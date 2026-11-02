#!/usr/bin/env python3
"""Content guard for the frozen step-120 candidate, preserving all 180 rows.

This version inventories every planned model/seed row before ASR. It hashes and
decodes every PASS generator WAV, groups only equal float32 sample arrays within
each clip, and retains unavailable rows. PCM16 request grouping is done after
the frozen FFmpeg conversion. A PCM mismatch among a clip's nine model/seed
rows is reported as an engineering control failure; it is never treated as a
new candidate or used for selection. Reference transcripts stay local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import asr_content_guard_v3_step120_candidate_attempt02 as frozen


ITEM_SCHEMA = "cross-tfg-audio-head-asr-content-items-step120-v3"
RESULT_SCHEMA = "cross-tfg-audio-head-asr-content-results-step120-v3"
SOURCE_SCHEMA = "cross-tfg-audio-head-step120-candidate-source-inventory-v1"
V3_METHOD_ID = frozen.V3_METHOD_ID
V3_MODEL_IDS = frozen.V3_MODEL_IDS
V3_SEEDS = frozen.V3_SEEDS


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(frozen._canonical_json(value) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    frozen._write_jsonl(path, rows)


def _source_inventory(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], Path, dict[str, Any]]:
    plan, results, manifest, artifact_root, materialization = frozen._validate_source_bundle(args)
    result_manifest_path = Path(args.result_manifest).resolve(strict=True)
    manifest_value = frozen._load_json(result_manifest_path)
    statuses = Counter(str(row.get("status")) for row in results)
    expected_counts = {state: int(statuses.get(state, 0)) for state in ("PASS", "FAILED", "INCOMPLETE")}
    if manifest_value.get("status_counts") != expected_counts:
        raise frozen.GuardError("candidate result-manifest status counts differ from all 180 result rows")
    if manifest_value.get("status") not in ("PASS", "INCOMPLETE"):
        raise frozen.GuardError("candidate result manifest has an unsupported terminal status")
    expected_manifest_status = "PASS" if expected_counts["FAILED"] == 0 and expected_counts["INCOMPLETE"] == 0 else "INCOMPLETE"
    if manifest_value.get("status") != expected_manifest_status:
        raise frozen.GuardError("candidate result-manifest terminal status conflicts with its row ledger")

    results_by_id = {str(row.get("row_id")): row for row in results}
    source_rows: list[dict[str, Any]] = []
    units_by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    sample_meta: dict[str, tuple[str, tuple[Any, ...], dict[str, Any]]] = {}
    method_registry = {str(item["method_id"]): item for item in plan.get("methods", [])}
    for planned in plan["rows"]:
        row_id = str(planned["row_id"])
        result = results_by_id[row_id]
        for field in ("sample_id", "speaker_id", "model_id", "seed", "method_id"):
            if result.get(field) != planned.get(field):
                raise frozen.GuardError(f"candidate result {field} differs from its frozen plan row: {row_id}")
        if result.get("status") not in ("PASS", "FAILED", "INCOMPLETE"):
            raise frozen.GuardError(f"candidate result row has an invalid producer status: {row_id}")
        sample_id = str(planned["sample_id"])
        natural = planned.get("natural_pcm")
        if not isinstance(natural, dict):
            raise frozen.GuardError(f"planned natural PCM reference is missing: {row_id}")
        natural_signature = tuple(natural.get(field) for field in
                                  ("file_sha256", "pcm_sha256", "sample_count", "sample_rate_hz", "channels", "subtype"))
        previous = sample_meta.setdefault(sample_id, (str(planned["speaker_id"]), natural_signature, dict(natural)))
        if previous[0] != str(planned["speaker_id"]) or previous[1] != natural_signature:
            raise frozen.GuardError(f"natural PCM/speaker identity differs within sample {sample_id}")

        record: dict[str, Any] = {
            "row_id": row_id,
            "split": "dev",
            "sample_id": sample_id,
            "speaker_id": str(planned["speaker_id"]),
            "method_id": V3_METHOD_ID,
            "model_id": str(planned["model_id"]),
            "seed": int(planned["seed"]),
            "natural_pcm": dict(natural),
            "producer_status": result.get("status"),
            "waveform_unit_id": None,
            "availability_status": None,
            "failure_reason": result.get("failure_reason"),
        }
        if result.get("status") != "PASS":
            if not isinstance(result.get("failure_reason"), str):
                raise frozen.GuardError(f"unavailable candidate row lacks a recorded failure reason: {row_id}")
            record["availability_status"] = "SOURCE_NOT_AVAILABLE"
            source_rows.append(record)
            continue

        ref = result.get("generator_pcm")
        method_config_sha = method_registry[V3_METHOD_ID].get("config_sha256")
        if (not isinstance(ref, dict)
                or result.get("generator_pcm_sha256") != ref.get("pcm_sha256")
                or result.get("method_config_sha256") != method_config_sha):
            raise frozen.GuardError(f"PASS row lacks exact generator PCM/method-config binding: {row_id}")
        if (ref.get("sample_rate_hz") != 16_000 or ref.get("channels") != 1
                or ref.get("sample_count") != natural.get("sample_count")
                or not isinstance(ref.get("pcm_sha256"), str)):
            raise frozen.GuardError(f"candidate WAV differs from the frozen mono 16-kHz/sample-count contract: {row_id}")
        wav_path = frozen._resolve_ref(artifact_root, ref, f"{row_id}.generator_pcm")
        if (wav_path.stat().st_size > args.max_input_wav_bytes
                or int(ref["sample_count"]) / 16_000 > args.max_duration_seconds):
            raise frozen.GuardError(f"candidate generator WAV exceeds frozen size/duration limit: {row_id}")
        # This verifies decoded little-endian float32 sample bytes against the
        # row's PCM pin, independently of WAV container metadata or file SHA.
        from scipy.io import wavfile
        audio = frozen._read_f32_wav(wav_path, ref, wavfile)
        pcm_digest = frozen._sha_bytes(np.ascontiguousarray(audio, dtype="<f4").tobytes(order="C"))
        if pcm_digest != ref.get("pcm_sha256"):
            raise frozen.GuardError(f"decoded generator sample-array SHA differs from row binding: {row_id}")
        unit = units_by_sample[sample_id].get(pcm_digest)
        if unit is None:
            unit_id = f"{sample_id}__pcm_{pcm_digest[:16]}"
            unit = {
                "item_id": unit_id,
                "waveform_unit_id": unit_id,
                "split": "dev",
                "sample_id": sample_id,
                "speaker_id": str(planned["speaker_id"]),
                "method_id": V3_METHOD_ID,
                "method_config_sha256": method_config_sha,
                "pcm_sha256": pcm_digest,
                "sample_count": int(ref["sample_count"]),
                "sample_rate_hz": 16_000,
                "channels": 1,
                "subtype": ref.get("subtype"),
                "natural_pcm": dict(natural),
                "source_rows": [],
                "wav_files": [],
                "status": "READY",
            }
            units_by_sample[sample_id][pcm_digest] = unit
        alias = {"row_id": row_id, "model_id": str(planned["model_id"]), "seed": int(planned["seed"]),
                 "status": "PASS", "generator_pcm": dict(ref), "resolved_path": str(wav_path)}
        unit["source_rows"].append(alias)
        file_ref = {"path": str(wav_path), "file_sha256": ref.get("file_sha256"), "bytes": wav_path.stat().st_size}
        if file_ref not in unit["wav_files"]:
            unit["wav_files"].append(file_ref)
        record.update({
            "availability_status": "READY",
            "waveform_unit_id": unit["waveform_unit_id"],
            "generator_pcm": dict(ref),
            "generator_pcm_path": str(wav_path),
            "generator_wav_file_sha256": ref.get("file_sha256"),
            "generator_float32_pcm_sha256": pcm_digest,
        })
        source_rows.append(record)

    if len(source_rows) != 180 or len({row["row_id"] for row in source_rows}) != 180:
        raise frozen.GuardError("source inventory did not retain exactly one record for every planned candidate row")
    waveform_control: dict[str, dict[str, Any]] = {}
    for sample_id, (speaker_id, _natural_signature, _natural_ref) in sample_meta.items():
        groups = units_by_sample.get(sample_id, {})
        source_group = [row for row in source_rows if row["sample_id"] == sample_id]
        unavailable = sum(row["availability_status"] != "READY" for row in source_group)
        state = "CONTROL_FAILED" if len(groups) > 1 else ("INCOMPLETE" if unavailable else "PASS")
        waveform_control[sample_id] = {
            "sample_id": sample_id,
            "speaker_id": speaker_id,
            "planned_model_seed_rows": len(source_group),
            "ready_rows": len(source_group) - unavailable,
            "unavailable_rows": unavailable,
            "distinct_float32_sample_arrays": len(groups),
            "waveform_identity_status": state,
            "waveform_unit_ids": [unit["waveform_unit_id"] for unit in groups.values()],
        }
    all_units = [unit for sample in sample_meta for unit in units_by_sample.get(sample, {}).values()]
    for unit in all_units:
        unit["waveform_identity_status"] = waveform_control[unit["sample_id"]]["waveform_identity_status"]
        unit["source_rows"].sort(key=lambda row: (V3_MODEL_IDS.index(row["model_id"]), V3_SEEDS.index(row["seed"])))
        unit["source_row_ids"] = [row["row_id"] for row in unit["source_rows"]]
    return plan, source_rows, artifact_root, {
        "manifest": manifest,
        "materialization": materialization,
        "sample_meta": sample_meta,
        "units": all_units,
        "waveform_control": waveform_control,
        "candidate_result_status_counts": {state: int(Counter(str(row.get("status")) for row in results).get(state, 0)) for state in ("PASS", "FAILED", "INCOMPLETE")},
    }


def _load_transcripts(args: argparse.Namespace, sample_ids: set[str], speakers: Mapping[str, str]) -> tuple[dict[str, tuple[str, str]], dict[str, Any], str]:
    transcript_root = Path(args.transcripts_root).resolve(strict=True)
    manifest_path = Path(args.transcript_manifest).resolve(strict=True)
    manifest = frozen._load_json(manifest_path)
    raw_sha = frozen._sha_file(manifest_path)
    if raw_sha != args.expected_transcript_manifest_raw_sha256:
        raise frozen.GuardError("AISHELL transcript-subset manifest raw SHA differs from the explicit pin")
    if (manifest.get("schema_version") != frozen.TRANSCRIPT_SCHEMA or manifest.get("status") != "PASS"
            or manifest.get("manifest_sha256") != frozen._payload_sha(manifest, "manifest_sha256")
            or manifest.get("sample_count") != 20):
        raise frozen.GuardError("transcript-subset manifest is not the sealed 20-reference PASS contract")
    files = manifest.get("files")
    rows = {row.get("sample_id"): row for row in manifest.get("rows", []) if isinstance(row, dict)}
    if not isinstance(files, dict) or set(files) != sample_ids or set(rows) != sample_ids:
        raise frozen.GuardError("transcript subset sample IDs differ from the frozen candidate cohort")
    transcripts: dict[str, tuple[str, str]] = {}
    for sample_id in sorted(sample_ids):
        ref = files[sample_id]
        if not isinstance(ref, dict) or ref.get("path") != f"{sample_id}.txt":
            raise frozen.GuardError(f"transcript filename mapping differs for {sample_id}")
        path = transcript_root / f"{sample_id}.txt"
        raw = path.read_bytes()
        if frozen._sha_bytes(raw) != ref.get("raw_sha256"):
            raise frozen.GuardError(f"transcript raw SHA mismatch for {sample_id}")
        text = raw.decode("utf-8-sig").strip()
        row = rows[sample_id]
        if (not text or row.get("text_raw_sha256") != frozen._sha_bytes(raw)
                or row.get("speaker_id") != speakers[sample_id]):
            raise frozen.GuardError(f"transcript text/speaker lineage invalid for {sample_id}")
        transcripts[sample_id] = (text, frozen._sha_bytes(raw))
    return transcripts, manifest, raw_sha


def _load_natural_baselines(args: argparse.Namespace, sample_natural: Mapping[str, dict[str, Any]], artifact_root: Path, ffmpeg: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    paths = {
        "asr": Path(args.natural_asr_results).resolve(strict=True),
        "items": Path(args.natural_content_items).resolve(strict=True),
        "input_manifest": Path(args.natural_content_input_manifest).resolve(strict=True),
        "result_manifest": Path(args.natural_content_result_manifest).resolve(strict=True),
    }
    expected = {
        "asr": args.expected_natural_asr_results_sha256,
        "items": args.expected_natural_content_items_sha256,
        "input_manifest": args.expected_natural_content_input_manifest_sha256,
        "result_manifest": args.expected_natural_content_result_manifest_sha256,
    }
    actual = {key: frozen._sha_file(path) for key, path in paths.items()}
    if actual != expected:
        raise frozen.GuardError("one or more reused natural identity ASR raw SHA pins differ")
    asr_rows = frozen._load_jsonl(paths["asr"])
    input_rows = frozen._load_jsonl(paths["items"])
    input_manifest = frozen._load_json(paths["input_manifest"])
    result_manifest = frozen._load_json(paths["result_manifest"])
    if (input_manifest.get("schema_version") != frozen.NATURAL_ITEM_SCHEMA
            or input_manifest.get("run_id") != "cross_tfg_audio_head_20261003_spectral_head_fit_v2_joint_attempt02"
            or input_manifest.get("manifest_sha256") != frozen._payload_sha(input_manifest, "manifest_sha256")):
        raise frozen.GuardError("natural identity input manifest is not the sealed prior run")
    if (result_manifest.get("schema_version") != frozen.NATURAL_RESULT_SCHEMA
            or result_manifest.get("status") != "PASS"
            or result_manifest.get("manifest_sha256") != frozen._payload_sha(result_manifest, "manifest_sha256")
            or result_manifest.get("asr_results_sha256") != actual["asr"]
            or result_manifest.get("content_items_manifest_sha256") != input_manifest.get("manifest_sha256")):
        raise frozen.GuardError("natural identity result manifest does not bind exact sealed rows/items")
    item_by_id = {row.get("item_id"): row for row in input_rows}
    asr_by_id = {row.get("item_id"): row for row in asr_rows}
    from scipy.io import wavfile
    import scipy
    import requests  # noqa: F401 - verify runtime import before any API request

    natural: dict[str, dict[str, Any]] = {}
    pcm16_by_sample: dict[str, dict[str, Any]] = {}
    temp = Path(args.tmp_root).resolve()
    if temp.exists():
        raise frozen.GuardError(f"temporary PCM16 directory must be fresh: {temp}")
    out = Path(args.out).resolve()
    if temp == out or temp in out.parents or out in temp.parents:
        raise frozen.GuardError("natural PCM16 conversion and output directories must be disjoint")
    temp.parent.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True)
    for sample_id, ref in sample_natural.items():
        prior = item_by_id.get(f"{sample_id}__identity")
        asr = asr_by_id.get(f"{sample_id}__identity")
        if (not isinstance(prior, dict) or not isinstance(asr, dict)
                or prior.get("status") != "READY" or prior.get("method_id") != frozen.NATURAL_METHOD_ID
                or prior.get("sample_id") != sample_id or asr.get("asr_status") != "PASS"
                or not isinstance(asr.get("hypothesis"), str) or not asr.get("hypothesis")):
            raise frozen.GuardError(f"sealed prior identity ASR row is unavailable for {sample_id}")
        prior_ref = prior.get("natural_pcm")
        if (not isinstance(prior_ref, dict) or prior_ref.get("pcm_sha256") != ref.get("pcm_sha256")
                or prior_ref.get("file_sha256") != ref.get("file_sha256")):
            raise frozen.GuardError(f"natural identity ASR PCM lineage differs for {sample_id}")
        path = frozen._resolve_ref(artifact_root, ref, f"{sample_id}.natural_pcm")
        audio = frozen._read_f32_wav(path, ref, wavfile)
        if path.stat().st_size > args.max_input_wav_bytes or len(audio) / 16_000 > args.max_duration_seconds:
            raise frozen.GuardError(f"natural PCM16 source exceeds frozen size/duration limit: {sample_id}")
        converted = frozen._pcm16_convert(ffmpeg, path, temp / f"natural_{sample_id}_{ref['pcm_sha256']}.pcm16.wav", int(ref["sample_count"]), wavfile)
        if (asr.get("input_pcm16_sha256") != converted.get("pcm16_sha256")
                or asr.get("input_pcm16_sample_count") != converted.get("sample_count")):
            raise frozen.GuardError(f"natural identity ASR is not the direct frozen PCM16 conversion: {sample_id}")
        natural[sample_id] = asr
        pcm16_by_sample[sample_id] = converted
    return natural, {"paths": {key: str(path) for key, path in paths.items()},
                     "raw_sha256": actual, "input_manifest_payload_sha256": input_manifest.get("manifest_sha256"),
                     "direct_pcm16_by_sample": pcm16_by_sample, "runtime": {"scipy": scipy.__version__, "requests": requests.__version__}}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "plan", "results", "result-manifest", "artifact-root", "materialization-receipt",
                 "expected-plan-raw-sha256", "expected-results-raw-sha256", "expected-manifest-raw-sha256",
                 "expected-consumer-source-sha256", "expected-attempt02-helper-sha256", "expected-ffmpeg-sha256",
                 "ffmpeg", "transcripts-root", "tmp-root", "out", "transcript-manifest",
                 "expected-transcript-manifest-raw-sha256", "natural-asr-results", "expected-natural-asr-results-sha256",
                 "natural-content-items", "expected-natural-content-items-sha256",
                 "natural-content-input-manifest", "expected-natural-content-input-manifest-sha256",
                 "natural-content-result-manifest", "expected-natural-content-result-manifest-sha256"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--expected-item-count", type=int, default=20, help="frozen dev sample count")
    parser.add_argument("--max-request-count", type=int, default=180)
    parser.add_argument("--max-input-wav-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--max-duration-seconds", type=float, default=300.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--delay-seconds", type=float, default=0.25)
    parser.add_argument("--confirm-api-calls", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve(strict=True)
    if project_root != Path(__file__).resolve(strict=True).parents[2]:
        raise frozen.GuardError("project root differs from the guarded repository")
    helper_path = Path(frozen.__file__).resolve(strict=True)
    if frozen._sha_file(helper_path) != args.expected_attempt02_helper_sha256:
        raise frozen.GuardError("reused frozen helper source SHA differs from its explicit pin")
    source_path = Path(__file__).resolve(strict=True)
    source_sha = frozen._sha_file(source_path)
    if source_sha != args.expected_consumer_source_sha256:
        raise frozen.GuardError("attempt03 source SHA differs from its explicit pin")
    if (args.expected_item_count != 20 or args.max_request_count != 180
            or args.max_input_wav_bytes != 10 * 1024 * 1024
            or args.max_duration_seconds != 300.0 or args.timeout_seconds != 120.0
            or args.delay_seconds != 0.25):
        raise frozen.GuardError("sample scope, input limits, timeout, and delay are fixed by the guarded protocol")
    if not args.confirm_api_calls:
        raise frozen.GuardError("Qwen ASR requires --confirm-api-calls after source inventory/preflight review")
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise frozen.GuardError("DASHSCOPE_API_KEY is not present in the private runtime environment")
    out = Path(args.out).resolve()
    if out.exists():
        raise frozen.GuardError(f"output directory must be fresh: {out}")
    temp = Path(args.tmp_root).resolve()
    candidate_temp = temp.parent / (temp.name + "_candidate")
    if temp.exists() or candidate_temp.exists():
        raise frozen.GuardError("both PCM16 conversion directories must be fresh")
    if (temp == out or temp in out.parents or out in temp.parents
            or candidate_temp == out or candidate_temp in out.parents or out in candidate_temp.parents
            or temp == candidate_temp or temp in candidate_temp.parents or candidate_temp in temp.parents):
        raise frozen.GuardError("PCM16 conversion and output directories must be disjoint")

    plan, source_rows, artifact_root, info = _source_inventory(args)
    sample_meta: Mapping[str, tuple[str, tuple[Any, ...], dict[str, Any]]] = info["sample_meta"]
    sample_ids = set(sample_meta)
    speakers = {sample: data[0] for sample, data in sample_meta.items()}
    if len(sample_ids) != args.expected_item_count or len(set(speakers.values())) != 10:
        raise frozen.GuardError("candidate inventory differs from frozen 20-clip/10-speaker cohort")
    transcripts, transcript_manifest, transcript_manifest_sha = _load_transcripts(args, sample_ids, speakers)
    ffmpeg = Path(args.ffmpeg).resolve(strict=True)
    ffmpeg_sha = frozen._sha_file(ffmpeg)
    if ffmpeg_sha != args.expected_ffmpeg_sha256:
        raise frozen.GuardError("FFmpeg binary SHA differs from the frozen PCM16 conversion pin")
    natural_asr, natural_info = _load_natural_baselines(args, {sample: data[2] for sample, data in sample_meta.items()}, artifact_root, str(ffmpeg))
    from scipy.io import wavfile
    import scipy
    import requests

    units = info["units"]
    unit_by_id = {unit["item_id"]: unit for unit in units}
    conversions: dict[str, dict[str, Any]] = {}
    temp = Path(args.tmp_root).resolve()
    if temp == out or temp in out.parents or out in temp.parents:
        raise frozen.GuardError("natural PCM16 conversion and output directories must be disjoint")
    # `_load_natural_baselines` created the only temp directory and converted
    # natural PCM. Candidate conversions are placed in a disjoint sibling.
    if candidate_temp.exists():
        raise frozen.GuardError("candidate PCM16 conversion directory is not fresh/disjoint")
    candidate_temp.parent.mkdir(parents=True, exist_ok=True)
    candidate_temp.mkdir()
    for unit in units:
        representative = unit["source_rows"][0]
        source = Path(representative["resolved_path"])
        ref = representative["generator_pcm"]
        if source.stat().st_size > args.max_input_wav_bytes or unit["sample_count"] / 16_000 > args.max_duration_seconds:
            raise frozen.GuardError(f"candidate ASR input exceeds frozen limits: {unit['item_id']}")
        converted = frozen._pcm16_convert(str(ffmpeg), source, candidate_temp / f"{unit['pcm_sha256']}.pcm16.wav", int(unit["sample_count"]), wavfile)
        if converted["file_bytes"] > args.max_input_wav_bytes:
            raise frozen.GuardError(f"canonical PCM16 payload exceeds frozen byte limit: {unit['item_id']}")
        unit["pcm16"] = converted
        conversions[unit["item_id"]] = converted

    # Qwen calls are grouped by actual PCM16 payload *within a sample*. A
    # different sample's identical bytes remain a distinct independent unit.
    payload_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for unit in units:
        payload_groups[(unit["sample_id"], unit["pcm16"]["pcm16_sha256"])].append(unit["item_id"])
    if len(payload_groups) > args.max_request_count:
        raise frozen.GuardError(f"actual unique sample×PCM16 requests exceed frozen cap: {len(payload_groups)}")
    for unit in units:
        unit["qwen_payload_key"] = f"{unit['sample_id']}:{unit['pcm16']['pcm16_sha256']}"
    payload_to_units = {key: list(value) for key, value in payload_groups.items()}

    waveform_states = {row["waveform_identity_status"] for row in info["waveform_control"].values()}
    if waveform_states == {"PASS"} and len(units) == 20 and len(payload_groups) == 20:
        equality_state = "PASS_20_SAMPLE_WAVEFORMS_IDENTICAL_ACROSS_9_CONTEXTS"
    elif "CONTROL_FAILED" in waveform_states:
        equality_state = "CONTROL_FAILED_MODEL_SEED_WAVEFORM_MISMATCH"
    else:
        equality_state = "INCOMPLETE_UNAVAILABLE_CANDIDATE_WAVEFORM_ROWS"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    source_path_out = out / "candidate_source_rows.jsonl"
    item_path_out = out / "content_items.jsonl"
    _write_jsonl(source_path_out, source_rows)
    _write_jsonl(item_path_out, units)
    input_manifest = {
        "schema_version": ITEM_SCHEMA,
        "run_id": plan.get("run_id"),
        "plan_raw_sha256": args.expected_plan_raw_sha256,
        "results_raw_sha256": args.expected_results_raw_sha256,
        "result_manifest_raw_sha256": args.expected_manifest_raw_sha256,
        "candidate_method_id": V3_METHOD_ID,
        "materialization_receipt": info["materialization"],
        "planned_source_rows": len(source_rows),
        "source_row_status_counts": {key: int(Counter(row["availability_status"] for row in source_rows).get(key, 0)) for key in ("READY", "SOURCE_NOT_AVAILABLE")},
        "planned_samples": len(sample_ids),
        "planned_speakers": len(set(speakers.values())),
        "actual_float32_audio_units": len(units),
        "actual_qwen_payload_groups": len(payload_groups),
        "waveform_identity_status": equality_state,
        "waveform_controls_by_sample": info["waveform_control"],
        "source_rows_raw_sha256": frozen._sha_file(source_path_out),
        "content_items_raw_sha256": frozen._sha_file(item_path_out),
        "transcripts": {"manifest_path": args.transcript_manifest, "manifest_raw_sha256": transcript_manifest_sha,
                        "manifest_payload_sha256": transcript_manifest.get("manifest_sha256"), "sent_to_provider": False},
        "natural_identity_asr": natural_info,
        "pcm16_conversion": {"ffmpeg_path": str(ffmpeg), "ffmpeg_sha256": ffmpeg_sha,
                             "sample_count_preserved": True, "natural_pcm16_reconversion_verified": True},
        "qwen_asr": {"provider": "Alibaba Cloud DashScope", "model": frozen.MODEL, "endpoint": frozen.API_URL,
                     "language": "zh", "enable_itn": False, "request_grouping": "exact PCM16 SHA within sample_id",
                     "max_input_wav_bytes": args.max_input_wav_bytes, "max_duration_seconds": args.max_duration_seconds,
                     "timeout_seconds": args.timeout_seconds, "delay_seconds": args.delay_seconds,
                     "api_key_source": "DASHSCOPE_API_KEY environment only", "reference_text_sent": False},
        "source_scope": {"dev_only": True, "confirm_opened": False, "score_selection": False,
                         "candidate_model_ids": list(V3_MODEL_IDS), "candidate_seeds": list(V3_SEEDS)},
        "consumer_source_sha256": source_sha,
        "attempt02_helper_source_sha256": args.expected_attempt02_helper_sha256,
        "ffmpeg_sha256": ffmpeg_sha,
        "transcripts_by_sample": {sample: {"speaker_id": speakers[sample], "text_raw_sha256": transcripts[sample][1]} for sample in sorted(sample_ids)},
    }
    input_manifest["manifest_sha256"] = frozen._payload_sha(input_manifest, "manifest_sha256")
    _write_json(out / "content_input_manifest.json", input_manifest)

    # At this point every PASS WAV/file/content reference and every unavailable
    # planned row is persisted. Only now may billable requests begin.
    asr_rows: list[dict[str, Any]] = []
    _write_jsonl(out / "asr_results.jsonl", [])
    for payload_key, unit_ids in payload_to_units.items():
        first = unit_by_id[unit_ids[0]]
        pcm16 = first["pcm16"]
        raw_audio = Path(pcm16["path"]).read_bytes()
        response = frozen._transcribe_once(raw_audio, api_key, args.timeout_seconds)
        response.update({"api_call_made": True, "pcm16_audio_sha256": pcm16["pcm16_sha256"],
                         "pcm16_audio_file_sha256": pcm16["file_sha256"], "sample_id": first["sample_id"],
                         "payload_key": f"{payload_key[0]}:{payload_key[1]}", "unit_ids": unit_ids})
        asr_rows.append({"sample_id": first["sample_id"], "payload_key": response["payload_key"],
                         "source_unit_ids": unit_ids, "source_row_ids": sorted({row_id for unit_id in unit_ids for row_id in unit_by_id[unit_id]["source_row_ids"]}),
                         "source_model_seed_rows": [alias for unit_id in unit_ids for alias in unit_by_id[unit_id]["source_rows"]],
                         "asr_status": response.get("status"), "asr_model": frozen.MODEL, "api_endpoint": frozen.API_URL,
                         "input_pcm16_sha256": pcm16["pcm16_sha256"], "input_pcm16_file_sha256": pcm16["file_sha256"],
                         "input_pcm16_sample_count": pcm16["sample_count"], "request_id": response.get("request_id"),
                         "http_status": response.get("http_status"), "api_error_code": response.get("api_error_code"),
                         "failure_reason": response.get("failure_reason"), "hypothesis": response.get("hypothesis"),
                         "usage": response.get("usage"), "api_call_made": True})
        _write_jsonl(out / "asr_results.jsonl", asr_rows)
        if args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

    asr_row_by_payload = {(row["sample_id"], row["input_pcm16_sha256"]): row for row in asr_rows}
    unit_score_rows: list[dict[str, Any]] = []
    unit_score_by_id: dict[str, dict[str, Any]] = {}
    for unit in units:
        payload = (unit["sample_id"], unit["pcm16"]["pcm16_sha256"])
        asr = asr_row_by_payload[payload]
        transcript, transcript_sha = transcripts[unit["sample_id"]]
        score = {"item_id": unit["item_id"], "waveform_unit_id": unit["waveform_unit_id"],
                 "sample_id": unit["sample_id"], "speaker_id": unit["speaker_id"], "method_id": V3_METHOD_ID,
                 "audio_pcm_sha256": unit["pcm_sha256"], "pcm16_audio_sha256": unit["pcm16"]["pcm16_sha256"],
                 "source_row_ids": unit["source_row_ids"], "source_model_seed_rows": unit["source_rows"],
                 "waveform_identity_status": unit["waveform_identity_status"],
                 "transcript_raw_sha256": transcript_sha, "asr_status": asr.get("asr_status"),
                 "planned_in_denominator": True}
        if asr.get("asr_status") == "PASS" and isinstance(asr.get("hypothesis"), str):
            score.update({"status": "PASS", "reference_status": "AVAILABLE", "cer_status": "DESCRIPTIVE",
                          "cer_vs_reference": frozen._cer(transcript, asr["hypothesis"]),
                          "asr_disagreement_vs_natural": frozen._cer(natural_asr[unit["sample_id"]]["hypothesis"], asr["hypothesis"]),
                          "asr_disagreement_status": "DESCRIPTIVE", "natural_asr_baseline_item_id": f"{unit['sample_id']}__identity"})
        else:
            score.update({"status": "INCOMPLETE", "reference_status": "AVAILABLE", "cer_status": "ASR_UNAVAILABLE",
                          "asr_disagreement_vs_natural": None, "asr_disagreement_status": "ASR_UNAVAILABLE",
                          "cer_vs_reference": None, "failure_reason": asr.get("failure_reason") or asr.get("asr_status")})
        unit_score_rows.append(score)
        unit_score_by_id[unit["item_id"]] = score
    _write_jsonl(out / "content_audio_unit_scores.jsonl", unit_score_rows)

    row_score_rows: list[dict[str, Any]] = []
    score_by_row = {alias["row_id"]: unit_score_by_id[unit["item_id"]]
                    for unit in units for alias in unit["source_rows"]}
    for source in source_rows:
        score = score_by_row.get(source["row_id"])
        if score is None:
            row_score_rows.append({**{field: source.get(field) for field in ("row_id", "split", "sample_id", "speaker_id", "method_id", "model_id", "seed")},
                                   "status": "INCOMPLETE", "planned_in_denominator": True,
                                   "producer_status": source["producer_status"], "failure_reason": source.get("failure_reason") or "SOURCE_NOT_AVAILABLE",
                                   "waveform_unit_id": None, "cer_vs_reference": None, "asr_disagreement_vs_natural": None})
        else:
            row_score_rows.append({**{field: source.get(field) for field in ("row_id", "split", "sample_id", "speaker_id", "method_id", "model_id", "seed")},
                                   "status": score["status"], "planned_in_denominator": True,
                                   "producer_status": source["producer_status"], "waveform_unit_id": source["waveform_unit_id"],
                                   "audio_pcm_sha256": score["audio_pcm_sha256"], "pcm16_audio_sha256": score["pcm16_audio_sha256"],
                                   "asr_status": score["asr_status"], "cer_vs_reference": score["cer_vs_reference"],
                                   "asr_disagreement_vs_natural": score["asr_disagreement_vs_natural"],
                                   "transcript_raw_sha256": score["transcript_raw_sha256"]})
    _write_jsonl(out / "content_scores.jsonl", row_score_rows)

    source_missing = sum(row["availability_status"] != "READY" for row in source_rows)
    unit_score_failed = sum(row["status"] != "PASS" for row in unit_score_rows)
    sample_unit_counts = Counter(unit["sample_id"] for unit in units)
    equality_failed = any(value["waveform_identity_status"] == "CONTROL_FAILED" for value in info["waveform_control"].values())
    if equality_failed:
        overall = "CONTROL_FAILED"
    elif source_missing or unit_score_failed or len(sample_unit_counts) != 20 or any(sample_unit_counts[sample] != 1 for sample in sample_ids):
        overall = "INCOMPLETE"
    else:
        overall = "PASS"
    summary = {
        "schema_version": RESULT_SCHEMA, "status": overall, "run_id": plan.get("run_id"),
        "plan_raw_sha256": args.expected_plan_raw_sha256, "results_raw_sha256": args.expected_results_raw_sha256,
        "result_manifest_raw_sha256": args.expected_manifest_raw_sha256, "consumer_source_sha256": source_sha,
        "planned_source_rows": len(source_rows), "ready_source_rows": len(source_rows) - source_missing,
        "unavailable_source_rows": source_missing, "planned_samples": len(sample_ids),
        "planned_speakers": len(set(speakers.values())), "actual_float32_audio_units": len(units),
        "distinct_qwen_payloads": len(payload_groups), "qwen_calls_made": len(asr_rows),
        "waveform_identity_status": equality_state, "sample_unit_counts": dict(sample_unit_counts),
        "unit_pass_count": sum(row["status"] == "PASS" for row in unit_score_rows),
        "unit_incomplete_count": unit_score_failed,
        "candidate_method_id": V3_METHOD_ID,
        "method_selection_from_content_or_visual_results": False,
        "reference_transcripts_sent_to_provider": False,
        "interpretation": "Development-only descriptive CER and ASR disagreement; waveform mismatch is an engineering control failure and no candidate promotion or selection follows.",
    }
    summary["summary_sha256"] = frozen._payload_sha(summary, "summary_sha256")
    _write_json(out / "content_summary.json", summary)
    file_list = []
    for path in sorted(item for item in out.iterdir() if item.is_file() and item.name != "content_result_manifest.json"):
        file_list.append({"path": path.name, "bytes": path.stat().st_size, "sha256": frozen._sha_file(path)})
    manifest_out = {
        "schema_version": RESULT_SCHEMA, "status": overall, "run_id": plan.get("run_id"),
        "content_summary_sha256": summary["summary_sha256"],
        "content_scores_sha256": frozen._sha_file(out / "content_scores.jsonl"),
        "content_audio_unit_scores_sha256": frozen._sha_file(out / "content_audio_unit_scores.jsonl"),
        "asr_results_sha256": frozen._sha_file(out / "asr_results.jsonl"),
        "content_input_manifest_sha256": frozen._sha_file(out / "content_input_manifest.json"),
        "files": file_list,
        "scope": {"planned_source_rows": 180, "planned_samples": 20, "candidate_method_id": V3_METHOD_ID,
                  "confirm_opened": False, "selection": False},
    }
    manifest_out["manifest_sha256"] = frozen._payload_sha(manifest_out, "manifest_sha256")
    _write_json(out / "content_result_manifest.json", manifest_out)
    print(json.dumps({"status": overall, "planned_source_rows": len(source_rows), "audio_units": len(units),
                      "qwen_calls": len(asr_rows), "unavailable_rows": source_missing, "out": str(out)},
                     sort_keys=True, separators=(",", ":")))
    return 0 if overall == "PASS" else 2


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(_parser().parse_args(argv))
    except Exception as exc:
        print(f"asr_content_guard_v3_step120_attempt03_failed:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
