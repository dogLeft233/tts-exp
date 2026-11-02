#!/usr/bin/env python3
"""Development-only ASR/CER guard for the frozen v3 candidate waveforms.

This program calls Qwen3-ASR-Flash only on canonicalized PCM16 WAV bytes for
the 20 candidate waveforms. It reuses the already sealed identity ASR rows
after proving they refer to the same natural PCM and direct PCM16 conversion;
it does not resend natural audio. Reference transcripts stay local and are
never included in request payloads.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
MODEL = "qwen3-asr-flash"
ITEM_SCHEMA = "cross-tfg-audio-head-asr-content-items-step120-v1"
RESULT_SCHEMA = "cross-tfg-audio-head-asr-content-results-step120-v1"
NATURAL_ITEM_SCHEMA = "cross-tfg-audio-head-asr-content-items-v1"
NATURAL_RESULT_SCHEMA = "cross-tfg-audio-head-asr-content-results-v1"
TRANSCRIPT_SCHEMA = "cross-tfg-audio-head-aishell1-transcript-subset-v1"
V3_METHOD_ID = "shared_recipient_transition_head_v3_fit40_step120"
V3_MODEL_IDS = ("wav2lip_gan_bac9a81e", "musetalk_fixed_portrait_attempt16", "ditto_hubert_fixed_portrait_attempt18")
V3_SEEDS = (20261003, 20261004, 20261005)
METHODS = (V3_METHOD_ID,)
NATURAL_METHOD_ID = "identity"
MATERIALIZATION_SCHEMA = "cross_tfg_v3_step120_method_candidate_materialization_v1"
CHECKPOINT_SCHEMA = "cross_tfg_v3_parameter_checkpoint_v1"
TRAINING_SCHEMA = "cross_tfg_v3_candidate_training_record_v1"
SELECTION_SCHEMA = "cross_tfg_v3_candidate_selection_record_v1"


class GuardError(RuntimeError):
    pass


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha(value: Mapping[str, Any], field: str) -> str:
    return _sha_bytes(_canonical_json({key: item for key, item in value.items() if key != field}))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GuardError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise GuardError(f"expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def _resolve_ref(root: Path, ref: Mapping[str, Any], where: str) -> Path:
    raw = ref.get("path")
    if not isinstance(raw, str) or not raw:
        raise GuardError(f"{where}.path is missing")
    path = Path(raw)
    path = (path if path.is_absolute() else root / path).resolve(strict=True)
    expected = ref.get("file_sha256")
    if not isinstance(expected, str) or _sha_file(path) != expected:
        raise GuardError(f"{where} file SHA mismatch: {path}")
    return path


def _project_pin(project_root: Path, pin: Mapping[str, Any], where: str) -> tuple[Path, bytes]:
    rel = pin.get("path")
    if not isinstance(rel, str) or not rel or "\\" in rel:
        raise GuardError(f"{where} path is missing or not canonical POSIX-relative text")
    rel_path = Path(rel)
    if rel_path.is_absolute() or ".." in rel_path.parts or not rel_path.parts:
        raise GuardError(f"{where} path escapes the declared project root")
    lexical = project_root
    for part in rel_path.parts:
        lexical = lexical / part
        if lexical.is_symlink():
            raise GuardError(f"{where} path traverses a symlink: {lexical}")
    path = lexical.resolve(strict=True)
    if project_root not in path.parents or not path.is_file():
        raise GuardError(f"{where} is not a regular file under the project root")
    raw = path.read_bytes()
    if type(pin.get("bytes")) is not int or len(raw) != pin["bytes"] or _sha_bytes(raw) != pin.get("sha256"):
        raise GuardError(f"{where} raw bytes/SHA differ from the sealed pin")
    return path, raw


def _validate_materialization(args: argparse.Namespace, plan: Mapping[str, Any]) -> dict[str, Any]:
    receipt_lexical = Path(args.materialization_receipt)
    if receipt_lexical.is_symlink():
        raise GuardError("materialization receipt must not be a symlink")
    receipt_path = receipt_lexical.resolve(strict=True)
    if not receipt_path.is_file():
        raise GuardError("materialization receipt is not a regular file")
    receipt_raw = receipt_path.read_bytes()
    receipt_sha = _sha_bytes(receipt_raw)
    if receipt_sha != args.expected_materialization_receipt_raw_sha256:
        raise GuardError("materialization receipt raw SHA differs from its explicit pin")
    receipt = json.loads(receipt_raw.decode("utf-8"))
    if not isinstance(receipt, dict):
        raise GuardError("materialization receipt is not a JSON object")
    for key, expected in (
        ("schema", MATERIALIZATION_SCHEMA),
        ("status", "MATERIALIZED_FIXED_STEP120_METHOD_CANDIDATE"),
        ("profile_role", "METHOD_CANDIDATE"),
        ("optimizer_steps", 120),
        ("fit_row_count", 40),
        ("fit_speaker_count", 20),
        ("zero_head_gate_rows_passed", 40),
        ("development_audio_opened", False),
        ("confirmation_audio_opened", False),
        ("method_selected_using_development", False),
        ("model_execution_performed", False),
    ):
        if receipt.get(key) != expected:
            raise GuardError(f"materialization receipt {key} differs from the fixed step120 contract")

    project_root = Path(args.project_root).resolve(strict=True)
    cfg_pin = receipt.get("candidate_config")
    provenance = receipt.get("candidate_provenance")
    if not isinstance(cfg_pin, dict) or not isinstance(provenance, dict) or set(provenance) != {"checkpoint", "training_record", "selection_record"}:
        raise GuardError("materialization receipt lacks the exact config/provenance pins")
    plan_refs = plan.get("source_reference_plans", {})
    plan_cfg = plan_refs.get("v3_cli_config") if isinstance(plan_refs, dict) else None
    if not isinstance(plan_cfg, dict) or cfg_pin.get("path") != plan_cfg.get("path") or cfg_pin.get("sha256") != plan_cfg.get("raw_sha256"):
        raise GuardError("materialization config pin differs from the frozen candidate plan")
    cfg_path, cfg_raw = _project_pin(project_root, cfg_pin, "candidate CLI config")
    config = json.loads(cfg_raw.decode("utf-8"))
    if not isinstance(config, dict) or config.get("schema") != "cross_tfg_v3_enhancement_cli_config_v1" or config.get("profile_role") != "METHOD_CANDIDATE":
        raise GuardError("materialized CLI config is not a V3 METHOD_CANDIDATE")
    if config.get("candidate_provenance") != provenance:
        raise GuardError("materialized CLI config and receipt provenance pins differ")

    plan_expected = {
        "checkpoint": plan.get("execution", {}).get("candidate_checkpoint_sha256"),
        "training_record": plan.get("execution", {}).get("candidate_training_record_sha256"),
        "selection_record": plan.get("execution", {}).get("candidate_selection_record_sha256"),
    }
    parsed: dict[str, dict[str, Any]] = {}
    raw_hashes: dict[str, str] = {}
    for name, pin in provenance.items():
        if not isinstance(pin, dict) or pin.get("sha256") != plan_expected[name]:
            raise GuardError(f"materialized {name} pin differs from the candidate plan")
        _path, raw = _project_pin(project_root, pin, f"candidate {name}")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise GuardError(f"candidate {name} record is not a JSON object")
        parsed[name] = value
        raw_hashes[name] = _sha_bytes(raw)

    checkpoint = parsed["checkpoint"]
    training = parsed["training_record"]
    selection = parsed["selection_record"]
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("head_schema") != config.get("head_schema") or checkpoint.get("parameter") != config.get("parameter"):
        raise GuardError("candidate checkpoint schema/head/parameter differs from the materialized config")
    if checkpoint.get("selection_status") != "UNSELECTED_FIT_CHECKPOINT_SELECTION_RECORD_NOT_CREATED":
        raise GuardError("candidate checkpoint is not the unselected fit-only terminal checkpoint")
    if training.get("schema") != TRAINING_SCHEMA or training.get("status") != "COMPLETE":
        raise GuardError("candidate training record is not complete")
    if training.get("completed_optimizer_steps") != 120 or training.get("planned_optimizer_steps") != 120:
        raise GuardError("candidate training record is not fixed at 120 optimizer steps")
    if training.get("checkpoint_sha256") != raw_hashes["checkpoint"]:
        raise GuardError("training record does not bind the exact candidate checkpoint")
    if training.get("development_audio_opened") is not False or training.get("confirmation_audio_opened") is not False:
        raise GuardError("fit record does not certify Dev and Confirm stayed closed")
    if training.get("selection_record_created") is not False or training.get("selection_status") != "NOT_SELECTED":
        raise GuardError("fit record selected weights before the fixed-step method freeze")
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("status") != "FROZEN":
        raise GuardError("candidate fixed-step selection record is not frozen")
    if selection.get("checkpoint_sha256") != raw_hashes["checkpoint"] or selection.get("training_record_sha256") != raw_hashes["training_record"]:
        raise GuardError("selection record does not bind the exact checkpoint and training record")
    if selection.get("selected_optimizer_step") != 120 or selection.get("selection_rule") != "fixed_terminal_optimizer_step_120_no_development_selection":
        raise GuardError("selection record is not the predeclared fixed terminal-step rule")
    if selection.get("development_audio_opened") is not False or selection.get("confirmation_audio_opened") is not False:
        raise GuardError("selection record does not certify Dev and Confirm stayed closed")

    method = plan.get("methods", [{}])[0]
    method_config = method.get("config", {}) if isinstance(method, dict) else {}
    if (
        method.get("method_id") != V3_METHOD_ID
        or method.get("checkpoint_sha256") != raw_hashes["checkpoint"]
        or method.get("training_record_sha256") != raw_hashes["training_record"]
        or method.get("fixed_optimizer_steps") != 120
        or method_config.get("v3_cli_config_sha256") != plan_cfg.get("raw_sha256")
        or method_config.get("candidate_checkpoint_sha256") != raw_hashes["checkpoint"]
        or method_config.get("candidate_training_record_sha256") != raw_hashes["training_record"]
        or method_config.get("candidate_fixed_step_selection_record_sha256") != raw_hashes["selection_record"]
    ):
        raise GuardError("candidate materialization provenance differs from the frozen Dev plan method")
    execution = plan.get("execution", {})
    candidate_scope = plan.get("analysis", {}).get("candidate_only_subrun", {})
    if (
        execution.get("candidate_only") is not True
        or execution.get("planned_rows") != 180
        or execution.get("fixed_optimizer_steps") != 120
        or execution.get("confirm_audio_read") is not False
        or execution.get("confirm_effects_read") is not False
        or execution.get("development_weights_selected") is not False
        or candidate_scope.get("weight_selection_using_dev") is not False
    ):
        raise GuardError("candidate plan is not the fixed-step candidate-only Dev scope")
    return {
        "path": str(receipt_path), "bytes": len(receipt_raw), "raw_sha256": receipt_sha,
        "candidate_config": {"path": str(cfg_path), "bytes": len(cfg_raw), "raw_sha256": _sha_bytes(cfg_raw)},
        "checkpoint_sha256": raw_hashes["checkpoint"],
        "training_record_sha256": raw_hashes["training_record"],
        "selection_record_sha256": raw_hashes["selection_record"],
        "optimizer_steps": 120,
    }


def _validate_source_bundle(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], Path]:
    plan_path = Path(args.plan).resolve(strict=True)
    results_path = Path(args.results).resolve(strict=True)
    manifest_path = Path(args.result_manifest).resolve(strict=True)
    root = Path(args.artifact_root).resolve(strict=True)
    plan, manifest = _load_json(plan_path), _load_json(manifest_path)
    rows = _load_jsonl(results_path)
    plan_raw, results_raw, manifest_raw = _sha_file(plan_path), _sha_file(results_path), _sha_file(manifest_path)
    for label, expected, actual in (
        ("plan", args.expected_plan_raw_sha256, plan_raw),
        ("results", args.expected_results_raw_sha256, results_raw),
        ("result manifest", args.expected_manifest_raw_sha256, manifest_raw),
    ):
        if actual != expected:
            raise GuardError(f"{label} raw SHA differs from the explicit pin")
    if plan.get("schema_version") != "cross-tfg-audio-eval-v1" or plan.get("status") != "FROZEN":
        raise GuardError("plan schema/status is not the frozen audio-evaluation contract")
    if plan.get("plan_sha256") != _payload_sha(plan, "plan_sha256"):
        raise GuardError("plan embedded payload fingerprint mismatch")
    if manifest.get("schema_version") != "cross-tfg-audio-eval-v1" or manifest.get("run_id") != plan.get("run_id"):
        raise GuardError("result manifest schema/run_id differs from plan")
    if manifest.get("manifest_sha256") != _payload_sha(manifest, "manifest_sha256"):
        raise GuardError("result-manifest embedded payload fingerprint mismatch")
    if (
        manifest.get("plan_raw_sha256") != plan_raw
        or manifest.get("plan_sha256") != plan.get("plan_sha256")
        or manifest.get("results_jsonl_sha256") != results_raw
        or manifest.get("planned_rows") != len(plan.get("rows", []))
        or manifest.get("result_rows") != len(rows)
    ):
        raise GuardError("result manifest does not bind exact plan/results/counts")
    planned = plan.get("rows")
    if not isinstance(planned, list) or len(planned) != len(rows):
        raise GuardError("plan/results row counts differ")
    planned_ids = [item.get("row_id") for item in planned]
    result_ids = [item.get("row_id") for item in rows]
    if len(planned_ids) != len(set(planned_ids)) or len(result_ids) != len(set(result_ids)):
        raise GuardError("plan or results contain duplicate row_id values")
    if planned_ids != result_ids:
        raise GuardError("result rows do not preserve frozen plan order")
    if any(item.get("split") != "dev" for item in planned) or any(item.get("split") != "dev" for item in rows):
        raise GuardError("this guard accepts development rows only")
    method_ids = [item.get("method_id") for item in plan.get("methods", [])]
    if tuple(method_ids) != METHODS:
        raise GuardError(f"frozen method order differs from expected five-method design: {method_ids}")
    declared_seeds = plan.get("generation_seed_family", {}).get("dev")
    if not isinstance(declared_seeds, list) or len(declared_seeds) != 3 or len(set(declared_seeds)) != 3:
        raise GuardError("development plan must declare exactly three distinct render seeds")
    if len(planned) != 20 * len(METHODS) * len(declared_seeds):
        raise GuardError("frozen development plan does not contain the expected 20×1×3 candidate row design")
    return plan, rows, manifest, root


def build_items(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    plan, results, result_manifest, root = _validate_source_bundle(args)
    results_by_id = {row.get("row_id"): row for row in results}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    planned_by_id: dict[str, dict[str, Any]] = {}
    for planned in plan["rows"]:
        key = (str(planned.get("sample_id")), str(planned.get("method_id")))
        groups[key].append(planned)
        planned_by_id[str(planned.get("row_id"))] = planned
    sample_meta: dict[str, tuple[str, dict[str, Any]]] = {}
    method_registry = {item["method_id"]: item for item in plan["methods"]}
    for planned in plan["rows"]:
        sample_id = str(planned.get("sample_id"))
        meta = (str(planned.get("speaker_id")), planned.get("natural_pcm"))
        old = sample_meta.setdefault(sample_id, meta)
        if old != meta:
            raise GuardError(f"sample natural PCM/speaker differs across methods: {sample_id}")
    seed_order: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for key, values in groups.items():
        values.sort(key=lambda row: int(row.get("seed")))
        expected_seeds = plan["generation_seed_family"]["dev"]
        if [row.get("seed") for row in values] != expected_seeds:
            raise GuardError(f"{key}: plan group does not contain the declared three-seed family")
        seed_order[key] = values
    items: list[dict[str, Any]] = []
    for (sample_id, method_id), planned_group in groups.items():
        speaker_id, natural_ref = sample_meta[sample_id]
        if not isinstance(natural_ref, dict):
            raise GuardError(f"{sample_id}: frozen natural PCM reference is missing")
        row_ids = [str(item["row_id"]) for item in planned_group]
        source_rows = [results_by_id.get(row_id) for row_id in row_ids]
        status_list = [row.get("status") if row else "MISSING" for row in source_rows]
        if method_id == "identity":
            audio_ref = dict(natural_ref)
        else:
            first_result = source_rows[0] if source_rows else None
            audio_ref = first_result.get("generator_pcm") if isinstance(first_result, dict) else None
        item_id = f"{sample_id}__{method_id}"
        common = {
            "item_id": item_id,
            "split": "dev",
            "sample_id": sample_id,
            "speaker_id": speaker_id,
            "method_id": method_id,
            "method_config_sha256": method_registry[method_id].get("config_sha256"),
            "seed_family": [int(item["seed"]) for item in planned_group],
            "source_row_ids": row_ids,
            "source_row_statuses": status_list,
            "natural_pcm": dict(natural_ref),
        }
        if method_id != "identity":
            refs = [row.get("generator_pcm") if isinstance(row, dict) else None for row in source_rows]
            if not refs or any(not isinstance(ref, dict) for ref in refs):
                items.append({**common, "status": "NOT_AVAILABLE", "failure_reason": "one or more planned rows lack generator_pcm"})
                continue
            first = refs[0]
            exact_keys = ("file_sha256", "pcm_sha256", "sample_count", "sample_rate_hz", "channels", "subtype")
            for other in refs[1:]:
                if any(other.get(field) != first.get(field) for field in exact_keys):
                    raise GuardError(f"{item_id}: transformed PCM differs across render seeds")
            audio_ref = dict(first)
        if not isinstance(audio_ref, dict):
            items.append({**common, "status": "NOT_AVAILABLE", "failure_reason": "audio reference is absent"})
            continue
        if any(status != "PASS" for status in status_list) and method_id != "identity":
            items.append({**common, "status": "NOT_AVAILABLE", "failure_reason": "one or more planned source rows did not PASS"})
            continue
        if (
            audio_ref.get("sample_rate_hz") != 16_000
            or audio_ref.get("channels") != 1
            or audio_ref.get("sample_count") != natural_ref.get("sample_count")
            or audio_ref.get("pcm_sha256") is None
        ):
            raise GuardError(f"{item_id}: audio metadata differs from the frozen 16-kHz same-length contract")
        if method_id == "identity" and (
            audio_ref.get("pcm_sha256") != natural_ref.get("pcm_sha256")
            or audio_ref.get("file_sha256") != natural_ref.get("file_sha256")
        ):
            raise GuardError(f"{item_id}: identity baseline is not byte-identical to natural PCM")
        audio_path = _resolve_ref(root, audio_ref, f"{item_id}.audio")
        common.update({
            "status": "READY",
            "audio": dict(audio_ref),
            "audio_path": str(audio_path),
            "audio_file_bytes": audio_path.stat().st_size,
            "audio_duration_seconds": float(audio_ref["sample_count"]) / 16_000,
        })
        items.append(common)
    if len(items) != len(groups):
        raise GuardError("content item builder did not preserve all planned sample/method groups")
    if not items or any(item.get("split") != "dev" for item in items):
        raise GuardError("empty/non-dev content item set")
    item_ids = [item["item_id"] for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise GuardError("duplicate planned content item identity")
    if len(sample_meta) != 20 or len({speaker for speaker, _ in sample_meta.values()}) != 10:
        raise GuardError("frozen development plan must contain 20 clips from 10 speakers")
    if len(items) != len(sample_meta) * len(METHODS):
        raise GuardError("plan does not contain one candidate method group per sample")
    return plan, items, root


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical_json(dict(row)) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_f32_wav(path: Path, expected: Mapping[str, Any], wavfile: Any) -> np.ndarray:
    rate, audio = wavfile.read(str(path), mmap=False)
    if rate != 16_000 or audio.ndim != 1 or audio.dtype != np.dtype("float32"):
        raise GuardError(f"expected mono IEEE float32 16-kHz WAV: {path}")
    audio = np.ascontiguousarray(audio, dtype="<f4")
    if audio.size != expected.get("sample_count") or _sha_bytes(audio.tobytes(order="C")) != expected.get("pcm_sha256"):
        raise GuardError(f"decoded float32 PCM/count mismatch: {path}")
    if not np.isfinite(audio).all():
        raise GuardError(f"non-finite audio samples: {path}")
    return audio


def _pcm16_convert(ffmpeg: str, source: Path, target: Path, expected_samples: int, wavfile: Any) -> dict[str, Any]:
    command = [
        ffmpeg, "-nostdin", "-n", "-v", "error", "-i", str(source), "-map", "0:a:0", "-vn",
        "-c:a", "pcm_s16le", "-f", "wav", "-ar", "16000", "-ac", "1", str(target),
    ]
    process = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0 or not target.is_file():
        raise GuardError(f"FFmpeg PCM16 conversion failed for {source} (exit={process.returncode})")
    rate, samples = wavfile.read(str(target), mmap=False)
    if rate != 16_000 or samples.ndim != 1 or samples.dtype != np.dtype("int16") or samples.size != expected_samples:
        raise GuardError(f"converted PCM16 WAV format/count differs for {source}")
    if not np.isfinite(samples).all():
        raise GuardError(f"converted PCM16 samples are non-finite for {source}")
    pcm_bytes = np.ascontiguousarray(samples, dtype="<i2").tobytes(order="C")
    with wave.open(str(target), "rb") as wf:
        if wf.getframerate() != 16_000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getnframes() != expected_samples:
            raise GuardError(f"converted PCM16 WAV header differs for {source}")
    return {
        "path": str(target),
        "file_sha256": _sha_file(target),
        "file_bytes": target.stat().st_size,
        "pcm16_sha256": _sha_bytes(pcm_bytes),
        "sample_count": int(samples.size),
        "sample_rate_hz": int(rate),
        "channels": 1,
        "encoding": "pcm_s16le",
        "ffmpeg_argv": command,
    }


def _transcribe_once(audio_bytes: bytes, api_key: str, timeout: float) -> dict[str, Any]:
    import requests

    data_uri = "data:audio/wav;base64," + base64.b64encode(audio_bytes).decode("ascii")
    payload = {
        "model": MODEL,
        "input": {"messages": [{"role": "user", "content": [{"audio": data_uri}]}]},
        "parameters": {"asr_options": {"language": "zh", "enable_itn": False}},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
    except Exception as exc:
        return {"status": "FAILED_TRANSPORT", "failure_reason": type(exc).__name__, "http_status": None, "request_id": None}
    request_id = response.headers.get("X-DashScope-Request-Id")
    try:
        body = response.json()
    except Exception:
        body = None
    if response.status_code < 200 or response.status_code >= 300:
        code = body.get("code") if isinstance(body, dict) and isinstance(body.get("code"), str) else None
        return {"status": "FAILED_HTTP", "failure_reason": "non_2xx_response", "http_status": response.status_code, "api_error_code": code, "request_id": request_id}
    if not isinstance(body, dict):
        return {"status": "FAILED_RESPONSE", "failure_reason": "response_not_json_object", "http_status": response.status_code, "request_id": request_id}
    output = body.get("output")
    choices = output.get("choices") if isinstance(output, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        code = body.get("code") if isinstance(body.get("code"), str) else None
        return {"status": "FAILED_RESPONSE", "failure_reason": "missing_output_choices", "http_status": response.status_code, "api_error_code": code, "request_id": request_id}
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    texts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                texts.append(block["text"])
    elif isinstance(content, str):
        texts.append(content)
    if not texts:
        return {"status": "FAILED_RESPONSE", "failure_reason": "no_text_transcription", "http_status": response.status_code, "request_id": request_id}
    text = "".join(texts).strip()
    return {
        "status": "PASS",
        "hypothesis": text,
        "http_status": response.status_code,
        "request_id": request_id or body.get("request_id"),
        "usage": body.get("usage"),
    }


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in normalized if not ch.isspace() and not unicodedata.category(ch).startswith(("P", "Z")))


def _edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, lch in enumerate(left, 1):
        current = [i]
        for j, rch in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (lch != rch)))
        previous = current
    return previous[-1]


def _cer(reference: str, hypothesis: str) -> dict[str, Any]:
    ref, hyp = _normalize(reference), _normalize(hypothesis)
    errors = _edit_distance(ref, hyp)
    return {
        "reference_characters": len(ref),
        "hypothesis_characters": len(hyp),
        "edit_errors": errors,
        "cer": float(errors / len(ref)) if ref else None,
        "normalization": "NFKC+casefold; remove Unicode whitespace/separators and punctuation; character-level Levenshtein",
    }


def run(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    if out.exists():
        raise GuardError(f"output directory already exists: {out}")
    consumer_source_path = Path(__file__).resolve(strict=True)
    consumer_source_sha = _sha_file(consumer_source_path)
    if consumer_source_sha != args.expected_consumer_source_sha256:
        raise GuardError("consumer source raw SHA differs from the explicit execution pin")
    plan, items, artifact_root = build_items(args)
    if len(items) != args.expected_item_count:
        raise GuardError(f"planned content items {len(items)} != expected {args.expected_item_count}")
    ffmpeg = Path(args.ffmpeg).resolve(strict=True)
    ffmpeg_sha = _sha_file(ffmpeg)
    if ffmpeg_sha != args.expected_ffmpeg_sha256:
        raise GuardError("FFmpeg binary SHA differs from the frozen PCM16 conversion pin")
    transcript_root = Path(args.transcripts_root).resolve(strict=True)
    transcript_manifest_path = Path(args.transcript_manifest).resolve(strict=True)
    transcript_manifest = _load_json(transcript_manifest_path)
    transcript_manifest_raw_sha = _sha_file(transcript_manifest_path)
    if transcript_manifest_raw_sha != args.expected_transcript_manifest_raw_sha256:
        raise GuardError("AISHELL transcript-subset manifest raw SHA differs from the explicit pin")
    if transcript_manifest.get("schema_version") != TRANSCRIPT_SCHEMA or transcript_manifest.get("status") != "PASS":
        raise GuardError("transcript-subset manifest schema/status is not qualified")
    if transcript_manifest.get("manifest_sha256") != _payload_sha(transcript_manifest, "manifest_sha256"):
        raise GuardError("transcript-subset manifest payload fingerprint mismatch")
    if transcript_manifest.get("sample_count") != 20:
        raise GuardError("transcript-subset manifest does not bind exactly 20 dev references")
    transcript_file_map = transcript_manifest.get("files")
    if not isinstance(transcript_file_map, dict):
        raise GuardError("transcript-subset manifest has no files mapping")
    transcripts: dict[str, tuple[str, str]] = {}
    transcript_source_rows = {row.get("sample_id"): row for row in transcript_manifest.get("rows", []) if isinstance(row, dict)}
    if set(transcript_file_map) != {item["sample_id"] for item in items} or set(transcript_source_rows) != set(transcript_file_map):
        raise GuardError("transcript subset sample IDs do not exactly match the frozen dev items")
    for item in items:
        sample_id = item["sample_id"]
        if sample_id not in transcripts:
            if Path(sample_id).name != sample_id or sample_id in (".", ".."):
                raise GuardError(f"invalid sample id for transcript lookup: {sample_id!r}")
            file_ref = transcript_file_map[sample_id]
            if not isinstance(file_ref, dict) or file_ref.get("path") != f"{sample_id}.txt":
                raise GuardError(f"transcript subset path mapping differs for {sample_id}")
            transcript_path = transcript_root / f"{sample_id}.txt"
            if not transcript_path.is_file() or _sha_file(transcript_path) != file_ref.get("raw_sha256"):
                raise GuardError(f"transcript subset text file is missing or hash-mismatched for {sample_id}")
            raw = transcript_path.read_bytes()
            text = raw.decode("utf-8-sig").strip()
            if not text:
                raise GuardError(f"empty transcript for {sample_id}")
            source_row = transcript_source_rows[sample_id]
            if source_row.get("text_raw_sha256") != _sha_bytes(raw) or source_row.get("speaker_id") != item["speaker_id"]:
                raise GuardError(f"transcript source-line or speaker binding differs for {sample_id}")
            transcripts[sample_id] = (text, _sha_bytes(raw))
    # Reuse the already completed identity ASR run. Verify the result ledger,
    # item ledger, manifest seals, natural float32 identities, and later the
    # exact PCM16 conversion before comparing v3 hypotheses to natural speech.
    natural_asr_path = Path(args.natural_asr_results).resolve(strict=True)
    natural_items_path = Path(args.natural_content_items).resolve(strict=True)
    natural_input_manifest_path = Path(args.natural_content_input_manifest).resolve(strict=True)
    natural_result_manifest_path = Path(args.natural_content_result_manifest).resolve(strict=True)
    natural_asr_raw = _sha_file(natural_asr_path)
    natural_items_raw = _sha_file(natural_items_path)
    natural_input_manifest_raw = _sha_file(natural_input_manifest_path)
    natural_result_manifest_raw = _sha_file(natural_result_manifest_path)
    expected_natural_pins = (
        ("natural ASR rows", args.expected_natural_asr_results_sha256, natural_asr_raw),
        ("natural content items", args.expected_natural_content_items_sha256, natural_items_raw),
        ("natural item manifest", args.expected_natural_content_input_manifest_sha256, natural_input_manifest_raw),
        ("natural result manifest", args.expected_natural_content_result_manifest_sha256, natural_result_manifest_raw),
    )
    for label, expected, actual in expected_natural_pins:
        if expected != actual:
            raise GuardError(f"{label} raw SHA differs from its explicit pin")
    natural_asr_rows = _load_jsonl(natural_asr_path)
    natural_items = _load_jsonl(natural_items_path)
    natural_input_manifest = _load_json(natural_input_manifest_path)
    natural_result_manifest = _load_json(natural_result_manifest_path)
    if natural_input_manifest.get("schema_version") != ITEM_SCHEMA or natural_input_manifest.get("run_id") != "cross_tfg_audio_head_20261003_spectral_head_fit_v2_joint_attempt02":
        raise GuardError("reused natural ASR item manifest is not the sealed prior W2L development run")
    if natural_input_manifest.get("manifest_sha256") != _payload_sha(natural_input_manifest, "manifest_sha256"):
        raise GuardError("reused natural ASR item-manifest payload fingerprint mismatch")
    if natural_result_manifest.get("schema_version") != RESULT_SCHEMA or natural_result_manifest.get("status") != "PASS":
        raise GuardError("reused natural ASR result manifest is not PASS")
    if natural_result_manifest.get("manifest_sha256") != _payload_sha(natural_result_manifest, "manifest_sha256"):
        raise GuardError("reused natural ASR result-manifest payload fingerprint mismatch")
    if (
        natural_result_manifest.get("asr_results_sha256") != natural_asr_raw
        or natural_result_manifest.get("content_items_manifest_sha256") != natural_input_manifest.get("manifest_sha256")
    ):
        raise GuardError("reused natural ASR manifest does not bind its exact row and item files")
    natural_item_by_id = {row.get("item_id"): row for row in natural_items}
    natural_asr_by_sample: dict[str, dict[str, Any]] = {}
    natural_pcm_by_sample = {item["sample_id"]: item["natural_pcm"] for item in items}
    for sample_id, natural_ref in natural_pcm_by_sample.items():
        item_id = f"{sample_id}__identity"
        prior_item = natural_item_by_id.get(item_id)
        prior_asr = next((row for row in natural_asr_rows if row.get("item_id") == item_id), None)
        if not isinstance(prior_item, dict) or not isinstance(prior_asr, dict):
            raise GuardError(f"prior identity ASR row is absent for {sample_id}")
        prior_ref = prior_item.get("natural_pcm")
        if (
            prior_item.get("method_id") != NATURAL_METHOD_ID
            or prior_item.get("sample_id") != sample_id
            or prior_item.get("status") != "READY"
            or not isinstance(prior_ref, dict)
            or prior_ref.get("pcm_sha256") != natural_ref.get("pcm_sha256")
            or prior_ref.get("file_sha256") != natural_ref.get("file_sha256")
            or prior_asr.get("method_id") != NATURAL_METHOD_ID
            or prior_asr.get("sample_id") != sample_id
            or prior_asr.get("asr_status") != "PASS"
            or not isinstance(prior_asr.get("hypothesis"), str)
            or not prior_asr.get("hypothesis")
        ):
            raise GuardError(f"prior natural identity ASR lineage is invalid for {sample_id}")
        natural_asr_by_sample[sample_id] = prior_asr
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise GuardError("DASHSCOPE_API_KEY is not present in the private execution environment")
    if not args.confirm_api_calls:
        raise GuardError("API execution requires --confirm-api-calls in the reviewed worker command")
    from scipy.io import wavfile
    import scipy
    import requests  # noqa: F401 - preflight before making any billable request

    temporary_root = Path(args.tmp_root).resolve()
    if temporary_root.exists():
        raise GuardError(f"temporary conversion directory must be fresh: {temporary_root}")
    if temporary_root == out or temporary_root in out.parents or out in temporary_root.parents:
        raise GuardError("temporary conversion and output directories must be disjoint")
    temporary_root.mkdir(parents=True)
    # Preflight all canonical float32 inputs and all PCM16 conversions before the first billable request.
    converted: dict[str, dict[str, Any]] = {}
    conversion_records: dict[str, dict[str, Any]] = {}
    natural_conversion_records: dict[str, dict[str, Any]] = {}
    for item in items:
        if item.get("status") != "READY":
            continue
        audio_ref = item["audio"]
        path = Path(item["audio_path"])
        _read_f32_wav(path, audio_ref, wavfile)
        if path.stat().st_size > args.max_input_wav_bytes:
            raise GuardError(f"ASR input exceeds frozen size limit: {item['item_id']}")
        if item["audio_duration_seconds"] > args.max_duration_seconds:
            raise GuardError(f"ASR input exceeds frozen duration limit: {item['item_id']}")
        cache_key = str(audio_ref["pcm_sha256"])
        if cache_key not in converted:
            pcm16_path = temporary_root / f"{cache_key}.pcm16.wav"
            converted[cache_key] = _pcm16_convert(str(ffmpeg), path, pcm16_path, int(audio_ref["sample_count"]), wavfile)
        conversion_records[item["item_id"]] = converted[cache_key]
    for item in items:
        sample_id = item["sample_id"]
        if sample_id in natural_conversion_records:
            continue
        natural_ref = item["natural_pcm"]
        natural_path = _resolve_ref(artifact_root, natural_ref, f"{sample_id}.natural_pcm")
        _read_f32_wav(natural_path, natural_ref, wavfile)
        pcm16_path = temporary_root / f"natural_{sample_id}_{natural_ref['pcm_sha256']}.pcm16.wav"
        natural_pcm16 = _pcm16_convert(str(ffmpeg), natural_path, pcm16_path, int(natural_ref["sample_count"]), wavfile)
        prior_asr = natural_asr_by_sample[sample_id]
        if (
            prior_asr.get("input_pcm16_sha256") != natural_pcm16.get("pcm16_sha256")
            or prior_asr.get("input_pcm16_sample_count") != natural_pcm16.get("sample_count")
        ):
            raise GuardError(f"reused natural ASR audio is not the pinned direct PCM16 conversion for {sample_id}")
        natural_conversion_records[sample_id] = natural_pcm16
    for pcm in converted.values():
        if pcm["file_bytes"] > args.max_input_wav_bytes:
            raise GuardError("canonical PCM16 ASR payload exceeds the frozen size limit")
    if len(converted) != args.expected_unique_audio_count:
        raise GuardError(f"unique audio PCM count {len(converted)} != frozen count {args.expected_unique_audio_count}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    item_raws = [_canonical_json(item) + b"\n" for item in items]
    item_bytes = b"".join(item_raws)
    item_manifest = {
        "schema_version": ITEM_SCHEMA,
        "run_id": plan.get("run_id"),
        "plan_raw_sha256": args.expected_plan_raw_sha256,
        "results_raw_sha256": args.expected_results_raw_sha256,
        "result_manifest_raw_sha256": args.expected_manifest_raw_sha256,
        "consumer_source_path": str(consumer_source_path),
        "consumer_source_sha256": consumer_source_sha,
        "split": "dev",
        "planned_items": len(items),
        "ready_items": sum(item.get("status") == "READY" for item in items),
        "not_available_items": sum(item.get("status") != "READY" for item in items),
        "method_ids": list(METHODS),
        "deduplicated_float32_audio_hashes": len(converted),
        "converted_pcm16_audio_hashes": len(converted),
        "items_jsonl_sha256": _sha_bytes(item_bytes),
        "pcm16_conversion": {
            "ffmpeg_path": str(ffmpeg), "ffmpeg_sha256": ffmpeg_sha,
            "argv_template": ["-v", "error", "-i", "{canonical_float32_wav}", "-map", "0:a:0", "-vn", "-c:a", "pcm_s16le", "-f", "wav", "-ar", "16000", "-ac", "1", "{output_pcm16_wav}"],
            "sample_count_preserved": True,
        },
        "asr_request": {
            "provider": "Alibaba Cloud DashScope",
            "model": MODEL,
            "endpoint": API_URL,
            "input_encoding": "audio/wav;base64 data URI; canonical mono PCM16LE 16 kHz WAV",
            "parameters": {"asr_options": {"language": "zh", "enable_itn": False}},
            "secret_source": "DASHSCOPE_API_KEY environment only; key is not written or logged",
        },
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__, "requests": requests.__version__},
        "reference_transcripts": {
            "root": str(transcript_root),
            "files": {sample_id: {"path": str(transcript_root / f"{sample_id}.txt"), "raw_sha256": digest} for sample_id, (_, digest) in sorted(transcripts.items())},
            "subset_manifest_path": str(transcript_manifest_path),
            "subset_manifest_raw_sha256": transcript_manifest_raw_sha,
            "subset_manifest_payload_sha256": transcript_manifest["manifest_sha256"],
            "source_file_sha256": transcript_manifest.get("source_file_sha256"),
            "source_revision": transcript_manifest.get("source_revision"),
            "source_url": transcript_manifest.get("source_url"),
            "source_line_sha256s": {sample_id: transcript_source_rows[sample_id].get("source_line_sha256") for sample_id in sorted(transcripts)},
            "sent_to_asr_provider": False,
        },
        "scope": {"score_outcomes_used_for_selection": False, "confirmation_audio_read": False, "gpu_forward": False},
        "natural_baseline_asr_reuse": {
            "asr_results_path": str(natural_asr_path),
            "asr_results_raw_sha256": natural_asr_raw,
            "content_items_path": str(natural_items_path),
            "content_items_raw_sha256": natural_items_raw,
            "content_input_manifest_path": str(natural_input_manifest_path),
            "content_input_manifest_raw_sha256": natural_input_manifest_raw,
            "content_result_manifest_path": str(natural_result_manifest_path),
            "content_result_manifest_raw_sha256": natural_result_manifest_raw,
            "reused_natural_identity_items": len(natural_asr_by_sample),
            "natural_asr_api_calls_repeated": 0,
            "direct_pcm16_sha256_by_sample": {sample_id: ref["pcm16_sha256"] for sample_id, ref in sorted(natural_conversion_records.items())},
        },
    }
    item_manifest["manifest_sha256"] = _payload_sha(item_manifest, "manifest_sha256")
    (out / "content_items.jsonl").write_bytes(item_bytes)
    (out / "content_input_manifest.json").write_bytes(_canonical_json(item_manifest) + b"\n")
    _write_jsonl(out / "asr_results.jsonl", [])
    request_cache: dict[str, tuple[dict[str, Any], str]] = {}
    asr_rows: list[dict[str, Any]] = []
    for item in items:
        base = {key: item.get(key) for key in ("item_id", "split", "sample_id", "speaker_id", "method_id", "seed_family", "source_row_ids", "status")}
        if item.get("status") != "READY":
            asr_rows.append({**base, "asr_status": "SOURCE_NOT_AVAILABLE", "failure_reason": item.get("failure_reason")})
            continue
        pcm16 = conversion_records[item["item_id"]]
        pcm_key = pcm16["pcm16_sha256"]
        if pcm_key in request_cache:
            cached_result, original_item_id = request_cache[pcm_key]
            call_result = dict(cached_result)
            call_result["transcription_reused_from_audio_sha256"] = pcm_key
            call_result["transcription_reused_from_item_id"] = original_item_id
            call_result["api_call_made"] = False
        else:
            raw = Path(pcm16["path"]).read_bytes()
            call_result = _transcribe_once(raw, key, args.timeout_seconds)
            call_result["api_call_made"] = True
            call_result["transcription_reused_from_audio_sha256"] = None
            call_result["transcription_reused_from_item_id"] = None
            call_result["pcm16_audio_sha256"] = pcm_key
            call_result["pcm16_audio_file_sha256"] = pcm16["file_sha256"]
            request_cache[pcm_key] = (dict(call_result), item["item_id"])
            if args.delay_seconds > 0:
                time.sleep(args.delay_seconds)
        asr_rows.append({
            **base,
            "asr_status": call_result.get("status"),
            "asr_model": MODEL,
            "api_endpoint": API_URL,
            "input_pcm16_sha256": pcm_key,
            "input_pcm16_file_sha256": pcm16["file_sha256"],
            "input_pcm16_sample_count": pcm16["sample_count"],
            "request_id": call_result.get("request_id"),
            "http_status": call_result.get("http_status"),
            "api_error_code": call_result.get("api_error_code"),
            "failure_reason": call_result.get("failure_reason"),
            "hypothesis": call_result.get("hypothesis"),
            "usage": call_result.get("usage"),
            "api_call_made": call_result.get("api_call_made"),
            "transcription_reused_from_audio_sha256": call_result.get("transcription_reused_from_audio_sha256"),
            "transcription_reused_from_item_id": call_result.get("transcription_reused_from_item_id"),
        })
        _write_jsonl(out / "asr_results.jsonl", asr_rows)
    asr_by_item = {row["item_id"]: row for row in asr_rows}
    item_by_id = {row["item_id"]: row for row in items}
    content_rows: list[dict[str, Any]] = []
    for item in items:
        asr = asr_by_item[item["item_id"]]
        transcript, transcript_sha = transcripts[item["sample_id"]]
        result = {
            "item_id": item["item_id"], "split": "dev", "sample_id": item["sample_id"],
            "speaker_id": item["speaker_id"], "method_id": item["method_id"],
            "audio_pcm_sha256": (item.get("audio") or {}).get("pcm_sha256"),
            "transcript_raw_sha256": transcript_sha,
            "asr_status": asr.get("asr_status"),
            "planned_in_denominator": True,
        }
        if asr.get("asr_status") == "PASS" and isinstance(asr.get("hypothesis"), str):
            result["status"] = "PASS"
            result["reference_status"] = "AVAILABLE"
            result["cer_status"] = "PASS"
            result["cer_vs_reference"] = _cer(transcript, asr["hypothesis"])
            baseline = natural_asr_by_sample.get(item["sample_id"])
            if baseline and baseline.get("asr_status") == "PASS" and isinstance(baseline.get("hypothesis"), str):
                result["asr_disagreement_vs_natural"] = _cer(baseline["hypothesis"], asr["hypothesis"])
                result["asr_disagreement_status"] = "PASS"
                result["natural_asr_baseline_item_id"] = baseline.get("item_id")
                result["natural_asr_baseline_input_pcm16_sha256"] = baseline.get("input_pcm16_sha256")
                result["natural_asr_baseline_source_results_sha256"] = natural_asr_raw
            else:
                result["asr_disagreement_vs_natural"] = None
                result["asr_disagreement_status"] = "BASELINE_ASR_UNAVAILABLE"
        else:
            result["cer_vs_reference"] = None
            result["asr_disagreement_vs_natural"] = None
            result["reference_status"] = "AVAILABLE"
            result["cer_status"] = "ASR_UNAVAILABLE"
            result["asr_disagreement_status"] = "ASR_UNAVAILABLE"
            result["status"] = "INCOMPLETE"
            result["failure_reason"] = asr.get("failure_reason") or asr.get("asr_status")
        content_rows.append(result)
    _write_jsonl(out / "content_scores.jsonl", content_rows)
    method_summaries = []
    for method_id in METHODS:
        method_rows = [row for row in content_rows if row["method_id"] == method_id]
        valid = [row for row in method_rows if row["status"] == "PASS"]
        speakers = sorted({row["speaker_id"] for row in method_rows})
        speaker_cer: list[float] = []
        speaker_disagreement: list[float] = []
        speaker_n = []
        disagreement_speaker_n = []
        for speaker in speakers:
            candidates = [row for row in valid if row["speaker_id"] == speaker]
            values = [row["cer_vs_reference"]["cer"] for row in candidates if row["cer_vs_reference"].get("cer") is not None]
            if values:
                speaker_cer.append(float(sum(values) / len(values)))
                speaker_n.append({"speaker_id": speaker, "planned_clips": sum(row["speaker_id"] == speaker for row in method_rows), "estimable_clips": len(values)})
            disagreements = [row["asr_disagreement_vs_natural"]["cer"] for row in candidates if isinstance(row.get("asr_disagreement_vs_natural"), dict) and row["asr_disagreement_vs_natural"].get("cer") is not None]
            if disagreements:
                speaker_disagreement.append(float(sum(disagreements) / len(disagreements)))
                disagreement_speaker_n.append({"speaker_id": speaker, "planned_clips": sum(row["speaker_id"] == speaker for row in method_rows), "estimable_clips": len(disagreements)})
        method_summaries.append({
            "method_id": method_id,
            "planned_clips": len(method_rows),
            "estimable_clips": len(valid),
            "failed_or_missing_clips": len(method_rows) - len(valid),
            "planned_speakers": len(speakers),
            "speaker_estimable_counts": speaker_n,
            "estimable_speakers": len(speaker_cer),
            "equal_speaker_mean_cer": float(sum(speaker_cer) / len(speaker_cer)) if speaker_cer else None,
            "disagreement_speaker_estimable_counts": disagreement_speaker_n,
            "disagreement_estimable_speakers": len(speaker_disagreement),
            "disagreement_estimable_clips": sum(item["estimable_clips"] for item in disagreement_speaker_n),
            "equal_speaker_mean_asr_disagreement_vs_natural": float(sum(speaker_disagreement) / len(speaker_disagreement)) if speaker_disagreement else None,
            "interpretation": "development descriptive only; no confirmatory CI or selection claim",
        })
    counts = Counter(row["status"] for row in content_rows)
    overall_status = "PASS" if len(content_rows) == len(items) and counts["PASS"] == len(items) else "INCOMPLETE"
    summary = {
        "schema_version": RESULT_SCHEMA,
        "status": overall_status,
        "run_id": plan.get("run_id"),
        "plan_raw_sha256": args.expected_plan_raw_sha256,
        "results_raw_sha256": args.expected_results_raw_sha256,
        "result_manifest_raw_sha256": args.expected_manifest_raw_sha256,
        "consumer_source_sha256": consumer_source_sha,
        "content_items_manifest_sha256": item_manifest["manifest_sha256"],
        "planned_content_items": len(items),
        "asr_calls_made": sum(row.get("api_call_made") is True for row in asr_rows),
        "asr_call_reuses": sum(row.get("api_call_made") is False for row in asr_rows),
        "content_status_counts": dict(counts),
        "method_summaries": method_summaries,
        "normalization": "NFKC+casefold; remove Unicode whitespace/separators and punctuation; character-level Levenshtein",
        "api": {"provider": "Alibaba Cloud DashScope", "model": MODEL, "endpoint": API_URL, "language": "zh", "enable_itn": False},
        "scope": {"split": "dev", "samples": len({item["sample_id"] for item in items}), "method_ids": list(METHODS), "confirmation_audio_read": False, "score_outcomes_used_for_selection": False, "reference_transcripts_sent_to_provider": False},
        "natural_baseline_asr_reuse": item_manifest["natural_baseline_asr_reuse"],
        "limitations": ["Character error rates measure ASR agreement with fixed references and do not by themselves prove perceptual intelligibility."],
    }
    summary["summary_sha256"] = _payload_sha(summary, "summary_sha256")
    (out / "content_summary.json").write_bytes(_canonical_json(summary) + b"\n")
    files = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "content_result_manifest.json":
            files.append({"path": path.name, "size_bytes": path.stat().st_size, "raw_sha256": _sha_file(path)})
    output_manifest = {
        "schema_version": RESULT_SCHEMA,
        "status": overall_status,
        "run_id": plan.get("run_id"),
        "content_summary_sha256": summary["summary_sha256"],
        "content_scores_sha256": _sha_file(out / "content_scores.jsonl"),
        "asr_results_sha256": _sha_file(out / "asr_results.jsonl"),
        "content_items_manifest_sha256": item_manifest["manifest_sha256"],
        "files": files,
        "scope": summary["scope"],
    }
    output_manifest["manifest_sha256"] = _payload_sha(output_manifest, "manifest_sha256")
    (out / "content_result_manifest.json").write_bytes(_canonical_json(output_manifest) + b"\n")
    print(json.dumps({"status": overall_status, "items": len(items), "api_calls": summary["asr_calls_made"], "estimable": counts["PASS"], "incomplete": counts["INCOMPLETE"], "out": str(out)}, ensure_ascii=False, sort_keys=True))
    return 0 if overall_status == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--result-manifest", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--expected-plan-raw-sha256", required=True)
    parser.add_argument("--expected-results-raw-sha256", required=True)
    parser.add_argument("--expected-manifest-raw-sha256", required=True)
    parser.add_argument("--expected-consumer-source-sha256", required=True)
    parser.add_argument("--expected-ffmpeg-sha256", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--transcripts-root", required=True)
    parser.add_argument("--tmp-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--transcript-manifest", required=True)
    parser.add_argument("--expected-transcript-manifest-raw-sha256", required=True)
    parser.add_argument("--natural-asr-results", required=True)
    parser.add_argument("--expected-natural-asr-results-sha256", required=True)
    parser.add_argument("--natural-content-items", required=True)
    parser.add_argument("--expected-natural-content-items-sha256", required=True)
    parser.add_argument("--natural-content-input-manifest", required=True)
    parser.add_argument("--expected-natural-content-input-manifest-sha256", required=True)
    parser.add_argument("--natural-content-result-manifest", required=True)
    parser.add_argument("--expected-natural-content-result-manifest-sha256", required=True)
    parser.add_argument("--expected-item-count", type=int, default=20)
    parser.add_argument("--expected-unique-audio-count", type=int, default=20)
    parser.add_argument("--max-input-wav-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--max-duration-seconds", type=float, default=300.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--delay-seconds", type=float, default=0.25)
    parser.add_argument("--confirm-api-calls", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except Exception as exc:
        print(f"asr_content_guard_failed:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
