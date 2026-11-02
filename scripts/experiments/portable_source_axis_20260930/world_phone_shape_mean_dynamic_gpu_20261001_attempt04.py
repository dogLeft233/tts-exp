#!/usr/bin/env python3
"""Fresh Wav2Lip/SyncNet execution for the frozen WORLD phone-shape design.

The CPU physical producer owns the five WORLD waveforms. This runner creates
all five videos fresh on the frozen primary face, embeds the two fixed audio
domains (the recipient's raw PCM and its original WORLD RT waveform), and
writes full V-by-A raw/unit distance arrays. It does not aggregate cohort
effects. Run preflight against the final, separately frozen EVAL manifest
before `run`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch
from scipy.io import wavfile

from . import evaluate, ops, scoring, wav2lip


DESIGN_SHA256 = "e17d257e43a78af6d7a291641ade6825b61c430ba51b8052383cab54dbea31fb"
METRIC_ADDENDUM_SHA256 = "9a6d9c14ad6d067911bd3e754d11e974787afe77a7b4bc05e278d92046ad33e6"
CAL_SOURCE_SHA256 = "4f4f1fe5d59565c87706cedf5531c3f29c649819d7a1c61413f0dee8508ed88e"
CELLS = ("q00", "q10", "q01", "q11", "SELF_GRID")
AUDIO_DOMAINS = ("A_RAW", "A_RT")
DIRECTIONS = ("tts_to_natural", "natural_to_tts")
SUPPORTS = ("full", "valid", "guard20", "guard15")
ANCHOR_COLUMN = 18
VSHIFT = 15
FIXED_RMS = 0.0376838172675746
EXPECTED_FLAGS = {
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "deterministic_algorithms": True,
}


class RunnerError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _fingerprint(value: Mapping[str, Any], field: str = "manifest_sha256") -> str:
    return hashlib.sha256(_canonical({key: item for key, item in value.items() if key != field})).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _must_sha(path: Path, expected: str, label: str) -> str:
    path = path.resolve(strict=True)
    observed = _sha(path)
    if observed != expected:
        raise RunnerError(f"{label} raw SHA mismatch: {observed} != {expected}")
    return observed


def _python_source_tree(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in sorted(p for p in root.rglob("*.py") if p.is_file() and "__pycache__" not in p.parts):
        relative = path.relative_to(root).as_posix()
        file_sha = _sha(path)
        size = path.stat().st_size
        digest.update(f"{relative}\0{file_sha}\0{size}\n".encode("utf-8"))
        file_count += 1
        total_bytes += size
    return {"path": str(root), "sha256": digest.hexdigest(), "file_count": file_count, "source_bytes": total_bytes}


def _inventory_map(path: Path) -> dict[str, str]:
    value = _json(path)
    if isinstance(value, dict) and isinstance(value.get("files"), dict):
        value = value["files"]
    if not isinstance(value, dict):
        raise RunnerError("baseline all-files inventory must be a path-to-SHA256 object")
    result: dict[str, str] = {}
    for name, item in value.items():
        if isinstance(item, str):
            digest = item
        elif isinstance(item, dict):
            digest = item.get("sha256")
        else:
            raise RunnerError(f"baseline inventory entry has invalid shape: {name}")
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise RunnerError(f"baseline inventory entry lacks a SHA256: {name}")
        result[name.replace("\\", "/")] = digest
    return result


def _verify_baseline(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str]]:
    """Bind the prior WORLD run to the independently verified seal values.

    The old run intentionally has no copied ``execution_plan.json``. Its
    external-plan provenance is established by the supplied protocol, rows,
    results and complete-file inventory hashes.
    """
    root = Path(args.baseline_run).resolve(strict=True)
    expected = {
        "evaluation_protocol.json": args.baseline_protocol_sha256,
        "evaluation_rows.jsonl": args.baseline_rows_sha256,
        "evaluation_results.jsonl": args.baseline_results_sha256,
        "all_files_sha256.json": args.baseline_all_files_sha256,
        "code_seal.json": args.baseline_code_seal_sha256,
        "run_status.json": args.baseline_run_status_sha256,
    }
    observed = {}
    for relative, wanted in expected.items():
        observed[relative] = _must_sha(root / relative, wanted, f"baseline/{relative}")
    inventory = _inventory_map(root / "all_files_sha256.json")
    for relative in ("evaluation_protocol.json", "evaluation_rows.jsonl", "evaluation_results.jsonl"):
        if inventory.get(relative) != observed[relative]:
            raise RunnerError(f"baseline inventory does not bind its {relative}")
    status_path = root / "run_status.json"
    status = _json(status_path)
    if (
        status.get("status") != "complete"
        or int(status.get("planned_contexts", -1)) != 160
        or int(status.get("completed_contexts", -1)) != 160
        or int(status.get("failed_contexts", -1)) != 0
        or int(status.get("planned_rows", -1)) != 12_160
        or int(status.get("scored_rows", -1)) != 12_160
        or int(status.get("failed_rows", -1)) != 0
    ):
        raise RunnerError("baseline WORLD run is not the sealed complete 160-context/12,160-row run")
    protocol = _json(root / "evaluation_protocol.json")
    anchor = protocol.get("anchor")
    if not isinstance(anchor, dict) or (anchor.get("k"), anchor.get("column"), anchor.get("official_offset")) != (3, 18, -3):
        raise RunnerError("baseline WORLD run does not preserve the frozen k=3/column18/official-offset=-3 anchor")
    if status.get("forward_flags_required_and_asserted") != EXPECTED_FLAGS:
        raise RunnerError("baseline WORLD run does not seal the required per-forward deterministic flags")
    forward_flags = status.get("flags_at_forward")
    if not isinstance(forward_flags, list) or not forward_flags:
        raise RunnerError("baseline WORLD run lacks observed per-forward flag records")
    for entry in forward_flags:
        try:
            parsed_flags = json.loads(entry) if isinstance(entry, str) else entry
        except (TypeError, json.JSONDecodeError) as exc:
            raise RunnerError("baseline per-forward flag record is not valid JSON") from exc
        if parsed_flags != EXPECTED_FLAGS:
            raise RunnerError("baseline WORLD run observed a non-frozen per-forward flag state")
    code_seal_path = root / "code_seal.json"
    code_seal = _json(code_seal_path)
    execution_plan_path = Path(args.baseline_execution_plan).resolve(strict=True)
    execution_plan_sha = _must_sha(execution_plan_path, args.baseline_execution_plan_sha256, "baseline/execution_plan")
    if (
        code_seal.get("protocol_sha256") != observed["evaluation_protocol.json"]
        or code_seal.get("row_manifest_sha256") != observed["evaluation_rows.jsonl"]
        or code_seal.get("execution_plan_sha256") != execution_plan_sha
    ):
        raise RunnerError("baseline code seal does not bind the frozen plan/protocol/row ledger")
    runtime_flags_path = Path(args.baseline_runtime_flags).resolve(strict=True)
    runtime_flags_sha = _must_sha(runtime_flags_path, args.baseline_runtime_flags_sha256, "baseline/runtime_flags")
    seal_metadata_path = Path(args.baseline_seal_metadata).resolve(strict=True)
    seal_metadata_sha = _must_sha(seal_metadata_path, args.baseline_seal_metadata_sha256, "baseline/seal_metadata")
    if inventory.get("code_seal.json") != observed["code_seal.json"]:
        raise RunnerError("baseline inventory does not bind code_seal.json")
    if inventory.get("run_status.json") != observed["run_status.json"]:
        raise RunnerError("baseline inventory does not bind run_status.json")
    if protocol.get("row_manifest_sha256") != observed["evaluation_rows.jsonl"]:
        raise RunnerError("baseline protocol does not bind the supplied row manifest")
    if status.get("protocol_sha256") != observed["evaluation_protocol.json"] or status.get("row_manifest_sha256") != observed["evaluation_rows.jsonl"]:
        raise RunnerError("baseline final status does not bind the supplied protocol and row ledger")
    if int(protocol.get("row_count", protocol.get("planned_rows", 12_160))) != 12_160:
        raise RunnerError("baseline protocol does not declare 12,160 planned rows")
    rows = [line for line in (root / "evaluation_rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    results = [line for line in (root / "evaluation_results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 12_160 or len(results) != 12_160:
        raise RunnerError(f"baseline row ledgers have wrong cardinality: {len(rows)}/{len(results)}")
    return {
        "path": str(root), "expected_file_sha256": observed,
        "status_sha256": _must_sha(status_path, inventory.get("run_status.json", ""), "baseline/run_status.json"),
        "code_seal_sha256": observed["code_seal.json"],
        "execution_plan_path": str(execution_plan_path), "execution_plan_sha256": execution_plan_sha,
        "runtime_flags_path": str(runtime_flags_path), "runtime_flags_sha256": runtime_flags_sha,
        "seal_metadata_path": str(seal_metadata_path), "seal_metadata_sha256": seal_metadata_sha,
        "context_count": 160, "planned_rows": 12_160, "scored_rows": 12_160,
        "inventory_entry_count": len(inventory),
    }, inventory


def _read_wave(ref: Mapping[str, Any], *, raw_pcm16: bool, role: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = Path(str(ref["path"])).resolve(strict=True)
    expected_file_sha = str(ref.get("file_sha256", ref.get("sha256", "")))
    if not expected_file_sha or _sha(path) != expected_file_sha:
        raise RunnerError(f"{role} file SHA mismatch: {path}")
    sample_rate, values = wavfile.read(str(path))
    if sample_rate != 16_000 or values.ndim != 1 or values.size == 0:
        raise RunnerError(f"{role} must be nonempty mono 16 kHz PCM: {sample_rate}/{values.shape}/{values.dtype}")
    expected_rate = int(ref.get("sample_rate_hz", sample_rate))
    if sample_rate != expected_rate:
        raise RunnerError(f"{role} sample rate differs from manifest: {sample_rate} != {expected_rate}")
    if raw_pcm16:
        if values.dtype != np.int16:
            raise RunnerError(f"{role} must be stored as PCM16")
        pcm16 = values.copy()
        wave = (pcm16.astype(np.float64) / 32768.0).astype(np.float32)
        units = pcm16.astype(np.float64)
        subtype = "PCM_16"
    else:
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise RunnerError(f"{role} must be finite float32 WORLD PCM")
        wave = values.copy()
        units, _ = ops.pcm_to_syncnet_integer_units(wave)
        pcm16 = None
        subtype = "FLOAT32"
    expected_samples = int(ref.get("samples", ref.get("sample_count", -1)))
    if expected_samples > 0 and len(wave) != expected_samples:
        raise RunnerError(f"{role} sample count differs from manifest: {len(wave)} != {expected_samples}")
    expected_pcm_sha = ref.get("float32_pcm_sha256", ref.get("raw_float32_pcm_sha256"))
    wave_sha = ops.array_sha256(wave)
    if expected_pcm_sha and wave_sha != expected_pcm_sha:
        raise RunnerError(f"{role} decoded float32 PCM identity mismatch")
    pcm16_sha = hashlib.sha256(np.asarray(pcm16, dtype="<i2").tobytes(order="C")).hexdigest() if pcm16 is not None else None
    if raw_pcm16:
        expected_pcm16_sha = ref.get("pcm16_sha256", ref.get("raw_decoded_pcm16_sha256"))
        if not expected_pcm16_sha or pcm16_sha != expected_pcm16_sha:
            raise RunnerError(f"{role} decoded PCM16 identity mismatch")
    unit_sha = ops.array_sha256(units)
    return wave, units, {
        "path": str(path), "file_sha256": expected_file_sha, "sample_rate_hz": int(sample_rate),
        "sample_count": int(len(wave)), "subtype": subtype, "float32_pcm_sha256": wave_sha,
        "syncnet_units_dtype": str(units.dtype), "syncnet_units_sha256": unit_sha,
        "pcm16_sha256": pcm16_sha,
        "peak_abs": float(np.max(np.abs(wave))), "rms": float(np.sqrt(np.mean(wave.astype(np.float64) ** 2))),
    }


def _source_manifests(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = {
        "shape_manifest": Path(args.shape_manifest).resolve(strict=True),
        "parent_manifest": Path(args.parent_manifest).resolve(strict=True),
        "input_manifest": Path(args.input_manifest).resolve(strict=True),
        "tts_manifest": Path(args.tts_manifest).resolve(strict=True),
        "shape_qa": Path(args.shape_qa).resolve(strict=True),
        "cal_qa": Path(args.cal_qa).resolve(strict=True),
        "evaluation_freeze": Path(args.evaluation_freeze).resolve(strict=True),
        "design": Path(args.design).resolve(strict=True),
        "metric_addendum": Path(args.metric_addendum).resolve(strict=True),
        "face_registry": Path(args.face_registry).resolve(strict=True),
    }
    expected = {
        "shape_manifest": args.shape_manifest_sha256,
        "parent_manifest": args.parent_manifest_sha256,
        "input_manifest": args.input_manifest_sha256,
        "tts_manifest": args.tts_manifest_sha256,
        "shape_qa": args.shape_qa_sha256,
        "cal_qa": args.cal_qa_sha256,
        "evaluation_freeze": args.evaluation_freeze_sha256,
        "design": DESIGN_SHA256,
        "metric_addendum": METRIC_ADDENDUM_SHA256,
        "face_registry": args.face_registry_sha256,
    }
    seals = {name: {"path": str(path), "sha256": _must_sha(path, expected[name], name)} for name, path in paths.items()}
    shape = _json(paths["shape_manifest"])
    parent = _json(paths["parent_manifest"])
    source = _json(paths["input_manifest"])
    tts = _json(paths["tts_manifest"])
    shape_qa = _json(paths["shape_qa"])
    cal_qa = _json(paths["cal_qa"])
    evaluation_freeze = _json(paths["evaluation_freeze"])
    face_registry = _json(paths["face_registry"])
    if shape.get("manifest_sha256") and _fingerprint(shape) != shape.get("manifest_sha256"):
        raise RunnerError("shape EVAL physical manifest self-hash is invalid")
    if source.get("manifest_sha256") and _fingerprint(source) != source.get("manifest_sha256"):
        raise RunnerError("natural/TTS source manifest self-hash is invalid")
    if tts.get("manifest_sha256") and _fingerprint(tts) != tts.get("manifest_sha256"):
        raise RunnerError("TTS manifest self-hash is invalid")
    if shape.get("status") != "EVAL_FACTORIAL_WAVEFORMS_READY_FOR_INDEPENDENT_REVIEW" or shape.get("split") != "evaluation":
        raise RunnerError("shape-factor EVAL physical manifest is not ready")
    if (
        shape.get("analysis_spec_raw_sha256") != DESIGN_SHA256
        or shape.get("metric_addendum_raw_sha256") != METRIC_ADDENDUM_SHA256
    ):
        raise RunnerError("shape-factor physical manifest is not bound to the frozen EVAL design/addendum")
    if (
        shape_qa.get("status") != "PASS"
        or shape_qa.get("scope") != "EVAL physical waveforms only; 80 utterances / 40 speakers / 160 directions / 800 WORLD waveforms; no video/model/scoring inputs"
        or shape_qa.get("evaluation_inputs_read") is not True
        or shape_qa.get("effects_read") is not False
        or shape_qa.get("evaluation_outcomes_read") is not False
        or shape_qa.get("model_forward_performed") is not False
        or shape_qa.get("input_report", {}).get("raw_sha256") != expected["shape_manifest"]
        or shape_qa.get("bound_world_manifest_raw_sha256") != expected["shape_manifest"]
        or shape_qa.get("bound_analysis_source_sha256") != shape.get("analysis_source_sha256")
        or shape_qa.get("bound_analysis_spec_sha256") != DESIGN_SHA256
    ):
        raise RunnerError("independent shape-factor physical QA is not a no-effect-read PASS")
    if (
        evaluation_freeze.get("status") != "FROZEN"
        or evaluation_freeze.get("scope") != "shape_factorial_physical_waveform_production_only"
        or evaluation_freeze.get("manifest_sha256") != _fingerprint(evaluation_freeze)
        or evaluation_freeze.get("evaluation_waveform_production_authorized") is not True
        or evaluation_freeze.get("gpu_model_execution_authorized") is not False
        or evaluation_freeze.get("intended_direction_rows") != 160
        or evaluation_freeze.get("intended_cell_waveforms") != 800
    ):
        raise RunnerError("physical EVAL root freeze must authorize waveforms and block GPU model execution")
    if (
        cal_qa.get("status") != "PASS"
        or not str(cal_qa.get("scope", "")).startswith("CAL only;")
        or len(cal_qa.get("records", [])) != 40
        or cal_qa.get("direction_rows_verified") != 40
        or cal_qa.get("cell_waveforms_verified") != 200
        or cal_qa.get("manifest_sha256") != _fingerprint(cal_qa)
        or any(cal_qa.get(name) is not False for name in ("evaluation_inputs_read", "evaluation_outcomes_read", "effects_read", "model_forward_performed"))
    ):
        raise RunnerError("required CAL-only QA is not a self-hashed 40-row/200-cell no-EVAL/no-forward PASS")
    if face_registry.get("face_id") != "primary":
        raise RunnerError("only the frozen primary face is supported by this design")
    return shape, parent, source, {"tts": tts, "shape_qa": shape_qa, "cal_qa": cal_qa, "face_registry": face_registry, "seals": seals}


def _input_rows(source: dict[str, Any], tts: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    source_records = source.get("records", [])
    tts_records = tts.get("records", [])
    eval_source = [row for row in source_records if row.get("split") == "evaluation"]
    eval_tts = {row.get("id"): row for row in tts_records if row.get("split") == "evaluation"}
    all_tts = {row.get("id"): row for row in tts_records}
    if len(eval_source) != 80 or len(eval_tts) != 80 or len(all_tts) != 100:
        raise RunnerError(f"cohort size mismatch: source_eval={len(eval_source)} tts_eval={len(eval_tts)} tts_all={len(all_tts)}")
    id_fields = tuple(tts.get("input_identity_fields", ("id", "speaker", "text", "split", "ref_sha256")))
    for row in source_records:
        trow = all_tts.get(row.get("id"))
        if trow is None or {field: row.get(field) for field in id_fields} != trow.get("input_identity"):
            raise RunnerError(f"natural/TTS source identity mismatch: {row.get('id')}")
    if len({row.get("speaker") for row in eval_source}) != 40:
        raise RunnerError("frozen EVAL source cohort is not 40 speakers")
    return eval_source, eval_tts, all_tts


def _audio_path_for_source(args: argparse.Namespace, sample_id: str, speaker: str, source_row: dict[str, Any], tts_row: dict[str, Any], condition: str) -> dict[str, Any]:
    utterance = sample_id.split("_", 1)[1]
    root = Path(args.input_root).resolve(strict=True)
    if condition == "natural":
        path = root / "audio" / speaker / f"{utterance}.wav"
        sha = str(source_row["ref_sha256"])
    else:
        path = root / "tts_outputs" / "audio_16k_canonical" / f"{sample_id}.wav"
        sha = str(tts_row["qwen_canonical_16k"]["sha256"])
    if _sha(path) != sha:
        raise RunnerError(f"raw source WAV identity differs from frozen manifest: {sample_id}/{condition}")
    return {"path": str(path.resolve()), "file_sha256": sha}


def _old_endpoint_ref(parent_record: dict[str, Any], direction: str, arm: str) -> dict[str, Any]:
    try:
        value = parent_record["directions"][direction]["arms"][arm]
        ref = {"path": str(value["path"]), "file_sha256": str(value["file_sha256"]), "float32_pcm_sha256": str(value["float32_pcm_sha256"]), "samples": int(value["samples"]), "sample_rate_hz": int(value["sample_rate_hz"])}
    except (KeyError, TypeError, ValueError) as exc:
        raise RunnerError(f"missing original {arm} endpoint for {direction}: {exc}") from exc
    return ref


def _plan_contexts(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shape, parent, source, aux = _source_manifests(args)
    baseline_seal, baseline_inventory = _verify_baseline(args)
    baseline_root = Path(args.baseline_run).resolve(strict=True)
    eval_source, eval_tts, all_tts = _input_rows(source, aux["tts"])
    # The frozen producer binds split at the manifest level. Its direction
    # rows carry identity/status but omit a redundant row-level split field.
    shape_rows = shape.get("records", [])
    parent_rows = {row.get("id"): row for row in parent.get("records", []) if row.get("split") == "evaluation"}
    if len(shape_rows) != 160 or len(parent_rows) != 80:
        raise RunnerError(f"manifest row counts differ: shape={len(shape_rows)} parent={len(parent_rows)}")
    expected_order = [(row["id"], direction) for row in eval_source for direction in DIRECTIONS]
    observed_order = [(row.get("id"), row.get("direction")) for row in shape_rows]
    if observed_order != expected_order:
        raise RunnerError("shape manifest order differs from frozen source IDs × direction order")
    contexts = []
    seen = set()
    for ordered_index, row in enumerate(shape_rows):
        sample_id, direction = str(row["id"]), str(row["direction"])
        if row.get("split") not in (None, "evaluation") or row.get("status") != "PASS":
            raise RunnerError(f"shape direction row is not a successful EVAL row: {sample_id}/{direction}")
        if (sample_id, direction) in seen:
            raise RunnerError(f"duplicate shape context: {sample_id}/{direction}")
        seen.add((sample_id, direction))
        source_row = eval_source[ordered_index // 2]
        reported_pair_index = row.get("ordered_index")
        if reported_pair_index is not None and int(reported_pair_index) != ordered_index // 2:
            raise RunnerError(f"shape direction row pair order differs from frozen source order: {sample_id}/{direction}")
        tts_row = eval_tts.get(sample_id)
        parent_record = parent_rows.get(sample_id)
        if tts_row is None or parent_record is None or row.get("speaker") != source_row.get("speaker"):
            raise RunnerError(f"shape row source identity mismatch: {sample_id}/{direction}")
        recipient = "natural" if direction == "tts_to_natural" else "tts"
        donor = "tts" if recipient == "natural" else "natural"
        if row.get("recipient_condition") != recipient or row.get("donor_condition") != donor:
            raise RunnerError(f"direction/recipient identity mismatch: {sample_id}/{direction}")
        if row.get("source_identity_sha256") != parent_record.get("source_identity_sha256"):
            raise RunnerError(f"shape row does not preserve parent source identity: {sample_id}")
        source_audio = _audio_path_for_source(args, sample_id, str(source_row["speaker"]), source_row, tts_row, recipient)
        parent_audio = parent_record.get(recipient, {})
        raw_ref = row.get("scoring_audio", {}).get("A_RAW", {})
        if str(Path(raw_ref.get("path", "")).resolve()) != str(Path(parent_audio.get("staged_audio_path", "")).resolve()) or raw_ref.get("file_sha256") != parent_audio.get("staged_audio_sha256"):
            raise RunnerError(f"A_RAW is not the frozen recipient staged PCM: {sample_id}/{direction}")
        if parent_audio.get("source_audio_sha256") != source_audio["file_sha256"]:
            raise RunnerError(f"parent raw source identity differs from source manifest: {sample_id}/{direction}")
        if (
            raw_ref.get("float32_pcm_sha256") != parent_audio.get("raw_float32_pcm_sha256")
            or raw_ref.get("pcm16_sha256") != parent_audio.get("raw_decoded_pcm16_sha256")
            or int(raw_ref.get("samples", -1)) != int(parent_audio.get("raw_decoded_samples", -2))
            or int(raw_ref.get("sample_rate_hz", -1)) != int(parent_audio.get("sample_rate_hz", -2))
        ):
            raise RunnerError(f"A_RAW decoded float/PCM16 hashes or clock differ from the sealed parent source: {sample_id}/{direction}")
        rt_ref = row.get("scoring_audio", {}).get("A_RT", {})
        expected_rt = _old_endpoint_ref(parent_record, direction, "RT")
        if (
            str(Path(rt_ref.get("path", "")).resolve()) != str(Path(expected_rt["path"]).resolve())
            or rt_ref.get("file_sha256") != expected_rt["file_sha256"]
            or rt_ref.get("float32_pcm_sha256") != expected_rt["float32_pcm_sha256"]
            or int(rt_ref.get("samples", -1)) != expected_rt["samples"]
            or int(rt_ref.get("sample_rate_hz", -1)) != expected_rt["sample_rate_hz"]
        ):
            raise RunnerError(f"A_RT differs from the original sealed recipient RT waveform: {sample_id}/{direction}")
        cells = row.get("cells")
        if not isinstance(cells, dict) or set(cells) != set(CELLS):
            raise RunnerError(f"shape row does not provide the exact five fresh generation cells: {sample_id}/{direction}")
        cell_refs = {}
        for cell in CELLS:
            item = cells[cell]
            if item.get("status") != "ok":
                raise RunnerError(f"physical cell is not successful: {sample_id}/{direction}/{cell}")
            wave = item.get("waveform", {})
            params = item.get("parameter_npz", {})
            if not wave.get("path") or not wave.get("file_sha256") or not params.get("path") or not params.get("file_sha256"):
                raise RunnerError(f"physical cell lacks sealed waveform/parameter refs: {sample_id}/{direction}/{cell}")
            cell_refs[cell] = {
                "waveform": {"path": str(Path(wave["path"]).resolve(strict=True)), "file_sha256": str(wave["file_sha256"]), "float32_pcm_sha256": str(wave["float32_pcm_sha256"]), "samples": int(wave["samples"]), "sample_rate_hz": int(wave["sample_rate_hz"])},
                "parameter_npz": {"path": str(Path(params["path"]).resolve(strict=True)), "file_sha256": str(params["file_sha256"])},
            }
            if _sha(cell_refs[cell]["waveform"]["path"]) != cell_refs[cell]["waveform"]["file_sha256"] or _sha(cell_refs[cell]["parameter_npz"]["path"]) != cell_refs[cell]["parameter_npz"]["file_sha256"]:
                raise RunnerError(f"physical cell artifact SHA mismatch: {sample_id}/{direction}/{cell}")
        if row.get("phone_map", {}).get("file_sha256") is None:
            raise RunnerError(f"shape row lacks exact phone-map identity: {sample_id}/{direction}")
        phone_map_path = Path(row["phone_map"]["path"]).resolve(strict=True)
        if _sha(phone_map_path) != row["phone_map"]["file_sha256"]:
            raise RunnerError(f"phone-map SHA mismatch: {sample_id}/{direction}")
        phone_map = _json(phone_map_path)
        if phone_map.get("id") != sample_id or phone_map.get("speaker") != source_row["speaker"] or phone_map.get("recipient_condition") != recipient or phone_map.get("donor_condition") != donor:
            raise RunnerError(f"phone-map identity differs from shape context: {sample_id}/{direction}")
        baseline_context_relative = f"contexts/{sample_id}__{direction}.npz"
        baseline_context_path = baseline_root / baseline_context_relative
        baseline_context_sha = baseline_inventory.get(baseline_context_relative)
        if not baseline_context_sha:
            raise RunnerError(f"baseline inventory omits required context: {baseline_context_relative}")
        _must_sha(baseline_context_path, baseline_context_sha, f"baseline/{baseline_context_relative}")
        baseline_videos = {}
        for old_arm in ("RT", "SP"):
            relative = f"videos/{sample_id}/{direction}/V_{old_arm}.mkv"
            path = baseline_root / relative
            expected_video_sha = baseline_inventory.get(relative)
            if not expected_video_sha:
                raise RunnerError(f"baseline inventory omits required video: {relative}")
            _must_sha(path, expected_video_sha, f"baseline/{relative}")
            baseline_videos[old_arm] = {"path": str(path), "relative_path": relative, "sha256": expected_video_sha}
        old_endpoints = {name: _old_endpoint_ref(parent_record, direction, name) for name in ("RT", "SP")}
        for old_arm, ref in old_endpoints.items():
            _read_wave(ref, raw_pcm16=False, role=f"preflight/{sample_id}/{direction}/old_{old_arm}")
        with np.load(baseline_context_path, allow_pickle=False) as old_context:
            for old_arm in ("RT", "SP"):
                z_key, v_key = f"arm_{old_arm}_z", f"arm_{old_arm}_lip_embeddings"
                if z_key not in old_context or v_key not in old_context:
                    raise RunnerError(f"baseline context lacks bridge arrays: {sample_id}/{direction}/{old_arm}")
                old_z, old_v = old_context[z_key], old_context[v_key]
                if (
                    old_z.ndim != 2 or old_z.shape[1] != 512 or old_v.ndim != 2
                    or old_v.shape[1] != 512 or old_z.shape[0] != old_v.shape[0] + 5
                    or not np.isfinite(old_z).all() or not np.isfinite(old_v).all()
                ):
                    raise RunnerError(f"baseline context bridge array shape/value mismatch: {sample_id}/{direction}/{old_arm}")
        ctx = {
            "ordered_eval_index": int(ordered_index), "sample_id": sample_id, "speaker_id": str(source_row["speaker"]),
            "direction": direction, "recipient_condition": recipient, "donor_condition": donor,
            "source_identity_sha256": str(row["source_identity_sha256"]), "source_audio": source_audio,
            "scoring_audio": {name: dict(row["scoring_audio"][name]) for name in AUDIO_DOMAINS},
            "phone_map": {"path": str(phone_map_path), "file_sha256": str(row["phone_map"]["file_sha256"])},
            "cells": cell_refs,
            "original_endpoints": old_endpoints,
            "baseline_artifacts": {
                "context": {"path": str(baseline_context_path), "relative_path": baseline_context_relative, "sha256": baseline_context_sha},
                "videos": baseline_videos,
            },
            "phone_map_record": phone_map,
        }
        contexts.append(ctx)
    seals = aux["seals"]
    return contexts, {"input_seals": seals, "baseline_seal": baseline_seal, "physical_status": shape.get("status"), "physical_row_count": len(shape_rows), "parent_row_count": len(parent_rows), "source_pair_count": len(eval_source), "tts_all_pair_count": len(all_tts), "face_registry": aux["face_registry"]}


def _expected_flags() -> dict[str, bool]:
    return {"cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled())}


def _set_flags() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    observed = _expected_flags()
    if observed != EXPECTED_FLAGS or torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32:
        raise RunnerError(f"deterministic policy is not active: {observed}")


def _audio_embeddings(units: np.ndarray, syncnet: Any, batch_size: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mfcc, embeddings, meta = evaluate._audio_forward(syncnet, units, device="cuda", batch_size=batch_size)
    expected = len(units) // 640 - 5
    if expected < 1 or len(embeddings) != expected:
        raise RunnerError(f"SyncNet audio window count differs from official endpoint: {len(embeddings)} != {expected}")
    return mfcc, embeddings.astype(np.float32, copy=False), {"mfcc_sha256": ops.array_sha256(mfcc), "embedding_sha256": ops.array_sha256(embeddings), "window_count": int(len(embeddings)), "frontend": meta}


def _render_one(
    run_dir: Path, temp_root: Path, ctx: dict[str, Any], cell: str, wave: np.ndarray,
    z: np.ndarray, frame: np.ndarray, face: dict[str, Any], face_cache: dict[int, list[Any]],
    model: Any, syncnet: Any, ffmpeg: Path, wav2lip_batch_size: int, syncnet_batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    sample_id, direction = ctx["sample_id"], ctx["direction"]
    video_path = run_dir / "videos" / sample_id / direction / f"{cell}.mkv"
    video_meta = wav2lip.render_video(model, z, frame, face["generation_box_xyxy"], face_cache, video_path, ffmpeg, batch_size=wav2lip_batch_size, device="cuda")
    frame_count = len(z)
    if int(video_meta.get("frame_count", -1)) != frame_count:
        raise RunnerError(f"encoded frame count differs from latent count: {sample_id}/{direction}/{cell}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RunnerError(f"generated video cannot be decoded: {video_path}")
    decoded = 0
    try:
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            decoded += 1
    finally:
        capture.release()
    if decoded != frame_count:
        raise RunnerError(f"decoded frame count differs from latent count: {sample_id}/{direction}/{cell}")
    temp = Path(tempfile.mkdtemp(prefix=f"shape_{sample_id}_{cell}_", dir=temp_root))
    crop_path, jpeg_dir = temp / "crop.mkv", temp / "jpegs"
    try:
        crop_meta = wav2lip.crop_score_video(video_path, face["score_box"], crop_path, ffmpeg)
        if int(crop_meta.get("frame_count", -1)) != frame_count:
            raise RunnerError(f"score-crop frame count differs: {sample_id}/{direction}/{cell}")
        jpeg_frames, jpeg_meta = scoring.extract_official_jpegs(crop_path, jpeg_dir, ffmpeg)
        if int(jpeg_meta.get("frame_count", -1)) != frame_count:
            raise RunnerError(f"official JPEG count differs: {sample_id}/{direction}/{cell}")
        lip = scoring.lip_embeddings_from_frames(syncnet, jpeg_frames, device="cuda", batch_size=syncnet_batch_size)
        expected_lip = frame_count - 5
        if len(lip) != expected_lip:
            raise RunnerError(f"official SyncNet V clock differs from N-5: {sample_id}/{direction}/{cell}: {len(lip)} != {expected_lip}")
        visual_path = run_dir / "visuals" / sample_id / direction / f"{cell}.npz"
        visual_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(visual_path, latent_z_float32=z.astype(np.float32, copy=False), lip_embeddings_float32=lip.astype(np.float32, copy=False))
        metadata = {
            "condition": cell, "video_path": str(video_path.relative_to(run_dir)), "video_sha256": _sha(video_path),
            "video_size_bytes": video_path.stat().st_size, "encoded_frame_count": int(video_meta["frame_count"]),
            "decoded_frame_count": decoded, "crop_frame_count": int(crop_meta["frame_count"]),
            "official_jpeg_frame_count": int(jpeg_meta["frame_count"]), "lip_window_count": int(len(lip)),
            "latent_shape": list(z.shape), "latent_sha256": ops.array_sha256(z),
            "lip_shape": list(lip.shape), "lip_sha256": ops.array_sha256(lip),
            "visual_npz": str(visual_path.relative_to(run_dir)), "visual_npz_sha256": _sha(visual_path),
            "crop_sha256": _sha(crop_path), "crop_policy": crop_meta.get("crop_method"),
            "jpeg_directory": "ephemeral; removed after scoring",
        }
        _write_json(visual_path.with_suffix(".json"), metadata)
        return lip.astype(np.float32, copy=False), metadata
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def _pixel_bridge(old_path: Path, fresh_path: Path) -> dict[str, Any]:
    def frames(path: Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RunnerError(f"cannot open bridge video: {path}")
        out = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                out.append(frame)
        finally:
            cap.release()
        return out
    old, fresh = frames(old_path), frames(fresh_path)
    if len(old) != len(fresh) or not old:
        raise RunnerError(f"q00/q11 video bridge frame count mismatch: {old_path} {len(old)} vs {fresh_path} {len(fresh)}")
    total_abs = 0.0
    total_pixels = 0
    max_abs = 0
    exact_frames = 0
    for a, b in zip(old, fresh):
        if a.shape != b.shape:
            raise RunnerError("bridge decoded frame shapes differ")
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
        total_abs += float(delta.sum(dtype=np.float64))
        total_pixels += int(delta.size)
        max_abs = max(max_abs, int(delta.max(initial=0)))
        exact_frames += int(np.array_equal(a, b))
    return {"old_video_sha256": _sha(old_path), "fresh_video_sha256": _sha(fresh_path), "frame_count": len(old), "pixel_bit_exact_frames": exact_frames, "mean_absolute_pixel_difference": total_abs / max(1, total_pixels), "maximum_absolute_pixel_difference": max_abs}


def _build_plan(args: argparse.Namespace) -> dict[str, Any]:
    contexts, source = _plan_contexts(args)
    assets = _execution_assets(args)
    rows = []
    for ctx in contexts:
        key = f"{ctx['sample_id']}__{ctx['direction']}"
        for cell in CELLS:
            for audio_domain in AUDIO_DOMAINS:
                row_id = f"{key}__{cell}__{audio_domain}"
                rows.append({
                    "ordered_eval_index": ctx["ordered_eval_index"], "row_id": row_id,
                    "context_key": key, "sample_id": ctx["sample_id"], "speaker_id": ctx["speaker_id"],
                    "split": "evaluation", "direction": ctx["direction"], "condition": cell,
                    "audio_domain": audio_domain, "face_id": "primary", "visual_id": cell,
                    "score_npz": f"scores/{key}__{cell}__{audio_domain}.npz",
                    "visual_npz": f"visuals/{key}/{cell}.npz", "context_npz": f"contexts/{key}.npz",
                })
    if len(contexts) != 160 or len(rows) != 1600 or len({row["row_id"] for row in rows}) != 1600:
        raise RunnerError(f"planned cardinality changed: contexts={len(contexts)} rows={len(rows)}")
    protocol = {
        "schema_version": 1, "classification": "fresh five-cell WORLD shape/dynamics generation and fixed RAW/RT scoring; no cohort effects",
        "created_at_utc": _now(), "design_sha256": DESIGN_SHA256, "metric_addendum_sha256": METRIC_ADDENDUM_SHA256,
        "calibration_source_sha256": CAL_SOURCE_SHA256, "input_seals": source["input_seals"],
        "baseline_seal": source["baseline_seal"], "execution_assets": assets,
        "direction_order": list(DIRECTIONS), "cell_order": list(CELLS), "audio_domains": list(AUDIO_DOMAINS),
        "cohort": {"pairs": 80, "speakers": 40, "contexts": 160, "videos": 800, "score_rows": 1600, "fixed_face": "primary"},
        "anchor": {"k": 3, "column": ANCHOR_COLUMN, "official_offset_frames": -3},
        "common_support_policy": "minimum full V-window count across all five fresh videos and full A-row counts for A_RAW/A_RT; matrices retain full V×A domains; padding uses each full A domain",
        "runtime_policy": {"device": "cuda", "wav2lip_batch_size": args.wav2lip_batch_size, "syncnet_batch_size": args.syncnet_batch_size, "forward_flags": EXPECTED_FLAGS, "cudnn_allow_tf32": False, "matmul_allow_tf32": False, "cublas_workspace_config": ":4096:8"},
        "contexts": contexts, "rows": rows,
    }
    protocol["protocol_sha256"] = hashlib.sha256(_canonical({key: value for key, value in protocol.items() if key != "protocol_sha256"})).hexdigest()
    return protocol


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    control = Path(args.control_dir).resolve()
    if control.exists() and any(control.iterdir()):
        raise RunnerError(f"control directory must be empty: {control}")
    protocol = _build_plan(args)
    control.mkdir(parents=True, exist_ok=True)
    rows_path = control / "evaluation_rows.jsonl"
    rows_path.write_bytes(b"".join(_canonical(row) + b"\n" for row in protocol["rows"]))
    protocol["row_manifest_sha256"] = _sha(rows_path)
    protocol["protocol_sha256"] = hashlib.sha256(_canonical({key: value for key, value in protocol.items() if key != "protocol_sha256"})).hexdigest()
    _write_json(control / "execution_plan.json", protocol)
    return {"status": "preflight_pass", "contexts": len(protocol["contexts"]), "rows": len(protocol["rows"]), "execution_plan": str(control / "execution_plan.json"), "execution_plan_sha256": _sha(control / "execution_plan.json"), "protocol_sha256": protocol["protocol_sha256"], "input_seal_count": len(protocol["input_seals"])}


def _execution_assets(args: argparse.Namespace) -> dict[str, Any]:
    import python_speech_features

    wav_root = Path(args.wav2lip_root).resolve(strict=True)
    sync_root = Path(args.syncnet_root).resolve(strict=True)
    wav_ckpt = Path(args.wav2lip_checkpoint).resolve(strict=True)
    sync_ckpt = Path(args.syncnet_checkpoint).resolve(strict=True)
    ffmpeg = Path(args.ffmpeg).resolve(strict=True)
    face_path = Path(args.primary_face).resolve(strict=True)
    face_registry = _json(Path(args.face_registry).resolve(strict=True))
    face = cv2.imread(str(face_path))
    if face is None or tuple(face.shape[:2]) != (512, 512):
        raise RunnerError("primary face must decode as 512x512")
    if _sha(face_path) != face_registry.get("face", {}).get("sha256"):
        raise RunnerError("primary face differs from the frozen face registry")
    for path, name in ((wav_ckpt, "Wav2Lip checkpoint"), (sync_ckpt, "SyncNet checkpoint"), (ffmpeg, "ffmpeg")):
        if not path.is_file():
            raise RunnerError(f"missing {name}: {path}")
    if not wav_root.is_dir() or not sync_root.is_dir():
        raise RunnerError("Wav2Lip and SyncNet source roots must be directories")
    wav_tree, sync_tree = _python_source_tree(wav_root), _python_source_tree(sync_root)
    if not wav_tree["file_count"] or not sync_tree["file_count"]:
        raise RunnerError("Wav2Lip or SyncNet source tree is empty")
    ffmpeg_version = subprocess.run([str(ffmpeg), "-version"], check=True, capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
    try:
        import librosa
        librosa_version = str(librosa.__version__)
    except Exception as exc:
        raise RunnerError(f"librosa frontend cannot be resolved during preflight: {exc}") from exc
    package_root = Path(__file__).resolve().parent
    package_sources = {
        name: {"path": str(package_root / name), "sha256": _sha(package_root / name)}
        for name in ("evaluate.py", "ops.py", "scoring.py", "wav2lip.py", "__init__.py")
        if (package_root / name).is_file()
    }
    if set(package_sources) != {"evaluate.py", "ops.py", "scoring.py", "wav2lip.py", "__init__.py"}:
        raise RunnerError("local portable scoring package is missing a bound source file")
    psf_tree = _python_source_tree(Path(python_speech_features.__file__).resolve().parent)
    return {
        "wav2lip_root": str(wav_root), "wav2lip_checkpoint": str(wav_ckpt), "wav2lip_checkpoint_sha256": _sha(wav_ckpt),
        "wav2lip_source_tree": wav_tree,
        "syncnet_root": str(sync_root), "syncnet_checkpoint": str(sync_ckpt), "syncnet_checkpoint_sha256": _sha(sync_ckpt),
        "syncnet_source_tree": sync_tree,
        "portable_scoring_package": package_sources, "python_speech_features_source_tree": psf_tree,
        "ffmpeg": str(ffmpeg), "ffmpeg_sha256": _sha(ffmpeg), "ffmpeg_version": ffmpeg_version,
        "primary_face": str(face_path), "primary_face_sha256": _sha(face_path),
        "primary_face_shape": list(face.shape), "torch": str(torch.__version__), "cuda_build": str(torch.version.cuda), "numpy": np.__version__,
        "librosa": librosa_version,
        "runner_path": str(Path(__file__).resolve()), "runner_sha256": _sha(__file__),
    }


def _load_plan(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.control_dir).resolve(strict=True) / "execution_plan.json"
    if _sha(path) != args.plan_sha256:
        raise RunnerError("execution plan raw hash changed")
    plan = _json(path)
    expected = plan.get("protocol_sha256")
    computed = hashlib.sha256(_canonical({key: value for key, value in plan.items() if key != "protocol_sha256"})).hexdigest()
    if expected != computed or plan.get("status") not in (None, "preflight_pass"):
        raise RunnerError("execution plan protocol fingerprint/status invalid")
    contexts, source = _plan_contexts(args)
    assets = _execution_assets(args)
    if (
        _canonical(contexts) != _canonical(plan.get("contexts"))
        or _canonical(source["input_seals"]) != _canonical(plan.get("input_seals"))
        or _canonical(source["baseline_seal"]) != _canonical(plan.get("baseline_seal"))
        or _canonical(assets) != _canonical(plan.get("execution_assets"))
    ):
        raise RunnerError("physical inputs changed after preflight")
    return plan


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RunnerError("CUDA is required")
    plan = _load_plan(args)
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty run directory: {run_dir}")
    temp_root = Path(args.jpeg_temp_root).resolve()
    temp_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("contexts", "visuals", "videos", "scores"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    rows_path = run_dir / "evaluation_rows.jsonl"
    rows = plan["rows"]
    rows_path.write_bytes(b"".join(_canonical(row) + b"\n" for row in rows))
    row_sha = _sha(rows_path)
    protocol = {key: value for key, value in plan.items() if key not in {"contexts", "rows"}}
    protocol.update({"row_manifest_sha256": row_sha, "run_dir": str(run_dir), "jpeg_temp_root": str(temp_root), "run_started_at_utc": _now()})
    protocol["protocol_sha256"] = hashlib.sha256(_canonical({key: value for key, value in protocol.items() if key != "protocol_sha256"})).hexdigest()
    _write_json(run_dir / "evaluation_protocol.json", protocol)
    _write_json(run_dir / "code_seal.json", {"runner_path": str(Path(__file__).resolve()), "runner_sha256": _sha(__file__), "row_manifest_sha256": row_sha, "execution_plan_sha256": args.plan_sha256, "protocol_sha256": _sha(run_dir / "evaluation_protocol.json")})
    _write_json(run_dir / "run_status.json", {"status": "running", "planned_contexts": 160, "completed_contexts": 0, "failed_contexts": 0, "planned_rows": 1600, "scored_rows": 0})
    _append_jsonl(run_dir / "attempts.jsonl", {"event": "run_started", "status": "running", "at": _now()})

    torch.manual_seed(20261001)
    np.random.seed(20261001)
    _set_flags()
    compat = wav2lip.install_librosa_compatibility_shim()
    model, model_meta = wav2lip.load_model(Path(args.wav2lip_root).resolve(), args.wav2lip_checkpoint, device="cuda")
    syncnet, syncnet_meta = scoring.load_syncnet(Path(args.syncnet_root).resolve(), args.syncnet_checkpoint, device="cuda")
    frame = cv2.imread(str(Path(args.primary_face).resolve()))
    face_registry = _json(Path(args.face_registry).resolve(strict=True))
    face = face_registry["face"]
    completed, failed = 0, []
    result_path = run_dir / "evaluation_results.jsonl"
    _append_jsonl(run_dir / "runtime_assets.jsonl", {"torch": str(torch.__version__), "cuda": str(torch.version.cuda), "device": torch.cuda.get_device_name(0), "wav2lip": model_meta, "syncnet": syncnet_meta, "frontend": compat})
    for ctx in plan["contexts"]:
        key = f"{ctx['sample_id']}__{ctx['direction']}"
        started = time.perf_counter()
        try:
            audio: dict[str, dict[str, Any]] = {}
            for domain in AUDIO_DOMAINS:
                wave, units, metadata = _read_wave(ctx["scoring_audio"][domain], raw_pcm16=(domain == "A_RAW"), role=f"{key}/{domain}")
                mfcc, emb, emb_meta = _audio_embeddings(units, syncnet, args.syncnet_batch_size)
                audio[domain] = {"wave": wave, "units": units, "mfcc": mfcc, "emb": emb, "metadata": {**metadata, **emb_meta}}
            cell_data: dict[str, dict[str, Any]] = {}
            for cell in CELLS:
                wave, _, meta = _read_wave(ctx["cells"][cell]["waveform"], raw_pcm16=False, role=f"{key}/{cell}")
                if abs(meta["rms"] - FIXED_RMS) > 1e-9:
                    raise RunnerError(f"cell RMS differs from frozen fixed RMS: {key}/{cell}")
                mel, chunks, starts, front = wav2lip.mel_chunks(wave, args.wav2lip_root)
                z = wav2lip.encode_latents(model, chunks, device="cuda", batch_size=args.wav2lip_batch_size)
                if len(z) != len(starts):
                    raise RunnerError(f"latent/mel frame count mismatch: {key}/{cell}")
                cell_data[cell] = {"wave": wave, "meta": meta, "mel": mel, "chunks": chunks, "starts": np.asarray(starts, dtype=np.int32), "z": z, "frontend": front}
            starts0 = cell_data[CELLS[0]]["starts"]
            for cell in CELLS[1:]:
                if not np.array_equal(starts0, cell_data[cell]["starts"]):
                    raise RunnerError(f"five cells do not share the exact Wav2Lip mel-start clock: {key}/{cell}")
            if len({len(cell_data[cell]["wave"]) for cell in CELLS}) != 1:
                raise RunnerError(f"five cell waveforms do not share the recipient sample clock: {key}")
            batch_counts = set()
            for cell in CELLS:
                n = len(cell_data[cell]["z"])
                batch_counts.add(min(args.wav2lip_batch_size, n))
                if n % args.wav2lip_batch_size:
                    batch_counts.add(n % args.wav2lip_batch_size)
            face_cache = wav2lip.cache_face_features(model, frame, face["generation_box_xyxy"], sorted(batch_counts), device="cuda")
            lips, visual_meta = {}, {}
            for cell in CELLS:
                _set_flags()
                lips[cell], visual_meta[cell] = _render_one(run_dir, temp_root, ctx, cell, cell_data[cell]["wave"], cell_data[cell]["z"], frame, face, face_cache, model, syncnet, Path(args.ffmpeg).resolve(), args.wav2lip_batch_size, args.syncnet_batch_size)
            common_count = min(*(len(lips[cell]) for cell in CELLS), *(len(audio[d]["emb"]) for d in AUDIO_DOMAINS))
            if common_count < 1:
                raise RunnerError(f"no common five-video/two-audio SyncNet prefix: {key}")
            supports = evaluate._support_indices(common_count)
            context_arrays: dict[str, np.ndarray] = {}
            for domain in AUDIO_DOMAINS:
                for name in ("wave", "units", "mfcc", "emb"):
                    context_arrays[f"{domain}_{name}"] = audio[domain][name]
            for cell in CELLS:
                for name in ("mel", "chunks", "starts", "z"):
                    context_arrays[f"{cell}_{name}"] = cell_data[cell][name]
                context_arrays[f"{cell}_V"] = lips[cell]
            for name, value in supports.items():
                context_arrays[f"support_{name}"] = value.astype(np.int32, copy=False)
            context_path = run_dir / "contexts" / f"{key}.npz"
            np.savez_compressed(context_path, **context_arrays)
            context_meta = {
                **ctx, "key": key, "common_query_count": int(common_count),
                "audio_metadata": {domain: audio[domain]["metadata"] for domain in AUDIO_DOMAINS},
                "cell_metadata": {cell: {"waveform": cell_data[cell]["meta"], "mel_frontend": cell_data[cell]["frontend"], "mel_sha256": ops.array_sha256(cell_data[cell]["mel"]), "chunks_sha256": ops.array_sha256(cell_data[cell]["chunks"]), "starts_sha256": ops.array_sha256(cell_data[cell]["starts"]), "z_sha256": ops.array_sha256(cell_data[cell]["z"]), "V_sha256": ops.array_sha256(lips[cell]), "visual": visual_meta[cell]} for cell in CELLS},
                "supports": {name: values.tolist() for name, values in supports.items()},
                "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": ops.array_sha256(value)} for name, value in context_arrays.items()},
                "context_npz": {"path": str(context_path.relative_to(run_dir)), "sha256": _sha(context_path), "bytes": context_path.stat().st_size},
                "mel_clock_policy": "all five fresh cell mel starts must be bit-identical; exact Wav2Lip frontend and official final chunk fallback",
            }
            _write_json(context_path.with_suffix(".json"), context_meta)
            score_meta = {}
            for cell in CELLS:
                for domain in AUDIO_DOMAINS:
                    _set_flags()
                    lip = lips[cell]
                    aud = audio[domain]["emb"]
                    raw_cross, raw_zero = evaluate._cross_distance_matrix(lip, aud, "cuda")
                    unit_cross, unit_zero = evaluate._cross_distance_matrix(evaluate._row_unit(lip), evaluate._row_unit(aud), "cuda")
                    lag_raw = evaluate._lag_matrix_from_cross(raw_cross, raw_zero, common_count, len(aud), vshift=VSHIFT)
                    lag_unit = evaluate._lag_matrix_from_cross(unit_cross, unit_zero, common_count, len(aud), vshift=VSHIFT)
                    if raw_cross.shape != (len(lip), len(aud)) or unit_cross.shape != raw_cross.shape or raw_zero.shape != (len(lip),) or unit_zero.shape != (len(lip),):
                        raise RunnerError(f"full V×A distance matrix was truncated: {key}/{cell}/{domain}")
                    row_id = f"{key}__{cell}__{domain}"
                    score_path = run_dir / "scores" / f"{row_id}.npz"
                    score_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(score_path,
                        cross_distance_raw_float32=raw_cross, cross_distance_to_zero_raw_float32=raw_zero, lag_distance_raw_float32=lag_raw,
                        cross_distance_unit_float32=unit_cross, cross_distance_to_zero_unit_float32=unit_zero, lag_distance_unit_float32=lag_unit,
                        common_query_prefix_indices_int32=np.arange(common_count, dtype=np.int32),
                        **{f"support_{name}_indices_int32": value.astype(np.int32, copy=False) for name, value in supports.items()},
                        lag_columns_int32=np.arange(31, dtype=np.int32), lag_k_int32=np.arange(-15, 16, dtype=np.int32))
                    result = {
                        "row_id": row_id, "context_key": key, "sample_id": ctx["sample_id"], "speaker_id": ctx["speaker_id"],
                        "ordered_eval_index": ctx["ordered_eval_index"], "direction": ctx["direction"], "condition": cell,
                        "audio_domain": domain, "face_id": "primary", "status": "scored",
                        "score_npz": str(score_path.relative_to(run_dir)), "score_npz_sha256": _sha(score_path),
                        "context_npz": str(context_path.relative_to(run_dir)), "context_npz_sha256": _sha(context_path),
                        "visual_npz": visual_meta[cell]["visual_npz"], "visual_npz_sha256": visual_meta[cell]["visual_npz_sha256"],
                        "video_path": visual_meta[cell]["video_path"], "video_sha256": visual_meta[cell]["video_sha256"],
                        "full_visual_rows": int(len(lip)), "full_audio_rows": int(len(aud)), "common_query_count": int(common_count),
                        "fixed_anchor_k": 3, "fixed_anchor_column": ANCHOR_COLUMN, "fixed_anchor_official_offset": -3,
                        "raw_support_metrics": evaluate._metric_supports(lag_raw, ANCHOR_COLUMN),
                        "unit_support_metrics": evaluate._metric_supports(lag_unit, ANCHOR_COLUMN),
                        "array_hashes": {"cross_raw": ops.array_sha256(raw_cross), "zero_raw": ops.array_sha256(raw_zero), "lag_raw": ops.array_sha256(lag_raw), "cross_unit": ops.array_sha256(unit_cross), "zero_unit": ops.array_sha256(unit_zero), "lag_unit": ops.array_sha256(lag_unit)},
                    }
                    _append_jsonl(result_path, result)
                    score_meta[row_id] = result

            # Explicit fresh q00/q11 bridges to original RT/SP generation artifacts.
            bridges = {}
            for cell, old_arm in (("q00", "RT"), ("q11", "SP")):
                old_ref = ctx["original_endpoints"][old_arm]
                old_wave, _, old_wave_meta = _read_wave(old_ref, raw_pcm16=False, role=f"{key}/old_{old_arm}")
                new_wave = cell_data[cell]["wave"]
                if old_wave.shape != new_wave.shape:
                    raise RunnerError(f"fresh {cell} waveform clock differs from old {old_arm}: {key}")
                diff = new_wave.astype(np.float64) - old_wave.astype(np.float64)
                if float(np.max(np.abs(diff), initial=0.0)) > 1e-6:
                    raise RunnerError(f"fresh {cell} does not bridge to original {old_arm} waveform within 1e-6: {key}")
                old_ctx_path = Path(ctx["baseline_artifacts"]["context"]["path"]).resolve(strict=True)
                if not old_ctx_path.is_file():
                    raise RunnerError(f"old WORLD context is missing for bridge: {old_ctx_path}")
                with np.load(old_ctx_path, allow_pickle=False) as old_ctx:
                    old_z = old_ctx[f"arm_{old_arm}_z"].copy()
                    old_v = old_ctx[f"arm_{old_arm}_lip_embeddings"].copy()
                new_z, new_v = cell_data[cell]["z"], lips[cell]
                if old_z.shape != new_z.shape or old_v.shape != new_v.shape:
                    raise RunnerError(f"fresh {cell}/old {old_arm} latent or V clock differs: {key}")
                old_video_path = Path(ctx["baseline_artifacts"]["videos"][old_arm]["path"]).resolve(strict=True)
                fresh_video_path = run_dir / visual_meta[cell]["video_path"]
                bridges[cell] = {
                    "old_arm": old_arm, "old_wave_file_sha256": old_wave_meta["file_sha256"],
                    "wave_max_abs_error": float(np.max(np.abs(diff), initial=0.0)), "wave_rms_error": float(np.sqrt(np.mean(diff * diff))),
                    "latent_bit_exact": bool(np.array_equal(old_z, new_z)), "latent_max_abs_error": float(np.max(np.abs(old_z.astype(np.float64) - new_z.astype(np.float64)), initial=0.0)),
                    "V_bit_exact": bool(np.array_equal(old_v, new_v)), "V_mean_abs_error": float(np.mean(np.abs(old_v.astype(np.float64) - new_v.astype(np.float64)))),
                    "pixel_bridge": _pixel_bridge(old_video_path, fresh_video_path),
                    "old_context_path": str(old_ctx_path), "old_context_sha256": _sha(old_ctx_path),
                }
            context_meta["endpoint_bridges"] = bridges
            _write_json(context_path.with_suffix(".json"), context_meta)
            completed += 1
            _append_jsonl(run_dir / "attempts.jsonl", {"event": "context_complete", "status": "complete", "context_key": key, "completed_contexts": completed, "planned_contexts": 160, "scored_rows": len(score_meta), "elapsed_seconds": time.perf_counter() - started})
            _write_json(run_dir / "run_status.json", {"status": "running", "planned_contexts": 160, "completed_contexts": completed, "failed_contexts": len(failed), "planned_rows": 1600, "scored_rows": sum(1 for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip())})
        except Exception as exc:
            failure = {"context_key": key, "sample_id": ctx["sample_id"], "direction": ctx["direction"], "error_type": type(exc).__name__, "error": str(exc)[:2000], "traceback": traceback.format_exc()[-4000:]}
            failed.append(failure)
            _append_jsonl(run_dir / "attempts.jsonl", {"event": "context_failed", "status": "failed", **failure})
            _write_json(run_dir / "run_status.json", {"status": "running", "planned_contexts": 160, "completed_contexts": completed, "failed_contexts": len(failed), "planned_rows": 1600, "scored_rows": sum(1 for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()) if result_path.exists() else 0})
    final_rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()] if result_path.exists() else []
    final = {
        "status": "complete_with_failures" if failed else "complete", "classification": "fresh five-cell Wav2Lip videos and fixed RAW/RT full-domain SyncNet matrices; no cohort effects",
        "planned_contexts": 160, "completed_contexts": completed, "failed_contexts": len(failed), "planned_rows": 1600,
        "scored_rows": len(final_rows), "failed_rows": 0, "unique_video_count": len(list((run_dir / "videos").rglob("*.mkv"))),
        "protocol_sha256": _sha(run_dir / "evaluation_protocol.json"), "row_manifest_sha256": row_sha,
        "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "cuda": str(torch.version.cuda), "device": torch.cuda.get_device_name(0), "numpy": np.__version__},
        "failed_rows": max(0, len(rows) - len(final_rows)), "failed_contexts_detail": failed, "finished_at_utc": _now(),
    }
    _write_json(run_dir / "run_status.json", final)
    inventory = {str(path.relative_to(run_dir)).replace("\\", "/"): _sha(path) for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "all_files_sha256.json"}
    _write_json(run_dir / "all_files_sha256.json", inventory)
    return final


def _shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shape-manifest", required=True)
    parser.add_argument("--shape-manifest-sha256", required=True)
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--parent-manifest-sha256", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--tts-manifest", required=True)
    parser.add_argument("--tts-manifest-sha256", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--shape-qa", required=True)
    parser.add_argument("--shape-qa-sha256", required=True)
    parser.add_argument("--cal-qa", required=True)
    parser.add_argument("--cal-qa-sha256", required=True)
    parser.add_argument("--evaluation-freeze", required=True)
    parser.add_argument("--evaluation-freeze-sha256", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--metric-addendum", required=True)
    parser.add_argument("--face-registry", required=True)
    parser.add_argument("--face-registry-sha256", required=True)
    parser.add_argument("--primary-face", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--baseline-protocol-sha256", required=True)
    parser.add_argument("--baseline-rows-sha256", required=True)
    parser.add_argument("--baseline-results-sha256", required=True)
    parser.add_argument("--baseline-all-files-sha256", required=True)
    parser.add_argument("--baseline-code-seal-sha256", required=True)
    parser.add_argument("--baseline-run-status-sha256", required=True)
    parser.add_argument("--baseline-execution-plan", required=True)
    parser.add_argument("--baseline-execution-plan-sha256", required=True)
    parser.add_argument("--baseline-runtime-flags", required=True)
    parser.add_argument("--baseline-runtime-flags-sha256", required=True)
    parser.add_argument("--baseline-seal-metadata", required=True)
    parser.add_argument("--baseline-seal-metadata-sha256", required=True)
    parser.add_argument("--wav2lip-root", required=True)
    parser.add_argument("--wav2lip-checkpoint", required=True)
    parser.add_argument("--syncnet-root", required=True)
    parser.add_argument("--syncnet-checkpoint", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--wav2lip-batch-size", type=int, default=64)
    parser.add_argument("--syncnet-batch-size", type=int, default=20)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("preflight")
    _shared(pre)
    pre.add_argument("--control-dir", required=True)
    run = sub.add_parser("run")
    _shared(run)
    run.add_argument("--control-dir", required=True)
    run.add_argument("--plan-sha256", required=True)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--jpeg-temp-root", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = _preflight(args) if args.command == "preflight" else _run(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        return 0 if result.get("status") in {"preflight_pass", "complete"} else 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:2000]}, ensure_ascii=False, separators=(",", ":")), file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
