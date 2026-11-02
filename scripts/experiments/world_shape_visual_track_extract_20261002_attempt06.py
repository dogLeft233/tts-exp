#!/usr/bin/env python3
"""Metadata plan, CPU tracking, and geometry-blind availability freeze for μ/ρ."""
from __future__ import annotations

import argparse
import atexit
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # metadata-only build-plan runs in the stock host Python
    np = None  # type: ignore[assignment]

DESIGN_SHA256 = "0394781ac97369b1dc8d3eaf79ec43f61b888d52cac8ff585eff4f0a14f896c8"
EVAL_DESIGN_SHA256 = "e17d257e43a78af6d7a291641ade6825b61c430ba51b8052383cab54dbea31fb"
EXECUTION_PIN_CORRECTION_SHA256 = "cfea922c2cba344b757005ab5f838ac6808db2c70808b3665da177b099e65de6"
WORLD_CELLS = ("q00", "q10", "q01", "q11", "SELF_GRID")
DIRECTIONS = ("tts_to_natural", "natural_to_tts")
EXPECTED_CONTEXTS = 160
EXPECTED_SAMPLES = 80
EXPECTED_SPEAKERS = 40
EXPECTED_VIDEOS = 800
FPS = 25.0
EXPECTED_FACE_PATH = "/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/inputs/primary_face.png"
EXPECTED_FACE_SHA256 = "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6"
EXPECTED_FACE_SHAPE = [512, 512, 3]
LATENCY_S = -0.085
SEED = 20260930
DRAWS = 20_000
CORE_SHA256 = "3a28f2e7060e8458859455bcb0283f8d1b98558f16cc4e185efd0902d73ea15b"
TONE_SHA256 = "b725b5feb5c3f30bea754f73bf897d606e04f59940fe8f4fdb13f269306fce98"
VIDEO_FEATURES_SHA256 = "2ce7006030e7f93b139cbbc8f5152c9b39719bea2fe7fbf16fbf18110cce5ff4"
LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
TONE_MARKS = frozenset(("\u02e5", "\u02e6", "\u02e7", "\u02e8", "\u02e9"))
_FEATURES: Any = None
_LANDMARKER: Any = None


class VisualRunError(RuntimeError):
    pass


def _ensure_numpy() -> Any:
    global np
    if np is None:
        import numpy as imported_numpy
        np = imported_numpy
    return np


def _configure_cpu_runtime() -> None:
    """Disable accelerator visibility and cap native CPU thread pools."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canon_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def embedded_fingerprint(value: Any) -> tuple[str | None, str | None]:
    if not isinstance(value, dict):
        return None, None
    for key in ("manifest_sha256", "plan_sha256", "mapping_sha256"):
        if isinstance(value.get(key), str):
            actual = bytes_sha256(canon_bytes({k: v for k, v in value.items() if k != key}))
            return key, actual
    return None, None


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise VisualRunError(f"expected_json_object:{path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise VisualRunError(f"expected_json_object:{path.name}:{number}")
                rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def seal_object(value: dict[str, Any], field: str) -> dict[str, Any]:
    out = dict(value)
    out[field] = bytes_sha256(canon_bytes({k: v for k, v in out.items() if k != field}))
    return out


def verify_object_fingerprint(value: dict[str, Any], field: str) -> str:
    expected = value.get(field)
    actual = bytes_sha256(canon_bytes({k: v for k, v in value.items() if k != field}))
    if expected != actual:
        raise VisualRunError(f"embedded_fingerprint_mismatch:{field}")
    return actual


def _read_inventory(path: Path) -> dict[str, str]:
    value = read_json(path)
    if "files" in value and isinstance(value["files"], dict):
        value = value["files"]
    if not all(isinstance(k, str) and isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for k, v in value.items()):
        raise VisualRunError("model_all_files_inventory_schema_invalid")
    return value


def _source_ref(path: Path, *, embedded_key: str | None = None) -> dict[str, Any]:
    target = path.resolve(strict=True)
    raw = sha256_file(target)
    entry: dict[str, Any] = {"path": str(target), "raw_sha256": raw, "bytes": target.stat().st_size}
    if target.suffix.lower() == ".json":
        value = read_json(target)
        key, computed = embedded_fingerprint(value)
        if embedded_key:
            if key != embedded_key:
                raise VisualRunError(f"embedded_hash_field_mismatch:{target.name}:{key}")
        if key:
            entry["embedded_fingerprint_field"] = key
            entry["embedded_fingerprint_value"] = value.get(key)
            entry["embedded_fingerprint_recomputed"] = computed
            if value.get(key) != computed:
                raise VisualRunError(f"embedded_fingerprint_invalid:{target.name}:{key}")
    return entry


def _find_records(root: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = root.get("records")
    if not isinstance(rows, list):
        raise VisualRunError("source_records_missing")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = str(row.get("id", row.get("sample_id", "")))
        if not identifier or identifier in out:
            raise VisualRunError("source_record_identity_duplicate_or_missing")
        out[identifier] = row
    return out


def _file_payload_value(ref: dict[str, Any]) -> dict[str, Any]:
    return {k: ref.get(k) for k in ("path", "sha256", "raw_sha256", "bytes", "payload_sha256") if k in ref}


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    model_root = args.model_run.resolve(strict=True)
    model_protocol_path = model_root / "evaluation_protocol.json"
    model_rows_path = model_root / "evaluation_rows.jsonl"
    model_status_path = model_root / "run_status.json"
    model_inventory_path = model_root / "all_files_sha256.json"
    model_code_seal_path = model_root / "code_seal.json"
    model_results_path = model_root / "evaluation_results.jsonl"
    model_execution_plan_path = args.model_execution_plan.resolve(strict=True)
    model_runtime_flags_path = args.model_runtime_flags.resolve(strict=True)
    protocol = read_json(model_protocol_path)
    status = read_json(model_status_path)
    model_rows = read_jsonl(model_rows_path)
    inventory = _read_inventory(model_inventory_path)
    model_execution_plan = read_json(model_execution_plan_path)
    model_runtime_flags = read_json(model_runtime_flags_path)
    model_code_seal = read_json(model_code_seal_path)
    if status.get("status") != "complete" or status.get("completed_contexts") != EXPECTED_CONTEXTS or status.get("failed_contexts") != 0 or status.get("unique_video_count") != EXPECTED_VIDEOS or status.get("scored_rows") != 1600 or status.get("failed_rows") != 0:
        raise VisualRunError("model14_run_status_not_complete")
    if protocol.get("cohort", {}).get("contexts") != EXPECTED_CONTEXTS or protocol.get("cohort", {}).get("videos") != EXPECTED_VIDEOS or tuple(protocol.get("cell_order", [])) != WORLD_CELLS or tuple(protocol.get("direction_order", [])) != DIRECTIONS or tuple(protocol.get("audio_domains", [])) != ("A_RAW", "A_RT"):
        raise VisualRunError("model14_protocol_design_mismatch")
    if protocol.get("design_sha256") != EVAL_DESIGN_SHA256:
        raise VisualRunError("model14_design_binding_mismatch")
    if protocol.get("run_dir") and args.model_runtime_run_dir and protocol["run_dir"] != args.model_runtime_run_dir:
        raise VisualRunError("model_runtime_run_path_differs_from_sealed_protocol")
    runtime_run = str(args.model_runtime_run_dir or protocol.get("run_dir") or "").rstrip("/")
    if not runtime_run.startswith("/"):
        raise VisualRunError("model_runtime_run_dir_must_be_absolute_posix")

    restore_receipt = read_json(args.model_restore_receipt.resolve(strict=True))
    if restore_receipt.get("status") != "FULL_SEAL_AND_ROW_COVERAGE_PASS" or restore_receipt.get("verified_file_count") != len(inventory) or restore_receipt.get("all_files_sha256_json_sha256") != sha256_file(model_inventory_path) or restore_receipt.get("run_status", {}).get("status") != "complete":
        raise VisualRunError("model14_restore_receipt_binding_mismatch")

    if (len(model_execution_plan.get("contexts", [])) != EXPECTED_CONTEXTS
            or len(model_execution_plan.get("rows", [])) != 1600
            or model_execution_plan.get("row_manifest_sha256") != sha256_file(model_rows_path)
            or tuple(model_execution_plan.get("cell_order", [])) != WORLD_CELLS
            or tuple(model_execution_plan.get("direction_order", [])) != DIRECTIONS
            or tuple(model_execution_plan.get("audio_domains", [])) != ("A_RAW", "A_RT")
            or model_execution_plan.get("design_sha256") != EVAL_DESIGN_SHA256):
        raise VisualRunError("model14_execution_plan_cohort_or_row_binding_mismatch")
    row_identity_fields = ("row_id", "sample_id", "speaker_id", "direction", "condition", "audio_domain", "face_id", "ordered_eval_index", "split", "context_key", "context_npz", "visual_id", "visual_npz")
    for index, (planned_row, materialized_row) in enumerate(zip(model_execution_plan["rows"], model_rows)):
        if any(planned_row.get(key) != materialized_row.get(key) for key in row_identity_fields):
            raise VisualRunError(f"model14_execution_plan_row_identity_mismatch:{index}")
    model_face = model_execution_plan.get("execution_assets", {})
    if (model_face.get("primary_face") != EXPECTED_FACE_PATH
            or model_face.get("primary_face_sha256") != EXPECTED_FACE_SHA256
            or model_face.get("primary_face_shape") != EXPECTED_FACE_SHAPE):
        raise VisualRunError("model14_execution_plan_primary_face_binding_mismatch")
    execution_assets = model_execution_plan.get("execution_assets", {})
    model_runner_sha = execution_assets.get("runner_sha256")
    required_flags = {"deterministic_algorithms": True, "cudnn_deterministic": True, "cudnn_benchmark": False, "cudnn_allow_tf32": False, "matmul_allow_tf32": False}
    def flags_match(record: Any) -> bool:
        return isinstance(record, dict) and {key: record.get(key) for key in required_flags} == required_flags
    if (model_code_seal.get("execution_plan_sha256") != sha256_file(model_execution_plan_path)
            or model_code_seal.get("protocol_sha256") != sha256_file(model_protocol_path)
            or model_code_seal.get("row_manifest_sha256") != sha256_file(model_rows_path)
            or model_code_seal.get("runner_sha256") != model_runner_sha
            or model_runtime_flags.get("runtime_forward_gate") != "pass"
            or model_runtime_flags.get("runner_exit_code") != 0
            or model_runtime_flags.get("flag_mismatch_count") != 0
            or model_runtime_flags.get("policy") != required_flags
            or not flags_match(model_runtime_flags.get("torch_flags_after_runner"))
            or model_runtime_flags.get("runner_sha256") != model_runner_sha
            or not flags_match(model_runtime_flags.get("torch_flags_after_policy_before_forward"))):
        raise VisualRunError("model14_execution_runtime_or_code_seal_mismatch")
    inventory_paths = (("evaluation_protocol.json", model_protocol_path), ("evaluation_rows.jsonl", model_rows_path), ("run_status.json", model_status_path), ("code_seal.json", model_code_seal_path), ("evaluation_results.jsonl", model_results_path))
    for rel, actual_path in inventory_paths:
        if inventory.get(rel) != sha256_file(actual_path):
            raise VisualRunError(f"model14_inventory_file_binding_mismatch:{rel}")

    physical = read_json(args.physical_manifest.resolve(strict=True))
    physical_plan = read_json(args.physical_plan.resolve(strict=True))
    physical_qa = read_json(args.physical_qa.resolve(strict=True))
    shape_manifest = read_json(args.shape_manifest.resolve(strict=True))
    shape_inventory = read_json(args.shape_output_inventory.resolve(strict=True))
    eligibility = read_json(args.eligibility.resolve(strict=True))
    grid_inventory = read_json(args.grid_inventory.resolve(strict=True))
    calibration = read_json(args.calibration.resolve(strict=True))
    calibration_freeze = read_json(args.calibration_freeze.resolve(strict=True))
    phone_mapping = read_json(args.phone_mapping.resolve(strict=True))
    for payload, key in ((physical, "manifest_sha256"), (physical_plan, "manifest_sha256"), (eligibility, "manifest_sha256"), (grid_inventory, None), (phone_mapping, "mapping_sha256")):
        if key and verify_object_fingerprint(payload, key) != payload.get(key):
            raise VisualRunError(f"source_payload_fingerprint_invalid:{key}")
    if physical.get("status") != "EVALUATION_WAVEFORMS_READY" and physical.get("status") != "PASS" and physical.get("status") != "COMPLETE":
        raise VisualRunError("physical_manifest_not_sealed_complete")
    if physical_qa.get("status") != "PASS" or physical_qa.get("direction_rows_verified") != EXPECTED_CONTEXTS or physical_qa.get("cell_waveforms_verified") != EXPECTED_VIDEOS or physical_qa.get("effects_read") is not False or physical_qa.get("evaluation_outcomes_read") is not False or physical_qa.get("model_forward_performed") is not False:
        raise VisualRunError("physical_qa_not_full_pass")
    shape_payload = verify_object_fingerprint(shape_manifest, "manifest_sha256")
    inventory_payload = verify_object_fingerprint(shape_inventory, "manifest_sha256")
    input_report = physical_qa.get("input_report", {})
    qa_inventory = physical_qa.get("input_output_inventory", {})
    if (input_report.get("raw_sha256") != sha256_file(args.shape_manifest.resolve(strict=True))
            or input_report.get("payload_sha256") != shape_payload
            or qa_inventory.get("raw_sha256") != sha256_file(args.shape_output_inventory.resolve(strict=True))
            or qa_inventory.get("payload_sha256") != inventory_payload
            or qa_inventory.get("entries_verified") != len(shape_inventory.get("files", {}))):
        raise VisualRunError("physical_qa_not_bound_to_generated_shape_manifest_and_inventory")
    if (shape_manifest.get("split") != "evaluation"
            or shape_manifest.get("status") != "EVAL_FACTORIAL_WAVEFORMS_READY_FOR_INDEPENDENT_REVIEW"
            or shape_manifest.get("intended_direction_rows") != EXPECTED_CONTEXTS
            or shape_manifest.get("complete_direction_rows") != EXPECTED_CONTEXTS
            or shape_manifest.get("intended_cell_waveforms") != EXPECTED_VIDEOS
            or shape_manifest.get("complete_cell_waveforms") != EXPECTED_VIDEOS
            or shape_manifest.get("failure_count") != 0
            or shape_manifest.get("effects_read") is not False
            or shape_manifest.get("evaluation_effects_read") is not False
            or shape_manifest.get("evaluation_outcomes_read") is not False
            or shape_manifest.get("model_forward_performed") is not False):
        raise VisualRunError("shape_evaluation_waveform_manifest_not_complete_or_scope_clean")
    if eligibility.get("status") != "MECHANICAL_APPLICATION_OF_FROZEN_CAL_RULES" or eligibility.get("records_total") != EXPECTED_SAMPLES:
        raise VisualRunError("original_eligibility_manifest_not_qualified")
    if grid_inventory.get("status") != "PASS" or len(grid_inventory.get("effective_recipient_grids", [])) != EXPECTED_CONTEXTS:
        raise VisualRunError("effective_grid_inventory_not_qualified")
    if calibration.get("status") != "CALIBRATION_COMPLETE" or calibration_freeze.get("status") != "FROZEN_CAL_ONLY" or calibration_freeze.get("evaluation_effects_read") is not False:
        raise VisualRunError("tone_visual_calibration_not_frozen_cal_only")
    if phone_mapping.get("mapping_sha256") != bytes_sha256(canon_bytes({k: v for k, v in phone_mapping.items() if k != "mapping_sha256"})):
        raise VisualRunError("tone_phone_mapping_fingerprint_invalid")
    if sha256_file(args.design.resolve(strict=True)) != DESIGN_SHA256:
        raise VisualRunError("visual_design_sha_mismatch")

    physical_by_id = _find_records(physical)
    eligibility_by_id = _find_records(eligibility)
    grid_by_context: dict[tuple[str, str], dict[str, Any]] = {}
    for ref in grid_inventory["effective_recipient_grids"]:
        key = (str(ref.get("sample_id")), str(ref.get("direction")))
        if key in grid_by_context:
            raise VisualRunError("duplicate_effective_recipient_grid")
        grid_by_context[key] = ref
    rows_by_context: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    context_order: list[tuple[str, str]] = []
    for row in model_rows:
        if row.get("split") != "evaluation" or row.get("face_id") != "primary" or row.get("audio_domain") not in ("A_RAW", "A_RT"):
            raise VisualRunError("model14_row_identity_outside_frozen_scope")
        key = (str(row.get("sample_id")), str(row.get("direction")))
        if key not in rows_by_context:
            context_order.append(key)
        rows_by_context[key].append(row)
    if len(rows_by_context) != EXPECTED_CONTEXTS or len(context_order) != EXPECTED_CONTEXTS:
        raise VisualRunError("model14_context_count_mismatch")
    for context_index, (planned_context, (sample, direction)) in enumerate(zip(model_execution_plan["contexts"], context_order)):
        grouped_context_rows = rows_by_context[(sample, direction)]
        context_speakers = {str(row.get("speaker_id")) for row in grouped_context_rows}
        context_indices = {int(row.get("ordered_eval_index", -1)) for row in grouped_context_rows}
        if (planned_context.get("sample_id") != sample
                or planned_context.get("direction") != direction
                or len(context_speakers) != 1
                or planned_context.get("speaker_id") != next(iter(context_speakers))
                or len(context_indices) != 1
                or planned_context.get("ordered_eval_index") != next(iter(context_indices))
                or set(planned_context.get("cells", {})) != set(WORLD_CELLS)):
            raise VisualRunError(f"model14_execution_plan_context_order_or_cell_mismatch:{context_index}")
    sample_speakers: dict[str, str] = {}
    speaker_samples: dict[str, set[str]] = defaultdict(set)
    row_records: list[dict[str, Any]] = []
    expected_cell_audio = {(cell, audio) for cell in WORLD_CELLS for audio in ("A_RAW", "A_RT")}
    for context_index, context in enumerate(context_order):
        sample, direction = context
        if direction not in DIRECTIONS or sample not in physical_by_id or sample not in eligibility_by_id or context not in grid_by_context:
            raise VisualRunError("model14_context_not_joined_to_physical_eligibility_grid")
        grouped = rows_by_context[context]
        identities = {(str(r.get("condition")), str(r.get("audio_domain"))) for r in grouped}
        if len(grouped) != 10 or identities != expected_cell_audio:
            raise VisualRunError("model14_context_cell_audio_coverage_mismatch")
        speaker_values = {str(r.get("speaker_id")) for r in grouped}
        orders = {int(r.get("ordered_eval_index", -1)) for r in grouped}
        if len(speaker_values) != 1 or len(orders) != 1:
            raise VisualRunError("model14_context_identity_inconsistent")
        speaker = next(iter(speaker_values))
        if sample in sample_speakers and sample_speakers[sample] != speaker:
            raise VisualRunError("model14_sample_speaker_alias_mismatch")
        sample_speakers[sample] = speaker
        speaker_samples[speaker].add(sample)
        physical_record = physical_by_id[sample]
        eligibility_record = eligibility_by_id[sample]
        recipient_condition = "natural" if direction == "tts_to_natural" else "tts"
        if physical_record.get("speaker") != speaker or eligibility_record.get("speaker") != speaker:
            raise VisualRunError("physical_or_eligibility_speaker_alias_mismatch")
        effective_grid = grid_by_context[context]
        if effective_grid.get("condition") != recipient_condition or effective_grid.get("timeline_policy") != "recipient clock":
            raise VisualRunError("effective_grid_condition_or_timeline_mismatch")
        direction_record = physical_record.get("directions", {}).get(direction)
        if not isinstance(direction_record, dict) or direction_record.get("recipient_condition") != recipient_condition:
            raise VisualRunError("physical_direction_record_mismatch")
        mapping = direction_record.get("mapping")
        if not isinstance(mapping, dict) or mapping.get("recipient_textgrid_sha256") != effective_grid.get("sha256"):
            raise VisualRunError("recipient_phone_map_grid_sha_mismatch")
        qualification_condition = eligibility_record.get(recipient_condition)
        if not isinstance(qualification_condition, dict):
            raise VisualRunError("original_eligibility_condition_missing")
        if not effective_grid.get("overlay_used") and qualification_condition.get("textgrid_sha256") != effective_grid.get("sha256"):
            raise VisualRunError("original_eligibility_textgrid_sha_mismatch")
        eligibility_qualified = eligibility_record.get("pair_qualified")
        physical_mfa_qualified = physical_record.get("mfa_pair_qualified")
        physical_pair_qualified = physical_record.get("pair_qualified")
        physical_alignment_qualified = physical_record.get("physical_alignment_pair_qualified")
        if not isinstance(eligibility_qualified, bool) or not isinstance(physical_mfa_qualified, bool):
            raise VisualRunError("original_mfa_pair_qualification_not_boolean")
        if physical_mfa_qualified != eligibility_qualified:
            raise VisualRunError("original_mfa_pair_qualification_conflict")
        if not isinstance(physical_pair_qualified, bool) or not isinstance(physical_alignment_qualified, bool):
            raise VisualRunError("physical_pair_qualification_provenance_not_boolean")
        # The physical overlay can qualify a row whose original MFA pair failed.
        # Keep this distinct from the original endpoint-eligibility decision.
        original_qualified = eligibility_qualified
        mapped_phones = mapping.get("recipient_phones")
        if not isinstance(mapped_phones, list) or not mapped_phones:
            raise VisualRunError("recipient_phone_inventory_missing")
        mapped_rows = []
        for phone in mapped_phones:
            interval = phone.get("interval_s")
            if phone.get("in_frozen_inventory") is not True or not isinstance(interval, list) or len(interval) != 2:
                raise VisualRunError("recipient_phone_inventory_entry_invalid")
            lo, hi = map(float, interval)
            if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
                raise VisualRunError("recipient_phone_interval_invalid")
            mapped_rows.append({"phone_index": int(phone["index"]), "label": str(phone.get("raw_label", "")), "normalized_label": str(phone.get("normalized_label", "")), "start_s": lo, "end_s": hi})
        if not effective_grid.get("overlay_used"):
            source_tokens = qualification_condition.get("tokens", [])
            if len(source_tokens) != len(mapped_rows):
                raise VisualRunError("grid_token_count_differs_from_original_qualification")
            duration = float(mapping.get("recipient_audio_duration_s"))
            phones = []
            for idx, (source, mapped) in enumerate(zip(source_tokens, mapped_rows)):
                label = str(source.get("raw_token", source.get("token", "")))
                source_start, source_end = float(source.get("start_s")), float(source.get("end_s"))
                clamped_end = min(source_end, duration)
                if label != mapped["label"] or source_start != mapped["start_s"] or not math.isclose(clamped_end, mapped["end_s"], rel_tol=0, abs_tol=1e-12):
                    raise VisualRunError(f"grid_token_identity_mismatch:{sample}:{direction}:{idx}")
                phones.append({"phone_index": idx, "label": label, "normalized_label": str(mapped["normalized_label"]), "start_s": source_start, "end_s": source_end})
        else:
            # Physical-only compound-token overlay is retained for full clock
            # provenance; original MFA qualification below remains false.
            phones = mapped_rows
        cell_refs: dict[str, Any] = {}
        for cell in WORLD_CELLS:
            pair = [r for r in grouped if r["condition"] == cell]
            by_audio = {r["audio_domain"]: r for r in pair}
            if set(by_audio) != {"A_RAW", "A_RT"}:
                raise VisualRunError("cell_audio_domain_alias_missing")
            raw_row, rt_row = by_audio["A_RAW"], by_audio["A_RT"]
            if raw_row.get("visual_npz") != rt_row.get("visual_npz") or raw_row.get("context_npz") != rt_row.get("context_npz"):
                raise VisualRunError("same_video_audio_domains_do_not_share_visual_artifact")
            cell_meta = read_json(model_root / str(raw_row["context_npz"]).replace(".npz", ".json"))
            if cell_meta.get("sample_id") != sample or cell_meta.get("direction") != direction:
                raise VisualRunError("model_context_sidecar_identity_mismatch")
            visual = cell_meta.get("cell_metadata", {}).get(cell, {}).get("visual")
            if not isinstance(visual, dict) or visual.get("condition") != cell:
                raise VisualRunError("model_visual_clock_metadata_missing")
            video_rel = str(visual.get("video_path", ""))
            video_sha = str(visual.get("video_sha256", ""))
            video_bytes = int(visual.get("video_size_bytes", -1))
            frame_count = int(visual.get("official_jpeg_frame_count", -1))
            if frame_count <= 0 or frame_count != int(visual.get("decoded_frame_count", -2)) or frame_count != int(visual.get("encoded_frame_count", -3)) or frame_count != int(visual.get("crop_frame_count", -4)):
                raise VisualRunError("model_video_clock_counts_inconsistent")
            if raw_row.get("visual_id") != cell:
                raise VisualRunError("model_visual_npz_row_sidecar_mismatch")
            if visual.get("video_sha256") != video_sha or int(visual.get("video_size_bytes", -1)) != video_bytes:
                raise VisualRunError("model_video_sidecar_identity_mismatch")
            visual_npz_rel = str(visual.get("visual_npz", ""))
            visual_npz_sha = str(visual.get("visual_npz_sha256", ""))
            if inventory.get(visual_npz_rel) != visual_npz_sha:
                raise VisualRunError("model_visual_npz_sidecar_not_in_inventory")
            if inventory.get(video_rel) != video_sha:
                raise VisualRunError("model_video_not_in_sealed_all_files_inventory")
            video_local = (model_root / video_rel).resolve(strict=True)
            if video_local.stat().st_size != video_bytes or sha256_file(video_local) != video_sha:
                raise VisualRunError("model_video_local_restore_sha_or_size_mismatch")
            cell_refs[cell] = {"video_relative_path": video_rel, "video_sha256": video_sha, "video_bytes": video_bytes, "official_frame_count": frame_count, "official_fps_hz": FPS, "visual_npz_relative_path": visual_npz_rel, "visual_npz_sha256": visual_npz_sha, "reported_a_raw_visual_npz_path": raw_row.get("visual_npz"), "reported_a_rt_visual_npz_path": rt_row.get("visual_npz"), "a_raw_row_id": raw_row["row_id"], "a_rt_row_id": rt_row["row_id"]}
        counts = {v["official_frame_count"] for v in cell_refs.values()}
        if len(counts) != 1:
            raise VisualRunError("five_cell_official_frame_clock_count_differs")
        orders = {int(r["ordered_eval_index"]) for r in grouped}
        if len(orders) != 1:
            raise VisualRunError("model_ordered_eval_index_mismatch")
        row_records.append({"context_order": context_index, "ordered_eval_index": next(iter(orders)), "sample_id": sample, "speaker_id": speaker, "split": "evaluation", "direction": direction, "recipient_condition": recipient_condition, "recipient_grid_sha256": effective_grid["sha256"], "recipient_grid_logical_path": effective_grid["path"], "overlay_used": bool(effective_grid.get("overlay_used")), "original_mfa_pair_qualified": original_qualified, "physical_pair_qualified": physical_pair_qualified, "physical_alignment_pair_qualified": physical_alignment_qualified, "physical_mfa_pair_qualified": physical_mfa_qualified, "original_eligibility_record_sha256": bytes_sha256(canon_bytes(eligibility_record)), "physical_record_sha256": bytes_sha256(canon_bytes(physical_record)), "recipient_textgrid_sha256": mapping["recipient_textgrid_sha256"], "recipient_phones": phones, "cells": cell_refs})
    if len(sample_speakers) != EXPECTED_SAMPLES or len(speaker_samples) != EXPECTED_SPEAKERS or any(len(v) != 2 for v in speaker_samples.values()):
        raise VisualRunError("cohort_must_be_80_clips_40_speakers_two_clips_each")
    if set(sample_speakers) != set(eligibility_by_id) or set(sample_speakers) != set(physical_by_id):
        raise VisualRunError("physical_eligibility_cohort_set_mismatch")
    source_paths = {
        "tracking_extractor_source": Path(__file__).resolve(),
        "visual_design": args.design,
        "evaluation_design": args.evaluation_design,
        "execution_pin_correction": args.execution_pin_correction,
        "waveform_source_analysis_plan": args.model_analysis_plan,
        "waveform_source_root_freeze": args.model_root_freeze,
        "model_execution_plan": model_execution_plan_path,
        "model_runtime_flags": model_runtime_flags_path,
        "model_code_seal": model_code_seal_path,
        "model_results": model_results_path,
        "model_protocol": model_protocol_path,
        "model_rows": model_rows_path,
        "model_status": model_status_path,
        "model_all_files_inventory": model_inventory_path,
        "model_restore_receipt": args.model_restore_receipt,
        "physical_manifest": args.physical_manifest,
        "physical_plan": args.physical_plan,
        "physical_independent_qa": args.physical_qa,
        "shape_evaluation_manifest": args.shape_manifest,
        "shape_output_inventory": args.shape_output_inventory,
        "original_mfa_eligibility": args.eligibility,
        "effective_grid_inventory": args.grid_inventory,
        "visual_calibration": args.calibration,
        "visual_calibration_freeze": args.calibration_freeze,
        "tone_phone_mapping": args.phone_mapping,
        "landmarker_asset": args.landmarker,
        "feature_helper": args.video_features,
        "visual_core_source": args.visual_core,
        "tone_adapter_source": args.tone_adapter,
    }
    source_refs = {}
    for role, path in source_paths.items():
        source_refs[role] = _source_ref(path)
    if (source_refs["visual_design"]["raw_sha256"] != DESIGN_SHA256
            or source_refs["evaluation_design"]["raw_sha256"] != EVAL_DESIGN_SHA256
            or source_refs["execution_pin_correction"]["raw_sha256"] != EXECUTION_PIN_CORRECTION_SHA256):
        raise VisualRunError("frozen_design_sources_changed")
    waveform_analysis_plan = read_json(args.model_analysis_plan.resolve(strict=True))
    waveform_root_freeze = read_json(args.model_root_freeze.resolve(strict=True))
    analysis_payload = verify_object_fingerprint(waveform_analysis_plan, "manifest_sha256")
    freeze_payload = verify_object_fingerprint(waveform_root_freeze, "manifest_sha256")
    if waveform_analysis_plan.get("status") != "FROZEN_FOR_EVALUATION_WAVEFORM_PRODUCTION" or waveform_analysis_plan.get("evaluation_waveform_production_authorized") is not True or waveform_analysis_plan.get("evaluation_effects_read") is not False:
        raise VisualRunError("model14_analysis_plan_not_frozen_or_not_authorized")
    if (waveform_root_freeze.get("status") != "FROZEN"
            or waveform_root_freeze.get("analysis_plan_raw_sha256") != source_refs["waveform_source_analysis_plan"]["raw_sha256"]
            or waveform_root_freeze.get("analysis_plan_sha256") != analysis_payload
            or waveform_root_freeze.get("evaluation_waveform_production_authorized") is not True
            or waveform_root_freeze.get("gpu_model_execution_authorized") is not False
            or waveform_root_freeze.get("model_forward_performed") is not False
            or waveform_root_freeze.get("evaluation_effects_read") is not False
            or waveform_root_freeze.get("evaluation_outcomes_read") is not False):
        raise VisualRunError("model14_root_freeze_not_bound_to_authorized_plan")
    expected_hashes = {"landmarker_asset": LANDMARKER_SHA256, "feature_helper": VIDEO_FEATURES_SHA256, "visual_core_source": CORE_SHA256, "tone_adapter_source": TONE_SHA256}
    for role, expected in expected_hashes.items():
        if source_refs[role]["raw_sha256"] != expected:
            raise VisualRunError(f"reusable_source_pin_mismatch:{role}")
    plan: dict[str, Any] = {
        "schema": "world_shape_visual_tracking_plan_v1",
        "status": "METADATA_PLAN_READY_FOR_ROOT_REVIEW",
        "classification": "metadata-only plan; no video pixels, audio waveforms, model effects, mouth geometry, or scores read",
        "design": {"visual_design_raw_sha256": DESIGN_SHA256, "evaluation_design_raw_sha256": EVAL_DESIGN_SHA256, "execution_pin_correction_raw_sha256": EXECUTION_PIN_CORRECTION_SHA256},
        "cohort": {"samples": EXPECTED_SAMPLES, "speakers": EXPECTED_SPEAKERS, "clips_per_speaker": 2, "directions": list(DIRECTIONS), "contexts": EXPECTED_CONTEXTS, "cells": list(WORLD_CELLS), "videos": EXPECTED_VIDEOS, "audio_domains_share_each_video": True},
        "clock_policy": {"reference": "q00 complete decoded timestamps", "fps_hz": FPS, "no_crop_or_timewarp": True, "require_each_cell_frames_equal_official_recipient_video_clock": True, "matching_tolerance": "0.5 * median(diff(q00 timestamps)) + 1e-12 seconds", "velocity": "absolute aperture difference divided by q00 timestamp delta only for q00-adjacent frames both valid in all five cells"},
        "availability_policy": {"latency_s": LATENCY_S, "minimum_common_frames_per_phone": int(calibration.get("protocol", {}).get("phone_min_common_frames", 1)), "min_phone_tokens_per_clip_by_class": calibration.get("protocol", {}).get("min_phone_tokens_per_clip_by_class", {}), "classes": ["bilabial", "open_vowel"], "original_mfa_eligibility_is_authoritative": True, "physical_overlay_never_restores_original_pair_eligibility": True, "availability_reads_geometry_values": False},
        "inference_policy": {"metrics": ["mean_aperture", "mean_absolute_aperture_velocity_per_s", "open_vowel_minus_bilabial_equal_token_mean"], "contrasts": ["mean_q10_minus_q00", "trajectory_q01_minus_q00", "joint_q11_minus_q00", "grid_cycle_SELF_GRID_minus_q00", "mean_minus_trajectory_q10_minus_q01", "mean_beyond_grid_q10_minus_SELF_GRID", "trajectory_beyond_grid_q01_minus_SELF_GRID", "joint_beyond_grid_q11_minus_SELF_GRID", "factorial_interaction_q11_minus_q10_minus_q01_plus_q00"], "speaker_roster": sorted(speaker_samples), "bootstrap": {"rng": "NumPy Generator(PCG64)", "seed": SEED, "draws": DRAWS, "fixed_roster_common_index_matrix": True, "point_estimator": "equal speakers after averaging two clips", "phone_estimator": "equal eligible speakers; each draw divides eligible sampled counts by that draw's eligible count; no NaN multiplied by zero", "interval": "studentized two-sided 99% simultaneous max-|t| across all estimable endpoints", "se": "bootstrap sample SD, ddof=1", "zero_se": "record deterministic state; assert all draws equal point estimate"}},
        "runtime": {"producer_run_dir_runtime": runtime_run, "producer_run_dir_metadata_source": str(model_root), "spawn_workers": True, "gpu_required": False, "score_forward_performed": False, "max_worker_count": 4, "threads_per_worker": 1},
        "input_seals": source_refs,
        "model_run_identity": {"status": status["status"], "completed_contexts": status["completed_contexts"], "failed_contexts": status["failed_contexts"], "planned_rows": status["planned_rows"], "scored_rows": status["scored_rows"], "unique_video_count": status["unique_video_count"], "execution_plan_raw_sha256": source_refs["model_execution_plan"]["raw_sha256"], "runtime_flags_raw_sha256": source_refs["model_runtime_flags"]["raw_sha256"], "code_seal_raw_sha256": source_refs["model_code_seal"]["raw_sha256"], "results_bytes_only_raw_sha256": source_refs["model_results"]["raw_sha256"], "protocol_sha256": source_refs["model_protocol"]["raw_sha256"], "rows_sha256": source_refs["model_rows"]["raw_sha256"], "status_sha256": source_refs["model_status"]["raw_sha256"], "all_files_inventory_sha256": source_refs["model_all_files_inventory"]["raw_sha256"], "restore_receipt_status": restore_receipt["status"], "restore_verified_file_count": restore_receipt["verified_file_count"], "waveform_source_analysis_plan_raw_sha256": source_refs["waveform_source_analysis_plan"]["raw_sha256"], "waveform_source_analysis_plan_payload_sha256": analysis_payload, "waveform_source_root_freeze_raw_sha256": source_refs["waveform_source_root_freeze"]["raw_sha256"], "waveform_source_root_freeze_payload_sha256": freeze_payload, "face_input_from_execution_plan": {"path": model_face["primary_face"], "sha256": model_face["primary_face_sha256"], "shape": model_face["primary_face_shape"]}},
        "physical_waveform_output_identity": {"parent_physical_manifest_raw_sha256": source_refs["physical_manifest"]["raw_sha256"], "parent_physical_plan_raw_sha256": source_refs["physical_plan"]["raw_sha256"], "evaluation_manifest_runtime_path": input_report.get("path"), "evaluation_manifest_raw_sha256": source_refs["shape_evaluation_manifest"]["raw_sha256"], "evaluation_manifest_payload_sha256": shape_payload, "evaluation_output_inventory_runtime_path": qa_inventory.get("path"), "evaluation_output_inventory_raw_sha256": source_refs["shape_output_inventory"]["raw_sha256"], "evaluation_output_inventory_payload_sha256": inventory_payload, "independent_qa_raw_sha256": source_refs["physical_independent_qa"]["raw_sha256"]},
        "context_rows": row_records,
    }
    plan = seal_object(plan, "plan_sha256")
    write_json(args.output, plan)
    rows_path = args.output.with_name("visual_tracking_rows.jsonl")
    if rows_path.exists():
        raise FileExistsError(rows_path)
    with rows_path.open("x", encoding="utf-8", newline="\n") as f:
        for row in row_records:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return {"status": plan["status"], "plan_path": str(args.output.resolve()), "plan_raw_sha256": sha256_file(args.output), "plan_payload_sha256": plan["plan_sha256"], "rows_path": str(rows_path.resolve()), "rows_raw_sha256": sha256_file(rows_path), "contexts": len(row_records), "videos": sum(len(r["cells"]) for r in row_records), "speakers": len(speaker_samples), "metadata_only": True}


def _verify_plan(path: Path) -> dict[str, Any]:
    plan = read_json(path)
    verify_object_fingerprint(plan, "plan_sha256")
    if plan.get("schema") != "world_shape_visual_tracking_plan_v1" or plan.get("status") != "METADATA_PLAN_READY_FOR_ROOT_REVIEW":
        raise VisualRunError("tracking_plan_not_ready")
    pins = {
        "tracking_extractor_source": sha256_file(Path(__file__).resolve()),
        "visual_design": DESIGN_SHA256,
        "evaluation_design": EVAL_DESIGN_SHA256,
        "execution_pin_correction": EXECUTION_PIN_CORRECTION_SHA256,
        "feature_helper": VIDEO_FEATURES_SHA256,
        "landmarker_asset": LANDMARKER_SHA256,
        "visual_core_source": CORE_SHA256,
        "tone_adapter_source": TONE_SHA256,
    }
    if any(plan.get("input_seals", {}).get(key, {}).get("raw_sha256") != value for key, value in pins.items()):
        raise VisualRunError("tracking_plan_source_pin_mismatch")
    if len(plan.get("context_rows", [])) != EXPECTED_CONTEXTS:
        raise VisualRunError("plan_context_count_mismatch")
    return plan


def _require_release(path: Path, plan: dict[str, Any], kind: str) -> dict[str, Any]:
    release = read_json(path)
    verify_object_fingerprint(release, "manifest_sha256")
    if release.get("status") != "ROOT_RELEASED" or release.get("scope") != kind or release.get("tracking_plan_sha256") != plan.get("plan_sha256"):
        raise VisualRunError(f"root_release_not_valid_for:{kind}")
    if release.get("geometry_read_authorized") is not (kind == "GEOMETRY_ANALYSIS"):
        raise VisualRunError("root_release_geometry_flag_mismatch")
    return release


def _safe_run_file(run_root: Path, relative: str) -> Path:
    path = (run_root / relative).resolve(strict=True)
    root = run_root.resolve(strict=True)
    if root not in path.parents:
        raise VisualRunError("video_path_escapes_model_run")
    return path


def _verify_model_run_seals(plan: dict[str, Any], run_root: Path, model_execution_plan_path: Path, model_runtime_flags_path: Path) -> dict[str, Any]:
    seals = plan["input_seals"]
    run_files = {
        "model_protocol": "evaluation_protocol.json",
        "model_rows": "evaluation_rows.jsonl",
        "model_status": "run_status.json",
        "model_all_files_inventory": "all_files_sha256.json",
        "model_code_seal": "code_seal.json",
        "model_results": "evaluation_results.jsonl",
    }
    for role, relative in run_files.items():
        path = _safe_run_file(run_root, relative)
        if sha256_file(path) != seals[role]["raw_sha256"]:
            raise VisualRunError(f"model_run_input_seal_changed:{role}")
    if (sha256_file(model_execution_plan_path) != seals["model_execution_plan"]["raw_sha256"]
            or sha256_file(model_runtime_flags_path) != seals["model_runtime_flags"]["raw_sha256"]):
        raise VisualRunError("model_execution_plan_or_runtime_flags_pin_mismatch")
    model_plan = read_json(model_execution_plan_path)
    expected_face = plan["model_run_identity"]["face_input_from_execution_plan"]
    face = model_plan.get("execution_assets", {})
    if (len(model_plan.get("contexts", [])) != EXPECTED_CONTEXTS
            or len(model_plan.get("rows", [])) != 1600
            or model_plan.get("row_manifest_sha256") != seals["model_rows"]["raw_sha256"]
            or face.get("primary_face") != expected_face["path"]
            or face.get("primary_face_sha256") != expected_face["sha256"]
            or face.get("primary_face_shape") != expected_face["shape"]):
        raise VisualRunError("model_execution_plan_remote_metadata_mismatch")
    flags = read_json(model_runtime_flags_path)
    if (flags.get("runtime_forward_gate") != "pass"
            or flags.get("runner_exit_code") != 0
            or flags.get("flag_mismatch_count") != 0
            or flags.get("runner_sha256") != model_plan.get("execution_assets", {}).get("runner_sha256")):
        raise VisualRunError("model_runtime_flags_remote_metadata_mismatch")
    return {"status": "MODEL_INPUT_SEALS_PASS", "model_files_hashed": len(run_files), "evaluation_results_parsed": False, "execution_plan_sha256": sha256_file(model_execution_plan_path), "runtime_flags_sha256": sha256_file(model_runtime_flags_path), "primary_face_sha256": face["primary_face_sha256"]}


def check_resources(output: Path, workers: int) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import shutil
        usage = shutil.disk_usage(output.parent)
        free = int(usage.free)
        available_mem = None
        cgroup_headroom = None
    else:
        import shutil
        free = int(shutil.disk_usage(output.parent).free)
        available_mem = None
        try:
            mem = Path("/proc/meminfo").read_text()
            match = re.search(r"^MemAvailable:\s+(\d+)", mem, re.M)
            available_mem = int(match.group(1)) * 1024 if match else None
        except OSError:
            pass
        current_path, max_path = Path("/sys/fs/cgroup/memory.current"), Path("/sys/fs/cgroup/memory.max")
        cgroup_headroom = None
        if current_path.exists() and max_path.exists():
            max_value = max_path.read_text().strip()
            if max_value != "max":
                cgroup_headroom = int(max_value) - int(current_path.read_text().strip())
    if free < 3 * (1024**3):
        raise VisualRunError("visual_output_disk_headroom_below_3GiB")
    if available_mem is not None and available_mem < 8 * (1024**3):
        raise VisualRunError("visual_host_memory_below_8GiB")
    if cgroup_headroom is not None and cgroup_headroom < 4 * (1024**3):
        raise VisualRunError("visual_cgroup_headroom_below_4GiB")
    return {"free_bytes": free, "available_memory_bytes": available_mem, "cgroup_headroom_bytes": cgroup_headroom, "workers": workers, "planned_video_count": EXPECTED_VIDEOS, "estimated_feature_output_bytes": 1_000_000_000}


def _video_metadata_check(path: Path, expected_frames: int) -> dict[str, Any]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VisualRunError(f"cannot_open_video:{path.name}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if not math.isfinite(fps) or not math.isclose(fps, FPS, rel_tol=0.0, abs_tol=1e-6):
        raise VisualRunError(f"video_fps_mismatch:{path.name}:{fps}")
    if n != expected_frames or n <= 0:
        raise VisualRunError(f"video_frame_count_mismatch:{path.name}:{n}:{expected_frames}")
    if width <= 0 or height <= 0:
        raise VisualRunError(f"video_dimensions_invalid:{path.name}")
    return {"fps_hz": fps, "declared_frames": n, "width": width, "height": height, "duration_s": n / fps}


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    # Keep the read-only preflight and MediaPipe smoke check CPU-only, matching
    # the extraction workers and the previously qualified visual runtime.
    _configure_cpu_runtime()
    _ensure_numpy()
    plan = _verify_plan(args.plan)
    run_root = args.run_root.resolve(strict=True)
    if args.run_root_runtime != plan["runtime"]["producer_run_dir_runtime"]:
        raise VisualRunError("runtime_run_root_differs_from_plan")
    model_run_seals = _verify_model_run_seals(plan, run_root, args.model_execution_plan.resolve(strict=True), args.model_runtime_flags.resolve(strict=True))
    output = args.output_dir.resolve(strict=False)
    if output.exists():
        raise VisualRunError("tracking_output_namespace_must_be_absent")
    if not 1 <= args.workers <= 4:
        raise VisualRunError("workers_outside_frozen_range")
    resources = check_resources(output, args.workers)
    helper = Path(args.video_features).resolve(strict=True)
    asset = Path(args.landmarker).resolve(strict=True)
    if sha256_file(helper) != VIDEO_FEATURES_SHA256 or sha256_file(asset) != LANDMARKER_SHA256:
        raise VisualRunError("feature_or_landmarker_pin_mismatch")
    if sha256_file(Path(args.visual_core)) != CORE_SHA256 or sha256_file(Path(args.tone_adapter)) != TONE_SHA256:
        raise VisualRunError("frozen_visual_rules_source_changed")
    import cv2
    import mediapipe as mp
    if np.__version__ != "1.26.4" or cv2.__version__ != "4.10.0" or mp.__version__ != "0.10.21":
        raise VisualRunError(f"pinned_visual_runtime_mismatch:numpy={np.__version__},opencv={cv2.__version__},mediapipe={mp.__version__}")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    refs = []
    frame_counts = Counter()
    for row in plan["context_rows"]:
        cell_results = []
        for cell in WORLD_CELLS:
            item = row["cells"][cell]
            video = _safe_run_file(run_root, item["video_relative_path"])
            if video.stat().st_size != item["video_bytes"] or sha256_file(video) != item["video_sha256"]:
                raise VisualRunError(f"video_hash_or_size_mismatch:{row['sample_id']}:{row['direction']}:{cell}")
            meta = _video_metadata_check(video, int(item["official_frame_count"]))
            frame_counts[meta["declared_frames"]] += 1
            cell_results.append({"cell": cell, "video_sha256": item["video_sha256"], **meta})
        if len({x["declared_frames"] for x in cell_results}) != 1 or len({x["fps_hz"] for x in cell_results}) != 1 or len({x["duration_s"] for x in cell_results}) != 1:
            raise VisualRunError(f"five_cell_video_clock_mismatch:{row['sample_id']}:{row['direction']}")
        refs.append({"sample_id": row["sample_id"], "speaker_id": row["speaker_id"], "direction": row["direction"], "cells": cell_results})
    if len(refs) != EXPECTED_CONTEXTS:
        raise VisualRunError("official_video_clock_cohort_mismatch")
    try:
        sys.path.insert(0, str(helper.parent))
        import video_features
        _mp, smoke = video_features.create_landmarker(asset)
        smoke.close()
    except Exception as exc:
        raise VisualRunError(f"landmarker_preflight_failed:{type(exc).__name__}") from exc
    return {"schema": "world_shape_visual_tracking_preflight_v1", "status": "PREFLIGHT_PASS", "tracking_plan_sha256": plan["plan_sha256"], "run_root_runtime": args.run_root_runtime, "model_run_seals": model_run_seals, "contexts": len(refs), "videos": EXPECTED_VIDEOS, "frame_count_distribution": {str(k): int(v) for k, v in sorted(frame_counts.items())}, "fps_hz": FPS, "feature_helper_sha256": sha256_file(helper), "landmarker_sha256": sha256_file(asset), "numpy_version": np.__version__, "opencv_version": cv2.__version__, "mediapipe_version": mp.__version__, "worker_method": "multiprocessing spawn", "resources": resources, "rows": refs, "video_pixels_or_mouth_geometry_read": False, "evaluation_results_parsed": False}


def _array_sha(value: np.ndarray) -> str:
    return bytes_sha256(np.ascontiguousarray(value).tobytes())


def _initialize_worker(helper_path: str, asset_path: str) -> None:
    global _FEATURES, _LANDMARKER
    _configure_cpu_runtime()
    import cv2
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    sys.path.insert(0, str(Path(helper_path).resolve().parent))
    import video_features
    _FEATURES = video_features
    _LANDMARKER = video_features.create_landmarker(Path(asset_path).resolve())
    atexit.register(_close_worker_landmarker)


def _close_worker_landmarker() -> None:
    global _LANDMARKER
    handle = _LANDMARKER
    _LANDMARKER = None
    if isinstance(handle, tuple) and len(handle) == 2:
        close = getattr(handle[1], "close", None)
        if callable(close):
            close()


def _extract_worker(task: dict[str, Any]) -> dict[str, Any]:
    video = Path(task["video_path"]).resolve(strict=True)
    if video.stat().st_size != task["video_bytes"] or sha256_file(video) != task["video_sha256"]:
        raise VisualRunError("video_changed_after_preflight")
    sequence = _FEATURES.extract_video_features(video, task["landmarker_path"], landmarker_handle=_LANDMARKER)
    if sequence.metadata.get("decoded_frame_count") != task["official_frame_count"] or sequence.metadata.get("declared_frame_count") != task["official_frame_count"] or not math.isclose(float(sequence.metadata.get("fps", 0.0)), FPS, rel_tol=0, abs_tol=1e-6):
        raise VisualRunError("decoded_video_clock_mismatch")
    identity = {k: task[k] for k in ("sample_id", "speaker_id", "split", "direction", "cell", "context_order", "ordered_eval_index")}
    metadata = dict(sequence.metadata)
    metadata.update({"protocol": "world_shape_visual_tracks_20261002_v1", **identity, "video_relative_path": task["video_relative_path"], "video_sha256": task["video_sha256"], "recipient_grid_sha256": task["recipient_grid_sha256"], "landmarker_sha256": LANDMARKER_SHA256, "video_features_sha256": VIDEO_FEATURES_SHA256, "visual_core_sha256": CORE_SHA256, "tone_adapter_sha256": TONE_SHA256, "full_recipient_clock": True, "fps_hz_required": FPS})
    # dataclasses.replace belongs to the helper's VisualSequence class module.
    from dataclasses import replace
    sequence = replace(sequence, metadata=metadata)
    feature_path = Path(task["feature_path"])
    availability_path = Path(task["availability_path"])
    if feature_path.exists() or availability_path.exists():
        raise FileExistsError("refusing_existing_track_output")
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    availability_path.parent.mkdir(parents=True, exist_ok=True)
    _FEATURES.save_visual_sequence(feature_path, sequence)
    np.savez_compressed(availability_path, timestamps_s=np.asarray(sequence.timestamps_s, dtype=np.float64), valid=np.asarray(sequence.valid, dtype=np.bool_))
    arrays = {name: getattr(sequence, name) for name in ("timestamps_s", "valid", "mouth_open", "mouth_width", "mouth_height", "mouth_area", "landmarks", "canonical_mouth")}
    return {**identity, "status": "complete", "video_relative_path": task["video_relative_path"], "video_sha256": task["video_sha256"], "video_bytes": video.stat().st_size, "official_frame_count": task["official_frame_count"], "fps_hz": float(sequence.metadata["fps"]), "feature_relative_path": feature_path.relative_to(Path(task["output_root"])).as_posix(), "feature_sha256": sha256_file(feature_path), "availability_relative_path": availability_path.relative_to(Path(task["output_root"])).as_posix(), "availability_sha256": sha256_file(availability_path), "array_inventory": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": _array_sha(value)} for name, value in arrays.items()}, "valid_frames": int(sequence.valid.sum()), "timestamp_sha256": _array_sha(np.asarray(sequence.timestamps_s, dtype=np.float64))}


def extract(args: argparse.Namespace) -> dict[str, Any]:
    _configure_cpu_runtime()
    _ensure_numpy()
    plan = _verify_plan(args.plan)
    release = _require_release(args.root_release, plan, "TRACKING_EXTRACTION")
    preflight = read_json(args.preflight_result.resolve(strict=True))
    if preflight.get("status") != "PREFLIGHT_PASS" or preflight.get("tracking_plan_sha256") != plan.get("plan_sha256") or preflight.get("videos") != EXPECTED_VIDEOS or preflight.get("contexts") != EXPECTED_CONTEXTS:
        raise VisualRunError("tracking_preflight_result_not_pass_or_bound")
    run_root = args.run_root.resolve(strict=True)
    if args.run_root_runtime != plan["runtime"]["producer_run_dir_runtime"]:
        raise VisualRunError("runtime_run_root_differs_from_plan")
    output = args.output_dir.resolve(strict=False)
    if output.exists():
        raise VisualRunError("tracking_output_namespace_must_be_absent")
    if not 1 <= args.workers <= 4:
        raise VisualRunError("workers_outside_frozen_range")
    helper, asset = Path(args.video_features).resolve(strict=True), Path(args.landmarker).resolve(strict=True)
    if sha256_file(helper) != VIDEO_FEATURES_SHA256 or sha256_file(asset) != LANDMARKER_SHA256:
        raise VisualRunError("feature_or_landmarker_pin_mismatch")
    import cv2
    import mediapipe as mp
    if np.__version__ != "1.26.4" or cv2.__version__ != "4.10.0" or mp.__version__ != "0.10.21":
        raise VisualRunError("pinned_visual_runtime_mismatch")
    resources = check_resources(output, args.workers)
    output.mkdir(parents=True, exist_ok=False)
    features_dir, availability_dir = output / "features", output / "availability"
    tasks = []
    for row in plan["context_rows"]:
        for cell in WORLD_CELLS:
            video_ref = row["cells"][cell]
            stem = f"{row['sample_id']}__{row['direction']}__{cell}"
            tasks.append({"sample_id": row["sample_id"], "speaker_id": row["speaker_id"], "split": row["split"], "direction": row["direction"], "cell": cell, "context_order": row["context_order"], "ordered_eval_index": row["ordered_eval_index"], "recipient_grid_sha256": row["recipient_grid_sha256"], "video_relative_path": video_ref["video_relative_path"], "video_sha256": video_ref["video_sha256"], "video_bytes": video_ref["video_bytes"], "official_frame_count": video_ref["official_frame_count"], "video_path": str(_safe_run_file(run_root, video_ref["video_relative_path"])), "landmarker_path": str(asset), "feature_path": str(features_dir / f"{stem}.npz"), "availability_path": str(availability_dir / f"{stem}.npz"), "output_root": str(output)})
    if len(tasks) != EXPECTED_VIDEOS:
        raise VisualRunError("track_task_count_mismatch")
    protocol = {"schema": "world_shape_visual_tracking_run_v1", "status": "TRACKING_STARTED", "tracking_plan_sha256": plan["plan_sha256"], "tracking_plan_raw_sha256": sha256_file(args.plan), "root_release_raw_sha256": sha256_file(args.root_release), "root_release_payload_sha256": release["manifest_sha256"], "extractor_source_sha256": sha256_file(Path(__file__).resolve()), "video_features_sha256": sha256_file(helper), "landmarker_sha256": sha256_file(asset), "runtime": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__, "mediapipe": mp.__version__, "start_method": "spawn", "workers": args.workers, "threads_per_worker": 1, "cuda_visible_devices": ""}, "resources": resources, "intended_videos": EXPECTED_VIDEOS, "effects_read": False, "model_forward_performed": False, "geometry_read_by_consumer": False}
    write_json(output / "visual_track_protocol.json", protocol)
    progress_path = output / "progress.jsonl"
    completed: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    with progress_path.open("x", encoding="utf-8", newline="\n") as journal:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"), initializer=_initialize_worker, initargs=(str(helper), str(asset))) as pool:
            future_map = {pool.submit(_extract_worker, task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_map):
                task = future_map[future]
                identity = {k: task[k] for k in ("sample_id", "speaker_id", "direction", "cell", "context_order")}
                try:
                    result = future.result()
                    completed.append(result)
                    journal_row = {"status": "complete", "sample_id": result["sample_id"], "direction": result["direction"], "cell": result["cell"], "feature_sha256": result["feature_sha256"], "availability_sha256": result["availability_sha256"]}
                except Exception as exc:
                    result = {**identity, "status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:500]}
                    failed.append(result)
                    journal_row = result
                journal.write(json.dumps(journal_row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                journal.flush(); os.fsync(journal.fileno())
                done = len(completed) + len(failed)
                if done % 20 == 0 or done == len(tasks):
                    print(json.dumps({"done": done, "complete": len(completed), "failed": len(failed), "planned": len(tasks)}, separators=(",", ":")), flush=True)
    records = sorted(completed + failed, key=lambda r: (int(r["context_order"]), WORLD_CELLS.index(r.get("cell", "q00"))))
    status = "COMPLETE" if len(completed) == EXPECTED_VIDEOS and not failed else "COMPLETE_WITH_FAILURES"
    manifest = {"schema": "world_shape_visual_track_manifest_v1", "status": status, "tracking_plan_sha256": plan["plan_sha256"], "tracking_plan_raw_sha256": sha256_file(args.plan), "tracking_protocol_sha256": sha256_file(output / "visual_track_protocol.json"), "progress_sha256": sha256_file(progress_path), "intended_videos": EXPECTED_VIDEOS, "completed_videos": len(completed), "failed_videos": len(failed), "effects_read": False, "model_forward_performed": False, "records": records, "failures": failed}
    manifest = seal_object(manifest, "manifest_sha256")
    write_json(output / "visual_track_manifest.json", manifest)
    files = {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name not in {"all_files_sha256.json", "visual_track_seal.json"}}
    write_json(output / "all_files_sha256.json", {"files": files})
    seal = {"schema": "world_shape_visual_track_seal_v1", "status": "SEALED" if status == "COMPLETE" else "SEALED_WITH_FAILURES", "manifest_sha256": sha256_file(output / "visual_track_manifest.json"), "manifest_payload_sha256": manifest["manifest_sha256"], "all_files_sha256": sha256_file(output / "all_files_sha256.json"), "files": len(files), "completed_videos": len(completed), "failed_videos": len(failed)}
    write_json(output / "visual_track_seal.json", seal)
    return {"status": status, "output_dir": str(output), "manifest_sha256": sha256_file(output / "visual_track_manifest.json"), "all_files_sha256": sha256_file(output / "all_files_sha256.json"), "completed_videos": len(completed), "failed_videos": len(failed)}


def normalize_tone_phone(label: Any, core: Any) -> str:
    value = str(label or "").strip().lower()
    value = "".join(ch for ch in value if ch not in TONE_MARKS)
    return core.normalize_phone(value)


def phone_class(label: Any, core: Any) -> str | None:
    normalized = normalize_tone_phone(label, core)
    if normalized in core.BILABIAL_BASES:
        return "bilabial"
    if normalized in core.OPEN_VOWEL_BASES:
        return "open_vowel"
    return None


def _load_pinned_modules(args: argparse.Namespace) -> tuple[Any, Any]:
    core_path, tone_path = Path(args.visual_core).resolve(strict=True), Path(args.tone_adapter).resolve(strict=True)
    if sha256_file(core_path) != CORE_SHA256 or sha256_file(tone_path) != TONE_SHA256:
        raise VisualRunError("visual_classification_sources_not_pinned")
    spec = importlib.util.spec_from_file_location("shape_visual_core_frozen", core_path)
    if spec is None or spec.loader is None:
        raise VisualRunError("cannot_import_frozen_visual_core")
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    try:
        spec.loader.exec_module(core)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return core, None


def _match_clock(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if reference.size > 1:
        tol = 0.5 * float(np.median(np.diff(reference))) + 1e-12
    else:
        tol = 0.5 / FPS
    out = np.full(reference.shape[0], -1, dtype=np.int32)
    i = j = 0
    while i < len(reference) and j < len(candidate):
        delta = float(candidate[j] - reference[i])
        if abs(delta) <= tol:
            out[i] = j; i += 1; j += 1
        elif delta < -tol:
            j += 1
        else:
            i += 1
    return out


def freeze_availability(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_numpy()
    plan = _verify_plan(args.plan)
    track_dir = args.track_dir.resolve(strict=True)
    track_manifest_path = track_dir / "visual_track_manifest.json"
    seal_path = track_dir / "visual_track_seal.json"
    inventory_path = track_dir / "all_files_sha256.json"
    track_manifest = read_json(track_manifest_path)
    track_seal = read_json(seal_path)
    if track_manifest.get("status") != "COMPLETE" or track_manifest.get("completed_videos") != EXPECTED_VIDEOS or track_manifest.get("failed_videos") != 0 or track_seal.get("status") != "SEALED":
        raise VisualRunError("track_run_not_complete_for_availability")
    verify_object_fingerprint(track_manifest, "manifest_sha256")
    inventory = _read_inventory(inventory_path)
    if track_seal.get("all_files_sha256") != sha256_file(inventory_path) or track_seal.get("manifest_sha256") != sha256_file(track_manifest_path):
        raise VisualRunError("track_seal_inventory_manifest_binding_mismatch")
    records = {(r["sample_id"], r["direction"], r["cell"]): r for r in track_manifest["records"] if r.get("status") == "complete"}
    if len(records) != EXPECTED_VIDEOS:
        raise VisualRunError("track_records_not_800_unique")
    core, _ = _load_pinned_modules(args)
    calibration = read_json(args.calibration)
    calibration_freeze = read_json(args.calibration_freeze)
    if sha256_file(args.calibration) != plan["input_seals"]["visual_calibration"]["raw_sha256"] or sha256_file(args.calibration_freeze) != plan["input_seals"]["visual_calibration_freeze"]["raw_sha256"]:
        raise VisualRunError("calibration_inputs_changed_after_plan")
    minimums = {str(k): int(v) for k, v in calibration.get("protocol", {}).get("min_phone_tokens_per_clip_by_class", {}).items()}
    frame_minimum = int(calibration.get("protocol", {}).get("phone_min_common_frames", 1))
    if set(minimums) != {"bilabial", "open_vowel"} or min(minimums.values()) < 1 or frame_minimum < 1 or not math.isclose(float(calibration.get("protocol", {}).get("latency", {}).get("latency_s")), LATENCY_S, abs_tol=1e-12):
        raise VisualRunError("calibration_support_rule_inconsistent")
    output = args.output_dir.resolve(strict=False)
    if output.exists():
        raise VisualRunError("availability_output_must_be_absent")
    output.mkdir(parents=True, exist_ok=False)
    context_rows = []
    sample_to_speaker: dict[str, str] = {}
    by_speaker_direction: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    ref_by_context = {(r["sample_id"], r["direction"]): r for r in plan["context_rows"]}
    for context_key, plan_row in ref_by_context.items():
        sample, direction = context_key
        cell_times: dict[str, np.ndarray] = {}
        cell_valid: dict[str, np.ndarray] = {}
        for cell in WORLD_CELLS:
            record = records.get((sample, direction, cell))
            if record is None:
                raise VisualRunError(f"track_record_missing:{sample}:{direction}:{cell}")
            if record["video_sha256"] != plan_row["cells"][cell]["video_sha256"] or record["video_relative_path"] != plan_row["cells"][cell]["video_relative_path"]:
                raise VisualRunError("track_record_does_not_match_plan")
            av_path = track_dir / record["availability_relative_path"]
            if sha256_file(av_path) != record["availability_sha256"] or inventory.get(record["availability_relative_path"]) != record["availability_sha256"]:
                raise VisualRunError("availability_array_file_hash_mismatch")
            # This archive deliberately contains only timestamps and validity; no mouth geometry.
            with np.load(av_path, allow_pickle=False) as archive:
                if set(archive.files) != {"timestamps_s", "valid"}:
                    raise VisualRunError("availability_npz_contains_unapproved_arrays")
                times = np.asarray(archive["timestamps_s"])
                valid = np.asarray(archive["valid"])
            if times.dtype != np.float64 or valid.dtype != np.bool_ or times.ndim != 1 or valid.shape != times.shape or len(times) != plan_row["cells"][cell]["official_frame_count"] or len(times) != record["official_frame_count"]:
                raise VisualRunError("availability_clock_shape_dtype_or_length_mismatch")
            if len(times) > 1 and (not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
                raise VisualRunError("availability_clock_not_finite_strictly_increasing")
            if not np.allclose(np.diff(times), 1.0 / FPS, rtol=0, atol=1e-9):
                raise VisualRunError("availability_clock_not_official_25fps")
            cell_times[cell], cell_valid[cell] = times, valid
        q00 = cell_times["q00"]
        maps = {cell: _match_clock(q00, cell_times[cell]) for cell in WORLD_CELLS}
        common = np.ones(len(q00), dtype=np.bool_)
        matched_valid = {}
        for cell in WORLD_CELLS:
            m = maps[cell]
            supported = np.zeros(len(q00), dtype=np.bool_)
            found = m >= 0
            supported[found] = cell_valid[cell][m[found]]
            matched_valid[cell] = supported
            common &= supported
        adjacent = np.zeros(max(0, len(q00) - 1), dtype=np.bool_)
        if len(q00) > 1:
            adjacent = common[:-1] & common[1:]
        phones = plan_row["recipient_phones"]
        token_info = []
        for phone in phones:
            cls = phone_class(phone["label"], core)
            if cls is None:
                continue
            shifted_start = float(phone["start_s"]) + LATENCY_S
            shifted_end = float(phone["end_s"]) + LATENCY_S
            token_mask = common & (q00 >= shifted_start) & (q00 < shifted_end)
            n = int(token_mask.sum())
            token_info.append({"phone_index": int(phone["phone_index"]), "raw_label": phone["label"], "normalized_label": phone["normalized_label"], "visual_class": cls, "start_s": phone["start_s"], "end_s": phone["end_s"], "shifted_start_s": shifted_start, "shifted_end_s": shifted_end, "common_frames": n, "supported": n >= frame_minimum, "q00_indices": np.flatnonzero(token_mask).astype(np.int32).tolist()})
        supported_counts = {cls: sum(1 for x in token_info if x["visual_class"] == cls and x["supported"]) for cls in ("bilabial", "open_vowel")}
        expected_counts = {cls: sum(1 for x in token_info if x["visual_class"] == cls) for cls in ("bilabial", "open_vowel")}
        original_qualified = bool(plan_row["original_mfa_pair_qualified"])
        endpoint_eligible = original_qualified and all(supported_counts[k] >= minimums[k] for k in minimums)
        # Persist clock/maps/support only. This file has no geometry values or references to geometry arrays.
        ctx_npz = output / "contexts" / f"{sample}__{direction}.npz"
        ctx_npz.parent.mkdir(parents=True, exist_ok=True)
        save_arrays = {"q00_timestamps_s": q00.astype(np.float64), "common_valid": common.astype(np.bool_), "common_adjacent": adjacent.astype(np.bool_)}
        for cell in WORLD_CELLS:
            save_arrays[f"index_map_{cell}"] = maps[cell].astype(np.int32)
            save_arrays[f"valid_on_q00_{cell}"] = matched_valid[cell].astype(np.bool_)
        np.savez_compressed(ctx_npz, **save_arrays)
        ctx_npz_sha = sha256_file(ctx_npz)
        row = {"context_order": plan_row["context_order"], "sample_id": sample, "speaker_id": plan_row["speaker_id"], "direction": direction, "recipient_condition": plan_row["recipient_condition"], "original_mfa_pair_qualified": original_qualified, "grid_overlay_used": plan_row["overlay_used"], "recipient_grid_sha256": plan_row["recipient_grid_sha256"], "recipient_phones": phones, "tokens": token_info, "expected_token_counts": expected_counts, "supported_token_counts": supported_counts, "minimum_token_counts": minimums, "minimum_common_frames_per_token": frame_minimum, "availability_endpoint_eligible": endpoint_eligible, "common_frame_count": int(common.sum()), "common_adjacent_pair_count": int(adjacent.sum()), "q00_frame_count": int(len(q00)), "q00_clock_sha256": _array_sha(q00), "common_valid_sha256": _array_sha(common), "common_adjacent_sha256": _array_sha(adjacent), "cell_index_map_sha256": {cell: _array_sha(maps[cell]) for cell in WORLD_CELLS}, "availability_context_path": ctx_npz.relative_to(output).as_posix(), "availability_context_sha256": ctx_npz_sha, "track_refs": {cell: {"feature_relative_path": records[(sample, direction, cell)]["feature_relative_path"], "feature_sha256": records[(sample, direction, cell)]["feature_sha256"], "availability_relative_path": records[(sample, direction, cell)]["availability_relative_path"], "availability_sha256": records[(sample, direction, cell)]["availability_sha256"], "array_inventory": records[(sample, direction, cell)]["array_inventory"]} for cell in WORLD_CELLS}}
        context_rows.append(row)
        sample_to_speaker[sample] = row["speaker_id"]
        by_speaker_direction[(row["speaker_id"], direction)].append(row)
    speaker_rows = []
    for speaker in sorted(set(sample_to_speaker.values())):
        for direction in DIRECTIONS:
            clips = sorted(by_speaker_direction[(speaker, direction)], key=lambda r: r["sample_id"])
            eligible = len(clips) == 2 and all(r["availability_endpoint_eligible"] for r in clips)
            speaker_rows.append({"speaker_id": speaker, "direction": direction, "clip_ids": [r["sample_id"] for r in clips], "both_clips_original_mfa_qualified": bool(len(clips) == 2 and all(r["original_mfa_pair_qualified"] for r in clips)), "complete_two_clip_phone_eligible": eligible})
    eligible_by_direction = {direction: sum(r["complete_two_clip_phone_eligible"] for r in speaker_rows if r["direction"] == direction) for direction in DIRECTIONS}
    output_manifest = {"schema": "world_shape_visual_availability_freeze_v1", "status": "FROZEN_BEFORE_GEOMETRY_READ", "scope": "timestamps, valid masks, recipient phone labels/intervals and identities only; no mouth geometry, scores, or outcomes parsed", "tracking_plan_raw_sha256": sha256_file(args.plan), "tracking_plan_payload_sha256": plan["plan_sha256"], "track_manifest_raw_sha256": sha256_file(track_manifest_path), "track_manifest_payload_sha256": track_manifest["manifest_sha256"], "track_seal_raw_sha256": sha256_file(seal_path), "track_files_inventory_raw_sha256": sha256_file(inventory_path), "visual_design_sha256": DESIGN_SHA256, "tone_adapter_sha256": TONE_SHA256, "visual_core_sha256": CORE_SHA256, "calibration_raw_sha256": sha256_file(args.calibration), "calibration_freeze_raw_sha256": sha256_file(args.calibration_freeze), "latency_s": LATENCY_S, "common_clock_policy": "q00 full timestamp vector; monotonically match all five cells with 0.5*median(q00 dt)+1e-12; common valid intersection; no crop or warp", "velocity_support_policy": "adjacent pairs only when both consecutive q00 frames are common-valid", "intended_denominator": {"samples": 80, "speakers": 40, "clips_per_speaker": 2, "directions": 2, "contexts": 160, "videos": 800}, "coverage": {"complete_contexts": len(context_rows), "complete_speakers": len({r['speaker_id'] for r in speaker_rows}), "complete_video_tracks": len(records), "phone_eligible_speakers_by_direction": eligible_by_direction, "original_mfa_unqualified_samples": sum(not bool(r["original_mfa_pair_qualified"]) for r in context_rows), "geometry_values_read": False, "evaluation_effects_read": False}, "speaker_rows": speaker_rows, "context_rows": context_rows}
    output_manifest = seal_object(output_manifest, "manifest_sha256")
    write_json(output / "availability_freeze.json", output_manifest)
    file_inventory = {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "all_files_sha256.json"}
    write_json(output / "all_files_sha256.json", {"files": file_inventory})
    seal = {"schema": "world_shape_visual_availability_seal_v1", "status": "FROZEN_BEFORE_GEOMETRY_READ", "manifest_raw_sha256": sha256_file(output / "availability_freeze.json"), "manifest_payload_sha256": output_manifest["manifest_sha256"], "all_files_sha256": sha256_file(output / "all_files_sha256.json"), "context_files": len(context_rows), "geometry_values_read": False}
    write_json(output / "availability_seal.json", seal)
    return {"status": seal["status"], "freeze_path": str((output / "availability_freeze.json").resolve()), "freeze_raw_sha256": seal["manifest_raw_sha256"], "freeze_payload_sha256": seal["manifest_payload_sha256"], "contexts": len(context_rows), "videos": len(records), "phone_eligible_speakers_by_direction": eligible_by_direction, "geometry_values_read": False}


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--visual-core", type=Path, required=True)
    parser.add_argument("--tone-adapter", type=Path, required=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    plan = subs.add_parser("build-plan")
    for name in ("design", "evaluation-design", "execution-pin-correction", "model-analysis-plan", "model-root-freeze", "model-execution-plan", "model-runtime-flags", "model-run", "model-restore-receipt", "physical-manifest", "physical-plan", "physical-qa", "shape-manifest", "shape-output-inventory", "eligibility", "grid-inventory", "calibration", "calibration-freeze", "phone-mapping", "landmarker", "video-features", "visual-core", "tone-adapter", "output"):
        plan.add_argument(f"--{name}", type=Path, required=True)
    plan.add_argument("--model-runtime-run-dir", required=False)
    pf = subs.add_parser("preflight")
    for name in ("plan", "run-root", "model-execution-plan", "model-runtime-flags", "video-features", "landmarker", "visual-core", "tone-adapter", "output-dir"):
        pf.add_argument(f"--{name}", type=Path, required=True)
    pf.add_argument("--run-root-runtime", required=True)
    pf.add_argument("--workers", type=int, default=4)
    ex = subs.add_parser("extract")
    for name in ("plan", "root-release", "preflight-result", "run-root", "video-features", "landmarker", "output-dir"):
        ex.add_argument(f"--{name}", type=Path, required=True)
    ex.add_argument("--run-root-runtime", required=True)
    ex.add_argument("--workers", type=int, default=4)
    av = subs.add_parser("freeze-availability")
    for name in ("plan", "track-dir", "calibration", "calibration-freeze", "visual-core", "tone-adapter", "output-dir"):
        av.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build-plan":
            result = build_plan(args)
        elif args.command == "preflight":
            result = preflight(args)
        elif args.command == "extract":
            result = extract(args)
        else:
            result = freeze_availability(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)[:1000]}, ensure_ascii=False, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
