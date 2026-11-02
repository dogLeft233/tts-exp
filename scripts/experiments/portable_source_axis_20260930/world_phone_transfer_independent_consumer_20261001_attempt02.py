#!/usr/bin/env python3
"""Independent CPU audit and cohort consumer for the frozen WORLD score run.

This reader imports no producer/evaluator project modules. It checks the sealed
row grid, artifact inventory, embeddings, full raw/unit cross matrices, lag
gathers, and alias map before producing prespecified cell decompositions,
speaker-block bootstrap intervals, and a fixed audio-row permutation null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_PLAN_SHA256 = "bba8bf428a1e5bc0a7e4eafa203976ea2c988d4332d2f82a02d2015e90f81d6d"
EXPECTED_PLAN_FINGERPRINT_SHA256 = "1410aface9e20f23421b42ec871f00344c424d56ea4a0d87c921b670bd4b11bc"
EXPECTED_QA04_SHA256 = "e78e11587ace920003163d04010670aa4d273f80f5b7df657d0b44eb143fd39d"
EXPECTED_RUNNER_SHA256 = "275a3169c7c7531a01d5942bd06278843faae70724f0e9ae5b7cf8c90ac7a6ac"
EXPECTED_CAL_SUMMARY_SHA256 = "34876249bb3463417f903ea49e11da03306b1c0785a28cb6a4f97da92d59f22a"
EXPECTED_CAL_AXES_SHA256 = "e71b669a67f435d951742cf67a9683e0385b77f5b97ab7a7408851389ade9244"
EXPECTED_FACE_REGISTRY_SHA256 = "8ada80c9c7a3628edef7428e9f218b475839a8a4f03079dcd9794e45ad0c4bbb"
EXPECTED_TTS_IDENTITY_SHA256 = "deb85e134904879d662ffce58990c04992420a25f0796184e907729f99af1f27"
EXPECTED_CONTEXTS = 160
EXPECTED_ROWS = 12_160
EXPECTED_MATRICES = 5_760
EXPECTED_VISUALS = 1_600
EXPECTED_PER_CONTEXT_ROWS = 76
ARMS = ("RT", "EX", "SP", "BOTH", "F0", "AP", "EX_EM", "SP_EM", "E")
FAMILIES = ("raw", "within_world")
CELLS = ("q00", "q10", "q01", "q11")
SUPPORTS = ("full", "valid", "guard20", "guard15")
METRICS = ("C_best", "C_fixed_anchor", "B_median", "D_min", "D0", "search_gain", "optimal_offset_frames")
DOMAIN_ARRAYS = {
    "raw": ("cross_distance_raw_float32", "cross_distance_to_zero_raw_float32", "lag_distance_raw_float32"),
    "unit": ("cross_distance_unit_float32", "cross_distance_to_zero_unit_float32", "lag_distance_unit_float32"),
}
EXPECTED_FLAGS = {"cudnn_benchmark": False, "cudnn_deterministic": True, "deterministic_algorithms": True}
EXPECTED_RUNTIME_FLAGS = {
    "deterministic_algorithms": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cudnn_allow_tf32": False,
    "matmul_allow_tf32": False,
}
EXPECTED_RUNTIME_WRAPPER_SHA256 = "001b5eefa0233ab70dd4187e1d65365c004e35aba23290faee4ccc3f28271456"
EXPECTED_SHA256_CONSTANTS = {
    name: value
    for name, value in globals().copy().items()
    if name.startswith("EXPECTED_") and name.endswith("SHA256") and isinstance(value, str)
}
VSHIFT = 15
ANCHOR_COLUMN = 18
BOOTSTRAP_SEED = 20260930
BOOTSTRAP_DRAWS = 20_000
NULL_DRAWS = 100
NULL_SEED_BASE = 20260930
NULL_METRICS = ("C_best", "C_fixed_anchor")
NULL_CONTRASTS = (
    "G_q10_minus_q00",
    "E_q01_minus_q00",
    "I_q11_minus_q10_minus_q01_plus_q00",
    "total_q11_minus_q00",
)
CROSS_ATOL = 1.0e-4
CROSS_RTOL = 1.0e-4
METRIC_ATOL = 2.0e-6


class AuditError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fail(errors: list[dict[str, Any]], code: str, **details: Any) -> None:
    if len(errors) < 500:
        errors.append({"code": code, **details})


def compare_hash(path: Path, expected: str, errors: list[dict[str, Any]], label: str) -> str | None:
    if not path.is_file():
        fail(errors, "missing_file", label=label, path=str(path))
        return None
    actual = file_sha256(path)
    if actual != expected:
        fail(errors, "file_hash_mismatch", label=label, path=str(path), expected=expected, actual=actual)
    return actual


def expected_supports(count: int) -> dict[str, np.ndarray]:
    full = np.arange(count, dtype=np.int32)
    g15_start = min(15, count)
    g20_start = min(20, count)
    return {
        "full": full,
        "valid": full.copy(),
        "guard20": np.arange(g20_start, max(g20_start, count - 20), dtype=np.int32),
        "guard15": np.arange(g15_start, max(g15_start, count - 15), dtype=np.int32),
    }


def cross_distance(lip: np.ndarray, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NumPy implementation of the frozen p=2 distance with PyTorch eps."""
    v = np.asarray(lip, dtype=np.float32)
    a = np.asarray(audio, dtype=np.float32)
    if v.ndim != 2 or a.ndim != 2 or not len(v) or not len(a) or v.shape[1] != a.shape[1]:
        raise AuditError(f"invalid embedding shapes: {v.shape}/{a.shape}")
    eps = np.float32(1.0e-6)
    delta = v[:, None, :] - a[None, :, :]
    delta += eps
    squared = np.sum(delta * delta, axis=2, dtype=np.float32)
    np.sqrt(squared, out=squared)
    zero_delta = v.copy()
    zero_delta += eps
    zero_squared = np.sum(zero_delta * zero_delta, axis=1, dtype=np.float32)
    np.sqrt(zero_squared, out=zero_squared)
    return squared.astype(np.float32, copy=False), zero_squared.astype(np.float32, copy=False)


def row_unit(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    norm = np.sqrt(np.sum(array * array, axis=1, keepdims=True, dtype=np.float32), dtype=np.float32)
    return (array / np.maximum(norm, np.float32(1.0e-12))).astype(np.float32, copy=False)


def lag_from_cross(cross: np.ndarray, zero: np.ndarray, query_count: int, audio_count: int) -> np.ndarray:
    matrix = np.asarray(cross, dtype=np.float32)
    zero_rows = np.asarray(zero, dtype=np.float32)
    if matrix.ndim != 2 or zero_rows.shape != (matrix.shape[0],):
        raise AuditError("cross/zero array shape mismatch")
    if query_count < 1 or query_count > matrix.shape[0] or audio_count < 1 or audio_count > matrix.shape[1]:
        raise AuditError("invalid common query or full audio domain")
    query = np.arange(query_count, dtype=np.int64)[:, None]
    offsets = np.arange(-VSHIFT, VSHIFT + 1, dtype=np.int64)[None, :]
    source = query + offsets
    in_domain = (source >= 0) & (source < audio_count)
    clipped = np.clip(source, 0, audio_count - 1)
    gathered = matrix[query, clipped]
    gathered = np.where(in_domain, gathered, zero_rows[:query_count, None])
    return np.asarray(gathered, dtype=np.float32)


def metric_curve(matrix: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float32)
    rows = np.asarray(indices, dtype=np.int64)
    if rows.size == 0:
        return {"status": "too_short", "support_n": 0, "window_indices": []}
    curve32 = np.mean(values[rows], axis=0, dtype=np.float32)
    curve = curve32.astype(np.float64)
    best = int(np.argmin(curve))
    median = float(np.median(curve))
    return {
        "status": "ok",
        "support_n": int(rows.size),
        "window_indices": rows.tolist(),
        "curve": curve.tolist(),
        "lag_offsets_frames": [int(VSHIFT - index) for index in range(values.shape[1])],
        "optimal_offset_frames": int(VSHIFT - best),
        "optimal_column": best,
        "D_min": float(curve[best]),
        "D0": float(curve[VSHIFT]),
        "B_median": median,
        "C_best": float(median - curve[best]),
        "fixed_anchor_column": ANCHOR_COLUMN,
        "fixed_anchor_offset_frames": int(VSHIFT - ANCHOR_COLUMN),
        "D_fixed_anchor": float(curve[ANCHOR_COLUMN]),
        "C_fixed_anchor": float(median - curve[ANCHOR_COLUMN]),
        "search_gain": float(curve[VSHIFT] - curve[best]),
    }


def compare_metric(actual: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(actual, dict) or actual.get("status") != expected["status"]:
        return False
    if actual.get("support_n") != expected["support_n"]:
        return False
    if expected["status"] == "too_short":
        return actual.get("window_indices") == []
    for key in ("window_indices", "lag_offsets_frames", "optimal_offset_frames", "optimal_column", "fixed_anchor_column", "fixed_anchor_offset_frames"):
        if actual.get(key) != expected.get(key):
            return False
    # D0 and search_gain are recomputed by this consumer from the sealed lag
    # matrix. The producer's legacy search_gain field is retained as evidence,
    # but is not an audit gate or an input to any consumer estimate.
    for key in ("curve", "D_min", "B_median", "C_best", "D_fixed_anchor", "C_fixed_anchor"):
        if key not in actual or expected.get(key) is None:
            return False
        lhs = np.asarray(actual.get(key), dtype=np.float64)
        rhs = np.asarray(expected.get(key), dtype=np.float64)
        if lhs.shape != rhs.shape or not np.allclose(lhs, rhs, atol=METRIC_ATOL, rtol=METRIC_ATOL, equal_nan=False):
            return False
    return True


def check_all_files(run: Path, errors: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, Any]]:
    inventory_path = run / "all_files_sha256.json"
    inventory = read_json(inventory_path)
    if "files" in inventory and isinstance(inventory["files"], dict):
        entries = inventory["files"]
    else:
        entries = inventory
    if not isinstance(entries, dict):
        raise AuditError("all_files_sha256.json is not a path-to-sha mapping")
    expected = {str(key).replace("\\", "/"): str(value) for key, value in entries.items()}
    actual_paths = {
        str(path.relative_to(run)).replace("\\", "/")
        for path in run.rglob("*") if path.is_file() and path.name != "all_files_sha256.json"
    }
    missing = sorted(set(expected) - actual_paths)
    unexpected = sorted(actual_paths - set(expected))
    if missing:
        fail(errors, "all_files_missing_paths", count=len(missing), examples=missing[:8])
    if unexpected:
        fail(errors, "all_files_unlisted_paths", count=len(unexpected), examples=unexpected[:8])
    mismatches = []
    for relative, expected_sha in expected.items():
        path = run / Path(relative)
        if not path.is_file():
            continue
        actual_sha = file_sha256(path)
        if actual_sha != expected_sha:
            mismatches.append({"path": relative, "expected": expected_sha, "actual": actual_sha})
            if len(mismatches) >= 100:
                break
    if mismatches:
        fail(errors, "all_files_hash_mismatches", count=len(mismatches), examples=mismatches[:8])
    return expected, {"inventory_count": len(expected), "present_file_count": len(actual_paths), "missing_path_count": len(missing), "unlisted_path_count": len(unexpected), "hash_mismatch_count": len(mismatches)}


def verify_input_wave_files(plan: dict[str, Any], errors: list[dict[str, Any]]) -> dict[str, Any]:
    records: dict[str, str] = {}
    for context in plan.get("contexts", []):
        for key in ("raw_audio", "fixed_audio"):
            item = context.get(key, {})
            if item.get("path") and item.get("sha256"):
                records[str(item["path"])] = str(item["sha256"])
        for arm, item in context.get("arms", {}).items():
            if item.get("path") and item.get("sha256"):
                records[str(item["path"])] = str(item["sha256"])
    mismatches = []
    for path_text, expected in records.items():
        path = Path(path_text)
        if not path.is_file():
            mismatches.append({"path": path_text, "failure": "missing"})
            continue
        actual = file_sha256(path)
        if actual != expected:
            mismatches.append({"path": path_text, "expected": expected, "actual": actual})
            if len(mismatches) >= 100:
                break
    if mismatches:
        fail(errors, "sealed_input_wave_hash_mismatch", count=len(mismatches), examples=mismatches[:8])
    return {"unique_wave_file_count": len(records), "hash_mismatch_count": len(mismatches)}


def verify_execution_assets(protocol: dict[str, Any], errors: list[dict[str, Any]]) -> dict[str, Any]:
    assets = protocol.get("input_seals", {}).get("execution_assets", {})
    checked_files = 0
    mismatches = []
    for label, record, path_key, sha_key in (
        ("python", assets.get("python", {}), "executable", "sha256"),
        ("wav2lip_checkpoint", assets.get("wav2lip", {}), "checkpoint", "checkpoint_sha256"),
        ("syncnet_checkpoint", assets.get("syncnet", {}), "checkpoint", "checkpoint_sha256"),
        ("ffmpeg", assets.get("ffmpeg", {}), "path", "sha256"),
        ("primary_face", assets.get("face", {}), "path", "sha256"),
    ):
        path_text, expected = record.get(path_key), record.get(sha_key)
        if not path_text or not expected:
            mismatches.append({"asset": label, "failure": "missing_binding"})
            continue
        path = Path(path_text)
        if not path.is_file():
            mismatches.append({"asset": label, "path": path_text, "failure": "missing_file"})
            continue
        actual = file_sha256(path)
        checked_files += 1
        if actual != expected:
            mismatches.append({"asset": label, "path": path_text, "expected": expected, "actual": actual})
    for label, record in (
        ("python_speech_features", assets.get("python_speech_features_source", {})),
        ("wav2lip_source", assets.get("wav2lip", {}).get("source_tree", {})),
        ("syncnet_source", assets.get("syncnet", {}).get("source_tree", {})),
    ):
        root_text = record.get("path")
        entries = record.get("files")
        if not root_text or not isinstance(entries, list):
            mismatches.append({"asset": label, "failure": "missing_source_tree_inventory"})
            continue
        root = Path(root_text)
        digest = hashlib.sha256()
        actual_relatives = set()
        tree_bad = False
        for entry in entries:
            relative = str(entry.get("relative_path", ""))
            candidate = (root / relative).resolve(strict=False)
            if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts or root.resolve(strict=False) not in candidate.parents:
                mismatches.append({"asset": label, "failure": "unsafe_source_relative_path", "relative_path": relative})
                tree_bad = True
                continue
            actual_relatives.add(relative)
            if not candidate.is_file() or candidate.stat().st_size != entry.get("size_bytes") or file_sha256(candidate) != entry.get("sha256"):
                mismatches.append({"asset": label, "failure": "source_file_hash_or_size_mismatch", "relative_path": relative})
                tree_bad = True
                continue
            digest.update(f"{relative}\0{entry['sha256']}\0{entry['size_bytes']}\n".encode("utf-8"))
        if len(entries) != record.get("file_count") or (not tree_bad and digest.hexdigest() != record.get("sha256")):
            mismatches.append({"asset": label, "failure": "source_tree_digest_mismatch", "expected_count": record.get("file_count"), "observed_count": len(entries), "expected_sha256": record.get("sha256"), "actual_sha256": None if tree_bad else digest.hexdigest()})
    if mismatches:
        fail(errors, "execution_asset_verification_failed", count=len(mismatches), examples=mismatches[:20])
    return {"checked_asset_files": checked_files, "mismatch_count": len(mismatches), "mismatches": mismatches[:20]}


def verify_runtime_flags(path: Path, expected_wrapper_path: Path, errors: list[dict[str, Any]]) -> dict[str, Any]:
    start_error_count = len(errors)
    if not path.is_file():
        fail(errors, "runtime_flag_sidecar_missing", path=str(path))
        return {"status": "missing", "path": str(path)}
    record = read_json(path)
    before = record.get("torch_flags_before_policy", {})
    after_policy = record.get("torch_flags_after_policy_before_forward", {})
    after_runner = record.get("torch_flags_after_runner", {})
    policy = record.get("policy", {})
    if record.get("schema_version") != 3:
        fail(errors, "runtime_flag_schema_mismatch", observed=record.get("schema_version"))
    if record.get("wrapper_sha256") != EXPECTED_RUNTIME_WRAPPER_SHA256:
        fail(errors, "runtime_wrapper_sha_mismatch", expected=EXPECTED_RUNTIME_WRAPPER_SHA256, actual=record.get("wrapper_sha256"))
    wrapper_path_text = record.get("wrapper_path")
    expected_wrapper_path = expected_wrapper_path.resolve(strict=False)
    if not wrapper_path_text or Path(wrapper_path_text).resolve(strict=False) != expected_wrapper_path:
        fail(errors, "runtime_wrapper_path_mismatch", expected=str(expected_wrapper_path), actual=wrapper_path_text)
    if not expected_wrapper_path.is_file() or file_sha256(expected_wrapper_path) != EXPECTED_RUNTIME_WRAPPER_SHA256:
        fail(errors, "runtime_wrapper_file_hash_mismatch", path=wrapper_path_text)
    if record.get("runner_sha256") != EXPECTED_RUNNER_SHA256:
        fail(errors, "runtime_runner_sha_mismatch", expected=EXPECTED_RUNNER_SHA256, actual=record.get("runner_sha256"))
    if record.get("runner_module") != "portable_source_axis_20260930.world_phone_transfer_gpu_evaluate_20261001":
        fail(errors, "runtime_runner_module_mismatch", observed=record.get("runner_module"))
    if record.get("cublas_workspace_config") != ":4096:8":
        fail(errors, "runtime_cublas_workspace_mismatch", observed=record.get("cublas_workspace_config"))
    for label, observed in (("policy", policy), ("after_policy", after_policy), ("after_runner", after_runner)):
        subset = {key: observed.get(key) for key in EXPECTED_RUNTIME_FLAGS} if isinstance(observed, dict) else None
        if subset != EXPECTED_RUNTIME_FLAGS:
            fail(errors, "runtime_five_flag_mismatch", stage=label, expected=EXPECTED_RUNTIME_FLAGS, actual=subset)
    forward_count = int(record.get("nn_module_forward_call_count", 0))
    counts_by_type = record.get("nn_module_call_counts_by_type", {})
    if forward_count <= 0 or sum(int(value) for value in counts_by_type.values()) != forward_count:
        fail(errors, "runtime_module_forward_count_mismatch", forward_count=forward_count, type_count_sum=sum(int(value) for value in counts_by_type.values()))
    if record.get("flag_mismatch_count") != 0 or record.get("runtime_forward_gate") != "pass" or record.get("runner_exit_code") != 0:
        fail(errors, "runtime_forward_flag_gate_failed", mismatch_count=record.get("flag_mismatch_count"), gate=record.get("runtime_forward_gate"), exit_code=record.get("runner_exit_code"))
    return {
        "status": "PASS" if len(errors) == start_error_count else "FAIL",
        "path": str(path),
        "sha256": file_sha256(path),
        "runner_sha256": record.get("runner_sha256"),
        "wrapper_sha256": record.get("wrapper_sha256"),
        "cublas_workspace_config": record.get("cublas_workspace_config"),
        "torch_flags_before_policy": before,
        "torch_flags_after_policy_before_forward": after_policy,
        "torch_flags_after_runner": after_runner,
        "nn_module_forward_call_count": forward_count,
        "flag_mismatch_count": record.get("flag_mismatch_count"),
        "runtime_forward_gate": record.get("runtime_forward_gate"),
    }


def verify_code_and_seals(run: Path, plan_path: Path, qa_path: Path, errors: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    malformed_constants = {
        name: value for name, value in EXPECTED_SHA256_CONSTANTS.items()
        if len(value) != 64 or re.fullmatch(r"[0-9a-f]{64}", value) is None
    }
    if malformed_constants:
        fail(errors, "consumer_expected_sha256_constant_malformed", constants=malformed_constants)
    plan_sha = file_sha256(plan_path)
    qa_sha = file_sha256(qa_path)
    if plan_sha != EXPECTED_PLAN_SHA256:
        fail(errors, "execution_plan_hash_mismatch", expected=EXPECTED_PLAN_SHA256, actual=plan_sha)
    if qa_sha != EXPECTED_QA04_SHA256:
        fail(errors, "qa04_hash_mismatch", expected=EXPECTED_QA04_SHA256, actual=qa_sha)
    plan = read_json(plan_path)
    protocol = read_json(run / "evaluation_protocol.json")
    rows = read_jsonl(run / "evaluation_rows.jsonl")
    status = read_json(run / "run_status.json")
    code_seal = read_json(run / "code_seal.json")
    qa = read_json(qa_path)
    if plan.get("status") != "preflight_pass" or plan.get("plan_sha256") != EXPECTED_PLAN_FINGERPRINT_SHA256:
        fail(errors, "execution_plan_status_or_embedded_sha_mismatch")
    plan_payload = {key: value for key, value in plan.items() if key != "plan_sha256"}
    computed_plan_fingerprint = hashlib.sha256(canonical_json(plan_payload)).hexdigest()
    if computed_plan_fingerprint != plan.get("plan_sha256") or computed_plan_fingerprint != EXPECTED_PLAN_FINGERPRINT_SHA256:
        fail(errors, "execution_plan_self_hash_mismatch")
    if protocol.get("execution_plan_sha256") != EXPECTED_PLAN_SHA256 or status.get("plan_sha256") != EXPECTED_PLAN_SHA256:
        fail(errors, "run_plan_binding_mismatch")
    if protocol.get("planned_context_count") != EXPECTED_CONTEXTS or protocol.get("planned_row_count") != EXPECTED_ROWS:
        fail(errors, "protocol_grid_count_mismatch")
    if protocol.get("planned_unique_matrix_count") != EXPECTED_MATRICES or protocol.get("planned_unique_video_count") != EXPECTED_VISUALS:
        fail(errors, "protocol_unique_count_mismatch")
    protocol_payload = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    protocol_fingerprint = hashlib.sha256(canonical_json(protocol_payload)).hexdigest()
    if protocol_fingerprint != protocol.get("protocol_sha256"):
        fail(errors, "protocol_self_hash_mismatch", expected=protocol.get("protocol_sha256"), actual=protocol_fingerprint)
    preflight_protocol = plan.get("protocol", {})
    preflight_protocol_sha = preflight_protocol.get("protocol_sha256")
    preflight_protocol_payload = {key: value for key, value in preflight_protocol.items() if key != "protocol_sha256"}
    computed_preflight_protocol_sha = hashlib.sha256(canonical_json(preflight_protocol_payload)).hexdigest()
    if plan.get("protocol_sha256") != preflight_protocol_sha or computed_preflight_protocol_sha != preflight_protocol_sha:
        fail(errors, "preflight_protocol_hash_binding_mismatch")
    run_only_fields = {
        "run_started_at_utc", "run_dir", "jpeg_temp_root",
        "runtime_parameters", "runtime_parameters_sha256",
        "row_manifest_sha256", "execution_plan_sha256", "protocol_sha256",
    }
    run_protocol_base = {key: value for key, value in protocol.items() if key not in run_only_fields}
    preflight_protocol_base = {key: value for key, value in preflight_protocol.items() if key != "protocol_sha256"}
    if canonical_json(run_protocol_base) != canonical_json(preflight_protocol_base):
        fail(errors, "run_protocol_preflight_payload_mismatch")
    if protocol.get("runtime_parameters") != protocol.get("planned_runtime_parameters"):
        fail(errors, "runtime_parameters_differ_from_preflight")
    raw_protocol_sha = file_sha256(run / "evaluation_protocol.json")
    row_sha = file_sha256(run / "evaluation_rows.jsonl")
    if hashlib.sha256(b"".join(canonical_json(row) + b"\n" for row in rows)).hexdigest() != row_sha:
        fail(errors, "row_manifest_raw_or_canonical_hash_mismatch")
    if row_sha != protocol.get("row_manifest_sha256") or row_sha != code_seal.get("row_manifest_sha256"):
        fail(errors, "row_manifest_seal_mismatch")
    if raw_protocol_sha != code_seal.get("protocol_sha256"):
        fail(errors, "protocol_raw_file_seal_mismatch")
    if code_seal.get("execution_plan_sha256") != EXPECTED_PLAN_SHA256:
        fail(errors, "code_seal_execution_plan_mismatch")
    if code_seal.get("code_files") != protocol.get("code_files"):
        fail(errors, "code_seal_file_map_mismatch")
    if len(rows) != EXPECTED_ROWS:
        fail(errors, "planned_row_count_mismatch", planned=len(rows), expected=EXPECTED_ROWS)
    if status.get("status") != "complete" or status.get("completed_contexts") != EXPECTED_CONTEXTS or status.get("failed_contexts") != 0:
        fail(errors, "run_not_complete", status=status.get("status"), completed_contexts=status.get("completed_contexts"), failed_contexts=status.get("failed_contexts"))
    if status.get("scored_rows") != EXPECTED_ROWS or status.get("failed_rows") != 0:
        fail(errors, "scored_row_coverage_mismatch", scored=status.get("scored_rows"), failed=status.get("failed_rows"))
    if status.get("unique_matrix_artifacts") != EXPECTED_MATRICES or status.get("unique_video_artifacts") != EXPECTED_VISUALS:
        fail(errors, "run_artifact_count_mismatch", matrices=status.get("unique_matrix_artifacts"), videos=status.get("unique_video_artifacts"))
    for key, value in EXPECTED_FLAGS.items():
        if protocol.get("model_flags", {}).get(key) is not value or status.get("forward_flags_required_and_asserted", {}).get(key) is not value:
            fail(errors, "forward_flag_binding_mismatch", key=key)
    actual_inputs = protocol.get("input_seals", {})
    expected_hashes = {
        "source_manifest": "6b0ce3252cf776a86cf84b19ab53afd936526216e4ddb30dc587a72a66409b4b",
        "tts_pair_manifest": "2aea4750a02125629d82f85b4842932465a202dd03eec96864d096d8856070d2",
        "physical_manifest": "27cb2cdf2246920c0667dd5bcc5d744253b3c6b7d407270c1254ffaf986aa3b3",
        "physical_plan": "f92dbcfc71ecb775d98c3cd33ec19b4dd99d206c27d155eba8ae9bc51204861e",
        "cal_qa_freeze": "0a5b694975efeb07b9d1e9c0beb10f0142bb4203ba4a026d7ff965477991ab9d",
        "spec": "f085e2915349523891a478e3fc4dd1dada8493d68d3d73608bcc042d4a4e62ca",
        "execution_bindings": "cde66fa11f140866fc87f07b7a39ca68ddefc403ff7839b239987549e9f7e47d",
        "alignment_overlay": "4822cf2dbaa1733c93447ef92446cc28cb5381f6c3c3d8d783cca5c477251b7e",
        "aliases_annex": "d14260e47869003756a0efd6e84719011c0a2550b3af07ead5f62b902fb22f3f",
    }
    for key, expected in expected_hashes.items():
        value = actual_inputs.get(key)
        observed = value.get("sha256") if isinstance(value, dict) else None
        if observed != expected:
            fail(errors, "input_seal_mismatch", key=key, expected=expected, actual=observed)
    tts_seal = actual_inputs.get("tts_pair_manifest", {})
    if tts_seal.get("input_identity_sha256") != EXPECTED_TTS_IDENTITY_SHA256:
        fail(errors, "tts_input_identity_seal_mismatch", expected=EXPECTED_TTS_IDENTITY_SHA256, actual=tts_seal.get("input_identity_sha256"))
    if actual_inputs.get("calibration_anchor") != {"k": 3, "column": 18, "official_offset": -3}:
        fail(errors, "calibration_anchor_seal_mismatch", actual=actual_inputs.get("calibration_anchor"))
    anchor = protocol.get("anchor", {})
    if (anchor.get("k"), anchor.get("column"), anchor.get("official_offset")) != (3, 18, -3):
        fail(errors, "protocol_fixed_anchor_mismatch", actual={key: anchor.get(key) for key in ("k", "column", "official_offset")})
    for seal_name, expected in (
        ("summary", EXPECTED_CAL_SUMMARY_SHA256),
        ("axes", EXPECTED_CAL_AXES_SHA256),
    ):
        record = actual_inputs.get("calibration", {}).get(seal_name, {})
        if record.get("sha256") != expected:
            fail(errors, "calibration_seal_mismatch", component=seal_name, expected=expected, actual=record.get("sha256"))
        path_text = record.get("path")
        if not path_text or not Path(path_text).is_file() or file_sha256(Path(path_text)) != expected:
            fail(errors, "calibration_file_hash_mismatch", component=seal_name, path=path_text)
    face_registry = actual_inputs.get("face_registry", {})
    if face_registry.get("sha256") != EXPECTED_FACE_REGISTRY_SHA256:
        fail(errors, "face_registry_seal_mismatch", expected=EXPECTED_FACE_REGISTRY_SHA256, actual=face_registry.get("sha256"))
    face_registry_path = face_registry.get("path")
    if not face_registry_path or not Path(face_registry_path).is_file() or file_sha256(Path(face_registry_path)) != EXPECTED_FACE_REGISTRY_SHA256:
        fail(errors, "face_registry_file_hash_mismatch", path=face_registry_path)
    face_geometry = face_registry.get("face", {})
    if face_geometry.get("generation_box_xyxy") != [168, 88, 381, 385] or face_geometry.get("score_box", {}).get("box") != [59, 16, 489, 446] or (face_geometry.get("width"), face_geometry.get("height")) != (512, 512):
        fail(errors, "face_geometry_binding_mismatch")
    if qa.get("status") != "PASS":
        fail(errors, "qa04_not_pass", status=qa.get("status"))
    qa_coverage = qa.get("coverage", {})
    for key, expected in {
        "sample_pairs": 80,
        "recipient_contexts": 160,
        "completed_recipient_contexts": 160,
        "unique_source_raw_wavs": 160,
        "expected_source_raw_wavs": 160,
        "unique_physical_wavs": 1920,
        "expected_physical_wavs": 1920,
        "unique_npz_artifacts": 1920,
        "expected_npz_artifacts": 1920,
        "original_mfa_qualified_pairs": 79,
        "physical_overlay_qualified_pairs": 80,
    }.items():
        if qa_coverage.get(key) != expected:
            fail(errors, "qa04_coverage_mismatch", field=key, expected=expected, actual=qa_coverage.get(key))
    qa_hashes = {key: value.get("sha256") for key, value in qa.get("inputs", {}).items() if isinstance(value, dict)}
    for key, expected in {
        "input_manifest": expected_hashes["source_manifest"],
        "tts_pair_manifest": expected_hashes["tts_pair_manifest"],
        "physical_manifest": expected_hashes["physical_manifest"],
        "physical_plan": expected_hashes["physical_plan"],
        "cal_freeze": expected_hashes["cal_qa_freeze"],
    }.items():
        if qa_hashes.get(key) != expected:
            fail(errors, "qa04_input_binding_mismatch", key=key, expected=expected, actual=qa_hashes.get(key))
    runner_ref = protocol.get("code_files", {}).get("world_phone_transfer_gpu_evaluate_20261001.py", {})
    if runner_ref.get("sha256") != EXPECTED_RUNNER_SHA256:
        fail(errors, "runner_source_seal_mismatch", expected=EXPECTED_RUNNER_SHA256, actual=runner_ref.get("sha256"))
    for name, record in protocol.get("code_files", {}).items():
        path_text = record.get("path")
        if not path_text:
            fail(errors, "code_file_path_missing", name=name)
            continue
        path = Path(path_text)
        if not path.is_file() or file_sha256(path) != record.get("sha256"):
            fail(errors, "sealed_code_file_changed", name=name, path=path_text)
    if status.get("forward_event_count", 0) <= 0:
        fail(errors, "forward_events_missing")
    return plan, protocol, status, {"qa04": {"sha256": qa_sha, "status": qa.get("status"), "coverage": qa.get("coverage")}, "raw_protocol_sha256": raw_protocol_sha, "row_manifest_sha256": row_sha, "code_seal": code_seal}


def expected_alias(family: str, arm: str, cell: str) -> tuple[str, str]:
    if family == "raw":
        table = {"q00": ("V_RT", "A_RAW"), "q10": (f"V_{arm}", "A_RAW"), "q01": ("V_RT", f"A_{arm}"), "q11": (f"V_{arm}", f"A_{arm}")}
    elif family == "within_world":
        table = {"q00": ("V_RT", "A_RT"), "q10": (f"V_{arm}", "A_RT"), "q01": ("V_RT", f"A_{arm}"), "q11": (f"V_{arm}", f"A_{arm}")}
    elif family == "bridge":
        table = {"q00": ("V_FIXED", "A_RAW"), "q10": ("V_RT", "A_RAW"), "q01": ("V_FIXED", "A_RT"), "q11": ("V_RT", "A_RT")}
    else:
        raise AuditError(f"unknown alias family: {family}")
    return table[cell]


def validate_support_arrays(npz: Any, common_count: int, errors: list[dict[str, Any]], context_key: str, matrix_id: str) -> dict[str, np.ndarray]:
    expected = expected_supports(common_count)
    supports: dict[str, np.ndarray] = {}
    for name in SUPPORTS:
        key = f"common_support_{name}_int32"
        actual = np.asarray(npz[key])
        if actual.dtype != np.int32 or not np.array_equal(actual, expected[name]):
            fail(errors, "matrix_support_mismatch", context=context_key, matrix=matrix_id, support=name)
        supports[name] = actual
    return supports


def array_shape_inventory(npz: Any) -> dict[str, dict[str, Any]]:
    return {key: {"shape": list(np.asarray(npz[key]).shape), "dtype": str(np.asarray(npz[key]).dtype)} for key in npz.files}


def seal_consumer_output(out: Path, consumer_sha: str, run: Path) -> dict[str, Any]:
    inventory_path = out / "consumer_all_files_sha256.json"
    files = {
        path.relative_to(out).as_posix(): file_sha256(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path not in {inventory_path, out / "consumer_output_seal.json"}
    }
    inventory_path.write_text(json.dumps({"files": files}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    seal = {
        "schema_version": 1,
        "status": "SEALED",
        "consumer_source_sha256": consumer_sha,
        "producer_run_root": str(run),
        "producer_plan_sha256": EXPECTED_PLAN_SHA256,
        "producer_plan_embedded_fingerprint_sha256": EXPECTED_PLAN_FINGERPRINT_SHA256,
        "producer_qa04_sha256": EXPECTED_QA04_SHA256,
        "inventory_path": inventory_path.name,
        "inventory_sha256": file_sha256(inventory_path),
        "inventory_count": len(files),
    }
    (out / "consumer_output_seal.json").write_text(json.dumps(seal, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return seal


def validate_metric_metadata(
    result: dict[str, Any],
    computed: dict[str, dict[str, dict[str, Any]]],
    errors: list[dict[str, Any]],
    legacy_search_gain: dict[str, dict[str, Any]],
) -> None:
    for domain in ("raw", "unit"):
        key = f"{domain}_support_metrics"
        actual = result.get(key)
        if not isinstance(actual, dict):
            fail(errors, "result_support_metrics_missing", row_id=result.get("row_id"), key=key)
            continue
        for support in SUPPORTS:
            actual_support = actual.get(support)
            expected_support = computed[domain][support]
            if expected_support.get("status") == "ok":
                key_id = f"{domain}/{support}"
                diagnostic = legacy_search_gain.setdefault(key_id, {
                    "rows_checked": 0, "rows_missing_legacy_value": 0,
                    "rows_differ_from_recomputed": 0, "max_abs_difference": 0.0,
                    "producer_D0_rows_checked": 0, "producer_D0_rows_missing": 0,
                    "producer_D0_rows_differ": 0, "producer_D0_max_abs_difference": 0.0,
                })
                stored = actual_support.get("search_gain") if isinstance(actual_support, dict) else None
                if isinstance(stored, (int, float)) and math.isfinite(float(stored)):
                    difference = float(stored) - float(expected_support["search_gain"])
                    diagnostic["rows_checked"] += 1
                    diagnostic["rows_differ_from_recomputed"] += int(abs(difference) > METRIC_ATOL)
                    diagnostic["max_abs_difference"] = max(diagnostic["max_abs_difference"], abs(difference))
                else:
                    diagnostic["rows_missing_legacy_value"] += 1
                producer_d0 = actual_support.get("D0") if isinstance(actual_support, dict) else None
                if isinstance(producer_d0, (int, float)) and math.isfinite(float(producer_d0)):
                    difference = float(producer_d0) - float(expected_support["D0"])
                    diagnostic["producer_D0_rows_checked"] += 1
                    diagnostic["producer_D0_rows_differ"] += int(abs(difference) > METRIC_ATOL)
                    diagnostic["producer_D0_max_abs_difference"] = max(diagnostic["producer_D0_max_abs_difference"], abs(difference))
                else:
                    diagnostic["producer_D0_rows_missing"] += 1
            if not compare_metric(actual_support, expected_support):
                fail(errors, "result_support_metrics_mismatch", row_id=result.get("row_id"), domain=domain, support=support)


def summarize_stats(
    result_rows: list[dict[str, Any]],
    metric_by_row: dict[str, dict[str, dict[str, dict[str, Any]]]],
    errors: list[dict[str, Any]],
    bootstrap_indices: np.ndarray,
) -> tuple[dict[str, Any], dict[tuple[Any, ...], dict[str, Any]]]:
    values: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    by_context: dict[tuple[str, str, str, str], dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)
    for row in result_rows:
        if row.get("status") != "scored" or row.get("row_id") not in metric_by_row:
            continue
        family = row["family"]
        arm = row["condition"]
        cell = row["cell"]
        sample, direction = row["sample_id"], row["direction"]
        by_context[(sample, direction, family, arm)][cell] = metric_by_row[row["row_id"]]
    for row in result_rows:
        if row.get("status") != "scored" or row.get("row_id") not in metric_by_row:
            continue
        family = row["family"]
        if family == "bridge":
            arm = "RT_bridge"
        else:
            arm = row["condition"]
        cells = by_context.get((row["sample_id"], row["direction"], family, arm), {})
        if any(cell not in cells for cell in CELLS):
            continue
        for domain in ("raw", "unit"):
            for support in SUPPORTS:
                for metric in METRICS:
                    q = {cell: cells[cell][domain][support].get(metric) for cell in CELLS}
                    if any(value is None for value in q.values()):
                        continue
                    contrasts = {
                        "q00": float(q["q00"]), "q10": float(q["q10"]), "q01": float(q["q01"]), "q11": float(q["q11"]),
                        "G_q10_minus_q00": float(q["q10"] - q["q00"]),
                        "E_q01_minus_q00": float(q["q01"] - q["q00"]),
                        "I_q11_minus_q10_minus_q01_plus_q00": float(q["q11"] - q["q10"] - q["q01"] + q["q00"]),
                        "total_q11_minus_q00": float(q["q11"] - q["q00"]),
                    }
                    base = (row["sample_id"], row["speaker_id"], row["direction"], family, arm, domain, support, metric)
                    if row["cell"] == "q00":
                        for contrast_name, value in contrasts.items():
                            values[base + (contrast_name,)].append({"value": value, "sample_id": row["sample_id"], "speaker_id": row["speaker_id"]})
    speaker_values: dict[tuple[Any, ...], dict[str, Any]] = {}
    grouped: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key, entries in values.items():
        # Each key is written once from q00; two paired clips are aggregated below.
        sample_id, speaker_id, direction, family, arm, domain, support, metric, contrast = key
        group_key = (direction, family, arm, domain, support, metric, contrast)
        grouped[group_key][speaker_id].append(entries[0]["value"])
    for group_key, per_speaker in grouped.items():
        complete = {speaker: vals for speaker, vals in per_speaker.items() if len(vals) == 2}
        speakers = sorted(complete)
        speaker_means = np.asarray([np.mean(complete[speaker], dtype=np.float64) for speaker in speakers], dtype=np.float64)
        speaker_values[group_key] = {
            "speakers": speakers,
            "speaker_means": speaker_means,
            "clip_counts": {speaker: len(complete[speaker]) for speaker in speakers},
            "observed_speakers_with_any_clip": len(per_speaker),
            "observed_clip_count": sum(len(vals) for vals in per_speaker.values()),
        }

    summaries = []
    for key, record in sorted(speaker_values.items(), key=lambda item: tuple(str(x) for x in item[0])):
        direction, family, arm, domain, support, metric, contrast = key
        speaker_means = record["speaker_means"]
        if len(speaker_means) != 40:
            summaries.append({
                "direction": direction, "family": family, "condition": arm,
                "domain": domain, "support": support, "metric": metric,
                "contrast": contrast, "status": "ineligible_incomplete_speakers",
                "speaker_count": len(speaker_means), "clip_count": record["observed_clip_count"],
                "speaker_count_with_any_clip": record["observed_speakers_with_any_clip"],
                "clip_counts_by_complete_speaker": record["clip_counts"],
            })
            continue
        boot = speaker_means[bootstrap_indices].mean(axis=1)
        ci = np.quantile(boot, [0.005, 0.995], method="linear")
        summaries.append({
            "direction": direction, "family": family, "condition": arm,
            "domain": domain, "support": support, "metric": metric,
            "contrast": contrast, "status": "ok", "clip_count": 80,
            "speaker_count": 40, "per_speaker_clip_count": 2,
            "estimate_speaker_equal_mean": float(speaker_means.mean(dtype=np.float64)),
            "ci99_percentile": [float(ci[0]), float(ci[1])],
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_draws": BOOTSTRAP_DRAWS,
        })

    gates = []
    for direction, sign in (("tts_to_natural", 1), ("natural_to_tts", -1)):
        for metric in ("C_best", "C_fixed_anchor"):
            key = (direction, "raw", "EX", "raw", "guard20", metric, "G_q10_minus_q00")
            record = speaker_values.get(key)
            if record is None or len(record["speaker_means"]) != 40:
                gates.append({"direction": direction, "metric": metric, "eligible": False, "gate_pass": False, "reason": "missing complete 40-speaker block"})
                continue
            vals = record["speaker_means"] * sign
            boot = vals[bootstrap_indices].mean(axis=1)
            ci = np.quantile(boot, [0.005, 0.995], method="linear")
            # Original-scale intervals are reported; direction is used only for pass logic.
            raw_ci = np.quantile(record["speaker_means"][bootstrap_indices].mean(axis=1), [0.005, 0.995], method="linear")
            passed = bool(ci[0] > 0)
            gates.append({"direction": direction, "metric": metric, "eligible": True, "estimate_speaker_equal_mean": float(record["speaker_means"].mean(dtype=np.float64)), "ci99_percentile": [float(raw_ci[0]), float(raw_ci[1])], "expected_sign": "positive" if sign > 0 else "negative", "gate_pass": passed})
    return {
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "draws": BOOTSTRAP_DRAWS,
            "method": "one shared 20,000 x 40 PCG64 speaker-index matrix reused for both directions and every endpoint; each speaker is the equal mean of two paired clips",
            "summaries": summaries,
            "primary_raw_EX_guard20_gates": gates,
        }
    }, speaker_values


def lag_from_permuted_cross(
    cross: np.ndarray,
    zero: np.ndarray,
    permutation: np.ndarray,
    query_count: int,
    audio_count: int,
) -> np.ndarray:
    """Gather lag distances after a full-domain row permutation of audio A."""
    matrix = np.asarray(cross, dtype=np.float32)
    zero_rows = np.asarray(zero, dtype=np.float32)
    perm = np.asarray(permutation, dtype=np.int64)
    if perm.shape != (audio_count,) or not np.array_equal(np.sort(perm), np.arange(audio_count)):
        raise AuditError("time-null permutation is not a full permutation of this audio domain")
    query = np.arange(query_count, dtype=np.int64)[:, None]
    offsets = np.arange(-VSHIFT, VSHIFT + 1, dtype=np.int64)[None, :]
    source = query + offsets
    in_domain = (source >= 0) & (source < audio_count)
    source_clipped = np.clip(source, 0, audio_count - 1)
    shuffled_audio_index = perm[source_clipped]
    gathered = matrix[query, shuffled_audio_index]
    return np.asarray(np.where(in_domain, gathered, zero_rows[:query_count, None]), dtype=np.float32)


def lag_from_permuted_cross_batch(
    cross: np.ndarray,
    zero: np.ndarray,
    permutations: np.ndarray,
    query_count: int,
    audio_count: int,
) -> np.ndarray:
    """Gather many full-domain audio permutations without repeated Python loops."""
    matrix = np.asarray(cross, dtype=np.float32)
    zero_rows = np.asarray(zero, dtype=np.float32)
    perms = np.asarray(permutations, dtype=np.int64)
    if matrix.ndim != 2 or zero_rows.shape != (matrix.shape[0],):
        raise AuditError("batched cross/zero array shape mismatch")
    if perms.ndim != 2 or perms.shape[1] != audio_count:
        raise AuditError("batched time-null permutations have the wrong shape")
    if matrix.shape[1] != audio_count or not np.all(np.sort(perms, axis=1) == np.arange(audio_count)[None, :]):
        raise AuditError("batched time-null input is not a full-domain permutation")
    if query_count < 1 or query_count > matrix.shape[0] or audio_count < 1:
        raise AuditError("invalid common query or full audio domain")
    query = np.arange(query_count, dtype=np.int64)[None, :, None]
    offsets = np.arange(-VSHIFT, VSHIFT + 1, dtype=np.int64)[None, None, :]
    source = query + offsets
    in_domain = (source >= 0) & (source < audio_count)
    clipped = np.clip(source, 0, audio_count - 1)
    shuffled_audio_index = perms[:, clipped[0]]
    query_index = np.broadcast_to(query, shuffled_audio_index.shape)
    gathered = matrix[query_index, shuffled_audio_index]
    zero_values = np.broadcast_to(zero_rows[:query_count][None, :, None], gathered.shape)
    return np.asarray(np.where(in_domain, gathered, zero_values), dtype=np.float32)


def metric_curves_batch(matrix: np.ndarray, indices: np.ndarray) -> dict[str, np.ndarray] | None:
    """Return per-draw metrics from a [draw, query, lag] matrix.

    The per-draw query mean is accumulated as float32 to match metric_curve;
    median and scalar summaries then use float64, also matching that routine.
    """
    values = np.asarray(matrix, dtype=np.float32)
    rows = np.asarray(indices, dtype=np.int64)
    if rows.size == 0:
        return None
    curve32 = np.mean(values[:, rows, :], axis=1, dtype=np.float32)
    curves = curve32.astype(np.float64)
    minima = np.min(curves, axis=1)
    medians = np.median(curves, axis=1)
    return {
        "C_best": medians - minima,
        "C_fixed_anchor": medians - curves[:, ANCHOR_COLUMN],
        "B_median": medians,
        "D_min": minima,
        "optimal_offset_frames": (VSHIFT - np.argmin(curves, axis=1)).astype(np.float64),
    }


def shared_bootstrap_indices() -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    return rng.integers(0, 40, size=(BOOTSTRAP_DRAWS, 40), dtype=np.int16)


def collect_context_time_null(
    sample_id: str,
    direction: str,
    ordered_index: int,
    rows: list[dict[str, Any]],
    matrix_data: dict[str, dict[str, Any]],
    supports: dict[str, np.ndarray],
    common_count: int,
) -> tuple[dict[tuple[Any, ...], np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    """Build all four-cell G/E/I/total null contrasts for one context."""
    lengths_to_streams: dict[int, set[str]] = defaultdict(set)
    for item in rows:
        if item.get("audio_id") and item.get("audio_domain_count") is not None:
            lengths_to_streams[int(item["audio_domain_count"])].add(str(item["audio_id"]))
    permutation_bank: dict[int, np.ndarray] = {}
    permutation_inventory = []
    for audio_count, stream_ids in sorted(lengths_to_streams.items()):
        rng = np.random.Generator(np.random.PCG64(NULL_SEED_BASE + ordered_index))
        permutations = np.stack([rng.permutation(audio_count) for _ in range(NULL_DRAWS)], axis=0).astype(np.int64, copy=False)
        permutation_bank[audio_count] = permutations
        permutation_inventory.append({
            "audio_domain_count": audio_count,
            "audio_stream_ids": sorted(stream_ids),
            "shape": list(permutations.shape),
            "dtype": str(permutations.dtype),
            "permutation_array_sha256": array_sha256(permutations),
            "rng": f"numpy.random.Generator(PCG64({NULL_SEED_BASE} + {ordered_index}))",
            "draw_order": "100 successive rng.permutation(audio_domain_count) calls",
        })

    matrix_null: dict[tuple[str, str, str, str], np.ndarray] = {}
    ineligible: list[dict[str, Any]] = []
    for matrix_id, matrix_data_item in sorted(matrix_data.items()):
        score = matrix_data_item["score"]
        audio_count = int(matrix_data_item["n_audio"])
        if audio_count not in permutation_bank:
            raise AuditError(f"no time-null permutation bank for A length {audio_count}")
        permutations = permutation_bank[audio_count]
        for domain in ("raw", "unit"):
            cross_key, zero_key, _ = DOMAIN_ARRAYS[domain]
            cross = score[cross_key]
            zero = score[zero_key]
            active_supports = [support for support in SUPPORTS if supports[support].size > 0]
            for support in SUPPORTS:
                if supports[support].size == 0:
                    ineligible.append({"matrix_id": matrix_id, "domain": domain, "support": support, "reason": "too_short_empty_support", "support_n": 0})
            permuted_lags = lag_from_permuted_cross_batch(
                cross, zero, permutations, common_count, audio_count,
            )
            draws_by_endpoint: dict[tuple[str, str], np.ndarray] = {}
            for support in active_supports:
                curves = metric_curves_batch(permuted_lags, supports[support])
                if curves is None:
                    raise AuditError(f"null metric support unexpectedly empty: {support}")
                for metric in NULL_METRICS:
                    draws_by_endpoint[(support, metric)] = curves[metric]
            for (support, metric), draws in draws_by_endpoint.items():
                matrix_null[(matrix_id, domain, support, metric)] = draws

    row_lookup = {(str(row["family"]), str(row["condition"]), str(row["cell"])): row for row in rows}
    context_null: dict[tuple[Any, ...], np.ndarray] = {}
    for family, arm in sorted({(str(row["family"]), str(row["condition"])) for row in rows}):
        cell_rows = {cell: row_lookup.get((family, arm, cell)) for cell in CELLS}
        if any(row is None for row in cell_rows.values()):
            raise AuditError(f"incomplete four-cell alias group: {sample_id}/{direction}/{family}/{arm}")
        for domain in ("raw", "unit"):
            for support in SUPPORTS:
                if supports[support].size == 0:
                    continue
                for metric in NULL_METRICS:
                    cell_draws: dict[str, np.ndarray] = {}
                    for cell, row in cell_rows.items():
                        matrix_id = str(row["matrix_id"])
                        value = matrix_null.get((matrix_id, domain, support, metric))
                        if value is None:
                            raise AuditError(f"null metric missing for {matrix_id}/{domain}/{support}/{metric}")
                        cell_draws[cell] = value
                    q00, q10, q01, q11 = (cell_draws[cell] for cell in CELLS)
                    contrasts = {
                        "G_q10_minus_q00": q10 - q00,
                        "E_q01_minus_q00": q01 - q00,
                        "I_q11_minus_q10_minus_q01_plus_q00": q11 - q10 - q01 + q00,
                        "total_q11_minus_q00": q11 - q00,
                    }
                    for contrast, values in contrasts.items():
                        key = (sample_id, direction, family, arm, domain, support, metric, contrast)
                        context_null[key] = np.asarray(values, dtype=np.float64)
    return context_null, {"ordered_eval_index": ordered_index, "audio_domains": permutation_inventory}, ineligible


def summarize_time_null(
    context_null: dict[tuple[Any, ...], np.ndarray],
    sample_to_speaker: dict[tuple[str, str], str],
    observed_speaker: dict[tuple[Any, ...], dict[str, Any]],
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    # context_null key: sample,direction,family,arm,domain,support,metric,contrast.
    grouped_null: dict[tuple[Any, ...], dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for (sample, direction, family, arm, domain, support, metric, contrast), draws in context_null.items():
        speaker = sample_to_speaker.get((sample, direction))
        if speaker is None:
            continue
        grouped_null[(direction, family, arm, domain, support, metric, contrast)][speaker].append(draws)
    output = []
    for key, by_speaker in sorted(grouped_null.items(), key=lambda item: tuple(str(x) for x in item[0])):
        direction, family, arm, domain, support, metric, contrast = key
        complete = {speaker: arrays for speaker, arrays in by_speaker.items() if len(arrays) == 2}
        observed = observed_speaker.get(key)
        if len(complete) != 40 or observed is None or len(observed.get("speaker_means", [])) != 40:
            output.append({
                "direction": direction, "family": family, "condition": arm,
                "domain": domain, "support": support, "metric": metric,
                "contrast": contrast, "status": "ineligible_incomplete_speakers",
                "complete_speaker_count": len(complete), "speakers_with_any_null_clip": len(by_speaker),
                "expected_speaker_count": 40,
            })
            continue
        speakers = sorted(complete)
        null_by_speaker = np.stack([
            np.stack(complete[speaker], axis=0).mean(axis=0, dtype=np.float64)
            for speaker in speakers
        ], axis=0)
        observed_by_speaker = dict(zip(observed["speakers"], observed["speaker_means"], strict=True))
        if set(observed_by_speaker) != set(speakers):
            output.append({
                "direction": direction, "family": family, "condition": arm,
                "domain": domain, "support": support, "metric": metric,
                "contrast": contrast, "status": "ineligible_observed_null_speaker_mismatch",
                "complete_speaker_count": len(complete),
            })
            continue
        observed_values = np.asarray([observed_by_speaker[speaker] for speaker in speakers], dtype=np.float64)
        null_draw_means = null_by_speaker.mean(axis=0, dtype=np.float64)
        observed_mean = float(observed_values.mean(dtype=np.float64))
        null_mean = float(null_draw_means.mean(dtype=np.float64))
        difference = observed_values - null_by_speaker.mean(axis=1, dtype=np.float64)
        difference_draws = difference[bootstrap_indices].mean(axis=1, dtype=np.float64)
        ci = np.quantile(difference_draws, [0.005, 0.995], method="linear")
        p_upper = (1 + int(np.sum(null_draw_means >= observed_mean))) / (NULL_DRAWS + 1)
        output.append({
            "direction": direction, "family": family, "condition": arm,
            "domain": domain, "support": support, "metric": metric,
            "contrast": contrast, "status": "ok", "speaker_count": 40,
            "observed_speaker_equal_mean": observed_mean,
            "null_mean_over_100_permutations": null_mean,
            "observed_minus_null_mean": observed_mean - null_mean,
            "observed_minus_null_ci99_speaker_bootstrap": [float(ci[0]), float(ci[1])],
            "permutation_p_plus_one_upper_tail_descriptive": float(p_upper),
            "permutations": NULL_DRAWS, "permutation_seed_base": NULL_SEED_BASE,
        })
    return {
        "draws": NULL_DRAWS,
        "seed": "PCG64(20260930 + ordered_eval_index)",
        "domain_policy": "Within each ordered eval context, each distinct complete audio-window length receives 100 successive full-domain permutations from a fresh PCG64(seed_base + ordered_eval_index) stream; all audio streams of equal length reuse the same permutation matrix.",
        "scope": "all arms and bridge labels; G/E/I/total four-cell contrasts; raw/unit C_best and C_fixed_anchor; full/valid/guard20/guard15",
        "support_policy": "The frozen recipient-common query support is unchanged. Empty guard support is reported as too_short/ineligible and is not treated as an audit failure.",
        "bootstrap": "20,000 x 40 shared speaker-index draws, two clips averaged within speaker before cohort averaging",
        "p_value": "descriptive one-sided upper-tail plus-one count; no confirmatory null claim",
        "summaries": output,
    }


def run_consumer(args: argparse.Namespace) -> dict[str, Any]:
    run = args.run_dir.resolve(strict=True)
    plan_path = args.execution_plan.resolve(strict=True)
    qa_path = args.qa04.resolve(strict=True)
    out = args.output_dir.resolve()
    if out == run or run in out.parents:
        raise AuditError("consumer output must be outside the sealed producer run")
    if out.exists() and any(out.iterdir()):
        raise AuditError(f"refusing nonempty output namespace: {out}")
    out.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, Any]] = []
    consumer_path = Path(__file__).resolve()
    code_sha = file_sha256(consumer_path)
    protocol_record = {
        "schema_version": 1,
        "classification": "independent CPU validation and prespecified WORLD score consumer; no producer imports",
        "consumer_source_path": str(consumer_path), "consumer_source_sha256": code_sha,
        "execution_plan_expected_sha256": EXPECTED_PLAN_SHA256,
        "execution_plan_embedded_fingerprint_sha256": EXPECTED_PLAN_FINGERPRINT_SHA256,
        "qa04_expected_sha256": EXPECTED_QA04_SHA256,
        "runner_source_expected_sha256": EXPECTED_RUNNER_SHA256,
        "runtime_wrapper_expected_sha256": EXPECTED_RUNTIME_WRAPPER_SHA256,
        "grid": {"contexts": EXPECTED_CONTEXTS, "labels": EXPECTED_ROWS, "unique_matrices": EXPECTED_MATRICES, "unique_videos": EXPECTED_VISUALS},
        "anchor": {"k": 3, "column": ANCHOR_COLUMN, "official_offset": -3},
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "speaker_blocks": 40, "clips_per_speaker": 2, "ci": "two-sided percentile 99%"},
        "runtime_flags": {"cublas_workspace_config": ":4096:8", "required_flags": EXPECTED_RUNTIME_FLAGS, "per_forward_check": "global nn.Module pre-hook in same runner process"},
        "metric_definitions": {
            "D0": "lag-curve value at column 15 (zero offset)",
            "search_gain": "D0 - D_min, recomputed from sealed lag matrices",
            "producer_D0": "if present, checked diagnostically; the consumer always computes D0 from the sealed lag matrix",
            "producer_search_gain": "retained only as a legacy diagnostic; never consumed as an estimate or audit equality gate",
            "too_short": "empty guard support is retained with status too_short and excluded only from that support-specific estimate",
        },
        "protocol_identity": {
            "raw_execution_plan_file_sha256": EXPECTED_PLAN_SHA256,
            "embedded_execution_plan_fingerprint_sha256": EXPECTED_PLAN_FINGERPRINT_SHA256,
            "run_and_preflight_protocol_self_hashes": "recomputed independently after removing only each protocol_sha256 self-field",
            "payload_comparison": "run-only execution fields are removed from run protocol; protocol_sha256 is removed from both payloads; all remaining immutable payload fields must match exactly",
        },
        "time_null": {
            "draws": NULL_DRAWS,
            "seed": "PCG64(20260930 + ordered_eval_index)",
            "scope": "all 9 arms in raw and within_WORLD plus bridge; G/E/I/total; raw/unit C_best and C_fixed_anchor; full/valid/guard20/guard15",
            "domain_policy": "same 100 full-A permutations reused for every stream with equal audio-window count; independent same-seed permutation stream per distinct count",
            "support_unchanged": True,
        },
        "score_reconstruction": {"distance": "float32 p=2 norm(x-y+1e-6), independent NumPy", "matrix": "full V x full A", "lag": "common visual query prefix against full A; 31 columns; zero distance only outside full A domain"},
    }
    (out / "consumer_protocol.json").write_text(json.dumps(protocol_record, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    plan, protocol, status, seal_record = verify_code_and_seals(run, plan_path, qa_path, errors)
    all_files, all_file_report = check_all_files(run, errors)
    input_wave_report = verify_input_wave_files(plan, errors)
    execution_asset_report = verify_execution_assets(protocol, errors)
    runner_record = protocol.get("code_files", {}).get("world_phone_transfer_gpu_evaluate_20261001.py", {})
    expected_wrapper_path = Path(runner_record.get("path", "")).with_name("world_gpu_runtime_wrapper_20261001_v3.py")
    runtime_flags_report = verify_runtime_flags(args.runtime_flags_json.resolve(strict=False), expected_wrapper_path, errors)
    if errors:
        # Do not parse result metrics or score arrays until the preflight plan,
        # runtime provenance, source waves, and every producer file hash pass.
        audit = {
            "schema_version": 1,
            "status": "FAIL_PRE_SCORE_PREFLIGHT",
            "classification": "seal and source preflight failed; no result metrics or score arrays were opened",
            "run_root": str(run),
            "consumer_source_sha256": code_sha,
            "run_seals": seal_record,
            "all_files_report": all_file_report,
            "input_wave_hash_report": input_wave_report,
            "execution_asset_report": execution_asset_report,
            "runtime_flags": runtime_flags_report,
            "error_count": len(errors),
            "errors": errors,
            "finished_epoch": time.time(),
        }
        audit_path = out / "world_consumer_audit.json"
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        stats_payload = {
            "schema_version": 1,
            "status": "NOT_RUN_PRE_SCORE_PREFLIGHT_FAILED",
            "source_audit_sha256": file_sha256(audit_path),
            "effects_computed": False,
            "finished_epoch": time.time(),
        }
        (out / "world_consumer_statistics.json").write_text(json.dumps(stats_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        seal_consumer_output(out, code_sha, run)
        return {"audit": audit, "statistics": stats_payload}

    expected_plan_rows = plan.get("rows", [])
    rows = read_jsonl(run / "evaluation_rows.jsonl")
    results = read_jsonl(run / "evaluation_results.jsonl")
    planned_by_id = {row.get("row_id"): row for row in expected_plan_rows}
    results_by_id = {row.get("row_id"): row for row in results}
    if len(planned_by_id) != EXPECTED_ROWS or len(results_by_id) != EXPECTED_ROWS:
        fail(errors, "row_id_cardinality_mismatch", plan=len(planned_by_id), results=len(results_by_id))
    frozen_fields = ("ordered_eval_index", "row_id", "sample_id", "speaker_id", "split", "family", "direction", "donor_condition", "recipient_condition", "condition", "cell", "face_id", "visual_id", "audio_id", "matrix_id")
    for row_id, planned in planned_by_id.items():
        actual = results_by_id.get(row_id)
        if actual is None:
            fail(errors, "planned_result_missing", row_id=row_id)
            continue
        for key in frozen_fields:
            if actual.get(key) != planned.get(key):
                fail(errors, "planned_label_changed", row_id=row_id, field=key)
                break
        if actual.get("status") != "scored":
            fail(errors, "planned_row_not_scored", row_id=row_id, status=actual.get("status"))
        if actual.get("fixed_anchor_k") != 3 or actual.get("fixed_anchor_column") != ANCHOR_COLUMN or actual.get("fixed_anchor_official_offset") != -3:
            fail(errors, "row_anchor_mismatch", row_id=row_id)
    if len(rows) == EXPECTED_ROWS:
        row_bytes = b"".join(canonical_json(row) + b"\n" for row in rows)
        if hashlib.sha256(row_bytes).hexdigest() != file_sha256(run / "evaluation_rows.jsonl"):
            fail(errors, "jsonl_row_canonicalization_mismatch")

    context_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        context_groups[(row.get("sample_id", ""), row.get("direction", ""))].append(row)
    if len(context_groups) != EXPECTED_CONTEXTS or any(len(group) != EXPECTED_PER_CONTEXT_ROWS for group in context_groups.values()):
        fail(errors, "context_alias_grid_mismatch", context_count=len(context_groups), per_context_counts=Counter(map(len, context_groups.values())).most_common(5))
    for key, group in context_groups.items():
        if len({row.get("matrix_id") for row in group}) != 36 or len({row.get("visual_id") for row in group}) != 10 or len({row.get("audio_id") for row in group}) != 10:
            fail(errors, "context_unique_artifact_cardinality_mismatch", context=list(key))
        for row in group:
            if row.get("family") == "bridge":
                expected_visual, expected_audio = expected_alias("bridge", "RT", row["cell"])
            else:
                expected_visual, expected_audio = expected_alias(row["family"], row["condition"], row["cell"])
            if (row.get("visual_id"), row.get("audio_id")) != (expected_visual, expected_audio):
                fail(errors, "alias_map_mismatch", row_id=row.get("row_id"), observed=[row.get("visual_id"), row.get("audio_id")], expected=[expected_visual, expected_audio])
            if row.get("matrix_id") != f"{expected_visual}__{expected_audio}":
                fail(errors, "matrix_id_mismatch", row_id=row.get("row_id"))

    if errors:
        audit = {
            "schema_version": 1,
            "status": "FAIL_RESULT_GRID_PREFLIGHT",
            "classification": "sealed producer files passed global hashes, but result identities/aliases failed; score arrays were not opened",
            "run_root": str(run),
            "consumer_source_sha256": code_sha,
            "run_seals": seal_record,
            "all_files_report": all_file_report,
            "input_wave_hash_report": input_wave_report,
            "execution_asset_report": execution_asset_report,
            "runtime_flags": runtime_flags_report,
            "row_coverage": {"planned": len(rows), "result": len(results)},
            "error_count": len(errors),
            "errors": errors,
            "finished_epoch": time.time(),
        }
        audit_path = out / "world_consumer_audit.json"
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        stats_payload = {
            "schema_version": 1,
            "status": "NOT_RUN_RESULT_GRID_PREFLIGHT_FAILED",
            "source_audit_sha256": file_sha256(audit_path),
            "effects_computed": False,
            "finished_epoch": time.time(),
        }
        (out / "world_consumer_statistics.json").write_text(json.dumps(stats_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        seal_consumer_output(out, code_sha, run)
        return {"audit": audit, "statistics": stats_payload}

    per_context_reports = []
    metrics_by_row: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    legacy_search_gain_diagnostic: dict[str, dict[str, Any]] = {}
    null_contexts: dict[tuple[Any, ...], np.ndarray] = {}
    null_permutation_inventory: list[dict[str, Any]] = []
    null_ineligible_supports: list[dict[str, Any]] = []
    forward_events_path = run / "forward_events.jsonl"
    forward_event_count = 0
    bad_event_count = 0
    if forward_events_path.is_file():
        for event in read_jsonl(forward_events_path):
            forward_event_count += 1
            if event.get("flags_at_call") != EXPECTED_FLAGS:
                bad_event_count += 1
        if forward_event_count != status.get("forward_event_count") or bad_event_count:
            fail(errors, "forward_event_inventory_mismatch", observed=forward_event_count, sealed=status.get("forward_event_count"), wrong_flags=bad_event_count)
    else:
        fail(errors, "forward_event_file_missing")

    ctx_plan = {(item.get("sample_id"), item.get("direction")): item for item in plan.get("contexts", [])}
    for context_key, group in sorted(context_groups.items()):
        sample_id, direction = context_key
        context_name = f"{sample_id}__{direction}"
        context_npz_path = run / "contexts" / f"{context_name}.npz"
        context_json_path = context_npz_path.with_suffix(".json")
        if not context_npz_path.is_file() or not context_json_path.is_file():
            fail(errors, "context_artifact_missing", context=context_name)
            continue
        context_meta = read_json(context_json_path)
        if file_sha256(context_npz_path) != context_meta.get("context_npz", {}).get("sha256"):
            fail(errors, "context_npz_hash_mismatch", context=context_name)
        if context_meta.get("sample_id") != sample_id or context_meta.get("direction") != direction:
            fail(errors, "context_identity_mismatch", context=context_name)
        context_relative = str(context_meta.get("context_npz", {}).get("path", "")).replace("\\", "/")
        context_sha = context_meta.get("context_npz", {}).get("sha256")
        for row in group:
            if row.get("context_npz") != context_relative or row.get("context_npz_sha256") != context_sha:
                fail(errors, "result_context_reference_mismatch", context=context_name, row_id=row.get("row_id"))
        if context_meta.get("fixed_anchor_k") not in (None, 3):
            fail(errors, "context_anchor_mismatch", context=context_name)
        flags = context_meta.get("actual_flags_at_forward", [])
        if not flags or any(json.loads(value) != EXPECTED_FLAGS for value in flags):
            fail(errors, "context_forward_flag_mismatch", context=context_name)
        with np.load(context_npz_path, allow_pickle=False) as archive:
            context_arrays = {key: np.asarray(archive[key]) for key in archive.files}
        meta_arrays = context_meta.get("context_arrays", {})
        for key, value in context_arrays.items():
            record = meta_arrays.get(key, {})
            if record.get("shape") != list(value.shape) or record.get("dtype") != str(value.dtype) or record.get("sha256") != array_sha256(value):
                fail(errors, "context_array_metadata_mismatch", context=context_name, array=key)
        common_count = int(context_meta.get("common_window_count", -1))
        visual_embeddings: dict[str, np.ndarray] = {}
        audio_embeddings: dict[str, np.ndarray] = {"A_RAW": context_arrays.get("raw_audio_embeddings_float32")}
        for arm in ARMS:
            audio_embeddings[f"A_{arm}"] = context_arrays.get(f"arm_{arm}_audio_embeddings")
            visual_embeddings[f"V_{arm}"] = context_arrays.get(f"arm_{arm}_lip_embeddings")
        visual_embeddings["V_FIXED"] = context_arrays.get("fixed_recipient_lip_embeddings")
        if any(value is None for value in audio_embeddings.values()) or any(value is None for value in visual_embeddings.values()):
            fail(errors, "context_embedding_key_missing", context=context_name)
            continue
        audio_lengths = {len(value) for value in audio_embeddings.values()}
        visual_lengths = {len(value) for value in visual_embeddings.values()}
        if not audio_lengths or not visual_lengths or common_count != min((*audio_lengths, *visual_lengths)):
            fail(errors, "common_clock_mismatch", context=context_name, common_count=common_count, audio_lengths=sorted(audio_lengths), visual_lengths=sorted(visual_lengths))
        if common_count < 1:
            fail(errors, "empty_common_clock", context=context_name)
            continue
        supports = expected_supports(common_count)
        if context_meta.get("support_indices") != {key: value.tolist() for key, value in supports.items()}:
            fail(errors, "context_support_indices_mismatch", context=context_name)
        for support_name in SUPPORTS:
            support_key = f"common_support_{support_name}_int32"
            if support_key not in context_arrays or context_arrays[support_key].dtype != np.int32 or not np.array_equal(context_arrays[support_key], supports[support_name]):
                fail(errors, "context_support_array_mismatch", context=context_name, support=support_name)
        context_rows_by_matrix: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            context_rows_by_matrix[row["matrix_id"]].append(row)
        matrix_data: dict[str, dict[str, Any]] = {}
        for matrix_id, aliases in sorted(context_rows_by_matrix.items()):
            representative = aliases[0]
            score_path = run / representative["score_npz"]
            if not score_path.is_file() or file_sha256(score_path) != representative.get("score_npz_sha256"):
                fail(errors, "score_npz_hash_mismatch", context=context_name, matrix=matrix_id)
                continue
            if any(row.get("score_npz") != representative.get("score_npz") or row.get("score_npz_sha256") != representative.get("score_npz_sha256") for row in aliases):
                fail(errors, "alias_score_artifact_mismatch", context=context_name, matrix=matrix_id)
            visual_id, audio_id = representative["visual_id"], representative["audio_id"]
            visual_npz_path = run / representative["visual_npz"]
            if not visual_npz_path.is_file() or file_sha256(visual_npz_path) != representative.get("visual_npz_sha256"):
                fail(errors, "visual_npz_hash_mismatch", context=context_name, visual=visual_id)
                continue
            visual_meta_path = visual_npz_path.with_suffix(".json")
            if not visual_meta_path.is_file():
                fail(errors, "visual_json_missing", context=context_name, visual=visual_id)
                continue
            visual_meta = read_json(visual_meta_path)
            if visual_meta.get("video_sha256") != representative.get("video_sha256") or visual_meta.get("video_path") != representative.get("video_path"):
                fail(errors, "visual_video_binding_mismatch", context=context_name, visual=visual_id)
            if visual_meta.get("visual_npz") != representative.get("visual_npz") or visual_meta.get("visual_npz_sha256") != representative.get("visual_npz_sha256"):
                fail(errors, "visual_npz_metadata_binding_mismatch", context=context_name, visual=visual_id)
            video_path = run / representative["video_path"]
            if not video_path.is_file() or file_sha256(video_path) != representative.get("video_sha256"):
                fail(errors, "video_hash_mismatch", context=context_name, visual=visual_id)
            with np.load(visual_npz_path, allow_pickle=False) as visual_archive:
                visual_npz = {key: np.asarray(visual_archive[key]) for key in visual_archive.files}
            visual_value = visual_embeddings[visual_id]
            if not np.array_equal(visual_npz.get("lip_embeddings_float32"), visual_value):
                fail(errors, "visual_context_embedding_mismatch", context=context_name, visual=visual_id)
            expected_z_key = "fixed_recipient_z_float32" if visual_id == "V_FIXED" else f"arm_{visual_id[2:]}_z"
            if expected_z_key not in context_arrays or not np.array_equal(visual_npz.get("latent_z_float32"), context_arrays[expected_z_key]):
                fail(errors, "visual_context_latent_mismatch", context=context_name, visual=visual_id)
            latent_frames = int(len(context_arrays[expected_z_key])) if expected_z_key in context_arrays else -1
            decoded_frames = visual_meta.get("decoded_video_frame_count")
            jpeg_frames = visual_meta.get("official_jpeg_frame_count")
            lip_frames = visual_meta.get("lip_window_count")
            if not (
                visual_meta.get("latent_frame_count") == latent_frames
                and visual_meta.get("encoded_video_frame_count") == latent_frames
                and decoded_frames == latent_frames
                and visual_meta.get("score_crop_frame_count") == decoded_frames
                and jpeg_frames == decoded_frames
                and isinstance(lip_frames, int)
                and lip_frames == jpeg_frames - 5
                and list(visual_npz.get("lip_embeddings_float32", np.empty((0,))).shape) == visual_meta.get("lip_embedding_shape")
                and str(visual_npz.get("lip_embeddings_float32", np.empty((0,))).dtype) == visual_meta.get("lip_embedding_dtype")
            ):
                fail(errors, "video_lip_clock_chain_mismatch", context=context_name, visual=visual_id)
            lip = visual_value
            audio = audio_embeddings[audio_id]
            recomputed_raw_cross, recomputed_raw_zero = cross_distance(lip, audio)
            lip_unit = row_unit(lip)
            audio_unit = row_unit(audio)
            recomputed_unit_cross, recomputed_unit_zero = cross_distance(lip_unit, audio_unit)
            with np.load(score_path, allow_pickle=False) as score_archive:
                score = {key: np.asarray(score_archive[key]) for key in score_archive.files}
            expected_keys = {key for domain in DOMAIN_ARRAYS.values() for key in domain} | {f"common_support_{name}_int32" for name in SUPPORTS}
            if set(score) != expected_keys:
                fail(errors, "score_npz_key_set_mismatch", context=context_name, matrix=matrix_id, observed=sorted(score), expected=sorted(expected_keys))
            stored_raw_cross = score["cross_distance_raw_float32"]
            stored_raw_zero = score["cross_distance_to_zero_raw_float32"]
            stored_unit_cross = score["cross_distance_unit_float32"]
            stored_unit_zero = score["cross_distance_to_zero_unit_float32"]
            for name, value in score.items():
                expected_dtype = np.dtype("int32") if name.startswith("common_support_") else np.dtype("float32")
                if value.dtype != expected_dtype:
                    fail(errors, "score_array_dtype_mismatch", context=context_name, matrix=matrix_id, array=name, expected=str(expected_dtype), actual=str(value.dtype))
            for name, actual, expected in (
                ("raw_cross", stored_raw_cross, recomputed_raw_cross),
                ("raw_zero", stored_raw_zero, recomputed_raw_zero),
                ("unit_cross", stored_unit_cross, recomputed_unit_cross),
                ("unit_zero", stored_unit_zero, recomputed_unit_zero),
            ):
                if actual.shape != expected.shape or not np.allclose(actual, expected, atol=CROSS_ATOL, rtol=CROSS_RTOL):
                    difference = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))) if actual.shape == expected.shape else None
                    fail(errors, "distance_recompute_mismatch", context=context_name, matrix=matrix_id, domain_array=name, max_abs_diff=difference)
            for name, actual, expected in (("raw_cross", stored_raw_cross, recomputed_raw_cross), ("unit_cross", stored_unit_cross, recomputed_unit_cross)):
                if actual.ndim != 2 or actual.shape != (len(lip), len(audio)):
                    fail(errors, "full_domain_matrix_shape_mismatch", context=context_name, matrix=matrix_id, domain_array=name, shape=list(actual.shape))
            if stored_raw_zero.shape != (len(lip),) or stored_unit_zero.shape != (len(lip),):
                fail(errors, "full_visual_zero_shape_mismatch", context=context_name, matrix=matrix_id)
            validate_support_arrays(score, common_count, errors, context_name, matrix_id)
            raw_lag_expected = lag_from_cross(stored_raw_cross, stored_raw_zero, common_count, len(audio))
            unit_lag_expected = lag_from_cross(stored_unit_cross, stored_unit_zero, common_count, len(audio))
            if score["lag_distance_raw_float32"].shape != (common_count, 31) or not np.array_equal(score["lag_distance_raw_float32"], raw_lag_expected):
                fail(errors, "raw_lag_reconstruction_mismatch", context=context_name, matrix=matrix_id)
            if score["lag_distance_unit_float32"].shape != (common_count, 31) or not np.array_equal(score["lag_distance_unit_float32"], unit_lag_expected):
                fail(errors, "unit_lag_reconstruction_mismatch", context=context_name, matrix=matrix_id)
            metric_arrays = {"raw": raw_lag_expected, "unit": unit_lag_expected}
            computed = {domain: {support: metric_curve(lag, supports[support]) for support in SUPPORTS} for domain, lag in metric_arrays.items()}
            for row in aliases:
                validate_metric_metadata(row, computed, errors, legacy_search_gain_diagnostic)
                metrics_by_row[row["row_id"]] = computed
            matrix_data[matrix_id] = {"score": score, "computed": computed, "visual_id": visual_id, "audio_id": audio_id, "n_visual": len(lip), "n_audio": len(audio)}
        if len(matrix_data) != 36:
            fail(errors, "context_unique_matrix_artifact_count_mismatch", context=context_name, observed=len(matrix_data))
        per_context_reports.append({
            "sample_id": sample_id, "direction": direction,
            "speaker_id": group[0].get("speaker_id") if group else None,
            "common_window_count": common_count,
            "raw_audio_window_count": len(audio_embeddings["A_RAW"]) if audio_embeddings.get("A_RAW") is not None else None,
            "audio_domain_counts": {key: len(value) for key, value in sorted(audio_embeddings.items())},
            "visual_window_counts": {key: len(value) for key, value in sorted(visual_embeddings.items())},
            "guard20_support_n": int(len(supports["guard20"])),
            "guard20_status": "ok" if len(supports["guard20"]) else "too_short",
            "score_matrix_count": len(matrix_data),
            "visual_count": len(visual_embeddings), "audio_stream_count": len(audio_embeddings),
        })

        if len(matrix_data) == 36 and common_count > 0:
            ordered_indices = {int(row["ordered_eval_index"]) for row in group}
            if len(ordered_indices) != 1:
                fail(errors, "time_null_ordered_eval_index_mismatch", context=context_name, values=sorted(ordered_indices))
            else:
                context_null, permutation_meta, context_ineligible = collect_context_time_null(
                    sample_id, direction, next(iter(ordered_indices)), group, matrix_data, supports, common_count,
                )
                null_contexts.update(context_null)
                null_permutation_inventory.append({"sample_id": sample_id, "direction": direction, **permutation_meta})
                null_ineligible_supports.extend({"sample_id": sample_id, "direction": direction, **item} for item in context_ineligible)

    if errors:
        # A failed independent reconstruction is an audit result, not a reason
        # to publish effect estimates from partially validated matrices.
        stats = {
            "bootstrap": {
                "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS,
                "summaries": [], "primary_raw_EX_guard20_gates": [],
            }
        }
        null_stats = {
            "status": "NOT_REPORTED_SCORE_RECONSTRUCTION_FAILED",
            "permutation_inventory": null_permutation_inventory,
            "ineligible_empty_support_matrix_count": len(null_ineligible_supports),
            "ineligible_empty_support_examples": null_ineligible_supports[:100],
            "summaries": [],
        }
    else:
        bootstrap_indices = shared_bootstrap_indices()
        stats, speaker_values = summarize_stats(results, metrics_by_row, errors, bootstrap_indices)
        sample_to_speaker = {(row["sample_id"], row["direction"]): row["speaker_id"] for row in results}
        null_stats = summarize_time_null(null_contexts, sample_to_speaker, speaker_values, bootstrap_indices)
        null_stats["permutation_inventory"] = null_permutation_inventory
        null_stats["ineligible_empty_support_matrix_count"] = len(null_ineligible_supports)
        null_stats["ineligible_empty_support_examples"] = null_ineligible_supports[:100]

    audit = {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "classification": "independent CPU raw/unit full-domain score and alias audit; effects opened only after producer all-files verification",
        "run_root": str(run),
        "consumer_source_sha256": code_sha,
        "run_seals": seal_record,
        "all_files_report": all_file_report,
        "input_wave_hash_report": input_wave_report,
        "execution_asset_report": execution_asset_report,
        "row_coverage": {"planned": len(rows), "result": len(results), "scored": sum(row.get("status") == "scored" for row in results), "failed": sum(row.get("status") != "scored" for row in results), "contexts": len(context_groups), "context_artifacts_audited": len(per_context_reports)},
        "matrix_coverage": {"planned_unique": EXPECTED_MATRICES, "audited_unique": sum(item["score_matrix_count"] for item in per_context_reports), "visuals_per_context": 10, "audio_streams_per_context": 10, "aliases_per_context": EXPECTED_PER_CONTEXT_ROWS},
        "forward_events": {"observed": forward_event_count, "wrong_flag_events": bad_event_count, "required_flags": EXPECTED_FLAGS},
        "runtime_flags": runtime_flags_report,
        "per_context_support_and_coverage": per_context_reports,
        "producer_legacy_search_gain_diagnostic": legacy_search_gain_diagnostic,
        "error_count": len(errors),
        "errors": errors,
        "finished_epoch": time.time(),
    }
    (out / "world_consumer_audit.json").write_text(json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    stats_payload = {
        "schema_version": 1,
        "status": "COMPLETE" if audit["status"] == "PASS" else "INCOMPLETE_AUDIT",
        "classification": "prespecified WORLD cell decomposition; descriptive secondary factors retained regardless of sign",
        "effects_computed": not bool(errors),
        "source_audit_sha256": file_sha256(out / "world_consumer_audit.json"),
        "estimands": {"G": "q10-q00", "E": "q01-q00", "I": "q11-q10-q01+q00", "total": "q11-q00", "groups": "per-sample paired clips -> two-clip speaker mean -> equal 40-speaker mean"},
        "bootstrap": stats["bootstrap"],
        "time_null": null_stats,
        "visual_endpoint": {"status": "not_recomputed_by_score_consumer", "validated": "all sealed MKV/video hashes and SyncNet lip/latent embeddings", "next_input": "sealed visual track producer must consume the same 160 contexts and 1,600 videos"},
        "scientific_eligibility": {"full_cohort_complete": audit["status"] == "PASS" and len(results) == EXPECTED_ROWS and status.get("failed_rows") == 0, "primary_gate": stats["bootstrap"]["primary_raw_EX_guard20_gates"]},
        "finished_epoch": time.time(),
    }
    (out / "world_consumer_statistics.json").write_text(json.dumps(stats_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    seal_consumer_output(out, code_sha, run)
    return {"audit": audit, "statistics": stats_payload}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--qa04", type=Path, required=True)
    parser.add_argument("--runtime-flags-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run_consumer(args)
        print(json.dumps({"status": result["audit"]["status"], "audit_path": str(args.output_dir / "world_consumer_audit.json"), "statistics_path": str(args.output_dir / "world_consumer_statistics.json"), "error_count": result["audit"]["error_count"]}, sort_keys=True))
        return 0 if result["audit"]["status"] == "PASS" else 2
    except Exception as exc:
        print(f"ERROR:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
