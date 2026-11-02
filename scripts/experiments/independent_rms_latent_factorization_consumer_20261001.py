#!/usr/bin/env python3
"""Independent CPU reconstruction for the frozen RMS latent-factorization run.

This consumer deliberately does not import the RMS producer, mel runner,
model helper, or producer scoring functions. It validates sealed latent and
feature files, rebuilds distance matrices and the specified full-A row
permutation nulls, then reports paired clip and complete-speaker summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


RMS_SPEC_SHA256 = "3eaf572afd83682e5dc9d09dc325313bc2e16b969183fc8f89d391424b861e23"
MEL_SPEC_SHA256 = "abeced0f6f50a94fe49fc085af40358071dacc5091c7c796c26c5edafbe63ee8"
TARGET_RMS = 0.0376838172675746
CONDITIONS = ("q00", "q10", "q01", "q11")
GEOMETRIES = ("raw", "unit")
SUPPORTS = ("full", "valid", "guard20", "guard15")
METRICS = ("C", "C_anchor", "B", "D", "D0", "search_gain")
CONTRASTS = {
    "direction_q10_minus_q00": {"q10": 1.0, "q00": -1.0},
    "radius_q01_minus_q00": {"q01": 1.0, "q00": -1.0},
    "joint_q11_minus_q00": {"q11": 1.0, "q00": -1.0},
    "interaction_q11_minus_q10_minus_q01_plus_q00": {"q11": 1.0, "q10": -1.0, "q01": -1.0, "q00": 1.0},
    "q10_minus_q01": {"q10": 1.0, "q01": -1.0},
}
VSHIFT = 15
ZERO_COLUMN = 15
ANCHOR_COLUMN = 18
EPS = np.float32(1e-6)
EMBEDDING_DIM = 1024
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260930
PERMUTATION_COUNT = 100
PERMUTATION_SEED = 20260930
RADIUS_ERROR_LIMIT = 1e-6


class ConsumerError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _payload_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical({key: item for key, item in value.items() if key != "payload_sha256"})).hexdigest()


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConsumerError(f"cannot read JSON object {path}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise ConsumerError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    result = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConsumerError(f"malformed JSONL at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ConsumerError(f"JSONL row is not an object at {path}:{line_number}")
        result.append(row)
    return result


def _require_sha(path: str | Path, expected: str, label: str) -> Path:
    target = Path(path).resolve(strict=True)
    actual = _sha_file(target)
    if actual != expected:
        raise ConsumerError(f"{label} SHA mismatch: {target} ({actual} != {expected})")
    return target


def _resolve_record_path(record: Mapping[str, Any], root: Path, label: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_absolute():
        path = root / path
    return _require_sha(path, str(record.get("file_sha256", "")), label)


def _load_npy(record: Mapping[str, Any], root: Path, label: str, *, ndim: int | None = None, dim: int | None = None) -> tuple[np.ndarray, Path]:
    path = _resolve_record_path(record, root, label)
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 or (ndim is not None and value.ndim != ndim):
        raise ConsumerError(f"{label} has invalid type/dtype/shape: {path}")
    if dim is not None and (value.ndim != 2 or value.shape[1] != dim):
        raise ConsumerError(f"{label} embedding dimension differs: {path} {value.shape}")
    if not np.isfinite(value).all():
        raise ConsumerError(f"{label} contains nonfinite values: {path}")
    if list(value.shape) != record.get("shape") or str(value.dtype) != record.get("dtype") or _sha_array(value) != record.get("array_sha256"):
        raise ConsumerError(f"{label} array metadata mismatch: {path}")
    return value, path


def _load_npz(record: Mapping[str, Any], root: Path, label: str) -> tuple[dict[str, np.ndarray], Path]:
    path = _resolve_record_path(record, root, label)
    try:
        with np.load(path, allow_pickle=False) as archive:
            values = {key: np.asarray(archive[key]) for key in archive.files}
    except Exception as exc:
        raise ConsumerError(f"cannot load {label}: {path}: {type(exc).__name__}") from exc
    metadata = record.get("arrays")
    if not isinstance(metadata, Mapping) or set(values) != set(metadata):
        raise ConsumerError(f"{label} array inventory differs: {path}")
    for name, value in values.items():
        expected = metadata[name]
        if list(value.shape) != expected.get("shape") or str(value.dtype) != expected.get("dtype") or _sha_array(value) != expected.get("array_sha256"):
            raise ConsumerError(f"{label} array metadata mismatch: {path}:{name}")
    return values, path


def _row_unit(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    norms = np.sqrt(np.sum(x * x, axis=1, keepdims=True, dtype=np.float32)).astype(np.float32, copy=False)
    return (x / np.maximum(norms, np.float32(1e-12))).astype(np.float32, copy=False)


def _cross_distance(video: np.ndarray, audio: np.ndarray) -> np.ndarray:
    v = np.asarray(video, dtype=np.float32)
    a = np.asarray(audio, dtype=np.float32)
    if v.ndim != 2 or a.ndim != 2 or v.shape[1] != a.shape[1] or len(v) < 1 or len(a) < 1:
        raise ConsumerError(f"invalid V/A embedding geometry: {v.shape}/{a.shape}")
    result = np.empty((len(v), len(a)), dtype=np.float32)
    for start in range(0, len(v), 8):
        stop = min(start + 8, len(v))
        delta = v[start:stop, None, :] - a[None, :, :]
        delta += EPS
        squared = np.sum(delta * delta, axis=-1, dtype=np.float32)
        result[start:stop] = np.sqrt(squared).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ConsumerError("reconstructed cross-distance matrix is nonfinite")
    return result


def _zero_distance(video: np.ndarray) -> np.ndarray:
    x = np.asarray(video, dtype=np.float32) + EPS
    result = np.sqrt(np.sum(x * x, axis=1, dtype=np.float32)).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ConsumerError("reconstructed zero-boundary vector is nonfinite")
    return result


def _lag_distance(cross: np.ndarray, zero: np.ndarray, query_count: int, audio_count: int) -> np.ndarray:
    result = np.empty((query_count, 31), dtype=np.float32)
    for query in range(query_count):
        for column, lag in enumerate(range(-VSHIFT, VSHIFT + 1)):
            audio_index = query + lag
            result[query, column] = cross[query, audio_index] if 0 <= audio_index < audio_count else zero[query]
    return result


def _supports(count: int) -> dict[str, np.ndarray]:
    full = np.arange(count, dtype=np.int32)
    return {
        "full": full,
        "valid": full.copy(),
        "guard20": np.arange(min(20, count), max(min(20, count), count - 20), dtype=np.int32),
        "guard15": np.arange(min(15, count), max(min(15, count), count - 15), dtype=np.int32),
    }


def _metrics(matrix: np.ndarray, supports: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in SUPPORTS:
        indices = supports[name]
        if not len(indices):
            result[name] = {"status": "too_short", "support_n": 0, "indices": []}
            continue
        curve = np.asarray(matrix[indices], dtype=np.float32).mean(axis=0, dtype=np.float32).astype(np.float64)
        min_index = int(np.argmin(curve))
        minimum = float(curve[min_index])
        baseline = float(curve[ZERO_COLUMN])
        median = float(np.median(curve))
        result[name] = {
            "status": "ok",
            "support_n": int(len(indices)),
            "indices": indices.tolist(),
            "curve": curve.tolist(),
            "C": median - minimum,
            "C_anchor": median - float(curve[ANCHOR_COLUMN]),
            "B": median,
            "D": minimum,
            "D0": baseline,
            "search_gain": baseline - minimum,
            "D_anchor": float(curve[ANCHOR_COLUMN]),
            "best_lag": int(min_index - VSHIFT),
        }
    return result


def _metric_match(observed: Mapping[str, Any], reconstructed: Mapping[str, Any], label: str) -> None:
    for support in SUPPORTS:
        left, right = observed.get(support), reconstructed.get(support)
        if not isinstance(left, Mapping) or not isinstance(right, Mapping) or left.get("status") != right.get("status"):
            raise ConsumerError(f"{label} support status mismatch: {support}")
        if left.get("status") != "ok":
            continue
        if left.get("support_n") != right.get("support_n") or left.get("indices") != right.get("indices") or left.get("best_lag") != right.get("best_lag"):
            raise ConsumerError(f"{label} support coordinates mismatch: {support}")
        for field in ("C", "C_anchor", "B", "D", "D0", "search_gain", "D_anchor"):
            if not math.isclose(float(left[field]), float(right[field]), rel_tol=1e-5, abs_tol=2e-5):
                raise ConsumerError(f"{label} {support}/{field} differs: {left[field]} vs {right[field]}")
        if not np.allclose(np.asarray(left["curve"], dtype=np.float64), np.asarray(right["curve"], dtype=np.float64), rtol=1e-5, atol=2e-5):
            raise ConsumerError(f"{label} {support} lag curve differs")


def _compare_array(actual: np.ndarray, expected: np.ndarray, label: str, *, exact: bool) -> dict[str, Any]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ConsumerError(f"{label} shape/dtype mismatch: {actual.shape}/{actual.dtype} != {expected.shape}/{expected.dtype}")
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    max_abs = float(np.max(np.abs(delta), initial=0.0))
    equal = bool(np.array_equal(actual, expected))
    if exact and not equal:
        raise ConsumerError(f"{label} is not bit-exact (max abs {max_abs})")
    if not exact and not np.allclose(actual, expected, rtol=2e-5, atol=2e-5):
        raise ConsumerError(f"{label} numerical reconstruction mismatch (max abs {max_abs})")
    return {"exact": equal, "max_abs_difference": max_abs, "actual_sha256": _sha_array(actual), "expected_sha256": _sha_array(expected)}


def _latent_geometry(z0: np.ndarray, z1: np.ndarray) -> dict[str, np.ndarray]:
    if z0.dtype != np.float32 or z1.dtype != np.float32 or z0.ndim != 2 or z1.shape != z0.shape or z0.shape[1] != 512:
        raise ConsumerError(f"latent endpoints must be same float32 [frames,512]: {z0.shape}/{z0.dtype} {z1.shape}/{z1.dtype}")
    if not np.isfinite(z0).all() or not np.isfinite(z1).all():
        raise ConsumerError("nonfinite latent endpoint")
    a, b = z0.astype(np.float64), z1.astype(np.float64)
    r0, r1 = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    if np.any(r0 <= 0) or np.any(r1 <= 0) or not np.isfinite(r0).all() or not np.isfinite(r1).all():
        raise ConsumerError("latent endpoint has nonpositive/nonfinite frame radius")
    unit0, unit1 = a / r0[:, None], b / r1[:, None]
    q10 = (b * (r0 / r1)[:, None]).astype(np.float32)
    q01 = (a * (r1 / r0)[:, None]).astype(np.float32)
    return {
        "q00": z0.copy(),
        "q10": q10,
        "q01": q01,
        "q11": z1.copy(),
        "r0_float64": r0,
        "r1_float64": r1,
        "radius_ratio_r1_over_r0_float64": r1 / r0,
        "cosine_float64": np.sum(unit0 * unit1, axis=1, dtype=np.float64),
        "direction_delta_l2_float64": np.linalg.norm(unit1 - unit0, axis=1),
        "radius_delta_float64": r1 - r0,
        "q10_delta_from_q00_l2_float64": np.linalg.norm(q10.astype(np.float64) - a, axis=1),
        "q01_delta_from_q00_l2_float64": np.linalg.norm(q01.astype(np.float64) - a, axis=1),
        "interaction_l2_float64": np.linalg.norm(b - q10.astype(np.float64) - q01.astype(np.float64) + a, axis=1),
        "analytic_interaction_l2_float64": np.linalg.norm((r1 - r0)[:, None] * (unit1 - unit0), axis=1),
        "q10_mixed_relative_radius_error_float64": np.abs(np.linalg.norm(q10.astype(np.float64), axis=1) - r0) / r0,
        "q01_mixed_relative_radius_error_float64": np.abs(np.linalg.norm(q01.astype(np.float64), axis=1) - r1) / r1,
    }


def _verify_file_ledger(run_manifest: Mapping[str, Any], artifact_root: Path) -> dict[str, Any]:
    ledger = run_manifest.get("artifact_ledger")
    if not isinstance(ledger, Mapping) or len(ledger) != int(run_manifest.get("artifact_count", -1)):
        raise ConsumerError("run artifact ledger/count mismatch")
    root = artifact_root.resolve(strict=True)
    verified = 0
    for rel, expected in ledger.items():
        path = (root / str(rel)).resolve(strict=True)
        if root not in path.parents:
            raise ConsumerError(f"artifact ledger path escapes the sealed output root: {rel}")
        if _sha_file(path) != expected:
            raise ConsumerError(f"artifact ledger file SHA mismatch: {rel}")
        verified += 1
    return {"verified_file_count": verified, "ledger_sha256": hashlib.sha256(_canonical(dict(ledger))).hexdigest()}


def _load_cell_arrays(row: Mapping[str, Any], artifact_root: Path, sample_id: str) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    shared = row.get("shared_A")
    if not isinstance(shared, Mapping):
        raise ConsumerError(f"missing shared RAW-N A embedding: {sample_id}")
    audio, _ = _load_npy(shared, artifact_root, f"shared A {sample_id}", ndim=2, dim=EMBEDDING_DIM)
    cells = row.get("cells")
    if not isinstance(cells, Mapping):
        raise ConsumerError(f"missing cells: {sample_id}")
    videos: dict[str, np.ndarray] = {}
    dist_npz: dict[str, dict[str, np.ndarray]] = {}
    for condition in CONDITIONS:
        cell = cells.get(condition)
        if not isinstance(cell, Mapping) or cell.get("status") != "generated_scored":
            raise ConsumerError(f"incomplete video cell: {sample_id}/{condition}")
        v_record = cell.get("V")
        if not isinstance(v_record, Mapping):
            raise ConsumerError(f"missing V record: {sample_id}/{condition}")
        v, _ = _load_npy(v_record, artifact_root, f"V {sample_id}/{condition}", ndim=2, dim=EMBEDDING_DIM)
        clock = cell.get("frame_clock_check")
        if not isinstance(clock, Mapping) or clock.get("passed") is not True:
            raise ConsumerError(f"missing/pass-failed exact frame clock audit: {sample_id}/{condition}")
        latent_frames = int(clock.get("latent_frames", -1))
        if any(int(clock.get(name, -1)) != latent_frames for name in ("decoded_video_frames", "score_crop_frames", "syncnet_v_decoded_frames")):
            raise ConsumerError(f"video/crop/SyncNet frame clock mismatch: {sample_id}/{condition}")
        if len(v) != latent_frames - 5 or int(clock.get("syncnet_v_feature_windows", -1)) != len(v):
            raise ConsumerError(f"SyncNet V window count mismatch: {sample_id}/{condition}")
        videos[condition] = v
    scores = row.get("scores")
    if not isinstance(scores, Mapping) or not isinstance(scores.get("geometries"), Mapping):
        raise ConsumerError(f"missing score geometry records: {sample_id}")
    for geometry in GEOMETRIES:
        geometry_record = scores["geometries"].get(geometry)
        if not isinstance(geometry_record, Mapping):
            raise ConsumerError(f"missing {geometry} score geometry: {sample_id}")
        for condition in CONDITIONS:
            cell_score = geometry_record.get(condition)
            if not isinstance(cell_score, Mapping) or not isinstance(cell_score.get("distance_artifact"), Mapping):
                raise ConsumerError(f"missing distance artifact: {sample_id}/{geometry}/{condition}")
            archive, _ = _load_npz(cell_score["distance_artifact"], artifact_root, f"distance {sample_id}/{geometry}/{condition}")
            required = {"cross_v_to_a_float32", "zero_boundary_float32", "lag_distance_float32", "support_full_int32", "support_valid_int32", "support_guard20_int32", "support_guard15_int32", "lag_columns_int32", "lag_k_int32"}
            if not required.issubset(archive):
                raise ConsumerError(f"distance artifact keys missing: {sample_id}/{geometry}/{condition}")
            dist_npz[f"{geometry}/{condition}"] = archive
    return audio, videos, dist_npz


def _distance_and_permutation_rows(
    sample: Mapping[str, Any],
    result: Mapping[str, Any],
    artifact_root: Path,
    ordered_index: int,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    sample_id = str(sample["id"])
    factor = sample.get("mixed_latents")
    endpoints = sample.get("endpoint_latents")
    if not isinstance(factor, Mapping) or not isinstance(endpoints, Mapping):
        raise ConsumerError(f"missing latent endpoints/factors: {sample_id}")
    z_by_condition = {}
    endpoint_paths = {}
    for condition in ("q00", "q11"):
        record = endpoints.get(condition)
        if not isinstance(record, Mapping):
            raise ConsumerError(f"missing exact saved endpoint {condition}: {sample_id}")
        z_by_condition[condition], path = _load_npy(record, artifact_root, f"endpoint {sample_id}/{condition}", ndim=2, dim=512)
        endpoint_paths[condition] = path
    mixed_arrays, mixed_path = _load_npz(factor, artifact_root, f"factor archive {sample_id}")
    geometry = _latent_geometry(z_by_condition["q00"], z_by_condition["q11"])
    for condition in CONDITIONS:
        actual = z_by_condition[condition] if condition in z_by_condition else mixed_arrays.get(condition)
        if actual is None:
            raise ConsumerError(f"missing mixed latent {condition}: {sample_id}")
        expected = geometry[condition]
        _compare_array(actual, expected, f"latent {sample_id}/{condition}", exact=True)
    for name, expected in geometry.items():
        if name in ("q00", "q11"):
            continue
        actual = mixed_arrays.get(name)
        if actual is None:
            raise ConsumerError(f"missing latent geometry array {name}: {sample_id}")
        exact = name in ("q10", "q01")
        _compare_array(actual, expected, f"latent geometry {sample_id}/{name}", exact=exact)
    q10_error = float(np.max(geometry["q10_mixed_relative_radius_error_float64"], initial=0.0))
    q01_error = float(np.max(geometry["q01_mixed_relative_radius_error_float64"], initial=0.0))
    if q10_error > RADIUS_ERROR_LIMIT or q01_error > RADIUS_ERROR_LIMIT:
        raise ConsumerError(f"mixed latent radius error exceeds the frozen limit: {sample_id}/{q10_error}/{q01_error}")

    cell_inputs = result.get("cells")
    if not isinstance(cell_inputs, Mapping):
        raise ConsumerError(f"missing decoder-cell inputs: {sample_id}")
    for condition in CONDITIONS:
        cell = cell_inputs.get(condition)
        latent_input = cell.get("latent_input") if isinstance(cell, Mapping) else None
        if not isinstance(latent_input, Mapping):
            raise ConsumerError(f"missing sealed decoder latent input metadata: {sample_id}/{condition}")
        if condition in endpoints:
            expected_meta = endpoints[condition]
        else:
            expected_meta = {
                "path": factor.get("path"),
                "file_sha256": factor.get("file_sha256"),
                **factor.get("arrays", {}).get(condition, {}),
            }
        for field in ("file_sha256", "array_sha256", "shape", "dtype"):
            if latent_input.get(field) != expected_meta.get(field):
                raise ConsumerError(f"decoder latent input metadata differs from sealed factor plan: {sample_id}/{condition}/{field}")
        if Path(str(latent_input.get("path", ""))).resolve(strict=True) != Path(str(expected_meta.get("path", ""))).resolve(strict=True):
            raise ConsumerError(f"decoder latent input path differs from sealed factor plan: {sample_id}/{condition}")
        if condition in ("q10", "q01") and latent_input.get("archive_key") != condition:
            raise ConsumerError(f"mixed decoder input selects the wrong archive key: {sample_id}/{condition}")

    audio, videos, saved_distances = _load_cell_arrays(result, artifact_root, sample_id)
    common_n = min(len(audio), *(len(videos[condition]) for condition in CONDITIONS))
    if common_n < 1:
        raise ConsumerError(f"no common A/V support: {sample_id}")
    supports = _supports(common_n)
    score_root = result["scores"]["geometries"]
    observed: dict[str, Any] = {"common_support_n": common_n, "per_geometry": {}}
    null_summary: dict[str, Any] = {"common_support_n": common_n, "permutation_count": PERMUTATION_COUNT, "per_geometry": {}}
    contrast_row: dict[str, Any] = {}
    rng = np.random.Generator(np.random.PCG64(PERMUTATION_SEED + ordered_index))
    permutations = np.stack([rng.permutation(len(audio)) for _ in range(PERMUTATION_COUNT)], axis=0).astype(np.int64, copy=False)
    permutation_sha = _sha_array(permutations)
    cell_metrics: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    null_cell_metrics: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}

    for geometry_name in GEOMETRIES:
        audio_geometry = audio if geometry_name == "raw" else _row_unit(audio)
        per_geometry: dict[str, Any] = {}
        null_geometry: dict[str, Any] = {}
        cell_metrics[geometry_name] = {}
        null_cell_metrics[geometry_name] = {}
        for condition in CONDITIONS:
            video_geometry = videos[condition] if geometry_name == "raw" else _row_unit(videos[condition])
            rebuilt_cross = _cross_distance(video_geometry, audio_geometry)
            rebuilt_zero = _zero_distance(video_geometry)
            rebuilt_lag = _lag_distance(rebuilt_cross, rebuilt_zero, common_n, len(audio_geometry))
            saved = saved_distances[f"{geometry_name}/{condition}"]
            stored_cross = saved["cross_v_to_a_float32"]
            stored_zero = saved["zero_boundary_float32"]
            stored_lag = saved["lag_distance_float32"]
            _compare_array(stored_cross, rebuilt_cross, f"cross matrix {sample_id}/{geometry_name}/{condition}", exact=False)
            _compare_array(stored_zero, rebuilt_zero, f"zero boundary {sample_id}/{geometry_name}/{condition}", exact=False)
            _compare_array(stored_lag, rebuilt_lag, f"lag matrix {sample_id}/{geometry_name}/{condition}", exact=False)
            for support in SUPPORTS:
                if not np.array_equal(saved[f"support_{support}_int32"], supports[support]):
                    raise ConsumerError(f"saved support indices differ: {sample_id}/{geometry_name}/{condition}/{support}")
            if not np.array_equal(saved["lag_columns_int32"], np.arange(31, dtype=np.int32)) or not np.array_equal(saved["lag_k_int32"], np.arange(-15, 16, dtype=np.int32)):
                raise ConsumerError(f"lag column identity differs: {sample_id}/{geometry_name}/{condition}")
            metrics = _metrics(rebuilt_lag, supports)
            observed_record = score_root[geometry_name][condition]["metrics"]
            _metric_match(observed_record, metrics, f"stored scores {sample_id}/{geometry_name}/{condition}")
            per_geometry[condition] = metrics
            cell_metrics[geometry_name][condition] = metrics

            null_metrics_accum: dict[str, dict[str, list[float]]] = {support: {metric: [] for metric in METRICS} for support in SUPPORTS}
            for permutation in permutations:
                permuted_cross = rebuilt_cross[:, permutation]
                permuted_lag = _lag_distance(permuted_cross, rebuilt_zero, common_n, len(audio_geometry))
                rep_metrics = _metrics(permuted_lag, supports)
                for support in SUPPORTS:
                    if rep_metrics[support]["status"] != "ok":
                        continue
                    for metric in METRICS:
                        null_metrics_accum[support][metric].append(float(rep_metrics[support][metric]))
            null_cell_metrics[geometry_name][condition] = {}
            null_condition_record: dict[str, Any] = {}
            for support in SUPPORTS:
                null_cell_metrics[geometry_name][condition][support] = {}
                null_support_record: dict[str, Any] = {}
                for metric in METRICS:
                    values = np.asarray(null_metrics_accum[support][metric], dtype=np.float64)
                    if not len(values):
                        null_cell_metrics[geometry_name][condition][support][metric] = None
                        null_support_record[metric] = {"status": "too_short", "permutation_n": 0}
                        continue
                    observed_value = float(metrics[support][metric])
                    null_mean = float(np.mean(values, dtype=np.float64))
                    record_value = {"status": "ok", "observed": observed_value, "null_mean": null_mean, "observed_minus_null_mean": observed_value - null_mean,
                                    "null_p05": float(np.quantile(values, 0.05)), "null_p50": float(np.quantile(values, 0.50)), "null_p95": float(np.quantile(values, 0.95)), "permutation_n": int(len(values))}
                    null_cell_metrics[geometry_name][condition][support][metric] = null_mean
                    null_support_record[metric] = record_value
                null_condition_record[support] = null_support_record
            null_geometry[condition] = null_condition_record
        observed["per_geometry"][geometry_name] = per_geometry
        null_summary["per_geometry"][geometry_name] = null_geometry

    for contrast_name, weights in CONTRASTS.items():
        contrast_row[contrast_name] = {}
        for geometry_name in GEOMETRIES:
            contrast_row[contrast_name][geometry_name] = {}
            for support in SUPPORTS:
                contrast_row[contrast_name][geometry_name][support] = {}
                for metric in METRICS:
                    values = []
                    adjusted = []
                    for condition, weight in weights.items():
                        observed_cell = cell_metrics[geometry_name][condition][support].get(metric)
                        null_cell = null_cell_metrics[geometry_name][condition][support].get(metric)
                        if observed_cell is None or observed_cell.get("status") != "ok" or null_cell is None:
                            values = []
                            adjusted = []
                            break
                        values.append(weight * float(observed_cell[metric]))
                        adjusted.append(weight * (float(observed_cell[metric]) - float(null_cell)))
                    contrast_row[contrast_name][geometry_name][support][metric] = {
                        "observed": float(sum(values)) if values else None,
                        "null_adjusted": float(sum(adjusted)) if adjusted else None,
                    }

    row_summary = {
        "id": sample_id,
        "speaker": str(sample["speaker"]),
        "ordered_index": ordered_index,
        "status": "complete",
        "latent_frames": int(len(z_by_condition["q00"])),
        "audio_rows": int(len(audio)),
        "V_rows": {condition: int(len(videos[condition])) for condition in CONDITIONS},
        "q10_relative_radius_error_max": q10_error,
        "q01_relative_radius_error_max": q01_error,
        "latent_archive_path": str(mixed_path),
        "latent_archive_sha256": _sha_file(mixed_path),
        "endpoint_sha256": {condition: _sha_file(path) for condition, path in endpoint_paths.items()},
        "permutation_seed": PERMUTATION_SEED + ordered_index,
        "permutation_count": PERMUTATION_COUNT,
        "permutation_indices_sha256": permutation_sha,
        "observed_metrics": observed,
        "permutation_null": null_summary,
        "contrast_metrics": contrast_row,
        "model_forward_performed_by_consumer": False,
        "effects_read_by_consumer": True,
    }
    return row_summary, contrast_row, permutation_sha


def _group_summary(
    plan_samples: list[dict[str, Any]],
    row_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    speaker_rows: dict[str, list[str]] = {}
    for sample in plan_samples:
        speaker_rows.setdefault(str(sample["speaker"]), []).append(str(sample["id"]))
    if len(speaker_rows) != 40 or any(len(ids) != 2 for ids in speaker_rows.values()):
        raise ConsumerError("planned EVAL cohort must retain exactly two planned rows per each of 40 speakers")

    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    bootstrap_cache: dict[int, np.ndarray] = {}
    group: dict[str, Any] = {}
    for contrast_name in CONTRASTS:
        group[contrast_name] = {}
        for geometry in GEOMETRIES:
            group[contrast_name][geometry] = {}
            for support in SUPPORTS:
                group[contrast_name][geometry][support] = {}
                for metric in METRICS:
                    group[contrast_name][geometry][support][metric] = {}
                    for basis in ("observed", "null_adjusted"):
                        speaker_values: list[tuple[str, float]] = []
                        incomplete: list[str] = []
                        paired_clips = 0
                        for speaker, ids in sorted(speaker_rows.items()):
                            values = []
                            for sample_id in ids:
                                row = row_summaries.get(sample_id)
                                value = None if row is None or row.get("status") != "complete" else row["contrast_metrics"][contrast_name][geometry][support][metric].get(basis)
                                if value is None:
                                    values = []
                                    break
                                values.append(float(value))
                            if len(values) == 2:
                                paired_clips += 2
                                speaker_values.append((speaker, float(np.mean(values, dtype=np.float64))))
                            else:
                                incomplete.append(speaker)
                        speaker_array = np.asarray([value for _, value in speaker_values], dtype=np.float64)
                        if len(speaker_array):
                            mean = float(np.mean(speaker_array, dtype=np.float64))
                            if len(speaker_array) not in bootstrap_cache:
                                bootstrap_cache[len(speaker_array)] = rng.integers(0, len(speaker_array), size=(BOOTSTRAP_DRAWS, len(speaker_array)), endpoint=False)
                            draws = bootstrap_cache[len(speaker_array)]
                            boot_means = speaker_array[draws].mean(axis=1, dtype=np.float64)
                            low, high = np.quantile(boot_means, [0.005, 0.995])
                            interval = {"lower_99": float(low), "upper_99": float(high), "draw_count": BOOTSTRAP_DRAWS, "rng": "PCG64", "seed": BOOTSTRAP_SEED}
                        else:
                            mean, interval = None, None
                        group[contrast_name][geometry][support][metric][basis] = {
                            "mean_equal_weight_complete_speakers": mean,
                            "bootstrap_99_ci": interval,
                            "intended_speakers": len(speaker_rows),
                            "complete_speakers": len(speaker_values),
                            "incomplete_speakers": incomplete,
                            "complete_two_clip_speakers": len(speaker_values),
                            "paired_clips_in_complete_speakers": paired_clips,
                            "all_intended_rows_retained": len(plan_samples) == 80,
                        }
    return group


def consume(args: argparse.Namespace) -> None:
    plan_path = args.plan.resolve(strict=True)
    run_path = args.run_manifest.resolve(strict=True)
    rows_path = args.rows.resolve(strict=True)
    producer_path = args.producer_code.resolve(strict=True)
    spec_path = args.spec.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise ConsumerError(f"consumer report path already exists; use a fresh immutable attempt: {output}")

    plan = _read_json(plan_path)
    run = _read_json(run_path)
    if plan.get("payload_sha256") != _payload_sha(plan) or run.get("payload_sha256") != _payload_sha(run):
        raise ConsumerError("producer plan or run manifest payload fingerprint mismatch")
    if plan.get("code", {}).get("sha256") != _sha_file(producer_path) or run.get("runner_code_sha256") != _sha_file(producer_path):
        raise ConsumerError("producer source hash differs from RMS execution plan/run manifest")
    if plan.get("spec", {}).get("sha256") != RMS_SPEC_SHA256:
        raise ConsumerError("RMS plan is bound to a different frozen spec")
    _require_sha(spec_path, RMS_SPEC_SHA256, "frozen RMS factorization spec")
    if Path(str(plan["spec"]["path"])).resolve() != spec_path.resolve():
        raise ConsumerError("consumer spec path differs from the bound RMS plan")
    if plan.get("mel_spec_sha256") != MEL_SPEC_SHA256:
        raise ConsumerError("RMS plan is bound to a different frozen mel spec")
    if run.get("status") not in ("complete", "complete_with_failures") or run.get("model_forward_performed") is not True or run.get("audio_encoder_forward_performed") is not False or run.get("mel_frontend_recomputed") is not False or run.get("effects_read") is not False:
        raise ConsumerError("RMS run is not a completed fresh-decoder/no-audio-encoder/no-effects producer run")
    if int(plan.get("planned_sample_count", -1)) != 80 or int(plan.get("planned_decoder_rows", -1)) != 320 or int(run.get("planned_rows", -1)) != 80 or int(run.get("planned_cells", -1)) != 320:
        raise ConsumerError("RMS source run does not retain the frozen 80x4 design")
    if Path(str(run.get("execution_plan_path", ""))).resolve() != plan_path.resolve() or run.get("execution_plan_raw_sha256") != _sha_file(plan_path):
        raise ConsumerError("RMS run manifest is not attached to the supplied execution plan")
    if Path(str(run.get("rows_path", ""))).resolve() != rows_path.resolve() or run.get("rows_raw_sha256") != _sha_file(rows_path):
        raise ConsumerError("RMS run rows path/hash differs from the supplied rows file")
    factor_rows_value = plan.get("rows_path")
    if not isinstance(factor_rows_value, str) or not factor_rows_value:
        raise ConsumerError("factorization plan lacks a factor rows path")
    factor_rows_path = Path(factor_rows_value).resolve(strict=True)
    if factor_rows_path == rows_path:
        raise ConsumerError("factor plan rows path must point to factorization metadata, not decoder results")
    if _sha_file(factor_rows_path) != plan.get("rows_raw_sha256"):
        raise ConsumerError("factorization row plan hash mismatch")

    mel_runner_record = plan.get("mel_runner")
    if not isinstance(mel_runner_record, Mapping):
        raise ConsumerError("RMS plan lacks bound mel runner record")
    _require_sha(str(mel_runner_record["path"]), str(mel_runner_record["sha256"]), "bound mel utility source")
    source_run = plan.get("mel_source_run")
    if not isinstance(source_run, Mapping):
        raise ConsumerError("RMS plan lacks bound mel source run")
    mel_plan_path = _require_sha(source_run["plan_path"], source_run["plan_raw_sha256"], "mel plan")
    mel_manifest_path = _require_sha(source_run["run_manifest_path"], source_run["run_manifest_raw_sha256"], "mel run manifest")
    mel_results_path = _require_sha(source_run["results_path"], source_run["results_raw_sha256"], "mel result rows")
    mel_plan = _read_json(mel_plan_path)
    mel_run = _read_json(mel_manifest_path)
    if mel_plan.get("payload_sha256") != _payload_sha(mel_plan) or mel_run.get("payload_sha256") != _payload_sha(mel_run):
        raise ConsumerError("bound mel source payload fingerprint mismatch")
    if mel_plan.get("code", {}).get("sha256") != mel_runner_record.get("sha256") or mel_plan.get("spec", {}).get("sha256") != MEL_SPEC_SHA256:
        raise ConsumerError("bound mel plan code/spec does not match the execution binding")
    _require_sha(mel_plan["spec"]["path"], MEL_SPEC_SHA256, "frozen mel partition spec")

    cal_qa = plan.get("calibration_geometry_qa")
    if not isinstance(cal_qa, Mapping):
        raise ConsumerError("RMS plan lacks CAL geometry QA binding")
    qa_path = _require_sha(cal_qa["path"], cal_qa["raw_sha256"], "RMS CAL QA")
    qa = _read_json(qa_path)
    if qa.get("payload_sha256") != _payload_sha(qa) or qa.get("status") != "PASS" or qa.get("model_forward_performed") is not False or qa.get("effects_read") is not False:
        raise ConsumerError("RMS CAL QA payload/state is invalid")
    if qa.get("producer_code", {}).get("sha256") != _sha_file(producer_path) or qa.get("spec", {}).get("sha256") != RMS_SPEC_SHA256 or qa.get("calibration_clip_count") != 20:
        raise ConsumerError("RMS CAL QA producer/spec/cohort binding mismatch")

    artifact_root = Path(str(run.get("artifact_root", ""))).resolve(strict=True)
    ledger_check = _verify_file_ledger(run, artifact_root)
    factor_samples = _read_jsonl(factor_rows_path)
    result_rows = _read_jsonl(rows_path)
    samples = plan.get("planned_samples")
    if not isinstance(samples, list) or len(samples) != 80 or len(factor_samples) != 80 or len(result_rows) != 80:
        raise ConsumerError("producer did not retain all 80 intended rows in plan, factors, and results")
    if len({str(row.get("id")) for row in samples}) != 80 or len({str(row.get("id")) for row in factor_samples}) != 80 or len({str(row.get("id")) for row in result_rows}) != 80:
        raise ConsumerError("duplicate/missing IDs among intended RMS rows")
    if [str(row.get("id")) for row in samples] != [str(row.get("id")) for row in factor_samples] or [str(row.get("id")) for row in samples] != [str(row.get("id")) for row in result_rows]:
        raise ConsumerError("RMS row ordering differs across source plan, factor plan and results")
    factor_by_id = {str(row["id"]): row for row in factor_samples}
    results_by_id = {str(row["id"]): row for row in result_rows}

    consumer_rows: list[dict[str, Any]] = []
    permutation_hashes: dict[str, str] = {}
    for ordered_index, sample in enumerate(samples):
        sample_id = str(sample["id"])
        factor_row = factor_by_id[sample_id]
        result = results_by_id[sample_id]
        if int(sample.get("ordered_index", ordered_index)) != ordered_index:
            raise ConsumerError(f"unexpected ordered sample index: {sample_id}")
        if factor_row.get("speaker") != sample.get("speaker") or result.get("speaker") != sample.get("speaker"):
            raise ConsumerError(f"speaker identity differs across RMS row records: {sample_id}")
        if factor_row.get("status") != "prepared" or result.get("status") != "complete":
            consumer_rows.append({"id": sample_id, "speaker": str(sample["speaker"]), "ordered_index": ordered_index,
                                  "status": "planned_failure", "factor_status": factor_row.get("status"), "factor_failure": factor_row.get("failure"),
                                  "run_status": result.get("status"), "run_failure": result.get("failure"),
                                  "retained_in_intended_denominator": True})
            continue
        consumer_row, _, permutation_sha = _distance_and_permutation_rows(factor_row, result, artifact_root, ordered_index)
        consumer_rows.append(consumer_row)
        permutation_hashes[sample_id] = permutation_sha

    consumer_by_id = {str(row["id"]): row for row in consumer_rows}
    group_summary = _group_summary(samples, consumer_by_id)
    report: dict[str, Any] = {
        "schema_version": 1,
        "consumer_id": "independent-rms-latent-factorization-consumer-20261001",
        "status": "PASS" if len(consumer_rows) == 80 else "INCOMPLETE",
        "created_at_utc": _utc_now(),
        "producer_source": {"path": str(producer_path), "sha256": _sha_file(producer_path)},
        "consumer_source": {"path": str(Path(__file__).resolve()), "sha256": _sha_file(Path(__file__))},
        "spec": {"path": str(spec_path), "sha256": _sha_file(spec_path)},
        "producer_bindings": {
            "factorization_plan_path": str(plan_path), "factorization_plan_sha256": _sha_file(plan_path), "factorization_plan_payload_sha256": plan.get("payload_sha256"),
            "run_manifest_path": str(run_path), "run_manifest_sha256": _sha_file(run_path), "run_manifest_payload_sha256": run.get("payload_sha256"),
            "factor_rows_path": str(factor_rows_path), "factor_rows_sha256": _sha_file(factor_rows_path),
            "evaluation_results_path": str(rows_path), "evaluation_results_sha256": _sha_file(rows_path),
            "mel_plan_path": str(mel_plan_path), "mel_plan_sha256": _sha_file(mel_plan_path),
            "mel_run_manifest_path": str(mel_manifest_path), "mel_run_manifest_sha256": _sha_file(mel_manifest_path),
            "mel_results_path": str(mel_results_path), "mel_results_sha256": _sha_file(mel_results_path),
            "rms_calibration_qa_path": str(qa_path), "rms_calibration_qa_sha256": _sha_file(qa_path),
            "artifact_ledger": ledger_check,
            "input_identity_sha256": plan.get("input_identity_sha256"),
            "execution_assets_raw_sha256": plan.get("execution_assets_raw_sha256"),
        },
        "reconstruction_protocol": {
            "producer_scoring_imported": False,
            "producer_ops_imported": False,
            "mel_frontend_recomputed": False,
            "audio_encoder_recomputed": False,
            "model_forward_performed_by_consumer": False,
            "effects_were_used_for_row_selection": False,
            "latent_radius_error_limit": RADIUS_ERROR_LIMIT,
            "distance_reconstruction_tolerance": {"rtol": 2e-5, "atol": 2e-5},
            "score_constants": {"vshift": VSHIFT, "zero_column": ZERO_COLUMN, "anchor_column": ANCHOR_COLUMN, "anchor_lag": 3, "embedding_dim": EMBEDDING_DIM, "pairwise_eps": float(EPS)},
            "permutation_null": {"rng": "PCG64", "base_seed": PERMUTATION_SEED, "per_clip_seed": "base_seed + ordered_eval_index", "permutations_per_clip": PERMUTATION_COUNT, "same_full_A_row_permutation_shared_by_all_four_video_cells": True},
            "bootstrap": {"rng": "PCG64", "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "percentile": [0.005, 0.995], "unit": "speaker means after two-clip average", "incomplete_speakers_retained_and_reported": True},
            "contrasts": CONTRASTS,
        },
        "intended_rows": 80,
        "processed_complete_rows": sum(row.get("status") == "complete" for row in consumer_rows),
        "planned_failures_retained": sum(row.get("status") != "complete" for row in consumer_rows),
        "intended_speaker_count": len({str(sample["speaker"]) for sample in samples}),
        "permutation_indices_array_sha256_by_id": permutation_hashes,
        "rows": consumer_rows,
        "equal_speaker_contrasts": group_summary,
        "effects_read": True,
        "consumer_selected_rows_by_effect": False,
    }
    report["payload_sha256"] = _payload_sha(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(output)
    print(json.dumps({"status": report["status"], "rows": report["intended_rows"], "complete_rows": report["processed_complete_rows"],
                      "planned_failures": report["planned_failures_retained"], "report": str(output), "report_sha256": _sha_file(output),
                      "model_forward_performed": False}, ensure_ascii=False, sort_keys=True), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--producer-code", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.set_defaults(func=consume)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:2000]}, ensure_ascii=False, sort_keys=True), file=__import__("sys").stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
