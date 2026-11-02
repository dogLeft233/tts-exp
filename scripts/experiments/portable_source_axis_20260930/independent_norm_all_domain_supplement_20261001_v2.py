#!/usr/bin/env python3
"""Independent NumPy consumer for the sealed latent angle/radius run.

Reads immutable four-cell artifacts, independently rebuilds the float32
distance and lag matrices, audits the per-frame radius factorization, and
computes frozen speaker-level summaries. It imports no producer, SyncNet,
Wav2Lip, or shared scoring code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

CONDITIONS = ("q00", "q10", "q01", "q11")
GEOMETRIES = ("raw", "unit")
SUPPORTS = ("full", "valid", "guard20", "guard15")
METRICS = ("C", "C_anchor", "B", "D", "search_gain")
CONTRASTS = ("q10_minus_q00", "q10_minus_q01", "q01_minus_q00", "q11_minus_q00", "interaction")
VISUAL_IDS = {"q00": "latent_radius_q00", "q10": "latent_radius_q10", "q01": "latent_radius_q01", "q11": "latent_radius_q11"}
LAGS = np.arange(-15, 16, dtype=np.int32)
ANCHOR_K, ANCHOR_COLUMN, OFFICIAL_OFFSET = 3, 18, -3
SPEC_SHA = "631e76042a523f598e9135de7ff9bf328f3656eabc800a0aa85f38f66bc01252"
GEOMETRY_SHA = "800d0a0d040df2337249958c907b28eb13ef75173392778ce7c674eed1fce17e"
SOURCE_PROTOCOL_SHA = "8ba27fcb395dccbb493ad4831e8b68ac18c485b61e39d6e3b7b3018fd668cf39"
SOURCE_ROWS_SHA = "19f805f6935c85475ad28319e17d2694a333a85d055d3e19a0eb93bda7390dab"
INPUT_MANIFEST_SHA = "6b0ce3252cf776a86cf84b19ab53afd936526216e4ddb30dc587a72a66409b4b"
RECOVERY_SEAL_SHA = "55c1512e35bf72563d0e01623f7e14f77aaa776dd94ee565926d9df9bea90417"
CONTINUATION_PARENT_SHA = "d5fbfa99865b958a006f504e0dd7c6e5120ec194651aa639acad4d540528b862"
NORM_PROTOCOL_SHA = "f1c082f7c255345ce027f231275dcf2917b4c390895517a8ed8aeca8d65c6562"
NORM_ROWS_SHA = "36727fde4ffb652ac592dd37a11a8a112cfb430c37cb7b4c96b3c28fd57571a3"
NORM_RESULTS_SHA = "5c389dfc421080b00e478b455bb5ec919b9ae7b5e22748835a3df84945d7a024"
NORM_STATUS_SHA = "28543b6a0cefd5a065cf05d368c912d9e61f2be43ab15bd0d444f92a7b2bec0f"
NORM_ALL_FILES_SHA = "ba11e903d18868781f562554b867d5fe062f58f3b65b1b9eab414518c2bcb50a"
NORM_CODE_SEAL_SHA = "a0c066f43e0f4e0223698d4d561e7b35a7c0efae75f16c5f92126c794b86458a"
EXPECTED_CODE = {
    "latent_angle_radius_attempt06.py": "23ee82bc421ad386765454e36222f78edffb8ace47923d1c94f9a7159ede3ecc",
    "latent_angle_radius.py": "cdc2f922903cfbb35cc9e8d57e24a3963c590a8d8d72b99f93a9cf476406ceb0",
    "evaluate.py": "e50a31eaa2493b83e87cb4426b742e8f28e73836cbb9374b2319e7b60c940880",
    "ops.py": "64eecefea72a494cc07573c8bbdd872351e23b9f7998e52b280483af3b9c0a3b",
    "scoring.py": "9411436d9fc199e91116a44523eeeccb66be0e59f7f20f491b0b9c8cc5a3ad87",
    "wav2lip.py": "f0a6cd180679aa1bb218df845632211731651e122b5baa706a38a3db32ccfe9d",
}
BOOTSTRAP_SEED, BOOTSTRAP_DRAWS = 20260930, 20_000
NULL_DRAWS = 100
EPS = np.float32(1e-6)
DISTANCE_TOL = 5e-4
METRIC_TOL = 5e-4


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Match the documented byte hash for a contiguous array, independently."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def safe_path(root: Path, ref: str) -> Path:
    p = PurePosixPath(str(ref))
    if p.is_absolute() or not p.parts or any(part in ("", ".", "..") for part in p.parts):
        raise AuditError(f"unsafe run-relative artifact path: {ref}")
    result = root.joinpath(*p.parts).resolve(strict=True)
    try:
        result.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise AuditError(f"artifact escapes run root: {ref}") from exc
    return result


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise AuditError(f"expected JSON object: {path}")
    return obj


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise AuditError(f"non-object JSONL at {path}:{line_no}")
                rows.append(item)
    return rows


def npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def file_manifest_check(root: Path, manifest_path: Path) -> tuple[dict[str, str], int]:
    manifest = read_json(manifest_path)
    if not manifest or not all(isinstance(k, str) and isinstance(v, str) for k, v in manifest.items()):
        raise AuditError("all_files_sha256.json must be a nonempty relative-path map")
    checked = 0
    for rel, digest in manifest.items():
        path = safe_path(root, rel)
        actual = sha256_file(path)
        if actual != digest:
            raise AuditError(f"all-files digest mismatch for {rel}: {actual} != {digest}")
        checked += 1
    found = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != manifest_path}
    if found != set(manifest):
        missing, extra = sorted(set(manifest) - found), sorted(found - set(manifest))
        raise AuditError(f"all-files inventory membership mismatch: missing={missing[:3]}, extra={extra[:3]}")
    return manifest, checked


def expected_supports(n: int) -> dict[str, np.ndarray]:
    full = np.arange(n, dtype=np.int32)
    return {
        "full": full,
        "valid": full.copy(),
        "guard20": np.arange(min(20, n), max(min(20, n), n - 20), dtype=np.int32),
        "guard15": np.arange(min(15, n), max(min(15, n), n - 15), dtype=np.int32),
    }


def row_unit(x: np.ndarray) -> np.ndarray:
    value = np.asarray(x, dtype=np.float32)
    norm = np.sqrt(np.sum(value * value, axis=1, keepdims=True, dtype=np.float32), dtype=np.float32)
    return (value / np.maximum(norm, np.float32(1e-12))).astype(np.float32, copy=False)


def cross_and_zero(v: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v, a = np.asarray(v, dtype=np.float32), np.asarray(a, dtype=np.float32)
    if v.ndim != 2 or a.ndim != 2 or v.shape[1] != a.shape[1] or not len(v) or not len(a):
        raise AuditError("embedding shapes do not form nonempty V×A geometry")
    cross = np.empty((len(v), len(a)), dtype=np.float32)
    for start in range(0, len(v), 4):
        stop = min(start + 4, len(v))
        delta = v[start:stop, None, :] - a[None, :, :]
        np.add(delta, EPS, out=delta)
        np.sqrt(np.sum(delta * delta, axis=2, dtype=np.float32), out=cross[start:stop])
    zero_v = v.copy()
    np.add(zero_v, EPS, out=zero_v)
    zero = np.sqrt(np.sum(zero_v * zero_v, axis=1, dtype=np.float32), dtype=np.float32)
    return cross, zero.astype(np.float32, copy=False)


def lag_matrix(cross: np.ndarray, zero: np.ndarray, query_n: int) -> np.ndarray:
    x, z = np.asarray(cross, dtype=np.float32), np.asarray(zero, dtype=np.float32)
    if x.ndim != 2 or len(z) != x.shape[0] or query_n < 1 or x.shape[0] < query_n:
        raise AuditError("malformed full cross or zero boundary arrays")
    out = np.empty((query_n, 31), dtype=np.float32)
    for q in range(query_n):
        for col, lag in enumerate(LAGS):
            j = q + int(lag)
            out[q, col] = x[q, j] if 0 <= j < x.shape[1] else z[q]
    return out


def metric_payload(matrix: np.ndarray, supports: dict[str, np.ndarray]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, indices in supports.items():
        if not len(indices):
            out[name] = {"status": "too_short", "support_n": 0}
            continue
        curve = np.asarray(matrix[indices], dtype=np.float32).mean(axis=0, dtype=np.float32).astype(np.float64)
        minimum, median = float(curve.min()), float(np.median(curve))
        out[name] = {
            "status": "ok", "support_n": int(len(indices)), "window_indices": indices.tolist(), "curve": curve.tolist(),
            "C": median - minimum, "C_anchor": median - float(curve[ANCHOR_COLUMN]), "B": median, "D": minimum,
            "search_gain": float(curve[15]) - minimum, "D0": float(curve[15]),
        }
    return out


def value_summary(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "max_abs": float(np.max(np.abs(x)))}


def compare_float32(actual: np.ndarray, expected: np.ndarray, label: str, tol: float = DISTANCE_TOL) -> float:
    a = np.asarray(actual)
    if a.dtype != np.float32 or a.shape != expected.shape or not np.isfinite(a).all():
        raise AuditError(f"{label} has invalid dtype/shape or nonfinite values")
    diff = float(np.max(np.abs(a - expected), initial=0.0))
    if diff > tol:
        raise AuditError(f"{label} differs by {diff:.8g} > {tol:.8g}")
    return diff


def compare_metrics(saved: dict[str, Any], rebuilt: dict[str, Any], label: str) -> float:
    biggest = 0.0
    for support in SUPPORTS:
        s, r = saved.get(support), rebuilt.get(support)
        if not isinstance(s, dict) or not isinstance(r, dict) or s.get("status") != r.get("status"):
            raise AuditError(f"{label} status missing/different for {support}")
        if int(s.get("support_n", -1)) != int(r.get("support_n", -2)):
            raise AuditError(f"{label} support count differs for {support}")
        if r["status"] == "too_short":
            continue
        if s.get("window_indices") != r.get("window_indices"):
            raise AuditError(f"{label} support indices differ for {support}")
        aliases = {"C": "C_best", "C_anchor": "C_fixed_anchor", "B": "B_median", "D": "D_min"}
        for key in ("C", "C_anchor", "B", "D"):
            delta = abs(float(s[aliases[key]]) - float(r[key]))
            biggest = max(biggest, delta)
            if delta > METRIC_TOL:
                raise AuditError(f"{label} {support}.{key} differs by {delta:.8g}")
        curve_diff = float(np.max(np.abs(np.asarray(s["curve"], dtype=np.float64) - np.asarray(r["curve"], dtype=np.float64)), initial=0.0))
        biggest = max(biggest, curve_diff)
        if curve_diff > METRIC_TOL:
            raise AuditError(f"{label} {support} curve differs by {curve_diff:.8g}")
    return biggest


def bootstrap_speakers(by_speaker: dict[str, list[float]], intended_speakers: int = 40) -> dict[str, Any]:
    speakers = sorted(s for s, vals in by_speaker.items() if len(vals) == 2)
    values = np.asarray([np.mean(by_speaker[s], dtype=np.float64) for s in speakers], dtype=np.float64)
    ci = None
    if len(speakers) == intended_speakers:
        rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
        draws = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)), dtype=np.int64)
        means = values[draws].mean(axis=1, dtype=np.float64)
        ci = [float(x) for x in np.quantile(means, [.005, .995], method="linear")]
    return {
        "equal_speaker_mean": float(values.mean(dtype=np.float64)) if len(values) else None,
        "speaker_count_complete_two_clip": len(speakers), "intended_speaker_count": intended_speakers,
        "expected_clip_count": intended_speakers * 2,
        "complete_clip_count": sum(len(v) for v in by_speaker.values()),
        "speaker_bootstrap_99_ci": ci, "bootstrap_draws": BOOTSTRAP_DRAWS if ci else None,
        "bootstrap_seed": BOOTSTRAP_SEED if ci else None,
        "eligible_for_intended_cohort": len(speakers) == intended_speakers and all(len(v) == 2 for v in by_speaker.values()),
    }


def factorization_audit(context: dict[str, np.ndarray]) -> dict[str, Any]:
    z0 = np.asarray(context["z0_saved_float32"])
    z1 = np.asarray(context["z1_saved_float32"])
    q10 = np.asarray(context["z_q10_float32"])
    q01 = np.asarray(context["z_q01_float32"])
    mask = np.asarray(context["natural_speech_mask_fixed_uint8"])
    if any(x.dtype != np.float32 for x in (z0, z1, q10, q01)) or z0.shape != z1.shape or z0.shape != q10.shape or z0.shape != q01.shape or z0.ndim < 2:
        raise AuditError("four latent arrays must share a frame×feature float32 shape")
    if mask.dtype != np.uint8 or mask.shape != (len(z0),) or not np.isin(mask, [0, 1]).all():
        raise AuditError("saved original fixed-N mask has wrong dtype/shape/value")
    if not all(np.isfinite(x).all() for x in (z0, z1, q10, q01)) or any(np.any(x < 0) for x in (z0, z1, q10, q01)):
        raise AuditError("latent arrays must be finite and nonnegative")
    if not np.array_equal(context["z0_saved_float32"], z0) or not np.array_equal(context["z1_saved_float32"], z1):
        raise AuditError("q00/q11 saved endpoint arrays are not stable float32 copies")
    inactive = mask == 0
    if np.any(inactive) and not np.array_equal(z0[inactive], z1[inactive]):
        raise AuditError("original source z0/z1 differ on inactive rows under the fixed-N mask")
    r0 = np.linalg.norm(z0.reshape(len(z0), -1).astype(np.float64), axis=1)
    r1 = np.linalg.norm(z1.reshape(len(z1), -1).astype(np.float64), axis=1)
    if np.any(r0 <= 0) or np.any(r1 <= 0):
        raise AuditError("zero radius in planned latent factorization")
    scale10, scale01 = r0 / r1, r1 / r0
    shape = (len(r0),) + (1,) * (z0.ndim - 1)
    expected10 = (z1.astype(np.float64) * scale10.reshape(shape)).astype(np.float32)
    expected01 = (z0.astype(np.float64) * scale01.reshape(shape)).astype(np.float32)
    if not np.array_equal(q10, expected10) or not np.array_equal(q01, expected01):
        raise AuditError("stored q10/q01 do not equal independently recomputed all-frame factorization")
    r10 = np.linalg.norm(q10.reshape(len(q10), -1).astype(np.float64), axis=1)
    r01 = np.linalg.norm(q01.reshape(len(q01), -1).astype(np.float64), axis=1)
    e10, e01 = np.abs(r10-r0)/r0, np.abs(r01-r1)/r1
    if float(e10.max(initial=0)) > 1e-6 or float(e01.max(initial=0)) > 1e-6:
        raise AuditError("stored float32 radius factorization exceeds frozen 1e-6 per-frame tolerance")
    u = np.asarray(context["calibration_u_level_float64"])
    if u.dtype != np.float64 or u.size != int(np.prod(z0.shape[1:])) or not np.isfinite(u).all():
        raise AuditError("saved calibration level axis is malformed")
    u = u.reshape(-1)
    z0f, z1f = z0.reshape(len(z0), -1).astype(np.float64), z1.reshape(len(z1), -1).astype(np.float64)
    q10f, q01f = q10.reshape(len(q10), -1).astype(np.float64), q01.reshape(len(q01), -1).astype(np.float64)
    q10pre, q01pre = z1f * scale10[:, None], z0f * scale01[:, None]
    active = mask.astype(bool)
    def dose(name: str, value: np.ndarray, pre: np.ndarray | None = None) -> dict[str, Any]:
        post_delta = value - z0f
        post_norm = np.linalg.norm(post_delta, axis=1)
        payload = {
            "delta_norm_all_frames": value_summary(post_norm),
            "delta_norm_active_masked": value_summary(post_norm[active]) if active.any() else None,
            "u_projection_post_float32_all_frames": value_summary(post_delta @ u),
            "u_projection_post_float32_active_masked": value_summary((post_delta @ u)[active]) if active.any() else None,
        }
        if pre is not None:
            pre_delta = pre - z0f
            pre_up = pre_delta @ u
            payload["u_projection_pre_float32_all_frames"] = value_summary(pre_up)
            payload["u_projection_pre_float32_active_masked"] = value_summary(pre_up[active]) if active.any() else None
        return payload
    cos01 = np.sum(z0f*z1f, axis=1)/(r0*r1)
    return {
        "frame_count": len(z0), "active_frames": int(mask.sum()), "inactive_frames": int(len(mask)-mask.sum()),
        "q00_q11_endpoint_bitexact": True, "original_z0_z1_inactive_bitexact": True,
        "q10_q01_all_frame_factorization_exact": True,
        "max_relative_radius_error": {"q10_to_r0": float(e10.max(initial=0)), "q01_to_r1": float(e01.max(initial=0))},
        "mean_r0": float(r0.mean()), "mean_r1": float(r1.mean()),
        "mean_ratio_r1_r0": float(np.mean(r1/r0)), "mean_cosine_z0_z1": float(np.mean(cos01)),
        "dose_geometry": {"q00": dose("q00", z0f), "q10": dose("q10", q10f, q10pre), "q01": dose("q01", q01f, q01pre), "q11": dose("q11", z1f)},
        "inactive_identity_applies_only_to_original_z0_z1": True,
    }


def null_metrics_for_support(
    cross: np.ndarray, zero: np.ndarray, queries: np.ndarray,
    permutations: list[np.ndarray],
) -> dict[str, np.ndarray] | None:
    """Compute all frozen lag metrics for each full-A permutation.

    Boundary lag cells outside the true A domain retain the scorer's saved
    distance-to-zero value. Guarded domains naturally have no such cells.
    """
    if len(queries) == 0:
        return None
    x = np.asarray(cross, dtype=np.float32)
    boundary = np.asarray(zero, dtype=np.float32)
    q = np.asarray(queries, dtype=np.int64)
    if x.ndim != 2 or boundary.shape != (x.shape[0],) or q.ndim != 1 or not len(x):
        raise AuditError("malformed temporal-null cross/zero/query arrays")
    if np.any(q < 0) or np.any(q >= x.shape[0]):
        raise AuditError("temporal-null query lies outside the V domain")
    source = q[:, None] + LAGS.astype(np.int64)[None, :]
    in_bounds = (source >= 0) & (source < x.shape[1])
    clipped = np.clip(source, 0, x.shape[1] - 1)
    row_index = np.broadcast_to(q[None, :, None], (len(permutations), len(q), len(LAGS)))
    if any(
        np.asarray(pi).shape != (x.shape[1],)
        or not np.array_equal(np.sort(np.asarray(pi, dtype=np.int64)), np.arange(x.shape[1], dtype=np.int64))
        for pi in permutations
    ):
        raise AuditError("temporal-null permutations do not match the complete audio row domain")
    permuted_source = np.stack([np.asarray(pi, dtype=np.int64)[clipped] for pi in permutations], axis=0)
    values = x[row_index, permuted_source]
    values = np.where(in_bounds[None, :, :], values, boundary[q][None, :, None]).astype(np.float32, copy=False)
    curves = values.mean(axis=1, dtype=np.float32).astype(np.float64)
    medians = np.median(curves, axis=1)
    minima = np.min(curves, axis=1)
    return {
        "C": medians - minima,
        "C_anchor": medians - curves[:, ANCHOR_COLUMN],
        "B": medians,
        "D": minima,
        "search_gain": curves[:, 15] - minima,
    }


def equal_speaker_mean(values_by_sample: dict[str, float], speakers: dict[str, str], expected_samples: list[str]) -> tuple[float | None, int, int]:
    by_sp: dict[str, list[float]] = defaultdict(list)
    for sample in expected_samples:
        if sample in values_by_sample:
            by_sp[speakers[sample]].append(float(values_by_sample[sample]))
    complete = {sp: float(np.mean(v, dtype=np.float64)) for sp, v in by_sp.items() if len(v) == 2}
    return (float(np.mean(list(complete.values()), dtype=np.float64)) if complete else None, len(complete), sum(len(v) for v in by_sp.values()))


def summarize_null(observed: dict[str, float], null: dict[str, np.ndarray | None], speakers: dict[str, str], samples: list[str], support: str) -> dict[str, Any]:
    null_mean = {s: float(np.mean(v, dtype=np.float64)) for s, v in null.items() if v is not None}
    paired = {s: float(observed[s] - null_mean[s]) for s in observed.keys() & null_mean.keys()}
    obs_mean, obs_sp, obs_clip = equal_speaker_mean(observed, speakers, samples)
    null_mean_group, null_sp, null_clip = equal_speaker_mean(null_mean, speakers, samples)
    paired_boot = defaultdict(list)
    for s, value in paired.items():
        paired_boot[speakers[s]].append(value)
    paired_summary = bootstrap_speakers(paired_boot)
    indices = list(range(NULL_DRAWS)) if any(v is not None for v in null.values()) else []
    perm_group: list[float] = []
    perm_speaker_counts: list[int] = []
    for i in indices:
        per_sample = {s: float(null[s][i]) for s in samples if null.get(s) is not None and len(null[s]) == NULL_DRAWS}
        mean, nsp, _ = equal_speaker_mean(per_sample, speakers, samples)
        if mean is not None:
            perm_group.append(mean)
            perm_speaker_counts.append(nsp)
    perm_ci = [float(x) for x in np.quantile(perm_group, [.005, .995], method="linear")] if perm_group else None
    p_upper = (1 + sum(x >= obs_mean for x in perm_group)) / (len(perm_group) + 1) if perm_group and obs_mean is not None else None
    return {
        "observed_equal_speaker_mean": obs_mean, "observed_complete_speakers": obs_sp, "observed_complete_clips": obs_clip,
        "mean_of_clip_level_null_means_equal_speaker": null_mean_group, "null_complete_speakers": null_sp, "null_complete_clips": null_clip,
        "paired_observed_minus_null_mean": paired_summary,
        "null_equal_speaker_means": perm_group, "null_speaker_count_each_draw": perm_speaker_counts, "null_percentile_99": perm_ci,
        "empirical_upper_tail_p_observed_ge_null": p_upper, "permutation_count": len(perm_group),
        "eligible_for_intended_cohort": obs_sp == 40 and null_sp == 40 and len(perm_group) == NULL_DRAWS and all(n == 40 for n in perm_speaker_counts),
        "support": f"{support}; original 100 full-A row permutations shared across all four cells for each recipient",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    root = args.run_root.resolve(strict=True)
    out = args.outdir.resolve()
    if out.exists() and any(out.iterdir()):
        raise AuditError(f"output directory must be empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    protocol_path, rows_path, results_path = (root / n for n in ("evaluation_protocol.json", "evaluation_rows.jsonl", "evaluation_results.jsonl"))
    status_path, seal_path, all_path = (root / n for n in ("run_status.json", "code_seal.json", "all_files_sha256.json"))
    protocol = read_json(protocol_path)
    plan, results, status, code_seal = read_jsonl(rows_path), read_jsonl(results_path), read_json(status_path), read_json(seal_path)
    exact_protocol_checks = (
        ("protocol", protocol_path, NORM_PROTOCOL_SHA), ("row plan", rows_path, NORM_ROWS_SHA),
        ("results", results_path, NORM_RESULTS_SHA), ("run status", status_path, NORM_STATUS_SHA),
        ("all-files manifest", all_path, NORM_ALL_FILES_SHA), ("code seal", seal_path, NORM_CODE_SEAL_SHA),
    )
    for label, path, expected in exact_protocol_checks:
        if sha256_file(path) != expected:
            raise AuditError(f"sealed {label} SHA mismatch")
    for label, observed, expected in (("frozen spec", protocol.get("spec", {}).get("sha256"), SPEC_SHA), ("CAL geometry", protocol.get("calibration_geometry_audit", {}).get("sha256"), GEOMETRY_SHA)):
        if observed != expected:
            raise AuditError(f"{label} seal mismatch: {observed}")
    if protocol.get("experiment_id") != "confirm-latent-angle-radius-20261001" or protocol.get("planned_samples") != 80 or protocol.get("planned_rows") != 320:
        raise AuditError("protocol does not bind the frozen 80×4 cohort")
    source = protocol.get("source_retry3", {})
    if source.get("protocol_sha256") != SOURCE_PROTOCOL_SHA or source.get("row_manifest_sha256") != SOURCE_ROWS_SHA:
        raise AuditError("retry3 source protocol/row plan differs from the frozen parent")
    if protocol.get("input_manifest", {}).get("sha256") != INPUT_MANIFEST_SHA or protocol.get("calibration_anchor") != {"k": 3, "column": 18, "official_syncnet_offset": -3}:
        raise AuditError("input cohort or k=3/column18 anchor binding differs from frozen spec")
    if protocol.get("score_audio") != "unchanged original RAW natural PCM for all four cells; one new A cache per sample" or protocol.get("runtime_policy", {}).get("all_conditions_freshly_rendered") is not True:
        raise AuditError("protocol does not bind fixed RAW-N scoring audio and fresh four-cell rendering")
    if code_seal.get("protocol_sha256") != sha256_file(protocol_path) or code_seal.get("code_sha256") != protocol.get("code_sha256"):
        raise AuditError("code_seal does not bind the protocol/code inventory")
    for name, expected in EXPECTED_CODE.items():
        if protocol.get("code_sha256", {}).get(name) != expected:
            raise AuditError(f"producer code seal mismatch for {name}")
    if status.get("status") != "complete" or status.get("planned_samples") != 80 or status.get("completed_samples") != 80 or status.get("failed_samples") != 0 or status.get("planned_rows") != 320 or status.get("completed_rows") != 320 or status.get("failed_rows") != 0:
        raise AuditError("producer status is not a complete 80-sample/320-row run")
    expected_flags = {"cudnn_benchmark": False, "cudnn_deterministic": True, "deterministic_algorithms": True}
    if status.get("face_flags", {}).get("after_override") != expected_flags:
        raise AuditError("post-detection deterministic flags are not recorded as false/true/true")
    events = status.get("forward_events", [])
    if not events or any(e.get("flags_at_call") != expected_flags for e in events):
        raise AuditError("one or more actual model/score forwards used unexpected deterministic flags")
    expected_flags_json = json.dumps(expected_flags, sort_keys=True)
    if status.get("forward_flag_values") != [expected_flags_json]:
        raise AuditError("run-level forward flag summary does not match per-call flags")
    if len(plan) != 320 or len(results) != 320:
        raise AuditError(f"expected 320 planned/result rows, got {len(plan)}/{len(results)}")
    all_hashes, inventory_n = file_manifest_check(root, all_path)
    for name, path in (("protocol", protocol_path), ("row plan", rows_path), ("results", results_path), ("run status", status_path), ("code seal", seal_path)):
        rel = path.relative_to(root).as_posix()
        if all_hashes.get(rel) != sha256_file(path):
            raise AuditError(f"sealed file inventory misses/mismatches {name}")
    expected_keys = {(str(r["sample_id"]), str(r["condition"])) for r in plan}
    if len(expected_keys) != 320 or {(str(r["sample_id"]), str(r["condition"])) for r in results} != expected_keys:
        raise AuditError("plan/results do not have unique matched sample×condition rows")
    row_by_id = {str(r["row_id"]): r for r in plan}
    result_by_id = {str(r["row_id"]): r for r in results}
    samples = list(dict.fromkeys(str(r["sample_id"]) for r in plan if r.get("condition") == "q00"))
    if len(samples) != 80 or len(row_by_id) != 320 or len(result_by_id) != 320:
        raise AuditError("row plan has duplicate IDs or incomplete 80-sample support")
    speaker_by_sample = {str(r["sample_id"]): str(r["speaker_id"]) for r in plan if r.get("condition") == "q00"}
    counts = Counter(speaker_by_sample.values())
    if len(counts) != 40 or set(counts.values()) != {2}:
        raise AuditError("intended cohort is not 40 speakers×2 clips")
    source_root = Path(str(source.get("run_dir", ""))).resolve(strict=True)
    if sha256_file(source_root / "evaluation_protocol.json") != SOURCE_PROTOCOL_SHA or sha256_file(source_root / "evaluation_rows.jsonl") != SOURCE_ROWS_SHA:
        raise AuditError("retry3 source protocol/rows hash mismatch")
    source_protocol = read_json(source_root / "evaluation_protocol.json")
    source_plan = read_jsonl(source_root / "evaluation_rows.jsonl")
    source_results = read_jsonl(source_root / "evaluation_results.jsonl")
    if sha256_file(source_root / "evaluation_results.jsonl") != source.get("evaluation_results_sha256"):
        raise AuditError("retry3 result seal differs from norm protocol source binding")
    calibration_axes_ref = source_protocol.get("calibration_axes", {})
    calibration_summary_ref = source_protocol.get("calibration_summary", {})
    if calibration_axes_ref.get("sha256") != source.get("calibration_axes_sha256") or calibration_summary_ref.get("sha256") != source.get("calibration_summary_sha256"):
        raise AuditError("retry3 protocol CAL artifact refs differ from norm protocol bindings")
    calibration_axes_path = Path(str(calibration_axes_ref.get("path", ""))).resolve(strict=True)
    calibration_summary_path = Path(str(calibration_summary_ref.get("path", ""))).resolve(strict=True)
    if sha256_file(calibration_axes_path) != source.get("calibration_axes_sha256") or sha256_file(calibration_summary_path) != source.get("calibration_summary_sha256"):
        raise AuditError("sealed CAL axes/summary file SHA mismatch")
    calibration_axes = npz(calibration_axes_path)
    if set(calibration_axes) != {"d_source", "u_level"}:
        raise AuditError("CAL axes archive must contain only d_source and u_level")
    d_source = np.asarray(calibration_axes["d_source"])
    u_level = np.asarray(calibration_axes["u_level"])
    if d_source.dtype != np.float64 or u_level.dtype != np.float64 or d_source.shape != (512,) or u_level.shape != (512,) or not np.isfinite(d_source).all() or not np.isfinite(u_level).all():
        raise AuditError("CAL d/u axes have unexpected dtype, shape, or finite values")
    if array_sha256(d_source) != calibration_axes_ref.get("d_sha256") or array_sha256(u_level) != calibration_axes_ref.get("u_sha256") or array_sha256(u_level) != source.get("u_level_array_sha256"):
        raise AuditError("CAL d/u arrays do not match protocol-bound content hashes")
    calibration_summary = read_json(calibration_summary_path)
    if calibration_summary.get("global_anchor", {}).get("selected_k") != ANCHOR_K or calibration_summary.get("global_anchor", {}).get("anchor_column") != ANCHOR_COLUMN or calibration_summary.get("global_anchor", {}).get("official_syncnet_offset") != OFFICIAL_OFFSET:
        raise AuditError("CAL summary does not independently bind the k=3/column18/offset=-3 anchor")
    source_plan_by_sample_cond = {(str(r["sample_id"]), str(r["condition"])): r for r in source_plan if r.get("family") == "source_axis" and r.get("source") == "natural" and r.get("face_id") == "primary" and r.get("condition") in ("baseline", "orth_plus")}
    source_result_by_sample_cond = {(str(r["sample_id"]), str(r["condition"])): r for r in source_results if r.get("family") == "source_axis" and r.get("source") == "natural" and r.get("face_id") == "primary" and r.get("condition") in ("baseline", "orth_plus")}
    if len(source_plan_by_sample_cond) != 160 or len(source_result_by_sample_cond) != 160:
        raise AuditError("retry3 baseline/orth_plus source row coverage incomplete")
    recovery_path, parent_path = source_root / "recovery_code_seal.json", source_root / "continuation_parent.json"
    if sha256_file(recovery_path) != RECOVERY_SEAL_SHA or sha256_file(parent_path) != CONTINUATION_PARENT_SHA:
        raise AuditError("retry3 recovery-code or continuation-parent seal mismatch")
    recovery, parent = read_json(recovery_path), read_json(parent_path)
    if recovery.get("status") != "sealed_before_retry3_model_forward" or recovery.get("production_protocol_sha256") != SOURCE_PROTOCOL_SHA or recovery.get("production_row_manifest_sha256") != SOURCE_ROWS_SHA:
        raise AuditError("retry3 recovery-code seal is not bound to the frozen source protocol and rows")
    if parent.get("status") != "parent_preserved_child_continuation" or parent.get("child_run_dir") != str(source_root) or parent.get("parent_protocol_sha256") != SOURCE_PROTOCOL_SHA or parent.get("parent_row_manifest_sha256") != SOURCE_ROWS_SHA or parent.get("child_protocol_rows_bytes_match_parent_after_detach") is not True or parent.get("parent_global_mutable_hashes_unchanged_before_retry3") is not True:
        raise AuditError("retry3 continuation-parent provenance failed")
    parent_root = Path(str(parent.get("parent_run_dir", ""))).resolve(strict=True)
    input_manifest_ref = protocol.get("input_manifest", {}).get("path")
    input_manifest_path = Path(str(input_manifest_ref)).resolve(strict=True)
    if sha256_file(input_manifest_path) != INPUT_MANIFEST_SHA:
        raise AuditError("frozen external input manifest file hash mismatch")
    input_manifest = read_json(input_manifest_path)
    natural_by_id = {str(r["id"]): r for r in input_manifest.get("records", []) if r.get("split") == "evaluation"}
    if len(natural_by_id) != 80:
        raise AuditError("external input manifest does not contain the frozen 80 natural evaluation rows")

    metrics: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    geometry_rows: dict[str, dict[str, Any]] = {}
    temporal_null: dict[str, dict[str, dict[str, dict[str, np.ndarray | None]]]] = defaultdict(lambda: defaultdict(dict))
    null_permutation_hashes: dict[str, list[str]] = {}
    checks = Counter()
    checks["calibration_axes"] = 1
    max_diffs: dict[str, float] = defaultdict(float)
    for sample in samples:
        sample_rows = [r for r in plan if str(r["sample_id"]) == sample]
        if {r.get("condition") for r in sample_rows} != set(CONDITIONS) or len(sample_rows) != 4:
            raise AuditError(f"bad four-cell plan for {sample}")
        result_rows = [result_by_id[str(r["row_id"])] for r in sample_rows]
        if any(r.get("status") != "scored" for r in result_rows):
            raise AuditError(f"unscored planned condition in {sample}")
        if str(result_rows[0].get("condition")) != "q00":
            raise AuditError(f"first per-sample result is not the q00 context anchor: {sample}")
        for r in result_rows:
            planned = row_by_id[str(r["row_id"])]
            for key in ("ordered_eval_index", "sample_id", "speaker_id", "split", "family", "condition", "face_id", "visual_id", "source_context_npz", "score_npz", "visual_npz", "natural_input_path", "natural_input_sha256"):
                if r.get(key) != planned.get(key):
                    raise AuditError(f"row/result identity differs for {r['row_id']} field {key}")
            if r.get("source_context_json") != planned.get("source_context_json") or r.get("source_context_npz_sha256") != planned.get("source_context_npz_sha256") or r.get("source_context_json_sha256") != planned.get("source_context_json_sha256"):
                raise AuditError(f"row/result source-context path/hash bindings differ for {r['row_id']}")
            if r.get("anchor") != {"k": 3, "column": 18, "official_syncnet_offset": -3}:
                raise AuditError(f"wrong anchor on {r['row_id']}")
            if r.get("family") != "latent_angle_radius" or r.get("split") != "evaluation" or r.get("face_id") != "primary" or r.get("visual_id") != VISUAL_IDS[str(r["condition"])] or r.get("audio_id") not in (None, "A_RAW_N"):
                raise AuditError(f"unexpected evaluation row identity on {r['row_id']}")
        first = result_rows[0]
        ctx_path = safe_path(root, str(first["context_npz"]))
        ctx_sha = str(first["context_npz_sha256"])
        if sha256_file(ctx_path) != ctx_sha or all_hashes.get(ctx_path.relative_to(root).as_posix()) != ctx_sha:
            raise AuditError(f"context hash mismatch for {sample}")
        ctx_meta_path = safe_path(root, str(first["context_json"]))
        if sha256_file(ctx_meta_path) != str(first["context_json_sha256"]):
            raise AuditError(f"context JSON hash mismatch for {sample}")
        context_meta = read_json(ctx_meta_path)
        if context_meta.get("npz_sha256") != ctx_sha:
            raise AuditError(f"scoring context JSON does not bind its latent NPZ: {sample}")
        ctx = npz(ctx_path)
        plan_row = row_by_id[str(first["row_id"])]
        factor_meta_path = safe_path(root, str(plan_row["context_json"]))
        factor_meta_sha = all_hashes.get(factor_meta_path.relative_to(root).as_posix())
        if factor_meta_sha is None:
            raise AuditError(f"factorized latent metadata missing from all-files seal: {sample}")
        factor_meta = read_json(factor_meta_path)
        factor_npz_ref = str(plan_row["context_npz"])
        factor_npz_path = safe_path(root, factor_npz_ref)
        factor_npz_sha = str(factor_meta.get("factorized_context_npz", {}).get("sha256", ""))
        if factor_meta.get("factorized_context_npz", {}).get("path") != factor_npz_ref or all_hashes.get(factor_npz_path.relative_to(root).as_posix()) != factor_npz_sha:
            raise AuditError(f"row plan and factorized-latent metadata do not bind the same NPZ: {sample}")
        factor_ctx = npz(factor_npz_path)
        # The scored-context NPZ deliberately contains scoring/audio arrays and
        # four z cells; the factorized-context NPZ separately stores u. d/u are
        # bound to the sealed CAL axes and retry3's original source context.
        for key in ("z0_saved_float32", "z1_saved_float32", "z_q10_float32", "z_q01_float32", "natural_speech_mask_fixed_uint8"):
            if not np.array_equal(ctx[key], factor_ctx[key]):
                raise AuditError(f"scoring context differs from frozen factorized latent context at {key}: {sample}")
        if not np.array_equal(factor_ctx.get("calibration_u_level_float64"), u_level):
            raise AuditError(f"factorized-context u differs from sealed CAL u axis: {sample}")
        source_result = source_result_by_sample_cond.get((sample, "baseline"))
        source_orth = source_result_by_sample_cond.get((sample, "orth_plus"))
        source_plan_row = source_plan_by_sample_cond.get((sample, "baseline"))
        source_orth_plan = source_plan_by_sample_cond.get((sample, "orth_plus"))
        if source_result is None or source_orth is None or source_plan_row is None or source_orth_plan is None:
            raise AuditError(f"missing retry3 baseline/orth_plus source cells for {sample}")
        if (plan_row.get("source_context_npz"), plan_row.get("source_context_json"), plan_row.get("source_context_npz_sha256"), plan_row.get("source_context_json_sha256")) != (source_result.get("context_npz"), source_result.get("context_json"), source_result.get("context_npz_sha256"), source_result.get("context_json_sha256")):
            raise AuditError(f"norm row plan source-context refs differ from sealed retry3 baseline result: {sample}")
        source_ctx_path = safe_path(source_root, str(source_plan_row["context_npz"]))
        source_ctx_json_ref = PurePosixPath(str(source_plan_row["context_npz"])).with_suffix(".json").as_posix()
        source_ctx_meta_path = safe_path(source_root, source_ctx_json_ref)
        source_ctx_sha = str(source_result["context_npz_sha256"])
        source_ctx_meta_sha = str(source_result["context_json_sha256"])
        if source_result.get("context_npz_sha256") != source_orth.get("context_npz_sha256") or source_result.get("context_json_sha256") != source_orth.get("context_json_sha256"):
            raise AuditError(f"retry3 baseline/orth_plus rows do not share source-context hashes: {sample}")
        # The continuation copied each sealed source artifact into retry3, while
        # some inherited result/JSON rows preserve retry2 absolute references.
        # The private audit namespace may not contain the historical retry2
        # files, so compare that exact lineage path lexically; read and hash only
        # the retry3 child artifact above.
        parent_ctx_path = (parent_root / str(source_plan_row["context_npz"])).resolve(strict=False)
        parent_ctx_json_path = (parent_root / source_ctx_json_ref).resolve(strict=False)
        allowed_source_npz_paths = {str(source_ctx_path.resolve(strict=True)), str(parent_ctx_path)}
        allowed_source_json_paths = {str(source_ctx_meta_path.resolve(strict=True)), str(parent_ctx_json_path)}
        result_source_npz_path = str(Path(str(source_result["context_npz"])).resolve(strict=False))
        result_source_json_path = str(Path(str(source_result["context_json"])).resolve(strict=False))
        orth_source_npz_path = str(Path(str(source_orth["context_npz"])).resolve(strict=False))
        orth_source_json_path = str(Path(str(source_orth["context_json"])).resolve(strict=False))
        if result_source_npz_path not in allowed_source_npz_paths or orth_source_npz_path not in allowed_source_npz_paths or result_source_json_path not in allowed_source_json_paths or orth_source_json_path not in allowed_source_json_paths:
            raise AuditError(f"retry3 result source-context path is outside the sealed parent/child lineage: {sample}")
        if sha256_file(source_ctx_path) != source_ctx_sha or sha256_file(source_ctx_meta_path) != source_ctx_meta_sha:
            raise AuditError(f"retry3 child context copy does not match sealed result hashes: {sample}")
        source_meta = read_json(source_ctx_meta_path)
        source_meta_npz = source_meta.get("context_npz", {})
        if source_meta_npz.get("path") not in allowed_source_npz_paths or source_meta_npz.get("sha256") != source_ctx_sha or source_meta.get("context_json") not in allowed_source_json_paths:
            raise AuditError(f"retry3 source JSON legacy provenance differs from its sealed result row: {sample}")
        source_arrays = npz(source_ctx_path)
        source_array_hashes = source_meta.get("array_sha256", {})
        if array_sha256(source_arrays["calibration_d_source_float64"]) != array_sha256(d_source) or array_sha256(source_arrays["calibration_u_level_float64"]) != array_sha256(u_level):
            raise AuditError(f"retry3 per-sample CAL d/u differ from the independently loaded axes archive: {sample}")
        if source_array_hashes.get("calibration_d_source_float64") != array_sha256(d_source) or source_array_hashes.get("calibration_u_level_float64") != array_sha256(u_level):
            raise AuditError(f"retry3 source JSON d/u array hashes differ from the sealed CAL axes: {sample}")
        factor_source_arrays = factor_meta.get("source_context_arrays", {})
        expected_source_array_hashes = {
            "z0_sha256": array_sha256(source_arrays["natural_z_fixed_float32"]),
            "z1_sha256": array_sha256(source_arrays["axis_z_orth_plus_float32"]),
            "u_sha256": array_sha256(u_level),
            "mask_sha256": array_sha256(source_arrays["natural_speech_mask_fixed_uint8"]),
        }
        if factor_source_arrays.get("z0_key") != "natural_z_fixed_float32" or factor_source_arrays.get("z1_key") != "axis_z_orth_plus_float32" or any(factor_source_arrays.get(key) != value for key, value in expected_source_array_hashes.items()):
            raise AuditError(f"factorized JSON source-array bindings disagree with sealed source context: {sample}")
        factor_source_npz = factor_meta.get("source_context_npz", {})
        factor_source_json = factor_meta.get("source_context_json", {})
        if factor_source_npz != {"path": str(source_result["context_npz"]), "sha256": source_ctx_sha} or factor_source_json != {"path": str(source_result["context_json"]), "sha256": source_ctx_meta_sha}:
            raise AuditError(f"factorized JSON source-context provenance differs from retry3 result bindings: {sample}")
        nested_source = context_meta.get("source_context", {})
        if nested_source.get("source_context_npz") != factor_source_npz or nested_source.get("source_context_json") != factor_source_json:
            raise AuditError(f"scoring JSON source-context provenance differs from factorized/retry3 binding: {sample}")
        if not np.array_equal(ctx["z0_saved_float32"], source_arrays["natural_z_fixed_float32"]) or not np.array_equal(ctx["z1_saved_float32"], source_arrays["axis_z_orth_plus_float32"]):
            raise AuditError(f"q00/q11 source latents differ from retry3 exact context: {sample}")
        if not np.array_equal(ctx["natural_speech_mask_fixed_uint8"], source_arrays["natural_speech_mask_fixed_uint8"]):
            raise AuditError(f"norm activity mask differs from retry3 fixed-N mask: {sample}")
        if not np.array_equal(ctx["natural_pcm16"], source_arrays["natural_pcm16"]):
            raise AuditError(f"norm RAW audio PCM16 differs from retry3 source audio: {sample}")
        factor = factorization_audit({**ctx, "calibration_u_level_float64": factor_ctx["calibration_u_level_float64"]})
        checks["factorizations"] += 1
        raw_info = natural_by_id.get(sample)
        if raw_info is None or str(raw_info.get("speaker")) != speaker_by_sample[sample]:
            raise AuditError(f"sample/speaker is absent from external input manifest: {sample}")
        units = np.asarray(ctx["natural_syncnet_units_raw_float64"])
        if units.dtype != np.float64 or not np.array_equal(units, np.asarray(ctx["natural_pcm16"], dtype=np.float64)):
            raise AuditError(f"RAW PCM16 integer-amplitude SyncNet units identity failed: {sample}")
        mfcc = np.asarray(ctx["natural_mfcc_raw_float64"])
        if mfcc.dtype != np.float64 or mfcc.ndim != 2 or not np.isfinite(mfcc).all():
            raise AuditError(f"RAW PCM MFCC cache has invalid float64 shape/values: {sample}")
        natural_path = Path(str(sample_rows[0]["natural_input_path"])).resolve(strict=True)
        expected_natural_sha = str(raw_info.get("ref_sha256"))
        if sample_rows[0].get("natural_input_sha256") != expected_natural_sha or sha256_file(natural_path) != expected_natural_sha:
            raise AuditError(f"RAW natural WAV binding differs from frozen input manifest: {sample}")
        if any(Path(str(r["natural_input_path"])).resolve() != natural_path or r.get("natural_input_sha256") != expected_natural_sha for r in sample_rows):
            raise AuditError(f"four score rows do not share the same RAW natural PCM: {sample}")
        with wave.open(str(natural_path), "rb") as wav:
            if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise AuditError(f"RAW natural WAV is not mono 16-kHz PCM16: {sample}")
            pcm_bytes = wav.readframes(wav.getnframes())
        if not np.array_equal(np.frombuffer(pcm_bytes, dtype="<i2"), ctx["natural_pcm16"]):
            raise AuditError(f"context PCM differs from the original natural WAV: {sample}")
        checks["source_context_bindings"] += 1
        q_by_cond = {"q00": ctx["z0_saved_float32"], "q10": ctx["z_q10_float32"], "q01": ctx["z_q01_float32"], "q11": ctx["z1_saved_float32"]}
        audio = np.asarray(ctx["natural_audio_embeddings_raw_float32"])
        n_audio = len(audio)
        lips_by_cond: dict[str, np.ndarray] = {}
        visual_files = {}
        for cond, row in ((str(r["condition"]), r) for r in result_rows):
            if row.get("flags_at_forward") != [expected_flags_json]:
                raise AuditError(f"row forward flag record differs from actual-call flags: {row['row_id']}")
            video_path = safe_path(root, str(row["video_path"]))
            if all_hashes.get(video_path.relative_to(root).as_posix()) != str(row.get("video_sha256")):
                raise AuditError(f"video hash differs from the sealed result row: {row['row_id']}")
            visual_path = safe_path(root, str(row["visual_npz"]))
            visual_sha = str(row["visual_npz_sha256"])
            if all_hashes.get(visual_path.relative_to(root).as_posix()) != visual_sha:
                raise AuditError(f"visual NPZ hash mismatch for {row['row_id']}")
            visual_artifact = npz(visual_path)
            lip = np.asarray(visual_artifact["lip_embeddings_float32"])
            if lip.dtype != np.float32 or lip.ndim != 2 or lip.shape[1] != audio.shape[1] or not np.isfinite(lip).all():
                raise AuditError(f"bad lip embedding artifact for {row['row_id']}")
            if not np.array_equal(visual_artifact["z_condition_float32"], q_by_cond[cond]):
                raise AuditError(f"visual latent does not match factorization context for {row['row_id']}")
            lips_by_cond[cond] = lip
            visual_files[cond] = (visual_path, visual_sha)
        query_n = min(n_audio, *(len(x) for x in lips_by_cond.values()))
        if query_n < 1:
            raise AuditError(f"no common visual/audio query support: {sample}")
        supports = expected_supports(query_n)
        order = int(next(r["ordered_eval_index"] for r in sample_rows if r["condition"] == "q00"))
        rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED + order))
        permutations = [rng.permutation(n_audio).astype(np.int64, copy=False) for _ in range(NULL_DRAWS)]
        null_permutation_hashes[sample] = [hashlib.sha256(np.asarray(pi, dtype="<i8").tobytes()).hexdigest() for pi in permutations]
        for cond, row in ((str(r["condition"]), r) for r in result_rows):
            if int(row.get("common_support_n", -1)) != query_n or int(row.get("audio_row_domain_n", -1)) != n_audio or int(row.get("visual_query_domain_n", -1)) != len(lips_by_cond[cond]):
                raise AuditError(f"recorded support domain differs from independently derived lengths: {sample}/{cond}")
            score_path = safe_path(root, str(row["score_npz"]))
            score_sha = str(row["score_npz_sha256"])
            if all_hashes.get(score_path.relative_to(root).as_posix()) != score_sha:
                raise AuditError(f"score NPZ hash mismatch for {row['row_id']}")
            score = npz(score_path)
            visual = lips_by_cond[cond][:query_n]
            if score["audio_embeddings_raw_float32"].shape != audio.shape or not np.array_equal(score["audio_embeddings_raw_float32"], audio):
                raise AuditError(f"fresh shared A cache differs in {row['row_id']}")
            if not np.array_equal(score["visual_embeddings_raw_float32"], visual):
                raise AuditError(f"V embeddings differ from visual/context artifact in {row['row_id']}")
            vunit, aunit = row_unit(visual), row_unit(audio)
            if not np.array_equal(score["visual_embeddings_unit_float32"], vunit) or not np.array_equal(score["audio_embeddings_unit_float32"], aunit):
                raise AuditError(f"unit embeddings are not independently row-normalized in {row['row_id']}")
            for name in SUPPORTS:
                key = f"common_support_{name}_int32"
                if name == "valid":
                    continue
                if score[key].dtype != np.int32 or not np.array_equal(score[key], supports[name]):
                    raise AuditError(f"support mismatch {name} in {row['row_id']}")
            if score["common_support_valid_int32"].dtype != np.int32 or not np.array_equal(score["common_support_valid_int32"], supports["valid"]):
                raise AuditError(f"valid support mismatch in {row['row_id']}")
            if score["lag_k_int32"].dtype != np.int32 or not np.array_equal(score["lag_k_int32"], LAGS):
                raise AuditError(f"lag k grid mismatch in {row['row_id']}")
            if score["lag_columns_int32"].dtype != np.int32 or not np.array_equal(score["lag_columns_int32"], np.arange(31, dtype=np.int32)):
                raise AuditError(f"lag column grid mismatch in {row['row_id']}")
            rebuilt = {}
            diffs = {}
            for geom, v_emb, a_emb in (("raw", visual, audio), ("unit", vunit, aunit)):
                cross, zero = cross_and_zero(v_emb, a_emb)
                lags = lag_matrix(cross, zero, query_n)
                cross_key = f"cross_distance_{geom}_float32"
                zero_key = f"cross_distance_to_zero_{geom}_float32"
                lag_key = f"distance_matrix_{geom}_float32"
                diffs[f"{geom}_cross"] = compare_float32(score[cross_key], cross, f"{row['row_id']} {geom} cross")
                diffs[f"{geom}_zero"] = compare_float32(score[zero_key], zero, f"{row['row_id']} {geom} zero")
                diffs[f"{geom}_lag"] = compare_float32(score[lag_key], lags, f"{row['row_id']} {geom} lag")
                rebuilt[geom] = metric_payload(lags, supports)
                producer_metrics = row[f"{geom}_support_metrics"]
                max_metric = compare_metrics(producer_metrics, rebuilt[geom], f"{row['row_id']} {geom}")
                diffs[f"{geom}_metric"] = max_metric
                for key, value in diffs.items():
                    max_diffs[key] = max(max_diffs[key], value)
                for support in SUPPORTS:
                    metrics[sample][geom][cond, support] = rebuilt[geom][support]
                temporal_null[sample][geom][cond] = {
                    support: null_metrics_for_support(cross, zero, supports[support], permutations)
                    for support in SUPPORTS
                }
            checks["score_rows"] += 1
            geometry_rows[f"{sample}__{cond}"] = {
                "sample_id": sample, "speaker_id": speaker_by_sample[sample], "condition": cond,
                "factorization": factor if cond == "q00" else None,
                "context_sha256": ctx_sha, "score_sha256": score_sha, "visual_sha256": visual_sha,
                "common_query_n": query_n, "audio_n": n_audio,
                "lip_rows_by_condition": {k: len(v) for k, v in lips_by_cond.items()},
                "max_abs_rebuild_difference": diffs, "metrics": rebuilt,
            }

    if checks["score_rows"] != 320 or checks["factorizations"] != 80 or checks["source_context_bindings"] != 80:
        raise AuditError("verified artifact counts differ from 80 contexts / 320 scores")
    gate_keys = {"q10_minus_q00": ("q10", "q00"), "q10_minus_q01": ("q10", "q01"), "q01_minus_q00": ("q01", "q00"), "q11_minus_q00": ("q11", "q00"), "interaction": None}
    summaries: dict[str, Any] = {"cells": {}, "contrasts": {}, "confirmatory_gates": {}}
    gates_complete = True
    for geom in GEOMETRIES:
        summaries["cells"][geom] = {}
        summaries["contrasts"][geom] = {}
        for support in SUPPORTS:
            summaries["cells"][geom][support] = {}
            summaries["contrasts"][geom][support] = {}
            for metric in METRICS:
                for cond in CONDITIONS:
                    by_speaker = defaultdict(list)
                    for sample in samples:
                        payload = metrics[sample][geom][cond, support]
                        if payload["status"] == "ok":
                            by_speaker[speaker_by_sample[sample]].append(float(payload[metric]))
                    summaries["cells"][geom][support].setdefault(cond, {})[metric] = bootstrap_speakers(by_speaker)
                for contrast, pair in gate_keys.items():
                    by_speaker = defaultdict(list)
                    for sample in samples:
                        cell = {cond: metrics[sample][geom][cond, support] for cond in CONDITIONS}
                        if any(cell[c]["status"] != "ok" for c in CONDITIONS):
                            continue
                        q00, q10, q01, q11 = (float(cell[c][metric]) for c in CONDITIONS)
                        value = {
                            "q10_minus_q00": q10-q00,
                            "q10_minus_q01": q10-q01,
                            "q01_minus_q00": q01-q00,
                            "q11_minus_q00": q11-q00,
                            "interaction": q11-q10-q01+q00,
                        }[contrast]
                        by_speaker[speaker_by_sample[sample]].append(value)
                    summaries["contrasts"][geom][support].setdefault(contrast, {})[metric] = bootstrap_speakers(by_speaker)
    for contrast in ("q10_minus_q00", "q10_minus_q01"):
        for metric in ("C", "C_anchor"):
            result = summaries["contrasts"]["raw"]["guard20"][contrast][metric]
            ci = result["speaker_bootstrap_99_ci"]
            eligible = result["eligible_for_intended_cohort"] and ci is not None
            passed = bool(eligible and ci[0] > 0)
            gates_complete &= eligible
            summaries["confirmatory_gates"][f"{contrast}:{metric}"] = {"estimate": result["equal_speaker_mean"], "ci99": ci, "eligible": eligible, "lower_bound_gt_zero": passed}
    summaries["confirmatory_gates"]["all_four_eligible"] = bool(gates_complete)
    summaries["confirmatory_gates"]["all_four_pass"] = bool(gates_complete and all(x["lower_bound_gt_zero"] for k,x in summaries["confirmatory_gates"].items() if k != "all_four_eligible"))

    null_summary: dict[str, Any] = {"rng": "NumPy PCG64", "draws": NULL_DRAWS, "seed_rule": "20260930 + ordered_eval_index; each full A-row permutation is shared by q00/q10/q01/q11 and raw/unit", "permutation_sha256_by_sample": null_permutation_hashes, "by_geometry": {}}
    for geom in GEOMETRIES:
        null_summary["by_geometry"][geom] = {"by_support": {}}
        for support in SUPPORTS:
            domain = {"cells": {}, "contrasts": {}}
            for metric in METRICS:
                domain["cells"][metric] = {}
                for cond in CONDITIONS:
                    observed, null_vals = {}, {}
                    for sample in samples:
                        cell = metrics[sample][geom][cond, support]
                        if cell["status"] == "ok":
                            observed[sample] = float(cell[metric])
                        current_null = temporal_null[sample][geom][cond][support]
                        null_vals[sample] = None if current_null is None else current_null[metric]
                    domain["cells"][metric][cond] = summarize_null(observed, null_vals, speaker_by_sample, samples, support)
                domain["contrasts"][metric] = {}
                for contrast in ("q10_minus_q00", "q10_minus_q01", "q01_minus_q00", "q11_minus_q00", "interaction"):
                    observed, null_vals = {}, {}
                    for sample in samples:
                        cells = {cond: metrics[sample][geom][cond, support] for cond in CONDITIONS}
                        if any(cells[c]["status"] != "ok" for c in CONDITIONS):
                            continue
                        q00, q10, q01, q11 = (float(cells[c][metric]) for c in CONDITIONS)
                        observed[sample] = {
                            "q10_minus_q00": q10-q00, "q10_minus_q01": q10-q01,
                            "q01_minus_q00": q01-q00, "q11_minus_q00": q11-q00,
                            "interaction": q11-q10-q01+q00,
                        }[contrast]
                        cell_nulls = [temporal_null[sample][geom][c][support] for c in CONDITIONS]
                        if any(x is None for x in cell_nulls):
                            null_vals[sample] = None
                        else:
                            n00, n10, n01, n11 = (x[metric] for x in cell_nulls)
                            null_vals[sample] = {
                                "q10_minus_q00": n10-n00, "q10_minus_q01": n10-n01,
                                "q01_minus_q00": n01-n00, "q11_minus_q00": n11-n00,
                                "interaction": n11-n10-n01+n00,
                            }[contrast]
                    domain["contrasts"][metric][contrast] = summarize_null(observed, null_vals, speaker_by_sample, samples, support)
            null_summary["by_geometry"][geom]["by_support"][support] = domain

    report = {
        "schema_version": 1, "analysis_type": "independent_norm_angle_radius_80x4_all_domain_supplement",
        "status": "COMPLETE", "created_utc": datetime.now(timezone.utc).isoformat(),
        "consumer": {"source_path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve()), "python": platform.python_version(), "numpy": np.__version__, "device": "CPU", "imports_producer_or_scoring": False},
        "input_seals": {"run_root": str(root), "protocol_sha256": sha256_file(protocol_path), "row_plan_sha256": sha256_file(rows_path), "results_sha256": sha256_file(results_path), "run_status_sha256": sha256_file(status_path), "code_seal_sha256": sha256_file(seal_path), "all_files_sha256": sha256_file(all_path), "source_retry3_protocol_sha256": SOURCE_PROTOCOL_SHA, "source_retry3_rows_sha256": SOURCE_ROWS_SHA, "source_retry3_calibration_summary_sha256": sha256_file(calibration_summary_path), "source_retry3_calibration_axes_sha256": sha256_file(calibration_axes_path), "source_retry3_calibration_d_array_sha256": array_sha256(d_source), "source_retry3_calibration_u_array_sha256": array_sha256(u_level), "spec_sha256": SPEC_SHA, "calibration_geometry_sha256": GEOMETRY_SHA, "input_manifest_sha256": INPUT_MANIFEST_SHA},
        "artifact_schema_mapping": {"planned_context_npz": "factorized latent context: q00/q10/q01/q11, fixed-N mask, CAL u", "scored_context_npz": "shared score/audio cache: q00/q10/q01/q11, fixed-N mask, RAW PCM/MFCC/A embeddings/support; intentionally no d/u arrays", "retry3_source_context_resolution": "child row-plan context_npz is the resolvable relative artifact; untouched result/context JSON preserve the historical parent absolute path and are checked against content hashes", "calibration_axes": "retry3 sealed protocol calibration_axes path/hash: d_source and u_level float64[512]"},
        "verification": {"planned_samples": 80, "completed_samples": 80, "planned_rows": 320, "scored_rows": 320, "failed_rows": 0, "speakers": 40, "clips_per_speaker": 2, "all_files_entries_verified": inventory_n, "calibration_axes_verified": checks["calibration_axes"], "latent_factorizations_verified": checks["factorizations"], "retry3_source_contexts_verified": checks["source_context_bindings"], "score_rows_rebuilt": checks["score_rows"], "max_abs_rebuild_differences": dict(max_diffs), "q10_q01_all_frame_identity": True, "q00_q11_endpoint_bitexact": True, "original_z0_z1_inactive_rows_bitexact": True, "q10_q01_inactive_identity_required": False},
        "bootstrap": {"unit": "40 speakers, equal weight; each speaker is the mean of its two paired clip contrasts", "rng": "PCG64", "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "ci99": "percentile [0.005,0.995]"},
        "metric_definitions": {"C": "B-Dmin", "C_anchor": "B-D at fixed column18 (k=+3; official offset=-3)", "B": "median across the 31 lag distances", "D": "minimum across the 31 lag distances", "search_gain": "D0-Dmin; D0 is column15 (k=0)"},
        "interpretation_constraint": "The q10/q01 direction-radius contrasts are factorial generator responses; their arithmetic decomposition is not a mediation percentage.",
        "summaries": summaries,
        "temporal_null_all_metrics": null_summary,
        "per_row_metrics": geometry_rows,
        "elapsed_seconds": time.monotonic()-started,
    }
    report_path = out / "independent_norm_reanalysis.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({"status": report["status"], "verification": report["verification"], "summary_path": str(args.outdir.resolve()/"independent_norm_reanalysis.json")}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
