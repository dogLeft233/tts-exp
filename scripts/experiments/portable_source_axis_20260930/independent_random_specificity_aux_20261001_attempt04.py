#!/usr/bin/env python3
"""Independent CPU audit and inference for the frozen random-specificity run.

This consumer imports no producer, ops, or scoring module. It verifies the
sealed 80x2x4 artifact grid, independently rebuilds PCM/V/A distance and lag
geometry, checks the corrected float32 dose contract, and only then writes the
prespecified equal-speaker estimates and cached temporal null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


SPEC_SHA256 = "679b11c8a67b18ac397d63b95900d55865b6dbf1ad7e8662bf3a5b2e915c8c79"
SOURCE_PROTOCOL_SHA256 = "8ba27fcb395dccbb493ad4831e8b68ac18c485b61e39d6e3b7b3018fd668cf39"
SOURCE_ROWS_SHA256 = "19f805f6935c85475ad28319e17d2694a333a85d055d3e19a0eb93bda7390dab"
SOURCE_RESULTS_SHA256 = "fd1f9ce1b23861de5eea7d87470f74e52451f1cc8bb050cbe85141a62975d3fa"
CONTINUATION_PARENT_SHA256 = "d5fbfa99865b958a006f504e0dd7c6e5120ec194651aa639acad4d540528b862"
RECOVERY_CODE_SEAL_SHA256 = "55c1512e35bf72563d0e01623f7e14f77aaa776dd94ee565926d9df9bea90417"
INPUT_MANIFEST_SHA256 = "6b0ce3252cf776a86cf84b19ab53afd936526216e4ddb30dc587a72a66409b4b"
CAL_SUMMARY_SHA256 = "34876249bb3463417f903ea49e11da03306b1c0785a28cb6a4f97da92d59f22a"
CAL_AXES_SHA256 = "e71b669a67f435d951742cf67a9683e0385b77f5b97ab7a7408851389ade9244"
AUX_RUN_ROOT = "/root/tts-research-system-20261001/source_axis/random_specificity_aux_20261001_attempt03"
AUX_RUN_SEALS = {
    "evaluation_protocol.json": "2a5f834939164a75b68b9d059c58580b4ad260dc7d1848536b3206e1018a579b",
    "evaluation_rows.jsonl": "33b0d36697e5c28f12210bcea4cbab164b2c12353f9b5df0bafbe2d0a62a1aa3",
    "evaluation_results.jsonl": "0e783920d5a7fae1883b005177ad467149c1896df263aab5d6973f76cf1977e3",
    "run_status.json": "e3ef70034ac08052c331a4d7910145f3302043e08c0ab58c610887ce8c3cb753",
    "code_seal.json": "73c45d80e1e904aa091f3577fe8574224dd48b17d1c5532e97b86b987782331d",
    "all_files_sha256.json": "555c854cb42508377c2de5cc689e63b308dd9ebf12d02c24969e1fa050d7b535",
    "source_artifact_inventory.json": "e1d239d20966deab34cb67ee66846f6c422b55d32069d442644d44769aa16f2a",
}
CODE_SHA256 = {
    "random_specificity_aux_20261001_v2.py": "792e0522572f5d02f476519d85139849af472348f723bd3c730a090a530a43e7",
    "randomdose_f32_v2.py": "c9af12003cd970899cb76d3147201b896352eb8a050389199e6d476f090027f4",
    "__init__.py": "788eaf7bc75753cd5e3d4eac5d9af7d0e55489f1520137ec0282072b1f5500e3",
    "evaluate.py": "e50a31eaa2493b83e87cb4426b742e8f28e73836cbb9374b2319e7b60c940880",
    "ops.py": "64eecefea72a494cc07573c8bbdd872351e23b9f7998e52b280483af3b9c0a3b",
    "scoring.py": "9411436d9fc199e91116a44523eeeccb66be0e59f7f20f491b0b9c8cc5a3ad87",
    "wav2lip.py": "f0a6cd180679aa1bb218df845632211731651e122b5baa706a38a3db32ccfe9d",
}
CONDITIONS = ("orth_plus", "random1", "random2", "random3")
FACES = ("primary", "secondary")
GEOMETRIES = ("raw", "unit")
SUPPORTS = ("full", "valid", "guard20", "guard15")
METRICS = ("C", "C_anchor", "B", "D", "search_gain")
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20_260_930
NULL_DRAWS = 100
VSHIFT = 15
ANCHOR_COLUMN = 18
DISTANCE_ATOL = 5e-4
METRIC_ATOL = 5e-4
DOSE_TOLERANCE_FLOOR = 1e-6
LATENT_DIM = 512
EMBEDDING_DIM = 1024
EPSILON = np.float32(1e-6)
REQUIRED_FLAGS = {
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "deterministic_algorithms": True,
}


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def expected_random_direction(source_axis: np.ndarray, level_axis: np.ndarray, seed: int) -> np.ndarray:
    d = np.asarray(source_axis, dtype=np.float64)
    u = np.asarray(level_axis, dtype=np.float64)
    unit_u = u / np.linalg.norm(u)
    d_perp = d - unit_u * float(unit_u @ d)
    target_norm = 0.5 * float(np.linalg.norm(d_perp))
    rng = np.random.default_rng(int(seed))
    candidate = rng.standard_normal(LATENT_DIM, dtype=np.float64)
    residual_d = d - unit_u * float(unit_u @ d)
    unit_d = residual_d / np.linalg.norm(residual_d)
    projected = candidate - unit_u * float(unit_u @ candidate) - unit_d * float(unit_d @ candidate)
    return projected * (target_norm / np.linalg.norm(projected))


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise AuditError(f"expected JSON object at {path}:{line_no}")
                rows.append(item)
    return rows


def safe_artifact(root: Path, rel: str) -> Path:
    parts = PurePosixPath(str(rel)).parts
    if not parts or PurePosixPath(str(rel)).is_absolute() or any(part in ("", "..") for part in parts):
        raise AuditError(f"unsafe artifact reference: {rel}")
    path = root.joinpath(*parts).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise AuditError(f"artifact escapes run root: {rel}") from exc
    return path


def check_file(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise AuditError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual.lower() != str(expected).lower():
        raise AuditError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")
    return actual


def read_pcm16_mono(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as stream:
        rate, channels, width = stream.getframerate(), stream.getnchannels(), stream.getsampwidth()
        if rate != 16000 or channels != 1 or width != 2:
            raise AuditError(f"expected mono 16-kHz PCM16 input WAV: {path}")
        payload = bytearray()
        while True:
            chunk = stream.readframes(1 << 20)
            if not chunk:
                break
            payload.extend(chunk)
    if len(payload) % 2:
        raise AuditError(f"PCM16 input WAV has odd payload length: {path}")
    return rate, np.frombuffer(payload, dtype="<i2").copy()


def resolve_source_ref(root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def supports_for(count: int) -> dict[str, np.ndarray]:
    full = np.arange(count, dtype=np.int32)
    return {
        "full": full,
        "valid": full.copy(),
        "guard20": np.arange(min(20, count), max(min(20, count), count - 20), dtype=np.int32),
        "guard15": np.arange(min(15, count), max(min(15, count), count - 15), dtype=np.int32),
    }


def unit_rows(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value, dtype=np.float32)
    norm = np.sqrt(np.sum(data * data, axis=1, keepdims=True, dtype=np.float32), dtype=np.float32)
    return (data / np.maximum(norm, np.float32(1e-12))).astype(np.float32, copy=False)


def cross_and_zero(v: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    visual = np.asarray(v, dtype=np.float32)
    audio = np.asarray(a, dtype=np.float32)
    if visual.ndim != 2 or audio.ndim != 2 or visual.shape[1] != audio.shape[1] or not len(visual) or not len(audio):
        raise AuditError("V/A embeddings must be nonempty [rows,features] arrays")
    cross = np.empty((len(visual), len(audio)), dtype=np.float32)
    for start in range(0, len(visual), 4):
        stop = min(start + 4, len(visual))
        delta = visual[start:stop, None, :] - audio[None, :, :]
        np.add(delta, EPSILON, out=delta)
        squared = np.multiply(delta, delta, dtype=np.float32)
        sums = np.sum(squared, axis=2, dtype=np.float32)
        np.sqrt(sums, out=cross[start:stop])
    zero = visual.copy()
    np.add(zero, EPSILON, out=zero)
    zero_sums = np.sum(np.multiply(zero, zero, dtype=np.float32), axis=1, dtype=np.float32)
    return cross, np.sqrt(zero_sums, dtype=np.float32).astype(np.float32, copy=False)


def lag_matrix(cross: np.ndarray, zero: np.ndarray, query_n: int, audio_n: int) -> np.ndarray:
    matrix = np.asarray(cross, dtype=np.float32)
    boundary = np.asarray(zero, dtype=np.float32)
    if query_n < 1 or query_n > matrix.shape[0] or audio_n < 1 or audio_n > matrix.shape[1] or boundary.shape != (matrix.shape[0],):
        raise AuditError("lag-domain sizes do not fit the saved full V×A matrix")
    output = np.empty((query_n, 2 * VSHIFT + 1), dtype=np.float32)
    for i in range(query_n):
        for col, k in enumerate(range(-VSHIFT, VSHIFT + 1)):
            a_index = i + k
            output[i, col] = matrix[i, a_index] if 0 <= a_index < audio_n else boundary[i]
    return output


def metric_payload(matrix: np.ndarray, supports: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, indices in supports.items():
        if len(indices) == 0:
            out[name] = {"status": "too_short", "support_n": 0}
            continue
        curve = np.asarray(matrix[indices], dtype=np.float32).mean(axis=0, dtype=np.float32).astype(np.float64)
        best = int(np.argmin(curve))
        median = float(np.median(curve))
        dmin = float(curve[best])
        d0 = float(curve[VSHIFT])
        danchor = float(curve[ANCHOR_COLUMN])
        out[name] = {
            "status": "ok", "support_n": int(len(indices)), "window_indices": indices.tolist(),
            "curve": curve.tolist(), "optimal_column": best, "optimal_offset_frames": int(VSHIFT - best),
            "D_min": dmin, "D": dmin, "B_median": median, "B": median,
            "C_best": median - dmin, "C": median - dmin,
            "D_fixed_anchor": danchor, "D_anchor": danchor,
            "C_fixed_anchor": median - danchor, "C_anchor": median - danchor,
            "D_zero": d0, "search_gain": d0 - dmin,
            "fixed_anchor_column": ANCHOR_COLUMN, "fixed_anchor_offset_frames": VSHIFT - ANCHOR_COLUMN,
        }
    return out


def compare_saved_metric(saved: dict[str, Any], rebuilt: dict[str, Any], label: str) -> float:
    max_diff = 0.0
    for support in SUPPORTS:
        left, right = saved.get(support), rebuilt.get(support)
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise AuditError(f"{label}: missing {support} metric record")
        if left.get("status") != right.get("status") or int(left.get("support_n", -1)) != int(right.get("support_n", -2)):
            raise AuditError(f"{label}: {support} status/support mismatch")
        if right["status"] == "too_short":
            continue
        if left.get("window_indices") != right.get("window_indices"):
            raise AuditError(f"{label}: {support} support indices mismatch")
        for key in ("D_min", "B_median", "C_best", "D_fixed_anchor", "C_fixed_anchor"):
            delta = abs(float(left[key]) - float(right[key]))
            max_diff = max(max_diff, delta)
            if delta > METRIC_ATOL:
                raise AuditError(f"{label}: {support}.{key} differs by {delta:.8g}")
        curve_diff = float(np.max(np.abs(np.asarray(left["curve"], dtype=np.float64) - np.asarray(right["curve"], dtype=np.float64))))
        max_diff = max(max_diff, curve_diff)
        if curve_diff > METRIC_ATOL:
            raise AuditError(f"{label}: {support} curve differs by {curve_diff:.8g}")
    return max_diff


def projection(base: np.ndarray, proposal: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, float]:
    """Independent scalar-dual solve for y=max(0, proposal-lambda*axis)."""
    z = np.asarray(base, dtype=np.float64)
    x = np.asarray(proposal, dtype=np.float64)
    u = np.asarray(axis, dtype=np.float64)
    if z.ndim != 1 or x.shape != z.shape or u.shape != z.shape:
        raise AuditError("KKT vectors are misaligned")
    tolerance = 2e-12 * max(1.0, float(np.linalg.norm(u) * np.linalg.norm(z)))

    def evaluate(lam: float) -> tuple[float, np.ndarray]:
        y = np.maximum(0.0, x - np.float64(lam) * u)
        return float(u @ (y - z)), y

    f0, y0 = evaluate(0.0)
    if abs(f0) <= tolerance:
        return y0, f0
    lo = hi = 0.0
    if f0 > 0.0:
        hi = 1.0
        fhi, _ = evaluate(hi)
        for _ in range(256):
            if fhi <= 0.0:
                break
            hi *= 2.0
            fhi, _ = evaluate(hi)
        else:
            raise AuditError("could not bracket positive KKT dual")
    else:
        lo = -1.0
        flo, _ = evaluate(lo)
        for _ in range(256):
            if flo >= 0.0:
                break
            lo *= 2.0
            flo, _ = evaluate(lo)
        else:
            raise AuditError("could not bracket negative KKT dual")
    residual = f0
    candidate = y0
    for _ in range(256):
        mid = (lo + hi) / 2.0
        residual, candidate = evaluate(mid)
        if abs(residual) <= tolerance:
            break
        if residual > 0.0:
            lo = mid
        else:
            hi = mid
    if abs(residual) > tolerance:
        raise AuditError(f"KKT residual {residual:.4g} exceeds tolerance {tolerance:.4g}")
    return candidate, residual


def dose_contract(z0: np.ndarray, orth: np.ndarray, mask: np.ndarray, d: np.ndarray, u: np.ndarray,
                  directions: list[np.ndarray], z_randoms: list[np.ndarray], metadata: dict[str, Any]) -> dict[str, Any]:
    active = np.flatnonzero(mask.astype(bool))
    target = np.linalg.norm(orth[active].astype(np.float64) - z0[active].astype(np.float64), axis=1)
    report: dict[str, Any] = {"active_frames": int(len(active)), "inactive_frames": int((~mask.astype(bool)).sum()), "arms": {}}
    for arm_index, (direction, zrand) in enumerate(zip(directions, z_randoms, strict=True), start=1):
        name = f"random{arm_index}"
        if zrand.dtype != np.float32 or zrand.shape != z0.shape or not np.isfinite(zrand).all() or np.any(zrand < 0):
            raise AuditError(f"{name}: random latent dtype/shape/finiteness/nonnegativity failure")
        if not np.array_equal(zrand[~mask.astype(bool)], z0[~mask.astype(bool)]):
            raise AuditError(f"{name}: inactive latent frames are not exact baseline copies")
        actual = np.linalg.norm(zrand[active].astype(np.float64) - z0[active].astype(np.float64), axis=1)
        errors = np.abs(actual - target)
        tolerances = np.maximum(DOSE_TOLERANCE_FLOOR, DOSE_TOLERANCE_FLOOR * target)
        passed = errors <= tolerances
        arm_meta = metadata.get(name, {})
        diagnostics = arm_meta.get("frames", [])
        if len(diagnostics) != len(active):
            raise AuditError(f"{name}: per-frame dose diagnostics do not cover every active frame")
        if not np.array_equal(np.asarray([int(x["frame_index"]) for x in diagnostics]), active):
            raise AuditError(f"{name}: per-frame dose diagnostics have a different frame order")
        capped = np.asarray([bool(x.get("capped_infeasible", x.get("infeasible", False))) for x in diagnostics])
        projection_max_difference = 0.0
        max_kkt_residual = 0.0
        for position, frame in enumerate(active):
            detail = diagnostics[position]
            base = z0[frame].astype(np.float64)
            scale = float(detail.get("scale", -1.0))
            if not 0.0 <= scale <= 64.0:
                raise AuditError(f"{name}: invalid scalar-search scale at frame {frame}")
            projected, residual = projection(base, base + scale * direction.astype(np.float64), u)
            reconstructed = projected.astype(np.float32)
            if not np.array_equal(reconstructed, zrand[frame]):
                diff = float(np.max(np.abs(reconstructed.astype(np.float64) - zrand[frame].astype(np.float64))))
                projection_max_difference = max(projection_max_difference, diff)
                raise AuditError(f"{name}: independently reconstructed KKT frame differs by {diff:.8g}")
            max_kkt_residual = max(max_kkt_residual, abs(residual))
            if abs(float(detail.get("actual_dose_error_float32", 0.0)) - float(actual[position] - target[position])) > 1e-10:
                raise AuditError(f"{name}: producer's per-frame actual dose diagnostic differs")
            if abs(float(detail.get("actual_float32_norm_float64", -1.0)) - float(actual[position])) > 1e-10:
                raise AuditError(f"{name}: producer's per-frame float32 norm differs")
            projected_norm = float(np.linalg.norm(projected - base))
            if abs(float(detail.get("projected_norm_float64", -1.0)) - projected_norm) > 1e-10:
                raise AuditError(f"{name}: producer's per-frame float64 projection norm differs")
            if abs(float(detail.get("projected_dose_error_float64", float("inf"))) - float(projected_norm - target[position])) > 1e-10:
                raise AuditError(f"{name}: producer's float64 dose error differs from independent projection")
            if abs(float(detail.get("target_norm_float64", detail.get("target_norm", -1.0))) - float(target[position])) > 1e-10:
                raise AuditError(f"{name}: producer's target dose diagnostic differs")
            if abs(float(detail.get("kkt_residual64", float("inf"))) - residual) > 1e-9:
                raise AuditError(f"{name}: producer KKT residual differs from independent solve")
            actual_constraint = abs(float(u @ (zrand[frame].astype(np.float64) - base)))
            constraint_bound = float(detail.get("float32_constraint_abs_bound", -1.0))
            if constraint_bound < 0.0 or abs(actual_constraint - float(detail.get("float32_constraint_abs", -1.0))) > 1e-10:
                raise AuditError(f"{name}: saved float32 KKT residual/bound differs from independent value")
            if actual_constraint > constraint_bound + 1e-10:
                raise AuditError(f"{name}: float32 level-axis residual exceeds its saved rounding bound")
            expected_bound = float(np.sum(np.abs(u) * np.abs(zrand[frame].astype(np.float64) - projected), dtype=np.float64))
            if abs(expected_bound - constraint_bound) > 1e-10:
                raise AuditError(f"{name}: float32 level-axis rounding bound differs from independent reconstruction")
            if bool(detail.get("tolerance_met")) != bool(passed[position]):
                raise AuditError(f"{name}: per-frame dose pass flag differs from independent float32 check")
        expected_misses = int((~passed).sum())
        expected_capped = int(capped.sum())
        if int(arm_meta.get("dose_contract_pass_frames", -1)) != int(passed.sum()):
            raise AuditError(f"{name}: aggregate dose pass count differs from saved frames")
        if int(arm_meta.get("dose_tolerance_miss_frames", -1)) != expected_misses:
            raise AuditError(f"{name}: aggregate dose miss count differs from saved frames")
        if int(arm_meta.get("capped_infeasible_frames", -1)) != expected_capped:
            raise AuditError(f"{name}: aggregate capped count differs from saved frames")
        direction_hash = array_sha256(direction)
        latent_hash = array_sha256(zrand)
        if arm_meta.get("random_direction_sha256") != direction_hash or arm_meta.get("output_z_sha256") != latent_hash:
            raise AuditError(f"{name}: dose summary does not bind the checked direction/output arrays")
        report["arms"][name] = {
            "pass_frames": int(passed.sum()), "miss_frames": expected_misses, "capped_infeasible_frames": expected_capped,
            "max_abs_actual_dose_error": float(errors.max(initial=0.0)),
            "p95_abs_actual_dose_error": float(np.percentile(errors, 95)) if len(errors) else 0.0,
            "max_tolerance_ratio": float(np.max(errors / tolerances, initial=0.0)),
            "max_kkt_residual": max_kkt_residual, "float32_level_dot_max_abs": float(max(abs(float(u @ (zrand[f].astype(np.float64) - z0[f].astype(np.float64)))) for f in active)),
            "reconstructed_kkt_max_abs_difference": projection_max_difference,
            "strict_equal_dose_eligible": bool(expected_misses == 0 and expected_capped == 0),
        }
    report["all_three_random_arms_strictly_equal_dose"] = all(item["strict_equal_dose_eligible"] for item in report["arms"].values())
    return report


def bootstrap_speakers(speaker_clip_values: dict[str, list[float]]) -> dict[str, Any]:
    speakers = sorted(speaker_clip_values)
    if len(speakers) != 40 or any(len(speaker_clip_values[s]) != 2 for s in speakers):
        raise AuditError("equal-speaker bootstrap requires 40 speakers with exactly two clip values")
    values = np.asarray([np.mean(speaker_clip_values[s], dtype=np.float64) for s in speakers], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, len(speakers), size=(BOOTSTRAP_DRAWS, len(speakers)), dtype=np.int64)
    draws = values[indices].mean(axis=1, dtype=np.float64)
    low, high = np.quantile(draws, [0.005, 0.995], method="linear")
    return {
        "equal_speaker_mean": float(values.mean(dtype=np.float64)),
        "speaker_mean_sd": float(values.std(ddof=1, dtype=np.float64)),
        "speaker_count": 40, "clips_per_speaker": 2,
        "bootstrap": {"method": "PCG64 cluster bootstrap of 40 speakers; each speaker contributes the mean of its two clip contrasts", "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "ci_level": 0.99, "percentiles": [0.5, 99.5], "ci_low": float(low), "ci_high": float(high)},
    }


def permutation_digest(values: np.ndarray) -> str:
    data = np.asarray(values, dtype="<i8")
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


def permutations_for(audio_n: int, ordered_index: int) -> tuple[list[np.ndarray], list[str]]:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED + int(ordered_index)))
    expected = np.arange(audio_n, dtype=np.int64)
    vectors, hashes = [], []
    for _ in range(NULL_DRAWS):
        value = rng.permutation(audio_n).astype(np.int64, copy=False)
        if not np.array_equal(np.sort(value), expected):
            raise AuditError("temporal-null vector is not a full RAW-N A permutation")
        vectors.append(value)
        hashes.append(permutation_digest(value))
    return vectors, hashes


def null_metric(cross: np.ndarray, queries: np.ndarray, permutation: np.ndarray) -> dict[str, float]:
    v = np.asarray(cross, dtype=np.float32)
    q = np.asarray(queries, dtype=np.int64)
    pi = np.asarray(permutation, dtype=np.int64)
    if not len(q):
        raise AuditError("guard20 is empty; temporal null unavailable for this face")
    audio_indices = q[:, None] + np.arange(-VSHIFT, VSHIFT + 1, dtype=np.int64)[None, :]
    if np.any(audio_indices < 0) or np.any(audio_indices >= len(pi)):
        raise AuditError("guard20 query references an out-of-domain RAW-N A row")
    matrix = v[q[:, None], pi[audio_indices]]
    curve = matrix.mean(axis=0, dtype=np.float32).astype(np.float64)
    d = float(np.min(curve))
    b = float(np.median(curve))
    return {"C": b - d, "C_anchor": b - float(curve[ANCHOR_COLUMN]), "B": b, "D": d, "search_gain": float(curve[VSHIFT]) - d}


def validate_expected_code(code_root: Path) -> None:
    if not code_root.is_dir():
        raise AuditError(f"producer code directory does not exist: {code_root}")
    for name, expected in CODE_SHA256.items():
        check_file(code_root / name, expected, f"producer source {name}")


def validate_complete_file_manifest(root: Path, manifest: dict[str, Any]) -> int:
    if not isinstance(manifest, dict) or not manifest:
        raise AuditError("all_files_sha256.json must be a nonempty relative-path to SHA256 object")
    keys = set(manifest)
    actual_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name != "all_files_sha256.json"}
    if keys != actual_files:
        raise AuditError(f"all-files path set mismatch: listed={len(keys)} actual={len(actual_files)} missing={len(keys-actual_files)} unlisted={len(actual_files-keys)}")
    for rel, expected in manifest.items():
        path = safe_artifact(root, rel)
        check_file(path, str(expected), f"all-files artifact {rel}")
    return len(keys)


def load_npz(path: Path, expected_sha: str, label: str) -> dict[str, np.ndarray]:
    check_file(path, expected_sha, label)
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.run_root.resolve(strict=True)
    out = args.outdir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite independent attempt: {out}")
    out.mkdir(parents=True, exist_ok=True)
    if str(root) != AUX_RUN_ROOT:
        raise AuditError(f"run root differs from the one sealed auxiliary execution: {root}")

    # Validate immutable producer and source seals before parsing any metric values.
    protocol_path = root / "evaluation_protocol.json"
    row_path = root / "evaluation_rows.jsonl"
    result_path = root / "evaluation_results.jsonl"
    status_path = root / "run_status.json"
    seal_path = root / "code_seal.json"
    allfiles_path = root / "all_files_sha256.json"
    inventory_path = root / "source_artifact_inventory.json"
    spec_path = args.spec.resolve(strict=True)
    check_file(spec_path, SPEC_SHA256, "frozen random-specificity spec")
    protocol = read_json(protocol_path)
    status = read_json(status_path)
    code_seal = read_json(seal_path)
    allfiles = read_json(allfiles_path)
    inventory = read_json(inventory_path)
    if protocol.get("status") != "frozen_pre_forward" or protocol.get("experiment_id") != "confirm-random-specificity-20261001":
        raise AuditError("producer protocol identity/status differs from the frozen experiment")
    row_sha = sha256_file(row_path)
    protocol_sha = sha256_file(protocol_path)
    results_sha = sha256_file(result_path)
    if protocol.get("row_manifest", {}).get("sha256") != row_sha or code_seal.get("row_manifest_sha256") != row_sha:
        raise AuditError("row plan bytes do not match protocol/code-seal bindings")
    if code_seal.get("protocol_sha256") != protocol_sha or status.get("protocol_sha256") != protocol_sha:
        raise AuditError("protocol bytes do not match code-seal/final-status bindings")
    if status.get("row_manifest_sha256") != row_sha or status.get("results_sha256") != results_sha:
        raise AuditError("final producer status does not bind exact row/result bytes")
    if protocol.get("code_sha256") != CODE_SHA256 or code_seal.get("code_sha256") != CODE_SHA256 or status.get("code_sha256") != CODE_SHA256:
        raise AuditError("producer code hash map differs from frozen reviewed sources")
    validate_expected_code(args.code_root.resolve(strict=True))

    expected_status = {
        "status": "complete", "planned_samples": 80, "completed_samples": 80,
        "failed_samples": 0, "planned_rows": 640, "completed_rows": 640,
        "score_artifacts_written": 640, "dose_contract_failed_rows": 0, "failed_rows": 0,
    }
    for key, expected in expected_status.items():
        if status.get(key) != expected:
            raise AuditError(f"final producer status {key}={status.get(key)!r}; expected {expected!r}")
    if protocol.get("spec", {}).get("sha256") != SPEC_SHA256:
        raise AuditError("protocol is not bound to the frozen random-specificity specification")
    if protocol.get("conditions") != list(CONDITIONS) or protocol.get("faces_ids") != list(FACES):
        raise AuditError("condition/face grid differs from the frozen 80x2x4 design")
    if (protocol.get("planned_samples"), protocol.get("planned_speakers"), protocol.get("planned_faces"), protocol.get("planned_rows")) != (80, 40, 2, 640):
        raise AuditError("protocol cohort/grid counts differ from frozen design")
    if protocol.get("calibration_anchor") != {"k": 3, "column": 18, "official_offset": -3}:
        raise AuditError("protocol anchor differs from frozen k=3/column18/official−3")
    if protocol.get("runtime_policy") != REQUIRED_FLAGS | {"actual_flags_recorded_at_model_and_scoring_calls": True}:
        raise AuditError("runtime flags differ from the frozen false/true/true policy")
    if protocol.get("random_operation", {}).get("tolerance") != "max(1e-6, 1e-6*target_norm)":
        raise AuditError("random operation dose tolerance changed")
    if (protocol.get("random_operation", {}).get("bounds"), protocol.get("random_operation", {}).get("bracket_start"), protocol.get("random_operation", {}).get("bisections")) != ([0.0, 64.0], 1.0, 64):
        raise AuditError("random operation scalar-search policy changed")
    if protocol.get("random_operation", {}).get("failure_policy") != "retain every planned row and report frame misses/capped infeasibility":
        raise AuditError("random dose failure policy differs from spec")

    source = protocol.get("source_retry3", {})
    fixed_source = {
        "protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "row_manifest_sha256": SOURCE_ROWS_SHA256,
        "results_sha256": SOURCE_RESULTS_SHA256,
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "calibration_summary_sha256": CAL_SUMMARY_SHA256,
        "calibration_axes_sha256": CAL_AXES_SHA256,
        "context_count": 80,
    }
    for key, expected in fixed_source.items():
        if source.get(key) != expected:
            raise AuditError(f"retry3 lineage protocol field {key} differs from its frozen hash")
    if source.get("continuation_parent", {}).get("sha256") != CONTINUATION_PARENT_SHA256 or source.get("recovery_code_seal", {}).get("sha256") != RECOVERY_CODE_SEAL_SHA256:
        raise AuditError("protocol does not bind exact retry3 continuation/recovery seals")
    source_root = args.source_root.resolve(strict=True)
    if str(source_root) != str(Path(source.get("run_dir", "")).resolve()):
        raise AuditError("source root argument differs from sealed retry3 parent path")
    source_expected = {
        "evaluation_protocol.json": SOURCE_PROTOCOL_SHA256,
        "evaluation_rows.jsonl": SOURCE_ROWS_SHA256,
        "evaluation_results.jsonl": SOURCE_RESULTS_SHA256,
        "continuation_parent.json": CONTINUATION_PARENT_SHA256,
        "recovery_code_seal.json": RECOVERY_CODE_SEAL_SHA256,
    }
    for rel, expected in source_expected.items():
        check_file(source_root / rel, expected, f"retry3 source {rel}")
    # The sealed AUX03 inventory only contains lineage and the 80 inherited
    # context records. The producer protocol carries the input root and hashes,
    # but does not serialize these three source paths. Require the exact paths
    # as CLI inputs and bind each to the frozen hash rather than resolving an
    # absent inventory field as Path("") (which silently resolves to cwd).
    input_manifest_path = args.input_manifest.resolve(strict=True)
    cal_summary_path = args.calibration_summary.resolve(strict=True)
    cal_axes_path = args.calibration_axes.resolve(strict=True)
    check_file(input_manifest_path, INPUT_MANIFEST_SHA256, "evaluation input manifest")
    check_file(cal_summary_path, CAL_SUMMARY_SHA256, "CAL summary")
    check_file(cal_axes_path, CAL_AXES_SHA256, "CAL axes")
    if inventory.get("schema_version") != 1 or len(inventory.get("source_artifacts", [])) != 80:
        raise AuditError("source artifact inventory does not contain exactly 80 inherited contexts")
    source_inventory_sha = sha256_file(inventory_path)
    if protocol.get("source_artifact_inventory", {}).get("sha256") != source_inventory_sha or protocol.get("source_artifact_inventory", {}).get("count") != 80:
        raise AuditError("source artifact inventory does not match its protocol binding")
    for rel, expected in AUX_RUN_SEALS.items():
        check_file(root / rel, expected, f"sealed AUX03 {rel}")

    # The all-files seal and full expected cohort are checked before score arrays.
    artifact_file_count = validate_complete_file_manifest(root, allfiles)
    if artifact_file_count != 2889:
        raise AuditError(f"all-files artifact count differs from frozen AUX03 inventory: {artifact_file_count}")
    if allfiles.get("evaluation_protocol.json") != protocol_sha or allfiles.get("evaluation_rows.jsonl") != row_sha or allfiles.get("evaluation_results.jsonl") != results_sha:
        raise AuditError("all-files inventory does not bind protocol, rows, and results")
    if allfiles.get("run_status.json") != sha256_file(status_path) or allfiles.get("code_seal.json") != sha256_file(seal_path) or allfiles.get("source_artifact_inventory.json") != source_inventory_sha:
        raise AuditError("all-files inventory does not bind final status, code seal, and source inventory")

    rows = read_jsonl(row_path)
    results = read_jsonl(result_path)
    if len(rows) != 640 or len(results) != 640:
        raise AuditError(f"planned/results rows have {len(rows)}/{len(results)} items; expected 640/640")
    row_by_id = {str(r.get("row_id")): r for r in rows}
    result_by_id = {str(r.get("row_id")): r for r in results}
    if len(row_by_id) != 640 or len(result_by_id) != 640 or set(row_by_id) != set(result_by_id):
        raise AuditError("planned/result row identities are not an exact unique 640-row match")
    input_manifest = read_json(input_manifest_path)
    input_eval = [x for x in input_manifest.get("records", []) if x.get("split") == "evaluation"]
    sample_order = [str(x["id"]) for x in input_eval]
    speaker_by_sample = {str(x["id"]): str(x["speaker"]) for x in input_eval}
    if len(sample_order) != 80 or len(set(sample_order)) != 80 or len(set(speaker_by_sample.values())) != 40:
        raise AuditError("frozen input manifest is not 80 clips / 40 speakers")
    source_rows = read_jsonl(source_root / "evaluation_rows.jsonl")
    source_results = read_jsonl(source_root / "evaluation_results.jsonl")
    source_plan_by_sample = {
        str(row["sample_id"]): row for row in source_rows
        if row.get("family") == "source_axis" and row.get("source") == "natural"
        and row.get("face_id") == "primary" and row.get("condition") == "baseline"
    }
    source_result_by_key = {
        (str(row.get("sample_id")), str(row.get("face_id")), str(row.get("condition"))): row
        for row in source_results
        if row.get("family") == "source_axis" and row.get("source") == "natural"
        and row.get("condition") in {"baseline", "orth_plus"}
    }
    if len(source_plan_by_sample) != 80 or len(source_result_by_key) != 320:
        raise AuditError("retry3 frozen source plan/results lack one or more inherited baseline/orth_plus cells")
    if any(count != 2 for count in Counter(speaker_by_sample.values()).values()):
        raise AuditError("frozen cohort does not have two clips per speaker")

    rows_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    results_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, sample_id in enumerate(sample_order):
        speaker = speaker_by_sample[sample_id]
        group = [row_by_id[rid] for rid in row_by_id if row_by_id[rid].get("sample_id") == sample_id]
        if len(group) != 8:
            raise AuditError(f"{sample_id}: expected 8 planned rows, found {len(group)}")
        expected_keys = {(face, condition) for face in FACES for condition in CONDITIONS}
        got_keys = {(str(r.get("face_id")), str(r.get("condition"))) for r in group}
        if got_keys != expected_keys:
            raise AuditError(f"{sample_id}: row condition/face grid differs from the frozen 2x4 design")
        for row in group:
            if row.get("ordered_eval_index") != index or row.get("speaker_id") != speaker or row.get("split") != "evaluation" or row.get("family") != "random_specificity_aux" or row.get("source") != "natural" or row.get("audio_level") != "RAW_N" or row.get("status") != "planned":
                raise AuditError(f"{row.get('row_id')}: planned row identity/status differs from frozen contract")
            if row.get("row_id") != f"{sample_id}__{row['face_id']}__random_specificity__{row['condition']}":
                raise AuditError(f"{row.get('row_id')}: row ID is not canonical")
            if row.get("natural_input_sha256") != next(x["ref_sha256"] for x in input_eval if str(x["id"]) == sample_id):
                raise AuditError(f"{row['row_id']}: natural input SHA differs from frozen evaluation manifest")
            check_file(Path(str(row["natural_input_path"])).resolve(strict=True), str(row["natural_input_sha256"]), f"{row['row_id']} RAW-N source WAV")
            source_plan = source_plan_by_sample.get(sample_id)
            if source_plan is None or source_plan.get("natural_input_sha256") != row["natural_input_sha256"] or Path(str(source_plan.get("natural_input_path"))).resolve() != Path(str(row["natural_input_path"])).resolve():
                raise AuditError(f"{row['row_id']}: natural input path/SHA differs from sealed retry3 source plan")
            rows_by_sample[sample_id].append(row)
            result = result_by_id[str(row["row_id"])]
            identity_keys = ("ordered_eval_index", "row_id", "sample_id", "speaker_id", "split", "family", "face_id", "condition", "visual_id", "source", "audio_level", "natural_input_sha256", "latent_context_npz", "latent_context_npz_sha256", "latent_context_json", "latent_context_json_sha256", "context_npz", "context_json", "visual_npz", "visual_json", "video_path", "score_npz")
            if any(result.get(key) != row.get(key) for key in identity_keys):
                raise AuditError(f"{row['row_id']}: result row identity/artifact refs differ from frozen row plan")
            if result.get("status") != "scored" or result.get("score_computed") is not True:
                raise AuditError(f"{row['row_id']}: result row is not an ordinary scored row")
            if (result.get("anchor") or {}) != {"k": 3, "column": 18, "official_syncnet_offset": -3}:
                raise AuditError(f"{row['row_id']}: result anchor differs from frozen +3/column18/−3")
            if result.get("flags_at_forward") != [json.dumps(REQUIRED_FLAGS, sort_keys=True)]:
                raise AuditError(f"{row['row_id']}: row does not bind actual false/true/true forward flags")
            results_by_key[(sample_id, str(row["face_id"]), str(row["condition"]))] = result
    if len(results_by_key) != 640:
        raise AuditError("row grid has duplicate sample/face/condition keys")

    events = status.get("forward_events", [])
    if not events or any(event.get("flags_at_call") != REQUIRED_FLAGS for event in events):
        raise AuditError("one or more instrumented model/scoring forwards lack false/true/true flags")
    operation_counts = Counter((str(event.get("module")), str(event.get("operation"))) for event in events)
    required_event_counts = {
        ("portable_source_axis_20260930.scoring", "audio_embeddings_from_syncnet_units"): 80,
        ("portable_source_axis_20260930.scoring", "lip_embeddings_from_frames"): 640,
        ("portable_source_axis_20260930.evaluate", "_cross_distance_matrix"): 1280,
        ("portable_source_axis_20260930.wav2lip", "cache_face_features"): 2,
    }
    for event_key, count in required_event_counts.items():
        if operation_counts[event_key] != count:
            raise AuditError(f"instrumented forward call count differs for {event_key}: {operation_counts[event_key]} != {count}")
    for face in FACES:
        face_flags = status.get("face_flags", {}).get(face, {})
        if face_flags.get("flags_before_detection") != REQUIRED_FLAGS:
            raise AuditError(f"{face}: pre-detection flags differ from frozen deterministic policy")
        if face_flags.get("flags_after_override") != REQUIRED_FLAGS:
            raise AuditError(f"{face}: post-detection override flags differ from frozen deterministic policy")

    # Verify all source/inherited arrays and corrected float32-dose diagnostics.
    inventory_by_sample = {str(x["sample_id"]): x for x in inventory["source_artifacts"]}
    if set(inventory_by_sample) != set(sample_order):
        raise AuditError("source inventory sample IDs differ from the frozen evaluation cohort")
    with np.load(cal_axes_path, allow_pickle=False) as archive:
        cal_d = np.asarray(archive["d_source"], dtype=np.float64)
        cal_u = np.asarray(archive["u_level"], dtype=np.float64)
    axes_hashes = {"d_source": array_sha256(cal_d), "u_level": array_sha256(cal_u)}
    cal_summary = read_json(cal_summary_path)
    cal_anchor = cal_summary.get("global_anchor", {})
    if (int(cal_anchor.get("selected_k", 999)), int(cal_anchor.get("anchor_column", -1)), int(cal_anchor.get("official_syncnet_offset", 999))) != (3, 18, -3):
        raise AuditError("CAL summary anchor differs from the frozen k=3/column18/official−3 contract")
    if source.get("d_sha256") != axes_hashes["d_source"] or source.get("u_sha256") != axes_hashes["u_level"]:
        raise AuditError("protocol source d/u hashes differ from independently loaded CAL axes")
    if cal_summary.get("axes", {}).get("d_sha256") != axes_hashes["d_source"] or cal_summary.get("axes", {}).get("u_sha256") != axes_hashes["u_level"]:
        raise AuditError("CAL summary axis hashes differ from independently loaded CAL axes")
    dose_by_sample: dict[str, Any] = {}
    all_context_sha_checks = 0
    for sample_id in sample_order:
        source_item = inventory_by_sample[sample_id]
        if source_item.get("speaker_id") != speaker_by_sample[sample_id]:
            raise AuditError(f"{sample_id}: source inventory speaker differs from evaluation manifest")
        old_base = source_result_by_key.get((sample_id, "primary", "baseline"))
        old_plus = source_result_by_key.get((sample_id, "primary", "orth_plus"))
        if not old_base or not old_plus or old_base.get("status") != "scored" or old_plus.get("status") != "scored":
            raise AuditError(f"{sample_id}: retry3 inherited primary baseline/orth_plus was not scored")
        inherited_context = resolve_source_ref(source_root, str(old_base["context_npz"]))
        inherited_meta = resolve_source_ref(source_root, str(old_base["context_json"]))
        if source_item.get("source_context_npz") != str(inherited_context) or source_item.get("source_context_json") != str(inherited_meta):
            raise AuditError(f"{sample_id}: corrected context lineage paths differ from exact retry3 baseline row")
        if source_item.get("source_context_npz_sha256") != old_base.get("context_npz_sha256") or source_item.get("source_context_json_sha256") != old_base.get("context_json_sha256"):
            raise AuditError(f"{sample_id}: corrected context lineage hashes differ from retry3 baseline row")
        if old_plus.get("context_npz") != old_base.get("context_npz") or old_plus.get("context_json") != old_base.get("context_json") or old_plus.get("context_npz_sha256") != old_base.get("context_npz_sha256") or old_plus.get("context_json_sha256") != old_base.get("context_json_sha256"):
            raise AuditError(f"{sample_id}: retry3 baseline and orth_plus do not share the same inherited context")
        latent_rel = str(source_item["output_context_npz"])
        latent_sha = str(source_item["output_context_npz_sha256"])
        latent_json_rel = str(source_item["output_context_json"])
        latent_json_sha = str(source_item["output_context_json_sha256"])
        latent_path = safe_artifact(root, latent_rel)
        latent_json_path = safe_artifact(root, latent_json_rel)
        latent_arrays = load_npz(latent_path, latent_sha, f"{sample_id} corrected latent context")
        latent_meta = read_json(latent_json_path)
        if sha256_file(latent_json_path) != latent_json_sha or source_item.get("output_context_npz_sha256") != latent_sha:
            raise AuditError(f"{sample_id}: corrected context JSON/NPZ binding mismatch")
        if latent_meta.get("output_context_npz", {}).get("sha256") != latent_sha or latent_meta.get("source_retry3_context", {}).get("sha256") != str(source_item["source_context_npz_sha256"]):
            raise AuditError(f"{sample_id}: corrected context JSON does not bind exact latent/source inputs")
        if Path(str(latent_meta.get("source_retry3_context", {}).get("path", ""))).resolve() != inherited_context:
            raise AuditError(f"{sample_id}: corrected context JSON points at a different inherited retry3 context")
        required_latent = {
            "z_natural_fixed_float32", "z_orth_plus_float32", "natural_speech_mask_fixed_uint8",
            "calibration_d_source_float64", "calibration_u_level_float64",
            "axis_direction_random1_float64", "axis_direction_random2_float64", "axis_direction_random3_float64",
            "z_orth_plus_float32", "z_random1_float32", "z_random2_float32", "z_random3_float32",
        }
        if set(latent_arrays) != required_latent:
            raise AuditError(f"{sample_id}: latent context key set differs from frozen schema")
        z0 = np.asarray(latent_arrays["z_natural_fixed_float32"])
        orth = np.asarray(latent_arrays["z_orth_plus_float32"])
        mask = np.asarray(latent_arrays["natural_speech_mask_fixed_uint8"])
        d = np.asarray(latent_arrays["calibration_d_source_float64"])
        u = np.asarray(latent_arrays["calibration_u_level_float64"])
        if z0.dtype != np.float32 or orth.dtype != np.float32 or z0.ndim != 2 or z0.shape[1] != LATENT_DIM or orth.shape != z0.shape:
            raise AuditError(f"{sample_id}: source latents have wrong float32 shape")
        if mask.dtype != np.uint8 or mask.shape != (len(z0),) or not np.isin(mask, (0, 1)).all():
            raise AuditError(f"{sample_id}: speech mask has wrong uint8 frame domain")
        if not np.isfinite(z0).all() or not np.isfinite(orth).all() or np.any(z0 < 0) or np.any(orth < 0) or not np.array_equal(orth[mask == 0], z0[mask == 0]):
            raise AuditError(f"{sample_id}: baseline/source latent violates finite, nonnegative or inactive-copy contract")
        if d.dtype != np.float64 or u.dtype != np.float64 or d.shape != (LATENT_DIM,) or u.shape != (LATENT_DIM,) or not np.array_equal(d, cal_d) or not np.array_equal(u, cal_u):
            raise AuditError(f"{sample_id}: d/u differ from sealed CAL axes")
        source_ctx_path = Path(str(source_item["source_context_npz"])).resolve(strict=True)
        source_ctx_meta_path = Path(str(source_item["source_context_json"])).resolve(strict=True)
        check_file(source_ctx_path, str(source_item["source_context_npz_sha256"]), f"{sample_id} inherited retry3 context")
        check_file(source_ctx_meta_path, str(source_item["source_context_json_sha256"]), f"{sample_id} inherited retry3 context metadata")
        with np.load(source_ctx_path, allow_pickle=False) as source_archive:
            source_arrays = {k: source_archive[k] for k in source_archive.files}
        inherited = {
            "z_natural_fixed_float32": "natural_z_fixed_float32",
            "z_orth_plus_float32": "axis_z_orth_plus_float32",
            "natural_speech_mask_fixed_uint8": "natural_speech_mask_fixed_uint8",
            "calibration_d_source_float64": "calibration_d_source_float64",
            "calibration_u_level_float64": "calibration_u_level_float64",
            "axis_direction_random1_float64": "axis_direction_random1_float64",
            "axis_direction_random2_float64": "axis_direction_random2_float64",
            "axis_direction_random3_float64": "axis_direction_random3_float64",
        }
        for new_key, old_key in inherited.items():
            old_value = np.asarray(source_arrays[old_key])
            if array_sha256(latent_arrays[new_key]) != array_sha256(old_value):
                raise AuditError(f"{sample_id}: inherited array differs in {new_key}")
        source_array_hashes = source_item.get("source_arrays_sha256", {})
        for key, value in latent_arrays.items():
            if source_array_hashes.get(key) != array_sha256(value):
                raise AuditError(f"{sample_id}: corrected latent context array hash mismatch for {key}")
        all_context_sha_checks += len(latent_arrays)
        directions = [np.asarray(latent_arrays[f"axis_direction_random{i}_float64"], dtype=np.float64) for i in (1, 2, 3)]
        randoms = [np.asarray(latent_arrays[f"z_random{i}_float32"], dtype=np.float32) for i in (1, 2, 3)]
        for i, direction in enumerate(directions, 1):
            expected_direction = expected_random_direction(d, u, 20260929 + i)
            if not np.allclose(direction, expected_direction, rtol=1e-13, atol=1e-13):
                raise AuditError(f"{sample_id}: random{i} direction differs from frozen PCG64 seed/projection construction")
            if abs(float(direction @ d)) > 1e-8 or abs(float(direction @ u)) > 1e-8:
                raise AuditError(f"{sample_id}: random{i} direction is not orthogonal to CAL d/u")
        computed_dose = dose_contract(z0, orth, mask, d, u, directions, randoms, latent_meta.get("corrected_random_dose", {}))
        # The inventory and row-plan summaries are separately bound to the same recomputed dose result.
        for cond in ("random1", "random2", "random3"):
            qc = source_item.get("random_dose_qc", {}).get(cond, {})
            summary = source_item.get("random_dose_diagnostics", {}).get(cond, {})
            if any(summary.get(key) != value for key, value in qc.items()):
                raise AuditError(f"{sample_id}/{cond}: source dose QA and diagnostic payloads differ")
        dose_by_sample[sample_id] = computed_dose
        for row in rows_by_sample[sample_id]:
            cond = str(row["condition"])
            if cond == "orth_plus":
                expected_z = orth
            else:
                expected_z = randoms[int(cond[-1]) - 1]
            if row.get("z_sha256") != array_sha256(expected_z) or row.get("random_dose_qc") != source_item.get("random_dose_qc", {}).get(cond):
                raise AuditError(f"{row['row_id']}: row dose/latent binding differs from independently checked context")
            if row.get("latent_context_npz_sha256") != latent_sha or row.get("latent_context_json_sha256") != latent_json_sha:
                raise AuditError(f"{row['row_id']}: latent context hash binding differs")

    # Verify one RAW-N PCM/A cache per sample, all V arrays and all 640 score matrices.
    metric_by_sample: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = defaultdict(lambda: defaultdict(dict))
    verified_audio_contexts = verified_visuals = verified_scores = 0
    matrix_max_diffs: dict[str, float] = defaultdict(float)
    raw_pcm_lengths: list[int] = []
    per_sample_null: dict[str, dict[str, Any]] = {}
    detailed_records: list[dict[str, Any]] = []
    score_npzs: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = defaultdict(lambda: defaultdict(dict))
    audio_npzs: dict[str, dict[str, np.ndarray]] = {}
    for ordered_index, sample_id in enumerate(sample_order):
        sample_results = [results_by_key[(sample_id, face, condition)] for face in FACES for condition in CONDITIONS]
        audio_npz_rel = {str(r["context_npz"]) for r in sample_results}
        audio_json_rel = {str(r["context_json"]) for r in sample_results}
        if len(audio_npz_rel) != 1 or len(audio_json_rel) != 1:
            raise AuditError(f"{sample_id}: conditions do not share one fresh RAW-N A context")
        audio_npz_rel_value = next(iter(audio_npz_rel))
        audio_json_rel_value = next(iter(audio_json_rel))
        audio_npz_sha = {str(r.get("context_npz_sha256")) for r in sample_results}
        audio_json_sha = {str(r.get("context_json_sha256")) for r in sample_results}
        if len(audio_npz_sha) != 1 or len(audio_json_sha) != 1:
            raise AuditError(f"{sample_id}: shared RAW-N context hash bindings differ")
        audio_path = safe_artifact(root, audio_npz_rel_value)
        audio_meta_path = safe_artifact(root, audio_json_rel_value)
        if allfiles.get(audio_npz_rel_value) != next(iter(audio_npz_sha)) or allfiles.get(audio_json_rel_value) != next(iter(audio_json_sha)):
            raise AuditError(f"{sample_id}: shared RAW-N context NPZ/JSON do not match all-files seal")
        audio_arrays = load_npz(audio_path, next(iter(audio_npz_sha)), f"{sample_id} RAW-N audio context")
        audio_meta = read_json(audio_meta_path)
        if audio_meta.get("audio_context_npz", {}).get("sha256") != sha256_file(audio_path) or audio_meta.get("sample_id") != sample_id:
            raise AuditError(f"{sample_id}: audio context JSON does not bind exact audio NPZ")
        required_audio = {"natural_pcm16", "natural_syncnet_units_raw_float64", "natural_mfcc_raw_float64", "natural_audio_embeddings_raw_float32"}
        support_keys = {f"{face}_common_support_{name}_int32" for face in FACES for name in SUPPORTS}
        if set(audio_arrays) != required_audio | support_keys:
            raise AuditError(f"{sample_id}: audio context lacks PCM/MFCC/A arrays")
        pcm = np.asarray(audio_arrays["natural_pcm16"])
        units = np.asarray(audio_arrays["natural_syncnet_units_raw_float64"])
        mfcc = np.asarray(audio_arrays["natural_mfcc_raw_float64"])
        audio = np.asarray(audio_arrays["natural_audio_embeddings_raw_float32"])
        if pcm.dtype != np.int16 or pcm.ndim != 1 or units.dtype != np.float64 or units.shape != pcm.shape or not np.array_equal(units, pcm.astype(np.float64)):
            raise AuditError(f"{sample_id}: RAW PCM16→integer-amplitude SyncNet units contract failed")
        if mfcc.dtype != np.float64 or mfcc.ndim != 2 or not np.isfinite(mfcc).all() or audio.dtype != np.float32 or audio.ndim != 2 or audio.shape[1] != EMBEDDING_DIM or not np.isfinite(audio).all():
            raise AuditError(f"{sample_id}: RAW-N MFCC/audio embedding shape, dtype or finiteness failed")
        raw_pcm_lengths.append(len(pcm))
        input_row = next(x for x in input_eval if str(x["id"]) == sample_id)
        expected_input_sha = str(input_row["ref_sha256"])
        natural_path = Path(str(sample_results[0]["natural_input_path"])).resolve(strict=True)
        check_file(natural_path, expected_input_sha, f"{sample_id} original RAW-N WAV")
        natural_rate, original_pcm = read_pcm16_mono(natural_path)
        if natural_rate != 16000 or not np.array_equal(pcm, original_pcm):
            raise AuditError(f"{sample_id}: saved RAW-N PCM16 differs from exact original WAV payload")
        expected_window_count = len(pcm) // 640 - 5
        if len(audio) != expected_window_count:
            raise AuditError(f"{sample_id}: RAW-N A rows do not match floor(N/640)-5 window count")
        for r in sample_results:
            if r.get("natural_input_sha256") != expected_input_sha:
                raise AuditError(f"{sample_id}: results bind different natural input SHA")
        if audio_meta.get("raw_audio_cache", {}).get("input", {}).get("sha256") != expected_input_sha:
            raise AuditError(f"{sample_id}: audio cache binds a different natural input")
        if audio_meta.get("raw_audio_cache", {}).get("forward", {}).get("audio_embedding_sha256") != array_sha256(audio):
            raise AuditError(f"{sample_id}: audio cache metadata does not hash the saved A embeddings")
        if audio_meta.get("raw_audio_cache", {}).get("mfcc_sha256") != array_sha256(mfcc):
            raise AuditError(f"{sample_id}: audio cache metadata does not hash the saved MFCC array")
        audio_npzs[sample_id] = audio_arrays
        verified_audio_contexts += 1

        for face in FACES:
            face_meta = audio_meta.get("faces", {}).get(face, {})
            face_support = face_meta.get("support_indices", {})
            visual_arrays_by_cond: dict[str, np.ndarray] = {}
            visual_meta_by_cond: dict[str, dict[str, Any]] = {}
            for condition in CONDITIONS:
                result = results_by_key[(sample_id, face, condition)]
                row = row_by_id[str(result["row_id"])]
                v_path = safe_artifact(root, str(row["visual_npz"]))
                v_json_path = safe_artifact(root, str(row["visual_json"]))
                v_sha = str(result.get("visual_npz_sha256", ""))
                v_json_sha = str(result.get("visual_json_sha256", ""))
                if allfiles.get(row["visual_npz"]) != v_sha or allfiles.get(row["visual_json"]) != v_json_sha:
                    raise AuditError(f"{row['row_id']}: visual NPZ/JSON hashes differ from all-files seal")
                v_arrays = load_npz(v_path, v_sha, f"{row['row_id']} visual embeddings")
                v_meta = read_json(v_json_path)
                if v_meta.get("feature_npz_sha256") != v_sha or v_meta.get("feature_npz") != row["visual_npz"]:
                    raise AuditError(f"{row['row_id']}: visual metadata does not bind the exact feature NPZ")
                if v_meta.get("sample_id") != sample_id or v_meta.get("face_id") != face or v_meta.get("condition") != condition:
                    raise AuditError(f"{row['row_id']}: visual JSON identity differs")
                if v_meta.get("video", {}).get("sha256") != result.get("video_sha256") or allfiles.get(row["video_path"]) != result.get("video_sha256"):
                    raise AuditError(f"{row['row_id']}: retained video hash does not match visual/result bindings")
                z_key = str(row["z_key"])
                latent_arrays = load_npz(safe_artifact(root, str(row["latent_context_npz"])), str(row["latent_context_npz_sha256"]), f"{sample_id} latent context")
                expected_z = np.asarray(latent_arrays[z_key])
                visual = np.asarray(v_arrays.get("lip_embeddings_float32"))
                # The AUX producer intentionally saves only lip embeddings in
                # visual NPZs; latent identity is carried by the adjacent JSON
                # and the scored result row. Verify both against the exact
                # corrected context array selected by this frozen row.
                expected_z_sha = array_sha256(expected_z)
                if (v_meta.get("latent_sha256") != expected_z_sha
                        or result.get("latent_sha256") != expected_z_sha):
                    raise AuditError(f"{row['row_id']}: visual/result latent SHA differs from its frozen corrected context")
                decode_events = [
                    event for event in events
                    if event.get("operation") == "decode_latents"
                    and event.get("sample_id") == sample_id
                    and event.get("face_id") == face
                    and event.get("condition") == condition
                ]
                if not decode_events:
                    raise AuditError(f"{row['row_id']}: no instrumented Wav2Lip decoder forward is bound to this context row")
                if any(event.get("flags_at_call") != REQUIRED_FLAGS for event in decode_events):
                    raise AuditError(f"{row['row_id']}: decoder event flags differ from the frozen policy")
                if visual.dtype != np.float32 or visual.ndim != 2 or visual.shape[1] != EMBEDDING_DIM or not len(visual) or not np.isfinite(visual).all():
                    raise AuditError(f"{row['row_id']}: visual V array is malformed or nonfinite")
                if result.get("visual_embedding_sha256") != array_sha256(visual) or v_meta.get("lip_embedding_sha256") != array_sha256(visual):
                    raise AuditError(f"{row['row_id']}: visual embedding hash mismatch")
                visual_arrays_by_cond[condition] = visual
                visual_meta_by_cond[condition] = v_meta
                verified_visuals += 1
            query_n = min(len(audio), *(len(v) for v in visual_arrays_by_cond.values()))
            if query_n != int(face_meta.get("common_query_windows", -1)) or query_n < 1:
                raise AuditError(f"{sample_id}/{face}: common V/A query support differs from recorded complete four-arm support")
            supports = supports_for(query_n)
            if int(face_meta.get("audio_embedding_rows", -1)) != len(audio):
                raise AuditError(f"{sample_id}/{face}: full RAW-N A domain length differs")
            for name, support in supports.items():
                saved = np.asarray(face_support.get(name, []), dtype=np.int32)
                if not np.array_equal(saved, support):
                    raise AuditError(f"{sample_id}/{face}: {name} support indices differ from frozen rule")
                if not np.array_equal(np.asarray(audio_arrays[f"{face}_common_support_{name}_int32"]), support):
                    raise AuditError(f"{sample_id}/{face}: audio NPZ {name} support differs from common support")
            expected_support_sha = {name: array_sha256(value) for name, value in supports.items()}
            if face_meta.get("support_sha256") != expected_support_sha:
                raise AuditError(f"{sample_id}/{face}: support hashes differ")

            for condition in CONDITIONS:
                result = results_by_key[(sample_id, face, condition)]
                row = row_by_id[str(result["row_id"])]
                visual = visual_arrays_by_cond[condition]
                visual_unit = unit_rows(visual)
                audio_unit = unit_rows(audio)
                raw_cross, raw_zero = cross_and_zero(visual, audio)
                unit_cross, unit_zero = cross_and_zero(visual_unit, audio_unit)
                raw_lags = lag_matrix(raw_cross, raw_zero, query_n, len(audio))
                unit_lags = lag_matrix(unit_cross, unit_zero, query_n, len(audio))
                score_path = safe_artifact(root, str(row["score_npz"]))
                score_sha = str(result.get("score_npz_sha256", ""))
                if allfiles.get(row["score_npz"]) != score_sha:
                    raise AuditError(f"{row['row_id']}: score hash differs from all-files seal")
                score = load_npz(score_path, score_sha, f"{row['row_id']} score matrix")
                required_score = {
                    "cross_distance_raw_float32", "cross_distance_to_zero_raw_float32", "distance_matrix_raw_float32",
                    "cross_distance_unit_float32", "cross_distance_to_zero_unit_float32", "distance_matrix_unit_float32",
                    "common_support_full_int32", "common_support_valid_int32", "common_support_guard20_int32", "common_support_guard15_int32",
                    "lag_columns_int32", "lag_k_int32",
                }
                if set(score) != required_score:
                    raise AuditError(f"{row['row_id']}: score matrix key set differs from frozen schema")
                for name, value in (("raw cross", score["cross_distance_raw_float32"]), ("raw zero", score["cross_distance_to_zero_raw_float32"]), ("raw lag", score["distance_matrix_raw_float32"]), ("unit cross", score["cross_distance_unit_float32"]), ("unit zero", score["cross_distance_to_zero_unit_float32"]), ("unit lag", score["distance_matrix_unit_float32"])):
                    if value.dtype != np.float32 or not np.isfinite(value).all():
                        raise AuditError(f"{row['row_id']}: {name} array must be finite float32")
                if score["cross_distance_raw_float32"].shape != (len(visual), len(audio)) or score["cross_distance_unit_float32"].shape != (len(visual), len(audio)):
                    raise AuditError(f"{row['row_id']}: cross matrix must cover full V×full RAW-N A domain")
                if score["distance_matrix_raw_float32"].shape != (query_n, 31) or score["distance_matrix_unit_float32"].shape != (query_n, 31):
                    raise AuditError(f"{row['row_id']}: lag matrix must cover common query clock ×31 lags")
                max_diffs = {
                    "raw_cross": float(np.max(np.abs(score["cross_distance_raw_float32"] - raw_cross))),
                    "raw_zero": float(np.max(np.abs(score["cross_distance_to_zero_raw_float32"] - raw_zero))),
                    "raw_lag": float(np.max(np.abs(score["distance_matrix_raw_float32"] - raw_lags))),
                    "unit_cross": float(np.max(np.abs(score["cross_distance_unit_float32"] - unit_cross))),
                    "unit_zero": float(np.max(np.abs(score["cross_distance_to_zero_unit_float32"] - unit_zero))),
                    "unit_lag": float(np.max(np.abs(score["distance_matrix_unit_float32"] - unit_lags))),
                }
                for key, value in max_diffs.items():
                    matrix_max_diffs[key] = max(matrix_max_diffs[key], value)
                    if value > DISTANCE_ATOL:
                        raise AuditError(f"{row['row_id']}: independent {key} differs from sealed matrix by {value:.8g}")
                if not np.array_equal(score["lag_columns_int32"], np.arange(31, dtype=np.int32)) or not np.array_equal(score["lag_k_int32"], np.arange(-15, 16, dtype=np.int32)):
                    raise AuditError(f"{row['row_id']}: saved lag column/k domains differ from frozen 31-lag mapping")
                for name, key in (("full", "common_support_full_int32"), ("valid", "common_support_valid_int32"), ("guard20", "common_support_guard20_int32"), ("guard15", "common_support_guard15_int32")):
                    if score[key].dtype != np.int32 or not np.array_equal(score[key], supports[name]):
                        raise AuditError(f"{row['row_id']}: saved {name} support differs from common four-arm support")
                raw_metrics = metric_payload(raw_lags, supports)
                unit_metrics = metric_payload(unit_lags, supports)
                diff_raw = compare_saved_metric(result.get("raw_support_metrics", {}), raw_metrics, f"{row['row_id']} raw")
                diff_unit = compare_saved_metric(result.get("unit_support_metrics", {}), unit_metrics, f"{row['row_id']} unit")
                matrix_max_diffs["raw_metric"] = max(matrix_max_diffs["raw_metric"], diff_raw)
                matrix_max_diffs["unit_metric"] = max(matrix_max_diffs["unit_metric"], diff_unit)
                if int(result.get("common_support_n", -1)) != query_n or int(result.get("audio_row_domain_n", -1)) != len(audio) or int(result.get("visual_query_domain_n", -1)) != len(visual):
                    raise AuditError(f"{row['row_id']}: saved domain counts differ from independently loaded arrays")
                if result.get("audio_embedding_sha256") != array_sha256(audio) or result.get("visual_embedding_sha256") != array_sha256(visual):
                    raise AuditError(f"{row['row_id']}: row result does not bind exact A/V embeddings")
                metric_by_sample[sample_id][face][condition] = {"raw": raw_metrics, "unit": unit_metrics}
                score_npzs[sample_id][face][condition] = score
                verified_scores += 1
                detailed_records.append({
                    "ordered_eval_index": ordered_index, "sample_id": sample_id, "speaker_id": speaker_by_sample[sample_id],
                    "face_id": face, "condition": condition, "row_id": row["row_id"],
                    "score_sha256": score_sha, "visual_sha256": str(result["visual_npz_sha256"]),
                    "latent_forward_binding": {
                        "latent_context_npz": str(row["latent_context_npz"]),
                        "latent_context_npz_sha256": str(row["latent_context_npz_sha256"]),
                        "z_key": z_key, "selected_context_array_sha256": expected_z_sha,
                        "visual_json_latent_sha256": str(v_meta.get("latent_sha256")),
                        "result_latent_sha256": str(result.get("latent_sha256")),
                        "decoder_event_count": len(decode_events),
                        "decoder_event_keys": ["sample_id", "face_id", "condition"],
                        "scope": "context z SHA is matched to visual/result metadata and row-keyed decode events; decoder event payload itself does not contain a latent-array hash",
                    },
                    "V_rows": len(visual), "A_rows": len(audio), "common_query_rows": query_n,
                    "support_counts": {name: int(len(values)) for name, values in supports.items()},
                    "max_abs_saved_matrix_difference": max_diffs,
                    "max_abs_saved_metric_difference": {"raw": diff_raw, "unit": diff_unit},
                    "metrics": {"raw": raw_metrics, "unit": unit_metrics},
                })

    if verified_audio_contexts != 80 or verified_visuals != 640 or verified_scores != 640:
        raise AuditError("independent source/A/V/score coverage is incomplete")
    for sample_id in sample_order:
        for face in FACES:
            if set(metric_by_sample[sample_id][face]) != set(CONDITIONS):
                raise AuditError(f"{sample_id}/{face}: incomplete four-condition score set")

    # Only after all seals, lineage, strict dose, support, and matrix parity pass,
    # compute the frozen equal-dose specificity contrast and cached null.
    contrasts: dict[str, Any] = {}
    clip_records: list[dict[str, Any]] = []
    for face in FACES:
        contrasts[face] = {}
        for geometry in GEOMETRIES:
            contrasts[face][geometry] = {}
            for support in SUPPORTS:
                contrasts[face][geometry][support] = {}
                for metric in METRICS:
                    speaker_values: dict[str, list[float]] = defaultdict(list)
                    clip_values = []
                    for sample_id in sample_order:
                        cell = metric_by_sample[sample_id][face]
                        plus = cell["orth_plus"][geometry][support][metric]
                        random_values = [cell[f"random{i}"][geometry][support][metric] for i in (1, 2, 3)]
                        effect = float(plus - np.mean(random_values, dtype=np.float64))
                        speaker = speaker_by_sample[sample_id]
                        speaker_values[speaker].append(effect)
                        clip_values.append(effect)
                        clip_records.append({
                            "sample_id": sample_id, "speaker_id": speaker, "face_id": face,
                            "geometry": geometry, "support": support, "metric": metric,
                            "orth_plus": plus, "random1": random_values[0], "random2": random_values[1], "random3": random_values[2],
                            "random_mean": float(np.mean(random_values, dtype=np.float64)), "orth_plus_minus_random_mean": effect,
                        })
                    contrasts[face][geometry][support][metric] = {
                        "intended_sample_count": 80,
                        "strict_equal_dose_eligible": all(dose_by_sample[s]["all_three_random_arms_strictly_equal_dose"] for s in sample_order),
                        "clip_mean_descriptive": float(np.mean(clip_values, dtype=np.float64)),
                        **bootstrap_speakers(dict(speaker_values)),
                    }

    null_records: list[dict[str, Any]] = []
    null_summary: dict[str, Any] = {"permutation_count": NULL_DRAWS, "rng": "numpy.PCG64", "seed_rule": "20260930 + ordered_eval_index", "by_face": {}}
    null_clip_contrasts: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    permutation_hashes_by_sample: dict[str, list[str]] = {}
    for face in FACES:
        null_summary["by_face"][face] = {}
        for geometry in GEOMETRIES:
            null_summary["by_face"][face][geometry] = {}
            for sample_index, sample_id in enumerate(sample_order):
                audio = audio_npzs[sample_id]["natural_audio_embeddings_raw_float32"]
                permutation_vectors, permutation_hashes = permutations_for(len(audio), sample_index)
                prior_hashes = permutation_hashes_by_sample.setdefault(sample_id, permutation_hashes)
                if prior_hashes != permutation_hashes:
                    raise AuditError(f"{sample_id}: temporal null permutation vectors differ across faces")
                face_query = results_by_key[(sample_id, face, "orth_plus")]["common_support_n"]
                supports = supports_for(int(face_query))
                queries = supports["guard20"]
                cross_by_condition = {cond: score_npzs[sample_id][face][cond][f"cross_distance_{geometry}_float32"] for cond in CONDITIONS}
                scores_by_arm: dict[str, dict[str, np.ndarray]] = {cond: {metric: np.empty(NULL_DRAWS, dtype=np.float64) for metric in METRICS} for cond in CONDITIONS}
                if len(queries):
                    for pi_index, permutation in enumerate(permutation_vectors):
                        for cond in CONDITIONS:
                            metrics = null_metric(cross_by_condition[cond], queries, permutation)
                            for metric in METRICS:
                                scores_by_arm[cond][metric][pi_index] = metrics[metric]
                    for metric in METRICS:
                        contrast_draws = scores_by_arm["orth_plus"][metric] - np.mean(np.stack([scores_by_arm[f"random{i}"][metric] for i in (1, 2, 3)]), axis=0, dtype=np.float64)
                        null_clip_contrasts[(face, geometry, metric, sample_id)]["values"] = contrast_draws.tolist()
                    for cond in CONDITIONS:
                        arm_summary = {
                            metric: {"mean_null": float(values.mean(dtype=np.float64)), "sd_over_permutations": float(values.std(ddof=1, dtype=np.float64))}
                            for metric, values in scores_by_arm[cond].items()
                        }
                        null_records.append({
                            "sample_id": sample_id, "ordered_eval_index": sample_index, "face_id": face,
                            "geometry": geometry, "condition": cond, "query_support": "guard20",
                            "query_count": int(len(queries)), "audio_domain_count": int(len(audio)),
                            "permutation_sha256": permutation_hashes, "metrics": arm_summary,
                        })
                else:
                    null_records.append({
                        "sample_id": sample_id, "ordered_eval_index": sample_index, "face_id": face,
                        "geometry": geometry, "query_support": "guard20", "query_count": 0,
                        "audio_domain_count": int(len(audio)), "permutation_sha256": permutation_hashes,
                        "status": "too_short_guard20",
                    })
            null_summary["by_face"][face][geometry] = {}
            for metric in METRICS:
                values_by_sample: dict[str, list[float]] = {}
                complete = True
                for sample_id in sample_order:
                    raw_values = null_clip_contrasts.get((face, geometry, metric, sample_id), {}).get("values")
                    if raw_values is None or len(raw_values) != NULL_DRAWS:
                        complete = False
                        continue
                    values_by_sample[sample_id] = [float(x) for x in raw_values]
                if complete and len(values_by_sample) == 80:
                    mean_null_by_sample = {sample: float(np.mean(vals, dtype=np.float64)) for sample, vals in values_by_sample.items()}
                    observed_by_sample = {
                        sample: float(metric_by_sample[sample][face]["orth_plus"][geometry]["guard20"][metric] - np.mean([
                            metric_by_sample[sample][face][f"random{i}"][geometry]["guard20"][metric] for i in (1, 2, 3)
                        ], dtype=np.float64))
                        for sample in sample_order
                    }
                    excess_by_sample = {sample: observed_by_sample[sample] - mean_null_by_sample[sample] for sample in sample_order}
                    speaker_groups: dict[str, list[float]] = defaultdict(list)
                    null_groups: dict[str, list[float]] = defaultdict(list)
                    excess_groups: dict[str, list[float]] = defaultdict(list)
                    for sample in sample_order:
                        speaker = speaker_by_sample[sample]
                        speaker_groups[speaker].append(observed_by_sample[sample])
                        null_groups[speaker].append(mean_null_by_sample[sample])
                        excess_groups[speaker].append(excess_by_sample[sample])
                    contrasts_out = {
                        "observed_specificity": bootstrap_speakers(dict(speaker_groups)),
                        "mean_permutation_null_specificity": bootstrap_speakers(dict(null_groups)),
                        "observed_minus_mean_null_sync_excess": bootstrap_speakers(dict(excess_groups)),
                    }
                    null_summary["by_face"][face][geometry][metric] = {
                        "guard20_query_count_complete_clips": 80,
                        "permutation_hashes_shared_across_conditions_and_faces": True,
                        **contrasts_out,
                    }
                else:
                    null_summary["by_face"][face][geometry][metric] = {
                        "guard20_query_count_complete_clips": len(values_by_sample),
                        "status": "INCOMPLETE_NULL_SUPPORT",
                    }

    # Write effects only after all validation succeeded.
    detail_path = out / "independent_aux_cell_metrics.jsonl"
    with detail_path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in detailed_records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    clip_path = out / "independent_aux_effect_rows.jsonl"
    with clip_path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in clip_records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    null_path = out / "independent_aux_temporal_null.jsonl"
    with null_path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in null_records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")

    dose_totals = {}
    for arm in ("random1", "random2", "random3"):
        arm_rows = [dose_by_sample[s]["arms"][arm] for s in sample_order]
        dose_totals[arm] = {
            "active_frames": int(sum(x["pass_frames"] + x["miss_frames"] for x in arm_rows)),
            "pass_frames": int(sum(x["pass_frames"] for x in arm_rows)),
            "miss_frames": int(sum(x["miss_frames"] for x in arm_rows)),
            "capped_infeasible_frames": int(sum(x["capped_infeasible_frames"] for x in arm_rows)),
            "max_abs_actual_dose_error": float(max(x["max_abs_actual_dose_error"] for x in arm_rows)),
            "p95_clip_abs_actual_dose_error": float(np.percentile([x["p95_abs_actual_dose_error"] for x in arm_rows], 95)),
            "strict_equal_dose_eligible_clips": int(sum(x["strict_equal_dose_eligible"] for x in arm_rows)),
        }
    report = {
        "schema_version": 1,
        "manifest_type": "independent_random_specificity_aux_cpu_consumer",
        "status": "COMPLETE",
        "classification": "independent CPU seal, dose, full-matrix parity, bootstrap and temporal-null audit; no model forwards",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "consumer": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve()), "numpy_version": np.__version__, "imports_producer_ops_or_scoring": False, "device": "CPU"},
        "input_seals": {
            "run_root": str(root), "spec_sha256": SPEC_SHA256,
            "protocol_sha256": protocol_sha, "row_manifest_sha256": row_sha, "results_jsonl_sha256": results_sha,
            "run_status_sha256": sha256_file(status_path), "code_seal_sha256": sha256_file(seal_path),
            "all_files_manifest_sha256": sha256_file(allfiles_path), "all_files_artifact_count": artifact_file_count,
            "source_artifact_inventory_sha256": source_inventory_sha,
            "retry3_protocol_sha256": SOURCE_PROTOCOL_SHA256, "retry3_rows_sha256": SOURCE_ROWS_SHA256,
            "retry3_results_sha256": SOURCE_RESULTS_SHA256, "continuation_parent_sha256": CONTINUATION_PARENT_SHA256,
            "recovery_code_seal_sha256": RECOVERY_CODE_SEAL_SHA256,
            "input_manifest_path": str(input_manifest_path), "input_manifest_sha256": INPUT_MANIFEST_SHA256,
            "calibration_summary_path": str(cal_summary_path), "calibration_summary_sha256": CAL_SUMMARY_SHA256,
            "calibration_axes_path": str(cal_axes_path), "calibration_axes_sha256": CAL_AXES_SHA256,
            "producer_code_sha256": CODE_SHA256,
        },
        "verification": {
            "planned_samples": 80, "complete_samples": 80, "failed_samples": 0,
            "planned_rows": 640, "scored_rows": 640, "score_artifacts_recomputed": verified_scores,
            "visual_artifacts_verified": verified_visuals, "audio_contexts_verified": verified_audio_contexts,
            "latent_arrays_hash_checked": all_context_sha_checks,
            "unique_speakers": 40, "clips_per_speaker": {"min": 2, "max": 2},
            "all_listed_files_exist_and_match_sha": True, "exact_all_files_path_set_match": True,
            "all_forward_events_false_true_true": True, "forward_event_count": len(events),
            "deterministic_flags": REQUIRED_FLAGS,
            "detector_flag_stages_by_face": status.get("face_flags", {}),
            "anchor": {"k": 3, "column": 18, "official_offset": -3},
            "all_random_frame_doses_reconstructed_from_independent_kkt": True,
            "strict_random_dose_totals": dose_totals,
            "pcm_to_syncnet_units_exact": True,
            "full_v_by_full_a_matrices_recomputed": True,
            "max_abs_saved_matrix_and_metric_differences": dict(matrix_max_diffs),
            "comparison_tolerances": {"saved_matrices_atol": DISTANCE_ATOL, "saved_metrics_atol": METRIC_ATOL},
            "raw_pcm_sample_count_min_max": [int(min(raw_pcm_lengths)), int(max(raw_pcm_lengths))],
        },
        "estimand": {
            "contrast": "orth_plus minus the within-clip arithmetic mean of random1/random2/random3",
            "random_dose_contract": "target is actual float32 orth_plus displacement per active frame; retain all intended rows and mark any arm/frame miss",
            "aggregation": "two clip contrasts averaged within speaker, then 40 speakers equally weighted",
            "geometries": list(GEOMETRIES), "supports": list(SUPPORTS), "metrics": list(METRICS),
            "confidence_interval": "99% percentile cluster bootstrap; 20,000 PCG64 speaker resamples, seed 20260930",
            "faces": list(FACES), "full_cohort_denominator": "80 clips / 40 speakers for every contrast; no complete-case row deletion",
        },
        "effects": contrasts,
        "temporal_null": null_summary,
        "output_files": {
            "cell_metrics": detail_path.name, "clip_effect_rows": clip_path.name, "temporal_null_rows": null_path.name,
        },
    }
    report_path = out / "independent_aux_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output_files = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        if path.name != "independent_aux_output_sha256.json":
            output_files[path.relative_to(out).as_posix()] = sha256_file(path)
    output_manifest = {
        "schema_version": 1, "manifest_type": "independent_aux_consumer_output_files",
        "consumer_sha256": sha256_file(Path(__file__).resolve()),
        "summary_sha256": sha256_file(report_path), "files": output_files,
    }
    output_manifest["manifest_sha256"] = hashlib.sha256(canonical_json(output_manifest)).hexdigest()
    (out / "independent_aux_output_sha256.json").write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--calibration-summary", type=Path, required=True)
    parser.add_argument("--calibration-axes", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args)
        print(json.dumps({"status": report["status"], "summary": str(args.outdir.resolve() / "independent_aux_summary.json"), "run_protocol_sha256": report["input_seals"]["protocol_sha256"], "rows": report["verification"]["scored_rows"], "scores_recomputed": report["verification"]["score_artifacts_recomputed"]}, separators=(",", ":"), allow_nan=False))
        return 0
    except Exception as exc:
        args.outdir.mkdir(parents=True, exist_ok=True)
        failure = {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "created_utc": datetime.now(timezone.utc).isoformat(), "consumer_source_sha256": sha256_file(Path(__file__).resolve())}
        (args.outdir / "independent_aux_failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False, separators=(",", ":")), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
