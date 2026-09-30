"""Independent, CPU-only checks for sealed source-axis artifacts.

This module deliberately does not import ``ops`` or ``scoring``. It recomputes
the constrained projections, SyncNet distance geometry, lag summaries, anchor
reduction, and speaker-level bootstrap from artifact arrays.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CONDITIONS = (
    "baseline", "full_plus", "full_minus", "orth_plus", "orth_minus",
    "random1", "random2", "random3",
)
LEVEL_CELLS = ("q00", "q10", "q01", "q11")


class ValidationError(AssertionError):
    """An independently recomputed invariant did not hold."""


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValidationError(f"{name} must be a nonempty finite vector")
    return result.astype(np.float64, copy=False)


def _independent_projection(
    base: np.ndarray,
    target: np.ndarray,
    axis: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Solve the scalar KKT root independently by bracketing and bisection."""

    z = _finite_vector(base, "base")
    p = _finite_vector(target, "target")
    u = _finite_vector(axis, "axis")
    if z.shape != p.shape or z.shape != u.shape or np.any(z < 0):
        raise ValidationError("projection inputs must align and base must be nonnegative")
    if float(u @ u) <= 1e-24:
        raise ValidationError("projection axis is degenerate")

    def evaluate(lam: float) -> tuple[float, np.ndarray]:
        y = np.maximum(0.0, p - np.float64(lam) * u)
        return float(u @ (y - z)), y

    f0, y0 = evaluate(0.0)
    scale = max(1.0, float(np.linalg.norm(u) * np.linalg.norm(z)))
    tolerance = 1e-12 * scale
    if abs(f0) <= tolerance:
        return y0, 0.0, f0, tolerance

    if f0 > 0:
        lo, hi = 0.0, 1.0
        fhi, _ = evaluate(hi)
        for _ in range(256):
            if fhi <= 0:
                break
            hi *= 2.0
            fhi, _ = evaluate(hi)
        else:
            raise ValidationError("independent KKT root could not be bracketed")
    else:
        lo, hi = -1.0, 0.0
        flo, _ = evaluate(lo)
        for _ in range(256):
            if flo >= 0:
                break
            lo *= 2.0
            flo, _ = evaluate(lo)
        else:
            raise ValidationError("independent KKT root could not be bracketed")

    lam = 0.5 * (lo + hi)
    residual = float("inf")
    y = y0
    for _ in range(256):
        lam = 0.5 * (lo + hi)
        residual, y = evaluate(lam)
        if abs(residual) <= tolerance:
            break
        if residual > 0:
            lo = lam
        else:
            hi = lam
    if abs(residual) > tolerance:
        raise ValidationError("independent KKT bisection did not converge")
    return y, lam, residual, tolerance


def validate_strict_projection(
    base: np.ndarray,
    target: np.ndarray,
    axis: np.ndarray,
    output_float32: np.ndarray,
) -> dict[str, float | int | bool]:
    """Check projection optimality, equality, nonnegativity and cast bound."""

    z = _finite_vector(base, "base")
    p = _finite_vector(target, "target")
    u = _finite_vector(axis, "axis")
    actual = np.asarray(output_float32)
    if actual.dtype != np.float32 or actual.shape != z.shape:
        raise ValidationError("projected output must be aligned float32")
    if not np.isfinite(actual).all() or np.any(actual < 0):
        raise ValidationError("projected output must be finite and nonnegative")

    expected, lam, residual64, tolerance64 = _independent_projection(z, p, u)
    free = expected > 1e-11
    stationarity = expected - p + lam * u
    free_error = float(np.max(np.abs(stationarity[free]))) if free.any() else 0.0
    bound_error = float(np.min(stationarity[~free])) if (~free).any() else 0.0
    stationarity_tol = 2e-9 * max(1.0, float(np.max(np.abs(p))))
    if free_error > stationarity_tol or bound_error < -stationarity_tol:
        raise ValidationError("independent KKT stationarity/complementarity check failed")

    rounded = actual.astype(np.float64)
    rounding_bound = float(np.sum(np.abs(u) * np.abs(rounded - expected), dtype=np.float64))
    equality_actual = float(u @ (rounded - z))
    numerical_slack = 64.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.linalg.norm(u) * (np.linalg.norm(rounded) + np.linalg.norm(z)))
    )
    if abs(equality_actual) > abs(residual64) + rounding_bound + numerical_slack:
        raise ValidationError("float32 equality residual exceeds the KKT rounding bound")

    f32_scale = max(1.0, float(np.max(np.abs(expected))))
    if np.max(np.abs(rounded - expected)) > 2.0 * np.finfo(np.float32).eps * f32_scale:
        raise ValidationError("stored output is not the float32 KKT projection")
    return {
        "lambda": float(lam),
        "residual64": float(residual64),
        "tolerance64": float(tolerance64),
        "stationarity_free_max_abs": free_error,
        "stationarity_zero_min": bound_error,
        "float32_constraint_abs": abs(equality_actual),
        "float32_rounding_constraint_abs_bound": rounding_bound,
        "float32_rounding_slack": numerical_slack,
        "active_positive_coordinates": int(np.count_nonzero(rounded > 0)),
        "valid": True,
    }


def validate_intervention(
    base: np.ndarray,
    mask: np.ndarray,
    output: np.ndarray,
    source_axis: np.ndarray,
    level_axis: np.ndarray,
    *,
    sign: int,
    level_orthogonal: bool,
) -> dict[str, Any]:
    """Recompute every active-row source intervention and exact inactive copy."""

    z = np.asarray(base)
    m = np.asarray(mask, dtype=bool)
    out = np.asarray(output)
    d = _finite_vector(source_axis, "source axis")
    u = _finite_vector(level_axis, "level axis")
    if z.dtype != np.float32 or out.dtype != np.float32 or z.ndim != 2 or out.shape != z.shape:
        raise ValidationError("latent arrays must be aligned float32 matrices")
    if m.shape != (len(z),) or not m.any() or d.shape != (z.shape[1],) or u.shape != d.shape:
        raise ValidationError("mask or axes do not align with latent rows")
    if sign not in (-1, 1) or not np.isfinite(z).all() or np.any(z < 0):
        raise ValidationError("invalid sign or native latent")
    if not np.array_equal(out[~m], z[~m]):
        raise ValidationError("inactive latent frames were not copied bitwise")

    row_checks = []
    for index in np.flatnonzero(m):
        base_row = z[index].astype(np.float64)
        target = base_row + np.float64(sign * 0.5) * d
        if level_orthogonal:
            row_checks.append(validate_strict_projection(base_row, target, u, out[index]))
        else:
            expected = np.maximum(0.0, target).astype(np.float32)
            if not np.array_equal(out[index], expected):
                raise ValidationError(f"full-direction row {index} differs from ReLU(z ± 0.5d)")
    return {
        "active_frames": int(m.sum()),
        "inactive_frames": int((~m).sum()),
        "inactive_exact": True,
        "strict_projection_rows": len(row_checks),
        "max_abs_float32_equality": max((float(row["float32_constraint_abs"]) for row in row_checks), default=0.0),
        "max_rounding_bound": max((float(row["float32_rounding_constraint_abs_bound"]) for row in row_checks), default=0.0),
        "valid": True,
    }


def validate_random_dose(
    base: np.ndarray,
    mask: np.ndarray,
    random_output: np.ndarray,
    orth_plus: np.ndarray,
    direction: np.ndarray,
    level_axis: np.ndarray,
    diagnostics: Sequence[Mapping[str, Any]],
    *,
    max_scale: float = 64.0,
) -> dict[str, Any]:
    """Validate dose from stored float32 artifacts, including capped failures."""

    z = np.asarray(base)
    m = np.asarray(mask, dtype=bool)
    rand = np.asarray(random_output)
    plus = np.asarray(orth_plus)
    r = _finite_vector(direction, "random direction")
    u = _finite_vector(level_axis, "level axis")
    if z.dtype != np.float32 or rand.dtype != np.float32 or plus.dtype != np.float32:
        raise ValidationError("random-dose arrays must be float32")
    if z.ndim != 2 or rand.shape != z.shape or plus.shape != z.shape or m.shape != (len(z),):
        raise ValidationError("random-dose arrays are misaligned")
    if not m.any() or r.shape != (z.shape[1],) or u.shape != r.shape:
        raise ValidationError("random-dose mask or axes are invalid")
    if not np.array_equal(rand[~m], z[~m]) or not np.array_equal(plus[~m], z[~m]):
        raise ValidationError("random and source arms must copy inactive frames exactly")
    frame_indices = np.flatnonzero(m)
    if len(diagnostics) != len(frame_indices):
        raise ValidationError("random-dose diagnostics dropped or added active frames")

    tolerance_max = 0.0
    errors: list[float] = []
    infeasible_count = 0
    for index, diagnostic in zip(frame_indices, diagnostics, strict=True):
        if int(diagnostic.get("frame_index", -1)) != int(index):
            raise ValidationError("random-dose diagnostic frame order is not canonical")
        target_norm = float(np.linalg.norm(plus[index].astype(np.float64) - z[index].astype(np.float64)))
        actual_norm = float(np.linalg.norm(rand[index].astype(np.float64) - z[index].astype(np.float64)))
        tolerance = max(1e-6, 1e-6 * target_norm)
        error = actual_norm - target_norm
        errors.append(error)
        tolerance_max = max(tolerance_max, tolerance)
        scale = float(diagnostic.get("scale", float("nan")))
        infeasible = bool(diagnostic.get("infeasible", False))
        if not np.isfinite(scale) or scale < 0 or scale > max_scale:
            raise ValidationError("random multiplier is outside its frozen [0,64] range")
        if target_norm <= 1e-15:
            if scale != 0.0 or infeasible or not np.array_equal(rand[index], z[index]):
                raise ValidationError("zero-dose row must copy native latent at scale zero")
            continue
        target = z[index].astype(np.float64) + scale * r
        projected, _lam, _resid, _tol = _independent_projection(z[index], target, u)
        expected = projected.astype(np.float32)
        if not np.array_equal(rand[index], expected):
            raise ValidationError(f"random frame {index} does not match its reported projected multiplier")
        if infeasible:
            infeasible_count += 1
            if scale != max_scale:
                raise ValidationError("unmatched random dose must retain the scale-cap output")
            if abs(error) <= tolerance:
                raise ValidationError("random dose marked infeasible despite meeting the post-cast norm tolerance")
        elif abs(error) > tolerance:
            raise ValidationError("post-cast random displacement misses orth_plus dose tolerance")

    return {
        "active_frames": int(len(frame_indices)),
        "matched_frames": int(len(frame_indices) - infeasible_count),
        "infeasible_frames": int(infeasible_count),
        "postcast_abs_dose_error_max": max((abs(item) for item in errors), default=0.0),
        "dose_tolerance_max": tolerance_max,
        "valid": True,
    }


def validate_random_direction(
    direction: np.ndarray,
    source_axis: np.ndarray,
    level_axis: np.ndarray,
) -> dict[str, float | bool]:
    """Check random controls span neither fitted axis and use the frozen dose."""

    r = _finite_vector(direction, "random direction")
    d = _finite_vector(source_axis, "source axis")
    u = _finite_vector(level_axis, "level axis")
    if r.shape != d.shape or u.shape != d.shape:
        raise ValidationError("random direction and calibration axes must align")
    gram = np.column_stack((d, u))
    coefficients, *_ = np.linalg.lstsq(gram, r, rcond=None)
    span_projection_norm = float(np.linalg.norm(gram @ coefficients))
    orthogonal_residual_norm = float(np.linalg.norm(r - gram @ coefficients))
    direction_norm = float(np.linalg.norm(r))
    d_perp = d - u * (float(u @ d) / float(u @ u))
    target_norm = 0.5 * float(np.linalg.norm(d_perp))
    orthogonality_d = float(d @ r)
    orthogonality_u = float(u @ r)
    orthogonal_tolerance = 2e-9 * max(
        1.0, float(np.linalg.norm(d) * direction_norm), float(np.linalg.norm(u) * direction_norm)
    )
    dose_tolerance = max(1e-10, 1e-8 * target_norm)
    if span_projection_norm > orthogonal_tolerance:
        raise ValidationError("random direction has a component in span(d,u)")
    if abs(orthogonal_residual_norm - direction_norm) > orthogonal_tolerance:
        raise ValidationError("random direction does not lie in the orthogonal complement of span(d,u)")
    if abs(direction_norm - target_norm) > dose_tolerance:
        raise ValidationError("random direction norm does not match half the level-orthogonal source dose")
    return {
        "direction_norm": direction_norm,
        "target_norm": target_norm,
        "span_projection_norm": span_projection_norm,
        "orthogonal_residual_norm": orthogonal_residual_norm,
        "dot_source": orthogonality_d,
        "dot_level": orthogonality_u,
        "valid": True,
    }


def validate_common_support(
    support_by_condition: Mapping[str, np.ndarray],
    declared_common: np.ndarray | None = None,
    *,
    required: Sequence[str] = CONDITIONS,
) -> dict[str, Any]:
    """Check the same full-clock frame support was used for every condition."""

    if set(support_by_condition) != set(required):
        raise ValidationError("support must retain every required condition")
    arrays = {name: np.asarray(support_by_condition[name], dtype=bool) for name in required}
    lengths = {value.shape for value in arrays.values()}
    if len(lengths) != 1 or next(iter(lengths)).__len__() != 1:
        raise ValidationError("condition support masks must be aligned vectors")
    common = np.logical_and.reduce([arrays[name] for name in required])
    if any(not np.array_equal(arrays[name], common) for name in required):
        raise ValidationError("condition scoring supports differ")
    if declared_common is not None and not np.array_equal(common, np.asarray(declared_common, dtype=bool)):
        raise ValidationError("declared common support differs from the recomputed support")
    return {"support_n": int(common.sum()), "clock_n": int(common.size), "valid": True}


def recompute_distance_matrix(
    lip: np.ndarray,
    audio: np.ndarray,
    *,
    vshift: int = 15,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Recreate SyncNet's 31-lag pairwise L2 geometry with its additive eps."""

    v = np.asarray(lip, dtype=np.float32)
    a = np.asarray(audio, dtype=np.float32)
    if v.ndim != 2 or a.ndim != 2 or v.shape[1] != a.shape[1]:
        raise ValidationError("lip and audio embeddings must be aligned matrices")
    if not len(v) or len(v) != len(a) or vshift < 0 or epsilon < 0:
        raise ValidationError("embedding support must be nonempty, equal and valid")
    padded = np.pad(a, ((vshift, vshift), (0, 0)), mode="constant")
    result = np.empty((len(v), 2 * vshift + 1), dtype=np.float32)
    eps32 = np.float32(epsilon)
    for index in range(len(v)):
        candidates = padded[index : index + 2 * vshift + 1]
        diff = (v[index][None, :] - candidates).astype(np.float32, copy=False)
        diff = (diff + eps32).astype(np.float32, copy=False)
        squared = np.sum(diff * diff, axis=1, dtype=np.float32)
        result[index] = np.sqrt(squared).astype(np.float32)
    return result


def row_unit_geometry(features: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    """Apply the diagnostic per-row unit normalization independently."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1] or not np.isfinite(values).all():
        raise ValidationError("unit-geometry features must be a finite nonempty matrix")
    if epsilon <= 0:
        raise ValidationError("unit-geometry epsilon must be positive")
    norms = np.sqrt(np.sum(values * values, axis=1, keepdims=True, dtype=np.float32), dtype=np.float32)
    denominator = np.maximum(norms, np.float32(epsilon))
    return (values / denominator).astype(np.float32, copy=False)


def validate_embedding_distance_artifacts(
    lip: np.ndarray,
    audio: np.ndarray,
    raw_matrix: np.ndarray,
    unit_matrix: np.ndarray,
    *,
    vshift: int = 15,
    atol: float = 5e-4,
) -> dict[str, Any]:
    """Rebuild raw and row-unit lag matrices from the saved A/V embeddings."""

    expected_raw = recompute_distance_matrix(lip, audio, vshift=vshift)
    expected_unit = recompute_distance_matrix(
        row_unit_geometry(lip), row_unit_geometry(audio), vshift=vshift
    )
    raw = np.asarray(raw_matrix, dtype=np.float32)
    unit = np.asarray(unit_matrix, dtype=np.float32)
    if raw.shape != expected_raw.shape or unit.shape != expected_unit.shape:
        raise ValidationError("stored distance matrix shape differs from embeddings and lag domain")
    raw_error = float(np.max(np.abs(raw - expected_raw)))
    unit_error = float(np.max(np.abs(unit - expected_unit)))
    if raw_error > atol or unit_error > atol:
        raise ValidationError("stored raw/unit distance matrix differs from independent reconstruction")
    return {
        "raw_max_abs_error": raw_error,
        "unit_max_abs_error": unit_error,
        "atol": float(atol),
        "raw_summary": curve_summary(raw),
        "unit_summary": curve_summary(unit),
        "valid": True,
    }


def curve_summary(matrix: np.ndarray, *, anchor_column: int | None = None) -> dict[str, Any]:
    """Recompute SyncNet B, D, C and optional calibration-anchor geometry."""

    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1] or not np.isfinite(values).all():
        raise ValidationError("distance matrix must be finite nonempty [windows,lag]")
    curve = values.mean(axis=0, dtype=np.float32).astype(np.float64)
    minimum = float(np.min(curve))
    median = float(np.median(curve))
    result: dict[str, Any] = {
        "support_n": int(values.shape[0]),
        "curve": curve,
        "D": minimum,
        "B": median,
        "C": median - minimum,
        "best_column": int(np.flatnonzero(curve == minimum)[0]),
    }
    if abs(result["C"] - (result["B"] - result["D"])) > 1e-12:
        raise ValidationError("C != B-D")
    if anchor_column is not None:
        if not 0 <= int(anchor_column) < values.shape[1]:
            raise ValidationError("anchor column is outside the stored lag curve")
        k = int(anchor_column) - (values.shape[1] // 2)
        result.update({
            "anchor_column": int(anchor_column),
            "anchor_k": k,
            "anchor_official_offset": -k,
            "D_anchor": float(curve[int(anchor_column)]),
            "C_anchor": float(median - curve[int(anchor_column)]),
        })
    return result


def recompute_four_cell_summary(
    matrices: Mapping[str, np.ndarray],
    *,
    anchor_column: int,
) -> dict[str, Any]:
    """Recompute G/E/I from the four declared generation/evaluator cells."""

    if set(matrices) != set(LEVEL_CELLS):
        raise ValidationError("four-cell score bundle must contain q00/q10/q01/q11")
    shapes = {np.asarray(matrices[name]).shape for name in LEVEL_CELLS}
    if len(shapes) != 1:
        raise ValidationError("four-cell matrices must use one common row/lag support")
    cells = {
        name: curve_summary(np.asarray(matrices[name]), anchor_column=anchor_column)
        for name in LEVEL_CELLS
    }
    contrasts = {}
    for metric in ("C", "C_anchor", "B", "D"):
        q00, q10, q01, q11 = (float(cells[name][metric]) for name in LEVEL_CELLS)
        contrasts[metric] = {
            "G": q10 - q00,
            "E": q01 - q00,
            "I": q11 - q10 - q01 + q00,
            "native": q00,
            "processed": q11,
        }
    return {"cells": cells, "contrasts": contrasts, "common_shape": list(next(iter(shapes))), "valid": True}


def _best_anchor_k(matrix: np.ndarray, center_k: int = 3) -> int:
    summary = curve_summary(matrix)
    curve = np.asarray(summary["curve"])
    minimum = np.min(curve)
    tied_columns = np.flatnonzero(curve == minimum)
    center_col = curve.size // 2
    return min((int(column) - center_col for column in tied_columns), key=lambda k: (abs(k - center_k), k))


def reduce_calibration_anchor(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen q10 clip → speaker → global anchor reduction."""

    if not rows:
        raise ValidationError("calibration anchor has no rows")
    speaker_clips: dict[str, list[float]] = defaultdict(list)
    clip_rows = []
    for row in rows:
        if str(row.get("cell")) != "q10" or str(row.get("split", "cal")).lower() not in {"cal", "calibration"}:
            raise ValidationError("anchor inputs must be calibration q10 (G(FIXED N), A_RAW N)")
        k = _best_anchor_k(np.asarray(row["matrix"]))
        speaker = str(row["speaker_id"])
        speaker_clips[speaker].append(float(k))
        clip_rows.append({"sample_id": str(row["sample_id"]), "speaker_id": speaker, "k": k})
    if any(len(values) != 2 for values in speaker_clips.values()):
        raise ValidationError("anchor needs exactly two clips per calibration speaker")
    speaker_values = {speaker: float(np.median(values)) for speaker, values in sorted(speaker_clips.items())}
    global_median = float(np.median(list(speaker_values.values())))
    lo, hi = int(np.floor(global_median)), int(np.ceil(global_median))
    if lo == hi:
        anchor_k = lo
    else:
        anchor_k = min((lo, hi), key=lambda k: (abs(k - 3), k))
    column = anchor_k + 15
    if not 0 <= column <= 30:
        raise ValidationError("global calibration anchor lies outside the 31-lag curve")
    return {
        "clip_k": clip_rows,
        "speaker_median_k": speaker_values,
        "global_median_k": global_median,
        "anchor_k": anchor_k,
        "anchor_column": column,
        "official_offset": -anchor_k,
        "speaker_count": len(speaker_values),
        "valid": True,
    }


def reduce_two_clip_speaker_effects(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
) -> dict[str, float]:
    """Equalize clips within speaker before assigning equal speaker weight."""

    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = float(row[value_key])
        if not np.isfinite(value):
            raise ValidationError("speaker endpoint values must be finite")
        groups[str(row["speaker_id"])].append(value)
    if not groups or any(len(items) != 2 for items in groups.values()):
        raise ValidationError("bootstrap source must retain exactly two clips per speaker")
    return {speaker: float(np.mean(items, dtype=np.float64)) for speaker, items in sorted(groups.items())}


def bootstrap_speaker_mean(
    speaker_values: Mapping[str, float],
    *,
    draws: int = 20_000,
    seed: int = 20260930,
) -> dict[str, Any]:
    """PCG64 percentile bootstrap over equal-weight speaker summaries."""

    speakers = sorted(speaker_values)
    values = np.asarray([speaker_values[name] for name in speakers], dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all() or draws <= 0:
        raise ValidationError("bootstrap needs finite speaker values, at least two speakers and positive draws")
    generator = np.random.Generator(np.random.PCG64(int(seed)))
    indices = generator.integers(0, len(values), size=(int(draws), len(values)))
    means = values[indices].mean(axis=1, dtype=np.float64)
    return {
        "speakers": speakers,
        "speaker_count": len(speakers),
        "mean": float(values.mean(dtype=np.float64)),
        "ci99": [float(np.quantile(means, 0.005, method="linear")), float(np.quantile(means, 0.995, method="linear"))],
        "draws": int(draws),
        "seed": int(seed),
        "unit": "speaker",
    }


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _float_close(actual: float, expected: float, *, atol: float = 1e-10) -> bool:
    return bool(np.isclose(float(actual), float(expected), rtol=1e-10, atol=atol))


def _raw_minimum_k(curve: Sequence[float], historical_k: int = 3) -> int:
    values = np.asarray(curve, dtype=np.float64)
    if values.shape != (31,) or not np.isfinite(values).all():
        raise ValidationError("calibration q10 curve must contain 31 finite lag values")
    minimum = np.min(values)
    columns = np.flatnonzero(values == minimum)
    return min((int(column) - 15 for column in columns), key=lambda k: (abs(k - historical_k), k))


def _recompute_active_mean(latent: np.ndarray, mask: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(latent)
    active = np.asarray(mask, dtype=bool)
    if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != 512:
        raise ValidationError(f"{name} latent must be float32 [frames,512]")
    if active.shape != (len(values),) or not active.any() or not np.isfinite(values).all():
        raise ValidationError(f"{name} latent and nonempty speech mask are misaligned")
    return values[active].astype(np.float64).mean(axis=0, dtype=np.float64)


def _recompute_audio_mask(
    waveform: np.ndarray,
    mel_starts: np.ndarray,
    *,
    threshold: float,
    window_samples: int = 3200,
    mel_hop_samples: int = 200,
) -> np.ndarray:
    pcm = np.asarray(waveform, dtype=np.float32)
    starts = np.asarray(mel_starts, dtype=np.int64)
    if pcm.ndim != 1 or not len(pcm) or not np.isfinite(pcm).all() or starts.ndim != 1:
        raise ValidationError("audio mask source waveform/mel starts are invalid")
    values = []
    for start_column in starts:
        start = int(start_column) * mel_hop_samples
        segment = pcm[max(0, start):min(len(pcm), start + window_samples)].astype(np.float64)
        if len(segment) < window_samples:
            segment = np.pad(segment, (0, window_samples - len(segment)))
        values.append(float(np.sqrt(np.mean(np.square(segment), dtype=np.float64))))
    return np.asarray(values, dtype=np.float64) > float(threshold)


def validate_calibration_artifacts(
    summary_path: str | Path,
    *,
    require_all_features: bool = False,
) -> dict[str, Any]:
    """Read-only independent audit of the sealed CAL axes and q10 anchor.

    All checks are CPU-only. When all 20 feature NPZs are present, d/u are
    reconstructed from their PCM-derived masks and latents. A partial restore
    still verifies summary/axes seals and recomputes the anchor from its 20
    stored q10 curves, while clearly reporting the missing axis inputs.
    """

    path = Path(summary_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "calibration_complete" or summary.get("classification") != "sealed calibration; no evaluation effects":
        raise ValidationError("summary is not the expected sealed CAL artifact")
    if int(summary.get("inputs", {}).get("calibration_rows", -1)) != 20 or int(summary.get("inputs", {}).get("calibration_speakers", -1)) != 10:
        raise ValidationError("CAL cohort must be exactly 20 clips and 10 speakers")
    if summary.get("inputs", {}).get("input_identity_rows_match") is not True:
        raise ValidationError("source/TTS identity seal failed")

    attempts_path = path.parent / "calibration_attempts.json"
    if not attempts_path.is_file():
        raise ValidationError("attempt ledger is missing")
    attempts = json.loads(attempts_path.read_text(encoding="utf-8"))
    successful = [row for row in attempts.get("attempts", []) if row.get("status") == "succeeded"]
    if len(successful) != 1:
        raise ValidationError("attempt ledger must identify exactly one successful CAL attempt")
    attempt_artifacts = successful[0].get("artifacts", {})
    summary_sha = _file_sha256(path)
    if attempt_artifacts.get("summary_sha256") != summary_sha:
        raise ValidationError("CAL summary SHA differs from successful attempt ledger")

    axes_path = path.parent / "calibration_axes.npz"
    if not axes_path.is_file() or _file_sha256(axes_path) != summary.get("axes", {}).get("npz_sha256"):
        raise ValidationError("CAL axes archive is missing or its file hash differs")
    with np.load(axes_path, allow_pickle=False) as archive:
        if set(archive.files) != {"d_source", "u_level"}:
            raise ValidationError("CAL axes archive has unexpected arrays")
        d_source = np.asarray(archive["d_source"])
        u_level = np.asarray(archive["u_level"])
    if d_source.shape != (512,) or u_level.shape != (512,) or not np.isfinite(d_source).all() or not np.isfinite(u_level).all():
        raise ValidationError("CAL axis arrays must be finite 512-vectors")
    axes_summary = summary["axes"]
    if _array_sha256(d_source) != axes_summary.get("d_sha256") or _array_sha256(u_level) != axes_summary.get("u_sha256"):
        raise ValidationError("CAL axis array hash differs from summary")
    if not _float_close(np.linalg.norm(d_source), axes_summary["d_norm"]):
        raise ValidationError("source-axis norm differs from summary")
    if not _float_close(np.linalg.norm(u_level), axes_summary["u_norm"]):
        raise ValidationError("level-axis norm differs from summary")
    cosine = float((d_source @ u_level) / (np.linalg.norm(d_source) * np.linalg.norm(u_level)))
    if not _float_close(cosine, axes_summary["d_u_cosine"]):
        raise ValidationError("axis cosine differs from summary")

    clips = summary.get("clips", {})
    if len(clips) != 20:
        raise ValidationError("CAL clip summary must contain exactly 20 utterances")
    speaker_ids = [str(record.get("speaker_id", "")) for record in clips.values()]
    counts: dict[str, int] = defaultdict(int)
    anchor_rows = []
    clip_anchor_checks = []
    for sample_id, record in clips.items():
        speaker = str(record.get("speaker_id", ""))
        counts[speaker] += 1
        anchor = record.get("anchor", {})
        if anchor.get("sample_id") != sample_id or anchor.get("speaker_id") != speaker or anchor.get("split") != "calibration":
            raise ValidationError(f"{sample_id}: q10 anchor row identity is not bound to CAL speaker")
        score = anchor.get("score_q10", {})
        curve = np.asarray(score.get("curve", []), dtype=np.float64)
        k = _raw_minimum_k(curve)
        column = k + 15
        if int(anchor.get("best_k", 10_000)) != k or int(anchor.get("best_column", -1)) != column:
            raise ValidationError(f"{sample_id}: exact q10 argmin/tie reduction differs")
        if int(anchor.get("official_offset", 10_000)) != -k or int(score.get("optimal_offset_frames", 10_000)) != -k:
            raise ValidationError(f"{sample_id}: k-to-official-offset sign is wrong")
        if int(score.get("optimal_column", -1)) != column:
            raise ValidationError(f"{sample_id}: stored q10 optimal column differs")
        if int(anchor.get("support_n", -1)) != int(score.get("support_n", -2)):
            raise ValidationError(f"{sample_id}: q10 anchor support count differs")
        if score.get("lag_offsets_frames") != [15 - index for index in range(31)]:
            raise ValidationError(f"{sample_id}: q10 lag order/sign map differs")
        if not _float_close(float(score["C_best"]), float(score["B_median"]) - float(score["D_min"])):
            raise ValidationError(f"{sample_id}: q10 C != B-D")
        anchor_rows.append({
            "sample_id": str(sample_id), "speaker_id": speaker,
            "split": "cal", "cell": "q10", "matrix": curve[None, :].astype(np.float32),
        })
        clip_anchor_checks.append({"sample_id": str(sample_id), "speaker_id": speaker, "k": k})
    if len(counts) != 10 or any(number != 2 for number in counts.values()):
        raise ValidationError("CAL q10 anchor must contain two clips for each of ten speakers")
    reduced_anchor = reduce_calibration_anchor(anchor_rows)
    recorded = summary.get("global_anchor", {})
    if int(recorded.get("selected_k", 10_000)) != reduced_anchor["anchor_k"]:
        raise ValidationError("global CAL anchor differs from independent clip/speaker reduction")
    if int(recorded.get("anchor_column", -1)) != reduced_anchor["anchor_column"] or int(recorded.get("official_syncnet_offset", 10_000)) != -reduced_anchor["anchor_k"]:
        raise ValidationError("global CAL anchor column or official offset sign differs")
    for speaker, median in reduced_anchor["speaker_median_k"].items():
        if not _float_close(recorded.get("speaker_medians", {}).get(speaker, float("nan")), median):
            raise ValidationError(f"{speaker}: stored two-clip anchor median differs")
    if not _float_close(recorded.get("median_before_rounding", float("nan")), reduced_anchor["global_median_k"]):
        raise ValidationError("global CAL anchor median differs")

    feature_dir = path.parent / "calibration_features"
    feature_paths = {sample_id: feature_dir / f"{sample_id}.npz" for sample_id in clips}
    available = {sample_id: target for sample_id, target in feature_paths.items() if target.is_file()}
    if require_all_features and len(available) != 20:
        raise ValidationError(f"independent axis rebuild needs all 20 feature NPZs; found {len(available)}")

    per_clip_axes: dict[str, dict[str, np.ndarray]] = {}
    per_clip_check_count = 0
    for sample_id, target_path in available.items():
        record = clips[sample_id]
        with np.load(target_path, allow_pickle=False) as archive:
            required = {
                "natural_pcm16", "natural_raw_float32", "natural_fixed_float32",
                "tts_pcm16", "tts_raw_float32", "tts_fixed_float32",
                "natural_mel_starts", "tts_mel_starts", "natural_z_fixed", "tts_z_fixed",
                "natural_z_plus3db", "natural_z_minus3db", "natural_speech_mask_fixed", "tts_speech_mask_fixed",
                "anchor_lip", "anchor_audio", "anchor_dists",
            }
            if not required.issubset(set(archive.files)):
                raise ValidationError(f"{sample_id}: calibration feature archive lacks required arrays")
            data = {name: np.asarray(archive[name]) for name in required}
        for raw_key, pcm_key in (("natural_raw_float32", "natural_pcm16"), ("tts_raw_float32", "tts_pcm16")):
            expected_pcm_float = (data[pcm_key].astype(np.float64) / 32768.0).astype(np.float32)
            if data[raw_key].dtype != np.float32 or not np.array_equal(data[raw_key], expected_pcm_float):
                raise ValidationError(f"{sample_id}: PCM16-to-float waveform identity differs")
        for level in ("natural", "tts"):
            fixed = data[f"{level}_fixed_float32"]
            raw = data[f"{level}_raw_float32"]
            meta = record["natural_rms_normalization" if level == "natural" else "tts_rms_normalization"]
            fixed_rms = float(np.sqrt(np.mean(np.square(fixed.astype(np.float64)), dtype=np.float64)))
            if fixed.dtype != np.float32 or not _float_close(fixed_rms, float(meta["target_rms"]), atol=2e-7):
                raise ValidationError(f"{sample_id}: {level} fixed waveform RMS differs from frozen target")
            mask = data[f"{level}_speech_mask_fixed"]
            meta_mask = record["natural_mask" if level == "natural" else "tts_mask"]
            starts = data[f"{level}_mel_starts"]
            expected_mask = _recompute_audio_mask(fixed, starts, threshold=10.0 ** (-45.0 / 20.0))
            if mask.dtype != np.bool_ or not np.array_equal(mask, expected_mask):
                raise ValidationError(f"{sample_id}: {level} audio-only fixed RMS mask differs")
            if _array_sha256(mask.astype(np.uint8)) != meta_mask.get("mask_sha256"):
                raise ValidationError(f"{sample_id}: {level} mask hash differs from summary")
            if not np.isfinite(raw).all():
                raise ValidationError(f"{sample_id}: {level} raw waveform contains nonfinite samples")

        natural_mean = _recompute_active_mean(data["natural_z_fixed"], data["natural_speech_mask_fixed"], "natural")
        tts_mean = _recompute_active_mean(data["tts_z_fixed"], data["tts_speech_mask_fixed"], "tts")
        plus_mean = _recompute_active_mean(data["natural_z_plus3db"], data["natural_speech_mask_fixed"], "natural+3dB")
        minus_mean = _recompute_active_mean(data["natural_z_minus3db"], data["natural_speech_mask_fixed"], "natural-3dB")
        per_clip_axes[sample_id] = {"speaker": np.asarray(record["speaker_id"]), "natural": natural_mean, "tts": tts_mean, "level": (plus_mean - minus_mean) / 6.0}

        anchor = record["anchor"]
        lip = data["anchor_lip"].astype(np.float32, copy=False)
        audio = data["anchor_audio"].astype(np.float32, copy=False)
        distance = data["anchor_dists"].astype(np.float32, copy=False)
        expected_anchor = recompute_distance_matrix(lip, audio, vshift=15)
        if distance.shape != expected_anchor.shape or float(np.max(np.abs(distance - expected_anchor))) > 5e-4:
            raise ValidationError(f"{sample_id}: q10 distance matrix differs from independent A/V reconstruction")
        if _array_sha256(distance) != anchor.get("distance_matrix_sha256"):
            raise ValidationError(f"{sample_id}: q10 distance matrix hash differs")
        if _array_sha256(lip) != anchor.get("lip_embedding_sha256") or _array_sha256(audio) != anchor.get("audio_embedding_sha256"):
            raise ValidationError(f"{sample_id}: q10 lip/audio embedding hash differs")
        audio_meta = anchor["audio_frontend"]
        raw_units = data["natural_pcm16"].astype(np.float64)
        if _array_sha256(raw_units) != audio_meta.get("input_units_sha256"):
            raise ValidationError(f"{sample_id}: q10 audio embedding provenance is not RAW natural PCM16 units")
        per_clip_check_count += 1

    axes_recomputed = False
    if len(available) == 20:
        groups: dict[str, list[dict[str, np.ndarray]]] = defaultdict(list)
        for values in per_clip_axes.values():
            groups[str(values["speaker"])].append(values)
        if len(groups) != 10 or any(len(rows) != 2 for rows in groups.values()):
            raise ValidationError("axis reconstruction needs two feature clips per CAL speaker")
        speaker_n = [np.stack([row["natural"] for row in rows]).mean(axis=0, dtype=np.float64) for _, rows in sorted(groups.items())]
        speaker_t = [np.stack([row["tts"] for row in rows]).mean(axis=0, dtype=np.float64) for _, rows in sorted(groups.items())]
        speaker_u = [np.stack([row["level"] for row in rows]).mean(axis=0, dtype=np.float64) for _, rows in sorted(groups.items())]
        d_recomputed = np.stack(speaker_t).mean(axis=0, dtype=np.float64) - np.stack(speaker_n).mean(axis=0, dtype=np.float64)
        u_recomputed = np.stack(speaker_u).mean(axis=0, dtype=np.float64)
        if not np.allclose(d_recomputed, d_source, rtol=1e-10, atol=1e-11) or not np.allclose(u_recomputed, u_level, rtol=1e-10, atol=1e-11):
            raise ValidationError("independent equal-speaker axis rebuild differs from sealed d/u")
        axes_recomputed = True

    return {
        "summary_sha256": summary_sha,
        "axes_npz_sha256": _file_sha256(axes_path),
        "calibration_clip_count": len(clips),
        "calibration_speaker_count": len(counts),
        "available_feature_npz_count": len(available),
        "q10_anchor_curve_rows_recomputed": len(anchor_rows),
        "q10_av_embedding_rows_recomputed": per_clip_check_count,
        "axis_values_rebuilt_from_features": axes_recomputed,
        "source_axis_norm": float(np.linalg.norm(d_source)),
        "level_axis_norm": float(np.linalg.norm(u_level)),
        "d_u_cosine": cosine,
        "anchor_k": int(reduced_anchor["anchor_k"]),
        "anchor_column": int(reduced_anchor["anchor_column"]),
        "official_syncnet_offset": int(reduced_anchor["official_offset"]),
        "missing_feature_npz": sorted(set(feature_paths) - set(available)),
        "valid": True,
    }
