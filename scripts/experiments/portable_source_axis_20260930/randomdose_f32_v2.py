"""Corrected float32-dose random source-direction operation (v2).

This module is intentionally separate from the historical executed ops.py.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import ops

OPERATION_VERSION = "random_direction_level_orthogonal_actual_float32_dose_v2"
MAX_SCALE = 64.0
MAX_BISECTIONS = 64


def _candidate(base: np.ndarray, direction: np.ndarray, level: np.ndarray, scale: float) -> tuple[np.ndarray, float, float, float, float]:
    projected64, kkt = ops.level_orthogonal_projection(
        base, base + np.float64(scale) * direction, level,
    )
    projected32 = projected64.astype(np.float32)
    actual32_norm = float(np.linalg.norm(projected32.astype(np.float64) - base))
    projected64_norm = float(np.linalg.norm(projected64 - base))
    residual64 = float(kkt["residual64"])
    cast_constraint_bound = float(np.sum(
        np.abs(level) * np.abs(projected32.astype(np.float64) - projected64), dtype=np.float64,
    ))
    return projected32, projected64_norm, actual32_norm, residual64, cast_constraint_bound


def apply_random_direction_f32_dose_v2(
    z: np.ndarray,
    speech_mask: np.ndarray,
    direction: np.ndarray,
    u: np.ndarray,
    orth_plus: np.ndarray,
    *,
    max_scale: float = MAX_SCALE,
    max_bisections: int = MAX_BISECTIONS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """KKT-project a fixed random direction and match actual float32 dose.

    Every candidate is projected in float64, cast to float32, then measured in
    float64 from the exact saved float32 baseline. Failed active frames remain
    in the output and are explicitly marked in the returned diagnostics.
    """
    native = np.asarray(z)
    mask = np.asarray(speech_mask, dtype=bool)
    random_delta = np.asarray(direction, dtype=np.float64)
    level_axis = np.asarray(u, dtype=np.float64)
    positive = np.asarray(orth_plus)
    if native.dtype != np.float32 or native.ndim != 2 or native.shape[1] != ops.LATENT_DIM:
        raise ValueError("native latent must be float32 [frames,512]")
    if positive.dtype != np.float32 or positive.shape != native.shape or not np.isfinite(positive).all():
        raise ValueError("orth_plus must be finite float32 with the native shape")
    if mask.shape != (native.shape[0],) or random_delta.shape != (ops.LATENT_DIM,) or level_axis.shape != (ops.LATENT_DIM,):
        raise ValueError("random dose inputs have invalid shapes")
    if not np.isfinite(native).all() or np.any(native < 0) or not np.isfinite(random_delta).all() or not np.isfinite(level_axis).all():
        raise ValueError("random dose inputs must be finite; native z must be nonnegative")
    if np.any(positive < 0) or np.any(positive[~mask] != native[~mask]):
        raise ValueError("orth_plus must be nonnegative and preserve inactive frames exactly")
    if max_scale != MAX_SCALE or max_bisections != MAX_BISECTIONS:
        raise ValueError("frozen v2 scalar search bounds are [0,64] with at most 64 bisections")

    output = native.copy()
    active_indices = np.flatnonzero(mask)
    targets = np.linalg.norm(
        positive[active_indices].astype(np.float64) - native[active_indices].astype(np.float64),
        axis=1,
    )
    rows: list[dict[str, Any]] = []
    for active_position, frame_index in enumerate(active_indices):
        base = native[frame_index].astype(np.float64)
        target_norm = float(targets[active_position])
        tolerance = max(1e-6, 1e-6 * target_norm)
        if target_norm <= 1e-15:
            output[frame_index] = native[frame_index]
            rows.append({
                "frame_index": int(frame_index), "target_norm_float64": target_norm,
                "projected_norm_float64": 0.0, "actual_float32_norm_float64": 0.0,
                "projected_dose_error_float64": 0.0, "actual_dose_error_float32": 0.0,
                "target_norm": target_norm, "actual_norm": 0.0, "dose_error": 0.0,
                "tolerance": tolerance, "tolerance_met": True, "capped_infeasible": False,
                "infeasible": False, "scale": 0.0, "bracket_expansions": 0, "bisections": 0,
                "kkt_residual64": 0.0, "float32_constraint_abs": 0.0,
                "float32_constraint_abs_bound": 0.0, "zero_target_exact_copy": True,
            })
            continue

        lo = 0.0
        y_lo = native[frame_index].copy()
        norm64_lo = norm32_lo = 0.0
        residual_lo = constraint_lo = bound_lo = 0.0
        hi = 1.0
        y_hi, norm64_hi, norm32_hi, residual_hi, bound_hi = _candidate(base, random_delta, level_axis, hi)
        constraint_hi = abs(float(level_axis @ (y_hi.astype(np.float64) - base)))
        expansions = 0
        while norm32_hi < target_norm and hi < MAX_SCALE:
            lo, y_lo, norm64_lo, norm32_lo = hi, y_hi, norm64_hi, norm32_hi
            residual_lo, constraint_lo, bound_lo = residual_hi, constraint_hi, bound_hi
            hi = min(MAX_SCALE, hi * 2.0)
            y_hi, norm64_hi, norm32_hi, residual_hi, bound_hi = _candidate(base, random_delta, level_axis, hi)
            constraint_hi = abs(float(level_axis @ (y_hi.astype(np.float64) - base)))
            expansions += 1

        capped_infeasible = norm32_hi < target_norm - tolerance and hi >= MAX_SCALE
        bisections = 0
        if capped_infeasible:
            selected = (hi, y_hi, norm64_hi, norm32_hi, residual_hi, constraint_hi, bound_hi)
        else:
            best = (abs(norm32_lo - target_norm), (lo, y_lo, norm64_lo, norm32_lo, residual_lo, constraint_lo, bound_lo))
            high_candidate = (hi, y_hi, norm64_hi, norm32_hi, residual_hi, constraint_hi, bound_hi)
            if (abs(norm32_hi - target_norm), hi) < (best[0], best[1][0]):
                best = (abs(norm32_hi - target_norm), high_candidate)
            for bisections in range(1, MAX_BISECTIONS + 1):
                mid = (lo + hi) / 2.0
                y_mid, norm64_mid, norm32_mid, residual_mid, bound_mid = _candidate(base, random_delta, level_axis, mid)
                constraint_mid = abs(float(level_axis @ (y_mid.astype(np.float64) - base)))
                err_mid = abs(norm32_mid - target_norm)
                candidate = (mid, y_mid, norm64_mid, norm32_mid, residual_mid, constraint_mid, bound_mid)
                if best is None or (err_mid, mid) < (best[0], best[1][0]):
                    best = (err_mid, candidate)
                if norm32_mid < target_norm:
                    lo, y_lo, norm64_lo, norm32_lo = mid, y_mid, norm64_mid, norm32_mid
                    residual_lo, constraint_lo, bound_lo = residual_mid, constraint_mid, bound_mid
                else:
                    hi, y_hi, norm64_hi, norm32_hi = mid, y_mid, norm64_mid, norm32_mid
                    residual_hi, constraint_hi, bound_hi = residual_mid, constraint_mid, bound_mid
            selected = best[1]

        scale, selected_y, norm64, norm32, residual, constraint_abs, constraint_bound = selected
        output[frame_index] = selected_y
        actual_error = norm32 - target_norm
        rows.append({
            "frame_index": int(frame_index), "target_norm_float64": target_norm,
            "projected_norm_float64": float(norm64), "actual_float32_norm_float64": float(norm32),
            "projected_dose_error_float64": float(norm64 - target_norm),
            "actual_dose_error_float32": float(actual_error), "tolerance": float(tolerance),
            "target_norm": target_norm, "actual_norm": float(norm64),
            "dose_error": float(norm64 - target_norm),
            "tolerance_met": bool(abs(actual_error) <= tolerance),
            "infeasible": bool(capped_infeasible), "capped_infeasible": bool(capped_infeasible),
            "scale": float(scale),
            "bracket_expansions": int(expansions), "bisections": int(bisections),
            "kkt_residual64": float(residual),
            "float32_constraint_abs": float(constraint_abs),
            "float32_constraint_abs_bound": float(constraint_bound),
            "zero_target_exact_copy": False,
        })

    errors = np.asarray([row["actual_dose_error_float32"] for row in rows], dtype=np.float64)
    projected_errors = np.asarray([row["projected_dose_error_float64"] for row in rows], dtype=np.float64)
    scales = np.asarray([row["scale"] for row in rows], dtype=np.float64)
    projected_norms = np.asarray([row["projected_norm_float64"] for row in rows], dtype=np.float64)
    infeasible = np.asarray([row["infeasible"] for row in rows], dtype=bool)
    target_values = np.asarray([row["target_norm_float64"] for row in rows], dtype=np.float64)
    matched = np.asarray([row["tolerance_met"] for row in rows], dtype=bool)
    output[~mask] = native[~mask]
    if not np.array_equal(output[~mask], native[~mask]):
        raise AssertionError("inactive latent frames changed")
    summary = {
        "operation_version": OPERATION_VERSION,
        "condition": "random_control_dose_matched_to_orth_plus",
        "direction_sha256": ops.array_sha256(random_delta),
        "u_level_sha256": ops.array_sha256(level_axis),
        "native_z_sha256": ops.array_sha256(native),
        "orth_plus_z_sha256": ops.array_sha256(positive),
        "output_z_sha256": ops.array_sha256(output),
        "random_direction_sha256": ops.array_sha256(random_delta),
        "random_direction_norm": float(np.linalg.norm(random_delta)),
        "random_direction_is_half_d_perp_norm": True,
        "input_z_sha256": ops.array_sha256(native),
        "inactive_exact": bool(np.array_equal(output[~mask], native[~mask])),
        "active_frames": int(len(rows)), "inactive_frames": int((~mask).sum()),
        "per_frame_target_dose_mean": float(np.mean(target_values)) if len(target_values) else 0.0,
        "per_frame_actual_dose_mean": float(np.mean(projected_norms)) if len(projected_norms) else 0.0,
        "per_frame_abs_dose_error_max": float(np.max(np.abs(projected_errors), initial=0.0)),
        "per_frame_abs_dose_error_mean": float(np.mean(np.abs(projected_errors))) if len(projected_errors) else 0.0,
        "matched_frame_count": int((~infeasible).sum()),
        "infeasible_frame_count": int(infeasible.sum()),
        "infeasible_mask_sha256": ops.array_sha256(infeasible.astype(np.uint8)),
        "scale_min": float(np.min(scales)) if len(scales) else 0.0,
        "scale_median": float(np.median(scales)) if len(scales) else 0.0,
        "scale_max": float(np.max(scales)) if len(scales) else 0.0,
        "scale_vector_sha256": ops.array_sha256(scales),
        "max_scale_cap": MAX_SCALE,
        "kkt_float64_residual_max_abs": float(max((abs(row["kkt_residual64"]) for row in rows), default=0.0)),
        "float32_level_dot_max_abs": float(max((row["float32_constraint_abs"] for row in rows), default=0.0)),
        "float32_rounding_constraint_abs_bound_max": float(max((row["float32_constraint_abs_bound"] for row in rows), default=0.0)),
        "float32_rounding_constraint_abs_max": float(max((row["float32_constraint_abs"] for row in rows), default=0.0)),
        "dose_contract_pass_frames": int(matched.sum()),
        "dose_tolerance_miss_frames": int((~matched).sum()),
        "capped_infeasible_frames": int(infeasible.sum()),
        "max_abs_actual_float32_dose_error": float(np.max(np.abs(errors), initial=0.0)),
        "p95_abs_actual_float32_dose_error": float(np.percentile(np.abs(errors), 95)) if len(errors) else 0.0,
        "target_norm_min_median_max": [float(np.min(target_values)), float(np.median(target_values)), float(np.max(target_values))] if len(target_values) else [0.0,0.0,0.0],
        "scale_min_median_p95_max": [float(np.min(scales)), float(np.median(scales)), float(np.percentile(scales,95)), float(np.max(scales))] if len(scales) else [0.0,0.0,0.0,0.0],
        "fraction_scale_outside_half_to_two": float(np.mean((scales < 0.5) | (scales > 2.0))) if len(scales) else 0.0,
        "inactive_exact_copy": bool(np.array_equal(output[~mask], native[~mask])),
        "frames": rows,
    }
    return output, summary
