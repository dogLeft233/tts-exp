#!/usr/bin/env python3
"""Audio-independent visual comparisons for the LRS3 TTS visual audit."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from video_features import VisualSequence


@dataclass(frozen=True)
class MetricResult:
    distance: float
    valid_frames: int
    coverage: float
    path_length: int | None = None
    warp_burden: float | None = None


def _frame_distance(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def _paired_indices(
    reference: VisualSequence,
    candidate: VisualSequence,
    tolerance_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair strictly ordered frames once, without shifting either time axis."""
    if reference.timestamps_s.size == 0 or candidate.timestamps_s.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    ref_times = np.asarray(reference.timestamps_s, dtype=np.float64)
    cand_times = np.asarray(candidate.timestamps_s, dtype=np.float64)
    if np.any(np.diff(ref_times) <= 0) or np.any(np.diff(cand_times) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if tolerance_s is None:
        ref_step = float(np.median(np.diff(ref_times))) if len(ref_times) > 1 else 0.02
        cand_step = float(np.median(np.diff(cand_times))) if len(cand_times) > 1 else ref_step
        tolerance_s = 0.5 * min(ref_step, cand_step)
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance_s must be finite and non-negative")

    left: list[int] = []
    right: list[int] = []
    ref_index = 0
    cand_index = 0
    while ref_index < len(ref_times) and cand_index < len(cand_times):
        delta = float(cand_times[cand_index] - ref_times[ref_index])
        if abs(delta) <= tolerance_s:
            left.append(ref_index)
            right.append(cand_index)
            ref_index += 1
            cand_index += 1
        elif delta < -tolerance_s:
            cand_index += 1
        else:
            ref_index += 1
    return np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)


def exact_time_distance(
    reference: VisualSequence,
    candidate: VisualSequence,
    *,
    tolerance_s: float | None = None,
) -> MetricResult:
    """Compare canonical mouth shape on the original clocks without time correction."""
    ref_indices, cand_indices = _paired_indices(reference, candidate, tolerance_s)
    denominator = max(len(reference.valid), len(candidate.valid))
    if not len(ref_indices):
        return MetricResult(float("nan"), 0, 0.0)
    valid = reference.valid[ref_indices] & candidate.valid[cand_indices]
    if not valid.any():
        return MetricResult(float("nan"), 0, float(valid.sum() / denominator) if denominator else 0.0)
    distances = [
        _frame_distance(reference.canonical_mouth[ref], candidate.canonical_mouth[cand])
        for ref, cand, keep in zip(ref_indices, cand_indices, valid, strict=True)
        if keep
    ]
    return MetricResult(
        distance=float(np.median(np.asarray(distances, dtype=np.float64))),
        valid_frames=len(distances),
        coverage=float(len(distances) / denominator) if denominator else 0.0,
    )


def _sequence_values(sequence: VisualSequence) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(sequence.canonical_mouth, dtype=np.float64).reshape(len(sequence.valid), -1)
    return values, np.asarray(sequence.valid, dtype=bool)


def _dtw_with_band(
    reference: VisualSequence,
    candidate: VisualSequence,
    band_fraction: float = 0.20,
    max_run: int = 2,
) -> MetricResult:
    if not (0.0 <= band_fraction <= 1.0) or max_run < 1:
        raise ValueError("invalid DTW configuration")
    ref_all, ref_valid = _sequence_values(reference)
    cand_all, cand_valid = _sequence_values(candidate)
    ref_indices = np.flatnonzero(ref_valid)
    cand_indices = np.flatnonzero(cand_valid)
    if not len(ref_indices) or not len(cand_indices):
        return MetricResult(float("nan"), 0, 0.0)
    ref = ref_all[ref_indices]
    cand = cand_all[cand_indices]
    n, m = len(ref), len(cand)
    state_count = max_run + 1
    inf = float("inf")
    diagonal, horizontal, vertical = range(3)
    dp = np.full((n, m, 3, state_count), inf, dtype=np.float64)
    lengths = np.zeros((n, m, 3, state_count), dtype=np.int32)

    def allowed(i: int, j: int) -> bool:
        original_i = int(ref_indices[i])
        original_j = int(cand_indices[j])
        ref_denominator = max(len(reference.valid) - 1, 1)
        cand_denominator = max(len(candidate.valid) - 1, 1)
        return abs(original_i / ref_denominator - original_j / cand_denominator) <= band_fraction + 1e-12

    def best_state(values: np.ndarray) -> tuple[float, int, int]:
        flat_index = int(np.argmin(values))
        state = np.unravel_index(flat_index, values.shape)
        return float(values[state]), int(state[0]), int(state[1])

    for i in range(n):
        for j in range(m):
            if not allowed(i, j):
                continue
            cost = _frame_distance(ref[i], cand[j])
            if i == 0 and j == 0:
                dp[i, j, diagonal, 0] = cost
                lengths[i, j, diagonal, 0] = 1
                continue

            if i > 0 and j > 0:
                previous_cost, axis, run = best_state(dp[i - 1, j - 1])
                if np.isfinite(previous_cost):
                    dp[i, j, diagonal, 0] = previous_cost + cost
                    lengths[i, j, diagonal, 0] = lengths[i - 1, j - 1, axis, run] + 1

            if j > 0:
                previous_states = [(diagonal, 0)] + [(vertical, value) for value in range(1, state_count)]
                previous_values = np.asarray(
                    [dp[i, j - 1, axis, run] for axis, run in previous_states],
                    dtype=np.float64,
                )
                previous_index = int(np.argmin(previous_values))
                previous_cost = float(previous_values[previous_index])
                if np.isfinite(previous_cost):
                    previous_axis, previous_run = previous_states[previous_index]
                    dp[i, j, horizontal, 1] = previous_cost + cost
                    lengths[i, j, horizontal, 1] = lengths[i, j - 1, previous_axis, previous_run] + 1
                for run in range(1, max_run):
                    previous_cost = dp[i, j - 1, horizontal, run]
                    if np.isfinite(previous_cost):
                        dp[i, j, horizontal, run + 1] = previous_cost + cost
                        lengths[i, j, horizontal, run + 1] = lengths[i, j - 1, horizontal, run] + 1

            if i > 0:
                previous_states = [(diagonal, 0)] + [(horizontal, value) for value in range(1, state_count)]
                previous_values = np.asarray(
                    [dp[i - 1, j, axis, run] for axis, run in previous_states],
                    dtype=np.float64,
                )
                previous_index = int(np.argmin(previous_values))
                previous_cost = float(previous_values[previous_index])
                if np.isfinite(previous_cost):
                    previous_axis, previous_run = previous_states[previous_index]
                    dp[i, j, vertical, 1] = previous_cost + cost
                    lengths[i, j, vertical, 1] = lengths[i - 1, j, previous_axis, previous_run] + 1
                for run in range(1, max_run):
                    previous_cost = dp[i - 1, j, vertical, run]
                    if np.isfinite(previous_cost):
                        dp[i, j, vertical, run + 1] = previous_cost + cost
                        lengths[i, j, vertical, run + 1] = lengths[i - 1, j, vertical, run] + 1

    final_cost, axis, run = best_state(dp[-1, -1])
    if not np.isfinite(final_cost):
        return MetricResult(float("nan"), 0, 0.0)
    path_length = int(lengths[-1, -1, axis, run])
    if path_length <= 0:
        return MetricResult(float("nan"), 0, 0.0)
    valid_frames = int(min(len(ref_indices), len(cand_indices)))
    denominator = max(len(reference.valid), len(candidate.valid))
    return MetricResult(
        distance=float(final_cost / path_length),
        valid_frames=valid_frames,
        coverage=float(valid_frames / denominator) if denominator else 0.0,
        path_length=path_length,
        warp_burden=float(abs(path_length - max(n, m)) / path_length),
    )


def visual_dtw_distance(
    reference: VisualSequence,
    candidate: VisualSequence,
    *,
    band_fraction: float = 0.20,
    max_run: int = 2,
) -> MetricResult:
    """Compare mouth shape with constrained visual-only monotonic DTW."""
    return _dtw_with_band(reference, candidate, band_fraction, max_run)


def pair_metrics(
    real: VisualSequence,
    natural: VisualSequence,
    tts: VisualSequence,
    *,
    band_fraction: float = 0.20,
    max_run: int = 2,
) -> dict[str, Any]:
    exact_natural = exact_time_distance(real, natural)
    exact_tts = exact_time_distance(real, tts)
    dtw_natural = visual_dtw_distance(real, natural, band_fraction=band_fraction, max_run=max_run)
    dtw_tts = visual_dtw_distance(real, tts, band_fraction=band_fraction, max_run=max_run)
    return {
        "exact_time": {
            "natural": exact_natural.__dict__,
            "tts": exact_tts.__dict__,
            "benefit": exact_natural.distance - exact_tts.distance,
        },
        "visual_dtw": {
            "natural": dtw_natural.__dict__,
            "tts": dtw_tts.__dict__,
            "benefit": dtw_natural.distance - dtw_tts.distance,
            "band_fraction": band_fraction,
            "max_run": max_run,
        },
    }
