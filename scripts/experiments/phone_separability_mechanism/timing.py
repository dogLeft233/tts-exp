"""Independent timing and pause checks for exact-length candidates."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .features import match_occurrences


def _interval_iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    left = max(first[0], second[0])
    right = min(first[1], second[1])
    intersection = max(0.0, right - left)
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union > 0 else 0.0


def pause_spans(tokens: Sequence[Mapping[str, Any]], *, min_ms: float = 50.0) -> list[tuple[float, float]]:
    return [(float(token["start_s"]), float(token["end_s"])) for token in tokens if not bool(token.get("speech", not token.get("silence", False))) and 1000.0 * (float(token["end_s"]) - float(token["start_s"])) >= min_ms]


def compare_boundaries(source_tokens: Sequence[Mapping[str, Any]], candidate_tokens: Sequence[Mapping[str, Any]], *, pause_min_ms: float = 50.0, pause_iou_min: float = 0.50, edge_error_ms: float = 20.0) -> dict[str, Any]:
    matching = match_occurrences(source_tokens, candidate_tokens)
    errors: list[float] = []
    matched = []
    for row in matching["matches"]:
        source = source_tokens[int(row["source_index"])]
        target = candidate_tokens[int(row["target_index"])]
        if not bool(source.get("speech", not source.get("silence", False))) or not bool(target.get("speech", not target.get("silence", False))):
            continue
        start_error = abs(float(source["start_s"]) - float(target["start_s"])) * 1000.0
        end_error = abs(float(source["end_s"]) - float(target["end_s"])) * 1000.0
        errors.extend([start_error, end_error])
        matched.append({"label": row.get("label"), "source_index": row["source_index"], "target_index": row["target_index"], "start_error_ms": start_error, "end_error_ms": end_error})
    source_pauses = pause_spans(source_tokens, min_ms=pause_min_ms)
    target_pauses = pause_spans(candidate_tokens, min_ms=pause_min_ms)
    pause_rows = []
    for source_pause in source_pauses:
        overlaps = [_interval_iou(source_pause, target_pause) for target_pause in target_pauses]
        best = max(overlaps, default=0.0)
        pause_rows.append({"source": source_pause, "best_iou": best, "pass": best >= pause_iou_min})
    for target_pause in target_pauses:
        if max((_interval_iou(target_pause, source_pause) for source_pause in source_pauses), default=0.0) < pause_iou_min:
            pause_rows.append({"source": None, "target": target_pause, "new_pause": True, "pass": False})
    median = float(np.median(errors)) if errors else None
    p95 = float(np.quantile(errors, 0.95)) if errors else None
    coverage = matching["matched_count"] / max(matching["source_count"], 1)
    return {"edit_rate": matching["edit_rate"], "matched_count": matching["matched_count"], "source_count": matching["source_count"], "occurrence_coverage": coverage, "ambiguous": matching["ambiguous"], "boundary_errors_ms": matched, "boundary_median_ms": median, "boundary_p95_ms": p95, "pauses": pause_rows, "pause_pass_fraction": float(np.mean([row["pass"] for row in pause_rows])) if pause_rows else 1.0, "timing_pass": bool(matching["edit_rate"] <= 0.05 and coverage >= 0.90 and (median is not None and median <= edge_error_ms) and (p95 is not None and p95 <= 2.0 * edge_error_ms) and all(row["pass"] for row in pause_rows))}


def waveform_contract(source_pcm: np.ndarray, candidate_pcm: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    source = np.asarray(source_pcm)
    candidate = np.asarray(candidate_pcm)
    edit_mask = np.asarray(mask, dtype=np.float64)
    if source.ndim != 1 or candidate.ndim != 1 or source.shape != candidate.shape or source.shape != edit_mask.shape:
        raise ValueError("source, candidate, and mask must have equal one-dimensional shapes")
    untouched = edit_mask <= 0.0
    return {"same_length": True, "mask_zero_pcm_equal": bool(np.array_equal(source[untouched], candidate[untouched])), "finite": bool(np.all(np.isfinite(candidate.astype(np.float64)))), "pcm_range_valid": bool(np.all(candidate >= -32768) and np.all(candidate <= 32767)), "changed_sample_count": int(np.count_nonzero(source != candidate)), "pass": bool(np.array_equal(source[untouched], candidate[untouched]) and np.all(np.isfinite(candidate.astype(np.float64))))}


def evaluate_timing_controls(natural_tokens: Sequence[Mapping[str, Any]], repeat_tokens: Sequence[Mapping[str, Any]] | None, shifted_tokens: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    repeat = compare_boundaries(natural_tokens, repeat_tokens) if repeat_tokens is not None else None
    shifted = compare_boundaries(natural_tokens, shifted_tokens) if shifted_tokens is not None else None
    calibrated = repeat is not None and repeat.get("timing_pass", False) and shifted is not None and not shifted.get("timing_pass", True)
    return {"repeat": repeat, "shifted": shifted, "status": "CALIBRATED" if calibrated else "TIMING_MEASUREMENT_UNCALIBRATED", "timing_measurement_calibrated": calibrated}


__all__ = ["compare_boundaries", "evaluate_timing_controls", "pause_spans", "waveform_contract"]
