"""Timeline rasterization, association metrics, bootstrap, and decisions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats


def build_grid(
    timestamps_s: Sequence[float] | np.ndarray,
    local_c: Sequence[float] | np.ndarray,
    error_spans: Sequence[Mapping[str, Any]],
    *,
    low_score_k: float = 1.0,
    std_ddof: int = 0,
) -> dict[str, Any]:
    times = np.asarray(timestamps_s, dtype=np.float64).reshape(-1)
    confidence = np.asarray(local_c, dtype=np.float64).reshape(-1)
    if times.shape != confidence.shape or times.size == 0:
        raise ValueError("timestamps and local confidence must be non-empty and equal length")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(confidence)):
        raise ValueError("grid inputs must be finite")
    if np.any(np.diff(times) <= 0):
        raise ValueError("grid timestamps must be strictly increasing")
    threshold = float(np.mean(confidence) - low_score_k * np.std(confidence, ddof=std_ddof))
    error_mask = np.zeros(times.shape, dtype=bool)
    for span in error_spans:
        start, end = float(span["start_s"]), float(span["end_s"])
        if not 0 <= start < end:
            raise ValueError("error span must be non-empty and nonnegative")
        error_mask |= (times >= start) & (times < end)
    low_mask = confidence < threshold
    return {
        "timestamps_s": times,
        "local_c": confidence,
        "badness": -confidence,
        "error_mask": error_mask,
        "low_sync_mask": low_mask,
        "threshold": threshold,
        "threshold_provenance": {
            "formula": "mean(local_c)-k*population_std(local_c)",
            "k": float(low_score_k),
            "std_ddof": int(std_ddof),
            "comparison": "strict_less_than",
        },
        "cell_count": int(times.size),
        "error_cell_count": int(np.sum(error_mask)),
        "low_sync_cell_count": int(np.sum(low_mask)),
    }


def _null(reason: str) -> tuple[None, str]:
    return None, reason


def compute_metrics(grid: Mapping[str, Any], *, word_error_rate: float | None = None) -> dict[str, Any]:
    error = np.asarray(grid["error_mask"], dtype=bool)
    low = np.asarray(grid["low_sync_mask"], dtype=bool)
    badness = np.asarray(grid["badness"], dtype=np.float64)
    if not (error.shape == low.shape == badness.shape):
        raise ValueError("metric arrays have different shapes")
    error_count = int(np.sum(error))
    low_count = int(np.sum(low))
    intersection = int(np.sum(error & low))
    union = int(np.sum(error | low))
    iou, iou_reason = (intersection / union, None) if union else _null("empty_union")
    recall, recall_reason = (intersection / error_count, None) if error_count else _null("no_asr_error_cells")
    precision, precision_reason = (intersection / low_count, None) if low_count else _null("no_low_sync_cells")

    spearman: float | None
    spearman_reason: str | None
    contrast: float | None
    contrast_reason: str | None
    if error_count == 0:
        spearman, spearman_reason = _null("no_asr_error_cells")
        contrast, contrast_reason = _null("no_asr_error_cells")
    elif error_count == len(error):
        spearman, spearman_reason = _null("no_correct_cells")
        contrast, contrast_reason = _null("no_correct_cells")
    elif np.ptp(badness) == 0:
        spearman, spearman_reason = _null("constant_sync_badness")
        contrast, contrast_reason = _null("constant_sync_badness")
    else:
        result = stats.spearmanr(error.astype(np.float64), badness)
        value = float(result.statistic)
        spearman, spearman_reason = (value, None) if np.isfinite(value) else _null("constant_input")
        contrast_value = float(np.mean(badness[error]) - np.mean(badness[~error]))
        contrast, contrast_reason = (contrast_value, None) if np.isfinite(contrast_value) else _null("non_finite_contrast")
    return {
        "intersection_cells": intersection,
        "union_cells": union,
        "error_cells": error_count,
        "low_sync_cells": low_count,
        "iou": iou,
        "iou_null_reason": iou_reason,
        "error_recall": recall,
        "error_recall_null_reason": recall_reason,
        "low_sync_precision": precision,
        "low_sync_precision_null_reason": precision_reason,
        "spearman_error_vs_negative_local": spearman,
        "spearman_null_reason": spearman_reason,
        "badness_contrast": contrast,
        "badness_contrast_null_reason": contrast_reason,
        "word_error_rate": word_error_rate,
        "degenerate_low_mask": low_count == 0 and float(np.std(np.asarray(grid["local_c"], dtype=np.float64), ddof=0)) == 0.0,
    }


def bootstrap_interval(values: Sequence[float], *, seed: int = 20260831, draws: int = 10000) -> dict[str, Any]:
    clean = np.asarray(values, dtype=np.float64)
    if clean.ndim != 1 or clean.size == 0 or not np.all(np.isfinite(clean)):
        return {"lower": None, "upper": None, "null_reason": "no_defined_values"}
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, clean.size, size=(int(draws), clean.size))
    estimates = np.median(clean[indices], axis=1)
    return {
        "lower": float(np.percentile(estimates, 2.5)),
        "upper": float(np.percentile(estimates, 97.5)),
        "draws": int(draws),
        "seed": int(seed),
        "unit": "whole_analyzable_sample_record",
        "null_reason": None,
    }


def _summary_stats(values: Sequence[float]) -> dict[str, Any]:
    clean = np.asarray(values, dtype=np.float64)
    if clean.size == 0:
        return {"count": 0, "mean": None, "median": None, "q1": None, "q3": None, "min": None, "max": None}
    return {
        "count": int(clean.size),
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "q1": float(np.percentile(clean, 25)),
        "q3": float(np.percentile(clean, 75)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def _arm_summary(records: Sequence[Mapping[str, Any]], *, seed: int, draws: int) -> dict[str, Any]:
    metric_names = ("iou", "error_recall", "low_sync_precision", "spearman_error_vs_negative_local", "badness_contrast", "word_error_rate")
    macro = {name: _summary_stats([float(row[name]) for row in records if row.get(name) is not None]) for name in metric_names}
    intersection = sum(int(row.get("intersection_cells", 0)) for row in records)
    union = sum(int(row.get("union_cells", 0)) for row in records)
    errors = sum(int(row.get("error_cells", 0)) for row in records)
    low = sum(int(row.get("low_sync_cells", 0)) for row in records)
    low_intersection = intersection
    micro = {
        "intersection_cells": intersection,
        "union_cells": union,
        "error_cells": errors,
        "low_sync_cells": low,
        "iou": intersection / union if union else None,
        "error_recall": low_intersection / errors if errors else None,
        "low_sync_precision": low_intersection / low if low else None,
    }
    defined_spearman = [float(row["spearman_error_vs_negative_local"]) for row in records if row.get("spearman_error_vs_negative_local") is not None]
    defined_contrast = [float(row["badness_contrast"]) for row in records if row.get("badness_contrast") is not None]
    groups = {str(row["source_group"]) for row in records if row.get("spearman_error_vs_negative_local") is not None and row.get("badness_contrast") is not None}
    return {
        "record_count": len(records),
        "analyzable_record_count": len(records),
        "source_group_count": len(groups),
        "macro": macro,
        "micro": micro,
        "bootstrap_median_spearman": bootstrap_interval(defined_spearman, seed=seed, draws=draws),
        "bootstrap_median_badness_contrast": bootstrap_interval(defined_contrast, seed=seed + 1, draws=draws),
        "defined_spearman_count": len(defined_spearman),
        "defined_badness_contrast_count": len(defined_contrast),
    }


def paired_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_sample: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        by_sample[str(record["sample_id"])][str(record["arm"])] = record
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(by_sample):
        pair = by_sample[sample_id]
        natural, tts = pair.get("natural"), pair.get("tts")
        if natural is None or tts is None:
            continue
        if natural.get("spearman_error_vs_negative_local") is None or tts.get("spearman_error_vs_negative_local") is None:
            continue
        if natural.get("badness_contrast") is None or tts.get("badness_contrast") is None:
            continue
        rows.append({
            "sample_id": sample_id,
            "source_group": str(natural["source_group"]),
            "tts_minus_natural_spearman": float(tts["spearman_error_vs_negative_local"]) - float(natural["spearman_error_vs_negative_local"]),
            "tts_minus_natural_badness_contrast": float(tts["badness_contrast"]) - float(natural["badness_contrast"]),
        })
    return rows


def aggregate_records(records: Sequence[Mapping[str, Any]], *, seed: int = 20260831, draws: int = 10000) -> dict[str, Any]:
    natural = [row for row in records if row.get("arm") == "natural"]
    tts = [row for row in records if row.get("arm") == "tts"]
    paired = paired_rows(records)
    paired_s = [row["tts_minus_natural_spearman"] for row in paired]
    paired_c = [row["tts_minus_natural_badness_contrast"] for row in paired]
    return {
        "natural": _arm_summary(natural, seed=seed, draws=draws),
        "tts": _arm_summary(tts, seed=seed, draws=draws),
        "paired": {
            "defined_pair_count": len(paired),
            "excluded_unpaired_or_undefined_count": len({str(row["sample_id"]) for row in records}) - len(paired),
            "rows": paired,
            "bootstrap_median_tts_minus_natural_spearman": bootstrap_interval(paired_s, seed=seed + 2, draws=draws),
            "bootstrap_median_tts_minus_natural_badness_contrast": bootstrap_interval(paired_c, seed=seed + 3, draws=draws),
        },
    }


def scientific_status(arm_summary: Mapping[str, Any], *, total_error_cells: int, required_groups: int = 8) -> dict[str, Any]:
    defined = min(int(arm_summary.get("defined_spearman_count", 0)), int(arm_summary.get("defined_badness_contrast_count", 0)))
    groups = int(arm_summary.get("source_group_count", 0))
    if defined < 12 or total_error_cells < 50 or groups < required_groups:
        return {"status": "INSUFFICIENT", "defined_metric_record_count": defined, "error_cells": int(total_error_cells), "source_group_count": groups}
    s_ci = arm_summary["bootstrap_median_spearman"]
    c_ci = arm_summary["bootstrap_median_badness_contrast"]
    support = float(s_ci["lower"]) > 0 and float(c_ci["lower"]) > 0
    return {
        "status": "SUPPORT" if support else "NO_SUPPORT",
        "defined_metric_record_count": defined,
        "error_cells": int(total_error_cells),
        "source_group_count": groups,
        "support_rule": "both median bootstrap 95% CI lower bounds > 0",
    }


def decisions(
    summaries: Mapping[str, Any],
    *,
    engineering_checks: Mapping[str, bool],
    expected_samples: int = 24,
    expected_arms: int = 48,
) -> dict[str, Any]:
    failed = sorted(key for key, value in engineering_checks.items() if not value)
    engineering_go = not failed and int(summaries.get("successful_arm_count", 0)) == expected_arms and int(summaries.get("sample_count", 0)) == expected_samples
    result = {
        "engineering": "GO" if engineering_go else "NO_GO",
        "engineering_failed_checks": failed,
        "scientific": {"natural": {"status": "NOT_EVALUATED"}, "tts": {"status": "NOT_EVALUATED"}},
    }
    if engineering_go:
        for arm in ("natural", "tts"):
            arm_summary = summaries[arm]
            result["scientific"][arm] = scientific_status(
                arm_summary,
                total_error_cells=int(arm_summary["micro"].get("error_cells", 0)),
            )
        statuses = {result["scientific"][arm]["status"] for arm in ("natural", "tts")}
        result["scientific_overall"] = (
            "SUPPORTED_BOTH_ARMS" if statuses == {"SUPPORT"} else
            "SUPPORTED_TTS_ONLY" if result["scientific"]["tts"]["status"] == "SUPPORT" else
            "SUPPORTED_NATURAL_ONLY" if result["scientific"]["natural"]["status"] == "SUPPORT" else
            "INSUFFICIENT" if "INSUFFICIENT" in statuses else "NO_SUPPORT"
        )
    else:
        result["scientific_overall"] = "NOT_EVALUATED"
    return result
