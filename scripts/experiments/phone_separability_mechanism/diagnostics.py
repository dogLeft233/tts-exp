"""Mechanism-specific diagnostics kept separate from acceptance decisions."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .metrics import flatten_view_records, paired_group_bootstrap


def duration_analysis(records: Sequence[Mapping[str, Any]], *, view: str = "full", split: str | None = None) -> dict[str, Any]:
    bins = ((0.0, 60.0), (60.0, 100.0), (100.0, 160.0), (160.0, float("inf")))
    rows = [row for row in flatten_view_records(records, view) if bool(row.get("speech", True)) and (split is None or str(row.get("analysis_split")) == split) and row.get("embedding") is not None]
    result: dict[str, Any] = {}
    for low, high in bins:
        key = f"{int(low)}_{int(high) if math.isfinite(high) else 'inf'}ms"
        grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in rows:
            duration_ms = 1000.0 * float(row.get("duration_s", 0.0))
            if low <= duration_ms < high:
                grouped[str(row.get("source_group", ""))][str(row.get("condition", ""))].append(float(duration_ms))
        result[key] = {
            "n_rows": sum(sum(len(values) for values in conditions.values()) for conditions in grouped.values()),
            "n_groups": len(grouped),
            "group_duration_mean": [{"source_group": group, "natural_ms": float(np.mean(values.get("natural", []))) if values.get("natural") else None, "tts_ms": float(np.mean(values.get("tts", []))) if values.get("tts") else None} for group, values in sorted(grouped.items())],
        }
    return {"view": view, "split": split, "bins": result}


def mask_dose_audit(source_pcm: np.ndarray, output_pcm: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    source = np.asarray(source_pcm)
    output = np.asarray(output_pcm)
    edit_mask = np.asarray(mask, dtype=np.float64)
    if source.shape != output.shape or source.shape != edit_mask.shape:
        raise ValueError("source, output, and mask must have equal shape")
    source_float = source.astype(np.float64) / 32768.0
    residual = (output.astype(np.float64) - source.astype(np.float64)) / 32768.0
    editable = edit_mask > 0.0
    return {
        "sample_count": int(source.size),
        "editable_fraction": float(np.mean(editable)) if source.size else 0.0,
        "changed_sample_count": int(np.count_nonzero(source != output)),
        "changed_fraction": float(np.mean(source != output)) if source.size else 0.0,
        "residual_rms_ratio": float(np.sqrt(np.mean(residual[editable] ** 2)) / max(np.sqrt(np.mean(source_float[editable] ** 2)), 1e-12)) if np.any(editable) else 0.0,
        "mask_zero_pcm_equal": bool(np.array_equal(source[~editable], output[~editable])),
        "peak_input": float(np.max(np.abs(source_float))) if source.size else 0.0,
        "peak_output": float(np.max(np.abs(output.astype(np.float64) / 32768.0))) if output.size else 0.0,
        "manipulation_too_weak": bool(np.mean(editable) < 0.20 or (np.any(editable) and np.sqrt(np.mean(residual[editable] ** 2)) / max(np.sqrt(np.mean(source_float[editable] ** 2)), 1e-12) < 0.005)),
    }


def historical_contrasts(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition_key: str = "condition",
    value_key: str = "accuracy",
    seed: int = 20260921,
    draws: int = 10_000,
) -> dict[str, Any]:
    """Summarize already available arms without treating them as new data."""

    by_group: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        group = str(row.get("source_group", ""))
        condition = str(row.get(condition_key, ""))
        value = row.get(value_key)
        if group and condition and value is not None and math.isfinite(float(value)):
            by_group[group][condition].append(float(value))
    conditions = sorted({condition for values in by_group.values() for condition in values})
    effects: dict[str, Any] = {}
    for left_index, left in enumerate(conditions):
        for right in conditions[left_index + 1 :]:
            group_effects = {group: float(np.mean(values[left]) - np.mean(values[right])) for group, values in by_group.items() if values.get(left) and values.get(right)}
            effects[f"{left}-{right}"] = paired_group_bootstrap(group_effects, seed=seed, draws=draws)
    return {"conditions": conditions, "effects": effects, "n_groups": len(by_group), "historical_only": True}


def link_syncnet(
    phone_rows: Sequence[Mapping[str, Any]],
    sync_rows: Sequence[Mapping[str, Any]],
    *,
    phone_value: str = "margin_delta",
    sync_value: str = "sync_c_delta",
) -> dict[str, Any]:
    """Join paired rows and report a descriptive rank correlation only."""

    sync_by = {str(row.get("pair_id", row.get("sample_id", ""))): row for row in sync_rows}
    joined = []
    for row in phone_rows:
        key = str(row.get("pair_id", row.get("sample_id", "")))
        other = sync_by.get(key)
        if other is None or row.get(phone_value) is None or other.get(sync_value) is None:
            continue
        try:
            joined.append((float(row[phone_value]), float(other[sync_value])))
        except (TypeError, ValueError):
            continue
    if len(joined) < 3:
        return {"n": len(joined), "spearman_rho": None, "reason": "FEWER_THAN_3_MATCHED_PAIRS"}
    left = np.asarray([value[0] for value in joined], dtype=np.float64)
    right = np.asarray([value[1] for value in joined], dtype=np.float64)
    left_rank = np.argsort(np.argsort(left, kind="mergesort"), kind="mergesort").astype(np.float64)
    right_rank = np.argsort(np.argsort(right, kind="mergesort"), kind="mergesort").astype(np.float64)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    return {"n": len(joined), "spearman_rho": float(np.dot(left_rank, right_rank) / denominator) if denominator > 0 else None, "interpretation": "descriptive_association_not_mediation"}


__all__ = ["duration_analysis", "historical_contrasts", "link_syncnet", "mask_dose_audit"]
