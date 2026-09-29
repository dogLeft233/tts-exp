"""Pure four-cell analysis for the MFA-linear scorer-effect experiment.

The experiment has two independently controlled factors:

``G``
    The audio used to generate the Wav2Lip video frames: natural (N) or
    MFA-linear candidate (M).
``E``
    The audio attached to the final scored media: natural (N) or MFA-linear
    candidate (M).

Sync-C is oriented higher-is-better and Sync-D is oriented lower-is-better.
All contrasts below therefore use a positive-is-better convention for both
metrics.  Repeated records from one LRS3 source group are summarized at the
group level for inference; record-level summaries remain descriptive.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

from scripts.experiments.lrs3_mfa_linear_replacement.statistics import (
    deterministic_bootstrap,
    exact_sign_flip_pvalue,
)

CELL_NAMES = ("G_N_E_N", "G_M_E_N", "G_N_E_M", "G_M_E_M")

CONTRAST_CELLS = {
    "visual_under_natural_audio": ("G_M_E_N", "G_N_E_N"),
    "visual_under_mfa_audio": ("G_M_E_M", "G_N_E_M"),
    "mfa_audio_effect_on_natural_video": ("G_N_E_M", "G_N_E_N"),
    "mfa_audio_effect_on_mfa_video": ("G_M_E_M", "G_M_E_N"),
}


def _finite_score(row: Mapping[str, Any], cell: str, metric: str) -> float:
    value = row.get(cell)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing score cell {cell}")
    score = float(value.get(metric))
    if not math.isfinite(score):
        raise ValueError(f"non-finite {cell}.{metric}")
    return score


def _utility(row: Mapping[str, Any], cell: str, metric: str) -> float:
    score = _finite_score(row, cell, metric)
    return score if metric == "sync_c" else -score


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("four-cell matrix must not be empty")
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    group_order: list[str] = []
    seen_groups: set[str] = set()
    for raw in rows:
        row = dict(raw)
        sample_id = str(row.get("sample_id", ""))
        source_group = str(row.get("source_group", ""))
        if not sample_id or not source_group or sample_id in seen_ids:
            raise ValueError("matrix sample ids must be non-empty and unique")
        for cell in CELL_NAMES:
            _finite_score(row, cell, "sync_c")
            _finite_score(row, cell, "sync_d")
        seen_ids.add(sample_id)
        if source_group not in seen_groups:
            seen_groups.add(source_group)
            group_order.append(source_group)
        result.append(row)
    return result


def _group_values(
    rows: Sequence[Mapping[str, Any]],
    values: Sequence[float],
) -> tuple[list[str], list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    order: list[str] = []
    seen: set[str] = set()
    for row, value in zip(rows, values, strict=True):
        group = str(row["source_group"])
        if group not in seen:
            seen.add(group)
            order.append(group)
        grouped[group].append(float(value))
    return order, [sum(grouped[group]) / len(grouped[group]) for group in order]


def _summarize_values(
    rows: Sequence[Mapping[str, Any]],
    values: Sequence[float],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    group_ids, group_values = _group_values(rows, values)
    if len(group_values) > 24:
        raise ValueError("exact group sign-flip inference requires at most 24 source groups")
    sign_flip = exact_sign_flip_pvalue(group_values)
    bootstrap = deterministic_bootstrap(group_values, draws=10_000, seed=bootstrap_seed)
    return {
        "record_mean": sum(values) / len(values),
        "group_mean": sum(group_values) / len(group_values),
        "record_positive_count": sum(value > 0 for value in values),
        "group_positive_count": sum(value > 0 for value in group_values),
        "record_count": len(values),
        "group_count": len(group_values),
        "group_ids": group_ids,
        "group_values": group_values,
        "bootstrap_group_mean_ci95": bootstrap["ci95"],
        "sign_flip_p_value_one_sided_positive": sign_flip["p_value_one_sided"],
    }


def _contrast_values(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    metric: str,
) -> list[float]:
    return [_utility(row, left, metric) - _utility(row, right, metric) for row in rows]


def _summarize_contrast(
    rows: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    *,
    seed: int,
) -> dict[str, Any]:
    return {
        "left_cell": left,
        "right_cell": right,
        "positive_is_better": True,
        "sync_c": _summarize_values(rows, _contrast_values(rows, left, right, "sync_c"), bootstrap_seed=seed),
        "sync_d": _summarize_values(rows, _contrast_values(rows, left, right, "sync_d"), bootstrap_seed=seed + 1),
    }


def analyze_scorer_effect(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int = 20260923,
) -> dict[str, Any]:
    """Summarize the complete four-cell matrix with source-group inference."""

    matrix = _validate_rows(rows)
    group_count = len({str(row["source_group"]) for row in matrix})
    contrasts = {
        name: _summarize_contrast(matrix, left, right, seed=bootstrap_seed + index * 10)
        for index, (name, (left, right)) in enumerate(CONTRAST_CELLS.items())
    }

    natural_visual = _contrast_values(matrix, *CONTRAST_CELLS["visual_under_natural_audio"], "sync_c")
    mfa_visual = _contrast_values(matrix, *CONTRAST_CELLS["visual_under_mfa_audio"], "sync_c")
    natural_visual_d = _contrast_values(matrix, *CONTRAST_CELLS["visual_under_natural_audio"], "sync_d")
    mfa_visual_d = _contrast_values(matrix, *CONTRAST_CELLS["visual_under_mfa_audio"], "sync_d")
    interaction_c = [mfa - natural for mfa, natural in zip(mfa_visual, natural_visual, strict=True)]
    interaction_d = [mfa - natural for mfa, natural in zip(mfa_visual_d, natural_visual_d, strict=True)]
    contrasts["scorer_visual_interaction"] = {
        "interpretation": "visual contrast under MFA audio minus visual contrast under natural audio",
        "left_cell_pair": ["G_M_E_M", "G_N_E_M"],
        "right_cell_pair": ["G_M_E_N", "G_N_E_N"],
        "positive_is_better": True,
        "sync_c": _summarize_values(matrix, interaction_c, bootstrap_seed=bootstrap_seed + 50),
        "sync_d": _summarize_values(matrix, interaction_d, bootstrap_seed=bootstrap_seed + 51),
    }

    audio_effect: dict[str, Any] = {
        "interpretation": "mean MFA scoring-audio effect across the two fixed video arms",
        "positive_is_better": True,
    }
    for index, metric in enumerate(("sync_c", "sync_d")):
        on_natural = _contrast_values(matrix, "G_N_E_M", "G_N_E_N", metric)
        on_mfa = _contrast_values(matrix, "G_M_E_M", "G_M_E_N", metric)
        average = [(natural + mfa) / 2 for natural, mfa in zip(on_natural, on_mfa, strict=True)]
        audio_effect[metric] = _summarize_values(matrix, average, bootstrap_seed=bootstrap_seed + 60 + index)
    contrasts["mfa_scoring_audio_main_effect"] = audio_effect

    cell_means = {
        cell: {
            metric: sum(_finite_score(row, cell, metric) for row in matrix) / len(matrix)
            for metric in ("sync_c", "sync_d")
        }
        for cell in CELL_NAMES
    }
    return {
        "schema_version": 1,
        "record_count": len(matrix),
        "source_group_count": group_count,
        "inference_unit": "source_group_mean",
        "metric_orientation": {"sync_c": "higher_is_better", "sync_d": "lower_is_better"},
        "cell_means_record_weighted": cell_means,
        "contrasts": contrasts,
        "matrix_cells": list(CELL_NAMES),
    }
