from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import config


def _finite(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def cluster_bootstrap(values: Sequence[Mapping[str, Any]], field: str, *, draws: int = config.BOOTSTRAP_DRAWS, seed: int = config.SEED) -> dict[str, Any]:
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    groups: dict[str, list[float]] = defaultdict(list)
    for row in values:
        group = str(row.get("source_group", ""))
        if not group:
            raise ValueError("source_group is required for cluster bootstrap")
        groups[group].append(_finite(row[field], field))
    if not groups:
        raise ValueError("bootstrap input is empty")
    group_values = [np.asarray(items, dtype=np.float64) for _, items in sorted(groups.items())]
    group_means = np.asarray([items.mean() for items in group_values], dtype=np.float64)
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(group_values), size=(draws, len(group_values)))
    means = group_means[selected].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    records = np.asarray([item for group in group_values for item in group], dtype=np.float64)
    return {
        "field": field,
        "draws": int(draws),
        "seed": int(seed),
        "source_group_count": len(group_values),
        "record_count": int(records.size),
        "mean": float(records.mean()),
        "ci95": [float(low), float(high)],
        "lower_bound_strictly_positive": bool(low > 0.0),
    }


def validate_diagonal_rows(rows: Sequence[Mapping[str, Any]], *, expected_count: int = config.EXPECTED_RECORD_COUNT) -> list[dict[str, Any]]:
    if len(rows) != expected_count:
        raise ValueError(f"diagonal matrix requires exactly {expected_count} rows")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        group = str(row.get("source_group", ""))
        if not sample_id or not group or sample_id in seen:
            raise ValueError("diagonal rows must have unique sample ids and source groups")
        for field in ("dtw_sync_c", "dtw_sync_d", "linear_sync_c", "linear_sync_d"):
            _finite(row.get(field), field)
        seen.add(sample_id)
        result.append(dict(row))
    return result


def analyze_diagonal(rows: Sequence[Mapping[str, Any]], *, engineering_complete: bool, expected_count: int = config.EXPECTED_RECORD_COUNT, draws: int = config.BOOTSTRAP_DRAWS, seed: int = config.SEED) -> dict[str, Any]:
    matrix = validate_diagonal_rows(rows, expected_count=expected_count)
    paired = []
    for row in matrix:
        paired.append({
            "sample_id": str(row["sample_id"]),
            "source_group": str(row["source_group"]),
            "delta_C": float(row["dtw_sync_c"] - row["linear_sync_c"]),
            "delta_D": float(row["linear_sync_d"] - row["dtw_sync_d"]),
        })
    bootstrap = {
        "delta_C": cluster_bootstrap(paired, "delta_C", draws=draws, seed=seed),
        "delta_D": cluster_bootstrap(paired, "delta_D", draws=draws, seed=seed),
    }
    gate = bool(engineering_complete and all(item["lower_bound_strictly_positive"] for item in bootstrap.values()))
    decision = "DTW_TFG_ADVANTAGE" if gate else ("NO_DTW_TFG_ADVANTAGE" if engineering_complete else "BLOCKED")
    return {
        "schema_version": 1,
        "analysis": "lrs3_mfa_dtw_diagonal",
        "record_count": len(matrix),
        "engineering_complete": bool(engineering_complete),
        "decision": decision,
        "replacement_authorized": gate,
        "comparison": {
            "dtw": "DTW video + DTW audio",
            "linear": "historical MFA-linear video + MFA-linear audio",
            "delta_C": "DTW Sync-C - linear Sync-C",
            "delta_D": "linear Sync-D - DTW Sync-D",
        },
        "statistics": {
            "method": "source-group cluster bootstrap",
            "draws": int(draws),
            "seed": int(seed),
            "confidence": 0.95,
            "bootstrap": bootstrap,
        },
        "wins": {
            "C_positive_count": int(sum(item["delta_C"] > 0 for item in paired)),
            "D_positive_count": int(sum(item["delta_D"] > 0 for item in paired)),
            "joint_positive_count": int(sum(item["delta_C"] > 0 and item["delta_D"] > 0 for item in paired)),
        },
        "paired_benefits": paired,
    }
