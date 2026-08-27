from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


CELL_NAMES = ("G_N_E_N", "G_M_E_N", "G_N_E_M", "G_M_E_M")


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def exact_sign_flip_pvalue(values: Sequence[float], *, chunk_size: int = 65536) -> dict[str, Any]:
    vector = _finite_vector(values, "benefits")
    n = int(vector.size)
    if n > 24:
        raise ValueError("exact sign-flip enumeration is restricted to n <= 24")
    observed_mean = float(np.mean(vector))
    assignment_count = 1 << n
    exceedances = 0
    for start in range(0, assignment_count, chunk_size):
        stop = min(assignment_count, start + chunk_size)
        masks = np.arange(start, stop, dtype=np.uint32)[:, None]
        bits = ((masks >> np.arange(n, dtype=np.uint32)) & 1).astype(np.float64)
        signed_means = ((1.0 - 2.0 * bits) * vector[None, :]).mean(axis=1)
        exceedances += int(np.count_nonzero(signed_means >= observed_mean))
    return {
        "n": n,
        "observed_mean": observed_mean,
        "assignment_count": assignment_count,
        "exceedance_count": exceedances,
        "p_value_one_sided": exceedances / assignment_count,
        "alternative": "mean_benefit_greater_than_zero",
    }


def deterministic_bootstrap(values: Sequence[float], *, draws: int = 10000, seed: int = 20260824) -> dict[str, Any]:
    vector = _finite_vector(values, "benefits")
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, vector.size, size=(draws, vector.size))
    means = vector[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"draws": draws, "seed": seed, "mean": float(np.mean(vector)), "ci95": [float(low), float(high)]}


def _score(row: Mapping[str, Any], cell: str, metric: str) -> float:
    cell_value = row.get(cell)
    if not isinstance(cell_value, Mapping):
        raise ValueError(f"missing cell {cell}")
    value = float(cell_value.get(metric))
    if not math.isfinite(value):
        raise ValueError(f"non-finite {cell}.{metric}")
    return value


def validate_complete_matrix(rows: Sequence[Mapping[str, Any]], *, expected_count: int = 24) -> list[dict[str, Any]]:
    if len(rows) != expected_count:
        raise ValueError(f"authoritative matrix requires exactly {expected_count} rows")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    groups: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        group = str(row.get("source_group", ""))
        if not sample_id or not group or sample_id in ids or group in groups:
            raise ValueError("matrix sample ids and source groups must be unique")
        for cell in CELL_NAMES:
            _score(row, cell, "sync_c")
            _score(row, cell, "sync_d")
        ids.add(sample_id)
        groups.add(group)
        result.append(dict(row))
    return result


def analyze_replacement(rows: Sequence[Mapping[str, Any]], *, engineering_complete: bool, alpha: float = 0.05, bootstrap_draws: int = 10000) -> dict[str, Any]:
    matrix = validate_complete_matrix(rows)
    benefit_c = [_score(row, "G_M_E_N", "sync_c") - _score(row, "G_N_E_N", "sync_c") for row in matrix]
    benefit_d = [_score(row, "G_N_E_N", "sync_d") - _score(row, "G_M_E_N", "sync_d") for row in matrix]
    tests = {"benefit_C": exact_sign_flip_pvalue(benefit_c), "benefit_D": exact_sign_flip_pvalue(benefit_d)}
    means = {"benefit_C": float(np.mean(benefit_c)), "benefit_D": float(np.mean(benefit_d))}
    p_values = {name: float(test["p_value_one_sided"]) for name, test in tests.items()}
    directional = means["benefit_C"] > 0.0 and means["benefit_D"] > 0.0
    significant = p_values["benefit_C"] <= alpha and p_values["benefit_D"] <= alpha
    if not engineering_complete:
        decision = "BLOCKED"
    else:
        decision = "GO" if directional and significant else "NO_GO"
    cell_means = {cell: {metric: float(np.mean([_score(row, cell, metric) for row in matrix])) for metric in ("sync_c", "sync_d")} for cell in CELL_NAMES}
    native_pair = {
        "sync_c": float(np.mean([_score(row, "G_M_E_M", "sync_c") - _score(row, "G_N_E_N", "sync_c") for row in matrix])),
        "sync_d": float(np.mean([_score(row, "G_M_E_M", "sync_d") - _score(row, "G_N_E_N", "sync_d") for row in matrix])),
    }
    replacement_loss = {
        "sync_c": float(np.mean([_score(row, "G_M_E_N", "sync_c") - _score(row, "G_M_E_M", "sync_c") for row in matrix])),
        "sync_d": float(np.mean([_score(row, "G_M_E_N", "sync_d") - _score(row, "G_M_E_M", "sync_d") for row in matrix])),
    }
    scorer_effect = {
        "sync_c": float(np.mean([_score(row, "G_N_E_M", "sync_c") - _score(row, "G_N_E_N", "sync_c") for row in matrix])),
        "sync_d": float(np.mean([_score(row, "G_N_E_M", "sync_d") - _score(row, "G_N_E_N", "sync_d") for row in matrix])),
    }
    return {
        "schema_version": 1,
        "record_count": len(matrix),
        "engineering_complete": engineering_complete,
        "decision": decision,
        "co_primary": {
            "means": means,
            "p_values_one_sided": p_values,
            "alpha": alpha,
            "directional_pass": directional,
            "significance_pass": significant,
            "intersection_union": True,
        },
        "benefits": {
            "benefit_C": benefit_c,
            "benefit_D": benefit_d,
            "sync_c_positive_count": int(sum(value > 0 for value in benefit_c)),
            "sync_d_positive_count": int(sum(value > 0 for value in benefit_d)),
            "joint_win_count": int(sum(c > 0 and d > 0 for c, d in zip(benefit_c, benefit_d, strict=True))),
            "bootstrap": {"benefit_C": deterministic_bootstrap(benefit_c, draws=bootstrap_draws), "benefit_D": deterministic_bootstrap(benefit_d, draws=bootstrap_draws)},
        },
        "four_cell_means": cell_means,
        "native_pair_difference": native_pair,
        "m_video_replacement_loss": replacement_loss,
        "n_video_scorer_effect": scorer_effect,
    }
