from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .protocol import DEFAULT_BOOTSTRAP_DRAWS, DEFAULT_SEED, NATURAL, ProtocolError, canonical_hash


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1:
        raise ProtocolError("statistical values must be one-dimensional")
    return array[np.isfinite(array)]


def group_means(rows: Iterable[Mapping[str, Any]], *, group_key: str = "source_group", value_key: str = "value") -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values[str(row[group_key])].append(numeric)
    return {group: float(np.mean(items, dtype=np.float64)) for group, items in sorted(values.items()) if items}


def cluster_bootstrap(
    values: Mapping[str, float] | Sequence[float],
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Equal-weight group bootstrap with a recorded PCG64 draw contract."""

    if draws < 1 or not 0.0 < confidence < 1.0:
        raise ValueError("draws and confidence are invalid")
    labels = list(values) if isinstance(values, Mapping) else [str(index) for index in range(len(values))]
    array = np.asarray([float(values[label]) if isinstance(values, Mapping) else float(value) for label, value in (zip(labels, values) if not isinstance(values, Mapping) else ((label, values[label]) for label in labels))], dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ProtocolError("bootstrap requires finite non-empty group values")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    indices = rng.integers(0, array.size, size=(int(draws), array.size))
    estimates = array[indices].mean(axis=1, dtype=np.float64)
    tail = (1.0 - confidence) / 2.0
    result = {
        "group_count": int(array.size),
        "mean": float(array.mean(dtype=np.float64)),
        "ci": [float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1.0 - tail))],
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "ci975": [float(np.quantile(estimates, 0.0125)), float(np.quantile(estimates, 0.9875))],
        "draws": int(draws),
        "seed": int(seed),
        "labels": [str(label) for label in labels],
        "indices_sha256": canonical_hash(indices.tolist()),
    }
    return result


def holm_adjust(pvalues: Mapping[str, float] | Sequence[float]) -> dict[str, float] | list[float]:
    """Holm step-down adjustment preserving the caller's labels/order."""

    is_mapping = isinstance(pvalues, Mapping)
    labels = list(pvalues) if is_mapping else list(range(len(pvalues)))
    raw = [float(pvalues[label]) if is_mapping else float(value) for label, value in (zip(labels, pvalues) if not is_mapping else ((label, pvalues[label]) for label in labels))]
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in raw):
        raise ValueError("p-values must be finite and in [0, 1]")
    order = sorted(range(len(raw)), key=lambda index: raw[index])
    adjusted = [0.0] * len(raw)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(raw) - rank) * raw[index]))
        adjusted[index] = running
    return {str(label): adjusted[index] for index, label in enumerate(labels)} if is_mapping else adjusted


def paired_sign_flip(values: Sequence[float], *, alternative: str = "two-sided") -> dict[str, Any]:
    """Exact paired sign-flip test for <=12 groups; deterministic otherwise."""

    array = _finite(values)
    if array.size != len(values) or array.size == 0:
        raise ProtocolError("sign-flip test requires finite non-empty values")
    n = int(array.size)
    if n > 20:
        raise ProtocolError("exact sign-flip is limited to 20 groups")
    observed = float(array.mean(dtype=np.float64))
    estimates = np.asarray(
        [float(np.mean(array * np.asarray(signs, dtype=np.float64), dtype=np.float64)) for signs in itertools.product((-1.0, 1.0), repeat=n)],
        dtype=np.float64,
    )
    if alternative == "two-sided":
        extreme = np.abs(estimates) >= abs(observed) - 1e-12
    elif alternative == "greater":
        extreme = estimates >= observed - 1e-12
    elif alternative == "less":
        extreme = estimates <= observed + 1e-12
    else:
        raise ValueError("alternative must be two-sided, greater, or less")
    return {"n": n, "estimate": observed, "p": float(np.mean(extreme)), "enumerated": int(len(estimates)), "alternative": alternative}


def pearson_spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, float | None]:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ProtocolError("correlation inputs have different shapes")
    if left.size < 3 or not np.isfinite(left).all() or not np.isfinite(right).all():
        return {"pearson": None, "spearman": None, "n": int(left.size)}
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        pearson = None
    else:
        pearson = float(np.corrcoef(left, right)[0, 1])
    left_rank = np.argsort(np.argsort(left)).astype(np.float64)
    right_rank = np.argsort(np.argsort(right)).astype(np.float64)
    if np.std(left_rank) <= 1e-12 or np.std(right_rank) <= 1e-12:
        spearman = None
    else:
        spearman = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return {"pearson": pearson, "spearman": spearman, "n": int(left.size)}


def _metric_by_group(rows: Iterable[Mapping[str, Any]], *, metric_key: str, group_key: str = "source_group") -> dict[str, float]:
    return group_means(({"source_group": row[group_key], "value": row[metric_key]} for row in rows if row.get(metric_key) is not None))


def _pair_deltas(rows: Iterable[Mapping[str, Any]], *, value_key: str, group_key: str = "source_group", natural_label: str = "natural") -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get(value_key) is None:
            continue
        values[(str(row[group_key]), str(row.get("tts_arm", row.get("condition"))))].append(float(row[value_key]))
    natural: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get(value_key) is not None and str(row.get("condition", row.get("arm"))) == natural_label:
            natural[str(row[group_key])].append(float(row[value_key]))
    natural_mean = {group: float(np.mean(items, dtype=np.float64)) for group, items in natural.items() if items}
    output: dict[tuple[str, str], float] = {}
    for (group, arm), items in values.items():
        if group in natural_mean and arm != natural_label:
            output[(group, arm)] = float(np.mean(items, dtype=np.float64) - natural_mean[group])
    return output


def summarize_evidence(
    official_rows: Sequence[Mapping[str, Any]],
    temporal_rows: Sequence[Mapping[str, Any]],
    ratings: Sequence[Mapping[str, Any]] | None = None,
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Summarize A without merging official, custom, visual, or human endpoints."""

    official_delta = _pair_deltas(official_rows, value_key="value")
    official_group = defaultdict(list)
    for (group, _arm), value in official_delta.items():
        official_group[group].append(value)
    official_group_values = {group: float(np.mean(items, dtype=np.float64)) for group, items in official_group.items()}
    temporal_group = defaultdict(list)
    for row in temporal_rows:
        if row.get("delta_r") is not None:
            temporal_group[str(row["source_group"])].append(float(row["delta_r"]))
    temporal_group_values = {group: float(np.mean(items, dtype=np.float64)) for group, items in temporal_group.items()}
    result: dict[str, Any] = {
        "schema_version": 1,
        "official_fulltrack": {
            "estimand": "group_mean(C_T-C_N)",
            "group_count": len(official_group_values),
            "bootstrap": cluster_bootstrap(official_group_values, draws=draws, seed=seed) if official_group_values else None,
            "rows": list(official_rows),
        },
        "temporal_rank": {
            "estimand": "group_mean(R_T-R_N)",
            "group_count": len(temporal_group_values),
            "bootstrap": cluster_bootstrap(temporal_group_values, draws=draws, seed=seed + 1) if temporal_group_values else None,
            "rows": list(temporal_rows),
        },
        "correlations": {},
        "human": {"status": "PENDING_HUMAN", "rows": [], "group_count": 0},
        "conclusion": "PENDING_HUMAN",
    }
    common = sorted(set(official_group_values) & set(temporal_group_values))
    result["correlations"]["official_deltaC_vs_deltaR"] = pearson_spearman(
        [official_group_values[group] for group in common],
        [temporal_group_values[group] for group in common],
    )
    if ratings is not None:
        human_groups: dict[str, list[float]] = defaultdict(list)
        human_arms: dict[str, set[str]] = defaultdict(set)
        for row in ratings:
            if row.get("status") not in {None, "MEASURED"}:
                continue
            if row.get("score") is None or row.get("source_group") is None:
                continue
            group = str(row["source_group"])
            human_groups[group].append(float(row["score"]))
            human_arms[group].add(str(row.get("tts_arm", "")))
        descriptive_values = {group: float(np.mean(items, dtype=np.float64) - 0.5) for group, items in human_groups.items() if items}
        complete_groups = sorted(group for group, arms in human_arms.items() if len({arm for arm in arms if arm != NATURAL}) >= 2)
        human_values = {group: descriptive_values[group] for group in complete_groups if group in descriptive_values}
        result["human"] = {
            "status": "MEASURED" if human_values else "PENDING_HUMAN",
            "group_count": len(human_values),
            "descriptive_group_count": len(descriptive_values),
            "complete_group_count": len(complete_groups),
            "inference_status": "ESTIMABLE" if len(complete_groups) >= 30 else "DESCRIPTIVE_ONLY",
            "estimand": "group_mean(H-0.5)",
            "bootstrap": cluster_bootstrap(human_values, draws=draws, seed=seed + 2) if human_values else None,
            "rows": list(ratings),
        }
        if human_values and len(complete_groups) >= 30:
            result["conclusion"] = "POSITIVE" if result["human"]["bootstrap"]["ci975"][0] > 0.0 else "INCONCLUSIVE"
        elif human_values:
            result["conclusion"] = "DESCRIPTIVE_ONLY"
    result["artifact_sha256"] = canonical_hash({key: value for key, value in result.items() if key != "artifact_sha256"})
    return result


def summarize_patch(
    rows: Sequence[Mapping[str, Any]],
    human_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    draws: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_SEED,
    quality_confounded_groups: Sequence[str] = (),
) -> dict[str, Any]:
    """Analyze the two pre-registered N←T comparisons at source-group level."""

    by_group_condition: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if str(row.get("direction")) != "natural_from_tts":
            continue
        if row.get("metric") is None:
            continue
        by_group_condition[(str(row["source_group"]), str(row["condition"]), str(row.get("metric_family", "primary")))].append(float(row["metric"]))
    metric_families = sorted({key[2] for key in by_group_condition})
    families: dict[str, Any] = {}
    for family in metric_families:
        base = {(group): float(np.mean(values, dtype=np.float64)) for (group, condition, current), values in by_group_condition.items() if current == family and condition == "BASE"}
        coherent = {(group): float(np.mean(values, dtype=np.float64)) for (group, condition, current), values in by_group_condition.items() if current == family and condition == "COHERENT"}
        scrambled = {(group): float(np.mean(values, dtype=np.float64)) for (group, condition, current), values in by_group_condition.items() if current == family and condition == "SCRAMBLED"}
        complete = sorted(set(base) & set(coherent) & set(scrambled))
        delta_base = {group: coherent[group] - base[group] for group in complete}
        delta_scrambled = {group: coherent[group] - scrambled[group] for group in complete}
        p_base = paired_sign_flip(list(delta_base.values())) if delta_base else None
        p_scrambled = paired_sign_flip(list(delta_scrambled.values())) if delta_scrambled else None
        raw_p = {name: result["p"] for name, result in (("coherent_minus_base", p_base), ("coherent_minus_scrambled", p_scrambled)) if result is not None}
        adjusted = holm_adjust(raw_p) if raw_p else {}
        families[family] = {
            "complete_groups": complete,
            "group_count": len(complete),
            "coherent_minus_base": {"groups": delta_base, "test": p_base, "bootstrap": cluster_bootstrap(delta_base, draws=draws, seed=seed) if delta_base else None, "holm_p": adjusted.get("coherent_minus_base")},
            "coherent_minus_scrambled": {"groups": delta_scrambled, "test": p_scrambled, "bootstrap": cluster_bootstrap(delta_scrambled, draws=draws, seed=seed + 1) if delta_scrambled else None, "holm_p": adjusted.get("coherent_minus_scrambled")},
        }
    quality = set(str(item) for item in quality_confounded_groups)
    human = {"status": "PENDING_HUMAN", "rows": [] if human_rows is None else list(human_rows), "quality_confounded_groups": sorted(quality)}
    if human_rows is not None:
        human["status"] = "MEASURED"
    return {
        "schema_version": 1,
        "families": families,
        "human": human,
        "scientific_interpretation": "ALGORITHM_EVIDENCE_ONLY" if human["status"] == "PENDING_HUMAN" else "HUMAN_EVIDENCE_AVAILABLE",
        "mediation_percentage": None,
        "artifact_sha256": canonical_hash({"families": families, "human": human, "scientific_interpretation": "ALGORITHM_EVIDENCE_ONLY" if human["status"] == "PENDING_HUMAN" else "HUMAN_EVIDENCE_AVAILABLE", "mediation_percentage": None}),
    }


def assert_c_decomposition(background: float, minimum: float, c_value: float, *, atol: float = 1e-8) -> None:
    if not math.isclose(float(background) - float(minimum), float(c_value), rel_tol=0.0, abs_tol=atol):
        raise ProtocolError("C=B-D decomposition failed for a single scorer/support")
