"""Corrected signed-margin, fixed-support, ABX, and paired statistics."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


def normalize_vector(value: Sequence[float] | np.ndarray, *, name: str = "vector") -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} is empty or non-finite")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} has zero norm")
    return vector / norm


def signed_phone_scores(vector: Sequence[float] | np.ndarray, label: str, centroids: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Score the correct class against the strongest *incorrect* class."""

    if label not in centroids or len(centroids) < 2:
        return {"prediction": None, "correct_score": None, "strongest_wrong_score": None, "signed_margin": -2.0, "legacy_top1_gap": None, "valid": False, "reason": "LABEL_OR_REFERENCE_MISSING"}
    z = normalize_vector(vector)
    labels = sorted(str(key) for key in centroids)
    scores = {candidate: float(np.dot(z, normalize_vector(centroids[candidate], name=f"centroid/{candidate}"))) for candidate in labels}
    ordered = sorted(labels, key=lambda candidate: (-scores[candidate], candidate))
    prediction = ordered[0]
    wrong = [candidate for candidate in labels if candidate != label]
    strongest_wrong = max(scores[candidate] for candidate in wrong)
    legacy = scores[ordered[0]] - scores[ordered[1]] if len(ordered) > 1 else None
    return {
        "prediction": prediction,
        "correct_score": scores[label],
        "strongest_wrong_score": strongest_wrong,
        "signed_margin": scores[label] - strongest_wrong,
        "legacy_top1_gap": legacy,
        "valid": True,
        "correct": bool(prediction == label),
        "scores": scores,
    }


def fit_group_equal_centroids(rows: Iterable[Mapping[str, Any]], *, condition: str | None = None, split: str = "fit", min_tokens: int = 1, min_groups: int = 1) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if condition is not None and str(row.get("condition")) != condition:
            continue
        if split is not None and str(row.get("analysis_split")) != split:
            continue
        if not bool(row.get("valid", row.get("embedding") is not None)) or not bool(row.get("speech", True)):
            continue
        label = str(row.get("label", ""))
        group = str(row.get("source_group", ""))
        value = row.get("embedding")
        if not label or not group or value is None:
            continue
        grouped[label][group].append(normalize_vector(value))
    centroids: dict[str, list[float]] = {}
    support: dict[str, Any] = {}
    for label in sorted(grouped):
        if sum(len(items) for items in grouped[label].values()) < min_tokens or len(grouped[label]) < min_groups:
            support[label] = {"eligible": False, "tokens": sum(len(items) for items in grouped[label].values()), "groups": len(grouped[label])}
            continue
        group_centers = [normalize_vector(np.mean(np.stack(items), axis=0)) for _, items in sorted(grouped[label].items())]
        centroid = normalize_vector(np.mean(np.stack(group_centers), axis=0))
        centroids[label] = centroid.astype(np.float32).tolist()
        support[label] = {"eligible": True, "tokens": sum(len(items) for items in grouped[label].values()), "groups": len(grouped[label])}
    return {"labels": sorted(centroids), "centroids": centroids, "support": support, "fit_scope": "group_equal"}


def score_fixed_support(
    rows: Iterable[Mapping[str, Any]],
    centroids: Mapping[str, Sequence[float]],
    *,
    expected: Iterable[Mapping[str, Any]] | None = None,
    vectors_by_key: Mapping[str, Sequence[float]] | None = None,
    condition: str | None = None,
) -> dict[str, Any]:
    """Score a pre-registered support; missing candidates stay in the denominator."""

    row_list = list(rows)
    actual: dict[str, Mapping[str, Any]] = {}
    for row in row_list:
        if condition is not None and str(row.get("condition")) != condition:
            continue
        key = str(row.get("support_key", row.get("token_id", "")))
        if key:
            actual[key] = row
    expected_rows = list(expected) if expected is not None else row_list
    group_label_values: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    flat: list[dict[str, Any]] = []
    missing = 0
    for spec in expected_rows:
        key = str(spec.get("support_key", spec.get("token_id", "")))
        row = actual.get(key)
        label = str(spec.get("label", row.get("label", "") if row else ""))
        group = str(spec.get("source_group", row.get("source_group", "") if row else ""))
        vector = vectors_by_key.get(key) if vectors_by_key is not None else (row.get("embedding") if row is not None else None)
        if vector is None or not label or label not in centroids:
            missing += 1
            result = {"prediction": None, "signed_margin": -2.0, "legacy_top1_gap": None, "correct": False, "valid": False, "reason": "MISSING_CANDIDATE"}
        else:
            try:
                result = signed_phone_scores(vector, label, centroids)
            except ValueError as exc:
                result = {"prediction": None, "signed_margin": -2.0, "legacy_top1_gap": None, "correct": False, "valid": False, "reason": f"INVALID_VECTOR:{type(exc).__name__}"}
        row_result = {"support_key": key, "label": label, "source_group": group, **result}
        flat.append(row_result)
        group_label_values[group][label].append(row_result)
    group_metrics: list[dict[str, Any]] = []
    for group in sorted(group_label_values):
        label_metrics = []
        for label in sorted(group_label_values[group]):
            values = group_label_values[group][label]
            label_metrics.append({
                "label": label,
                "accuracy": float(np.mean([float(value.get("correct", False)) for value in values])),
                "margin": float(np.mean([float(value.get("signed_margin", -2.0)) for value in values])),
                "n_tokens": len(values),
            })
        group_metrics.append({
            "source_group": group,
            "accuracy": float(np.mean([value["accuracy"] for value in label_metrics])) if label_metrics else 0.0,
            "margin": float(np.mean([value["margin"] for value in label_metrics])) if label_metrics else -2.0,
            "label_metrics": label_metrics,
            "n_tokens": sum(value["n_tokens"] for value in label_metrics),
        })
    return {
        "accuracy": float(np.mean([row["accuracy"] for row in group_metrics])) if group_metrics else None,
        "margin": float(np.mean([row["margin"] for row in group_metrics])) if group_metrics else None,
        "legacy_top1_gap": float(np.mean([float(row["legacy_top1_gap"]) for row in flat if row.get("legacy_top1_gap") is not None])) if any(row.get("legacy_top1_gap") is not None for row in flat) else None,
        "group_metrics": group_metrics,
        "eligible_groups": [row["source_group"] for row in group_metrics],
        "support_count": len(expected_rows),
        "valid_count": len(expected_rows) - missing,
        "missing_count": missing,
        "coverage": (len(expected_rows) - missing) / max(len(expected_rows), 1),
        "predictions": flat,
        "labels": sorted(str(label) for label in centroids),
        "support_ok": bool(group_metrics),
        "reason": None if group_metrics else "NO_SUPPORT",
    }


def paired_group_bootstrap(group_effects: Mapping[str, float] | Sequence[float], *, draws: int = 10000, seed: int = 20260921) -> dict[str, Any]:
    values = np.asarray(list(group_effects.values()) if isinstance(group_effects, Mapping) else list(group_effects), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"estimate": None, "ci_low": None, "ci_high": None, "n_groups": 0, "draws": 0, "seed": int(seed)}
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    sample_index = rng.integers(0, values.size, size=(int(draws), values.size))
    boot = values[sample_index].mean(axis=1)
    return {"estimate": float(values.mean()), "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975)), "n_groups": int(values.size), "draws": int(draws), "seed": int(seed)}


def sign_flip_p(group_effects: Mapping[str, float] | Sequence[float], *, seed: int = 20260921, permutations: int = 10000) -> float | None:
    values = np.asarray(list(group_effects.values()) if isinstance(group_effects, Mapping) else list(group_effects), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    observed = abs(float(values.mean()))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(int(permutations), values.size))
    simulated = np.abs((signs * values).mean(axis=1))
    return float((1.0 + np.count_nonzero(simulated >= observed)) / (simulated.size + 1.0))


def benjamini_hochberg(p_values: Mapping[str, float | None]) -> dict[str, float | None]:
    valid = sorted((float(value), key) for key, value in p_values.items() if value is not None and math.isfinite(float(value)))
    result = {key: None for key in p_values}
    running = 1.0
    for index in range(len(valid) - 1, -1, -1):
        value, key = valid[index]
        rank = index + 1
        running = min(running, value * len(valid) / rank)
        result[key] = float(min(1.0, running))
    return result


def build_abx_triplets(rows: Iterable[Mapping[str, Any]], *, triplets_per_group: int = 20, seed: int = 20260921) -> list[dict[str, Any]]:
    by_group_label: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not bool(row.get("valid", row.get("embedding") is not None)) or not bool(row.get("speech", True)):
            continue
        group = str(row.get("source_group", ""))
        label = str(row.get("label", ""))
        occurrence = str(row.get("occurrence_key", row.get("token_id", "")))
        if group and label and occurrence:
            by_group_label[group][label].append(row)
    result: list[dict[str, Any]] = []
    for group in sorted(by_group_label):
        labels = sorted(by_group_label[group])
        local: list[dict[str, Any]] = []
        for label in labels:
            same = sorted(by_group_label[group][label], key=lambda row: str(row.get("occurrence_key", row.get("token_id", ""))))
            if len(same) < 2:
                continue
            for other_label in labels:
                if other_label == label or not by_group_label[group][other_label]:
                    continue
                other = sorted(by_group_label[group][other_label], key=lambda row: str(row.get("occurrence_key", row.get("token_id", ""))))
                for index, x in enumerate(same):
                    a = same[(index + 1) % len(same)]
                    if str(a.get("occurrence_key", a.get("token_id"))) == str(x.get("occurrence_key", x.get("token_id"))):
                        continue
                    b = other[index % len(other)]
                    local.append({"group": group, "a": str(a.get("occurrence_key", a.get("token_id"))), "b": str(b.get("occurrence_key", b.get("token_id"))), "x": str(x.get("occurrence_key", x.get("token_id"))), "label_a": label, "label_b": other_label, "label_x": label})
        local.sort(key=lambda row: hashlib.sha256(f"{seed}|{row['group']}|{row['a']}|{row['b']}|{row['x']}".encode()).hexdigest())
        result.extend(local[: int(triplets_per_group)])
    return result


def score_abx_triplets(triplets: Sequence[Mapping[str, Any]], vectors: Mapping[str, Sequence[float]], *, distance: str = "cosine") -> dict[str, Any]:
    errors: list[float] = []
    kept: list[dict[str, Any]] = []
    for triplet in triplets:
        a_key, b_key, x_key = str(triplet["a"]), str(triplet["b"]), str(triplet["x"])
        if a_key == x_key or any(key not in vectors for key in (a_key, b_key, x_key)):
            continue
        xa, xb, xx = (normalize_vector(vectors[key]) for key in (a_key, b_key, x_key))
        da, db = 1.0 - float(np.dot(xa, xx)), 1.0 - float(np.dot(xb, xx))
        error = 1.0 if db < da else (0.0 if da < db else 0.5)
        errors.append(error)
        kept.append({**dict(triplet), "error": error})
    return {"error": float(np.mean(errors)) if errors else None, "n_triplets": len(errors), "triplets": kept, "distance": distance, "reason": None if errors else "NO_VALID_TRIPLETS"}


def decide_gain(natural: Mapping[str, Any], candidate: Mapping[str, Any], *, min_accuracy_gain: float = 0.010, min_groups: int = 30) -> dict[str, Any]:
    groups_n = {str(row["source_group"]): float(row["accuracy"]) for row in natural.get("group_metrics", [])}
    groups_c = {str(row["source_group"]): float(row["accuracy"]) for row in candidate.get("group_metrics", [])}
    common = sorted(set(groups_n) & set(groups_c))
    deltas = {group: groups_c[group] - groups_n[group] for group in common}
    stats = paired_group_bootstrap(deltas)
    return {"estimate": stats["estimate"], "ci_low": stats["ci_low"], "ci_high": stats["ci_high"], "n_groups": len(common), "coverage": len(common) / max(len(groups_n), 1), "passes_exploratory": bool(len(common) >= min_groups and stats["estimate"] is not None and stats["ci_low"] >= min_accuracy_gain)}


__all__ = [
    "benjamini_hochberg",
    "build_abx_triplets",
    "decide_gain",
    "fit_group_equal_centroids",
    "normalize_vector",
    "paired_group_bootstrap",
    "score_abx_triplets",
    "score_fixed_support",
    "sign_flip_p",
    "signed_phone_scores",
]
