"""Pure, leakage-aware metrics for the phone separability protocol."""

from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def normalize_phone(label: object) -> str:
    return unicodedata.normalize("NFC", str(label)).strip()


def _vector(row: Mapping[str, Any]) -> np.ndarray | None:
    value = row.get("embedding")
    if value is None or not bool(row.get("valid", True)):
        return None
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else None


def _unit(value: np.ndarray, name: str = "vector") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} is empty or non-finite")
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise ValueError(f"{name} has zero norm")
    return array / norm


def _quantile(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="linear"))
    except TypeError:  # pragma: no cover - old NumPy compatibility
        return float(np.quantile(values, q, interpolation="linear"))


def paired_group_bootstrap(
    effects: Mapping[str, float] | Sequence[Mapping[str, Any]],
    *,
    seed: int = 20260921,
    draws: int = 10_000,
) -> dict[str, Any]:
    """Bootstrap group means; token rows never become independent samples."""

    by_group: dict[str, float] = {}
    if isinstance(effects, Mapping):
        by_group = {str(key): float(value) for key, value in effects.items()}
    else:
        for row in effects:
            group = str(row["source_group"])
            if group in by_group:
                raise ValueError(f"duplicate source group: {group}")
            by_group[group] = float(row["effect"])
    groups = sorted(by_group)
    values = np.asarray([by_group[group] for group in groups], dtype=np.float64)
    if not groups or not np.all(np.isfinite(values)):
        return {"estimate": None, "ci_low": None, "ci_high": None, "n_groups": 0, "groups": groups, "seed": int(seed), "draws": int(draws), "reason": "EMPTY_OR_NONFINITE"}
    if draws <= 0:
        raise ValueError("draws must be positive")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    samples = rng.integers(0, len(values), size=(int(draws), len(values)))
    boot = values[samples].mean(axis=1)
    return {"estimate": float(values.mean()), "ci_low": _quantile(boot, 0.025), "ci_high": _quantile(boot, 0.975), "n_groups": len(groups), "groups": groups, "seed": int(seed), "draws": int(draws), "method": "source_group_paired_percentile_bootstrap"}


def sign_flip_p(effects: Mapping[str, float] | Sequence[float], *, seed: int = 20260921, draws: int = 10_000) -> float | None:
    values = np.asarray(list(effects.values()) if isinstance(effects, Mapping) else list(effects), dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    observed = abs(float(values.mean()))
    if values.size <= 16:
        masks = np.arange(1 << values.size, dtype=np.uint64)[:, None]
        bits = ((masks >> np.arange(values.size, dtype=np.uint64)) & 1).astype(np.float64)
        signs = bits * 2.0 - 1.0
        null = np.abs((signs * values[None, :]).mean(axis=1))
    else:
        rng = np.random.Generator(np.random.PCG64(int(seed)))
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(int(draws), values.size))
        null = np.abs((signs * values[None, :]).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (null.size + 1))


def benjamini_hochberg(p_values: Sequence[float | None]) -> list[float | None]:
    result: list[float | None] = [None] * len(p_values)
    finite = [(index, float(value)) for index, value in enumerate(p_values) if value is not None and math.isfinite(float(value))]
    finite.sort(key=lambda item: item[1])
    running = 1.0
    for rank in range(len(finite), 0, -1):
        index, value = finite[rank - 1]
        running = min(running, value * len(finite) / rank)
        result[index] = min(1.0, running)
    return result


def _record_view(row: Mapping[str, Any], view: str) -> Mapping[str, Any] | None:
    views = row.get("views")
    if isinstance(views, Mapping):
        candidate = views.get(view)
        return candidate if isinstance(candidate, Mapping) else None
    return row if str(row.get("view", "full")) == view else None


def flatten_view_records(rows: Sequence[Mapping[str, Any]], view: str = "full") -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        selected = _record_view(row, view)
        if selected is None:
            continue
        merged = dict(row)
        merged.update(dict(selected))
        merged["view"] = view
        result.append(merged)
    return result


def fit_probe_bundle(
    records: Sequence[Mapping[str, Any]],
    *,
    view: str = "full",
    reference: str = "mixed",
    fit_splits: Sequence[str] = ("fit",),
    min_tokens: int = 20,
    min_groups: int = 3,
) -> dict[str, Any]:
    """Fit group-equal phone centroids without inspecting DEV/E_SEEN scores."""

    if reference not in {"mixed", "natural", "tts"}:
        raise ValueError("reference must be mixed, natural, or tts")
    rows = flatten_view_records(records, view)
    allowed = set(str(value) for value in fit_splits)
    buckets: dict[str, dict[str, dict[str, list[np.ndarray]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        if str(row.get("analysis_split")) not in allowed:
            continue
        condition = str(row.get("condition", ""))
        if condition not in {"natural", "tts"}:
            continue
        label = normalize_phone(row.get("label", ""))
        group = str(row.get("source_group", ""))
        vector = _vector(row)
        if not label or not group or vector is None:
            continue
        buckets[label][condition][group].append(vector)
    labels: list[str] = []
    centroids: dict[str, list[float]] = {}
    support: dict[str, Any] = {}
    condition_centroids: dict[str, dict[str, list[float]]] = {}
    for label in sorted(buckets):
        info: dict[str, Any] = {}
        means: dict[str, np.ndarray] = {}
        eligible_conditions = [reference] if reference in {"natural", "tts"} else ["natural", "tts"]
        label_ok = True
        for condition in ("natural", "tts"):
            groups = buckets[label].get(condition, {})
            token_count = sum(len(values) for values in groups.values())
            group_names = sorted(groups)
            info[condition] = {"token_count": token_count, "group_count": len(group_names), "groups": group_names}
            if condition not in eligible_conditions:
                continue
            if token_count < min_tokens or len(group_names) < min_groups:
                label_ok = False
                continue
            group_means = [_unit(np.mean(np.stack(groups[group]), axis=0), f"{label}/{condition}/{group}") for group in group_names]
            means[condition] = _unit(np.mean(np.stack(group_means), axis=0), f"{label}/{condition}")
            info[condition]["centroid"] = means[condition].tolist()
        support[label] = info
        if not label_ok or not means:
            continue
        if reference == "mixed":
            if set(means) != {"natural", "tts"}:
                continue
            center = _unit((means["natural"] + means["tts"]) / 2.0, f"{label}/mixed")
        else:
            center = means[reference]
        labels.append(label)
        centroids[label] = center.tolist()
        condition_centroids[label] = {key: value.tolist() for key, value in means.items()}
    return {
        "schema_version": 1,
        "view": view,
        "reference": reference,
        "labels": labels,
        "centroids": centroids,
        "condition_centroids": condition_centroids,
        "support": support,
        "fit_splits": list(fit_splits),
        "min_tokens": int(min_tokens),
        "min_groups": int(min_groups),
        "fit_scope": "group_equal_then_condition_equal",
    }


def _predict(vector: np.ndarray, centroids: Mapping[str, Sequence[float]]) -> tuple[str, float, float, float]:
    unit = _unit(vector, "prediction")
    labels = sorted(str(label) for label in centroids)
    if not labels:
        raise ValueError("empty reference")
    matrix = np.stack([_unit(np.asarray(centroids[label]), f"centroid/{label}") for label in labels])
    scores = matrix @ unit
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    second = int(order[1]) if len(order) > 1 else best
    return labels[best], float(scores[best]), float(scores[best] - scores[second]), float(scores[second])


def _prepare_centroids(centroids: Mapping[str, Sequence[float]]) -> tuple[list[str], np.ndarray]:
    labels = sorted(str(label) for label in centroids)
    if not labels:
        raise ValueError("empty reference")
    matrix = np.stack([_unit(np.asarray(centroids[label]), f"centroid/{label}") for label in labels])
    return labels, matrix


def _predict_prepared(vector: np.ndarray, labels: Sequence[str], matrix: np.ndarray) -> tuple[str, float, float, float]:
    unit = _unit(vector, "prediction")
    scores = matrix @ unit
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    second = int(order[1]) if len(order) > 1 else best
    return str(labels[best]), float(scores[best]), float(scores[best] - scores[second]), float(scores[second])


def score_frozen_support(
    records: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
    *,
    view: str = "full",
    condition: str | None = None,
    split: str | None = None,
    min_labels: int = 5,
    min_tokens: int = 10,
    min_coverage: float = 0.70,
) -> dict[str, Any]:
    """Score fixed occurrences with macro-phone accuracy and cosine margin."""

    labels = {str(label) for label in reference.get("labels", [])}
    if not labels or not reference.get("centroids"):
        return {
            "view": view,
            "condition": condition,
            "split": split,
            "group_metrics": [],
            "eligible_groups": [],
            "n_groups": 0,
            "accuracy": None,
            "margin": None,
            "coverage": None,
            "predictions": [],
            "support_ok": False,
            "reason": "EMPTY_REFERENCE",
        }
    prediction_labels, prediction_matrix = _prepare_centroids(reference["centroids"])
    rows = flatten_view_records(records, view)
    if condition is not None:
        rows = [row for row in rows if str(row.get("condition")) == condition]
    if split is not None:
        rows = [row for row in rows if str(row.get("analysis_split")) == split]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if bool(row.get("speech", True)) and str(row.get("label", "")) in labels:
            grouped[str(row.get("source_group", row.get("pair_id", "")))].append(row)
    group_metrics: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for group in sorted(grouped):
        valid = [row for row in grouped[group] if _vector(row) is not None]
        by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in valid:
            by_label[str(row["label"])].append(row)
        accuracies: list[float] = []
        margins: list[float] = []
        within_scores: list[float] = []
        for label in sorted(by_label):
            label_correct = 0
            for row in by_label[label]:
                predicted, correct_score, margin, wrong_score = _predict_prepared(_vector(row), prediction_labels, prediction_matrix)
                label_correct += int(predicted == label)
                margins.append(margin)
                within_scores.append(correct_score if predicted == label else wrong_score)
                predictions.append({"token_id": row.get("token_id"), "source_group": group, "label": label, "prediction": predicted, "correct": predicted == label, "margin": margin})
            if by_label[label]:
                accuracies.append(label_correct / len(by_label[label]))
        total_speech = sum(1 for row in rows if str(row.get("source_group", row.get("pair_id", ""))) == group and bool(row.get("speech", True)))
        coverage = len(valid) / total_speech if total_speech else 0.0
        reasons: list[str] = []
        if len(by_label) < min_labels:
            reasons.append("COMMON_LABELS_BELOW_MIN")
        if len(valid) < min_tokens:
            reasons.append("VALID_TOKENS_BELOW_MIN")
        if coverage < min_coverage:
            reasons.append("SPEECH_TOKEN_COVERAGE_BELOW_MIN")
        group_metrics.append({"source_group": group, "accuracy": float(np.mean(accuracies)) if accuracies else None, "margin": float(np.mean(margins)) if margins else None, "within_similarity": float(np.mean(within_scores)) if within_scores else None, "valid_tokens": len(valid), "speech_tokens": total_speech, "coverage": coverage, "label_count": len(by_label), "eligible": not reasons, "reason_codes": reasons})
    eligible = [row for row in group_metrics if row["eligible"] and row["accuracy"] is not None]
    return {
        "view": view,
        "condition": condition,
        "split": split,
        "group_metrics": group_metrics,
        "eligible_groups": [row["source_group"] for row in eligible],
        "n_groups": len(eligible),
        "accuracy": float(np.mean([row["accuracy"] for row in eligible])) if eligible else None,
        "margin": float(np.mean([row["margin"] for row in eligible])) if eligible else None,
        "coverage": float(np.mean([row["coverage"] for row in eligible])) if eligible else None,
        "predictions": predictions,
        "support_ok": bool(eligible),
    }


def contrast_scores(
    score_rows: Mapping[str, Mapping[str, Any]],
    left: str,
    right: str,
    *,
    seed: int = 20260921,
    draws: int = 10_000,
) -> dict[str, Any]:
    groups = sorted(set(score_rows.get(left, {}).get("eligible_groups", [])) & set(score_rows.get(right, {}).get("eligible_groups", [])))
    left_by = {str(row["source_group"]): row for row in score_rows.get(left, {}).get("group_metrics", [])}
    right_by = {str(row["source_group"]): row for row in score_rows.get(right, {}).get("group_metrics", [])}
    effects = {group: float(left_by[group]["accuracy"] - right_by[group]["accuracy"]) for group in groups if left_by[group].get("accuracy") is not None and right_by[group].get("accuracy") is not None}
    margins = {group: float(left_by[group]["margin"] - right_by[group]["margin"]) for group in groups if left_by[group].get("margin") is not None and right_by[group].get("margin") is not None}
    return {"left": left, "right": right, "accuracy": paired_group_bootstrap(effects, seed=seed, draws=draws), "margin": paired_group_bootstrap(margins, seed=seed, draws=draws), "sign_flip_p": sign_flip_p(effects, seed=seed, draws=draws), "group_effects": effects}


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def domain_auc(records: Sequence[Mapping[str, Any]], *, view: str = "full", split: str | None = "dev") -> dict[str, Any]:
    """Report a source-domain AUC separately from phone identity metrics."""

    rows = [row for row in flatten_view_records(records, view) if (split is None or str(row.get("analysis_split")) == split) and str(row.get("condition")) in {"natural", "tts"} and _vector(row) is not None]
    if len({str(row.get("condition")) for row in rows}) < 2:
        return {"auc": None, "n": len(rows), "reason": "ONE_DOMAIN_ONLY"}
    # A nearest-centroid domain diagnostic is deterministic, dependency-light,
    # and avoids fitting a high-dimensional classifier on hundreds of
    # thousands of correlated token rows.  It is deliberately not used as a
    # phone identity score.
    rows = sorted(rows, key=lambda row: (str(row.get("source_group", "")), str(row.get("token_id", "")), str(row.get("condition", ""))))
    max_rows = 24_000
    if len(rows) > max_rows:
        indices = np.linspace(0, len(rows) - 1, max_rows, dtype=np.int64)
        rows = [rows[int(index)] for index in indices]
    natural = np.stack([_vector(row) for row in rows if row.get("condition") == "natural"])
    tts = np.stack([_vector(row) for row in rows if row.get("condition") == "tts"])
    center_n, center_t = natural.mean(axis=0), tts.mean(axis=0)
    scores = np.asarray([float(np.dot(_vector(row), center_t - center_n)) for row in rows])
    labels = np.asarray([int(row.get("condition") == "tts") for row in rows])
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    auc = float(np.mean(positives[:, None] > negatives[None, :]) + 0.5 * np.mean(positives[:, None] == negatives[None, :])) if positives.size and negatives.size else None
    return {"auc": auc, "n": len(rows), "classifier": "nearest_domain_centroid", "scope": "within_requested_split", "metric": "domain_auc_N_vs_T"}


def abx_score(
    records: Sequence[Mapping[str, Any]],
    *,
    view: str = "full",
    split: str | None = None,
    seed: int = 20260921,
    triplets_per_group: int = 20,
) -> dict[str, Any]:
    """Fixed deterministic ABX error; lower is better."""

    rows = [row for row in flatten_view_records(records, view) if (split is None or str(row.get("analysis_split")) == split) and bool(row.get("speech", True)) and _vector(row) is not None]
    by_group_label: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_group_label[str(row.get("source_group", ""))][normalize_phone(row.get("label", ""))].append(row)
    triplets: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for group in sorted(by_group_label):
        labels = sorted(label for label in by_group_label[group] if label)
        if len(labels) < 2:
            continue
        local: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
        for label in labels:
            same = sorted(by_group_label[group][label], key=lambda row: str(row.get("token_id", "")))
            other_labels = [value for value in labels if value != label]
            for index, x in enumerate(same):
                for other in other_labels[: max(1, min(2, len(other_labels)) )]:
                    opposite = sorted(by_group_label[group][other], key=lambda row: str(row.get("token_id", "")))
                    if not opposite:
                        continue
                    a = same[(index + 1) % len(same)]
                    b = opposite[index % len(opposite)]
                    local.append((a, b, x))
        triplets.extend(local[:triplets_per_group])
    if not triplets:
        return {"error": None, "n_triplets": 0, "triplets": [], "reason": "NO_TRIPLETS"}
    errors = []
    serial = []
    for a, b, x in triplets:
        xa, xb, xx = _vector(a), _vector(b), _vector(x)
        dist_a = 1.0 - float(np.dot(xa, xx))
        dist_b = 1.0 - float(np.dot(xb, xx))
        error = dist_b <= dist_a
        errors.append(float(error))
        serial.append({"a": a.get("token_id"), "b": b.get("token_id"), "x": x.get("token_id"), "error": bool(error)})
    return {"error": float(np.mean(errors)), "n_triplets": len(errors), "triplets": serial, "seed": int(seed), "view": view, "split": split}


def distance_diagnostics(records: Sequence[Mapping[str, Any]], *, view: str = "full", split: str | None = None) -> dict[str, Any]:
    rows = [row for row in flatten_view_records(records, view) if (split is None or str(row.get("analysis_split")) == split) and _vector(row) is not None]
    by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        by_label[normalize_phone(row.get("label", ""))].append(_vector(row))
    within: list[float] = []
    between: list[float] = []
    labels = sorted(label for label in by_label if label)
    for label in labels:
        values = by_label[label]
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                within.append(1.0 - float(np.dot(values[i], values[j])))
    for i, first in enumerate(labels):
        for second in labels[i + 1 :]:
            if by_label[first] and by_label[second]:
                between.append(1.0 - float(np.dot(np.mean(by_label[first], axis=0), np.mean(by_label[second], axis=0))))
    return {"within_distance": float(np.mean(within)) if within else None, "between_distance": float(np.mean(between)) if between else None, "n_within": len(within), "n_between": len(between), "fisher_like": (float(np.mean(between)) / max(float(np.mean(within)), 1e-12)) if within and between else None}


__all__ = [
    "abx_score",
    "benjamini_hochberg",
    "contrast_scores",
    "distance_diagnostics",
    "domain_auc",
    "fit_probe_bundle",
    "flatten_view_records",
    "paired_group_bootstrap",
    "score_frozen_support",
    "sign_flip_p",
]
