"""Deterministic source-group bootstrap used by both C routes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .common import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    TimingError,
    quantile,
)


def bootstrap_indices(
    groups: Sequence[str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[list[str], np.ndarray]:
    """Return sorted group labels and PCG64 group-resampling indices.

    ``groups`` may contain one or more rows per group.  Resampling is always at
    the group level, so repeated rows belonging to one source group remain
    together.  The returned index matrix can be shared by every C contrast,
    which makes the paired model difference use exactly the same random draws.
    """

    labels = sorted({str(group) for group in groups})
    if not labels:
        raise TimingError("bootstrap requires at least one source group")
    if isinstance(draws, bool) or int(draws) != draws or int(draws) <= 0:
        raise TimingError("bootstrap draws must be a positive integer")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    indices = rng.integers(0, len(labels), size=(int(draws), len(labels)), endpoint=False)
    return labels, np.asarray(indices, dtype=np.int64)


def _group_values(
    values: Mapping[str, float] | Sequence[float],
    groups: Sequence[str] | None,
) -> tuple[list[str], np.ndarray]:
    if isinstance(values, Mapping):
        grouped: dict[str, list[float]] = {}
        for key, value in values.items():
            grouped.setdefault(str(key), []).append(float(value))
        labels = sorted(grouped)
        means = np.asarray([np.mean(grouped[label], dtype=np.float64) for label in labels], dtype=np.float64)
    else:
        if groups is None or len(values) != len(groups):
            raise TimingError("bootstrap values/groups length mismatch")
        grouped = {}
        for value, group in zip(values, groups, strict=True):
            grouped.setdefault(str(group), []).append(float(value))
        labels = sorted(grouped)
        means = np.asarray([np.mean(grouped[label], dtype=np.float64) for label in labels], dtype=np.float64)
    if means.size == 0 or not np.isfinite(means).all():
        raise TimingError("bootstrap values must be finite and non-empty")
    return labels, means


def bootstrap_group_ci(
    values: Mapping[str, float] | Sequence[float],
    groups: Sequence[str] | None = None,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    indices: np.ndarray | None = None,
    level: float = 0.99,
) -> dict[str, Any]:
    """Summarize a group mean with fixed PCG64 and linear quantiles."""

    labels, means = _group_values(values, groups)
    if indices is None:
        index_labels, indices = bootstrap_indices(labels, draws=draws, seed=seed)
        if index_labels != labels:
            raise TimingError("bootstrap labels changed while constructing indices")
    else:
        indices = np.asarray(indices)
        if indices.ndim != 2 or indices.shape[1] != len(labels):
            raise TimingError("bootstrap index shape does not match source groups")
        if indices.size and (indices.min() < 0 or indices.max() >= len(labels)):
            raise TimingError("bootstrap indices are outside source-group support")
        draws = int(indices.shape[0])
    estimates = np.mean(means[indices.astype(np.int64, copy=False)], axis=1, dtype=np.float64)
    alpha = (1.0 - float(level)) / 2.0
    if not 0.0 < alpha < 0.5:
        raise TimingError("level must be between 0 and 1")
    result = {
        "mean": float(np.mean(means, dtype=np.float64)),
        "ci": [quantile(estimates, alpha), quantile(estimates, 1.0 - alpha)],
        "ci99": [quantile(estimates, 0.005), quantile(estimates, 0.995)] if abs(level - 0.99) < 1e-12 else None,
        "level": float(level),
        "group_labels": labels,
        "group_means": {label: float(value) for label, value in zip(labels, means, strict=True)},
        "group_positive_count": int(np.sum(means > 0.0)),
        "group_count": len(labels),
        "draws": int(draws),
        "seed": int(seed),
        "rng": "numpy.PCG64",
        "quantile_method": "linear",
    }
    return result


def bootstrap_group_means(
    values: Sequence[float],
    groups: Sequence[str],
    indices: np.ndarray,
    *,
    level: float = 0.99,
) -> dict[str, Any]:
    """Validator-friendly alias accepting pre-frozen indices."""

    return bootstrap_group_ci(values, groups, indices=indices, level=level)


def paired_model_difference(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    indices: np.ndarray | None = None,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    level: float = 0.99,
) -> dict[str, Any]:
    """Bootstrap ``left - right`` on the common source-group denominator."""

    left_labels = {str(key) for key in left}
    right_labels = {str(key) for key in right}
    if left_labels != right_labels:
        missing_left = sorted(right_labels - left_labels)
        missing_right = sorted(left_labels - right_labels)
        raise TimingError(f"paired models have different groups: left-missing={missing_left}, right-missing={missing_right}")
    labels = sorted(left_labels)
    values = {label: float(left[label]) - float(right[label]) for label in labels}
    return bootstrap_group_ci(values, indices=indices, draws=draws, seed=seed, level=level)


def ci_lower(summary: Mapping[str, Any], *, key: str = "ci99") -> float | None:
    value = summary.get(key)
    if not isinstance(value, Sequence) or len(value) != 2:
        return None
    try:
        number = float(value[0])
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None
