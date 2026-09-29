from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import config
from .common import ExperimentError


def endpoint(matrix: np.ndarray, rows: Sequence[int], anchor_offset: int | None = None) -> dict[str, Any]:
    value = np.asarray(matrix)
    selected = np.asarray([int(row) for row in rows], dtype=np.int64)
    if value.ndim != 2 or value.shape[1] != config.MATRIX_COLUMNS or selected.size == 0:
        raise ExperimentError("matrix or endpoint rows are malformed")
    if int(selected.min()) < 0 or int(selected.max()) >= value.shape[0]:
        raise ExperimentError("endpoint rows are outside the matrix")
    if not np.isfinite(value).all():
        raise ExperimentError("matrix contains non-finite values")
    z = np.mean(value[selected, :], axis=0, dtype=np.float64)
    minimum_index = int(np.argmin(z))
    minimum = float(z[minimum_index])
    median = float(np.median(z))
    result: dict[str, Any] = {
        "rows": [int(item) for item in selected],
        "curve": [float(item) for item in z],
        "D": minimum,
        "M": median,
        "C": median - minimum,
        "offset": int(config.VSHIFT - minimum_index),
        "min_index": minimum_index,
        "second_min": float(np.partition(z, 1)[1]),
        "clear": bool(float(np.partition(z, 1)[1] - minimum) > config.PEAK_GAP_THRESHOLD and abs(config.VSHIFT - minimum_index) < config.VSHIFT),
    }
    if anchor_offset is not None:
        column = config.VSHIFT - int(anchor_offset)
        if not 0 <= column < config.MATRIX_COLUMNS:
            raise ExperimentError(f"anchor offset is outside SyncNet columns: {anchor_offset}")
        result["anchor_offset"] = int(anchor_offset)
        result["D_anchor"] = float(z[column])
        result["C_anchor"] = median - float(z[column])
        result["anchor_column"] = int(column)
    return result


def compare(candidate: Mapping[str, Any], baseline: Mapping[str, Any], name: str, expected_shift: int | None = None) -> dict[str, Any]:
    delta_m = float(candidate["M"] - baseline["M"])
    value: dict[str, Any] = {
        "name": name,
        "C": float(candidate["C"] - baseline["C"]),
        "D": float(baseline["D"] - candidate["D"]),
        "anchor": float(baseline.get("D_anchor", 0.0) - candidate.get("D_anchor", 0.0)) if "D_anchor" in candidate and "D_anchor" in baseline else None,
        "delta_M": delta_m,
        "offset_delta": int(candidate["offset"] - baseline["offset"]),
        "clear": bool(candidate.get("clear", False) and baseline.get("clear", False)),
    }
    value["identity_check"] = bool(abs(value["C"] - (value["delta_M"] + value["D"])) <= 1e-10)
    if expected_shift is not None:
        value["expected_offset_delta"] = int(expected_shift)
        value["sign_error"] = int(value["offset_delta"] - expected_shift)
        value["sign_pass"] = bool(value["clear"] and abs(value["sign_error"]) <= config.OFFSET_TOLERANCE_FRAMES)
    return value


def make_bootstrap(groups: Sequence[str], *, seed: int = config.BOOTSTRAP_SEED, draws: int = config.BOOTSTRAP_DRAWS) -> tuple[list[str], np.ndarray]:
    labels = sorted({str(group) for group in groups})
    if len(labels) != len(groups) or len(set(groups)) != len(groups):
        raise ExperimentError("new bootstrap requires one record per source group")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(labels), size=(draws, len(labels)), endpoint=False, dtype=np.int64)
    return labels, indices


def bootstrap(values: Sequence[float], groups: Sequence[str], labels: Sequence[str], indices: np.ndarray) -> dict[str, Any]:
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.ndim != 1 or values_array.size != len(groups) or not np.isfinite(values_array).all():
        raise ExperimentError("bootstrap values are malformed")
    if list(map(str, groups)) != list(map(str, labels)):
        # Records are deliberately sorted by source_group before this function.
        raise ExperimentError("bootstrap group order is not the frozen sorted order")
    if indices.ndim != 2 or indices.shape[1] != len(labels):
        raise ExperimentError("bootstrap index matrix has the wrong shape")
    estimates = values_array[indices].mean(axis=1)
    return {
        "mean": float(np.mean(values_array)),
        "ci95": [float(np.quantile(estimates, 0.025, method="linear")), float(np.quantile(estimates, 0.975, method="linear"))],
        "draws": int(indices.shape[0]),
        "seed": config.BOOTSTRAP_SEED,
        "rng": "numpy_default_rng_pcg64",
        "quantile_method": "linear",
        "source_group_count": len(labels),
        "draw_indices_sha256": hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest(),
    }
