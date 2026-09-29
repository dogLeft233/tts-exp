from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from . import config
from .common import ProtocolError, assert_finite


def legacy_distance_matrix(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
    """Rebuild the historical [88,31] matrix, including its fixed zero padding."""
    visual_value = np.asarray(visual, dtype=np.float32)
    audio_value = np.asarray(audio, dtype=np.float32)
    if visual_value.shape != (config.EMBEDDING_ROWS, config.EMBEDDING_DIM) or audio_value.shape != visual_value.shape:
        raise ProtocolError(f"embedding shape mismatch: {visual_value.shape} vs {audio_value.shape}")
    padded = np.pad(audio_value, ((config.VSHIFT, config.VSHIFT), (0, 0)), mode="constant")
    result = np.empty((config.EMBEDDING_ROWS, config.MATRIX_COLUMNS), dtype=np.float32)
    epsilon = np.float32(1e-6)
    for row in range(config.EMBEDDING_ROWS):
        candidate = padded[row : row + config.MATRIX_COLUMNS]
        difference = visual_value[row : row + 1] - candidate
        squared = np.square(difference + epsilon, dtype=np.float32)
        result[row] = np.sqrt(np.sum(squared, axis=1, dtype=np.float32)).astype(np.float32)
    return result


def matched_distance_matrix(visual: np.ndarray, audio: np.ndarray, lag_start: int) -> np.ndarray:
    """Build a bounded, unpadded [28,31] matrix on the supplied physical lag domain."""
    visual_value = np.asarray(visual, dtype=np.float32)
    audio_value = np.asarray(audio, dtype=np.float32)
    if visual_value.shape != (config.EMBEDDING_ROWS, config.EMBEDDING_DIM) or audio_value.shape != visual_value.shape:
        raise ProtocolError(f"embedding shape mismatch: {visual_value.shape} vs {audio_value.shape}")
    lags = list(range(int(lag_start), int(lag_start) + config.MATRIX_COLUMNS))
    result = np.empty((len(config.U_ROWS), config.MATRIX_COLUMNS), dtype=np.float32)
    epsilon = np.float32(1e-6)
    for output_row, visual_row in enumerate(config.U_ROWS):
        indices = np.asarray([visual_row + lag for lag in lags], dtype=np.int64)
        if int(indices.min()) < 0 or int(indices.max()) >= audio_value.shape[0]:
            raise ProtocolError(f"matched lag domain lacks real support at row {visual_row}: {lags[0]}..{lags[-1]}")
        difference = visual_value[visual_row : visual_row + 1] - audio_value[indices]
        squared = np.square(difference + epsilon, dtype=np.float32)
        result[output_row] = np.sqrt(np.sum(squared, axis=1, dtype=np.float32)).astype(np.float32)
    return result


def curve(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != config.MATRIX_COLUMNS or not np.isfinite(value).all():
        raise ProtocolError(f"invalid distance matrix: {value.shape}")
    return np.mean(value, axis=0, dtype=np.float64)


def metrics_from_curve(values: np.ndarray, *, offset_base: int) -> dict[str, Any]:
    curve_value = np.asarray(values, dtype=np.float64)
    if curve_value.shape != (config.MATRIX_COLUMNS,) or not np.isfinite(curve_value).all():
        raise ProtocolError(f"invalid curve shape: {curve_value.shape}")
    minimum_index = int(np.argmin(curve_value))
    ordered = np.sort(curve_value)
    minimum = float(curve_value[minimum_index])
    return {
        "curve": [float(item) for item in curve_value],
        "min_index": minimum_index,
        "offset": int(offset_base - minimum_index),
        "D": minimum,
        "C": float(np.median(curve_value) - minimum),
        "minimum": minimum,
        "second_minimum": float(ordered[1]),
        "peak_gap": float(ordered[1] - ordered[0]),
    }


def metrics_from_matrix(matrix: np.ndarray, *, offset_base: int) -> dict[str, Any]:
    return metrics_from_curve(curve(matrix), offset_base=offset_base)


def bootstrap_indices() -> np.ndarray:
    return np.random.default_rng(config.BOOTSTRAP_SEED).integers(
        0,
        config.EXPECTED_GROUP_COUNT,
        size=(config.BOOTSTRAP_DRAWS, config.EXPECTED_GROUP_COUNT),
        endpoint=False,
    )


def bootstrap_group_means(values: Sequence[float], groups: Sequence[str], indices: np.ndarray) -> dict[str, Any]:
    if len(values) != len(groups):
        raise ProtocolError("bootstrap values/groups length mismatch")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        grouped[str(group)].append(float(value))
    labels = sorted(grouped)
    if len(labels) != config.EXPECTED_GROUP_COUNT or any(len(grouped[label]) != 2 for label in labels):
        raise ProtocolError("bootstrap requires eight source groups with two records each")
    means = np.asarray([np.mean(grouped[label], dtype=np.float64) for label in labels], dtype=np.float64)
    if indices.shape != (config.BOOTSTRAP_DRAWS, config.EXPECTED_GROUP_COUNT):
        raise ProtocolError(f"bootstrap index shape changed: {indices.shape}")
    estimates = np.mean(means[indices], axis=1, dtype=np.float64)
    return {
        "mean": float(np.mean(means, dtype=np.float64)),
        "ci95": [float(np.quantile(estimates, 0.025, method="linear")), float(np.quantile(estimates, 0.975, method="linear"))],
        "group_labels": labels,
        "group_means": {label: float(value) for label, value in zip(labels, means, strict=True)},
        "group_positive_count": int(np.sum(means > 0.0)),
        "draws": config.BOOTSTRAP_DRAWS,
        "seed": config.BOOTSTRAP_SEED,
        "rng": "numpy_default_rng_pcg64",
        "quantile_method": "linear",
    }


def decision_flags(
    *,
    anchor_pass: bool,
    matched_pass_count: int,
    old_failure_ids: set[str],
    expected_failure_ids: set[str],
    recovered_ids: set[str],
    record_count: int = config.EXPECTED_RECORD_COUNT,
) -> dict[str, Any]:
    corrected_control_pass = bool(anchor_pass and int(matched_pass_count) >= 14)
    boundary_explanation_complete = bool(
        old_failure_ids == expected_failure_ids
        and recovered_ids == expected_failure_ids
        and int(matched_pass_count) == int(record_count)
    )
    recovered = bool(corrected_control_pass and boundary_explanation_complete)
    return {
        "corrected_control_pass": corrected_control_pass,
        "boundary_explanation_complete": boundary_explanation_complete,
        "terminal_decision": "SEARCH_SUPPORT_RECOVERED" if recovered else "CONTROL_UNRESOLVED",
        "content_probe_revision_eligible": recovered,
    }


def analyze_record(
    *,
    sample_id: str,
    source_group: str,
    visual: np.ndarray,
    natural_audio: np.ndarray,
    delayed_audio: np.ndarray,
    cached_n_matrix: np.ndarray,
    cached_delay_matrix: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    rebuilt_n = legacy_distance_matrix(visual, natural_audio)
    rebuilt_delay = legacy_distance_matrix(visual, delayed_audio)
    old_n_curve = curve(rebuilt_n[np.asarray(config.U_ROWS)])
    old_delay_curve = curve(rebuilt_delay[np.asarray(config.U_ROWS)])
    old_n = metrics_from_curve(old_n_curve, offset_base=config.VSHIFT)
    old_delay = metrics_from_curve(old_delay_curve, offset_base=config.VSHIFT)
    natural_matched = matched_distance_matrix(visual, natural_audio, config.NATURAL_LAG_START)
    delay_matched = matched_distance_matrix(visual, delayed_audio, config.DELAY_LAG_START)
    natural_curve = curve(natural_matched)
    delay_curve = curve(delay_matched)
    matched_n = metrics_from_curve(natural_curve, offset_base=config.VSHIFT)
    matched_delay = metrics_from_curve(delay_curve, offset_base=config.VSHIFT - config.DELAY_LAG_START + config.NATURAL_LAG_START)
    # The formula above is 10-j for the delayed domain (lag_start=-10).
    matched_delay["offset"] = int(10 - matched_delay["min_index"])
    old_difference = int(old_delay["offset"] - old_n["offset"])
    matched_difference = int(matched_delay["offset"] - matched_n["offset"])
    expected_column = int(old_n["min_index"] + config.DELAY_FRAMES)
    q = np.arange(15, 73, dtype=np.int64)
    embedding_delta = np.asarray(delayed_audio[q + config.DELAY_FRAMES], dtype=np.float32) - np.asarray(natural_audio[q], dtype=np.float32)
    curve_delta = delay_curve - natural_curve
    result = {
        "sample_id": sample_id,
        "source_group": source_group,
        "legacy": {
            "natural": old_n,
            "delay": old_delay,
            "offset_difference": old_difference,
            "expected_column": expected_column,
            "expected_in_legacy_domain": bool(0 <= expected_column < config.MATRIX_COLUMNS),
            "old_failure": not (config.OFFSET_LOW <= old_difference <= config.OFFSET_HIGH),
            "natural_matrix_max_abs": float(np.max(np.abs(rebuilt_n.astype(np.float64) - np.asarray(cached_n_matrix, dtype=np.float64)))),
            "delay_matrix_max_abs": float(np.max(np.abs(rebuilt_delay.astype(np.float64) - np.asarray(cached_delay_matrix, dtype=np.float64)))),
        },
        "matched": {
            "natural_lag_start": config.NATURAL_LAG_START,
            "delay_lag_start": config.DELAY_LAG_START,
            "natural_lags": list(range(config.NATURAL_LAG_START, config.NATURAL_LAG_START + config.MATRIX_COLUMNS)),
            "delay_lags": list(range(config.DELAY_LAG_START, config.DELAY_LAG_START + config.MATRIX_COLUMNS)),
            "natural": matched_n,
            "delay": matched_delay,
            "offset_difference": matched_difference,
            "offset_pass": bool(config.OFFSET_LOW <= matched_difference <= config.OFFSET_HIGH),
            "same_array_index": bool(matched_n["min_index"] == matched_delay["min_index"]),
        },
        "embedding_shift": {
            "q_start": 15,
            "q_end": 72,
            "max_abs": float(np.max(np.abs(embedding_delta))),
            "rms": float(np.sqrt(np.mean(np.square(embedding_delta, dtype=np.float64)))),
        },
        "curve_shift": {
            "max_abs": float(np.max(np.abs(curve_delta))),
            "rms": float(np.sqrt(np.mean(np.square(curve_delta, dtype=np.float64)))),
        },
        "anchor_damage": float(old_delay_curve[old_n["min_index"]] - old_n_curve[old_n["min_index"]]),
    }
    assert_finite(result)
    return result, natural_matched, delay_matched
