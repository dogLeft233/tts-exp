"""Fixed-support SyncNet arithmetic for the fresh-source replacement probe.

Only the first 140 video frames are part of this endpoint.  The functions in
this file intentionally operate on already extracted embeddings, which makes
the numerical contract easy to exercise with small synthetic arrays and lets
the validator recompute it without invoking a model.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .common import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    EPSILON,
    LAG_MAX,
    LAG_MIN,
    SEEDS,
    SHIFT_FRAMES,
    U_START,
    U_STOP,
    cell_key,
    sha256_bytes,
)


class ScoringError(ValueError):
    """Invalid embeddings or an incomplete fixed-support cell."""


def _embedding(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ScoringError(f"{name} must be a non-empty 2-D embedding array")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ScoringError(f"{name} contains non-finite/non-numeric values")
    return np.asarray(array, dtype=np.float32)


def _support(visual: np.ndarray, audio: np.ndarray, *, lag_min: int, lag_max: int) -> None:
    if visual.shape[1] != audio.shape[1]:
        raise ScoringError(f"embedding widths differ: visual={visual.shape}, audio={audio.shape}")
    min_q = U_START + lag_min
    max_q = (U_STOP - 1) + lag_max
    if visual.shape[0] < U_STOP:
        raise ScoringError(f"visual embedding support ends at {visual.shape[0] - 1}; need r<{U_STOP}")
    if min_q < 0 or audio.shape[0] <= max_q:
        raise ScoringError(f"audio embedding support [{min_q}, {max_q}] is unavailable: rows={audio.shape[0]}")


def distance(visual_row: np.ndarray, audio_row: np.ndarray) -> np.float32:
    """One frozen float32 distance, including the specified +1e-6 term."""

    left = np.asarray(visual_row, dtype=np.float32)
    right = np.asarray(audio_row, dtype=np.float32)
    if left.shape != right.shape:
        raise ScoringError(f"distance row shape mismatch: {left.shape} != {right.shape}")
    delta = left - right + np.float32(EPSILON)
    return np.sqrt(np.sum(delta * delta, dtype=np.float32), dtype=np.float32)


def distance_matrix(
    visual: np.ndarray,
    audio: np.ndarray,
    *,
    r_values: Iterable[int] = range(U_START, U_STOP),
    lag_min: int = LAG_MIN,
    lag_max: int = LAG_MAX,
) -> np.ndarray:
    """Return genuinely supported distances, without padded edge rows."""

    visual_array = _embedding(visual, "visual")
    audio_array = _embedding(audio, "audio")
    starts = tuple(int(value) for value in r_values)
    if not starts:
        raise ScoringError("at least one visual support row is required")
    _support(visual_array, audio_array, lag_min=lag_min, lag_max=lag_max)
    if min(starts) < 0 or max(starts) >= visual_array.shape[0]:
        raise ScoringError("visual support row is outside the embedding")
    lags = tuple(range(int(lag_min), int(lag_max) + 1))
    matrix = np.empty((len(starts), len(lags)), dtype=np.float32)
    for row_index, r in enumerate(starts):
        for lag_index, lag in enumerate(lags):
            q = r + lag
            if q < 0 or q >= audio_array.shape[0]:
                raise ScoringError(f"unsupported audio row q={q} for r={r}, lag={lag}")
            matrix[row_index, lag_index] = distance(visual_array[r], audio_array[q])
    return matrix


def lag_curve(
    visual: np.ndarray,
    audio: np.ndarray,
    *,
    lag_min: int = LAG_MIN,
    lag_max: int = LAG_MAX,
    r_values: Iterable[int] = range(U_START, U_STOP),
) -> np.ndarray:
    """Compute z[j] as a float64 mean over U, preserving float32 distances."""

    matrix = distance_matrix(visual, audio, r_values=r_values, lag_min=lag_min, lag_max=lag_max)
    return np.mean(matrix, axis=0, dtype=np.float64)


def curve_metrics(curve: Sequence[float], *, lag_min: int = LAG_MIN) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ScoringError("lag curve must be a finite one-dimensional array")
    best_index = int(np.argmin(values))  # NumPy argmin keeps the first tie.
    return {
        "z": [float(value) for value in values],
        "C": float(np.median(values) - np.min(values)),
        "D": float(np.min(values)),
        "offset": int(-int(lag_min) - best_index),
        "best_index": best_index,
        "best_lag": int(lag_min) + best_index,
    }


def fixed_metrics(visual: np.ndarray, audio: np.ndarray) -> dict[str, Any]:
    """The A endpoint: U=25..114, q=r+j-15, j=0..30."""

    curve = lag_curve(visual, audio, lag_min=LAG_MIN, lag_max=LAG_MAX)
    return curve_metrics(curve, lag_min=LAG_MIN)


def fixed_anchor_gain(natural: Mapping[str, Any], candidate: Mapping[str, Any], k0: int) -> float:
    return float(natural["z"][int(k0)] - candidate["z"][int(k0)])


def sync_c_gain(natural: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    return float(candidate["C"] - natural["C"])


def sync_d_gain(natural: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    return float(natural["D"] - candidate["D"])


def delay_lag_curve(
    visual: np.ndarray,
    audio: np.ndarray,
    *,
    delay_frames: int = SHIFT_FRAMES,
    lag_min: int = LAG_MIN,
    lag_max: int = LAG_MAX,
    r_values: Iterable[int] = range(U_START, U_STOP),
) -> np.ndarray:
    """Curve for ``A_delay[q] = A_N[q-delay_frames]`` without padding.

    The index domain remains the original video/audio clock.  Consequently a
    positive delay moves the first minimum to a larger lag and decreases the
    reported ``15-argmin`` offset by the same amount.
    """

    visual_array = _embedding(visual, "visual")
    audio_array = _embedding(audio, "audio")
    starts = tuple(int(value) for value in r_values)
    lags = tuple(range(int(lag_min), int(lag_max) + 1))
    if delay_frames < 0:
        raise ScoringError("delay_frames must be non-negative")
    _support(visual_array, audio_array, lag_min=lag_min - delay_frames, lag_max=lag_max - delay_frames)
    matrix = np.empty((len(starts), len(lags)), dtype=np.float32)
    for row_index, r in enumerate(starts):
        for lag_index, lag in enumerate(lags):
            q = r + lag - int(delay_frames)
            if q < 0 or q >= audio_array.shape[0]:
                raise ScoringError(f"unsupported delayed audio row q={q}")
            matrix[row_index, lag_index] = distance(visual_array[r], audio_array[q])
    return np.mean(matrix, axis=0, dtype=np.float64)


def matched_delay_curves(
    visual: np.ndarray,
    audio: np.ndarray,
    *,
    delay_frames: int = SHIFT_FRAMES,
) -> dict[str, Any]:
    """Return the natural/delayed curves on the frozen matched domain.

    The delayed curve is evaluated at ``q=r+lag-5`` as required by the A
    contract; ``delayed_audio[q]`` is reconstructed as ``natural[q-5]``.
    ``delay_lag_curve`` exposes the original clock and is used for the shift
    direction/damage checks.
    """

    if delay_frames != SHIFT_FRAMES:
        raise ScoringError("A timing control is frozen at a five-frame delay")
    # The matched domain is -10..20.  For the delayed signal, q=r+lag-5 and
    # A_delay[q]=A_N[q-5], so the natural index is r+lag-10.
    natural = lag_curve(visual, audio, lag_min=LAG_MIN + delay_frames, lag_max=LAG_MAX + delay_frames)
    delayed = delay_lag_curve(visual, audio, delay_frames=delay_frames, lag_min=LAG_MIN + delay_frames, lag_max=LAG_MAX + delay_frames)
    return {
        "lag_min": int(LAG_MIN + delay_frames),
        "lag_max": int(LAG_MAX + delay_frames),
        "q_formula": "q=r+lag-5",
        "natural": [float(value) for value in natural],
        "delayed": [float(value) for value in delayed],
        "max_abs_difference": float(np.max(np.abs(natural - delayed))),
    }


def synthetic_delay_control(visual: np.ndarray, audio: np.ndarray, *, delay_frames: int = SHIFT_FRAMES) -> dict[str, Any]:
    natural_curve = lag_curve(visual, audio, lag_min=LAG_MIN, lag_max=LAG_MAX)
    delayed_curve = delay_lag_curve(visual, audio, delay_frames=delay_frames, lag_min=LAG_MIN, lag_max=LAG_MAX)
    natural = curve_metrics(natural_curve, lag_min=LAG_MIN)
    delayed = curve_metrics(delayed_curve, lag_min=LAG_MIN)
    k0 = int(natural["best_index"])
    matched = matched_delay_curves(visual, audio, delay_frames=delay_frames)
    # The damage is deliberately uncompensated at the original natural k0.
    damage = float(delayed_curve[k0] - natural_curve[k0])
    return {
        "natural": natural,
        "delayed": delayed,
        "natural_k0": k0,
        "uncompensated_damage": damage,
        "best_index_shift": int(delayed["best_index"] - natural["best_index"]),
        "offset_change": int(delayed["offset"] - natural["offset"]),
        "matched_domain": matched,
    }


def make_bootstrap_indices(
    group_count: int,
    *,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
) -> np.ndarray:
    if int(group_count) < 1 or int(draws) < 1:
        raise ScoringError("bootstrap group_count/draws must be positive")
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    return rng.integers(0, int(group_count), size=(int(draws), int(group_count)), dtype=np.int64)


def bootstrap_indices_sha256(indices: np.ndarray) -> str:
    array = np.asarray(indices, dtype=np.int64)
    if array.ndim != 2:
        raise ScoringError("bootstrap indices must be a matrix")
    return sha256_bytes(np.ascontiguousarray(array).tobytes())


def bootstrap_summary(
    values: Sequence[float],
    indices: np.ndarray,
    *,
    confidence: float,
    positive_threshold: float = 0.0,
) -> dict[str, Any]:
    values_array = np.asarray(values, dtype=np.float64)
    index_array = np.asarray(indices, dtype=np.int64)
    if values_array.ndim != 1 or values_array.size < 1 or not np.isfinite(values_array).all():
        raise ScoringError("bootstrap values must be finite and one-dimensional")
    if index_array.ndim != 2 or index_array.shape[1] != values_array.size:
        raise ScoringError("bootstrap indices do not match group values")
    if np.any(index_array < 0) or np.any(index_array >= values_array.size):
        raise ScoringError("bootstrap index is out of range")
    if not 0.0 < confidence < 1.0:
        raise ScoringError("confidence must be between zero and one")
    estimates = np.mean(values_array[index_array], axis=1, dtype=np.float64)
    alpha = (1.0 - float(confidence)) / 2.0
    interval = np.quantile(estimates, [alpha, 1.0 - alpha], method="linear")
    return {
        "mean": float(np.mean(values_array, dtype=np.float64)),
        "ci": [float(interval[0]), float(interval[1])],
        "confidence": float(confidence),
        "draws": int(index_array.shape[0]),
        "group_count": int(values_array.size),
        "group_positive_count": int(np.count_nonzero(values_array > positive_threshold)),
        "indices_sha256": bootstrap_indices_sha256(index_array),
        "seed": BOOTSTRAP_SEED,
        "method": "numpy.PCG64; np.quantile(method=linear)",
    }


def _lookup_cell(cells: Any, sample_id: str, arm: str, seed: int, repeat_index: int = 0) -> Mapping[str, Any] | None:
    key = cell_key(sample_id, "wav2lip", arm, seed, repeat_index)
    if isinstance(cells, Mapping):
        for candidate in (key, (sample_id, arm, int(seed), int(repeat_index)), f"{sample_id}/{arm}/{seed}/{repeat_index}"):
            try:
                value = cells.get(candidate)  # type: ignore[arg-type]
            except TypeError:
                value = None
            if isinstance(value, Mapping):
                return value
        # Accept a keyed list-like map from JSON where producer omitted model.
        for value in cells.values():
            if isinstance(value, Mapping) and str(value.get("sample_id")) == sample_id and str(value.get("arm")) == arm and int(value.get("seed", -1)) == int(seed) and int(value.get("repeat_index", 0)) == int(repeat_index):
                return value
    if isinstance(cells, Sequence) and not isinstance(cells, (str, bytes)):
        for value in cells:
            if isinstance(value, Mapping) and str(value.get("sample_id")) == sample_id and str(value.get("arm")) == arm and int(value.get("seed", -1)) == int(seed) and int(value.get("repeat_index", 0)) == int(repeat_index):
                return value
    return None


def _cell_metric(cell: Mapping[str, Any]) -> dict[str, Any]:
    if "metrics" in cell and isinstance(cell["metrics"], Mapping) and "z" in cell["metrics"]:
        value = dict(cell["metrics"])
        return value
    visual = cell.get("visual")
    audio = cell.get("audio")
    if visual is None or audio is None:
        raise ScoringError(f"cell lacks embeddings: {cell.get('sample_id')}/{cell.get('arm')}/{cell.get('seed')}")
    return fixed_metrics(np.asarray(visual), np.asarray(audio))


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and bool(np.isfinite(float(value)))


def analyze_model_cells(
    records: Sequence[Mapping[str, Any]],
    cells: Any,
    *,
    model: str = "wav2lip",
    bootstrap_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a complete model analysis from cell embeddings/metrics.

    This is the producer-side calculation.  ``validate.py`` repeats these
    operations in its own code path before accepting an analysis artifact.
    """

    formal = list(records[:12])
    group_labels = [str(row.get("source_group", row.get("sample_id", ""))) for row in formal]
    sample_ids = [str(row.get("sample_id", "")) for row in formal]
    if len(group_labels) != 12 or len(set(group_labels)) != 12 or len(set(sample_ids)) != 12:
        raise ScoringError("analysis requires twelve unique formal source groups")
    indices = bootstrap_indices if bootstrap_indices is not None else make_bootstrap_indices(len(group_labels))
    if np.asarray(indices).shape[1] != 12:
        raise ScoringError("analysis bootstrap indices must have twelve columns")
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for record, group, sample_id in zip(formal, group_labels, sample_ids, strict=True):
        values: dict[str, Mapping[str, Any]] = {}
        for arm in ("N", "C"):
            for seed in SEEDS:
                item = _lookup_cell(cells, sample_id, arm, seed, 0)
                if item is None:
                    missing.append(cell_key(sample_id, model, arm, seed, 0))
                else:
                    values[f"{arm}{seed}"] = item
        repeat = _lookup_cell(cells, sample_id, "N", 42, 1)
        if sample_id in sample_ids[:2] and repeat is None:
            missing.append(cell_key(sample_id, model, "N", 42, 1))
        if any(name not in values for name in ("N42", "N43", "C42", "C43")):
            continue
        natural42 = _cell_metric(values["N42"])
        natural43 = _cell_metric(values["N43"])
        candidate42 = _cell_metric(values["C42"])
        candidate43 = _cell_metric(values["C43"])
        k0 = int(natural42["best_index"])
        if not 0 <= k0 < len(natural42["z"]):
            raise ScoringError(f"invalid natural k0 for {sample_id}")
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "source_group": group,
            "k0": k0,
            "k0_lag": int(natural42.get("best_lag", k0 + LAG_MIN)),
            "natural_42": natural42,
            "natural_43": natural43,
            "candidate_42": candidate42,
            "candidate_43": candidate43,
            "fixed_anchor_gain_42": fixed_anchor_gain(natural42, candidate42, k0),
            "fixed_anchor_gain_43": fixed_anchor_gain(natural43, candidate43, k0),
            "sync_c_gain_42": sync_c_gain(natural42, candidate42),
            "sync_c_gain_43": sync_c_gain(natural43, candidate43),
            "sync_d_gain_42": sync_d_gain(natural42, candidate42),
            "sync_d_gain_43": sync_d_gain(natural43, candidate43),
            "natural_seed_difference": {
                "A": float(natural43["z"][k0] - natural42["z"][k0]),
                "C": float(natural43["C"] - natural42["C"]),
                "D": float(natural43["D"] - natural42["D"]),
            },
            "candidate_seed_difference": {
                "C": float(candidate43["C"] - candidate42["C"]),
                "D": float(candidate43["D"] - candidate42["D"]),
                "A": float(candidate43["z"][k0] - candidate42["z"][k0]),
            },
        }
        delay_control: dict[str, Any] | None = None
        n42_visual = values["N42"].get("visual")
        n42_audio = values["N42"].get("audio")
        if n42_visual is not None and n42_audio is not None:
            delay_control = synthetic_delay_control(np.asarray(n42_visual), np.asarray(n42_audio))
        elif isinstance(values["N42"].get("delay_control"), Mapping):
            delay_control = dict(values["N42"]["delay_control"])
        row["delay_control"] = delay_control
        if repeat is not None:
            repeat_metric = _cell_metric(repeat)
            row["natural_42_repeat"] = repeat_metric
            row["repeat_differences"] = {
                "C": float(repeat_metric["C"] - natural42["C"]),
                "D": float(repeat_metric["D"] - natural42["D"]),
                "A_at_k0": float(repeat_metric["z"][k0] - natural42["z"][k0]),
                "offset": int(repeat_metric["offset"] - natural42["offset"]),
                "pixel_max_abs": repeat.get("pixel_max_abs"),
                "pixel_mean_abs": repeat.get("pixel_mean_abs"),
                "pixel_different_bytes": repeat.get("pixel_different_bytes"),
            }
        rows.append(row)

    complete = not missing and len(rows) == 12
    if complete:
        k0_values = [int(row["k0"]) for row in rows]
        damage = [float(row["delay_control"]["uncompensated_damage"]) for row in rows if isinstance(row.get("delay_control"), Mapping)]
        shifted = [float(row["delay_control"]["uncompensated_damage"]) > 0 for row in rows if isinstance(row.get("delay_control"), Mapping)]
        natural_interior = all(1 <= value <= 29 for value in k0_values)
        delay_summary = bootstrap_summary(damage, indices, confidence=0.95) if len(damage) == 12 else None
        repeat_rows = [row for row in rows if "repeat_differences" in row]
        repeat_pass = len(repeat_rows) == 2 and all(
            abs(float(row["repeat_differences"][name])) <= 0.01 for row in repeat_rows for name in ("C", "D", "A_at_k0")
        ) and all(abs(int(row["repeat_differences"]["offset"])) <= 1 for row in repeat_rows)
        controls_pass = bool(natural_interior and len(damage) == 12 and sum(shifted) >= 10 and delay_summary and delay_summary["ci"][0] > 0 and repeat_pass)
    else:
        k0_values = []
        delay_summary = None
        natural_interior = False
        shifted = []
        repeat_pass = False
        controls_pass = False

    def group_mean(key: str) -> list[float]:
        return [float((float(row[key + "_42"]) + float(row[key + "_43"])) / 2.0) for row in rows]

    gains = {
        "fixed_anchor": group_mean("fixed_anchor_gain"),
        "sync_c": group_mean("sync_c_gain"),
        "sync_d": group_mean("sync_d_gain"),
    }
    # Natural seed noise uses the same endpoint definitions as the scientific
    # contrasts; A is explicitly evaluated at each group's frozen k0.
    noise_floor = {
        "fixed_anchor": float(np.mean([abs(float(row["natural_seed_difference"]["A"])) for row in rows], dtype=np.float64)) if rows else None,
        "sync_c": float(np.mean([abs(float(row["natural_seed_difference"]["C"])) for row in rows], dtype=np.float64)) if rows else None,
        "sync_d": float(np.mean([abs(float(row["natural_seed_difference"]["D"])) for row in rows], dtype=np.float64)) if rows else None,
    }
    gain_bootstrap = {
        "fixed_anchor": bootstrap_summary(gains["fixed_anchor"], indices, confidence=0.99) if len(gains["fixed_anchor"]) == 12 else None,
        "sync_c": bootstrap_summary(gains["sync_c"], indices, confidence=0.99) if len(gains["sync_c"]) == 12 else None,
        "sync_d": bootstrap_summary(gains["sync_d"], indices, confidence=0.99) if len(gains["sync_d"]) == 12 else None,
    }
    joint_positive = int(sum(a > 0 and c > 0 for a, c in zip(gains["fixed_anchor"], gains["sync_c"], strict=True)))
    delta_sync_c = gains["sync_c"]
    if complete and controls_pass and gain_bootstrap["fixed_anchor"] and gain_bootstrap["sync_c"] and gain_bootstrap["sync_d"]:
        signal = bool(
            float(np.mean(delta_sync_c)) > 0.050
            and gain_bootstrap["sync_c"]["ci"][0] > 0
            and gain_bootstrap["fixed_anchor"]["ci"][0] > 0
            and float(np.mean(gains["sync_c"])) > float(noise_floor["sync_c"])
            and float(np.mean(gains["fixed_anchor"])) > float(noise_floor["fixed_anchor"])
            and gain_bootstrap["sync_d"]["ci"][0] > -0.100
            and joint_positive >= 10
        )
    else:
        signal = False
    if not complete:
        status = "ENGINEERING_INCOMPLETE"
    elif not controls_pass:
        status = "CONTROL_FAILED"
    elif signal:
        status = "SIGNAL"
    else:
        status = "NO_REPLACEMENT_SIGNAL_ESTABLISHED"
    return {
        "schema_version": 1,
        "model": model,
        "status": status,
        "formal_groups": len(rows),
        "formal_cells": int(len(rows) * 4 + (2 if complete else 0)),
        "expected_formal_cells": 50,
        "missing_cells": missing,
        "rows": rows,
        "controls": {
            "pass": controls_pass,
            "natural_k0_interior": natural_interior,
            "k0_values": k0_values,
            "shifted_damage_positive_count": int(sum(shifted)),
            "shifted_damage": damage,
            "shifted_damage_bootstrap_95": delay_summary,
            "repeat_pass": repeat_pass,
            "repeat_tolerance": {"C": 0.01, "D": 0.01, "fixed_anchor": 0.01, "offset_frames": 1},
        },
        "noise_floor": noise_floor,
        "gains": gains,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "draws": BOOTSTRAP_DRAWS,
            "indices_sha256": bootstrap_indices_sha256(np.asarray(indices)),
            "fixed_anchor": gain_bootstrap["fixed_anchor"],
            "sync_c": gain_bootstrap["sync_c"],
            "sync_d": gain_bootstrap["sync_d"],
        },
        "signal_gate": {
            "pass": signal,
            "mean_delta_sync_c_threshold": 0.050,
            "mean_delta_sync_c": float(np.mean(delta_sync_c)) if delta_sync_c else None,
            "joint_positive_count": joint_positive,
            "joint_positive_required": 10,
            "replacement_confirmed": False,
        },
        "replacement_confirmed": False,
        "training_authorized": False,
        "generalization_established": False,
        "human_status": "pending",
    }


__all__ = [
    "ScoringError", "analyze_model_cells", "bootstrap_indices_sha256", "bootstrap_summary", "curve_metrics",
    "delay_lag_curve", "distance", "distance_matrix", "fixed_anchor_gain", "fixed_metrics", "lag_curve",
    "make_bootstrap_indices", "matched_delay_curves", "sync_c_gain", "sync_d_gain", "synthetic_delay_control",
]
