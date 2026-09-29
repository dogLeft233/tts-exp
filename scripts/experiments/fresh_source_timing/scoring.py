"""SyncNet embedding scoring for the integer-delay timing transfer.

The actual GPU forward is owned by A.  C only consumes the immutable visual
and audio embeddings emitted by that forward.  Keeping this module array based
makes the score reconstructable on CPU and makes it impossible to answer C by
re-indexing a scorer-only audio delay.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    DELAY_LAG_START,
    LAG_COUNT,
    NATURAL_LAG_START,
    U_ROWS,
    TimingError,
    file_sha256,
)


def _embedding(value: Any, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] <= 0 or result.shape[0] <= max(U_ROWS) + 5:
        raise TimingError(f"{name} must be a 2-D embedding with support through U+5: {result.shape}")
    if not np.isfinite(result).all():
        raise TimingError(f"{name} contains non-finite values")
    return result


def _rows(rows: Sequence[int] | None, limit: int) -> tuple[int, ...]:
    selected = tuple(int(row) for row in (U_ROWS if rows is None else rows))
    if not selected or min(selected) < 0 or max(selected) >= limit:
        raise TimingError(f"scoring rows outside embedding support: {selected}")
    return selected


def distance_matrix(
    visual: np.ndarray,
    audio: np.ndarray,
    *,
    lag_start: int,
    rows: Sequence[int] | None = None,
) -> np.ndarray:
    """Build the supported distance matrix for one physical lag domain.

    For column ``j`` the audio row is ``q=r+lag_start+j``.  This is the same
    float32 distance contract as the A SyncNet worker, with the average over
    selected rows deferred to :func:`metrics_from_matrix`.
    """

    visual_value = _embedding(visual, name="visual embedding")
    audio_value = _embedding(audio, name="audio embedding")
    if visual_value.shape[1] != audio_value.shape[1]:
        raise TimingError(f"embedding dimensions differ: {visual_value.shape}/{audio_value.shape}")
    selected = _rows(rows, visual_value.shape[0])
    lags = np.arange(int(lag_start), int(lag_start) + LAG_COUNT, dtype=np.int64)
    result = np.empty((len(selected), LAG_COUNT), dtype=np.float32)
    epsilon = np.float32(1e-6)
    for output_row, row in enumerate(selected):
        indices = np.int64(row) + lags
        if int(indices.min()) < 0 or int(indices.max()) >= audio_value.shape[0]:
            raise TimingError(
                f"lag domain lacks real support for row {row}: "
                f"{int(lags[0])}..{int(lags[-1])}"
            )
        difference = visual_value[row : row + 1] - audio_value[indices]
        # Keep arithmetic in float32 to match the persisted SyncNet matrices.
        squared = np.square(difference + epsilon, dtype=np.float32)
        result[output_row] = np.sqrt(np.sum(squared, axis=1, dtype=np.float32), dtype=np.float32)
    return result


def build_distance_matrix(
    visual: np.ndarray,
    audio: np.ndarray,
    lag_start: int,
    rows: Sequence[int] | None = None,
) -> np.ndarray:
    """Positional-argument alias used by older scoring callers."""

    return distance_matrix(visual, audio, lag_start=lag_start, rows=rows)


def curve(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != LAG_COUNT or value.shape[0] < 1 or not np.isfinite(value).all():
        raise TimingError(f"distance matrix is malformed: {value.shape}")
    return np.mean(value, axis=0, dtype=np.float64)


def metrics_from_curve(values: np.ndarray, *, lag_start: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (LAG_COUNT,) or not np.isfinite(values).all():
        raise TimingError(f"distance curve must have {LAG_COUNT} finite values")
    minimum_index = int(np.argmin(values))
    ordered = np.sort(values)
    best_lag = int(lag_start) + minimum_index
    minimum = float(values[minimum_index])
    return {
        "curve": [float(item) for item in values],
        "lag_start": int(lag_start),
        "lags": [int(lag_start) + index for index in range(LAG_COUNT)],
        "min_index": minimum_index,
        "best_lag": best_lag,
        "offset": -best_lag,
        "D": minimum,
        "C": float(np.median(values) - minimum),
        "sync_d": minimum,
        "sync_c": float(np.median(values) - minimum),
        "minimum": minimum,
        "second_minimum": float(ordered[1]),
        "peak_gap": float(ordered[1] - ordered[0]),
    }


def metrics_from_matrix(matrix: np.ndarray, *, lag_start: int) -> dict[str, Any]:
    return metrics_from_curve(curve(matrix), lag_start=lag_start)


def _row_distances(visual: np.ndarray, audio: np.ndarray, rows: np.ndarray, lag: int) -> np.ndarray:
    indices = rows.astype(np.int64) + int(lag)
    if int(indices.min()) < 0 or int(indices.max()) >= audio.shape[0]:
        raise TimingError("fixed-anchor support is outside audio embedding")
    difference = visual[rows] - audio[indices]
    return np.sqrt(np.sum(np.square(difference + np.float32(1e-6), dtype=np.float32), axis=1, dtype=np.float32), dtype=np.float32).astype(np.float64)


def score_embeddings(
    natural_visual: np.ndarray,
    delay_visual: np.ndarray,
    natural_audio: np.ndarray,
    *,
    rows: Sequence[int] = U_ROWS,
    delay_shift_frames: int = 5,
) -> dict[str, Any]:
    """Score N42 and a fresh DELAY video against untouched N audio.

    The DELAY audio embedding (if one exists in A) is intentionally not an
    argument here.  Passing it would turn C into a scorer-only audio shift and
    would not test a generator forward driven by DELAY PCM.
    """

    if int(delay_shift_frames) != 5:
        raise TimingError("C compensation is fixed at +5 frames")
    natural_video = _embedding(natural_visual, name="natural visual embedding")
    delayed_video = _embedding(delay_visual, name="delay visual embedding")
    natural_audio_value = _embedding(natural_audio, name="natural audio embedding")
    if natural_video.shape[1] != delayed_video.shape[1] or natural_video.shape[1] != natural_audio_value.shape[1]:
        raise TimingError("N/DELAY visual and natural-audio dimensions differ")
    selected = np.asarray(_rows(rows, min(natural_video.shape[0], delayed_video.shape[0])), dtype=np.int64)
    natural_matrix = distance_matrix(natural_video, natural_audio_value, lag_start=NATURAL_LAG_START, rows=selected)
    delay_matrix = distance_matrix(delayed_video, natural_audio_value, lag_start=DELAY_LAG_START, rows=selected)
    natural = metrics_from_matrix(natural_matrix, lag_start=NATURAL_LAG_START)
    delay = metrics_from_matrix(delay_matrix, lag_start=DELAY_LAG_START)
    lag_n = int(natural["best_lag"])
    row_values = selected.astype(np.int64)
    anchor_distances = _row_distances(natural_video, natural_audio_value, row_values, lag_n)
    uncompensated = _row_distances(delayed_video, natural_audio_value, row_values, lag_n)
    compensated_rows = row_values + int(delay_shift_frames)
    if int(compensated_rows.min()) < 0 or int(compensated_rows.max()) >= delayed_video.shape[0]:
        raise TimingError("+5 compensation is outside DELAY visual support")
    compensated = _row_distances(delayed_video, natural_audio_value, compensated_rows, lag_n)
    d_anchor = float(np.mean(anchor_distances, dtype=np.float64))
    d_unc = float(np.mean(uncompensated, dtype=np.float64))
    d_comp = float(np.mean(compensated, dtype=np.float64))
    rescue = d_unc - d_comp
    return {
        "natural": natural,
        "delay": delay,
        "matrix_natural": natural_matrix,
        "matrix_delay": delay_matrix,
        "rows": [int(row) for row in selected],
        "lag_n": lag_n,
        "offset_delta": int(delay["offset"] - natural["offset"]),
        "expected_offset_delta": int(delay_shift_frames),
        "d_anchor": d_anchor,
        "d_unc": d_unc,
        "d_comp": d_comp,
        "rescue": float(rescue),
        "residual": float(d_comp - d_anchor),
        "compensation_frames": int(delay_shift_frames),
    }


def score_syncnet_embeddings(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Explicit alias documenting that this is the SyncNet embedding route."""

    return score_embeddings(*args, **kwargs)


def score_embedding_pair(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return score_embeddings(*args, **kwargs)


def parse_syncnet_log(path: str | Path) -> dict[str, float | int]:
    """Parse the optional raw SyncNet log without running a model."""

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    matches = {
        "sync_c": re.findall(r"Confidence:\s+([-+]?\d+(?:\.\d+)?)", text),
        "sync_d": re.findall(r"Min dist:\s+([-+]?\d+(?:\.\d+)?)", text),
        "av_offset": re.findall(r"AV offset:\s+(-?\d+)", text),
    }
    if any(len(values) != 1 for values in matches.values()):
        raise TimingError(f"SyncNet log is missing or ambiguous: {path}")
    return {
        "sync_c": float(matches["sync_c"][0]),
        "sync_d": float(matches["sync_d"][0]),
        "av_offset": int(matches["av_offset"][0]),
    }


def _load_array(path: Path, expected_sha: str | None = None, *, key: str | None = None) -> np.ndarray:
    if not path.is_file() or (expected_sha is not None and file_sha256(path) != expected_sha):
        raise TimingError(f"embedding artifact hash binding failed: {path}")
    try:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.ndarray):
            result = loaded
        else:
            with loaded:
                candidates = [key] if key else []
                candidates.extend(["visual", "visual_embedding", "audio", "audio_embedding", "embedding"])
                selected = next((name for name in candidates if name and name in loaded.files), None)
                if selected is None:
                    raise TimingError(f"embedding key is missing: {path}")
                result = loaded[selected]
    except (ValueError, OSError) as exc:
        raise TimingError(f"embedding artifact is malformed: {path}") from exc
    return np.asarray(result)


def load_embedding_array(item: Any, *, name: str, key: str | None = None, base: Path | None = None) -> np.ndarray:
    """Load a hash-bound ``.npy``/``.npz`` entry or an inline synthetic array."""

    if isinstance(item, (np.ndarray, list, tuple)):
        return _embedding(item, name=name)
    if not isinstance(item, Mapping):
        raise TimingError(f"{name} binding is malformed")
    if "array" in item:
        return _embedding(item["array"], name=name)
    path_value = item.get("path") or item.get("file")
    if path_value is None:
        raise TimingError(f"{name} path is missing")
    path = Path(str(path_value))
    if not path.is_absolute() and base is not None:
        path = base / path
    selected_key = key or (str(item.get("key")) if item.get("key") else None)
    value = _load_array(path.resolve(), str(item.get("sha256")) if item.get("sha256") else None, key=selected_key)
    return _embedding(value, name=name)


def load_worker_embeddings(worker: Mapping[str, Any], *, base: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read A's worker-style visual/audio bindings with their hashes."""

    visual_item = worker.get("visual") or worker.get("visual_embedding")
    audio_item = worker.get("audio_embedding") or worker.get("audio")
    if visual_item is None or audio_item is None:
        raise TimingError("A worker row lacks visual/audio embeddings")
    visual = load_embedding_array(visual_item, name="visual embedding", key="visual", base=base)
    audio = load_embedding_array(audio_item, name="audio embedding", key="audio_embedding", base=base)
    if visual.shape != audio.shape:
        raise TimingError(f"worker visual/audio shape differs: {visual.shape}/{audio.shape}")
    return visual, audio
