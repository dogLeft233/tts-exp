"""Frozen SyncNet support-window and metric calculations."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def frozen_support_windows(audio_samples: int, video_frames: int, *, vshift: int = 15, fps: float = 25.0, mel_hz: float = 80.0) -> dict[str, Any]:
    length = int(audio_samples)
    if length <= 0 or video_frames < 5:
        return {"windows": [], "support_count": 0, "reason": "SHORT_INPUT"}
    windows: list[int] = []
    for t in range(int(video_frames) - 4):
        generator_start = int(np.floor(3.2 * t) * 200 - 401)
        generator_end = int((np.floor(3.2 * (t + 4)) + 15) * 200 + 400)
        all_lags_supported = True
        for lag in range(-int(vshift), int(vshift) + 1):
            audio_start = (t + lag) * 640 - 1
            audio_end = (t + lag) * 640 + 3440
            if not (generator_start >= 0 and generator_end <= length and audio_start >= 0 and audio_end <= length):
                all_lags_supported = False
                break
        if all_lags_supported:
            windows.append(t)
    return {"windows": sorted(set(windows)), "support_count": len(set(windows)), "audio_samples": length, "video_frames": int(video_frames), "vshift": int(vshift)}


def score_distance_matrix(matrix: np.ndarray, support: Sequence[int], *, lags: Sequence[int], anchor_lag: int | None = None) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(lags):
        raise ValueError("distance matrix/lag shape mismatch")
    support_idx = np.asarray(list(support), dtype=np.int64)
    if support_idx.size == 0 or np.any(support_idx < 0) or np.any(support_idx >= values.shape[0]):
        return {"status": "BASELINE_SUPPORT_INELIGIBLE", "support_count": int(support_idx.size)}
    selected = values[support_idx]
    if not np.isfinite(selected).all():
        return {"status": "CELL_FAILURE", "support_count": int(support_idx.size)}
    curve = selected.mean(axis=0)
    min_value = float(curve.min())
    candidates = np.flatnonzero(curve == min_value)
    lag_index = int(candidates[0])
    lag = int(lags[lag_index])
    d0_index = list(lags).index(0) if 0 in lags else None
    anchor_index = list(lags).index(int(anchor_lag)) if anchor_lag is not None and int(anchor_lag) in lags else None
    return {"status": "COMPLETE", "support_count": int(support_idx.size), "lag": lag, "D": min_value, "C": float(np.median(curve) - min_value), "D0": float(curve[d0_index]) if d0_index is not None else None, "D_anchor": float(curve[anchor_index]) if anchor_index is not None else None, "anchor_lag": int(anchor_lag) if anchor_index is not None else None, "curve": curve.tolist(), "lags": [int(value) for value in lags]}


def cross_cell_interaction(nn: float, en: float, ne: float, ee: float) -> dict[str, float]:
    return {"delta_gen": float(en - nn), "delta_audio": float(ne - nn), "interaction": float(ee - en - ne + nn)}


__all__ = ["cross_cell_interaction", "frozen_support_windows", "score_distance_matrix"]
