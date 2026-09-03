"""SyncNet distance-matrix parity and original-clock local confidence."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import signal


def compute_local_scores(dists: np.ndarray, *, vshift: int = 15, median_width: int = 9) -> dict[str, Any]:
    values = np.asarray(dists, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 2 * vshift + 1:
        raise ValueError(f"expected distance matrix [T,{2 * vshift + 1}], found {values.shape}")
    if values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("distance matrix must be non-empty and finite")
    import torch
    values_tensor = torch.from_numpy(values.T.copy())
    mdist_tensor = torch.mean(values_tensor, dim=1)
    mdist = mdist_tensor.numpy()
    j_star = int(torch.argmin(mdist_tensor).item())
    median_distance_value = torch.median(mdist_tensor)
    median_distance = float(median_distance_value.item())
    sync_d = float(mdist_tensor[j_star].item())
    sync_c = float((median_distance_value - mdist_tensor[j_star]).item())
    av_offset_frames = int(vshift - j_star)
    local_c_raw = (median_distance_value - values_tensor[j_star, :]).numpy()
    local_c_filtered = signal.medfilt(local_c_raw, kernel_size=median_width)
    if not np.all(np.isfinite(local_c_filtered)):
        raise ValueError("local confidence is non-finite")
    return {
        "mdist": mdist,
        "j_star": j_star,
        "av_offset_frames": av_offset_frames,
        "sync_d": sync_d,
        "sync_c": float(sync_c),
        "local_c_raw": np.asarray(local_c_raw, dtype=np.float64),
        "local_c_filtered": np.asarray(local_c_filtered, dtype=np.float64),
        "median_distance": median_distance,
        "vshift": int(vshift),
        "median_filter_width": int(median_width),
    }


def parity_report(
    scores: Mapping[str, Any],
    *,
    upstream_offset: int | None = None,
    upstream_confidence: float | None = None,
    upstream_distance: float | None = None,
    log_tolerance: float = 5e-4,
) -> dict[str, Any]:
    offset_ok = upstream_offset is None or int(scores["av_offset_frames"]) == int(upstream_offset)
    conf_error = None if upstream_confidence is None else abs(float(scores["sync_c"]) - float(upstream_confidence))
    dist_error = None if upstream_distance is None else abs(float(scores["sync_d"]) - float(upstream_distance))
    confidence_ok = conf_error is None or conf_error <= 1e-6
    distance_ok = dist_error is None or dist_error <= log_tolerance
    return {
        "offset_exact": bool(offset_ok),
        "confidence_abs_error": conf_error,
        "confidence_within_1e-6": bool(confidence_ok),
        "distance_abs_error": dist_error,
        "distance_within_tolerance": bool(distance_ok),
        "passed": bool(offset_ok and confidence_ok and distance_ok),
    }


def normalize_tracks(tracks: Any) -> list[tuple[int, Mapping[str, Any]]]:
    if isinstance(tracks, Mapping):
        pairs = [(int(key), value) for key, value in tracks.items()]
    elif isinstance(tracks, Sequence) and not isinstance(tracks, (str, bytes)):
        pairs = list(enumerate(tracks))
    else:
        raise ValueError("tracks must be a mapping or sequence")
    normalized: list[tuple[int, Mapping[str, Any]]] = []
    for index, track in pairs:
        if not isinstance(track, Mapping) or "frame" not in track:
            raise ValueError(f"track {index} has no frame list")
        frames = list(track["frame"])
        if not frames:
            continue
        normalized.append((int(index), track))
    return normalized


def select_track(tracks: Any, *, min_track: int = 50) -> dict[str, Any]:
    candidates = normalize_tracks(tracks)
    eligible = [(index, track) for index, track in candidates if len(track["frame"]) > int(min_track)]
    if not eligible:
        raise ValueError(f"no face track longer than min_track={min_track}")
    index, selected = min(eligible, key=lambda item: (-len(item[1]["frame"]), item[0]))
    all_ranges = []
    for track_index, track in sorted(candidates):
        frames = list(track["frame"])
        all_ranges.append({
            "track_index": int(track_index),
            "frame_count": len(frames),
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "eligible": len(frames) > int(min_track),
        })
    selected_frames = list(selected["frame"])
    return {
        "selected_track_index": int(index),
        "selected_frame_count": len(selected_frames),
        "track_start_frame": int(selected_frames[0]),
        "track_end_frame": int(selected_frames[-1]),
        "track_ranges": all_ranges,
        "selection_reason": "max_frame_count_then_min_track_index",
    }


def map_local_scores(
    scores: Mapping[str, Any],
    *,
    track_start_frame: int,
    track_end_frame: int,
    audio_duration_s: float,
    fps: float = 25.0,
) -> dict[str, Any]:
    raw = np.asarray(scores["local_c_raw"], dtype=np.float64)
    filtered = np.asarray(scores["local_c_filtered"], dtype=np.float64)
    if raw.shape != filtered.shape:
        raise ValueError("raw and filtered local scores have different shapes")
    n = len(filtered)
    vshift = int(scores["vshift"])
    j_star = int(scores["j_star"])
    width = int(scores["median_filter_width"])
    edge = width // 2
    timestamps: list[float] = []
    values: list[float] = []
    source_rows: list[int] = []
    reasons: Counter[str] = Counter()
    for t in range(n):
        a = t + j_star - vshift
        if a < 0 or a >= n:
            reasons["shift_padding"] += 1
            continue
        if t < edge or t >= n - edge:
            reasons["median_filter_edge"] += 1
            continue
        center = (int(track_start_frame) + a + 2) / float(fps)
        if center < int(track_start_frame) / float(fps) or center >= (int(track_end_frame) + 1) / float(fps):
            reasons["outside_selected_track"] += 1
            continue
        if center < 0 or center >= float(audio_duration_s):
            reasons["outside_audio"] += 1
            continue
        timestamps.append(float(center))
        values.append(float(filtered[t]))
        source_rows.append(t)
    return {
        "timestamps_s": np.asarray(timestamps, dtype=np.float64),
        "local_c": np.asarray(values, dtype=np.float64),
        "source_rows": np.asarray(source_rows, dtype=np.int64),
        "excluded_counts": dict(sorted(reasons.items())),
        "retained_count": len(timestamps),
        "excluded_count": int(sum(reasons.values())),
        "audio_duration_s": float(audio_duration_s),
        "track_support_s": [int(track_start_frame) / float(fps), (int(track_end_frame) + 1) / float(fps)],
    }


def validate_paired_track_metadata(natural: Mapping[str, Any], tts: Mapping[str, Any]) -> None:
    keys = ("selected_track_index", "selected_frame_count", "track_start_frame", "track_end_frame", "track_ranges")
    for key in keys:
        if natural.get(key) != tts.get(key):
            raise ValueError(f"paired selected-track metadata mismatch: {key}")
