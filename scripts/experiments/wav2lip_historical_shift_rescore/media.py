from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_roi_retiming_oracle import media as parent_media

from . import config
from .common import RescoreError, bytes_sha256, file_sha256


def decode_frames(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    frames, evidence = parent_media.extract_bgr24_frames(path)
    if frames.ndim != 4 or frames.dtype != np.uint8 or frames.shape[1:] != (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3):
        raise RescoreError(f"historical video is not 224x224 BGR24: {path}")
    return frames, evidence


def validate_source_timeline(path: Path, evidence: dict[str, Any], expected_count: int) -> dict[str, Any]:
    probe = evidence.get("probe", {})
    stream = probe.get("stream")
    timestamps = probe.get("frames")
    if not isinstance(stream, dict) or not isinstance(timestamps, list) or len(timestamps) != expected_count:
        raise RescoreError(f"source video timeline is incomplete: {path}")
    if stream.get("width") != config.FRAME_WIDTH or stream.get("height") != config.FRAME_HEIGHT or stream.get("r_frame_rate") != "25/1":
        raise RescoreError(f"source video geometry/timing changed: {path}")
    pts = [int(item["best_effort_timestamp"]) for item in timestamps if isinstance(item, dict) and item.get("best_effort_timestamp") is not None]
    times = [float(item["best_effort_timestamp_time"]) for item in timestamps if isinstance(item, dict) and item.get("best_effort_timestamp_time") is not None]
    if len(pts) != expected_count or len(times) != expected_count or pts[0] != 0:
        raise RescoreError(f"source video PTS are incomplete: {path}")
    if any(right <= left for left, right in zip(pts, pts[1:])):
        raise RescoreError(f"source video PTS are not increasing: {path}")
    max_ms = max(abs(actual - index / config.FPS) * 1000.0 for index, actual in enumerate(times))
    if max_ms > 1.0:
        raise RescoreError(f"source video PTS deviate by more than 1 ms: {path}: {max_ms}")
    return {"frame_count": expected_count, "pts": pts, "pts_time": times, "max_deviation_ms": max_ms, "stream": dict(stream)}


def encode_lossless(frames: np.ndarray, output: Path) -> dict[str, Any]:
    if output.exists():
        raise RescoreError(f"refusing to overwrite derived video: {output}")
    evidence = parent_media.encode_video(frames, output)
    decoded, decoded_evidence = decode_frames(output)
    if not np.array_equal(decoded, frames):
        mismatch = np.argwhere(decoded != frames)
        location = tuple(int(item) for item in mismatch[0]) if mismatch.size else ()
        raise RescoreError(f"lossless pixel identity failed at {location}: {output}")
    timeline = validate_source_timeline(output, decoded_evidence, int(frames.shape[0]))
    evidence.update({"output_sha256": file_sha256(output), "pixel_sha256": bytes_sha256(np.ascontiguousarray(decoded).tobytes()), "pixel_identity_verified": True, "timeline": timeline})
    return evidence


def mux_n(video: Path, audio: Path, output: Path, expected_pcm: bytes) -> dict[str, Any]:
    if output.exists():
        raise RescoreError(f"refusing to overwrite target media: {output}")
    evidence = parent_media.mux_audio(video, audio, output, expected_pcm)
    if evidence.get("audio_pcm_sha256") != bytes_sha256(expected_pcm):
        raise RescoreError(f"mux PCM hash mismatch: {output}")
    if evidence.get("video_stream_copy") is not True or evidence.get("audio_modified") is not False:
        raise RescoreError(f"mux policy changed: {output}")
    return evidence
