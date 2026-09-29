from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_roi_retiming_oracle.common import (
    OracleError,
    bytes_sha256,
    file_sha256,
)
from scripts.experiments.wav2lip_roi_retiming_oracle.media import (
    encode_video as _encode_video,
)
from scripts.experiments.wav2lip_roi_retiming_oracle.media import forward_map
from scripts.experiments.wav2lip_roi_retiming_oracle.media import (
    mux_audio as _mux_audio,
)

from . import config


def build_linear_indices(sample_count: int, frame_count: int) -> dict[str, Any]:
    if sample_count < 2 or frame_count < 1:
        raise OracleError("sample_count and frame_count must be positive")
    mapped = forward_map(sample_count)
    t = config.SAMPLES_PER_FRAME * np.arange(frame_count, dtype=np.float64)
    if float(t[-1]) > float(sample_count - 1):
        raise OracleError("frame timestamp is outside the audio sample range")
    u = np.interp(t, np.arange(sample_count, dtype=np.float64), mapped) / config.SAMPLES_PER_FRAME
    if not np.isfinite(u).all() or np.any(np.diff(u) < 0.0):
        raise OracleError("linear frame coordinates are not finite and monotone")
    j = np.floor(u).astype(np.int64)
    k = np.ceil(u).astype(np.int64)
    if np.any(j < 0) or np.any(k >= frame_count):
        raise OracleError("linear interpolation frame index is out of bounds")
    w = u - j.astype(np.float64)
    return {
        "u": [float(value) for value in u],
        "j": [int(value) for value in j],
        "k": [int(value) for value in k],
        "w": [float(value) for value in w],
        "monotone": True,
        "in_bounds": True,
        "mapping_sha256": bytes_sha256(np.asarray(mapped, dtype="<f8").tobytes()),
        "mapping_length": int(mapped.size),
        "sample_count": int(sample_count),
        "frame_count": int(frame_count),
    }


def interpolate_frames(source: np.ndarray, indices: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(source)
    if values.ndim != 4 or values.shape[-1] != 3 or values.dtype != np.uint8:
        raise OracleError(f"source frames must be uint8 BGR: {values.shape} {values.dtype}")
    j = np.asarray(indices["j"], dtype=np.int64)
    k = np.asarray(indices["k"], dtype=np.int64)
    w = np.asarray(indices["w"], dtype=np.float64)
    if j.shape != k.shape or j.shape != w.shape or j.size != values.shape[0]:
        raise OracleError("linear interpolation index shape does not match source frames")
    if np.any(j < 0) or np.any(k >= values.shape[0]) or np.any(k < j):
        raise OracleError("linear interpolation index is invalid")
    output_float = (
        (1.0 - w[:, None, None, None]) * values[j].astype(np.float64)
        + w[:, None, None, None] * values[k].astype(np.float64)
    )
    output = np.floor(output_float + 0.5).astype(np.uint8)
    if not np.isfinite(output_float).all():
        raise OracleError("linear interpolation produced non-finite pixels")
    return output, {
        "source_frame_sha256": [bytes_sha256(frame.tobytes()) for frame in values],
        "output_frame_sha256": [bytes_sha256(frame.tobytes()) for frame in output],
        "pixel_dtype": "uint8",
        "pixel_arithmetic": "float64; floor(value+0.5) per channel",
        "frame_count": int(output.shape[0]),
    }


def verify_video_hash(path: Path, expected: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise OracleError(f"video hash mismatch: {path}")


def encode_video(frames: np.ndarray, output: Path) -> dict[str, Any]:
    return _encode_video(frames, output)


def mux_audio(video: Path, audio_wav: Path, output: Path, expected_pcm: bytes) -> dict[str, Any]:
    return _mux_audio(video, audio_wav, output, expected_pcm)
