from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_roi_retiming_oracle import media as parent_media

from . import config
from .common import ExperimentError, bytes_sha256, file_sha256, read_pcm16_wav


def mux_audio(video: Path, audio_wav: Path, output: Path, expected_pcm: bytes) -> dict[str, Any]:
    if output.exists():
        raise ExperimentError(f"refusing to overwrite media artifact: {output}")
    evidence = parent_media.mux_audio(video, audio_wav, output, expected_pcm)
    if evidence.get("audio_pcm_sha256") != bytes_sha256(expected_pcm):
        raise ExperimentError(f"mux PCM binding failed: {output}")
    return evidence


def decode_frames(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    return parent_media.extract_bgr24_frames(path)


def video_identity(path: Path, expected_frame_count: int) -> dict[str, Any]:
    frames, evidence = decode_frames(path)
    if frames.shape != (expected_frame_count, config.FRAME_HEIGHT, config.FRAME_WIDTH, 3):
        raise ExperimentError(f"video frame contract failed: {path}: {frames.shape}")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "frame_count": int(frames.shape[0]),
        "pixel_sha256": bytes_sha256(np.ascontiguousarray(frames).tobytes()),
        "probe": evidence.get("probe"),
    }


def verify_audio(path: Path, expected_pcm: bytes, expected_container_sha256: str | None = None) -> dict[str, Any]:
    if expected_container_sha256:
        actual = file_sha256(path)
        if actual != expected_container_sha256:
            raise ExperimentError(f"audio container hash changed: {path}")
    pcm, values, params = read_pcm16_wav(path)
    if pcm != expected_pcm:
        raise ExperimentError(f"audio decoded PCM differs: {path}")
    return {
        "path": str(path.resolve()),
        "container_sha256": file_sha256(path),
        "decoded_pcm_sha256": bytes_sha256(pcm),
        "sample_count": int(values.size),
        "sample_rate": int(params["sample_rate"]),
        "channels": int(params["channels"]),
        "sample_width": int(params["sample_width"]),
    }
