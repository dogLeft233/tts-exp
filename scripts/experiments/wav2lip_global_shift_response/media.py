from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_roi_retiming_oracle import media as parent_media

from . import config
from .common import ExperimentError, bytes_sha256, file_sha256


def encode_face(frames: np.ndarray, output: Path) -> dict[str, Any]:
    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[1:] != (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3) or values.dtype != np.uint8:
        raise ExperimentError(f"face frames do not match BGR24 contract: {values.shape} {values.dtype}")
    return parent_media.encode_video(values, output)


def decode_frames(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    return parent_media.extract_bgr24_frames(path)


def mux_audio(video: Path, audio: Path, output: Path, expected_pcm: bytes) -> dict[str, Any]:
    if output.exists():
        raise ExperimentError(f"refusing to overwrite media cell: {output}")
    evidence = parent_media.mux_audio(video, audio, output, expected_pcm)
    if evidence.get("audio_pcm_sha256") != bytes_sha256(expected_pcm):
        raise ExperimentError(f"muxed PCM hash mismatch: {output}")
    if evidence.get("video_stream_copy") is not True or evidence.get("audio_modified") is not False:
        raise ExperimentError(f"mux policy changed: {output}")
    return evidence


def video_binding(path: Path, expected_frames: np.ndarray) -> dict[str, Any]:
    decoded, evidence = decode_frames(path)
    if not np.array_equal(decoded, expected_frames):
        raise ExperimentError(f"decoded video pixels differ from expected source: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "frame_count": int(decoded.shape[0]),
        "pixel_sha256": bytes_sha256(np.ascontiguousarray(decoded).tobytes()),
        "probe": evidence.get("probe"),
        "pixel_identity_verified": True,
    }
