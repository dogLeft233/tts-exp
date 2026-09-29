from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_roi_retiming_oracle import media as parent_media

from . import config
from .common import PlateauError, bytes_sha256, read_pcm16_wav, write_pcm16_wav


def make_plateau_pcm(pcm: np.ndarray, source_indices: np.ndarray) -> tuple[bytes, dict[str, Any]]:
    source = np.asarray(pcm, dtype=np.int16)
    indices = np.asarray(source_indices, dtype=np.int64)
    if source.ndim != 1 or indices.shape != source.shape or int(indices.min()) < 0 or int(indices.max()) >= source.size:
        raise PlateauError("plateau PCM source indices are invalid")
    output = np.ascontiguousarray(source[indices], dtype=np.int16)
    return output.astype("<i2", copy=False).tobytes(), {
        "sample_count": int(output.size),
        "source_index_sha256": bytes_sha256(np.asarray(indices, dtype="<i8").tobytes()),
        "source_index_length": int(indices.size),
        "source_index_rule": "+3200 before break, -3200 at/after break",
        "pcm_sha256": bytes_sha256(output.astype("<i2", copy=False).tobytes()),
        "sample_shift": config.AUDIO_SHIFT,
    }


def copy_or_write_audio(
    natural_path: Path,
    natural_pcm: bytes,
    plateau_pcm: bytes,
    output_dir: Path,
    sample_id: str,
) -> dict[str, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_path = output_dir / f"{sample_id}__N.wav"
    p_path = output_dir / f"{sample_id}__P.wav"
    if n_path.exists() or p_path.exists():
        raise PlateauError(f"audio artifacts already exist: {sample_id}")
    n_path.write_bytes(natural_path.read_bytes())
    observed, values, params = read_pcm16_wav(n_path)
    if observed != natural_pcm:
        raise PlateauError(f"N audio copy changed: {sample_id}")
    p_meta = write_pcm16_wav(p_path, plateau_pcm, config.SAMPLE_RATE)
    return {
        config.AUDIO_N: {
            "path": str(n_path.resolve()),
            "container_sha256": __import__("hashlib").sha256(n_path.read_bytes()).hexdigest(),
            "decoded_pcm_sha256": bytes_sha256(observed),
            "sample_count": int(values.size),
            "sample_rate": int(params["sample_rate"]),
            "construction": "byte-identical copy of parent N",
        },
        config.AUDIO_P: {
            **p_meta,
            "construction": "direct int16 source indexing; no resampling/interpolation/normalization",
        },
    }


def encode_video(frames: np.ndarray, output: Path) -> dict[str, Any]:
    return parent_media.encode_video(np.asarray(frames, dtype=np.uint8), output)


def mux_audio(video: Path, audio_wav: Path, output: Path, expected_pcm: bytes) -> dict[str, Any]:
    return parent_media.mux_audio(video, audio_wav, output, expected_pcm)


def decode_frames(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    return parent_media.extract_bgr24_frames(path)
