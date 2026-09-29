from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import DiagnosticError, bytes_sha256, file_sha256


def read_pcm16(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            count = handle.getnframes()
            payload = handle.readframes(count)
    except (OSError, wave.Error) as exc:
        raise DiagnosticError(f"cannot read PCM16 WAV: {path}") from exc
    if channels != 1 or width != 2 or rate != config.SAMPLE_RATE:
        raise DiagnosticError(f"audio format is not mono PCM16/16k: {path}")
    values = np.frombuffer(payload, dtype="<i2").copy()
    if values.size != count or values.size == 0:
        raise DiagnosticError(f"audio sample count is invalid: {path}")
    return values, {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "pcm_sha256": bytes_sha256(payload),
        "sample_count": int(values.size),
        "sample_rate": rate,
        "channels": channels,
        "sample_width": width,
    }


def write_pcm16(path: Path, values: np.ndarray) -> dict[str, Any]:
    pcm = np.asarray(values)
    if pcm.dtype != np.int16 or pcm.ndim != 1 or pcm.size == 0:
        raise ValueError("PCM output must be a non-empty one-dimensional int16 array")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(config.SAMPLE_RATE)
        handle.writeframes(np.asarray(pcm, dtype="<i2").tobytes())
    temporary.replace(path)
    _, metadata = read_pcm16(path)
    return metadata


def local_swap(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    pcm = np.asarray(values)
    if pcm.dtype != np.int16 or pcm.ndim != 1:
        raise ValueError("LOCAL_SWAP input must be one-dimensional int16")
    length = int(pcm.size)
    b1, b2, b3 = length // 4, length // 2, (3 * length) // 4
    result = np.concatenate((pcm[:b1], pcm[b2:b3], pcm[b1:b2], pcm[b3:])).astype("<i2")
    if result.size != length:
        raise DiagnosticError("LOCAL_SWAP changed sample count")
    return result, {"length": length, "b1": b1, "b2": b2, "b3": b3}


def quarter_swap(values: np.ndarray, k: int) -> tuple[np.ndarray, dict[str, list[int]]]:
    value = np.asarray(values)
    if value.ndim < 1 or value.shape[0] != 4 * k:
        raise ValueError("quarter_swap requires exactly four equal chunks")
    chunks = [value[index * k : (index + 1) * k] for index in range(4)]
    order = [0, 2, 1, 3]
    result = np.concatenate([chunks[index] for index in order], axis=0)
    mapping = {"output_to_input": [index for index in range(4 * k) if False]}
    mapping["output_to_input"] = [chunk * k + offset for chunk in order for offset in range(k)]
    mapping["chunk_order"] = order
    return result, mapping


def slice_pcm(values: np.ndarray, start_frame: int, end_frame: int) -> np.ndarray:
    return np.asarray(values)[start_frame * config.SAMPLES_PER_FRAME : end_frame * config.SAMPLES_PER_FRAME].copy()
