"""Exact integer-delay PCM construction and source-index provenance."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    DELAY_SAMPLES,
    SAMPLE_RATE,
    TimingError,
    bytes_sha256,
    file_sha256,
)


def _pcm_values(natural: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    if isinstance(natural, (bytes, bytearray, memoryview)):
        raw = bytes(natural)
        if not raw or len(raw) % 2:
            raise TimingError("natural PCM must be non-empty and int16 aligned")
        return np.frombuffer(raw, dtype="<i2").copy()
    values = np.asarray(natural)
    if values.ndim != 1 or values.size == 0:
        raise TimingError("natural PCM must be a non-empty one-dimensional array")
    if values.dtype.kind not in "iu" or values.dtype.itemsize != 2:
        raise TimingError(f"natural PCM must be int16, got {values.dtype}")
    # The copy also normalizes a native big-endian input to little-endian PCM.
    return np.asarray(values, dtype="<i2").copy()


def build_delay_pcm(
    natural: bytes | bytearray | memoryview | np.ndarray,
    *,
    shift_samples: int = DELAY_SAMPLES,
) -> tuple[bytes, np.ndarray]:
    """Return ``N[n-shift]`` with zero outside support and its exact map.

    The map is indexed by output sample.  ``-1`` denotes a zero-filled sample;
    all other values are the source index in the untouched natural PCM.  It is
    intentionally an integer map so the validator can reconstruct the byte
    sequence without relying on floating point audio code.
    """

    if isinstance(shift_samples, bool) or int(shift_samples) != shift_samples or shift_samples < 1:
        raise TimingError("shift_samples must be a positive integer")
    shift = int(shift_samples)
    values = _pcm_values(natural)
    source_index = np.arange(values.size, dtype="<i8") - np.int64(shift)
    valid = source_index >= 0
    source_index[~valid] = -1
    result = np.zeros(values.size, dtype="<i2")
    if np.any(valid):
        result[valid] = values[source_index[valid]]
    return result.tobytes(), source_index


def construct_delay_pcm(
    natural: bytes | bytearray | memoryview | np.ndarray,
    *,
    shift_samples: int = DELAY_SAMPLES,
) -> tuple[bytes, np.ndarray]:
    """Alias used by callers that describe the operation as construction."""

    return build_delay_pcm(natural, shift_samples=shift_samples)


def delay_pcm(
    natural: bytes | bytearray | memoryview | np.ndarray,
    *,
    shift_samples: int = DELAY_SAMPLES,
) -> bytes:
    """Return only the delayed PCM bytes for mux/scorer adapters."""

    return build_delay_pcm(natural, shift_samples=shift_samples)[0]


def exact_source_index_map(length: int, *, shift_samples: int = DELAY_SAMPLES) -> np.ndarray:
    """Descriptive alias for the persisted output-to-source map."""

    return source_index_map(length, shift_samples=shift_samples)


def source_index_map(length: int, *, shift_samples: int = DELAY_SAMPLES) -> np.ndarray:
    if isinstance(length, bool) or int(length) != length or int(length) <= 0:
        raise TimingError("length must be a positive integer")
    _, mapping = build_delay_pcm(np.zeros(int(length), dtype="<i2"), shift_samples=shift_samples)
    return mapping


def read_pcm16_wav(path: str | Path) -> tuple[bytes, np.ndarray, dict[str, Any]]:
    target = Path(path).resolve()
    try:
        with wave.open(str(target), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            count = handle.getnframes()
            raw = handle.readframes(count)
    except (OSError, wave.Error) as exc:
        raise TimingError(f"cannot read PCM16 WAV: {target}") from exc
    if channels != 1 or sample_width != 2 or sample_rate != SAMPLE_RATE:
        raise TimingError(
            f"WAV must be mono/{SAMPLE_RATE}Hz/PCM16: {target} "
            f"(channels={channels}, width={sample_width}, rate={sample_rate})"
        )
    if len(raw) != int(count) * 2:
        raise TimingError(f"WAV frame count does not match PCM bytes: {target}")
    values = np.frombuffer(raw, dtype="<i2").copy()
    return raw, values, {
        "sample_count": int(count),
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "sample_width": int(sample_width),
        "pcm_sha256": bytes_sha256(raw),
        "container_sha256": file_sha256(target),
    }


def write_pcm16_wav(path: str | Path, pcm: bytes | np.ndarray) -> dict[str, Any]:
    target = Path(path).resolve()
    values = _pcm_values(pcm)
    raw = values.tobytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        existing, _values, _meta = read_pcm16_wav(target)
        if existing != raw:
            raise TimingError(f"existing delay WAV differs; refusing overwrite: {target}")
    else:
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            with wave.open(str(temporary), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(SAMPLE_RATE)
                handle.writeframes(raw)
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
    decoded, values_read, metadata = read_pcm16_wav(target)
    if decoded != raw or not np.array_equal(values_read, values):
        raise TimingError(f"delay WAV readback differs: {target}")
    return metadata


def write_source_index_map(
    path: str | Path,
    mapping: np.ndarray,
    *,
    shift_samples: int = DELAY_SAMPLES,
) -> dict[str, Any]:
    target = Path(path).resolve()
    values = np.asarray(mapping)
    if values.ndim != 1 or values.dtype.kind not in "iu" or values.size == 0:
        raise TimingError("source-index map must be a non-empty integer vector")
    values = np.asarray(values, dtype="<i8")
    if np.any((values < -1) | (values >= values.size)):
        raise TimingError("source-index map contains an out-of-range source index")
    verify_source_index_map(values, length=values.size, shift_samples=shift_samples)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        with target.open("rb") as handle:
            existing = np.load(handle, allow_pickle=False)
        if not np.array_equal(np.asarray(existing, dtype="<i8"), values):
            raise TimingError(f"existing source-index map differs: {target}")
    else:
        temporary = target.with_name(f".{target.name}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        temporary.replace(target)
    with target.open("rb") as handle:
        loaded = np.asarray(np.load(handle, allow_pickle=False), dtype="<i8")
    if not np.array_equal(loaded, values):
        raise TimingError(f"source-index map readback differs: {target}")
    return {
        "path": str(target),
        "sha256": file_sha256(target),
        "dtype": "<i8",
        "length": int(values.size),
        "zero_count": int(np.sum(values < 0)),
        "bytes_sha256": bytes_sha256(values.tobytes()),
    }


def verify_source_index_map(
    mapping: np.ndarray,
    *,
    length: int,
    shift_samples: int = DELAY_SAMPLES,
) -> None:
    values = np.asarray(mapping)
    expected = source_index_map(length, shift_samples=shift_samples)
    if values.shape != expected.shape or not np.array_equal(np.asarray(values, dtype="<i8"), expected):
        raise TimingError("source-index map does not match the exact delay contract")


def build_delay_artifacts(
    natural_wav: str | Path,
    output_wav: str | Path,
    output_map: str | Path,
    *,
    shift_samples: int = DELAY_SAMPLES,
) -> dict[str, Any]:
    """Construct, write, and read back one frozen DELAY pair."""

    _natural_raw, natural_values, natural_meta = read_pcm16_wav(natural_wav)
    delayed_raw, mapping = build_delay_pcm(natural_values, shift_samples=shift_samples)
    delayed_meta = write_pcm16_wav(output_wav, delayed_raw)
    mapping_binding = write_source_index_map(output_map, mapping, shift_samples=shift_samples)
    verify_source_index_map(mapping, length=natural_values.size, shift_samples=shift_samples)
    rebuilt = np.zeros(natural_values.size, dtype="<i2")
    valid = mapping >= 0
    rebuilt[valid] = natural_values[mapping[valid]]
    if rebuilt.tobytes() != delayed_raw:
        raise TimingError("delay PCM does not follow its source-index map")
    return {
        "natural": {
            "path": str(Path(natural_wav).resolve()),
            "sha256": file_sha256(natural_wav),
            "pcm_sha256": natural_meta["pcm_sha256"],
            "sample_count": int(natural_values.size),
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "sample_width": 2,
        },
        "delay": {
            "path": str(Path(output_wav).resolve()),
            "sha256": delayed_meta["container_sha256"],
            "pcm_sha256": delayed_meta["pcm_sha256"],
            "sample_count": int(natural_values.size),
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "sample_width": 2,
        },
        "source_index_map": mapping_binding,
        "delay_samples": int(shift_samples),
        "delay_frames": int(shift_samples // (SAMPLE_RATE // 25)) if shift_samples % (SAMPLE_RATE // 25) == 0 else None,
        "length_preserved": True,
        "formula": "DELAY[n]=N[n-3200] when n>=3200, else zero",
    }
