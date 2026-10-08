"""Strict 16 kHz mono float32 WAV boundary for enhancement experiments.

Only uncompressed RIFF/WAVE PCM and IEEE-float inputs are accepted.  The
resampler is a deterministic NumPy windowed-sinc implementation so the same
input does not depend on an optional audio package's version.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

CANONICAL_SAMPLE_RATE = 16_000
FLOAT32_WAV_SUBTYPE = "IEEE_FLOAT_32_LE"


class AudioFormatError(ValueError):
    """Raised when a WAV cannot be represented by the canonical contract."""


@dataclass(frozen=True)
class AudioReadInfo:
    path: str
    file_sha256: str
    source_sample_rate: int
    sample_rate: int
    source_channels: int
    source_frames: int
    frames: int
    source_encoding: str
    resampler: str
    pcm_float32_sha256: str


@dataclass(frozen=True)
class AudioWriteInfo:
    path: str
    file_sha256: str
    sample_rate: int
    channels: int
    frames: int
    subtype: str
    peak_before_limit: float
    peak_after_limit: float
    applied_peak_gain: float
    pcm_float32_sha256: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pcm_float32_sha256(samples: np.ndarray) -> str:
    """Hash canonical little-endian float32 PCM samples, without WAV headers."""

    arr = np.asarray(samples, dtype=np.dtype("<f4"))
    if arr.ndim != 1:
        raise AudioFormatError("canonical PCM hash expects a mono 1-D array")
    return _sha256_bytes(np.ascontiguousarray(arr).tobytes())


def _parse_wave(data: bytes) -> tuple[int, int, int, int, int, bytes]:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioFormatError("expected a little-endian RIFF/WAVE file")
    declared_size = struct.unpack_from("<I", data, 4)[0] + 8
    if declared_size > len(data):
        raise AudioFormatError("WAV RIFF size exceeds bytes on disk")

    fmt: tuple[int, int, int, int, int] | None = None
    data_chunks: list[bytes] = []
    pos = 12
    limit = min(len(data), declared_size)
    while pos + 8 <= limit:
        chunk_id = data[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", data, pos + 4)[0]
        start = pos + 8
        end = start + chunk_size
        if end > limit:
            raise AudioFormatError(f"truncated WAV chunk {chunk_id!r}")
        chunk = data[start:end]
        if chunk_id == b"fmt ":
            if len(chunk) < 16:
                raise AudioFormatError("WAV fmt chunk is shorter than WAVEFORMAT")
            format_code, channels, sample_rate, _, block_align, bits = struct.unpack_from(
                "<HHIIHH", chunk, 0
            )
            if format_code == 0xFFFE:
                if len(chunk) < 40:
                    raise AudioFormatError("truncated WAVE_FORMAT_EXTENSIBLE fmt chunk")
                # The first WORD of the SubFormat GUID is the underlying
                # WAVEFORMAT tag (PCM=1, IEEE_FLOAT=3).
                format_code = struct.unpack_from("<H", chunk, 24)[0]
            fmt = (format_code, channels, sample_rate, bits, block_align)
        elif chunk_id == b"data":
            data_chunks.append(chunk)
        pos = end + (chunk_size & 1)

    if fmt is None or not data_chunks:
        raise AudioFormatError("WAV must contain fmt and data chunks")
    format_code, channels, sample_rate, bits, block_align = fmt
    if channels < 1 or sample_rate < 1 or block_align < 1:
        raise AudioFormatError("invalid WAV channel/rate/block alignment")
    if bits not in (8, 16, 24, 32, 64):
        raise AudioFormatError(f"unsupported WAV sample width: {bits} bits")
    if format_code == 1 and bits == 64:
        raise AudioFormatError("64-bit integer PCM WAV is not supported")
    if format_code == 3 and bits not in (32, 64):
        raise AudioFormatError("IEEE-float WAV input must be 32- or 64-bit")
    if format_code not in (1, 3):
        raise AudioFormatError(f"unsupported WAV format code: {format_code}")
    expected_align = channels * ((bits + 7) // 8)
    if block_align != expected_align:
        raise AudioFormatError("WAV block alignment does not match channel/sample width")
    payload = b"".join(data_chunks)
    if len(payload) % block_align:
        raise AudioFormatError("WAV data size is not a whole number of frames")
    return format_code, channels, sample_rate, bits, block_align, payload


def _decode_wave_samples(
    format_code: int, channels: int, bits: int, block_align: int, payload: bytes
) -> np.ndarray:
    frame_count = len(payload) // block_align
    if format_code == 1 and bits == 8:
        ints = np.frombuffer(payload, dtype=np.uint8).reshape(frame_count, channels)
        decoded = (ints.astype(np.float64) - 128.0) / 128.0
    elif format_code == 1 and bits == 16:
        ints = np.frombuffer(payload, dtype="<i2").reshape(frame_count, channels)
        decoded = ints.astype(np.float64) / 32768.0
    elif format_code == 1 and bits == 24:
        raw = np.frombuffer(payload, dtype=np.uint8).reshape(frame_count, channels, 3)
        packed = (
            raw[:, :, 0].astype(np.int32)
            | (raw[:, :, 1].astype(np.int32) << 8)
            | (raw[:, :, 2].astype(np.int32) << 16)
        )
        signed = (packed ^ 0x800000) - 0x800000
        decoded = signed.astype(np.float64) / 8388608.0
    elif format_code == 1 and bits == 32:
        ints = np.frombuffer(payload, dtype="<i4").reshape(frame_count, channels)
        decoded = ints.astype(np.float64) / 2147483648.0
    elif format_code == 3 and bits == 32:
        decoded = np.frombuffer(payload, dtype="<f4").reshape(frame_count, channels).astype(
            np.float64
        )
    elif format_code == 3 and bits == 64:
        decoded = np.frombuffer(payload, dtype="<f8").reshape(frame_count, channels)
    else:  # guarded by _parse_wave; retain fail-closed behavior if called alone
        raise AudioFormatError(f"unsupported WAV encoding {format_code}/{bits}")
    if not np.isfinite(decoded).all():
        raise AudioFormatError("WAV contains NaN or infinite PCM values")
    return decoded


def _kaiser_resample_mono(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample with a deterministic truncated-sinc low-pass and edge renormalization."""

    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float64).copy()
    n_input = int(samples.shape[0])
    if n_input == 0:
        return np.empty(0, dtype=np.float64)
    n_output = (n_input * target_rate + source_rate // 2) // source_rate
    if n_output <= 0:
        raise AudioFormatError("resampling would produce an empty waveform")

    cutoff = min(1.0, target_rate / source_rate)
    radius = int(math.ceil(12.0 / cutoff))
    beta = 8.6
    i0_beta = np.i0(beta)
    output = np.empty(n_output, dtype=np.float64)
    source = np.asarray(samples, dtype=np.float64)
    block = 2048
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    for start in range(0, n_output, block):
        stop = min(n_output, start + block)
        positions = np.arange(start, stop, dtype=np.float64) * source_rate / target_rate
        centers = np.floor(positions).astype(np.int64)
        indices = centers[:, None] + offsets[None, :]
        delta = indices.astype(np.float64) - positions[:, None]
        relative = delta / radius
        inside = np.abs(relative) <= 1.0
        window = np.zeros_like(relative)
        window[inside] = np.i0(beta * np.sqrt(1.0 - relative[inside] ** 2)) / i0_beta
        weights = cutoff * np.sinc(cutoff * delta) * window
        in_bounds = (indices >= 0) & (indices < n_input)
        clipped_indices = np.clip(indices, 0, n_input - 1)
        weights *= in_bounds
        denom = weights.sum(axis=1)
        if np.any(np.abs(denom) < 1e-15):
            raise AudioFormatError("resampler produced a zero-weight output frame")
        output[start:stop] = (source[clipped_indices] * weights).sum(axis=1) / denom
    return output


def canonicalize_samples(
    samples: np.ndarray,
    sample_rate: int,
    *,
    target_rate: int = CANONICAL_SAMPLE_RATE,
) -> np.ndarray:
    """Convert sample-major PCM to mono float32 at ``target_rate``.

    Stereo is downmixed by the arithmetic channel mean before deterministic
    resampling. Input values are never clipped or peak-normalized here.
    """

    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise AudioFormatError("sample_rate must be a positive integer")
    if not isinstance(target_rate, int) or target_rate <= 0:
        raise AudioFormatError("target_rate must be a positive integer")
    arr = np.asarray(samples)
    if arr.ndim == 1:
        mono = arr.astype(np.float64, copy=False)
    elif arr.ndim == 2 and arr.shape[1] >= 1:
        mono = arr.astype(np.float64, copy=False).mean(axis=1, dtype=np.float64)
    else:
        raise AudioFormatError("samples must have shape [frames] or [frames, channels]")
    if mono.size == 0:
        raise AudioFormatError("empty audio is not valid input")
    if not np.isfinite(mono).all():
        raise AudioFormatError("audio contains NaN or infinite values")
    converted = _kaiser_resample_mono(mono, sample_rate, target_rate)
    result = np.ascontiguousarray(converted, dtype=np.float32)
    if not np.isfinite(result).all():
        raise AudioFormatError("canonicalization produced NaN or infinite values")
    return result


def read_canonical_wav(
    path: str | os.PathLike[str], *, target_rate: int = CANONICAL_SAMPLE_RATE
) -> tuple[np.ndarray, AudioReadInfo]:
    """Read PCM/float RIFF WAV and return mono float32 samples plus source seals."""

    source_path = Path(path)
    raw = source_path.read_bytes()
    fmt_code, channels, sample_rate, bits, block_align, payload = _parse_wave(raw)
    decoded = _decode_wave_samples(fmt_code, channels, bits, block_align, payload)
    samples = canonicalize_samples(decoded, sample_rate, target_rate=target_rate)
    encoding = f"{'PCM' if fmt_code == 1 else 'FLOAT'}_{bits}"
    info = AudioReadInfo(
        path=str(source_path.resolve()),
        file_sha256=_sha256_bytes(raw),
        source_sample_rate=sample_rate,
        sample_rate=target_rate,
        source_channels=channels,
        source_frames=decoded.shape[0],
        frames=int(samples.size),
        source_encoding=encoding,
        resampler=("none" if sample_rate == target_rate else "numpy_kaiser_sinc_beta8.6_radius12_over_cutoff"),
        pcm_float32_sha256=pcm_float32_sha256(samples),
    )
    return samples, info


def _as_float32_mono(samples: np.ndarray | Iterable[float]) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        raise AudioFormatError("writer expects a non-empty mono 1-D waveform")
    if not np.isfinite(arr).all():
        raise AudioFormatError("refusing to write non-finite PCM")
    return np.ascontiguousarray(arr)


def write_float32_wav(
    path: str | os.PathLike[str],
    samples: np.ndarray | Iterable[float],
    *,
    sample_rate: int = CANONICAL_SAMPLE_RATE,
    peak_ceiling: float | None = None,
    overwrite: bool = False,
) -> AudioWriteInfo:
    """Atomically write mono IEEE-float32 WAV and verify exact PCM readback.

    If ``peak_ceiling`` is supplied, over-peak audio is uniformly attenuated;
    it is never hard-clipped. Set it to ``None`` to retain exact input values.
    """

    if sample_rate <= 0:
        raise AudioFormatError("sample_rate must be positive")
    if peak_ceiling is not None and (not math.isfinite(peak_ceiling) or peak_ceiling <= 0):
        raise AudioFormatError("peak_ceiling must be positive and finite or None")
    arr = _as_float32_mono(samples)
    peak_before = float(np.max(np.abs(arr.astype(np.float64))))
    gain = 1.0
    if peak_ceiling is not None and peak_before > peak_ceiling:
        gain = peak_ceiling / peak_before
        arr = np.ascontiguousarray(arr * np.float32(gain), dtype=np.float32)
    peak_after = float(np.max(np.abs(arr.astype(np.float64))))

    data_bytes = arr.astype(np.dtype("<f4"), copy=False).tobytes()
    if len(data_bytes) > 0xFFFFFFFF - 36:
        raise AudioFormatError("audio is too large for a standard RIFF/WAVE file")
    fmt = struct.pack(
        "<HHIIHH",
        3,  # WAVE_FORMAT_IEEE_FLOAT
        1,
        int(sample_rate),
        int(sample_rate) * 4,
        4,
        32,
    )
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data_bytes)) + data_bytes
    riff = b"RIFF" + struct.pack("<I", len(body)) + body

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(riff)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    persisted = destination.read_bytes()
    parsed = _parse_wave(persisted)
    code, channels, read_rate, bits, align, payload = parsed
    reread = _decode_wave_samples(code, channels, bits, align, payload)
    reread = np.ascontiguousarray(reread[:, 0], dtype=np.float32)
    if (code, channels, read_rate, bits) != (3, 1, int(sample_rate), 32):
        raise AudioFormatError("float WAV readback header differs from requested format")
    if reread.shape != arr.shape or not np.array_equal(reread, arr):
        raise AudioFormatError("float WAV readback PCM is not bit-identical")
    return AudioWriteInfo(
        path=str(destination.resolve()),
        file_sha256=_sha256_bytes(persisted),
        sample_rate=int(sample_rate),
        channels=1,
        frames=int(arr.size),
        subtype=FLOAT32_WAV_SUBTYPE,
        peak_before_limit=peak_before,
        peak_after_limit=peak_after,
        applied_peak_gain=float(gain),
        pcm_float32_sha256=pcm_float32_sha256(arr),
    )
