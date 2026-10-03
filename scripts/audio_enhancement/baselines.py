"""Independent RMS and phase-preserving low-dimensional profile baselines."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from .audio_io import CANONICAL_SAMPLE_RATE

FIXED_POSITIVE_CONTROL_RMS = 0.0376838172675746
DEFAULT_N_FFT = 512
DEFAULT_HOP_LENGTH = 128


def waveform_rms(samples: np.ndarray) -> float:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("RMS expects a non-empty finite waveform")
    return float(np.sqrt(np.mean(arr * arr, dtype=np.float64)))


@dataclass(frozen=True)
class RMSGainInfo:
    input_rms: float
    target_rms: float
    requested_gain: float
    applied_gain: float
    output_rms: float
    peak_limited: bool


def apply_global_rms(
    samples: np.ndarray,
    target_rms: float = FIXED_POSITIVE_CONTROL_RMS,
    *,
    peak_ceiling: float | None = None,
    silence_epsilon: float = 1e-12,
) -> tuple[np.ndarray, RMSGainInfo]:
    """Apply the historic fixed RMS positive control without hard clipping."""

    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError("RMS baseline expects non-empty finite mono float32 audio")
    if not math.isfinite(target_rms) or target_rms <= 0:
        raise ValueError("target_rms must be positive and finite")
    if peak_ceiling is not None and (not math.isfinite(peak_ceiling) or peak_ceiling <= 0):
        raise ValueError("peak_ceiling must be positive and finite or None")
    input_rms = waveform_rms(arr)
    input_peak = float(np.max(np.abs(arr.astype(np.float64))))
    if input_rms <= silence_epsilon:
        output = arr.copy()
        gain = 1.0
        limited = False
    else:
        requested = target_rms / input_rms
        gain = requested
        limited = False
        if peak_ceiling is not None and input_peak > 0 and input_peak * gain > peak_ceiling:
            gain = peak_ceiling / input_peak
            limited = True
        output = np.ascontiguousarray(arr * np.float32(gain), dtype=np.float32)
    info = RMSGainInfo(
        input_rms=input_rms,
        target_rms=float(target_rms),
        requested_gain=(1.0 if input_rms <= silence_epsilon else target_rms / input_rms),
        applied_gain=float(gain),
        output_rms=waveform_rms(output),
        peak_limited=limited,
    )
    return output, info


@dataclass(frozen=True)
class FlatGainInfo:
    input_rms: float
    gain_db: float
    applied_gain: float
    output_rms: float
    peak_limited: bool


def apply_flat_gain(
    samples: np.ndarray,
    gain_db: float,
    *,
    max_abs_gain_db: float = 6.0,
    peak_ceiling: float | None = None,
) -> tuple[np.ndarray, FlatGainInfo]:
    """Apply a declared, bounded utterance-wide gain without hard clipping."""

    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("flat gain expects non-empty finite mono float32 audio")
    if not math.isfinite(gain_db) or not math.isfinite(max_abs_gain_db) or max_abs_gain_db <= 0:
        raise ValueError("gain_db and positive max_abs_gain_db must be finite")
    if abs(gain_db) > max_abs_gain_db:
        raise ValueError("flat gain is outside the configured dB bound")
    if peak_ceiling is not None and (not math.isfinite(peak_ceiling) or peak_ceiling <= 0):
        raise ValueError("peak_ceiling must be positive and finite or None")
    input_rms = waveform_rms(x)
    peak = float(np.max(np.abs(x.astype(np.float64))))
    requested = 10.0 ** (float(gain_db) / 20.0)
    applied = requested
    limited = False
    if peak_ceiling is not None and peak > 0 and peak * applied > peak_ceiling:
        applied = peak_ceiling / peak
        limited = True
    output = x.copy() if input_rms <= 1e-12 else np.ascontiguousarray(x * np.float32(applied))
    if not np.isfinite(output).all():
        raise FloatingPointError("flat gain returned non-finite audio")
    return output, FlatGainInfo(
        input_rms=input_rms,
        gain_db=float(gain_db),
        applied_gain=float(1.0 if input_rms <= 1e-12 else applied),
        output_rms=waveform_rms(output),
        peak_limited=bool(limited and input_rms > 1e-12),
    )


@dataclass(frozen=True)
class LocalCompressionInfo:
    input_rms: float
    output_rms: float
    envelope_window_samples: int
    ratio: float
    minimum_gain_db: float
    maximum_gain_db: float
    silence_unchanged: bool


def apply_local_envelope_compression(
    samples: np.ndarray,
    *,
    sample_rate: int = CANONICAL_SAMPLE_RATE,
    window_ms: float = 32.0,
    ratio: float = 0.8,
    min_gain_db: float = -6.0,
    max_gain_db: float = 6.0,
    envelope_floor: float = 1e-5,
) -> tuple[np.ndarray, LocalCompressionInfo]:
    """Compress centered local RMS with a fixed short window and no retiming."""

    x = np.asarray(samples, dtype=np.float32)
    if sample_rate != CANONICAL_SAMPLE_RATE:
        raise ValueError("local envelope is defined at the canonical 16 kHz rate")
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("local compressor expects non-empty finite mono float32 audio")
    if not 20.0 <= window_ms <= 50.0:
        raise ValueError("window_ms must be within the frozen 20--50 ms range")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    if not (math.isfinite(min_gain_db) and math.isfinite(max_gain_db)):
        raise ValueError("gain bounds must be finite")
    if min_gain_db > 0 or max_gain_db < 0 or min_gain_db > max_gain_db:
        raise ValueError("gain bounds must include 0 dB and be ordered")
    if not math.isfinite(envelope_floor) or envelope_floor <= 0:
        raise ValueError("envelope_floor must be positive and finite")

    n_window = max(1, int(round(sample_rate * window_ms / 1000.0)))
    n_window = min(n_window, max(1, (x.size // 2) * 2 + 1))
    left = n_window // 2 if n_window % 2 else n_window // 2 - 1
    right = n_window - 1 - left
    power = x.astype(np.float64) ** 2
    padded = np.pad(power, (left, right), mode="edge")
    cumulative = np.concatenate(([0.0], np.cumsum(padded, dtype=np.float64)))
    local_power = (cumulative[n_window:] - cumulative[:-n_window]) / n_window
    if local_power.size != x.size:
        raise RuntimeError("local RMS smoother changed sample count")
    envelope = np.sqrt(np.maximum(local_power, envelope_floor**2))
    input_rms = waveform_rms(x)
    if input_rms <= 1e-12:
        output = x.copy()
        low_gain_db = high_gain_db = 0.0
        silence_unchanged = True
    else:
        raw_gain_db = 20.0 * (ratio - 1.0) * np.log10(envelope / max(input_rms, envelope_floor))
        bounded_gain_db = np.clip(raw_gain_db, min_gain_db, max_gain_db)
        output = np.ascontiguousarray(
            x.astype(np.float64) * np.power(10.0, bounded_gain_db / 20.0), dtype=np.float32
        )
        low_gain_db = float(np.min(bounded_gain_db))
        high_gain_db = float(np.max(bounded_gain_db))
        silence_unchanged = False
    if output.shape != x.shape or not np.isfinite(output).all():
        raise RuntimeError("local compressor violated exact-length/finite output contract")
    info = LocalCompressionInfo(
        input_rms=input_rms,
        output_rms=waveform_rms(output),
        envelope_window_samples=n_window,
        ratio=float(ratio),
        minimum_gain_db=low_gain_db,
        maximum_gain_db=high_gain_db,
        silence_unchanged=silence_unchanged,
    )
    return output, info


@dataclass(frozen=True)
class SpectralProfile:
    """Bounded few-parameter magnitude profile for deterministic black-box search.

    ``band_gains_db`` controls low/mid/high frequency anchors and is linearly
    interpolated in log frequency. ``envelope_gains_db`` is a coarse temporal
    envelope interpolated over normalized clip time. Complex STFT phase is
    unchanged; the profile does not time-warp the waveform.
    """

    global_gain_db: float = 0.0
    band_gains_db: tuple[float, float, float] = (0.0, 0.0, 0.0)
    envelope_gains_db: tuple[float, ...] = (0.0,) * 8
    max_abs_gain_db: float = 6.0
    max_total_gain_db: float = 6.0

    def __post_init__(self) -> None:
        values = (self.global_gain_db, *self.band_gains_db, *self.envelope_gains_db)
        if len(self.band_gains_db) != 3 or len(self.envelope_gains_db) < 2:
            raise ValueError("profile requires 3 band controls and at least 2 envelope knots")
        if not math.isfinite(self.max_abs_gain_db) or self.max_abs_gain_db <= 0:
            raise ValueError("max_abs_gain_db must be positive and finite")
        if not math.isfinite(self.max_total_gain_db) or self.max_total_gain_db <= 0:
            raise ValueError("max_total_gain_db must be positive and finite")
        if any(not math.isfinite(float(v)) for v in values):
            raise ValueError("profile values must be finite")
        if any(abs(float(v)) > self.max_abs_gain_db for v in values):
            raise ValueError("profile gain is outside the configured dB bound")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _stft(
    samples: np.ndarray, *, n_fft: int = DEFAULT_N_FFT, hop_length: int = DEFAULT_HOP_LENGTH
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("STFT expects a non-empty finite mono waveform")
    if n_fft < 32 or n_fft % 2 or hop_length < 1 or hop_length > n_fft:
        raise ValueError("invalid STFT dimensions")
    window = np.hanning(n_fft + 1)[:-1]
    half = n_fft // 2
    padded = np.pad(x, (half, half), mode="constant")
    frame_count = 1 + (padded.size - n_fft) // hop_length
    starts = np.arange(frame_count, dtype=np.int64) * hop_length
    indices = starts[:, None] + np.arange(n_fft, dtype=np.int64)[None, :]
    frames = padded[indices] * window[None, :]
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1).T
    centers = starts.astype(np.float64)  # zero frame begins at original t=0 center
    return spectrum, centers, window


def _istft(
    spectrum: np.ndarray,
    output_length: int,
    *,
    window: np.ndarray,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    n_fft = int(window.size)
    frames = np.fft.irfft(np.asarray(spectrum).T, n=n_fft, axis=1)
    total = (frames.shape[0] - 1) * hop_length + n_fft
    out = np.zeros(total, dtype=np.float64)
    norm = np.zeros(total, dtype=np.float64)
    for frame_index, frame in enumerate(frames):
        start = frame_index * hop_length
        out[start : start + n_fft] += frame * window
        norm[start : start + n_fft] += window * window
    safe = norm > 1e-12
    out[safe] /= norm[safe]
    out[~safe] = 0.0
    half = n_fft // 2
    trimmed = out[half : half + output_length]
    if trimmed.size < output_length:
        trimmed = np.pad(trimmed, (0, output_length - trimmed.size))
    return trimmed


def apply_spectral_profile(
    samples: np.ndarray,
    profile: SpectralProfile,
    *,
    sample_rate: int = CANONICAL_SAMPLE_RATE,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    """Apply a low-dimensional real magnitude mask while preserving STFT phase."""

    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("profile expects non-empty finite mono float32 audio")
    if sample_rate != CANONICAL_SAMPLE_RATE:
        raise ValueError("spectral profiles are defined only at the canonical 16 kHz rate")
    controls = (profile.global_gain_db, *profile.band_gains_db, *profile.envelope_gains_db)
    if all(float(v) == 0.0 for v in controls):
        return x.copy()  # exact identity for the baseline profile

    spectrum, frame_centers, window = _stft(x, n_fft=n_fft, hop_length=hop_length)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    band_centers = np.array([250.0, 1_600.0, min(6_500.0, sample_rate / 2)], dtype=np.float64)
    interp_x = np.log1p(frequencies)
    anchor_x = np.log1p(band_centers)
    band_db = np.interp(
        interp_x,
        anchor_x,
        np.asarray(profile.band_gains_db, dtype=np.float64),
        left=float(profile.band_gains_db[0]),
        right=float(profile.band_gains_db[-1]),
    )
    frame_times = frame_centers / sample_rate
    duration = max((x.size - 1) / sample_rate, 1.0 / sample_rate)
    normalized_times = np.clip(frame_times / duration, 0.0, 1.0)
    envelope_times = np.linspace(0.0, 1.0, len(profile.envelope_gains_db))
    envelope_db = np.interp(normalized_times, envelope_times, profile.envelope_gains_db)
    total_db = profile.global_gain_db + band_db[:, None] + envelope_db[None, :]
    total_db = np.clip(total_db, -profile.max_total_gain_db, profile.max_total_gain_db)
    magnitude_gain = np.power(10.0, total_db / 20.0)
    enhanced_spectrum = spectrum * magnitude_gain
    output = _istft(
        enhanced_spectrum,
        int(x.size),
        window=window,
        hop_length=hop_length,
    ).astype(np.float32)
    if output.shape != x.shape or not np.isfinite(output).all():
        raise RuntimeError("spectral profile violated finite/exact-length output contract")
    return np.ascontiguousarray(output)


@dataclass(frozen=True)
class ProfileSearchResult:
    split_role: str
    selected_index: int
    profile: Mapping[str, object]
    objective: float
    candidate_results: tuple[Mapping[str, object], ...]


def search_profiles(
    profiles: Sequence[SpectralProfile],
    evaluate: Callable[[SpectralProfile], float],
    *,
    split_role: str,
    higher_is_better: bool = True,
) -> ProfileSearchResult:
    """Evaluate a finite low-dimensional profile grid on fit data only.

    The explicit role gate prevents this convenience API from tuning on a
    dev/confirm/evaluation set. The evaluation callback is responsible for
    hashing and recording its fit-only input rows and objective definition.
    """

    if split_role != "fit":
        raise ValueError("black-box profile search is permitted only on the fit split")
    if not profiles:
        raise ValueError("at least one profile candidate is required")
    rows: list[dict[str, object]] = []
    for index, profile in enumerate(profiles):
        score = float(evaluate(profile))
        if not math.isfinite(score):
            raise ValueError(f"candidate {index} produced a non-finite objective")
        rows.append({"index": index, "profile": profile.to_dict(), "objective": score})
    best = max(rows, key=lambda row: float(row["objective"])) if higher_is_better else min(
        rows, key=lambda row: float(row["objective"])
    )
    return ProfileSearchResult(
        split_role=split_role,
        selected_index=int(best["index"]),
        profile=dict(best["profile"]),
        objective=float(best["objective"]),
        candidate_results=tuple(rows),
    )
