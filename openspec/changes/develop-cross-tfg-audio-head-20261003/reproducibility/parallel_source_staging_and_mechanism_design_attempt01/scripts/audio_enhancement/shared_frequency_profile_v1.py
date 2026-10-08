"""Small, recipient-adaptive, phase-preserving frequency-profile family.

This is a new, isolated candidate family.  It does not alter the existing
identity, fixed-RMS, local-compressor, or learned-head paths.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from .audio_io import CANONICAL_SAMPLE_RATE
from .baselines import (
    FIXED_POSITIVE_CONTROL_RMS,
    _istft,
    _stft,
    apply_global_rms,
    apply_local_envelope_compression,
    waveform_rms,
)


SCHEMA = "cross_tfg_shared_frequency_profile_v1"
N_FFT = 512
HOP_LENGTH = 128
SMOOTH_FREQUENCY_SIGMA_BINS = 2.0
SMOOTH_TIME_SIGMA_FRAMES = 2.0
MAX_MASK_DB = 3.0
LOCAL_WINDOW_MS = 32.0
LOCAL_COMPRESSION_RATIO = 0.90
LOCAL_GAIN_MIN_DB = -6.0
LOCAL_GAIN_MAX_DB = 6.0
DEFAULT_PEAK_CEILING = 0.98


@dataclass(frozen=True)
class FrequencyProfile:
    profile_id: str
    contrast_strength: float
    low_high_tilt_db: float
    local_compression_ratio: float


def _make_profiles() -> tuple[FrequencyProfile, ...]:
    specs: list[FrequencyProfile] = []
    for magnitude in (0.10, 0.20, 0.30):
        for sign, suffix in ((1.0, "flatten"), (-1.0, "accent")):
            strength = sign * magnitude
            specs.append(
                FrequencyProfile(
                    profile_id=f"local_r090_{suffix}_{magnitude:.2f}_tilt0",
                    contrast_strength=strength,
                    low_high_tilt_db=0.0,
                    local_compression_ratio=LOCAL_COMPRESSION_RATIO,
                )
            )
    specs.extend(
        (
            FrequencyProfile("local_r090_tilt_m025", 0.0, -0.25, LOCAL_COMPRESSION_RATIO),
            FrequencyProfile("local_r090_tilt_p025", 0.0, 0.25, LOCAL_COMPRESSION_RATIO),
            FrequencyProfile("local_r090_flatten015_tilt_m025", 0.15, -0.25, LOCAL_COMPRESSION_RATIO),
            FrequencyProfile("local_r090_accent015_tilt_p025", -0.15, 0.25, LOCAL_COMPRESSION_RATIO),
            FrequencyProfile("raw_flatten020_tilt0", 0.20, 0.0, 1.0),
            FrequencyProfile("raw_accent020_tilt0", -0.20, 0.0, 1.0),
        )
    )
    if len(specs) != 12 or len({profile.profile_id for profile in specs}) != 12:
        raise RuntimeError("the v1 shared profile family must contain 12 unique candidates")
    return tuple(specs)


PROFILES = _make_profiles()
PROFILES_BY_ID = {profile.profile_id: profile for profile in PROFILES}


def profile_dict(profile: FrequencyProfile | str) -> dict[str, Any]:
    if isinstance(profile, str):
        try:
            profile = PROFILES_BY_ID[profile]
        except KeyError as exc:
            raise ValueError(f"unknown shared profile id: {profile}") from exc
    return {
        "schema": SCHEMA,
        **asdict(profile),
        "sample_rate_hz": CANONICAL_SAMPLE_RATE,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "smoothing": {
            "frequency_sigma_bins": SMOOTH_FREQUENCY_SIGMA_BINS,
            "time_sigma_frames": SMOOTH_TIME_SIGMA_FRAMES,
            "boundary": "reflect",
        },
        "mask": {
            "definition": "-contrast_strength*(log_magnitude_db-gaussian_smooth(log_magnitude_db))+log_frequency_tilt_db",
            "zero_mean_per_frame": True,
            "hard_bound_db": [-MAX_MASK_DB, MAX_MASK_DB],
            "phase": "input_stft_phase_preserved",
        },
        "postprocess": {
            "fixed_rms_target": FIXED_POSITIVE_CONTROL_RMS,
            "peak_ceiling": DEFAULT_PEAK_CEILING,
            "clipping": "uniform_peak_attenuation_only",
        },
    }


def candidate_profiles() -> tuple[dict[str, Any], ...]:
    """Return the frozen, ordered 12-candidate grid as JSON-safe objects."""

    return tuple(profile_dict(profile) for profile in PROFILES)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Gaussian smoothing sigma must be positive and finite")
    radius = max(1, int(math.ceil(3.0 * sigma)))
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * np.square(positions / sigma))
    kernel /= kernel.sum(dtype=np.float64)
    return kernel


def _smooth_axis(values: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(np.asarray(values, dtype=np.float64), axis, -1)
    radius = kernel.size // 2
    padded = np.pad(moved, [(0, 0)] * (moved.ndim - 1) + [(radius, radius)], mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.size, axis=-1)
    smoothed = np.einsum("...k,k->...", windows, kernel, optimize=True, dtype=np.float64)
    return np.moveaxis(smoothed, -1, axis)


def _profile_from_value(profile: FrequencyProfile | str) -> FrequencyProfile:
    if isinstance(profile, FrequencyProfile):
        if profile.profile_id not in PROFILES_BY_ID or PROFILES_BY_ID[profile.profile_id] != profile:
            raise ValueError("profile is not one of the frozen v1 family entries")
        return profile
    try:
        return PROFILES_BY_ID[profile]
    except KeyError as exc:
        raise ValueError(f"unknown shared profile id: {profile}") from exc


def transform_shared_profile(
    samples: np.ndarray,
    profile: FrequencyProfile | str,
    *,
    sample_rate_hz: int = CANONICAL_SAMPLE_RATE,
    target_rms: float = FIXED_POSITIVE_CONTROL_RMS,
    peak_ceiling: float = DEFAULT_PEAK_CEILING,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a bounded local spectral profile while preserving phase and clock.

    The optional ratio-0.90 local compressor is applied before the spectral
    mask.  A single utterance-wide RMS/peak stage follows every profile.
    """

    spec = _profile_from_value(profile)
    x = np.asarray(samples)
    if sample_rate_hz != CANONICAL_SAMPLE_RATE:
        raise ValueError("shared frequency profiles require 16 kHz canonical audio")
    if x.dtype != np.float32 or x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("profile input must be non-empty finite mono float32")
    if not math.isfinite(target_rms) or target_rms <= 0:
        raise ValueError("target_rms must be positive and finite")
    if not math.isfinite(peak_ceiling) or peak_ceiling <= 0:
        raise ValueError("peak_ceiling must be positive and finite")

    before_rms = waveform_rms(x)
    if spec.local_compression_ratio < 1.0:
        working, compression = apply_local_envelope_compression(
            x,
            sample_rate=sample_rate_hz,
            window_ms=LOCAL_WINDOW_MS,
            ratio=spec.local_compression_ratio,
            min_gain_db=LOCAL_GAIN_MIN_DB,
            max_gain_db=LOCAL_GAIN_MAX_DB,
        )
    else:
        working = x.copy()
        compression = None

    spectrum, frame_centers, window = _stft(working, n_fft=N_FFT, hop_length=HOP_LENGTH)
    magnitude = np.abs(spectrum)
    log_magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-8))
    smoothed = _smooth_axis(
        log_magnitude_db,
        _gaussian_kernel(SMOOTH_FREQUENCY_SIGMA_BINS),
        axis=0,
    )
    smoothed = _smooth_axis(
        smoothed,
        _gaussian_kernel(SMOOTH_TIME_SIGMA_FRAMES),
        axis=1,
    )
    local_contrast = log_magnitude_db - smoothed

    frequencies = np.fft.rfftfreq(N_FFT, d=1.0 / sample_rate_hz)
    log_frequency = np.log1p(frequencies)
    tilt_axis = 2.0 * (log_frequency - log_frequency.min()) / max(
        float(log_frequency.max() - log_frequency.min()), 1e-12
    ) - 1.0
    tilt_axis -= tilt_axis.mean(dtype=np.float64)
    mask_db = -float(spec.contrast_strength) * local_contrast
    mask_db = mask_db + float(spec.low_high_tilt_db) * tilt_axis[:, None]

    # Project each frame onto the zero-mean, bounded mask set. Scaling after
    # centering keeps the exact per-frame mean and the declared +/-3 dB bound.
    mask_db -= mask_db.mean(axis=0, keepdims=True, dtype=np.float64)
    peak_per_frame = np.max(np.abs(mask_db), axis=0, keepdims=True)
    scale = np.minimum(1.0, MAX_MASK_DB / np.maximum(peak_per_frame, 1e-12))
    mask_db *= scale
    if np.any(np.abs(mask_db) > MAX_MASK_DB + 1e-10):
        raise RuntimeError("projected local spectral mask exceeded its frozen dB bound")

    gain = np.power(10.0, mask_db / 20.0)
    modified_spectrum = spectrum * gain
    reconstructed = _istft(
        modified_spectrum,
        int(x.size),
        window=window,
        hop_length=HOP_LENGTH,
    ).astype(np.float32)
    if reconstructed.shape != x.shape or not np.isfinite(reconstructed).all():
        raise RuntimeError("local spectral profile changed the sample clock or returned non-finite audio")
    output, rms_info = apply_global_rms(
        reconstructed,
        target_rms=target_rms,
        peak_ceiling=peak_ceiling,
    )
    if output.shape != x.shape or output.dtype != np.float32:
        raise RuntimeError("shared profile did not preserve the canonical float32 waveform contract")

    diagnostics = {
        "schema": SCHEMA,
        "profile": profile_dict(spec),
        "sample_count": int(x.size),
        "sample_rate_hz": int(sample_rate_hz),
        "input_rms": before_rms,
        "after_local_compression_rms": waveform_rms(working),
        "before_fixed_rms_stage_rms": waveform_rms(reconstructed),
        "output_rms": waveform_rms(output),
        "output_peak": float(np.max(np.abs(output.astype(np.float64)))),
        "fixed_rms_stage": asdict(rms_info),
        "local_compression": asdict(compression) if compression is not None else None,
        "mask_db": {
            "minimum": float(mask_db.min()),
            "maximum": float(mask_db.max()),
            "maximum_absolute": float(np.max(np.abs(mask_db))),
            "max_absolute_frame_mean": float(np.max(np.abs(mask_db.mean(axis=0, dtype=np.float64)))),
            "max_absolute_bin_mean": float(np.max(np.abs(mask_db.mean(axis=1, dtype=np.float64)))),
            "frame_count": int(mask_db.shape[1]),
            "frequency_bin_count": int(mask_db.shape[0]),
        },
        "stft_frame_centers_sample_indices": [int(frame_centers[0]), int(frame_centers[-1])],
        "clock_preserved": int(output.size) == int(x.size),
        "phase_policy": "input_stft_phase_preserved",
        "selection_role": "fit_only_profile_family_candidate",
    }
    return np.ascontiguousarray(output, dtype=np.float32), diagnostics
