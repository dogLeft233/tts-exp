"""Descriptive phase and source-clock diagnostics for exact-length PCM pairs.

Preparation helper only: importing this module performs no audio work. A caller
must pass the pinned STFT function and the exact complex coefficients produced
by its transform before ISTFT. The returned arrays distinguish coefficient-time
phase from the phase measured after waveform reconstruction and reanalysis.

All lag and transient-centroid values are acoustic sample-clock descriptors,
not phoneme alignment, biological timing, or acceptance thresholds. The helper
does not shift audio, alter a transform, or make a method-selection decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Callable, Mapping

import numpy as np


SCHEMA = "cross_tfg_phase_clock_diagnostics_v1"


@dataclass(frozen=True)
class PhaseClockDiagnosticsV1:
    """Raw numeric arrays and their descriptive, threshold-free summary."""

    arrays: Mapping[str, np.ndarray]
    summary: Mapping[str, Any]


def _canonical_f32_sha(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _canonical_c128_sha(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(values, dtype="<c16"))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256")
    return value


def _exact_pcm(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.float32 or array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be nonempty mono float32 PCM")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite PCM")
    return np.ascontiguousarray(array, dtype=np.float32)


def _stft_result(
    stft: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    pcm: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = stft(pcm.copy(), n_fft=n_fft, hop_length=hop_length)
    if not isinstance(result, tuple) or len(result) < 3:
        raise ValueError(f"{name} STFT must return (complex_spectrum, centers, window)")
    spectrum = np.asarray(result[0])
    centers_raw = np.asarray(result[1])
    window = np.asarray(result[2])
    if spectrum.ndim != 2 or not np.iscomplexobj(spectrum):
        raise ValueError(f"{name} STFT spectrum must be complex frequency-by-time data")
    if not np.isfinite(spectrum.real).all() or not np.isfinite(spectrum.imag).all():
        raise ValueError(f"{name} STFT spectrum contains non-finite values")
    if centers_raw.ndim != 1 or centers_raw.size != spectrum.shape[1]:
        raise ValueError(f"{name} STFT centers do not match its time dimension")
    if not np.issubdtype(centers_raw.dtype, np.integer):
        if not np.isfinite(centers_raw).all() or not np.equal(
            centers_raw, np.rint(centers_raw)
        ).all():
            raise ValueError(f"{name} STFT centers must be integer source samples")
    centers = np.asarray(centers_raw, dtype=np.int64)
    if centers.size > 1 and not np.all(np.diff(centers) > 0):
        raise ValueError(f"{name} STFT centers must increase strictly")
    if window.ndim != 1 or window.size == 0 or not np.isfinite(window).all():
        raise ValueError(f"{name} STFT window must be finite one-dimensional data")
    return np.ascontiguousarray(spectrum), centers, np.ascontiguousarray(window)


def _phase_pair(
    left: np.ndarray,
    right: np.ndarray,
    *,
    name: str,
    shared_support: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if left.shape != right.shape or left.shape != shared_support.shape:
        raise ValueError(f"phase pair {name} has mismatched coefficient shapes")
    left64 = np.asarray(left, dtype=np.complex128)
    right64 = np.asarray(right, dtype=np.complex128)
    weight = np.sqrt(np.abs(left64) * np.abs(right64))
    delta = np.angle(left64 * np.conjugate(right64))
    supported = (
        np.asarray(shared_support, dtype=bool)
        & np.isfinite(weight)
        & np.isfinite(delta)
        & (weight > 0.0)
    )
    weight = np.where(supported, weight, 0.0)
    delta = np.where(supported, delta, 0.0)
    weight_sum = float(np.sum(weight, dtype=np.float64))
    if weight_sum > 0.0:
        sine = float(np.sum(weight * np.sin(delta), dtype=np.float64))
        cosine = float(np.sum(weight * np.cos(delta), dtype=np.float64))
        circular_mean = math.atan2(sine, cosine)
        resultant = math.hypot(sine, cosine) / weight_sum
        mean_abs = float(np.sum(weight * np.abs(delta), dtype=np.float64) / weight_sum)
        rms = float(np.sqrt(np.sum(weight * delta * delta, dtype=np.float64) / weight_sum))
    else:
        circular_mean = resultant = mean_abs = rms = None
    arrays = {
        f"{name}_phase_delta_rad": delta.astype(np.float64),
        f"{name}_geometric_magnitude_weight": weight.astype(np.float64),
        f"{name}_phase_supported": supported.astype(np.uint8),
    }
    summary = {
        "weight_definition": "sqrt(abs(left_coefficient) * abs(right_coefficient))",
        "phase_delta_definition": "angle(left * conjugate(right)) radians",
        "weighted_support_bins": int(np.count_nonzero(supported)),
        "weight_sum": weight_sum,
        "weighted_circular_mean_error_rad": circular_mean,
        "weighted_circular_resultant_length": resultant,
        "weighted_mean_absolute_error_rad": mean_abs,
        "weighted_rms_error_rad": rms,
    }
    return arrays, summary


def _local_waveform_xcorr(
    source: np.ndarray,
    output: np.ndarray,
    *,
    window_samples: int,
    hop_samples: int,
    max_lag_samples: int,
) -> dict[str, np.ndarray]:
    if window_samples <= 1 or hop_samples <= 0 or max_lag_samples < 0:
        raise ValueError("local correlation window/hop/lag parameters are invalid")
    if window_samples > source.size:
        raise ValueError("local correlation window exceeds the exact PCM sample count")
    starts = np.arange(0, source.size - window_samples + 1, hop_samples, dtype=np.int64)
    lags = np.arange(-max_lag_samples, max_lag_samples + 1, dtype=np.int64)
    curves = np.full((starts.size, lags.size), np.nan, dtype=np.float64)
    valid = np.zeros((starts.size, lags.size), dtype=np.uint8)
    best_lag = np.full(starts.size, np.iinfo(np.int64).min, dtype=np.int64)
    best_corr = np.full(starts.size, np.nan, dtype=np.float64)
    tie_count = np.zeros(starts.size, dtype=np.int64)

    for row, start_value in enumerate(starts):
        start = int(start_value)
        reference = np.asarray(source[start : start + window_samples], dtype=np.float64)
        reference = reference - float(np.mean(reference, dtype=np.float64))
        reference_norm = float(np.linalg.norm(reference))
        if reference_norm == 0.0:
            continue
        for column, lag_value in enumerate(lags):
            lag = int(lag_value)
            output_start = start + lag
            output_stop = output_start + window_samples
            if output_start < 0 or output_stop > output.size:
                continue
            candidate = np.asarray(output[output_start:output_stop], dtype=np.float64)
            candidate = candidate - float(np.mean(candidate, dtype=np.float64))
            candidate_norm = float(np.linalg.norm(candidate))
            if candidate_norm == 0.0:
                continue
            correlation = float(np.dot(reference, candidate) / (reference_norm * candidate_norm))
            curves[row, column] = correlation
            valid[row, column] = 1
        row_valid = np.flatnonzero(valid[row])
        if row_valid.size:
            row_values = curves[row, row_valid]
            maximum = float(np.max(row_values))
            ties = row_valid[row_values == maximum]
            tie_count[row] = ties.size
            chosen = min(ties.tolist(), key=lambda idx: (abs(int(lags[idx])), int(lags[idx])))
            best_lag[row] = int(lags[chosen])
            best_corr[row] = maximum

    return {
        "local_xcorr_window_start_samples": starts,
        "local_xcorr_window_samples": np.asarray([window_samples], dtype=np.int64),
        "local_xcorr_hop_samples": np.asarray([hop_samples], dtype=np.int64),
        "local_xcorr_lag_grid_samples": lags,
        "local_xcorr_curve": curves,
        "local_xcorr_valid": valid,
        "local_xcorr_argmax_output_delay_samples": best_lag,
        "local_xcorr_argmax_value": best_corr,
        "local_xcorr_exact_tie_count": tie_count,
    }


def _energy_centroid_by_frame(
    source: np.ndarray,
    output: np.ndarray,
    *,
    sample_rate_hz: int,
) -> dict[str, np.ndarray]:
    frame_samples = sample_rate_hz // 25
    frame_count = source.size // frame_samples
    used_samples = frame_count * frame_samples
    source_frames = np.asarray(source[:used_samples], dtype=np.float64).reshape(
        frame_count, frame_samples
    )
    output_frames = np.asarray(output[:used_samples], dtype=np.float64).reshape(
        frame_count, frame_samples
    )
    source_power = source_frames * source_frames
    output_power = output_frames * output_frames
    positions = np.arange(frame_samples, dtype=np.float64) + 0.5
    source_total = np.sum(source_power, axis=1, dtype=np.float64)
    output_total = np.sum(output_power, axis=1, dtype=np.float64)
    source_centroid = np.full(frame_count, np.nan, dtype=np.float64)
    output_centroid = np.full(frame_count, np.nan, dtype=np.float64)
    source_valid = source_total > 0.0
    output_valid = output_total > 0.0
    source_centroid[source_valid] = (
        source_power[source_valid] @ positions / source_total[source_valid]
    )
    output_centroid[output_valid] = (
        output_power[output_valid] @ positions / output_total[output_valid]
    )
    delta = output_centroid - source_centroid
    return {
        "acoustic_energy_frame_start_samples_40ms": (
            np.arange(frame_count, dtype=np.int64) * frame_samples
        ),
        "source_acoustic_energy_centroid_within_frame_samples_40ms": source_centroid,
        "output_acoustic_energy_centroid_within_frame_samples_40ms": output_centroid,
        "output_minus_source_acoustic_energy_centroid_samples_40ms": delta,
        "source_frame_nonzero_energy": source_valid.astype(np.uint8),
        "output_frame_nonzero_energy": output_valid.astype(np.uint8),
        "trailing_samples_excluded_from_40ms_frames": np.asarray(
            [source.size - used_samples], dtype=np.int64
        ),
    }


def phase_clock_reanalysis_diagnostics_v1(
    source_pcm_float32: np.ndarray,
    output_pcm_float32: np.ndarray,
    *,
    constructed_output_stft: np.ndarray,
    stft: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]],
    stft_source_sha256: str,
    coefficient_transform_source_sha256: str,
    sample_rate_hz: int,
    n_fft: int,
    hop_length: int,
    local_xcorr_window_samples: int,
    local_xcorr_hop_samples: int,
    max_lag_samples: int,
) -> PhaseClockDiagnosticsV1:
    """Compare original, constructed, and reanalyzed coefficients on exact support.

    Parameters are mandatory so callers cannot silently inherit a lag window or
    feature setting. ``constructed_output_stft`` is the transformer's complex
    coefficient array before ISTFT; ``stft`` is the exact pinned analysis
    function used both for source and reconstructed output. Its implementation
    SHA and the coefficient-transform source SHA are mandatory provenance pins.
    Positive local-xcorr lag means output samples at ``t + lag`` best match
    source samples at ``t``. No lag or phase threshold is applied.
    """

    source = _exact_pcm("source_pcm_float32", source_pcm_float32)
    output = _exact_pcm("output_pcm_float32", output_pcm_float32)
    stft_source_sha256 = _require_sha256("stft_source_sha256", stft_source_sha256)
    coefficient_transform_source_sha256 = _require_sha256(
        "coefficient_transform_source_sha256", coefficient_transform_source_sha256
    )
    if output.size != source.size:
        raise ValueError("source and output must have identical exact sample counts")
    if sample_rate_hz <= 0 or sample_rate_hz % 25 != 0:
        raise ValueError("sample rate must yield an integer number of samples per 40 ms")
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_fft and hop_length must be positive")
    if hop_length > n_fft:
        raise ValueError("hop_length must not exceed n_fft")

    source_stft, centers, source_window = _stft_result(
        stft, source, n_fft=n_fft, hop_length=hop_length, name="source"
    )
    reanalysis_stft, output_centers, output_window = _stft_result(
        stft, output, n_fft=n_fft, hop_length=hop_length, name="reanalysis"
    )
    if not np.array_equal(centers, output_centers):
        raise ValueError("source/reanalysis STFT frame centers differ; support is not shared")
    if not np.array_equal(source_window, output_window):
        raise ValueError("source/reanalysis STFT windows differ")
    constructed = np.asarray(constructed_output_stft)
    if constructed.shape != source_stft.shape or not np.iscomplexobj(constructed):
        raise ValueError("constructed output STFT must match source complex support exactly")
    if not np.isfinite(constructed.real).all() or not np.isfinite(constructed.imag).all():
        raise ValueError("constructed output STFT contains non-finite values")
    constructed = np.ascontiguousarray(constructed)
    common_phase_support = (
        (np.abs(source_stft) > 0.0)
        & (np.abs(constructed) > 0.0)
        & (np.abs(reanalysis_stft) > 0.0)
    )

    arrays: dict[str, np.ndarray] = {
        "source_pcm_float32": source.copy(),
        "output_pcm_float32": output.copy(),
        "source_stft_complex": source_stft.copy(),
        "constructed_output_stft_complex": constructed.copy(),
        "reanalysis_output_stft_complex": reanalysis_stft.copy(),
        "source_clock_stft_frame_centers_samples": centers.copy(),
        "stft_window": source_window.copy(),
        "stft_shared_nonzero_phase_support_all_three": common_phase_support.astype(
            np.uint8
        ),
    }
    phase_summaries: dict[str, Any] = {}
    for label, left, right in (
        ("construction_vs_source", constructed, source_stft),
        ("reanalysis_vs_constructed", reanalysis_stft, constructed),
        ("reanalysis_vs_source", reanalysis_stft, source_stft),
    ):
        phase_arrays, phase_summary = _phase_pair(
            left,
            right,
            name=label,
            shared_support=common_phase_support,
        )
        arrays.update(phase_arrays)
        phase_summaries[label] = phase_summary

    frame_samples = sample_rate_hz // 25
    frame_count = source.size // frame_samples
    used_samples = frame_count * frame_samples
    source_frames = np.asarray(source[:used_samples], dtype=np.float64).reshape(
        frame_count, frame_samples
    )
    output_frames = np.asarray(output[:used_samples], dtype=np.float64).reshape(
        frame_count, frame_samples
    )
    source_power = np.mean(source_frames * source_frames, axis=1, dtype=np.float64)
    output_power = np.mean(output_frames * output_frames, axis=1, dtype=np.float64)
    frame_ratio_valid = (source_power > 0.0) & (output_power > 0.0)
    frame_ratio_db = np.full(frame_count, np.nan, dtype=np.float64)
    frame_ratio_db[frame_ratio_valid] = 10.0 * np.log10(
        output_power[frame_ratio_valid] / source_power[frame_ratio_valid]
    )
    arrays.update(
        {
            "physical_frame_start_samples_40ms": (
                np.arange(frame_count, dtype=np.int64) * frame_samples
            ),
            "physical_source_frame_mean_square_40ms": source_power,
            "physical_output_frame_mean_square_40ms": output_power,
            "physical_frame_power_ratio_db_40ms": frame_ratio_db,
            "physical_frame_power_ratio_valid_both_nonzero_40ms": (
                frame_ratio_valid.astype(np.uint8)
            ),
        }
    )
    arrays.update(
        _energy_centroid_by_frame(source, output, sample_rate_hz=sample_rate_hz)
    )
    arrays.update(
        _local_waveform_xcorr(
            source,
            output,
            window_samples=local_xcorr_window_samples,
            hop_samples=local_xcorr_hop_samples,
            max_lag_samples=max_lag_samples,
        )
    )

    summary = {
        "schema": SCHEMA,
        "sample_rate_hz": int(sample_rate_hz),
        "sample_count_source": int(source.size),
        "sample_count_output": int(output.size),
        "source_pcm_float32_sha256": _canonical_f32_sha(source),
        "output_pcm_float32_sha256": _canonical_f32_sha(output),
        "stft_source_sha256": stft_source_sha256,
        "coefficient_transform_source_sha256": coefficient_transform_source_sha256,
        "source_stft_canonical_complex128_sha256": _canonical_c128_sha(source_stft),
        "constructed_output_stft_canonical_complex128_sha256": _canonical_c128_sha(
            constructed
        ),
        "reanalysis_output_stft_canonical_complex128_sha256": _canonical_c128_sha(
            reanalysis_stft
        ),
        "stft_frame_count": int(centers.size),
        "stft_frequency_bin_count": int(source_stft.shape[0]),
        "phase_shared_nonzero_support_bins_all_three": int(
            np.count_nonzero(common_phase_support)
        ),
        "stft_centers_identical_source_and_reanalysis": True,
        "stft_window_identical_source_and_reanalysis": True,
        "phase_comparisons": phase_summaries,
        "frame_power_40ms": {
            "frame_samples": int(frame_samples),
            "frame_count": int(frame_count),
            "trailing_samples_excluded": int(source.size - used_samples),
            "source_zero_power_frames": int(np.count_nonzero(source_power == 0.0)),
            "output_zero_power_frames": int(np.count_nonzero(output_power == 0.0)),
            "ratio_valid_both_nonzero_frames": int(np.count_nonzero(frame_ratio_valid)),
            "mean_ratio_db": (
                float(np.mean(frame_ratio_db[frame_ratio_valid]))
                if np.any(frame_ratio_valid)
                else None
            ),
            "mean_absolute_ratio_db": (
                float(np.mean(np.abs(frame_ratio_db[frame_ratio_valid])))
                if np.any(frame_ratio_valid)
                else None
            ),
        },
        "local_waveform_xcorr": {
            "window_samples": int(local_xcorr_window_samples),
            "hop_samples": int(local_xcorr_hop_samples),
            "lag_grid_min_samples": -int(max_lag_samples),
            "lag_grid_max_samples": int(max_lag_samples),
            "positive_lag_means": "output[t + lag] compared with source[t]",
            "argmax_is_descriptive_only": True,
            "no_threshold_or_audio_shift_applied": True,
        },
        "transient_centroid_semantics": (
            "40 ms within-frame squared-waveform-energy centroid; acoustic descriptor, "
            "not a phoneme or articulation time"
        ),
        "phase_semantics": (
            "construction compares complex coefficients before ISTFT; reanalysis compares "
            "STFT(output PCM) with those constructed coefficients and source; no waveform "
            "phase identity is asserted"
        ),
        "scope": "descriptive acoustic diagnostics only; no method selection or thresholds",
    }
    return PhaseClockDiagnosticsV1(arrays=arrays, summary=summary)
