"""Deterministic, equal-clock audio interventions for the attribution study.

The implementation in this module deliberately avoids a signal-processing
library's hidden defaults.  The protocol uses a 512 point periodic Hann STFT,
float64 arithmetic, and an explicit overlap-add inverse.  The resulting WAVs
are the only audio inputs consumed by the new experiment.
"""

from __future__ import annotations

import hashlib
import math
import platform
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    ProtocolError,
    decode_media_pcm16,
    file_sha256,
    pcm_hash,
    read_json,
    read_self_hashed_json,
    write_json,
    write_self_hashed_json,
)


def pcm16_to_float(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype != np.int16 or array.ndim != 1 or array.size == 0:
        raise ProtocolError("PCM input must be a non-empty one-dimensional int16 array")
    result = array.astype(np.float64) / 32768.0
    if not np.isfinite(result).all():
        raise ProtocolError("PCM input contains non-finite samples")
    return result


def quantize_pcm16(values: np.ndarray) -> np.ndarray:
    """Round to nearest int16 with NumPy's ties-to-even rule, without clipping."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ProtocolError("waveform is empty or contains non-finite samples")
    scaled = np.rint(array * 32768.0)
    if float(scaled.min()) < -32768.0 or float(scaled.max()) > 32767.0:
        raise ProtocolError("PCM quantization would exceed int16 range; clipping is forbidden")
    return scaled.astype("<i2")


def read_pcm16_wav(path: str | Path) -> np.ndarray:
    target = Path(path)
    try:
        with wave.open(str(target), "rb") as handle:
            channels = int(handle.getnchannels())
            sample_width = int(handle.getsampwidth())
            sample_rate = int(handle.getframerate())
            sample_count = int(handle.getnframes())
            payload = handle.readframes(sample_count)
    except (OSError, wave.Error) as exc:
        raise ProtocolError(f"cannot read WAV: {target}") from exc
    if (channels, sample_width, sample_rate) != (1, 2, config.SAMPLE_RATE):
        raise ProtocolError(f"WAV violates 16 kHz mono PCM16 contract: {target}")
    values = np.frombuffer(payload, dtype="<i2").copy()
    if values.size != sample_count or values.size == 0:
        raise ProtocolError(f"WAV frame count is invalid: {target}")
    return values


def write_pcm16_wav(path: str | Path, values: np.ndarray) -> str:
    target = Path(path)
    array = np.asarray(values)
    if array.dtype != np.int16 or array.ndim != 1 or array.size == 0:
        raise ProtocolError("WAV output must be a non-empty one-dimensional int16 array")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{__import__('os').getpid()}.tmp")
    try:
        with wave.open(str(temporary), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(config.SAMPLE_RATE)
            handle.writeframes(np.asarray(array, dtype="<i2").tobytes())
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(target)


def periodic_hann() -> np.ndarray:
    # endpoint=False is the definition of a periodic Hann window.
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(config.WINDOW, dtype=np.float64) / config.WINDOW)).astype(np.float64)


def _reflect_pad(values: np.ndarray, amount: int = config.PAD) -> np.ndarray:
    if values.size < 2:
        raise ProtocolError("reflect padding requires at least two source samples")
    return np.pad(values, (amount, amount), mode="reflect")


def stft(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unnormalised positive-frequency spectra and padded frame starts."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 2 or not np.isfinite(array).all():
        raise ProtocolError("STFT input is too short or non-finite")
    padded = _reflect_pad(array)
    frame_count = 1 + (padded.size - config.WINDOW) // config.HOP
    if frame_count < 1:
        raise ProtocolError("STFT produced no frames")
    starts = np.arange(frame_count, dtype=np.int64) * config.HOP
    frames = np.stack([padded[start : start + config.WINDOW] for start in starts], axis=0)
    spectra = np.fft.rfft(frames * periodic_hann()[None, :], n=config.NFFT, axis=1)
    return np.asarray(spectra, dtype=np.complex128), starts


def istft(spectra: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """Explicit inverse STFT with window-square overlap-add and exact trimming."""

    value = np.asarray(spectra, dtype=np.complex128)
    starts_array = np.asarray(starts, dtype=np.int64).reshape(-1)
    if value.ndim != 2 or value.shape[1] != config.NFFT // 2 + 1 or value.shape[0] != starts_array.size:
        raise ProtocolError(f"invalid STFT shape for inverse: {value.shape}")
    if length < 1 or starts_array.size == 0:
        raise ProtocolError("invalid iSTFT length")
    output_length = int(starts_array[-1]) + config.WINDOW
    output = np.zeros(output_length, dtype=np.float64)
    window_square = np.zeros(output_length, dtype=np.float64)
    window = periodic_hann()
    for row, start in zip(value, starts_array, strict=True):
        frame = np.fft.irfft(row, n=config.NFFT).astype(np.float64, copy=False)
        output[start : start + config.WINDOW] += frame * window
        window_square[start : start + config.WINDOW] += window * window
    reconstructed = np.divide(output, window_square, out=np.zeros_like(output), where=window_square > 1e-12)
    left = config.PAD
    right = left + int(length)
    if right > reconstructed.size:
        raise ProtocolError("iSTFT did not cover the requested length")
    result = reconstructed[left:right]
    if result.size != length or not np.isfinite(result).all():
        raise ProtocolError("iSTFT reconstruction is invalid")
    return result


def frame_rms_and_activity(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw-frame RMS, active-frame flags, and the union mask in source time."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    _, starts = stft(array)
    padded = _reflect_pad(array)
    frames = np.stack([padded[start : start + config.WINDOW] for start in starts], axis=0)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1, dtype=np.float64))
    peak = float(frame_rms.max())
    threshold = max(peak * 0.1, 10.0 ** (-50.0 / 20.0))
    active_frames = frame_rms > threshold
    mask = np.zeros(array.size, dtype=bool)
    for start, active in zip(starts, active_frames, strict=True):
        if not active:
            continue
        source_start = max(0, int(start) - config.PAD)
        source_end = min(array.size, int(start) - config.PAD + config.WINDOW)
        if source_end > source_start:
            mask[source_start:source_end] = True
    return frame_rms, active_frames, mask


def low_energy_noise_power(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    spectra, starts = stft(values)
    frames = _reflect_pad(np.asarray(values, dtype=np.float64))
    raw = np.stack([frames[start : start + config.WINDOW] for start in starts], axis=0)
    frame_energy = np.mean(raw * raw, axis=1, dtype=np.float64)
    order = np.lexsort((np.arange(frame_energy.size), frame_energy))
    count = max(1, math.ceil(0.10 * frame_energy.size))
    selected = np.asarray(order[:count], dtype=np.int64)
    power = np.mean(np.abs(spectra[selected]) ** 2, axis=0, dtype=np.float64)
    if not np.isfinite(power).all() or power.size != config.NFFT // 2 + 1:
        raise ProtocolError("low-energy noise power is invalid")
    return power, {
        "frame_count": int(frame_energy.size),
        "selected_count": count,
        "selected_frame_indices": [int(item) for item in selected],
        "selection_rule": "lowest raw-frame energy; ties by frame index",
        "power_sha256": hashlib.sha256(np.asarray(power, dtype="<f8").tobytes()).hexdigest(),
    }


def noise_realization(
    length: int,
    natural_noise_power: np.ndarray,
    *,
    sample_id: int,
    source: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if length < 2:
        raise ProtocolError("noise realization is too short")
    base = np.asarray(natural_noise_power, dtype=np.float64).reshape(-1)
    if base.size != config.NFFT // 2 + 1 or not np.isfinite(base).all() or float(base.mean()) <= 0.0:
        raise ProtocolError("natural low-energy noise power is invalid")
    target_frequency = np.fft.rfftfreq(length, d=1.0 / config.SAMPLE_RATE)
    source_frequency = np.fft.rfftfreq(config.NFFT, d=1.0 / config.SAMPLE_RATE)
    colored_power = np.interp(target_frequency, source_frequency, base)
    colored_power = colored_power + 1e-6 * float(base.mean())
    seed_material = f"native-gain-v1|{int(sample_id)}|{source}".encode()
    seed_digest = hashlib.sha256(seed_material).digest()
    seed = int.from_bytes(seed_digest[:8], "little", signed=False)
    generator = np.random.Generator(np.random.PCG64(seed))
    white = generator.standard_normal(length, dtype=np.float64)
    filtered = np.fft.irfft(np.fft.rfft(white) * np.sqrt(colored_power), n=length).astype(np.float64)
    filtered -= float(filtered.mean())
    if not np.isfinite(filtered).all() or float(np.sqrt(np.mean(filtered * filtered))) <= 0.0:
        raise ProtocolError("colored noise realization is invalid")
    return filtered, {
        "seed_material": seed_material.decode("utf-8"),
        "seed_sha256": seed_digest.hex(),
        "seed_uint64_little_endian": seed,
        "rng": "numpy.random.Generator(PCG64)",
        "target_rfft_bins": int(colored_power.size),
        "colored_power_mean": float(colored_power.mean()),
        "colored_power_sha256": hashlib.sha256(np.asarray(colored_power, dtype="<f8").tobytes()).hexdigest(),
        "noise_mean_after_centering": float(filtered.mean()),
    }


def active_rms(values: np.ndarray, mask: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    active = np.asarray(mask, dtype=bool)
    if array.shape != active.shape:
        raise ProtocolError("waveform and active mask shapes differ")
    if not np.any(active):
        return 0.0
    return float(np.sqrt(np.mean(array[active] * array[active], dtype=np.float64)))


def inactive_rms(values: np.ndarray, mask: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    inactive = ~np.asarray(mask, dtype=bool)
    if int(np.count_nonzero(inactive)) < 1600:
        return None
    return float(np.sqrt(np.mean(array[inactive] * array[inactive], dtype=np.float64)))


def envelope_10ms(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    count = array.size // 160
    if count < 1:
        return np.asarray([], dtype=np.float64)
    frames = array[: count * 160].reshape(count, 160)
    return np.sqrt(np.mean(frames * frames, axis=1, dtype=np.float64))


def normalized_lag_curve(left: np.ndarray, right: np.ndarray, *, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size == 0:
        raise ProtocolError("cross-correlation inputs have different or zero lengths")
    lags = np.arange(-max_lag, max_lag + 1, dtype=np.int64)
    values: list[float] = []
    x_energy = np.concatenate(([0.0], np.cumsum(x * x, dtype=np.float64)))
    y_energy = np.concatenate(([0.0], np.cumsum(y * y, dtype=np.float64)))
    fast_full: np.ndarray | None = None
    if x.size > 20_000 and max_lag > 100:
        # Computing every dot product independently is quadratic in the clip
        # length.  FFT correlation gives the same numerator, while the
        # overlap-specific norms below preserve the normalized definition.
        try:
            from scipy.signal import fftconvolve

            fast_full = np.asarray(fftconvolve(y, x[::-1], mode="full"), dtype=np.float64)
        except ImportError:  # pragma: no cover - scipy is a declared runtime dependency
            fast_full = None
    for lag in lags:
        shift = int(lag)
        if shift < 0:
            x_start, x_end, y_start, y_end = -shift, x.size, 0, y.size + shift
        elif shift > 0:
            x_start, x_end, y_start, y_end = 0, x.size - shift, shift, y.size
        else:
            x_start, x_end, y_start, y_end = 0, x.size, 0, y.size
        denominator = float(np.sqrt(max(0.0, (x_energy[x_end] - x_energy[x_start]) * (y_energy[y_end] - y_energy[y_start]))))
        numerator = float(fast_full[x.size - 1 + shift]) if fast_full is not None else float(np.dot(x[x_start:x_end], y[y_start:y_end]))
        values.append(float(numerator / denominator) if denominator > 0.0 else 0.0)
    return lags, np.asarray(values, dtype=np.float64)


def spectral_flux(values: np.ndarray) -> float:
    spectra, _ = stft(values)
    magnitude = np.abs(spectra)
    normalized = magnitude / np.maximum(np.sum(magnitude, axis=1, keepdims=True), config.EPSILON)
    if normalized.shape[0] < 2:
        return 0.0
    return float(np.mean(np.sqrt(np.sum(np.diff(normalized, axis=0) ** 2, axis=1)), dtype=np.float64))


def manipulation_metrics(original: np.ndarray, candidate: np.ndarray, mask: np.ndarray, *, candidate_name: str, source: str) -> dict[str, Any]:
    x = np.asarray(original, dtype=np.float64)
    y = np.asarray(candidate, dtype=np.float64)
    if x.shape != y.shape or x.shape != np.asarray(mask, dtype=bool).shape:
        raise ProtocolError(f"{candidate_name} manipulation shape differs")
    x_env = envelope_10ms(x)
    y_env = envelope_10ms(y)
    envelope_length = min(x_env.size, y_env.size)
    source_active = np.asarray(mask, dtype=bool)
    active_envelope = np.asarray(
        [bool(np.any(source_active[index * config.HOP : (index + 1) * config.HOP])) for index in range(envelope_length)],
        dtype=bool,
    )
    # Silence outside the shared source activity mask cannot create a false
    # temporal match.  The mask is defined on the source clock, so both
    # envelopes use the same 10 ms bins and no resampling or trimming occurs.
    x_envelope_active = np.where(active_envelope, x_env[:envelope_length], 0.0)
    y_envelope_active = np.where(active_envelope, y_env[:envelope_length], 0.0)
    envelope_lags, envelope_curve = normalized_lag_curve(x_envelope_active, y_envelope_active, max_lag=10)
    peak_index = int(np.argmax(envelope_curve))
    # The protocol's temporal tolerance is one 10 ms hop.  Keep the full
    # +/-100 ms curve in the machine-readable artifact.
    waveform_lags, waveform_curve = normalized_lag_curve(x, y, max_lag=min(1600, max(1, x.size - 1)))
    waveform_peak_index = int(np.argmax(waveform_curve))
    x_active = active_rms(x, source_active)
    y_active = active_rms(y, source_active)
    difference = x - y
    active_difference_rms = active_rms(difference, source_active)
    candidate_inactive = inactive_rms(y, source_active)
    original_inactive = inactive_rms(x, source_active)
    envelope_corr = float(np.corrcoef(x_envelope_active, y_envelope_active)[0, 1]) if envelope_length > 1 and np.std(x_envelope_active) > 0.0 and np.std(y_envelope_active) > 0.0 else None
    normalized_envelope_lag = int(envelope_lags[peak_index])
    return {
        "source": source,
        "candidate": candidate_name,
        "sample_count": int(x.size),
        "active_sample_count": int(np.count_nonzero(source_active)),
        "inactive_sample_count": int(np.count_nonzero(~source_active)),
        "original_active_rms": x_active,
        "candidate_active_rms": y_active,
        "active_rms_ratio_db": None if x_active <= 0.0 or y_active <= 0.0 else float(20.0 * np.log10(y_active / x_active)),
        "original_inactive_rms": original_inactive,
        "candidate_inactive_rms": candidate_inactive,
        "background_rms_change_db": None if original_inactive is None or candidate_inactive is None or original_inactive <= 0.0 or candidate_inactive <= 0.0 else float(20.0 * np.log10(candidate_inactive / original_inactive)),
        "background_rms_status": "NOT_ESTIMABLE" if candidate_inactive is None or original_inactive is None else "ESTIMABLE",
        "active_difference_rms": active_difference_rms,
        "active_difference_energy": float(np.mean(difference[source_active] ** 2, dtype=np.float64)) if np.any(source_active) else None,
        "waveform_rms_correlation": float(np.corrcoef(x[source_active], y[source_active])[0, 1]) if np.count_nonzero(source_active) > 1 and np.std(x[source_active]) > 0.0 and np.std(y[source_active]) > 0.0 else None,
        "envelope_10ms_correlation": envelope_corr,
        "envelope_lag_hops": normalized_envelope_lag,
        "envelope_lag_ms": float(normalized_envelope_lag * 10.0),
        "envelope_lag_within_one_hop": bool(abs(normalized_envelope_lag) <= 1),
        "envelope_cross_correlation_lags_ms": [float(item * 10.0) for item in envelope_lags],
        "envelope_cross_correlation": [float(item) for item in envelope_curve],
        "temporal_envelope_definition": "10 ms RMS envelopes; bins outside the shared source activity mask are set to zero; no trimming or resampling",
        "waveform_cross_correlation_lags_ms": [float(item * 1000.0 / config.SAMPLE_RATE) for item in waveform_lags],
        "waveform_cross_correlation": [float(item) for item in waveform_curve],
        "waveform_peak_lag_samples": int(waveform_lags[waveform_peak_index]),
        "waveform_peak_lag_ms": float(waveform_lags[waveform_peak_index] * 1000.0 / config.SAMPLE_RATE),
        "spectral_flux_original": spectral_flux(x),
        "spectral_flux_candidate": spectral_flux(y),
        "spectral_flux_change": float(spectral_flux(y) - spectral_flux(x)),
        "temporal_check": "PASS" if abs(normalized_envelope_lag) <= 1 else "MANIPULATION_FAILED",
        "human_listening_status": "NOT_ASSESSED",
    }


def _source_validity(values: np.ndarray, mask: np.ndarray) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if values.size < config.MIN_ACTIVE_SAMPLES:
        reasons.append(f"sample_count<{config.MIN_ACTIVE_SAMPLES}")
    if int(np.count_nonzero(mask)) < config.MIN_ACTIVE_SAMPLES:
        reasons.append(f"active_sample_count<{config.MIN_ACTIVE_SAMPLES}")
    if active_rms(values, mask) <= 0.0:
        reasons.append("active_rms=0")
    return ("AUDIO_VALID" if not reasons else "AUDIO_INVALID"), reasons


def _variant_record(path: Path, values: np.ndarray, *, source: str, sample_id: int, condition: str, g: float, unquantized: np.ndarray, original: np.ndarray, mask: np.ndarray, operation: Mapping[str, Any], manipulation: Mapping[str, Any] | None) -> dict[str, Any]:
    pcm = np.asarray(values, dtype=np.int16)
    record: dict[str, Any] = {
        "sample_id": int(sample_id),
        "source": source,
        "condition": condition,
        "path": str(path),
        "file_sha256": file_sha256(path),
        "pcm_sha256": pcm_hash(pcm),
        "sample_count": int(pcm.size),
        "sample_rate": config.SAMPLE_RATE,
        "channels": 1,
        "sample_width": 2,
        "headroom_g": float(g),
        "unquantized_peak": float(np.max(np.abs(unquantized))),
        "quantized_peak": float(np.max(np.abs(pcm.astype(np.float64) / 32768.0))),
        "quantization_max_abs_error": float(np.max(np.abs(unquantized * g - pcm.astype(np.float64) / 32768.0))) if condition != "ORIGINAL" else float(np.max(np.abs(original - pcm.astype(np.float64) / 32768.0))),
        "rms": float(np.sqrt(np.mean((pcm.astype(np.float64) / 32768.0) ** 2, dtype=np.float64))),
        "active_rms": active_rms(pcm.astype(np.float64) / 32768.0, mask),
        "operation": dict(operation),
        "manipulation": None if manipulation is None else dict(manipulation),
        "status": "COMPLETE",
    }
    return record


def _write_npy(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{__import__('os').getpid()}.tmp")
    np.save(temporary, np.asarray(value), allow_pickle=False)
    produced = Path(str(temporary) + ".npy") if not temporary.name.endswith(".npy") else temporary
    produced.replace(path)
    return file_sha256(path)


def construct_source(sample_id: int, source: str, original_pcm: np.ndarray, natural_power: np.ndarray, output_dir: Path, *, common_g: float | None = None) -> dict[str, Any]:
    if source not in config.SOURCES:
        raise ProtocolError(f"unknown audio source: {source}")
    x = pcm16_to_float(np.asarray(original_pcm, dtype=np.int16))
    spectrum, starts = stft(x)
    identity = istft(spectrum, starts, x.size)
    identity_error = float(np.max(np.abs(identity - x)))
    if identity_error >= 1e-8:
        raise ProtocolError(f"float64 identity reconstruction error is too large for {sample_id}/{source}: {identity_error}")
    frame_rms, active_frames, mask = frame_rms_and_activity(x)
    validity, invalid_reasons = _source_validity(x, mask)
    if validity != "AUDIO_VALID":
        raise ProtocolError(f"{sample_id}/{source} is invalid for P1: {invalid_reasons}")

    noise, noise_meta = noise_realization(x.size, natural_power, sample_id=sample_id, source=source)
    noise_rms = active_rms(noise, mask)
    signal_rms = active_rms(x, mask)
    if noise_rms <= 0.0 or signal_rms <= 0.0:
        raise ProtocolError(f"{sample_id}/{source} cannot construct NOISE20")
    beta = signal_rms / (10.0 * noise_rms)
    noise_mix = x + beta * noise

    natural_power_target = np.asarray(natural_power, dtype=np.float64)
    magnitude_sq = np.abs(spectrum) ** 2
    gain = np.clip(np.sqrt(np.maximum(magnitude_sq - natural_power_target[None, :], 0.0) / (magnitude_sq + 1e-12)), 0.25, 1.0)
    denoised = istft(spectrum * gain, starts, x.size)
    denoised_active_rms = active_rms(denoised, mask)
    if denoised_active_rms <= 0.0:
        raise ProtocolError(f"{sample_id}/{source} DENOISE active RMS is zero")
    denoise_scale = signal_rms / denoised_active_rms
    denoised *= denoise_scale

    unquantized = {"NATURAL": x, "NOISE20": noise_mix, "DENOISE": denoised}
    max_abs = max(float(np.max(np.abs(value))) for value in unquantized.values())
    own_g = min(1.0, 0.98 / max_abs) if max_abs > 0.0 else 1.0
    g = own_g if common_g is None else float(common_g)
    if not 0.0 < g <= 1.0:
        raise ProtocolError(f"invalid common headroom factor for {sample_id}/{source}: {g}")
    conditions = {
        "ORIGINAL": x,
        "A0": x,
        "GAIN": x * (10.0 ** (-6.0 / 20.0)),
        "NOISE": noise_mix,
        "DENOISE": denoised,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    power_path = output_dir / "natural_low_energy_power.npy"
    _write_npy(power_path, natural_power)
    frame_rms_path = output_dir / "frame_rms.npy"
    activity_mask_path = output_dir / "activity_mask.npy"
    frame_rms_hash = _write_npy(frame_rms_path, frame_rms)
    activity_mask_hash = _write_npy(activity_mask_path, mask.astype(np.uint8))
    a0_pcm = quantize_pcm16(g * x)
    records: list[dict[str, Any]] = []
    for condition, unscaled in conditions.items():
        pcm = quantize_pcm16(unscaled if condition == "ORIGINAL" else g * unscaled)
        path = output_dir / f"{condition}.wav"
        write_pcm16_wav(path, pcm)
        if pcm.size != original_pcm.size:
            raise ProtocolError(f"{sample_id}/{source}/{condition} changed sample count")
        manipulation = None
        if condition in {"GAIN", "NOISE", "DENOISE"}:
            manipulation = manipulation_metrics(x, pcm.astype(np.float64) / 32768.0, mask, candidate_name=condition, source=source)
        operation: dict[str, Any] = {
            "kind": {"ORIGINAL": "identity_quantization", "A0": "shared_headroom_baseline", "GAIN": "fixed_minus_6db", "NOISE": "colored_noise_snr20", "DENOISE": "fixed_spectral_subtraction"}[condition],
            "processing_dtype": "float64",
            "quantization": "numpy.rint(x*32768), ties-to-even, int16 range checked, no clipping",
            "common_headroom_applied": condition != "ORIGINAL",
        }
        if condition == "GAIN":
            a0_values = a0_pcm.astype(np.float64) / 32768.0
            candidate_values = pcm.astype(np.float64) / 32768.0
            measured_gain = 20.0 * np.log10(active_rms(candidate_values, mask) / active_rms(a0_values, mask))
            operation.update({"gain_db": -6.0, "gain_linear": 10.0 ** (-6.0 / 20.0), "measured_gain_db": float(measured_gain), "measurement": "20log10(RMS_M(Q(g*x*gain))/RMS_M(Q(g*x)))"})
        if condition == "NOISE":
            a0_values = a0_pcm.astype(np.float64) / 32768.0
            candidate_values = pcm.astype(np.float64) / 32768.0
            quantized_noise = candidate_values - a0_values
            measured_snr = 10.0 * np.log10(np.sum(a0_values[mask] ** 2) / np.sum(quantized_noise[mask] ** 2))
            operation.update({"beta": float(beta), "target_snr_db": 20.0, "measured_snr_db": float(measured_snr), "measurement": "10log10(sum_M(Q(g*x)^2)/sum_M((Q(g*(x+beta*e))-Q(g*x))^2))", "noise": noise_meta})
        if condition == "DENOISE":
            operation.update({"denoise_scale": float(denoise_scale), "gain_floor": 0.25, "gain_ceiling": 1.0, "pnoise_source": "natural_low_energy_power"})
        records.append(_variant_record(path, pcm, source=source, sample_id=sample_id, condition=condition, g=g, unquantized=unscaled, original=x, mask=mask, operation=operation, manipulation=manipulation))

    noise_record = next(item for item in records if item["condition"] == "NOISE")
    measured_snr = float(noise_record["operation"]["measured_snr_db"])
    gain_record = next(item for item in records if item["condition"] == "GAIN")
    gain_db = float(gain_record["operation"]["measured_gain_db"])
    if abs(measured_snr - 20.0) > 0.1:
        raise ProtocolError(f"{sample_id}/{source} NOISE20 SNR check failed: {measured_snr}")
    if abs(gain_db + 6.0) > 0.01:
        raise ProtocolError(f"{sample_id}/{source} GAIN check failed: {gain_db}")
    return {
        "sample_id": int(sample_id),
        "source": source,
        "status": "COMPLETE",
        "sample_count": int(x.size),
        "sample_rate": config.SAMPLE_RATE,
        "original_pcm_sha256": pcm_hash(np.asarray(original_pcm, dtype=np.int16)),
        "active_sample_count": int(np.count_nonzero(mask)),
        "inactive_sample_count": int(np.count_nonzero(~mask)),
        "validity": validity,
        "invalid_reasons": invalid_reasons,
        "activity": {
            "frame_count": int(frame_rms.size),
            "active_frame_count": int(np.count_nonzero(active_frames)),
            "peak_frame_rms": float(frame_rms.max()),
            "threshold": float(max(float(frame_rms.max()) * 0.1, 10.0 ** (-50.0 / 20.0))),
            "mask_rule": "union of centered active 512-sample frames, clipped to source clock",
        },
        "activity_artifacts": {
            "frame_rms": {"path": str(frame_rms_path), "sha256": frame_rms_hash},
            "activity_mask": {"path": str(activity_mask_path), "sha256": activity_mask_hash},
        },
        "stft": {
            "sample_rate": config.SAMPLE_RATE,
            "n_fft": config.NFFT,
            "window_length": config.WINDOW,
            "hop": config.HOP,
            "window": "periodic Hann",
            "center": True,
            "pad_mode": "reflect",
            "pad": config.PAD,
            "forward_normalization": "none",
            "inverse_normalization": "1/512 via numpy.irfft",
            "identity_max_abs_error": identity_error,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "noise_power": {
            "path": str(power_path),
            "sha256": file_sha256(power_path),
            "source": "natural N low-energy frames for this ID",
            "details": read_json(power_path.with_suffix(".json")) if power_path.with_suffix(".json").is_file() else None,
        },
        "common_headroom": {
            "g": float(g),
            "max_abs_unquantized": max_abs,
            "all_sources_and_variants": True,
        },
        "conditions": records,
        "status_checks": {
            "identity_float64_pass": identity_error < 1e-8,
            "gain_minus6db_pass": abs(gain_db + 6.0) <= 0.01,
            "noise20_snr_pass": abs(measured_snr - 20.0) <= 0.1,
            "sample_counts_unchanged": all(int(item["sample_count"]) == int(x.size) for item in records),
            "no_variant_temporal_failure": all(item["manipulation"] is None or item["manipulation"]["temporal_check"] == "PASS" for item in records),
        },
    }


def _binding_path(value: Mapping[str, Any], key: str) -> Path:
    path_value = value.get(key)
    if isinstance(path_value, Mapping):
        path_value = path_value.get("path")
    if not isinstance(path_value, str):
        raise ProtocolError(f"audio binding field is missing: {key}")
    target = Path(path_value)
    if not target.is_absolute():
        target = config.REPO / target
    return target.resolve()


def audio_stage(paths: config.RunPaths, assets: Mapping[str, Any]) -> dict[str, Any]:
    """Decode all three source clocks and materialise P1 audio artifacts."""

    root = paths.audio
    manifest_path = root / "manifest.json"
    current_code_hash = file_sha256(Path(__file__))
    if manifest_path.is_file():
        try:
            cached = read_self_hashed_json(manifest_path)
            if cached.get("status") == "COMPLETE" and cached.get("audio_code_hash") == current_code_hash and int(cached.get("record_count", -1)) == len(config.SAMPLE_IDS) * len(config.SOURCES) * len(config.AUDIO_CONDITIONS):
                return cached
        except ProtocolError:
            pass
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for item in assets.get("records", []):
        sample_id = int(item["sample_id"])
        by_source = item.get("sources")
        if not isinstance(by_source, Mapping):
            raise ProtocolError(f"assets has no source bindings for {sample_id}")
        source_inputs: dict[str, np.ndarray] = {}
        for source in config.SOURCES:
            source_binding = by_source.get(source)
            if not isinstance(source_binding, Mapping):
                raise ProtocolError(f"assets has no {source} binding for {sample_id}")
            media = _binding_path(source_binding, "audio_media") if source_binding.get("audio_media") is not None else _binding_path(source_binding, "media")
            if not media.is_file():
                raise ProtocolError(f"audio source media is missing: {media}")
            values = decode_media_pcm16(media)
            expected = source_binding.get("original_pcm_sha256")
            actual = pcm_hash(values)
            if expected and actual != str(expected):
                raise ProtocolError(f"decoded PCM hash mismatch for {sample_id}/{source}: {actual} != {expected}")
            source_inputs[source] = values
        natural_power, natural_meta = low_energy_noise_power(pcm16_to_float(source_inputs["N"]))
        id_dir = root / str(sample_id)
        id_dir.mkdir(parents=True, exist_ok=True)
        np.save(id_dir / "natural_low_energy_power.npy", natural_power, allow_pickle=False)
        write_json(id_dir / "natural_low_energy_power.json", natural_meta)
        unquantized: dict[str, np.ndarray] = {}
        # Constructing independently first is required by the common g rule.
        for source in config.SOURCES:
            x = pcm16_to_float(source_inputs[source])
            spectrum, starts = stft(x)
            noise, _ = noise_realization(x.size, natural_power, sample_id=sample_id, source=source)
            mask = frame_rms_and_activity(x)[2]
            beta = active_rms(x, mask) / (10.0 * active_rms(noise, mask))
            noise_mix = x + beta * noise
            gain = np.clip(np.sqrt(np.maximum(np.abs(spectrum) ** 2 - natural_power[None, :], 0.0) / (np.abs(spectrum) ** 2 + 1e-12)), 0.25, 1.0)
            denoised = istft(spectrum * gain, starts, x.size)
            denoised *= active_rms(x, mask) / active_rms(denoised, mask)
            unquantized[f"{source}:NATURAL"] = x
            unquantized[f"{source}:NOISE20"] = noise_mix
            unquantized[f"{source}:DENOISE"] = denoised
        common_max = max(float(np.max(np.abs(value))) for value in unquantized.values())
        common_g = min(1.0, 0.98 / common_max) if common_max > 0.0 else 1.0
        for source in config.SOURCES:
            source_dir = id_dir / source
            # construct_source recomputes the fixed operations and accepts the
            # precomputed natural spectrum.  The assertion below protects the
            # cross-source headroom from accidental per-source scaling.
            source_result = construct_source(sample_id, source, source_inputs[source], natural_power, source_dir, common_g=common_g)
            if abs(float(source_result["common_headroom"]["g"]) - common_g) > 1e-12:
                # Rewrite the source using the common value rather than allow
                # a silent per-source headroom.  This branch is expected only
                # when a future numerical backend changes an operation.
                raise ProtocolError(f"common headroom mismatch for {sample_id}/{source}")
            records.extend(source_result["conditions"])
            write_self_hashed_json(source_dir / "manifest.json", source_result)
    expected_count = len(config.SAMPLE_IDS) * len(config.SOURCES) * len(config.AUDIO_CONDITIONS)
    if len(records) != expected_count:
        raise ProtocolError(f"audio record count is {len(records)}/{expected_count}")
    result = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "stage": "01_audio",
        "status": "COMPLETE",
        "sample_ids": list(config.SAMPLE_IDS),
        "sources": list(config.SOURCES),
        "conditions": list(config.AUDIO_CONDITIONS),
        "record_count": len(records),
        "records": records,
        "audio_code_hash": current_code_hash,
        "algorithm": {
            "audio_clock": "16 kHz mono PCM16; decoded once per source; all variants exact source length",
            "headroom": "one g per ID across N/T/R and NATURAL/NOISE20/DENOISE unquantized waveforms",
            "no_new_tts_or_cloud_calls": True,
        },
    }
    return write_self_hashed_json(manifest_path, result)
