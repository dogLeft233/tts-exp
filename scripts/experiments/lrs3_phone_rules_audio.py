"""Deterministic, time-preserving waveform rules for the LRS3 experiment."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

PCM_FLOAT_MIN = -1.0
PCM_FLOAT_MAX = 32767.0 / 32768.0


def _as_pcm16(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype != np.int16 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional int16 array")
    return array


def _as_float_waveform(values: np.ndarray, name: str = "waveform") -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite non-empty one-dimensional array")
    return array


def _pcm_to_float(values: np.ndarray) -> np.ndarray:
    return _as_pcm16(values, "PCM input").astype(np.float64) / 32768.0


def _float_to_pcm(values: np.ndarray) -> np.ndarray:
    array = _as_float_waveform(values, "float waveform")
    if np.any(array < PCM_FLOAT_MIN - 1e-12) or np.any(array > PCM_FLOAT_MAX + 1e-12):
        raise ValueError("float waveform is outside the representable PCM16 range")
    return np.rint(array * 32768.0).astype("<i2")


def _raised_cosine_up(length: int) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    if length == 1:
        return np.asarray([1.0], dtype=np.float64)
    phase = np.arange(length, dtype=np.float64) / float(length - 1)
    return 0.5 - 0.5 * np.cos(np.pi * phase)


def _raised_cosine_down(length: int) -> np.ndarray:
    return _raised_cosine_up(length)[::-1]


def build_edit_mask(
    sample_count: int,
    tokens: Sequence[Mapping[str, Any]],
    *,
    sample_rate: int = 16_000,
    edge_guard_s: float = 0.010,
    taper_s: float = 0.005,
) -> np.ndarray:
    """Build a speech-only mask with protected token boundaries.

    Every interval is processed independently.  A short phone that cannot
    contain both guards and tapers receives no editable samples; neighbouring
    phones are never merged to reclaim that space.
    """

    if int(sample_count) <= 0 or int(sample_rate) <= 0:
        raise ValueError("sample_count and sample_rate must be positive")
    if edge_guard_s < 0.0 or taper_s < 0.0:
        raise ValueError("edge_guard_s and taper_s must be non-negative")
    mask = np.zeros(int(sample_count), dtype=np.float64)
    guard = round(edge_guard_s * sample_rate)
    taper = round(taper_s * sample_rate)
    for token in tokens:
        if not bool(token.get("speech", token.get("is_speech", not token.get("silence", False)))):
            continue
        start = float(token["start_s"])
        end = float(token["end_s"])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("token spans must be finite and positive")
        begin = max(0, min(sample_count, math.ceil(start * sample_rate)))
        finish = max(0, min(sample_count, math.floor(end * sample_rate)))
        if finish <= begin:
            continue
        left_zero_end = min(finish, begin + guard)
        right_zero_start = max(begin, finish - guard)
        active_start = left_zero_end + taper
        active_end = right_zero_start - taper
        if active_end <= active_start:
            continue
        mask[active_start:active_end] = np.maximum(mask[active_start:active_end], 1.0)
        if taper:
            left = _raised_cosine_up(taper)
            right = _raised_cosine_down(taper)
            left_start = left_zero_end
            left_stop = min(active_start, sample_count)
            right_start = max(active_end, 0)
            right_stop = min(right_start + taper, sample_count)
            if left_stop > left_start:
                mask[left_start:left_stop] = np.maximum(
                    mask[left_start:left_stop], left[: left_stop - left_start]
                )
            if right_stop > right_start:
                mask[right_start:right_stop] = np.maximum(
                    mask[right_start:right_stop], right[: right_stop - right_start]
                )
    return np.clip(mask, 0.0, 1.0)


def _spectral_gain_db(frequencies: np.ndarray) -> np.ndarray:
    """Fixed +2 dB mid-band emphasis with cosine transitions."""

    freq = np.asarray(frequencies, dtype=np.float64)
    gain = np.zeros_like(freq)
    mid = (freq >= 1000.0) & (freq <= 4000.0)
    gain[mid] = 2.0
    rising = (freq > 300.0) & (freq < 1000.0)
    phase = (freq[rising] - 300.0) / 700.0
    gain[rising] = 2.0 * (0.5 - 0.5 * np.cos(np.pi * phase))
    falling = (freq > 4000.0) & (freq < 6000.0)
    phase = (freq[falling] - 4000.0) / 2000.0
    gain[falling] = 2.0 * (0.5 + 0.5 * np.cos(np.pi * phase))
    return gain


def _rfft_energy(spectrum: Any, n_fft: int) -> Any:
    """One-sided rFFT energy with DC/Nyquist weight one and others two."""

    import torch

    weights = torch.full((spectrum.shape[0],), 2.0, dtype=spectrum.real.dtype, device=spectrum.device)
    weights[0] = 1.0
    if n_fft % 2 == 0 and weights.numel() > 1:
        weights[-1] = 1.0
    return (spectrum.abs().square() * weights[:, None]).sum(dim=0)


def spectral_emphasis(
    waveform: np.ndarray,
    *,
    sample_rate: int = 16_000,
    n_fft: int = 512,
    win_length: int = 512,
    hop_length: int = 128,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply phase-preserving fixed spectral emphasis using torch STFT/ISTFT."""

    x = _as_float_waveform(waveform)
    if sample_rate <= 0 or n_fft <= 0 or win_length <= 0 or hop_length <= 0:
        raise ValueError("STFT parameters must be positive")
    if win_length != n_fft or win_length > n_fft:
        raise ValueError("the frozen protocol requires win_length == n_fft")
    if x.size <= n_fft // 2:
        raise ValueError("waveform is too short for reflect-centered STFT")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only in minimal envs
        raise RuntimeError("torch is required for the frozen STFT implementation") from exc

    tensor = torch.from_numpy(x).to(dtype=torch.float64)
    window = torch.hann_window(win_length, periodic=True, dtype=torch.float64)
    spectrum = torch.stft(
        tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    gain_db = torch.from_numpy(_spectral_gain_db(frequencies)).to(dtype=torch.float64)
    gain = torch.pow(torch.tensor(10.0, dtype=torch.float64), gain_db / 20.0)[:, None]
    modified = spectrum * gain
    original_energy = _rfft_energy(spectrum, n_fft)
    modified_energy = _rfft_energy(modified, n_fft)
    scale = torch.ones_like(original_energy)
    nonzero = (original_energy > 1e-12) & (modified_energy > 1e-12)
    scale[nonzero] = torch.sqrt(original_energy[nonzero] / modified_energy[nonzero])
    modified = modified * scale[None, :]
    output = torch.istft(
        modified,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        length=x.size,
    )
    result = output.detach().cpu().numpy().astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("spectral emphasis generated non-finite samples")
    return result, {
        "algorithm": "torch_stft_phase_preserving_spectral_emphasis",
        "sample_rate": int(sample_rate),
        "n_fft": int(n_fft),
        "win_length": int(win_length),
        "hop_length": int(hop_length),
        "window": "periodic_hann",
        "center": True,
        "pad_mode": "reflect",
        "gain_db": "<=300:0;300-1000:cosine_to_2;1000-4000:2;4000-6000:cosine_to_0;>=6000:0",
        "phase_policy": "unchanged",
    }


def _reflect_convolve(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    half = kernel.size // 2
    if values.size <= half:
        raise ValueError("waveform is too short for reflect smoothing")
    padded = np.pad(values, (half, half), mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def upward_compression(
    waveform: np.ndarray,
    edit_mask: np.ndarray,
    *,
    sample_rate: int = 16_000,
    window_s: float = 0.020,
    threshold_quantile: float = 0.75,
    ratio: float = 1.5,
    max_gain_db: float = 3.0,
    smoothing_length: int = 801,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the fixed weak upward-compression rule."""

    x = _as_float_waveform(waveform)
    mask = np.asarray(edit_mask, dtype=np.float64)
    if mask.shape != x.shape or not np.all(np.isfinite(mask)) or np.any(mask < 0.0) or np.any(mask > 1.0):
        raise ValueError("edit_mask must be finite, [0,1], and match waveform length")
    if not 0.0 < threshold_quantile < 1.0 or ratio <= 1.0 or max_gain_db < 0.0:
        raise ValueError("invalid upward-compression parameters")
    if smoothing_length <= 0 or smoothing_length % 2 != 1:
        raise ValueError("smoothing_length must be a positive odd integer")
    window_samples = max(1, round(window_s * sample_rate))
    left = window_samples // 2
    right = window_samples - left
    if x.size <= max(left, smoothing_length // 2):
        raise ValueError("waveform is too short for centered RMS/smoothing")
    square = np.pad(x * x, (left, right - 1), mode="reflect")
    rms = np.sqrt(np.convolve(square, np.ones(window_samples) / window_samples, mode="valid"))
    speech = mask > 0.0
    if not np.any(speech) or float(np.max(rms[speech])) <= 1e-12:
        raise ValueError("speech region is silent; compression threshold is undefined")
    speech_rms = np.maximum(rms[speech], 10.0 ** (-80.0 / 20.0))
    threshold = float(np.quantile(speech_rms, threshold_quantile))
    rms_db = 20.0 * np.log10(np.maximum(rms, 10.0 ** (-80.0 / 20.0)))
    threshold_db = 20.0 * math.log10(max(threshold, 10.0 ** (-80.0 / 20.0)))
    gain_db = np.clip((1.0 - 1.0 / ratio) * (threshold_db - rms_db), 0.0, max_gain_db)
    smoothing_window = np.hanning(smoothing_length).astype(np.float64)
    smoothing_window /= max(float(smoothing_window.sum()), 1e-12)
    gain_db = _reflect_convolve(gain_db, smoothing_window)
    gain = np.power(10.0, gain_db / 20.0)
    output = x * gain
    return output, {
        "algorithm": "upward_compression",
        "window_samples": int(window_samples),
        "threshold_quantile": float(threshold_quantile),
        "threshold_rms": threshold,
        "threshold_db": threshold_db,
        "ratio": float(ratio),
        "max_gain_db": float(max_gain_db),
        "smoothing_length": int(smoothing_length),
    }


def _safe_residual_alpha(source: np.ndarray, candidate: np.ndarray) -> float:
    source = _as_float_waveform(source, "source")
    candidate = _as_float_waveform(candidate, "candidate")
    if source.shape != candidate.shape:
        raise ValueError("source and candidate must have equal shape")
    delta = candidate - source
    alpha = 1.0
    positive = delta > 0.0
    negative = delta < 0.0
    if np.any(positive):
        alpha = min(alpha, float(np.min((PCM_FLOAT_MAX - source[positive]) / delta[positive])))
    if np.any(negative):
        alpha = min(alpha, float(np.min((PCM_FLOAT_MIN - source[negative]) / delta[negative])))
    if not math.isfinite(alpha):
        raise ValueError("residual alpha is non-finite")
    return max(0.0, min(1.0, alpha))


def _quantize_masked(source_pcm: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    source = _as_pcm16(source_pcm, "source PCM")
    x = _pcm_to_float(source)
    y = _as_float_waveform(candidate, "candidate")
    m = np.asarray(mask, dtype=np.float64)
    if y.shape != x.shape or m.shape != x.shape:
        raise ValueError("candidate/mask shape mismatch")
    z = x + m * (y - x)
    alpha = _safe_residual_alpha(x, z)
    rendered = x + alpha * (z - x)
    # Preserve untouched PCM values exactly, including values that would move
    # by a float round-trip at a zero mask.
    output = source.copy()
    editable = m > 0.0
    if np.any(editable):
        output[editable] = _float_to_pcm(rendered[editable])
    return output, float(alpha)


def render_rule_arm(
    source_pcm: np.ndarray,
    tokens: Sequence[Mapping[str, Any]],
    arm: str,
    *,
    sample_rate: int = 16_000,
    edge_guard_s: float = 0.010,
    taper_s: float = 0.005,
    n_fft: int = 512,
    win_length: int = 512,
    hop_length: int = 128,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render identity or one of the frozen rule arms."""

    source = _as_pcm16(source_pcm, "source PCM")
    if arm == "identity":
        return source.copy(), {
            "arm": arm,
            "sample_count": int(source.size),
            "alpha": 0.0,
            "mask_sha256": None,
            "changed_sample_count": 0,
            "effective_edit": False,
        }
    if arm not in {"spectral_drc", "spectral_only", "drc_only"}:
        raise ValueError(f"unknown rule arm: {arm}")
    mask = build_edit_mask(
        source.size,
        tokens,
        sample_rate=sample_rate,
        edge_guard_s=edge_guard_s,
        taper_s=taper_s,
    )
    x = _pcm_to_float(source)
    rule = x
    audio_meta: dict[str, Any] = {}
    if arm in {"spectral_drc", "spectral_only"}:
        rule, spectral_meta = spectral_emphasis(
            rule,
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
        audio_meta["spectral"] = spectral_meta
    if arm in {"spectral_drc", "drc_only"}:
        rule, drc_meta = upward_compression(rule, mask, sample_rate=sample_rate)
        audio_meta["drc"] = drc_meta
    output, alpha = _quantize_masked(source, rule, mask)
    mask_bytes = np.asarray(mask, dtype="<f8").tobytes()
    import hashlib

    changed = int(np.count_nonzero(output != source))
    speech_source = x[mask > 0.0]
    speech_output = output.astype(np.float64)[mask > 0.0] / 32768.0
    source_rms = float(np.sqrt(np.mean(speech_source**2))) if speech_source.size else 0.0
    output_rms = float(np.sqrt(np.mean(speech_output**2))) if speech_output.size else 0.0
    return output, {
        "arm": arm,
        "sample_count": int(source.size),
        "alpha": alpha,
        "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
        "mask_edit_fraction": float(np.mean(mask > 0.0)),
        "changed_sample_count": changed,
        "effective_edit": bool(changed > 0 and alpha > 0.0),
        "input_peak": float(np.max(np.abs(x))),
        "output_peak": float(np.max(np.abs(output.astype(np.float64) / 32768.0))),
        "speech_rms_input": source_rms,
        "speech_rms_output": output_rms,
        "audio_rules": audio_meta,
    }


def make_gain_control(
    source_pcm: np.ndarray,
    edit_mask: np.ndarray,
    target_energy: float,
    *,
    tolerance_db: float = 0.05,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Match a target speech energy with one constant gain, without limiting."""

    source = _as_pcm16(source_pcm, "source PCM")
    x = _pcm_to_float(source)
    mask = np.asarray(edit_mask, dtype=np.float64)
    if mask.shape != x.shape or not np.all(np.isfinite(mask)) or np.any(mask < 0.0) or np.any(mask > 1.0):
        raise ValueError("edit_mask must match source and lie in [0,1]")
    target = float(target_energy)
    if not math.isfinite(target) or target < 0.0:
        raise ValueError("target_energy must be finite and non-negative")
    speech = mask > 0.0
    if not np.any(speech):
        return source.copy(), {
            "arm": "gain_control",
            "valid": False,
            "status": "CONTROL_INVALID",
            "reason": "EMPTY_SPEECH_MASK",
            "g": None,
            "target_energy": target,
        }
    mx = mask[speech] * x[speech]
    speech_x = x[speech]
    a = float(np.sum(mx * mx))
    b = float(2.0 * np.sum(speech_x * mx))
    c = float(np.sum(speech_x * speech_x) - target)
    roots: list[float] = []
    if abs(a) <= 1e-15:
        if abs(b) > 1e-15:
            roots.append(-c / b)
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant >= -1e-12:
            discriminant = max(0.0, discriminant)
            roots.extend([(-b - math.sqrt(discriminant)) / (2.0 * a), (-b + math.sqrt(discriminant)) / (2.0 * a)])
    # The quadratic is written in delta=(g-1), while the contract records the
    # non-negative absolute gain g.
    candidates = [root + 1.0 for root in roots if math.isfinite(root + 1.0) and root + 1.0 >= 0.0]
    if not candidates:
        return source.copy(), {
            "arm": "gain_control",
            "valid": False,
            "status": "CONTROL_INVALID",
            "reason": "NO_NONNEGATIVE_ROOT",
            "g": None,
            "target_energy": target,
        }
    gain = min(candidates, key=lambda value: abs(value - 1.0))
    candidate = x + mask * (gain - 1.0) * x
    if np.any(candidate < PCM_FLOAT_MIN - 1e-12) or np.any(candidate > PCM_FLOAT_MAX + 1e-12):
        return source.copy(), {
            "arm": "gain_control",
            "valid": False,
            "status": "CONTROL_INVALID",
            "reason": "PEAK_WOULD_OVERFLOW_PCM",
            "g": gain,
            "target_energy": target,
        }
    output = source.copy()
    editable = mask > 0.0
    output[editable] = _float_to_pcm(candidate[editable])
    actual = float(np.sum((output[speech].astype(np.float64) / 32768.0) ** 2))
    error_db = 10.0 * math.log10(max(actual, 1e-30) / max(target, 1e-30)) if target > 0 else 0.0
    valid = abs(error_db) <= tolerance_db
    return output, {
        "arm": "gain_control",
        "valid": valid,
        "status": "OK" if valid else "CONTROL_INVALID",
        "reason": None if valid else "QUANTIZED_ENERGY_OUTSIDE_TOLERANCE",
        "g": gain,
        "target_energy": target,
        "quantized_energy": actual,
        "energy_error_db": error_db,
        "tolerance_db": tolerance_db,
    }


def validate_waveform(
    source_pcm: np.ndarray,
    output_pcm: np.ndarray,
    edit_mask: np.ndarray,
    *,
    sample_rate: int = 16_000,
) -> dict[str, Any]:
    """Validate the hard waveform contract independently of scientific scores."""

    source = _as_pcm16(source_pcm, "source PCM")
    output = _as_pcm16(output_pcm, "output PCM")
    mask = np.asarray(edit_mask, dtype=np.float64)
    if source.shape != output.shape or mask.shape != source.shape:
        raise ValueError("source, output, and edit_mask must have equal shape")
    zero = mask <= 0.0
    outside_equal = bool(np.array_equal(source[zero], output[zero]))
    finite = bool(np.all(np.isfinite(output.astype(np.float64))))
    return {
        "sample_rate": int(sample_rate),
        "sample_count": int(source.size),
        "same_length": True,
        "mask_zero_pcm_equal": outside_equal,
        "finite": finite,
        "pcm_range_valid": bool(np.all(output >= -32768) and np.all(output <= 32767)),
        "changed_sample_count": int(np.count_nonzero(source != output)),
        "pass": bool(outside_equal and finite),
    }


__all__ = [
    "build_edit_mask",
    "make_gain_control",
    "render_rule_arm",
    "spectral_emphasis",
    "upward_compression",
    "validate_waveform",
]
