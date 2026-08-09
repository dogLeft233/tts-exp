#!/usr/bin/env python3
"""Deterministic acoustic controls for TTS/natural timing experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import librosa
import numpy as np

from pnp_alignment_warp import (
    Span,
    WarpResult,
    as_spans,
    match_token_spans,
    phone_local_warp,
)


@dataclass(frozen=True)
class ControlResult:
    """Waveform control and construction metadata."""

    audio: np.ndarray
    metadata: dict[str, Any]


def _exact_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if target_length < 0:
        raise ValueError("target_length must be non-negative")
    if values.size == target_length:
        return values.copy()
    if values.size > target_length:
        return values[:target_length].copy()
    return np.pad(values, (0, target_length - values.size)).astype(np.float32)


def _sample_bounds(start_s: float, end_s: float, sample_rate: int, length: int) -> tuple[int, int]:
    start = int(np.rint(start_s * sample_rate))
    end = int(np.rint(end_s * sample_rate))
    start = max(0, min(length, start))
    end = max(start, min(length, end))
    return start, end


def _phase_vocoder_resample(
    audio: np.ndarray,
    target_length: int,
    *,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> tuple[np.ndarray, bool, str | None]:
    """Change duration with phase vocoder while retaining approximate F0."""
    source = np.asarray(audio, dtype=np.float32).reshape(-1)
    if target_length < 0:
        raise ValueError("target_length must be non-negative")
    if target_length == 0:
        return np.empty(0, dtype=np.float32), False, None
    if source.size == 0:
        return np.empty(0, dtype=np.float32), True, "empty_source"
    if source.size == target_length:
        return source.copy(), False, None
    if source.size < max(16, hop_length // 2):
        return _exact_length(source, target_length), True, "short_segment_exact_fit"
    rate = source.size / target_length
    try:
        stretched = librosa.effects.time_stretch(
            source.astype(np.float64), rate=float(rate),
            n_fft=n_fft, hop_length=hop_length,
        )
        return _exact_length(stretched, target_length), False, None
    except (ValueError, RuntimeError, librosa.util.exceptions.ParameterError) as exc:
        return _exact_length(source, target_length), True, f"phase_vocoder_failed:{type(exc).__name__}"


def pitch_preserving_resample(audio: np.ndarray, target_length: int) -> ControlResult:
    """Return an exact-length, phase-vocoder duration transform."""
    output, fallback, reason = _phase_vocoder_resample(audio, target_length)
    return ControlResult(
        audio=output,
        metadata={
            "transform": "phase_vocoder_time_stretch",
            "source_sample_count": int(np.asarray(audio).size),
            "target_sample_count": int(target_length),
            "fallback": bool(fallback),
            "fallback_reason": reason,
            "pitch_preserving_approximation": True,
        },
    )


def pitch_preserving_phone_local_warp(
    natural_audio: np.ndarray,
    tts_audio: np.ndarray,
    sample_rate: int,
    natural_spans: Iterable[Span | Mapping[str, Any] | Any],
    tts_spans: Iterable[Span | Mapping[str, Any] | Any],
    *,
    min_confidence: float = 0.8,
) -> WarpResult:
    """Copy TTS phone segments with pitch-preserving duration conversion."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    natural = np.asarray(natural_audio, dtype=np.float32).reshape(-1)
    tts = np.asarray(tts_audio, dtype=np.float32).reshape(-1)
    if natural.size == 0 or tts.size == 0:
        raise ValueError("natural_audio and tts_audio must not be empty")

    matches, stats = match_token_spans(
        natural_spans, tts_spans, min_confidence=min_confidence
    )
    output = natural.copy()
    fallbacks: list[dict[str, Any]] = []
    used_spans: list[dict[str, Any]] = []
    for match in matches:
        nat_start, nat_end = _sample_bounds(
            match.natural_start_s, match.natural_end_s, sample_rate, len(natural)
        )
        tts_start, tts_end = _sample_bounds(
            match.tts_start_s, match.tts_end_s, sample_rate, len(tts)
        )
        target_length = nat_end - nat_start
        source = tts[tts_start:tts_end]
        if target_length <= 0 or source.size <= 0:
            fallbacks.append({"label": match.label, "reason": "empty_segment"})
            continue
        transformed = pitch_preserving_resample(source, target_length)
        if transformed.metadata["fallback"]:
            fallbacks.append({"label": match.label, **transformed.metadata})
            continue
        output[nat_start:nat_end] = transformed.audio
        used_spans.append(
            {
                "label": match.label,
                "natural_start_sample": nat_start,
                "natural_end_sample": nat_end,
                "tts_start_sample": tts_start,
                "tts_end_sample": tts_end,
                "natural_duration_s": (nat_end - nat_start) / sample_rate,
                "tts_duration_s": (tts_end - tts_start) / sample_rate,
            }
        )

    metadata = {
        **stats,
        "transform": "pitch_preserving_phone_local_warp",
        "sample_rate": int(sample_rate),
        "natural_sample_count": int(natural.size),
        "output_sample_count": int(output.size),
        "exact_sample_count": bool(output.size == natural.size),
        "fallback_count": len(fallbacks),
        "fallbacks": fallbacks,
        "used_spans": used_spans,
    }
    return WarpResult(audio=output, matched_spans=tuple(matches), metadata=metadata)


def ordinary_phone_local_warp(*args: Any, **kwargs: Any) -> WarpResult:
    """Expose the existing ordinary waveform warp as a named control."""
    return phone_local_warp(*args, **kwargs)


def _duration_targets(
    matches: Sequence[Any],
    alpha: float,
    sample_rate: int,
) -> tuple[list[int], dict[str, Any]]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    raw_durations = []
    for match in matches:
        natural_duration = match.natural_end_s - match.natural_start_s
        tts_duration = match.tts_end_s - match.tts_start_s
        raw_duration = natural_duration * np.exp(
            alpha * (np.log(max(tts_duration, 1e-9)) - np.log(max(natural_duration, 1e-9)))
        )
        raw_durations.append(float(raw_duration))
    natural_total = sum(
        int(np.rint((match.natural_end_s - match.natural_start_s) * sample_rate))
        for match in matches
    )
    raw_total = max(1, int(np.rint(sum(raw_durations) * sample_rate)))
    scale = natural_total / raw_total
    target_samples = [
        max(1, int(np.rint(duration * sample_rate * scale)))
        for duration in raw_durations
    ]
    if target_samples:
        target_samples[-1] += natural_total - sum(target_samples)
        target_samples[-1] = max(1, target_samples[-1])
    return target_samples, {
        "alpha": float(alpha),
        "raw_target_durations_s": raw_durations,
        "duration_scale_for_exact_n": float(scale),
        "matched_natural_sample_count": int(natural_total),
        "target_sample_counts": target_samples,
    }


def duration_alpha_target(
    natural_audio: np.ndarray,
    tts_audio: np.ndarray,
    sample_rate: int,
    natural_spans: Iterable[Span | Mapping[str, Any] | Any],
    tts_spans: Iterable[Span | Mapping[str, Any] | Any],
    *,
    alpha: float,
    min_confidence: float = 0.8,
) -> ControlResult:
    """Transfer relative TTS phone timing while preserving natural gaps and length."""
    natural = np.asarray(natural_audio, dtype=np.float32).reshape(-1)
    tts = np.asarray(tts_audio, dtype=np.float32).reshape(-1)
    natural_records = as_spans(natural_spans)
    tts_records = as_spans(tts_spans)
    matches, stats = match_token_spans(
        natural_records, tts_records, min_confidence=min_confidence
    )
    target_samples, duration_meta = _duration_targets(
        matches, alpha, sample_rate
    )
    match_iter = iter(matches)
    current_match = next(match_iter, None)
    output_parts: list[np.ndarray] = []
    cursor = 0
    match_index = 0
    fallback_count = 0
    for natural_span in natural_records:
        nat_start, nat_end = _sample_bounds(
            natural_span.start_s, natural_span.end_s, sample_rate, len(natural)
        )
        output_parts.append(natural[cursor:nat_start])
        if (
            current_match is not None
            and abs(current_match.natural_start_s - natural_span.start_s) <= 1e-7
            and current_match.label == natural_span.label
        ):
            tts_start, tts_end = _sample_bounds(
                current_match.tts_start_s, current_match.tts_end_s, sample_rate, len(tts)
            )
            transformed = pitch_preserving_resample(
                tts[tts_start:tts_end], target_samples[match_index]
            )
            if transformed.metadata["fallback"]:
                output_parts.append(natural[nat_start:nat_end])
                fallback_count += 1
            else:
                output_parts.append(transformed.audio)
            match_index += 1
            current_match = next(match_iter, None)
        else:
            output_parts.append(natural[nat_start:nat_end])
        cursor = nat_end
    output_parts.append(natural[cursor:])
    output = _exact_length(np.concatenate(output_parts) if output_parts else natural, len(natural))
    metadata = {
        **stats,
        **duration_meta,
        "transform": "duration_alpha_target",
        "sample_rate": int(sample_rate),
        "natural_sample_count": int(len(natural)),
        "output_sample_count": int(len(output)),
        "exact_sample_count": True,
        "natural_gap_policy": "preserve_natural_gap_samples",
        "fallback_count": int(fallback_count),
    }
    return ControlResult(audio=output, metadata=metadata)


def _energy_envelope(audio: np.ndarray, sample_rate: int, window_ms: float = 50.0) -> np.ndarray:
    window = max(1, int(round(window_ms * sample_rate / 1000.0)))
    kernel = np.ones(window, dtype=np.float32) / window
    return np.sqrt(np.maximum(np.convolve(audio * audio, kernel, mode="same"), 1e-12))


def energy_envelope_transfer(
    natural_audio: np.ndarray, tts_audio: np.ndarray, sample_rate: int
) -> ControlResult:
    """Transfer normalized TTS energy shape onto the natural waveform."""
    natural = np.asarray(natural_audio, dtype=np.float32).reshape(-1)
    tts = np.asarray(tts_audio, dtype=np.float32).reshape(-1)
    natural_env = _energy_envelope(natural, sample_rate)
    tts_env = _energy_envelope(tts, sample_rate)
    if not len(natural) or not len(tts) or not np.isfinite(tts_env).all():
        return ControlResult(
            audio=natural.copy(),
            metadata={
                "transform": "energy_envelope_transfer",
                "fallback": True,
                "fallback_reason": "empty_or_nonfinite_envelope",
                "sample_count": int(len(natural)),
            },
        )
    positions = np.linspace(0.0, 1.0, len(tts_env), dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, len(natural), dtype=np.float32)
    target_env = np.interp(target_positions, positions, tts_env)
    source_mean = float(np.mean(natural_env))
    target_mean = float(np.mean(target_env))
    if source_mean <= 1e-8 or target_mean <= 1e-8:
        return ControlResult(
            audio=natural.copy(),
            metadata={
                "transform": "energy_envelope_transfer",
                "fallback": True,
                "fallback_reason": "silent_source_or_target",
                "sample_count": int(len(natural)),
            },
        )
    target_env = target_env * (source_mean / target_mean)
    ratio = target_env / np.maximum(natural_env, 1e-5)
    output = natural * ratio
    source_rms = float(np.sqrt(np.mean(np.square(natural))))
    output_rms = float(np.sqrt(np.mean(np.square(output))))
    if output_rms > 0:
        output *= source_rms / output_rms
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 0.99:
        output *= 0.99 / peak
    return ControlResult(
        audio=output.astype(np.float32),
        metadata={
            "transform": "energy_envelope_transfer",
            "fallback": False,
            "window_ms": 50.0,
            "source_energy_mean": source_mean,
            "target_energy_mean_after_normalization": float(np.mean(target_env)),
            "energy_envelope_correlation": float(np.corrcoef(target_env, _energy_envelope(output, sample_rate))[0, 1]),
            "sample_count": int(len(output)),
        },
    )


def _lufs(audio: np.ndarray, sample_rate: int) -> float:
    import pyloudnorm as pyln

    return float(pyln.Meter(sample_rate).integrated_loudness(audio))


def loudness_transfer(natural_audio: np.ndarray, tts_audio: np.ndarray, sample_rate: int) -> ControlResult:
    """Match natural waveform gain to the TTS integrated loudness."""
    natural = np.asarray(natural_audio, dtype=np.float32).reshape(-1)
    target_lufs = _lufs(tts_audio, sample_rate)
    source_lufs = _lufs(natural, sample_rate)
    if not np.isfinite(source_lufs) or not np.isfinite(target_lufs):
        return ControlResult(
            audio=natural.copy(),
            metadata={
                "transform": "loudness_transfer",
                "fallback": True,
                "fallback_reason": "undefined_lufs",
                "source_lufs": source_lufs,
                "target_lufs": target_lufs,
                "gain_linear": 1.0,
                "sample_count": int(len(natural)),
            },
        )
    gain = 10.0 ** ((target_lufs - source_lufs) / 20.0)
    output = natural * gain
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 0.99:
        output = output * (0.99 / peak)
    return ControlResult(
        audio=output.astype(np.float32),
        metadata={
            "transform": "loudness_transfer",
            "source_lufs": source_lufs,
            "target_lufs": target_lufs,
            "gain_linear": gain,
            "sample_count": int(len(output)),
        },
    )


def _spectral_tilt(audio: np.ndarray, sample_rate: int) -> float:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    spectrum = np.abs(np.fft.rfft(values))
    frequencies = np.fft.rfftfreq(len(values), 1.0 / sample_rate)
    mask = (frequencies > 80.0) & (frequencies < sample_rate / 2 - 1) & (spectrum > 1e-12)
    if int(np.count_nonzero(mask)) < 4:
        return 0.0
    return float(np.polyfit(np.log10(frequencies[mask]), np.log10(spectrum[mask]), 1)[0])


def spectral_timbre_transfer(
    natural_audio: np.ndarray, tts_audio: np.ndarray, sample_rate: int
) -> ControlResult:
    """Match only a long-term spectral tilt, keeping natural duration and phase structure."""
    natural = np.asarray(natural_audio, dtype=np.float32).reshape(-1)
    target = _spectral_tilt(tts_audio, sample_rate)
    current = _spectral_tilt(natural, sample_rate)
    frequencies = np.fft.rfftfreq(len(natural), 1.0 / sample_rate)
    spectrum = np.fft.rfft(natural)
    gain_db = 20.0 * (target - current) * (
        np.log10(np.maximum(frequencies, 1.0)) - np.log10(1000.0)
    )
    gain = 10.0 ** (gain_db / 20.0)
    gain[0] = 1.0
    output = np.fft.irfft(spectrum * gain, n=len(natural)).astype(np.float32)
    source_rms = float(np.sqrt(np.mean(np.square(natural))))
    output_rms = float(np.sqrt(np.mean(np.square(output))))
    if output_rms > 0:
        output *= source_rms / output_rms
    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if peak > 0.99:
        output *= 0.99 / peak
    return ControlResult(
        audio=output.astype(np.float32),
        metadata={
            "transform": "spectral_tilt_transfer",
            "source_spectral_tilt": current,
            "target_spectral_tilt": target,
            "sample_count": int(len(output)),
        },
    )


def _median_f0(audio: np.ndarray, sample_rate: int) -> float | None:
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(values) < 1024:
        return None
    try:
        f0, voiced, _ = librosa.pyin(
            values, fmin=50.0, fmax=600.0, sr=sample_rate,
            frame_length=1024, hop_length=256,
        )
    except (ValueError, RuntimeError, librosa.util.exceptions.ParameterError):
        return None
    voiced_f0 = f0[voiced & np.isfinite(f0)]
    return float(np.median(voiced_f0)) if len(voiced_f0) else None


def f0_transfer(natural_audio: np.ndarray, tts_audio: np.ndarray, sample_rate: int) -> ControlResult:
    """Shift natural F0 toward the TTS median while preserving natural duration."""
    natural = np.asarray(natural_audio, dtype=np.float32).reshape(-1)
    natural_f0 = _median_f0(natural, sample_rate)
    tts_f0 = _median_f0(tts_audio, sample_rate)
    if natural_f0 is None or tts_f0 is None:
        return ControlResult(
            audio=natural.copy(),
            metadata={
                "transform": "f0_transfer",
                "fallback": True,
                "fallback_reason": "missing_voiced_f0",
                "natural_median_f0": natural_f0,
                "tts_median_f0": tts_f0,
                "sample_count": int(len(natural)),
            },
        )
    n_steps = 12.0 * np.log2(tts_f0 / natural_f0)
    output = librosa.effects.pitch_shift(
        natural.astype(np.float64), sr=sample_rate, n_steps=float(n_steps)
    )
    output = _exact_length(output, len(natural))
    return ControlResult(
        audio=output,
        metadata={
            "transform": "f0_transfer",
            "fallback": False,
            "natural_median_f0": natural_f0,
            "tts_median_f0": tts_f0,
            "pitch_shift_semitones": float(n_steps),
            "sample_count": int(len(output)),
        },
    )


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural", type=Path, required=True)
    parser.add_argument("--tts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--condition", choices=("loudness", "energy", "spectral_tilt", "f0"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import soundfile as sf

    args = _cli().parse_args(argv)
    natural, natural_sr = sf.read(args.natural, dtype="float32", always_2d=False)
    tts, tts_sr = sf.read(args.tts, dtype="float32", always_2d=False)
    if natural_sr != args.sample_rate or tts_sr != args.sample_rate:
        raise ValueError("input sample rates must equal --sample-rate")
    if natural.ndim != 1 or tts.ndim != 1:
        raise ValueError("inputs must be mono")
    builders = {
        "loudness": loudness_transfer,
        "energy": energy_envelope_transfer,
        "spectral_tilt": spectral_timbre_transfer,
        "f0": f0_transfer,
    }
    result = builders[args.condition](natural, tts, args.sample_rate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, result.audio, args.sample_rate, subtype="PCM_16")
    print(json.dumps(result.metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
