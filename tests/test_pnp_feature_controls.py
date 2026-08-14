"""Tests for interpretable TTS acoustic controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pnp_alignment_warp import Span
from pnp_feature_controls import (
    duration_alpha_target,
    energy_envelope_transfer,
    f0_transfer,
    loudness_transfer,
    ordinary_phone_local_warp,
    pitch_preserving_phone_local_warp,
    pitch_preserving_resample,
    reallocate_pause_samples,
    spectral_timbre_transfer,
)


def _tone(frequency: float, length: int, sample_rate: int = 16_000) -> np.ndarray:
    time = np.arange(length, dtype=np.float32) / sample_rate
    return (0.2 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def test_pitch_preserving_resample_is_exact_and_finite() -> None:
    source = _tone(220.0, 4096)
    result = pitch_preserving_resample(source, 6144)
    assert result.audio.shape == (6144,)
    assert result.audio.dtype == np.float32
    assert np.isfinite(result.audio).all()
    assert result.metadata["fallback"] is False


def test_pitch_preserving_resample_keeps_single_tone_frequency() -> None:
    sample_rate = 16_000
    source = _tone(220.0, 8192, sample_rate)
    result = pitch_preserving_resample(source, 12_288)
    spectrum = np.abs(np.fft.rfft(result.audio * np.hanning(len(result.audio))))
    frequencies = np.fft.rfftfreq(len(result.audio), 1.0 / sample_rate)
    peak_frequency = frequencies[int(np.argmax(spectrum))]
    assert abs(peak_frequency - 220.0) < 2.0


def test_nonfinite_phase_vocoder_output_is_explicit_fallback(monkeypatch) -> None:
    import pnp_feature_controls as controls

    def fake_time_stretch(*args, **kwargs):
        return np.array([np.nan, np.inf], dtype=np.float64)

    monkeypatch.setattr(controls.librosa.effects, "time_stretch", fake_time_stretch)
    source = np.ones(2048, dtype=np.float32)
    result = pitch_preserving_resample(source, 3072)
    assert result.metadata["fallback"] is True
    assert result.metadata["fallback_reason"] == "phase_vocoder_nonfinite"
    assert np.isfinite(result.audio).all()
    assert len(result.audio) == 3072


def test_pitch_preserving_local_warp_preserves_natural_length_and_gap() -> None:
    sr = 1000
    natural = np.zeros(1500, dtype=np.float32)
    tts = np.concatenate([_tone(100.0, 400, sr), np.zeros(100, dtype=np.float32), _tone(100.0, 700, sr)])
    natural_spans = [Span("a", 0.2, 0.6), Span("b", 0.9, 1.3)]
    tts_spans = [Span("a", 0.0, 0.4), Span("b", 0.5, 1.2)]
    result = pitch_preserving_phone_local_warp(natural, tts, sr, natural_spans, tts_spans)
    assert len(result.audio) == len(natural)
    assert result.metadata["exact_sample_count"]
    assert result.metadata["fallback_count"] == 0
    assert np.array_equal(result.audio[:200], natural[:200])
    assert np.array_equal(result.audio[600:900], natural[600:900])
    assert np.array_equal(result.audio[1300:], natural[1300:])


def test_pause_reallocation_is_identity_for_source_counts() -> None:
    audio = np.arange(20, dtype=np.float32)
    spans = [Span("a", 0.004, 0.008), Span("b", 0.012, 0.016)]
    result = reallocate_pause_samples(audio, 1000, spans, [4, 4, 4])
    assert np.array_equal(result.audio, audio)
    assert result.metadata["exact_sample_count"]
    assert result.metadata["pause_sample_count"] == 12


def test_pause_reallocation_preserves_every_speech_and_gap_sample() -> None:
    audio = np.arange(20, dtype=np.float32)
    spans = [Span("a", 0.004, 0.008), Span("b", 0.012, 0.016)]
    result = reallocate_pause_samples(audio, 1000, spans, [0, 8, 4])
    assert len(result.audio) == len(audio)
    assert np.isfinite(result.audio).all()
    assert np.array_equal(np.sort(result.audio), np.sort(audio))
    assert np.array_equal(result.audio[:4], audio[4:8])
    assert np.array_equal(result.audio[12:16], audio[12:16])


def test_pause_reallocation_rejects_nonconserving_targets() -> None:
    audio = np.arange(20, dtype=np.float32)
    spans = [Span("a", 0.004, 0.008), Span("b", 0.012, 0.016)]
    import pytest

    with pytest.raises(ValueError, match="preserve"):
        reallocate_pause_samples(audio, 1000, spans, [1, 2, 3])


def test_duration_alpha_zero_matches_natural_phone_durations() -> None:
    sr = 1000
    natural = np.zeros(1000, dtype=np.float32)
    tts = np.ones(700, dtype=np.float32)
    natural_spans = [Span("a", 0.1, 0.3), Span("b", 0.5, 0.7)]
    tts_spans = [Span("a", 0.0, 0.1), Span("b", 0.15, 0.65)]
    result = duration_alpha_target(natural, tts, sr, natural_spans, tts_spans, alpha=0.0)
    assert len(result.audio) == len(natural)
    assert result.metadata["target_sample_counts"] == [200, 200]
    assert result.metadata["exact_sample_count"]
    assert np.array_equal(result.audio[:100], natural[:100])
    assert np.array_equal(result.audio[300:500], natural[300:500])
    assert np.array_equal(result.audio[700:], natural[700:])


def test_duration_alpha_reports_normalized_tts_relative_durations() -> None:
    sr = 1000
    natural = np.zeros(1000, dtype=np.float32)
    tts = np.ones(1000, dtype=np.float32)
    natural_spans = [Span("a", 0.1, 0.3), Span("b", 0.5, 0.7)]
    tts_spans = [Span("a", 0.0, 0.1), Span("b", 0.1, 0.6)]
    result = duration_alpha_target(natural, tts, sr, natural_spans, tts_spans, alpha=1.0)
    assert sum(result.metadata["target_sample_counts"]) == 400
    assert result.metadata["target_sample_counts"][1] > result.metadata["target_sample_counts"][0]


def test_named_ordinary_control_matches_existing_warp() -> None:
    natural = np.zeros(200, dtype=np.float32)
    tts = np.ones(100, dtype=np.float32)
    spans = [Span("a", 0.05, 0.1)]
    result = ordinary_phone_local_warp(natural, tts, 1000, spans, spans, crossfade_ms=0)
    assert len(result.audio) == len(natural)


def test_loudness_transfer_preserves_length_and_finite() -> None:
    natural = _tone(220.0, 16_000)
    tts = 2.0 * _tone(220.0, 16_000)
    result = loudness_transfer(natural, tts, 16_000)
    assert len(result.audio) == len(natural)
    assert np.isfinite(result.audio).all()
    assert result.metadata["target_lufs"] > result.metadata["source_lufs"]


def test_spectral_timbre_transfer_preserves_length() -> None:
    natural = _tone(220.0, 16_000)
    tts = _tone(440.0, 16_000)
    result = spectral_timbre_transfer(natural, tts, 16_000)
    assert len(result.audio) == len(natural)
    assert np.isfinite(result.audio).all()
    assert result.metadata["sample_count"] == len(natural)


def test_energy_transfer_preserves_length_and_tracks_envelope() -> None:
    natural = _tone(220.0, 16_000)
    tts = np.concatenate([_tone(220.0, 8_000), _tone(440.0, 8_000)])
    result = energy_envelope_transfer(natural, tts, 16_000)
    assert len(result.audio) == len(natural)
    assert np.isfinite(result.audio).all()
    assert result.metadata["sample_count"] == len(natural)


    natural = _tone(180.0, 16_000)
    tts = _tone(260.0, 16_000)
    result = f0_transfer(natural, tts, 16_000)
    assert len(result.audio) == len(natural)
    assert np.isfinite(result.audio).all()
    assert result.metadata["sample_count"] == len(natural)


def test_runner_writes_provenance_and_conditions(tmp_path: Path) -> None:
    import soundfile as sf
    from run_pnp_feature_control_experiment import run_experiment

    manifest = tmp_path / "manifest.json"
    natural_dir = tmp_path / "natural"
    tts_dir = tmp_path / "tts"
    natural_dir.mkdir()
    tts_dir.mkdir()
    sr = 1_000
    sf.write(natural_dir / "1.wav", np.zeros(1_000, dtype=np.float32), sr)
    sf.write(tts_dir / "1.wav", np.ones(700, dtype=np.float32), sr)
    manifest.write_text(json.dumps({"records": [
        {"paired_key": "1", "sample_id": 1, "condition": "natural", "audio_path": "remote/natural/1.wav", "tokens": [
            {"token": "a", "start_s": 0.1, "end_s": 0.3}, {"token": "b", "start_s": 0.5, "end_s": 0.7}]},
        {"paired_key": "1", "sample_id": 1, "condition": "tts", "tts_provider": "provider-a", "audio_path": "remote/tts/1.wav", "tokens": [
            {"token": "a", "start_s": 0.0, "end_s": 0.1}, {"token": "b", "start_s": 0.1, "end_s": 0.6}]},
    ]}), encoding="utf-8")
    result = run_experiment(manifest, tmp_path / "out", natural_dir=natural_dir, tts_dir=tts_dir, provider="provider-a", alpha_values=(0.0, 1.0), sample_rate=sr)
    assert result["row_count"] == 11
    assert (tmp_path / "out" / "manifest.json").exists()
    assert all(item["exact_sample_count"] for item in result["items"] if item["condition"] != "tts_raw_oracle")
    assert all(item["natural_source_sample_rate"] == sr for item in result["items"])
