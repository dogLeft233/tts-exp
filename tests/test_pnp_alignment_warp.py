"""Tests for the CPU-only phone-local TTS target construction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pnp_alignment_warp import (
    Span,
    global_duration_match,
    identity_target,
    local_resample,
    match_token_spans,
    phone_local_warp,
)


def _spans(labels: list[str], durations: list[float], confidence: float = 1.0) -> list[Span]:
    result: list[Span] = []
    start = 0.0
    for label, duration in zip(labels, durations):
        result.append(Span(label, start, start + duration, confidence))
        start += duration
    return result


def test_local_resample_is_exact_and_finite() -> None:
    source = np.sin(np.linspace(0, 2 * np.pi, 17, dtype=np.float32))
    result = local_resample(source, 31)
    assert result.shape == (31,)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_local_warp_preserves_natural_timeline_and_unmatched_pause() -> None:
    sr = 1000
    natural = np.zeros(1000, dtype=np.float32)
    natural[100:300] = 0.25
    natural[500:700] = -0.25
    tts = np.concatenate(
        [
            np.full(100, 0.8, dtype=np.float32),
            np.zeros(50, dtype=np.float32),
            np.full(300, -0.8, dtype=np.float32),
        ]
    )
    natural_spans = [Span("a", 0.1, 0.3), Span("b", 0.5, 0.7)]
    tts_spans = [Span("a", 0.0, 0.1), Span("b", 0.15, 0.45)]
    result = phone_local_warp(
        natural,
        tts,
        sr,
        natural_spans,
        tts_spans,
        crossfade_ms=0,
    )
    assert len(result.audio) == len(natural)
    assert result.metadata["exact_sample_count"]
    assert np.allclose(result.audio[:100], natural[:100])
    assert np.allclose(result.audio[300:500], natural[300:500])
    assert np.allclose(result.audio[700:], natural[700:])
    assert result.audio[150].mean() > 0.5
    assert result.audio[600].mean() < -0.5


def test_local_warp_crossfade_reduces_splice_jump() -> None:
    sr = 1000
    natural = np.zeros(200, dtype=np.float32)
    tts = np.concatenate([np.ones(50, dtype=np.float32), -np.ones(50, dtype=np.float32)])
    spans = [Span("a", 0.05, 0.1), Span("b", 0.1, 0.15)]
    tts_spans = [Span("a", 0.0, 0.05), Span("b", 0.05, 0.1)]
    no_fade = phone_local_warp(natural, tts, sr, spans, tts_spans, crossfade_ms=0).audio
    faded = phone_local_warp(natural, tts, sr, spans, tts_spans, crossfade_ms=10).audio
    assert np.max(np.abs(np.diff(faded))) < np.max(np.abs(np.diff(no_fade)))


def test_low_confidence_span_falls_back_to_natural() -> None:
    natural = np.linspace(-1, 1, 100, dtype=np.float32)
    tts = np.ones(100, dtype=np.float32)
    result = phone_local_warp(
        natural,
        tts,
        1000,
        [Span("a", 0.02, 0.08, confidence=0.4)],
        [Span("a", 0.02, 0.08)],
        min_confidence=0.8,
        crossfade_ms=0,
    )
    assert np.array_equal(result.audio, natural)
    assert result.metadata["matched_count"] == 0
    assert result.metadata["alignment_source"] == "mfa"


def test_matching_does_not_pair_different_labels() -> None:
    matches, stats = match_token_spans(
        [Span("a", 0, 0.1), Span("b", 0.1, 0.2)],
        [Span("x", 0, 0.1), Span("b", 0.1, 0.2)],
    )
    assert [match.label for match in matches] == ["b"]
    assert stats["label_mismatch_count"] == 1


def test_controls_have_expected_lengths() -> None:
    natural = np.arange(13, dtype=np.float32)
    tts = np.arange(7, dtype=np.float32)
    assert np.array_equal(identity_target(natural), natural)
    assert len(global_duration_match(tts, len(natural))) == len(natural)


def test_invalid_spans_are_rejected() -> None:
    with pytest.raises(ValueError, match="sorted"):
        phone_local_warp(
            np.zeros(100, dtype=np.float32),
            np.zeros(100, dtype=np.float32),
            1000,
            [Span("a", 0.1, 0.2), Span("b", 0.15, 0.3)],
            [Span("a", 0, 0.1), Span("b", 0.1, 0.2)],
        )
