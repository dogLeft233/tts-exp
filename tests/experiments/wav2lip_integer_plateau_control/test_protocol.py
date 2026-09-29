from __future__ import annotations

import numpy as np

from scripts.experiments.wav2lip_integer_plateau_control import (
    analysis,
    config,
    media,
    protocol,
    validate,
)


def test_video_and_audio_shifts_are_toward_the_middle() -> None:
    frame_count = 119
    sample_count = 100_000
    midpoint = frame_count // 2
    frame_shift = np.where(np.arange(frame_count) < midpoint, 5, -5)
    sample_break = 640 * midpoint
    sample_shift = np.where(np.arange(sample_count) < sample_break, 3200, -3200)
    assert np.all((np.arange(frame_count) + frame_shift >= 0) & (np.arange(frame_count) + frame_shift < frame_count))
    assert np.all((np.arange(sample_count) + sample_shift >= 0) & (np.arange(sample_count) + sample_shift < sample_count))


def test_support_audit_covers_all_31_columns_and_predecessor() -> None:
    rows = [protocol._support_for_row(row, F=119, L=100_000, m=59, b=37_760, direction=1) for row in range(25, 34)]
    assert all(row["all_31_columns_legal"] for row in rows)
    assert all(row["same_platform"] for row in rows)
    assert rows[0]["columns"][0]["sample_start"] == ((25 - 15) * 4) * 160 - 1


def test_audio_plateau_is_exact_integer_source_indexing() -> None:
    source = np.arange(20_000, dtype=np.int16)
    indices = np.arange(source.size, dtype=np.int64) + np.where(np.arange(source.size) < 10_000, 3200, -3200)
    payload, evidence = media.make_plateau_pcm(source, indices)
    observed = np.frombuffer(payload, dtype="<i2")
    assert np.array_equal(observed, source[indices])
    assert evidence["sample_count"] == source.size


def test_peak_clearance_is_strictly_greater_than_threshold() -> None:
    curve = np.full(31, 2.0, dtype=np.float64)
    curve[15] = 1.0
    curve[14] = 1.0
    assert analysis.peak(curve)["clear"] is False
    curve[14] = 1.011
    assert analysis.peak(curve)["clear"] is True


def test_distance_reconstruction_matches_frozen_shape() -> None:
    rng = np.random.default_rng(4)
    visual = rng.normal(size=(7, config.EMBEDDING_DIM)).astype(np.float32)
    audio = rng.normal(size=(7, config.EMBEDDING_DIM)).astype(np.float32)
    result = validate._reconstruct_distance(visual, audio)
    assert result.shape == (7, 31)
    assert np.isfinite(result).all()


def test_timing_signs_follow_spec() -> None:
    left = {"PLUS": {"offset": 5, "clear": True}, "MINUS": {"offset": -5, "clear": True}}
    right = {"PLUS": {"offset": 0, "clear": True}, "MINUS": {"offset": 0, "clear": True}}
    result = analysis._timing(left, right, {"PLUS": 5, "MINUS": -5})
    assert result["passes"] is True


def test_short_segments_are_rejected() -> None:
    result = protocol._support_for_row(2, F=20, L=10_000, m=10, b=6400, direction=1)
    assert result["support_legal"] is False
