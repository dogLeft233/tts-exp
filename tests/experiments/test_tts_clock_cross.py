import numpy as np
import pytest

from scripts.experiments.tts_clock_cross import (
    estimate_delay,
    interval,
    shift_pcm,
    summarize_matrix,
)


def test_interior_excludes_padded_boundary_and_decomposes():
    matrix = np.tile(np.abs(np.arange(31) - 13), (80, 1)).astype(float)
    matrix[:15, :] = 1e5
    matrix[-15:, :] = 1e5
    result = summarize_matrix(matrix)["interior"]
    assert result["rows"] == 50
    assert result["offset"] == 2
    assert result["D"] == 0
    assert result["C"] == result["B"] - result["D"]


def test_paired_interval_preserves_zero_difference():
    result = interval([0, 0, 0])
    assert result["ci95"] == [0, 0]
    assert result["mean"] == 0


def test_short_matrix_rejected():
    with pytest.raises(ValueError):
        summarize_matrix(np.zeros((20, 31)))


def test_waveform_delay_recovery_independent_of_scores():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=10000)
    delayed = shift_pcm(signal, 737) * 0.8
    result = estimate_delay(signal, delayed)
    assert result["lag_samples"] == 737
    assert result["aligned_correlation"] > 0.999999
    assert result["least_squares_gain"] == pytest.approx(0.8)


def test_shift_direction_and_identity():
    x = np.arange(10, dtype=np.int16)
    assert np.array_equal(shift_pcm(x, 2), [0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    assert np.array_equal(shift_pcm(x, -2), [2, 3, 4, 5, 6, 7, 8, 9, 0, 0])
    assert np.array_equal(shift_pcm(x, 0), x)
