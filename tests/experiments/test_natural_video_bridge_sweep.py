import numpy as np
import pytest

from scripts.experiments.natural_video_bridge_sweep import distance_matrix, metrics


def test_identical_embeddings_peak_at_zero():
    x = np.random.default_rng(1).normal(size=(100, 16)).astype(np.float32)
    matrix = distance_matrix(x, x, 90)
    assert matrix.shape == (60, 31)
    assert np.isfinite(matrix).all()
    result = metrics(matrix, 15)
    assert result["offset"] == 0
    assert result["D"] < 1e-4
    assert result["DN"] == result["D0"]


def test_delayed_audio_has_negative_official_offset():
    x = np.random.default_rng(2).normal(size=(100, 16)).astype(np.float32)
    y = np.concatenate((np.zeros((5, 16), dtype=np.float32), x[:-5]))
    result = metrics(distance_matrix(x, y, 90), 15)
    assert result["offset"] == -5
    assert result["D0"] > result["D"]


def test_short_support_rejected():
    with pytest.raises(ValueError, match="support"):
        distance_matrix(np.zeros((40, 16)), np.zeros((40, 16)), 40)
