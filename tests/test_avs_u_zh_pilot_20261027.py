"""Causal time controls and cosine-definition checks for the AVS replication."""
import importlib.util
import json
import os
from pathlib import Path
import numpy as np


def load():
    p = Path(__file__).parents[1] / "scripts/experiments/avs_u_zh_pilot_20261027.py"
    s = importlib.util.spec_from_file_location("avs_pilot", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def test_pcm_shift_preserves_clock_without_wrap():
    m = load()
    x = np.arange(12, dtype=np.int16) + 1
    assert np.array_equal(m.shift_pcm(x, 3), [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert np.array_equal(m.shift_pcm(x, -3), [4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0])
    assert np.array_equal(m.shift_pcm(x, 0), x)


def test_score_preserves_negative_cosines_and_distinct_reductions():
    m = load()
    a = np.array([[1., 0], [0, 4.]])
    v = np.array([[-1., 0], [0, 4.]])
    result = m.similarities(a, v)
    assert abs(result["frame_mean"]) < 1e-12
    assert abs(result["flatten_cosine"] - 15 / 17) < 1e-12
    assert m.similarities(a, -a)["frame_mean"] == -1


def test_real_weight_forward_and_saved_embedding_replay():
    """Optional integration check after model load and visual transform setup."""
    run = os.environ.get("AVS_U_PILOT_RUN")
    if not run:
        import pytest
        pytest.skip("AVS_U_PILOT_RUN required for real-weight integration check")
    root = Path(run)
    assert json.loads((root / "model_load.json").read_text())["strict_encoder"]
    scores = json.loads((root / "scores.json").read_text())
    assert scores
    for key, result in scores.items():
        with np.load(root / "embeddings" / (key + ".npz")) as data:
            assert data["audio"].shape == data["video"].shape
            assert data["video"].shape == (result["frames"], 768)
            replay = load().similarities(data["audio"], data["video"])
        assert abs(replay["frame_mean"] - result["frame_mean"]) < 1e-12
