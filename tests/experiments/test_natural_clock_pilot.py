import numpy as np
import pytest

from scripts.experiments.natural_clock_pilot import comparison
from scripts.experiments.natural_clock_pilot_audio import (
    canonical_phone,
    duration_grid,
    encode,
    equalize,
)


def test_identity_and_length():
    pcm = np.random.default_rng(42).integers(-3000, 3000, 96007, dtype=np.int16)
    rt, _ = equalize(pcm, 0)
    assert np.array_equal(rt, pcm)
    eq, _ = equalize(pcm)
    assert len(eq) == len(pcm) and not np.array_equal(eq, pcm)
    assert (
        abs(
            np.sqrt(np.mean(eq.astype(float) ** 2))
            / np.sqrt(np.mean(pcm.astype(float) ** 2))
            - 1
        )
        < 0.001
    )


def test_no_peak_time_shift():
    x = np.zeros(16000, dtype=np.int16)
    x[8000] = 16000
    y, _ = equalize(x)
    assert np.argmax(np.abs(y)) == 8000


@pytest.mark.parametrize("x", [[np.nan], [1.0], [-1.01]])
def test_reject_bad_audio(x):
    with pytest.raises(ValueError):
        encode(x)


def test_duration_grid():
    intervals = [(0, 0.4, ""), (0.4, 0.46, "ɪ"), (0.46, 1.0, "n")]
    edges, d = duration_grid(intervals, 16000)
    assert (d >= 0).all() and d.sum() == 87
    assert np.max(np.abs(edges[:-1] * 256 / 22050 - np.array([0, 0.4, 0.46]))) < 0.006
    assert canonical_phone("ɪ") == "@IH1" and canonical_phone("kʰ") == "@K"
    assert canonical_phone("") == " "
    with pytest.raises(ValueError):
        canonical_phone("unknown")


def test_anchor_does_not_research_candidate_lag():
    n = {"C": 3, "D": 7, "lag": 0, "curve": [8] * 15 + [7] + [8] * 15}
    a = {"C": 4, "D": 6, "lag": 1, "curve": [9] * 16 + [6] + [9] * 14}
    r = comparison(n, a)
    assert (
        r["delta_C"] == 1 and r["D_improvement"] == 1 and r["anchor_improvement"] == -2
    )


def test_independent_validator_math():
    from scripts.experiments.natural_clock_pilot_verify import deltas, support
    from scripts.experiments.static_image_bridge.frontal_probe import frozen_support

    n = np.array([9.0] * 15 + [7.0] + [9.0] * 15)
    a = np.array([10.0] * 16 + [6.0] + [10.0] * 14)
    v = deltas(n, a)
    assert v == {
        "delta_C": 2.0,
        "D_improvement": 1.0,
        "anchor_improvement": -3.0,
        "lag_change": 1,
    }
    assert support(96007) == frozen_support(96007)["primary"]
