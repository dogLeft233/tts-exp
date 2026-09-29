import numpy as np
import pytest

from scripts.experiments.static_image_bridge.frontal_probe import frozen_support, generator_support, metrics, summary


def test_support_includes_full_frontends_and_excludes_delay_padding():
    length = 96256
    p = frozen_support(length)
    assert len(p["primary"]) > 20
    assert set(p["delay"]) < set(p["primary"])
    for t in p["primary"]:
        assert generator_support(t)[0] >= 0
        assert generator_support(t)[1] <= length
        assert (t - 15) * 640 - 1 >= 0
        assert (t + 15) * 640 + 3440 <= length
    for t in p["delay"]:
        assert (t - 15) * 640 - 1 >= 3200


@pytest.mark.parametrize("length", [96256, 96257, 96258, 96259])
def test_local_support_intersects_unequal_swapped_quarters(length):
    p = frozen_support(length)
    b1, b2, b3 = length // 4, length // 2, 3 * length // 4
    for label, lo, hi in (("2", b1, min(b2, b1 + b3 - b2)), ("3", max(b2, b1 + b3 - b2), b3)):
        for lag, windows in p["local_by_lag"][label].items():
            for t in windows:
                assert generator_support(t)[0] >= lo
                assert generator_support(t)[1] <= hi
                assert (t + int(lag)) * 640 - 1 >= lo
                assert (t + int(lag)) * 640 + 3440 <= hi


def test_metrics_and_lag_sign_and_no_adaptive_finite_mask():
    matrix = np.full((50, 31), 8.0)
    matrix[:, 20] = 2.0
    result = metrics(matrix, list(range(20, 30)))
    assert result["C"] == 6 and result["D"] == 2 and result["lag"] == 5
    matrix[20, 0] = np.nan
    with pytest.raises(ValueError, match="cannot adapt"):
        metrics(matrix, list(range(20, 30)))


def test_bootstrap_is_reproducible_and_counts_signed_improvement():
    result = summary([-1, -2, 1])
    assert result == summary([-1, -2, 1])
    assert result["positive"] == 1 and result["n"] == 3
    assert result["mean"] == pytest.approx(-2 / 3)
