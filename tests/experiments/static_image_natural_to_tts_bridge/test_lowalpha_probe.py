import numpy as np
import pytest

from scripts.experiments.static_image_bridge.frontal_probe import frozen_support
from scripts.experiments.static_image_bridge.lowalpha_probe import (
    aggregate,
    pair_metrics,
)
from scripts.experiments.static_image_bridge.lowalpha_subset import summarize
from scripts.experiments.static_image_bridge.lowalpha_verify import support


def test_replacement_uses_natural_audio_not_own_score():
    nn = np.tile(np.arange(31, dtype=float), (50, 1))
    bn = nn + 2
    bb = nn * 2
    row = pair_metrics(nn, bn, bb, list(range(10, 40)))
    assert row["delta_C"] == 0
    assert row["D_improvement"] == -2
    assert row["own_vs_N_diagonal_C"] == 15


def test_grouping_averages_portraits_before_bootstrap():
    rows = [{"source_group": "one", "image_id": str(i), "delta_C": i,
             "D_improvement": i, "anchor_improvement": i,
             "own_vs_N_diagonal_C": i, "offset_change": 0} for i in (1, 2, 3)]
    result = aggregate(rows)
    assert result["grouped"]["delta_C"]["n"] == 1
    assert result["grouped"]["delta_C"]["mean"] == 2
    with pytest.raises(ValueError):
        aggregate(rows[:-1])


def test_support_has_full_lag_and_frontend_margins():
    length = 160000
    w = frozen_support(length)["primary"]
    assert min(w) * 640 - 15 * 640 - 1 >= 0
    assert max(w) * 640 + 15 * 640 + 3440 <= length
    assert support(length) == w


def test_native_and_replacement_are_distinct_and_grouped():
    n = {"C": 5., "D": 7., "lag": 0, "curve": [7.] * 31}
    own = {"C": 6., "D": 6., "lag": 0, "curve": [6.] * 31}
    replaced = {"C": 4., "D": 8., "lag": 0, "curve": [8.] * 31}
    rows = [{"sample_id": "sample", "source_group": "source", "image_id": im,
             "N": n, "B": replaced, "B_own": own} for im in ("3", "6", "9")]
    assert summarize(rows, "native")["delta_C"]["mean"] == 1
    assert summarize(rows, "replacement")["delta_C"]["mean"] == -1
    assert summarize(rows, "native")["delta_C"]["n"] == 1
