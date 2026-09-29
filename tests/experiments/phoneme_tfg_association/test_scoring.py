from __future__ import annotations

import numpy as np

from scripts.experiments.phoneme_tfg_association.scoring import distance_from_embeddings, summarize_pair_matrices


def test_dynamic_three_arm_support_and_curve_decomposition():
    rng = np.random.default_rng(7)
    natural = rng.normal(size=(70, 31)).astype(np.float32)
    cloud = rng.normal(size=(83, 31)).astype(np.float32)
    local = rng.normal(size=(61, 31)).astype(np.float32)
    rows = summarize_pair_matrices(natural, {"qwen_cloud": cloud, "qwen_local": local})
    assert {(row["condition"], row["support"]) for row in rows} == {(condition, support) for condition in ("natural", "qwen_cloud", "qwen_local") for support in ("FULL", "INTERIOR", "EQUAL_COUNT")}
    for row in rows:
        assert len(row["curve"]) == 31
        assert abs(float(row["sync_c"]) - (float(row["background_b"]) - float(row["sync_d"]))) < 1e-6
    assert len({tuple(row["support_rows"]) for row in rows if row["support"] == "EQUAL_COUNT"}) == 3


def test_distance_has_dynamic_t_and_lag_width():
    visual = np.ones((37, 4), dtype=np.float32)
    audio = np.zeros((37, 4), dtype=np.float32)
    result = distance_from_embeddings(visual, audio)
    assert result.shape == (37, 31)
    assert np.isfinite(result).all()
