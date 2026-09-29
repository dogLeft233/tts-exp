from __future__ import annotations

import ast
import inspect

import numpy as np

from scripts.experiments import check_ditto_timing_rate as checker
from scripts.experiments import ditto_timing_rate as runner
from scripts.experiments import ditto_timing_rate_metrics as metrics


def test_video_identity_and_l_over_m_mapping_are_explicit() -> None:
    frames = np.zeros((4, 224, 224, 3), dtype=np.uint8)
    for index in range(4):
        frames[index] = index * 40
    same, receipt = runner.retime_frames(frames, 4)
    assert np.array_equal(same, frames)
    assert receipt["endpoint_hold_count"] == 1
    fast, _ = runner.retime_frames(frames, 2)
    assert np.array_equal(fast[:, 0, 0, 0], np.asarray([0, 80], dtype=np.uint8))
    slow, _ = runner.retime_frames(frames, 8)
    assert np.array_equal(slow[:, 0, 0, 0], np.asarray([0, 20, 40, 60, 80, 100, 120, 120], dtype=np.uint8))


def test_support_requires_complete_temporal_intervals() -> None:
    short = metrics.support_for_halves(70)
    assert short["complete"] is False
    long = metrics.support_for_halves(100)
    assert long["complete"] is True
    left = set(long["halves"]["H0"]["rows"])
    right = set(long["halves"]["H1"]["rows"])
    assert left.isdisjoint(right)
    for row in left:
        assert row - metrics.GUARD_LEFT >= 0
        assert row + metrics.GUARD_RIGHT <= long["boundary"]


def test_known_lag_is_recovered_with_the_declared_sign() -> None:
    rng = np.random.default_rng(13)
    visual = rng.normal(size=(100, 8)).astype(np.float32)
    audio = np.zeros_like(visual)
    audio[5:] = visual[:-5]
    result = metrics.calibrate_lag(visual, audio, list(range(20, 70)))
    assert result["k0"] == 5


def test_common_phase_support_and_rank_ties() -> None:
    counts = {key: 100 for key in ("N_O", "N_SHORT", "N_LONG", "T_O", "T_SHORT", "T_LONG")}
    phases = metrics.build_common_phases(counts)
    assert phases["complete"]
    assert phases["distinct_index_counts"]["N_O"]["H0"] >= 10
    assert phases["distinct_index_counts"]["T_LONG"]["H1"] >= 10
    assert metrics._win(1.0, 1.0) == 0.5
    assert metrics._win(1.1, 1.0) == 1.0
    assert metrics._win(0.9, 1.0) == 0.0


def test_bootstrap_primary_family_and_insufficient_shape() -> None:
    summary = metrics.bootstrap_summary([1.0, -1.0] * 15, primary=True)
    assert summary["status"] == "COMPLETE"
    assert summary["ci_percentiles"] == [1.25, 98.75]
    assert metrics.bootstrap_summary([], primary=True)["status"] == "NOT_ESTIMABLE"


def test_rank_is_invariant_under_positive_affine_embedding_transform() -> None:
    rng = np.random.default_rng(7)
    v = rng.normal(size=(100, 8))
    a = rng.normal(size=(100, 8))
    bundle = metrics.build_common_phases({"X": 100})
    support = metrics.support_for_halves(100)
    cal = {"H0": metrics.calibrate_lag(v, a, support["halves"]["H1"]["rows"]), "H1": metrics.calibrate_lag(v, a, support["halves"]["H0"]["rows"])}
    first = metrics.rank_cell(v, a, bundle["records"], "X", cal)
    second = metrics.rank_cell(3.0 * v + 7.0, 3.0 * a + 7.0, bundle["records"], "X", cal)
    assert first["R"] == second["R"]


def test_checker_does_not_import_producer_metrics() -> None:
    tree = ast.parse(inspect.getsource(checker))
    imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert "scripts.experiments.ditto_timing_rate_metrics" not in imported
