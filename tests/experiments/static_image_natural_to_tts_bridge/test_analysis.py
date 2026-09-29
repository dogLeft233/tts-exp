import numpy as np

from scripts.experiments.static_image_bridge import config
from scripts.experiments.static_image_bridge.analysis import (
    analyze_stage_a,
    bootstrap_stats,
    common_w,
    score_metrics,
)
from scripts.experiments.static_image_bridge.config import RunPaths


def _row(path) -> dict[str, str]:
    return {"matrix_path": str(path)}


def test_common_w_intersects_finite_support_and_score_uses_lowest_tied_lag(tmp_path) -> None:
    first = np.full((30, 31), 4.0, dtype=np.float64)
    second = np.full((30, 31), 4.0, dtype=np.float64)
    # Equal minima at lag -1 and 0: the frozen tie rule must choose -1.
    first[:, config.SYNCNET_VSHIFT - 1] = 1.0
    first[:, config.SYNCNET_VSHIFT] = 1.0
    second[:2, :] = np.nan
    first_path = tmp_path / "first.npy"
    second_path = tmp_path / "second.npy"
    np.save(first_path, first)
    np.save(second_path, second)
    W = common_w((_row(first_path), _row(second_path)))
    assert W.tolist() == list(range(2, 30))
    metrics = score_metrics(_row(first_path), W)
    assert metrics["k"] == -1
    assert metrics["D"] == 1.0
    assert metrics["D0"] == 1.0


def test_bootstrap_is_seeded_and_uses_source_group_means() -> None:
    values = [1.0, 3.0, 10.0]
    groups = ["a", "a", "b"]
    first = bootstrap_stats(values, groups, draws=500, seed=123)
    second = bootstrap_stats(values, groups, draws=500, seed=123)
    assert first == second
    assert first["mean"] == 6.0
    assert first["group_means"] == {"a": 2.0, "b": 10.0}


def test_fake_repeat_fails_the_stage_a_gate(tmp_path) -> None:
    inputs = {"records": []}
    scores = {}
    for index in range(config.EXPECTED_RECORD_COUNT):
        sid = f"sample_{index:02d}"
        inputs["records"].append({"sample_id": sid, "source_group": sid})
        for video, audio, minimum in (("N", "N", 15), ("N_REPEAT", "N", 10), ("N", "ND", 20)):
            matrix = np.full((30, 31), 4.0, dtype=np.float64)
            matrix[:, minimum] = 1.0
            path = tmp_path / f"{sid}_{video}_{audio}.npy"
            np.save(path, matrix)
            scores[(sid, video, audio)] = {"matrix_path": str(path)}
    result = analyze_stage_a(RunPaths(tmp_path), inputs, scores)
    assert result["repeatability"]["record_pass_count"] == 0
    assert result["measurement_decision"] == "MEASUREMENT_CONTROL_FAILED"
