from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.experiments.lrs3_tts_gain_mechanism import config, runner
from scripts.experiments.lrs3_tts_gain_mechanism.analysis import (
    _curve_metrics,
    bootstrap_summary,
    curve_analysis,
    decompose_pair,
    equal_count_rows,
    group_pairs,
    pair_curve_metrics,
)
from scripts.experiments.lrs3_tts_gain_mechanism.common import write_self_hashed_json
from scripts.experiments.lrs3_tts_gain_mechanism.runner import (
    ProtocolError,
    build_cohort,
    load_historical_records,
    load_input_bindings,
)


def test_algebraic_decomposition_has_two_explicit_components():
    natural = {
        "model": "LeapTalk",
        "sample_id": 151,
        "source_group": "g",
        "condition": "natural_raw",
        "sync_c": 2.0,
        "sync_d": 10.0,
        "background_b_hat": 12.0,
    }
    tts = {
        **natural,
        "condition": "tts_raw",
        "sync_c": 3.5,
        "sync_d": 9.0,
        "background_b_hat": 12.5,
    }
    result = decompose_pair(natural, tts)
    assert result["gain_c"] == pytest.approx(1.5)
    assert result["gain_match"] == pytest.approx(1.0)
    assert result["gain_background"] == pytest.approx(0.5)
    assert result["decomposition_residual"] == pytest.approx(0.0, abs=1e-10)


def test_curve_support_uses_mean_then_median_and_correct_offset_sign():
    matrix = np.zeros((50, 31), dtype=np.float32)
    matrix[:, :] = np.arange(31, dtype=np.float32)
    metrics = _curve_metrics(matrix, range(15, 35), label="test")
    assert metrics["sync_d"] == pytest.approx(0.0)
    assert metrics["sync_c"] == pytest.approx(15.0)
    assert metrics["official_offset"] == 15
    assert metrics["offsets"] == list(range(15, -16, -1))
    assert metrics["c5"] == pytest.approx(5.0)


def test_curve_components_separate_valley_and_background_changes():
    natural_curve = np.full(31, 10.0)
    natural_curve[15] = 1.0
    valley_curve = natural_curve.copy()
    valley_curve[15] = 0.0
    background_curve = natural_curve.copy()
    background_curve[0:17] = 12.0
    background_curve[15] = 1.0
    shifted_curve = natural_curve + 1.0
    natural = _curve_metrics(np.tile(natural_curve, (50, 1)), range(20, 30), label="natural")
    valley = _curve_metrics(np.tile(valley_curve, (50, 1)), range(20, 30), label="valley")
    background = _curve_metrics(np.tile(background_curve, (50, 1)), range(20, 30), label="background")
    shifted = _curve_metrics(np.tile(shifted_curve, (50, 1)), range(20, 30), label="shifted")
    assert natural["background_b"] - valley["background_b"] == pytest.approx(0.0)
    assert natural["sync_d"] - valley["sync_d"] == pytest.approx(1.0)
    assert background["background_b"] - natural["background_b"] == pytest.approx(2.0)
    assert background["sync_d"] == pytest.approx(natural["sync_d"])
    assert shifted["sync_d"] - natural["sync_d"] == pytest.approx(1.0)
    assert shifted["background_b"] - natural["background_b"] == pytest.approx(1.0)


def test_flat_and_boundary_troughs_are_explicitly_flagged():
    flat = _curve_metrics(np.zeros((40, 31)), range(20, 30), label="flat")
    boundary_curve = np.full(31, 10.0)
    boundary_curve[0] = 1.0
    boundary = _curve_metrics(np.tile(boundary_curve, (40, 1)), range(20, 30), label="boundary")
    tied_curve = np.full(31, 10.0)
    tied_curve[3:5] = 1.0
    tied = _curve_metrics(np.tile(tied_curve, (40, 1)), range(20, 30), label="tied")
    assert flat["trough_width_status"] == "flat_curve" and flat["trough_width_ms"] is None
    assert boundary["trough_width_status"] == "censored"
    assert tied["min_index"] == 3 and tied["trough_ties"] == [3, 4]


def test_equal_count_is_unique_and_pair_analysis_has_three_supports():
    assert equal_count_rows(7, target_count=4) == [0, 2, 4, 6]
    natural = np.tile(np.arange(31, dtype=np.float64), (60, 1))
    tts = np.tile(np.arange(31, dtype=np.float64) + 0.5, (63, 1))
    endpoints, support = pair_curve_metrics(natural, tts, sample_id=151, source_group="g")
    assert support["equal_count"] == 30
    assert len(endpoints) == 6
    assert {row["support"] for row in endpoints} == {"FULL", "INTERIOR", "EQUAL_COUNT"}
    assert all(len(row["curve"]) == 31 for row in endpoints)


def test_group_aggregation_gives_each_source_equal_weight_after_within_group_mean():
    rows = []
    for sample_id, group, value in ((1, "a", 1.0), (2, "b", 3.0), (3, "b", 5.0), (4, "b", 7.0)):
        rows.append({"model": "Ditto", "sample_id": sample_id, "source_group": group, "gain_c": value, "gain_match": 0.0, "gain_background": value, "c_positive": value > 0})
    grouped = group_pairs(rows)
    assert [row["source_group"] for row in grouped] == ["a", "b"]
    assert [row["gain_c"] for row in grouped] == pytest.approx([1.0, 5.0])
    assert np.mean([row["gain_c"] for row in grouped]) == pytest.approx(3.0)


def test_curve_bootstrap_uses_three_comparison_bonferroni_interval():
    indices = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.int64)
    summary = bootstrap_summary(
        {"a": 1.0, "b": 3.0, "c": 5.0},
        indices,
        metric="test",
        bonferroni_comparisons=3,
    )
    assert summary["bonferroni_comparisons"] == 3
    assert summary["bonferroni_tail_probability"] == pytest.approx(1 / 120)
    assert summary["ci98_333"] == pytest.approx([1.0333333333, 4.9666666667])


def test_curve_analysis_records_stage_b_bonferroni_contract():
    endpoints = []
    for sample_id in config.CURVE_SAMPLE_IDS:
        for condition, value in (("natural_raw", 10.0), ("tts_raw", 9.0)):
            for support in ("FULL", "INTERIOR", "EQUAL_COUNT"):
                endpoints.append(
                    {
                        "model": "LeapTalk",
                        "sample_id": sample_id,
                        "source_group": f"g{sample_id}",
                        "condition": condition,
                        "support": support,
                        "sync_c": value,
                        "sync_d": 1.0,
                        "background_b": value + 1.0,
                        "c5": value,
                        "d0": 1.0,
                        "search_gain_s": 0.0,
                        "trough_width_ms": None,
                        "official_offset": 0,
                        "boundary_best": False,
                    }
                )
    result = curve_analysis(endpoints, {sample_id: f"g{sample_id}" for sample_id in config.CURVE_SAMPLE_IDS})
    assert result["bonferroni"] == {"comparisons": 3, "interval": "98.333333%", "tail_probability": 1 / 120}
    assert all(item["bonferroni_comparisons"] == 3 for item in result["core_summaries"].values())


def test_delayed_control_expected_offset_direction_is_minus_five():
    from scripts.experiments.lrs3_tts_gain_mechanism.runner import _control_metrics

    original_curve = np.full(31, 10.0)
    original_curve[15] = 1.0
    delayed_curve = np.full(31, 10.0)
    delayed_curve[20] = 1.0
    original = np.tile(original_curve, (70, 1))
    delayed = np.tile(delayed_curve, (70, 1))
    result = _control_metrics(original, delayed)
    assert result["status"] == "CONTROL_PASS"
    assert result["offset_change_delayed_minus_original"] == -5
    assert result["original_argmin_delayed_distance_higher"] is True


def test_delayed_control_at_search_boundary_is_explicitly_failed():
    from scripts.experiments.lrs3_tts_gain_mechanism.runner import _control_metrics

    original_curve = np.full(31, 10.0)
    original_curve[0] = 1.0
    delayed_curve = np.full(31, 10.0)
    delayed_curve[5] = 1.0
    result = _control_metrics(np.tile(original_curve, (70, 1)), np.tile(delayed_curve, (70, 1)))
    assert result["offset_change_delayed_minus_original"] == -5
    assert result["original_boundary_best"] is True
    assert result["status"] == "CONTROL_FAILED"


def test_resource_wait_cleanup_removes_only_uncommitted_cell_artifacts(tmp_path):
    cell_dir = tmp_path / "cell"
    official = cell_dir / "official"
    worker_tmp = cell_dir / "worker_tmp"
    crop = cell_dir / "crop.avi"
    selection = cell_dir / "crop_selection.json"
    matrix = tmp_path / "matrix.npy"
    result = cell_dir / "worker.json"
    for path in (official / "pywork", worker_tmp):
        path.mkdir(parents=True)
    for path in (crop, selection, matrix, result):
        path.write_bytes(b"partial")

    runner._cleanup_interrupted_cell(
        cell_dir,
        matrix_path=matrix,
        result_path=result,
        crop=crop,
        selection_path=selection,
        repeat=False,
    )

    assert not official.exists()
    assert not worker_tmp.exists()
    assert not any(path.exists() for path in (crop, selection, matrix, result))


def test_syncnet_worker_declares_raw_pcm_format_for_ffmpeg_pipe(monkeypatch, tmp_path):
    from scripts.experiments.lrs3_tts_gain_mechanism import syncnet_worker

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return type("Completed", (), {"stdout": b"pcm"})()

    monkeypatch.setattr(syncnet_worker.subprocess, "run", fake_run)
    assert syncnet_worker.media_pcm(tmp_path / "crop.avi", Path("/usr/bin/ffmpeg")) == b"pcm"
    assert captured["command"][-3:-1] == ["-f", "s16le"]
    assert captured["command"][-1] == "pipe:1"
    assert captured["kwargs"] == {"capture_output": True, "check": True}


def test_bound_lrs3_cohort_and_historical_checks_are_reproducible():
    bindings = load_input_bindings()
    cohort, _ = build_cohort(bindings)
    assert [row["sample_id"] for row in cohort] == list(config.HISTORICAL_SAMPLE_IDS)
    assert len({row["source_group"] for row in cohort}) == 45
    cells, pairs = load_historical_records(cohort)
    assert len(cells) == 200
    assert len(pairs) == 100
    assert np.mean([row["gain_c"] for row in pairs if row["model"] == "Ditto"]) == pytest.approx(1.1238)
    assert np.mean([row["gain_c"] for row in pairs if row["model"] == "LeapTalk"]) == pytest.approx(1.38894)


def test_resource_wait_run_can_resume_but_completed_run_is_immutable(tmp_path, monkeypatch):
    run_root = tmp_path / "lrs3_tts_gain_mechanism_resume"
    run_root.mkdir()
    final_path = run_root / "final.json"
    monkeypatch.setattr(config, "run_root_for", lambda _run_id: run_root)
    monkeypatch.setattr(runner, "report", lambda _paths: {"status": "resumed"})

    write_self_hashed_json(final_path, {"status": "RESOURCE_WAIT"})
    assert runner.run("resume", "report") == {"status": "resumed"}

    write_self_hashed_json(final_path, {"status": "complete"})
    with pytest.raises(ProtocolError, match="terminal run"):
        runner.run("resume", "curves")
