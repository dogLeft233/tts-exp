from pathlib import Path

from scripts.experiments.lrs3_local_timing_control_calibration import config
from scripts.experiments.lrs3_local_timing_control_calibration.common import (
    write_self_hashed_json,
)
from scripts.experiments.lrs3_local_timing_control_calibration.stats import (
    analyze_scores,
    cluster_bootstrap,
)


def test_cluster_bootstrap_is_deterministic_and_grouped() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    groups = ["g1", "g2", "g3", "g4"]
    first = cluster_bootstrap(values, groups)
    second = cluster_bootstrap(values, groups)
    assert first == second
    assert first["mean"] == 2.5
    assert first["draws"] == config.BOOTSTRAP_DRAWS
    assert first["seed"] == config.BOOTSTRAP_SEED
    assert first["source_group_count"] == 4


def _synthetic_inputs(tmp_path: Path, own_c: float, replacement_c: float) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    records = [{"sample_id": f"s{index}", "source_group": f"g{index}"} for index in range(config.EXPECTED_RECORD_COUNT)]
    cohort = {"status": "complete", "records": records}
    protocol = {"status": "locked", "branch": config.SMOOTH_BRANCH, "control_arm": config.SMOOTH_WARP_ARM}
    rows: list[dict[str, object]] = []
    cells = config.matrix_cells(config.SMOOTH_WARP_ARM)
    for record in records:
        sample_id = str(record["sample_id"])
        values = {
            cells[0]: (5.0, 8.0, 0),
            cells[1]: (5.0, 8.0, 0),
            cells[2]: (own_c, 8.0 - (own_c - 5.0), 0),
            cells[3]: (replacement_c, 8.0 + (5.0 - replacement_c), 0),
        }
        for cell, (sync_c, sync_d, offset) in values.items():
            rows.append({"sample_id": sample_id, "cell": cell, "sync_c": sync_c, "sync_d": sync_d, "av_offset": offset})
    score_path = tmp_path / "04_scores/scores_manifest.json"
    write_self_hashed_json(score_path, {"status": "complete", "stage_id": "04_scores", "protocol_id": config.PROTOCOL_ID, "scores": rows})
    return cohort, protocol, {"status": "complete", "scores": rows}


def test_analysis_reports_all_gates_and_can_calibrate(tmp_path: Path) -> None:
    cohort, protocol, scores = _synthetic_inputs(tmp_path, own_c=5.2, replacement_c=4.0)
    result = analyze_scores(cohort, protocol, scores, tmp_path / "05_final")
    assert result["decisions"]["scientific_decision"] == "CONTROL_CALIBRATED"
    assert result["decisions"]["reference_conditioned_audio_head_spec_eligible"] is False
    assert result["controls"][config.SMOOTH_WARP_ARM]["sensitivity"]["passes"] is True


def test_analysis_keeps_control_failure_when_replacement_is_better(tmp_path: Path) -> None:
    cohort, protocol, scores = _synthetic_inputs(tmp_path, own_c=4.0, replacement_c=5.2)
    result = analyze_scores(cohort, protocol, scores, tmp_path / "05_final")
    assert result["decisions"]["scientific_decision"] == "CONTROL_FAILED"
    assert result["controls"][config.SMOOTH_WARP_ARM]["sensitivity"]["passes"] is False


def test_own_validity_can_fail_while_sensitivity_passes(tmp_path: Path) -> None:
    cohort, protocol, scores = _synthetic_inputs(tmp_path, own_c=4.0, replacement_c=3.0)
    result = analyze_scores(cohort, protocol, scores, tmp_path / "05_final")
    control = result["controls"][config.SMOOTH_WARP_ARM]
    assert control["own_audio_validity"]["passes"] is False
    assert control["sensitivity"]["passes"] is True
    assert result["decisions"]["scientific_decision"] == "CONTROL_FAILED"


def test_strict_non_inferiority_boundary_does_not_pass(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "REPLACEMENT_MARGIN", -0.09999999999999964)
    cohort, protocol, scores = _synthetic_inputs(tmp_path, own_c=4.9, replacement_c=3.9)
    result = analyze_scores(cohort, protocol, scores, tmp_path / "05_final")
    assert result["controls"][config.SMOOTH_WARP_ARM]["own_audio_validity"]["passes"] is False
