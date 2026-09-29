from pathlib import Path

import pytest

from scripts.experiments.lrs3_local_timing_control_calibration import config
from scripts.experiments.lrs3_local_timing_control_calibration.common import (
    CalibrationError,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from scripts.experiments.lrs3_local_timing_control_calibration.protocol import (
    run_protocol,
)


def _history() -> dict[str, object]:
    return {
        "final": {"artifact_sha256": "final"},
        "cohort": {"records": []},
        "bindings": {
            name: {"path": f"/tmp/{name}.json", "sha256": name}
            for name in ("final", "cohort", "audio_manifest", "videos_manifest", "scores_manifest")
        },
    }


def test_protocol_selects_smooth_branch_only_after_clean_audit(tmp_path: Path) -> None:
    paths = config.RunPaths(tmp_path / "run")
    audit = {"audit_decision": "NO_DEFECT_FOUND", "artifact_sha256": "audit"}
    result = run_protocol(paths, audit, _history())
    assert result["branch"] == config.SMOOTH_BRANCH
    assert result["control_arm"] == config.SMOOTH_WARP_ARM
    assert result["matrix_cells"] == list(config.matrix_cells(config.SMOOTH_WARP_ARM))


def test_protocol_does_not_enter_repair_branch_without_verified_repair(tmp_path: Path) -> None:
    paths = config.RunPaths(tmp_path / "run")
    with pytest.raises(CalibrationError, match="verified minimal repair"):
        run_protocol(paths, {"audit_decision": "DEFECT_FOUND", "artifact_sha256": "audit"}, _history())


def test_run_id_does_not_allow_path_traversal() -> None:
    with pytest.raises(ValueError, match="run-id"):
        config.run_root_for("../outside")


def test_runner_writes_blocked_terminal_for_a_new_failed_run(tmp_path: Path, monkeypatch) -> None:
    from scripts.experiments.lrs3_local_timing_control_calibration import runner

    monkeypatch.setattr(config, "REPO", tmp_path)

    def fail(_stage, _paths):
        raise CalibrationError("synthetic failure")

    monkeypatch.setattr(runner, "_run_stage", fail)
    with pytest.raises(CalibrationError, match="synthetic failure"):
        runner.run("new-failure", "audit")
    terminal = verify_self_hashed_json(config.run_root_for("new-failure") / "05_final/final.json")
    assert terminal["status"] == "blocked"


def test_runner_preserves_an_existing_terminal_artifact_on_failure(tmp_path: Path, monkeypatch) -> None:
    from scripts.experiments.lrs3_local_timing_control_calibration import runner

    monkeypatch.setattr(config, "REPO", tmp_path)
    root = config.run_root_for("preserve-terminal")
    write_self_hashed_json(root / "00_audit/audit.json", {"stage_id": "00_audit", "protocol_id": config.PROTOCOL_ID})
    final_path = root / "05_final/final.json"
    write_self_hashed_json(final_path, {"stage_id": "05_final", "protocol_id": config.PROTOCOL_ID, "status": "complete"})
    before = final_path.read_bytes()

    def fail(_stage, _paths):
        raise CalibrationError("later failure")

    monkeypatch.setattr(runner, "_run_stage", fail)
    with pytest.raises(CalibrationError, match="later failure"):
        runner.run("preserve-terminal", "audit")
    assert final_path.read_bytes() == before
