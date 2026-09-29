from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiments.mfa_linear_sync_generalization_200.config import BLOCKED_DATASET_SCALE, ScaleConfig
from scripts.experiments.mfa_linear_sync_generalization_200.protocol import ScaleProtocolError, select_scale_cohorts
from scripts.experiments.mfa_linear_sync_generalization_200.validate import resume_terminal_run, validate_data_manifest, validate_run


def _row(group: str, index: int) -> dict[str, object]:
    sample_id = f"lrs3_{group}_{index:05d}"
    return {
        "sample_id": sample_id,
        "source_group": group,
        "protocol_split": "train",
        "mfa_audio": f"{sample_id}.wav",
        "mfa_audio_sha256": "mfa",
        "natural_target": {"target_offset": 0, "best_second_gap": 0.01, "curve": [1.0] * 31},
        "visual": {"path": f"{sample_id}.avi", "sha256": "video", "decoded_bgr_frame_sha256": ["x"] * 96},
    }


def _manifest(tmp_path: Path) -> Path:
    rows = [_row(f"group-{group:02d}", index) for group in range(60) for index in range(10)]
    select_scale_cohorts(rows, [], tmp_path, excluded_source_groups=[])
    return tmp_path / "01_data_lock/manifest.json"


def test_manifest_tampering_fails_closed(tmp_path: Path) -> None:
    manifest_path = _manifest(tmp_path)
    payload = json.loads(manifest_path.read_text())
    payload["selection_uses_outcomes"] = True
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ScaleProtocolError, match="BLOCKED_DATASET_SCALE"):
        validate_data_manifest(manifest_path)


def test_blocked_run_is_valid_without_downstream_artifacts(tmp_path: Path) -> None:
    decision_path = tmp_path / "decision.json"
    decision = {
        "schema_version": 1,
        "status": BLOCKED_DATASET_SCALE,
        "pass": False,
        "error": "need 40 groups",
        "config": ScaleConfig().to_dict(),
    }
    decision_path.write_text(json.dumps(decision))
    validation = validate_run(tmp_path)
    assert validation["status"] == "valid"
    (tmp_path / "validation.json").write_text(json.dumps(validation))
    assert resume_terminal_run(tmp_path)["status"] == BLOCKED_DATASET_SCALE


def test_blocked_run_rejects_training_artifact(tmp_path: Path) -> None:
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps({
        "schema_version": 1,
        "status": BLOCKED_DATASET_SCALE,
        "pass": False,
        "error": "need more groups",
        "config": ScaleConfig().to_dict(),
    }))
    (tmp_path / "02_training").mkdir()
    validation = validate_run(tmp_path)
    assert validation["status"] == "invalid"
