from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiments.mfa_linear_sync_existing_data.config import BLOCKED_EXISTING_DATA, ExistingDataConfig
from scripts.experiments.mfa_linear_sync_existing_data.protocol import ExistingDataProtocolError, select_cohort
from scripts.experiments.mfa_linear_sync_existing_data.validate import resume_terminal_run, validate_manifest, validate_run


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


def test_manifest_tampering_fails_closed(tmp_path: Path) -> None:
    rows = [_row("group-00", index) for index in range(12)]
    rows += [_row(f"group-{group:02d}", index) for group in range(1, 10) for index in range(10)]
    select_cohort(rows, [], tmp_path, excluded_source_groups=[])
    path = tmp_path / "01_data_lock/manifest.json"
    payload = json.loads(path.read_text())
    payload["selection_uses_outcomes"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ExistingDataProtocolError, match="BLOCKED_EXISTING_DATA"):
        validate_manifest(path)


def test_blocked_run_validates_and_resumes_read_only(tmp_path: Path) -> None:
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps({
        "schema_version": 1,
        "status": BLOCKED_EXISTING_DATA,
        "pass": False,
        "error": "inventory incomplete",
        "config": ExistingDataConfig().to_dict(),
    }))
    validation = validate_run(tmp_path)
    assert validation["status"] == "valid"
    (tmp_path / "validation.json").write_text(json.dumps(validation))
    assert resume_terminal_run(tmp_path)["status"] == BLOCKED_EXISTING_DATA


def test_blocked_run_rejects_training_artifact(tmp_path: Path) -> None:
    (tmp_path / "decision.json").write_text(json.dumps({
        "schema_version": 1,
        "status": BLOCKED_EXISTING_DATA,
        "pass": False,
        "error": "inventory incomplete",
        "config": ExistingDataConfig().to_dict(),
    }))
    (tmp_path / "02_training").mkdir()
    assert validate_run(tmp_path)["status"] == "invalid"
