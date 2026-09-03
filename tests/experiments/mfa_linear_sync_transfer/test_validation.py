from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.experiments.mfa_linear_sync_transfer.config import TransferConfig
from scripts.experiments.mfa_linear_sync_transfer.validate import resume_terminal_run, validate_run
from scripts.experiments.mfa_linear_sync_transfer.protocol import TransferProtocolError


def test_blocked_parent_graph_is_valid_and_resume_is_read_only(tmp_path: Path) -> None:
    decision = {
        "schema_version": 1,
        "status": "BLOCKED_P2_NOT_PASSED",
        "pass": False,
        "engineering_status": "failed",
        "error": "missing parent",
        "config": TransferConfig().to_dict(),
        "p2_prerequisite_status": "BLOCKED_P2_NOT_PASSED",
        "real_video_transfer_status": "REAL_VIDEO_TRANSFER_NOT_EVALUATED",
        "replacement_status": "REPLACEMENT_NOT_EVALUATED",
    }
    (tmp_path / "decision.json").write_text(json.dumps(decision), encoding="utf-8")
    graph = validate_run(tmp_path)
    assert graph["artifact_graph_valid"] is True
    (tmp_path / "validation.json").write_text(json.dumps(graph), encoding="utf-8")
    assert resume_terminal_run(tmp_path)["status"] == "BLOCKED_P2_NOT_PASSED"


def test_resume_rejects_tampered_decision(tmp_path: Path) -> None:
    decision = {
        "schema_version": 1,
        "status": "BLOCKED_P2_NOT_PASSED",
        "pass": False,
        "config": TransferConfig().to_dict(),
    }
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(decision), encoding="utf-8")
    graph = validate_run(tmp_path)
    (tmp_path / "validation.json").write_text(json.dumps(graph), encoding="utf-8")
    path.write_text(json.dumps({**decision, "pass": True}), encoding="utf-8")
    with pytest.raises(TransferProtocolError, match="hash"):
        resume_terminal_run(tmp_path)
