"""Fail-closed validation for the post-hoc diagnostic."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..mfa_linear_sync_existing_data.validate import validate_manifest
from .protocol import read_object, sha256_file
from .config import (
    DIAGNOSTIC_COMPLETE,
    DIAGNOSTIC_NOT_EVALUATED,
    DIAGNOSTIC_PARENT_INVALID,
    EXPECTED_MATRIX_CELLS,
    REPLACEMENT_NOT_EVALUATED,
    REPLACEMENT_NOT_RUN,
    REPLACEMENT_OBSERVED,
    DiagnosticConfig,
)
from .protocol import sha256_file


def validate_run(root: str | Path) -> dict[str, Any]:
    run = Path(root).resolve()
    decision_path = run / "decision.json"
    decision = read_object(decision_path)
    status = str(decision.get("status", ""))
    checks: dict[str, bool] = {
        "terminal_decision": decision_path.is_file(),
        "pass_is_false": decision.get("pass") is False,
        "config": decision.get("config") == DiagnosticConfig().to_dict(),
        "recognized_status": status in {DIAGNOSTIC_PARENT_INVALID, DIAGNOSTIC_NOT_EVALUATED, DIAGNOSTIC_COMPLETE},
    }
    if status == DIAGNOSTIC_PARENT_INVALID:
        checks.update({
            "error_recorded": isinstance(decision.get("error"), str) and bool(decision["error"]),
            "no_candidates": not (run / "02_candidates").exists(),
            "no_real_video": not (run / "03_real_video").exists(),
            "no_replacement": not (run / "04_replacement").exists(),
        })
    else:
        checks["parent_lock"] = (run / "01_parent/parent_lock.json").is_file()
        checks["parent_manifest"] = (run / "01_parent/manifest.json").is_file()
        if checks["parent_manifest"]:
            manifest = read_object(run / "01_parent/manifest.json")
            checks["parent_manifest"] = len(manifest.get("evaluation", [])) == 8
        if status == DIAGNOSTIC_NOT_EVALUATED:
            checks["diagnostic_decision"] = (run / "03_real_video/decision.json").is_file()
        else:
            candidate_files = list((run / "02_candidates").glob("*/candidate.json")) if (run / "02_candidates").is_dir() else []
            checks["candidate_denominator"] = len(candidate_files) == 8 or status == DIAGNOSTIC_NOT_EVALUATED
            checks["diagnostic_decision"] = (run / "03_real_video/decision.json").is_file()
        replacement_status = str(decision.get("replacement_status", REPLACEMENT_NOT_EVALUATED))
        if replacement_status in {REPLACEMENT_OBSERVED, REPLACEMENT_NOT_EVALUATED, REPLACEMENT_NOT_RUN}:
            cells_dir = run / "04_replacement/matrix"
            cells = list(cells_dir.glob("*/**/cell.json")) if cells_dir.is_dir() else []
            has_replacement_decision = (run / "04_replacement/decision.json").is_file()
            if cells_dir.is_dir() or has_replacement_decision:
                checks["replacement_cells"] = len(cells) == EXPECTED_MATRIX_CELLS
                checks["replacement_decision"] = has_replacement_decision
            else:
                checks["replacement_not_run"] = replacement_status == REPLACEMENT_NOT_RUN
        else:
            checks["replacement_not_run"] = not (run / "04_replacement/matrix").exists()
    valid = all(checks.values())
    return {
        "status": "valid" if valid else "invalid",
        "artifact_graph_valid": valid,
        "run": str(run),
        "terminal_status": status,
        "checks": checks,
        "decision_sha256": sha256_file(decision_path),
    }


def resume_terminal_run(root: str | Path) -> dict[str, Any]:
    run = Path(root).resolve()
    validation = read_object(run / "validation.json")
    decision = read_object(run / "decision.json")
    if validation.get("status") != "valid" or validation.get("artifact_graph_valid") is not True:
        raise ValueError("resume requires a complete hash-valid diagnostic run")
    if validation.get("decision_sha256") != sha256_file(run / "decision.json"):
        raise ValueError("resume terminal decision hash mismatch")
    if validation.get("terminal_status") != decision.get("status"):
        raise ValueError("resume terminal status mismatch")
    return decision
