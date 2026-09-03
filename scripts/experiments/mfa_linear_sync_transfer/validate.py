"""Recompute the follow-up artifact graph without inference or rendering."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .config import (
    BLOCKED_COHORT,
    BLOCKED_P2,
    COHORT_SIZE,
    EXPECTED_MATRIX_CELLS,
    NO_REAL_VIDEO_TRANSFER,
    NO_REPLACEMENT,
    REAL_VIDEO_NOT_EVALUATED,
    REAL_VIDEO_TRANSFER,
    REPLACEMENT_GATE_FAILED,
    REPLACEMENT_NOT_EVALUATED,
    REPLACEMENT_OBSERVED,
    SELECTION_SALT,
    TransferConfig,
)
from .protocol import TransferProtocolError, read_object, selection_key, sha256_file


def _check_manifest(manifest: Mapping[str, Any]) -> dict[str, bool]:
    selected = manifest.get("selected", [])
    ids = manifest.get("selected_sample_ids", [])
    groups = manifest.get("selected_source_groups", [])
    rows = manifest.get("eligible_universe", [])
    return {
        "schema_version": manifest.get("schema_version") == 1,
        "complete": manifest.get("status") == "complete",
        "salt": manifest.get("selection_salt") == SELECTION_SALT,
        "outcome_independent": manifest.get("selection_uses_outcomes") is False,
        "denominator": manifest.get("denominator") == COHORT_SIZE and len(selected) == COHORT_SIZE,
        "unique_ids": len(ids) == len(set(ids)) == COHORT_SIZE,
        "unique_groups": len(groups) == len(set(groups)) == COHORT_SIZE,
        "p2_disjoint": not set(groups).intersection(set(manifest.get("p2_source_groups", []))),
        "selected_bindings": all(
            str(row.get("sample_id")) == str(ids[index])
            and str(row.get("source_group")) == str(groups[index])
            and str(row.get("selection_key")) == selection_key(str(row.get("source_group")), str(row.get("sample_id")))
            for index, row in enumerate(selected)
        ),
        "universe_has_no_outcome_fields": not any(
            any(token in key.lower() for token in ("candidate", "loss", "score", "wav2lip"))
            for row in rows if isinstance(row, Mapping) for key in row
        ),
    }


def validate_cohort_manifest(path: str | Path) -> dict[str, Any]:
    manifest = read_object(path)
    checks = _check_manifest(manifest)
    valid = all(checks.values())
    if not valid:
        raise TransferProtocolError(f"{BLOCKED_COHORT}: cohort manifest failed validation: {checks}")
    return {"status": "valid", "artifact_graph_valid": True, "checks": checks, "manifest_sha256": sha256_file(path)}


def _json(path: Path) -> dict[str, Any]:
    return read_object(path)


def validate_run(root: str | Path) -> dict[str, Any]:
    run = Path(root).resolve()
    decision_path = run / "decision.json"
    decision = _json(decision_path)
    status = str(decision.get("status", ""))
    checks: dict[str, bool] = {
        "terminal_decision": decision_path.is_file(),
        "boolean_pass": isinstance(decision.get("pass"), bool),
        "config": decision.get("config") == TransferConfig().to_dict(),
    }
    if status == BLOCKED_P2:
        checks.update({
            "no_cohort": not (run / "01_cohort_lock/manifest.json").exists(),
            "no_candidates": not (run / "02_candidates").exists(),
            "no_replacement": not (run / "04_replacement").exists(),
        })
    elif status == BLOCKED_COHORT:
        checks.update({
            "cohort_failure_recorded": isinstance(decision.get("error"), str) and bool(decision["error"]),
            "no_candidates": not (run / "02_candidates").exists(),
            "no_replacement": not (run / "04_replacement").exists(),
        })
    elif status in {REAL_VIDEO_TRANSFER, NO_REAL_VIDEO_TRANSFER, REAL_VIDEO_NOT_EVALUATED, REPLACEMENT_GATE_FAILED}:
        manifest_path = run / "01_cohort_lock/manifest.json"
        checks.update({"cohort": False})
        if manifest_path.is_file():
            checks["cohort"] = all(validate_cohort_manifest(manifest_path)["checks"].values())
        checks["real_video_decision"] = (run / "03_real_video/decision.json").is_file()
        checks["no_replacement_after_gate_miss"] = status != NO_REAL_VIDEO_TRANSFER or not (run / "04_replacement/matrix").exists()
    elif status in {REPLACEMENT_OBSERVED, NO_REPLACEMENT, REPLACEMENT_NOT_EVALUATED}:
        manifest_path = run / "01_cohort_lock/manifest.json"
        checks["cohort"] = manifest_path.is_file() and all(validate_cohort_manifest(manifest_path)["checks"].values())
        checks["real_video_decision"] = (run / "03_real_video/decision.json").is_file()
        matrix = run / "04_replacement/matrix"
        cells = list(matrix.glob("*/**/cell.json")) if matrix.is_dir() else []
        checks["matrix_denominator"] = len(cells) == EXPECTED_MATRIX_CELLS or status == REPLACEMENT_NOT_EVALUATED
        checks["replacement_decision"] = (run / "04_replacement/decision.json").is_file() or status == REPLACEMENT_NOT_EVALUATED
    else:
        checks["recognized_status"] = False
    valid = all(checks.values())
    return {
        "status": "valid" if valid else "invalid",
        "artifact_graph_valid": valid,
        "run": str(run),
        "terminal_status": status,
        "checks": checks,
        "decision_sha256": sha256_file(decision_path),
        "scientific_pass": bool(decision.get("pass", False)),
    }


def resume_terminal_run(root: str | Path) -> dict[str, Any]:
    run = Path(root).resolve()
    validation = _json(run / "validation.json")
    decision = _json(run / "decision.json")
    if validation.get("status") != "valid" or validation.get("artifact_graph_valid") is not True:
        raise TransferProtocolError("resume requires a complete hash-valid terminal run")
    if validation.get("decision_sha256") != sha256_file(run / "decision.json"):
        raise TransferProtocolError("resume terminal decision hash mismatch")
    if validation.get("terminal_status") != decision.get("status"):
        raise TransferProtocolError("resume terminal status mismatch")
    return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    result = validate_run(args.run)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["artifact_graph_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
