"""Fail-closed validation and read-only resume for the scale run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..mfa_linear_sync_transfer.protocol import read_object
from ..mfa_linear_sync_transfer.protocol import sha256_file
from .config import (
    BLOCKED_DATASET_SCALE,
    EVAL_RECORD_COUNT,
    EXPECTED_MATRIX_CELLS,
    NO_REAL_VIDEO_TRANSFER,
    REAL_VIDEO_NOT_EVALUATED,
    REAL_VIDEO_TRANSFER,
    REPLACEMENT_GATE_FAILED,
    REPLACEMENT_NOT_EVALUATED,
    REPLACEMENT_OBSERVED,
    ScaleConfig,
    TRAIN_RECORD_COUNT,
)
from .protocol import ScaleProtocolError, group_key, selection_key


def validate_data_manifest(path: str | Path) -> dict[str, Any]:
    manifest = read_object(path)
    training = manifest.get("training", [])
    evaluation = manifest.get("evaluation", [])
    train_ids = [str(row.get("sample_id")) for row in training]
    eval_ids = [str(row.get("sample_id")) for row in evaluation]
    train_groups = {str(row.get("source_group")) for row in training}
    eval_groups = [str(row.get("source_group")) for row in evaluation]
    checks = {
        "schema_version": manifest.get("schema_version") == 1,
        "complete": manifest.get("status") == "complete",
        "manifest_type": manifest.get("manifest_type") == "mfa_linear_sync_generalization_scale_200",
        "selection_salt": manifest.get("selection_salt") == ScaleConfig().selection_salt,
        "outcome_independent": manifest.get("selection_uses_outcomes") is False,
        "sealed_splits_closed": manifest.get("sealed_splits_accessed") is False,
        "training_denominator": len(training) == TRAIN_RECORD_COUNT and manifest.get("training_record_count") == TRAIN_RECORD_COUNT,
        "evaluation_denominator": len(evaluation) == EVAL_RECORD_COUNT and manifest.get("evaluation_record_count") == EVAL_RECORD_COUNT,
        "training_ids_unique": len(train_ids) == len(set(train_ids)) == TRAIN_RECORD_COUNT,
        "evaluation_ids_unique": len(eval_ids) == len(set(eval_ids)) == EVAL_RECORD_COUNT,
        "training_split": all(row.get("protocol_split") == "train" for row in training),
        "evaluation_split": all(row.get("protocol_split") == "train" for row in evaluation),
        "evaluation_groups_unique": len(eval_groups) == len(set(eval_groups)) == EVAL_RECORD_COUNT,
        "evaluation_group_order": eval_groups == sorted(
            eval_groups, key=lambda group: (bytes.fromhex(group_key(group)), group.encode())
        ),
        "group_disjoint": not train_groups.intersection(eval_groups),
        "training_group_bindings": manifest.get("training_source_groups") == sorted(train_groups),
        "evaluation_group_bindings": manifest.get("evaluation_groups") == eval_groups,
        "training_selection_keys": all(
            str(row.get("selection_key")) == selection_key(
                str(row.get("source_group")), str(row.get("sample_id")), role="train"
            )
            for row in training
        ),
        "evaluation_group_keys": all(
            str(row.get("source_group")) in eval_groups
            and str(row.get("sample_id")) == eval_ids[index]
            and str(row.get("selection_key")) == selection_key(
                str(row.get("source_group")), str(row.get("sample_id")), role="evaluation"
            )
            for index, row in enumerate(evaluation)
        ),
        "universe_has_no_outcome_fields": not any(
            any(token in key.lower() for token in ("candidate", "loss", "score", "wav2lip"))
            for row in manifest.get("eligible_universe", [])
            if isinstance(row, Mapping)
            for key in row
        ),
        "no_natural_waveform_fields": not any(
            "natural_audio_values" in row for row in training + evaluation if isinstance(row, Mapping)
        ),
    }
    if not all(checks.values()):
        raise ScaleProtocolError(f"{BLOCKED_DATASET_SCALE}: manifest validation failed: {checks}")
    return {
        "status": "valid",
        "artifact_graph_valid": True,
        "checks": checks,
        "manifest_sha256": sha256_file(path),
    }


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
        "config": decision.get("config") == ScaleConfig().to_dict(),
        "recognized_status": status in {
            BLOCKED_DATASET_SCALE, REAL_VIDEO_TRANSFER, NO_REAL_VIDEO_TRANSFER,
            REAL_VIDEO_NOT_EVALUATED, REPLACEMENT_GATE_FAILED, REPLACEMENT_OBSERVED,
            REPLACEMENT_NOT_EVALUATED,
        },
    }
    if status == BLOCKED_DATASET_SCALE:
        checks.update({
            "error_recorded": isinstance(decision.get("error"), str) and bool(decision["error"]),
            "no_data_manifest": not (run / "01_data_lock/manifest.json").exists(),
            "no_training_checkpoint": not (run / "02_training").exists(),
            "no_candidates": not (run / "03_candidates").exists(),
            "no_replacement": not (run / "04_replacement").exists(),
        })
    else:
        manifest_path = run / "01_data_lock/manifest.json"
        checks["data_manifest"] = manifest_path.is_file()
        if manifest_path.is_file():
            checks["data_manifest"] = all(validate_data_manifest(manifest_path)["checks"].values())
        checks["real_video_decision"] = (run / "05_real_video/decision.json").is_file()
        if status in {REPLACEMENT_OBSERVED, REPLACEMENT_NOT_EVALUATED}:
            matrix = run / "06_replacement/matrix"
            cells = list(matrix.glob("*/**/cell.json")) if matrix.is_dir() else []
            checks["matrix_denominator"] = len(cells) == EXPECTED_MATRIX_CELLS or status == REPLACEMENT_NOT_EVALUATED
            checks["replacement_decision"] = (run / "06_replacement/decision.json").is_file() or status == REPLACEMENT_NOT_EVALUATED
        else:
            checks["no_replacement_after_gate_miss"] = not (run / "06_replacement").exists()
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
        raise ScaleProtocolError("resume requires a complete hash-valid terminal run")
    if validation.get("decision_sha256") != sha256_file(run / "decision.json"):
        raise ScaleProtocolError("resume terminal decision hash mismatch")
    if validation.get("terminal_status") != decision.get("status"):
        raise ScaleProtocolError("resume terminal status mismatch")
    return decision
