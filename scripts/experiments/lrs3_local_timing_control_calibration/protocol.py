from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import config
from .common import (
    CalibrationError,
    assert_not_sealed,
    assert_run_root_compatible,
    file_sha256,
    sample_ids_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def _bound_artifact(path: Path, expected_sha256: str, name: str) -> dict[str, str]:
    assert_not_sealed(path)
    if not path.is_file():
        raise CalibrationError(f"historical artifact is missing: {name}: {path}")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise CalibrationError(f"historical artifact hash changed: {name}")
    try:
        verify_self_hashed_json(path)
    except (OSError, TypeError, ValueError) as exc:
        raise CalibrationError(f"historical artifact self-hash is invalid: {name}") from exc
    return {"path": str(path.resolve()), "sha256": actual}


def _asset(value: Any, name: str, sample_id: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise CalibrationError(f"historical asset binding is malformed: {sample_id}/{name}")
    path = Path(str(value.get("path", ""))).resolve()
    expected = str(value.get("sha256", ""))
    assert_not_sealed(path)
    if not path.is_file() or file_sha256(path) != expected:
        raise CalibrationError(f"historical asset binding changed: {sample_id}/{name}")
    return {"path": str(path), "sha256": expected}


def load_history() -> dict[str, Any]:
    """Load and bind the completed control run without writing to it."""

    _bound_artifact(config.HISTORY_FINAL, config.HISTORY_FINAL_SHA256, "final")
    final = verify_self_hashed_json(config.HISTORY_FINAL)
    if (
        final.get("status") != "complete"
        or final.get("protocol_id") != config.HISTORY_PROTOCOL_ID
        or final.get("record_count") != config.EXPECTED_RECORD_COUNT
        or final.get("source_group_count") != config.EXPECTED_SOURCE_GROUP_COUNT
        or final.get("scientific_decision") != "CONTROL_FAILED"
        or final.get("reference_conditioned_audio_head_spec_eligible") is not False
    ):
        raise CalibrationError("historical final is not the registered CONTROL_FAILED run")

    bindings = {
        "final": {"path": str(config.HISTORY_FINAL.resolve()), "sha256": config.HISTORY_FINAL_SHA256},
        "cohort": _bound_artifact(config.HISTORY_COHORT, config.HISTORY_COHORT_SHA256, "cohort"),
        "audio_manifest": _bound_artifact(config.HISTORY_AUDIO, config.HISTORY_AUDIO_SHA256, "audio_manifest"),
        "videos_manifest": _bound_artifact(config.HISTORY_VIDEOS, config.HISTORY_VIDEOS_SHA256, "videos_manifest"),
        "scores_manifest": _bound_artifact(config.HISTORY_SCORES, config.HISTORY_SCORES_SHA256, "scores_manifest"),
    }
    expected_from_final = {
        "cohort": final.get("cohort_sha256"),
        "audio_manifest": final.get("audio_manifest_sha256"),
        "videos_manifest": final.get("videos_manifest_sha256"),
        "scores_manifest": final.get("scores_manifest_sha256"),
    }
    for name, expected in expected_from_final.items():
        if bindings[name]["sha256"] != expected:
            raise CalibrationError(f"historical final binding differs: {name}")

    cohort = verify_self_hashed_json(config.HISTORY_COHORT)
    audio = verify_self_hashed_json(config.HISTORY_AUDIO)
    videos = verify_self_hashed_json(config.HISTORY_VIDEOS)
    scores = verify_self_hashed_json(config.HISTORY_SCORES)
    records = cohort.get("records")
    if (
        cohort.get("status") != "complete"
        or not isinstance(records, list)
        or len(records) != config.EXPECTED_RECORD_COUNT
        or cohort.get("sample_ids_sha256") != config.EXPECTED_SAMPLE_ID_SHA256
        or cohort.get("source_group_count") != config.EXPECTED_SOURCE_GROUP_COUNT
    ):
        raise CalibrationError("historical cohort identity is incomplete")
    ids = [str(row.get("sample_id", "")) for row in records]
    groups = [str(row.get("source_group", "")) for row in records]
    if len(set(ids)) != len(ids) or len(set(groups)) != len(groups) or sample_ids_sha256(ids) != config.EXPECTED_SAMPLE_ID_SHA256:
        raise CalibrationError("historical cohort ordering or groups changed")
    for row in records:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or not str(row.get("source_group", "")):
            raise CalibrationError("historical cohort contains an incomplete record")
        for name in ("natural_audio", "mfa_linear_audio", "face_video"):
            _asset(row.get(name), name, sample_id)

    if audio.get("status") != "complete" or audio.get("record_count") != config.EXPECTED_RECORD_COUNT or audio.get("audio_cell_count") != 88:
        raise CalibrationError("historical audio manifest is incomplete")
    if videos.get("status") != "complete" or videos.get("record_count") != config.EXPECTED_RECORD_COUNT or videos.get("video_count") != 88:
        raise CalibrationError("historical video manifest is incomplete")
    if scores.get("status") != "complete" or scores.get("record_count") != config.EXPECTED_RECORD_COUNT or scores.get("cell_count") != 132:
        raise CalibrationError("historical score manifest is incomplete")
    return {"final": final, "cohort": cohort, "audio": audio, "videos": videos, "scores": scores, "bindings": bindings}


def _source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {path.name: file_sha256(path) for path in sorted(package.glob("*.py"))}


def protocol_payload(audit: Mapping[str, Any], history: Mapping[str, Any], branch: str) -> dict[str, Any]:
    if branch not in config.BRANCHES:
        raise CalibrationError(f"unknown selected branch: {branch}")
    control_arm = config.control_arm_for_branch(branch)
    return {
        "schema_version": 1,
        "stage_id": "01_protocol",
        "protocol_id": config.PROTOCOL_ID,
        "status": "locked",
        "experiment": "lrs3-local-timing-control-calibration",
        "classification": "fit_only_control_calibration",
        "branch": branch,
        "control_arm": control_arm,
        "arms": list(config.arms_for_branch(branch)),
        "matrix_cells": list(config.matrix_cells(control_arm)),
        "config": config.FrozenConfig().to_dict(),
        "audit_decision": audit.get("audit_decision"),
        "audit_sha256": audit.get("artifact_sha256"),
        "history": dict(history["bindings"]),
        "history_final_artifact_sha256": history["final"].get("artifact_sha256"),
        "source_hashes": _source_hashes(),
        "selection": {
            "record_count": config.EXPECTED_RECORD_COUNT,
            "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
            "sample_ids_sha256": config.EXPECTED_SAMPLE_ID_SHA256,
            "uses_scores": False,
            "uses_sealed_media": False,
            "source": "bound historical confirmation cohort, preserved in order",
        },
        "sealed_scope": {
            "validation_test_accessed": False,
            "training": False,
            "fine_tuning": False,
            "tts_generation": False,
            "mfa": False,
            "dtw": False,
            "bridge_rescoring": False,
            "score_based_selection": False,
            "score_based_retry": False,
            "branch_switch": False,
            "parent_or_history_overwrite": False,
        },
    }


def run_protocol(paths: config.RunPaths, audit: Mapping[str, Any], history: Mapping[str, Any] | None = None) -> dict[str, Any]:
    assert_run_root_compatible(paths.root)
    if audit.get("audit_decision") == "INCONCLUSIVE":
        raise CalibrationError("audit is inconclusive; protocol is blocked")
    if audit.get("audit_decision") not in {"NO_DEFECT_FOUND", "DEFECT_FOUND"}:
        raise CalibrationError("audit decision is not selectable")
    if audit.get("audit_decision") == "DEFECT_FOUND" and audit.get("repair_verified") is not True:
        raise CalibrationError("a found defect requires a verified minimal repair before REPAIR_ONLY")
    history = history or load_history()
    branch = config.SMOOTH_BRANCH if audit.get("audit_decision") == "NO_DEFECT_FOUND" else config.REPAIR_BRANCH
    payload = protocol_payload(audit, history, branch)
    paths.protocol.mkdir(parents=True, exist_ok=True)
    protocol_path = paths.protocol / "protocol.json"
    if protocol_path.is_file():
        existing = verify_self_hashed_json(protocol_path)
        existing_body = dict(existing)
        existing_body.pop("artifact_sha256", None)
        if existing_body != payload:
            raise CalibrationError("existing protocol binding differs")
    else:
        write_self_hashed_json(protocol_path, payload)
    protocol = verify_self_hashed_json(protocol_path)
    protocol["_path"] = str(protocol_path)

    cohort_payload = {
        "schema_version": 1,
        "stage_id": "01_protocol",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "classification": "fit_only_control_calibration",
        "branch": branch,
        "control_arm": config.control_arm_for_branch(branch),
        "record_count": config.EXPECTED_RECORD_COUNT,
        "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
        "sample_ids_sha256": config.EXPECTED_SAMPLE_ID_SHA256,
        "selection": "bound historical confirmation cohort preserved in order",
        "records": history["cohort"]["records"],
        "history_cohort_sha256": history["bindings"]["cohort"]["sha256"],
        "history_final_sha256": history["bindings"]["final"]["sha256"],
    }
    cohort_path = paths.protocol / "cohort.json"
    if cohort_path.is_file():
        existing = verify_self_hashed_json(cohort_path)
        body = dict(existing)
        body.pop("artifact_sha256", None)
        if body != cohort_payload:
            raise CalibrationError("existing calibration cohort binding differs")
    else:
        write_self_hashed_json(cohort_path, cohort_payload)
    write_self_hashed_json(
        paths.protocol / "access.json",
        {
            "schema_version": 1,
            "stage_id": "01_protocol",
            "protocol_id": config.PROTOCOL_ID,
            "status": "locked",
            "history_read_only": True,
            "new_audio_decode_count": 0,
            "new_video_decode_count": 0,
            "new_score_read_count": 0,
            "sealed_media_accessed": False,
            "branch_switch": False,
        },
    )
    return protocol


def load_protocol(paths: config.RunPaths) -> dict[str, Any]:
    path = paths.protocol / "protocol.json"
    payload = verify_self_hashed_json(path)
    if payload.get("protocol_id") != config.PROTOCOL_ID or payload.get("status") != "locked":
        raise CalibrationError("calibration protocol is not locked")
    payload["_path"] = str(path)
    return payload


def load_cohort(paths: config.RunPaths) -> dict[str, Any]:
    payload = verify_self_hashed_json(paths.protocol / "cohort.json")
    if payload.get("protocol_id") != config.PROTOCOL_ID or payload.get("status") != "complete":
        raise CalibrationError("calibration cohort is incomplete")
    return payload
