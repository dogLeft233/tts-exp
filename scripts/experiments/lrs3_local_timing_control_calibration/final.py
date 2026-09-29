from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import config
from .common import CalibrationError, file_sha256, write_self_hashed_json


def _result_markdown(protocol: Mapping[str, Any], analysis: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    branch = str(protocol["branch"])
    control_arm = str(protocol["control_arm"])
    controls = analysis["controls"]
    audit_summary = audit.get("summary", {})
    lines = [
        "# LRS3 local timing control calibration",
        "",
        f"- classification: `{analysis['classification']}`",
        f"- audit: `{audit.get('audit_decision')}`; evidence items: {audit_summary.get('pass', 0)} pass / {audit_summary.get('fail', 0)} fail / {audit_summary.get('inconclusive', 0)} inconclusive",
        f"- audit artifact SHA-256: `{audit.get('artifact_sha256')}`",
        f"- branch: `{branch}`; control arm: `{control_arm}`",
        f"- records/groups: {analysis['record_count']}/{analysis['source_group_count']}",
        f"- scientific decision: `{analysis['decisions']['scientific_decision']}`",
        "- reference-conditioned audio-head eligibility: `false`",
        "",
        "| Gate | Pass |",
        "|---|---:|",
        f"| N_REPEAT repeatability | {controls['N_REPEAT']['passes']} |",
        f"| {control_arm} own-audio validity | {controls[control_arm]['own_audio_validity']['passes']} |",
        f"| {control_arm} replacement sensitivity | {controls[control_arm]['sensitivity']['passes']} |",
        "",
        "本结果只校准固定 fit-only cohort 与固定 SyncNet endpoint。校准成功也不构成 bridge 收益证明；后续 bridge 收益需要单独的 OpenSpec。",
    ]
    return "\n".join(lines) + "\n"


def finalize(protocol: Mapping[str, Any], cohort: Mapping[str, Any], audio: Mapping[str, Any], videos: Mapping[str, Any], scores: Mapping[str, Any], analysis: Mapping[str, Any], paths: config.RunPaths, audit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if protocol.get("status") != "locked" or any(payload.get("status") != "complete" for payload in (cohort, audio, videos, scores, analysis)):
        raise CalibrationError("cannot finalize an incomplete calibration run")
    decision = str(analysis.get("decisions", {}).get("scientific_decision", ""))
    if decision not in {"CONTROL_FAILED", "CONTROL_CALIBRATED"}:
        raise CalibrationError(f"invalid calibration decision: {decision}")
    protocol_path = paths.protocol / "protocol.json"
    cohort_path = paths.protocol / "cohort.json"
    audio_path = paths.audio / "audio_manifest.json"
    videos_path = paths.videos / "videos_manifest.json"
    scores_path = paths.scores / "scores_manifest.json"
    analysis_path = paths.final / "analysis.json"
    audit = audit or {}
    terminal = {
        "schema_version": 1,
        "stage_id": "05_final",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "classification": "fit_only_control_calibration",
        "engineering_decision": "GO",
        "scientific_decision": decision,
        "reference_conditioned_audio_head_spec_eligible": False,
        "audit_decision": audit.get("audit_decision", protocol.get("audit_decision")),
        "audit_item_count": len(audit.get("items", [])) if isinstance(audit.get("items"), list) else None,
        "branch": protocol["branch"],
        "control_arm": protocol["control_arm"],
        "scope": "fixed 22-record/22-source-group LRS3 fit-only control calibration on the registered endpoint",
        "limitations": [
            "reuses the historical 22-record fit-only cohort",
            "calibrates only the selected transform and exact Wav2Lip/SyncNet endpoint",
            "does not rescore BRIDGE_075 or prove a bridge gain",
            "does not authorize training, deployment, or held-out evaluation",
        ],
        "history_final_sha256": config.HISTORY_FINAL_SHA256,
        "history_cohort_sha256": config.HISTORY_COHORT_SHA256,
        "history_audio_sha256": config.HISTORY_AUDIO_SHA256,
        "history_videos_sha256": config.HISTORY_VIDEOS_SHA256,
        "history_scores_sha256": config.HISTORY_SCORES_SHA256,
        "audit_sha256": file_sha256(paths.audit / "audit.json"),
        "protocol_sha256": file_sha256(protocol_path),
        "cohort_sha256": file_sha256(cohort_path),
        "audio_manifest_sha256": file_sha256(audio_path),
        "videos_manifest_sha256": file_sha256(videos_path),
        "scores_manifest_sha256": file_sha256(scores_path),
        "analysis_sha256": file_sha256(analysis_path),
        "record_count": config.EXPECTED_RECORD_COUNT,
        "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
        "video_count": config.expected_video_count(str(protocol["branch"])),
        "cell_count": config.expected_cell_count(str(protocol["branch"])),
        "controls": analysis["controls"],
        "bootstrap": analysis["bootstrap"],
        "sealed_scope": protocol["sealed_scope"],
    }
    paths.final.mkdir(parents=True, exist_ok=True)
    write_self_hashed_json(paths.final / "final.json", terminal)
    (paths.final / "result.md").write_text(_result_markdown(protocol, analysis, audit), encoding="utf-8")
    return terminal


def write_blocked_terminal(paths: config.RunPaths, error: Exception, failed_stage: str | None = None) -> dict[str, Any]:
    terminal = {
        "schema_version": 1,
        "stage_id": "05_final",
        "protocol_id": config.PROTOCOL_ID,
        "status": "blocked",
        "classification": "fit_only_control_calibration",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "BLOCKED",
        "reference_conditioned_audio_head_spec_eligible": False,
        "failed_stage": failed_stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "run_root": str(paths.root),
        "scope": "fixed historical fit-only cohort; no scientific result issued",
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
    paths.final.mkdir(parents=True, exist_ok=True)
    write_self_hashed_json(paths.final / "final.json", terminal)
    return terminal
