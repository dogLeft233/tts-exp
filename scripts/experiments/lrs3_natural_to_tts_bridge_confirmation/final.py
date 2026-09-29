from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import config
from .common import file_sha256, write_self_hashed_json
from .protocol import ProtocolError


def run_stage04_final(
    protocol: Mapping[str, Any],
    cohort: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    matrix: Mapping[str, Any],
    analysis: Mapping[str, Any],
    output_stage: Path,
) -> dict[str, Any]:
    if (
        protocol.get("protocol_id") != config.PROTOCOL_ID
        or cohort.get("status") != "complete"
        or diagnostics.get("status") != "complete"
        or matrix.get("status") != "complete"
        or analysis.get("status") != "complete"
    ):
        raise ProtocolError("cannot finalize an incomplete natural-to-TTS bridge run")
    decision = str(analysis.get("decisions", {}).get("scientific_decision", ""))
    allowed = {"CONTROL_FAILED", "NATURAL_TO_TTS_BRIDGE_CONFIRMED", "NATURAL_TO_TTS_BRIDGE_NOT_CONFIRMED"}
    if decision not in allowed:
        raise ProtocolError(f"unknown terminal scientific decision: {decision}")
    terminal = {
        "schema_version": 1,
        "stage_id": "04_final",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": decision,
        "reference_conditioned_audio_head_spec_eligible": decision == "NATURAL_TO_TTS_BRIDGE_CONFIRMED",
        "scope": analysis["decisions"]["scope"],
        "limitations": [
            "fit-only fixed 22-record/22-source-group confirmation cohort",
            "result applies only to BRIDGE_075 and the exact registered media endpoint",
            "TTS-direction movement is not speaker-identity transfer",
            "a positive result authorizes a separate spec but no model training, deployment, or held-out evaluation",
        ],
        "parent_protocol_sha256": config.EXPECTED_PARENT_PROTOCOL_SHA256,
        "parent_replacement_sha256": config.EXPECTED_PARENT_REPLACEMENT_SHA256,
        "discovery_final_sha256": config.EXPECTED_DISCOVERY_FINAL_SHA256,
        "protocol_sha256": file_sha256(config.STAGES["00_protocol"] / "protocol.json"),
        "cohort_sha256": file_sha256(config.STAGES["00_protocol"] / "cohort.json"),
        "audio_manifest_sha256": file_sha256(config.STAGES["01_audio"] / "audio_manifest.json"),
        "diagnostics_sha256": file_sha256(config.STAGES["01_audio"] / "diagnostics.json"),
        "videos_manifest_sha256": file_sha256(config.STAGES["02_videos"] / "videos_manifest.json"),
        "scores_manifest_sha256": file_sha256(config.STAGES["03_scores"] / "scores_manifest.json"),
        "record_count": config.EXPECTED_RECORD_COUNT,
        "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
        "video_count": config.EXPECTED_VIDEO_COUNT,
        "cell_count": config.EXPECTED_CELL_COUNT,
        "controls": analysis["controls"],
        "bridge": analysis["bridge"],
        "bootstrap": analysis["bootstrap"],
        "sealed_scope": protocol["sealed_scope"],
        "analysis_sha256": file_sha256(output_stage / "analysis.json"),
    }
    write_self_hashed_json(output_stage / "final.json", terminal)
    return terminal


def write_blocked_terminal(error: Exception, output_stage: Path = config.STAGES["04_final"]) -> dict[str, Any]:
    terminal = {
        "schema_version": 1,
        "stage_id": "04_final",
        "protocol_id": config.PROTOCOL_ID,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "BLOCKED",
        "reference_conditioned_audio_head_spec_eligible": False,
        "error_type": type(error).__name__,
        "error": str(error),
        "sealed_scope": {
            "validation_test_accessed": False,
            "training": False,
            "fine_tuning": False,
            "tts_generation": False,
            "mfa": False,
            "dtw": False,
            "score_based_selection": False,
            "score_based_retry": False,
            "parent_overwrite": False,
        },
    }
    write_self_hashed_json(output_stage / "final.json", terminal)
    return terminal
