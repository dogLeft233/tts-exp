from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import config
from .common import file_sha256, write_self_hashed_json
from .protocol import ProtocolError


def run_stage06(protocol: Mapping[str, Any], cohort: Mapping[str, Any], fidelity: Mapping[str, Any], matrix: Mapping[str, Any], analysis: Mapping[str, Any], output_stage: Any) -> dict[str, Any]:
    if analysis.get("status") != "complete":
        raise ProtocolError("analysis is incomplete")
    if matrix.get("status") != "complete":
        raise ProtocolError("matrix is incomplete")
    decision = str(analysis["decisions"]["scientific_decision"])
    eligible = decision == "DAC_CODEC_IDENTITY_COMPATIBLE"
    terminal = {
        "schema_version": 1,
        "stage_id": "06_final",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": decision,
        "future_tts_alignment_experiment_eligible": eligible,
        "scope": "fixed 50-record LRS3 fit-only codec identity cohort",
        "limitations": ["not a population-wide LRS3 claim", "does not test TTS or alignment transfer", "identity compatibility is defined only by registered endpoints and margin"],
        "parent_summary_sha256": config.EXPECTED_PARENT_SUMMARY_SHA256,
        "source_manifest_sha256": config.EXPECTED_SOURCE_MANIFEST_SHA256,
        "protocol_sha256": file_sha256(config.STAGES["00_protocol"] / "protocol.json"),
        "cohort_sha256": file_sha256(config.STAGES["00_protocol"] / "cohort.json"),
        "audio_manifest_sha256": file_sha256(config.STAGES["01_audio"] / "audio_manifest.json"),
        "fidelity_sha256": file_sha256(config.STAGES["02_fidelity"] / "fidelity.json"),
        "video_manifest_sha256": file_sha256(config.STAGES["03_videos"] / "videos_manifest.json"),
        "matrix_manifest_sha256": file_sha256(config.STAGES["04_matrix"] / "matrix_manifest.json"),
        "analysis_sha256": file_sha256(config.STAGES["05_analysis"] / "analysis.json"),
        "endpoints": analysis["endpoints"],
        "wins": analysis["wins"],
        "decisions": analysis["decisions"],
        "sealed_scope": protocol["sealed_scope"],
    }
    write_self_hashed_json(output_stage / "final.json", terminal)
    return terminal
