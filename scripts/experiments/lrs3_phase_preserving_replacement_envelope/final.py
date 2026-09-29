from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import config
from .common import file_sha256, write_self_hashed_json
from .protocol import ProtocolError


def run_stage05_final(protocol: Mapping[str, Any], cohort: Mapping[str, Any], diagnostics: Mapping[str, Any], matrix: Mapping[str, Any], analysis: Mapping[str, Any], output_stage: Path) -> dict[str, Any]:
    if protocol.get("protocol_id") != config.PROTOCOL_ID or cohort.get("status") != "complete" or diagnostics.get("status") != "complete" or matrix.get("status") != "complete" or analysis.get("status") != "complete":
        raise ProtocolError("cannot finalize an incomplete replacement-envelope run")
    decision = str(analysis.get("decisions", {}).get("scientific_decision", ""))
    allowed = {"CONTROL_FAILED", "PHASE_PRESERVING_REPLACEMENT_ENVELOPE_FOUND", "ONLY_TRIVIAL_REPLACEMENT_EQUIVALENCE_FOUND"}
    if decision not in allowed:
        raise ProtocolError(f"unknown terminal scientific decision: {decision}")
    terminal = {
        "schema_version": 1,
        "stage_id": "05_final",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": decision,
        "identity_characterization_experiment_eligible": decision == "PHASE_PRESERVING_REPLACEMENT_ENVELOPE_FOUND",
        "largest_qualifying_registered_blend_strength": analysis.get("max_compatible_alpha"),
        "scope": analysis["decisions"]["scope"],
        "limitations": [
            "fit-only fixed 23-record/23-source-group cohort",
            "result applies only to the registered phase-preserving magnitude construction",
            "TTS-direction movement is not speaker-identity transfer",
            "a positive result authorizes a separate spec but no model training or deployment",
        ],
        "parent_protocol_sha256": config.EXPECTED_PARENT_PROTOCOL_SHA256,
        "parent_replacement_sha256": config.EXPECTED_PARENT_REPLACEMENT_SHA256,
        "protocol_sha256": file_sha256(config.STAGES["00_protocol"] / "protocol.json"),
        "cohort_sha256": file_sha256(config.STAGES["00_protocol"] / "cohort.json"),
        "audio_manifest_sha256": file_sha256(config.STAGES["01_candidates"] / "audio_manifest.json"),
        "diagnostics_sha256": file_sha256(config.STAGES["02_audio_diagnostics"] / "diagnostics.json"),
        "scores_manifest_sha256": file_sha256(config.STAGES["04_scores"] / "scores_manifest.json"),
        "record_count": config.EXPECTED_RECORD_COUNT,
        "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
        "endpoints": analysis["compatibility"],
        "movement": analysis["movement"],
        "controls": analysis["controls"],
        "bootstrap": analysis["bootstrap"],
        "sealed_scope": protocol["sealed_scope"],
    }
    analysis_path = output_stage / "analysis.json"
    if analysis_path.is_file():
        terminal["analysis_sha256"] = file_sha256(analysis_path)
    else:
        terminal.pop("analysis_sha256")
    write_self_hashed_json(output_stage / "final.json", terminal)
    return terminal
