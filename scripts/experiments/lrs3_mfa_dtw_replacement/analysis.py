from __future__ import annotations

from pathlib import Path
from typing import Any

from . import config
from .protocol import (
    ProtocolError,
    canonical_sha256,
    load_json,
    sha256_file,
    write_json_once,
)
from .statistics import analyze_diagonal


def run_analysis(stage00_dir: str | Path = config.STAGE00, diagonal_dir: str | Path = config.STAGE02, output_dir: str | Path = config.STAGE03) -> dict[str, Any]:
    stage00_root = Path(stage00_dir).resolve()
    diagonal_root = Path(diagonal_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "analysis.json").exists() or (output / "decision.json").exists():
        raise FileExistsError(f"diagonal analysis stage is already finalized: {output}")
    stage00 = load_json(stage00_root / "manifest.json")
    diagonal = load_json(diagonal_root / "diagonal_manifest.json")
    if stage00.get("protocol_id") != config.PROTOCOL_ID or stage00.get("cohort", {}).get("ordered_sample_ids_sha256") != config.EXPECTED_COHORT_HASH:
        raise ProtocolError("Stage 00 binding is invalid")
    if diagonal.get("protocol_id") != config.PROTOCOL_ID or diagonal.get("status") != "complete" or len(diagonal.get("scores", [])) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("diagonal matrix is incomplete")
    stage00_rows = stage00["cohort"]["records"]
    diagonal_rows = diagonal["scores"]
    if [str(row["sample_id"]) for row in diagonal_rows] != [str(row["sample_id"]) for row in stage00_rows]:
        raise ProtocolError("diagonal order differs from frozen cohort")
    rows = []
    for score, source in zip(diagonal_rows, stage00_rows, strict=True):
        linear = source["historical_scores"]["G_M_E_M"]
        if score.get("historical_linear", {}).get("score_row_sha256") != linear.get("score_row_sha256"):
            raise ProtocolError(f"historical linear binding changed: {score['sample_id']}")
        rows.append({
            "sample_id": str(score["sample_id"]),
            "source_group": str(score["source_group"]),
            "dtw_sync_c": float(score["sync_c"]),
            "dtw_sync_d": float(score["sync_d"]),
            "linear_sync_c": float(linear["sync_c"]),
            "linear_sync_d": float(linear["sync_d"]),
            "dtw_score_row_sha256": canonical_sha256(score),
            "linear_score_row_sha256": str(linear["score_row_sha256"]),
        })
    result = analyze_diagonal(rows, engineering_complete=True)
    result.update({
        "protocol_id": config.PROTOCOL_ID,
        "stage_id": "03_diagonal_analysis",
        "cohort_sha256": config.EXPECTED_COHORT_HASH,
        "parent_bindings": {
            "stage00_manifest": {"path": str(stage00_root / "manifest.json"), "sha256": sha256_file(stage00_root / "manifest.json")},
            "diagonal_manifest": {"path": str(diagonal_root / "diagonal_manifest.json"), "sha256": sha256_file(diagonal_root / "diagonal_manifest.json")},
        },
    })
    analysis_hash = write_json_once(output / "analysis.json", result)
    decision = {
        "schema_version": 1,
        "stage_id": "03_diagonal_analysis",
        "protocol_id": config.PROTOCOL_ID,
        "engineering_decision": "GO",
        "scientific_decision": result["decision"],
        "replacement_authorized": bool(result["replacement_authorized"]),
        "analysis_sha256": analysis_hash,
        "cohort_sha256": config.EXPECTED_COHORT_HASH,
        "next_allowed_stage": "04_replacement" if result["replacement_authorized"] else "05_final",
        "reason": "both source-group cluster-bootstrap lower bounds are strictly positive" if result["replacement_authorized"] else "DTW did not establish simultaneous positive lower bounds for Sync-C and Sync-D",
    }
    decision["decision_payload_sha256"] = canonical_sha256(decision)
    write_json_once(output / "decision.json", decision)
    write_json_once(output / "decision.sha256", {"sha256": sha256_file(output / "decision.json")})
    return decision
