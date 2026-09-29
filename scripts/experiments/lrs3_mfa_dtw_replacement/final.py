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
from .replacement import replacement_contrast_rows
from .statistics import cluster_bootstrap


def _contrast_summary(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    c = cluster_bootstrap(rows, "delta_C", draws=config.BOOTSTRAP_DRAWS, seed=config.SEED)
    d = cluster_bootstrap(rows, "delta_D", draws=config.BOOTSTRAP_DRAWS, seed=config.SEED)
    return {
        "name": name,
        "mean_delta_C": c["mean"],
        "mean_delta_D": d["mean"],
        "delta_C_ci95": c["ci95"],
        "delta_D_ci95": d["ci95"],
        "delta_C_lower_bound_strictly_positive": c["lower_bound_strictly_positive"],
        "delta_D_lower_bound_strictly_positive": d["lower_bound_strictly_positive"],
        "C_positive_count": int(sum(float(row["delta_C"]) > 0.0 for row in rows)),
        "D_positive_count": int(sum(float(row["delta_D"]) > 0.0 for row in rows)),
        "joint_positive_count": int(sum(float(row["delta_C"]) > 0.0 and float(row["delta_D"]) > 0.0 for row in rows)),
        "per_record": rows,
    }


def run_final(stage00_dir: str | Path = config.STAGE00, analysis_dir: str | Path = config.STAGE03, replacement_dir: str | Path = config.STAGE04, output_dir: str | Path = config.STAGE05) -> dict[str, Any]:
    stage00_root = Path(stage00_dir).resolve()
    analysis_root = Path(analysis_dir).resolve()
    replacement_root = Path(replacement_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"final stage is already populated: {output}")
    stage00 = load_json(stage00_root / "manifest.json")
    if not analysis_root.joinpath("analysis.json").is_file():
        candidate_summary = load_json(config.STAGE01 / "summary.json")
        failure_path = config.STAGE01 / "failure_ledger.json"
        if candidate_summary.get("engineering_decision") != "BLOCKED" or not failure_path.is_file():
            raise ProtocolError("diagonal analysis is missing without a valid blocked candidate stage")
        if replacement_root.exists():
            raise ProtocolError("replacement output must remain absent after candidate-stage blockage")
        failure = load_json(failure_path)
        result = {
            "schema_version": 1,
            "manifest_type": "lrs3_mfa_dtw_comparison_final",
            "stage_id": "05_final",
            "protocol_id": config.PROTOCOL_ID,
            "status": "complete",
            "engineering_decision": "BLOCKED",
            "scientific_decision": "not_available",
            "terminal_reason": "candidate_stage_incomplete",
            "replacement_status": "sealed_not_run",
            "replacement_authorized": False,
            "record_count": int(candidate_summary.get("record_count", 0)),
            "expected_record_count": config.EXPECTED_RECORD_COUNT,
            "candidate_summary": candidate_summary,
            "failure_ledger": failure,
            "stage00_manifest_sha256": sha256_file(stage00_root / "manifest.json"),
        }
        output.mkdir(parents=True, exist_ok=True)
        write_json_once(output / "final.json", result)
        write_json_once(output / "decision.json", {
            "schema_version": 1,
            "stage_id": "05_final",
            "protocol_id": config.PROTOCOL_ID,
            "engineering_decision": "BLOCKED",
            "scientific_decision": "not_available",
            "replacement_status": "sealed_not_run",
            "reason": "Stage 01 could not produce a complete 133-record hard-DTW candidate matrix",
            "final_payload_sha256": canonical_sha256(result),
        })
        return result
    analysis = load_json(analysis_root / "analysis.json")
    decision = load_json(analysis_root / "decision.json")
    if stage00.get("cohort", {}).get("ordered_sample_ids_sha256") != config.EXPECTED_COHORT_HASH or analysis.get("cohort_sha256") != config.EXPECTED_COHORT_HASH or decision.get("cohort_sha256") != config.EXPECTED_COHORT_HASH:
        raise ProtocolError("final parent cohort binding is invalid")
    if load_json(analysis_root / "decision.sha256").get("sha256") != sha256_file(analysis_root / "decision.json"):
        raise ProtocolError("final decision binding self-hash changed")
    if decision.get("replacement_authorized") is not True:
        if replacement_root.exists():
            raise ProtocolError("replacement output must remain absent after a sealed diagonal decision")
        result = {
            "schema_version": 1,
            "manifest_type": "lrs3_mfa_dtw_comparison_final",
            "stage_id": "05_final",
            "protocol_id": config.PROTOCOL_ID,
            "status": "complete",
            "engineering_decision": "GO",
            "scientific_decision": analysis.get("decision"),
            "replacement_status": "sealed_not_run",
            "replacement_authorized": False,
            "record_count": config.EXPECTED_RECORD_COUNT,
            "diagonal_analysis": analysis,
            "historical_context": {
                "mfa_linear_parent": str(config.REPLACEMENT_MANIFEST),
                "prior_analysis": str(config.HISTORICAL_ANALYSIS),
            },
        }
        output.mkdir(parents=True, exist_ok=True)
        write_json_once(output / "final.json", result)
        write_json_once(output / "decision.json", {
            "schema_version": 1,
            "stage_id": "05_final",
            "protocol_id": config.PROTOCOL_ID,
            "engineering_decision": "GO",
            "scientific_decision": analysis.get("decision"),
            "replacement_status": "sealed_not_run",
            "reason": "DTW diagonal promotion gate did not pass; strict replacement was intentionally not run",
            "final_payload_sha256": canonical_sha256(result),
        })
        return result
    if not replacement_root.is_dir():
        raise ProtocolError("authorized replacement is missing")
    replacement = load_json(replacement_root / "replacement_manifest.json")
    contrasts = replacement_contrast_rows(stage00, replacement)
    summaries = {name: _contrast_summary(rows, name) for name, rows in contrasts.items()}
    safe = summaries["dtw_vs_natural"]["delta_C_lower_bound_strictly_positive"] and summaries["dtw_vs_natural"]["delta_D_lower_bound_strictly_positive"]
    result = {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa_dtw_comparison_final",
        "stage_id": "05_final",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "REPLACEMENT_SAFE_GO" if safe else "NO_REPLACEMENT_SAFE_GAIN",
        "replacement_status": "complete",
        "replacement_authorized": True,
        "record_count": config.EXPECTED_RECORD_COUNT,
        "contrasts": summaries,
        "parent_bindings": {
            "analysis": {"path": str(analysis_root / "decision.json"), "sha256": sha256_file(analysis_root / "decision.json")},
            "replacement": {"path": str(replacement_root / "replacement_manifest.json"), "sha256": sha256_file(replacement_root / "replacement_manifest.json")},
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json_once(output / "final.json", result)
    write_json_once(output / "decision.json", {
        "schema_version": 1,
        "stage_id": "05_final",
        "protocol_id": config.PROTOCOL_ID,
        "engineering_decision": "GO",
        "scientific_decision": result["scientific_decision"],
        "replacement_status": "complete",
        "reason": "DTW replacement was evaluated against both MFA-linear replacement and untouched natural baseline",
        "final_payload_sha256": canonical_sha256(result),
    })
    return result
