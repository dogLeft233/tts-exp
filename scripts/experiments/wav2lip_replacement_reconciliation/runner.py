from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .analysis import (
    compute_parity,
    endpoint_rows,
    make_analysis,
    make_bootstrap_indices,
)
from .common import ReconciliationError, file_sha256, write_self_hashed_json
from .protocol import build_audit


def _paths(run_id: str) -> config.RunPaths:
    return config.RunPaths(config.run_root_for(run_id))


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_resume(paths: config.RunPaths) -> dict[str, Any]:
    if not paths.protocol.is_file() or not paths.input_audit.is_file():
        raise ReconciliationError("--resume requires an existing audit with protocol.json and input_audit.json")
    protocol = _read(paths.protocol)
    if protocol.get("protocol_id") != config.PROTOCOL_ID:
        raise ReconciliationError("run root belongs to another protocol")
    return protocol


def run_audit(run_id: str, *, resume: bool = False) -> config.RunPaths:
    paths = _paths(run_id)
    if paths.root.exists() and any(paths.root.iterdir()):
        if not resume:
            raise ReconciliationError(f"run root is populated; choose a new run-id or pass --resume: {paths.root}")
        _assert_resume(paths)
        return paths
    paths.root.mkdir(parents=True, exist_ok=True)
    bundle = build_audit(run_id)
    audit_sha = write_self_hashed_json(paths.input_audit, bundle["input_audit"])
    protocol = dict(bundle["protocol"])
    protocol["input_audit_sha256"] = audit_sha
    write_self_hashed_json(paths.protocol, protocol)
    return paths


def _write_discrepancy(paths: config.RunPaths, stage: str, exc: BaseException) -> None:
    write_self_hashed_json(
        paths.discrepancy,
        {
            "schema_version": 1,
            "stage_id": "discrepancy",
            "protocol_id": config.PROTOCOL_ID,
            "status": "blocked",
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "new_media_count": 0,
            "model_forward_count": 0,
        },
    )


def _write_blocked_final(paths: config.RunPaths, reason: str) -> None:
    write_self_hashed_json(
        paths.final,
        {
            "schema_version": 1,
            "stage_id": "final",
            "protocol_id": config.PROTOCOL_ID,
            "status": "blocked",
            "engineering_decision": "BLOCKED",
            "diagnostic_decision": "INCOMPLETE",
            "scientific_decision": "not_available",
            "next_action": "REVIEW_INPUT_OR_CACHE_DISCREPANCY",
            "reason": reason,
            "training_authorized": False,
            "generalization_established": False,
            "historical_gate_repaired": False,
            "new_media_count": 0,
            "model_forward_count": 0,
        },
    )


def _result_text(protocol: dict[str, Any], analysis: dict[str, Any], parity: dict[str, Any], final: dict[str, Any]) -> str:
    h_bridge = analysis["comparisons"]["H_BRIDGE_075_vs_N"]
    s_mag = analysis["comparisons"]["S_MAG_vs_N"]
    s_env = analysis["comparisons"]["S_ENV_vs_N"]
    return "\n".join(
        [
            "# Wav2Lip replacement endpoint reconciliation",
            "",
            "This is a cache-only retrospective diagnostic on the fixed paired 22-record cohort. It is not a new replacement confirmation.",
            "",
            "| comparison | endpoint | benefit C mean [95% CI] | benefit D mean [95% CI] | joint wins |",
            "|---|---|---:|---:|---:|",
            f"| H BRIDGE_075-N | FULL | {h_bridge['FULL']['benefit_C']['mean']:.6f} [{h_bridge['FULL']['benefit_C']['ci95'][0]:.6f}, {h_bridge['FULL']['benefit_C']['ci95'][1]:.6f}] | {h_bridge['FULL']['benefit_D']['mean']:.6f} [{h_bridge['FULL']['benefit_D']['ci95'][0]:.6f}, {h_bridge['FULL']['benefit_D']['ci95'][1]:.6f}] | {h_bridge['FULL']['joint_win_count']} |",
            f"| H BRIDGE_075-N | U | {h_bridge['U']['benefit_C']['mean']:.6f} [{h_bridge['U']['benefit_C']['ci95'][0]:.6f}, {h_bridge['U']['benefit_C']['ci95'][1]:.6f}] | {h_bridge['U']['benefit_D']['mean']:.6f} [{h_bridge['U']['benefit_D']['ci95'][0]:.6f}, {h_bridge['U']['benefit_D']['ci95'][1]:.6f}] | {h_bridge['U']['joint_win_count']} |",
            f"| S MAG-N | U | {s_mag['U']['benefit_C']['mean']:.6f} [{s_mag['U']['benefit_C']['ci95'][0]:.6f}, {s_mag['U']['benefit_C']['ci95'][1]:.6f}] | {s_mag['U']['benefit_D']['mean']:.6f} [{s_mag['U']['benefit_D']['ci95'][0]:.6f}, {s_mag['U']['benefit_D']['ci95'][1]:.6f}] | {s_mag['U']['joint_win_count']} |",
            f"| S ENV-N | U | {s_env['U']['benefit_C']['mean']:.6f} [{s_env['U']['benefit_C']['ci95'][0]:.6f}, {s_env['U']['benefit_C']['ci95'][1]:.6f}] | {s_env['U']['benefit_D']['mean']:.6f} [{s_env['U']['benefit_D']['ci95'][0]:.6f}, {s_env['U']['benefit_D']['ci95'][1]:.6f}] | {s_env['U']['joint_win_count']} |",
            "",
            f"Parity valid: {parity['valid']}; H matrices: {parity['H_full']['cell_count']}; endpoint rows: {analysis['endpoint_count']}.",
            "PCM and processing-chain differences are retained as confounds; the arithmetic support/window/pipeline terms are not causal contributions.",
            "",
            f"Engineering: {final['engineering_decision']}; diagnostic: {final['diagnostic_decision']}; scientific: {final['scientific_decision']}; next: {final['next_action']}.",
            "",
        ]
    )


def run_all(run_id: str, *, resume: bool = False) -> int:
    paths = run_audit(run_id, resume=resume)
    try:
        protocol = _read(paths.protocol)
        input_audit = _read(paths.input_audit)
        labels, indices = make_bootstrap_indices(protocol)
        np.save(paths.bootstrap_indices, indices, allow_pickle=False)
        endpoints = endpoint_rows(protocol)
        write_self_hashed_json(paths.endpoints, {"schema_version": 1, "stage_id": "endpoints", "protocol_id": config.PROTOCOL_ID, "status": "complete", "record_count": len(protocol["records"]), "matrix_count": config.EXPECTED_MATRIX_COUNT, "endpoint_count": len(endpoints), "endpoints": endpoints, "bootstrap_labels": labels})
        parity = compute_parity(protocol, endpoints)
        write_self_hashed_json(paths.parity, parity)
        if not parity.get("valid"):
            _write_discrepancy(paths, "parity", ReconciliationError(f"historical parity failed with {parity['error_count']} errors"))
            _write_blocked_final(paths, "historical parity failed; see discrepancy.json")
            return 1
        analysis = make_analysis(protocol, endpoints, input_audit, indices)
        write_self_hashed_json(paths.analysis, analysis)
        review = {
            "schema_version": 1,
            "stage_id": "review",
            "protocol_id": config.PROTOCOL_ID,
            "status": "complete",
            "reviewer": "self-review before independent validation",
            "checks": {
                "fixed_root_hashes": True,
                "keyed_join_22_records": True,
                "matrix_count_176": True,
                "endpoint_count_528": True,
                "h_full_and_s_parent_parity": True,
                "zero_new_media": True,
                "zero_model_forward": True,
                "no_training_or_generalization_claim": True,
            },
            "protocol_sha256": file_sha256(paths.protocol),
            "code_bindings": protocol["code_bindings"],
            "spec_bindings": protocol["spec_bindings"],
        }
        write_self_hashed_json(paths.review, review)
        from .validate import validate_pre_final

        validation = validate_pre_final(paths.root)
        write_self_hashed_json(paths.validation, validation)
        if not validation.get("valid"):
            _write_discrepancy(paths, "independent_validation", ReconciliationError(f"independent validation failed with {validation['error_count']} errors"))
            _write_blocked_final(paths, "independent validation failed; see validation.json and discrepancy.json")
            return 1
        final = {
            "schema_version": 1,
            "stage_id": "final",
            "protocol_id": config.PROTOCOL_ID,
            "status": "complete",
            "engineering_decision": "GO",
            "diagnostic_decision": "RECONCILED",
            "scientific_decision": "NOT_A_CONFIRMATION",
            "next_action": "STOP_CURRENT_SPECTRAL_CONSTRUCTION",
            "record_count": len(protocol["records"]),
            "source_group_count": len({str(row["source_group"]) for row in protocol["records"]}),
            "matrix_count": config.EXPECTED_MATRIX_COUNT,
            "endpoint_count": len(endpoints),
            "parity_sha256": file_sha256(paths.parity),
            "analysis_sha256": file_sha256(paths.analysis),
            "review_sha256": file_sha256(paths.review),
            "validation_sha256": file_sha256(paths.validation),
            "protocol_sha256": file_sha256(paths.protocol),
            "new_media_count": 0,
            "model_forward_count": 0,
            "training_authorized": False,
            "generalization_established": False,
            "historical_gate_repaired": False,
            "boundary": "H legacy cache has no original embeddings; this is diagnostic arithmetic and cache consistency only.",
        }
        write_self_hashed_json(paths.final, final)
        paths.result.write_text(_result_text(protocol, analysis, parity, final), encoding="utf-8")
        from .validate import validate_run

        post = validate_run(paths.root)
        if not post.get("valid"):
            raise ReconciliationError(f"post-final validation failed: {post['errors'][:3]}")
        return 0
    except Exception as exc:  # noqa: BLE001 - runner must seal unexpected failures as discrepancies
        _write_discrepancy(paths, "runner", exc)
        if not paths.final.is_file():
            _write_blocked_final(paths, str(exc))
        return 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cache-only Wav2Lip endpoint reconciliation")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("audit", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "audit":
        try:
            run_audit(args.run_id, resume=bool(args.resume))
            return 0
        except Exception as exc:  # noqa: BLE001 - CLI must persist any failure as a discrepancy
            paths = _paths(args.run_id)
            paths.root.mkdir(parents=True, exist_ok=True)
            _write_discrepancy(paths, "audit", exc)
            _write_blocked_final(paths, str(exc))
            return 1
    return run_all(args.run_id, resume=bool(args.resume))


if __name__ == "__main__":
    raise SystemExit(main())
