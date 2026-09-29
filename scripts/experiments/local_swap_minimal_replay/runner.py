from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import config
from .analysis import analyze, report_markdown, update_support, write_scores_csv
from .audit import audit_history
from .common import DiagnosticError, file_sha256, read_json, verify_self_hashed_json, write_self_hashed_json
from .prepare import prepare_media
from .protocol import freeze_inputs, load_history, write_protocol
from .review import create_playback
from .scoring import score_all
from .validate import validate_outputs


def _guard_root(paths: config.RunPaths) -> None:
    if paths.final.is_file():
        raise DiagnosticError(f"terminal run already exists; choose a new run id: {paths.root}")
    if paths.root.exists() and any(paths.root.iterdir()):
        raise DiagnosticError(f"run root is non-empty; fresh-cache evidence requires a new run id: {paths.root}")
    paths.root.mkdir(parents=True, exist_ok=True)


def _final_payload(paths: config.RunPaths, analysis_payload: dict[str, Any], audit: dict[str, Any], media: dict[str, Any], scores: dict[str, Any], validation: dict[str, Any], review: dict[str, Any], error: dict[str, Any] | None = None) -> dict[str, Any]:
    input_payload = verify_self_hashed_json(paths.inputs)
    support = verify_self_hashed_json(paths.support)
    input_ok = bool(audit.get("claims", {}).get("history_record_verified"))
    media_ok = bool(media.get("status") == "complete")
    score_ok = bool(scores.get("status") == "complete" and scores.get("complete_count") == config.EXPECTED_NEW_CELL_COUNT)
    validation_ok = bool(validation.get("status") == "complete")
    if error is not None:
        status = "IMPLEMENTATION_ERROR"
    elif not input_ok or not media_ok or not score_ok:
        status = "BLOCKED_INPUT"
    elif not validation_ok:
        status = "IMPLEMENTATION_ERROR"
    else:
        status = "COMPLETE"
    return {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": status,
        "engineering_decision": "GO" if validation_ok else "BLOCKED",
        "scope": "three fixed samples; historical audit all 22; new A/B/C matrix 24 cells",
        "historical_record_verified": bool(analysis_payload.get("historical_record_verified")),
        "replay_direction": analysis_payload.get("replay_direction"),
        "replay_numeric_agreement": analysis_payload.get("replay_numeric_agreement"),
        "known_pairing_sensitivity": analysis_payload.get("known_pairing_sensitivity"),
        "generated_audio_preference": analysis_payload.get("generated_audio_preference"),
        "localization": analysis_payload.get("localization"),
        "new_cell_count": int(scores.get("cell_count", 0)),
        "new_complete_cell_count": int(scores.get("complete_count", 0)),
        "new_failure_count": int(scores.get("failure_count", 0)),
        "audit_sha256": file_sha256(paths.historical_audit),
        "inputs_sha256": file_sha256(paths.inputs),
        "protocol_sha256": file_sha256(paths.protocol),
        "media_manifest_sha256": file_sha256(paths.media / "manifest.json"),
        "score_manifest_sha256": file_sha256(paths.scores / "manifest.json"),
        "support_sha256": file_sha256(paths.support),
        "analysis_sha256": file_sha256(paths.root / "analysis.json"),
        "validation_sha256": file_sha256(paths.root / "validation.json"),
        "report_sha256": file_sha256(paths.report) if paths.report.is_file() else None,
        "review_status": review.get("human_review_status", "NOT_HUMAN_REVIEWED"),
        "review_manifest_sha256": file_sha256(paths.review / "manifest.json"),
        "input_runtime": input_payload.get("runtime"),
        "forbidden_operations_observed": ["wav2lip_inference", "tts_generation", "training", "bridge_rescoring", "score_based_retry"],
        "errors": [error] if error else [],
        "validation": validation,
        "support_status": support.get("status"),
    }


def run(run_id: str) -> dict[str, Any]:
    paths = config.RunPaths(config.run_root_for(run_id))
    _guard_root(paths)
    try:
        history = load_history()
        inputs = freeze_inputs(history, paths.root)
        write_self_hashed_json(paths.inputs, inputs)
        audit = audit_history(history, paths.historical_audit)
        protocol = write_protocol(paths, verify_self_hashed_json(paths.inputs), verify_self_hashed_json(paths.historical_audit))
        media = prepare_media(paths, history, config.SAMPLE_IDS, defer_c=True)
        scores = score_all(paths, protocol, media, history=history)
        support = verify_self_hashed_json(paths.support)
        analysis_payload = analyze(verify_self_hashed_json(paths.inputs), verify_self_hashed_json(paths.historical_audit), verify_self_hashed_json(paths.media / "manifest.json"), verify_self_hashed_json(paths.scores / "manifest.json"), support)
        write_self_hashed_json(paths.root / "analysis.json", analysis_payload)
        write_scores_csv(paths, verify_self_hashed_json(paths.media / "manifest.json"), verify_self_hashed_json(paths.scores / "manifest.json"))
        update_support(paths, analysis_payload)
        review = create_playback(paths, verify_self_hashed_json(paths.media / "manifest.json"), verify_self_hashed_json(paths.support))
        paths.report.write_text(report_markdown(analysis_payload, verify_self_hashed_json(paths.historical_audit), str(review.get("human_review_status"))), encoding="utf-8")
        validation = validate_outputs(paths)
        final = _final_payload(paths, analysis_payload, audit, verify_self_hashed_json(paths.media / "manifest.json"), verify_self_hashed_json(paths.scores / "manifest.json"), validation, review)
        return write_self_hashed_json(paths.final, final) and verify_self_hashed_json(paths.final)
    except Exception as exc:
        error = {"error_type": type(exc).__name__, "error": str(exc)}
        try:
            if not paths.inputs.is_file():
                write_self_hashed_json(paths.inputs, {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": "blocked", "error": error})
            if not paths.historical_audit.is_file():
                write_self_hashed_json(paths.historical_audit, {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": "blocked", "error": error})
            if not paths.protocol.is_file():
                write_self_hashed_json(paths.protocol, {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": "blocked", "error": error})
            if not paths.support.is_file():
                write_self_hashed_json(paths.support, {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": "blocked", "samples": {}, "failures": [error]})
            if not (paths.media / "manifest.json").is_file():
                write_self_hashed_json(paths.media / "manifest.json", {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": "blocked", "cell_count": 0, "cells": [], "failures": [error]})
            if not (paths.scores / "manifest.json").is_file():
                write_self_hashed_json(paths.scores / "manifest.json", {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "status": "blocked", "cell_count": 0, "complete_count": 0, "failure_count": 0, "scores": [], "failures": [error]})
            if not paths.report.is_file():
                paths.report.write_text(f"# LOCAL_SWAP 最小重放诊断\n\n实验执行异常：`{type(exc).__name__}: {exc}`\n", encoding="utf-8")
            if not (paths.review / "manifest.json").is_file():
                paths.review.mkdir(parents=True, exist_ok=True)
                write_self_hashed_json(paths.review / "manifest.json", {"schema_version": 1, "status": "blocked", "human_review_status": "NOT_HUMAN_REVIEWED", "entry_count": 0})
            validation = validate_outputs(paths)
            final = {
                "schema_version": 1,
                "protocol_id": config.PROTOCOL_ID,
                "protocol_revision": config.PROTOCOL_REVISION,
                "status": "IMPLEMENTATION_ERROR",
                "engineering_decision": "BLOCKED",
                "error": error,
                "validation": validation,
                "review_status": "NOT_HUMAN_REVIEWED",
            }
            write_self_hashed_json(paths.final, final)
        except Exception as finalize_exc:
            raise DiagnosticError(f"experiment failed and blocked-finalization also failed: {exc}; {finalize_exc}") from finalize_exc
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the minimal LOCAL_SWAP replay diagnostic")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run(args.run_id)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
