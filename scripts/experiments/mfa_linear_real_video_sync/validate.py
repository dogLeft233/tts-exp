#!/usr/bin/env python3
"""Validate the immutable prototype artifact graph without rerunning training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import ProtocolError, sha256_file, write_json_once


def _read_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"missing or corrupt artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ProtocolError(f"artifact is not a JSON object: {path}")
    return payload


def validate_run(root: str | Path) -> dict[str, Any]:
    run = Path(root).resolve()
    decision_path = run / "decision.json"
    decision = _read_object(decision_path)
    status = str(decision.get("status", ""))
    checks: dict[str, bool] = {
        "one_terminal_decision": decision_path.is_file(),
        "decision_has_boolean_pass": isinstance(decision.get("pass"), bool),
        "p2_not_implicitly_run": not (run / "03_p2_shared_four").exists() or bool(decision.get("p1_status")),
    }
    if status == "ENGINEERING_NO_GO":
        tests = _read_object(run / "01_p0_seam/test_results.json")
        checks.update(
            {
                "failed_stage_is_p0": decision.get("failed_stage") == "P0_SEAM",
                "p0_failure_preserved": tests.get("status") in {"failed", "complete"}
                and isinstance(decision.get("error"), str)
                and bool(decision.get("error")),
                "no_p1_optimizer_history": not (run / "02_p1_one_record/history.json").exists(),
                "no_p2_artifacts": not (run / "03_p2_shared_four").exists(),
            }
        )
    elif status in {
        "FIXED_DATA_TTS_ONLY_SYNC_CONSTRUCTED",
        "INPUT_LOCK_FAILURE",
        "NATURAL_INPUT_LEAKAGE",
        "VISUAL_COORDINATE_MISMATCH",
        "TARGET_OFFSET_AMBIGUOUS",
        "OFFSET_SIGN_MISMATCH",
        "MFCC_PARITY_FAILURE",
        "FILE_PROXY_PARITY_FAILURE",
        "IDENTITY_FAILURE",
        "FROZEN_STATE_MUTATION",
        "NO_CANDIDATE_GRADIENT",
        "NONFINITE_TRAINING",
        "TRUST_REGION_FAILURE",
        "NO_PROXY_LOSS_DESCENT",
        "OFFICIAL_D_MARGIN_FAILURE",
        "OFFICIAL_C_MARGIN_FAILURE",
        "OFFICIAL_OFFSET_FAILURE",
    }:
        p0 = _read_object(run / "01_p0_seam/evidence.json")
        p1 = _read_object(run / "02_p1_one_record/decision.json")
        history = _read_object(run / "02_p1_one_record/history.json")
        checks.update(
            {
                "p0_go": p0.get("status") == "GO",
                "p1_decision_matches_terminal": p1.get("status") == status,
                "exact_p1_steps": history.get("steps") == 20,
                "step0_checkpoint": (run / "02_p1_one_record/step0/model.pt").is_file(),
                "step20_checkpoint": (run / "02_p1_one_record/step20/model.pt").is_file(),
                "step0_official_curve": (run / "02_p1_one_record/step0/official_curve.json").is_file(),
                "step20_official_curve": (run / "02_p1_one_record/step20/official_curve.json").is_file(),
            }
        )
    elif status in {
        "FIXED_DATA_SHARED_TTS_ONLY_SYNC_CONSTRUCTED",
        "FIXED_DATA_SHARED_TTS_ONLY_SYNC_NOT_CONSTRUCTED",
    }:
        p0 = _read_object(run / "01_p0_seam/evidence.json")
        p1 = _read_object(run / "02_p1_one_record/decision.json")
        p2 = _read_object(run / "03_p2_shared_four/decision.json")
        history = _read_object(run / "03_p2_shared_four/history.json")
        manifest = _read_object(run / "00_lock/p2_manifest.json")
        checks.update(
            {
                "p0_go": p0.get("status") == "GO",
                "p1_passed": p1.get("pass") is True,
                "p2_decision_matches_terminal": p2.get("status") == status and p2.get("pass") == decision.get("pass"),
                "exact_p2_steps": history.get("stage") == "P2_SHARED_FOUR" and history.get("steps") == 100,
                "exact_p2_records": history.get("record_order") == manifest.get("record_order") and len(history.get("record_order", [])) == 4,
                "p2_manifest_train": manifest.get("protocol_split") == "train" and manifest.get("natural_in_training_records") is False,
                "step0_checkpoint": (run / "03_p2_shared_four/step0/model.pt").is_file(),
                "step100_checkpoint": (run / "03_p2_shared_four/step100/model.pt").is_file(),
                "p2_official_curves": all(
                    (run / "03_p2_shared_four" / str(sample_id) / label / "official_curve.json").is_file()
                    for sample_id in manifest.get("record_order", [])
                    for label in ("step0", "step100")
                ),
            }
        )
    else:
        checks["recognized_terminal_status"] = False
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    result = validate_run(args.run)
    if args.write:
        write_json_once(args.run / "validation.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result["artifact_graph_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
