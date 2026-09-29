from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.experiments.fresh_source_inputs.protocol import read_json, write_json


def _read_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def _shared_root(run_root: Path) -> Path:
    """Resolve the shared P artifacts for both the CLI and unit-test layouts."""
    candidates = (run_root / "run" / "shared", run_root)
    for candidate in candidates:
        if (candidate / "cohort.json").is_file():
            return candidate
    return candidates[0]


def _measurement_gate(run_root: Path) -> tuple[bool, list[str], dict[str, Any]]:
    root = run_root / "run"
    shared = _shared_root(run_root)
    cohort = _read_optional(shared / "cohort.json")
    a_analysis = _read_optional(root / "A" / "analysis.json")
    a_final = _read_optional(root / "A" / "final.json")
    b_calibration = _read_optional(root / "B" / "calibration.json")
    b_final = _read_optional(root / "B" / "final.json")
    c_final = _read_optional(root / "C" / "final.json")
    blockers: list[str] = []
    if str(cohort.get("status")) not in {"GO", "COHORT_READY"} and str(cohort.get("readiness")) != "COHORT_READY":
        blockers.append(f"P cohort status={cohort.get('status', 'missing')}")
    controls = a_analysis.get("controls")
    if not isinstance(controls, dict) or controls.get("pass") is not True:
        blockers.append(f"A natural controls status={a_analysis.get('status', 'missing')}")
    if str(a_final.get("cross_generator_status")) != "READY":
        blockers.append(f"A cross-generator status={a_final.get('cross_generator_status', 'missing')}")
    if b_calibration.get("calibrated") is not True and b_calibration.get("status") != "METRIC_CALIBRATED":
        blockers.append(f"B calibration status={b_calibration.get('status', 'missing')}")
    # B's human review is deliberately not an entry gate; it is required for
    # the later visual claim, not for constructing the audio M arm.
    snapshot = {
        "cohort_status": cohort.get("status", cohort.get("readiness")),
        "a_control_status": a_analysis.get("status"),
        "a_control_pass": controls.get("pass") if isinstance(controls, dict) else False,
        "a_cross_generator_status": a_final.get("cross_generator_status"),
        "a_final_status": a_final.get("status"),
        "b_calibration_status": b_calibration.get("status"),
        "b_calibrated": bool(b_calibration.get("calibrated")),
        "b_final_status": b_final.get("status"),
        "c_final_status": c_final.get("status"),
    }
    return not blockers, blockers, snapshot


def run_deferred(run_root: Path) -> int:
    root = run_root / "run" / "D"
    ready, blockers, snapshot = _measurement_gate(run_root)
    # This implementation intentionally stops before TTS/MFA.  Even if a
    # future run satisfies the measurement gates, the asset/budget gate must
    # be explicitly implemented before any provider can be called.
    status = "DEFERRED_MEASUREMENT" if not ready else "BLOCKED_TTS_ASSET"
    reason = blockers or ["D TTS/MFA asset and budget gate is not implemented; no provider was invoked"]
    write_json(root / "tts_inputs.json", {
        "schema_version": 1,
        "status": status,
        "records": [],
        "formal_cells": 0,
        "reason": reason,
        "measurement_gate": snapshot,
        "provider_invoked": False,
        "cloud_fallback": False,
    })
    write_json(root / "final.json", {
        "schema_version": 1,
        "status": status,
        "reason": reason,
        "measurement_gate": snapshot,
        "replacement_confirmed": False,
        "training_authorized": False,
        "generalization_established": False,
        "formal_cells": 0,
        "provider_invoked": False,
    })
    write_json(root / "validation.json", {
        "schema_version": 1,
        "status": "GO",
        "errors": [],
        "scientific_cells": 0,
        "deferred_status": status,
        "provider_invoked": False,
    })
    print(f"D status={status}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("tts", "analyze"), default="tts")
    args = parser.parse_args(argv)
    return run_deferred(args.run_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
