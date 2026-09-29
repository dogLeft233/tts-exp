#!/usr/bin/env python3
"""Deterministic gate fixtures for branches that the real run may not enter.

These fixtures contain no media, model output, or scientific observations.  A
fixture record is always labelled ``SYNTHETIC_NOT_SCIENCE`` and is used only to
exercise the pre-registered gate decisions when calibration or native support
stops the formal run early.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from tts_visual_timing import (
    _generation_skip,
    evaluate_replacement_gate,
    evaluate_wav2lip_baseline_gate,
)


SYNTHETIC_LABEL = "SYNTHETIC_NOT_SCIENCE"


def _ci(left: float, right: float, *, mean: float = 0.0) -> Dict[str, Any]:
    return {"status": "COMPLETE", "mean": mean, "ci": [left, right]}


def _baseline(*, identity_ok: bool = True, source_count: int = 8) -> Dict[str, Any]:
    identity_c = [-0.05, 0.05] if identity_ok else [0.25, 0.35]
    identity_e = [-5.0, 5.0] if identity_ok else [25.0, 35.0]
    return {
        "valid_source_count": source_count,
        "native_delta_c": _ci(0.02, 0.12, mean=0.07),
        "BV_W": _ci(25.0, 55.0, mean=40.0),
        "identity_n_delta_c": _ci(*identity_c),
        "identity_t_delta_c": _ci(*identity_c),
        "identity_n_delta_E_ms": _ci(*identity_e),
        "identity_t_delta_E_ms": _ci(*identity_e),
    }


def _decision(mean: float, left: float) -> Dict[str, Any]:
    return {"science": "SUPPORTED", "mean": mean, "ci": [left, mean + 10.0]}


def run_synthetic_fixtures() -> Dict[str, Dict[str, Any]]:
    """Return expected outcomes for all unentered gate branches."""

    failed_calibration = {"science": "NO_CLEAR_SUPPORT"}
    calibration_ok = {"science": "SUPPORTED"}
    native_unsupported = {"gate": {"status": "NO_CLEAR_SUPPORT"}}
    native_ok = {"gate": {"status": "VISUAL_NATIVE_SUPPORT"}, "decision": {"delta_c_95": {"ci": [0.01, 0.20]}}}
    records = {"main_records": [{"sample_id": 151, "source_group": "synthetic"}]}
    cases: Dict[str, Dict[str, Any]] = {}
    cases["A_calibration_failed"] = {
        "label": SYNTHETIC_LABEL,
        "expected": "VISUAL_TIMING_CALIBRATION_GATE_NOT_PASSED",
        "actual": _generation_skip(records, "VISUAL_TIMING_CALIBRATION_GATE_NOT_PASSED")["reason"],
        "calibration": failed_calibration,
    }
    cases["B_native_no_support"] = {
        "label": SYNTHETIC_LABEL,
        "expected": "NATIVE_BV_OR_DELTA_C_GATE_NOT_PASSED",
        "actual": _generation_skip(records, "NATIVE_BV_OR_DELTA_C_GATE_NOT_PASSED")["reason"],
        "calibration": calibration_ok,
        "native": native_unsupported,
    }
    cases["Wav2Lip_native_unconfirmed"] = {
        "label": SYNTHETIC_LABEL,
        "expected": "WAV2LIP_BASELINE_UNCONFIRMED",
        "actual": evaluate_wav2lip_baseline_gate(_baseline(source_count=7), mfa_qc_status="COMPLETE")["status"],
    }
    cases["identity_not_equivalent"] = {
        "label": SYNTHETIC_LABEL,
        "expected": "WAV2LIP_BASELINE_UNCONFIRMED",
        "actual": evaluate_wav2lip_baseline_gate(_baseline(identity_ok=False), mfa_qc_status="COMPLETE")["status"],
    }
    rv_values = {"g%d" % index: 25.0 + index for index in range(8)}
    rc_values = {"g%d" % index: 0.05 + index * 0.01 for index in range(8)}
    replacement = evaluate_replacement_gate(
        _decision(28.0, 20.0), _decision(0.09, 0.02), rv_values, rc_values,
        mfa_qc_status="COMPLETE", rv_id_values=rv_values, rc_id_values=rc_values,
    )
    cases["C_all_gates_pass"] = {
        "label": SYNTHETIC_LABEL,
        "expected": "REPLACEMENT_SUPPORT",
        "actual": replacement["status"],
    }
    return cases


if __name__ == "__main__":
    import json

    print(json.dumps(run_synthetic_fixtures(), ensure_ascii=False, indent=2))
