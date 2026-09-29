from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import ProtocolError, file_sha256, load_self_hashed, write_json
from .scoring import score_metrics
from .transform import temporal_contrast


def validate(run_root: Path, stage: str = "all") -> dict[str, Any]:
    errors: list[str] = []
    try:
        protocol = load_self_hashed(run_root / "protocol.json")
        drivers = load_self_hashed(run_root / "drivers/manifest.json")
        audit = load_self_hashed(run_root / "input_audit.json")
        if protocol.get("record_count") != 16 or protocol.get("source_group_count") != 8:
            errors.append("cohort_count")
        if audit.get("record_count") != 16 or audit.get("status") != "complete":
            errors.append("input_audit")
        if len(drivers.get("rows", [])) != 16:
            errors.append("driver_count")
        for row in drivers.get("rows", []):
            n = np.asarray(np.load(row["arms"]["N"]["path"], allow_pickle=False), dtype=np.float64)
            expected = temporal_contrast(n)
            for arm in ("SMOOTH", "SHARP"):
                actual = np.asarray(np.load(row["arms"][arm]["path"], allow_pickle=False), dtype=np.float64)
                if actual.shape != (80, 308) or not np.allclose(actual, expected[arm], atol=0.0, rtol=0.0):
                    errors.append(f"transform:{row['sample_id']}:{arm}")
                if file_sha256(row["arms"][arm]["path"]) != row["arms"][arm]["sha256"]:
                    errors.append(f"hash:{row['sample_id']}:{arm}")
        if stage in {"controls", "candidates", "analyze", "all"}:
            control = load_self_hashed(run_root / "control_analysis.json")
            if control.get("fresh_video_count") != 2 or control.get("fresh_score_count") != 4:
                errors.append("control_budget")
            if not control.get("control_pass") and (run_root / "candidate_scores/manifest.json").is_file():
                errors.append("candidate_present_after_control_failure")
        if stage in {"candidates", "analyze", "all"} and (run_root / "candidate_scores/manifest.json").is_file():
            candidates = load_self_hashed(run_root / "candidate_scores/manifest.json")
            if len(candidates.get("rows", [])) != 32:
                errors.append("candidate_count")
            for row in candidates.get("rows", []):
                score = row.get("score", {})
                if score.get("audio_arm") != "N" or list(score.get("U", {}).get("rows", [])) != list(config.U_ROWS):
                    errors.append(f"candidate_binding:{row.get('sample_id')}:{row.get('video_arm')}")
        if stage in {"analyze", "all"} and not (run_root / "analysis.json").is_file():
            errors.append("analysis_missing")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"exception:{type(exc).__name__}:{exc}")
    return {"schema_version": 1, "protocol_id": "wav2lip_natural_temporal_contrast", "stage": stage, "status": "PASS" if not errors else "FAIL", "error_count": len(errors), "errors": errors, "independent": True, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", default="all")
    args = parser.parse_args(argv)
    result = validate(args.run_root, args.stage)
    write_json(args.run_root / "validation.json", result)
    print(result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
