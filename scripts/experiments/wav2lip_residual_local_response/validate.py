from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import load_self_hashed, write_json
from .support import chunk_columns, row_supports


def validate(run_root: Path, stage: str = "all") -> dict[str, Any]:
    errors: list[str] = []
    try:
        protocol = load_self_hashed(run_root / "protocol.json")
        audit = load_self_hashed(run_root / "input_audit.json")
        exposure = load_self_hashed(run_root / "exposure.json")
        if protocol.get("record_count") != 16 or protocol.get("source_group_count") != 8: errors.append("cohort_count")
        if audit.get("record_count") != 16 or len(exposure.get("rows", [])) != 16: errors.append("input_or_exposure_count")
        expected = row_supports()
        for record in exposure.get("rows", []):
            for row in record.get("rows", []):
                actual = np.asarray(row.get("support", []), dtype=np.int64)
                if not np.array_equal(actual, expected[int(row["row"])]): errors.append(f"support:{record.get('sample_id')}:{row.get('row')}")
        if (run_root / "reused_manifest.json").is_file():
            reused = load_self_hashed(run_root / "reused_manifest.json")
            if reused.get("fresh_video_count") != 0 or reused.get("fresh_score_count") != 0: errors.append("fresh_budget")
        if stage in {"analyze", "all"} and not (run_root / "analysis.json").is_file(): errors.append("analysis_missing")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"exception:{type(exc).__name__}:{exc}")
    return {"schema_version": 1, "protocol_id": "wav2lip_residual_local_response", "stage": stage, "status": "PASS" if not errors else "FAIL", "error_count": len(errors), "errors": errors, "independent": True, "gpu_used": False, "fresh_video_count": 0, "fresh_score_count": 0, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", type=Path, required=True); parser.add_argument("--stage", default="all"); args = parser.parse_args(argv)
    result = validate(args.run_root, args.stage); write_json(args.run_root / "validation.json", result); print(result); return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
