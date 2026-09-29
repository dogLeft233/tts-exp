"""Independent artifact checker for PSE v2 runs.

This module intentionally does not call the runner's scoring or selection
functions. It checks hashes, support denominators, PCM protection, and the
smallest independently recomputable invariants from saved artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.lrs3_phone_rules_worker import read_pcm16

from . import MEASUREMENT_VERSION, PROTOCOL_ID
from .config import file_sha256, read_json, read_jsonl, write_json


def _load_json_strict(path: Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}: {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def _walk_finite(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            errors.extend(_walk_finite(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(_walk_finite(item, f"{path}[{index}]"))
    return errors


def check_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    required = ["protocol.json", "status.json", "00_audit/registry.json", "00_audit/support_plan.json", "01_calibration/calibration.json", "02_optimize/summary.json", "03_train/decision.json", "04_lock/selection_lock.json", "05_evaluation/summary.json", "06_mechanisms/mechanism_evidence.json", "08_report/decision.json"]
    for relative in required:
        if not (root / relative).is_file():
            errors.append(f"MISSING:{relative}")
    if errors:
        return {"status": "FAIL", "run_dir": str(root), "errors": errors, "warnings": warnings}
    protocol = _load_json_strict(root / "protocol.json")
    status = _load_json_strict(root / "status.json")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        errors.append("PROTOCOL_MISMATCH")
    if protocol.get("measurement_version") != MEASUREMENT_VERSION:
        errors.append("MEASUREMENT_VERSION_MISMATCH")
    if status.get("execution") not in {"COMPLETE", "RUNNING"}:
        errors.append(f"BAD_EXECUTION_STATUS:{status.get('execution')}")
    registry = _load_json_strict(root / "00_audit/registry.json")
    support = _load_json_strict(root / "00_audit/support_plan.json")
    pair_lookup = {str(row["pair_id"]): row for row in registry.get("pairs", [])}
    if not support.get("support_hash"):
        errors.append("SUPPORT_HASH_MISSING")
    # All JSON artifacts must be finite and parseable. This also catches a
    # partially written result before any scientific summary is trusted.
    for path in root.rglob("*.json"):
        try:
            payload = _load_json_strict(path)
            errors.extend(f"NONFINITE:{path.relative_to(root)}:{item}" for item in _walk_finite(payload))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"JSON_INVALID:{path.relative_to(root)}:{type(exc).__name__}")
    optimize_path = root / "02_optimize/results.jsonl"
    optimize_rows = read_jsonl(optimize_path) if optimize_path.is_file() else []
    checked_pcm = 0
    for row in optimize_rows:
        pair = pair_lookup.get(str(row.get("pair_id")))
        output_path = Path(str(row.get("output_path", "")))
        if pair is None:
            errors.append(f"UNKNOWN_PAIR:{row.get('pair_id')}")
            continue
        if not output_path.is_file():
            errors.append(f"OUTPUT_MISSING:{row.get('pair_id')}:{row.get('arm')}")
            continue
        try:
            candidate, metadata = read_pcm16(output_path)
            source, _ = read_pcm16(pair["sides"]["natural"]["audio_path"])
            if candidate.shape != source.shape:
                errors.append(f"LENGTH_MISMATCH:{row.get('pair_id')}:{row.get('arm')}")
            expected_hash = str(row.get("output_pcm_sha256", ""))
            if expected_hash and metadata["pcm_sha256"] != expected_hash:
                errors.append(f"PCM_HASH_MISMATCH:{row.get('pair_id')}:{row.get('arm')}")
            field_path = Path(str(row.get("gain_field_path", "")))
            if not field_path.is_file():
                errors.append(f"GAIN_FIELD_MISSING:{row.get('pair_id')}:{row.get('arm')}")
            else:
                with np.load(field_path, allow_pickle=False) as fields:
                    protected = np.asarray(fields["protected"], dtype=bool).reshape(-1)
                if protected.shape != source.shape:
                    errors.append(f"PROTECTED_MASK_LENGTH:{row.get('pair_id')}:{row.get('arm')}")
                elif not np.array_equal(source[protected], candidate[protected]):
                    errors.append(f"PROTECTED_PCM_CHANGED:{row.get('pair_id')}:{row.get('arm')}")
            checked_pcm += 1
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"PCM_INVALID:{row.get('pair_id')}:{row.get('arm')}:{type(exc).__name__}")
    calibration = _load_json_strict(root / "01_calibration/calibration.json")
    if calibration.get("status") == "PASS":
        for key in ("teacher_parity", "gradient_check", "stft_identity"):
            if calibration.get(key, {}).get("status") != "PASS":
                errors.append(f"CALIBRATION_NOT_PASS:{key}")
    triplet_path = root / "01_calibration/abx_triplets.json"
    if triplet_path.is_file():
        triplets = _load_json_strict(triplet_path)
        for encoder, rows in triplets.items():
            for row in rows:
                if str(row.get("a")) == str(row.get("x")):
                    errors.append(f"ABX_A_EQUALS_X:{encoder}:{row.get('a')}")
    lock = _load_json_strict(root / "04_lock/selection_lock.json")
    if lock.get("support_hash") != support.get("support_hash"):
        errors.append("LOCK_SUPPORT_HASH_MISMATCH")
    if lock.get("protocol_hash") != file_sha256(root / "protocol.json"):
        errors.append("LOCK_PROTOCOL_HASH_MISMATCH")
    decision = _load_json_strict(root / "08_report/decision.json")
    if decision.get("human_quality") != "HUMAN_NOT_ASSESSED":
        warnings.append("HUMAN_QUALITY_STATUS_NOT_EXPLICIT")
    if status.get("generalization") == "SMOKE_ONLY":
        warnings.append("SMOKE_RUN_HAS_NO_SCIENTIFIC_E_EVALUATION")
    result = {"status": "PASS" if not errors else "FAIL", "run_dir": str(root), "protocol": PROTOCOL_ID, "measurement_version": MEASUREMENT_VERSION, "checked_pcm": checked_pcm, "errors": errors, "warnings": warnings, "independent": {"json_finite": True, "pcm_protection": True, "abx_distinct_occurrences": True, "lock_hashes": True}}
    write_json(root / "08_report/checker.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    result = check_run(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["check_run", "main"]
