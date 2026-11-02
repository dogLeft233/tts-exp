"""Fixed lag-zero extraction from already accepted report-41/report-42 JSON."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "original_event_alignment_diagnostic_attempt01.json"
SAMPLES = ("BAC009S0022W0123", "BAC009S0024W0122")
ARMS = ("native_height_zero", "native_height_plus", "native_height_minus")
INPUTS = {
    "report41_old_I_effect": (
        "geometry_effect_attempt03.json",
        "1b88218d53f88a73575e3acfbd413fd8870b843748c872b6016a38cc174473b4",
    ),
    "report42_I_plus_4_effect": (
        "output_owned_geometry_effect_attempt01.json",
        "bb5fa06e5fd6b1d8a8f1340752b1fdad4c4d6cc733434cb0ba95a20e06fc4967",
    ),
    "report41_existing_score_QA": (
        "root_actual_response_qa.json",
        "72ef059b13a3a0cfd4f398b2f203224ec104a2072e18635fa5f124a26ad2e538",
    ),
    "report42_existing_score_QA": (
        "root_actual_output_owned_response_qa.json",
        "f470d1aa4a926cafb65f73555e6957797bf3b7176b7bbaf8a30348593d3f32d9",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pinned(name: str) -> tuple[dict, dict]:
    rel, expected = INPUTS[name]
    path = ROOT / rel
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"input hash mismatch for {rel}: {actual}")
    return json.loads(path.read_text(encoding="utf-8")), {
        "path": str(path),
        "sha256": actual,
        "bytes": path.stat().st_size,
    }


def indexed_curve(cell: dict, arm: str, scale: str) -> dict[str, float]:
    values = cell["C_curves"][arm][scale]["C_high_minus_low_by_absolute_lag"]
    if len(values) != 17:
        raise ValueError(f"expected frozen absolute S17, got {len(values)} points")
    c0 = float(values[8])
    h0 = c0 - (float(values[4]) + float(values[12])) / 2.0
    return {"C0": c0, "H0": h0}


def score_extract(score_qa: dict, sample_id: str, arm: str) -> dict:
    row = next(
        r for r in score_qa["score_rows"]
        if r["model"] == "musetalk" and r["sample_id"] == sample_id and r["arm"] == arm
    )
    pair = next(
        (r for r in score_qa["score_pairs"]
         if r["model"] == "musetalk" and r["sample_id"] == sample_id and r["arm"] == arm),
        None,
    )
    delta = pair["delta_vs_new_zero"] if pair else None
    result = {}
    for scope in ("full", "guard20"):
        values = row[scope]
        d = delta[scope] if delta else {"D0": 0.0, "Cbest": 0.0}
        result[scope] = {
            "D0": float(values["D0"]),
            "Cbest": float(values["Cbest"]),
            "delta_D0_vs_zero": float(d["D0"]),
            "delta_Cbest_vs_zero": float(d["Cbest"]),
        }
    return result


def main() -> int:
    start = utc_now()
    source_path = Path(__file__).resolve()
    source_sha = sha256(source_path)
    old_effect, old_effect_pin = load_pinned("report41_old_I_effect")
    new_effect, new_effect_pin = load_pinned("report42_I_plus_4_effect")
    old_score, old_score_pin = load_pinned("report41_existing_score_QA")
    new_score, new_score_pin = load_pinned("report42_existing_score_QA")

    cases = []
    operations = (
        ("report41_old_I", old_effect, old_score),
        ("report42_output_owner_I_plus_4", new_effect, new_score),
    )
    for operation, effect, score_qa in operations:
        for sample_id in SAMPLES:
            cell = next(
                c for c in effect["cells"]
                if c["model"] == "musetalk" and c["sample_id"] == sample_id
            )
            arms = {}
            for arm in ARMS:
                arms[arm] = {
                    "raw": indexed_curve(cell, arm, "raw"),
                    "z": indexed_curve(cell, arm, "z"),
                    "existing_score": score_extract(score_qa, sample_id, arm),
                }
            cases.append({
                "operation": operation,
                "model": "musetalk",
                "sample_id": sample_id,
                "fixed_S17_lags": [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                "event_lag_frames": 0,
                "curve_definition": "C = C_high_minus_low_by_absolute_lag; H0=C(0)-mean(C(-4),C(+4))",
                "arms": arms,
            })

    output = {
        "schema": "original_event_alignment_diagnostic_v1",
        "status": "ORIGINAL_EVENT_ALIGNMENT_DIAGNOSTIC_ONLY",
        "interpretation_limit": "Does not change the fixed-p4 primary screen, infer p4 as physical delay, or select a lag peak.",
        "extraction": {
            "curves_rebuilt": False,
            "only_existing_JSON_values_used": True,
            "fixed_index_rule": {"C0": 8, "C_minus4": 4, "C_plus4": 12},
            "zero_C0_may_be_negative": True,
            "score_delta_source": "existing ROOT score_pairs delta_vs_new_zero; zero arm delta recorded as 0",
        },
        "input_pins": {
            "report41_old_I_effect": old_effect_pin,
            "report42_I_plus_4_effect": new_effect_pin,
            "report41_existing_score_QA": old_score_pin,
            "report42_existing_score_QA": new_score_pin,
        },
        "execution": {
            "pid": os.getpid(),
            "start_utc": start,
            "end_utc": None,
            "argv": sys.argv,
            "cwd": os.getcwd(),
            "source_path": str(source_path),
            "source_sha256": source_sha,
            "return_code": None,
        },
        "cases": cases,
    }
    output["execution"]["end_utc"] = utc_now()
    output["execution"]["return_code"] = 0
    tmp = OUT.with_suffix(OUT.suffix + ".tmp")
    tmp.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
