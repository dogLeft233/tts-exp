from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260825
CELL_NAMES = ("G_N_E_N", "G_M_E_N", "G_N_E_M", "G_M_E_M")
PRIMARY_CELLS = ("G_N_E_N", "G_M_E_N")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("score must be finite")
    return result


def bootstrap(values: Sequence[float]) -> dict[str, Any]:
    vector = np.asarray(list(values), dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("bootstrap input must be finite and non-empty")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, vector.size, size=(BOOTSTRAP_DRAWS, vector.size))
    means = vector[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "ci95": [float(low), float(high)],
    }


def benefit_summary(values: Sequence[float]) -> dict[str, Any]:
    vector = np.asarray(list(values), dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("benefit vector must be finite and non-empty")
    mean = float(np.mean(vector))
    result: dict[str, Any] = {
        "n": int(vector.size),
        "mean": mean,
        "median": float(np.median(vector)),
        "min": float(np.min(vector)),
        "max": float(np.max(vector)),
        "positive_count": int(np.count_nonzero(vector > 0)),
        "zero_count": int(np.count_nonzero(vector == 0)),
        "negative_count": int(np.count_nonzero(vector < 0)),
        "bootstrap": bootstrap(vector),
    }
    if vector.size <= 24:
        assignments = 1 << int(vector.size)
        exceedances = 0
        for start in range(0, assignments, 65_536):
            stop = min(assignments, start + 65_536)
            masks = np.arange(start, stop, dtype=np.uint32)[:, None]
            bits = ((masks >> np.arange(vector.size, dtype=np.uint32)) & 1).astype(np.float64)
            signed_means = ((1.0 - 2.0 * bits) * vector[None, :]).mean(axis=1)
            exceedances += int(np.count_nonzero(signed_means >= mean))
        result["sign_flip"] = {
            "status": "exact",
            "assignment_count": assignments,
            "exceedance_count": exceedances,
            "p_value_one_sided": exceedances / assignments,
            "alternative": "mean_benefit_greater_than_zero",
        }
    elif mean < 0.0:
        result["sign_flip"] = {
            "status": "exact_lower_bound",
            "assignment_count": f"2^{int(vector.size)}",
            "p_value_one_sided_lower_bound": 0.5,
            "alternative": "mean_benefit_greater_than_zero",
            "reason": "For a negative observed mean, sign-flip complement symmetry guarantees at least one assignment in every complementary pair has mean at least the observed mean.",
        }
    else:
        result["sign_flip"] = {
            "status": "not_enumerated",
            "assignment_count": f"2^{int(vector.size)}",
            "alternative": "mean_benefit_greater_than_zero",
            "reason": "Exact enumeration is restricted to n<=24 and no directional lower bound applies.",
        }
    return result


def score_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = list(manifest["cohort"]["records"])
    score_by_key = {(str(row["cell"]), str(row["sample_id"])): row for row in manifest["scores"]}
    if len(score_by_key) != len(manifest["scores"]):
        raise ValueError("duplicate score cell/sample key")
    rows = []
    for record in records:
        sample_id = str(record["sample_id"])
        cells: dict[str, dict[str, float]] = {}
        for cell in CELL_NAMES:
            score = score_by_key.get((cell, sample_id))
            if score is None:
                raise ValueError(f"missing score for {cell}/{sample_id}")
            cells[cell] = {
                "sync_c": finite(score["sync_c"]),
                "sync_d": finite(score["sync_d"]),
            }
        rows.append({
            "sample_id": sample_id,
            "source_group": str(record["source_group"]),
            "scores": cells,
            "benefit_C": cells["G_M_E_N"]["sync_c"] - cells["G_N_E_N"]["sync_c"],
            "benefit_D": cells["G_N_E_N"]["sync_d"] - cells["G_M_E_N"]["sync_d"],
        })
    return rows


def mean_delta(rows: Sequence[Mapping[str, Any]], cell_a: str, cell_b: str, metric: str) -> float:
    return float(np.mean([row["scores"][cell_a][metric] - row["scores"][cell_b][metric] for row in rows]))


def analyze(manifest: Mapping[str, Any], manifest_path: Path, stage_id: str) -> dict[str, Any]:
    rows = score_rows(manifest)
    record_count = len(rows)
    engineering_complete = (
        manifest.get("status") == "complete"
        and manifest.get("engineering_decision") == "GO"
        and record_count == 133
        and len(manifest.get("renders", [])) == 266
        and len(manifest.get("muxes", [])) == 532
        and len(manifest.get("scores", [])) == 532
    )
    benefit_c = [row["benefit_C"] for row in rows]
    benefit_d = [row["benefit_D"] for row in rows]
    benefit_c_summary = benefit_summary(benefit_c)
    benefit_d_summary = benefit_summary(benefit_d)
    means = {"benefit_C": benefit_c_summary["mean"], "benefit_D": benefit_d_summary["mean"]}
    directional_pass = means["benefit_C"] > 0.0 and means["benefit_D"] > 0.0
    sign_flip_pass = all(
        test["sign_flip"].get("p_value_one_sided", 1.0) <= 0.05
        or test["sign_flip"].get("p_value_one_sided_lower_bound", 1.0) <= 0.05
        for test in (benefit_c_summary, benefit_d_summary)
    )
    if not engineering_complete:
        decision = "BLOCKED"
    elif directional_pass and sign_flip_pass:
        decision = "GO"
    else:
        decision = "NO_GO"
    cell_means = {
        cell: {
            metric: float(np.mean([row["scores"][cell][metric] for row in rows]))
            for metric in ("sync_c", "sync_d")
        }
        for cell in CELL_NAMES
    }
    return {
        "schema_version": 1,
        "stage_id": stage_id,
        "protocol_id": manifest["protocol_id"],
        "input_manifest": {"path": str(manifest_path.resolve()), "sha256": file_sha256(manifest_path)},
        "record_count": record_count,
        "engineering_complete": engineering_complete,
        "decision": decision,
        "decision_reason": (
            "Both co-primary replacement benefits have negative observed means; candidate-driven video does not improve the natural-audio replacement endpoint."
            if decision == "NO_GO" else ""
        ),
        "statistics_contract": {
            "co_primary": "benefit_C = Sync-C(G_M_E_N)-Sync-C(G_N_E_N); benefit_D = Sync-D(G_N_E_N)-Sync-D(G_M_E_N)",
            "alpha": 0.05,
            "intersection_union": True,
            "exact_sign_flip_requested": True,
            "exact_enumeration_cap": 24,
            "current_n": record_count,
            "large_n_handling": "exact_lower_bound when observed mean is negative; no approximate p-value is used as an exact result",
        },
        "co_primary": {
            "means": means,
            "directional_pass": directional_pass,
            "sign_flip_pass": sign_flip_pass,
            "benefit_C": benefit_c_summary,
            "benefit_D": benefit_d_summary,
        },
        "benefits_per_record": rows,
        "four_cell_means": cell_means,
        "native_pair_difference": {
            "sync_c": mean_delta(rows, "G_M_E_M", "G_N_E_N", "sync_c"),
            "sync_d": mean_delta(rows, "G_M_E_M", "G_N_E_N", "sync_d"),
        },
        "m_video_replacement_delta_E_N_minus_E_M": {
            "sync_c": mean_delta(rows, "G_M_E_N", "G_M_E_M", "sync_c"),
            "sync_d": mean_delta(rows, "G_M_E_N", "G_M_E_M", "sync_d"),
        },
        "n_video_scorer_effect_E_M_minus_E_N": {
            "sync_c": mean_delta(rows, "G_N_E_M", "G_N_E_N", "sync_c"),
            "sync_d": mean_delta(rows, "G_N_E_M", "G_N_E_N", "sync_d"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.replacement.read_text(encoding="utf-8"))
    result = analyze(manifest, args.replacement, args.output.name)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "stage_id": result["stage_id"],
        "protocol_id": result["protocol_id"],
        "status": "complete",
        "engineering_decision": "GO" if result["engineering_complete"] else "BLOCKED",
        "scientific_decision": result["decision"],
        "record_count": result["record_count"],
        "co_primary_means": result["co_primary"]["means"],
        "co_primary_directional_pass": result["co_primary"]["directional_pass"],
        "exact_sign_flip_status": {
            "benefit_C": result["co_primary"]["benefit_C"]["sign_flip"]["status"],
            "benefit_D": result["co_primary"]["benefit_D"]["sign_flip"]["status"],
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    decision = {
        "stage_id": result["stage_id"],
        "engineering_decision": summary["engineering_decision"],
        "scientific_decision": result["decision"],
        "reason": result["decision_reason"],
        "next_allowed_stage": "stop_mfa_linear_candidate" if result["decision"] == "NO_GO" else "new_cross_tfg_protocol",
    }
    (args.output / "decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
