#!/usr/bin/env python3
"""Analyze pause-allocation SyncNet interventions against strict identity."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = REPO / "runs/rhythm_timing/20260812_pause_syncnet_smoke10/summary.json"
DEFAULT_SOURCE = REPO / "runs/rhythm_timing/20260812_pause_counterfactual_smoke10/summary.json"
DEFAULT_OUTPUT = REPO / "runs/rhythm_timing/20260812_pause_syncnet_smoke10/analysis.json"
SEED = 42
DRAWS = 10_000
BASELINE = "pause_alpha_0"


def bootstrap_ci(values: list[float], seed: int) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    estimates = sorted(
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(DRAWS)
    )
    return [estimates[int(0.025 * (DRAWS - 1))], estimates[int(0.975 * (DRAWS - 1))]]


def magnitude(record: dict[str, Any]) -> float:
    source = record["source_gap_sample_counts"]
    target = record["target_gap_sample_counts"]
    total = sum(source)
    if total <= 0:
        return 0.0
    return sum(abs(int(left) - int(right)) for left, right in zip(source, target)) / (2.0 * total)


def summarize(
    condition: str,
    by_score: dict[str, dict[int, dict[str, Any]]],
    by_source: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, Any]:
    shared = sorted(set(by_score.get(BASELINE, {})) & set(by_score.get(condition, {})))
    rows = []
    for sample_id in shared:
        source_record = by_source[condition][sample_id]
        baseline = by_score[BASELINE][sample_id]
        intervention = by_score[condition][sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "magnitude": magnitude(source_record),
                "delta_sync_c": float(intervention["sync_c"]) - float(baseline["sync_c"]),
                "delta_sync_d": float(intervention["sync_d"]) - float(baseline["sync_d"]),
                "delta_av_offset": int(intervention["av_offset"]) - int(baseline["av_offset"]),
            }
        )

    def aggregate(selected: list[dict[str, Any]], seed_offset: int) -> dict[str, Any]:
        delta_c = [row["delta_sync_c"] for row in selected]
        delta_d = [row["delta_sync_d"] for row in selected]
        return {
            "count": len(selected),
            "mean_intervention_magnitude": statistics.fmean(row["magnitude"] for row in selected) if selected else None,
            "mean_delta_sync_c": statistics.fmean(delta_c) if delta_c else None,
            "bootstrap_95_ci_delta_sync_c": bootstrap_ci(delta_c, SEED + seed_offset),
            "sync_c_better_count": sum(value > 0.0 for value in delta_c),
            "mean_delta_sync_d": statistics.fmean(delta_d) if delta_d else None,
            "bootstrap_95_ci_delta_sync_d": bootstrap_ci(delta_d, SEED + seed_offset + 1),
            "sync_d_better_count": sum(value < 0.0 for value in delta_d),
            "av_offset_changed_count": sum(row["delta_av_offset"] != 0 for row in selected),
        }

    active = [row for row in rows if row["magnitude"] > 0.0]
    return {
        "condition": condition,
        "baseline": BASELINE,
        "all_samples": aggregate(rows, 10),
        "active_intervention_samples": aggregate(active, 20),
        "per_sample": {str(row["sample_id"]): row for row in rows},
    }


def analyze(matrix_path: Path, source_path: Path, output_path: Path) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if matrix.get("heldout_excluded") is not True or source.get("heldout_excluded") is not True:
        raise ValueError("pause analysis requires heldout-excluded artifacts")
    if matrix.get("failures"):
        raise ValueError("pause matrix contains failures")
    if len(matrix.get("scores", [])) != int(matrix["expected_scores"]):
        raise ValueError("pause matrix is incomplete")
    by_score: dict[str, dict[int, dict[str, Any]]] = {}
    for row in matrix["scores"]:
        by_score.setdefault(str(row["condition"]), {})[int(row["sample_id"])] = row
    by_source: dict[str, dict[int, dict[str, Any]]] = {}
    for row in source["records"]:
        by_source.setdefault(str(row["condition"]), {})[int(row["sample_id"])] = row
    conditions = sorted(set(by_score) - {BASELINE})
    missing_source = [condition for condition in conditions if condition not in by_source]
    if missing_source:
        raise ValueError(f"missing source metadata for {missing_source}")
    result = {
        "schema_version": 1,
        "analysis": "valid_only_pause_allocation_paired",
        "matrix": str(matrix_path.resolve()),
        "source": str(source_path.resolve()),
        "valid_speaker": "S0765",
        "heldout_excluded": True,
        "baseline": BASELINE,
        "bootstrap": {"draws": DRAWS, "seed": SEED, "interval": "percentile_95"},
        "active_policy": "target gap vector differs from source gap vector",
        "comparisons": {
            condition: summarize(condition, by_score, by_source)
            for condition in conditions
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    matrix = args.matrix if args.matrix.is_absolute() else (REPO / args.matrix).resolve()
    source = args.source if args.source.is_absolute() else (REPO / args.source).resolve()
    output = args.output if args.output.is_absolute() else (REPO / args.output).resolve()
    result = analyze(matrix, source, output)
    for condition, comparison in result["comparisons"].items():
        print(condition, json.dumps(comparison["active_intervention_samples"], ensure_ascii=False))
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
