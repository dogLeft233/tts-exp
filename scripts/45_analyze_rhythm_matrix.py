#!/usr/bin/env python3
"""Analyze a completed valid-only rhythm counterfactual SyncNet matrix."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO / "runs/rhythm_timing/20260812_syncnet_smoke10/summary.json"
DEFAULT_OUTPUT = REPO / "runs/rhythm_timing/20260812_syncnet_smoke10/analysis.json"
BOOTSTRAP_SEED = 42
BOOTSTRAP_DRAWS = 10_000


def mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.fmean(values) if values else None


def bootstrap_mean_ci(values: list[float], *, seed: int, draws: int) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        estimates.append(statistics.fmean(values[rng.randrange(len(values))] for _ in values))
    estimates.sort()
    low = estimates[int(0.025 * (draws - 1))]
    high = estimates[int(0.975 * (draws - 1))]
    return [float(low), float(high)]


def paired_delta(
    by_condition: dict[str, dict[int, dict[str, Any]]],
    condition: str,
    baseline: str,
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    condition_rows = by_condition.get(condition, {})
    baseline_rows = by_condition.get(baseline, {})
    shared = sorted(set(condition_rows) & set(baseline_rows))
    delta_c = [float(condition_rows[sid]["sync_c"]) - float(baseline_rows[sid]["sync_c"]) for sid in shared]
    delta_d = [float(condition_rows[sid]["sync_d"]) - float(baseline_rows[sid]["sync_d"]) for sid in shared]
    return {
        "condition": condition,
        "baseline": baseline,
        "count": len(shared),
        "mean_delta_sync_c": mean(delta_c),
        "mean_delta_sync_d": mean(delta_d),
        "bootstrap_95_ci_delta_sync_c": bootstrap_mean_ci(delta_c, seed=seed, draws=draws),
        "bootstrap_95_ci_delta_sync_d": bootstrap_mean_ci(delta_d, seed=seed + 1, draws=draws),
        "sync_c_better_count": sum(value > 0.0 for value in delta_c),
        "sync_d_better_count": sum(value < 0.0 for value in delta_d),
        "per_sample": {
            str(sid): {"delta_sync_c": delta_c[index], "delta_sync_d": delta_d[index]}
            for index, sid in enumerate(shared)
        },
    }


def dose_response(by_condition: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    names = ["duration_alpha_0", "duration_alpha_0.5", "duration_alpha_1"]
    shared = sorted(set.intersection(*(set(by_condition.get(name, {})) for name in names)))
    increasing_c = 0
    decreasing_c = 0
    improving_d = 0
    worsening_d = 0
    per_sample: dict[str, Any] = {}
    for sid in shared:
        c = [float(by_condition[name][sid]["sync_c"]) for name in names]
        d = [float(by_condition[name][sid]["sync_d"]) for name in names]
        increasing_c += c[0] <= c[1] <= c[2]
        decreasing_c += c[0] >= c[1] >= c[2]
        improving_d += d[0] >= d[1] >= d[2]
        worsening_d += d[0] <= d[1] <= d[2]
        per_sample[str(sid)] = {"sync_c": c, "sync_d": d}
    return {
        "conditions": names,
        "count": len(shared),
        "sync_c_monotonic_increase_count": increasing_c,
        "sync_c_monotonic_decrease_count": decreasing_c,
        "sync_d_monotonic_improvement_count": improving_d,
        "sync_d_monotonic_worsening_count": worsening_d,
        "per_sample": per_sample,
    }


def analyze(input_path: Path, output_path: Path, *, draws: int) -> dict[str, Any]:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload.get("heldout_excluded") is not True or payload.get("valid_speaker") != "S0765":
        raise ValueError("analysis requires the valid-only S0765 matrix with heldout excluded")
    if payload.get("failures"):
        raise ValueError(f"matrix has {len(payload['failures'])} failures; do not analyze as complete")
    expected = int(payload["expected_scores"])
    if len(payload.get("scores", [])) != expected:
        raise ValueError(f"matrix incomplete: {len(payload.get('scores', []))}/{expected}")

    by_condition: dict[str, dict[int, dict[str, Any]]] = {}
    for score in payload["scores"]:
        by_condition.setdefault(str(score["condition"]), {})[int(score["sample_id"])] = score
    condition_summary = {}
    for condition, rows in sorted(by_condition.items()):
        ordered = list(rows.values())
        condition_summary[condition] = {
            "count": len(ordered),
            "mean_sync_c": mean(row["sync_c"] for row in ordered),
            "mean_sync_d": mean(row["sync_d"] for row in ordered),
            "mean_av_offset": mean(row["av_offset"] for row in ordered),
        }

    comparisons = {
        "global_length_vs_processed_identity": paired_delta(
            by_condition, "tts_global_natural_length", "tts_noop", seed=BOOTSTRAP_SEED, draws=draws
        ),
        "alpha_0_vs_processed_identity": paired_delta(
            by_condition, "duration_alpha_0", "tts_noop", seed=BOOTSTRAP_SEED + 10, draws=draws
        ),
        "alpha_0.5_vs_alpha_0": paired_delta(
            by_condition, "duration_alpha_0.5", "duration_alpha_0", seed=BOOTSTRAP_SEED + 20, draws=draws
        ),
        "alpha_1_vs_alpha_0": paired_delta(
            by_condition, "duration_alpha_1", "duration_alpha_0", seed=BOOTSTRAP_SEED + 30, draws=draws
        ),
        "alpha_1_vs_alpha_0.5": paired_delta(
            by_condition, "duration_alpha_1", "duration_alpha_0.5", seed=BOOTSTRAP_SEED + 40, draws=draws
        ),
        "natural_vs_processed_tts_identity": paired_delta(
            by_condition, "natural_noop", "tts_noop", seed=BOOTSTRAP_SEED + 50, draws=draws
        ),
    }
    result = {
        "schema_version": 1,
        "analysis": "valid_only_rhythm_counterfactual_paired",
        "input": str(input_path.resolve()),
        "valid_speaker": "S0765",
        "heldout_excluded": True,
        "bootstrap": {"draws": draws, "seed": BOOTSTRAP_SEED, "interval": "percentile_95"},
        "baseline_policy": {
            "processed_interventions": "compare against tts_noop from the same 16 kHz writeback chain",
            "phone_duration_allocation": "compare alpha=0.5 and alpha=1 against alpha=0",
            "raw_tts": "positive control only; not a direct baseline for processed interventions",
        },
        "condition_summary": condition_summary,
        "comparisons": comparisons,
        "duration_alpha_dose_response": dose_response(by_condition),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    input_path = args.input if args.input.is_absolute() else (REPO / args.input).resolve()
    output_path = args.output if args.output.is_absolute() else (REPO / args.output).resolve()
    if args.bootstrap_draws <= 0:
        raise SystemExit("--bootstrap-draws must be positive")
    result = analyze(input_path, output_path, draws=args.bootstrap_draws)
    print(json.dumps({"condition_summary": result["condition_summary"], "comparisons": result["comparisons"]}, ensure_ascii=False, indent=2))
    print(f"output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
