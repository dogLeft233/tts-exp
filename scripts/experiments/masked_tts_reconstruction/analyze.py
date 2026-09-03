"""Hierarchical paired analysis and conservative feasibility decision."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, EVAL_GROUPS
from .protocol import canonical_json, sha256_text, write_json


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    result = float(np.median(np.asarray(values, dtype=np.float64)))
    if not np.isfinite(result):
        raise FloatingPointError("median is not finite")
    return result


def _relative_reduction(natural: float, full: float) -> float:
    if natural <= 0:
        raise ValueError("NAT_ONLY loss must be positive for relative reduction")
    return (natural - full) / natural


def analyze_evaluation(
    evaluation: Mapping[str, Any],
    mask_manifest: Mapping[str, Any],
    output_dir: Path,
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], dict[str, Any]]:
    records = [row for row in evaluation.get("records", []) if row.get("status") == "complete"]
    required_conditions = {"FULL_CORRECT", "FULL_ZERO", "NAT_ONLY"}
    cells: dict[tuple[int, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    shuffled_cells: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in records:
        condition = str(row.get("condition"))
        key = (int(row["seed"]), str(row["mask_sha256"]))
        if condition in required_conditions:
            if condition in cells[key]:
                raise ValueError(f"duplicate required cell {key} {condition}")
            cells[key][condition] = row
        elif condition == "FULL_SHUFFLED":
            if key in shuffled_cells:
                raise ValueError(f"duplicate shuffled diagnostic cell {key}")
            shuffled_cells[key] = row
    masks = {str(row["mask_sha256"]): row for row in mask_manifest["masks"] if row["prototype_split"] == "evaluation"}
    if not masks:
        raise ValueError("evaluation mask manifest is empty")
    mask_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for (seed, mask_hash), condition_rows in sorted(cells.items()):
        if set(condition_rows) != required_conditions:
            missing.append({"seed": seed, "mask_sha256": mask_hash, "conditions": sorted(condition_rows)})
            continue
        mask = masks.get(mask_hash)
        if mask is None:
            raise ValueError(f"evaluation cell references unknown mask {mask_hash}")
        full = condition_rows["FULL_CORRECT"]
        zero = condition_rows["FULL_ZERO"]
        natural = condition_rows["NAT_ONLY"]
        identity_fields = ("sample_id", "source_group", "target_sha256", "natural_input_sha256", "masked_support_sha256", "target_core_sha256")
        for field in identity_fields:
            if not (full.get(field) == zero.get(field) == natural.get(field)):
                raise ValueError(f"shared evaluation input mismatch at {mask_hash}: {field}")
        losses = {name: float(condition_rows[name]["loss"]["total"]) for name in required_conditions}
        if not all(np.isfinite(value) for value in losses.values()):
            raise FloatingPointError(f"non-finite required loss at {mask_hash}")
        shuf = shuffled_cells.get((seed, mask_hash))
        shuf_gap = None if shuf is None else float(shuf["loss"]["total"]) - losses["FULL_CORRECT"]
        row = {
            "seed": seed,
            "mask_sha256": mask_hash,
            "sample_id": str(mask["sample_id"]),
            "source_group": str(mask["source_group"]),
            "full_correct": losses["FULL_CORRECT"],
            "full_zero": losses["FULL_ZERO"],
            "nat_only": losses["NAT_ONLY"],
            "nat_gain": losses["NAT_ONLY"] - losses["FULL_CORRECT"],
            "zero_gap": losses["FULL_ZERO"] - losses["FULL_CORRECT"],
            "shuf_gap": shuf_gap,
        }
        mask_rows.append(row)
    expected_keys = {(int(seed), mask_hash) for seed in evaluation.get("seeds", []) for mask_hash in masks}
    missing.extend({"seed": seed, "mask_sha256": mask_hash, "conditions": []} for seed, mask_hash in sorted(expected_keys - set(cells)))
    if missing:
        raise ValueError(f"incomplete required evaluation cells: {missing[:3]}")
    by_group_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in mask_rows:
        by_group_seed[(row["source_group"], int(row["seed"]))].append(row)
    record_rows: list[dict[str, Any]] = []
    group_seed_rows: list[dict[str, Any]] = []
    for (group, seed), group_masks in sorted(by_group_seed.items()):
        by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_masks:
            by_record[row["sample_id"]].append(row)
        for sample_id, sample_rows in sorted(by_record.items(), key=lambda item: item[0].encode("utf-8")):
            record_rows.append({
                "source_group": group,
                "seed": seed,
                "sample_id": sample_id,
                "mask_count": len(sample_rows),
                **{key: _median([float(row[key]) for row in sample_rows]) for key in ("full_correct", "full_zero", "nat_only", "nat_gain", "zero_gap")},
            })
        record_group = [row for row in record_rows if row["source_group"] == group and row["seed"] == seed]
        group_seed_rows.append({
            "source_group": group,
            "seed": seed,
            "record_count": len(record_group),
            "mask_count": len(group_masks),
            **{key: _median([float(row[key]) for row in record_group]) for key in ("full_correct", "full_zero", "nat_only", "nat_gain", "zero_gap")},
        })
    final_group_rows: list[dict[str, Any]] = []
    for group in EVAL_GROUPS:
        seed_rows = [row for row in group_seed_rows if row["source_group"] == group]
        if len(seed_rows) != 3:
            raise ValueError(f"evaluation group {group} does not have three seed rows")
        final_group_rows.append({
            "source_group": group,
            "seed_count": len(seed_rows),
            **{key: _median([float(row[key]) for row in seed_rows]) for key in ("full_correct", "full_zero", "nat_only", "nat_gain", "zero_gap")},
        })
    seed_overall = {
        str(seed): {
            "seed": seed,
            "group_count": len([row for row in group_seed_rows if int(row["seed"]) == seed]),
            "overall_median_nat_gain": _median([float(row["nat_gain"]) for row in group_seed_rows if int(row["seed"]) == seed]),
        }
        for seed in sorted({int(row["seed"]) for row in group_seed_rows})
    }
    group_gains = np.asarray([float(row["nat_gain"]) for row in final_group_rows], dtype=np.float64)
    if len(group_gains) != 4 or tuple(row["source_group"] for row in final_group_rows) != EVAL_GROUPS:
        raise ValueError("final evaluation groups do not match frozen order")
    rng = np.random.Generator(np.random.PCG64(int(bootstrap_seed)))
    boot = np.empty(int(bootstrap_draws), dtype=np.float64)
    for index in range(int(bootstrap_draws)):
        draw = rng.integers(0, len(group_gains), size=len(group_gains))
        boot[index] = np.median(group_gains[draw])
    interval = [float(np.quantile(boot, 0.025, method="linear")), float(np.quantile(boot, 0.975, method="linear"))]
    median_nat = _median([float(row["nat_only"]) for row in final_group_rows])
    median_full = _median([float(row["full_correct"]) for row in final_group_rows])
    relative_reduction = _relative_reduction(median_nat, median_full)
    gates = {
        "bootstrap_lower_bound_gt_zero": interval[0] > 0,
        "all_evaluation_groups_positive": bool(np.all(group_gains > 0)),
        "median_relative_reduction_ge_5_percent": relative_reduction >= 0.05,
        "all_seeds_overall_median_positive": all(float(row["overall_median_nat_gain"]) > 0 for row in seed_overall.values()),
    }
    readiness = dict(mask_manifest.get("readiness", {}))
    sufficient = bool(readiness.get("sufficient", False))
    engineering = "GO"
    if not sufficient:
        science = "INSUFFICIENT"
    else:
        science = "ALGORITHM_FEASIBLE" if all(gates.values()) else "NO_ALGORITHM_FEASIBILITY_SUPPORT"
    shuf_values = [float(row["shuf_gap"]) for row in mask_rows if row["shuf_gap"] is not None]
    analysis = {
        "schema_version": 1,
        "status": "GO",
        "primary": "FULL_CORRECT_vs_NAT_ONLY",
        "mask_rows": mask_rows,
        "record_rows": record_rows,
        "group_seed_rows": group_seed_rows,
        "final_group_rows": final_group_rows,
        "seed_overall": seed_overall,
        "bootstrap": {"draws": int(bootstrap_draws), "seed": int(bootstrap_seed), "unit": "whole_source_group", "group_count": len(group_gains), "nat_gain_median": float(np.median(group_gains)), "ci95": interval, "method": "numpy_quantile_linear"},
        "relative_reduction": {"median_nat_only": median_nat, "median_full_correct": median_full, "value": relative_reduction},
        "sensitivity_non_promoting": {"zero_gap": "reported in mask/record/group rows only", "shuf_gap": {"available_mask_seed_cells": len(shuf_values), "median": _median(shuf_values) if shuf_values else None, "diagnostic_only": True}},
        "gates": gates,
        "readiness": readiness,
        "sealed_splits_accessed": False,
    }
    decision = {
        "schema_version": 1,
        "engineering": engineering,
        "science": science,
        "failed_gates": [name for name, passed in gates.items() if not passed] if science == "NO_ALGORITHM_FEASIBILITY_SUPPORT" else [],
        "primary": "nat_gain = L_total(NAT_ONLY) - L_total(FULL_CORRECT)",
        "claim": "paired phone-aligned TTS features improve masked natural log-mel reconstruction under this frozen fit-only task" if science == "ALGORITHM_FEASIBLE" else "no feasibility claim",
        "claim_boundary": ["not phonetic-content causality", "not TTS clarity retention", "not prosody disentanglement", "not waveform reachability", "not Wav2Lip/SyncNet gain", "not replacement effect", "not cross-TFG transfer", "not population generalization"],
        "gates": gates,
        "sealed_splits_accessed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "analysis.json", analysis, overwrite=True)
    write_json(output_dir / "decision.json", decision, overwrite=True)
    return analysis, decision
