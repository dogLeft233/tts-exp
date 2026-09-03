from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.experiments.asr_sync_error_correlation.io import atomic_write_json


ESTIMANDS = ("baseline_C_gain", "baseline_D_gain", "control_C_adv", "control_D_adv")


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric: {name}")
    return result


def compute_estimands(scores: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    for condition in ("natural", "asr_targeted", "target_control_0", "target_control_1"):
        if condition not in scores:
            raise ValueError(f"missing condition score: {condition}")
    natural_c = _finite(scores["natural"]["sync_c"], "natural.sync_c")
    targeted_c = _finite(scores["asr_targeted"]["sync_c"], "asr_targeted.sync_c")
    natural_d = _finite(scores["natural"]["sync_d"], "natural.sync_d")
    targeted_d = _finite(scores["asr_targeted"]["sync_d"], "asr_targeted.sync_d")
    control_c = np.mean([_finite(scores[name]["sync_c"], f"{name}.sync_c") for name in ("target_control_0", "target_control_1")])
    control_d = np.mean([_finite(scores[name]["sync_d"], f"{name}.sync_d") for name in ("target_control_0", "target_control_1")])
    return {
        "baseline_C_gain": float(targeted_c - natural_c),
        "baseline_D_gain": float(natural_d - targeted_d),
        "control_C_adv": float(targeted_c - control_c),
        "control_D_adv": float(control_d - targeted_d),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no complete sample rows")
    unit_ids = [str(row["unit_id"]) for row in rows]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("source_group unit is not unique")
    summary: dict[str, Any] = {"count": len(rows), "unit_id": "source_group", "units": unit_ids, "estimands": {}}
    for name in ESTIMANDS:
        values = np.asarray([_finite(row[name], name) for row in rows], dtype=np.float64)
        summary["estimands"][name] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "wins_gt_zero": int(np.count_nonzero(values > 0)),
            "wins_ge_zero": int(np.count_nonzero(values >= 0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summary


def bootstrap_medians(rows: Sequence[Mapping[str, Any]], *, draws: int = 10000, seed: int = 20260901) -> dict[str, dict[str, Any]]:
    if not rows:
        raise ValueError("cannot bootstrap empty rows")
    unit_ids = [str(row["unit_id"]) for row in rows]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("bootstrap requires one row per source group")
    values = {name: np.asarray([_finite(row[name], name) for row in rows], dtype=np.float64) for name in ESTIMANDS}
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    indices = rng.integers(0, len(rows), size=(int(draws), len(rows)))
    output: dict[str, dict[str, Any]] = {}
    for name, array in values.items():
        medians = np.median(array[indices], axis=1)
        lower, upper = np.quantile(medians, [0.025, 0.975], method="linear")
        output[name] = {
            "observed_median": float(np.median(array)),
            "lower": float(lower),
            "upper": float(upper),
            "draws": int(draws),
            "seed": int(seed),
            "unit": "source_group",
            "percentile_method": "linear",
        }
    return output


def make_analysis_rows(samples: Sequence[Mapping[str, Any]], sync_records: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        source_group = str(sample["source_group"])
        scores: dict[str, Mapping[str, Any]] = {}
        for condition in ("natural", "asr_targeted", "target_control_0", "target_control_1"):
            record = sync_records.get(sample_id, {}).get(condition)
            if record is None or "official" not in record or record.get("parity", {}).get("passed") is not True:
                raise ValueError(f"incomplete or unverified SyncNet score: {sample_id}/{condition}")
            scores[condition] = {
                "sync_c": record["official"]["sync_c"],
                "sync_d": record["official"]["sync_d"],
            }
        row = {"sample_id": sample_id, "unit_id": source_group, "source_group": source_group, **compute_estimands(scores)}
        row["condition_scores"] = {name: dict(value) for name, value in scores.items()}
        rows.append(row)
    return rows


def decide(*, engineering_status: str, rows: Sequence[Mapping[str, Any]], bootstrap: Mapping[str, Mapping[str, Any]], min_units: int = 12) -> dict[str, Any]:
    if engineering_status != "GO":
        return {"engineering": "NO_GO", "science": "NOT_EVALUATED", "reason": "engineering_incomplete"}
    unit_ids = {str(row["unit_id"]) for row in rows}
    if len(rows) != len(unit_ids):
        return {"engineering": "NO_GO", "science": "NOT_EVALUATED", "reason": "source_group_unit_mismatch"}
    if len(unit_ids) < int(min_units):
        return {"engineering": "GO", "science": "INSUFFICIENT", "complete_units": len(unit_ids), "minimum_units": int(min_units)}
    lower_bounds = {name: float(bootstrap[name]["lower"]) for name in ESTIMANDS}
    science = "PROTOTYPE_SUPPORT" if all(value > 0.0 for value in lower_bounds.values()) else "NO_PROTOTYPE_SUPPORT"
    interval_interpretation = {
        name: ("strictly_positive" if float(bootstrap[name]["lower"]) > 0 else ("crosses_zero" if float(bootstrap[name]["upper"]) > 0 else "at_or_below_zero"))
        for name in ESTIMANDS
    }
    return {"engineering": "GO", "science": science, "complete_units": len(unit_ids), "minimum_units": int(min_units), "lower_bounds": lower_bounds, "interval_interpretation": interval_interpretation, "claim_boundary": "discovery-reuse fit-only conditional prototype; not cross-TFG or population-level replacement-safe evidence"}


def write_analysis(output_dir: Path, rows: Sequence[Mapping[str, Any]], *, engineering_status: str = "GO", draws: int = 10000, seed: int = 20260901, min_units: int = 12) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows)
    bootstrap = bootstrap_medians(rows, draws=draws, seed=seed)
    decision = decide(engineering_status=engineering_status, rows=rows, bootstrap=bootstrap, min_units=min_units)
    payload = {"schema_version": 1, "unit": "source_group", "rows": list(rows), "summary": summary, "bootstrap": bootstrap, "decision": decision}
    atomic_write_json(output_dir / "analysis.json", payload)
    return payload


def plot_sample(path: Path, sample_id: str, scores: Mapping[str, Mapping[str, Any]], budgets: Mapping[str, float], local: Mapping[str, Any] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = ("natural", "asr_targeted", "target_control_0", "target_control_1")
    c_values = [float(scores[name]["sync_c"]) for name in conditions]
    d_values = [float(scores[name]["sync_d"]) for name in conditions]
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    axes[0].bar(range(4), c_values, color="#315f8c")
    axes[0].set_title("Replacement Sync-C")
    axes[1].bar(range(4), d_values, color="#a45a3f")
    axes[1].set_title("Replacement Sync-D")
    local_values: list[float] = []
    local_labels: list[str] = []
    if local:
        for condition in conditions:
            diagnostic = local.get(condition)
            if isinstance(diagnostic, Mapping) and diagnostic.get("status") == "ok":
                inside = diagnostic.get("inside", {}).get("mean")
                outside = diagnostic.get("outside", {}).get("mean")
                if inside is not None and outside is not None:
                    local_values.extend([float(inside), float(outside)])
                    local_labels.extend([f"{condition}:in", f"{condition}:out"])
    if local_values:
        axes[2].bar(range(len(local_values)), local_values, color="#4c8c5a")
        axes[2].set_xticks(range(len(local_values)), local_labels, rotation=55, ha="right", fontsize=7)
        axes[2].set_title("Fixed-column Δdistance")
    else:
        axes[2].text(0.5, 0.5, "unavailable", ha="center", va="center")
        axes[2].set_title("Fixed-column Δdistance")
    for axis in axes[:2]:
        axis.set_xticks(range(4), ["N", "T", "C0", "C1"])
        axis.grid(axis="y", alpha=0.25)
    axes[2].grid(axis="y", alpha=0.25)
    figure.suptitle(f"{sample_id} | edit " + ", ".join(f"{name}={value:.3f}s" for name, value in budgets.items()))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
