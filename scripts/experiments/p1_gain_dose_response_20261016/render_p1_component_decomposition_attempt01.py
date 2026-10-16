"""Build a derived component table and one combined P1 curve/decomposition SVG.

Inputs are the frozen analyzer's JSON/CSV outputs only. This script does not
read score_results.json, run models, change the primary contrast, or derive C
from a cohort-averaged distance curve. It reuses the analyzer's primary C
interval and 31-lag curve-summary rows verbatim; bootstrap intervals are added
only for the predeclared descriptive component contrasts other than C.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SEED = 20261016
BOOTSTRAP_DRAWS = 20_000
PRIMARY_CONDITIONS = ("RAW_R1", "RAW_R2", "FIXED_R1", "FIXED_R2")
DOSE_ORDER = ("DOSE_M6", "DOSE_M3", "DOSE_P3", "DOSE_P6")
DOSE_DB = {"DOSE_M6": -6.0, "DOSE_M3": -3.0, "DOSE_P3": 3.0, "DOSE_P6": 6.0}
METRICS = ("C", "B", "Dmin", "D0", "anchor_D", "anchor_C", "search_gain")
CURVE_ORDER = ("RAW_R1", "RAW_MEAN", "FIXED_MEAN", "DOSE_M6", "DOSE_M3", "DOSE_P3", "DOSE_P6")
CURVE_LABELS = {
    "RAW_R1": "RAW_R1",
    "RAW_MEAN": "RAW mean (r1/r2)",
    "FIXED_MEAN": "FIXED mean (r1/r2)",
    "DOSE_M6": "−6 dB",
    "DOSE_M3": "−3 dB",
    "DOSE_P3": "+3 dB",
    "DOSE_P6": "+6 dB",
}
CURVE_COLORS = {
    "RAW_R1": "#4c5661",
    "RAW_MEAN": "#277da1",
    "FIXED_MEAN": "#e76f51",
    "DOSE_M6": "#43aa8b",
    "DOSE_M3": "#90be6d",
    "DOSE_P3": "#577590",
    "DOSE_P6": "#9b5de5",
}
COMPONENT_LABELS = {
    "C": "ΔC · Sync-C",
    "B": "ΔB · median baseline",
    "Dmin": "ΔDmin · best-lag distance",
    "D0": "ΔD0 · zero-lag distance",
    "anchor_D": "Δanchor_D · fixed RAW_R1 anchor distance",
    "anchor_C": "Δanchor_C · fixed-anchor score",
    "search_gain": "Δsearch_gain · D0 − Dmin",
}


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _number(value: Any, *, label: str) -> float:
    result = _finite(value)
    if result is None:
        raise ValueError(f"{label} must be finite")
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path.name} contains no data rows")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bootstrap_indices(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("descriptive paired bootstrap requires at least one complete speaker")
    rng = np.random.Generator(np.random.PCG64(SEED))
    return rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n), endpoint=False)


def _percentile_ci(values: list[float], indices: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    draws = array[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975], method="linear")
    return float(low), float(high)


def _read_optional_metric(row: dict[str, str], metric: str, *, speaker_id: str) -> float | None:
    value = row.get(metric)
    if value is None or not str(value).strip():
        return None
    result = _finite(value)
    if result is None:
        raise ValueError(f"{speaker_id}/{row.get('condition_id')}.{metric} must be blank or finite")
    return result


def compute_decomposition(
    summary: dict[str, Any],
    condition_rows: list[dict[str, str]],
    primary_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Compute repeat-averaged paired deltas, and only the predeclared descriptors."""
    planned_ids = summary.get("planned_speakers")
    if not isinstance(planned_ids, list) or not planned_ids:
        raise ValueError("analysis_summary.json must carry planned_speakers")
    planned_ids = [str(x) for x in planned_ids]
    n_planned = len(planned_ids)
    if len(set(planned_ids)) != n_planned or int(summary.get("n_planned_speakers", -1)) != n_planned:
        raise ValueError("planned speaker order/count is invalid")

    condition_map: dict[tuple[str, str], dict[str, str]] = {}
    for row in condition_rows:
        speaker_id, condition_id = str(row.get("speaker_id", "")), str(row.get("condition_id", ""))
        key = (speaker_id, condition_id)
        if not speaker_id or not condition_id or key in condition_map:
            raise ValueError("speaker_conditions.csv contains a blank or duplicate speaker-condition row")
        if speaker_id not in set(planned_ids):
            raise ValueError(f"unexpected speaker in speaker_conditions.csv: {speaker_id}")
        condition_map[key] = row

    primary_map: dict[str, dict[str, str]] = {}
    for row in primary_rows:
        speaker_id = str(row.get("speaker_id", ""))
        if not speaker_id or speaker_id in primary_map:
            raise ValueError("speaker_primary.csv contains a blank or duplicate speaker_id")
        primary_map[speaker_id] = row
    if list(primary_map) != planned_ids:
        raise ValueError("speaker_primary.csv must contain every planned speaker in frozen order")

    primary_summary = summary.get("primary_fixed_minus_raw")
    if not isinstance(primary_summary, dict):
        raise ValueError("analysis_summary.json lacks primary_fixed_minus_raw")
    if int(primary_summary.get("n_planned", -1)) != n_planned:
        raise ValueError("primary planned denominator differs from planned_speakers")
    if int(primary_summary.get("bootstrap_seed", SEED)) != SEED or int(primary_summary.get("bootstrap_draws", BOOTSTRAP_DRAWS)) != BOOTSTRAP_DRAWS:
        raise ValueError("primary bootstrap metadata differs from the frozen SPEC")

    component_rows: list[dict[str, Any]] = []
    dose_component_rows: list[dict[str, Any]] = []
    complete_ids: list[str] = []
    family_complete_ids: list[str] = []
    for speaker_id in planned_ids:
        primary_row = primary_map[speaker_id]
        declared_complete = str(primary_row.get("primary_complete", "")).strip().lower() == "true"
        declared_family_complete = str(primary_row.get("family_complete", "")).strip().lower() == "true"
        values_by_condition: dict[str, dict[str, float]] = {}
        conditions_pass = True
        for condition_id in PRIMARY_CONDITIONS:
            row = condition_map.get((speaker_id, condition_id))
            if row is None or str(row.get("status", "")).strip().upper() != "PASS":
                conditions_pass = False
                continue
            metrics: dict[str, float] = {}
            for metric in METRICS:
                value = _read_optional_metric(row, metric, speaker_id=speaker_id)
                if value is None:
                    raise ValueError(f"{speaker_id}/{condition_id}.{metric} is blank on a PASS row")
                metrics[metric] = value
            values_by_condition[condition_id] = metrics
        if declared_complete and (not conditions_pass or len(values_by_condition) != len(PRIMARY_CONDITIONS)):
            raise ValueError(f"{speaker_id}: primary_complete disagrees with the four condition rows")
        if not declared_complete:
            if primary_row.get("primary_delta_C", "").strip():
                raise ValueError(f"{speaker_id}: incomplete primary row has a primary_delta_C")
            component_rows.append({"speaker_id": speaker_id, "primary_complete": False, **{f"raw_mean_{m}": None for m in METRICS}, **{f"fixed_mean_{m}": None for m in METRICS}, **{f"delta_{m}": None for m in METRICS}})
        else:
            complete_ids.append(speaker_id)
            row_out = {"speaker_id": speaker_id, "primary_complete": True}
            for metric in METRICS:
                raw_mean = (values_by_condition["RAW_R1"][metric] + values_by_condition["RAW_R2"][metric]) / 2.0
                fixed_mean = (values_by_condition["FIXED_R1"][metric] + values_by_condition["FIXED_R2"][metric]) / 2.0
                row_out[f"raw_mean_{metric}"] = raw_mean
                row_out[f"fixed_mean_{metric}"] = fixed_mean
                row_out[f"delta_{metric}"] = fixed_mean - raw_mean
            saved_primary_delta = _number(primary_row.get("primary_delta_C"), label=f"{speaker_id}.primary_delta_C")
            if not math.isclose(float(row_out["delta_C"]), saved_primary_delta, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"{speaker_id}: recomputed ΔC differs from frozen speaker_primary.csv")
            component_rows.append(row_out)

        dose_row: dict[str, Any] = {"speaker_id": speaker_id, "family_complete": declared_family_complete}
        if declared_family_complete:
            raw_row = condition_map.get((speaker_id, "RAW_R1"))
            if raw_row is None or str(raw_row.get("status", "")).strip().upper() != "PASS":
                raise ValueError(f"{speaker_id}: family_complete but RAW_R1 did not pass")
            raw_metrics: dict[str, float] = {}
            for metric in METRICS:
                value = _read_optional_metric(raw_row, metric, speaker_id=speaker_id)
                if value is None:
                    raise ValueError(f"{speaker_id}/RAW_R1.{metric} is blank on a PASS row")
                raw_metrics[metric] = value
            for dose_id in DOSE_ORDER:
                dose_condition = condition_map.get((speaker_id, dose_id))
                if dose_condition is None or str(dose_condition.get("status", "")).strip().upper() != "PASS":
                    raise ValueError(f"{speaker_id}: family_complete but {dose_id} did not pass")
                for metric in METRICS:
                    dose_value = _read_optional_metric(dose_condition, metric, speaker_id=speaker_id)
                    if dose_value is None:
                        raise ValueError(f"{speaker_id}/{dose_id}.{metric} is blank on a PASS row")
                    dose_row[f"{dose_id}_delta_{metric}"] = dose_value - raw_metrics[metric]
                saved_c = _number(primary_row.get(f"{dose_id}_dose_delta_C"), label=f"{speaker_id}.{dose_id}_dose_delta_C")
                if not math.isclose(float(dose_row[f"{dose_id}_delta_C"]), saved_c, rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(f"{speaker_id}/{dose_id}: component-derived ΔC differs from frozen speaker_primary.csv")
            family_complete_ids.append(speaker_id)
        else:
            for dose_id in DOSE_ORDER:
                for metric in METRICS:
                    dose_row[f"{dose_id}_delta_{metric}"] = None
        dose_component_rows.append(dose_row)

    n_complete = len(complete_ids)
    if int(primary_summary.get("n_complete", -1)) != n_complete:
        raise ValueError("condition-derived complete speaker count differs from the frozen primary summary")
    family_summary = summary.get("dose_family_vs_raw_r1")
    if not isinstance(family_summary, dict):
        raise ValueError("analysis_summary.json lacks dose_family_vs_raw_r1")
    n_family_complete = len(family_complete_ids)
    if int(summary.get("n_family_complete", -1)) != n_family_complete:
        raise ValueError("condition-derived complete dose-family count differs from analysis summary")
    if int(family_summary.get("n_planned", -1)) != n_planned or int(family_summary.get("n_complete", -1)) != n_family_complete:
        raise ValueError("dose-family completion counts disagree across analyzer outputs")
    metric_values = {
        metric: [float(row[f"delta_{metric}"]) for row in component_rows if row["primary_complete"]]
        for metric in METRICS
    }

    # The C interval is inherited exactly from the frozen primary. One common
    # PCG64 index matrix supplies descriptive percentile intervals for the
    # remaining score components; no max-t family or dose analysis is changed.
    shared_indices = _bootstrap_indices(n_complete) if n_complete else None
    metric_summary: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        vals = metric_values[metric]
        if metric == "C":
            if primary_summary.get("status") == "COMPUTABLE":
                mean = _number(primary_summary.get("mean"), label="primary mean")
                ci_raw = primary_summary.get("ci95")
                if not isinstance(ci_raw, list) or len(ci_raw) != 2:
                    raise ValueError("computable frozen primary summary must include ci95")
                ci = [_number(ci_raw[0], label="primary ci95 lower"), _number(ci_raw[1], label="primary ci95 upper")]
                if not math.isclose(mean, float(np.mean(vals)), rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError("condition-derived ΔC mean differs from frozen primary summary")
                metric_summary[metric] = {
                    "label": COMPONENT_LABELS[metric],
                    "n": n_complete,
                    "n_planned": n_planned,
                    "mean_delta": mean,
                    "ci95": ci,
                    "interval": "reused unchanged from frozen primary_fixed_minus_raw; speaker-bootstrap percentile 95%",
                    "ci_source": "analysis_summary.json:primary_fixed_minus_raw.ci95",
                }
            else:
                metric_summary[metric] = {
                    "label": COMPONENT_LABELS[metric],
                    "n": n_complete,
                    "n_planned": n_planned,
                    "mean_delta": None,
                    "ci95": None,
                    "interval": "withheld by frozen primary planned-denominator rule",
                    "ci_source": "frozen primary summary; not recomputed",
                }
            continue
        if vals and shared_indices is not None:
            mean = float(np.mean(np.asarray(vals, dtype=np.float64)))
            lo, hi = _percentile_ci(vals, shared_indices)
            ci: list[float] | None = [lo, hi]
        else:
            mean = None
            ci = None
        metric_summary[metric] = {
            "label": COMPONENT_LABELS[metric],
            "n": n_complete,
            "n_planned": n_planned,
            "mean_delta": mean,
            "ci95": ci,
            "interval": "paired speaker-bootstrap percentile 95%; descriptive point interval only; shared indices",
            "ci_source": "derived from repeat-averaged per-speaker condition metrics",
        }

    def max_abs(values: list[float]) -> float | None:
        return max((abs(v) for v in values), default=None)

    residual_c = [
        float(row["delta_C"]) - (float(row["delta_B"]) - float(row["delta_Dmin"]))
        for row in component_rows if row["primary_complete"]
    ]
    residual_anchor_c = [
        float(row["delta_anchor_C"]) - (float(row["delta_B"]) - float(row["delta_anchor_D"]))
        for row in component_rows if row["primary_complete"]
    ]
    residual_search = [
        float(row["delta_search_gain"]) - (float(row["delta_D0"]) - float(row["delta_Dmin"]))
        for row in component_rows if row["primary_complete"]
    ]
    algebra = {
        "delta_C_equals_delta_B_minus_delta_Dmin_max_abs_residual": max_abs(residual_c),
        "delta_anchor_C_equals_delta_B_minus_delta_anchor_D_max_abs_residual": max_abs(residual_anchor_c),
        "delta_search_gain_equals_delta_D0_minus_delta_Dmin_max_abs_residual": max_abs(residual_search),
        "interpretation": "algebraic score decomposition only; not a mediator share",
    }
    for key, value in algebra.items():
        if key.endswith("max_abs_residual") and value is not None and value > 5e-6:
            raise ValueError(f"component algebra check failed: {key}={value}")

    family_indices = _bootstrap_indices(n_family_complete) if n_family_complete else None
    dose_metric_summary: dict[str, dict[str, Any]] = {}
    family_contrasts = family_summary.get("contrasts", {}) if isinstance(family_summary.get("contrasts"), dict) else {}
    for dose_id in DOSE_ORDER:
        dose_metric_summary[dose_id] = {}
        for metric in METRICS:
            vals = [float(row[f"{dose_id}_delta_{metric}"]) for row in dose_component_rows if row["family_complete"]]
            if metric == "C":
                if family_summary.get("status") == "COMPUTABLE":
                    contrast = family_contrasts.get(dose_id)
                    if not isinstance(contrast, dict) or int(contrast.get("n", -1)) != n_family_complete:
                        raise ValueError(f"frozen C dose-family result is missing or miscounted for {dose_id}")
                    mean = _number(contrast.get("mean"), label=f"{dose_id} C family mean")
                    if not math.isclose(mean, float(np.mean(vals)), rel_tol=1e-10, abs_tol=1e-10):
                        raise ValueError(f"{dose_id}: component-derived C mean differs from frozen dose family")
                    interval_raw = contrast.get("simultaneous_ci95")
                    if not isinstance(interval_raw, list) or len(interval_raw) != 2:
                        raise ValueError(f"{dose_id}: frozen C dose family lacks simultaneous_ci95")
                    ci = [_number(interval_raw[0], label=f"{dose_id} CI low"), _number(interval_raw[1], label=f"{dose_id} CI high")]
                    ci_type = "simultaneous_max_t_family"
                    interval = "reused unchanged from frozen four-contrast C dose family; simultaneous max-|t| 95% CI"
                else:
                    mean, ci = None, None
                    ci_type = "withheld"
                    interval = "withheld by frozen four-contrast C family planned-denominator rule"
            else:
                if vals and family_indices is not None:
                    mean = float(np.mean(np.asarray(vals, dtype=np.float64)))
                    low, high = _percentile_ci(vals, family_indices)
                    ci = [low, high]
                else:
                    mean, ci = None, None
                ci_type = "pointwise_descriptive"
                interval = "paired speaker-bootstrap percentile 95%; descriptive point interval only; shared indices across four doses and components"
            dose_metric_summary[dose_id][metric] = {
                "label": COMPONENT_LABELS[metric],
                "dose_db": DOSE_DB[dose_id],
                "n": n_family_complete,
                "n_planned": n_planned,
                "mean_delta": mean,
                "ci95": ci,
                "ci_type": ci_type,
                "interval": interval,
            }

    return {
        "schema": "p1_component_decomposition_attempt01",
        "planned_speakers": planned_ids,
        "n_planned": n_planned,
        "n_primary_component_complete": n_complete,
        "primary_summary_status": primary_summary.get("status"),
        "bootstrap": {"generator": "numpy.random.PCG64", "seed": SEED, "draws": BOOTSTRAP_DRAWS, "shared_indices_for_non_C_components": True},
        "component_summary": metric_summary,
        "speaker_components": component_rows,
        "algebra_checks": algebra,
        "dose_component_summary": dose_metric_summary,
        "dose_speaker_components": dose_component_rows,
        "dose_bootstrap": {"generator": "numpy.random.PCG64", "seed": SEED, "draws": BOOTSTRAP_DRAWS, "same_speaker_indices_across_all_four_doses_and_non_C_components": True},
        "protocol_sha256": str(summary.get("protocol_sha256", "")),
        "analyzer_input_sha256": str(summary.get("input_sha256", "")),
        "statistical_scope": "C interval is reused from frozen primary; other intervals are descriptive pointwise speaker-bootstrap intervals; no individual significance tests or dose selection",
    }


def read_curve_summary(path: Path, *, expected_groups: list[str], n_planned: int) -> dict[str, list[dict[str, Any]]]:
    rows = _read_csv(path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = str(row.get("curve_id", ""))
        if group not in expected_groups:
            raise ValueError(f"unexpected curve group in distance_curve_summary.csv: {group}")
        record = {
            "curve_id": group,
            "lag": _number(row.get("lag"), label=f"{group}.lag"),
            "n_speakers": int(row.get("n_speakers", -1)),
            "n_planned": int(row.get("n_planned", -1)),
            "mean_distance": _number(row.get("mean_distance"), label=f"{group}.mean_distance"),
            "ci95_low_pointwise": _number(row.get("ci95_low_pointwise"), label=f"{group}.ci95_low_pointwise"),
            "ci95_high_pointwise": _number(row.get("ci95_high_pointwise"), label=f"{group}.ci95_high_pointwise"),
            "interval": str(row.get("interval", "")),
        }
        if record["n_planned"] != n_planned or record["n_speakers"] < 1 or record["n_speakers"] > n_planned:
            raise ValueError(f"{group}: invalid curve summary denominator")
        if record["ci95_low_pointwise"] > record["ci95_high_pointwise"]:
            raise ValueError(f"{group}: curve pointwise interval is unordered")
        if "pointwise" not in record["interval"].lower() or "descriptive" not in record["interval"].lower():
            raise ValueError(f"{group}: expected descriptive pointwise interval metadata")
        grouped.setdefault(group, []).append(record)

    missing = [group for group in expected_groups if group not in grouped]
    if missing:
        raise ValueError(f"distance_curve_summary.csv is missing planned curve groups: {missing}")
    lag_grid: list[float] | None = None
    for group in expected_groups:
        records = sorted(grouped[group], key=lambda item: item["lag"])
        lags = [float(item["lag"]) for item in records]
        if len(lags) != 31 or len(set(lags)) != 31:
            raise ValueError(f"{group}: expected exactly 31 unique lag rows")
        if lag_grid is None:
            lag_grid = lags
        elif lags != lag_grid:
            raise ValueError(f"{group}: lag grid differs from the other analyzer curve groups")
        grouped[group] = records
    return grouped


def _svg_text(
    x: float,
    y: float,
    text: str,
    size: int = 13,
    anchor: str = "start",
    color: str = "#222",
    weight: str = "normal",
    transform: str | None = None,
) -> str:
    transform_attr = f' transform="{html.escape(transform, quote=True)}"' if transform else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{transform_attr} '
        f'font-family="Arial,sans-serif" font-size="{size}" fill="{color}" font-weight="{weight}">'
        f'{html.escape(text)}</text>'
    )


def _svg_line(x1: float, y1: float, x2: float, y2: float, color: str = "#d9dee3", width: float = 1.0, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr}/>'


def _range(values: list[float]) -> tuple[float, float]:
    if not values:
        return -1.0, 1.0
    low, high = min(values), max(values)
    if high - low < 1e-12:
        padding = max(abs(low) * 0.1, 0.1)
    else:
        padding = (high - low) * 0.14
    return low - padding, high + padding


def render_svg(decomposition: dict[str, Any], curves: dict[str, list[dict[str, Any]]]) -> str:
    width, height = 1420, 1165
    comp_left, comp_right = 330.0, 1000.0
    comp_width = comp_right - comp_left
    comp_stats = decomposition["component_summary"]
    comp_rows = [comp_stats[key] for key in METRICS]
    comp_values = [0.0]
    for row in comp_rows:
        if row["mean_delta"] is not None:
            comp_values.append(float(row["mean_delta"]))
        if row["ci95"] is not None:
            comp_values.extend(float(v) for v in row["ci95"])
    xmin, xmax = _range(comp_values)
    def xcomp(value: float) -> float:
        return comp_left + (value - xmin) / (xmax - xmin) * comp_width

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 34, "Amplitude P1: score-component decomposition and complete lag curves", 23, "middle", weight="bold"),
        _svg_text(width / 2, 59, f"Primary paired unit: speaker · complete components n={decomposition['n_primary_component_complete']}/{decomposition['n_planned']}", 13, "middle", "#555"),
        _svg_text(38, 99, "A · Repeat-averaged FIXED − RAW component contrasts", 17, weight="bold"),
        _svg_text(38, 121, "Non-C bars are descriptive paired speaker-bootstrap 95% point intervals; C reuses the frozen primary CI.", 12, color="#555"),
        _svg_text(1025, 121, "mean Δ [95% CI] · n/N", 10, color="#555"),
    ]

    plot_top, row_step = 165.0, 34.0
    plot_bottom = plot_top + (len(METRICS) - 1) * row_step
    zero_x = xcomp(0.0)
    parts.append(_svg_line(zero_x, plot_top - 15, zero_x, plot_bottom + 14, "#4a535c", 1.4, "5,4"))
    for i in range(6):
        tick = xmin + (xmax - xmin) * i / 5
        xx = xcomp(tick)
        parts.append(_svg_line(xx, plot_top - 12, xx, plot_bottom + 14, "#edf0f2", 0.8))
        parts.append(_svg_text(xx, plot_top - 21, f"{tick:+.3f}", 10, "middle", "#666"))
    for i, metric in enumerate(METRICS):
        y = plot_top + i * row_step
        row = comp_stats[metric]
        parts.append(_svg_text(38, y + 4, COMPONENT_LABELS[metric], 12, color="#39434c", weight="bold" if metric == "C" else "normal"))
        if row["mean_delta"] is None or row["ci95"] is None:
            parts.append(_svg_text(1025, y + 4, f"withheld · n={row['n']}/{row['n_planned']}", 10, color="#737d85"))
            continue
        mean = float(row["mean_delta"])
        low, high = map(float, row["ci95"])
        color = "#246b8e" if metric == "C" else "#687e8d"
        parts.append(_svg_line(xcomp(low), y, xcomp(high), y, color, 2.2))
        parts.append(_svg_line(xcomp(low), y - 5, xcomp(low), y + 5, color, 1.4))
        parts.append(_svg_line(xcomp(high), y - 5, xcomp(high), y + 5, color, 1.4))
        parts.append(f'<circle cx="{xcomp(mean):.1f}" cy="{y:.1f}" r="5.2" fill="{color}" stroke="white" stroke-width="1.2"/>')
        parts.append(_svg_text(1025, y + 4, f"{mean:+.3f} [{low:+.3f}, {high:+.3f}] · {row['n']}/{row['n_planned']}", 10, color=color))
    algebra = decomposition["algebra_checks"]
    parts.append(_svg_text(38, 425, "Algebra: ΔC = ΔB − ΔDmin; Δanchor_C = ΔB − Δanchor_D; Δsearch_gain = ΔD0 − ΔDmin.", 12, color="#303942"))
    parts.append(_svg_text(38, 446, f"Max |residual|: C identity {algebra['delta_C_equals_delta_B_minus_delta_Dmin_max_abs_residual']}; anchor identity {algebra['delta_anchor_C_equals_delta_B_minus_delta_anchor_D_max_abs_residual']}; search identity {algebra['delta_search_gain_equals_delta_D0_minus_delta_Dmin_max_abs_residual']}. Algebraic decomposition only, not mediator shares.", 10, color="#5c646b"))

    curve_left, curve_top, curve_width, curve_height = 110.0, 545.0, 1075.0, 370.0
    lag_grid = [float(row["lag"]) for row in curves[CURVE_ORDER[0]]]
    curve_values = [
        float(v)
        for rows in curves.values()
        for row in rows
        for v in (row["mean_distance"], row["ci95_low_pointwise"], row["ci95_high_pointwise"])
    ]
    y_min, y_max = _range(curve_values)
    x_min, x_max = min(lag_grid), max(lag_grid)
    if x_max <= x_min:
        raise ValueError("curve lag range must be increasing")
    def xcurve(value: float) -> float:
        return curve_left + (value - x_min) / (x_max - x_min) * curve_width
    def ycurve(value: float) -> float:
        return curve_top + curve_height - (value - y_min) / (y_max - y_min) * curve_height
    parts.append(_svg_text(38, 504, "B · Complete 31-lag distance curves", 17, weight="bold"))
    parts.append(_svg_text(38, 526, "Mean curves and pointwise 95% ribbons are reused from distance_curve_summary.csv (speaker-level aggregation).", 12, color="#555"))
    for i in range(6):
        tick = y_min + (y_max - y_min) * i / 5
        yy = ycurve(tick)
        parts.append(_svg_line(curve_left, yy, curve_left + curve_width, yy, "#e8ecef", 0.8))
        parts.append(_svg_text(curve_left - 9, yy + 4, f"{tick:.3f}", 10, "end", "#666"))
    for tick in [x_min + (x_max - x_min) * i / 6 for i in range(7)]:
        xx = xcurve(tick)
        parts.append(_svg_line(xx, curve_top + curve_height, xx, curve_top + curve_height + 5, "#555", 0.8))
        parts.append(_svg_text(xx, curve_top + curve_height + 20, f"{tick:g}", 10, "middle", "#555"))
    parts.append(_svg_text(curve_left + curve_width / 2, curve_top + curve_height + 43, "Lag index · locked 31-column scorer grid", 11, "middle"))
    axis_label_x = curve_left - 50
    axis_label_y = curve_top + curve_height / 2
    parts.append(_svg_text(
        axis_label_x,
        axis_label_y,
        "Mean distance",
        11,
        "middle",
        transform=f"rotate(-90 {axis_label_x:.1f} {axis_label_y:.1f})",
    ))

    for group in CURVE_ORDER:
        rows = curves[group]
        color = CURVE_COLORS[group]
        upper = [(xcurve(float(row["lag"])), ycurve(float(row["ci95_high_pointwise"]))) for row in rows]
        lower = [(xcurve(float(row["lag"])), ycurve(float(row["ci95_low_pointwise"]))) for row in reversed(rows)]
        polygon = " ".join(f"{x:.1f},{y:.1f}" for x, y in [*upper, *lower])
        parts.append(f'<polygon points="{polygon}" fill="{color}" fill-opacity="0.08" stroke="none"/>')
        points = " ".join(f"{xcurve(float(row['lag'])):.1f},{ycurve(float(row['mean_distance'])):.1f}" for row in rows)
        dash = ' stroke-dasharray="5,4"' if group == "RAW_R1" else ""
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2"{dash}/>')

    legend_positions = ((150, 990), (450, 990), (790, 990), (1070, 990), (260, 1022), (590, 1022), (900, 1022))
    for group, (lx, ly) in zip(CURVE_ORDER, legend_positions):
        color = CURVE_COLORS[group]
        parts.append(_svg_line(lx, ly - 4, lx + 27, ly - 4, color, 2.4, "5,4" if group == "RAW_R1" else ""))
        n = int(curves[group][0]["n_speakers"])
        parts.append(_svg_text(lx + 34, ly, f"{CURVE_LABELS[group]} · n={n}/{decomposition['n_planned']}", 10, color="#424a51"))
    parts.append(_svg_text(38, 1070, "Do not derive C from the cohort-mean curve: the component table uses each speaker’s saved condition metrics and repeat averages.", 11, color="#303942"))
    parts.append(_svg_text(38, 1091, "Curve ribbons and non-C component intervals are descriptive pointwise 95% intervals, not simultaneous tests. A rise in B alone is not evidence of gaming or lower audio quality.", 11, color="#555"))
    parts.append(_svg_text(38, 1122, f"Frozen protocol SHA-256: {decomposition['protocol_sha256']} · analyzer input SHA-256: {decomposition['analyzer_input_sha256'] or 'not recorded'}", 9, color="#7a8289"))
    parts.append("</svg>")
    return "\n".join(parts)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty derived component table")
    keys = list(rows[0])
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(analysis_dir: Path, output_dir: Path) -> dict[str, Any]:
    summary_path = analysis_dir / "analysis_summary.json"
    conditions_path = analysis_dir / "speaker_conditions.csv"
    primary_path = analysis_dir / "speaker_primary.csv"
    curves_path = analysis_dir / "distance_curve_summary.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("analysis_summary.json root must be an object")
    decomposition = compute_decomposition(summary, _read_csv(conditions_path), _read_csv(primary_path))
    expected_groups = [str(x) for x in summary.get("curve_groups", [])]
    if tuple(expected_groups) != CURVE_ORDER:
        raise ValueError(f"frozen curve group order differs: {expected_groups}")
    curves = read_curve_summary(curves_path, expected_groups=expected_groups, n_planned=decomposition["n_planned"])
    svg = render_svg(decomposition, curves)

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"preserving existing derived output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=False)
    table_rows = decomposition["speaker_components"]
    _write_csv(output_dir / "component_decomposition_by_speaker.csv", table_rows)
    _write_csv(output_dir / "dose_component_decomposition_by_speaker.csv", decomposition["dose_speaker_components"])
    dose_summary_rows = []
    for dose_id in DOSE_ORDER:
        for metric in METRICS:
            row = decomposition["dose_component_summary"][dose_id][metric]
            ci = row["ci95"]
            dose_summary_rows.append({
                "dose_id": dose_id,
                "dose_db": row["dose_db"],
                "metric": metric,
                "label": row["label"],
                "mean_delta": row["mean_delta"],
                "ci95_low": ci[0] if ci is not None else None,
                "ci95_high": ci[1] if ci is not None else None,
                "ci_type": row["ci_type"],
                "n": row["n"],
                "n_planned": row["n_planned"],
                "interval": row["interval"],
            })
    _write_csv(output_dir / "dose_component_decomposition_summary.csv", dose_summary_rows)
    summary_path_out = output_dir / "component_decomposition_summary.json"
    summary_path_out.write_text(json.dumps(decomposition, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    figure_path = output_dir / "p1_full_curves_and_component_decomposition.svg"
    figure_path.write_text(svg, encoding="utf-8")
    manifest = {
        "schema": "p1_component_decomposition_derived_manifest_attempt01",
        "input_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (summary_path, conditions_path, primary_path, curves_path)
        },
        "helper_source": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        "derived_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (
                output_dir / "component_decomposition_by_speaker.csv",
                output_dir / "dose_component_decomposition_by_speaker.csv",
                output_dir / "dose_component_decomposition_summary.csv",
                summary_path_out,
                figure_path,
            )
        },
        "no_score_results_read": True,
        "no_models_run": True,
        "dose_family_unchanged": True,
    }
    (output_dir / "derived_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "n_planned": decomposition["n_planned"],
        "n_primary_component_complete": decomposition["n_primary_component_complete"],
        "n_family_component_complete": len([row for row in decomposition["dose_speaker_components"] if row["family_complete"]]),
        "dose_component_rows": sum(len(metrics) for metrics in decomposition["dose_component_summary"].values()),
        "curve_groups": list(curves),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True, help="existing frozen analyzer output directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="new create-only derived output directory")
    args = parser.parse_args(argv)
    result = run(args.analysis_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
