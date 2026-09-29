"""Pre-registered group-level summaries and interpretation labels."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.experiments.phone_separability_enhancement.metrics import paired_group_bootstrap


BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260922


def _bootstrap_group_effects(effects: Mapping[str, float], *, confidence: float = 0.95) -> dict[str, Any]:
    groups = sorted(str(group) for group in effects)
    values = np.asarray([float(effects[group]) for group in groups], dtype=np.float64)
    if not groups or not np.isfinite(values).all():
        return {"estimate": None, "ci_low": None, "ci_high": None, "n_groups": 0, "groups": groups, "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED}
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    tail = (1.0 - float(confidence)) / 2.0
    return {"estimate": float(values.mean()), "ci_low": float(np.quantile(means, tail)), "ci_high": float(np.quantile(means, 1.0 - tail)), "n_groups": len(groups), "groups": groups, "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "confidence": confidence, "method": "source_group_paired_percentile_bootstrap"}


def _group_effects(group_rows: Mapping[str, Sequence[Mapping[str, Any]]], function: Any) -> dict[str, float]:
    effects: dict[str, float] = {}
    for group, rows in group_rows.items():
        values = []
        for row in rows:
            try:
                value = float(function(row))
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        if values:
            effects[str(group)] = float(np.mean(values))
    return effects


def _cell(row: Mapping[str, Any], video_arm: str, audio_arm: str) -> Mapping[str, Any]:
    return row["cells"][f"{video_arm}_{audio_arm}"]


def _tfg_stat(group_rows: Mapping[str, Sequence[Mapping[str, Any]]], function: Any) -> dict[str, Any]:
    effects = _group_effects(group_rows, function)
    result = _bootstrap_group_effects(effects, confidence=0.95)
    strict = _bootstrap_group_effects(effects, confidence=1.0 - 2.0 * (1.0 - 0.9833333333333333) / 2.0)
    result["ci98_333_low"] = strict["ci_low"]
    result["ci98_333_high"] = strict["ci_high"]
    result["group_effects"] = effects
    return result


def analyze_tfg_scores(score_rows: Sequence[Mapping[str, Any]], *, expected_pairs: int | None = None, required_cells: Sequence[tuple[str, str]] | None = None) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        if row.get("status") != "COMPLETE":
            continue
        grouped[str(row["source_group"])].append(dict(row))
    group_rows: dict[str, list[dict[str, Any]]] = {}
    incomplete_pairs: list[str] = []
    required = tuple(required_cells or ("N", "N"))
    for group, rows in grouped.items():
        by_pair: dict[str, dict[str, Any]] = {}
        for row in rows:
            by_pair.setdefault(str(row["pair_id"]), {"source_group": group, "cells": {}})["cells"][f"{row['video_arm']}_{row['audio_arm']}"] = row
        complete = []
        for pair_id, value in by_pair.items():
            if all(f"{video}_{audio}" in value["cells"] for video, audio in required):
                complete.append(value)
            else:
                incomplete_pairs.append(pair_id)
        if complete:
            group_rows[group] = complete

    arms: dict[str, Any] = {}
    for arm in ("D", "A", "B", "C"):
        c_gain = _tfg_stat(group_rows, lambda row, arm=arm: float(_cell(row, arm, "N")["C"]) - float(_cell(row, "N", "N")["C"]))
        d_improvement = _tfg_stat(group_rows, lambda row, arm=arm: float(_cell(row, "N", "N")["D"]) - float(_cell(row, arm, "N")["D"]))
        d_anchor_improvement = _tfg_stat(group_rows, lambda row, arm=arm: float(_cell(row, "N", "N")["D_anchor"]) - float(_cell(row, arm, "N")["D_anchor"]))
        audio_effect = _tfg_stat(group_rows, lambda row, arm=arm: float(_cell(row, "N", arm)["C"]) - float(_cell(row, "N", "N")["C"]))
        audio_d_anchor_improvement = _tfg_stat(group_rows, lambda row, arm=arm: float(_cell(row, "N", "N")["D_anchor"]) - float(_cell(row, "N", arm)["D_anchor"]))
        interaction = _tfg_stat(group_rows, lambda row, arm=arm: float(_cell(row, arm, arm)["C"]) - float(_cell(row, arm, "N")["C"]) - float(_cell(row, "N", arm)["C"]) + float(_cell(row, "N", "N")["C"]))
        lag_stable = _group_effects(group_rows, lambda row, arm=arm: 1.0 if abs(int(_cell(row, arm, "N")["lag"]) - int(_cell(row, "N", "N")["lag"])) <= 1 else 0.0)
        lag_rate = _bootstrap_group_effects(lag_stable, confidence=0.95)
        calibration_missing = any(stat.get("estimate") is None for stat in (c_gain, d_improvement, d_anchor_improvement))
        classification = "INCONCLUSIVE_MISSING" if not group_rows or calibration_missing else ("GAIN" if c_gain["estimate"] > 0.05 and c_gain.get("ci98_333_low") is not None and c_gain["ci98_333_low"] > 0.0 and d_improvement.get("estimate", -np.inf) >= 0.0 and d_improvement.get("ci_low", -np.inf) >= -0.05 and d_anchor_improvement.get("estimate", -np.inf) >= 0.0 and d_anchor_improvement.get("ci_low", -np.inf) >= -0.05 and lag_rate.get("estimate", 0.0) >= 0.90 else "NO_CLEAR_GAIN")
        arms[arm] = {"C_gain_video_N": c_gain, "D_improvement_video_N": d_improvement, "D_anchor_improvement_video_N": d_anchor_improvement, "C_audio_N_video": audio_effect, "D_anchor_audio_N_video": audio_d_anchor_improvement, "C_interaction": interaction, "lag_within_one_frame_rate": lag_rate, "classification": classification}
    c_vs_a = _tfg_stat(group_rows, lambda row: float(_cell(row, "C", "N")["C"]) - float(_cell(row, "A", "N")["C"]))
    d_vs_a = _tfg_stat(group_rows, lambda row: float(_cell(row, "A", "N")["D"]) - float(_cell(row, "C", "N")["D"]))
    d_anchor_vs_a = _tfg_stat(group_rows, lambda row: float(_cell(row, "A", "N")["D_anchor"]) - float(_cell(row, "C", "N")["D_anchor"]))
    tts_native = {"C_TT_vs_NN": _tfg_stat(group_rows, lambda row: float(_cell(row, "T", "T")["C"]) - float(_cell(row, "N", "N")["C"])), "D_TT_vs_NN": _tfg_stat(group_rows, lambda row: float(_cell(row, "T", "T")["D"]) - float(_cell(row, "N", "N")["D"])), "note": "descriptive native-axis T/T versus N/N; not a natural-clock replacement contrast"}
    primary = arms["C"]
    complete_pair_count = sum(len(rows) for rows in group_rows.values())
    complete = bool(group_rows) and (expected_pairs is None or complete_pair_count == int(expected_pairs))
    status = "COMPLETE" if complete else ("INCONCLUSIVE_MISSING" if incomplete_pairs or expected_pairs is not None else "PARTIAL")
    return {"status": status, "complete_score_rows": len([row for row in score_rows if row.get("status") == "COMPLETE"]), "complete_pair_count": complete_pair_count, "expected_pair_count": expected_pairs, "incomplete_pairs": sorted(set(incomplete_pairs)), "source_group_count": len(group_rows), "arms": arms, "C_vs_A_video_N": c_vs_a, "D_vs_A_video_N": d_vs_a, "D_anchor_vs_A_video_N": d_anchor_vs_a, "tts_native": tts_native, "classification": primary["classification"] if complete else status, "contract": "group-level paired bootstrap; natural-clock endpoint is V(E),A(N) minus V(N),A(N); D_anchor uses each baseline N/N k; T/T is descriptive only"}


def classify_phone(phone: Mapping[str, Any]) -> dict[str, Any]:
    summary = phone.get("pairwise", {}).get("xlsr_natural_primary", {})
    c_n = summary.get("C-N", {})
    c_a = summary.get("C-A", {})
    if not c_n or not c_a or c_n.get("estimate") is None or c_a.get("estimate") is None:
        classification = "INCONCLUSIVE_MISSING"
    else:
        gain = bool(float(c_n["estimate"]) >= 0.01 and c_n.get("ci_low") is not None and float(c_n["ci_low"]) > 0.0 and float(c_a["estimate"]) >= 0.01 and c_a.get("ci_low") is not None and float(c_a["ci_low"]) > 0.0)
        classification = "GAIN" if gain else "NO_CLEAR_GAIN"
    return {"classification": classification, "encoder": "xlsr", "C-N": c_n, "C-A": c_a, "threshold": 0.01, "note": "natural_primary excludes T; matched_nt is the TTS timing comparison"}


def group_mean(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if value is not None and np.isfinite(float(value)):
            grouped[str(row["source_group"])].append(float(value))
    return {group: float(np.mean(values)) for group, values in grouped.items() if values}


def paired_delta(rows: Sequence[Mapping[str, Any]], candidate: str, baseline: str, metric: str = "C") -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        values = row.get("arms", row.get("scores", {}))
        if candidate not in values or baseline not in values:
            continue
        left, right = values[candidate].get(metric), values[baseline].get(metric)
        if left is not None and right is not None and np.isfinite([left, right]).all():
            grouped[str(row["source_group"])].append(float(left) - float(right))
    effects = {group: float(np.mean(values)) for group, values in grouped.items() if values}
    return paired_group_bootstrap(effects, draws=20000, seed=20260922)


def classify_delta(summary: Mapping[str, Any], *, threshold: float = 0.05, lower_bound_required: bool = True) -> str:
    estimate = summary.get("estimate")
    lower = summary.get("ci_low")
    if estimate is None or lower is None:
        return "INCONCLUSIVE_MISSING"
    if float(estimate) > threshold and (not lower_bound_required or float(lower) > 0):
        return "GAIN"
    return "NO_CLEAR_GAIN"


def interpret(phone: Mapping[str, Any], tfg: Mapping[str, Any]) -> str:
    phone_gain = phone.get("classification") in {"GAIN", "NONINFERIOR"}
    tfg_gain = tfg.get("classification") == "GAIN"
    if phone.get("classification", "").startswith("INCONCLUSIVE") or tfg.get("status", "").startswith("INCONCLUSIVE"):
        return "INCONCLUSIVE_MISSING_OR_UNCALIBRATED"
    if phone_gain and tfg_gain:
        return "PHONE_UP_TFG_UP_ASSOCIATED_NOT_MEDIATED"
    if phone_gain:
        return "PHONE_UP_TFG_NO_CLEAR_UP"
    if tfg_gain:
        return "PHONE_NO_CLEAR_UP_TFG_UP"
    return "NEITHER_CLEARLY_UP"


__all__ = ["analyze_tfg_scores", "classify_delta", "classify_phone", "group_mean", "interpret", "paired_delta"]
