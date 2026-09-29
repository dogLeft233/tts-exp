#!/usr/bin/env python3
"""Independent, dependency-light audit of saved acoustic-reorganization outputs.

This checker intentionally does not import the experiment implementation.  It
recomputes the saved curve scalars, the ten paired effects, and the shared
bootstrap intervals from the output files alone.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

EVALUATION_IDS = tuple(range(6, 16))
ARMS = ("N_RAW", "N_ID", "N_R_DOWN", "N_P_DOWN", "T_ID", "T_R_DOWN", "T_P_DOWN")
RESYNTH_ARMS = ARMS[1:]
N_ARMS = ("N_ID", "N_R_DOWN", "N_P_DOWN")
T_ARMS = ("T_ID", "T_R_DOWN", "T_P_DOWN")
VSHIFT = 15
MIN_COMMON_WINDOWS = 30
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_DRAWS = 20_000
BONFERRONI_ALPHA = 0.05 / 6.0


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_cells() -> set[tuple[int, str, str]]:
    cells: set[tuple[int, str, str]] = set()
    for sid in EVALUATION_IDS:
        cells.update((sid, arm, arm) for arm in ARMS)
        cells.update((sid, arm, "N_RAW") for arm in RESYNTH_ARMS)
        for ids in (N_ARMS, T_ARMS):
            identity = ids[0]
            cells.update((sid, arm, identity) for arm in ids[1:])
            cells.update((sid, identity, arm) for arm in ids[1:])
    return cells


def _curve_metrics(curve: dict[str, Any]) -> dict[str, float | int]:
    expected_lags = list(range(-VSHIFT, VSHIFT + 1))
    if list(curve.get("lags", [])) != expected_lags:
        raise ValueError("curve lag grid differs from -15..15")
    values = np.asarray(curve.get("values", []), dtype=np.float64)
    if values.shape != (31,) or not np.isfinite(values).all():
        raise ValueError("curve must contain 31 finite values")
    minimum = float(np.min(values))
    index = int(np.flatnonzero(values == minimum)[0])
    return {"C": float(np.median(values) - minimum), "D": minimum, "B": float(np.median(values)), "k_star": expected_lags[index], "d_zero": float(values[VSHIFT])}


def _read_scores(output: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    with (output / "scores.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (int(row["sample_id"]), row["video_arm"], row["audio_arm"])
        parsed = dict(row)
        for field in ("C", "D", "B", "d_zero"):
            parsed[field] = float(row[f"{field}_raw"] if row.get(f"{field}_raw") not in (None, "") else row[field])
        parsed["k_star"] = int(row["k_star"])
        result[key] = parsed
    return result


def _bootstrap_summary(values: list[float], indices: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    estimates = np.mean(array[indices], axis=1)
    return {"mean": float(np.mean(array)), "ci95": np.quantile(estimates, [0.025, 0.975], method="linear").tolist(), "ci_bonferroni": np.quantile(estimates, [BONFERRONI_ALPHA / 2.0, 1.0 - BONFERRONI_ALPHA / 2.0], method="linear").tolist()}


def audit(output: Path) -> dict[str, Any]:
    failures: list[str] = []
    curves = _read_json(output / "curves.json").get("curves", [])
    curve_map = {(int(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row for row in curves}
    scores = _read_scores(output)
    expected = _expected_cells()
    if set(scores) != expected or set(curve_map) != expected or len(scores) != 210:
        failures.append(f"cell_set scores={len(scores)} curves={len(curve_map)} expected={len(expected)}")
    for key in expected:
        if key not in scores or key not in curve_map:
            continue
        try:
            metrics = _curve_metrics(curve_map[key]); score = scores[key]
            for field in ("C", "D", "B", "d_zero"):
                if abs(float(score[field]) - float(metrics[field])) > 1e-6:
                    failures.append(f"curve_scalar:{key}:{field}")
            if int(score["k_star"]) != int(metrics["k_star"]):
                failures.append(f"curve_scalar:{key}:k_star")
        except Exception as exc:  # noqa: BLE001 - preserve audit detail
            failures.append(f"curve:{key}:{exc}")
    id6 = [key for key in expected if key[0] == 6]
    if len(id6) != 21:
        failures.append(f"id6_cells:{len(id6)}")

    feature_manifest = _read_json(output / "feature_manifest.json")
    diagnostics: dict[int, dict[str, Any]] = {}
    for record in feature_manifest.get("records", []):
        sid = int(record["sample_id"])
        metadata_path = Path(str(record["feature_path"])).with_suffix(".json")
        diagnostics[sid] = _read_json(metadata_path)["source_diagnostic"]
    with (output / "paired_effects.csv").open(newline="", encoding="utf-8") as handle:
        paired_rows = {int(row["sample_id"]): row for row in csv.DictReader(handle)}
    paired_values: dict[str, list[float]] = {name: [] for name in ("S", "A", "b_N", "h_N", "h_T", "J")}
    for sid in EVALUATION_IDS:
        diag = diagnostics[sid]
        q = {arm: float(scores[(sid, arm, "N_RAW")]["C"]) for arm in ARMS}
        expected_row = {"S": float(diag["f_R_N"]) - float(diag["f_R_T_RAW"]), "A": float(diag["semantic_A"]), "b_N": q["N_R_DOWN"] - q["N_ID"], "h_N": q["N_ID"] - q["N_P_DOWN"], "h_T": q["T_ID"] - q["T_P_DOWN"], "J": (q["N_R_DOWN"] - q["N_ID"]) - (q["T_R_DOWN"] - q["T_ID"])}
        row = paired_rows.get(sid)
        if row is None:
            failures.append(f"paired_missing:{sid}")
            continue
        for field, value in expected_row.items():
            if abs(float(row[field]) - value) > 1e-10:
                failures.append(f"paired:{sid}:{field}")
            paired_values[field].append(value)

    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 10, size=(BOOTSTRAP_DRAWS, 10), endpoint=False)
    analysis = _read_json(output / "analysis.json")
    for name, values in paired_values.items():
        expected_summary = _bootstrap_summary(values, indices)
        actual = analysis.get("primary", {}).get(name, {})
        if actual.get("mean") is None or abs(float(actual["mean"]) - expected_summary["mean"]) > 1e-12 or not np.allclose(actual.get("ci95"), expected_summary["ci95"], atol=1e-12, rtol=1e-12) or not np.allclose(actual.get("ci_bonferroni"), expected_summary["ci_bonferroni"], atol=1e-12, rtol=1e-12):
            failures.append(f"bootstrap:{name}")
    return {"status": "complete" if not failures else "incomplete", "protocol": "tts_acoustic_reorganization_v1", "score_cells_checked": len(scores), "id6_cells_checked": len(id6), "primary_statistics_checked": len(paired_values), "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.output_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
