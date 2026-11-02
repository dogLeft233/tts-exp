#!/usr/bin/env python3
"""Plot and export frozen functional lag-response summaries only."""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

# Keep Matplotlib's font/cache writes outside the report asset directory.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "codex-report40-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
EFFECT_PATH = ROOT / ".codex/scratch/functional_lag_response_20261015_attempt01/lag_effect_attempt01.json"
QA_PATH = ROOT / ".codex/scratch/functional_lag_response_20261015_attempt01/root_actual_lag_qa.json"
EXPECTED_EFFECT_SHA = "3055e950cd51820288e04f9f4607256333cf128dca95af18b802543b819ddc22"
EXPECTED_QA_SHA = "4dd93b5e56a24c7b4998904aeeebf5d1e59534a11d38dbfc54702286610e894c"
ENDPOINT_CSV = OUT / "endpoint_summary.csv"
CURVE_CSV = OUT / "lag_response_curves.csv"
SPEAKER_CSV = OUT / "lag_response_speaker_rms.csv"
PNG = OUT / "functional_lag_response.png"
SVG = OUT / "functional_lag_response.svg"
RECEIPT = OUT / "asset_receipt.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fnum(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return format(float(value), ".17g")
    return str(value)


def require_sha(path: Path, expected: str) -> str:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"Frozen input SHA mismatch for {path}: {observed}")
    return observed


effect_sha = require_sha(EFFECT_PATH, EXPECTED_EFFECT_SHA)
qa_sha = require_sha(QA_PATH, EXPECTED_QA_SHA)
effect = json.loads(EFFECT_PATH.read_text(encoding="utf-8"))
qa = json.loads(QA_PATH.read_text(encoding="utf-8"))
if qa.get("effect_sha256") != effect_sha:
    raise ValueError("ROOT QA does not bind the supplied effect JSON")
if len(qa.get("six_cell_summary", [])) != 6 or len(effect.get("model_attribute_results", {})) != 6:
    raise ValueError("Frozen six-cell inputs do not match expected shape")

qa_rows = {(r["model"], r["attribute"]): r for r in qa["six_cell_summary"]}
cells = effect["model_attribute_results"]
model_order = ["Wav2Lip", "MuseTalk", "Ditto"]
attribute_order = ["vowel_height_aperture_proxy", "vowel_rounding_proxy"]
attribute_labels = {
    "vowel_height_aperture_proxy": "Vowel-height / aperture proxy",
    "vowel_rounding_proxy": "Vowel-rounding proxy",
}

# Fail closed on source shape only; no score, support, or curve is recalculated.
for model in model_order:
    for attribute in attribute_order:
        key = f"{model}|{attribute}"
        cell = cells[key]
        if cell.get("support_status") != "SUPPORTED":
            raise ValueError(f"Unexpected frozen support status: {key}")
        z_rows = [r for r in cell["curve_mean_median"] if r.get("scale") == "z"]
        lags = sorted(int(r["lag"]) for r in z_rows)
        if lags != list(range(-8, 9)):
            raise ValueError(f"Unexpected saved z-lag grid: {key}")
        if len([r for r in cell["speaker_curves"][0]["lag_curves"] if r.get("scale") == "z"]) != 17:
            raise ValueError(f"Unexpected saved speaker z-lag grid: {key}")
        if (model, attribute) not in qa_rows:
            raise ValueError(f"ROOT QA cell missing: {key}")

# Endpoint summaries: direct saved raw/z means, medians, deltas and gate fields.
endpoint_fields = [
    "model", "attribute", "support_status", "n_speakers_with_both_classes", "endpoint", "scale", "n_speakers",
    "rms_mean", "rms_median", "head_mean", "head_median", "head_minus_rms_mean", "head_minus_rms_median",
    "rms_positive_speakers", "rms_required_positive", "rms_gate_pass",
    "head_positive_speakers", "head_required_positive", "head_gate_pass",
    "head_minus_rms_positive_speakers", "head_minus_rms_required_positive", "head_minus_rms_gate_pass",
    "root_qa_RMS_C0_median", "root_qa_RMS_Cp_median", "root_qa_RMS_Hp_median",
    "root_qa_RMS_Cp_gate", "root_qa_RMS_Hp_gate", "root_qa_head_delta_Cp_mean", "root_qa_head_delta_Hp_mean",
]
endpoint_rows = []
for model in model_order:
    for attribute in attribute_order:
        cell = cells[f"{model}|{attribute}"]
        qrow = qa_rows[(model, attribute)]
        gate_map = cell["endpoint_mean_median_and_descriptive_gates"]["positive_gates"]
        for summary in cell["endpoint_mean_median_and_descriptive_gates"]["summaries"]:
            scale, endpoint = summary["scale"], summary["endpoint"]
            gates = gate_map[f"{scale}_{endpoint}"]
            rms_gate = gates["rms"]
            head_gate = gates["head"]
            delta_gate = gates["head_minus_rms"]
            endpoint_rows.append({
                "model": model, "attribute": attribute, "support_status": cell["support_status"],
                "n_speakers_with_both_classes": cell["n_speakers_with_both_classes"],
                "endpoint": endpoint, "scale": scale,
                "n_speakers": summary["rms"]["n_speakers"],
                "rms_mean": summary["rms"]["mean"], "rms_median": summary["rms"]["median"],
                "head_mean": summary["head"]["mean"], "head_median": summary["head"]["median"],
                "head_minus_rms_mean": summary["head_minus_rms"]["mean"], "head_minus_rms_median": summary["head_minus_rms"]["median"],
                "rms_positive_speakers": rms_gate["positive_speakers"], "rms_required_positive": rms_gate["required_positive"],
                "rms_gate_pass": rms_gate["passes_descriptive_baseline_rule"],
                "head_positive_speakers": head_gate["positive_speakers"], "head_required_positive": head_gate["required_positive"],
                "head_gate_pass": head_gate["passes_descriptive_baseline_rule"],
                "head_minus_rms_positive_speakers": delta_gate["positive_speakers"],
                "head_minus_rms_required_positive": delta_gate["required_positive"],
                "head_minus_rms_gate_pass": delta_gate["passes_descriptive_baseline_rule"],
                "root_qa_RMS_C0_median": qrow["RMS_C0_median"], "root_qa_RMS_Cp_median": qrow["RMS_Cp_median"],
                "root_qa_RMS_Hp_median": qrow["RMS_Hp_median"], "root_qa_RMS_Cp_gate": qrow["RMS_Cp_gate"],
                "root_qa_RMS_Hp_gate": qrow["RMS_Hp_gate"], "root_qa_head_delta_Cp_mean": qrow["head_delta_Cp_mean"],
                "root_qa_head_delta_Hp_mean": qrow["head_delta_Hp_mean"],
            })
with ENDPOINT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=endpoint_fields)
    writer.writeheader()
    for row in endpoint_rows:
        writer.writerow({k: fnum(v) for k, v in row.items()})

# Full 17-lag standardized curves: all values are copied from the saved summaries.
curve_fields = ["model", "attribute", "scale", "lag", "n_speakers", "rms_mean", "rms_median", "head_mean", "head_median", "head_minus_rms_mean", "head_minus_rms_median"]
speaker_fields = ["model", "attribute", "speaker", "lag", "scale", "rms_z"]
curve_rows, speaker_rows = [], []
curve_lookup = {}
for model in model_order:
    for attribute in attribute_order:
        key = f"{model}|{attribute}"
        cell = cells[key]
        z_rows = sorted((r for r in cell["curve_mean_median"] if r["scale"] == "z"), key=lambda r: int(r["lag"]))
        curve_lookup[(model, attribute)] = {int(r["lag"]): r for r in z_rows}
        for r in z_rows:
            curve_rows.append({
                "model": model, "attribute": attribute, "scale": "z", "lag": int(r["lag"]),
                "n_speakers": r["rms"]["n_speakers"],
                "rms_mean": r["rms"]["mean"], "rms_median": r["rms"]["median"],
                "head_mean": r["head"]["mean"], "head_median": r["head"]["median"],
                "head_minus_rms_mean": r["head_minus_rms"]["mean"], "head_minus_rms_median": r["head_minus_rms"]["median"],
            })
        for sp in cell["speaker_curves"]:
            for r in sorted((r for r in sp["lag_curves"] if r["scale"] == "z"), key=lambda r: int(r["lag"])):
                speaker_rows.append({"model": model, "attribute": attribute, "speaker": sp["speaker"], "lag": int(r["lag"]), "scale": "z", "rms_z": r["rms"]})
with CURVE_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=curve_fields)
    writer.writeheader()
    for row in curve_rows:
        writer.writerow({k: fnum(v) for k, v in row.items()})
with SPEAKER_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=speaker_fields)
    writer.writeheader()
    for row in speaker_rows:
        writer.writerow({k: fnum(v) for k, v in row.items()})

# 2 x 3 figure, shared scales; no peak or lag is selected from the curves.
matplotlib.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(16, 9.4), sharex=True, sharey=True)
color_rms = "#2878A8"
color_head = "#D57922"
color_speaker = "#91AFC0"
all_y = []
for model in model_order:
    for attribute in attribute_order:
        cell = cells[f"{model}|{attribute}"]
        z_by_lag = curve_lookup[(model, attribute)]
        all_y.extend(float(r["rms"]["median"]) for r in z_by_lag.values())
        all_y.extend(float(r["head"]["median"]) for r in z_by_lag.values())
        for sp in cell["speaker_curves"]:
            all_y.extend(float(r["rms"]) for r in sp["lag_curves"] if r["scale"] == "z")
max_abs = max(abs(v) for v in all_y) if all_y else 1.0
y_bound = max(1.0, np.ceil(max_abs * 2.0) / 2.0 + 0.1)
for row_idx, attribute in enumerate(attribute_order):
    for col_idx, model in enumerate(model_order):
        ax = axes[row_idx, col_idx]
        cell = cells[f"{model}|{attribute}"]
        lags = np.arange(-8, 9)
        z_by_lag = curve_lookup[(model, attribute)]
        # Optional person-level RMS trajectories are faint; no person-level head line is implied.
        for sp in cell["speaker_curves"]:
            srows = {int(r["lag"]): r for r in sp["lag_curves"] if r["scale"] == "z"}
            ax.plot(lags, [srows[int(k)]["rms"] for k in lags], color=color_speaker, alpha=0.34, linewidth=0.9, linestyle=(0, (2, 2)), zorder=1)
        ax.plot(lags, [z_by_lag[int(k)]["rms"]["median"] for k in lags], color=color_rms, marker="o", markersize=3.2, linewidth=2.0, label="RMS speaker median", zorder=3)
        ax.plot(lags, [z_by_lag[int(k)]["head"]["median"] for k in lags], color=color_head, marker="s", markersize=3.0, linewidth=2.0, label="v4 head speaker median", zorder=4)
        ax.axvline(0, color="#59656f", linewidth=1.05, linestyle="--", alpha=0.9, zorder=2)
        ax.axhline(0, color="#9aa5ad", linewidth=0.8, alpha=0.55, zorder=0)
        ax.set_xlim(-8, 8)
        ax.set_ylim(-y_bound, y_bound)
        ax.set_xticks([-8, -4, 0, 4, 8])
        ax.grid(axis="y", color="#dce2e6", linewidth=0.75, alpha=0.75)
        ns = int(cell["n_speakers_with_both_classes"])
        ax.set_title(f"{model}\n{attribute_labels[attribute]}  ·  n={ns} speakers", fontsize=10.4, pad=8)
        ax.tick_params(labelsize=8.5)
        if col_idx == 0:
            ax.set_ylabel("Signed standardized contrast (z)", fontsize=9.5)
        if row_idx == 1:
            ax.set_xlabel("Visual read lag (frames)", fontsize=9.5)

fig.suptitle("Phone-conditioned visual response across the saved 17-lag grid", fontsize=17, fontweight="bold", y=0.975)
fig.text(0.5, 0.925, "Median across supported speakers; faint curves show individual RMS speaker trajectories. Vertical 0 = original clock.", ha="center", fontsize=10.5, color="#4b5963")
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color=color_rms, marker="o", linewidth=2, markersize=4, label="RMS speaker median"),
    Line2D([0], [0], color=color_head, marker="s", linewidth=2, markersize=4, label="v4 head speaker median"),
    Line2D([0], [0], color=color_speaker, linewidth=1, linestyle=(0, (2, 2)), alpha=0.7, label="RMS speaker curves (faint)"),
]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.238), ncol=3, frameon=False, fontsize=9.5)
all_baseline_gates = all(bool(r["RMS_Cp_gate"]) and bool(r["RMS_Hp_gate"]) for r in qa["six_cell_summary"])
if not all_baseline_gates:
    raise ValueError("ROOT QA baseline Cp/Hp gate state differs from frozen task scope")
fig.text(0.5, 0.139, "Per-utterance p varies; no common p or single response peak is selected. Curves use saved row-standardized values without lag-wise restandardization.", ha="center", fontsize=9.0, color="#34424c")
fig.text(0.5, 0.108, "RMS Cₚ and Hₚ descriptive gates: all six model × attribute cells pass (as recorded in ROOT QA).", ha="center", fontsize=9.2, color="#34424c")
fig.text(0.5, 0.078, "Exploratory functional response only; no best-lag, causal-accuracy, or real-mouth accuracy claim.", ha="center", fontsize=9.0, color="#66737c")
fig.subplots_adjust(left=0.085, right=0.99, top=0.865, bottom=0.31, hspace=0.36, wspace=0.22)
fig.savefig(PNG, dpi=180, facecolor="white", bbox_inches="tight")
fig.savefig(SVG, facecolor="white", bbox_inches="tight")
plt.close(fig)

# Final, compact provenance receipt. Output metrics are not repeated here.
outputs = {}
for path in [ENDPOINT_CSV, CURVE_CSV, SPEAKER_CSV, PNG, SVG]:
    outputs[path.name] = {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
qa_status = qa.get("status", "")
receipt_data = {
    "schema": "functional_lag_response_report40_assets_v1",
    "status": "CREATED_FROM_FROZEN_JSON_SUMMARIES_ONLY",
    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    "inputs": [
        {"path": str(EFFECT_PATH.resolve()), "sha256": effect_sha, "expected_sha256": EXPECTED_EFFECT_SHA, "schema": effect.get("schema"), "status": effect.get("status")},
        {"path": str(QA_PATH.resolve()), "sha256": qa_sha, "expected_sha256": EXPECTED_QA_SHA, "schema": qa.get("schema"), "status": qa_status},
    ],
    "plot_runtime": {"python": platform.python_version(), "python_executable": sys.executable, "matplotlib": matplotlib.__version__, "matplotlib_path": str(Path(matplotlib.__file__).resolve()), "numpy": np.__version__, "numpy_path": str(Path(np.__file__).resolve()), "no_install": True},
    "figure": {"panels": "2x3; rows are vowel-height/aperture and vowel-rounding proxies; columns are Wav2Lip, MuseTalk, Ditto", "curve_source": "saved z-scale lag rows; speaker median RMS/v4 head plus faint saved individual RMS speaker curves", "lag_grid": "-8..8 inclusive from frozen source", "original_clock_marker": 0, "per_utterance_p": "varies; no common p or selected peak", "baseline_rms_cp_hp_descriptive_gates_passed": all_baseline_gates, "claim_boundary": "No best-lag, causal-accuracy, or real-mouth accuracy claim."},
    "rows": {"endpoint_summary": len(endpoint_rows), "aggregate_lag_curve_points": len(curve_rows), "speaker_rms_curve_points": len(speaker_rows), "cells": len(cells)},
    "outputs": outputs,
    "work_scope": {"model_scoring_or_source_effect_rerun": False, "new_analysis": False, "gpu": False, "tests": False, "dependency_install": False},
}
RECEIPT.write_text(json.dumps(receipt_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"receipt_path": str(RECEIPT.resolve()), "receipt_sha256": sha256(RECEIPT), "inputs": [{"path": str(EFFECT_PATH.resolve()), "sha256": effect_sha}, {"path": str(QA_PATH.resolve()), "sha256": qa_sha}], "outputs": outputs, "row_counts": receipt_data["rows"], "matplotlib": matplotlib.__version__, "python": platform.python_version()}, ensure_ascii=False, indent=2))

