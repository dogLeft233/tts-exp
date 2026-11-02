#!/usr/bin/env python3
"""Create a compact, frozen-summary-only scatter figure and CSV."""
from __future__ import annotations
import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
INPUTS = [
    ROOT / ".codex/scratch/native_axis_model_factor_scope_render_20261014_attempt02/root_complete_native_coordinate_grid_summary.json",
    ROOT / ".codex/scratch/physical_phone_spectral_shape_fit_20261014_attempt04/root_physical_six_endpoint_summary.json",
]
CSV_PATH = OUT / "native_physical_endpoint_rows.csv"
SVG_PATH = OUT / "native_physical_endpoint_joint_figure.svg"
PNG_PATH = OUT / "native_physical_endpoint_joint_figure.png"
RECEIPT_PATH = OUT / "figure_receipt.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fmt(value: float) -> str:
    return format(float(value), ".17g")


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def read_rows(path: Path, dataset: str) -> tuple[dict, list[dict]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    expect = "root_complete_native_coordinate_grid_v1" if dataset == "native" else "root_physical_phone_shape_six_endpoint_summary_v1"
    if raw.get("schema") != expect or len(raw.get("rows", [])) != (18 if dataset == "native" else 6):
        raise ValueError(f"Unexpected frozen summary schema/row count: {path}")
    rows = []
    for source_row in raw["rows"]:
        d = source_row["delta"]
        rows.append({
            "dataset": dataset,
            "model": source_row["model"],
            "speaker": source_row["speaker"],
            "factor": source_row["factor"],
            "delta_D0": float(d["D0"]),
            "delta_Cbest": float(d["Cbest"]),
            "delta_Dmin": float(d["Dmin"]),
            "delta_aperture_median": float(d["aperture_median"]),
            "delta_velocity_rms": float(d["velocity_rms"]),
            "joint_gate": bool(source_row.get("gate", source_row.get("joint_gate", False))),
            "source_summary": source_row.get("source_summary", ""),
        })
    return raw, rows


NATIVE, native_rows = read_rows(INPUTS[0], "native")
PHYSICAL, physical_rows = read_rows(INPUTS[1], "physical")
ALL_ROWS = native_rows + physical_rows
if len(ALL_ROWS) != 24:
    raise ValueError(f"Expected 24 summary rows, found {len(ALL_ROWS)}")

# CSV values are direct copies of the precomputed delta fields in ROOT summaries.
with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
    fields = ["dataset", "model", "speaker", "factor", "delta_D0", "delta_Cbest", "delta_Dmin", "delta_aperture_median", "delta_velocity_rms", "joint_gate", "source_summary"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in ALL_ROWS:
        writer.writerow({
            "dataset": row["dataset"], "model": row["model"], "speaker": row["speaker"], "factor": row["factor"],
            "delta_D0": fmt(row["delta_D0"]), "delta_Cbest": fmt(row["delta_Cbest"]), "delta_Dmin": fmt(row["delta_Dmin"]),
            "delta_aperture_median": fmt(row["delta_aperture_median"]), "delta_velocity_rms": fmt(row["delta_velocity_rms"]),
            "joint_gate": str(row["joint_gate"]).lower(), "source_summary": row["source_summary"],
        })

# Shared axes make horizontal/vertical direction and panel locations comparable.
W, H = 1600, 1120
XMIN, XMAX = -0.27, 0.18
YMIN, YMAX = -0.19, 0.15
PLOT_TOP, PLOT_BOTTOM = 165, 710
PANEL_X = {"native": 45, "physical": 815}
PANEL_W = 740
PLOT_LEFT_PAD, PLOT_RIGHT_PAD = 72, 20
PLOT_W = PANEL_W - PLOT_LEFT_PAD - PLOT_RIGHT_PAD
PLOT_H = PLOT_BOTTOM - PLOT_TOP
FACTOR_COLORS = {
    "SEP_ANG_ACTIVE05": "#2774AE",
    "STAB_ANG_FIT": "#D57A12",
    "ENV_RADIAL_V2": "#19866B",
    "PHYSICAL_PHONE_SPECTRAL_SHAPE": "#7656A7",
}
MODEL_LETTERS = {"Wav2Lip": "W", "Ditto": "D", "MuseTalk": "M"}
MODEL_SHAPES = {"Wav2Lip": "circle", "Ditto": "square", "MuseTalk": "triangle"}


def px(panel: str, x: float) -> float:
    left = PANEL_X[panel] + PLOT_LEFT_PAD
    return left + (x - XMIN) / (XMAX - XMIN) * PLOT_W


def py(panel: str, y: float) -> float:
    return PLOT_TOP + (YMAX - y) / (YMAX - YMIN) * PLOT_H


def text(x: float, y: float, content: object, size: int = 14, weight: str = "400", fill: str = "#25313b", anchor: str = "start", extra: str = "") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}px" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>{esc(content)}</text>'


def marker(x: float, y: float, model: str, color: str, size: float = 6.0, stroke: str = "#17212b", sw: float = 1.2) -> str:
    if MODEL_SHAPES[model] == "circle":
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" fill="{color}" stroke="{stroke}" stroke-width="{sw:.1f}"/>'
    if MODEL_SHAPES[model] == "square":
        return f'<rect x="{x-size:.2f}" y="{y-size:.2f}" width="{2*size:.2f}" height="{2*size:.2f}" rx="0.8" fill="{color}" stroke="{stroke}" stroke-width="{sw:.1f}"/>'
    return f'<path d="M {x:.2f} {y-size-0.8:.2f} L {x+size+0.8:.2f} {y+size:.2f} L {x-size-0.8:.2f} {y+size:.2f} Z" fill="{color}" stroke="{stroke}" stroke-width="{sw:.1f}"/>'


def label_layout(points: list[dict], panel: str) -> dict[int, tuple[float, float]]:
    offsets = [(8, -9), (8, 7), (-31, -9), (-31, 7), (-1, -20), (-1, 14), (13, -20), (-35, -20), (13, 14), (-35, 14), (20, -2), (-43, -2), (2, -28), (2, 22)]
    left = PANEL_X[panel] + PLOT_LEFT_PAD
    right = left + PLOT_W
    occupied: list[tuple[float, float, float, float]] = []
    layout: dict[int, tuple[float, float]] = {}
    order = sorted(range(len(points)), key=lambda i: (py(panel, points[i]["delta_Cbest"]), px(panel, points[i]["delta_D0"])))
    for i in order:
        row = points[i]
        x, y = px(panel, row["delta_D0"]), py(panel, row["delta_Cbest"])
        best = None
        for dx, dy in offsets:
            lx = min(max(x + dx, left + 1), right - 29)
            ly = min(max(y + dy - 8, PLOT_TOP + 1), PLOT_BOTTOM - 14)
            box = (lx, ly, lx + 28, ly + 14)
            overlap = 0.0
            for ox1, oy1, ox2, oy2 in occupied:
                overlap += max(0.0, min(box[2], ox2) - max(box[0], ox1)) * max(0.0, min(box[3], oy2) - max(box[1], oy1))
            # Prefer nearby annotations while avoiding existing label boxes.
            cost = overlap * 1000.0 + math.hypot(lx - (x + 8), ly - (y - 17))
            if best is None or cost < best[0]:
                best = (cost, box)
            if overlap == 0:
                break
        assert best is not None
        box = best[1]
        occupied.append(box)
        layout[i] = (box[0], box[1])
    return layout


parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
         '<rect width="100%" height="100%" fill="#ffffff"/>',
         '<style>text{font-kerning:normal} .tick{font-size:12px;fill:#4a5863}</style>']
parts.append(text(52, 48, "Native-coordinate vs physical spectral-shape endpoint screen", 27, "700", "#17212b"))
parts.append(text(53, 76, "Per-endpoint changes from frozen ROOT summaries; both panels use identical axes", 15, "400", "#53616c"))

for panel, rows, heading in [
    ("native", native_rows, "Native-coordinate interventions · 18 endpoints"),
    ("physical", physical_rows, "Physical phone spectral shape · 6 endpoints"),
]:
    base_x = PANEL_X[panel]
    left, right = base_x + PLOT_LEFT_PAD, base_x + PLOT_LEFT_PAD + PLOT_W
    parts.append(f'<rect x="{base_x}" y="104" width="{PANEL_W}" height="718" rx="10" fill="#fbfcfd" stroke="#d3dbe1"/>')
    parts.append(text(base_x + 18, 136, heading, 17, "700", "#17212b"))
    x_ticks = [-0.25, -0.15, -0.05, 0, 0.05, 0.15]
    y_ticks = [-0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15]
    for tick in x_ticks:
        x = px(panel, tick)
        iszero = abs(tick) < 1e-12
        parts.append(f'<line x1="{x:.2f}" y1="{PLOT_TOP}" x2="{x:.2f}" y2="{PLOT_BOTTOM}" stroke="{"#7b8790" if iszero else "#e5e9ec"}" stroke-width="{"1.7" if iszero else "0.9"}"/>')
        parts.append(text(x, PLOT_BOTTOM + 21, f"{tick:.2f}" if tick else "0", 12, "400", "#4a5863", "middle"))
    for tick in y_ticks:
        y = py(panel, tick)
        iszero = abs(tick) < 1e-12
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="{"#7b8790" if iszero else "#e5e9ec"}" stroke-width="{"1.7" if iszero else "0.9"}"/>')
        parts.append(text(left - 10, y + 4, f"{tick:.2f}" if tick else "0", 12, "400", "#4a5863", "end"))
    parts.append(f'<rect x="{left}" y="{PLOT_TOP}" width="{PLOT_W}" height="{PLOT_H}" fill="none" stroke="#9ba6ae" stroke-width="1"/>')
    parts.append(text((left + right)/2, PLOT_BOTTOM + 48, "ΔD₀  (better ←)", 14, "600", "#26343e", "middle"))
    parts.append(text(base_x + 20, (PLOT_TOP + PLOT_BOTTOM)/2, "ΔCbest  (higher ↑)", 14, "600", "#26343e", "middle", f'transform="rotate(-90 {base_x+20} {(PLOT_TOP+PLOT_BOTTOM)/2:.1f})"'))
    # Data point labels encode model and speaker; color encodes factor; marker shape encodes model.
    positions = label_layout(rows, panel)
    for i, row in enumerate(rows):
        color = FACTOR_COLORS[row["factor"]]
        x, y = px(panel, row["delta_D0"]), py(panel, row["delta_Cbest"])
        parts.append(marker(x, y, row["model"], color))
        lx, ly = positions[i]
        label = MODEL_LETTERS[row["model"]] + row["speaker"][-2:]
        parts.append(f'<rect x="{lx:.1f}" y="{ly:.1f}" width="28" height="14" rx="3" fill="#ffffff" fill-opacity="0.92" stroke="{color}" stroke-width="0.85"/>')
        parts.append(text(lx + 14, ly + 10.5, label, 9, "700", "#202a32", "middle"))
    if panel == "native":
        note1 = "Joint rows: 4/18  ·  both-speaker model×factor pairs: 0/9"
        note2 = "Colors: factor; marker/label: model and speaker (44/52)"
    else:
        note1 = "Joint rows: 0/6"
        note2 = "Both ΔCbest gains coincide with worse ΔD₀ (Ditto 44; Wav2Lip 52)"
    parts.append(text(base_x + 18, 786, note1, 13, "700", "#26343e"))
    parts.append(text(base_x + 18, 806, note2, 12, "400", "#53616c"))

# Compact factor/model keys, then interpretation limits.
parts.append(text(52, 858, "Factor", 13, "700", "#26343e"))
legend_factors = [
    ("SEP_ANG_ACTIVE05", "SEP_ANG_ACTIVE05"),
    ("STAB_ANG_FIT", "STAB_ANG_FIT"),
    ("ENV_RADIAL_V2", "ENV_RADIAL_V2"),
    ("PHYSICAL_PHONE_SPECTRAL_SHAPE", "Physical phone spectral shape"),
]
x = 108
for key, label in legend_factors:
    parts.append(f'<rect x="{x}" y="{846}" width="12" height="12" rx="2" fill="{FACTOR_COLORS[key]}"/>')
    parts.append(text(x + 18, 857, label, 12, "400", "#34424c"))
    x += 184 if key != "PHYSICAL_PHONE_SPECTRAL_SHAPE" else 240
parts.append(text(52, 891, "Model markers", 13, "700", "#26343e"))
x = 170
for model, label in [("Wav2Lip", "Wav2Lip (W)"), ("Ditto", "Ditto (D)"), ("MuseTalk", "MuseTalk (M)")]:
    parts.append(marker(x, 886, model, "#c8d0d6", size=5.3))
    parts.append(text(x + 14, 891, label, 12, "400", "#34424c"))
    x += 160
parts.append(text(52, 931, "Joint screen requires ΔCbest > 0, ΔDmin ≤ 0, and ΔD₀ ≤ 0; the scatter shows only ΔD₀ and ΔCbest.", 14, "600", "#26343e"))
parts.append(text(52, 960, "n=2 speakers (S0044, S0052) per endpoint; exploratory screen. Mouth aperture/velocity are movement descriptions, not accuracy against a ground truth.", 13, "400", "#53616c"))
parts.append(text(52, 1002, "Interpretation is limited to these frozen doses and endpoints; no inferential tests were run for this figure.", 12, "400", "#6b7780"))
parts.append(text(W - 52, 1068, "Source: frozen ROOT summary JSONs · 24 endpoints", 11, "400", "#7b8790", "end"))
parts.append("</svg>")
SVG_PATH.write_text("\n".join(parts), encoding="utf-8")

# Try the already-installed Edge renderer for a raster companion; no packages are installed.
edge_candidates = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]
edge = next((p for p in edge_candidates if p.exists()), None)
png_status = {"status": "UNAVAILABLE", "reason": "No compatible raster renderer found"}
if edge:
    try:
        PNG_PATH.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(prefix="endpoint-figure-edge-", ignore_cleanup_errors=True) as tmp:
            profile = Path(tmp) / "profile"
            cmd = [str(edge), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", "--no-default-browser-check", f"--user-data-dir={profile}", f"--window-size={W},{H}", f"--screenshot={PNG_PATH}", SVG_PATH.resolve().as_uri()]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45, check=False)
            for _ in range(30):
                if PNG_PATH.exists() and PNG_PATH.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                    break
                time.sleep(0.1)
            if proc.returncode == 0 and PNG_PATH.exists():
                blob = PNG_PATH.read_bytes()
                if blob.startswith(b"\x89PNG\r\n\x1a\n") and len(blob) >= 24:
                    dims = struct.unpack(">II", blob[16:24])
                    png_status = {"status": "RENDERED", "sha256": sha256(PNG_PATH), "bytes": len(blob), "width": dims[0], "height": dims[1], "renderer": str(edge), "returncode": proc.returncode}
                else:
                    PNG_PATH.unlink(missing_ok=True)
                    png_status = {"status": "FAILED", "returncode": proc.returncode, "reason": "Renderer did not produce a valid PNG"}
            else:
                PNG_PATH.unlink(missing_ok=True)
                png_status = {"status": "FAILED", "returncode": proc.returncode, "stderr_tail": proc.stderr.decode("utf-8", "replace")[-1200:]}
    except Exception as exc:
        PNG_PATH.unlink(missing_ok=True)
        png_status = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}

input_records = []
for path, raw in zip(INPUTS, [NATIVE, PHYSICAL]):
    input_records.append({"path": str(path), "sha256": sha256(path), "schema": raw["schema"], "row_count": len(raw["rows"])})
outputs = {
    "csv": {"path": str(CSV_PATH), "sha256": sha256(CSV_PATH), "bytes": CSV_PATH.stat().st_size},
    "svg": {"path": str(SVG_PATH), "sha256": sha256(SVG_PATH), "bytes": SVG_PATH.stat().st_size},
    "png": {"path": str(PNG_PATH), **png_status} if png_status["status"] == "RENDERED" else png_status,
}
receipt = {
    "schema": "native_physical_endpoint_joint_figure_v1",
    "status": "COMPLETE",
    "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "source": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
    "inputs": input_records,
    "row_count": len(ALL_ROWS),
    "outputs": outputs,
    "rendering": {"method": "direct SVG from Python standard library; PNG companion from installed Edge headless renderer when available", "python": platform.python_version(), "no_dependency_install": True},
    "derivation": "Copied precomputed delta.D0, delta.Cbest, delta.Dmin, delta.aperture_median, delta.velocity_rms and gate fields directly from the frozen ROOT summaries; no model scores or statistical tests recomputed.",
    "displayed_frozen_counts": {"native_joint_rows": NATIVE["joint_pass_count"], "native_two_speaker_model_factor_passes": sum(bool(r["two_speaker_gate"]) for r in NATIVE["replication"]), "native_model_factor_pairs": len(NATIVE["replication"]), "physical_joint_rows": PHYSICAL["joint_pass_count"], "physical_c_gain_rows": sum(float(r["delta"]["Cbest"]) > 0 for r in PHYSICAL["rows"]), "physical_c_gain_rows_with_worse_D0": sum(float(r["delta"]["Cbest"]) > 0 and float(r["delta"]["D0"]) > 0 for r in PHYSICAL["rows"])},
    "caveats": ["Joint screen requires ΔCbest>0, ΔDmin<=0, and ΔD0<=0; the plot shows only ΔD0 and ΔCbest.", "Two-speaker results are exploratory n=2 endpoint screens.", "Mouth aperture and velocity are descriptive movement metrics with no ground-truth accuracy label."],
}
RECEIPT_PATH.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"receipt": str(RECEIPT_PATH), "receipt_sha256": sha256(RECEIPT_PATH), "inputs": input_records, "outputs": outputs, "csv_rows": len(ALL_ROWS)}, ensure_ascii=False, indent=2))





