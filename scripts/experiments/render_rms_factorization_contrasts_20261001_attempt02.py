#!/usr/bin/env python3
"""Render frozen RMS mean/trajectory/joint contrast summaries to PNG."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image


SOURCE_DIR = Path("runs/recovered_20260930/independent_validation/rms_consumer_attempt07_pass_readback")
SOURCE_REPORT = SOURCE_DIR / "independent_rms_factorization_report.json"
SOURCE_COMPACT = SOURCE_DIR / "independent_rms_compact_sidecar_v3.json"
EXPECTED_REPORT_SHA256 = "6371c540e15ff9f1a0ba8a0bda21f7836be5d92305d100e20712c02bdaa9afe6"
EXPECTED_COMPACT_SHA256 = "06176334245bfe97925f027dceccac35a3f1214844b3553658d14aaf5f400249"
OUTPUT_DIR = Path("runs/recovered_20260930/independent_validation/mechanism_figures_20261001_attempt06")

SUPPORTS = ("full", "valid", "guard15", "guard20")
SUPPORT_COLORS = {
    "full": "#49566b",
    "valid": "#8b6faf",
    "guard15": "#198c8c",
    "guard20": "#d17a22",
}
CONTRASTS = (
    ("direction_q10_minus_q00", "Direction · q10−q00", "o"),
    ("radius_q01_minus_q00", "Trajectory · q01−q00", "s"),
    ("joint_q11_minus_q00", "Joint · q11−q00", "D"),
    ("q10_minus_q01", "Direction−trajectory · q10−q01", "^"),
    ("interaction_q11_minus_q10_minus_q01_plus_q00", "Factor response · q11−q10−q01+q00", "P"),
)
METRICS = (("C", "C"), ("C_anchor", "C anchor"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data() -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = SOURCE_REPORT.resolve(strict=True)
    compact_path = SOURCE_COMPACT.resolve(strict=True)
    report_sha = sha256_file(report_path)
    compact_sha = sha256_file(compact_path)
    if report_sha != EXPECTED_REPORT_SHA256 or compact_sha != EXPECTED_COMPACT_SHA256:
        raise ValueError("RMS report/compact sidecar differs from the frozen attempt07 readback")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    compact = json.loads(compact_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("processed_complete_rows") != 80 or report.get("intended_rows") != 80:
        raise ValueError("RMS independent report is not the complete 80-row pass")
    if compact.get("status") != "PASS" or compact.get("processed_complete_rows") != 80 or compact.get("intended_rows") != 80:
        raise ValueError("RMS compact sidecar is not the complete 80-row pass")
    source = compact.get("source_report", {})
    if source.get("sha256") != report_sha or source.get("payload_sha256") != report.get("payload_sha256"):
        raise ValueError("compact sidecar is not bound to the expected report")
    return report, compact


def collect_points(compact: dict[str, Any], metric: str, estimand: str) -> list[dict[str, Any]]:
    source = compact["equal_speaker_contrasts"]
    output: list[dict[str, Any]] = []
    for contrast_key, contrast_label, marker in CONTRASTS:
        for geometry in ("raw", "unit"):
            for support in SUPPORTS:
                record = source[contrast_key][geometry][support][metric][estimand]
                if record.get("complete_speakers") != 40 or record.get("intended_speakers") != 40:
                    raise ValueError(f"incomplete frozen endpoint: {contrast_key}/{geometry}/{support}/{metric}/{estimand}")
                ci = record.get("ci99")
                if not isinstance(ci, dict) or not {"lower", "upper"}.issubset(ci):
                    raise ValueError(f"missing frozen 99% CI: {contrast_key}/{geometry}/{support}/{metric}/{estimand}")
                estimate = float(record["mean_equal_weight_complete_speakers"])
                low, high = float(ci["lower"]), float(ci["upper"])
                if not (low <= estimate <= high):
                    raise ValueError(f"point estimate lies outside CI: {contrast_key}/{geometry}/{support}/{metric}/{estimand}")
                output.append({
                    "contrast": contrast_key,
                    "contrast_label": contrast_label,
                    "marker": marker,
                    "geometry": geometry,
                    "support": support,
                    "estimand": estimand,
                    "metric": metric,
                    "estimate": estimate,
                    "ci99": [low, high],
                    "complete_speakers": 40,
                    "paired_clips": 80,
                })
    if len(output) != 40:
        raise ValueError(f"expected 40 plotted points, got {len(output)}")
    return output


def plot_panel(ax: Any, points: list[dict[str, Any]], geometry: str, estimand: str, show_labels: bool) -> None:
    subset = [p for p in points if p["geometry"] == geometry and p["estimand"] == estimand]
    contrast_order = {key: i for i, (key, _, _) in enumerate(CONTRASTS)}
    support_order = {key: i for i, key in enumerate(SUPPORTS)}
    subset.sort(key=lambda p: (contrast_order[p["contrast"]], support_order[p["support"]]))
    for y, point in enumerate(subset):
        estimate = point["estimate"]
        low, high = point["ci99"]
        color = SUPPORT_COLORS[point["support"]]
        ax.errorbar(
            estimate,
            y,
            xerr=[[estimate - low], [high - estimate]],
            fmt=point["marker"],
            color=color,
            ecolor=color,
            elinewidth=1.35,
            capsize=2.5,
            markersize=5.2,
            markeredgewidth=0.6,
            zorder=3,
        )
    labels = [f'{p["contrast_label"]} · {p["support"]}' for p in subset]
    ax.set_yticks(range(len(labels)), labels if show_labels else [""] * len(labels), fontsize=8.4)
    ax.invert_yaxis()
    ax.axvline(0.0, color="#202733", linewidth=1.1, linestyle="--", zorder=1)
    lows = [p["ci99"][0] for p in subset]
    highs = [p["ci99"][1] for p in subset]
    left, right = min(0.0, min(lows)), max(0.0, max(highs))
    margin = max((right - left) * 0.08, 0.001)
    ax.set_xlim(left - margin, right + margin)
    ax.grid(axis="x", color="#dfe3e9", linewidth=0.75)
    ax.set_axisbelow(True)
    stat_title = "Observed contrast" if estimand == "observed" else "Observed minus mean time permutation"
    ax.set_title(f"{geometry.capitalize()} geometry · {stat_title}", loc="left", fontsize=12.2, fontweight="bold", pad=8)
    ax.set_xlabel("SyncNet distance contrast", fontsize=9.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    report, compact = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    report_sha = sha256_file(SOURCE_REPORT)
    compact_sha = sha256_file(SOURCE_COMPACT)
    images: dict[str, Any] = {}

    for metric, metric_label in METRICS:
        observed = collect_points(compact, metric, "observed")
        excess = collect_points(compact, metric, "null_adjusted")
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
        for ax, geometry, estimand, show_labels in (
            (axes[0, 0], "raw", "observed", True),
            (axes[0, 1], "unit", "observed", False),
            (axes[1, 0], "raw", "null_adjusted", True),
            (axes[1, 1], "unit", "null_adjusted", False),
        ):
            plot_panel(ax, observed if estimand == "observed" else excess, geometry, estimand, show_labels)
        fig.suptitle(
            f"RMS WORLD factorization contrasts · {metric_label}",
            fontsize=17,
            fontweight="bold",
        )
        contrast_handles = [
            Line2D([0], [0], marker=marker, color="#303744", linestyle="none", label=label)
            for _, label, marker in CONTRASTS
        ]
        support_handles = [
            Line2D([0], [0], marker="o", color=SUPPORT_COLORS[support], linestyle="none", label=support)
            for support in SUPPORTS
        ]
        fig.legend(
            handles=contrast_handles + support_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.982),
            ncol=5,
            frameon=False,
            fontsize=8.5,
        )
        fig.text(
            0.01,
            0.004,
            "80 clips / 40 speakers; two clips averaged within speaker. 99% paired speaker-bootstrap intervals. "
            "Null-adjusted = observed contrast minus mean of 100 full-A time permutations. "
            "The factorial interaction is a coordinated factor response, not a decoder-nonlinearity estimate.",
            fontsize=8.2,
            color="#505a68",
        )
        name = f"rms_factorization_{metric.lower()}_contrasts.png"
        path = OUTPUT_DIR / name
        fig.savefig(path, dpi=100, facecolor="white")
        plt.close(fig)
        with Image.open(path) as image:
            dimensions = list(image.size)
        if dimensions[0] != 1800:
            raise ValueError(f"unexpected render width for {name}: {dimensions}")
        images[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "dimensions_px": dimensions,
            "plotted_points": 80,
            "metric": metric,
            "contrasts": [key for key, _, _ in CONTRASTS],
            "supports": list(SUPPORTS),
            "geometries": ["raw", "unit"],
            "estimands": ["observed", "null_adjusted"],
        }

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "renderer": "Matplotlib Agg; values copied from the independently validated RMS attempt07 compact sidecar",
        "render_source": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "source_report": {"path": str(SOURCE_REPORT.resolve()), "sha256": report_sha, "payload_sha256": report["payload_sha256"]},
        "source_compact_sidecar": {"path": str(SOURCE_COMPACT.resolve()), "sha256": compact_sha},
        "cohort": {"clips": 80, "speakers": 40, "clips_per_speaker": 2},
        "intervals": "Per-endpoint 99% paired speaker bootstrap intervals from the frozen consumer; not a max-t family interval",
        "images": images,
    }
    manifest_path = OUTPUT_DIR / "png_manifest.json"
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "manifest": str(manifest_path.resolve()), "images": images}, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
