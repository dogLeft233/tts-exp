"""Deterministic non-interactive paired timeline plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import atomic_write_json


def plot_sample(sample_id: str, arm_records: Mapping[str, Mapping[str, Any]], *, output_path: Path, config_hash: str) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    if not hasattr(axes, "__len__"):
        axes = [axes]
    max_time = 0.0
    for axis, arm in zip(axes, ("natural", "tts")):
        record = arm_records.get(arm)
        axis.set_title(f"{sample_id} / {arm}")
        if record is None:
            axis.text(0.5, 0.5, "unavailable", transform=axis.transAxes, ha="center", va="center")
            continue
        grid = record.get("grid", record)
        times = np.asarray(grid.get("timestamps_s", []), dtype=float)
        confidence = np.asarray(grid.get("local_c", []), dtype=float)
        if times.size:
            axis.plot(times, confidence, color="#263238", linewidth=0.9, label="local_c")
            threshold = record.get("threshold")
            if threshold is not None:
                axis.axhline(float(threshold), color="#455a64", linestyle="--", linewidth=0.8, label="threshold")
            max_time = max(max_time, float(times[-1]))
        for span in record.get("error_spans", []):
            axis.axvspan(float(span["start_s"]), float(span["end_s"]), color="#e53935", alpha=0.18)
            max_time = max(max_time, float(span["end_s"]))
        for span in record.get("excluded_regions", []):
            axis.axvspan(float(span[0]), float(span[1]), color="#b0bec5", alpha=0.28)
        metric = record.get("metrics", {})
        title_suffix = " ".join(
            f"{name}={metric[name]}" for name in ("word_error_rate", "spearman_error_vs_negative_local") if metric.get(name) is not None
        )
        if title_suffix:
            axis.text(0.995, 0.95, title_suffix, transform=axis.transAxes, ha="right", va="top", fontsize=8)
        axis.grid(alpha=0.18)
        axis.set_ylabel("local_c")
    axes[-1].set_xlabel("audio time (s)")
    figure.suptitle(f"ASR error / local SyncNet diagnostic — {sample_id}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=120, format="png")
    plt.close(figure)
    return {"sample_id": sample_id, "path": str(output_path), "config_hash": config_hash}


def write_plot_index(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_json(path, {"schema_version": 1, "plots": [dict(row) for row in rows]})
