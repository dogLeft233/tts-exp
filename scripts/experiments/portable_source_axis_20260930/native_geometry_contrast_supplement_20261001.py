#!/usr/bin/env python3
"""Paired all-cohort summaries for frozen native visual geometry means."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260930
METRICS = (
    "aperture",
    "absolute_aperture_velocity_per_s",
    "mouth_width",
    "yaw_proxy",
    "pitch_proxy",
)
CONTRASTS = (
    "native_t_raw_minus_n_raw",
    "natural_fixed_minus_natural_raw",
    "tts_fixed_minus_tts_raw",
    "change_in_t_minus_n_gap_after_fixed",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def speaker_summary(values: dict[str, float], speaker_by_sample: dict[str, str], samples: list[str], speakers: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        grouped[speaker_by_sample[sample]].append(float(values[sample]))
    if set(grouped) != set(speakers) or any(len(grouped[speaker]) != 2 for speaker in speakers):
        raise ValueError("paired geometry contrast does not cover exactly two clips for every intended speaker")
    per_speaker = {speaker: float(np.mean(grouped[speaker], dtype=np.float64)) for speaker in speakers}
    data = np.asarray([per_speaker[speaker] for speaker in speakers], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, len(data), size=(BOOTSTRAP_DRAWS, len(data)), endpoint=False)
    draws = data[indices].mean(axis=1, dtype=np.float64)
    return {
        "equal_speaker_mean": float(data.mean(dtype=np.float64)),
        "ci99": [float(x) for x in np.quantile(draws, [0.005, 0.995], method="linear")],
        "complete_clips": len(samples),
        "complete_two_clip_speakers": len(speakers),
        "intended_clips": 80,
        "intended_speakers": 40,
        "speaker_values": per_speaker,
        "clip_values": values,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "unit": "speaker; each speaker is the mean of their two clip-level paired contrasts",
    }


def run(report_path: Path, out_path: Path) -> dict[str, Any]:
    report_path = report_path.resolve(strict=True)
    out_path = out_path.resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to replace immutable supplement: {out_path}")
    source = json.loads(report_path.read_text(encoding="utf-8"))
    if source.get("status") != "COMPLETE" or source.get("analysis_type") != "native_4cell_visual_phone_and_geometry_diagnostic":
        raise ValueError("source report is not the completed frozen native visual analysis")
    clip_rows = source.get("per_clip")
    if not isinstance(clip_rows, dict) or len(clip_rows) != 80:
        raise ValueError("source report does not contain all 80 intended clips")
    samples = sorted(clip_rows)
    speaker_by_sample = {sample: str(clip_rows[sample]["speaker_id"]) for sample in samples}
    speakers = sorted(set(speaker_by_sample.values()))
    if len(speakers) != 40 or any(sum(speaker_by_sample[sample] == speaker for sample in samples) != 2 for speaker in speakers):
        raise ValueError("source report cohort is not 40 speakers × 2 clips")

    cell_values: dict[str, dict[str, dict[str, float]]] = {
        cell: {metric: {} for metric in METRICS}
        for cell in ("natural_raw", "natural_fixed", "tts_raw", "tts_fixed")
    }
    for sample in samples:
        source_rows = clip_rows[sample]["source_results"]
        for cell, source_name, level in (
            ("natural_raw", "natural", "raw"), ("natural_fixed", "natural", "fixed"),
            ("tts_raw", "tts", "raw"), ("tts_fixed", "tts", "fixed"),
        ):
            track_metrics = source_rows[source_name]["overall_track_metrics"][level]
            for metric in METRICS:
                value = track_metrics[metric].get("mean")
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(f"missing/nonfinite all-track mean for {sample}/{cell}/{metric}")
                cell_values[cell][metric][sample] = float(value)

    contrast_values: dict[str, dict[str, dict[str, float]]] = {
        contrast: {metric: {} for metric in METRICS} for contrast in CONTRASTS
    }
    for metric in METRICS:
        for sample in samples:
            nr = cell_values["natural_raw"][metric][sample]
            nf = cell_values["natural_fixed"][metric][sample]
            tr = cell_values["tts_raw"][metric][sample]
            tf = cell_values["tts_fixed"][metric][sample]
            contrast_values["native_t_raw_minus_n_raw"][metric][sample] = tr - nr
            contrast_values["natural_fixed_minus_natural_raw"][metric][sample] = nf - nr
            contrast_values["tts_fixed_minus_tts_raw"][metric][sample] = tf - tr
            contrast_values["change_in_t_minus_n_gap_after_fixed"][metric][sample] = (tf - nf) - (tr - nr)

    summaries = {
        contrast: {
            metric: speaker_summary(contrast_values[contrast][metric], speaker_by_sample, samples, speakers)
            for metric in METRICS
        }
        for contrast in CONTRASTS
    }
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "analysis_type": "native_visual_all_cohort_paired_geometry_mean_supplement",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "input": "existing immutable native v2 per-clip full-track metrics; no feature NPZ reopened",
            "cell_metric": "mean over the source-specific RAW/FIXED common valid clock or adjacent valid frame pairs as already computed",
            "contrasts": {
                "native_t_raw_minus_n_raw": "T_RAW - N_RAW",
                "natural_fixed_minus_natural_raw": "N_FIXED - N_RAW",
                "tts_fixed_minus_tts_raw": "T_FIXED - T_RAW",
                "change_in_t_minus_n_gap_after_fixed": "(T_FIXED - N_FIXED) - (T_RAW - N_RAW)",
            },
            "statistics": "80 clip-level paired differences -> each speaker's two-clip mean -> equal-speaker mean; 20,000 PCG64 speaker bootstrap draws, seed 20260930, percentile 99% CI",
            "scope": "descriptive visible mouth/head geometry, not phone accuracy, perception, or event timing",
            "median_field_note": "The frozen v2 report's cell-level median entries are null because _summary_values emits mean/sd/p10/p90 while the aggregator requests median. This supplement uses only the stored per-clip means and does not modify or backfill the original report.",
        },
        "intended_denominator": {"clips": 80, "speakers": 40, "clips_per_speaker": 2},
        "source_report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "input_seals": source["input_seals"],
        "cell_means": {
            cell: {
                metric: speaker_summary(cell_values[cell][metric], speaker_by_sample, samples, speakers)
                for metric in METRICS
            }
            for cell in cell_values
        },
        "paired_contrasts": summaries,
    }
    atomic_json(out_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.report, args.out)
    print(json.dumps({"status": result["status"], "source_report": result["source_report"], "output": str(args.out.resolve()), "sha256": sha256_file(args.out.resolve())}, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
