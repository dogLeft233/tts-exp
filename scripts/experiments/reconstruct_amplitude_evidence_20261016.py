#!/usr/bin/env python3
"""Reconstruct compact tables and Figure 2 from accepted report artifacts.

This script only reads already accepted RMS, MEL-partition, and visual-analysis
JSON reports. It does not call a model, score media, or decode videos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[2]
RMS_PATH = ROOT / "runs/recovered_20260930/independent_validation/rms_consumer_attempt07_pass_readback/independent_rms_factorization_report.json"
MEL_PATH = ROOT / "runs/recovered_20260930/independent_validation/mel_partition_consumer_20261001_attempt07/independent_mel_report.json"
VISUAL_PATH = ROOT / "runs/recovered_20260930/independent_validation/native_geometry_visual_20261001_attempt02/analysis_v2/native_4cell_visual_analysis.json"
DEFAULT_OUT = ROOT / "runs/recovered_20260930/independent_validation/amplitude_evidence_reconstruction_20261016_attempt03"

EXPECTED_SHA256 = {
    RMS_PATH: "6371c540e15ff9f1a0ba8a0bda21f7836be5d92305d100e20712c02bdaa9afe6",
    MEL_PATH: "112ff916b3c6b1790f3cf3db1b75721f08f84486aca0e16d828ec432bf4bc6fb",
    VISUAL_PATH: "72e7f68df2920a062f8703855bdc6d994f59223851f266e5a94713356e9ac341",
}

GEOMETRIES = ("raw", "unit")
SUPPORTS = ("full", "valid", "guard15", "guard20")
CELLS = ("q00", "q10", "q01", "q11")
METRICS = ("B", "C", "C_anchor", "D", "D0", "D_anchor", "search_gain")
MEL_METRICS = ("B", "C", "C_anchor", "D", "D0", "search_gain")
CONTRASTS = {
    "direction_q10_minus_q00": {"q10": 1.0, "q00": -1.0},
    "radius_q01_minus_q00": {"q01": 1.0, "q00": -1.0},
    "joint_q11_minus_q00": {"q11": 1.0, "q00": -1.0},
    "q10_minus_q01": {"q10": 1.0, "q01": -1.0},
    "interaction_q11_minus_q10_minus_q01_plus_q00": {
        "q11": 1.0, "q10": -1.0, "q01": -1.0, "q00": 1.0,
    },
}
MEL_CONTRAST_LABELS = {
    "interior_q10_minus_q00": {"q10": 1.0, "q00": -1.0},
    "boundary_q01_minus_q00": {"q01": 1.0, "q00": -1.0},
    "joint_q11_minus_q00": {"q11": 1.0, "q00": -1.0},
    "q10_minus_q01": {"q10": 1.0, "q01": -1.0},
    "interaction_q11_minus_q10_minus_q01_plus_q00": {
        "q11": 1.0, "q10": -1.0, "q01": -1.0, "q00": 1.0,
    },
}
RMS_Q_LABELS = {
    "q00": "RAW endpoint",
    "q10": "direction-only",
    "q01": "radius-only",
    "q11": "FIXED endpoint",
}
MEL_Q_LABELS = {
    "q00": "baseline",
    "q10": "interior-only",
    "q01": "boundary-only",
    "q11": "interior + boundary",
}


def derive_rms_decomposition(
    curve: list[float], zero_column: int = 15, anchor_column: int = 18
) -> dict[str, float | int]:
    """Derive the accepted report's scalar decomposition from one lag curve."""
    values = [float(x) for x in curve]
    if len(values) != 31 or any(not math.isfinite(x) for x in values):
        raise ValueError("RMS curve must contain 31 finite distances")
    if not (0 <= zero_column < len(values) and 0 <= anchor_column < len(values)):
        raise ValueError("zero and anchor columns must fall inside the curve")
    min_column = min(range(len(values)), key=values.__getitem__)
    b = float(median(values))
    d = values[min_column]
    d0 = values[zero_column]
    d_anchor = values[anchor_column]
    return {
        "B": b,
        "D": d,
        "D0": d0,
        "D_anchor": d_anchor,
        "best_lag": min_column - zero_column,
        "C": b - d,
        "C_anchor": b - d_anchor,
        "search_gain": d0 - d,
    }


def validate_common_support(cells_by_q: dict[str, dict], context: str = "") -> tuple[int, ...]:
    """Reject q-cell distance comparisons that do not share frame support."""
    if set(cells_by_q) != set(CELLS):
        raise ValueError(f"missing q cells for common-support check: {context}")
    signatures = {tuple(int(i) for i in cell.get("indices", [])) for cell in cells_by_q.values()}
    support_ns = {int(cell.get("support_n", -1)) for cell in cells_by_q.values()}
    if len(signatures) != 1 or len(support_ns) != 1:
        raise ValueError(f"q-cell supports differ: {context}")
    signature = next(iter(signatures))
    if not signature or len(signature) != next(iter(support_ns)):
        raise ValueError(f"empty or inconsistent frame support: {context}")
    return signature


def equal_speaker_curve_mean(
    by_speaker: dict[str, list[float]],
    expected_speakers: set[str],
    clips_per_speaker: int = 2,
) -> float:
    """Average clips within speaker, then speakers; refuse silent missingness."""
    if set(by_speaker) != expected_speakers:
        raise ValueError("curve aggregate is missing or adding speakers")
    if any(len(values) != clips_per_speaker for values in by_speaker.values()):
        raise ValueError("curve aggregate requires the declared clip count per speaker")
    return mean(mean(values) for values in by_speaker.values())


def forest_rows(mel_reported: list[dict], rms_reported: list[dict]) -> list[dict]:
    """Select the predeclared intervention ΔC estimates and accepted 99% CIs."""
    selections = [
        ("MEL partition", "interior_q10_minus_q00", "Interior-only (q10−q00)", "primary", mel_reported, ""),
        ("MEL partition", "boundary_q01_minus_q00", "Boundary-only (q01−q00)", "primary", mel_reported, ""),
        ("MEL partition", "joint_q11_minus_q00", "Joint (q11−q00)", "secondary", mel_reported, ""),
        ("RMS factorization", "direction_q10_minus_q00", "Direction-only (q10−q00)", "primary", rms_reported, "observed"),
        ("RMS factorization", "radius_q01_minus_q00", "Radius-only (q01−q00)", "primary", rms_reported, "observed"),
        ("RMS factorization", "joint_q11_minus_q00", "Joint (q11−q00)", "secondary", rms_reported, "observed"),
    ]
    selected = []
    for experiment, contrast, label, priority, rows, estimand in selections:
        matches = [r for r in rows if r["contrast"] == contrast and r["geometry"] == "raw"
                   and r["support"] == "guard20" and r["metric"] == "C"
                   and r.get("estimand", "") == estimand]
        if len(matches) != 1:
            raise ValueError(f"expected one accepted forest estimate for {experiment} / {contrast}")
        r = matches[0]
        if r.get("equal_speaker_mean") is None or r.get("ci99_lower") is None or r.get("ci99_upper") is None:
            raise ValueError(f"accepted 99% interval missing for {experiment} / {contrast}")
        selected.append({
            "experiment": experiment, "contrast": contrast, "label": label,
            "priority": priority, "geometry": "raw", "support": "guard20", "metric": "C",
            "equal_speaker_mean_delta_C": float(r["equal_speaker_mean"]),
            "ci99_lower": float(r["ci99_lower"]), "ci99_upper": float(r["ci99_upper"]),
            "speaker_count": int(r.get("complete_speakers") or r.get("speaker_count") or 0),
        })
    return selected


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_pinned(path: Path) -> tuple[dict, str]:
    actual = sha256(path)
    expected = EXPECTED_SHA256[path]
    if actual.lower() != expected.lower():
        raise RuntimeError(f"input SHA mismatch: {path} expected={expected} actual={actual}")
    return json.loads(path.read_text(encoding="utf-8")), actual


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def mean_or_none(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return mean(clean) if clean else None


def equal_speaker_mean(by_speaker: dict[str, list[float]]) -> tuple[float | None, int, int]:
    per_speaker = [mean(v) for v in by_speaker.values() if v]
    return (mean(per_speaker) if per_speaker else None, len(per_speaker), sum(map(len, by_speaker.values())))


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def speaker_clip_counts(rows: list[dict], speaker_key: str = "speaker") -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row[speaker_key])] += 1
    return dict(counts)


def build_rms(rms: dict, out: Path, ids: set[str]) -> tuple[list[dict], list[dict], list[dict], dict]:
    rows = rms["rows"]
    assert {r["id"] for r in rows} == ids and len(rows) == 80
    assert len({r["speaker"] for r in rows}) == 40
    curve_rows: list[dict] = []
    contrast_clip_rows: list[dict] = []
    curve_buckets: dict[tuple[str, str, str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    lag_contract = rms["reconstruction_protocol"]["score_constants"]
    zero_column = int(lag_contract["zero_column"])
    anchor_column = int(lag_contract["anchor_column"])
    vshift = int(lag_contract["vshift"])
    if (zero_column, anchor_column, vshift) != (15, 18, 15):
        raise RuntimeError(f"unexpected RMS lag contract: {lag_contract}")
    lag_offsets = list(range(-vshift, vshift + 1))
    expected_speakers = {r["speaker"] for r in rows}

    for row in rows:
        sid = row["speaker"]
        for geom in GEOMETRIES:
            for support in SUPPORTS:
                validate_common_support(
                    {q: row["observed_metrics"]["per_geometry"][geom][q][support] for q in CELLS},
                    f"{row['id']} / {geom} / {support}",
                )
            for q in CELLS:
                for support in SUPPORTS:
                    cell = row["observed_metrics"]["per_geometry"][geom][q][support]
                    curve = [float(x) for x in cell["curve"]]
                    if len(curve) != 2 * vshift + 1:
                        raise RuntimeError(f"unexpected curve width {row['id']} {geom} {q} {support}: {len(curve)}")
                    best_lag = int(cell["best_lag"])
                    if not -vshift <= best_lag <= vshift:
                        raise RuntimeError(f"best_lag outside search grid: {row['id']} {best_lag}")
                    derived = derive_rms_decomposition(curve, zero_column, anchor_column)
                    checks = {name: derived[name] for name in ("B", "D", "D0", "D_anchor", "C", "C_anchor", "search_gain")}
                    for name, expected in checks.items():
                        if abs(float(cell[name]) - expected) > 2e-5:
                            raise RuntimeError(f"RMS {name}/curve mismatch {row['id']} {geom} {q} {support}")
                    if int(derived["best_lag"]) != best_lag:
                        raise RuntimeError(f"RMS best_lag/curve mismatch {row['id']} {geom} {q} {support}")
                    record = {
                        "clip_id": row["id"], "speaker_id": sid, "geometry": geom,
                        "support": support, "cell": q,
                        **{metric: finite(cell.get(metric)) for metric in METRICS},
                        "best_lag": best_lag, "support_n": cell.get("support_n"),
                        "curve_lag_offsets": json.dumps(lag_offsets, separators=(",", ":")),
                        "curve_distances": json.dumps(curve, separators=(",", ":")),
                        "fixed_support_frame_indices": json.dumps(cell.get("indices", []), separators=(",", ":")),
                    }
                    curve_rows.append(record)
                    for ix, val in enumerate(curve):
                        curve_buckets[(geom, support, q, lag_offsets[ix])][sid].append(val)

            # Per-clip contrasts use the report's frozen q-cell algebra.
            for support in SUPPORTS:
                cellset = row["observed_metrics"]["per_geometry"][geom]
                for contrast, weights in CONTRASTS.items():
                    for metric in METRICS:
                        vals = [weights[q] * float(cellset[q][support][metric]) for q in weights]
                        contrast_clip_rows.append({
                            "clip_id": row["id"], "speaker_id": sid,
                            "geometry": geom, "support": support,
                            "contrast": contrast, "metric": metric,
                            "observed_contrast": sum(vals),
                        })

    write_csv(out / "rms_clip_cells_curves.csv", [
        "clip_id", "speaker_id", "geometry", "support", "cell", "B", "C", "C_anchor",
        "D", "D0", "D_anchor", "search_gain", "best_lag", "support_n",
        "curve_lag_offsets", "curve_distances", "fixed_support_frame_indices",
    ], curve_rows)
    write_csv(out / "rms_clip_contrasts.csv", list(contrast_clip_rows[0]), contrast_clip_rows)

    aggregate_curves = []
    for (geom, support, q, lag), by_speaker in sorted(curve_buckets.items()):
        curve_mean = equal_speaker_curve_mean(by_speaker, expected_speakers, clips_per_speaker=2)
        aggregate_curves.append({
            "geometry": geom, "support": support, "cell": q,
            "lag_offset_frames": lag,
            "equal_speaker_mean_distance": curve_mean,
            "speaker_count": len(expected_speakers),
            "clip_count": sum(len(v) for v in by_speaker.values()),
            "weighting": "mean of two clips within each speaker, then equal mean across speakers",
        })
    write_csv(out / "rms_equal_speaker_curves.csv", list(aggregate_curves[0]), aggregate_curves)

    # Include the already accepted report-level bootstrap summaries; no new CI is computed.
    official = []
    for contrast, geoms in rms["equal_speaker_contrasts"].items():
        for geom, supports in geoms.items():
            for support, metrics in supports.items():
                for metric, estimands in metrics.items():
                    for estimand, value in estimands.items():
                        ci = value.get("bootstrap_99_ci", {})
                        official.append({
                            "contrast": contrast, "geometry": geom, "support": support,
                            "metric": metric, "estimand": estimand,
                            "equal_speaker_mean": value.get("mean_equal_weight_complete_speakers"),
                            "ci99_lower": ci.get("lower_99"), "ci99_upper": ci.get("upper_99"),
                            "complete_speakers": value.get("complete_speakers"),
                            "intended_speakers": value.get("intended_speakers"),
                            "paired_clips": value.get("paired_clips_in_complete_speakers"),
                        })
    write_csv(out / "rms_reported_contrasts.csv", list(official[0]), official)
    return curve_rows, aggregate_curves, official, {
        "row_count": len(rows), "speaker_count": len({r["speaker"] for r in rows}),
        "cell_curve_rows": len(curve_rows), "curve_width": 2 * vshift + 1,
        "lag_offsets_frames": lag_offsets, "score_constants": lag_contract,
        "all_rows_complete": all(r["status"] == "complete" for r in rows),
    }


def build_mel(mel: dict, out: Path, ids: set[str]) -> tuple[list[dict], list[dict], dict]:
    rows = mel["evaluation_score_reconstruction"]["rows"]
    assert {r["id"] for r in rows} == ids and len(rows) == 80
    clip_rows: list[dict] = []
    speaker_contrasts: list[dict] = []
    for row in rows:
        for geom in GEOMETRIES:
            for q in CELLS:
                for support in SUPPORTS:
                    cell = row["cell_metrics"][q][geom][support]
                    clip_rows.append({
                        "clip_id": row["id"], "speaker_id": row["speaker"],
                        "geometry": geom, "support": support, "cell": q,
                        **{metric: finite(cell.get(metric)) for metric in METRICS if metric in cell},
                        "best_lag": "not_present_in_accepted_MEL_report",
                        "full_curve": "not_present_in_accepted_MEL_report",
                    })
            for support in SUPPORTS:
                for contrast, weights in MEL_CONTRAST_LABELS.items():
                    for metric in MEL_METRICS:
                        values = [
                            weights[q] * float(row["cell_metrics"][q][geom][support][metric])
                            for q in weights
                        ]
                        speaker_contrasts.append({
                            "clip_id": row["id"], "speaker_id": row["speaker"],
                            "geometry": geom, "support": support,
                            "contrast": contrast, "metric": metric,
                            "clip_contrast": sum(values),
                        })
    mel_fields = ["clip_id", "speaker_id", "geometry", "support", "cell", "B", "C", "C_anchor", "D", "D0", "D_anchor", "search_gain", "best_lag", "full_curve"]
    write_csv(out / "mel_clip_cells.csv", mel_fields, clip_rows)
    write_csv(out / "mel_clip_contrasts.csv", list(speaker_contrasts[0]), speaker_contrasts)

    reported = []
    for contrast, geoms in mel["aggregate_statistics"]["contrasts"].items():
        for geom, supports in geoms.items():
            for support, metrics in supports.items():
                for metric, value in metrics.items():
                    boot = value.get("bootstrap", {})
                    ci = boot.get("ci99", [])
                    reported.append({
                        "contrast": contrast, "geometry": geom, "support": support,
                        "metric": metric, "equal_speaker_mean": value.get("equal_speaker_mean"),
                        "ci99_lower": ci[0] if len(ci) == 2 else None,
                        "ci99_upper": ci[1] if len(ci) == 2 else None,
                        "speaker_count": value.get("speaker_count"),
                        "complete_clips": value.get("complete_clips"),
                    })
    write_csv(out / "mel_reported_contrasts.csv", list(reported[0]), reported)
    return clip_rows, reported, {
        "row_count": len(rows), "speaker_count": len({r["speaker"] for r in rows}),
        "cell_metric_rows": len(clip_rows), "curve_arrays_recovered": False,
        "curve_gap": "2000-array ledger is in report, but local accepted consumer folder contains only the aggregate report; do not infer curves from cell metrics.",
        "cells": {"q00": "baseline", "q10": "interior-only MEL change", "q01": "boundary-only MEL change", "q11": "interior+boundary"},
    }


def build_motion(visual: dict, out: Path, ids: set[str]) -> tuple[list[dict], list[dict], dict]:
    per_clip = visual["per_clip"]
    assert set(per_clip) == ids and len(per_clip) == 80
    clips: list[dict] = []
    bucket: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    speaker_values: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: {"raw": [], "fixed": []})
    for clip_id, item in per_clip.items():
        speaker = item.get("speaker_id") or clip_id.split("_")[0]
        for source in ("natural", "tts"):
            sr = item.get("source_results", {}).get(source)
            for arm in ("raw", "fixed"):
                track = (sr or {}).get("overall_track_metrics", {}).get(arm, {})
                clock = (sr or {}).get("common_clock", {})
                aperture = finite(track.get("aperture", {}).get("mean"))
                velocity = finite(track.get("absolute_aperture_velocity_per_s", {}).get("mean"))
                record = {
                    "clip_id": clip_id, "speaker_id": speaker, "source": source, "arm": arm,
                    "status": "observed" if aperture is not None and velocity is not None else "missing_track_metric",
                    "original_mfa_pair_qualified": item.get("original_mfa_pair_qualified"),
                    "valid_common_frames": clock.get("valid_common_frames"),
                    "common_clock_coverage": clock.get("common_coverage"),
                    "matched_frames_baseline": clock.get("matched_frames_by_arm", {}).get("baseline"),
                    "matched_frames_fixed": clock.get("matched_frames_by_arm", {}).get("fixed"),
                    "aperture_mean": aperture,
                    "aperture_p10": finite(track.get("aperture", {}).get("p10")),
                    "aperture_p90": finite(track.get("aperture", {}).get("p90")),
                    "absolute_aperture_velocity_per_s_mean": velocity,
                    "absolute_aperture_velocity_per_s_p10": finite(track.get("absolute_aperture_velocity_per_s", {}).get("p10")),
                    "absolute_aperture_velocity_per_s_p90": finite(track.get("absolute_aperture_velocity_per_s", {}).get("p90")),
                    "common_frame_count": track.get("common_frame_count"),
                    "valid_fraction_on_full_clock": track.get("valid_fraction_on_full_clock"),
                }
                clips.append(record)
                for metric, val in (("aperture_mean", aperture), ("absolute_aperture_velocity_per_s_mean", velocity)):
                    if val is not None:
                        speaker_values[(source, metric, speaker)][arm].append(val)

    write_csv(out / "mouth_motion_clip.csv", list(clips[0]), clips)
    per_speaker = []
    summary_buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    complete_counts: dict[tuple[str, str], int] = defaultdict(int)
    for (source, metric, speaker), arms in sorted(speaker_values.items()):
        raw = mean(arms["raw"]) if arms["raw"] else None
        fixed = mean(arms["fixed"]) if arms["fixed"] else None
        matched_n = min(len(arms["raw"]), len(arms["fixed"]))
        delta = fixed - raw if raw is not None and fixed is not None and len(arms["raw"]) == 2 and len(arms["fixed"]) == 2 else None
        per_speaker.append({
            "speaker_id": speaker, "source": source, "metric": metric,
            "raw_mean_across_clips": raw, "fixed_mean_across_clips": fixed,
            "fixed_minus_raw_two_clip_paired": delta,
            "raw_clip_count": len(arms["raw"]), "fixed_clip_count": len(arms["fixed"]),
            "paired_complete_two_clip": delta is not None,
        })
        if raw is not None:
            summary_buckets[(source, metric, "raw")].append(raw)
        if fixed is not None:
            summary_buckets[(source, metric, "fixed")].append(fixed)
        if delta is not None:
            summary_buckets[(source, metric, "fixed_minus_raw")].append(delta)
            complete_counts[(source, metric)] += 1
    write_csv(out / "mouth_motion_speaker.csv", list(per_speaker[0]), per_speaker)

    summary = []
    for source in ("natural", "tts"):
        for metric in ("aperture_mean", "absolute_aperture_velocity_per_s_mean"):
            for arm in ("raw", "fixed", "fixed_minus_raw"):
                vals = summary_buckets[(source, metric, arm)]
                summary.append({
                    "source": source, "metric": metric, "arm_or_contrast": arm,
                    "equal_speaker_mean": mean(vals) if vals else None,
                    "speaker_count": len(vals),
                    "complete_two_clip_speakers_for_paired_delta": complete_counts[(source, metric)] if arm == "fixed_minus_raw" else None,
                })
    write_csv(out / "mouth_motion_summary.csv", list(summary[0]), summary)
    return clips, summary, {
        "clip_arm_rows": len(clips), "intended_clip_count": len(per_clip),
        "intended_speakers": len({x.get("speaker_id") for x in per_clip.values()}),
        "physical_overlay_applied": False,
        "summary_support": "equal speaker; paired raw/fixed contrast requires two clips in both arms",
    }


def cell_mean(rows: list[dict], key: str, metric: str, geom: str = "raw", support: str = "guard20") -> dict[str, float]:
    by_speaker: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["geometry"] == geom and row["support"] == support and row["cell"] == key:
            by_speaker[row["speaker_id"]].append(float(row[metric]))
    return {"mean": mean([mean(v) for v in by_speaker.values() if v]), "n_speakers": len(by_speaker)}


def scale_y(value: float, low: float, high: float, top: float, height: float) -> float:
    if high <= low:
        return top + height / 2
    return top + height - (value - low) / (high - low) * height


def svg_figure(out: Path, rms_curves: list[dict], motion_summary: list[dict], intervention_points: list[dict]) -> None:
    width, height = 1240, 1090
    colors = {"q00": "#334155", "q10": "#2f7ea8", "q01": "#d17a22", "q11": "#7b61a8"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="#ffffff"/>',
             '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2937}.small{font-size:12px}.axis{stroke:#9ca3af;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.title{font-size:18px;font-weight:700}.sub{font-size:13px;fill:#4b5563}.legend{font-size:12px}</style>',
             '<text x="42" y="38" font-size="24" font-weight="700">Figure 2. Amplitude mechanism evidence</text>',
             '<text x="42" y="61" class="sub">Accepted reports; 40 speakers / 80 clips; means use equal speaker weights (two clips each) where available</text>']

    def panel(title: str, x: int, y: int, w: int, h: int) -> tuple[float, float, float, float]:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#fff" stroke="#cbd5e1"/>')
        parts.append(f'<text x="{x+18}" y="{y+26}" class="title">{escape(title)}</text>')
        return x + 56, y + 58, w - 82, h - 112

    def lineplot(title: str, x: int, y: int, w: int, h: int,
                 series: list[tuple[str, list[float], list[float], str]], xlabel: str, ylabel: str,
                 xticks: list[tuple[float, str]] | None = None) -> None:
        left, top, pw, ph = panel(title, x, y, w, h)
        all_y = [v for _, _, ys, _ in series for v in ys if math.isfinite(v)]
        low, high = min(all_y), max(all_y)
        pad = (high-low)*0.12 if high > low else max(abs(high)*0.1, 0.1)
        low -= pad; high += pad
        all_x = [v for _, xs, _, _ in series for v in xs]
        xmin, xmax = min(all_x), max(all_x)
        for frac in (0, .25, .5, .75, 1):
            gy = top + ph * frac
            val = high - frac * (high-low)
            parts.append(f'<line x1="{left}" y1="{gy:.2f}" x2="{left+pw}" y2="{gy:.2f}" class="grid"/>')
            parts.append(f'<text x="{left-8}" y="{gy+4:.2f}" text-anchor="end" class="small">{val:.3g}</text>')
        if xticks is None:
            xticks = [(xv, str(xv)) for xv in sorted(set(all_x))]
        for xv, label in xticks:
            gx = left + (xv-xmin)/(xmax-xmin if xmax != xmin else 1)*pw
            parts.append(f'<line x1="{gx:.2f}" y1="{top}" x2="{gx:.2f}" y2="{top+ph}" class="grid"/>')
            parts.append(f'<text x="{gx:.2f}" y="{top+ph+18}" text-anchor="middle" class="small">{escape(label)}</text>')
        parts.append(f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" class="axis"/>')
        parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}" class="axis"/>')
        for si, (label, xs, ys, color) in enumerate(series):
            points = []
            for xv, val in zip(xs, ys):
                px = left + (xv-xmin)/(xmax-xmin if xmax != xmin else 1)*pw
                py = scale_y(val, low, high, top, ph)
                points.append(f'{px:.2f},{py:.2f}')
            parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.7"/>')
            for xv, val in zip(xs, ys):
                px = left + (xv-xmin)/(xmax-xmin if xmax != xmin else 1)*pw
                py = scale_y(val, low, high, top, ph)
                parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.3" fill="{color}"/>')
            ly = y + 48
            lx = x + 22 + si * max(120, w // max(len(series), 1))
            parts.append(f'<line x1="{lx}" y1="{ly-4}" x2="{lx+18}" y2="{ly-4}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<text x="{lx+24}" y="{ly}" class="legend">{escape(label)}</text>')
        parts.append(f'<text x="{left+pw/2:.2f}" y="{top+ph+37}" text-anchor="middle" class="small">{escape(xlabel)}</text>')
        parts.append(f'<text x="{x+16}" y="{top+ph/2:.2f}" transform="rotate(-90 {x+16} {top+ph/2:.2f})" text-anchor="middle" class="small">{escape(ylabel)}</text>')

    def forest_panel(title: str, x: int, y: int, w: int, h: int, experiment: str) -> None:
        panel(title, x, y, w, h)
        points = [r for r in intervention_points if r["experiment"] == experiment]
        if not points:
            raise ValueError(f"forest panel missing points for {experiment}")
        xmin = min(0.0, *(float(r["ci99_lower"]) for r in points))
        xmax = max(0.0, *(float(r["ci99_upper"]) for r in points))
        margin = max((xmax - xmin) * 0.10, 0.01)
        xmin -= margin
        xmax += margin
        left = x + 225
        right = x + w - 25
        top = y + 58
        bottom = y + h - 55
        plot_w = right - left
        plot_h = bottom - top
        px = lambda value: left + (float(value) - xmin) / (xmax - xmin) * plot_w
        for frac in (0, .25, .5, .75, 1):
            value = xmin + frac * (xmax - xmin)
            gx = px(value)
            parts.append(f'<line x1="{gx:.2f}" y1="{top}" x2="{gx:.2f}" y2="{bottom}" class="grid"/>')
            parts.append(f'<text x="{gx:.2f}" y="{bottom+18}" text-anchor="middle" class="small">{value:.2f}</text>')
        zero_x = px(0.0)
        parts.append(f'<line x1="{zero_x:.2f}" y1="{top-5}" x2="{zero_x:.2f}" y2="{bottom+2}" stroke="#111827" stroke-width="1.8" stroke-dasharray="5 4"/>')
        parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')
        for i, point in enumerate(points):
            cy = top + (i + .5) * plot_h / len(points)
            color = "#64748b" if point["priority"] == "secondary" else ("#2f7ea8" if experiment == "MEL partition" else "#7b61a8")
            low = px(point["ci99_lower"])
            high = px(point["ci99_upper"])
            estimate = px(point["equal_speaker_mean_delta_C"])
            parts.append(f'<text x="{x+16}" y="{cy+4:.2f}" class="small">{escape(point["label"])}</text>')
            parts.append(f'<line x1="{low:.2f}" y1="{cy:.2f}" x2="{high:.2f}" y2="{cy:.2f}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<line x1="{low:.2f}" y1="{cy-5:.2f}" x2="{low:.2f}" y2="{cy+5:.2f}" stroke="{color}" stroke-width="1.5"/>')
            parts.append(f'<line x1="{high:.2f}" y1="{cy-5:.2f}" x2="{high:.2f}" y2="{cy+5:.2f}" stroke="{color}" stroke-width="1.5"/>')
            parts.append(f'<circle cx="{estimate:.2f}" cy="{cy:.2f}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{(left+right)/2:.2f}" y="{y+h-17}" text-anchor="middle" class="small">ΔC (raw, guard20; accepted 99% CI)</text>')

    forest_panel("A. MEL partition · intervention ΔC", 35, 82, 570, 320, "MEL partition")
    forest_panel("B. RMS factorization · intervention ΔC", 635, 82, 570, 320, "RMS factorization")

    curve_series = []
    for q in CELLS:
        entries = sorted([r for r in rms_curves if r["geometry"] == "raw" and r["support"] == "guard20" and r["cell"] == q], key=lambda z: int(z["lag_offset_frames"]))
        curve_series.append((f"{q} {RMS_Q_LABELS[q]}", [int(z["lag_offset_frames"]) for z in entries], [float(z["equal_speaker_mean_distance"]) for z in entries], colors[q]))
    lineplot("C. RMS factorization · 40-speaker mean distance curves", 35, 424, 1170, 300,
             curve_series, "Lag offset from zero (frames; fixed anchor at +3)", "Mean distance per lag (report units)",
             [(-15, "−15"), (-10, "−10"), (-5, "−5"), (0, "0"), (3, "+3"), (5, "+5"), (10, "+10"), (15, "+15")])

    # Motion summary: compare raw/fixed track aperture and absolute aperture velocity.
    motion_x = [0, 1]
    motion_series = []
    for source, label, color in (("natural", "Natural", "#2f7ea8"), ("tts", "TTS", "#d17a22")):
        for metric, suffix in (("aperture_mean", "aperture"), ("absolute_aperture_velocity_per_s_mean", "velocity")):
            vals = {}
            for arm in ("raw", "fixed"):
                rec = next(x for x in motion_summary if x["source"] == source and x["metric"] == metric and x["arm_or_contrast"] == arm)
                vals[arm] = float(rec["equal_speaker_mean"])
            motion_series.append((f"{label} {suffix}: raw→fixed", motion_x, [vals["raw"], vals["fixed"]], color if suffix == "aperture" else ("#74a9cf" if source == "natural" else "#e7a866")))
    # Two compact motion plots are rendered directly to keep the different units separate.
    # Reuse the panel frame and axes geometry from the standard line plot for aperture.
    aperture_series = [x for x in motion_series if "aperture" in x[0]]
    velocity_series = [x for x in motion_series if "velocity" in x[0]]
    lineplot("D1. Mean mouth aperture", 35, 750, 570, 260, aperture_series,
             "Pair arm", "Normalized aperture", [(0, "raw"), (1, "fixed")])
    lineplot("D2. Mean absolute aperture speed", 635, 750, 570, 260, velocity_series,
             "Pair arm", "Aperture units / second", [(0, "raw"), (1, "fixed")])
    parts.append('<text x="42" y="1045" class="small">Panels A/B: four primary intervention ΔC estimates with accepted 99% CIs; gray joint contrasts are secondary. Curve panel: equal-speaker mean at each lag; C is not recalculated from the mean curve.</text>')
    parts.append('<text x="42" y="1064" class="small">MEL q10/q01 = interior/boundary; RMS q10/q01 = direction/radius. Mouth aperture and speed have separate axes; tracks are observational and no physical overlay was applied.</text>')
    parts.append('</svg>')
    (out / "figure2_amplitude_mechanism.svg").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    rms, rms_sha = read_pinned(RMS_PATH)
    mel, mel_sha = read_pinned(MEL_PATH)
    visual, visual_sha = read_pinned(VISUAL_PATH)
    rms_ids = {row["id"] for row in rms["rows"]}
    mel_ids = {row["id"] for row in mel["evaluation_score_reconstruction"]["rows"]}
    visual_ids = set(visual["per_clip"])
    if rms_ids != mel_ids or rms_ids != visual_ids:
        raise RuntimeError(f"ID denominator mismatch: RMS={len(rms_ids)} MEL={len(mel_ids)} visual={len(visual_ids)}")
    speakers = {row["speaker"] for row in rms["rows"]}
    if len(rms_ids) != 80 or len(speakers) != 40:
        raise RuntimeError(f"unexpected intended cohort: clips={len(rms_ids)} speakers={len(speakers)}")

    out.mkdir(parents=True, exist_ok=False)

    rms_cells, rms_curves, rms_reported, rms_summary = build_rms(rms, out, rms_ids)
    mel_cells, mel_reported, mel_summary = build_mel(mel, out, rms_ids)
    motion_clip, motion_summary, motion_meta = build_motion(visual, out, rms_ids)
    points = forest_rows(mel_reported, rms_reported)
    write_csv(out / "forest_intervention_points.csv", list(points[0]), points)
    svg_figure(out, rms_curves, motion_summary, points)

    metadata = {
        "schema": "amplitude-evidence-derived-v1",
        "created_from_existing_reports_only": True,
        "no_model_scoring_or_video_decoding": True,
        "inputs": {
            "rms_factorization_report": {"path": str(RMS_PATH.relative_to(ROOT)), "sha256": rms_sha},
            "mel_partition_report": {"path": str(MEL_PATH.relative_to(ROOT)), "sha256": mel_sha},
            "native_geometry_visual_analysis": {"path": str(VISUAL_PATH.relative_to(ROOT)), "sha256": visual_sha},
        },
        "intended_cohort": {"clips": 80, "speakers": 40, "speaker_ids": sorted(speakers), "clip_ids": sorted(rms_ids)},
        "rms_factorization": rms_summary,
        "mel_partition": mel_summary,
        "mouth_motion": motion_meta,
        "interpretation_scope": [
            "RMS factorization and MEL partition are distinct precomputed intervention schemes and are shown separately.",
            "q10/q01 labels differ between those schemes; do not compare their labels across panels.",
            "The geometry report measures mouth tracks on source-pair common timestamp support; physical video overlay was not applied.",
            "MEL full lag curves are not present in the accepted local report and were not reconstructed or guessed.",
            "Per-clip records retain all intended IDs; summary means use equal-speaker weighting and report their available support.",
            "Figure panels A/B use the reports' accepted raw/guard20 ΔC estimates and 99% intervals; joint contrasts are secondary.",
            "Figure panel C plots the equal-speaker mean distance at each lag; C is taken from per-clip report cells, not recomputed from the mean curve.",
        ],
        "outputs": [
            "rms_clip_cells_curves.csv", "rms_equal_speaker_curves.csv", "rms_clip_contrasts.csv", "rms_reported_contrasts.csv",
            "mel_clip_cells.csv", "mel_clip_contrasts.csv", "mel_reported_contrasts.csv",
            "mouth_motion_clip.csv", "mouth_motion_speaker.csv", "mouth_motion_summary.csv",
            "forest_intervention_points.csv", "figure2_amplitude_mechanism.svg",
        ],
    }
    (out / "analysis_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "rms": rms_summary, "mel": mel_summary, "motion": motion_meta, "outputs": metadata["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
