"""Render a derived, speaker-level summary from the frozen P1 analyzer outputs.

This utility never reads score_results.json and never recomputes inferential
statistics. Primary confidence limits and the four-dose simultaneous family
are copied from analyze_p1_results.py outputs; the sign counts are descriptive
counts over the planned speaker denominator.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any


DOSE_ROWS = (
    ("−6 dB", "DOSE_M6_dose_delta_C"),
    ("−3 dB", "DOSE_M3_dose_delta_C"),
    ("RAW_R1 · 0 dB reference", None),
    ("+3 dB", "DOSE_P3_dose_delta_C"),
    ("+6 dB", "DOSE_P6_dose_delta_C"),
)
DOSE_CONTRAST_IDS = ("DOSE_M6", "DOSE_M3", "DOSE_P3", "DOSE_P6")


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _csv_float(value: Any, *, field: str, speaker_id: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    result = _finite(value)
    if result is None:
        raise ValueError(f"{speaker_id}.{field} must be blank or finite")
    return result


def _read_speaker_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("speaker_primary.csv must contain the frozen planned cohort")
    return rows


def _required_interval(value: Any, *, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be a two-value interval")
    lo, hi = (_finite(v) for v in value)
    if lo is None or hi is None or lo > hi:
        raise ValueError(f"{label} must be finite and ordered")
    return lo, hi


def build_payload(summary: dict[str, Any], speaker_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Validate and assemble only values already emitted by the primary analyzer."""
    planned_ids = summary.get("planned_speakers")
    if not isinstance(planned_ids, list) or not planned_ids:
        raise ValueError("analysis_summary.json must include planned_speakers")
    planned_ids = [str(s) for s in planned_ids]
    if len(set(planned_ids)) != len(planned_ids):
        raise ValueError("analysis_summary.json contains duplicate planned speakers")
    if int(summary.get("n_planned_speakers", -1)) != len(planned_ids):
        raise ValueError("summary planned-speaker count differs from planned_speakers")

    row_map: dict[str, dict[str, str]] = {}
    for row in speaker_rows:
        speaker_id = str(row.get("speaker_id", ""))
        if not speaker_id or speaker_id in row_map:
            raise ValueError("speaker_primary.csv has a blank or duplicate speaker_id")
        row_map[speaker_id] = row
    if list(row_map) != planned_ids:
        raise ValueError("speaker_primary.csv rows must exactly follow the frozen planned speaker order")

    primary_summary = summary.get("primary_fixed_minus_raw")
    family_summary = summary.get("dose_family_vs_raw_r1")
    if not isinstance(primary_summary, dict) or not isinstance(family_summary, dict):
        raise ValueError("summary is missing primary or dose-family results")

    effects: list[dict[str, Any]] = []
    counts = {"higher": 0, "lower": 0, "zero": 0, "missing": 0}
    for speaker_id in planned_ids:
        row = row_map[speaker_id]
        delta = _csv_float(row.get("primary_delta_C"), field="primary_delta_C", speaker_id=speaker_id)
        is_complete = str(row.get("primary_complete", "")).strip().lower() == "true"
        if is_complete != (delta is not None):
            raise ValueError(f"{speaker_id}: primary_complete and primary_delta_C disagree")
        if delta is None:
            counts["missing"] += 1
        elif delta > 0.0:
            counts["higher"] += 1
        elif delta < 0.0:
            counts["lower"] += 1
        else:
            counts["zero"] += 1
        effects.append({"speaker_id": speaker_id, "delta": delta})

    n_planned = len(planned_ids)
    n_complete = n_planned - counts["missing"]
    if int(primary_summary.get("n_planned", -1)) != n_planned or int(primary_summary.get("n_complete", -1)) != n_complete:
        raise ValueError("speaker primary completion count differs from analysis summary")
    primary_ci = None
    primary_mean = None
    if primary_summary.get("status") == "COMPUTABLE":
        if n_complete != n_planned:
            raise ValueError("primary CI is marked computable with missing planned speakers")
        primary_mean = _finite(primary_summary.get("mean"))
        primary_ci = _required_interval(primary_summary.get("ci95"), label="primary ci95")
        if primary_mean is None:
            raise ValueError("computable primary mean must be finite")

    dose_n = int(family_summary.get("n_complete", -1))
    if int(family_summary.get("n_planned", -1)) != n_planned:
        raise ValueError("dose family planned-speaker count differs from planned_speakers")
    dose_intervals: dict[str, dict[str, Any]] = {}
    if family_summary.get("status") == "COMPUTABLE":
        if dose_n != n_planned:
            raise ValueError("dose family CI is marked computable with missing planned speakers")
        contrast_map = family_summary.get("contrasts")
        if not isinstance(contrast_map, dict):
            raise ValueError("computable dose family must contain contrast results")
        for contrast_id in DOSE_CONTRAST_IDS:
            row = contrast_map.get(contrast_id)
            if not isinstance(row, dict) or int(row.get("n", -1)) != dose_n:
                raise ValueError(f"missing or miscounted frozen dose contrast: {contrast_id}")
            mean = _finite(row.get("mean"))
            interval = _required_interval(row.get("simultaneous_ci95"), label=f"{contrast_id} simultaneous_ci95")
            if mean is None:
                raise ValueError(f"{contrast_id} mean must be finite")
            dose_intervals[contrast_id] = {"mean": mean, "ci": interval}

    dose_values: dict[str, list[dict[str, Any]]] = {}
    for label, csv_field in DOSE_ROWS:
        if csv_field is None:
            continue
        dose_id = csv_field.removesuffix("_dose_delta_C")
        values = []
        for speaker_id in planned_ids:
            value = _csv_float(row_map[speaker_id].get(csv_field), field=csv_field, speaker_id=speaker_id)
            if value is not None:
                values.append({"speaker_id": speaker_id, "delta": value})
        dose_values[dose_id] = values

    protocol_sha = str(summary.get("protocol_sha256", ""))
    if len(protocol_sha) != 64 or any(c not in "0123456789abcdefABCDEF" for c in protocol_sha):
        raise ValueError("summary protocol_sha256 must be a SHA-256 digest")
    return {
        "planned_ids": planned_ids,
        "primary_effects": effects,
        "counts": counts,
        "n_planned": n_planned,
        "n_primary_complete": n_complete,
        "primary_status": primary_summary.get("status"),
        "primary_mean": primary_mean,
        "primary_ci": primary_ci,
        "dose_status": family_summary.get("status"),
        "dose_n": dose_n,
        "dose_intervals": dose_intervals,
        "dose_values": dose_values,
        "protocol_sha256": protocol_sha,
        "input_sha256": str(summary.get("input_sha256", "")),
    }


def _text(x: float, y: float, value: str, size: int = 14, anchor: str = "start", color: str = "#222", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial,sans-serif" font-size="{size}" fill="{color}" font-weight="{weight}">'
        f'{html.escape(value)}</text>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, color: str = "#d9dee3", width: float = 1.0, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr}/>'


def _range(values: list[float]) -> tuple[float, float]:
    if not values:
        return -1.0, 1.0
    low, high = min(values), max(values)
    if high - low < 1e-12:
        padding = max(abs(low) * 0.1, 0.5)
    else:
        padding = (high - low) * 0.16
    return low - padding, high + padding


def render_svg(payload: dict[str, Any]) -> str:
    """Create an SVG from validated analyzer outputs; no statistics are fit here."""
    width, height = 1280, 1115
    left, right = 300.0, 1015.0
    plot_w = right - left
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _text(width / 2, 34, "New-speaker amplitude response: speaker effects and dose family", 23, "middle", weight="bold"),
        _text(width / 2, 59, "C = SyncNet Sync-C; positive ΔC is a higher-score direction, not a speaker-level significance test.", 13, "middle", "#555"),
    ]

    # Panel 1: explicit paired FIXED-minus-RAW effect for every planned speaker.
    ptop = 135.0
    pstep = 18.0
    effects = payload["primary_effects"]
    vals = [float(row["delta"]) for row in effects if row["delta"] is not None]
    pvals = vals[:]
    if payload["primary_ci"] is not None:
        pvals.extend(payload["primary_ci"])
    if payload["primary_mean"] is not None:
        pvals.append(float(payload["primary_mean"]))
    xmin, xmax = _range(pvals)
    def px(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * plot_w
    parts.append(_text(42, 83, "1 · Per-speaker primary effect", 17, weight="bold"))
    parts.append(_text(42, 105, "ΔC = mean(FIXED_R1, FIXED_R2) − mean(RAW_R1, RAW_R2)", 12, color="#555"))
    parts.append(_text(1035, 105, "ΔC", 11, color="#555"))
    zero_x = px(0.0)
    parts.append(_line(zero_x, ptop - 9, zero_x, ptop + (payload["n_planned"] - 1) * pstep + 8, "#4b5563", 1.5, "5,4"))
    for tick in [xmin + (xmax - xmin) * i / 5 for i in range(6)]:
        xx = px(tick)
        parts.append(_line(xx, ptop - 8, xx, ptop + (payload["n_planned"] - 1) * pstep + 8, "#edf0f2", 0.8))
        parts.append(_text(xx, ptop - 15, f"{tick:.2f}", 10, "middle", "#666"))
    for i, effect in enumerate(effects):
        y = ptop + i * pstep
        parts.append(_text(42, y + 4, effect["speaker_id"], 11, color="#39434c"))
        delta = effect["delta"]
        if delta is None:
            parts.append(_text(px(0.0), y + 4, "×", 16, "middle", "#929ba3", "bold"))
            parts.append(_text(1035, y + 4, "missing", 10, color="#737d85"))
        else:
            color = "#2b7a78" if delta > 0 else ("#bb5a48" if delta < 0 else "#333333")
            parts.append(f'<circle cx="{px(delta):.1f}" cy="{y:.1f}" r="4.3" fill="{color}" stroke="white" stroke-width="1.1"/>')
            parts.append(_text(1035, y + 4, f"{delta:+.3f}", 10, color=color))

    counts = payload["counts"]
    nvalid = payload["n_primary_complete"]
    if nvalid:
        direction = (
            f"Direction counts among complete pairs (n={nvalid}/{payload['n_planned']} planned): "
            f"ΔC>0 {counts['higher']}/{nvalid} ({100 * counts['higher'] / nvalid:.1f}%), "
            f"ΔC<0 {counts['lower']}/{nvalid} ({100 * counts['lower'] / nvalid:.1f}%), "
            f"exact ΔC=0 {counts['zero']}/{nvalid} ({100 * counts['zero'] / nvalid:.1f}%); "
            f"missing {counts['missing']}/{payload['n_planned']} planned. Direction only."
        )
    else:
        direction = f"No complete primary pairs; missing {counts['missing']}/{payload['n_planned']} planned speakers."
    parts.append(_text(42, ptop + payload["n_planned"] * pstep + 14, direction, 11, color="#303942"))
    if payload["primary_status"] == "COMPUTABLE" and payload["primary_ci"] is not None:
        lo, hi = payload["primary_ci"]
        ptext = f"Speaker-bootstrap percentile 95% CI for mean ΔC: {payload['primary_mean']:+.3f} [{lo:+.3f}, {hi:+.3f}] · n={nvalid}/{payload['n_planned']}"
    else:
        ptext = f"Primary mean CI withheld by frozen analyzer · n={nvalid}/{payload['n_planned']}"
    parts.append(_text(42, ptop + payload["n_planned"] * pstep + 36, ptext, 11, color="#303942"))

    # Panel 2: display all five planned levels. Only the four nonzero levels
    # have intervals, copied unchanged from the frozen simultaneous family.
    dtop = 635.0
    dstep = 47.0
    all_dose_values = [
        float(item["delta"])
        for group in payload["dose_values"].values()
        for item in group
    ]
    for entry in payload["dose_intervals"].values():
        all_dose_values.extend(entry["ci"])
        all_dose_values.append(entry["mean"])
    dxmin, dxmax = _range([0.0, *all_dose_values])
    def dx(value: float) -> float:
        return left + (value - dxmin) / (dxmax - dxmin) * plot_w
    dose_plot_bottom = dtop + (len(DOSE_ROWS) - 1) * dstep
    parts.append(_text(42, 572, "2 · Five planned evaluation levels relative to RAW_R1", 17, weight="bold"))
    parts.append(_text(42, 594, "Four nonzero-dose simultaneous 95% CIs are reused from the frozen analyzer; gray dots are speaker contrasts.", 12, color="#555"))
    parts.append(_text(1035, 594, "mean [simultaneous CI]", 10, color="#555"))
    dose_zero_x = dx(0.0)
    parts.append(_line(dose_zero_x, dtop - 11, dose_zero_x, dose_plot_bottom + 12, "#4b5563", 1.5, "5,4"))
    for tick in [dxmin + (dxmax - dxmin) * i / 5 for i in range(6)]:
        xx = dx(tick)
        parts.append(_line(xx, dtop - 10, xx, dose_plot_bottom + 11, "#edf0f2", 0.8))
        parts.append(_text(xx, dtop - 15, f"{tick:+.2f}", 10, "middle", "#666"))
    dose_colors = {"DOSE_M6": "#397f64", "DOSE_M3": "#a77b25", "DOSE_P3": "#9f4e62", "DOSE_P6": "#573e8f"}
    for row_i, (label, csv_field) in enumerate(DOSE_ROWS):
        y = dtop + row_i * dstep
        parts.append(_text(42, y + 4, label, 12, color="#39434c"))
        if csv_field is None:
            parts.append(f'<circle cx="{dose_zero_x:.1f}" cy="{y:.1f}" r="5.0" fill="#222" stroke="white" stroke-width="1.2"/>')
            parts.append(_text(1035, y + 4, "exact zero reference · no CI", 10, color="#555"))
            continue
        dose_id = csv_field.removesuffix("_dose_delta_C")
        values = payload["dose_values"].get(dose_id, [])
        n_values = len(values)
        color = dose_colors[dose_id]
        for i, item in enumerate(values):
            jitter = (i - (n_values - 1) / 2) * 1.15
            parts.append(f'<circle cx="{dx(float(item["delta"])):.1f}" cy="{y + jitter:.1f}" r="2.4" fill="#8b949c" fill-opacity="0.65"/>')
        interval = payload["dose_intervals"].get(dose_id)
        if interval is None:
            parts.append(_text(1035, y + 4, f"CI withheld · n={n_values}/{payload['n_planned']}", 10, color="#737d85"))
        else:
            lo, hi = interval["ci"]
            mean = interval["mean"]
            parts.append(_line(dx(lo), y, dx(hi), y, color, 2.4))
            parts.append(_line(dx(lo), y - 6, dx(lo), y + 6, color, 1.4))
            parts.append(_line(dx(hi), y - 6, dx(hi), y + 6, color, 1.4))
            parts.append(f'<path d="M {dx(mean):.1f} {y - 5:.1f} L {dx(mean) + 5:.1f} {y:.1f} L {dx(mean):.1f} {y + 5:.1f} L {dx(mean) - 5:.1f} {y:.1f} Z" fill="{color}" stroke="white" stroke-width="1"/>')
            parts.append(_text(1035, y + 4, f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}] · n={n_values}/{payload['n_planned']}", 10, color=color))
    parts.append(_text(left + plot_w / 2, dose_plot_bottom + 35, "ΔC = C(dose) − C(RAW_R1); positive values indicate higher Sync-C", 11, "middle", "#444"))

    parts.append(_text(42, 1008, "Evaluation baseline: per frozen protocol, each condition uses the same speaker’s original natural PCM as the evaluation audio.", 11, color="#303942"))
    parts.append(_text(42, 1030, "FIXED’s measured per-sentence RMS dBFS is reported separately in p1_measured_audio_levels.svg / descriptive_metrics.csv.", 11, color="#555"))
    parts.append(_text(42, 1052, "All planned levels are shown; RAW_R1 is the zero reference, not a fifth contrast. No best dose is selected; the response is not forced to be monotonic.", 11, color="#555"))
    parts.append(_text(42, 1086, f"Frozen protocol SHA-256: {payload['protocol_sha256']}   ·   Analyzer input SHA-256: {payload['input_sha256'] or 'not recorded'}", 9, color="#7a8289"))
    parts.append("</svg>")
    return "\n".join(parts)


def render_from_dir(analysis_dir: Path, output_path: Path) -> None:
    summary_path = analysis_dir / "analysis_summary.json"
    rows_path = analysis_dir / "speaker_primary.csv"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("analysis_summary.json root must be an object")
    payload = build_payload(summary, _read_speaker_rows(rows_path))
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"preserving existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_svg(payload), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True, help="existing frozen analyzer output directory")
    parser.add_argument("--output", type=Path, required=True, help="new SVG path; existing files are preserved")
    args = parser.parse_args(argv)
    render_from_dir(args.analysis_dir, args.output)
    print(json.dumps({"output": str(args.output), "format": "svg", "source": str(args.analysis_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
