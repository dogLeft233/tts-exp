"""Summarize frozen local-clock crossover deliveries without model/API calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROLES = ("N1", "N2", "T1", "T2")
REFERENCES = ("N1", "T1")
BOOTSTRAP_SEED = 20261016
BOOTSTRAP_REPLICATES = 2000

ROW_FIELDS = (
    "run_id", "speaker", "reference_role", "source_role", "within_cross",
    "pair_status", "local_map_status", "clock_status", "clock_qualified",
    "speaker_extension_eligible", "endpoint_affine_status", "local_status",
    "affine_reference_coverage", "affine_actual_coverage", "affine_clock_mae_s",
    "affine_clock_median_abs_error_s", "local_reference_coverage", "local_actual_coverage",
    "local_clock_mae_s", "local_clock_median_abs_error_s", "local_slope_count",
    "local_slope_min", "local_slope_max", "primary_index", "primary_lag_frames",
    "affine_primary_index", "local_primary_index", "fixed_primary_index_match",
    "raw_status", "raw_distance_path", "raw_fixedlag_D", "raw_C", "affine_fixedlag_D", "affine_C",
    "local_fixedlag_D", "local_C", "D_gain_affine_minus_local", "C_gain_local_minus_affine",
    "D_computable", "C_computable", "strict_clock_qualified",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact_path(raw: Any, repo_root: Path, matrix_dir: Path) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    direct = Path(raw)
    if direct.is_file():
        return direct.resolve()
    normalized = raw.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    for marker in ("runs", "data"):
        if marker in parts:
            candidate = repo_root.joinpath(*parts[parts.index(marker):])
            if candidate.is_file():
                return candidate.resolve()
    for candidate in (repo_root / direct, matrix_dir / direct):
        if candidate.is_file():
            return candidate.resolve()
    return None


def pair_category(reference: str, source: str) -> str:
    if reference not in ROLES or source not in ROLES:
        return "unknown"
    return "within" if reference[0] == source[0] else "cross"


def raw_pair_values(
    speaker: str,
    reference: str,
    source: str,
    matrix_by_speaker: dict[str, Any],
    repo_root: Path,
    matrix_dir: Path,
) -> dict[str, Any]:
    speaker_doc = matrix_by_speaker.get(speaker)
    if not isinstance(speaker_doc, dict):
        return {"status": "MISSING_SPEAKER"}
    cells = speaker_doc.get("cells", {})
    self_cell = cells.get(f"V_{reference}_A_{reference}") if isinstance(cells, dict) else None
    cross_cell = cells.get(f"V_{reference}_A_{source}") if isinstance(cells, dict) else None
    if not isinstance(self_cell, dict):
        return {"status": "MISSING_REFERENCE_SELF_CELL"}
    primary: int | None = None
    primary_lag: int | None = None
    try:
        primary = int(self_cell["argmin_index"])
        if not 0 <= primary < 31:
            raise ValueError("frozen self-cell primary index is outside 0..30")
        lag_frames = self_cell.get("lag_frames", [])
        if isinstance(lag_frames, list) and len(lag_frames) == 31:
            primary_lag = int(lag_frames[primary])
    except Exception as exc:
        return {"status": f"FAILED_{type(exc).__name__}", "error": str(exc)}
    index_record = {"primary_index": primary, "primary_lag_frames": primary_lag}
    if not isinstance(cross_cell, dict):
        return {"status": "MISSING_RAW_CROSS_CELL", **index_record}
    try:
        support = speaker_doc.get("support", {})
        q0 = int(support["query_start_window_inclusive"])
        q1 = int(support["query_end_window_exclusive"])
        if q0 < 0 or q1 <= q0:
            raise ValueError("invalid report44 shared support interval")
        path = resolve_artifact_path(cross_cell.get("distance_path"), repo_root, matrix_dir)
        if path is None:
            curve = cross_cell.get("curve")
            metrics = cross_cell.get("metrics", {})
            if curve is None and isinstance(metrics, dict):
                curve = metrics.get("curve")
            if isinstance(curve, list) and len(curve) == 31:
                values = np.asarray(curve, dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError("raw cross-cell curve contains nonfinite values")
                raw_d = float(values[primary])
                raw_c = float(np.median(values) - np.min(values))
                return {"status": "PASS_CURVE", **index_record, "raw_D": raw_d, "raw_C": raw_c}
            raise FileNotFoundError("raw cross-cell distance artifact is not locally available")
        expected_sha = cross_cell.get("distance_sha256")
        if isinstance(expected_sha, str) and expected_sha and sha256_file(path) != expected_sha:
            raise ValueError("raw cross-cell distance artifact hash mismatch")
        matrix = np.load(path, allow_pickle=False)
        if matrix.ndim != 2 or matrix.shape[1] != 31 or not np.isfinite(matrix).all():
            raise ValueError(f"raw cross-cell distance matrix has invalid geometry: {matrix.shape}")
        if matrix.shape[0] == q1 - q0:
            rows = matrix
        elif matrix.shape[0] >= q1:
            rows = matrix[q0:q1]
        else:
            raise ValueError("raw cross-cell matrix does not cover the shared q support")
        curve_values = np.mean(rows, axis=0, dtype=np.float64)
        return {
            "status": "PASS_MATRIX", **index_record,
            "raw_D": float(np.mean(rows[:, primary], dtype=np.float64)),
            "raw_C": float(np.median(curve_values) - np.min(curve_values)),
            "distance_path": str(path),
        }
    except Exception as exc:
        return {"status": f"FAILED_{type(exc).__name__}", **index_record, "error": str(exc)}


def score_values(pair_output: Any) -> tuple[str, float | None, float | None, int | None]:
    if not isinstance(pair_output, dict):
        return "MISSING_OUTPUT", None, None, None
    output_status = str(pair_output.get("status", "MISSING_STATUS"))
    score = pair_output.get("score")
    if not isinstance(score, dict) or score.get("status") != "PASS":
        return output_status, None, None, None
    metrics = score.get("metrics", {})
    if not isinstance(metrics, dict):
        return output_status, None, None, None
    index = metrics.get("primary_index_frozen_from_report44")
    try:
        primary_index = int(index) if index is not None else None
    except (TypeError, ValueError):
        primary_index = None
    return output_status, as_float(metrics.get("primary_fixedlag_distance")), as_float(metrics.get("C")), primary_index


def coverage_values(pair_output: Any, prefix: str) -> dict[str, Any]:
    output = pair_output if isinstance(pair_output, dict) else {}
    check = output.get("clock_check", {})
    if not isinstance(check, dict):
        check = {}
    return {
        f"{prefix}_reference_coverage": as_float(check.get("reference_coverage")),
        f"{prefix}_actual_coverage": as_float(check.get("actual_coverage")),
        f"{prefix}_clock_mae_s": as_float(check.get("mae_s")),
        f"{prefix}_clock_median_abs_error_s": as_float(check.get("median_abs_error_s")),
    }


def slope_range(pair: dict[str, Any]) -> tuple[int, float | None, float | None]:
    local_map = pair.get("local_map", {})
    if not isinstance(local_map, dict):
        local_map = {}
    slopes = local_map.get("segment_slopes", [])
    if not isinstance(slopes, list):
        slopes = []
    finite = [number for number in (as_float(item) for item in slopes) if number is not None]
    low = min(finite) if finite else as_float(local_map.get("slope_min"))
    high = max(finite) if finite else as_float(local_map.get("slope_max"))
    return len(finite), low, high


def make_row(
    run_id: str,
    speaker: str,
    reference: str,
    source: str,
    pair: dict[str, Any] | None,
    speaker_eligible: Any,
    raw: dict[str, Any],
) -> dict[str, Any]:
    pair = pair or {}
    outputs = pair.get("outputs", {})
    if not isinstance(outputs, dict):
        outputs = {}
    affine = outputs.get("endpoint_affine", {})
    local = outputs.get("local", {})
    affine_status, affine_d, affine_c, affine_index = score_values(affine)
    local_status, local_d, local_c, local_index = score_values(local)
    clock = pair.get("clock_comparison", {})
    if not isinstance(clock, dict):
        clock = {}
    clock_status = str(clock.get("status", "MISSING_PAIR_ROW" if not pair else "MISSING_CLOCK_STATUS"))
    qualified = clock_status == "PASS"
    slope_count, slope_min, slope_max = slope_range(pair)
    raw_d, raw_c = as_float(raw.get("raw_D")), as_float(raw.get("raw_C"))
    expected_index = raw.get("primary_index")
    fixed_index_match = (
        isinstance(expected_index, int) and affine_index == expected_index and local_index == expected_index
    ) if affine_d is not None and local_d is not None else None
    d_computable = affine_d is not None and local_d is not None and fixed_index_match is True
    c_computable = affine_c is not None and local_c is not None
    row = {
        "run_id": run_id, "speaker": speaker, "reference_role": reference, "source_role": source,
        "within_cross": pair_category(reference, source), "pair_status": str(pair.get("status", "MISSING_PAIR_ROW")),
        "local_map_status": str(pair.get("local_map", {}).get("status", "MISSING")) if isinstance(pair.get("local_map"), dict) else "MISSING",
        "clock_status": clock_status, "clock_qualified": qualified,
        "speaker_extension_eligible": speaker_eligible if isinstance(speaker_eligible, bool) else None,
        "endpoint_affine_status": affine_status, "local_status": local_status,
        **coverage_values(affine, "affine"), **coverage_values(local, "local"),
        "local_slope_count": slope_count, "local_slope_min": slope_min, "local_slope_max": slope_max,
        "primary_index": raw.get("primary_index"), "primary_lag_frames": raw.get("primary_lag_frames"),
        "affine_primary_index": affine_index, "local_primary_index": local_index,
        "fixed_primary_index_match": fixed_index_match,
        "raw_status": str(raw.get("status", "NOT_COMPUTED")), "raw_distance_path": raw.get("distance_path"),
        "raw_fixedlag_D": raw_d, "raw_C": raw_c,
        "affine_fixedlag_D": affine_d, "affine_C": affine_c, "local_fixedlag_D": local_d, "local_C": local_c,
        "D_gain_affine_minus_local": affine_d - local_d if d_computable else None,
        "C_gain_local_minus_affine": local_c - affine_c if c_computable else None,
        "D_computable": d_computable, "C_computable": c_computable,
        "strict_clock_qualified": qualified and d_computable and c_computable,
    }
    return row


def speaker_sort_key(value: str) -> tuple[str, int | str]:
    prefix = value.rstrip("0123456789")
    suffix = value[len(prefix):]
    return prefix, int(suffix) if suffix.isdigit() else suffix


def read_run(run_argument: Path) -> tuple[str, dict[str, Any], dict[str, Any], Path]:
    path = run_argument.resolve()
    receipt_path = path if path.name == "main_receipt.json" else path / "main_receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"main receipt not found: {receipt_path}")
    run_dir = receipt_path.parent
    technical_path = run_dir / "main_technical_summary.json"
    if not technical_path.is_file():
        raise FileNotFoundError(f"main technical summary not found: {technical_path}")
    run_id = run_dir.parent.name or run_dir.name
    return run_id, read_json(receipt_path), read_json(technical_path), run_dir


def expected_pair_keys() -> list[tuple[str, str]]:
    return [(reference, source) for reference in REFERENCES for source in ROLES if source != reference]


def build_rows(
    run_arguments: Iterable[Path],
    matrix_scores_path: Path,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrix_doc = read_json(matrix_scores_path)
    matrix_by = {str(row["speaker"]): row for row in matrix_doc.get("speaker_results", []) if isinstance(row, dict) and "speaker" in row}
    all_rows: list[dict[str, Any]] = []
    run_metadata: list[dict[str, Any]] = []
    seen_speakers: dict[str, str] = {}
    for run_argument in run_arguments:
        run_id, receipt, technical, run_dir = read_run(run_argument)
        outputs = receipt.get("outputs", {})
        tech_speakers = technical.get("speaker_results", {})
        if not isinstance(outputs, dict) or not isinstance(tech_speakers, dict):
            raise ValueError(f"invalid main output/technical-summary schema for {run_dir}")
        speakers = sorted(set(map(str, outputs)) | set(map(str, tech_speakers)), key=speaker_sort_key)
        for speaker in speakers:
            pairs = outputs.get(speaker, [])
            by_key: dict[tuple[str, str], dict[str, Any]] = {}
            if isinstance(pairs, list):
                for pair in pairs:
                    if not isinstance(pair, dict):
                        continue
                    key = (str(pair.get("reference_role", "")), str(pair.get("source_role", "")))
                    if key in by_key:
                        raise ValueError(f"duplicate pair in {run_dir}: {speaker}/{key}")
                    by_key[key] = pair
            if speaker in seen_speakers:
                raise ValueError(f"source speaker appears in more than one input run: {speaker} ({seen_speakers[speaker]}, {run_id})")
            seen_speakers[speaker] = run_id
            speaker_technical = tech_speakers.get(speaker, {})
            eligible = speaker_technical.get("eligible_for_extension") if isinstance(speaker_technical, dict) else None
            for reference, source in expected_pair_keys():
                pair = by_key.get((reference, source))
                raw = raw_pair_values(speaker, reference, source, matrix_by, repo_root, matrix_scores_path.parent)
                all_rows.append(make_row(run_id, speaker, reference, source, pair, eligible, raw))
            for (reference, source), pair in sorted(by_key.items()):
                if (reference, source) in expected_pair_keys():
                    continue
                raw = raw_pair_values(speaker, reference, source, matrix_by, repo_root, matrix_scores_path.parent)
                all_rows.append(make_row(run_id, speaker, reference, source, pair, eligible, raw))
        run_metadata.append({
            "run_id": run_id, "run_dir": str(run_dir), "main_receipt": str(run_dir / "main_receipt.json"),
            "technical_summary": str(run_dir / "main_technical_summary.json"),
            "extension_gate_pass": technical.get("extension_gate_pass"),
            "extension_eligible_speakers": technical.get("extension_eligible_speakers"),
            "receipt_audio_job_count": receipt.get("audio_job_count"),
            "speaker_ids": speakers,
        })
    all_rows.sort(key=lambda row: (speaker_sort_key(row["speaker"]), row["reference_role"], row["source_role"], row["run_id"]))
    return all_rows, run_metadata


def expected_sources(reference: str, relation: str) -> list[str]:
    return [role for role in ROLES if role != reference and pair_category(reference, role) == relation]


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires values")
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def bootstrap_ci(values: list[float]) -> dict[str, Any]:
    if len(values) != 8:
        return {"status": "NOT_ESTIMATED_N_NOT_8", "ci_95": None}
    rng = random.Random(BOOTSTRAP_SEED)
    means = [statistics.mean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP_REPLICATES)]
    means.sort()
    return {
        "status": "PASS_ORDINARY_PERCENTILE_BOOTSTRAP", "method": "source-speaker resampling with replacement",
        "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES,
        "ci_95": [percentile(means, 0.025), percentile(means, 0.975)],
    }


def group_summaries(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = sorted({row["speaker"] for row in rows}, key=speaker_sort_key)
    values_out: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for population in ("all_computable", "strict_clock_qualified"):
        for reference in REFERENCES:
            for relation in ("within", "cross"):
                expected_roles = expected_sources(reference, relation)
                for endpoint, field in (("D_gain", "D_gain_affine_minus_local"), ("C_gain", "C_gain_local_minus_affine")):
                    per_source: list[dict[str, Any]] = []
                    for speaker in sources:
                        source_rows = [r for r in rows if r["speaker"] == speaker and r["reference_role"] == reference and r["source_role"] in expected_roles]
                        by_role = {r["source_role"]: r for r in source_rows}
                        used: list[float] = []
                        missing_roles: list[str] = []
                        for role in expected_roles:
                            item = by_role.get(role)
                            value = as_float(item.get(field)) if item else None
                            if population == "strict_clock_qualified" and item and item.get("clock_status") != "PASS":
                                value = None
                            if value is None:
                                missing_roles.append(role)
                            else:
                                used.append(value)
                        complete = not missing_roles and len(used) == len(expected_roles)
                        source_value = statistics.mean(used) if complete else None
                        item = {
                            "population": population, "endpoint": endpoint, "reference_role": reference,
                            "within_cross": relation, "speaker": speaker, "value": source_value,
                            "complete": complete, "expected_source_roles": expected_roles,
                            "missing_source_roles": missing_roles,
                        }
                        per_source.append(item)
                        values_out.append(item)
                    complete_values = [float(item["value"]) for item in per_source if item["complete"]]
                    fully_complete = len(complete_values) == len(sources) and len(sources) in (4, 8)
                    observed_mean = statistics.mean(complete_values) if complete_values else None
                    all_source_mean = observed_mean if fully_complete else None
                    missing_sources = [item["speaker"] for item in per_source if not item["complete"]]
                    if len(sources) == 8 and fully_complete:
                        ci = bootstrap_ci(complete_values)
                    else:
                        reason = "NOT_ESTIMATED_N_NOT_8" if len(sources) != 8 else "NOT_ESTIMATED_INCOMPLETE_BALANCE"
                        ci = {"status": reason, "ci_95": None}
                    summaries.append({
                        "population": population, "endpoint": endpoint, "reference_role": reference,
                        "within_cross": relation, "expected_source_count": len(sources),
                        "complete_source_count": len(complete_values), "complete": fully_complete,
                        "mean": all_source_mean, "observed_complete_source_mean": observed_mean,
                        "source_values": per_source, "missing_source_ids": missing_sources,
                        "bootstrap_95": ci,
                    })
    return values_out, summaries


def overall_source_values(summaries: list[dict[str, Any]], endpoint: str, population: str) -> list[dict[str, Any]]:
    selected = [s for s in summaries if s["endpoint"] == endpoint and s["population"] == population]
    source_ids = sorted({v["speaker"] for s in selected for v in s["source_values"]}, key=speaker_sort_key)
    result = []
    for speaker in source_ids:
        cells = [
            v for s in selected for v in s["source_values"]
            if v["speaker"] == speaker
        ]
        complete = len(cells) == 4 and all(item["complete"] for item in cells)
        vals = [float(item["value"]) for item in cells if item["complete"]]
        result.append({"speaker": speaker, "value": statistics.mean(vals) if complete else None, "complete": complete})
    return result


def svg_panel(title: str, all_values: list[dict[str, Any]], strict_values: list[dict[str, Any]], x: int, width: int, height: int) -> str:
    left, right, top, bottom = x + 62, x + width - 18, 62, height - 62
    plotted = [as_float(v.get("value")) for v in all_values + strict_values]
    nums = [v for v in plotted if v is not None]
    if not nums:
        return f'<g><text x="{x + width / 2:.1f}" y="32" text-anchor="middle" class="title">{title}</text><text x="{x + width / 2:.1f}" y="{height / 2:.1f}" text-anchor="middle">No computable gains</text></g>'
    low = min(0.0, min(nums)); high = max(0.0, max(nums))
    span = max(high - low, 0.02)
    low -= span * 0.12; high += span * 0.12
    ids = [item["speaker"] for item in all_values]
    count = max(1, len(ids))
    x_positions = {speaker: (left + (right - left) * (i + 0.5) / count) for i, speaker in enumerate(ids)}

    def y_pos(value: float) -> float:
        return bottom - (value - low) * (bottom - top) / (high - low)

    pieces = [f'<g><text x="{x + width / 2:.1f}" y="30" text-anchor="middle" class="title">{title}</text>']
    for tick in range(5):
        value = low + (high - low) * tick / 4
        yy = y_pos(value)
        pieces.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" class="grid"/>')
        pieces.append(f'<text x="{left - 7}" y="{yy + 4:.1f}" text-anchor="end" class="tick">{value:.3f}</text>')
    if low <= 0 <= high:
        yy = y_pos(0.0)
        pieces.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{right}" y2="{yy:.1f}" class="zero"/>')
    all_by = {item["speaker"]: as_float(item.get("value")) for item in all_values}
    strict_by = {item["speaker"]: as_float(item.get("value")) for item in strict_values}
    for speaker in ids:
        xx = x_positions[speaker]
        a, s = all_by.get(speaker), strict_by.get(speaker)
        if a is not None and s is not None:
            pieces.append(f'<line x1="{xx:.1f}" y1="{y_pos(a):.1f}" x2="{xx:.1f}" y2="{y_pos(s):.1f}" class="pair"/>')
        if a is not None:
            pieces.append(f'<circle cx="{xx:.1f}" cy="{y_pos(a):.1f}" r="5" class="all"/>')
        if s is not None:
            pieces.append(f'<rect x="{xx - 4:.1f}" y="{y_pos(s) - 4:.1f}" width="8" height="8" class="strict"/>')
        pieces.append(f'<text x="{xx:.1f}" y="{bottom + 22}" text-anchor="middle" class="tick">{speaker}</text>')
    pieces.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/><line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')
    pieces.append("</g>")
    return "".join(pieces)


def write_svg(path: Path, summaries: list[dict[str, Any]]) -> None:
    d_all = overall_source_values(summaries, "D_gain", "all_computable")
    d_strict = overall_source_values(summaries, "D_gain", "strict_clock_qualified")
    c_all = overall_source_values(summaries, "C_gain", "all_computable")
    c_strict = overall_source_values(summaries, "C_gain", "strict_clock_qualified")
    width, height = 1040, 470
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    svg.append('<style>text{font:13px Arial,sans-serif;fill:#202124}.title{font-size:16px;font-weight:bold}.tick{font-size:11px}.grid{stroke:#e3e6ea;stroke-width:1}.axis{stroke:#53565a;stroke-width:1}.zero{stroke:#777;stroke-width:1.3;stroke-dasharray:5 4}.pair{stroke:#98a0aa;stroke-width:1.2}.all{fill:#1769aa}.strict{fill:#e07022}</style>')
    svg.append(svg_panel("D gain = affine fixed-lag D − local fixed-lag D", d_all, d_strict, 0, 510, height))
    svg.append(svg_panel("C gain = local C − affine C", c_all, c_strict, 510, 510, height))
    svg.append('<circle cx="390" cy="452" r="5" class="all"/><text x="402" y="456">all computable</text><rect x="530" y="448" width="8" height="8" class="strict"/><text x="544" y="456">strict clock-qualified</text></svg>')
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def analyze(args: argparse.Namespace) -> None:
    matrix_path = args.matrix_scores.resolve(strict=True)
    repo_root = args.repo_root.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, run_metadata = build_rows(args.main_run, matrix_path, repo_root)
    source_values, summaries = group_summaries(rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["clock_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "schema_version": 1,
        "interpretation": "The local condition combines a phone-center MFA time map with pitch-preserving waveform phase-vocoder rendering; the cached visual/video input is fixed. This does not isolate a pure clock intervention.",
        "primary_metrics": {"D_gain": "affine fixed-lag D minus local fixed-lag D (positive favors local)", "C_gain": "local C minus affine C (positive favors local)"},
        "fixed_lag_policy": "Use the original report44 reference self-cell argmin index for raw, affine, and local D; C is the all-lag median curve distance minus its minimum on the same common q support.",
        "source_equalization_policy": "For each N1/T1 reference by within/cross stratum, average the expected source-role rows equally within each speaker, then report the mean across independent source speakers. A group mean is null unless every source speaker is complete; observed complete-source means and missing IDs are reported separately.",
        "inputs": {"matrix_scores": str(matrix_path), "matrix_scores_sha256": sha256_file(matrix_path), "runs": run_metadata},
        "counts": {
            "rows": len(rows),
            "D_computable": sum(bool(row["D_computable"]) for row in rows),
            "C_computable": sum(bool(row["C_computable"]) for row in rows),
            "strict_clock_qualified": sum(bool(row["strict_clock_qualified"]) for row in rows),
            "clock_status": status_counts,
            "raw_computable": sum(row["raw_fixedlag_D"] is not None and row["raw_C"] is not None for row in rows),
        },
        "source_equal_group_summaries": summaries,
        "bootstrap_policy": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "interval": "ordinary percentile 95%", "only_when": "exactly 8 unique source speakers and all 8 have complete balanced values for that population, endpoint, reference, and within/cross group"},
        "source_equal_values": source_values,
        "plot_values": {
            endpoint: {
                population: overall_source_values(summaries, endpoint, population)
                for population in ("all_computable", "strict_clock_qualified")
            }
            for endpoint in ("D_gain", "C_gain")
        },
    }
    (output_dir / "delivery_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "delivery_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "delivery_rows.csv", rows, ROW_FIELDS)
    write_csv(output_dir / "source_equal_values.csv", source_values, ("population", "endpoint", "reference_role", "within_cross", "speaker", "value", "complete", "expected_source_roles", "missing_source_roles"))
    write_csv(output_dir / "source_equal_summary.csv", summaries, ("population", "endpoint", "reference_role", "within_cross", "expected_source_count", "complete_source_count", "complete", "mean", "observed_complete_source_mean", "missing_source_ids", "bootstrap_95"))
    write_svg(output_dir / "paired_gains.svg", summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run", action="append", type=Path, required=True, help="main stage directory or main_receipt.json; repeat for additional source runs")
    parser.add_argument("--matrix-scores", type=Path, required=True, help="report44 matrix_scores.json")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, required=True, help="new or empty directory for analysis artifacts")
    args = parser.parse_args()
    analyze(args)


if __name__ == "__main__":
    main()
