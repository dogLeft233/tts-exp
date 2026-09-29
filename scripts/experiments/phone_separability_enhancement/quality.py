"""Independent timing/PCM/content quality contracts for candidate outputs."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.phone_separability_mechanism.features import match_occurrences


def _iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union if union > 0 else 0.0


def pause_spans(tokens: Sequence[Mapping[str, Any]], *, min_ms: float = 50.0) -> list[tuple[float, float]]:
    return [(float(token["start_s"]), float(token["end_s"])) for token in tokens if not bool(token.get("speech", not token.get("silence", False))) and (float(token["end_s"]) - float(token["start_s"])) * 1000.0 >= float(min_ms)]


def one_to_one_pause_match(source: Sequence[tuple[float, float]], target: Sequence[tuple[float, float]], *, iou_min: float = 0.5, edge_error_ms: float = 20.0) -> list[dict[str, Any]]:
    candidates = sorted(((float(_iou(left, right)), i, j) for i, left in enumerate(source) for j, right in enumerate(target)), reverse=True)
    used_source: set[int] = set()
    used_target: set[int] = set()
    matches: list[dict[str, Any]] = []
    for iou, i, j in candidates:
        if i in used_source or j in used_target:
            continue
        used_source.add(i)
        used_target.add(j)
        left, right = source[i], target[j]
        start_error = abs(left[0] - right[0]) * 1000.0
        end_error = abs(left[1] - right[1]) * 1000.0
        matches.append({"source_index": i, "target_index": j, "iou": iou, "start_error_ms": start_error, "end_error_ms": end_error, "pass": bool(iou >= iou_min and start_error <= edge_error_ms and end_error <= edge_error_ms)})
    matches.sort(key=lambda row: row["source_index"])
    matches.extend({"source_index": i, "target_index": None, "iou": 0.0, "start_error_ms": None, "end_error_ms": None, "pass": False, "missing": True} for i in range(len(source)) if i not in used_source)
    matches.extend({"source_index": None, "target_index": j, "iou": 0.0, "start_error_ms": None, "end_error_ms": None, "pass": False, "new": True} for j in range(len(target)) if j not in used_target)
    return matches


def audit_timing(source_tokens: Sequence[Mapping[str, Any]], candidate_tokens: Sequence[Mapping[str, Any]], *, pause_min_ms: float = 50.0, pause_iou_min: float = 0.5, edge_error_ms: float = 20.0, max_edit_rate: float = 0.05, min_coverage: float = 0.90, median_error_ms: float = 20.0, p95_error_ms: float = 40.0) -> dict[str, Any]:
    matching = match_occurrences(source_tokens, candidate_tokens, speech_only=True)
    boundary_errors: list[float] = []
    rows = []
    for pair in matching.get("matches", []):
        source = source_tokens[int(pair["source_index"])]
        candidate = candidate_tokens[int(pair["target_index"])]
        start_error = abs(float(source["start_s"]) - float(candidate["start_s"])) * 1000.0
        end_error = abs(float(source["end_s"]) - float(candidate["end_s"])) * 1000.0
        boundary_errors.extend((start_error, end_error))
        rows.append({"source_index": pair["source_index"], "target_index": pair["target_index"], "label": pair.get("label"), "start_error_ms": start_error, "end_error_ms": end_error})
    source_pauses = pause_spans(source_tokens, min_ms=pause_min_ms)
    target_pauses = pause_spans(candidate_tokens, min_ms=pause_min_ms)
    pause_rows = one_to_one_pause_match(source_pauses, target_pauses, iou_min=pause_iou_min, edge_error_ms=edge_error_ms)
    median = float(np.median(boundary_errors)) if boundary_errors else None
    p95 = float(np.quantile(boundary_errors, 0.95)) if boundary_errors else None
    coverage = float(matching.get("matched_count", 0)) / max(int(matching.get("source_count", 0)), 1)
    passed = bool(float(matching.get("edit_rate", 1.0)) <= max_edit_rate and coverage >= min_coverage and median is not None and p95 is not None and median <= median_error_ms and p95 <= p95_error_ms and all(bool(row["pass"]) for row in pause_rows))
    return {"edit_rate": float(matching.get("edit_rate", 1.0)), "occurrence_coverage": coverage, "matched_count": int(matching.get("matched_count", 0)), "source_count": int(matching.get("source_count", 0)), "ambiguous": bool(matching.get("ambiguous", False)), "boundary_errors_ms": rows, "boundary_median_ms": median, "boundary_p95_ms": p95, "pauses": pause_rows, "timing_pass": passed}


def timing_calibration(natural_tokens: Sequence[Mapping[str, Any]], repeat_tokens: Sequence[Mapping[str, Any]] | None, global_shift_tokens: Sequence[Mapping[str, Any]] | None, local_shift_tokens: Sequence[Mapping[str, Any]] | None, *, expected_sentences: int = 8) -> dict[str, Any]:
    repeat = audit_timing(natural_tokens, repeat_tokens) if repeat_tokens is not None else None
    global_shift = audit_timing(natural_tokens, global_shift_tokens) if global_shift_tokens is not None else None
    local_shift = audit_timing(natural_tokens, local_shift_tokens) if local_shift_tokens is not None else None
    calibrated = bool(repeat and repeat["timing_pass"] and global_shift and not global_shift["timing_pass"] and local_shift and not local_shift["timing_pass"])
    return {"repeat": repeat, "global_shift": global_shift, "local_shift": local_shift, "expected_sentences": int(expected_sentences), "status": "CALIBRATED" if calibrated else "TIMING_MEASUREMENT_UNCALIBRATED"}


def pcm_contract(source: np.ndarray, candidate: np.ndarray, protected: np.ndarray, *, max_residual_ratio: float = 0.01, min_snr_db: float = 20.0, max_rms_change_db: float = 1.0) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.int16).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.int16).reshape(-1)
    protected = np.asarray(protected, dtype=bool).reshape(-1)
    if source.shape != candidate.shape or source.shape != protected.shape:
        raise ValueError("PCM contract arrays have different shapes")
    editable = ~protected
    residual = candidate.astype(np.float64) - source.astype(np.float64)
    signal = source.astype(np.float64)
    ratio = float(np.sum(residual[editable] ** 2) / max(np.sum(signal[editable] ** 2), 1e-8)) if np.any(editable) else 0.0
    snr = float(10 * np.log10(max(np.sum(signal[editable] ** 2), 1e-8) / max(np.sum(residual[editable] ** 2), 1e-8))) if np.any(editable) else float("inf")
    rms_source = np.sqrt(np.mean(signal[editable] ** 2)) if np.any(editable) else 0.0
    rms_candidate = np.sqrt(np.mean(candidate.astype(np.float64)[editable] ** 2)) if np.any(editable) else 0.0
    rms_change = float(20 * np.log10(max(rms_candidate, 1e-12) / max(rms_source, 1e-12))) if np.any(editable) else 0.0
    return {"same_length": True, "protected_pcm_equal": bool(np.array_equal(source[protected], candidate[protected])), "residual_energy_ratio": ratio, "snr_db": snr, "rms_change_db": rms_change, "new_saturated_samples": int(np.count_nonzero((np.abs(candidate) >= 32767) & (np.abs(source) < 32767))), "pass": bool(np.array_equal(source[protected], candidate[protected]) and ratio <= max_residual_ratio and snr >= min_snr_db and abs(rms_change) <= max_rms_change_db and np.all(np.isfinite(candidate)))}


def blind_pack(rows: Sequence[Mapping[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, row in enumerate(rows):
        source = str(row.get("path", ""))
        alias = hashlib.sha256(f"blind|{index}|{row.get('pair_id')}|{row.get('arm')}".encode()).hexdigest()[:12]
        manifest.append({"alias": alias, "pair_id": row.get("pair_id"), "arm": row.get("arm"), "path": source, "anonymous": True})
    (target / "ratings_template.csv").write_text("alias,clarity,naturalness,content_error,pause_position,notes\n" + "\n".join(f"{row['alias']},,,," for row in manifest) + "\n", encoding="utf-8")
    return {"status": "HUMAN_NOT_ASSESSED", "manifest": manifest, "ratings": str(target / "ratings_template.csv")}


__all__ = ["audit_timing", "blind_pack", "one_to_one_pause_match", "pcm_contract", "pause_spans", "timing_calibration"]
