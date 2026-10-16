"""CPU-only, pre-specified analysis for the 2026-10-16 new-speaker P1 run.

This reads only score_results.json; it never runs a model or reads media. The
statistics are frozen to the BM SPEC: speaker-level FIXED-minus-RAW after
averaging two repeats, plus a four-contrast dose family against RAW_R1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SEED = 20261016
BOOTSTRAP_DRAWS = 20_000
CONDITION_ORDER = (
    "RAW_R1",
    "RAW_R2",
    "FIXED_R1",
    "FIXED_R2",
    "DOSE_M6",
    "DOSE_M3",
    "DOSE_P3",
    "DOSE_P6",
)
DOSE_ORDER = ("DOSE_M6", "DOSE_M3", "DOSE_P3", "DOSE_P6")
LOCKED_SPEAKERS = tuple([f"S{i:04d}" for i in range(102, 112)] + [f"S{i:04d}" for i in range(144, 154)])
SUCCESS_STATUSES = {"PASS", "SCORED", "COMPLETE", "OK"}
METRIC_NAMES = ("C", "B", "Dmin", "D0", "anchor_D", "anchor_C", "search_gain", "bestlag")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    out = [_finite(x) for x in value]
    if any(x is None for x in out):
        return None
    return [float(x) for x in out]


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def _conditions_as_map(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items() if isinstance(v, dict)}
    if isinstance(value, list):
        out: dict[str, dict[str, Any]] = {}
        for row in value:
            if isinstance(row, dict) and row.get("condition_id") is not None:
                key = str(row["condition_id"])
                if key in out:
                    raise ValueError(f"duplicate condition row: {key}")
                out[key] = row
        return out
    return {}


def _speakers_as_map(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items() if isinstance(v, dict)}
    if isinstance(value, list):
        out: dict[str, dict[str, Any]] = {}
        for row in value:
            if isinstance(row, dict) and row.get("speaker_id") is not None:
                key = str(row["speaker_id"])
                if key in out:
                    raise ValueError(f"duplicate speaker row: {key}")
                out[key] = row
        return out
    return {}


def bootstrap_indices(n: int, draws: int = BOOTSTRAP_DRAWS, seed: int = SEED) -> np.ndarray:
    if n < 1:
        raise ValueError("bootstrap requires at least one speaker")
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0, n, size=(draws, n), endpoint=False)


def percentile_mean_interval(values: Iterable[float], indices: np.ndarray) -> dict[str, Any]:
    x = np.asarray(list(values), dtype=np.float64)
    if x.ndim != 1 or len(x) < 1 or not np.isfinite(x).all():
        raise ValueError("values must be a nonempty finite vector")
    means = x[indices].mean(axis=1)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "ci95": [float(v) for v in np.quantile(means, [0.025, 0.975], method="linear")],
        "interval": "speaker-bootstrap percentile; descriptive unless named primary",
    }


def paired_primary_stats(delta: Iterable[float], indices: np.ndarray) -> dict[str, Any]:
    x = np.asarray(list(delta), dtype=np.float64)
    out = percentile_mean_interval(x, indices)
    out.update(
        {
            "contrast": "mean(FIXED_R1,FIXED_R2) - mean(RAW_R1,RAW_R2), per speaker",
            "bootstrap_seed": SEED,
            "bootstrap_draws": int(indices.shape[0]),
            "unit": "speaker",
        }
    )
    return out


def max_t_family_stats(matrix: np.ndarray, labels: list[str], indices: np.ndarray) -> dict[str, Any]:
    """Centered paired bootstrap max-|t| family with fixed original sample SE."""
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(labels) or x.shape[0] < 2:
        raise ValueError("family matrix must be speakers by contrast, with at least two speakers")
    if not np.isfinite(x).all():
        raise ValueError("family matrix must be finite")
    if indices.shape[1] != x.shape[0]:
        raise ValueError("bootstrap speaker count does not match family matrix")
    observed = x.mean(axis=0)
    se = x.std(axis=0, ddof=1) / math.sqrt(x.shape[0])
    boot_means = x[indices].mean(axis=1)
    centered = boot_means - observed[None, :]
    t = np.zeros_like(centered)
    nonzero = se > 0.0
    t[:, nonzero] = centered[:, nonzero] / se[None, nonzero]
    max_abs_t = np.max(np.abs(t), axis=1)
    critical = float(np.quantile(max_abs_t, 0.95, method="linear"))
    contrasts: dict[str, Any] = {}
    for j, label in enumerate(labels):
        contrasts[label] = {
            "n": int(x.shape[0]),
            "mean": float(observed[j]),
            "sample_se": float(se[j]),
            "simultaneous_ci95": [
                float(observed[j] - critical * se[j]),
                float(observed[j] + critical * se[j]),
            ],
            "zero_variance": bool(se[j] == 0.0),
            "zero_variance_policy": "centered bootstrap t is 0; interval degenerates to the observed constant",
        }
    return {
        "status": "COMPUTABLE",
        "unit": "speaker",
        "family": labels,
        "method": "paired speaker bootstrap; centered bootstrap means / original sample SE; max-|t| 95th percentile",
        "critical_value": critical,
        "bootstrap_seed": SEED,
        "bootstrap_draws": int(indices.shape[0]),
        "contrasts": contrasts,
    }


def _score_errors(row: dict[str, Any], primary_q: list[Any], full_q: list[Any] | None = None) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    curve = _float_list(row.get("curve31"))
    lags = _float_list(row.get("lags"))
    q_support = row.get("q_support")
    if curve is None or len(curve) != 31:
        errors.append("curve31 must contain 31 finite values")
    if lags is None or len(lags) != 31:
        errors.append("lags must contain 31 finite values")
    if curve is not None and lags is not None and len(curve) == len(lags) == 31:
        if any(lags[i] >= lags[i + 1] for i in range(30)):
            errors.append("lags must be strictly increasing")
        zero_indices = [i for i, lag in enumerate(lags) if abs(lag) <= 1e-12]
        if len(zero_indices) != 1:
            errors.append("lags must contain exactly one zero position")
    if not isinstance(q_support, list) or q_support != primary_q:
        errors.append("condition q_support must exactly equal speaker primary_guard20_q_indices")
    distance_shape = row.get("distance_shape")
    allowed_q_counts = {len(primary_q)}
    if isinstance(full_q, list) and full_q:
        allowed_q_counts.add(len(full_q))
    if not isinstance(distance_shape, (list, tuple)) or len(distance_shape) != 2:
        errors.append("distance_shape must be [q,31]")
    else:
        try:
            shape = [int(distance_shape[0]), int(distance_shape[1])]
        except (TypeError, ValueError):
            shape = []
        if not shape or shape[1] != 31 or shape[0] not in allowed_q_counts:
            errors.append(f"distance_shape must use a declared common q set and 31 lags; got {distance_shape}")
    vals = {name: _finite(row.get(name)) for name in ("C", "B", "Dmin", "D0")}
    if any(value is None for value in vals.values()):
        errors.append("C/B/Dmin/D0 must be finite")
    anchor = row.get("anchor") if isinstance(row.get("anchor"), dict) else {}
    anchor_d = _finite(anchor.get("D"))
    anchor_c = _finite(anchor.get("C"))
    anchor_bestlag = _finite(anchor.get("bestlag"))
    search_gain = _finite(anchor.get("search_gain"))
    audio_rms_dbfs = _finite(row.get("audio_rms_dbfs"))
    relative_dose_db = _finite(row.get("relative_dose_db"))
    if any(value is None for value in (anchor_d, anchor_c, anchor_bestlag, search_gain)):
        errors.append("anchor must contain finite D, C, bestlag, and search_gain")
    if audio_rms_dbfs is None or relative_dose_db is None:
        errors.append("audio_rms_dbfs and measured relative_dose_db are required")
    for field in ("video_sha256", "visual_sha256", "audio_variant_sha256", "distance_sha256"):
        if not _is_sha256(row.get(field)):
            errors.append(f"{field} must be a SHA-256 digest")
    if not isinstance(row.get("distance_path"), str) or not row.get("distance_path"):
        errors.append("distance_path is required")
    if not isinstance(row.get("engine_metadata"), dict) or not row.get("engine_metadata"):
        errors.append("engine_metadata is required")
    if not isinstance(row.get("lag_frame_convention"), str) or not row.get("lag_frame_convention"):
        errors.append("lag_frame_convention is required")
    if curve is not None and len(curve) == 31 and lags is not None and len(lags) == 31:
        b = float(np.median(curve))
        dmin = float(np.min(curve))
        zero_indices = [i for i, lag in enumerate(lags) if abs(lag) <= 1e-12]
        checks = [("B", b), ("Dmin", dmin), ("C", b - dmin)]
        if len(zero_indices) == 1:
            checks.append(("D0", float(curve[zero_indices[0]])))
        for name, expected in checks:
            if vals[name] is not None and abs(float(vals[name]) - expected) > 1e-6:
                errors.append(f"{name} inconsistent with curve31/lags (error>{1e-6:g})")
        if anchor_d is not None and anchor_c is not None and abs(anchor_c - (b - anchor_d)) > 1e-6:
            errors.append("anchor.C inconsistent with B - anchor.D")
        if search_gain is not None and vals["D0"] is not None and abs(search_gain - (float(vals["D0"]) - dmin)) > 1e-6:
            errors.append("anchor.search_gain inconsistent with D0 - Dmin")
    if errors:
        return None, errors
    return {
        **{name: float(value) for name, value in vals.items()},
        "curve31": curve,
        "lags": lags,
        "anchor_D": float(anchor_d),
        "anchor_C": float(anchor_c),
        "bestlag": float(anchor_bestlag),
        "search_gain": float(search_gain),
        "q_support": list(q_support),
        "audio_rms_dbfs": float(audio_rms_dbfs),
        "relative_dose_db": float(relative_dose_db),
        "motion": row.get("motion") if isinstance(row.get("motion"), dict) else {},
        "encoder": row.get("encoder") if isinstance(row.get("encoder"), dict) else {},
        "video_sha256": row.get("video_sha256"),
        "feature_sha256": row.get("feature_sha256"),
        "visual_sha256": row.get("visual_sha256"),
        "visual_path": row.get("visual_path"),
        "audio_variant_sha256": row.get("audio_variant_sha256"),
        "distance_path": row.get("distance_path"),
        "distance_sha256": row.get("distance_sha256"),
        "distance_shape": row.get("distance_shape"),
        "engine_metadata": row.get("engine_metadata") if isinstance(row.get("engine_metadata"), dict) else {},
        "lag_frame_convention": row.get("lag_frame_convention"),
    }, []


def normalize_manifest(doc: dict[str, Any], *, require_locked_cohort: bool = True) -> dict[str, Any]:
    if not _is_sha256(doc.get("protocol_sha256")):
        raise ValueError("score_results.json must bind a valid protocol_sha256")
    planned_speakers = doc.get("planned_speakers")
    if not isinstance(planned_speakers, list) or not planned_speakers:
        raise ValueError("score_results.json must carry the complete planned_speakers list")
    speaker_ids = [str(s) for s in planned_speakers]
    if len(set(speaker_ids)) != len(speaker_ids):
        raise ValueError("planned_speakers contains duplicate IDs")
    if require_locked_cohort and tuple(speaker_ids) != LOCKED_SPEAKERS:
        raise ValueError(f"planned_speakers must exactly match the frozen 20-speaker order {LOCKED_SPEAKERS}")
    planned_conditions = doc.get("planned_conditions")
    if not isinstance(planned_conditions, list) or not planned_conditions:
        raise ValueError("score_results.json must carry planned_conditions, including failed planned rows")
    cond_plan: dict[str, dict[str, Any]] = {}
    for row in planned_conditions:
        if not isinstance(row, dict) or row.get("condition_id") is None:
            raise ValueError("every planned_conditions entry needs condition_id")
        key = str(row["condition_id"])
        if key in cond_plan:
            raise ValueError(f"duplicate planned condition: {key}")
        cond_plan[key] = row
    if tuple(cond_plan) != CONDITION_ORDER:
        raise ValueError(f"conditions must match the locked 8-row order {CONDITION_ORDER}; got {tuple(cond_plan)}")
    dose_values = {key: _finite(cond_plan[key].get("relative_dose_db")) for key in DOSE_ORDER}
    if dose_values != {"DOSE_M6": -6.0, "DOSE_M3": -3.0, "DOSE_P3": 3.0, "DOSE_P6": 6.0}:
        raise ValueError(f"locked nonzero dose levels must be -6, -3, +3, +6 dB; got {dose_values}")
    if _finite(cond_plan["RAW_R1"].get("relative_dose_db")) != 0.0:
        raise ValueError("RAW_R1 is the predeclared zero-dose point")

    source_map = _speakers_as_map(doc.get("speakers"))
    unexpected_speakers = sorted(set(source_map) - set(speaker_ids))
    if unexpected_speakers:
        raise ValueError(f"score_results.json contains speakers outside the frozen cohort: {unexpected_speakers}")
    speakers: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for speaker_id in speaker_ids:
        speaker_doc = source_map.get(speaker_id)
        if speaker_doc is None:
            speaker_doc = {}
            speaker_status = "MISSING_SPEAKER_ROW"
        else:
            speaker_status = str(speaker_doc.get("status", "PRESENT"))
        support = speaker_doc.get("support") if isinstance(speaker_doc.get("support"), dict) else {}
        primary_q = support.get("primary_guard20_q_indices", support.get("query_indices"))
        full_q = support.get("full_q_indices")
        support_status = str(support.get("status", "UNKNOWN"))
        source_sha256 = speaker_doc.get("source_sha256")
        source_hash_ok = _is_sha256(source_sha256)
        support_ok = (
            support_status in SUCCESS_STATUSES
            and isinstance(primary_q, list)
            and len(primary_q) > 0
            and len(set(primary_q)) == len(primary_q)
            and isinstance(full_q, list)
            and len(full_q) > 0
            and len(set(full_q)) == len(full_q)
            and set(primary_q).issubset(set(full_q))
        )
        if not support_ok:
            support_status = "SUPPORT_NOT_PASS"
        raw_conditions = _conditions_as_map(speaker_doc.get("conditions"))
        unexpected_conditions = sorted(set(raw_conditions) - set(CONDITION_ORDER))
        if unexpected_conditions:
            raise ValueError(f"{speaker_id} has conditions outside the frozen 8-row plan: {unexpected_conditions}")
        normalized_conditions: dict[str, dict[str, Any] | None] = {}
        errors_by_condition: dict[str, list[str]] = {}
        for cond_id in CONDITION_ORDER:
            row = raw_conditions.get(cond_id)
            if row is None:
                normalized_conditions[cond_id] = None
                errors_by_condition[cond_id] = ["planned condition row missing"]
                output_row = {
                    "speaker_id": speaker_id,
                    "condition_id": cond_id,
                    "status": "MISSING_CONDITION_ROW",
                    "missing_reason": "planned condition row missing",
                    "relative_dose_db": _finite(cond_plan[cond_id].get("relative_dose_db")),
                }
            else:
                worker_status = str(row.get("status", "UNKNOWN"))
                if worker_status not in SUCCESS_STATUSES:
                    normalized_conditions[cond_id] = None
                    errors_by_condition[cond_id] = [str(row.get("missing_reason") or f"worker status {worker_status}")]
                    output_row = {
                        "speaker_id": speaker_id,
                        "condition_id": cond_id,
                        "status": worker_status,
                        "missing_reason": row.get("missing_reason"),
                        "relative_dose_db": _finite(row.get("relative_dose_db", cond_plan[cond_id].get("relative_dose_db"))),
                        "audio_rms_dbfs": _finite(row.get("audio_rms_dbfs")),
                        "video_sha256": row.get("video_sha256"),
                        "feature_sha256": row.get("feature_sha256"),
                        "visual_sha256": row.get("visual_sha256"),
                        "audio_variant_sha256": row.get("audio_variant_sha256"),
                    }
                else:
                    if support_ok:
                        score, errors = _score_errors(row, primary_q, full_q)
                    else:
                        score, errors = None, ["speaker primary support is not valid"]
                    if score is None:
                        normalized_conditions[cond_id] = None
                        errors_by_condition[cond_id] = errors
                        output_row = {
                            "speaker_id": speaker_id,
                            "condition_id": cond_id,
                            "status": "VALIDATION_FAILED",
                            "missing_reason": "; ".join(errors),
                            "relative_dose_db": _finite(row.get("relative_dose_db", cond_plan[cond_id].get("relative_dose_db"))),
                            "audio_rms_dbfs": _finite(row.get("audio_rms_dbfs")),
                            "video_sha256": row.get("video_sha256"),
                            "feature_sha256": row.get("feature_sha256"),
                            "visual_sha256": row.get("visual_sha256"),
                            "audio_variant_sha256": row.get("audio_variant_sha256"),
                        }
                    else:
                        normalized_conditions[cond_id] = score
                        errors_by_condition[cond_id] = []
                        output_row = {
                            "speaker_id": speaker_id,
                            "condition_id": cond_id,
                            "status": "PASS",
                            "missing_reason": None,
                            "relative_dose_db": score["relative_dose_db"] if score["relative_dose_db"] is not None else _finite(cond_plan[cond_id].get("relative_dose_db")),
                            **{k: score.get(k) for k in (*METRIC_NAMES, "audio_rms_dbfs")},
                            "video_sha256": score.get("video_sha256"),
                            "feature_sha256": score.get("feature_sha256"),
                            "visual_sha256": score.get("visual_sha256"),
                            "visual_path": score.get("visual_path"),
                            "audio_variant_sha256": score.get("audio_variant_sha256"),
                            "distance_path": score.get("distance_path"),
                            "distance_sha256": score.get("distance_sha256"),
                            "distance_shape": json.dumps(score.get("distance_shape"), ensure_ascii=False),
                            "q_count": len(score["q_support"]),
                            "motion_json": json.dumps(score["motion"], ensure_ascii=False, sort_keys=True),
                            "encoder_json": json.dumps(score["encoder"], ensure_ascii=False, sort_keys=True),
                            "engine_metadata_json": json.dumps(score["engine_metadata"], ensure_ascii=False, sort_keys=True),
                            "lag_frame_convention": score.get("lag_frame_convention"),
                        }
            condition_rows.append(output_row)
            status_counts[str(output_row["status"])] += 1
        # Verify measured sentence RMS against each declared amplitude intervention.
        raw_r1 = normalized_conditions.get("RAW_R1")
        if raw_r1 is not None:
            raw_level = raw_r1["audio_rms_dbfs"]
            locked_relative = {"RAW_R1": 0.0, "RAW_R2": 0.0, "DOSE_M6": -6.0, "DOSE_M3": -3.0, "DOSE_P3": 3.0, "DOSE_P6": 6.0}
            for cond_id in CONDITION_ORDER:
                score = normalized_conditions.get(cond_id)
                if score is None:
                    continue
                measured_delta = score["audio_rms_dbfs"] - raw_level
                if abs(measured_delta - score["relative_dose_db"]) > 0.03:
                    normalized_conditions[cond_id] = None
                    errors_by_condition[cond_id].append("relative_dose_db disagrees with measured RMS dBFS")
                elif cond_id in locked_relative and abs(score["relative_dose_db"] - locked_relative[cond_id]) > 0.03:
                    normalized_conditions[cond_id] = None
                    errors_by_condition[cond_id].append("measured relative_dose_db disagrees with locked dose")
            # The predeclared fixed anchor is each speaker's RAW_R1 argmin column.
            raw_r1 = normalized_conditions.get("RAW_R1")
            if raw_r1 is not None:
                anchor_index = int(np.argmin(raw_r1["curve31"]))
                for cond_id in CONDITION_ORDER:
                    score = normalized_conditions.get(cond_id)
                    if score is not None and abs(score["anchor_D"] - score["curve31"][anchor_index]) > 1e-6:
                        normalized_conditions[cond_id] = None
                        errors_by_condition[cond_id].append("anchor.D does not use this speaker's RAW_R1 argmin column")
        if not source_hash_ok:
            for cond_id in CONDITION_ORDER:
                if normalized_conditions.get(cond_id) is not None:
                    normalized_conditions[cond_id] = None
                    errors_by_condition[cond_id].append("speaker source_sha256 is missing or invalid")
        # Keep emitted status rows synchronized with the validations above.
        for output_row in condition_rows[-len(CONDITION_ORDER):]:
            cond_id = output_row["condition_id"]
            if output_row["status"] == "PASS" and normalized_conditions.get(cond_id) is None:
                output_row["status"] = "VALIDATION_FAILED"
                output_row["missing_reason"] = "; ".join(errors_by_condition[cond_id])
        status_counts = Counter(row["status"] for row in condition_rows)
        # A participant is primary-complete only when their common support passes and all four repeated RAW/FIXED scores pass.
        primary_complete = support_ok and source_hash_ok and all(normalized_conditions[c] is not None for c in CONDITION_ORDER[:4])
        family_complete = support_ok and source_hash_ok and all(normalized_conditions[c] is not None for c in ("RAW_R1", *DOSE_ORDER))
        all_condition_complete = support_ok and source_hash_ok and all(normalized_conditions[c] is not None for c in CONDITION_ORDER)
        raw_mean = fixed_mean = primary_delta = None
        if primary_complete:
            raw_mean = float(np.mean([normalized_conditions["RAW_R1"]["C"], normalized_conditions["RAW_R2"]["C"]]))
            fixed_mean = float(np.mean([normalized_conditions["FIXED_R1"]["C"], normalized_conditions["FIXED_R2"]["C"]]))
            primary_delta = fixed_mean - raw_mean
        dose_contrasts = {}
        if normalized_conditions["RAW_R1"] is not None:
            for cond_id in DOSE_ORDER:
                if normalized_conditions[cond_id] is not None:
                    dose_contrasts[cond_id] = normalized_conditions[cond_id]["C"] - normalized_conditions["RAW_R1"]["C"]
        speakers.append(
            {
                "speaker_id": speaker_id,
                "source_sha256": source_sha256,
                "source_hash_valid": bool(source_hash_ok),
                "speaker_status": speaker_status,
                "support_status": support_status,
                "primary_q_count": len(primary_q) if isinstance(primary_q, list) else None,
                "full_q_count": len(full_q) if isinstance(full_q, list) else None,
                "primary_complete": bool(primary_complete),
                "family_complete": bool(family_complete),
                "all_condition_complete": bool(all_condition_complete),
                "raw_mean_C": raw_mean,
                "fixed_mean_C": fixed_mean,
                "primary_delta_C": primary_delta,
                "dose_contrasts": dose_contrasts,
                "conditions": normalized_conditions,
                "condition_errors": errors_by_condition,
            }
        )
    return {
        "planned_speakers": speaker_ids,
        "condition_plan": cond_plan,
        "speakers": speakers,
        "condition_rows": condition_rows,
        "condition_status_counts": dict(status_counts),
        "protocol_sha256": doc["protocol_sha256"],
    }


def _valid_speakers(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    return [row for row in rows if row[key]]


def _curve_speaker_groups(speaker: dict[str, Any]) -> dict[str, list[float]]:
    cond = speaker["conditions"]
    out: dict[str, list[float]] = {}
    for cond_id in CONDITION_ORDER:
        if cond[cond_id] is not None:
            out[cond_id] = cond[cond_id]["curve31"]
    for alias, left, right in (
        ("RAW_MEAN", "RAW_R1", "RAW_R2"),
        ("FIXED_MEAN", "FIXED_R1", "FIXED_R2"),
    ):
        if cond[left] is not None and cond[right] is not None:
            out[alias] = list(np.mean([cond[left]["curve31"], cond[right]["curve31"]], axis=0))
    return out


def _metric_values(speaker: dict[str, Any], condition_id: str) -> dict[str, float]:
    row = speaker["conditions"].get(condition_id)
    if row is None:
        return {}
    values = {name: float(row[name]) for name in METRIC_NAMES if row.get(name) is not None}
    for prefix in ("motion", "encoder"):
        for key, value in row.get(prefix, {}).items():
            numeric = _finite(value)
            if numeric is not None:
                values[f"{prefix}.{key}"] = numeric
    for key in ("audio_rms_dbfs", "relative_dose_db"):
        numeric = _finite(row.get(key))
        if numeric is not None:
            values[key] = numeric
    return values


def _repeatability(speakers: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[float]] = {}
    for speaker in speakers:
        for arm, first, second in (("RAW", "RAW_R1", "RAW_R2"), ("FIXED", "FIXED_R1", "FIXED_R2")):
            a = speaker["conditions"].get(first)
            b = speaker["conditions"].get(second)
            if a is None or b is None:
                continue
            av, bv = _metric_values(speaker, first), _metric_values(speaker, second)
            for metric in sorted(set(av) & set(bv)):
                delta = float(bv[metric] - av[metric])
                rows.append({"speaker_id": speaker["speaker_id"], "arm": arm, "metric": metric, "r2_minus_r1": delta})
                groups.setdefault((arm, metric), []).append(delta)
            rows.append(
                {
                    "speaker_id": speaker["speaker_id"],
                    "arm": arm,
                    "metric": "asset_hash_equality",
                    "r2_minus_r1": None,
                    "video_sha256_equal": bool(a.get("video_sha256") and a.get("video_sha256") == b.get("video_sha256")),
                    "visual_sha256_equal": bool(a.get("visual_sha256") and a.get("visual_sha256") == b.get("visual_sha256")),
                    "audio_variant_sha256_equal": bool(a.get("audio_variant_sha256") and a.get("audio_variant_sha256") == b.get("audio_variant_sha256")),
                }
            )
    summaries = []
    for index, ((arm, metric), values) in enumerate(sorted(groups.items())):
        d = _description(values, SEED + 1300 + index)
        summaries.append(
            {
                "arm": arm,
                "metric": metric,
                "n_speakers": len(values),
                "n_planned": len(speakers),
                "mean_r2_minus_r1": d["mean"],
                "ci95_pointwise_descriptive": d["ci95"],
                "purpose": "repeatability description only; one repeat pair per speaker",
            }
        )
    return rows, summaries


def _description(values: list[float], seed: int) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 1:
        return {"n": 1, "mean": float(x[0]), "ci95": None}
    inds = bootstrap_indices(len(x), BOOTSTRAP_DRAWS, seed)
    return percentile_mean_interval(x, inds)


def analyze(dataset: dict[str, Any], input_sha256: str, source_sha256: str) -> dict[str, Any]:
    rows = dataset["speakers"]
    n_planned = len(dataset["planned_speakers"])
    primary_speakers = _valid_speakers(rows, "primary_complete")
    family_speakers = _valid_speakers(rows, "family_complete")
    primary: dict[str, Any] = {
        "status": "COMPUTABLE" if len(primary_speakers) == n_planned else "WITHHELD_INCOMPLETE_PLANNED_DENOMINATOR",
        "n_planned": n_planned,
        "n_complete": len(primary_speakers),
        "contrast": "per-speaker mean(FIXED_R1,FIXED_R2) C minus mean(RAW_R1,RAW_R2) C",
        "unit": "speaker",
        "bootstrap_seed": SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }
    if len(primary_speakers) == n_planned:
        vals = [float(row["primary_delta_C"]) for row in primary_speakers]
        primary.update(paired_primary_stats(vals, bootstrap_indices(len(vals))))
    family: dict[str, Any] = {
        "status": "COMPUTABLE" if len(family_speakers) == n_planned else "WITHHELD_INCOMPLETE_PLANNED_DENOMINATOR",
        "n_planned": n_planned,
        "n_complete": len(family_speakers),
        "contrast_definition": "C(dose) - C(RAW_R1), per speaker",
        "planned_family_order": list(DOSE_ORDER),
    }
    if len(family_speakers) == n_planned:
        matrix = np.asarray(
            [[float(row["dose_contrasts"][cond]) for cond in DOSE_ORDER] for row in family_speakers],
            dtype=np.float64,
        )
        family.update(max_t_family_stats(matrix, list(DOSE_ORDER), bootstrap_indices(n_planned)))

    descriptive: dict[str, dict[str, Any]] = {}
    curve_rows: list[dict[str, Any]] = []
    curve_summary: list[dict[str, Any]] = []
    for cond_id in CONDITION_ORDER:
        cond_speakers = [s for s in rows if s["conditions"][cond_id] is not None]
        metric_names = sorted({k for s in cond_speakers for k in _metric_values(s, cond_id)})
        for metric_idx, metric in enumerate(metric_names):
            vals = [_metric_values(s, cond_id)[metric] for s in cond_speakers if metric in _metric_values(s, cond_id)]
            descr = _description(vals, SEED + 100 + list(CONDITION_ORDER).index(cond_id) * 100 + metric_idx)
            descr.update({"condition_id": cond_id, "metric": metric, "n_planned": n_planned, "purpose": "descriptive point interval only"})
            descriptive[f"{cond_id}.{metric}"] = descr

    # Full 31-lag curves: aggregate each speaker first, then provide descriptive pointwise intervals.
    curve_groups_order = ("RAW_R1", "RAW_MEAN", "FIXED_MEAN", *DOSE_ORDER)
    for group_idx, group_id in enumerate(curve_groups_order):
        per_speaker = []
        for speaker in rows:
            curves = _curve_speaker_groups(speaker)
            if group_id in curves:
                per_speaker.append((speaker["speaker_id"], curves[group_id]))
        if not per_speaker:
            continue
        matrix = np.asarray([curve for _, curve in per_speaker], dtype=np.float64)
        inds = bootstrap_indices(len(per_speaker), BOOTSTRAP_DRAWS, SEED + 500 + group_idx)
        boot_means = matrix[inds].mean(axis=1)
        lags = None
        for speaker in rows:
            condition_key = group_id if group_id in CONDITION_ORDER else ("RAW_R1" if group_id == "RAW_MEAN" else "FIXED_R1")
            value = speaker["conditions"].get(condition_key)
            if value is not None:
                lags = value["lags"]
                break
        if lags is None:
            continue
        for i, lag in enumerate(lags):
            lo, hi = np.quantile(boot_means[:, i], [0.025, 0.975], method="linear")
            curve_summary.append(
                {
                    "curve_id": group_id,
                    "lag": float(lag),
                    "n_speakers": len(per_speaker),
                    "n_planned": n_planned,
                    "mean_distance": float(matrix[:, i].mean()),
                    "ci95_low_pointwise": float(lo),
                    "ci95_high_pointwise": float(hi),
                    "interval": "descriptive speaker-bootstrap pointwise 95%; not simultaneous",
                }
            )
        for speaker_id, curve in per_speaker:
            for lag, dist in zip(lags, curve):
                curve_rows.append({"speaker_id": speaker_id, "curve_id": group_id, "lag": float(lag), "distance": float(dist)})

    repeat_rows, repeat_summary = _repeatability(rows)
    return {
        "analysis_version": "p1_gain_dose_response_20261016_v1",
        "analysis_source_sha256": source_sha256,
        "input_sha256": input_sha256,
        "protocol_sha256": dataset["protocol_sha256"],
        "bootstrap": {"generator": "numpy.random.PCG64", "seed": SEED, "draws": BOOTSTRAP_DRAWS},
        "planned_speakers": dataset["planned_speakers"],
        "n_planned_speakers": n_planned,
        "n_planned_condition_rows": n_planned * len(CONDITION_ORDER),
        "n_condition_rows_materialized": len(dataset["condition_rows"]),
        "n_condition_rows_pass": sum(row["status"] == "PASS" for row in dataset["condition_rows"]),
        "condition_status_counts": dataset["condition_status_counts"],
        "n_primary_complete": len(primary_speakers),
        "n_family_complete": len(family_speakers),
        "primary_fixed_minus_raw": primary,
        "dose_family_vs_raw_r1": family,
        "descriptive_metrics": descriptive,
        "repeatability_summary": repeat_summary,
        "curve_groups": curve_groups_order,
        "curve_group_summary": curve_summary,
        "claim_scope": "new-speaker waveform-amplitude result under this fixed audio/video/scorer pipeline; combined conditioning and score-map response, not cross-model or perceptual benefit",
    }, curve_rows, repeat_rows


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _svg_text(x: float, y: float, text: str, size: int = 14, anchor: str = "middle", color: str = "#222") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="Arial,sans-serif" font-size="{size}" fill="{color}">{html.escape(text)}</text>'


def _line(x1: float, y1: float, x2: float, y2: float, color: str = "#777", width: float = 1.0, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr}/>'


def _plot_range(values: list[float]) -> tuple[float, float]:
    if not values:
        return -1.0, 1.0
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        pad = max(abs(lo) * 0.1, 0.5)
    else:
        pad = (hi - lo) * 0.12
    return lo - pad, hi + pad


def _render_effects_svg(summary: dict[str, Any], speaker_rows: list[dict[str, Any]], out: Path) -> None:
    width, height = 1180, 760
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    chunks.append(_svg_text(width / 2, 36, "New-speaker amplitude response: paired P1 and dose curve", 23))
    chunks.append(_svg_text(width / 2, 61, "C is higher-is-better; lines connect repeated measurements within speaker. No monotone fit.", 13, color="#555"))
    # Left: primary paired C comparison.
    x0, y0, pw, ph = 80, 120, 465, 470
    chunks.append(_svg_text(x0 + pw / 2, 102, "Primary: FIXED mean repeats − RAW mean repeats", 16))
    vals = []
    for s in speaker_rows:
        for cid in ("RAW_R1", "RAW_R2", "FIXED_R1", "FIXED_R2"):
            row = s["conditions"].get(cid)
            if row is not None:
                vals.append(row["C"])
    ymin, ymax = _plot_range(vals)
    def yleft(v: float) -> float:
        return y0 + ph - (v - ymin) / (ymax - ymin) * ph
    xa, xb = x0 + 120, x0 + 350
    chunks.extend([_line(x0, y0 + ph, x0 + pw, y0 + ph), _line(x0, y0, x0, y0 + ph)])
    for tick in np.linspace(ymin, ymax, 6):
        yy = yleft(float(tick))
        chunks.append(_line(x0, yy, x0 + pw, yy, "#ddd"))
        chunks.append(_svg_text(x0 - 10, yy + 4, f"{tick:.2f}", 11, "end", "#555"))
    chunks.append(_svg_text(x0 + 16, y0 + ph / 2, "C", 13))
    for s in speaker_rows:
        rvals = [s["conditions"].get(c) for c in ("RAW_R1", "RAW_R2")]
        fvals = [s["conditions"].get(c) for c in ("FIXED_R1", "FIXED_R2")]
        if all(v is not None for v in (*rvals, *fvals)):
            rv = float(np.mean([v["C"] for v in rvals]))
            fv = float(np.mean([v["C"] for v in fvals]))
            chunks.append(_line(xa, yleft(rv), xb, yleft(fv), "#c7cdd2", 1.1))
            chunks.append(f'<circle cx="{xa}" cy="{yleft(rv):.1f}" r="3" fill="#6889a6"/>')
            chunks.append(f'<circle cx="{xb}" cy="{yleft(fv):.1f}" r="3" fill="#c26d55"/>')
    pair_means = []
    for cid, x, color in (("RAW_R1", xa, "#26618a"), ("FIXED_R1", xb, "#a84830")):
        conds = ("RAW_R1", "RAW_R2") if cid == "RAW_R1" else ("FIXED_R1", "FIXED_R2")
        speaker_vals = [float(np.mean([s["conditions"][c]["C"] for c in conds])) for s in speaker_rows if all(s["conditions"].get(c) is not None for c in conds)]
        if speaker_vals:
            mean = float(np.mean(speaker_vals))
            pair_means.append((cid, mean, len(speaker_vals)))
            chunks.append(f'<circle cx="{x}" cy="{yleft(mean):.1f}" r="7" fill="{color}" stroke="white" stroke-width="2"/>')
    chunks.append(_svg_text(xa, y0 + ph + 28, "RAW (mean of r1/r2)", 12))
    chunks.append(_svg_text(xb, y0 + ph + 28, "FIXED (mean of r1/r2)", 12))
    primary = summary["primary_fixed_minus_raw"]
    ptxt = "Primary CI withheld: incomplete planned speaker denominator"
    if primary.get("status") == "COMPUTABLE":
        lo, hi = primary["ci95"]
        ptxt = f"ΔC = {primary['mean']:.3f}  (paired speaker bootstrap 95% CI {lo:.3f}, {hi:.3f}; n={primary['n']})"
    chunks.append(_svg_text(x0 + pw / 2, y0 + ph + 64, ptxt, 12, color="#333"))
    chunks.append(_svg_text(x0 + pw / 2, y0 + ph + 88, "Pair summaries use the two repeats; individual gray lines are speaker-level pairs.", 11, color="#555"))

    # Right: dose curve against RAW_R1 at zero; FIXED is deliberately absent here.
    rx, ry, rw, rh = 650, 120, 450, 470
    chunks.append(_svg_text(rx + rw / 2, 102, f"Dose response: RAW_R1 is zero; planned n={len(speaker_rows)}", 16))
    ids = ("DOSE_M6", "DOSE_M3", "RAW_R1", "DOSE_P3", "DOSE_P6")
    dose_x = [-6.0, -3.0, 0.0, 3.0, 6.0]
    # Use the locked plan labels; x-axis is relative intervention dose, not absolute RMS dBFS.
    all_c = [s["conditions"][cid]["C"] for s in speaker_rows for cid in ids if s["conditions"].get(cid) is not None]
    cy0, cy1 = _plot_range(all_c)
    xlo, xhi = -7.0, 7.0
    def xright(v: float) -> float:
        return rx + (v - xlo) / (xhi - xlo) * rw
    def yright(v: float) -> float:
        return ry + rh - (v - cy0) / (cy1 - cy0) * rh
    chunks.extend([_line(rx, ry + rh, rx + rw, ry + rh), _line(rx, ry, rx, ry + rh)])
    for tick in np.linspace(cy0, cy1, 6):
        yy = yright(float(tick))
        chunks.append(_line(rx, yy, rx + rw, yy, "#ddd"))
        chunks.append(_svg_text(rx - 9, yy + 4, f"{tick:.2f}", 11, "end", "#555"))
    chunks.append(_svg_text(rx + rw / 2, ry + rh + 33, "Relative dose from RAW (dB)", 13))
    chunks.append(_svg_text(rx + 12, ry + rh / 2, "C", 13))
    for speaker in speaker_rows:
        pts: list[tuple[float, float] | None] = []
        for cid, dose in zip(ids, dose_x):
            row = speaker["conditions"].get(cid)
            pts.append((xright(dose), yright(row["C"])) if row is not None else None)
        for left_point, right_point in zip(pts, pts[1:]):
            if left_point is not None and right_point is not None:
                chunks.append(_line(left_point[0], left_point[1], right_point[0], right_point[1], "#d1d5d8", 0.9))
        for point in pts:
            if point is not None:
                x, y = point
                chunks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.3" fill="#aab0b4"/>')
    dose_means: dict[str, tuple[float, float, str]] = {}
    colors = ("#246b8e", "#397f64", "#a77b25", "#9f4e62", "#573e8f")
    for cid, dose, color in zip(ids, dose_x, colors):
        v = [s["conditions"][cid]["C"] for s in speaker_rows if s["conditions"].get(cid) is not None]
        if not v:
            continue
        d = _description([float(x) for x in v], SEED + 900 + ids.index(cid))
        mean = d["mean"]
        if d["ci95"]:
            yl, yh = yright(d["ci95"][0]), yright(d["ci95"][1])
            chunks.append(_line(xright(dose), yl, xright(dose), yh, color, 2.2))
            chunks.append(_line(xright(dose) - 5, yl, xright(dose) + 5, yl, color, 1.4))
            chunks.append(_line(xright(dose) - 5, yh, xright(dose) + 5, yh, color, 1.4))
        chunks.append(f'<circle cx="{xright(dose):.1f}" cy="{yright(mean):.1f}" r="5.5" fill="{color}" stroke="white" stroke-width="1.5"/>')
        chunks.append(_svg_text(xright(dose), ry + rh + 47, f"n={len(v)}/{len(speaker_rows)}", 9, color="#555"))
        dose_means[cid] = (xright(dose), yright(mean), color)
    for left_id, right_id in zip(ids, ids[1:]):
        if left_id in dose_means and right_id in dose_means:
            x1, y1, _ = dose_means[left_id]
            x2, y2, color = dose_means[right_id]
            chunks.append(_line(x1, y1, x2, y2, color, 2.0))
    for tick in (-6, -3, 0, 3, 6):
        chunks.append(_svg_text(xright(float(tick)), ry + rh + 17, f"{tick:+d}" if tick else "0", 11, color="#555"))
    chunks.append(_svg_text(rx + rw / 2, ry + rh + 61, "Curves connect the predeclared levels; pointwise intervals are descriptive, not simultaneous.", 11, color="#555"))
    chunks.append(_svg_text(rx + rw / 2, ry + rh + 82, "FIXED absolute per-sentence dBFS is reported in a separate figure/table.", 11, color="#555"))
    chunks.append("</svg>")
    out.write_text("\n".join(chunks), encoding="utf-8")


def _render_levels_svg(dataset: dict[str, Any], out: Path) -> None:
    rows = dataset["speakers"]
    width, height = 1100, 470
    left, top, plot_w, plot_h = 90, 95, 930, 270
    vals = []
    for speaker in rows:
        for cid in ("RAW_R1", "FIXED_R1", "FIXED_R2"):
            row = speaker["conditions"].get(cid)
            if row is not None and row.get("audio_rms_dbfs") is not None:
                vals.append(float(row["audio_rms_dbfs"]))
    ymin, ymax = _plot_range(vals)
    def y(v: float) -> float:
        return top + plot_h - (v - ymin) / (ymax - ymin) * plot_h
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append(_svg_text(width / 2, 36, "Measured audio RMS level by sentence and condition", 21))
    parts.append(_svg_text(width / 2, 61, "FIXED shows each sentence’s actual RMS dBFS; this axis is separate from the uniform relative-dose response.", 12, color="#555"))
    parts.extend([_line(left, top + plot_h, left + plot_w, top + plot_h), _line(left, top, left, top + plot_h)])
    for tick in np.linspace(ymin, ymax, 6):
        yy = y(float(tick))
        parts.append(_line(left, yy, left + plot_w, yy, "#ddd"))
        parts.append(_svg_text(left - 8, yy + 4, f"{tick:.1f}", 11, "end", "#555"))
    colors = {"RAW_R1": "#54738a", "FIXED_R1": "#b5533c", "FIXED_R2": "#e3a48b"}
    xstep = plot_w / max(1, len(rows))
    for i, speaker in enumerate(rows):
        x = left + (i + 0.5) * xstep
        for offset, cid in zip((-8, 0, 8), ("RAW_R1", "FIXED_R1", "FIXED_R2")):
            row = speaker["conditions"].get(cid)
            if row is None or row.get("audio_rms_dbfs") is None:
                continue
            yy = y(float(row["audio_rms_dbfs"]))
            parts.append(f'<circle cx="{x + offset:.1f}" cy="{yy:.1f}" r="3.5" fill="{colors[cid]}"/>')
        parts.append(_svg_text(x, top + plot_h + 19, speaker["speaker_id"], 9, color="#555"))
    parts.append(_svg_text(28, top + plot_h / 2, "RMS (dBFS)", 12))
    parts.append(_svg_text(width / 2, height - 34, "Speaker ID; each point is an actual saved sentence/condition measurement", 12, color="#555"))
    legend_y = 83
    for i, (label, cid) in enumerate((("RAW r1", "RAW_R1"), ("FIXED r1", "FIXED_R1"), ("FIXED r2", "FIXED_R2"))):
        x = 400 + i * 130
        parts.append(f'<circle cx="{x}" cy="{legend_y}" r="4" fill="{colors[cid]}"/>')
        parts.append(_svg_text(x + 10, legend_y + 4, label, 11, "start"))
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")


def _render_dose_family_svg(summary: dict[str, Any], out: Path) -> None:
    family = summary["dose_family_vs_raw_r1"]
    width, height = 920, 390
    left, top, pw, ph = 245, 95, 580, 210
    rows = family.get("contrasts", {}) if family.get("status") == "COMPUTABLE" else {}
    extent_values = [0.0]
    for cid in DOSE_ORDER:
        row = rows.get(cid)
        if row:
            extent_values.extend(row["simultaneous_ci95"])
            extent_values.append(row["mean"])
    xmin, xmax = _plot_range(extent_values)
    if xmin > 0:
        xmin = 0.0
    if xmax < 0:
        xmax = 0.0
    def xp(value: float) -> float:
        return left + (value - xmin) / (xmax - xmin) * pw
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append(_svg_text(width / 2, 35, "Predeclared dose family: ΔC against RAW_R1", 21))
    parts.append(_svg_text(width / 2, 61, "Centered paired speaker bootstrap; fixed original SE; simultaneous 95% max-|t| intervals.", 12, color="#555"))
    xzero = xp(0.0)
    parts.append(_line(xzero, top - 8, xzero, top + ph + 4, "#333", 1.5, "5,4"))
    n = len(DOSE_ORDER)
    colors = ("#397f64", "#a77b25", "#9f4e62", "#573e8f")
    for i, (cid, color) in enumerate(zip(DOSE_ORDER, colors)):
        y = top + 28 + i * 47
        parts.append(_svg_text(left - 14, y + 4, f"{cid.replace('DOSE_', '')} dB − RAW_R1", 12, "end"))
        row = rows.get(cid)
        if row is None:
            parts.append(_svg_text(left + 8, y + 4, "WITHHELD: incomplete planned denominator", 12, "start", "#6b737a"))
            continue
        lo, hi = row["simultaneous_ci95"]
        mean = row["mean"]
        parts.append(_line(xp(lo), y, xp(hi), y, color, 2.4))
        parts.append(_line(xp(lo), y - 6, xp(lo), y + 6, color, 1.5))
        parts.append(_line(xp(hi), y - 6, xp(hi), y + 6, color, 1.5))
        parts.append(f'<circle cx="{xp(mean):.1f}" cy="{y}" r="5.5" fill="{color}" stroke="white" stroke-width="1.5"/>')
        parts.append(_svg_text(left + pw + 10, y + 4, f"{mean:.3f} [{lo:.3f}, {hi:.3f}]", 11, "start"))
    for tick in np.linspace(xmin, xmax, 6):
        xx = xp(float(tick))
        parts.append(_line(xx, top + ph + 3, xx, top + ph + 8, "#555"))
        parts.append(_svg_text(xx, top + ph + 25, f"{tick:.2f}", 10, color="#555"))
    parts.append(_svg_text(left + pw / 2, top + ph + 51, "ΔC = C(dose) − C(RAW_R1); C higher is better", 12))
    parts.append(_svg_text(width / 2, height - 24, f"Family status: {family['status']} · n={family['n_complete']}/{family['n_planned']} speakers", 12, color="#555"))
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")


def _render_curves_svg(summary: dict[str, Any], out: Path) -> None:
    curves = summary["curve_group_summary"]
    width, height = 1180, 640
    left, top, pw, ph = 80, 105, 830, 430
    lag_values = sorted({float(row["lag"]) for row in curves})
    all_values = [v for row in curves for v in (row["ci95_low_pointwise"], row["ci95_high_pointwise"])]
    ymin, ymax = _plot_range(all_values)
    xmin, xmax = min(lag_values, default=-15.0), max(lag_values, default=15.0)
    if abs(xmax - xmin) < 1e-12:
        xmin, xmax = -1.0, 1.0
    def xp(v: float) -> float:
        return left + (v - xmin) / (xmax - xmin) * pw
    def yp(v: float) -> float:
        return top + ph - (v - ymin) / (ymax - ymin) * ph
    palette = {"RAW_R1": "#246b8e", "RAW_MEAN": "#72a7bf", "FIXED_MEAN": "#a84830", "DOSE_M6": "#397f64", "DOSE_M3": "#a77b25", "DOSE_P3": "#9f4e62", "DOSE_P6": "#573e8f"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    parts.append(_svg_text(width / 2, 35, "Complete 31-lag distance curves", 22))
    parts.append(_svg_text(width / 2, 61, "Each curve first averages q within sentence, then averages speakers equally; ribbons are descriptive pointwise 95% intervals.", 12, color="#555"))
    parts.extend([_line(left, top + ph, left + pw, top + ph), _line(left, top, left, top + ph)])
    for tick in np.linspace(ymin, ymax, 6):
        yy = yp(float(tick))
        parts.append(_line(left, yy, left + pw, yy, "#ddd"))
        parts.append(_svg_text(left - 8, yy + 4, f"{tick:.2f}", 11, "end", "#555"))
    for tick in np.linspace(xmin, xmax, 7):
        xx = xp(float(tick))
        parts.append(_line(xx, top + ph, xx, top + ph + 4, "#555"))
        parts.append(_svg_text(xx, top + ph + 20, f"{tick:g}", 10, color="#555"))
    by_group: dict[str, list[dict[str, Any]]] = {}
    for row in curves:
        by_group.setdefault(row["curve_id"], []).append(row)
    for group, vals in by_group.items():
        vals.sort(key=lambda r: r["lag"])
        color = palette.get(group, "#333")
        upper = [(xp(r["lag"]), yp(r["ci95_high_pointwise"])) for r in vals]
        lower = [(xp(r["lag"]), yp(r["ci95_low_pointwise"])) for r in reversed(vals)]
        polygon = " ".join(f"{x:.1f},{y:.1f}" for x, y in [*upper, *lower])
        parts.append(f'<polygon points="{polygon}" fill="{color}" fill-opacity="0.10" stroke="none"/>')
        pts = " ".join(f'{xp(r["lag"]):.1f},{yp(r["mean_distance"]):.1f}' for r in vals)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.3"/>')
    parts.append(_svg_text(left + pw / 2, top + ph + 44, "Lag index (locked 31-column scorer grid)", 12))
    parts.append(_svg_text(26, top + ph / 2, "Mean distance", 12))
    legend = list(by_group)
    for i, group in enumerate(legend):
        x, y = 950, 125 + i * 34
        color = palette.get(group, "#333")
        parts.append(_line(x, y, x + 26, y, color, 3))
        n = next(r["n_speakers"] for r in by_group[group])
        parts.append(_svg_text(x + 36, y + 4, f"{group} (n={n})", 12, "start"))
    parts.append(_svg_text(1055, 450, "Dose curves compare against RAW_R1 at 0 dB.", 11, color="#555"))
    parts.append(_svg_text(1055, 470, "FIXED is the mean of its two repeats.", 11, color="#555"))
    parts.append(_svg_text(1055, 490, "Do not take median/min of this mean curve to infer mean C.", 11, color="#555"))
    parts.append("</svg>")
    out.write_text("\n".join(parts), encoding="utf-8")


def _write_outputs(doc: dict[str, Any], dataset: dict[str, Any], summary: dict[str, Any], curve_rows: list[dict[str, Any]], repeat_rows: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "analysis_summary.json", summary)
    _write_json(
        output / "analysis_lock.json",
        {
            "analysis_version": summary["analysis_version"],
            "analysis_source_sha256": summary["analysis_source_sha256"],
            "input_sha256": summary["input_sha256"],
            "protocol_sha256": summary["protocol_sha256"],
            "bootstrap": summary["bootstrap"],
            "statistics_frozen_before_score_read": True,
            "primary": summary["primary_fixed_minus_raw"]["contrast"],
            "dose_family": summary["dose_family_vs_raw_r1"].get("method", "four predeclared C contrasts; centered paired max-|t|"),
        },
    )
    _csv(output / "speaker_conditions.csv", dataset["condition_rows"])
    speaker_rows = []
    for speaker in dataset["speakers"]:
        speaker_rows.append(
            {
                k: speaker[k]
                for k in (
                    "speaker_id", "source_sha256", "source_hash_valid", "speaker_status", "support_status", "primary_q_count", "full_q_count",
                    "primary_complete", "family_complete", "all_condition_complete", "raw_mean_C", "fixed_mean_C", "primary_delta_C",
                )
            }
            | {f"{key}_dose_delta_C": speaker["dose_contrasts"].get(key) for key in DOSE_ORDER}
        )
    _csv(output / "speaker_primary.csv", speaker_rows)
    _csv(output / "distance_curves.csv", curve_rows)
    _csv(output / "distance_curve_summary.csv", summary["curve_group_summary"])
    _csv(output / "repeatability_by_speaker.csv", repeat_rows)
    _csv(output / "repeatability_summary.csv", summary["repeatability_summary"])
    descriptive_rows = []
    for key, value in summary["descriptive_metrics"].items():
        descriptive_rows.append({"key": key, **value})
    _csv(output / "descriptive_metrics.csv", descriptive_rows)
    _render_effects_svg(summary, dataset["speakers"], output / "p1_primary_and_dose.svg")
    _render_dose_family_svg(summary, output / "p1_dose_family_maxT.svg")
    _render_levels_svg(dataset, output / "p1_measured_audio_levels.svg")
    _render_curves_svg(summary, output / "p1_distance_curves.svg")


def run(input_path: Path, output_path: Path) -> dict[str, Any]:
    doc = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("score_results.json root must be an object")
    input_hash = sha256_file(input_path)
    source_hash = sha256_file(Path(__file__).resolve())
    dataset = normalize_manifest(doc)
    summary, curve_rows, repeat_rows = analyze(dataset, input_hash, source_hash)
    _write_outputs(doc, dataset, summary, curve_rows, repeat_rows, output_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="score_results.json")
    parser.add_argument("--output", type=Path, required=True, help="new output directory; must not exist")
    args = parser.parse_args(argv)
    summary = run(args.input, args.output)
    print(json.dumps({"output": str(args.output), "primary_status": summary["primary_fixed_minus_raw"]["status"], "dose_family_status": summary["dose_family_vs_raw_r1"]["status"], "n_planned": summary["n_planned_speakers"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
