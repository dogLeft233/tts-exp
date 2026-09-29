from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import config
from .common import file_sha256, write_self_hashed_json
from .protocol import ProtocolError


def cluster_bootstrap(values: Sequence[float], groups: Sequence[str], seed: int = config.BOOTSTRAP_SEED, draws: int = config.BOOTSTRAP_DRAWS) -> dict[str, Any]:
    if len(values) != len(groups) or not values or draws <= 0:
        raise ValueError("bootstrap inputs are empty, mismatched, or invalid")
    by_group: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("bootstrap values must be finite")
        by_group[str(group)].append(value)
    labels = sorted(by_group)
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sampled = rng.choice(labels, size=len(labels), replace=True)
        means[index] = np.mean([value for label in sampled for value in by_group[label]])
    ci = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    return {
        "draws": int(draws),
        "seed": int(seed),
        "source_group_count": len(labels),
        "record_count": len(values),
        "mean": float(np.mean(values)),
        "ci95": ci,
        "group_means": {label: float(np.mean(by_group[label])) for label in labels},
    }


def endpoint(values: Sequence[float], groups: Sequence[str], field: str) -> dict[str, Any]:
    result = cluster_bootstrap(values, groups)
    result["field"] = field
    result["wins"] = {
        "positive_count": int(sum(float(value) > 0.0 for value in values)),
        "record_count": len(values),
    }
    return result


def _rows_by_key(matrix: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    if matrix.get("status") != "complete" or len(matrix.get("scores", [])) != config.EXPECTED_CELL_COUNT:
        raise ProtocolError("score matrix is incomplete")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in matrix["scores"]:
        key = (str(row.get("sample_id")), str(row.get("cell")))
        if key in result or key[1] not in config.MATRIX_CELLS:
            raise ProtocolError(f"duplicate or unknown score cell: {key}")
        for field in ("sync_c", "sync_d", "av_offset"):
            if not isinstance(row.get(field), (int, float)) or not np.isfinite(float(row[field])):
                raise ProtocolError(f"invalid score value: {key}/{field}")
        result[key] = row
    if len(result) != config.EXPECTED_CELL_COUNT:
        raise ProtocolError("score cell identity is incomplete")
    return result


def _diagnostic_rows(diagnostics: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if diagnostics.get("status") != "complete" or diagnostics.get("frozen_before_scoring") is not True:
        raise ProtocolError("audio diagnostics are incomplete or not frozen")
    result: dict[str, Mapping[str, Any]] = {}
    for row in diagnostics.get("rows", []):
        sample_id = str(row.get("sample_id"))
        if sample_id in result:
            raise ProtocolError(f"duplicate diagnostics row: {sample_id}")
        result[sample_id] = row
    if len(result) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("diagnostic row identity is incomplete")
    return result


def _replacement_endpoint(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows]
    groups = [str(row["source_group"]) for row in rows]
    return endpoint(values, groups, field)


def _compatibility(
    replacement_rows: Sequence[Mapping[str, Any]],
    baseline_offsets: Sequence[int],
) -> dict[str, Any]:
    gap_c = _replacement_endpoint(replacement_rows, "gap_c")
    gap_d = _replacement_endpoint(replacement_rows, "gap_d")
    offset_differences = [abs(int(row["av_offset"]) - int(base)) for row, base in zip(replacement_rows, baseline_offsets)]
    offset_agreement_count = int(sum(value <= config.OFFSET_TOLERANCE_FRAMES for value in offset_differences))
    return {
        "gap_C": gap_c,
        "gap_D": gap_d,
        "offset_agreement": {
            "tolerance_frames": config.OFFSET_TOLERANCE_FRAMES,
            "count": offset_agreement_count,
            "required": config.MIN_OFFSET_AGREEMENT_RECORDS,
            "differences": offset_differences,
            "passes": offset_agreement_count >= config.MIN_OFFSET_AGREEMENT_RECORDS,
        },
        "passes": bool(
            gap_c["ci95"][0] > config.REPLACEMENT_MARGIN
            and gap_d["ci95"][0] > config.REPLACEMENT_MARGIN
            and offset_agreement_count >= config.MIN_OFFSET_AGREEMENT_RECORDS
        ),
    }


def _movement(rows: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    values = [float(row["arms"][arm]["progress"]) for row in rows]
    groups = [str(row["source_group"]) for row in rows]
    stats = endpoint(values, groups, f"{arm}.progress")
    count = int(sum(value >= config.MOVEMENT_THRESHOLD for value in values))
    stats.update(
        {
            "threshold": config.MOVEMENT_THRESHOLD,
            "record_count_at_or_above_threshold": count,
            "required_record_count": config.MIN_MOVEMENT_RECORDS,
            "passes": bool(count >= config.MIN_MOVEMENT_RECORDS and stats["ci95"][0] > config.MOVEMENT_THRESHOLD),
            "per_record": values,
        }
    )
    return stats


def _score(matrix_rows: Mapping[tuple[str, str], Mapping[str, Any]], sample_id: str, video_arm: str, audio_arm: str) -> Mapping[str, Any]:
    key = (sample_id, f"V_{video_arm}/A_{audio_arm}")
    try:
        return matrix_rows[key]
    except KeyError as exc:
        raise ProtocolError(f"missing required score cell: {key}") from exc


def run_stage05(
    cohort: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_stage: Any,
) -> dict[str, Any]:
    matrix_rows = _rows_by_key(matrix)
    diagnostic_rows = _diagnostic_rows(diagnostics)
    records = list(cohort.get("records", []))
    if len(records) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("cohort is incomplete")
    baseline_offsets: list[int] = []
    baseline_scores: dict[str, Mapping[str, Any]] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        baseline = _score(matrix_rows, sample_id, "N", "N")
        baseline_scores[sample_id] = baseline
        baseline_offsets.append(int(baseline["av_offset"]))
    endpoint_rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in config.ARMS}
    descriptive: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        group = str(record["source_group"])
        baseline = baseline_scores[sample_id]
        for arm in config.ARMS:
            replacement = _score(matrix_rows, sample_id, arm, "N")
            own = _score(matrix_rows, sample_id, arm, arm)
            endpoint_rows[arm].append(
                {
                    "sample_id": sample_id,
                    "source_group": group,
                    "sync_c": float(replacement["sync_c"]),
                    "sync_d": float(replacement["sync_d"]),
                    "av_offset": int(replacement["av_offset"]),
                    "gap_c": float(replacement["sync_c"]) - float(baseline["sync_c"]),
                    "gap_d": float(baseline["sync_d"]) - float(replacement["sync_d"]),
                    "own_sync_c": float(own["sync_c"]),
                    "own_sync_d": float(own["sync_d"]),
                    "own_av_offset": int(own["av_offset"]),
                }
            )
        descriptive.append(
            {
                "sample_id": sample_id,
                "source_group": group,
                "baseline": {"sync_c": float(baseline["sync_c"]), "sync_d": float(baseline["sync_d"]), "av_offset": int(baseline["av_offset"])},
                "arms": {
                    arm: {
                        "replacement": endpoint_rows[arm][-1],
                        "diagnostics": diagnostic_rows[sample_id]["arms"][arm],
                    }
                    for arm in config.ARMS
                },
            }
        )
    compatibility = {arm: _compatibility(endpoint_rows[arm], baseline_offsets) for arm in config.ARMS}
    movement = {arm: _movement([diagnostic_rows[str(record["sample_id"])] for record in records], arm) for arm in config.MAG_ARMS}
    inv_mel_values = [float(diagnostic_rows[str(record["sample_id"])]["arms"]["INV"]["mel_mae_to_natural"]) for record in records]
    inv_mel = endpoint(inv_mel_values, [str(record["source_group"]) for record in records], "INV.mel_mae_to_natural")
    inv_mel["limit"] = config.INV_MEL_MAE_LIMIT
    inv_mel["passes"] = bool(inv_mel["mean"] <= config.INV_MEL_MAE_LIMIT)
    inv_control = {
        "mel_equivalence": inv_mel,
        "replacement_compatibility": compatibility["INV"],
        "passes": bool(inv_mel["passes"] and compatibility["INV"]["passes"]),
    }
    shift_rows = endpoint_rows["SHIFT_200"]
    shift_offset_changes = [abs(row["av_offset"] - base) for row, base in zip(shift_rows, baseline_offsets)]
    shift_offset_count = int(sum(value >= config.SHIFT_OFFSET_CHANGE_FRAMES for value in shift_offset_changes))
    shift_c_mean = float(np.mean([row["gap_c"] for row in shift_rows]))
    shift_d_mean = float(np.mean([row["gap_d"] for row in shift_rows]))
    shift_control = {
        "offset_changes": shift_offset_changes,
        "offset_change_threshold_frames": config.SHIFT_OFFSET_CHANGE_FRAMES,
        "offset_change_count": shift_offset_count,
        "offset_change_required": config.SHIFT_OFFSET_CHANGE_RECORDS,
        "gap_C_mean": shift_c_mean,
        "gap_D_mean": shift_d_mean,
        "degradation_threshold": config.SHIFT_MEAN_DEGRADATION,
        "passes": bool(
            shift_offset_count >= config.SHIFT_OFFSET_CHANGE_RECORDS
            or (shift_c_mean < config.SHIFT_MEAN_DEGRADATION and shift_d_mean < config.SHIFT_MEAN_DEGRADATION)
        ),
    }
    qualifying: list[str] = []
    for arm in config.MAG_ARMS:
        if movement[arm]["passes"] and compatibility[arm]["passes"]:
            qualifying.append(arm)
    max_compatible_alpha = max((config.ARM_STRENGTHS[arm] for arm in qualifying), default=None)
    if not inv_control["passes"] or not shift_control["passes"]:
        decision = "CONTROL_FAILED"
    elif qualifying:
        decision = "PHASE_PRESERVING_REPLACEMENT_ENVELOPE_FOUND"
    else:
        decision = "ONLY_TRIVIAL_REPLACEMENT_EQUIVALENCE_FOUND"
    analysis = {
        "schema_version": 1,
        "stage_id": "05_analysis",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "record_count": len(records),
        "source_group_count": len({str(record["source_group"]) for record in records}),
        "bootstrap": {"draws": config.BOOTSTRAP_DRAWS, "seed": config.BOOTSTRAP_SEED, "unit": "source_group_cluster", "confidence": 0.95},
        "replacement_margin": config.REPLACEMENT_MARGIN,
        "offset_gate": {"tolerance_frames": config.OFFSET_TOLERANCE_FRAMES, "required_records": config.MIN_OFFSET_AGREEMENT_RECORDS},
        "compatibility": compatibility,
        "movement": movement,
        "controls": {"INV": inv_control, "SHIFT_200": shift_control},
        "qualifying_mag_arms": qualifying,
        "max_compatible_alpha": max_compatible_alpha,
        "decisions": {
            "scientific_decision": decision,
            "identity_characterization_experiment_eligible": decision == "PHASE_PRESERVING_REPLACEMENT_ENVELOPE_FOUND",
            "scope": "fixed 23-record/23-source-group LRS3 fit-only cohort and exact registered phase-preserving construction",
        },
        "per_record": {arm: endpoint_rows[arm] for arm in config.ARMS},
        "descriptive_diagnostics": descriptive,
        "diagnostics_sha256": file_sha256(config.STAGES["02_audio_diagnostics"] / "diagnostics.json"),
        "matrix_manifest_sha256": file_sha256(config.STAGES["04_scores"] / "scores_manifest.json"),
    }
    write_self_hashed_json(output_stage / "analysis.json", analysis)
    return analysis
