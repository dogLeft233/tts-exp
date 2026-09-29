from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import file_sha256, write_self_hashed_json
from .protocol import ProtocolError


def cluster_bootstrap(values: Sequence[float], groups: Sequence[str], seed: int = config.BOOTSTRAP_SEED, draws: int = config.BOOTSTRAP_DRAWS) -> dict[str, Any]:
    if len(values) != len(groups) or len(values) == 0 or draws <= 0:
        raise ValueError("bootstrap inputs are empty, mismatched, or invalid")
    by_group: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("bootstrap values must be finite")
        by_group[str(group)].append(value)
    labels = sorted(by_group)
    group_means = {label: float(np.mean(items)) for label, items in by_group.items()}
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sampled = rng.choice(labels, size=len(labels), replace=True)
        means[index] = np.mean([group_means[label] for label in sampled])
    ci = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    return {
        "draws": int(draws),
        "seed": int(seed),
        "source_group_count": len(labels),
        "record_count": len(values),
        "mean": float(np.mean(values)),
        "ci95": ci,
        "group_means": group_means,
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


def _compatibility(replacement_rows: Sequence[Mapping[str, Any]], baseline_offsets: Sequence[int]) -> dict[str, Any]:
    gap_c = endpoint([float(row["gap_c"]) for row in replacement_rows], [str(row["source_group"]) for row in replacement_rows], "gap_C")
    gap_d = endpoint([float(row["gap_d"]) for row in replacement_rows], [str(row["source_group"]) for row in replacement_rows], "gap_D")
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


def _pair_row(record: Mapping[str, Any], score: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(record["sample_id"]),
        "source_group": str(record["source_group"]),
        "sync_c": float(score["sync_c"]),
        "sync_d": float(score["sync_d"]),
        "av_offset": int(score["av_offset"]),
        "gap_c": float(score["sync_c"]) - float(baseline["sync_c"]),
        "gap_d": float(baseline["sync_d"]) - float(score["sync_d"]),
    }


def run_stage04_analysis(
    cohort: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    matrix: Mapping[str, Any],
    output_stage: Path,
) -> dict[str, Any]:
    matrix_rows = _rows_by_key(matrix)
    diagnostic_rows = _diagnostic_rows(diagnostics)
    records = list(cohort.get("records", []))
    if len(records) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("cohort is incomplete")
    baseline_offsets: list[int] = []
    baseline_scores: dict[str, Mapping[str, Any]] = {}
    repeat_rows: list[dict[str, Any]] = []
    local_own_rows: list[dict[str, Any]] = []
    local_natural_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    local_damage_c: list[float] = []
    local_damage_d: list[float] = []
    descriptive: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        baseline = _score(matrix_rows, sample_id, "N", "N")
        repeat = _score(matrix_rows, sample_id, "N_REPEAT", "N")
        local_natural = _score(matrix_rows, sample_id, "LOCAL_SWAP", "N")
        local_own = _score(matrix_rows, sample_id, "LOCAL_SWAP", "LOCAL_SWAP")
        bridge_natural = _score(matrix_rows, sample_id, config.BRIDGE_ARM, "N")
        bridge_own = _score(matrix_rows, sample_id, config.BRIDGE_ARM, config.BRIDGE_ARM)
        baseline_scores[sample_id] = baseline
        baseline_offsets.append(int(baseline["av_offset"]))
        repeat_rows.append(_pair_row(record, repeat, baseline))
        local_natural_row = _pair_row(record, local_natural, baseline)
        local_own_row = _pair_row(record, local_own, baseline)
        bridge_row = _pair_row(record, bridge_natural, baseline)
        local_natural_rows.append(local_natural_row)
        local_own_rows.append(local_own_row)
        bridge_rows.append(bridge_row)
        local_damage_c.append(float(local_own["sync_c"]) - float(local_natural["sync_c"]))
        local_damage_d.append(float(local_natural["sync_d"]) - float(local_own["sync_d"]))
        descriptive.append(
            {
                "sample_id": sample_id,
                "source_group": str(record["source_group"]),
                "baseline": {"sync_c": float(baseline["sync_c"]), "sync_d": float(baseline["sync_d"]), "av_offset": int(baseline["av_offset"])},
                "arms": {
                    "N_REPEAT": {"replacement": repeat_rows[-1]},
                    "LOCAL_SWAP": {
                        "replacement": local_natural_row,
                        "own_audio": {"sync_c": float(local_own["sync_c"]), "sync_d": float(local_own["sync_d"]), "av_offset": int(local_own["av_offset"])},
                    },
                    "BRIDGE_075": {
                        "replacement": bridge_row,
                        "own_audio": {"sync_c": float(bridge_own["sync_c"]), "sync_d": float(bridge_own["sync_d"]), "av_offset": int(bridge_own["av_offset"])},
                        "diagnostics": diagnostic_rows[sample_id]["arms"][config.BRIDGE_ARM],
                    },
                },
            }
        )
    repeatability = _compatibility(repeat_rows, baseline_offsets)
    repeatability["passes"] = bool(repeatability["passes"])
    local_own_validity = _compatibility(local_own_rows, baseline_offsets)
    damage_c = endpoint(local_damage_c, [str(record["source_group"]) for record in records], "damage_C")
    damage_d = endpoint(local_damage_d, [str(record["source_group"]) for record in records], "damage_D")
    damage_positive_count = int(sum(c > 0.0 and d > 0.0 for c, d in zip(local_damage_c, local_damage_d)))
    local_sensitivity = {
        "damage_C": damage_c,
        "damage_D": damage_d,
        "both_damage_positive_count": damage_positive_count,
        "required_positive_count": config.MIN_LOCAL_SENSITIVITY_RECORDS,
        "passes": bool(
            damage_c["ci95"][0] > config.LOCAL_DAMAGE_THRESHOLD
            and damage_d["ci95"][0] > config.LOCAL_DAMAGE_THRESHOLD
            and damage_positive_count >= config.MIN_LOCAL_SENSITIVITY_RECORDS
        ),
        "per_record_damage_C": local_damage_c,
        "per_record_damage_D": local_damage_d,
    }
    controls = {
        "N_REPEAT": {
            "repeatability": repeatability,
            "passes": repeatability["passes"],
        },
        "LOCAL_SWAP": {
            "own_audio_validity": local_own_validity,
            "sensitivity": local_sensitivity,
            "passes": bool(local_own_validity["passes"] and local_sensitivity["passes"]),
        },
    }
    controls_pass = bool(controls["N_REPEAT"]["passes"] and controls["LOCAL_SWAP"]["passes"])
    bridge_compatibility = _compatibility(bridge_rows, baseline_offsets)
    bridge_movement = _movement([diagnostic_rows[str(record["sample_id"])] for record in records], config.BRIDGE_ARM)
    primary_gain = {
        "endpoint": bridge_compatibility["gap_C"],
        "threshold": config.PRIMARY_GAIN_THRESHOLD,
        "passes": bool(bridge_compatibility["gap_C"]["ci95"][0] > config.PRIMARY_GAIN_THRESHOLD),
    }
    bridge = {
        "movement": bridge_movement,
        "replacement_compatibility": bridge_compatibility,
        "primary_sync_C_gain": primary_gain,
        "passes": bool(bridge_movement["passes"] and bridge_compatibility["passes"] and primary_gain["passes"]),
    }
    if not controls_pass:
        decision = "CONTROL_FAILED"
    elif bridge["passes"]:
        decision = "NATURAL_TO_TTS_BRIDGE_CONFIRMED"
    else:
        decision = "NATURAL_TO_TTS_BRIDGE_NOT_CONFIRMED"
    analysis = {
        "schema_version": 1,
        "stage_id": "04_analysis",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "record_count": len(records),
        "source_group_count": len({str(record["source_group"]) for record in records}),
        "bootstrap": {"draws": config.BOOTSTRAP_DRAWS, "seed": config.BOOTSTRAP_SEED, "unit": "source_group_cluster", "confidence": 0.95, "interval": "two_sided_percentile"},
        "replacement_margin": config.REPLACEMENT_MARGIN,
        "offset_gate": {"tolerance_frames": config.OFFSET_TOLERANCE_FRAMES, "required_records": config.MIN_OFFSET_AGREEMENT_RECORDS},
        "controls": controls,
        "controls_pass": controls_pass,
        "bridge": bridge,
        "decisions": {
            "scientific_decision": decision,
            "reference_conditioned_audio_head_spec_eligible": decision == "NATURAL_TO_TTS_BRIDGE_CONFIRMED",
            "scope": "fixed 22-record/22-source-group LRS3 fit-only confirmation cohort and exact registered natural-to-TTS bridge construction",
        },
        "per_record": {"N_REPEAT": repeat_rows, "LOCAL_SWAP_replacement": local_natural_rows, "LOCAL_SWAP_own_audio": local_own_rows, "BRIDGE_075_replacement": bridge_rows},
        "descriptive_diagnostics": descriptive,
        "diagnostics_sha256": file_sha256(config.STAGES["01_audio"] / "diagnostics.json"),
        "matrix_manifest_sha256": file_sha256(config.STAGES["03_scores"] / "scores_manifest.json"),
    }
    write_self_hashed_json(output_stage / "analysis.json", analysis)
    return analysis
