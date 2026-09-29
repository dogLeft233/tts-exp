from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import config
from .common import CalibrationError, file_sha256, write_self_hashed_json


def cluster_bootstrap(values: Sequence[float], groups: Sequence[str], seed: int = config.BOOTSTRAP_SEED, draws: int = config.BOOTSTRAP_DRAWS) -> dict[str, Any]:
    if len(values) != len(groups) or not values or draws <= 0:
        raise ValueError("bootstrap inputs are empty, mismatched, or invalid")
    by_group: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("bootstrap values must be finite")
        by_group[str(group)].append(numeric)
    labels = sorted(by_group)
    group_means = {label: float(np.mean(items)) for label, items in by_group.items()}
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sampled = rng.choice(labels, size=len(labels), replace=True)
        means[index] = float(np.mean([group_means[label] for label in sampled]))
    return {
        "draws": int(draws),
        "seed": int(seed),
        "source_group_count": len(labels),
        "record_count": len(values),
        "mean": float(np.mean(values)),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "group_means": group_means,
    }


def endpoint(values: Sequence[float], groups: Sequence[str], field: str) -> dict[str, Any]:
    result = cluster_bootstrap(values, groups)
    result["field"] = field
    result["wins"] = {"positive_count": int(sum(float(value) > 0.0 for value in values)), "record_count": len(values)}
    return result


def _offset_gate(rows: Sequence[Mapping[str, Any]], baseline_offsets: Sequence[int]) -> dict[str, Any]:
    differences = [abs(int(row["av_offset"]) - int(base)) for row, base in zip(rows, baseline_offsets)]
    count = int(sum(value <= config.OFFSET_TOLERANCE_FRAMES for value in differences))
    return {
        "tolerance_frames": config.OFFSET_TOLERANCE_FRAMES,
        "count": count,
        "required": config.MIN_OFFSET_AGREEMENT_RECORDS,
        "differences": differences,
        "passes": count >= config.MIN_OFFSET_AGREEMENT_RECORDS,
    }


def _two_metric_gate(rows: Sequence[Mapping[str, Any]], groups: Sequence[str], baseline_offsets: Sequence[int], field_a: str, field_b: str, name: str) -> dict[str, Any]:
    metric_a = endpoint([float(row[field_a]) for row in rows], groups, field_a)
    metric_b = endpoint([float(row[field_b]) for row in rows], groups, field_b)
    offsets = _offset_gate(rows, baseline_offsets)
    return {
        field_a: metric_a,
        field_b: metric_b,
        "offset_agreement": offsets,
        "passes": bool(metric_a["ci95"][0] > config.REPLACEMENT_MARGIN and metric_b["ci95"][0] > config.REPLACEMENT_MARGIN and offsets["passes"]),
        "gate": name,
    }


def _score_rows(matrix: Mapping[str, Any], control_arm: str) -> dict[tuple[str, str], Mapping[str, Any]]:
    expected_cells = set(config.matrix_cells(control_arm))
    rows = matrix.get("scores")
    if matrix.get("status") != "complete" or not isinstance(rows, list) or len(rows) != config.expected_cell_count(config.SMOOTH_BRANCH):
        # Both branches have the same four-cell matrix size.
        raise CalibrationError("score manifest is incomplete")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("sample_id")), str(row.get("cell")))
        if key in result or key[1] not in expected_cells:
            raise CalibrationError(f"duplicate or unknown score cell: {key}")
        for field in ("sync_c", "sync_d", "av_offset"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
                raise CalibrationError(f"invalid score value: {key}/{field}")
        result[key] = row
    if len(result) != config.expected_cell_count(config.SMOOTH_BRANCH):
        raise CalibrationError("score cell identity is incomplete")
    return result


def _score(rows: Mapping[tuple[str, str], Mapping[str, Any]], sample_id: str, video_arm: str, audio_arm: str) -> Mapping[str, Any]:
    key = (sample_id, f"V_{video_arm}/A_{audio_arm}")
    if key not in rows:
        raise CalibrationError(f"required score cell is missing: {key}")
    return rows[key]


def _pair(score: Mapping[str, Any], baseline: Mapping[str, Any], sample_id: str, source_group: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "source_group": source_group,
        "sync_c": float(score["sync_c"]),
        "sync_d": float(score["sync_d"]),
        "av_offset": int(score["av_offset"]),
        "gap_c": float(score["sync_c"]) - float(baseline["sync_c"]),
        "gap_d": float(baseline["sync_d"]) - float(score["sync_d"]),
    }


def analyze_scores(cohort: Mapping[str, Any], protocol: Mapping[str, Any], score_manifest: Mapping[str, Any], output_stage) -> dict[str, Any]:
    branch = str(protocol.get("branch", ""))
    control_arm = config.control_arm_for_branch(branch)
    matrix = _score_rows(score_manifest, control_arm)
    records = list(cohort.get("records", []))
    if len(records) != config.EXPECTED_RECORD_COUNT:
        raise CalibrationError("cohort is incomplete")
    groups = [str(record["source_group"]) for record in records]
    baseline_offsets: list[int] = []
    repeat_rows: list[dict[str, Any]] = []
    own_rows: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, Any]] = []
    per_record: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        source_group = str(record["source_group"])
        baseline = _score(matrix, sample_id, config.NATURAL_ARM, config.NATURAL_ARM)
        repeat = _score(matrix, sample_id, config.REPEAT_ARM, config.NATURAL_ARM)
        own = _score(matrix, sample_id, control_arm, control_arm)
        replacement = _score(matrix, sample_id, control_arm, config.NATURAL_ARM)
        baseline_offsets.append(int(baseline["av_offset"]))
        repeat_rows.append(_pair(repeat, baseline, sample_id, source_group))
        own_rows.append(_pair(own, baseline, sample_id, source_group))
        replacement_rows.append(_pair(replacement, baseline, sample_id, source_group))
        per_record.append(
            {
                "sample_id": sample_id,
                "source_group": source_group,
                "baseline": {key: baseline[key] for key in ("sync_c", "sync_d", "av_offset")},
                "repeat": {key: repeat[key] for key in ("sync_c", "sync_d", "av_offset")},
                "control_own_audio": {key: own[key] for key in ("sync_c", "sync_d", "av_offset")},
                "control_natural_audio": {key: replacement[key] for key in ("sync_c", "sync_d", "av_offset")},
                "damage_C": float(own["sync_c"]) - float(replacement["sync_c"]),
                "damage_D": float(replacement["sync_d"]) - float(own["sync_d"]),
            }
        )
    repeatability = _two_metric_gate(repeat_rows, groups, baseline_offsets, "gap_c", "gap_d", "repeatability")
    own_validity = _two_metric_gate(own_rows, groups, baseline_offsets, "gap_c", "gap_d", "own_audio_validity")
    damage_c = [float(row["damage_C"]) for row in per_record]
    damage_d = [float(row["damage_D"]) for row in per_record]
    damage_c_stats = endpoint(damage_c, groups, "damage_C")
    damage_d_stats = endpoint(damage_d, groups, "damage_D")
    both_positive = int(sum(c > 0.0 and d > 0.0 for c, d in zip(damage_c, damage_d)))
    sensitivity = {
        "damage_C": damage_c_stats,
        "damage_D": damage_d_stats,
        "both_damage_positive_count": both_positive,
        "required_positive_count": config.MIN_LOCAL_SENSITIVITY_RECORDS,
        "passes": bool(damage_c_stats["ci95"][0] > config.LOCAL_DAMAGE_THRESHOLD and damage_d_stats["ci95"][0] > config.LOCAL_DAMAGE_THRESHOLD and both_positive >= config.MIN_LOCAL_SENSITIVITY_RECORDS),
        "per_record_damage_C": damage_c,
        "per_record_damage_D": damage_d,
    }
    controls = {
        "N_REPEAT": {"repeatability": repeatability, "passes": repeatability["passes"]},
        control_arm: {"own_audio_validity": own_validity, "sensitivity": sensitivity, "passes": bool(own_validity["passes"] and sensitivity["passes"])},
    }
    controls_pass = bool(repeatability["passes"] and own_validity["passes"] and sensitivity["passes"])
    decision = "CONTROL_CALIBRATED" if controls_pass else "CONTROL_FAILED"
    analysis = {
        "schema_version": 1,
        "stage_id": "05_analysis",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "classification": "fit_only_control_calibration",
        "branch": branch,
        "control_arm": control_arm,
        "record_count": len(records),
        "source_group_count": len(set(groups)),
        "bootstrap": {"draws": config.BOOTSTRAP_DRAWS, "seed": config.BOOTSTRAP_SEED, "unit": "source_group_cluster", "confidence": 0.95, "interval": "two_sided_percentile", "rng": "numpy_default_rng_pcg64", "labels": "sorted"},
        "controls": controls,
        "controls_pass": controls_pass,
        "decisions": {"engineering_decision": "GO", "scientific_decision": decision, "reference_conditioned_audio_head_spec_eligible": False},
        "per_record": per_record,
        "score_manifest_sha256": file_sha256(output_stage.parent / "04_scores/scores_manifest.json"),
    }
    output_stage.mkdir(parents=True, exist_ok=True)
    write_self_hashed_json(output_stage / "analysis.json", analysis)
    return analysis
