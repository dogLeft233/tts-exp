from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    ProtocolError,
    assert_finite,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .scoring import reconstruct_global


def cluster_bootstrap(values: Sequence[float], groups: Sequence[str], *, seed: int = config.BOOTSTRAP_SEED, draws: int = config.BOOTSTRAP_DRAWS) -> dict[str, Any]:
    if len(values) != len(groups) or not values or draws <= 0:
        raise ValueError("bootstrap inputs are empty, mismatched, or invalid")
    by_group: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values, strict=True):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("bootstrap values must be finite")
        by_group[str(group)].append(number)
    labels = sorted(by_group)
    group_means = {label: float(np.mean(items)) for label, items in by_group.items()}
    rng = np.random.default_rng(seed)
    estimates = np.empty(int(draws), dtype=np.float64)
    for index in range(int(draws)):
        sampled = rng.choice(labels, size=len(labels), replace=True)
        estimates[index] = float(np.mean([group_means[label] for label in sampled]))
    return {
        "draws": int(draws),
        "seed": int(seed),
        "rng": "numpy_default_rng_pcg64",
        "labels": "sorted",
        "record_count": len(values),
        "source_group_count": len(labels),
        "mean": float(statistics.fmean(float(value) for value in values)),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "group_means": group_means,
    }


def _score_index(score_manifest: Mapping[str, Any], *, stage: str) -> dict[tuple[str, str, str, bool], Mapping[str, Any]]:
    expected = config.EXPECTED_CONTROL_SCORE_COUNT if stage == "control" else config.EXPECTED_BRIDGE_CELL_COUNT
    rows = score_manifest.get("scores")
    if score_manifest.get("status") != "complete" or score_manifest.get("stage") != stage or not isinstance(rows, list) or len(rows) != expected:
        raise ProtocolError(f"{stage} score manifest is incomplete")
    result: dict[tuple[str, str, str, bool], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProtocolError("score row is malformed")
        key = (str(row.get("sample_id")), str(row.get("video_arm")), str(row.get("audio_arm")), bool(row.get("repeat", False)))
        if key in result:
            raise ProtocolError(f"duplicate score cell: {key}")
        result[key] = row
    return result


def _cell(index: Mapping[tuple[str, str, str, bool], Mapping[str, Any]], sample_id: str, video_arm: str, audio_arm: str, repeat: bool = False) -> Mapping[str, Any]:
    key = (sample_id, video_arm, audio_arm, repeat)
    if key not in index:
        raise ProtocolError(f"score cell is missing: {key}")
    return index[key]


def _matrix(row: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(row.get("matrix", "")))
    if not path.is_file() or str(row.get("matrix_sha256")) != file_sha256(path):
        raise ProtocolError(f"score matrix is missing or changed: {path}")
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 31 or value.shape[0] < 1 or not np.isfinite(value).all():
        raise ProtocolError(f"score matrix is malformed: {path}")
    return value


def _evidence(record: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    matrix = _matrix(row)
    video_arm = str(row["video_arm"])
    frame_count = int(record["predicted_frame_counts"][video_arm])
    audio_count = int(record["natural_sample_count"])
    expected_rows = min(frame_count, audio_count // config.SAMPLES_PER_FRAME) - config.WINDOW_FRAMES
    if matrix.shape != (expected_rows, 31):
        raise ProtocolError(f"score matrix shape differs from protocol: {record.get('sample_id')}/{video_arm}: {matrix.shape}")
    masks = record.get("masks")
    if not isinstance(masks, Mapping):
        raise ProtocolError(f"timing masks are missing: {record.get('sample_id')}")
    common_rows = [int(value) for value in masks.get("common_window_rows", [])]
    if not common_rows or min(common_rows) < 0 or max(common_rows) >= matrix.shape[0]:
        raise ProtocolError(f"common window is outside matrix: {record.get('sample_id')}/{video_arm}")
    local = row.get("local")
    common = row.get("common_global")
    reconstructed = row.get("reconstructed")
    if not isinstance(local, Mapping) or not isinstance(common, Mapping) or not isinstance(reconstructed, Mapping):
        raise ProtocolError(f"score evidence is incomplete: {record.get('sample_id')}/{video_arm}")
    independent_common = reconstruct_global(matrix[common_rows, :])
    if abs(float(independent_common["sync_c"]) - float(common["sync_c"])) > 0.001 or abs(float(independent_common["sync_d"]) - float(common["sync_d"])) > 0.001 or int(independent_common["offset"]) != int(common["offset"]):
        raise ProtocolError(f"stored common-window result differs from matrix: {record.get('sample_id')}/{video_arm}")
    return {"matrix": matrix, "common": common, "reconstructed": reconstructed, "local": local}


def _baseline_info(left: Mapping[str, Any], right: Mapping[str, Any], left_evidence: Mapping[str, Any], right_evidence: Mapping[str, Any]) -> dict[str, Any]:
    left_matrix = left_evidence["matrix"]
    right_matrix = right_evidence["matrix"]
    shape_equal = left_matrix.shape == right_matrix.shape
    max_abs = float(np.max(np.abs(left_matrix - right_matrix))) if shape_equal else None
    offsets_equal = {
        "global": int(left_evidence["common"]["offset"]) == int(right_evidence["common"]["offset"]),
        "plus": int(left_evidence["local"]["PLUS"]["offset"]) == int(right_evidence["local"]["PLUS"]["offset"]),
        "minus": int(left_evidence["local"]["MINUS"]["offset"]) == int(right_evidence["local"]["MINUS"]["offset"]),
    }
    return {"matrix_shape_equal": shape_equal, "matrix_max_abs": max_abs, "offsets_equal": offsets_equal, "passes": bool(shape_equal and max_abs is not None and max_abs <= 0.001 and all(offsets_equal.values()))}


def _baseline_explainable(evidence: Mapping[str, Any]) -> dict[str, Any]:
    plus = evidence["local"]["PLUS"]
    minus = evidence["local"]["MINUS"]
    return {
        "plus": {key: plus[key] for key in ("offset", "peak_gap", "clear", "rows")},
        "minus": {key: minus[key] for key in ("offset", "peak_gap", "clear", "rows")},
        "direction_difference": abs(int(plus["offset"]) - int(minus["offset"])),
        "passes": bool(plus["clear"] and minus["clear"] and abs(int(plus["offset"]) - int(minus["offset"])) <= config.OFFSET_TOLERANCE_FRAMES),
    }


def _check_pair(left: Mapping[str, Any], right: Mapping[str, Any], expected: Mapping[str, Any], expected_keys: tuple[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, expected_key in zip(("PLUS", "MINUS"), expected_keys, strict=True):
        actual = float(left["local"][name]["offset"]) - float(right["local"][name]["offset"])
        target = float(expected[expected_key])
        result[name] = {"actual": actual, "expected": target, "error": actual - target, "passes": bool(left["local"][name]["clear"] and right["local"][name]["clear"] and abs(actual - target) <= config.OFFSET_TOLERANCE_FRAMES)}
    result["passes"] = bool(result["PLUS"]["passes"] and result["MINUS"]["passes"])
    return result


def _offset_noninferiority(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return abs(int(left["common"]["offset"]) - int(right["common"]["offset"])) <= config.OFFSET_TOLERANCE_FRAMES


def _gate_ci(value: Mapping[str, Any], *, lower: float | None = None, upper: float | None = None, strict_lower: bool = False, strict_upper: bool = False) -> bool:
    ci = value.get("ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        return False
    low, high = float(ci[0]), float(ci[1])
    lower_ok = True if lower is None else (low > lower if strict_lower else low >= lower)
    upper_ok = True if upper is None else (high < upper if strict_upper else high <= upper)
    return bool(lower_ok and upper_ok)


def _pair_bootstrap(rows: Sequence[Mapping[str, Any]], groups: Sequence[str], left: str, right: str, metric: str, *, direction: int = 1) -> dict[str, Any]:
    values = [direction * (float(row[left]["common"][metric]) - float(row[right]["common"][metric])) for row in rows]
    return cluster_bootstrap(values, groups)


def analyze_control(protocol: Mapping[str, Any], score_manifest: Mapping[str, Any]) -> dict[str, Any]:
    records = protocol.get("records")
    if not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("protocol record count is invalid")
    index = _score_index(score_manifest, stage="control")
    groups: list[str] = []
    per_record: list[dict[str, Any]] = []
    repeatability_count = 0
    r_baseline_count = 0
    gn_baseline_count = 0
    check_counts = {name: 0 for name in ("A", "B", "C", "O")}
    gnr_offset_count = 0
    gnr_c_values: list[float] = []
    gnr_d_values: list[float] = []
    own_c: list[float] = []
    own_d: list[float] = []
    own_offsets = 0
    damage_c: list[float] = []
    damage_d: list[float] = []
    damage_positive = 0
    for record in records:
        sample_id = str(record["sample_id"])
        group = str(record["source_group"])
        groups.append(group)
        names = {
            "rn": (config.VIDEO_R, config.AUDIO_N, False),
            "rw": (config.VIDEO_R, config.AUDIO_W, False),
            "gnn": (config.VIDEO_GN, config.AUDIO_N, False),
            "gnw": (config.VIDEO_GN, config.AUDIO_W, False),
            "gwn": (config.VIDEO_GW, config.AUDIO_N, False),
            "gww": (config.VIDEO_GW, config.AUDIO_W, False),
            "gnr": (config.VIDEO_GNR, config.AUDIO_N, False),
            "rn_repeat": (config.VIDEO_R, config.AUDIO_N, True),
            "gnn_repeat": (config.VIDEO_GN, config.AUDIO_N, True),
        }
        score_rows = {name: _cell(index, sample_id, video, audio, repeat) for name, (video, audio, repeat) in names.items()}
        ev = {name: _evidence(record, row) for name, row in score_rows.items()}
        r_repeat = _baseline_info(score_rows["rn"], score_rows["rn_repeat"], ev["rn"], ev["rn_repeat"])
        gn_repeat = _baseline_info(score_rows["gnn"], score_rows["gnn_repeat"], ev["gnn"], ev["gnn_repeat"])
        repeat_pass = bool(r_repeat["passes"] and gn_repeat["passes"])
        repeatability_count += int(repeat_pass)
        r_baseline = _baseline_explainable(ev["rn"])
        gn_baseline = _baseline_explainable(ev["gnn"])
        r_baseline_count += int(r_baseline["passes"])
        gn_baseline_count += int(gn_baseline["passes"])
        masks = record["masks"]
        check_a = _check_pair(ev["rw"], ev["rn"], masks, ("plus_expected_offset", "minus_expected_offset"))
        check_b = _check_pair(ev["gnw"], ev["gnn"], masks, ("plus_expected_offset", "minus_expected_offset"))
        check_c = _check_pair(ev["gwn"], ev["gnn"], masks, ("plus_expected_video_response", "minus_expected_video_response"))
        check_o = _check_pair(ev["gww"], ev["gnn"], {"plus_expected_offset": 0.0, "minus_expected_offset": 0.0}, ("plus_expected_offset", "minus_expected_offset"))
        for name, value, baseline in (("A", check_a, r_baseline), ("B", check_b, gn_baseline), ("C", check_c, gn_baseline), ("O", check_o, gn_baseline)):
            value["passes"] = bool(value["passes"] and baseline["passes"])
            check_counts[name] += int(value["passes"])
        gnr_c = float(ev["gnr"]["common"]["sync_c"]) - float(ev["gnn"]["common"]["sync_c"])
        gnr_d = float(ev["gnn"]["common"]["sync_d"]) - float(ev["gnr"]["common"]["sync_d"])
        gnr_c_values.append(gnr_c)
        gnr_d_values.append(gnr_d)
        gnr_offset_count += int(_offset_noninferiority(ev["gnr"], ev["gnn"]))
        own_c_value = float(ev["gww"]["common"]["sync_c"]) - float(ev["gnn"]["common"]["sync_c"])
        own_d_value = float(ev["gnn"]["common"]["sync_d"]) - float(ev["gww"]["common"]["sync_d"])
        own_c.append(own_c_value)
        own_d.append(own_d_value)
        own_offsets += int(_offset_noninferiority(ev["gww"], ev["gnn"]))
        damage_c_value = float(ev["gww"]["common"]["sync_c"]) - float(ev["gwn"]["common"]["sync_c"])
        damage_d_value = float(ev["gwn"]["common"]["sync_d"]) - float(ev["gww"]["common"]["sync_d"])
        damage_c.append(damage_c_value)
        damage_d.append(damage_d_value)
        damage_positive += int(damage_c_value > 0.0 and damage_d_value > 0.0)
        per_record.append({"sample_id": sample_id, "source_group": group, "repeatability": {"R_N": r_repeat, "G_N_N": gn_repeat, "passes": repeat_pass}, "baseline": {"R_N": r_baseline, "G_N_N": gn_baseline}, "checks": {"A": check_a, "B": check_b, "C": check_c, "O": check_o}, "generated_repeat": {"c_difference": gnr_c, "d_difference": gnr_d, "offset_difference": int(ev["gnr"]["common"]["offset"]) - int(ev["gnn"]["common"]["offset"])}, "own_audio": {"c": own_c_value, "d": own_d_value, "offset_noninferior": bool(_offset_noninferiority(ev["gww"], ev["gnn"]))}, "replacement_damage": {"c": damage_c_value, "d": damage_d_value, "both_positive": bool(damage_c_value > 0.0 and damage_d_value > 0.0)}})
    gnr_ci_c = cluster_bootstrap(gnr_c_values, groups)
    gnr_ci_d = cluster_bootstrap(gnr_d_values, groups)
    own_ci_c = cluster_bootstrap(own_c, groups)
    own_ci_d = cluster_bootstrap(own_d, groups)
    damage_ci_c = cluster_bootstrap(damage_c, groups)
    damage_ci_d = cluster_bootstrap(damage_d, groups)
    gates = {
        "repeatability": {"count": repeatability_count, "required": config.EXPECTED_RECORD_COUNT, "passes": repeatability_count == config.EXPECTED_RECORD_COUNT},
        "baseline": {"R_N_count": r_baseline_count, "G_N_N_count": gn_baseline_count, "required_each": config.MIN_BASELINE_RECORDS, "passes": r_baseline_count >= config.MIN_BASELINE_RECORDS and gn_baseline_count >= config.MIN_BASELINE_RECORDS},
        "generated_repeat": {"c_difference": gnr_ci_c, "d_difference_N_minus_NR": gnr_ci_d, "offset_count": gnr_offset_count, "required_offset_count": config.MIN_BASELINE_RECORDS, "ci_open_interval": bool(_gate_ci(gnr_ci_c, lower=-0.10, upper=0.10, strict_lower=True, strict_upper=True) and _gate_ci(gnr_ci_d, lower=-0.10, upper=0.10, strict_lower=True, strict_upper=True)), "passes": bool(_gate_ci(gnr_ci_c, lower=-0.10, upper=0.10, strict_lower=True, strict_upper=True) and _gate_ci(gnr_ci_d, lower=-0.10, upper=0.10, strict_lower=True, strict_upper=True) and gnr_offset_count >= config.MIN_BASELINE_RECORDS)},
        "inherited": {
            **{
                name: {
                    "count": check_counts[name],
                    "required": config.MIN_SUCCESS_RECORDS,
                    "denominator": config.EXPECTED_RECORD_COUNT,
                    "passes": check_counts[name] >= config.MIN_SUCCESS_RECORDS,
                }
                for name in ("A", "B", "C", "O")
            },
            "passes": all(check_counts[name] >= config.MIN_SUCCESS_RECORDS for name in ("A", "B", "C", "O")),
        },
        "own_audio": {"c": own_ci_c, "d": own_ci_d, "offset_count": own_offsets, "required_offset_count": config.MIN_BASELINE_RECORDS, "passes": bool(_gate_ci(own_ci_c, lower=-0.10, strict_lower=True) and _gate_ci(own_ci_d, lower=-0.10, strict_lower=True) and own_offsets >= config.MIN_BASELINE_RECORDS)},
        "replacement_damage": {"c": damage_ci_c, "d": damage_ci_d, "both_positive_count": damage_positive, "required_positive_count": config.MIN_SUCCESS_RECORDS, "passes": bool(_gate_ci(damage_ci_c, lower=0.10, strict_lower=True) and _gate_ci(damage_ci_d, lower=0.10, strict_lower=True) and damage_positive >= config.MIN_SUCCESS_RECORDS)},
    }
    control_pass = bool(all(bool(value["passes"]) for value in gates.values()))
    result = {
        "schema_version": 1,
        "analysis": "wav2lip_face_roi_replacement_control",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "engineering_decision": "GO",
        "scientific_decision": "CONTROL_PASS" if control_pass else "CONTROL_FAILED",
        "control_pass": control_pass,
        "record_count": config.EXPECTED_RECORD_COUNT,
        "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
        "score_count": config.EXPECTED_CONTROL_SCORE_COUNT,
        "main_score_count": config.EXPECTED_CONTROL_MAIN_CELL_COUNT,
        "repeat_score_count": config.EXPECTED_CONTROL_REPEAT_CELL_COUNT,
        "gates": gates,
        "per_record": per_record,
        "input_audit_sha256": str(protocol["input_audit"]["sha256"]),
        "score_manifest_sha256": str(score_manifest.get("artifact_sha256", "")),
        "training_authorized": False,
        "reference_conditioned_audio_head_spec_eligible": False,
        "generalization_established": False,
        "limits": ["seen-fit 22-record pilot", "ROI generation and tail_v2 scoring geometry are fixed contracts", "bootstrap CIs are exploratory precision gates, not a power claim", "control pass would only authorize this pilot's fixed bridge stage"],
    }
    assert_finite(result)
    return result


def analyze_bridge(protocol: Mapping[str, Any], control: Mapping[str, Any], control_scores: Mapping[str, Any], bridge_scores: Mapping[str, Any]) -> dict[str, Any]:
    if not bool(control.get("control_pass")):
        raise ProtocolError("bridge analysis is forbidden before control pass")
    records = protocol.get("records")
    if not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("protocol record count is invalid")
    control_index = _score_index(control_scores, stage="control")
    bridge_index = _score_index(bridge_scores, stage="bridge")
    groups: list[str] = []
    gain_c: list[float] = []
    gain_d: list[float] = []
    movement: list[float] = []
    per_record: list[dict[str, Any]] = []
    offset_count = 0
    movement_count = 0
    first_three: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        sample_id = str(record["sample_id"])
        group = str(record["source_group"])
        groups.append(group)
        base_row = _cell(control_index, sample_id, config.VIDEO_GN, config.AUDIO_N)
        bridge_n_row = _cell(bridge_index, sample_id, config.VIDEO_GB, config.AUDIO_N)
        bridge_row = _cell(bridge_index, sample_id, config.VIDEO_GB, config.AUDIO_BRIDGE)
        base = _evidence(record, base_row)
        bridge_n = _evidence(record, bridge_n_row)
        bridge_audio = _evidence(record, bridge_row)
        gain_c_value = float(bridge_n["common"]["sync_c"]) - float(base["common"]["sync_c"])
        gain_d_value = float(base["common"]["sync_d"]) - float(bridge_n["common"]["sync_d"])
        gain_c.append(gain_c_value)
        gain_d.append(gain_d_value)
        offset_ok = _offset_noninferiority(bridge_n, base)
        offset_count += int(offset_ok)
        audio_row = record.get("audio", {}).get("manifest_row")
        progress = None
        if isinstance(audio_row, Mapping) and isinstance(audio_row.get("mel_diagnostics"), Mapping):
            movement_info = audio_row["mel_diagnostics"].get("bridge_movement")
            if isinstance(movement_info, Mapping):
                progress = float(movement_info.get("progress", float("nan")))
        if progress is None or not np.isfinite(progress):
            raise ProtocolError(f"bridge movement is missing: {sample_id}")
        movement.append(progress)
        movement_ok = progress >= 0.15
        movement_count += int(movement_ok)
        per_record.append({"sample_id": sample_id, "source_group": group, "movement_progress": progress, "movement_pass": movement_ok, "gain_C": gain_c_value, "gain_D": gain_d_value, "offset_noninferior": offset_ok, "bridge_audio_descriptive": {"c_bridge_minus_n": float(bridge_audio["common"]["sync_c"]) - float(bridge_n["common"]["sync_c"]), "d_n_minus_bridge": float(bridge_n["common"]["sync_d"]) - float(bridge_audio["common"]["sync_d"])}})
        if index < 3:
            first_three.append({"sample_id": sample_id, "natural_audio": str(base_row["audio"]), "bridge_audio": str(bridge_row["audio"]), "natural_video": str(base_row["media"]), "bridge_video": str(bridge_row["media"]), "visual_quality": "UNREVIEWED"})
    movement_ci = cluster_bootstrap(movement, groups)
    gain_ci_c = cluster_bootstrap(gain_c, groups)
    gain_ci_d = cluster_bootstrap(gain_d, groups)
    gates = {
        "movement": {"count": movement_count, "required": config.MIN_BASELINE_RECORDS, "mean_ci": movement_ci, "passes": bool(movement_count >= config.MIN_BASELINE_RECORDS and _gate_ci(movement_ci, lower=0.15, strict_lower=True))},
        "natural_endpoint_noninferiority": {"gain_C": gain_ci_c, "gain_D": gain_ci_d, "passes": bool(_gate_ci(gain_ci_c, lower=-0.10, strict_lower=True) and _gate_ci(gain_ci_d, lower=-0.10, strict_lower=True))},
        "offset": {"count": offset_count, "required": config.MIN_BASELINE_RECORDS, "passes": offset_count >= config.MIN_BASELINE_RECORDS},
        "positive_primary_gain": {"gain_C": gain_ci_c, "passes": _gate_ci(gain_ci_c, lower=0.0, strict_lower=True)},
    }
    replacement_pass = bool(all(bool(value["passes"]) for value in gates.values()))
    result = {
        "schema_version": 1,
        "analysis": "wav2lip_face_roi_replacement_bridge",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "engineering_decision": "GO",
        "scientific_decision": "WAV2LIP_REPLACEMENT_PILOT_PASS" if replacement_pass else "REPLACEMENT_NOT_ESTABLISHED",
        "replacement_pass": replacement_pass,
        "record_count": config.EXPECTED_RECORD_COUNT,
        "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
        "score_count": config.EXPECTED_BRIDGE_CELL_COUNT,
        "gates": gates,
        "per_record": per_record,
        "first_three_playable_pairs": first_three,
        "visual_quality": "UNREVIEWED",
        "control_scientific_decision": control.get("scientific_decision"),
        "training_authorized": False,
        "reference_conditioned_audio_head_spec_eligible": False,
        "generalization_established": False,
        "cross_model_spec_eligible": replacement_pass,
        "limits": ["replacement result is only a seen-fit Wav2Lip pilot", "visual pairs are exported but not human-reviewed", "bridge self-audio condition is descriptive and not a primary gate"],
    }
    assert_finite(result)
    return result


def write_analysis(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    write_self_hashed_json(path, payload)
    return verify_self_hashed_json(path)
