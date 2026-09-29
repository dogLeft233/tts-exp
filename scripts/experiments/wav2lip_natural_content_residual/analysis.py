from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import config
from .common import ProtocolError, assert_finite
from .scoring import gain


def _shared_bootstrap(values: Sequence[float], groups: Sequence[str], indices: np.ndarray) -> dict[str, Any]:
    if len(values) != len(groups) or not values:
        raise ProtocolError("bootstrap inputs are empty or mismatched")
    by_group: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        if not np.isfinite(float(value)):
            raise ProtocolError("bootstrap input is non-finite")
        by_group[str(group)].append(float(value))
    labels = sorted(by_group)
    if len(labels) != config.EXPECTED_GROUP_COUNT:
        raise ProtocolError("bootstrap group count is not 8")
    group_means = np.asarray([np.mean(by_group[label]) for label in labels], dtype=np.float64)
    estimates = np.mean(group_means[indices], axis=1)
    return {"mean": float(np.mean(group_means)), "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))], "group_labels": labels, "group_means": {label: float(value) for label, value in zip(labels, group_means, strict=True)}, "draws": int(len(estimates)), "seed": config.BOOTSTRAP_SEED, "rng": "numpy_default_rng_pcg64", "quantile_method": "linear"}


def bootstrap_indices(groups: Sequence[str]) -> tuple[list[str], np.ndarray]:
    labels = sorted({str(group) for group in groups})
    if len(labels) != config.EXPECTED_GROUP_COUNT:
        raise ProtocolError("expected eight source groups")
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    indices = rng.integers(0, len(labels), size=(config.BOOTSTRAP_DRAWS, len(labels)), endpoint=False)
    return labels, indices


def _contrast(values: Mapping[str, Sequence[float]], groups: Sequence[str], indices: np.ndarray) -> dict[str, Any]:
    metrics = {name: _shared_bootstrap(value, groups, indices) for name, value in values.items()}
    group_matrix = {}
    for group_index, group in enumerate(sorted(set(groups))):
        positions = [index for index, item in enumerate(groups) if item == group]
        group_matrix[group] = {name: float(np.mean([float(values[name][position]) for position in positions])) for name in values}
    positive = sum(all(group_matrix[group][name] > 0.0 for name in values) for group in group_matrix)
    return {"metrics": metrics, "group_joint_positive_count": int(positive), "group_count": len(group_matrix), "group_values": group_matrix}


def _ci_lower(result: Mapping[str, Any]) -> float:
    ci = result.get("ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        raise ProtocolError("missing bootstrap CI")
    return float(ci[0])


def _pass_contrast(result: Mapping[str, Any]) -> bool:
    metrics = result["metrics"]
    return bool(all(_ci_lower(metrics[name]) > 0.0 for name in ("C", "D", "A")) and int(result["group_joint_positive_count"]) >= 7)


def analyze_controls(protocol: Mapping[str, Any], score_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {(str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row for row in score_rows}
    if len(by_key) != config.EXPECTED_CONTROL_SCORE_COUNT:
        raise ProtocolError(f"control score count is {len(by_key)}, expected {config.EXPECTED_CONTROL_SCORE_COUNT}")
    records = protocol["records"]
    parity_rows = [row for row in score_rows if str(row.get("video_arm", "")).startswith("PARITY_")]
    parity_pass = all(bool(row.get("parity", {}).get("passes")) for row in parity_rows) and len(parity_rows) == 2
    groups: list[str] = []
    delay_anchor: list[float] = []
    delay_offset_differences: list[int] = []
    repeat_differences: list[dict[str, float]] = []
    per_record: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        group = str(record["source_group"])
        groups.append(group)
        n = by_key[(sample_id, "N", "N")]
        delay = by_key[(sample_id, "N", "A_DELAY")]
        repeat = by_key.get((sample_id, "N_REPEAT", "N"))
        n_u = n["U"]
        d_u = delay["U"]
        k_n = int(n_u["min_index"])
        anchor = float(d_u["curve"][k_n] - n_u["curve"][k_n])
        offset_diff = int(d_u["offset"]) - int(n_u["offset"])
        delay_anchor.append(anchor)
        delay_offset_differences.append(offset_diff)
        repeat_info: dict[str, Any] = {"available": False}
        if repeat is not None:
            r_u = repeat["U"]
            repeat_diff = {"C": float(r_u["C"] - n_u["C"]), "D": float(r_u["D"] - n_u["D"]), "A": float(r_u["curve"][k_n] - n_u["curve"][k_n])}
            repeat_differences.append(repeat_diff)
            repeat_info = {"available": True, "difference": repeat_diff, "abs_max": max(abs(repeat_diff[name]) for name in repeat_diff), "offset_equal": int(r_u["offset"]) == int(n_u["offset"])}
        per_record.append({"sample_id": sample_id, "source_group": group, "k_N": k_n, "delay_anchor_damage": anchor, "delay_offset_difference": offset_diff, "repeat": repeat_info})
    labels, indices = bootstrap_indices(groups)
    delay_bootstrap = _shared_bootstrap(delay_anchor, groups, indices)
    delay_group_positive = sum(float(value) > 0 for value in delay_bootstrap["group_means"].values())
    repeat_pass = bool(len(repeat_differences) == 2 and all(max(abs(value[name]) for name in value) <= 1e-4 for value in repeat_differences) and sum(1 for item in per_record if item["repeat"].get("available")) == 2)
    offset_pass = bool(sum(-6 <= value <= -4 for value in delay_offset_differences) >= 14)
    anchor_pass = bool(_ci_lower(delay_bootstrap) > 0.0 and delay_group_positive >= 7)
    control_pass = bool(parity_pass and repeat_pass and anchor_pass and offset_pass)
    result = {"schema_version": 1, "protocol_id": "wav2lip_natural_content_residual", "stage": "A", "engineering_status": "GO" if parity_pass and repeat_pass else "BLOCKED", "scientific_decision": "CONTROL_PASS" if control_pass else "CONTROL_FAILED", "control_pass": control_pass, "stage_b_authorized": control_pass, "counts": {"parity": len(parity_rows), "natural": 16, "repeat": 2, "delay": 16, "total": len(score_rows)}, "parity": {"count": len(parity_rows), "passes": parity_pass, "rows": parity_rows}, "repeatability": {"passes": repeat_pass, "per_record": per_record}, "delay_sensitivity": {"anchor_damage": delay_bootstrap, "group_positive_count": delay_group_positive, "offset_differences": delay_offset_differences, "offset_pass_count": sum(-6 <= value <= -4 for value in delay_offset_differences), "anchor_pass": anchor_pass, "offset_pass": offset_pass}, "budget": {"videos": 18, "scores": len(score_rows), "max_videos": config.MAX_VIDEO_COUNT, "max_scores": config.MAX_SCORE_COUNT}, "limits": ["Stage A control only", "free-offset compensation is not a failure", "no replacement confirmation"]}
    assert_finite(result)
    return result


def analyze_candidates(protocol: Mapping[str, Any], driver_manifest: Mapping[str, Any], control: Mapping[str, Any], score_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not bool(control.get("stage_b_authorized")):
        raise ProtocolError("candidate analysis is unauthorized because Stage A failed")
    score_index = {(str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row for row in score_rows}
    driver_index = {str(row["sample_id"]): row for row in driver_manifest["rows"]}
    if len(score_index) != config.EXPECTED_CANDIDATE_SCORE_COUNT + config.EXPECTED_CONTROL_SCORE_COUNT:
        raise ProtocolError(f"candidate analysis score count is {len(score_index)}, expected {config.EXPECTED_CANDIDATE_SCORE_COUNT + config.EXPECTED_CONTROL_SCORE_COUNT}")
    baseline_index = {(str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row for row in []}
    # The baseline rows are supplied in score_rows with the N arm from Stage A
    # by runner._merge_score_rows; retain the explicit branch for auditability.
    for row in score_rows:
        if str(row.get("video_arm")) == "N":
            baseline_index[(str(row["sample_id"]), "N", "N")] = row
    if len(baseline_index) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("candidate analysis did not receive all natural baselines")
    groups = [str(record["source_group"]) for record in protocol["records"]]
    labels, indices = bootstrap_indices(groups)
    values: dict[str, dict[str, list[float]]] = {"CORRECT_vs_N": {name: [] for name in ("C", "D", "A")}, "CORRECT_vs_WRONG": {name: [] for name in ("C", "D", "A")}, "CORRECT_vs_SHUFFLE": {name: [] for name in ("C", "D", "A")}}
    per_record: list[dict[str, Any]] = []
    norm_valid = True
    for record in protocol["records"]:
        sample_id = str(record["sample_id"])
        row_driver = driver_index[sample_id]
        norm_valid = norm_valid and bool(row_driver["construct"]["norm_control_valid"])
        baseline = baseline_index[(sample_id, "N", "N")]
        candidate_rows = {arm: score_index[(sample_id, arm, "N")] for arm in config.CANDIDATE_ARMS}
        gains_n = {arm: gain(candidate_rows[arm], baseline) for arm in config.CANDIDATE_ARMS}
        metrics_correct_wrong = {name: float(candidate_rows["CORRECT"]["U"][name] - candidate_rows["WRONG"]["U"][name]) for name in ("C",)}
        k_n = int(baseline["U"]["min_index"])
        metrics_correct_wrong.update({"D": float(candidate_rows["WRONG"]["U"]["D"] - candidate_rows["CORRECT"]["U"]["D"]), "A": float(candidate_rows["WRONG"]["U"]["curve"][k_n] - candidate_rows["CORRECT"]["U"]["curve"][k_n])})
        metrics_correct_shuffle = {"C": float(candidate_rows["CORRECT"]["U"]["C"] - candidate_rows["SHUFFLE"]["U"]["C"]), "D": float(candidate_rows["SHUFFLE"]["U"]["D"] - candidate_rows["CORRECT"]["U"]["D"]), "A": float(candidate_rows["SHUFFLE"]["U"]["curve"][k_n] - candidate_rows["CORRECT"]["U"]["curve"][k_n])}
        for name in ("C", "D", "A"):
            values["CORRECT_vs_N"][name].append(gains_n["CORRECT"][name])
            values["CORRECT_vs_WRONG"][name].append(metrics_correct_wrong[name])
            values["CORRECT_vs_SHUFFLE"][name].append(metrics_correct_shuffle[name])
        per_record.append({"sample_id": sample_id, "source_group": str(record["source_group"]), "k_N": k_n, "gains_vs_N": gains_n, "correct_vs_wrong": metrics_correct_wrong, "correct_vs_shuffle": metrics_correct_shuffle, "norm_control_valid": bool(row_driver["construct"]["norm_control_valid"])})
    contrasts = {name: _contrast(values[name], groups, indices) for name in values}
    primary = contrasts["CORRECT_vs_N"]
    primary_pass = bool(_pass_contrast(primary) and float(primary["metrics"]["C"]["mean"]) > 0.05)
    content_signal = bool(primary_pass and norm_valid and _pass_contrast(contrasts["CORRECT_vs_WRONG"]) and _pass_contrast(contrasts["CORRECT_vs_SHUFFLE"]))
    if not primary_pass:
        decision = "NO_INCREMENT_ESTABLISHED"
    elif content_signal:
        decision = "CONTENT_RESIDUAL_SIGNAL"
    else:
        decision = "NATURAL_GAIN_MECHANISM_UNRESOLVED"
    result = {"schema_version": 1, "protocol_id": "wav2lip_natural_content_residual", "stage": "B", "scientific_decision": decision, "primary_pass": primary_pass, "norm_control_valid": norm_valid, "content_signal": content_signal, "contrasts": contrasts, "per_record": per_record, "shared_bootstrap": {"labels": labels, "seed": config.BOOTSTRAP_SEED, "draws": config.BOOTSTRAP_DRAWS, "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()}, "limits": ["seen records only", "3.84-second mel support", "exploratory, not independent confirmation", "replacement_confirmed=false", "waveform_head_authorized=false", "generalization_established=false", "historical_shift_gate_repaired=false"]}
    assert_finite(result)
    return result
