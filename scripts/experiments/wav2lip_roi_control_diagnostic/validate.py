from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .analysis import load_parent
from .common import (
    DiagnosticError,
    compare_values,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def _timing(sample_count: int, frame_counts: Mapping[str, Any]) -> dict[str, Any]:
    n = np.arange(sample_count, dtype=np.float64)
    mapped = n + config.WARP_AMPLITUDE_SAMPLES * np.sin(2.0 * np.pi * n / float(sample_count - 1))
    mapped[0] = 0.0
    mapped[-1] = float(sample_count - 1)
    q = min(*(int(frame_counts[arm]) for arm in config.VIDEO_ARMS), sample_count // config.SAMPLES_PER_FRAME)
    candidate = list(range(config.VSHIFT, q - 20))
    plus: list[int] = []
    minus: list[int] = []
    d_by_row: dict[str, float] = {}
    a_by_row: dict[str, float] = {}
    for row in candidate:
        center = float(config.SAMPLES_PER_FRAME * (row + 2))
        d = (float(np.interp(center, n, mapped)) - center) / config.SAMPLES_PER_FRAME
        a = (center - float(np.interp(center, mapped, n))) / config.SAMPLES_PER_FRAME
        d_by_row[str(row)] = d
        a_by_row[str(row)] = a
        if d >= 2.5 and a >= 2.5:
            plus.append(row)
        if d <= -2.5 and a <= -2.5:
            minus.append(row)
    if len(plus) < config.MIN_LOCAL_ROWS or len(minus) < config.MIN_LOCAL_ROWS:
        raise DiagnosticError("validator timing support is too small")
    return {
        "sample_count": int(sample_count),
        "q_frames": int(q),
        "candidate_rows": candidate,
        "common_window_rows": list(range(q - config.WINDOW_FRAMES)),
        "plus_rows": plus,
        "minus_rows": minus,
        "d_by_row": d_by_row,
        "a_by_row": a_by_row,
        "plus_expected_offset": float(np.mean([a_by_row[str(row)] for row in plus])),
        "minus_expected_offset": float(np.mean([a_by_row[str(row)] for row in minus])),
        "plus_expected_video_response": float(-np.mean([d_by_row[str(row)] for row in plus])),
        "minus_expected_video_response": float(-np.mean([d_by_row[str(row)] for row in minus])),
        "forward_mapping_sha256": hashlib.sha256(np.asarray(mapped, dtype="<f8").tobytes()).hexdigest(),
        "forward_mapping_dtype": "float64-little-endian",
        "forward_mapping_length": int(mapped.size),
    }


def _curve(matrix: np.ndarray, rows: Sequence[int] | None = None) -> dict[str, Any]:
    selected = list(range(matrix.shape[0])) if rows is None else [int(row) for row in rows]
    curve = np.mean(matrix[np.asarray(selected, dtype=np.int64), :], axis=0)
    index = int(np.argmin(curve))
    ordered = np.sort(curve, kind="stable")
    minimum = float(curve[index])
    second = float(ordered[1])
    offset = config.VSHIFT - index
    return {
        "rows": selected,
        "curve": [float(value) for value in curve],
        "offsets": [config.VSHIFT - item for item in range(curve.size)],
        "min_index": index,
        "offset": int(offset),
        "sync_d": minimum,
        "sync_c": float(np.median(curve) - minimum),
        "min": minimum,
        "second_min": second,
        "peak_gap": second - minimum,
        "clear": bool(second - minimum > config.PEAK_GAP_THRESHOLD and offset not in (-15, 15)),
    }


def _evidence(parent: Mapping[str, Any], row: Mapping[str, Any], record: Mapping[str, Any], timing: Mapping[str, Any], video_arm: str) -> dict[str, Any]:
    matrix = parent["matrix_cache"][str(row["matrix_sha256"])]
    frame_count = int(record["predicted_frame_counts"][video_arm])
    expected_rows = min(frame_count, int(record["natural_sample_count"]) // config.SAMPLES_PER_FRAME) - config.WINDOW_FRAMES
    if matrix.shape != (expected_rows, 31):
        raise DiagnosticError(f"validator matrix shape mismatch: {record['sample_id']}/{video_arm}")
    common = list(timing["common_window_rows"])
    return {
        "matrix": matrix,
        "reconstructed": _curve(matrix),
        "common_global": _curve(matrix, common),
        "local": {
            "supported_rows": list(range(matrix.shape[0])),
            "common_window_rows": common,
            "PLUS": _curve(matrix, timing["plus_rows"]),
            "MINUS": _curve(matrix, timing["minus_rows"]),
        },
    }


def _bootstrap(values: Sequence[float], groups: Sequence[str]) -> dict[str, Any]:
    by_group: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values, strict=True):
        by_group[str(group)].append(float(value))
    labels = sorted(by_group)
    means = {label: float(np.mean(items)) for label, items in by_group.items()}
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    estimates = np.empty(config.BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(config.BOOTSTRAP_DRAWS):
        sampled = rng.choice(labels, size=len(labels), replace=True)
        estimates[index] = float(np.mean([means[label] for label in sampled]))
    return {
        "draws": config.BOOTSTRAP_DRAWS,
        "seed": config.BOOTSTRAP_SEED,
        "rng": "numpy_default_rng_pcg64",
        "labels": "sorted",
        "record_count": len(values),
        "source_group_count": len(labels),
        "mean": float(statistics.fmean(float(value) for value in values)),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "group_means": means,
    }


def _check_pair(left: Mapping[str, Any], right: Mapping[str, Any], timing: Mapping[str, Any], keys: tuple[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, expected_key in zip(("PLUS", "MINUS"), keys, strict=True):
        actual = float(left["local"][name]["offset"]) - float(right["local"][name]["offset"])
        expected = float(timing[expected_key])
        result[name] = {"actual": actual, "expected": expected, "error": actual - expected, "passes": bool(left["local"][name]["clear"] and right["local"][name]["clear"] and abs(actual - expected) <= config.OFFSET_TOLERANCE_FRAMES)}
    result["passes"] = bool(result["PLUS"]["passes"] and result["MINUS"]["passes"])
    return result


def _baseline(evidence: Mapping[str, Any]) -> dict[str, Any]:
    plus = evidence["local"]["PLUS"]
    minus = evidence["local"]["MINUS"]
    return {"plus": {key: plus[key] for key in ("offset", "peak_gap", "clear", "rows")}, "minus": {key: minus[key] for key in ("offset", "peak_gap", "clear", "rows")}, "direction_difference": abs(int(plus["offset"]) - int(minus["offset"])), "passes": bool(plus["clear"] and minus["clear"] and abs(int(plus["offset"]) - int(minus["offset"])) <= config.OFFSET_TOLERANCE_FRAMES)}


def _pair_baseline(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    same_shape = left["matrix"].shape == right["matrix"].shape
    max_abs = float(np.max(np.abs(left["matrix"] - right["matrix"]))) if same_shape else None
    equal = {"global": left["common_global"]["offset"] == right["common_global"]["offset"], "plus": left["local"]["PLUS"]["offset"] == right["local"]["PLUS"]["offset"], "minus": left["local"]["MINUS"]["offset"] == right["local"]["MINUS"]["offset"]}
    return {"matrix_shape_equal": same_shape, "matrix_max_abs": max_abs, "offsets_equal": equal, "passes": bool(same_shape and max_abs is not None and max_abs <= 0.001 and all(equal.values()))}


def _noninferior(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return abs(int(left["common_global"]["offset"]) - int(right["common_global"]["offset"])) <= config.OFFSET_TOLERANCE_FRAMES


def _ci_gate(value: Mapping[str, Any], lower: float | None = None, upper: float | None = None, strict_lower: bool = False, strict_upper: bool = False) -> bool:
    ci = value["ci95"]
    return bool((lower is None or (ci[0] > lower if strict_lower else ci[0] >= lower)) and (upper is None or (ci[1] < upper if strict_upper else ci[1] <= upper)))


def _independent_core(parent: Mapping[str, Any]) -> dict[str, Any]:
    records = parent["records"]
    score_index = parent["score_index"]
    groups: list[str] = []
    rows: list[dict[str, Any]] = []
    own_rows: list[dict[str, Any]] = []
    c_flags = {name: [] for name in ("baseline_invalid", "boundary_peak", "unclear_peak", "offset_error")}
    checks = {name: 0 for name in ("A", "B", "C", "O")}
    counts = {name: 0 for name in ("repeatability", "r_baseline", "gn_baseline", "gnr_offset", "own_offset", "damage_positive")}
    gnr_c: list[float] = []
    gnr_d: list[float] = []
    own_c: list[float] = []
    own_d: list[float] = []
    damage_c: list[float] = []
    damage_d: list[float] = []
    for record in records:
        sample_id = str(record["sample_id"])
        group = str(record["source_group"])
        groups.append(group)
        timing = _timing(int(record["natural_sample_count"]), record["predicted_frame_counts"])
        names = {"rn": ("R", "N", False), "rw": ("R", "W", False), "gnn": ("G_N", "N", False), "gnw": ("G_N", "W", False), "gwn": ("G_W", "N", False), "gww": ("G_W", "W", False), "gnr": ("G_NR", "N", False), "rn_repeat": ("R", "N", True), "gnn_repeat": ("G_N", "N", True)}
        ev = {name: _evidence(parent, score_index[(sample_id, video, audio, repeat)], record, timing, video) for name, (video, audio, repeat) in names.items()}
        repeat_r = _pair_baseline(ev["rn"], ev["rn_repeat"])
        repeat_gn = _pair_baseline(ev["gnn"], ev["gnn_repeat"])
        repeat_pass = bool(repeat_r["passes"] and repeat_gn["passes"])
        counts["repeatability"] += int(repeat_pass)
        baseline_r = _baseline(ev["rn"])
        baseline_gn = _baseline(ev["gnn"])
        counts["r_baseline"] += int(baseline_r["passes"])
        counts["gn_baseline"] += int(baseline_gn["passes"])
        check_a = _check_pair(ev["rw"], ev["rn"], timing, ("plus_expected_offset", "minus_expected_offset"))
        check_b = _check_pair(ev["gnw"], ev["gnn"], timing, ("plus_expected_offset", "minus_expected_offset"))
        check_c = _check_pair(ev["gwn"], ev["gnn"], timing, ("plus_expected_video_response", "minus_expected_video_response"))
        check_o = _check_pair(ev["gww"], ev["gnn"], {"plus_expected_offset": 0.0, "minus_expected_offset": 0.0}, ("plus_expected_offset", "minus_expected_offset"))
        for name, value, base in (("A", check_a, baseline_r), ("B", check_b, baseline_gn), ("C", check_c, baseline_gn), ("O", check_o, baseline_gn)):
            value["passes"] = bool(value["passes"] and base["passes"])
            checks[name] += int(value["passes"])
        segments: dict[str, Any] = {}
        flags = {"baseline_invalid": not baseline_gn["passes"], "boundary_peak": False, "unclear_peak": False, "offset_error": False}
        for name, expected_key in (("PLUS", "plus_expected_video_response"), ("MINUS", "minus_expected_video_response")):
            left, right = ev["gwn"]["local"][name], ev["gnn"]["local"][name]
            actual = float(left["offset"] - right["offset"])
            expected = float(timing[expected_key])
            residual = actual - expected
            segment_flags = {"boundary_peak": abs(int(left["offset"])) == 15 or abs(int(right["offset"])) == 15, "unclear_peak": left["peak_gap"] <= config.PEAK_GAP_THRESHOLD or right["peak_gap"] <= config.PEAK_GAP_THRESHOLD, "offset_error": abs(residual) > config.OFFSET_TOLERANCE_FRAMES}
            for key, value in segment_flags.items():
                flags[key] = bool(flags[key] or value)
            segments[name] = {"rows": list(right["rows"]), "gnn_curve": list(right["curve"]), "gwn_curve": list(left["curve"]), "gnn": {key: right[key] for key in ("offset", "peak_gap", "clear")}, "gwn": {key: left[key] for key in ("offset", "peak_gap", "clear")}, "expected": expected, "actual": actual, "residual": residual, "flags": segment_flags}
        c_diag = {"segments": segments, "flags": flags, "passes": bool(check_c["passes"] and baseline_gn["passes"])}
        for flag, present in flags.items():
            if present:
                c_flags[flag].append(sample_id)
        gnr_c_value = float(ev["gnr"]["common_global"]["sync_c"] - ev["gnn"]["common_global"]["sync_c"])
        gnr_d_value = float(ev["gnn"]["common_global"]["sync_d"] - ev["gnr"]["common_global"]["sync_d"])
        gnr_c.append(gnr_c_value)
        gnr_d.append(gnr_d_value)
        gnr_offset = int(ev["gnr"]["common_global"]["offset"] - ev["gnn"]["common_global"]["offset"])
        counts["gnr_offset"] += int(abs(gnr_offset) <= config.OFFSET_TOLERANCE_FRAMES)
        own_c_value = float(ev["gww"]["common_global"]["sync_c"] - ev["gnn"]["common_global"]["sync_c"])
        own_d_value = float(ev["gnn"]["common_global"]["sync_d"] - ev["gww"]["common_global"]["sync_d"])
        n_curve, w_curve = ev["gnn"]["common_global"]["curve"], ev["gww"]["common_global"]["curve"]
        median_change = float(np.median(w_curve) - np.median(n_curve))
        identity_error = own_c_value - median_change - own_d_value
        if abs(identity_error) > 1e-9:
            raise DiagnosticError(f"validator own-audio identity failed: {sample_id}")
        own_c.append(own_c_value)
        own_d.append(own_d_value)
        own_offset = _noninferior(ev["gww"], ev["gnn"])
        counts["own_offset"] += int(own_offset)
        own_rows.append({"sample_id": sample_id, "source_group": group, "N_curve": list(n_curve), "W_curve": list(w_curve), "median_N": float(np.median(n_curve)), "median_W": float(np.median(w_curve)), "median_change": median_change, "own_C": own_c_value, "own_D": own_d_value, "identity_error": identity_error, "N": {key: ev["gnn"]["common_global"][key] for key in ("offset", "sync_c", "sync_d")}, "W": {key: ev["gww"]["common_global"][key] for key in ("offset", "sync_c", "sync_d")}, "offset_noninferior": own_offset})
        damage_c_value = float(ev["gww"]["common_global"]["sync_c"] - ev["gwn"]["common_global"]["sync_c"])
        damage_d_value = float(ev["gwn"]["common_global"]["sync_d"] - ev["gww"]["common_global"]["sync_d"])
        damage_c.append(damage_c_value)
        damage_d.append(damage_d_value)
        damage_positive = damage_c_value > 0.0 and damage_d_value > 0.0
        counts["damage_positive"] += int(damage_positive)
        rows.append({"sample_id": sample_id, "source_group": group, "repeatability": {"R_N": repeat_r, "G_N_N": repeat_gn, "passes": repeat_pass}, "baseline": {"R_N": baseline_r, "G_N_N": baseline_gn}, "checks": {"A": check_a, "B": check_b, "C": check_c, "O": check_o}, "generated_repeat": {"c_difference": gnr_c_value, "d_difference": gnr_d_value, "offset_difference": gnr_offset}, "own_audio": {"c": own_c_value, "d": own_d_value, "offset_noninferior": own_offset}, "replacement_damage": {"c": damage_c_value, "d": damage_d_value, "both_positive": damage_positive}, "c_diagnostic": c_diag})
    gnr_ci_c, gnr_ci_d = _bootstrap(gnr_c, groups), _bootstrap(gnr_d, groups)
    own_ci_c, own_ci_d = _bootstrap(own_c, groups), _bootstrap(own_d, groups)
    damage_ci_c, damage_ci_d = _bootstrap(damage_c, groups), _bootstrap(damage_d, groups)
    gates = {"repeatability": {"count": counts["repeatability"], "required": config.EXPECTED_RECORD_COUNT, "passes": counts["repeatability"] == config.EXPECTED_RECORD_COUNT}, "baseline": {"R_N_count": counts["r_baseline"], "G_N_N_count": counts["gn_baseline"], "required_each": config.MIN_BASELINE_RECORDS, "passes": counts["r_baseline"] >= config.MIN_BASELINE_RECORDS and counts["gn_baseline"] >= config.MIN_BASELINE_RECORDS}, "generated_repeat": {"c_difference": gnr_ci_c, "d_difference_N_minus_NR": gnr_ci_d, "offset_count": counts["gnr_offset"], "required_offset_count": config.MIN_BASELINE_RECORDS, "ci_open_interval": bool(_ci_gate(gnr_ci_c, -0.10, 0.10, True, True) and _ci_gate(gnr_ci_d, -0.10, 0.10, True, True)), "passes": bool(_ci_gate(gnr_ci_c, -0.10, 0.10, True, True) and _ci_gate(gnr_ci_d, -0.10, 0.10, True, True) and counts["gnr_offset"] >= config.MIN_BASELINE_RECORDS)}, "inherited": {**{name: {"count": checks[name], "required": config.MIN_SUCCESS_RECORDS, "denominator": config.EXPECTED_RECORD_COUNT, "passes": checks[name] >= config.MIN_SUCCESS_RECORDS} for name in ("A", "B", "C", "O")}, "passes": all(checks[name] >= config.MIN_SUCCESS_RECORDS for name in ("A", "B", "C", "O"))}, "own_audio": {"c": own_ci_c, "d": own_ci_d, "offset_count": counts["own_offset"], "required_offset_count": config.MIN_BASELINE_RECORDS, "passes": bool(_ci_gate(own_ci_c, -0.10, None, True) and _ci_gate(own_ci_d, -0.10, None, True) and counts["own_offset"] >= config.MIN_BASELINE_RECORDS)}, "replacement_damage": {"c": damage_ci_c, "d": damage_ci_d, "both_positive_count": counts["damage_positive"], "required_positive_count": config.MIN_SUCCESS_RECORDS, "passes": bool(_ci_gate(damage_ci_c, 0.10, None, True) and _ci_gate(damage_ci_d, 0.10, None, True) and counts["damage_positive"] >= config.MIN_SUCCESS_RECORDS)}}
    return {"groups": groups, "rows": rows, "own_rows": own_rows, "gates": gates, "c_flags": c_flags, "control_pass": bool(all(bool(item["passes"]) for item in gates.values()))}


def _stored_score_differences(parent: Mapping[str, Any]) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    for record in parent["records"]:
        sample_id = str(record["sample_id"])
        timing = _timing(int(record["natural_sample_count"]), record["predicted_frame_counts"])
        for video, audio, repeat in config.ALL_CELL_SPECS:
            key = (sample_id, video, audio, repeat)
            row = parent["score_index"][key]
            evidence = _evidence(parent, row, record, timing, video)
            compare_values(row.get("reconstructed"), evidence["reconstructed"], f"score[{key}].reconstructed", differences)
            compare_values(row.get("common_global"), evidence["common_global"], f"score[{key}].common_global", differences)
            compare_values(row.get("local"), evidence["local"], f"score[{key}].local", differences)
    return differences


def _check_final_contract(root: Path, final: Mapping[str, Any]) -> None:
    for name in ("protocol", "audit", "diagnostics", "result"):
        path = root / f"{name}.json" if name != "result" else root / "result.md"
        if file_sha256(path) != str(final.get(f"{name}_sha256")):
            raise DiagnosticError(f"final/{name} hash binding changed")
    if final.get("new_generated_videos") != 0 or final.get("new_score_cells") != 0 or final.get("bridge_executed") is not False:
        raise DiagnosticError("final records forbidden new work")
    for key in ("training_authorized", "cross_model_spec_eligible", "generalization_established"):
        if final.get(key) is not False:
            raise DiagnosticError(f"final unsafe flag is not false: {key}")


def validate_run(run_root: Path) -> dict[str, Any]:
    root = run_root.resolve()
    final = verify_self_hashed_json(root / "final.json")
    if final.get("status") == "blocked":
        if final.get("diagnostic_decision") != "BLOCKED":
            raise DiagnosticError("blocked final marker is malformed")
        result = {"status": "valid", "mode": "blocked", "diagnostic_decision": "BLOCKED", "final_sha256": file_sha256(root / "final.json")}
        write_self_hashed_json(root / "validation.json", {"schema_version": 1, "stage_id": "validation", "protocol_id": config.PROTOCOL_ID, **result})
        return result
    if final.get("status") != "complete" or final.get("protocol_id") != config.PROTOCOL_ID:
        raise DiagnosticError("complete diagnostic final marker is malformed")
    _check_final_contract(root, final)
    protocol = verify_self_hashed_json(root / "protocol.json")
    audit = verify_self_hashed_json(root / "audit.json")
    diagnostics = verify_self_hashed_json(root / "diagnostics.json")
    if protocol.get("status") != "locked" or audit.get("status") != "complete" or diagnostics.get("status") != "complete":
        raise DiagnosticError("diagnostic artifacts are not complete")
    if protocol.get("audit_sha256") != file_sha256(root / "audit.json") or diagnostics.get("protocol_sha256") != file_sha256(root / "protocol.json"):
        raise DiagnosticError("diagnostic artifact chain is broken")
    for key, expected in (("record_count", config.EXPECTED_RECORD_COUNT), ("source_group_count", config.EXPECTED_SOURCE_GROUP_COUNT), ("score_cell_count", config.EXPECTED_CELL_COUNT)):
        diagnostic_key = "score_count" if key == "score_cell_count" else key
        if int(final.get(key, -1)) != expected or int(diagnostics.get(diagnostic_key, -1)) != expected:
            raise DiagnosticError(f"count binding is invalid: {key}")
    parent = load_parent()
    if audit.get("matrix_file_count") != len(parent["matrix_cache"]):
        raise DiagnosticError("audit matrix count differs from frozen inputs")
    score_differences = _stored_score_differences(parent)
    core = _independent_core(parent)
    differences: list[dict[str, Any]] = list(score_differences)
    compare_values(diagnostics.get("gates"), core["gates"], "diagnostics.gates", differences)
    compare_values(diagnostics.get("per_record"), core["rows"], "diagnostics.per_record", differences)
    compare_values(diagnostics.get("own_audio_decomposition", {}).get("rows"), core["own_rows"], "diagnostics.own_audio_decomposition.rows", differences)
    expected_history = "match" if not differences else "mismatch"
    expected_decision = "CONTROL_FAILURE_REPRODUCED" if expected_history == "match" else "AUDIT_MISMATCH"
    if diagnostics.get("history_comparison", {}).get("status") != expected_history or final.get("diagnostic_decision") != expected_decision:
        raise DiagnosticError("diagnostic terminal does not match independent recomputation")
    if diagnostics.get("c_failure_summary", {}).get("flag_counts") != {name: len(ids) for name, ids in core["c_flags"].items()}:
        raise DiagnosticError("C failure flag counts differ from independent recomputation")
    own_rows = diagnostics.get("own_audio_decomposition", {}).get("rows")
    if not isinstance(own_rows, list) or any(abs(float(row.get("identity_error", 1.0))) > 1e-9 for row in own_rows):
        raise DiagnosticError("own-audio decomposition identity is invalid")
    expected_scientific = parent["payloads"]["final"].get("scientific_decision")
    if expected_scientific != "CONTROL_FAILED" or final.get("historical_scientific_decision") != "CONTROL_FAILED":
        raise DiagnosticError("historical scientific decision was changed")
    result = {"status": "valid", "mode": "complete", "diagnostic_decision": expected_decision, "historical_scientific_decision": "CONTROL_FAILED", "final_sha256": file_sha256(root / "final.json"), "independent_difference_count": len(differences), "new_generated_videos": 0, "new_score_cells": 0, "bridge_executed": False}
    write_self_hashed_json(root / "validation.json", {"schema_version": 1, "stage_id": "validation", "protocol_id": config.PROTOCOL_ID, **result})
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently validate the Wav2Lip ROI control diagnostic")
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = validate_run(args.run_root)
    except (DiagnosticError, OSError, ValueError, KeyError, IndexError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 2 if result.get("mode") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
