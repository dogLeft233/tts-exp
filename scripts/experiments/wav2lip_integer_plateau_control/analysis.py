from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import PlateauError, file_sha256


def load_matrix(row: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(row.get("matrix", ""))).resolve()
    expected = str(row.get("matrix_sha256", ""))
    if not path.is_file() or not expected or file_sha256(path) != expected:
        raise PlateauError(f"matrix binding is invalid: {path}")
    matrix = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 31 or matrix.shape[0] < 1 or not np.isfinite(matrix).all():
        raise PlateauError(f"matrix is malformed: {path}")
    return matrix


def peak(curve: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float64)
    if values.shape != (31,) or not np.isfinite(values).all():
        raise PlateauError(f"curve must be finite with shape [31], got {values.shape}")
    order = np.sort(values, kind="stable")
    index = int(np.argmin(values))
    offset = config.VSHIFT - index
    gap = float(order[1] - order[0])
    return {
        "min_index": index,
        "offset": int(offset),
        "sync_d": float(order[0]),
        "sync_c": float(np.median(values) - order[0]),
        "minimum": float(order[0]),
        "second_minimum": float(order[1]),
        "peak_gap": gap,
        "clear": bool(gap > config.PEAK_GAP_THRESHOLD and abs(offset) < config.VSHIFT),
        "curve": [float(item) for item in values],
    }


def summarize(matrix: np.ndarray, rows: Sequence[int]) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    selected = [int(row) for row in rows]
    if values.ndim != 2 or values.shape[1] != 31 or not np.isfinite(values).all():
        raise PlateauError(f"distance matrix is malformed: {values.shape}")
    if not selected or min(selected) < 0 or max(selected) >= values.shape[0]:
        raise PlateauError(f"mask rows are outside distance matrix: {selected[:3]} / {values.shape[0]}")
    result = peak(np.mean(values[selected, :], axis=0, dtype=np.float64))
    result["rows"] = selected
    return result


def _source_rows(record: Mapping[str, Any]) -> dict[str, list[int]]:
    plus = [int(row) for row in record["plus_rows"]]
    minus = [int(row) for row in record["minus_rows"]]
    return {
        "PLUS": [row + config.FRAME_SHIFT for row in plus],
        "MINUS": [row - config.FRAME_SHIFT for row in minus],
    }


def _target_rows(record: Mapping[str, Any]) -> dict[str, list[int]]:
    return {
        "PLUS": [int(row) for row in record["plus_rows"]],
        "MINUS": [int(row) for row in record["minus_rows"]],
    }


def _signature(matrix: np.ndarray, rows: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    plus = summarize(matrix, rows["PLUS"])
    minus = summarize(matrix, rows["MINUS"])
    common = summarize(matrix, [*rows["PLUS"], *rows["MINUS"]])
    return {"PLUS": plus, "MINUS": minus, "common": common}


def _timing(left: Mapping[str, Any], right: Mapping[str, Any], expected: Mapping[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for segment in ("PLUS", "MINUS"):
        actual = int(left[segment]["offset"] - right[segment]["offset"])
        error = actual - int(expected[segment])
        result[segment] = {
            "actual": actual,
            "expected": int(expected[segment]),
            "error": error,
            "passes": bool(left[segment]["clear"] and right[segment]["clear"] and abs(error) <= config.OFFSET_TOLERANCE_FRAMES),
        }
    result["passes"] = bool(result["PLUS"]["passes"] and result["MINUS"]["passes"])
    return result


def _bootstrap(values: Sequence[float], groups: Sequence[str]) -> dict[str, Any]:
    if len(values) != len(groups) or not values:
        raise PlateauError("invalid bootstrap input")
    grouped: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values, strict=True):
        if not np.isfinite(float(value)):
            raise PlateauError("bootstrap value is not finite")
        grouped[str(group)].append(float(value))
    labels = sorted(grouped)
    means = {label: float(np.mean(grouped[label])) for label in labels}
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    estimates = np.empty(config.BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(config.BOOTSTRAP_DRAWS):
        sampled = rng.choice(labels, size=len(labels), replace=True)
        estimates[index] = float(np.mean([means[str(label)] for label in sampled]))
    return {
        "draws": config.BOOTSTRAP_DRAWS,
        "seed": config.BOOTSTRAP_SEED,
        "rng": "numpy_default_rng_pcg64",
        "quantile_method": "linear",
        "record_count": len(values),
        "source_group_count": len(labels),
        "mean": float(statistics.fmean(float(value) for value in values)),
        "ci95": [
            float(np.quantile(estimates, 0.025, method="linear")),
            float(np.quantile(estimates, 0.975, method="linear")),
        ],
        "group_means": means,
    }


def _cell_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"]))
        if key in result:
            raise PlateauError(f"duplicate score cell: {key}")
        result[key] = row
    return result


def _gate_ci(bootstrap: Mapping[str, Any], threshold: float, direction: str) -> bool:
    lower = float(bootstrap["ci95"][0])
    return lower > threshold if direction == "gt" else lower < threshold


def _fixed_flags() -> dict[str, Any]:
    return {
        "historical_scientific_decision": "CONTROL_FAILED",
        "historical_gate_repaired": False,
        "bridge_executed": False,
        "training_authorized": False,
        "reference_conditioned_audio_head_spec_eligible": False,
        "generalization_established": False,
    }


def analyze_oracle(protocol: Mapping[str, Any], score_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fresh = _cell_index(score_rows)
    expected = {(str(record["sample_id"]), video, audio) for record in protocol["records"] for video, audio in config.FRESH_CELL_SPECS}
    if set(fresh) != expected:
        raise PlateauError(f"fresh score cell set differs: {len(fresh)}/{len(expected)}")
    per_record: list[dict[str, Any]] = []
    transport: list[dict[str, Any]] = []
    for record in protocol["records"]:
        sid = str(record["sample_id"])
        target = _target_rows(record)
        source = _source_rows(record)
        baseline = load_matrix(record["parent_v_id_n"])
        matrices = {
            (video, audio): load_matrix(fresh[(sid, video, audio)])
            for video, audio in config.FRESH_CELL_SPECS
        }
        U = [*target["PLUS"], *target["MINUS"]]
        Q = [*source["PLUS"], *source["MINUS"]]
        delta = np.abs(matrices[(config.VIDEO_P, config.AUDIO_P)][U, :] - baseline[Q, :])
        location = np.unravel_index(int(np.argmax(delta)), delta.shape)
        parity = {
            "max_abs": float(np.max(delta)),
            "tolerance": config.MATRIX_DIFF_TOLERANCE,
            "argmax": {"row_in_U": int(location[0]), "column": int(location[1])},
            "pass": bool(np.max(delta) <= config.MATRIX_DIFF_TOLERANCE),
        }
        id_n = _signature(baseline, target)
        id_n_source = _signature(baseline, source)
        id_p = _signature(matrices[(config.VIDEO_ID, config.AUDIO_P)], target)
        p_n = _signature(matrices[(config.VIDEO_P, config.AUDIO_N)], target)
        p_p = _signature(matrices[(config.VIDEO_P, config.AUDIO_P)], target)
        baseline_pass = bool(
            all(id_n[name]["clear"] for name in ("PLUS", "MINUS"))
            and all(id_n_source[segment]["clear"] for segment in ("PLUS", "MINUS"))
            and abs(id_n["PLUS"]["offset"] - id_n["MINUS"]["offset"]) <= config.OFFSET_TOLERANCE_FRAMES
            and abs(id_n_source["PLUS"]["offset"] - id_n_source["MINUS"]["offset"]) <= config.OFFSET_TOLERANCE_FRAMES
        )
        checks = {
            "B": _timing(id_p, id_n, {"PLUS": config.FRAME_SHIFT, "MINUS": -config.FRAME_SHIFT}),
            "C": _timing(p_n, id_n_source, {"PLUS": -config.FRAME_SHIFT, "MINUS": config.FRAME_SHIFT}),
            "O": _timing(p_p, id_n_source, {"PLUS": 0, "MINUS": 0}),
        }
        own = {
            "c": float(p_p["common"]["sync_c"] - id_n_source["common"]["sync_c"]),
            "d": float(id_n_source["common"]["sync_d"] - p_p["common"]["sync_d"]),
            "offset_agreement": bool(abs(p_p["common"]["offset"] - id_n_source["common"]["offset"]) <= config.OFFSET_TOLERANCE_FRAMES),
        }
        damage = {
            "c": float(p_p["common"]["sync_c"] - p_n["common"]["sync_c"]),
            "d": float(p_n["common"]["sync_d"] - p_p["common"]["sync_d"]),
            "both_positive": bool(p_p["common"]["sync_c"] > p_n["common"]["sync_c"] and p_n["common"]["sync_d"] > p_p["common"]["sync_d"]),
        }
        chronological = {
            "c": float(p_p["common"]["sync_c"] - id_n["common"]["sync_c"]),
            "d": float(id_n["common"]["sync_d"] - p_p["common"]["sync_d"]),
            "offset_agreement": bool(abs(p_p["common"]["offset"] - id_n["common"]["offset"]) <= config.OFFSET_TOLERANCE_FRAMES),
        }
        per_record.append(
            {
                "sample_id": sid,
                "source_group": str(record["source_group"]),
                "transport_parity": parity,
                "baseline": baseline_pass,
                "checks": checks,
                "own": own,
                "damage": damage,
                "chronological_own": chronological,
                "signatures": {
                    "V_ID__N__U": id_n,
                    "V_ID__N__Q": id_n_source,
                    "V_ID__P__U": id_p,
                    "V_P__N__U": p_n,
                    "V_P__P__U": p_p,
                },
            }
        )
        transport.append(parity)
    groups = [str(row["source_group"]) for row in per_record]
    own_c = [float(row["own"]["c"]) for row in per_record]
    own_d = [float(row["own"]["d"]) for row in per_record]
    damage_c = [float(row["damage"]["c"]) for row in per_record]
    damage_d = [float(row["damage"]["d"]) for row in per_record]
    own_bootstrap = {"c": _bootstrap(own_c, groups), "d": _bootstrap(own_d, groups)}
    damage_bootstrap = {"c": _bootstrap(damage_c, groups), "d": _bootstrap(damage_d, groups)}
    baseline_count = sum(bool(row["baseline"]) for row in per_record)
    timing_counts = {name: sum(bool(row["checks"][name]["passes"]) for row in per_record) for name in ("B", "C", "O")}
    own_gate = {
        "ci_lower_gt_negative_0_10": bool(own_bootstrap["c"]["ci95"][0] > -0.10 and own_bootstrap["d"]["ci95"][0] > -0.10),
        "offset_agreement_count": sum(bool(row["own"]["offset_agreement"]) for row in per_record),
    }
    own_gate["offset_agreement_pass"] = own_gate["offset_agreement_count"] >= config.MIN_BASELINE_RECORDS
    own_gate["passes"] = bool(own_gate["ci_lower_gt_negative_0_10"] and own_gate["offset_agreement_pass"])
    damage_gate = {
        "ci_lower_gt_0_10": bool(damage_bootstrap["c"]["ci95"][0] > 0.10 and damage_bootstrap["d"]["ci95"][0] > 0.10),
        "both_positive_count": sum(bool(row["damage"]["both_positive"]) for row in per_record),
    }
    damage_gate["both_positive_pass"] = damage_gate["both_positive_count"] >= config.MIN_SUCCESS_RECORDS
    damage_gate["passes"] = bool(damage_gate["ci_lower_gt_0_10"] and damage_gate["both_positive_pass"])
    parity_pass = len(transport) == config.EXPECTED_RECORD_COUNT and all(bool(row["pass"]) for row in transport)
    timing_pass = baseline_count >= config.MIN_BASELINE_RECORDS and all(count >= config.MIN_SUCCESS_RECORDS for count in timing_counts.values())
    decision = "INTEGER_PLATEAU_CONTROL_SUPPORTED" if bool(parity_pass and timing_pass and own_gate["passes"] and damage_gate["passes"]) else "ORACLE_PLATEAU_UNRESOLVED"
    return {
        "schema_version": 1,
        "status": "complete",
        "stage_id": "oracle",
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "record_count": len(per_record),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "fresh_score_cell_count": len(score_rows),
        "expected_fresh_score_cell_count": config.EXPECTED_RECORD_COUNT * len(config.FRESH_CELL_SPECS),
        "cached_score_cell_count": config.EXPECTED_RECORD_COUNT,
        "transport_parity": {
            "pass_count": sum(bool(row["pass"]) for row in transport),
            "expected_count": config.EXPECTED_RECORD_COUNT,
            "max_abs_difference": float(max(row["max_abs"] for row in transport)),
            "all_pass": parity_pass,
        },
        "baseline_count": baseline_count,
        "baseline_minimum": config.MIN_BASELINE_RECORDS,
        "timing_counts": timing_counts,
        "timing_minimum": config.MIN_SUCCESS_RECORDS,
        "own_bootstrap": own_bootstrap,
        "own_gate": own_gate,
        "damage_bootstrap": damage_bootstrap,
        "damage_gate": damage_gate,
        "per_record": per_record,
        "chronological_own": {
            "c_mean": float(statistics.fmean(float(row["chronological_own"]["c"]) for row in per_record)),
            "d_mean": float(statistics.fmean(float(row["chronological_own"]["d"]) for row in per_record)),
            "offset_agreement_count": sum(bool(row["chronological_own"]["offset_agreement"]) for row in per_record),
        },
        "interpretation": "matched own compares V_P/P at U with V_ID/N at Q; chronological own at U is descriptive and is not a repair of the historical gate",
        **_fixed_flags(),
    }


def analyze_generated(
    protocol: Mapping[str, Any],
    oracle_score_rows: Sequence[Mapping[str, Any]],
    generated_score_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    oracle = _cell_index(oracle_score_rows)
    generated = _cell_index(generated_score_rows)
    expected_oracle = {(str(r["sample_id"]), config.VIDEO_P, audio) for r in protocol["records"] for audio in (config.AUDIO_N, config.AUDIO_P)}
    expected_generated = {(str(r["sample_id"]), "E_P", audio) for r in protocol["records"] for audio in (config.AUDIO_N, config.AUDIO_P)}
    if not expected_oracle.issubset(set(oracle)) or set(generated) != expected_generated:
        raise PlateauError("generated-stage score cells are incomplete")
    rows: list[dict[str, Any]] = []
    for record in protocol["records"]:
        sid = str(record["sample_id"])
        target = _target_rows(record)
        v_p_n = _signature(load_matrix(oracle[(sid, config.VIDEO_P, config.AUDIO_N)]), target)
        v_p_p = _signature(load_matrix(oracle[(sid, config.VIDEO_P, config.AUDIO_P)]), target)
        e_p_n = _signature(load_matrix(generated[(sid, "E_P", config.AUDIO_N)]), target)
        e_p_p = _signature(load_matrix(generated[(sid, "E_P", config.AUDIO_P)]), target)
        n_timing = _timing(e_p_n, v_p_n, {"PLUS": 0, "MINUS": 0})
        p_timing = _timing(e_p_p, v_p_p, {"PLUS": 0, "MINUS": 0})
        own = {
            "c": float(e_p_p["common"]["sync_c"] - v_p_p["common"]["sync_c"]),
            "d": float(v_p_p["common"]["sync_d"] - e_p_p["common"]["sync_d"]),
            "offset_agreement": bool(abs(e_p_p["common"]["offset"] - v_p_p["common"]["offset"]) <= config.OFFSET_TOLERANCE_FRAMES),
        }
        damage = {
            "c": float(e_p_p["common"]["sync_c"] - e_p_n["common"]["sync_c"]),
            "d": float(e_p_n["common"]["sync_d"] - e_p_p["common"]["sync_d"]),
            "both_positive": bool(e_p_p["common"]["sync_c"] > e_p_n["common"]["sync_c"] and e_p_n["common"]["sync_d"] > e_p_p["common"]["sync_d"]),
        }
        rows.append({"sample_id": sid, "source_group": str(record["source_group"]), "timing_N": n_timing, "timing_P": p_timing, "own": own, "damage": damage})
    groups = [str(row["source_group"]) for row in rows]
    own_bootstrap = {metric: _bootstrap([float(row["own"][metric]) for row in rows], groups) for metric in ("c", "d")}
    damage_bootstrap = {metric: _bootstrap([float(row["damage"][metric]) for row in rows], groups) for metric in ("c", "d")}
    timing_counts = {
        "E_P_N_vs_V_P_N": sum(bool(row["timing_N"]["passes"]) for row in rows),
        "E_P_P_vs_V_P_P": sum(bool(row["timing_P"]["passes"]) for row in rows),
    }
    own_gate = {
        "ci_lower_gt_negative_0_10": bool(own_bootstrap["c"]["ci95"][0] > -0.10 and own_bootstrap["d"]["ci95"][0] > -0.10),
        "offset_agreement_count": sum(bool(row["own"]["offset_agreement"]) for row in rows),
    }
    own_gate["offset_agreement_pass"] = own_gate["offset_agreement_count"] >= config.MIN_BASELINE_RECORDS
    own_gate["passes"] = bool(own_gate["ci_lower_gt_negative_0_10"] and own_gate["offset_agreement_pass"])
    damage_gate = {
        "ci_lower_gt_0_10": bool(damage_bootstrap["c"]["ci95"][0] > 0.10 and damage_bootstrap["d"]["ci95"][0] > 0.10),
        "both_positive_count": sum(bool(row["damage"]["both_positive"]) for row in rows),
    }
    damage_gate["both_positive_pass"] = damage_gate["both_positive_count"] >= config.MIN_SUCCESS_RECORDS
    damage_gate["passes"] = bool(damage_gate["ci_lower_gt_0_10"] and damage_gate["both_positive_pass"])
    timing_pass = all(count >= config.MIN_SUCCESS_RECORDS for count in timing_counts.values())
    decision = "INTEGER_PLATEAU_CONTROL_SUPPORTED" if bool(timing_pass and own_gate["passes"] and damage_gate["passes"]) else "GENERATED_PLATEAU_UNRESOLVED"
    return {
        "schema_version": 1,
        "status": "complete",
        "stage_id": "generated",
        "protocol_id": protocol["protocol_id"],
        "decision": decision,
        "record_count": len(rows),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "fresh_score_cell_count": len(generated_score_rows),
        "expected_fresh_score_cell_count": config.EXPECTED_RECORD_COUNT * 2,
        "timing_counts": timing_counts,
        "timing_minimum": config.MIN_SUCCESS_RECORDS,
        "own_bootstrap": own_bootstrap,
        "own_gate": own_gate,
        "damage_bootstrap": damage_bootstrap,
        "damage_gate": damage_gate,
        "per_record": rows,
        "interpretation": "E_P is compared against V_P at the same target rows; this does not isolate the generator from face, crop, or paste-back effects",
        **_fixed_flags(),
    }
