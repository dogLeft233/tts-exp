from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from . import config
from .common import ExperimentError, file_sha256


def load_matrix(row: Mapping[str, Any]) -> np.ndarray:
    path = __import__("pathlib").Path(str(row.get("matrix", ""))).resolve()
    expected = str(row.get("matrix_sha256", ""))
    if not path.is_file() or not expected or file_sha256(path) != expected:
        raise ExperimentError(f"matrix binding is invalid: {path}")
    matrix = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != config.MATRIX_COLUMNS or matrix.shape[0] < 1 or not np.isfinite(matrix).all():
        raise ExperimentError(f"matrix is malformed: {path}")
    return matrix


def peak(curve: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float64)
    if values.shape != (config.MATRIX_COLUMNS,) or not np.isfinite(values).all():
        raise ExperimentError(f"distance curve must be finite [{config.MATRIX_COLUMNS}]")
    index = int(np.argmin(values))
    ordered = np.sort(values, kind="stable")
    return {
        "curve": [float(value) for value in values],
        "min_index": index,
        "offset": int(config.VSHIFT - index),
        "sync_c": float(np.median(values) - ordered[0]),
        "sync_d": float(ordered[0]),
        "minimum": float(ordered[0]),
        "second_minimum": float(ordered[1]),
        "peak_gap": float(ordered[1] - ordered[0]),
        "clear": bool(ordered[1] - ordered[0] > config.PEAK_GAP_THRESHOLD and abs(config.VSHIFT - index) < config.VSHIFT),
    }


def summarize(matrix: np.ndarray, rows: Sequence[int]) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    selected = [int(row) for row in rows]
    if values.ndim != 2 or values.shape[1] != config.MATRIX_COLUMNS or not selected or min(selected) < 0 or max(selected) >= values.shape[0]:
        raise ExperimentError("score rows are outside the distance matrix")
    # The protocol requires averaging rows first, then deriving C/D/offset.
    result = peak(np.mean(values[selected, :], axis=0, dtype=np.float64))
    result["rows"] = selected
    return result


def signature(matrix: np.ndarray, record: Mapping[str, Any]) -> dict[str, Any]:
    plus = summarize(matrix, record["plus_rows"])
    minus = summarize(matrix, record["minus_rows"])
    common = summarize(matrix, record["u"])
    return {"PLUS": plus, "MINUS": minus, "U": common}


def _cell_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"]))
        if key in result:
            raise ExperimentError(f"duplicate score cell: {key}")
        result[key] = row
    return result


def _shared_indices(groups: Sequence[str]) -> tuple[list[str], np.ndarray]:
    labels = sorted({str(group) for group in groups})
    if len(labels) != config.EXPECTED_SOURCE_GROUP_COUNT:
        raise ExperimentError("bootstrap source-group count differs from 22")
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    indices = rng.integers(0, len(labels), size=(config.BOOTSTRAP_DRAWS, len(labels)), endpoint=False)
    return labels, indices


def bootstrap(values: Sequence[float], groups: Sequence[str], *, level: float, labels: Sequence[str], indices: np.ndarray) -> dict[str, Any]:
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.ndim != 1 or len(values_array) != len(groups) or not np.isfinite(values_array).all():
        raise ExperimentError("invalid bootstrap values")
    per_group: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values_array, strict=True):
        per_group[str(group)].append(float(value))
    if sorted(per_group) != list(labels):
        raise ExperimentError("bootstrap labels differ")
    means = np.asarray([np.mean(per_group[label]) for label in labels], dtype=np.float64)
    estimates = means[indices].mean(axis=1)
    tail = (1.0 - level) / 2.0
    return {
        "mean": float(np.mean(values_array)),
        "ci": [float(np.quantile(estimates, tail, method="linear")), float(np.quantile(estimates, 1.0 - tail, method="linear"))],
        "level": float(level),
        "draws": config.BOOTSTRAP_DRAWS,
        "seed": config.BOOTSTRAP_SEED,
        "rng": "numpy_default_rng_pcg64",
        "quantile_method": "linear",
        "source_group_count": len(labels),
        "group_means": {label: float(np.mean(per_group[label])) for label in labels},
        "draw_indices_sha256": hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest(),
    }


def _comparison(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], groups: Sequence[str], *, level: float, labels: Sequence[str], indices: np.ndarray, name: str) -> dict[str, Any]:
    rows = []
    c_values: list[float] = []
    d_values: list[float] = []
    for item_left, item_right, group in zip(left, right, groups, strict=True):
        left_u = item_left["signature"]["U"]
        right_u = item_right["signature"]["U"]
        c = float(left_u["sync_c"] - right_u["sync_c"])
        d = float(right_u["sync_d"] - left_u["sync_d"])
        offset_delta = int(left_u["offset"] - right_u["offset"])
        rows.append({"sample_id": item_left["sample_id"], "source_group": str(group), "delta_c": c, "delta_d": d, "offset_delta": offset_delta, "offset_pass": abs(offset_delta) <= config.OFFSET_TOLERANCE_FRAMES})
        c_values.append(c)
        d_values.append(d)
    return {
        "name": name,
        "record_count": len(rows),
        "per_record": rows,
        "c": bootstrap(c_values, groups, level=level, labels=labels, indices=indices),
        "d": bootstrap(d_values, groups, level=level, labels=labels, indices=indices),
        "offset_agreement_count": sum(bool(row["offset_pass"]) for row in rows),
        "offset_agreement_pass": sum(bool(row["offset_pass"]) for row in rows) >= config.MIN_BASELINE_RECORDS,
    }


def _pair_signatures(records: Sequence[Mapping[str, Any]], cells: Mapping[tuple[str, str, str], Mapping[str, Any]], video_arm: str, audio_arm: str) -> list[dict[str, Any]]:
    result = []
    for record in records:
        sid = str(record["sample_id"])
        cell = cells[(sid, video_arm, audio_arm)]
        result.append({"sample_id": sid, "source_group": str(record["source_group"]), "signature": signature(load_matrix(cell), record)})
    return result


def _timing(left: Mapping[str, Any], right: Mapping[str, Any], expected_plus: int, expected_minus: int) -> dict[str, Any]:
    result = {}
    for segment, expected in (("PLUS", expected_plus), ("MINUS", expected_minus)):
        actual = int(left[segment]["offset"] - right[segment]["offset"])
        result[segment] = {
            "actual": actual,
            "expected": expected,
            "error": actual - expected,
            "passes": bool(left[segment]["clear"] and right[segment]["clear"] and abs(actual - expected) <= config.OFFSET_TOLERANCE_FRAMES),
        }
    result["passes"] = bool(result["PLUS"]["passes"] and result["MINUS"]["passes"])
    return result


def _fixed_flags() -> dict[str, Any]:
    return {
        "training_authorized": False,
        "generalization_established": False,
        "historical_gate_repaired": False,
        "legacy_bridge_executed": False,
    }


def analyze_stage_a(protocol: Mapping[str, Any], score_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = protocol["records"]
    cells = _cell_index(score_rows)
    expected = {(str(record["sample_id"]), arm, config.ARM_N) for record in records for arm in config.STAGE_A_ARMS}
    expected |= {(str(record["sample_id"]), config.ARM_N, config.ARM_P) for record in records}
    if set(cells) != expected:
        raise ExperimentError(f"Stage A cell set differs: {len(cells)}/{len(expected)}")
    arms = {arm: _pair_signatures(records, cells, arm, config.ARM_N) for arm in config.STAGE_A_ARMS}
    n_p = _pair_signatures(records, cells, config.ARM_N, config.ARM_P)
    groups = [str(record["source_group"]) for record in records]
    labels, indices = _shared_indices(groups)
    per_record = []
    baseline_count = 0
    sensitivity_count = 0
    damage_c: list[float] = []
    damage_d: list[float] = []
    repeat_c: list[float] = []
    repeat_d: list[float] = []
    rt_c: list[float] = []
    rt_d: list[float] = []
    for index, record in enumerate(records):
        n_sig = arms[config.ARM_N][index]["signature"]
        repeat_sig = arms[config.ARM_N_REPEAT][index]["signature"]
        rt_sig = arms[config.ARM_RT][index]["signature"]
        p_sig = n_p[index]["signature"]
        baseline = bool(n_sig["PLUS"]["clear"] and n_sig["MINUS"]["clear"] and abs(n_sig["PLUS"]["offset"] - n_sig["MINUS"]["offset"]) <= config.OFFSET_TOLERANCE_FRAMES)
        sensitivity = _timing(p_sig, n_sig, 5, -5)
        if baseline:
            baseline_count += 1
        if sensitivity["passes"]:
            sensitivity_count += 1
        repeat_c.append(float(repeat_sig["U"]["sync_c"] - n_sig["U"]["sync_c"]))
        repeat_d.append(float(n_sig["U"]["sync_d"] - repeat_sig["U"]["sync_d"]))
        rt_c.append(float(rt_sig["U"]["sync_c"] - n_sig["U"]["sync_c"]))
        rt_d.append(float(n_sig["U"]["sync_d"] - rt_sig["U"]["sync_d"]))
        damage_c.append(float(n_sig["U"]["sync_c"] - p_sig["U"]["sync_c"]))
        damage_d.append(float(p_sig["U"]["sync_d"] - n_sig["U"]["sync_d"]))
        per_record.append({
            "sample_id": str(record["sample_id"]),
            "source_group": str(record["source_group"]),
            "baseline": baseline,
            "sensitivity": sensitivity,
            "repeat_delta_c": repeat_c[-1],
            "repeat_delta_d": repeat_d[-1],
            "rt_delta_c": rt_c[-1],
            "rt_delta_d": rt_d[-1],
            "damage_c": damage_c[-1],
            "damage_d": damage_d[-1],
            "damage_both_positive": bool(damage_c[-1] > 0 and damage_d[-1] > 0),
            "signatures": {"N_N": n_sig, "N_REPEAT_N": repeat_sig, "RT_N": rt_sig, "N_P": p_sig},
        })
    repeat_boot = {"c": bootstrap(repeat_c, groups, level=0.95, labels=labels, indices=indices), "d": bootstrap(repeat_d, groups, level=0.95, labels=labels, indices=indices)}
    rt_boot = {"c": bootstrap(rt_c, groups, level=0.95, labels=labels, indices=indices), "d": bootstrap(rt_d, groups, level=0.95, labels=labels, indices=indices)}
    damage_boot = {"c": bootstrap(damage_c, groups, level=0.95, labels=labels, indices=indices), "d": bootstrap(damage_d, groups, level=0.95, labels=labels, indices=indices)}
    repeat_offsets = sum(abs(row["signatures"]["N_REPEAT_N"]["U"]["offset"] - row["signatures"]["N_N"]["U"]["offset"]) <= 1 for row in per_record)
    rt_offsets = sum(abs(row["signatures"]["RT_N"]["U"]["offset"] - row["signatures"]["N_N"]["U"]["offset"]) <= 1 for row in per_record)
    damage_positive = sum(bool(row["damage_both_positive"]) for row in per_record)
    controls = {
        "baseline_clarity": {"count": baseline_count, "minimum": config.MIN_BASELINE_RECORDS, "passes": baseline_count >= config.MIN_BASELINE_RECORDS},
        "sensitivity": {"count": sensitivity_count, "minimum": config.MIN_SUCCESS_RECORDS, "passes": sensitivity_count >= config.MIN_SUCCESS_RECORDS},
        "repeat_equivalence": {
            "c": repeat_boot["c"], "d": repeat_boot["d"], "offset_count": repeat_offsets,
            "passes": bool(-0.05 < repeat_boot["c"]["ci"][0] and repeat_boot["c"]["ci"][1] < 0.05 and -0.05 < repeat_boot["d"]["ci"][0] and repeat_boot["d"]["ci"][1] < 0.05 and repeat_offsets >= config.MIN_BASELINE_RECORDS),
        },
        "rt_equivalence": {
            "c": rt_boot["c"], "d": rt_boot["d"], "offset_count": rt_offsets,
            "passes": bool(-0.05 < rt_boot["c"]["ci"][0] and rt_boot["c"]["ci"][1] < 0.05 and -0.05 < rt_boot["d"]["ci"][0] and rt_boot["d"]["ci"][1] < 0.05 and rt_offsets >= config.MIN_BASELINE_RECORDS),
        },
        "damage": {
            "c": damage_boot["c"], "d": damage_boot["d"], "both_positive_count": damage_positive,
            "passes": bool(damage_boot["c"]["ci"][0] > 0.10 and damage_boot["d"]["ci"][0] > 0.10 and damage_positive >= config.MIN_SUCCESS_RECORDS),
        },
    }
    return {
        "schema_version": 1,
        "stage_id": "A",
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "record_count": len(per_record),
        "score_cell_count": len(score_rows),
        "controls": controls,
        "passes": bool(all(bool(value["passes"]) for value in controls.values())),
        "per_record": per_record,
        "interpretation": "N/P is a same-video local audio mismatch sensitivity control; it does not validate generated temporal equivariance or repair the historical bridge gate.",
        **_fixed_flags(),
    }


def _candidate_comparison(candidate: Sequence[Mapping[str, Any]], reference: Sequence[Mapping[str, Any]], groups: Sequence[str], labels: Sequence[str], indices: np.ndarray, name: str) -> dict[str, Any]:
    return _comparison(candidate, reference, groups, level=0.975, labels=labels, indices=indices, name=name)


def analyze_stage_b(protocol: Mapping[str, Any], score_rows: Sequence[Mapping[str, Any]], control: Mapping[str, Any]) -> dict[str, Any]:
    if not bool(control.get("passes")):
        raise ExperimentError("Stage B cannot be analyzed when Stage A controls fail")
    records = protocol["records"]
    cells = _cell_index(score_rows)
    expected = {(str(record["sample_id"]), arm, config.ARM_N) for record in records for arm in (*config.STAGE_A_ARMS, *config.STAGE_B_ARMS)}
    expected |= {(str(record["sample_id"]), config.ARM_N, config.ARM_P) for record in records}
    if set(cells) != expected:
        raise ExperimentError("Stage B score cell set is incomplete")
    groups = [str(record["source_group"]) for record in records]
    labels, indices = _shared_indices(groups)
    signatures = {arm: _pair_signatures(records, cells, arm, config.ARM_N) for arm in (*config.STAGE_A_ARMS, *config.STAGE_B_ARMS)}
    comparisons: dict[str, Any] = {}
    for candidate in config.STAGE_B_ARMS:
        comparisons[f"{candidate}_vs_N"] = _candidate_comparison(signatures[candidate], signatures[config.ARM_N], groups, labels, indices, f"{candidate}-N")
        comparisons[f"{candidate}_vs_RT"] = _candidate_comparison(signatures[candidate], signatures[config.ARM_RT], groups, labels, indices, f"{candidate}-RT")
    comparisons["MAG_vs_ENV"] = _candidate_comparison(signatures[config.ARM_MAG], signatures[config.ARM_ENV], groups, labels, indices, "MAG-ENV")

    def gain(arm: str) -> dict[str, Any]:
        routes = {}
        for reference in (config.ARM_N, config.ARM_RT):
            item = comparisons[f"{arm}_vs_{reference}"]
            routes[reference] = {
                "c_mean_gt_0_05": bool(item["c"]["mean"] > 0.05),
                "c_ci_lower_gt_0": bool(item["c"]["ci"][0] > 0.0),
                "d_ci_lower_gt_neg_0_10": bool(item["d"]["ci"][0] > -0.10),
                "offset_count": item["offset_agreement_count"],
                "offset_pass": bool(item["offset_agreement_pass"]),
                "passes": bool(item["c"]["mean"] > 0.05 and item["c"]["ci"][0] > 0.0 and item["d"]["ci"][0] > -0.10 and item["offset_agreement_pass"]),
            }
        return {"routes": routes, "passes": bool(all(route["passes"] for route in routes.values()))}

    gains = {arm: gain(arm) for arm in config.STAGE_B_ARMS}
    increment = comparisons["MAG_vs_ENV"]
    increment_positive = bool(
        increment["c"]["ci"][0] > 0.0
        and increment["d"]["ci"][0] > -0.10
        and increment["offset_agreement_pass"]
    )
    average_equivalent = bool(
        -0.05 < increment["c"]["ci"][0]
        and increment["c"]["ci"][1] < 0.05
        and -0.05 < increment["d"]["ci"][0]
        and increment["d"]["ci"][1] < 0.05
        and increment["offset_agreement_pass"]
    )
    temporal = bool(gains[config.ARM_MAG]["passes"] and increment_positive)
    average = bool(not temporal and gains[config.ARM_ENV]["passes"] and average_equivalent)
    if temporal:
        decision = "TEMPORAL_SPECTRAL_INCREMENT_SUPPORTED"
    elif average:
        decision = "AVERAGE_SPECTRUM_SUFFICIENT_IN_SCOPE"
    elif gains[config.ARM_MAG]["passes"] or gains[config.ARM_ENV]["passes"]:
        decision = "GAIN_WITH_MECHANISM_UNRESOLVED"
    else:
        decision = "NO_USEFUL_GAIN_ESTABLISHED"
    return {
        "schema_version": 1,
        "stage_id": "B",
        "status": "complete",
        "protocol_id": protocol["protocol_id"],
        "record_count": len(records),
        "score_cell_count": len(score_rows),
        "comparisons": comparisons,
        "gain": gains,
        "mag_env": {"increment_positive": increment_positive, "average_equivalent": average_equivalent},
        "decision": decision,
        "temporal_spectral_increment_supported": temporal,
        "average_spectrum_sufficient_in_scope": average,
        "interpretation": "All candidate decisions are scoped to seen-fit records and the frozen U local endpoint; no pure content-specific or generalized replacement claim is made.",
        **_fixed_flags(),
    }
