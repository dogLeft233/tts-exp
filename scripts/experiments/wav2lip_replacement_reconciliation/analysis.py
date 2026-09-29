from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import ReconciliationError, load_matrix, verify_self_hashed_json
from .protocol import _cell_index, load_bound_matrices

_DEFAULT_MEAN_DTYPE = np.dtype(np.float64)


def summarize_matrix(matrix: np.ndarray, rows: Sequence[int], *, mean_dtype: np.dtype[Any] = _DEFAULT_MEAN_DTYPE) -> dict[str, Any]:
    values = np.asarray(matrix)
    selected = [int(row) for row in rows]
    if values.ndim != 2 or values.shape[1] != config.MATRIX_COLUMNS or not selected or min(selected) < 0 or max(selected) >= values.shape[0]:
        raise ReconciliationError("endpoint rows are outside the distance matrix")
    if not np.isfinite(values[selected, :]).all():
        raise ReconciliationError("endpoint matrix contains non-finite values")
    curve = np.mean(values[selected, :], axis=0, dtype=mean_dtype)
    curve = np.asarray(curve, dtype=np.float64)
    index = int(np.argmin(curve))
    minimum = float(curve[index])
    return {
        "curve": [float(value) for value in curve],
        "curve_sha256": hashlib.sha256(np.ascontiguousarray(curve, dtype=np.float64).tobytes()).hexdigest(),
        "min_index": index,
        "offset": int(config.VSHIFT - index),
        "sync_c": float(np.median(curve) - minimum),
        "sync_d": minimum,
        "minimum": minimum,
        "row_count": len(selected),
        "rows": selected,
        "mean_dtype": str(np.dtype(mean_dtype)),
    }


def endpoint_rows(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    matrices = load_bound_matrices(protocol)
    records = list(protocol.get("records", []))
    if len(records) != config.EXPECTED_RECORD_COUNT:
        raise ReconciliationError("protocol record count differs")
    result: list[dict[str, Any]] = []
    for record in records:
        sid = str(record["sample_id"])
        source_group = str(record["source_group"])
        interior = [int(value) for value in record["common_interior_rows"]]
        u_rows = [int(value) for value in record["u_rows"]]
        for origin, arms in (("H", config.H_VIDEO_ARMS), ("S", config.S_VIDEO_ARMS)):
            for arm in arms:
                key = (origin, sid, arm, "N")
                matrix = matrices[key]
                row_sets = {"FULL": list(range(matrix.shape[0])), "COMMON_INTERIOR": interior, "U": u_rows}
                for endpoint, rows in row_sets.items():
                    summary = summarize_matrix(matrix, rows)
                    result.append(
                        {
                            "sample_id": sid,
                            "source_group": source_group,
                            "origin": origin,
                            "video_arm": arm,
                            "audio_arm": "N",
                            "endpoint": endpoint,
                            **summary,
                        }
                    )
    expected = config.EXPECTED_MATRIX_COUNT * len(config.ENDPOINTS)
    if len(result) != expected or len({(row["sample_id"], row["origin"], row["video_arm"], row["audio_arm"], row["endpoint"]) for row in result}) != expected:
        raise ReconciliationError("endpoint row identity is incomplete")
    return result


def shared_bootstrap_indices() -> tuple[list[str], np.ndarray]:
    labels = sorted({str(row["source_group"]) for row in []})
    del labels
    raise ReconciliationError("shared_bootstrap_indices needs protocol records")


def make_bootstrap_indices(protocol: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
    labels = sorted({str(row["source_group"]) for row in protocol.get("records", [])})
    if len(labels) != config.EXPECTED_SOURCE_GROUP_COUNT:
        raise ReconciliationError("bootstrap source-group count differs from 22")
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    indices = rng.integers(0, len(labels), size=(config.BOOTSTRAP_DRAWS, len(labels)), endpoint=False)
    return labels, np.asarray(indices, dtype=np.int64)


def bootstrap_summary(values: Sequence[float], groups: Sequence[str], labels: Sequence[str], indices: np.ndarray) -> dict[str, Any]:
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.ndim != 1 or len(values_array) != len(groups) or not np.isfinite(values_array).all():
        raise ReconciliationError("invalid bootstrap values")
    by_group: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values_array, strict=True):
        by_group[str(group)].append(float(value))
    if sorted(by_group) != list(labels):
        raise ReconciliationError("bootstrap labels differ")
    group_means = np.asarray([np.mean(by_group[label], dtype=np.float64) for label in labels], dtype=np.float64)
    estimates = group_means[indices].mean(axis=1, dtype=np.float64)
    return {
        "mean": float(np.mean(values_array, dtype=np.float64)),
        "ci95": [float(np.quantile(estimates, 0.025, method="linear")), float(np.quantile(estimates, 0.975, method="linear"))],
        "draws": int(config.BOOTSTRAP_DRAWS),
        "seed": int(config.BOOTSTRAP_SEED),
        "rng": "numpy_default_rng_pcg64",
        "quantile_method": "linear",
        "source_group_count": len(labels),
        "record_count": len(values_array),
        "group_means": {label: float(value) for label, value in zip(labels, group_means, strict=True)},
        "draw_indices_sha256": hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest(),
    }


def _interval_from_group_means(summary: Mapping[str, Any], labels: Sequence[str], indices: np.ndarray, level: float) -> list[float]:
    if not 0.0 < float(level) < 1.0:
        raise ReconciliationError(f"invalid confidence level: {level}")
    group_means = np.asarray([float(summary["group_means"][label]) for label in labels], dtype=np.float64)
    estimates = group_means[indices].mean(axis=1, dtype=np.float64)
    tail = (1.0 - float(level)) / 2.0
    return [float(np.quantile(estimates, tail, method="linear")), float(np.quantile(estimates, 1.0 - tail, method="linear"))]


def _endpoint_index(endpoints: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str, str, str], Mapping[str, Any]] = {}
    for row in endpoints:
        key = (str(row["sample_id"]), str(row["origin"]), str(row["video_arm"]), str(row["audio_arm"]), str(row["endpoint"]))
        if key in result:
            raise ReconciliationError(f"duplicate endpoint: {key}")
        result[key] = row
    return result


def _benefit_rows(
    endpoint_index: Mapping[tuple[str, str, str, str, str], Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    origin: str,
    candidate: str,
    baseline: str,
    endpoint: str,
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        sid = str(record["sample_id"])
        candidate_row = endpoint_index[(sid, origin, candidate, "N", endpoint)]
        baseline_row = endpoint_index[(sid, origin, baseline, "N", endpoint)]
        benefit_c = float(candidate_row["sync_c"]) - float(baseline_row["sync_c"])
        benefit_d = float(baseline_row["sync_d"]) - float(candidate_row["sync_d"])
        offset_delta = int(candidate_row["offset"]) - int(baseline_row["offset"])
        rows.append(
            {
                "sample_id": sid,
                "source_group": str(record["source_group"]),
                "candidate": {"sync_c": float(candidate_row["sync_c"]), "sync_d": float(candidate_row["sync_d"]), "offset": int(candidate_row["offset"])},
                "baseline": {"sync_c": float(baseline_row["sync_c"]), "sync_d": float(baseline_row["sync_d"]), "offset": int(baseline_row["offset"])},
                "benefit_c": benefit_c,
                "benefit_d": benefit_d,
                "offset_delta": offset_delta,
                "offset_pass": abs(offset_delta) <= 1,
                "c_positive": benefit_c > 0.0,
                "d_positive": benefit_d > 0.0,
                "joint_win": benefit_c > 0.0 and benefit_d > 0.0,
            }
        )
    return rows


def comparison_summary(rows: Sequence[Mapping[str, Any]], labels: Sequence[str], indices: np.ndarray, name: str, endpoint: str) -> dict[str, Any]:
    c_values = [float(row["benefit_c"]) for row in rows]
    d_values = [float(row["benefit_d"]) for row in rows]
    return {
        "name": name,
        "endpoint": endpoint,
        "record_count": len(rows),
        "per_record": [dict(row) for row in rows],
        "benefit_C": bootstrap_summary(c_values, [str(row["source_group"]) for row in rows], labels, indices),
        "benefit_D": bootstrap_summary(d_values, [str(row["source_group"]) for row in rows], labels, indices),
        "raw_D": {"candidate": [float(row["candidate"]["sync_d"]) for row in rows], "baseline": [float(row["baseline"]["sync_d"]) for row in rows]},
        "positive_record_count": {"C": int(sum(bool(row["c_positive"]) for row in rows)), "D": int(sum(bool(row["d_positive"]) for row in rows))},
        "joint_win_count": int(sum(bool(row["joint_win"]) for row in rows)),
        "offset_agreement_count": int(sum(bool(row["offset_pass"]) for row in rows)),
        "offset_tolerance_frames": 1,
    }


def _historical_bootstrap(values: Sequence[float], groups: Sequence[str]) -> dict[str, Any]:
    values_array = np.asarray(values, dtype=np.float64)
    by_group: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values_array, strict=True):
        by_group[str(group)].append(float(value))
    labels = sorted(by_group)
    group_means = {label: float(np.mean(by_group[label])) for label in labels}
    rng = np.random.default_rng(config.HISTORICAL_BOOTSTRAP_SEED)
    estimates = np.empty(config.HISTORICAL_BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(config.HISTORICAL_BOOTSTRAP_DRAWS):
        sampled = rng.choice(labels, size=len(labels), replace=True)
        estimates[index] = np.mean([group_means[label] for label in sampled])
    return {"mean": float(np.mean(values_array)), "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))], "draws": config.HISTORICAL_BOOTSTRAP_DRAWS, "seed": config.HISTORICAL_BOOTSTRAP_SEED}


def _compare_stat(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str, errors: list[dict[str, Any]], tolerance: float = 1e-6) -> None:
    expected_ci = expected.get("ci95", expected.get("ci"))
    if not isinstance(expected_ci, (list, tuple)) or len(expected_ci) != 2:
        errors.append({"label": label, "field": "ci95", "actual": actual.get("ci95"), "expected": expected_ci, "tolerance": tolerance, "error": "missing two-element confidence interval"})
        return
    actual_ci = actual.get("ci_parent", actual.get("ci95"))
    if not isinstance(actual_ci, (list, tuple)) or len(actual_ci) != 2:
        errors.append({"label": label, "field": "ci95", "actual": actual_ci, "expected": expected_ci, "tolerance": tolerance, "error": "missing actual confidence interval"})
        return
    for field in ("mean",):
        if abs(float(actual[field]) - float(expected[field])) > tolerance:
            errors.append({"label": label, "field": field, "actual": actual[field], "expected": expected[field], "tolerance": tolerance})
    for index, (actual_value, expected_value) in enumerate(zip(actual_ci, expected_ci, strict=True)):
        if abs(float(actual_value) - float(expected_value)) > tolerance:
            errors.append({"label": label, "field": f"ci95[{index}]", "actual": actual_value, "expected": expected_value, "tolerance": tolerance})


def compute_parity(protocol: Mapping[str, Any], endpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    endpoint_index = _endpoint_index(endpoints)
    records = list(protocol["records"])
    h_bindings = [row for row in protocol["matrix_bindings"] if row["origin"] == "H"]
    h_full_errors: list[dict[str, Any]] = []
    for binding in h_bindings:
        matrix = load_matrix(Path(str(binding["matrix"])), str(binding["matrix_sha256"]), f"H parity {binding['cell']}")
        actual = summarize_matrix(matrix, range(matrix.shape[0]), mean_dtype=np.dtype(np.float32))
        expected = binding["log_score"]
        item = {
            "sample_id": binding["sample_id"],
            "video_arm": binding["video_arm"],
            "sync_c_abs_error": abs(float(actual["sync_c"]) - float(expected["sync_c"])),
            "sync_d_abs_error": abs(float(actual["sync_d"]) - float(expected["sync_d"])),
            "offset_equal": int(actual["offset"]) == int(expected["av_offset"]),
        }
        if item["sync_c_abs_error"] > 0.000501 or item["sync_d_abs_error"] > 0.000501 or not item["offset_equal"]:
            h_full_errors.append(item)
    h_score_manifest = verify_self_hashed_json(config.ROOT_INPUTS["H/scores_manifest"][0])
    h_analysis = verify_self_hashed_json(config.ROOT_INPUTS["H/analysis"][0])
    h_score_index = _cell_index(h_score_manifest["scores"], ("sample_id", "cell"))
    groups = [str(record["source_group"]) for record in records]
    def h_deltas(candidate: str) -> tuple[list[float], list[float]]:
        c_values = []
        d_values = []
        for record in records:
            sid = str(record["sample_id"])
            candidate_row = h_score_index[(sid, f"V_{candidate}/A_N")]
            baseline_row = h_score_index[(sid, "V_N/A_N")]
            c_values.append(float(candidate_row["sync_c"]) - float(baseline_row["sync_c"]))
            d_values.append(float(baseline_row["sync_d"]) - float(candidate_row["sync_d"]))
        return c_values, d_values
    h_stat_checks = []
    for candidate in ("N_REPEAT", "BRIDGE_075"):
        c_values, d_values = h_deltas(candidate)
        if candidate == "N_REPEAT":
            parent = h_analysis["controls"]["N_REPEAT"]["repeatability"]
        else:
            parent = h_analysis["bridge"]["replacement_compatibility"]
        for metric, values in (("gap_C", c_values), ("gap_D", d_values)):
            actual = _historical_bootstrap(values, groups)
            expected = parent[metric]
            h_stat_checks.append({"candidate": candidate, "metric": metric, "mean_abs_error": abs(actual["mean"] - expected["mean"]), "ci_abs_error": [abs(actual["ci95"][0] - expected["ci95"][0]), abs(actual["ci95"][1] - expected["ci95"][1])]})
            if abs(actual["mean"] - expected["mean"]) > 1e-6 or any(abs(a - b) > 1e-6 for a, b in zip(actual["ci95"], expected["ci95"], strict=True)):
                errors.append({"type": "H_historical_bootstrap", "candidate": candidate, "metric": metric, "actual": actual, "expected": expected})
    s_analysis = verify_self_hashed_json(config.ROOT_INPUTS["S/analysis"][0])
    labels, indices = make_bootstrap_indices(protocol)
    s_checks = []
    comparisons = {
        "N_REPEAT_vs_N": ("N_REPEAT", "N", s_analysis["stage_a"]["controls"]["repeat_equivalence"]),
        "RT_vs_N": ("RT", "N", s_analysis["stage_a"]["controls"]["rt_equivalence"]),
        "MAG_vs_N": ("MAG", "N", s_analysis["stage_b"]["comparisons"]["MAG_vs_N"]),
        "ENV_vs_N": ("ENV", "N", s_analysis["stage_b"]["comparisons"]["ENV_vs_N"]),
        "MAG_vs_RT": ("MAG", "RT", s_analysis["stage_b"]["comparisons"]["MAG_vs_RT"]),
        "ENV_vs_RT": ("ENV", "RT", s_analysis["stage_b"]["comparisons"]["ENV_vs_RT"]),
        "MAG_vs_ENV": ("MAG", "ENV", s_analysis["stage_b"]["comparisons"]["MAG_vs_ENV"]),
    }
    for name, (candidate, baseline, parent) in comparisons.items():
        rows = _benefit_rows(endpoint_index, records, "S", candidate, baseline, "U")
        c_values = [float(row["benefit_c"]) for row in rows]
        d_values = [float(row["benefit_d"]) for row in rows]
        actual_c = bootstrap_summary(c_values, groups, labels, indices)
        actual_d = bootstrap_summary(d_values, groups, labels, indices)
        expected_c = parent["c"]
        expected_d = parent["d"]
        actual_c["ci_parent"] = _interval_from_group_means(actual_c, labels, indices, float(expected_c.get("level", 0.95)))
        actual_d["ci_parent"] = _interval_from_group_means(actual_d, labels, indices, float(expected_d.get("level", 0.95)))
        _compare_stat(actual_c, expected_c, f"S/{name}/C", errors)
        _compare_stat(actual_d, expected_d, f"S/{name}/D", errors)
        expected_offset = parent.get("offset_count", parent.get("offset_agreement_count"))
        actual_offset = sum(bool(row["offset_pass"]) for row in rows)
        if expected_offset is not None and int(expected_offset) != actual_offset:
            errors.append({"type": "S_offset", "name": name, "actual": actual_offset, "expected": expected_offset})
        s_checks.append({"name": name, "actual": {"C": actual_c, "D": actual_d, "offset_count": actual_offset}, "expected_parent": {"C": expected_c, "D": expected_d, "offset_count": expected_offset}})
    errors.extend({"type": "H_full", **item} for item in h_full_errors)
    return {
        "schema_version": 1,
        "stage_id": "parity",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "valid": not errors,
        "tolerances": {"h_full": 0.000501, "summary": 1e-6, "s_matrix_rebuild": 1e-4},
        "H_full": {"cell_count": len(h_bindings), "error_count": len(h_full_errors), "errors": h_full_errors, "max_sync_c_abs_error": max((item["sync_c_abs_error"] for item in h_full_errors), default=0.0), "max_sync_d_abs_error": max((item["sync_d_abs_error"] for item in h_full_errors), default=0.0)},
        "H_historical_summary": {"checks": h_stat_checks, "valid": not any(item.get("candidate") and item.get("metric") for item in errors if item.get("type") == "H_historical_bootstrap")},
        "S_parent_summary": {"checks": s_checks, "valid": not any(item.get("label", "").startswith("S/") or item.get("type") == "S_offset" for item in errors)},
        "error_count": len(errors),
        "errors": errors,
    }


def _term_summary(values: Sequence[float], groups: Sequence[str], labels: Sequence[str], indices: np.ndarray) -> dict[str, Any]:
    return bootstrap_summary(values, groups, labels, indices)


def _decomposition(
    endpoint_index: Mapping[tuple[str, str, str, str, str], Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    indices: np.ndarray,
    metric: str,
) -> dict[str, Any]:
    groups = [str(record["source_group"]) for record in records]
    values: dict[str, list[float]] = {name: [] for name in ("total_gap", "support_term", "window_term", "pipeline_term", "interaction", "common_pipeline_gap", "identity_check")}
    per_record: list[dict[str, Any]] = []
    for record in records:
        sid = str(record["sample_id"])
        def benefit(origin: str, candidate: str, baseline: str, endpoint: str, sample_id: str = sid) -> float:
            candidate_row = endpoint_index[(sample_id, origin, candidate, "N", endpoint)]
            baseline_row = endpoint_index[(sample_id, origin, baseline, "N", endpoint)]
            if metric == "C":
                return float(candidate_row["sync_c"]) - float(baseline_row["sync_c"])
            return float(baseline_row["sync_d"]) - float(candidate_row["sync_d"])
        h_full = benefit("H", "BRIDGE_075", "N", "FULL")
        h_i = benefit("H", "BRIDGE_075", "N", "COMMON_INTERIOR")
        h_u = benefit("H", "BRIDGE_075", "N", "U")
        s_i = benefit("S", "MAG", "N", "COMMON_INTERIOR")
        s_u = benefit("S", "MAG", "N", "U")
        row = {
            "sample_id": sid,
            "source_group": str(record["source_group"]),
            "h_full": h_full,
            "h_i": h_i,
            "h_u": h_u,
            "s_i": s_i,
            "s_u": s_u,
            "total_gap": s_u - h_full,
            "support_term": h_i - h_full,
            "window_term": h_u - h_i,
            "pipeline_term": s_u - h_u,
            "identity_check": (s_u - h_full) - ((h_i - h_full) + (h_u - h_i) + (s_u - h_u)),
            "interaction": (s_u - s_i) - (h_u - h_i),
            "common_pipeline_gap": s_i - h_i,
        }
        per_record.append(row)
        for name, bucket in values.items():
            bucket.append(float(row[name]))
    summary = {name: _term_summary(values[name], groups, labels, indices) for name in values if name != "identity_check"}
    summary["identity_check"] = {"max_abs": float(np.max(np.abs(values["identity_check"]))), "mean_abs": float(np.mean(np.abs(values["identity_check"]))), "passes": bool(np.max(np.abs(values["identity_check"])) <= 1e-6)}
    return {"metric": metric, "per_record": per_record, "terms": summary}


def make_analysis(protocol: Mapping[str, Any], endpoints: Sequence[Mapping[str, Any]], input_audit: Mapping[str, Any], indices: np.ndarray) -> dict[str, Any]:
    endpoint_index = _endpoint_index(endpoints)
    records = list(protocol["records"])
    labels = sorted({str(row["source_group"]) for row in records})
    comparison_specs = (
        ("H_BRIDGE_075_vs_N", "H", "BRIDGE_075", "N"),
        ("H_N_REPEAT_vs_N", "H", "N_REPEAT", "N"),
        ("S_MAG_vs_N", "S", "MAG", "N"),
        ("S_ENV_vs_N", "S", "ENV", "N"),
        ("S_MAG_vs_RT", "S", "MAG", "RT"),
        ("S_ENV_vs_RT", "S", "ENV", "RT"),
        ("S_N_REPEAT_vs_N", "S", "N_REPEAT", "N"),
        ("S_RT_vs_N", "S", "RT", "N"),
    )
    comparisons: dict[str, Any] = {}
    for name, origin, candidate, baseline in comparison_specs:
        comparisons[name] = {
            endpoint: comparison_summary(_benefit_rows(endpoint_index, records, origin, candidate, baseline, endpoint), labels, indices, name, endpoint)
            for endpoint in config.ENDPOINTS
        }
    decomposition = {metric: _decomposition(endpoint_index, records, labels, indices, metric) for metric in ("C", "D")}
    return {
        "schema_version": 1,
        "stage_id": "analysis",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "record_count": len(records),
        "source_group_count": len(labels),
        "matrix_count": config.EXPECTED_MATRIX_COUNT,
        "endpoint_count": len(endpoints),
        "bootstrap": {"draws": config.BOOTSTRAP_DRAWS, "seed": config.BOOTSTRAP_SEED, "unit": "sorted source_group shared index matrix", "confidence": 0.95, "interval": "two_sided_percentile", "draw_indices_sha256": hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest()},
        "comparisons": comparisons,
        "decomposition": decomposition,
        "audio_summary": input_audit.get("audio", {}),
        "processing_chain": input_audit.get("processing_chain", []),
        "interpretation": {
            "diagnostic_only": True,
            "historical_control_failed_preserved": True,
            "latest_scientific_negative_preserved": True,
            "common_support_is_not_causal": True,
            "pipeline_term_is_mixed_generation_crop_codec_frontend_difference": True,
            "positive_descriptive_endpoint_does_not_authorize_training": True,
        },
        "training_authorized": False,
        "generalization_established": False,
        "historical_gate_repaired": False,
    }
