from __future__ import annotations

import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import RescoreError, bytes_sha256, write_self_hashed_json


def load_old_matrix(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, list) or len(value) != 1:
        raise RescoreError(f"legacy matrix is not one-track: {path}")
    matrix = np.asarray(value[0], dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != config.MATRIX_COLUMNS or not np.isfinite(matrix).all():
        raise RescoreError(f"legacy matrix is malformed: {path}")
    return matrix


def endpoint(matrix: np.ndarray, rows: Sequence[int], anchor: int | None = None) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float32)
    indices = np.asarray([int(row) for row in rows], dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != config.MATRIX_COLUMNS or indices.size == 0 or int(indices.min()) < 0 or int(indices.max()) >= values.shape[0] or not np.isfinite(values).all():
        raise RescoreError(f"endpoint input is malformed: shape={values.shape}, rows={indices.size}")
    curve = values[indices].mean(axis=0, dtype=np.float64)
    minimum_index = int(np.argmin(curve))
    result: dict[str, Any] = {
        "rows": [int(item) for item in indices],
        "curve": [float(item) for item in curve],
        "D": float(curve[minimum_index]),
        "M": float(np.median(curve)),
        "C": float(np.median(curve) - curve[minimum_index]),
        "offset": int(config.VSHIFT - minimum_index),
        "min_index": minimum_index,
    }
    if anchor is not None:
        column = config.VSHIFT - int(anchor)
        if not 0 <= column < config.MATRIX_COLUMNS:
            raise RescoreError(f"anchor is outside SyncNet columns: {anchor}")
        result["anchor_offset"] = int(anchor)
        result["anchor_column"] = column
        result["D_anchor"] = float(curve[column])
        result["C_anchor"] = float(np.median(curve) - curve[column])
    return result


def benefits(natural: Mapping[str, Any], shifted: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "benefit_C": float(shifted["C"] - natural["C"]),
        "benefit_D": float(natural["D"] - shifted["D"]),
        "benefit_anchor": float(natural["D_anchor"] - shifted["D_anchor"]),
        "delta_M": float(shifted["M"] - natural["M"]),
        "offset_delta": int(shifted["offset"] - natural["offset"]),
    }
    if abs(result["benefit_C"] - (result["delta_M"] + result["benefit_D"])) > 1e-6:
        raise RescoreError("benefit identity C = delta_M + D failed")
    return result


def _group_values(rows: Sequence[Mapping[str, Any]], field: str) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["source_group"]), []).append(float(row[field]))
    labels = sorted(grouped)
    return labels, np.asarray([np.mean(grouped[label], dtype=np.float64) for label in labels], dtype=np.float64)


def make_bootstrap(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]]:
    labels, _ = _group_values(rows, fields[0])
    index_by_group = {label: index for index, label in enumerate(labels)}
    group_index = np.asarray([index_by_group[str(row["source_group"])] for row in rows], dtype=np.int64)
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    draws = rng.integers(0, len(labels), size=(config.BOOTSTRAP_DRAWS, len(labels)), dtype=np.int64)
    result: dict[str, Any] = {"source_groups": labels, "draws": int(config.BOOTSTRAP_DRAWS), "seed": config.BOOTSTRAP_SEED, "quantile_method": "linear"}
    for field in fields:
        _, values = _group_values(rows, field)
        sampled = values[draws].mean(axis=1, dtype=np.float64)
        result[field] = {"mean": float(values.mean(dtype=np.float64)), "ci95": [float(item) for item in np.percentile(sampled, [2.5, 97.5], method="linear")], "positive_count": int(np.count_nonzero(values > 0.0))}
    return draws, result


def _signal(aggregate: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    c = aggregate["benefit_C"]
    d = aggregate["benefit_D"]
    anchor = aggregate["benefit_anchor"]
    offset_ok = sum(abs(int(row["offset_delta"])) <= 1 for row in rows)
    positive = float(c["mean"]) > 0.05 and float(c["ci95"][0]) > 0.0
    anchored = positive and float(d["ci95"][0]) > 0.0 and float(anchor["ci95"][0]) > 0.0 and offset_ok >= 20
    return {"positive_score_signal": bool(positive), "anchored_joint_signal": bool(anchored), "offset_within_1_frame_count": int(offset_ok)}


def _path_summary(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    fields = ("benefit_C", "benefit_D", "benefit_anchor", "delta_M")
    _draws, aggregate = make_bootstrap(rows, fields)
    aggregate["offset_delta"] = {"mean": float(np.mean([int(row["offset_delta"]) for row in rows])), "exact_values": [int(row["offset_delta"]) for row in rows]}
    aggregate["path"] = prefix
    aggregate["signals"] = _signal(aggregate, rows)
    return aggregate


def _compare_pair(natural: Mapping[str, Any], shifted: Mapping[str, Any], sample_id: str, source_group: str) -> dict[str, Any]:
    return {"sample_id": sample_id, "source_group": source_group, **benefits(natural, shifted)}


def _matrix_for_score(scores: Mapping[str, Any], key: tuple[str, str]) -> np.ndarray:
    row = scores.get("|".join(key))
    if not isinstance(row, Mapping):
        raise RescoreError(f"new score is missing: {key}")
    matrix_path = Path(str(row["matrix"]))
    matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != config.MATRIX_COLUMNS or not np.isfinite(matrix).all():
        raise RescoreError(f"new matrix is malformed: {matrix_path}")
    return matrix


def analyze(protocol: Mapping[str, Any], history: Mapping[str, Any], score_index: Mapping[str, Any], paths: config.RunPaths) -> tuple[dict[str, Any], dict[str, Any]]:
    endpoint_rows: list[dict[str, Any]] = []
    j_legacy: list[dict[str, Any]] = []
    j_new: list[dict[str, Any]] = []
    frontend_rows: list[dict[str, Any]] = []
    old_full: list[dict[str, Any]] = []
    old_ih: list[dict[str, Any]] = []
    new_full: list[dict[str, Any]] = []

    history_by_id = {str(row["sample_id"]): row for row in history["rows"]}
    for record in protocol["records"]:
        sid = str(record["sample_id"])
        source_group = str(record["source_group"])
        hrow = history_by_id[sid]
        support = [int(item) for item in record["support"]["J"]]
        if not support:
            raise RescoreError(f"J is empty: {sid}")
        old_cells = hrow["cells"]
        old_n_meta = old_cells["V_N/A_N"]
        old_s_meta = old_cells["V_SHIFT_200/A_N"]
        old_n = load_old_matrix(Path(str(old_n_meta["matrix"])))
        old_s = load_old_matrix(Path(str(old_s_meta["matrix"])))
        n_f0 = int(old_n_meta["track_start_frame"])
        s_f0 = int(old_s_meta["track_start_frame"])
        old_n_j = [g - n_f0 for g in support]
        old_s_j = [g - s_f0 for g in support]
        legacy_n = endpoint(old_n, old_n_j)
        anchor = int(legacy_n["offset"])
        legacy_n = endpoint(old_n, old_n_j, anchor)
        legacy_s = endpoint(old_s, old_s_j, anchor)
        legacy_gain = _compare_pair(legacy_n, legacy_s, sid, source_group)
        legacy_gain["support"] = {"absolute_first": support[0], "absolute_last_exclusive": support[-1] + 1, "count": len(support), "natural_local_first": old_n_j[0], "shift_local_first": old_s_j[0], "anchor_offset": anchor}
        j_legacy.append(legacy_gain)

        new_n = _matrix_for_score(score_index, (sid, config.VIDEO_N))
        new_s = _matrix_for_score(score_index, (sid, config.VIDEO_SHIFT))
        new_n_ep = endpoint(new_n, support, anchor)
        new_s_ep = endpoint(new_s, support, anchor)
        new_gain = _compare_pair(new_n_ep, new_s_ep, sid, source_group)
        new_gain["support"] = legacy_gain["support"]
        j_new.append(new_gain)
        frontend_rows.append({"sample_id": sid, "source_group": source_group, **{field: float(new_gain[field] - legacy_gain[field]) for field in ("benefit_C", "benefit_D", "benefit_anchor", "delta_M")}, "offset_delta": int(new_gain["offset_delta"] - legacy_gain["offset_delta"])})
        endpoint_rows.append({"sample_id": sid, "source_group": source_group, "J": support, "anchor_offset": anchor, "legacy": {"N": legacy_n, "SHIFT": legacy_s, "benefit": legacy_gain}, "full_frame_v4": {"N": new_n_ep, "SHIFT": new_s_ep, "benefit": new_gain}})

        full_n0 = endpoint(old_n, range(old_n.shape[0]))
        full_n = endpoint(old_n, range(old_n.shape[0]), int(full_n0["offset"]))
        full_s = endpoint(old_s, range(old_s.shape[0]), int(full_n0["offset"]))
        old_full.append(_compare_pair(full_n, full_s, sid, source_group))
        ih = range(*[int(item) - n_f0 for item in hrow["common_absolute_support"]])
        ih_n0 = endpoint(old_n, ih)
        ih_n = endpoint(old_n, ih, int(ih_n0["offset"]))
        ih_s = endpoint(old_s, range(int(hrow["common_absolute_support"][0]) - s_f0, int(hrow["common_absolute_support"][1]) - s_f0), int(ih_n0["offset"]))
        old_ih.append(_compare_pair(ih_n, ih_s, sid, source_group))
        full_new_n0 = endpoint(new_n, range(new_n.shape[0]))
        full_new_n = endpoint(new_n, range(new_n.shape[0]), int(full_new_n0["offset"]))
        full_new_s = endpoint(new_s, range(new_s.shape[0]), int(full_new_n0["offset"]))
        new_full.append(_compare_pair(full_new_n, full_new_s, sid, source_group))

    all_rows = j_legacy + j_new
    _draws, legacy_aggregate = make_bootstrap(j_legacy, ("benefit_C", "benefit_D", "benefit_anchor", "delta_M"))
    _draws, new_aggregate = make_bootstrap(j_new, ("benefit_C", "benefit_D", "benefit_anchor", "delta_M"))
    _draws, frontend_aggregate = make_bootstrap(frontend_rows, ("benefit_C", "benefit_D", "benefit_anchor", "delta_M"))
    legacy_aggregate["offset_delta"] = {"mean": float(np.mean([row["offset_delta"] for row in j_legacy])), "exact_values": [int(row["offset_delta"]) for row in j_legacy]}
    new_aggregate["offset_delta"] = {"mean": float(np.mean([row["offset_delta"] for row in j_new])), "exact_values": [int(row["offset_delta"]) for row in j_new]}
    frontend_aggregate["offset_delta"] = {"mean": float(np.mean([row["offset_delta"] for row in frontend_rows])), "exact_values": [int(row["offset_delta"]) for row in frontend_rows]}
    legacy_aggregate["signals"] = _signal(legacy_aggregate, j_legacy)
    new_aggregate["signals"] = _signal(new_aggregate, j_new)
    for metric in ("benefit_C", "benefit_D", "benefit_anchor"):
        lower = float(frontend_aggregate[metric]["ci95"][0])
        upper = float(frontend_aggregate[metric]["ci95"][1])
        frontend_aggregate[metric]["direction"] = "POSITIVE" if lower > 0 else "NEGATIVE" if upper < 0 else "UNRESOLVED"

    labels = [str(row["source_group"]) for row in j_legacy]
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    draws = rng.integers(0, len(sorted(set(labels))), size=(config.BOOTSTRAP_DRAWS, len(sorted(set(labels)))), dtype=np.int64)
    np.save(paths.bootstrap_indices, draws, allow_pickle=False)
    summary = {
        "schema_version": 1,
        "stage_id": "historical_shift_rescore_analysis",
        "status": "complete",
        "record_count": len(j_legacy),
        "source_group_count": len(set(labels)),
        "main_support": "J; same absolute rows for legacy and FULL_FRAME_V4",
        "legacy_J": {"aggregate": legacy_aggregate, "per_record": j_legacy},
        "full_frame_v4_J": {"aggregate": new_aggregate, "per_record": j_new},
        "frontend_change_J": {"aggregate": frontend_aggregate, "per_record": frontend_rows},
        "descriptive": {
            "legacy_FULL": {"aggregate": _path_summary(old_full, "LEGACY_TRACKED/FULL"), "per_record": old_full},
            "legacy_I_H": {"aggregate": _path_summary(old_ih, "LEGACY_TRACKED/I_H"), "per_record": old_ih},
            "full_frame_v4_FULL": {"aggregate": _path_summary(new_full, "FULL_FRAME_V4/FULL"), "per_record": new_full},
            "historical_original_expected": history.get("scalar_original_shift_vs_n"),
        },
        "bootstrap": {"draws": config.BOOTSTRAP_DRAWS, "seed": config.BOOTSTRAP_SEED, "source_group_count": len(set(labels)), "indices_sha256": bytes_sha256(np.ascontiguousarray(draws).tobytes()), "intervals": "retrospective exploratory; no multiple-comparison correction"},
        "flags": {"training_authorized": False, "generalization_established": False, "historical_gate_repaired": False},
        "scientific_decision": "NOT_A_CONFIRMATION",
        "next_action": "CLOSE_SHIFT_DIAGNOSTIC",
    }
    write_self_hashed_json(paths.endpoints, {"schema_version": 1, "status": "complete", "rows": endpoint_rows})
    write_self_hashed_json(paths.analysis, summary)
    return summary, {"rows": endpoint_rows}
