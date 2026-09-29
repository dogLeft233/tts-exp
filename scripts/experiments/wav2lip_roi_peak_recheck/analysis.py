from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import RecheckError, file_sha256


def _peak(curve: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float64)
    if values.shape != (31,) or not np.isfinite(values).all():
        raise RecheckError(f"local curve must be finite with shape [31], got {values.shape}")
    order = np.sort(values)
    min_index = int(np.argmin(values))
    offset = config.VSHIFT - min_index
    gap = float(order[1] - order[0])
    return {
        "min_index": min_index,
        "offset": offset,
        "minimum": float(order[0]),
        "second_minimum": float(order[1]),
        "peak_gap": gap,
        "clear": bool(gap > config.PEAK_GAP_THRESHOLD and -config.VSHIFT < offset < config.VSHIFT),
    }


def summarize_segment(matrix: np.ndarray, rows: Sequence[int], d_by_row: Mapping[str, Any]) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 31 or value.shape[0] < 1 or not np.isfinite(value).all():
        raise RecheckError(f"distance matrix is malformed: {value.shape}")
    selected = [int(row) for row in rows]
    if not selected or min(selected) < 0 or max(selected) >= value.shape[0]:
        raise RecheckError(f"mask rows are outside matrix: {selected[:3]} / {value.shape[0]}")
    curve = np.mean(value[selected, :], axis=0, dtype=np.float64)
    evidence = {"rows": selected, "curve": [float(item) for item in curve]}
    evidence.update(_peak(curve))
    evidence["expected"] = float(-np.mean([float(d_by_row[str(row)]) for row in selected], dtype=np.float64))
    return evidence


def _c_pass(
    plus: Mapping[str, Mapping[str, Any]],
    minus: Mapping[str, Mapping[str, Any]],
    baseline_pass: bool,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not baseline_pass:
        reasons.append("baseline_invalid")
    for name, pair in (("PLUS", plus), ("MINUS", minus)):
        if not bool(pair["gn"]["clear"]):
            reasons.append(f"{name.lower()}_gn_unclear_or_boundary")
        if not bool(pair["gw"]["clear"]):
            reasons.append(f"{name.lower()}_gw_unclear_or_boundary")
        if abs(float(pair["gn"]["residual"])) > config.OFFSET_TOLERANCE_FRAMES:
            reasons.append(f"{name.lower()}_offset_error")
    return not reasons, reasons


def _protocol_record(protocol: Mapping[str, Any], sample_id: str) -> Mapping[str, Any]:
    for record in protocol.get("selected_records", []):
        if isinstance(record, Mapping) and str(record.get("sample_id")) == sample_id:
            return record
    raise RecheckError(f"protocol selected record is missing: {sample_id}")


def _segment_pair(
    old_matrix: np.ndarray,
    new_matrix: np.ndarray,
    rows: Sequence[int],
    d_by_row: Mapping[str, Any],
) -> dict[str, Any]:
    old = summarize_segment(old_matrix, rows, d_by_row)
    new = summarize_segment(new_matrix, rows, d_by_row)
    old["actual"] = None
    new["actual"] = None
    old["residual"] = None
    new["residual"] = None
    return {"historical": old, "new": new}


def _attach_response(pair: dict[str, Any], other_old: Mapping[str, Any], other_new: Mapping[str, Any]) -> None:
    for key in ("historical", "new"):
        current = pair[key]
        other = other_old if key == "historical" else other_new
        current["actual"] = float(other["offset"] - current["offset"])
        current["residual"] = float(current["actual"] - current["expected"])


def _media_row(parent: Mapping[str, Any], sample_id: str) -> Mapping[str, Any]:
    for row in parent["media_manifest"].get("rows", []):
        if isinstance(row, Mapping) and str(row.get("sample_id")) == sample_id:
            return row
    raise RecheckError(f"parent media row is missing: {sample_id}")


def _historical_matrix(parent: Mapping[str, Any], sample_id: str, video_arm: str) -> tuple[np.ndarray, Mapping[str, Any]]:
    key = (sample_id, video_arm, config.AUDIO_ARM, config.REPEAT)
    row = parent["score_index"].get(key)
    if not isinstance(row, Mapping):
        raise RecheckError(f"parent score cell is missing: {key}")
    matrix_hash = str(row.get("matrix_sha256"))
    matrix = parent["matrix_cache"].get(matrix_hash)
    if not isinstance(matrix, np.ndarray):
        path = Path(str(row.get("matrix")))
        if file_sha256(path) != matrix_hash:
            raise RecheckError(f"parent matrix hash changed: {path}")
        matrix = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    return matrix, row


def selected_records(parent: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    wanted = set(config.FAIL_IDS + config.PASS_IDS)
    records = [record for record in parent["records"] if str(record.get("sample_id")) in wanted]
    if len(records) != config.EXPECTED_RECORD_COUNT or {str(item["sample_id"]) for item in records} != wanted:
        raise RecheckError("selected sample set does not match the frozen 8-fail plus 2-pass cohort")
    return records


def build_input_rows(parent: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in selected_records(parent):
        sample_id = str(record["sample_id"])
        media_row = _media_row(parent, sample_id)
        source_audio = record.get("natural_audio")
        if not isinstance(source_audio, Mapping):
            raise RecheckError(f"natural audio binding is missing: {sample_id}")
        masks = record.get("masks")
        if not isinstance(masks, Mapping) or not isinstance(masks.get("d_by_row"), Mapping):
            raise RecheckError(f"frozen local masks are missing: {sample_id}")
        for video_arm in config.VIDEO_ARMS:
            cell_name = f"{video_arm}__{config.AUDIO_ARM}"
            cell = media_row.get("cells", {}).get(cell_name)
            old_matrix, old_score = _historical_matrix(parent, sample_id, video_arm)
            if not isinstance(cell, Mapping):
                raise RecheckError(f"parent media cell is missing: {sample_id}/{cell_name}")
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_group": str(record["source_group"]),
                    "video_arm": video_arm,
                    "audio_arm": config.AUDIO_ARM,
                    "repeat": config.REPEAT,
                    "media": str(Path(str(cell["output"])).resolve()),
                    "media_sha256": str(cell["output_sha256"]),
                    "media_pcm_sha256": str(cell["audio_pcm_sha256"]),
                    "source_audio": str(Path(str(source_audio["path"])).resolve()),
                    "source_audio_sha256": str(source_audio["sha256"]),
                    "source_audio_pcm_sha256": str(source_audio["decoded_pcm_sha256"]),
                    "parent_matrix": str(Path(str(old_score["matrix"])).resolve()),
                    "parent_matrix_sha256": str(old_score["matrix_sha256"]),
                    "parent_matrix_shape": [int(item) for item in old_matrix.shape],
                    "masks": {
                        "plus_rows": [int(item) for item in masks["plus_rows"]],
                        "minus_rows": [int(item) for item in masks["minus_rows"]],
                        "d_by_row": {str(key): float(value) for key, value in masks["d_by_row"].items()},
                    },
                }
            )
    if len(rows) != config.EXPECTED_CELL_COUNT:
        raise RecheckError(f"expected {config.EXPECTED_CELL_COUNT} input cells, found {len(rows)}")
    return rows


def analyze(parent: Mapping[str, Any], protocol: Mapping[str, Any], score_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    score_index = {(str(row["sample_id"]), str(row["video_arm"])): row for row in score_rows}
    if len(score_index) != config.EXPECTED_CELL_COUNT:
        raise RecheckError(f"new score cell set is incomplete: {len(score_index)}/{config.EXPECTED_CELL_COUNT}")
    per_record: list[dict[str, Any]] = []
    matrix_comparisons: list[dict[str, Any]] = []
    curve_comparisons: list[dict[str, Any]] = []
    for record in selected_records(parent):
        sample_id = str(record["sample_id"])
        masks = record["masks"]
        protocol_record = _protocol_record(protocol, sample_id)
        baseline = protocol_record.get("baseline_g_n_n")
        if not isinstance(baseline, Mapping) or not isinstance(baseline.get("passes"), bool):
            raise RecheckError(f"frozen G_N/N baseline is missing: {sample_id}")
        baseline_pass = bool(baseline["passes"])
        plus_rows = [int(item) for item in masks["plus_rows"]]
        minus_rows = [int(item) for item in masks["minus_rows"]]
        d_by_row = masks["d_by_row"]
        old_gn, _old_gn_row = _historical_matrix(parent, sample_id, "G_N")
        old_gw, _old_gw_row = _historical_matrix(parent, sample_id, "G_W")
        new_gn = np.asarray(np.load(Path(str(score_index[(sample_id, "G_N")]["matrix"])), allow_pickle=False), dtype=np.float64)
        new_gw = np.asarray(np.load(Path(str(score_index[(sample_id, "G_W")]["matrix"])), allow_pickle=False), dtype=np.float64)
        if old_gn.shape != new_gn.shape or old_gw.shape != new_gw.shape:
            raise RecheckError(f"new/old matrix shape differs: {sample_id}")
        for arm, old_matrix, new_matrix in (("G_N", old_gn, new_gn), ("G_W", old_gw, new_gw)):
            absolute = np.abs(old_matrix - new_matrix)
            flat_index = int(np.argmax(absolute))
            difference = float(absolute.reshape(-1)[flat_index])
            peak_row, peak_col = np.unravel_index(flat_index, absolute.shape)
            matrix_comparisons.append(
                {
                    "sample_id": sample_id,
                    "video_arm": arm,
                    "location_name": arm,
                    "max_abs": difference,
                    "tolerance": config.MATRIX_DIFF_TOLERANCE,
                    "pass": difference <= config.MATRIX_DIFF_TOLERANCE,
                    "argmax": {"row": int(peak_row), "column": int(peak_col)},
                    "old_value": float(old_matrix[peak_row, peak_col]),
                    "new_value": float(new_matrix[peak_row, peak_col]),
                }
            )
        historical_plus = summarize_segment(old_gn, plus_rows, d_by_row)
        historical_minus = summarize_segment(old_gn, minus_rows, d_by_row)
        new_plus = summarize_segment(new_gn, plus_rows, d_by_row)
        new_minus = summarize_segment(new_gn, minus_rows, d_by_row)
        old_gw_plus = summarize_segment(old_gw, plus_rows, d_by_row)
        old_gw_minus = summarize_segment(old_gw, minus_rows, d_by_row)
        new_gw_plus = summarize_segment(new_gw, plus_rows, d_by_row)
        new_gw_minus = summarize_segment(new_gw, minus_rows, d_by_row)
        for name, old_left, new_left, old_right, new_right in (
            ("PLUS", historical_plus, new_plus, old_gw_plus, new_gw_plus),
            ("MINUS", historical_minus, new_minus, old_gw_minus, new_gw_minus),
        ):
            for label, left, right in (("G_N", old_left, new_left), ("G_W", old_right, new_right)):
                curve_delta = np.abs(np.asarray(left["curve"], dtype=np.float64) - np.asarray(right["curve"], dtype=np.float64))
                curve_index = int(np.argmax(curve_delta))
                max_curve = float(curve_delta[curve_index])
                curve_comparisons.append(
                    {
                        "sample_id": sample_id,
                        "segment": name,
                        "video_arm": label,
                        "location_name": label,
                        "max_abs": max_curve,
                        "tolerance": config.CURVE_DIFF_TOLERANCE,
                        "pass": max_curve <= config.CURVE_DIFF_TOLERANCE,
                        "argmax_column": curve_index,
                        "argmax_offset": config.VSHIFT - curve_index,
                        "old_value": float(np.asarray(left["curve"])[curve_index]),
                        "new_value": float(np.asarray(right["curve"])[curve_index]),
                    }
                )
        old_plus = {"gn": historical_plus, "gw": old_gw_plus}
        old_minus = {"gn": historical_minus, "gw": old_gw_minus}
        new_plus = {"gn": new_plus, "gw": new_gw_plus}
        new_minus = {"gn": new_minus, "gw": new_gw_minus}
        for pair in (old_plus, old_minus, new_plus, new_minus):
            pair["gn"]["actual"] = None
            pair["gn"]["residual"] = None
            pair["gw"]["actual"] = None
            pair["gw"]["residual"] = None
        for old_pair, new_pair in ((old_plus, new_plus), (old_minus, new_minus)):
            for pair in (old_pair, new_pair):
                pair["gn"]["actual"] = float(pair["gw"]["offset"] - pair["gn"]["offset"])
                pair["gn"]["residual"] = float(pair["gn"]["actual"] - pair["gn"]["expected"])
                pair["gw"]["actual"] = float(pair["gw"]["offset"] - pair["gn"]["offset"])
                pair["gw"]["residual"] = float(pair["gn"]["residual"])
        old_c_pass, old_reasons = _c_pass(old_plus, old_minus, baseline_pass)
        new_c_pass, new_reasons = _c_pass(new_plus, new_minus, baseline_pass)
        selected_label = "historical_fail" if sample_id in config.FAIL_IDS else "historical_pass"
        offsets_match = all(
            old_pair["gn"]["offset"] == new_pair["gn"]["offset"]
            and old_pair["gw"]["offset"] == new_pair["gw"]["offset"]
            for old_pair, new_pair in ((old_plus, new_plus), (old_minus, new_minus))
        )
        clear_flags_match = all(
            old_pair[arm]["clear"] == new_pair[arm]["clear"]
            for old_pair, new_pair in ((old_plus, new_plus), (old_minus, new_minus))
            for arm in ("gn", "gw")
        )
        per_record.append(
            {
                "sample_id": sample_id,
                "source_group": str(record["source_group"]),
                "historical_label": selected_label,
                "baseline_g_n_n": baseline,
                "historical": {"PLUS": old_plus, "MINUS": old_minus, "c_pass": old_c_pass, "failure_reasons": old_reasons},
                "new": {"PLUS": new_plus, "MINUS": new_minus, "c_pass": new_c_pass, "failure_reasons": new_reasons},
                "offsets_match": offsets_match,
                "clear_flags_match": clear_flags_match,
                "c_decision_match": old_c_pass == new_c_pass,
                "matrix_diff_pass": all(item["pass"] for item in matrix_comparisons if item["sample_id"] == sample_id),
                "curve_diff_pass": all(item["pass"] for item in curve_comparisons if item["sample_id"] == sample_id),
                "interpretation": "响应偏差/误差；本实验不将局部峰结果归因为生成器响应不足",
            }
        )
    failure_rows = [row for row in per_record if row["historical_label"] == "historical_fail"]
    pass_rows = [row for row in per_record if row["historical_label"] == "historical_pass"]
    failure_reproduced = sum(not bool(row["new"]["c_pass"]) for row in failure_rows)
    pass_reproduced = sum(bool(row["new"]["c_pass"]) for row in pass_rows)
    matrix_differences = [item for item in matrix_comparisons if not item["pass"]]
    curve_differences = [item for item in curve_comparisons if not item["pass"]]
    all_matrix_pass = not matrix_differences
    all_curve_pass = not curve_differences
    all_peak_pass = all(row["offsets_match"] and row["clear_flags_match"] and row["c_decision_match"] for row in per_record)
    decision = "PEAKS_REPRODUCED" if all_matrix_pass and all_curve_pass and all_peak_pass and failure_reproduced == 8 and pass_reproduced == 2 else "SCORER_MISMATCH"
    return {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "decision": decision,
        "record_count": len(per_record),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "score_cell_count": len(score_rows),
        "expected_score_cell_count": config.EXPECTED_CELL_COUNT,
        "historical_failure_count": len(failure_rows),
        "historical_failure_reproduced": failure_reproduced,
        "historical_pass_count": len(pass_rows),
        "historical_pass_reproduced": pass_reproduced,
        "matrix_max_abs_difference": float(max([item["max_abs"] for item in matrix_comparisons], default=0.0)),
        "curve_max_abs_difference": float(max([item["max_abs"] for item in curve_comparisons], default=0.0)),
        "matrix_diff_pass": all_matrix_pass,
        "curve_diff_pass": all_curve_pass,
        "peak_and_decision_pass": all_peak_pass,
        "matrix_differences": matrix_differences,
        "curve_differences": curve_differences,
        "matrix_comparisons": matrix_comparisons,
        "curve_comparisons": curve_comparisons,
        "per_record": per_record,
        "causal_boundary": "PEAKS_REPRODUCED 只说明独立实现复现了 SyncNet 所见的局部响应误差；不能区分 SyncNet 表征限制、warp 声学变化与生成器响应，不能写成生成器响应不足。",
        "own_audio_retested": False,
    }
