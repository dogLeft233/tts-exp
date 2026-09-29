from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    RecheckError,
    compare_values,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def _fixed_parent() -> dict[str, Any]:
    try:
        from scripts.experiments.wav2lip_roi_control_diagnostic.analysis import (
            load_parent,
        )

        return load_parent()
    except Exception as exc:
        raise RecheckError(f"fixed parent audit failed: {exc}") from exc


def _fixed_baselines() -> dict[str, Mapping[str, Any]]:
    final_path = config.DIAGNOSTIC_ROOT / "final.json"
    diagnostics_path = config.DIAGNOSTIC_ROOT / "diagnostics.json"
    final = verify_self_hashed_json(final_path, config.DIAGNOSTIC_FINAL_SHA256)
    diagnostics = verify_self_hashed_json(diagnostics_path, config.DIAGNOSTIC_DIAGNOSTICS_SHA256)
    validation = verify_self_hashed_json(config.DIAGNOSTIC_VALIDATION)
    if final.get("diagnostic_decision") != "CONTROL_FAILURE_REPRODUCED" or validation.get("status") != "valid":
        raise RecheckError("fixed diagnostic evidence is not valid")
    if validation.get("final_sha256") != config.DIAGNOSTIC_FINAL_SHA256:
        raise RecheckError("fixed diagnostic validation binding changed")
    result: dict[str, Mapping[str, Any]] = {}
    for row in diagnostics.get("per_record", []):
        if not isinstance(row, Mapping):
            raise RecheckError("fixed diagnostic row is malformed")
        baseline = row.get("baseline", {}).get("G_N_N") if isinstance(row.get("baseline"), Mapping) else None
        if not isinstance(baseline, Mapping) or not isinstance(baseline.get("passes"), bool):
            raise RecheckError(f"fixed G_N/N baseline is missing: {row.get('sample_id')}")
        result[str(row["sample_id"])] = {
            "passes": bool(baseline["passes"]),
            "plus": baseline.get("plus"),
            "minus": baseline.get("minus"),
            "direction_difference": baseline.get("direction_difference"),
        }
    return result


def expected_cell_keys(sample_ids: Sequence[str]) -> set[tuple[str, str, str, bool]]:
    return {(str(sample_id), arm, config.AUDIO_ARM, config.REPEAT) for sample_id in sample_ids for arm in config.VIDEO_ARMS}


def check_cell_keys(rows: Sequence[Mapping[str, Any]], sample_ids: Sequence[str]) -> None:
    observed: list[tuple[str, str, str, bool]] = []
    for row in rows:
        key = (str(row.get("sample_id")), str(row.get("video_arm")), str(row.get("audio_arm")), bool(row.get("repeat", False)))
        observed.append(key)
    if len(observed) != len(set(observed)):
        raise RecheckError("score manifest contains duplicate cells")
    expected = expected_cell_keys(sample_ids)
    if set(observed) != expected:
        raise RecheckError(f"score cell set mismatch: missing={sorted(expected - set(observed))} extra={sorted(set(observed) - expected)}")


def _independent_distance(visual: np.ndarray, audio: np.ndarray) -> np.ndarray:
    visual_value = np.asarray(visual, dtype=np.float32)
    audio_value = np.asarray(audio, dtype=np.float32)
    if visual_value.ndim != 2 or audio_value.ndim != 2 or visual_value.shape != audio_value.shape:
        raise RecheckError(f"embedding shape mismatch: {visual_value.shape} vs {audio_value.shape}")
    padded = np.pad(audio_value, ((config.VSHIFT, config.VSHIFT), (0, 0)), mode="constant")
    result = np.empty((visual_value.shape[0], 2 * config.VSHIFT + 1), dtype=np.float32)
    epsilon = np.float32(config.EPSILON)
    for row in range(visual_value.shape[0]):
        difference = visual_value[row : row + 1] - padded[row : row + 2 * config.VSHIFT + 1]
        squared = np.square(difference + epsilon, dtype=np.float32)
        result[row] = np.sqrt(np.sum(squared, axis=1, dtype=np.float32)).astype(np.float32)
    return result


def validate_stored_arrays(visual: np.ndarray, audio: np.ndarray, matrix: np.ndarray) -> float:
    for name, value in (("visual", visual), ("audio", audio), ("matrix", matrix)):
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise RecheckError(f"{name} must be finite float32")
    if visual.ndim != 2 or audio.ndim != 2 or matrix.ndim != 2:
        raise RecheckError("stored arrays must be rank 2")
    if visual.shape != audio.shape or visual.shape[1] != config.EMBEDDING_DIM:
        raise RecheckError(f"stored embeddings have invalid shape: {visual.shape}, {audio.shape}")
    if matrix.shape != (visual.shape[0], 31):
        raise RecheckError(f"stored matrix shape is invalid: {matrix.shape}")
    expected = _independent_distance(visual, audio)
    maximum = float(np.max(np.abs(expected.astype(np.float64) - matrix.astype(np.float64))))
    if maximum > config.VALIDATOR_MATRIX_TOLERANCE:
        raise RecheckError(f"stored matrix differs from independent embedding distance by {maximum}")
    return maximum


def _peak(curve: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float64)
    if values.shape != (31,) or not np.isfinite(values).all():
        raise RecheckError("invalid local curve")
    order = np.sort(values)
    index = int(np.argmin(values))
    offset = config.VSHIFT - index
    gap = float(order[1] - order[0])
    return {
        "min_index": index,
        "offset": offset,
        "minimum": float(order[0]),
        "second_minimum": float(order[1]),
        "peak_gap": gap,
        "clear": bool(gap > config.PEAK_GAP_THRESHOLD and -config.VSHIFT < offset < config.VSHIFT),
    }


def _segment(matrix: np.ndarray, rows: Sequence[int], d_by_row: Mapping[str, Any]) -> dict[str, Any]:
    selected = [int(row) for row in rows]
    curve = np.mean(np.asarray(matrix, dtype=np.float64)[selected, :], axis=0, dtype=np.float64)
    result = {"rows": selected, "curve": [float(item) for item in curve]}
    result.update(_peak(curve))
    result["expected"] = float(-np.mean([float(d_by_row[str(row)]) for row in selected], dtype=np.float64))
    return result


def _c_pass(plus: Mapping[str, Mapping[str, Any]], minus: Mapping[str, Mapping[str, Any]], baseline_pass: bool) -> tuple[bool, list[str]]:
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


def _selected_records(parent: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    wanted = set(config.FAIL_IDS + config.PASS_IDS)
    records = [record for record in parent["records"] if str(record.get("sample_id")) in wanted]
    if len(records) != config.EXPECTED_RECORD_COUNT or {str(row["sample_id"]) for row in records} != wanted:
        raise RecheckError("parent selected records changed")
    return records


def _parent_matrix(parent: Mapping[str, Any], sample_id: str, video_arm: str) -> np.ndarray:
    row = parent["score_index"].get((sample_id, video_arm, config.AUDIO_ARM, config.REPEAT))
    if not isinstance(row, Mapping):
        raise RecheckError(f"parent matrix binding missing: {sample_id}/{video_arm}")
    matrix = parent["matrix_cache"].get(str(row["matrix_sha256"]))
    if not isinstance(matrix, np.ndarray):
        raise RecheckError(f"parent matrix cache missing: {sample_id}/{video_arm}")
    return np.asarray(matrix, dtype=np.float64)


def _protocol_record(protocol: Mapping[str, Any], sample_id: str) -> Mapping[str, Any]:
    for row in protocol.get("selected_records", []):
        if isinstance(row, Mapping) and str(row.get("sample_id")) == sample_id:
            return row
    raise RecheckError(f"protocol record missing: {sample_id}")


def _comparison(old: np.ndarray, new: np.ndarray, tolerance: float, location_name: str) -> dict[str, Any]:
    delta = np.abs(old.astype(np.float64) - new.astype(np.float64))
    flat = int(np.argmax(delta))
    if delta.ndim == 2:
        location = np.unravel_index(flat, delta.shape)
        argmax = {"row": int(location[0]), "column": int(location[1])}
        old_value = float(old[location])
        new_value = float(new[location])
    else:
        argmax = {"column": int(flat), "offset": config.VSHIFT - flat}
        old_value = float(old[flat])
        new_value = float(new[flat])
    return {
        **({"location_name": location_name} if location_name else {}),
        "max_abs": float(delta.reshape(-1)[flat]),
        "tolerance": tolerance,
        "pass": bool(float(delta.reshape(-1)[flat]) <= tolerance),
        "argmax" if delta.ndim == 2 else "argmax_column": argmax if delta.ndim == 2 else int(flat),
        **({} if delta.ndim == 2 else {"argmax_offset": config.VSHIFT - flat}),
        "old_value": old_value,
        "new_value": new_value,
    }


def _derive_core(parent: Mapping[str, Any], protocol: Mapping[str, Any], score_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    score_index = {(str(row["sample_id"]), str(row["video_arm"])): row for row in score_rows}
    records: list[dict[str, Any]] = []
    matrix_comparisons: list[dict[str, Any]] = []
    curve_comparisons: list[dict[str, Any]] = []
    for record in _selected_records(parent):
        sample_id = str(record["sample_id"])
        protocol_record = _protocol_record(protocol, sample_id)
        baseline = protocol_record.get("baseline_g_n_n")
        if not isinstance(baseline, Mapping) or not isinstance(baseline.get("passes"), bool):
            raise RecheckError(f"protocol baseline missing: {sample_id}")
        d_by_row = record["masks"]["d_by_row"]
        plus_rows = record["masks"]["plus_rows"]
        minus_rows = record["masks"]["minus_rows"]
        old_gn = _parent_matrix(parent, sample_id, "G_N")
        old_gw = _parent_matrix(parent, sample_id, "G_W")
        new_gn = np.asarray(np.load(Path(str(score_index[(sample_id, "G_N")]["matrix"])), allow_pickle=False), dtype=np.float64)
        new_gw = np.asarray(np.load(Path(str(score_index[(sample_id, "G_W")]["matrix"])), allow_pickle=False), dtype=np.float64)
        if old_gn.shape != new_gn.shape or old_gw.shape != new_gw.shape:
            raise RecheckError(f"old/new matrix shape mismatch: {sample_id}")
        for arm, old_matrix, new_matrix in (("G_N", old_gn, new_gn), ("G_W", old_gw, new_gw)):
            comparison = _comparison(old_matrix, new_matrix, config.MATRIX_DIFF_TOLERANCE, arm)
            comparison.update({"sample_id": sample_id, "video_arm": arm})
            matrix_comparisons.append(comparison)
        old_plus = {"gn": _segment(old_gn, plus_rows, d_by_row), "gw": _segment(old_gw, plus_rows, d_by_row)}
        old_minus = {"gn": _segment(old_gn, minus_rows, d_by_row), "gw": _segment(old_gw, minus_rows, d_by_row)}
        new_plus = {"gn": _segment(new_gn, plus_rows, d_by_row), "gw": _segment(new_gw, plus_rows, d_by_row)}
        new_minus = {"gn": _segment(new_gn, minus_rows, d_by_row), "gw": _segment(new_gw, minus_rows, d_by_row)}
        for pair in (old_plus, old_minus, new_plus, new_minus):
            pair["gn"]["actual"] = float(pair["gw"]["offset"] - pair["gn"]["offset"])
            pair["gn"]["residual"] = float(pair["gn"]["actual"] - pair["gn"]["expected"])
            pair["gw"]["actual"] = pair["gn"]["actual"]
            pair["gw"]["residual"] = pair["gn"]["residual"]
        for segment_name, old_pair, new_pair in (("PLUS", old_plus, new_plus), ("MINUS", old_minus, new_minus)):
            for arm, label in (("gn", "G_N"), ("gw", "G_W")):
                comparison = _comparison(
                    np.asarray(old_pair[arm]["curve"], dtype=np.float64),
                    np.asarray(new_pair[arm]["curve"], dtype=np.float64),
                    config.CURVE_DIFF_TOLERANCE,
                    label,
                )
                comparison.update({"sample_id": sample_id, "segment": segment_name, "video_arm": label})
                curve_comparisons.append(comparison)
        baseline_pass = bool(baseline["passes"])
        old_c, old_reasons = _c_pass(old_plus, old_minus, baseline_pass)
        new_c, new_reasons = _c_pass(new_plus, new_minus, baseline_pass)
        records.append(
            {
                "sample_id": sample_id,
                "source_group": str(record["source_group"]),
                "historical_label": "historical_fail" if sample_id in config.FAIL_IDS else "historical_pass",
                "baseline_g_n_n": baseline,
                "historical": {"PLUS": old_plus, "MINUS": old_minus, "c_pass": old_c, "failure_reasons": old_reasons},
                "new": {"PLUS": new_plus, "MINUS": new_minus, "c_pass": new_c, "failure_reasons": new_reasons},
                "offsets_match": all(
                    old_pair[arm]["offset"] == new_pair[arm]["offset"]
                    for old_pair, new_pair in ((old_plus, new_plus), (old_minus, new_minus))
                    for arm in ("gn", "gw")
                ),
                "clear_flags_match": all(
                    old_pair[arm]["clear"] == new_pair[arm]["clear"]
                    for old_pair, new_pair in ((old_plus, new_plus), (old_minus, new_minus))
                    for arm in ("gn", "gw")
                ),
                "c_decision_match": old_c == new_c,
                "matrix_diff_pass": all(item["pass"] for item in matrix_comparisons if item["sample_id"] == sample_id),
                "curve_diff_pass": all(item["pass"] for item in curve_comparisons if item["sample_id"] == sample_id),
                "interpretation": "响应偏差/误差；本实验不将局部峰结果归因为生成器响应不足",
            }
        )
    failures = [row for row in records if row["historical_label"] == "historical_fail"]
    passes = [row for row in records if row["historical_label"] == "historical_pass"]
    failure_reproduced = sum(not bool(row["new"]["c_pass"]) for row in failures)
    pass_reproduced = sum(bool(row["new"]["c_pass"]) for row in passes)
    matrix_differences = [item for item in matrix_comparisons if not item["pass"]]
    curve_differences = [item for item in curve_comparisons if not item["pass"]]
    peak_pass = all(row["offsets_match"] and row["clear_flags_match"] and row["c_decision_match"] for row in records)
    decision = "PEAKS_REPRODUCED" if not matrix_differences and not curve_differences and peak_pass and failure_reproduced == 8 and pass_reproduced == 2 else "SCORER_MISMATCH"
    return {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "decision": decision,
        "record_count": len(records),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "score_cell_count": len(score_rows),
        "expected_score_cell_count": config.EXPECTED_CELL_COUNT,
        "historical_failure_count": len(failures),
        "historical_failure_reproduced": failure_reproduced,
        "historical_pass_count": len(passes),
        "historical_pass_reproduced": pass_reproduced,
        "matrix_max_abs_difference": float(max([item["max_abs"] for item in matrix_comparisons], default=0.0)),
        "curve_max_abs_difference": float(max([item["max_abs"] for item in curve_comparisons], default=0.0)),
        "matrix_diff_pass": not matrix_differences,
        "curve_diff_pass": not curve_differences,
        "peak_and_decision_pass": peak_pass,
        "matrix_differences": matrix_differences,
        "curve_differences": curve_differences,
        "matrix_comparisons": matrix_comparisons,
        "curve_comparisons": curve_comparisons,
        "per_record": records,
        "causal_boundary": "PEAKS_REPRODUCED 只说明独立实现复现了 SyncNet 所见的局部响应误差；不能区分 SyncNet 表征限制、warp 声学变化与生成器响应，不能写成生成器响应不足。",
        "own_audio_retested": False,
    }


def _relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_bindings(root: Path, protocol: Mapping[str, Any], parent: Mapping[str, Any]) -> None:
    if protocol.get("protocol_id") != config.PROTOCOL_ID or protocol.get("status") != "frozen":
        raise RecheckError("protocol identity/status is invalid")
    if protocol.get("parent", {}).get("scientific_decision") != "CONTROL_FAILED":
        raise RecheckError("protocol does not preserve historical CONTROL_FAILED")
    if protocol.get("parent", {}).get("fixed_entry_hashes") != config.PARENT_HASHES:
        raise RecheckError("protocol parent hash bindings differ")
    if file_sha256(parent["root"] + "/final.json") != config.PARENT_HASHES["final"]:
        raise RecheckError("parent final hash changed")
    for name, path in (("proposal", config.PROPOSAL), ("design", config.DESIGN), ("spec", config.SPEC)):
        binding = protocol.get("change_bindings", {}).get(name)
        if not isinstance(binding, Mapping) or binding.get("sha256") != file_sha256(path):
            raise RecheckError(f"change binding changed: {name}")
    for name, binding in protocol.get("source_bindings", {}).items():
        path = Path(str(binding.get("path", "")))
        if not path.is_file() or not isinstance(binding, Mapping) or binding.get("sha256") != file_sha256(path):
            raise RecheckError(f"source binding changed: {name}")
    diagnostic = protocol.get("diagnostic_binding")
    if not isinstance(diagnostic, Mapping) or diagnostic.get("decision") != "CONTROL_FAILURE_REPRODUCED":
        raise RecheckError("diagnostic binding is invalid")
    _fixed_baselines()


def validate_run(run_root: Path) -> dict[str, Any]:
    root = Path(run_root)
    errors: list[str] = []
    final_sha: str | None = None
    try:
        final_path = root / "final.json"
        final_sha = file_sha256(final_path) if final_path.is_file() else None
        final = verify_self_hashed_json(final_path)
        protocol = verify_self_hashed_json(root / "protocol.json")
        audit = verify_self_hashed_json(root / "audit.json")
        score_manifest = verify_self_hashed_json(root / "scores/manifest.json")
        analysis = verify_self_hashed_json(root / "analysis.json")
        parent = _fixed_parent()
        _validate_bindings(root, protocol, parent)
        if audit.get("status") != "complete" or audit.get("selected_cell_count") != config.EXPECTED_CELL_COUNT:
            raise RecheckError("audit is incomplete")
        selected_ids = [str(row["sample_id"]) for row in protocol.get("selected_records", [])]
        parent_ids = [str(row["sample_id"]) for row in _selected_records(parent)]
        if selected_ids != parent_ids:
            raise RecheckError("protocol selected record order differs from parent order")
        rows = score_manifest.get("scores")
        if score_manifest.get("status") != "complete" or not isinstance(rows, list):
            raise RecheckError("score manifest is incomplete")
        check_cell_keys(rows, selected_ids)
        if len(rows) != config.EXPECTED_CELL_COUNT:
            raise RecheckError("score count is not exactly 20")
        protocol_cells = {(str(row["sample_id"]), str(row["video_arm"])): row for row in protocol["selected_cells"]}
        root_resolved = root.resolve()
        for row in rows:
            key = (str(row["sample_id"]), str(row["video_arm"]))
            item = protocol_cells.get(key)
            if item is None or row.get("protocol_sha256") != file_sha256(root / "protocol.json"):
                raise RecheckError(f"score protocol binding mismatch: {key}")
            for field in ("media", "source_audio", "visual", "audio_embedding", "matrix", "worker_result"):
                path = Path(str(row[field]))
                if field in ("visual", "audio_embedding", "matrix", "worker_result") and not _relative_to(path, root_resolved):
                    raise RecheckError(f"new artifact escapes run root: {key}/{field}")
                if not path.is_file():
                    raise RecheckError(f"score artifact is missing: {key}/{field}")
            if file_sha256(Path(row["media"])) != str(item["media_sha256"]):
                raise RecheckError(f"media binding changed: {key}")
            if file_sha256(Path(row["source_audio"])) != str(item["source_audio_sha256"]):
                raise RecheckError(f"source audio binding changed: {key}")
            for field, hash_field in (("visual", "visual_sha256"), ("audio_embedding", "audio_embedding_sha256"), ("matrix", "matrix_sha256"), ("worker_result", "worker_result_sha256")):
                if file_sha256(Path(row[field])) != str(row[hash_field]):
                    raise RecheckError(f"score hash changed: {key}/{field}")
            worker = verify_self_hashed_json(Path(row["worker_result"]))
            if worker.get("new_forward") is not True or worker.get("extracted_pcm_verified") is not True:
                raise RecheckError(f"worker does not prove fresh forward/PCM verification: {key}")
            if worker.get("protocol_sha256") != file_sha256(root / "protocol.json"):
                raise RecheckError(f"worker protocol binding mismatch: {key}")
            if worker.get("model_sha256") != config.SYNCNET_MODEL_SHA256:
                raise RecheckError(f"worker model binding mismatch: {key}")
            if worker.get("media_sha256") != str(item["media_sha256"]):
                raise RecheckError(f"worker media binding mismatch: {key}")
            if worker.get("extracted_pcm_sha256") != str(item["media_pcm_sha256"]):
                raise RecheckError(f"worker PCM binding mismatch: {key}")
            if worker.get("visual") != row["visual"] or worker.get("audio_embedding") != row["audio_embedding"] or worker.get("matrix") != row["matrix"]:
                raise RecheckError(f"worker output path binding mismatch: {key}")
            visual = np.load(Path(row["visual"]), allow_pickle=False)
            audio = np.load(Path(row["audio_embedding"]), allow_pickle=False)
            matrix = np.load(Path(row["matrix"]), allow_pickle=False)
            validate_stored_arrays(visual, audio, matrix)
            if [int(item) for item in matrix.shape] != [int(item) for item in row["matrix_shape"]]:
                raise RecheckError(f"score matrix shape binding changed: {key}")
        recomputed = _derive_core(parent, protocol, rows)
        core_keys = (
            "decision",
            "record_count",
            "expected_record_count",
            "score_cell_count",
            "expected_score_cell_count",
            "historical_failure_count",
            "historical_failure_reproduced",
            "historical_pass_count",
            "historical_pass_reproduced",
            "matrix_max_abs_difference",
            "curve_max_abs_difference",
            "matrix_diff_pass",
            "curve_diff_pass",
            "peak_and_decision_pass",
            "matrix_differences",
            "curve_differences",
            "matrix_comparisons",
            "curve_comparisons",
            "per_record",
            "causal_boundary",
            "own_audio_retested",
        )
        differences: list[dict[str, Any]] = []
        compare_values({key: recomputed[key] for key in core_keys}, {key: analysis.get(key) for key in core_keys}, "analysis", differences, 1e-6)
        if differences:
            raise RecheckError(f"analysis differs from independent validator: {differences[:3]}")
        expected_final = {
            "status": "complete",
            "diagnostic_decision": recomputed["decision"],
            "historical_scientific_decision": "CONTROL_FAILED",
            "expected_record_count": config.EXPECTED_RECORD_COUNT,
            "record_count": config.EXPECTED_RECORD_COUNT,
            "expected_score_cell_count": config.EXPECTED_CELL_COUNT,
            "score_cell_count": config.EXPECTED_CELL_COUNT,
            "new_generated_videos": 0,
            "new_score_cells": config.EXPECTED_CELL_COUNT,
            "own_audio_retested": False,
            "bridge_executed": False,
            "training_authorized": False,
            "generalization_established": False,
        }
        for key, value in expected_final.items():
            if final.get(key) != value:
                raise RecheckError(f"final contract mismatch at {key}: {final.get(key)} != {value}")
        if final.get("protocol_sha256") != file_sha256(root / "protocol.json") or final.get("analysis_sha256") != file_sha256(root / "analysis.json"):
            raise RecheckError("final evidence binding mismatch")
        result_path = root / "result.md"
        if not result_path.is_file() or final.get("result_sha256") != file_sha256(result_path):
            raise RecheckError("result binding mismatch")
        result = {
            "schema_version": 1,
            "stage_id": "validation",
            "protocol_id": config.PROTOCOL_ID,
            "status": "valid",
            "valid": True,
            "mode": "independent_embedding_matrix_peak_recompute",
            "final_sha256": final_sha,
            "diagnostic_decision": recomputed["decision"],
            "historical_scientific_decision": "CONTROL_FAILED",
            "independent_difference_count": 0,
            "new_generated_videos": 0,
            "new_score_cells": config.EXPECTED_CELL_COUNT,
            "bridge_executed": False,
            "errors": [],
        }
    except Exception as exc:  # noqa: BLE001 - validator must emit an invalid artifact for every failure
        errors.append(str(exc))
        result = {
            "schema_version": 1,
            "stage_id": "validation",
            "protocol_id": config.PROTOCOL_ID,
            "status": "invalid",
            "valid": False,
            "mode": "independent_embedding_matrix_peak_recompute",
            "final_sha256": final_sha,
            "independent_difference_count": None,
            "errors": errors,
        }
    write_self_hashed_json(root / "validation.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Wav2Lip ROI local peak recheck run offline")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate_run(args.run_root)
    print(result)
    return 0 if result.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
