from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import config, media
from .common import ExperimentError, bytes_sha256, file_sha256, write_self_hashed_json
from .metrics import bootstrap, compare, endpoint, make_bootstrap


def _matrix(row: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(row["matrix"])).resolve()
    if not path.is_file() or file_sha256(path) != str(row["matrix_sha256"]):
        raise ExperimentError(f"score matrix binding changed: {path}")
    value = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != config.MATRIX_COLUMNS or not np.isfinite(value).all():
        raise ExperimentError(f"score matrix malformed: {path}")
    return value


def _score_index(scores: Mapping[str, Any]) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    rows = scores.get("scores")
    if not isinstance(rows, list):
        raise ExperimentError("score manifest rows are missing")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ExperimentError("score manifest row is malformed")
        key = (str(row["sample_id"]), str(row["face_mode"]), str(row["video_arm"]), str(row["audio_arm"]))
        if key in result:
            raise ExperimentError(f"duplicate score key: {key}")
        result[key] = row
    expected = {(str(record["sample_id"]), mode, video, audio) for record in [] for mode in config.FACE_MODES for video, audio in config.SCORE_CELLS}
    if len(result) != config.EXPECTED_SCORE_COUNT:
        raise ExperimentError(f"score denominator differs: {len(result)} != {config.EXPECTED_SCORE_COUNT}")
    return result


def _video_index(videos: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    rows = videos.get("rows")
    if not isinstance(rows, list):
        raise ExperimentError("video manifest rows are missing")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ExperimentError("video manifest row is malformed")
        key = (str(row["sample_id"]), str(row["face_mode"]), str(row["video_arm"]))
        if key in result:
            raise ExperimentError(f"duplicate video key: {key}")
        result[key] = row
    if len(result) != config.EXPECTED_VIDEO_COUNT:
        raise ExperimentError(f"video denominator differs: {len(result)} != {config.EXPECTED_VIDEO_COUNT}")
    return result


def _support(frame_count: int, matrix: np.ndarray) -> list[int]:
    expected_rows = frame_count - config.WINDOW_FRAMES
    if matrix.shape[0] != expected_rows:
        raise ExperimentError(f"SyncNet row count differs from F-5: {matrix.shape[0]} != {expected_rows}")
    rows = list(range(30, expected_rows - 30))
    if not rows:
        raise ExperimentError("fixed interior support is empty")
    # The 31 columns use MFCC rows around each visual row.  This explicit audit
    # keeps all selected audio windows away from PCM boundaries.
    for row in rows:
        for column in range(config.MATRIX_COLUMNS):
            audio_row = row + column - config.VSHIFT
            first = audio_row * config.MFCC_STRIDE * config.MFCC_FRAME_STEP - 1
            last = (audio_row * config.MFCC_STRIDE + config.MFCC_AUDIO_FRAMES - 1) * config.MFCC_FRAME_STEP + config.MFCC_FRAME_LENGTH - 1
            if first < 0 or last >= (frame_count * config.SAMPLES_PER_FRAME):
                raise ExperimentError(f"fixed interior audio support is invalid at row {row}")
    return rows


def _record_endpoint(record: Mapping[str, Any], score: Mapping[str, Any], support: Sequence[int], anchor: int | None) -> dict[str, Any]:
    matrix = _matrix(score)
    return {"full": endpoint(matrix, range(matrix.shape[0]), anchor_offset=anchor), "I": endpoint(matrix, support, anchor_offset=anchor), "matrix_shape": [int(x) for x in matrix.shape], "score_key": str(score["cell_key"])}


def _aggregate(rows: Sequence[Mapping[str, Any]], groups: Sequence[str], name: str) -> dict[str, Any]:
    ordered = list(rows)
    if list(groups) != sorted(map(str, groups)):
        raise ExperimentError("aggregation rows must be sorted by source_group")
    labels, indices = make_bootstrap(groups)
    result: dict[str, Any] = {"name": name, "record_count": len(ordered), "per_record": []}
    fields = ("C", "D", "anchor", "delta_M")
    for field in fields:
        values = [float(row[field]) for row in ordered]
        result[field] = bootstrap(values, groups, labels, indices)
    offsets = [int(row["offset_delta"]) for row in ordered]
    result["offset_delta"] = {"mean": float(np.mean(offsets)), "values": offsets, "within_one_count": int(sum(abs(value) <= config.OFFSET_TOLERANCE_FRAMES for value in offsets)), "record_count": len(offsets)}
    result["clear_count"] = int(sum(bool(row.get("clear", False)) for row in ordered))
    result["sign_pass_count"] = int(sum(bool(row.get("sign_pass", False)) for row in ordered))
    result["per_record"] = [dict(row) for row in ordered]
    return result


def _pair(records: Sequence[Mapping[str, Any]], endpoints: Mapping[tuple[str, str, str, str], Mapping[str, Any]], mode: str, candidate: tuple[str, str], baseline: tuple[str, str], support: str, expected: int | None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in records:
        sid = str(record["sample_id"])
        left = endpoints[(sid, mode, candidate[0], candidate[1])][support]
        right = endpoints[(sid, mode, baseline[0], baseline[1])][support]
        item = compare(left, right, f"{candidate[0]}/{candidate[1]}-{baseline[0]}/{baseline[1]}-{support}", expected_shift=expected)
        item.update({"sample_id": sid, "source_group": str(record["source_group"])})
        rows.append(item)
    groups = [str(record["source_group"]) for record in records]
    return _aggregate(rows, groups, f"{candidate[0]}/{candidate[1]}-{baseline[0]}/{baseline[1]}-{support}")


def _flag_repeat(item: Mapping[str, Any]) -> bool:
    return bool(item["C"]["ci95"][0] > -0.05 and item["C"]["ci95"][1] < 0.05 and item["D"]["ci95"][0] > -0.05 and item["D"]["ci95"][1] < 0.05 and item["offset_delta"]["within_one_count"] == config.EXPECTED_RECORD_COUNT and all(int(row["offset_delta"]) == 0 for row in item["per_record"]))


def _flag_sign(item: Mapping[str, Any], denominator: int) -> bool:
    return bool(item["sign_pass_count"] >= 10 and denominator == config.EXPECTED_RECORD_COUNT)


def _positive_signal(item: Mapping[str, Any]) -> dict[str, Any]:
    return {"mean_C_gt_0_05": bool(item["C"]["mean"] > 0.05), "C_ci_lower_gt_0": bool(item["C"]["ci95"][0] > 0.0), "D_ci_lower_gt_0": bool(item["D"]["ci95"][0] > 0.0), "anchor_ci_lower_gt_0": bool(item["anchor"]["ci95"][0] > 0.0), "offset_count_at_least_10": bool(item["offset_delta"]["within_one_count"] >= 10), "passes": bool(item["C"]["mean"] > 0.05 and item["C"]["ci95"][0] > 0.0 and item["D"]["ci95"][0] > 0.0 and item["anchor"]["ci95"][0] > 0.0 and item["offset_delta"]["within_one_count"] >= 10)}


def _visual_patch(frame: np.ndarray, box: Sequence[int]) -> np.ndarray:
    top, bottom, left, right = (int(value) for value in box)
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        raise ExperimentError("visual response ROI is empty")
    resized = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized[48:], dtype=np.float32)


def _visual_response(protocol_payload: Mapping[str, Any], video_index: Mapping[tuple[str, str, str], Mapping[str, Any]], root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    records = sorted(protocol_payload["records"], key=lambda row: (str(row["source_group"]), str(row["sample_id"])))
    for record in records:
        sid = str(record["sample_id"])
        for mode in config.FACE_MODES:
            mode_payload = record["face_modes"][mode]
            boxes = mode_payload["boxes"]["boxes"]
            baseline_path = Path(str(video_index[(sid, mode, config.VIDEO_N)]["output"]))
            baseline, _ = media.decode_frames(baseline_path)
            repeat, _ = media.decode_frames(Path(str(video_index[(sid, mode, config.VIDEO_N_REPEAT)]["output"])))
            item: dict[str, Any] = {"sample_id": sid, "source_group": str(record["source_group"]), "face_mode": mode, "lags": list(range(-8, 9)), "shifts": {}}
            repeat_errors = []
            for frame in range(min(baseline.shape[0], repeat.shape[0])):
                repeat_errors.append(float(np.mean(np.abs(_visual_patch(repeat[frame], boxes[frame]) - _visual_patch(baseline[frame], boxes[frame])))))
            item["n_repeat_E0"] = float(np.mean(repeat_errors)) if repeat_errors else None
            t = int(record["frame_count"]) - config.WINDOW_FRAMES
            interior = range(30, t - 30)
            for video_arm, expected in ((config.VIDEO_DELAY, config.FRAME_SHIFT), (config.VIDEO_ADVANCE, -config.FRAME_SHIFT)):
                shifted, _ = media.decode_frames(Path(str(video_index[(sid, mode, video_arm)]["output"])))
                errors: dict[str, float] = {}
                for lag in range(-8, 9):
                    values = []
                    for frame in interior:
                        reference_index = frame - lag
                        if 0 <= reference_index < baseline.shape[0] and frame < shifted.shape[0]:
                            values.append(float(np.mean(np.abs(_visual_patch(shifted[frame], boxes[frame]) - _visual_patch(baseline[reference_index], boxes[reference_index])))))
                    errors[str(lag)] = float(np.mean(values)) if values else float("nan")
                finite = {int(key): value for key, value in errors.items() if np.isfinite(value)}
                best = min(finite, key=lambda key: (finite[key], key)) if finite else None
                item["shifts"][video_arm] = {"expected_lag": expected, "errors": errors, "best_lag": best, "E0": errors["0"], "E_expected": errors[str(expected)]}
            rows.append(item)
    payload = {"schema_version": 1, "stage_id": "visual_response", "protocol_id": "wav2lip_global_shift_response", "status": "complete", "record_count": len(rows), "rows": rows, "description": "Lower-half 96x96 ROI pixel lag; descriptive only, not a mouth-motion truth signal."}
    write_self_hashed_json(root / "visual_response.json", payload)
    return payload


def analyze(protocol_payload: Mapping[str, Any], history_payload: Mapping[str, Any], videos: Mapping[str, Any], scores: Mapping[str, Any], paths: Any) -> dict[str, Any]:
    records = sorted(protocol_payload.get("records", []), key=lambda row: (str(row["source_group"]), str(row["sample_id"])))
    if len(records) != config.EXPECTED_RECORD_COUNT:
        raise ExperimentError("analysis record denominator differs")
    score_index = _score_index(scores)
    video_index = _video_index(videos)
    endpoints: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    endpoint_rows: list[dict[str, Any]] = []
    for record in records:
        sid = str(record["sample_id"])
        for mode in config.FACE_MODES:
            baseline_score = score_index[(sid, mode, config.VIDEO_N, config.AUDIO_N)]
            baseline_matrix = _matrix(baseline_score)
            support = _support(int(record["frame_count"]), baseline_matrix)
            baseline_full = endpoint(baseline_matrix, range(baseline_matrix.shape[0]))
            baseline_i = endpoint(baseline_matrix, support)
            for video_arm, audio_arm in config.SCORE_CELLS:
                score = score_index[(sid, mode, video_arm, audio_arm)]
                matrix = _matrix(score)
                if matrix.shape != baseline_matrix.shape:
                    raise ExperimentError(f"matrix shape differs within record: {sid}/{mode}/{video_arm}/{audio_arm}")
                item = {"sample_id": sid, "source_group": str(record["source_group"]), "face_mode": mode, "video_arm": video_arm, "audio_arm": audio_arm, "cell_key": str(score["cell_key"]), "support_rows": support, "full": endpoint(matrix, range(matrix.shape[0]), anchor_offset=int(baseline_full["offset"])), "I": endpoint(matrix, support, anchor_offset=int(baseline_i["offset"]))}
                endpoints[(sid, mode, video_arm, audio_arm)] = item
                endpoint_rows.append(item)
    write_self_hashed_json(paths.endpoints, {"schema_version": 1, "stage_id": "endpoints", "protocol_id": "wav2lip_global_shift_response", "status": "complete", "record_count": len(records), "cell_count": len(endpoint_rows), "rows": endpoint_rows})

    face_results: dict[str, Any] = {}
    for mode in config.FACE_MODES:
        repeat_i = _pair(records, endpoints, mode, (config.VIDEO_N_REPEAT, config.AUDIO_N), (config.VIDEO_N, config.AUDIO_N), "I", None)
        audio_delay_i = _pair(records, endpoints, mode, (config.VIDEO_N, config.AUDIO_DELAY), (config.VIDEO_N, config.AUDIO_N), "I", -config.FRAME_SHIFT)
        audio_advance_i = _pair(records, endpoints, mode, (config.VIDEO_N, config.AUDIO_ADVANCE), (config.VIDEO_N, config.AUDIO_N), "I", config.FRAME_SHIFT)
        generated_delay_i = _pair(records, endpoints, mode, (config.VIDEO_DELAY, config.AUDIO_N), (config.VIDEO_N, config.AUDIO_N), "I", config.FRAME_SHIFT)
        generated_advance_i = _pair(records, endpoints, mode, (config.VIDEO_ADVANCE, config.AUDIO_N), (config.VIDEO_N, config.AUDIO_N), "I", -config.FRAME_SHIFT)
        both_delay_i = _pair(records, endpoints, mode, (config.VIDEO_DELAY, config.AUDIO_DELAY), (config.VIDEO_N, config.AUDIO_N), "I", 0)
        both_advance_i = _pair(records, endpoints, mode, (config.VIDEO_ADVANCE, config.AUDIO_ADVANCE), (config.VIDEO_N, config.AUDIO_N), "I", 0)
        # FULL is retained as a descriptive endpoint and a paired sensitivity check.
        generated_delay_full = _pair(records, endpoints, mode, (config.VIDEO_DELAY, config.AUDIO_N), (config.VIDEO_N, config.AUDIO_N), "full", config.FRAME_SHIFT)
        generated_advance_full = _pair(records, endpoints, mode, (config.VIDEO_ADVANCE, config.AUDIO_N), (config.VIDEO_N, config.AUDIO_N), "full", -config.FRAME_SHIFT)
        stable = _flag_repeat(repeat_i)
        audio_detectable = _flag_sign(audio_delay_i, len(records)) and _flag_sign(audio_advance_i, len(records))
        generated_follows = _flag_sign(generated_delay_i, len(records)) and _flag_sign(generated_advance_i, len(records))
        if not stable:
            audio_status: Any = "UNINTERPRETABLE"
            generated_status: Any = "UNINTERPRETABLE"
        else:
            audio_status = "PASS" if audio_detectable else "FAIL"
            generated_status = "PASS" if generated_follows else "FAIL"
        positive = {config.VIDEO_DELAY: _positive_signal(generated_delay_i), config.VIDEO_ADVANCE: _positive_signal(generated_advance_i)}
        face_results[mode] = {"comparisons": {"repeat": {"I": repeat_i}, "audio_delay": {"I": audio_delay_i}, "audio_advance": {"I": audio_advance_i}, config.VIDEO_DELAY: {"I": generated_delay_i, "FULL": generated_delay_full}, config.VIDEO_ADVANCE: {"I": generated_advance_i, "FULL": generated_advance_full}, f"{config.VIDEO_DELAY}_both": {"I": both_delay_i}, f"{config.VIDEO_ADVANCE}_both": {"I": both_advance_i}}, "positive_signal": positive, "flags": {"repeat_stable": stable, "audio_shift_detectable": audio_status, "generated_shift_follows": generated_status, "exploratory_positive_signal": {key: bool(value["passes"]) and stable and audio_detectable for key, value in positive.items()}}, "interpretation": "Any sign/positive result is diagnostic only; dynamic pose, static distribution, free offset and SyncNet endpoint confounds remain."}

    # Paired interaction is a descriptive difference of the per-record benefits;
    # it is not inferred from separate significance decisions.
    interactions: dict[str, Any] = {}
    for video_arm in (config.VIDEO_DELAY, config.VIDEO_ADVANCE):
        dynamic = face_results[config.FACE_DYNAMIC]["comparisons"][video_arm]["I"]["per_record"]
        static = face_results[config.FACE_STATIC]["comparisons"][video_arm]["I"]["per_record"]
        rows = [{"C": float(d["C"] - s["C"]), "D": float(d["D"] - s["D"]), "anchor": float((d["anchor"] or 0.0) - (s["anchor"] or 0.0)), "delta_M": float(d["delta_M"] - s["delta_M"]), "offset_delta": int(d["offset_delta"] - s["offset_delta"]), "clear": bool(d["clear"] and s["clear"]), "sample_id": d["sample_id"], "source_group": d["source_group"]} for d, s in zip(dynamic, static, strict=True)]
        interactions[video_arm] = _aggregate(rows, [str(record["source_group"]) for record in records], f"{video_arm}:DYNAMIC-STATIC")

    history_rows = sorted(history_payload.get("rows", []), key=lambda row: (str(row["source_group"]), str(row["sample_id"])))
    history_pairs = []
    for row in history_rows:
        baseline = row["cells"]["V_N/A_N"]["interior"]
        shifted = row["cells"]["V_SHIFT_200/A_N"]["interior"]
        value = compare(shifted, baseline, "historical_SHIFT_200", expected_shift=None)
        value.update({"sample_id": row["sample_id"], "source_group": row["source_group"]})
        history_pairs.append(value)
    hgroups = [str(row["source_group"]) for row in history_rows]
    hlabels, hindices = make_bootstrap(hgroups)
    np.save(paths.history_bootstrap_indices, hindices, allow_pickle=False)
    history_i = {"C": bootstrap([row["C"] for row in history_pairs], hgroups, hlabels, hindices), "D": bootstrap([row["D"] for row in history_pairs], hgroups, hlabels, hindices), "anchor": bootstrap([row["anchor"] or 0.0 for row in history_pairs], hgroups, hlabels, hindices), "per_record": history_pairs, "support": "common absolute-frame interior"}
    labels, indices = make_bootstrap([str(record["source_group"]) for record in records])
    np.save(paths.bootstrap_indices, indices, allow_pickle=False)
    visual = _visual_response(protocol_payload, video_index, paths.root)
    result = {"schema_version": 1, "stage_id": "analysis", "protocol_id": "wav2lip_global_shift_response", "status": "complete", "record_count": len(records), "face_modes": face_results, "interactions": interactions, "history": {"original_full": history_payload.get("scalar_original_shift_vs_n"), "derived_interior": history_i, "parity_pass": history_payload.get("parity_pass")}, "bootstrap": {"draws": config.BOOTSTRAP_DRAWS, "seed": config.BOOTSTRAP_SEED, "shared_indices_sha256": bytes_sha256(np.ascontiguousarray(indices).tobytes()), "source_group_count": len(labels), "confidence": 0.95, "intervals": "exploratory"}, "visual_response_sha256": file_sha256(paths.visual_response), "flags": {"training_authorized": False, "generalization_established": False, "historical_gate_repaired": False}, "next_action": "STOP_AND_REVIEW", "scientific_decision": "NOT_A_CONFIRMATION"}
    write_self_hashed_json(paths.analysis, result)
    return result
