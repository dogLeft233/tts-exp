from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "scripts.experiments.lrs3_tts_gain_mechanism"

from . import config
from .analysis import curve_analysis, group_pairs, pair_curve_metrics
from .common import (
    ProtocolError,
    csv_read,
    read_self_hashed_json,
    sample_ids_sha256,
    sha256_file,
    write_self_hashed_json,
)
from .runner import (
    _control_metrics,
    build_cohort,
    current_run_identity,
    load_historical_records,
    load_input_bindings,
    verify_bound_files,
    verify_bound_score_cells,
)


def _float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise ProtocolError(f"{label} is non-finite")
    return result


def _validate_historical(root: Path, cohort: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cells_path = root / "01_decomposition" / "cells.csv"
    paired_path = root / "01_decomposition" / "paired.csv"
    groups_path = root / "01_decomposition" / "groups.csv"
    for path in (cells_path, paired_path, groups_path):
        if not path.is_file():
            raise ProtocolError(f"historical artifact is missing: {path}")
    raw_cells, expected_pairs = load_historical_records(cohort)
    expected_groups = group_pairs(expected_pairs)
    actual_cells = csv_read(cells_path)
    actual_pairs = csv_read(paired_path)
    actual_groups = csv_read(groups_path)
    if len(actual_cells) != 200 or len(actual_pairs) != 100 or len(actual_groups) != 90:
        raise ProtocolError(f"historical CSV counts are wrong: {len(actual_cells)}/{len(actual_pairs)}/{len(actual_groups)}")
    cell_key = lambda row: (str(row["model"]), int(row["sample_id"]), str(row["condition"]))
    expected_cell_map = {cell_key(row): row for row in raw_cells}
    actual_cell_map = {cell_key(row): row for row in actual_cells}
    if set(expected_cell_map) != set(actual_cell_map):
        raise ProtocolError("historical cell identities differ from raw bound scores")
    for key, expected in expected_cell_map.items():
        actual = actual_cell_map[key]
        if str(actual["source_group"]) != str(expected["source_group"]):
            raise ProtocolError(f"historical cell value differs: {key}/source_group")
        expected_source = expected.get("source_path")
        if expected_source is None:
            if actual.get("source_path", "") != "":
                raise ProtocolError(f"historical cell source path differs: {key}")
        elif actual.get("source_path") != str(expected_source):
            raise ProtocolError(f"historical cell source path differs: {key}")
        for field in ("sync_c", "sync_d", "background_b_hat"):
            if abs(_float(actual[field], f"cell {key}/{field}") - float(expected[field])) > 1e-10:
                raise ProtocolError(f"historical cell value differs: {key}/{field}")
    pair_key = lambda row: (str(row["model"]), int(row["sample_id"]))
    expected_pair_map = {pair_key(row): row for row in expected_pairs}
    actual_pair_map = {pair_key(row): row for row in actual_pairs}
    if set(expected_pair_map) != set(actual_pair_map):
        raise ProtocolError("historical pair identities differ from raw bound scores")
    for key, expected in expected_pair_map.items():
        actual = actual_pair_map[key]
        if str(actual["source_group"]) != str(expected["source_group"]):
            raise ProtocolError(f"historical pair source group differs: {key}")
        for field in ("gain_c", "gain_match", "gain_background", "decomposition_residual"):
            if abs(_float(actual[field], f"pair {key}/{field}") - float(expected[field])) > 1e-10:
                raise ProtocolError(f"historical pair value differs: {key}/{field}")
    expected_group_map = {(str(row["model"]), str(row["source_group"])): row for row in expected_groups}
    actual_group_map = {(str(row["model"]), str(row["source_group"])): row for row in actual_groups}
    if set(expected_group_map) != set(actual_group_map):
        raise ProtocolError("historical group identities differ")
    for key, expected in expected_group_map.items():
        actual = actual_group_map[key]
        if int(actual["record_count"]) != int(expected["record_count"]):
            raise ProtocolError(f"historical group record count differs: {key}")
        try:
            actual_sample_ids = json.loads(actual["sample_ids"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"historical group sample IDs are malformed: {key}") from exc
        if actual_sample_ids != expected["sample_ids"] or int(actual["c_positive_count"]) != int(expected["c_positive_count"]):
            raise ProtocolError(f"historical group metadata differs: {key}")
        for field in ("gain_c", "gain_match", "gain_background"):
            if abs(_float(actual[field], f"group {key}/{field}") - float(expected[field])) > 1e-10:
                raise ProtocolError(f"historical group value differs: {key}/{field}")
    summary_path = root / "01_decomposition" / "summary.json"
    if not summary_path.is_file():
        raise ProtocolError("historical summary is missing")
    summary = read_self_hashed_json(summary_path)
    group_indices = _bootstrap_matrix(45)
    expected_record_means: dict[str, Any] = {}
    expected_group_summaries: dict[str, Any] = {}
    for model in config.MODELS:
        model_pairs = [row for row in expected_pairs if str(row["model"]) == model]
        model_groups = [row for row in expected_groups if str(row["model"]) == model]
        expected_record_means[model] = {
            "n": len(model_pairs),
            "gain_c": float(np.mean([float(row["gain_c"]) for row in model_pairs], dtype=np.float64)),
            "benefit_d": float(np.mean([float(row["gain_match"]) for row in model_pairs], dtype=np.float64)),
            "gain_background": float(np.mean([float(row["gain_background"]) for row in model_pairs], dtype=np.float64)),
            "c_positive": int(sum(bool(row["c_positive"]) for row in model_pairs)),
        }
        for metric in ("gain_c", "gain_match", "gain_background"):
            values = {str(row["source_group"]): float(row[metric]) for row in model_groups}
            expected_group_summaries[f"{model}.{metric}"] = _independent_group_summary(
                values,
                group_indices,
                metric=f"{model}.{metric}",
                bonferroni_comparisons=4,
            )
    expected_summary = {
        "status": "complete",
        "estimand": "record_pair_then_source_group_mean_then_equal_group_bootstrap",
        "record_count": len(cohort),
        "pair_count": len(expected_pairs),
        "source_group_count": 45,
        "record_means": expected_record_means,
        "group_summaries": expected_group_summaries,
        "primary_components": ["gain_match", "gain_background"],
        "bonferroni": {"comparisons": 4, "interval": "98.75%", "tail_probability": 0.00625},
        "decomposition_identity_max_abs": float(max(abs(float(row["decomposition_residual"])) for row in expected_pairs)),
    }
    for field, expected_value in expected_summary.items():
        _assert_value(expected_value, summary.get(field), f"historical.summary.{field}")
    return {"status": "valid", "cell_count": len(actual_cells), "pair_count": len(actual_pairs), "source_group_count": 45}


def _run_path(root: Path, value: Any, label: str) -> Path:
    target = Path(str(value)).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ProtocolError(f"{label} escapes the run directory: {target}") from exc
    return target


def _run_link_path(root: Path, value: Any, label: str) -> Path:
    target = Path(str(value))
    if not target.is_absolute():
        target = root / target
    target = target.absolute()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ProtocolError(f"{label} escapes the run directory: {target}") from exc
    return target


def _assert_value(expected: Any, actual: Any, label: str, *, tolerance: float = 1e-8) -> None:
    if expected is None or actual is None:
        if expected is not actual:
            raise ProtocolError(f"{label} differs: {expected!r} != {actual!r}")
        return
    if isinstance(expected, bool):
        if not isinstance(actual, bool) or expected != actual:
            raise ProtocolError(f"{label} differs: {expected!r} != {actual!r}")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(expected) != set(actual):
            raise ProtocolError(f"{label} keys differ")
        for key in expected:
            _assert_value(expected[key], actual[key], f"{label}.{key}", tolerance=tolerance)
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            raise ProtocolError(f"{label} length differs")
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _assert_value(left, right, f"{label}[{index}]", tolerance=tolerance)
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if abs(_float(actual, label) - float(expected)) > tolerance:
            raise ProtocolError(f"{label} differs: {expected!r} != {actual!r}")
        return
    if expected != actual:
        raise ProtocolError(f"{label} differs: {expected!r} != {actual!r}")


def _csv_value(raw: str, expected: Any, label: str) -> Any:
    if expected is None:
        if raw != "":
            raise ProtocolError(f"{label} should be empty")
        return None
    if isinstance(expected, bool):
        if raw.lower() not in {"true", "false"}:
            raise ProtocolError(f"{label} is not boolean")
        return raw.lower() == "true"
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(raw)
        except ValueError as exc:
            raise ProtocolError(f"{label} is not an integer") from exc
    if isinstance(expected, float):
        return _float(raw, label)
    if isinstance(expected, (list, tuple, dict)):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"{label} is not JSON") from exc
    return raw


def _quantile(values: np.ndarray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="linear"))
    except TypeError:
        return float(np.quantile(values, probability, interpolation="linear"))


def _independent_group_summary(
    values: Mapping[str, float],
    indices: np.ndarray,
    *,
    metric: str,
    bonferroni_comparisons: int | None = None,
) -> dict[str, Any]:
    labels = sorted(str(label) for label in values)
    if not labels:
        raise ProtocolError(f"cannot validate an empty group summary: {metric}")
    array = np.asarray([float(values[label]) for label in labels], dtype=np.float64)
    sampled = array[np.asarray(indices, dtype=np.int64)].mean(axis=1, dtype=np.float64)
    result: dict[str, Any] = {
        "metric": metric,
        "mean": float(array.mean(dtype=np.float64)),
        "ci95": [_quantile(sampled, 0.025), _quantile(sampled, 0.975)],
        "group_labels": labels,
        "group_means": {label: float(value) for label, value in zip(labels, array, strict=True)},
        "group_count": len(labels),
        "draws": int(indices.shape[0]),
        "seed": config.BOOTSTRAP_SEED,
        "rng": "numpy.default_rng(PCG64)",
        "quantile_method": "linear",
    }
    if bonferroni_comparisons is not None:
        tail_probability = 0.05 / (2.0 * bonferroni_comparisons)
        interval = [_quantile(sampled, tail_probability), _quantile(sampled, 1.0 - tail_probability)]
        interval_name = "ci98_75" if bonferroni_comparisons == 4 else "ci98_333" if bonferroni_comparisons == 3 else "ci_bonferroni"
        result[interval_name] = interval
        result["ci_bonferroni"] = interval
        result["bonferroni_comparisons"] = int(bonferroni_comparisons)
        result["bonferroni_tail_probability"] = tail_probability
    return result


def _bootstrap_matrix(group_count: int) -> np.ndarray:
    if group_count < 1:
        raise ProtocolError("cannot bootstrap an empty group set")
    return np.random.default_rng(config.BOOTSTRAP_SEED).integers(
        0,
        group_count,
        size=(config.BOOTSTRAP_DRAWS, group_count),
        dtype=np.int64,
    )


def _validate_matrix(root: Path, row: Mapping[str, Any], label: str) -> np.ndarray:
    matrix_path = _run_path(root, row.get("matrix"), f"{label} matrix")
    if not matrix_path.is_file() or sha256_file(matrix_path) != row.get("matrix_sha256"):
        raise ProtocolError(f"{label} matrix hash changed")
    raw_matrix = np.load(matrix_path, allow_pickle=False)
    if str(raw_matrix.dtype) != "float64":
        raise ProtocolError(f"{label} matrix dtype is not float64: {raw_matrix.dtype}")
    matrix = np.asarray(raw_matrix, dtype=np.float64)
    expected_shape = [int(item) for item in matrix.shape]
    if matrix.ndim != 2 or matrix.shape[1] != config.LAG_COUNT or matrix.shape[0] < 1 or not np.isfinite(matrix).all() or row.get("matrix_shape") != expected_shape:
        raise ProtocolError(f"{label} matrix shape is invalid: {matrix.shape}")
    return matrix


def _validate_worker(root: Path, row: Mapping[str, Any], matrix: np.ndarray, label: str) -> dict[str, Any]:
    result_path = _run_path(root, row.get("worker_result"), f"{label} worker result")
    if not result_path.is_file() or sha256_file(result_path) != row.get("worker_result_sha256"):
        raise ProtocolError(f"{label} worker result hash changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ProtocolError(f"{label} worker result is malformed")
    if row.get("worker") != result:
        # The run row carries the same immutable worker payload; compare it recursively so
        # an edited in-memory copy cannot make the manifest look consistent.
        _assert_value(result, row.get("worker"), f"{label}.worker")
    if result.get("matrix_sha256") != row.get("matrix_sha256") or result.get("matrix_shape") != [int(item) for item in matrix.shape] or result.get("matrix_dtype") != str(matrix.dtype) or result.get("vshift") != config.VSHIFT or result.get("batch_size") != config.BATCH_SIZE or result.get("extracted_pcm_verified") is not True:
        raise ProtocolError(f"{label} worker metadata is inconsistent")
    if result.get("new_forward") is not True:
        raise ProtocolError(f"{label} worker does not prove a fresh forward")
    if result.get("worker_code_sha256") != sha256_file(Path(__file__).with_name("syncnet_worker.py")):
        raise ProtocolError(f"{label} worker code hash changed")
    if result.get("syncnet_instance_sha256") != sha256_file(config.SYNCNET_ROOT / "SyncNetInstance.py") or result.get("syncnet_model_definition_sha256") != sha256_file(config.SYNCNET_ROOT / "SyncNetModel.py"):
        raise ProtocolError(f"{label} SyncNet implementation hash changed")
    ffmpeg = _ffmpeg_path()
    if result.get("ffmpeg") != str(ffmpeg.resolve()) or result.get("ffmpeg_sha256") != sha256_file(ffmpeg):
        raise ProtocolError(f"{label} ffmpeg binding changed")
    if result.get("model_sha256") != sha256_file(config.SYNCNET_MODEL):
        raise ProtocolError(f"{label} worker model hash changed")
    worker_media = Path(str(row.get("worker_media", row.get("media", "")))).resolve()
    if not worker_media.is_file() or result.get("media") != str(worker_media) or result.get("media_sha256") != sha256_file(worker_media):
        raise ProtocolError(f"{label} worker media is missing")
    media_pcm = _decode_pcm(worker_media)
    if result.get("input_pcm_sha256") != hashlib.sha256(media_pcm).hexdigest() or result.get("extracted_pcm_sha256") != hashlib.sha256(media_pcm).hexdigest():
        raise ProtocolError(f"{label} worker PCM binding differs from media")
    mean_curve = np.asarray(matrix.astype(np.float32).mean(axis=0, dtype=np.float32), dtype=np.float64)
    expected_offset = config.VSHIFT - int(np.argmin(mean_curve))
    expected_confidence = float(np.median(mean_curve) - np.min(mean_curve))
    if result.get("offset") != expected_offset or abs(_float(result.get("confidence"), f"{label}.confidence") - expected_confidence) > config.MATRIX_TOLERANCE:
        raise ProtocolError(f"{label} worker scalar summary differs from its matrix")
    return result


def _validate_selection(root: Path, row: Mapping[str, Any], label: str) -> None:
    selection_path = _run_path(root, row.get("crop_selection"), f"{label} crop selection")
    if not selection_path.is_file() or sha256_file(selection_path) != row.get("selection_sha256"):
        raise ProtocolError(f"{label} crop selection hash changed")
    selection = read_self_hashed_json(selection_path)
    if selection.get("official_pipeline_config") != config.OFFICIAL_PIPELINE_CONFIG:
        raise ProtocolError(f"{label} official pipeline configuration differs")
    if selection.get("selection_rule") != "sort by actual frame count descending, start frame ascending, crop filename lexicographically":
        raise ProtocolError(f"{label} track selection rule differs")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ProtocolError(f"{label} has no track candidates")
    ordered = sorted(candidates, key=lambda item: (-int(item["frame_count"]), int(item["start_frame"]), Path(str(item["crop_path"])).name))
    if candidates != ordered:
        raise ProtocolError(f"{label} track candidates are not in frozen selection order")
    track_indices: set[int] = set()
    for candidate in candidates:
        try:
            frame_indices = candidate["frame_indices"]
            frame_count = int(candidate["frame_count"])
            track_index = int(candidate["track_index"])
            start_frame = int(candidate["start_frame"])
            end_frame = int(candidate["end_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"{label} track candidate metadata is malformed") from exc
        if (
            track_index < 0
            or track_index in track_indices
            or not isinstance(frame_indices, list)
            or not frame_indices
            or frame_count != len(frame_indices)
            or frame_count <= config.MIN_TRACK
            or start_frame != int(frame_indices[0])
            or end_frame != int(frame_indices[-1])
        ):
            raise ProtocolError(f"{label} track candidate metadata is invalid")
        _run_link_path(root, candidate.get("crop_path"), f"{label} candidate crop")
        track_indices.add(track_index)
    selected = selection.get("selected")
    if not isinstance(selected, Mapping):
        raise ProtocolError(f"{label} selected track is missing")
    first = candidates[0]
    for field in ("track_index", "frame_count", "start_frame", "end_frame", "frame_indices", "bbox_sha256", "crop_sha256"):
        _assert_value(first.get(field), selected.get(field), f"{label}.selected.{field}")
    crop = _run_path(root, row.get("crop"), f"{label} crop")
    if str(crop) != str(selected.get("crop_path")) or row.get("crop_sha256") != selected.get("crop_sha256"):
        raise ProtocolError(f"{label} selected crop binding differs")
    if row.get("official_config") != {**config.OFFICIAL_PIPELINE_CONFIG, "vshift": config.VSHIFT, "batch_size": config.BATCH_SIZE}:
        raise ProtocolError(f"{label} scoring configuration differs")


def _validate_curve_row(root: Path, row: Mapping[str, Any], media_by_key: Mapping[tuple[int, str], Mapping[str, Any]], source_groups: Mapping[int, str], label: str) -> np.ndarray:
    sample_id = int(row.get("sample_id", -1))
    condition = str(row.get("condition", ""))
    if (sample_id, condition) not in media_by_key or sample_id not in source_groups:
        raise ProtocolError(f"{label} has an unexpected identity")
    if row.get("status") != "complete" or str(row.get("source_group")) != source_groups[sample_id]:
        raise ProtocolError(f"{label} identity or status is invalid")
    bound = media_by_key[(sample_id, condition)]
    if str(row.get("media")) != str((config.REPO / str(bound["path"])).resolve()) or row.get("media_sha256") != bound.get("sha256"):
        raise ProtocolError(f"{label} source media binding differs")
    matrix = _validate_matrix(root, row, label)
    _validate_selection(root, row, label)
    _validate_worker(root, {**row, "worker_media": row.get("crop")}, matrix, label)
    return matrix


def _validate_failed_curve_row(row: Mapping[str, Any], media_by_key: Mapping[tuple[int, str], Mapping[str, Any]], source_groups: Mapping[int, str], label: str) -> None:
    sample_id = int(row.get("sample_id", -1))
    condition = str(row.get("condition", ""))
    if (sample_id, condition) not in media_by_key or sample_id not in source_groups:
        raise ProtocolError(f"{label} has an unexpected failed-cell identity")
    bound = media_by_key[(sample_id, condition)]
    if row.get("status") != "FAILED" or str(row.get("source_group")) != source_groups[sample_id] or str(row.get("media")) != str((config.REPO / str(bound["path"])).resolve()) or row.get("media_sha256") != bound.get("sha256") or not str(row.get("reason", "")).strip():
        raise ProtocolError(f"{label} failed-cell record is malformed")


def _ffmpeg_path() -> Path:
    return config.FFMPEG if config.FFMPEG.is_file() else Path(shutil.which("ffmpeg") or "ffmpeg")


def _ffprobe_path() -> Path:
    return config.FFPROBE if config.FFPROBE.is_file() else Path(shutil.which("ffprobe") or "ffprobe")


def _decode_pcm(path: Path) -> bytes:
    ffmpeg = _ffmpeg_path()
    result = subprocess.run([str(ffmpeg), "-v", "error", "-i", str(path), "-map", "0:a:0", "-async", "1", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(config.SAMPLE_RATE), "-ac", "1", "pipe:1"], capture_output=True, check=False)
    if result.returncode != 0:
        raise ProtocolError(f"cannot decode control audio: {path}")
    return bytes(result.stdout)


def _probe_video(path: Path) -> dict[str, Any]:
    ffprobe = _ffprobe_path()
    result = subprocess.run([str(ffprobe), "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, check=False)
    if result.returncode != 0:
        raise ProtocolError(f"cannot probe control video: {path}")
    try:
        payload = json.loads(result.stdout.decode("utf-8"))
        videos = [item for item in payload.get("streams", []) if isinstance(item, Mapping) and item.get("codec_type") == "video"]
        if len(videos) != 1:
            raise ValueError("expected one video stream")
        video = videos[0]
        fps = float(Fraction(str(video.get("r_frame_rate", "0/1"))))
        frame_count = int(video.get("nb_read_frames", video.get("nb_frames", 0)) or 0)
    except (AttributeError, KeyError, StopIteration, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"control video metadata is malformed: {path}") from exc
    pts_result = subprocess.run([str(ffprobe), "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1", "-show_entries", "frame=best_effort_timestamp,best_effort_timestamp_time,pts,pts_time", "-of", "json", str(path)], capture_output=True, check=False)
    if pts_result.returncode != 0:
        raise ProtocolError(f"cannot probe first video PTS: {path}")
    try:
        frames = json.loads(pts_result.stdout.decode("utf-8")).get("frames", [])
        if not frames:
            raise ValueError("no video frame PTS")
        first_pts = dict(frames[0])
    except (AttributeError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError(f"first video PTS is unavailable: {path}") from exc
    if int(video.get("width", 0) or 0) <= 0 or int(video.get("height", 0) or 0) <= 0 or frame_count <= 0 or fps <= 0:
        raise ProtocolError(f"control video dimensions or frame rate are invalid: {path}")
    return {
        "codec_name": video.get("codec_name"),
        "width": int(video.get("width", 0) or 0),
        "height": int(video.get("height", 0) or 0),
        "frame_count": frame_count,
        "fps": fps,
        "start_time": video.get("start_time"),
        "duration": video.get("duration"),
        "time_base": video.get("time_base"),
        "first_frame_pts": first_pts,
    }


def _validate_media_audit(root: Path, bindings: Mapping[str, Any]) -> str:
    audit_path = root / "02_curves" / "media_audit.json"
    resource_path = root / "00_audit" / "resource_plan.json"
    if not audit_path.is_file() or not resource_path.is_file():
        raise ProtocolError("curve media/resource audit artifacts are missing")
    audit = read_self_hashed_json(audit_path)
    resource = read_self_hashed_json(resource_path)
    if audit.get("protocol_id") != config.PROTOCOL_ID or audit.get("expected_cell_count") != 24 or audit.get("cell_count") != 24:
        raise ProtocolError("curve media audit denominator is wrong")
    if resource.get("protocol_id") != config.PROTOCOL_ID or resource.get("media_audit_sha256") != sha256_file(audit_path) or resource.get("official_pipeline_config") != config.OFFICIAL_PIPELINE_CONFIG:
        raise ProtocolError("curve resource audit does not bind the media audit")
    expected = {(int(item["sample_id"]), str(item["condition"])): item for item in bindings.get("curve_media", [])}
    rows = audit.get("rows", [])
    if not isinstance(rows, list) or len(rows) != config.MAIN_CURVE_CELLS:
        raise ProtocolError("curve media audit rows are malformed")
    actual = {(int(row.get("sample_id", -1)), str(row.get("condition", ""))): row for row in rows}
    if set(actual) != set(expected):
        raise ProtocolError("curve media audit identities are incomplete")
    status = str(audit.get("status"))
    if status not in {"complete", "BLOCKED_INPUT_BINDING"}:
        raise ProtocolError("curve media audit status is invalid")
    failed_count = 0
    for key, bound in expected.items():
        row = actual[key]
        if row.get("status") != "PASS":
            failed_count += 1
            if status == "BLOCKED_INPUT_BINDING":
                continue
            raise ProtocolError(f"curve media audit has an unexpected failure: {key}")
        if row.get("path") != str((config.REPO / str(bound["path"])).resolve()) or row.get("media_sha256") != bound.get("sha256") or not Path(str(row.get("path"))).is_file() or sha256_file(Path(str(row["path"]))) != bound.get("sha256"):
            raise ProtocolError(f"curve media audit binding differs: {key}")
        path = Path(str(row["path"]))
        if row.get("video") != _probe_video(path):
            raise ProtocolError(f"curve media video metadata differs: {key}")
        audio = row.get("audio")
        if not isinstance(audio, Mapping):
            raise ProtocolError(f"curve media audio metadata is missing: {key}")
        pcm = _decode_pcm(path)
        if audio.get("decoded_sample_count") != len(pcm) // 2 or audio.get("decoded_pcm_sha256") != hashlib.sha256(pcm).hexdigest():
            raise ProtocolError(f"curve media PCM metadata differs: {key}")
    if status == "complete" and failed_count:
        raise ProtocolError("complete curve media audit contains failures")
    if status == "BLOCKED_INPUT_BINDING" and not failed_count:
        raise ProtocolError("blocked curve media audit contains no failure")
    return status


def _validate_controls(root: Path, controls: Mapping[str, Any], main_rows: Mapping[tuple[int, str], Mapping[str, Any]], *, allow_partial: bool = False) -> None:
    rows = controls.get("rows", [])
    if not isinstance(rows, list):
        raise ProtocolError("control manifest rows are malformed")
    run_identity = controls.get("run_identity")
    expected_statuses = {"partial", "RESOURCE_WAIT"} if allow_partial else {"complete"}
    expected_count = len(rows) if allow_partial else config.REPEAT_CELLS + config.DELAY_CELLS
    if (
        controls.get("protocol_id") != config.PROTOCOL_ID
        or controls.get("status") not in expected_statuses
        or controls.get("completed_cells") != expected_count
        or len(rows) != expected_count
        or controls.get("budget") != config.MAX_NEW_CELLS
        or controls.get("main_cells") != config.MAIN_CURVE_CELLS
        or controls.get("repeat_cells") != config.REPEAT_CELLS
        or controls.get("delay_cells") != config.DELAY_CELLS
        or not isinstance(run_identity, Mapping)
        or run_identity.get("identity_sha256") != current_run_identity().get("identity_sha256")
    ):
        raise ProtocolError("control manifest is incomplete")
    expected_keys = {(control, condition) for control in ("repeat", "delay_200ms") for condition in config.CONDITIONS}
    actual_keys = {(str(row.get("control")), str(row.get("condition"))) for row in rows}
    if (
        len(actual_keys) != len(rows)
        or (not allow_partial and actual_keys != expected_keys)
        or (allow_partial and not actual_keys.issubset(expected_keys))
    ):
        raise ProtocolError("control identities are incomplete")
    for row in rows:
        if int(row.get("sample_id", -1)) != 151:
            raise ProtocolError("control sample identity is invalid")
        condition = str(row["condition"])
        original = main_rows.get((151, condition))
        if original is None:
            raise ProtocolError(f"control has no matching main cell: {condition}")
        original_matrix = np.asarray(np.load(_run_path(root, original["matrix"], "original control matrix"), allow_pickle=False), dtype=np.float64)
        _validate_selection(root, {**row, "official_config": row.get("official_config", {})}, f"control/{row['control']}/{condition}")
        if row.get("execution_failed") is True:
            if row.get("status") != "CONTROL_FAILED" or not str(row.get("reason", "")).strip():
                raise ProtocolError(f"failed control row is malformed: {row['control']}/{condition}")
            continue
        matrix = _validate_matrix(root, row, f"control/{row['control']}/{condition}")
        worker_row: Mapping[str, Any] = {**row, "worker_media": row.get("crop")}
        if row["control"] == "delay_200ms":
            delay_media = row.get("media")
            if not isinstance(delay_media, Mapping):
                raise ProtocolError(f"delay control media metadata is missing: {condition}")
            worker_row = {**row, "media": delay_media.get("output"), "worker_media": delay_media.get("output")}
        worker = _validate_worker(root, worker_row, matrix, f"control/{row['control']}/{condition}")
        if row["control"] == "repeat":
            same_shape = bool(original_matrix.shape == matrix.shape)
            max_abs_error = float(np.max(np.abs(original_matrix - matrix))) if same_shape else None
            same_offset = worker.get("offset") == main_rows[(151, condition)].get("worker", {}).get("offset")
            if row.get("same_shape") != same_shape or row.get("max_abs_error") != max_abs_error or row.get("same_offset") != same_offset:
                raise ProtocolError(f"repeat control metadata differs: {condition}")
            expected_status = "CONTROL_PASS" if same_shape and max_abs_error <= config.MATRIX_TOLERANCE and same_offset else "CONTROL_FAILED"
        else:
            media = row.get("media")
            if not isinstance(media, Mapping):
                raise ProtocolError(f"delay control media metadata is missing: {condition}")
            output = _run_path(root, media.get("output"), f"delay/{condition} media")
            if not output.is_file() or sha256_file(output) != media.get("output_sha256"):
                raise ProtocolError(f"delay control media hash changed: {condition}")
            original_crop = _run_path(root, original["crop"], f"original/{condition} crop")
            original_pcm = _decode_pcm(original_crop)
            delayed_pcm = _decode_pcm(output)
            expected_pcm = b"\0" * (3_200 * 2) + original_pcm[: -(3_200 * 2)]
            if delayed_pcm != expected_pcm or media.get("input_pcm_sha256") != hashlib.sha256(original_pcm).hexdigest() or media.get("delayed_pcm_sha256") != hashlib.sha256(expected_pcm).hexdigest() or media.get("delay_samples") != 3_200 or media.get("sample_count") != len(original_pcm) // 2:
                raise ProtocolError(f"delay control audio transform differs: {condition}")
            original_video = _probe_video(original_crop)
            delayed_video = _probe_video(output)
            video_fields = ("codec_name", "width", "height", "frame_count", "fps", "start_time", "duration", "time_base", "first_frame_pts")
            video_unchanged = all(original_video.get(field) == delayed_video.get(field) for field in video_fields)
            if media.get("original_video") != original_video or media.get("delayed_video") != delayed_video or media.get("video_unchanged") != video_unchanged:
                raise ProtocolError(f"delay control video transform differs: {condition}")
            expected_metrics = _control_metrics(original_matrix, matrix)
            if not video_unchanged:
                expected_metrics["status"] = "CONTROL_FAILED"
            for field in (
                "status",
                "common_rows",
                "offset_change_delayed_minus_original",
                "expected_offset_change",
                "original_argmin_delayed_distance_higher",
                "original_boundary_best",
                "delayed_boundary_best",
                "boundary_ok",
                "rule",
            ):
                _assert_value(expected_metrics.get(field), row.get(field), f"delay/{condition}.{field}")
            expected_status = expected_metrics["status"]
        if row.get("status") != expected_status:
            raise ProtocolError(f"control status differs: {row['control']}/{condition}")


def _validate_curve(root: Path, cohort: Sequence[Mapping[str, Any]], bindings: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = root / "02_curves" / "manifest.json"
    if not manifest_path.is_file():
        return {"status": "not_run", "main_cell_count": 0, "control_cell_count": 0}
    manifest = read_self_hashed_json(manifest_path)
    status = str(manifest.get("status"))
    if manifest.get("run_identity", {}).get("identity_sha256") != current_run_identity().get("identity_sha256"):
        raise ProtocolError("curve manifest run identity differs from current code/config")
    media_audit_status = _validate_media_audit(root, bindings)
    rows = manifest.get("rows", [])
    if not isinstance(rows, list) or len(rows) > config.MAIN_CURVE_CELLS:
        raise ProtocolError("curve manifest count is inconsistent")
    complete_main_count = sum(row.get("status") == "complete" for row in rows if isinstance(row, Mapping))
    failed_main_count = sum(row.get("status") == "FAILED" for row in rows if isinstance(row, Mapping))
    if manifest.get("completed_main_cells") != complete_main_count or manifest.get("failed_main_cells", 0) != failed_main_count:
        raise ProtocolError("curve manifest main-cell counts are inconsistent")
    if manifest.get("protocol_id") != config.PROTOCOL_ID or manifest.get("expected_main_cells") != config.MAIN_CURVE_CELLS or manifest.get("input_audit_sha256") != sha256_file(root / "00_audit" / "inputs.json"):
        raise ProtocolError("curve manifest input binding is inconsistent")
    media_bindings = {(int(item["sample_id"]), str(item["condition"])): item for item in bindings.get("curve_media", [])}
    source_groups = {int(row["sample_id"]): str(row["source_group"]) for row in cohort}
    if status == "complete":
        for key, bound in media_bindings.items():
            media_path = Path(str((config.REPO / str(bound["path"])).resolve()))
            if not media_path.is_file() or sha256_file(media_path) != bound.get("sha256"):
                raise ProtocolError(f"curve source media changed: {key}")
    if status in {"RESOURCE_WAIT", "BLOCKED_INPUT_BINDING", "PARTIAL"}:
        if (status == "RESOURCE_WAIT" and media_audit_status != "complete") or (status == "BLOCKED_INPUT_BINDING" and media_audit_status != "BLOCKED_INPUT_BINDING"):
            raise ProtocolError("curve manifest status disagrees with media audit")
        if status == "PARTIAL" and media_audit_status != "complete":
            raise ProtocolError("partial curve manifest has an incomplete media audit")
        if status == "BLOCKED_INPUT_BINDING" and rows:
            raise ProtocolError("blocked input manifest contains completed rows")
        completed_controls = int(manifest.get("completed_control_cells", 0))
        if status == "BLOCKED_INPUT_BINDING" and completed_controls != 0:
            raise ProtocolError("blocked curve manifest contains control cells")
        if completed_controls < 0 or completed_controls > config.REPEAT_CELLS + config.DELAY_CELLS:
            raise ProtocolError("resource-wait curve manifest control count is invalid")
        if manifest.get("new_syncnet_cells_attempted") != len(rows) + completed_controls or manifest.get("new_syncnet_cells_attempted") > config.MAX_NEW_CELLS:
            raise ProtocolError("partial curve manifest exceeds the SyncNet cell budget")
        expected_keys = {(sample_id, condition) for sample_id in config.CURVE_SAMPLE_IDS for condition in config.CONDITIONS}
        actual_keys = {(int(row.get("sample_id", -1)), str(row.get("condition", ""))) for row in rows}
        if len(actual_keys) != len(rows) or not actual_keys.issubset(expected_keys):
            raise ProtocolError("partial curve manifest cell identities are invalid")
        if status == "PARTIAL" and actual_keys != expected_keys:
            raise ProtocolError("partial curve manifest does not retain every failed cell")
        for row in rows:
            label = f"partial/{row.get('cell_name')}"
            if row.get("status") == "complete":
                _validate_curve_row(root, row, media_bindings, source_groups, label)
            elif row.get("status") == "FAILED":
                _validate_failed_curve_row(row, media_bindings, source_groups, label)
            else:
                raise ProtocolError(f"{label} has an invalid status")
        controls_path = root / "02_curves" / "controls.json"
        if not controls_path.is_file():
            raise ProtocolError("resource-wait curve manifest is missing its control checkpoint")
        controls = read_self_hashed_json(controls_path)
        expected_control_status = "RESOURCE_WAIT" if status == "RESOURCE_WAIT" else "BLOCKED_INPUT_BINDING" if status == "BLOCKED_INPUT_BINDING" else "complete" if completed_controls == config.REPEAT_CELLS + config.DELAY_CELLS else "partial"
        control_rows = controls.get("rows")
        control_identity = controls.get("run_identity")
        if (
            controls.get("status") != expected_control_status
            or controls.get("completed_cells") != completed_controls
            or not isinstance(control_rows, list)
            or len(control_rows) != completed_controls
            or not isinstance(control_identity, Mapping)
            or control_identity.get("identity_sha256") != current_run_identity().get("identity_sha256")
        ):
            raise ProtocolError("resource-wait control checkpoint is inconsistent")
        if completed_controls:
            id151_rows = [row for row in rows if int(row.get("sample_id", -1)) == 151]
            if {str(row.get("condition")) for row in id151_rows} != set(config.CONDITIONS) or not all(row.get("status") == "complete" for row in id151_rows):
                raise ProtocolError("resource-wait controls require all main curve cells")
            _validate_controls(
                root,
                controls,
                {(int(row["sample_id"]), str(row["condition"])): row for row in rows},
                allow_partial=controls.get("status") != "complete",
            )
        analysis_path = root / "03_analysis" / "summary.json"
        if not analysis_path.is_file():
            raise ProtocolError("incomplete curve analysis summary is missing")
        analysis = read_self_hashed_json(analysis_path)
        expected_analysis = {
            "status": status,
            "protocol_id": config.PROTOCOL_ID,
            "reason": manifest.get("failures", []),
            "main_cells": complete_main_count,
            "controls": completed_controls,
        }
        for field, expected_value in expected_analysis.items():
            _assert_value(expected_value, analysis.get(field), f"incomplete.analysis.{field}")
        return {"status": status, "main_cell_count": complete_main_count, "failed_main_cell_count": failed_main_count, "control_cell_count": completed_controls, "reason": manifest.get("failures", [])}
    if status != "complete" or len(rows) != config.MAIN_CURVE_CELLS or manifest.get("completed_control_cells") != 4:
        raise ProtocolError("curve manifest is incomplete")
    if manifest.get("new_syncnet_cells_attempted") != config.MAX_NEW_CELLS:
        raise ProtocolError("complete curve manifest has an invalid SyncNet cell count")
    if manifest.get("model_sha256") != sha256_file(config.SYNCNET_MODEL):
        raise ProtocolError("curve model hash changed")
    current_identity = current_run_identity()
    if manifest.get("total_new_syncnet_cells") != config.MAX_NEW_CELLS or manifest.get("code_sha256") != current_identity["code_sha256"]:
        raise ProtocolError("curve implementation or budget binding changed")
    expected_keys = {(sample_id, condition) for sample_id in config.CURVE_SAMPLE_IDS for condition in config.CONDITIONS}
    actual_keys = {(int(row["sample_id"]), str(row["condition"])) for row in rows}
    if actual_keys != expected_keys:
        raise ProtocolError("curve main cell identity is incomplete")
    by_key = {(int(row["sample_id"]), str(row["condition"])): row for row in rows}
    for key, row in by_key.items():
        _validate_curve_row(root, row, media_bindings, source_groups, f"curve/{key[0]}/{key[1]}")
    endpoints: list[dict[str, Any]] = []
    for sample_id in config.CURVE_SAMPLE_IDS:
        n = np.asarray(np.load(_run_path(root, by_key[(sample_id, "natural_raw")]["matrix"], "natural curve matrix"), allow_pickle=False), dtype=np.float64)
        t = np.asarray(np.load(_run_path(root, by_key[(sample_id, "tts_raw")]["matrix"], "tts curve matrix"), allow_pickle=False), dtype=np.float64)
        current, _ = pair_curve_metrics(n, t, sample_id=sample_id, source_group=source_groups[sample_id])
        endpoints.extend(current)
    expected_analysis = curve_analysis(endpoints, source_groups)
    analysis_summary_path = root / "03_analysis" / "summary.json"
    if not analysis_summary_path.is_file():
        raise ProtocolError("curve analysis summary is missing")
    analysis = read_self_hashed_json(analysis_summary_path)
    for field in ("status", "record_count", "source_group_count", "endpoints_count", "paired_count", "exploratory_metrics", "bonferroni"):
        _assert_value(expected_analysis[field], analysis.get(field), f"analysis.{field}")
    _assert_value(expected_analysis["paired"], analysis.get("paired"), "analysis.paired")
    _assert_value(expected_analysis["core_summaries"], analysis.get("core_summaries"), "analysis.core_summaries")
    if analysis.get("manifest_sha256") != sha256_file(manifest_path):
        raise ProtocolError("analysis does not bind the current curve manifest")
    actual_endpoints = csv_read(root / "03_analysis" / "endpoints.csv")
    if len(actual_endpoints) != len(endpoints):
        raise ProtocolError("curve endpoint table count is wrong")
    endpoint_fields = ("model", "sample_id", "source_group", "condition", "support", "support_count", "sync_d", "sync_c", "background_b", "d0", "search_gain_s", "c5", "official_offset", "min_index", "boundary_best", "trough_width_ms", "trough_width_status", "curve", "support_rows", "offsets", "trough_ties")
    for actual, expected in zip(actual_endpoints, endpoints, strict=True):
        for field in endpoint_fields:
            _assert_value(expected[field], _csv_value(actual.get(field, ""), expected[field], f"endpoint.{field}"), f"endpoint.{field}")
    actual_paired = csv_read(root / "03_analysis" / "paired.csv")
    if len(actual_paired) != len(expected_analysis["paired"]):
        raise ProtocolError("curve paired table count is wrong")
    for actual, expected in zip(actual_paired, expected_analysis["paired"], strict=True):
        for field, expected_value in expected.items():
            _assert_value(expected_value, _csv_value(actual.get(field, ""), expected_value, f"paired.{field}"), f"paired.{field}")
    controls = read_self_hashed_json(root / "02_curves" / "controls.json")
    _validate_controls(root, controls, by_key)
    figures = root / "03_analysis" / "figures"
    for stem in ("historical_decomposition", "leaptalk_curves", "support_sensitivity"):
        for suffix in (".png", ".svg"):
            if not (figures / f"{stem}{suffix}").is_file():
                raise ProtocolError(f"required figure is missing: {stem}{suffix}")
    return {"status": "valid", "main_cell_count": 24, "control_cell_count": 4, "endpoint_count": 72}


def _validate_perception(root: Path, bindings: Mapping[str, Any], cohort: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    directory = root / "04_perception"
    status_path = directory / "status.json"
    if status_path.is_file():
        status = read_self_hashed_json(status_path)
        if status.get("protocol_id") != config.PROTOCOL_ID or status.get("status") != "BLOCKED_INPUT_BINDING" or status.get("pair_count") != len(config.CURVE_SAMPLE_IDS) or status.get("clip_count") != 0 or not isinstance(status.get("failures"), list):
            raise ProtocolError("perception blocked-status artifact is inconsistent")
        expected_failures = []
        for item in bindings.get("curve_media", []):
            path = config.REPO / str(item["path"])
            if not path.is_file() or sha256_file(path) != str(item["sha256"]):
                expected_failures.append(f"{int(item['sample_id'])}/{item['condition']}: media is missing or its hash changed")
        if not expected_failures or status.get("failures") != expected_failures:
            raise ProtocolError("perception blocked-status failures do not match current media bindings")
        return {"status": "valid", "perception": "BLOCKED_INPUT_BINDING", "pair_count": 12}
    required = [directory / name for name in ("clips.json", "assignments.csv", "ratings_template.csv", "private_key.json")]
    if not all(path.is_file() for path in required):
        return {"status": "not_run"}
    clips = read_self_hashed_json(directory / "clips.json")
    assignments = csv_read(directory / "assignments.csv")
    key = read_self_hashed_json(directory / "private_key.json")
    if (
        clips.get("protocol_id") != config.PROTOCOL_ID
        or clips.get("status") != "READY"
        or clips.get("clip_count") != 24
        or clips.get("pair_count") != len(config.CURVE_SAMPLE_IDS)
        or len(assignments) != len(config.CURVE_SAMPLE_IDS)
        or key.get("schema_version") != 1
        or key.get("protocol_id") != config.PROTOCOL_ID
        or key.get("seed") != config.BOOTSTRAP_SEED
        or len(key.get("rows", [])) != len(config.CURVE_SAMPLE_IDS)
    ):
        raise ProtocolError("perception package count is wrong")
    if stat.S_IMODE((directory / "private_key.json").stat().st_mode) != 0o600:
        raise ProtocolError("perception private key permissions are not 0600")
    assignment_map = {int(row["sample_id"]): row for row in assignments}
    key_map = {int(row["sample_id"]): row for row in key["rows"]}
    if set(assignment_map) != set(config.CURVE_SAMPLE_IDS) or set(key_map) != set(config.CURVE_SAMPLE_IDS):
        raise ProtocolError("perception assignment IDs are wrong")
    source_groups = {int(row["sample_id"]): str(row["source_group"]) for row in cohort}
    media_bindings = {(int(item["sample_id"]), str(item["condition"])): item for item in bindings.get("curve_media", [])}
    clip_map = {str(row.get("clip_id")): row for row in clips.get("clips", [])}
    if not isinstance(clips.get("clips"), list) or len(clips["clips"]) != 24 or len(clip_map) != 24:
        raise ProtocolError("perception clip identities are duplicated")
    expected_rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    for sample_id in config.CURVE_SAMPLE_IDS:
        mapping = key_map[sample_id]
        expected_order = list(expected_rng.permutation(list(config.CONDITIONS)))
        expected_mapping = {
            "sample_id": sample_id,
            "source_group": source_groups[sample_id],
            "a_condition": expected_order[0],
            "b_condition": expected_order[1],
        }
        _assert_value(expected_mapping, mapping, f"perception.private_key.{sample_id}")
        assignment = assignment_map[sample_id]
        if (
            assignment.get("source_group") != source_groups[sample_id]
            or assignment.get("a_id") != "A"
            or assignment.get("b_id") != "B"
            or assignment.get("clip_a") != str(directory / "media" / f"{sample_id}_A.mp4")
            or assignment.get("clip_b") != str(directory / "media" / f"{sample_id}_B.mp4")
        ):
            raise ProtocolError(f"perception assignment binding is invalid: {sample_id}")
        for label in ("A", "B"):
            clip_id = f"{sample_id}_{label}"
            clip = clip_map.get(clip_id)
            if clip is None or clip.get("label") != label or int(clip.get("sample_id", -1)) != sample_id or clip.get("source_group") != source_groups[sample_id] or clip.get("full_clip_reference") is not True:
                raise ProtocolError(f"perception clip identity is invalid: {clip_id}")
            public_path = _run_link_path(root, clip.get("path"), f"perception/{clip_id}")
            if not public_path.is_file() or sha256_file(public_path) != clip.get("sha256"):
                raise ProtocolError(f"perception clip hash changed: {clip_id}")
            condition = mapping[f"{label.lower()}_condition"]
            bound = media_bindings[(sample_id, condition)]
            if public_path.resolve() != (config.REPO / str(bound["path"])).resolve() or clip.get("sha256") != bound.get("sha256"):
                raise ProtocolError(f"perception clip source binding differs: {clip_id}")
            expected_path = directory / "media" / f"{sample_id}_{label}.mp4"
            if public_path != expected_path.absolute():
                raise ProtocolError(f"perception clip path is not opaque: {clip_id}")
            if assignment.get(f"clip_{label.lower()}") != str(expected_path):
                raise ProtocolError(f"perception assignment path is invalid: {sample_id}/{label}")
    ratings = csv_read(directory / "ratings_template.csv")
    analysis_path = directory / "analysis.json"
    if not analysis_path.is_file():
        raise ProtocolError("perception analysis artifact is missing")
    analysis = read_self_hashed_json(analysis_path)
    if analysis.get("ratings_sha256") != sha256_file(directory / "ratings_template.csv") or analysis.get("assignments_sha256") != sha256_file(directory / "assignments.csv") or analysis.get("private_key_sha256") != sha256_file(directory / "private_key.json"):
        raise ProtocolError("perception analysis does not bind its input files")
    if not ratings:
        expected_empty = {
            "status": "NOT_ASSESSED",
            "pair_count": len(config.CURVE_SAMPLE_IDS),
            "rating_row_count": 0,
            "rater_count": 0,
            "complete_rater_count": 0,
            "complete_raters": [],
            "ratings_by_rater": {},
            "unjudgeable_count": 0,
        }
        for field, expected_value in expected_empty.items():
            _assert_value(expected_value, analysis.get(field), f"perception.analysis.{field}")
        return {"status": "valid", "perception": "NOT_ASSESSED", "pair_count": 12}
    by_rater: dict[str, dict[int, Mapping[str, str]]] = {}
    seen: set[tuple[str, int]] = set()
    preferences = {"a", "b", "tie", "unjudgeable"}
    for row in ratings:
        rater = str(row.get("rater_id", "")).strip()
        try:
            sample_id = int(row.get("sample_id", ""))
        except ValueError as exc:
            raise ProtocolError("perception rating has an invalid sample_id") from exc
        if not rater or sample_id not in set(config.CURVE_SAMPLE_IDS) or (rater, sample_id) in seen:
            raise ProtocolError("perception rating identity is invalid or duplicated")
        seen.add((rater, sample_id))
        by_rater.setdefault(rater, {})[sample_id] = row
        if str(row.get("sync_preference", "")).strip().lower() not in preferences:
            raise ProtocolError("perception preference is invalid")
        for field in ("sync_rating_a_1_5", "sync_rating_b_1_5", "naturalness_rating_a_1_5", "naturalness_rating_b_1_5"):
            value = _float(row.get(field), f"perception/{rater}/{sample_id}/{field}")
            if value < 1 or value > 5 or value != round(value):
                raise ProtocolError(f"perception rating is outside 1..5: {field}")
    expected_ids = set(config.CURVE_SAMPLE_IDS)
    complete_raters = sorted(rater for rater, rating_rows in by_rater.items() if set(rating_rows) == expected_ids)
    ratings_by_rater = {rater: len(rating_rows) for rater, rating_rows in sorted(by_rater.items())}
    unjudgeable_count = sum(
        str(row.get("sync_preference", "")).strip().lower() == "unjudgeable"
        for rater in complete_raters
        for row in by_rater[rater].values()
    )
    expected_status = "COMPLETE" if len(complete_raters) >= 3 else "PARTIAL"
    expected_base = {
        "status": expected_status,
        "pair_count": len(config.CURVE_SAMPLE_IDS),
        "rating_row_count": len(ratings),
        "rater_count": len(by_rater),
        "complete_rater_count": len(complete_raters),
        "complete_raters": complete_raters,
        "ratings_by_rater": ratings_by_rater,
        "unjudgeable_count": unjudgeable_count,
    }
    for field, expected_value in expected_base.items():
        _assert_value(expected_value, analysis.get(field), f"perception.analysis.{field}")
    if expected_status == "PARTIAL":
        if "pair_summaries" in analysis or "group_bootstrap" in analysis:
            raise ProtocolError("partial perception analysis contains inferential summaries")
        return {"status": "valid", "perception": expected_status, "pair_count": 12, "complete_rater_count": len(complete_raters)}
    pair_rows: list[dict[str, Any]] = []
    for sample_id in sorted(expected_ids):
        mapping = key_map[sample_id]
        sync_deltas: list[float] = []
        naturalness_deltas: list[float] = []
        preference_values: list[bool] = []
        ties = 0
        unjudgeable = 0
        for rater in complete_raters:
            row = by_rater[rater][sample_id]
            a_sync = float(row["sync_rating_a_1_5"])
            b_sync = float(row["sync_rating_b_1_5"])
            a_naturalness = float(row["naturalness_rating_a_1_5"])
            b_naturalness = float(row["naturalness_rating_b_1_5"])
            sync_by_condition = {mapping["a_condition"]: a_sync, mapping["b_condition"]: b_sync}
            naturalness_by_condition = {mapping["a_condition"]: a_naturalness, mapping["b_condition"]: b_naturalness}
            sync_deltas.append(sync_by_condition["tts_raw"] - sync_by_condition["natural_raw"])
            naturalness_deltas.append(naturalness_by_condition["tts_raw"] - naturalness_by_condition["natural_raw"])
            preference = str(row["sync_preference"]).strip().lower()
            if preference == "tie":
                ties += 1
            elif preference == "unjudgeable":
                unjudgeable += 1
            else:
                preference_values.append(mapping[f"{preference}_condition"] == "tts_raw")
        pair_rows.append({
            "sample_id": sample_id,
            "source_group": mapping["source_group"],
            "sync_rating_delta_tts_minus_natural": float(np.mean(sync_deltas)),
            "naturalness_delta_tts_minus_natural": float(np.mean(naturalness_deltas)),
            "tts_preference_rate_among_judged": None if not preference_values else float(np.mean(preference_values)),
            "tie_rate": float(ties / len(complete_raters)),
            "unjudgeable_rate": float(unjudgeable / len(complete_raters)),
        })
    _assert_value(pair_rows, analysis.get("pair_summaries"), "perception.analysis.pair_summaries")
    group_indices = _bootstrap_matrix(len(pair_rows))
    expected_bootstrap = {
        metric: _independent_group_summary(
            {str(row["source_group"]): float(row[metric]) for row in pair_rows if row[metric] is not None},
            group_indices,
            metric=metric,
        )
        for metric in ("sync_rating_delta_tts_minus_natural", "naturalness_delta_tts_minus_natural")
    }
    preference_values = {str(row["source_group"]): float(row["tts_preference_rate_among_judged"]) for row in pair_rows if row["tts_preference_rate_among_judged"] is not None}
    if len(preference_values) == len(pair_rows):
        expected_bootstrap["tts_preference_rate_among_judged"] = _independent_group_summary(
            preference_values,
            group_indices,
            metric="tts_preference_rate_among_judged",
        )
    _assert_value(expected_bootstrap, analysis.get("group_bootstrap"), "perception.analysis.group_bootstrap")
    return {"status": "valid", "perception": expected_status, "pair_count": 12, "complete_rater_count": len(complete_raters)}


def validate_run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    inputs_path = root / "00_audit" / "inputs.json"
    cohort_path = root / "00_audit" / "cohort.json"
    if not inputs_path.is_file() or not cohort_path.is_file():
        raise ProtocolError("audit artifacts are missing")
    inputs = read_self_hashed_json(inputs_path)
    if inputs.get("status") != "complete" or inputs.get("record_count") != 50 or inputs.get("source_group_count") != 45:
        raise ProtocolError("audit status or denominator is wrong")
    bindings = load_input_bindings()
    current_identity = current_run_identity()
    if inputs.get("run_identity") != current_identity:
        raise ProtocolError("audit run identity differs from current code/config")
    input_binding_meta = inputs.get("bindings")
    if not isinstance(input_binding_meta, Mapping):
        raise ProtocolError("audit input binding metadata is missing")
    if (
        input_binding_meta.get("sha256") != sha256_file(config.INPUT_BINDINGS)
        or input_binding_meta.get("file_count") != len(bindings.get("files", []))
        or input_binding_meta.get("curve_media_count") != len(bindings.get("curve_media", []))
        or inputs.get("historical_cell_count") != len(config.HISTORICAL_SAMPLE_IDS) * len(config.MODELS) * len(config.CONDITIONS)
        or inputs.get("historical_pair_count") != len(config.HISTORICAL_SAMPLE_IDS) * len(config.MODELS)
    ):
        raise ProtocolError("audit artifact counts or input binding is inconsistent")
    verify_bound_files(bindings, include_curve_media=False)
    cohort, group_for = build_cohort(bindings)
    raw_cells, _ = load_historical_records(cohort)
    score_crosscheck = verify_bound_score_cells(raw_cells, bindings)
    cohort_artifact = read_self_hashed_json(cohort_path)
    if cohort_artifact.get("sample_ids_sha256") != sample_ids_sha256([int(row["sample_id"]) for row in cohort]):
        raise ProtocolError("cohort sample identity changed")
    _assert_value(cohort, cohort_artifact.get("records"), "cohort.records")
    _assert_value(sorted({str(row["source_group"]) for row in cohort}), cohort_artifact.get("source_groups"), "cohort.source_groups")
    artifact_group_for = cohort_artifact.get("group_for_sample_id")
    if not isinstance(artifact_group_for, Mapping):
        raise ProtocolError("cohort.group_for_sample_id is missing")
    try:
        artifact_group_for = {int(sample_id): str(source_group) for sample_id, source_group in artifact_group_for.items()}
    except (TypeError, ValueError) as exc:
        raise ProtocolError("cohort.group_for_sample_id is malformed") from exc
    _assert_value(group_for, artifact_group_for, "cohort.group_for_sample_id")
    historical = _validate_historical(root, cohort)
    curve = _validate_curve(root, cohort, bindings)
    perception = _validate_perception(root, bindings, cohort)
    final_result: dict[str, Any]
    final_path = root / "final.json"
    report_path = root / "report.md"
    if not final_path.is_file() or not report_path.is_file():
        raise ProtocolError("final report artifacts are missing")
    final = read_self_hashed_json(final_path)
    if final.get("protocol_id") != config.PROTOCOL_ID or final.get("report_sha256") != sha256_file(report_path):
        raise ProtocolError("final artifact does not bind this protocol/report")
    if final.get("run_identity") != inputs.get("run_identity") or final.get("run_identity", {}).get("identity_sha256") != current_identity.get("identity_sha256"):
        raise ProtocolError("final artifact run identity differs from current code/config")
    curve_artifact_status = "complete" if curve.get("status") == "valid" else str(curve.get("status"))
    expected_final_status = "complete" if curve_artifact_status == "complete" else "RESOURCE_WAIT" if curve_artifact_status == "RESOURCE_WAIT" else "partial"
    if final.get("status") != expected_final_status or bool(final.get("automatic_complete")) != bool(curve.get("status") == "valid"):
        raise ProtocolError("final status disagrees with independent stage validation")
    expected_counts = {
        "historical_cells": 200,
        "historical_pairs": 100,
        "historical_source_groups": 45,
        "curve_main_cells": 24,
        "curve_control_cells": 4,
        "new_syncnet_cells_max": config.MAX_NEW_CELLS,
        "perception_pairs": 12,
    }
    if final.get("counts") != expected_counts:
        raise ProtocolError("final counts do not match the protocol denominator")
    expected_attempted_cells = int(curve.get("main_cell_count", 0)) + int(curve.get("failed_main_cell_count", 0)) + int(curve.get("control_cell_count", 0))
    if final.get("budget") != {"max_new_syncnet_cells": config.MAX_NEW_CELLS, "attempted_new_syncnet_cells": expected_attempted_cells}:
        raise ProtocolError("final SyncNet budget binding is inconsistent")
    if final.get("historical_decomposition", {}).get("status") != "complete" or final.get("curve_analysis", {}).get("status") != curve_artifact_status:
        raise ProtocolError("final stage status binding is inconsistent")
    controls_path = root / "02_curves" / "controls.json"
    controls_status = read_self_hashed_json(controls_path).get("status") if controls_path.is_file() else "NOT_RUN"
    if final.get("controls", {}).get("status") != controls_status:
        raise ProtocolError("final control status binding is inconsistent")
    summary_bindings = (
        ("historical_decomposition", root / "01_decomposition" / "summary.json"),
        ("curve_analysis", root / "03_analysis" / "summary.json"),
        ("controls", controls_path),
    )
    for field, path in summary_bindings:
        descriptor = final.get(field)
        if not isinstance(descriptor, Mapping):
            raise ProtocolError(f"final {field} binding is missing")
        expected_path = str(path.resolve()) if path.is_file() else None
        expected_hash = sha256_file(path) if path.is_file() else None
        if descriptor.get("summary") != expected_path or descriptor.get("summary_sha256") != expected_hash:
            raise ProtocolError(f"final {field} summary binding is inconsistent")
    perception_status = str(final.get("perception", {}).get("status", "NOT_ASSESSED"))
    if perception_status != str(perception.get("perception", "NOT_ASSESSED")):
        raise ProtocolError("final perception status binding is inconsistent")
    final_perception = final.get("perception")
    if not isinstance(final_perception, Mapping):
        raise ProtocolError("final perception payload is missing")
    perception_artifact_path = root / "04_perception" / ("status.json" if perception_status == "BLOCKED_INPUT_BINDING" else "analysis.json")
    perception_artifact = read_self_hashed_json(perception_artifact_path)
    perception_body = {key: value for key, value in perception_artifact.items() if key != "artifact_sha256"}
    for key, expected_value in perception_body.items():
        _assert_value(expected_value, final_perception.get(key), f"final.perception.{key}")
    for field in ("zero_new_tts", "zero_new_tfg", "zero_training", "zero_cloud_calls"):
        if final.get(field) is not True:
            raise ProtocolError(f"final resource invariant is false: {field}")
    final_result = {"status": "valid", "automatic_complete": bool(final.get("automatic_complete"))}
    return {"status": "valid", "protocol_id": config.PROTOCOL_ID, "stages": {"audit": "valid", "historical": {**historical, "score_crosscheck": score_crosscheck}, "curves": curve, "perception": perception}, "final": final_result}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently validate the LRS3 TTS gain mechanism run")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate_run(args.root)
    payload = {"schema_version": 1, **result}
    write_self_hashed_json(args.root.resolve() / "validation.json", payload)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
