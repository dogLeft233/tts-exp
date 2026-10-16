#!/usr/bin/env python3
"""Describe mouth aperture and adjacent-frame speed for the frozen P1 videos.

This is an independent, read-only video postprocessor. It never changes or
rewrites P1 score inputs. The production observer matches the existing Tasks
API mouth code: landmark 13--14 gap normalized by eye points 33--263.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence


FPS = Fraction(25, 1)
CONDITION_IDS = (
    "RAW_R1",
    "RAW_R2",
    "FIXED_R1",
    "FIXED_R2",
    "DOSE_M6",
    "DOSE_M3",
    "DOSE_P3",
    "DOSE_P6",
)
SPEAKER_IDS = tuple(
    [f"S{i:04d}" for i in range(102, 112)]
    + [f"S{i:04d}" for i in range(144, 154)]
)
LANDMARK_INDICES = (13, 14, 33, 263)
MIN_EYE_DISTANCE_PX = 1e-6
VALIDITY_REASON_CODES = {
    "valid": 0,
    "no_unique_face": 1,
    "required_landmark_missing_or_nonfinite": 2,
    "eye_distance_not_positive": 3,
    "opening_ratio_nonfinite": 4,
    "landmark_count_below_required_index": 5,
}

# Frozen P1 measurement-qualification rule. This is a measurement gate only;
# it does not filter or qualify SyncNet score rows.
MIN_COMMON_COVERAGE = 0.90
MIN_COMMON_VALID_FRAMES = 25
MIN_COMMON_ADJACENT_EDGES = 10

INPUT_SCHEMA = "p1_mouth_video_input_v1"
OUTPUT_SCHEMA = "p1_mouth_geometry_overlay_v1"
SUCCESS_RENDER_STATUSES = frozenset({"PASS", "RENDERED"})


class MouthPostprocessError(RuntimeError):
    pass


def _is_render_success(row: dict[str, Any]) -> bool:
    return row.get("status") in SUCCESS_RENDER_STATUSES


@dataclass(frozen=True)
class GeometryPoint:
    valid: bool
    opening_ratio: float | None
    reason: str | None


def _finite_xy(landmarks: Sequence[Sequence[float]], index: int) -> tuple[float, float] | None:
    if index >= len(landmarks):
        return None
    point = landmarks[index]
    if len(point) < 2:
        return None
    x, y = float(point[0]), float(point[1])
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return x, y


def opening_geometry(
    landmarks: Sequence[Sequence[float]], width: int, height: int
) -> GeometryPoint:
    """Return pixel-space lip gap / pixel-space inter-eye distance."""
    if width <= 0 or height <= 0:
        return GeometryPoint(False, None, "invalid_frame_dimensions")
    mouth_upper = _finite_xy(landmarks, 13)
    mouth_lower = _finite_xy(landmarks, 14)
    eye_left = _finite_xy(landmarks, 33)
    eye_right = _finite_xy(landmarks, 263)
    if any(point is None for point in (mouth_upper, mouth_lower, eye_left, eye_right)):
        return GeometryPoint(False, None, "required_landmark_missing_or_nonfinite")

    assert mouth_upper is not None and mouth_lower is not None
    assert eye_left is not None and eye_right is not None
    upper_px = (mouth_upper[0] * width, mouth_upper[1] * height)
    lower_px = (mouth_lower[0] * width, mouth_lower[1] * height)
    left_px = (eye_left[0] * width, eye_left[1] * height)
    right_px = (eye_right[0] * width, eye_right[1] * height)
    mouth_gap = math.hypot(lower_px[0] - upper_px[0], lower_px[1] - upper_px[1])
    eye_distance = math.hypot(right_px[0] - left_px[0], right_px[1] - left_px[1])
    if not math.isfinite(eye_distance) or eye_distance <= MIN_EYE_DISTANCE_PX:
        return GeometryPoint(False, None, "eye_distance_not_positive")
    ratio = mouth_gap / eye_distance
    if not math.isfinite(ratio):
        return GeometryPoint(False, None, "opening_ratio_nonfinite")
    return GeometryPoint(True, ratio, None)


def common_valid_mask(condition_masks: Sequence[Sequence[bool]]) -> list[bool]:
    """AND the exact per-frame masks for the eight ordered P1 conditions."""
    if len(condition_masks) != len(CONDITION_IDS):
        raise ValueError(f"expected exactly {len(CONDITION_IDS)} condition masks")
    lengths = {len(mask) for mask in condition_masks}
    if len(lengths) != 1:
        raise ValueError("condition masks must share an identical frame grid")
    return [all(bool(mask[i]) for mask in condition_masks) for i in range(next(iter(lengths), 0))]


def adjacent_speed(
    opening: Sequence[float | None], valid: Sequence[bool], fps: Fraction = FPS
) -> tuple[list[float | None], list[bool]]:
    """Return absolute opening speed on original adjacent valid frames only."""
    if len(opening) != len(valid):
        raise ValueError("opening and validity arrays must have identical length")
    if fps <= 0:
        raise ValueError("fps must be positive")
    velocities: list[float | None] = []
    interval_valid: list[bool] = []
    rate = float(fps)
    for i in range(max(0, len(opening) - 1)):
        okay = bool(valid[i]) and bool(valid[i + 1])
        left, right = opening[i], opening[i + 1]
        okay = okay and left is not None and right is not None
        if okay:
            value = abs(float(right) - float(left)) * rate
            okay = math.isfinite(value)
        else:
            value = 0.0
        interval_valid.append(bool(okay))
        velocities.append(float(value) if okay else None)
    return velocities, interval_valid


def qualification(
    *,
    frame_count: int,
    common_mask: Sequence[bool],
    metadata_aligned: bool,
) -> dict[str, Any]:
    support_known = bool(metadata_aligned and len(common_mask) == frame_count)
    common_n = sum(bool(value) for value in common_mask) if support_known else None
    edge_n = (
        sum(bool(common_mask[i]) and bool(common_mask[i + 1]) for i in range(max(0, len(common_mask) - 1)))
        if support_known
        else None
    )
    coverage = common_n / frame_count if support_known and frame_count else None
    passed = bool(
        support_known
        and coverage is not None
        and coverage >= MIN_COMMON_COVERAGE
        and common_n is not None
        and common_n >= MIN_COMMON_VALID_FRAMES
        and edge_n is not None
        and edge_n >= MIN_COMMON_ADJACENT_EDGES
    )
    return {
        "status": "PASS_MOUTH_MEASUREMENT_QUALIFICATION" if passed else "UNMEASURABLE",
        "eligible_for_mouth_description": passed,
        "score_rows_affected": False,
        "metadata_aligned": bool(metadata_aligned),
        "frame_count": int(frame_count),
        "common_valid_frames": int(common_n) if common_n is not None else None,
        "common_coverage": coverage,
        "common_adjacent_valid_edges": int(edge_n) if edge_n is not None else None,
        "frozen_thresholds": {
            "common_coverage_min": MIN_COMMON_COVERAGE,
            "common_valid_frames_min": MIN_COMMON_VALID_FRAMES,
            "common_adjacent_valid_edges_min": MIN_COMMON_ADJACENT_EDGES,
        },
    }


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MouthPostprocessError(f"cannot read manifest: {type(exc).__name__}: {exc}") from exc
    if not isinstance(value, dict):
        raise MouthPostprocessError("manifest_root_not_object")
    return value


def _safe_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise MouthPostprocessError(f"invalid_sha256:{field}")
    return value


def _probe_video(path: Path, ffprobe: str) -> dict[str, Any]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,time_base,start_time,nb_frames:frame=best_effort_timestamp",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        data = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise MouthPostprocessError(f"ffprobe_failed:{type(exc).__name__}") from exc
    streams = data.get("streams", [])
    frames = data.get("frames", [])
    if len(streams) != 1 or not frames:
        raise MouthPostprocessError("video_stream_or_frames_missing")
    stream = streams[0]
    try:
        time_base = Fraction(stream["time_base"])
        fps_avg = Fraction(stream["avg_frame_rate"])
        fps_r = Fraction(stream["r_frame_rate"])
        start_time = Fraction(str(stream["start_time"]))
        width, height = int(stream["width"]), int(stream["height"])
        timestamps = [Fraction(int(frame["best_effort_timestamp"])) * time_base for frame in frames]
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise MouthPostprocessError("video_clock_metadata_invalid") from exc
    expected = [Fraction(i, FPS) for i in range(len(frames))]
    exact_clock = timestamps == expected
    return {
        "width": width,
        "height": height,
        "frame_count": len(frames),
        "fps_avg": _fraction_text(fps_avg),
        "fps_r": _fraction_text(fps_r),
        "time_base": _fraction_text(time_base),
        "start_time": str(start_time),
        "first_pts_time": str(timestamps[0]),
        "timestamps_exact_25fps_from_zero": exact_clock,
        "frame_timestamps_num_den": [[int(value.numerator), int(value.denominator)] for value in timestamps],
    }


def _load_rows(manifest_path: Path) -> tuple[dict[str, Any], list[str], dict[tuple[str, str], dict[str, Any]]]:
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != INPUT_SCHEMA:
        raise MouthPostprocessError(f"input_schema_mismatch:{manifest.get('schema')!r}")
    if manifest.get("fps") != {"numerator": 25, "denominator": 1}:
        raise MouthPostprocessError("input_fps_must_be_locked_25_1")
    speakers = manifest.get("planned_speakers")
    rows = manifest.get("conditions")
    if not isinstance(speakers, list) or not speakers or not isinstance(rows, list):
        raise MouthPostprocessError("planned_speakers_or_conditions_missing")
    if len(speakers) != len(set(speakers)) or any(sid not in SPEAKER_IDS for sid in speakers):
        raise MouthPostprocessError("planned_speaker_ids_invalid")
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise MouthPostprocessError("condition_row_not_object")
        sid, condition_id = row.get("speaker_id"), row.get("condition_id")
        if sid not in speakers or condition_id not in CONDITION_IDS:
            raise MouthPostprocessError("unexpected_speaker_or_condition")
        key = (sid, condition_id)
        if key in lookup:
            raise MouthPostprocessError(f"duplicate_condition:{sid}:{condition_id}")
        if _is_render_success(row):
            _safe_sha(row.get("video_sha256"), f"{sid}/{condition_id}/video")
            if not isinstance(row.get("video_path"), str) or not row["video_path"]:
                raise MouthPostprocessError(f"video_path_missing:{sid}:{condition_id}")
        lookup[key] = row
    return manifest, speakers, lookup


def _load_runtime(model_path: Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    try:
        import cv2  # type: ignore[import-not-found]
        import mediapipe as mp  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError as exc:
        raise MouthPostprocessError(f"runtime_dependency_missing:{exc.name}") from exc
    if not model_path.is_file():
        raise MouthPostprocessError(f"model_asset_missing:{model_path}")
    options_type = mp.tasks.vision.FaceLandmarkerOptions
    options = options_type(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    confidence_keys = (
        "min_face_detection_confidence",
        "min_face_presence_confidence",
        "min_tracking_confidence",
    )
    metadata = {
        "mediapipe_version": str(getattr(mp, "__version__", "unknown")),
        "opencv_version": str(getattr(cv2, "__version__", "unknown")),
        "numpy_version": str(getattr(np, "__version__", "unknown")),
        "api": "mediapipe.tasks.vision.FaceLandmarker",
        "running_mode": "IMAGE",
        "num_faces": 1,
        "confidence_parameters": {
            "source": "omitted as in existing mfa_linear_phoneme_video_audit mouth() source; actual runtime defaults recorded",
            "values": {key: getattr(options, key, None) for key in confidence_keys},
        },
        "model_asset_path": str(model_path.resolve()),
        "model_asset_sha256": _sha_file(model_path),
        "delegate": "BaseOptions default",
    }
    return cv2, mp, np, {"options": options, **metadata}


def _track_video(path: Path, expected_sha: str, cv2: Any, mp: Any, np: Any, detector: Any, probe: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise MouthPostprocessError("video_file_missing_or_symlink")
    video_sha = _sha_file(path)
    if video_sha != expected_sha:
        raise MouthPostprocessError("video_sha256_mismatch")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise MouthPostprocessError("video_open_failed")
    landmarks: list[Any] = []
    geometry_valid: list[bool] = []
    face_detected: list[bool] = []
    reason_codes: list[int] = []
    opening: list[float] = []
    invalid_reason_counts: dict[str, int] = {}
    dimensions: tuple[int, int] | None = None
    try:
        frame_index = 0
        while True:
            okay, frame = capture.read()
            if not okay:
                break
            height, width = frame.shape[:2]
            if dimensions is None:
                dimensions = (width, height)
            elif dimensions != (width, height):
                raise MouthPostprocessError("decoded_frame_dimensions_changed")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(image)
            face_rows = result.face_landmarks or []
            face_detected.append(len(face_rows) == 1)
            if len(face_rows) != 1:
                points: list[tuple[float, float, float]] = []
                geom = GeometryPoint(False, None, "no_unique_face")
            else:
                face = face_rows[0]
                points = [(float(point.x), float(point.y), float(point.z)) for point in face]
                geom = opening_geometry(points, width, height)
            if points:
                points_array = np.asarray(points, dtype=np.float32)
            else:
                count = int(probe.get("landmark_count", 478))
                points_array = np.full((count, 3), np.nan, dtype=np.float32)
            if len(points_array) <= max(LANDMARK_INDICES):
                geom = GeometryPoint(False, None, "landmark_count_below_required_index")
            landmarks.append(points_array)
            geometry_valid.append(bool(geom.valid))
            reason_codes.append(VALIDITY_REASON_CODES["valid"] if geom.valid else VALIDITY_REASON_CODES.get(geom.reason or "", 255))
            opening.append(float(geom.opening_ratio) if geom.valid else float("nan"))
            if geom.reason:
                invalid_reason_counts[geom.reason] = invalid_reason_counts.get(geom.reason, 0) + 1
            frame_index += 1
    finally:
        capture.release()
    if not landmarks:
        raise MouthPostprocessError("decoded_video_has_no_frames")
    if len(landmarks) != probe["frame_count"]:
        raise MouthPostprocessError(f"decoded_frame_count_mismatch:{len(landmarks)}:{probe['frame_count']}")
    if dimensions != (probe["width"], probe["height"]):
        raise MouthPostprocessError("decoded_dimensions_differ_from_ffprobe")
    counts = {len(value) for value in landmarks}
    if len(counts) != 1:
        raise MouthPostprocessError("landmark_count_changed_within_video")
    return {
        "video_path": str(path.resolve()),
        "video_sha256": video_sha,
        "probe": probe,
        "landmark_count": counts.pop(),
        "landmarks": landmarks,
        "valid": geometry_valid,
        "face_detected": face_detected,
        "validity_reason_code": reason_codes,
        "opening": opening,
        "invalid_reason_counts": invalid_reason_counts,
    }


def _percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _build_condition_summary(
    sid: str,
    condition_id: str,
    tracked: dict[str, Any],
    common_mask: Sequence[bool] | None,
    common_status: str,
) -> dict[str, Any]:
    n = len(tracked["valid"])
    own_valid = [bool(value) for value in tracked["valid"]]
    own_open = [float(value) if own_valid[i] else None for i, value in enumerate(tracked["opening"])]
    if common_mask is not None:
        open_mask = [bool(common_mask[i]) and own_valid[i] for i in range(n)]
    else:
        open_mask = [False] * n
    common_open = [own_open[i] if open_mask[i] else None for i in range(n)]
    speed, speed_valid = adjacent_speed(common_open, open_mask, FPS)
    supported_aperture = [float(v) for v in common_open if v is not None]
    supported_speed = [float(v) for v, valid in zip(speed, speed_valid) if valid and v is not None]
    frame_count = len(own_valid)
    return {
        "speaker_id": sid,
        "condition_id": condition_id,
        "status": "PASS_RAW_LANDMARKS",
        "common_support_status": common_status,
        "video_path": tracked["video_path"],
        "video_sha256": tracked["video_sha256"],
        "frame_count": frame_count,
        "fps": "25/1",
        "width": tracked["probe"]["width"],
        "height": tracked["probe"]["height"],
        "start_time": tracked["probe"]["start_time"],
        "pts_exact_i_over_25": tracked["probe"]["timestamps_exact_25fps_from_zero"],
        "tracking_valid_frames": sum(own_valid),
        "tracking_coverage": sum(own_valid) / frame_count if frame_count else None,
        "common_support_known": common_mask is not None,
        "common_valid_frames": sum(open_mask) if common_mask is not None else None,
        "common_coverage": sum(open_mask) / frame_count if common_mask is not None and frame_count else None,
        "common_adjacent_valid_edges": sum(speed_valid) if common_mask is not None else None,
        "opening_mean_common": _mean(supported_aperture),
        "opening_median_common": _percentile(supported_aperture, 0.5),
        "opening_p90_common": _percentile(supported_aperture, 0.9),
        "speed_mean_common": _mean(supported_speed),
        "speed_rms_common": math.sqrt(_mean([v * v for v in supported_speed])) if supported_speed else None,
        "speed_p95_common": _percentile(supported_speed, 0.95),
        "invalid_reason_counts": tracked["invalid_reason_counts"],
    }


def _save_condition_npz(path: Path, tracked: dict[str, Any], common_mask: Sequence[bool] | None, np: Any) -> dict[str, Any]:
    n = len(tracked["valid"])
    own_valid = [bool(value) for value in tracked["valid"]]
    common = [bool(common_mask[i]) and own_valid[i] for i in range(n)] if common_mask is not None else [False] * n
    opening = [float(value) if own_valid[i] else float("nan") for i, value in enumerate(tracked["opening"])]
    opening_common = [opening[i] if common[i] else float("nan") for i in range(n)]
    speed, speed_valid = adjacent_speed(
        [opening_common[i] if common[i] else None for i in range(n)], common, FPS
    )
    speed_values = [float(value) if valid and value is not None else float("nan") for value, valid in zip(speed, speed_valid)]
    array_path = path
    np.savez_compressed(
        array_path,
        landmarks_xyz_float32=np.stack(tracked["landmarks"]).astype(np.float32, copy=False),
        frame_index_int32=np.arange(n, dtype=np.int32),
        timestamp_num_int64=np.arange(n, dtype=np.int64),
        timestamp_den_int64=np.asarray(25, dtype=np.int64),
        tracking_valid_bool=np.asarray(own_valid, dtype=np.bool_),
        face_detected_bool=np.asarray(tracked["face_detected"], dtype=np.bool_),
        validity_reason_code_uint8=np.asarray(tracked["validity_reason_code"], dtype=np.uint8),
        common_valid_bool=np.asarray(common, dtype=np.bool_),
        common_support_known_bool=np.full(n, common_mask is not None, dtype=np.bool_),
        opening_ratio_float64=np.asarray(opening, dtype=np.float64),
        opening_ratio_common_float64=np.asarray(opening_common, dtype=np.float64),
        speed_ratio_per_s_float64=np.asarray(speed_values, dtype=np.float64),
        speed_valid_bool=np.asarray(speed_valid, dtype=np.bool_),
        interval_start_index_int32=np.arange(max(0, n - 1), dtype=np.int32),
        interval_midpoint_timestamp_num_int64=2 * np.arange(max(0, n - 1), dtype=np.int64) + 1,
        interval_timestamp_den_int64=np.asarray(50, dtype=np.int64),
    )
    return {"path": str(array_path), "bytes": array_path.stat().st_size, "sha256": _sha_file(array_path)}


def run(manifest_path: Path, output_dir: Path, model_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    model_path = model_path.resolve(strict=True)
    if output_dir.exists():
        raise MouthPostprocessError(f"refusing_to_overwrite_output:{output_dir}")
    manifest, speakers, lookup = _load_rows(manifest_path)
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise MouthPostprocessError("ffprobe_missing_from_existing_runtime")
    cv2, mp, np, runtime = _load_runtime(model_path)
    options = runtime.pop("options")

    input_sha = _sha_file(manifest_path)
    model_sha = _sha_file(model_path)
    source_path = Path(__file__).resolve()
    source_sha = _sha_file(source_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    all_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    speaker_results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for sid in speakers:
            row_inputs = {cid: lookup.get((sid, cid)) for cid in CONDITION_IDS}
            tracked: dict[str, dict[str, Any]] = {}
            probes: dict[str, dict[str, Any]] = {}
            cell_status: dict[str, str] = {}
            for cid in CONDITION_IDS:
                row = row_inputs[cid]
                if row is None:
                    cell_status[cid] = "MISSING_RENDER_ROW"
                    failures.append({"speaker_id": sid, "condition_id": cid, "reason": "missing_render_row"})
                    continue
                if not _is_render_success(row):
                    cell_status[cid] = f"RENDER_{row.get('status', 'STATUS_MISSING')}"
                    failures.append({"speaker_id": sid, "condition_id": cid, "reason": cell_status[cid]})
                    continue
                path = Path(row["video_path"])
                try:
                    if not path.is_file() or path.is_symlink():
                        raise MouthPostprocessError("video_file_missing_or_symlink")
                    observed_sha = _sha_file(path)
                    if observed_sha != row["video_sha256"]:
                        raise MouthPostprocessError("video_sha256_mismatch")
                    probe = _probe_video(path, ffprobe)
                    probes[cid] = probe
                    cell = _track_video(path, row["video_sha256"], cv2, mp, np, detector, probe)
                    tracked[cid] = cell
                    cell_status[cid] = "PASS_RAW_LANDMARKS"
                except (OSError, MouthPostprocessError) as exc:
                    cell_status[cid] = f"FAILED:{str(exc)}"
                    failures.append({"speaker_id": sid, "condition_id": cid, "reason": str(exc)})

            all_eight_present = all(cid in tracked for cid in CONDITION_IDS)
            metadata_aligned = False
            common_mask: list[bool] | None = None
            common_status = "UNMEASURABLE_EIGHT_CONDITION_SET_INCOMPLETE"
            qual: dict[str, Any] | None = None
            if all_eight_present:
                first = probes[CONDITION_IDS[0]]
                metadata_equal = all(
                    probe["width"] == first["width"]
                    and probe["height"] == first["height"]
                    and probe["frame_count"] == first["frame_count"]
                    and probe["fps_avg"] == "25/1"
                    and probe["fps_r"] == "25/1"
                    and probe["start_time"] == "0"
                    and probe["timestamps_exact_25fps_from_zero"]
                    for probe in probes.values()
                ) and all(
                    probe["width"] == first["width"]
                    and probe["height"] == first["height"]
                    and probe["frame_count"] == first["frame_count"]
                    and probe["start_time"] == first["start_time"]
                    for probe in probes.values()
                )
                metadata_aligned = bool(metadata_equal)
                if metadata_aligned:
                    n = first["frame_count"]
                    masks = [tracked[cid]["valid"] for cid in CONDITION_IDS]
                    common_mask = common_valid_mask(masks)
                    common_status = "COMMON_EIGHT_CONDITION_FRAME_SUPPORT"
                    qual = qualification(frame_count=n, common_mask=common_mask, metadata_aligned=True)
                else:
                    common_status = "UNMEASURABLE_FRAME_CLOCK_OR_SIZE_MISMATCH"
                    n = first["frame_count"]
                    qual = qualification(frame_count=n, common_mask=[False] * n, metadata_aligned=False)
            else:
                qual = qualification(frame_count=0, common_mask=[], metadata_aligned=False)

            for cid in CONDITION_IDS:
                if cid not in tracked:
                    source_row = row_inputs[cid]
                    all_rows.append(
                        {
                            "speaker_id": sid,
                            "condition_id": cid,
                            "status": cell_status.get(cid, "MISSING_RENDER_ROW"),
                            "input_video_path": source_row.get("video_path") if source_row else None,
                            "input_video_sha256": source_row.get("video_sha256") if source_row else None,
                            "input_render_status": source_row.get("status") if source_row else None,
                            "common_support_status": common_status,
                            "common_support_known": False,
                            "mouth_eligible_for_description": False,
                            "score_rows_affected": False,
                            "metrics": None,
                            "arrays": None,
                        }
                    )
                    continue
                cell = tracked[cid]
                summary = _build_condition_summary(sid, cid, cell, common_mask, common_status)
                relative_array = Path("mouth_arrays") / sid / f"{cid}.npz"
                output_array = output_dir / relative_array
                output_array.parent.mkdir(parents=True, exist_ok=True)
                pin = _save_condition_npz(output_array, cell, common_mask, np)
                summary["arrays"] = {
                    "relative_path": relative_array.as_posix(),
                    "bytes": pin["bytes"],
                    "sha256": pin["sha256"],
                }
                summary["mouth_eligible_for_description"] = bool(qual and qual["eligible_for_mouth_description"])
                summary["score_rows_affected"] = False
                all_rows.append(summary)

                frame_count = len(cell["valid"])
                own_valid = [bool(value) for value in cell["valid"]]
                opening_velocity_support = (
                    [bool(common_mask[i]) and own_valid[i] for i in range(frame_count)]
                    if common_mask is not None
                    else [False] * frame_count
                )
                common_opening = [
                    float(cell["opening"][i]) if opening_velocity_support[i] else None
                    for i in range(frame_count)
                ]
                speed_values, speed_valid = adjacent_speed(common_opening, opening_velocity_support, FPS)
                for i in range(frame_count):
                    curve_rows.append(
                        {
                            "speaker_id": sid,
                            "condition_id": cid,
                            "metric": "opening_ratio",
                            "sample_index": i,
                            "time_num": i,
                            "time_den": 25,
                            "value": cell["opening"][i] if own_valid[i] else None,
                            "valid": own_valid[i],
                            "common_support_valid": opening_velocity_support[i],
                            "common_support_known": common_mask is not None,
                        }
                    )
                for i, (value, valid) in enumerate(zip(speed_values, speed_valid)):
                    curve_rows.append(
                        {
                            "speaker_id": sid,
                            "condition_id": cid,
                            "metric": "absolute_speed_ratio_per_s",
                            "sample_index": i,
                            "time_num": 2 * i + 1,
                            "time_den": 50,
                            "value": value if valid else None,
                            "valid": bool(valid),
                            "common_support_valid": bool(valid),
                            "common_support_known": common_mask is not None,
                        }
                    )

            speaker_results.append(
                {
                    "speaker_id": sid,
                    "planned_condition_ids": list(CONDITION_IDS),
                    "condition_status": cell_status,
                    "metadata_aligned": metadata_aligned,
                    "common_support_status": common_status,
                    "qualification": qual,
                    "score_rows_affected": False,
                }
            )

    csv_path = output_dir / "mouth_condition_rows.csv"
    csv_fields = [
        "speaker_id",
        "condition_id",
        "status",
        "common_support_status",
        "frame_count",
        "fps",
        "width",
        "height",
        "start_time",
        "pts_exact_i_over_25",
        "tracking_valid_frames",
        "tracking_coverage",
        "common_valid_frames",
        "common_coverage",
        "common_support_known",
        "common_adjacent_valid_edges",
        "opening_mean_common",
        "opening_median_common",
        "opening_p90_common",
        "speed_mean_common",
        "speed_rms_common",
        "speed_p95_common",
        "mouth_eligible_for_description",
        "score_rows_affected",
        "arrays_path",
        "arrays_sha256",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            arrays = row.get("arrays") or {}
            writer.writerow({**row, "arrays_path": arrays.get("relative_path"), "arrays_sha256": arrays.get("sha256")})

    curve_path = output_dir / "mouth_curve_rows.csv"
    curve_fields = [
        "speaker_id",
        "condition_id",
        "metric",
        "sample_index",
        "time_num",
        "time_den",
        "value",
        "valid",
        "common_support_valid",
        "common_support_known",
    ]
    with curve_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=curve_fields)
        writer.writeheader()
        writer.writerows(curve_rows)

    output = {
        "schema": OUTPUT_SCHEMA,
        "status": "COMPLETE_WITH_PRESERVED_MISSING_CELLS" if failures or any(row.get("status") != "PASS_RAW_LANDMARKS" for row in all_rows) else "COMPLETE",
        "scope": "Descriptive mouth aperture and adjacent-frame speed; not mouth accuracy, perception, or synchronization truth.",
        "input_manifest": {"path": str(manifest_path), "sha256": input_sha},
        "protocol_sha256": manifest.get("protocol_sha256"),
        "model": {"path": str(model_path), "sha256": model_sha},
        "source": {"path": str(source_path), "sha256": source_sha, "bytes": source_path.stat().st_size},
        "runtime": runtime,
        "ffprobe": ffprobe,
        "fps_clock": "frame i is time i/25 seconds; adjacent absolute speed at midpoint (i+0.5)/25.",
        "opening_definition": "Euclidean pixel distance between landmarks 13 and 14 divided by Euclidean pixel distance between landmarks 33 and 263.",
        "validity_definition": "All four required landmark x/y coordinates finite and inter-eye pixel distance > 1e-6; all eight condition masks are intersected per speaker.",
        "qualification_rule": {
            "common_coverage_min": MIN_COMMON_COVERAGE,
            "common_valid_frames_min": MIN_COMMON_VALID_FRAMES,
            "common_adjacent_valid_edges_min": MIN_COMMON_ADJACENT_EDGES,
            "purpose": "measurement eligibility only; does not alter SyncNet inclusion or score analysis",
        },
        "planned_speakers": speakers,
        "planned_conditions": list(CONDITION_IDS),
        "speaker_results": speaker_results,
        "condition_rows": all_rows,
        "failures": failures,
    }
    _write_json(output_dir / "mouth_overlay.json", output)
    file_inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "output_manifest.json":
            relative = path.relative_to(output_dir).as_posix()
            file_inventory[relative] = {"bytes": path.stat().st_size, "sha256": _sha_file(path)}
    output["files"] = file_inventory
    _write_json(output_dir / "output_manifest.json", output)
    return {
        "status": output["status"],
        "output": str(output_dir.resolve()),
        "speaker_count": len(speakers),
        "condition_rows": len(all_rows),
        "eligible_speakers": sum(bool(row["qualification"]["eligible_for_mouth_description"]) for row in speaker_results),
        "failed_cells": len(failures),
        "output_manifest_sha256": _sha_file(output_dir / "output_manifest.json"),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="sealed render-receipt-derived input manifest")
    parser.add_argument("--model", required=True, type=Path, help="existing face_landmarker.task asset")
    parser.add_argument("--output", required=True, type=Path, help="new output directory; existing paths are refused")
    args = parser.parse_args(argv)
    try:
        result = run(args.manifest, args.output, args.model)
    except MouthPostprocessError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
