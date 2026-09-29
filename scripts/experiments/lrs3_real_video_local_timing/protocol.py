from __future__ import annotations

import json
import math
import subprocess
import wave
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import config
from .common import (
    DiagnosticError,
    assert_finite,
    assert_not_sealed,
    bytes_sha256,
    canonical_json_sha256,
    file_sha256,
    read_json,
    require_file,
    sample_ids_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def ffprobe_json(path: Path, *, count_frames: bool = False, show_frames: bool = False) -> dict[str, Any]:
    command = [str(config.FFPROBE), "-v", "error"]
    if show_frames:
        command.extend(["-select_streams", "v:0"])
    if count_frames:
        command.extend(["-count_frames"])
    command.extend(["-show_streams", "-show_format"])
    if show_frames:
        command.append("-show_frames")
    command.extend(["-of", "json", str(path)])
    try:
        result = subprocess.run(command, capture_output=True, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise DiagnosticError(f"ffprobe failed: {path}") from exc
    try:
        value = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"ffprobe returned invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"ffprobe returned a non-object: {path}")
    return value


def _number(value: Any, label: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise DiagnosticError(f"{label} is invalid: {path}") from None
    if not math.isfinite(result):
        raise DiagnosticError(f"{label} is non-finite: {path}")
    return result


def _fraction(value: Any, label: str, path: Path) -> Fraction:
    try:
        result = Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError):
        raise DiagnosticError(f"{label} is invalid: {path}") from None
    if result.denominator == 0:
        raise DiagnosticError(f"{label} is invalid: {path}")
    return result


def _timestamp(value: Any, label: str, path: Path) -> int:
    if isinstance(value, bool):
        raise DiagnosticError(f"{label} is invalid: {path}")
    try:
        result = int(str(value))
    except (TypeError, ValueError):
        raise DiagnosticError(f"{label} is invalid: {path}") from None
    return result


def video_timeline(path: Path) -> dict[str, Any]:
    assert_not_sealed(path)
    require_file(path, "face video")
    payload = ffprobe_json(path, count_frames=True, show_frames=True)
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise DiagnosticError(f"video stream list is missing: {path}")
    videos = [stream for stream in streams if isinstance(stream, Mapping) and stream.get("codec_type") == "video"]
    if len(videos) != 1:
        raise DiagnosticError(f"face video must contain exactly one video stream: {path}")
    stream = videos[0]
    rate_text = str(stream.get("r_frame_rate", ""))
    try:
        frame_rate = float(Fraction(rate_text))
    except (ValueError, ZeroDivisionError):
        raise DiagnosticError(f"video frame rate is invalid: {path}") from None
    if abs(frame_rate - config.FPS) > 1e-6:
        raise DiagnosticError(f"video is not {config.FPS} fps: {path}")
    frame_value = stream.get("nb_read_frames", stream.get("nb_frames"))
    try:
        frame_count = int(frame_value)
    except (TypeError, ValueError):
        raise DiagnosticError(f"video frame count is unavailable: {path}") from None
    if frame_count < config.WINDOW_FRAMES + 2:
        raise DiagnosticError(f"video is too short: {path}")
    start_time = _number(stream.get("start_time"), "video start_time", path)
    duration = _number(stream.get("duration"), "video duration", path)
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise DiagnosticError(f"video geometry is invalid: {path}")
    if abs(start_time) > 0.001:
        raise DiagnosticError(f"video does not start at zero: {path}")
    time_base = _fraction(stream.get("time_base"), "video time_base", path)
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list) or len(raw_frames) != frame_count:
        raise DiagnosticError(f"video frame PTS evidence is incomplete: {path}")
    pts: list[int] = []
    pts_time: list[float] = []
    frame_evidence: list[dict[str, Any]] = []
    first_pts: Fraction | None = None
    maximum_deviation = Fraction(0, 1)
    for index, frame in enumerate(raw_frames):
        if not isinstance(frame, Mapping):
            raise DiagnosticError(f"video frame PTS evidence is malformed: {path}")
        field = "best_effort_timestamp" if frame.get("best_effort_timestamp") is not None else "pts"
        raw_pts = frame.get(field)
        if raw_pts is None:
            raise DiagnosticError(f"video frame PTS is missing at index {index}: {path}")
        value = _timestamp(raw_pts, f"video frame PTS {index}", path)
        actual = value * time_base
        if first_pts is None:
            first_pts = actual
        else:
            if value <= pts[-1]:
                raise DiagnosticError(f"video frame PTS is not strictly increasing at index {index}: {path}")
            deviation = abs((actual - first_pts) - Fraction(index, config.FPS))
            maximum_deviation = max(maximum_deviation, deviation)
            if deviation > Fraction(1, 1000):
                raise DiagnosticError(f"video frame PTS is not a 25 fps timeline at index {index}: {path}")
        pts.append(value)
        pts_time.append(float(actual))
        frame_evidence.append(
            {
                "index": index,
                "field": field,
                "pts": value,
                "pts_time": float(actual),
            }
        )
    if first_pts is None:
        raise DiagnosticError(f"video has no frame PTS evidence: {path}")
    return {
        "frame_count": frame_count,
        "frame_rate": frame_rate,
        "width": width,
        "height": height,
        "start_time": start_time,
        "duration": duration,
        "time_base": str(stream.get("time_base")),
        "codec_name": stream.get("codec_name"),
        "first_pts": pts[0],
        "first_pts_time": pts_time[0],
        "pts": pts,
        "pts_time": pts_time,
        "pts_strictly_increasing": True,
        "pts_max_deviation_ms": float(maximum_deviation * 1000),
        "pts_evidence": {
            "time_base": str(stream.get("time_base")),
            "frame_count": frame_count,
            "frames": frame_evidence,
        },
    }


def pcm_wav_metadata(path: Path) -> dict[str, Any]:
    assert_not_sealed(path)
    require_file(path, "natural audio")
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            sample_count = handle.getnframes()
            pcm = handle.readframes(sample_count)
    except (OSError, wave.Error) as exc:
        raise DiagnosticError(f"natural audio is not a readable WAV: {path}") from exc
    if channels != config.PCM_CHANNELS or sample_width != config.PCM_SAMPLE_WIDTH or sample_rate != config.SAMPLE_RATE:
        raise DiagnosticError(f"natural audio format is not 16 kHz mono PCM16: {path}")
    if len(pcm) != sample_count * config.PCM_SAMPLE_WIDTH * config.PCM_CHANNELS:
        raise DiagnosticError(f"natural audio PCM length is inconsistent: {path}")
    return {
        "sample_count": int(sample_count),
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "sample_width": int(sample_width),
        "container_sha256": file_sha256(path),
        "pcm_sha256": bytes_sha256(pcm),
    }


def audio_video_duration_within_one_frame(
    audio_sample_count: int,
    video_frame_count: int,
) -> bool:
    """Check the duration contract without introducing float-boundary errors."""
    return (
        abs(audio_video_tail_delta(audio_sample_count, video_frame_count))
        <= config.SAMPLES_PER_FRAME
    )


def audio_video_tail_delta(audio_sample_count: int, video_frame_count: int) -> int:
    """Return the signed PCM tail in samples using only integer arithmetic."""
    return int(audio_sample_count) - int(video_frame_count) * config.SAMPLES_PER_FRAME


def audio_video_duration_within_bounded_tail(
    audio_sample_count: int,
    video_frame_count: int,
) -> bool:
    delta = audio_video_tail_delta(audio_sample_count, video_frame_count)
    return config.AUDIO_TAIL_MIN_SAMPLES <= delta <= config.AUDIO_TAIL_MAX_SAMPLES


def audio_video_tail_contract(audio_sample_count: int, video_frame_count: int) -> dict[str, Any]:
    delta = audio_video_tail_delta(audio_sample_count, video_frame_count)
    if delta < config.AUDIO_TAIL_MIN_SAMPLES:
        tail_class = "invalid_short_tail"
    elif delta > config.AUDIO_TAIL_MAX_SAMPLES:
        tail_class = "invalid_long_tail"
    elif delta > config.SAMPLES_PER_FRAME:
        tail_class = "extended_audio_tail"
    else:
        tail_class = "within_original_bound"
    return {
        "revision": config.PROTOCOL_REVISION,
        "delta_samples": delta,
        "delta_ms": float(delta * 1000 / config.SAMPLE_RATE),
        "accepted": bool(config.AUDIO_TAIL_MIN_SAMPLES <= delta <= config.AUDIO_TAIL_MAX_SAMPLES),
        "tail_class": tail_class,
        "reason": "unattributed" if tail_class == "extended_audio_tail" else None,
        "bounds_samples": [config.AUDIO_TAIL_MIN_SAMPLES, config.AUDIO_TAIL_MAX_SAMPLES],
    }


def _asset(value: Any, name: str, sample_id: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise DiagnosticError(f"history asset is malformed: {sample_id}/{name}")
    path = Path(str(value.get("path", ""))).resolve()
    expected = str(value.get("sha256", ""))
    assert_not_sealed(path)
    if not path.is_file() or not expected or file_sha256(path) != expected:
        raise DiagnosticError(f"history asset hash changed: {sample_id}/{name}")
    return {"path": str(path), "sha256": expected}


def _resolve_history_path(value: Any) -> Path:
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = config.REPO / candidate
    return candidate.resolve()


def source_pairing_evidence(
    source: Mapping[str, Any],
    video: Mapping[str, str],
    audio: Mapping[str, str],
    timeline: Mapping[str, Any],
    audio_meta: Mapping[str, Any],
) -> dict[str, Any]:
    audio_path = Path(str(audio["path"]))
    manifest_path = audio_path.parent.parent / "manifest.json"
    require_file(manifest_path, "historical source manifest")
    if file_sha256(manifest_path) != config.HISTORY_SOURCE_MANIFEST_SHA256:
        raise DiagnosticError("historical source manifest hash changed")
    manifest = read_json(manifest_path)
    records = manifest.get("records")
    if not isinstance(records, list):
        raise DiagnosticError("historical source manifest records are missing")
    sample_id = str(source.get("sample_id", ""))
    matches = [(index, row) for index, row in enumerate(records) if isinstance(row, Mapping) and str(row.get("sample_id")) == sample_id]
    if len(matches) != 1:
        raise DiagnosticError(f"historical source manifest pairing is ambiguous: {sample_id}")
    record_index, manifest_record = matches[0]
    expected_video = Path(str(video["path"])).resolve()
    expected_audio = Path(str(audio["path"])).resolve()
    checks = {
        "sample_id": str(manifest_record.get("sample_id")) == sample_id,
        "source_group": str(manifest_record.get("source_group")) == str(source.get("source_group")),
        "face_video_path": _resolve_history_path(manifest_record.get("video_local_path")) == expected_video,
        "face_video_hash": str(manifest_record.get("video_sha256")) == str(video["sha256"]),
        "natural_audio_path": _resolve_history_path(manifest_record.get("natural_audio_path")) == expected_audio,
        "natural_audio_hash": str(manifest_record.get("natural_audio_sha256")) == str(audio["sha256"]),
        "natural_audio_samples": int(manifest_record.get("natural_audio_samples", -1)) == int(audio_meta["sample_count"]),
        "natural_audio_rate": int(manifest_record.get("natural_audio_sample_rate_hz", -1)) == config.SAMPLE_RATE,
        "natural_audio_channels": int(manifest_record.get("natural_audio_channels", -1)) == config.PCM_CHANNELS,
    }
    if source.get("transcript") is not None:
        checks["transcript"] = str(manifest_record.get("transcript")) == str(source.get("transcript"))
    if not all(checks.values()):
        failed = ",".join(name for name, passed in checks.items() if not passed)
        raise DiagnosticError(f"historical source pairing differs: {sample_id} ({failed})")
    video_start = float(timeline["first_pts_time"])
    audio_start = 0.0
    start_difference_ms = abs(video_start - audio_start) * 1000.0
    return {
        "source_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": file_sha256(manifest_path),
            "record_index": int(record_index),
            "record_sha256": canonical_json_sha256(manifest_record),
        },
        "checks": checks,
        "audio_start_s": audio_start,
        "video_start_pts_s": video_start,
        "start_difference_ms": float(start_difference_ms),
        "audio_start_evidence": "natural WAV sample 0 paired by the historical source manifest",
    }


def load_history_cohort() -> dict[str, Any]:
    path = config.HISTORY_COHORT
    require_file(path, "historical cohort")
    if file_sha256(path) != config.HISTORY_COHORT_SHA256:
        raise DiagnosticError("historical cohort hash changed")
    cohort = verify_self_hashed_json(path)
    records = cohort.get("records")
    if (
        cohort.get("status") != "complete"
        or cohort.get("record_count") != config.EXPECTED_RECORD_COUNT
        or cohort.get("source_group_count") != config.EXPECTED_SOURCE_GROUP_COUNT
        or not isinstance(records, list)
        or len(records) != config.EXPECTED_RECORD_COUNT
    ):
        raise DiagnosticError("historical cohort is incomplete")
    ids = [str(row.get("sample_id", "")) for row in records if isinstance(row, Mapping)]
    groups = [str(row.get("source_group", "")) for row in records if isinstance(row, Mapping)]
    if (
        len(ids) != config.EXPECTED_RECORD_COUNT
        or len(set(ids)) != len(ids)
        or len(set(groups)) != len(groups)
        or sample_ids_sha256(ids) != config.EXPECTED_SAMPLE_ID_SHA256
        or cohort.get("sample_ids_sha256") != config.EXPECTED_SAMPLE_ID_SHA256
    ):
        raise DiagnosticError("historical cohort order or identity changed")
    for row in records:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or not str(row.get("source_group", "")):
            raise DiagnosticError("historical cohort contains an incomplete record")
        _asset(row.get("natural_audio"), "natural_audio", sample_id)
        _asset(row.get("face_video"), "face_video", sample_id)
    return cohort


def runtime_bindings() -> dict[str, Any]:
    paths = {
        "syncnet_python": config.SYNCNET_PYTHON,
        "syncnet_model": config.SYNCNET_MODEL,
        "syncnet_instance": config.SYNCNET_ROOT / "SyncNetInstance.py",
        "syncnet_model_code": config.SYNCNET_ROOT / "SyncNetModel.py",
        "run_syncnet": config.SYNCNET_ROOT / "run_syncnet.py",
        "s3fd_weights": config.SYNCNET_ROOT / "detectors/s3fd/weights/sfd_face.pth",
        "ffmpeg": config.FFMPEG,
        "ffprobe": config.FFPROBE,
    }
    result: dict[str, Any] = {}
    for name, path in paths.items():
        require_file(path, f"runtime binding {name}")
        result[name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    if result["syncnet_model"]["sha256"] != config.SYNCNET_MODEL_SHA256:
        raise DiagnosticError("SyncNet model hash differs from the registered weight")
    return result


def package_source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {path.name: file_sha256(path) for path in sorted(package.glob("*.py"))}


def registered_spec_bindings() -> dict[str, dict[str, str]]:
    bindings = {
        "parent_spec": (config.PARENT_SPEC, config.PARENT_SPEC_SHA256),
        "amendment_spec": (config.AMENDMENT_SPEC, config.AMENDMENT_SPEC_SHA256),
    }
    result: dict[str, dict[str, str]] = {}
    for name, (path, expected) in bindings.items():
        require_file(path, f"{name} source")
        actual = file_sha256(path)
        if actual != expected:
            raise DiagnosticError(f"{name} hash changed")
        result[name] = {"path": str(path.resolve()), "sha256": expected}
    return result


def parent_blocked_binding() -> dict[str, str]:
    require_file(config.PARENT_BLOCKED_FINAL, "parent blocked final")
    actual = file_sha256(config.PARENT_BLOCKED_FINAL)
    if actual != config.PARENT_BLOCKED_FINAL_SHA256:
        raise DiagnosticError("parent blocked final hash changed")
    return {
        "run_id": "lrs3_real_video_local_timing_20260905_v2",
        "path": str(config.PARENT_BLOCKED_FINAL.resolve()),
        "sha256": config.PARENT_BLOCKED_FINAL_SHA256,
        "result_sha256": config.PARENT_BLOCKED_RESULT_SHA256,
    }


def _audit_asset(value: Any, name: str, sample_id: str) -> tuple[dict[str, Any], Path | None, list[str]]:
    details: dict[str, Any] = {"path": None, "expected_sha256": None, "actual_sha256": None, "hash_match": False}
    errors: list[str] = []
    if not isinstance(value, Mapping):
        errors.append(f"history asset is malformed: {sample_id}/{name}")
        return details, None, errors
    raw_path = str(value.get("path", ""))
    expected = str(value.get("sha256", ""))
    path = Path(raw_path).resolve()
    details.update({"path": str(path), "expected_sha256": expected or None})
    try:
        assert_not_sealed(path)
    except DiagnosticError as exc:
        errors.append(str(exc))
        return details, path, errors
    if not path.is_file():
        errors.append(f"history asset is missing: {sample_id}/{name}")
        return details, path, errors
    actual = file_sha256(path)
    details["actual_sha256"] = actual
    details["hash_match"] = bool(expected and actual == expected)
    if not details["hash_match"]:
        errors.append(f"history asset hash changed: {sample_id}/{name}")
        return details, path, errors
    return details, path, errors


def _audit_record(source: Mapping[str, Any], index: int) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sample_id = str(source.get("sample_id", "")) or None
    source_group = str(source.get("source_group", "")) or None
    label = sample_id or f"record_{index}"
    face_details, face_path, face_errors = _audit_asset(source.get("face_video"), "face_video", label)
    audio_details, audio_path, audio_errors = _audit_asset(source.get("natural_audio"), "natural_audio", label)
    errors = [*face_errors, *audio_errors]
    checks: dict[str, str] = {
        "face_video_hash": "PASS" if not face_errors else "FAIL",
        "natural_audio_hash": "PASS" if not audio_errors else "FAIL",
        "video_timeline": "NOT_RUN",
        "video_pts": "NOT_RUN",
        "natural_audio_format": "NOT_RUN",
        "tail_contract": "NOT_RUN",
        "source_pairing": "NOT_RUN",
        "audio_video_start": "NOT_RUN",
        "common_support": "NOT_RUN",
    }
    # The expression above intentionally keeps the two asset checks independent even
    # when both assets fail on the same record.
    timeline: dict[str, Any] | None = None
    if face_path is not None and face_details["hash_match"]:
        try:
            timeline = video_timeline(face_path)
            checks["video_timeline"] = "PASS"
            checks["video_pts"] = "PASS"
        except (DiagnosticError, OSError, ValueError) as exc:
            checks["video_timeline"] = "FAIL"
            checks["video_pts"] = "FAIL"
            errors.append(str(exc))
    audio_meta: dict[str, Any] | None = None
    if audio_path is not None and audio_details["hash_match"]:
        try:
            audio_meta = pcm_wav_metadata(audio_path)
            checks["natural_audio_format"] = "PASS"
        except (DiagnosticError, OSError, ValueError) as exc:
            checks["natural_audio_format"] = "FAIL"
            errors.append(str(exc))
    duration: dict[str, Any] | None = None
    common_support: list[int] | None = None
    if timeline is not None and audio_meta is not None:
        duration = audio_video_tail_contract(int(audio_meta["sample_count"]), int(timeline["frame_count"]))
        checks["tail_contract"] = "PASS" if duration["accepted"] else "FAIL"
        if not duration["accepted"]:
            errors.append(
                f"audio/video duration is outside {config.PROTOCOL_REVISION}: {label} d={duration['delta_samples']}"
            )
        common_support = [0, min(int(audio_meta["sample_count"]), int(timeline["frame_count"]) * config.SAMPLES_PER_FRAME)]
        checks["common_support"] = "PASS" if common_support[1] > 0 else "FAIL"
        if common_support[1] <= 0:
            errors.append(f"common audio/video support is empty: {label}")
        try:
            pairing = source_pairing_evidence(
                source,
                {"path": str(face_path), "sha256": str(source.get("face_video", {}).get("sha256", ""))},
                {"path": str(audio_path), "sha256": str(source.get("natural_audio", {}).get("sha256", ""))},
                timeline,
                audio_meta,
            )
            checks["source_pairing"] = "PASS"
            checks["audio_video_start"] = "PASS" if pairing["start_difference_ms"] <= 1.0 else "FAIL"
            if checks["audio_video_start"] == "FAIL":
                errors.append(f"audio/video start differs by more than 1 ms: {label}")
        except (DiagnosticError, OSError, ValueError, TypeError) as exc:
            pairing = None
            checks["source_pairing"] = "FAIL"
            checks["audio_video_start"] = "FAIL"
            errors.append(str(exc))
    else:
        pairing = None
    passed = not errors and all(value == "PASS" for value in checks.values())
    audit_row: dict[str, Any] = {
        "index": index,
        "sample_id": sample_id,
        "source_group": source_group,
        "face_video": face_details,
        "natural_audio": audio_details,
        "frame_count": int(timeline["frame_count"]) if timeline is not None else None,
        "sample_count": int(audio_meta["sample_count"]) if audio_meta is not None else None,
        "duration_delta_samples": duration["delta_samples"] if duration is not None else None,
        "tail_difference_ms": duration["delta_ms"] if duration is not None else None,
        "tail_class": duration["tail_class"] if duration is not None else None,
        "source_video_timeline": timeline,
        "natural_audio_format": audio_meta,
        "source_pairing": pairing,
        "common_support_samples": common_support,
        "checks": checks,
        "passed": passed,
        "status": "PASS" if passed else "BLOCKED",
        "errors": errors,
    }
    if not passed:
        return audit_row, None
    assert sample_id is not None and source_group is not None and timeline is not None and audio_meta is not None and pairing is not None and duration is not None
    return audit_row, {
        "sample_id": sample_id,
        "source_group": source_group,
        "paired_key": source.get("paired_key"),
        "speaker_id": source.get("speaker_id"),
        "split": source.get("split"),
        "transcript": source.get("transcript"),
        "face_video": {"path": str(face_path), "sha256": str(source["face_video"]["sha256"])},
        "natural_audio": {"path": str(audio_path), "sha256": str(source["natural_audio"]["sha256"])},
        "source_video_timeline": timeline,
        "natural_audio_format": audio_meta,
        "source_pairing": pairing,
        "duration_contract": duration,
        "common_support_samples": common_support,
    }


def audit_cohort(cohort: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = cohort.get("records")
    if not isinstance(records, list):
        raise DiagnosticError("historical cohort records are missing")
    audit_records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for index, source in enumerate(records):
        if not isinstance(source, Mapping):
            source = {}
        audit_row, source_record = _audit_record(source, index)
        audit_records.append(audit_row)
        if source_record is not None:
            sources.append(source_record)
    passed_count = sum(bool(row["passed"]) for row in audit_records)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "stage_id": "input_audit",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": "complete" if passed_count == len(audit_records) else "blocked",
        "cohort": {
            "path": str(config.HISTORY_COHORT.resolve()),
            "sha256": config.HISTORY_COHORT_SHA256,
            "record_count": config.EXPECTED_RECORD_COUNT,
            "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
            "sample_ids_sha256": config.EXPECTED_SAMPLE_ID_SHA256,
            "order": "historical cohort order",
        },
        "record_count": len(audit_records),
        "source_group_count": len({row["source_group"] for row in audit_records if row["source_group"]}),
        "passed_count": passed_count,
        "blocked_count": len(audit_records) - passed_count,
        "source_manifest": {
            "path": str(config.HISTORY_SOURCE_MANIFEST.resolve()),
            "sha256": config.HISTORY_SOURCE_MANIFEST_SHA256,
        },
        "common_support_definition": "[0,min(N,640*F)) samples; all visual and 31 audio-lag windows must fit",
        "records": audit_records,
    }
    assert_finite(payload)
    return payload, sources


def write_input_audit(paths, payload: Mapping[str, Any]) -> dict[str, Any]:
    paths.root.mkdir(parents=True, exist_ok=True)
    if paths.input_audit.is_file():
        existing = verify_self_hashed_json(paths.input_audit)
        old = dict(existing)
        old.pop("artifact_sha256", None)
        if old != dict(payload):
            raise DiagnosticError("existing input audit differs; use a new run id")
    elif paths.final.is_file():
        raise DiagnosticError("terminal run cannot be audited again")
    else:
        write_self_hashed_json(paths.input_audit, payload)
    result = verify_self_hashed_json(paths.input_audit)
    result["_path"] = str(paths.input_audit)
    result["_sha256"] = file_sha256(paths.input_audit)
    return result


def build_protocol(
    records: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
    crop_records: Sequence[Mapping[str, Any]],
    input_audit: Mapping[str, Any],
) -> dict[str, Any]:
    if len(records) != config.EXPECTED_RECORD_COUNT or len(crop_records) != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("protocol input record count is invalid")
    crop_by_id = {str(row.get("sample_id")): row for row in crop_records}
    frozen: list[dict[str, Any]] = []
    for source in records:
        sample_id = str(source["sample_id"])
        crop = crop_by_id.get(sample_id)
        if crop is None:
            raise DiagnosticError(f"crop record is missing: {sample_id}")
        row = dict(source)
        row["crop"] = dict(crop)
        frozen.append(row)
    audit_sha = str(input_audit.get("_sha256", ""))
    audit_artifact_sha = str(input_audit.get("artifact_sha256", ""))
    if not audit_sha or not audit_artifact_sha or input_audit.get("status") != "complete":
        raise DiagnosticError("input audit is not complete or hashed")
    payload: dict[str, Any] = {
        "schema_version": 2,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": "locked",
        "experiment": "lrs3-real-video-local-timing-diagnostic",
        "classification": "fit_only_forward_endpoint_diagnostic",
        "cohort": {
            "path": str(config.HISTORY_COHORT.resolve()),
            "sha256": config.HISTORY_COHORT_SHA256,
            "record_count": config.EXPECTED_RECORD_COUNT,
            "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
            "sample_ids_sha256": config.EXPECTED_SAMPLE_ID_SHA256,
            "order": "historical cohort order",
        },
        "config": config.FrozenConfig().to_dict(),
        "spec_bindings": registered_spec_bindings(),
        "parent_blocked_run": parent_blocked_binding(),
        "input_audit": {
            "path": str(input_audit.get("_path", "")),
            "sha256": audit_sha,
            "artifact_sha256": audit_artifact_sha,
        },
        "arms": list(config.ARMS),
        "cells": list(config.ARMS),
        "runtime_bindings": dict(bindings),
        "source_hashes": package_source_hashes(),
        "records": frozen,
        "sealed_scope": {
            "natural_audio_modified": False,
            "global_offset_alignment": False,
            "tts_generation": False,
            "wav2lip": False,
            "training": False,
            "heldout_or_sealed_media_accessed": False,
            "score_based_retry": False,
            "record_filtering": False,
            "history_overwrite": False,
        },
    }
    assert_finite(payload)
    return payload


def write_protocol(paths, payload: Mapping[str, Any]) -> dict[str, Any]:
    paths.root.mkdir(parents=True, exist_ok=True)
    if paths.protocol.is_file():
        existing = verify_self_hashed_json(paths.protocol)
        old = dict(existing)
        old.pop("artifact_sha256", None)
        if old != dict(payload):
            raise DiagnosticError("existing protocol differs; use a new run id")
    elif paths.final.is_file():
        raise DiagnosticError("terminal run cannot be replaced")
    else:
        write_self_hashed_json(paths.protocol, payload)
    result = verify_self_hashed_json(paths.protocol)
    result["_path"] = str(paths.protocol)
    result["_sha256"] = file_sha256(paths.protocol)
    return result


def load_protocol(paths) -> dict[str, Any]:
    payload = verify_self_hashed_json(paths.protocol)
    if (
        payload.get("schema_version") != 2
        or payload.get("protocol_id") != config.PROTOCOL_ID
        or payload.get("protocol_revision") != config.PROTOCOL_REVISION
        or payload.get("status") != "locked"
    ):
        raise DiagnosticError("protocol is not locked")
    if payload.get("cohort", {}).get("sha256") != config.HISTORY_COHORT_SHA256:
        raise DiagnosticError("protocol is bound to an unexpected cohort")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("protocol record count is invalid")
    audit = payload.get("input_audit")
    if (
        not isinstance(audit, Mapping)
        or Path(str(audit.get("path", ""))).resolve() != paths.input_audit.resolve()
        or not paths.input_audit.is_file()
        or str(audit.get("sha256")) != file_sha256(paths.input_audit)
    ):
        raise DiagnosticError("protocol input audit binding is invalid")
    audit_payload = verify_self_hashed_json(paths.input_audit)
    if audit_payload.get("protocol_revision") != config.PROTOCOL_REVISION or audit_payload.get("status") != "complete":
        raise DiagnosticError("protocol input audit is not complete")
    payload["_path"] = str(paths.protocol)
    payload["_sha256"] = file_sha256(paths.protocol)
    return payload


def source_records(cohort: Mapping[str, Any]) -> list[dict[str, Any]]:
    audit, result = audit_cohort(cohort)
    if audit.get("status") != "complete":
        failures = [
            f"{row.get('sample_id')}: {row.get('errors', ['unknown input audit error'])[0]}"
            for row in audit.get("records", [])
            if not row.get("passed")
        ]
        detail = "; ".join(failures[:3])
        suffix = "" if len(failures) <= 3 else f"; ... {len(failures)} records blocked"
        raise DiagnosticError(f"input audit is blocked: {detail}{suffix}")
    if len(result) != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("input audit did not produce the complete cohort")
    return result
