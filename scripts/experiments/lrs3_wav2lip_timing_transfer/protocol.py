from __future__ import annotations

import math
import wave
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.lrs3_real_video_local_timing.protocol import (
    ffprobe_json,
    video_timeline,
)

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


def _resolve_path(value: Any) -> Path:
    candidate = Path(str(value))
    if not candidate.is_absolute():
        candidate = config.REPO / candidate
    return candidate.resolve()


def media_video_timeline(path: Path) -> dict[str, Any]:
    """Read a video-only or muxed output without requiring stream.duration."""
    payload = ffprobe_json(path, count_frames=True, show_frames=True)
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise DiagnosticError(f"video stream list is missing: {path}")
    videos = [stream for stream in streams if isinstance(stream, Mapping) and stream.get("codec_type") == "video"]
    if len(videos) != 1:
        raise DiagnosticError(f"media must contain exactly one video stream: {path}")
    stream = videos[0]
    try:
        frame_rate = float(Fraction(str(stream.get("r_frame_rate", ""))))
        time_base = Fraction(str(stream.get("time_base", "")))
    except (ValueError, ZeroDivisionError) as exc:
        raise DiagnosticError(f"media video rate/time base is invalid: {path}") from exc
    if abs(frame_rate - config.FPS) > 1e-6:
        raise DiagnosticError(f"media video is not {config.FPS} fps: {path}")
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise DiagnosticError(f"media video PTS evidence is missing: {path}")
    frame_count = len(raw_frames)
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise DiagnosticError(f"media video geometry is invalid: {path}")
    pts: list[int] = []
    pts_time: list[float] = []
    evidence: list[dict[str, Any]] = []
    first_time: Fraction | None = None
    maximum_deviation = Fraction(0, 1)
    for index, frame in enumerate(raw_frames):
        if not isinstance(frame, Mapping):
            raise DiagnosticError(f"media frame evidence is malformed: {path}")
        field = "best_effort_timestamp" if frame.get("best_effort_timestamp") is not None else "pts"
        raw_pts = frame.get(field)
        if raw_pts is None:
            raise DiagnosticError(f"media frame PTS is missing: {path} frame={index}")
        try:
            value = int(str(raw_pts))
        except (TypeError, ValueError) as exc:
            raise DiagnosticError(f"media frame PTS is invalid: {path} frame={index}") from exc
        actual = value * time_base
        if first_time is None:
            first_time = actual
        elif value <= pts[-1]:
            raise DiagnosticError(f"media frame PTS is not increasing: {path} frame={index}")
        deviation = abs((actual - first_time) - Fraction(index, config.FPS))
        maximum_deviation = max(maximum_deviation, deviation)
        if deviation > Fraction(1, 1000):
            raise DiagnosticError(f"media frame PTS is not a 25 fps timeline: {path} frame={index}")
        pts.append(value)
        pts_time.append(float(actual))
        evidence.append({"index": index, "field": field, "pts": value, "pts_time": float(actual)})
    assert first_time is not None
    start_time_value = stream.get("start_time")
    try:
        start_time = float(start_time_value) if start_time_value not in (None, "N/A") else float(first_time)
    except (TypeError, ValueError) as exc:
        raise DiagnosticError(f"media start time is invalid: {path}") from exc
    if abs(float(first_time)) > 0.001:
        raise DiagnosticError(f"media does not start at zero: {path}")
    duration_value = stream.get("duration")
    duration = None if duration_value in (None, "N/A") else float(duration_value)
    return {"frame_count": frame_count, "frame_rate": frame_rate, "width": width, "height": height, "start_time": start_time, "duration": duration, "time_base": str(stream.get("time_base")), "codec_name": stream.get("codec_name"), "first_pts": pts[0], "first_pts_time": pts_time[0], "pts": pts, "pts_time": pts_time, "pts_strictly_increasing": True, "pts_max_deviation_ms": float(maximum_deviation * 1000), "pts_evidence": {"time_base": str(stream.get("time_base")), "frame_count": frame_count, "frames": evidence}}


def _verify_json_asset(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    require_file(path, label)
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise DiagnosticError(f"{label} hash changed: {path}")
    return verify_self_hashed_json(path)


def _asset_detail(value: Any, label: str) -> tuple[dict[str, Any], Path | None, list[str]]:
    detail: dict[str, Any] = {"path": None, "expected_sha256": None, "actual_sha256": None, "hash_match": False}
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return detail, None, [f"asset is malformed: {label}"]
    path = _resolve_path(value.get("path", ""))
    expected = str(value.get("sha256", ""))
    detail.update({"path": str(path), "expected_sha256": expected or None})
    try:
        assert_not_sealed(path)
    except DiagnosticError as exc:
        return detail, path, [str(exc)]
    if not path.is_file():
        return detail, path, [f"asset is missing: {label}: {path}"]
    actual = file_sha256(path)
    detail["actual_sha256"] = actual
    detail["hash_match"] = bool(expected and actual == expected)
    if not detail["hash_match"]:
        errors.append(f"asset hash changed: {label}")
    return detail, path, errors


def pcm_wav_metadata(path: Path) -> tuple[dict[str, Any], bytes]:
    require_file(path, "WAV audio")
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            count = handle.getnframes()
            pcm = handle.readframes(count)
    except (OSError, wave.Error) as exc:
        raise DiagnosticError(f"audio is not a readable WAV: {path}") from exc
    if (channels, width, rate) != (config.PCM_CHANNELS, config.PCM_SAMPLE_WIDTH, config.SAMPLE_RATE):
        raise DiagnosticError(f"audio is not 16 kHz mono PCM16: {path}")
    if len(pcm) != count * config.PCM_CHANNELS * config.PCM_SAMPLE_WIDTH:
        raise DiagnosticError(f"audio PCM byte length is inconsistent: {path}")
    return (
        {
            "sample_count": int(count),
            "sample_rate": int(rate),
            "channels": int(channels),
            "sample_width": int(width),
            "container_sha256": file_sha256(path),
            "decoded_pcm_sha256": bytes_sha256(pcm),
        },
        pcm,
    )


def audio_forward_map(sample_count: int) -> np.ndarray:
    if sample_count < 2 or sample_count - 1 <= 2.0 * math.pi * config.WARP_AMPLITUDE_SAMPLES:
        raise DiagnosticError("natural audio is too short for the registered warp")
    coordinates = np.arange(sample_count, dtype=np.float64)
    mapped = coordinates + config.WARP_AMPLITUDE_SAMPLES * np.sin(2.0 * np.pi * coordinates / float(sample_count - 1))
    mapped[0] = 0.0
    mapped[-1] = float(sample_count - 1)
    if not np.isfinite(mapped).all() or np.any(np.diff(mapped) <= 0.0) or mapped[0] != 0.0 or mapped[-1] != sample_count - 1:
        raise DiagnosticError("registered audio map is not strictly increasing")
    return mapped


def reconstruct_warped_pcm(natural_pcm: bytes) -> tuple[bytes, np.ndarray]:
    if len(natural_pcm) % config.PCM_SAMPLE_WIDTH:
        raise DiagnosticError("natural PCM is not aligned to int16 samples")
    samples = np.frombuffer(natural_pcm, dtype="<i2")
    mapped = audio_forward_map(int(samples.size))
    original = np.arange(samples.size, dtype=np.float64)
    warped = np.interp(mapped, original, samples.astype(np.float64))
    output = np.rint(warped).astype("<i2", copy=False).tobytes()
    return output, mapped


def _interpolate_forward(mapped: np.ndarray, coordinate: float) -> float:
    return float(np.interp(float(coordinate), np.arange(mapped.size, dtype=np.float64), mapped))


def _interpolate_inverse(mapped: np.ndarray, coordinate: float) -> float:
    return float(np.interp(float(coordinate), mapped, np.arange(mapped.size, dtype=np.float64)))


def timing_masks(sample_count: int, frame_counts: Mapping[str, int]) -> dict[str, Any]:
    mapped = audio_forward_map(sample_count)
    q = min(int(frame_counts[config.VIDEO_R]), int(frame_counts[config.VIDEO_GN]), int(frame_counts[config.VIDEO_GW]), sample_count // config.SAMPLES_PER_FRAME)
    candidate = list(range(config.VSHIFT, q - 20))
    plus: list[int] = []
    minus: list[int] = []
    d_by_row: dict[str, float] = {}
    a_by_row: dict[str, float] = {}
    for row in candidate:
        center = float(config.SAMPLES_PER_FRAME * (row + 2))
        d = (_interpolate_forward(mapped, center) - center) / config.SAMPLES_PER_FRAME
        a = (center - _interpolate_inverse(mapped, center)) / config.SAMPLES_PER_FRAME
        d_by_row[str(row)] = d
        a_by_row[str(row)] = a
        if d >= 2.5 and a >= 2.5:
            plus.append(row)
        if d <= -2.5 and a <= -2.5:
            minus.append(row)
    if len(plus) < config.MIN_LOCAL_ROWS or len(minus) < config.MIN_LOCAL_ROWS:
        raise DiagnosticError(f"timing mask has fewer than {config.MIN_LOCAL_ROWS} rows")
    common_rows = list(range(q - config.WINDOW_FRAMES))
    return {
        "sample_count": int(sample_count),
        "q_frames": int(q),
        "candidate_rows": candidate,
        "common_window_rows": common_rows,
        "plus_rows": plus,
        "minus_rows": minus,
        "d_by_row": d_by_row,
        "a_by_row": a_by_row,
        "plus_expected_offset": float(np.mean([d_by_row[str(row)] * 0.0 + a_by_row[str(row)] for row in plus])),
        "minus_expected_offset": float(np.mean([d_by_row[str(row)] * 0.0 + a_by_row[str(row)] for row in minus])),
        "plus_expected_video_response": float(-np.mean([d_by_row[str(row)] for row in plus])),
        "minus_expected_video_response": float(-np.mean([d_by_row[str(row)] for row in minus])),
        "forward_mapping_sha256": bytes_sha256(np.asarray(mapped, dtype="<f8").tobytes()),
        "forward_mapping_dtype": "float64-little-endian",
        "forward_mapping_length": int(mapped.size),
    }


def _audio_arm_map(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    arms = row.get("arms")
    if not isinstance(arms, list):
        raise DiagnosticError(f"audio manifest arms are malformed: {row.get('sample_id')}")
    result: dict[str, Mapping[str, Any]] = {}
    for item in arms:
        if not isinstance(item, Mapping):
            raise DiagnosticError(f"audio manifest arm is malformed: {row.get('sample_id')}")
        arm = str(item.get("arm", ""))
        if arm in result:
            raise DiagnosticError(f"duplicate audio arm: {row.get('sample_id')}/{arm}")
        result[arm] = item
    return result


def _video_arm_map(row: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    arms = row.get("arms")
    if not isinstance(arms, Mapping):
        raise DiagnosticError(f"video manifest arms are malformed: {row.get('sample_id')}")
    return arms  # type: ignore[return-value]


def _runtime_bindings() -> dict[str, Any]:
    sync_root = config.REPO / "third_party/syncnet_python"
    sync_python = Path.home() / ".venvs/syncnet/bin/python"
    worker = config.REPO / "scripts/experiments/lrs3_real_video_local_timing/syncnet_worker.py"
    paths = {
        "syncnet_python": sync_python,
        "syncnet_model": sync_root / "data/syncnet_v2.model",
        "syncnet_worker": worker,
        "syncnet_instance": sync_root / "SyncNetInstance.py",
        "syncnet_model_code": sync_root / "SyncNetModel.py",
        "run_syncnet": sync_root / "run_syncnet.py",
        "ffmpeg": Path("/home/wjj/miniconda3/bin/ffmpeg"),
        "ffprobe": Path("/home/wjj/miniconda3/bin/ffprobe"),
    }
    result: dict[str, Any] = {}
    for name, path in paths.items():
        require_file(path, f"runtime binding {name}")
        result[name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    if result["syncnet_model"]["sha256"] != "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442":
        raise DiagnosticError("SyncNet V2 model hash differs from the frozen model")
    return result


def _package_source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {path.name: file_sha256(path) for path in sorted(package.glob("*.py"))}


def _verify_spec_bindings() -> dict[str, dict[str, str]]:
    bindings = config.spec_bindings()
    for name, item in bindings.items():
        path = Path(item["path"])
        require_file(path, f"spec binding {name}")
        if file_sha256(path) != item["sha256"]:
            raise DiagnosticError(f"spec binding hash changed: {name}")
    return bindings


def _load_history_assets() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    cohort = _verify_json_asset(config.COHORT, config.COHORT_SHA256, "cohort")
    calibration_final = _verify_json_asset(config.CALIBRATION_FINAL, config.CALIBRATION_FINAL_SHA256, "calibration final")
    audio_manifest = _verify_json_asset(config.AUDIO_MANIFEST, config.AUDIO_MANIFEST_SHA256, "audio manifest")
    videos_manifest = _verify_json_asset(config.VIDEOS_MANIFEST, config.VIDEOS_MANIFEST_SHA256, "videos manifest")
    tail_final = _verify_json_asset(config.TAIL_FINAL, config.TAIL_FINAL_SHA256, "real-video tail final")
    if calibration_final.get("status") != "complete" or calibration_final.get("engineering_decision") != "GO":
        raise DiagnosticError("calibration final is not a complete engineering run")
    if calibration_final.get("history_cohort_sha256") != config.COHORT_SHA256 or calibration_final.get("audio_manifest_sha256") != config.AUDIO_MANIFEST_SHA256 or calibration_final.get("videos_manifest_sha256") != config.VIDEOS_MANIFEST_SHA256:
        raise DiagnosticError("calibration final does not bind the frozen manifests")
    if audio_manifest.get("status") != "complete" or audio_manifest.get("record_count") != config.EXPECTED_RECORD_COUNT or audio_manifest.get("audio_cell_count") != 66:
        raise DiagnosticError("audio manifest is incomplete")
    if videos_manifest.get("status") != "complete" or videos_manifest.get("record_count") != config.EXPECTED_RECORD_COUNT or videos_manifest.get("video_count") != 66:
        raise DiagnosticError("video manifest is incomplete")
    if tail_final.get("status") != "complete" or tail_final.get("scientific_decision") != "LOCAL_TIMING_DETECTED":
        raise DiagnosticError("tail_v2 final is not the registered completed real-video diagnostic")
    if tail_final.get("protocol_sha256") != file_sha256(config.TAIL_PROTOCOL):
        raise DiagnosticError("tail_v2 final does not bind its protocol")
    tail_protocol = verify_self_hashed_json(config.TAIL_PROTOCOL)
    if tail_protocol.get("status") != "locked" or tail_protocol.get("protocol_revision") != "bounded_audio_tail_v2":
        raise DiagnosticError("tail_v2 protocol is not locked")
    if cohort.get("status") != "complete" or cohort.get("record_count") != config.EXPECTED_RECORD_COUNT or cohort.get("source_group_count") != config.EXPECTED_SOURCE_GROUP_COUNT:
        raise DiagnosticError("cohort is incomplete")
    return cohort, calibration_final, audio_manifest, videos_manifest, tail_protocol


def _audit_record(
    source: Mapping[str, Any],
    tail: Mapping[str, Any] | None,
    audio_row: Mapping[str, Any] | None,
    video_row: Mapping[str, Any] | None,
    index: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sample_id = str(source.get("sample_id", "")) or None
    source_group = str(source.get("source_group", "")) or None
    label = sample_id or f"record_{index}"
    errors: list[str] = []
    checks: dict[str, str] = {}
    face_detail, face_path, face_errors = _asset_detail(source.get("face_video"), f"{label}/face_video")
    natural_detail, natural_path, natural_errors = _asset_detail(source.get("natural_audio"), f"{label}/natural_audio")
    errors.extend([*face_errors, *natural_errors])
    checks["cohort_assets"] = "PASS" if not face_errors and not natural_errors else "FAIL"
    real_timeline: dict[str, Any] | None = None
    natural_meta: dict[str, Any] | None = None
    natural_pcm: bytes | None = None
    if face_path is not None and face_detail["hash_match"]:
        try:
            real_timeline = video_timeline(face_path)
            checks["real_video_timeline"] = "PASS"
        except (DiagnosticError, OSError, ValueError) as exc:
            checks["real_video_timeline"] = "FAIL"
            errors.append(str(exc))
    else:
        checks["real_video_timeline"] = "NOT_RUN"
    if natural_path is not None and natural_detail["hash_match"]:
        try:
            natural_meta, natural_pcm = pcm_wav_metadata(natural_path)
            checks["natural_audio_format"] = "PASS"
        except (DiagnosticError, OSError, ValueError) as exc:
            checks["natural_audio_format"] = "FAIL"
            errors.append(str(exc))
    else:
        checks["natural_audio_format"] = "NOT_RUN"

    generated: dict[str, Any] = {}
    audio_assets: dict[str, Any] = {}
    audio_pcm: dict[str, bytes] = {}
    if audio_row is None:
        errors.append(f"audio manifest row is missing: {label}")
        checks["audio_manifest_join"] = "FAIL"
    else:
        try:
            if str(audio_row.get("sample_id")) != sample_id or str(audio_row.get("source_group")) != source_group:
                raise DiagnosticError(f"audio manifest identity differs: {label}")
            arm_map = _audio_arm_map(audio_row)
            for arm in (config.AUDIO_N, "N_REPEAT", config.AUDIO_W):
                manifest_arm = "LOCAL_WARP_120" if arm == config.AUDIO_W else arm
                item = arm_map.get(manifest_arm)
                if item is None:
                    raise DiagnosticError(f"audio arm is missing: {label}/{manifest_arm}")
                path = _resolve_path(item.get("output"))
                expected = str(item.get("output_sha256", ""))
                if not path.is_file() or not expected or file_sha256(path) != expected:
                    raise DiagnosticError(f"audio manifest output hash changed: {label}/{arm}")
                meta, pcm = pcm_wav_metadata(path)
                if int(item.get("sample_count", -1)) != meta["sample_count"]:
                    raise DiagnosticError(f"audio manifest sample count differs: {label}/{arm}")
                # The historical field named format.pcm_sha256 is retained as
                # provenance only: in this manifest it equals the WAV container
                # hash.  The decoded PCM hash above is the new verified value.
                audio_assets[arm] = {
                    "path": str(path),
                    "container_sha256": expected,
                    "decoded_pcm_sha256": meta["decoded_pcm_sha256"],
                    "legacy_manifest_pcm_sha256": item.get("format", {}).get("pcm_sha256"),
                    "manifest": dict(item),
                }
                audio_pcm[arm] = pcm
            if natural_pcm is not None and audio_pcm[config.AUDIO_N] != natural_pcm:
                raise DiagnosticError(f"N PCM differs from cohort natural audio: {label}")
            if natural_path is not None and str(arm_map[config.AUDIO_N].get("natural_audio_sha256")) != str(source.get("natural_audio", {}).get("sha256")):
                raise DiagnosticError(f"N manifest source binding differs: {label}")
            reconstructed, mapped = reconstruct_warped_pcm(audio_pcm[config.AUDIO_N])
            if reconstructed != audio_pcm[config.AUDIO_W]:
                raise DiagnosticError(f"W cannot be exactly reconstructed from N: {label}")
            audio_assets["W"]["decoded_pcm_sha256"] = bytes_sha256(audio_pcm[config.AUDIO_W])
            audio_assets["N"]["decoded_pcm_sha256"] = bytes_sha256(audio_pcm[config.AUDIO_N])
            audio_assets["N"]["source_container_sha256"] = str(source.get("natural_audio", {}).get("sha256"))
            audio_assets["W"]["reconstruction"] = {
                "formula": "s[n]=n+1920*sin(2*pi*n/(L-1)); endpoints forced",
                "interpolation": "float64 linear",
                "rounding": "nearest_ties_to_even",
                "mapping_sha256": bytes_sha256(np.asarray(mapped, dtype="<f8").tobytes()),
            }
            checks["audio_manifest_join"] = "PASS"
            checks["warp_reconstruction"] = "PASS"
        except (DiagnosticError, OSError, ValueError, TypeError) as exc:
            checks["audio_manifest_join"] = "FAIL"
            checks["warp_reconstruction"] = "FAIL"
            errors.append(str(exc))
    if "audio_manifest_join" not in checks:
        checks["warp_reconstruction"] = "NOT_RUN"

    if video_row is None:
        errors.append(f"video manifest row is missing: {label}")
        checks["video_manifest_join"] = "FAIL"
    else:
        try:
            if str(video_row.get("sample_id")) != sample_id or str(video_row.get("source_group")) != source_group:
                raise DiagnosticError(f"video manifest identity differs: {label}")
            arm_map = _video_arm_map(video_row)
            for arm, video_key in ((config.VIDEO_GN, "N"), (config.VIDEO_GW, "LOCAL_WARP_120")):
                item = arm_map.get(video_key)
                if not isinstance(item, Mapping):
                    raise DiagnosticError(f"generated video arm is missing: {label}/{video_key}")
                path = _resolve_path(item.get("output"))
                expected = str(item.get("output_sha256", ""))
                if not path.is_file() or not expected or file_sha256(path) != expected:
                    raise DiagnosticError(f"generated video hash changed: {label}/{video_key}")
                timeline = video_timeline(path)
                manifest_count = int(str(item.get("video_stream", {}).get("nb_frames", -1)))
                if timeline["frame_count"] != manifest_count:
                    raise DiagnosticError(f"generated video frame count differs from manifest: {label}/{video_key}")
                generated[arm] = {
                    "path": str(path),
                    "sha256": expected,
                    "manifest": dict(item),
                    "timeline": timeline,
                }
            if generated[config.VIDEO_GN]["timeline"]["frame_count"] != generated[config.VIDEO_GW]["timeline"]["frame_count"]:
                raise DiagnosticError(f"G_N/G_W frame counts differ: {label}")
            checks["video_manifest_join"] = "PASS"
        except (DiagnosticError, OSError, ValueError, TypeError) as exc:
            checks["video_manifest_join"] = "FAIL"
            errors.append(str(exc))

    track: list[dict[str, float]] | None = None
    if tail is None:
        errors.append(f"tail_v2 record is missing: {label}")
        checks["tail_join"] = "FAIL"
    else:
        try:
            if str(tail.get("sample_id")) != sample_id or str(tail.get("source_group")) != source_group:
                raise DiagnosticError(f"tail_v2 identity differs: {label}")
            tail_face = tail.get("face_video")
            tail_audio = tail.get("natural_audio")
            if not isinstance(tail_face, Mapping) or not isinstance(tail_audio, Mapping) or str(tail_face.get("path")) != str(source.get("face_video", {}).get("path")) or str(tail_audio.get("path")) != str(source.get("natural_audio", {}).get("path")) or str(tail_face.get("sha256")) != str(source.get("face_video", {}).get("sha256")) or str(tail_audio.get("sha256")) != str(source.get("natural_audio", {}).get("sha256")):
                raise DiagnosticError(f"tail_v2 source binding differs: {label}")
            crop = tail.get("crop")
            raw_track = crop.get("processed_track") if isinstance(crop, Mapping) else None
            if not isinstance(raw_track, list):
                raise DiagnosticError(f"tail_v2 processed track is missing: {label}")
            track = []
            for item in raw_track:
                if not isinstance(item, Mapping):
                    raise DiagnosticError(f"tail_v2 processed track item is malformed: {label}")
                value = {"x": float(item["x"]), "y": float(item["y"]), "s": float(item["s"])}
                if not np.isfinite(list(value.values())).all() or value["s"] <= 0:
                    raise DiagnosticError(f"tail_v2 processed track item is invalid: {label}")
                track.append(value)
            checks["tail_join"] = "PASS"
        except (DiagnosticError, KeyError, TypeError, ValueError) as exc:
            checks["tail_join"] = "FAIL"
            errors.append(str(exc))

    source_frame_count = real_timeline["frame_count"] if real_timeline is not None else None
    if track is not None and source_frame_count is not None and len(track) != source_frame_count:
        errors.append(f"processed track length differs from real video: {label}")
        checks["track_length"] = "FAIL"
    elif track is not None:
        checks["track_length"] = "PASS"
    else:
        checks["track_length"] = "NOT_RUN"

    duration: dict[str, Any] | None = None
    masks: dict[str, Any] | None = None
    if natural_meta is not None and source_frame_count is not None and generated:
        generated_frames = {key: int(value["timeline"]["frame_count"]) for key, value in generated.items()}
        if generated_frames[config.VIDEO_GN] != generated_frames[config.VIDEO_GW] or generated_frames[config.VIDEO_GN] > source_frame_count:
            errors.append(f"generated videos are not equal-length prefixes: {label}")
            checks["generated_prefix"] = "FAIL"
        else:
            checks["generated_prefix"] = "PASS"
        delta_real = int(natural_meta["sample_count"]) - source_frame_count * config.SAMPLES_PER_FRAME
        duration = {
            "real_delta_samples": delta_real,
            "real_delta_ms": float(delta_real * 1000 / config.SAMPLE_RATE),
            "real_bounds_samples": [config.AUDIO_TAIL_MIN_SAMPLES, config.AUDIO_TAIL_MAX_SAMPLES],
            "real_accepted": bool(config.AUDIO_TAIL_MIN_SAMPLES <= delta_real <= config.AUDIO_TAIL_MAX_SAMPLES),
            "generated_delta_samples": {key: int(natural_meta["sample_count"]) - count * config.SAMPLES_PER_FRAME for key, count in generated_frames.items()},
        }
        if not duration["real_accepted"]:
            errors.append(f"real video tail is outside bounded_audio_tail_v2: {label}")
        checks["real_tail_contract"] = "PASS" if duration["real_accepted"] else "FAIL"
        if source_frame_count != real_timeline["frame_count"] or generated_frames[config.VIDEO_GN] <= 0:
            checks["common_support"] = "FAIL"
            errors.append(f"common support is empty: {label}")
        else:
            frame_counts = {config.VIDEO_R: source_frame_count, **generated_frames}
            try:
                masks = timing_masks(int(natural_meta["sample_count"]), frame_counts)
                checks["common_support"] = "PASS"
            except DiagnosticError as exc:
                checks["common_support"] = "FAIL"
                errors.append(str(exc))
    else:
        for name in ("generated_prefix", "real_tail_contract", "common_support"):
            checks[name] = "NOT_RUN"

    if real_timeline is not None and generated:
        if any(int(item["timeline"]["width"]) != int(real_timeline["width"]) or int(item["timeline"]["height"]) != int(real_timeline["height"]) for item in generated.values()):
            errors.append(f"generated canvas differs from real video: {label}")
            checks["canvas"] = "FAIL"
        else:
            checks["canvas"] = "PASS"
    else:
        checks["canvas"] = "NOT_RUN"

    passed = not errors and all(value == "PASS" for value in checks.values())
    audit_row: dict[str, Any] = {
        "index": index,
        "sample_id": sample_id,
        "source_group": source_group,
        "cohort_assets": {"face_video": face_detail, "natural_audio": natural_detail},
        "real_video_timeline": real_timeline,
        "natural_audio_format": natural_meta,
        "generated_video_timeline": {key: value["timeline"] for key, value in generated.items()},
        "audio_assets": audio_assets,
        "duration": duration,
        "processed_track_length": len(track) if track is not None else None,
        "masks": masks,
        "checks": checks,
        "passed": passed,
        "status": "PASS" if passed else "BLOCKED",
        "errors": errors,
    }
    if not passed or sample_id is None or source_group is None or track is None or natural_meta is None or duration is None or masks is None:
        return audit_row, None
    return audit_row, {
        "sample_id": sample_id,
        "source_group": source_group,
        "natural_audio": {
            "path": str(natural_path),
            "cohort_sha256": str(source["natural_audio"]["sha256"]),
            "pcm_sha256": str(natural_meta["decoded_pcm_sha256"]),
            "sample_count": int(natural_meta["sample_count"]),
        },
        "warp_audio": dict(audio_assets[config.AUDIO_W]),
        "real_video": {
            "path": str(face_path),
            "sha256": str(source["face_video"]["sha256"]),
            "timeline": real_timeline,
        },
        "generated_videos": generated,
        "processed_track": track,
        "track_sha256": canonical_json_sha256(track),
        "duration": duration,
        "masks": masks,
    }


def audit_inputs() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    global_errors: list[str] = []
    try:
        cohort, _calibration_final, audio_manifest, videos_manifest, tail_protocol = _load_history_assets()
    except (DiagnosticError, OSError, ValueError, TypeError) as exc:
        global_errors.append(str(exc))
        cohort = {}
        audio_manifest = {}
        videos_manifest = {}
        tail_protocol = {}
        for path in (config.COHORT, config.CALIBRATION_FINAL, config.AUDIO_MANIFEST, config.VIDEOS_MANIFEST, config.TAIL_FINAL, config.TAIL_PROTOCOL):
            try:
                candidate = read_json(path)
                if path == config.COHORT:
                    cohort = candidate
                elif path == config.TAIL_PROTOCOL:
                    tail_protocol = candidate
            except DiagnosticError:
                pass
    raw_records = cohort.get("records") if isinstance(cohort, Mapping) else None
    records = raw_records if isinstance(raw_records, list) else []
    if len(records) != config.EXPECTED_RECORD_COUNT:
        global_errors.append(f"cohort record count is not {config.EXPECTED_RECORD_COUNT}")
    tail_map = {str(row.get("sample_id")): row for row in tail_protocol.get("records", []) if isinstance(row, Mapping)} if isinstance(tail_protocol, Mapping) else {}
    audio_map = {str(row.get("sample_id")): row for row in audio_manifest.get("rows", []) if isinstance(row, Mapping)} if isinstance(audio_manifest, Mapping) else {}
    video_map = {str(row.get("sample_id")): row for row in videos_manifest.get("rows", []) if isinstance(row, Mapping)} if isinstance(videos_manifest, Mapping) else {}
    audit_records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for index in range(config.EXPECTED_RECORD_COUNT):
        source = records[index] if index < len(records) and isinstance(records[index], Mapping) else {}
        sample_id = str(source.get("sample_id", "")) or None
        if global_errors:
            row = {
                "index": index,
                "sample_id": sample_id,
                "source_group": str(source.get("source_group", "")) or None,
                "checks": {},
                "passed": False,
                "status": "BLOCKED",
                "errors": list(global_errors),
            }
            audit_records.append(row)
            continue
        audit_row, source_record = _audit_record(source, tail_map.get(sample_id or ""), audio_map.get(sample_id or ""), video_map.get(sample_id or ""), index)
        audit_records.append(audit_row)
        if source_record is not None:
            sources.append(source_record)
    ids = [str(row.get("sample_id", "")) for row in records if isinstance(row, Mapping)]
    groups = [str(row.get("source_group", "")) for row in records if isinstance(row, Mapping)]
    if len(ids) == config.EXPECTED_RECORD_COUNT and (len(set(ids)) != len(ids) or sample_ids_sha256(ids) != config.ORDERED_SAMPLE_ID_SHA256):
        global_errors.append("cohort order or sample ID hash changed")
    passed_count = sum(bool(row.get("passed")) for row in audit_records)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "stage_id": "input_audit",
        "status": "complete" if passed_count == config.EXPECTED_RECORD_COUNT and not global_errors else "blocked",
        "classification": "fit_only_diagnostic",
        "global_errors": global_errors,
        "record_count": config.EXPECTED_RECORD_COUNT,
        "source_group_count": len(set(groups)),
        "passed_count": passed_count,
        "blocked_count": config.EXPECTED_RECORD_COUNT - passed_count,
        "ordered_sample_id_sha256": config.ORDERED_SAMPLE_ID_SHA256,
        "frozen_inputs": {
            "cohort": {"path": str(config.COHORT.resolve()), "sha256": config.COHORT_SHA256},
            "calibration_final": {"path": str(config.CALIBRATION_FINAL.resolve()), "sha256": config.CALIBRATION_FINAL_SHA256},
            "audio_manifest": {"path": str(config.AUDIO_MANIFEST.resolve()), "sha256": config.AUDIO_MANIFEST_SHA256},
            "videos_manifest": {"path": str(config.VIDEOS_MANIFEST.resolve()), "sha256": config.VIDEOS_MANIFEST_SHA256},
            "tail_final": {"path": str(config.TAIL_FINAL.resolve()), "sha256": config.TAIL_FINAL_SHA256},
            "tail_protocol": {"path": str(config.TAIL_PROTOCOL.resolve()), "sha256": file_sha256(config.TAIL_PROTOCOL) if config.TAIL_PROTOCOL.is_file() else None},
        },
        "records": audit_records,
    }
    assert_finite(payload)
    return payload, sources


def write_input_audit(paths: config.RunPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    paths.root.mkdir(parents=True, exist_ok=True)
    if paths.input_audit.is_file():
        previous = verify_self_hashed_json(paths.input_audit)
        old = dict(previous)
        old.pop("artifact_sha256", None)
        if old != dict(payload):
            raise DiagnosticError("existing input audit differs; use a new run id")
    elif paths.final.is_file():
        raise DiagnosticError("terminal run cannot be audited again")
    else:
        write_self_hashed_json(paths.input_audit, payload)
    result = verify_self_hashed_json(paths.input_audit)
    result["_path"] = str(paths.input_audit.resolve())
    result["_sha256"] = file_sha256(paths.input_audit)
    return result


def build_protocol(sources: Sequence[Mapping[str, Any]], runtime: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    if len(sources) != config.EXPECTED_RECORD_COUNT or audit.get("status") != "complete":
        raise DiagnosticError("cannot lock protocol before all 22 inputs pass")
    audit_sha = str(audit.get("_sha256", ""))
    if not audit_sha:
        raise DiagnosticError("input audit hash is missing")
    records: list[dict[str, Any]] = []
    for source in sources:
        row = dict(source)
        frame_counts = {
            config.VIDEO_R: int(source["real_video"]["timeline"]["frame_count"]),
            config.VIDEO_GN: int(source["generated_videos"][config.VIDEO_GN]["timeline"]["frame_count"]),
            config.VIDEO_GW: int(source["generated_videos"][config.VIDEO_GW]["timeline"]["frame_count"]),
        }
        mapped = audio_forward_map(int(source["natural_audio"]["sample_count"]))
        masks = dict(source["masks"])
        masks["frame_counts"] = frame_counts
        masks["mapping_formula"] = "s[n]=n+1920*sin(2*pi*n/(L-1)); endpoints forced"
        masks["inverse_formula"] = "interp(t, s, n)"
        masks["center_formula"] = "t_r=640*(r+2)"
        masks["d_formula"] = "(s(t_r)-t_r)/640"
        masks["a_formula"] = "(t_r-s_inverse(t_r))/640"
        row["audio_map"] = {
            "sample_count": int(mapped.size),
            "forward_mapping_sha256": bytes_sha256(np.asarray(mapped, dtype="<f8").tobytes()),
            "dtype": "float64-little-endian",
            "interpolation": "piecewise_linear",
        }
        row["masks"] = masks
        row["crop"] = {
            "size": config.CROP_SIZE,
            "scale": config.CROP_SCALE,
            "processed_track": source["processed_track"],
            "processed_track_sha256": source["track_sha256"],
            "track_source": "tail_v2 protocol records[].crop.processed_track",
            "generated_track_policy": "prefix of the same track; no redetection",
        }
        records.append(row)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": "locked",
        "classification": "fit_only_diagnostic",
        "experiment": "lrs3-wav2lip-timing-transfer-diagnostic",
        "cohort": {"path": str(config.COHORT.resolve()), "sha256": config.COHORT_SHA256, "record_count": config.EXPECTED_RECORD_COUNT, "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT, "sample_ids_sha256": config.ORDERED_SAMPLE_ID_SHA256},
        "parent_runs": {
            "calibration_final": {"path": str(config.CALIBRATION_FINAL.resolve()), "sha256": config.CALIBRATION_FINAL_SHA256},
            "tail_final": {"path": str(config.TAIL_FINAL.resolve()), "sha256": config.TAIL_FINAL_SHA256},
            "tail_protocol": {"path": str(config.TAIL_PROTOCOL.resolve()), "sha256": file_sha256(config.TAIL_PROTOCOL)},
        },
        "input_audit": {"path": str(audit["_path"]), "sha256": audit_sha, "artifact_sha256": str(audit["artifact_sha256"])},
        "manifests": {
            "audio": {"path": str(config.AUDIO_MANIFEST.resolve()), "sha256": config.AUDIO_MANIFEST_SHA256},
            "videos": {"path": str(config.VIDEOS_MANIFEST.resolve()), "sha256": config.VIDEOS_MANIFEST_SHA256},
        },
        "spec_bindings": _verify_spec_bindings(),
        "runtime_bindings": dict(runtime),
        "source_hashes": _package_source_hashes(),
        "config": config.FrozenConfig().to_dict(),
        "arms": {"videos": list(config.VIDEO_ARMS), "audios": list(config.AUDIO_ARMS), "main_cells": [[v, a] for v, a in config.MAIN_CELL_SPECS], "repeat_cells": [[v, a] for v, a in config.REPEAT_CELL_SPECS]},
        "cell_counts": {"main": config.EXPECTED_MAIN_CELL_COUNT, "repeat": config.EXPECTED_REPEAT_CELL_COUNT, "total": config.EXPECTED_TOTAL_CELL_COUNT},
        "records": records,
        "sealed_scope": {"fit_only": True, "tts_generation": False, "wav2lip_generation": False, "training": False, "mfa": False, "dtw": False, "heldout": False, "history_overwrite": False, "reference_conditioned_audio_head_spec_eligible": False},
    }
    assert_finite(payload)
    return payload


def write_protocol(paths: config.RunPaths, payload: Mapping[str, Any]) -> dict[str, Any]:
    if paths.protocol.is_file():
        previous = verify_self_hashed_json(paths.protocol)
        old = dict(previous)
        old.pop("artifact_sha256", None)
        expected = dict(payload)
        expected.pop("artifact_sha256", None)
        if old != expected:
            raise DiagnosticError("existing protocol differs; use a new run id")
    elif paths.final.is_file():
        raise DiagnosticError("terminal run cannot be locked again")
    else:
        write_self_hashed_json(paths.protocol, payload)
    result = verify_self_hashed_json(paths.protocol)
    result["_path"] = str(paths.protocol.resolve())
    result["_sha256"] = file_sha256(paths.protocol)
    return result


def prepare(paths: config.RunPaths) -> dict[str, Any]:
    if paths.final.is_file():
        raise DiagnosticError("terminal run cannot be prepared again")
    audit_payload, sources = audit_inputs()
    audit = write_input_audit(paths, audit_payload)
    print(f"INPUT_AUDIT {audit_payload['passed_count']}/{config.EXPECTED_RECORD_COUNT} status={audit_payload['status']}", flush=True)
    if audit_payload.get("status") != "complete":
        failures = [f"{row.get('sample_id')}: {row.get('errors', ['unknown'])[0]}" for row in audit_payload.get("records", []) if not row.get("passed")]
        raise DiagnosticError("input audit is blocked: " + "; ".join(failures[:3]))
    runtime = _runtime_bindings()
    payload = build_protocol(sources, runtime, audit)
    result = write_protocol(paths, payload)
    print(f"PROTOCOL locked records={len(result['records'])}", flush=True)
    return result


def load_protocol(paths: config.RunPaths) -> dict[str, Any]:
    payload = verify_self_hashed_json(paths.protocol)
    if payload.get("protocol_id") != config.PROTOCOL_ID or payload.get("protocol_revision") != config.PROTOCOL_REVISION or payload.get("status") != "locked":
        raise DiagnosticError("protocol marker is invalid")
    if payload.get("cell_counts") != {"main": config.EXPECTED_MAIN_CELL_COUNT, "repeat": config.EXPECTED_REPEAT_CELL_COUNT, "total": config.EXPECTED_TOTAL_CELL_COUNT}:
        raise DiagnosticError("protocol cell counts are invalid")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("protocol records are incomplete")
    result = dict(payload)
    result["_path"] = str(paths.protocol.resolve())
    result["_sha256"] = file_sha256(paths.protocol)
    return result
