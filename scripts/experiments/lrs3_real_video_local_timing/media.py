from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import signal

from . import config
from .common import (
    DiagnosticError,
    bytes_sha256,
    canonical_json_sha256,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .protocol import (
    audit_cohort,
    build_protocol,
    ffprobe_json,
    load_history_cohort,
    runtime_bindings,
    write_input_audit,
    write_protocol,
)


def read_video_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise DiagnosticError(f"cannot open source video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise DiagnosticError(f"source video has an invalid frame: {path}")
            frames.append(np.ascontiguousarray(frame))
    finally:
        capture.release()
    if not frames:
        raise DiagnosticError(f"source video has no decodable frames: {path}")
    return frames


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    denominator = left_area + right_area - intersection
    return intersection / denominator if denominator > 0 else 0.0


def _track_shot(detections: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    remaining = [[dict(face) for face in frame_faces] for frame_faces in detections]
    tracks: list[dict[str, Any]] = []
    while True:
        track: list[dict[str, Any]] = []
        for frame_faces in remaining:
            for face in list(frame_faces):
                if not track:
                    track.append(face)
                    frame_faces.remove(face)
                    continue
                if int(face["frame"]) - int(track[-1]["frame"]) <= config.NUM_FAILED_DET and _iou(face["bbox"], track[-1]["bbox"]) > config.TRACK_IOU_THRESHOLD:
                    track.append(face)
                    frame_faces.remove(face)
        if not track:
            break
        if len(track) > config.MIN_TRACK:
            frames = np.asarray([int(face["frame"]) for face in track], dtype=np.int64)
            boxes = np.asarray([np.asarray(face["bbox"], dtype=np.float64) for face in track], dtype=np.float64)
            full_frames = np.arange(int(frames[0]), int(frames[-1]) + 1, dtype=np.int64)
            interpolated = np.stack([np.interp(full_frames, frames, boxes[:, column]) for column in range(4)], axis=1)
            mean_size = max(float(np.mean(interpolated[:, 2] - interpolated[:, 0])), float(np.mean(interpolated[:, 3] - interpolated[:, 1])))
            if mean_size > config.MIN_FACE_SIZE:
                tracks.append(
                    {
                        "frame": full_frames,
                        "bbox": interpolated,
                        "detection_count": len(track),
                        "mean_face_size": mean_size,
                    }
                )
    return tracks


def make_face_detector() -> Any:
    try:
        import torch

        if str(config.SYNCNET_ROOT) not in sys.path:
            sys.path.insert(0, str(config.SYNCNET_ROOT))
        from detectors import S3FD

        device = "cuda" if torch.cuda.is_available() else "cpu"
        previous_cwd = os.getcwd()
        try:
            # The vendored S3FD loader keeps its registered weight as a path
            # relative to the SyncNet checkout.
            os.chdir(config.SYNCNET_ROOT)
            return S3FD(device=device)
        finally:
            os.chdir(previous_cwd)
    except (ImportError, OSError) as exc:
        raise DiagnosticError("official S3FD detector is unavailable") from exc


def detect_full_track(frames: Sequence[np.ndarray], detector: Any) -> dict[str, Any]:
    detections: list[list[dict[str, Any]]] = []
    for frame_index, image in enumerate(frames):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = detector.detect_faces(rgb, conf_th=0.9, scales=[config.DETECTION_SCALE])
        frame_faces: list[dict[str, Any]] = []
        for box in boxes:
            values = np.asarray(box, dtype=np.float64).tolist()
            frame_faces.append({"frame": frame_index, "bbox": values[:4], "confidence": float(values[4])})
        detections.append(frame_faces)
    tracks = _track_shot(detections)
    full = [track for track in tracks if int(track["frame"][0]) == 0 and int(track["frame"][-1]) == len(frames) - 1]
    if len(full) != 1:
        raise DiagnosticError(f"expected exactly one full single-speaker crop track, found {len(full)}")
    track = full[0]
    if len(track["frame"]) != len(frames):
        raise DiagnosticError("crop track does not cover every source frame")
    return {
        "source_frame_indices": [int(value) for value in track["frame"]],
        "raw_bbox": [[float(value) for value in row] for row in np.asarray(track["bbox"])],
        "detection_count": int(track["detection_count"]),
        "mean_face_size": float(track["mean_face_size"]),
    }


def processed_crop_track(raw_bbox: Sequence[Sequence[float]], frame_count: int) -> list[dict[str, float]]:
    boxes = np.asarray(raw_bbox, dtype=np.float64)
    if boxes.shape != (frame_count, 4) or not np.isfinite(boxes).all():
        raise DiagnosticError("crop bounding-box track is malformed")
    sizes = np.maximum(boxes[:, 3] - boxes[:, 1], boxes[:, 2] - boxes[:, 0]) / 2.0
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    kernel = 13 if frame_count >= 13 else (frame_count if frame_count % 2 else frame_count - 1)
    if kernel >= 3:
        sizes = signal.medfilt(sizes, kernel_size=kernel)
        centers_x = signal.medfilt(centers_x, kernel_size=kernel)
        centers_y = signal.medfilt(centers_y, kernel_size=kernel)
    return [
        {"x": float(x), "y": float(y), "s": float(size)}
        for x, y, size in zip(centers_x, centers_y, sizes, strict=True)
    ]


def crop_frames_from_track(frames: Sequence[np.ndarray], proc_track: Sequence[Mapping[str, float]]) -> list[np.ndarray]:
    if len(frames) != len(proc_track):
        raise DiagnosticError("source frames and crop track length differ")
    result: list[np.ndarray] = []
    for image, item in zip(frames, proc_track, strict=True):
        scale = float(item["s"])
        center_y = float(item["y"])
        center_x = float(item["x"])
        if scale <= 0:
            raise DiagnosticError("crop track contains a non-positive scale")
        border = int(scale * (1.0 + 2.0 * config.CROP_SCALE))
        padded = np.pad(image, ((border, border), (border, border), (0, 0)), mode="constant", constant_values=110)
        padded_y = center_y + border
        padded_x = center_x + border
        face = padded[
            int(padded_y - scale) : int(padded_y + scale * (1.0 + 2.0 * config.CROP_SCALE)),
            int(padded_x - scale * (1.0 + config.CROP_SCALE)) : int(padded_x + scale * (1.0 + config.CROP_SCALE)),
        ]
        if face.size == 0:
            raise DiagnosticError("crop track produced an empty crop")
        resized = cv2.resize(face, (config.CROP_SIZE, config.CROP_SIZE))
        result.append(np.ascontiguousarray(resized))
    return result


def build_mapping(frame_count: int) -> dict[str, list[int]]:
    if frame_count < 2:
        raise DiagnosticError("video needs at least two frames for local warp")
    indices = np.arange(frame_count, dtype=np.float64)
    warped = indices + config.WARP_AMPLITUDE_FRAMES * np.sin(2.0 * np.pi * indices / float(frame_count - 1))
    warped[0] = 0.0
    warped[-1] = float(frame_count - 1)
    selected = np.rint(warped).astype(np.int64)
    displacement = selected - np.arange(frame_count, dtype=np.int64)
    if (
        int(selected[0]) != 0
        or int(selected[-1]) != frame_count - 1
        or int(selected.min()) < 0
        or int(selected.max()) >= frame_count
        or bool(np.any(np.diff(selected) < 0))
        or not bool(np.any(displacement == 3))
        or not bool(np.any(displacement == -3))
    ):
        raise DiagnosticError("registered video mapping is invalid")
    return {
        config.REAL_ARM: list(range(frame_count)),
        config.REPEAT_ARM: list(range(frame_count)),
        config.WARP_ARM: [int(value) for value in selected],
    }


def _raw_frame_bytes(frames: Iterable[np.ndarray]) -> bytes:
    chunks: list[bytes] = []
    for frame in frames:
        value = np.asarray(frame)
        if value.shape != (config.CROP_SIZE, config.CROP_SIZE, 3) or value.dtype != np.uint8:
            raise DiagnosticError("all crop frames must be uint8 224x224 BGR")
        chunks.append(np.ascontiguousarray(value).tobytes())
    if not chunks:
        raise DiagnosticError("cannot encode an empty video")
    return b"".join(chunks)


def encode_ffv1(frames: Sequence[np.ndarray], output: Path, log_path: Path | None = None) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.partial{output.suffix}")
    command = [
        str(config.FFMPEG),
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{config.CROP_SIZE}x{config.CROP_SIZE}",
        "-r",
        str(config.FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-g",
        "1",
        "-pix_fmt",
        "bgr0",
        "-f",
        "matroska",
        str(temporary),
    ]
    raw = _raw_frame_bytes(frames)
    result = subprocess.run(command, input=raw, capture_output=True, check=False)
    if result.returncode != 0 or not temporary.is_file():
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_bytes(result.stdout + result.stderr)
        temporary.unlink(missing_ok=True)
        raise DiagnosticError(f"FFV1 encoding failed: {output}")
    temporary.replace(output)
    return {
        "output": str(output),
        "output_sha256": file_sha256(output),
        "frame_count": len(frames),
        "raw_frame_sha256": bytes_sha256(raw),
        "command": command,
        "command_sha256": canonical_json_sha256(command),
    }


def mux_pcm(video_only: Path, audio: Path, output: Path, log_path: Path | None = None) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.partial{output.suffix}")
    command = [
        str(config.FFMPEG),
        "-y",
        "-v",
        "error",
        "-i",
        str(video_only),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-f",
        "matroska",
        str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0 or not temporary.is_file():
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_bytes(result.stdout + result.stderr)
        temporary.unlink(missing_ok=True)
        raise DiagnosticError(f"PCM mux failed: {output}")
    temporary.replace(output)
    return {"output": str(output), "output_sha256": file_sha256(output), "command": command, "command_sha256": canonical_json_sha256(command)}


def decode_video_frames(path: Path) -> list[np.ndarray]:
    payload = ffprobe_json(path, count_frames=True)
    streams = payload.get("streams", [])
    videos = [stream for stream in streams if isinstance(stream, Mapping) and stream.get("codec_type") == "video"]
    if len(videos) != 1:
        raise DiagnosticError(f"media must contain exactly one video stream: {path}")
    width = int(videos[0].get("width", 0))
    height = int(videos[0].get("height", 0))
    result = subprocess.run(
        [str(config.FFMPEG), "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"],
        capture_output=True,
        check=True,
    )
    stride = width * height * 3
    if width <= 0 or height <= 0 or len(result.stdout) % stride != 0:
        raise DiagnosticError(f"decoded media frame bytes are malformed: {path}")
    return [np.frombuffer(result.stdout[start : start + stride], dtype=np.uint8).reshape(height, width, 3).copy() for start in range(0, len(result.stdout), stride)]


def decode_pcm16(path: Path) -> bytes:
    result = subprocess.run(
        [str(config.FFMPEG), "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(config.SAMPLE_RATE), "-ac", "1", "pipe:1"],
        capture_output=True,
        check=True,
    )
    return bytes(result.stdout)


def source_pcm16(path: Path) -> bytes:
    import wave

    with wave.open(str(path), "rb") as handle:
        return handle.readframes(handle.getnframes())


def verify_muxed_media(path: Path, expected_frames: Sequence[np.ndarray], expected_pcm: bytes, expected_mapping: Sequence[int]) -> dict[str, Any]:
    payload = ffprobe_json(path, count_frames=True)
    streams = payload.get("streams", [])
    videos = [stream for stream in streams if isinstance(stream, Mapping) and stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if isinstance(stream, Mapping) and stream.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise DiagnosticError(f"media must contain exactly one video and one audio stream: {path}")
    video = videos[0]
    audio = audios[0]
    try:
        frame_rate = float(Fraction(str(video.get("r_frame_rate", ""))))
    except (ValueError, ZeroDivisionError):
        raise DiagnosticError(f"media frame rate is invalid: {path}") from None
    if str(video.get("codec_name")) != "ffv1" or abs(frame_rate - config.FPS) > 1e-6 or int(video.get("nb_read_frames", video.get("nb_frames", -1))) != len(expected_frames):
        raise DiagnosticError(f"media video stream does not match the frozen timeline: {path}")
    if str(audio.get("codec_name")) != "pcm_s16le" or int(audio.get("sample_rate", 0)) != config.SAMPLE_RATE or int(audio.get("channels", 0)) != config.PCM_CHANNELS:
        raise DiagnosticError(f"media audio stream is not mono PCM16: {path}")
    actual_pcm = decode_pcm16(path)
    if actual_pcm != expected_pcm:
        raise DiagnosticError(f"muxed PCM differs from natural audio: {path}")
    actual_frames = decode_video_frames(path)
    if len(actual_frames) != len(expected_mapping):
        raise DiagnosticError(f"media frame count differs from mapping: {path}")
    for index, source_index in enumerate(expected_mapping):
        if not np.array_equal(actual_frames[index], expected_frames[source_index]):
            raise DiagnosticError(f"encoded frame differs from the frozen crop sequence: {path} frame {index}")
    return {
        "sha256": file_sha256(path),
        "video_stream": {key: video.get(key) for key in ("codec_name", "width", "height", "r_frame_rate", "time_base", "nb_frames", "nb_read_frames")},
        "audio_stream": {key: audio.get(key) for key in ("codec_name", "sample_rate", "channels", "channel_layout")},
        "audio_pcm_sha256": bytes_sha256(actual_pcm),
        "audio_pcm_verified": True,
        "frame_count": len(actual_frames),
        "mapping_sha256": canonical_json_sha256([int(value) for value in expected_mapping]),
    }


def _crop_record(source: Mapping[str, Any], detector: Any) -> dict[str, Any]:
    sample_id = str(source["sample_id"])
    video = Path(str(source["face_video"]["path"]))
    timeline = source["source_video_timeline"]
    frames = read_video_frames(video)
    if len(frames) != int(timeline["frame_count"]):
        raise DiagnosticError(f"decoded frame count differs from ffprobe: {sample_id}")
    track = detect_full_track(frames, detector)
    proc_track = processed_crop_track(track["raw_bbox"], len(frames))
    mappings = build_mapping(len(frames))
    return {
        "frame_count": len(frames),
        "crop_size": config.CROP_SIZE,
        "crop_scale": config.CROP_SCALE,
        "detection_scale": config.DETECTION_SCALE,
        "track": track,
        "processed_track": proc_track,
        "mappings": mappings,
        "mapping_sha256": {arm: canonical_json_sha256(mapping) for arm, mapping in mappings.items()},
    }


def prepare(paths) -> dict[str, Any]:
    if paths.final.is_file():
        raise DiagnosticError("terminal run cannot be prepared again")
    cohort = load_history_cohort()
    audit_payload, sources = audit_cohort(cohort)
    audit = write_input_audit(paths, audit_payload)
    print(
        f"INPUT_AUDIT {audit_payload['passed_count']}/{audit_payload['record_count']} "
        f"status={audit_payload['status']}",
        flush=True,
    )
    if audit_payload.get("status") != "complete":
        failures = [
            f"{row.get('sample_id')}: {row.get('errors', ['unknown input audit error'])[0]}"
            for row in audit_payload.get("records", [])
            if not row.get("passed")
        ]
        detail = "; ".join(failures[:3])
        suffix = "" if len(failures) <= 3 else f"; ... {len(failures)} records blocked"
        raise DiagnosticError(f"input audit is blocked: {detail}{suffix}")
    bindings = runtime_bindings()
    detector = make_face_detector()
    crop_records: list[dict[str, Any]] = []
    for index, source in enumerate(sources, 1):
        crop_records.append({"sample_id": source["sample_id"], **_crop_record(source, detector)})
        print(f"PREPARE {index}/{len(sources)} {source['sample_id']}", flush=True)
    payload = build_protocol(sources, bindings, crop_records, audit)
    return write_protocol(paths, payload)


def _expected_mapping(protocol_record: Mapping[str, Any], arm: str) -> list[int]:
    mappings = protocol_record.get("crop", {}).get("mappings", {})
    mapping = mappings.get(arm)
    if not isinstance(mapping, list):
        raise DiagnosticError(f"mapping is missing: {protocol_record.get('sample_id')}/{arm}")
    return [int(value) for value in mapping]


def _media_sidecar_matches(sidecar: Mapping[str, Any], protocol_sha: str, sample_id: str, arm: str, output: Path, crop_base: Path, audio: Path, mapping: Sequence[int]) -> bool:
    return bool(
        sidecar.get("protocol_id") == config.PROTOCOL_ID
        and sidecar.get("protocol_sha256") == protocol_sha
        and sidecar.get("sample_id") == sample_id
        and sidecar.get("arm") == arm
        and sidecar.get("output_sha256") == file_sha256(output)
        and sidecar.get("crop_base_sha256") == file_sha256(crop_base)
        and sidecar.get("audio_sha256") == file_sha256(audio)
        and sidecar.get("mapping_sha256") == canonical_json_sha256([int(value) for value in mapping])
    )


def materialize(paths, protocol: Mapping[str, Any]) -> dict[str, Any]:
    if protocol.get("status") != "locked":
        raise DiagnosticError("protocol is not locked for media materialization")
    protocol_sha = str(protocol.get("_sha256", ""))
    if not protocol_sha:
        raise DiagnosticError("protocol hash is unavailable")
    paths.media.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    expected_pcm_by_id: dict[str, bytes] = {}
    for index, record in enumerate(protocol["records"], 1):
        sample_id = str(record["sample_id"])
        source_video = Path(str(record["face_video"]["path"]))
        source_audio = Path(str(record["natural_audio"]["path"]))
        frames = read_video_frames(source_video)
        crop = record["crop"]
        if len(frames) != int(crop["frame_count"]):
            raise DiagnosticError(f"source frame count changed: {sample_id}")
        proc_track = crop.get("processed_track")
        if not isinstance(proc_track, list):
            raise DiagnosticError(f"processed crop track is missing: {sample_id}")
        crop_frames = crop_frames_from_track(frames, proc_track)
        crop_base = paths.media / "crops" / f"{sample_id}.mkv"
        crop_sidecar = crop_base.with_suffix(".json")
        if crop_base.is_file() and crop_sidecar.is_file():
            crop_meta = verify_self_hashed_json(crop_sidecar)
            if crop_meta.get("protocol_sha256") != protocol_sha or crop_meta.get("sample_id") != sample_id or crop_meta.get("output_sha256") != file_sha256(crop_base):
                raise DiagnosticError(f"existing crop base identity changed: {sample_id}")
            actual_crop = decode_video_frames(crop_base)
            if len(actual_crop) != len(crop_frames) or any(not np.array_equal(left, right) for left, right in zip(actual_crop, crop_frames, strict=True)):
                raise DiagnosticError(f"existing crop base pixels changed: {sample_id}")
        elif crop_base.exists() or crop_sidecar.exists():
            raise DiagnosticError(f"partial crop base cannot be resumed: {sample_id}")
        else:
            encoded = encode_ffv1(crop_frames, crop_base, paths.media / "logs" / f"{sample_id}.crop.log")
            crop_meta = {
                "schema_version": 1,
                "protocol_id": config.PROTOCOL_ID,
                "protocol_sha256": protocol_sha,
                "sample_id": sample_id,
                "output": str(crop_base),
                **encoded,
            }
            write_self_hashed_json(crop_sidecar, crop_meta)
        expected_pcm = source_pcm16(source_audio)
        expected_pcm_by_id[sample_id] = expected_pcm
        mapping_path = paths.media / "mapping" / f"{sample_id}.json"
        mapping_payload = {
            "schema_version": 1,
            "protocol_id": config.PROTOCOL_ID,
            "protocol_sha256": protocol_sha,
            "sample_id": sample_id,
            "frame_count": len(crop_frames),
            "mappings": {arm: _expected_mapping(record, arm) for arm in config.ARMS},
        }
        if mapping_path.is_file():
            prior_mapping = verify_self_hashed_json(mapping_path)
            body = dict(prior_mapping)
            body.pop("artifact_sha256", None)
            if body != mapping_payload:
                raise DiagnosticError(f"existing mapping differs: {sample_id}")
        else:
            write_self_hashed_json(mapping_path, mapping_payload)
        arm_rows: dict[str, Any] = {}
        for arm in config.ARMS:
            mapping = _expected_mapping(record, arm)
            output = paths.media / arm / f"{sample_id}.mkv"
            sidecar = output.with_suffix(".json")
            if output.is_file() and sidecar.is_file():
                media_meta = verify_self_hashed_json(sidecar)
                if not _media_sidecar_matches(media_meta, protocol_sha, sample_id, arm, output, crop_base, source_audio, mapping):
                    raise DiagnosticError(f"existing media identity changed: {sample_id}/{arm}")
            elif output.exists() or sidecar.exists():
                raise DiagnosticError(f"partial media cell cannot be resumed: {sample_id}/{arm}")
            else:
                video_only = paths.media / "temporary" / arm / f"{sample_id}.mkv"
                encode_ffv1([crop_frames[int(source_index)] for source_index in mapping], video_only, paths.media / "logs" / f"{sample_id}.{arm}.encode.log")
                muxed = mux_pcm(video_only, source_audio, output, paths.media / "logs" / f"{sample_id}.{arm}.mux.log")
                video_only.unlink(missing_ok=True)
                media_meta = {
                    "schema_version": 2,
                    "protocol_id": config.PROTOCOL_ID,
                    "protocol_sha256": protocol_sha,
                    "sample_id": sample_id,
                    "source_group": str(record["source_group"]),
                    "arm": arm,
                    "output": str(output),
                    "output_sha256": muxed["output_sha256"],
                    "crop_base": str(crop_base),
                    "crop_base_sha256": file_sha256(crop_base),
                    "audio": str(source_audio),
                    "audio_sha256": file_sha256(source_audio),
                    "audio_pcm_sha256": bytes_sha256(expected_pcm),
                    "mapping": mapping,
                    "mapping_sha256": canonical_json_sha256(mapping),
                    "source_frame_sha256": [bytes_sha256(crop_frames[int(source_index)].tobytes()) for source_index in mapping],
                    "encode_command_sha256": muxed.get("command_sha256"),
                }
                verification = verify_muxed_media(output, crop_frames, expected_pcm, mapping)
                media_meta.update(verification)
                write_self_hashed_json(sidecar, media_meta)
            expected = verify_muxed_media(output, crop_frames, expected_pcm, mapping)
            arm_rows[arm] = {
                "output": str(output),
                "output_sha256": file_sha256(output),
                "sidecar": str(sidecar),
                "mapping": mapping,
                "mapping_sha256": canonical_json_sha256(mapping),
                "verification": expected,
            }
        rows.append(
            {
                "sample_id": sample_id,
                "source_group": str(record["source_group"]),
                "crop_base": str(crop_base),
                "crop_base_sha256": file_sha256(crop_base),
                "mapping": str(mapping_path),
                "arms": arm_rows,
            }
        )
        print(f"MEDIA {index}/{len(protocol['records'])} {sample_id}", flush=True)
    manifest = {
        "schema_version": 2,
        "stage_id": "media",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "protocol_sha256": protocol_sha,
        "status": "complete",
        "record_count": len(rows),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "cell_count": sum(len(row["arms"]) for row in rows),
        "expected_cell_count": config.expected_cell_count(),
        "arms": list(config.ARMS),
        "rows": rows,
        "audio_modified": False,
        "source_audio_byte_identity_verified": True,
    }
    write_self_hashed_json(paths.media / "manifest.json", manifest)
    return manifest
