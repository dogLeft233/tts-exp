from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import config
from .audio import write_pcm16
from .common import DiagnosticError, bytes_sha256, canonical_json_sha256, file_sha256


def ffprobe_streams(path: Path) -> list[dict[str, Any]]:
    result = subprocess.run([str(config.FFPROBE), "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, check=True)
    payload = json.loads(result.stdout.decode("utf-8"))
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise DiagnosticError(f"ffprobe returned no stream list: {path}")
    return streams


def probe_video(path: Path) -> dict[str, Any]:
    streams = ffprobe_streams(path)
    videos = [row for row in streams if row.get("codec_type") == "video"]
    if len(videos) != 1:
        raise DiagnosticError(f"expected one video stream: {path}")
    stream = videos[0]
    try:
        rate = float(Fraction(str(stream.get("r_frame_rate"))))
    except (ValueError, ZeroDivisionError):
        raise DiagnosticError(f"invalid frame rate: {path}") from None
    frame_count = int(stream.get("nb_frames", stream.get("nb_read_frames", -1)))
    start = float(stream.get("start_time", 0.0))
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "codec_name": str(stream.get("codec_name", "")),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "frame_rate": rate,
        "frame_count": frame_count,
        "start_time": start,
        "time_base": str(stream.get("time_base", "")),
    }


def validate_zero_based_pts(path: Path) -> dict[str, Any]:
    """Verify that the source video has a zero-based, contiguous 25 fps PTS axis."""
    result = subprocess.run(
        [
            str(config.FFPROBE),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout.decode("utf-8"))
    timestamps = [float(row["best_effort_timestamp_time"]) for row in payload.get("frames", []) if row.get("best_effort_timestamp_time") is not None]
    if not timestamps:
        raise DiagnosticError(f"source video has no readable frame timestamps: {path}")
    expected_step = 1.0 / config.FPS
    if abs(timestamps[0]) > 1e-6:
        raise DiagnosticError(f"source video PTS does not start at zero: {path} ({timestamps[0]:.9f})")
    if any(abs((right - left) - expected_step) > 1e-5 for left, right in zip(timestamps, timestamps[1:], strict=False)):
        raise DiagnosticError(f"source video PTS is not contiguous at {config.FPS} fps: {path}")
    return {
        "frame_count": len(timestamps),
        "first_pts": timestamps[0],
        "last_pts": timestamps[-1],
        "expected_step": expected_step,
        "zero_based": True,
        "contiguous": True,
    }


def read_video_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise DiagnosticError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise DiagnosticError(f"invalid decoded frame: {path}")
            frames.append(np.ascontiguousarray(frame))
    finally:
        capture.release()
    if not frames:
        raise DiagnosticError(f"video has no frames: {path}")
    return frames


def decode_video_frames(path: Path) -> list[np.ndarray]:
    info = probe_video(path)
    width, height = int(info["width"]), int(info["height"])
    raw = subprocess.run([str(config.FFMPEG), "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], capture_output=True, check=True).stdout
    stride = width * height * 3
    if width <= 0 or height <= 0 or len(raw) % stride:
        raise DiagnosticError(f"decoded frame bytes are malformed: {path}")
    frames = [np.frombuffer(raw[start : start + stride], dtype=np.uint8).reshape(height, width, 3).copy() for start in range(0, len(raw), stride)]
    if info["frame_count"] > 0 and len(frames) != int(info["frame_count"]):
        raise DiagnosticError(f"decoded frame count differs from ffprobe: {path}")
    return frames


def decode_pcm16(path: Path) -> bytes:
    return bytes(subprocess.run([str(config.FFMPEG), "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(config.SAMPLE_RATE), "-ac", "1", "pipe:1"], capture_output=True, check=True).stdout)


def video_bitstream(path: Path, codec_name: str | None = None) -> bytes:
    codec = codec_name or probe_video(path)["codec_name"]
    formats = {"h264": "h264", "hevc": "hevc", "mpeg4": "m4v", "vp8": "ivf", "vp9": "ivf", "ffv1": "matroska"}
    output_format = formats.get(codec)
    if output_format is None:
        raise DiagnosticError(f"unsupported video codec for elementary stream check: {codec}")
    return bytes(subprocess.run([str(config.FFMPEG), "-v", "error", "-i", str(path), "-map", "0:v:0", "-c:v", "copy", "-f", output_format, "pipe:1"], capture_output=True, check=True).stdout)


def strict_mux(video: Path, audio: Path, output: Path, log_path: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    command = [str(config.FFMPEG), "-y", "-v", "error", "-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "pcm_s16le", "-ar", str(config.SAMPLE_RATE), "-ac", "1", "-f", "matroska", str(temporary)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        raise DiagnosticError(f"strict mux failed: {output}")
    expected_pcm = decode_pcm16(audio)
    actual_pcm = decode_pcm16(temporary)
    if expected_pcm != actual_pcm:
        temporary.unlink(missing_ok=True)
        raise DiagnosticError(f"strict mux changed PCM: {output}")
    source_codec = str(probe_video(video)["codec_name"])
    source_bits = video_bitstream(video, source_codec)
    mux_bits = video_bitstream(temporary, source_codec)
    if source_bits != mux_bits:
        temporary.unlink(missing_ok=True)
        raise DiagnosticError(f"strict mux changed video elementary stream: {output}")
    temporary.replace(output)
    info = probe_video(output)
    return {
        "path": str(output.resolve()),
        "sha256": file_sha256(output),
        "command": command,
        "command_sha256": canonical_json_sha256(command),
        "source_video": str(video.resolve()),
        "source_video_sha256": file_sha256(video),
        "source_audio": str(audio.resolve()),
        "source_audio_sha256": file_sha256(audio),
        "audio_pcm_sha256": bytes_sha256(actual_pcm),
        "audio_pcm_verified": True,
        "video_elementary_sha256": bytes_sha256(mux_bits),
        "video_stream_copy_verified": True,
        "video": info,
        "audio_sample_count": len(actual_pcm) // 2,
    }


def mux_pcm(video: Path, audio: Path, output: Path) -> dict[str, Any]:
    """Mux a frozen FFV1 crop and WAV without changing either decoded stream."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    command = [str(config.FFMPEG), "-y", "-v", "error", "-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "pcm_s16le", "-ar", str(config.SAMPLE_RATE), "-ac", "1", "-f", "matroska", str(temporary)]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        raise DiagnosticError(f"PCM mux failed: {output}: {result.stderr.decode(errors='replace')}")
    expected_pcm = decode_pcm16(audio)
    actual_pcm = decode_pcm16(temporary)
    if expected_pcm != actual_pcm:
        temporary.unlink(missing_ok=True)
        raise DiagnosticError(f"PCM mux changed decoded audio: {output}")
    codec_name = str(probe_video(video)["codec_name"])
    if codec_name == "ffv1":
        source_bits = actual_bits = b""
    else:
        source_bits = video_bitstream(video, codec_name)
        actual_bits = video_bitstream(temporary, codec_name)
        if source_bits != actual_bits:
            temporary.unlink(missing_ok=True)
            raise DiagnosticError(f"PCM mux changed video stream: {output}")
    temporary.replace(output)
    return {
        "path": str(output.resolve()),
        "sha256": file_sha256(output),
        "command": command,
        "command_sha256": canonical_json_sha256(command),
        "video_source_sha256": file_sha256(video),
        "audio_source_sha256": file_sha256(audio),
        "video_elementary_sha256": bytes_sha256(actual_bits),
        "audio_pcm_sha256": bytes_sha256(actual_pcm),
        "video_stream_copy_verified": True,
        "audio_pcm_verified": True,
    }


def _raw_frame_bytes(frames: Sequence[np.ndarray]) -> bytes:
    if len(frames) == 0:
        raise DiagnosticError("cannot encode empty frame sequence")
    shape = np.asarray(frames[0]).shape
    if len(shape) != 3 or shape[2] != 3:
        raise DiagnosticError("frames must be BGR images")
    chunks: list[bytes] = []
    for frame in frames:
        value = np.asarray(frame)
        if value.shape != shape or value.dtype != np.uint8:
            raise DiagnosticError("frame shape or dtype changed")
        chunks.append(np.ascontiguousarray(value).tobytes())
    return b"".join(chunks)


def encode_ffv1(frames: Sequence[np.ndarray], output: Path) -> dict[str, Any]:
    height, width = np.asarray(frames[0]).shape[:2]
    raw = _raw_frame_bytes(frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{output.suffix.lstrip('.')}.partial{output.suffix}")
    command = [str(config.FFMPEG), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(config.FPS), "-i", "pipe:0", "-an", "-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "bgr0", "-f", "matroska", str(temporary)]
    result = subprocess.run(command, input=raw, capture_output=True, check=False)
    if result.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        raise DiagnosticError(f"FFV1 encoding failed: {output}: {result.stderr.decode(errors='replace')}")
    temporary.replace(output)
    decoded = decode_video_frames(output)
    if len(decoded) != len(frames) or any(not np.array_equal(left, right) for left, right in zip(decoded, frames, strict=True)):
        raise DiagnosticError(f"FFV1 round-trip changed frame data: {output}")
    return {
        "path": str(output.resolve()),
        "sha256": file_sha256(output),
        "frame_count": len(frames),
        "shape": [height, width, 3],
        "raw_frame_sha256": bytes_sha256(raw),
        "command": command,
        "command_sha256": canonical_json_sha256(command),
        "frame_hashes": [bytes_sha256(np.ascontiguousarray(frame).tobytes()) for frame in frames],
    }


def verify_media(path: Path, expected_frames: Sequence[np.ndarray], expected_pcm: bytes) -> dict[str, Any]:
    actual_frames = decode_video_frames(path)
    actual_pcm = decode_pcm16(path)
    if len(actual_frames) != len(expected_frames) or any(not np.array_equal(left, right) for left, right in zip(actual_frames, expected_frames, strict=True)):
        raise DiagnosticError(f"media video frames differ from frozen frames: {path}")
    if actual_pcm != expected_pcm:
        raise DiagnosticError(f"media PCM differs from frozen samples: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "frame_count": len(actual_frames),
        "frame_hashes": [bytes_sha256(np.ascontiguousarray(frame).tobytes()) for frame in actual_frames],
        "pcm_sha256": bytes_sha256(actual_pcm),
        "pcm_sample_count": len(actual_pcm) // 2,
    }


def lock_source_crop(source_video: Path, detector: Any | None = None) -> tuple[dict[str, Any], list[np.ndarray], dict[str, Any]]:
    from scripts.experiments.lrs3_real_video_local_timing import media as real_media

    info = probe_video(source_video)
    if abs(float(info["frame_rate"]) - config.FPS) > 1e-6 or abs(float(info["start_time"])) > 0.001:
        raise DiagnosticError(f"source video does not provide a zero-based 25fps timeline: {source_video}")
    pts_validation = validate_zero_based_pts(source_video)
    frames = read_video_frames(source_video)
    if int(info["frame_count"]) > 0 and len(frames) != int(info["frame_count"]):
        raise DiagnosticError(f"source video frame count changed: {source_video}")
    if detector is None:
        detector = real_media.make_face_detector()
    track = real_media.detect_full_track(frames, detector)
    processed = real_media.processed_crop_track(track["raw_bbox"], len(frames))
    crops = real_media.crop_frames_from_track(frames, processed)
    if len(crops) != len(frames):
        raise DiagnosticError("locked crop does not cover every source frame")
    return track, crops, {
        "source_video": info,
        "pts_validation": pts_validation,
        "frame_count": len(frames),
        "crop_count": len(crops),
        "crop_size": [int(crops[0].shape[0]), int(crops[0].shape[1])] if crops else [],
        "track_selection": "exactly_one_full_track_from_official_S3FD_before_scoring",
        "raw_bbox_sha256": canonical_json_sha256(track["raw_bbox"]),
        "processed_track_sha256": canonical_json_sha256(processed),
        "crop_frame_hashes": [bytes_sha256(np.ascontiguousarray(frame).tobytes()) for frame in crops],
    }
