from __future__ import annotations

import hashlib
import json
import math
import subprocess
import wave
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import config


class ProtocolError(RuntimeError):
    """A frozen-input, provenance, or diagnostic contract failure."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{__import__('os').getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return file_sha256(path)


def write_self_hashed_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = canonical_json_sha256(body)
    write_json_atomic(path, body)
    return body


def verify_self_hashed_json(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    recorded = payload.get("artifact_sha256")
    body = dict(payload)
    body.pop("artifact_sha256", None)
    if not isinstance(recorded, str) or recorded != canonical_json_sha256(body):
        raise ProtocolError(f"self-hash mismatch: {path}")
    return payload


def assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ProtocolError(f"non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")


def source_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != config.SAMPLE_RATE:
            raise ProtocolError(f"audio is not 16kHz mono PCM16: {path}")
        return handle.readframes(handle.getnframes())


def _run(command: list[str]) -> bytes:
    try:
        result = subprocess.run(command, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProtocolError(f"command failed: {' '.join(command)}") from exc
    return bytes(result.stdout)


def _probe_media(path: Path) -> dict[str, Any]:
    output = _run(
        [
            "/home/wjj/miniconda3/bin/ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type,codec_name,width,height,r_frame_rate,time_base,start_time,nb_read_frames",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        ]
    )
    try:
        payload = json.loads(output.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"ffprobe output is malformed: {path}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"ffprobe output is not an object: {path}")
    return payload


def _probe_streams(path: Path) -> list[dict[str, Any]]:
    output = _run(["/home/wjj/miniconda3/bin/ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)])
    try:
        streams = json.loads(output.decode("utf-8")).get("streams")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"ffprobe stream output is malformed: {path}") from exc
    if not isinstance(streams, list) or any(not isinstance(item, dict) for item in streams):
        raise ProtocolError(f"ffprobe returned malformed streams: {path}")
    return streams


def decode_video_frames(path: Path, width: int, height: int) -> list[np.ndarray]:
    raw = _run(
        [
            "/home/wjj/miniconda3/bin/ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
    )
    stride = int(width) * int(height) * 3
    if stride <= 0 or len(raw) % stride:
        raise ProtocolError(f"decoded video bytes are malformed: {path}")
    return [np.frombuffer(raw[start : start + stride], dtype=np.uint8).reshape(height, width, 3).copy() for start in range(0, len(raw), stride)]


def decode_pcm16(path: Path) -> bytes:
    return _run(
        [
            "/home/wjj/miniconda3/bin/ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(config.SAMPLE_RATE),
            "-ac",
            "1",
            "pipe:1",
        ]
    )


def inspect_media(path: Path, expected_pcm: bytes) -> tuple[dict[str, Any], list[np.ndarray]]:
    payload = _probe_media(path)
    streams = _probe_streams(path)
    videos = [item for item in streams if item.get("codec_type") == "video"]
    audios = [item for item in streams if item.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise ProtocolError(f"media must have one video and one audio stream: {path}")
    video = videos[0]
    audio = audios[0]
    try:
        frame_rate = float(Fraction(str(video.get("r_frame_rate", ""))))
        start_time = float(video.get("start_time", "0"))
        audio_start_time = float(audio.get("start_time", "0"))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ProtocolError(f"media stream timing is malformed: {path}") from exc
    width = int(video.get("width", 0))
    height = int(video.get("height", 0))
    frames = decode_video_frames(path, width, height)
    timestamps = [float(item.get("best_effort_timestamp_time", "nan")) for item in payload.get("frames", [])]
    if (
        str(video.get("codec_name")) != "ffv1"
        or width != config.FRAME_WIDTH
        or height != config.FRAME_HEIGHT
        or abs(frame_rate - config.FPS) > 1e-6
        or abs(start_time) > 1e-6
        or abs(audio_start_time) > 1e-6
        or len(frames) != config.FRAME_COUNT
        or len(timestamps) != config.FRAME_COUNT
        or any(abs(value - index / config.FPS) > 1e-6 for index, value in enumerate(timestamps))
        or str(audio.get("codec_name")) != "pcm_s16le"
        or int(audio.get("sample_rate", 0)) != config.SAMPLE_RATE
        or int(audio.get("channels", 0)) != 1
    ):
        raise ProtocolError(f"media timeline/stream contract failed: {path}")
    actual_pcm = decode_pcm16(path)
    if actual_pcm != expected_pcm:
        raise ProtocolError(f"media PCM differs from frozen source: {path}")
    frame_bytes = b"".join(frame.tobytes() for frame in frames)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "frame_count": len(frames),
        "frame_shape": [config.FRAME_HEIGHT, config.FRAME_WIDTH, 3],
        "frame_sha256": bytes_sha256(frame_bytes),
        "pts_start": timestamps[0],
        "pts_end": timestamps[-1],
        "pts_step": 1.0 / config.FPS,
        "audio_pcm_sha256": bytes_sha256(actual_pcm),
        "stream_contract": "ffv1/224x224/25fps + mono pcm_s16le/16kHz/start0",
    }, frames


def verify_media_pair(n_path: Path, d_path: Path, n_pcm: bytes, d_pcm: bytes) -> dict[str, Any]:
    n_evidence, n_frames = inspect_media(n_path, n_pcm)
    d_evidence, d_frames = inspect_media(d_path, d_pcm)
    if len(n_frames) != len(d_frames) or any(not np.array_equal(left, right) for left, right in zip(n_frames, d_frames, strict=True)):
        raise ProtocolError("natural and delayed control video pixels differ")
    n_evidence["paired_video_equal"] = True
    d_evidence["paired_video_equal"] = True
    return {"N": n_evidence, "A_DELAY": d_evidence, "video_pixels_equal": True}
