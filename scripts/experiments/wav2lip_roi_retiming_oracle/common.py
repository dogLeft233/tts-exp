from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config


class OracleError(RuntimeError):
    """A frozen-input, media, scoring, or validation error."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sample_ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise OracleError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise OracleError(f"expected JSON object: {path}")
    return payload


def verify_self_hashed_json(path: Path, expected_file_hash: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise OracleError(f"missing JSON: {path}")
    actual = file_sha256(path)
    if expected_file_hash is not None and actual != expected_file_hash:
        raise OracleError(f"fixed hash mismatch: {path}: expected {expected_file_hash}, actual {actual}")
    payload = read_json(path)
    recorded = payload.get("artifact_sha256")
    body = dict(payload)
    body.pop("artifact_sha256", None)
    if not isinstance(recorded, str) or recorded != canonical_json_sha256(body):
        raise OracleError(f"self-hash mismatch: {path}")
    return payload


def write_self_hashed_json(path: Path, payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = canonical_json_sha256(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return file_sha256(path)


def resolve_path(value: Any, repo: Path = config.REPO) -> Path:
    path = Path(str(value))
    return (repo / path).resolve() if not path.is_absolute() else path.resolve()


def run_command(command: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(command, input=input_bytes, capture_output=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or b"").decode("utf-8", errors="replace")[-2000:]
        raise OracleError(f"command failed ({result.returncode}): {' '.join(command)}\n{detail}")
    return result


def read_pcm16_wav(path: Path) -> tuple[bytes, np.ndarray, dict[str, Any]]:
    try:
        with wave.open(str(path), "rb") as handle:
            params = {
                "channels": handle.getnchannels(),
                "sample_width": handle.getsampwidth(),
                "sample_rate": handle.getframerate(),
                "frame_count": handle.getnframes(),
            }
            pcm = handle.readframes(handle.getnframes())
    except (OSError, wave.Error) as exc:
        raise OracleError(f"cannot read PCM16 WAV: {path}") from exc
    expected = {"channels": 1, "sample_width": 2, "sample_rate": config.SAMPLE_RATE}
    if any(params[key] != value for key, value in expected.items()):
        raise OracleError(f"WAV is not mono PCM16/16k: {path}: {params}")
    audio = np.frombuffer(pcm, dtype="<i2").copy()
    if audio.size != params["frame_count"] or audio.size == 0:
        raise OracleError(f"WAV frame count is inconsistent: {path}")
    return pcm, audio, params


def write_pcm16_wav(path: Path, pcm: bytes) -> dict[str, Any]:
    if len(pcm) == 0 or len(pcm) % 2:
        raise OracleError("PCM16 payload is empty or unaligned")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(config.SAMPLE_RATE)
        handle.writeframes(pcm)
    _pcm, audio, params = read_pcm16_wav(path)
    return {
        "path": str(path.resolve()),
        "container_sha256": file_sha256(path),
        "decoded_pcm_sha256": bytes_sha256(pcm),
        "sample_count": int(audio.size),
        "sample_rate": int(params["sample_rate"]),
        "channels": int(params["channels"]),
        "sample_width": int(params["sample_width"]),
    }


def extract_pcm_from_media(path: Path) -> bytes:
    command = [
        str(config.FFMPEG),
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
    return run_command(command).stdout


def probe_video(path: Path, *, include_frames: bool = True) -> dict[str, Any]:
    entries = "codec_name,pix_fmt,color_range,color_space,color_primaries,color_trc,width,height,r_frame_rate,time_base,start_pts,nb_frames"
    command = [
        str(config.FFPROBE),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        f"stream={entries}",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(run_command(command).stdout.decode("utf-8"))
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise OracleError(f"video stream is missing: {path}")
    result: dict[str, Any] = {"stream": dict(streams[0])}
    if include_frames:
        frame_command = [
            str(config.FFPROBE),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp,best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        ]
        frame_payload = json.loads(run_command(frame_command).stdout.decode("utf-8"))
        frames = frame_payload.get("frames")
        if not isinstance(frames, list):
            raise OracleError(f"video frame timestamps are missing: {path}")
        result["frames"] = frames
    return result


def extract_bgr24_frames(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    probe = probe_video(path)
    stream = probe["stream"]
    width = int(stream.get("width", -1))
    height = int(stream.get("height", -1))
    if (width, height) != (config.FRAME_WIDTH, config.FRAME_HEIGHT):
        raise OracleError(f"unexpected video dimensions: {path}: {width}x{height}")
    raw_command = [
        str(config.FFMPEG),
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
    raw = run_command(raw_command).stdout
    frame_size = width * height * 3
    if len(raw) % frame_size:
        raise OracleError(f"raw BGR24 size is not frame aligned: {path}")
    frames = np.frombuffer(raw, dtype=np.uint8).reshape((-1, height, width, 3)).copy()
    timestamps = probe.get("frames", [])
    if len(timestamps) != frames.shape[0]:
        raise OracleError(f"frame timestamp count differs from decoded frames: {path}")
    return frames, {
        "probe": probe,
        "frame_count": int(frames.shape[0]),
        "raw_bgr24_sha256": bytes_sha256(raw),
        "frame_sha256": [bytes_sha256(frame.tobytes()) for frame in frames],
    }


def compare_values(expected: Any, actual: Any, path: str, differences: list[dict[str, Any]], tolerance: float) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            differences.append({"path": path, "expected": expected, "actual": actual})
            return
        for key, value in expected.items():
            if key not in actual:
                differences.append({"path": f"{path}.{key}", "expected": value, "actual": "<missing>"})
            else:
                compare_values(value, actual[key], f"{path}.{key}", differences, tolerance)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            differences.append({"path": path, "expected": expected, "actual": actual})
            return
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare_values(left, right, f"{path}[{index}]", differences, tolerance)
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            differences.append({"path": path, "expected": expected, "actual": actual})
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isfinite(float(expected)) or not math.isfinite(float(actual)) or abs(float(expected) - float(actual)) > tolerance:
            differences.append({"path": path, "expected": expected, "actual": actual})
        return
    if expected != actual:
        differences.append({"path": path, "expected": expected, "actual": actual})
