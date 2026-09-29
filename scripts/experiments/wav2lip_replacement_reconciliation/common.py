from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import subprocess
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config


class ReconciliationError(RuntimeError):
    """A frozen input, cache, parity, or validation contract failed."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ReconciliationError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)
    return file_sha256(path)


def write_self_hashed_json(path: Path, payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = canonical_json_sha256(body)
    return write_json(path, body)


def verify_self_hashed_json(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    recorded = payload.get("artifact_sha256")
    if not isinstance(recorded, str):
        raise ReconciliationError(f"self-hash missing: {path}")
    body = dict(payload)
    body.pop("artifact_sha256", None)
    if recorded != canonical_json_sha256(body):
        raise ReconciliationError(f"self-hash mismatch: {path}")
    return payload


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ReconciliationError(f"missing {label}: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ReconciliationError(f"{label} hash changed: expected {expected}, actual {actual}: {path}")
    return actual


def require_asset(path: Path, expected: str, label: str) -> dict[str, str]:
    actual = require_hash(path, expected, label)
    lowered = str(path.resolve()).lower()
    if any(token in lowered for token in config.NO_SEALED_MEDIA_TOKENS):
        raise ReconciliationError(f"sealed path crossed: {path}")
    return {"path": str(path.resolve()), "sha256": actual, "label": label}


def read_pcm16_wav(path: Path) -> tuple[bytes, np.ndarray, dict[str, int]]:
    try:
        with wave.open(str(path), "rb") as handle:
            params = {
                "channels": handle.getnchannels(),
                "sample_width": handle.getsampwidth(),
                "sample_rate": handle.getframerate(),
                "frame_count": handle.getnframes(),
            }
            raw = handle.readframes(handle.getnframes())
    except (OSError, wave.Error) as exc:
        raise ReconciliationError(f"cannot decode WAV: {path}: {exc}") from exc
    if params != {**params, "frame_count": params["frame_count"]}:
        raise AssertionError("unreachable")
    if params["channels"] != 1 or params["sample_width"] != 2 or params["sample_rate"] != 16_000:
        raise ReconciliationError(f"WAV format is not mono/16k/PCM16: {path}: {params}")
    values = np.frombuffer(raw, dtype="<i2")
    if values.size != params["frame_count"]:
        raise ReconciliationError(f"WAV byte/frame count differs: {path}")
    return raw, values, params


def probe_media(path: Path) -> dict[str, Any]:
    command = [
        "/home/wjj/miniconda3/bin/ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ReconciliationError(f"ffprobe failed: {path}: {exc}") from exc
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    if not isinstance(video, Mapping):
        raise ReconciliationError(f"no video stream: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "video": {
            "codec_name": video.get("codec_name"),
            "width": video.get("width"),
            "height": video.get("height"),
            "r_frame_rate": video.get("r_frame_rate"),
            "time_base": video.get("time_base"),
            "nb_frames": video.get("nb_frames"),
        },
        "audio": next((item for item in streams if item.get("codec_type") == "audio"), None),
    }


def load_matrix(path: Path, expected_sha256: str, label: str) -> np.ndarray:
    require_hash(path, expected_sha256, label)
    try:
        if path.suffix.lower() in {".pckl", ".pickle"}:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, list) or len(payload) != 1:
                raise ValueError("legacy cache must contain exactly one matrix")
            matrix = np.asarray(payload[0])
        else:
            matrix = np.load(path, allow_pickle=False)
    except (OSError, EOFError, pickle.PickleError, ValueError) as exc:
        raise ReconciliationError(f"cannot load matrix: {path}: {exc}") from exc
    if matrix.ndim != 2 or matrix.shape[1] != config.MATRIX_COLUMNS or matrix.shape[0] < 1 or not np.isfinite(matrix).all():
        raise ReconciliationError(f"malformed distance matrix: {path}: {matrix.shape}")
    return np.asarray(matrix)


def assert_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ReconciliationError("non-finite output")
    if isinstance(value, Mapping):
        for item in value.values():
            assert_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_finite(item)
