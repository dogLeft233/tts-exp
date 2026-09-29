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


class ProtocolError(RuntimeError):
    """Raised when an auditable experiment contract is violated."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(bytes(value)).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sample_ids_sha256(sample_ids: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(data, encoding="utf-8")
    temporary.replace(path)
    return file_sha256(path)


def write_self_hashed_json(path: Path, payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = canonical_json_sha256(body)
    return write_json_atomic(path, body)


def verify_self_hashed_json(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    recorded = payload.get("artifact_sha256")
    if not isinstance(recorded, str):
        raise ProtocolError(f"self-hash missing: {path}")
    body = dict(payload)
    body.pop("artifact_sha256", None)
    if recorded != canonical_json_sha256(body):
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


def assert_not_sealed(path: str | Path) -> None:
    lowered = str(Path(path).resolve()).lower()
    if any(token in lowered for token in config.NO_SEALED_MEDIA_TOKENS):
        raise ProtocolError(f"sealed media path crossed: {path}")


def read_pcm16(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        count = handle.getnframes()
        payload = handle.readframes(count)
    if channels != config.PCM_CHANNELS or width != config.PCM_SAMPLE_WIDTH or rate != config.SAMPLE_RATE:
        raise ProtocolError(f"audio format is not 16 kHz mono PCM16: {path}")
    values = np.frombuffer(payload, dtype="<i2").copy()
    if values.size != count or values.size < config.MIN_AUDIO_SAMPLES:
        raise ProtocolError(f"audio sample count is invalid: {path}")
    return values, {
        "sample_count": int(values.size),
        "sample_rate": int(rate),
        "channels": int(channels),
        "sample_width": int(width),
        "container_sha256": file_sha256(path),
        "pcm_sha256": bytes_sha256(values.astype("<i2", copy=False).tobytes()),
    }


def write_pcm16(path: Path, values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    if values.dtype != np.int16 or values.ndim != 1 or values.size < config.MIN_AUDIO_SAMPLES:
        raise ProtocolError("PCM16 output must be a one-dimensional int16 array")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(config.PCM_CHANNELS)
        handle.setsampwidth(config.PCM_SAMPLE_WIDTH)
        handle.setframerate(config.SAMPLE_RATE)
        handle.writeframes(values.astype("<i2", copy=False).tobytes())
    temporary.replace(path)
    pcm = values.astype("<i2", copy=False).tobytes()
    return {
        "path": str(path.resolve()),
        "container_sha256": file_sha256(path),
        "pcm_sha256": bytes_sha256(pcm),
        "sample_count": int(values.size),
        "sample_rate": config.SAMPLE_RATE,
        "channels": config.PCM_CHANNELS,
        "sample_width": config.PCM_SAMPLE_WIDTH,
    }


def run_logged(command: list[str], cwd: Path, log_path: Path, *, env: Mapping[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(cwd), env=dict(env or os.environ), stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise ProtocolError(f"command failed with exit {result.returncode}: {log_path}")


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ProtocolError(f"{label} is missing: {path}")


def require_hash(path: Path, expected: str, label: str) -> None:
    require_file(path, label)
    actual = file_sha256(path)
    if actual != expected:
        raise ProtocolError(f"{label} hash changed: expected {expected}, got {actual}")


def npy_sha256(path: Path) -> str:
    return file_sha256(path)
