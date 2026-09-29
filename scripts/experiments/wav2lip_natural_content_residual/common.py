from __future__ import annotations

import hashlib
import json
import math
import os
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ProtocolError(RuntimeError):
    """A frozen-protocol or artifact error that must stop the run."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return bytes_sha256(canonical_json_bytes(value))


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return file_sha256(path)


def write_self_hashed_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = canonical_json_sha256(body)
    write_json_atomic(path, body)
    return body


def verify_self_hashed_json(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    actual = payload.get("artifact_sha256")
    body = dict(payload)
    body.pop("artifact_sha256", None)
    if not isinstance(actual, str) or actual != canonical_json_sha256(body):
        raise ProtocolError(f"self-hash mismatch: {path}")
    return payload


def write_or_verify(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if path.is_file():
        old = verify_self_hashed_json(path)
        old_body = dict(old)
        old_body.pop("artifact_sha256", None)
        if old_body != dict(payload):
            raise ProtocolError(f"existing artifact differs: {path}")
        return old
    if path.exists():
        raise ProtocolError(f"artifact is not a file: {path}")
    return write_self_hashed_json(path, payload)


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
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2 or handle.getframerate() != 16_000:
            raise ProtocolError(f"audio is not 16kHz mono PCM16: {path}")
        return handle.readframes(handle.getnframes())


def write_delayed_pcm16(source: Path, output: Path, delay_samples: int) -> dict[str, Any]:
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        if params.nchannels != 1 or params.sampwidth != 2 or params.framerate != 16_000:
            raise ProtocolError(f"delay source is not 16kHz mono PCM16: {source}")
        raw = reader.readframes(reader.getnframes())
    step = int(delay_samples) * 2
    if step <= 0 or len(raw) <= step:
        raise ProtocolError("delay must be positive and shorter than the audio")
    delayed = b"\0" * step + raw[:-step]
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(delayed)
    return {"path": str(output.resolve()), "sha256": file_sha256(output), "pcm_sha256": bytes_sha256(delayed), "delay_samples": int(delay_samples), "sample_count": len(delayed) // 2}
