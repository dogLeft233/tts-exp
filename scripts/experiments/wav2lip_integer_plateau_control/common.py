from __future__ import annotations

import json
import subprocess
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_roi_retiming_oracle.common import (
    bytes_sha256,
    canonical_json_sha256,
    extract_pcm_from_media,
    file_sha256,
    read_pcm16_wav,
    verify_self_hashed_json,
    write_self_hashed_json,
)


class PlateauError(RuntimeError):
    """Frozen-input, media, scoring, or validation failure."""


def run_command(command: list[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(command, input=input_bytes, capture_output=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or b"").decode("utf-8", errors="replace")[-3000:]
        raise PlateauError(f"command failed ({result.returncode}): {' '.join(command)}\n{detail}")
    return result


def write_pcm16_wav(path: Path, pcm: bytes, sample_rate: int = 16_000) -> dict[str, Any]:
    if not pcm or len(pcm) % 2:
        raise PlateauError("PCM16 payload is empty or unaligned")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PlateauError(f"refusing to overwrite audio artifact: {path}")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    observed, values, params = read_pcm16_wav(path)
    if observed != pcm:
        raise PlateauError(f"PCM changed during WAV write: {path}")
    return {
        "path": str(path.resolve()),
        "container_sha256": file_sha256(path),
        "decoded_pcm_sha256": bytes_sha256(observed),
        "sample_count": int(values.size),
        "sample_rate": int(params["sample_rate"]),
        "channels": int(params["channels"]),
        "sample_width": int(params["sample_width"]),
    }


def json_payload(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlateauError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PlateauError(f"expected JSON object: {path}")
    return value


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise PlateauError(f"missing {label}: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise PlateauError(f"{label} hash changed: expected {expected}, actual {actual}: {path}")
    return actual


def copy_identity(source: Path, destination: Path, expected_pcm: bytes) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise PlateauError(f"refusing to overwrite copied audio: {destination}")
    destination.write_bytes(source.read_bytes())
    observed, _values, _params = read_pcm16_wav(destination)
    if observed != expected_pcm:
        raise PlateauError(f"copied PCM changed: {destination}")
    return {
        "path": str(destination.resolve()),
        "container_sha256": file_sha256(destination),
        "decoded_pcm_sha256": bytes_sha256(observed),
        "sample_count": len(observed) // 2,
        "construction": "byte-identical copy of frozen natural PCM16",
    }


def np_hash(values: np.ndarray, dtype: str | None = None) -> str:
    array = np.asarray(values, dtype=dtype) if dtype else np.asarray(values)
    return bytes_sha256(np.ascontiguousarray(array).tobytes())


def safe_resolve(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve()


def self_hash_body(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(dict(payload))


__all__ = [
    "PlateauError",
    "bytes_sha256",
    "canonical_json_sha256",
    "copy_identity",
    "extract_pcm_from_media",
    "file_sha256",
    "json_payload",
    "np_hash",
    "read_pcm16_wav",
    "require_hash",
    "run_command",
    "safe_resolve",
    "verify_self_hashed_json",
    "write_pcm16_wav",
    "write_self_hashed_json",
]
