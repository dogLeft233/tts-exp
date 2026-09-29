from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from scripts.experiments.wav2lip_roi_retiming_oracle.common import (
    bytes_sha256,
    canonical_json_sha256,
    extract_bgr24_frames,
    extract_pcm_from_media,
    file_sha256,
    read_pcm16_wav,
    run_command,
    verify_self_hashed_json,
    write_self_hashed_json,
)


class ExperimentError(RuntimeError):
    """Frozen input, construction, media, scoring, or validation failure."""


def json_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ExperimentError(f"expected JSON object: {path}")
    return payload


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise ExperimentError(f"missing {label}: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ExperimentError(f"{label} hash changed: expected {expected}, actual {actual}: {path}")
    return actual


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ExperimentError(f"missing {label}: {path}")
    return path


def clone_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False))


def free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


__all__ = [
    "ExperimentError",
    "bytes_sha256",
    "canonical_json_sha256",
    "clone_json",
    "extract_bgr24_frames",
    "extract_pcm_from_media",
    "file_sha256",
    "free_bytes",
    "json_payload",
    "read_pcm16_wav",
    "require_file",
    "require_hash",
    "run_command",
    "verify_self_hashed_json",
    "write_self_hashed_json",
]
