from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class DiagnosticError(RuntimeError):
    """A protocol, input, or artifact error that must be preserved in the run."""


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
    return bytes_sha256(canonical_json_bytes(payload))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))
    if not isinstance(value, dict):
        raise DiagnosticError(f"JSON root is not an object: {path}")
    return value


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
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
        raise DiagnosticError(f"self-hash is missing: {path}")
    body = dict(payload)
    body.pop("artifact_sha256", None)
    if recorded != canonical_json_sha256(body):
        raise DiagnosticError(f"self-hash mismatch: {path}")
    return payload


def assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise DiagnosticError(f"non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise DiagnosticError(f"{label} is missing: {path}")
    return path


def sample_ids_sha256(sample_ids: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def git_commit(repo: Path) -> str:
    import subprocess

    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"
