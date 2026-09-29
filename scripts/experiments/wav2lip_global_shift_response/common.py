from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ExperimentError(RuntimeError):
    """A frozen protocol or artifact failure that must stop the run."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return bytes_sha256(canonical_bytes(value))


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ExperimentError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ExperimentError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return file_sha256(path)


def write_self_hashed_json(path: Path, payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = canonical_sha256(body)
    return write_json(path, body)


def verify_self_hashed_json(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    recorded = payload.get("artifact_sha256")
    body = dict(payload)
    body.pop("artifact_sha256", None)
    if not isinstance(recorded, str) or recorded != canonical_sha256(body):
        raise ExperimentError(f"self-hash mismatch: {path}")
    return payload


def assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ExperimentError(f"non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")


def require_hash(path: Path, expected: str, label: str) -> Path:
    if not path.is_file():
        raise ExperimentError(f"missing {label}: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ExperimentError(f"{label} hash changed: expected {expected}, actual {actual}: {path}")
    return path


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ExperimentError(f"missing {label}: {path}")
    return path


def write_or_verify(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if path.is_file():
        prior = verify_self_hashed_json(path)
        old = dict(prior)
        old.pop("artifact_sha256", None)
        if old != dict(payload):
            raise ExperimentError(f"existing artifact differs: {path}")
    elif path.exists():
        raise ExperimentError(f"artifact is not a regular file: {path}")
    else:
        write_self_hashed_json(path, payload)
    return verify_self_hashed_json(path)


def resolve(path_value: Any, repo: Path) -> Path:
    path = Path(str(path_value))
    return path.resolve() if path.is_absolute() else (repo / path).resolve()
