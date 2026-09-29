from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import config


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


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return bytes_sha256(canonical_json_bytes(payload))


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {item}")))
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


def write_self_hashed_json(path: Path, payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = canonical_json_sha256(body)
    return write_json_atomic(path, body)


def verify_self_hashed_json(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    recorded = payload.get("artifact_sha256")
    if not isinstance(recorded, str):
        raise ProtocolError(f"self-hash is missing: {path}")
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


def require_file(path: Path, label: str) -> Path:
    assert_not_sealed(path)
    if not path.is_file():
        raise ProtocolError(f"{label} is missing: {path}")
    return path


def sample_ids_sha256(sample_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def assert_run_root_compatible(root: Path) -> None:
    if not root.exists():
        return
    if not root.is_dir():
        raise ProtocolError(f"run root is not a directory: {root}")
    entries = [entry for entry in root.iterdir() if entry.name != ".DS_Store"]
    if not entries:
        return
    for marker in (root / "protocol.json", root / "final.json"):
        if marker.is_file():
            payload = verify_self_hashed_json(marker)
            if payload.get("protocol_id") == config.PROTOCOL_ID:
                return
    raise ProtocolError(f"populated run root has no compatible marker: {root}")


def write_or_verify(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if path.is_file():
        prior = verify_self_hashed_json(path)
        old = dict(prior)
        old.pop("artifact_sha256", None)
        if old != dict(payload):
            raise ProtocolError(f"existing artifact differs: {path}")
    elif path.exists():
        raise ProtocolError(f"artifact marker is not a regular file: {path}")
    else:
        write_self_hashed_json(path, payload)
    return verify_self_hashed_json(path)
