from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class ProtocolError(RuntimeError):
    pass


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = canonical_hash(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return body


def load_self_hashed(path: Path) -> dict[str, Any]:
    value = read_json(path)
    actual = value.get("artifact_sha256")
    body = dict(value)
    body.pop("artifact_sha256", None)
    if not isinstance(actual, str) or actual != canonical_hash(body):
        raise ProtocolError(f"self-hash mismatch: {path}")
    return value


def tree_manifest(root: Path) -> tuple[list[dict[str, str]], str]:
    repo = Path(__file__).resolve().parents[3]
    files = sorted(path for path in root.rglob("*") if path.is_file())
    rows = [
        {"path": path.relative_to(repo).as_posix(), "sha256": file_sha256(path)}
        for path in files
    ]
    joined = "".join(f"{row['sha256']}  {row['path']}\n" for row in rows).encode(
        "utf-8"
    )
    return rows, hashlib.sha256(joined).hexdigest()


def finite_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ProtocolError(f"{name} is non-finite")
    return array
