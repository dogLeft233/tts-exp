from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config


class ProtocolError(RuntimeError):
    """An input, resource, or artifact contract violation."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ProtocolError(f"non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")


def read_json(path: str | Path) -> Any:
    target = Path(path)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        return json.loads(target.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid JSON: {target}") from exc


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    assert_finite(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def write_self_hashed_json(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = canonical_sha256(body)
    write_json(path, body)
    return body


def read_self_hashed_json(path: str | Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    recorded = value.get("artifact_sha256")
    body = dict(value)
    body.pop("artifact_sha256", None)
    if not isinstance(recorded, str) or recorded != canonical_sha256(body):
        raise ProtocolError(f"artifact hash mismatch: {path}")
    return value


def require_file(path: str | Path, label: str) -> Path:
    target = Path(path)
    if not target.is_file():
        raise ProtocolError(f"{label} is missing: {target}")
    return target


def relative_binding(path: str | Path) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = config.REPO / target
    return target.resolve()


def sample_ids_sha256(sample_ids: Sequence[int | str]) -> str:
    return sha256_bytes("\n".join(str(value) for value in sample_ids).encode("utf-8"))


def csv_write(path: str | Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)


def csv_read(path: str | Path) -> list[dict[str, str]]:
    import csv

    with Path(path).open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def finite_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ProtocolError(f"{name} must have ndim={ndim}, got {array.shape}")
    if array.size == 0 or not np.isfinite(array).all():
        raise ProtocolError(f"{name} must be non-empty and finite")
    return array
