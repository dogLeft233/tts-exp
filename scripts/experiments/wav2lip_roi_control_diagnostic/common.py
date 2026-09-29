from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class DiagnosticError(RuntimeError):
    """A frozen-input, computation, or validation error."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DiagnosticError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"expected JSON object: {path}")
    return value


def verify_self_hashed_json(path: Path, expected_file_hash: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise DiagnosticError(f"missing JSON: {path}")
    actual_file_hash = file_sha256(path)
    if expected_file_hash is not None and actual_file_hash != expected_file_hash:
        raise DiagnosticError(
            f"fixed hash mismatch: {path}: expected {expected_file_hash}, actual {actual_file_hash}"
        )
    payload = read_json(path)
    recorded = payload.get("artifact_sha256")
    body = dict(payload)
    body.pop("artifact_sha256", None)
    if not isinstance(recorded, str) or recorded != canonical_json_sha256(body):
        raise DiagnosticError(f"self-hash mismatch: {path}")
    return payload


def write_self_hashed_json(path: Path, payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = canonical_json_sha256(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(body, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return file_sha256(path)


def resolve_path(value: Any, repo: Path) -> Path:
    path = Path(str(value))
    return (repo / path).resolve() if not path.is_absolute() else path.resolve()


def sample_ids_sha256(sample_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()


def assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise DiagnosticError(f"non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")


def compare_values(
    expected: Any,
    actual: Any,
    path: str,
    differences: list[dict[str, Any]],
    *,
    tolerance: float = 1e-6,
) -> None:
    """Compare a stored derived tree without replacing independent values."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            differences.append({"path": path, "expected": expected, "actual": actual})
            return
        for key, value in expected.items():
            if key not in actual:
                differences.append({"path": f"{path}.{key}", "expected": value, "actual": "<missing>"})
            else:
                compare_values(value, actual[key], f"{path}.{key}", differences, tolerance=tolerance)
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            differences.append({"path": path, "expected": expected, "actual": actual})
            return
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            compare_values(left, right, f"{path}[{index}]", differences, tolerance=tolerance)
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            differences.append({"path": path, "expected": expected, "actual": actual})
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if abs(float(expected) - float(actual)) > tolerance:
            differences.append({"path": path, "expected": expected, "actual": actual})
        return
    if expected != actual:
        differences.append({"path": path, "expected": expected, "actual": actual})
