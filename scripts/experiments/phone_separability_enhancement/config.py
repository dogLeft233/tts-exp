"""Configuration, hashing, and run-contract helpers for PSE v2."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import yaml

from . import MEASUREMENT_VERSION, PROTOCOL_ID


RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "detach") and callable(value.detach):
        tensor = value.detach()
        if int(getattr(tensor, "numel", lambda: 0)()) == 1:
            return _json_safe(tensor.cpu().item())
        raise TypeError("non-scalar tensor cannot be written to JSON run artifacts")
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError, RuntimeError):
            pass
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path, repo_root: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else Path(repo_root) / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return payload


def write_json(path: str | Path, payload: Any) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    return file_sha256(target)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(target)
    return file_sha256(target)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row is not an object: {path}")
                rows.append(value)
    return rows


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run-id must contain only letters, numbers, '_' or '-' and be at most 128 chars")
    return run_id


def protocol_snapshot(config: dict[str, Any], config_path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    source = dict(config.get("source", {}))
    input_hashes: dict[str, str | None] = {}
    for key in ("manifest", "tokens"):
        value = source.get(key)
        path = resolve_path(value, repo_root) if value else None
        input_hashes[key] = file_sha256(path) if path is not None and path.is_file() else None
    code_files = sorted(
        str(path.relative_to(repo_root))
        for path in (Path(repo_root) / "scripts/experiments/phone_separability_enhancement").rglob("*.py")
    )
    code_hashes = {
        name: file_sha256(Path(repo_root) / name)
        for name in code_files
        if (Path(repo_root) / name).is_file()
    }
    return {
        "schema_version": 2,
        "protocol_id": PROTOCOL_ID,
        "measurement_version": MEASUREMENT_VERSION,
        "config_path": str(config_file),
        "config_sha256": file_sha256(config_file),
        "config": config,
        "input_hashes": input_hashes,
        "declared_input_hashes": {key: source.get(f"{key}_sha256") for key in ("manifest", "tokens")},
        "code_hashes": code_hashes,
    }


def ensure_protocol_unchanged(previous: dict[str, Any], current: dict[str, Any]) -> None:
    for key in ("protocol_id", "measurement_version", "config_sha256", "input_hashes", "code_hashes"):
        if previous.get(key) != current.get(key):
            raise ValueError(f"resume rejected: protocol field changed: {key}")


__all__ = [
    "MEASUREMENT_VERSION",
    "PROTOCOL_ID",
    "canonical_hash",
    "canonical_json",
    "ensure_protocol_unchanged",
    "file_sha256",
    "load_yaml",
    "read_json",
    "read_jsonl",
    "resolve_path",
    "validate_run_id",
    "write_json",
    "write_jsonl",
    "protocol_snapshot",
]
