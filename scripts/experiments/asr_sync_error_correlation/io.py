"""Strict, inspectable artifact IO helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON value at {path}")
    elif isinstance(value, Mapping):
        for key, child in value.items():
            _finite(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite(child, f"{path}[{index}]")


def canonical_json(value: Any) -> bytes:
    _finite(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return bytes_sha256(canonical_json(value))


def _atomic_replace(path: Path, writer: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    _atomic_replace(Path(path), lambda handle: handle.write(data))


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, payload: Mapping[str, Any], *, indent: int = 2) -> None:
    _finite(payload)
    text = json.dumps(payload, indent=indent, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_text(path, text)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON constant: {value}")))
    _finite(payload)
    return payload


def atomic_write_npz(path: str | Path, arrays: Mapping[str, np.ndarray], *, compressed: bool = True) -> None:
    normalized: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"object arrays are forbidden in NPZ: {key}")
        if not np.all(np.isfinite(array)) and np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"non-finite NPZ array: {key}")
        normalized[str(key)] = array

    def writer(handle: Any) -> None:
        saver = np.savez_compressed if compressed else np.savez
        saver(handle, **normalized)

    _atomic_replace(Path(path), writer)


def read_npz(path: str | Path, required: Sequence[str] = ()) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    missing = set(required) - set(arrays)
    if missing:
        raise ValueError(f"NPZ missing arrays: {sorted(missing)}")
    for key, array in arrays.items():
        if array.dtype.hasobject:
            raise TypeError(f"object array in NPZ: {key}")
        if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
            raise ValueError(f"non-finite array in NPZ: {key}")
    return arrays


def output_hashes(paths: Mapping[str, str | Path]) -> dict[str, str]:
    return {name: file_sha256(path) for name, path in sorted(paths.items())}


def write_success_marker(path: str | Path, *, stage: str, sample_id: str, arm: str, bindings: Mapping[str, Any], outputs: Mapping[str, str | Path]) -> dict[str, Any]:
    marker = {
        "schema_version": 1,
        "stage": stage,
        "sample_id": str(sample_id),
        "arm": str(arm),
        "bindings": dict(bindings),
        "binding_hash": canonical_hash(bindings),
        "outputs": output_hashes(outputs),
    }
    atomic_write_json(path, marker)
    return marker


def success_marker_valid(path: str | Path, *, bindings: Mapping[str, Any], outputs: Mapping[str, str | Path], stage: str | None = None, sample_id: str | None = None, arm: str | None = None) -> bool:
    marker_path = Path(path)
    if not marker_path.is_file():
        return False
    try:
        marker = read_json(marker_path)
        if stage is not None and marker.get("stage") != stage:
            return False
        if sample_id is not None and str(marker.get("sample_id")) != str(sample_id):
            return False
        if arm is not None and str(marker.get("arm")) != str(arm):
            return False
        if marker.get("binding_hash") != canonical_hash(bindings):
            return False
        expected = output_hashes(outputs)
        return marker.get("outputs") == expected
    except (OSError, ValueError, TypeError, KeyError):
        return False


def append_failure(path: str | Path, *, stage: str, sample_id: str, arm: str, exception: BaseException, log_path: str | Path | None = None, retryable: bool = True) -> dict[str, Any]:
    ledger_path = Path(path)
    rows: list[dict[str, Any]] = []
    if ledger_path.is_file():
        old = read_json(ledger_path)
        if not isinstance(old, list):
            raise ValueError("failure ledger must be a JSON list")
        rows = old
    row = {
        "stage": stage,
        "sample_id": str(sample_id),
        "arm": str(arm),
        "exception_type": type(exception).__name__,
        "message": str(exception).replace("\n", " ")[:500],
        "log_path": str(log_path) if log_path is not None else None,
        "retryable": bool(retryable),
    }
    rows.append(row)
    atomic_write_json(ledger_path, rows)
    return row
