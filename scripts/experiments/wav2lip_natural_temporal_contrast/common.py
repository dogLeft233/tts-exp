from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

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
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProtocolError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"expected object JSON: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = canonical_hash(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
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


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def pcm16(path: Path) -> bytes:
    import wave

    with wave.open(str(path), "rb") as handle:
        if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) != (1, 2, 16_000):
            raise ProtocolError(f"audio is not mono PCM16/16k: {path}")
        return handle.readframes(handle.getnframes())


def free_gpu_mib() -> int:
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise ProtocolError("nvidia-smi unavailable; GPU stage is queued/blocked")
    try:
        return int(float(result.stdout.splitlines()[0].strip()))
    except ValueError as exc:
        raise ProtocolError(f"invalid nvidia-smi output: {result.stdout!r}") from exc


@contextmanager
def gpu_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        if free_gpu_mib() < 5120:
            raise ProtocolError("GPU free memory is below the frozen 5 GiB gate")
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def metrics(matrix: np.ndarray, rows: tuple[int, ...] | list[int] | None = None) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 31 or not np.isfinite(value).all():
        raise ProtocolError(f"invalid matrix: {value.shape}")
    chosen = value if rows is None else value[np.asarray(rows, dtype=np.int64)]
    curve = np.mean(chosen, axis=0, dtype=np.float64)
    index = int(np.argmin(curve))
    return {"curve": [float(x) for x in curve], "min_index": index, "offset": 15 - index, "D": float(curve[index]), "C": float(np.median(curve) - curve[index])}


def group_bootstrap(values: list[float], groups: list[str]) -> dict[str, Any]:
    if len(values) != len(groups):
        raise ProtocolError("values/groups length mismatch")
    grouped: dict[str, list[float]] = {}
    for value, group in zip(values, groups, strict=True):
        grouped.setdefault(str(group), []).append(float(value))
    labels = sorted(grouped)
    if len(labels) != 8 or any(len(grouped[label]) != 2 for label in labels):
        raise ProtocolError("expected eight source groups with two records each")
    means = np.asarray([np.mean(grouped[label], dtype=np.float64) for label in labels], dtype=np.float64)
    indices = np.random.Generator(np.random.PCG64(20260910)).integers(0, 8, size=(20_000, 8), endpoint=False)
    estimates = np.mean(means[indices], axis=1, dtype=np.float64)
    return {"mean": float(np.mean(means)), "ci99": [float(np.quantile(estimates, 0.005, method="linear")), float(np.quantile(estimates, 0.995, method="linear"))], "ci95": [float(np.quantile(estimates, 0.025, method="linear")), float(np.quantile(estimates, 0.975, method="linear"))], "group_labels": labels, "group_means": {label: float(item) for label, item in zip(labels, means, strict=True)}, "group_positive_count": int(np.sum(means > 0)), "draws": 20_000, "seed": 20260910, "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()}
