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
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(value, dict): raise ProtocolError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    body = dict(value); body.pop("artifact_sha256", None); body["artifact_sha256"] = canonical_hash(body); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp"); temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"); os.replace(temporary, path); return body


def load_self_hashed(path: Path) -> dict[str, Any]:
    value = read_json(path); actual = value.get("artifact_sha256"); body = dict(value); body.pop("artifact_sha256", None)
    if not isinstance(actual, str) or actual != canonical_hash(body): raise ProtocolError(f"self-hash mismatch: {path}")
    return value


def group_bootstrap(values: list[float], groups: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for value, group in zip(values, groups, strict=True): grouped.setdefault(str(group), []).append(float(value))
    labels = sorted(grouped)
    if len(labels) != 8: raise ProtocolError("expected 8 source groups")
    means = np.asarray([np.mean(grouped[label], dtype=np.float64) for label in labels])
    indices = np.random.Generator(np.random.PCG64(20260910)).integers(0, 8, size=(20_000, 8), endpoint=False)
    estimates = means[indices].mean(axis=1)
    return {"mean": float(means.mean()), "ci99": [float(np.quantile(estimates, .005, method="linear")), float(np.quantile(estimates, .995, method="linear"))], "ci95": [float(np.quantile(estimates, .025, method="linear")), float(np.quantile(estimates, .975, method="linear"))], "group_labels": labels, "group_means": {label: float(value) for label, value in zip(labels, means, strict=True)}, "group_positive_count": int(np.sum(means > 0)), "draws": 20_000, "seed": 20260910, "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest()}
