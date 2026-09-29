"""Small, producer-independent helpers for the fresh-source timing branch.

The timing branch deliberately has its own protocol utilities.  In particular,
it does not import the fresh-source input producer/selection runner: a C run
must be able to audit a frozen input without accidentally opening the P gate or
changing the cohort denominator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


class TimingError(RuntimeError):
    """A timing protocol or artifact contract violation."""


SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
DELAY_SAMPLES = 3_200
DELAY_FRAMES = 5
U_START = 25
U_STOP = 115
U_ROWS = tuple(range(U_START, U_STOP))
NATURAL_LAG_START = -15
NATURAL_LAG_STOP = 15
DELAY_LAG_START = -20
DELAY_LAG_STOP = 10
LAG_COUNT = 31
EXPECTED_GROUP_COUNT = 12
MIN_VISUAL_ROWS = 81
BOOTSTRAP_SEED = 20_260_910
BOOTSTRAP_DRAWS = 20_000
MATRIX_TOLERANCE = 1e-4
SCALAR_TOLERANCE = 1e-6
PROTOCOL_ID = "fresh_source_timing_transfer"
PROTOCOL_REVISION = "fresh_source_timing_transfer_v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash JSON values using the same canonical representation as P artifacts."""

    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(target.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise TimingError(f"invalid JSON: {target}") from exc
    if not isinstance(value, dict):
        raise TimingError(f"expected JSON object: {target}")
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Write an atomically replaced self-hashed JSON artifact."""

    target = Path(path)
    body = dict(payload)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = canonical_hash(body)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return body


def load_self_hashed(path: str | Path) -> dict[str, Any]:
    value = read_json(path)
    recorded = value.get("artifact_sha256")
    body = dict(value)
    body.pop("artifact_sha256", None)
    if not isinstance(recorded, str) or recorded != canonical_hash(body):
        raise TimingError(f"self-hash mismatch: {path}")
    return value


def assert_finite(value: Any, path: str = "root") -> None:
    """Reject non-finite nested values before they enter a signed artifact."""

    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise TimingError(f"non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")


def resolve_path(value: Any, *, base: Path | None = None) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TimingError(f"artifact path is not a string: {value!r}")
    path = Path(value)
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


def file_binding(path: str | Path, **extra: Any) -> dict[str, Any]:
    target = Path(path).resolve()
    if not target.is_file():
        raise TimingError(f"artifact is missing: {target}")
    result: dict[str, Any] = {"path": str(target), "sha256": file_sha256(target)}
    result.update(extra)
    return result


def verify_file_binding(item: Mapping[str, Any], *, label: str, base: Path | None = None) -> Path:
    path = resolve_path(item.get("path"), base=base)
    expected = item.get("sha256")
    if not isinstance(expected, str) or not path.is_file() or file_sha256(path) != expected:
        raise TimingError(f"{label} hash binding failed: {path}")
    return path


def as_finite_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if ndim is not None and result.ndim != ndim:
        raise TimingError(f"{name} must have ndim={ndim}, got {result.shape}")
    if result.size == 0 or not np.isfinite(result).all():
        raise TimingError(f"{name} must be non-empty and finite")
    return result


def quantile(values: np.ndarray, q: float) -> float:
    """NumPy's linear interpolation with a compatibility fallback."""

    try:
        return float(np.quantile(values, q, method="linear"))
    except TypeError:  # NumPy < 1.22
        return float(np.quantile(values, q, interpolation="linear"))
