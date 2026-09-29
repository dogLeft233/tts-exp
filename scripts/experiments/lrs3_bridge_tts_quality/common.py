from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config


class ProtocolError(RuntimeError):
    """A registered input, artifact, or protocol contract is invalid."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return bytes_sha256(canonical_json_bytes(value))


def sample_ids_sha256(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in sample_ids).encode("utf-8")).hexdigest()


def decoded_pcm_sha256(values: np.ndarray) -> str:
    array = np.asarray(values)
    if array.dtype != np.int16 or array.ndim != 1:
        raise ValueError("decoded PCM hash requires one-dimensional int16 samples")
    return bytes_sha256(np.asarray(array, dtype="<i2").tobytes())


def assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")


def read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(target.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {target}")
    assert_finite(value)
    return value


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, target)
    return file_sha256(target)


def write_self_hashed_json(path: str | Path, value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = canonical_json_sha256(body)
    return write_json_atomic(path, body)


def verify_self_hashed_json(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    recorded = payload.get("artifact_sha256")
    if not isinstance(recorded, str):
        raise ProtocolError(f"artifact self-hash is missing: {path}")
    body = dict(payload)
    body.pop("artifact_sha256", None)
    if recorded != canonical_json_sha256(body):
        raise ProtocolError(f"artifact self-hash mismatch: {path}")
    return payload


def asset_binding(path: str | Path, *, expected_sha256: str | None = None, name: str = "asset") -> dict[str, str]:
    target = Path(path).resolve()
    if not target.is_file():
        raise ProtocolError(f"{name} is missing: {target}")
    actual = file_sha256(target)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ProtocolError(f"{name} hash mismatch: {target}")
    assert_not_sealed(target)
    return {"path": str(target), "sha256": actual}


def verify_asset(value: Any, *, name: str = "asset", allow_sealed: bool = False) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"malformed {name} binding")
    path = Path(str(value.get("path", ""))).resolve()
    expected = str(value.get("sha256", ""))
    if not path.is_file() or not expected or file_sha256(path) != expected:
        raise ProtocolError(f"bound {name} changed or is missing: {path}")
    if not allow_sealed:
        assert_not_sealed(path)
    return {"path": str(path), "sha256": expected}


def assert_not_sealed(path: str | Path, tokens: tuple[str, ...] = config.NO_SEALED_MEDIA_TOKENS) -> None:
    lowered = str(Path(path).resolve()).lower()
    if any(token in lowered for token in tokens):
        raise ProtocolError(f"sealed media path crossed: {path}")


def copy_verified(source: str | Path, destination: str | Path, expected_sha256: str) -> str:
    source_path = Path(source).resolve()
    destination_path = Path(destination)
    if file_sha256(source_path) != expected_sha256:
        raise ProtocolError(f"source changed before copy: {source_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(f".{destination_path.name}.{os.getpid()}.tmp")
    shutil.copy2(source_path, temporary)
    if file_sha256(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ProtocolError(f"copy changed bytes: {source_path}")
    os.replace(temporary, destination_path)
    return expected_sha256


def record_seed(sample_id: str) -> int:
    return int(hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()[:8], 16)


def command_sha256(command: Sequence[str]) -> str:
    return canonical_json_sha256([str(item) for item in command])


def runtime_binding(path: str | Path, *, expected: str | None = None) -> dict[str, str]:
    return asset_binding(path, expected_sha256=expected, name="runtime binding")


def runtime_versions() -> dict[str, str]:
    import importlib.metadata

    names = ("numpy", "scipy", "torch", "soundfile", "opencv-python", "pytest")
    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def assert_run_root_compatible(root: Path) -> None:
    if not root.exists():
        return
    marker = root / "00_protocol" / "setup.json"
    entries = [item for item in root.iterdir() if item.name != ".DS_Store"]
    if entries and not marker.is_file():
        raise ProtocolError(f"populated run root has no setup marker: {root}")
    if marker.is_file():
        payload = verify_self_hashed_json(marker)
        if payload.get("protocol_id") != config.PROTOCOL_ID:
            raise ProtocolError(f"run root belongs to another protocol: {root}")
