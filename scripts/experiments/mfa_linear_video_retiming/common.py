from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


class ProtocolError(RuntimeError):
    """Raised when a frozen experiment contract is not satisfied."""


SYNCNET_REQUEST_EXEC_CODE = (
    "import sys; from scripts.experiments.mfa_linear_video_retiming.search_worker "
    "import run_request; run_request(sys.argv[1])"
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return bytes_sha256(canonical_json_bytes(value))


def read_json(path: str | Path) -> Any:
    def reject(value: str) -> None:
        raise ProtocolError(f"non-finite JSON constant in {path}: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def write_json(path: str | Path, value: Any, *, self_hash: bool = False) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value) if isinstance(value, Mapping) else value
    if self_hash:
        if not isinstance(payload, dict):
            raise TypeError("self-hashed JSON payload must be an object")
        payload.pop("artifact_sha256", None)
        payload["artifact_sha256"] = canonical_json_sha256(payload)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    return payload


def verify_json(path: str | Path, *, self_hash: bool = False) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    if self_hash:
        recorded = payload.get("artifact_sha256")
        body = dict(payload)
        body.pop("artifact_sha256", None)
        if not isinstance(recorded, str) or recorded != canonical_json_sha256(body):
            raise ProtocolError(f"self-hash mismatch: {path}")
    return payload


def path_binding(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "exists": resolved.is_file(),
        "sha256": file_sha256(resolved) if resolved.is_file() else None,
        "size_bytes": resolved.stat().st_size if resolved.is_file() else None,
    }
