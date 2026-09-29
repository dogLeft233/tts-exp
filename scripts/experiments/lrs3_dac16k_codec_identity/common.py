from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


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


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(data, encoding="utf-8")
    os.replace(temporary, path)
    return file_sha256(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_self_hashed_json(path: Path, payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = canonical_json_sha256(body)
    return write_json_atomic(path, body)


def verify_self_hashed_json(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    recorded = payload.get("artifact_sha256")
    if not isinstance(recorded, str):
        raise TypeError(f"self-hash is missing: {path}")
    body = dict(payload)
    body.pop("artifact_sha256")
    if recorded != canonical_json_sha256(body):
        raise ValueError(f"self-hash mismatch: {path}")
    return payload


def assert_run_root_compatible(root: Path, protocol_id: str) -> None:
    if not root.exists():
        return
    entries = [entry for entry in root.iterdir() if entry.name != ".DS_Store"]
    if not entries:
        return
    marker = root / "00_protocol" / "protocol.json"
    if not marker.is_file():
        failures = root / "00_protocol" / "failures.json"
        if failures.is_file():
            failure_payload = read_json(failures)
            if failure_payload.get("protocol_id") == protocol_id:
                return
        raise ValueError(f"populated run root has no protocol marker: {root}")
    marker_payload = read_json(marker)
    if marker_payload.get("protocol_id") != protocol_id:
        raise ValueError(f"run root belongs to another protocol: {root}")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
