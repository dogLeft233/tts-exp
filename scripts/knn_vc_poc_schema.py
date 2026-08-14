#!/usr/bin/env python3
"""Strict JSON and hash helpers for kNN-VC PoC artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

EXPECTED_CONDITIONS = (
    "natural_direct_resynthesis",
    "paired_tts_unconstrained",
    "paired_tts_same_phone",
    "paired_tts_same_phone_position",
    "paired_tts_mfa_linear",
    "paired_tts_wrong_phone",
)
SUMMARY_KEYS = {
    "schema_version", "experiment_type", "status", "manifest", "selection", "model",
    "policies", "runtime", "conditions", "items", "heldout_excluded",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite_json(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_json(item, f"{path}[{index}]")


def write_json(path: str | Path, value: Mapping[str, Any]) -> str:
    assert_finite_json(value)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return file_sha256(output)


def load_json(path: str | Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    payload = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    assert_finite_json(payload)
    return payload


def validate_summary(summary: Mapping[str, Any]) -> None:
    if set(summary) != SUMMARY_KEYS:
        raise ValueError(f"summary root schema mismatch: {sorted(set(summary) ^ SUMMARY_KEYS)}")
    if summary.get("schema_version") != 1 or summary.get("experiment_type") != "wavlm_knn_vc_valid_poc":
        raise ValueError("incompatible summary schema or experiment type")
    if summary.get("heldout_excluded") is not True:
        raise ValueError("heldout must be excluded")
    conditions = tuple(summary.get("conditions", []))
    if conditions != EXPECTED_CONDITIONS:
        raise ValueError("summary condition order is incompatible")
    selection = summary.get("selection")
    if not isinstance(selection, Mapping) or selection.get("split") != "valid" or selection.get("speaker_group") != "S0765":
        raise ValueError("summary selection is not the valid S0765 contract")
    keys = selection.get("ordered_paired_keys")
    if not isinstance(keys, list) or not keys or len(keys) != len(set(keys)):
        raise ValueError("selection requires unique ordered paired keys")
    if selection.get("ordered_paired_keys_sha256") != canonical_sha256(keys):
        raise ValueError("selection key hash mismatch")
    items = summary.get("items")
    if not isinstance(items, list) or len(items) != len(keys):
        raise ValueError("summary item count does not match selection")
    for key, item in zip(keys, items, strict=True):
        if not isinstance(item, Mapping) or item.get("paired_key") != key:
            raise ValueError("summary item order does not match selection")
        rows = item.get("conditions")
        if not isinstance(rows, list) or tuple(row.get("condition") for row in rows) != EXPECTED_CONDITIONS:
            raise ValueError(f"condition set mismatch for {key}")
        for row in rows:
            if row.get("paired_oracle") is not True:
                raise ValueError("every conversion row must declare paired_oracle=true")
