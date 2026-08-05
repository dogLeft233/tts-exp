"""Versioned utterance manifest helpers for speech and AV analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json

MANIFEST_SCHEMA_VERSION = 1
REQUIRED_FIELDS = (
    "dataset",
    "utterance_id",
    "speaker_id",
    "paired_key",
    "condition",
    "transcript",
    "audio_path",
    "split",
    "alignment_source",
    "license",
)
VALID_CONDITIONS = {"natural", "tts"}
VALID_ALIGNMENT_SOURCES = {"mfa", "uniform_fallback", "provided", "missing"}


def _as_text(value: Any, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest record {index}: {field} must be non-empty text")
    return value


def _validate_paired_key(record: dict[str, Any], index: int) -> None:
    paired_key = record["paired_key"]
    representation_only = bool(record.get("representation_only", False))
    if paired_key is None and representation_only and record["condition"] == "natural":
        return
    _as_text(paired_key, "paired_key", index)


def validate_record(record: dict[str, Any], index: int = 0) -> None:
    """Validate one manifest record and its controlled categorical fields."""
    if not isinstance(record, dict):
        raise ValueError(f"manifest record {index}: expected object")
    for field in REQUIRED_FIELDS:
        if field not in record:
            raise ValueError(f"manifest record {index}: missing {field}")
        if field != "paired_key":
            _as_text(record[field], field, index)
    _validate_paired_key(record, index)
    if record["condition"] not in VALID_CONDITIONS:
        raise ValueError(f"manifest record {index}: invalid condition {record['condition']!r}")
    if record["alignment_source"] not in VALID_ALIGNMENT_SOURCES:
        raise ValueError(
            f"manifest record {index}: invalid alignment_source {record['alignment_source']!r}"
        )
    if "sample_rate" in record and (not isinstance(record["sample_rate"], int) or record["sample_rate"] <= 0):
        raise ValueError(f"manifest record {index}: sample_rate must be a positive integer")
    if "tokens" in record and not isinstance(record["tokens"], list):
        raise ValueError(f"manifest record {index}: tokens must be a list")


def validate_manifest(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate records and reject duplicate dataset/utterance/condition keys."""
    materialized = list(records)
    seen: set[tuple[str, str, str]] = set()
    for index, record in enumerate(materialized):
        validate_record(record, index)
        key = (record["dataset"], record["utterance_id"], record["condition"])
        if key in seen:
            raise ValueError(f"manifest record {index}: duplicate key {key!r}")
        seen.add(key)
    return materialized


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a versioned manifest while accepting legacy alignment wrappers."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        version = payload.get("schema_version")
        if version is not None and version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema_version: {version!r}")
        records = payload.get("records")
        if records is None:
            records = payload.get("manifest", payload.get("entries"))
    else:
        version = 0
        records = payload
    if not isinstance(records, list):
        raise ValueError("manifest must contain a records list")
    if version == MANIFEST_SCHEMA_VERSION:
        return validate_manifest(records)
    if all(isinstance(record, dict) and all(field in record for field in REQUIRED_FIELDS) for record in records):
        return validate_manifest(records)
    return records


def write_manifest(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Validate and write a versioned JSON manifest."""
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "records": validate_manifest(records),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def check_speaker_leakage(records: Iterable[dict[str, Any]]) -> list[str]:
    """Return speakers appearing in more than one split."""
    splits: dict[str, set[str]] = {}
    for record in records:
        splits.setdefault(record["speaker_id"], set()).add(record["split"])
    return sorted(speaker for speaker, values in splits.items() if len(values) > 1)


def paired_keys(records: Iterable[dict[str, Any]]) -> set[str]:
    """Return pairing keys that have both natural and TTS conditions."""
    by_key: dict[str, set[str]] = {}
    for record in records:
        paired_key = record.get("paired_key")
        if paired_key is None:
            continue
        by_key.setdefault(str(paired_key), set()).add(record["condition"])
    return {key for key, conditions in by_key.items() if VALID_CONDITIONS <= conditions}
