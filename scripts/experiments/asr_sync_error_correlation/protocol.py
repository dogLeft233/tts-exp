"""Frozen LRS3 parent joins and natural/TTS arm expansion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import (
    CANONICAL_SOURCE_MANIFEST,
    CANONICAL_TTS_METADATA,
    PARENT_COHORT_MANIFEST,
)
from .io import canonical_hash, file_sha256
from .word_errors import parse_lrs3_txt, validate_official_timing


def _path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _repo_path(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {path}")


def validate_parent_lock(repo_root: Path, parent: Mapping[str, Any], lock: Mapping[str, Any]) -> None:
    if parent.get("status") != "complete" or parent.get("decision") != "GO":
        raise ValueError("parent cohort lock is not complete GO")
    cohort = parent.get("cohort", {})
    records = cohort.get("records", [])
    if cohort.get("record_count") != 24 or cohort.get("source_group_count") != 24 or len(records) != 24:
        raise ValueError("parent cohort is not exactly 24 records/groups")
    if cohort.get("selection") != "existing ordered parent fresh_confirmation records":
        raise ValueError("parent cohort selection is not frozen fresh_confirmation")
    if cohort.get("sealed_splits_unvisited") is not True:
        raise ValueError("parent lock does not prove sealed splits were unvisited")
    if lock.get("status") != "sealed_unvisited":
        raise ValueError("test lock is not sealed_unvisited")
    for key in ("test_media_opened", "validation_media_opened", "internal_dev_media_opened"):
        if lock.get(key) is not False:
            raise ValueError(f"sealed media access lock is not false: {key}")
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    for row in records:
        sample_id = str(row.get("sample_id"))
        group = str(row.get("source_group"))
        if sample_id in seen_ids or group in seen_groups:
            raise ValueError("parent cohort sample_id/source_group is not unique")
        seen_ids.add(sample_id)
        seen_groups.add(group)
        if row.get("protocol_split") != "fit":
            raise ValueError(f"cohort record is not fit-only: {sample_id}")
        for key in ("face", "natural_audio", "tts_audio", "face_sha256", "natural_audio_sha256", "tts_audio_sha256"):
            if not row.get(key):
                raise ValueError(f"parent cohort missing {key}: {sample_id}")


def build_frozen_manifest(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    parent_path = _path(repo_root, PARENT_COHORT_MANIFEST)
    parent = _load(parent_path)
    parent_lock = _load(parent_path.with_name("test_lock.json"))
    validate_parent_lock(repo_root, parent, parent_lock)
    source_path = _path(repo_root, CANONICAL_SOURCE_MANIFEST)
    tts_path = _path(repo_root, CANONICAL_TTS_METADATA)
    parent_files = parent.get("parents", {}).get("parent_files", {})
    for parent_key, path in (("source_manifest", source_path), ("tts_meta", tts_path)):
        expected = parent_files.get(parent_key, {}).get("sha256")
        if not expected:
            raise ValueError(f"parent lock has no hash for {parent_key}")
        _assert_file_hash(path, str(expected), parent_key)
    source = _load(source_path)
    tts = _load(tts_path)
    if source.get("sample_count") != 500 or len(source.get("records", [])) != 500:
        raise ValueError("canonical source manifest is not n=500")
    if tts.get("sample_count") != 500 or len(tts.get("results", {})) != 500:
        raise ValueError("canonical TTS metadata is not n=500")
    source_by_id = {str(row["sample_id"]): row for row in source["records"]}
    tts_by_id = {str(key): value for key, value in tts["results"].items()}
    rows: list[dict[str, Any]] = []
    for parent_row in parent["cohort"]["records"]:
        sample_id = str(parent_row["sample_id"])
        source_row = source_by_id.get(sample_id)
        tts_row = tts_by_id.get(sample_id)
        if source_row is None or tts_row is None:
            raise ValueError(f"missing canonical join: {sample_id}")
        if source_row.get("source_group") != parent_row["source_group"] or tts_row.get("source_group") != parent_row["source_group"]:
            raise ValueError(f"source group mismatch: {sample_id}")
        if source_row.get("transcript") != tts_row.get("transcript"):
            raise ValueError(f"natural/TTS transcript mismatch: {sample_id}")
        if source_row.get("video_sha256") != parent_row["face_sha256"] or tts_row.get("video_sha256") != parent_row["face_sha256"]:
            raise ValueError(f"video hash mismatch: {sample_id}")
        if source_row.get("natural_audio_sha256") != parent_row["natural_audio_sha256"]:
            raise ValueError(f"natural audio hash mismatch: {sample_id}")
        if tts_row.get("reference_audio_sha256") != parent_row["natural_audio_sha256"]:
            raise ValueError(f"TTS reference audio mismatch: {sample_id}")
        if tts_row.get("canonical_audio_sha256") != parent_row["tts_audio_sha256"] or tts_row.get("status") != "ok":
            raise ValueError(f"TTS canonical audio mismatch: {sample_id}")
        official_path = _path(repo_root, source_row["official_transcript_path"])
        parsed = parse_lrs3_txt(official_path)
        validate_official_timing(parsed, str(source_row["transcript"]))
        for path_value, hash_value, label in (
            (parent_row["face"], parent_row["face_sha256"], "video"),
            (parent_row["natural_audio"], parent_row["natural_audio_sha256"], "natural audio"),
            (parent_row["tts_audio"], parent_row["tts_audio_sha256"], "TTS audio"),
            (source_row["official_transcript_path"], file_sha256(official_path), "official transcript"),
        ):
            _assert_file_hash(_path(repo_root, path_value), hash_value, label)
        common = {
            "sample_id": sample_id,
            "source_group": str(parent_row["source_group"]),
            "split": "fit",
            "transcript": str(source_row["transcript"]),
            "normalized_transcript": parsed["text"],
            "official_transcript_path": _repo_path(repo_root, official_path),
            "official_word_timings": parsed["words"],
            "video_path": _repo_path(repo_root, _path(repo_root, parent_row["face"])),
            "video_sha256": str(parent_row["face_sha256"]),
            "video_fps": float(source_row.get("video_fps", 25.0)),
            "video_duration_s": float(source_row["video_duration_s"]),
            "natural_audio_path": _repo_path(repo_root, _path(repo_root, parent_row["natural_audio"])),
            "natural_audio_sha256": str(parent_row["natural_audio_sha256"]),
            "tts_audio_path": _repo_path(repo_root, _path(repo_root, parent_row["tts_audio"])),
            "tts_audio_sha256": str(parent_row["tts_audio_sha256"]),
            "natural_audio_duration_s": float(source_row["natural_audio_duration_s"]),
            "tts_audio_duration_s": float(tts_row["canonical_duration_s"]),
            "sample_provenance": {
                "parent_manifest": _repo_path(repo_root, parent_path),
                "parent_manifest_sha256": file_sha256(parent_path),
                "source_manifest": _repo_path(repo_root, source_path),
                "source_manifest_sha256": file_sha256(source_path),
                "tts_metadata": _repo_path(repo_root, tts_path),
                "tts_metadata_sha256": file_sha256(tts_path),
                "parent_record_hash": canonical_hash(parent_row),
            },
        }
        for arm, audio_key, hash_key, duration_key in (
            ("natural", "natural_audio_path", "natural_audio_sha256", "natural_audio_duration_s"),
            ("tts", "tts_audio_path", "tts_audio_sha256", "tts_audio_duration_s"),
        ):
            rows.append({
                **common,
                "arm": arm,
                "arm_key": f"{sample_id}:{arm}",
                "audio_path": common[audio_key],
                "audio_sha256": common[hash_key],
                "audio_duration_s": common[duration_key],
            })
    validate_manifest_rows(rows)
    return {
        "schema_version": 1,
        "manifest_type": "lrs3_asr_sync_error_correlation",
        "selection": "frozen parent fresh_confirmation fit-only cohort",
        "sample_count": 24,
        "source_group_count": 24,
        "arm_count": 48,
        "arms": ["natural", "tts"],
        "sealed_splits_accessed": False,
        "parent_manifest_sha256": file_sha256(parent_path),
        "parent_test_lock_sha256": file_sha256(parent_path.with_name("test_lock.json")),
        "records": rows,
    }


def validate_manifest_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 48:
        raise ValueError(f"expected 48 arm records, found {len(rows)}")
    expected_arms = {"natural", "tts"}
    by_sample: dict[str, list[Mapping[str, Any]]] = {}
    seen_keys: set[str] = set()
    for row in rows:
        key = str(row.get("arm_key"))
        if key in seen_keys:
            raise ValueError(f"duplicate arm key: {key}")
        seen_keys.add(key)
        if row.get("split") != "fit":
            raise ValueError("manifest crossed sealed split")
        by_sample.setdefault(str(row["sample_id"]), []).append(row)
    if len(by_sample) != 24:
        raise ValueError("manifest does not contain 24 samples")
    if len({str(row["source_group"]) for row in rows}) != 24:
        raise ValueError("manifest does not preserve 24 source groups")
    for sample_id, sample_rows in by_sample.items():
        if {str(row["arm"]) for row in sample_rows} != expected_arms:
            raise ValueError(f"incomplete natural/tts arm pair: {sample_id}")
        if len({str(row["video_sha256"]) for row in sample_rows}) != 1:
            raise ValueError(f"paired video mismatch: {sample_id}")


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload = _load(Path(path))
    validate_manifest_rows(payload.get("records", []))
    if payload.get("sealed_splits_accessed") is not False:
        raise ValueError("manifest records sealed media access")
    return payload
