#!/usr/bin/env python3
"""Freeze the fit-only cohorts for the LRS3 TTS visual-advantage audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[3]
STAGE_ID = "00_protocol_lock"
SCHEMA_VERSION = 1

DEFAULT_SOURCE_MANIFEST = REPO / "runs/lrs3_qwen_cloud_n500_20260817/00_manifest/manifest.json"
DEFAULT_TTS_META = REPO / "runs/lrs3_qwen_cloud_n500_20260817/02_tts/tts_meta.json"
DEFAULT_PARENT_DIR = REPO / "runs/lrs3_motion_teacher_20260822/00_protocol_audit_retry1"
DEFAULT_TEACHER_MANIFEST = (
    REPO / "runs/lrs3_motion_teacher_20260822/01_wav2lip_teacher_pilot_retry3/teacher_manifest.json"
)
DEFAULT_LANDMARK_DIR = REPO / "runs/lrs3_motion_teacher_20260822/02_landmark_teacher_audit_retry2"
DEFAULT_STRICT_DIR = REPO / "runs/lrs3_motion_teacher_20260822/11_raw_tts_strict_2x2_retry1"
DEFAULT_WAV2LIP_CHECKPOINT = REPO / "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth"
DEFAULT_LANDMARKER_ASSET = REPO / "checkpoints/mediapipe/face_landmarker.task"
DEFAULT_CODE_REVIEW = REPO / "_reviews/lrs3_tts_visual_advantage/00_protocol_lock/code_review.json"

EXPECTED_COUNTS = {
    "source_records": 500,
    "source_groups": 43,
    "working_records": 416,
    "fit_groups": 24,
    "internal_dev_groups": 6,
    "validation_groups": 6,
    "test_groups": 7,
    "test_records": 84,
    "teacher_records": 60,
    "teacher_groups": 30,
    "pilot_fit_records": 48,
    "fresh_fit_records": 24,
}


def file_sha256(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def assert_finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite_json(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_json(item, f"{path}[{index}]")


def load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    payload = json.loads(target.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {target}")
    assert_finite_json(payload)
    return payload


def write_json(path: str | Path, value: Mapping[str, Any]) -> str:
    assert_finite_json(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return file_sha256(target)


def _unique_text(values: Sequence[Any], name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or any(not value for value in result) or len(result) != len(set(result)):
        raise ValueError(f"{name} must be a non-empty unique list")
    return result


def _sample_group(sample_id: str) -> str:
    parts = str(sample_id).split("_")
    if len(parts) != 3 or parts[0] != "lrs3" or not parts[1] or not parts[2].isdigit():
        raise ValueError(f"invalid LRS3 sample id: {sample_id!r}")
    return parts[1]


def _resolve_repo_path(value: str | Path) -> Path:
    target = Path(value)
    return (target if target.is_absolute() else REPO / target).resolve()


def _record_path(record: Mapping[str, Any], key: str) -> Path:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"record path {key} must be non-empty text")
    return _resolve_repo_path(value)


def validate_test_lock(
    test_lock: Mapping[str, Any],
    canonical_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if test_lock.get("status") != "sealed_unvisited":
        raise ValueError("test lock is not sealed_unvisited")
    required_top = ("media_opened", "derived_features_created", "scores_created")
    for key in required_top:
        if key not in test_lock:
            raise ValueError(f"test lock missing required access flag: {key}")
        if test_lock[key] is not False:
            raise ValueError(f"test lock {key} must be false")
    split_flags = (
        "test_media_opened",
        "test_derived_features_created",
        "test_scores_created",
        "validation_media_opened",
        "validation_derived_features_created",
        "validation_scores_created",
        "internal_dev_media_opened",
        "internal_dev_derived_features_created",
        "internal_dev_scores_created",
        "fit_media_opened",
        "fit_derived_features_created",
        "fit_scores_created",
    )
    has_split_schema = any(key in test_lock for key in split_flags)
    if has_split_schema:
        for key in split_flags:
            if key not in test_lock:
                raise ValueError(f"test lock missing split access flag: {key}")
            if test_lock[key] is not False:
                raise ValueError(f"test lock {key} must be false")
    groups = _unique_text(test_lock.get("test_groups", []), "test_groups")
    sample_ids = _unique_text(test_lock.get("test_sample_ids", []), "test_sample_ids")
    if len(groups) != EXPECTED_COUNTS["test_groups"]:
        raise ValueError("unexpected test group count")
    if len(sample_ids) != EXPECTED_COUNTS["test_records"]:
        raise ValueError("unexpected test sample count")
    if any(_sample_group(sample_id) not in groups for sample_id in sample_ids):
        raise ValueError("test sample membership does not match locked groups")
    if canonical_records is not None:
        canonical_by_id = {str(row.get("sample_id")): row for row in canonical_records}
        if len(canonical_by_id) != len(canonical_records):
            raise ValueError("canonical source sample ids must be unique")
        expected_ids = {
            sample_id
            for sample_id, row in canonical_by_id.items()
            if str(row.get("source_group")) in groups
        }
        if set(sample_ids) != expected_ids:
            raise ValueError("test lock is not the exact canonical membership")
    return {"groups": list(groups), "sample_ids": list(sample_ids)}


def _audit_record(record: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "sample_id",
        "source_group",
        "protocol_split",
        "face",
        "face_sha256",
        "natural_audio",
        "natural_audio_sha256",
        "tts_audio",
        "tts_audio_sha256",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"record missing fields: {missing}")
    if record["protocol_split"] != "fit":
        raise ValueError(f"audit cohort must be fit-only: {record['sample_id']}")
    return {key: record[key] for key in required}


def _validate_cross_manifest_joins(
    parent_manifest: Mapping[str, Any],
    teacher_manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    tts_meta: Mapping[str, Any],
    canonical_test_lock: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    source_records = source_manifest.get("records")
    if not isinstance(source_records, list) or len(source_records) != EXPECTED_COUNTS["source_records"]:
        raise ValueError("canonical source records are incomplete")
    source_by_id = {str(row.get("sample_id")): row for row in source_records}
    if len(source_by_id) != len(source_records) or any(not key for key in source_by_id):
        raise ValueError("canonical source sample ids must be unique")
    tts_results = tts_meta.get("results")
    if not isinstance(tts_results, Mapping) or len(tts_results) != EXPECTED_COUNTS["source_records"]:
        raise ValueError("canonical TTS results are incomplete")
    tts_by_id = {str(key): value for key, value in tts_results.items()}
    if any(not isinstance(value, Mapping) for value in tts_by_id.values()):
        raise ValueError("TTS results must be mappings")

    protocol = parent_manifest.get("protocol")
    if not isinstance(protocol, Mapping) or not isinstance(protocol.get("working_records"), list):
        raise ValueError("parent protocol working records are missing")
    working_by_id = {str(row.get("sample_id")): row for row in protocol["working_records"]}
    if len(working_by_id) != len(protocol["working_records"]):
        raise ValueError("parent working sample ids must be unique")

    def check_join(row: Mapping[str, Any], expected_split: str | None = None) -> None:
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in source_by_id or sample_id not in tts_by_id:
            raise ValueError(f"record is absent from canonical source/TTS manifests: {sample_id}")
        source = source_by_id[sample_id]
        tts = tts_by_id[sample_id]
        if str(row.get("source_group")) != str(source.get("source_group")):
            raise ValueError(f"source group mismatch for {sample_id}")
        if _sample_group(sample_id) != str(source.get("source_group")):
            raise ValueError(f"sample id/group mismatch for {sample_id}")
        if expected_split is not None and row.get("protocol_split") != expected_split:
            raise ValueError(f"unexpected split for {sample_id}")
        if row.get("face_sha256") != source.get("video_sha256"):
            raise ValueError(f"face hash mismatch for {sample_id}")
        if row.get("natural_audio_sha256") != source.get("natural_audio_sha256"):
            raise ValueError(f"natural audio hash mismatch for {sample_id}")
        if row.get("tts_audio_sha256") != tts.get("canonical_audio_sha256"):
            raise ValueError(f"TTS audio hash mismatch for {sample_id}")
        if _record_path(row, "face") != _resolve_repo_path(source["video_local_path"]):
            raise ValueError(f"face path mismatch for {sample_id}")
        if _record_path(row, "natural_audio") != _resolve_repo_path(source["natural_audio_path"]):
            raise ValueError(f"natural audio path mismatch for {sample_id}")
        if _record_path(row, "tts_audio") != _resolve_repo_path(tts["canonical_16k_audio"]):
            raise ValueError(f"TTS audio path mismatch for {sample_id}")
        if tts.get("source_group") != source.get("source_group") or tts.get("video_sha256") != source.get("video_sha256"):
            raise ValueError(f"TTS/source metadata mismatch for {sample_id}")
        if tts.get("reference_audio_sha256") != source.get("natural_audio_sha256"):
            raise ValueError(f"TTS reference audio mismatch for {sample_id}")

    for row in protocol["working_records"]:
        check_join(row)
    teacher_records = teacher_manifest.get("records")
    if not isinstance(teacher_records, list):
        raise ValueError("teacher records are missing")
    for row in teacher_records:
        sample_id = str(row.get("sample_id", ""))
        if sample_id not in working_by_id:
            raise ValueError(f"teacher record is absent from parent working records: {sample_id}")
        check_join(row, working_by_id[sample_id].get("protocol_split"))
        parent_row = working_by_id[sample_id]
        for key in ("source_group", "face_sha256", "natural_audio_sha256", "tts_audio_sha256"):
            if row.get(key) != parent_row.get(key):
                raise ValueError(f"teacher/parent field mismatch for {sample_id}: {key}")

    test_ids = set(canonical_test_lock["sample_ids"])
    return source_by_id, tts_by_id, working_by_id


def build_cohorts(
    parent_manifest: Mapping[str, Any],
    teacher_manifest: Mapping[str, Any],
    test_lock: Mapping[str, Any],
    source_manifest: Mapping[str, Any] | None = None,
    tts_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if source_manifest is None or tts_meta is None:
        raise ValueError("canonical source and TTS metadata are required for cohort construction")
    source_records = source_manifest.get("records")
    if not isinstance(source_records, list):
        raise ValueError("canonical source records are missing")
    locked_test = validate_test_lock(test_lock, source_records)
    parent_protocol = parent_manifest.get("protocol")
    if not isinstance(parent_protocol, Mapping) or not isinstance(parent_protocol.get("working_records"), list):
        raise ValueError("parent protocol working records are missing")
    working_ids = {str(row.get("sample_id", "")) for row in parent_protocol["working_records"]}
    if working_ids & set(locked_test["sample_ids"]):
        raise ValueError("working records intersect the sealed test set")
    _validate_cross_manifest_joins(
        parent_manifest,
        teacher_manifest,
        source_manifest,
        tts_meta,
        locked_test,
    )
    protocol = parent_manifest.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("parent manifest is missing protocol")
    if parent_manifest.get("status") != "complete" or parent_manifest.get("decision") != "GO":
        raise ValueError("parent protocol audit is not complete GO")

    split = protocol.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("parent protocol is missing split")
    fit_groups = _unique_text(split.get("fit_groups", []), "fit_groups")
    internal_groups = _unique_text(split.get("internal_dev_groups", []), "internal_dev_groups")
    validation_groups = _unique_text(split.get("validation_groups", []), "validation_groups")
    parent_test_groups = _unique_text(split.get("test_groups", []), "parent test_groups")
    if len(fit_groups) != EXPECTED_COUNTS["fit_groups"]:
        raise ValueError("unexpected fit group count")
    if len(internal_groups) != EXPECTED_COUNTS["internal_dev_groups"]:
        raise ValueError("unexpected internal-dev group count")
    if len(validation_groups) != EXPECTED_COUNTS["validation_groups"]:
        raise ValueError("unexpected validation group count")
    if set(parent_test_groups) != set(locked_test["groups"]):
        raise ValueError("parent split and test lock disagree")
    all_groups = [set(fit_groups), set(internal_groups), set(validation_groups), set(parent_test_groups)]
    for left in range(len(all_groups)):
        for right in range(left + 1, len(all_groups)):
            if all_groups[left] & all_groups[right]:
                raise ValueError("protocol group splits overlap")

    working_records = protocol.get("working_records")
    if not isinstance(working_records, list) or len(working_records) != EXPECTED_COUNTS["working_records"]:
        raise ValueError("unexpected working record count")
    sample_ids = [str(row.get("sample_id", "")) for row in working_records]
    if any(not value for value in sample_ids) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("working record sample ids must be unique")
    if set(sample_ids) & set(locked_test["sample_ids"]):
        raise ValueError("working records intersect the sealed test set")
    if any(str(row.get("source_group", "")) in parent_test_groups for row in working_records):
        raise ValueError("working records contain a sealed test group")

    if teacher_manifest.get("status") != "complete" or teacher_manifest.get("decision") != "GO":
        raise ValueError("teacher manifest is not complete GO")
    teacher_records = teacher_manifest.get("records")
    if not isinstance(teacher_records, list) or len(teacher_records) != EXPECTED_COUNTS["teacher_records"]:
        raise ValueError("unexpected teacher record count")
    teacher_ids = [str(row.get("sample_id", "")) for row in teacher_records]
    if any(not value for value in teacher_ids) or len(teacher_ids) != len(set(teacher_ids)):
        raise ValueError("teacher sample ids must be unique")
    if set(teacher_ids) & set(locked_test["sample_ids"]):
        raise ValueError("teacher cache intersects sealed test ids")
    if teacher_manifest.get("syncnet_used_for_selection") is not False:
        raise ValueError("teacher selection must be score-independent")

    teacher_counts = Counter(str(row.get("source_group", "")) for row in teacher_records)
    if len(teacher_counts) != EXPECTED_COUNTS["teacher_groups"] or set(teacher_counts.values()) != {2}:
        raise ValueError("teacher cache must contain exactly two records per group")

    pilot_rows = [row for row in teacher_records if str(row.get("source_group")) in fit_groups]
    pilot_rows.sort(key=lambda row: (fit_groups.index(str(row["source_group"])), str(row["sample_id"])))
    pilot_counts = Counter(str(row["source_group"]) for row in pilot_rows)
    if len(pilot_rows) != EXPECTED_COUNTS["pilot_fit_records"]:
        raise ValueError("unexpected pilot fit record count")
    if set(pilot_counts) != set(fit_groups) or set(pilot_counts.values()) != {2}:
        raise ValueError("pilot cohort must have two records for every fit group")

    fit_candidates: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in working_records:
        if row.get("protocol_split") == "fit" and str(row.get("source_group")) in fit_groups:
            fit_candidates[str(row["source_group"])].append(row)
    fresh_rows: list[Mapping[str, Any]] = []
    used_teacher_ids = set(teacher_ids)
    for group in fit_groups:
        candidates = sorted(
            (row for row in fit_candidates[group] if str(row["sample_id"]) not in used_teacher_ids),
            key=lambda row: str(row["sample_id"]),
        )
        if not candidates:
            raise ValueError(f"fit group has no fresh record: {group}")
        fresh_rows.append(candidates[0])
    if len(fresh_rows) != EXPECTED_COUNTS["fresh_fit_records"]:
        raise ValueError("unexpected fresh confirmation count")
    if {str(row["sample_id"]) for row in fresh_rows} & set(teacher_ids):
        raise ValueError("fresh confirmation cohort overlaps teacher cache")

    pilot = [_audit_record(row) for row in pilot_rows]
    fresh = [_audit_record(row) for row in fresh_rows]
    return {
        "split": {
            "unit": "source_group_directory_not_verified_speaker_id",
            "fit_groups": list(fit_groups),
            "internal_dev_groups": list(internal_groups),
            "validation_groups": list(validation_groups),
            "test_groups": list(parent_test_groups),
        },
        "pilot": {
            "role": "seen_engineering_parity_only",
            "selection": "existing_teacher_cache_fit_records",
            "record_count": len(pilot),
            "source_group_count": len(pilot_counts),
            "records": pilot,
        },
        "fresh_confirmation": {
            "role": "fit_only_confirmatory_visual_endpoint",
            "selection": "first_sample_id_ascending_not_in_teacher_cache_per_fit_group",
            "score_based_substitution": False,
            "record_count": len(fresh),
            "source_group_count": len(fit_groups),
            "records": fresh,
        },
    }


def stage_test_lock(parent_lock: Mapping[str, Any], cohorts: Mapping[str, Any]) -> dict[str, Any]:
    locked = validate_test_lock(parent_lock)
    split = cohorts["split"]
    return {
        "status": "sealed_unvisited",
        "scope": "internal_dev_validation_and_test_media",
        "media_opened": False,
        "derived_features_created": False,
        "scores_created": False,
        "test_group_count": len(locked["groups"]),
        "test_groups": locked["groups"],
        "test_sample_ids": locked["sample_ids"],
        "test_media_opened": False,
        "test_derived_features_created": False,
        "test_scores_created": False,
        "validation_groups": split["validation_groups"],
        "validation_media_opened": False,
        "validation_derived_features_created": False,
        "validation_scores_created": False,
        "internal_dev_groups": split["internal_dev_groups"],
        "internal_dev_media_opened": False,
        "internal_dev_derived_features_created": False,
        "internal_dev_scores_created": False,
        "fit_media_opened": False,
        "fit_derived_features_created": False,
        "fit_scores_created": False,
    }


def source_manifest(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        rows.append({"path": str(resolved.relative_to(REPO)), "sha256": file_sha256(resolved)})
    return rows


def validate_code_review(path: Path, expected_source_manifest: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    review = load_json(path)
    if review.get("stage_id") != STAGE_ID or review.get("status") != "PASS":
        raise ValueError("Stage 00 code review is not PASS")
    expected_hash = canonical_sha256(list(expected_source_manifest))
    if review.get("source_manifest_sha256") != expected_hash:
        raise ValueError("Stage 00 code review source hash mismatch")
    if review.get("test_status") != "PASS":
        raise ValueError("Stage 00 code review tests are not PASS")
    return review


def git_info() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"available": True, "revision": revision, "branch": branch, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"available": False}


def _asset(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    outdir = args.outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {outdir}")

    own_sources = source_manifest((Path(__file__), args.test_source))
    review = validate_code_review(args.code_review.resolve(), own_sources)

    source = load_json(args.source_manifest)
    tts_meta = load_json(args.tts_meta)
    parent_manifest = load_json(args.parent_dir / "manifest.json")
    parent_lock = load_json(args.parent_dir / "test_lock.json")
    teacher_manifest = load_json(args.teacher_manifest)
    old_landmark_summary = load_json(args.landmark_dir / "summary.json")
    old_landmark_decision = load_json(args.landmark_dir / "decision.json")
    old_landmark_audit = load_json(args.landmark_dir / "landmark_audit.json")
    old_strict_summary = load_json(args.strict_dir / "summary.json")
    old_strict_decision = load_json(args.strict_dir / "decision.json")

    if source.get("sample_count") != EXPECTED_COUNTS["source_records"]:
        raise ValueError("canonical source manifest is not n=500")
    if source.get("source_group_count") != EXPECTED_COUNTS["source_groups"]:
        raise ValueError("canonical source manifest does not contain 43 groups")
    if tts_meta.get("source_manifest_sha256") != file_sha256(args.source_manifest):
        raise ValueError("TTS metadata is not bound to the supplied source manifest")
    if _resolve_repo_path(tts_meta.get("source_manifest", "")) != args.source_manifest.resolve():
        raise ValueError("TTS metadata source manifest path mismatch")
    if tts_meta.get("sample_count") != EXPECTED_COUNTS["source_records"] or not tts_meta.get("complete"):
        raise ValueError("TTS metadata is not complete n=500")
    if parent_manifest.get("manifest_type") != "lrs3_motion_teacher_protocol_audit":
        raise ValueError("unexpected parent protocol manifest type")
    if teacher_manifest.get("manifest_type") != "lrs3_wav2lip_natural_tts_teacher_cache":
        raise ValueError("unexpected teacher manifest type")
    if teacher_manifest.get("parent_protocol_manifest_sha256") != file_sha256(args.parent_dir / "manifest.json"):
        raise ValueError("teacher manifest is not bound to the supplied parent protocol")
    if old_landmark_audit.get("manifest_type") != "lrs3_landmark_teacher_audit":
        raise ValueError("unexpected old landmark audit type")
    if old_landmark_audit.get("run_id") != "02_landmark_teacher_audit_retry2":
        raise ValueError("unexpected old landmark audit run")
    if old_landmark_audit.get("status") != "complete" or old_landmark_audit.get("decision") != "NO_GO":
        raise ValueError("old landmark audit is incomplete")
    teacher_hash = file_sha256(args.teacher_manifest)
    if old_landmark_audit.get("parent_teacher_manifest_sha256") != teacher_hash:
        raise ValueError("old landmark audit teacher parent hash mismatch")
    if _resolve_repo_path(old_landmark_audit.get("parent_teacher_manifest", "")) != args.teacher_manifest.resolve():
        raise ValueError("old landmark audit teacher parent path mismatch")
    if old_landmark_summary.get("status") != "complete" or old_landmark_decision.get("status") != "complete":
        raise ValueError("old landmark parent is incomplete")
    if old_landmark_summary.get("decision") != "NO_GO" or old_landmark_decision.get("decision") != "NO_GO":
        raise ValueError("old landmark parent is not the expected NO_GO")
    if old_strict_summary.get("manifest_type") != "lrs3_wav2lip_raw_tts_strict_pcm_2x2":
        raise ValueError("unexpected old strict 2x2 type")
    if old_strict_summary.get("run_id") != "11_raw_tts_strict_2x2_retry1":
        raise ValueError("unexpected old strict 2x2 run")
    if old_strict_summary.get("status") != "complete" or old_strict_decision.get("status") != "complete":
        raise ValueError("old strict parent is incomplete")
    if old_strict_summary.get("teacher_manifest_sha256") != teacher_hash:
        raise ValueError("old strict 2x2 teacher parent hash mismatch")
    if _resolve_repo_path(old_strict_summary.get("teacher_manifest", "")) != args.teacher_manifest.resolve():
        raise ValueError("old strict 2x2 teacher parent path mismatch")
    if old_strict_summary.get("decision") != "NO_GO" or old_strict_decision.get("decision") != "NO_GO":
        raise ValueError("old strict 2x2 parent is not the expected NO_GO")

    cohorts = build_cohorts(parent_manifest, teacher_manifest, parent_lock, source, tts_meta)
    test_lock = stage_test_lock(parent_lock, cohorts)
    validate_test_lock(test_lock, source.get("records"))
    parents = {
        "canonical_source_manifest": _asset(args.source_manifest),
        "tts_metadata": _asset(args.tts_meta),
        "motion_teacher_protocol_manifest": _asset(args.parent_dir / "manifest.json"),
        "motion_teacher_test_lock": _asset(args.parent_dir / "test_lock.json"),
        "teacher_manifest": _asset(args.teacher_manifest),
        "old_landmark_summary": _asset(args.landmark_dir / "summary.json"),
        "old_landmark_decision": _asset(args.landmark_dir / "decision.json"),
        "old_landmark_audit": _asset(args.landmark_dir / "landmark_audit.json"),
        "old_strict_summary": _asset(args.strict_dir / "summary.json"),
        "old_strict_decision": _asset(args.strict_dir / "decision.json"),
    }
    assets = {
        "wav2lip_checkpoint": _asset(args.wav2lip_checkpoint),
        "mediapipe_landmarker": _asset(args.landmarker_asset),
    }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "lrs3_tts_visual_advantage_protocol_lock",
        "stage_id": STAGE_ID,
        "status": "complete",
        "decision": "GO",
        "next_allowed_stage": "01_metric_parity",
        "scientific_question": "whether V_T is visually closer than V_N to real paired LRS3 video",
        "endpoints": {
            "primary": "natural_clock_exact_time_canonical_mouth_landmark_distance",
            "secondary": "visual_only_constrained_dtw_canonical_mouth_shape_distance",
            "benefit_direction": "D(real,V_N)-D(real,V_T)>0",
        },
        "parents": parents,
        "assets": assets,
        "cohorts": cohorts,
        "selection_constraints": {
            "fit_only": True,
            "score_based_substitution": False,
            "syncnet_used": False,
            "audio_or_phone_alignment_used": False,
            "raw_tts_time_correction": False,
        },
        "media_access": {
            "fit_media_opened": False,
            "internal_dev_media_opened": False,
            "validation_media_opened": False,
            "test_media_opened": False,
        },
        "source_manifest": own_sources,
        "source_manifest_sha256": canonical_sha256(own_sources),
        "code_review": _asset(args.code_review),
        "code_review_payload_sha256": canonical_sha256(review),
        "git": git_info(),
    }
    summary = {
        "stage_id": STAGE_ID,
        "status": "complete",
        "decision": "GO",
        "next_allowed_stage": "01_metric_parity",
        "pilot_record_count": cohorts["pilot"]["record_count"],
        "pilot_source_group_count": cohorts["pilot"]["source_group_count"],
        "fresh_record_count": cohorts["fresh_confirmation"]["record_count"],
        "fresh_source_group_count": cohorts["fresh_confirmation"]["source_group_count"],
        "pilot_fresh_disjoint": not bool(
            {row["sample_id"] for row in cohorts["pilot"]["records"]}
            & {row["sample_id"] for row in cohorts["fresh_confirmation"]["records"]}
        ),
        "internal_dev_validation_test_unopened": True,
    }
    decision = {
        "stage_id": STAGE_ID,
        "status": "complete",
        "decision": "GO",
        "next_allowed_stage": "01_metric_parity",
        "reason": "fit-only pilot and fresh confirmation cohorts frozen without media access",
    }

    outdir.mkdir(parents=True, exist_ok=True)
    write_json(outdir / "manifest.json", manifest)
    write_json(outdir / "summary.json", summary)
    write_json(outdir / "decision.json", decision)
    write_json(outdir / "test_lock.json", test_lock)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--tts-meta", type=Path, default=DEFAULT_TTS_META)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--teacher-manifest", type=Path, default=DEFAULT_TEACHER_MANIFEST)
    parser.add_argument("--landmark-dir", type=Path, default=DEFAULT_LANDMARK_DIR)
    parser.add_argument("--strict-dir", type=Path, default=DEFAULT_STRICT_DIR)
    parser.add_argument("--wav2lip-checkpoint", type=Path, default=DEFAULT_WAV2LIP_CHECKPOINT)
    parser.add_argument("--landmarker-asset", type=Path, default=DEFAULT_LANDMARKER_ASSET)
    parser.add_argument("--code-review", type=Path, default=DEFAULT_CODE_REVIEW)
    parser.add_argument(
        "--test-source",
        type=Path,
        default=REPO / "tests/experiments/lrs3_tts_visual_advantage/test_protocol.py",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
