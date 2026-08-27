from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROTOCOL_ID = "lrs3_mfa_linear_replacement_20260824"
STAGE_ID = "00_protocol_lock_retry1"
REPO = Path(__file__).resolve().parents[3]
EXPECTED_RECORDS = 24


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


def load_json(path: str | Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {target}")
    assert_finite_json(payload)
    return payload


def write_json(path: str | Path, value: Mapping[str, Any]) -> str:
    assert_finite_json(value)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return file_sha256(target)


def _sample_group(sample_id: str) -> str:
    parts = str(sample_id).split("_")
    if len(parts) != 3 or parts[0] != "lrs3" or not parts[1] or not parts[2].isdigit():
        raise ValueError(f"invalid LRS3 sample id: {sample_id!r}")
    return parts[1]


def _unique(values: Sequence[Any], name: str) -> list[str]:
    result = [str(value) for value in values]
    if not result or any(not value for value in result) or len(result) != len(set(result)):
        raise ValueError(f"{name} must be non-empty and unique")
    return result


def validate_test_lock(test_lock: Mapping[str, Any], canonical_records: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    if test_lock.get("status") != "sealed_unvisited":
        raise ValueError("test lock must be sealed_unvisited")
    required_flags = ("media_opened", "derived_features_created", "scores_created")
    for key in required_flags:
        if test_lock.get(key) is not False:
            raise ValueError(f"test lock {key} must be false")
    split_flags = (
        "test_media_opened", "test_derived_features_created", "test_scores_created",
        "validation_media_opened", "validation_derived_features_created", "validation_scores_created",
        "internal_dev_media_opened", "internal_dev_derived_features_created", "internal_dev_scores_created",
    )
    if any(key in test_lock for key in split_flags):
        for key in split_flags:
            if test_lock.get(key) is not False:
                raise ValueError(f"test lock {key} must be false")
    groups = _unique(test_lock.get("test_groups", []), "test_groups")
    sample_ids = _unique(test_lock.get("test_sample_ids", []), "test_sample_ids")
    if test_lock.get("test_group_count") != len(groups):
        raise ValueError("test_group_count disagrees with test_groups")
    if any(_sample_group(sample_id) not in groups for sample_id in sample_ids):
        raise ValueError("test sample is outside locked groups")
    if canonical_records is not None:
        source_ids = {str(row.get("sample_id")): str(row.get("source_group")) for row in canonical_records}
        if len(source_ids) != len(canonical_records) or any(not sample_id for sample_id in source_ids):
            raise ValueError("canonical source sample ids must be unique and non-empty")
        expected = {sample_id for sample_id, group in source_ids.items() if group in set(groups)}
        if set(sample_ids) != expected:
            raise ValueError("test lock does not match canonical membership")
    return {
        "test_groups": groups,
        "test_sample_ids": sample_ids,
        "validation_groups": [str(value) for value in test_lock["validation_groups"]] if "validation_groups" in test_lock else None,
        "internal_dev_groups": [str(value) for value in test_lock["internal_dev_groups"]] if "internal_dev_groups" in test_lock else None,
    }


def validate_fresh_cohort(
    records: Sequence[Mapping[str, Any]],
    *,
    teacher_sample_ids: Sequence[str] = (),
    expected_count: int = EXPECTED_RECORDS,
) -> list[dict[str, Any]]:
    if len(records) != expected_count:
        raise ValueError(f"fresh cohort must contain exactly {expected_count} records")
    seen_ids: set[str] = set()
    seen_groups: set[str] = set()
    result: list[dict[str, Any]] = []
    teacher_ids = {str(value) for value in teacher_sample_ids}
    required = ("sample_id", "source_group", "protocol_split", "face", "face_sha256", "natural_audio", "natural_audio_sha256", "tts_audio", "tts_audio_sha256")
    for row in records:
        missing = [key for key in required if not row.get(key)]
        if missing:
            raise ValueError(f"cohort record missing fields: {missing}")
        sample_id = str(row["sample_id"])
        group = str(row["source_group"])
        if sample_id in seen_ids or group in seen_groups:
            raise ValueError("fresh cohort must have unique sample ids and one record per source group")
        if _sample_group(sample_id) != group:
            raise ValueError(f"sample/group mismatch: {sample_id}")
        if row["protocol_split"] != "fit":
            raise ValueError(f"fresh cohort is not fit-only: {sample_id}")
        if sample_id in teacher_ids:
            raise ValueError(f"fresh cohort overlaps teacher cache: {sample_id}")
        seen_ids.add(sample_id)
        seen_groups.add(group)
        result.append({key: row[key] for key in required})
    return result


def validate_cross_manifest_joins(
    cohort: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    tts_meta: Mapping[str, Any],
) -> None:
    source_rows = source_manifest.get("records")
    tts_rows = tts_meta.get("results")
    if not isinstance(source_rows, list) or not isinstance(tts_rows, Mapping):
        raise ValueError("canonical source/TTS manifests are incomplete")
    source_by_id = {str(row.get("sample_id")): row for row in source_rows}
    for row in cohort:
        sample_id = str(row["sample_id"])
        source = source_by_id.get(sample_id)
        tts = tts_rows.get(sample_id)
        if not isinstance(source, Mapping) or not isinstance(tts, Mapping):
            raise ValueError(f"missing source/TTS join: {sample_id}")
        checks = {
            "source_group": source.get("source_group"),
            "face_sha256": source.get("video_sha256"),
            "natural_audio_sha256": source.get("natural_audio_sha256"),
            "tts_audio_sha256": tts.get("canonical_audio_sha256"),
        }
        for key, expected in checks.items():
            if row.get(key) != expected:
                raise ValueError(f"{key} mismatch for {sample_id}")
        if tts.get("source_group") != source.get("source_group") or tts.get("video_sha256") != source.get("video_sha256"):
            raise ValueError(f"TTS/source metadata mismatch for {sample_id}")
        if tts.get("reference_audio_sha256") != source.get("natural_audio_sha256"):
            raise ValueError(f"TTS reference mismatch for {sample_id}")
        if tts.get("transcript") != source.get("transcript") or tts.get("tts_transcript") != source.get("tts_transcript"):
            raise ValueError(f"transcript mismatch for {sample_id}")


def _path_hashes(paths: Mapping[str, str | Path]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, value in paths.items():
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            stored_path = str(path.relative_to(REPO))
        except ValueError:
            stored_path = str(path)
        result[name] = {"path": stored_path, "sha256": file_sha256(path)}
    return result


def _runtime() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "environment_variable_names": sorted(key for key in __import__("os").environ if "KEY" not in key.upper() and "TOKEN" not in key.upper() and "PASSWORD" not in key.upper()),
    }


def derive_stage_test_lock(parent_lock: Mapping[str, Any]) -> dict[str, Any]:
    locked = validate_test_lock(parent_lock)
    return {
        "status": "sealed_unvisited",
        "scope": "internal_dev_validation_and_test_media",
        "media_opened": False,
        "derived_features_created": False,
        "scores_created": False,
        "test_group_count": len(locked["test_groups"]),
        "test_groups": locked["test_groups"],
        "test_sample_ids": locked["test_sample_ids"],
        "test_media_opened": False,
        "test_derived_features_created": False,
        "test_scores_created": False,
        "validation_groups": list(parent_lock.get("validation_groups", [])),
        "validation_media_opened": False,
        "validation_derived_features_created": False,
        "validation_scores_created": False,
        "internal_dev_groups": list(parent_lock.get("internal_dev_groups", [])),
        "internal_dev_media_opened": False,
        "internal_dev_derived_features_created": False,
        "internal_dev_scores_created": False,
        "fit_media_opened": False,
        "fit_derived_features_created": False,
        "fit_scores_created": False,
    }


def build_stage00_artifacts(
    *,
    parent_visual_manifest: Mapping[str, Any],
    parent_visual_lock: Mapping[str, Any],
    parent_motion_manifest: Mapping[str, Any],
    parent_motion_lock: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    tts_meta: Mapping[str, Any],
    asset_paths: Mapping[str, str | Path],
    source_paths: Mapping[str, str | Path],
    test_paths: Mapping[str, str | Path],
    parent_files: Mapping[str, str | Path] | None = None,
    review_files: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    if parent_visual_manifest.get("status") != "complete" or parent_visual_manifest.get("decision") != "GO":
        raise ValueError("parent visual Stage00 must be complete GO")
    if parent_visual_manifest.get("manifest_type") != "lrs3_tts_visual_advantage_protocol_lock":
        raise ValueError("unexpected parent visual manifest type")
    if parent_motion_manifest.get("status") != "complete" or parent_motion_manifest.get("decision") != "GO":
        raise ValueError("parent motion-teacher audit must be complete GO")
    if parent_motion_manifest.get("manifest_type") != "lrs3_motion_teacher_protocol_audit":
        raise ValueError("unexpected parent motion-teacher manifest type")
    if parent_motion_manifest.get("protocol", {}).get("manifest_type") != "lrs3_motion_teacher_protocol":
        raise ValueError("motion parent protocol identity mismatch")
    motion_protocol = parent_motion_manifest.get("protocol")
    visual_split = parent_visual_manifest.get("cohorts", {}).get("split")
    motion_split = motion_protocol.get("split") if isinstance(motion_protocol, Mapping) else None
    if not isinstance(motion_protocol, Mapping) or not isinstance(motion_split, Mapping) or not isinstance(visual_split, Mapping):
        raise ValueError("parent manifests are missing split provenance")
    for split_key in ("fit_groups", "internal_dev_groups", "validation_groups", "test_groups"):
        if list(visual_split.get(split_key, [])) != list(motion_split.get(split_key, [])):
            raise ValueError(f"parent manifests disagree on {split_key}")
    visual_locked = validate_test_lock(parent_visual_lock, source_manifest.get("records"))
    if list(parent_visual_lock.get("validation_groups", [])) != list(visual_split.get("validation_groups", [])):
        raise ValueError("visual lock validation groups disagree with parent manifest")
    if list(parent_visual_lock.get("internal_dev_groups", [])) != list(visual_split.get("internal_dev_groups", [])):
        raise ValueError("visual lock internal-dev groups disagree with parent manifest")
    if list(parent_visual_lock.get("test_groups", [])) != list(visual_split.get("test_groups", [])):
        raise ValueError("visual lock test groups disagree with parent manifest")
    motion_locked = validate_test_lock(parent_motion_lock, source_manifest.get("records"))
    if visual_locked["test_groups"] != motion_locked["test_groups"] or visual_locked["test_sample_ids"] != motion_locked["test_sample_ids"]:
        raise ValueError("visual and motion parent test locks disagree")
    for split_key in ("validation_groups", "internal_dev_groups"):
        motion_value = motion_locked[split_key]
        if motion_value is not None and motion_value != visual_locked[split_key]:
            raise ValueError(f"visual and motion parent locks disagree on {split_key}")
    fresh = parent_visual_manifest.get("cohorts", {}).get("fresh_confirmation", {})
    pilot = parent_visual_manifest.get("cohorts", {}).get("pilot", {})
    fresh_records = validate_fresh_cohort(fresh.get("records", []), teacher_sample_ids=[row.get("sample_id") for row in pilot.get("records", [])])
    validate_cross_manifest_joins(fresh_records, source_manifest, tts_meta)
    stage_lock = derive_stage_test_lock(parent_visual_lock)
    cohort_hash = canonical_sha256(fresh_records)
    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa_linear_replacement_protocol_lock",
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "complete",
        "decision": "GO",
        "scientific_decision": "not_available",
        "parents": {
            "visual_manifest_sha256": canonical_sha256(parent_visual_manifest),
            "visual_test_lock_sha256": canonical_sha256(parent_visual_lock),
            "motion_manifest_sha256": canonical_sha256(parent_motion_manifest),
            "motion_test_lock_sha256": canonical_sha256(parent_motion_lock),
            "source_manifest_sha256": canonical_sha256(source_manifest),
            "tts_metadata_sha256": canonical_sha256(tts_meta),
            "parent_files": _path_hashes(parent_files or {}),
        },
        "cohort": {
            "selection": "existing ordered parent fresh_confirmation records",
            "record_count": len(fresh_records),
            "source_group_count": len({row["source_group"] for row in fresh_records}),
            "records": fresh_records,
            "ordered_records_sha256": cohort_hash,
            "score_based_substitution": False,
            "sealed_splits_unvisited": True,
        },
        "endpoints": {
            "authoritative_baseline": "G_N_E_N",
            "authoritative_target": "G_M_E_N",
            "diagnostics": ["G_N_E_M", "G_M_E_M"],
            "benefit_C": "SyncC(G_M_E_N)-SyncC(G_N_E_N)",
            "benefit_D": "SyncD(G_N_E_N)-SyncD(G_M_E_N)",
            "sync_c_higher_is_better": True,
            "sync_d_lower_is_better": True,
        },
        "candidate_contract": {
            "natural_sample_rate_hz": 16000,
            "natural_channels": 1,
            "wavlm_layer": 6,
            "wavlm_feature_dim": 1024,
            "wavlm_hop_samples": 320,
            "prematched_vocoder": True,
            "loudness_normalization": False,
            "speech_fallback_frames_required": 0,
            "silence_fallback_requires_tts_silence": True,
            "exact_length_policy": "right_crop_or_right_zero_pad_only",
        },
        "strict_mux_contract": {
            "container": "matroska",
            "video": "copy",
            "audio_codec": "pcm_s16le",
            "audio_sample_rate_hz": 16000,
            "audio_channels": 1,
            "forbidden": ["atempo", "-shortest", "-t", "crop", "pad", "filter", "video_reencode"],
        },
        "statistics": {
            "method": "exact_one_sided_paired_sign_flip_on_mean",
            "alpha": 0.05,
            "bootstrap_draws": 10000,
            "intersection_union": True,
            "post_hoc_effect_gate": False,
        },
        "assets": _path_hashes(asset_paths),
        "source_manifest": _path_hashes(source_paths),
        "test_manifest": _path_hashes(test_paths),
        "code_review": _path_hashes(review_files or {}),
        "media_access": {
            "fit_media_opened": False,
            "internal_dev_media_opened": False,
            "validation_media_opened": False,
            "test_media_opened": False,
            "mfa_run": False,
            "features_created": False,
            "scores_created": False,
        },
        "runtime": _runtime(),
    }
    summary = {
        "schema_version": 1,
        "stage_id": STAGE_ID,
        "status": "complete",
        "decision": "GO",
        "scientific_decision": "not_available",
        "record_count": len(fresh_records),
        "source_group_count": len({row["source_group"] for row in fresh_records}),
        "ordered_records_sha256": cohort_hash,
        "media_access": manifest["media_access"],
    }
    decision = {
        "schema_version": 1,
        "stage_id": STAGE_ID,
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "next_allowed_stage": "01_candidate_audio_retry1",
        "reason": "CPU-only protocol lock is complete; no media, MFA, features, or scores were accessed.",
    }
    return {"manifest": manifest, "test_lock": stage_lock, "summary": summary, "decision": decision}


def write_stage00_artifacts(output_dir: str | Path, artifacts: Mapping[str, Mapping[str, Any]]) -> None:
    target = Path(output_dir)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Stage00 output directory is non-empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    for name in ("manifest", "test_lock", "summary", "decision"):
        write_json(target / f"{name}.json", artifacts[name])


def _load_map(paths: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in paths:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"path must be NAME=PATH: {value}")
        result[name] = Path(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-visual-manifest", required=True)
    parser.add_argument("--parent-visual-lock", required=True)
    parser.add_argument("--parent-motion-manifest", required=True)
    parser.add_argument("--parent-motion-lock", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--tts-meta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asset", action="append", default=[])
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--test", action="append", default=[])
    parser.add_argument("--parent-file", action="append", default=[])
    parser.add_argument("--review", action="append", default=[])
    args = parser.parse_args()
    artifacts = build_stage00_artifacts(
        parent_visual_manifest=load_json(args.parent_visual_manifest),
        parent_visual_lock=load_json(args.parent_visual_lock),
        parent_motion_manifest=load_json(args.parent_motion_manifest),
        parent_motion_lock=load_json(args.parent_motion_lock),
        source_manifest=load_json(args.source_manifest),
        tts_meta=load_json(args.tts_meta),
        asset_paths=_load_map(args.asset),
        source_paths=_load_map(args.source),
        test_paths=_load_map(args.test),
        parent_files=_load_map(args.parent_file),
        review_files=_load_map(args.review),
    )
    write_stage00_artifacts(args.output, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
