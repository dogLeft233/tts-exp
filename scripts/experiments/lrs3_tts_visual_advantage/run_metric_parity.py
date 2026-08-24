#!/usr/bin/env python3
"""Run the CPU visual comparator on the frozen pilot teacher cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
REPO = MODULE_DIR.parents[2]
sys.path.insert(0, str(MODULE_DIR))

from protocol import _sample_group, file_sha256  # noqa: E402
from video_features import extract_video_features, save_visual_sequence  # noqa: E402
from visual_metrics import pair_metrics  # noqa: E402

SCHEMA_VERSION = 1
STAGE_ID = "01_metric_parity"
MIN_RECORD_COVERAGE = 0.90
MIN_VALID_FRACTION = 0.90
MIN_EXACT_COVERAGE = 0.85
FROZEN_PROTOCOL_MANIFEST_PATH = (
    REPO / "runs/lrs3_tts_visual_advantage_20260824/00_protocol_lock_retry2/manifest.json"
)
FROZEN_PROTOCOL_MANIFEST_SHA256 = "89ee465f981419da48d22c119e08bae45b099479523648e3c87fca7bb0255e79"
FROZEN_TEST_LOCK_PATH = (
    REPO / "runs/lrs3_tts_visual_advantage_20260824/00_protocol_lock_retry2/test_lock.json"
)
FROZEN_TEST_LOCK_SHA256 = "0db2a6da818a51229e106ab9cd15a8bee37c1782822e39b22758e085272d53ba"


def load_json_with_sha256(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value, digest


def load_json(path: Path) -> dict[str, Any]:
    return load_json_with_sha256(path)[0]


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_number(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, dict):
        return {str(key): _json_number(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_number(item) for item in value]
    return value


def _metric_payload(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return _json_number(dict(metrics))


def _teacher_index(teacher_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = teacher_manifest.get("records")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("teacher manifest records must be objects")
    indexed = {str(row.get("sample_id")): row for row in records}
    if len(indexed) != len(records) or any(not key for key in indexed):
        raise ValueError("teacher manifest sample ids must be unique")
    return indexed


def _validate_teacher_record(
    record: Mapping[str, Any],
    teacher: Mapping[str, Any],
    teacher_cache_root: Path,
) -> tuple[dict[str, Path], dict[str, str]]:
    sample_id = str(record["sample_id"])
    for field in (
        "source_group",
        "protocol_split",
        "face",
        "face_sha256",
        "natural_audio",
        "natural_audio_sha256",
        "tts_audio",
        "tts_audio_sha256",
    ):
        if teacher.get(field) != record.get(field):
            raise ValueError(f"teacher/pilot provenance mismatch for {field}: {sample_id}")
    natural_teacher = teacher.get("natural_teacher")
    tts_teacher = teacher.get("tts_teacher")
    if not isinstance(natural_teacher, dict) or not isinstance(tts_teacher, dict):
        raise ValueError(f"teacher arms missing: {sample_id}")
    paths: dict[str, Path] = {}
    expected_hashes: dict[str, str] = {"real": str(record["face_sha256"])}
    for arm, teacher_arm, driver in (
        ("natural", natural_teacher, "natural_driver"),
        ("tts", tts_teacher, "tts_driver"),
    ):
        path = Path(str(teacher_arm.get("video"))).resolve()
        expected_path = (teacher_cache_root / driver / "video" / f"{sample_id}.mp4").resolve()
        if path != expected_path:
            raise ValueError(f"{arm} teacher video provenance mismatch: {sample_id}")
        video_sha256 = str(teacher_arm.get("video_sha256", ""))
        if not video_sha256:
            raise ValueError(f"{arm} teacher video hash is missing: {sample_id}")
        paths[arm] = path
        expected_hashes[arm] = video_sha256
    if paths["natural"] == paths["tts"] or expected_hashes["natural"] == expected_hashes["tts"]:
        raise ValueError(f"natural and TTS teacher arms are not distinct: {sample_id}")
    paths["real"] = Path(str(record["face"])).resolve()
    return paths, expected_hashes


def _verify_video(path: Path, expected_sha256: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} video missing: {path}")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} video hash mismatch: {path}")
    return actual
def _snapshot_file(
    source: Path,
    expected_sha256: str,
    target: Path,
    label: str,
) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"{label} missing: {source}")
    if file_sha256(source) != expected_sha256:
        raise ValueError(f"{label} hash mismatch: {source}")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite snapshot: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if file_sha256(target) != expected_sha256:
        raise ValueError(f"{label} snapshot hash mismatch: {target}")
    return target


def _snapshot_video(
    source: Path,
    expected_sha256: str,
    target: Path,
    label: str,
) -> Path:
    return _snapshot_file(source, expected_sha256, target, f"{label} video")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))


def _sequence_gate(
    real: Any,
    natural: Any,
    tts: Any,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    valid_fraction = {
        "real": float(real.valid.mean()),
        "natural": float(natural.valid.mean()),
        "tts": float(tts.valid.mean()),
    }
    exact = metrics["exact_time"]
    exact_coverage = {
        "natural": float(exact["natural"]["coverage"]),
        "tts": float(exact["tts"]["coverage"]),
    }
    dtw = metrics["visual_dtw"]
    checks = {
        "valid_fraction_at_least_0_90": all(
            value >= MIN_VALID_FRACTION for value in valid_fraction.values()
        ),
        "exact_time_coverage_at_least_0_85": all(
            value >= MIN_EXACT_COVERAGE for value in exact_coverage.values()
        ),
        "finite_primary_metrics": all(
            _finite_number(exact[arm]["distance"])
            for arm in ("natural", "tts")
        ) and _finite_number(exact.get("benefit")),
        "finite_secondary_metrics": all(
            all(_finite_number(dtw[arm].get(field)) for field in ("distance", "coverage", "path_length", "warp_burden"))
            and float(dtw[arm]["coverage"]) > 0.0
            and float(dtw[arm]["path_length"]) > 0.0
            and float(dtw[arm]["warp_burden"]) >= 0.0
            for arm in ("natural", "tts")
        ) and _finite_number(dtw.get("benefit")),
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "valid_fraction": valid_fraction,
        "exact_time_coverage": exact_coverage,
    }


def _assert_bound_artifact(
    protocol_manifest: Mapping[str, Any],
    parent_key: str,
    supplied_path: Path,
    supplied_sha256: str | None = None,
) -> None:
    parents = protocol_manifest.get("parents")
    if not isinstance(parents, dict) or not isinstance(parents.get(parent_key), dict):
        raise ValueError(f"protocol parent binding is missing: {parent_key}")
    binding = parents[parent_key]
    expected_path = Path(str(binding.get("path"))).resolve()
    supplied = supplied_path.resolve()
    if supplied != expected_path:
        raise ValueError(f"{parent_key} path is not the frozen protocol path")
    expected_sha256 = str(binding.get("sha256", ""))
    actual_sha256 = supplied_sha256 or file_sha256(supplied)
    if not expected_sha256 or actual_sha256 != expected_sha256:
        raise ValueError(f"{parent_key} hash does not match the frozen protocol")


def _assert_bound_asset(
    protocol_manifest: Mapping[str, Any],
    asset_key: str,
    supplied_path: Path,
) -> str:
    assets = protocol_manifest.get("assets")
    if not isinstance(assets, dict) or not isinstance(assets.get(asset_key), dict):
        raise ValueError(f"protocol asset binding is missing: {asset_key}")
    binding = assets[asset_key]
    expected_path = Path(str(binding.get("path"))).resolve()
    supplied = supplied_path.resolve()
    if supplied != expected_path:
        raise ValueError(f"{asset_key} path is not the frozen protocol path")
    expected_sha256 = str(binding.get("sha256", ""))
    actual_sha256 = file_sha256(supplied)
    if not expected_sha256 or actual_sha256 != expected_sha256:
        raise ValueError(f"{asset_key} hash does not match the frozen protocol")
    return actual_sha256


def _assert_frozen_protocol_lock(
    protocol_manifest_path: Path,
    protocol_manifest: Mapping[str, Any],
    protocol_manifest_sha256: str,
) -> dict[str, str]:
    supplied_protocol = protocol_manifest_path.resolve()
    if supplied_protocol != FROZEN_PROTOCOL_MANIFEST_PATH.resolve():
        raise ValueError("protocol manifest path is not the frozen Stage 00 manifest")
    if protocol_manifest_sha256 != FROZEN_PROTOCOL_MANIFEST_SHA256:
        raise ValueError("protocol manifest hash does not match the frozen Stage 00 manifest")

    test_lock_path = FROZEN_TEST_LOCK_PATH.resolve()
    test_lock, test_lock_sha256 = load_json_with_sha256(test_lock_path)
    if test_lock_sha256 != FROZEN_TEST_LOCK_SHA256:
        raise ValueError("Stage 00 test lock hash does not match the frozen lock")
    if test_lock.get("status") != "sealed_unvisited":
        raise ValueError("Stage 00 test lock is not sealed_unvisited")
    if test_lock.get("scope") != "internal_dev_validation_and_test_media":
        raise ValueError("Stage 00 test lock scope is unexpected")
    for field in ("media_opened", "derived_features_created", "scores_created"):
        if test_lock.get(field) is not False:
            raise ValueError(f"Stage 00 test lock records top-level access: {field}")
    for split_name in ("fit", "internal_dev", "validation", "test"):
        for field in ("media_opened", "derived_features_created", "scores_created"):
            if test_lock.get(f"{split_name}_{field}") is not False:
                raise ValueError(f"Stage 00 test lock records {split_name} access")

    cohorts = protocol_manifest.get("cohorts")
    split = cohorts.get("split") if isinstance(cohorts, dict) else None
    if not isinstance(split, dict):
        raise ValueError("protocol cohort split is missing")
    expected_counts = {
        "fit_groups": 24,
        "internal_dev_groups": 6,
        "validation_groups": 6,
        "test_groups": 7,
    }
    split_groups: dict[str, set[str]] = {}
    for key, expected_count in expected_counts.items():
        values = split.get(key)
        if (
            not isinstance(values, list)
            or len(values) != expected_count
            or len(set(values)) != expected_count
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise ValueError(f"protocol {key} are not the frozen groups")
        split_groups[key] = set(values)
    if any(
        split_groups[left] & split_groups[right]
        for index, left in enumerate(expected_counts)
        for right in list(expected_counts)[index + 1 :]
    ):
        raise ValueError("protocol cohort groups overlap across sealed splits")
    for split_name in ("internal_dev", "validation", "test"):
        key = f"{split_name}_groups"
        locked_groups = test_lock.get(key)
        if not isinstance(locked_groups, list) or set(locked_groups) != split_groups[key]:
            raise ValueError(f"Stage 00 test lock does not match protocol {key}")
    test_groups = test_lock.get("test_groups")
    if (
        test_lock.get("test_group_count") != 7
        or not isinstance(test_groups, list)
        or len(test_groups) != 7
        or len(set(test_groups)) != 7
    ):
        raise ValueError("Stage 00 test lock test groups are incomplete")
    test_sample_ids = test_lock.get("test_sample_ids")
    if (
        not isinstance(test_sample_ids, list)
        or len(test_sample_ids) != 84
        or len(set(test_sample_ids)) != 84
        or any(
            not isinstance(sample_id, str)
            or not sample_id
            or _sample_group(sample_id) not in split_groups["test_groups"]
            for sample_id in test_sample_ids
        )
    ):
        raise ValueError("Stage 00 test lock test sample membership is invalid")
    parents = protocol_manifest.get("parents")
    source_binding = parents.get("canonical_source_manifest") if isinstance(parents, dict) else None
    if not isinstance(source_binding, dict):
        raise ValueError("canonical source manifest binding is missing")
    source_path = Path(str(source_binding.get("path"))).resolve()
    source_manifest, source_sha256 = load_json_with_sha256(source_path)
    if source_sha256 != str(source_binding.get("sha256", "")):
        raise ValueError("canonical source manifest hash does not match Stage 00")
    if source_manifest.get("manifest_type") != "lrs3_qwen_cloud_tts_n500":
        raise ValueError("unexpected canonical source manifest type")
    source_records = source_manifest.get("records")
    if not isinstance(source_records, list) or any(not isinstance(row, dict) for row in source_records):
        raise ValueError("canonical source manifest records are invalid")
    canonical_test_ids = {
        str(row.get("sample_id"))
        for row in source_records
        if row.get("source_group") in split_groups["test_groups"]
    }
    if canonical_test_ids != set(test_sample_ids):
        raise ValueError("Stage 00 test lock samples do not match canonical source manifest")
    return {
        "protocol_manifest_sha256": FROZEN_PROTOCOL_MANIFEST_SHA256,
        "test_lock_sha256": FROZEN_TEST_LOCK_SHA256,
    }


def _validate_pilot_cohort(protocol_manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cohorts = protocol_manifest.get("cohorts")
    if not isinstance(cohorts, dict):
        raise ValueError("protocol cohorts are missing")
    split = cohorts.get("split")
    if not isinstance(split, dict):
        raise ValueError("protocol cohort split is missing")
    fit_groups = split.get("fit_groups")
    if not isinstance(fit_groups, list) or len(fit_groups) != 24 or len(set(fit_groups)) != 24:
        raise ValueError("protocol fit groups are not the frozen 24 unique groups")
    pilot = cohorts.get("pilot")
    if not isinstance(pilot, dict) or pilot.get("record_count") != 48 or pilot.get("source_group_count") != 24:
        raise ValueError("pilot cohort is not the frozen 48-record cohort")
    records = pilot.get("records")
    if not isinstance(records, list) or len(records) != 48:
        raise ValueError("pilot cohort records are incomplete")
    sample_ids = [str(row.get("sample_id", "")) for row in records if isinstance(row, dict)]
    if len(sample_ids) != 48 or any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != 48:
        raise ValueError("pilot sample ids must be unique")
    group_counts: Counter[str] = Counter()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("pilot records must be objects")
        if row.get("protocol_split") != "fit":
            raise ValueError(f"pilot record is not fit-only: {row.get('sample_id')}")
        source_group = str(row.get("source_group", ""))
        if source_group not in fit_groups:
            raise ValueError(f"pilot record is outside frozen fit groups: {row.get('sample_id')}")
        if _sample_group(str(row.get("sample_id"))) != source_group:
            raise ValueError(f"pilot sample/group mismatch: {row.get('sample_id')}")
        group_counts[source_group] += 1
    if set(group_counts) != set(fit_groups) or set(group_counts.values()) != {2}:
        raise ValueError("pilot cohort must contain exactly two records per fit group")
    return records


def _validate_old_landmark_audit(
    old_audit: Mapping[str, Any],
    teacher_manifest: Mapping[str, Any],
    teacher_manifest_path: Path,
    teacher_manifest_sha256: str,
) -> None:
    if old_audit.get("manifest_type") != "lrs3_landmark_teacher_audit":
        raise ValueError("unexpected old landmark audit type")
    if old_audit.get("run_id") != "02_landmark_teacher_audit_retry2":
        raise ValueError("unexpected old landmark audit run")
    if old_audit.get("status") != "complete" or old_audit.get("decision") != "NO_GO":
        raise ValueError("old landmark audit is not the immutable prior NO_GO")
    if old_audit.get("parent_teacher_manifest_sha256") != teacher_manifest_sha256:
        raise ValueError("old landmark audit teacher parent hash mismatch")
    expected_teacher_path = str(old_audit.get("parent_teacher_manifest", ""))
    if Path(expected_teacher_path).resolve() != teacher_manifest_path.resolve():
        raise ValueError("old landmark audit teacher parent path mismatch")
    rows = old_audit.get("rows")
    if not isinstance(rows, list) or len(rows) != 60 or any(not isinstance(row, dict) for row in rows):
        raise ValueError("old landmark audit rows are incomplete")
    if teacher_manifest.get("sample_count") != 60:
        raise ValueError("teacher manifest sample count is not the immutable 60-record cache")
def _old_direction_map(old_audit: Mapping[str, Any]) -> dict[str, float]:
    rows = old_audit["rows"]
    result: dict[str, float] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        benefit = row.get("teacher_benefit")
        if sample_id and isinstance(benefit, (int, float)) and math.isfinite(float(benefit)):
            result[sample_id] = float(benefit)
    return result


def _compare_old_direction(
    rows: list[Mapping[str, Any]], old_benefit: Mapping[str, float]
) -> dict[str, Any]:
    paired: list[tuple[float, float]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        benefit = row.get("exact_time_benefit")
        old = old_benefit.get(sample_id)
        if isinstance(benefit, (int, float)) and math.isfinite(float(benefit)) and old is not None:
            paired.append((float(old), float(benefit)))
    if not paired:
        return {"overlap_count": 0, "sign_agreement_fraction": None}
    agreement = sum((old >= 0) == (new >= 0) for old, new in paired) / len(paired)
    return {
        "overlap_count": len(paired),
        "sign_agreement_fraction": agreement,
        "old_median_benefit": float(np.median([old for old, _ in paired])),
        "new_exact_time_median_benefit": float(np.median([new for _, new in paired])),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    outdir = args.outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    protocol_manifest, protocol_manifest_sha256 = load_json_with_sha256(args.protocol_manifest.resolve())
    frozen_lock = _assert_frozen_protocol_lock(
        args.protocol_manifest, protocol_manifest, protocol_manifest_sha256
    )
    if protocol_manifest.get("stage_id") != "00_protocol_lock":
        raise ValueError("unexpected protocol manifest stage")
    if protocol_manifest.get("status") != "complete" or protocol_manifest.get("decision") != "GO":
        raise ValueError("protocol lock is not complete GO")
    if protocol_manifest.get("next_allowed_stage") != STAGE_ID:
        raise ValueError("protocol lock does not authorize metric parity")
    records = _validate_pilot_cohort(protocol_manifest)
    teacher_manifest_path = args.teacher_manifest.resolve()
    teacher_manifest, teacher_manifest_sha256 = load_json_with_sha256(teacher_manifest_path)
    _assert_bound_artifact(
        protocol_manifest,
        "teacher_manifest",
        teacher_manifest_path,
        teacher_manifest_sha256,
    )
    if teacher_manifest.get("manifest_type") != "lrs3_wav2lip_natural_tts_teacher_cache":
        raise ValueError("unexpected teacher manifest type")
    if teacher_manifest.get("status") != "complete" or teacher_manifest.get("decision") != "GO":
        raise ValueError("teacher manifest is not complete GO")
    if teacher_manifest.get("syncnet_used_for_selection") is not False:
        raise ValueError("teacher cache was selected with SyncNet")
    protocol_parent = protocol_manifest["parents"]["motion_teacher_protocol_manifest"]
    if teacher_manifest.get("parent_protocol_manifest_sha256") != protocol_parent["sha256"]:
        raise ValueError("teacher manifest is not bound to the frozen parent protocol")
    teachers = _teacher_index(teacher_manifest)
    old_audit_path = args.old_landmark_audit.resolve()
    old_audit, old_audit_sha256 = load_json_with_sha256(old_audit_path)
    _assert_bound_artifact(
        protocol_manifest,
        "old_landmark_audit",
        old_audit_path,
        old_audit_sha256,
    )
    _validate_old_landmark_audit(
        old_audit,
        teacher_manifest,
        teacher_manifest_path,
        teacher_manifest_sha256,
    )
    old_benefit = _old_direction_map(old_audit)
    asset_path = args.landmarker_asset.resolve()
    asset_sha256 = _assert_bound_asset(protocol_manifest, "mediapipe_landmarker", asset_path)
    snapshot_root = outdir / "snapshots"
    asset_snapshot_path = _snapshot_file(
        asset_path,
        asset_sha256,
        snapshot_root / "mediapipe_landmarker.task",
        "MediaPipe landmarker asset",
    )
    feature_root = outdir / "features"
    metric_root = outdir / "records"
    result_rows: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []

    for record in records:
        sample_id = str(record["sample_id"])
        source_group = str(record["source_group"])
        try:
            teacher = teachers.get(sample_id)
            if teacher is None:
                raise ValueError(f"pilot record missing from teacher manifest: {sample_id}")
            paths, expected_hashes = _validate_teacher_record(
                record, teacher, teacher_manifest_path.parent / "teachers"
            )
            snapshot_paths = {
                arm: _snapshot_video(
                    paths[arm],
                    expected_hashes[arm],
                    snapshot_root / sample_id / f"{arm}.mp4",
                    f"{arm} {sample_id}",
                )
                for arm in ("real", "natural", "tts")
            }
            sequences = {
                arm: extract_video_features(snapshot_paths[arm], asset_snapshot_path)
                for arm in ("real", "natural", "tts")
            }
            for arm in ("real", "natural", "tts"):
                if file_sha256(snapshot_paths[arm]) != expected_hashes[arm]:
                    raise ValueError(f"{arm} video snapshot changed during extraction: {sample_id}")
            feature_paths = {}
            for arm, sequence in sequences.items():
                target = feature_root / sample_id / f"{arm}.npz"
                save_visual_sequence(target, sequence)
                feature_paths[arm] = str(target.resolve())
            metrics = _metric_payload(pair_metrics(sequences["real"], sequences["natural"], sequences["tts"]))
            gate = _sequence_gate(sequences["real"], sequences["natural"], sequences["tts"], metrics)
            row = {
                "sample_id": sample_id,
                "source_group": source_group,
                "protocol_split": record["protocol_split"],
                "real_video": str(paths["real"]),
                "natural_teacher_video": str(paths["natural"]),
                "tts_teacher_video": str(paths["tts"]),
                "snapshot_videos": {
                    arm: str(snapshot_paths[arm].resolve())
                    for arm in ("real", "natural", "tts")
                },
                "real_video_sha256": expected_hashes["real"],
                "natural_teacher_video_sha256": expected_hashes["natural"],
                "tts_teacher_video_sha256": expected_hashes["tts"],
                "feature_paths": feature_paths,
                "landmarker_asset_sha256": asset_sha256,
                "metrics": metrics,
                "exact_time_benefit": metrics["exact_time"]["benefit"],
                "visual_dtw_benefit": metrics["visual_dtw"]["benefit"],
                "engineering_gate": gate,
            }
            write_json(metric_root / f"{sample_id}.json", row)
            result_rows.append(row)
            if gate["eligible"]:
                group_counts[source_group] += 1
        except Exception as exc:
            failure = {"sample_id": sample_id, "source_group": source_group, "error": str(exc)}
            failures.append(failure)
            write_json(metric_root / f"{sample_id}.failure.json", failure)

    if file_sha256(asset_snapshot_path) != asset_sha256:
        raise ValueError("MediaPipe landmarker snapshot changed during extraction")
    eligible_rows = [row for row in result_rows if row["engineering_gate"]["eligible"]]
    required_count = math.ceil(len(records) * MIN_RECORD_COVERAGE)
    expected_groups = {
        str(row["source_group"])
        for row in records
    }
    coverage_gate = {
        "record_count": len(records),
        "successful_extraction_count": len(result_rows),
        "eligible_record_count": len(eligible_rows),
        "minimum_eligible_record_count": required_count,
        "eligible_record_fraction": len(eligible_rows) / len(records),
        "fit_group_count": len(expected_groups),
        "groups_with_eligible_record_count": len(group_counts),
        "every_fit_group_has_eligible_record": set(group_counts) == expected_groups,
    }
    engineering_pass = (
        coverage_gate["eligible_record_count"] >= required_count
        and coverage_gate["every_fit_group_has_eligible_record"]
    )
    comparison = _compare_old_direction(eligible_rows, old_benefit)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": "complete",
        "decision": "GO" if engineering_pass else "BLOCKED",
        "scientific_decision": "not_available",
        "next_allowed_stage": "02_fresh_confirmation" if engineering_pass else None,
        "cohort": "pilot_seen_engineering_parity_only",
        "protocol_manifest": str(args.protocol_manifest.resolve()),
        "protocol_manifest_sha256": frozen_lock["protocol_manifest_sha256"],
        "stage00_test_lock": str(FROZEN_TEST_LOCK_PATH.resolve()),
        "stage00_test_lock_sha256": frozen_lock["test_lock_sha256"],
        "teacher_manifest": str(args.teacher_manifest.resolve()),
        "teacher_manifest_sha256": teacher_manifest_sha256,
        "old_landmark_audit": str(args.old_landmark_audit.resolve()),
        "old_landmark_audit_sha256": old_audit_sha256,
        "landmarker_asset": str(args.landmarker_asset.resolve()),
        "landmarker_asset_sha256": asset_sha256,
        "landmarker_asset_snapshot": str(asset_snapshot_path.resolve()),
        "snapshot_root": str(snapshot_root.resolve()),
        "coverage_gate": coverage_gate,
        "failures": failures,
        "old_direction_comparison": comparison,
        "media_access": {
            "pilot_media_opened": True,
            "internal_dev_media_opened": False,
            "validation_media_opened": False,
            "test_media_opened": False,
        },
    }
    decision = {
        "schema_version": SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": "complete",
        "decision": "GO" if engineering_pass else "BLOCKED",
        "scientific_decision": "not_available",
        "next_allowed_stage": summary["next_allowed_stage"],
        "reason": (
            "pilot comparator parity engineering gate passed; fresh fit confirmation is authorized"
            if engineering_pass
            else "pilot comparator parity engineering gate failed; fresh confirmation is blocked"
        ),
        "coverage_gate": coverage_gate,
        "old_direction_comparison": comparison,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_type": "lrs3_tts_visual_advantage_metric_parity",
        "stage_id": STAGE_ID,
        "cohort": "pilot_seen_engineering_parity_only",
        "protocol_manifest_sha256": summary["protocol_manifest_sha256"],
        "stage00_test_lock_sha256": summary["stage00_test_lock_sha256"],
        "teacher_manifest_sha256": summary["teacher_manifest_sha256"],
        "old_landmark_audit_sha256": summary["old_landmark_audit_sha256"],
        "landmarker_asset_sha256": asset_sha256,
        "record_count": len(records),
        "feature_record_count": len(result_rows),
        "eligible_record_count": len(eligible_rows),
        "record_files": sorted(str((metric_root / f"{row['sample_id']}.json").resolve()) for row in result_rows),
    }
    write_json(outdir / "manifest.json", manifest)
    write_json(outdir / "summary.json", summary)
    write_json(outdir / "decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--old-landmark-audit", type=Path, required=True)
    parser.add_argument("--landmarker-asset", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
