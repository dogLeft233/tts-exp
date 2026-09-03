"""Score-independent cohort freezing for the direct-mel probe."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .phase_a import file_sha256, read_json, write_json


SELECTION_SALT = "masked-tts-tfg-probe-v1\0"
EXPECTED_GROUPS = (
    "6vLrreR6YOE",
    "6zVS8HIPUng",
    "73jPh0eRPSY",
    "79tRTivyMSM",
    "6ORDQFh0Byw",
    "6VWPHKABRQA",
    "79zra755WgA",
    "6yR5OUVb2gY",
)


def selection_key(sample_id: str) -> str:
    return hashlib.sha256((SELECTION_SALT + str(sample_id)).encode("utf-8")).hexdigest()


def _source_pool_path(parent: Path, lock: Mapping[str, Any]) -> Path:
    value = lock.get("metadata", {}).get("source_pool")
    if not value:
        raise ValueError("parent lock has no source pool binding")
    path = Path(str(value))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def freeze_cohort(parent: Path, diagnosis_path: Path, output_path: Path) -> dict[str, Any]:
    parent = parent.resolve()
    diagnosis = read_json(diagnosis_path)
    if diagnosis.get("status") != "complete" or diagnosis.get("read_only") is not True:
        raise ValueError("Phase A diagnosis is not complete and read-only")
    if diagnosis.get("loss_validation", {}).get("passed") is not True:
        raise ValueError("Phase A arithmetic validation did not pass")
    lock = read_json(parent / "00_lock/lock.json")
    masks = read_json(parent / "01_masks/mask_manifest.json")
    eval_masks = [row for row in masks["masks"] if row.get("prototype_split") == "evaluation"]
    by_record: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_group: dict[str, list[str]] = defaultdict(list)
    for mask in eval_masks:
        sample_id = str(mask["sample_id"])
        group = str(mask["source_group"])
        by_record[sample_id].append(mask)
        by_group[group].append(sample_id)
    source_pool_path = _source_pool_path(parent, lock)
    source_pool = read_json(source_pool_path)
    if source_pool.get("status") != "complete":
        raise ValueError("source pool is not complete")
    source_rows = {str(row["sample_id"]): row for row in source_pool.get("records", [])}
    lock_rows = {str(row["sample_id"]): row for row in lock.get("records", []) if row.get("prototype_split") == "evaluation"}
    selected: list[dict[str, Any]] = []
    for group in EXPECTED_GROUPS:
        candidates = []
        for sample_id in sorted(set(by_group.get(group, []))):
            candidates.append((-
                len(by_record[sample_id]),
                selection_key(sample_id),
                sample_id,
            ))
        if not candidates:
            raise ValueError(f"missing evaluation records for {group}")
        _, tie_key, sample_id = min(candidates)
        masks_for_record = sorted(by_record[sample_id], key=lambda row: int(row["canonical_index"]))
        source = source_rows.get(sample_id)
        lock_row = lock_rows.get(sample_id)
        if source is None or lock_row is None:
            raise ValueError(f"selected record is missing from frozen bindings: {sample_id}")
        if str(source["source_group"]) != group or str(lock_row["source_group"]) != group:
            raise ValueError(f"source-group mismatch for {sample_id}")
        face = Path(str(source["face"])).resolve()
        natural_audio = Path(str(lock_row["paths"]["natural_audio"])).resolve()
        if not face.is_file() or file_sha256(face) != str(source["face_sha256"]):
            raise ValueError(f"face binding mismatch for {sample_id}")
        if not natural_audio.is_file() or file_sha256(natural_audio) != str(lock_row["sha256"]["natural_audio"]):
            raise ValueError(f"natural audio binding mismatch for {sample_id}")
        if Path(str(source["natural_audio"])).resolve() != natural_audio:
            raise ValueError(f"source-pool natural audio mismatch for {sample_id}")
        selected.append({
            "sample_id": sample_id,
            "source_group": group,
            "evaluation_mask_count": len(masks_for_record),
            "evaluation_mask_sha256": [str(row["mask_sha256"]) for row in masks_for_record],
            "selection_key_sha256": tie_key,
            "selection_rule": "maximum_evaluation_mask_count_then_sha256_salted_sample_id",
            "face": str(face),
            "face_sha256": str(source["face_sha256"]),
            "natural_audio": str(natural_audio),
            "natural_audio_sha256": str(lock_row["sha256"]["natural_audio"]),
            "natural_mel_frames": int(masks_for_record[0]["natural_mel_frames"]),
            "parent_eval_split": str(lock_row["prototype_split"]),
        })
    if len(selected) != 8 or {row["source_group"] for row in selected} != set(EXPECTED_GROUPS):
        raise ValueError("cohort does not contain exactly one record per evaluation group")
    if not {row["source_group"] for row in selected}.issuperset({"6ORDQFh0Byw", "6yR5OUVb2gY"}):
        raise ValueError("token-negative groups are absent from cohort")
    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_masked_tts_direct_mel_tfg_cohort",
        "status": "complete",
        "score_independent": True,
        "selection_salt": SELECTION_SALT,
        "parent_run": str(parent),
        "parent_lock_sha256": file_sha256(parent / "00_lock/lock.json"),
        "parent_mask_manifest_sha256": file_sha256(parent / "01_masks/mask_manifest.json"),
        "phase_a_analysis_sha256": file_sha256(diagnosis_path),
        "source_pool": str(source_pool_path),
        "source_pool_sha256": file_sha256(source_pool_path),
        "groups": list(EXPECTED_GROUPS),
        "record_count": len(selected),
        "records": selected,
        "selection_uses": ["evaluation mask metadata only", "maximum mask count", "salted sample-id hash tie-break"],
        "selection_does_not_use": ["loss", "prediction quality", "Phase-A labels", "Wav2Lip", "SyncNet"],
        "sealed_splits_accessed": False,
    }
    write_json(output_path, manifest)
    return manifest


def freeze_confirmation_cohort(parent: Path, excluded_path: Path, output_path: Path) -> dict[str, Any]:
    parent = parent.resolve()
    excluded = read_json(excluded_path)
    if excluded.get("status") != "complete" or int(excluded.get("record_count", -1)) != 8:
        raise ValueError("first probe cohort is not complete")
    excluded_ids = {str(row["sample_id"]) for row in excluded.get("records", [])}
    if len(excluded_ids) != 8:
        raise ValueError("first probe cohort has duplicate records")
    lock = read_json(parent / "00_lock/lock.json")
    masks = read_json(parent / "01_masks/mask_manifest.json")
    eval_masks = [row for row in masks["masks"] if row.get("prototype_split") == "evaluation"]
    by_record: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_group: dict[str, set[str]] = defaultdict(set)
    for mask in eval_masks:
        sample_id = str(mask["sample_id"])
        group = str(mask["source_group"])
        by_record[sample_id].append(mask)
        by_group[group].add(sample_id)
    source_pool_path = _source_pool_path(parent, lock)
    source_pool = read_json(source_pool_path)
    if source_pool.get("status") != "complete":
        raise ValueError("source pool is not complete")
    source_rows = {str(row["sample_id"]): row for row in source_pool.get("records", [])}
    lock_rows = {str(row["sample_id"]): row for row in lock.get("records", []) if row.get("prototype_split") == "evaluation"}
    selected: list[dict[str, Any]] = []
    for group in EXPECTED_GROUPS:
        candidates = sorted(set(by_group.get(group, set())) - excluded_ids, key=lambda sample_id: (selection_key(sample_id), sample_id))
        if len(candidates) != 2:
            raise ValueError(f"expected two confirmation records for {group}, found {len(candidates)}")
        for sample_id in candidates:
            masks_for_record = sorted(by_record[sample_id], key=lambda row: int(row["canonical_index"]))
            source = source_rows.get(sample_id)
            lock_row = lock_rows.get(sample_id)
            if source is None or lock_row is None:
                raise ValueError(f"confirmation record is missing from frozen bindings: {sample_id}")
            if str(source["source_group"]) != group or str(lock_row["source_group"]) != group:
                raise ValueError(f"source-group mismatch for {sample_id}")
            face = Path(str(source["face"])).resolve()
            natural_audio = Path(str(lock_row["paths"]["natural_audio"])).resolve()
            if not face.is_file() or file_sha256(face) != str(source["face_sha256"]):
                raise ValueError(f"face binding mismatch for {sample_id}")
            if not natural_audio.is_file() or file_sha256(natural_audio) != str(lock_row["sha256"]["natural_audio"]):
                raise ValueError(f"natural audio binding mismatch for {sample_id}")
            if Path(str(source["natural_audio"])).resolve() != natural_audio:
                raise ValueError(f"source-pool natural audio mismatch for {sample_id}")
            selected.append({
                "sample_id": sample_id,
                "source_group": group,
                "evaluation_mask_count": len(masks_for_record),
                "evaluation_mask_sha256": [str(row["mask_sha256"]) for row in masks_for_record],
                "selection_key_sha256": selection_key(sample_id),
                "selection_rule": "all_remaining_records_after_first_probe_exclusion",
                "face": str(face),
                "face_sha256": str(source["face_sha256"]),
                "natural_audio": str(natural_audio),
                "natural_audio_sha256": str(lock_row["sha256"]["natural_audio"]),
                "natural_mel_frames": int(masks_for_record[0]["natural_mel_frames"]),
                "parent_eval_split": str(lock_row["prototype_split"]),
            })
    if len(selected) != 16 or {row["source_group"] for row in selected} != set(EXPECTED_GROUPS):
        raise ValueError("confirmation cohort does not contain 16 records across eight groups")
    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_masked_tts_direct_mel_tfg_confirmation_cohort",
        "status": "complete",
        "score_independent": True,
        "selection_salt": SELECTION_SALT,
        "parent_run": str(parent),
        "parent_lock_sha256": file_sha256(parent / "00_lock/lock.json"),
        "parent_mask_manifest_sha256": file_sha256(parent / "01_masks/mask_manifest.json"),
        "excluded_probe_cohort_sha256": file_sha256(excluded_path),
        "excluded_probe_sample_ids": sorted(excluded_ids),
        "source_pool": str(source_pool_path),
        "source_pool_sha256": file_sha256(source_pool_path),
        "groups": list(EXPECTED_GROUPS),
        "record_count": len(selected),
        "records_per_group": 2,
        "records": selected,
        "selection_uses": ["parent evaluation membership", "exclusion of first probe records", "deterministic group/sample ordering"],
        "selection_does_not_use": ["loss", "prediction quality", "Wav2Lip", "SyncNet"],
        "sealed_splits_accessed": False,
    }
    write_json(output_path, manifest)
    return manifest


__all__ = ["freeze_cohort", "freeze_confirmation_cohort", "selection_key", "EXPECTED_GROUPS", "SELECTION_SALT"]
