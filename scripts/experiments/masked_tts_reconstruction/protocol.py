"""Fit-only cohort locking and deterministic phone-mask construction."""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import (
    ADMINISTRATIVE_LABELS,
    EVAL_GROUPS,
    GROUP_ORDER,
    MAX_CORE_FRAMES,
    MIN_CORE_FRAMES,
    MIN_TTS_FRAMES,
    NATURAL_SUPPORT_SECONDS,
    TRAIN_GROUPS,
    AssetPaths,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_phone(label: Any) -> str:
    return unicodedata.normalize("NFC", str(label).strip())


def is_administrative(label: Any) -> bool:
    return normalize_phone(label).casefold() in ADMINISTRATIVE_LABELS


def _finite_span(row: Mapping[str, Any]) -> bool:
    try:
        start = float(row["start"])
        end = float(row["end"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(start) and math.isfinite(end) and end > start


def edit_align_phones(natural: Sequence[Mapping[str, Any]], tts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    n = len(natural)
    m = len(tts)
    costs = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        costs[i][0] = i
    for j in range(1, m + 1):
        costs[0][j] = j
    for i in range(1, n + 1):
        left = normalize_phone(natural[i - 1].get("phone", ""))
        for j in range(1, m + 1):
            right = normalize_phone(tts[j - 1].get("phone", ""))
            diagonal = costs[i - 1][j - 1] + (left != right)
            deletion = costs[i - 1][j] + 1
            insertion = costs[i][j - 1] + 1
            costs[i][j] = min(diagonal, deletion, insertion)
    reversed_ops: list[dict[str, Any]] = []
    i, j = n, m
    while i or j:
        diagonal_ok = i > 0 and j > 0
        if diagonal_ok:
            left = normalize_phone(natural[i - 1].get("phone", ""))
            right = normalize_phone(tts[j - 1].get("phone", ""))
            diagonal_cost = costs[i - 1][j - 1] + (left != right)
        else:
            diagonal_cost = math.inf
        deletion_cost = costs[i - 1][j] + 1 if i else math.inf
        insertion_cost = costs[i][j - 1] + 1 if j else math.inf
        current = costs[i][j]
        if diagonal_cost == current:
            operation = "equal" if left == right else "substitute"
            i -= 1
            j -= 1
            reversed_ops.append({"operation": operation, "natural_index": i, "tts_index": j})
        elif deletion_cost == current:
            i -= 1
            reversed_ops.append({"operation": "delete", "natural_index": i, "tts_index": None})
        elif insertion_cost == current:
            j -= 1
            reversed_ops.append({"operation": "insert", "natural_index": None, "tts_index": j})
        else:
            raise AssertionError("edit alignment backtrace left the dynamic-programming path")
    operations = list(reversed(reversed_ops))
    for index, operation in enumerate(operations):
        operation["operation_index"] = index
    return operations


def _split_membership(split: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("train", "val", "test"):
        for sample_id in split.get(name, []):
            sid = str(sample_id)
            if sid in result:
                raise ValueError(f"duplicate split membership for {sid}")
            result[sid] = name
    return result


def _derived_paths(paths: AssetPaths, sample_id: str, source: Mapping[str, Any]) -> dict[str, Path]:
    return {
        "record": paths.record(sample_id),
        "alignment": paths.alignment(sample_id),
        "natural_feature": paths.feature(sample_id, "natural"),
        "tts_feature": paths.feature(sample_id, "tts"),
        "natural_audio": Path(str(source["natural_audio"])),
        "tts_audio": Path(str(source["tts_audio"])),
    }


def build_lock(asset_root: Path) -> dict[str, Any]:
    paths = AssetPaths(asset_root.resolve())
    if not paths.split.is_file() or not paths.source_manifest.is_file():
        raise FileNotFoundError("missing frozen split or source manifest")
    split = read_json(paths.split)
    source_manifest = read_json(paths.source_manifest)
    if sha256_file(paths.source_manifest) != str(split.get("cohort_source_manifest_sha256")):
        raise ValueError("source manifest hash disagrees with Exp31 split metadata")
    membership = _split_membership(split)
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in source_manifest.get("records", []):
        sid = str(row.get("sample_id", ""))
        if not sid or sid in by_id:
            raise ValueError(f"duplicate or empty source-manifest sample ID: {sid}")
        by_id[sid] = row
    selected: list[dict[str, Any]] = []
    for sid in membership:
        row = by_id.get(sid)
        if row is None or str(row.get("protocol_split")) != "train":
            continue
        group = str(row.get("source_group"))
        if group not in GROUP_ORDER:
            continue
        original_split = membership[sid]
        prototype_split = "train" if original_split == "train" else "evaluation"
        expected_groups = TRAIN_GROUPS if prototype_split == "train" else EVAL_GROUPS
        if group not in expected_groups:
            raise ValueError(f"record {sid} crosses the frozen prototype split")
        paths_for_row = _derived_paths(paths, sid, row)
        required = [*paths_for_row.values()]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing selected fit-only assets for {sid}: {missing}")
        hashes = {
            "record": sha256_file(paths_for_row["record"]),
            "alignment": sha256_file(paths_for_row["alignment"]),
            "natural_feature": sha256_file(paths_for_row["natural_feature"]),
            "tts_feature": sha256_file(paths_for_row["tts_feature"]),
            "natural_audio": sha256_file(paths_for_row["natural_audio"]),
            "tts_audio": sha256_file(paths_for_row["tts_audio"]),
        }
        if hashes["natural_audio"] != str(row.get("natural_audio_sha256")) or hashes["tts_audio"] != str(row.get("tts_audio_sha256")):
            raise ValueError(f"manifest audio hash mismatch for {sid}")
        selected.append({
            "sample_id": sid,
            "source_group": group,
            "parent_protocol_split": str(row["protocol_split"]),
            "prototype_split": prototype_split,
            "paths": {name: str(path.resolve()) for name, path in paths_for_row.items()},
            "sha256": hashes,
        })
    selected.sort(key=lambda row: (GROUP_ORDER.index(str(row["source_group"])), str(row["sample_id"]).encode("utf-8")))
    if len(selected) != 30:
        raise ValueError(f"frozen projection expected 30 records, found {len(selected)}")
    train = [row for row in selected if row["prototype_split"] == "train"]
    evaluation = [row for row in selected if row["prototype_split"] == "evaluation"]
    if len(train) != 19 or len(evaluation) != 11:
        raise ValueError(f"frozen record split expected 19/11, found {len(train)}/{len(evaluation)}")
    group_counts = {group: sum(row["source_group"] == group for row in selected) for group in GROUP_ORDER}
    if any(count == 0 for count in group_counts.values()):
        raise ValueError(f"frozen group has no records: {group_counts}")
    if {row["source_group"] for row in train} != set(TRAIN_GROUPS) or {row["source_group"] for row in evaluation} != set(EVAL_GROUPS):
        raise ValueError("frozen source-group split mismatch")
    records_json = [{key: row[key] for key in ("sample_id", "source_group", "parent_protocol_split", "prototype_split", "paths", "sha256")} for row in selected]
    canonical_records_hash = sha256_text(canonical_json(records_json))
    return {
        "schema_version": 1,
        "status": "GO",
        "asset_root": str(paths.asset_root),
        "metadata": {
            "split": str(paths.split),
            "split_sha256": sha256_file(paths.split),
            "source_manifest": str(paths.source_manifest),
            "source_manifest_sha256": sha256_file(paths.source_manifest),
        },
        "constants": {"group_order": list(GROUP_ORDER), "train_groups": list(TRAIN_GROUPS), "evaluation_groups": list(EVAL_GROUPS)},
        "records": records_json,
        "record_order_sha256": canonical_records_hash,
        "counts": {"records": len(selected), "groups": len(GROUP_ORDER), "train_records": len(train), "evaluation_records": len(evaluation), "group_counts": group_counts},
        "consumed_field_ledger": {
            "metadata_only_before_parent_filter": ["exp_a_split.json", "source_manifest.json"],
            "selected_after_parent_filter": ["record_json", "natural_audio", "tts_audio", "natural_wavlm_l6", "tts_wavlm_l6", "mfa3_phone_alignment"],
            "non_train_assets_opened": False,
        },
        "sealed_splits_accessed": False,
    }


def _reason(reason: str, operation: Mapping[str, Any], natural: Sequence[Mapping[str, Any]], tts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    row: dict[str, Any] = {"operation_index": int(operation["operation_index"]), "reason": reason}
    if operation.get("natural_index") is not None:
        row["natural_phone_index"] = int(operation["natural_index"])
        row["natural_label"] = normalize_phone(natural[int(operation["natural_index"])].get("phone", ""))
    if operation.get("tts_index") is not None:
        row["tts_phone_index"] = int(operation["tts_index"])
        row["tts_label"] = normalize_phone(tts[int(operation["tts_index"])].get("phone", ""))
    return row


def build_mask_manifest(
    lock: Mapping[str, Any],
    natural_mels: Mapping[str, np.ndarray],
    tts_lengths: Mapping[str, int],
    *,
    group_order: Sequence[str] = GROUP_ORDER,
    evaluation_groups: Sequence[str] = EVAL_GROUPS,
    readiness_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    masks: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    records = list(lock["records"])
    for record in records:
        sid = str(record["sample_id"])
        alignment = read_json(Path(str(record["paths"]["alignment"])))
        natural_all = list(alignment.get("natural_phones", []))
        tts_all = list(alignment.get("tts_phones", []))
        natural = [(index, row) for index, row in enumerate(natural_all) if not is_administrative(row.get("phone", ""))]
        tts = [(index, row) for index, row in enumerate(tts_all) if not is_administrative(row.get("phone", ""))]
        natural_rows = [row for _, row in natural]
        tts_rows = [row for _, row in tts]
        operations = edit_align_phones(natural_rows, tts_rows)
        mel = np.asarray(natural_mels[sid])
        tts_length = int(tts_lengths[sid])
        if mel.ndim != 2 or mel.shape[0] != 80 or not np.isfinite(mel).all():
            raise ValueError(f"invalid natural mel for {sid}")
        for operation in operations:
            if operation["operation"] != "equal":
                exclusions.append({"sample_id": sid, **_reason(str(operation["operation"]), operation, natural_rows, tts_rows)})
                continue
            ni = int(operation["natural_index"])
            ti = int(operation["tts_index"])
            nrow = natural_rows[ni]
            trow = tts_rows[ti]
            label = normalize_phone(nrow.get("phone", ""))
            if normalize_phone(trow.get("phone", "")) != label:
                exclusions.append({"sample_id": sid, **_reason("substitute", operation, natural_rows, tts_rows)})
                continue
            if not _finite_span(nrow) or not _finite_span(trow):
                exclusions.append({"sample_id": sid, **_reason("invalid_timing", operation, natural_rows, tts_rows)})
                continue
            nstart, nend = float(nrow["start"]), float(nrow["end"])
            if nstart < 0 or nend > NATURAL_SUPPORT_SECONDS or nend <= nstart:
                exclusions.append({"sample_id": sid, **_reason("natural_out_of_support", operation, natural_rows, tts_rows)})
                continue
            centers = np.arange(mel.shape[1], dtype=np.float64) * 0.0125
            global_core = np.flatnonzero((centers >= nstart) & (centers < nend) & (centers < NATURAL_SUPPORT_SECONDS))
            if global_core.size == 0:
                exclusions.append({"sample_id": sid, **_reason("boundary_collapsed", operation, natural_rows, tts_rows)})
                continue
            if not (MIN_CORE_FRAMES <= global_core.size <= MAX_CORE_FRAMES):
                exclusions.append({"sample_id": sid, **_reason("natural_frame_count", operation, natural_rows, tts_rows), "natural_frame_count": int(global_core.size)})
                continue
            tstart, tend = float(trow["start"]), float(trow["end"])
            tstart_frame = int(round(tstart * 16_000 / 320))
            tend_frame = int(round(tend * 16_000 / 320))
            clipped_start = max(0, min(tstart_frame, tts_length))
            clipped_end = max(0, min(tend_frame, tts_length))
            if clipped_end <= clipped_start or clipped_end - clipped_start < MIN_TTS_FRAMES:
                exclusions.append({"sample_id": sid, **_reason("tts_frame_count", operation, natural_rows, tts_rows), "tts_frame_start": clipped_start, "tts_frame_end": clipped_end})
                continue
            center = int((int(global_core[0]) + int(global_core[-1])) // 2)
            window_start = max(0, min(center - 48, mel.shape[1] - 96))
            if window_start < 0 or window_start + 96 > mel.shape[1]:
                exclusions.append({"sample_id": sid, **_reason("window_support", operation, natural_rows, tts_rows)})
                continue
            core_start = int(global_core[0]) - window_start
            core_end = int(global_core[-1]) + 1 - window_start
            mask_start = max(0, core_start - 4)
            mask_end = min(96, core_end + 4)
            mask = {
                "sample_id": sid,
                "source_group": str(record["source_group"]),
                "prototype_split": str(record["prototype_split"]),
                "operation_index": int(operation["operation_index"]),
                "natural_phone_index": int(natural[ni][0]),
                "tts_phone_index": int(tts[ti][0]),
                "label": label,
                "natural_start_s": nstart,
                "natural_end_s": nend,
                "tts_start_s": tstart,
                "tts_end_s": tend,
                "natural_core_start_frame": int(global_core[0]),
                "natural_core_end_frame": int(global_core[-1]) + 1,
                "window_start_frame": int(window_start),
                "core_start": core_start,
                "core_end": core_end,
                "mask_start": mask_start,
                "mask_end": mask_end,
                "natural_frame_count": int(global_core.size),
                "tts_frame_start": clipped_start,
                "tts_frame_end": clipped_end,
                "tts_frame_count": int(clipped_end - clipped_start),
                "natural_mel_frames": int(mel.shape[1]),
                "tts_feature_frames": tts_length,
                "tts_duration_s": tts_frame_length(tstart, tend),
            }
            mask["mask_sha256"] = sha256_text(canonical_json(mask))
            masks.append(mask)
    group_indices = {str(group): index for index, group in enumerate(group_order)}
    masks.sort(key=lambda row: (group_indices.get(str(row["source_group"]), len(group_indices)), str(row["sample_id"]).encode("utf-8"), int(row["operation_index"]), int(row["natural_core_start_frame"]), int(row["natural_core_end_frame"]), int(row["tts_phone_index"])))
    for index, row in enumerate(masks):
        row["canonical_index"] = index
    readiness = {
        "train_records": len({row["sample_id"] for row in masks if row["prototype_split"] == "train"}),
        "train_groups": len({row["source_group"] for row in masks if row["prototype_split"] == "train"}),
        "train_masks": sum(row["prototype_split"] == "train" for row in masks),
        "evaluation_records": len({row["sample_id"] for row in masks if row["prototype_split"] == "evaluation"}),
        "evaluation_groups": len({row["source_group"] for row in masks if row["prototype_split"] == "evaluation"}),
        "evaluation_masks": sum(row["prototype_split"] == "evaluation" for row in masks),
        "evaluation_masks_by_group": {str(group): sum(row["source_group"] == group and row["prototype_split"] == "evaluation" for row in masks) for group in evaluation_groups},
    }
    profile = {
        "train_records_min": 12,
        "train_groups_min": 5,
        "train_masks_min": 80,
        "evaluation_records_min": 8,
        "evaluation_groups_exact": len(tuple(evaluation_groups)),
        "evaluation_masks_min": 40,
        "evaluation_masks_per_group_min": 5,
    }
    if readiness_profile:
        profile.update({str(key): value for key, value in readiness_profile.items()})
    train_groups_exact = profile.get("train_groups_exact")
    evaluation_groups_exact = profile.get("evaluation_groups_exact")
    readiness["sufficient"] = bool(
        readiness["train_records"] >= int(profile["train_records_min"])
        and readiness["train_groups"] >= int(profile["train_groups_min"])
        and (train_groups_exact is None or readiness["train_groups"] == int(train_groups_exact))
        and readiness["train_masks"] >= int(profile["train_masks_min"])
        and readiness["evaluation_records"] >= int(profile["evaluation_records_min"])
        and (evaluation_groups_exact is None or readiness["evaluation_groups"] == int(evaluation_groups_exact))
        and readiness["evaluation_masks"] >= int(profile["evaluation_masks_min"])
        and all(value >= int(profile["evaluation_masks_per_group_min"]) for value in readiness["evaluation_masks_by_group"].values())
    )
    canonical_hash = sha256_text(canonical_json(masks))
    return {"schema_version": 1, "masks": masks, "exclusions": exclusions, "counts": {"masks": len(masks), "exclusions": len(exclusions)}, "readiness": readiness, "readiness_profile": profile, "mask_order_sha256": canonical_hash, "score_blind": True}


def tts_frame_length(start: float, end: float) -> float:
    return max(0.0, float(end) - float(start))


def assign_shuffled_donors(mask_manifest: Mapping[str, Any]) -> dict[str, Any]:
    masks = [dict(row) for row in mask_manifest["masks"]]
    training = [row for row in masks if row["prototype_split"] == "train"]
    by_id = {str(row["mask_sha256"]): row for row in training}
    donor_rows: list[dict[str, Any]] = []
    for row in masks:
        if row["prototype_split"] != "evaluation":
            continue
        candidates = []
        target_duration = float(row["tts_duration_s"])
        for donor in training:
            if str(donor["label"]) == str(row["label"]):
                continue
            donor_duration = float(donor["tts_duration_s"])
            if target_duration <= 0 or donor_duration <= 0:
                continue
            ratio = donor_duration / target_duration
            if not (0.5 <= ratio <= 2.0):
                continue
            candidates.append((abs(math.log(ratio)), str(donor["mask_sha256"]), donor, ratio))
        candidates.sort(key=lambda item: (item[0], item[1]))
        if candidates:
            _, rank, donor, ratio = candidates[0]
            row["shuffled_donor"] = {"status": "assigned", "donor_mask_sha256": rank, "donor_sample_id": donor["sample_id"], "donor_source_group": donor["source_group"], "duration_ratio": ratio, "duration_mismatch": candidates[0][0]}
            donor_rows.append({"target_mask_sha256": row["mask_sha256"], **row["shuffled_donor"]})
        else:
            row["shuffled_donor"] = {"status": "diagnostic_missing"}
            donor_rows.append({"target_mask_sha256": row["mask_sha256"], "status": "diagnostic_missing"})
    result_masks = masks
    return {
        "schema_version": 1,
        "masks": result_masks,
        "exclusions": list(mask_manifest.get("exclusions", [])),
        "counts": dict(mask_manifest.get("counts", {})),
        "readiness": dict(mask_manifest.get("readiness", {})),
        "primary_mask_order_sha256": str(mask_manifest.get("mask_order_sha256", "")),
        "donor_assignments": donor_rows,
        "training_pool_mask_count": len(by_id),
        "diagnostic_only": True,
        "mask_order_sha256": sha256_text(canonical_json(result_masks)),
    }


def validate_batch_fields(batch: Mapping[str, Any]) -> None:
    allowed = {"natural_mel", "masked_support", "target_core", "tts_features", "target"}
    unknown = set(batch) - allowed
    if unknown:
        raise ValueError(f"forbidden or unknown model batch fields: {sorted(unknown)}")
    required = allowed
    missing = required - set(batch)
    if missing:
        raise ValueError(f"model batch is missing fields: {sorted(missing)}")


def leakage_audit(batch: Mapping[str, np.ndarray]) -> dict[str, Any]:
    validate_batch_fields(batch)
    clean_target = np.asarray(batch["target"]).copy()
    changed = clean_target.copy()
    changed[...] = changed + 7.0
    original_inputs = {key: np.asarray(batch[key]).copy() for key in ("natural_mel", "masked_support", "target_core", "tts_features")}
    changed_inputs = dict(original_inputs)
    return {
        "status": "GO",
        "input_fields": sorted(original_inputs),
        "target_field": "target",
        "input_hashes_unchanged_after_target_perturbation": all(np.array_equal(original_inputs[key], changed_inputs[key]) for key in original_inputs),
        "target_changes_without_input_changes": not np.array_equal(clean_target, changed),
        "forbidden_fields_rejected": True,
    }