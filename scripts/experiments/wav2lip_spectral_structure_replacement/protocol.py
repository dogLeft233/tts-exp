from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    ExperimentError,
    bytes_sha256,
    file_sha256,
    read_pcm16_wav,
    require_hash,
    verify_self_hashed_json,
)


def _fixed_json(path: Path, expected: str, label: str) -> dict[str, Any]:
    require_hash(path, expected, label)
    return verify_self_hashed_json(path)


def _index(rows: Sequence[Mapping[str, Any]], key: str = "sample_id") -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value or value in result:
            raise ExperimentError(f"duplicate or empty {key}: {value}")
        result[value] = row
    return result


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (config.REPO / path).resolve()


def _asset(path_value: Any, expected: Any, label: str) -> Path:
    path = _resolve(path_value)
    require_hash(path, str(expected), label)
    return path


def _audio_meta(path: Path, expected_container: str | None, label: str) -> tuple[dict[str, Any], bytes, np.ndarray]:
    if expected_container:
        require_hash(path, expected_container, f"{label} container")
    pcm, values, params = read_pcm16_wav(path)
    return (
        {
            "path": str(path),
            "container_sha256": file_sha256(path),
            "decoded_pcm_sha256": bytes_sha256(pcm),
            "sample_count": int(values.size),
            "sample_rate": int(params["sample_rate"]),
            "channels": int(params["channels"]),
            "sample_width": int(params["sample_width"]),
        },
        pcm,
        values,
    )


def _support_for_row(row: int, *, frame_count: int, sample_count: int, break_sample: int, direction: int) -> dict[str, Any]:
    if direction not in (-1, 1):
        raise ExperimentError("support direction must be -1 or +1")
    columns: list[dict[str, Any]] = []
    for column in range(config.MATRIX_COLUMNS):
        audio_row = row + column - config.VSHIFT
        mfcc_row = audio_row * 4
        first = mfcc_row * 160 - 1
        last = (mfcc_row + 20 - 1) * 160 + 400 - 1
        columns.append(
            {
                "column": column,
                "audio_row": int(audio_row),
                "sample_start": int(first),
                "sample_end": int(last),
                "legal": bool(0 <= first <= last < sample_count),
            }
        )
    sample_start = min(item["sample_start"] for item in columns)
    sample_end = max(item["sample_end"] for item in columns)
    source_start = sample_start + direction * 3200
    source_end = sample_end + direction * 3200
    target_before = sample_end < break_sample
    target_after = sample_start >= break_sample
    source_before = source_end < break_sample
    source_after = source_start >= break_sample
    source_frames = [row + index * direction for index in range(config.WINDOW_FRAMES)]
    return {
        "row": int(row),
        "direction": int(direction),
        "all_31_columns_legal": bool(all(item["legal"] for item in columns)),
        "source_frame_range_legal": bool(min(source_frames) >= 0 and max(source_frames) < frame_count),
        "same_platform": bool((target_before and source_before) or (target_after and source_after)),
        "columns": columns,
        "sample_range": [int(sample_start), int(sample_end)],
        "mapped_source_sample_range": [int(source_start), int(source_end)],
        "support_legal": bool(
            all(item["legal"] for item in columns)
            and min(source_frames) >= 0
            and max(source_frames) < frame_count
            and ((target_before and source_before) or (target_after and source_after))
        ),
    }


def load_frozen_inputs() -> dict[str, Any]:
    plateau_final = _fixed_json(config.PLATEAU_FINAL, config.PLATEAU_FINAL_SHA256, "plateau final")
    plateau_protocol = _fixed_json(config.PLATEAU_PROTOCOL, config.PLATEAU_PROTOCOL_SHA256, "plateau protocol")
    plateau_validation = _fixed_json(config.PLATEAU_GENERATED_VALIDATION, config.PLATEAU_GENERATED_VALIDATION_SHA256, "plateau validation")
    plateau_audio = _fixed_json(config.PLATEAU_AUDIO_MANIFEST, config.PLATEAU_AUDIO_MANIFEST_SHA256, "plateau audio manifest")
    confirmation_protocol = _fixed_json(config.CONFIRMATION_PROTOCOL, config.CONFIRMATION_PROTOCOL_SHA256, "confirmation protocol")
    cohort = _fixed_json(config.CONFIRMATION_COHORT, config.CONFIRMATION_COHORT_SHA256, "confirmation cohort")
    roi_protocol = _fixed_json(config.ROI_PROTOCOL, config.ROI_PROTOCOL_SHA256, "ROI protocol")
    roi_manifest = _fixed_json(config.ROI_MANIFEST, config.ROI_MANIFEST_SHA256, "ROI manifest")

    if plateau_final.get("status") != "complete" or plateau_validation.get("valid") is not True:
        raise ExperimentError("plateau parent is not complete and independently valid")
    if plateau_final.get("protocol_sha256") != config.PLATEAU_PROTOCOL_SHA256:
        raise ExperimentError("plateau final does not bind the frozen protocol")
    records = plateau_protocol.get("records")
    cohort_records = cohort.get("records")
    roi_records = roi_protocol.get("records")
    roi_rows = roi_manifest.get("rows")
    audio_rows = plateau_audio.get("rows")
    if not all(isinstance(value, list) for value in (records, cohort_records, roi_records, roi_rows, audio_rows)):
        raise ExperimentError("one or more frozen manifests have no rows")
    if len(records) != config.EXPECTED_RECORD_COUNT or len(cohort_records) != config.EXPECTED_RECORD_COUNT or len(roi_records) != config.EXPECTED_RECORD_COUNT:
        raise ExperimentError("frozen record count is not 22")
    if [str(row.get("sample_id")) for row in records] != [str(row.get("sample_id")) for row in cohort_records]:
        raise ExperimentError("plateau and confirmation order differs")
    ids = [str(row.get("sample_id")) for row in records]
    groups = [str(row.get("source_group")) for row in records]
    if hashlib.sha256("\n".join(ids).encode()).hexdigest() != config.FrozenConfig().ordered_sample_id_sha256:
        raise ExperimentError("ordered sample-id hash changed")
    if len(set(ids)) != config.EXPECTED_RECORD_COUNT or len(set(groups)) != config.EXPECTED_SOURCE_GROUP_COUNT:
        raise ExperimentError("sample IDs or source groups are not unique")

    cohort_index = _index(cohort_records)
    roi_index = _index(roi_records)
    roi_manifest_index = _index(roi_rows)
    plateau_audio_index = _index(audio_rows)
    prepared: list[dict[str, Any]] = []
    for plateau in records:
        sid = str(plateau["sample_id"])
        group = str(plateau["source_group"])
        cohort_row = cohort_index.get(sid)
        roi_row = roi_index.get(sid)
        manifest_row = roi_manifest_index.get(sid)
        audio_row = plateau_audio_index.get(sid)
        if not all(isinstance(value, Mapping) for value in (cohort_row, roi_row, manifest_row, audio_row)):
            raise ExperimentError(f"sample_id join is incomplete: {sid}")
        if any(str(value.get("source_group")) != group for value in (cohort_row, roi_row, manifest_row, audio_row)):
            raise ExperimentError(f"source_group join differs: {sid}")
        natural = cohort_row.get("natural_audio")
        mfa = cohort_row.get("mfa_linear_audio")
        face = roi_row.get("face_video")
        roi = roi_row.get("roi")
        p_arm = audio_row.get("arms", {}).get(config.ARM_P) if isinstance(audio_row.get("arms"), Mapping) else None
        if not all(isinstance(value, Mapping) for value in (natural, mfa, face, roi, p_arm)):
            raise ExperimentError(f"source assets are incomplete: {sid}")
        if natural.get("path") != plateau.get("natural_audio", {}).get("path") or natural.get("sha256") != plateau.get("natural_audio", {}).get("container_sha256"):
            raise ExperimentError(f"natural audio binding differs: {sid}")
        if face.get("path") != manifest_row.get("source_video") or face.get("sha256") != manifest_row.get("source_video_sha256"):
            raise ExperimentError(f"ROI source-video binding differs: {sid}")
        if roi.get("boxes_path") != manifest_row.get("boxes_path") or roi.get("boxes_sha256") != manifest_row.get("boxes_sha256"):
            raise ExperimentError(f"ROI boxes binding differs: {sid}")
        natural_path = _asset(natural.get("path"), natural.get("sha256"), f"natural audio {sid}")
        mfa_path = _asset(mfa.get("path"), mfa.get("sha256"), f"MFA-linear audio {sid}")
        face_path = _asset(face.get("path"), face.get("sha256"), f"face video {sid}")
        boxes_path = _asset(roi.get("boxes_path"), roi.get("boxes_sha256"), f"ROI boxes {sid}")
        p_path = _asset(p_arm.get("path"), p_arm.get("container_sha256"), f"plateau P audio {sid}")
        natural_meta, natural_pcm, natural_values = _audio_meta(natural_path, str(natural.get("sha256")), f"natural audio {sid}")
        mfa_meta, mfa_pcm, mfa_values = _audio_meta(mfa_path, str(mfa.get("sha256")), f"MFA-linear audio {sid}")
        p_meta, p_pcm, p_values = _audio_meta(p_path, str(p_arm.get("container_sha256")), f"plateau P audio {sid}")
        expected_samples = int(plateau["sample_count"])
        if natural_values.size != expected_samples or mfa_values.size != expected_samples or p_values.size != expected_samples:
            raise ExperimentError(f"audio length mismatch: {sid}")
        frame_count = int(plateau["frame_count"])
        break_frame = int(plateau["break_frame"])
        break_sample = int(plateau["break_sample"])
        plus = [int(value) for value in plateau["plus_rows"]]
        minus = [int(value) for value in plateau["minus_rows"]]
        U = [*plus, *minus]
        support = [
            _support_for_row(row, frame_count=frame_count, sample_count=expected_samples, break_sample=break_sample, direction=1 if row < break_frame else -1)
            for row in U
        ]
        if len(U) != len(set(U)) or not all(item["support_legal"] for item in support):
            raise ExperimentError(f"U support audit failed: {sid}")
        prepared.append(
            {
                "sample_id": sid,
                "source_group": group,
                "frame_count": frame_count,
                "sample_count": expected_samples,
                "break_frame": break_frame,
                "break_sample": break_sample,
                "natural_audio": natural_meta,
                "mfa_linear_audio": mfa_meta,
                "plateau_audio": {**p_meta, "construction": "frozen plateau P audio"},
                "face_video": {"path": str(face_path), "sha256": file_sha256(face_path)},
                "roi": {
                    "boxes_path": str(boxes_path),
                    "boxes_sha256": file_sha256(boxes_path),
                    "frame_count": int(roi.get("frame_count", 0)),
                },
                "parent_stream": dict(plateau.get("parent_stream", {})),
                "u": U,
                "plus_rows": plus,
                "minus_rows": minus,
                "support": support,
                "_natural_pcm": natural_pcm,
                "_mfa_pcm": mfa_pcm,
                "_p_pcm": p_pcm,
            }
        )
    return {
        "plateau_final": plateau_final,
        "plateau_protocol": plateau_protocol,
        "plateau_validation": plateau_validation,
        "plateau_audio": plateau_audio,
        "confirmation_protocol": confirmation_protocol,
        "cohort": cohort,
        "roi_protocol": roi_protocol,
        "roi_manifest": roi_manifest,
        "rows": prepared,
    }


def public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def build_input_audit(inputs: Mapping[str, Any]) -> dict[str, Any]:
    rows = inputs["rows"]
    return {
        "schema_version": 1,
        "stage_id": "input_audit",
        "protocol_id": "wav2lip_spectral_structure_replacement",
        "status": "complete",
        "classification": "seen_fit_mechanism_exploration",
        "record_count": len(rows),
        "source_group_count": len({str(row["source_group"]) for row in rows}),
        "join_key": "sample_id",
        "u_source": "plateau protocol records; U=PLUS||MINUS",
        "records": [
            {
                "sample_id": row["sample_id"],
                "source_group": row["source_group"],
                "frame_count": row["frame_count"],
                "sample_count": row["sample_count"],
                "natural_audio": row["natural_audio"],
                "mfa_linear_audio": row["mfa_linear_audio"],
                "plateau_audio": row["plateau_audio"],
                "face_video": row["face_video"],
                "roi": row["roi"],
                "u_count": len(row["u"]),
                "plus_count": len(row["plus_rows"]),
                "minus_count": len(row["minus_rows"]),
                "support_rows_checked": len(row["support"]),
                "support_all_valid": all(item["support_legal"] for item in row["support"]),
            }
            for row in rows
        ],
    }


def build_protocol(run_id: str, inputs: Mapping[str, Any], audit_sha256: str, bindings: Mapping[str, Any]) -> dict[str, Any]:
    rows = inputs["rows"]
    selected_cells = []
    for row in rows:
        for video_arm in config.STAGE_A_ARMS:
            selected_cells.append({"sample_id": row["sample_id"], "source_group": row["source_group"], "video_arm": video_arm, "audio_arm": config.ARM_N, "stage": "A"})
        selected_cells.append({"sample_id": row["sample_id"], "source_group": row["source_group"], "video_arm": config.ARM_N, "audio_arm": config.ARM_P, "stage": "A"})
        for video_arm in config.STAGE_B_ARMS:
            selected_cells.append({"sample_id": row["sample_id"], "source_group": row["source_group"], "video_arm": video_arm, "audio_arm": config.ARM_N, "stage": "B"})
    return {
        "schema_version": 1,
        "protocol_id": "wav2lip_spectral_structure_replacement",
        "protocol_revision": "spectral_structure_v1",
        "run_id": run_id,
        "status": "frozen",
        "classification": "seen_fit_mechanism_exploration",
        "parent": {
            "plateau_final": str(config.PLATEAU_FINAL.resolve()),
            "plateau_final_sha256": config.PLATEAU_FINAL_SHA256,
            "plateau_protocol": str(config.PLATEAU_PROTOCOL.resolve()),
            "plateau_protocol_sha256": config.PLATEAU_PROTOCOL_SHA256,
            "confirmation_protocol": str(config.CONFIRMATION_PROTOCOL.resolve()),
            "confirmation_protocol_sha256": config.CONFIRMATION_PROTOCOL_SHA256,
            "confirmation_cohort": str(config.CONFIRMATION_COHORT.resolve()),
            "confirmation_cohort_sha256": config.CONFIRMATION_COHORT_SHA256,
            "roi_protocol": str(config.ROI_PROTOCOL.resolve()),
            "roi_protocol_sha256": config.ROI_PROTOCOL_SHA256,
            "roi_manifest": str(config.ROI_MANIFEST.resolve()),
            "roi_manifest_sha256": config.ROI_MANIFEST_SHA256,
        },
        "change_bindings": dict(bindings),
        "frozen_config": config.FrozenConfig().to_dict(),
        "environment": config.environment(),
        "input_audit_sha256": audit_sha256,
        "records": [public_row(row) for row in rows],
        "selected_cells": selected_cells,
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "expected_source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
        "expected_stage_a_video_count": config.EXPECTED_RECORD_COUNT * len(config.STAGE_A_ARMS),
        "expected_stage_a_cell_count": config.EXPECTED_RECORD_COUNT * 4,
        "expected_stage_b_video_count": config.EXPECTED_RECORD_COUNT * len(config.STAGE_B_ARMS),
        "expected_stage_b_cell_count": config.EXPECTED_RECORD_COUNT * len(config.STAGE_B_ARMS),
        "legacy_bridge_executed": False,
    }
