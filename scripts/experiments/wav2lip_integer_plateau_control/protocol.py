from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_roi_retiming_oracle import media as parent_media

from . import config
from .common import (
    PlateauError,
    bytes_sha256,
    file_sha256,
    read_pcm16_wav,
    require_hash,
    verify_self_hashed_json,
)


def _fixed_json(path: Path, expected: str, label: str) -> dict[str, Any]:
    require_hash(path, expected, label)
    return verify_self_hashed_json(path)


def _index(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> dict[tuple[str, ...], Mapping[str, Any]]:
    result: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(key_name)) for key_name in keys)
        if key in result:
            raise PlateauError(f"duplicate manifest key: {key}")
        result[key] = row
    return result


def load_frozen_inputs() -> dict[str, Any]:
    oracle_final = _fixed_json(config.ORACLE_FINAL, config.ORACLE_HASHES["final"], "oracle final")
    oracle_validation = _fixed_json(config.ORACLE_VALIDATION, config.ORACLE_HASHES["validation"], "oracle validation")
    oracle_protocol = _fixed_json(config.ORACLE_PROTOCOL, config.ORACLE_HASHES["protocol"], "oracle protocol")
    roi_final = _fixed_json(config.ROI_FINAL, config.ROI_HASHES["final"], "ROI final")
    roi_validation = _fixed_json(config.ROI_VALIDATION, config.ROI_HASHES["validation"], "ROI validation")
    require_hash(config.ROI_PROTOCOL, config.ROI_HASHES["protocol"], "ROI protocol")
    roi_protocol = verify_self_hashed_json(config.ROI_PROTOCOL)
    linear_final = _fixed_json(config.LINEAR_FINAL, config.LINEAR_HASHES["final"], "linear oracle final")
    linear_validation = _fixed_json(config.LINEAR_VALIDATION, config.LINEAR_HASHES["validation"], "linear oracle validation")
    linear_protocol = _fixed_json(config.LINEAR_PROTOCOL, config.LINEAR_HASHES["protocol"], "linear oracle protocol")

    if oracle_final.get("status") != "complete" or not (oracle_validation.get("valid") is True or oracle_validation.get("status") == "valid"):
        raise PlateauError("frozen oracle parent is not complete and independently valid")
    if roi_final.get("status") != "complete" or not (roi_validation.get("valid") is True or roi_validation.get("status") == "valid"):
        raise PlateauError("frozen ROI parent is not complete and independently valid")
    if len(oracle_protocol.get("records", [])) != config.EXPECTED_RECORD_COUNT:
        raise PlateauError("oracle parent record count changed")
    if len(roi_protocol.get("records", [])) != config.EXPECTED_RECORD_COUNT:
        raise PlateauError("ROI parent record count changed")
    if oracle_final.get("protocol_sha256") != config.ORACLE_HASHES["protocol"]:
        raise PlateauError("oracle final does not bind the frozen oracle protocol")
    if roi_final.get("protocol_sha256") != config.ROI_HASHES["protocol"]:
        raise PlateauError("ROI final does not bind the frozen ROI protocol")

    frame_manifest = verify_self_hashed_json(config.ORACLE_FRAMES)
    media_manifest = verify_self_hashed_json(config.ORACLE_MEDIA)
    score_manifest = verify_self_hashed_json(config.ORACLE_SCORES)
    if frame_manifest.get("protocol_sha256") != config.ORACLE_HASHES["protocol"]:
        raise PlateauError("oracle frame manifest is bound to a different protocol")
    if media_manifest.get("protocol_sha256") != config.ORACLE_HASHES["protocol"]:
        raise PlateauError("oracle media manifest is bound to a different protocol")
    if score_manifest.get("protocol_sha256") != config.ORACLE_HASHES["protocol"]:
        raise PlateauError("oracle score manifest is bound to a different protocol")
    records = oracle_protocol.get("records")
    if not isinstance(records, list):
        raise PlateauError("oracle protocol records are missing")
    ids = [str(row.get("sample_id")) for row in records if isinstance(row, Mapping)]
    if len(ids) != config.EXPECTED_RECORD_COUNT or hashlib.sha256("\n".join(ids).encode()).hexdigest() != config.ORDERED_SAMPLE_ID_SHA256:
        raise PlateauError("ordered sample-id hash changed")
    return {
        "oracle_final": oracle_final,
        "oracle_validation": oracle_validation,
        "oracle_protocol": oracle_protocol,
        "oracle_frames": frame_manifest,
        "oracle_media": media_manifest,
        "oracle_scores": score_manifest,
        "roi_final": roi_final,
        "roi_validation": roi_validation,
        "roi_protocol": roi_protocol,
        "linear_final": linear_final,
        "linear_validation": linear_validation,
        "linear_protocol": linear_protocol,
    }


def _support_for_row(row: int, *, F: int, L: int, m: int, b: int, direction: int) -> dict[str, Any]:
    if direction not in (-1, 1):
        raise PlateauError(f"support direction must be -1 or +1, got {direction}")
    columns: list[dict[str, Any]] = []
    for column in range(2 * config.VSHIFT + 1):
        mfcc_row = (row + column - config.VSHIFT) * config.MFCC_STRIDE
        first = mfcc_row * config.MFCC_FRAME_STEP - config.MFCC_PREEMPHASIS_PREDECESSOR
        last = (mfcc_row + config.MFCC_AUDIO_FRAMES - 1) * config.MFCC_FRAME_STEP + config.MFCC_FRAME_LENGTH - 1
        columns.append(
            {
                "column": column,
                "audio_row": int(row + column - config.VSHIFT),
                "mfcc_start": int(mfcc_row),
                "mfcc_end": int(mfcc_row + config.MFCC_AUDIO_FRAMES - 1),
                "sample_start": int(first),
                "sample_end": int(last),
                "legal": bool(0 <= first <= last < L),
            }
        )
    sample_start = min(item["sample_start"] for item in columns)
    sample_end = max(item["sample_end"] for item in columns)
    source_start = sample_start + direction * config.AUDIO_SHIFT
    source_end = sample_end + direction * config.AUDIO_SHIFT
    target_before = sample_end < b
    target_after = sample_start >= b
    source_before = source_end < b
    source_after = source_start >= b
    output_frames = list(range(row, row + config.WINDOW_FRAMES))
    source_frames = [index + direction * config.FRAME_SHIFT for index in output_frames]
    return {
        "row": int(row),
        "direction": int(direction),
        "output_frames": output_frames,
        "source_frames": source_frames,
        "source_frame_range_legal": bool(min(source_frames) >= 0 and max(source_frames) < F),
        "columns": columns,
        "all_31_columns_legal": bool(all(item["legal"] for item in columns)),
        "sample_range": [int(sample_start), int(sample_end)],
        "mapped_source_sample_range": [int(source_start), int(source_end)],
        "target_platform_single": bool(target_before or target_after),
        "mapped_source_platform_single": bool(source_before or source_after),
        "same_platform": bool((target_before and source_before) or (target_after and source_after)),
        "support_legal": bool(
            all(item["legal"] for item in columns)
            and min(source_frames) >= 0
            and max(source_frames) < F
            and ((target_before and source_before) or (target_after and source_after))
        ),
    }


def _build_row(
    record: Mapping[str, Any],
    media_row: Mapping[str, Any],
    frame_row: Mapping[str, Any],
    cached_score: Mapping[str, Any],
) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    F = int(record["frame_count"])
    m = F // 2
    L = int(record["sample_count"])
    b = config.SAMPLES_PER_FRAME * m
    if F <= 2 * config.FRAME_SHIFT + 50 or m <= config.FRAME_SHIFT or L <= b + config.AUDIO_SHIFT:
        raise PlateauError(f"source is too short for the fixed plateau: {sample_id}")
    stream = media_row.get("streams", {}).get(config.VIDEO_ID)
    if not isinstance(stream, Mapping):
        raise PlateauError(f"V_ID stream is missing: {sample_id}")
    stream_path = Path(str(stream["output"])).resolve()
    require_hash(stream_path, str(stream["output_sha256"]), f"V_ID stream {sample_id}")
    frames, timeline = parent_media.audit_parent_stream(stream_path, F)
    natural_path = Path(str(record["natural_audio"]["path"])).resolve()
    require_hash(natural_path, str(record["natural_audio"]["container_sha256"]), f"natural audio {sample_id}")
    pcm_bytes, pcm, params = read_pcm16_wav(natural_path)
    if pcm.size != L or bytes_sha256(pcm_bytes) != str(record["natural_audio"]["decoded_pcm_sha256"]):
        raise PlateauError(f"natural PCM binding changed: {sample_id}")
    video_shift = np.where(np.arange(F) < m, config.FRAME_SHIFT, -config.FRAME_SHIFT).astype(np.int64)
    q = np.arange(F, dtype=np.int64) + video_shift
    audio_shift = np.where(np.arange(L) < b, config.AUDIO_SHIFT, -config.AUDIO_SHIFT).astype(np.int64)
    audio_source_indices = np.arange(L, dtype=np.int64) + audio_shift
    if int(q.min()) < 0 or int(q.max()) >= F or int(audio_source_indices.min()) < 0 or int(audio_source_indices.max()) >= L:
        raise PlateauError(f"plateau source indices are illegal: {sample_id}")
    plus = np.arange(25, m - 25, dtype=np.int64)
    minus = np.arange(m + 25, F - 25, dtype=np.int64)
    U = np.concatenate((plus, minus))
    Q = U + video_shift[U]
    if plus.size < 5 or minus.size < 5:
        raise PlateauError(f"plateau segment has fewer than five rows: {sample_id}")
    support = [_support_for_row(int(row), F=F, L=L, m=m, b=b, direction=1 if int(video_shift[row]) > 0 else -1) for row in U]
    if not all(item["support_legal"] for item in support):
        bad = next(item for item in support if not item["support_legal"])
        raise PlateauError(f"SyncNet support crosses an illegal boundary: {sample_id}/{bad['row']}")
    cached_matrix = Path(str(cached_score.get("matrix", ""))).resolve()
    require_hash(cached_matrix, str(cached_score.get("matrix_sha256")), f"cached V_ID/N matrix {sample_id}")
    return {
        "sample_id": sample_id,
        "source_group": str(record["source_group"]),
        "frame_count": F,
        "sample_count": L,
        "break_frame": m,
        "break_sample": b,
        "natural_audio": {
            "path": str(natural_path),
            "container_sha256": file_sha256(natural_path),
            "decoded_pcm_sha256": bytes_sha256(pcm_bytes),
            "sample_count": L,
            "sample_rate": int(params["sample_rate"]),
        },
        "parent_stream": {
            "path": str(stream_path),
            "sha256": file_sha256(stream_path),
            "frame_count": F,
            "timeline": timeline,
        },
        "parent_v_id_n": dict(cached_score),
        "q": [int(value) for value in q],
        "video_shift_rule": "+5 before floor(F/2), -5 at/after floor(F/2)",
        "audio_shift_rule": "+3200 before 640*floor(F/2), -3200 at/after that sample",
        "audio_source_index_sha256": bytes_sha256(np.asarray(audio_source_indices, dtype="<i8").tobytes()),
        "audio_source_index_length": L,
        "u": [int(value) for value in U],
        "q_rows": [int(value) for value in Q],
        "plus_rows": [int(value) for value in plus],
        "minus_rows": [int(value) for value in minus],
        "support": support,
        "source_frame_sha256": [bytes_sha256(frame.tobytes()) for frame in frames],
        "source_pixels_sha256": bytes_sha256(np.ascontiguousarray(frames).tobytes()),
        "q_sha256": bytes_sha256(np.asarray(q, dtype="<i8").tobytes()),
        "_frames": frames,
        "_pcm": pcm,
        "_pcm_bytes": pcm_bytes,
        "_q": q,
        "_audio_source_indices": audio_source_indices,
        "_cached_matrix": np.asarray(np.load(cached_matrix, allow_pickle=False), dtype=np.float32),
    }


def build_parent_rows(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    protocol = inputs["oracle_protocol"]
    media_index = _index(inputs["oracle_media"]["rows"], ("sample_id",))
    frame_index = _index(
        [row for row in inputs["oracle_frames"]["rows"] if str(row.get("video_arm")) == config.VIDEO_ID],
        ("sample_id",),
    )
    score_index = _index(
        [row for row in inputs["oracle_scores"]["scores"] if str(row.get("video_arm")) == config.VIDEO_ID and str(row.get("audio_arm")) == config.AUDIO_N],
        ("sample_id",),
    )
    rows: list[dict[str, Any]] = []
    for record in protocol["records"]:
        sid = str(record["sample_id"])
        if (sid,) not in media_index or (sid,) not in frame_index or (sid,) not in score_index:
            raise PlateauError(f"parent V_ID/N assets are incomplete: {sid}")
        media_row = media_index[(sid,)]
        frame_row = frame_index[(sid,)]
        cached = score_index[(sid,)]
        if int(record["frame_count"]) != int(media_row["streams"][config.VIDEO_ID]["frame_count"]):
            raise PlateauError(f"parent frame count differs between protocol and media: {sid}")
        if str(frame_row.get("evidence", {}).get("output_sha256")) != str(media_row["streams"][config.VIDEO_ID]["output_sha256"]):
            raise PlateauError(f"parent frame manifest differs from media manifest: {sid}")
        row = _build_row(record, media_row, frame_row, cached)
        rows.append(row)
    return rows


def public_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in row.items() if not key.startswith("_")}


def build_input_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": "wav2lip_integer_plateau_control",
        "stage_id": "input_audit",
        "status": "complete",
        "classification": "seen_fit_diagnostic",
        "record_count": len(rows),
        "source_group_count": len({str(row["source_group"]) for row in rows}),
        "new_generated_videos": 0,
        "new_score_cells": 0,
        "records": [
            {
                "sample_id": row["sample_id"],
                "source_group": row["source_group"],
                "frame_count": row["frame_count"],
                "sample_count": row["sample_count"],
                "source_stream": row["parent_stream"],
                "natural_audio": row["natural_audio"],
                "q_sha256": row["q_sha256"],
                "audio_source_index_sha256": row["audio_source_index_sha256"],
                "plus_rows": row["plus_rows"],
                "minus_rows": row["minus_rows"],
                "support_rows_checked": len(row["support"]),
                "support_all_valid": all(item["support_legal"] for item in row["support"]),
                "parent_cached_matrix": {
                    "path": row["parent_v_id_n"]["matrix"],
                    "sha256": row["parent_v_id_n"]["matrix_sha256"],
                },
            }
            for row in rows
        ],
    }


def build_protocol(
    run_id: str,
    rows: Sequence[Mapping[str, Any]],
    input_audit_sha256: str,
    change_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    selected_cells: list[dict[str, Any]] = []
    for row in rows:
        for video, audio in (*config.FRESH_CELL_SPECS, config.CACHED_CELL_SPEC):
            selected_cells.append(
                {
                    "sample_id": row["sample_id"],
                    "source_group": row["source_group"],
                    "video_arm": video,
                    "audio_arm": audio,
                    "fresh": (video, audio) in config.FRESH_CELL_SPECS,
                    "media": None if (video, audio) in config.FRESH_CELL_SPECS else row["parent_v_id_n"]["media"],
                    "media_sha256": None if (video, audio) in config.FRESH_CELL_SPECS else row["parent_v_id_n"]["media_sha256"],
                    "matrix": None if (video, audio) in config.FRESH_CELL_SPECS else row["parent_v_id_n"]["matrix"],
                    "matrix_sha256": None if (video, audio) in config.FRESH_CELL_SPECS else row["parent_v_id_n"]["matrix_sha256"],
                }
            )
    return {
        "schema_version": 1,
        "protocol_id": "wav2lip_integer_plateau_control",
        "protocol_revision": "integer_plateau_v1",
        "run_id": run_id,
        "status": "frozen",
        "classification": "seen_fit_diagnostic",
        "parent": {
            "oracle_root": str(config.ORACLE_ROOT.resolve()),
            "oracle_fixed_hashes": dict(config.ORACLE_HASHES),
            "roi_final": str(config.ROI_FINAL.resolve()),
            "roi_fixed_hashes": dict(config.ROI_HASHES),
            "linear_fixed_hashes": dict(config.LINEAR_HASHES),
            "historical_scientific_decision": "CONTROL_FAILED",
            "record_count": config.EXPECTED_RECORD_COUNT,
            "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
        },
        "change_bindings": dict(change_bindings),
        "frozen_config": config.FrozenConfig().to_dict(),
        "environment": config.environment(),
        "input_audit_sha256": input_audit_sha256,
        "records": [public_record(row) for row in rows],
        "selected_cells": selected_cells,
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "expected_source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
        "expected_new_video_count_stage_a": config.EXPECTED_RECORD_COUNT,
        "expected_fresh_score_cell_count_stage_a": config.EXPECTED_RECORD_COUNT * len(config.FRESH_CELL_SPECS),
        "expected_cached_score_cell_count_stage_a": config.EXPECTED_RECORD_COUNT,
        "expected_new_video_count_stage_b": config.EXPECTED_RECORD_COUNT,
        "expected_fresh_score_cell_count_stage_b": config.EXPECTED_RECORD_COUNT * 2,
    }
