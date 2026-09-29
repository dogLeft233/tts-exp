from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import config
from .common import (
    assert_not_sealed,
    assert_run_root_compatible,
    canonical_json_sha256,
    file_sha256,
    read_json,
    sample_ids_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)


class ProtocolError(RuntimeError):
    pass


def _parent_binding(path: Path, expected: str, name: str) -> dict[str, str]:
    if not path.is_file():
        raise ProtocolError(f"parent artifact missing: {name}: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ProtocolError(f"parent artifact hash changed: {name}")
    return {"path": str(path.resolve()), "sha256": actual}


def _asset_binding(value: Any, name: str, sample_id: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"malformed {name} binding: {sample_id}")
    path = Path(str(value.get("path", ""))).resolve()
    expected = str(value.get("sha256", ""))
    if not path.is_file():
        raise ProtocolError(f"bound {name} is missing: {sample_id}: {path}")
    if file_sha256(path) != expected:
        raise ProtocolError(f"bound {name} hash mismatch: {sample_id}")
    assert_not_sealed(path, config.NO_SEALED_MEDIA_TOKENS)
    return {"path": str(path), "sha256": expected}


def _string_asset_binding(value: Any, name: str, expected: Any, sample_id: str) -> dict[str, str]:
    path = Path(str(value)).resolve()
    expected_hash = str(expected)
    if not path.is_file():
        raise ProtocolError(f"bound {name} is missing: {sample_id}: {path}")
    if file_sha256(path) != expected_hash:
        raise ProtocolError(f"bound {name} hash mismatch: {sample_id}")
    assert_not_sealed(path, config.NO_SEALED_MEDIA_TOKENS)
    return {"path": str(path), "sha256": expected_hash}


def _record_from_parent(protocol_row: Mapping[str, Any], replacement_row: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = str(protocol_row.get("sample_id", ""))
    if not sample_id or sample_id != str(replacement_row.get("sample_id", "")):
        raise ProtocolError(f"parent sample-ID join mismatch: {sample_id}")
    source_group = str(protocol_row.get("source_group", ""))
    if not source_group or source_group != str(replacement_row.get("source_group", "")):
        raise ProtocolError(f"parent source-group join mismatch: {sample_id}")
    natural = _asset_binding(protocol_row.get("natural_audio"), "natural audio", sample_id)
    candidate_value = protocol_row.get("mfa_linear_audio")
    candidate = _asset_binding(candidate_value, "MFA-linear audio", sample_id)
    replacement_candidate = _string_asset_binding(
        replacement_row.get("candidate_audio"),
        "replacement MFA-linear audio",
        replacement_row.get("candidate_audio_sha256"),
        sample_id,
    )
    if candidate != replacement_candidate:
        raise ProtocolError(f"MFA-linear candidate parent join mismatch: {sample_id}")
    face = _asset_binding(protocol_row.get("face_video"), "face video", sample_id)
    replacement_face = _string_asset_binding(
        replacement_row.get("face_video"),
        "replacement face video",
        replacement_row.get("face_video_sha256"),
        sample_id,
    )
    if face != replacement_face:
        raise ProtocolError(f"face-video parent join mismatch: {sample_id}")
    sample_count = int(protocol_row.get("natural_samples", -1))
    if sample_count < config.MIN_AUDIO_SAMPLES or sample_count != int(replacement_row.get("natural_samples", -1)):
        raise ProtocolError(f"natural sample-count binding mismatch: {sample_id}")
    return {
        "sample_id": sample_id,
        "source_group": source_group,
        "transcript": str(protocol_row.get("transcript", "")),
        "natural_audio": natural,
        "mfa_linear_audio": candidate,
        "face_video": face,
        "natural_sample_count": sample_count,
        "candidate_mapping": protocol_row.get("mfa_linear_mapping"),
        "parent_protocol_record_sha256": canonical_json_sha256(
            {key: value for key, value in protocol_row.items() if key != "historical_scores"}
        ),
        "parent_replacement_record_sha256": canonical_json_sha256(replacement_row),
    }


def load_frozen_cohort(
    parent_protocol: Path = config.PARENT_PROTOCOL,
    parent_replacement: Path = config.PARENT_REPLACEMENT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _parent_binding(parent_protocol, config.EXPECTED_PARENT_PROTOCOL_SHA256, "parent_protocol")
    _parent_binding(parent_replacement, config.EXPECTED_PARENT_REPLACEMENT_SHA256, "parent_replacement")
    protocol = read_json(parent_protocol)
    replacement = read_json(parent_replacement)
    if protocol.get("status") != "complete" or protocol.get("manifest_type") != "lrs3_mfa_dtw_comparison_protocol":
        raise ProtocolError("parent protocol is not complete")
    if replacement.get("status") != "complete" or replacement.get("manifest_type") != "lrs3_mfa3_exploratory_strict_replacement_protocol":
        raise ProtocolError("parent replacement manifest is not complete")
    protocol_cohort = protocol.get("cohort")
    replacement_cohort = replacement.get("cohort")
    if not isinstance(protocol_cohort, Mapping) or not isinstance(replacement_cohort, Mapping):
        raise ProtocolError("parent cohort is missing")
    protocol_rows = protocol_cohort.get("records")
    replacement_rows = replacement_cohort.get("records")
    if not isinstance(protocol_rows, list) or len(protocol_rows) != config.PARENT_RECORD_COUNT:
        raise ProtocolError("parent protocol record count changed")
    if not isinstance(replacement_rows, list) or len(replacement_rows) != config.PARENT_RECORD_COUNT:
        raise ProtocolError("parent replacement record count changed")
    protocol_ids = [str(row.get("sample_id", "")) for row in protocol_rows]
    replacement_ids = [str(row.get("sample_id", "")) for row in replacement_rows]
    if protocol_ids != replacement_ids:
        raise ProtocolError("parent cohort order differs")
    if int(protocol_cohort.get("source_group_count", -1)) != config.PARENT_SOURCE_GROUP_COUNT:
        raise ProtocolError("parent source-group count changed")
    if len(set(protocol_ids)) != config.PARENT_RECORD_COUNT:
        raise ProtocolError("parent sample IDs are not unique")
    replacement_by_id = {str(row.get("sample_id", "")): row for row in replacement_rows}
    records: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for protocol_row in protocol_rows:
        source_group = str(protocol_row.get("source_group", ""))
        if source_group in seen_groups:
            continue
        seen_groups.add(source_group)
        sample_id = str(protocol_row.get("sample_id", ""))
        replacement_row = replacement_by_id.get(sample_id)
        if replacement_row is None:
            raise ProtocolError(f"selected sample absent from replacement parent: {sample_id}")
        records.append(_record_from_parent(protocol_row, replacement_row))
    selected_ids = [row["sample_id"] for row in records]
    groups = [row["source_group"] for row in records]
    if len(records) != config.EXPECTED_RECORD_COUNT or len(set(groups)) != config.EXPECTED_SOURCE_GROUP_COUNT:
        raise ProtocolError("selected cohort count or group count changed")
    if sample_ids_sha256(selected_ids) != config.EXPECTED_SAMPLE_ID_SHA256:
        raise ProtocolError("selected ordered sample-ID hash changed")
    return records, {
        "record_count": len(records),
        "source_group_count": len(set(groups)),
        "sample_ids_sha256": sample_ids_sha256(selected_ids),
        "selection": "first ordered record per source group from parent protocol cohort",
    }


def protocol_payload(parent_bindings: Mapping[str, Any]) -> dict[str, Any]:
    config_payload = config.FrozenConfig().to_dict()
    return {
        "schema_version": 1,
        "stage_id": "00_protocol",
        "protocol_id": config.PROTOCOL_ID,
        "experiment": config.EXPERIMENT,
        "dataset": "lrs3",
        "config": config_payload,
        "parents": dict(parent_bindings),
        "selection_policy": {
            "parent_record_count": config.PARENT_RECORD_COUNT,
            "parent_source_group_count": config.PARENT_SOURCE_GROUP_COUNT,
            "selected_record_count": config.EXPECTED_RECORD_COUNT,
            "selected_source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
            "rule": "first ordered record per source group",
            "ordered_sample_id_sha256": config.EXPECTED_SAMPLE_ID_SHA256,
            "uses_scores": False,
        },
        "sealed_scope": {
            "validation_test_accessed": False,
            "training": False,
            "fine_tuning": False,
            "tts_generation": False,
            "mfa": False,
            "dtw": False,
            "score_based_selection": False,
            "score_based_retry": False,
            "parent_overwrite": False,
        },
    }


def run_stage00(output_stage: Path = config.STAGES["00_protocol"]) -> dict[str, Any]:
    assert_run_root_compatible(config.RUN_ROOT, config.PROTOCOL_ID)
    parent_bindings = {
        "parent_protocol": _parent_binding(
            config.PARENT_PROTOCOL, config.EXPECTED_PARENT_PROTOCOL_SHA256, "parent_protocol"
        ),
        "parent_replacement": _parent_binding(
            config.PARENT_REPLACEMENT, config.EXPECTED_PARENT_REPLACEMENT_SHA256, "parent_replacement"
        ),
    }
    try:
        records, selection = load_frozen_cohort()
        protocol = protocol_payload(parent_bindings)
        protocol_path = output_stage / "protocol.json"
        if protocol_path.is_file():
            existing = verify_self_hashed_json(protocol_path)
            existing_body = dict(existing)
            existing_body.pop("artifact_sha256", None)
            if existing_body != protocol:
                raise ProtocolError("existing Stage 00 protocol binding differs")
        else:
            write_self_hashed_json(protocol_path, protocol)
        cohort = {
            "schema_version": 1,
            "stage_id": "00_protocol",
            "protocol_id": config.PROTOCOL_ID,
            "status": "complete",
            "parent_protocol_sha256": config.EXPECTED_PARENT_PROTOCOL_SHA256,
            "parent_replacement_sha256": config.EXPECTED_PARENT_REPLACEMENT_SHA256,
            **selection,
            "records": records,
        }
        write_self_hashed_json(output_stage / "cohort.json", cohort)
        media_access = {
            "schema_version": 1,
            "stage_id": "00_protocol",
            "protocol_id": config.PROTOCOL_ID,
            "cohort_audio_decode_count": 0,
            "cohort_video_decode_count": 0,
            "cohort_audio_hash_read_count": config.EXPECTED_RECORD_COUNT * 2,
            "cohort_video_hash_read_count": config.EXPECTED_RECORD_COUNT,
            "score_read_count": 0,
            "sealed_media_accessed": False,
            "selection_used_scores": False,
        }
        write_self_hashed_json(output_stage / "media_access.json", media_access)
        write_self_hashed_json(
            output_stage / "failures.json",
            {
                "schema_version": 1,
                "stage_id": "00_protocol",
                "protocol_id": config.PROTOCOL_ID,
                "status": "complete",
                "failures": [],
            },
        )
        return {"status": "complete", **selection}
    except Exception as exc:
        output_stage.mkdir(parents=True, exist_ok=True)
        write_self_hashed_json(
            output_stage / "media_access.json",
            {
                "schema_version": 1,
                "stage_id": "00_protocol",
                "protocol_id": config.PROTOCOL_ID,
                "cohort_audio_decode_count": 0,
                "cohort_video_decode_count": 0,
                "score_read_count": 0,
                "sealed_media_accessed": False,
                "selection_used_scores": False,
            },
        )
        write_self_hashed_json(
            output_stage / "failures.json",
            {
                "schema_version": 1,
                "stage_id": "00_protocol",
                "protocol_id": config.PROTOCOL_ID,
                "status": "blocked",
                "failures": [{"error_type": type(exc).__name__, "error": str(exc)}],
            },
        )
        raise
