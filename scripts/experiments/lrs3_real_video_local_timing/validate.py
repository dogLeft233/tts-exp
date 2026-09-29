from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    DiagnosticError,
    assert_not_sealed,
    bytes_sha256,
    canonical_json_sha256,
    file_sha256,
    read_json,
    sample_ids_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .media import decode_pcm16, decode_video_frames, source_pcm16, verify_muxed_media
from .protocol import (
    load_history_cohort,
    load_protocol,
    pcm_wav_metadata,
    video_timeline,
)
from .scoring import local_evidence, parse_syncnet_log, reconstruct_global


def _artifact(path: Path) -> dict[str, Any]:
    return verify_self_hashed_json(path)


def _independent_asset(value: Any, name: str, sample_id: str) -> tuple[dict[str, Any], Path | None, list[str]]:
    details: dict[str, Any] = {"path": None, "expected_sha256": None, "actual_sha256": None, "hash_match": False}
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return details, None, [f"history asset is malformed: {sample_id}/{name}"]
    path = Path(str(value.get("path", ""))).resolve()
    expected = str(value.get("sha256", ""))
    details.update({"path": str(path), "expected_sha256": expected or None})
    try:
        assert_not_sealed(path)
    except DiagnosticError as exc:
        return details, path, [str(exc)]
    if not path.is_file():
        return details, path, [f"history asset is missing: {sample_id}/{name}"]
    actual = file_sha256(path)
    details["actual_sha256"] = actual
    details["hash_match"] = bool(expected and actual == expected)
    if not details["hash_match"]:
        errors.append(f"history asset hash changed: {sample_id}/{name}")
    return details, path, errors


def _independent_resolve_history_path(value: Any) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = config.REPO / path
    return path.resolve()


def _independent_pairing(
    source: Mapping[str, Any],
    video: Mapping[str, str],
    audio: Mapping[str, str],
    timeline: Mapping[str, Any],
    audio_meta: Mapping[str, Any],
) -> dict[str, Any]:
    audio_path = Path(str(audio["path"]))
    manifest_path = audio_path.parent.parent / "manifest.json"
    if not manifest_path.is_file():
        raise DiagnosticError(f"historical source manifest is missing: {manifest_path}")
    manifest_sha = file_sha256(manifest_path)
    if manifest_sha != config.HISTORY_SOURCE_MANIFEST_SHA256:
        raise DiagnosticError("historical source manifest hash changed")
    manifest = read_json(manifest_path)
    records = manifest.get("records")
    if not isinstance(records, list):
        raise DiagnosticError("historical source manifest records are missing")
    sample_id = str(source.get("sample_id", ""))
    matches = [(index, row) for index, row in enumerate(records) if isinstance(row, Mapping) and str(row.get("sample_id")) == sample_id]
    if len(matches) != 1:
        raise DiagnosticError(f"historical source manifest pairing is ambiguous: {sample_id}")
    record_index, manifest_record = matches[0]
    checks = {
        "sample_id": str(manifest_record.get("sample_id")) == sample_id,
        "source_group": str(manifest_record.get("source_group")) == str(source.get("source_group")),
        "face_video_path": _independent_resolve_history_path(manifest_record.get("video_local_path")) == Path(str(video["path"])).resolve(),
        "face_video_hash": str(manifest_record.get("video_sha256")) == str(video["sha256"]),
        "natural_audio_path": _independent_resolve_history_path(manifest_record.get("natural_audio_path")) == audio_path.resolve(),
        "natural_audio_hash": str(manifest_record.get("natural_audio_sha256")) == str(audio["sha256"]),
        "natural_audio_samples": int(manifest_record.get("natural_audio_samples", -1)) == int(audio_meta["sample_count"]),
        "natural_audio_rate": int(manifest_record.get("natural_audio_sample_rate_hz", -1)) == config.SAMPLE_RATE,
        "natural_audio_channels": int(manifest_record.get("natural_audio_channels", -1)) == config.PCM_CHANNELS,
    }
    if source.get("transcript") is not None:
        checks["transcript"] = str(manifest_record.get("transcript")) == str(source.get("transcript"))
    if not all(checks.values()):
        failed = ",".join(name for name, passed in checks.items() if not passed)
        raise DiagnosticError(f"historical source pairing differs: {sample_id} ({failed})")
    video_start = float(timeline["first_pts_time"])
    start_difference_ms = abs(video_start) * 1000.0
    return {
        "source_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha,
            "record_index": int(record_index),
            "record_sha256": canonical_json_sha256(manifest_record),
        },
        "checks": checks,
        "audio_start_s": 0.0,
        "video_start_pts_s": video_start,
        "start_difference_ms": float(start_difference_ms),
        "audio_start_evidence": "natural WAV sample 0 paired by the historical source manifest",
    }


def _independent_tail_contract(sample_count: int, frame_count: int) -> dict[str, Any]:
    delta = int(sample_count) - int(frame_count) * config.SAMPLES_PER_FRAME
    if delta < config.AUDIO_TAIL_MIN_SAMPLES:
        tail_class = "invalid_short_tail"
    elif delta > config.AUDIO_TAIL_MAX_SAMPLES:
        tail_class = "invalid_long_tail"
    elif delta > config.SAMPLES_PER_FRAME:
        tail_class = "extended_audio_tail"
    else:
        tail_class = "within_original_bound"
    return {
        "revision": config.PROTOCOL_REVISION,
        "delta_samples": delta,
        "delta_ms": float(delta * 1000 / config.SAMPLE_RATE),
        "accepted": bool(config.AUDIO_TAIL_MIN_SAMPLES <= delta <= config.AUDIO_TAIL_MAX_SAMPLES),
        "tail_class": tail_class,
        "reason": "unattributed" if tail_class == "extended_audio_tail" else None,
        "bounds_samples": [config.AUDIO_TAIL_MIN_SAMPLES, config.AUDIO_TAIL_MAX_SAMPLES],
    }


def _independent_audit_record(source: Mapping[str, Any], index: int) -> dict[str, Any]:
    sample_id = str(source.get("sample_id", "")) or None
    source_group = str(source.get("source_group", "")) or None
    label = sample_id or f"record_{index}"
    face_details, face_path, face_errors = _independent_asset(source.get("face_video"), "face_video", label)
    audio_details, audio_path, audio_errors = _independent_asset(source.get("natural_audio"), "natural_audio", label)
    errors = [*face_errors, *audio_errors]
    checks: dict[str, str] = {
        "face_video_hash": "PASS" if not face_errors else "FAIL",
        "natural_audio_hash": "PASS" if not audio_errors else "FAIL",
        "video_timeline": "NOT_RUN",
        "video_pts": "NOT_RUN",
        "natural_audio_format": "NOT_RUN",
        "tail_contract": "NOT_RUN",
        "source_pairing": "NOT_RUN",
        "audio_video_start": "NOT_RUN",
        "common_support": "NOT_RUN",
    }
    timeline: dict[str, Any] | None = None
    if face_path is not None and face_details["hash_match"]:
        try:
            timeline = video_timeline(face_path)
            checks["video_timeline"] = "PASS"
            checks["video_pts"] = "PASS"
        except (DiagnosticError, OSError, ValueError) as exc:
            checks["video_timeline"] = "FAIL"
            checks["video_pts"] = "FAIL"
            errors.append(str(exc))
    audio_meta: dict[str, Any] | None = None
    if audio_path is not None and audio_details["hash_match"]:
        try:
            audio_meta = pcm_wav_metadata(audio_path)
            checks["natural_audio_format"] = "PASS"
        except (DiagnosticError, OSError, ValueError) as exc:
            checks["natural_audio_format"] = "FAIL"
            errors.append(str(exc))
    duration: dict[str, Any] | None = None
    common_support: list[int] | None = None
    pairing: dict[str, Any] | None = None
    if timeline is not None and audio_meta is not None:
        duration = _independent_tail_contract(int(audio_meta["sample_count"]), int(timeline["frame_count"]))
        checks["tail_contract"] = "PASS" if duration["accepted"] else "FAIL"
        if not duration["accepted"]:
            errors.append(f"audio/video duration is outside {config.PROTOCOL_REVISION}: {label} d={duration['delta_samples']}")
        common_support = [0, min(int(audio_meta["sample_count"]), int(timeline["frame_count"]) * config.SAMPLES_PER_FRAME)]
        checks["common_support"] = "PASS" if common_support[1] > 0 else "FAIL"
        if common_support[1] <= 0:
            errors.append(f"common audio/video support is empty: {label}")
        try:
            pairing = _independent_pairing(
                source,
                {"path": str(face_path), "sha256": str(source.get("face_video", {}).get("sha256", ""))},
                {"path": str(audio_path), "sha256": str(source.get("natural_audio", {}).get("sha256", ""))},
                timeline,
                audio_meta,
            )
            checks["source_pairing"] = "PASS"
            checks["audio_video_start"] = "PASS" if pairing["start_difference_ms"] <= 1.0 else "FAIL"
            if checks["audio_video_start"] == "FAIL":
                errors.append(f"audio/video start differs by more than 1 ms: {label}")
        except (DiagnosticError, OSError, ValueError, TypeError, AttributeError) as exc:
            errors.append(str(exc))
            checks["source_pairing"] = "FAIL"
            checks["audio_video_start"] = "FAIL"
    audit_row: dict[str, Any] = {
        "index": index,
        "sample_id": sample_id,
        "source_group": source_group,
        "face_video": face_details,
        "natural_audio": audio_details,
        "frame_count": int(timeline["frame_count"]) if timeline is not None else None,
        "sample_count": int(audio_meta["sample_count"]) if audio_meta is not None else None,
        "duration_delta_samples": duration["delta_samples"] if duration is not None else None,
        "tail_difference_ms": duration["delta_ms"] if duration is not None else None,
        "tail_class": duration["tail_class"] if duration is not None else None,
        "source_video_timeline": timeline,
        "natural_audio_format": audio_meta,
        "source_pairing": pairing,
        "common_support_samples": common_support,
        "checks": checks,
        "passed": not errors and all(value == "PASS" for value in checks.values()),
        "status": "PASS" if not errors and all(value == "PASS" for value in checks.values()) else "BLOCKED",
        "errors": errors,
    }
    return audit_row


def validate_input_audit(paths, cohort: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit = _artifact(paths.input_audit)
    if audit.get("schema_version") != 2 or audit.get("stage_id") != "input_audit" or audit.get("protocol_id") != config.PROTOCOL_ID or audit.get("protocol_revision") != config.PROTOCOL_REVISION:
        raise DiagnosticError("input audit identity or revision is invalid")
    if audit.get("cohort", {}).get("sha256") != config.HISTORY_COHORT_SHA256 or audit.get("record_count") != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("input audit cohort binding is invalid")
    records = cohort.get("records")
    audit_records = audit.get("records")
    if not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT or not isinstance(audit_records, list) or len(audit_records) != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("input audit must contain the complete cohort")
    expected = [_independent_audit_record(source if isinstance(source, Mapping) else {}, index) for index, source in enumerate(records)]
    if audit_records != expected:
        raise DiagnosticError("input audit differs from the independently recomputed source evidence")
    passed_count = sum(bool(row["passed"]) for row in expected)
    if audit.get("passed_count") != passed_count or audit.get("blocked_count") != len(expected) - passed_count or audit.get("status") != ("complete" if passed_count == len(expected) else "blocked"):
        raise DiagnosticError("input audit summary does not match its records")
    if audit.get("source_manifest") != {"path": str(config.HISTORY_SOURCE_MANIFEST.resolve()), "sha256": config.HISTORY_SOURCE_MANIFEST_SHA256}:
        raise DiagnosticError("input audit source manifest binding is invalid")
    return audit, expected


def validate_protocol(paths) -> tuple[dict[str, Any], dict[str, Any]]:
    cohort = load_history_cohort()
    audit, audit_records = validate_input_audit(paths, cohort)
    protocol = load_protocol(paths)
    records = protocol.get("records")
    if not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("protocol record count is invalid")
    if (
        protocol.get("schema_version") != 2
        or protocol.get("protocol_revision") != config.PROTOCOL_REVISION
        or protocol.get("cohort", {}).get("sha256") != config.HISTORY_COHORT_SHA256
        or protocol.get("config") != config.FrozenConfig().to_dict()
    ):
        raise DiagnosticError("protocol cohort binding is invalid")
    audit_binding = protocol.get("input_audit")
    if (
        not isinstance(audit_binding, Mapping)
        or audit_binding.get("path") != str(paths.input_audit)
        or audit_binding.get("sha256") != file_sha256(paths.input_audit)
        or audit_binding.get("artifact_sha256") != audit.get("artifact_sha256")
    ):
        raise DiagnosticError("protocol input audit binding is invalid")
    for name, expected in (
        ("parent_spec", config.PARENT_SPEC_SHA256),
        ("amendment_spec", config.AMENDMENT_SPEC_SHA256),
    ):
        binding = protocol.get("spec_bindings", {}).get(name)
        if not isinstance(binding, Mapping) or binding.get("sha256") != expected or not Path(str(binding.get("path", ""))).is_file() or file_sha256(Path(str(binding["path"]))) != expected:
            raise DiagnosticError(f"protocol spec binding is invalid: {name}")
    parent = protocol.get("parent_blocked_run")
    if (
        not isinstance(parent, Mapping)
        or parent.get("sha256") != config.PARENT_BLOCKED_FINAL_SHA256
        or parent.get("result_sha256") != config.PARENT_BLOCKED_RESULT_SHA256
        or not Path(str(parent.get("path", ""))).is_file()
        or file_sha256(Path(str(parent["path"]))) != config.PARENT_BLOCKED_FINAL_SHA256
    ):
        raise DiagnosticError("protocol parent blocked-run binding is invalid")
    expected_ids = [str(row.get("sample_id")) for row in audit_records]
    actual_ids = [str(row.get("sample_id", "")) for row in records]
    if actual_ids != expected_ids or sample_ids_sha256(actual_ids) != config.EXPECTED_SAMPLE_ID_SHA256:
        raise DiagnosticError("protocol record order is invalid")
    from .media import build_mapping

    for source, audit_row, actual in zip(cohort["records"], audit_records, records, strict=True):
        sample_id = str(audit_row["sample_id"])
        expected_source = {
            "sample_id": sample_id,
            "source_group": str(audit_row["source_group"]),
            "paired_key": source.get("paired_key"),
            "speaker_id": source.get("speaker_id"),
            "split": source.get("split"),
            "transcript": source.get("transcript"),
            "face_video": {"path": audit_row["face_video"]["path"], "sha256": str(source["face_video"]["sha256"])},
            "natural_audio": {"path": audit_row["natural_audio"]["path"], "sha256": str(source["natural_audio"]["sha256"])},
            "source_video_timeline": audit_row["source_video_timeline"],
            "natural_audio_format": audit_row["natural_audio_format"],
            "source_pairing": audit_row["source_pairing"],
            "duration_contract": {
                "revision": config.PROTOCOL_REVISION,
                "delta_samples": audit_row["duration_delta_samples"],
                "delta_ms": audit_row["tail_difference_ms"],
                "accepted": True,
                "tail_class": audit_row["tail_class"],
                "reason": "unattributed" if audit_row["tail_class"] == "extended_audio_tail" else None,
                "bounds_samples": [config.AUDIO_TAIL_MIN_SAMPLES, config.AUDIO_TAIL_MAX_SAMPLES],
            },
            "common_support_samples": audit_row["common_support_samples"],
        }
        for field, expected in expected_source.items():
            if actual.get(field) != expected:
                raise DiagnosticError(f"protocol source binding differs: {sample_id}/{field}")
        crop = actual.get("crop")
        if not isinstance(crop, Mapping):
            raise DiagnosticError(f"protocol crop binding is missing: {sample_id}")
        frame_count = int(audit_row["source_video_timeline"]["frame_count"])
        if int(crop.get("frame_count", -1)) != frame_count:
            raise DiagnosticError(f"protocol crop frame count is invalid: {sample_id}")
        mappings = crop.get("mappings")
        if not isinstance(mappings, Mapping):
            raise DiagnosticError(f"protocol mapping is missing: {sample_id}")
        expected_mapping = build_mapping(frame_count)
        for arm in config.ARMS:
            if [int(value) for value in mappings.get(arm, [])] != expected_mapping[arm]:
                raise DiagnosticError(f"protocol mapping changed: {sample_id}/{arm}")
        processed = crop.get("processed_track")
        if not isinstance(processed, list) or len(processed) != frame_count:
            raise DiagnosticError(f"protocol crop track is incomplete: {sample_id}")
    return protocol, cohort


def validate_media(paths, protocol: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _artifact(paths.media / "manifest.json")
    if (
        manifest.get("status") != "complete"
        or manifest.get("protocol_id") != config.PROTOCOL_ID
        or manifest.get("protocol_revision") != config.PROTOCOL_REVISION
        or manifest.get("protocol_sha256") != file_sha256(paths.protocol)
        or manifest.get("record_count") != config.EXPECTED_RECORD_COUNT
        or manifest.get("cell_count") != config.expected_cell_count()
        or manifest.get("arms") != list(config.ARMS)
    ):
        raise DiagnosticError("media manifest is incomplete")
    manifest_rows = {str(row.get("sample_id")): row for row in manifest.get("rows", [])}
    if len(manifest_rows) != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("media record identity is incomplete")
    for record in protocol["records"]:
        sample_id = str(record["sample_id"])
        row = manifest_rows.get(sample_id)
        if row is None:
            raise DiagnosticError(f"media row is missing: {sample_id}")
        crop_base = Path(str(row.get("crop_base")))
        if not crop_base.is_file() or file_sha256(crop_base) != row.get("crop_base_sha256"):
            raise DiagnosticError(f"crop base hash changed: {sample_id}")
        crop_sidecar = _artifact(crop_base.with_suffix(".json"))
        crop_frames = decode_video_frames(crop_base)
        expected_count = int(record["crop"]["frame_count"])
        if crop_sidecar.get("protocol_sha256") != file_sha256(paths.protocol) or crop_sidecar.get("sample_id") != sample_id or len(crop_frames) != expected_count:
            raise DiagnosticError(f"crop base identity is invalid: {sample_id}")
        mapping_path = Path(str(row.get("mapping")))
        mapping_payload = _artifact(mapping_path)
        if mapping_payload.get("protocol_sha256") != file_sha256(paths.protocol) or mapping_payload.get("sample_id") != sample_id:
            raise DiagnosticError(f"mapping identity is invalid: {sample_id}")
        audio = Path(str(record["natural_audio"]["path"]))
        expected_pcm = source_pcm16(audio)
        arms = row.get("arms")
        if not isinstance(arms, Mapping) or set(arms) != set(config.ARMS):
            raise DiagnosticError(f"media arm set is incomplete: {sample_id}")
        for arm in config.ARMS:
            item = arms[arm]
            output = Path(str(item["output"]))
            sidecar = _artifact(Path(str(item["sidecar"])))
            mapping = [int(value) for value in item.get("mapping", [])]
            expected_mapping = [int(value) for value in record["crop"]["mappings"][arm]]
            if mapping != expected_mapping or sidecar.get("mapping") != expected_mapping:
                raise DiagnosticError(f"media mapping changed: {sample_id}/{arm}")
            if not output.is_file() or file_sha256(output) != item.get("output_sha256") or file_sha256(output) != sidecar.get("output_sha256"):
                raise DiagnosticError(f"media hash changed: {sample_id}/{arm}")
            if sidecar.get("protocol_sha256") != file_sha256(paths.protocol) or sidecar.get("sample_id") != sample_id or sidecar.get("arm") != arm or sidecar.get("crop_base_sha256") != file_sha256(crop_base) or sidecar.get("audio_sha256") != file_sha256(audio) or sidecar.get("audio_pcm_sha256") != bytes_sha256(expected_pcm):
                raise DiagnosticError(f"media sidecar identity is invalid: {sample_id}/{arm}")
            expected = verify_muxed_media(output, crop_frames, expected_pcm, expected_mapping)
            if expected["sha256"] != file_sha256(output):
                raise DiagnosticError(f"media verification hash changed: {sample_id}/{arm}")
            expected_frame_hashes = [hashlib.sha256(crop_frames[source].tobytes()).hexdigest() for source in expected_mapping]
            if sidecar.get("source_frame_sha256") != expected_frame_hashes:
                raise DiagnosticError(f"media source-frame evidence changed: {sample_id}/{arm}")
    return {"stage": "media", "status": "valid", "record_count": config.EXPECTED_RECORD_COUNT, "cell_count": config.expected_cell_count()}


def validate_scores(paths, protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[tuple[str, str], Mapping[str, Any]]]:
    manifest = _artifact(paths.scores / "manifest.json")
    if (
        manifest.get("status") != "complete"
        or manifest.get("protocol_id") != config.PROTOCOL_ID
        or manifest.get("protocol_revision") != config.PROTOCOL_REVISION
        or manifest.get("protocol_sha256") != file_sha256(paths.protocol)
        or manifest.get("media_manifest_sha256") != file_sha256(paths.media / "manifest.json")
        or manifest.get("cell_count") != config.expected_cell_count()
        or len(manifest.get("scores", [])) != config.expected_cell_count()
    ):
        raise DiagnosticError("score manifest is incomplete")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in manifest["scores"]:
        key = (str(row.get("sample_id")), str(row.get("arm")))
        if key in result or key[1] not in config.ARMS:
            raise DiagnosticError(f"duplicate or unknown score cell: {key}")
        # The manifest stores the complete sidecar row; derive the canonical cell path from the matrix.
        matrix = Path(str(row.get("matrix", "")))
        if not matrix.is_file() or file_sha256(matrix) != row.get("matrix_sha256"):
            raise DiagnosticError(f"score matrix hash changed: {key}")
        values = np.asarray(np.load(matrix, allow_pickle=False), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 31 or not np.isfinite(values).all() or row.get("matrix_shape") != [int(value) for value in values.shape]:
            raise DiagnosticError(f"score matrix shape or values are invalid: {key}")
        log = Path(str(row.get("score_log")))
        parsed = parse_syncnet_log(log)
        reconstructed = reconstruct_global(values)
        if abs(float(parsed["sync_c"]) - reconstructed["sync_c"]) > 0.001 or abs(float(parsed["sync_d"]) - reconstructed["sync_d"]) > 0.001 or int(parsed["av_offset"]) != int(reconstructed["offset"]):
            raise DiagnosticError(f"score log parity failed: {key}")
        record = next((item for item in protocol["records"] if str(item["sample_id"]) == key[0]), None)
        if record is None:
            raise DiagnosticError(f"score record is unknown: {key}")
        mask_mapping = record["crop"]["mappings"][config.WARP_ARM]
        local = local_evidence(
            values,
            mask_mapping,
            int(record["crop"]["frame_count"]),
            int(record["natural_audio_format"]["sample_count"]),
            record.get("common_support_samples"),
        )
        if row.get("window_frame_starts") != list(range(values.shape[0])) or row.get("column_offsets") != [config.VSHIFT - index for index in range(31)] or row.get("local_mask_mapping_sha256") != canonical_json_sha256(mask_mapping) or row.get("reconstructed") != reconstructed or row.get("local") != local:
            raise DiagnosticError(f"score evidence changed: {key}")
        media = Path(str(row.get("media")))
        if not media.is_file() or file_sha256(media) != row.get("media_sha256"):
            raise DiagnosticError(f"score media hash changed: {key}")
        audio_path = Path(str(record["natural_audio"]["path"]))
        if row.get("natural_audio_sha256") != file_sha256(audio_path) or row.get("natural_audio_pcm_sha256") != bytes_sha256(source_pcm16(audio_path)) or decode_pcm16(media) != source_pcm16(audio_path):
            raise DiagnosticError(f"score media audio binding changed: {key}")
        result[key] = row
    return manifest, result


def _independent_decision(protocol: Mapping[str, Any], scores: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    repeatability = 0
    baseline = 0
    recovery = 0
    per_record: list[dict[str, Any]] = []
    c_values: list[float] = []
    d_values: list[float] = []
    groups: list[str] = []
    for record in protocol["records"]:
        sample_id = str(record["sample_id"])
        real = scores[(sample_id, config.REAL_ARM)]
        repeat = scores[(sample_id, config.REPEAT_ARM)]
        warp = scores[(sample_id, config.WARP_ARM)]
        real_matrix = np.asarray(np.load(Path(str(real["matrix"])), allow_pickle=False), dtype=np.float64)
        repeat_matrix = np.asarray(np.load(Path(str(repeat["matrix"])), allow_pickle=False), dtype=np.float64)
        real_plus = real["local"]["PLUS"]
        real_minus = real["local"]["MINUS"]
        repeat_plus = repeat["local"]["PLUS"]
        repeat_minus = repeat["local"]["MINUS"]
        warp_plus = warp["local"]["PLUS"]
        warp_minus = warp["local"]["MINUS"]
        repeat_pass = bool(
            real_matrix.shape == repeat_matrix.shape
            and float(np.max(np.abs(real_matrix - repeat_matrix))) <= 0.001
            and int(real["reconstructed"]["offset"]) == int(repeat["reconstructed"]["offset"])
            and int(real_plus["offset"]) == int(repeat_plus["offset"])
            and int(real_minus["offset"]) == int(repeat_minus["offset"])
        )
        repeatability += int(repeat_pass)
        baseline_pass = bool(real_plus["clear"] and real_minus["clear"] and abs(int(real_plus["offset"]) - int(real_minus["offset"])) <= config.OFFSET_TOLERANCE_FRAMES)
        baseline += int(baseline_pass)
        delta_plus = int(warp_plus["offset"]) - int(real_plus["offset"])
        delta_minus = int(warp_minus["offset"]) - int(real_minus["offset"])
        recovery_pass = bool(baseline_pass and warp_plus["clear"] and warp_minus["clear"] and abs(delta_plus + 3) <= config.OFFSET_TOLERANCE_FRAMES and abs(delta_minus - 3) <= config.OFFSET_TOLERANCE_FRAMES)
        recovery += int(recovery_pass)
        c_value = float(real["reconstructed"]["sync_c"]) - float(warp["reconstructed"]["sync_c"])
        d_value = float(warp["reconstructed"]["sync_d"]) - float(real["reconstructed"]["sync_d"])
        c_values.append(c_value)
        d_values.append(d_value)
        groups.append(str(record["source_group"]))
        per_record.append({"sample_id": sample_id, "repeatability": repeat_pass, "baseline": baseline_pass, "delta_plus": delta_plus, "delta_minus": delta_minus, "recovery": recovery_pass})
    if repeatability != config.EXPECTED_RECORD_COUNT:
        decision = "REPEATABILITY_FAILED"
    elif baseline < config.MIN_BASELINE_RECORDS:
        decision = "BASELINE_INCONCLUSIVE"
    elif recovery >= config.MIN_SUCCESS_RECORDS:
        decision = "LOCAL_TIMING_DETECTED"
    else:
        decision = "LOCAL_TIMING_NOT_ESTABLISHED"
    return {"decision": decision, "repeatability": repeatability, "baseline": baseline, "recovery": recovery, "per_record": per_record, "c_values": c_values, "d_values": d_values, "groups": groups}


def _validate_new_revision_binding(final: Mapping[str, Any], paths) -> dict[str, Any]:
    if final.get("schema_version") != 2 or final.get("protocol_revision") != config.PROTOCOL_REVISION:
        raise DiagnosticError("final artifact is not the bounded-tail v2 schema")
    for name, expected in (
        ("parent_spec", config.PARENT_SPEC_SHA256),
        ("amendment_spec", config.AMENDMENT_SPEC_SHA256),
    ):
        binding = final.get("spec_bindings", {}).get(name)
        if not isinstance(binding, Mapping) or binding.get("path") != str(getattr(config, "PARENT_SPEC" if name == "parent_spec" else "AMENDMENT_SPEC").resolve()) or binding.get("sha256") != expected or not Path(str(binding.get("path", ""))).is_file() or file_sha256(Path(str(binding["path"]))) != expected:
            raise DiagnosticError(f"final spec binding is invalid: {name}")
    parent = final.get("parent_blocked_run")
    if (
        not isinstance(parent, Mapping)
        or parent.get("run_id") != "lrs3_real_video_local_timing_20260905_v2"
        or parent.get("path") != str(config.PARENT_BLOCKED_FINAL.resolve())
        or parent.get("sha256") != config.PARENT_BLOCKED_FINAL_SHA256
        or parent.get("result_sha256") != config.PARENT_BLOCKED_RESULT_SHA256
        or not config.PARENT_BLOCKED_FINAL.is_file()
        or file_sha256(config.PARENT_BLOCKED_FINAL) != config.PARENT_BLOCKED_FINAL_SHA256
        or not config.PARENT_BLOCKED_FINAL.with_name("result.md").is_file()
        or file_sha256(config.PARENT_BLOCKED_FINAL.with_name("result.md")) != config.PARENT_BLOCKED_RESULT_SHA256
    ):
        raise DiagnosticError("final parent blocked-run binding is invalid")
    audit = final.get("input_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("path") != str(paths.input_audit.resolve())
        or not paths.input_audit.is_file()
        or audit.get("sha256") != file_sha256(paths.input_audit)
        or final.get("input_audit_sha256") != file_sha256(paths.input_audit)
    ):
        raise DiagnosticError("final input audit binding is invalid")
    return {"stage": "revision", "status": "valid", "protocol_revision": config.PROTOCOL_REVISION}


def validate_final(paths, protocol: Mapping[str, Any], score_index: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    final = _artifact(paths.final)
    if final.get("status") != "complete" or final.get("engineering_decision") != "GO" or final.get("record_count") != config.EXPECTED_RECORD_COUNT or final.get("cell_count") != config.expected_cell_count():
        raise DiagnosticError("final artifact is incomplete")
    _validate_new_revision_binding(final, paths)
    analysis = _artifact(paths.root / "analysis.json")
    recomputed = _independent_decision(protocol, score_index)
    if final.get("scientific_decision") != recomputed["decision"] or analysis.get("scientific_decision") != recomputed["decision"] or analysis.get("repeatability", {}).get("count") != recomputed["repeatability"] or analysis.get("baseline", {}).get("clear_and_aligned_count") != recomputed["baseline"] or analysis.get("recovery", {}).get("success_count") != recomputed["recovery"]:
        raise DiagnosticError("final decision does not match the independently recomputed gates")
    for field, path in (("protocol_sha256", paths.protocol), ("media_manifest_sha256", paths.media / "manifest.json"), ("score_manifest_sha256", paths.scores / "manifest.json"), ("analysis_sha256", paths.root / "analysis.json"), ("result_sha256", paths.result)):
        if final.get(field) != file_sha256(path):
            raise DiagnosticError(f"final binding changed: {field}")
    if analysis.get("schema_version") != 2 or analysis.get("protocol_revision") != config.PROTOCOL_REVISION or analysis.get("input_audit_sha256") != file_sha256(paths.input_audit):
        raise DiagnosticError("analysis input-audit binding is invalid")
    review = _artifact(paths.review / "manifest.json")
    if final.get("review_status") != "PENDING" or final.get("review_manifest_sha256") != file_sha256(paths.review / "manifest.json") or review.get("status") != "PENDING":
        raise DiagnosticError("review status or binding is invalid")
    return {"stage": "final", "status": "valid", "scientific_decision": recomputed["decision"], "recovery_count": recomputed["recovery"]}


def validate_blocked(paths) -> dict[str, Any]:
    final = _artifact(paths.final)
    if final.get("status") != "blocked" or final.get("engineering_decision") != "BLOCKED" or final.get("scientific_decision") is not None or final.get("eligibility") is not False or not paths.result.is_file() or final.get("result_sha256") != file_sha256(paths.result):
        raise DiagnosticError("blocked terminal is invalid")
    if final.get("protocol_revision") != config.PROTOCOL_REVISION:
        return {"stage": "final", "status": "valid", "schema": "legacy", "new_contract": "not_checked", "scientific_decision": None}
    revision = _validate_new_revision_binding(final, paths)
    cohort = load_history_cohort()
    audit, rows = validate_input_audit(paths, cohort)
    if audit.get("status") not in {"complete", "blocked"} or len(rows) != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("blocked input audit is incomplete")
    return {"stage": "final", "status": "valid", "schema": "bounded_tail_v2", "new_contract": audit.get("status"), "scientific_decision": None, "audit_record_count": len(rows), "protocol_revision": revision["protocol_revision"]}


def validate_run(root: Path) -> dict[str, Any]:
    paths = config.RunPaths(root.resolve())
    final = _artifact(paths.final)
    if final.get("protocol_revision") != config.PROTOCOL_REVISION:
        if final.get("status") == "blocked":
            return {"status": "valid", "stages": [validate_blocked(paths)]}
        raise DiagnosticError("legacy completed run requires its legacy validator; new contract was not asserted")
    if final.get("status") == "blocked":
        return {"status": "valid", "stages": [validate_blocked(paths)]}
    protocol, _ = validate_protocol(paths)
    media_result = validate_media(paths, protocol)
    score_manifest, score_index = validate_scores(paths, protocol)
    final_result = validate_final(paths, protocol, score_index)
    return {"status": "valid", "stages": [{"stage": "input_audit", "status": "valid", "record_count": config.EXPECTED_RECORD_COUNT}, {"stage": "protocol", "status": "valid"}, media_result, {"stage": "score", "status": "valid", "cell_count": len(score_manifest["scores"])}, final_result]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently validate an LRS3 real-video local timing diagnostic")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate_run(args.run_root)
    payload = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, **result}
    write_self_hashed_json(args.run_root.resolve() / "validation.json", payload)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
