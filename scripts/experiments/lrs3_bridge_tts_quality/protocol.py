from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    ProtocolError,
    asset_binding,
    canonical_json_sha256,
    file_sha256,
    read_json,
    runtime_binding,
    runtime_versions,
    sample_ids_sha256,
    verify_asset,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def _parent_binding(path: Path, name: str) -> dict[str, str]:
    expected = config.PARENT_HASHES[name]
    return asset_binding(path, expected_sha256=expected, name=f"parent {name}")


def _validate_input_bindings() -> dict[str, str]:
    binding = asset_binding(config.INPUT_BINDINGS, expected_sha256=config.INPUT_BINDINGS_SHA256, name="input-bindings")
    payload = read_json(config.INPUT_BINDINGS)
    if payload.get("change_id") != "compare-lrs3-bridge-tts-quality" or payload.get("binding_hash_kind") != "sha256_of_file_bytes":
        raise ProtocolError("input-bindings identity changed")
    parents = payload.get("parents")
    cohort = payload.get("cohort")
    if not isinstance(parents, Mapping) or not isinstance(cohort, Mapping):
        raise ProtocolError("input-bindings is missing parent/cohort declarations")
    expected_parent_names = {
        "confirmation_cohort": "confirmation_cohort",
        "cloud_tts_metadata": "cloud_tts_metadata",
        "historical_cloud_target_manifest": "historical_cloud_target",
        "parent_protocol": "parent_protocol",
    }
    for declared_name, config_name in expected_parent_names.items():
        row = parents.get(declared_name)
        if not isinstance(row, Mapping) or row.get("file_sha256") != config.PARENT_HASHES[config_name]:
            raise ProtocolError(f"input-bindings parent declaration changed: {declared_name}")
    if cohort.get("record_count") != config.EXPECTED_RECORD_COUNT or cohort.get("source_group_count") != config.EXPECTED_SOURCE_GROUP_COUNT or cohort.get("ordered_sample_ids_sha256_from_parent") != config.EXPECTED_SAMPLE_ID_SHA256 or cohort.get("sample_ids") != list(config.EXPECTED_SAMPLE_IDS):
        raise ProtocolError("input-bindings cohort declaration changed")
    return binding


def _bound_asset(value: Any, name: str, *, allow_sealed: bool = False) -> dict[str, str]:
    return verify_asset(value, name=name, allow_sealed=allow_sealed)


def _read_pcm_header(path: Path) -> dict[str, int]:
    import wave

    try:
        with wave.open(str(path), "rb") as handle:
            return {
                "sample_rate": int(handle.getframerate()),
                "channels": int(handle.getnchannels()),
                "sample_width": int(handle.getsampwidth()),
                "samples": int(handle.getnframes()),
            }
    except (OSError, wave.Error) as exc:
        raise ProtocolError(f"cannot read WAV header: {path}") from exc


def _asset_from_path(path: str | Path, expected: str, name: str) -> dict[str, str]:
    return asset_binding(path, expected_sha256=expected, name=name)


def _validate_parent_cohort() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = config.CONFIRMATION_COHORT
    _parent_binding(path, "confirmation_cohort")
    cohort = verify_self_hashed_json(path)
    if (
        cohort.get("status") != "complete"
        or cohort.get("record_count") != config.EXPECTED_RECORD_COUNT
        or cohort.get("source_group_count") != config.EXPECTED_SOURCE_GROUP_COUNT
        or cohort.get("sample_ids_sha256") != config.EXPECTED_SAMPLE_ID_SHA256
    ):
        raise ProtocolError("confirmation cohort is not the registered complete 22-record cohort")
    records = cohort.get("records")
    if not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("confirmation cohort records are incomplete")
    ids = [str(row.get("sample_id", "")) for row in records]
    groups = [str(row.get("source_group", "")) for row in records]
    if ids != list(config.EXPECTED_SAMPLE_IDS) or sample_ids_sha256(ids) != config.EXPECTED_SAMPLE_ID_SHA256:
        raise ProtocolError("confirmation cohort order or ordered-ID hash changed")
    if len(set(ids)) != len(ids) or len(set(groups)) != len(groups):
        raise ProtocolError("confirmation cohort IDs or source groups are not unique")
    return [dict(row) for row in records], cohort


def _validate_parent_protocol() -> dict[str, str]:
    path = config.PARENT_PROTOCOL
    binding = _parent_binding(path, "parent_protocol")
    payload = read_json(path)
    if payload.get("status") != "complete" or payload.get("engineering_decision") != "GO":
        raise ProtocolError("parent protocol is not a complete engineering GO artifact")
    if payload.get("protocol_id") != "lrs3_mfa_dtw_short_phone_support_20260904":
        raise ProtocolError("parent protocol identity changed")
    return binding


def _cloud_rows() -> dict[str, Mapping[str, Any]]:
    path = config.CLOUD_TTS_METADATA
    _parent_binding(path, "cloud_tts_metadata")
    payload = read_json(path)
    if (
        payload.get("dataset") != "lrs3"
        or payload.get("language") != config.LANGUAGE
        or payload.get("provider") != config.CLOUD_PROVIDER
        or payload.get("model") != config.CLOUD_MODEL
    ):
        raise ProtocolError("cloud TTS metadata top-level identity is not the registered provider")
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ProtocolError("cloud TTS metadata has no result map")
    return {str(key): value for key, value in results.items() if isinstance(value, Mapping)}


def _historical_rows() -> dict[str, Mapping[str, Any]]:
    path = config.HISTORICAL_CLOUD_TARGET
    _parent_binding(path, "historical_cloud_target")
    payload = read_json(path)
    if (
        payload.get("status") != "complete"
        or payload.get("engineering_decision") != "GO"
        or payload.get("protocol_id") != "lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
    ):
        raise ProtocolError("historical target manifest is not the registered complete artifact")
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise ProtocolError("historical target manifest has no result list")
    return {str(row.get("sample_id")): row for row in rows if isinstance(row, Mapping)}


def _record(
    parent_row: Mapping[str, Any],
    cloud: Mapping[str, Any],
    historical: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id = str(parent_row.get("sample_id", ""))
    group = str(parent_row.get("source_group", ""))
    if sample_id != str(cloud.get("sample_id", "")) or sample_id != str(historical.get("sample_id", "")):
        raise ProtocolError(f"sample-ID provenance join mismatch: {sample_id}")
    if group != str(cloud.get("source_group", "")) or group != str(historical.get("source_group", "")):
        raise ProtocolError(f"source-group provenance join mismatch: {sample_id}")
    transcript = str(parent_row.get("transcript", ""))
    if transcript != str(cloud.get("transcript", "")) or transcript != str(cloud.get("tts_transcript", "")):
        raise ProtocolError(f"cloud TTS transcript mismatch: {sample_id}")
    natural = _bound_asset(parent_row.get("natural_audio"), f"natural audio {sample_id}")
    face = _bound_asset(parent_row.get("face_video"), f"face video {sample_id}")
    if str(cloud.get("reference_audio_sha256")) != natural["sha256"]:
        raise ProtocolError(f"cloud reference audio is not the paired natural PCM: {sample_id}")
    if str(cloud.get("video_sha256")) != face["sha256"]:
        raise ProtocolError(f"cloud video identity differs from the paired face video: {sample_id}")
    if cloud.get("reference_role") != "paired_natural_audio":
        raise ProtocolError(f"cloud reference role is not paired natural audio: {sample_id}")
    if cloud.get("provider") != config.CLOUD_PROVIDER or cloud.get("model") != config.CLOUD_MODEL:
        raise ProtocolError(f"cloud provider/model mismatch: {sample_id}")
    cloud_audio = _asset_from_path(
        str(cloud.get("canonical_16k_audio", "")),
        str(cloud.get("canonical_audio_sha256", "")),
        f"cloud canonical audio {sample_id}",
    )
    cloud_header = _read_pcm_header(Path(cloud_audio["path"]))
    if cloud_header != {
        "sample_rate": config.SAMPLE_RATE,
        "channels": config.PCM_CHANNELS,
        "sample_width": config.PCM_SAMPLE_WIDTH,
        "samples": int(cloud.get("canonical_samples", -1)),
    }:
        raise ProtocolError(f"cloud canonical audio format/count mismatch: {sample_id}")
    provider_audio = _asset_from_path(
        str(cloud.get("provider_audio", "")),
        str(cloud.get("provider_audio_sha256", "")),
        f"cloud provider raw audio {sample_id}",
    )
    historical_tts_hash = str(historical.get("tts_audio_sha256", ""))
    if historical_tts_hash != cloud_audio["sha256"]:
        raise ProtocolError(f"historical target does not point to this cloud TTS canonical audio: {sample_id}")
    historical_audio = _asset_from_path(
        str(historical.get("candidate_audio", "")),
        str(historical.get("candidate_audio_sha256", "")),
        f"historical cloud target {sample_id}",
    )
    if historical.get("natural_audio_sha256") != natural["sha256"] or historical.get("natural_samples") != parent_row.get("natural_sample_count"):
        raise ProtocolError(f"historical target natural binding mismatch: {sample_id}")
    record = {
        "sample_id": sample_id,
        "source_group": group,
        "transcript": transcript,
        "language": config.LANGUAGE,
        "historical_exposure": True,
        "scope": "fit_only_paired_pilot",
        "natural_audio": {
            **natural,
            "sample_count": int(parent_row.get("natural_sample_count", cloud.get("natural_samples", -1))),
            "header": _read_pcm_header(Path(natural["path"])),
        },
        "face_video": face,
        "cloud_tts": {
            "provider": config.CLOUD_PROVIDER,
            "model": config.CLOUD_MODEL,
            "voice_id": str(cloud.get("voice_id", "")),
            "provider_audio": provider_audio,
            "provider_sample_rate_hz": int(cloud.get("provider_sample_rate_hz", -1)),
            "provider_duration_s": float(cloud.get("provider_duration_s", -1.0)),
            "canonical_audio": cloud_audio,
            "canonical_samples": int(cloud.get("canonical_samples", -1)),
            "duration_ratio": float(cloud.get("duration_ratio", -1.0)),
            "metadata_record_sha256": canonical_json_sha256(dict(cloud)),
        },
        "historical_cloud_target": {
            **historical_audio,
            "tts_audio_sha256": historical_tts_hash,
            "mapping": historical.get("mapping"),
        },
    }
    audit = {
        "sample_id": sample_id,
        "source_group": group,
        "metadata_chain": {
            "cloud_reference_matches_natural": True,
            "cloud_video_matches_face": True,
            "historical_target_tts_matches_cloud": True,
            "historical_target_natural_matches_natural": True,
        },
        "bound_media": [
            {"kind": "natural", **natural},
            {"kind": "face_video", **face},
            {"kind": "cloud_provider_audio", **provider_audio},
            {"kind": "cloud_canonical_audio", **cloud_audio},
            {"kind": "historical_cloud_target", **historical_audio},
        ],
        "local_paired_assets_verified": False,
    }
    return record, audit


def load_frozen_cohort() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_binding = _validate_input_bindings()
    parent_rows, parent_payload = _validate_parent_cohort()
    _validate_parent_protocol()
    cloud_by_id = _cloud_rows()
    historical_by_id = _historical_rows()
    records: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for row in parent_rows:
        sample_id = str(row["sample_id"])
        cloud = cloud_by_id.get(sample_id)
        historical = historical_by_id.get(sample_id)
        if cloud is None or historical is None:
            raise ProtocolError(f"missing cloud or historical target record: {sample_id}")
        record, audit = _record(row, cloud, historical)
        records.append(record)
        audits.append(audit)
    return records, {
        "parent_cohort_sha256": file_sha256(config.CONFIRMATION_COHORT),
        "parent_protocol_sha256": file_sha256(config.PARENT_PROTOCOL),
        "cloud_tts_metadata_sha256": file_sha256(config.CLOUD_TTS_METADATA),
        "historical_cloud_target_sha256": file_sha256(config.HISTORICAL_CLOUD_TARGET),
        "parent_record_count": len(parent_rows),
        "record_count": len(records),
        "source_group_count": len({str(row["source_group"]) for row in records}),
        "sample_ids_sha256": sample_ids_sha256([str(row["sample_id"]) for row in records]),
        "selection": "all 22 records in confirmation cohort original order",
        "uses_scores": False,
        "historical_exposure": True,
        "scope": "fit_only_paired_pilot",
        "audits": audits,
        "parent_cohort_self_hash": parent_payload.get("artifact_sha256"),
        "input_bindings_sha256": input_binding["sha256"],
    }


def _runtime_bindings() -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    missing: list[str] = []
    for name, path in config.runtime_paths().items():
        if path.is_file():
            bindings[name] = runtime_binding(path)
        else:
            bindings[name] = {"path": str(path.resolve()), "sha256": None, "missing": True}
            missing.append(name)
    return {"bindings": bindings, "missing": missing, "versions": runtime_versions()}


def build_setup(records: list[Mapping[str, Any]], selection: Mapping[str, Any]) -> dict[str, Any]:
    execution_items = [
        {"sample_id": str(row["sample_id"]), "render_repeat": repeat, "video_arm": arm}
        for repeat in config.REPEATS
        for row in records
        for arm in config.ARMS
    ]
    return {
        "schema_version": 1,
        "stage_id": "00_protocol",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": "complete",
        "historical_exposure": True,
        "scope": "fit_only_paired_pilot",
        "config": config.FrozenConfig().to_dict(),
        "cohort": {
            "record_count": len(records),
            "source_group_count": len({str(row["source_group"]) for row in records}),
            "ordered_sample_ids": [str(row["sample_id"]) for row in records],
            "ordered_sample_ids_sha256": sample_ids_sha256([str(row["sample_id"]) for row in records]),
        },
        "providers": {
            "LOCAL": {
                "provider": config.LOCAL_PROVIDER,
                "model": config.LOCAL_MODEL,
                "clone_mode": config.LOCAL_CLONE_MODE,
                "language": config.LANGUAGE,
                "reference": "same natural audio and cohort transcript",
                "new_success_budget": config.EXPECTED_RECORD_COUNT,
            },
            "CLOUD": {
                "provider": config.CLOUD_PROVIDER,
                "model": config.CLOUD_MODEL,
                "language": config.LANGUAGE,
                "new_call_budget": 0,
                "metadata_source": str(config.CLOUD_TTS_METADATA.resolve()),
            },
        },
        "selection": dict(selection),
        "runtime": _runtime_bindings(),
        "randomness": {
            "local_seed_formula": "int(sha256(sample_id UTF-8).hexdigest()[:8], 16)",
            "render_seed_by_repeat": {"0": 20260913, "1": 20260914},
            "render_schedule_seed": config.BOOTSTRAP_SEED,
            "quality_blind_seed": config.BOOTSTRAP_SEED,
        },
        "matrix": {
            "driver_arms": list(config.ARMS),
            "render_repeats": list(config.REPEATS),
            "score_cells_per_record_repeat": [
                {"video_arm": video, "score_audio_arm": audio, "purpose": purpose}
                for video, audio, purpose in config.SCORE_CELLS
            ],
            "expected_videos": config.EXPECTED_VIDEO_COUNT,
            "expected_score_cells": config.EXPECTED_CELL_COUNT,
        },
        "quality": {
            "stimuli": ["T_LOCAL", "T_CLOUD", "M_LOCAL", "M_CLOUD"],
            "expected_stimuli": config.EXPECTED_QUALITY_STIMULI,
            "minimum_raters_per_pair": config.MIN_RATERS_PER_PAIR,
            "ratings_do_not_enter_model_or_syncnet": True,
        },
        "selection_forbidden": [
            "historical_score_read",
            "quality_based_selection",
            "provider_swap_after_failure",
            "record_substitution",
            "score_based_retry",
        ],
        "records": list(records),
        "execution_item_count": len(execution_items),
    }


def build_execution_order(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    items = [
        {"sample_id": str(row["sample_id"]), "render_repeat": repeat, "video_arm": arm}
        for repeat in config.REPEATS
        for row in records
        for arm in config.ARMS
    ]
    rng = np.random.Generator(np.random.PCG64(config.BOOTSTRAP_SEED))
    order = rng.permutation(len(items))
    return [{"ordinal": ordinal, **items[int(index)]} for ordinal, index in enumerate(order)]


def run_audit(run_root: Path) -> dict[str, Any]:
    from .common import assert_run_root_compatible

    assert_run_root_compatible(run_root)
    paths = config.RunPaths(run_root)
    paths.protocol.mkdir(parents=True, exist_ok=True)
    try:
        records, selection = load_frozen_cohort()
        setup = build_setup(records, selection)
        cohort = {
            "schema_version": 1,
            "stage_id": "00_protocol",
            "protocol_id": config.PROTOCOL_ID,
            "status": "complete",
            "record_count": len(records),
            "source_group_count": len({str(row["source_group"]) for row in records}),
            "sample_ids_sha256": sample_ids_sha256([str(row["sample_id"]) for row in records]),
            "selection": selection,
            "records": records,
        }
        input_audit = {
            "schema_version": 1,
            "stage_id": "00_protocol",
            "protocol_id": config.PROTOCOL_ID,
            "status": "complete",
            "metadata_chain_checks": len(selection["audits"]),
            "bound_media_file_checks": sum(len(row["bound_media"]) for row in selection["audits"]),
            "errors": [],
            "local_paired_assets_verified": False,
            "local_asset_action": "audit current-run compatible assets, otherwise synthesize once in Stage 01",
            "perceptual_quality_assessed": False,
            "new_inference_executed": False,
            "score_read_count": 0,
            "sealed_media_accessed": False,
            "records": selection["audits"],
        }
        setup_hash = write_self_hashed_json(paths.protocol / "setup.json", setup)
        write_self_hashed_json(paths.protocol / "cohort.json", cohort)
        write_self_hashed_json(paths.protocol / "input_audit.json", input_audit)
        order = build_execution_order(records)
        write_self_hashed_json(
            paths.protocol / "execution_order.json",
            {
                "schema_version": 1,
                "stage_id": "00_protocol",
                "protocol_id": config.PROTOCOL_ID,
                "seed": config.BOOTSTRAP_SEED,
                "policy": "one serial item at a time; PCG64 permutation of sample/repeat/arm",
                "items": order,
            },
        )
        write_self_hashed_json(
            paths.protocol / "failures.json",
            {"schema_version": 1, "stage_id": "00_protocol", "protocol_id": config.PROTOCOL_ID, "status": "complete", "failures": []},
        )
        return {"status": "complete", "setup_sha256": setup_hash, "record_count": len(records)}
    except Exception as exc:
        write_self_hashed_json(
            paths.protocol / "input_audit.json",
            {
                "schema_version": 1,
                "stage_id": "00_protocol",
                "protocol_id": config.PROTOCOL_ID,
                "status": "blocked",
                "errors": [{"error_type": type(exc).__name__, "error": str(exc)}],
                "score_read_count": 0,
                "sealed_media_accessed": False,
            },
        )
        write_self_hashed_json(
            paths.protocol / "failures.json",
            {
                "schema_version": 1,
                "stage_id": "00_protocol",
                "protocol_id": config.PROTOCOL_ID,
                "status": "blocked",
                "failures": [{"error_type": type(exc).__name__, "error": str(exc)}],
            },
        )
        raise
