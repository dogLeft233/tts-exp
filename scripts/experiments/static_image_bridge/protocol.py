from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import config
from .common import (
    ProtocolError,
    assert_not_sealed,
    file_sha256,
    read_pcm16,
    require_hash,
    sample_ids_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def _asset(path_value: Any, expected_hash: Any, label: str, *, allow_sealed: bool = False) -> dict[str, str]:
    path = Path(str(path_value)).resolve()
    expected = str(expected_hash)
    if not path.is_file() or file_sha256(path) != expected:
        raise ProtocolError(f"{label} binding changed: {path}")
    if not allow_sealed:
        assert_not_sealed(path)
    return {"path": str(path), "container_sha256": expected}


def _audio_asset(audio_row: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = Path(str(audio_row.get("output", ""))).resolve()
    expected_container = str(audio_row.get("output_sha256", ""))
    if not path.is_file() or file_sha256(path) != expected_container:
        raise ProtocolError(f"{label} container binding changed: {path}")
    assert_not_sealed(path)
    values, meta = read_pcm16(path)
    return {
        "path": str(path),
        "container_sha256": expected_container,
        "pcm_sha256": meta["pcm_sha256"],
        "historical_declared_pcm_sha256": str(audio_row.get("format", {}).get("pcm_sha256", "")),
        "sample_count": int(values.size),
        "sample_rate": config.SAMPLE_RATE,
        "channels": config.PCM_CHANNELS,
        "sample_width": config.PCM_SAMPLE_WIDTH,
    }


def load_frozen_inputs() -> dict[str, Any]:
    require_hash(config.PARENT_COHORT, config.PARENT_COHORT_SHA256, "parent cohort")
    require_hash(config.PARENT_AUDIO, config.PARENT_AUDIO_SHA256, "parent audio manifest")
    require_hash(config.PARENT_ANALYSIS, config.PARENT_ANALYSIS_SHA256, "parent analysis")
    require_hash(config.DISCOVERY_ANALYSIS, config.DISCOVERY_ANALYSIS_SHA256, "discovery analysis")
    cohort = verify_self_hashed_json(config.PARENT_COHORT)
    audio_manifest = verify_self_hashed_json(config.PARENT_AUDIO)
    parent_analysis = verify_self_hashed_json(config.PARENT_ANALYSIS)
    discovery_analysis = verify_self_hashed_json(config.DISCOVERY_ANALYSIS)
    records = cohort.get("records")
    if cohort.get("status") != "complete" or not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("historical cohort is not the frozen complete 22-record cohort")
    if int(cohort.get("source_group_count", -1)) != config.EXPECTED_SOURCE_GROUP_COUNT:
        raise ProtocolError("historical source-group count changed")
    ids = [str(row.get("sample_id", "")) for row in records]
    groups = [str(row.get("source_group", "")) for row in records]
    if len(set(ids)) != len(ids) or len(set(groups)) != len(groups):
        raise ProtocolError("historical sample IDs or source groups are not unique")
    if sample_ids_sha256(ids) != config.EXPECTED_SAMPLE_IDS_SHA256:
        raise ProtocolError("historical sample order hash changed")
    audio_rows = {str(row.get("sample_id")): row for row in audio_manifest.get("rows", [])}
    if audio_manifest.get("status") != "complete" or len(audio_rows) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("historical audio manifest is incomplete")

    output: list[dict[str, Any]] = []
    for cohort_row in records:
        sid = str(cohort_row["sample_id"])
        source_group = str(cohort_row["source_group"])
        face = cohort_row.get("face_video", {})
        face_asset = _asset(face.get("path"), face.get("sha256"), f"face video {sid}")
        audio_row = audio_rows.get(sid)
        if not isinstance(audio_row, Mapping):
            raise ProtocolError(f"historical audio row missing: {sid}")
        arms = {str(row.get("arm")): row for row in audio_row.get("arms", [])}
        if set(arms) != {"N", "N_REPEAT", "LOCAL_SWAP", "BRIDGE_075"}:
            raise ProtocolError(f"historical audio arms incomplete: {sid}")
        n = _audio_asset(arms["N"], f"N {sid}")
        n_repeat = _audio_asset(arms["N_REPEAT"], f"N_REPEAT {sid}")
        bridge = _audio_asset(arms["BRIDGE_075"], f"BRIDGE_075 {sid}")
        swap = _audio_asset(arms["LOCAL_SWAP"], f"LOCAL_SWAP {sid}")
        if n["pcm_sha256"] != n_repeat["pcm_sha256"] or n["sample_count"] != int(cohort_row.get("natural_sample_count", -1)):
            raise ProtocolError(f"historical N/N_REPEAT identity or length changed: {sid}")
        if bridge["sample_count"] != n["sample_count"] or swap["sample_count"] != n["sample_count"]:
            raise ProtocolError(f"historical arm length changed: {sid}")
        output.append({
            "sample_id": sid,
            "source_group": source_group,
            "seen_fit": True,
            "face_video": face_asset,
            "audio": {"N": n, "B": bridge, "S": swap},
            "historical_audio": {"N_REPEAT": n_repeat, "BRIDGE_075": bridge, "LOCAL_SWAP": swap},
            "natural_sample_count": int(n["sample_count"]),
            "transcript": str(cohort_row.get("transcript", "")),
        })
    return {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "seen_fit": True,
        "record_count": len(output),
        "source_group_count": len({row["source_group"] for row in output}),
        "ordered_sample_id_sha256": sample_ids_sha256([row["sample_id"] for row in output]),
        "parents": {
            "cohort": {"path": str(config.PARENT_COHORT.resolve()), "sha256": config.PARENT_COHORT_SHA256},
            "audio_manifest": {"path": str(config.PARENT_AUDIO.resolve()), "sha256": config.PARENT_AUDIO_SHA256},
            "analysis": {"path": str(config.PARENT_ANALYSIS.resolve()), "sha256": config.PARENT_ANALYSIS_SHA256},
            "discovery_analysis": {"path": str(config.DISCOVERY_ANALYSIS.resolve()), "sha256": config.DISCOVERY_ANALYSIS_SHA256},
        },
        "historical_decision": parent_analysis.get("decisions", parent_analysis.get("scientific_decision")),
        "discovery_reference": {"analysis_sha256": config.DISCOVERY_ANALYSIS_SHA256, "status": discovery_analysis.get("status")},
        "records": output,
    }


def build_protocol(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "status": "complete",
        "seen_fit": True,
        "record_count": config.EXPECTED_RECORD_COUNT,
        "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
        "expected_sample_ids_sha256": config.EXPECTED_SAMPLE_IDS_SHA256,
        "cohort_order": [row["sample_id"] for row in inputs["records"]],
        "config": {
            "sample_rate": config.SAMPLE_RATE,
            "channels": config.PCM_CHANNELS,
            "sample_width": config.PCM_SAMPLE_WIDTH,
            "fps": config.FPS,
            "seed": config.SEED,
            "face_detection_threshold": config.FACE_DET_THRESHOLD,
            "generation_bottom_pad": config.GENERATION_BOTTOM_PAD,
            "syncnet_vshift": config.SYNCNET_VSHIFT,
            "bootstrap_draws": config.BOOTSTRAP_DRAWS,
            "bootstrap_seed": config.BOOTSTRAP_SEED,
            "arms": list(config.ARMS),
            "stage_a_videos": list(config.STAGE_A_VIDEOS),
            "stage_b_videos": list(config.STAGE_B_VIDEOS),
            "stage_a_cells": [[v, a] for v, a in config.STAGE_A_CELLS],
            "stage_b_cells": [[v, a] for v, a in config.STAGE_B_CELLS],
            "no_dynamic_input": True,
            "source_frame_index": 0,
            "audio_hash_semantics": "decoded_pcm_sha256_is distinct from container_sha256",
        },
        "runtime": config.runtime_bindings(),
        "parents": inputs["parents"],
        "forbidden_operations": [
            "source_frame_fallback", "dynamic_frame_input", "candidate_specific_crop", "score_based_retry",
            "record_substitution", "strength_search", "training", "fine_tuning", "tts", "asr", "mfa", "dtw",
        ],
        "inputs_sha256": inputs["artifact_sha256"],
    }


def prepare_protocol(paths: config.RunPaths, inputs_override: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    inputs = dict(inputs_override) if inputs_override is not None else load_frozen_inputs()
    if paths.inputs.is_file():
        existing = verify_self_hashed_json(paths.inputs)
        if existing.get("records") != inputs["records"] or existing.get("parents") != inputs["parents"]:
            raise ProtocolError("existing inputs.json differs from the frozen parent cohort")
        inputs = existing
    else:
        write_self_hashed_json(paths.inputs, inputs)
        inputs = verify_self_hashed_json(paths.inputs)
    protocol = build_protocol(inputs)
    if paths.protocol.is_file():
        existing_protocol = verify_self_hashed_json(paths.protocol)
        # ``protocol.json`` is self-hashed, while ``build_protocol`` returns
        # the unhashed body. Compare semantic payloads so a prepared run is
        # genuinely idempotent and can be resumed without a false mismatch.
        existing_body = dict(existing_protocol)
        existing_body.pop("artifact_sha256", None)
        if existing_body != protocol:
            raise ProtocolError("existing protocol.json differs from the frozen protocol")
        protocol = existing_protocol
    else:
        write_self_hashed_json(paths.protocol, protocol)
        protocol = verify_self_hashed_json(paths.protocol)
    return protocol, inputs


def load_protocol(paths: config.RunPaths) -> dict[str, Any]:
    protocol = verify_self_hashed_json(paths.protocol)
    if protocol.get("protocol_id") != config.PROTOCOL_ID or protocol.get("record_count") != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("protocol identity or count is invalid")
    return protocol


def load_inputs(paths: config.RunPaths) -> dict[str, Any]:
    inputs = verify_self_hashed_json(paths.inputs)
    if inputs.get("status") != "complete" or len(inputs.get("records", [])) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("inputs are incomplete")
    return inputs


def expected_runtime_hashes() -> dict[str, str]:
    return {
        "wav2lip_checkpoint": config.WAV2LIP_CHECKPOINT_SHA256,
        "syncnet_model": config.SYNCNET_MODEL_SHA256,
        "ffmpeg": config.FFMPEG_SHA256,
        "ffprobe": config.FFPROBE_SHA256,
        "wav2lip_python": config.PYTHON_SHA256,
        "syncnet_python": config.PYTHON_SHA256,
    }
