from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.lrs3_real_video_local_timing.media import (
    crop_frames_from_track,
    decode_video_frames,
    encode_ffv1,
    mux_pcm,
    read_video_frames,
    source_pcm16,
    verify_muxed_media,
)
from scripts.experiments.lrs3_wav2lip_timing_transfer.protocol import (
    media_video_timeline,
)

from . import config
from .common import (
    ProtocolError,
    bytes_sha256,
    canonical_json_sha256,
    file_sha256,
    verify_self_hashed_json,
    write_or_verify,
    write_self_hashed_json,
)


def _video_row(video_manifest: Mapping[str, Any], sample_id: str, arm: str) -> Mapping[str, Any]:
    rows = video_manifest.get("rows")
    if not isinstance(rows, list):
        raise ProtocolError("video manifest rows are missing")
    row = next((item for item in rows if isinstance(item, Mapping) and str(item.get("sample_id")) == sample_id), None)
    if not isinstance(row, Mapping):
        raise ProtocolError(f"video manifest row is missing: {sample_id}")
    arms = row.get("arms")
    if not isinstance(arms, Mapping) or not isinstance(arms.get(arm), Mapping):
        raise ProtocolError(f"video arm is missing: {sample_id}/{arm}")
    return arms[arm]


def _audio_row(record: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    row = record.get("audio", {}).get("manifest_row")
    if not isinstance(row, Mapping):
        raise ProtocolError(f"audio row is missing: {record.get('sample_id')}")
    arms = row.get("arms")
    if not isinstance(arms, list):
        raise ProtocolError(f"audio arms are missing: {record.get('sample_id')}")
    value = next((item for item in arms if isinstance(item, Mapping) and str(item.get("arm")) == arm), None)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"audio arm is missing: {record.get('sample_id')}/{arm}")
    return value


def _stream_source(record: Mapping[str, Any], video_manifest: Mapping[str, Any], stage: str, video_arm: str) -> tuple[Path, str, int]:
    if video_arm == config.VIDEO_R:
        item = record["face_video"]
        path = Path(str(item["path"]))
        expected = str(item["sha256"])
        frame_count = int(record["predicted_frame_counts"][config.VIDEO_R])
    else:
        item = _video_row(video_manifest, str(record["sample_id"]), video_arm)
        path = Path(str(item["output"]))
        expected = str(item["output_sha256"])
        frame_count = int(record["predicted_frame_counts"][video_arm])
    if not path.is_file() or file_sha256(path) != expected:
        raise ProtocolError(f"source video hash changed: {record.get('sample_id')}/{video_arm}")
    return path, expected, frame_count


def _materialize_stream(record: Mapping[str, Any], video_manifest: Mapping[str, Any], paths: config.RunPaths, stage: str, video_arm: str) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    source, source_sha, frame_count = _stream_source(record, video_manifest, stage, video_arm)
    source_frames = read_video_frames(source)
    if len(source_frames) != frame_count:
        raise ProtocolError(f"video frame count differs from protocol: {sample_id}/{video_arm}: {len(source_frames)} != {frame_count}")
    track = record["processed_track"]
    if not isinstance(track, list) or len(track) < frame_count:
        raise ProtocolError(f"processed crop track is too short: {sample_id}/{video_arm}")
    cropped = crop_frames_from_track(source_frames, track[:frame_count])
    output = paths.media / stage / "streams" / f"{sample_id}__{video_arm}.mkv"
    sidecar = output.with_suffix(".json")
    expected_track_hash = canonical_json_sha256(track[:frame_count])
    if output.is_file() and sidecar.is_file():
        prior = verify_self_hashed_json(sidecar)
        if prior.get("protocol_id") != config.PROTOCOL_ID or prior.get("stage") != stage or prior.get("sample_id") != sample_id or prior.get("video_arm") != video_arm or prior.get("source_video_sha256") != source_sha or prior.get("track_sha256") != expected_track_hash or prior.get("output_sha256") != file_sha256(output):
            raise ProtocolError(f"existing stream identity changed: {sample_id}/{video_arm}")
        actual = decode_video_frames(output)
        if len(actual) != len(cropped) or any(not np.array_equal(a, b) for a, b in zip(actual, cropped, strict=True)):
            raise ProtocolError(f"existing stream pixels changed: {sample_id}/{video_arm}")
        encoded = prior
    elif output.exists() or sidecar.exists():
        raise ProtocolError(f"partial stream cannot be resumed: {sample_id}/{video_arm}")
    else:
        encoded = encode_ffv1(cropped, output, paths.media / stage / "logs" / f"{sample_id}.{video_arm}.encode.log")
        write_self_hashed_json(sidecar, {"schema_version": 1, "stage_id": "media_stream", "protocol_id": config.PROTOCOL_ID, "stage": stage, "sample_id": sample_id, "source_group": str(record["source_group"]), "video_arm": video_arm, "source_video": str(source.resolve()), "source_video_sha256": source_sha, "track_sha256": expected_track_hash, "frame_count": frame_count, "crop_size": config.CROP_SIZE, "crop_scale": config.CROP_SCALE, **encoded})
        encoded = verify_self_hashed_json(sidecar)
    timeline = media_video_timeline(output)
    if int(timeline["frame_count"]) != frame_count or int(timeline["width"]) != config.CROP_SIZE or int(timeline["height"]) != config.CROP_SIZE:
        raise ProtocolError(f"materialized stream timeline is invalid: {sample_id}/{video_arm}")
    return {"video_arm": video_arm, "output": str(output.resolve()), "output_sha256": file_sha256(output), "sidecar": str(sidecar.resolve()), "source_video": str(source.resolve()), "source_video_sha256": source_sha, "track_sha256": expected_track_hash, "frame_count": frame_count, "timeline": timeline, "encoded": {key: encoded.get(key) for key in ("raw_frame_sha256", "command_sha256", "frame_count")}}


def _materialize_cell(record: Mapping[str, Any], stream: Mapping[str, Any], paths: config.RunPaths, stage: str, audio_arm: str) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    video_arm = str(stream["video_arm"])
    audio_item = _audio_row(record, audio_arm)
    audio = Path(str(audio_item.get("output", audio_item.get("path", ""))))
    audio_sha = str(audio_item.get("output_sha256", audio_item.get("container_sha256", "")))
    if not audio.is_file() or not audio_sha or file_sha256(audio) != audio_sha:
        raise ProtocolError(f"audio hash changed: {sample_id}/{audio_arm}")
    expected_pcm = source_pcm16(audio)
    output = paths.media / stage / "cells" / f"{sample_id}__{video_arm}__{audio_arm}.mkv"
    sidecar = output.with_suffix(".json")
    stream_path = Path(str(stream["output"]))
    if output.is_file() and sidecar.is_file():
        prior = verify_self_hashed_json(sidecar)
        if prior.get("protocol_id") != config.PROTOCOL_ID or prior.get("stage") != stage or prior.get("sample_id") != sample_id or prior.get("video_arm") != video_arm or prior.get("audio_arm") != audio_arm or prior.get("video_stream_sha256") != str(stream["output_sha256"]) or prior.get("audio_container_sha256") != audio_sha or prior.get("audio_pcm_sha256") != bytes_sha256(expected_pcm) or prior.get("output_sha256") != file_sha256(output):
            raise ProtocolError(f"existing media cell identity changed: {sample_id}/{video_arm}/{audio_arm}")
    elif output.exists() or sidecar.exists():
        raise ProtocolError(f"partial media cell cannot be resumed: {sample_id}/{video_arm}/{audio_arm}")
    else:
        mux = mux_pcm(stream_path, audio, output, paths.media / stage / "logs" / f"{sample_id}.{video_arm}.{audio_arm}.mux.log")
        write_self_hashed_json(sidecar, {"schema_version": 1, "stage_id": "media_cell", "protocol_id": config.PROTOCOL_ID, "stage": stage, "sample_id": sample_id, "source_group": str(record["source_group"]), "video_arm": video_arm, "audio_arm": audio_arm, "output": str(output.resolve()), "video_stream": str(stream_path.resolve()), "video_stream_sha256": str(stream["output_sha256"]), "audio": str(audio.resolve()), "audio_container_sha256": audio_sha, "audio_pcm_sha256": bytes_sha256(expected_pcm), "audio_modified": False, **mux})
    frames = decode_video_frames(stream_path)
    verification = verify_muxed_media(output, frames, expected_pcm, list(range(len(frames))))
    timeline = media_video_timeline(output)
    if int(timeline["frame_count"]) != len(frames):
        raise ProtocolError(f"muxed cell frame count changed: {sample_id}/{video_arm}/{audio_arm}")
    return {"video_arm": video_arm, "audio_arm": audio_arm, "output": str(output.resolve()), "output_sha256": file_sha256(output), "sidecar": str(sidecar.resolve()), "video_stream": str(stream_path.resolve()), "video_stream_sha256": str(stream["output_sha256"]), "audio": str(audio.resolve()), "audio_container_sha256": audio_sha, "audio_pcm_sha256": bytes_sha256(expected_pcm), "frame_count": len(frames), "timeline": timeline, "verification": verification}


def materialize_stage(protocol: Mapping[str, Any], video_manifest: Mapping[str, Any], paths: config.RunPaths, stage: str) -> dict[str, Any]:
    if stage == "control":
        video_arms = config.CONTROL_VIDEO_ARMS
        cell_specs = config.CONTROL_MAIN_CELL_SPECS
        expected_cells = config.EXPECTED_CONTROL_MAIN_CELL_COUNT
    elif stage == "bridge":
        video_arms = (config.VIDEO_GB,)
        cell_specs = config.BRIDGE_CELL_SPECS
        expected_cells = config.EXPECTED_BRIDGE_CELL_COUNT
    else:
        raise ValueError(f"unknown media stage: {stage}")
    if video_manifest.get("status") != "complete":
        raise ProtocolError(f"{stage} video manifest is incomplete")
    protocol_sha = file_sha256(paths.protocol)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(protocol["records"], 1):
        streams = {arm: _materialize_stream(record, video_manifest, paths, stage, arm) for arm in video_arms}
        cells = {config.cell_key(video, audio): _materialize_cell(record, streams[video], paths, stage, audio) for video, audio in cell_specs}
        rows.append({"sample_id": str(record["sample_id"]), "source_group": str(record["source_group"]), "streams": streams, "cells": cells})
        print(f"MEDIA {index}/{len(protocol['records'])} {record['sample_id']} stage={stage}", flush=True)
    payload = {"schema_version": 1, "stage_id": "media", "protocol_id": config.PROTOCOL_ID, "stage": stage, "protocol_sha256": protocol_sha, "video_manifest_sha256": file_sha256(paths.videos / stage / "manifest.json"), "status": "complete", "record_count": len(rows), "expected_record_count": config.EXPECTED_RECORD_COUNT, "video_stream_count": len(rows) * len(video_arms), "cell_count": sum(len(row["cells"]) for row in rows), "expected_cell_count": expected_cells, "audio_modified": False, "rows": rows}
    return write_or_verify(paths.media / stage / "manifest.json", payload)
