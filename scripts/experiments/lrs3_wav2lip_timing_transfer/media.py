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

from . import config
from .common import (
    DiagnosticError,
    bytes_sha256,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .protocol import media_video_timeline


def _record_index(protocol: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = protocol.get("records")
    if not isinstance(records, list):
        raise DiagnosticError("protocol records are missing")
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise DiagnosticError("protocol record is malformed")
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in result:
            raise DiagnosticError(f"duplicate protocol sample: {sample_id}")
        result[sample_id] = record
    return result


def _stream_spec(record: Mapping[str, Any], video_arm: str) -> tuple[Path, str, list[dict[str, float]]]:
    if video_arm == config.VIDEO_R:
        item = record.get("real_video")
    else:
        item = record.get("generated_videos", {}).get(video_arm)
    if not isinstance(item, Mapping):
        raise DiagnosticError(f"video source is missing: {record.get('sample_id')}/{video_arm}")
    path = Path(str(item.get("path", "")))
    expected = str(item.get("sha256", ""))
    if not path.is_file() or not expected or file_sha256(path) != expected:
        raise DiagnosticError(f"video source hash changed: {record.get('sample_id')}/{video_arm}")
    track = record.get("processed_track")
    if not isinstance(track, list) or video_arm == config.VIDEO_R and len(track) != int(item.get("timeline", {}).get("frame_count", -1)):
        raise DiagnosticError(f"processed track is invalid: {record.get('sample_id')}/{video_arm}")
    return path, expected, [dict(value) for value in track[: int(item.get("timeline", {}).get("frame_count", 0))] if isinstance(value, Mapping)]


def _stream_sidecar_matches(sidecar: Mapping[str, Any], protocol_sha: str, sample_id: str, video_arm: str, source_sha256: str, track_sha256: str, output: Path) -> bool:
    return bool(
        sidecar.get("protocol_id") == config.PROTOCOL_ID
        and sidecar.get("protocol_sha256") == protocol_sha
        and sidecar.get("sample_id") == sample_id
        and sidecar.get("video_arm") == video_arm
        and sidecar.get("source_video_sha256") == source_sha256
        and sidecar.get("track_sha256") == track_sha256
        and sidecar.get("output_sha256") == file_sha256(output)
    )


def _materialize_video_stream(paths: config.RunPaths, protocol_sha: str, record: Mapping[str, Any], video_arm: str) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    source, source_sha256, track = _stream_spec(record, video_arm)
    source_frames = read_video_frames(source)
    expected_count = len(track)
    if video_arm == config.VIDEO_R:
        expected_count = len(source_frames)
    if expected_count <= 0 or len(source_frames) < expected_count:
        raise DiagnosticError(f"video source frame count is invalid: {sample_id}/{video_arm}")
    if len(track) != expected_count:
        raise DiagnosticError(f"video track prefix length is invalid: {sample_id}/{video_arm}")
    cropped = crop_frames_from_track(source_frames[:expected_count], track)
    track_sha256 = str(record["track_sha256"])
    output = paths.media / "streams" / f"{sample_id}__{video_arm}.mkv"
    sidecar_path = output.with_suffix(".json")
    if output.is_file() and sidecar_path.is_file():
        sidecar = verify_self_hashed_json(sidecar_path)
        if not _stream_sidecar_matches(sidecar, protocol_sha, sample_id, video_arm, source_sha256, track_sha256, output):
            raise DiagnosticError(f"existing video stream identity changed: {sample_id}/{video_arm}")
        actual = decode_video_frames(output)
        if len(actual) != len(cropped) or any(not np.array_equal(left, right) for left, right in zip(actual, cropped, strict=True)):
            raise DiagnosticError(f"existing video stream pixels changed: {sample_id}/{video_arm}")
        encoded = dict(sidecar)
    elif output.exists() or sidecar_path.exists():
        raise DiagnosticError(f"partial video stream cannot be resumed: {sample_id}/{video_arm}")
    else:
        encoded = encode_ffv1(cropped, output, paths.media / "logs" / f"{sample_id}.{video_arm}.encode.log")
        sidecar_body = {
            "schema_version": 1,
            "protocol_id": config.PROTOCOL_ID,
            "protocol_sha256": protocol_sha,
            "sample_id": sample_id,
            "source_group": str(record["source_group"]),
            "video_arm": video_arm,
            "source_video": str(source),
            "source_video_sha256": source_sha256,
            "track_sha256": track_sha256,
            "frame_count": len(cropped),
            "crop_size": config.CROP_SIZE,
            "crop_scale": config.CROP_SCALE,
            **encoded,
        }
        write_self_hashed_json(sidecar_path, sidecar_body)
        encoded = verify_self_hashed_json(sidecar_path)
    timeline = media_video_timeline(output)
    if int(timeline["frame_count"]) != len(cropped) or int(timeline["width"]) != config.CROP_SIZE or int(timeline["height"]) != config.CROP_SIZE:
        raise DiagnosticError(f"materialized video stream timeline is invalid: {sample_id}/{video_arm}")
    return {
        "video_arm": video_arm,
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "sidecar": str(sidecar_path.resolve()),
        "source_video": str(source.resolve()),
        "source_video_sha256": source_sha256,
        "track_sha256": track_sha256,
        "frame_count": len(cropped),
        "timeline": timeline,
        "encoded": {key: encoded.get(key) for key in ("raw_frame_sha256", "command_sha256", "frame_count")},
    }


def _cell_sidecar_matches(sidecar: Mapping[str, Any], protocol_sha: str, sample_id: str, video_arm: str, audio_arm: str, stream_sha256: str, audio_sha256: str, audio_pcm_sha256: str, output: Path) -> bool:
    return bool(
        sidecar.get("protocol_id") == config.PROTOCOL_ID
        and sidecar.get("protocol_sha256") == protocol_sha
        and sidecar.get("sample_id") == sample_id
        and sidecar.get("video_arm") == video_arm
        and sidecar.get("audio_arm") == audio_arm
        and sidecar.get("video_stream_sha256") == stream_sha256
        and sidecar.get("audio_container_sha256") == audio_sha256
        and sidecar.get("audio_pcm_sha256") == audio_pcm_sha256
        and sidecar.get("output_sha256") == file_sha256(output)
    )


def _materialize_cell(paths: config.RunPaths, protocol_sha: str, record: Mapping[str, Any], stream: Mapping[str, Any], audio_arm: str) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    video_arm = str(stream["video_arm"])
    audio_path = Path(str(record["natural_audio"]["path"])) if audio_arm == config.AUDIO_N else Path(str(record["warp_audio"]["path"]))
    if not audio_path.is_file():
        raise DiagnosticError(f"audio input is missing: {sample_id}/{audio_arm}")
    audio_sha256 = file_sha256(audio_path)
    expected_pcm = source_pcm16(audio_path)
    audio_pcm_sha256 = bytes_sha256(expected_pcm)
    output = paths.media / "cells" / f"{sample_id}__{video_arm}__{audio_arm}.mkv"
    sidecar_path = output.with_suffix(".json")
    stream_path = Path(str(stream["output"]))
    if output.is_file() and sidecar_path.is_file():
        sidecar = verify_self_hashed_json(sidecar_path)
        if not _cell_sidecar_matches(sidecar, protocol_sha, sample_id, video_arm, audio_arm, str(stream["output_sha256"]), audio_sha256, audio_pcm_sha256, output):
            raise DiagnosticError(f"existing media cell identity changed: {sample_id}/{video_arm}/{audio_arm}")
    elif output.exists() or sidecar_path.exists():
        raise DiagnosticError(f"partial media cell cannot be resumed: {sample_id}/{video_arm}/{audio_arm}")
    else:
        mux = mux_pcm(stream_path, audio_path, output, paths.media / "logs" / f"{sample_id}.{video_arm}.{audio_arm}.mux.log")
        sidecar_body = {
            "schema_version": 1,
            "protocol_id": config.PROTOCOL_ID,
            "protocol_sha256": protocol_sha,
            "sample_id": sample_id,
            "source_group": str(record["source_group"]),
            "video_arm": video_arm,
            "audio_arm": audio_arm,
            "output": str(output.resolve()),
            "video_stream": str(stream_path.resolve()),
            "video_stream_sha256": str(stream["output_sha256"]),
            "audio": str(audio_path.resolve()),
            "audio_container_sha256": audio_sha256,
            "audio_pcm_sha256": audio_pcm_sha256,
            "audio_modified": False,
            **mux,
        }
        write_self_hashed_json(sidecar_path, sidecar_body)
    frames = decode_video_frames(stream_path)
    verification = verify_muxed_media(output, frames, expected_pcm, list(range(len(frames))))
    timeline = media_video_timeline(output)
    if int(timeline["frame_count"]) != len(frames):
        raise DiagnosticError(f"muxed cell frame count changed: {sample_id}/{video_arm}/{audio_arm}")
    return {
        "video_arm": video_arm,
        "audio_arm": audio_arm,
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "sidecar": str(sidecar_path.resolve()),
        "video_stream": str(stream_path.resolve()),
        "video_stream_sha256": str(stream["output_sha256"]),
        "audio": str(audio_path.resolve()),
        "audio_container_sha256": audio_sha256,
        "audio_pcm_sha256": audio_pcm_sha256,
        "frame_count": len(frames),
        "timeline": timeline,
        "verification": verification,
    }


def materialize(paths: config.RunPaths, protocol: Mapping[str, Any]) -> dict[str, Any]:
    if protocol.get("status") != "locked":
        raise DiagnosticError("protocol is not locked for media materialization")
    protocol_sha = str(protocol.get("_sha256", ""))
    if not protocol_sha:
        raise DiagnosticError("protocol hash is unavailable")
    paths.media.mkdir(parents=True, exist_ok=True)
    records = _record_index(protocol)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(protocol["records"], 1):
        sample_id = str(record["sample_id"])
        streams = {arm: _materialize_video_stream(paths, protocol_sha, record, arm) for arm in config.VIDEO_ARMS}
        cells: dict[str, Any] = {}
        for video_arm, audio_arm in config.MAIN_CELL_SPECS:
            key = config.cell_key(video_arm, audio_arm)
            cells[key] = _materialize_cell(paths, protocol_sha, record, streams[video_arm], audio_arm)
        rows.append({"sample_id": sample_id, "source_group": str(record["source_group"]), "streams": streams, "cells": cells})
        print(f"MEDIA {index}/{len(records)} {sample_id}", flush=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage_id": "media",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "protocol_sha256": protocol_sha,
        "status": "complete",
        "record_count": len(rows),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "video_stream_count": len(rows) * len(config.VIDEO_ARMS),
        "expected_video_stream_count": config.EXPECTED_RECORD_COUNT * len(config.VIDEO_ARMS),
        "cell_count": sum(len(row["cells"]) for row in rows),
        "expected_cell_count": config.EXPECTED_MAIN_CELL_COUNT,
        "audio_modified": False,
        "rows": rows,
    }
    write_self_hashed_json(paths.media / "manifest.json", manifest)
    return verify_self_hashed_json(paths.media / "manifest.json")
