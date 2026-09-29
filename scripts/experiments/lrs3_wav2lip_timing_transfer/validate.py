from __future__ import annotations

import argparse
import json
import math
import re
import wave
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.lrs3_real_video_local_timing.media import (
    crop_frames_from_track,
    decode_pcm16,
    decode_video_frames,
    source_pcm16,
    verify_muxed_media,
)
from scripts.experiments.lrs3_real_video_local_timing.protocol import video_timeline

from . import config
from .common import (
    DiagnosticError,
    bytes_sha256,
    canonical_json_sha256,
    file_sha256,
    read_json,
    sample_ids_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .protocol import media_video_timeline

SYNCNET_MODEL = config.REPO / "third_party/syncnet_python/data/syncnet_v2.model"


def _artifact(path: Path) -> dict[str, Any]:
    return verify_self_hashed_json(path)


def _resolve(value: Any) -> Path:
    path = Path(str(value))
    return (config.REPO / path).resolve() if not path.is_absolute() else path.resolve()


def _assert_file(path: Path, expected: str | None, label: str) -> None:
    if not path.is_file():
        raise DiagnosticError(f"{label} is missing: {path}")
    if expected is not None and file_sha256(path) != expected:
        raise DiagnosticError(f"{label} hash changed: {path}")


def _assert_close(left: Any, right: Any, tolerance: float = 1e-12, label: str = "value") -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise DiagnosticError(f"{label} keys differ")
        for key in left:
            _assert_close(left[key], right[key], tolerance, f"{label}.{key}")
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise DiagnosticError(f"{label} length differs")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _assert_close(a, b, tolerance, f"{label}[{index}]")
        return
    if isinstance(left, (float, int)) and isinstance(right, (float, int)) and not isinstance(left, bool) and not isinstance(right, bool):
        if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance):
            raise DiagnosticError(f"{label} differs: {left} != {right}")
        return
    if left != right:
        raise DiagnosticError(f"{label} differs: {left} != {right}")


def _wav(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            count = handle.getnframes()
            pcm = handle.readframes(count)
    except (OSError, wave.Error) as exc:
        raise DiagnosticError(f"invalid WAV: {path}") from exc
    if (channels, width, rate) != (1, 2, config.SAMPLE_RATE) or len(pcm) != count * 2:
        raise DiagnosticError(f"WAV format is not 16 kHz mono PCM16: {path}")
    return {"sample_count": count, "sample_rate": rate, "channels": channels, "sample_width": width, "container_sha256": file_sha256(path), "decoded_pcm_sha256": bytes_sha256(pcm)}, pcm


def _forward_map(sample_count: int) -> np.ndarray:
    if sample_count < 2 or sample_count - 1 <= 2.0 * math.pi * config.WARP_AMPLITUDE_SAMPLES:
        raise DiagnosticError("audio map length is invalid")
    n = np.arange(sample_count, dtype=np.float64)
    mapped = n + config.WARP_AMPLITUDE_SAMPLES * np.sin(2.0 * np.pi * n / float(sample_count - 1))
    mapped[0] = 0.0
    mapped[-1] = float(sample_count - 1)
    if not np.isfinite(mapped).all() or np.any(np.diff(mapped) <= 0.0) or mapped[0] != 0.0 or mapped[-1] != sample_count - 1:
        raise DiagnosticError("audio map is not strictly increasing")
    return mapped


def _reconstruct_warp(natural_pcm: bytes) -> tuple[bytes, np.ndarray]:
    samples = np.frombuffer(natural_pcm, dtype="<i2")
    mapped = _forward_map(int(samples.size))
    source = np.arange(samples.size, dtype=np.float64)
    warped = np.interp(mapped, source, samples.astype(np.float64))
    return np.rint(warped).astype("<i2", copy=False).tobytes(), mapped


def _independent_masks(sample_count: int, frame_counts: Mapping[str, int]) -> dict[str, Any]:
    mapped = _forward_map(sample_count)
    n = np.arange(sample_count, dtype=np.float64)
    q = min(int(frame_counts[config.VIDEO_R]), int(frame_counts[config.VIDEO_GN]), int(frame_counts[config.VIDEO_GW]), sample_count // config.SAMPLES_PER_FRAME)
    candidate = list(range(config.VSHIFT, q - 20))
    plus: list[int] = []
    minus: list[int] = []
    d_by_row: dict[str, float] = {}
    a_by_row: dict[str, float] = {}
    for row in candidate:
        center = float(config.SAMPLES_PER_FRAME * (row + 2))
        d = (float(np.interp(center, n, mapped)) - center) / config.SAMPLES_PER_FRAME
        a = (center - float(np.interp(center, mapped, n))) / config.SAMPLES_PER_FRAME
        d_by_row[str(row)] = d
        a_by_row[str(row)] = a
        if d >= 2.5 and a >= 2.5:
            plus.append(row)
        if d <= -2.5 and a <= -2.5:
            minus.append(row)
    if len(plus) < config.MIN_LOCAL_ROWS or len(minus) < config.MIN_LOCAL_ROWS:
        raise DiagnosticError("independent mask has too few rows")
    return {
        "sample_count": sample_count,
        "q_frames": q,
        "candidate_rows": candidate,
        "common_window_rows": list(range(q - config.WINDOW_FRAMES)),
        "plus_rows": plus,
        "minus_rows": minus,
        "d_by_row": d_by_row,
        "a_by_row": a_by_row,
        "plus_expected_offset": float(np.mean([a_by_row[str(row)] for row in plus])),
        "minus_expected_offset": float(np.mean([a_by_row[str(row)] for row in minus])),
        "plus_expected_video_response": float(-np.mean([d_by_row[str(row)] for row in plus])),
        "minus_expected_video_response": float(-np.mean([d_by_row[str(row)] for row in minus])),
        "forward_mapping_sha256": bytes_sha256(np.asarray(mapped, dtype="<f8").tobytes()),
        "forward_mapping_dtype": "float64-little-endian",
        "forward_mapping_length": sample_count,
    }


def _parse_log(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = {
        "sync_c": re.findall(r"Confidence:\s+([-+]?\d+(?:\.\d+)?)", text),
        "sync_d": re.findall(r"Min dist:\s+([-+]?\d+(?:\.\d+)?)", text),
        "av_offset": re.findall(r"AV offset:\s+(-?\d+)", text),
    }
    if any(len(items) != 1 for items in matches.values()):
        raise DiagnosticError(f"SyncNet log is missing or ambiguous: {path}")
    return {"sync_c": float(matches["sync_c"][0]), "sync_d": float(matches["sync_d"][0]), "av_offset": int(matches["av_offset"][0])}


def _global(matrix: np.ndarray) -> dict[str, Any]:
    if matrix.ndim != 2 or matrix.shape[1] != 31 or matrix.shape[0] < 1 or not np.isfinite(matrix).all():
        raise DiagnosticError("distance matrix is malformed")
    curve = np.mean(matrix, axis=0)
    index = int(np.argmin(curve))
    minimum = float(curve[index])
    return {"curve": [float(value) for value in curve], "offsets": [config.VSHIFT - i for i in range(31)], "min_index": index, "offset": config.VSHIFT - index, "sync_d": minimum, "sync_c": float(np.median(curve) - minimum)}


def _peak(curve: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float64)
    ordered = np.sort(values)
    index = int(np.argmin(values))
    offset = config.VSHIFT - index
    minimum = float(ordered[0])
    second = float(ordered[1])
    return {"min_index": index, "offset": offset, "min": minimum, "second_min": second, "peak_gap": float(second - minimum), "clear": bool(second - minimum > config.PEAK_GAP_THRESHOLD and -config.VSHIFT < offset < config.VSHIFT)}


def _local(matrix: np.ndarray, masks: Mapping[str, Any], name: str) -> dict[str, Any]:
    rows = [int(value) for value in masks[f"{name.lower()}_rows"]]
    if len(rows) < config.MIN_LOCAL_ROWS or max(rows, default=-1) >= matrix.shape[0]:
        raise DiagnosticError(f"{name} mask is outside matrix support")
    curve = np.mean(matrix[rows, :], axis=0)
    return {"rows": rows, "curve": [float(value) for value in curve], **_peak(curve)}


def _common(matrix: np.ndarray, masks: Mapping[str, Any]) -> dict[str, Any]:
    rows = [int(value) for value in masks["common_window_rows"]]
    if not rows or max(rows) >= matrix.shape[0]:
        raise DiagnosticError("common mask is outside matrix support")
    return _global(matrix[rows, :])


def _validate_spec_bindings(bindings: Mapping[str, Any]) -> None:
    expected = config.spec_bindings()
    if set(bindings) != set(expected):
        raise DiagnosticError("spec binding set differs")
    for name, item in expected.items():
        actual = bindings.get(name)
        if actual != item:
            raise DiagnosticError(f"spec binding differs: {name}")
        _assert_file(Path(item["path"]), item["sha256"], f"spec binding {name}")


def _validate_parent_runs(parent_runs: Mapping[str, Any]) -> None:
    expected = {
        "calibration_final": (config.CALIBRATION_FINAL, config.CALIBRATION_FINAL_SHA256),
        "tail_final": (config.TAIL_FINAL, config.TAIL_FINAL_SHA256),
        "tail_protocol": (config.TAIL_PROTOCOL, None),
    }
    for name, (path, expected_hash) in expected.items():
        item = parent_runs.get(name)
        if not isinstance(item, Mapping) or _resolve(item.get("path")) != path.resolve():
            raise DiagnosticError(f"parent binding path differs: {name}")
        if expected_hash is not None and item.get("sha256") != expected_hash:
            raise DiagnosticError(f"parent binding hash differs: {name}")
        _assert_file(path, expected_hash, f"parent binding {name}")
    tail_protocol = _artifact(config.TAIL_PROTOCOL)
    if parent_runs["tail_protocol"].get("sha256") != file_sha256(config.TAIL_PROTOCOL) or tail_protocol.get("status") != "locked":
        raise DiagnosticError("tail protocol binding is invalid")


def _validate_input_audit(paths: config.RunPaths) -> dict[str, Any]:
    audit = _artifact(paths.input_audit)
    if audit.get("protocol_id") != config.PROTOCOL_ID or audit.get("status") != "complete" or audit.get("record_count") != config.EXPECTED_RECORD_COUNT or audit.get("passed_count") != config.EXPECTED_RECORD_COUNT or len(audit.get("records", [])) != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("input audit is incomplete")
    if audit.get("ordered_sample_id_sha256") != config.ORDERED_SAMPLE_ID_SHA256:
        raise DiagnosticError("input audit sample ID binding differs")
    frozen = audit.get("frozen_inputs")
    expected_inputs = {
        "cohort": (config.COHORT, config.COHORT_SHA256),
        "calibration_final": (config.CALIBRATION_FINAL, config.CALIBRATION_FINAL_SHA256),
        "audio_manifest": (config.AUDIO_MANIFEST, config.AUDIO_MANIFEST_SHA256),
        "videos_manifest": (config.VIDEOS_MANIFEST, config.VIDEOS_MANIFEST_SHA256),
        "tail_final": (config.TAIL_FINAL, config.TAIL_FINAL_SHA256),
    }
    if not isinstance(frozen, Mapping):
        raise DiagnosticError("input audit frozen inputs are missing")
    for name, (path, expected_hash) in expected_inputs.items():
        item = frozen.get(name)
        if not isinstance(item, Mapping) or _resolve(item.get("path")) != path.resolve() or item.get("sha256") != expected_hash:
            raise DiagnosticError(f"input audit binding differs: {name}")
        _assert_file(path, expected_hash, f"input audit input {name}")
    tail_item = frozen.get("tail_protocol")
    if not isinstance(tail_item, Mapping) or _resolve(tail_item.get("path")) != config.TAIL_PROTOCOL.resolve() or tail_item.get("sha256") != file_sha256(config.TAIL_PROTOCOL):
        raise DiagnosticError("input audit tail protocol binding differs")
    for row in audit["records"]:
        if not row.get("passed") or row.get("status") != "PASS" or row.get("errors"):
            raise DiagnosticError(f"input audit record is not clean: {row.get('sample_id')}")
    return audit


def _validate_protocol(paths: config.RunPaths) -> dict[str, Any]:
    protocol = _artifact(paths.protocol)
    if protocol.get("protocol_id") != config.PROTOCOL_ID or protocol.get("protocol_revision") != config.PROTOCOL_REVISION or protocol.get("status") != "locked":
        raise DiagnosticError("protocol marker is invalid")
    records = protocol.get("records")
    if not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("protocol record count is invalid")
    ids = [str(record.get("sample_id")) for record in records]
    groups = [str(record.get("source_group")) for record in records]
    if len(set(ids)) != config.EXPECTED_RECORD_COUNT or len(set(groups)) != config.EXPECTED_SOURCE_GROUP_COUNT or sample_ids_sha256(ids) != config.ORDERED_SAMPLE_ID_SHA256:
        raise DiagnosticError("protocol order or identity differs")
    _validate_spec_bindings(protocol.get("spec_bindings", {}))
    _validate_parent_runs(protocol.get("parent_runs", {}))
    package = Path(__file__).resolve().parent
    current_source_hashes = {path.name: file_sha256(path) for path in sorted(package.glob("*.py"))}
    if protocol.get("source_hashes") != current_source_hashes:
        raise DiagnosticError("implementation source hashes changed")
    input_binding = protocol.get("input_audit")
    if not isinstance(input_binding, Mapping) or _resolve(input_binding.get("path")) != paths.input_audit.resolve():
        raise DiagnosticError("protocol input audit path differs")
    audit = _validate_input_audit(paths)
    if input_binding.get("sha256") != file_sha256(paths.input_audit) or input_binding.get("artifact_sha256") != audit.get("artifact_sha256"):
        raise DiagnosticError("protocol input audit hash differs")
    for record in records:
        sample_id = str(record["sample_id"])
        natural = record.get("natural_audio")
        warp = record.get("warp_audio")
        real = record.get("real_video")
        generated = record.get("generated_videos")
        if not all(isinstance(item, Mapping) for item in (natural, warp, real, generated)):
            raise DiagnosticError(f"protocol source binding is malformed: {sample_id}")
        n_path = _resolve(natural["path"])
        w_path = _resolve(warp["path"])
        r_path = _resolve(real["path"])
        _assert_file(n_path, natural["cohort_sha256"], f"natural audio {sample_id}")
        _assert_file(w_path, warp["container_sha256"], f"warp audio {sample_id}")
        _assert_file(r_path, real["sha256"], f"real video {sample_id}")
        n_meta, n_pcm = _wav(n_path)
        w_meta, w_pcm = _wav(w_path)
        if n_meta["sample_count"] != int(natural["sample_count"]) or n_meta["decoded_pcm_sha256"] != natural["pcm_sha256"] or w_meta["decoded_pcm_sha256"] != warp["decoded_pcm_sha256"]:
            raise DiagnosticError(f"protocol audio evidence differs: {sample_id}")
        reconstructed, mapped = _reconstruct_warp(n_pcm)
        if reconstructed != w_pcm or bytes_sha256(np.asarray(mapped, dtype="<f8").tobytes()) != record["audio_map"]["forward_mapping_sha256"]:
            raise DiagnosticError(f"warp reconstruction or map hash differs: {sample_id}")
        real_timeline = video_timeline(r_path)
        if real_timeline["frame_count"] != int(real["timeline"]["frame_count"]):
            raise DiagnosticError(f"real video timeline differs: {sample_id}")
        frame_counts: dict[str, int] = {config.VIDEO_R: real_timeline["frame_count"]}
        for arm in (config.VIDEO_GN, config.VIDEO_GW):
            item = generated.get(arm)
            if not isinstance(item, Mapping):
                raise DiagnosticError(f"generated video binding is missing: {sample_id}/{arm}")
            path = _resolve(item["path"])
            _assert_file(path, item["sha256"], f"generated video {sample_id}/{arm}")
            timeline = video_timeline(path)
            frame_counts[arm] = timeline["frame_count"]
            if timeline["frame_count"] != int(item["timeline"]["frame_count"]):
                raise DiagnosticError(f"generated video timeline differs: {sample_id}/{arm}")
            if timeline["width"] != real_timeline["width"] or timeline["height"] != real_timeline["height"]:
                raise DiagnosticError(f"generated canvas differs: {sample_id}/{arm}")
        if frame_counts[config.VIDEO_GN] != frame_counts[config.VIDEO_GW] or frame_counts[config.VIDEO_GN] > frame_counts[config.VIDEO_R]:
            raise DiagnosticError(f"generated videos are not equal prefixes: {sample_id}")
        track = record.get("processed_track")
        if not isinstance(track, list) or len(track) != frame_counts[config.VIDEO_R] or record.get("track_sha256") != canonical_json_sha256(track):
            raise DiagnosticError(f"processed track binding differs: {sample_id}")
        expected_masks = _independent_masks(n_meta["sample_count"], frame_counts)
        masks = record.get("masks")
        if not isinstance(masks, Mapping):
            raise DiagnosticError(f"protocol masks are missing: {sample_id}")
        for key in ("sample_count", "q_frames", "candidate_rows", "common_window_rows", "plus_rows", "minus_rows", "forward_mapping_sha256", "forward_mapping_dtype", "forward_mapping_length"):
            _assert_close(masks.get(key), expected_masks[key], 1e-12, f"protocol masks {sample_id}.{key}")
        for key in ("d_by_row", "a_by_row", "plus_expected_offset", "minus_expected_offset", "plus_expected_video_response", "minus_expected_video_response"):
            _assert_close(masks.get(key), expected_masks[key], 1e-12, f"protocol masks {sample_id}.{key}")
        if record["masks"].get("frame_counts") != frame_counts:
            raise DiagnosticError(f"protocol frame-count mask binding differs: {sample_id}")
    return protocol


def _validate_media(paths: config.RunPaths, protocol: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _artifact(paths.media / "manifest.json")
    if manifest.get("protocol_id") != config.PROTOCOL_ID or manifest.get("protocol_sha256") != file_sha256(paths.protocol) or manifest.get("status") != "complete" or manifest.get("record_count") != config.EXPECTED_RECORD_COUNT or manifest.get("video_stream_count") != config.EXPECTED_RECORD_COUNT * 3 or manifest.get("cell_count") != config.EXPECTED_MAIN_CELL_COUNT:
        raise DiagnosticError("media manifest is incomplete")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("media record rows are incomplete")
    records = {str(record["sample_id"]): record for record in protocol["records"]}
    seen: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id"))
        if sample_id in seen or sample_id not in records:
            raise DiagnosticError(f"media sample identity is invalid: {sample_id}")
        seen.add(sample_id)
        record = records[sample_id]
        tracks = record["processed_track"]
        expected_frames_by_arm: dict[str, list[np.ndarray]] = {}
        for video_arm in config.VIDEO_ARMS:
            stream = row.get("streams", {}).get(video_arm)
            if not isinstance(stream, Mapping):
                raise DiagnosticError(f"media stream is missing: {sample_id}/{video_arm}")
            stream_path = _resolve(stream["output"])
            sidecar_path = _resolve(stream["sidecar"])
            sidecar = _artifact(sidecar_path)
            source = record["real_video"] if video_arm == config.VIDEO_R else record["generated_videos"][video_arm]
            source_path = _resolve(source["path"])
            if stream.get("output_sha256") != file_sha256(stream_path):
                raise DiagnosticError(f"media stream manifest hash differs: {sample_id}/{video_arm}")
            source_frames = decode_video_frames(source_path)
            count = len(source_frames) if video_arm == config.VIDEO_R else int(source["timeline"]["frame_count"])
            track_prefix = [dict(item) for item in tracks[:count]]
            cropped = crop_frames_from_track(source_frames[:count], track_prefix)
            if len(cropped) != int(stream["frame_count"]):
                raise DiagnosticError(f"media stream frame count differs: {sample_id}/{video_arm}")
            actual = decode_video_frames(stream_path)
            if len(actual) != len(cropped) or any(not np.array_equal(left, right) for left, right in zip(actual, cropped, strict=True)):
                raise DiagnosticError(f"media stream pixels differ: {sample_id}/{video_arm}")
            if sidecar.get("protocol_sha256") != file_sha256(paths.protocol) or sidecar.get("sample_id") != sample_id or sidecar.get("video_arm") != video_arm or sidecar.get("source_video_sha256") != source["sha256"] or sidecar.get("track_sha256") != record["track_sha256"] or sidecar.get("output_sha256") != file_sha256(stream_path):
                raise DiagnosticError(f"media stream sidecar binding differs: {sample_id}/{video_arm}")
            timeline = media_video_timeline(stream_path)
            if timeline["frame_count"] != len(cropped) or timeline["width"] != config.CROP_SIZE or timeline["height"] != config.CROP_SIZE:
                raise DiagnosticError(f"media stream timeline differs: {sample_id}/{video_arm}")
            expected_frames_by_arm[video_arm] = actual
        cells = row.get("cells")
        if not isinstance(cells, Mapping) or set(cells) != {config.cell_key(video, audio) for video, audio in config.MAIN_CELL_SPECS}:
            raise DiagnosticError(f"media cell set differs: {sample_id}")
        for video_arm, audio_arm in config.MAIN_CELL_SPECS:
            key = config.cell_key(video_arm, audio_arm)
            cell = cells[key]
            output = _resolve(cell["output"])
            sidecar = _artifact(_resolve(cell["sidecar"]))
            if cell.get("output_sha256") != file_sha256(output):
                raise DiagnosticError(f"media cell manifest hash differs: {sample_id}/{key}")
            audio_path = _resolve(record["natural_audio"]["path"] if audio_arm == config.AUDIO_N else record["warp_audio"]["path"])
            expected_pcm = source_pcm16(audio_path)
            if sidecar.get("protocol_sha256") != file_sha256(paths.protocol) or sidecar.get("sample_id") != sample_id or sidecar.get("video_arm") != video_arm or sidecar.get("audio_arm") != audio_arm or sidecar.get("video_stream_sha256") != cell["video_stream_sha256"] or sidecar.get("audio_container_sha256") != file_sha256(audio_path) or sidecar.get("audio_pcm_sha256") != bytes_sha256(expected_pcm) or sidecar.get("output_sha256") != file_sha256(output):
                raise DiagnosticError(f"media cell sidecar binding differs: {sample_id}/{key}")
            verification = verify_muxed_media(output, expected_frames_by_arm[video_arm], expected_pcm, list(range(len(expected_frames_by_arm[video_arm]))))
            if verification["sha256"] != file_sha256(output) or cell.get("output_sha256") != file_sha256(output):
                raise DiagnosticError(f"media cell verification differs: {sample_id}/{key}")
            if cell.get("frame_count") != len(expected_frames_by_arm[video_arm]):
                raise DiagnosticError(f"media cell frame count differs: {sample_id}/{key}")
    if len(seen) != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("media sample count differs")
    return {"stage": "media", "status": "valid", "record_count": len(seen), "cell_count": manifest["cell_count"]}


def _expected_score_keys() -> set[tuple[str, str, str, bool]]:
    keys: set[tuple[str, str, str, bool]] = set()
    for sample_id in []:
        keys.add((sample_id, "", "", False))
    return keys


def _validate_score_row(row: Mapping[str, Any], record: Mapping[str, Any], paths: config.RunPaths) -> None:
    cell_id = str(row.get("cell_id"))
    matrix_path = _resolve(row["matrix"])
    worker_path = _resolve(row["worker_result"])
    log_path = _resolve(row["score_log"])
    media_path = _resolve(row["media"])
    audio_path = _resolve(row["audio"])
    for path, label in ((matrix_path, "score matrix"), (worker_path, "worker result"), (log_path, "score log"), (media_path, "score media"), (audio_path, "score audio")):
        _assert_file(path, None, f"{label} {cell_id}")
    expected_media = paths.media / "cells" / f"{row['sample_id']}__{row['video_arm']}__{row['audio_arm']}.mkv"
    if media_path != expected_media.resolve():
        raise DiagnosticError(f"score media path is swapped: {cell_id}")
    if row.get("matrix_sha256") != file_sha256(matrix_path) or row.get("worker_result_sha256") != file_sha256(worker_path) or row.get("score_log_sha256") != file_sha256(log_path) or row.get("media_sha256") != file_sha256(media_path) or row.get("audio_container_sha256") != file_sha256(audio_path):
        raise DiagnosticError(f"score file hash binding differs: {cell_id}")
    expected_pcm = source_pcm16(audio_path)
    if decode_pcm16(media_path) != expected_pcm or row.get("audio_pcm_sha256") != bytes_sha256(expected_pcm):
        raise DiagnosticError(f"score media audio differs: {cell_id}")
    matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float64)
    video_arm = str(row["video_arm"])
    expected_rows = min(int(record["masks"]["frame_counts"][video_arm]), int(record["natural_audio"]["sample_count"]) // config.SAMPLES_PER_FRAME) - config.WINDOW_FRAMES
    if matrix.shape != (expected_rows, 31) or row.get("matrix_shape") != [expected_rows, 31] or row.get("window_frame_starts") != list(range(expected_rows)) or row.get("column_offsets") != [config.VSHIFT - index for index in range(31)]:
        raise DiagnosticError(f"score matrix support binding differs: {cell_id}")
    worker = read_json(worker_path)
    parsed = _parse_log(log_path)
    global_score = _global(matrix)
    if abs(float(parsed["sync_c"]) - global_score["sync_c"]) > 0.001 or abs(float(parsed["sync_d"]) - global_score["sync_d"]) > 0.001 or int(parsed["av_offset"]) != int(global_score["offset"]):
        raise DiagnosticError(f"official score reconstruction differs: {cell_id}")
    if worker.get("matrix_sha256") != file_sha256(matrix_path) or worker.get("media_sha256") != file_sha256(media_path) or worker.get("input_pcm_sha256") != bytes_sha256(expected_pcm) or worker.get("extracted_pcm_sha256") != bytes_sha256(expected_pcm) or int(worker.get("offset", 10000)) != int(parsed["av_offset"]):
        raise DiagnosticError(f"worker binding differs: {cell_id}")
    masks = record["masks"]
    expected_local = {"PLUS": _local(matrix, masks, "PLUS"), "MINUS": _local(matrix, masks, "MINUS")}
    expected_common = _common(matrix, masks)
    _assert_close(row.get("official_global"), global_score, 1e-12, f"official global {cell_id}")
    _assert_close(row.get("common_global"), expected_common, 1e-12, f"common global {cell_id}")
    _assert_close(row.get("local"), expected_local, 1e-12, f"local evidence {cell_id}")
    if row.get("protocol_sha256") != file_sha256(paths.protocol) or row.get("sample_id") != record["sample_id"] or row.get("source_group") != record["source_group"] or row.get("runtime", {}).get("syncnet_model_sha256") != file_sha256(SYNCNET_MODEL):
        raise DiagnosticError(f"score protocol binding differs: {cell_id}")
    sidecar_path = matrix_path.parent / "score.json"
    sidecar = _artifact(sidecar_path)
    if dict(sidecar) != dict(row):
        raise DiagnosticError(f"score manifest row differs from its sidecar: {cell_id}")


def _validate_scores(paths: config.RunPaths, protocol: Mapping[str, Any]) -> tuple[dict[str, Any], dict[tuple[str, str, str, bool], Mapping[str, Any]]]:
    manifest = _artifact(paths.scores / "manifest.json")
    if manifest.get("protocol_id") != config.PROTOCOL_ID or manifest.get("protocol_sha256") != file_sha256(paths.protocol) or manifest.get("media_manifest_sha256") != file_sha256(paths.media / "manifest.json") or manifest.get("status") != "complete" or manifest.get("cell_count") != config.EXPECTED_TOTAL_CELL_COUNT or manifest.get("main_cell_count") != config.EXPECTED_MAIN_CELL_COUNT or manifest.get("repeat_cell_count") != config.EXPECTED_REPEAT_CELL_COUNT:
        raise DiagnosticError("score manifest is incomplete")
    rows = manifest.get("scores")
    if not isinstance(rows, list) or len(rows) != config.EXPECTED_TOTAL_CELL_COUNT:
        raise DiagnosticError("score rows are incomplete")
    records = {str(record["sample_id"]): record for record in protocol["records"]}
    index: dict[tuple[str, str, str, bool], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise DiagnosticError("score row is malformed")
        key = (str(row.get("sample_id")), str(row.get("video_arm")), str(row.get("audio_arm")), bool(row.get("repeat", False)))
        if key in index or key[0] not in records or key[1] not in config.VIDEO_ARMS or key[2] not in config.AUDIO_ARMS:
            raise DiagnosticError(f"score identity is invalid: {key}")
        expected_id = f"{key[0]}__{key[1]}__{key[2]}" + ("__repeat" if key[3] else "")
        if row.get("cell_id") != expected_id:
            raise DiagnosticError(f"score cell ID differs: {key}")
        if bool(row.get("repeat")) and row.get("repeat_of") != f"{key[0]}__{key[1]}__{key[2]}":
            raise DiagnosticError(f"repeat binding differs: {key}")
        _validate_score_row(row, records[key[0]], paths)
        index[key] = row
    expected: set[tuple[str, str, str, bool]] = set()
    for sample_id in records:
        expected.update((sample_id, video, audio, False) for video, audio in config.MAIN_CELL_SPECS)
        expected.update((sample_id, video, audio, True) for video, audio in config.REPEAT_CELL_SPECS)
    if set(index) != expected:
        raise DiagnosticError("score cell set differs")
    for sample_id in records:
        for video, audio in config.REPEAT_CELL_SPECS:
            main = index[(sample_id, video, audio, False)]
            repeat = index[(sample_id, video, audio, True)]
            if main["matrix"] == repeat["matrix"] or main["score_log"] == repeat["score_log"] or main["worker_result"] == repeat["worker_result"] or main["worker_tmp_dir"] == repeat["worker_tmp_dir"]:
                raise DiagnosticError(f"repeat execution artifacts were copied: {sample_id}/{video}/{audio}")
    return manifest, index


def _baseline(main: Mapping[str, Any], repeat: Mapping[str, Any]) -> dict[str, Any]:
    left = np.asarray(np.load(_resolve(main["matrix"]), allow_pickle=False), dtype=np.float64)
    right = np.asarray(np.load(_resolve(repeat["matrix"]), allow_pickle=False), dtype=np.float64)
    same_shape = left.shape == right.shape
    max_abs = float(np.max(np.abs(left - right))) if same_shape else None
    offsets = {
        "global": int(main["official_global"]["offset"]) == int(repeat["official_global"]["offset"]),
        "plus": int(main["local"]["PLUS"]["offset"]) == int(repeat["local"]["PLUS"]["offset"]),
        "minus": int(main["local"]["MINUS"]["offset"]) == int(repeat["local"]["MINUS"]["offset"]),
    }
    return {"matrix_shape_equal": same_shape, "matrix_max_abs": max_abs, "offsets_equal": offsets, "passes": bool(same_shape and max_abs is not None and max_abs <= 0.001 and all(offsets.values()))}


def _base_clear(row: Mapping[str, Any]) -> bool:
    local = row["local"]
    return bool(local["PLUS"]["clear"] and local["MINUS"]["clear"] and abs(int(local["PLUS"]["offset"]) - int(local["MINUS"]["offset"])) <= config.OFFSET_TOLERANCE_FRAMES)


def _pair_check(left: Mapping[str, Any], right: Mapping[str, Any], masks: Mapping[str, Any], expected_plus: float, expected_minus: float) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, target in (("PLUS", expected_plus), ("MINUS", expected_minus)):
        actual = float(left["local"][name]["offset"]) - float(right["local"][name]["offset"])
        error = actual - float(target)
        result[name] = {"actual": actual, "expected": float(target), "error": error, "passes": bool(left["local"][name]["clear"] and right["local"][name]["clear"] and abs(error) <= config.OFFSET_TOLERANCE_FRAMES)}
    result["passes"] = bool(result["PLUS"]["passes"] and result["MINUS"]["passes"])
    return result


def _independent_analysis(protocol: Mapping[str, Any], scores: Mapping[tuple[str, str, str, bool], Mapping[str, Any]]) -> dict[str, Any]:
    repeatability = 0
    r_baseline = 0
    gn_baseline = 0
    counts = {"A": 0, "B": 0, "C": 0, "O": 0}
    per_record: list[dict[str, Any]] = []
    groups: list[str] = []
    descriptive: dict[str, list[float]] = {key: [] for key in ("self_C_GW_W_minus_GN_N", "self_D_GN_N_minus_GW_W", "replacement_C_GW_W_minus_GW_N", "replacement_D_GW_N_minus_GW_W")}
    for record in protocol["records"]:
        sample_id = str(record["sample_id"])
        groups.append(str(record["source_group"]))
        def get(video: str, audio: str, repeat: bool = False, _sample_id: str = sample_id) -> Mapping[str, Any]:
            return scores[(_sample_id, video, audio, repeat)]
        rn, rw = get(config.VIDEO_R, config.AUDIO_N), get(config.VIDEO_R, config.AUDIO_W)
        gnn, gnw = get(config.VIDEO_GN, config.AUDIO_N), get(config.VIDEO_GN, config.AUDIO_W)
        gwn, gww = get(config.VIDEO_GW, config.AUDIO_N), get(config.VIDEO_GW, config.AUDIO_W)
        rn_rep, gnn_rep = get(config.VIDEO_R, config.AUDIO_N, True), get(config.VIDEO_GN, config.AUDIO_N, True)
        r_repeat = _baseline(rn, rn_rep)
        gn_repeat = _baseline(gnn, gnn_rep)
        repeat_pass = bool(r_repeat["passes"] and gn_repeat["passes"])
        repeatability += int(repeat_pass)
        r_clear = _base_clear(rn)
        gn_clear = _base_clear(gnn)
        r_baseline += int(r_clear)
        gn_baseline += int(gn_clear)
        masks = record["masks"]
        a = _pair_check(rw, rn, masks, masks["plus_expected_offset"], masks["minus_expected_offset"])
        b = _pair_check(gnw, gnn, masks, masks["plus_expected_offset"], masks["minus_expected_offset"])
        c = _pair_check(gwn, gnn, masks, masks["plus_expected_video_response"], masks["minus_expected_video_response"])
        o = _pair_check(gww, gnn, masks, 0.0, 0.0)
        a["passes"] = bool(a["passes"] and r_clear)
        b["passes"] = bool(b["passes"] and gn_clear)
        c["passes"] = bool(c["passes"] and gn_clear)
        o["passes"] = bool(o["passes"] and gn_clear)
        checks = {"A": a, "B": b, "C": c, "O": o}
        for name, value in checks.items():
            counts[name] += int(value["passes"])
        descriptive["self_C_GW_W_minus_GN_N"].append(float(gww["common_global"]["sync_c"]) - float(gnn["common_global"]["sync_c"]))
        descriptive["self_D_GN_N_minus_GW_W"].append(float(gnn["common_global"]["sync_d"]) - float(gww["common_global"]["sync_d"]))
        descriptive["replacement_C_GW_W_minus_GW_N"].append(float(gww["common_global"]["sync_c"]) - float(gwn["common_global"]["sync_c"]))
        descriptive["replacement_D_GW_N_minus_GW_W"].append(float(gwn["common_global"]["sync_d"]) - float(gww["common_global"]["sync_d"]))
        per_record.append({"sample_id": sample_id, "source_group": str(record["source_group"]), "repeatability": {"R_N": r_repeat, "G_N_N": gn_repeat, "passes": repeat_pass}, "baseline": {"R_N": r_clear, "G_N_N": gn_clear}, "checks": checks})
    baseline_pass = r_baseline >= config.MIN_BASELINE_RECORDS and gn_baseline >= config.MIN_BASELINE_RECORDS
    if repeatability != config.EXPECTED_RECORD_COUNT:
        decision = "REPEATABILITY_FAILED"
    elif not baseline_pass:
        decision = "BASELINE_INCONCLUSIVE"
    elif counts["A"] < config.MIN_SUCCESS_RECORDS:
        decision = "AUDIO_CONTROL_UNRESOLVED"
    elif counts["B"] < config.MIN_SUCCESS_RECORDS:
        decision = "GENERATED_ENDPOINT_UNRESOLVED"
    elif counts["C"] < config.MIN_SUCCESS_RECORDS or counts["O"] < config.MIN_SUCCESS_RECORDS:
        decision = "GENERATED_RESPONSE_UNRESOLVED"
    else:
        decision = "LOCAL_RESPONSE_ESTABLISHED"
    return {"scientific_decision": decision, "repeatability": repeatability, "baseline": {"R_N_count": r_baseline, "G_N_N_count": gn_baseline}, "checks": counts, "per_record": per_record, "descriptive_values": descriptive, "groups": groups}


def _bootstrap(values: Sequence[float], groups: Sequence[str]) -> dict[str, Any]:
    by_group: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values, strict=True):
        by_group[str(group)].append(float(value))
    labels = sorted(by_group)
    means = {label: float(np.mean(items)) for label, items in by_group.items()}
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    estimates = np.empty(config.BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(config.BOOTSTRAP_DRAWS):
        selected = rng.choice(labels, size=len(labels), replace=True)
        estimates[index] = float(np.mean([means[label] for label in selected]))
    return {"draws": config.BOOTSTRAP_DRAWS, "seed": config.BOOTSTRAP_SEED, "rng": "numpy_default_rng_pcg64", "labels": "sorted", "source_group_count": len(labels), "record_count": len(values), "mean": float(np.mean(values)), "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))], "group_means": means}


def _validate_analysis(paths: config.RunPaths, protocol: Mapping[str, Any], scores: Mapping[tuple[str, str, str, bool], Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    analysis = _artifact(paths.analysis)
    recomputed = _independent_analysis(protocol, scores)
    if analysis.get("scientific_decision") != recomputed["scientific_decision"] or analysis.get("repeatability", {}).get("count") != recomputed["repeatability"] or analysis.get("baseline", {}).get("R_N_count") != recomputed["baseline"]["R_N_count"] or analysis.get("baseline", {}).get("G_N_N_count") != recomputed["baseline"]["G_N_N_count"]:
        raise DiagnosticError("analysis decision or baseline counts differ")
    for name, count in recomputed["checks"].items():
        if analysis.get("checks", {}).get(name, {}).get("count") != count or analysis.get("checks", {}).get(name, {}).get("denominator") != config.EXPECTED_RECORD_COUNT:
            raise DiagnosticError(f"analysis gate count differs: {name}")
    expected_descriptive = {key: _bootstrap(values, recomputed["groups"]) for key, values in recomputed["descriptive_values"].items()}
    _assert_close(analysis.get("descriptive"), expected_descriptive, 1e-12, "analysis descriptive")
    if len(analysis.get("per_record", [])) != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("analysis per-record count differs")
    return analysis, recomputed


def _validate_final(paths: config.RunPaths, protocol: Mapping[str, Any], score_index: Mapping[tuple[str, str, str, bool], Mapping[str, Any]], analysis: Mapping[str, Any], recomputed: Mapping[str, Any]) -> dict[str, Any]:
    final = _artifact(paths.final)
    if final.get("status") != "complete" or final.get("engineering_decision") != "GO" or final.get("protocol_id") != config.PROTOCOL_ID or final.get("record_count") != config.EXPECTED_RECORD_COUNT or final.get("main_cell_count") != config.EXPECTED_MAIN_CELL_COUNT or final.get("repeat_cell_count") != config.EXPECTED_REPEAT_CELL_COUNT or final.get("cell_count") != config.EXPECTED_TOTAL_CELL_COUNT:
        raise DiagnosticError("final artifact is incomplete")
    if final.get("scientific_decision") != recomputed["scientific_decision"] or final.get("reference_conditioned_audio_head_spec_eligible") is not False or final.get("eligibility") is not False:
        raise DiagnosticError("final decision or eligibility differs")
    if final.get("expected_counts") != {"records": config.EXPECTED_RECORD_COUNT, "main_cells": config.EXPECTED_MAIN_CELL_COUNT, "repeat_cells": config.EXPECTED_REPEAT_CELL_COUNT, "total_cells": config.EXPECTED_TOTAL_CELL_COUNT}:
        raise DiagnosticError("final expected counts differ")
    for name, path in (("protocol_sha256", paths.protocol), ("input_audit_sha256", paths.input_audit), ("media_manifest_sha256", paths.media / "manifest.json"), ("score_manifest_sha256", paths.scores / "manifest.json"), ("analysis_sha256", paths.analysis), ("result_sha256", paths.result)):
        if final.get(name) != file_sha256(path):
            raise DiagnosticError(f"final binding differs: {name}")
    expected_baseline_counts = {"R_N": recomputed["baseline"]["R_N_count"], "G_N_N": recomputed["baseline"]["G_N_N_count"]}
    if final.get("spec_bindings") != protocol.get("spec_bindings") or final.get("parent_runs") != protocol.get("parent_runs") or final.get("gate_counts") != recomputed["checks"] or final.get("baseline_counts") != expected_baseline_counts or final.get("repeatability_count") != recomputed["repeatability"]:
        raise DiagnosticError("final evidence binding differs")
    return {"stage": "final", "status": "valid", "scientific_decision": recomputed["scientific_decision"], "record_count": config.EXPECTED_RECORD_COUNT, "main_cell_count": config.EXPECTED_MAIN_CELL_COUNT, "repeat_cell_count": config.EXPECTED_REPEAT_CELL_COUNT}


def _validate_blocked(paths: config.RunPaths) -> dict[str, Any]:
    final = _artifact(paths.final)
    if final.get("status") != "blocked" or final.get("engineering_decision") != "BLOCKED" or final.get("scientific_decision") is not None or final.get("eligibility") is not False or not paths.result.is_file() or final.get("result_sha256") != file_sha256(paths.result):
        raise DiagnosticError("blocked terminal is invalid")
    if paths.input_audit.is_file():
        _artifact(paths.input_audit)
    return {"stage": "final", "status": "valid", "engineering_decision": "BLOCKED", "scientific_decision": None}


def validate_run(root: Path) -> dict[str, Any]:
    paths = config.RunPaths(root.resolve())
    final = _artifact(paths.final)
    if final.get("status") == "blocked":
        return {"status": "valid", "stages": [_validate_blocked(paths)]}
    protocol = _validate_protocol(paths)
    media = _validate_media(paths, protocol)
    score_manifest, score_index = _validate_scores(paths, protocol)
    analysis, recomputed = _validate_analysis(paths, protocol, score_index)
    final_result = _validate_final(paths, protocol, score_index, analysis, recomputed)
    return {"status": "valid", "stages": [{"stage": "input_audit", "status": "valid", "record_count": config.EXPECTED_RECORD_COUNT}, {"stage": "protocol", "status": "valid"}, media, {"stage": "score", "status": "valid", "cell_count": len(score_manifest["scores"])}, {"stage": "analysis", "status": "valid", "scientific_decision": recomputed["scientific_decision"]}, final_result]}


def write_validation(paths: config.RunPaths, result: Mapping[str, Any]) -> dict[str, Any]:
    payload = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, **dict(result)}
    write_self_hashed_json(paths.validation, payload)
    return verify_self_hashed_json(paths.validation)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently validate an LRS3 Wav2Lip timing-transfer diagnostic")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = config.RunPaths(args.run_root.resolve())
    result = validate_run(paths.root)
    payload = write_validation(paths, result)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
