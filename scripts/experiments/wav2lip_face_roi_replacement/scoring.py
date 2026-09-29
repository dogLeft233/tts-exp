from __future__ import annotations

import math
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.lrs3_real_video_local_timing.media import (
    decode_pcm16,
    source_pcm16,
)

from . import config
from .common import (
    ProtocolError,
    assert_finite,
    bytes_sha256,
    file_sha256,
    read_json,
    verify_self_hashed_json,
    write_or_verify,
    write_self_hashed_json,
)


def parse_syncnet_log(path: Path) -> dict[str, float | int] | None:
    """Parse the human-readable official SyncNet summary when it is present."""
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = {
        "sync_c": re.findall(r"Confidence:\s+([-+]?\d+(?:\.\d+)?)", text),
        "sync_d": re.findall(r"Min dist:\s+([-+]?\d+(?:\.\d+)?)", text),
        "av_offset": re.findall(r"AV offset:\s+(-?\d+)", text),
    }
    if not any(matches.values()):
        return None
    if any(len(values) != 1 for values in matches.values()):
        raise ProtocolError(f"SyncNet output is missing or ambiguous: {path}")
    result: dict[str, float | int] = {
        "sync_c": float(matches["sync_c"][0]),
        "sync_d": float(matches["sync_d"][0]),
        "av_offset": int(matches["av_offset"][0]),
    }
    if not math.isfinite(float(result["sync_c"])) or not math.isfinite(float(result["sync_d"])):
        raise ProtocolError(f"SyncNet output is non-finite: {path}")
    return result


def reconstruct_global(matrix: np.ndarray) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 31 or value.shape[0] < 1 or not np.isfinite(value).all():
        raise ProtocolError(f"distance matrix must be finite with shape [window,31], got {value.shape}")
    curve = np.mean(value, axis=0)
    minimum_index = int(np.argmin(curve))
    minimum = float(curve[minimum_index])
    return {
        "curve": [float(item) for item in curve],
        "offsets": [config.VSHIFT - index for index in range(31)],
        "min_index": minimum_index,
        "offset": config.VSHIFT - minimum_index,
        "sync_d": minimum,
        "sync_c": float(np.median(curve) - minimum),
    }


def supported_window_rows(window_count: int, frame_count: int, audio_sample_count: int) -> list[int]:
    coordinate_count = min(int(frame_count), int(audio_sample_count) // config.SAMPLES_PER_FRAME)
    expected = coordinate_count - config.WINDOW_FRAMES
    if int(window_count) != expected or expected <= 0:
        raise ProtocolError(f"distance matrix window count {window_count} does not match frozen coordinates {expected}")
    return list(range(config.VSHIFT, window_count - config.VSHIFT))


def _peak(curve: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float64)
    if values.ndim != 1 or values.size != 31 or not np.isfinite(values).all():
        raise ProtocolError("local curve must contain 31 finite values")
    ordered = np.sort(values)
    minimum_index = int(np.argmin(values))
    minimum = float(ordered[0])
    second = float(ordered[1])
    offset = config.VSHIFT - minimum_index
    gap = second - minimum
    return {
        "min_index": minimum_index,
        "offset": offset,
        "min": minimum,
        "second_min": second,
        "peak_gap": float(gap),
        "clear": bool(gap > config.PEAK_GAP_THRESHOLD and -config.VSHIFT < offset < config.VSHIFT),
    }


def local_evidence(matrix: np.ndarray, masks: Mapping[str, Any]) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64)
    plus = [int(item) for item in masks.get("plus_rows", [])]
    minus = [int(item) for item in masks.get("minus_rows", [])]
    if len(plus) < config.MIN_LOCAL_ROWS or len(minus) < config.MIN_LOCAL_ROWS:
        raise ProtocolError("local evidence has fewer than the registered minimum rows")
    for name, rows in (("PLUS", plus), ("MINUS", minus)):
        if min(rows, default=-1) < 0 or max(rows, default=-1) >= value.shape[0]:
            raise ProtocolError(f"{name} rows are not supported by the distance matrix")
    evidence: dict[str, Any] = {
        "supported_rows": list(range(value.shape[0])),
        "common_window_rows": [int(item) for item in masks.get("common_window_rows", [])],
    }
    for name, rows in (("PLUS", plus), ("MINUS", minus)):
        curve = np.mean(value[rows, :], axis=0)
        evidence[name] = {"rows": rows, "curve": [float(item) for item in curve], **_peak(curve)}
    return evidence


def common_global(matrix: np.ndarray, masks: Mapping[str, Any]) -> dict[str, Any]:
    rows = [int(item) for item in masks.get("common_window_rows", [])]
    if not rows or min(rows) < 0 or max(rows) >= matrix.shape[0]:
        raise ProtocolError("common window rows are not supported by the distance matrix")
    return reconstruct_global(matrix[rows, :])


def _run_worker(media: Path, matrix: Path, worker_result: Path, tmp_dir: Path, log: Path, reference: str) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(config.SYNCNET_PYTHON),
        str(config.SYNCNET_WORKER),
        "--media", str(media),
        "--model", str(config.SYNCNET_MODEL),
        "--tmp-dir", str(tmp_dir),
        "--reference", reference,
        "--matrix", str(matrix),
        "--result", str(worker_result),
        "--batch-size", str(config.SYNCNET_BATCH_SIZE),
        "--vshift", str(config.VSHIFT),
    ]
    environment = dict(os.environ)
    environment["PATH"] = f"{config.FFMPEG.parent}:{environment.get('PATH', '')}"
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(config.SYNCNET_ROOT), env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not matrix.is_file() or not worker_result.is_file():
        raise ProtocolError(f"official SyncNet worker failed: {log}")


def _audio_item(record: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    manifest_row = record.get("audio", {}).get("manifest_row")
    if not isinstance(manifest_row, Mapping) or not isinstance(manifest_row.get("arms"), list):
        raise ProtocolError(f"audio manifest row is missing: {record.get('sample_id')}")
    value = next((item for item in manifest_row["arms"] if isinstance(item, Mapping) and str(item.get("arm")) == arm), None)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"audio arm is missing: {record.get('sample_id')}/{arm}")
    return value


def _score_sidecar_matches(sidecar: Mapping[str, Any], *, protocol_sha: str, media_manifest_sha: str, sample_id: str, video_arm: str, audio_arm: str, repeat: bool, media: Path, matrix: Path) -> bool:
    return bool(
        sidecar.get("protocol_id") == config.PROTOCOL_ID
        and sidecar.get("protocol_sha256") == protocol_sha
        and sidecar.get("media_manifest_sha256") == media_manifest_sha
        and sidecar.get("sample_id") == sample_id
        and sidecar.get("video_arm") == video_arm
        and sidecar.get("audio_arm") == audio_arm
        and bool(sidecar.get("repeat", False)) is repeat
        and sidecar.get("media_sha256") == file_sha256(media)
        and sidecar.get("matrix_sha256") == file_sha256(matrix)
    )


def score_one(record: Mapping[str, Any], media_row: Mapping[str, Any], protocol: Mapping[str, Any], paths: config.RunPaths, *, stage: str, repeat: bool = False) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    video_arm = str(media_row.get("video_arm"))
    audio_arm = str(media_row.get("audio_arm"))
    media = Path(str(media_row.get("output", "")))
    protocol_sha = str(protocol.get("_sha256", ""))
    media_manifest_path = paths.media / stage / "manifest.json"
    media_manifest_sha = file_sha256(media_manifest_path)
    if not protocol_sha or not media.is_file() or not media_manifest_path.is_file():
        raise ProtocolError(f"score input is missing: {sample_id}/{video_arm}/{audio_arm}")
    audio_item = _audio_item(record, audio_arm)
    audio = Path(str(audio_item.get("output", audio_item.get("path", ""))))
    audio_sha = str(audio_item.get("output_sha256", audio_item.get("container_sha256", "")))
    expected_pcm = source_pcm16(audio)
    if not audio.is_file() or not audio_sha or file_sha256(audio) != audio_sha:
        raise ProtocolError(f"audio input hash changed: {sample_id}/{audio_arm}")
    if decode_pcm16(media) != expected_pcm:
        raise ProtocolError(f"media PCM differs from selected audio arm: {sample_id}/{video_arm}/{audio_arm}")
    frame_count = int(media_row.get("frame_count", -1))
    expected_matrix_rows = min(frame_count, len(expected_pcm) // (config.PCM_SAMPLE_WIDTH * config.SAMPLES_PER_FRAME)) - config.WINDOW_FRAMES
    if expected_matrix_rows <= 0:
        raise ProtocolError(f"video/audio support is too short: {sample_id}/{video_arm}/{audio_arm}")
    cell_id = config.cell_key(video_arm, audio_arm, repeat)
    cell_dir = paths.scores / stage / ("repeat_cells" if repeat else "cells") / f"{sample_id}__{cell_id}"
    matrix = cell_dir / "distance.npy"
    worker_result = cell_dir / "worker.json"
    score_log = paths.scores / stage / "logs" / f"{sample_id}__{cell_arm_name(video_arm, audio_arm, repeat)}.log"
    sidecar = cell_dir / "score.json"
    if sidecar.is_file() and matrix.is_file() and worker_result.is_file():
        prior = verify_self_hashed_json(sidecar)
        if not _score_sidecar_matches(prior, protocol_sha=protocol_sha, media_manifest_sha=media_manifest_sha, sample_id=sample_id, video_arm=video_arm, audio_arm=audio_arm, repeat=repeat, media=media, matrix=matrix):
            raise ProtocolError(f"existing score identity changed: {sample_id}/{video_arm}/{audio_arm}/{repeat}")
        return prior
    if any(path.exists() for path in (matrix, worker_result, sidecar)):
        raise ProtocolError(f"partial score cell cannot be resumed: {sample_id}/{video_arm}/{audio_arm}/{repeat}")
    _run_worker(media, matrix, worker_result, cell_dir / "tmp", score_log, f"{stage}_{sample_id}_{cell_id}")
    worker = read_json(worker_result)
    actual_matrix = np.asarray(np.load(matrix, allow_pickle=False), dtype=np.float64)
    if actual_matrix.shape != (expected_matrix_rows, 31):
        raise ProtocolError(f"distance matrix shape differs from frozen support: {sample_id}/{video_arm}/{audio_arm}: {actual_matrix.shape} != {(expected_matrix_rows, 31)}")
    global_metrics = reconstruct_global(actual_matrix)
    worker_metrics = {"sync_c": float(worker.get("confidence", float("nan"))), "sync_d": float(np.mean(actual_matrix, axis=0).min()), "av_offset": int(worker.get("offset", 10_000))}
    if not math.isfinite(worker_metrics["sync_c"]) or worker_metrics["av_offset"] != global_metrics["offset"] or abs(worker_metrics["sync_c"] - global_metrics["sync_c"]) > 0.001:
        raise ProtocolError(f"worker result does not reproduce the official matrix result: {sample_id}/{video_arm}/{audio_arm}")
    parsed = parse_syncnet_log(score_log)
    if parsed is not None and (abs(float(parsed["sync_c"]) - global_metrics["sync_c"]) > 0.001 or abs(float(parsed["sync_d"]) - global_metrics["sync_d"]) > 0.001 or int(parsed["av_offset"]) != int(global_metrics["offset"])):
        raise ProtocolError(f"SyncNet log does not reproduce the distance matrix: {sample_id}/{video_arm}/{audio_arm}")
    masks = record.get("masks")
    if not isinstance(masks, Mapping):
        raise ProtocolError(f"timing masks are missing: {sample_id}")
    supported = supported_window_rows(actual_matrix.shape[0], frame_count, len(expected_pcm) // config.PCM_SAMPLE_WIDTH)
    local_masks = {**masks, "common_window_rows": [row for row in masks.get("common_window_rows", []) if int(row) < actual_matrix.shape[0]]}
    local = local_evidence(actual_matrix, local_masks)
    common = common_global(actual_matrix, local_masks)
    row: dict[str, Any] = {
        "schema_version": 1,
        "stage_id": "score",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_sha256": protocol_sha,
        "media_manifest_sha256": media_manifest_sha,
        "stage": stage,
        "sample_id": sample_id,
        "source_group": str(record["source_group"]),
        "video_arm": video_arm,
        "audio_arm": audio_arm,
        "repeat": bool(repeat),
        "media": str(media.resolve()),
        "media_sha256": file_sha256(media),
        "audio": str(audio.resolve()),
        "audio_sha256": audio_sha,
        "audio_pcm_sha256": bytes_sha256(expected_pcm),
        "matrix": str(matrix.resolve()),
        "matrix_sha256": file_sha256(matrix),
        "matrix_shape": [int(value) for value in actual_matrix.shape],
        "supported_window_rows": supported,
        "vshift": config.VSHIFT,
        "column_offsets": [config.VSHIFT - index for index in range(31)],
        "worker_result": str(worker_result.resolve()),
        "worker_result_sha256": file_sha256(worker_result),
        "official_log": parsed or {"source": "worker_result"},
        "worker_official": worker_metrics,
        "reconstructed": global_metrics,
        "common_global": common,
        "local": local,
        "runtime": {"syncnet_python": str(config.SYNCNET_PYTHON), "syncnet_model": str(config.SYNCNET_MODEL), "syncnet_model_sha256": config.SYNCNET_MODEL_SHA256, "batch_size": config.SYNCNET_BATCH_SIZE, "vshift": config.VSHIFT},
    }
    assert_finite(row)
    write_self_hashed_json(sidecar, row)
    return verify_self_hashed_json(sidecar)


def cell_arm_name(video_arm: str, audio_arm: str, repeat: bool) -> str:
    suffix = "_repeat" if repeat else ""
    return f"{video_arm}_{audio_arm}{suffix}"


def _media_cell(media_manifest: Mapping[str, Any], sample_id: str, video_arm: str, audio_arm: str) -> Mapping[str, Any]:
    row = next((item for item in media_manifest.get("rows", []) if isinstance(item, Mapping) and str(item.get("sample_id")) == sample_id), None)
    if not isinstance(row, Mapping) or not isinstance(row.get("cells"), Mapping):
        raise ProtocolError(f"media record is missing: {sample_id}")
    key = config.cell_key(video_arm, audio_arm)
    cell = row["cells"].get(key)
    if not isinstance(cell, Mapping):
        raise ProtocolError(f"media cell is missing: {sample_id}/{key}")
    return cell


def _score_specs(stage: str) -> tuple[Sequence[tuple[str, str]], Sequence[tuple[str, str]]]:
    if stage == "control":
        return config.CONTROL_MAIN_CELL_SPECS, config.CONTROL_REPEAT_CELL_SPECS
    if stage == "bridge":
        return config.BRIDGE_CELL_SPECS, ()
    raise ValueError(f"unknown score stage: {stage}")


def run_scores(protocol: Mapping[str, Any], media_manifest: Mapping[str, Any], paths: config.RunPaths, stage: str) -> dict[str, Any]:
    if media_manifest.get("status") != "complete" or media_manifest.get("stage") != stage:
        raise ProtocolError(f"{stage} media manifest is incomplete")
    main_specs, repeat_specs = _score_specs(stage)
    score_rows: list[dict[str, Any]] = []
    records = protocol.get("records")
    if not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("protocol records are incomplete")
    for index, record in enumerate(records, 1):
        sample_id = str(record["sample_id"])
        for video_arm, audio_arm in main_specs:
            media_row = _media_cell(media_manifest, sample_id, video_arm, audio_arm)
            score_rows.append(score_one(record, media_row, protocol, paths, stage=stage, repeat=False))
        for video_arm, audio_arm in repeat_specs:
            media_row = _media_cell(media_manifest, sample_id, video_arm, audio_arm)
            score_rows.append(score_one(record, media_row, protocol, paths, stage=stage, repeat=True))
        print(f"SCORE {index}/{len(records)} {sample_id} stage={stage}", flush=True)
    expected_main = config.EXPECTED_CONTROL_MAIN_CELL_COUNT if stage == "control" else config.EXPECTED_BRIDGE_CELL_COUNT
    expected_repeat = config.EXPECTED_CONTROL_REPEAT_CELL_COUNT if stage == "control" else 0
    payload = {
        "schema_version": 1,
        "stage_id": "scores",
        "protocol_id": config.PROTOCOL_ID,
        "stage": stage,
        "status": "complete",
        "protocol_sha256": str(protocol["_sha256"]),
        "media_manifest_sha256": file_sha256(paths.media / stage / "manifest.json"),
        "record_count": len(records),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "main_score_count": sum(not bool(row.get("repeat")) for row in score_rows),
        "repeat_score_count": sum(bool(row.get("repeat")) for row in score_rows),
        "score_count": len(score_rows),
        "expected_main_score_count": expected_main,
        "expected_repeat_score_count": expected_repeat,
        "expected_score_count": expected_main + expected_repeat,
        "scores": score_rows,
    }
    assert_finite(payload)
    return write_or_verify(paths.scores / stage / "manifest.json", payload)
