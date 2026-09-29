from __future__ import annotations

import math
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    DiagnosticError,
    bytes_sha256,
    canonical_json_sha256,
    file_sha256,
    read_json,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .media import decode_pcm16, source_pcm16


def parse_syncnet_log(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = {
        "sync_c": re.findall(r"Confidence:\s+([-+]?\d+(?:\.\d+)?)", text),
        "sync_d": re.findall(r"Min dist:\s+([-+]?\d+(?:\.\d+)?)", text),
        "av_offset": re.findall(r"AV offset:\s+(-?\d+)", text),
    }
    if any(len(values) != 1 for values in matches.values()):
        raise DiagnosticError(f"SyncNet output is missing or ambiguous: {path}")
    sync_c = float(matches["sync_c"][0])
    sync_d = float(matches["sync_d"][0])
    if not math.isfinite(sync_c) or not math.isfinite(sync_d):
        raise DiagnosticError(f"SyncNet output is non-finite: {path}")
    return {"sync_c": sync_c, "sync_d": sync_d, "av_offset": int(matches["av_offset"][0])}


def reconstruct_global(matrix: np.ndarray) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 31 or value.shape[0] < 1 or not np.isfinite(value).all():
        raise DiagnosticError(f"distance matrix must be finite with shape [window,31], got {value.shape}")
    curve = np.mean(value, axis=0)
    minimum_index = int(np.argmin(curve))
    minimum = float(curve[minimum_index])
    confidence = float(np.median(curve) - minimum)
    return {
        "curve": [float(item) for item in curve],
        "offsets": [config.VSHIFT - index for index in range(31)],
        "min_index": minimum_index,
        "offset": config.VSHIFT - minimum_index,
        "sync_d": minimum,
        "sync_c": confidence,
    }


def supported_window_rows(
    window_count: int,
    frame_count: int,
    audio_sample_count: int,
    common_support_samples: Sequence[int] | None = None,
) -> list[int]:
    expected_support = [0, min(frame_count * config.SAMPLES_PER_FRAME, audio_sample_count)]
    if common_support_samples is not None:
        if len(common_support_samples) != 2 or [int(value) for value in common_support_samples] != expected_support:
            raise DiagnosticError("common audio/video support does not match the frozen input")
        support_stop = int(common_support_samples[1])
    else:
        support_stop = expected_support[1]
    coordinate_count = min(frame_count, support_stop // config.SAMPLES_PER_FRAME)
    expected_windows = coordinate_count - config.WINDOW_FRAMES
    if window_count != expected_windows:
        raise DiagnosticError(f"distance matrix window count {window_count} does not match frozen coordinates {expected_windows}")
    start = config.VSHIFT
    stop = window_count - config.VSHIFT
    return list(range(start, stop))


def _peak(curve: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float64)
    if values.ndim != 1 or values.size != 31 or not np.isfinite(values).all():
        raise DiagnosticError("local curve must contain 31 finite values")
    order = np.sort(values)
    minimum_index = int(np.argmin(values))
    minimum = float(order[0])
    second = float(order[1])
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


def local_evidence(
    matrix: np.ndarray,
    mapping: Sequence[int],
    frame_count: int,
    audio_sample_count: int,
    common_support_samples: Sequence[int] | None = None,
) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64)
    support = [0, min(frame_count * config.SAMPLES_PER_FRAME, audio_sample_count)]
    rows = supported_window_rows(value.shape[0], frame_count, audio_sample_count, common_support_samples)
    map_array = np.asarray(mapping, dtype=np.int64)
    if map_array.shape != (frame_count,) or int(map_array.min()) < 0 or int(map_array.max()) >= frame_count:
        raise DiagnosticError("video mapping does not match the frame count")
    displacement = map_array - np.arange(frame_count, dtype=np.int64)
    plus = [row for row in rows if np.all(displacement[row : row + config.WINDOW_FRAMES] == 3)]
    minus = [row for row in rows if np.all(displacement[row : row + config.WINDOW_FRAMES] == -3)]
    if len(plus) < config.MIN_LOCAL_ROWS or len(minus) < config.MIN_LOCAL_ROWS:
        raise DiagnosticError(f"local evidence has fewer than {config.MIN_LOCAL_ROWS} supported rows")
    evidence: dict[str, Any] = {"supported_rows": rows, "common_support_samples": support}
    for name, selected in (("PLUS", plus), ("MINUS", minus)):
        curve = np.mean(value[selected, :], axis=0)
        evidence[name] = {"rows": selected, "curve": [float(item) for item in curve], **_peak(curve)}
    return evidence


def _run_worker(media: Path, matrix: Path, worker_result: Path, tmp_dir: Path, log: Path, reference: str) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(config.SYNCNET_PYTHON),
        str(Path(__file__).with_name("syncnet_worker.py")),
        "--media",
        str(media),
        "--model",
        str(config.SYNCNET_MODEL),
        "--tmp-dir",
        str(tmp_dir),
        "--reference",
        reference,
        "--matrix",
        str(matrix),
        "--result",
        str(worker_result),
        "--batch-size",
        "20",
        "--vshift",
        str(config.VSHIFT),
    ]
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(config.SYNCNET_ROOT), stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not matrix.is_file() or not worker_result.is_file():
        raise DiagnosticError(f"official SyncNet worker failed: {log}")


def _score_sidecar_matches(sidecar: Mapping[str, Any], protocol_sha: str, sample_id: str, arm: str, media: Path, matrix: Path) -> bool:
    return bool(
        sidecar.get("protocol_id") == config.PROTOCOL_ID
        and sidecar.get("protocol_sha256") == protocol_sha
        and sidecar.get("sample_id") == sample_id
        and sidecar.get("arm") == arm
        and sidecar.get("media_sha256") == file_sha256(media)
        and sidecar.get("matrix_sha256") == file_sha256(matrix)
    )


def score_one(record: Mapping[str, Any], media_row: Mapping[str, Any], protocol: Mapping[str, Any], paths) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    arm = str(media_row["arm"])
    media = Path(str(media_row["output"]))
    audio = Path(str(record["natural_audio"]["path"]))
    protocol_sha = str(protocol.get("_sha256", ""))
    if not protocol_sha or not media.is_file():
        raise DiagnosticError(f"score input is missing: {sample_id}/{arm}")
    if decode_pcm16(media) != source_pcm16(audio):
        raise DiagnosticError(f"media PCM is not byte-identical to natural audio: {sample_id}/{arm}")
    cell_id = f"{sample_id}__{arm}"
    cell_dir = paths.scores / "cells" / cell_id
    matrix = cell_dir / "distance.npy"
    worker_result = cell_dir / "worker.json"
    score_log = paths.scores / "logs" / f"{cell_id}.log"
    sidecar = cell_dir / "score.json"
    if sidecar.is_file() and matrix.is_file() and worker_result.is_file():
        prior = verify_self_hashed_json(sidecar)
        if not _score_sidecar_matches(prior, protocol_sha, sample_id, arm, media, matrix):
            raise DiagnosticError(f"existing score identity changed: {cell_id}")
        return prior
    if any(path.exists() for path in (matrix, worker_result, sidecar)):
        raise DiagnosticError(f"partial score cell cannot be resumed: {cell_id}")
    _run_worker(media, matrix, worker_result, cell_dir / "tmp", score_log, cell_id)
    worker = read_json(worker_result)
    parsed = parse_syncnet_log(score_log)
    actual_matrix = np.load(matrix, allow_pickle=False)
    global_metrics = reconstruct_global(actual_matrix)
    for field, tolerance in (("sync_c", 0.001), ("sync_d", 0.001)):
        if abs(float(parsed[field]) - float(global_metrics[field])) > tolerance:
            raise DiagnosticError(f"distance matrix does not reproduce official {field}: {cell_id}")
    if int(parsed["av_offset"]) != int(global_metrics["offset"]):
        raise DiagnosticError(f"distance matrix does not reproduce official offset: {cell_id}")
    if int(worker.get("offset", 10_000)) != int(parsed["av_offset"]) or abs(float(worker.get("confidence", float("nan"))) - float(global_metrics["sync_c"])) > 0.001:
        raise DiagnosticError(f"worker result differs from the reconstructed official result: {cell_id}")
    frame_count = int(record["crop"]["frame_count"])
    audio_sample_count = int(record["natural_audio_format"]["sample_count"])
    local_mask_mapping = [int(value) for value in record["crop"]["mappings"][config.WARP_ARM]]
    local = local_evidence(
        actual_matrix,
        local_mask_mapping,
        frame_count,
        audio_sample_count,
        record.get("common_support_samples"),
    )
    row = {
        "schema_version": 1,
        "stage_id": "score",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_sha256": protocol_sha,
        "sample_id": sample_id,
        "source_group": str(record["source_group"]),
        "arm": arm,
        "media": str(media),
        "media_sha256": file_sha256(media),
        "natural_audio_sha256": file_sha256(audio),
        "natural_audio_pcm_sha256": bytes_sha256(source_pcm16(audio)),
        "matrix": str(matrix),
        "matrix_sha256": file_sha256(matrix),
        "matrix_shape": [int(value) for value in actual_matrix.shape],
        "window_frame_starts": list(range(int(actual_matrix.shape[0]))),
        "vshift": config.VSHIFT,
        "column_offsets": [config.VSHIFT - index for index in range(31)],
        "local_mask_mapping_sha256": canonical_json_sha256(local_mask_mapping),
        "official_log": parsed,
        "worker_result": str(worker_result),
        "worker_result_sha256": file_sha256(worker_result),
        "reconstructed": global_metrics,
        "local": local,
        "score_log": str(score_log),
        "runtime": {
            "syncnet_python": str(config.SYNCNET_PYTHON),
            "syncnet_model": str(config.SYNCNET_MODEL),
            "syncnet_model_sha256": config.SYNCNET_MODEL_SHA256,
            "vshift": config.VSHIFT,
        },
    }
    write_self_hashed_json(sidecar, row)
    return verify_self_hashed_json(sidecar)


def run_scores(protocol: Mapping[str, Any], media_manifest: Mapping[str, Any], paths) -> dict[str, Any]:
    if media_manifest.get("status") != "complete" or media_manifest.get("cell_count") != config.expected_cell_count():
        raise DiagnosticError("media manifest is incomplete")
    paths.scores.mkdir(parents=True, exist_ok=True)
    records = {str(record["sample_id"]): record for record in protocol.get("records", [])}
    rows: list[dict[str, Any]] = []
    for index, media_record in enumerate(media_manifest.get("rows", []), 1):
        sample_id = str(media_record["sample_id"])
        record = records.get(sample_id)
        if record is None:
            raise DiagnosticError(f"protocol record is missing for media row: {sample_id}")
        for arm in config.ARMS:
            arm_row = media_record.get("arms", {}).get(arm)
            if not isinstance(arm_row, Mapping):
                raise DiagnosticError(f"media arm is missing: {sample_id}/{arm}")
            row = score_one(record, {"arm": arm, **arm_row}, protocol, paths)
            rows.append(row)
        print(f"SCORE {index}/{len(media_manifest['rows'])} {sample_id}", flush=True)
    if len(rows) != config.expected_cell_count():
        raise DiagnosticError(f"score cell count is incomplete: {len(rows)}/{config.expected_cell_count()}")
    manifest = {
        "schema_version": 2,
        "stage_id": "score",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "protocol_sha256": protocol.get("_sha256"),
        "media_manifest_sha256": file_sha256(paths.media / "manifest.json"),
        "status": "complete",
        "record_count": config.EXPECTED_RECORD_COUNT,
        "cell_count": len(rows),
        "expected_cell_count": config.expected_cell_count(),
        "arms": list(config.ARMS),
        "scores": rows,
    }
    write_self_hashed_json(paths.scores / "manifest.json", manifest)
    return manifest
