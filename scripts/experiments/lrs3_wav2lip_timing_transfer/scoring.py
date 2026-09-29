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
    DiagnosticError,
    bytes_sha256,
    file_sha256,
    read_json,
    verify_self_hashed_json,
    write_self_hashed_json,
)

SYNCNET_ROOT = config.REPO / "third_party/syncnet_python"
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"
SYNCNET_PYTHON = Path.home() / ".venvs/syncnet/bin/python"
SYNCNET_WORKER = config.REPO / "scripts/experiments/lrs3_real_video_local_timing/syncnet_worker.py"


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
    return {
        "curve": [float(item) for item in curve],
        "offsets": [config.VSHIFT - index for index in range(31)],
        "min_index": minimum_index,
        "offset": config.VSHIFT - minimum_index,
        "sync_d": minimum,
        "sync_c": float(np.median(curve) - minimum),
    }


def _peak(curve: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float64)
    if values.ndim != 1 or values.size != 31 or not np.isfinite(values).all():
        raise DiagnosticError("local curve must contain 31 finite values")
    ordered = np.sort(values)
    minimum_index = int(np.argmin(values))
    offset = config.VSHIFT - minimum_index
    minimum = float(ordered[0])
    second = float(ordered[1])
    return {
        "min_index": minimum_index,
        "offset": offset,
        "min": minimum,
        "second_min": second,
        "peak_gap": float(second - minimum),
        "clear": bool(second - minimum > config.PEAK_GAP_THRESHOLD and -config.VSHIFT < offset < config.VSHIFT),
    }


def _local_evidence(matrix: np.ndarray, mask: Mapping[str, Any], name: str) -> dict[str, Any]:
    rows = [int(value) for value in mask[f"{name.lower()}_rows"]]
    if len(rows) < config.MIN_LOCAL_ROWS or min(rows, default=-1) < 0 or max(rows, default=-1) >= matrix.shape[0]:
        raise DiagnosticError(f"{name} rows are not supported by the distance matrix")
    curve = np.mean(matrix[rows, :], axis=0)
    return {"rows": rows, "curve": [float(value) for value in curve], **_peak(curve)}


def _common_global(matrix: np.ndarray, mask: Mapping[str, Any]) -> dict[str, Any]:
    rows = [int(value) for value in mask["common_window_rows"]]
    if not rows or min(rows) < 0 or max(rows) >= matrix.shape[0]:
        raise DiagnosticError("common window rows are not supported by the distance matrix")
    return reconstruct_global(matrix[rows, :])


def _expected_matrix_rows(record: Mapping[str, Any], video_arm: str) -> int:
    frame_count = int(record["masks"]["frame_counts"][video_arm])
    sample_count = int(record["natural_audio"]["sample_count"])
    expected = min(frame_count, sample_count // config.SAMPLES_PER_FRAME) - config.WINDOW_FRAMES
    if expected <= 0:
        raise DiagnosticError(f"video/audio support is too short: {record.get('sample_id')}/{video_arm}")
    return expected


def _run_worker(media: Path, matrix: Path, worker_result: Path, tmp_dir: Path, log: Path, reference: str) -> None:
    for path in (media, SYNCNET_MODEL, SYNCNET_WORKER, SYNCNET_PYTHON):
        if not path.is_file():
            raise DiagnosticError(f"SyncNet input is missing: {path}")
    matrix.parent.mkdir(parents=True, exist_ok=True)
    worker_result.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(SYNCNET_PYTHON),
        str(SYNCNET_WORKER),
        "--media",
        str(media),
        "--model",
        str(SYNCNET_MODEL),
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
    environment = dict(os.environ)
    environment["PATH"] = f"/home/wjj/miniconda3/bin:{environment.get('PATH', '')}"
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(SYNCNET_ROOT), env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not matrix.is_file() or not worker_result.is_file():
        raise DiagnosticError(f"official SyncNet worker failed: {log}")


def _score_identity(row: Mapping[str, Any], cell: Mapping[str, Any], media: Path, audio: Path, matrix: Path, worker_result: Path, log: Path) -> bool:
    return bool(
        row.get("protocol_id") == config.PROTOCOL_ID
        and row.get("protocol_sha256") == cell["protocol_sha256"]
        and row.get("sample_id") == cell["sample_id"]
        and row.get("video_arm") == cell["video_arm"]
        and row.get("audio_arm") == cell["audio_arm"]
        and bool(row.get("repeat", False)) == bool(cell.get("repeat", False))
        and row.get("media_sha256") == file_sha256(media)
        and row.get("audio_container_sha256") == file_sha256(audio)
        and row.get("matrix_sha256") == file_sha256(matrix)
        and row.get("worker_result_sha256") == file_sha256(worker_result)
        and row.get("score_log_sha256") == file_sha256(log)
    )


def _build_score_row(cell: Mapping[str, Any], protocol: Mapping[str, Any], paths: config.RunPaths, matrix: np.ndarray, worker: Mapping[str, Any], parsed: Mapping[str, Any], matrix_path: Path, worker_result_path: Path, log: Path, media: Path, audio: Path, reference: str, tmp_dir: Path) -> dict[str, Any]:
    record = cell["record"]
    mask = record["masks"]
    expected_rows = _expected_matrix_rows(record, str(cell["video_arm"]))
    if matrix.shape != (expected_rows, 31):
        raise DiagnosticError(f"distance matrix shape differs from frozen support: {cell['sample_id']}/{cell['video_arm']}/{cell['audio_arm']}")
    official_global = reconstruct_global(matrix)
    for field in ("sync_c", "sync_d"):
        if abs(float(parsed[field]) - float(official_global[field])) > 0.001:
            raise DiagnosticError(f"distance matrix does not reproduce official {field}: {cell['sample_id']}/{cell['video_arm']}/{cell['audio_arm']}")
    if int(parsed["av_offset"]) != int(official_global["offset"]):
        raise DiagnosticError(f"distance matrix does not reproduce official offset: {cell['sample_id']}/{cell['video_arm']}/{cell['audio_arm']}")
    if int(worker.get("offset", 10_000)) != int(parsed["av_offset"]) or abs(float(worker.get("confidence", float("nan"))) - float(official_global["sync_c"])) > 0.001:
        raise DiagnosticError(f"worker result differs from reconstructed score: {cell['sample_id']}/{cell['video_arm']}/{cell['audio_arm']}")
    if worker.get("matrix_sha256") != file_sha256(matrix_path) or worker.get("media_sha256") != file_sha256(media) or worker.get("input_pcm_sha256") != bytes_sha256(source_pcm16(audio)) or worker.get("extracted_pcm_sha256") != worker.get("input_pcm_sha256"):
        raise DiagnosticError(f"worker input binding differs: {cell['sample_id']}/{cell['video_arm']}/{cell['audio_arm']}")
    local = {
        "PLUS": _local_evidence(matrix, mask, "PLUS"),
        "MINUS": _local_evidence(matrix, mask, "MINUS"),
    }
    common_global = _common_global(matrix, mask)
    return {
        "schema_version": 1,
        "stage_id": "score",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_sha256": str(cell["protocol_sha256"]),
        "sample_id": str(cell["sample_id"]),
        "source_group": str(record["source_group"]),
        "video_arm": str(cell["video_arm"]),
        "audio_arm": str(cell["audio_arm"]),
        "cell_id": str(cell["cell_id"]),
        "repeat": bool(cell.get("repeat", False)),
        "repeat_of": cell.get("repeat_of"),
        "media": str(media.resolve()),
        "media_sha256": file_sha256(media),
        "audio": str(audio.resolve()),
        "audio_container_sha256": file_sha256(audio),
        "audio_pcm_sha256": bytes_sha256(source_pcm16(audio)),
        "matrix": str(matrix_path.resolve()),
        "matrix_sha256": file_sha256(matrix_path),
        "matrix_shape": [int(value) for value in matrix.shape],
        "window_frame_starts": list(range(int(matrix.shape[0]))),
        "vshift": config.VSHIFT,
        "batch_size": 20,
        "column_offsets": [config.VSHIFT - index for index in range(31)],
        "official_log": dict(parsed),
        "score_log": str(log.resolve()),
        "score_log_sha256": file_sha256(log),
        "worker_result": str(worker_result_path.resolve()),
        "worker_result_sha256": file_sha256(worker_result_path),
        "worker_reference": reference,
        "worker_tmp_dir": str(tmp_dir.resolve()),
        "official_global": official_global,
        "common_global": common_global,
        "local": local,
        "runtime": {
            "syncnet_python": str(SYNCNET_PYTHON.resolve()),
            "syncnet_model": str(SYNCNET_MODEL.resolve()),
            "syncnet_model_sha256": file_sha256(SYNCNET_MODEL),
            "syncnet_worker": str(SYNCNET_WORKER.resolve()),
            "syncnet_worker_sha256": file_sha256(SYNCNET_WORKER),
        },
    }


def _score_one(cell: Mapping[str, Any], protocol: Mapping[str, Any], paths: config.RunPaths) -> dict[str, Any]:
    cell_id = str(cell["cell_id"])
    media = Path(str(cell["media"]))
    audio = Path(str(cell["audio"]))
    if not media.is_file() or not audio.is_file():
        raise DiagnosticError(f"score media/audio is missing: {cell_id}")
    expected_pcm = source_pcm16(audio)
    if decode_pcm16(media) != expected_pcm:
        raise DiagnosticError(f"media PCM differs from bound audio: {cell_id}")
    cell_dir = paths.scores / "cells" / cell_id
    matrix_path = cell_dir / "distance.npy"
    worker_result_path = cell_dir / "worker.json"
    log = paths.scores / "logs" / f"{cell_id}.log"
    sidecar_path = cell_dir / "score.json"
    reference = f"lrs3_w2l_transfer_{cell_id}"
    tmp_dir = cell_dir / "tmp"
    complete = all(path.is_file() for path in (matrix_path, worker_result_path, log, sidecar_path))
    any_outputs = any(path.exists() for path in (matrix_path, worker_result_path, log, sidecar_path))
    if complete:
        sidecar = verify_self_hashed_json(sidecar_path)
        if not _score_identity(sidecar, cell, media, audio, matrix_path, worker_result_path, log):
            raise DiagnosticError(f"existing score identity changed: {cell_id}")
    elif any_outputs:
        raise DiagnosticError(f"partial score cell cannot be resumed: {cell_id}")
    else:
        _run_worker(media, matrix_path, worker_result_path, tmp_dir, log, reference)
    matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float64)
    worker = read_json(worker_result_path)
    parsed = parse_syncnet_log(log)
    row = _build_score_row(cell, protocol, paths, matrix, worker, parsed, matrix_path, worker_result_path, log, media, audio, reference, tmp_dir)
    if complete:
        if dict(sidecar) != {**row, "artifact_sha256": sidecar["artifact_sha256"]}:
            raise DiagnosticError(f"existing score evidence changed: {cell_id}")
        return sidecar
    write_self_hashed_json(sidecar_path, row)
    return verify_self_hashed_json(sidecar_path)


def _manifest_cell_specs(protocol: Mapping[str, Any], media_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    record_map = {str(record["sample_id"]): record for record in protocol["records"]}
    rows = media_manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != config.EXPECTED_RECORD_COUNT:
        raise DiagnosticError("media rows are incomplete")
    cells: list[dict[str, Any]] = []
    for media_row in rows:
        sample_id = str(media_row.get("sample_id"))
        record = record_map.get(sample_id)
        if record is None:
            raise DiagnosticError(f"media row has unknown record: {sample_id}")
        row_cells = media_row.get("cells")
        if not isinstance(row_cells, Mapping):
            raise DiagnosticError(f"media cells are missing: {sample_id}")
        for video_arm, audio_arm in config.MAIN_CELL_SPECS:
            key = config.cell_key(video_arm, audio_arm)
            media = row_cells.get(key)
            if not isinstance(media, Mapping):
                raise DiagnosticError(f"media cell is missing: {sample_id}/{key}")
            audio = record["natural_audio"]["path"] if audio_arm == config.AUDIO_N else record["warp_audio"]["path"]
            cells.append({"cell_id": f"{sample_id}__{key}", "sample_id": sample_id, "video_arm": video_arm, "audio_arm": audio_arm, "repeat": False, "repeat_of": None, "media": str(media["output"]), "audio": str(audio), "protocol_sha256": str(protocol["_sha256"]), "record": record})
        for video_arm, audio_arm in config.REPEAT_CELL_SPECS:
            key = config.cell_key(video_arm, audio_arm)
            media = row_cells.get(key)
            if not isinstance(media, Mapping):
                raise DiagnosticError(f"repeat media source cell is missing: {sample_id}/{key}")
            audio = record["natural_audio"]["path"]
            cells.append({"cell_id": f"{sample_id}__{key}__repeat", "sample_id": sample_id, "video_arm": video_arm, "audio_arm": audio_arm, "repeat": True, "repeat_of": f"{sample_id}__{key}", "media": str(media["output"]), "audio": str(audio), "protocol_sha256": str(protocol["_sha256"]), "record": record})
    if len(cells) != config.EXPECTED_TOTAL_CELL_COUNT:
        raise DiagnosticError(f"score cell plan is incomplete: {len(cells)}/{config.EXPECTED_TOTAL_CELL_COUNT}")
    return cells


def run_scores(protocol: Mapping[str, Any], media_manifest: Mapping[str, Any], paths: config.RunPaths) -> dict[str, Any]:
    if media_manifest.get("status") != "complete" or media_manifest.get("cell_count") != config.EXPECTED_MAIN_CELL_COUNT:
        raise DiagnosticError("media manifest is incomplete")
    manifest_path = paths.scores / "manifest.json"
    if manifest_path.exists() and not manifest_path.is_file():
        raise DiagnosticError("score manifest marker is not a file")
    if manifest_path.is_file():
        prior = verify_self_hashed_json(manifest_path)
        if prior.get("status") != "complete" or prior.get("cell_count") != config.EXPECTED_TOTAL_CELL_COUNT:
            raise DiagnosticError("partial score manifest cannot be resumed")
    cells = _manifest_cell_specs(protocol, media_manifest)
    paths.scores.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, cell in enumerate(cells, 1):
        rows.append(_score_one(cell, protocol, paths))
        if index % 6 == 0 or index == len(cells):
            print(f"SCORE {index}/{len(cells)} {cell['sample_id']}", flush=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage_id": "score",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "protocol_sha256": str(protocol["_sha256"]),
        "media_manifest_sha256": file_sha256(paths.media / "manifest.json"),
        "status": "complete",
        "record_count": config.EXPECTED_RECORD_COUNT,
        "main_cell_count": sum(not bool(row["repeat"]) for row in rows),
        "expected_main_cell_count": config.EXPECTED_MAIN_CELL_COUNT,
        "repeat_cell_count": sum(bool(row["repeat"]) for row in rows),
        "expected_repeat_cell_count": config.EXPECTED_REPEAT_CELL_COUNT,
        "cell_count": len(rows),
        "expected_cell_count": config.EXPECTED_TOTAL_CELL_COUNT,
        "scores": rows,
    }
    write_self_hashed_json(manifest_path, manifest)
    return verify_self_hashed_json(manifest_path)
