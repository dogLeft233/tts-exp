from __future__ import annotations

import json
import math
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import DiagnosticError, bytes_sha256, canonical_json_sha256, file_sha256, read_json, verify_self_hashed_json, write_self_hashed_json


def pairwise_distance_reference(visual: np.ndarray, audio: np.ndarray, vshift: int = config.VSHIFT) -> np.ndarray:
    """Numpy reference for SyncNet's official ``calc_pdist`` implementation."""
    visual_value = np.asarray(visual, dtype=np.float64)
    audio_value = np.asarray(audio, dtype=np.float64)
    if visual_value.ndim != 2 or audio_value.ndim != 2 or visual_value.shape[1] != audio_value.shape[1]:
        raise DiagnosticError("visual/audio features must be two-dimensional with equal width")
    if vshift < 0:
        raise DiagnosticError("vshift must be non-negative")
    padded = np.pad(audio_value, ((vshift, vshift), (0, 0)))
    rows = []
    for index in range(visual_value.shape[0]):
        # torch.nn.functional.pairwise_distance adds eps to every component
        # before taking the p=2 norm (rather than adding eps to the squared
        # norm).  Keep this reference exact enough to catch a future change
        # from the official SyncNet implementation.
        delta = padded[index : index + 2 * vshift + 1] - visual_value[index] + 1e-6
        rows.append(np.sqrt(np.sum(delta * delta, axis=1)))
    return np.asarray(rows, dtype=np.float64)


def parse_syncnet_metrics(text: str) -> list[dict[str, float | int]]:
    confidence = re.findall(r"Confidence:\s+([-+]?\d+(?:\.\d+)?)", text)
    distance = re.findall(r"Min dist:\s+([-+]?\d+(?:\.\d+)?)", text)
    offset = re.findall(r"AV offset:\s+(-?\d+)", text)
    if not (len(confidence) == len(distance) == len(offset)):
        return []
    result = [{"sync_c": float(c), "sync_d": float(d), "av_offset": int(o)} for c, d, o in zip(confidence, distance, offset, strict=True)]
    if any(not math.isfinite(float(item["sync_c"])) or not math.isfinite(float(item["sync_d"])) for item in result):
        raise DiagnosticError("SyncNet output contains non-finite metrics")
    return result


def reconstruct_global(matrix: np.ndarray) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 * config.VSHIFT + 1 or value.shape[0] < 1 or not np.isfinite(value).all():
        raise DiagnosticError(f"distance matrix must be finite with shape [window,31], got {value.shape}")
    curve = np.mean(value, axis=0)
    index = int(np.argmin(curve))
    minimum = float(curve[index])
    return {
        "curve": [float(item) for item in curve],
        "offsets": [config.VSHIFT - column for column in range(2 * config.VSHIFT + 1)],
        "min_index": index,
        "offset": config.VSHIFT - index,
        "sync_d": minimum,
        "sync_c": float(np.median(curve) - minimum),
    }


def _run(command: list[str], cwd: Path, log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr
    log_path.write_text(output, encoding="utf-8")
    if result.returncode != 0:
        raise DiagnosticError(f"command failed ({result.returncode}): {log_path}")
    return output


def _run_worker(media: Path, cell_dir: Path, reference: str) -> tuple[dict[str, Any], np.ndarray, Path, Path]:
    matrix = cell_dir / "distance.npy"
    result_path = cell_dir / "worker.json"
    worker_log = cell_dir / "forward.log"
    tmp_dir = cell_dir / "fresh_tmp"
    command = [
        str(config.SYNCNET_PYTHON),
        str(config.SYNCNET_WORKER),
        "--media", str(media),
        "--model", str(config.SYNCNET_MODEL),
        "--tmp-dir", str(tmp_dir),
        "--reference", reference,
        "--matrix", str(matrix),
        "--result", str(result_path),
        "--batch-size", "20",
        "--vshift", str(config.VSHIFT),
    ]
    _run(command, config.SYNCNET_ROOT, worker_log)
    if not matrix.is_file() or not result_path.is_file():
        raise DiagnosticError(f"official SyncNet worker did not produce artifacts: {reference}")
    worker = read_json(result_path)
    value = np.load(matrix, allow_pickle=False)
    return worker, np.asarray(value, dtype=np.float64), worker_log, result_path


def _run_pipeline(media: Path, cell_dir: Path, reference: str) -> tuple[Path, list[dict[str, float | int]], int]:
    data_dir = cell_dir / "official_pipeline"
    pipeline_log = cell_dir / "pipeline.log"
    score_log = cell_dir / "official_score.log"
    pipeline_command = [str(config.SYNCNET_PYTHON), str(config.SYNCNET_PIPELINE), "--videofile", str(media), "--reference", reference, "--data_dir", str(data_dir), "--min_track", "50", "--overwrite"]
    pipeline_output = _run(pipeline_command, config.SYNCNET_ROOT, pipeline_log)
    score_command = [str(config.SYNCNET_PYTHON), str(config.SYNCNET_SCORE), "--videofile", str(media), "--reference", reference, "--data_dir", str(data_dir), "--initial_model", str(config.SYNCNET_MODEL)]
    score_output = _run(score_command, config.SYNCNET_ROOT, score_log)
    crops = sorted((data_dir / "pycrop" / reference).glob("0*.avi"))
    metrics = parse_syncnet_metrics(score_output)
    if not crops or not metrics:
        raise DiagnosticError(f"official pipeline produced no scoreable tracks: {reference}")
    if len(crops) != len(metrics):
        raise DiagnosticError(f"official pipeline track/metric count differs: {reference} {len(crops)} != {len(metrics)}")
    return crops[0], metrics, len(crops)


def _cell_score(media_row: Mapping[str, Any], paths: config.RunPaths, protocol: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = str(media_row["sample_id"])
    family = str(media_row["family"])
    cell = str(media_row["cell"])
    cell_key = config.cell_id(sample_id, family, cell)
    cell_dir = paths.scores / cell_key
    cell_dir.mkdir(parents=True, exist_ok=True)
    media = Path(str(media_row["media"]))
    if not media.is_file() or file_sha256(media) != str(media_row["media_sha256"]):
        raise DiagnosticError(f"media identity changed before scoring: {cell_key}")
    pipeline_metrics: list[dict[str, float | int]] = []
    pipeline_track_count: int | None = None
    scored_media = media
    if family == "A":
        scored_media, pipeline_metrics, pipeline_track_count = _run_pipeline(media, cell_dir, cell_key)
        worker, matrix, worker_log, worker_result_path = _run_worker(scored_media, cell_dir, cell_key)
        official = pipeline_metrics[0]
    else:
        worker, matrix, worker_log, worker_result_path = _run_worker(media, cell_dir, cell_key)
        official = None
    reconstructed = reconstruct_global(matrix)
    worker_metrics = {
        "sync_c": float(worker.get("confidence")),
        "sync_d": float(reconstructed["sync_d"]),
        "av_offset": int(worker.get("offset")),
    }
    parity = {
        "worker_matrix_reconstruction_c_abs_error": abs(worker_metrics["sync_c"] - float(reconstructed["sync_c"])),
        "worker_matrix_reconstruction_d_abs_error": abs(float(worker.get("min_dist", reconstructed["sync_d"])) - float(reconstructed["sync_d"])),
        "worker_matrix_reconstruction_offset_equal": int(worker_metrics["av_offset"]) == int(reconstructed["offset"]),
        "official_forward": "SyncNetInstance.evaluate -> SyncNetInstance.calc_pdist",
    }
    if parity["worker_matrix_reconstruction_c_abs_error"] > 0.001 or not parity["worker_matrix_reconstruction_offset_equal"]:
        raise DiagnosticError(f"official worker result differs from distance matrix: {cell_key}")
    if official is not None:
        parity.update(
            {
                "pipeline_first_track_c_abs_error": abs(float(official["sync_c"]) - float(reconstructed["sync_c"])),
                "pipeline_first_track_d_abs_error": abs(float(official["sync_d"]) - float(reconstructed["sync_d"])),
                "pipeline_first_track_offset_equal": int(official["av_offset"]) == int(reconstructed["offset"]),
            }
        )
    row = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_sha256": protocol.get("artifact_sha256"),
        "sample_id": sample_id,
        "family": family,
        "cell": cell,
        "cell_id": cell_key,
        "media": str(media.resolve()),
        "media_sha256": file_sha256(media),
        "scored_media": str(scored_media.resolve()),
        "scored_media_sha256": file_sha256(scored_media),
        "matrix": str((cell_dir / "distance.npy").resolve()),
        "matrix_sha256": file_sha256(cell_dir / "distance.npy"),
        "matrix_shape": [int(value) for value in matrix.shape],
        "window_frame_starts": list(range(int(matrix.shape[0]))),
        "offsets": reconstructed["offsets"],
        "official_worker": str(worker_result_path.resolve()),
        "official_worker_sha256": file_sha256(worker_result_path),
        "forward_log": str(worker_log.resolve()),
        "pipeline_log": str((cell_dir / "pipeline.log").resolve()) if family == "A" else None,
        "official_score_log": str((cell_dir / "official_score.log").resolve()) if family == "A" else None,
        "pipeline_track_count": pipeline_track_count,
        "pipeline_metrics": pipeline_metrics,
        "official": official,
        "reconstructed": reconstructed,
        "parity": parity,
        "runtime": {
            "syncnet_python": str(config.SYNCNET_PYTHON.resolve()),
            "syncnet_python_sha256": file_sha256(config.SYNCNET_PYTHON),
            "syncnet_model": str(config.SYNCNET_MODEL.resolve()),
            "syncnet_model_sha256": file_sha256(config.SYNCNET_MODEL),
            "worker_sha256": file_sha256(config.SYNCNET_WORKER),
        },
    }
    write_self_hashed_json(cell_dir / "score.json", row)
    return row


def _score_failure(media_row: Mapping[str, Any], error: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "blocked",
        "protocol_id": config.PROTOCOL_ID,
        "sample_id": str(media_row.get("sample_id", "")),
        "family": str(media_row.get("family", "")),
        "cell": str(media_row.get("cell", "")),
        "cell_id": config.cell_id(str(media_row.get("sample_id", "")), str(media_row.get("family", "")), str(media_row.get("cell", ""))),
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _score_batch(paths: config.RunPaths, protocol: Mapping[str, Any], media_rows: list[Mapping[str, Any]], display_offset: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, media_row in enumerate(media_rows, display_offset + 1):
        try:
            if media_row.get("status") != "ready":
                raise DiagnosticError(f"media cell is not ready: {media_row.get('cell_id', media_row.get('cell'))}")
            scored = _cell_score(media_row, paths, protocol)
            rows.append(scored)
            print(f"SCORE {index}/{config.EXPECTED_NEW_CELL_COUNT} {media_row['sample_id']} {media_row['family']}/{media_row['cell']}", flush=True)
        except Exception as exc:
            failure = _score_failure(media_row, exc)
            failures.append(failure)
            rows.append(failure)
    return rows, failures


def score_all(
    paths: config.RunPaths,
    protocol: Mapping[str, Any],
    media_manifest: Mapping[str, Any],
    history: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score A/B first, then materialize C from A/G_N's selected crop."""
    initial_media_rows = list(media_manifest.get("cells", []))
    expected = config.EXPECTED_NEW_CELL_COUNT
    score_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    first_rows = [row for row in initial_media_rows if str(row.get("family")) in {"A", "B"}]
    first_scored, first_failures = _score_batch(paths, protocol, first_rows, 0)
    failures.extend(first_failures)
    for row in first_scored:
        score_by_key[(str(row.get("sample_id")), str(row.get("family")), str(row.get("cell")))] = row

    if history is not None and any(str(row.get("family")) == "C" for row in initial_media_rows):
        from .prepare import materialize_c_from_a_pipeline

        media_manifest = materialize_c_from_a_pipeline(paths, history, media_manifest, score_by_key)
    media_rows = list(media_manifest.get("cells", []))
    c_rows = [row for row in media_rows if str(row.get("family")) == "C"]
    c_scored, c_failures = _score_batch(paths, protocol, c_rows, len(first_rows))
    failures.extend(c_failures)
    for row in c_scored:
        score_by_key[(str(row.get("sample_id")), str(row.get("family")), str(row.get("cell")))] = row

    rows = [
        score_by_key.get(
            (str(media_row.get("sample_id")), str(media_row.get("family")), str(media_row.get("cell"))),
            _score_failure(media_row, DiagnosticError("score row was not produced")),
        )
        for media_row in media_rows
    ]
    complete = len(media_rows) == expected and not failures and all(row.get("status") == "complete" for row in rows)
    manifest = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if complete else "blocked",
        "expected_cell_count": expected,
        "cell_count": len(rows),
        "complete_count": sum(row.get("status") == "complete" for row in rows),
        "failure_count": len(failures),
        "scores": rows,
        "failures": failures,
        "fresh_cache_root": str(paths.scores.resolve()),
        "score_based_retry": False,
        "c_materialized_from_a_gn": history is not None,
    }
    write_self_hashed_json(paths.scores / "manifest.json", manifest)
    return manifest
