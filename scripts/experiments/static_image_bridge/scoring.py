from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    ProtocolError,
    file_sha256,
    run_logged,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def _audio_path(audio_row: Mapping[str, Any], arm: str) -> Path:
    if arm == "ND":
        return Path(str(audio_row["controls"]["ND"]["path"]))
    return Path(str(audio_row["arms"][arm]["path"]))


def _cell_sidecar(paths: config.RunPaths, sample_id: str, video_arm: str, audio_arm: str) -> Path:
    return paths.matrices / sample_id / video_arm / f"{audio_arm}.json"


def _load_existing(path: Path) -> dict[str, Any]:
    row = verify_self_hashed_json(path)
    matrix = Path(str(row.get("matrix_path", "")))
    if not matrix.is_file() or file_sha256(matrix) != str(row.get("matrix_sha256")):
        raise ProtocolError(f"score matrix changed: {path}")
    if row.get("status") != "complete":
        raise ProtocolError(f"score cell is not complete: {path}")
    return row


def _worker_result_path(output_dir: Path, video_arm: str, audio_arm: str) -> Path:
    return output_dir / f"{video_arm}__{audio_arm}__worker.json"


def _run_video_score(
    paths: config.RunPaths,
    sample: Mapping[str, Any],
    video_row: Mapping[str, Any],
    audio_row: Mapping[str, Any],
    audio_arms: list[str],
) -> list[dict[str, Any]]:
    sid = str(sample["sample_id"])
    video_arm = str(video_row["video_arm"])
    work = paths.matrices / sid / video_arm
    work.mkdir(parents=True, exist_ok=True)
    score_box_path = paths.support / sid / "score_box.json"
    score_box_path.parent.mkdir(parents=True, exist_ok=True)
    score_box_path.write_text(json.dumps(sample["static_reference"]["score_box"], indent=2) + "\n", encoding="utf-8")
    audio_map_path = paths.logs_dir / "score_requests" / f"{sid}__{video_arm}.json"
    audio_map_path.parent.mkdir(parents=True, exist_ok=True)
    audio_map = {arm: str(_audio_path(audio_row, arm).resolve()) for arm in audio_arms}
    audio_map_path.write_text(json.dumps(audio_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log = paths.logs_dir / "syncnet" / f"{sid}__{video_arm}.log"
    command = [
        str(config.SYNCNET_PYTHON), str(Path(__file__).with_name("score_worker.py")),
        "--video", str(Path(str(video_row["output"])).resolve()), "--video-arm", video_arm, "--sample-id", sid,
        "--score-box", str(score_box_path), "--audio-json", str(audio_map_path), "--model", str(config.SYNCNET_MODEL),
        "--output-dir", str(work), "--vshift", str(config.SYNCNET_VSHIFT), "--batch-size", str(config.SYNCNET_BATCH_SIZE),
    ]
    run_logged(command, config.REPO, log, env={**dict(os.environ), "CUDA_VISIBLE_DEVICES": "0"})
    rows: list[dict[str, Any]] = []
    for audio_arm in audio_arms:
        worker_path = _worker_result_path(work, video_arm, audio_arm)
        if not worker_path.is_file():
            raise ProtocolError(f"SyncNet worker did not produce a result: {worker_path}")
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
        matrix_path = Path(str(worker.get("matrix", ""))).resolve()
        if not matrix_path.is_file() or file_sha256(matrix_path) != str(worker.get("matrix_sha256")):
            raise ProtocolError(f"SyncNet matrix binding is invalid: {worker_path}")
        matrix = np.asarray(np.load(matrix_path, allow_pickle=False), dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != 2 * config.SYNCNET_VSHIFT + 1:
            raise ProtocolError(f"SyncNet matrix shape is invalid: {matrix_path}")
        audio_path = _audio_path(audio_row, audio_arm)
        row = {
            "schema_version": 1,
            "status": "complete",
            "protocol_id": config.PROTOCOL_ID,
            "sample_id": sid,
            "source_group": str(sample["source_group"]),
            "video_arm": video_arm,
            "audio_arm": audio_arm,
            "cell": config.cell_key(video_arm, audio_arm),
            "video_path": str(Path(str(video_row["output"])).resolve()),
            "video_sha256": str(video_row["output_sha256"]),
            "audio_path": str(audio_path.resolve()),
            "audio_sha256": file_sha256(audio_path),
            "audio_pcm_sha256": str(audio_row["controls"]["ND"]["pcm_sha256"] if audio_arm == "ND" else audio_row["arms"][audio_arm]["pcm_sha256"]),
            "fixed_score_box": dict(sample["static_reference"]["score_box"]),
            "score_crop_png_sha256": str(sample["static_reference"]["score_crop"]["container_sha256"]),
            "frontend": "official SyncNet V2 model plus python_speech_features MFCC; fixed crop supplied by protocol",
            "device": worker.get("device"),
            "model_sha256": worker.get("model_sha256"),
            "vshift": worker.get("vshift"),
            "matrix_path": str(matrix_path),
            "matrix_sha256": str(worker["matrix_sha256"]),
            "matrix_shape": list(worker["matrix_shape"]),
            "finite_mask": worker["finite_mask"],
            "visual_path": worker.get("visual"),
            "visual_sha256": worker.get("visual_sha256"),
            "visual_count": worker.get("visual_count"),
            "audio_feature_count": worker.get("audio_feature_count"),
            "audio_sample_count": worker.get("audio_sample_count"),
            "worker_result": str(worker_path.resolve()),
            "worker_result_sha256": file_sha256(worker_path),
            "score_log": str(log),
        }
        sidecar = _cell_sidecar(paths, sid, video_arm, audio_arm)
        write_self_hashed_json(sidecar, row)
        rows.append(verify_self_hashed_json(sidecar))
    return rows


def _expected_cells(stage: str) -> tuple[tuple[str, str], ...]:
    if stage == "A":
        return config.STAGE_A_CELLS
    if stage == "B":
        return config.STAGE_B_CELLS
    raise ValueError(stage)


def score_stage(paths: config.RunPaths, inputs: Mapping[str, Any], audio_manifest: Mapping[str, Any], video_manifest: Mapping[str, Any], stage: str) -> dict[str, Any]:
    expected = _expected_cells(stage)
    video_by_key = {(str(row["sample_id"]), str(row["video_arm"])): row for row in video_manifest.get("rows", [])}
    audio_by_id = {str(row["sample_id"]): row for row in audio_manifest.get("rows", [])}
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    if paths.scores_manifest.is_file():
        previous = verify_self_hashed_json(paths.scores_manifest)
        for row in previous.get("rows", []):
            key = (str(row.get("sample_id")), str(row.get("video_arm")), str(row.get("audio_arm")))
            if key in rows:
                raise ProtocolError(f"duplicate score manifest row: {key}")
            rows[key] = dict(row)

    for sample in inputs["records"]:
        sid = str(sample["sample_id"])
        audio_row = audio_by_id.get(sid)
        if audio_row is None:
            raise ProtocolError(f"audio row missing for scoring: {sid}")
        by_video: defaultdict[str, list[str]] = defaultdict(list)
        for video_arm, audio_arm in expected:
            key = (sid, video_arm, audio_arm)
            sidecar = _cell_sidecar(paths, sid, video_arm, audio_arm)
            if sidecar.is_file():
                rows[key] = _load_existing(sidecar)
            elif key in rows:
                raise ProtocolError(f"score manifest references missing cell sidecar: {key}")
            else:
                by_video[video_arm].append(audio_arm)
        for video_arm, audio_arms in by_video.items():
            video_row = video_by_key.get((sid, video_arm))
            if video_row is None:
                raise ProtocolError(f"video arm missing for score cell: {sid}/{video_arm}")
            for row in _run_video_score(paths, sample, video_row, audio_row, audio_arms):
                rows[(sid, str(row["video_arm"]), str(row["audio_arm"]))] = row
        print(f"SCORE {sid} stage={stage}", flush=True)

    required_keys = {(str(item["sample_id"]), video, audio) for item in inputs["records"] for video, audio in expected}
    # A B-stage manifest intentionally contains the already completed A-stage
    # cells as well.  Requiring exact equality here makes a perfectly complete
    # cumulative manifest fail with ``missing=[]`` because it has valid extras.
    # Require the current stage's cells, while still rejecting cells outside
    # the two frozen stage contracts.
    missing = sorted(required_keys - set(rows))
    allowed_keys = {
        (str(item["sample_id"]), video, audio)
        for item in inputs["records"]
        for video, audio in (set(config.STAGE_A_CELLS) | set(config.STAGE_B_CELLS))
    }
    unexpected = sorted(set(rows) - allowed_keys)
    if missing:
        raise ProtocolError(f"score matrix incomplete: missing {missing[:5]}")
    if unexpected:
        raise ProtocolError(f"score manifest contains unexpected cells: {unexpected[:5]}")
    ordered = [rows[key] for key in sorted(rows, key=lambda item: (item[0], item[1], item[2]))]
    payload = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "record_count": config.EXPECTED_RECORD_COUNT,
        "stage": stage,
        "expected_cells": [[video, audio] for video, audio in expected],
        "cell_count": len(ordered),
        "rows": ordered,
        "fixed_crop": True,
        "common_w_required": True,
        "score_matrix_semantics": "d(t,k)=||v(t)-a(t+k)||_2; invalid boundary cells are NaN and excluded by frozen support",
    }
    write_self_hashed_json(paths.scores_manifest, payload)
    return verify_self_hashed_json(paths.scores_manifest)


def load_score_rows(paths: config.RunPaths) -> dict[tuple[str, str, str], dict[str, Any]]:
    manifest = verify_self_hashed_json(paths.scores_manifest)
    rows = {}
    for row in manifest.get("rows", []):
        key = (str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"]))
        if key in rows:
            raise ProtocolError(f"duplicate score row: {key}")
        rows[key] = row
    return rows
