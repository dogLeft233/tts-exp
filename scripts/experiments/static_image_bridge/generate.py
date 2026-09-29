from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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


def _video_sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def _video_row(sample_id: str, video_arm: str, static: Mapping[str, Any], audio_row: Mapping[str, Any], output: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "complete",
        "sample_id": sample_id,
        "video_arm": video_arm,
        "input_mode": "one_png_only",
        "image_path": str(static["image"]["path"]),
        "image_container_sha256": str(static["image"]["container_sha256"]),
        "image_rgb_pixel_sha256": str(static["image"]["rgb_pixel_sha256"]),
        "source_frame_index": 0,
        "source_frame_indices": list(result["source_frame_indices"]),
        "generation_box_xyxy": list(static["generation_box_xyxy"]),
        "score_box": dict(static["score_box"]),
        "audio_arm": video_arm,
        "audio_path": str(_audio_path(audio_row, video_arm)),
        "audio_container_sha256": str(audio_row["arms"][video_arm]["container_sha256"]),
        "audio_pcm_sha256": str(audio_row["arms"][video_arm]["pcm_sha256"]),
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "output_codec": result.get("output_codec"),
        "fps": result.get("fps"),
        "frame_count": result.get("frames_rendered"),
        "checkpoint_sha256": config.WAV2LIP_CHECKPOINT_SHA256,
        "device": result.get("device"),
        "seed": result.get("seed"),
        "worker_result": str(_video_sidecar(output).resolve()),
    }


def render_one(paths: config.RunPaths, sample: Mapping[str, Any], audio_row: Mapping[str, Any], video_arm: str) -> dict[str, Any]:
    sid = str(sample["sample_id"])
    static = sample["static_reference"]
    output = paths.video_dir / video_arm / f"{sid}.mkv"
    sidecar = _video_sidecar(output)
    if output.is_file() and sidecar.is_file():
        row = verify_self_hashed_json(sidecar)
        if row.get("sample_id") != sid or row.get("video_arm") != video_arm or file_sha256(output) != row.get("output_sha256"):
            raise ProtocolError(f"resumed video identity changed: {sid}/{video_arm}")
        if row.get("source_frame_indices") != [0] * int(row.get("frame_count", 0)):
            raise ProtocolError(f"resumed video used a dynamic source frame: {sid}/{video_arm}")
        return row
    if output.exists() or sidecar.exists():
        raise ProtocolError(f"partial video cell cannot be resumed: {sid}/{video_arm}")
    audio_path = _audio_path(audio_row, video_arm)
    work = paths.video_dir / "work" / sid / video_arm
    work.mkdir(parents=True, exist_ok=True)
    worker_result = work / "worker_result.json"
    log = paths.logs_dir / "wav2lip" / f"{sid}__{video_arm}.log"
    command = [
        str(config.WAV2LIP_PYTHON), str(Path(__file__).with_name("render_worker.py")),
        "--image", str(static["image"]["path"]), "--image-rgb-sha256", str(static["image"]["rgb_pixel_sha256"]),
        "--audio", str(audio_path), "--box", *[str(value) for value in static["generation_box_xyxy"]],
        "--checkpoint", str(config.WAV2LIP_CHECKPOINT), "--ffmpeg", str(config.FFMPEG),
        "--outfile", str(output), "--result", str(worker_result), "--batch-size", str(config.WAV2LIP_BATCH_SIZE), "--seed", str(config.SEED),
    ]
    run_logged(command, config.REPO, log, env={**dict(os.environ), "CUDA_VISIBLE_DEVICES": "0", "PYTHONHASHSEED": str(config.SEED)})
    result = json.loads(worker_result.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("device") != "cuda" or not output.is_file():
        raise ProtocolError(f"Wav2Lip render did not produce a CUDA result: {sid}/{video_arm}")
    if result.get("source_frame_indices") != [0] * int(result.get("frames_rendered", 0)):
        raise ProtocolError(f"Wav2Lip worker used a nonzero source frame: {sid}/{video_arm}")
    row = _video_row(sid, video_arm, static, audio_row, output, result)
    row["command"] = command
    row["log"] = str(log)
    row["worker_result_sha256"] = file_sha256(worker_result)
    write_self_hashed_json(sidecar, row)
    return verify_self_hashed_json(sidecar)


def render_stage(paths: config.RunPaths, inputs: Mapping[str, Any], audio_manifest: Mapping[str, Any], arms: tuple[str, ...]) -> dict[str, Any]:
    rows_by_id = {str(row["sample_id"]): row for row in audio_manifest.get("rows", [])}
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if paths.video_manifest.is_file():
        previous = verify_self_hashed_json(paths.video_manifest)
        for row in previous.get("rows", []):
            key = (str(row.get("sample_id")), str(row.get("video_arm")))
            if key in existing:
                raise ProtocolError(f"duplicate video manifest row: {key}")
            existing[key] = dict(row)
    completed = 0
    for sample in inputs["records"]:
        sid = str(sample["sample_id"])
        audio_row = rows_by_id.get(sid)
        if audio_row is None:
            raise ProtocolError(f"audio row missing for render: {sid}")
        for arm in arms:
            row = render_one(paths, sample, audio_row, arm)
            existing[(sid, arm)] = row
            completed += 1
            print(f"VIDEO {completed}/{len(inputs['records']) * len(arms)} {sid} {arm}", flush=True)
    rows = [existing[key] for key in sorted(existing, key=lambda item: (item[0], item[1]))]
    arms_seen = sorted({str(row["video_arm"]) for row in rows})
    payload = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "record_count": config.EXPECTED_RECORD_COUNT,
        "video_count": len(rows),
        "arms": arms_seen,
        "rows": rows,
        "audio_manifest_sha256": file_sha256(paths.audio_manifest),
    }
    write_self_hashed_json(paths.video_manifest, payload)
    return verify_self_hashed_json(paths.video_manifest)
