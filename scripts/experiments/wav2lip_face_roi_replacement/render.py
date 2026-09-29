from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import config
from .common import (
    ProtocolError,
    file_sha256,
    verify_self_hashed_json,
    write_or_verify,
    write_self_hashed_json,
)


def _run(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["NUMBA_DISABLE_JIT"] = "1"
    environment["NUMBA_CACHE_DIR"] = str(cwd / "numba_cache")
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(cwd), env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise ProtocolError(f"Wav2Lip generation failed: {log_path}")


_VIDEO_TO_AUDIO_ARM = {
    config.VIDEO_GN: config.AUDIO_N,
    config.VIDEO_GNR: config.AUDIO_N_REPEAT,
    config.VIDEO_GW: config.AUDIO_W,
    config.VIDEO_GB: config.AUDIO_N,
}


def _audio_paths(record: Mapping[str, Any], arms: Sequence[str]) -> dict[str, str]:
    audio_row = record.get("audio", {}).get("manifest_row")
    if not isinstance(audio_row, Mapping):
        raise ProtocolError(f"audio manifest row is missing: {record.get('sample_id')}")
    items = audio_row.get("arms")
    if not isinstance(items, list):
        raise ProtocolError(f"audio arm list is missing: {record.get('sample_id')}")
    result: dict[str, str] = {}
    for video_arm in arms:
        video_arm = str(video_arm)
        audio_arm = _VIDEO_TO_AUDIO_ARM.get(video_arm, video_arm)
        item = next((value for value in items if isinstance(value, Mapping) and str(value.get("arm")) == audio_arm), None)
        if not isinstance(item, Mapping):
            raise ProtocolError(f"audio arm is missing: {record.get('sample_id')}/{audio_arm} for video arm {video_arm}")
        path = Path(str(item.get("output", item.get("path", ""))))
        expected = str(item.get("output_sha256", item.get("container_sha256", "")))
        if not path.is_file() or not expected or file_sha256(path) != expected:
            raise ProtocolError(f"audio input hash changed: {record.get('sample_id')}/{audio_arm} for video arm {video_arm}")
        result[video_arm] = str(path.resolve())
    return result


def _roi(record: Mapping[str, Any], roi_by_id: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    sample_id = str(record["sample_id"])
    row = roi_by_id.get(sample_id)
    if not isinstance(row, Mapping):
        raise ProtocolError(f"ROI row is missing: {sample_id}")
    path = Path(str(row.get("boxes_path", "")))
    if not path.is_file() or file_sha256(path) != str(row.get("boxes_sha256")):
        raise ProtocolError(f"ROI cache hash changed: {sample_id}")
    if int(row.get("frame_count", -1)) != int(record["roi"]["frame_count"]):
        raise ProtocolError(f"ROI frame count changed: {sample_id}")
    return row


def _render_bundle(record: Mapping[str, Any], roi: Mapping[str, Any], stage: str, arms: Sequence[str], paths: config.RunPaths) -> dict[str, Mapping[str, Any]]:
    sample_id = str(record["sample_id"])
    expected = {arm: int(record["predicted_frame_counts"][arm]) for arm in arms}
    if len(set(expected.values())) != 1:
        raise ProtocolError(f"generated frame predictions differ: {sample_id}")
    output_paths = {arm: paths.videos / stage / arm / f"{sample_id}.mkv" for arm in arms}
    sidecars = {arm: output.with_suffix(".json") for arm, output in output_paths.items()}
    complete = all(output.is_file() and sidecars[arm].is_file() for arm, output in output_paths.items())
    if complete:
        rows: dict[str, Mapping[str, Any]] = {}
        for arm in arms:
            prior = verify_self_hashed_json(sidecars[arm])
            if prior.get("protocol_id") != config.PROTOCOL_ID or prior.get("stage") != stage or prior.get("sample_id") != sample_id or prior.get("arm") != arm or prior.get("output_sha256") != file_sha256(output_paths[arm]) or prior.get("boxes_sha256") != str(roi["boxes_sha256"]):
                raise ProtocolError(f"existing generated video identity changed: {sample_id}/{arm}")
            rows[arm] = prior
        return rows
    if any(path.exists() for path in (*output_paths.values(), *sidecars.values())):
        raise ProtocolError(f"partial generated video cannot be resumed: {sample_id}/{stage}")
    audio_paths = _audio_paths(record, arms)
    work_dir = paths.videos / "work" / stage / ("_".join(arms)) / sample_id
    work_dir.mkdir(parents=True, exist_ok=True)
    audio_plan = work_dir / "audio.json"
    output_plan = work_dir / "outputs.json"
    audio_plan.write_text(json.dumps(audio_paths, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_plan.write_text(json.dumps({arm: str(output_paths[arm].resolve()) for arm in arms}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    command = [str(config.WAV2LIP_PYTHON), str(Path(__file__).with_name("generation_worker.py")), "--face", str(Path(str(record["face_video"]["path"])).resolve()), "--boxes", str(Path(str(roi["boxes_path"])).resolve()), "--audio-json", str(audio_plan), "--outputs-json", str(output_plan), "--checkpoint", str(config.WAV2LIP_CHECKPOINT), "--ffmpeg", str(config.FFMPEG), "--batch-size", "4", "--expected-frame-count", str(next(iter(expected.values())))]
    log = paths.videos / "logs" / stage / f"{sample_id}__{'_'.join(arms)}.log"
    _run(command, work_dir, log)
    result_path = work_dir / "generation_result.json"
    result = verify_self_hashed_json(result_path)
    result_rows = result.get("rows")
    if not isinstance(result_rows, Mapping) or set(result_rows) != set(arms):
        raise ProtocolError(f"generation result is incomplete: {sample_id}/{stage}")
    rows: dict[str, Mapping[str, Any]] = {}
    for arm in arms:
        item = result_rows[arm]
        if not isinstance(item, Mapping) or int(item.get("frame_count", -1)) != expected[arm] or not output_paths[arm].is_file() or str(item.get("output_sha256")) != file_sha256(output_paths[arm]):
            raise ProtocolError(f"generated output violates frame/hash contract: {sample_id}/{arm}")
        row = {"schema_version": 1, "stage_id": "videos", "protocol_id": config.PROTOCOL_ID, "stage": stage, "sample_id": sample_id, "source_group": str(record["source_group"]), "arm": arm, "output": str(output_paths[arm].resolve()), "output_sha256": file_sha256(output_paths[arm]), "face": str(Path(str(record["face_video"]["path"])).resolve()), "face_sha256": str(record["face_video"]["sha256"]), "boxes": str(Path(str(roi["boxes_path"])).resolve()), "boxes_sha256": str(roi["boxes_sha256"]), "box_track_sha256": str(roi["box_track_sha256"]), "audio": audio_paths[arm], "audio_sha256": str(item.get("audio_sha256")), "frame_count": int(item["frame_count"]), "width": int(item["width"]), "height": int(item["height"]), "predicted_frame_count": expected[arm], "checkpoint_sha256": str(item.get("checkpoint_sha256")), "runtime": {"device": item.get("device"), "batch_size": item.get("batch_size"), "generation_result": str(result_path.resolve()), "generation_result_sha256": file_sha256(result_path), "command": command, "command_sha256": __import__("hashlib").sha256(json.dumps(command, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "log": str(log.resolve())}}
        write_self_hashed_json(sidecars[arm], row)
        rows[arm] = verify_self_hashed_json(sidecars[arm])
    return rows


def render_stage(protocol: Mapping[str, Any], paths: config.RunPaths, stage: str) -> dict[str, Any]:
    if stage not in ("control", "bridge"):
        raise ValueError(f"unknown render stage: {stage}")
    roi_manifest = verify_self_hashed_json(paths.roi_manifest)
    roi_by_id = {str(row.get("sample_id")): row for row in roi_manifest.get("rows", []) if isinstance(row, Mapping)}
    records = protocol.get("records")
    if not isinstance(records, list) or len(records) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("protocol records are incomplete")
    manifest_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        roi = _roi(record, roi_by_id)
        if stage == "control":
            first = _render_bundle(record, roi, stage, (config.VIDEO_GN, config.VIDEO_GW), paths)
            repeat = _render_bundle(record, roi, stage, (config.VIDEO_GNR,), paths)
            generated = {**first, **repeat}
        else:
            generated = _render_bundle(record, roi, stage, (config.VIDEO_GB,), paths)
        manifest_rows.append({"sample_id": str(record["sample_id"]), "source_group": str(record["source_group"]), "arms": generated})
        print(f"VIDEO {index}/{len(records)} {record['sample_id']} stage={stage}", flush=True)
    expected_count = config.EXPECTED_CONTROL_VIDEO_COUNT if stage == "control" else config.EXPECTED_BRIDGE_VIDEO_COUNT
    payload = {"schema_version": 1, "stage_id": "videos", "protocol_id": config.PROTOCOL_ID, "stage": stage, "status": "complete", "record_count": len(manifest_rows), "expected_record_count": config.EXPECTED_RECORD_COUNT, "video_count": sum(len(row["arms"]) for row in manifest_rows), "expected_video_count": expected_count, "rows": manifest_rows, "checkpoint_sha256": config.WAV2LIP_CHECKPOINT_SHA256, "roi_manifest_sha256": file_sha256(paths.roi_manifest)}
    manifest_path = paths.videos / stage / "manifest.json"
    return write_or_verify(manifest_path, payload)
