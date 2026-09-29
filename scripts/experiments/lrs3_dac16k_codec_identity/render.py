from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import config
from .common import file_sha256, verify_self_hashed_json, write_self_hashed_json
from .protocol import ProtocolError


def ffprobe_streams(path: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        [config.FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        check=True,
    )
    streams = json.loads(result.stdout.decode("utf-8")).get("streams")
    if not isinstance(streams, list):
        raise ProtocolError(f"ffprobe returned no streams: {path}")
    return streams


def shared_face_geometry(video: Path) -> dict[str, Any]:
    streams = ffprobe_streams(video)
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    if len(video_streams) != 1:
        raise ProtocolError(f"expected one source video stream: {video}")
    stream = video_streams[0]
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise ProtocolError(f"invalid source video geometry: {video}")
    return {
        "mode": "constant_full_frame_fallback",
        "box_order": ["top", "bottom", "left", "right"],
        "box": [0, height, 0, width],
        "width": width,
        "height": height,
        "source_video_sha256": file_sha256(video),
    }


def _command(face: Path, audio: Path, output: Path, geometry: Mapping[str, Any]) -> list[str]:
    top, bottom, left, right = (str(int(value)) for value in geometry["box"])
    return [
        str(config.WAV2LIP_PYTHON),
        str(config.WAV2LIP_ROOT / "inference.py"),
        "--checkpoint_path", str(config.WAV2LIP_CHECKPOINT),
        "--face", str(face),
        "--audio", str(audio),
        "--outfile", str(output),
        "--box", top, bottom, left, right,
        "--face_det_batch_size", "16",
        "--wav2lip_batch_size", "16",
        "--nosmooth",
    ]


def render_driver(face: Path, audio: Path, output: Path, geometry: Mapping[str, Any], work_dir: Path, log_path: Path) -> dict[str, Any]:
    if not config.WAV2LIP_PYTHON.is_file() or not config.WAV2LIP_CHECKPOINT.is_file():
        raise FileNotFoundError("Wav2Lip environment or checkpoint is missing")
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "temp").mkdir(exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    command = _command(face, audio, temporary, geometry)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(work_dir), stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        raise ProtocolError(f"Wav2Lip render failed ({result.returncode}): {output}")
    temporary.replace(output)
    return {
        "output": str(output),
        "output_sha256": file_sha256(output),
        "face_sha256": file_sha256(face),
        "audio_sha256": file_sha256(audio),
        "geometry_sha256": hashlib.sha256(json.dumps(dict(geometry), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "checkpoint_sha256": file_sha256(config.WAV2LIP_CHECKPOINT),
        "command": command,
        "command_sha256": hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest(),
        "log": str(log_path),
    }


def run_stage03(cohort: Mapping[str, Any], audio_manifest: Mapping[str, Any], output_stage: Path) -> dict[str, Any]:
    if audio_manifest.get("status") != "complete":
        raise ProtocolError("Stage 01 is incomplete")
    rows: list[dict[str, Any]] = []
    for index, (record, audio_row) in enumerate(zip(cohort["records"], audio_manifest["rows"]), 1):
        sample_id = str(record["sample_id"])
        if sample_id != str(audio_row["sample_id"]):
            raise ProtocolError("cohort and audio order differs")
        face = Path(str(record["video"]))
        geometry = shared_face_geometry(face)
        arms: dict[str, Any] = {}
        for arm, audio_key in (("N", "natural_audio"), ("W", "wavlm_audio"), ("D", "dac")):
            audio = Path(str(record[audio_key])) if arm != "D" else Path(str(audio_row["dac"]["output_path"]))
            output = output_stage / "videos" / arm / f"{sample_id}.mp4"
            log = output_stage / "logs" / arm / f"{sample_id}.log"
            sidecar = output.with_suffix(".json")
            if output.is_file() and sidecar.is_file():
                prior = verify_self_hashed_json(sidecar)
                if prior.get("protocol_id") != config.PROTOCOL_ID or prior.get("audio_sha256") != file_sha256(audio) or prior.get("face_sha256") != file_sha256(face) or prior.get("output_sha256") != file_sha256(output):
                    raise ProtocolError(f"render sidecar identity changed: {sample_id}/{arm}")
                arms[arm] = prior
            elif output.is_file() and not sidecar.is_file():
                streams = ffprobe_streams(output)
                if len([stream for stream in streams if stream.get("codec_type") == "video"]) != 1:
                    raise ProtocolError(f"interrupted render is not a valid video: {sample_id}/{arm}")
                temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
                command = _command(face, audio, temporary, geometry)
                recovered = {
                    "schema_version": 1,
                    "stage_id": "03_videos",
                    "protocol_id": config.PROTOCOL_ID,
                    "sample_id": sample_id,
                    "arm": arm,
                    "output": str(output),
                    "output_sha256": file_sha256(output),
                    "face_sha256": file_sha256(face),
                    "audio_sha256": file_sha256(audio),
                    "geometry_sha256": hashlib.sha256(json.dumps(dict(geometry), sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    "checkpoint_sha256": file_sha256(config.WAV2LIP_CHECKPOINT),
                    "command": command,
                    "command_sha256": hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest(),
                    "log": str(log),
                    "recovered_after_interruption": True,
                }
                write_self_hashed_json(sidecar, recovered)
                arms[arm] = recovered
            elif output.exists() or sidecar.exists() or output.with_name(f".{output.stem}.partial{output.suffix}").exists():
                raise ProtocolError(f"partial render cannot be resumed: {sample_id}/{arm}")
            else:
                rendered = render_driver(face, audio, output, geometry, output_stage / "work" / arm / sample_id, log)
                rendered = {"schema_version": 1, "stage_id": "03_videos", "protocol_id": config.PROTOCOL_ID, "sample_id": sample_id, "arm": arm, **rendered}
                write_self_hashed_json(sidecar, rendered)
                arms[arm] = rendered
            if arms[arm]["geometry_sha256"] != hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
                raise ProtocolError(f"driver geometry differs: {sample_id}/{arm}")
        rows.append({"sample_id": sample_id, "source_group": str(record["source_group"]), "face": str(face), "face_sha256": record["video_sha256"], "geometry": geometry, "arms": arms})
        print(f"VIDEO {index}/{config.EXPECTED_RECORD_COUNT} {sample_id}", flush=True)
    result = {
        "schema_version": 1,
        "stage_id": "03_videos",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if len(rows) == config.EXPECTED_RECORD_COUNT else "blocked",
        "record_count": len(rows),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "driver_arms": list(config.DRIVER_ARMS),
        "rows": rows,
        "wav2lip_checkpoint_sha256": file_sha256(config.WAV2LIP_CHECKPOINT),
        "audio_manifest_sha256": file_sha256(config.STAGES["01_audio"] / "audio_manifest.json"),
    }
    write_self_hashed_json(output_stage / "videos_manifest.json", result)
    if result["status"] != "complete":
        raise ProtocolError("Stage 03 is incomplete")
    return result
