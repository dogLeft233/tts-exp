from __future__ import annotations

import subprocess
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import config
from .common import (
    canonical_json_sha256,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .protocol import ProtocolError


def _run(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise ProtocolError(f"command failed ({result.returncode}): {log_path}")


def ffprobe_streams(path: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        [str(config.FFPROBE), "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        check=True,
    )
    payload = __import__("json").loads(result.stdout.decode("utf-8"))
    streams = payload.get("streams")
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


def _binding_hashes() -> dict[str, str]:
    paths = {
        "wav2lip_python": config.WAV2LIP_PYTHON,
        "wav2lip_checkpoint": config.WAV2LIP_CHECKPOINT,
        "wav2lip_inference": config.WAV2LIP_ROOT / "inference.py",
    }
    result = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = file_sha256(path)
    if result["wav2lip_checkpoint"] != config.WAV2LIP_CHECKPOINT_SHA256 or result["wav2lip_python"] != config.WAV2LIP_PYTHON_SHA256:
        raise ProtocolError("Wav2Lip runtime binding differs from registered binding")
    return result


def render_driver(face: Path, audio: Path, output: Path, geometry: Mapping[str, Any], work_dir: Path, log_path: Path) -> dict[str, Any]:
    binding = _binding_hashes()
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "temp").mkdir(exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    command = _command(face, audio, temporary, geometry)
    _run(command, work_dir, log_path)
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise ProtocolError(f"Wav2Lip render produced no output: {output}")
    temporary.replace(output)
    return {
        "output": str(output),
        "output_sha256": file_sha256(output),
        "face_sha256": file_sha256(face),
        "audio_sha256": file_sha256(audio),
        "geometry_sha256": canonical_json_sha256(dict(geometry)),
        "checkpoint_sha256": binding["wav2lip_checkpoint"],
        "command": command,
        "command_sha256": canonical_json_sha256(command),
        "runtime_bindings": binding,
        "log": str(log_path),
    }


def _audio_by_id(audio_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if audio_manifest.get("status") != "complete":
        raise ProtocolError("Stage 01 audio manifest is incomplete")
    result: dict[str, Mapping[str, Any]] = {}
    for row in audio_manifest.get("rows", []):
        sample_id = str(row.get("sample_id", ""))
        if sample_id in result:
            raise ProtocolError(f"duplicate audio row: {sample_id}")
        result[sample_id] = row
    if len(result) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("audio manifest record count is incomplete")
    return result


def _render_one(record: Mapping[str, Any], arm_row: Mapping[str, Any], arm: str, output_stage: Path, geometry: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    face = Path(str(record["face_video"]["path"]))
    audio = Path(str(arm_row["output"]))
    output = output_stage / "videos" / arm / f"{sample_id}.mp4"
    sidecar = output.with_suffix(".json")
    expected_geometry_sha256 = canonical_json_sha256(dict(geometry))
    if output.is_file() and sidecar.is_file():
        prior = verify_self_hashed_json(sidecar)
        if (
            prior.get("protocol_id") != config.PROTOCOL_ID
            or prior.get("sample_id") != sample_id
            or prior.get("arm") != arm
            or prior.get("face_sha256") != file_sha256(face)
            or prior.get("audio_sha256") != file_sha256(audio)
            or prior.get("geometry_sha256") != expected_geometry_sha256
            or prior.get("output_sha256") != file_sha256(output)
            or prior.get("checkpoint_sha256") != config.WAV2LIP_CHECKPOINT_SHA256
        ):
            raise ProtocolError(f"existing render sidecar identity changed: {sample_id}/{arm}")
        return prior
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    if output.exists() or sidecar.exists() or temporary.exists():
        raise ProtocolError(f"partial render cannot be resumed: {sample_id}/{arm}")
    rendered = render_driver(
        face,
        audio,
        output,
        geometry,
        output_stage / "work" / arm / sample_id,
        output_stage / "logs" / arm / f"{sample_id}.log",
    )
    row = {
        "schema_version": 1,
        "stage_id": "02_videos",
        "protocol_id": config.PROTOCOL_ID,
        "sample_id": sample_id,
        "source_group": str(record["source_group"]),
        "arm": arm,
        **rendered,
    }
    write_self_hashed_json(sidecar, row)
    return row


def run_stage02(cohort: Mapping[str, Any], audio_manifest: Mapping[str, Any], output_stage: Path) -> dict[str, Any]:
    audio_by_id = _audio_by_id(audio_manifest)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(cohort.get("records", []), 1):
        sample_id = str(record["sample_id"])
        audio_row = audio_by_id.get(sample_id)
        if audio_row is None:
            raise ProtocolError(f"missing audio row: {sample_id}")
        arm_rows = {str(row["arm"]): row for row in audio_row.get("arms", [])}
        if tuple(arm_rows) != config.ARMS:
            raise ProtocolError(f"audio arm order is incomplete: {sample_id}")
        face = Path(str(record["face_video"]["path"]))
        geometry = shared_face_geometry(face)
        arms: dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                arm: executor.submit(_render_one, record, arm_rows[arm], arm, output_stage, geometry)
                for arm in config.ARMS
            }
            for arm in config.ARMS:
                try:
                    arms[arm] = futures[arm].result()
                except (OSError, ProtocolError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
                    failures.append({"sample_id": sample_id, "arm": arm, "error_type": type(exc).__name__, "error": str(exc)})
        if len(arms) == len(config.ARMS):
            geometry_hash = canonical_json_sha256(geometry)
            if any(row["geometry_sha256"] != geometry_hash for row in arms.values()):
                raise ProtocolError(f"shared geometry changed across arms: {sample_id}")
            rows.append({"sample_id": sample_id, "source_group": str(record["source_group"]), "face": str(face), "face_sha256": file_sha256(face), "geometry": geometry, "arms": arms})
        print(f"VIDEO {index}/{config.EXPECTED_RECORD_COUNT} {sample_id}", flush=True)
    video_count = sum(len(row["arms"]) for row in rows)
    complete = not failures and len(rows) == config.EXPECTED_RECORD_COUNT and video_count == config.EXPECTED_VIDEO_COUNT
    result = {
        "schema_version": 1,
        "stage_id": "02_videos",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if complete else "blocked",
        "record_count": len(rows),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "video_count": video_count,
        "expected_video_count": config.EXPECTED_VIDEO_COUNT,
        "arms": list(config.ARMS),
        "rows": rows,
        "failures": failures,
        "wav2lip_checkpoint_sha256": config.WAV2LIP_CHECKPOINT_SHA256,
        "audio_manifest_sha256": file_sha256(config.STAGES["01_audio"] / "audio_manifest.json"),
    }
    write_self_hashed_json(output_stage / "videos_manifest.json", result)
    write_self_hashed_json(output_stage / "failures.json", {"schema_version": 1, "stage_id": "02_videos", "protocol_id": config.PROTOCOL_ID, "status": result["status"], "failures": failures})
    if not complete:
        raise ProtocolError(f"video render incomplete: {video_count}/{config.EXPECTED_VIDEO_COUNT}")
    return result
