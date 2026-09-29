from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import config
from .common import (
    CalibrationError,
    canonical_json_sha256,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def runtime_environment(work_dir: Path) -> dict[str, str]:
    return {
        **config.WAV2LIP_ENV,
        "NUMBA_CACHE_DIR": str((work_dir / "numba_cache").resolve()),
    }


def _run(command: list[str], cwd: Path, log_path: Path, environment_overrides: Mapping[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(environment_overrides)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(cwd), env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise CalibrationError(f"command failed ({result.returncode}): {log_path}")


def ffprobe_streams(path: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        [str(config.FFPROBE), "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        check=True,
    )
    streams = json.loads(result.stdout.decode("utf-8")).get("streams")
    if not isinstance(streams, list):
        raise CalibrationError(f"ffprobe returned no stream list: {path}")
    return streams


def validate_geometry(geometry: Mapping[str, Any], face: Path) -> dict[str, Any]:
    streams = ffprobe_streams(face)
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    if len(videos) != 1:
        raise CalibrationError(f"face video does not have one video stream: {face}")
    stream = videos[0]
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise CalibrationError(f"face video geometry is invalid: {face}")
    expected_box = [0, height, 0, width]
    if list(geometry.get("box", [])) != expected_box or int(geometry.get("width", -1)) != width or int(geometry.get("height", -1)) != height:
        raise CalibrationError(f"historical face geometry changed: {face}")
    if geometry.get("source_video_sha256") != file_sha256(face):
        raise CalibrationError(f"historical face hash changed: {face}")
    return {
        "mode": str(geometry.get("mode", "")),
        "box": expected_box,
        "width": width,
        "height": height,
        "source_video_sha256": file_sha256(face),
    }


def _render_command(face: Path, audio: Path, output: Path, geometry: Mapping[str, Any]) -> list[str]:
    top, bottom, left, right = (str(int(value)) for value in geometry["box"])
    return [
        str(config.WAV2LIP_PYTHON),
        str(config.WAV2LIP_ROOT / "inference.py"),
        "--checkpoint_path",
        str(config.WAV2LIP_CHECKPOINT),
        "--face",
        str(face),
        "--audio",
        str(audio),
        "--outfile",
        str(output),
        "--box",
        top,
        bottom,
        left,
        right,
        "--face_det_batch_size",
        "16",
        "--wav2lip_batch_size",
        "16",
        "--nosmooth",
    ]


def _runtime_bindings() -> dict[str, str]:
    paths = {
        "wav2lip_python": config.WAV2LIP_PYTHON,
        "wav2lip_checkpoint": config.WAV2LIP_CHECKPOINT,
        "wav2lip_inference": config.WAV2LIP_ROOT / "inference.py",
    }
    result: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise CalibrationError(f"Wav2Lip binding is missing: {path}")
        result[name] = file_sha256(path)
    if result["wav2lip_python"] != config.WAV2LIP_PYTHON_SHA256 or result["wav2lip_checkpoint"] != config.WAV2LIP_CHECKPOINT_SHA256 or result["wav2lip_inference"] != config.WAV2LIP_INFERENCE_SHA256:
        raise CalibrationError("Wav2Lip runtime binding differs from the registered binding")
    return result


def render_driver(face: Path, audio: Path, output: Path, geometry: Mapping[str, Any], work_dir: Path, log_path: Path) -> dict[str, Any]:
    binding = _runtime_bindings()
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "temp").mkdir(exist_ok=True)
    runtime_env = runtime_environment(work_dir)
    Path(runtime_env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    command = _render_command(face, audio, temporary, geometry)
    try:
        _run(command, work_dir, log_path, runtime_env)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise CalibrationError(f"Wav2Lip produced no output: {output}")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    streams = ffprobe_streams(output)
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    try:
        frame_rate = float(Fraction(str(videos[0].get("r_frame_rate", "")))) if len(videos) == 1 else 0.0
    except (ValueError, ZeroDivisionError):
        frame_rate = 0.0
    if len(videos) != 1 or abs(frame_rate - 25.0) > 1e-6:
        raise CalibrationError(f"rendered video is not a single 25-fps stream: {output}")
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
        "runtime_environment": runtime_env,
        "video_stream": {
            "codec_name": videos[0].get("codec_name"),
            "width": videos[0].get("width"),
            "height": videos[0].get("height"),
            "r_frame_rate": videos[0].get("r_frame_rate"),
            "time_base": videos[0].get("time_base"),
            "nb_frames": videos[0].get("nb_frames"),
        },
        "log": str(log_path),
    }


def _history_video_rows(history_videos: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in history_videos.get("rows", []):
        sample_id = str(row.get("sample_id", ""))
        if sample_id in result:
            raise CalibrationError(f"duplicate historical video row: {sample_id}")
        result[sample_id] = row
    if len(result) != config.EXPECTED_RECORD_COUNT:
        raise CalibrationError("historical video rows are incomplete")
    return result


def _render_one(record: Mapping[str, Any], audio_row: Mapping[str, Any], history_video_row: Mapping[str, Any], arm: str, branch: str, paths: config.RunPaths, protocol_sha256: str) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    arms = {str(row.get("arm")): row for row in audio_row.get("arms", [])}
    audio_row_for_arm = arms.get(arm)
    if audio_row_for_arm is None:
        raise CalibrationError(f"audio arm is missing: {sample_id}/{arm}")
    face = Path(str(record["face_video"]["path"]))
    audio = Path(str(audio_row_for_arm["output"]))
    geometry = validate_geometry(history_video_row["geometry"], face)
    output = paths.videos / "videos" / arm / f"{sample_id}.mp4"
    sidecar = output.with_suffix(".json")
    work_dir = paths.videos / "work" / arm / sample_id
    if output.is_file() and sidecar.is_file():
        prior = verify_self_hashed_json(sidecar)
        if (
            prior.get("protocol_id") != config.PROTOCOL_ID
            or prior.get("protocol_sha256") != protocol_sha256
            or prior.get("sample_id") != sample_id
            or prior.get("branch") != branch
            or prior.get("arm") != arm
            or prior.get("face_sha256") != file_sha256(face)
            or prior.get("audio_sha256") != file_sha256(audio)
            or prior.get("geometry_sha256") != canonical_json_sha256(geometry)
            or prior.get("output_sha256") != file_sha256(output)
            or prior.get("runtime_environment") != runtime_environment(work_dir)
        ):
            raise CalibrationError(f"existing video identity changed: {sample_id}/{arm}")
        return prior
    if output.exists() or sidecar.exists():
        raise CalibrationError(f"partial video cannot be resumed: {sample_id}/{arm}")
    rendered = render_driver(
        face,
        audio,
        output,
        geometry,
        work_dir,
        paths.videos / "logs" / arm / f"{sample_id}.log",
    )
    row = {
        "schema_version": 1,
        "stage_id": "03_videos",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "sample_id": sample_id,
        "source_group": str(record["source_group"]),
        "branch": branch,
        "arm": arm,
        **rendered,
    }
    write_self_hashed_json(sidecar, row)
    return row


def render_videos(cohort: Mapping[str, Any], protocol: Mapping[str, Any], audio_manifest: Mapping[str, Any], history_videos: Mapping[str, Any], paths: config.RunPaths) -> dict[str, Any]:
    branch = str(protocol.get("branch", ""))
    arms = config.arms_for_branch(branch)
    if protocol.get("status") != "locked" or audio_manifest.get("status") != "complete":
        raise CalibrationError("protocol or audio manifest is incomplete")
    protocol_sha256 = file_sha256(Path(str(protocol["_path"]))) if "_path" in protocol else str(protocol.get("artifact_sha256", ""))
    history_rows = _history_video_rows(history_videos)
    audio_rows = {str(row.get("sample_id")): row for row in audio_manifest.get("rows", [])}
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(cohort.get("records", []), 1):
        sample_id = str(record["sample_id"])
        audio_row = audio_rows.get(sample_id)
        history_row = history_rows.get(sample_id)
        if audio_row is None or history_row is None:
            raise CalibrationError(f"upstream row is missing: {sample_id}")
        rendered: dict[str, Any] = {}
        for arm in arms:
            try:
                rendered[arm] = _render_one(record, audio_row, history_row, arm, branch, paths, protocol_sha256)
            except (OSError, CalibrationError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
                failures.append({"sample_id": sample_id, "arm": arm, "error_type": type(exc).__name__, "error": str(exc)})
        if len(rendered) == len(arms):
            geometry = validate_geometry(history_row["geometry"], Path(str(record["face_video"]["path"])))
            rows.append({"sample_id": sample_id, "source_group": str(record["source_group"]), "face": str(record["face_video"]["path"]), "face_sha256": file_sha256(Path(str(record["face_video"]["path"]))), "geometry": geometry, "arms": rendered})
        print(f"VIDEO {index}/{config.EXPECTED_RECORD_COUNT} {sample_id}", flush=True)
    video_count = sum(len(row["arms"]) for row in rows)
    complete = not failures and len(rows) == config.EXPECTED_RECORD_COUNT and video_count == config.expected_video_count(branch)
    manifest = {
        "schema_version": 1,
        "stage_id": "03_videos",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if complete else "blocked",
        "branch": branch,
        "control_arm": config.control_arm_for_branch(branch),
        "arms": list(arms),
        "record_count": len(rows),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "video_count": video_count,
        "expected_video_count": config.expected_video_count(branch),
        "rows": rows,
        "failures": failures,
        "protocol_sha256": protocol_sha256,
        "history_videos_sha256": file_sha256(config.HISTORY_VIDEOS),
    }
    write_self_hashed_json(paths.videos / "videos_manifest.json", manifest)
    if not complete:
        raise CalibrationError(f"video render incomplete: {video_count}/{config.expected_video_count(branch)}")
    return manifest
