from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import config
from .common import (
    CalibrationError,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def _run(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT, check=False)
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


def decode_pcm16(path: Path) -> bytes:
    result = subprocess.run(
        [
            str(config.FFMPEG),
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(config.SAMPLE_RATE),
            "-ac",
            "1",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return bytes(result.stdout)


def video_bitstream(path: Path, codec_name: str | None = None) -> bytes:
    if codec_name is None:
        streams = ffprobe_streams(path)
        videos = [stream for stream in streams if stream.get("codec_type") == "video"]
        if len(videos) != 1:
            raise CalibrationError(f"video stream count is invalid: {path}")
        codec_name = str(videos[0].get("codec_name", ""))
    output_format = {"h264": "h264", "hevc": "hevc", "mpeg4": "m4v", "vp8": "ivf", "vp9": "ivf"}.get(codec_name)
    if output_format is None:
        raise CalibrationError(f"unsupported video codec for elementary-stream check: {codec_name}")
    result = subprocess.run(
        [str(config.FFMPEG), "-v", "error", "-i", str(path), "-map", "0:v:0", "-c:v", "copy", "-f", output_format, "pipe:1"],
        capture_output=True,
        check=True,
    )
    return bytes(result.stdout)


def _finite_float(value: Any, name: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise CalibrationError(f"{name} is missing or invalid: {path}") from None
    if not math.isfinite(result):
        raise CalibrationError(f"{name} is non-finite: {path}")
    return result


def runtime_bindings() -> dict[str, str]:
    expected = {
        "syncnet_python": (config.SYNCNET_PYTHON, config.SYNCNET_PYTHON_SHA256),
        "syncnet_model": (config.SYNCNET_MODEL, config.SYNCNET_MODEL_SHA256),
        "ffmpeg": (config.FFMPEG, config.FFMPEG_SHA256),
        "ffprobe": (config.FFPROBE, config.FFPROBE_SHA256),
    }
    result: dict[str, str] = {}
    for name, (path, expected_sha256) in expected.items():
        if not path.is_file():
            raise CalibrationError(f"runtime binding is missing: {path}")
        actual = file_sha256(path)
        if actual != expected_sha256:
            raise CalibrationError(f"runtime binding differs from the registered binding: {name}")
        result[name] = actual
    return result


def _stream_duration(stream: Mapping[str, Any], name: str, path: Path) -> float:
    value = stream.get("duration")
    if value is None:
        tags = stream.get("tags")
        if isinstance(tags, Mapping):
            text = tags.get("DURATION")
            if isinstance(text, str):
                parts = text.split(":")
                if len(parts) == 3:
                    try:
                        hours, minutes, seconds = (float(part) for part in parts)
                        value = hours * 3600.0 + minutes * 60.0 + seconds
                    except ValueError:
                        value = None
    return _finite_float(value, name, path)


def _timeline(streams: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise CalibrationError(f"mux must contain exactly one video and one audio stream: {path}")
    video = videos[0]
    audio = audios[0]
    video_start = _finite_float(video.get("start_time", 0.0), "video start_time", path)
    audio_start = _finite_float(audio.get("start_time", 0.0), "audio start_time", path)
    video_duration = _stream_duration(video, "video duration", path)
    audio_duration = _stream_duration(audio, "audio duration", path)
    frame_rate_text = str(video.get("r_frame_rate", ""))
    try:
        frame_rate = float(Fraction(frame_rate_text))
    except (ValueError, ZeroDivisionError):
        raise CalibrationError(f"video frame rate is invalid: {path}") from None
    if abs(frame_rate - 25.0) > 1e-6:
        raise CalibrationError(f"video frame rate is not 25 fps: {path}")
    if abs(video_start) > 1e-3 or abs(audio_start) > 1e-3 or abs(video_start - audio_start) > 1.0 / 25.0:
        raise CalibrationError(f"audio/video start times are not aligned to zero: {path}")
    return {
        "video_start_time": video_start,
        "audio_start_time": audio_start,
        "video_duration": video_duration,
        "audio_duration": audio_duration,
        "duration_difference": video_duration - audio_duration,
        "frame_rate": frame_rate,
        "video_nb_frames": video.get("nb_frames"),
        "time_base": video.get("time_base"),
    }


def verify_mux(video: Path, audio: Path, mux: Path) -> dict[str, Any]:
    """Verify an already-created mux without changing it."""

    if not video.is_file() or not audio.is_file() or not mux.is_file():
        raise CalibrationError(f"mux input is missing: {video}, {audio}, {mux}")
    mux_streams = ffprobe_streams(mux)
    timeline = _timeline(mux_streams, mux)
    videos = [stream for stream in mux_streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in mux_streams if stream.get("codec_type") == "audio"]
    audio_stream = audios[0]
    if audio_stream.get("codec_name") != "pcm_s16le" or int(audio_stream.get("sample_rate", 0)) != config.SAMPLE_RATE or int(audio_stream.get("channels", 0)) != config.PCM_CHANNELS:
        raise CalibrationError(f"mux audio contract is invalid: {mux}")
    source_streams = ffprobe_streams(video)
    source_videos = [stream for stream in source_streams if stream.get("codec_type") == "video"]
    if len(source_videos) != 1:
        raise CalibrationError(f"source video stream count is invalid: {video}")
    codec_name = str(videos[0].get("codec_name", ""))
    expected_pcm = decode_pcm16(audio)
    actual_pcm = decode_pcm16(mux)
    if actual_pcm != expected_pcm:
        raise CalibrationError(f"mux decoded PCM differs from source audio: {mux}")
    expected_video = video_bitstream(video, codec_name)
    actual_video = video_bitstream(mux, codec_name)
    if expected_video != actual_video:
        raise CalibrationError(f"mux video elementary stream differs from source video: {mux}")
    source_rate = str(source_videos[0].get("r_frame_rate", ""))
    try:
        source_frame_rate = float(Fraction(source_rate))
    except (ValueError, ZeroDivisionError):
        raise CalibrationError(f"source video frame rate is invalid: {video}") from None
    if abs(source_frame_rate - 25.0) > 1e-6:
        raise CalibrationError(f"source video is not 25 fps: {video}")
    return {
        "path": str(mux),
        "sha256": file_sha256(mux),
        "video_stream": {key: videos[0].get(key) for key in ("codec_name", "width", "height", "r_frame_rate", "time_base", "nb_frames")},
        "audio_stream": {key: audio_stream.get(key) for key in ("codec_name", "sample_rate", "channels", "channel_layout")},
        "expected_audio_pcm_sha256": hashlib.sha256(expected_pcm).hexdigest(),
        "muxed_audio_pcm_sha256": hashlib.sha256(actual_pcm).hexdigest(),
        "audio_pcm_verified": True,
        "video_source_elementary_sha256": hashlib.sha256(expected_video).hexdigest(),
        "muxed_video_elementary_sha256": hashlib.sha256(actual_video).hexdigest(),
        "video_stream_copy_verified": True,
        "video_source_sha256": file_sha256(video),
        "audio_source_sha256": file_sha256(audio),
        "audio_sample_count": len(expected_pcm) // config.PCM_SAMPLE_WIDTH,
        "timeline": timeline,
    }


def strict_mux(video: Path, audio: Path, output: Path, log_path: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    try:
        _run(
            [
                str(config.FFMPEG),
                "-y",
                "-i",
                str(video),
                "-i",
                str(audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "pcm_s16le",
                "-ar",
                str(config.SAMPLE_RATE),
                "-ac",
                "1",
                "-f",
                "matroska",
                str(temporary),
            ],
            output.parent,
            log_path,
        )
        evidence = verify_mux(video, audio, temporary)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    evidence["path"] = str(output)
    evidence["sha256"] = file_sha256(output)
    return evidence


def parse_syncnet(log_path: Path) -> dict[str, float | int]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = {
        "sync_c": re.findall(r"Confidence:\s+([-+]?\d+(?:\.\d+)?)", text),
        "sync_d": re.findall(r"Min dist:\s+([-+]?\d+(?:\.\d+)?)", text),
        "av_offset": re.findall(r"AV offset:\s+(-?\d+)", text),
    }
    if any(len(values) != 1 for values in matches.values()):
        raise CalibrationError(f"SyncNet output is missing or ambiguous: {log_path}")
    result: dict[str, float | int] = {
        "sync_c": float(matches["sync_c"][0]),
        "sync_d": float(matches["sync_d"][0]),
        "av_offset": int(matches["av_offset"][0]),
    }
    if not math.isfinite(float(result["sync_c"])) or not math.isfinite(float(result["sync_d"])):
        raise CalibrationError(f"SyncNet output is non-finite: {log_path}")
    return result


def score_syncnet(media: Path, output_stage: Path, reference: str) -> dict[str, Any]:
    bindings = runtime_bindings()
    data_dir = output_stage / "syncnet" / reference
    pipeline_log = output_stage / "logs" / "syncnet" / f"{reference}.pipeline.log"
    score_log = output_stage / "logs" / "syncnet" / f"{reference}.score.log"
    data_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(config.SYNCNET_PYTHON),
            "run_pipeline.py",
            "--videofile",
            str(media),
            "--reference",
            reference,
            "--data_dir",
            str(data_dir),
            "--min_track",
            str(config.MIN_TRACK),
            "--overwrite",
        ],
        config.SYNCNET_ROOT,
        pipeline_log,
    )
    _run(
        [
            str(config.SYNCNET_PYTHON),
            "run_syncnet.py",
            "--videofile",
            str(media),
            "--reference",
            reference,
            "--data_dir",
            str(data_dir),
            "--initial_model",
            str(config.SYNCNET_MODEL),
        ],
        config.SYNCNET_ROOT,
        score_log,
    )
    return {
        "reference": reference,
        "media": str(media),
        "media_sha256": file_sha256(media),
        "pipeline_log": str(pipeline_log),
        "score_log": str(score_log),
        "syncnet_model_sha256": bindings["syncnet_model"],
        "syncnet_python_sha256": bindings["syncnet_python"],
        "runtime_bindings": bindings,
        "min_track": config.MIN_TRACK,
        **parse_syncnet(score_log),
    }


def expected_cells_for_sample(sample_id: str, control_arm: str) -> list[dict[str, str]]:
    return [
        {
            "sample_id": sample_id,
            "cell": cell,
            "video_arm": cell.split("/")[0].removeprefix("V_"),
            "audio_arm": cell.split("/")[1].removeprefix("A_"),
        }
        for cell in config.matrix_cells(control_arm)
    ]


def _find_row(rows: list[Mapping[str, Any]], sample_id: str, name: str) -> Mapping[str, Any]:
    matches = [row for row in rows if str(row.get("sample_id")) == sample_id]
    if len(matches) != 1:
        raise CalibrationError(f"{name} row is missing or duplicated: {sample_id}")
    return matches[0]


def _score_cell(record: Mapping[str, Any], audio_row: Mapping[str, Any], video_row: Mapping[str, Any], cell_info: Mapping[str, str], protocol: Mapping[str, Any], paths: config.RunPaths) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id = str(record["sample_id"])
    cell = str(cell_info["cell"])
    video_arm = str(cell_info["video_arm"])
    audio_arm = str(cell_info["audio_arm"])
    cell_id = f"{sample_id}__{video_arm}_{audio_arm}"
    audio_by_arm = {str(row.get("arm")): row for row in audio_row.get("arms", [])}
    video_by_arm = {str(arm): row for arm, row in video_row.get("arms", {}).items()}
    if audio_arm not in audio_by_arm or video_arm not in video_by_arm:
        raise CalibrationError(f"matrix source arm is missing: {cell_id}")
    video = Path(str(video_by_arm[video_arm]["output"]))
    audio = Path(str(audio_by_arm[audio_arm]["output"]))
    mux = paths.scores / "mux" / video_arm / audio_arm / f"{sample_id}.mkv"
    mux_sidecar = mux.with_suffix(".json")
    score_sidecar = paths.scores / "scores" / video_arm / audio_arm / f"{sample_id}.json"
    if mux.is_file() and mux_sidecar.is_file():
        mux_payload = verify_self_hashed_json(mux_sidecar)
        if (
            mux_payload.get("protocol_id") != config.PROTOCOL_ID
            or mux_payload.get("protocol_sha256") != protocol.get("_sha256")
            or mux_payload.get("sample_id") != sample_id
            or mux_payload.get("cell") != cell
            or mux_payload.get("video_source_sha256") != file_sha256(video)
            or mux_payload.get("audio_source_sha256") != file_sha256(audio)
            or mux_payload.get("sha256") != file_sha256(mux)
        ):
            raise CalibrationError(f"resumed mux identity changed: {cell_id}")
        mux_evidence = verify_mux(video, audio, mux)
    elif mux.exists() or mux_sidecar.exists() or score_sidecar.exists():
        raise CalibrationError(f"partial matrix cell cannot be resumed: {cell_id}")
    else:
        mux_evidence = strict_mux(video, audio, mux, paths.scores / "logs" / "mux" / f"{cell_id}.log")
        mux_payload = {"schema_version": 1, "stage_id": "04_scores", "protocol_id": config.PROTOCOL_ID, "protocol_sha256": protocol.get("_sha256"), "sample_id": sample_id, "cell": cell, **mux_evidence}
        write_self_hashed_json(mux_sidecar, mux_payload)
    if score_sidecar.is_file():
        score = verify_self_hashed_json(score_sidecar)
        if score.get("protocol_id") != config.PROTOCOL_ID or score.get("protocol_sha256") != protocol.get("_sha256") or score.get("sample_id") != sample_id or score.get("cell") != cell or score.get("media_sha256") != file_sha256(mux):
            raise CalibrationError(f"resumed score identity changed: {cell_id}")
    elif score_sidecar.exists():
        raise CalibrationError(f"partial score cell cannot be resumed: {cell_id}")
    else:
        score = score_syncnet(mux, paths.scores, cell_id)
        score = {"schema_version": 1, "stage_id": "04_scores", "protocol_id": config.PROTOCOL_ID, "protocol_sha256": protocol.get("_sha256"), "sample_id": sample_id, "cell": cell, **score}
        write_self_hashed_json(score_sidecar, score)
    return {"sample_id": sample_id, "cell": cell, **mux_evidence}, {"sample_id": sample_id, "source_group": str(record["source_group"]), "cell": cell, "video_arm": video_arm, "audio_arm": audio_arm, **score}


def run_scores(cohort: Mapping[str, Any], protocol: Mapping[str, Any], audio_manifest: Mapping[str, Any], video_manifest: Mapping[str, Any], paths: config.RunPaths) -> dict[str, Any]:
    branch = str(protocol.get("branch", ""))
    control_arm = config.control_arm_for_branch(branch)
    cells = config.matrix_cells(control_arm)
    if protocol.get("status") != "locked" or audio_manifest.get("status") != "complete" or video_manifest.get("status") != "complete":
        raise CalibrationError("upstream protocol or media stage is incomplete")
    runtime_bindings()
    protocol_path = Path(str(protocol.get("_path", "")))
    protocol_hash = file_sha256(protocol_path) if protocol_path.is_file() else str(protocol.get("artifact_sha256", ""))
    protocol_for_cells = dict(protocol)
    protocol_for_cells["_sha256"] = protocol_hash
    audio_rows = list(audio_manifest.get("rows", []))
    video_rows = list(video_manifest.get("rows", []))
    scores: list[dict[str, Any]] = []
    muxes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(cohort.get("records", []), 1):
        sample_id = str(record["sample_id"])
        audio_row = _find_row(audio_rows, sample_id, "audio")
        video_row = _find_row(video_rows, sample_id, "video")
        for cell_info in expected_cells_for_sample(sample_id, control_arm):
            try:
                mux, score = _score_cell(record, audio_row, video_row, cell_info, protocol_for_cells, paths)
                muxes.append(mux)
                scores.append(score)
            except (OSError, CalibrationError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
                failures.append({"sample_id": sample_id, "cell": str(cell_info["cell"]), "error_type": type(exc).__name__, "error": str(exc)})
        print(f"SCORE {index}/{config.EXPECTED_RECORD_COUNT} {sample_id}", flush=True)
    expected = config.expected_cell_count(branch)
    complete = not failures and len(scores) == expected and len(muxes) == expected
    manifest = {
        "schema_version": 1,
        "stage_id": "04_scores",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if complete else "blocked",
        "branch": branch,
        "control_arm": control_arm,
        "record_count": config.EXPECTED_RECORD_COUNT,
        "cell_count": len(scores),
        "expected_cell_count": expected,
        "cells": list(cells),
        "muxes": muxes,
        "scores": scores,
        "failures": failures,
        "protocol_sha256": protocol_hash,
        "audio_manifest_sha256": file_sha256(paths.audio / "audio_manifest.json"),
        "videos_manifest_sha256": file_sha256(paths.videos / "videos_manifest.json"),
    }
    write_self_hashed_json(paths.scores / "scores_manifest.json", manifest)
    if not complete:
        raise CalibrationError(f"score matrix incomplete: {len(scores)}/{expected}")
    return manifest
