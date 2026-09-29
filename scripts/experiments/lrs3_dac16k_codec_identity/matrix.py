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


def _run(command: list[str], cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise ProtocolError(f"command failed ({result.returncode}): {log}")


def ffprobe_streams(path: Path) -> list[dict[str, Any]]:
    result = subprocess.run([config.FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(path)], capture_output=True, check=True)
    streams = json.loads(result.stdout.decode("utf-8")).get("streams")
    if not isinstance(streams, list):
        raise ProtocolError(f"no stream list: {path}")
    return streams


def decode_pcm16(path: Path) -> bytes:
    result = subprocess.run([config.FFMPEG, "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(config.SAMPLE_RATE), "-ac", "1", "pipe:1"], capture_output=True, check=True)
    return bytes(result.stdout)


def video_bitstream(path: Path, codec_name: str) -> bytes:
    format_by_codec = {"h264": "h264", "hevc": "hevc", "mpeg4": "m4v", "vp8": "ivf", "vp9": "ivf"}
    output_format = format_by_codec.get(codec_name)
    if output_format is None:
        raise ProtocolError(f"unsupported video codec for elementary-stream check: {codec_name}")
    result = subprocess.run([config.FFMPEG, "-v", "error", "-i", str(path), "-map", "0:v:0", "-c:v", "copy", "-f", output_format, "pipe:1"], capture_output=True, check=True)
    return bytes(result.stdout)


def strict_mux(video: Path, audio: Path, output: Path, log: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    _run([config.FFMPEG, "-y", "-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "pcm_s16le", "-ar", str(config.SAMPLE_RATE), "-ac", "1", "-f", "matroska", str(temporary)], output.parent, log)
    streams = ffprobe_streams(temporary)
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise ProtocolError(f"strict mux stream count mismatch: {output}")
    audio_stream = audios[0]
    if audio_stream.get("codec_name") != "pcm_s16le" or int(audio_stream.get("sample_rate", 0)) != config.SAMPLE_RATE or int(audio_stream.get("channels", 0)) != 1:
        raise ProtocolError(f"strict mux audio contract mismatch: {output}")
    expected_pcm = decode_pcm16(audio)
    actual_pcm = decode_pcm16(temporary)
    expected_hash = hashlib.sha256(expected_pcm).hexdigest()
    actual_hash = hashlib.sha256(actual_pcm).hexdigest()
    if expected_pcm != actual_pcm:
        raise ProtocolError(f"strict mux changed decoded PCM: {output}")
    codec_name = str(videos[0].get("codec_name", ""))
    expected_video = video_bitstream(video, codec_name)
    actual_video = video_bitstream(temporary, codec_name)
    expected_video_hash = hashlib.sha256(expected_video).hexdigest()
    actual_video_hash = hashlib.sha256(actual_video).hexdigest()
    if expected_video != actual_video:
        raise ProtocolError(f"strict mux changed video elementary stream: {output}")
    temporary.replace(output)
    return {
        "path": str(output),
        "sha256": file_sha256(output),
        "video_stream": {key: videos[0].get(key) for key in ("codec_name", "width", "height", "r_frame_rate", "time_base")},
        "audio_stream": {key: audio_stream.get(key) for key in ("codec_name", "sample_rate", "channels", "channel_layout")},
        "expected_audio_pcm_sha256": expected_hash,
        "muxed_audio_pcm_sha256": actual_hash,
        "audio_pcm_verified": True,
        "video_source_elementary_sha256": expected_video_hash,
        "muxed_video_elementary_sha256": actual_video_hash,
        "video_stream_copy_verified": True,
        "video_source_sha256": file_sha256(video),
        "audio_source_sha256": file_sha256(audio),
        "audio_sample_count": len(expected_pcm) // 2,
    }


def parse_syncnet(log: Path) -> dict[str, float | int]:
    import re

    text = log.read_text(encoding="utf-8", errors="replace")
    confidence = re.search(r"Confidence:\s+([0-9.]+)", text)
    distance = re.search(r"Min dist:\s+([0-9.]+)", text)
    offset = re.search(r"AV offset:\s+(-?\d+)", text)
    if confidence is None or distance is None:
        raise ProtocolError(f"could not parse SyncNet output: {log}")
    return {"sync_c": float(confidence.group(1)), "sync_d": float(distance.group(1)), "av_offset": int(offset.group(1)) if offset else 0}


def score_syncnet(media: Path, output_stage: Path, reference: str) -> dict[str, Any]:
    data_dir = output_stage / "syncnet" / reference
    log_dir = output_stage / "logs" / "syncnet"
    data_dir.mkdir(parents=True, exist_ok=True)
    pipeline_log = log_dir / f"{reference}.pipeline.log"
    score_log = log_dir / f"{reference}.score.log"
    _run([str(config.SYNCNET_PYTHON), "run_pipeline.py", "--videofile", str(media), "--reference", reference, "--data_dir", str(data_dir), "--min_track", str(config.MIN_TRACK), "--overwrite"], config.SYNCNET_ROOT, pipeline_log)
    _run([str(config.SYNCNET_PYTHON), "run_syncnet.py", "--videofile", str(media), "--reference", reference, "--data_dir", str(data_dir), "--initial_model", str(config.SYNCNET_MODEL)], config.SYNCNET_ROOT, score_log)
    return {"reference": reference, "media": str(media), "media_sha256": file_sha256(media), "log": str(score_log), "syncnet_model_sha256": file_sha256(config.SYNCNET_MODEL), "min_track": config.MIN_TRACK, **parse_syncnet(score_log)}


def matrix_paths(record: Mapping[str, Any], video_row: Mapping[str, Any], audio_row: Mapping[str, Any]) -> tuple[dict[str, Path], dict[str, Path]]:
    audio = {"N": Path(str(record["natural_audio"])), "W": Path(str(record["wavlm_audio"])), "D": Path(str(audio_row["dac"]["output_path"]))}
    videos = {arm: Path(str(video_row["arms"][arm]["output"])) for arm in config.DRIVER_ARMS}
    return videos, audio


def run_stage04(cohort: Mapping[str, Any], audio_manifest: Mapping[str, Any], video_manifest: Mapping[str, Any], output_stage: Path) -> dict[str, Any]:
    if not config.SYNCNET_PYTHON.is_file() or not config.SYNCNET_MODEL.is_file() or not (config.SYNCNET_ROOT / "run_pipeline.py").is_file() or not (config.SYNCNET_ROOT / "run_syncnet.py").is_file():
        raise FileNotFoundError("official SyncNet V2 environment is incomplete")
    if audio_manifest.get("status") != "complete" or video_manifest.get("status") != "complete":
        raise ProtocolError("upstream media stage is incomplete")
    scores: list[dict[str, Any]] = []
    muxes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, (record, audio_row, video_row) in enumerate(zip(cohort["records"], audio_manifest["rows"], video_manifest["rows"]), 1):
        sample_id = str(record["sample_id"])
        videos, audios = matrix_paths(record, video_row, audio_row)
        for cell in config.MATRIX_CELLS:
            video_label, audio_label = cell.split("/")
            video_arm = video_label.removeprefix("V_")
            audio_arm = audio_label.removeprefix("A_")
            cell_id = f"{sample_id}__{video_arm}_{audio_arm}"
            mux_path = output_stage / "mux" / video_arm / audio_arm / f"{sample_id}.mkv"
            mux_sidecar = mux_path.with_suffix(".json")
            score_sidecar = output_stage / "scores" / video_arm / audio_arm / f"{sample_id}.json"
            try:
                if mux_path.is_file() and mux_sidecar.is_file() and score_sidecar.is_file():
                    mux = verify_self_hashed_json(mux_sidecar)
                    score = verify_self_hashed_json(score_sidecar)
                    if mux.get("video_source_sha256") != file_sha256(videos[video_arm]) or mux.get("audio_source_sha256") != file_sha256(audios[audio_arm]) or score.get("media_sha256") != file_sha256(mux_path):
                        raise ProtocolError(f"resumed matrix identity changed: {cell_id}")
                elif mux_path.exists() or mux_sidecar.exists() or score_sidecar.exists():
                    raise ProtocolError(f"partial matrix cell cannot be resumed: {cell_id}")
                else:
                    mux = strict_mux(videos[video_arm], audios[audio_arm], mux_path, output_stage / "logs" / "mux" / f"{cell_id}.log")
                    write_self_hashed_json(mux_sidecar, {"schema_version": 1, "stage_id": "04_matrix", "protocol_id": config.PROTOCOL_ID, "sample_id": sample_id, "cell": cell, **mux})
                    score = score_syncnet(mux_path, output_stage, cell_id)
                    write_self_hashed_json(score_sidecar, {"schema_version": 1, "stage_id": "04_matrix", "protocol_id": config.PROTOCOL_ID, "sample_id": sample_id, "cell": cell, **score})
                muxes.append({"sample_id": sample_id, "cell": cell, **mux})
                scores.append({"sample_id": sample_id, "source_group": str(record["source_group"]), "cell": cell, **score})
            except (OSError, ProtocolError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
                failures.append({"sample_id": sample_id, "cell": cell, "error_type": type(exc).__name__, "error": str(exc)})
                print(f"FAIL {index}/{config.EXPECTED_RECORD_COUNT} {cell_id}: {exc}", flush=True)
        print(f"MATRIX {index}/{config.EXPECTED_RECORD_COUNT} {sample_id}", flush=True)
    expected = config.EXPECTED_RECORD_COUNT * len(config.MATRIX_CELLS)
    complete = len(scores) == expected and len(muxes) == expected and not failures
    result = {
        "schema_version": 1,
        "stage_id": "04_matrix",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if complete else "blocked",
        "record_count": config.EXPECTED_RECORD_COUNT,
        "cell_count": len(scores),
        "expected_cell_count": expected,
        "cells": list(config.MATRIX_CELLS),
        "muxes": muxes,
        "scores": scores,
        "failures": failures,
        "video_manifest_sha256": file_sha256(config.STAGES["03_videos"] / "videos_manifest.json"),
        "audio_manifest_sha256": file_sha256(config.STAGES["01_audio"] / "audio_manifest.json"),
    }
    write_self_hashed_json(output_stage / "matrix_manifest.json", result)
    write_self_hashed_json(output_stage / "failures.json", {"schema_version": 1, "stage_id": "04_matrix", "status": result["status"], "failures": failures})
    if not complete:
        raise ProtocolError(f"matrix incomplete: {len(scores)}/{expected}")
    return result
