from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import config
from .common import (
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
    result = subprocess.run([str(config.FFPROBE), "-v", "error", "-show_streams", "-of", "json", str(path)], capture_output=True, check=True)
    streams = json.loads(result.stdout.decode("utf-8")).get("streams")
    if not isinstance(streams, list):
        raise ProtocolError(f"no stream list: {path}")
    return streams


def decode_pcm16(path: Path) -> bytes:
    result = subprocess.run(
        [str(config.FFMPEG), "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(config.SAMPLE_RATE), "-ac", "1", "pipe:1"],
        capture_output=True,
        check=True,
    )
    return bytes(result.stdout)


def video_bitstream(path: Path, codec_name: str) -> bytes:
    format_by_codec = {"h264": "h264", "hevc": "hevc", "mpeg4": "m4v", "vp8": "ivf", "vp9": "ivf"}
    output_format = format_by_codec.get(codec_name)
    if output_format is None:
        raise ProtocolError(f"unsupported video codec for elementary-stream check: {codec_name}")
    result = subprocess.run([str(config.FFMPEG), "-v", "error", "-i", str(path), "-map", "0:v:0", "-c:v", "copy", "-f", output_format, "pipe:1"], capture_output=True, check=True)
    return bytes(result.stdout)


def strict_mux(video: Path, audio: Path, output: Path, log_path: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    try:
        _run([str(config.FFMPEG), "-y", "-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "pcm_s16le", "-ar", str(config.SAMPLE_RATE), "-ac", "1", "-f", "matroska", str(temporary)], output.parent, log_path)
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
        if expected_pcm != actual_pcm:
            raise ProtocolError(f"strict mux changed decoded PCM: {output}")
        codec_name = str(videos[0].get("codec_name", ""))
        expected_video = video_bitstream(video, codec_name)
        actual_video = video_bitstream(temporary, codec_name)
        if expected_video != actual_video:
            raise ProtocolError(f"strict mux changed video elementary stream: {output}")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(output),
        "sha256": file_sha256(output),
        "video_stream": {key: videos[0].get(key) for key in ("codec_name", "width", "height", "r_frame_rate", "time_base")},
        "audio_stream": {key: audio_stream.get(key) for key in ("codec_name", "sample_rate", "channels", "channel_layout")},
        "expected_audio_pcm_sha256": hashlib.sha256(expected_pcm).hexdigest(),
        "muxed_audio_pcm_sha256": hashlib.sha256(actual_pcm).hexdigest(),
        "audio_pcm_verified": True,
        "video_source_elementary_sha256": hashlib.sha256(expected_video).hexdigest(),
        "muxed_video_elementary_sha256": hashlib.sha256(actual_video).hexdigest(),
        "video_stream_copy_verified": True,
        "video_source_sha256": file_sha256(video),
        "audio_source_sha256": file_sha256(audio),
        "audio_sample_count": len(expected_pcm) // 2,
    }


def parse_syncnet(log_path: Path) -> dict[str, float | int]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    confidence = re.search(r"Confidence:\s+([0-9.]+)", text)
    distance = re.search(r"Min dist:\s+([0-9.]+)", text)
    offset = re.search(r"AV offset:\s+(-?\d+)", text)
    if confidence is None or distance is None:
        raise ProtocolError(f"could not parse SyncNet output: {log_path}")
    return {"sync_c": float(confidence.group(1)), "sync_d": float(distance.group(1)), "av_offset": int(offset.group(1)) if offset else 0}


def score_syncnet(media: Path, output_stage: Path, reference: str) -> dict[str, Any]:
    data_dir = output_stage / "syncnet" / reference
    pipeline_log = output_stage / "logs" / "syncnet" / f"{reference}.pipeline.log"
    score_log = output_stage / "logs" / "syncnet" / f"{reference}.score.log"
    data_dir.mkdir(parents=True, exist_ok=True)
    _run([str(config.SYNCNET_PYTHON), "run_pipeline.py", "--videofile", str(media), "--reference", reference, "--data_dir", str(data_dir), "--min_track", str(config.MIN_TRACK), "--overwrite"], config.SYNCNET_ROOT, pipeline_log)
    _run([str(config.SYNCNET_PYTHON), "run_syncnet.py", "--videofile", str(media), "--reference", reference, "--data_dir", str(data_dir), "--initial_model", str(config.SYNCNET_MODEL)], config.SYNCNET_ROOT, score_log)
    return {
        "reference": reference,
        "media": str(media),
        "media_sha256": file_sha256(media),
        "score_log": str(score_log),
        "syncnet_model_sha256": file_sha256(config.SYNCNET_MODEL),
        "syncnet_python_sha256": file_sha256(config.SYNCNET_PYTHON),
        "min_track": config.MIN_TRACK,
        **parse_syncnet(score_log),
    }


def expected_cells_for_sample(sample_id: str) -> list[dict[str, str]]:
    return [
        {
            "sample_id": sample_id,
            "cell": cell,
            "video_arm": cell.split("/")[0].removeprefix("V_"),
            "audio_arm": cell.split("/")[1].removeprefix("A_"),
        }
        for cell in config.MATRIX_CELLS
    ]


def _binding_check() -> None:
    required = {
        config.FFMPEG: config.FFMPEG_SHA256,
        config.FFPROBE: config.FFPROBE_SHA256,
        config.WAV2LIP_CHECKPOINT: config.WAV2LIP_CHECKPOINT_SHA256,
        config.SYNCNET_MODEL: config.SYNCNET_MODEL_SHA256,
        config.SYNCNET_PYTHON: config.SYNCNET_PYTHON_SHA256,
    }
    for path, expected in required.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise ProtocolError(f"frozen runtime binding changed: {path}")
    for path in (config.SYNCNET_ROOT / "run_pipeline.py", config.SYNCNET_ROOT / "run_syncnet.py"):
        if not path.is_file():
            raise ProtocolError(f"SyncNet executable is missing: {path}")


def _find_row(rows: list[Mapping[str, Any]], sample_id: str) -> Mapping[str, Any]:
    matches = [row for row in rows if str(row.get("sample_id")) == sample_id]
    if len(matches) != 1:
        raise ProtocolError(f"expected one media row for {sample_id}")
    return matches[0]


def _score_cell(
    record: Mapping[str, Any],
    audio_manifest_row: Mapping[str, Any],
    video_manifest_row: Mapping[str, Any],
    cell_info: Mapping[str, str],
    output_stage: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id = str(record["sample_id"])
    cell = str(cell_info["cell"])
    video_arm = str(cell_info["video_arm"])
    audio_arm = str(cell_info["audio_arm"])
    cell_id = f"{sample_id}__{video_arm}_{audio_arm}"
    audio_by_arm = {str(row["arm"]): row for row in audio_manifest_row.get("arms", [])}
    if set(audio_by_arm) != set(config.ARMS):
        raise ProtocolError(f"audio arms are incomplete: {sample_id}")
    video = Path(str(video_manifest_row["arms"][video_arm]["output"]))
    audio = Path(str(audio_by_arm[audio_arm]["output"]))
    mux_path = output_stage / "mux" / video_arm / audio_arm / f"{sample_id}.mkv"
    mux_sidecar = mux_path.with_suffix(".json")
    score_sidecar = output_stage / "scores" / video_arm / audio_arm / f"{sample_id}.json"
    temporary = mux_path.with_name(f".{mux_path.stem}.partial{mux_path.suffix}")
    if mux_path.is_file() and mux_sidecar.is_file():
        mux = verify_self_hashed_json(mux_sidecar)
        if (
            mux.get("protocol_id") != config.PROTOCOL_ID
            or mux.get("sample_id") != sample_id
            or mux.get("cell") != cell
            or mux.get("video_source_sha256") != file_sha256(video)
            or mux.get("audio_source_sha256") != file_sha256(audio)
            or mux.get("sha256") != file_sha256(mux_path)
            or mux.get("audio_pcm_verified") is not True
            or mux.get("video_stream_copy_verified") is not True
        ):
            raise ProtocolError(f"resumed mux identity changed: {cell_id}")
    elif mux_path.exists() or mux_sidecar.exists() or score_sidecar.exists() or temporary.exists():
        raise ProtocolError(f"partial matrix cell cannot be resumed: {cell_id}")
    else:
        mux = strict_mux(video, audio, mux_path, output_stage / "logs" / "mux" / f"{cell_id}.log")
        write_self_hashed_json(mux_sidecar, {"schema_version": 1, "stage_id": "04_scores", "protocol_id": config.PROTOCOL_ID, "sample_id": sample_id, "cell": cell, **mux})
    if score_sidecar.is_file():
        score = verify_self_hashed_json(score_sidecar)
        if score.get("protocol_id") != config.PROTOCOL_ID or score.get("sample_id") != sample_id or score.get("cell") != cell or score.get("media_sha256") != file_sha256(mux_path):
            raise ProtocolError(f"resumed score identity changed: {cell_id}")
    elif score_sidecar.exists():
        raise ProtocolError(f"partial score cell cannot be resumed: {cell_id}")
    else:
        score = score_syncnet(mux_path, output_stage, cell_id)
        write_self_hashed_json(score_sidecar, {"schema_version": 1, "stage_id": "04_scores", "protocol_id": config.PROTOCOL_ID, "sample_id": sample_id, "cell": cell, **score})
    return {"sample_id": sample_id, "cell": cell, **mux}, {"sample_id": sample_id, "source_group": str(record["source_group"]), "cell": cell, **score}


def run_stage04(cohort: Mapping[str, Any], audio_manifest: Mapping[str, Any], diagnostics: Mapping[str, Any], video_manifest: Mapping[str, Any], output_stage: Path) -> dict[str, Any]:
    _binding_check()
    if diagnostics.get("status") != "complete" or diagnostics.get("frozen_before_scoring") is not True or diagnostics.get("score_read_count") != 0:
        raise ProtocolError("Stage 02 diagnostics were not frozen before scoring")
    if audio_manifest.get("status") != "complete" or video_manifest.get("status") != "complete":
        raise ProtocolError("upstream media stage is incomplete")
    jobs: list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, str]]] = []
    for record in cohort.get("records", []):
        sample_id = str(record["sample_id"])
        audio_row = _find_row(list(audio_manifest.get("rows", [])), sample_id)
        video_row = _find_row(list(video_manifest.get("rows", [])), sample_id)
        jobs.extend((record, audio_row, video_row, cell_info) for cell_info in expected_cells_for_sample(sample_id))
    results: list[tuple[dict[str, Any], dict[str, Any]] | None] = [None] * len(jobs)
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_score_cell, record, audio_row, video_row, cell_info, output_stage) for record, audio_row, video_row, cell_info in jobs]
        for index, future in enumerate(futures):
            record, _, _, cell_info = jobs[index]
            try:
                results[index] = future.result()
            except (OSError, ProtocolError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as exc:
                failures.append({"sample_id": str(record["sample_id"]), "cell": str(cell_info["cell"]), "error_type": type(exc).__name__, "error": str(exc)})
    completed = [result for result in results if result is not None]
    muxes = [result[0] for result in completed]
    scores = [result[1] for result in completed]
    complete = not failures and len(scores) == config.EXPECTED_CELL_COUNT and len(muxes) == config.EXPECTED_CELL_COUNT
    result = {
        "schema_version": 1,
        "stage_id": "04_scores",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if complete else "blocked",
        "record_count": config.EXPECTED_RECORD_COUNT,
        "cell_count": len(scores),
        "expected_cell_count": config.EXPECTED_CELL_COUNT,
        "cells": list(config.MATRIX_CELLS),
        "muxes": muxes,
        "scores": scores,
        "failures": failures,
        "video_manifest_sha256": file_sha256(config.STAGES["03_videos"] / "videos_manifest.json"),
        "audio_manifest_sha256": file_sha256(config.STAGES["01_candidates"] / "audio_manifest.json"),
        "diagnostics_sha256": file_sha256(config.STAGES["02_audio_diagnostics"] / "diagnostics.json"),
        "syncnet_model_sha256": config.SYNCNET_MODEL_SHA256,
        "score_workers": 8,
    }
    write_self_hashed_json(output_stage / "scores_manifest.json", result)
    write_self_hashed_json(output_stage / "failures.json", {"schema_version": 1, "stage_id": "04_scores", "protocol_id": config.PROTOCOL_ID, "status": result["status"], "failures": failures})
    if not complete:
        raise ProtocolError(f"score matrix incomplete: {len(scores)}/{config.EXPECTED_CELL_COUNT}")
    return result
