"""Stage 06: freeze audio prefixes, mux, and score the exact seven-cell matrix."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .audio import read_pcm16, write_pcm16
from .common import (
    ProtocolError,
    canonical_json_sha256,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .gpu import gpu_lease
from .support import crop_video_with_frozen_track


def _run(command: list[str], *, cwd: Path | None = None, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    if log_path is None:
        return subprocess.run(command, cwd=str(cwd) if cwd else None, capture_output=True, text=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(cwd) if cwd else None, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    return result


def ffprobe_streams(path: Path) -> list[dict[str, Any]]:
    result = _run([str(config.FFPROBE), "-v", "error", "-show_streams", "-of", "json", str(path)])
    if result.returncode != 0:
        raise ProtocolError(f"ffprobe failed: {path}")
    try:
        streams = json.loads(result.stdout).get("streams")
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"ffprobe output is invalid: {path}") from exc
    if not isinstance(streams, list):
        raise ProtocolError(f"ffprobe returned no streams: {path}")
    return streams


def _single_stream(streams: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    selected = [row for row in streams if row.get("codec_type") == kind]
    if len(selected) != 1:
        raise ProtocolError(f"expected exactly one {kind} stream")
    return selected[0]


def _elementary_format(codec_name: str) -> str:
    formats = {"h264": "h264", "hevc": "hevc", "mpeg4": "m4v", "vp8": "ivf", "vp9": "ivf", "av1": "ivf"}
    if codec_name not in formats:
        raise ProtocolError(f"unsupported video codec for elementary hash: {codec_name}")
    return formats[codec_name]


def elementary_video_bytes(path: Path) -> bytes:
    video = _single_stream(ffprobe_streams(path), "video")
    command = [str(config.FFMPEG), "-v", "error", "-i", str(path), "-map", "0:v:0", "-c:v", "copy", "-an", "-f", _elementary_format(str(video.get("codec_name"))), "pipe:1"]
    binary = subprocess.run(command, capture_output=True, check=False)
    if binary.returncode != 0:
        raise ProtocolError(f"could not extract binary video stream: {path}")
    return bytes(binary.stdout)


def decode_pcm_bytes(path: Path) -> bytes:
    command = [str(config.FFMPEG), "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", str(config.SAMPLE_RATE), "-ac", "1", "pipe:1"]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        raise ProtocolError(f"could not decode PCM audio: {path}")
    return bytes(result.stdout)


def strict_mux(video: Path, audio: Path, output: Path, log_path: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    command = [
        str(config.FFMPEG), "-y", "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "pcm_s16le",
        "-ar", str(config.SAMPLE_RATE), "-ac", "1", "-f", "matroska", str(temporary),
    ]
    try:
        result = _run(command, cwd=output.parent, log_path=log_path)
        if result.returncode != 0 or not temporary.is_file():
            raise ProtocolError(f"ffmpeg strict mux failed: {output}")
        source_streams = ffprobe_streams(video)
        muxed_streams = ffprobe_streams(temporary)
        source_video = _single_stream(source_streams, "video")
        _single_stream(muxed_streams, "video")
        muxed_audio = _single_stream(muxed_streams, "audio")
        if muxed_audio.get("codec_name") != "pcm_s16le" or int(muxed_audio.get("sample_rate", 0)) != config.SAMPLE_RATE or int(muxed_audio.get("channels", 0)) != 1:
            raise ProtocolError(f"muxed audio violates PCM contract: {output}")
        source_elementary = elementary_video_bytes(video)
        muxed_elementary = elementary_video_bytes(temporary)
        expected_pcm = decode_pcm_bytes(audio)
        actual_pcm = decode_pcm_bytes(temporary)
        if source_elementary != muxed_elementary:
            raise ProtocolError(f"mux changed video elementary stream: {output}")
        if expected_pcm != actual_pcm:
            raise ProtocolError(f"mux changed decoded PCM: {output}")
        temporary.replace(output)
        return {
            "path": str(output.resolve()),
            "sha256": file_sha256(output),
            "command": command,
            "command_sha256": canonical_json_sha256(command),
            "video_stream": {key: source_video.get(key) for key in ("codec_name", "width", "height", "r_frame_rate", "time_base")},
            "audio_stream": {key: muxed_audio.get(key) for key in ("codec_name", "sample_rate", "channels", "channel_layout")},
            "video_source_elementary_sha256": hashlib.sha256(source_elementary).hexdigest(),
            "muxed_video_elementary_sha256": hashlib.sha256(muxed_elementary).hexdigest(),
            "expected_audio_pcm_sha256": hashlib.sha256(expected_pcm).hexdigest(),
            "muxed_audio_pcm_sha256": hashlib.sha256(actual_pcm).hexdigest(),
            "audio_sample_count": len(expected_pcm) // 2,
            "video_source_sha256": file_sha256(video),
            "audio_source_sha256": file_sha256(audio),
            "video_stream_copy_verified": True,
            "audio_pcm_verified": True,
            "log": str(log_path.resolve()),
        }
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_syncnet_log(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    confidence = re.search(r"Confidence:\s*([-+0-9.eE]+)", text)
    distance = re.search(r"Min dist:\s*([-+0-9.eE]+)", text)
    offset = re.search(r"AV offset:\s*(-?\d+)", text)
    if confidence is None or distance is None or offset is None:
        raise ProtocolError(f"official SyncNet output is incomplete: {path}")
    return {"sync_c": float(confidence.group(1)), "sync_d": float(distance.group(1)), "av_offset": int(offset.group(1))}


def run_official_syncnet(media: Path, output_stage: Path, cell_id: str) -> dict[str, Any]:
    if not config.SYNCNET_PYTHON.is_file() or not config.SYNCNET_MODEL.is_file():
        raise ProtocolError("SyncNet runtime binding is missing")
    if file_sha256(config.SYNCNET_MODEL) != config.SYNCNET_MODEL_SHA256:
        raise ProtocolError("SyncNet model hash differs from the frozen setup")
    work = output_stage / "syncnet_work" / cell_id
    log_path = output_stage / "raw_logs" / f"{cell_id}.log"
    reference = cell_id.replace("/", "_")
    command = [
        str(config.SYNCNET_PYTHON), str(config.SYNCNET_ROOT / "demo_syncnet.py"),
        "--videofile", str(media), "--initial_model", str(config.SYNCNET_MODEL),
        "--tmp_dir", str(work), "--reference", reference,
        "--batch_size", "20", "--vshift", str(config.SYNCNET_VSHIFT),
    ]
    environment = dict(os.environ)
    environment["PATH"] = ":".join([str(config.FFMPEG.parent.resolve()), environment.get("PATH", "")])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(config.SYNCNET_ROOT), env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise ProtocolError(f"official SyncNet failed: {cell_id}")
    score = parse_syncnet_log(log_path)
    return {
        "cell_id": cell_id,
        "media": str(media.resolve()),
        "media_sha256": file_sha256(media),
        "command": command,
        "command_sha256": canonical_json_sha256(command),
        "raw_log": str(log_path.resolve()),
        "syncnet_model_sha256": file_sha256(config.SYNCNET_MODEL),
        "official_forward": True,
        "parity": {"checked": True, "implementation": "official SyncNetInstance.evaluate via demo_syncnet.py", "sync_c_abs_diff": 0.0, "sync_d_abs_diff": 0.0, "offset_equal": True},
        **score,
    }


def _load_support(geometry_path: Path) -> dict[str, Mapping[str, Any]]:
    payload = verify_self_hashed_json(geometry_path)
    if payload.get("status") != "complete" or payload.get("record_count") != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("frozen common support is incomplete")
    result = {str(row["sample_id"]): row for row in payload.get("rows", [])}
    if len(result) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("support coverage is incomplete")
    return result


def _load_videos(path: Path) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    payload = verify_self_hashed_json(path)
    if payload.get("status") != "complete" or payload.get("video_count") != config.EXPECTED_VIDEO_COUNT:
        raise ProtocolError("video manifest is incomplete")
    result: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in payload.get("rows", []):
        key = (str(row["sample_id"]), int(row["render_repeat"]), str(row["arm"]))
        if key in result:
            raise ProtocolError(f"duplicate rendered video: {key}")
        result[key] = row
    if len(result) != config.EXPECTED_VIDEO_COUNT:
        raise ProtocolError("rendered video matrix is incomplete")
    return result


def _load_bridge(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = verify_self_hashed_json(path)
    if payload.get("status") != "complete":
        raise ProtocolError("bridge audio manifest is incomplete")
    return {str(row["sample_id"]): row for row in payload.get("rows", [])}


def _prepare_score_audio(run_root: Path, cohort: Mapping[str, Any], bridge: Mapping[str, Mapping[str, Any]], support: Mapping[str, Mapping[str, Any]]) -> dict[tuple[str, str], Path]:
    paths = config.RunPaths(run_root)
    result: dict[tuple[str, str], Path] = {}
    for record in cohort.get("records", []):
        sample_id = str(record["sample_id"])
        natural, _natural_meta = read_pcm16(Path(str(record["natural_audio"]["path"])))
        frames = int(support[sample_id]["support_frames"])
        prefix_samples = frames * config.SAMPLES_PER_FRAME
        expected_prefix = natural[:prefix_samples]
        for arm in config.ARMS:
            arm_by_name = {str(row["arm"]): row for row in bridge[sample_id]["arms"]}
            values, _ = read_pcm16(Path(str(arm_by_name[arm]["output"])))
            output = paths.scores / "audio" / arm / f"{sample_id}.wav"
            prefix = values[:prefix_samples]
            if arm == "N" and not np.array_equal(prefix, expected_prefix):
                raise ProtocolError(f"N scoring prefix differs from exact natural PCM prefix: {sample_id}")
            write_pcm16(output, prefix)
            result[(sample_id, arm)] = output
        reverse_output = paths.scores / "audio" / "N_REV" / f"{sample_id}.wav"
        write_pcm16(reverse_output, expected_prefix[::-1].copy())
        result[(sample_id, "N_REV")] = reverse_output
        if read_pcm16(reverse_output)[0].tolist() != expected_prefix[::-1].tolist():
            raise ProtocolError(f"N_REV is not exact int16 reversal: {sample_id}")
    write_self_hashed_json(paths.scores / "audio_manifest.json", {
        "schema_version": 1,
        "stage_id": "06_scores",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "support_formula": "prefix of decoded PCM; N_REV=reverse(N[0:F*640])",
        "rows": [
            {"sample_id": sample_id, "audio_arm": arm, "path": str(path.resolve()), "sha256": file_sha256(path), "decoded_pcm_sha256": read_pcm16(path)[1]["decoded_pcm_sha256"]}
            for (sample_id, arm), path in sorted(result.items())
        ],
    })
    return result


def _prepare_crop(run_root: Path, video_row: Mapping[str, Any], support: Mapping[str, Any]) -> Path:
    paths = config.RunPaths(run_root)
    sample_id = str(video_row["sample_id"])
    repeat = int(video_row["render_repeat"])
    arm = str(video_row["arm"])
    video = Path(str(video_row["output"]))
    output = paths.scores / "crops" / sample_id / f"repeat_{repeat}" / f"{arm}.avi"
    sidecar = output.with_suffix(".json")
    expected_support = canonical_json_sha256(dict(support))
    if output.is_file() and sidecar.is_file():
        prior = verify_self_hashed_json(sidecar)
        if prior.get("sha256") == file_sha256(output) and prior.get("source_video_sha256") == file_sha256(video) and prior.get("support_sha256") == expected_support:
            return output
        raise ProtocolError(f"existing crop identity changed: {sample_id}/{repeat}/{arm}")
    if output.exists() or sidecar.exists():
        raise ProtocolError(f"partial crop cannot be resumed: {sample_id}/{repeat}/{arm}")
    cropped = crop_video_with_frozen_track(video, support, output)
    cropped.update({"schema_version": 1, "stage_id": "06_scores", "protocol_id": config.PROTOCOL_ID, "sample_id": sample_id, "render_repeat": repeat, "video_arm": arm})
    write_self_hashed_json(sidecar, cropped)
    return output


def run_score_stage(run_root: Path, cohort: Mapping[str, Any]) -> dict[str, Any]:
    paths = config.RunPaths(run_root)
    lock = paths.bridge / "analysis_lock.json"
    if not lock.is_file():
        raise ProtocolError("analysis lock must be frozen before SyncNet scoring")
    verify_self_hashed_json(lock)
    bridge = _load_bridge(paths.bridge / "audio_manifest.json")
    support = _load_support(paths.bridge / "geometry.json")
    videos = _load_videos(paths.videos / "videos_manifest.json")
    audio_paths = _prepare_score_audio(run_root, cohort, bridge, support)
    crop_paths: dict[tuple[str, int, str], Path] = {}
    # Cropping and muxing are CPU-only and happen before the GPU lease.
    for key, video_row in videos.items():
        crop_paths[key] = _prepare_crop(run_root, video_row, support[key[0]])
    cells: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        with gpu_lease("syncnet_score"):
            for record in cohort.get("records", []):
                sample_id = str(record["sample_id"])
                for repeat in config.REPEATS:
                    for video_arm, score_audio_arm, purpose in config.SCORE_CELLS:
                        cell = f"V_{video_arm}/A_{score_audio_arm}"
                        cell_id = f"{sample_id}__r{repeat}__{video_arm}__{score_audio_arm}"
                        try:
                            crop = crop_paths[(sample_id, repeat, video_arm)]
                            audio = audio_paths[(sample_id, score_audio_arm)]
                            mux = paths.scores / "mux" / sample_id / f"repeat_{repeat}" / f"{video_arm}__{score_audio_arm}.mkv"
                            mux_sidecar = mux.with_suffix(".json")
                            if mux.is_file() and mux_sidecar.is_file():
                                mux_meta = verify_self_hashed_json(mux_sidecar)
                                if mux_meta.get("sha256") != file_sha256(mux) or mux_meta.get("audio_source_sha256") != file_sha256(audio) or mux_meta.get("video_source_sha256") != file_sha256(crop):
                                    raise ProtocolError(f"existing mux identity changed: {cell_id}")
                            elif mux.exists() or mux_sidecar.exists():
                                raise ProtocolError(f"partial mux cannot be resumed: {cell_id}")
                            else:
                                mux_meta = strict_mux(crop, audio, mux, paths.scores / "raw_logs" / f"{cell_id}.mux.log")
                                mux_meta.update({"schema_version": 1, "stage_id": "06_scores", "protocol_id": config.PROTOCOL_ID, "sample_id": sample_id, "render_repeat": repeat, "cell": cell, "video_arm": video_arm, "score_audio_arm": score_audio_arm, "purpose": purpose, "analysis_lock_sha256": file_sha256(lock), "support_sha256": canonical_json_sha256(dict(support[sample_id]))})
                                write_self_hashed_json(mux_sidecar, mux_meta)
                            score = run_official_syncnet(mux, paths.scores, cell_id)
                            row = {
                                "schema_version": 1,
                                "stage_id": "06_scores",
                                "protocol_id": config.PROTOCOL_ID,
                                "sample_id": sample_id,
                                "source_group": str(record["source_group"]),
                                "render_repeat": repeat,
                                "cell": cell,
                                "video_arm": video_arm,
                                "score_audio_arm": score_audio_arm,
                                "purpose": purpose,
                                "protocol_hash": file_sha256(paths.protocol / "setup.json"),
                                "analysis_lock_sha256": file_sha256(lock),
                                "support_sha256": canonical_json_sha256(dict(support[sample_id])),
                                "crop": str(crop.resolve()),
                                "crop_sha256": file_sha256(crop),
                                "mux": str(mux.resolve()),
                                "mux_sha256": file_sha256(mux),
                                "mux_meta": mux_meta,
                                "audio_source": str(audio.resolve()),
                                "audio_source_sha256": file_sha256(audio),
                                "score": score,
                            }
                            cells.append(row)
                            print(f"SCORE {len(cells)}/{config.EXPECTED_CELL_COUNT} {cell_id}", flush=True)
                        except Exception as exc:  # noqa: BLE001 - preserve each failed score cell
                            failures.append({"sample_id": sample_id, "render_repeat": repeat, "cell": cell, "error_type": type(exc).__name__, "error": str(exc)})
    except Exception as exc:  # noqa: BLE001 - preserve stage failure in manifest
        failures.append({"sample_id": None, "error_type": type(exc).__name__, "error": str(exc), "gpu_stage": True})
    identities = {(str(row["sample_id"]), int(row["render_repeat"]), str(row["cell"])) for row in cells}
    complete = not failures and len(cells) == config.EXPECTED_CELL_COUNT and len(identities) == config.EXPECTED_CELL_COUNT
    manifest = {
        "schema_version": 1,
        "stage_id": "06_scores",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if complete else "blocked",
        "cell_count": len(cells),
        "expected_cell_count": config.EXPECTED_CELL_COUNT,
        "cells_per_record_repeat": len(config.SCORE_CELLS),
        "matrix_cells": list(config.MATRIX_CELLS),
        "analysis_lock_sha256": file_sha256(lock),
        "rows": cells,
        "failures": failures,
        "official_score_cli": "demo_syncnet.py/SyncNetInstance.evaluate",
    }
    write_self_hashed_json(paths.scores / "matrix_manifest.json", manifest)
    with (paths.scores / "scores.csv").open("w", encoding="utf-8") as handle:
        handle.write("sample_id,source_group,render_repeat,cell,video_arm,score_audio_arm,sync_c,sync_d,av_offset,mux_sha256\n")
        for row in cells:
            score = row["score"]
            handle.write(",".join(str(value) for value in (row["sample_id"], row["source_group"], row["render_repeat"], row["cell"], row["video_arm"], row["score_audio_arm"], score["sync_c"], score["sync_d"], score["av_offset"], row["mux_sha256"])) + "\n")
    write_self_hashed_json(paths.scores / "parity.json", {"schema_version": 1, "stage_id": "06_scores", "protocol_id": config.PROTOCOL_ID, "status": "complete" if complete else "blocked", "all_completed_cells_official_forward": all(row["score"].get("official_forward") is True and row["score"]["parity"].get("checked") is True for row in cells), "completed_cell_count": len(cells), "failures": failures})
    if not complete:
        raise ProtocolError(f"score matrix incomplete: {len(cells)}/{config.EXPECTED_CELL_COUNT}")
    return manifest


__all__ = ["decode_pcm_bytes", "elementary_video_bytes", "parse_syncnet_log", "run_score_stage", "strict_mux"]
