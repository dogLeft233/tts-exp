#!/usr/bin/env python3
"""Run the frozen Wav2Lip Stage-A dev plan with original-PCM SyncNet scoring.

This executor transforms natural audio only for the Wav2Lip generator, then
losslessly muxes the original canonical float32 PCM. It asks the official
SyncNet tracker for its selected crop, converts the full natural WAV to PCM16
and slices the exact crop source-clock interval, and calls the pinned
SyncNetInstance.evaluate API directly on a fresh crop-video/PCM16 Matroska.
All failures remain in the ordered result ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import wave
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "cross_tfg_audio_head_stagea_v1"
LAG_OFFSETS = np.arange(-15, 16, dtype="<i4")
ZERO_COLUMN = 15
IDENTITY_FIELDS = (
    "row_id", "split", "sample_id", "speaker_id", "model_id", "audio_frontend",
    "model_role", "face_id", "seed", "method_id", "method_config_sha256", "checkpoint_sha256",
)
ARRAY_DTYPES = {
    "query_frame_index": np.dtype("<i4"),
    "query_timestamp_num": np.dtype("<i8"),
    "query_timestamp_den": np.dtype("<i8"),
    "track_frame_index": np.dtype("<i4"),
    "valid_query_mask": np.dtype("bool"),
    "distance_by_query_lag": np.dtype("<f4"),
    "lag_offsets_frames": np.dtype("<i4"),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run_logged(argv: Sequence[str], *, cwd: Path, env: Mapping[str, str] | None,
               stdout_path: Path, stderr_path: Path, timeout: float) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(
                list(argv), cwd=str(cwd), env=dict(env) if env is not None else None,
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                timeout=timeout, check=False,
            )
        return {
            "argv": list(argv), "cwd": str(cwd), "returncode": completed.returncode,
            "started_unix": started, "ended_unix": time.time(),
            "stdout_path": str(stdout_path), "stdout_sha256": sha256_file(stdout_path),
            "stdout_bytes": stdout_path.stat().st_size,
            "stderr_path": str(stderr_path), "stderr_sha256": sha256_file(stderr_path),
            "stderr_bytes": stderr_path.stat().st_size,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": list(argv), "cwd": str(cwd), "returncode": None, "timed_out": True,
            "started_unix": started, "ended_unix": time.time(),
            "stdout_path": str(stdout_path), "stdout_sha256": sha256_file(stdout_path) if stdout_path.exists() else None,
            "stderr_path": str(stderr_path), "stderr_sha256": sha256_file(stderr_path) if stderr_path.exists() else None,
            "error": f"TimeoutExpired: {exc}",
        }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _ffprobe(path: Path, ffprobe: str, *, frames: bool = False) -> dict[str, Any]:
    argv = [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json"]
    if frames:
        argv += ["-select_streams", "v:0", "-show_frames", "-show_entries", "frame=best_effort_timestamp_time,pkt_pts_time"]
    argv.append(str(path))
    proc = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    if proc.returncode:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.decode('utf-8', errors='replace')[-3000:]}")
    return json.loads(proc.stdout.decode("utf-8"))


def _stream_ref(probe: Mapping[str, Any], kind: str) -> dict[str, Any]:
    streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == kind]
    if len(streams) != 1:
        raise RuntimeError(f"expected one {kind} stream, found {len(streams)}")
    return streams[0]


def _video_es_sha(path: Path, ffmpeg: str) -> str:
    argv = [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-c:v", "copy",
            "-f", "hash", "-hash", "sha256", "pipe:1"]
    proc = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    if proc.returncode:
        raise RuntimeError(f"cannot hash video elementary stream: {proc.stderr.decode('utf-8', errors='replace')[-2000:]}")
    return proc.stdout.decode("ascii").strip().split("=", 1)[-1].lower()


def _decode_bytes(path: Path, ffmpeg: str, *, codec: str) -> bytes:
    argv = [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn", "-f", codec, "pipe:1"]
    proc = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    if proc.returncode:
        raise RuntimeError(f"FFmpeg audio readback failed for {path}: {proc.stderr.decode('utf-8', errors='replace')[-3000:]}")
    return proc.stdout


def _frame_clock(path: Path, ffprobe: str, fps_num: int, fps_den: int) -> tuple[list[dict[str, int]], dict[str, Any]]:
    probe = _ffprobe(path, ffprobe, frames=True)
    stream = _stream_ref(probe, "video")
    fps_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if not fps_text or Fraction(fps_text) != Fraction(fps_num, fps_den):
        raise RuntimeError(f"video FPS does not match the frozen clock: {fps_text}")
    frame_records = probe.get("frames", [])
    if not frame_records:
        raise RuntimeError("ffprobe found no decoded rendered video frames")
    output: list[dict[str, int]] = []
    for index, frame in enumerate(frame_records):
        raw_time = frame.get("best_effort_timestamp_time", frame.get("pkt_pts_time"))
        if raw_time is None:
            raise RuntimeError(f"rendered frame {index} has no exact ffprobe timestamp")
        timestamp = Fraction(str(raw_time))
        expected = Fraction(index * fps_den, fps_num)
        if timestamp != expected:
            raise RuntimeError(f"render timestamp {timestamp} at row {index} differs from frozen clock {expected}")
        output.append({"frame_index": index, "timestamp_num": timestamp.numerator, "timestamp_den": timestamp.denominator})
    return output, stream


def _reference(ref: Mapping[str, Any]) -> tuple[Path, np.ndarray, dict[str, Any]]:
    from scripts.audio_enhancement.audio_io import read_canonical_wav

    path = Path(str(ref.get("path", ""))).resolve(strict=True)
    samples, info = read_canonical_wav(path)
    if info.source_sample_rate != 16000 or info.source_channels != 1 or info.source_frames != samples.size:
        raise ValueError("planned natural WAV is not already canonical 16 kHz mono")
    if info.file_sha256 != ref.get("file_sha256") or info.pcm_float32_sha256 != ref.get("pcm_sha256"):
        raise ValueError(f"natural PCM/file hash differs from frozen plan at {path}")
    if info.frames != ref.get("sample_count"):
        raise ValueError(f"natural PCM sample count differs from frozen plan at {path}")
    return path, samples, {
        "path": str(path), "file_sha256": info.file_sha256, "pcm_sha256": info.pcm_float32_sha256,
        "sample_rate_hz": 16000, "channels": 1, "sample_count": info.frames,
        "subtype": "IEEE_FLOAT_32_LE",
    }


def _transform(x: np.ndarray, method: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    from scripts.audio_enhancement.baselines import apply_global_rms, apply_local_envelope_compression, waveform_rms

    config = method["config"]
    method_name = config.get("method")
    if method_name == "identity":
        y = x.copy()
        details: dict[str, Any] = {"operation": "exact_identity"}
    elif method_name == "rms":
        y, info = apply_global_rms(x, target_rms=float(config["target_rms"]), peak_ceiling=config.get("peak_ceiling"))
        details = {"operation": "fixed_global_rms", **info.__dict__}
    elif method_name == "rms_then_localcompress":
        pre, pre_info = apply_global_rms(x, target_rms=float(config["pre_rms_target"]), peak_ceiling=None)
        compressed, compressor_info = apply_local_envelope_compression(
            pre,
            sample_rate=int(config["sample_rate_hz"]),
            window_ms=float(config["local_window_ms"]),
            ratio=float(config["local_ratio"]),
            min_gain_db=float(config["local_min_gain_db"]),
            max_gain_db=float(config["local_max_gain_db"]),
            envelope_floor=float(config["envelope_floor"]),
        )
        y, post_info = apply_global_rms(compressed, target_rms=float(config["post_rms_target"]), peak_ceiling=None)
        details = {
            "operation": "global_rms_then_centered_local_envelope_compressor_then_global_rms",
            "pre_rms": pre_info.__dict__, "local_compressor": compressor_info.__dict__,
            "post_rms": post_info.__dict__,
        }
    else:
        raise ValueError(f"unknown frozen Stage-A method configuration: {method_name!r}")
    y = np.ascontiguousarray(y, dtype=np.float32)
    if y.shape != x.shape or not np.isfinite(y).all():
        raise RuntimeError("generator condition violated finite float32 and exact sample-count contract")
    details.update({"input_rms": waveform_rms(x), "output_rms": waveform_rms(y),
                    "peak_abs": float(np.max(np.abs(y.astype(np.float64))))})
    return y, details


def _write_f32_wav(path: Path, samples: np.ndarray) -> dict[str, Any]:
    from scripts.audio_enhancement.audio_io import write_float32_wav

    info = write_float32_wav(path, samples, sample_rate=16000, peak_ceiling=None, overwrite=False)
    return {
        "path": info.path, "file_sha256": info.file_sha256, "pcm_sha256": info.pcm_float32_sha256,
        "sample_rate_hz": info.sample_rate, "channels": 1, "sample_count": info.frames,
        "subtype": "IEEE_FLOAT_32_LE",
    }


def _run_generator(*, row: Mapping[str, Any], model: Mapping[str, Any], audio_path: Path,
                   output_video: Path, work: Path, args: argparse.Namespace) -> dict[str, Any]:
    helper = Path(args.generator_child).resolve(strict=True)
    generator = model["generator"]
    python = str(generator["python"])  # preserve the venv symlink; never resolve it
    wav2lip_root = str(generator["repo"])
    checkpoint = str(generator["checkpoint"])
    face = str(row["source_face"]["path"])
    child_cwd = work / "generator_cwd"
    child_cwd.mkdir(parents=True, exist_ok=False)
    (child_cwd / "temp").mkdir()
    argv = [python, str(helper), "--child", "--wav2lip-root-child", wav2lip_root,
            "--cwd", str(child_cwd), "--checkpoint", checkpoint, "--face-child", face,
            "--audio", str(audio_path), "--outfile", str(output_video), "--seed", str(row["seed"])]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONPATH": args.project_root + os.pathsep + env.get("PYTHONPATH", ""),
    })
    execution = run_logged(argv, cwd=child_cwd, env=env, stdout_path=work / "logs/generator.stdout.log",
                           stderr_path=work / "logs/generator.stderr.log", timeout=args.generator_timeout)
    if execution["returncode"] != 0 or not output_video.is_file() or output_video.stat().st_size == 0:
        raise RuntimeError(f"Wav2Lip child failed ({execution['returncode']}); stderr={execution.get('stderr_path')}")
    return execution


def _run_pipeline(muxed: Path, work: Path, args: argparse.Namespace, reference: str) -> tuple[Path, list[Any], list[np.ndarray], dict[str, Any]]:
    syncnet_root = Path(args.syncnet_root).resolve(strict=True)
    data_dir = work / "syncnet_tracking"
    data_dir.mkdir(parents=True, exist_ok=False)
    pipeline = syncnet_root / "run_pipeline.py"
    argv = [args.syncnet_python, str(pipeline), "--videofile", str(muxed), "--reference", reference,
            "--data_dir", str(data_dir), "--facedet_scale", "0.25", "--crop_scale", "0.40",
            "--min_track", "5", "--frame_rate", "25", "--num_failed_det", "25",
            "--min_face_size", "100", "--overwrite"]
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    pipeline_run = run_logged(argv, cwd=syncnet_root, env=env,
        stdout_path=work / "logs/syncnet_tracking.stdout.log", stderr_path=work / "logs/syncnet_tracking.stderr.log",
        timeout=args.syncnet_timeout)
    if pipeline_run["returncode"] != 0:
        raise RuntimeError(f"official SyncNet tracking failed ({pipeline_run['returncode']}); stderr={pipeline_run['stderr_path']}")
    tracks_path = data_dir / "pywork" / reference / "tracks.pckl"
    if not tracks_path.is_file():
        raise RuntimeError("official SyncNet run_pipeline did not write tracks.pckl")
    with tracks_path.open("rb") as stream:
        tracks = pickle.load(stream)
    if not isinstance(tracks, list) or not tracks:
        raise RuntimeError("official SyncNet found no selected face tracks")
    candidates: list[tuple[int, int, list[int], np.ndarray]] = []
    for track_index, record in enumerate(tracks):
        if not isinstance(record, Mapping):
            continue
        track = record.get("track", record)
        frames_array = np.asarray(track.get("frame", []), dtype=np.int64).reshape(-1)
        boxes = np.asarray(track.get("bbox", []), dtype=np.float32)
        if frames_array.size < 5 or boxes.shape != (frames_array.size, 4):
            continue
        frames = frames_array.tolist()
        if np.any(np.diff(frames_array) != 1):
            continue
        candidates.append((int(frames_array.size), track_index, frames, boxes))
    if not candidates:
        raise RuntimeError("official SyncNet tracker returned no contiguous crop track of at least five frames")
    _, selected_index, selected_frames, selected_boxes = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
    crop_dir = data_dir / "pycrop" / reference
    crop_path = crop_dir / f"{selected_index:05d}.avi"
    if not crop_path.is_file():
        paths = sorted(crop_dir.glob("*.avi"))
        if selected_index >= len(paths):
            raise RuntimeError(f"selected track crop AVI not found for track index {selected_index}")
        crop_path = paths[selected_index]
    return tracks_path, tracks, [selected_boxes], {
        "data_dir": data_dir, "reference": reference, "selected_track_index": selected_index,
        "selected_track_id": f"track_{selected_index:05d}", "selected_frames": selected_frames,
        "crop_avi_path": crop_path, "tracks_pickle_sha256": sha256_file(tracks_path),
        "pipeline_run": pipeline_run,
    }


def _make_primary_input(*, natural_path: Path, natural_meta: Mapping[str, Any],
                        silent_path: Path, muxed_path: Path, render_timestamps: list[dict[str, int]],
                        render_video_sha: str, render_video_es_sha: str, track_meta: Mapping[str, Any],
                        row_root: Path, args: argparse.Namespace, reference: str) -> dict[str, Any]:
    from scripts.experiments.cross_tfg_audio_head_20261003.stagea_pcm16_lineage import decode_and_slice, write_pcm16_wav

    frames = list(track_meta["selected_frames"])
    if not frames or any(b != a + 1 for a, b in zip(frames, frames[1:])):
        raise RuntimeError("selected official track source-clock frame indices are not contiguous")
    crop_video = row_root / "syncnet/selected_crop_video.mkv"
    crop_video.parent.mkdir(parents=True, exist_ok=False)
    crop_avi = Path(track_meta["crop_avi_path"])
    strip_argv = [args.ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(crop_avi),
                  "-map", "0:v:0", "-c:v", "copy", "-an", "-map_metadata", "-1", "-f", "matroska", str(crop_video)]
    stripped = run_logged(strip_argv, cwd=row_root, env=None,
        stdout_path=row_root / "logs/crop_video.stdout.log", stderr_path=row_root / "logs/crop_video.stderr.log",
        timeout=args.ffmpeg_timeout)
    if stripped["returncode"] != 0 or not crop_video.is_file():
        raise RuntimeError(f"selected crop video extraction failed ({stripped['returncode']})")
    crop_probe = _ffprobe(crop_video, args.ffprobe, frames=True)
    crop_stream = _stream_ref(crop_probe, "video")
    crop_decoded_frames = crop_probe.get("frames", [])
    if len(crop_decoded_frames) != len(frames):
        raise RuntimeError(f"selected crop has {len(crop_decoded_frames)} decoded frames but tracks.pckl maps {len(frames)}")
    crop_video_es_sha = _video_es_sha(crop_video, args.ffmpeg)
    selected_index = int(track_meta["selected_track_index"])
    selected_id = str(track_meta["selected_track_id"])
    fps_num, fps_den = 25, 1
    frame_map_rows = []
    for local_index, source_clock_index in enumerate(frames):
        stamp = Fraction(source_clock_index * fps_den, fps_num)
        if source_clock_index < len(render_timestamps):
            render_frame_index: int | None = source_clock_index
            frame_origin = "decoded_render"
            actual_stamp = Fraction(render_timestamps[source_clock_index]["timestamp_num"], render_timestamps[source_clock_index]["timestamp_den"])
            if actual_stamp != stamp:
                raise RuntimeError("selected track frame PTS differs from its decoded render timestamp")
        else:
            render_frame_index = None
            frame_origin = "nominal_clock_outside_render_decode"
        frame_map_rows.append({
            "crop_frame_index": local_index,
            "source_clock_frame_index": int(source_clock_index),
            "timestamp_num": stamp.numerator,
            "timestamp_den": stamp.denominator,
            "render_frame_index": render_frame_index,
            "frame_origin": frame_origin,
        })
    frame_map_path = row_root / "syncnet/selected_track_frame_map.json"
    crop_file_sha = sha256_file(crop_video)
    frame_map_payload = {
        "schema_version": "cross-tfg-track-frame-map-v2",
        "selected_crop_video_path": str(crop_video.resolve()),
        "selected_crop_video_file_sha256": crop_file_sha,
        "selected_crop_video_elementary_sha256": crop_video_es_sha,
        "selected_track_id": selected_id,
        "selected_track_index": selected_index,
        "render_video_path": str(muxed_path.resolve()),
        "render_video_file_sha256": sha256_file(muxed_path),
        "render_video_elementary_sha256": render_video_es_sha,
        "render_timestamps_sha256": sha256_file(row_root / "render/video_timestamps.json"),
        "fps_num": fps_num,
        "fps_den": fps_den,
        "frames": frame_map_rows,
    }
    write_json(frame_map_path, frame_map_payload)

    requested_start = int(frames[0]) * 640
    requested_stop = int(frames[-1] + 1) * 640
    natural_count = int(natural_meta["sample_count"])
    actual_stop = min(requested_stop, natural_count)
    if requested_start >= natural_count or actual_stop <= requested_start:
        raise RuntimeError("selected track has no natural PCM samples in its requested source-clock span")
    tail_clipped = actual_stop < requested_stop
    pcm_bytes, lineage = decode_and_slice(
        natural_path,
        start_sample=requested_start,
        stop_sample_exclusive=actual_stop,
        ffmpeg=args.ffmpeg,
    )
    pcm16_path = row_root / "syncnet/primary_pcm16_window.wav"
    pcm16_ref = write_pcm16_wav(pcm16_path, pcm_bytes)
    if lineage["source_file_sha256"] != natural_meta["file_sha256"] or lineage["pcm_sha256"] != pcm16_ref["pcm_sha256"]:
        raise RuntimeError("direct natural PCM16 lineage does not match the frozen source/window")

    primary_mkv = row_root / "syncnet/primary_crop_video_original_pcm16.mkv"
    mux_argv = [args.ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(crop_video), "-i", str(pcm16_path),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy", "-map_metadata", "0",
                "-f", "matroska", str(primary_mkv)]
    remux = run_logged(mux_argv, cwd=row_root, env=None,
        stdout_path=row_root / "logs/primary_syncnet_remux.stdout.log", stderr_path=row_root / "logs/primary_syncnet_remux.stderr.log",
        timeout=args.ffmpeg_timeout)
    if remux["returncode"] != 0 or not primary_mkv.is_file():
        raise RuntimeError(f"primary direct-PCM16 SyncNet remux failed ({remux['returncode']})")
    input_probe = _ffprobe(primary_mkv, args.ffprobe)
    input_video = _stream_ref(input_probe, "video")
    input_audio = _stream_ref(input_probe, "audio")
    if input_video.get("codec_name") != crop_stream.get("codec_name"):
        raise RuntimeError("primary SyncNet remux changed selected crop video codec")
    if input_audio.get("codec_name") != "pcm_s16le" or int(input_audio.get("sample_rate", 0)) != 16000 or int(input_audio.get("channels", 0)) != 1:
        raise RuntimeError("primary SyncNet input lost mono 16 kHz PCM16 audio")
    remux_video_es_sha = _video_es_sha(primary_mkv, args.ffmpeg)
    if remux_video_es_sha != crop_video_es_sha:
        raise RuntimeError("primary SyncNet remux changed the selected crop video elementary stream")
    remux_audio = _decode_bytes(primary_mkv, args.ffmpeg, codec="s16le")
    if remux_audio != pcm_bytes:
        raise RuntimeError("primary SyncNet remux PCM16 readback differs from direct natural-source window")

    source_start = int(frames[0])
    source_stop = int(frames[-1]) + 1
    start_pts = Fraction(source_start, 25)
    stop_pts = Fraction(source_stop, 25)
    score_refs = {
        "primary_pcm16_window": {
            **pcm16_ref,
            "source_float32_pcm_sha256": natural_meta["pcm_sha256"],
            "start_sample": requested_start,
            "requested_stop_sample_exclusive": requested_stop,
            "stop_sample_exclusive": actual_stop,
            "tail_clipped_to_natural_pcm": tail_clipped,
            "source_start_pts_num": start_pts.numerator,
            "source_start_pts_den": start_pts.denominator,
            "source_stop_pts_num": stop_pts.numerator,
            "source_stop_pts_den": stop_pts.denominator,
            "source_frame_start": source_start,
            "source_frame_stop_exclusive": source_stop,
            "fps_num": fps_num,
            "fps_den": fps_den,
            "conversion": {
                "ffmpeg_sha256": args.ffmpeg_sha256,
                "argv_template": ["-v", "error", "-i", "{natural_pcm_path}", "-map", "0:a:0", "-vn", "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1"],
                "code_sha256": args.pcm16_lineage_sha256,
                "operation": "ffmpeg_decode_full_source_then_slice_sample_window",
            },
        },
        "primary_syncnet_input": {
            "path": str(primary_mkv.resolve()),
            "file_sha256": sha256_file(primary_mkv),
            "container": "matroska",
            "video_codec": input_video.get("codec_name"),
            "video_elementary_sha256": remux_video_es_sha,
            "audio_codec": "pcm_s16le",
            "audio_pcm16_sha256": pcm16_ref["pcm_sha256"],
            "audio_sample_count": pcm16_ref["sample_count"],
            "sample_rate_hz": 16000,
            "channels": 1,
            "crop_video_path": str(crop_video.resolve()),
            "crop_video_file_sha256": crop_file_sha,
            "crop_video_elementary_sha256": crop_video_es_sha,
            "selected_track_id": selected_id,
            "selected_track_index": selected_index,
            "frame_map_path": str(frame_map_path.resolve()),
            "frame_map_sha256": sha256_file(frame_map_path),
        },
    }
    selected_crop_ref = {
        "path": str(crop_video.resolve()),
        "file_sha256": crop_file_sha,
        "video_elementary_sha256": crop_video_es_sha,
        "selected_track_id": selected_id,
        "selected_track_index": selected_index,
        "frame_map_path": str(frame_map_path.resolve()),
        "frame_map_sha256": sha256_file(frame_map_path),
        "crop_avi_path": str(crop_avi.resolve()),
        "crop_avi_file_sha256": sha256_file(crop_avi),
        "tracks_pickle_path": str(track_meta["data_dir"] / "pywork" / reference / "tracks.pckl"),
        "tracks_pickle_sha256": track_meta["tracks_pickle_sha256"],
        "source_clock_frame_start": source_start,
        "source_clock_frame_stop_exclusive": source_stop,
    }
    extra = {
        "frame_map_path": str(frame_map_path.resolve()),
        "frame_map_sha256": sha256_file(frame_map_path),
        "selected_crop_ref": selected_crop_ref,
        "primary_refs": score_refs,
        "primary_mkv": primary_mkv,
        "pcm16_bytes": pcm_bytes,
        "crop_video": crop_video,
        "crop_video_file_sha": crop_file_sha,
        "crop_video_es_sha": crop_video_es_sha,
        "crop_video_frame_count": len(crop_decoded_frames),
        "selected_track_index": selected_index,
        "selected_track_id": selected_id,
        "frame_map_payload": frame_map_payload,
        "mux_log": remux,
    }
    return extra


def _score_direct(primary: Mapping[str, Any], track_meta: Mapping[str, Any], row_root: Path,
                  args: argparse.Namespace, reference: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    evaluator = Path(args.syncnet_evaluator).resolve(strict=True)
    run_dir = row_root / "syncnet_direct_eval"
    run_dir.mkdir(parents=True, exist_ok=False)
    raw_npz = row_root / "score/raw_matrix_arrays.npz"
    receipt_path = row_root / "score/syncnet_evaluator_receipt.json"
    raw_npz.parent.mkdir(parents=True, exist_ok=True)
    pcm_ref = primary["primary_refs"]["primary_pcm16_window"]
    video_frame_count = int(primary["crop_video_frame_count"])
    argv = [args.syncnet_python, str(evaluator), "--syncnet-root", args.syncnet_root,
            "--checkpoint", args.syncnet_checkpoint, "--input-mkv", primary["primary_mkv"],
            "--tmp-dir", str(run_dir), "--output-npz", str(raw_npz), "--output-json", str(receipt_path),
            "--reference", reference, "--expected-pcm16-sha256", pcm_ref["pcm_sha256"],
            "--expected-sample-count", str(pcm_ref["sample_count"]),
            "--expected-video-frame-count", str(video_frame_count), "--batch-size", "20", "--vshift", "15"]
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"})
    execution = run_logged(argv, cwd=Path(args.syncnet_root), env=env,
        stdout_path=row_root / "logs/syncnet_score.stdout.log", stderr_path=row_root / "logs/syncnet_score.stderr.log",
        timeout=args.syncnet_timeout)
    if execution["returncode"] != 0 or not raw_npz.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"direct official SyncNet evaluate failed ({execution['returncode']}); stderr={execution.get('stderr_path')}")
    with np.load(raw_npz, allow_pickle=False) as archive:
        matrix = np.ascontiguousarray(archive["distance_by_query_lag"], dtype="<f4")
        lag = np.ascontiguousarray(archive["lag_offsets_frames"], dtype="<i4")
    if lag.tolist() != list(range(-15, 16)):
        raise RuntimeError("official evaluator returned unexpected lag axis")
    count = matrix.shape[0]
    frame_map = primary["frame_map_payload"]["frames"]
    if count != max(0, min(video_frame_count, int(pcm_ref["sample_count"]) // 640) - 5):
        raise RuntimeError("official query count differs from min(video_frames,audio_samples//640)-5")
    if count > len(frame_map):
        raise RuntimeError("official query count exceeds selected crop source-frame map")
    arrays = {
        "query_frame_index": np.asarray([frame_map[i]["source_clock_frame_index"] for i in range(count)], dtype="<i4"),
        "query_timestamp_num": np.asarray([frame_map[i]["timestamp_num"] for i in range(count)], dtype="<i8"),
        "query_timestamp_den": np.asarray([frame_map[i]["timestamp_den"] for i in range(count)], dtype="<i8"),
        "track_frame_index": np.asarray([frame_map[i]["source_clock_frame_index"] for i in range(count)], dtype="<i4"),
        "valid_query_mask": np.ones((count,), dtype="bool"),
        "distance_by_query_lag": matrix,
        "lag_offsets_frames": lag,
    }
    return arrays, {"execution": execution, "receipt": read_json(receipt_path), "raw_npz_path": str(raw_npz.resolve()),
                    "raw_npz_sha256": sha256_file(raw_npz)}


def _render_and_score(row: Mapping[str, Any], model: Mapping[str, Any], method: Mapping[str, Any] | None,
                      natural_path: Path, natural_samples: np.ndarray, natural_ref: Mapping[str, Any],
                      row_root: Path, args: argparse.Namespace, *, qc_repeat: bool = False) -> dict[str, Any]:
    from scripts.audio_enhancement.audio_io import pcm_float32_sha256

    row_root.mkdir(parents=True, exist_ok=False)
    (row_root / "logs").mkdir()
    (row_root / "audio").mkdir()
    (row_root / "render").mkdir()
    if method is None:
        enhanced = natural_samples.copy()
        transform_diagnostics = {
            "operation": "identity repeat",
            "input_rms": float(np.sqrt(np.mean(natural_samples.astype(np.float64) ** 2))),
        }
    else:
        enhanced, transform_diagnostics = _transform(natural_samples, method)
    enhanced_path = row_root / "audio/generator_condition_f32.wav"
    enhanced_ref = _write_f32_wav(enhanced_path, enhanced)
    if enhanced_ref["sample_count"] != natural_ref["sample_count"]:
        raise RuntimeError("enhanced generator condition changed sample count")
    generator_video = row_root / "render/generated_video.mp4"
    generation = _run_generator(row=row, model=model, audio_path=enhanced_path,
                                 output_video=generator_video, work=row_root, args=args)
    silent = row_root / "render/silent_video.mkv"
    strip_argv = [args.ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(generator_video),
                  "-map", "0:v:0", "-c:v", "copy", "-an", "-map_metadata", "-1", "-f", "matroska", str(silent)]
    strip = run_logged(strip_argv, cwd=row_root, env=None,
        stdout_path=row_root / "logs/strip_audio.stdout.log", stderr_path=row_root / "logs/strip_audio.stderr.log",
        timeout=args.ffmpeg_timeout)
    if strip["returncode"] != 0 or not silent.is_file():
        raise RuntimeError("could not remove Wav2Lip's generated audio stream")
    silent_frames, _ = _frame_clock(silent, args.ffprobe, 25, 1)
    muxed = row_root / "render/natural_pcm_f32_mux.mkv"
    mux_argv = [args.ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(silent), "-i", str(natural_path),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy", "-map_metadata", "0",
                "-f", "matroska", str(muxed)]
    mux = run_logged(mux_argv, cwd=row_root, env=None,
        stdout_path=row_root / "logs/natural_pcm_mux.stdout.log", stderr_path=row_root / "logs/natural_pcm_mux.stderr.log",
        timeout=args.ffmpeg_timeout)
    if mux["returncode"] != 0 or not muxed.is_file():
        raise RuntimeError("could not losslessly mux the original canonical float32 PCM")
    mux_probe = _ffprobe(muxed, args.ffprobe)
    mux_video_stream = _stream_ref(mux_probe, "video")
    mux_audio_stream = _stream_ref(mux_probe, "audio")
    if mux_probe.get("format", {}).get("format_name", "").find("matroska") < 0:
        raise RuntimeError("original audio delivery mux is not Matroska")
    if mux_audio_stream.get("codec_name") != "pcm_f32le" or int(mux_audio_stream.get("sample_rate", 0)) != 16000 or int(mux_audio_stream.get("channels", 0)) != 1:
        raise RuntimeError("original audio delivery mux is not mono 16 kHz PCM_F32LE")
    natural_bytes = _decode_bytes(natural_path, args.ffmpeg, codec="f32le")
    mux_audio_bytes = _decode_bytes(muxed, args.ffmpeg, codec="f32le")
    if mux_audio_bytes != natural_bytes or sha256_bytes(mux_audio_bytes) != natural_ref["pcm_sha256"]:
        raise RuntimeError("original PCM_F32LE mux readback is not byte-identical to natural PCM")
    mux_video_es_sha = _video_es_sha(muxed, args.ffmpeg)
    silent_video_es_sha = _video_es_sha(silent, args.ffmpeg)
    if mux_video_es_sha != silent_video_es_sha:
        raise RuntimeError("natural audio mux changed the generated video elementary stream")
    timestamps_path = row_root / "render/video_timestamps.json"
    write_json(timestamps_path, {
        "schema_version": "cross_tfg_video_clock_v1",
        "frame_count": len(silent_frames),
        "fps_num": 25, "fps_den": 1,
        "frames": silent_frames,
    })
    crop_policy = {
        "schema_version": "cross_tfg_native_official_full_frame_crop_policy_v1",
        "roi_policy": "full decoded rendered frame; SyncNet native face track is separately selected and bound",
        "fps_num": 25, "fps_den": 1,
    }
    crop_policy_sha = sha256_bytes(canonical(crop_policy))
    roi_path = row_root / "render/crop_roi.json"
    first_probe = _ffprobe(silent, args.ffprobe)
    full_width, full_height = int(mux_video_stream.get("width", 0)), int(mux_video_stream.get("height", 0))
    if full_width <= 0 or full_height <= 0:
        raise RuntimeError("generated video dimensions are invalid")
    write_json(roi_path, {
        "schema_version": "tfg-crop-roi-v1", "crop_policy_sha256": crop_policy_sha,
        "frames": [{**stamp, "x": 0.0, "y": 0.0, "width": float(full_width), "height": float(full_height)} for stamp in silent_frames],
    })
    reference = f"stagea_{row['row_id']}"
    tracks_path, tracks, _, track_meta = _run_pipeline(muxed, row_root, args, reference)
    primary = _make_primary_input(
        natural_path=natural_path, natural_meta=natural_ref, silent_path=silent, muxed_path=muxed,
        render_timestamps=silent_frames, render_video_sha=sha256_file(muxed),
        render_video_es_sha=mux_video_es_sha, track_meta=track_meta,
        row_root=row_root, args=args, reference=reference,
    )
    arrays, score_run = _score_direct(primary, track_meta, row_root, args, reference + "_primary")
    raw_npz_path = Path(primary["primary_mkv"]).parent / "raw_score_arrays.npz"
    if raw_npz_path.exists():
        raise FileExistsError(raw_npz_path)
    np.savez_compressed(raw_npz_path, **arrays)
    array_meta = {
        key: {"dtype": array.dtype.str, "shape": list(array.shape), "array_sha256": sha256_bytes(np.ascontiguousarray(array).tobytes(order="C"))}
        for key, array in arrays.items()
    }
    array_ref = {"path": str(raw_npz_path.resolve()), "file_sha256": sha256_file(raw_npz_path), "arrays": array_meta}
    frame_sidecar_sha = sha256_file(timestamps_path)
    row_result = {
        **{key: row.get(key) for key in IDENTITY_FIELDS},
        "schema_version": SCHEMA,
        "status": "PASS",
        "failure_reason": None,
        "natural_pcm_sha256": natural_ref["pcm_sha256"],
        "score_audio_sha256": natural_ref["pcm_sha256"],
        "source_face_sha256": row["source_face"]["file_sha256"],
        "source_video_sha256": row["source_video"]["file_sha256"],
        "generator_pcm": enhanced_ref,
        "generator_pcm_sha256": enhanced_ref["pcm_sha256"],
        "transform_diagnostics": transform_diagnostics,
        "render": {
            "status": "PASS",
            "generated_video_path": str(generator_video.resolve()),
            "generated_video_sha256": sha256_file(generator_video),
            "silent_video_path": str(silent.resolve()), "silent_video_sha256": sha256_file(silent),
            "muxed_video_path": str(muxed.resolve()), "muxed_video_sha256": sha256_file(muxed),
            "video_stream_sha256": mux_video_es_sha,
            "generator_pcm_sha256": enhanced_ref["pcm_sha256"],
            "fps_num": 25, "fps_den": 1, "frame_count": len(silent_frames),
            "crop_frame_count": len(silent_frames),
            "timestamps_path": str(timestamps_path.resolve()), "timestamps_sha256": frame_sidecar_sha,
            "crop_timestamps_sha256": frame_sidecar_sha,
            "crop_roi_path": str(roi_path.resolve()), "crop_roi_sha256": sha256_file(roi_path),
            "crop_policy_sha256": crop_policy_sha,
            "selected_track_id": primary["selected_track_id"],
            "selected_track_index": primary["selected_track_index"],
            "selected_track_crop": primary["selected_crop_ref"],
            "generation": generation, "strip_audio": strip,
            "natural_pcm_mux": mux, "syncnet_tracking": track_meta["pipeline_run"],
            "render_video_path": str(muxed.resolve()),
            "render_video_file_sha256": sha256_file(muxed),
        },
        "mux_audio_readback": {
            "status": "PASS", "codec": "pcm_f32le", "sample_rate_hz": 16000, "channels": 1,
            "sample_count": len(natural_bytes) // 4,
            "decoded_float32_pcm_sha256": sha256_bytes(natural_bytes),
            "matches_natural_canonical_pcm": True,
            "video_stream_identity_verified": True,
        },
        "score": {
            "status": "PASS",
            "score_model_sha256": model["score_model_sha256"],
            "preprocessing_sha256": model["preprocessing_sha256"],
            "score_audio_pcm_sha256": natural_ref["pcm_sha256"],
            "score_preprocessed_pcm16_sha256": primary["primary_refs"]["primary_pcm16_window"]["pcm_sha256"],
            "score_preprocessed_sample_count": primary["primary_refs"]["primary_pcm16_window"]["sample_count"],
            **primary["primary_refs"],
            "raw_matrix_npz": array_ref,
            "official_evaluator": score_run,
        },
    }
    row_result["_internal_arrays"] = arrays
    row_result["_internal_video_path"] = str(silent.resolve())
    row_result["_internal_matrix_sha256"] = array_meta["distance_by_query_lag"]["array_sha256"]
    row_result["_internal_crop_policy_sha256"] = crop_policy_sha
    return row_result


def _identity_base(row: Mapping[str, Any], natural_ref: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **{key: row.get(key) for key in IDENTITY_FIELDS},
        "schema_version": SCHEMA,
        "status": "FAILED",
        "failure_reason": "not_started",
        "natural_pcm_sha256": natural_ref.get("pcm_sha256"),
        "score_audio_sha256": natural_ref.get("pcm_sha256"),
        "source_face_sha256": row.get("source_face", {}).get("file_sha256"),
        "source_video_sha256": row.get("source_video", {}).get("file_sha256"),
        "render": {"status": "NOT_STARTED"},
        "mux_audio_readback": {"status": "NOT_STARTED"},
        "score": {"status": "NOT_RUN"},
    }


def _query_keys(arrays: Mapping[str, np.ndarray]) -> list[tuple[int, int, int, int]]:
    return [
        (int(f), Fraction(int(n), int(d)).numerator, Fraction(int(n), int(d)).denominator, int(t))
        for f, n, d, t in zip(arrays["query_frame_index"], arrays["query_timestamp_num"], arrays["query_timestamp_den"], arrays["track_frame_index"])
    ]


def _support_sha(support_id: str, keys: Sequence[tuple[int, int, int, int]]) -> str:
    return sha256_bytes(canonical({"support_id": support_id, "keys": [list(key) for key in keys]}))


def _metric_summary(matrix: np.ndarray, indices: Sequence[int], lag: np.ndarray, anchor_column: int) -> tuple[list[float], dict[str, Any]]:
    import torch

    if not indices:
        return [], {"status": "EMPTY", "reason": "no common valid queries", "B": None, "Dmin": None, "D0": None,
                    "Cbest": None, "Canchor": None, "best_lag_frames": None, "offset_frames": None,
                    "official_av_offset_frames": None, "anchor_column": None, "zero_lag_column": None}
    selected = np.ascontiguousarray(matrix[np.asarray(indices, dtype=np.int64)], dtype=np.float32)
    torch.set_num_threads(1)
    curve_tensor = torch.mean(torch.from_numpy(np.ascontiguousarray(selected.T)), dim=1)
    min_result = torch.min(curve_tensor, dim=0)
    median = torch.median(curve_tensor)
    dmin = min_result.values
    d0 = curve_tensor[15]
    danchor = curve_tensor[anchor_column]
    best_column = int(min_result.indices.item())
    best_lag = int(lag[best_column])
    curve = curve_tensor.numpy().astype(np.float32)
    return curve.tolist(), {
        "B": float(np.float32(median.item())),
        "Dmin": float(np.float32(dmin.item())),
        "D0": float(np.float32(d0.item())),
        "Cbest": float(np.float32((median - dmin).item())),
        "Canchor": float(np.float32((median - danchor).item())),
        "best_lag_frames": best_lag,
        "offset_frames": -best_lag,
        "official_av_offset_frames": -best_lag,
        "anchor_column": int(anchor_column),
        "zero_lag_column": 15,
    }


def _write_partial_results(out: Path, rows: list[dict[str, Any]]) -> None:
    path = out / "results.partial.jsonl"
    tmp = out / ".results.partial.jsonl.tmp"
    with tmp.open("wb") as stream:
        for row in rows:
            public = {key: value for key, value in row.items() if not key.startswith("_internal_")}
            if "_internal_arrays" in public:
                public.pop("_internal_arrays", None)
            stream.write(canonical(public) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _finish_supports(plan: Mapping[str, Any], results: list[dict[str, Any]]) -> None:
    torch_anchor = int(plan["protocol"]["syncnet"]["anchor_column"])
    by_id = {row["row_id"]: row for row in results}
    plan_by_id = {row["row_id"]: row for row in plan["rows"]}
    contexts: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for row in plan["rows"]:
        key = (row["split"], row["sample_id"], row["speaker_id"], row["model_id"], row["face_id"], row["seed"])
        contexts[key].append(row["row_id"])
    for _, row_ids in contexts.items():
        planned_methods = {plan_by_id[row_id]["method_id"]: row_id for row_id in row_ids}
        all_pass = len(planned_methods) == len(plan["methods"]) and all(by_id[row_id]["status"] == "PASS" for row_id in planned_methods.values())
        if all_pass:
            indexes = {method: {key: i for i, key in enumerate(_query_keys(by_id[row_id]["_internal_arrays"]))
                                if bool(by_id[row_id]["_internal_arrays"]["valid_query_mask"][i])}
                       for method, row_id in planned_methods.items()}
            common = set.intersection(*(set(indexes[method]) for method in planned_methods)) if indexes else set()
            full_keys = sorted(common, key=lambda key: (key[0], Fraction(key[1], key[2]), key[3]))
        else:
            indexes = {}
            full_keys = []
        for method, row_id in planned_methods.items():
            result = by_id[row_id]
            if result["status"] != "PASS":
                continue
            arrays = result["_internal_arrays"]
            local_index = {key: i for i, key in enumerate(_query_keys(arrays))}
            support_keys = {
                "full": list(full_keys),
                "guard20": list(full_keys[20:-20]) if len(full_keys) >= 41 else [],
            }
            curves, supports, metrics = {}, {}, {}
            for support_id, keys in support_keys.items():
                indices = [local_index[key] for key in keys]
                mask_sha = _support_sha(support_id, keys)
                curve, metric = _metric_summary(arrays["distance_by_query_lag"], indices, arrays["lag_offsets_frames"], torch_anchor)
                curves[support_id] = curve
                supports[support_id] = {
                    "query_count": len(keys), "support_mask_sha256": mask_sha,
                    "eligible": len(keys) >= (1 if support_id == "full" else 41),
                }
                metrics[support_id] = metric
            result["score"]["lag_curve_by_support"] = curves
            result["score"]["support_by_support"] = supports
            result["score"]["metrics_by_support"] = metrics


def _decode_rgb(path: Path, ffmpeg: str, width: int, height: int) -> tuple[np.ndarray, int]:
    argv = [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    proc = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode:
        return np.empty((0, 0, 0, 0), dtype=np.uint8), -1
    raw = proc.stdout
    frame_bytes = width * height * 3
    if frame_bytes <= 0 or len(raw) % frame_bytes:
        return np.empty((0, 0, 0, 0), dtype=np.uint8), -1
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3).copy(), 0


def _repeat_record(plan: Mapping[str, Any], baseline: Mapping[str, Any], natural_path: Path,
                   natural_samples: np.ndarray, natural_ref: Mapping[str, Any], model: Mapping[str, Any],
                   out: Path, args: argparse.Namespace) -> dict[str, Any]:
    declaration = plan["generation_repeat_qc"][0]
    row = next(item for item in plan["rows"] if item["row_id"] == declaration["reference_row_id"])
    repeat_root = out / "generation_repeat_qc" / row["row_id"]
    repeated = _render_and_score(row, model, None, natural_path, natural_samples, natural_ref,
                                 repeat_root, args, qc_repeat=True)
    ref_render = baseline["render"]
    rep_render = repeated["render"]
    raw_ref = repeated["score"]["raw_matrix_npz"]
    arrays = repeated["_internal_arrays"]
    baseline_arrays = baseline["_internal_arrays"]
    exact_video = rep_render["video_stream_sha256"] == ref_render["video_stream_sha256"]
    exact_timestamps = rep_render["timestamps_sha256"] == ref_render["timestamps_sha256"]
    raw_sha = sha256_bytes(np.ascontiguousarray(arrays["distance_by_query_lag"], dtype="<f4").tobytes())
    baseline_sha = sha256_bytes(np.ascontiguousarray(baseline_arrays["distance_by_query_lag"], dtype="<f4").tobytes())
    exact_distance = raw_sha == baseline_sha
    rep_keys = _query_keys(arrays)
    base_keys = _query_keys(baseline_arrays)
    exact_query = rep_keys == base_keys and np.array_equal(arrays["lag_offsets_frames"], baseline_arrays["lag_offsets_frames"])
    width = int(_ffprobe(Path(rep_render["silent_video_path"]), args.ffprobe)["streams"][0]["width"])
    height = int(_ffprobe(Path(rep_render["silent_video_path"]), args.ffprobe)["streams"][0]["height"])
    va, _ = _decode_rgb(Path(ref_render["silent_video_path"]), args.ffmpeg, width, height)
    vb, _ = _decode_rgb(Path(rep_render["silent_video_path"]), args.ffmpeg, width, height)
    max_frame_diff: float | None = None
    if va.shape == vb.shape and va.size:
        max_frame_diff = float(np.max(np.abs(va.astype(np.int16) - vb.astype(np.int16))))
    max_distance_diff: float | None = None
    if arrays["distance_by_query_lag"].shape == baseline_arrays["distance_by_query_lag"].shape:
        max_distance_diff = float(np.max(np.abs(arrays["distance_by_query_lag"].astype(np.float64) - baseline_arrays["distance_by_query_lag"].astype(np.float64))))
    common = sorted(set(rep_keys) & set(base_keys), key=lambda key: (key[0], Fraction(key[1], key[2]), key[3]))
    cbest_delta = None
    if common:
        rep_idx = {key: i for i, key in enumerate(rep_keys)}
        base_idx = {key: i for i, key in enumerate(base_keys)}
        _, rep_metric = _metric_summary(arrays["distance_by_query_lag"], [rep_idx[key] for key in common], arrays["lag_offsets_frames"], 15)
        _, base_metric = _metric_summary(baseline_arrays["distance_by_query_lag"], [base_idx[key] for key in common], baseline_arrays["lag_offsets_frames"], 15)
        cbest_delta = abs(float(rep_metric["Cbest"]) - float(base_metric["Cbest"]))
    exact_all = exact_video and exact_timestamps and exact_distance and exact_query
    derived_status = "PASS" if exact_all else "NONDETERMINISTIC"
    qc_attempt = {
        "status": "PASS",
        "video_path": rep_render["silent_video_path"],
        "video_sha256": rep_render["silent_video_sha256"],
        "elementary_video_sha256": rep_render["video_stream_sha256"],
        "timestamp_sidecar_path": rep_render["timestamps_path"],
        "timestamp_sidecar_sha256": rep_render["timestamps_sha256"],
        "crop_policy_sha256": rep_render["crop_policy_sha256"],
        "raw_matrix_npz": raw_ref,
        "raw_distance_matrix_sha256": raw_sha,
    }
    return {
        "run_id": plan["run_id"], "model_id": declaration["model_id"],
        "input_sample_id": declaration["input_sample_id"], "face_id": declaration["face_id"],
        "seed": declaration["seed"], "deterministic": declaration["deterministic"],
        "followup_protocol": declaration["followup_protocol"], "attempts": [qc_attempt],
        "status": derived_status,
        "comparison": {
            "exact_video": exact_video, "exact_timestamps": exact_timestamps,
            "exact_query_map": exact_query, "exact_distance": exact_distance,
            "max_frame_absdiff": max_frame_diff,
            "max_distance_absdiff": max_distance_diff,
            "abs_delta_Cbest": cbest_delta,
        },
    }


def run(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan).resolve(strict=True)
    rows_path = Path(args.planned_rows).resolve(strict=True)
    out = Path(args.output_root).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty Stage-A output root: {out}")
    out.mkdir(parents=True, exist_ok=True)
    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw.decode("utf-8"))
    if plan.get("status") != "FROZEN" or plan.get("plan_sha256") != sha256_bytes(canonical({k: v for k, v in plan.items() if k != "plan_sha256"})):
        raise ValueError("frozen plan canonical payload seal is invalid")
    if sha256_file(plan_path) != args.expected_plan_raw_sha256:
        raise ValueError("caller plan raw SHA does not match exact frozen plan file")
    planned_rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(planned_rows) != len(plan["rows"]) or [r["row_id"] for r in planned_rows] != [r["row_id"] for r in plan["rows"]]:
        raise ValueError("planned_rows.jsonl differs from exact ordered plan rows")
    models = {item["model_id"]: item for item in plan["models"]}
    methods = {item["method_id"]: item for item in plan["methods"]}
    checkpoint = Path(args.syncnet_checkpoint).resolve(strict=True)
    if sha256_file(checkpoint) != plan["models"][0]["score_model_sha256"]:
        raise ValueError("SyncNet checkpoint differs from frozen plan hash")
    ffmpeg_path = Path(args.ffmpeg).resolve(strict=True)
    if sha256_file(ffmpeg_path) != plan["protocol"]["syncnet"]["primary_audio_input"]["ffmpeg_sha256"]:
        raise ValueError("FFmpeg executable differs from frozen PCM16 conversion pin")
    import sys as _sys
    _sys.path.insert(0, args.project_root)
    from scripts.audio_enhancement.audio_io import read_canonical_wav

    (out / "control").mkdir()
    write_json(out / "control/run_status.json", {
        "schema_version": SCHEMA, "status": "RUNNING", "run_id": plan["run_id"],
        "plan_path": str(plan_path), "plan_raw_sha256": sha256_file(plan_path),
        "planned_rows_path": str(rows_path), "planned_rows_sha256": sha256_file(rows_path),
        "started_unix": time.time(), "pid": os.getpid(), "completed_rows": 0,
        "planned_rows": len(planned_rows), "current_row_id": None,
    })
    result_by_id: dict[str, dict[str, Any]] = {}
    ordered_results: list[dict[str, Any]] = []
    source_cache: dict[str, tuple[np.ndarray, dict[str, Any], Path]] = {}
    for index, row in enumerate(planned_rows):
        model = models[row["model_id"]]
        method = methods[row["method_id"]]
        source_path_text = row["natural_pcm"]["path"]
        try:
            if source_path_text not in source_cache:
                path = Path(source_path_text).resolve(strict=True)
                samples, info = read_canonical_wav(path)
                ref = {
                    "path": str(path), "file_sha256": info.file_sha256,
                    "pcm_sha256": info.pcm_float32_sha256, "sample_rate_hz": 16000,
                    "channels": 1, "sample_count": info.frames, "subtype": "IEEE_FLOAT_32_LE",
                }
                if ref["file_sha256"] != row["natural_pcm"]["file_sha256"] or ref["pcm_sha256"] != row["natural_pcm"]["pcm_sha256"]:
                    raise RuntimeError("plan source input hash mismatch")
                source_cache[source_path_text] = (samples, ref, path)
            samples, natural_ref, natural_path = source_cache[source_path_text]
            write_json(out / "control/run_status.json", {
                "schema_version": SCHEMA, "status": "RUNNING", "run_id": plan["run_id"],
                "plan_path": str(plan_path), "plan_raw_sha256": sha256_file(plan_path),
                "planned_rows_path": str(rows_path), "planned_rows_sha256": sha256_file(rows_path),
                "started_unix": time.time(), "pid": os.getpid(), "completed_rows": index,
                "planned_rows": len(planned_rows), "current_row_id": row["row_id"],
            })
            result = _render_and_score(
                row, model, method, natural_path, samples, natural_ref,
                out / "rows" / row["row_id"], args,
                qc_repeat=False,
            )
        except Exception as exc:
            natural_ref = row.get("natural_pcm", {})
            result = _identity_base(row, natural_ref)
            result["failure_reason"] = f"{type(exc).__name__}: {exc}"
            result["failure_traceback"] = traceback.format_exc()
            failed_dir = out / "row_failures" / row["row_id"]
            failed_dir.mkdir(parents=True, exist_ok=True)
            write_json(failed_dir / "failure.json", {"row_id": row["row_id"], "error": str(exc), "traceback": result["failure_traceback"], "unix_time": time.time()})
        result_by_id[row["row_id"]] = result
        ordered_results.append(result)
        _write_partial_results(out, ordered_results)
        write_json(out / "control/run_status.json", {
            "schema_version": SCHEMA, "status": "RUNNING", "run_id": plan["run_id"],
            "plan_path": str(plan_path), "plan_raw_sha256": sha256_file(plan_path),
            "planned_rows_path": str(rows_path), "planned_rows_sha256": sha256_file(rows_path),
            "started_unix": time.time(), "pid": os.getpid(), "completed_rows": index + 1,
            "planned_rows": len(planned_rows), "current_row_id": row["row_id"],
            "last_row_status": result["status"],
        })

    _finish_supports(plan, ordered_results)
    _write_partial_results(out, ordered_results)
    final_results = out / "results.jsonl"
    partial_path = out / "results.partial.jsonl"
    os.replace(partial_path, final_results)

    # One same-seed identity repeat per model, outside efficacy rows.
    repeat_decl = plan["generation_repeat_qc"][0]
    reference_row = result_by_id[repeat_decl["reference_row_id"]]
    repeat_failure: str | None = None
    try:
        repeat_plan_row = next(r for r in plan["rows"] if r["row_id"] == repeat_decl["reference_row_id"])
        repeat_samples, repeat_natural_ref, repeat_natural_path = source_cache[repeat_plan_row["natural_pcm"]["path"]]
        repeat_record = _repeat_record(plan, reference_row, repeat_natural_path, repeat_samples,
                                       repeat_natural_ref, models[repeat_plan_row["model_id"]], out, args)
    except Exception as exc:
        repeat_failure = f"{type(exc).__name__}: {exc}"
        repeat_record = {
            "run_id": plan["run_id"], "model_id": repeat_decl["model_id"],
            "input_sample_id": repeat_decl["input_sample_id"], "face_id": repeat_decl["face_id"],
            "seed": repeat_decl["seed"], "deterministic": repeat_decl["deterministic"],
            "followup_protocol": repeat_decl["followup_protocol"], "status": "FAILED",
            "attempts": [{"status": "FAILED", "failure_reason": repeat_failure}],
            "comparison": {"exact_video": False, "exact_timestamps": False, "exact_query_map": False,
                           "exact_distance": False, "max_frame_absdiff": None, "max_distance_absdiff": None,
                           "abs_delta_Cbest": None},
        }
    repeat_path = out / "generation_repeat_qc.jsonl"
    repeat_path.write_bytes(canonical(repeat_record) + b"\n")

    status_counts = Counter(result["status"] for result in ordered_results)
    counts_by_model_method = []
    for model_id in models:
        for method_id in methods:
            subset = [r for r in ordered_results if r["model_id"] == model_id and r["method_id"] == method_id]
            counts_by_model_method.append({
                "model_id": model_id, "method_id": method_id, "planned_rows": len(subset), "result_rows": len(subset),
                "status_counts": {status: sum(r["status"] == status for r in subset) for status in ("PASS", "FAILED", "INCOMPLETE")},
            })
    result_manifest = {
        "schema_version": "cross-tfg-audio-eval-v1",
        "run_id": plan["run_id"],
        "status": "PASS" if status_counts["FAILED"] == 0 and repeat_record["status"] in ("PASS", "NONDETERMINISTIC") else "INCOMPLETE",
        "plan_sha256": plan["plan_sha256"],
        "plan_raw_sha256": sha256_file(plan_path),
        "results_jsonl_sha256": sha256_file(final_results),
        "generation_repeat_qc_jsonl_sha256": sha256_file(repeat_path),
        "planned_rows": len(planned_rows), "result_rows": len(ordered_results),
        "generation_repeat_qc_rows": len(plan["generation_repeat_qc"]),
        "status_counts": {status: int(status_counts.get(status, 0)) for status in ("PASS", "FAILED", "INCOMPLETE")},
        "counts_by_model_method": counts_by_model_method,
        "failure_reasons": [{"row_id": result["row_id"], "reason": result["failure_reason"]}
                             for result in ordered_results if result["status"] != "PASS"],
        "repeat_qc_status": repeat_record["status"],
    }
    result_manifest["manifest_sha256"] = sha256_bytes(canonical({k: v for k, v in result_manifest.items() if k != "manifest_sha256"}))
    write_json(out / "result_manifest.json", result_manifest)
    write_json(out / "control/run_status.json", {
        "schema_version": SCHEMA, "status": "COMPLETE" if result_manifest["status"] == "PASS" else "COMPLETE_WITH_FAILURES",
        "run_id": plan["run_id"], "plan_path": str(plan_path), "plan_raw_sha256": sha256_file(plan_path),
        "planned_rows_path": str(rows_path), "planned_rows_sha256": sha256_file(rows_path),
        "ended_unix": time.time(), "pid": os.getpid(), "completed_rows": len(ordered_results),
        "planned_rows": len(planned_rows), "status_counts": result_manifest["status_counts"],
        "repeat_qc_status": repeat_record["status"], "result_manifest_path": str((out / "result_manifest.json").resolve()),
        "result_manifest_sha256": sha256_file(out / "result_manifest.json"),
    })
    print(json.dumps({"status": result_manifest["status"], "planned_rows": len(planned_rows),
                      "status_counts": result_manifest["status_counts"], "output_root": str(out),
                      "manifest": str(out / "result_manifest.json")}, ensure_ascii=False, sort_keys=True))
    return 0 if result_manifest["status"] == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--planned-rows", required=True)
    parser.add_argument("--expected-plan-raw-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--project-root", default="/root/tts-research-system-20261001/cross_tfg_audio_head_20261003")
    parser.add_argument("--wav2lip-python", default="/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python")
    parser.add_argument("--wav2lip-root", default="/root/autodl-tmp/wav2lip-syncnet/Wav2Lip")
    parser.add_argument("--generator-child", default="/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/engineering_qc_oldpilot_attempt04.py")
    parser.add_argument("--syncnet-root", default="/root/autodl-tmp/wav2lip-syncnet/syncnet_python")
    parser.add_argument("--syncnet-python", default="/root/autodl-tmp/wav2lip-syncnet/envs/syncnet/bin/python")
    parser.add_argument("--syncnet-checkpoint", default="/root/autodl-tmp/wav2lip-syncnet/syncnet_python/data/syncnet_v2.model")
    parser.add_argument("--syncnet-evaluator", default="/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py")
    parser.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    parser.add_argument("--ffprobe", default="/usr/bin/ffprobe")
    parser.add_argument("--ffmpeg-sha256", required=True)
    parser.add_argument("--pcm16-lineage-sha256", required=True)
    parser.add_argument("--generator-timeout", type=float, default=900)
    parser.add_argument("--syncnet-timeout", type=float, default=1800)
    parser.add_argument("--ffmpeg-timeout", type=float, default=300)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
