from __future__ import annotations

import shutil
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    OracleError,
    bytes_sha256,
    extract_bgr24_frames,
    extract_pcm_from_media,
    file_sha256,
    read_pcm16_wav,
    run_command,
    write_pcm16_wav,
)


def forward_map(sample_count: int) -> np.ndarray:
    if sample_count < 2 or sample_count - 1 <= 2.0 * np.pi * config.WARP_AMPLITUDE_SAMPLES:
        raise OracleError("audio is too short for the registered warp")
    coordinates = np.arange(sample_count, dtype=np.float64)
    mapped = coordinates + config.WARP_AMPLITUDE_SAMPLES * np.sin(
        2.0 * np.pi * coordinates / float(sample_count - 1)
    )
    mapped[0] = 0.0
    mapped[-1] = float(sample_count - 1)
    if not np.isfinite(mapped).all() or np.any(np.diff(mapped) <= 0.0):
        raise OracleError("registered audio map is not strictly increasing")
    return mapped


def reconstruct_warped_pcm(natural_pcm: bytes) -> tuple[bytes, np.ndarray]:
    if len(natural_pcm) % 2:
        raise OracleError("natural PCM is not int16 aligned")
    samples = np.frombuffer(natural_pcm, dtype="<i2")
    mapped = forward_map(int(samples.size))
    original = np.arange(samples.size, dtype=np.float64)
    warped = np.interp(mapped, original, samples.astype(np.float64))
    return np.rint(warped).astype("<i2", copy=False).tobytes(), mapped


def nearest_half_up(values: np.ndarray) -> np.ndarray:
    """Round non-negative frame coordinates with the frozen half-up rule."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise OracleError("frame coordinates must be finite and non-negative")
    return np.floor(values + 0.5).astype(np.int64)


def build_oracle_indices(sample_count: int, frame_count: int) -> dict[str, Any]:
    mapped = forward_map(sample_count)
    t = config.SAMPLES_PER_FRAME * np.arange(frame_count, dtype=np.float64)
    sampled = np.interp(t, np.arange(sample_count, dtype=np.float64), mapped)
    q_float = sampled / config.SAMPLES_PER_FRAME
    q_oracle = nearest_half_up(q_float)
    if not np.all(np.diff(q_oracle) >= 0):
        raise OracleError("oracle frame indices are not monotone")
    if np.any(q_oracle < 0) or np.any(q_oracle >= frame_count):
        raise OracleError("oracle frame index is outside the source frame range")
    quant_error = q_oracle.astype(np.float64) - q_float
    if np.max(np.abs(quant_error), initial=0.0) > 0.5 + 1e-9:
        raise OracleError("oracle frame quantization error exceeds the frozen bound")
    return {
        "q_id": list(range(frame_count)),
        "q_oracle": [int(value) for value in q_oracle],
        "sample_positions": [float(value) for value in sampled],
        "q_float": [float(value) for value in q_float],
        "quantization_error": [float(value) for value in quant_error],
        "max_abs_quantization_error": float(np.max(np.abs(quant_error), initial=0.0)),
        "monotone": True,
        "in_bounds": True,
        "mapping_sha256": bytes_sha256(np.asarray(mapped, dtype="<f8").tobytes()),
        "mapping_length": int(mapped.size),
    }


def _validate_timeline(probe: dict[str, Any], expected_count: int | None = None) -> dict[str, Any]:
    stream = probe.get("stream")
    frames = probe.get("frames")
    if not isinstance(stream, dict) or not isinstance(frames, list) or not frames:
        raise OracleError("video timeline is incomplete")
    required = {
        "codec_name": "ffv1",
        "pix_fmt": "bgr0",
        "color_range": "pc",
        "color_space": "gbr",
        "width": config.FRAME_WIDTH,
        "height": config.FRAME_HEIGHT,
        "r_frame_rate": "25/1",
        "start_pts": 0,
    }
    for key, value in required.items():
        if stream.get(key) != value:
            raise OracleError(f"video metadata mismatch for {key}: {stream.get(key)!r} != {value!r}")
    if expected_count is not None and len(frames) != expected_count:
        raise OracleError(f"video frame count mismatch: {len(frames)} != {expected_count}")
    pts = [int(item.get("best_effort_timestamp")) for item in frames if isinstance(item, dict) and item.get("best_effort_timestamp") is not None]
    pts_time = [float(item.get("best_effort_timestamp_time")) for item in frames if isinstance(item, dict) and item.get("best_effort_timestamp_time") is not None]
    if len(pts) != len(frames) or len(pts_time) != len(frames) or pts[0] != 0 or any(right <= left for left, right in pairwise(pts)):
        raise OracleError("video PTS are missing or not strictly increasing")
    max_deviation_ms = max(
        abs(float(actual) - index / config.FPS) * 1000.0
        for index, actual in enumerate(pts_time)
    )
    if max_deviation_ms > 1.0:
        raise OracleError(f"video PTS deviate by more than 1 ms: {max_deviation_ms}")
    return {
        "frame_count": len(frames),
        "pts": pts,
        "pts_time": pts_time,
        "pts_max_deviation_ms": float(max_deviation_ms),
        "stream": dict(stream),
    }


def audit_parent_stream(path: Path, expected_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    frames, evidence = extract_bgr24_frames(path)
    if frames.shape[0] != expected_count:
        raise OracleError(f"parent G_N frame count mismatch: {path}")
    timeline = _validate_timeline(evidence["probe"], expected_count)
    evidence["timeline"] = timeline
    evidence["path"] = str(path.resolve())
    evidence["sha256"] = file_sha256(path)
    return frames, evidence


def encode_video(frames: np.ndarray, output: Path) -> dict[str, Any]:
    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[1:] != (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3) or values.dtype != np.uint8:
        raise OracleError(f"frames do not match BGR24 contract: {values.shape} {values.dtype}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(config.FFMPEG),
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{config.FRAME_WIDTH}x{config.FRAME_HEIGHT}",
        "-r",
        str(config.FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-pix_fmt",
        "bgr0",
        "-color_range",
        "pc",
        "-colorspace",
        # FFmpeg's CLI names the RGB colorspace `rgb`; ffprobe exposes the
        # resulting stream metadata as `gbr`, which is the frozen contract.
        "rgb",
        "-video_track_timescale",
        "1000",
        str(output),
    ]
    run_command(command, input_bytes=values.tobytes())
    decoded, evidence = extract_bgr24_frames(output)
    if not np.array_equal(decoded, values):
        mismatch = np.argwhere(decoded != values)
        location = tuple(int(value) for value in mismatch[0]) if mismatch.size else ()
        raise OracleError(f"encoded video pixels differ at {location}: {output}")
    evidence.update(
        {
            "output": str(output.resolve()),
            "output_sha256": file_sha256(output),
            "source_frame_sha256": [bytes_sha256(frame.tobytes()) for frame in values],
            "decoded_frame_sha256": [bytes_sha256(frame.tobytes()) for frame in decoded],
            "pixel_identity_verified": True,
            "encoding_command": command,
        }
    )
    _validate_timeline(evidence["probe"], int(values.shape[0]))
    return evidence


def mux_audio(video: Path, audio_wav: Path, output: Path, expected_pcm: bytes) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(config.FFMPEG),
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-i",
        str(audio_wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "pcm_s16le",
        "-map_metadata",
        "-1",
        str(output),
    ]
    run_command(command)
    frames, video_evidence = extract_bgr24_frames(output)
    observed_pcm = extract_pcm_from_media(output)
    if observed_pcm != expected_pcm:
        raise OracleError(f"muxed PCM differs from requested audio: {output}")
    if frames.shape[0] != int(video_evidence["frame_count"]):
        raise OracleError(f"muxed frame evidence is inconsistent: {output}")
    return {
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "video": video_evidence,
        "audio_pcm_sha256": bytes_sha256(observed_pcm),
        "audio_pcm_sample_count": len(observed_pcm) // 2,
        "audio_pcm_verified": True,
        "video_stream_copy": True,
        "audio_modified": False,
        "mux_command": command,
    }


def materialize_audio(
    natural_audio: Path,
    warped_pcm: bytes,
    output_dir: Path,
    *,
    artifact_stem: str | None = None,
) -> dict[str, dict[str, Any]]:
    natural_pcm, _natural, _params = read_pcm16_wav(natural_audio)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = artifact_stem or natural_audio.stem
    natural_target = output_dir / f"{stem}__N.wav"
    if natural_target.exists():
        raise OracleError(f"refusing to overwrite audio artifact: {natural_target}")
    shutil.copyfile(natural_audio, natural_target)
    copied_pcm, _a, _p = read_pcm16_wav(natural_target)
    if copied_pcm != natural_pcm:
        raise OracleError("copied natural PCM changed")
    warped_target = output_dir / f"{stem}__W.wav"
    warped_meta = write_pcm16_wav(warped_target, warped_pcm)
    return {
        "N": {
            "path": str(natural_target.resolve()),
            "container_sha256": file_sha256(natural_target),
            "decoded_pcm_sha256": bytes_sha256(natural_pcm),
            "sample_count": len(natural_pcm) // 2,
            "construction": "copied untouched parent natural PCM16",
        },
        "W": {
            **warped_meta,
            "construction": "reconstructed registered LOCAL_WARP_120 PCM16",
        },
    }
