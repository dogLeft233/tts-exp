"""Independent consumer for cross-TFG original-PCM enhancement evaluations.

The consumer trusts neither producer-reported Sync-C summaries nor support
counts. It verifies the immutable plan/result rows, decodes the canonical
waveforms and final muxes, rebuilds the paired common query supports from the
saved per-query distance matrices, and then performs equal-speaker inference.

This module is read-only with respect to all producer inputs. It creates a new
output directory and refuses to overwrite an existing one.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "cross-tfg-audio-eval-v1"
CONSUMER_VERSION = "1.0.1"
IDENTITY_FIELDS = (
    "row_id",
    "split",
    "sample_id",
    "speaker_id",
    "model_id",
    "audio_frontend",
    "model_role",
    "face_id",
    "seed",
    "method_id",
    "method_config_sha256",
    "checkpoint_sha256",
)
REQUIRED_SCORE_ARRAYS = {
    "query_frame_index",
    "query_timestamp_num",
    "query_timestamp_den",
    "track_frame_index",
    "valid_query_mask",
    "distance_by_query_lag",
    "lag_offsets_frames",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class ConsumerError(RuntimeError):
    """Raised for a frozen-contract or artifact integrity failure."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(value: Mapping[str, Any], self_field: str) -> str:
    return _sha256_bytes(_canonical_json({k: v for k, v in value.items() if k != self_field}))


def _require_sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise ConsumerError(f"{where} must be a lowercase 64-character SHA-256")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # preserve filename and parser reason in the audit
        raise ConsumerError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConsumerError(f"JSON root must be an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except Exception as exc:
                    raise ConsumerError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
                if not isinstance(item, dict):
                    raise ConsumerError(f"JSONL row must be an object at {path}:{line_number}")
                rows.append(item)
    except OSError as exc:
        raise ConsumerError(f"cannot read JSONL {path}: {exc}") from exc
    return rows


def _resolve_path(root: Path, value: Any, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ConsumerError(f"{where} path must be a non-empty string")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _check_file_ref(
    root: Path,
    ref: Mapping[str, Any],
    *,
    where: str,
    path_key: str = "path",
    sha_key: str = "file_sha256",
    cache: dict[str, str] | None = None,
) -> Path:
    path = _resolve_path(root, ref.get(path_key), f"{where}.{path_key}")
    expected = _require_sha(ref.get(sha_key), f"{where}.{sha_key}")
    cache_key = str(path.resolve())
    actual = cache.get(cache_key) if cache is not None else None
    if actual is None:
        if not path.is_file():
            raise ConsumerError(f"missing input file for {where}: {path}")
        actual = _sha256_file(path)
        if cache is not None:
            cache[cache_key] = actual
    if actual != expected:
        raise ConsumerError(f"file SHA mismatch for {where}: expected {expected}, got {actual}")
    return path


def _read_canonical_f32_wav(path: Path) -> tuple[np.ndarray, str, int]:
    """Read a canonical mono 16 kHz IEEE-float32 WAV without audio libraries."""

    raw = path.read_bytes()
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ConsumerError(f"not a little-endian RIFF/WAVE file: {path}")
    declared_end = int.from_bytes(raw[4:8], "little") + 8
    if declared_end > len(raw):
        raise ConsumerError(f"truncated RIFF payload: {path}")

    fmt: tuple[int, int, int, int, int] | None = None
    chunks: list[bytes] = []
    pos = 12
    while pos + 8 <= min(declared_end, len(raw)):
        chunk_id = raw[pos : pos + 4]
        chunk_size = int.from_bytes(raw[pos + 4 : pos + 8], "little")
        start = pos + 8
        end = start + chunk_size
        if end > min(declared_end, len(raw)):
            raise ConsumerError(f"truncated WAV chunk {chunk_id!r}: {path}")
        payload = raw[start:end]
        if chunk_id == b"fmt ":
            if len(payload) < 16:
                raise ConsumerError(f"short WAV fmt chunk: {path}")
            format_code = int.from_bytes(payload[0:2], "little")
            channels = int.from_bytes(payload[2:4], "little")
            sample_rate = int.from_bytes(payload[4:8], "little")
            block_align = int.from_bytes(payload[12:14], "little")
            bits = int.from_bytes(payload[14:16], "little")
            if format_code == 0xFFFE:
                if len(payload) < 40:
                    raise ConsumerError(f"truncated extensible WAV fmt chunk: {path}")
                format_code = int.from_bytes(payload[24:26], "little")
            fmt = (format_code, channels, sample_rate, bits, block_align)
        elif chunk_id == b"data":
            chunks.append(payload)
        pos = end + (chunk_size & 1)

    if fmt is None or not chunks:
        raise ConsumerError(f"WAV lacks fmt or data chunk: {path}")
    format_code, channels, sample_rate, bits, block_align = fmt
    if (format_code, channels, sample_rate, bits, block_align) != (3, 1, 16_000, 32, 4):
        raise ConsumerError(
            f"canonical WAV must be mono 16 kHz float32; got format={format_code}, "
            f"channels={channels}, rate={sample_rate}, bits={bits}, align={block_align}: {path}"
        )
    pcm = b"".join(chunks)
    if len(pcm) % 4:
        raise ConsumerError(f"float32 WAV payload is not a whole number of samples: {path}")
    samples = np.frombuffer(pcm, dtype=np.dtype("<f4"))
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ConsumerError(f"canonical WAV is empty or non-finite: {path}")
    return samples, _sha256_bytes(pcm), int(samples.size)


def _canonical_pcm_ref(
    root: Path,
    ref: Mapping[str, Any],
    *,
    where: str,
    cache: dict[str, str],
) -> tuple[Path, np.ndarray, str, int]:
    path = _check_file_ref(root, ref, where=where, cache=cache)
    samples, pcm_hash, sample_count = _read_canonical_f32_wav(path)
    if pcm_hash != _require_sha(ref.get("pcm_sha256"), f"{where}.pcm_sha256"):
        raise ConsumerError(f"canonical float32 PCM SHA mismatch for {where}")
    if ref.get("sample_rate_hz") != 16_000 or ref.get("channels") != 1:
        raise ConsumerError(f"canonical audio metadata must be 16 kHz mono for {where}")
    if ref.get("sample_count") != sample_count:
        raise ConsumerError(f"sample_count mismatch for {where}: metadata={ref.get('sample_count')}, actual={sample_count}")
    return path, samples, pcm_hash, sample_count


def _run_bytes(command: Sequence[str]) -> bytes:
    try:
        result = subprocess.run(list(command), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise ConsumerError(f"required media executable is unavailable: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")[-2000:]
        raise ConsumerError(f"media command failed ({command[0]}): {stderr}") from exc
    return result.stdout


def _probe(path: Path, ffprobe: str) -> dict[str, Any]:
    raw = _run_bytes([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("streams"), list):
        raise ConsumerError(f"ffprobe returned malformed stream data for {path}")
    return data


def _probe_video_pts_clock(path: Path, ffprobe: str, fps_num: int, fps_den: int) -> list[tuple[int, int, int]]:
    """Read actual decoded video PTS and require the declared rational frame clock."""
    command = [
        ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json",
        "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time,pkt_pts_time", str(path),
    ]
    payload = json.loads(_run_bytes(command).decode("utf-8"))
    streams = [item for item in payload.get("streams", []) if isinstance(item, dict) and item.get("codec_type") == "video"]
    if len(streams) != 1:
        raise ConsumerError(f"expected one probed video stream while reading PTS: {path}")
    fps_text = streams[0].get("avg_frame_rate") or streams[0].get("r_frame_rate")
    if not isinstance(fps_text, str) or Fraction(fps_text) != Fraction(fps_num, fps_den):
        raise ConsumerError(f"actual video stream FPS differs from frozen rational clock: {path}")
    frame_records = payload.get("frames")
    if not isinstance(frame_records, list) or not frame_records:
        raise ConsumerError(f"ffprobe returned no decoded video PTS rows: {path}")
    actual: list[tuple[int, int, int]] = []
    for index, frame in enumerate(frame_records):
        if not isinstance(frame, dict):
            raise ConsumerError(f"malformed decoded video PTS row {index}: {path}")
        raw_time = frame.get("best_effort_timestamp_time", frame.get("pkt_pts_time"))
        if raw_time is None:
            raise ConsumerError(f"decoded video frame {index} has no exact PTS: {path}")
        timestamp = Fraction(str(raw_time))
        expected = Fraction(index * fps_den, fps_num)
        if timestamp != expected:
            raise ConsumerError(f"decoded video PTS {timestamp} at row {index} differs from frozen clock {expected}: {path}")
        actual.append((index, timestamp.numerator, timestamp.denominator))
    return actual


def _clock_sha256(clock: Sequence[tuple[int, int, int]]) -> str:
    return _sha256_bytes(_canonical_json([list(row) for row in clock]))


def _one_stream(probe: Mapping[str, Any], stream_type: str, where: str) -> Mapping[str, Any]:
    streams = [s for s in probe.get("streams", []) if isinstance(s, dict) and s.get("codec_type") == stream_type]
    if len(streams) != 1:
        raise ConsumerError(f"expected exactly one {stream_type} stream in {where}, found {len(streams)}")
    return streams[0]


def _video_elementary_sha256(path: Path, ffmpeg: str, ffprobe: str) -> str:
    stream = _one_stream(_probe(path, ffprobe), "video", str(path))
    codec_name = stream.get("codec_name")
    elementary = {
        "h264": "h264",
        "hevc": "hevc",
        "mpeg4": "m4v",
        "vp8": "ivf",
        "vp9": "ivf",
        "av1": "ivf",
    }.get(codec_name)
    if elementary is None:
        raise ConsumerError(f"unsupported video codec for elementary-stream verification: {codec_name}")
    command = [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-c:v", "copy", "-an", "-f", elementary, "pipe:1"]
    return _sha256_bytes(_run_bytes(command))


def _video_hash_muxer_packet_sha256(path: Path, ffmpeg: str) -> str:
    """Recompute the producer's `ffmpeg -f hash -hash sha256` video digest.

    This is a hash over FFmpeg's copied video packets. It is intentionally
    distinct from `_video_elementary_sha256`, which hashes codec-specific
    elementary-container bytes and can differ because of serialization such
    as Annex-B/extradata headers.
    """
    command = [
        ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-c:v", "copy",
        "-f", "hash", "-hash", "sha256", "pipe:1",
    ]
    output = _run_bytes(command).decode("ascii").strip()
    if not output.startswith("SHA256="):
        raise ConsumerError(f"FFmpeg hash muxer returned an unexpected digest for {path}: {output!r}")
    digest = output.split("=", 1)[1].lower()
    return _require_sha(digest, f"{path} ffmpeg hash-muxer packet digest")


def _decode_rgb24_video(path: Path, ffmpeg: str, ffprobe: str) -> np.ndarray:
    stream = _one_stream(_probe(path, ffprobe), "video", str(path))
    width = stream.get("width")
    height = stream.get("height")
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ConsumerError(f"video stream has invalid dimensions: {path}")
    command = [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-an", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = _run_bytes(command)
    frame_bytes = width * height * 3
    if frame_bytes == 0 or len(raw) == 0 or len(raw) % frame_bytes:
        raise ConsumerError(f"RGB24 decode length does not contain whole frames: {path}")
    return np.frombuffer(raw, dtype=np.uint8).reshape((-1, height, width, 3))


def _max_rgb_frame_difference(path_a: Path, path_b: Path, ffmpeg: str, ffprobe: str) -> int | None:
    frames_a = _decode_rgb24_video(path_a, ffmpeg, ffprobe)
    frames_b = _decode_rgb24_video(path_b, ffmpeg, ffprobe)
    if frames_a.shape != frames_b.shape:
        return None
    difference = np.abs(frames_a.astype(np.int16) - frames_b.astype(np.int16))
    return int(np.max(difference))


def _pcm16_readback_sha256(path: Path, ffmpeg: str, sample_rate: int, channels: int) -> tuple[str, int]:
    pcm, sample_count = _pcm16_readback_bytes(path, ffmpeg, sample_rate, channels)
    return _sha256_bytes(pcm), sample_count


def _pcm16_readback_bytes(path: Path, ffmpeg: str, sample_rate: int, channels: int) -> tuple[bytes, int]:
    command = [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn", "-f", "s16le", "-ar", str(sample_rate), "-ac", str(channels), "pipe:1"]
    pcm = _run_bytes(command)
    if len(pcm) % (2 * channels):
        raise ConsumerError(f"decoded PCM16 length does not align to {channels} channel(s): {path}")
    return pcm, len(pcm) // (2 * channels)


def _resolve_executable(executable: str) -> Path:
    candidate = Path(executable)
    if candidate.is_file():
        return candidate.resolve()
    located = shutil.which(executable)
    if located is None:
        raise ConsumerError(f"required media executable is unavailable: {executable}")
    return Path(located).resolve()


def _verify_primary_pcm16_window(
    score: Mapping[str, Any],
    render: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    root: Path,
    natural_path: Path,
    natural_pcm_sha256: str,
    natural_sample_count: int,
    timestamps: Sequence[tuple[int, int, int]],
    ffmpeg: str,
    ffprobe: str,
    file_cache: dict[str, str],
) -> dict[str, Any]:
    """Verify direct-natural PCM16 crop lineage and its SyncNet Matroska input."""

    syncnet = plan["protocol"]["syncnet"]
    conversion = syncnet.get("primary_audio_input")
    if not isinstance(conversion, dict):
        raise ConsumerError("plan.protocol.syncnet.primary_audio_input must freeze PCM16 conversion")
    fixed_conversion = {
        "mode": "canonical_natural_pcm16_crop_window",
        "source": "natural_pcm",
        "codec": "pcm_s16le",
        "sample_rate_hz": 16_000,
        "channels": 1,
        "operation": "ffmpeg_decode_full_source_then_slice_sample_window",
        "frame_clock": "original_render_source_pts",
    }
    for key, expected in fixed_conversion.items():
        if conversion.get(key) != expected:
            raise ConsumerError(f"primary_audio_input.{key} must be frozen as {expected!r}")
    conversion_sha = _require_sha(conversion.get("ffmpeg_sha256"), "primary_audio_input.ffmpeg_sha256")
    conversion_code_sha = _require_sha(conversion.get("code_sha256"), "primary_audio_input.code_sha256")
    argv_template = conversion.get("argv_template")
    expected_argv = [
        "-v", "error", "-i", "{natural_pcm_path}", "-map", "0:a:0", "-vn",
        "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1",
    ]
    if argv_template != expected_argv:
        raise ConsumerError("primary_audio_input.argv_template differs from the frozen full-source PCM16 conversion")
    actual_ffmpeg = _resolve_executable(ffmpeg)
    actual_ffmpeg_sha = _sha256_file(actual_ffmpeg)
    if actual_ffmpeg_sha != conversion_sha:
        raise ConsumerError(
            f"FFmpeg binary SHA differs from frozen PCM16 conversion: expected={conversion_sha}, actual={actual_ffmpeg_sha}"
        )

    selected_crop = render.get("selected_track_crop")
    if not isinstance(selected_crop, dict):
        raise ConsumerError("render.selected_track_crop must bind the exact SyncNet crop video and frame map")
    crop_video = _check_file_ref(
        root,
        {"path": selected_crop.get("path"), "file_sha256": selected_crop.get("file_sha256")},
        where="render.selected_track_crop",
        cache=file_cache,
    )
    crop_video_packet_sha = _video_hash_muxer_packet_sha256(crop_video, ffmpeg)
    crop_video_elementary_sha = _video_elementary_sha256(crop_video, ffmpeg, ffprobe)
    if _require_sha(selected_crop.get("video_elementary_sha256"), "render.selected_track_crop.video_elementary_sha256") != crop_video_packet_sha:
        raise ConsumerError("selected track crop reported FFmpeg hash-muxer packet SHA differs from its file")
    for field in ("selected_track_id", "selected_track_index"):
        if selected_crop.get(field) != render.get(field):
            raise ConsumerError(f"selected track crop {field} differs from render metadata")

    frame_map_path = _resolve_path(root, selected_crop.get("frame_map_path"), "render.selected_track_crop.frame_map_path")
    frame_map_sha = _require_sha(selected_crop.get("frame_map_sha256"), "render.selected_track_crop.frame_map_sha256")
    frame_map_bytes = frame_map_path.read_bytes()
    if _sha256_bytes(frame_map_bytes) != frame_map_sha:
        raise ConsumerError("selected track source-frame map SHA mismatch")
    try:
        frame_map_payload = json.loads(frame_map_bytes.decode("utf-8"))
    except Exception as exc:
        raise ConsumerError(f"cannot parse selected track frame map: {exc}") from exc
    if not isinstance(frame_map_payload, dict) or frame_map_payload.get("schema_version") != "cross-tfg-track-frame-map-v2":
        raise ConsumerError("selected track frame map has an unexpected schema")
    mapped_render_path = frame_map_payload.get("render_video_path")
    if mapped_render_path == render.get("silent_video_path"):
        mapped_render_sha = render.get("silent_video_sha256")
    elif mapped_render_path == render.get("muxed_video_path"):
        mapped_render_sha = render.get("muxed_video_sha256")
    else:
        raise ConsumerError("selected track frame-map render_video_path is not a rendered input bound by the result")
    map_bindings = {
        "selected_crop_video_path": selected_crop.get("path"),
        "selected_crop_video_file_sha256": selected_crop.get("file_sha256"),
        "selected_crop_video_elementary_sha256": selected_crop.get("video_elementary_sha256"),
        "selected_track_id": selected_crop.get("selected_track_id"),
        "selected_track_index": selected_crop.get("selected_track_index"),
        "render_video_path": mapped_render_path,
        "render_video_file_sha256": mapped_render_sha,
        "render_video_elementary_sha256": render.get("video_stream_sha256"),
        "render_timestamps_sha256": render.get("timestamps_sha256"),
        "fps_num": render.get("fps_num"),
        "fps_den": render.get("fps_den"),
    }
    for key, expected in map_bindings.items():
        actual = frame_map_payload.get(key)
        if actual != expected:
            raise ConsumerError(f"selected track frame-map binding {key} differs from render metadata")
    frame_rows = frame_map_payload.get("frames")
    if not isinstance(frame_rows, list) or not frame_rows:
        raise ConsumerError("selected track frame map must contain at least one crop frame")
    fps_num = render.get("fps_num")
    fps_den = render.get("fps_den")
    if not isinstance(fps_num, int) or isinstance(fps_num, bool) or not isinstance(fps_den, int) or isinstance(fps_den, bool) or fps_num <= 0 or fps_den <= 0:
        raise ConsumerError("render FPS is invalid for selected-track source-clock mapping")
    crop_actual_pts = _probe_video_pts_clock(crop_video, ffprobe, fps_num, fps_den)
    if len(crop_actual_pts) != len(frame_rows):
        raise ConsumerError("selected crop decoded PTS count differs from its source-frame map")
    crop_source_clock_frames: list[int] = []
    crop_clock_by_local_index: list[tuple[int, int, int]] = []
    for local_index, frame in enumerate(frame_rows):
        if not isinstance(frame, dict) or frame.get("crop_frame_index") != local_index:
            raise ConsumerError("selected track crop frame indices must be contiguous from zero")
        source_clock_index = frame.get("source_clock_frame_index")
        if not isinstance(source_clock_index, int) or isinstance(source_clock_index, bool) or source_clock_index < 0:
            raise ConsumerError("selected track map has an invalid nominal source-clock frame index")
        if crop_source_clock_frames and source_clock_index != crop_source_clock_frames[-1] + 1:
            raise ConsumerError("selected crop does not map to a contiguous nominal source-clock span")
        raw_num = frame.get("timestamp_num")
        raw_den = frame.get("timestamp_den")
        stamp = _rational_pair(frame, "render.selected_track_crop.frame_map")
        if (raw_num, raw_den) != stamp:
            raise ConsumerError("selected track frame-map PTS must be stored as a reduced rational")
        expected_stamp = Fraction(source_clock_index * fps_den, fps_num)
        if stamp != (expected_stamp.numerator, expected_stamp.denominator):
            raise ConsumerError("selected track frame-map PTS differs from the nominal source-video clock")
        frame_origin = frame.get("frame_origin")
        render_frame_index = frame.get("render_frame_index")
        if frame_origin == "decoded_render":
            if (
                not isinstance(render_frame_index, int)
                or isinstance(render_frame_index, bool)
                or render_frame_index != source_clock_index
                or not 0 <= source_clock_index < len(timestamps)
            ):
                raise ConsumerError("decoded-render frame mapping does not match an available source-clock frame")
            if stamp != (timestamps[source_clock_index][1], timestamps[source_clock_index][2]):
                raise ConsumerError("decoded-render frame-map PTS differs from the render timestamp sidecar")
        elif frame_origin == "nominal_clock_outside_render_decode":
            if render_frame_index is not None or source_clock_index < len(timestamps):
                raise ConsumerError("outside-render nominal frame must have no render index and lie beyond decoded frames")
        else:
            raise ConsumerError(f"unsupported selected-track frame_origin: {frame_origin!r}")
        crop_source_clock_frames.append(source_clock_index)
        crop_clock_by_local_index.append((source_clock_index, stamp[0], stamp[1]))

    first_source_frame = crop_source_clock_frames[0]
    stop_source_frame = crop_source_clock_frames[-1] + 1
    fps_num = render.get("fps_num")
    fps_den = render.get("fps_den")
    if not isinstance(fps_num, int) or not isinstance(fps_den, int) or fps_num <= 0 or fps_den <= 0:
        raise ConsumerError("render FPS is invalid for score-audio window mapping")
    sample_rate = int(conversion["sample_rate_hz"])
    samples_per_frame = Fraction(sample_rate * fps_den, fps_num)
    if samples_per_frame.denominator != 1:
        raise ConsumerError("frozen source FPS does not map video frame boundaries to integer audio samples")
    first_pts = Fraction(crop_clock_by_local_index[0][1], crop_clock_by_local_index[0][2])
    stop_pts = Fraction(stop_source_frame * fps_den, fps_num)
    start_sample_fraction = first_pts * sample_rate
    stop_sample_fraction = stop_pts * sample_rate
    if start_sample_fraction.denominator != 1 or stop_sample_fraction.denominator != 1:
        raise ConsumerError("selected crop PTS boundaries do not land on exact source audio sample boundaries")
    expected_start_sample = int(start_sample_fraction)
    requested_stop_sample = int(stop_sample_fraction)
    if expected_start_sample >= natural_sample_count:
        raise ConsumerError("selected crop starts at or after the end of canonical natural PCM")
    expected_stop_sample = min(requested_stop_sample, natural_sample_count)
    tail_clipped_to_natural_pcm = expected_stop_sample < requested_stop_sample
    if expected_stop_sample <= expected_start_sample:
        raise ConsumerError("selected crop has no original natural PCM samples in its requested window")

    audio_window = score.get("primary_pcm16_window")
    if not isinstance(audio_window, dict):
        raise ConsumerError("score.primary_pcm16_window must reference directly derived natural-source PCM16")
    if audio_window.get("source_float32_pcm_sha256") != natural_pcm_sha256:
        raise ConsumerError("primary PCM16 window is not derived from the original natural float32 PCM")
    window_fields = {
        "start_sample": expected_start_sample,
        "requested_stop_sample_exclusive": requested_stop_sample,
        "stop_sample_exclusive": expected_stop_sample,
        "sample_count": expected_stop_sample - expected_start_sample,
        "tail_clipped_to_natural_pcm": tail_clipped_to_natural_pcm,
        "sample_rate_hz": sample_rate,
        "channels": 1,
        "source_frame_start": first_source_frame,
        "source_frame_stop_exclusive": stop_source_frame,
        "fps_num": fps_num,
        "fps_den": fps_den,
        "source_start_pts_num": first_pts.numerator,
        "source_start_pts_den": first_pts.denominator,
        "source_stop_pts_num": stop_pts.numerator,
        "source_stop_pts_den": stop_pts.denominator,
    }
    for field, expected in window_fields.items():
        if audio_window.get(field) != expected:
            raise ConsumerError(f"primary PCM16 window {field} differs from selected crop clock")
    conversion_ref = audio_window.get("conversion")
    if not isinstance(conversion_ref, dict):
        raise ConsumerError("primary PCM16 window must record conversion provenance")
    expected_conversion_ref = {
        "ffmpeg_sha256": conversion_sha,
        "argv_template": argv_template,
        "code_sha256": conversion_code_sha,
        "operation": conversion["operation"],
    }
    if conversion_ref != expected_conversion_ref:
        raise ConsumerError("primary PCM16 conversion provenance differs from frozen plan")

    window_wav = _check_file_ref(
        root,
        {"path": audio_window.get("path"), "file_sha256": audio_window.get("file_sha256")},
        where="score.primary_pcm16_window",
        cache=file_cache,
    )
    window_probe = _probe(window_wav, ffprobe)
    window_audio_stream = _one_stream(window_probe, "audio", str(window_wav))
    if window_audio_stream.get("codec_name") != "pcm_s16le" or int(window_audio_stream.get("sample_rate", 0)) != sample_rate or int(window_audio_stream.get("channels", 0)) != 1:
        raise ConsumerError("primary PCM16 window WAV must be mono 16 kHz pcm_s16le")
    window_bytes, window_count = _pcm16_readback_bytes(window_wav, ffmpeg, sample_rate, 1)
    window_pcm_sha = _sha256_bytes(window_bytes)
    if window_count != expected_stop_sample - expected_start_sample:
        raise ConsumerError("primary PCM16 WAV sample count differs from selected crop window")
    if _require_sha(audio_window.get("pcm_sha256"), "score.primary_pcm16_window.pcm_sha256") != window_pcm_sha:
        raise ConsumerError("primary PCM16 WAV decoded hash differs from its reference")

    full_source_bytes, full_source_count = _pcm16_readback_bytes(natural_path, ffmpeg, sample_rate, 1)
    if full_source_count != natural_sample_count:
        raise ConsumerError("direct natural-source PCM16 conversion changed the source sample count")
    expected_window_bytes = full_source_bytes[expected_start_sample * 2 : expected_stop_sample * 2]
    if len(expected_window_bytes) != len(window_bytes) or expected_window_bytes != window_bytes:
        raise ConsumerError("primary PCM16 window bytes are not an exact direct-source sample slice")

    primary_input = score.get("primary_syncnet_input")
    if not isinstance(primary_input, dict):
        raise ConsumerError("score.primary_syncnet_input must reference the exact official scorer input")
    input_path = _check_file_ref(
        root,
        {"path": primary_input.get("path"), "file_sha256": primary_input.get("file_sha256")},
        where="score.primary_syncnet_input",
        cache=file_cache,
    )
    input_probe = _probe(input_path, ffprobe)
    input_formats = str(input_probe.get("format", {}).get("format_name", "")).split(",")
    if primary_input.get("container") != "matroska" or "matroska" not in input_formats:
        raise ConsumerError("primary SyncNet input must be the frozen Matroska crop/PCM16 remux")
    input_video = _one_stream(input_probe, "video", str(input_path))
    input_audio = _one_stream(input_probe, "audio", str(input_path))
    if input_audio.get("codec_name") != "pcm_s16le" or int(input_audio.get("sample_rate", 0)) != sample_rate or int(input_audio.get("channels", 0)) != 1:
        raise ConsumerError("primary SyncNet Matroska audio must remain mono 16 kHz PCM16")
    input_video_packet_sha = _video_hash_muxer_packet_sha256(input_path, ffmpeg)
    input_video_elementary_sha = _video_elementary_sha256(input_path, ffmpeg, ffprobe)
    input_actual_pts = _probe_video_pts_clock(input_path, ffprobe, fps_num, fps_den)
    if input_actual_pts != crop_actual_pts:
        raise ConsumerError("primary SyncNet Matroska changed selected crop decoded PTS")
    if input_video_packet_sha != crop_video_packet_sha:
        raise ConsumerError("primary SyncNet Matroska changed the selected crop video packet stream")
    if input_video_elementary_sha != crop_video_elementary_sha:
        raise ConsumerError("primary SyncNet Matroska changed the selected crop video elementary stream")
    if primary_input.get("video_elementary_sha256") != input_video_packet_sha or primary_input.get("crop_video_elementary_sha256") != crop_video_packet_sha:
        raise ConsumerError("primary SyncNet input video hashes differ from the selected track crop")
    if primary_input.get("crop_video_path") != selected_crop.get("path") or primary_input.get("crop_video_file_sha256") != selected_crop.get("file_sha256"):
        raise ConsumerError("primary SyncNet input does not bind the selected crop video file")
    if primary_input.get("selected_track_id") != selected_crop.get("selected_track_id") or primary_input.get("selected_track_index") != selected_crop.get("selected_track_index"):
        raise ConsumerError("primary SyncNet input selected-track identity differs from render crop")
    input_pcm_bytes, input_pcm_count = _pcm16_readback_bytes(input_path, ffmpeg, sample_rate, 1)
    input_pcm_sha = _sha256_bytes(input_pcm_bytes)
    if input_pcm_bytes != window_bytes or input_pcm_count != window_count:
        raise ConsumerError("primary SyncNet Matroska audio is not the exact source-derived PCM16 window")
    if _require_sha(primary_input.get("audio_pcm16_sha256"), "score.primary_syncnet_input.audio_pcm16_sha256") != input_pcm_sha:
        raise ConsumerError("primary SyncNet input PCM16 hash differs from decoded stream")
    if primary_input.get("audio_sample_count") != input_pcm_count or primary_input.get("sample_rate_hz") != sample_rate or primary_input.get("channels") != 1 or primary_input.get("audio_codec") != "pcm_s16le":
        raise ConsumerError("primary SyncNet input PCM16 metadata differs from its decoded stream")
    if _require_sha(score.get("score_preprocessed_pcm16_sha256"), "score.score_preprocessed_pcm16_sha256") != window_pcm_sha:
        raise ConsumerError("score summary does not identify the direct natural-source PCM16 window")
    if score.get("score_preprocessed_sample_count") != window_count:
        raise ConsumerError("score PCM16 sample count differs from the direct natural-source window")

    return {
        "primary_pcm16_window_file_sha256": _sha256_file(window_wav),
        "primary_pcm16_window_sha256": window_pcm_sha,
        "primary_pcm16_sample_count": window_count,
        "primary_pcm16_start_sample": expected_start_sample,
        "primary_pcm16_requested_stop_sample_exclusive": requested_stop_sample,
        "primary_pcm16_stop_sample_exclusive": expected_stop_sample,
        "primary_pcm16_tail_clipped_to_natural_pcm": tail_clipped_to_natural_pcm,
        "source_clock_frame_start": first_source_frame,
        "source_clock_frame_stop_exclusive": stop_source_frame,
        "primary_syncnet_input_file_sha256": _sha256_file(input_path),
        "primary_syncnet_input_video_elementary_sha256": input_video_packet_sha,
        "primary_syncnet_input_video_packet_sha256": input_video_packet_sha,
        "primary_syncnet_input_video_serialized_elementary_sha256": input_video_elementary_sha,
        "primary_syncnet_input_actual_pts_sha256": _clock_sha256(input_actual_pts),
        "selected_track_crop_video_elementary_sha256": crop_video_packet_sha,
        "selected_track_crop_video_packet_sha256": crop_video_packet_sha,
        "selected_track_crop_video_serialized_elementary_sha256": crop_video_elementary_sha,
        "selected_track_crop_actual_pts_sha256": _clock_sha256(crop_actual_pts),
        "selected_track_index": selected_crop.get("selected_track_index"),
        "selected_track_frame_map_sha256": frame_map_sha,
        "official_query_count": max(0, min(len(frame_rows), window_count // 640) - 5),
        # Private in-memory map used below to verify every valid SyncNet query.
        # It is intentionally not copied into the public audit summary.
        "_query_clock_by_track_frame": crop_clock_by_local_index,
    }


def _pcm_f32_readback_sha256(path: Path, ffmpeg: str, sample_rate: int = 16_000, channels: int = 1) -> tuple[str, int]:
    """Decode a mux's audio stream as canonical little-endian float32 PCM."""

    command = [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn", "-f", "f32le", "-ar", str(sample_rate), "-ac", str(channels), "pipe:1"]
    pcm = _run_bytes(command)
    frame_bytes = 4 * channels
    if len(pcm) % frame_bytes:
        raise ConsumerError(f"decoded float32 PCM length does not align to {channels} channel(s): {path}")
    values = np.frombuffer(pcm, dtype=np.dtype("<f4"))
    if not np.isfinite(values).all():
        raise ConsumerError(f"decoded float32 PCM contains non-finite samples: {path}")
    return _sha256_bytes(pcm), len(pcm) // frame_bytes


def _rational_pair(row: Mapping[str, Any], where: str) -> tuple[int, int]:
    num = row.get("timestamp_num")
    den = row.get("timestamp_den")
    if not isinstance(num, int) or isinstance(num, bool) or not isinstance(den, int) or isinstance(den, bool) or den <= 0:
        raise ConsumerError(f"invalid rational timestamp in {where}")
    fraction = Fraction(num, den)
    return fraction.numerator, fraction.denominator


def _read_timestamp_sidecar(path: Path, expected_sha: str, frame_count: int, fps_num: int, fps_den: int) -> list[tuple[int, int, int]]:
    raw = path.read_bytes()
    if _sha256_bytes(raw) != expected_sha:
        raise ConsumerError(f"timestamp sidecar SHA mismatch: {path}")
    payload = json.loads(raw.decode("utf-8"))
    frames = payload.get("frames") if isinstance(payload, dict) else payload
    if not isinstance(frames, list) or len(frames) != frame_count:
        raise ConsumerError(f"timestamp sidecar frame count mismatch: {path}")
    result: list[tuple[int, int, int]] = []
    for expected_index, frame in enumerate(frames):
        if not isinstance(frame, dict) or frame.get("frame_index") != expected_index:
            raise ConsumerError(f"timestamp sidecar indices must be contiguous from zero: {path}")
        num, den = _rational_pair(frame, str(path))
        if Fraction(num, den) != Fraction(expected_index * fps_den, fps_num):
            raise ConsumerError(f"timestamp sidecar does not match the frozen rational FPS clock: {path}")
        result.append((expected_index, num, den))
    return result


def _check_roi_sidecar(path: Path, expected_sha: str, timestamps: list[tuple[int, int, int]], crop_policy_sha: str) -> str:
    raw = path.read_bytes()
    if _sha256_bytes(raw) != expected_sha:
        raise ConsumerError(f"crop ROI sidecar SHA mismatch: {path}")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "tfg-crop-roi-v1":
        raise ConsumerError(f"unexpected crop ROI sidecar schema: {path}")
    if payload.get("crop_policy_sha256") != crop_policy_sha:
        raise ConsumerError(f"crop policy hash mismatch: {path}")
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != len(timestamps):
        raise ConsumerError(f"crop ROI frame count mismatch: {path}")
    geometry: list[dict[str, Any]] = []
    for i, (frame, clock) in enumerate(zip(frames, timestamps)):
        if not isinstance(frame, dict) or frame.get("frame_index") != i:
            raise ConsumerError(f"invalid crop ROI frame index at {path}:{i}")
        if _rational_pair(frame, str(path)) != (clock[1], clock[2]):
            raise ConsumerError(f"crop ROI clock differs from rendered frame clock: {path}:{i}")
        for key in ("x", "y", "width", "height"):
            value = frame.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ConsumerError(f"invalid crop ROI coordinate {key} at {path}:{i}")
        if float(frame["width"]) <= 0 or float(frame["height"]) <= 0:
            raise ConsumerError(f"non-positive crop ROI size at {path}:{i}")
        geometry.append({
            "frame_index": i,
            "timestamp_num": clock[1],
            "timestamp_den": clock[2],
            "x": float(frame["x"]),
            "y": float(frame["y"]),
            "width": float(frame["width"]),
            "height": float(frame["height"]),
        })
    return _sha256_bytes(_canonical_json(geometry))


def _verify_mux_and_clock(
    row: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    root: Path,
    natural_path: Path,
    natural_pcm_sha256: str,
    natural_sample_count: int,
    ffmpeg: str,
    ffprobe: str,
    file_cache: dict[str, str],
) -> dict[str, Any]:
    render = result.get("render")
    mux_audio = result.get("mux_audio_readback")
    if not isinstance(render, dict) or not isinstance(mux_audio, dict):
        raise ConsumerError("PASS result row must contain render and mux_audio_readback objects")
    if render.get("status") != "PASS" or mux_audio.get("status") != "PASS":
        raise ConsumerError("PASS result row has a non-PASS render or mux readback")
    generator_pcm_sha = _require_sha(result.get("generator_pcm_sha256"), f"{row['row_id']}.generator_pcm_sha256")
    if render.get("generator_pcm_sha256") != generator_pcm_sha:
        raise ConsumerError(f"render does not bind the actual generator PCM for {row['row_id']}")

    silent = _check_file_ref(
        root,
        {"path": render.get("silent_video_path"), "file_sha256": render.get("silent_video_sha256")},
        where=f"{row['row_id']}.render.silent_video",
        cache=file_cache,
    )
    muxed = _check_file_ref(
        root,
        {"path": render.get("muxed_video_path"), "file_sha256": render.get("muxed_video_sha256")},
        where=f"{row['row_id']}.render.muxed_video",
        cache=file_cache,
    )
    silent_video_packet_sha = _video_hash_muxer_packet_sha256(silent, ffmpeg)
    muxed_video_packet_sha = _video_hash_muxer_packet_sha256(muxed, ffmpeg)
    silent_video_elementary_sha = _video_elementary_sha256(silent, ffmpeg, ffprobe)
    muxed_video_elementary_sha = _video_elementary_sha256(muxed, ffmpeg, ffprobe)
    if silent_video_packet_sha != muxed_video_packet_sha:
        raise ConsumerError(f"audio mux changed the FFmpeg video packet digest for {row['row_id']}")
    if silent_video_elementary_sha != muxed_video_elementary_sha:
        raise ConsumerError(f"mux changed the video elementary stream for {row['row_id']}")
    if _require_sha(render.get("video_stream_sha256"), f"{row['row_id']}.render.video_stream_sha256") != silent_video_packet_sha:
        raise ConsumerError(f"reported FFmpeg hash-muxer video packet SHA differs for {row['row_id']}")

    mux_probe = _probe(muxed, ffprobe)
    format_names = str(mux_probe.get("format", {}).get("format_name", "")).split(",")
    if "matroska" not in format_names:
        raise ConsumerError(f"final mux is not Matroska for {row['row_id']}")
    video_stream = _one_stream(mux_probe, "video", str(muxed))
    audio_stream = _one_stream(mux_probe, "audio", str(muxed))
    if audio_stream.get("codec_name") != "pcm_f32le":
        raise ConsumerError(f"final mux audio is not PCM float32 for {row['row_id']}")
    sample_rate = int(audio_stream.get("sample_rate", 0))
    channels = int(audio_stream.get("channels", 0))
    if sample_rate != 16_000 or channels != 1:
        raise ConsumerError(f"final mux audio clock is not 16 kHz mono for {row['row_id']}")

    expected_pcm_sha, expected_sample_count = _pcm_f32_readback_sha256(natural_path, ffmpeg, sample_rate, channels)
    actual_pcm_sha, actual_sample_count = _pcm_f32_readback_sha256(muxed, ffmpeg, sample_rate, channels)
    if expected_pcm_sha != natural_pcm_sha256 or expected_sample_count != natural_sample_count:
        raise ConsumerError(f"canonical natural PCM reader disagrees with its frozen reference for {row['row_id']}")
    if actual_pcm_sha != expected_pcm_sha or actual_sample_count != expected_sample_count:
        raise ConsumerError(f"muxed score audio is not the original natural PCM for {row['row_id']}")
    if _require_sha(mux_audio.get("decoded_float32_pcm_sha256"), f"{row['row_id']}.mux_audio_readback.decoded_float32_pcm_sha256") != actual_pcm_sha:
        raise ConsumerError(f"reported mux float32 PCM hash differs for {row['row_id']}")
    if mux_audio.get("sample_count") != actual_sample_count:
        raise ConsumerError(f"reported mux PCM sample count differs for {row['row_id']}")
    if mux_audio.get("sample_rate_hz") != sample_rate or mux_audio.get("channels") != channels:
        raise ConsumerError(f"reported mux PCM format differs for {row['row_id']}")
    if mux_audio.get("codec") != "pcm_f32le":
        raise ConsumerError(f"reported mux codec differs for {row['row_id']}")

    score = result.get("score")
    if not isinstance(score, dict):
        raise ConsumerError(f"missing score record for {row['row_id']}")
    if score.get("score_audio_pcm_sha256") != natural_pcm_sha256:
        raise ConsumerError(f"SyncNet score audio is not bound to original natural PCM for {row['row_id']}")

    fps_num = render.get("fps_num")
    fps_den = render.get("fps_den")
    frame_count = render.get("frame_count")
    if not isinstance(fps_num, int) or not isinstance(fps_den, int) or fps_num <= 0 or fps_den <= 0:
        raise ConsumerError(f"invalid rendered FPS for {row['row_id']}")
    if not isinstance(frame_count, int) or frame_count <= 0:
        raise ConsumerError(f"invalid rendered frame count for {row['row_id']}")
    model = next((item for item in plan["models"] if item.get("model_id") == row["model_id"]), None)
    if not isinstance(model, dict) or model.get("fps_num") != fps_num or model.get("fps_den") != fps_den:
        raise ConsumerError(f"render FPS differs from frozen model registry for {row['row_id']}")
    if row["source_video"].get("fps_num") != fps_num or row["source_video"].get("fps_den") != fps_den:
        raise ConsumerError(f"render FPS differs from frozen source-video plan for {row['row_id']}")
    if row["source_video"].get("expected_frame_count") is not None and row["source_video"].get("expected_frame_count") != frame_count:
        raise ConsumerError(f"render frame count differs from frozen plan for {row['row_id']}")
    timestamp_path = _resolve_path(root, render.get("timestamps_path"), f"{row['row_id']}.render.timestamps_path")
    timestamps = _read_timestamp_sidecar(
        timestamp_path,
        _require_sha(render.get("timestamps_sha256"), f"{row['row_id']}.render.timestamps_sha256"),
        frame_count,
        fps_num,
        fps_den,
    )
    silent_actual_pts = _probe_video_pts_clock(silent, ffprobe, fps_num, fps_den)
    muxed_actual_pts = _probe_video_pts_clock(muxed, ffprobe, fps_num, fps_den)
    if silent_actual_pts != timestamps:
        raise ConsumerError(f"silent render decoded PTS differ from the sealed timestamp sidecar for {row['row_id']}")
    if muxed_actual_pts != timestamps:
        raise ConsumerError(f"final mux decoded PTS differ from the sealed timestamp sidecar for {row['row_id']}")
    if silent_actual_pts != muxed_actual_pts:
        raise ConsumerError(f"audio mux changed the actual video PTS clock for {row['row_id']}")
    roi_path = _resolve_path(root, render.get("crop_roi_path"), f"{row['row_id']}.render.crop_roi_path")
    roi_sha = _require_sha(render.get("crop_roi_sha256"), f"{row['row_id']}.render.crop_roi_sha256")
    crop_policy_sha = _require_sha(render.get("crop_policy_sha256"), f"{row['row_id']}.render.crop_policy_sha256")
    roi_geometry_sha = _check_roi_sidecar(roi_path, roi_sha, timestamps, crop_policy_sha)
    if render.get("crop_frame_count") != frame_count:
        raise ConsumerError(f"crop frame count differs from render for {row['row_id']}")
    if render.get("crop_timestamps_sha256") != render.get("timestamps_sha256"):
        raise ConsumerError(f"crop timestamp hash differs from render for {row['row_id']}")

    # The final mux itself must preserve the complete original float32 source.
    # The SyncNet input is a different artifact: a direct-source PCM16 window
    # aligned to the exact selected crop track, remuxed with that crop video.
    source_pcm16_bytes, source_pcm16_count = _pcm16_readback_bytes(natural_path, ffmpeg, sample_rate, channels)
    mux_pcm16_bytes, mux_pcm16_count = _pcm16_readback_bytes(muxed, ffmpeg, sample_rate, channels)
    if source_pcm16_bytes != mux_pcm16_bytes or source_pcm16_count != mux_pcm16_count:
        raise ConsumerError(f"full-source PCM16 decode differs between natural input and exact-float32 mux for {row['row_id']}")
    primary_audio_audit = _verify_primary_pcm16_window(
        score,
        render,
        plan=plan,
        root=root,
        natural_path=natural_path,
        natural_pcm_sha256=natural_pcm_sha256,
        natural_sample_count=natural_sample_count,
        timestamps=timestamps,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        file_cache=file_cache,
    )

    stream_fps_text = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")
    if not isinstance(stream_fps_text, str) or "/" not in stream_fps_text:
        raise ConsumerError(f"ffprobe did not report rational FPS for {row['row_id']}")
    stream_fps = Fraction(stream_fps_text)
    if stream_fps != Fraction(fps_num, fps_den):
        raise ConsumerError(f"ffprobe FPS differs from result metadata for {row['row_id']}")
    audio_start = float(audio_stream.get("start_time", mux_probe.get("format", {}).get("start_time", 0.0)))
    mux_protocol = plan["protocol"]["mux"]
    expected_start = float(mux_protocol["audio_start_time_s"])
    start_tolerance = float(mux_protocol["audio_start_time_tolerance_s"])
    if abs(audio_start - expected_start) > start_tolerance:
        raise ConsumerError(f"mux audio start time differs from plan for {row['row_id']}")

    return {
        "muxed_video_sha256": _sha256_file(muxed),
        "video_stream_sha256": silent_video_packet_sha,
        "video_packet_hash_scheme": "ffmpeg_hash_muxer_copy_packets_sha256",
        "video_elementary_serialized_sha256": silent_video_elementary_sha,
        "silent_video_actual_pts_sha256": _clock_sha256(silent_actual_pts),
        "muxed_video_actual_pts_sha256": _clock_sha256(muxed_actual_pts),
        "muxed_pcm_float32_sha256": actual_pcm_sha,
        "muxed_full_source_pcm16_sha256": _sha256_bytes(mux_pcm16_bytes),
        "score_preprocessed_pcm16_sha256": primary_audio_audit["primary_pcm16_window_sha256"],
        "sample_count": actual_sample_count,
        **primary_audio_audit,
        "fps_num": fps_num,
        "fps_den": fps_den,
        "frame_count": frame_count,
        "timestamps_sha256": render.get("timestamps_sha256"),
        "crop_roi_sha256": roi_sha,
        "crop_roi_geometry_sha256": roi_geometry_sha,
        "crop_policy_sha256": crop_policy_sha,
        "query_clock": timestamps,
        "query_clock_by_track_frame": primary_audio_audit["_query_clock_by_track_frame"],
    }


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return _sha256_bytes(contiguous.tobytes(order="C"))


def _load_score_arrays(root: Path, score: Mapping[str, Any], *, row_id: str, file_cache: dict[str, str]) -> dict[str, np.ndarray]:
    ref = score.get("raw_matrix_npz")
    if not isinstance(ref, dict):
        raise ConsumerError(f"missing score.raw_matrix_npz for {row_id}")
    npz_path = _check_file_ref(root, ref, where=f"{row_id}.score.raw_matrix_npz", cache=file_cache)
    arrays_meta = ref.get("arrays")
    if not isinstance(arrays_meta, dict):
        raise ConsumerError(f"score raw matrix array metadata missing for {row_id}")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            missing = REQUIRED_SCORE_ARRAYS.difference(archive.files)
            if missing:
                raise ConsumerError(f"score NPZ missing arrays for {row_id}: {sorted(missing)}")
            arrays = {key: np.asarray(archive[key]) for key in REQUIRED_SCORE_ARRAYS}
    except ConsumerError:
        raise
    except Exception as exc:
        raise ConsumerError(f"cannot read score NPZ for {row_id}: {exc}") from exc

    for key, array in arrays.items():
        meta = arrays_meta.get(key)
        if not isinstance(meta, dict):
            raise ConsumerError(f"missing array seal for {row_id}.{key}")
        if meta.get("dtype") not in (array.dtype.str, array.dtype.name):
            raise ConsumerError(f"array dtype mismatch for {row_id}.{key}: {meta.get('dtype')} vs {array.dtype}")
        if meta.get("shape") != list(array.shape):
            raise ConsumerError(f"array shape mismatch for {row_id}.{key}")
        if _require_sha(meta.get("array_sha256"), f"{row_id}.{key}.array_sha256") != _array_sha256(array):
            raise ConsumerError(f"array byte SHA mismatch for {row_id}.{key}")

    n = arrays["query_frame_index"].size
    expected_shapes = {
        "query_frame_index": (n,),
        "query_timestamp_num": (n,),
        "query_timestamp_den": (n,),
        "track_frame_index": (n,),
        "valid_query_mask": (n,),
        "distance_by_query_lag": (n, 31),
        "lag_offsets_frames": (31,),
    }
    for key, expected in expected_shapes.items():
        if arrays[key].shape != expected:
            raise ConsumerError(f"score array {row_id}.{key} has shape {arrays[key].shape}, expected {expected}")
    if arrays["query_frame_index"].dtype.kind not in "iu" or arrays["query_timestamp_num"].dtype.kind not in "iu" or arrays["query_timestamp_den"].dtype.kind not in "iu" or arrays["track_frame_index"].dtype.kind not in "iu":
        raise ConsumerError(f"score query identity arrays have non-integer dtype for {row_id}")
    if arrays["valid_query_mask"].dtype.kind != "b":
        raise ConsumerError(f"valid_query_mask must be boolean for {row_id}")
    if arrays["distance_by_query_lag"].dtype != np.dtype("<f4") and arrays["distance_by_query_lag"].dtype != np.dtype("float32"):
        raise ConsumerError(f"distance matrix must be float32 for {row_id}")
    if arrays["lag_offsets_frames"].dtype.kind not in "iu":
        raise ConsumerError(f"lag axis must be integer for {row_id}")
    exact_dtypes = {
        "query_frame_index": np.dtype("<i4"),
        "query_timestamp_num": np.dtype("<i8"),
        "query_timestamp_den": np.dtype("<i8"),
        "track_frame_index": np.dtype("<i4"),
        "valid_query_mask": np.dtype("bool"),
        "distance_by_query_lag": np.dtype("<f4"),
        "lag_offsets_frames": np.dtype("<i4"),
    }
    for key, expected_dtype in exact_dtypes.items():
        if arrays[key].dtype != expected_dtype:
            raise ConsumerError(f"score array {row_id}.{key} dtype {arrays[key].dtype} != frozen {expected_dtype}")
    if np.any(arrays["query_timestamp_den"] <= 0):
        raise ConsumerError(f"query timestamp denominator must be positive for {row_id}")
    if np.any(arrays["valid_query_mask"] & (arrays["track_frame_index"] < 0)):
        raise ConsumerError(f"valid query has negative track frame index for {row_id}")
    if not np.isfinite(arrays["distance_by_query_lag"][arrays["valid_query_mask"]]).all():
        raise ConsumerError(f"valid query distance values are non-finite for {row_id}")
    keys = _query_keys(arrays)
    valid_keys = [key for key, valid in zip(keys, arrays["valid_query_mask"].tolist()) if valid]
    if len(valid_keys) != len(set(valid_keys)):
        raise ConsumerError(f"duplicate valid query identity in score NPZ for {row_id}")
    return arrays


def _validate_query_clock_map(
    arrays: Mapping[str, np.ndarray],
    media: Mapping[str, Any],
    *,
    row_id: str,
) -> None:
    """Bind each valid SyncNet row to its crop-local row and global source clock."""

    frame_map = media.get("query_clock_by_track_frame")
    if not isinstance(frame_map, list) or not frame_map:
        raise ConsumerError(f"selected crop frame map is unavailable for score queries in {row_id}")
    expected_query_count = media.get("official_query_count")
    if not isinstance(expected_query_count, int) or arrays["query_frame_index"].size != expected_query_count:
        raise ConsumerError(
            f"score query count for {row_id} differs from official min(video_frames, audio_frames)-5 rule: "
            f"expected={expected_query_count}, actual={arrays['query_frame_index'].size}"
        )
    valid = arrays["valid_query_mask"]
    for row_index in np.flatnonzero(valid):
        crop_local_index = int(row_index)
        if not 0 <= crop_local_index < len(frame_map):
            raise ConsumerError(f"valid query {row_id}[{row_index}] row index is outside selected crop")
        source_clock_frame, timestamp_num, timestamp_den = frame_map[crop_local_index]
        query_frame = int(arrays["query_frame_index"][row_index])
        track_frame = int(arrays["track_frame_index"][row_index])
        query_stamp = Fraction(
            int(arrays["query_timestamp_num"][row_index]),
            int(arrays["query_timestamp_den"][row_index]),
        )
        if query_frame != source_clock_frame:
            raise ConsumerError(
                f"valid query {row_id}[{row_index}] source-clock frame {query_frame} does not map from "
                f"crop-local row {crop_local_index} (expected {source_clock_frame})"
            )
        if track_frame != source_clock_frame:
            raise ConsumerError(
                f"valid query {row_id}[{row_index}] track-frame coordinate {track_frame} "
                f"does not match mapped source-clock frame {source_clock_frame}"
            )
        if query_stamp != Fraction(timestamp_num, timestamp_den):
            raise ConsumerError(
                f"valid query {row_id}[{row_index}] timestamp does not match its crop-local source frame"
            )


def _query_keys(arrays: Mapping[str, np.ndarray]) -> list[tuple[int, int, int, int]]:
    keys: list[tuple[int, int, int, int]] = []
    for frame, num, den, track in zip(
        arrays["query_frame_index"],
        arrays["query_timestamp_num"],
        arrays["query_timestamp_den"],
        arrays["track_frame_index"],
    ):
        stamp = Fraction(int(num), int(den))
        keys.append((int(frame), stamp.numerator, stamp.denominator, int(track)))
    return keys


def _query_sort_key(key: tuple[int, int, int, int]) -> tuple[int, Fraction, int]:
    return key[0], Fraction(key[1], key[2]), key[3]


def _configure_torch_reduction(syncnet: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Pin PyTorch CPU semantics for the official SyncNet curve reduction."""
    reduction = syncnet.get("score_reduction")
    if not isinstance(reduction, dict):
        raise ConsumerError("plan.protocol.syncnet.score_reduction must be an object")
    fixed = {
        "backend": "pytorch_cpu",
        "matrix_layout": "query_by_lag_float32",
        "stacked_layout": "lag_by_query_contiguous_float32",
        "curve_operation": "torch.mean(torch.stack(query_distance_vectors, dim=1), dim=1)",
        "median_operation": "torch.median(curve)",
        "minimum_operation": "torch.min(curve, dim=0)",
        "contrast_dtype": "float32",
        "median_semantics": "lower_middle_order_statistic",
        "num_threads": 1,
    }
    for key, expected in fixed.items():
        if reduction.get(key) != expected:
            raise ConsumerError(f"score_reduction.{key} must be frozen as {expected!r}")
    expected_version = reduction.get("torch_version")
    if not isinstance(expected_version, str) or not expected_version:
        raise ConsumerError("score_reduction.torch_version must pin the exact PyTorch runtime version")
    try:
        import torch
    except Exception as exc:
        raise ConsumerError(f"the frozen SyncNet score reduction requires PyTorch: {exc}") from exc
    actual_version = str(torch.__version__)
    if actual_version != expected_version:
        raise ConsumerError(
            f"PyTorch runtime version differs from the frozen score reduction: "
            f"actual={actual_version!r}, expected={expected_version!r}"
        )
    try:
        torch.set_num_threads(1)
    except Exception as exc:
        raise ConsumerError(f"cannot set frozen PyTorch CPU thread count: {exc}") from exc
    actual_threads = int(torch.get_num_threads())
    if actual_threads != 1:
        raise ConsumerError(f"PyTorch CPU thread count is {actual_threads}, expected frozen 1")
    return torch, {
        **fixed,
        "torch_version": actual_version,
        "torch_git_version": getattr(getattr(torch, "version", None), "git_version", None),
        "device": "cpu",
    }


def _compute_curve_metrics(
    matrix: np.ndarray,
    row_indices: Sequence[int],
    *,
    torch: Any,
    lag_offsets: np.ndarray,
    zero_lag_column: int,
    anchor_column: int,
) -> tuple[list[float], dict[str, float | int]]:
    selected = matrix[np.asarray(row_indices, dtype=np.int64)]
    if selected.dtype != np.dtype("float32"):
        raise ConsumerError("official score reduction requires the frozen float32 distance matrix")
    if selected.ndim != 2 or selected.shape[1] != 31 or not len(selected):
        raise ConsumerError("score reduction requires a nonempty query-by-31-lag matrix")
    # Official SyncNet creates torch.stack(dists, 1), a contiguous [31, N]
    # tensor, then applies torch.mean(..., 1). Keep the same operator, layout,
    # query order, and dtype instead of substituting a NumPy reduction.
    stacked = torch.from_numpy(np.ascontiguousarray(selected.T))
    if stacked.dtype != torch.float32 or tuple(stacked.shape) != (31, len(selected)):
        raise ConsumerError("score matrix is not representable as frozen [31, queries] float32")
    curve_tensor = torch.mean(stacked, dim=1)
    minimum_result = torch.min(curve_tensor, dim=0)
    best_column = int(minimum_result.indices.item())
    b_tensor = torch.median(curve_tensor)
    d_min_tensor = minimum_result.values
    d_zero_tensor = curve_tensor[zero_lag_column]
    d_anchor_tensor = curve_tensor[anchor_column]
    c_best_tensor = b_tensor - d_min_tensor
    c_anchor_tensor = b_tensor - d_anchor_tensor
    curve = curve_tensor.detach().cpu().numpy()
    b32 = np.float32(b_tensor.item())
    d_min32 = np.float32(d_min_tensor.item())
    d_zero32 = np.float32(d_zero_tensor.item())
    d_anchor32 = np.float32(d_anchor_tensor.item())
    c_best32 = np.float32(c_best_tensor.item())
    c_anchor32 = np.float32(c_anchor_tensor.item())
    official_offset = -int(lag_offsets[best_column])

    curve64 = np.mean(selected.astype(np.float64), axis=0, dtype=np.float64)
    b64 = float(np.median(curve64))
    d_min64 = float(np.min(curve64))
    c_best64 = b64 - d_min64
    c_anchor64 = b64 - float(curve64[anchor_column])
    float64_diagnostic = {
        "mean_curve_max_abs_difference": float(np.max(np.abs(curve64 - curve.astype(np.float64)))),
        "Cbest_abs_difference": abs(c_best64 - float(c_best32)),
        "Canchor_abs_difference": abs(c_anchor64 - float(c_anchor32)),
    }
    return curve.astype(float).tolist(), {
        "B": float(b32),
        "Dmin": float(d_min32),
        "D0": float(d_zero32),
        "Cbest": float(c_best32),
        "Canchor": float(c_anchor32),
        "best_lag_frames": int(lag_offsets[best_column]),
        "offset_frames": official_offset,
        "official_av_offset_frames": official_offset,
        "anchor_column": int(anchor_column),
        "zero_lag_column": int(zero_lag_column),
        "independent_float64_diagnostic": float64_diagnostic,
    }


def _close(actual: Any, expected: Any, atol: float, rtol: float, where: str) -> None:
    try:
        actual_f = float(actual)
        expected_f = float(expected)
    except (TypeError, ValueError) as exc:
        raise ConsumerError(f"non-numeric stored metric at {where}") from exc
    if not math.isfinite(actual_f) or not math.isfinite(expected_f):
        raise ConsumerError(f"non-finite metric at {where}")
    if abs(actual_f - expected_f) > atol + rtol * abs(expected_f):
        raise ConsumerError(f"stored metric mismatch at {where}: stored={actual_f}, rebuilt={expected_f}")


def _support_mask_sha(support_id: str, keys: Sequence[tuple[int, int, int, int]]) -> str:
    return _sha256_bytes(_canonical_json({"support_id": support_id, "keys": [list(k) for k in keys]}))


def _compare_saved_score_summary(
    result: Mapping[str, Any],
    support_id: str,
    curve: Sequence[float],
    metrics: Mapping[str, Any],
    support_hash: str,
    query_count: int,
    *,
    atol: float,
    rtol: float,
) -> None:
    score = result.get("score")
    if not isinstance(score, dict):
        raise ConsumerError(f"missing score object for {result.get('row_id')}")
    saved_curves = score.get("lag_curve_by_support")
    saved_supports = score.get("support_by_support")
    saved_metrics = score.get("metrics_by_support")
    if not all(isinstance(v, dict) for v in (saved_curves, saved_supports, saved_metrics)):
        raise ConsumerError(f"score support summaries malformed for {result.get('row_id')}")
    if support_id not in saved_curves or support_id not in saved_supports or support_id not in saved_metrics:
        raise ConsumerError(f"missing {support_id} summary for {result.get('row_id')}")
    saved_curve = saved_curves[support_id]
    if not isinstance(saved_curve, list) or len(saved_curve) != 31:
        raise ConsumerError(f"stored lag curve must have 31 values for {result.get('row_id')}/{support_id}")
    for i, (got, rebuilt) in enumerate(zip(saved_curve, curve)):
        _close(got, rebuilt, atol, rtol, f"{result.get('row_id')}.{support_id}.curve[{i}]")
    support_record = saved_supports[support_id]
    if not isinstance(support_record, dict):
        raise ConsumerError(f"support record malformed for {result.get('row_id')}/{support_id}")
    if support_record.get("support_mask_sha256") != support_hash:
        raise ConsumerError(f"support mask hash mismatch for {result.get('row_id')}/{support_id}")
    if support_record.get("query_count") != query_count:
        raise ConsumerError(f"query count mismatch for {result.get('row_id')}/{support_id}")
    stored_metrics = saved_metrics[support_id]
    if not isinstance(stored_metrics, dict):
        raise ConsumerError(f"metric record malformed for {result.get('row_id')}/{support_id}")
    for key in ("B", "Dmin", "D0", "Cbest", "Canchor", "best_lag_frames", "offset_frames", "official_av_offset_frames", "anchor_column", "zero_lag_column"):
        if key not in stored_metrics or key not in metrics:
            raise ConsumerError(f"missing metric {key} for {result.get('row_id')}/{support_id}")
        if key in ("best_lag_frames", "offset_frames", "official_av_offset_frames", "anchor_column", "zero_lag_column"):
            if int(stored_metrics[key]) != int(metrics[key]):
                raise ConsumerError(f"integer metric mismatch {key} for {result.get('row_id')}/{support_id}")
        else:
            _close(stored_metrics[key], metrics[key], atol, rtol, f"{result.get('row_id')}.{support_id}.{key}")


def _validate_plan(plan: Mapping[str, Any]) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    Any,
    dict[str, Any],
    list[dict[str, Any]],
]:
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("status") != "FROZEN":
        raise ConsumerError("plan schema/status must be cross-tfg-audio-eval-v1/FROZEN")
    if plan.get("plan_sha256") != _payload_sha256(plan, "plan_sha256"):
        raise ConsumerError("embedded plan_sha256 does not match canonical plan payload")
    _require_sha(plan.get("split_manifest_sha256"), "plan.split_manifest_sha256")
    _require_sha(plan.get("config_sha256"), "plan.config_sha256")
    planned_rows_raw = plan.get("rows", [])
    has_dev_rows = any(isinstance(row, dict) and row.get("split") == "dev" for row in planned_rows_raw)
    has_confirm_rows = any(isinstance(row, dict) and row.get("split") == "confirm" for row in planned_rows_raw)
    for key in ("fit_speaker_ids", "dev_speaker_ids", "confirm_speaker_ids"):
        value = plan.get(key)
        allow_empty = key == "confirm_speaker_ids" or (key == "dev_speaker_ids" and not has_dev_rows)
        if not isinstance(value, list) or (not allow_empty and not value) or any(not isinstance(speaker, str) or not speaker for speaker in value) or len(value) != len(set(value)):
            qualifier = "unique ordered string list" if allow_empty else "non-empty unique ordered string list"
            raise ConsumerError(f"plan.{key} must be a {qualifier}")
    fit = set(plan["fit_speaker_ids"])
    dev = set(plan["dev_speaker_ids"])
    confirm = set(plan["confirm_speaker_ids"])
    if fit & dev or fit & confirm or dev & confirm:
        raise ConsumerError("speaker IDs leak across fit/dev/confirm splits")
    seed_family = plan.get("generation_seed_family")
    if not isinstance(seed_family, dict):
        raise ConsumerError("plan.generation_seed_family must predeclare render/evaluation seeds by split")
    seeds_by_split: dict[str, tuple[Any, ...]] = {}
    seed_coverage_issues: list[dict[str, Any]] = []
    for split_id, seed_values in seed_family.items():
        if split_id not in ("fit", "dev", "confirm"):
            raise ConsumerError(f"unknown generation_seed_family split: {split_id}")
        if not isinstance(seed_values, list):
            raise ConsumerError(f"generation_seed_family.{split_id} must be an ordered list")
        for seed_value in seed_values:
            if isinstance(seed_value, bool) or not (
                isinstance(seed_value, int) or (isinstance(seed_value, str) and seed_value)
            ):
                raise ConsumerError(f"generation_seed_family.{split_id} entries must be integers or nonempty strings")
        if len(seed_values) != len(set((type(value).__name__, value) for value in seed_values)):
            raise ConsumerError(f"generation_seed_family.{split_id} contains duplicate values")
        if split_id == "dev" and has_dev_rows and len(seed_values) < 3:
            raise ConsumerError("development generation seed family must contain at least three predeclared seeds")
        seeds_by_split[split_id] = tuple(seed_values)
    if has_dev_rows and not seeds_by_split.get("dev"):
        raise ConsumerError("a plan with development evaluation rows must predeclare at least three development render seeds")
    exposed_speakers_raw = plan.get("historical_exposed_speaker_ids", [])
    exposed_samples_raw = plan.get("historical_exposed_sample_ids", [])
    if not isinstance(exposed_speakers_raw, list) or not isinstance(exposed_samples_raw, list):
        raise ConsumerError("historical exposure rosters must be ordered lists")
    if len(exposed_speakers_raw) != len(set(exposed_speakers_raw)) or len(exposed_samples_raw) != len(set(exposed_samples_raw)):
        raise ConsumerError("historical exposure rosters must be unique")
    exposed_speakers = set(exposed_speakers_raw)
    exposed_samples = set(exposed_samples_raw)
    if any(not isinstance(value, str) or not value for value in exposed_speakers | exposed_samples):
        raise ConsumerError("historical exposure rosters must contain only non-empty string IDs")
    if confirm & exposed_speakers:
        raise ConsumerError("historically exposed speakers appear in confirmation")
    if not isinstance(plan.get("models"), list) or not plan["models"]:
        raise ConsumerError("plan.models must be a non-empty list")
    if not isinstance(plan.get("methods"), list) or not plan["methods"]:
        raise ConsumerError("plan.methods must be a non-empty list")
    if not isinstance(plan.get("rows"), list) or not plan["rows"]:
        raise ConsumerError("plan.rows must be a non-empty list")

    models: dict[str, dict[str, Any]] = {}
    for model in plan["models"]:
        if not isinstance(model, dict) or not isinstance(model.get("model_id"), str):
            raise ConsumerError("every model must have a model_id")
        model_id = model["model_id"]
        if model_id in models:
            raise ConsumerError(f"duplicate model_id: {model_id}")
        for key in ("model_sha256", "code_sha256", "score_model_sha256", "preprocessing_sha256"):
            _require_sha(model.get(key), f"plan.models[{model_id}].{key}")
        if not isinstance(model.get("fps_num"), int) or not isinstance(model.get("fps_den"), int) or model["fps_num"] <= 0 or model["fps_den"] <= 0:
            raise ConsumerError(f"model {model_id} lacks rational FPS")
        models[model_id] = model

    methods: dict[str, dict[str, Any]] = {}
    for method in plan["methods"]:
        if not isinstance(method, dict) or not isinstance(method.get("method_id"), str):
            raise ConsumerError("every method must have a method_id")
        method_id = method["method_id"]
        if method_id in methods:
            raise ConsumerError(f"duplicate method_id: {method_id}")
        _require_sha(method.get("config_sha256"), f"plan.methods[{method_id}].config_sha256")
        checkpoint = method.get("checkpoint_sha256")
        if checkpoint is not None:
            _require_sha(checkpoint, f"plan.methods[{method_id}].checkpoint_sha256")
        if method.get("kind") == "rms_fixed":
            target_rms = method.get("target_rms")
            tolerance = method.get("target_rms_tolerance")
            if not isinstance(target_rms, (int, float)) or not math.isfinite(float(target_rms)) or float(target_rms) <= 0:
                raise ConsumerError(f"fixed-RMS method {method_id} needs a positive finite target_rms")
            if not isinstance(tolerance, (int, float)) or not math.isfinite(float(tolerance)) or float(tolerance) < 0:
                raise ConsumerError(f"fixed-RMS method {method_id} needs a finite nonnegative target_rms_tolerance")
        methods[method_id] = method
    method_kinds = {str(method.get("kind")) for method in methods.values()}
    if "identity" not in method_kinds or "rms_fixed" not in method_kinds:
        raise ConsumerError("the frozen method roster must include identity and fixed-RMS controls")
    if "identity_repeat" in method_kinds:
        raise ConsumerError("identity repeats are QC records, not efficacy methods")

    protocol = plan.get("protocol")
    if not isinstance(protocol, dict):
        raise ConsumerError("plan.protocol must be an object")
    audio = protocol.get("audio")
    if not isinstance(audio, dict) or audio.get("sample_rate_hz") != 16_000 or audio.get("channels") != 1 or audio.get("dtype") != "float32_le":
        raise ConsumerError("plan must freeze canonical mono 16 kHz float32 audio")
    syncnet = protocol.get("syncnet")
    if not isinstance(syncnet, dict):
        raise ConsumerError("plan.protocol.syncnet must be an object")
    primary_audio_input = syncnet.get("primary_audio_input")
    if not isinstance(primary_audio_input, dict):
        raise ConsumerError("plan.protocol.syncnet.primary_audio_input must freeze the primary source PCM16 window")
    fixed_primary_audio_input = {
        "mode": "canonical_natural_pcm16_crop_window",
        "source": "natural_pcm",
        "codec": "pcm_s16le",
        "sample_rate_hz": 16_000,
        "channels": 1,
        "operation": "ffmpeg_decode_full_source_then_slice_sample_window",
        "frame_clock": "original_render_source_pts",
        "argv_template": [
            "-v", "error", "-i", "{natural_pcm_path}", "-map", "0:a:0", "-vn",
            "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1",
        ],
    }
    for key, expected in fixed_primary_audio_input.items():
        if primary_audio_input.get(key) != expected:
            raise ConsumerError(f"plan.protocol.syncnet.primary_audio_input.{key} must be frozen as {expected!r}")
    _require_sha(primary_audio_input.get("ffmpeg_sha256"), "plan.protocol.syncnet.primary_audio_input.ffmpeg_sha256")
    _require_sha(primary_audio_input.get("code_sha256"), "plan.protocol.syncnet.primary_audio_input.code_sha256")
    torch_module, score_reduction_runtime = _configure_torch_reduction(syncnet)
    lag_axis = syncnet.get("lag_offsets_frames")
    if lag_axis != list(range(-15, 16)):
        raise ConsumerError("SyncNet lag axis must be the frozen 31 columns -15..15")
    if syncnet.get("best_lag_definition") != "lag_offsets_frames[best_column]":
        raise ConsumerError("plan must define best_lag_frames from the frozen lag column axis")
    if syncnet.get("official_offset_definition") != "negative_best_lag_frames":
        raise ConsumerError("plan must freeze official_av_offset_frames = -best_lag_frames")
    zero_column = syncnet.get("zero_lag_column")
    anchor_column = syncnet.get("anchor_column")
    if zero_column != 15 or not isinstance(anchor_column, int) or not 0 <= anchor_column < 31:
        raise ConsumerError("plan must freeze zero-lag column 15 and a valid fixed-anchor column")
    atol = float(syncnet.get("metric_atol", 2e-6))
    rtol = float(syncnet.get("metric_rtol", 2e-6))
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ConsumerError("invalid frozen metric tolerances")
    roi_protocol = protocol.get("roi")
    if not isinstance(roi_protocol, dict) or roi_protocol.get("mode") not in ("shared_fixed", "native_official"):
        raise ConsumerError("plan.protocol.roi.mode must be shared_fixed or native_official")
    mux = protocol.get("mux")
    if not isinstance(mux, dict) or mux.get("container") != "matroska" or mux.get("audio_codec") != "pcm_f32le":
        raise ConsumerError("plan must freeze a Matroska/PCM_F32LE validation mux")
    start = mux.get("audio_start_time_s")
    start_tolerance = mux.get("audio_start_time_tolerance_s")
    if not isinstance(start, (int, float)) or not math.isfinite(float(start)):
        raise ConsumerError("plan.protocol.mux.audio_start_time_s must be finite")
    if not isinstance(start_tolerance, (int, float)) or not math.isfinite(float(start_tolerance)) or float(start_tolerance) < 0:
        raise ConsumerError("plan.protocol.mux.audio_start_time_tolerance_s must be finite and nonnegative")
    supports_raw = plan.get("supports")
    if not isinstance(supports_raw, list):
        raise ConsumerError("plan.supports must be a list")
    supports: dict[str, dict[str, Any]] = {}
    for support in supports_raw:
        if not isinstance(support, dict) or not isinstance(support.get("support_id"), str):
            raise ConsumerError("malformed plan support")
        sid = support["support_id"]
        if sid in supports:
            raise ConsumerError(f"duplicate support_id: {sid}")
        trim = support.get("trim_each_end")
        minimum = support.get("minimum_queries")
        if not isinstance(trim, int) or trim < 0 or not isinstance(minimum, int) or minimum < 1:
            raise ConsumerError(f"invalid support definition for {sid}")
        supports[sid] = support
    if supports.get("full", {}).get("trim_each_end") != 0 or supports.get("guard20", {}).get("trim_each_end") != 20:
        raise ConsumerError("plan must freeze full and historical guard20 supports")

    rows: dict[str, dict[str, Any]] = {}
    row_group_methods: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    row_group_sources: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    seeds_by_context: dict[tuple[Any, ...], set[tuple[str, Any]]] = defaultdict(set)
    pcm_splits: dict[str, set[str]] = defaultdict(set)
    for row in plan["rows"]:
        if not isinstance(row, dict):
            raise ConsumerError("planned row must be an object")
        missing = [key for key in IDENTITY_FIELDS if key not in row]
        if missing:
            raise ConsumerError(f"planned row missing identity fields: {missing}")
        row_id = row["row_id"]
        if not isinstance(row_id, str) or not row_id or row_id in rows:
            raise ConsumerError(f"invalid or duplicate row_id: {row_id}")
        if row.get("model_id") not in models or row.get("method_id") not in methods:
            raise ConsumerError(f"row {row_id} refers to undeclared model or method")
        split = row.get("split")
        speaker = row.get("speaker_id")
        sample_id = row.get("sample_id")
        if (split == "fit" and speaker not in fit) or (split == "dev" and speaker not in dev) or (split == "confirm" and speaker not in confirm):
            raise ConsumerError(f"row {row_id} has split/speaker mismatch")
        if split not in ("fit", "dev", "confirm"):
            raise ConsumerError(f"row {row_id} has unknown split")
        seed_value = row.get("seed")
        if isinstance(seed_value, bool) or not (
            isinstance(seed_value, int) or (isinstance(seed_value, str) and seed_value)
        ):
            raise ConsumerError(f"row {row_id} seed must be an integer or nonempty string")
        if split not in seeds_by_split or seed_value not in seeds_by_split[split]:
            raise ConsumerError(f"row {row_id} seed is absent from its frozen generation_seed_family")
        seeds_by_context[(split, sample_id, speaker, row.get("model_id"), row.get("face_id"))].add(
            (type(seed_value).__name__, seed_value)
        )
        is_historically_exposed = speaker in exposed_speakers or sample_id in exposed_samples
        if row.get("historical_exposed") is not is_historically_exposed:
            raise ConsumerError(f"historical_exposed flag disagrees with frozen history roster for {row_id}")
        if not isinstance(row.get("historical_exposed"), bool):
            raise ConsumerError(f"historical_exposed must be a JSON boolean for {row_id}")
        if split == "confirm" and is_historically_exposed:
            raise ConsumerError(f"historically exposed sample/speaker appears in confirmation: {row_id}")
        _require_sha(row.get("method_config_sha256"), f"{row_id}.method_config_sha256")
        if row.get("method_config_sha256") != methods[row["method_id"]].get("config_sha256"):
            raise ConsumerError(f"method config hash differs from plan method table for {row_id}")
        if row.get("checkpoint_sha256") != methods[row["method_id"]].get("checkpoint_sha256"):
            raise ConsumerError(f"checkpoint hash differs from plan method table for {row_id}")
        for ref_key in ("natural_pcm", "source_face", "source_video"):
            if not isinstance(row.get(ref_key), dict):
                raise ConsumerError(f"planned row {row_id} missing {ref_key} reference")
        if not isinstance(row.get("score_audio"), dict):
            raise ConsumerError(f"planned row {row_id} missing score_audio reference")
        if row["score_audio"].get("pcm_sha256") != row["natural_pcm"].get("pcm_sha256"):
            raise ConsumerError(f"planned score audio is not original natural PCM for {row_id}")
        if row["score_audio"].get("file_sha256") != row["natural_pcm"].get("file_sha256"):
            raise ConsumerError(f"planned score audio file is not the original natural file for {row_id}")
        model = models[row["model_id"]]
        if row.get("audio_frontend") != model.get("audio_frontend") or row.get("model_role") != model.get("role"):
            raise ConsumerError(f"row {row_id} frontend/model role differs from model registry")
        if row["score_audio"].get("sample_count") != row["natural_pcm"].get("sample_count"):
            raise ConsumerError(f"planned score audio sample count differs from natural PCM for {row_id}")
        group_key = _pair_group_key(row)
        row_group_methods[group_key].append(row["method_id"])
        source_signature = tuple(
            row[field].get(key)
            for field, key in (
                ("natural_pcm", "file_sha256"),
                ("natural_pcm", "pcm_sha256"),
                ("score_audio", "file_sha256"),
                ("score_audio", "pcm_sha256"),
                ("source_face", "file_sha256"),
                ("source_video", "file_sha256"),
            )
        )
        prior_signature = row_group_sources.setdefault(group_key, source_signature)
        if source_signature != prior_signature:
            raise ConsumerError(f"paired methods do not share identical natural PCM/face/video inputs: {group_key}")
        pcm_splits[row["natural_pcm"].get("pcm_sha256")].add(split)
        rows[row_id] = row

    method_ids = set(methods)
    for group_key, group_methods in row_group_methods.items():
        if len(group_methods) != len(set(group_methods)) or set(group_methods) != method_ids:
            seed_coverage_issues.append({
                "row_id": "plan",
                "message": f"paired plan group has missing or duplicate method rows and is retained as unestimable: {group_key}",
                "issue_code": "INCOMPLETE_PLANNED_METHOD_FAMILY",
            })
    for context_key, observed_seed_tokens in seeds_by_context.items():
        split_id = context_key[0]
        expected_seed_tokens = {(type(value).__name__, value) for value in seeds_by_split[split_id]}
        if observed_seed_tokens != expected_seed_tokens:
            seed_coverage_issues.append({
                "row_id": "plan",
                "message": (
                    "planned context does not contain the complete predeclared seed family; "
                    f"missing seeds remain in the paired denominator: context={context_key}, "
                    f"observed={sorted(observed_seed_tokens, key=str)}, "
                    f"expected={sorted(expected_seed_tokens, key=str)}"
                ),
                "issue_code": "INCOMPLETE_PLANNED_SEED_FAMILY",
            })
    cross_split_pcm = [pcm_sha for pcm_sha, split_ids in pcm_splits.items() if len(split_ids) > 1]
    if cross_split_pcm:
        raise ConsumerError(f"identical canonical natural PCM is assigned across splits: {cross_split_pcm[:5]}")

    analysis = plan.get("analysis")
    if not isinstance(analysis, dict):
        raise ConsumerError("plan.analysis must be an object")
    contrast_map: dict[str, dict[str, Any]] = {}
    for contrast in analysis.get("contrasts", []):
        if not isinstance(contrast, dict) or not isinstance(contrast.get("contrast_id"), str):
            raise ConsumerError("malformed planned contrast")
        cid = contrast["contrast_id"]
        if cid in contrast_map or contrast.get("method_a") not in methods or contrast.get("method_b") not in methods:
            raise ConsumerError(f"invalid or duplicate contrast {cid}")
        if contrast["method_a"] == contrast["method_b"]:
            raise ConsumerError(f"contrast {cid} compares a method to itself")
        contrast_map[cid] = contrast
    endpoints: dict[str, dict[str, Any]] = {}
    if not isinstance(analysis.get("summary_endpoints"), list):
        raise ConsumerError("analysis.summary_endpoints must be a frozen list")
    if not analysis["summary_endpoints"] and (has_dev_rows or has_confirm_rows):
        raise ConsumerError("an evaluation plan with dev/confirm rows must declare at least one endpoint")
    for endpoint in analysis.get("summary_endpoints", []):
        if not isinstance(endpoint, dict) or not isinstance(endpoint.get("endpoint_id"), str):
            raise ConsumerError("malformed summary endpoint")
        eid = endpoint["endpoint_id"]
        if eid in endpoints:
            raise ConsumerError(f"duplicate summary endpoint {eid}")
        if endpoint.get("contrast_id") not in contrast_map or endpoint.get("support_id") not in supports:
            raise ConsumerError(f"endpoint {eid} refers to unknown contrast or support")
        split_scope = endpoint.get("split", "confirm")
        if split_scope not in ("dev", "confirm"):
            raise ConsumerError(f"endpoint {eid} has unsupported split {split_scope}")
        if endpoint.get("roi_estimand", roi_protocol["mode"]) != roi_protocol["mode"]:
            raise ConsumerError(f"endpoint {eid} ROI estimand differs from frozen protocol")
        if endpoint.get("metric") not in ("Cbest", "Canchor", "B", "Dmin", "D0", "offset_frames"):
            raise ConsumerError(f"endpoint {eid} has unknown metric")
        scope = endpoint.get("model_scope")
        if scope == "model" and endpoint.get("model_id") not in models:
            raise ConsumerError(f"endpoint {eid} refers to unknown model")
        if scope == "equal_model_macro":
            model_ids = endpoint.get("model_ids")
            if not isinstance(model_ids, list) or not model_ids or any(m not in models for m in model_ids):
                raise ConsumerError(f"endpoint {eid} has invalid macro model_ids")
        elif scope != "model":
            raise ConsumerError(f"endpoint {eid} has unknown model_scope")
        endpoints[eid] = endpoint
    family_ids = analysis.get("simultaneous_family_ids")
    if not isinstance(family_ids, list) or len(family_ids) != len(set(family_ids)):
        raise ConsumerError("analysis.simultaneous_family_ids must be a unique list")
    if any(eid not in endpoints for eid in family_ids):
        raise ConsumerError("simultaneous family refers to an undeclared endpoint")
    for endpoint_id in family_ids:
        if endpoints[endpoint_id].get("split", "confirm") != "confirm":
            raise ConsumerError(f"non-confirmatory endpoint {endpoint_id} enters the simultaneous family")
    confirm_endpoint_ids = {
        endpoint_id for endpoint_id, endpoint in endpoints.items()
        if endpoint.get("split", "confirm") == "confirm"
    }
    if set(family_ids) != confirm_endpoint_ids:
        raise ConsumerError("the simultaneous family must contain every and only frozen confirmatory endpoint")

    bootstrap = analysis.get("bootstrap")
    if not isinstance(bootstrap, dict):
        raise ConsumerError("analysis.bootstrap must be an object")
    if bootstrap.get("bit_generator") != "PCG64" or not isinstance(bootstrap.get("seed"), int):
        raise ConsumerError("plan must freeze PCG64 and an integer bootstrap seed")
    if not isinstance(bootstrap.get("draw_count"), int) or bootstrap["draw_count"] < 1000:
        raise ConsumerError("bootstrap draw_count must be at least 1000")
    if not 0 < float(bootstrap.get("confidence_level", 0)) < 1:
        raise ConsumerError("invalid bootstrap confidence_level")
    # Each model gets one prespecified direct-drive identity repeat QC. These
    # records are separately emitted and never enter score contrasts.
    repeat_qc = plan.get("generation_repeat_qc")
    if not isinstance(repeat_qc, list) or (has_dev_rows and len(repeat_qc) != len(models)) or (not has_dev_rows and repeat_qc):
        raise ConsumerError("generation_repeat_qc must contain one QC declaration per model with dev rows, and be empty otherwise")
    qc_models: set[str] = set()
    for qc in repeat_qc:
        if not isinstance(qc, dict) or qc.get("model_id") not in models or qc["model_id"] in qc_models:
            raise ConsumerError("generation_repeat_qc has an unknown or duplicated model")
        qc_models.add(qc["model_id"])
        if not isinstance(qc.get("deterministic"), bool):
            raise ConsumerError(f"generation_repeat_qc deterministic flag must be boolean for {qc['model_id']}")
        if qc.get("split") != "dev" or qc.get("method_id") not in methods or methods[qc["method_id"]].get("kind") != "identity":
            raise ConsumerError(f"identity repeat QC must reference a dev identity generation for {qc['model_id']}")
        reference_row = rows.get(qc.get("reference_row_id"))
        if (
            reference_row is None
            or reference_row.get("split") != "dev"
            or reference_row.get("model_id") != qc["model_id"]
            or methods[reference_row["method_id"]].get("kind") != "identity"
            or reference_row.get("sample_id") != qc.get("input_sample_id")
            or reference_row.get("face_id") != qc.get("face_id")
            or reference_row.get("seed") != qc.get("seed")
        ):
            raise ConsumerError(f"identity repeat QC must bind a planned dev identity row for {qc['model_id']}")
        followup = qc.get("followup_protocol")
        if qc["deterministic"]:
            if followup is not None:
                raise ConsumerError(f"deterministic identity QC should not declare nondeterminism followup for {qc['model_id']}")
        else:
            if not isinstance(followup, dict) or not isinstance(followup.get("dev_seed_ids"), list) or len(set(followup["dev_seed_ids"])) < 3:
                raise ConsumerError(f"nondeterministic model {qc['model_id']} requires a followup protocol with at least three dev seeds")
            _require_sha(followup.get("config_sha256"), f"generation_repeat_qc[{qc['model_id']}].followup_protocol.config_sha256")
    if has_dev_rows and qc_models != set(models):
        raise ConsumerError("generation_repeat_qc does not cover every declared model")
    return models, methods, rows, endpoints, torch_module, score_reduction_runtime, seed_coverage_issues


def _pair_group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("split"),
        row.get("sample_id"),
        row.get("speaker_id"),
        row.get("model_id"),
        row.get("face_id"),
        row.get("seed"),
    )


def _verify_generation_repeat_qc(
    plan: Mapping[str, Any],
    repeat_qc_path: Path,
    *,
    root: Path,
    result_by_id: Mapping[str, Mapping[str, Any]],
    score_arrays_by_id: Mapping[str, Mapping[str, np.ndarray]],
    plan_rows_by_id: Mapping[str, Mapping[str, Any]],
    torch_module: Any,
    file_cache: dict[str, str],
    ffmpeg: str,
    ffprobe: str,
) -> tuple[list[dict[str, Any]], str]:
    """Verify one same-seed identity repeat per model, outside efficacy rows."""

    qc_rows = _load_jsonl(repeat_qc_path)
    expected = plan["generation_repeat_qc"]
    if len(qc_rows) != len(expected):
        raise ConsumerError("generation-repeat QC JSONL row count differs from frozen plan")
    summaries: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    for declaration, record in zip(expected, qc_rows):
        model_id = declaration["model_id"]
        if model_id in seen_models:
            raise ConsumerError(f"duplicate repeat-QC row for {model_id}")
        seen_models.add(model_id)
        for field in ("run_id", "model_id", "input_sample_id", "face_id", "seed", "deterministic", "followup_protocol"):
            expected_value = plan["run_id"] if field == "run_id" else declaration.get(field)
            if record.get(field) != expected_value:
                raise ConsumerError(f"repeat-QC identity mismatch for {model_id}.{field}")
        attempts = record.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != 1:
            raise ConsumerError(f"repeat-QC must record exactly one same-seed repeat attempt for {model_id}")

        reference_id = declaration["reference_row_id"]
        baseline_result = result_by_id[reference_id]
        baseline_plan_row = plan_rows_by_id[reference_id]
        baseline_arrays = score_arrays_by_id.get(reference_id)
        baseline_media = baseline_result.get("render") if baseline_result.get("status") == "PASS" else None
        attempt = attempts[0]
        if not isinstance(attempt, dict):
            raise ConsumerError(f"repeat-QC attempt is malformed for {model_id}")
        attempt_status = attempt.get("status")
        if attempt_status not in ("PASS", "FAILED", "INCOMPLETE"):
            raise ConsumerError(f"invalid repeat-QC attempt status for {model_id}")
        if attempt_status != "PASS" or baseline_arrays is None or not isinstance(baseline_media, dict):
            derived_status = "FAILED" if attempt_status == "FAILED" else "INCOMPLETE"
            if record.get("status") != derived_status:
                raise ConsumerError(f"repeat-QC status mismatch for unavailable baseline/attempt {model_id}")
            summaries.append({
                "model_id": model_id,
                "reference_row_id": reference_id,
                "status": derived_status,
                "failure_reason": attempt.get("failure_reason"),
                "attempt": attempt,
                "excluded_from_efficacy": True,
            })
            continue

        video_path = _check_file_ref(
            root,
            {"path": attempt.get("video_path"), "file_sha256": attempt.get("video_sha256")},
            where=f"repeat_qc.{model_id}.video",
            cache=file_cache,
        )
        packet_sha = _video_hash_muxer_packet_sha256(video_path, ffmpeg)
        elementary_sha = _video_elementary_sha256(video_path, ffmpeg, ffprobe)
        if _require_sha(attempt.get("elementary_video_sha256"), f"repeat_qc.{model_id}.elementary_video_sha256") != packet_sha:
            raise ConsumerError(f"repeat-QC FFmpeg hash-muxer video packet SHA mismatch for {model_id}")
        timestamp_path = _resolve_path(root, attempt.get("timestamp_sidecar_path"), f"repeat_qc.{model_id}.timestamp_sidecar_path")
        timestamp_sha = _require_sha(attempt.get("timestamp_sidecar_sha256"), f"repeat_qc.{model_id}.timestamp_sidecar_sha256")
        if _sha256_file(timestamp_path) != timestamp_sha:
            raise ConsumerError(f"repeat-QC timestamp sidecar SHA mismatch for {model_id}")
        try:
            repeat_timestamp_payload = json.loads(timestamp_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ConsumerError(f"cannot parse repeat-QC timestamp sidecar for {model_id}: {exc}") from exc
        repeat_timestamp_rows = repeat_timestamp_payload.get("frames") if isinstance(repeat_timestamp_payload, dict) else repeat_timestamp_payload
        if not isinstance(repeat_timestamp_rows, list) or not repeat_timestamp_rows:
            raise ConsumerError(f"repeat-QC timestamp sidecar has no frames for {model_id}")
        repeat_model = next(model for model in plan["models"] if model["model_id"] == model_id)
        repeat_expected_clock = _read_timestamp_sidecar(
            timestamp_path, timestamp_sha, len(repeat_timestamp_rows),
            int(repeat_model["fps_num"]), int(repeat_model["fps_den"]),
        )
        repeat_actual_pts = _probe_video_pts_clock(
            video_path, ffprobe, int(repeat_model["fps_num"]), int(repeat_model["fps_den"]),
        )
        if repeat_actual_pts != repeat_expected_clock:
            raise ConsumerError(f"repeat-QC decoded PTS differ from its timestamp sidecar for {model_id}")

        score = {"raw_matrix_npz": attempt.get("raw_matrix_npz")}
        repeat_arrays = _load_score_arrays(root, score, row_id=f"repeat_qc.{model_id}", file_cache=file_cache)
        repeat_matrix_sha = _array_sha256(repeat_arrays["distance_by_query_lag"])
        if _require_sha(attempt.get("raw_distance_matrix_sha256"), f"repeat_qc.{model_id}.raw_distance_matrix_sha256") != repeat_matrix_sha:
            raise ConsumerError(f"repeat-QC distance matrix SHA mismatch for {model_id}")

        baseline_matrix_sha = _array_sha256(baseline_arrays["distance_by_query_lag"])
        baseline_render = baseline_media
        exact_video = packet_sha == baseline_render.get("video_stream_sha256")
        baseline_video_path = _resolve_path(root, baseline_result["render"].get("silent_video_path"), f"repeat_qc.{model_id}.baseline_video")
        baseline_serialized_elementary_sha = _video_elementary_sha256(baseline_video_path, ffmpeg, ffprobe)
        exact_serialized_elementary = elementary_sha == baseline_serialized_elementary_sha
        exact_timestamps = timestamp_sha == baseline_render.get("timestamps_sha256")
        baseline_timestamp_path = _resolve_path(root, baseline_render.get("timestamps_path"), f"repeat_qc.{model_id}.baseline_timestamps")
        baseline_expected_clock = _read_timestamp_sidecar(
            baseline_timestamp_path,
            _require_sha(baseline_render.get("timestamps_sha256"), f"repeat_qc.{model_id}.baseline_timestamps_sha256"),
            int(baseline_render.get("frame_count", 0)),
            int(repeat_model["fps_num"]),
            int(repeat_model["fps_den"]),
        )
        baseline_actual_pts = _probe_video_pts_clock(
            baseline_video_path, ffprobe, int(repeat_model["fps_num"]), int(repeat_model["fps_den"]),
        )
        if baseline_actual_pts != baseline_expected_clock:
            raise ConsumerError(f"baseline decoded PTS differ from its timestamp sidecar for {model_id}")
        exact_actual_pts = repeat_actual_pts == baseline_actual_pts
        exact_distance = repeat_matrix_sha == baseline_matrix_sha
        repeat_index = {key: i for i, (key, valid) in enumerate(zip(_query_keys(repeat_arrays), repeat_arrays["valid_query_mask"].tolist())) if valid}
        baseline_index = {key: i for i, (key, valid) in enumerate(zip(_query_keys(baseline_arrays), baseline_arrays["valid_query_mask"].tolist())) if valid}
        repeat_valid_keys = list(repeat_index)
        baseline_valid_keys = list(baseline_index)
        exact_query_map = (
            repeat_valid_keys == baseline_valid_keys
            and np.array_equal(repeat_arrays["lag_offsets_frames"], baseline_arrays["lag_offsets_frames"])
        )
        if attempt.get("crop_policy_sha256") != baseline_render.get("crop_policy_sha256"):
            raise ConsumerError(f"repeat-QC crop policy differs from identity reference for {model_id}")
        if exact_actual_pts != exact_timestamps:
            raise ConsumerError(f"repeat-QC actual PTS equality disagrees with sidecar equality for {model_id}")

        frame_difference = _max_rgb_frame_difference(video_path, baseline_video_path, ffmpeg, ffprobe)
        distance_a = repeat_arrays["distance_by_query_lag"]
        distance_b = baseline_arrays["distance_by_query_lag"]
        distance_difference = float(np.max(np.abs(distance_a.astype(np.float64) - distance_b.astype(np.float64)))) if distance_a.shape == distance_b.shape else None

        common_keys = sorted(set(repeat_index) & set(baseline_index), key=_query_sort_key)
        delta_cbest = None
        if common_keys and np.array_equal(repeat_arrays["lag_offsets_frames"], baseline_arrays["lag_offsets_frames"]):
            repeat_metric = _compute_curve_metrics(
                distance_a,
                [repeat_index[key] for key in common_keys],
                torch=torch_module,
                lag_offsets=repeat_arrays["lag_offsets_frames"],
                zero_lag_column=int(plan["protocol"]["syncnet"]["zero_lag_column"]),
                anchor_column=int(plan["protocol"]["syncnet"]["anchor_column"]),
            )[1]
            baseline_metric = _compute_curve_metrics(
                distance_b,
                [baseline_index[key] for key in common_keys],
                torch=torch_module,
                lag_offsets=baseline_arrays["lag_offsets_frames"],
                zero_lag_column=int(plan["protocol"]["syncnet"]["zero_lag_column"]),
                anchor_column=int(plan["protocol"]["syncnet"]["anchor_column"]),
            )[1]
            delta_cbest = abs(float(repeat_metric["Cbest"]) - float(baseline_metric["Cbest"]))
        if not isinstance(record.get("comparison"), dict):
            raise ConsumerError(f"repeat-QC comparison object missing for {model_id}")
        comparison = record["comparison"]
        for field, actual in (("exact_video", exact_video), ("exact_timestamps", exact_timestamps), ("exact_distance", exact_distance)):
            if comparison.get(field) is not actual:
                raise ConsumerError(f"repeat-QC {field} differs from artifact hashes for {model_id}")
        for field, rebuilt_value in (
            ("max_frame_absdiff", frame_difference),
            ("max_distance_absdiff", distance_difference),
            ("abs_delta_Cbest", delta_cbest),
        ):
            value = comparison.get(field)
            if rebuilt_value is None:
                if value is not None:
                    raise ConsumerError(f"repeat-QC comparison.{field} must be null when it cannot be rebuilt for {model_id}")
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
                raise ConsumerError(f"repeat-QC comparison.{field} must be finite and nonnegative for {model_id}")
            tolerance = float(plan["protocol"]["syncnet"].get("metric_atol", 2e-6)) + float(plan["protocol"]["syncnet"].get("metric_rtol", 2e-6)) * abs(float(rebuilt_value))
            if abs(float(value) - float(rebuilt_value)) > tolerance:
                raise ConsumerError(f"repeat-QC comparison.{field} differs from artifact rebuild for {model_id}")
        if exact_video and frame_difference not in (0, None):
            raise ConsumerError(f"repeat-QC exact video hash has nonzero reported frame difference for {model_id}")
        if exact_distance and distance_difference not in (0, None):
            raise ConsumerError(f"repeat-QC exact distance hash has nonzero reported matrix difference for {model_id}")

        deterministic = bool(declaration["deterministic"])
        if deterministic:
            derived_status = "PASS" if exact_video and exact_timestamps and exact_distance and exact_query_map else "FAILED"
        else:
            derived_status = "PASS" if exact_video and exact_timestamps and exact_distance and exact_query_map else "NONDETERMINISTIC"
        if record.get("status") != derived_status:
            raise ConsumerError(f"repeat-QC status differs from measured exactness for {model_id}")
        summaries.append({
            "model_id": model_id,
            "reference_row_id": reference_id,
            "status": derived_status,
            "deterministic_declared": deterministic,
            "exact_video": exact_video,
            "exact_serialized_elementary_video": exact_serialized_elementary,
            "exact_timestamps": exact_timestamps,
            "exact_actual_pts": exact_actual_pts,
            "exact_distance": exact_distance,
            "exact_query_map": exact_query_map,
            "attempt_video_path": attempt.get("video_path"),
            "attempt_video_sha256": attempt.get("video_sha256"),
            "elementary_video_sha256": elementary_sha,
            "timestamp_sidecar_path": str(timestamp_path),
            "timestamp_sidecar_sha256": timestamp_sha,
            "crop_policy_sha256": attempt.get("crop_policy_sha256"),
            "raw_matrix_npz": attempt.get("raw_matrix_npz"),
            "raw_distance_matrix_sha256": repeat_matrix_sha,
            "comparison": comparison,
            "followup_protocol": declaration.get("followup_protocol"),
            "excluded_from_efficacy": True,
        })
    return summaries, _sha256_file(repeat_qc_path)


def _load_and_validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    plan_path = Path(args.plan).resolve()
    results_path = Path(args.results).resolve()
    manifest_path = Path(args.result_manifest).resolve()
    repeat_qc_path = Path(args.generation_repeat_qc).resolve()
    root = Path(args.artifact_root).resolve()
    plan = _load_json(plan_path)
    plan_raw_sha = _sha256_file(plan_path)
    results_raw_sha = _sha256_file(results_path)
    repeat_qc_raw_sha = _sha256_file(repeat_qc_path)
    manifest = _load_json(manifest_path)
    manifest_raw_sha = _sha256_file(manifest_path)
    models, methods, plan_rows_by_id, endpoints, torch_module, score_reduction_runtime, seed_coverage_issues = _validate_plan(plan)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("run_id") != plan.get("run_id"):
        raise ConsumerError("result manifest schema/run_id differs from the frozen plan")
    if manifest.get("manifest_sha256") != _payload_sha256(manifest, "manifest_sha256"):
        raise ConsumerError("embedded result-manifest SHA does not match canonical payload")
    if manifest.get("plan_sha256") != plan.get("plan_sha256") or manifest.get("plan_raw_sha256") != plan_raw_sha:
        raise ConsumerError("result manifest does not bind the exact plan file and payload")
    if manifest.get("results_jsonl_sha256") != results_raw_sha:
        raise ConsumerError("result manifest does not bind the exact results JSONL")
    if manifest.get("generation_repeat_qc_jsonl_sha256") != repeat_qc_raw_sha:
        raise ConsumerError("result manifest does not bind the generation-repeat QC JSONL")
    if manifest.get("generation_repeat_qc_rows") != len(plan.get("generation_repeat_qc", [])):
        raise ConsumerError("result manifest repeat-QC row count differs from plan")

    planned_rows = plan.get("rows", [])
    result_rows = _load_jsonl(results_path)
    if manifest.get("planned_rows") != len(planned_rows) or manifest.get("result_rows") != len(result_rows):
        raise ConsumerError("result manifest row totals do not match actual inputs")
    planned_ids = [row["row_id"] for row in planned_rows]
    result_ids = [row.get("row_id") for row in result_rows]
    if len(result_ids) != len(set(result_ids)):
        raise ConsumerError("results JSONL contains duplicate row_id values")
    if result_ids != planned_ids:
        missing = sorted(set(planned_ids) - set(result_ids))
        extra = sorted(set(result_ids) - set(planned_ids))
        raise ConsumerError(f"results rows are not in exact plan order; missing={missing[:8]}, extra={extra[:8]}")

    status_counts = Counter()
    status_counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    counts_by_model_method: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    result_by_id: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, Any]] = list(seed_coverage_issues)
    file_cache: dict[str, str] = {}
    audio_cache: dict[str, tuple[np.ndarray, str, int]] = {}
    media_by_id: dict[str, dict[str, Any]] = {}
    score_arrays_by_id: dict[str, dict[str, np.ndarray]] = {}

    model_map = models
    for planned, result in zip(planned_rows, result_rows):
        row_id = planned["row_id"]
        result_by_id[row_id] = result
        status = result.get("status")
        if status not in ("PASS", "FAILED", "INCOMPLETE"):
            raise ConsumerError(f"result {row_id} has invalid status {status!r}")
        status_counts[status] += 1
        status_counts_by_split[planned["split"]][status] += 1
        counts_by_model_method[(planned["model_id"], planned["method_id"])][status] += 1
        for field in IDENTITY_FIELDS:
            if result.get(field) != planned.get(field):
                raise ConsumerError(f"result identity mismatch for {row_id}.{field}")
        expected_refs = {
            "natural_pcm_sha256": planned["natural_pcm"]["pcm_sha256"],
            "score_audio_sha256": planned["score_audio"]["pcm_sha256"],
            "source_face_sha256": planned["source_face"]["file_sha256"],
            "source_video_sha256": planned["source_video"]["file_sha256"],
        }
        for field, expected in expected_refs.items():
            if result.get(field) != expected:
                raise ConsumerError(f"result artifact identity mismatch for {row_id}.{field}")
        if result.get("score_audio_sha256") != planned["natural_pcm"]["pcm_sha256"]:
            raise ConsumerError(f"score audio is not the natural PCM for {row_id}")

        # Immutable source/generator inputs are checked for every planned row,
        # including rows whose downstream render or score later failed.
        for name in ("source_face", "source_video"):
            _check_file_ref(root, planned[name], where=f"{row_id}.{name}", cache=file_cache)
        natural_path, natural_samples, natural_hash, natural_count = _canonical_pcm_ref(
            root, planned["natural_pcm"], where=f"{row_id}.natural_pcm", cache=file_cache
        )
        if planned["score_audio"].get("path") != planned["natural_pcm"].get("path"):
            raise ConsumerError(f"score_audio path is not the natural PCM path for {row_id}")
        if planned["score_audio"].get("file_sha256") != planned["natural_pcm"].get("file_sha256"):
            raise ConsumerError(f"score_audio file bytes differ from natural PCM for {row_id}")
        if natural_hash != planned["score_audio"].get("pcm_sha256"):
            raise ConsumerError(f"score_audio float32 PCM hash differs from natural PCM for {row_id}")
        audio_cache[str(natural_path.resolve())] = (natural_samples, natural_hash, natural_count)

        generator_ref = result.get("generator_pcm")
        if status == "PASS" and not isinstance(generator_ref, dict):
            raise ConsumerError(f"PASS result lacks actual generator PCM for {row_id}")
        if isinstance(generator_ref, dict):
            generator_path, generator_samples, generator_hash, generator_count = _canonical_pcm_ref(
                root, generator_ref, where=f"{row_id}.generator_pcm", cache=file_cache
            )
            if generator_count != natural_count or generator_samples.shape != natural_samples.shape:
                raise ConsumerError(f"generator waveform changed canonical sample count for {row_id}")
            if result.get("generator_pcm_sha256") != generator_hash:
                raise ConsumerError(f"result generator PCM hash differs from its audio reference for {row_id}")
            if methods[planned["method_id"]].get("kind") == "identity" and (generator_hash != natural_hash or not np.array_equal(generator_samples, natural_samples)):
                raise ConsumerError(f"identity method is not bit-exact for {row_id}")
            if methods[planned["method_id"]].get("kind") == "rms_fixed":
                target_rms = float(methods[planned["method_id"]]["target_rms"])
                tolerance = float(methods[planned["method_id"]]["target_rms_tolerance"])
                actual_rms = float(np.sqrt(np.mean(generator_samples.astype(np.float64) ** 2, dtype=np.float64)))
                if abs(actual_rms - target_rms) > tolerance:
                    raise ConsumerError(f"fixed-RMS method output RMS differs from frozen target for {row_id}: {actual_rms} vs {target_rms}")
            audio_cache[str(generator_path.resolve())] = (generator_samples, generator_hash, generator_count)
        elif result.get("generator_pcm_sha256") not in (None, ""):
            raise ConsumerError(f"generator PCM SHA is present without an artifact for {row_id}")

        if status != "PASS":
            if not isinstance(result.get("failure_reason"), str) or not result.get("failure_reason"):
                raise ConsumerError(f"non-PASS result lacks failure_reason for {row_id}")
            continue
        score = result.get("score")
        if not isinstance(score, dict) or score.get("status") != "PASS":
            raise ConsumerError(f"PASS result lacks a PASS score record for {row_id}")
        model = model_map[planned["model_id"]]
        if score.get("score_model_sha256") != model["score_model_sha256"]:
            raise ConsumerError(f"score model hash differs from frozen model registry for {row_id}")
        if score.get("preprocessing_sha256") != model["preprocessing_sha256"]:
            raise ConsumerError(f"score preprocessing hash differs from frozen model registry for {row_id}")
        media_by_id[row_id] = _verify_mux_and_clock(
            planned,
            result,
            plan=plan,
            root=root,
            natural_path=natural_path,
            natural_pcm_sha256=natural_hash,
            natural_sample_count=natural_count,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            file_cache=file_cache,
        )
        score_arrays_by_id[row_id] = _load_score_arrays(root, score, row_id=row_id, file_cache=file_cache)
        _validate_query_clock_map(score_arrays_by_id[row_id], media_by_id[row_id], row_id=row_id)

    manifest_status_counts = manifest.get("status_counts")
    actual_status_counts = {key: int(status_counts.get(key, 0)) for key in ("PASS", "FAILED", "INCOMPLETE")}
    if manifest_status_counts != actual_status_counts:
        raise ConsumerError("result-manifest status_counts disagree with JSONL")
    expected_counts: list[dict[str, Any]] = []
    for model_id in models:
        for method_id in methods:
            counts = counts_by_model_method.get((model_id, method_id), Counter())
            expected_counts.append({
                "model_id": model_id,
                "method_id": method_id,
                "planned_rows": sum(1 for row in planned_rows if row["model_id"] == model_id and row["method_id"] == method_id),
                "result_rows": sum(1 for row in result_rows if row.get("model_id") == model_id and row.get("method_id") == method_id),
                "status_counts": {key: int(counts.get(key, 0)) for key in ("PASS", "FAILED", "INCOMPLETE")},
            })
    if manifest.get("counts_by_model_method") != expected_counts:
        raise ConsumerError("result-manifest per-model/method counts disagree with planned/result JSONL rows")

    repeat_qc_summaries, verified_repeat_qc_raw_sha = _verify_generation_repeat_qc(
        plan,
        repeat_qc_path,
        root=root,
        result_by_id=result_by_id,
        score_arrays_by_id=score_arrays_by_id,
        plan_rows_by_id=plan_rows_by_id,
        torch_module=torch_module,
        file_cache=file_cache,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
    )

    return {
        "plan": plan,
        "torch_module": torch_module,
        "score_reduction_runtime": score_reduction_runtime,
        "plan_path": str(plan_path),
        "plan_raw_sha256": plan_raw_sha,
        "results_path": str(results_path),
        "results_raw_sha256": results_raw_sha,
        "result_manifest_path": str(manifest_path),
        "result_manifest_raw_sha256": manifest_raw_sha,
        "result_manifest_payload_sha256": manifest["manifest_sha256"],
        "generation_repeat_qc_path": str(repeat_qc_path),
        "generation_repeat_qc_raw_sha256": verified_repeat_qc_raw_sha,
        "generation_repeat_qc": repeat_qc_summaries,
        "models": models,
        "methods": methods,
        "plan_rows_by_id": plan_rows_by_id,
        "result_by_id": result_by_id,
        "media_by_id": media_by_id,
        "score_arrays_by_id": score_arrays_by_id,
        "status_counts": actual_status_counts,
        "status_counts_by_split": {
            split: {key: int(counts.get(key, 0)) for key in ("PASS", "FAILED", "INCOMPLETE")}
            for split, counts in status_counts_by_split.items()
        },
        "counts_by_model_method": expected_counts,
        "issues": issues,
        "file_cache": file_cache,
    }


def _rebuild_all_supports(state: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    plan = state["plan"]
    protocol = plan["protocol"]["syncnet"]
    lag_offsets = np.asarray(protocol["lag_offsets_frames"], dtype=np.int32)
    zero_column = int(protocol["zero_lag_column"])
    anchor_column = int(protocol["anchor_column"])
    atol = float(protocol.get("metric_atol", 2e-6))
    rtol = float(protocol.get("metric_rtol", 2e-6))
    methods = list(state["methods"])
    groups: dict[tuple[Any, ...], dict[str, str]] = defaultdict(dict)
    for row_id, row in state["plan_rows_by_id"].items():
        groups[_pair_group_key(row)][row["method_id"]] = row_id
    rebuilt: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    group_audit: list[dict[str, Any]] = []

    for group_key, row_ids_by_method in groups.items():
        if set(row_ids_by_method) != set(methods):
            group_audit.append({
                "pair_key": list(group_key),
                "status": "INCOMPLETE_PLANNED_METHOD_GROUP",
                "planned_method_ids": methods,
                "present_method_ids": sorted(row_ids_by_method),
            })
            continue
        if any(row_id not in state["score_arrays_by_id"] for row_id in row_ids_by_method.values()):
            group_audit.append({"pair_key": list(group_key), "status": "INCOMPLETE_METHOD_ROW"})
            continue
        media_rows = [state["media_by_id"][row_id] for row_id in row_ids_by_method.values()]
        clock_fields = ("fps_num", "fps_den", "frame_count", "crop_policy_sha256")
        for field in clock_fields:
            if len({media.get(field) for media in media_rows}) != 1:
                raise ConsumerError(f"paired methods have different {field} in context {group_key}")
        reference_clock = media_rows[0].get("query_clock")
        if any(media.get("query_clock") != reference_clock for media in media_rows[1:]):
            raise ConsumerError(f"paired methods have different rational frame clocks in context {group_key}")
        if plan["protocol"]["roi"]["mode"] == "shared_fixed":
            if len({media.get("crop_roi_geometry_sha256") for media in media_rows}) != 1:
                raise ConsumerError(f"paired methods have different frozen shared ROI geometry in context {group_key}")
        keys_by_method: dict[str, list[tuple[int, int, int, int]]] = {}
        index_by_method: dict[str, dict[tuple[int, int, int, int], int]] = {}
        for method_id, row_id in row_ids_by_method.items():
            arrays = state["score_arrays_by_id"][row_id]
            if not np.array_equal(arrays["lag_offsets_frames"], lag_offsets):
                raise ConsumerError(f"result lag axis differs from frozen plan for {row_id}")
            keys = _query_keys(arrays)
            valid = arrays["valid_query_mask"].tolist()
            valid_keys = [key for key, flag in zip(keys, valid) if flag]
            keys_by_method[method_id] = valid_keys
            index_by_method[method_id] = {key: index for index, (key, flag) in enumerate(zip(keys, valid)) if flag}
        common = set(keys_by_method[methods[0]])
        for method_id in methods[1:]:
            common.intersection_update(keys_by_method[method_id])
        ordered_common = sorted(common, key=_query_sort_key)
        group_audit.append({"pair_key": list(group_key), "status": "PASS", "common_full_queries": len(ordered_common)})
        for support_id, support in state["supports"].items():
            trim = int(support["trim_each_end"])
            start = min(trim, len(ordered_common))
            stop = max(start, len(ordered_common) - trim)
            selected = ordered_common[start:stop]
            mask_hash = _support_mask_sha(support_id, selected)
            eligible = len(selected) >= int(support["minimum_queries"])
            for method_id, row_id in row_ids_by_method.items():
                arrays = state["score_arrays_by_id"][row_id]
                indices = [index_by_method[method_id][key] for key in selected]
                if selected:
                    curve, metrics = _compute_curve_metrics(
                        arrays["distance_by_query_lag"],
                        indices,
                        torch=state["torch_module"],
                        lag_offsets=lag_offsets,
                        zero_lag_column=zero_column,
                        anchor_column=anchor_column,
                    )
                    _compare_saved_score_summary(
                        state["result_by_id"][row_id],
                        support_id,
                        curve,
                        metrics,
                        mask_hash,
                        len(selected),
                        atol=atol,
                        rtol=rtol,
                    )
                    rebuilt[row_id][support_id] = {
                        "curve": curve,
                        "metrics": metrics,
                        "support_mask_sha256": mask_hash,
                        "query_count": len(selected),
                        "eligible": eligible,
                    }
                else:
                    # Empty support is a planned ineligible outcome. The
                    # producer must serialize it as empty, not as a recovered
                    # shorter support or a fabricated score.
                    score = state["result_by_id"][row_id]["score"]
                    if score.get("lag_curve_by_support", {}).get(support_id) not in ([], None):
                        raise ConsumerError(f"empty common support has a non-empty saved curve for {row_id}/{support_id}")
                    saved_support = score.get("support_by_support", {}).get(support_id, {})
                    if saved_support.get("query_count") != 0 or saved_support.get("support_mask_sha256") != mask_hash:
                        raise ConsumerError(f"empty common support metadata differs for {row_id}/{support_id}")
                    saved_empty_metrics = score.get("metrics_by_support", {}).get(support_id)
                    if isinstance(saved_empty_metrics, dict):
                        populated = [
                            key for key, value in saved_empty_metrics.items()
                            if key not in ("status", "reason") and value is not None
                        ]
                        if populated:
                            raise ConsumerError(
                                f"empty common support has populated metrics for {row_id}/{support_id}: {populated}"
                            )
                    rebuilt[row_id][support_id] = {
                        "curve": [],
                        "metrics": None,
                        "support_mask_sha256": mask_hash,
                        "query_count": 0,
                        "eligible": False,
                    }

    state["pair_group_audit"] = group_audit
    state["supports"] = state["supports"]
    return rebuilt


def _endpoint_speaker_values(
    endpoint: Mapping[str, Any],
    state: dict[str, Any],
    rebuilt: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    plan = state["plan"]
    analysis = plan["analysis"]
    contrast_by_id = {item["contrast_id"]: item for item in analysis["contrasts"]}
    contrast = contrast_by_id[endpoint["contrast_id"]]
    method_a = contrast["method_a"]
    method_b = contrast["method_b"]
    support_id = endpoint["support_id"]
    metric = endpoint["metric"]
    split_id = endpoint.get("split", "confirm")
    speaker_roster = list(plan[f"{split_id}_speaker_ids"])
    scope = endpoint["model_scope"]
    model_ids = [endpoint["model_id"]] if scope == "model" else list(endpoint["model_ids"])

    row_index: dict[tuple[str, str, str, str, str], str] = {}
    planned_units: dict[tuple[str, str], set[tuple[Any, ...]]] = defaultdict(set)
    context_rows: dict[tuple[Any, ...], dict[str, str]] = defaultdict(dict)
    base_contexts: set[tuple[Any, ...]] = set()
    for row_id, row in state["plan_rows_by_id"].items():
        if row.get("split") != split_id or row.get("model_id") not in model_ids:
            continue
        if row.get("method_id") not in (method_a, method_b):
            continue
        ctx = _pair_group_key(row)
        base_contexts.add((split_id, row["sample_id"], row["speaker_id"], row["model_id"], row["face_id"]))
        context_rows[ctx][row["method_id"]] = row_id
    expected_seeds = state["plan"].get("generation_seed_family", {}).get(split_id, [])
    for base in base_contexts:
        _, sample_id, speaker_id, model_id, face_id = base
        for seed in expected_seeds:
            ctx = (*base, seed)
            context_rows.setdefault(ctx, {})
            planned_units[(model_id, speaker_id)].add((sample_id, face_id, seed))

    context_delta: dict[tuple[Any, ...], float | None] = {}
    context_status: dict[tuple[Any, ...], str] = {}
    for ctx, by_method in context_rows.items():
        if set(by_method) != {method_a, method_b}:
            context_delta[ctx] = None
            context_status[ctx] = "MISSING_METHOD_OR_SEED_PLAN_ROW"
            continue
        row_a = by_method[method_a]
        row_b = by_method[method_b]
        support_a = rebuilt.get(row_a, {}).get(support_id)
        support_b = rebuilt.get(row_b, {}).get(support_id)
        if not support_a or not support_b or not support_a.get("eligible") or not support_b.get("eligible"):
            context_delta[ctx] = None
            context_status[ctx] = "INELIGIBLE_SUPPORT_OR_FAILED_ROW"
            continue
        if support_a["support_mask_sha256"] != support_b["support_mask_sha256"] or support_a["query_count"] != support_b["query_count"]:
            raise ConsumerError(f"paired support mismatch in endpoint {endpoint['endpoint_id']}: {ctx}")
        value_a = support_a["metrics"][metric]
        value_b = support_b["metrics"][metric]
        context_delta[ctx] = float(value_a) - float(value_b)
        context_status[ctx] = "PASS"

    model_speaker_values: dict[str, dict[str, float | None]] = {}
    model_speaker_complete: dict[str, dict[str, bool]] = {}
    model_speaker_counts: dict[str, dict[str, dict[str, int]]] = {}
    for model_id in model_ids:
        values_for_model: dict[str, float | None] = {}
        complete_for_model: dict[str, bool] = {}
        count_for_model: dict[str, dict[str, int]] = {}
        for speaker_id in speaker_roster:
            sample_values: list[float] = []
            planned_samples = sorted({unit[0] for unit in planned_units.get((model_id, speaker_id), set())})
            valid_samples = 0
            planned_context_count = 0
            valid_context_count = 0
            for sample_id in planned_samples:
                units = sorted(
                    [unit for unit in planned_units[(model_id, speaker_id)] if unit[0] == sample_id],
                    key=lambda unit: (str(unit[1]), str(unit[2])),
                )
                per_unit = []
                for face_id, seed in [(u[1], u[2]) for u in units]:
                    ctx = (split_id, sample_id, speaker_id, model_id, face_id, seed)
                    planned_context_count += 1
                    value = context_delta.get(ctx)
                    if value is None:
                        continue
                    valid_context_count += 1
                    per_unit.append(value)
                if per_unit and len(per_unit) == len(units):
                    sample_values.append(float(np.mean(np.asarray(per_unit, dtype=np.float64))))
                    valid_samples += 1
            speaker_value = float(np.mean(np.asarray(sample_values, dtype=np.float64))) if sample_values else None
            speaker_complete = bool(planned_samples) and valid_samples == len(planned_samples) and valid_context_count == planned_context_count
            values_for_model[speaker_id] = speaker_value
            complete_for_model[speaker_id] = speaker_complete
            count_for_model[speaker_id] = {
                "planned_samples": len(planned_samples),
                "estimable_samples": valid_samples,
                "planned_contexts": planned_context_count,
                "estimable_contexts": valid_context_count,
            }
        model_speaker_values[model_id] = values_for_model
        model_speaker_complete[model_id] = complete_for_model
        model_speaker_counts[model_id] = count_for_model

    speaker_values: list[float | None] = []
    speaker_complete_list: list[bool] = []
    speaker_counts: dict[str, dict[str, Any]] = {}
    for speaker_id in speaker_roster:
        values = [model_speaker_values[model_id][speaker_id] for model_id in model_ids]
        model_complete = [model_speaker_complete[model_id][speaker_id] for model_id in model_ids]
        if all(v is not None for v in values):
            speaker_values.append(float(np.mean(np.asarray(values, dtype=np.float64))))
        else:
            speaker_values.append(None)
        speaker_complete_list.append(all(model_complete))
        speaker_counts[speaker_id] = {
            "by_model": {m: model_speaker_counts[m][speaker_id] for m in model_ids},
            "estimable_models": sum(v is not None for v in values),
        }
    complete = bool(speaker_roster) and all(speaker_complete_list)
    estimable_values = [v for v in speaker_values if v is not None]
    descriptive_mean = float(np.mean(np.asarray(estimable_values, dtype=np.float64))) if estimable_values else None
    return {
        "endpoint_id": endpoint["endpoint_id"],
        "split": split_id,
        "model_scope": scope,
        "model_ids": model_ids,
        "contrast_id": endpoint["contrast_id"],
        "method_a": method_a,
        "method_b": method_b,
        "support_id": support_id,
        "metric": metric,
        "speaker_ids": speaker_roster,
        "speaker_values": speaker_values,
        "speaker_complete": speaker_complete_list,
        "speaker_counts": speaker_counts,
        "planned_speakers": len(speaker_roster),
        "estimable_speakers": len(estimable_values),
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "mean": descriptive_mean,
        "interval_scope": "confirmatory_equal_speaker" if complete else "descriptive_available_complete_pairs_only",
    }


def _bootstrap_summary(
    endpoint_rows: dict[str, dict[str, Any]],
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    analysis = plan["analysis"]
    bootstrap = analysis["bootstrap"]
    speakers = list(plan["confirm_speaker_ids"])
    n_speakers = len(speakers)
    draws = int(bootstrap["draw_count"])
    rng = np.random.Generator(np.random.PCG64(int(bootstrap["seed"])))
    index_matrix = (
        rng.integers(0, n_speakers, size=(draws, n_speakers), dtype=np.int64)
        if n_speakers
        else np.empty((draws, 0), dtype=np.int64)
    )
    confidence = float(bootstrap["confidence_level"])
    alpha = 1.0 - confidence
    boot_values: dict[str, np.ndarray] = {}
    for endpoint_id, row in endpoint_rows.items():
        values = row["speaker_values"]
        if row.get("split", "confirm") != "confirm" or row["status"] != "COMPLETE" or any(value is None for value in values):
            row["pointwise_interval"] = None
            row["bootstrap_se"] = None
            if row.get("split", "confirm") != "confirm":
                row["interval_scope"] = "exploratory_dev_descriptive_only"
            continue
        array = np.asarray(values, dtype=np.float64)
        estimates = array[index_matrix].mean(axis=1)
        row["pointwise_interval"] = [float(np.quantile(estimates, alpha / 2)), float(np.quantile(estimates, 1 - alpha / 2))]
        row["bootstrap_se"] = float(np.std(estimates, ddof=1))
        boot_values[endpoint_id] = estimates

    family_ids = list(analysis["simultaneous_family_ids"])
    family_complete = all(endpoint_rows[eid]["status"] == "COMPLETE" for eid in family_ids)
    family_result: dict[str, Any] = {
        "family_ids": family_ids,
        "status": "COMPLETE" if family_complete else "INCOMPLETE",
        "method": "max_abs_studentized_speaker_bootstrap",
        "confidence_level": confidence,
        "draw_count": draws,
        "critical_value": None,
    }
    if not family_ids:
        family_result["status"] = "NOT_APPLICABLE"
        family_result["reason"] = "no confirmatory endpoints are planned in this development-only run"
        return family_result, index_matrix
    if family_complete:
        observed = np.asarray([float(endpoint_rows[eid]["mean"]) for eid in family_ids], dtype=np.float64)
        matrix = np.column_stack([boot_values[eid] for eid in family_ids])
        standard_errors = np.std(matrix, axis=0, ddof=1)
        undefined_zero_se = [
            family_ids[j]
            for j, se in enumerate(standard_errors)
            if se == 0 and np.any(matrix[:, j] != observed[j])
        ]
        if undefined_zero_se:
            family_result["status"] = "INCOMPLETE_ZERO_SE"
            family_result["reason"] = "studentized bootstrap is undefined for a zero-SE endpoint whose resamples differ from its observed estimate"
            family_result["undefined_zero_se_endpoint_ids"] = undefined_zero_se
            for endpoint_id in family_ids:
                endpoint_rows[endpoint_id]["simultaneous_interval"] = None
                endpoint_rows[endpoint_id]["simultaneous_family_status"] = "INCOMPLETE_ZERO_SE"
            return family_result, index_matrix
        max_t = np.zeros(draws, dtype=np.float64)
        for j, se in enumerate(standard_errors):
            if se == 0:
                t_values = np.zeros(draws, dtype=np.float64)
            else:
                t_values = np.abs((matrix[:, j] - observed[j]) / se)
            max_t = np.maximum(max_t, t_values)
        critical = float(np.quantile(max_t, confidence))
        family_result["critical_value"] = critical
        family_result["critical_value_definition"] = "linear empirical quantile of per-draw maximum absolute studentized deviation"
        for j, endpoint_id in enumerate(family_ids):
            half_width = critical * float(standard_errors[j])
            endpoint_rows[endpoint_id]["simultaneous_interval"] = [float(observed[j] - half_width), float(observed[j] + half_width)]
            endpoint_rows[endpoint_id]["simultaneous_family_status"] = "COMPLETE"
    else:
        for endpoint_id in family_ids:
            endpoint_rows[endpoint_id]["simultaneous_interval"] = None
            endpoint_rows[endpoint_id]["simultaneous_family_status"] = "INCOMPLETE_FAMILY_NOT_SHORTENED"
    return family_result, index_matrix


def _write_summary(
    out: Path,
    state: Mapping[str, Any],
    endpoints: Mapping[str, dict[str, Any]],
    family: Mapping[str, Any],
) -> None:
    lines = [
        f"# Audio enhancement evaluation: {state['plan']['run_id']}",
        "",
        f"Consumer status: **{state['audit_status']}**",
        f"Plan payload SHA-256: `{state['plan']['plan_sha256']}`",
        f"Results JSONL SHA-256: `{state['results_raw_sha256']}`",
        "",
        "## Planned and completed rows",
        "",
        "| Model | Method | Planned | PASS | FAILED | INCOMPLETE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in state["counts_by_model_method"]:
        c = row["status_counts"]
        lines.append(f"| {row['model_id']} | {row['method_id']} | {row['planned_rows']} | {c['PASS']} | {c['FAILED']} | {c['INCOMPLETE']} |")
    lines += ["", "## Frozen contrast endpoints", "", "| Endpoint | Split | Model scope | Contrast | Support | Metric | Mean | Pointwise interval | Simultaneous interval | Status |", "|---|---|---|---|---|---:|---|---|---|---|"]
    for eid, row in endpoints.items():
        mean = "" if row.get("mean") is None else f"{row['mean']:.6f}"
        point = row.get("pointwise_interval")
        simultaneous = row.get("simultaneous_interval")
        point_text = "" if point is None else f"[{point[0]:.6f}, {point[1]:.6f}]"
        sim_text = "" if simultaneous is None else f"[{simultaneous[0]:.6f}, {simultaneous[1]:.6f}]"
        lines.append(f"| {eid} | {row.get('split', 'confirm')} | {row['model_scope']}:{','.join(row['model_ids'])} | {row['method_a']} − {row['method_b']} | {row['support_id']} | {row['metric']} | {mean} | {point_text} | {sim_text} | {row['status']} |")
    lines += ["", "## Same-seed identity-repeat QC", "", "| Model | Status | Deterministic declared | Exact video | Exact timestamps | Exact query map | Exact distance matrix | Max frame difference | Max distance difference | Absolute Cbest delta |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in state.get("generation_repeat_qc", []):
        comparison = row.get("comparison", {})
        lines.append(
            f"| {row['model_id']} | {row['status']} | {row.get('deterministic_declared', '')} | "
            f"{row.get('exact_video', '')} | {row.get('exact_timestamps', '')} | {row.get('exact_query_map', '')} | {row.get('exact_distance', '')} | "
            f"{comparison.get('max_frame_absdiff', '')} | {comparison.get('max_distance_absdiff', '')} | {comparison.get('abs_delta_Cbest', '')} |"
        )
    lines += [
        "",
        f"Simultaneous family: **{family['status']}**; method `{family['method']}`; "
        f"planned endpoints={len(family['family_ids'])}; critical value={family['critical_value']}",
        "",
        "## Integrity notes",
        "",
        ("- SyncNet curve reduction: PyTorch "
         f"`{state['score_reduction_runtime']['torch_version']}` on CPU with "
         f"{state['score_reduction_runtime']['num_threads']} thread(s), using the frozen stack/mean/median/min sequence."),
        "- The enhancement waveform is used only as the generator condition. The muxed SyncNet audio was independently decoded as PCM_F32LE and checked against the original natural PCM; scorer PCM16 is recorded separately as a derived input.",
        "- Per-query SyncNet distances were reduced on the common FULL and frozen GUARD20 supports. No alignment or lag retiming was applied.",
        "- `best_lag_frames` is the selected lag-axis value; official AV `offset_frames` is its negative. Neither value was used to retime the video or audio.",
        "- Every planned row remains in the denominator. Endpoints with incomplete rows/supports are marked incomplete; the simultaneous family is never shortened.",
        "",
    ]
    if state["issues"]:
        lines += ["## Integrity issues", ""]
        for issue in state["issues"]:
            lines.append(f"- `{issue.get('row_id', 'run')}`: {issue.get('message', issue)}")
        lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_outputs(out: Path, state: dict[str, Any], rebuilt: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> None:
    plan = state["plan"]
    endpoint_rows: dict[str, dict[str, Any]] = {}
    for endpoint in plan["analysis"]["summary_endpoints"]:
        endpoint_rows[endpoint["endpoint_id"]] = _endpoint_speaker_values(endpoint, state, rebuilt)
    family, index_matrix = _bootstrap_summary(endpoint_rows, plan)

    qc_hard_failure = any(row["status"] in ("FAILED", "INCOMPLETE") for row in state.get("generation_repeat_qc", []))
    family_ready = family["status"] in ("COMPLETE", "NOT_APPLICABLE")
    if not endpoint_rows:
        output_status = "NO_EVALUATION_ENDPOINTS"
    else:
        output_status = "PASS" if not qc_hard_failure and not state["issues"] and all(row["status"] == "COMPLETE" for row in endpoint_rows.values()) and family_ready else "INCOMPLETE"
    state["audit_status"] = output_status

    with (out / "speaker_contrasts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("endpoint_id", "speaker_id", "speaker_delta", "speaker_complete", "planned_samples", "estimable_samples", "planned_contexts", "estimable_contexts"),
        )
        writer.writeheader()
        for endpoint_id, endpoint in endpoint_rows.items():
            for speaker_id, value, complete in zip(endpoint["speaker_ids"], endpoint["speaker_values"], endpoint["speaker_complete"]):
                counts = endpoint["speaker_counts"].get(speaker_id, {})
                by_model = counts.get("by_model", {})
                flattened = {
                    "planned_samples": sum(v["planned_samples"] for v in by_model.values()),
                    "estimable_samples": sum(v["estimable_samples"] for v in by_model.values()),
                    "planned_contexts": sum(v["planned_contexts"] for v in by_model.values()),
                    "estimable_contexts": sum(v["estimable_contexts"] for v in by_model.values()),
                }
                writer.writerow({
                    "endpoint_id": endpoint_id,
                    "speaker_id": speaker_id,
                    "speaker_delta": "" if value is None else f"{value:.17g}",
                    "speaker_complete": int(bool(complete)),
                    **flattened,
                })

    endpoint_speaker_union = list(dict.fromkeys(
        speaker_id
        for speaker_list in (plan["dev_speaker_ids"], plan["confirm_speaker_ids"])
        for speaker_id in speaker_list
    ))
    endpoint_speaker_index = {speaker_id: i for i, speaker_id in enumerate(endpoint_speaker_union)}
    endpoint_speaker_values = np.full((len(endpoint_rows), len(endpoint_speaker_union)), np.nan, dtype=np.float64)
    for endpoint_index, endpoint in enumerate(endpoint_rows.values()):
        for speaker_id, value in zip(endpoint["speaker_ids"], endpoint["speaker_values"]):
            endpoint_speaker_values[endpoint_index, endpoint_speaker_index[speaker_id]] = np.nan if value is None else float(value)

    np.savez_compressed(
        out / "speaker_bootstrap.npz",
        speaker_ids=np.asarray(plan["confirm_speaker_ids"], dtype=f"U{max(1, max(map(len, plan['confirm_speaker_ids']), default=0))}"),
        bootstrap_indices_int64=index_matrix,
        endpoint_ids=np.asarray(list(endpoint_rows), dtype=f"U{max(1, max(map(len, endpoint_rows), default=0))}"),
        endpoint_speaker_union_ids=np.asarray(endpoint_speaker_union, dtype=f"U{max(1, max(map(len, endpoint_speaker_union), default=0))}"),
        endpoint_speaker_values_float64=endpoint_speaker_values,
    )
    npz_sha = _sha256_file(out / "speaker_bootstrap.npz")

    audit = {
        "schema_version": SCHEMA_VERSION,
        "consumer_version": CONSUMER_VERSION,
        "run_id": plan["run_id"],
        "status": output_status,
        "plan_path": state["plan_path"],
        "plan_raw_sha256": state["plan_raw_sha256"],
        "plan_payload_sha256": plan["plan_sha256"],
        "results_path": state["results_path"],
        "results_raw_sha256": state["results_raw_sha256"],
        "result_manifest_path": state["result_manifest_path"],
        "result_manifest_raw_sha256": state["result_manifest_raw_sha256"],
        "result_manifest_payload_sha256": state["result_manifest_payload_sha256"],
        "score_reduction_runtime": state["score_reduction_runtime"],
        "generation_repeat_qc_path": state["generation_repeat_qc_path"],
        "generation_repeat_qc_raw_sha256": state["generation_repeat_qc_raw_sha256"],
        "generation_repeat_qc": state["generation_repeat_qc"],
        "generation_seed_family": plan.get("generation_seed_family", {}),
        "planned_rows": len(plan["rows"]),
        "result_rows": sum(state["status_counts"].values()),
        "status_counts": state["status_counts"],
        "status_counts_by_split": state["status_counts_by_split"],
        "counts_by_model_method": state["counts_by_model_method"],
        "model_count": len(state["models"]),
        "model_registry": [
            {
                "model_id": model_id,
                "audio_frontend": model.get("audio_frontend"),
                "role": model.get("role"),
                "model_sha256": model.get("model_sha256"),
                "code_sha256": model.get("code_sha256"),
                "score_model_sha256": model.get("score_model_sha256"),
                "preprocessing_sha256": model.get("preprocessing_sha256"),
                "fps_num": model.get("fps_num"),
                "fps_den": model.get("fps_den"),
            }
            for model_id, model in state["models"].items()
        ],
        "verified_source_file_count": len(state["file_cache"]),
        "verified_scored_row_count": len(state["score_arrays_by_id"]),
        "verified_pair_group_count": len(state.get("pair_group_audit", [])),
        "primary_audio_window_audit_by_row": [
            {
                "row_id": row_id,
                "window_sha256": media["primary_pcm16_window_sha256"],
                "sample_count": media["primary_pcm16_sample_count"],
                "start_sample": media["primary_pcm16_start_sample"],
                "requested_stop_sample_exclusive": media["primary_pcm16_requested_stop_sample_exclusive"],
                "actual_stop_sample_exclusive": media["primary_pcm16_stop_sample_exclusive"],
                "tail_clipped_to_natural_pcm": media["primary_pcm16_tail_clipped_to_natural_pcm"],
                "source_clock_frame_start": media["source_clock_frame_start"],
                "source_clock_frame_stop_exclusive": media["source_clock_frame_stop_exclusive"],
                "frame_map_sha256": media["selected_track_frame_map_sha256"],
                "expected_query_count": media["official_query_count"],
            }
            for row_id, media in state["media_by_id"].items()
        ],
        "pair_group_audit": state.get("pair_group_audit", []),
        "recomputed_support_metrics_by_row": rebuilt,
        "summary_endpoints": endpoint_rows,
        "simultaneous_family": family,
        "bootstrap": {
            "bit_generator": "PCG64",
            "seed": plan["analysis"]["bootstrap"]["seed"],
            "draw_count": plan["analysis"]["bootstrap"]["draw_count"],
            "speaker_order": plan["confirm_speaker_ids"],
            "index_matrix_npz": "speaker_bootstrap.npz",
            "index_matrix_file_sha256": npz_sha,
            "index_array_sha256": _array_sha256(index_matrix),
        },
        "issues": state["issues"],
    }
    (out / "audit.json").write_bytes(_canonical_json(audit) + b"\n")
    _write_summary(out, state, endpoint_rows, family)

    output_files = []
    for name in ("audit.json", "summary.md", "speaker_contrasts.csv", "speaker_bootstrap.npz"):
        path = out / name
        output_files.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "consumer_version": CONSUMER_VERSION,
        "run_id": plan["run_id"],
        "status": output_status,
        "score_reduction_runtime": state["score_reduction_runtime"],
        "files": output_files,
    }
    output_manifest["manifest_sha256"] = _payload_sha256(output_manifest, "manifest_sha256")
    (out / "consumer_manifest.json").write_bytes(_canonical_json(output_manifest) + b"\n")


def run_consumer(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    if out.exists():
        raise ConsumerError(f"output directory already exists; refusing overwrite: {out}")
    out.mkdir(parents=True, exist_ok=False)
    try:
        state = _load_and_validate_inputs(args)
        state["supports"] = {item["support_id"]: item for item in state["plan"]["supports"]}
        rebuilt = _rebuild_all_supports(state)
        _write_outputs(out, state, rebuilt)
        return 0 if state["audit_status"] == "PASS" else 2
    except Exception as exc:
        audit = {
            "schema_version": SCHEMA_VERSION,
            "consumer_version": CONSUMER_VERSION,
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "argv": sys.argv,
        }
        (out / "audit.json").write_bytes(_canonical_json(audit) + b"\n")
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="frozen plan.json")
    parser.add_argument("--results", required=True, help="results.jsonl with one row per plan row")
    parser.add_argument("--result-manifest", required=True, help="result_manifest.json")
    parser.add_argument("--generation-repeat-qc", required=True, help="separate generation_repeat_qc.jsonl, one row per model")
    parser.add_argument("--artifact-root", required=True, help="base for relative producer artifact paths")
    parser.add_argument("--out", required=True, help="new consumer output directory; must not exist")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="pinned ffmpeg executable")
    parser.add_argument("--ffprobe", default="ffprobe", help="pinned ffprobe executable")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_consumer(args)


if __name__ == "__main__":
    raise SystemExit(main())
