#!/usr/bin/env python3
"""Capture official SyncNet forward embeddings as a separate read-only sidecar.

This replays the pinned scorer on the already sealed primary Matroska inputs.
It never edits the producer plan/results or substitutes replay distances for
the primary score arrays. Each sidecar carries the frozen source-clock arrays,
the replayed distance matrix, and explicit audio/video lineage checks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "cross-tfg-syncnet-embedding-replay-v1"
SCORE_ARRAYS = (
    "query_frame_index", "query_timestamp_num", "query_timestamp_den",
    "track_frame_index", "valid_query_mask", "distance_by_query_lag",
    "lag_offsets_frames",
)
EMBEDDING_ARRAYS = (
    "video_embedding_forward_float32", "audio_embedding_forward_float32",
    "query_frame_index_int32",
)
LAGS = np.arange(-15, 16, dtype=np.int32)


class ReplayError(RuntimeError):
    pass


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha(value: Mapping[str, Any], field: str) -> str:
    return _sha_bytes(_canonical_json({key: item for key, item in value.items() if key != field}))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReplayError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ReplayError(f"expected JSON object at {path}:{line_no}")
            rows.append(row)
    return rows


def _path_ref(root: Path, ref: Mapping[str, Any], where: str) -> Path:
    raw_path = ref.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ReplayError(f"{where}.path must be a nonempty string")
    path = Path(raw_path)
    return (path if path.is_absolute() else root / path).resolve(strict=True)


def _checked_file(root: Path, ref: Mapping[str, Any], where: str) -> Path:
    path = _path_ref(root, ref, where)
    expected = ref.get("file_sha256")
    if not isinstance(expected, str) or len(expected) != 64 or _sha_file(path) != expected:
        raise ReplayError(f"{where} raw file SHA mismatch: {path}")
    return path


def _array_sha(array: np.ndarray) -> str:
    return _sha_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def _reduced_pair(numerator: int, denominator: int) -> tuple[int, int]:
    if type(numerator) is not int or type(denominator) is not int or numerator < 0 or denominator <= 0:
        raise ReplayError("invalid nominal frame-clock rational")
    common = math.gcd(numerator, denominator)
    return numerator // common, denominator // common


def _load_score_npz(root: Path, ref: Mapping[str, Any], row_id: str) -> tuple[Path, dict[str, np.ndarray]]:
    path = _checked_file(root, ref, f"{row_id}.score.raw_matrix_npz")
    seals = ref.get("arrays")
    if not isinstance(seals, dict):
        raise ReplayError(f"{row_id}: raw score NPZ lacks per-array seals")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(SCORE_ARRAYS) - set(archive.files):
                raise ReplayError(f"{row_id}: raw score NPZ lacks required arrays")
            arrays = {name: np.asarray(archive[name]) for name in SCORE_ARRAYS}
    except ReplayError:
        raise
    except Exception as exc:
        raise ReplayError(f"{row_id}: cannot load raw score NPZ: {exc}") from exc
    n = arrays["query_frame_index"].shape[0]
    expected_shapes = {
        "query_frame_index": (n,), "query_timestamp_num": (n,),
        "query_timestamp_den": (n,), "track_frame_index": (n,),
        "valid_query_mask": (n,), "distance_by_query_lag": (n, 31),
        "lag_offsets_frames": (31,),
    }
    for name, array in arrays.items():
        seal = seals.get(name)
        if not isinstance(seal, dict):
            raise ReplayError(f"{row_id}: missing array seal for {name}")
        if seal.get("dtype") not in (array.dtype.str, array.dtype.name) or seal.get("shape") != list(array.shape):
            raise ReplayError(f"{row_id}: dtype/shape seal mismatch for {name}")
        if seal.get("array_sha256") != _array_sha(array):
            raise ReplayError(f"{row_id}: array SHA mismatch for {name}")
        if array.shape != expected_shapes[name]:
            raise ReplayError(f"{row_id}: unexpected shape for {name}: {array.shape}")
    if arrays["distance_by_query_lag"].dtype != np.dtype("float32") or not np.isfinite(arrays["distance_by_query_lag"]).all():
        raise ReplayError(f"{row_id}: saved score matrix must be finite float32")
    expected_dtypes = {
        "query_frame_index": np.dtype("<i4"),
        "query_timestamp_num": np.dtype("<i8"),
        "query_timestamp_den": np.dtype("<i8"),
        "track_frame_index": np.dtype("<i4"),
        "valid_query_mask": np.dtype("bool"),
        "distance_by_query_lag": np.dtype("<f4"),
        "lag_offsets_frames": np.dtype("<i4"),
    }
    for name, expected_dtype in expected_dtypes.items():
        if arrays[name].dtype != expected_dtype:
            raise ReplayError(f"{row_id}: raw score array {name} dtype {arrays[name].dtype} != {expected_dtype}")
    if arrays["valid_query_mask"].dtype.kind != "b":
        raise ReplayError(f"{row_id}: valid_query_mask must be boolean")
    if not np.array_equal(arrays["lag_offsets_frames"], LAGS):
        raise ReplayError(f"{row_id}: saved lag axis is not exactly -15..15")
    if np.any(arrays["query_timestamp_den"] <= 0):
        raise ReplayError(f"{row_id}: query timestamps must have positive denominators")
    return path, arrays


def _verify_natural_pcm(root: Path, planned: Mapping[str, Any], row_id: str, wavfile: Any) -> Path:
    ref = planned.get("natural_pcm")
    if not isinstance(ref, dict):
        raise ReplayError(f"{row_id}: planned natural_pcm ref is missing")
    path = _checked_file(root, ref, f"{row_id}.natural_pcm")
    rate, samples = wavfile.read(str(path), mmap=False)
    if rate != 16_000 or samples.ndim != 1 or samples.dtype != np.dtype("float32"):
        raise ReplayError(f"{row_id}: natural PCM must be mono float32 at 16 kHz")
    canonical = np.ascontiguousarray(samples, dtype="<f4")
    pcm_sha = _sha_bytes(canonical.tobytes(order="C"))
    if pcm_sha != ref.get("pcm_sha256") or int(samples.size) != int(ref.get("sample_count", -1)):
        raise ReplayError(f"{row_id}: natural PCM content/count differs from frozen plan")
    return path


def _read_pcm16_window(root: Path, score: Mapping[str, Any], planned: Mapping[str, Any], row_id: str) -> tuple[Path, bytes, dict[str, Any]]:
    window = score.get("primary_pcm16_window")
    if not isinstance(window, dict):
        raise ReplayError(f"{row_id}: score.primary_pcm16_window is missing")
    path = _checked_file(root, window, f"{row_id}.score.primary_pcm16_window")
    if window.get("source_float32_pcm_sha256") != planned.get("natural_pcm", {}).get("pcm_sha256"):
        raise ReplayError(f"{row_id}: PCM16 window is not bound to original natural float32 PCM")
    if window.get("sample_rate_hz") != 16_000 or window.get("channels") != 1:
        raise ReplayError(f"{row_id}: PCM16 window format differs from frozen mono 16-kHz contract")
    try:
        with wave.open(str(path), "rb") as source:
            if source.getframerate() != 16_000 or source.getnchannels() != 1 or source.getsampwidth() != 2:
                raise ReplayError(f"{row_id}: PCM16 window WAV format is not mono 16-kHz signed PCM16")
            count = source.getnframes()
            samples = source.readframes(count)
    except ReplayError:
        raise
    except Exception as exc:
        raise ReplayError(f"{row_id}: cannot read PCM16 window WAV: {exc}") from exc
    pcm_sha = _sha_bytes(samples)
    if count != window.get("sample_count") or pcm_sha != window.get("pcm_sha256"):
        raise ReplayError(f"{row_id}: PCM16 window decoded bytes/count differ from their seals")
    if score.get("score_preprocessed_pcm16_sha256") != pcm_sha or score.get("score_preprocessed_sample_count") != count:
        raise ReplayError(f"{row_id}: score preprocessing summary does not bind the PCM16 window")
    return path, samples, window


def _decode_input_pcm16(ffmpeg: str, input_path: Path) -> bytes:
    command = [
        ffmpeg, "-v", "error", "-i", str(input_path), "-map", "0:a:0", "-vn",
        "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1",
    ]
    process = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process.stdout


def _packet_video_sha(ffmpeg: str, path: Path) -> str:
    command = [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-c:v", "copy", "-f", "hash", "-hash", "sha256", "-"]
    process = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    text = process.stdout.strip()
    if not text.startswith("SHA256="):
        raise ReplayError(f"cannot parse FFmpeg packet hash for {path}: {text!r}")
    value = text.split("=", 1)[1].lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ReplayError(f"invalid FFmpeg packet hash for {path}")
    return value


def _verify_frame_map(
    root: Path, result: Mapping[str, Any], score: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray], row_id: str, natural_sample_count: int,
) -> dict[str, Any]:
    render = result.get("render")
    if not isinstance(render, dict):
        raise ReplayError(f"{row_id}: render object is missing")
    crop = render.get("selected_track_crop")
    primary = score.get("primary_syncnet_input")
    if not isinstance(crop, dict) or not isinstance(primary, dict):
        raise ReplayError(f"{row_id}: selected crop / primary scorer input ref is missing")
    map_path_text = crop.get("frame_map_path")
    map_sha = crop.get("frame_map_sha256")
    if not isinstance(map_path_text, str) or not isinstance(map_sha, str):
        raise ReplayError(f"{row_id}: selected frame-map path/hash is missing")
    frame_map_path = Path(map_path_text)
    frame_map_path = (frame_map_path if frame_map_path.is_absolute() else root / frame_map_path).resolve(strict=True)
    if _sha_file(frame_map_path) != map_sha:
        raise ReplayError(f"{row_id}: selected frame-map file SHA mismatch")
    if primary.get("frame_map_sha256") != map_sha or primary.get("frame_map_path") != map_path_text:
        raise ReplayError(f"{row_id}: primary scorer input does not bind the selected frame map")
    fmap = _load_json(frame_map_path)
    if fmap.get("schema_version") != "cross-tfg-track-frame-map-v2":
        raise ReplayError(f"{row_id}: unsupported frame-map schema")
    mapped_render_path = fmap.get("render_video_path")
    if mapped_render_path == render.get("silent_video_path"):
        mapped_render_sha = render.get("silent_video_sha256")
    elif mapped_render_path == render.get("muxed_video_path"):
        mapped_render_sha = render.get("muxed_video_sha256")
    else:
        raise ReplayError(f"{row_id}: frame map references a render path not bound by the result")
    expected_bindings = {
        "selected_crop_video_path": crop.get("path"),
        "selected_crop_video_file_sha256": crop.get("file_sha256"),
        "selected_crop_video_elementary_sha256": crop.get("video_elementary_sha256"),
        "selected_track_id": crop.get("selected_track_id"),
        "selected_track_index": crop.get("selected_track_index"),
        "render_video_path": mapped_render_path,
        "render_video_file_sha256": mapped_render_sha,
        "render_video_elementary_sha256": render.get("video_stream_sha256"),
        "render_timestamps_sha256": render.get("timestamps_sha256"),
        "fps_num": render.get("fps_num"),
        "fps_den": render.get("fps_den"),
    }
    for key, expected in expected_bindings.items():
        if fmap.get(key) != expected:
            raise ReplayError(f"{row_id}: frame-map binding differs for {key}")
    frames = fmap.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ReplayError(f"{row_id}: frame map has no frames")
    n = int(arrays["query_frame_index"].shape[0])
    window = score.get("primary_pcm16_window") or {}
    expected_n = max(0, min(len(frames), int(window.get("sample_count", -1)) // 640) - 5)
    if n != expected_n:
        raise ReplayError(f"{row_id}: score rows {n} differ from frozen crop/audio query count {expected_n}")
    source_frames: list[int] = []
    timestamp_pairs: list[tuple[int, int]] = []
    for index, frame in enumerate(frames[:n]):
        if frame.get("crop_frame_index") != index:
            raise ReplayError(f"{row_id}: frame-map crop indices are not ordered")
        source_frame = frame.get("source_clock_frame_index")
        numerator, denominator = frame.get("timestamp_num"), frame.get("timestamp_den")
        if (
            type(source_frame) is not int or source_frame < 0
            or type(numerator) is not int or type(denominator) is not int or denominator <= 0
        ):
            raise ReplayError(f"{row_id}: invalid frame-map source-clock tuple at {index}")
        if math.gcd(numerator, denominator) != 1:
            raise ReplayError(f"{row_id}: frame-map timestamp is not reduced at {index}")
        if (numerator, denominator) != _reduced_pair(source_frame * render.get("fps_den"), render.get("fps_num")):
            raise ReplayError(f"{row_id}: frame-map timestamp differs from nominal source clock at {index}")
        source_frames.append(source_frame)
        timestamp_pairs.append((numerator, denominator))
    expected_frames = np.asarray(source_frames, dtype=np.int32)
    if not np.array_equal(arrays["query_frame_index"], expected_frames) or not np.array_equal(arrays["track_frame_index"], expected_frames):
        raise ReplayError(f"{row_id}: saved score query/track frames differ from crop frame map")
    expected_num = np.asarray([pair[0] for pair in timestamp_pairs], dtype=np.int64)
    expected_den = np.asarray([pair[1] for pair in timestamp_pairs], dtype=np.int64)
    if not np.array_equal(arrays["query_timestamp_num"], expected_num) or not np.array_equal(arrays["query_timestamp_den"], expected_den):
        raise ReplayError(f"{row_id}: saved score timestamps differ from crop frame map")
    fps_num, fps_den = render.get("fps_num"), render.get("fps_den")
    if type(fps_num) is not int or type(fps_den) is not int or fps_num <= 0 or fps_den <= 0:
        raise ReplayError(f"{row_id}: render FPS is invalid")
    samples_numerator = 16_000 * fps_den
    if samples_numerator % fps_num:
        raise ReplayError(f"{row_id}: frozen frame clock does not map to integer audio samples")
    samples_per_frame = samples_numerator // fps_num
    first_source_frame = crop.get("source_clock_frame_start")
    stop_source_frame = crop.get("source_clock_frame_stop_exclusive")
    if (
        type(first_source_frame) is not int or first_source_frame < 0
        or type(stop_source_frame) is not int or stop_source_frame <= first_source_frame
    ):
        raise ReplayError(f"{row_id}: selected crop source-clock span is invalid")
    if frames[0].get("source_clock_frame_index") != first_source_frame:
        raise ReplayError(f"{row_id}: selected crop start disagrees with its frame map")
    if stop_source_frame != first_source_frame + len(frames):
        raise ReplayError(f"{row_id}: selected crop stop disagrees with its frame-map length")
    requested_start = int(first_source_frame) * samples_per_frame
    requested_stop = int(stop_source_frame) * samples_per_frame
    if requested_start >= natural_sample_count:
        raise ReplayError(f"{row_id}: selected crop starts after original natural PCM ends")
    actual_stop = min(requested_stop, natural_sample_count)
    expected_window = {
        "start_sample": requested_start,
        "requested_stop_sample_exclusive": requested_stop,
        "stop_sample_exclusive": actual_stop,
        "sample_count": actual_stop - requested_start,
        "tail_clipped_to_natural_pcm": actual_stop < requested_stop,
        "source_frame_start": first_source_frame,
        "source_frame_stop_exclusive": stop_source_frame,
        "fps_num": fps_num,
        "fps_den": fps_den,
    }
    if actual_stop <= requested_start:
        raise ReplayError(f"{row_id}: selected crop has no original PCM samples")
    window = score.get("primary_pcm16_window") or {}
    for field, expected in expected_window.items():
        if window.get(field) != expected:
            raise ReplayError(f"{row_id}: PCM16 window {field} disagrees with frozen source-frame map")
    return {
        "path": str(frame_map_path), "file_sha256": map_sha,
        "frame_count": len(frames), "query_count": n,
        "source_frame_start": first_source_frame, "source_frame_stop_exclusive": stop_source_frame,
        **expected_window,
    }


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    seals = {
        name: {"dtype": array.dtype.str, "shape": list(array.shape), "array_sha256": _array_sha(array)}
        for name, array in arrays.items()
    }
    return {"path": path.name, "file_sha256": _sha_file(path), "file_bytes": path.stat().st_size, "arrays": seals}


def _row_replay(
    *, planned: Mapping[str, Any], result: Mapping[str, Any], plan: Mapping[str, Any], root: Path,
    instance: Any, capture: Any, ffmpeg: str, tmp_root: Path, out: Path, index: int,
    wavfile: Any, batch_size: int, vshift: int, pcm16_cache: dict[str, bytes],
) -> dict[str, Any]:
    row_id = str(planned.get("row_id"))
    row: dict[str, Any] = {key: planned.get(key) for key in ("row_id", "split", "sample_id", "speaker_id", "model_id", "face_id", "seed", "method_id")}
    row["source_result_status"] = result.get("status")
    if result.get("status") != "PASS":
        row.update({"status": "SOURCE_ROW_NOT_PASS", "reason": result.get("failure_reason")})
        return row
    score = result.get("score")
    if not isinstance(score, dict) or score.get("status") != "PASS":
        raise ReplayError(f"{row_id}: producer row is PASS but score object is not")
    render = result.get("render")
    crop = render.get("selected_track_crop") if isinstance(render, dict) else None
    if not isinstance(crop, dict):
        raise ReplayError(f"{row_id}: selected-track crop metadata is missing")
    models = {str(item.get("model_id")): item for item in plan.get("models", []) if isinstance(item, dict)}
    frozen_model = models.get(str(planned.get("model_id")))
    if frozen_model is None or score.get("score_model_sha256") != frozen_model.get("score_model_sha256"):
        raise ReplayError(f"{row_id}: score model SHA differs from the frozen model registry")
    if score.get("preprocessing_sha256") != frozen_model.get("preprocessing_sha256"):
        raise ReplayError(f"{row_id}: score preprocessing differs from the frozen model registry")
    natural_path = _verify_natural_pcm(root, planned, row_id, wavfile)
    natural_ref = planned["natural_pcm"]
    if score.get("score_audio_pcm_sha256") != natural_ref.get("pcm_sha256"):
        raise ReplayError(f"{row_id}: SyncNet score audio is not the original natural PCM")
    pcm_window_path, window_bytes, window = _read_pcm16_window(root, score, planned, row_id)
    primary_ref = score.get("primary_syncnet_input")
    if not isinstance(primary_ref, dict):
        raise ReplayError(f"{row_id}: primary SyncNet input ref is missing")
    primary_path = _checked_file(root, primary_ref, f"{row_id}.score.primary_syncnet_input")
    if primary_ref.get("audio_pcm16_sha256") != window.get("pcm_sha256"):
        raise ReplayError(f"{row_id}: primary input PCM16 digest differs from direct-natural window")
    for field, expected in (
        ("crop_video_path", crop.get("path")),
        ("crop_video_file_sha256", crop.get("file_sha256")),
        ("crop_video_elementary_sha256", crop.get("video_elementary_sha256")),
        ("selected_track_id", crop.get("selected_track_id")),
        ("selected_track_index", crop.get("selected_track_index")),
    ):
        if primary_ref.get(field) != expected:
            raise ReplayError(f"{row_id}: primary scorer input does not bind selected-crop field {field}")
    decoded_pcm = _decode_input_pcm16(ffmpeg, primary_path)
    if decoded_pcm != window_bytes:
        raise ReplayError(f"{row_id}: primary Matroska audio does not byte-match the direct-natural PCM16 window")
    if int(primary_ref.get("audio_sample_count", -1)) * 2 != len(decoded_pcm):
        raise ReplayError(f"{row_id}: primary input PCM16 sample count differs from decoded bytes")
    crop_path = _checked_file(root, {"path": crop.get("path"), "file_sha256": crop.get("file_sha256")}, f"{row_id}.selected_track_crop")
    primary_video_packet_sha = _packet_video_sha(ffmpeg, primary_path)
    crop_video_packet_sha = _packet_video_sha(ffmpeg, crop_path)
    if primary_video_packet_sha != crop_video_packet_sha:
        raise ReplayError(f"{row_id}: primary SyncNet input changed selected crop video packets")
    if primary_ref.get("video_elementary_sha256") != primary_video_packet_sha or primary_ref.get("crop_video_elementary_sha256") != crop_video_packet_sha:
        raise ReplayError(f"{row_id}: primary/crop video packet SHA differs from producer metadata")
    score_path, score_arrays = _load_score_npz(root, score.get("raw_matrix_npz") or {}, row_id)
    frame_map = _verify_frame_map(root, result, score, score_arrays, row_id, int(natural_ref.get("sample_count", -1)))
    direct_full_pcm16 = pcm16_cache.get(str(natural_ref.get("pcm_sha256")))
    if direct_full_pcm16 is None:
        direct_full_pcm16 = _decode_input_pcm16(ffmpeg, natural_path)
        if len(direct_full_pcm16) != int(natural_ref.get("sample_count", -1)) * 2:
            raise ReplayError(f"{row_id}: direct-natural PCM16 conversion changed the full sample count")
        pcm16_cache[str(natural_ref.get("pcm_sha256"))] = direct_full_pcm16
    direct_slice = direct_full_pcm16[frame_map["start_sample"] * 2 : frame_map["stop_sample_exclusive"] * 2]
    if direct_slice != window_bytes:
        raise ReplayError(f"{row_id}: PCM16 window is not an exact direct-natural sample slice")
    query_frames = np.asarray(score_arrays["query_frame_index"], dtype=np.int32)
    tmp_ref = "embedding_replay_" + hashlib.sha256(row_id.encode("utf-8")).hexdigest()[:20]
    opt = SimpleNamespace(tmp_dir=str(tmp_root), reference=tmp_ref, batch_size=batch_size, vshift=vshift)
    official, captured = capture.evaluate_with_forward_capture(instance, opt, str(primary_path), query_frames)
    replay_distance = np.asarray(official[2])
    saved_distance = score_arrays["distance_by_query_lag"]
    if replay_distance.shape != saved_distance.shape or replay_distance.dtype != np.dtype("float32"):
        raise ReplayError(f"{row_id}: replay matrix shape/dtype differs from sealed primary matrix")
    if not np.isfinite(replay_distance).all():
        raise ReplayError(f"{row_id}: replay distance matrix contains non-finite values")
    replay_offset = float(np.asarray(official[0]).item())
    replay_confidence = float(np.asarray(official[1]).item())
    if not math.isfinite(replay_offset) or not math.isfinite(replay_confidence):
        raise ReplayError(f"{row_id}: official replay offset/confidence is non-finite")
    difference = np.abs(replay_distance.astype(np.float64) - saved_distance.astype(np.float64))
    exact = bool(np.array_equal(replay_distance, saved_distance))
    max_abs = float(difference.max(initial=0.0))
    mismatch_count = int(np.count_nonzero(replay_distance != saved_distance))
    sidecar_arrays: dict[str, np.ndarray] = {
        **{name: np.ascontiguousarray(score_arrays[name]) for name in SCORE_ARRAYS},
        **{name: np.ascontiguousarray(captured[name]) for name in EMBEDDING_ARRAYS},
        "distance_by_query_lag_replay_float32": np.ascontiguousarray(replay_distance, dtype="<f4"),
    }
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", row_id)
    npz_path = out / f"{index:04d}_{safe_id}.npz"
    npz_ref = _write_npz(npz_path, sidecar_arrays)
    tmp_evaluation = tmp_root / tmp_ref
    if tmp_evaluation.exists():
        shutil.rmtree(tmp_evaluation)
    row.update({
        "status": "PASS",
        "forward_stage": "official_forward_outputs_before_distance_no_extra_normalization",
        "primary_audio_source": {
            "natural_pcm_path": str(natural_path), "natural_pcm_file_sha256": natural_ref.get("file_sha256"),
            "natural_pcm_float32_sha256": natural_ref.get("pcm_sha256"), "score_audio_pcm_sha256": score.get("score_audio_pcm_sha256"),
            "pcm16_window_path": str(pcm_window_path), "pcm16_window_file_sha256": window.get("file_sha256"),
            "pcm16_window_sha256": window.get("pcm_sha256"), "sample_count": window.get("sample_count"),
            "primary_input_path": str(primary_path), "primary_input_file_sha256": primary_ref.get("file_sha256"),
            "primary_input_audio_pcm16_sha256": primary_ref.get("audio_pcm16_sha256"),
            "decoded_primary_input_pcm16_sha256": _sha_bytes(decoded_pcm), "pcm16_exact_match": True,
        },
        "primary_video_source": {
            "primary_input_packet_sha256": primary_video_packet_sha,
            "selected_crop_packet_sha256": crop_video_packet_sha,
            "packet_exact_match": True, "frame_map": frame_map,
        },
        "primary_score_matrix": {
            "path": str(score_path), "file_sha256": score["raw_matrix_npz"]["file_sha256"],
            "array_sha256": score["raw_matrix_npz"]["arrays"]["distance_by_query_lag"]["array_sha256"],
            "replay_exact_match": exact, "replay_max_abs_difference": max_abs,
            "replay_mismatched_elements": mismatch_count, "replay_shape": list(replay_distance.shape),
        },
        "official_replay": {"offset": replay_offset, "confidence": replay_confidence},
        "embedding_npz": npz_ref,
    })
    return row


def run(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    if out.exists():
        raise ReplayError(f"output directory already exists: {out}")
    plan_path = Path(args.plan).resolve(strict=True)
    results_path = Path(args.results).resolve(strict=True)
    manifest_path = Path(args.result_manifest).resolve(strict=True)
    root = Path(args.artifact_root).resolve(strict=True)
    plan = _load_json(plan_path)
    manifest = _load_json(manifest_path)
    plan_raw_sha, results_raw_sha, manifest_raw_sha = _sha_file(plan_path), _sha_file(results_path), _sha_file(manifest_path)
    for field, expected, actual in (
        ("plan", args.expected_plan_raw_sha256, plan_raw_sha),
        ("results", args.expected_results_raw_sha256, results_raw_sha),
        ("result manifest", args.expected_manifest_raw_sha256, manifest_raw_sha),
    ):
        if actual != expected:
            raise ReplayError(f"{field} raw SHA differs from explicit replay pin")
    if plan.get("schema_version") != "cross-tfg-audio-eval-v1" or plan.get("status") != "FROZEN" or plan.get("plan_sha256") != _payload_sha(plan, "plan_sha256"):
        raise ReplayError("plan schema/status/fingerprint is invalid")
    if manifest.get("schema_version") != "cross-tfg-audio-eval-v1" or manifest.get("run_id") != plan.get("run_id") or manifest.get("manifest_sha256") != _payload_sha(manifest, "manifest_sha256"):
        raise ReplayError("result-manifest schema/status/fingerprint is invalid")
    if manifest.get("plan_raw_sha256") != plan_raw_sha or manifest.get("plan_sha256") != plan.get("plan_sha256") or manifest.get("results_jsonl_sha256") != results_raw_sha:
        raise ReplayError("result manifest does not bind exact plan/results")
    plan_rows, results_rows = plan.get("rows"), _load_jsonl(results_path)
    if not isinstance(plan_rows, list) or len(plan_rows) != len(results_rows) or manifest.get("planned_rows") != len(plan_rows) or manifest.get("result_rows") != len(results_rows):
        raise ReplayError("plan/results/manifest row counts differ")
    if [row.get("row_id") for row in plan_rows] != [row.get("row_id") for row in results_rows]:
        raise ReplayError("results do not preserve frozen plan row order")
    models = plan.get("models")
    if not isinstance(models, list) or len(models) != 1:
        raise ReplayError("expected one frozen scoring model for this replay")
    model_by_id = {item.get("model_id"): item for item in models if isinstance(item, dict)}
    if len(model_by_id) != 1:
        raise ReplayError("frozen scoring model IDs are invalid")
    syncnet = plan.get("protocol", {}).get("syncnet", {})
    source_pins = syncnet.get("source_sha256", {})
    source_paths = {
        "SyncNetInstance.py": Path(args.syncnet_source_root) / "SyncNetInstance.py",
        "SyncNetModel.py": Path(args.syncnet_source_root) / "SyncNetModel.py",
    }
    source_hashes: dict[str, str] = {}
    for name, path in source_paths.items():
        path = path.resolve(strict=True)
        actual = _sha_file(path)
        if actual != source_pins.get(name):
            raise ReplayError(f"{name} source SHA differs from the frozen plan")
        source_paths[name] = path
        source_hashes[name] = actual
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    checkpoint_sha = _sha_file(checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise ReplayError("SyncNet checkpoint differs from explicit replay pin")
    if model_by_id[next(iter(model_by_id))].get("score_model_sha256") != checkpoint_sha:
        raise ReplayError("SyncNet checkpoint does not match the frozen model registry")
    ffmpeg = shutil.which(args.ffmpeg)
    if ffmpeg is None:
        raise ReplayError(f"ffmpeg is unavailable: {args.ffmpeg}")
    ffmpeg_path = Path(ffmpeg).resolve(strict=True)
    conversion = syncnet.get("primary_audio_input")
    expected_conversion_argv = ["-v", "error", "-i", "{natural_pcm_path}", "-map", "0:a:0", "-vn", "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1"]
    if (
        not isinstance(conversion, dict)
        or conversion.get("mode") != "canonical_natural_pcm16_crop_window"
        or conversion.get("source") != "natural_pcm"
        or conversion.get("codec") != "pcm_s16le"
        or conversion.get("argv_template") != expected_conversion_argv
    ):
        raise ReplayError("plan does not freeze the expected direct-natural PCM16 conversion")
    expected_ffmpeg_sha = conversion.get("ffmpeg_sha256")
    if not isinstance(expected_ffmpeg_sha, str) or _sha_file(ffmpeg_path) != expected_ffmpeg_sha:
        raise ReplayError("actual FFmpeg binary differs from the frozen PCM16 conversion source")
    out.mkdir(parents=True)
    tmp_root = Path(args.tmp_root).resolve()
    tmp_root.mkdir(parents=True, exist_ok=True)
    capture_path = Path(args.capture_wrapper_source).resolve(strict=True)
    if _sha_file(capture_path) != args.expected_capture_wrapper_sha256:
        raise ReplayError("embedding capture wrapper SHA differs from explicit pin")
    sys.path.insert(0, str(capture_path.parent))
    sys.path.insert(0, str(source_paths["SyncNetInstance.py"].parent))
    capture = importlib.import_module(capture_path.stem)
    from scipy.io import wavfile
    import torch
    from SyncNetInstance import SyncNetInstance

    torch.set_num_threads(1)
    instance = SyncNetInstance(device=args.device)
    instance.loadParameters(str(checkpoint))
    instance.eval()
    output_rows: list[dict[str, Any]] = []
    pcm16_cache: dict[str, bytes] = {}
    for index, (planned, result) in enumerate(zip(plan_rows, results_rows, strict=True)):
        try:
            for field in ("split", "sample_id", "speaker_id", "model_id", "face_id", "seed", "method_id"):
                if result.get(field) != planned.get(field):
                    raise ReplayError(f"result identity differs from plan at {field}")
            row = _row_replay(
                planned=planned, result=result, plan=plan, root=root, instance=instance,
                capture=capture, ffmpeg=str(ffmpeg_path), tmp_root=tmp_root, out=out,
                index=index, wavfile=wavfile, batch_size=args.batch_size, vshift=args.vshift,
                pcm16_cache=pcm16_cache,
            )
        except Exception as exc:
            row = {key: planned.get(key) for key in ("row_id", "split", "sample_id", "speaker_id", "model_id", "face_id", "seed", "method_id")}
            row.update({"source_result_status": result.get("status"), "status": "FAILED", "failure_reason": f"{type(exc).__name__}: {exc}"})
        output_rows.append(row)
    with (out / "embedding_replay_rows.jsonl").open("wb") as handle:
        for row in output_rows:
            handle.write(_canonical_json(row) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    status_counts = Counter(row.get("status", "MISSING") for row in output_rows)
    exact_count = sum(row.get("primary_score_matrix", {}).get("replay_exact_match") is True for row in output_rows)
    matrix_parity_counts = Counter(
        "EXACT" if row.get("primary_score_matrix", {}).get("replay_exact_match") is True
        else "DIFFERENT" if isinstance(row.get("primary_score_matrix"), dict) else "NOT_AVAILABLE"
        for row in output_rows
    )
    files = []
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "embedding_replay_manifest.json":
            files.append({"path": path.name, "file_sha256": _sha_file(path), "size_bytes": path.stat().st_size})
    runtime = {
        "python": sys.version.split()[0], "torch": str(torch.__version__),
        "numpy": np.__version__, "device": args.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") and torch.cuda.is_available() else None,
        "torch_num_threads": int(torch.get_num_threads()),
        "ffmpeg_path": str(ffmpeg_path), "ffmpeg_sha256": _sha_file(ffmpeg_path),
    }
    output_status = "PASS" if len(output_rows) == len(plan_rows) and status_counts["PASS"] == len(plan_rows) and matrix_parity_counts["DIFFERENT"] == 0 else "INCOMPLETE"
    manifest_out: dict[str, Any] = {
        "schema_version": SCHEMA, "status": output_status, "run_id": plan.get("run_id"),
        "plan_path": str(plan_path), "plan_raw_sha256": plan_raw_sha,
        "plan_payload_sha256": plan.get("plan_sha256"), "results_path": str(results_path),
        "results_raw_sha256": results_raw_sha, "result_manifest_path": str(manifest_path),
        "result_manifest_raw_sha256": manifest_raw_sha, "result_manifest_payload_sha256": manifest.get("manifest_sha256"),
        "artifact_root": str(root), "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "syncnet_source_paths": {key: str(value) for key, value in source_paths.items()},
        "syncnet_source_sha256": source_hashes,
        "capture_wrapper_path": str(capture_path), "capture_wrapper_sha256": _sha_file(capture_path),
        "rows_planned": len(plan_rows), "rows_captured": status_counts["PASS"],
        "rows_failed": status_counts["FAILED"], "rows_not_pass_in_source": status_counts["SOURCE_ROW_NOT_PASS"],
        "rows_matrix_replay_exact": exact_count, "status_counts": dict(status_counts),
        "matrix_parity_counts": dict(matrix_parity_counts), "runtime": runtime,
        "scope": {"split": "dev", "score_outcomes_used_for_selection": False, "primary_results_modified": False, "gpu_forward": args.device.startswith("cuda")},
        "files": files,
        "interpretation": "Forward-vector sidecar only. Primary distance matrices, audio, video, lags, and endpoints remain the sealed producer values; replay parity is reported separately.",
    }
    manifest_out["manifest_sha256"] = _payload_sha(manifest_out, "manifest_sha256")
    (out / "embedding_replay_manifest.json").write_bytes(_canonical_json(manifest_out) + b"\n")
    print(json.dumps({"status": output_status, "rows": len(output_rows), "captured": status_counts["PASS"], "failed": status_counts["FAILED"], "matrix_exact": exact_count, "out": str(out)}, sort_keys=True))
    return 0 if output_status == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--result-manifest", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--expected-plan-raw-sha256", required=True)
    parser.add_argument("--expected-results-raw-sha256", required=True)
    parser.add_argument("--expected-manifest-raw-sha256", required=True)
    parser.add_argument("--syncnet-source-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--capture-wrapper-source", required=True)
    parser.add_argument("--expected-capture-wrapper-sha256", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--tmp-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--vshift", type=int, default=15)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"syncnet_embedding_replay_failed:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
