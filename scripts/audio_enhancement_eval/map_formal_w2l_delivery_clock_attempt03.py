#!/usr/bin/env python3
"""Read-only frame-clock audit for one frozen W2L RMS delivery row.

The input bundle contains the exact producer result line and its local seals.
This script verifies the referenced remote media/matrix bytes, then compares
the decoded generated MP4 with the full-frame JPEG sequence actually consumed
by SyncNet's run_pipeline path. It performs no render, mux, model forward,
audio transform, or score recomputation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class AuditError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def pin_file(path_text: str, expected_sha: str, label: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.is_file():
        raise AuditError(f"missing frozen {label}: {path}")
    size = path.stat().st_size
    actual = sha256_file(path)
    if actual != expected_sha:
        raise AuditError(f"SHA-256 mismatch for frozen {label}: {path}")
    return {"path": str(path), "bytes": size, "sha256": actual}


def run_json(argv: list[str]) -> dict[str, Any]:
    result = subprocess.run(argv, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise AuditError(f"read-only probe failed ({Path(argv[0]).name}, exit {result.returncode})")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AuditError(f"invalid JSON from {Path(argv[0]).name}") from exc


def ffprobe_video(path: Path, ffprobe: str) -> dict[str, Any]:
    payload = run_json([
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,time_base,start_time,duration,nb_frames:frame=best_effort_timestamp_time,pkt_pts_time,pkt_dts_time,key_frame",
        "-of", "json", str(path),
    ])
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise AuditError(f"ffprobe did not return exactly one selected video stream: {path}")
    frames = payload.get("frames", [])
    pts = []
    for frame in frames:
        value = frame.get("best_effort_timestamp_time", frame.get("pkt_pts_time"))
        pts.append(value)
    return {"stream": streams[0], "decoded_frame_pts_seconds": pts, "frame_count": len(frames)}


def decode_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise AuditError(f"OpenCV cannot decode pinned video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                raise AuditError(f"unexpected decoded frame shape in {path}")
            frames.append(np.ascontiguousarray(frame))
    finally:
        capture.release()
    if not frames:
        raise AuditError(f"video decoded to zero frames: {path}")
    return frames


def read_jpegs(folder: Path) -> tuple[list[Path], list[np.ndarray], list[dict[str, Any]]]:
    paths = sorted(path for path in folder.glob("*.jpg") if path.is_file())
    if not paths:
        raise AuditError(f"SyncNet JPEG input directory is empty: {folder}")
    names = [path.name for path in paths]
    if any(re.fullmatch(r"\d{6}\.jpg", name) is None for name in names):
        raise AuditError("run_pipeline JPEG names do not match six-digit frame sequence")
    numeric = [int(path.stem) for path in paths]
    if numeric != list(range(numeric[0], numeric[0] + len(numeric))):
        raise AuditError("run_pipeline JPEG sequence has missing or duplicate indices")
    frames: list[np.ndarray] = []
    inventory: list[dict[str, Any]] = []
    for path in paths:
        payload = path.read_bytes()
        frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise AuditError(f"cannot decode run_pipeline JPEG: {path.name}")
        frames.append(np.ascontiguousarray(frame))
        inventory.append({"name": path.name, "bytes": len(payload), "sha256": sha256_bytes(payload)})
    return paths, frames, inventory


def replay_run_pipeline_jpegs(row: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Replay only run_pipeline's two pinned FFmpeg conversion steps.

    The original run_pipeline scratch JPEG folder may be deleted after scoring.
    These files are explicitly labeled replayed; they are never represented as
    the historical JPEG files. No detector, tracker, model, or scorer is run.
    """
    render = row["render"]
    argv = render["syncnet_tracking"]["argv"]
    pipeline_path = Path(require_argv_value(argv, "--videofile"))
    reference = require_argv_value(argv, "--reference")
    ffmpeg = Path("/usr/bin/ffmpeg")
    if not ffmpeg.is_file():
        raise AuditError("pinned FFmpeg binary is missing")
    pipeline_source = Path(argv[1])
    expected_pipeline_sha = "3654a3cf22c74cb3071fca6fb97ca9fb7dd1a5943fe864169ff8c0b57eb80acf"
    expected_ffmpeg_sha = "36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45"
    if sha256_file(pipeline_source) != expected_pipeline_sha:
        raise AuditError("pinned run_pipeline source changed before replay")
    if sha256_file(ffmpeg) != expected_ffmpeg_sha:
        raise AuditError("pinned FFmpeg binary changed before replay")
    if sha256_file(pipeline_path) != render["muxed_video_sha256"]:
        raise AuditError("frozen natural-PCM mux changed before replay")

    replay_root = output_dir / "replayed_syncnet_pipeline"
    avi_dir = replay_root / "pyavi" / reference
    jpeg_dir = replay_root / "pyframes" / reference
    avi_dir.mkdir(parents=True, exist_ok=False)
    jpeg_dir.mkdir(parents=True, exist_ok=False)
    replay_avi = avi_dir / "video.avi"
    avi_argv = [str(ffmpeg), "-y", "-loglevel", "error", "-i", str(pipeline_path),
                "-qscale:v", "2", "-async", "1", "-r", "25", str(replay_avi)]
    jpeg_argv = [str(ffmpeg), "-y", "-loglevel", "error", "-i", str(replay_avi),
                 "-qscale:v", "2", "-threads", "1", "-f", "image2", str(jpeg_dir / "%06d.jpg")]
    command_receipts = []
    for name, command in (("run_pipeline_video_avi_replay", avi_argv),
                          ("run_pipeline_jpeg_replay", jpeg_argv)):
        completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command_receipts.append({
            "name": name,
            "argv": command,
            "returncode": int(completed.returncode),
            "stdout_bytes": len(completed.stdout),
            "stdout_sha256": sha256_bytes(completed.stdout),
            "stderr_bytes": len(completed.stderr),
            "stderr_sha256": sha256_bytes(completed.stderr),
        })
        if completed.returncode != 0:
            raise AuditError(f"pinned {name} failed with return code {completed.returncode}")
    paths, frames, inventory = read_jpegs(jpeg_dir)
    if not replay_avi.is_file() or not paths:
        raise AuditError("pinned replay did not produce its expected AVI/JPEG files")
    return {
        "classification": "REPLAYED_PIPELINE_FRAMES_NOT_HISTORICAL_FILES",
        "run_pipeline_source": {"path": str(pipeline_source), "bytes": pipeline_source.stat().st_size,
                                 "sha256": sha256_file(pipeline_source)},
        "ffmpeg_binary": {"path": str(ffmpeg), "bytes": ffmpeg.stat().st_size,
                          "sha256": sha256_file(ffmpeg)},
        "source_mux": {"path": str(pipeline_path), "bytes": pipeline_path.stat().st_size,
                       "sha256": sha256_file(pipeline_path)},
        "replayed_video_avi": {"path": str(replay_avi), "bytes": replay_avi.stat().st_size,
                               "sha256": sha256_file(replay_avi)},
        "replayed_jpeg_dir": str(jpeg_dir),
        "replayed_jpeg_count": len(paths),
        "replayed_jpeg_inventory": inventory,
        "commands": command_receipts,
        "_jpeg_frames": frames,
    }


def crop_replayed_frames_from_track(jpeg_frames: list[np.ndarray], tracks_path: Path,
                                    track_index: int) -> dict[str, Any]:
    """Apply the pinned run_pipeline crop_video geometry to replayed JPEGs."""
    with tracks_path.open("rb") as handle:
        tracks = pickle.load(handle)
    if not isinstance(tracks, list) or not 0 <= track_index < len(tracks):
        raise AuditError("selected track index is absent from the frozen tracks pickle")
    entry = tracks[track_index]
    if not isinstance(entry, dict) or not isinstance(entry.get("track"), dict) or not isinstance(entry.get("proc_track"), dict):
        raise AuditError("frozen tracks pickle entry does not expose track/proc_track mappings")
    track = entry["track"]
    proc = entry["proc_track"]
    frame_indices = [int(value) for value in np.asarray(track.get("frame")).tolist()]
    track_frames = np.asarray(track.get("frame"))
    bbox = np.asarray(track.get("bbox"))
    if track_frames.ndim != 1 or bbox.ndim != 2 or bbox.shape[0] != track_frames.size or bbox.shape[1] != 4:
        raise AuditError("frozen selected track frame/bbox arrays have unexpected geometry")
    if len(frame_indices) == 0 or min(frame_indices) < 0 or max(frame_indices) >= len(jpeg_frames):
        raise AuditError("frozen selected track references replayed JPEG indices outside the sequence")
    if any(key not in proc for key in ("x", "y", "s")):
        raise AuditError("frozen selected proc_track is missing x/y/s arrays")
    values = {key: np.asarray(proc[key], dtype=np.float64) for key in ("x", "y", "s")}
    if any(array.shape != (len(frame_indices),) or not np.isfinite(array).all() for array in values.values()):
        raise AuditError("frozen selected proc_track x/y/s arrays do not align to its track frames")
    crops: list[np.ndarray] = []
    geometry_rows: list[dict[str, Any]] = []
    for local_index, source_jpeg_index in enumerate(frame_indices):
        bs = float(values["s"][local_index])
        cx = float(values["x"][local_index])
        cy = float(values["y"][local_index])
        bsi = int(bs * (1.0 + 2.0 * 0.40))
        image = jpeg_frames[source_jpeg_index]
        padded = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), mode="constant", constant_values=110)
        my, mx = cy + bsi, cx + bsi
        y0, y1 = int(my - bs), int(my + bs * (1.0 + 2.0 * 0.40))
        x0, x1 = int(mx - bs * (1.0 + 0.40)), int(mx + bs * (1.0 + 0.40))
        crop = padded[y0:y1, x0:x1]
        if crop.size == 0:
            raise AuditError(f"official selected-track crop is empty at local frame {local_index}")
        resized = np.ascontiguousarray(cv2.resize(crop, (224, 224)))
        crops.append(resized)
        geometry_rows.append({
            "crop_local_frame_index": local_index,
            "replayed_jpeg_index_zero_based": source_jpeg_index,
            "track_bbox_xyxy": [float(value) for value in bbox[local_index].tolist()],
            "proc_track_x": cx,
            "proc_track_y": cy,
            "proc_track_s": bs,
            "padding_bsi": bsi,
            "crop_xyxy_padded": [x0, y0, x1, y1],
            "recreated_crop_array_sha256": array_sha256(resized),
        })
    return {"crops": crops, "frames": geometry_rows, "frame_count": len(crops)}


def minimum_mae_monotone_map(source: list[np.ndarray], observed: list[np.ndarray]) -> dict[str, Any]:
    """Map JPEG rows to MP4 source rows by constrained minimum-MAE order.

    The output may repeat or skip source frames. This is an image
    correspondence estimate, not a claim of exact frame identity.
    """
    n_source, n_observed = len(source), len(observed)
    if min(n_source, n_observed) <= 0 or abs(n_observed - n_source) > 8:
        raise AuditError(f"unexpected source/JPEG frame-count relation {n_source}->{n_observed}")
    for frame in source + observed:
        if frame.shape != source[0].shape:
            raise AuditError("source MP4 and run_pipeline JPEG frame geometries differ")
    size = (64, 64)
    src_small = [cv2.resize(frame, size, interpolation=cv2.INTER_AREA).astype(np.int16) for frame in source]
    obs_small = [cv2.resize(frame, size, interpolation=cv2.INTER_AREA).astype(np.int16) for frame in observed]
    cost = np.empty((n_observed, n_source), dtype=np.float64)
    for j, current in enumerate(obs_small):
        for i, ref in enumerate(src_small):
            cost[j, i] = float(np.mean(np.abs(current - ref), dtype=np.float64))

    inf = np.inf
    dp = np.full((n_observed, n_source), inf, dtype=np.float64)
    parent = np.full((n_observed, n_source), -1, dtype=np.int32)
    dp[0, 0] = cost[0, 0]
    for j in range(1, n_observed):
        for i in range(n_source):
            previous = [k for k in (i - 1, i, i - 2) if 0 <= k < n_source and np.isfinite(dp[j - 1, k])]
            if previous:
                best = min(previous, key=lambda k: (dp[j - 1, k], k))
                dp[j, i] = dp[j - 1, best] + cost[j, i]
                parent[j, i] = best
    if not np.isfinite(dp[-1, -1]):
        raise AuditError("no endpoint-constrained monotone map satisfies actual frame counts")
    mapping = [n_source - 1]
    for j in range(n_observed - 1, 0, -1):
        previous = int(parent[j, mapping[-1]])
        if previous < 0:
            raise AuditError("failed to backtrace minimum-MAE frame map")
        mapping.append(previous)
    mapping.reverse()
    steps = np.diff(np.asarray(mapping, dtype=np.int64))
    if mapping[0] != 0 or mapping[-1] != n_source - 1 or np.any((steps < 0) | (steps > 2)):
        raise AuditError("minimum-MAE map violates endpoint or monotonicity contract")

    mapped = []
    for j, i in enumerate(mapping):
        ranked = np.sort(cost[j])
        source_frame = source[i]
        observed_frame = observed[j]
        full_mae = float(np.mean(np.abs(source_frame.astype(np.int16) - observed_frame.astype(np.int16)), dtype=np.float64))
        mapped.append({
            "jpeg_index_zero_based": j,
            "source_mp4_frame_index_zero_based": int(i),
            "source_mp4_pts_index_nominal": int(i),
            "jpeg_order_index_nominal": int(j),
            "downsample_64x64_mae_0_255": float(cost[j, i]),
            "best_candidate_mae_0_255": float(ranked[0]),
            "second_candidate_mae_0_255": float(ranked[1]) if n_source > 1 else None,
            "full_frame_mae_0_255": full_mae,
            "decoded_source_frame_sha256": array_sha256(source_frame),
            "decoded_jpeg_frame_sha256": array_sha256(observed_frame),
        })
    return {
        "method": "minimum_mae_monotone_correspondence_64x64_BGR_allow_repeat_or_skip_step_2",
        "interpretation": "estimated pixel correspondence; not asserted to be exact source identity",
        "source_mp4_frame_count": n_source,
        "pipeline_jpeg_frame_count": n_observed,
        "total_downsample_mae_0_255": float(dp[-1, -1]),
        "repeat_steps": int(np.sum(steps == 0)),
        "one_frame_advances": int(np.sum(steps == 1)),
        "two_frame_advances": int(np.sum(steps == 2)),
        "frames": mapped,
    }


def require_argv_value(argv: list[str], name: str) -> str:
    try:
        index = argv.index(name)
        return argv[index + 1]
    except (ValueError, IndexError) as exc:
        raise AuditError(f"frozen producer argv is missing {name}") from exc


def verify_npz(row: dict[str, Any]) -> dict[str, Any]:
    npz_ref = row["score"]["raw_matrix_npz"]
    pin = pin_file(npz_ref["path"], npz_ref["file_sha256"], "official raw matrix NPZ")
    with np.load(pin["path"], allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    expected = npz_ref.get("arrays", {})
    if set(arrays) != set(expected):
        raise AuditError("official raw matrix NPZ key set differs from frozen result row")
    array_receipts: dict[str, Any] = {}
    for key, array in arrays.items():
        actual_sha = array_sha256(array)
        spec = expected[key]
        if list(array.shape) != spec.get("shape") or array.dtype.str != spec.get("dtype") or actual_sha != spec.get("array_sha256"):
            raise AuditError(f"official raw matrix NPZ array seal mismatch: {key}")
        array_receipts[key] = {"shape": list(array.shape), "dtype": array.dtype.str, "sha256": actual_sha}
    distances = arrays.get("distance_by_query_lag")
    if distances is None or distances.ndim != 2 or distances.shape[1] != 31 or distances.dtype != np.float32:
        raise AuditError("official stored SyncNet matrix is not float32 [queries,31]")
    return {"file": pin, "arrays": array_receipts, "query_count": int(distances.shape[0]), "lag_count": 31,
            "query_frame_index": arrays["query_frame_index"].astype(np.int64).tolist(),
            "track_frame_index": arrays["track_frame_index"].astype(np.int64).tolist(),
            "query_timestamp_num": arrays["query_timestamp_num"].astype(np.int64).tolist(),
            "query_timestamp_den": arrays["query_timestamp_den"].astype(np.int64).tolist(),
            "valid_query_count": int(np.asarray(arrays["valid_query_mask"], dtype=np.bool_).sum())}


def collect(bundle: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    row_line = bundle.get("producer_row_line")
    if not isinstance(row_line, str) or sha256_bytes(row_line.encode("utf-8")) != bundle.get("producer_row_line_sha256"):
        raise AuditError("producer result-line bytes do not match the local row-line seal")
    row = json.loads(row_line)
    if (row.get("status") != "PASS" or row.get("method_id") != "rms_fixed"
            or row.get("model_id") != "wav2lip_gan_bac9a81e" or row.get("split") != "dev"):
        raise AuditError("input is not the frozen dev W2L RMS delivery row")
    render = row["render"]
    crop = render["selected_track_crop"]
    source_specs = [
        (render["generated_video_path"], render["generated_video_sha256"], "generated W2L MP4"),
        (render["silent_video_path"], render["silent_video_sha256"], "silent lossless container"),
        (render["muxed_video_path"], render["muxed_video_sha256"], "natural-PCM mux input to run_pipeline"),
        (render["timestamps_path"], render["timestamps_sha256"], "render timestamp sidecar"),
        (render["crop_roi_path"], render["crop_roi_sha256"], "official crop ROI sidecar"),
        (crop["crop_avi_path"], crop["crop_avi_file_sha256"], "official selected-track crop AVI"),
        (crop["path"], crop["file_sha256"], "official selected-track crop MKV"),
        (crop["frame_map_path"], crop["frame_map_sha256"], "selected-track source-clock map"),
        (crop["tracks_pickle_path"], crop["tracks_pickle_sha256"], "official selected-track geometry pickle"),
        (row["score"]["primary_syncnet_input"]["path"], row["score"]["primary_syncnet_input"]["file_sha256"], "official primary SyncNet input"),
    ]
    argv = render["syncnet_tracking"]["argv"]
    pipeline_source = Path(argv[1])
    ffmpeg = Path("/usr/bin/ffmpeg")
    expected_pipeline_sha = "3654a3cf22c74cb3071fca6fb97ca9fb7dd1a5943fe864169ff8c0b57eb80acf"
    expected_ffmpeg_sha = "36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45"
    source_specs.extend([
        (str(pipeline_source), expected_pipeline_sha, "pinned run_pipeline source used for JPEG replay"),
        (str(ffmpeg), expected_ffmpeg_sha, "pinned FFmpeg binary used for JPEG replay"),
    ])
    source_receipts = [pin_file(path, digest, label) for path, digest, label in source_specs]
    npz_receipt = verify_npz(row)
    ts_payload = json.loads(Path(render["timestamps_path"]).read_text(encoding="utf-8"))
    frame_map_payload = json.loads(Path(crop["frame_map_path"]).read_text(encoding="utf-8"))
    if frame_map_payload.get("schema_version") != "cross-tfg-track-frame-map-v2":
        raise AuditError("selected-track frame map schema differs from the frozen consumer contract")
    map_bindings = {
        "selected_crop_video_path": crop.get("path"),
        "selected_crop_video_file_sha256": crop.get("file_sha256"),
        "selected_crop_video_elementary_sha256": crop.get("video_elementary_sha256"),
        "selected_track_id": crop.get("selected_track_id"),
        "selected_track_index": crop.get("selected_track_index"),
        "render_video_path": render.get("muxed_video_path"),
        "render_video_file_sha256": render.get("muxed_video_sha256"),
        "render_video_elementary_sha256": render.get("video_stream_sha256"),
        "render_timestamps_sha256": render.get("timestamps_sha256"),
        "fps_num": render.get("fps_num"),
        "fps_den": render.get("fps_den"),
    }
    for key, expected in map_bindings.items():
        if frame_map_payload.get(key) != expected:
            raise AuditError(f"selected-track frame-map binding differs: {key}")
    frame_rows = frame_map_payload.get("frames")
    if not isinstance(frame_rows, list) or len(frame_rows) != int(crop["source_clock_frame_stop_exclusive"]) - int(crop["source_clock_frame_start"]):
        raise AuditError("selected-track map frame count differs from frozen crop clock")
    source_clock = []
    for local_index, item in enumerate(frame_rows):
        if item.get("crop_frame_index") != local_index:
            raise AuditError("selected-track crop frame indices are not contiguous")
        source_index = item.get("source_clock_frame_index")
        if not isinstance(source_index, int) or source_index != int(crop["source_clock_frame_start"]) + local_index:
            raise AuditError("selected-track source-clock indices differ from the frozen contiguous interval")
        source_clock.append(source_index)
    # The original run_pipeline pyframes directory is commonly cleaned after
    # scoring. Replay its two pinned FFmpeg extraction commands into this
    # fresh output tree; never claim the replayed files are historical inputs.
    replay = replay_run_pipeline_jpegs(row, out_dir)
    jpeg_frames = replay.pop("_jpeg_frames")
    jpeg_inventory = replay["replayed_jpeg_inventory"]
    jpeg_dir = Path(replay["replayed_jpeg_dir"])
    mp4_path = Path(render["generated_video_path"])
    mp4_frames = decode_video(mp4_path)
    mux_path = Path(render["muxed_video_path"])
    mux_frames = decode_video(mux_path)
    if len(mp4_frames) != int(render["frame_count"]):
        raise AuditError("generated MP4 decoded frame count differs from frozen result row")
    if len(mux_frames) != len(mp4_frames):
        raise AuditError("lossless natural-audio mux changed the W2L video frame count")
    mux_identical = all(np.array_equal(left, right) for left, right in zip(mp4_frames, mux_frames))
    if not mux_identical:
        raise AuditError("MP4 and lossless mux decoded frames differ; cannot attribute pipeline map")
    mapping = minimum_mae_monotone_map(mp4_frames, jpeg_frames)
    matrix_queries = np.asarray(npz_receipt["query_frame_index"], dtype=np.int64)
    matrix_tracks = np.asarray(npz_receipt["track_frame_index"], dtype=np.int64)
    if not np.array_equal(matrix_queries, matrix_tracks):
        raise AuditError("stored official matrix query and track frame indices differ")
    if matrix_queries.size != npz_receipt["query_count"]:
        raise AuditError("matrix query count mismatch")
    qmaps = []
    mapped_frames = mapping["frames"]
    for query_position, query in enumerate(matrix_queries.tolist()):
        if query < 0 or query + 4 >= len(mapped_frames):
            raise AuditError(f"official matrix query has no complete 5-frame JPEG window: q={query}")
        if query + 4 >= len(source_clock) or source_clock[query] != query:
            raise AuditError("matrix query does not use the expected zero-based source-clock frame")
        stamp_num = int(npz_receipt["query_timestamp_num"][query_position])
        stamp_den = int(npz_receipt["query_timestamp_den"][query_position])
        if stamp_den <= 0 or stamp_num * int(render["fps_num"]) != query * int(render["fps_den"]) * stamp_den:
            raise AuditError("stored SyncNet query timestamp differs from its frozen 25-fps source clock")
        window = mapped_frames[query:query + 5]
        qmaps.append({
            "query_frame_index_from_raw_matrix": int(query),
            "nominal_timestamp_num": int(query),
            "nominal_timestamp_den": int(render["fps_num"]),
            "jpeg_indices": list(range(query, query + 5)),
            "mapped_w2l_mp4_source_indices": [int(item["source_mp4_frame_index_zero_based"]) for item in window],
            "source_minus_query_offsets": [int(item["source_mp4_frame_index_zero_based"]) - (query + k) for k, item in enumerate(window)],
            "full_window_is_identity_map": all(int(item["source_mp4_frame_index_zero_based"]) == query + k for k, item in enumerate(window)),
        })

    frame_rows = frame_map_payload["frames"]
    origins: dict[str, int] = {}
    for item in frame_rows:
        origins[str(item.get("frame_origin"))] = origins.get(str(item.get("frame_origin")), 0) + 1
    pipeline_avi = Path(replay["replayed_video_avi"]["path"])
    pipeline_avi_info = {"file": replay["replayed_video_avi"],
                          "ffprobe": ffprobe_video(pipeline_avi, shutil.which("ffprobe") or "/usr/bin/ffprobe"),
                          "decoded_frame_count": len(decode_video(pipeline_avi))}
    replayed_track_crops = crop_replayed_frames_from_track(
        jpeg_frames, Path(crop["tracks_pickle_path"]), int(crop["selected_track_index"]))
    saved_crop_avi_frames = decode_video(Path(crop["crop_avi_path"]))
    saved_crop_mkv_frames = decode_video(Path(crop["path"]))
    if len(replayed_track_crops["crops"]) != len(saved_crop_avi_frames) or len(saved_crop_avi_frames) != len(saved_crop_mkv_frames):
        raise AuditError("replayed selected-track crop frame count differs from the frozen crop AVI/MKV")
    crop_frame_comparisons = []
    for index, (replayed_crop, avi_frame, mkv_frame) in enumerate(
            zip(replayed_track_crops["crops"], saved_crop_avi_frames, saved_crop_mkv_frames)):
        avi_mae = float(np.mean(np.abs(replayed_crop.astype(np.int16) - avi_frame.astype(np.int16)), dtype=np.float64))
        mkv_mae = float(np.mean(np.abs(replayed_crop.astype(np.int16) - mkv_frame.astype(np.int16)), dtype=np.float64))
        crop_frame_comparisons.append({
            "crop_local_frame_index": index,
            "replayed_crop_array_sha256": array_sha256(replayed_crop),
            "saved_crop_avi_decoded_sha256": array_sha256(avi_frame),
            "saved_crop_mkv_decoded_sha256": array_sha256(mkv_frame),
            "replayed_to_saved_crop_avi_mae_0_255": avi_mae,
            "replayed_to_saved_crop_mkv_mae_0_255": mkv_mae,
            "saved_crop_avi_and_mkv_pixel_identical": bool(np.array_equal(avi_frame, mkv_frame)),
        })
    ffprobe = shutil.which("ffprobe") or "/usr/bin/ffprobe"
    media_probes = {
        "generated_mp4": ffprobe_video(mp4_path, ffprobe),
        "mux_input": ffprobe_video(mux_path, ffprobe),
        "crop_avi": ffprobe_video(Path(crop["crop_avi_path"]), ffprobe),
        "selected_crop_mkv": ffprobe_video(Path(crop["path"]), ffprobe),
        "primary_syncnet_input": ffprobe_video(Path(row["score"]["primary_syncnet_input"]["path"]), ffprobe),
    }
    return {
        "schema": "cross_tfg_formal_w2l_delivery_clock_audit_v2",
        "status": "PASS_FORMAL_W2L_DELIVERY_CLOCK_AUDIT_REPLAYED_JPEGS",
        "scope": "one already-frozen W2L RMS dev delivery row; replayed only the pinned run_pipeline FFmpeg video/JPEG extraction; no render/model forward/scoring/mux performed",
        "sample_id": row["sample_id"],
        "speaker_id": row["speaker_id"],
        "row_id": row["row_id"],
        "method_id": row["method_id"],
        "model_id": row["model_id"],
        "seed": row["seed"],
        "frozen_row_line_sha256": bundle["producer_row_line_sha256"],
        "frozen_result_manifest_sha256": bundle["result_manifest_sha256"],
        "frozen_joint_manifest_sha256": bundle["joint_manifest_sha256"],
        "frozen_plan_sha256": bundle["plan_sha256"],
        "source_files": source_receipts,
        "matrix": npz_receipt,
        "video": {
            "generated_mp4_decoded_frame_count": len(mp4_frames),
            "natural_pcm_mux_decoded_frame_count": len(mux_frames),
            "mux_frames_pixel_identical_to_mp4_decode": mux_identical,
            "pipeline_jpeg_classification": replay["classification"],
            "pipeline_jpeg_directory": str(jpeg_dir),
            "pipeline_jpeg_count": len(jpeg_frames),
            "pipeline_jpeg_inventory": jpeg_inventory,
            "replayed_run_pipeline_commands": replay["commands"],
            "replayed_run_pipeline_source": replay["run_pipeline_source"],
            "replayed_ffmpeg_binary": replay["ffmpeg_binary"],
            "replayed_crop_geometry": replayed_track_crops["frames"],
            "replayed_crop_vs_saved_crop_frames": crop_frame_comparisons,
            "replayed_crop_vs_saved_crop_mae_mean_0_255": float(np.mean([item["replayed_to_saved_crop_avi_mae_0_255"] for item in crop_frame_comparisons], dtype=np.float64)),
            "replayed_crop_vs_saved_crop_mae_max_0_255": float(np.max([item["replayed_to_saved_crop_avi_mae_0_255"] for item in crop_frame_comparisons])),
            "official_frame_map_schema": frame_map_payload["schema_version"],
            "official_frame_map_row_count": len(frame_rows),
            "official_frame_map_origin_counts": origins,
            "official_frame_map_clock_start": frame_rows[0].get("source_clock_frame_index"),
            "official_frame_map_clock_end_exclusive": int(frame_rows[-1]["source_clock_frame_index"]) + 1,
            "ffprobe": media_probes,
            "run_pipeline_intermediate_avi": pipeline_avi_info,
            "minimum_mae_mp4_to_pipeline_jpeg_map": mapping,
            "matrix_query_windows": qmaps,
            "timestamp_sidecar_payload_sha256": sha256_bytes(Path(render["timestamps_path"]).read_bytes()),
            "timestamp_sidecar_frame_count": len(ts_payload) if isinstance(ts_payload, list) else None,
        },
        "runtime": {"python": os.sys.version, "numpy": np.__version__, "opencv": cv2.__version__, "ffprobe": ffprobe},
        "started_unix": bundle.get("started_unix"),
        "finished_unix": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    bundle_bytes = args.input_bundle.read_bytes()
    bundle = json.loads(bundle_bytes.decode("utf-8"))
    if bundle.get("schema") != "cross_tfg_formal_w2l_delivery_clock_input_v1":
        raise AuditError("input bundle schema mismatch")
    receipt = collect(bundle, args.output_dir)
    receipt["input_bundle"] = {"path": str(args.input_bundle), "bytes": len(bundle_bytes), "sha256": sha256_bytes(bundle_bytes)}
    receipt_path = args.output_dir / "formal_delivery_clock_receipt.json"
    receipt_tmp = receipt_path.with_suffix(".json.tmp")
    with receipt_tmp.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(receipt_tmp, receipt_path)
    print(json.dumps({"status": receipt["status"], "receipt_path": str(receipt_path),
                      "receipt_bytes": receipt_path.stat().st_size, "receipt_sha256": sha256_file(receipt_path),
                      "row_id": receipt["row_id"], "mp4_frames": receipt["video"]["generated_mp4_decoded_frame_count"],
                      "jpeg_frames": receipt["video"]["pipeline_jpeg_count"], "queries": receipt["matrix"]["query_count"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
