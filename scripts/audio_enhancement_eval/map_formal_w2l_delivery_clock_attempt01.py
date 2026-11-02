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
    argv = render["syncnet_tracking"]["argv"]
    data_dir = Path(require_argv_value(argv, "--data_dir"))
    reference = require_argv_value(argv, "--reference")
    jpeg_dir = data_dir / "pyframes" / reference
    jpeg_paths, jpeg_frames, jpeg_inventory = read_jpegs(jpeg_dir)
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
    pipeline_avi = data_dir / "pyavi" / f"{reference}.avi"
    pipeline_avi_info = None
    if pipeline_avi.is_file():
        pipeline_avi_pin = {"path": str(pipeline_avi), "bytes": pipeline_avi.stat().st_size, "sha256": sha256_file(pipeline_avi)}
        pipeline_avi_info = {"file": pipeline_avi_pin, "ffprobe": ffprobe_video(pipeline_avi, shutil.which("ffprobe") or "/usr/bin/ffprobe"),
                             "decoded_frame_count": len(decode_video(pipeline_avi))}
    ffprobe = shutil.which("ffprobe") or "/usr/bin/ffprobe"
    media_probes = {
        "generated_mp4": ffprobe_video(mp4_path, ffprobe),
        "mux_input": ffprobe_video(mux_path, ffprobe),
        "crop_avi": ffprobe_video(Path(crop["crop_avi_path"]), ffprobe),
        "selected_crop_mkv": ffprobe_video(Path(crop["path"]), ffprobe),
        "primary_syncnet_input": ffprobe_video(Path(row["score"]["primary_syncnet_input"]["path"]), ffprobe),
    }
    return {
        "schema": "cross_tfg_formal_w2l_delivery_clock_audit_v1",
        "status": "PASS_FORMAL_W2L_DELIVERY_CLOCK_AUDIT",
        "scope": "one already-frozen W2L RMS dev delivery row; no render/model forward/scoring/mux performed",
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
            "pipeline_jpeg_directory": str(jpeg_dir),
            "pipeline_jpeg_count": len(jpeg_paths),
            "pipeline_jpeg_inventory": jpeg_inventory,
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
