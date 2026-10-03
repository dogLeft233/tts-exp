#!/usr/bin/env python3
"""CPU-only, read-only scoring of eight frozen full512 RMS fit contexts.

Compares proxy visual embeddings, hard full512-to-official-224 pre-codec
re-crops, and saved official 224 visual embeddings against one shared,
saved original-PCM SyncNet audio embedding track and the same 27 source-clock
queries. No Wav2Lip forward, tracker run, training, dev, or confirmation read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SCHEMA = "cross_tfg_full512_precodec_shared_audio_parity_v2"
SAVED_MATRIX_MAX_ABS_ATOL = 3.0e-5
CPU_GPU_EMBEDDING_MAX_ABS_ATOL = 1.0e-3
CPU_GPU_EMBEDDING_MIN_MEAN_COSINE = 0.99999
SUMMARY_RAW_SHA256 = "9ae4cb8976461cee579b5606f8e20c09f60d8d5f24c6449fed6f55c90f307cfb"
SUMMARY_PAYLOAD_SHA256 = "5490b94e4e927ffab79c39ad6f48d26f720080c951542974588c0b9c956eebb4"
CONTEXT_NPZ_SHA256 = "6616e649412cdf5c5f78ce51fa7d8403364c56528f93777a2e35eb82fc941ec1"
SYNCNET_INSTANCE_SHA256 = "bd722bf844700d6a923dec2f10d7cacd65c0c6b22d09aa5d44d2509660e900d6"
SYNCNET_MODEL_SOURCE_SHA256 = "41b85aed0006546cb2d6dc9180c5bc3dac737ab80eff643022e397a349c30c22"
SYNCNET_CHECKPOINT_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
SYNCNET_PIPELINE_SHA256 = "3654a3cf22c74cb3071fca6fb97ca9fb7dd1a5943fe864169ff8c0b57eb80acf"
NATIVE_CROP_SOURCE_SHA256 = "1e89da5bdf2bee83938b717a084b55a699f1902925be3290d23317bfa562fa27"
WAV2LIP_INFERENCE_SHA256 = "99662bfce2fdaf17c1cff8e71977ba02cb80c684058cd114c4e0648b12059fc6"
LAG_RADIUS = 15
LAG_OFFSETS = np.arange(-LAG_RADIUS, LAG_RADIUS + 1, dtype=np.int32)
QUERY_COUNT = 27
FRAME_COUNT = 32
FPS = 25
EXPECTED_ROW_IDS = [
    "S0052_BAC009S0052W0121", "S0052_BAC009S0052W0122",
    "S0053_BAC009S0053W0121", "S0053_BAC009S0053W0122",
    "S0054_BAC009S0054W0121", "S0054_BAC009S0054W0122",
    "S0055_BAC009S0055W0121", "S0055_BAC009S0055W0122",
]


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256_bytes(raw)


def atomic_json(path: Path, payload: Any) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def require_file(path: Path, expected_sha: str, expected_size: int | None = None, label: str = "file") -> None:
    if not path.is_file():
        raise AuditError(f"{label} is missing: {path}")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise AuditError(f"{label} size mismatch: {path}")
    if sha256_file(path) != expected_sha:
        raise AuditError(f"{label} sha256 mismatch: {path}")


def finite_float32(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32 or (ndim is not None and array.ndim != ndim) or not np.isfinite(array).all():
        raise AuditError(f"{name} must be finite float32 (ndim={ndim}), got {array.dtype}/{array.shape}")
    return np.ascontiguousarray(array, dtype="<f4")


def metric_summary(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if matrix.dtype != np.float32 or matrix.shape != (QUERY_COUNT, len(LAG_OFFSETS)) or not np.isfinite(matrix).all():
        raise AuditError(f"distance matrix must be finite float32 27x31, got {matrix.dtype}/{matrix.shape}")
    # Match evaluate(): torch.stack(dists, dim=1).mean(dim=1).
    curve_t = torch.from_numpy(np.ascontiguousarray(matrix.T, dtype=np.float32)).mean(dim=1)
    median_t = curve_t.median()
    min_t, min_idx_t = curve_t.min(dim=0)
    zero_t = curve_t[LAG_RADIUS]
    curve = np.ascontiguousarray(curve_t.detach().cpu().numpy(), dtype="<f4")
    far = torch.from_numpy(np.abs(LAG_OFFSETS) >= 4)
    return curve, {
        "B_median": float(median_t.item()),
        "Dmin": float(min_t.item()),
        "D0": float(zero_t.item()),
        "Cbest": float((median_t - min_t).item()),
        "Canchor": float((median_t - zero_t).item()),
        "best_lag_frames": int(min_idx_t.item()) - LAG_RADIUS,
        "official_av_offset_frames": LAG_RADIUS - int(min_idx_t.item()),
        "far_lag_minus_D0_mean_abs_lag_ge_4": float((curve_t[far] - zero_t).mean().item()),
        "curve_float32": curve.tolist(),
    }


def distance_matrix(visual: np.ndarray, audio: np.ndarray, query_start_local: int) -> np.ndarray:
    visual = finite_float32(visual, "visual embedding", ndim=2)
    audio = finite_float32(audio, "audio embedding", ndim=2)
    if visual.shape != (QUERY_COUNT, 1024) or audio.shape[1] != 1024:
        raise AuditError(f"embedding dimensions mismatch: {visual.shape}/{audio.shape}")
    if query_start_local < LAG_RADIUS or query_start_local + QUERY_COUNT - 1 + LAG_RADIUS >= audio.shape[0]:
        raise AuditError("center queries do not have complete 31-lag audio support")
    visual_t, audio_t = torch.from_numpy(visual), torch.from_numpy(audio)
    rows = []
    for i in range(QUERY_COUNT):
        q = query_start_local + i
        window = audio_t[q - LAG_RADIUS : q + LAG_RADIUS + 1]
        if tuple(window.shape) != (31, 1024):
            raise AuditError("lag window is not the full 31-column audio support")
        repeated = visual_t[i : i + 1].expand(31, -1)
        rows.append(F.pairwise_distance(repeated, window).detach().cpu().numpy().astype("<f4", copy=True))
    return np.ascontiguousarray(np.stack(rows, axis=0), dtype="<f4")


def embed_32_frames(syncnet, frames_bgr_u8: np.ndarray) -> np.ndarray:
    frames = np.asarray(frames_bgr_u8)
    if frames.dtype != np.uint8 or frames.shape != (FRAME_COUNT, 224, 224, 3):
        raise AuditError(f"SyncNet frames must be uint8 [32,224,224,3], got {frames.dtype}/{frames.shape}")
    # evaluate(): BGR 0..255, [T,H,W,C] -> [1,C,T,H,W], then 5-frame windows.
    video = np.ascontiguousarray(frames.transpose(3, 0, 1, 2), dtype=np.float32)[None, ...]
    video_t = torch.from_numpy(video)
    model_input = torch.cat([video_t[:, :, i : i + 5] for i in range(QUERY_COUNT)], dim=0)
    with torch.inference_mode():
        embedding = syncnet.__S__.forward_lip(model_input)
    output = finite_float32(embedding.detach().cpu().contiguous().numpy(), "SyncNet visual embedding", ndim=2)
    if output.shape != (QUERY_COUNT, 1024):
        raise AuditError(f"SyncNet visual embedding shape mismatch: {output.shape}")
    return output


def embedding_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = finite_float32(left, "left embedding", 2).astype(np.float64)
    right = finite_float32(right, "right embedding", 2).astype(np.float64)
    if left.shape != right.shape:
        raise AuditError(f"embedding comparison shape mismatch: {left.shape}/{right.shape}")
    ln, rn = np.linalg.norm(left, axis=1), np.linalg.norm(right, axis=1)
    cos = np.sum(left * right, axis=1) / np.maximum(ln * rn, np.finfo(np.float64).tiny)
    delta = np.abs(left - right)
    return {
        "left_mean_l2_norm": float(ln.mean()),
        "right_mean_l2_norm": float(rn.mean()),
        "mean_cosine": float(cos.mean()),
        "min_cosine": float(cos.min()),
        "mean_angle_degrees": float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).mean()),
        "max_angle_degrees": float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))).max()),
        "max_abs_embedding_delta": float(delta.max(initial=0.0)),
        "mean_abs_embedding_delta": float(delta.mean()),
    }


def pixel_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left, right = np.asarray(left), np.asarray(right)
    if left.dtype != np.uint8 or right.dtype != np.uint8 or left.shape != right.shape:
        raise AuditError(f"pixel comparison requires same-shape uint8: {left.dtype}/{left.shape}, {right.dtype}/{right.shape}")
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return {
        "shape": list(left.shape),
        "mae_0_255": float(delta.astype(np.float64).mean()),
        "max_abs": int(delta.max(initial=0)),
        "different_pixel_fraction": float(np.count_nonzero(delta) / delta.size),
    }


def ffmpeg_jpeg_frames(ffmpeg: Path, crop_video: Path, temp_parent: Path, expected_count: int) -> np.ndarray:
    temp = Path(tempfile.mkdtemp(prefix="syncnet_jpeg_", dir=temp_parent))
    try:
        pattern = temp / "%06d.jpg"
        argv = [str(ffmpeg), "-y", "-loglevel", "error", "-i", str(crop_video), "-threads", "1", "-f", "image2", str(pattern)]
        proc = subprocess.run(argv, capture_output=True, check=False, timeout=180)
        if proc.returncode != 0:
            raise AuditError(f"official ffmpeg image2 extraction failed, exit={proc.returncode}")
        files = sorted(temp.glob("*.jpg"))
        if len(files) != expected_count:
            raise AuditError(f"selected crop JPEG count mismatch: {len(files)} != {expected_count}")
        images = []
        for path in files:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.dtype != np.uint8 or image.shape != (224, 224, 3):
                raise AuditError(f"ffmpeg image2 frame is not 224x224 BGR uint8: {path.name}")
            images.append(image)
        return np.ascontiguousarray(np.stack(images), dtype=np.uint8)
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def load_track(path: Path, expected_sha: str, expected_size: int, selected_index: int,
               expected_start: int, expected_stop: int, expected_frames_sha: str) -> dict[str, Any]:
    require_file(path, expected_sha, expected_size, "selected SyncNet tracks.pckl")
    import pickle
    with path.open("rb") as stream:
        tracks = pickle.load(stream)
    if not isinstance(tracks, list) or not 0 <= selected_index < len(tracks):
        raise AuditError("selected track index is missing from tracks.pckl")
    selected = tracks[selected_index]
    if not isinstance(selected, dict) or not isinstance(selected.get("track"), dict) or not isinstance(selected.get("proc_track"), dict):
        raise AuditError("selected tracks.pckl entry lacks track/proc_track mappings")
    frames = np.asarray(selected["track"].get("frame"), dtype=np.int64).reshape(-1)
    if frames.size == 0 or int(frames[0]) != expected_start or int(frames[-1]) + 1 != expected_stop:
        raise AuditError("selected source-frame bounds differ from parity summary")
    if frames.size > 1 and not np.array_equal(np.diff(frames), np.ones(frames.size - 1, dtype=np.int64)):
        raise AuditError("selected source frames are not contiguous")
    if sha256_bytes(np.ascontiguousarray(frames, dtype="<i8").tobytes()) != expected_frames_sha:
        raise AuditError("selected source-frame array SHA differs from parity summary")
    proc = selected["proc_track"]
    if any(np.asarray(proc.get(key)).reshape(-1).size != frames.size for key in ("x", "y", "s")):
        raise AuditError("selected proc_track x/y/s lengths differ from the frame map")
    return {"frames": frames, "proc_track": proc}


def run(args: argparse.Namespace) -> int:
    summary_path = Path(args.summary).resolve(strict=True)
    syncnet_root = Path(args.syncnet_root).resolve(strict=True)
    checkpoint = Path(args.syncnet_checkpoint).resolve(strict=True)
    repo_root = Path(args.repo_root).resolve(strict=True)
    crop_source = repo_root / "scripts" / "audio_enhancement" / "differentiable_native_crop.py"
    ffmpeg = Path(args.ffmpeg).resolve(strict=True)
    output, work = Path(args.output).resolve(), Path(args.work).resolve()
    if output.exists() or work.exists():
        raise AuditError("refusing to reuse an output/work directory")
    if sha256_file(summary_path) != SUMMARY_RAW_SHA256:
        raise AuditError("full512 parity summary raw SHA mismatch")
    if sha256_file(checkpoint) != SYNCNET_CHECKPOINT_SHA256:
        raise AuditError("SyncNet checkpoint SHA mismatch")
    if sha256_file(syncnet_root / "SyncNetInstance.py") != SYNCNET_INSTANCE_SHA256:
        raise AuditError("SyncNetInstance source SHA mismatch")
    if sha256_file(syncnet_root / "SyncNetModel.py") != SYNCNET_MODEL_SOURCE_SHA256:
        raise AuditError("SyncNetModel source SHA mismatch")
    if sha256_file(syncnet_root / "run_pipeline.py") != SYNCNET_PIPELINE_SHA256:
        raise AuditError("run_pipeline source SHA mismatch")
    if sha256_file(crop_source) != NATIVE_CROP_SOURCE_SHA256:
        raise AuditError("native crop helper source SHA mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("summary_payload_sha256") != SUMMARY_PAYLOAD_SHA256 or summary.get("status") != "COMPLETE":
        raise AuditError("full512 summary status/payload differs from frozen source")
    if summary.get("context_npz_sha256") != CONTEXT_NPZ_SHA256:
        raise AuditError("full512 summary context pin differs")
    if summary.get("syncnet_checkpoint_sha256") != SYNCNET_CHECKPOINT_SHA256 or summary.get("syncnet_instance_sha256") != SYNCNET_INSTANCE_SHA256:
        raise AuditError("full512 summary SyncNet pins differ")
    if summary.get("wav2lip_inference_sha256") != WAV2LIP_INFERENCE_SHA256:
        raise AuditError("full512 summary Wav2Lip inference pin differs")
    rows = summary.get("rows")
    if not isinstance(rows, list) or [row.get("sample_id") for row in rows] != EXPECTED_ROW_IDS:
        raise AuditError("source is not the exact eight historical fit contexts in frozen order")
    if any(row.get("split") != "fit" or row.get("method_id") != "rms_fixed" or not row.get("historical_exposed") for row in rows):
        raise AuditError("row split/method/exposure differs from the same-RMS historical-fit scope")

    input_seals: dict[str, dict[str, Any]] = {}
    resolved_rows = []
    for row in rows:
        sid, art = row["sample_id"], row["artifacts"]
        video, tracking = row["native_render_details"]["video"], row["native_render_details"]["tracking"]
        paths = {
            "parity_arrays": (Path(art["parity_arrays_path"]), art["parity_arrays_sha256"], art["parity_arrays_bytes"]),
            "tracks_pickle": (Path(tracking["tracks_pickle_path"]), tracking["tracks_pickle_sha256"], tracking["tracks_pickle_bytes"]),
            "selected_track_crop_video": (Path(video["selected_track_crop_path"]), video["selected_track_crop_sha256"], video["selected_track_crop_bytes"]),
            "native_direct_pcm16_wav": (Path(art["direct_pcm16_wav_path"]), art["direct_pcm16_wav_sha256"], art["direct_pcm16_wav_bytes"]),
            "native_direct_pcm16_mux": (Path(art["native_mux_path"]), art["native_mux_sha256"], art["native_mux_bytes"]),
        }
        for label, (path, expected_sha, expected_size) in paths.items():
            require_file(path, expected_sha, expected_size, f"{sid}/{label}")
            input_seals[f"{sid}/{label}"] = {"path": str(path), "bytes": int(expected_size), "sha256": expected_sha}
        resolved_rows.append({"row": row, "paths": paths})

    output.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(syncnet_root))
    sys.path.insert(0, str(repo_root))
    from SyncNetInstance import SyncNetInstance
    from scripts.audio_enhancement.differentiable_native_crop import crop_official_numpy, official_track_crop_geometry

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    scorer = SyncNetInstance(device="cpu")
    scorer.loadParameters(str(checkpoint))
    scorer.__S__.eval()
    results = []
    try:
        for resolved in resolved_rows:
            row = resolved["row"]
            sid = row["sample_id"]
            video_ref, tracking_ref = row["native_render_details"]["video"], row["native_render_details"]["tracking"]
            q0_source = int(row["center_query_start_source_frame"])
            track_start = int(video_ref["selected_track_start_source_frame"])
            q0_local = q0_source - track_start
            if q0_source != int(video_ref["center_query_start_source_clock_frame"]):
                raise AuditError(f"{sid}: source-clock q0 differs between summary fields")
            if q0_local < LAG_RADIUS or q0_local + QUERY_COUNT - 1 + LAG_RADIUS >= int(row["selected_track_query_count"]):
                raise AuditError(f"{sid}: center 27 queries lack complete 31-lag support")
            if int(row["center_query_count"]) != QUERY_COUNT or int(video_ref["center_query_start_local"]) != q0_local:
                raise AuditError(f"{sid}: local q0/query count differs from the frozen source clock")
            track = load_track(
                resolved["paths"]["tracks_pickle"][0], tracking_ref["tracks_pickle_sha256"], tracking_ref["tracks_pickle_bytes"],
                int(video_ref["selected_track_index"]), track_start, int(video_ref["selected_track_stop_source_frame_exclusive"]),
                video_ref["selected_track_source_frames_sha256"],
            )
            with np.load(resolved["paths"]["parity_arrays"][0], allow_pickle=False) as z:
                required = {
                    "proxy_distance_by_query_lag_float32", "native_distance_by_query_lag_float32",
                    "proxy_video_embedding_float32", "native_video_embedding_center_float32",
                    "proxy_audio_embedding_aligned_segment_float32", "native_audio_embedding_full_float32",
                    "proxy_center_frames_bgr_u8", "official_center_frames_bgr_u8",
                    "official_full512_center_generated_u8", "official_full512_center_decoded_u8",
                }
                missing = sorted(required - set(z.files))
                if missing:
                    raise AuditError(f"{sid}: source parity NPZ misses keys {missing}")
                saved_proxy_matrix = finite_float32(z["proxy_distance_by_query_lag_float32"], "saved proxy matrix", 2)
                saved_native_matrix = finite_float32(z["native_distance_by_query_lag_float32"], "saved native matrix", 2)
                proxy_visual = finite_float32(z["proxy_video_embedding_float32"], "proxy visual embedding", 2)
                saved_native_visual = finite_float32(z["native_video_embedding_center_float32"], "native saved visual embedding", 2)
                proxy_audio = finite_float32(z["proxy_audio_embedding_aligned_segment_float32"], "proxy aligned audio embedding", 2)
                native_audio = finite_float32(z["native_audio_embedding_full_float32"], "native audio embedding", 2)
                proxy_frames = np.asarray(z["proxy_center_frames_bgr_u8"])
                saved_native224 = np.asarray(z["official_center_frames_bgr_u8"])
                full512_generated = np.asarray(z["official_full512_center_generated_u8"])
                full512_decoded = np.asarray(z["official_full512_center_decoded_u8"])
            if saved_proxy_matrix.shape != (QUERY_COUNT, 31) or saved_native_matrix.shape != (QUERY_COUNT, 31):
                raise AuditError(f"{sid}: saved matrix shape differs from 27x31")
            if proxy_visual.shape != (QUERY_COUNT, 1024) or saved_native_visual.shape != (QUERY_COUNT, 1024):
                raise AuditError(f"{sid}: visual embedding shape differs from 27x1024")
            if proxy_audio.shape != native_audio.shape or proxy_audio.shape[1] != 1024:
                raise AuditError(f"{sid}: paired original-audio embedding support differs")
            if full512_generated.shape != (FRAME_COUNT, 512, 512, 3) or full512_generated.dtype != np.uint8:
                raise AuditError(f"{sid}: saved generated full512 pixels have wrong shape/dtype")
            if full512_decoded.shape != full512_generated.shape or full512_decoded.dtype != np.uint8:
                raise AuditError(f"{sid}: decoded full512 pixels have wrong shape/dtype")
            if saved_native224.shape != (FRAME_COUNT, 224, 224, 3) or saved_native224.dtype != np.uint8:
                raise AuditError(f"{sid}: saved official 224 crop pixels have wrong shape/dtype")
            if proxy_frames.ndim != 4 or proxy_frames.shape != saved_native224.shape or proxy_frames.dtype != np.uint8:
                raise AuditError(f"{sid}: saved proxy crop pixel stack has wrong shape/dtype")
            if not np.array_equal(np.asarray(row["lag_offsets_frames"], dtype=np.int32), LAG_OFFSETS):
                raise AuditError(f"{sid}: lag-column order differs from -15..+15")

            generated_crops, decoded_crops = [], []
            for j in range(FRAME_COUNT):
                xyxy, bsi = official_track_crop_geometry(track["proc_track"], q0_local + j)
                generated_crops.append(crop_official_numpy(full512_generated[j], xyxy, bsi))
                decoded_crops.append(crop_official_numpy(full512_decoded[j], xyxy, bsi))
            generated_crops = np.ascontiguousarray(np.stack(generated_crops), dtype=np.uint8)
            decoded_crops = np.ascontiguousarray(np.stack(decoded_crops), dtype=np.uint8)
            jpeg_all = ffmpeg_jpeg_frames(args.ffmpeg, resolved["paths"]["selected_track_crop_video"][0], work, int(video_ref["selected_track_frame_count"]))
            if jpeg_all.shape[0] != int(video_ref["selected_track_frame_count"]):
                raise AuditError(f"{sid}: official JPEG full track frame count differs")
            official_jpeg = np.ascontiguousarray(jpeg_all[q0_local : q0_local + FRAME_COUNT], dtype=np.uint8)
            if official_jpeg.shape != saved_native224.shape:
                raise AuditError(f"{sid}: official ffmpeg JPEG window shape differs from the saved 224 crop window")

            proxy_reembedded_visual = embed_32_frames(scorer, proxy_frames)
            saved_native224_reembedded_visual = embed_32_frames(scorer, saved_native224)
            precodec_visual = embed_32_frames(scorer, generated_crops)
            decoded_visual = embed_32_frames(scorer, decoded_crops)
            official_jpeg_visual = embed_32_frames(scorer, official_jpeg)
            proxy_shared = distance_matrix(proxy_visual, native_audio, q0_local)
            proxy_reembedded_shared = distance_matrix(proxy_reembedded_visual, native_audio, q0_local)
            saved_native224_reembedded_shared = distance_matrix(saved_native224_reembedded_visual, native_audio, q0_local)
            precodec_matrix = distance_matrix(precodec_visual, native_audio, q0_local)
            decoded_matrix = distance_matrix(decoded_visual, native_audio, q0_local)
            native_saved_visual_matrix = distance_matrix(saved_native_visual, native_audio, q0_local)
            native_jpeg_matrix = distance_matrix(official_jpeg_visual, native_audio, q0_local)
            proxy_saved_rebuilt = distance_matrix(proxy_visual, proxy_audio, q0_local)
            native_saved_rebuilt = distance_matrix(saved_native_visual, native_audio, q0_local)
            metric_matrices = {
                "proxy_saved_visual_shared_native_audio": proxy_shared,
                "proxy_reembedded_shared_native_audio": proxy_reembedded_shared,
                "hard_precodec_512_to_official224": precodec_matrix,
                "decoded512_then_hard224_pre_crop_codec": decoded_matrix,
                "official224_saved_visual_shared_native_audio": native_saved_visual_matrix,
                "official224_saved_cv2_reembedded_shared_native_audio": saved_native224_reembedded_shared,
                "official224_jpeg_recomputed_shared_native_audio": native_jpeg_matrix,
                "proxy_saved_audio_matrix_reconstruction": proxy_saved_rebuilt,
                "official_saved_matrix_reconstruction": native_saved_rebuilt,
            }
            matrix_metrics = {key: metric_summary(value)[1] for key, value in metric_matrices.items()}
            proxy_diff = np.abs(proxy_saved_rebuilt.astype(np.float64) - saved_proxy_matrix.astype(np.float64))
            native_diff = np.abs(native_saved_rebuilt.astype(np.float64) - saved_native_matrix.astype(np.float64))
            proxy_embedding_replay = embedding_comparison(proxy_reembedded_visual, proxy_visual)
            native_embedding_replay = embedding_comparison(official_jpeg_visual, saved_native_visual)
            saved_cv2_embedding_replay = embedding_comparison(saved_native224_reembedded_visual, saved_native_visual)
            feature_diff = np.abs(official_jpeg_visual.astype(np.float64) - saved_native_visual.astype(np.float64))
            control_checks = {
                "proxy_saved_matrix_reconstruction": {
                    "max_abs": float(proxy_diff.max(initial=0.0)),
                    "atol": SAVED_MATRIX_MAX_ABS_ATOL,
                    "passed": bool(proxy_diff.max(initial=0.0) <= SAVED_MATRIX_MAX_ABS_ATOL),
                },
                "official_saved_matrix_reconstruction": {
                    "max_abs": float(native_diff.max(initial=0.0)),
                    "atol": SAVED_MATRIX_MAX_ABS_ATOL,
                    "passed": bool(native_diff.max(initial=0.0) <= SAVED_MATRIX_MAX_ABS_ATOL),
                },
                "proxy_embedding_cpu_replay": {
                    **proxy_embedding_replay,
                    "max_abs_atol": CPU_GPU_EMBEDDING_MAX_ABS_ATOL,
                    "mean_cosine_min": CPU_GPU_EMBEDDING_MIN_MEAN_COSINE,
                    "passed": bool(proxy_embedding_replay["max_abs_embedding_delta"] <= CPU_GPU_EMBEDDING_MAX_ABS_ATOL and proxy_embedding_replay["mean_cosine"] >= CPU_GPU_EMBEDDING_MIN_MEAN_COSINE),
                },
                "official_jpeg_embedding_cpu_replay": {
                    **native_embedding_replay,
                    "max_abs_atol": CPU_GPU_EMBEDDING_MAX_ABS_ATOL,
                    "mean_cosine_min": CPU_GPU_EMBEDDING_MIN_MEAN_COSINE,
                    "passed": bool(native_embedding_replay["max_abs_embedding_delta"] <= CPU_GPU_EMBEDDING_MAX_ABS_ATOL and native_embedding_replay["mean_cosine"] >= CPU_GPU_EMBEDDING_MIN_MEAN_COSINE),
                },
            }
            row_control_pass = all(item["passed"] for item in control_checks.values())
            row_result = {
                "sample_id": sid,
                "speaker_id": row["speaker_id"],
                "scope": "historical_fit_only_engineering_parity",
                "control_status": "PASS" if row_control_pass else "CONTROL_FAILURE",
                "control_checks": control_checks,
                "method_id": "rms_fixed",
                "query_clock": {
                    "fps_num": FPS, "fps_den": 1,
                    "center_source_frame_start": q0_source,
                    "selected_track_source_frame_start": track_start,
                    "selected_track_local_query_start": q0_local,
                    "query_count": QUERY_COUNT, "precodec_frame_count": FRAME_COUNT,
                    "query_timestamp_num": [q0_source + i for i in range(QUERY_COUNT)],
                    "query_timestamp_den": FPS,
                    "lag_offsets_frames": LAG_OFFSETS.tolist(),
                    "all_31_lags_valid": True,
                    "score_audio_float32_sha256": row["syncnet_scored_natural_float32_sha256"],
                    "score_audio_pcm16_sha256": row["syncnet_scored_natural_pcm16_sha256"],
                    "selected_track_pcm16_sha256": row["native_render_details"]["pcm16"]["selected_track_pcm16_sha256"],
                },
                "source_bindings": {key: seal for key, seal in input_seals.items() if key.startswith(sid + "/")},
                "metrics": matrix_metrics,
                "pixels": {
                    "hard_precodec_512crop_vs_official224_saved": pixel_comparison(generated_crops, saved_native224),
                    "decoded512_recrop_vs_official224_saved": pixel_comparison(decoded_crops, saved_native224),
                    "hard_precodec_vs_decoded512_crop": pixel_comparison(generated_crops, decoded_crops),
                    "proxy_crop_vs_official224_saved": pixel_comparison(proxy_frames, saved_native224),
                    "official224_saved_cv2_decode_vs_ffmpeg_jpeg_readback": pixel_comparison(saved_native224, official_jpeg),
                },
                "embedding_alignment": {
                    "proxy_reembedded_vs_saved_proxy": proxy_embedding_replay,
                    "official_saved_cv2_reembedded_vs_saved": saved_cv2_embedding_replay,
                    "hard_precodec_vs_saved_official224": embedding_comparison(precodec_visual, saved_native_visual),
                    "decoded512_crop_vs_saved_official224": embedding_comparison(decoded_visual, saved_native_visual),
                    "proxy_vs_saved_official224": embedding_comparison(proxy_visual, saved_native_visual),
                    "official224_jpeg_recomputed_vs_saved": embedding_comparison(official_jpeg_visual, saved_native_visual),
                },
                "reconstruction_residuals": {
                    "proxy_recomputed_on_saved_audio_max_abs": float(proxy_diff.max(initial=0.0)),
                    "proxy_recomputed_on_saved_audio_rmse": float(np.sqrt(np.mean(np.square(proxy_diff)))) if proxy_diff.size else 0.0,
                    "official_matrix_recomputed_max_abs": float(native_diff.max(initial=0.0)),
                    "official_matrix_recomputed_rmse": float(np.sqrt(np.mean(np.square(native_diff)))) if native_diff.size else 0.0,
                    "official_jpeg_visual_vs_saved_embedding_max_abs": float(feature_diff.max(initial=0.0)),
                    "official_jpeg_visual_vs_saved_embedding_mean_abs": float(feature_diff.mean()),
                },
                "definitions": {
                    "hard_precodec_512_to_official224": "saved official generated uint8 512x512 frames, cropped by selected run_pipeline proc_track with the pinned hard OpenCV 224 crop helper; before full-frame and crop-video codecs",
                    "decoded512_then_hard224_pre_crop_codec": "saved DIVX-decoded 512x512 frames cropped by the same selected proc_track before crop AVI encode/decode",
                    "official224_saved_visual_shared_native_audio": "saved official evaluator visual embeddings from the selected XVID crop (ffmpeg image2 JPEG inputs), rescored against the same saved direct-original-PCM16 native audio embedding track",
                    "official224_saved_cv2_reembedded_shared_native_audio": "CPU SyncNet embeddings recomputed from OpenCV-decoded saved selected-crop frames, rescored against that same native audio embedding track",
                    "official224_jpeg_recomputed_shared_native_audio": "CPU SyncNet embeddings recomputed from the official ffmpeg image2 JPEG extraction and OpenCV readback, rescored against the same native audio embedding track",
                    "proxy_saved_visual_shared_native_audio": "saved proxy visual embeddings rescored against the same native audio embedding track",
                    "proxy_reembedded_shared_native_audio": "CPU SyncNet embeddings recomputed from saved proxy BGR frames, rescored against the same native audio embedding track",
                    "native_audio": "one unchanged official SyncNet audio embedding array from the exact selected-track direct slice of the original natural PCM16",
                    "offset": "official AV offset = negative best_lag_frames; no video/audio retiming is applied",
                },
            }
            npz_path = output / f"{sid}_precodec_parity_arrays.npz"
            np.savez_compressed(
                npz_path,
                hard_precodec_crop_bgr_u8=generated_crops,
                decoded512_recrop_bgr_u8=decoded_crops,
                official224_saved_crop_bgr_u8=saved_native224,
                official224_ffmpeg_jpeg_crop_bgr_u8=official_jpeg,
                proxy_saved_visual_embedding_float32=proxy_visual,
                proxy_reembedded_visual_embedding_float32=proxy_reembedded_visual,
                hard_precodec_visual_embedding_float32=precodec_visual,
                decoded512_visual_embedding_float32=decoded_visual,
                official224_saved_visual_embedding_float32=saved_native_visual,
                official224_saved_cv2_reembedded_visual_embedding_float32=saved_native224_reembedded_visual,
                official224_jpeg_recomputed_visual_embedding_float32=official_jpeg_visual,
                native_audio_embedding_full_float32=native_audio,
                proxy_saved_visual_shared_native_audio_distance_by_query_lag_float32=proxy_shared,
                proxy_reembedded_shared_native_audio_distance_by_query_lag_float32=proxy_reembedded_shared,
                hard_precodec_distance_by_query_lag_float32=precodec_matrix,
                decoded512_distance_by_query_lag_float32=decoded_matrix,
                official224_saved_visual_distance_by_query_lag_float32=native_saved_visual_matrix,
                official224_saved_cv2_reembedded_distance_by_query_lag_float32=saved_native224_reembedded_shared,
                official224_jpeg_recomputed_distance_by_query_lag_float32=native_jpeg_matrix,
                proxy_saved_audio_recomputed_distance_by_query_lag_float32=proxy_saved_rebuilt,
                official_saved_matrix_recomputed_distance_by_query_lag_float32=native_saved_rebuilt,
                lag_offsets_frames=LAG_OFFSETS,
            )
            row_result["derived_arrays"] = {"path": npz_path.name, "bytes": npz_path.stat().st_size, "sha256": sha256_file(npz_path)}
            atomic_json(output / f"{sid}_result.json", row_result)
            results.append(row_result)
            print(json.dumps({
                "sample_id": sid, "status": "SCORED",
                "precodec_Cbest": matrix_metrics["hard_precodec_512_to_official224"]["Cbest"],
                "official_Cbest": matrix_metrics["official224_saved_visual_shared_native_audio"]["Cbest"],
            }, sort_keys=True), flush=True)

        for key, seal in input_seals.items():
            require_file(Path(seal["path"]), seal["sha256"], seal["bytes"], f"post-run {key}")
        payload = {
            "schema": SCHEMA,
            "status": ("COMPLETE" if len(results) == 8 and all(r["control_status"] == "PASS" for r in results) else
                       "COMPLETE_WITH_CONTROL_FAILURE" if len(results) == 8 else "INCOMPLETE"),
            "control_gate": {
                "saved_matrix_reconstruction_max_abs_atol": SAVED_MATRIX_MAX_ABS_ATOL,
                "cpu_gpu_embedding_max_abs_atol": CPU_GPU_EMBEDDING_MAX_ABS_ATOL,
                "cpu_gpu_embedding_mean_cosine_min": CPU_GPU_EMBEDDING_MIN_MEAN_COSINE,
                "all_rows_control_pass": all(r["control_status"] == "PASS" for r in results),
            },
            "scope": "historical_fit_only_engineering_parity; no development or confirmation audio/effects; no Wav2Lip forward or training",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_summary_path": str(summary_path),
            "source_summary_raw_sha256": SUMMARY_RAW_SHA256,
            "source_summary_payload_sha256": SUMMARY_PAYLOAD_SHA256,
            "fit_context_npz_sha256": CONTEXT_NPZ_SHA256,
            "syncnet_checkpoint_sha256": SYNCNET_CHECKPOINT_SHA256,
            "syncnet_instance_sha256": SYNCNET_INSTANCE_SHA256,
            "syncnet_model_source_sha256": SYNCNET_MODEL_SOURCE_SHA256,
            "syncnet_pipeline_sha256": SYNCNET_PIPELINE_SHA256,
            "native_crop_source_sha256": NATIVE_CROP_SOURCE_SHA256,
            "torch_version": torch.__version__, "numpy_version": np.__version__, "opencv_version": cv2.__version__,
            "syncnet_device": "cpu", "ffmpeg_path": str(ffmpeg), "ffmpeg_sha256": sha256_file(ffmpeg),
            "row_count": len(results),
            "primary_branches": ["proxy_saved_visual_shared_native_audio", "proxy_reembedded_shared_native_audio", "hard_precodec_512_to_official224", "official224_saved_visual_shared_native_audio", "official224_jpeg_recomputed_shared_native_audio"],
            "secondary_controls": ["decoded512_then_hard224_pre_crop_codec", "official224_saved_cv2_reembedded_shared_native_audio", "proxy_saved_matrix_reconstruction", "official_saved_matrix_reconstruction"],
            "metric_definitions": {
                "curve": "float32 torch mean over the 27 q0-centered queries, per lag column, matching SyncNetInstance.evaluate stack(dim=1).mean(dim=1)",
                "Cbest": "B_median-Dmin", "Canchor": "B_median-D0",
                "far_lag_minus_D0_mean_abs_lag_ge_4": "mean(curve[|lag|>=4]-D0)",
                "offset": "official AV offset equals negative best_lag_frames; no retiming",
            },
            "input_files": input_seals,
            "rows": results,
        }
        payload["payload_sha256"] = canonical_sha(payload)
        atomic_json(output / "result_summary.json", payload)
        outputs = sorted(path for path in output.iterdir() if path.is_file() and path.name != "result_manifest.json")
        file_records = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in outputs]
        manifest = {
            "schema": "cross_tfg_full512_precodec_parity_output_manifest_v2",
            "status": payload["status"],
            "files": file_records,
            "payload_sha256": canonical_sha({"schema": "cross_tfg_full512_precodec_parity_output_manifest_v2", "files": file_records}),
        }
        atomic_json(output / "result_manifest.json", manifest)
        return 0 if payload["status"] == "COMPLETE" else 2
    except Exception as exc:
        if output.is_dir():
            atomic_json(output / "analysis_failure.json", {
                "schema": "cross_tfg_full512_precodec_parity_failure_v2", "status": "FAILED",
                "error_type": type(exc).__name__, "error": str(exc)[:2000],
                "completed_rows": [r["sample_id"] for r in results],
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_summary_raw_sha256": SUMMARY_RAW_SHA256,
            })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--syncnet-root", required=True)
    parser.add_argument("--syncnet-checkpoint", required=True)
    parser.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    parser.add_argument("--output", required=True)
    parser.add_argument("--work", required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)[:2000]}, sort_keys=True), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
