#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporal purification experiment for SyncDrift-protected videos.

Ported from nas-cch-code/voice_def/tools/purify_syncdrift.py
(SHA256 6cfbf5855b87ea97c34f690d15671e31ad97c185a2b2cf95124c6d9629a7f30e).
The offset search, smoothing, frame mapping, and rendering algorithms are unchanged.
The local SyncNet scoring entry point, default paths, and decoded-frame count
handling are adapted below. Scoring uses the NAS run_sync.sh default
min_track=100; offset estimation retains the script's min_track=40 default.

Correction sign convention used everywhere in this script:
  positive c means output_frame[n] = protected_frame[n + c_frames].

The score minimized during offset search is the mean L2 distance between
SyncNet audio embeddings at time t and lip embeddings sampled at t + c.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from scipy.io import wavfile

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional at runtime.
    plt = None

try:
    from scipy.interpolate import PchipInterpolator
except Exception:  # pragma: no cover
    PchipInterpolator = None

try:
    import python_speech_features
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "python_speech_features is required. Use the syncnet environment, e.g. "
        "/data/cuichenhao/miniconda3/envs/syncnet/bin/python"
    ) from exc


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SYNCNET_ROOT = REPO_ROOT / "third_party" / "syncnet_python"
DEFAULT_SYNCNET_MODEL = DEFAULT_SYNCNET_ROOT / "data" / "syncnet_v2.model"
EVAL_MIN_TRACK = 100


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int
    video_duration_sec: float
    audio_duration_sec: Optional[float]


@dataclass
class SyncNetCrop:
    crop_avi: Path
    track_start_frame: int
    track_end_frame: int
    crop_index: int


@dataclass
class Embeddings:
    audio: np.ndarray
    lip: np.ndarray
    crop: SyncNetCrop


@dataclass
class EstimateResult:
    frame_correction_ms: np.ndarray
    raw_frame_correction_ms: np.ndarray
    median_window_ms: np.ndarray
    mapped_indices: np.ndarray
    window_rows: List[Dict[str, object]]
    score_matrix: Optional[np.ndarray]
    candidate_ms: np.ndarray
    valid_window_ratio: float
    best_global_ms: Optional[float]


def run_cmd(
    cmd: Sequence[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(proc.stdout or "", encoding="utf-8")
    if check and proc.returncode != 0:
        tail = (proc.stdout or "")[-4000:]
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n{tail}"
        )
    return proc


def device_to_env(device: str) -> Dict[str, str]:
    env = os.environ.copy()
    if device.startswith("cuda"):
        parts = device.split(":", 1)
        if len(parts) == 2 and parts[1] != "":
            env["CUDA_VISIBLE_DEVICES"] = parts[1]
    return env


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "video"


def ffprobe_duration(path: Path, stream_selector: Optional[str] = None) -> Optional[float]:
    cmd = ["ffprobe", "-v", "error"]
    if stream_selector:
        cmd += ["-select_streams", stream_selector]
    cmd += ["-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def get_video_info(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    # Some containers report duration-derived frames that cannot be decoded.
    # The mapping must use the same frame count that rendering can actually read.
    frame_count = 0
    while cap.grab():
        frame_count += 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {path}")
    return VideoInfo(
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        video_duration_sec=frame_count / fps,
        audio_duration_sec=ffprobe_duration(path, "a:0"),
    )


def collect_videos(input_dir: Path, recursive: bool) -> List[Path]:
    it = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def ensure_syncnet_import(syncnet_root: Path) -> None:
    root = str(syncnet_root)
    if root not in sys.path:
        sys.path.insert(0, root)


def run_syncnet_pipeline(
    input_video: Path,
    work_dir: Path,
    syncnet_root: Path,
    device: str,
    reference: str,
    min_track: int,
) -> SyncNetCrop:
    data_dir = work_dir / "syncnet_pipeline"
    log_path = work_dir / "logs" / "syncnet_pipeline.log"
    data_dir.mkdir(parents=True, exist_ok=True)
    env = device_to_env(device)
    cmd = [
        sys.executable,
        "run_pipeline.py",
        "--videofile",
        str(input_video),
        "--reference",
        reference,
        "--data_dir",
        str(data_dir),
        "--min_track",
        str(min_track),
        "--overwrite",
    ]
    run_cmd(cmd, cwd=syncnet_root, env=env, log_path=log_path)

    crop_dir = data_dir / "pycrop" / reference
    crops = sorted(crop_dir.glob("0*.avi"))
    if not crops:
        raise RuntimeError(f"SyncNet pipeline did not produce face crops. See {log_path}")

    tracks_path = data_dir / "pywork" / reference / "tracks.pckl"
    if not tracks_path.exists():
        raise RuntimeError(f"Missing SyncNet track file: {tracks_path}")
    with tracks_path.open("rb") as f:
        tracks = pickle.load(f)

    candidates: List[Tuple[int, Path, int, int, int]] = []
    for idx, crop in enumerate(crops):
        start = 0
        end = -1
        if idx < len(tracks):
            frames = np.asarray(tracks[idx]["track"]["frame"], dtype=np.int64)
            if frames.size:
                start = int(frames[0])
                end = int(frames[-1])
        cap = cv2.VideoCapture(str(crop))
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        candidates.append((nframes, crop, start, end, idx))

    nframes, crop, start, end, idx = max(candidates, key=lambda item: item[0])
    if nframes <= 5:
        raise RuntimeError(f"Chosen SyncNet crop is too short: {crop}")
    if end < start:
        end = start + nframes - 1
    return SyncNetCrop(crop_avi=crop, track_start_frame=start, track_end_frame=end, crop_index=idx)


def load_syncnet_model(syncnet_root: Path, model_path: Path, device: str) -> torch.nn.Module:
    ensure_syncnet_import(syncnet_root)
    from SyncNetModel import S  # type: ignore

    torch_device = torch.device(device)
    model = S(num_layers_in_fc_layers=1024).to(torch_device)
    state = torch.load(str(model_path), map_location=torch_device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def extract_wav(video_path: Path, wav_path: Path) -> None:
    run_cmd(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-ac",
            "1",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            str(wav_path),
        ]
    )


def read_video_frames_224(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[0] != 224 or frame.shape[1] != 224:
            frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    cap.release()
    if len(frames) < 6:
        raise RuntimeError(f"Too few frames for SyncNet embedding: {path}")
    return np.asarray(frames, dtype=np.float32)


def extract_embeddings(
    crop: SyncNetCrop,
    model: torch.nn.Module,
    device: str,
    batch_size: int,
    work_dir: Path,
) -> Embeddings:
    frames = read_video_frames_224(crop.crop_avi)
    wav_path = work_dir / "syncnet_crop_audio.wav"
    extract_wav(crop.crop_avi, wav_path)
    sample_rate, audio = wavfile.read(str(wav_path))
    if sample_rate != 16000:
        raise RuntimeError(f"Expected 16 kHz audio, got {sample_rate}: {wav_path}")

    mfcc_iter = zip(*python_speech_features.mfcc(audio, sample_rate))
    mfcc = np.stack([np.asarray(x) for x in mfcc_iter]).astype(np.float32)
    min_length = min(len(frames), int(math.floor(len(audio) / 640)))
    lastframe = min_length - 5
    if lastframe < 8:
        raise RuntimeError(
            f"Not enough overlapping audio/video embeddings: frames={len(frames)} audio={len(audio)}"
        )

    im = np.expand_dims(np.stack(frames, axis=3), axis=0)
    im = np.transpose(im, (0, 3, 4, 1, 2))
    imtv = torch.from_numpy(im.astype(np.float32))

    cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
    cct = torch.from_numpy(cc.astype(np.float32))

    torch_device = torch.device(device)
    lip_feats: List[torch.Tensor] = []
    aud_feats: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, lastframe, batch_size):
            stop = min(lastframe, start + batch_size)
            im_batch = [imtv[:, :, vframe : vframe + 5, :, :] for vframe in range(start, stop)]
            im_in = torch.cat(im_batch, 0).to(torch_device)
            lip_feats.append(model.forward_lip(im_in).detach().cpu())

            cc_batch = [cct[:, :, :, vframe * 4 : vframe * 4 + 20] for vframe in range(start, stop)]
            cc_in = torch.cat(cc_batch, 0).to(torch_device)
            aud_feats.append(model.forward_aud(cc_in).detach().cpu())

    lip = torch.cat(lip_feats, 0).numpy()
    aud = torch.cat(aud_feats, 0).numpy()
    return Embeddings(audio=aud, lip=lip, crop=crop)


def candidate_offsets(fps: float, search_min_ms: float, search_max_ms: float) -> Tuple[np.ndarray, np.ndarray]:
    min_frames = int(math.ceil(search_min_ms * fps / 1000.0))
    max_frames = int(math.floor(search_max_ms * fps / 1000.0))
    if min_frames > max_frames:
        raise ValueError("Search range is smaller than one frame step.")
    frames = np.arange(min_frames, max_frames + 1, dtype=np.int64)
    return frames, frames.astype(np.float64) * 1000.0 / fps


def score_candidate(
    audio: np.ndarray,
    lip: np.ndarray,
    start: int,
    stop: int,
    c_frames: int,
    min_valid_feats: int,
) -> Tuple[float, int]:
    idx = np.arange(start, stop, dtype=np.int64)
    shifted = idx + int(c_frames)
    valid = (idx >= 0) & (idx < len(audio)) & (shifted >= 0) & (shifted < len(lip))
    if int(valid.sum()) < min_valid_feats:
        return float("nan"), int(valid.sum())
    diff = audio[idx[valid]] - lip[shifted[valid]]
    return float(np.linalg.norm(diff, axis=1).mean()), int(valid.sum())


def best_offset_for_range(
    audio: np.ndarray,
    lip: np.ndarray,
    start: int,
    stop: int,
    cand_frames: np.ndarray,
    min_valid_feats: int,
    flat_score_epsilon: float,
) -> Tuple[float, float, float, float, bool, np.ndarray]:
    scores = []
    valid_counts = []
    for c in cand_frames:
        score, nvalid = score_candidate(audio, lip, start, stop, int(c), min_valid_feats)
        scores.append(score)
        valid_counts.append(nvalid)
    scores_np = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores_np)
    if finite.sum() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), False, scores_np
    finite_scores = scores_np[finite]
    order = np.argsort(finite_scores)
    best_score = float(finite_scores[order[0]])
    second = float(finite_scores[order[1]]) if len(order) > 1 else float("nan")
    gap = second - best_score if np.isfinite(second) else float("nan")
    best_finite_indices = np.where(finite)[0]
    best_idx = int(best_finite_indices[order[0]])
    best_c = float(cand_frames[best_idx])
    spread = float(np.nanmax(scores_np) - np.nanmin(scores_np))
    valid = bool(np.isfinite(best_score) and spread >= flat_score_epsilon)
    return best_c, best_score, second, gap, valid, scores_np


def median_filter_valid(values: np.ndarray, valid: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return values.copy()
    if size % 2 == 0:
        size += 1
    radius = size // 2
    out = np.full_like(values, np.nan, dtype=np.float64)
    for i in range(len(values)):
        if not valid[i]:
            continue
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        local = values[lo:hi][valid[lo:hi]]
        if local.size:
            out[i] = float(np.median(local))
    return out


def interpolate_to_frames(
    centers_sec: np.ndarray,
    values_ms: np.ndarray,
    frame_times_sec: np.ndarray,
) -> np.ndarray:
    good = np.isfinite(values_ms)
    if good.sum() == 0:
        return np.zeros_like(frame_times_sec, dtype=np.float64)
    x = centers_sec[good]
    y = values_ms[good]
    if good.sum() == 1:
        return np.full_like(frame_times_sec, float(y[0]), dtype=np.float64)

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if PchipInterpolator is not None and len(np.unique(x)) >= 2:
        fn = PchipInterpolator(x, y, extrapolate=False)
        out = np.asarray(fn(frame_times_sec), dtype=np.float64)
        left = frame_times_sec < x[0]
        right = frame_times_sec > x[-1]
        out[left] = y[0]
        out[right] = y[-1]
        middle_nan = ~np.isfinite(out)
        if middle_nan.any():
            out[middle_nan] = np.interp(frame_times_sec[middle_nan], x, y)
        return out
    return np.interp(frame_times_sec, x, y, left=y[0], right=y[-1])


def build_mapped_indices(correction_ms: np.ndarray, fps: float, num_frames: int) -> np.ndarray:
    k = np.rint(correction_ms * fps / 1000.0).astype(np.int64)
    raw = np.arange(num_frames, dtype=np.int64) + k
    raw = np.clip(raw, 0, num_frames - 1)
    mapped = raw.copy()
    for i in range(1, len(mapped)):
        if mapped[i] < mapped[i - 1]:
            mapped[i] = mapped[i - 1]
    return np.clip(mapped, 0, num_frames - 1)


def estimate_correction(
    embeddings: Embeddings,
    info: VideoInfo,
    mode: str,
    window_sec: float,
    stride_sec: float,
    search_min_ms: float,
    search_max_ms: float,
    median_filter_size: int,
    min_valid_feats: int,
    flat_score_epsilon: float,
) -> EstimateResult:
    audio = embeddings.audio
    lip = embeddings.lip
    cand_frames, cand_ms = candidate_offsets(25.0, search_min_ms, search_max_ms)
    frame_times = np.arange(info.frame_count, dtype=np.float64) / info.fps

    if mode == "global":
        best_c, best_score, second, gap, valid, scores = best_offset_for_range(
            audio, lip, 0, min(len(audio), len(lip)), cand_frames, min_valid_feats, flat_score_epsilon
        )
        best_ms = 0.0 if not valid else float(best_c * 1000.0 / 25.0)
        correction = np.full(info.frame_count, best_ms, dtype=np.float64)
        mapped = build_mapped_indices(correction, info.fps, info.frame_count)
        row = {
            "window_index": 0,
            "center_time_sec": (embeddings.crop.track_start_frame + embeddings.crop.track_end_frame) / 2.0 / 25.0,
            "estimated_correction_ms": best_ms,
            "best_score": best_score,
            "second_best_score": second,
            "score_gap": gap,
            "valid_window": bool(valid),
        }
        return EstimateResult(
            frame_correction_ms=correction,
            raw_frame_correction_ms=correction.copy(),
            median_window_ms=np.asarray([best_ms], dtype=np.float64),
            mapped_indices=mapped,
            window_rows=[row],
            score_matrix=np.asarray([scores], dtype=np.float64),
            candidate_ms=cand_ms,
            valid_window_ratio=1.0 if valid else 0.0,
            best_global_ms=best_ms,
        )

    win = max(int(round(window_sec * 25.0)), min_valid_feats)
    stride = max(int(round(stride_sec * 25.0)), 1)
    nfeat = min(len(audio), len(lip))
    starts = list(range(0, max(1, nfeat - win + 1), stride))
    if starts and starts[-1] + win < nfeat:
        starts.append(max(0, nfeat - win))
    if not starts:
        starts = [0]

    raw_ms: List[float] = []
    centers_sec: List[float] = []
    rows: List[Dict[str, object]] = []
    matrices: List[np.ndarray] = []
    valids: List[bool] = []
    track_start_sec = embeddings.crop.track_start_frame / 25.0

    for wi, start in enumerate(starts):
        stop = min(nfeat, start + win)
        center_feat = 0.5 * (start + stop - 1)
        center_time = track_start_sec + center_feat / 25.0
        best_c, best_score, second, gap, valid, scores = best_offset_for_range(
            audio, lip, start, stop, cand_frames, min_valid_feats, flat_score_epsilon
        )
        est_ms = float(best_c * 1000.0 / 25.0) if valid else float("nan")
        raw_ms.append(est_ms)
        centers_sec.append(center_time)
        matrices.append(scores)
        valids.append(valid)
        rows.append(
            {
                "window_index": wi,
                "center_time_sec": center_time,
                "estimated_correction_ms": est_ms,
                "best_score": best_score,
                "second_best_score": second,
                "score_gap": gap,
                "valid_window": bool(valid),
            }
        )

    raw_np = np.asarray(raw_ms, dtype=np.float64)
    valid_np = np.asarray(valids, dtype=bool)
    centers_np = np.asarray(centers_sec, dtype=np.float64)
    median_np = median_filter_valid(raw_np, valid_np, median_filter_size)
    correction = interpolate_to_frames(centers_np, median_np, frame_times)
    mapped = build_mapped_indices(correction, info.fps, info.frame_count)
    return EstimateResult(
        frame_correction_ms=correction,
        raw_frame_correction_ms=interpolate_to_frames(centers_np, raw_np, frame_times),
        median_window_ms=median_np,
        mapped_indices=mapped,
        window_rows=rows,
        score_matrix=np.stack(matrices, axis=0) if matrices else None,
        candidate_ms=cand_ms,
        valid_window_ratio=float(valid_np.mean()) if valid_np.size else 0.0,
        best_global_ms=None,
    )


def render_purified_video(
    input_video: Path,
    output_video: Path,
    info: VideoInfo,
    mapped_indices: np.ndarray,
    work_dir: Path,
) -> None:
    if input_video.resolve() == output_video.resolve():
        raise ValueError("Refusing to overwrite input video.")
    output_video.parent.mkdir(parents=True, exist_ok=True)
    temp_video = work_dir / "purified_video_only.mp4"

    cap = cv2.VideoCapture(str(input_video))
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if len(frames) != info.frame_count:
        raise RuntimeError(f"Frame read mismatch: metadata={info.frame_count}, decoded={len(frames)}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_video), fourcc, info.fps, (info.width, info.height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {temp_video}")
    for idx in mapped_indices:
        writer.write(frames[int(idx)])
    writer.release()

    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(temp_video),
        "-i",
        str(input_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    run_cmd(cmd, log_path=work_dir / "logs" / "ffmpeg_mux.log")

    out_info = get_video_info(output_video)
    if out_info.frame_count != info.frame_count:
        raise RuntimeError(f"Output frame count changed: {out_info.frame_count} != {info.frame_count}")
    out_audio = out_info.audio_duration_sec
    if out_audio is not None:
        diff = abs(out_audio - out_info.video_duration_sec)
        if diff > (1.0 / info.fps + 1e-3):
            raise RuntimeError(
                f"Output A/V duration differs by {diff:.4f}s, more than one frame at {info.fps:.3f} fps."
            )


def link_or_copy(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def write_estimate_csv(path: Path, info: VideoInfo, estimate: EstimateResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "frame_index",
            "timestamp_sec",
            "raw_correction_ms",
            "smoothed_correction_ms",
            "mapped_frame_index",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(info.frame_count):
            writer.writerow(
                {
                    "frame_index": idx,
                    "timestamp_sec": idx / info.fps,
                    "raw_correction_ms": float(estimate.raw_frame_correction_ms[idx]),
                    "smoothed_correction_ms": float(estimate.frame_correction_ms[idx]),
                    "mapped_frame_index": int(estimate.mapped_indices[idx]),
                }
            )


def write_window_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_curve(path: Path, info: VideoInfo, estimate: EstimateResult) -> None:
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_times = np.arange(info.frame_count, dtype=np.float64) / info.fps
    fig, ax = plt.subplots(figsize=(10, 4))
    centers = []
    raw = []
    med = []
    for i, row in enumerate(estimate.window_rows):
        centers.append(float(row["center_time_sec"]))
        raw.append(float(row["estimated_correction_ms"]))
        if i < len(estimate.median_window_ms):
            med.append(float(estimate.median_window_ms[i]))
    if centers:
        ax.scatter(centers, raw, s=18, label="raw window estimate")
        ax.plot(centers, med, marker="o", linewidth=1.2, label="median filtered")
    ax.plot(frame_times, estimate.frame_correction_ms, linewidth=1.5, label="per-frame PCHIP")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("time (sec)")
    ax.set_ylabel("correction (ms)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_syncnet_scores(text: str) -> Tuple[float, float]:
    # The local run_syncnet.py prints the same LSE-D/LSE-C quantities by name.
    distances = re.findall(r"Min dist:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))", text)
    confidences = re.findall(r"Confidence:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))", text)
    if distances and confidences:
        return float(distances[-1]), float(confidences[-1])
    pat = re.compile(
        r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
        re.M,
    )
    pairs = [(float(a), float(b)) for a, b in pat.findall(text)]
    if not pairs:
        raise RuntimeError(f"Cannot parse SyncNet LSE-D/LSE-C from output:\n{text[-2000:]}")
    return pairs[-1]


def evaluate_lse(
    video_path: Path,
    tag: str,
    save_dir: Path,
    syncnet_root: Path,
    device: str,
    syncnet_model: Path,
) -> Tuple[float, float]:
    eval_dir = save_dir / "syncnet_eval_inputs"
    eval_dir.mkdir(parents=True, exist_ok=True)
    staged = eval_dir / f"{safe_name(tag)}.mp4"
    shutil.copy2(video_path, staged)
    env = device_to_env(device)
    log_path = save_dir / "logs" / f"syncnet_eval_{safe_name(tag)}.log"
    reference = safe_name(tag)
    run_syncnet_pipeline(staged, eval_dir / reference, syncnet_root, device, reference, EVAL_MIN_TRACK)
    proc = run_cmd(
        [
            sys.executable, "run_syncnet.py", "--videofile", str(staged),
            "--reference", reference, "--data_dir", str(eval_dir / reference / "syncnet_pipeline"),
            "--initial_model", str(syncnet_model),
        ],
        cwd=syncnet_root,
        env=env,
        log_path=log_path,
    )
    return parse_syncnet_scores(proc.stdout or "")


def repeated_and_skipped_ratios(mapped: np.ndarray) -> Tuple[float, float]:
    if len(mapped) <= 1:
        return 0.0, 0.0
    diffs = np.diff(mapped)
    return float(np.mean(diffs == 0)), float(np.mean(diffs > 1))


def process_one_video(
    input_video: Path,
    output_video: Path,
    save_dir: Path,
    args: argparse.Namespace,
    model: torch.nn.Module,
) -> Dict[str, object]:
    save_dir.mkdir(parents=True, exist_ok=True)
    info = get_video_info(input_video)
    print(f"[video] {input_video}")
    print(f"[meta] fps={info.fps:.3f} frames={info.frame_count} size={info.width}x{info.height}")

    reference = safe_name(input_video.stem)[:80] or "syncdrift"
    crop = run_syncnet_pipeline(
        input_video=input_video,
        work_dir=save_dir,
        syncnet_root=Path(args.syncnet_root),
        device=args.device,
        reference=reference,
        min_track=args.min_track,
    )
    embeddings = extract_embeddings(
        crop=crop,
        model=model,
        device=args.device,
        batch_size=args.batch_size,
        work_dir=save_dir,
    )
    estimate = estimate_correction(
        embeddings=embeddings,
        info=info,
        mode=args.mode,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        search_min_ms=args.search_min_ms,
        search_max_ms=args.search_max_ms,
        median_filter_size=args.median_filter_size,
        min_valid_feats=args.min_valid_feats,
        flat_score_epsilon=args.flat_score_epsilon,
    )

    render_purified_video(input_video, output_video, info, estimate.mapped_indices, save_dir)
    link_or_copy(output_video, save_dir / "purified.mp4")

    write_estimate_csv(save_dir / "estimated_correction.csv", info, estimate)
    write_window_csv(save_dir / "window_estimates.csv", estimate.window_rows)
    np.save(save_dir / "estimated_correction.npy", estimate.frame_correction_ms)
    np.save(save_dir / "candidate_offsets_ms.npy", estimate.candidate_ms)
    if estimate.score_matrix is not None:
        np.save(save_dir / "offset_score_matrix.npy", estimate.score_matrix)
    plot_curve(save_dir / "correction_curve.png", info, estimate)

    protected_lse_d, protected_lse_c = evaluate_lse(
        input_video, "protected_" + input_video.stem, save_dir, Path(args.syncnet_root), args.device,
        Path(args.syncnet_model),
    )
    purified_lse_d, purified_lse_c = evaluate_lse(
        output_video, "purified_" + input_video.stem, save_dir, Path(args.syncnet_root), args.device,
        Path(args.syncnet_model),
    )

    repeated_ratio, skipped_ratio = repeated_and_skipped_ratios(estimate.mapped_indices)
    result = {
        "input_video": str(input_video),
        "output_video": str(output_video),
        "mode": args.mode,
        "protected_lse_c": protected_lse_c,
        "protected_lse_d": protected_lse_d,
        "purified_lse_c": purified_lse_c,
        "purified_lse_d": purified_lse_d,
        "recovery_c": purified_lse_c - protected_lse_c,
        "recovery_d": protected_lse_d - purified_lse_d,
        "mean_abs_correction_ms": float(np.mean(np.abs(estimate.frame_correction_ms))),
        "max_abs_correction_ms": float(np.max(np.abs(estimate.frame_correction_ms))),
        "valid_window_ratio": estimate.valid_window_ratio,
        "repeated_frame_ratio": repeated_ratio,
        "skipped_frame_ratio": skipped_ratio,
        "best_global_correction_ms": estimate.best_global_ms,
        "syncnet_crop_avi": str(crop.crop_avi),
        "syncnet_track_start_frame": crop.track_start_frame,
        "syncnet_track_end_frame": crop.track_end_frame,
    }
    (save_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[result] "
        f"LSE-C {protected_lse_c:.4f}->{purified_lse_c:.4f}, "
        f"LSE-D {protected_lse_d:.4f}->{purified_lse_d:.4f}"
    )
    return result


def build_jobs(args: argparse.Namespace) -> List[Tuple[Path, Path, Path]]:
    save_root = Path(args.save_dir).resolve()
    jobs: List[Tuple[Path, Path, Path]] = []
    if args.input_video:
        inp = Path(args.input_video).resolve()
        if args.output_video:
            out = Path(args.output_video).resolve()
        else:
            out_root = Path(args.output_dir).resolve() if args.output_dir else save_root
            out = out_root / f"{inp.stem}_purified_{args.mode}{inp.suffix}"
        video_save = save_root
        jobs.append((inp, out, video_save))
        return jobs

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    videos = collect_videos(input_dir, bool(args.recursive))
    if not videos:
        raise RuntimeError(f"No videos found under {input_dir}")
    for inp in videos:
        rel = inp.relative_to(input_dir)
        out = output_dir / rel.with_name(f"{rel.stem}_purified_{args.mode}{rel.suffix}")
        video_save = save_root / rel.with_suffix("")
        jobs.append((inp, out, video_save))
    return jobs


def write_summary(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple temporal purification for SyncDrift videos.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_video", type=str, help="Single protected input video.")
    src.add_argument("--input_dir", type=str, help="Directory of protected videos.")
    parser.add_argument("--output_video", type=str, help="Output video for --input_video.")
    parser.add_argument("--output_dir", type=str, help="Output directory for --input_dir.")
    parser.add_argument("--recursive", action="store_true", help="Recursively process mp4 files.")
    parser.add_argument("--mode", choices=["global", "window"], default="window")
    parser.add_argument("--window_sec", type=float, default=5.0)
    parser.add_argument("--stride_sec", type=float, default=1.0)
    parser.add_argument("--search_min_ms", type=float, default=-240.0)
    parser.add_argument("--search_max_ms", type=float, default=240.0)
    parser.add_argument("--median_filter_size", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--syncnet_root", type=str, default=str(DEFAULT_SYNCNET_ROOT))
    parser.add_argument("--syncnet_model", type=str, default=str(DEFAULT_SYNCNET_MODEL))
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--min_track", type=int, default=40)
    parser.add_argument("--min_valid_feats", type=int, default=8)
    parser.add_argument("--flat_score_epsilon", type=float, default=1e-4)
    parser.add_argument("--continue_on_error", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.input_dir and not args.output_dir:
        raise ValueError("--output_dir is required with --input_dir")
    if args.output_video and not args.input_video:
        raise ValueError("--output_video can only be used with --input_video")
    syncnet_root = Path(args.syncnet_root)
    if not syncnet_root.exists():
        raise FileNotFoundError(f"syncnet_root not found: {syncnet_root}")
    if not (syncnet_root / "run_pipeline.py").exists():
        raise FileNotFoundError(f"run_pipeline.py not found under {syncnet_root}")
    if not (syncnet_root / "run_syncnet.py").exists():
        raise FileNotFoundError(f"run_syncnet.py not found under {syncnet_root}")
    if not Path(args.syncnet_model).exists():
        raise FileNotFoundError(f"syncnet_model not found: {args.syncnet_model}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false.")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    save_root = Path(args.save_dir).resolve()
    save_root.mkdir(parents=True, exist_ok=True)

    model = load_syncnet_model(Path(args.syncnet_root), Path(args.syncnet_model), args.device)
    jobs = build_jobs(args)
    print(f"[jobs] {len(jobs)} video(s)")

    rows: List[Dict[str, object]] = []
    failed: List[Dict[str, object]] = []
    for idx, (inp, out, video_save) in enumerate(jobs, start=1):
        print(f"\n[{idx}/{len(jobs)}] {inp}")
        try:
            result = process_one_video(inp, out, video_save, args, model)
            rows.append(result)
            write_summary(save_root / "purification_summary.csv", rows)
        except Exception as exc:
            err = {"input_video": str(inp), "output_video": str(out), "error": str(exc)}
            failed.append(err)
            write_summary(save_root / "purification_failed.csv", failed)
            print(f"[error] {inp}: {exc}")
            if not args.continue_on_error:
                raise

    write_summary(save_root / "purification_summary.csv", rows)
    if failed:
        write_summary(save_root / "purification_failed.csv", failed)
    print(f"\n[done] ok={len(rows)} failed={len(failed)} save_dir={save_root}")


if __name__ == "__main__":
    main()
