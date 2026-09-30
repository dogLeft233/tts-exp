"""SyncNet V2 frontend helpers for the frozen portable experiment.

Visual inputs follow the official evaluator's FFmpeg-JPEG/OpenCV path. Audio
MFCC receives float64 integer-amplitude units directly, preserving normalized
float PCM without int16 rounding or clipping.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import ops


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_syncnet(syncnet_root: str | Path, checkpoint: str | Path, device: str = "cuda"):
    import torch

    root = str(Path(syncnet_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from SyncNetInstance import SyncNetInstance

    network = SyncNetInstance(device=device)
    network.loadParameters(str(checkpoint))
    network.eval()
    try:
        psf_version = importlib.metadata.version("python_speech_features")
    except importlib.metadata.PackageNotFoundError:
        psf_version = "unknown"
    return network, {
        "repo_root": root,
        "checkpoint_path": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "device": device,
        "frontend": "official SyncNetInstance JPEG/OpenCV video path; direct float64 integer-amplitude MFCC input",
        "batch_size": 20,
        "vshift": 15,
        "torch_version": str(torch.__version__),
        "python_speech_features_version": psf_version,
    }


def extract_official_jpegs(video_path: str | Path, work_dir: str | Path, ffmpeg: str | Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    import cv2

    destination = Path(work_dir)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    command = [
        str(ffmpeg), "-y", "-loglevel", "error", "-i", str(video_path), "-threads", "1",
        "-f", "image2", str(destination / "%06d.jpg"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"SyncNet JPEG extraction failed: {completed.stderr[-2000:]}")
    paths = sorted(destination.glob("*.jpg"))
    frames = [cv2.imread(str(path)) for path in paths]
    if not frames or any(frame is None for frame in frames):
        raise RuntimeError(f"SyncNet JPEG extraction produced invalid frames: {destination}")
    shapes = sorted({tuple(frame.shape) for frame in frames})
    if len(shapes) != 1:
        raise RuntimeError(f"SyncNet JPEG frames have mixed shapes: {shapes}")
    return frames, {
        "frame_count": len(frames),
        "frame_shape_bgr": list(frames[0].shape),
        "jpeg_first_sha256": sha256_file(paths[0]),
        "jpeg_last_sha256": sha256_file(paths[-1]),
        "ffmpeg_command": command,
        "jpeg_directory": str(destination.resolve()),
    }


def lip_embeddings_from_frames(network: Any, frames: list[np.ndarray], *, device: str = "cuda", batch_size: int = 20) -> np.ndarray:
    import torch

    if len(frames) < 6 or batch_size <= 0:
        raise ValueError("SyncNet needs at least six video frames and a positive batch size")
    stacked = np.stack(frames, axis=3)
    im = np.transpose(np.expand_dims(stacked, axis=0), (0, 3, 4, 1, 2))
    imtv = torch.from_numpy(im.astype(float)).float()
    window_count = len(frames) - 5
    outputs = []
    network.__S__.eval()
    with torch.inference_mode():
        for start in range(0, window_count, batch_size):
            stop = min(window_count, start + batch_size)
            windows = [imtv[:, :, index : index + 5, :, :] for index in range(start, stop)]
            batch = torch.cat(windows, dim=0).to(device)
            outputs.append(network.__S__.forward_lip(batch).detach().float().cpu().numpy().copy())
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def audio_embeddings_from_syncnet_units(
    network: Any,
    integer_amplitude_units: np.ndarray,
    *,
    window_count: int,
    device: str = "cuda",
    batch_size: int = 20,
    sample_rate: int = 16_000,
) -> tuple[np.ndarray, dict[str, Any]]:
    import python_speech_features
    import torch

    units = np.asarray(integer_amplitude_units)
    if units.ndim != 1 or units.size == 0 or not np.isfinite(units).all():
        raise ValueError("SyncNet audio units must be finite nonempty mono data")
    if window_count < 1 or window_count > (len(units) // 640) - 5:
        raise ValueError("requested SyncNet audio window count is outside official support")
    mfcc_rows = python_speech_features.mfcc(units, sample_rate)
    mfcc = np.stack([np.asarray(row) for row in zip(*mfcc_rows, strict=True)], axis=0)
    needed_mfcc_frames = (window_count - 1) * 4 + 20
    if mfcc.shape[0] != 13 or mfcc.shape[1] < needed_mfcc_frames:
        raise RuntimeError(f"SyncNet MFCC shape cannot supply requested windows: {mfcc.shape}/{window_count}")
    if mfcc[:, (window_count - 1) * 4 : (window_count - 1) * 4 + 20].shape[1] != 20:
        raise AssertionError("final SyncNet MFCC window is shorter than the official 20-frame slice")
    cct = torch.from_numpy(np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0).astype(float)).float()
    outputs = []
    network.__S__.eval()
    with torch.inference_mode():
        for start in range(0, window_count, batch_size):
            stop = min(window_count, start + batch_size)
            windows = [cct[:, :, :, index * 4 : index * 4 + 20] for index in range(start, stop)]
            batch = torch.cat(windows, dim=0).to(device)
            outputs.append(network.__S__.forward_aud(batch).detach().float().cpu().numpy().copy())
    features = np.concatenate(outputs, axis=0).astype(np.float32, copy=False)
    return features, {
        "sample_rate_hz": int(sample_rate),
        "input_units_dtype": str(units.dtype),
        "input_units_sha256": ops.array_sha256(units),
        "mfcc_shape": list(mfcc.shape),
        "mfcc_dtype": str(mfcc.dtype),
        "mfcc_sha256": ops.array_sha256(mfcc),
        "audio_embedding_shape": list(features.shape),
        "audio_embedding_sha256": ops.array_sha256(features),
        "window_count": int(window_count),
        "quantization": "none; float64 waveform multiplied by 32768 before MFCC",
        "clipping": "none",
    }


def pairwise_distance_matrix(lip: np.ndarray, audio: np.ndarray, vshift: int = 15, device: str = "cuda") -> np.ndarray:
    import torch
    import torch.nn.functional as F

    v = np.asarray(lip, dtype=np.float32)
    a = np.asarray(audio, dtype=np.float32)
    if v.ndim != 2 or a.ndim != 2 or v.shape[1] != a.shape[1] or len(v) < 1 or len(a) < 1:
        raise ValueError("lip and audio embeddings must be nonempty aligned-dimension matrices")
    if vshift < 0:
        raise ValueError("vshift must be nonnegative")
    vt = torch.from_numpy(v).to(device)
    at = F.pad(torch.from_numpy(a).to(device), (0, 0, vshift, vshift))
    rows = []
    for index in range(len(v)):
        row = F.pairwise_distance(vt[index : index + 1].expand(2 * vshift + 1, -1), at[index : index + 2 * vshift + 1])
        rows.append(row.detach().cpu().numpy())
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def curve_metrics(matrix: np.ndarray, anchor_column: int | None = None) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
        raise ValueError("distance matrix must be nonempty [windows,lag]")
    curve = values.mean(axis=0, dtype=np.float32).astype(np.float64)
    best = int(np.argmin(curve))
    result = {
        "support_n": int(values.shape[0]),
        "curve": curve.tolist(),
        "lag_offsets_frames": [int(15 - index) for index in range(values.shape[1])],
        "optimal_offset_frames": int(15 - best),
        "optimal_column": best,
        "D_min": float(curve[best]),
        "B_median": float(np.median(curve)),
        "C_best": float(np.median(curve) - curve[best]),
    }
    if anchor_column is not None:
        if not 0 <= int(anchor_column) < values.shape[1]:
            raise ValueError("anchor column is outside the lag curve")
        result.update({
            "fixed_anchor_column": int(anchor_column),
            "fixed_anchor_offset_frames": int(15 - int(anchor_column)),
            "D_fixed_anchor": float(curve[int(anchor_column)]),
            "C_fixed_anchor": float(np.median(curve) - curve[int(anchor_column)]),
        })
    return result


def official_evaluate_capture(network: Any, options: Any, video_path: str | Path) -> dict[str, Any]:
    """Run the repository evaluator and retain embeddings for one parity check."""

    import torch

    lip_batches: list[np.ndarray] = []
    audio_batches: list[np.ndarray] = []
    original_lip = network.__S__.forward_lip
    original_audio = network.__S__.forward_aud

    def capture_lip(value):
        output = original_lip(value)
        lip_batches.append(output.detach().float().cpu().numpy().copy())
        return output

    def capture_audio(value):
        output = original_audio(value)
        audio_batches.append(output.detach().float().cpu().numpy().copy())
        return output

    network.__S__.forward_lip = capture_lip
    network.__S__.forward_aud = capture_audio
    try:
        offset, confidence, dists = network.evaluate(options, videofile=str(video_path))
    finally:
        network.__S__.forward_lip = original_lip
        network.__S__.forward_aud = original_audio
    return {
        "offset": int(np.asarray(offset).item()),
        "confidence": float(np.asarray(confidence).item()),
        "dists": np.asarray(dists, dtype=np.float32),
        "lip": np.concatenate(lip_batches, axis=0).astype(np.float32),
        "audio": np.concatenate(audio_batches, axis=0).astype(np.float32),
    }
