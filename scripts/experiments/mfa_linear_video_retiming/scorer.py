from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .common import ProtocolError, canonical_json_sha256, file_sha256


VSHIFT = 15
WINDOW_FRAMES = 5
MFCC_WINDOW = 20
MFCC_PER_FRAME = 4
FEATURE_DIM = 1024
PAIRWISE_EPS = 1e-6
ALLOWED_MISSING_COUNTER_KEYS = (
    "netcnnaud.1.num_batches_tracked",
    "netcnnaud.5.num_batches_tracked",
    "netcnnaud.9.num_batches_tracked",
    "netcnnaud.12.num_batches_tracked",
    "netcnnaud.15.num_batches_tracked",
    "netcnnaud.19.num_batches_tracked",
    "netcnnlip.1.num_batches_tracked",
    "netcnnlip.5.num_batches_tracked",
    "netcnnlip.9.num_batches_tracked",
    "netcnnlip.12.num_batches_tracked",
    "netcnnlip.15.num_batches_tracked",
    "netcnnlip.19.num_batches_tracked",
    "netfcaud.1.num_batches_tracked",
    "netfclip.1.num_batches_tracked",
)


def validate_checkpoint_state(
    checkpoint_state: Mapping[str, Any],
    model_state: Mapping[str, Any],
    *,
    allowed_missing: Sequence[str] = ALLOWED_MISSING_COUNTER_KEYS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_keys = set(checkpoint_state)
    model_keys = set(model_state)
    unexpected = sorted(checkpoint_keys - model_keys)
    missing = sorted(model_keys - checkpoint_keys)
    allowed = set(allowed_missing)
    if unexpected:
        raise ProtocolError(f"SyncNet checkpoint has unexpected keys: {unexpected}")
    if missing != sorted(allowed):
        raise ProtocolError(f"SyncNet checkpoint keyset mismatch; missing={missing}, allowed_batchnorm_counters={sorted(allowed)}")
    shape_mismatches = [
        (key, tuple(checkpoint_state[key].shape), tuple(model_state[key].shape))
        for key in sorted(checkpoint_keys)
        if tuple(checkpoint_state[key].shape) != tuple(model_state[key].shape)
    ]
    if shape_mismatches:
        raise ProtocolError(f"SyncNet checkpoint tensor shapes differ: {shape_mismatches[:8]}")
    strict_state = dict(checkpoint_state)
    for key in missing:
        if not key.endswith(".num_batches_tracked"):
            raise ProtocolError(f"only enumerated BatchNorm counters may be absent: {key}")
        strict_state[key] = model_state[key]
    if set(strict_state) != model_keys:
        raise ProtocolError("normalized SyncNet state does not have the model's exact keyset")
    normalization = {
        "checkpoint_key_count": len(checkpoint_keys),
        "model_key_count": len(model_keys),
        "missing_batchnorm_counter_keys": missing,
        "unexpected_keys": unexpected,
        "common_tensor_shapes_match": True,
        "strict_load_keyset_match_after_buffer_completion": set(strict_state) == model_keys,
        "buffer_compatibility": "initialize only enumerated evaluation-inert num_batches_tracked buffers from model defaults",
    }
    return strict_state, normalization


def _state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_frozen_syncnet(
    syncnet_root: str | Path,
    checkpoint: str | Path,
    *,
    device: str = "cuda",
) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    root = Path(syncnet_root).resolve()
    model_path = Path(checkpoint).resolve()
    if not root.is_dir() or not model_path.is_file():
        raise ProtocolError("frozen SyncNet source tree or checkpoint is missing")
    if device == "cuda" and not torch.cuda.is_available():
        raise ProtocolError("SyncNet scoring requires CUDA; CPU fallback is forbidden")
    root_string = str(root)
    if root_string in sys.path:
        sys.path.remove(root_string)
    sys.path.insert(0, root_string)
    instance_module = importlib.import_module("SyncNetInstance")
    model_module = importlib.import_module("SyncNetModel")
    expected_instance = (root / "SyncNetInstance.py").resolve()
    expected_model = (root / "SyncNetModel.py").resolve()
    actual_instance = Path(str(instance_module.__file__)).resolve()
    actual_model = Path(str(model_module.__file__)).resolve()
    if actual_instance != expected_instance or actual_model != expected_model:
        raise ProtocolError(f"SyncNet import shadowing: instance={actual_instance}, model={actual_model}")

    scorer = instance_module.SyncNetInstance(device=device)
    scorer.eval()
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    raw_state = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(raw_state, Mapping):
        raise ProtocolError("SyncNet checkpoint is not a state dictionary")
    strict_state, key_audit = validate_checkpoint_state(raw_state, scorer.__S__.state_dict())
    scorer.__S__.load_state_dict(strict_state, strict=True)
    scorer.__S__.eval()
    for parameter in scorer.__S__.parameters():
        parameter.requires_grad_(False)
    parameter_sha = _state_sha256(scorer.__S__.state_dict())
    if scorer.training or scorer.__S__.training or any(parameter.requires_grad for parameter in scorer.parameters()):
        raise ProtocolError("SyncNet did not enter frozen evaluation mode")
    metadata = {
        "device": device,
        "checkpoint": str(model_path),
        "checkpoint_sha256": file_sha256(model_path),
        "instance_module": str(actual_instance),
        "instance_module_sha256": file_sha256(actual_instance),
        "model_module": str(actual_model),
        "model_module_sha256": file_sha256(actual_model),
        "parameter_state_sha256": parameter_sha,
        "checkpoint_key_audit": key_audit,
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "opencv_version": str(cv2.__version__),
    }
    metadata["scorer_fingerprint"] = canonical_json_sha256(metadata)
    return scorer, torch, metadata


def read_video_frames(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ProtocolError(f"cannot open video for scoring: {path}")
    frames: list[np.ndarray] = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.dtype != np.uint8 or frame.shape != (height, width, 3):
                raise ProtocolError("scoring input is not full uint8 BGR video")
            frames.append(frame)
    finally:
        capture.release()
    if not frames or abs(fps - 25.0) > 0.01:
        raise ProtocolError(f"SyncNet input must be 25 fps: {path} frames={len(frames)} fps={fps}")
    return np.stack(frames), {"frame_count": len(frames), "fps": fps, "width": width, "height": height}


def crop_zero_padded(frame: np.ndarray, box: Sequence[int], output_size: int = 224) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ProtocolError("score crop input must be uint8 BGR")
    if len(box) != 4:
        raise ProtocolError("score crop must be XYXY")
    x1, y1, x2, y2 = [int(value) for value in box]
    side = x2 - x1
    if side <= 0 or y2 - y1 != side:
        raise ProtocolError("score crop box must be a positive square in XYXY order")
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
    if sx2 > sx1 and sy2 > sy1:
        canvas[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = frame[sy1:sy2, sx1:sx2]
    return cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def crop_frames(frames: np.ndarray, score_box: Sequence[int], output_size: int = 224) -> np.ndarray:
    source = np.asarray(frames)
    if source.ndim != 4 or source.dtype != np.uint8 or source.shape[-1] != 3:
        raise ProtocolError("video must be full uint8 BGR [F,H,W,3]")
    return np.stack([crop_zero_padded(frame, score_box, output_size) for frame in source], axis=0)


def make_video_tensor(cropped_frames: np.ndarray, *, start: int = 0, stop: int | None = None) -> np.ndarray:
    cropped = np.asarray(cropped_frames)
    if cropped.ndim != 4 or cropped.shape[1:] != (224, 224, 3) or cropped.dtype != np.uint8:
        raise ProtocolError("cropped video must be uint8 [F,224,224,3]")
    count = cropped.shape[0] - WINDOW_FRAMES + 1
    end = count if stop is None else min(count, int(stop))
    begin = int(start)
    if begin < 0 or end <= begin:
        raise ProtocolError("video has no requested five-frame windows")
    windows = np.stack([cropped[index:index + WINDOW_FRAMES] for index in range(begin, end)], axis=0)
    return np.transpose(windows, (0, 4, 1, 2, 3)).astype(np.float32, copy=False)


def make_audio_tensor(mfcc: np.ndarray, *, start: int = 0, stop: int | None = None) -> np.ndarray:
    values = np.asarray(mfcc, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 13:
        raise ProtocolError(f"SyncNet audio MFCC must have shape [13,T], got {values.shape}")
    count = (values.shape[1] - MFCC_WINDOW) // MFCC_PER_FRAME + 1
    end = count if stop is None else min(count, int(stop))
    begin = int(start)
    if begin < 0 or end <= begin:
        raise ProtocolError("audio has no requested 13x20 MFCC windows")
    windows = np.stack([
        values[:, index * MFCC_PER_FRAME:index * MFCC_PER_FRAME + MFCC_WINDOW]
        for index in range(begin, end)
    ], axis=0)
    return windows[:, None, :, :].astype(np.float32, copy=False)


def mfcc_from_pcm16(pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    import python_speech_features

    values = np.asarray(pcm)
    if sample_rate != 16000 or values.ndim != 1 or values.dtype != np.int16:
        raise ProtocolError("SyncNet audio input must be original mono 16 kHz PCM16")
    mfcc = np.asarray(list(zip(*python_speech_features.mfcc(values, sample_rate))), dtype=np.float32)
    if mfcc.ndim != 2 or mfcc.shape[0] != 13 or not np.isfinite(mfcc).all():
        raise ProtocolError(f"unexpected official SyncNet MFCC result {mfcc.shape}")
    return mfcc


def forward_video(cropped_frames: np.ndarray, scorer: Any, torch: Any, *, batch_size: int = 20) -> np.ndarray:
    count = int(cropped_frames.shape[0] - WINDOW_FRAMES + 1)
    if count < 1:
        raise ProtocolError("video must contain at least five cropped frames")
    embeddings: list[np.ndarray] = []
    for start in range(0, count, max(1, int(batch_size))):
        tensor = make_video_tensor(cropped_frames, start=start, stop=min(count, start + max(1, int(batch_size))))
        with torch.inference_mode():
            output = scorer.__S__.forward_lip(torch.from_numpy(tensor).to(scorer.device))
        batch = output.detach().cpu().numpy().astype(np.float32, copy=False)
        if batch.ndim != 2 or batch.shape[1] != FEATURE_DIM or not np.isfinite(batch).all():
            raise ProtocolError(f"SyncNet visual tower returned invalid shape/data: {batch.shape}")
        embeddings.append(batch)
    return np.concatenate(embeddings, axis=0)


def forward_audio(mfcc: np.ndarray, scorer: Any, torch: Any, *, batch_size: int = 20) -> np.ndarray:
    count = (int(mfcc.shape[1]) - MFCC_WINDOW) // MFCC_PER_FRAME + 1
    if count < 1:
        raise ProtocolError("audio has insufficient MFCC windows")
    embeddings: list[np.ndarray] = []
    for start in range(0, count, max(1, int(batch_size))):
        tensors = make_audio_tensor(mfcc, start=start, stop=min(count, start + max(1, int(batch_size))))
        with torch.inference_mode():
            output = scorer.__S__.forward_aud(torch.from_numpy(tensors).to(scorer.device))
        batch = output.detach().cpu().numpy().astype(np.float32, copy=False)
        if batch.ndim != 2 or batch.shape[1] != FEATURE_DIM or not np.isfinite(batch).all():
            raise ProtocolError(f"SyncNet audio tower returned invalid shape/data: {batch.shape}")
        embeddings.append(batch)
    return np.concatenate(embeddings, axis=0)


def load_audio_embeddings(path: str | Path, scorer: Any, torch: Any, *, batch_size: int = 20) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.io import wavfile

    target = Path(path).resolve()
    sample_rate, pcm = wavfile.read(str(target))
    if np.asarray(pcm).dtype != np.int16:
        raise ProtocolError(f"audio is not PCM16: {target}")
    mfcc = mfcc_from_pcm16(np.asarray(pcm), int(sample_rate))
    embeddings = forward_audio(mfcc, scorer, torch, batch_size=batch_size)
    return embeddings, {
        "audio_path": str(target),
        "audio_sha256": file_sha256(target),
        "pcm_sha256": hashlib.sha256(np.ascontiguousarray(pcm).tobytes()).hexdigest(),
        "sample_rate": int(sample_rate),
        "sample_count": int(np.asarray(pcm).size),
        "mfcc_shape": [int(value) for value in mfcc.shape],
        "mfcc_sha256": hashlib.sha256(np.ascontiguousarray(mfcc).tobytes()).hexdigest(),
        "embedding_shape": [int(value) for value in embeddings.shape],
        "embedding_sha256": hashlib.sha256(np.ascontiguousarray(embeddings).tobytes()).hexdigest(),
    }


def distance_matrix(
    visual: np.ndarray,
    audio: np.ndarray,
    *,
    n_support: int,
    torch: Any,
    vshift: int = VSHIFT,
    eps: float = PAIRWISE_EPS,
) -> np.ndarray:
    v = np.asarray(visual, dtype=np.float32)
    a = np.asarray(audio, dtype=np.float32)
    if v.ndim != 2 or a.ndim != 2 or v.shape[1] != FEATURE_DIM or a.shape[1] != FEATURE_DIM:
        raise ProtocolError("SyncNet embeddings must be [N,1024] without extra normalization")
    n = int(n_support)
    if n < 1 or n > v.shape[0] or n > a.shape[0]:
        raise ProtocolError("frozen support exceeds available visual/audio embeddings")
    matrix = np.full((n, 2 * int(vshift) + 1), np.nan, dtype=np.float32)
    for index in range(n):
        for column, lag in enumerate(range(-int(vshift), int(vshift) + 1)):
            audio_index = index + lag
            if 0 <= audio_index < a.shape[0]:
                left = torch.from_numpy(v[index:index + 1])
                right = torch.from_numpy(a[audio_index:audio_index + 1])
                value = torch.nn.functional.pairwise_distance(left, right, p=2, eps=float(eps))[0]
                matrix[index, column] = np.float32(value.item())
    return matrix


def support_indices(n_support: int, *, vshift: int = VSHIFT) -> np.ndarray:
    n = int(n_support)
    indices = np.arange(int(vshift), n - int(vshift), dtype=np.int64)
    if indices.size < 25:
        raise ProtocolError(f"INSUFFICIENT_SUPPORT: fixed support has {indices.size} rows (<25)")
    return indices


def _curve_metrics(curve: np.ndarray, *, lags: Sequence[int] | None = None) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float64)
    expected = 2 * VSHIFT + 1
    if values.shape != (expected,) or not np.isfinite(values).all():
        raise ProtocolError("SyncNet curve must contain exactly 31 finite entries")
    minimum_index = int(np.flatnonzero(values == np.min(values))[0])
    actual_lags = list(range(-VSHIFT, VSHIFT + 1)) if lags is None else [int(value) for value in lags]
    median = float(np.median(values))
    minimum = float(values[minimum_index])
    return {
        "sync_c": median - minimum,
        "sync_d": minimum,
        "curve_median": median,
        "d0": float(values[VSHIFT]),
        "best_lag": int(actual_lags[minimum_index]),
        "offset": int(VSHIFT - minimum_index),
        "curve": [float(value) for value in values],
    }


def score_embeddings(
    visual: np.ndarray,
    audio: np.ndarray,
    *,
    frame_count: int,
    valid_frame_count: int,
    audio_sample_count: int,
    torch: Any,
    vshift: int = VSHIFT,
    local_window_frames: int = 25,
) -> dict[str, Any]:
    n = min(int(valid_frame_count), int(audio_sample_count) // 640) - 5
    if n <= 2 * int(vshift):
        raise ProtocolError(f"INSUFFICIENT_SUPPORT: conservative window count is {n}")
    if n > np.asarray(visual).shape[0] or n > np.asarray(audio).shape[0]:
        raise ProtocolError("conservative support exceeds embedding coverage")
    rows = support_indices(n, vshift=vshift)
    matrix = distance_matrix(visual, audio, n_support=n, torch=torch, vshift=vshift)
    fixed = matrix[rows]
    if not np.isfinite(fixed).all():
        raise ProtocolError("frozen W has non-finite entries in the 31-lag distance matrix")
    curve = np.mean(fixed, axis=0, dtype=np.float32).astype(np.float64)
    metrics = _curve_metrics(curve)
    local: list[dict[str, Any]] = []
    window_size = int(local_window_frames)
    for start in range(0, rows.size, window_size):
        chunk = rows[start:start + window_size]
        if chunk.size < window_size:
            continue
        values = np.mean(matrix[chunk], axis=0, dtype=np.float32).astype(np.float64)
        row = _curve_metrics(values)
        local.append({"row_start": int(chunk[0]), "row_stop_exclusive": int(chunk[-1] + 1), **row})
    if not local:
        raise ProtocolError("INSUFFICIENT_SUPPORT: no complete local 25-row window")
    metrics.update({
        "frame_count": int(frame_count),
        "valid_frame_count": int(valid_frame_count),
        "audio_sample_count": int(audio_sample_count),
        "conservative_window_count": n,
        "W": [int(value) for value in rows],
        "W_sha256": canonical_json_sha256([int(value) for value in rows]),
        "distance_matrix_shape": [int(value) for value in matrix.shape],
        "distance_matrix": matrix.tolist(),
        "local_windows": local,
        "local_median_d0": float(np.median([value["d0"] for value in local])),
        "local_abs_offset_q90": float(np.quantile([abs(value["offset"]) for value in local], 0.9, method="linear")),
    })
    return metrics


def score_video_frames(
    frames: np.ndarray,
    audio_embeddings: np.ndarray,
    *,
    score_box: Sequence[int],
    audio_sample_count: int,
    scorer: Any,
    torch: Any,
    batch_size: int = 20,
    valid_frame_count: int | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    source = np.asarray(frames)
    cropped = crop_frames(source, score_box)
    visual = forward_video(cropped, scorer, torch, batch_size=batch_size)
    valid = source.shape[0] if valid_frame_count is None else int(valid_frame_count)
    result = score_embeddings(
        visual,
        audio_embeddings,
        frame_count=source.shape[0],
        valid_frame_count=valid,
        audio_sample_count=audio_sample_count,
        torch=torch,
    )
    return result, visual, cropped


def calibration_fingerprint(
    model_metadata: Mapping[str, Any],
    *,
    score_box: Sequence[int],
    support: Sequence[int],
    legacy_score_worker: str | Path,
) -> str:
    return canonical_json_sha256({
        "model": dict(model_metadata),
        "score_box_xyxy": [int(value) for value in score_box],
        "W": [int(value) for value in support],
        "legacy_score_worker_sha256": file_sha256(legacy_score_worker),
        "protocol": "fixed_crop_syncnet_v2_wxyz_v1",
    })
