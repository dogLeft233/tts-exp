"""Portable GPU-side Wav2Lip operations for the 2026-09-30 run."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from . import ops


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_librosa_compatibility_shim() -> dict[str, Any]:
    """Map the old Wav2Lip positional mel API to librosa 0.10 keyword args."""

    import librosa

    original = librosa.filters.mel
    if getattr(original, "_tts_source_axis_positional_compat", False):
        return {
            "librosa_version": str(librosa.__version__),
            "shim": "positional sr,n_fft mapped to keyword arguments; remaining defaults unchanged",
        }

    def compatible_mel(*args: Any, **kwargs: Any) -> np.ndarray:
        positional = list(args)
        if positional and "sr" not in kwargs:
            kwargs["sr"] = positional.pop(0)
        if positional and "n_fft" not in kwargs:
            kwargs["n_fft"] = positional.pop(0)
        if positional:
            raise TypeError("unexpected positional arguments in librosa.filters.mel compatibility shim")
        return original(**kwargs)

    setattr(compatible_mel, "_tts_source_axis_positional_compat", True)
    librosa.filters.mel = compatible_mel
    return {
        "librosa_version": str(librosa.__version__),
        "shim": "positional sr,n_fft mapped to keyword arguments; remaining defaults unchanged",
    }


def import_wav2lip(root: str | Path):
    root_path = Path(root).resolve()
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))
    install_librosa_compatibility_shim()
    import audio
    from models import Wav2Lip

    return audio, Wav2Lip


def load_model(root: str | Path, checkpoint: str | Path, device: str = "cuda"):
    import torch

    _, Wav2Lip = import_wav2lip(root)
    model = Wav2Lip()
    try:
        loaded = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(str(checkpoint), map_location="cpu")
    state = loaded.get("state_dict", loaded)
    model.load_state_dict({key.replace("module.", ""): value for key, value in state.items()})
    model = model.to(device).eval()
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return model, {
        "checkpoint_path": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "loaded_state_sha256": digest.hexdigest(),
        "model_class": "models.wav2lip.Wav2Lip",
        "device": device,
    }


def detect_static_face(frame: np.ndarray, wav2lip_root: str | Path, device: str = "cuda", threshold: float = 0.9) -> dict[str, Any]:
    """Use the bundled S3FD detector and frozen static-image box recipe."""

    import cv2
    import torch

    root = str(Path(wav2lip_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from face_detection.detection.sfd import FaceDetector

    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA is unavailable for the frozen face detector")
    detector = FaceDetector(device=device, verbose=False)
    try:
        detections = detector.detect_from_image(frame)
    finally:
        del detector
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    candidates = []
    for detection in detections:
        values = [float(value) for value in detection]
        if len(values) == 5:
            candidates.append({"x1": values[0], "y1": values[1], "x2": values[2], "y2": values[3], "score": values[4]})
    eligible = [row for row in candidates if row["score"] >= float(threshold)]
    eligible.sort(key=lambda row: (-row["score"], row["x1"], row["y1"]))
    if not eligible:
        raise RuntimeError(f"no static face reaches frozen S3FD threshold {threshold}")
    selected = eligible[0]
    width, height = int(frame.shape[1]), int(frame.shape[0])
    generation = [
        max(0, int(np.floor(selected["x1"]))),
        max(0, int(np.floor(selected["y1"]))),
        min(width, int(np.ceil(selected["x2"]))),
        min(height, int(np.ceil(selected["y2"] + 10.0))),
    ]
    if generation[2] <= generation[0] or generation[3] <= generation[1]:
        raise RuntimeError("static face generation box is empty")
    box_width = selected["x2"] - selected["x1"]
    box_height = selected["y2"] - selected["y1"]
    side = max(1, round(1.5 * max(box_width, box_height)))
    center_x = (selected["x1"] + selected["x2"]) / 2.0
    center_y = (selected["y1"] + selected["y2"]) / 2.0
    left = int(np.floor(center_x - side / 2.0))
    top = int(np.floor(center_y - side / 2.0))
    score_box = {"box": [left, top, left + side, top + side], "side": side, "center": [center_x, center_y]}
    return {
        "detector": "Wav2Lip S3FD FaceDetector",
        "threshold": float(threshold),
        "detections": candidates,
        "selected": selected,
        "generation_box_xyxy": generation,
        "score_box": score_box,
        "zero_padding": True,
    }


def image_tensor_from_frame(frame: np.ndarray, generation_box: Sequence[int], count: int, device: str):
    import cv2
    import torch

    if count <= 0:
        raise ValueError("face batch size must be positive")
    x1, y1, x2, y2 = [int(value) for value in generation_box]
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        raise ValueError("static generation box produced an empty image crop")
    resized = cv2.resize(face, (96, 96), interpolation=cv2.INTER_LINEAR)
    batch = np.repeat(resized[None, ...], count, axis=0)
    masked = batch.copy()
    masked[:, 48:] = 0
    images = np.concatenate((masked, batch), axis=3).astype(np.float32) / np.float32(255.0)
    return torch.from_numpy(np.transpose(images, (0, 3, 1, 2)).copy()).to(device)


def cache_face_features(model: Any, frame: np.ndarray, generation_box: Sequence[int], counts: Sequence[int], device: str = "cuda") -> dict[int, list[Any]]:
    """Cache all face-encoder skip tensors once per needed batch shape."""

    import torch

    cache: dict[int, list[Any]] = {}
    with torch.inference_mode():
        for count in sorted({int(value) for value in counts if int(value) > 0}):
            tensor = image_tensor_from_frame(frame, generation_box, count, device)
            x = tensor
            skips = []
            for block in model.face_encoder_blocks:
                x = block(x)
                skips.append(x.detach())
            cache[count] = skips
    return cache


def mel_chunks(waveform: np.ndarray, wav2lip_root: str | Path) -> tuple[np.ndarray, np.ndarray, list[int], dict[str, Any]]:
    """Run the repository Wav2Lip mel frontend and official chunk indexing."""

    audio, _ = import_wav2lip(wav2lip_root)
    values = np.asarray(waveform, dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Wav2Lip frontend requires finite nonempty mono float32 PCM")
    mel = np.asarray(audio.melspectrogram(values), dtype=np.float32)
    if mel.ndim != 2 or mel.shape[0] != 80 or mel.shape[1] < 16 or not np.isfinite(mel).all():
        raise ValueError(f"Wav2Lip mel frontend returned invalid shape: {mel.shape}")
    starts = ops.mel_chunk_starts(mel.shape[1])
    chunks = np.stack([mel[:, start : start + 16] for start in starts], axis=0).astype(np.float32)
    metadata = {
        "mel_shape": list(mel.shape),
        "mel_dtype": str(mel.dtype),
        "mel_sha256": ops.array_sha256(mel),
        "chunk_shape": list(chunks.shape),
        "chunk_sha256": ops.array_sha256(chunks),
        "mel_starts": starts,
        "frontend": "Wav2Lip audio.melspectrogram; librosa 0.10 positional compatibility shim",
    }
    return mel, chunks, starts, metadata


def encode_latents(model: Any, chunks: np.ndarray, *, device: str = "cuda", batch_size: int = 32) -> np.ndarray:
    """Run Wav2Lip audio_encoder once and retain its exact float32 output."""

    import torch

    value = np.asarray(chunks)
    if value.dtype != np.float32 or value.ndim != 3 or value.shape[1:] != (80, 16) or not np.isfinite(value).all():
        raise ValueError(f"mel chunks must be float32 [frames,80,16], got {value.shape}/{value.dtype}")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    outputs = []
    model.audio_encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(value), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(value[start : start + batch_size, None, :, :])).to(device)
            output = model.audio_encoder(batch)
            if output.ndim != 4 or output.shape[1:] != (ops.LATENT_DIM, 1, 1):
                raise RuntimeError(f"Wav2Lip audio encoder returned unexpected shape {tuple(output.shape)}")
            outputs.append(output[:, :, 0, 0].detach().cpu().numpy().astype(np.float32, copy=True))
    latents = np.concatenate(outputs, axis=0)
    if not np.isfinite(latents).all():
        raise RuntimeError("Wav2Lip audio embeddings contain NaN or infinity")
    return latents


def decode_latents(model: Any, latents: np.ndarray, face_skips: Sequence[Any], *, device: str = "cuda"):
    """Decode retained 512-D latents through Wav2Lip's unchanged decoder."""

    import torch

    value = np.asarray(latents)
    if value.dtype != np.float32 or value.ndim != 2 or value.shape[1] != ops.LATENT_DIM or not np.isfinite(value).all():
        raise ValueError("decoder latents must be finite float32 [batch,512]")
    if not face_skips:
        raise ValueError("face encoder skip tensors were not supplied")
    batch_size = len(value)
    x = torch.from_numpy(np.ascontiguousarray(value[:, :, None, None])).to(device)
    features = list(face_skips)
    if len(features) != len(model.face_decoder_blocks):
        raise ValueError("cached face skips do not match the Wav2Lip decoder")
    with torch.inference_mode():
        for block in model.face_decoder_blocks:
            x = block(x)
            skip = features.pop()
            if skip.shape[0] != batch_size:
                raise ValueError("face skip batch size differs from audio latent batch")
            x = torch.cat((x, skip), dim=1)
        output = model.output_block(x)
    if tuple(output.shape) != (batch_size, 3, 96, 96):
        raise RuntimeError(f"Wav2Lip decoder returned unexpected output {tuple(output.shape)}")
    return output


def parity_check(model: Any, chunks: np.ndarray, face_frame: np.ndarray, generation_box: Sequence[int], face_skips: Sequence[Any], *, device: str = "cuda") -> dict[str, Any]:
    """Compare cached-face direct decode to the unchanged Wav2Lip forward."""

    import torch

    count = min(4, len(chunks))
    mel = torch.from_numpy(np.ascontiguousarray(chunks[:count, None, :, :])).to(device)
    image = image_tensor_from_frame(face_frame, generation_box, count, device)
    with torch.inference_mode():
        native_z = model.audio_encoder(mel)[:, :, 0, 0]
        full = model(mel, image)
        cached = decode_latents(model, native_z.detach().cpu().numpy().astype(np.float32), face_skips[:], device=device)
    a = full.detach().cpu().numpy()
    b = cached.detach().cpu().numpy()
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    ua = (a.transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
    ub = (b.transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
    return {
        "frames_compared": int(count),
        "max_abs_float_prediction": float(difference.max(initial=0.0)),
        "float_prediction_exact": bool(np.array_equal(a, b)),
        "uint8_prediction_exact": bool(np.array_equal(ua, ub)),
        "uint8_differing_values": int(np.count_nonzero(ua != ub)),
        "native_latent_shape": list(native_z.shape),
        "device": device,
    }


def parity_check_batches(
    model: Any,
    chunks: np.ndarray,
    face_frame: np.ndarray,
    generation_box: Sequence[int],
    face_features: dict[int, list[Any]],
    *,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict[str, Any]:
    """Check native forward against encode/cache/decode on head and tail batches."""

    import torch

    values = np.asarray(chunks)
    if values.dtype != np.float32 or values.ndim != 3 or values.shape[1:] != (80, 16) or not len(values):
        raise ValueError("parity chunks must be nonempty float32 [frames,80,16]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    count = min(batch_size, len(values))
    ranges = [(0, count)]
    if len(values) > count:
        tail_start = (len(values) // batch_size) * batch_size
        if tail_start == len(values):
            tail_start = max(0, len(values) - batch_size)
        tail_range = (tail_start, len(values))
        if tail_range not in ranges:
            ranges.append(tail_range)
    rows = []
    checked: set[int] = set()
    with torch.inference_mode():
        for start, stop in ranges:
            current = np.ascontiguousarray(values[start:stop])
            current_count = len(current)
            if current_count not in face_features:
                raise RuntimeError(f"face features for parity batch size {current_count} are missing")
            mel = torch.from_numpy(current[:, None, :, :]).to(device)
            image = image_tensor_from_frame(face_frame, generation_box, current_count, device)
            z = model.audio_encoder(mel)
            native = model(mel, image)
            direct = decode_latents(model, z[:, :, 0, 0].detach().cpu().numpy().astype(np.float32), face_features[current_count], device=device)
            native_pixels = (native.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
            direct_pixels = (direct.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
            difference = np.abs(native.detach().cpu().numpy().astype(np.float64) - direct.detach().cpu().numpy().astype(np.float64))
            rows.append({
                "start": int(start),
                "stop_exclusive": int(stop),
                "batch_size": int(current_count),
                "max_abs_float_prediction": float(difference.max(initial=0.0)),
                "float_prediction_exact": bool(np.array_equal(native.detach().cpu().numpy(), direct.detach().cpu().numpy())),
                "uint8_prediction_exact": bool(np.array_equal(native_pixels, direct_pixels)),
                "uint8_differing_values": int(np.count_nonzero(native_pixels != direct_pixels)),
            })
            checked.update(range(start, stop))
    return {
        "total_frames": int(len(values)),
        "checked_frames": int(len(checked)),
        "all_frames_checked": len(checked) == len(values),
        "checked_ranges": [[int(start), int(stop)] for start, stop in ranges],
        "float_prediction_exact": all(row["float_prediction_exact"] for row in rows),
        "uint8_prediction_exact": all(row["uint8_prediction_exact"] for row in rows),
        "max_abs_float_prediction": max(row["max_abs_float_prediction"] for row in rows),
        "uint8_differing_values": sum(row["uint8_differing_values"] for row in rows),
        "batch_checks": rows,
        "device": device,
    }


class _PipeEncoder:
    def __init__(self, output: str | Path, ffmpeg: str | Path, width: int, height: int, *, codec: str = "libx264") -> None:
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = int(width), int(height)
        command = [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.width}x{self.height}", "-r", "25", "-i", "pipe:0",
            "-an", "-c:v", codec,
        ]
        if codec == "libx264":
            command.extend(["-pix_fmt", "yuv420p"])
        else:
            command.extend(["-level", "3", "-g", "1", "-pix_fmt", "bgr0", "-f", "matroska"])
        command.append(str(self.output))
        self.command = command
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if self.process.stdin is None or self.process.stderr is None:
            self.process.kill()
            raise RuntimeError("FFmpeg rawvideo encoder failed to create pipes")
        self.stderr_tail: deque[bytes] = deque(maxlen=30)
        self.stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self.stderr_thread.start()
        self.frames = 0
        self.pixel_digest = hashlib.sha256()

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        while True:
            chunk = self.process.stderr.readline()
            if not chunk:
                break
            self.stderr_tail.append(chunk)

    def write(self, frame: np.ndarray) -> None:
        if frame.dtype != np.uint8 or frame.shape != (self.height, self.width, 3):
            raise ValueError(f"frame shape/dtype invalid for {self.output}: {frame.shape}/{frame.dtype}")
        raw = np.ascontiguousarray(frame).tobytes()
        assert self.process.stdin is not None
        self.process.stdin.write(raw)
        self.pixel_digest.update(raw)
        self.frames += 1

    def close(self) -> dict[str, Any]:
        assert self.process.stdin is not None
        self.process.stdin.close()
        returncode = self.process.wait()
        self.stderr_thread.join(timeout=10)
        stderr = b"".join(self.stderr_tail).decode("utf-8", errors="replace")
        if returncode != 0 or self.frames < 1 or not self.output.is_file():
            raise RuntimeError(f"FFmpeg encode failed ({returncode}) for {self.output}: {stderr[-2000:]}")
        return {
            "path": str(self.output.resolve()),
            "sha256": sha256_file(self.output),
            "raw_bgr_pixel_sha256": self.pixel_digest.hexdigest(),
            "frame_count": self.frames,
            "width": self.width,
            "height": self.height,
            "codec": self.command[self.command.index("-c:v") + 1],
            "ffmpeg_command": self.command,
        }


def render_video(
    model: Any,
    latents: np.ndarray,
    frame: np.ndarray,
    generation_box: Sequence[int],
    face_features: dict[int, list[Any]],
    output: str | Path,
    ffmpeg: str | Path,
    *,
    batch_size: int = 32,
    device: str = "cuda",
    codec: str = "libx264",
) -> dict[str, Any]:
    """Render retained latents once through a reusable model and face cache."""

    import cv2
    import torch

    value = np.asarray(latents)
    if value.dtype != np.float32 or value.ndim != 2 or value.shape[1] != ops.LATENT_DIM or len(value) < 1:
        raise ValueError("video latents must be float32 [frames,512]")
    x1, y1, x2, y2 = [int(v) for v in generation_box]
    encoder = _PipeEncoder(output, ffmpeg, int(frame.shape[1]), int(frame.shape[0]), codec=codec)
    start = 0
    try:
        while start < len(value):
            end = min(start + batch_size, len(value))
            count = end - start
            if count not in face_features:
                raise RuntimeError(f"face features for final batch size {count} were not cached")
            prediction = decode_latents(model, value[start:end], face_features[count], device=device)
            pixels = (prediction.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
            for output_face in pixels:
                full = frame.copy()
                patch = cv2.resize(output_face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                full[y1:y2, x1:x2] = patch
                encoder.write(full)
            start = end
    except BaseException:
        if encoder.process.poll() is None:
            encoder.process.kill()
        encoder.process.wait()
        raise
    return encoder.close()


def crop_score_video(
    video_path: str | Path,
    score_box: dict[str, Any],
    output: str | Path,
    ffmpeg: str | Path,
) -> dict[str, Any]:
    """Make a fixed 224px face video using project OpenCV crop semantics."""

    import cv2

    x1, y1, x2, y2 = [int(item) for item in score_box["box"]]
    side = int(score_box["side"])
    if side <= 0 or x2 - x1 != side or y2 - y1 != side:
        raise ValueError("score_box must be a fixed unbounded square")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not read Wav2Lip video {video_path}")
    encoder = _PipeEncoder(output, ffmpeg, 224, 224, codec="ffv1")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            canvas = np.zeros((side, side, 3), dtype=np.uint8)
            sx1, sy1 = max(0, x1), max(0, y1)
            sx2, sy2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if sx2 > sx1 and sy2 > sy1:
                canvas[sy1 - y1 : sy2 - y1, sx1 - x1 : sx2 - x1] = frame[sy1:sy2, sx1:sx2]
            cropped = cv2.resize(canvas, (224, 224), interpolation=cv2.INTER_LINEAR)
            encoder.write(cropped)
    except BaseException:
        if encoder.process.poll() is None:
            encoder.process.kill()
        encoder.process.wait()
        raise
    finally:
        capture.release()
    result = encoder.close()
    result["source_video_sha256"] = sha256_file(video_path)
    result["crop_box"] = dict(score_box)
    result["crop_method"] = "OpenCV VideoCapture BGR; zero-padded square; INTER_LINEAR to 224x224"
    return result
