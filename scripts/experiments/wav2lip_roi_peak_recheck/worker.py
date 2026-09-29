from __future__ import annotations

import argparse
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import python_speech_features
import torch

from . import config
from .common import RecheckError, bytes_sha256, file_sha256, write_self_hashed_json

if str(config.SYNCNET_ROOT) not in sys.path:
    sys.path.insert(0, str(config.SYNCNET_ROOT))

from SyncNetModel import S


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-1000:]
        raise RecheckError(f"command failed ({result.returncode}): {' '.join(command)}\n{detail}")


def _pcm_from_wav(path: Path) -> tuple[bytes, np.ndarray, dict[str, Any]]:
    try:
        with wave.open(str(path), "rb") as handle:
            params = {
                "channels": handle.getnchannels(),
                "sample_width": handle.getsampwidth(),
                "sample_rate": handle.getframerate(),
                "frame_count": handle.getnframes(),
            }
            pcm = handle.readframes(handle.getnframes())
    except (OSError, wave.Error) as exc:
        raise RecheckError(f"cannot read extracted WAV: {path}") from exc
    if params != {"channels": 1, "sample_width": 2, "sample_rate": config.SAMPLE_RATE, "frame_count": params["frame_count"]}:
        raise RecheckError(f"extracted WAV format is not frozen PCM16/16k mono: {path}: {params}")
    audio = np.frombuffer(pcm, dtype="<i2").copy()
    if audio.size != params["frame_count"]:
        raise RecheckError(f"extracted WAV frame count is inconsistent: {path}")
    return pcm, audio, params


def _source_pcm(path: Path) -> bytes:
    pcm, _audio, params = _pcm_from_wav(path)
    if params["sample_rate"] != config.SAMPLE_RATE:
        raise RecheckError(f"frozen natural audio is not 16 kHz: {path}")
    return pcm


def _crop_visual_frame(
    image: np.ndarray,
    visual_box: tuple[int, int, int, int] | None,
    *,
    output_size: int = 224,
) -> np.ndarray:
    """Return the visual frame in the geometry expected by SyncNet.

    Historical callers score videos that are already 224x224 crops.  The
    phoneme/TFG association protocol renders into the original external
    frame, so it supplies the frozen detector box explicitly.  Keeping this
    conversion here makes the model input contract explicit without changing
    the legacy no-box path.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise RecheckError(f"visual frame must be uint8 BGR HxWx3, got {image.shape}/{image.dtype}")
    if visual_box is None:
        if image.shape[:2] != (output_size, output_size):
            raise RecheckError(
                f"unboxed SyncNet frame must be {output_size}x{output_size}, got {image.shape[:2]}"
            )
        return np.ascontiguousarray(image)
    top, bottom, left, right = (int(value) for value in visual_box)
    height, width = image.shape[:2]
    if not (0 <= top < bottom <= height and 0 <= left < right <= width):
        raise RecheckError(f"visual crop box is outside decoded frame: {visual_box} vs {image.shape[:2]}")
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise RecheckError(f"visual crop is empty: {visual_box}")
    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized)


def _extract_media(
    media: Path,
    source_audio: Path,
    output_dir: Path,
    visual_box: tuple[int, int, int, int] | None = None,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
    if not media.is_file() or not source_audio.is_file():
        raise RecheckError(f"missing scoring input: media={media}, audio={source_audio}")
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / "audio.wav"
    frame_pattern = str(frame_dir / "%06d.jpg")
    frame_command = [
        str(config.FFMPEG),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(media),
        "-threads",
        "1",
        "-f",
        "image2",
        frame_pattern,
    ]
    audio_command = [
        str(config.FFMPEG),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(media),
        "-async",
        "1",
        "-ac",
        "1",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(config.SAMPLE_RATE),
        str(audio_path),
    ]
    _run(frame_command)
    _run(audio_command)

    frame_paths = sorted(frame_dir.glob("*.jpg"))
    expected_names = [f"{index:06d}.jpg" for index in range(1, len(frame_paths) + 1)]
    if [path.name for path in frame_paths] != expected_names or not frame_paths:
        raise RecheckError(f"extracted JPEG sequence is missing or non-contiguous: {frame_dir}")
    frames: list[np.ndarray] = []
    decoded_shape: tuple[int, ...] | None = None
    shape: tuple[int, ...] | None = None
    for path in frame_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.dtype != np.uint8:
            raise RecheckError(f"cv2 could not read BGR JPEG: {path}")
        if decoded_shape is None:
            decoded_shape = image.shape
        if image.shape != decoded_shape:
            raise RecheckError(f"decoded frame shape changed at {path}: {image.shape} != {decoded_shape}")
        processed = _crop_visual_frame(image, visual_box)
        if shape is None:
            shape = processed.shape
        if processed.shape != shape:
            raise RecheckError(f"scoring frame shape changed at {path}: {processed.shape} != {shape}")
        frames.append(processed)

    raw_command = [
        str(config.FFMPEG),
        "-v",
        "error",
        "-i",
        str(media),
        "-map",
        "0:a:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(config.SAMPLE_RATE),
        "-ac",
        "1",
        "pipe:1",
    ]
    raw_result = subprocess.run(raw_command, capture_output=True, check=False)
    if raw_result.returncode != 0:
        raise RecheckError(f"raw PCM extraction failed: {media}")
    extracted_pcm, audio, audio_format = _pcm_from_wav(audio_path)
    if extracted_pcm != raw_result.stdout:
        raise RecheckError(f"WAV extraction differs from raw PCM extraction: {media}")
    frozen_pcm = _source_pcm(source_audio)
    if raw_result.stdout != frozen_pcm:
        raise RecheckError(f"decoded media PCM differs from frozen N PCM: {media}")
    metadata = {
        "frame_count": len(frames),
        "frame_shape": list(shape or ()),
        "decoded_frame_shape": list(decoded_shape or ()),
        "frame_dtype": "uint8",
        "frame_layout": "cv2 BGR, 0..255, fixed box crop + INTER_AREA resize" if visual_box is not None else "cv2 BGR, 0..255, no resize",
        "visual_box": list(visual_box) if visual_box is not None else None,
        "visual_crop_size": 224 if visual_box is not None else None,
        "frame_pattern": frame_pattern,
        "frame_command": frame_command,
        "audio_path": str(audio_path.resolve()),
        "audio_format": audio_format,
        "audio_command": audio_command,
        "raw_pcm_command": raw_command,
        "media_pcm_sha256": bytes_sha256(raw_result.stdout),
        "frozen_pcm_sha256": bytes_sha256(frozen_pcm),
        "extracted_pcm_verified": True,
        "frame_files_sha256": [file_sha256(path) for path in frame_paths],
    }
    return frames, audio, metadata


def _forward(
    model: S,
    frames: list[np.ndarray],
    audio: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(frames) < config.WINDOW_FRAMES:
        raise RecheckError("video has fewer than five frames")
    mfcc = np.asarray(python_speech_features.mfcc(audio, config.SAMPLE_RATE), dtype=np.float32).T
    sample_limit = min(len(frames), int(audio.size // config.SAMPLES_PER_FRAME))
    window_count = sample_limit - config.WINDOW_FRAMES
    if window_count < 1 or mfcc.shape[1] < (window_count - 1) * config.MFCC_STRIDE + config.MFCC_AUDIO_FRAMES:
        raise RecheckError(f"media does not support the required SyncNet windows: frames={len(frames)} mfcc={mfcc.shape}")

    image_stack = np.stack(frames, axis=3)
    image_stack = np.transpose(np.expand_dims(image_stack, axis=0), (0, 3, 4, 1, 2))
    image_tensor = torch.from_numpy(image_stack.astype(np.float32, copy=False))
    audio_tensor = torch.from_numpy(mfcc[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False))
    visual_batches: list[torch.Tensor] = []
    audio_batches: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, window_count, batch_size):
            stop = min(window_count, start + batch_size)
            image_batch = torch.cat([image_tensor[:, :, row : row + 5, :, :] for row in range(start, stop)], dim=0)
            audio_batch = torch.cat(
                [audio_tensor[:, :, :, row * config.MFCC_STRIDE : row * config.MFCC_STRIDE + config.MFCC_AUDIO_FRAMES] for row in range(start, stop)],
                dim=0,
            )
            visual_batches.append(model.forward_lip(image_batch.to(device)).detach().cpu().to(torch.float32))
            audio_batches.append(model.forward_aud(audio_batch.to(device)).detach().cpu().to(torch.float32))
    visual = torch.cat(visual_batches, dim=0).numpy().astype(np.float32, copy=False)
    audio_embedding = torch.cat(audio_batches, dim=0).numpy().astype(np.float32, copy=False)
    if visual.shape != (window_count, config.EMBEDDING_DIM) or audio_embedding.shape != (window_count, config.EMBEDDING_DIM):
        raise RecheckError(f"unexpected embedding shapes: visual={visual.shape}, audio={audio_embedding.shape}")
    if not np.isfinite(visual).all() or not np.isfinite(audio_embedding).all():
        raise RecheckError("SyncNet embeddings contain non-finite values")
    return visual, audio_embedding


def pairwise_distance(visual: np.ndarray, audio: np.ndarray, vshift: int = config.VSHIFT) -> np.ndarray:
    """Implement the frozen distance contract without SyncNetInstance.calc_pdist."""
    visual_tensor = torch.as_tensor(visual, dtype=torch.float32)
    audio_tensor = torch.as_tensor(audio, dtype=torch.float32)
    if visual_tensor.ndim != 2 or audio_tensor.ndim != 2 or visual_tensor.shape != audio_tensor.shape:
        raise RecheckError(f"embedding shape mismatch: visual={visual_tensor.shape}, audio={audio_tensor.shape}")
    if visual_tensor.shape[1] != config.EMBEDDING_DIM:
        raise RecheckError(f"embedding width is not {config.EMBEDDING_DIM}: {visual_tensor.shape}")
    padded = torch.nn.functional.pad(audio_tensor, (0, 0, vshift, vshift))
    rows: list[torch.Tensor] = []
    for row in range(visual_tensor.shape[0]):
        candidate = padded[row : row + 2 * vshift + 1]
        difference = visual_tensor[row : row + 1] - candidate
        rows.append(torch.sqrt(torch.sum((difference + config.EPSILON) ** 2, dim=1)))
    return torch.stack(rows, dim=0).to(torch.float32).numpy()


def load_model(model_path: Path, device: str, threads: int = config.TORCH_THREADS) -> tuple[S, torch.device]:
    if not model_path.is_file() or file_sha256(model_path) != config.SYNCNET_MODEL_SHA256:
        raise RecheckError(f"SyncNet model is missing or hash changed: {model_path}")
    torch.set_num_threads(int(threads))
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RecheckError("CUDA was requested but is unavailable")
    model = S(num_layers_in_fc_layers=config.EMBEDDING_DIM).to(selected_device)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, selected_device


class SyncNetScorer:
    def __init__(self, model_path: Path, device: str = "cpu", batch_size: int = config.BATCH_SIZE, threads: int = config.TORCH_THREADS):
        self.model_path = model_path
        self.batch_size = int(batch_size)
        self.threads = int(threads)
        self.model, self.device = load_model(model_path, device, threads)

    def score(
        self,
        media: Path,
        source_audio: Path,
        output_dir: Path,
        expected_media_sha256: str,
        expected_pcm_sha256: str,
        visual_box: tuple[int, int, int, int] | None = None,
    ) -> dict[str, Any]:
        if file_sha256(media) != expected_media_sha256:
            raise RecheckError(f"media hash changed before fresh scoring: {media}")
        frames, pcm_audio, extraction = _extract_media(media, source_audio, output_dir / "extract", visual_box)
        if extraction["media_pcm_sha256"] != expected_pcm_sha256:
            raise RecheckError(f"decoded PCM hash differs from frozen media binding: {media}")
        visual, audio_embedding = _forward(self.model, frames, pcm_audio, self.device, self.batch_size)
        distance = pairwise_distance(visual, audio_embedding)
        visual_path = output_dir / "visual.npy"
        audio_path = output_dir / "audio.npy"
        matrix_path = output_dir / "distance.npy"
        np.save(visual_path, visual, allow_pickle=False)
        np.save(audio_path, audio_embedding, allow_pickle=False)
        np.save(matrix_path, distance, allow_pickle=False)
        return {
            "schema_version": 1,
            "stage_id": "fresh_syncnet_forward",
            "media": str(media.resolve()),
            "media_sha256": file_sha256(media),
            "source_audio": str(source_audio.resolve()),
            "source_audio_sha256": file_sha256(source_audio),
            "model": str(self.model_path.resolve()),
            "model_sha256": config.SYNCNET_MODEL_SHA256,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "torch_threads": self.threads,
            "new_forward": True,
            "extraction": extraction,
            "visual": str(visual_path.resolve()),
            "visual_sha256": file_sha256(visual_path),
            "visual_shape": list(visual.shape),
            "audio_embedding": str(audio_path.resolve()),
            "audio_embedding_sha256": file_sha256(audio_path),
            "audio_embedding_shape": list(audio_embedding.shape),
            "matrix": str(matrix_path.resolve()),
            "matrix_sha256": file_sha256(matrix_path),
            "matrix_shape": list(distance.shape),
            "dtype": "float32",
            "distance_epsilon": config.EPSILON,
            "extracted_pcm_sha256": extraction["media_pcm_sha256"],
            "frozen_pcm_sha256": extraction["frozen_pcm_sha256"],
            "extracted_pcm_verified": extraction["extracted_pcm_verified"],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fresh independent SyncNet forward for one frozen media cell")
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--source-audio", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=config.SYNCNET_MODEL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--threads", type=int, default=config.TORCH_THREADS)
    parser.add_argument("--expected-media-sha256", required=True)
    parser.add_argument("--expected-pcm-sha256", required=True)
    args = parser.parse_args(argv)
    scorer = SyncNetScorer(args.model, args.device, args.batch_size, args.threads)
    result = scorer.score(args.media, args.source_audio, args.output_dir, args.expected_media_sha256, args.expected_pcm_sha256)
    write_self_hashed_json(args.output_dir / "worker.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
