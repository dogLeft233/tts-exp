"""Local Wav2Lip wrapper that accepts precomputed mel chunks."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


WAV2LIP_ROOT = Path(__file__).resolve().parents[3] / "third_party/Wav2Lip"
sys.path.insert(0, str(WAV2LIP_ROOT))


def chunk_mels(mel: np.ndarray, fps: float, mel_step_size: int = 16) -> list[np.ndarray]:
    values = np.asarray(mel)
    if values.ndim != 2 or values.shape[0] != 80 or not np.isfinite(values).all():
        raise ValueError(f"mel must have shape [80,N], got {values.shape}")
    if not math_is_positive(fps) or values.shape[1] < mel_step_size:
        raise ValueError("invalid FPS or mel length")
    chunks = []
    multiplier = 80.0 / float(fps)
    index = 0
    while True:
        start = int(index * multiplier)
        if start + mel_step_size > len(values[0]):
            chunks.append(values[:, len(values[0]) - mel_step_size:])
            break
        chunks.append(values[:, start:start + mel_step_size])
        index += 1
    return chunks


def math_is_positive(value: float) -> bool:
    return np.isfinite(value) and float(value) > 0


def load_frames(face_path: Path, *, resize_factor: int = 1, crop: Sequence[int] = (0, -1, 0, -1), rotate: bool = False) -> tuple[list[np.ndarray], float]:
    import cv2

    if not face_path.is_file():
        raise ValueError(f"face path is not a file: {face_path}")
    suffix = face_path.suffix.lower()
    if suffix in {".jpg", ".png", ".jpeg"}:
        frame = cv2.imread(str(face_path))
        if frame is None:
            raise ValueError(f"could not read face image: {face_path}")
        return [frame], 25.0
    stream = cv2.VideoCapture(str(face_path))
    fps = float(stream.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    try:
        while True:
            reading, frame = stream.read()
            if not reading:
                break
            if resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            y1, y2, x1, x2 = (int(value) for value in crop)
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]
            frames.append(frame[y1:y2, x1:x2])
    finally:
        stream.release()
    if not frames or not math_is_positive(fps):
        raise ValueError(f"face video has no usable frames or FPS: {face_path}")
    return frames, fps


def detect_boxes(frames: list[np.ndarray], face_det_batch_size: int, pads: Sequence[int], nosmooth: bool) -> list[list[int]]:
    import face_detection

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)
    predictions = []
    batch_size = int(face_det_batch_size)
    while True:
        try:
            predictions = []
            for start in range(0, len(frames), batch_size):
                predictions.extend(detector.get_detections_for_batch(np.asarray(frames[start:start + batch_size])))
            break
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError("face detection failed at batch size 1")
            batch_size //= 2
    pady1, pady2, padx1, padx2 = (int(value) for value in pads)
    boxes: list[list[int]] = []
    for rect, image in zip(predictions, frames):
        if rect is None:
            raise ValueError("Face not detected!")
        x1 = max(0, int(rect[0]) - padx1)
        y1 = max(0, int(rect[1]) - pady1)
        x2 = min(image.shape[1], int(rect[2]) + padx2)
        y2 = min(image.shape[0], int(rect[3]) + pady2)
        boxes.append([x1, y1, x2, y2])
    if not nosmooth:
        for index in range(len(boxes)):
            window = boxes[len(boxes) - 5:] if index + 5 > len(boxes) else boxes[index:index + 5]
            boxes[index] = [int(round(value)) for value in np.mean(np.asarray(window), axis=0)]
    del detector
    return boxes


def _load_model(checkpoint_path: Path, device: str):
    import torch
    from models import Wav2Lip

    model = Wav2Lip()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state = checkpoint["state_dict"]
    model.load_state_dict({key.replace("module.", ""): value for key, value in state.items()})
    return model.to(device).eval()


def render(args: argparse.Namespace) -> dict[str, Any]:
    import cv2
    import torch

    mel = np.asarray(np.load(args.mel, allow_pickle=False), dtype=np.float32)
    frames, fps = load_frames(Path(args.face), resize_factor=args.resize_factor, crop=args.crop, rotate=args.rotate)
    chunks = chunk_mels(mel, fps)
    frames = frames[:len(chunks)]
    if not frames:
        raise ValueError("mel/video intersection is empty")
    if args.boxes_input:
        boxes = json.loads(Path(args.boxes_input).read_text(encoding="utf-8"))
        boxes = [[int(value) for value in row] for row in boxes]
        if len(boxes) < len(frames):
            raise ValueError("cached face boxes do not cover rendered frames")
        boxes = boxes[:len(frames)]
    else:
        boxes = detect_boxes(frames, args.face_det_batch_size, args.pads, args.nosmooth)
        if args.boxes_output:
            Path(args.boxes_output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.boxes_output).write_text(json.dumps(boxes), encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(Path(args.checkpoint), device)
    frame_h, frame_w = frames[0].shape[:2]
    outfile = Path(args.outfile).resolve()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    avi = outfile.with_suffix(".avi")
    writer = cv2.VideoWriter(str(avi), cv2.VideoWriter_fourcc(*"DIVX"), fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer: {avi}")
    try:
        for start in range(0, len(chunks), args.wav2lip_batch_size):
            local_chunks = chunks[start:start + args.wav2lip_batch_size]
            local_indices = [(start + offset) % len(frames) for offset in range(len(local_chunks))]
            local_frames = [frames[index].copy() for index in local_indices]
            local_boxes = [boxes[index] for index in local_indices]
            faces = []
            for frame, (x1, y1, x2, y2) in zip(local_frames, local_boxes):
                face = cv2.resize(frame[y1:y2, x1:x2], (96, 96))
                faces.append(face)
            img_batch = np.asarray(faces)
            mel_batch = np.asarray(local_chunks)
            img_masked = img_batch.copy()
            img_masked[:, 48:] = 0
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            img_tensor = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_tensor = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
            with torch.no_grad():
                pred = model(mel_tensor, img_tensor)
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            for prediction, frame, (x1, y1, x2, y2) in zip(pred, local_frames, local_boxes):
                replacement = cv2.resize(prediction.astype(np.uint8), (x2 - x1, y2 - y1))
                frame[y1:y2, x1:x2] = replacement
                writer.write(frame)
    finally:
        writer.release()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(avi), "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(outfile)], check=True)
    avi.unlink(missing_ok=True)
    return {"outfile": str(outfile), "fps": fps, "mel_chunks": len(chunks), "frames_rendered": len(frames), "boxes": str(args.boxes_output or args.boxes_input or "")}


def parity(args: argparse.Namespace) -> dict[str, Any]:
    import audio
    import soundfile as sf

    waveform, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if int(sample_rate) != 16000 or np.asarray(waveform).ndim != 1 or len(waveform) < args.support_samples:
        raise ValueError("natural audio violates the frozen 16 kHz mono support contract")
    waveform = np.ascontiguousarray(np.asarray(waveform[:args.support_samples], dtype=np.float32))
    generated = np.asarray(audio.melspectrogram(waveform), dtype=np.float32)
    stored = np.asarray(np.load(args.mel, allow_pickle=False), dtype=np.float32)
    if generated.shape != stored.shape:
        raise ValueError(f"natural mel shape mismatch: generated={generated.shape} stored={stored.shape}")
    max_abs = float(np.max(np.abs(generated - stored)))
    chunks_generated = chunk_mels(generated, args.fps)
    chunks_stored = chunk_mels(stored, args.fps)
    if max_abs > 1e-6 or len(chunks_generated) != len(chunks_stored) or any(not np.array_equal(a, b) for a, b in zip(chunks_generated, chunks_stored)):
        raise ValueError(f"natural mel parity failed: max_abs={max_abs}")
    return {"status": "PASS", "shape": list(stored.shape), "max_abs": max_abs, "fps": float(args.fps), "chunk_count": len(chunks_stored)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    parity_parser = subparsers.add_parser("parity")
    parity_parser.add_argument("--audio", required=True)
    parity_parser.add_argument("--mel", required=True)
    parity_parser.add_argument("--fps", type=float, required=True)
    parity_parser.add_argument("--support-samples", type=int, default=61440)
    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--checkpoint", required=True)
    render_parser.add_argument("--face", required=True)
    render_parser.add_argument("--mel", required=True)
    render_parser.add_argument("--outfile", required=True)
    render_parser.add_argument("--boxes-input", default="")
    render_parser.add_argument("--boxes-output", default="")
    render_parser.add_argument("--face-det-batch-size", type=int, default=4)
    render_parser.add_argument("--wav2lip-batch-size", type=int, default=4)
    render_parser.add_argument("--pads", nargs=4, type=int, default=[0, 10, 0, 0])
    render_parser.add_argument("--nosmooth", action="store_true")
    render_parser.add_argument("--resize-factor", type=int, default=1)
    render_parser.add_argument("--crop", nargs=4, type=int, default=[0, -1, 0, -1])
    render_parser.add_argument("--rotate", action="store_true")
    args = parser.parse_args(argv)
    result = parity(args) if args.command == "parity" else render(args)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
