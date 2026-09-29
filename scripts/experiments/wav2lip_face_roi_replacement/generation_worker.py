from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
ROOT = Path(__file__).resolve().parents[3] / "third_party/Wav2Lip"
sys.path.insert(0, str(ROOT))

import audio
import cv2
import numpy as np
import torch

from models import Wav2Lip


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError(f"invalid source frame: {path}")
            frames.append(np.ascontiguousarray(frame))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"source video has no frames: {path}")
    return frames


def mel_chunks(path: Path) -> tuple[list[np.ndarray], dict[str, object]]:
    waveform = audio.load_wav(str(path), 16000)
    mel = np.asarray(audio.melspectrogram(waveform), dtype=np.float32)
    if mel.ndim != 2 or mel.shape[0] != 80 or not np.isfinite(mel).all():
        raise RuntimeError(f"invalid Wav2Lip mel: {path}")
    chunks: list[np.ndarray] = []
    index = 0
    while True:
        start = int(index * 80.0 / 25.0)
        if start + 16 > len(mel[0]):
            chunks.append(mel[:, len(mel[0]) - 16:])
            break
        chunks.append(mel[:, start : start + 16])
        index += 1
    return chunks, {"sample_count": int(np.asarray(waveform).size), "mel_shape": [int(x) for x in mel.shape], "mel_chunk_count": len(chunks)}


def encode_video(frames: list[np.ndarray], output: Path, ffmpeg: Path) -> dict[str, object]:
    height, width = frames[0].shape[:2]
    raw = b"".join(np.ascontiguousarray(frame).tobytes() for frame in frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    command = [str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", "25", "-i", "pipe:0", "-an", "-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "bgr0", "-f", "matroska", str(temporary)]
    result = __import__("subprocess").run(command, input=raw, capture_output=True, check=False)
    if result.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"FFV1 encoding failed for {output}: {result.stderr.decode('utf-8', 'replace')[-500:]}")
    temporary.replace(output)
    return {"output": str(output.resolve()), "output_sha256": file_sha256(output), "frame_count": len(frames), "width": width, "height": height, "raw_frame_sha256": hashlib.sha256(raw).hexdigest(), "command": command, "command_sha256": canonical_hash(command), "codec": "ffv1"}


def render_arm(model: Wav2Lip, frames: list[np.ndarray], boxes: list[list[int]], chunks: list[np.ndarray], batch_size: int, device: str) -> list[np.ndarray]:
    if len(chunks) > len(frames) or len(boxes) < len(chunks):
        raise RuntimeError(f"mel frame count {len(chunks)} exceeds source/box count")
    result: list[np.ndarray] = []
    for start in range(0, len(chunks), batch_size):
        chunk_batch = chunks[start : start + batch_size]
        images: list[np.ndarray] = []
        frame_batch: list[np.ndarray] = []
        for offset in range(len(chunk_batch)):
            index = start + offset
            top, bottom, left, right = (int(value) for value in boxes[index])
            frame = frames[index].copy()
            face = frame[top:bottom, left:right]
            if face.size == 0:
                raise RuntimeError(f"empty ROI at frame {index}")
            face = cv2.resize(face, (96, 96))
            masked = face.copy()
            masked[48:] = 0
            images.append(np.concatenate((masked, face), axis=2).astype(np.float32) / 255.0)
            frame_batch.append(frame)
        image_tensor = torch.from_numpy(np.transpose(np.asarray(images), (0, 3, 1, 2))).to(device=device, dtype=torch.float32)
        mel_tensor = torch.from_numpy(np.asarray(chunk_batch)[:, None, :, :]).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            prediction = model(mel_tensor, image_tensor).detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        for offset, predicted in enumerate(prediction):
            index = start + offset
            top, bottom, left, right = (int(value) for value in boxes[index])
            frame = frame_batch[offset]
            patch = cv2.resize(predicted.astype(np.uint8), (right - left, bottom - top))
            frame[top:bottom, left:right] = patch
            result.append(frame)
    return result


def load_model(path: Path, device: str) -> Wav2Lip:
    model = Wav2Lip()
    checkpoint = torch.load(str(path), map_location="cpu")
    state = checkpoint["state_dict"]
    model.load_state_dict({key.replace("module.", ""): value for key, value in state.items()})
    return model.to(device).eval()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--face", required=True)
    parser.add_argument("--boxes", required=True)
    parser.add_argument("--audio-json", required=True)
    parser.add_argument("--outputs-json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--expected-frame-count", type=int, required=True)
    args = parser.parse_args()
    face = Path(args.face)
    boxes_path = Path(args.boxes)
    box_payload = json.loads(boxes_path.read_text(encoding="utf-8"))
    boxes = box_payload.get("boxes")
    if not isinstance(boxes, list) or len(boxes) < args.expected_frame_count:
        raise RuntimeError("ROI cache is incomplete")
    frames = load_frames(face)
    if len(frames) < args.expected_frame_count:
        raise RuntimeError("source video has fewer frames than expected")
    audio_paths = json.loads(Path(args.audio_json).read_text(encoding="utf-8"))
    if not isinstance(audio_paths, dict) or not audio_paths:
        raise RuntimeError("audio plan is empty")
    all_chunks: dict[str, list[np.ndarray]] = {}
    metadata: dict[str, dict[str, object]] = {}
    for arm, value in audio_paths.items():
        chunks, info = mel_chunks(Path(str(value)))
        if len(chunks) != args.expected_frame_count:
            raise RuntimeError(f"predicted frame count changed for {arm}: {len(chunks)} != {args.expected_frame_count}")
        all_chunks[str(arm)] = chunks
        metadata[str(arm)] = info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.batch_size != 4:
        raise RuntimeError("registered Wav2Lip batch size is 4")
    model = load_model(Path(args.checkpoint), device)
    outputs = json.loads(Path(args.outputs_json).read_text(encoding="utf-8"))
    if not isinstance(outputs, dict):
        raise TypeError("outputs plan is malformed")
    rows: dict[str, object] = {}
    for arm, chunks in all_chunks.items():
        output = Path(str(outputs[arm]))
        if output.exists():
            raise RuntimeError(f"output already exists: {output}")
        generated = render_arm(model, frames, [[int(value) for value in box] for box in boxes], chunks, 4, device)
        encoded = encode_video(generated, output, Path(args.ffmpeg))
        rows[arm] = {**encoded, "arm": arm, "audio": str(Path(str(audio_paths[arm])).resolve()), "audio_sha256": file_sha256(Path(str(audio_paths[arm]))), "face": str(face.resolve()), "face_sha256": file_sha256(face), "boxes": str(boxes_path.resolve()), "boxes_sha256": file_sha256(boxes_path), "checkpoint_sha256": file_sha256(Path(args.checkpoint)), "predicted_frame_count": len(chunks), "mel": metadata[arm], "device": device, "batch_size": 4}
    body = {"schema_version": 1, "stage_id": "videos", "protocol_id": "wav2lip_face_roi_replacement", "status": "complete", "source_video": str(face.resolve()), "source_video_sha256": file_sha256(face), "expected_frame_count": args.expected_frame_count, "rows": rows, "runtime": {"checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": file_sha256(Path(args.checkpoint)), "device": device, "batch_size": 4}}
    body["artifact_sha256"] = canonical_hash(body)
    output_manifest = Path(args.outputs_json).with_name("generation_result.json")
    output_manifest.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(body, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
