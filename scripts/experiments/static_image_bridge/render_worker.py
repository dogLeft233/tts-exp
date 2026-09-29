from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
WAV2LIP_ROOT = REPO / "third_party/Wav2Lip"
sys.path.insert(0, str(WAV2LIP_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encode_ffv1_stream(frames: object, *, width: int, height: int, output: Path, ffmpeg: Path) -> dict[str, object]:
    temporary = output.with_name(f".{output.name}.partial")
    command = [
        str(ffmpeg), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", "25", "-i", "pipe:0", "-an", "-c:v", "ffv1",
        "-level", "3", "-g", "1", "-pix_fmt", "bgr0", "-f", "matroska", str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    digest = hashlib.sha256()
    count = 0
    try:
        assert process.stdin is not None
        for frame in frames:
            value = np.ascontiguousarray(frame)
            if value.shape != (height, width, 3) or value.dtype != np.uint8:
                raise ValueError("streamed frame shape/dtype mismatch")
            raw = value.tobytes()
            digest.update(raw)
            process.stdin.write(raw)
            count += 1
        process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr is not None else ""
        returncode = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        temporary.unlink(missing_ok=True)
        raise
    if returncode != 0 or count == 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"FFV1 encoding failed: {stderr[-1000:]}")
    temporary.replace(output)
    return {"command": command, "raw_frame_sha256": digest.hexdigest(), "frame_count": count, "width": width, "height": height}


def chunk_mels(mel: np.ndarray, fps: float = 25.0) -> list[np.ndarray]:
    if mel.ndim != 2 or mel.shape[0] != 80 or mel.shape[1] < 16 or not np.isfinite(mel).all():
        raise ValueError(f"invalid Wav2Lip mel shape: {mel.shape}")
    multiplier = 80.0 / fps
    chunks: list[np.ndarray] = []
    index = 0
    while True:
        start = int(index * multiplier)
        if start + 16 > mel.shape[1]:
            chunks.append(mel[:, mel.shape[1] - 16:])
            break
        chunks.append(mel[:, start:start + 16])
        index += 1
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--image-rgb-sha256", required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--box", nargs=4, type=int, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    import audio
    import torch

    import models
    from models import Wav2Lip

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Wav2Lip static render requested CUDA, but CUDA is unavailable")
    device = str(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"cannot read static image: {args.image}")
    actual_rgb_hash = sha256_bytes(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())
    if actual_rgb_hash != args.image_rgb_sha256:
        raise ValueError("static image RGB hash does not match the frozen input")
    x1, y1, x2, y2 = [int(value) for value in args.box]
    if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
        raise ValueError("generation box is outside the static image")
    wav = audio.load_wav(str(args.audio), 16000)
    mel = np.asarray(audio.melspectrogram(wav), dtype=np.float32)
    chunks = chunk_mels(mel)
    if not chunks:
        raise ValueError("audio produced no mel chunks")

    checkpoint_sha256 = sha256_file(args.checkpoint)
    model = Wav2Lip()
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu")
    state = checkpoint["state_dict"]
    model.load_state_dict({key.replace("module.", ""): value for key, value in state.items()})
    model = model.to(device).eval()
    parameter_digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        parameter_digest.update(name.encode("utf-8"))
        parameter_digest.update(str(tensor.dtype).encode("ascii"))
        parameter_digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        parameter_digest.update(tensor.numpy().tobytes())
    def frame_stream():
        for start in range(0, len(chunks), max(1, int(args.batch_size))):
            local_chunks = chunks[start:start + max(1, int(args.batch_size))]
            faces = [cv2.resize(frame[y1:y2, x1:x2], (96, 96), interpolation=cv2.INTER_LINEAR) for _ in local_chunks]
            images = np.asarray(faces, dtype=np.uint8)
            masked = images.copy()
            masked[:, 48:] = 0
            image_batch = np.concatenate((masked, images), axis=3) / 255.0
            mel_batch = np.asarray(local_chunks, dtype=np.float32)[:, :, :, None]
            image_tensor = torch.FloatTensor(np.transpose(image_batch, (0, 3, 1, 2))).to(device)
            mel_tensor = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
            with torch.no_grad():
                prediction = model(mel_tensor, image_tensor).detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.0
            for pred in prediction:
                rendered = frame.copy()
                replacement = cv2.resize(pred.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                rendered[y1:y2, x1:x2] = replacement
                yield np.ascontiguousarray(rendered)
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    encode = encode_ffv1_stream(frame_stream(), width=int(frame.shape[1]), height=int(frame.shape[0]), output=args.outfile, ffmpeg=args.ffmpeg)
    capture = cv2.VideoCapture(str(args.outfile))
    decoded_digest = hashlib.sha256()
    decoded_count = 0
    decoded_fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    decoded_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outside_mask = np.ones(frame.shape[:2], dtype=bool)
    outside_mask[y1:y2, x1:x2] = False
    while True:
        ok, decoded = capture.read()
        if not ok:
            break
        if decoded.shape != frame.shape or not np.array_equal(decoded[outside_mask], frame[outside_mask]):
            capture.release()
            raise RuntimeError("decoded static video changed pixels outside generation box")
        decoded_digest.update(np.ascontiguousarray(decoded).tobytes())
        decoded_count += 1
    capture.release()
    if decoded_count != int(encode["frame_count"]) or decoded_width != int(frame.shape[1]) or decoded_height != int(frame.shape[0]) or abs(decoded_fps - 25.0) > 0.01:
        raise RuntimeError(f"decoded static video contract mismatch: frames={decoded_count}, fps={decoded_fps}, size={decoded_width}x{decoded_height}")
    result = {
        "schema_version": 1,
        "status": "complete",
        "device": device,
        "seed": int(args.seed),
        "input_mode": "one_png_only",
        "image": str(args.image.resolve()),
        "image_rgb_sha256": actual_rgb_hash,
        "audio": str(args.audio.resolve()),
        "audio_sha256": sha256_file(args.audio),
        "box_xyxy": [x1, y1, x2, y2],
        "source_frame_indices": [0] * int(encode["frame_count"]),
        "frames_rendered": int(encode["frame_count"]),
        "fps": 25.0,
        "mel_shape": [int(value) for value in mel.shape],
        "mel_chunks": len(chunks),
        "output": str(args.outfile.resolve()),
        "output_sha256": sha256_file(args.outfile),
        "output_codec": "ffv1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "loaded_parameter_sha256": parameter_digest.hexdigest(),
        "render_worker": str(Path(__file__).resolve()),
        "render_worker_sha256": sha256_file(Path(__file__).resolve()),
        "python_executable": str(Path(sys.executable).absolute()),
        "python_realpath": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "opencv_version": str(cv2.__version__),
        "audio_module": str(Path(audio.__file__).resolve()),
        "audio_module_sha256": sha256_file(Path(audio.__file__).resolve()),
        "models_module": str(Path(models.__file__).resolve()),
        "models_module_sha256": sha256_file(Path(models.__file__).resolve()),
        "wav2lip_class_source": str(Path(inspect.getsourcefile(Wav2Lip) or "").resolve()),
        "wav2lip_class_source_sha256": sha256_file(Path(inspect.getsourcefile(Wav2Lip) or "")),
        "encode": encode,
        "decoded_frame_sha256": decoded_digest.hexdigest(),
        "decoded_frame_count": decoded_count,
        "decoded_fps": decoded_fps,
        "outside_generation_box_identity": True,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "device": device, "frames": int(encode["frame_count"]), "output": str(args.outfile)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
