from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
SYNCNET_ROOT = REPO / "third_party/syncnet_python"
sys.path.insert(0, str(SYNCNET_ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def crop_zero_padded(frame: np.ndarray, box: list[int], output_size: int = 224) -> np.ndarray:
    x1, y1, x2, y2 = [int(value) for value in box]
    side = x2 - x1
    if side <= 0 or y2 - y1 != side:
        raise ValueError("score crop box must be a positive square")
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
    if sx2 > sx1 and sy2 > sy1:
        canvas[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = frame[sy1:sy2, sx1:sx2]
    return cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def read_video(path: Path, score_box: dict[str, object]) -> tuple[np.ndarray, list[str], dict[str, object]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open generated video: {path}")
    frames: list[np.ndarray] = []
    hashes: list[str] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cropped = crop_zero_padded(frame, [int(value) for value in score_box["box"]])
            frames.append(cropped)
            hashes.append(sha256_bytes(cropped.tobytes()))
    finally:
        capture.release()
    if len(frames) < 25:
        raise ValueError(f"generated video has insufficient frames: {path}")
    metadata_capture = cv2.VideoCapture(str(path))
    try:
        fps = float(metadata_capture.get(cv2.CAP_PROP_FPS))
    finally:
        metadata_capture.release()
    if abs(fps - 25.0) > 0.01:
        raise ValueError(f"generated video is not 25 fps: {path} ({fps})")
    return np.stack(frames, axis=0), hashes, {"frame_count": len(frames), "fps": fps, "crop_box": score_box}


def audio_embedding(path: Path, scorer: object, device: str, torch: object) -> tuple[object, int, int]:
    import python_speech_features
    from scipy.io import wavfile

    rate, audio = wavfile.read(str(path))
    if int(rate) != 16000 or np.asarray(audio).ndim != 1:
        raise ValueError(f"audio is not mono 16 kHz PCM: {path}")
    mfcc = np.asarray(list(zip(*python_speech_features.mfcc(audio, rate))), dtype=np.float32)
    if mfcc.ndim != 2 or mfcc.shape[0] != 13:
        raise ValueError(f"unexpected MFCC shape: {mfcc.shape}")
    count = (mfcc.shape[1] - 20) // 4 + 1
    if count <= 0:
        raise ValueError(f"audio has insufficient frontend support: {path}")
    batches = []
    batch_size = 20
    for start in range(0, count, batch_size):
        values = [mfcc[:, index * 4:index * 4 + 20] for index in range(start, min(count, start + batch_size))]
        tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)[:, None, :, :]).to(device)
        with torch.no_grad():
            batches.append(scorer.__S__.forward_aud(tensor).detach().cpu())
    return torch.cat(batches, dim=0), int(count), len(audio)


def visual_embedding(cropped: np.ndarray, scorer: object, device: str, torch: object) -> tuple[object, int]:
    count = cropped.shape[0] - 4
    batches = []
    for start in range(0, count, 20):
        values = cropped[start:min(count, start + 20)]
        tensor = torch.from_numpy(np.transpose(values, (0, 4, 1, 2, 3)) if values.ndim == 5 else np.transpose(values, (0, 3, 1, 2))).to(device)
        # SyncNet lip expects [N, 3, 5, 224, 224].
        if values.ndim == 4:
            sequences = np.stack([cropped[index:index + 5] for index in range(start, min(count, start + 20))], axis=0)
            tensor = torch.from_numpy(np.transpose(sequences, (0, 4, 1, 2, 3))).float().to(device)
        with torch.no_grad():
            batches.append(scorer.__S__.forward_lip(tensor).detach().cpu())
    return torch.cat(batches, dim=0), int(count)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-arm", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--score-box", type=Path, required=True)
    parser.add_argument("--audio-json", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vshift", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    import torch
    from SyncNetInstance import SyncNetInstance

    if not torch.cuda.is_available():
        raise RuntimeError("SyncNet scoring requires CUDA; CPU fallback is forbidden")
    device = "cuda"
    score_box = json.loads(args.score_box.read_text(encoding="utf-8"))
    if not isinstance(score_box, dict):
        raise TypeError("score box must be an object")
    cropped, crop_hashes, video_meta = read_video(args.video, score_box)
    scorer = SyncNetInstance(device=device)
    scorer.loadParameters(str(args.model))
    scorer.eval()
    v_count = cropped.shape[0] - 4
    v_batches = []
    for start in range(0, v_count, max(1, int(args.batch_size))):
        sequences = np.stack([cropped[index:index + 5] for index in range(start, min(v_count, start + max(1, int(args.batch_size))))], axis=0)
        tensor = torch.from_numpy(np.transpose(sequences, (0, 4, 1, 2, 3))).float().to(device)
        with torch.no_grad():
            v_batches.append(scorer.__S__.forward_lip(tensor).detach().cpu())
    visual = torch.cat(v_batches, dim=0)
    audio_map = json.loads(args.audio_json.read_text(encoding="utf-8"))
    if not isinstance(audio_map, dict) or not audio_map:
        raise ValueError("audio-json must contain at least one audio arm")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_path = args.output_dir / f"{args.video_arm}__visual.npy"
    np.save(visual_path, visual.numpy().astype(np.float64), allow_pickle=False)
    rows = []
    for audio_arm, audio_value in audio_map.items():
        audio_path = Path(str(audio_value))
        aud, a_count, audio_samples = audio_embedding(audio_path, scorer, device, torch)
        rows_for_audio = min(int(v_count), int(a_count))
        matrix = np.full((rows_for_audio, 2 * int(args.vshift) + 1), np.nan, dtype=np.float64)
        for t in range(rows_for_audio):
            for column, lag in enumerate(range(-int(args.vshift), int(args.vshift) + 1)):
                index = t + lag
                if 0 <= index < a_count:
                    matrix[t, column] = float(torch.nn.functional.pairwise_distance(visual[t:t + 1].float(), aud[index:index + 1].float()).item())
        matrix_path = args.output_dir / f"{args.video_arm}__{audio_arm}__matrix.npy"
        np.save(matrix_path, matrix, allow_pickle=False)
        row = {
            "schema_version": 1,
            "status": "complete",
            "sample_id": args.sample_id,
            "video_arm": args.video_arm,
            "audio_arm": str(audio_arm),
            "video": str(args.video.resolve()),
            "video_sha256": sha256_file(args.video),
            "audio": str(audio_path.resolve()),
            "audio_sha256": sha256_file(audio_path),
            "device": device,
            "model_sha256": sha256_file(args.model),
            "vshift": int(args.vshift),
            "matrix": str(matrix_path.resolve()),
            "matrix_sha256": sha256_file(matrix_path),
            "matrix_shape": [int(value) for value in matrix.shape],
            "finite_mask": np.isfinite(matrix).all(axis=1).astype(np.uint8).tolist(),
            "visual": str(visual_path.resolve()),
            "visual_sha256": sha256_file(visual_path),
            "visual_count": int(v_count),
            "audio_feature_count": int(a_count),
            "audio_sample_count": int(audio_samples),
            "video_meta": video_meta,
            "score_crop_hashes": crop_hashes,
        }
        result_path = args.output_dir / f"{args.video_arm}__{audio_arm}__worker.json"
        result_path.write_text(json.dumps(row, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        row["worker_result"] = str(result_path.resolve())
        row["worker_result_sha256"] = sha256_file(result_path)
        rows.append(row)
    print(json.dumps({"status": "PASS", "device": device, "video_arm": args.video_arm, "cells": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
