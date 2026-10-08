#!/usr/bin/env python3
"""Invoke the pinned official SyncNet evaluator on a crop-video/PCM16 MKV."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--syncnet-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-mkv", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--expected-pcm16-sha256", required=True)
    parser.add_argument("--expected-sample-count", type=int, required=True)
    parser.add_argument("--expected-video-frame-count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--vshift", type=int, default=15)
    args = parser.parse_args(argv)

    import numpy as np
    import torch
    from scipy.io import wavfile

    syncnet_root = args.syncnet_root.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    input_mkv = args.input_mkv.resolve(strict=True)
    tmp_dir = args.tmp_dir.resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.output_npz.exists() or args.output_json.exists():
        raise FileExistsError("refusing to overwrite SyncNet evaluator outputs")
    if len(args.expected_pcm16_sha256) != 64:
        raise ValueError("expected PCM16 hash is malformed")
    sys.path.insert(0, str(syncnet_root))
    torch.set_num_threads(1)
    from SyncNetInstance import SyncNetInstance  # imported only after root and runtime are pinned

    net = SyncNetInstance(device="cuda" if torch.cuda.is_available() else "cpu")
    net.loadParameters(str(checkpoint))
    net.eval()
    opt = SimpleNamespace(
        tmp_dir=str(tmp_dir),
        reference=args.reference,
        batch_size=int(args.batch_size),
        vshift=int(args.vshift),
    )
    offset, confidence, distance = net.evaluate(opt, str(input_mkv))
    matrix = np.ascontiguousarray(np.asarray(distance, dtype=np.float32))
    if matrix.ndim != 2 or matrix.shape[1] != 2 * args.vshift + 1:
        raise RuntimeError(f"official evaluator returned unexpected distance shape {matrix.shape}")
    extracted_audio = tmp_dir / args.reference / "audio.wav"
    rate, audio = wavfile.read(str(extracted_audio), mmap=False)
    audio = np.ascontiguousarray(audio)
    if rate != 16000 or audio.ndim != 1 or audio.dtype != np.dtype("int16"):
        raise RuntimeError(f"official evaluator decoded audio format mismatch: {rate}, {audio.dtype}, {audio.shape}")
    audio_le = audio.astype("<i2", copy=False).tobytes()
    actual_audio_sha = sha256_bytes(audio_le)
    if actual_audio_sha != args.expected_pcm16_sha256 or audio.size != args.expected_sample_count:
        raise RuntimeError(
            "official SyncNetInstance.evaluate FFmpeg extraction differs from the primary direct-source PCM16 window: "
            f"sha={actual_audio_sha}, samples={audio.size}, expected_sha={args.expected_pcm16_sha256}, "
            f"expected_samples={args.expected_sample_count}"
        )
    frame_files = sorted((tmp_dir / args.reference).glob("*.jpg"))
    if len(frame_files) != args.expected_video_frame_count:
        raise RuntimeError(
            f"official evaluator decoded {len(frame_files)} video frames; expected {args.expected_video_frame_count}"
        )
    expected_queries = max(0, min(len(frame_files), int(audio.size) // 640) - 5)
    if matrix.shape[0] != expected_queries:
        raise RuntimeError(
            f"official query count mismatch: evaluator={matrix.shape[0]}, official rule={expected_queries}"
        )
    np.savez_compressed(
        args.output_npz,
        distance_by_query_lag=matrix.astype("<f4", copy=False),
        lag_offsets_frames=np.arange(-args.vshift, args.vshift + 1, dtype="<i4"),
    )
    receipt = {
        "schema_version": "cross_tfg_official_syncnet_evaluation_v1",
        "status": "PASS",
        "evaluator": "SyncNetInstance.evaluate",
        "syncnet_root": str(syncnet_root),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "input_mkv_path": str(input_mkv),
        "input_mkv_sha256": sha256_file(input_mkv),
        "torch_version": str(torch.__version__),
        "torch_device": str(net.device),
        "torch_num_threads": int(torch.get_num_threads()),
        "video_frame_count": len(frame_files),
        "audio_sample_rate_hz": int(rate),
        "audio_sample_count": int(audio.size),
        "audio_pcm16_sha256": actual_audio_sha,
        "expected_query_count": expected_queries,
        "matrix_shape": list(matrix.shape),
        "matrix_dtype": str(matrix.dtype),
        "matrix_sha256": sha256_bytes(matrix.astype("<f4", copy=False).tobytes(order="C")),
        "offset_numpy": np.asarray(offset).tolist(),
        "confidence_numpy": np.asarray(confidence).tolist(),
        "distance_npz_path": str(args.output_npz.resolve()),
        "distance_npz_sha256": sha256_file(args.output_npz),
        "extracted_audio_path": str(extracted_audio),
        "extracted_audio_file_sha256": sha256_file(extracted_audio),
    }
    args.output_json.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
