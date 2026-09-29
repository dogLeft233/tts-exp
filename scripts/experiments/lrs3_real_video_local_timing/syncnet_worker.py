from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SYNCNET_ROOT = REPO / "third_party/syncnet_python"
if str(SYNCNET_ROOT) not in sys.path:
    sys.path.insert(0, str(SYNCNET_ROOT))

from SyncNetInstance import SyncNetInstance


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def media_pcm(path: Path) -> bytes:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "pipe:1"],
        capture_output=True,
        check=True,
    )
    return bytes(result.stdout)


def extracted_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        return handle.readframes(handle.getnframes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the official SyncNet V2 forward scorer and export its full distance matrix")
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--vshift", type=int, default=15)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
    args.matrix.parent.mkdir(parents=True, exist_ok=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    input_pcm = media_pcm(args.media)
    scorer = SyncNetInstance()
    scorer.loadParameters(str(args.model))
    options = SimpleNamespace(
        tmp_dir=str(args.tmp_dir),
        reference=args.reference,
        batch_size=int(args.batch_size),
        vshift=int(args.vshift),
    )
    offset, confidence, distances = scorer.evaluate(options, str(args.media))
    matrix = np.asarray(distances, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 31 or not np.isfinite(matrix).all():
        raise ValueError(f"SyncNet returned an invalid matrix: {matrix.shape}")
    np.save(args.matrix, matrix, allow_pickle=False)
    extracted_path = args.tmp_dir / args.reference / "audio.wav"
    actual_pcm = extracted_pcm(extracted_path)
    if actual_pcm != input_pcm:
        raise ValueError("SyncNet's extracted PCM differs from the muxed media PCM")
    payload = {
        "schema_version": 1,
        "media": str(args.media.resolve()),
        "media_sha256": file_sha256(args.media),
        "model": str(args.model.resolve()),
        "tmp_dir": str(args.tmp_dir.resolve()),
        "reference": args.reference,
        "vshift": int(args.vshift),
        "batch_size": int(args.batch_size),
        "offset": int(np.asarray(offset).item()),
        "confidence": float(np.asarray(confidence).item()),
        "matrix_shape": [int(value) for value in matrix.shape],
        "matrix_sha256": file_sha256(args.matrix),
        "input_pcm_sha256": hashlib.sha256(input_pcm).hexdigest(),
        "extracted_pcm_sha256": hashlib.sha256(actual_pcm).hexdigest(),
        "extracted_pcm_verified": True,
    }
    args.result.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
