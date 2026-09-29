from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def media_pcm(path: Path, ffmpeg: Path) -> bytes:
    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-i",
        str(path),
        "-async",
        "1",
        "-ac",
        "1",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-f",
        "s16le",
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        check=True,
    )
    return bytes(result.stdout)


def extracted_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as handle:
        return handle.readframes(handle.getnframes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the official SyncNet V2 [T,31] distance matrix for one fixed crop")
    parser.add_argument("--media", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--vshift", type=int, default=15)
    parser.add_argument("--ffmpeg", type=Path, default=Path("ffmpeg"))
    args = parser.parse_args(argv)
    ffmpeg = args.ffmpeg if args.ffmpeg.is_file() else Path(shutil.which(str(args.ffmpeg)) or "")
    if not ffmpeg.is_file():
        raise FileNotFoundError(f"ffmpeg executable is missing: {args.ffmpeg}")
    args.matrix.parent.mkdir(parents=True, exist_ok=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    input_pcm = media_pcm(args.media, ffmpeg)
    scorer = SyncNetInstance()
    scorer.loadParameters(str(args.model))
    options = SimpleNamespace(tmp_dir=str(args.tmp_dir), reference=args.reference, batch_size=int(args.batch_size), vshift=int(args.vshift))
    offset, confidence, distances = scorer.evaluate(options, str(args.media))
    matrix = np.asarray(distances, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 31 or matrix.shape[0] < 1 or not np.isfinite(matrix).all():
        raise ValueError(f"SyncNet returned an invalid matrix: {matrix.shape}")
    np.save(args.matrix, matrix, allow_pickle=False)
    extracted_path = args.tmp_dir / args.reference / "audio.wav"
    actual_pcm = extracted_pcm(extracted_path)
    if actual_pcm != input_pcm:
        raise ValueError("SyncNet extracted PCM differs from the crop media PCM")
    payload = {
        "schema_version": 1,
        "media": str(args.media.resolve()),
        "media_sha256": sha256_file(args.media),
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "worker_code_sha256": sha256_file(Path(__file__)),
        "syncnet_instance_sha256": sha256_file(SYNCNET_ROOT / "SyncNetInstance.py"),
        "syncnet_model_definition_sha256": sha256_file(SYNCNET_ROOT / "SyncNetModel.py"),
        "ffmpeg": str(ffmpeg.resolve()),
        "ffmpeg_sha256": sha256_file(ffmpeg),
        "tmp_dir": str(args.tmp_dir.resolve()),
        "reference": args.reference,
        "vshift": int(args.vshift),
        "batch_size": int(args.batch_size),
        "offset": int(np.asarray(offset).item()),
        "confidence": float(np.asarray(confidence).item()),
        "matrix_shape": [int(value) for value in matrix.shape],
        "matrix_dtype": str(matrix.dtype),
        "matrix_sha256": sha256_file(args.matrix),
        "input_pcm_sha256": hashlib.sha256(input_pcm).hexdigest(),
        "extracted_pcm_sha256": hashlib.sha256(actual_pcm).hexdigest(),
        "extracted_pcm_verified": True,
        "new_forward": True,
    }
    args.result.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
