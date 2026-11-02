#!/usr/bin/env python3
"""Reproduce the frozen direct-natural PCM16 window used by Stage A.

The full canonical natural float32 WAV is decoded with the pinned FFmpeg build
to mono 16 kHz signed little-endian PCM16. Only then is the half-open
source-clock sample window sliced. No audio is padded or retimed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import wave
from pathlib import Path
from typing import Sequence

ARGV_TEMPLATE = [
    "-v", "error", "-i", "{natural_pcm_path}", "-map", "0:a:0", "-vn",
    "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1",
]
OPERATION = "ffmpeg_decode_full_source_then_slice_sample_window"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def decode_and_slice(
    natural_pcm_path: Path,
    *,
    start_sample: int,
    stop_sample_exclusive: int,
    ffmpeg: str = "/usr/bin/ffmpeg",
) -> tuple[bytes, dict[str, object]]:
    if start_sample < 0 or stop_sample_exclusive <= start_sample:
        raise ValueError("PCM16 window must be a nonempty nonnegative half-open sample range")
    path = natural_pcm_path.resolve(strict=True)
    argv = [
        ffmpeg, "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn",
        "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1",
    ]
    proc = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False)
    if proc.returncode:
        raise RuntimeError(
            f"FFmpeg full-source PCM16 conversion failed ({proc.returncode}): "
            f"{proc.stderr.decode('utf-8', errors='replace')[-4000:]}"
        )
    full = proc.stdout
    if len(full) % 2:
        raise RuntimeError("FFmpeg PCM16 output ended on a partial sample")
    source_count = len(full) // 2
    if stop_sample_exclusive > source_count:
        raise ValueError("requested PCM16 window exceeds the decoded source; caller must clip explicitly")
    window = full[start_sample * 2:stop_sample_exclusive * 2]
    if len(window) != (stop_sample_exclusive - start_sample) * 2:
        raise RuntimeError("sliced PCM16 byte count differs from the requested sample window")
    return window, {
        "source_path": str(path),
        "source_file_sha256": sha256_file(path),
        "full_source_pcm16_sha256": sha256_bytes(full),
        "full_source_sample_count": source_count,
        "start_sample": start_sample,
        "stop_sample_exclusive": stop_sample_exclusive,
        "sample_count": len(window) // 2,
        "pcm_sha256": sha256_bytes(window),
        "argv": argv,
        "argv_template": ARGV_TEMPLATE,
        "operation": OPERATION,
    }


def write_pcm16_wav(path: Path, pcm16_bytes: bytes) -> dict[str, object]:
    if not pcm16_bytes or len(pcm16_bytes) % 2:
        raise ValueError("PCM16 WAV requires a nonempty even number of bytes")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm16_bytes)
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "pcm_sha256": sha256_bytes(pcm16_bytes),
        "sample_count": len(pcm16_bytes) // 2,
        "sample_rate_hz": 16000,
        "channels": 1,
        "codec": "pcm_s16le",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-pcm", type=Path, required=True)
    parser.add_argument("--start-sample", type=int, required=True)
    parser.add_argument("--stop-sample-exclusive", type=int, required=True)
    parser.add_argument("--ffmpeg", default="/usr/bin/ffmpeg")
    parser.add_argument("--wav-out", type=Path, required=True)
    args = parser.parse_args(argv)
    data, lineage = decode_and_slice(
        args.natural_pcm,
        start_sample=args.start_sample,
        stop_sample_exclusive=args.stop_sample_exclusive,
        ffmpeg=args.ffmpeg,
    )
    audio = write_pcm16_wav(args.wav_out, data)
    print(json.dumps({"lineage": lineage, "audio": audio}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
