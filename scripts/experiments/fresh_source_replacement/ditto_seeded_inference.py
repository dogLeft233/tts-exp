"""Deterministic offline Ditto wrapper with lossless natural-audio muxing.

The upstream example leaves seeding commented out and muxes AAC through
``os.system``.  This adapter is intentionally small: it fixes the seed before
constructing/running the SDK and muxes the original PCM stream with ``check=True``.
It is copied to the pinned Ditto checkout only for the engineering smoke and
future A-stage generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import librosa
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_ditto(*, repo: Path, cfg: Path, data_root: Path, audio: Path, source: Path, output: Path, seed: int) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(repo))
    from inference import seed_everything  # type: ignore
    from stream_pipeline_offline import StreamSDK  # type: ignore

    seed_everything(seed)
    sdk = StreamSDK(str(cfg), str(data_root))
    values, sample_rate = librosa.load(str(audio), sr=16000, mono=True)
    values = np.asarray(values, dtype=np.float32)
    if sample_rate != 16000 or values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Ditto smoke audio must be finite mono 16kHz")
    num_frames = math.ceil(len(values) / 16000 * 25)
    sdk.setup(str(source), str(output),)
    sdk.setup_Nd(N_d=num_frames, fade_in=-1, fade_out=-1, ctrl_info={})
    if sdk.online_mode:
        raise RuntimeError("online Ditto backend is not allowed for this offline probe")
    sdk.audio2motion_queue.put(sdk.wav2feat.wav2feat(values))
    sdk.close()
    tmp_output = Path(sdk.tmp_output_path)
    if not tmp_output.is_file():
        raise FileNotFoundError(tmp_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "/usr/bin/ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp_output), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", str(output),
    ]
    subprocess.run(command, check=True)
    return {
        "status": "GO",
        "seed": seed,
        "audio_path": str(audio),
        "audio_sha256": sha256(audio),
        "source_path": str(source),
        "output_path": str(output),
        "output_sha256": sha256(output),
        "tmp_video_path": str(tmp_output),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args(argv)
    result = run_ditto(repo=args.repo, cfg=args.cfg, data_root=args.data_root, audio=args.audio, source=args.source, output=args.output, seed=args.seed)
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
