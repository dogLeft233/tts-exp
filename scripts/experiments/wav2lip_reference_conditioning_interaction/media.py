from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

from scripts.experiments.lrs3_real_video_local_timing.media import (
    decode_video_frames,
    mux_pcm,
    verify_muxed_media,
)

from .common import file_sha256, pcm16


def mux(video_only: Path, audio: Path, output: Path) -> dict[str, Any]:
    mux_pcm(video_only, audio, output, output.with_suffix(".mux.log"))
    frames = decode_video_frames(video_only)
    verify = verify_muxed_media(output, frames, pcm16(audio), list(range(len(frames))))
    return {
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "video_only": str(video_only.resolve()),
        "video_only_sha256": file_sha256(video_only),
        "audio": str(audio.resolve()),
        "audio_sha256": file_sha256(audio),
        "frame_count": len(frames),
        "verification": verify,
    }


def delayed_audio(source: Path, output: Path, samples: int = 3200) -> dict[str, Any]:
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        raw = reader.readframes(reader.getnframes())
    if (params.nchannels, params.sampwidth, params.framerate) != (1, 2, 16_000):
        raise ValueError("source is not PCM16/16k mono")
    step = samples * 2
    delayed = b"\0" * step + raw[:-step]
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(delayed)
    return {
        "path": str(output.resolve()),
        "sha256": file_sha256(output),
        "pcm_sha256": __import__("hashlib").sha256(delayed).hexdigest(),
        "delay_samples": samples,
    }
