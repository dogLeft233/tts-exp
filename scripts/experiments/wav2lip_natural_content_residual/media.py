from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.experiments.lrs3_real_video_local_timing.media import (
    decode_video_frames,
    encode_ffv1,
    mux_pcm,
    verify_muxed_media,
)

from . import config
from .common import ProtocolError, file_sha256, source_pcm16, write_delayed_pcm16


def delay_audio(source: Path, output: Path) -> dict[str, Any]:
    return write_delayed_pcm16(source, output, 3_200)


def mux_and_verify(video_only: Path, audio: Path, output: Path, *, sample_id: str, arm: str) -> dict[str, Any]:
    if not video_only.is_file() or not audio.is_file():
        raise ProtocolError(f"media input is missing: {sample_id}/{arm}")
    if output.exists():
        raise ProtocolError(f"media output already exists: {output}")
    mux_pcm(video_only, audio, output, output.with_suffix(".mux.log"))
    frames = decode_video_frames(video_only)
    pcm = source_pcm16(audio)
    verification = verify_muxed_media(output, frames, pcm, list(range(len(frames))))
    return {"sample_id": sample_id, "arm": arm, "output": str(output.resolve()), "output_sha256": file_sha256(output), "video_only": str(video_only.resolve()), "video_only_sha256": file_sha256(video_only), "audio": str(audio.resolve()), "audio_sha256": file_sha256(audio), "audio_pcm_sha256": verification["audio_pcm_sha256"], "frame_count": len(frames), "verification": verification, "audio_modified": arm == "A_DELAY"}
