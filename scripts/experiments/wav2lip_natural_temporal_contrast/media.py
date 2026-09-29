from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.experiments.lrs3_real_video_local_timing.media import decode_video_frames, mux_pcm, source_pcm16, verify_muxed_media

from .common import file_sha256


def mux_natural(video_only: Path, audio: Path, output: Path) -> dict[str, Any]:
    result = mux_pcm(video_only, audio, output, output.with_suffix(".mux.log"))
    frames = decode_video_frames(video_only)
    verification = verify_muxed_media(output, frames, source_pcm16(audio), list(range(len(frames))))
    return {"output": str(output.resolve()), "output_sha256": file_sha256(output), "video_only": str(video_only.resolve()), "video_only_sha256": file_sha256(video_only), "audio": str(audio.resolve()), "audio_sha256": file_sha256(audio), "frame_count": len(frames), "verification": verification}
