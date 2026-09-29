from __future__ import annotations

import wave

import numpy as np

from scripts.experiments.lrs3_real_video_local_timing.media import (
    decode_pcm16,
    decode_video_frames,
    encode_ffv1,
    mux_pcm,
    verify_muxed_media,
)


def test_media_mux_preserves_video_and_pcm(tmp_path) -> None:
    frames = [np.full((224, 224, 3), value, dtype=np.uint8) for value in (10, 20, 30, 40, 50, 60)]
    audio = tmp_path / "audio.wav"
    pcm = (np.arange(3_840, dtype=np.int16) - 1_920).astype("<i2").tobytes()
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(pcm)
    video = tmp_path / "video.mkv"
    muxed = tmp_path / "muxed.mkv"
    encode_ffv1(frames, video)
    mux_pcm(video, audio, muxed)
    assert decode_pcm16(muxed) == pcm
    decoded = decode_video_frames(muxed)
    assert all(np.array_equal(left, right) for left, right in zip(decoded, frames, strict=True))
    assert verify_muxed_media(muxed, frames, pcm, list(range(len(frames))))["audio_pcm_verified"]
