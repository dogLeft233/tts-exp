from pathlib import Path

import numpy as np

from scripts.experiments.lrs3_real_video_local_timing import config
from scripts.experiments.lrs3_real_video_local_timing.media import (
    build_mapping,
    crop_frames_from_track,
    decode_pcm16,
    decode_video_frames,
    encode_ffv1,
    mux_pcm,
    source_pcm16,
    verify_muxed_media,
)


def test_video_mapping_has_registered_endpoints_monotonicity_and_plateaus() -> None:
    mappings = build_mapping(150)
    assert mappings[config.REAL_ARM] == list(range(150))
    assert mappings[config.REPEAT_ARM] == mappings[config.REAL_ARM]
    warp = np.asarray(mappings[config.WARP_ARM])
    displacement = warp - np.arange(150)
    assert int(warp[0]) == 0
    assert int(warp[-1]) == 149
    assert np.all(np.diff(warp) >= 0)
    assert int(displacement.min()) == -3
    assert int(displacement.max()) == 3
    assert np.any(np.diff(warp) == 0)


def test_crop_track_is_reused_for_each_frame_selection() -> None:
    frames = [np.full((4, 4, 3), index, dtype=np.uint8) for index in range(6)]
    track = [{"x": 2.0, "y": 2.0, "s": 2.0} for _ in frames]
    crops = crop_frames_from_track(frames, track)
    assert len(crops) == len(frames)
    assert all(frame.shape == (config.CROP_SIZE, config.CROP_SIZE, 3) for frame in crops)


def test_ffv1_and_pcm_mux_preserve_frames_and_audio(tmp_path: Path) -> None:
    import wave

    frames = [np.full((224, 224, 3), index, dtype=np.uint8) for index in range(7)]
    video_only = tmp_path / "video.mkv"
    media = tmp_path / "media.mkv"
    audio = tmp_path / "audio.wav"
    samples = np.arange(640 * 7, dtype=np.int16)
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(samples.tobytes())
    encode_ffv1(frames, video_only)
    mux_pcm(video_only, audio, media)
    assert all(np.array_equal(left, right) for left, right in zip(frames, decode_video_frames(video_only), strict=True))
    evidence = verify_muxed_media(media, frames, source_pcm16(audio), list(range(7)))
    assert evidence["audio_pcm_verified"] is True
    assert evidence["frame_count"] == 7


def test_pcm_mux_preserves_an_approximately_1_4_frame_audio_tail(tmp_path: Path) -> None:
    import wave

    frames = [np.full((224, 224, 3), index, dtype=np.uint8) for index in range(7)]
    video_only = tmp_path / "video.mkv"
    media = tmp_path / "media.mkv"
    audio = tmp_path / "audio.wav"
    samples = np.arange(7 * config.SAMPLES_PER_FRAME + 896, dtype=np.int16)
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(config.SAMPLE_RATE)
        handle.writeframes(samples.tobytes())
    encode_ffv1(frames, video_only)
    mux_pcm(video_only, audio, media)
    assert decode_pcm16(media) == source_pcm16(audio)
    assert len(decode_pcm16(media)) == samples.size * config.PCM_SAMPLE_WIDTH
