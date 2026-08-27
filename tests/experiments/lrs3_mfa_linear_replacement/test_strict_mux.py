from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.experiments.lrs3_mfa_linear_replacement.strict_mux import (
    build_strict_mux_command,
    validate_strict_mux_command,
    verify_mux_integrity,
)


def test_strict_mux_command_has_only_registered_operations() -> None:
    command = build_strict_mux_command("video.mp4", "audio.wav", "out.mkv")
    text = " ".join(command)
    assert "-map 0:v:0" in text
    assert "-map 1:a:0" in text
    assert "-c:v copy" in text
    assert "-c:a pcm_s16le" in text
    assert "-shortest" not in text
    assert "atempo" not in text


def test_strict_mux_rejects_duration_or_filter_operations() -> None:
    command = build_strict_mux_command("video.mp4", "audio.wav", "out.mkv")
    with pytest.raises(ValueError, match="unexpected arguments"):
        validate_strict_mux_command(command + ["-shortest"])
    with pytest.raises(ValueError, match="unexpected arguments"):
        validate_strict_mux_command(command + ["-af", "atempo=1.0"])


def test_mux_verification_checks_streams_format_and_identity(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    muxed = tmp_path / "muxed.mkv"
    expected = tmp_path / "expected.wav"
    for path in (source, muxed, expected):
        path.write_bytes(path.name.encode())

    def runner(command, *args, **kwargs):
        command = [str(value) for value in command]
        if "-show_streams" in command:
            target = command[-1]
            if target.endswith("expected.wav"):
                streams = [{"codec_type": "audio", "codec_name": "pcm_s16le", "sample_rate": "16000", "channels": 1}]
            elif target.endswith("muxed.mkv"):
                streams = [
                    {"codec_type": "video", "codec_name": "h264"},
                    {"codec_type": "audio", "codec_name": "pcm_s16le", "sample_rate": "16000", "channels": 1},
                ]
            else:
                streams = [{"codec_type": "video", "codec_name": "h264"}]
            return SimpleNamespace(returncode=0, stdout=json.dumps({"streams": streams}).encode(), stderr=b"")
        if "-show_format" in command:
            return SimpleNamespace(returncode=0, stdout=json.dumps({"format": {"format_name": "matroska,webm"}}).encode(), stderr=b"")
        if "-f" in command and command[command.index("-f") + 1] == "h264":
            return SimpleNamespace(returncode=0, stdout=b"video-elementary", stderr=b"")
        if "-f" in command and command[command.index("-f") + 1] == "s16le":
            return SimpleNamespace(returncode=0, stdout=b"canonical-audio", stderr=b"")
        raise AssertionError(command)

    result = verify_mux_integrity(source_video=source, muxed_file=muxed, expected_audio=expected, runner=runner)
    assert result["video_stream_copy_verified"] is True
    assert result["audio_pcm_verified"] is True
