from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

FORBIDDEN_FLAGS = ("-shortest", "-t", "-filter:v", "-filter:a", "-vf", "-af", "atempo", "crop", "pad")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_strict_mux_command(video_path: str | Path, audio_path: str | Path, output_path: str | Path, *, ffmpeg: str = "ffmpeg") -> list[str]:
    command = [
        ffmpeg, "-y", "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-f", "matroska", str(output_path),
    ]
    validate_strict_mux_command(command)
    return command


def validate_strict_mux_command(command: Sequence[str]) -> None:
    tokens = [str(value) for value in command]
    if len(tokens) != 21:
        raise ValueError("strict mux command has unexpected arguments")
    expected_flags = [
        "-y", "-i", None, "-i", None,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-f", "matroska", None,
    ]
    for index, expected in enumerate(expected_flags):
        if expected is not None and tokens[index + 1] != expected:
            raise ValueError(f"strict mux command violates contract at argument {index + 1}")
    if not tokens[0] or not tokens[2] or not tokens[4] or not tokens[20]:
        raise ValueError("strict mux command has empty executable or path")


def _run(command: Sequence[str], *, capture_output: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(list(command), check=True, capture_output=capture_output)


def ffprobe_streams(path: str | Path, *, ffprobe: str = "ffprobe", runner=_run) -> list[dict[str, Any]]:
    command = [ffprobe, "-v", "error", "-show_streams", "-of", "json", str(path)]
    result = runner(command)
    payload = json.loads(result.stdout.decode("utf-8"))
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise ValueError("ffprobe did not return streams")
    return streams


def ffprobe_format_name(path: str | Path, *, ffprobe: str = "ffprobe", runner=_run) -> str:
    command = [ffprobe, "-v", "error", "-show_format", "-of", "json", str(path)]
    result = runner(command)
    payload = json.loads(result.stdout.decode("utf-8"))
    value = payload.get("format", {}).get("format_name")
    if not isinstance(value, str) or not value:
        raise ValueError("ffprobe did not return a format name")
    return value


def _video_stream(streams: Sequence[dict[str, Any]]) -> dict[str, Any]:
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    if len(videos) != 1:
        raise ValueError("expected exactly one video stream")
    return videos[0]


def _audio_stream(streams: Sequence[dict[str, Any]]) -> dict[str, Any]:
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(audios) != 1:
        raise ValueError("expected exactly one audio stream")
    return audios[0]


def _elementary_format(codec_name: str) -> str:
    formats = {"h264": "h264", "hevc": "hevc", "mpeg4": "m4v", "vp8": "ivf", "vp9": "ivf", "av1": "ivf"}
    if codec_name not in formats:
        raise ValueError(f"unsupported video codec for elementary hash: {codec_name}")
    return formats[codec_name]


def elementary_video_sha256(path: str | Path, *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe", runner=_run) -> str:
    streams = ffprobe_streams(path, ffprobe=ffprobe, runner=runner)
    video = _video_stream(streams)
    command = [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-c:v", "copy", "-an", "-f", _elementary_format(str(video.get("codec_name"))), "pipe:1"]
    result = runner(command)
    return hashlib.sha256(result.stdout).hexdigest()


def canonical_pcm_bytes(path: str | Path, *, ffmpeg: str = "ffmpeg", runner=_run) -> bytes:
    command = [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1"]
    return runner(command).stdout


def verify_mux_integrity(
    *,
    source_video: str | Path,
    muxed_file: str | Path,
    expected_audio: str | Path,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    runner=_run,
) -> dict[str, Any]:
    source_streams = ffprobe_streams(source_video, ffprobe=ffprobe, runner=runner)
    muxed_streams = ffprobe_streams(muxed_file, ffprobe=ffprobe, runner=runner)
    if "matroska" not in ffprobe_format_name(muxed_file, ffprobe=ffprobe, runner=runner).split(","):
        raise ValueError("muxed output is not Matroska")
    expected_audio_stream = _audio_stream(ffprobe_streams(expected_audio, ffprobe=ffprobe, runner=runner))
    if expected_audio_stream.get("codec_name") != "pcm_s16le" or int(expected_audio_stream.get("sample_rate", 0)) != 16000 or int(expected_audio_stream.get("channels", 0)) != 1:
        raise ValueError("expected audio is not canonical 16 kHz mono PCM s16le")
    source_video_stream = _video_stream(source_streams)
    muxed_video_stream = _video_stream(muxed_streams)
    audio_stream = _audio_stream(muxed_streams)
    if muxed_video_stream.get("codec_name") != source_video_stream.get("codec_name"):
        raise ValueError("muxed video codec differs from source")
    if audio_stream.get("codec_name") != "pcm_s16le" or int(audio_stream.get("sample_rate", 0)) != 16000 or int(audio_stream.get("channels", 0)) != 1:
        raise ValueError("muxed audio stream violates PCM contract")
    source_video_hash = elementary_video_sha256(source_video, ffmpeg=ffmpeg, ffprobe=ffprobe, runner=runner)
    muxed_video_hash = elementary_video_sha256(muxed_file, ffmpeg=ffmpeg, ffprobe=ffprobe, runner=runner)
    if source_video_hash != muxed_video_hash:
        raise ValueError("video elementary stream changed during mux")
    expected_pcm = canonical_pcm_bytes(expected_audio, ffmpeg=ffmpeg, runner=runner)
    muxed_pcm = canonical_pcm_bytes(muxed_file, ffmpeg=ffmpeg, runner=runner)
    if expected_pcm != muxed_pcm:
        raise ValueError("muxed audio PCM differs from expected audio")
    return {
        "source_video_sha256": file_sha256(source_video),
        "muxed_file_sha256": file_sha256(muxed_file),
        "source_video_elementary_sha256": source_video_hash,
        "muxed_video_elementary_sha256": muxed_video_hash,
        "expected_audio_pcm_sha256": hashlib.sha256(expected_pcm).hexdigest(),
        "muxed_audio_pcm_sha256": hashlib.sha256(muxed_pcm).hexdigest(),
        "expected_audio_pcm_bytes": len(expected_pcm),
        "muxed_audio_pcm_bytes": len(muxed_pcm),
        "video_stream_copy_verified": True,
        "audio_pcm_verified": True,
    }


def mux_and_verify(
    *,
    source_video: str | Path,
    expected_audio: str | Path,
    output_path: str | Path,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    runner=_run,
) -> dict[str, Any]:
    command = build_strict_mux_command(source_video, expected_audio, output_path, ffmpeg=ffmpeg)
    result = runner(command)
    verification = verify_mux_integrity(source_video=source_video, muxed_file=output_path, expected_audio=expected_audio, ffmpeg=ffmpeg, ffprobe=ffprobe, runner=runner)
    return {"command": command, "returncode": result.returncode, **verification}
