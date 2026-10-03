"""Mux generated video with the untouched natural WAV using lossless stream copy.

Matroska is used because it can carry PCM audio without lossy AAC encoding.
The decoded audio stream is compared byte-for-byte after conversion to float32;
this helper does not select, retime, resample, or score the audio.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


class MuxError(RuntimeError):
    pass


@dataclass(frozen=True)
class MuxInfo:
    output_path: str
    output_file_sha256: str
    video_codec: str
    video_stream_sha256: str
    video_frame_count: int
    video_timestamps_sha256: str
    audio_codec: str
    audio_sample_rate: int
    audio_channels: int
    decoded_audio_sample_count: int
    decoded_audio_float32_sha256: str
    audio_matches_original_decoded_pcm: bool


def _run_json(command: list[str], *, timeout: float) -> dict:
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise MuxError(f"command failed ({completed.returncode}): {message}")
    try:
        return json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MuxError("ffprobe did not return valid UTF-8 JSON") from exc


def _probe(path: Path, ffprobe: str, timeout: float) -> dict:
    return _run_json(
        [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        timeout=timeout,
    )


def _stream(probe: dict, kind: str) -> dict:
    found = [item for item in probe.get("streams", []) if item.get("codec_type") == kind]
    if len(found) != 1:
        raise MuxError(f"expected exactly one {kind} stream, found {len(found)}")
    return found[0]


def _decoded_audio(
    path: Path, ffmpeg: str, ffprobe: str, timeout: float
) -> tuple[bytes, int, int]:
    probe = _probe(path, ffprobe, timeout)
    audio = _stream(probe, "audio")
    sample_rate = int(audio["sample_rate"])
    channels = int(audio["channels"])
    if sample_rate <= 0 or channels <= 0:
        raise MuxError("ffprobe returned invalid audio sample-rate/channel metadata")
    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise MuxError(f"audio decode failed ({completed.returncode}): {message}")
    if len(completed.stdout) % (4 * channels):
        raise MuxError("decoded float32 audio byte count is not frame aligned")
    frame_count = len(completed.stdout) // (4 * channels)
    return completed.stdout, sample_rate, frame_count


def _stream_sha256(path: Path, ffmpeg: str, timeout: float) -> str:
    completed = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-f",
            "hash",
            "-hash",
            "sha256",
            "pipe:1",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace")[-4000:]
        raise MuxError(f"video stream hashing failed ({completed.returncode}): {message}")
    result = completed.stdout.decode("ascii", errors="strict").strip()
    prefix = "SHA256="
    if not result.startswith(prefix) or len(result) != len(prefix) + 64:
        raise MuxError("ffmpeg did not return the expected video SHA-256")
    return result[len(prefix) :].lower()


def _video_timestamps(path: Path, ffprobe: str, timeout: float) -> tuple[Decimal, ...]:
    document = _run_json(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(path),
        ],
        timeout=timeout,
    )
    frames = document.get("frames")
    if not isinstance(frames, list) or not frames:
        raise MuxError("ffprobe did not return a non-empty video frame timeline")
    values: list[Decimal] = []
    for frame in frames:
        try:
            values.append(Decimal(str(frame["best_effort_timestamp_time"])))
        except (KeyError, InvalidOperation) as exc:
            raise MuxError("a video frame has no numeric timestamp") from exc
    return tuple(values)


def _timestamp_digest(values: tuple[Decimal, ...]) -> str:
    canonical = "\n".join(format(value.normalize(), "f") for value in values).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mux_original_pcm(
    generated_video: str | os.PathLike[str],
    original_audio_wav: str | os.PathLike[str],
    output_mkv: str | os.PathLike[str],
    *,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    timeout_seconds: float = 600.0,
    overwrite: bool = False,
) -> MuxInfo:
    """Stream-copy video and original PCM into MKV, then verify decoded PCM.

    The destination must end in ``.mkv`` or ``.mov``. The generated video's audio, if any,
    is deliberately not mapped. The natural WAV stream is copied unchanged;
    decoded float32 PCM bytes are compared before atomically publishing output.
    """

    video_path = Path(generated_video)
    audio_path = Path(original_audio_wav)
    destination = Path(output_mkv)
    container = destination.suffix.lower()
    if container not in (".mkv", ".mov"):
        raise MuxError("lossless PCM mux output must use a .mkv or .mov container")
    if not video_path.is_file() or not audio_path.is_file():
        raise FileNotFoundError("generated video and original audio WAV must both exist")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    if shutil.which(ffmpeg) is None and not Path(ffmpeg).is_file():
        raise FileNotFoundError(f"ffmpeg executable not found: {ffmpeg}")
    if shutil.which(ffprobe) is None and not Path(ffprobe).is_file():
        raise FileNotFoundError(f"ffprobe executable not found: {ffprobe}")
    timeout = float(timeout_seconds)
    if timeout <= 0:
        raise ValueError("timeout_seconds must be positive")

    original_video_info = _probe(video_path, ffprobe, timeout)
    video_stream = _stream(original_video_info, "video")
    audio_probe = _probe(audio_path, ffprobe, timeout)
    audio_stream = _stream(audio_probe, "audio")
    if not str(audio_stream.get("codec_name", "")).startswith("pcm_"):
        raise MuxError("original audio must be a PCM WAV stream for lossless stream-copy mux")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.tmp{container}")
    try:
        command = [
            ffmpeg,
            "-v",
            "error",
            "-nostdin",
            "-copyts",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            "-map_metadata",
            "0",
            "-f",
            ("matroska" if container == ".mkv" else "mov"),
            str(temporary),
        ]
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", errors="replace")[-4000:]
            raise MuxError(f"ffmpeg mux failed ({completed.returncode}): {message}")
        output_probe = _probe(temporary, ffprobe, timeout)
        output_video = _stream(output_probe, "video")
        output_audio = _stream(output_probe, "audio")
        if output_video.get("codec_name") != video_stream.get("codec_name"):
            raise MuxError("video codec changed during stream-copy mux")
        if output_audio.get("codec_name") != audio_stream.get("codec_name"):
            raise MuxError("natural PCM codec changed during stream-copy mux")
        for key in ("sample_rate", "channels"):
            if str(output_audio.get(key)) != str(audio_stream.get(key)):
                raise MuxError(f"natural audio {key} changed during stream-copy mux")
        original_pcm, original_rate, original_count = _decoded_audio(
            audio_path, ffmpeg, ffprobe, timeout
        )
        muxed_pcm, muxed_rate, muxed_count = _decoded_audio(temporary, ffmpeg, ffprobe, timeout)
        if (original_rate, original_count, len(original_pcm)) != (
            muxed_rate,
            muxed_count,
            len(muxed_pcm),
        ):
            raise MuxError("decoded original and muxed audio frame clocks differ")
        if original_pcm != muxed_pcm:
            raise MuxError("decoded mux audio differs from original natural PCM")
        original_timestamps = _video_timestamps(video_path, ffprobe, timeout)
        muxed_timestamps = _video_timestamps(temporary, ffprobe, timeout)
        if len(original_timestamps) != len(muxed_timestamps):
            raise MuxError("video frame count changed during stream-copy mux")
        timestamp_deltas = [abs(a - b) for a, b in zip(original_timestamps, muxed_timestamps)]
        if any(delta > Decimal("0.000001") for delta in timestamp_deltas):
            raise MuxError("video frame timestamps changed by more than 1 microsecond")
        original_video_sha = _stream_sha256(video_path, ffmpeg, timeout)
        muxed_video_sha = _stream_sha256(temporary, ffmpeg, timeout)
        if original_video_sha != muxed_video_sha:
            raise MuxError("video elementary stream changed during stream-copy mux")
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return MuxInfo(
        output_path=str(destination.resolve()),
        output_file_sha256=_file_sha256(destination),
        video_codec=str(video_stream.get("codec_name", "unknown")),
        video_stream_sha256=original_video_sha,
        video_frame_count=len(original_timestamps),
        video_timestamps_sha256=_timestamp_digest(original_timestamps),
        audio_codec=str(audio_stream.get("codec_name", "unknown")),
        audio_sample_rate=original_rate,
        audio_channels=int(audio_stream["channels"]),
        decoded_audio_sample_count=original_count,
        decoded_audio_float32_sha256=hashlib.sha256(original_pcm).hexdigest(),
        audio_matches_original_decoded_pcm=True,
    )
