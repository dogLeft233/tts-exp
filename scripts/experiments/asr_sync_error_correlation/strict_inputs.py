"""PCM-preserving arm muxing using the repository's strict contract."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Sequence

from .io import atomic_write_json, file_sha256


def _strict_mux_module():
    from scripts.experiments.lrs3_mfa_linear_replacement.strict_mux import mux_and_verify
    return mux_and_verify


def validate_arm_files(record: dict[str, Any], repo_root: Path) -> None:
    video = repo_root / record["video_path"] if not Path(record["video_path"]).is_absolute() else Path(record["video_path"])
    audio = repo_root / record["audio_path"] if not Path(record["audio_path"]).is_absolute() else Path(record["audio_path"])
    if not video.is_file() or file_sha256(video) != record["video_sha256"]:
        raise ValueError(f"video hash mismatch: {record['sample_id']}:{record['arm']}")
    if not audio.is_file() or file_sha256(audio) != record["audio_sha256"]:
        raise ValueError(f"audio hash mismatch: {record['sample_id']}:{record['arm']}")


def mux_arm(record: dict[str, Any], *, repo_root: Path, output_path: Path, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict[str, Any]:
    validate_arm_files(record, repo_root)
    video = repo_root / record["video_path"] if not Path(record["video_path"]).is_absolute() else Path(record["video_path"])
    audio = repo_root / record["audio_path"] if not Path(record["audio_path"]).is_absolute() else Path(record["audio_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = _strict_mux_module()(source_video=video, expected_audio=audio, output_path=output_path, ffmpeg=ffmpeg, ffprobe=ffprobe)
    return {
        "sample_id": str(record["sample_id"]),
        "arm": str(record["arm"]),
        "video_path": str(video),
        "audio_path": str(audio),
        "output_path": str(output_path),
        "video_sha256": file_sha256(video),
        "audio_sha256": file_sha256(audio),
        "output_sha256": file_sha256(output_path),
        "strict_contract": result,
        "audio_duration_s": float(record["audio_duration_s"]),
        "video_duration_s": float(record["video_duration_s"]),
        "audio_outside_video_s": max(0.0, float(record["audio_duration_s"]) - float(record["video_duration_s"])),
    }


def run_command(command: Sequence[str], *, log_path: Path, cwd: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(list(command), cwd=str(cwd) if cwd else None, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed with return code {result.returncode}: {command[0]}")


def write_input_record(path: Path, record: dict[str, Any]) -> None:
    atomic_write_json(path, record)
