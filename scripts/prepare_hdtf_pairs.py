#!/usr/bin/env python3
"""Prepare matched HDTF audio crops and source frames for one run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml


CANONICAL_SAMPLE_RATE = 24000
CANONICAL_SUBTYPE = "PCM_16"
CROP_DURATION_S = 6.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config_override(run_id: str, sample_count: int = 13) -> dict[str, Any]:
    return {
        "paths": {
            "audio_dir": f"runs/{run_id}/00_pairs/natural",
            "tts_audio_dir": f"runs/{run_id}/02_tts",
            "image_dir": f"runs/{run_id}/00_pairs/images",
        },
        "samples": {
            "count": sample_count,
            "ids": list(range(1, sample_count + 1)),
            "smoke_id": 1,
        },
        "tts": {
            "language": "English",
            "seed": 42,
            "retry": 1,
        },
        "ditto": {
            "loudnorm": False,
            "conditions": ["natural_raw", "tts_raw"],
            "retry": 1,
            "seed": 42,
        },
        "eval": {
            "require_complete_pairs": True,
        },
    }


def probe_duration(path: Path, ffprobe_bin: str = "ffprobe") -> float:
    output = subprocess.check_output(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
        timeout=60,
    ).strip()
    duration = float(output)
    if duration <= 0:
        raise ValueError(f"invalid duration for {path}: {duration}")
    return duration


def crop_audio(
    source: Path,
    destination: Path,
    start_s: float,
    ffmpeg_bin: str,
    overwrite: bool = False,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return
    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start_s:.3f}",
            "-t",
            f"{CROP_DURATION_S:.3f}",
            "-i",
            str(source),
            "-ar",
            str(CANONICAL_SAMPLE_RATE),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )


def extract_frame(
    source: Path,
    destination: Path,
    timestamp_s: float,
    ffmpeg_bin: str,
    overwrite: bool = False,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return
    subprocess.run(
        [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-ss",
            f"{timestamp_s:.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            "scale=512:-2",
            str(destination),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )


def validate_audio(path: Path) -> dict[str, Any]:
    info = sf.info(path)
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.size == 0 or info.frames == 0:
        raise ValueError(f"empty audio: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"non-finite audio: {path}")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms <= 1e-4 or peak > 1.0:
        raise ValueError(f"invalid audio level: {path}")
    if int(sample_rate) != CANONICAL_SAMPLE_RATE:
        raise ValueError(f"unexpected sample rate: {path}")
    if info.channels != 1 or info.subtype != CANONICAL_SUBTYPE:
        raise ValueError(f"unexpected audio format: {path}")
    return {
        "sample_rate_hz": int(sample_rate),
        "channels": int(info.channels),
        "subtype": info.subtype,
        "duration_s": round(float(info.duration), 6),
        "sha256": sha256_file(path),
    }


def validate_image(path: Path) -> dict[str, Any]:
    from PIL import Image

    with Image.open(path) as image:
        image.load()
        width, height = image.size
        if width < 32 or height < 32:
            raise ValueError(f"image too small: {path}")
    return {
        "width": width,
        "height": height,
        "sha256": sha256_file(path),
    }


def _resolve_path(repo: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo / path


def prepare_pairs(
    repo: Path,
    run_id: str,
    video_manifest_path: Path,
    audio_manifest_path: Path | None = None,
    count: int = 13,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    payload = json.loads(video_manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list) or len(records) < count:
        raise ValueError(f"video manifest has fewer than {count} candidate records")
    print(f"Candidate pool has {len(records)} HDTF pairs; preparing {count}:")
    for index, record in enumerate(records, start=1):
        print(
            f"  {index}: {record['stem']} speaker={record['speaker_key']} "
            f"duration={float(record['duration_s']):.3f}s"
        )
    if dry_run:
        return {
            "run_id": run_id,
            "sample_count": count,
            "candidate_count": len(records),
            "records": records,
        }

    run_dir = repo / "runs" / run_id
    natural_dir = run_dir / "00_pairs" / "natural"
    image_dir = run_dir / "00_pairs" / "images"
    prepared: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    slot_failures: list[dict[str, Any]] = []
    for record in records:
        if len(prepared) >= count:
            break
        numeric_id = len(prepared) + 1
        temporary_audio = natural_dir / f".{numeric_id}_{record['stem']}.wav"
        temporary_image = image_dir / f".{numeric_id}_{record['stem']}.png"
        try:
            audio_source = _resolve_path(repo, record["audio_path"])
            video_source = _resolve_path(repo, record["video_local_path"])
            if not audio_source.exists():
                raise FileNotFoundError(audio_source)
            if not video_source.exists():
                raise FileNotFoundError(video_source)

            source_audio_sha256 = sha256_file(audio_source)
            expected_audio_sha256 = record.get("audio_sha256")
            if expected_audio_sha256 and source_audio_sha256 != expected_audio_sha256:
                raise ValueError(f"audio SHA256 mismatch: {audio_source}")

            try:
                audio_duration = probe_duration(audio_source, ffprobe_bin)
            except (OSError, subprocess.SubprocessError, ValueError):
                if record.get("duration_s") is None:
                    raise
                audio_duration = float(record["duration_s"])
            video_duration = probe_duration(video_source, ffprobe_bin)
            available_duration = min(audio_duration, video_duration)
            if available_duration < CROP_DURATION_S:
                raise ValueError(
                    f"matched source is shorter than {CROP_DURATION_S}s: "
                    f"{record['stem']} audio={audio_duration:.3f}s "
                    f"video={video_duration:.3f}s"
                )
            crop_start = max(0.0, (available_duration - CROP_DURATION_S) / 2.0)
            frame_time = crop_start + CROP_DURATION_S / 2.0
            audio_destination = natural_dir / f"{numeric_id}.wav"
            image_destination = image_dir / f"{numeric_id}.png"
            crop_audio(
                audio_source,
                temporary_audio,
                crop_start,
                ffmpeg_bin,
                overwrite=True,
            )
            extract_frame(
                video_source,
                temporary_image,
                frame_time,
                ffmpeg_bin,
                overwrite=True,
            )
            audio_qc = validate_audio(temporary_audio)
            image_qc = validate_image(temporary_image)
            temporary_audio.replace(audio_destination)
            temporary_image.replace(image_destination)
            prepared_record = {
                "sample_id": numeric_id,
                "stem": record["stem"],
                "speaker_key": record["speaker_key"],
                "remote_video_path": record["remote_path"],
                "source_audio": str(audio_source),
                "source_video": str(video_source),
                "source_audio_sha256": source_audio_sha256,
                "source_video_sha256": record["video_sha256"],
                "source_audio_duration_s": round(audio_duration, 6),
                "source_video_duration_s": round(video_duration, 6),
                "crop_start_s": round(crop_start, 6),
                "crop_duration_s": CROP_DURATION_S,
                "frame_time_s": round(frame_time, 6),
                "natural_audio": str(audio_destination.relative_to(repo)),
                "source_image": str(image_destination.relative_to(repo)),
                "audio_qc": audio_qc,
                "image_qc": image_qc,
            }
            if slot_failures:
                prepared_record["replacement_for"] = {
                    "sample_id": numeric_id,
                    "stem": slot_failures[-1]["stem"],
                    "reason": slot_failures[-1]["error"],
                }
                prepared_record["rejected_candidates"] = slot_failures.copy()
            prepared.append(prepared_record)
            slot_failures = []
            print(f"Prepared sample {numeric_id}: {record['stem']}")
        except Exception as exc:
            failure = {
                "attempted_sample_id": numeric_id,
                "stem": record["stem"],
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(failure)
            slot_failures.append(failure)
            print(f"Rejected candidate {record['stem']}: {failure['error']}")
        finally:
            temporary_audio.unlink(missing_ok=True)
            temporary_image.unlink(missing_ok=True)

    if len(prepared) < count:
        raise RuntimeError(
            f"prepared {len(prepared)} of {count} HDTF pairs; "
            f"candidate failures: {json.dumps(failures, ensure_ascii=False)}"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    pair_manifest = {
        "schema_version": 1,
        "design": "hdtf_paired_self_clone",
        "strict_unseen": False,
        "video_manifest": str(video_manifest_path),
        "audio_manifest": str(audio_manifest_path) if audio_manifest_path else None,
        "candidate_count": len(records),
        "rejected_candidates": failures,
        "records": prepared,
    }
    (run_dir / "pair_manifest.json").write_text(
        json.dumps(pair_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "run_id": run_id,
        "experiment_name": "hdtf_paired_tts_tfg",
        "design": "hdtf_paired_self_clone",
        "language": "English",
        "sample_description": (
            f"HDTF paired source, {len(prepared)} distinct speakers, "
            "centered six-second crops"
        ),
        "sample_count": len(prepared),
        "strict_unseen": False,
        "known_overlap": "HDTF is part of Ditto's published evaluation ecosystem",
        "candidate_count": len(records),
        "rejected_candidates": failures,
        "records": prepared,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "config_override.yaml").write_text(
        yaml.safe_dump(
            build_config_override(run_id, sample_count=len(prepared)),
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--video-manifest", required=True, type=Path)
    parser.add_argument("--audio-manifest", type=Path)
    parser.add_argument("--count", type=int, default=13)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")

    repo = Path(__file__).resolve().parents[1]
    prepare_pairs(
        repo=repo,
        run_id=args.run_id,
        video_manifest_path=args.video_manifest,
        audio_manifest_path=args.audio_manifest,
        count=args.count,
        ffmpeg_bin=args.ffmpeg,
        ffprobe_bin=args.ffprobe,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
