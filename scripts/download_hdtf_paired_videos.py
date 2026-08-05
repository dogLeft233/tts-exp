#!/usr/bin/env python3
"""Download a deterministic set of matched HDTF talking-face videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from download_hdtf_english_audio import download_file, is_directory, list_folder, login


VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
DEFAULT_REMOTE_ROOT = "/dhg_mm/HDTF/_videos_raw"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def enumerate_videos(remote_root: str, sid: str, token: str) -> list[dict[str, Any]]:
    pending = [remote_root]
    videos: list[dict[str, Any]] = []
    while pending:
        folder = pending.pop()
        for entry in list_folder(folder, sid, token):
            path = str(entry.get("path") or f"{folder}/{entry['name']}")
            if is_directory(entry):
                pending.append(path)
            elif PurePosixPath(path).suffix.lower() in VIDEO_SUFFIXES:
                videos.append({**entry, "path": path})
    return sorted(videos, key=lambda entry: str(entry["path"]))


def speaker_key(stem: str) -> str:
    return re.sub(r"[0-9]+$", "", stem)


def select_pairs(
    audio_records: list[dict[str, Any]],
    video_stems: set[str],
    count: int = 13,
    minimum_count: int | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for record in audio_records:
        stem = Path(record["local_path"]).stem
        if stem not in video_stems:
            continue
        if not (stem.startswith("WRA_") or stem.startswith("WDA_")):
            continue
        if float(record["duration_s"]) < 6.0:
            continue
        candidates.append(
            {
                "stem": stem,
                "speaker_key": speaker_key(stem),
                "duration_s": float(record["duration_s"]),
                "audio_path": record["local_path"],
                "audio_sha256": record.get("sha256"),
            }
        )
    best_by_speaker: dict[str, dict[str, Any]] = {}
    for candidate in sorted(
        candidates, key=lambda item: (item["duration_s"], item["stem"])
    ):
        best_by_speaker.setdefault(candidate["speaker_key"], candidate)
    selected = sorted(
        best_by_speaker.values(), key=lambda item: (item["duration_s"], item["stem"])
    )
    required_count = count if minimum_count is None else minimum_count
    if len(selected) < required_count:
        raise ValueError(f"only {len(selected)} unique matched WRA/WDA speakers")
    return selected[:count]


def probe_duration(path: Path, ffprobe_bin: str = "ffprobe") -> float:
    command = [
        ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    output = subprocess.check_output(command, text=True, timeout=60).strip()
    duration = float(output)
    if duration <= 0:
        raise ValueError(f"invalid duration for {path}: {duration}")
    return duration


def load_audio_records(
    repo: Path, manifest_path: Path, ffprobe_bin: str = "ffprobe"
) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"invalid HDTF audio manifest: {manifest_path}")
    result = []
    for record in records:
        path = Path(record["local_path"])
        if not path.is_absolute():
            path = repo / path
        if not path.exists():
            raise FileNotFoundError(path)
        expected_sha256 = record.get("sha256")
        actual_sha256 = sha256_file(path)
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(f"audio SHA256 mismatch: {path}")
        result.append(
            {
                **record,
                "local_path": str(path),
                "duration_s": probe_duration(path, ffprobe_bin),
                "sha256": actual_sha256,
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-manifest", required=True, type=Path)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=13)
    parser.add_argument(
        "--replacement-buffer",
        type=int,
        default=2,
        help="extra speakers to download for preparation-time replacement",
    )
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    if args.replacement_buffer < 0:
        raise SystemExit("--replacement-buffer must be non-negative")
    password = os.environ.get("NAS_DHG_MM_PASSWORD")
    if not password:
        raise SystemExit("Set NAS_DHG_MM_PASSWORD in the current shell")

    repo = Path(__file__).resolve().parents[1]
    audio_manifest = args.audio_manifest
    if not audio_manifest.is_absolute():
        audio_manifest = repo / audio_manifest
    audio_records = load_audio_records(repo, audio_manifest, args.ffprobe)

    sid, token = login(password)
    remote_videos = enumerate_videos(args.remote_root, sid, token)
    by_stem = {PurePosixPath(entry["path"]).stem: entry for entry in remote_videos}
    selected = select_pairs(
        audio_records,
        set(by_stem),
        args.count + args.replacement_buffer,
        minimum_count=args.count,
    )
    print(
        f"Selected {len(selected)} matched HDTF speakers "
        f"({args.count} required, {args.replacement_buffer} replacements reserved):"
    )
    for index, item in enumerate(selected, start=1):
        print(
            f"  {index}: {item['stem']} speaker={item['speaker_key']} "
            f"duration={item['duration_s']:.3f}s"
        )
    if args.dry_run:
        return 0

    video_dir = args.output_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    output_records: list[dict[str, Any]] = []
    for index, item in enumerate(selected, start=1):
        remote_entry = by_stem[item["stem"]]
        remote_path = str(remote_entry["path"])
        destination = video_dir / f"{index:02d}_{item['stem']}.mp4"
        if destination.exists():
            print(f"Reusing {destination}")
        else:
            download_file(remote_path, destination, sid, token)
        output_records.append(
            {
                "numeric_id": index,
                **item,
                "remote_path": remote_path,
                "remote_size_bytes": remote_entry.get("size"),
                "video_local_path": str(destination),
                "video_sha256": sha256_file(destination),
                "video_size_bytes": destination.stat().st_size,
            }
        )
        print(f"Downloaded {destination}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = {
        "schema_version": 1,
        "audio_manifest": str(audio_manifest),
        "remote_root": args.remote_root,
        "selection": "shortest matched WRA/WDA clip per speaker",
        "requested_count": args.count,
        "replacement_buffer": args.replacement_buffer,
        "records": output_records,
    }
    manifest_path = args.output_dir / "video_manifest.json"
    manifest_path.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
