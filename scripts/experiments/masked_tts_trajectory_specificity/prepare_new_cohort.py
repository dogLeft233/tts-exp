"""Prepare an independent LRS3 trajectory-confirmation cohort from raw tar shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Any


EXPECTED_RECORDS = 16
EXPECTED_GROUPS = 8
SALT = "lrs3-trajectory-confirmation-new-v1\0"
OLD_SOURCE_POOL = Path(".claude/worktrees/lrs3-wavlm-resynthesis-50/tmp/lrs3_policy_a1_200_20260828/source_pool.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank(value: str) -> str:
    return hashlib.sha256((SALT + value).encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def transcript_from_text(raw: str) -> str:
    match = re.search(r"^Text:\s*(.*)$", raw, re.MULTILINE)
    if match is None:
        raise ValueError("LRS3 text has no Text line")
    text = re.sub(r"\{[^}]*\}", " ", match.group(1))
    text = text.replace("’", "'").replace("‘", "'")
    text = " ".join(text.split())
    if not text or not re.search(r"[A-Za-z]", text):
        raise ValueError("transcript is empty or non-lexical")
    return text


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    value = float(result.stdout.strip())
    if value < 3.84:
        raise ValueError(f"video shorter than frozen support: {value:.3f}s")
    return value


def extract_member(tar: tarfile.TarFile, member_name: str, destination: Path) -> None:
    source = tar.extractfile(member_name)
    if source is None:
        raise FileNotFoundError(member_name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source, destination.open("wb") as handle:
        while True:
            chunk = source.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)


def canonical_audio(video: Path, audio: Path) -> None:
    audio.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-map", "0:a:0", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(audio)],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", action="append", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    old_pool_path = repo / OLD_SOURCE_POOL
    old_pool = json.loads(old_pool_path.read_text(encoding="utf-8"))
    old_groups = {str(row["source_group"]) for row in old_pool.get("records", [])}
    shard_paths = [(repo / value).resolve() if not value.is_absolute() else value.resolve() for value in args.shard]
    if any(not path.is_file() for path in shard_paths):
        raise FileNotFoundError([str(path) for path in shard_paths if not path.is_file()])

    entries: dict[str, tuple[Path, str, str, str]] = {}
    group_members: dict[str, list[str]] = {}
    for shard_path in shard_paths:
        with tarfile.open(shard_path, "r:") as archive:
            for name in archive.getnames():
                match = re.fullmatch(r"pretrain/([^/]+)/([^/]+)\.mp4", name)
                if match is None:
                    continue
                group, clip = match.groups()
                if group in old_groups:
                    continue
                txt_name = f"pretrain/{group}/{clip}.txt"
                try:
                    raw_text = archive.extractfile(txt_name).read().decode("utf-8", "replace")
                    transcript = transcript_from_text(raw_text)
                except Exception:
                    continue
                group_members.setdefault(group, []).append(f"{group}/{clip}")
                entries[f"{group}/{clip}"] = (shard_path, name, txt_name, transcript)

    fresh_groups = sorted(group_members, key=rank)
    if len(fresh_groups) < EXPECTED_GROUPS:
        raise ValueError(f"only {len(fresh_groups)} fresh source groups available")
    selected_groups = fresh_groups[:EXPECTED_GROUPS]
    selected: list[dict[str, Any]] = []
    for group in selected_groups:
        candidates: list[tuple[str, float]] = []
        for key in sorted(group_members[group], key=rank):
            shard_path, video_name, _, transcript = entries[key]
            scratch = output / "selection_probe" / f"{key.replace('/', '__')}.mp4"
            with tarfile.open(shard_path, "r:") as archive:
                extract_member(archive, video_name, scratch)
            try:
                duration = probe_duration(scratch)
            except Exception:
                scratch.unlink(missing_ok=True)
                continue
            if len(transcript.split()) < 3:
                scratch.unlink(missing_ok=True)
                continue
            candidates.append((key, duration))
            scratch.unlink(missing_ok=True)
        if len(candidates) < 2:
            raise ValueError(f"source group {group} has fewer than two eligible records")
        for key, duration in sorted(candidates, key=lambda item: rank(item[0]))[:2]:
            selected.append({"key": key, "source_group": group, "duration_s": duration})

    records: list[dict[str, Any]] = []
    for selection in selected:
        key = selection["key"]
        group, clip = key.split("/", 1)
        shard_path, video_name, txt_name, transcript = entries[key]
        sample_id = f"lrs3_{group}_{clip}"
        video_path = output / "raw" / "video" / f"{sample_id}.mp4"
        txt_path = output / "raw" / "transcript" / f"{sample_id}.txt"
        audio_path = output / "raw" / "natural_audio" / f"{sample_id}.wav"
        with tarfile.open(shard_path, "r") as archive:
            extract_member(archive, video_name, video_path)
            extract_member(archive, txt_name, txt_path)
        canonical_audio(video_path, audio_path)
        duration = probe_duration(video_path)
        records.append({
            "sample_id": sample_id,
            "source_group": group,
            "protocol_split": "new_confirmation",
            "transcript": transcript,
            "source_shard": str(shard_path),
            "source_video_member": video_name,
            "source_text_member": txt_name,
            "face": str(video_path.resolve()),
            "natural_audio": str(audio_path.resolve()),
            "transcript_file": str(txt_path.resolve()),
            "video_sha256": sha256_file(video_path),
            "face_sha256": sha256_file(video_path),
            "natural_audio_sha256": sha256_file(audio_path),
            "transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
            "duration_s": duration,
            "fps": 25,
            "selection_rank_sha256": rank(key),
        })
    records.sort(key=lambda row: (selected_groups.index(str(row["source_group"])), str(row["sample_id"])))
    if len(records) != EXPECTED_RECORDS or len({row["source_group"] for row in records}) != EXPECTED_GROUPS:
        raise ValueError("selected cohort does not satisfy the fixed 16-record/8-group contract")
    if any(sum(row["source_group"] == group for row in records) != 2 for group in selected_groups):
        raise ValueError("selected source groups do not have exactly two records")
    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_masked_tts_trajectory_new_record_confirmation_cohort",
        "status": "complete",
        "dataset": "LRS3-TED",
        "source_repository": "TheNHz/ellipsis-lrs3-raw",
        "source_shards": [{"path": str(path), "sha256": sha256_file(path)} for path in shard_paths],
        "prior_source_pool": {"path": str(old_pool_path), "sha256": sha256_file(old_pool_path), "excluded_group_count": len(old_groups)},
        "records": records,
        "groups": selected_groups,
        "record_count": len(records),
        "group_count": len(selected_groups),
        "selection": {
            "salt": SALT,
            "fresh_group_rule": "source_group absent from prior source pool",
            "group_order": "sha256(salt + source_group)",
            "record_order": "sha256(salt + source_group/clip)",
            "eligibility": "video duration >= 3.84 seconds and lexical transcript with >= 3 words",
            "score_free": True,
            "excluded_old_source_groups": sorted(old_groups),
        },
        "required_for_confirmation": ["face video", "untouched natural audio", "transcript"],
        "sealed_splits_accessed": False,
    }
    write_json(output / "00_cohort/manifest.json", manifest)
    write_json(output / "00_cohort/selection.json", {"selected_groups": selected_groups, "records": records})
    print(json.dumps({"records": len(records), "groups": len(selected_groups), "groups_selected": selected_groups}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
