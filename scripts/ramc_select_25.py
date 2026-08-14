#!/usr/bin/env python3
"""Select 25 RAMC conversations from the full local tarball.

Uses stride sampling over archive member order so the selection spreads
across batch dirs / regions / topics instead of taking the first 25 files
(the first batch is heavily biased, e.g. Sichuan speakers only).

For each selected conversation (.wav member) the script extracts:
  - the conversation wav,
  - any TXT member matching the same stem (transcript),
  - every SPKINFO.txt member (speaker metadata).

Produces <outdir>/selection.json with per-file SHA-256 and metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Any

TARGET = 25
WAV_RE = re.compile(r"^\.?/?(MDT2021S\d+)/WAV/(?P<stem>.+\.wav)$")
TXT_RE = re.compile(r"^\.?/?(MDT2021S\d+)/TXT/(?P<stem>.+\.(txt|json))$")
SPK_RE = re.compile(r"^\.?/?(MDT2021S\d+)/SPKINFO\.txt$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pick_indices(total: int, count: int) -> list[int]:
    if total <= count:
        return list(range(total))
    return [round(i * (total - 1) / (count - 1)) for i in range(count)]


def select(archive_path: Path, outdir: Path) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
    wav_members = [m for m in members if WAV_RE.match(m.name)]
    if len(wav_members) < TARGET:
        raise ValueError(f"archive contains only {len(wav_members)} wav members")
    picks = {wav_members[i].name: i for i in _pick_indices(len(wav_members), TARGET)}
    stems = {}
    for name in picks:
        stem = WAV_RE.match(name).group("stem")[: -len(".wav")]
        stems[stem] = name
    txt_by_stem: dict[str, list[str]] = {}
    for m in members:
        match = TXT_RE.match(m.name)
        if match:
            stem = match.group("stem").rsplit(".", 1)[0]
            txt_by_stem.setdefault(stem, []).append(m.name)
    spk_members = sorted(m.name for m in members if SPK_RE.match(m.name))
    wanted = set(picks) | {name for stem, names in txt_by_stem.items() if stem in stems for name in names} | set(spk_members)

    records: list[dict[str, Any]] = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or member.name not in wanted:
                continue
            target = outdir / member.name.lstrip("./")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot extract {member.name}")
            with target.open("wb") as handle:
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    handle.write(chunk)
            records.append({
                "member": member.name,
                "tar_size": int(member.size),
                "path": str(target),
                "sha256": sha256_file(target),
                "role": "wav" if member.name in picks else ("spkinfo" if SPK_RE.match(member.name) else "transcript"),
                "archive_index": picks.get(member.name),
            })
    selection = {
        "schema_version": 1,
        "archive": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "total_conversation_wavs": len(wav_members),
        "sampling_rule": f"stride_over_archive_order_{TARGET}",
        "selected_indices": sorted(picks.values()),
        "selected_wavs": sorted(picks),
        "spkinfo_members": spk_members,
        "records": records,
        "wav_records": len([r for r in records if r["role"] == "wav"]),
        "transcript_records": len([r for r in records if r["role"] == "transcript"]),
        "spkinfo_records": len([r for r in records if r["role"] == "spkinfo"]),
    }
    (outdir / "selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps({
        "total_conversation_wavs": selection["total_conversation_wavs"],
        "selected": selection["wav_records"],
        "transcripts": selection["transcript_records"],
        "spkinfo": selection["spkinfo_records"],
    }, indent=2))
    return selection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    select(args.archive.resolve(), args.outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
