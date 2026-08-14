#!/usr/bin/env python3
"""Streaming partial extraction of the RAMC / AliMeeting pilot archives.

Usage (curl pipes the archive into stdin; the script aborts the stream once
the stop condition is met, which kills curl via SIGPIPE — nothing else of the
archive is downloaded or stored):

    curl -s URL | python scripts/pilot_partial_download.py ramc \
        --outdir data/ramc_alimeeting_pilot50/ramc --budget 6000000000
    curl -s URL | python scripts/pilot_partial_download.py alimeeting \
        --outdir data/ramc_alimeeting_pilot50/alimeeting --budget 3600000000

Selection rules (deterministic, archive order):

- ramc: members matching `MDT2021S<id>/WAV/*.wav`, `MDT2021S<id>/TXT/*.txt`
  and `MDT2021S<id>/SPKINFO.txt`. Take at most 2 wavs per conversation dir
  (spread across speakers), extract each wav's same-stem .txt if present,
  stop after 25 wavs or when the byte budget is consumed.
- alimeeting: members matching `Eval_Ali_near/audio_dir/*.wav` and
  `Eval_Ali_near/textgrid_file/*.TextGrid`. Extract all near channels until
  the byte budget is consumed.

Provenance (source url, byte budget/consumed, member sizes, per-file SHA-256,
truncated flag) is written to <outdir>/provenance.json.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

RAMC_WAV_RE = re.compile(r"^\.?/?(MDT2021S\d+)/WAV/(.+\.wav)$")
RAMC_TXT_RE = re.compile(r"^\.?/?(MDT2021S\d+)/TXT/(.+\.txt)$")
RAMC_SPK_RE = re.compile(r"^\.?/?(MDT2021S\d+)/SPKINFO\.txt$")
ALI_WAV_RE = re.compile(r"^\.?/?Eval_Ali/Eval_Ali_near/audio_dir/(.+\.wav)$")
ALI_GRID_RE = re.compile(r"^\.?/?Eval_Ali/Eval_Ali_near/textgrid_file/(.+\.TextGrid)$")

RAMC_TARGET_WAVS = 25
RAMC_MAX_WAVS_PER_CONVERSATION = 2


class CountingReader:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.count = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.stream.read(size)
        self.count += len(chunk)
        return chunk


class StopExtraction(Exception):
    pass


@dataclass
class Extraction:
    outdir: Path
    budget: int
    records: list[dict[str, Any]] = field(default_factory=list)
    reader: CountingReader | None = None
    skipped_members_log: list[str] = field(default_factory=list)

    def _sha256_stream(self, member: tarfile.TarInfo, extractor: Any) -> tuple[Path, str]:
        digest = hashlib.sha256()
        target = self.outdir / member.name.lstrip("./")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            while True:
                chunk = extractor.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
        return target, digest.hexdigest()

    def add(self, member: tarfile.TarInfo, extractor: Any) -> None:
        target, sha256 = self._sha256_stream(member, extractor)
        self.records.append({
            "member": member.name,
            "tar_size": int(member.size),
            "path": str(target),
            "sha256": sha256,
            "stream_bytes_consumed_at_start": self.reader.count if self.reader else 0,
        })

    def finish(self, reason: str, truncated: bool) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "reason": reason,
            "truncated": truncated,
            "byte_budget": self.budget,
            "stream_bytes_consumed": self.reader.count if self.reader else 0,
            "extracted_members": len(self.records),
            "records": self.records,
        }
        write_json(self.outdir / "provenance.json", payload)
        return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _run(mode: str, outdir: Path, budget: int) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    reader = CountingReader(sys.stdin.buffer)
    extraction = Extraction(outdir=outdir, budget=budget, reader=reader)
    reason = "completed_archive"
    truncated = False
    conversation_counts: dict[str, int] = {}
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as archive:  # type: ignore[arg-type]
            for member in archive:
                if member.isfile():
                    if mode == "ramc":
                        wav = RAMC_WAV_RE.match(member.name)
                        txt = RAMC_TXT_RE.match(member.name)
                        spk = RAMC_SPK_RE.match(member.name)
                        if wav:
                            conversation = wav.group(1)
                            count = conversation_counts.get(conversation, 0)
                            if count < RAMC_MAX_WAVS_PER_CONVERSATION and len(extraction.records) < RAMC_TARGET_WAVS:
                                extraction.add(member, archive.extractfile(member))
                                conversation_counts[conversation] = count + 1
                                if len(extraction.records) >= RAMC_TARGET_WAVS:
                                    raise StopExtraction("ramc_target_25_wavs_reached")
                            else:
                                extraction.skipped_members_log.append(member.name)
                        elif txt:
                            # Stream mode cannot defer member reads, so extract
                            # every .txt unconditionally — they are tiny and
                            # guarantee each selected wav has its transcript
                            # regardless of archive order.
                            extraction.add(member, archive.extractfile(member))
                        elif spk:
                            extraction.add(member, archive.extractfile(member))
                        else:
                            extraction.skipped_members_log.append(member.name)
                    elif mode == "alimeeting":
                        if ALI_WAV_RE.match(member.name) or ALI_GRID_RE.match(member.name):
                            extraction.add(member, archive.extractfile(member))
                        else:
                            extraction.skipped_members_log.append(member.name)
                    else:
                        raise ValueError(f"unsupported mode: {mode}")
                if reader.count >= budget:
                    raise StopExtraction("byte_budget_exhausted")
    except StopExtraction as stop:
        reason = str(stop)
    except (tarfile.ReadError, EOFError, OSError) as error:
        truncated = True
        reason = f"archive_stream_ended: {type(error).__name__}"
    payload = extraction.finish(reason, truncated)
    (outdir / "members.log").write_text("\n".join(extraction.skipped_members_log[:2000]), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("ramc", "alimeeting"))
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=6_000_000_000, help="stream byte budget")
    args = parser.parse_args(argv)
    _run(args.mode, args.outdir.resolve(), args.budget)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
