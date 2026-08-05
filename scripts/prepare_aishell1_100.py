#!/usr/bin/env python3
"""Prepare a deterministic 100-utterance AISHELL-1 test corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
import wave
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATASET = "TwinkStart/AISHELL-1"
REVISION = "2a509ef3a7a88d3234205eaeecba39fdd9d50518"
SOURCE_URL = "https://huggingface.co/datasets/TwinkStart/AISHELL-1"
ROWS_API = "https://datasets-server.huggingface.co/rows"


def _speaker(name: str) -> str:
    match = re.search(r"(S\d+)", name)
    if not match:
        raise ValueError(f"cannot parse speaker from {name!r}")
    return match.group(1)


def _fetch_rows(offset: int, length: int = 100) -> list[dict]:
    query = urllib.parse.urlencode(
        {"dataset": DATASET, "config": "default", "split": "test", "offset": offset, "length": length}
    )
    with urllib.request.urlopen(f"{ROWS_API}?{query}", timeout=120) as response:
        return json.load(response)["rows"]


def load_rows(cache_path: Path | None) -> list[dict]:
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for offset in range(0, 7200, 100):
        try:
            batch = _fetch_rows(offset)
        except Exception:
            continue
        rows.extend(batch)
        if len(batch) < 100:
            break
    rows.sort(key=lambda item: item["row_idx"])
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def select_rows(rows: list[dict], count: int, seed: int = 42) -> list[dict]:
    del seed
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for item in rows:
        row = item["row"]
        by_speaker[_speaker(row["name"])].append(item)
    speakers = sorted(by_speaker)
    if len(speakers) < 10:
        raise ValueError(f"only found {len(speakers)} speakers in metadata")
    per_speaker = count // len(speakers)
    remainder = count % len(speakers)
    selected: list[dict] = []
    for index, speaker in enumerate(speakers):
        bucket = sorted(by_speaker[speaker], key=lambda item: item["row"]["name"])
        target = per_speaker + (1 if index < remainder else 0)
        if len(bucket) < target:
            raise ValueError(f"speaker {speaker} has only {len(bucket)} rows")
        for position in range(target):
            selected.append(bucket[round(position * (len(bucket) - 1) / max(target - 1, 1))])
    return selected


def _download(item: tuple[int, dict, Path]) -> dict:
    sample_id, item_data, output_path = item
    row = item_data["row"]
    source = row["audio"][0]["src"]
    with urllib.request.urlopen(source, timeout=120) as response:
        data = response.read()
    output_path.write_bytes(data)
    with wave.open(str(output_path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        duration_s = handle.getnframes() / sample_rate
    return {
        "sample_id": sample_id,
        "source_utterance_id": row["name"],
        "source_path": row["WavPath"],
        "speaker_id": _speaker(row["name"]),
        "transcript": "".join(row["text"].split()),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "duration_s": round(duration_s, 6),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "audio_path": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/aishell1_100_zh"))
    parser.add_argument("--metadata-cache", type=Path, default=Path("tmp/aishell1_test_rows.json"))
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")

    rows = load_rows(args.metadata_cache)
    selected = select_rows(rows, args.count)
    audio_dir = args.output / "audio"
    transcript_dir = args.output / "transcripts"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(index, item, audio_dir / f"{index}.wav") for index, item in enumerate(selected, 1)]
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_download, job) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: item["sample_id"])

    for record in records:
        (transcript_dir / f"{record['sample_id']}.txt").write_text(
            record["transcript"] + "\n", encoding="utf-8"
        )
    payload = {
        "schema_version": 1,
        "dataset": DATASET,
        "dataset_revision": REVISION,
        "source_url": SOURCE_URL,
        "split": "test",
        "selection": "all available test-set speakers, evenly sampled per speaker",
        "count": len(records),
        "records": records,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "count": len(records),
        "speakers": len({record['speaker_id'] for record in records}),
        "min_duration_s": min(record['duration_s'] for record in records),
        "max_duration_s": max(record['duration_s'] for record in records),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
