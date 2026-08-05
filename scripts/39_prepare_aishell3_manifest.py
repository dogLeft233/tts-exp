#!/usr/bin/env python3
"""Prepare a deterministic, representation-only AISHELL-3 manifest.

This adapter deliberately does not create TTS pairs or TFG records.  Its output
is intended for speaker-held-out SSL and phonetic representation analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import wave
from collections import defaultdict
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import validate_manifest, write_manifest

logger = logging.getLogger(__name__)

_SPLITS = ("train", "dev", "test")
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_component(value: str) -> str:
    return _SAFE.sub("_", value).strip("._") or "unknown"


def _read_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[1].strip():
            raise ValueError(f"invalid label at {path}:{line_number}")
        utterance_id, transcript = parts
        if utterance_id in labels:
            raise ValueError(f"duplicate utterance label {utterance_id!r} in {path}")
        labels[utterance_id] = transcript.strip()
    return labels


def _find_labels(dataset_root: Path, split: str, labels: Path | None) -> Path:
    if labels is not None:
        return labels if labels.is_absolute() else dataset_root / labels
    candidates = []
    for split_name in (_SPLITS if split == "all" else (split,)):
        candidates.extend(
            dataset_root / split_name / name
            for name in ("content.txt", "transcript.txt", "transcripts.txt")
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("could not find AISHELL-3 content.txt; pass --labels")


def _find_audio(dataset_root: Path, split: str, utterance_id: str) -> Path | None:
    split_names = _SPLITS if split == "all" else (split,)
    for split_name in split_names:
        roots = [
            dataset_root / split_name / "wav",
            dataset_root / split_name / "audio",
            dataset_root / split_name,
            dataset_root / "wav",
            dataset_root / "audio",
        ]
        for root in roots:
            matches = list(root.glob(f"**/{utterance_id}.wav")) if root.exists() else []
            if matches:
                return sorted(matches)[0]
    return None


def _speaker_id(audio_path: Path, dataset_root: Path, utterance_id: str) -> str:
    try:
        relative = audio_path.relative_to(dataset_root)
        for component in relative.parts[:-1]:
            if component.upper().startswith("S"):
                return component
        if len(relative.parts) >= 2:
            return relative.parts[-2]
    except ValueError:
        pass
    prefix = utterance_id.split("S", 1)
    return f"S{prefix[1][:4]}" if len(prefix) == 2 and len(prefix[1]) >= 4 else "unknown"


def _audio_info(path: Path) -> tuple[int, float]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        duration_s = handle.getnframes() / float(sample_rate) if sample_rate else 0.0
    return sample_rate, duration_s


def _uniform_tokens(transcript: str, duration_s: float) -> list[dict]:
    tokens = list(transcript.replace(" ", ""))
    if not tokens or duration_s <= 0:
        return []
    step = duration_s / len(tokens)
    return [
        {
            "token": token,
            "initial": "",
            "final": "",
            "tone": 0,
            "start_s": i * step,
            "end_s": (i + 1) * step,
            "confidence": 0.0,
            "viseme": "other",
        }
        for i, token in enumerate(tokens)
    ]


def _select_records(records: list[dict], speaker_limit: int | None, max_utterances: int | None, seed: int) -> list[dict]:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_speaker[record["speaker_id"]].append(record)
    speakers = sorted(by_speaker)
    if speaker_limit is not None:
        if speaker_limit <= 0:
            raise ValueError("--speaker-limit must be positive")
        rng = random.Random(seed)
        rng.shuffle(speakers)
        speakers = sorted(speakers[:speaker_limit])
    selected = []
    buckets = [by_speaker[speaker] for speaker in speakers]
    for bucket in buckets:
        bucket.sort(key=lambda item: item["utterance_id"])
    if max_utterances is None:
        return [item for bucket in buckets for item in bucket]
    if max_utterances <= 0:
        raise ValueError("--max-utterances must be positive")
    offsets = [0] * len(buckets)
    while len(selected) < max_utterances:
        added = False
        for bucket_index, bucket in enumerate(buckets):
            if len(selected) >= max_utterances:
                break
            if offsets[bucket_index] < len(bucket):
                selected.append(bucket[offsets[bucket_index]])
                offsets[bucket_index] += 1
                added = True
        if not added:
            break
    return selected


def build_manifest(
    dataset_root: Path,
    split: str = "train",
    labels_path: Path | None = None,
    speaker_limit: int | None = None,
    max_utterances: int | None = None,
    seed: int = 42,
    allow_missing_text: bool = False,
    uniform_fallback: bool = False,
    check_audio: bool = True,
) -> list[dict]:
    if split not in {*_SPLITS, "all"}:
        raise ValueError(f"invalid split {split!r}")
    dataset_root = dataset_root.resolve()
    split_names = _SPLITS if split == "all" else (split,)
    records: list[dict] = []
    for split_name in split_names:
        split_root = dataset_root / split_name
        if not split_root.exists():
            raise FileNotFoundError(f"split directory not found: {split_root}")
        current_labels = _read_labels(_find_labels(dataset_root, split_name, labels_path))
        for raw_utt, transcript in sorted(current_labels.items()):
            audio_path = _find_audio(dataset_root, split_name, raw_utt)
            if audio_path is None:
                if check_audio:
                    raise FileNotFoundError(f"audio not found for {split_name}/{raw_utt}")
                continue
            sample_id = f"{split_name}/{raw_utt}"
            speaker = _speaker_id(audio_path, dataset_root, raw_utt)
            sample_rate, duration_s = _audio_info(audio_path) if check_audio else (16000, 0.0)
            tokens = _uniform_tokens(transcript, duration_s) if uniform_fallback else []
            records.append(
                {
                    "dataset": "aishell3",
                    "utterance_id": sample_id,
                    "sample_id": sample_id,
                    "speaker_id": speaker,
                    "paired_key": None,
                    "condition": "natural",
                    "tts_provider": None,
                    "variant": "raw",
                    "transcript": transcript,
                    "text": transcript,
                    "audio_path": str(audio_path),
                    "filepath": str(audio_path),
                    "split": split_name,
                    "alignment_source": "uniform_fallback" if uniform_fallback else "missing",
                    "alignment_confidence": 0.0 if uniform_fallback else None,
                    "sample_rate": sample_rate,
                    "duration_s": duration_s,
                    "license": "Apache-2.0",
                    "representation_only": True,
                    "tokens": tokens,
                    "embedding_stem": "aishell3_{}_{}_{}".format(
                        _safe_component(split_name), _safe_component(speaker), _safe_component(raw_utt)
                    ),
                }
            )
    records = _select_records(records, speaker_limit, max_utterances, seed)
    if not allow_missing_text and any(not record["transcript"].strip() for record in records):
        raise ValueError("manifest contains an empty transcript")
    validate_manifest(records)
    return records


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", "--root", required=True, type=Path)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--split", choices=[*_SPLITS, "all"], default="train")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--speaker-limit", type=int, default=None)
    parser.add_argument("--max-utterances", "--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing-text", action="store_true")
    parser.add_argument("--uniform-fallback", action="store_true")
    parser.add_argument("--no-check-audio", dest="check_audio", action="store_false")
    parser.set_defaults(check_audio=True)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        records = build_manifest(
            args.dataset_root,
            split=args.split,
            labels_path=args.labels,
            speaker_limit=args.speaker_limit,
            max_utterances=args.max_utterances,
            seed=args.seed,
            allow_missing_text=args.allow_missing_text,
            uniform_fallback=args.uniform_fallback,
            check_audio=args.check_audio,
        )
        if args.output is not None and not args.check_only:
            write_manifest(args.output, records)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    summary = {
        "dataset": "aishell3",
        "split": args.split,
        "entries": len(records),
        "speakers": len({record["speaker_id"] for record in records}),
        "alignment_source": "uniform_fallback" if args.uniform_fallback else "missing",
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
