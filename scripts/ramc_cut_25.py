#!/usr/bin/env python3
"""Cut 25 utterances from the 25 selected RAMC conversations.

Each conversation .wav is a ~32-minute dyadic recording and its .txt holds
per-turn lines: `[start,end]\tSPEAKER_ID\t方言,普通话\ttext`.

One utterance per conversation (25 conversations = up to 50 speakers).
Deterministic selection: among turns labeled 普通话 with duration in
[8, 20]s pick the longest; fall back to [5, 30]s; then to the longest
overall. A 0.25s margin is added around the turn, clamped to bounds.

Outputs <outdir>/<sample_id:02d>.{wav,txt} plus manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

LINE_RE = re.compile(r"^\[([\d.]+),([\d.]+)\]\s+(\S+)\s+(\S+)\s+(.+)$")
PREFERRED_RANGE = (8.0, 20.0)
FALLBACK_RANGE = (5.0, 30.0)
MARGIN_S = 0.25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_turns(path: Path) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE_RE.match(line.strip())
        if not match:
            continue
        start, end, speaker, dialect, text = float(match.group(1)), float(match.group(2)), match.group(3), match.group(4), match.group(5).strip()
        if text and end > start:
            turns.append({
                "start_s": start, "end_s": end, "speaker_id": speaker,
                "dialect": dialect, "text": text, "duration_s": end - start,
                "putonghua": "普通话" in dialect,
            })
    return turns


def pick_turn(turns: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    for rule, (low, high) in (("preferred_8_20s", PREFERRED_RANGE), ("fallback_5_30s", FALLBACK_RANGE)):
        candidates = [t for t in turns if t["putonghua"] and low <= t["duration_s"] <= high]
        if candidates:
            return max(candidates, key=lambda t: t["duration_s"]), rule
        candidates = [t for t in turns if low <= t["duration_s"] <= high]
        if candidates:
            return max(candidates, key=lambda t: t["duration_s"]), rule + "_any_dialect"
    chosen = max(turns, key=lambda t: t["duration_s"])
    return chosen, "longest_overall"


def cut(selection_path: Path, outdir: Path) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    wav_records = [r for r in selection["records"] if r["role"] == "wav"]
    if len(wav_records) != 25:
        raise ValueError(f"expected 25 selected conversations, found {len(wav_records)}")
    records: list[dict[str, Any]] = []
    for index, wav_record in enumerate(wav_records, 1):
        wav = Path(wav_record["path"])
        txt = wav.parent.parent / "TXT" / (wav.stem + ".txt")
        if not txt.is_file():
            raise FileNotFoundError(txt)
        turns = parse_turns(txt)
        if not turns:
            raise ValueError(f"no turns in {txt}")
        turn, rule = pick_turn(turns)
        audio, sample_rate = sf.read(wav, dtype="float32", always_2d=False)
        if sample_rate != 16000:
            raise ValueError(f"unexpected sample rate {sample_rate} in {wav}")
        start = max(0.0, turn["start_s"] - MARGIN_S)
        end = min(len(audio) / sample_rate, turn["end_s"] + MARGIN_S)
        frames = audio[int(start * sample_rate): int(end * sample_rate)]
        sample_id = f"{index:02d}"
        out_wav = outdir / f"{sample_id}.wav"
        out_txt = outdir / f"{sample_id}.txt"
        sf.write(out_wav, frames, sample_rate, subtype="FLOAT")
        out_txt.write_text(turn["text"] + "\n", encoding="utf-8")
        conversation = wav.stem
        records.append({
            "sample_id": sample_id,
            "utterance_id": f"ramc_{sample_id}",
            "dataset": "magicdata_ramc",
            "conversation": conversation,
            "speaker_id": turn["speaker_id"],
            "dialect": turn["dialect"],
            "transcript": turn["text"],
            "selection_rule": rule,
            "turn_start_s": turn["start_s"],
            "turn_end_s": turn["end_s"],
            "cut_start_s": round(start, 4),
            "cut_end_s": round(end, 4),
            "cut_duration_s": round(len(frames) / sample_rate, 4),
            "audio_path": str(out_wav),
            "audio_sha256": sha256_file(out_wav),
            "transcript_path": str(out_txt),
            "source_conversation_wav": str(wav),
            "source_conversation_wav_sha256": sha256_file(wav),
            "source_turn_transcript": str(txt),
            "encoder_leakage": {
                "hubert_base_ls960": "none_english_only",
                "wavlm_large": "none_english_eu_only",
                "xls_r": "n_a_self_recorded_corpus",
            },
        })
    manifest = {
        "schema_version": 1,
        "dataset": "magicdata_ramc",
        "license": "CC BY-NC-ND 4.0",
        "recorded": "in_house_recordings_zero_web_scrape_overlap",
        "sample_count": len(records),
        "speaker_count": len({r["speaker_id"] for r in records}),
        "records": records,
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    print(json.dumps({
        "samples": len(records),
        "speakers": manifest["speaker_count"],
        "rules": {rule: sum(1 for r in records if r["selection_rule"] == rule) for rule in sorted({r["selection_rule"] for r in records})},
        "putonghua_turns": sum(1 for r in records if "普通话" in r["dialect"]),
        "durations_s": [r["cut_duration_s"] for r in records],
    }, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    cut(args.selection.resolve(), args.outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
