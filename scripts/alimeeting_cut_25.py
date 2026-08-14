#!/usr/bin/env python3
"""Cut 25 clean utterances from AliMeeting Eval near-field channels.

One utterance per speaker channel (25 channels = 25 distinct speakers).
Deterministic selection per channel: among intervals with duration in
[8, 20]s, pick the longest; fall back to [5, 30]s, then to the longest
overall (rule recorded per utterance). A 0.25s margin is added around the
interval, clamped to the channel bounds.

Outputs <outdir>/<sample_id:02d>.{wav,txt} plus manifest.json with
provenance (source channel/session/speaker, interval times, rule,
per-file SHA-256).
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

INTERVAL_RE = re.compile(
    r'intervals\s*\[\s*\d+\s*\]\s*:\s*xmin\s*=\s*([\d.]+)\s*'
    r'xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"',
    re.DOTALL,
)
PREFERRED_RANGE = (8.0, 20.0)
FALLBACK_RANGE = (5.0, 30.0)
MARGIN_S = 0.25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_textgrid(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    intervals: list[dict[str, Any]] = []
    for match in INTERVAL_RE.finditer(raw):
        start, end, text = float(match.group(1)), float(match.group(2)), match.group(3).strip()
        if text and end > start:
            intervals.append({"start_s": start, "end_s": end, "text": text, "duration_s": end - start})
    return intervals


def pick_interval(intervals: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    for rule, (low, high) in (("preferred_8_20s", PREFERRED_RANGE), ("fallback_5_30s", FALLBACK_RANGE)):
        candidates = [i for i in intervals if low <= i["duration_s"] <= high]
        if candidates:
            return max(candidates, key=lambda i: i["duration_s"]), rule
    chosen = max(intervals, key=lambda i: i["duration_s"])
    return chosen, "longest_overall"


def cut(audio_dir: Path, textgrid_dir: Path, outdir: Path) -> dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(audio_dir.glob("*.wav"))
    if len(wavs) != 25:
        raise ValueError(f"expected 25 near-field channels, found {len(wavs)}")
    records: list[dict[str, Any]] = []
    for index, wav in enumerate(wavs, 1):
        grid = textgrid_dir / (wav.stem + ".TextGrid")
        if not grid.is_file():
            raise FileNotFoundError(grid)
        intervals = parse_textgrid(grid)
        if not intervals:
            raise ValueError(f"no intervals in {grid}")
        interval, rule = pick_interval(intervals)
        audio, sample_rate = sf.read(wav, dtype="float32", always_2d=False)
        if sample_rate != 16000:
            raise ValueError(f"unexpected sample rate {sample_rate} in {wav}")
        start = max(0.0, interval["start_s"] - MARGIN_S)
        end = min(len(audio) / sample_rate, interval["end_s"] + MARGIN_S)
        frames = audio[int(start * sample_rate): int(end * sample_rate)]
        sample_id = f"{index:02d}"
        out_wav = outdir / f"{sample_id}.wav"
        out_txt = outdir / f"{sample_id}.txt"
        sf.write(out_wav, frames, sample_rate, subtype="FLOAT")
        out_txt.write_text(interval["text"] + "\n", encoding="utf-8")
        session = wav.stem.split("_")[0] + "_" + wav.stem.split("_")[1]
        speaker = wav.stem.split("_")[3]
        records.append({
            "sample_id": sample_id,
            "utterance_id": f"alimeeting_{sample_id}",
            "dataset": "alimeeting_eval",
            "speaker_id": speaker,
            "session": session,
            "transcript": interval["text"],
            "selection_rule": rule,
            "interval_start_s": interval["start_s"],
            "interval_end_s": interval["end_s"],
            "cut_start_s": round(start, 4),
            "cut_end_s": round(end, 4),
            "cut_duration_s": round(len(frames) / sample_rate, 4),
            "audio_path": str(out_wav),
            "audio_sha256": sha256_file(out_wav),
            "transcript_path": str(out_txt),
            "source_channel": str(wav),
            "source_channel_sha256": sha256_file(wav),
            "source_textgrid": str(grid),
            "source_textgrid_sha256": sha256_file(grid),
            "encoder_leakage": {
                "hubert_base_ls960": "none_english_only",
                "wavlm_large": "none_english_eu_only",
                "xls_r": "n_a_self_recorded_corpus",
            },
        })
    manifest = {
        "schema_version": 1,
        "dataset": "alimeeting_eval",
        "license": "CC BY-SA 4.0",
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
        "rules": {rule: sum(1 for r in records if r["selection_rule"] == rule) for rule in ("preferred_8_20s", "fallback_5_30s", "longest_overall")},
        "durations_s": [r["cut_duration_s"] for r in records],
    }, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--textgrid-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    cut(args.audio_dir.resolve(), args.textgrid_dir.resolve(), args.outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
