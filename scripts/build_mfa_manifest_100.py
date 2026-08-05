#!/usr/bin/env python3
"""Build a phoneme-analysis manifest from MFA TextGrid outputs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml
from pypinyin import Style, pinyin

_INITIALS = ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w")


def split_pinyin(raw: str) -> tuple[str, str, int]:
    base = raw.strip()
    tone = 5
    if base and base[-1].isdigit():
        tone = int(base[-1])
        base = base[:-1]
    initial = "∅"
    final = base
    for candidate in _INITIALS:
        if base.startswith(candidate):
            initial = candidate
            final = base[len(candidate):]
            break
    if final == "i" and initial in {"zh", "ch", "sh", "r", "z", "c", "s"}:
        final = "-i"
    return initial, final, tone


def viseme(initial: str) -> str:
    if initial in {"b", "p", "m"}:
        return "bilabial"
    if initial == "f":
        return "labiodental"
    if initial == "w":
        return "rounded_vowel"
    return "nonlabial_consonant"


def parse_textgrid(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    tiers: dict[str, list[tuple[float, float, str]]] = {}
    current: str | None = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("name = "):
            current = line.split("=", 1)[1].strip().strip('"')
            tiers.setdefault(current, [])
        elif current and line.startswith("xmin = ") and i + 2 < len(lines):
            next_line = lines[i + 1].strip()
            text_line = lines[i + 2].strip()
            if next_line.startswith("xmax = ") and text_line.startswith("text = "):
                start = float(line.split("=", 1)[1])
                end = float(next_line.split("=", 1)[1])
                label = text_line.split("=", 1)[1].strip().strip('"')
                tiers[current].append((start, end, label))
                i += 2
        i += 1
    return tiers


def nonempty(intervals: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    return [(a, b, t) for a, b, t in intervals if t not in {"", "<eps>", "sil", "sp", "spn"}]


def build_tokens(text: str, tiers: dict[str, list[tuple[float, float, str]]]) -> tuple[list[dict], str]:
    chars = list("".join(text.split()))
    readings = pinyin("".join(chars), style=Style.TONE3)
    words = nonempty(tiers.get("words", []))
    phones = nonempty(tiers.get("phones", []))
    if not chars or not words:
        return [], "mfa_textgrid_missing_words"

    tokens: list[dict] = []
    char_index = 0
    used_fallback = False
    for word_start, word_end, word in words:
        word_chars = list("".join(word.split()))
        if not word_chars:
            continue
        word_phones = [(a, b, t) for a, b, t in phones if a >= word_start - 1e-5 and b <= word_end + 1e-5]
        if len(word_phones) < len(word_chars):
            used_fallback = True
            word_phones = []
        if word_phones:
            # MFA phones provide the true speech span. Distribute only the
            # within-word boundary among characters when phone-to-character
            # segmentation is not represented explicitly in the TextGrid.
            phone_edges = [word_start] + [b for _, b, _ in word_phones]
            for offset, _char in enumerate(word_chars):
                start_pos = round(offset * len(word_phones) / len(word_chars))
                end_pos = round((offset + 1) * len(word_phones) / len(word_chars))
                start = phone_edges[start_pos]
                end = phone_edges[end_pos]
                if end <= start:
                    used_fallback = True
                    start = word_start + (word_end - word_start) * offset / len(word_chars)
                    end = word_start + (word_end - word_start) * (offset + 1) / len(word_chars)
                if char_index < len(readings):
                    initial, final, tone = split_pinyin(readings[char_index][0])
                else:
                    initial, final, tone = "∅", "", 5
                tokens.append({"token": chars[char_index], "initial": initial, "final": final, "tone": tone, "viseme": viseme(initial), "start_s": round(start, 6), "end_s": round(end, 6), "confidence": 1.0})
                char_index += 1
        else:
            used_fallback = True
            for offset, _char in enumerate(word_chars):
                start = word_start + (word_end - word_start) * offset / len(word_chars)
                end = word_start + (word_end - word_start) * (offset + 1) / len(word_chars)
                initial, final, tone = split_pinyin(readings[char_index][0])
                tokens.append({"token": chars[char_index], "initial": initial, "final": final, "tone": tone, "viseme": viseme(initial), "start_s": round(start, 6), "end_s": round(end, 6), "confidence": 0.5})
                char_index += 1

    if char_index != len(chars):
        return [], "mfa_textgrid_character_mismatch"
    return tokens, "mfa_textgrid_word_phone_boundary" if not used_fallback else "mfa_textgrid_word_boundary_fallback"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    base = args.base
    records: list[dict] = []
    failures: list[dict] = []
    for sample_id in range(1, 101):
        text = (base / "aishell1_100_zh" / "transcripts" / f"{sample_id}.txt").read_text(encoding="utf-8").strip()
        for condition in ("natural", "tts"):
            grid = base / "mfa_full" / "out" / condition / f"{sample_id}.TextGrid"
            try:
                tokens, source = build_tokens(text, parse_textgrid(grid))
                if not tokens:
                    raise ValueError(source)
                audio = base / "aishell1_100_zh" / "audio" / f"{sample_id}.wav" if condition == "natural" else base / "run" / "02_tts" / f"{sample_id}.wav"
                records.append({"schema_version": 1, "dataset": "aishell1_100_zh", "utterance_id": f"{sample_id}:{condition}:raw", "speaker_id": f"aishell1_{sample_id}", "paired_key": str(sample_id), "sample_id": sample_id, "condition": "natural" if condition == "natural" else "tts", "tts_provider": None if condition == "natural" else "faster_qwen3", "variant": "raw", "transcript": text, "audio_path": str(audio), "filepath": str(audio), "split": "test", "license": "Apache-2.0", "alignment_source": "mfa", "alignment_method": source, "alignment_confidence": min(float(t["confidence"]) for t in tokens), "sample_rate": 16000 if condition == "natural" else 24000, "tokens": tokens})
            except Exception as exc:
                failures.append({"sample_id": sample_id, "condition": condition, "error": str(exc)})
    out = base / "alignment" / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema_version": 1, "records": records, "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "failures": len(failures), "natural": sum(r["condition"] == "natural" for r in records), "tts": sum(r["condition"] == "tts" for r in records), "fallback_records": sum("fallback" in r["alignment_source"] for r in records)}, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
