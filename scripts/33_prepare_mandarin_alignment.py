#!/usr/bin/env python3
"""Build Mandarin Wav2Sem alignment manifest from MFA TextGrids.

Reads MFA TextGrid phone tiers (IPA with tones), strips tones,
maps IPA phones to Preston Blair 13 visemes, and writes a manifest
compatible with scripts 15 (SSL embeddings) and 16 (separability).

Usage
-----
    python scripts/33_prepare_mandarin_alignment.py
        [--mfa-dir DIR] [--output-dir DIR] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg_feature_common import OUTPUT_BASE_ZH

NATURAL_AUDIO_DIR = Path("data/data/audio")
TTS_AUDIO_DIR = Path("runs/r2_dashscope_vc_20260707T143930Z/02_tts")
DEFAULT_MFA_DIR = Path("/root/autodl-tmp/mandarin_output")
DEFAULT_OUTPUT_DIR = OUTPUT_BASE_ZH / "manifest"

SILENCE_LABELS = frozenset({"", "sp", "sil", "SIL", "SPN", "spn", "<sil>", "<unk>"})

LOG_REX = re.compile(r"xmin\s*=\s*([\d.]+)\s*\n\s*xmax\s*=\s*([\d.]+)\s*\n\s*text\s*=\s*\"([^\"]*)\"", re.DOTALL)

IPA_TO_VISEME: dict[str, str] = {
    # Silence
    "": "sil", "sp": "sil", "sil": "sil", "spn": "sil",
    # /pbmv/ — bilabial
    "p": "pbmv", "pʰ": "pbmv", "pʲ": "pbmv", "pʷ": "pbmv",
    "m": "pbmv", "mʲ": "pbmv",
    # /fv/ — labiodental
    "f": "fv",
    # /th/ — dental/interdental (none in Mandarin)
    # /cdsz/ — alveolar
    "t": "cdsz", "tʰ": "cdsz", "tʲ": "cdsz", "tʷ": "cdsz",
    "ts": "cdsz", "tsʰ": "cdsz",
    "s": "cdsz", "n": "cdsz", "l": "cdsz",
    # /kg/ — velar
    "k": "kg", "kʰ": "kg", "kʷ": "kg",
    "x": "kg", "ŋ": "kg",
    # /chjsh/ — palatal/retroflex
    "tɕ": "chjsh", "tɕʰ": "chjsh", "ɕ": "chjsh",
    "ʂ": "chjsh", "ʈʂ": "chjsh", "ʈʂʰ": "chjsh",
    "ɻ": "chjsh", "ʐ": "chjsh",
    # /e/ — mid front lax
    "e": "e", "ej": "e", "ə": "e",
    # /o/ — mid back rounded
    "o": "o", "ow": "o",
    # /i/ — close front spread (wide stretch)
    "i": "i", "j": "i",
    # /u/ — close back rounded (small round)
    "u": "u", "w": "u",
    # /r/ — retroflex (erhua)
    "ɚ": "r",
    # /ai/ — open vowel (jaw drop)
    "a": "ai", "aj": "ai",
    # /aw/ — open back
    "aw": "aw",
    # /y/ — close front rounded (ü) → rounded like /u/
    "y": "u",
    "ɥ": "u",
    # Syllabic consonants
    "z̩": "cdsz",
}

TONE_STRIP = re.compile(r"[˥˦˧˨˩]+")


def strip_tone(ipa: str) -> str:
    """Remove tone diacritics from IPA string."""
    return TONE_STRIP.sub("", ipa)


def ipa_to_viseme(ipa: str) -> str:
    """Map IPA phone (with optional tone) to Preston Blair viseme class."""
    if ipa in SILENCE_LABELS:
        return "sil"
    bare = strip_tone(ipa)
    if not bare:
        return "e"  # rare: empty after tone strip (shouldn't happen)
    if bare in IPA_TO_VISEME:
        return IPA_TO_VISEME[bare]
    # try removing diacritics
    for key, vis in IPA_TO_VISEME.items():
        if len(key) == 1 and bare.startswith(key):
            return vis
    return "e"


def parse_textgrid(path: Path) -> list[dict]:
    """Parse MFA TextGrid phone tier → list of {token, viseme, start_s, end_s}."""
    content = path.read_text(encoding="utf-8")
    tokens: list[dict] = []

    # Find phone tier boundaries
    import re as _re
    phone_start = content.index('name = "phones"')
    # Find next tier start (end of phones tier)
    try:
        next_item = content.index('item [', phone_start + 1)
    except ValueError:
        next_item = len(content)
    phone_section = content[phone_start:next_item]

    blocks = phone_section.split("intervals [")
    for block in blocks[1:]:
        m = LOG_REX.search(block)
        if not m:
            continue
        start_s = float(m.group(1))
        end_s = float(m.group(2))
        phone = m.group(3)
        if phone in {"", "sp", "sil", "spn", "SPN"}:
            continue
        viseme = ipa_to_viseme(phone)
        tokens.append({
            "token": phone,
            "viseme": viseme,
            "start_s": start_s,
            "end_s": end_s,
            "confidence": 1.0,
        })
    return tokens


def build_manifest(
    sample_ids: list[int],
    mfa_dir: Path,
    output_dir: Path,
) -> list[dict]:
    """Build manifest entries for natural and TTS conditions."""
    import csv, wave

    # Load texts from AISHELL-1 CSV
    texts: dict[int, str] = {}
    csv_path = Path(__file__).resolve().parent.parent / "data" / "test.csv"
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader, 1):
            if i <= 13:
                texts[i] = row[1].replace(" ", "")
            else:
                break

    entries: list[dict] = []
    for sid in sample_ids:
        nat_wav = Path(str(NATURAL_AUDIO_DIR)) / f"{sid}.wav"
        tts_wav = Path(str(TTS_AUDIO_DIR)) / f"{sid}.wav"

        for condition, wav_path, mfa_suffix in [
            ("natural", nat_wav, f"{sid}.TextGrid"),
            ("tts", tts_wav, f"{sid}.TextGrid"),
        ]:
            tg_path = mfa_dir / mfa_suffix
            if not tg_path.exists():
                print(f"  [SKIP] {condition}/{sid}: TextGrid not found at {tg_path}")
                continue

            # Duration
            with wave.open(str(wav_path), "rb") as wf:
                duration_s = wf.getnframes() / wf.getframerate()

            tokens = parse_textgrid(tg_path)
            if not tokens:
                print(f"  [WARN] {condition}/{sid}: no tokens parsed")
                continue

            unique_visemes = len(set(t["viseme"] for t in tokens))
            print(f"  {condition}/{sid:>2}: {len(tokens)} phones, {unique_visemes} visemes, {duration_s:.1f}s")

            entries.append({
                "sample_id": sid,
                "condition": condition,
                "variant": "raw",
                "filepath": str(wav_path),
                "duration_s": round(duration_s, 3),
                "text": texts.get(sid, ""),
                "tokens": tokens,
            })

    return entries


def main():
    parser = argparse.ArgumentParser(description="Build Mandarin Wav2Sem alignment manifest")
    parser.add_argument("--mfa-dir", default=str(DEFAULT_MFA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--smoke", action="store_true", help="Only first 3 samples")
    args = parser.parse_args()

    mfa_dir = Path(args.mfa_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from tfg_feature_common import CHINESE_STUDY_SAMPLES
    sample_ids = CHINESE_STUDY_SAMPLES[:3] if args.smoke else CHINESE_STUDY_SAMPLES

    print(f"MFAs: {mfa_dir}  ({len(list(mfa_dir.glob('*.TextGrid')))} found)")
    print(f"Output: {output_dir}")
    print(f"Samples: {sample_ids}")

    entries = build_manifest(sample_ids, mfa_dir, output_dir)

    # Write
    output_path = output_dir / "alignment.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(entries)} entries → {output_path}")

    # Summary
    viseme_counts: dict[str, dict[str, int]] = {}
    unmapped: set[str] = set()
    for e in entries:
        for t in e["tokens"]:
            v = t["viseme"]
            if v not in viseme_counts:
                viseme_counts[v] = {"natural": 0, "tts": 0}
            viseme_counts[v][e["condition"]] += 1
            if v == "e" and t["token"] not in SILENCE_LABELS and strip_tone(t["token"]) not in IPA_TO_VISEME:
                unmapped.add(t["token"])

    print("\nViseme distribution:")
    for vis, counts in sorted(viseme_counts.items()):
        print(f"  {vis:>6}: nat={counts['natural']:>4}, tts={counts['tts']:>4}")

    if unmapped:
        print(f"\nUnmapped IPA phones ({len(unmapped)}): {sorted(unmapped)}")


if __name__ == "__main__":
    main()
