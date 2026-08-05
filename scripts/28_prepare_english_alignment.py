#!/usr/bin/env python3
"""Build English Wav2Sem alignment manifest from MFA TextGrid alignment.

Runs MFA forced alignment (or reads pre-existing TextGrids), parses the
phone tier, maps IPA phones to Preston Blair 13 visemes, and writes a
manifest compatible with scripts 15 (SSL embeddings) and 16 (separability).

Each manifest entry matches the Chinese manifest schema:

    {
      "sample_id": 1,
      "condition": "natural",
      "variant": "raw",
      "filepath": "data/data/audio_en/1.wav",
      "duration_s": 5.765,
      "text": "...",
      "tokens": [
        {"token": "w", "viseme": "cdsz", "start_s": 0.0,
         "end_s": 0.12, "confidence": 1.0},
        ...
      ]
    }

Usage
-----
    python scripts/28_prepare_english_alignment.py [--samples IDS] [--smoke]
        [--mfa-dir DIR] [--output-dir DIR] [--run-mfa] [--mfa-env NAME]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg_feature_common import ENGLISH_STUDY_SAMPLES, OUTPUT_BASE_EN

try:
    import yaml
except ImportError:
    yaml = None

AUDIO_DIR_REL = Path("data/data/audio_en")
TTS_DIR_REL = Path("data/data/audio_en_qwen3_tts")
AUDIO_MANIFEST_REL = Path("data/data/audio_en/manifest.json")
TTS_MANIFEST_REL = Path("data/data/audio_en_qwen3_tts/manifest.json")
VISEME_MAP_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "data" / "english_viseme_map.yaml"
)

DEFAULT_MFA_DIR = OUTPUT_BASE_EN / "mfa_textgrid"
OUTPUT_MANIFEST_REL = OUTPUT_BASE_EN / "manifest" / "alignment.json"

SILENCE_LABELS = frozenset({"", "sp", "sil", "SIL", "SPN", "spn", "<sil>", "<unk>"})

MFA_MFA_ENV = "mfa"
MFA_DICT = "english_mfa"
MFA_ACOUSTIC = "english_mfa"


# ---------------------------------------------------------------------------
# IPA → Preston Blair 13 viseme mapping
# ---------------------------------------------------------------------------
# MFA's english_mfa model outputs a mix of IPA and X-SAMPA-like symbols
# (e.g. "aj" = /aɪ/, "ej" = /eɪ/, "ow" = /oʊ/, "aw" = /aʊ/). We strip
# diacritics (dental ◌̪, palatalization ʲ, length ː, aspiration ʰ) before
# lookup. See data/data/english_viseme_map.yaml for ARPABET → viseme defs.

BASE_IPA_TO_VISEME: dict[str, str] = {
    # Silence
    "": "sil",
    "sp": "sil",
    "sil": "sil",
    # /pbmv/ — lips together
    "p": "pbmv",
    "b": "pbmv",
    "m": "pbmv",
    # /fv/ — lower lip to upper teeth
    "f": "fv",
    "v": "fv",
    # /th/ — tongue between teeth
    "θ": "th",
    "ð": "th",
    # /cdsz/ — tongue tip to alveolar ridge
    "t": "cdsz",
    "d": "cdsz",
    "s": "cdsz",
    "z": "cdsz",
    "n": "cdsz",
    "l": "cdsz",
    "ɫ": "cdsz",
    "ɾ": "cdsz",
    # /kg/ — back of tongue to soft palate
    "k": "kg",
    "g": "kg",
    "ɡ": "kg",
    "ɟ": "kg",
    "c": "kg",
    "ŋ": "kg",
    "ɲ": "kg",
    "x": "kg",
    # /chjsh/ — blade to post-alveolar
    "ʃ": "chjsh",
    "ʒ": "chjsh",
    "tʃ": "chjsh",
    "dʒ": "chjsh",
    # /e/ — relaxed open mouth
    "ɛ": "e",
    "æ": "e",
    "ʌ": "e",
    "ə": "e",
    "ɐ": "e",
    "a": "e",  # also /ä/ in some conventions
    "ɜ": "e",
    # /o/ — open round (also back vowels)
    "ɑ": "o",
    "ɒ": "o",
    "ɔ": "o",
    "o": "o",
    "ʊ": "o",
    # /i/ — wide stretch
    "i": "i",
    "ɪ": "i",
    "e": "i",
    # /u/ — small round
    "u": "u",
    "ʉ": "u",
    "w": "u",
    # /r/ — slightly cupped
    "ɹ": "r",
    "ɚ": "r",
    "ɝ": "r",
    # Diphthongs — offglide viseme per spec
    "aɪ": "ai",
    "eɪ": "ai",
    "aʊ": "aw",
    "oʊ": "o",
    "ɔɪ": "o",
    # Glottal / no visible shape → neutral
    "h": "e",
    "ɦ": "e",
    "ʔ": "e",
    "ç": "e",
    # Palatal glide
    "j": "i",
    "ʎ": "i",
}

# X-SAMPA / non-IPA variants MFA sometimes outputs
XSAMPA_TO_VISEME: dict[str, str] = {
    "aj": "ai",
    "ej": "ai",
    "aw": "aw",
    "ow": "o",
    "oj": "o",
    "ɔj": "o",
    "Ij": "i",
    "əw": "o",
    "əu": "o",
}

# Syllabic consonant markers (strip before lookup)
_DIACRITIC_RE = re.compile(r"[̪̺̻̼̟̠̬̪ʰʲːˑ̩ʷˠˤ̃]")


def _normalize_phone(raw: str) -> str:
    """Strip diacritics and length marks to get the base phone symbol."""
    s = _DIACRITIC_RE.sub("", raw.strip())
    return s


def ipa_to_viseme(phone: str) -> str:
    """Map an MFA IPA phone to a Preston Blair viseme class."""
    raw = phone.strip()
    if not raw or raw in SILENCE_LABELS:
        return "sil"
    n = _normalize_phone(raw)
    # Check base IPA table first
    v = BASE_IPA_TO_VISEME.get(n)
    if v is not None:
        return v
    # Check X-SAMPA variants
    v = XSAMPA_TO_VISEME.get(n)
    if v is not None:
        return v
    # Fallback: check without stress or diacritics
    simpler = re.sub(r"[ˈˌ]", "", n)
    if simpler != n:
        v = BASE_IPA_TO_VISEME.get(simpler)
        if v is not None:
            return v
        v = XSAMPA_TO_VISEME.get(simpler)
        if v is not None:
            return v
    return "other"


# LibriSpeech ARPABET phone labels used by the legacy .phn adapter.
ARPABET_TO_VISEME: dict[str, str] = {
    "P": "pbmv", "B": "pbmv", "M": "pbmv",
    "F": "fv", "V": "fv",
    "TH": "th", "DH": "th",
    "T": "cdsz", "D": "cdsz", "S": "cdsz", "Z": "cdsz",
    "N": "cdsz", "L": "cdsz",
    "K": "kg", "G": "kg", "NG": "kg",
    "CH": "chjsh", "JH": "chjsh", "SH": "chjsh", "ZH": "chjsh",
    "HH": "e",
    "IY": "i", "IH": "i", "EH": "e", "AE": "e", "AH": "e",
    "AA": "o", "AO": "o", "OW": "o", "UH": "o",
    "UW": "u", "ER": "r", "R": "r", "W": "u", "Y": "i",
}


def arpabet_to_viseme(phone: str) -> str:
    """Map a LibriSpeech ARPABET phone to a Preston Blair viseme."""
    normalized = phone.strip().upper()
    if normalized in {"", "H#", "SIL", "SP", "SPN"}:
        return "sil"
    normalized = re.sub(r"[0-9]$", "", normalized)
    return ARPABET_TO_VISEME.get(normalized, "other")


def parse_phn_file(phn_path: Path, sample_rate: int = 16000) -> list[dict]:
    """Parse LibriSpeech sample-index phone boundaries into token spans."""
    tokens: list[dict] = []
    for line_number, raw_line in enumerate(phn_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw_line.split()
        if len(parts) != 3:
            raise ValueError(f"invalid .phn line {phn_path}:{line_number}")
        try:
            start_sample = int(parts[0])
            end_sample = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"invalid .phn boundary {phn_path}:{line_number}") from exc
        if end_sample < start_sample or sample_rate <= 0:
            raise ValueError(f"invalid .phn interval {phn_path}:{line_number}")
        start_s = start_sample / sample_rate
        end_s = end_sample / sample_rate
        tokens.append({
            "token": parts[2],
            "viseme": arpabet_to_viseme(parts[2]),
            "start_s": round(start_s, 6),
            "end_s": round(end_s, 6),
            "duration_s": round(end_s - start_s, 6),
            "confidence": 1.0,
        })
    return tokens


# MFA TextGrid format:
#   intervals [N]:
#       xmin = 0.54
#       xmax = 0.66
#       text = "w"
#
# We extract the "phones" tier and return a list of token dicts.

TEXTGRID_INTERVAL_RE = re.compile(
    r"intervals\s*\[\s*\d+\s*\]\s*:\s*"
    r"xmin\s*=\s*([\d.eE+-]+)\s*"
    r"xmax\s*=\s*([\d.eE+-]+)\s*"
    r'text\s*=\s*"([^"]*)"',
    re.DOTALL,
)


def parse_textgrid(textgrid_path: Path) -> list[dict]:
    """Parse the phones tier from an MFA TextGrid."""
    raw = textgrid_path.read_text(encoding="utf-8")

    phones_section = raw.split('name = "phones"')[1]
    phones_section = phones_section.split("name =")[0]

    tokens: list[dict] = []
    for m in TEXTGRID_INTERVAL_RE.finditer(phones_section):
        start_s = float(m.group(1))
        end_s = float(m.group(2))
        phone = m.group(3)
        viseme = ipa_to_viseme(phone)
        tokens.append({
            "token": phone,
            "viseme": viseme,
            "start_s": round(start_s, 6),
            "end_s": round(end_s, 6),
            "duration_s": round(end_s - start_s, 6),
            "confidence": 1.0,
        })
    return tokens


# ---------------------------------------------------------------------------
# MFA runner
# ---------------------------------------------------------------------------


def run_mfa_alignment(
    input_dir: Path,
    output_dir: Path,
    dictionary: str = MFA_DICT,
    acoustic: str = MFA_ACOUSTIC,
    conda_env: str = MFA_MFA_ENV,
    conda_root: str = "/root/miniconda3",
) -> None:
    """Run MFA alignment via subprocess."""
    conda_sh = Path(conda_root) / "etc" / "profile.d" / "conda.sh"
    cmd = (
        f"source {conda_sh} && "
        f"conda activate {conda_env} && "
        f"mfa align --clean --overwrite "
        f"{input_dir} {dictionary} {acoustic} {output_dir}"
    )
    result = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MFA alignment failed (exit {result.returncode}):\n"
            f"{result.stderr[:2000]}"
        )


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def _build_alignment_input(
    sample_ids: list[int], repo_root: Path, align_dir: Path,
) -> None:
    """Prepare .wav + .lab files for MFA alignment (natural only)."""
    audio_manifest = json.loads((repo_root / AUDIO_MANIFEST_REL).read_text())
    records = {int(r["sample_id"]): r for r in audio_manifest}
    align_dir.mkdir(parents=True, exist_ok=True)
    for sid in sample_ids:
        rec = records.get(sid)
        if rec is None:
            continue
        wav = repo_root / AUDIO_DIR_REL / f"{sid}.wav"
        if not wav.exists():
            continue
        dst_wav = align_dir / f"{sid}.wav"
        if not dst_wav.exists():
            dst_wav.symlink_to(wav.resolve())
        lab = align_dir / f"{sid}.lab"
        lab.write_text(rec["text"].strip() + "\n")


def _load_manifest_records(repo_root: Path) -> tuple[dict, dict]:
    """Load natural and TTS manifests; handle both list and dict formats."""
    audio_manifest_raw = json.loads(
        (repo_root / AUDIO_MANIFEST_REL).read_text()
    )
    tts_manifest_raw = json.loads(
        (repo_root / TTS_MANIFEST_REL).read_text()
    )

    if isinstance(audio_manifest_raw, list):
        audio_records = {int(r["sample_id"]): r for r in audio_manifest_raw}
    else:
        audio_records = {int(r["sample_id"]): r for r in audio_manifest_raw.get("results", [])}

    if isinstance(tts_manifest_raw, list):
        tts_records = {int(r["sample_id"]): r for r in tts_manifest_raw}
    else:
        tts_records = {int(r["sample_id"]): r for r in tts_manifest_raw.get("results", [])}

    return audio_records, tts_records


def build_english_manifest(
    sample_ids: list[int],
    repo_root: Path,
    mfa_dir: Path = DEFAULT_MFA_DIR,
    output_path: Path = OUTPUT_MANIFEST_REL,
    run_mfa: bool = False,
    conda_env: str = MFA_MFA_ENV,
    phn_dir: Path | None = None,
) -> dict:
    """Build alignment manifest from MFA TextGrids."""
    audio_records, tts_records = _load_manifest_records(repo_root)

    if run_mfa:
        align_in = output_path.parent / "_mfa_input"
        _build_alignment_input(sample_ids, repo_root, align_in)
        print(f"[28] running MFA alignment on {len(sample_ids)} files ...")
        run_mfa_alignment(align_in, mfa_dir, conda_env=conda_env)

    manifest: list[dict] = []
    failures: list[dict] = []

    for sid in sorted(sample_ids):
        librispeech_id = audio_records.get(sid, {}).get("librispeech_id", str(sid))
        phn_path = (phn_dir / f"{librispeech_id}.phn") if phn_dir is not None else None
        textgrid_path = mfa_dir / f"{sid}.TextGrid"
        if phn_path is not None and phn_path.exists():
            tokens = parse_phn_file(phn_path)
        elif textgrid_path.exists():
            tokens = parse_textgrid(textgrid_path)
        else:
            failures.append({"sample_id": sid, "error": f"alignment not found: {phn_path or textgrid_path}"})
            continue
        if not tokens:
            failures.append({"sample_id": sid, "error": "empty TextGrid"})
            continue

        duration_s = tokens[-1]["end_s"]
        rec = audio_records.get(sid, {})
        text = rec.get("text", "").strip()

        for condition, rel_dir in [("natural", AUDIO_DIR_REL), ("tts", TTS_DIR_REL)]:
            audio_path = repo_root / rel_dir / f"{sid}.wav"
            if not audio_path.exists():
                failures.append({
                    "sample_id": sid, "condition": condition,
                    "error": "audio missing",
                })
                continue

            entry: dict = {
                "sample_id": sid,
                "condition": condition,
                "variant": "raw",
                "filepath": str(audio_path.relative_to(repo_root)),
                "duration_s": duration_s,
                "text": text,
                "tokens": tokens,
            }
            if "librispeech_id" in rec:
                entry["librispeech_id"] = rec["librispeech_id"]
            manifest.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "samples_requested": sorted(sample_ids),
        "entries_written": len(manifest),
        "failures": failures,
        "manifest": manifest,
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=str,
        default=",".join(str(s) for s in ENGLISH_STUDY_SAMPLES),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--mfa-dir", type=str, default=str(DEFAULT_MFA_DIR),
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_MANIFEST_REL.parent),
    )
    parser.add_argument("--run-mfa", action="store_true")
    parser.add_argument("--mfa-env", type=str, default=MFA_MFA_ENV)
    args = parser.parse_args()

    sample_ids = [1] if args.smoke else [
        int(x) for x in args.samples.split(",") if x.strip()
    ]
    repo_root = Path(__file__).resolve().parent.parent
    mfa_dir = Path(args.mfa_dir)
    if not mfa_dir.is_absolute():
        mfa_dir = (repo_root / mfa_dir).resolve()
    output_path = Path(args.output_dir) / "alignment.json"

    summary = build_english_manifest(
        sample_ids, repo_root,
        mfa_dir=mfa_dir,
        output_path=output_path,
        run_mfa=args.run_mfa,
        conda_env=args.mfa_env,
    )

    print(f"[28] wrote {summary['entries_written']} entries; {len(summary['failures'])} failures")
    for f in summary["failures"]:
        print(f"  FAIL sample {f.get('sample_id', '?')} {f.get('condition', ''):8s} {f.get('error', '')}")
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
