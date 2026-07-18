#!/usr/bin/env python3
"""Build English Wav2Sem alignment manifest from LibriSpeech .phn files.

Reads .phn files extracted by script 27, maps ARPABET phones to Preston Blair
13 visemes (plus a 'sil' class for silence tokens), and writes a manifest
compatible with script 15 (SSL embeddings) and script 16 (separability).

Each manifest entry mirrors the Chinese manifest schema so downstream scripts
can consume both languages uniformly:

    {
      "sample_id": 1,
      "condition": "natural",        # or "tts"
      "variant": "raw",
      "filepath": "data/data/audio_en/1.wav",
      "duration_s": 5.765,
      "tokens": [
        {"token": "h#", "viseme": "sil", "start_s": 0.0, "end_s": 0.29, "confidence": 1.0},
        {"token": "k",  "viseme": "kg",  "start_s": 0.29, "end_s": 0.62, "confidence": 1.0},
        ...
      ]
    }

Usage
-----
    python scripts/28_prepare_english_alignment.py [--samples IDS] [--smoke] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg_feature_common import (
    ENGLISH_STUDY_SAMPLES,
    ENGLISH_SILENCE_LABELS,
    OUTPUT_BASE_EN,
)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

VISEME_MAP_PATH = Path(__file__).resolve().parent.parent / "data" / "data" / "english_viseme_map.yaml"
PHN_DIR = OUTPUT_BASE_EN / "manifest" / "librispeech_phn"
AUDIO_DIR_REL = Path("data/data/audio_en")
TTS_DIR_REL = Path("data/data/audio_en_qwen3_tts")
AUDIO_MANIFEST_REL = Path("data/data/audio_en/manifest.json")
TTS_MANIFEST_REL = Path("data/data/audio_en_qwen3_tts/manifest.json")

OUTPUT_MANIFEST_REL = OUTPUT_BASE_EN / "manifest" / "alignment.json"

# ---------------------------------------------------------------------------
# Viseme map
# ---------------------------------------------------------------------------


_viseme_map_cache: dict | None = None


def load_viseme_map(path: Path = VISEME_MAP_PATH) -> dict:
    global _viseme_map_cache
    if _viseme_map_cache is not None:
        return _viseme_map_cache
    if yaml is None:
        raise ImportError("PyYAML required")
    with open(path, "r", encoding="utf-8") as f:
        _viseme_map_cache = yaml.safe_load(f)
    return _viseme_map_cache


def arpabet_to_viseme(arpabet: str) -> str:
    """Map ARPABET phone to viseme; silence tokens → 'sil'; unmapped → 'other'."""
    if arpabet in ENGLISH_SILENCE_LABELS:
        return "sil"
    vm = load_viseme_map()
    mapping = vm["arpabet_to_viseme"]
    return mapping.get(arpabet, "other")


# ---------------------------------------------------------------------------
# .phn parsing (reuses 27)
# ---------------------------------------------------------------------------


def parse_phn_file(phn_path: Path, sample_rate: int = 16000) -> list[dict]:
    """Parse a LibriSpeech .phn file: <start> <end> <phone> lines.

    Imports parse_phn_file from script 27 via importlib to avoid code
    duplication (script 27's filename starts with a digit so direct import
    by name is not possible).
    """
    script27 = Path(__file__).resolve().parent / "27_download_librispeech_phn.py"
    spec = importlib.util.spec_from_file_location("librispeech_phn", script27)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_phn_file(phn_path, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------


def find_phn_for_sample(
    sample_id: int,
    audio_manifest_records: dict[int, dict],
    phn_dir: Path,
) -> Path | None:
    rec = audio_manifest_records.get(sample_id)
    if rec is None:
        return None
    uid = rec["librispeech_id"]
    phn_path = phn_dir / f"{uid}.phn"
    return phn_path if phn_path.exists() else None


def build_english_manifest(
    sample_ids: list[int],
    repo_root: Path,
    phn_dir: Path = PHN_DIR,
    output_path: Path = OUTPUT_MANIFEST_REL,
) -> dict:
    audio_manifest_path = repo_root / AUDIO_MANIFEST_REL
    tts_manifest_path = repo_root / TTS_MANIFEST_REL

    audio_records = {
        int(r["sample_id"]): r
        for r in json.loads(audio_manifest_path.read_text())
    }
    tts_records = {
        int(r["sample_id"]): r
        for r in json.loads(tts_manifest_path.read_text())
    }

    audio_dir = repo_root / AUDIO_DIR_REL
    tts_dir = repo_root / TTS_DIR_REL

    manifest: list[dict] = []
    failures: list[dict] = []
    conditions = ["natural", "tts"]

    for sid in sorted(sample_ids):
        rec = audio_records.get(sid)
        if rec is None:
            failures.append({"sample_id": sid, "error": "missing in audio manifest"})
            continue
        phn_path = find_phn_for_sample(sid, audio_records, phn_dir)
        if phn_path is None:
            failures.append({"sample_id": sid, "error": "missing .phn"})
            continue
        phn_tokens = parse_phn_file(phn_path)
        if not phn_tokens:
            failures.append({"sample_id": sid, "error": "empty .phn"})
            continue

        text = rec.get("text", "").strip()
        duration_s = float(rec.get("duration_s", phn_tokens[-1]["end_s"]))

        for condition in conditions:
            if condition == "natural":
                audio_path = audio_dir / f"{sid}.wav"
            else:
                audio_path = tts_dir / f"{sid}.wav"
            if not audio_path.exists():
                failures.append({"sample_id": sid, "condition": condition, "error": "audio missing"})
                continue

            entry: dict = {
                "sample_id": sid,
                "condition": condition,
                "variant": "raw",
                "filepath": str(audio_path.relative_to(repo_root)),
                "duration_s": duration_s,
                "text": text,
                "librispeech_id": rec["librispeech_id"],
                "tokens": [
                    {
                        "token": t["token"],
                        "viseme": arpabet_to_viseme(t["token"]),
                        "start_s": round(t["start_s"], 6),
                        "end_s": round(t["end_s"], 6),
                        "duration_s": round(t["duration_s"], 6),
                        "confidence": 1.0,
                    }
                    for t in phn_tokens
                ],
            }
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
        "--samples", type=str, default=",".join(str(s) for s in ENGLISH_STUDY_SAMPLES),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_BASE_EN / "manifest"),
    )
    args = parser.parse_args()

    sample_ids = [1] if args.smoke else [int(x) for x in args.samples.split(",") if x.strip()]
    repo_root = Path(__file__).resolve().parent.parent
    output_path = Path(args.output_dir) / "alignment.json"

    summary = build_english_manifest(sample_ids, repo_root, output_path=output_path)
    print(f"[28] wrote {summary['entries_written']} entries; {len(summary['failures'])} failures")
    for f in summary["failures"]:
        print(f"  {f}")
    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
