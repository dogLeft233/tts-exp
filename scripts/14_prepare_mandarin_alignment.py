#!/usr/bin/env python3
"""Mandarin forced alignment and viseme mapping for Wav2Sem-style analysis.

Usage
-----
    python scripts/14_prepare_mandarin_alignment.py [--check-only]
        [--audio-dir DIR] [--transcript-dir DIR] [--output-dir DIR]
        [--sample-ids IDS] [--mfa-model MODEL]

Dependencies
------------
    pip install pypinyin PyYAML
    pip install montreal-forced-aligner  # optional, for MFA alignment
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared module
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg_feature_common import STUDY_SAMPLES, OUTPUT_BASE
from manifest import MANIFEST_SCHEMA_VERSION, write_manifest

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
# pip install pypinyin
try:
    from pypinyin import pinyin as _pypinyin_fn, Style as _pypinyin_Style
    pinyin = _pypinyin_fn
    Style = _pypinyin_Style
except ImportError:
    pinyin = None  # type: ignore[assignment]
    Style = None  # type: ignore[assignment]

# pip install PyYAML
try:
    import yaml
except ImportError:
    yaml = None

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VISEME_MAP_PATH = PROJECT_ROOT / "data" / "wav2sem_analysis" / "mandarin_viseme_map.yaml"

# Ordered longest-first so we match "zh" before "z", "ch" before "c", etc.
_PINYIN_INITIALS: tuple[str, ...] = (
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x",
    "r", "z", "c", "s",
    "y", "w",
)

# Initials whose "i" final is an apical/retroflex vowel (written "-i").
_APICAL_VOWEL_INITIALS: frozenset[str] = frozenset({"zh", "ch", "sh", "r", "z", "c", "s"})

# TTS run paths keyed by engine name.
_TTS_RUNS: dict[str, Path] = {
    "faster_qwen3": PROJECT_ROOT / "runs" / "r2_faster_qwen3_20260707T145233Z" / "02_tts",
    "f5_tts": PROJECT_ROOT / "runs" / "r5_f5_tts" / "02_tts",
}

# ---------------------------------------------------------------------------
# Pinyin helpers
# ---------------------------------------------------------------------------


def _split_pinyin_syllable(raw: str) -> tuple[str, str, int]:
    """Split a single pinyin syllable into (initial, final, tone).

    Examples
    --------
    >>> _split_pinyin_syllable("ni3")
    ('n', 'i', 3)
    >>> _split_pinyin_syllable("shi4")
    ('sh', '-i', 4)
    >>> _split_pinyin_syllable("a1")
    ('∅', 'a', 1)
    """
    tone = 5
    base = raw.strip()
    if not base:
        return ("∅", "", tone)

    if base[-1].isdigit():
        tone = int(base[-1])
        base = base[:-1]

    if not base:
        return ("∅", "", tone)

    initial = "∅"
    final = base
    for init in _PINYIN_INITIALS:
        if base.startswith(init):
            initial = init
            final = base[len(init):]
            break

    if final == "i" and initial in _APICAL_VOWEL_INITIALS:
        final = "-i"

    return (initial, final, tone)


def _clean_text(text: str) -> str:
    """Strip punctuation and whitespace, keeping only Chinese characters."""
    return re.sub(r"[^\u4e00-\u9fff]", "", text)


def text_to_pinyin_tokens(text: str) -> list[tuple[str, str, str, int]]:
    """Convert Chinese *text* to a list of (character, initial, final, tone).

    Returns empty list if pypinyin is not installed.
    """
    if pinyin is None:
        return []

    cleaned = _clean_text(text)
    if not cleaned:
        return []

    pinyin_list = pinyin(cleaned, style=Style.TONE3)  # type: ignore[union-attr]
    if not pinyin_list:
        return []

    result: list[tuple[str, str, str, int]] = []
    for i, readings in enumerate(pinyin_list):
        char = cleaned[i] if i < len(cleaned) else ""
        raw = readings[0] if readings else ""
        if not raw:
            continue
        initial, final, tone = _split_pinyin_syllable(raw)
        result.append((char, initial, final, tone))

    return result


# ---------------------------------------------------------------------------
# Viseme mapping
# ---------------------------------------------------------------------------

_viseme_map_cache: dict | None = None  # type: ignore[type-arg]


def load_viseme_map() -> dict:  # type: ignore[type-arg]
    """Load and cache the viseme map from YAML."""
    global _viseme_map_cache
    if _viseme_map_cache is not None:
        return _viseme_map_cache
    if yaml is None:
        raise ImportError("PyYAML is required; install with: pip install PyYAML")
    with open(VISEME_MAP_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for key in ("initial_to_viseme", "final_to_viseme"):
        if key not in raw:
            raise ValueError(
                f"Viseme map file {VISEME_MAP_PATH} is missing required key: {key!r}"
            )
    _viseme_map_cache = raw
    return _viseme_map_cache  # type: ignore[return-value]


def pinyin_to_viseme(initial: str, final: str) -> str:
    """Return the viseme label for a pinyin (initial, final) pair.

    Prefers the initial's viseme for consonant initials; falls back to
    the final's viseme for zero-initial syllables.
    """
    vm = load_viseme_map()
    init_map: dict[str, str] = vm["initial_to_viseme"]
    final_map: dict[str, str] = vm["final_to_viseme"]

    if initial != "∅":
        return init_map.get(initial, "nonlabial_consonant")
    return final_map.get(final, "silence")


# ---------------------------------------------------------------------------
# CTM parsing
# ---------------------------------------------------------------------------

_CTM_LINE_RE = re.compile(
    r"(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)"
)


def parse_ctm_file(
    ctm_path: Path,
    min_duration_s: float = 0.04,
    min_confidence: float = 1.0,
) -> tuple[list[dict], dict]:
    """Parse a CTM alignment file.

    Returns (tokens, stats) where *tokens* is a list of per-token dicts
    and *stats* is a dict with ``total``, ``short_dropped``,
    ``confidence_dropped``, ``kept``.
    """
    tokens: list[dict] = []
    stats: dict[str, int] = {"total": 0, "short_dropped": 0, "confidence_dropped": 0, "kept": 0}

    if not ctm_path.exists():
        logger.warning("CTM file not found: %s", ctm_path)
        return tokens, stats

    with open(ctm_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _CTM_LINE_RE.match(line)
            if m is None:
                continue
            utterance, channel_str, start_str, dur_str, token_str = m.groups()
            channel = int(channel_str)
            start_s = float(start_str)
            duration_s = float(dur_str)

            stats["total"] += 1

            if duration_s < min_duration_s:
                stats["short_dropped"] += 1
                continue

            confidence = 1.0
            parts = token_str.split("_")
            token_label = parts[0]
            if len(parts) >= 2:
                try:
                    confidence = float(parts[-1])
                except ValueError:
                    pass

            if confidence < min_confidence:
                stats["confidence_dropped"] += 1
                continue

            tokens.append({
                "token": token_label,
                "channel": channel,
                "start_s": start_s,
                "end_s": start_s + duration_s,
                "confidence": confidence,
            })
            stats["kept"] += 1

    return tokens, stats


# ---------------------------------------------------------------------------
# MFA integration
# ---------------------------------------------------------------------------

def _mfa_available() -> bool:
    return shutil.which("mfa") is not None


def run_mfa_alignment(
    audio_path: Path,
    transcript_path: Path,
    output_dir: Path,
    mfa_model: str = "mandarin_mfa",
    num_jobs: int = 4,
) -> Path:
    """Run MFA forced alignment on a single audio file.

    Returns the path to the expected CTM output file, or raises
    ``subprocess.CalledProcessError`` on failure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mfa", "align",
        str(audio_path.parent),
        str(transcript_path.parent),
        str(mfa_model),
        str(output_dir),
        "--single_speaker",
        "--clean",
        "--num_jobs", str(num_jobs),
    ]
    logger.info("Running MFA: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=False)

    base = audio_path.stem
    ctm_path = output_dir / f"{base}.ctm"
    return ctm_path


# ---------------------------------------------------------------------------
# Source audio discovery
# ---------------------------------------------------------------------------


def _discover_audio(
    sample_id: int, condition: str, audio_dir: Path
) -> Path | None:
    """Return the audio file path for a given (sample_id, condition)."""
    if condition == "natural":
        path = audio_dir / f"{sample_id}.wav"
        return path if path.exists() else None
    run_dir = _TTS_RUNS.get(condition)
    if run_dir is None:
        logger.debug("Unknown TTS condition: %s", condition)
        return None
    path = run_dir / f"{sample_id}.wav"
    return path if path.exists() else None


# ---------------------------------------------------------------------------
# Main manifest builder
# ---------------------------------------------------------------------------


def _load_transcript(transcript_dir: Path, sample_id: int) -> str | None:
    """Load the Chinese transcript for a sample."""
    for name in (f"{sample_id}.txt", f"{sample_id:04d}.txt"):
        path = transcript_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    logger.warning("No transcript found for sample %d in %s", sample_id, transcript_dir)
    return None


def _get_audio_duration(path: Path) -> float:
    """Get audio duration in seconds using ffprobe or soundfile."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, timeout=15,
        )
        return float(result.stdout.strip())
    except Exception:
        try:
            import soundfile as sf
            info = sf.info(str(path))
            return info.duration
        except Exception:
            return 0.0


def build_manifest(
    sample_ids: list[int],
    audio_dir: Path,
    transcript_dir: Path,
    output_dir: Path,
    mfa_model: str = "mandarin_mfa",
    mfa_num_jobs: int = 4,
    check_only: bool = False,
) -> list[dict]:
    """Build the full alignment manifest.

    Parameters
    ----------
    sample_ids : list[int]
        Sample identifiers to process.
    audio_dir : Path
        Directory containing natural audio (data/data/audio/).
    transcript_dir : Path
        Directory containing .txt transcripts.
    output_dir : Path
        Output directory for MFA alignments and manifest.
    mfa_model : str
        MFA acoustic model name or path.
    mfa_num_jobs : int
        Number of parallel MFA jobs.
    check_only : bool
        If True, only validate dependencies and print status.

    Returns
    -------
    manifest : list[dict]
        JSON-serialisable manifest entries.
    """
    mfa_ok = _mfa_available()
    pinyin_ok = pinyin is not None
    yaml_ok = yaml is not None

    logger.info("=== Dependency check ===")
    logger.info("  MFA available: %s", "yes" if mfa_ok else "no")
    logger.info("  pypinyin available: %s", "yes" if pinyin_ok else "no (pip install pypinyin)")
    logger.info("  PyYAML available: %s", "yes" if yaml_ok else "no (pip install PyYAML)")

    if check_only:
        return []

    if not yaml_ok:
        logger.error("PyYAML is required.  Install with: pip install PyYAML")
        return []

    conditions = ["natural", "faster_qwen3", "f5_tts"]
    variants = ["raw", "gain_matched"]

    manifest: list[dict] = []
    total_filtered_short = 0
    total_filtered_confidence = 0
    total_tokens_processed = 0
    total_tokens_kept = 0

    for sample_id in sorted(sample_ids):
        transcript_text = _load_transcript(transcript_dir, sample_id)
        pinyin_tokens = text_to_pinyin_tokens(transcript_text) if transcript_text and pinyin_ok else []

        for condition in conditions:
            audio_path = _discover_audio(sample_id, condition, audio_dir)
            if audio_path is None:
                logger.debug(
                    "Skipping sample %d condition %s: no audio found", sample_id, condition
                )
                continue

            duration_s = _get_audio_duration(audio_path)
            alignment_dir = output_dir / "alignments" / f"{sample_id}_{condition}"

            ctm_path: Path | None = None
            if mfa_ok and transcript_text:
                transcript_file = alignment_dir / f"{sample_id}.txt"
                transcript_file.parent.mkdir(parents=True, exist_ok=True)
                transcript_file.write_text(transcript_text, encoding="utf-8")
                try:
                    ctm_path = run_mfa_alignment(
                        audio_path, transcript_file, alignment_dir,
                        mfa_model=mfa_model, num_jobs=mfa_num_jobs,
                    )
                except subprocess.CalledProcessError as exc:
                    logger.warning("MFA alignment failed for sample %d: %s", sample_id, exc)
                except FileNotFoundError:
                    logger.warning("MFA binary not found; skipping alignment for sample %d", sample_id)
            elif not mfa_ok:
                logger.debug(
                    "MFA not available; using uniform segmentation for sample %d", sample_id,
                )

            for variant in variants:
                entry: dict = {
                    "schema_version": MANIFEST_SCHEMA_VERSION,
                    "dataset": "aishell1",
                    "utterance_id": f"{sample_id}:{condition}:{variant}",
                    "speaker_id": f"aishell1_{sample_id}",
                    "paired_key": str(sample_id),
                    "sample_id": sample_id,
                    "condition": "natural" if condition == "natural" else "tts",
                    "tts_provider": None if condition == "natural" else condition,
                    "variant": variant,
                    "transcript": transcript_text,
                    "filepath": str(audio_path),
                    "audio_path": str(audio_path),
                    "duration_s": duration_s,
                    "sample_rate": 16000,
                    "split": "legacy",
                    "license": "Apache-2.0",
                    "alignment_source": "missing",
                    "alignment_confidence": None,
                    "tokens": [],
                }

                if not mfa_ok and pinyin_tokens and duration_s > 0:
                    entry["alignment_source"] = "uniform_fallback"
                    entry["alignment_confidence"] = 0.0
                    n = len(pinyin_tokens)
                    segment_s = duration_s / n
                    for i, (char, init, fin, tone) in enumerate(pinyin_tokens):
                        start = round(i * segment_s, 6)
                        end = round((i + 1) * segment_s, 6)
                        viseme = pinyin_to_viseme(init, fin)
                        entry["tokens"].append({
                            "token": char,
                            "initial": init,
                            "final": fin,
                            "tone": tone,
                            "viseme": viseme,
                            "start_s": start,
                            "end_s": end,
                            "confidence": 1.0,
                        })

                if ctm_path is not None and ctm_path.exists() and pinyin_tokens:
                    ctm_tokens, stats = parse_ctm_file(ctm_path)
                    total_tokens_processed += stats["total"]
                    total_filtered_short += stats["short_dropped"]
                    total_filtered_confidence += stats["confidence_dropped"]
                    total_tokens_kept += stats["kept"]

                    token_count = min(len(ctm_tokens), len(pinyin_tokens))
                    entry["alignment_source"] = "mfa"
                    entry["alignment_confidence"] = (
                        sum(t["confidence"] for t in ctm_tokens) / len(ctm_tokens)
                        if ctm_tokens else None
                    )
                    for i in range(token_count):
                        ct = ctm_tokens[i]
                        char, init, fin, tone = pinyin_tokens[i]
                        viseme = pinyin_to_viseme(init, fin)
                        entry["tokens"].append({
                            "token": char,
                            "initial": init,
                            "final": fin,
                            "tone": tone,
                            "viseme": viseme,
                            "start_s": round(ct["start_s"], 6),
                            "end_s": round(ct["end_s"], 6),
                            "confidence": round(ct["confidence"], 4),
                        })

                manifest.append(entry)

    logger.info("=== Filtering statistics ===")
    logger.info("  Total CTM tokens processed: %d", total_tokens_processed)
    logger.info("  Dropped (duration < 40 ms): %d", total_filtered_short)
    logger.info("  Dropped (low confidence):  %d", total_filtered_confidence)
    logger.info("  Kept:                        %d", total_tokens_kept)
    logger.info("  Manifest entries:            %d", len(manifest))

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mandarin forced alignment and viseme mapping for Wav2Sem analysis.",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Validate dependencies and exit without processing.",
    )
    parser.add_argument(
        "--audio-dir", type=str,
        default=str(PROJECT_ROOT / "data" / "data" / "audio"),
        help="Directory containing natural audio files (default: data/data/audio/).",
    )
    parser.add_argument(
        "--transcript-dir", type=str,
        default=str(PROJECT_ROOT / "data" / "data" / "transcript"),
        help="Directory containing .txt transcripts (default: data/data/transcript/).",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(OUTPUT_BASE),
        help="Root output directory (default: data/wav2sem_analysis/).",
    )
    parser.add_argument(
        "--sample-ids", type=str, default=None,
        help="Comma-separated sample IDs (default: STUDY_SAMPLES).",
    )
    parser.add_argument(
        "--mfa-model", type=str, default="mandarin_mfa",
        help="MFA acoustic model name or path (default: mandarin_mfa).",
    )
    parser.add_argument(
        "--mfa-num-jobs", type=int, default=4,
        help="Number of parallel MFA jobs (default: 4).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    audio_dir = Path(args.audio_dir)
    transcript_dir = Path(args.transcript_dir)
    output_dir = Path(args.output_dir)

    if args.sample_ids is not None:
        sample_ids = []
        for x in args.sample_ids.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                sample_ids.append(int(x))
            except ValueError:
                print(f"Invalid sample ID: {x}", file=sys.stderr)
                continue
    else:
        sample_ids = STUDY_SAMPLES

    try:
        manifest = build_manifest(
            sample_ids=sample_ids,
            audio_dir=audio_dir,
            transcript_dir=transcript_dir,
            output_dir=output_dir,
            mfa_model=args.mfa_model,
            mfa_num_jobs=args.mfa_num_jobs,
            check_only=args.check_only,
        )

        if args.check_only:
            logger.info("Check-only mode: no manifest written.")
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "alignments").mkdir(exist_ok=True)
        manifest_path = output_dir / "manifest" / "alignment.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        write_manifest(manifest_path, manifest)

        logger.info("Wrote %d entries to %s", len(manifest), manifest_path)
        return 0
    except (IOError, OSError) as exc:
        print(f"Error writing output: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
