#!/usr/bin/env python3
"""Acquire small, reproducible multilingual TTS audio subsets.

The Hugging Face datasets-server endpoint exposes signed, short-lived audio
URLs. This script stores only stable dataset/row metadata and local checksums,
never the signed URLs themselves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - optional fallback
    sf = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "multilingual_tts"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
FIRST_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/first-rows"

QWEN_LANGUAGES = [
    "Chinese",
    "English",
    "German",
    "Italian",
    "Portuguese",
    "Spanish",
    "Japanese",
    "Korean",
    "French",
    "Russian",
]

# The selected sources are TTS-oriented corpora, not LibriSpeech, MLS,
# Common Voice, FLEURS, or VoxPopuli. Proprietary model-training overlap is
# not knowable from public dataset cards and is recorded as an open risk.
SOURCES: dict[str, dict[str, Any]] = {
    "zh": {
        "language": "Chinese",
        "qwen_language": "Chinese",
        "dataset": "davidguzmanr/CSS10-Multilingual-LJSpeech",
        "config": "Chinese",
        "split": "test",
        "license": "Apache-2.0 (derived repository)",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "dataset_url": "https://huggingface.co/datasets/davidguzmanr/CSS10-Multilingual-LJSpeech",
        "min_duration_s": 5.0,
        "max_duration_s": 10.0,
        "source_family": "CSS10 TTS corpus",
    },
    "de": {
        "language": "German",
        "qwen_language": "German",
        "dataset": "ylacombe/cml-tts",
        "config": "german",
        "split": "test",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "dataset_url": "https://huggingface.co/datasets/ylacombe/cml-tts",
        "min_duration_s": 5.0,
        "max_duration_s": 20.0,
        "source_family": "CML-TTS public-domain audiobook corpus",
    },
    "it": {
        "language": "Italian",
        "qwen_language": "Italian",
        "dataset": "ylacombe/cml-tts",
        "config": "italian",
        "split": "test",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "dataset_url": "https://huggingface.co/datasets/ylacombe/cml-tts",
        "min_duration_s": 5.0,
        "max_duration_s": 20.0,
        "source_family": "CML-TTS public-domain audiobook corpus",
    },
    "pt": {
        "language": "Portuguese",
        "qwen_language": "Portuguese",
        "dataset": "ylacombe/cml-tts",
        "config": "portuguese",
        "split": "test",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "dataset_url": "https://huggingface.co/datasets/ylacombe/cml-tts",
        "min_duration_s": 5.0,
        "max_duration_s": 20.0,
        "source_family": "CML-TTS public-domain audiobook corpus",
    },
    "es": {
        "language": "Spanish",
        "qwen_language": "Spanish",
        "dataset": "ylacombe/cml-tts",
        "config": "spanish",
        "split": "test",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "dataset_url": "https://huggingface.co/datasets/ylacombe/cml-tts",
        "min_duration_s": 5.0,
        "max_duration_s": 20.0,
        "source_family": "CML-TTS public-domain audiobook corpus",
    },
    "ja": {
        "language": "Japanese",
        "qwen_language": "Japanese",
        "dataset": "davidguzmanr/CSS10-Multilingual-LJSpeech",
        "config": "Japanese",
        "split": "test",
        "license": "Apache-2.0 (derived repository)",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "dataset_url": "https://huggingface.co/datasets/davidguzmanr/CSS10-Multilingual-LJSpeech",
        "min_duration_s": 5.0,
        "max_duration_s": 10.0,
        "source_family": "CSS10 TTS corpus",
    },
    "ko": {
        "language": "Korean",
        "qwen_language": "Korean",
        "dataset": "Bingsu/KSS_Dataset",
        "config": "default",
        "split": "train",
        "license": "CC-BY-NC-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "dataset_url": "https://huggingface.co/datasets/Bingsu/KSS_Dataset",
        # KSS metadata rounds some files to 5.0 s; use a margin so decoded
        # duration still satisfies the project's strict >=5 s gate.
        "min_duration_s": 5.05,
        "max_duration_s": 12.0,
        "source_family": "KSS single-speaker TTS corpus",
    },
    "fr": {
        "language": "French",
        "qwen_language": "French",
        "dataset": "ylacombe/cml-tts",
        "config": "french",
        "split": "test",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "dataset_url": "https://huggingface.co/datasets/ylacombe/cml-tts",
        "min_duration_s": 5.0,
        "max_duration_s": 20.0,
        "source_family": "CML-TTS public-domain audiobook corpus",
    },
    "ru": {
        "language": "Russian",
        "qwen_language": "Russian",
        "dataset": "davidguzmanr/CSS10-Multilingual-LJSpeech",
        "config": "Russian",
        "split": "test",
        "license": "Apache-2.0 (derived repository)",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "dataset_url": "https://huggingface.co/datasets/davidguzmanr/CSS10-Multilingual-LJSpeech",
        "min_duration_s": 5.0,
        "max_duration_s": 10.0,
        "source_family": "CSS10 TTS corpus",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_rows_url(source: dict[str, Any], offset: int, length: int) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": source["dataset"],
            "config": source["config"],
            "split": source["split"],
            "offset": offset,
            "length": length,
        }
    )
    return f"{ROWS_ENDPOINT}?{query}"


def first_rows_url(source: dict[str, Any]) -> str:
    query = urllib.parse.urlencode(
        {
            "dataset": source["dataset"],
            "config": source["config"],
            "split": source["split"],
        }
    )
    return f"{FIRST_ROWS_ENDPOINT}?{query}"


def request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "tts-exp-multilingual-acquisition/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except Exception as exc:  # pragma: no cover - network-dependent
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f"Dataset API request failed: {url}") from last_error


def fetch_rows(source: dict[str, Any], offset: int, length: int) -> list[dict[str, Any]]:
    try:
        payload = request_json(stable_rows_url(source, offset, length))
    except RuntimeError:
        # Some converted Parquet shards intermittently fail through /rows at
        # offset zero while the service's /first-rows endpoint remains usable.
        if offset != 0:
            raise
        payload = request_json(first_rows_url(source))
    return [entry["row"] for entry in payload.get("rows", [])]


def row_audio_url(row: dict[str, Any]) -> str:
    audio = row.get("audio")
    if isinstance(audio, list) and audio:
        audio = audio[0]
    if isinstance(audio, dict) and isinstance(audio.get("src"), str):
        return audio["src"]
    raise ValueError("Dataset row has no downloadable audio URL")


def row_text(row: dict[str, Any]) -> str:
    for key in ("text", "original_script", "expanded_script"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("Dataset row has no text transcript")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wav_metadata(path: Path) -> dict[str, Any]:
    if sf is not None:
        info = sf.info(str(path))
        if info.channels < 1 or info.samplerate < 1 or info.frames < 1:
            raise ValueError(f"Invalid or empty audio file: {path}")
        return {
            "duration_s": info.frames / info.samplerate,
            "sample_rate_hz": info.samplerate,
            "channels": info.channels,
            "subtype": info.subtype,
            "file_size_bytes": path.stat().st_size,
        }
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        sample_width = handle.getsampwidth()
    if channels < 1 or sample_rate < 1 or frames < 1:
        raise ValueError(f"Invalid or empty WAV file: {path}")
    return {
        "duration_s": frames / sample_rate,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "file_size_bytes": path.stat().st_size,
    }


def download_audio(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=".part",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "tts-exp-multilingual-acquisition/1.0"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temporary.write(chunk)
            temporary.flush()
            os.replace(temporary_path, destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise


def select_rows(
    source: dict[str, Any], sample_count: int, max_rows: int
) -> list[tuple[int, dict[str, Any]]]:
    selected: list[tuple[int, dict[str, Any]]] = []
    offset = 0
    while offset < max_rows and len(selected) < sample_count:
        batch_size = min(100, max_rows - offset)
        rows = fetch_rows(source, offset, batch_size)
        if not rows:
            break
        for index, row in enumerate(rows):
            advertised_duration = row.get("duration")
            if not isinstance(advertised_duration, (float, int)):
                continue
            if not (
                source["min_duration_s"]
                <= float(advertised_duration)
                <= source["max_duration_s"]
            ):
                continue
            try:
                row_text(row)
                row_audio_url(row)
            except ValueError:
                continue
            row_index = int(row.get("row_idx", offset + index))
            selected.append((row_index, row))
            if len(selected) >= sample_count:
                break
        offset += len(rows)
        if len(rows) < batch_size:
            break
    if len(selected) < sample_count:
        raise RuntimeError(
            f"Only found {len(selected)} usable rows for {source['language']}; "
            f"needed {sample_count} within {source['min_duration_s']}-"
            f"{source['max_duration_s']} seconds"
        )
    return selected


def build_sample(
    language_code: str,
    source: dict[str, Any],
    ordinal: int,
    row_index: int,
    row: dict[str, Any],
    audio_dir: Path,
    force: bool,
) -> dict[str, Any]:
    sample_id = f"{language_code}_{ordinal:03d}"
    destination = audio_dir / language_code / f"{sample_id}.wav"
    if not destination.exists() or force:
        download_audio(row_audio_url(row), destination)
    metadata = wav_metadata(destination)
    if metadata["duration_s"] < 5.0:
        raise ValueError(
            f"Decoded audio is shorter than 5 seconds: {destination} "
            f"({metadata['duration_s']:.4f}s)"
        )
    return {
        "sample_id": sample_id,
        "language": source["language"],
        "language_code": language_code,
        "qwen_language": source["qwen_language"],
        "ditto_language_validation": "pending_empirical_test",
        "audio_role": "multilingual_tts_reference",
        "local_path": str(destination.relative_to(ROOT)),
        "text": row_text(row),
        "advertised_duration_s": float(row["duration"]),
        "source_row_index": row_index,
        "source_dataset": source["dataset"],
        "source_config": source["config"],
        "source_split": source["split"],
        "source_url": source["dataset_url"],
        "source_rows_url": stable_rows_url(source, row_index, 1),
        "license": source["license"],
        "license_url": source["license_url"],
        "source_family": source["source_family"],
        "audio_encoder_overlap_risk": (
            "Named ASR/SSL benchmark avoided; proprietary Qwen/encoder overlap "
            "cannot be established from public information"
        ),
        "sha256": sha256_file(destination),
        **metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--languages",
        default=",".join(SOURCES),
        help="Comma-separated language codes (default: all nine public sources)",
    )
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query and report selected rows without downloading audio",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sample_count < 1:
        raise SystemExit("--sample-count must be positive")
    language_codes = [code.strip() for code in args.languages.split(",") if code.strip()]
    unknown = sorted(set(language_codes) - set(SOURCES))
    if unknown:
        raise SystemExit(f"Unknown language code(s): {', '.join(unknown)}")

    selected_by_language: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for code in language_codes:
        source = SOURCES[code]
        selected = select_rows(source, args.sample_count, args.max_rows)
        selected_by_language[code] = selected
        print(
            f"{code}: selected {len(selected)} rows from "
            f"{source['dataset']}[{source['config']}/{source['split']}]"
        )

    if args.dry_run:
        return 0

    output_dir = args.output_dir
    audio_dir = output_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    for code in language_codes:
        source = SOURCES[code]
        for ordinal, (row_index, row) in enumerate(selected_by_language[code], start=1):
            sample = build_sample(
                code,
                source,
                ordinal,
                row_index,
                row,
                audio_dir,
                args.force,
            )
            samples.append(sample)
            print(f"  {sample['sample_id']}: {sample['duration_s']:.2f}s")

    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "sample_count_per_public_language": args.sample_count,
        "public_languages": [SOURCES[code]["language"] for code in language_codes],
        "qwen3_tts_official_languages": QWEN_LANGUAGES,
        "english_status": "pending_hdtf_nas_download",
        "ditto_status": (
            "Ditto has no official language list in the project documentation; "
            "validate each language empirically before claiming support"
        ),
        "selection_policy": {
            "minimum_duration_s": 5.0,
            "text_required": True,
            "source_policy": (
                "Prefer TTS-specific corpora and exclude named LibriSpeech, MLS, "
                "Common Voice, FLEURS, and VoxPopuli sources"
            ),
        },
        "sources": [
            {
                "language_code": code,
                "language": SOURCES[code]["language"],
                "dataset": SOURCES[code]["dataset"],
                "config": SOURCES[code]["config"],
                "split": SOURCES[code]["split"],
                "dataset_url": SOURCES[code]["dataset_url"],
                "license": SOURCES[code]["license"],
                "license_url": SOURCES[code]["license_url"],
                "duration_filter_s": [
                    SOURCES[code]["min_duration_s"],
                    SOURCES[code]["max_duration_s"],
                ],
            }
            for code in language_codes
        ],
        "samples": samples,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(samples)} samples to {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("Interrupted")
