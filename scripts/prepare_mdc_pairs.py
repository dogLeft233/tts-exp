#!/usr/bin/env python3
"""Prepare one deterministic MDC language run for the legacy TTS pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

from utils import resolve_repo_path


LANGUAGE_CONFIG = {
    "en": {"label": "English", "qwen": "English"},
    "de": {"label": "German", "qwen": "German"},
    "it": {"label": "Italian", "qwen": "Italian"},
    "es": {"label": "Spanish", "qwen": "Spanish"},
    "ko": {"label": "Korean", "qwen": "Korean"},
    "pt": {"label": "Portuguese", "qwen": "Portuguese"},
    "ja": {"label": "Japanese", "qwen": "Japanese"},
    "fr": {"label": "French", "qwen": "French"},
    "ru": {"label": "Russian", "qwen": "Russian"},
}

CANONICAL_SAMPLE_RATE = 24000
CANONICAL_SUBTYPE = "PCM_16"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"invalid MDC manifest: {path}")
    return payload


def select_samples(
    records: list[dict[str, Any]],
    language_code: str,
    count: int = 13,
    exclude_sample_ids: set[str] | None = None,
    min_text_chars_per_s: float = 0.0,
) -> list[dict[str, Any]]:
    excluded = exclude_sample_ids or set()
    candidates = [
        (index, record)
        for index, record in enumerate(records)
        if record.get("language_code") == language_code
        and str(record.get("sample_id")) not in excluded
        and (
            min_text_chars_per_s <= 0
            or len(str(record.get("text", "")).strip())
            / max(float(record.get("duration_s", 0.0)), 1e-9)
            >= min_text_chars_per_s
        )
    ]
    if len(candidates) < count:
        raise ValueError(f"{language_code}: need {count} records, found {len(candidates)}")
    ranked = sorted(
        candidates,
        key=lambda item: (-float(item[1].get("duration_s", 0.0)), item[0]),
    )
    return [record for _, record in ranked[:count]]


def _record_source(repo: Path, record: dict[str, Any]) -> Path:
    raw_path = record.get("file_path") or record.get("local_path")
    if not raw_path:
        raise ValueError(f"record has no local audio path: {record.get('sample_id')}")
    return resolve_repo_path(repo, raw_path)


def canonicalize_audio(
    source: Path,
    destination: Path,
    ffmpeg_bin: str,
    overwrite: bool = False,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return
    command = [
        ffmpeg_bin,
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-ar",
        str(CANONICAL_SAMPLE_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=120)


def validate_audio(path: Path) -> dict[str, Any]:
    info = sf.info(path)
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.size == 0 or info.frames == 0:
        raise ValueError(f"empty audio: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"non-finite audio: {path}")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms <= 1e-4:
        raise ValueError(f"silent audio: {path}")
    if peak > 1.0:
        raise ValueError(f"out-of-range audio: {path}")
    if int(sample_rate) != CANONICAL_SAMPLE_RATE:
        raise ValueError(f"sample-rate mismatch: {path} ({sample_rate})")
    if int(info.channels) != 1 or info.subtype != CANONICAL_SUBTYPE:
        raise ValueError(
            f"format mismatch: {path} ({info.channels}ch {info.subtype})"
        )
    return {
        "sample_rate_hz": int(sample_rate),
        "channels": int(info.channels),
        "subtype": info.subtype,
        "duration_s": round(float(info.duration), 6),
        "frames": int(info.frames),
        "peak": round(peak, 8),
        "rms": round(rms, 8),
        "shorter_than_5s": bool(float(info.duration) < 5.0),
        "sha256": sha256_file(path),
    }


def build_config_override(
    language_code: str,
    run_id: str,
    sample_count: int = 13,
    sample_ids: list[int] | None = None,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    try:
        language = LANGUAGE_CONFIG[language_code]
    except KeyError as exc:
        raise ValueError(f"unsupported language: {language_code}") from exc
    ids = list(sample_ids) if sample_ids is not None else list(range(1, sample_count + 1))
    if len(ids) != sample_count:
        raise ValueError("sample_ids length must match sample_count")
    tts_config: dict[str, Any] = {
        "language": language["qwen"],
        "seed": 42,
        "retry": 1,
    }
    if max_new_tokens is not None:
        tts_config["faster_qwen3"] = {"max_new_tokens": int(max_new_tokens)}
    return {
        "paths": {
            "audio_dir": f"runs/{run_id}/00_pairs/natural",
            "tts_audio_dir": f"runs/{run_id}/02_tts",
            "image_dir": "data/data/image",
        },
        "samples": {
            "count": sample_count,
            "ids": ids,
            "smoke_id": 1,
        },
        "tts": tts_config,
        "ditto": {
            "loudnorm": False,
            "conditions": ["natural_raw", "tts_raw"],
            "retry": 1,
            "seed": 42,
        },
        "eval": {
            "require_complete_pairs": True,
        },
    }


def _resolve_ffmpeg(repo: Path, configured: str) -> str:
    if configured:
        return configured
    ditto_ffmpeg = Path("/root/autodl-tmp/envs/ditto/bin/ffmpeg")
    if ditto_ffmpeg.exists():
        return str(ditto_ffmpeg)
    return shutil.which("ffmpeg") or "ffmpeg"


def prepare_language(
    repo: Path,
    run_id: str,
    language_code: str,
    manifest_path: Path,
    image_dir: Path,
    sample_count: int = 13,
    ffmpeg_bin: str = "",
    overwrite: bool = False,
    dry_run: bool = False,
    sample_ids: list[int] | None = None,
    exclude_sample_ids: set[str] | None = None,
    source_label: str = "MDC",
    risk_label: str = "",
    min_text_chars_per_s: float = 0.0,
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    if language_code not in LANGUAGE_CONFIG:
        raise ValueError(f"unsupported language: {language_code}")
    payload = load_manifest(manifest_path)
    selected = select_samples(
        payload["samples"],
        language_code,
        sample_count,
        exclude_sample_ids=exclude_sample_ids,
        min_text_chars_per_s=min_text_chars_per_s,
    )
    target_ids = list(sample_ids) if sample_ids is not None else list(range(1, len(selected) + 1))
    if len(target_ids) != len(selected):
        raise ValueError("sample_ids length must match selected sample count")
    print(f"[{language_code}] selected {len(selected)} records:")
    for numeric_id, record in zip(target_ids, selected):
        print(
            f"  {numeric_id}: {record['sample_id']} "
            f"duration={float(record.get('duration_s', 0.0)):.3f}s"
        )
    if dry_run:
        return {
            "language_code": language_code,
            "sample_count": len(selected),
            "sample_ids": target_ids,
            "selected": selected,
        }

    run_dir = repo / "runs" / run_id
    pair_dir = run_dir / "00_pairs"
    natural_dir = pair_dir / "natural"
    transcript_dir = run_dir / "01_transcript"
    resolved_image_dir = resolve_repo_path(repo, image_dir)
    ffmpeg = _resolve_ffmpeg(repo, ffmpeg_bin)
    records: list[dict[str, Any]] = []

    for numeric_id, source_record in zip(target_ids, selected):
        source = _record_source(repo, source_record)
        image = resolved_image_dir / f"{numeric_id}.png"
        if not source.exists():
            raise FileNotFoundError(source)
        if not image.exists():
            raise FileNotFoundError(image)
        text = str(source_record.get("text", "")).strip()
        if not text:
            raise ValueError(f"empty manifest text: {source_record['sample_id']}")

        destination = natural_dir / f"{numeric_id}.wav"
        canonicalize_audio(source, destination, ffmpeg, overwrite=overwrite)
        qc = validate_audio(destination)
        transcript_path = transcript_dir / f"{numeric_id}.txt"
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(text + "\n", encoding="utf-8")

        records.append(
            {
                "sample_id": numeric_id,
                "source_sample_id": source_record["sample_id"],
                "language_code": language_code,
                "language": LANGUAGE_CONFIG[language_code]["label"],
                "qwen_language": LANGUAGE_CONFIG[language_code]["qwen"],
                "text": text,
                "source_audio": str(source.relative_to(repo)),
                "canonical_audio": str(destination.relative_to(repo)),
                "source_audio_sha256": sha256_file(source),
                "image": str(image.relative_to(repo)),
                "image_sha256": sha256_file(image),
                "manifest_duration_s": float(source_record.get("duration_s", 0.0)),
                "source_author_id": source_record.get("source_author_id"),
                "source_dataset": source_record.get("source_dataset"),
                "canonical_qc": qc,
            }
        )

    transcript_payload = {
        "source": "data/mdc_tts/manifest.json",
        "language_code": language_code,
        "results": {
            str(record["sample_id"]): {
                "sample_id": record["sample_id"],
                "text": record["text"],
            }
            for record in records
        },
    }
    transcript_dir.mkdir(parents=True, exist_ok=True)
    (transcript_dir / "transcript.json").write_text(
        json.dumps(transcript_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    short_ids = [
        record["sample_id"]
        for record in records
        if record["canonical_qc"]["shorter_than_5s"]
    ]
    language_label = LANGUAGE_CONFIG[language_code]["label"]
    metadata = {
        "run_id": run_id,
        "experiment_name": "mdc_multilingual_tts_tfg",
        "design": "self_clone",
        "language_code": language_code,
        "language": language_label,
        "qwen_language": LANGUAGE_CONFIG[language_code]["qwen"],
        "sample_description": (
            f"{source_label} {language_label}, {len(records)} paired samples, "
            "longest-duration selection"
        ),
        "source_label": source_label,
        "risk_label": risk_label or None,
        "sample_count": len(records),
        "sample_ids": target_ids,
        "excluded_source_sample_ids": sorted(exclude_sample_ids or set()),
        "min_text_chars_per_s": min_text_chars_per_s,
        "shorter_than_5s_ids": short_ids,
        "canonicalization": {
            "sample_rate_hz": CANONICAL_SAMPLE_RATE,
            "channels": 1,
            "subtype": CANONICAL_SUBTYPE,
            "loudness_normalization": False,
        },
        "manifest": {
            "path": str(manifest_path.relative_to(repo)),
            "sha256": sha256_file(manifest_path),
        },
        "records": records,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "pair_manifest.json").write_text(
        json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "config_override.yaml").write_text(
        yaml.safe_dump(
            build_config_override(
                language_code,
                run_id,
                sample_count=len(records),
                sample_ids=target_ids,
                max_new_tokens=max_new_tokens,
            ),
            sort_keys=False,
            allow_unicode=False,
        ),
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--language", required=True, choices=sorted(LANGUAGE_CONFIG))
    parser.add_argument("--manifest", default="data/mdc_tts/manifest.json")
    parser.add_argument("--sample-count", type=int, default=13)
    parser.add_argument("--image-dir", default="data/data/image")
    parser.add_argument("--ffmpeg", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--target-ids",
        default="",
        help="Comma-separated output/image IDs; useful for replacement samples.",
    )
    parser.add_argument(
        "--exclude-sample-ids",
        default="",
        help="Comma-separated source manifest IDs to exclude before selection.",
    )
    parser.add_argument("--source-label", default="MDC")
    parser.add_argument("--risk-label", default="")
    parser.add_argument("--min-text-chars-per-s", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()
    if args.sample_count < 1:
        raise SystemExit("--sample-count must be positive")
    target_ids = [int(value) for value in args.target_ids.split(",") if value.strip()]
    if target_ids and len(target_ids) != args.sample_count:
        raise SystemExit("--target-ids length must match --sample-count")
    exclude_sample_ids = {
        value.strip() for value in args.exclude_sample_ids.split(",") if value.strip()
    }

    repo = Path(__file__).resolve().parent.parent
    prepare_language(
        repo=repo,
        run_id=args.run_id,
        language_code=args.language,
        manifest_path=resolve_repo_path(repo, args.manifest),
        image_dir=Path(args.image_dir),
        sample_count=args.sample_count,
        ffmpeg_bin=args.ffmpeg,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        sample_ids=target_ids or None,
        exclude_sample_ids=exclude_sample_ids or None,
        source_label=args.source_label,
        risk_label=args.risk_label,
        min_text_chars_per_s=args.min_text_chars_per_s,
        max_new_tokens=args.max_new_tokens,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
