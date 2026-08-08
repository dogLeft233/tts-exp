#!/usr/bin/env python3
"""Prepare stable-ID natural English pairs from the MDC manifest."""

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

from utils import resolve_repo_path

CANONICAL_SAMPLE_RATE = 16000
CANONICAL_CHANNELS = 1
CANONICAL_SUBTYPE = "PCM_16"
EXPECTED_PREFIX = "en_"
EXPECTED_SAMPLE_IDS = [f"en_{index:03d}" for index in range(1, 51)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_english_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"invalid MDC manifest: {path}")
    records = [
        record for record in payload["samples"]
        if record.get("language_code") == "en"
    ]
    records.sort(key=lambda record: str(record.get("sample_id", "")))
    return records


def load_english_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise ValueError(f"invalid MDC manifest: {path}")
    language_meta = payload.get("languages", {}).get("en", {})
    if not isinstance(language_meta, dict):
        language_meta = {}
    records = [
        record for record in payload["samples"]
        if record.get("language_code") == "en"
    ]
    records.sort(key=lambda record: str(record.get("sample_id", "")))
    return records, language_meta


def _sample_sort_key(sample_id: str) -> tuple[str, int | str]:
    prefix, _, suffix = sample_id.partition("_")
    try:
        return prefix, int(suffix)
    except ValueError:
        return prefix, suffix


def select_records(
    records: list[dict[str, Any]], sample_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    by_id = {str(record.get("sample_id")): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate English sample_id in MDC manifest")
    available = sorted(by_id, key=_sample_sort_key)
    selected_ids = sample_ids or available
    selected_ids = [str(value) for value in selected_ids]
    missing = sorted(set(selected_ids) - set(by_id))
    if missing:
        raise ValueError(f"sample IDs missing from MDC manifest: {missing}")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("duplicate sample IDs requested")
    if any(not sample_id.startswith(EXPECTED_PREFIX) for sample_id in selected_ids):
        raise ValueError("MDC English IDs must use the en_### prefix")
    return [by_id[sample_id] for sample_id in selected_ids]


def require_complete_sample_set(sample_ids: list[str]) -> None:
    expected = set(EXPECTED_SAMPLE_IDS)
    actual = set(sample_ids)
    if actual != expected or len(sample_ids) != len(EXPECTED_SAMPLE_IDS):
        missing = sorted(expected - actual, key=_sample_sort_key)
        extra = sorted(actual - expected, key=_sample_sort_key)
        raise ValueError(
            "full MDC English run must contain exactly en_001..en_050 "
            f"(missing={missing}, extra={extra})"
        )


def validate_audio(path: Path) -> dict[str, Any]:
    info = sf.info(path)
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.size == 0 or info.frames == 0:
        raise ValueError(f"empty audio: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"non-finite audio: {path}")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    if rms <= 1e-4:
        raise ValueError(f"silent audio: {path}")
    if peak > 1.0:
        raise ValueError(f"out-of-range audio: {path}")
    if int(sample_rate) != CANONICAL_SAMPLE_RATE:
        raise ValueError(f"sample-rate mismatch: {path} ({sample_rate})")
    if int(info.channels) != CANONICAL_CHANNELS or info.subtype != CANONICAL_SUBTYPE:
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
        "clipped_fraction": round(float(np.mean(np.abs(audio) >= 0.9999)), 10),
        "rms": round(rms, 8),
        "sha256": sha256_file(path),
    }


def canonicalize_audio(
    source: Path, destination: Path, ffmpeg_bin: str, overwrite: bool = False,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    marker = destination.with_name(f".{destination.name}.source.sha256")
    reusable = (
        destination.exists()
        and not overwrite
        and marker.exists()
        and marker.read_text(encoding="ascii").strip() == source_hash
    )
    if not reusable:
        temp = destination.with_name(f".{destination.name}.tmp.wav")
        if temp.exists():
            temp.unlink()
        subprocess.run(
            [
                ffmpeg_bin, "-y", "-v", "error", "-i", str(source),
                "-ar", str(CANONICAL_SAMPLE_RATE), "-ac", str(CANONICAL_CHANNELS),
                "-c:a", "pcm_s16le", str(temp),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        temp.replace(destination)
        marker.write_text(source_hash + "\n", encoding="ascii")
    return validate_audio(destination)


def prepare_natural_pairs(
    repo: Path,
    manifest_path: Path,
    run_id: str,
    sample_ids: list[str] | None = None,
    ffmpeg_bin: str = "",
    overwrite: bool = False,
) -> dict[str, Any]:
    records, language_meta = load_english_manifest(manifest_path)
    records = select_records(records, sample_ids)
    if sample_ids is None:
        require_complete_sample_set([str(record["sample_id"]) for record in records])
    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
    run_dir = repo / "runs" / run_id
    natural_dir = run_dir / "00_pairs" / "natural"
    prepared: list[dict[str, Any]] = []
    clipping_flags: list[str] = []
    for source_record in records:
        sample_key = str(source_record["sample_id"])
        text = str(source_record.get("text", "")).strip()
        if not text:
            raise ValueError(f"empty text for {sample_key}")
        source = resolve_repo_path(repo, source_record.get("file_path", ""))
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = natural_dir / f"{sample_key}.wav"
        qc = canonicalize_audio(source, destination, ffmpeg, overwrite=overwrite)
        source_author = source_record.get("source_author_id")
        archive_name = source_record.get("source_archive") or language_meta.get("archive_filename")
        dataset_id = source_record.get("source_dataset_id") or language_meta.get("dataset_id")
        if qc["clipped_fraction"] > 0:
            clipping_flags.append(sample_key)
        prepared.append(
            {
                "dataset": "mdc_tts",
                "source_dataset": source_record.get("source_dataset"),
                "source_dataset_id": dataset_id,
                "source_sample_id": sample_key,
                "sample_key": sample_key,
                "paired_key": sample_key,
                "utterance_id": f"mdc_tts/{sample_key}/natural",
                "condition": "natural",
                "variant": "raw",
                "speaker_id": f"author:{source_author}" if source_author is not None else None,
                "source_author_id": source_author,
                "source_row_index": source_record.get("source_row_index"),
                "source_archive": archive_name,
                "source_archive_member": source_record.get("source_archive_member"),
                "source_audio_id": source_record.get("source_audio_id"),
                "text": text,
                "audio_path": str(destination.relative_to(repo)),
                "source_audio_path": str(source.relative_to(repo)),
                "source_audio_sha256": sha256_file(source),
                "source_duration_s": source_record.get("source_duration_s"),
                "duration_s": qc["duration_s"],
                "canonical_qc": qc,
                "tts_status": "pending",
            }
        )

    sample_keys = [record["sample_key"] for record in prepared]
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError("duplicate prepared sample keys")
    output = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset": "mdc_tts",
        "language_code": "en",
        "source_manifest": str(manifest_path.relative_to(repo)),
        "source_manifest_sha256": sha256_file(manifest_path),
        "canonicalization": {
            "sample_rate_hz": CANONICAL_SAMPLE_RATE,
            "channels": CANONICAL_CHANNELS,
            "subtype": CANONICAL_SUBTYPE,
            "clipping_policy": "record_clipped_fraction; do not silently exclude",
        },
        "sample_keys": sample_keys,
        "sample_count": len(prepared),
        "clipped_sample_keys": clipping_flags,
        "complete_natural": sample_keys == EXPECTED_SAMPLE_IDS,
        "complete_pairs": False,
        "records": prepared,
    }
    pair_dir = run_dir / "00_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    output_path = pair_dir / "pair_manifest.json"
    temp_path = output_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(output_path)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest", default="data/mdc_tts/manifest.json")
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--ffmpeg", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    sample_ids = [value.strip() for value in args.sample_ids.split(",") if value.strip()] or None
    result = prepare_natural_pairs(
        repo=repo,
        manifest_path=resolve_repo_path(repo, args.manifest),
        run_id=args.run_id,
        sample_ids=sample_ids,
        ffmpeg_bin=args.ffmpeg,
        overwrite=args.overwrite,
    )
    print(f"[mdc-en-pairs] natural={result['sample_count']} complete_pairs={result['complete_pairs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
