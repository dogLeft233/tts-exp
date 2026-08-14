#!/usr/bin/env python3
"""Canonicalize the frozen strict AISHELL-1 TTS sources for MFA-linear."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(cohort_path: Path, outdir: Path, ffmpeg: str = "") -> dict[str, Any]:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    records = list(cohort.get("records", []))
    if cohort.get("manifest_type") != "aishell1_mfa_linear_predefined_cohort" or len(records) != 25:
        raise ValueError("expected frozen AISHELL-1 n=25 cohort")
    ffmpeg_bin = ffmpeg or shutil.which("ffmpeg") or "ffmpeg"
    tts_dir = outdir / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for row in records:
        sample_id = str(row["sample_id"])
        source = Path(row["tts_source_path"])
        if not source.is_file() or sha256_file(source) != row["tts_source_sha256"]:
            raise ValueError(f"strict TTS source hash mismatch for {row['paired_key']}")
        canonical = tts_dir / f"{sample_id}.wav"
        temp = canonical.with_suffix(".tmp.wav")
        subprocess.run([
            ffmpeg_bin, "-y", "-v", "error", "-i", str(source),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(temp),
        ], check=True, capture_output=True, timeout=120)
        temp.replace(canonical)
        results[sample_id] = {
            "sample_id": sample_id,
            "paired_key": row["paired_key"],
            "speaker_id": row["speaker_id"],
            "split": row["split"],
            "condition": "tts",
            "transcript": row["transcript"],
            "provider": "frozen_strict_manifest_source",
            "source_audio": str(source),
            "source_audio_sha256": row["tts_source_sha256"],
            "canonical_16k_audio": str(canonical),
            "canonical_audio_sha256": sha256_file(canonical),
            "status": "ok",
        }
    summary = {
        "schema_version": 1,
        "manifest_type": "aishell1_n25_strict_tts_canonical",
        "source_manifest": str(cohort_path.resolve()),
        "source_manifest_sha256": sha256_file(cohort_path.resolve()),
        "provider": "frozen_strict_manifest_source",
        "samples_total": len(records),
        "samples_ok": len(results),
        "samples_failed": 0,
        "complete": len(results) == 25,
        "results": results,
        "failed": [],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "tts_meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="")
    args = parser.parse_args(argv)
    result = build(args.cohort.resolve(), args.outdir.resolve(), args.ffmpeg)
    print(json.dumps({"samples_ok": result["samples_ok"], "complete": result["complete"]}))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
