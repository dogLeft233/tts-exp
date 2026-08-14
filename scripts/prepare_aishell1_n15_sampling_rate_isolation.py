#!/usr/bin/env python3
"""Freeze original n15 strict pairs and canonicalize TTS to n25's 16k contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(repo: Path, value: str) -> Path:
    path = Path(value)
    prefix = "/mnt/e/Documents/tts-audio/tts-exp/"
    if str(path).startswith(prefix):
        path = repo / str(path)[len(prefix):]
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build(source_manifest: Path, outdir: Path, repo: Path, ffmpeg: str = "") -> dict[str, Any]:
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    grouped: dict[int, dict[str, Mapping[str, Any]]] = {}
    for row in source.get("records", []):
        if row.get("split") != "valid" or row.get("speaker_id") != "S0765" or int(row["sample_id"]) > 15:
            continue
        condition = str(row.get("condition"))
        if condition not in ("natural", "tts"):
            continue
        sample_id = int(row["sample_id"])
        if condition in grouped.setdefault(sample_id, {}):
            raise ValueError(f"duplicate {condition} record for sample {sample_id}")
        grouped[sample_id][condition] = row
    if set(grouped) != set(range(1, 16)) or any(set(rows) != {"natural", "tts"} for rows in grouped.values()):
        raise ValueError("strict source does not contain exactly 15 complete pairs")
    ordered_keys = [str(grouped[i]["natural"]["paired_key"]) for i in range(1, 16)]
    if ordered_keys != sorted(ordered_keys):
        raise ValueError("source sample ids are not the original sorted paired-key prefix")
    ffmpeg_bin = ffmpeg or shutil.which("ffmpeg") or "ffmpeg"
    outdir = outdir.resolve()
    canonical_dir = outdir / "tts_canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    token_records: dict[str, Any] = {}
    for sample_id in range(1, 16):
        natural = grouped[sample_id]["natural"]
        tts = grouped[sample_id]["tts"]
        if natural["paired_key"] != tts["paired_key"] or natural["transcript"] != tts["transcript"]:
            raise ValueError(f"pair identity mismatch at sample {sample_id}")
        natural_path = local_path(repo, str(natural["audio_path"]))
        tts_path = local_path(repo, str(tts["audio_path"]))
        if sha256_file(natural_path) != str(natural["source_sha256"] or natural["source_sha256"]):
            raise ValueError(f"natural source hash mismatch at sample {sample_id}")
        if sha256_file(tts_path) != str(tts["source_sha256"]):
            raise ValueError(f"TTS source hash mismatch at sample {sample_id}")
        canonical = canonical_dir / f"{sample_id}.wav"
        temp = canonical.with_suffix(".tmp.wav")
        subprocess.run([
            ffmpeg_bin, "-y", "-v", "error", "-i", str(tts_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(temp),
        ], check=True, capture_output=True, timeout=120)
        temp.replace(canonical)
        records.append({
            "sample_id": sample_id,
            "paired_key": natural["paired_key"],
            "speaker_id": natural["speaker_id"],
            "split": natural["split"],
            "transcript": natural["transcript"],
            "audio_path": str(natural_path),
            "natural_source_path": str(natural_path),
            "natural_source_sha256": natural["source_sha256"],
            "tts_source_path": str(tts_path),
            "tts_source_sha256": tts["source_sha256"],
            "strict_alignment_source": str(source_manifest.resolve()),
            "strict_natural_tokens": natural["tokens"],
            "strict_tts_tokens": tts["tokens"],
            "canonical_16k_audio": str(canonical),
            "canonical_audio_sha256": sha256_file(canonical),
        })
        token_records[str(sample_id)] = {
            "sample_id": sample_id,
            "paired_key": natural["paired_key"],
            "speaker_id": natural["speaker_id"],
            "split": natural["split"],
            "transcript": natural["transcript"],
            "natural": {"tokens": natural["tokens"]},
            "tts": {"tokens": tts["tokens"], "audio_sha256": tts["source_sha256"]},
        }
    manifest = {
        "schema_version": 1,
        "manifest_type": "aishell1_mfa_linear_original_n15_sampling_rate_isolation",
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest.resolve()),
        "selection": {
            "speaker_id": "S0765",
            "split": "valid",
            "count": 15,
            "sample_ids": list(range(1, 16)),
            "ordered_paired_keys": ordered_keys,
            "selection_rule": "original_strict_manifest_sample_ids_1_to_15_sorted_prefix",
        },
        "records": records,
    }
    tokens = {
        "schema_version": 1,
        "manifest_type": "aishell1_mfa_linear_original_n15_tokens",
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest.resolve()),
        "samples_ok": 15,
        "failures": [],
        "records": token_records,
    }
    tts_meta = {
        "schema_version": 1,
        "manifest_type": "aishell1_original_n15_tts_canonical_16k",
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest.resolve()),
        "provider": "frozen_strict_manifest_source",
        "canonicalization": {"tool": "ffmpeg", "args": ["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"]},
        "samples_total": 15,
        "samples_ok": 15,
        "complete": True,
        "results": {
            str(row["sample_id"]): {
                "sample_id": row["sample_id"],
                "paired_key": row["paired_key"],
                "speaker_id": row["speaker_id"],
                "split": row["split"],
                "condition": "tts",
                "transcript": row["transcript"],
                "provider": "frozen_strict_manifest_source",
                "source_audio": row["tts_source_path"],
                "source_audio_sha256": row["tts_source_sha256"],
                "canonical_16k_audio": row["canonical_16k_audio"],
                "canonical_audio_sha256": row["canonical_audio_sha256"],
                "status": "ok",
            }
            for row in records
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "n15_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (outdir / "tokens.json").write_text(json.dumps(tokens, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (outdir / "tts_meta.json").write_text(json.dumps(tts_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"manifest": manifest, "tokens": tokens, "tts_meta": tts_meta}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    build(args.source_manifest.resolve(), args.outdir.resolve(), repo, args.ffmpeg)
