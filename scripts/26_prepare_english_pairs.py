#!/usr/bin/env python3
"""Prepare paired English natural and TTS inputs for mechanism experiments."""

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

from utils import detect_sample_ids, load_config, resolve_repo_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text())
    records = payload.get("results", []) if isinstance(payload, dict) else payload
    return {int(record["sample_id"]): record for record in records}


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
    return {
        "sample_rate_hz": int(sample_rate),
        "channels": int(info.channels),
        "subtype": info.subtype,
        "duration_s": round(float(info.duration), 6),
        "frames": int(info.frames),
        "peak": round(peak, 8),
        "rms": round(rms, 8),
        "clipped_fraction": round(float(np.mean(np.abs(audio) >= 0.9999)), 10),
        "sha256": sha256_file(path),
    }


def canonicalize_audio(
    source: Path, destination: Path, sample_rate: int, ffmpeg_bin: str = "ffmpeg"
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin, "-y", "-v", "error", "-i", str(source),
        "-ar", str(sample_rate), "-ac", "1", "-c:a", "pcm_s16le",
        str(destination),
    ]
    subprocess.run(command, check=True, capture_output=True, timeout=60)


def prepare_pairs(
    repo: Path,
    run_id: str,
    cfg: dict[str, Any],
    smoke: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    paths = cfg["paths"]
    natural_manifest_path = resolve_repo_path(repo, paths["natural_manifest"])
    tts_manifest_path = resolve_repo_path(repo, paths["tts_manifest"])
    image_dir = resolve_repo_path(repo, paths["image_dir"])
    natural_records = load_manifest(natural_manifest_path)
    tts_records = load_manifest(tts_manifest_path)
    sample_ids = detect_sample_ids(repo, smoke=smoke, cfg=cfg)

    out_dir = repo / "runs" / run_id / "00_pairs"
    natural_out = out_dir / "natural"
    tts_out = out_dir / "tts"
    target_rate = int(cfg["experiment"]["canonical_sample_rate"])
    env_ffmpeg = (
        Path(cfg["paths"]["envs_dir"]) / "ditto" / "bin" / "ffmpeg"
        if cfg.get("paths", {}).get("envs_dir") else None
    )
    ffmpeg_bin = str(env_ffmpeg) if env_ffmpeg and env_ffmpeg.exists() else (
        shutil.which("ffmpeg") or "ffmpeg"
    )
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for sample_id in sample_ids:
        try:
            natural = natural_records[sample_id]
            tts = tts_records[sample_id]
            natural_text = str(natural["text"]).strip()
            tts_text = str(tts["text"]).strip()
            if natural_text != tts_text:
                raise ValueError("natural/TTS transcript mismatch")

            natural_source = resolve_repo_path(
                repo, natural.get("local_path") or natural.get("reference_audio")
            )
            tts_source = resolve_repo_path(
                repo, tts.get("generated_audio") or tts.get("local_path")
            )
            image_path = image_dir / f"{sample_id}.png"
            for required in (natural_source, tts_source, image_path):
                if not required.exists():
                    raise FileNotFoundError(required)

            nat_dst = natural_out / f"{sample_id}.wav"
            tts_dst = tts_out / f"{sample_id}.wav"
            if overwrite or not nat_dst.exists():
                canonicalize_audio(natural_source, nat_dst, target_rate, ffmpeg_bin)
            if overwrite or not tts_dst.exists():
                canonicalize_audio(tts_source, tts_dst, target_rate, ffmpeg_bin)

            natural_qc = validate_audio(nat_dst)
            tts_qc = validate_audio(tts_dst)
            expected = cfg["experiment"]
            for arm, qc in (("natural", natural_qc), ("tts", tts_qc)):
                if qc["sample_rate_hz"] != target_rate or qc["channels"] != 1:
                    raise ValueError(f"{arm} canonical format mismatch")
                if qc["subtype"] != expected["canonical_subtype"]:
                    raise ValueError(f"{arm} subtype mismatch: {qc['subtype']}")

            records.append({
                "sample_id": sample_id,
                "librispeech_id": natural.get("librispeech_id"),
                "speaker_id": natural.get("speaker_id"),
                "chapter_id": natural.get("chapter_id"),
                "text": natural_text,
                "image": str(image_path.relative_to(repo)),
                "image_sha256": sha256_file(image_path),
                "natural": {
                    "source": str(natural_source.relative_to(repo)),
                    "canonical": str(nat_dst.relative_to(repo)),
                    "source_qc": validate_audio(natural_source),
                    "canonical_qc": natural_qc,
                },
                "tts": {
                    "source": str(tts_source.relative_to(repo)),
                    "canonical": str(tts_dst.relative_to(repo)),
                    "source_qc": validate_audio(tts_source),
                    "canonical_qc": tts_qc,
                    "provider": tts.get("backend"),
                    "model": tts.get("model"),
                    "voice_id": tts.get("voice_id"),
                },
            })
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": str(exc)})

    image_groups: dict[str, list[int]] = {}
    for record in records:
        image_groups.setdefault(record["image_sha256"], []).append(record["sample_id"])

    result = {
        "run_id": run_id,
        "config": "scripts/configs/english_wav2sem.yaml",
        "samples_expected": sample_ids,
        "samples_ok": len(records),
        "samples_failed": len(failures),
        "complete": len(records) == len(sample_ids) and not failures,
        "image_identity_groups": [
            ids for ids in image_groups.values() if len(ids) > 1
        ],
        "records": records,
        "failed": failures,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pair_manifest.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    provenance = {
        "run_id": run_id,
        "natural_manifest": str(natural_manifest_path.relative_to(repo)),
        "natural_manifest_sha256": sha256_file(natural_manifest_path),
        "tts_manifest": str(tts_manifest_path.relative_to(repo)),
        "tts_manifest_sha256": sha256_file(tts_manifest_path),
        "canonicalization": {
            "sample_rate_hz": target_rate,
            "channels": 1,
            "codec": "pcm_s16le",
            "loudness_normalization": False,
        },
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--config", default="scripts/configs/english_wav2sem.yaml"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    cfg = load_config(repo, args.config)
    result = prepare_pairs(repo, args.run_id, cfg, args.smoke, args.overwrite)
    print(
        f"[pairs] {result['samples_ok']}/{len(result['samples_expected'])} ok, "
        f"{result['samples_failed']} failed"
    )
    for failure in result["failed"]:
        print(f"  sample {failure['sample_id']}: {failure['error']}")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
