#!/usr/bin/env python3
"""Generate a fixed-reference-voice TTS control without touching self-clone outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from tts.faster_qwen3 import FasterQwen3TTSProvider  # noqa: E402

SELECTED_SPEAKERS = ("S0765", "S0901", "S0912")
LANGUAGE = "Chinese"
SEED = 42


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(manifest_path: Path, outdir: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [row for row in manifest["records"] if str(row["speaker_id"]) in SELECTED_SPEAKERS]
    if len(records) != 15 or any(sum(str(row["speaker_id"]) == speaker for row in records) != 5 for speaker in SELECTED_SPEAKERS):
        raise ValueError("expected S0765/S0901/S0912 five-sample control cohort")
    records.sort(key=lambda row: int(row["sample_id"]))
    reference = next(row for row in records if str(row["speaker_id"]) == "S0765")
    reference_audio = Path(reference["audio_path"]).resolve()
    reference_text = str(reference["transcript"])
    if not reference_audio.is_file():
        raise FileNotFoundError(reference_audio)
    outdir = outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    audio_dir = outdir / "tts"
    audio_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    import torch
    torch.manual_seed(SEED)
    provider = FasterQwen3TTSProvider(cfg={"faster_qwen3": {}})
    results: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    for record in records:
        sample_id = str(record["sample_id"])
        text = str(record["transcript"])
        try:
            generated = provider.generate_voice_clone(
                text=text,
                ref_audio_path=reference_audio,
                ref_text=reference_text,
                language=LANGUAGE,
            )
            audio = np.asarray(generated.audio, dtype=np.float32)
            provider_wav = audio_dir / f"{sample_id}.provider.wav"
            sf.write(provider_wav, audio, int(generated.sample_rate), subtype="FLOAT")
            canonical = audio_dir / f"{sample_id}.wav"
            subprocess.run(
                [shutil.which("ffmpeg") or "ffmpeg", "-y", "-v", "error", "-i", str(provider_wav),
                 "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(canonical)],
                check=True, capture_output=True, timeout=120,
            )
            results[sample_id] = {
                "sample_id": sample_id,
                "paired_key": record["paired_key"],
                "speaker_id": record["speaker_id"],
                "split": record["split"],
                "condition": "tts_fixed_reference_voice",
                "transcript": text,
                "transcript_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "fixed_reference_audio": str(reference_audio),
                "fixed_reference_audio_sha256": sha256_file(reference_audio),
                "fixed_reference_speaker_id": reference["speaker_id"],
                "fixed_reference_text": reference_text,
                "provider": provider.name,
                "model_id": provider.model_id,
                "language": LANGUAGE,
                "seed": SEED,
                "provider_sample_rate_hz": int(generated.sample_rate),
                "provider_audio": str(provider_wav),
                "provider_audio_sha256": sha256_file(provider_wav),
                "canonical_16k_audio": str(canonical),
                "canonical_audio_sha256": sha256_file(canonical),
                "duration_s": round(len(audio) / int(generated.sample_rate), 3),
                "status": "ok",
            }
            print(f"OK {sample_id}", flush=True)
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": str(exc)})
            print(f"FAIL {sample_id}: {exc}", flush=True)
    summary = {
        "schema_version": 1,
        "manifest_type": "aishell1_fixed_reference_voice_tts_control",
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "fixed_reference_audio": str(reference_audio),
        "fixed_reference_audio_sha256": sha256_file(reference_audio),
        "fixed_reference_speaker_id": reference["speaker_id"],
        "fixed_reference_text": reference_text,
        "selected_speakers": list(SELECTED_SPEAKERS),
        "provider": provider.name,
        "model_id": provider.model_id,
        "language": LANGUAGE,
        "seed": SEED,
        "samples_total": len(records),
        "samples_ok": len(results),
        "samples_failed": len(failures),
        "complete": len(results) == len(records) and not failures,
        "elapsed_s": round(time.monotonic() - started, 1),
        "results": results,
        "failed": failures,
    }
    (outdir / "tts_meta.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = generate(args.manifest.resolve(), args.outdir.resolve())
    print(json.dumps({"complete": result["complete"], "samples_ok": result["samples_ok"], "samples_failed": result["samples_failed"]}, ensure_ascii=False))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
