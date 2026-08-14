#!/usr/bin/env python3
"""Generate paired faster_qwen3 ICL self-clone TTS for the pilot utterances.

Reads a pilot manifest (alimeeting25/manifest.json), synthesizes each
transcript in the voice cloned from its own natural audio, and writes:
  - provider raw output  (<outdir>/tts/{id}.provider.wav, 24 kHz float)
  - canonical 16 kHz     (<outdir>/tts/{id}.wav, s16le via ffmpeg)
  - metadata             (<outdir>/tts_meta.json)

Run with the qwen3 venv: ~/.venvs/qwen3/bin/python
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tts.faster_qwen3 import FasterQwen3TTSProvider  # noqa: E402

LANGUAGE = "Chinese"
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"]
    if manifest.get("manifest_type") == "aishell1_mfa_linear_predefined_cohort":
        if manifest.get("cohort", {}).get("sample_count") != len(records):
            raise ValueError("cohort count does not match records")
        if len({str(r["paired_key"]) for r in records}) != len(records):
            raise ValueError("cohort paired_key values must be unique")
        if len({str(r["sample_id"]) for r in records}) != len(records):
            raise ValueError("cohort sample_id values must be unique")
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    tts_dir = outdir / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)
    import torch
    torch.manual_seed(SEED)

    provider = FasterQwen3TTSProvider(cfg={"faster_qwen3": {}})
    results: dict[str, Any] = {}
    failed: list[dict[str, Any]] = []
    started = time.monotonic()
    for record in records:
        sample_id = str(record["sample_id"])
        text = str(record["transcript"]).strip()
        ref = Path(record["audio_path"])
        if not ref.is_file():
            failed.append({"sample_id": sample_id, "error": f"missing reference {ref}"})
            continue
        error = None
        for attempt in range(3):
            try:
                generated = provider.generate_voice_clone(
                    text=text, ref_audio_path=ref, ref_text=text, language=LANGUAGE,
                )
                audio = np.asarray(generated.audio, dtype=np.float32)
                provider_wav = tts_dir / f"{sample_id}.provider.wav"
                sf.write(provider_wav, audio, int(generated.sample_rate), subtype="FLOAT")
                canonical = tts_dir / f"{sample_id}.wav"
                subprocess.run(
                    [shutil.which("ffmpeg") or "ffmpeg", "-y", "-v", "error", "-i", str(provider_wav),
                     "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(canonical)],
                    check=True, capture_output=True, timeout=120,
                )
                results[sample_id] = {
                    "sample_id": sample_id,
                    "paired_key": str(record.get("paired_key", sample_id)),
                    "speaker_id": record.get("speaker_id"),
                    "split": record.get("split"),
                    "condition": "tts",
                    "transcript_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "natural_source": str(ref.resolve()),
                    "natural_source_sha256": sha256_file(ref),
                    "provider": provider.name,
                    "language": LANGUAGE,
                    "seed": SEED,
                    "provider_sample_rate_hz": int(generated.sample_rate),
                    "provider_audio": str(provider_wav),
                    "provider_audio_sha256": sha256_file(provider_wav),
                    "canonical_16k_audio": str(canonical),
                    "canonical_audio_sha256": sha256_file(canonical),
                    "duration_s": round(len(audio) / int(generated.sample_rate), 3),
                    "status": "ok",
                    "attempt": attempt + 1,
                }
                break
            except Exception as exc:  # provider boundary; keep failure for rerun
                error = str(exc)
        else:
            failed.append({"sample_id": sample_id, "error": error})
        print(f"OK {sample_id}" if sample_id in results else f"FAIL {sample_id}", flush=True)

    summary = {
        "schema_version": 1,
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.manifest.resolve()),
        "manifest_type": manifest.get("manifest_type"),
        "provider": provider.name,
        "language": LANGUAGE,
        "seed": SEED,
        "samples_total": len(records),
        "samples_ok": len(results),
        "samples_failed": len(failed),
        "complete": len(results) == len(records) and not failed,
        "elapsed_s": round(time.monotonic() - started, 1),
        "results": results,
        "failed": failed,
    }
    (outdir / "tts_meta.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )
    print(f"[pilot-tts] {summary['samples_ok']}/{summary['samples_total']} ok; complete={summary['complete']}")
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
