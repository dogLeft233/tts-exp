#!/usr/bin/env python3
"""Generate same-speaker multi-reference TTS variants for style auditing."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
import sys

sys.path.insert(0, str(SCRIPT_DIR))
from tts.faster_qwen3 import FasterQwen3TTSProvider  # noqa: E402

SPEAKERS = ("S0765", "S0901", "S0912")
SEED = 42
LANGUAGE = "Chinese"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_references(records: Sequence[dict[str, Any]], refs_per_target: int) -> dict[str, list[dict[str, Any]]]:
    if refs_per_target <= 0:
        raise ValueError("refs_per_target must be positive")
    grouped: dict[str, list[dict[str, Any]]] = {speaker: [] for speaker in SPEAKERS}
    for row in records:
        speaker = str(row["speaker_id"])
        if speaker in grouped:
            grouped[speaker].append(row)
    for speaker, rows in grouped.items():
        rows.sort(key=lambda row: int(row["sample_id"]))
        if len(rows) < refs_per_target + 1:
            raise ValueError(f"speaker {speaker} needs at least refs_per_target+1 records")
    selected: dict[str, list[dict[str, Any]]] = {}
    for speaker, rows in grouped.items():
        for index, target in enumerate(rows):
            candidates = [rows[(index + offset) % len(rows)] for offset in range(1, len(rows))]
            selected[str(target["sample_id"])] = candidates[:refs_per_target]
    return selected


def generate(manifest_path: Path, outdir: Path, refs_per_target: int = 3) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [row for row in manifest["records"] if str(row["speaker_id"]) in SPEAKERS]
    records.sort(key=lambda row: int(row["sample_id"]))
    if len(records) != 15 or any(sum(str(row["speaker_id"]) == speaker for row in records) != 5 for speaker in SPEAKERS):
        raise ValueError("expected S0765/S0901/S0912 five-sample cohort")
    outdir = outdir.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    audio_dir = outdir / "tts"
    audio_dir.mkdir(parents=True, exist_ok=True)
    references = select_references(records, refs_per_target)
    rows_by_id = {str(row["sample_id"]): row for row in records}
    np.random.seed(SEED)
    import torch

    torch.manual_seed(SEED)
    provider = FasterQwen3TTSProvider(cfg={"faster_qwen3": {}})
    results: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    for target in records:
        target_id = str(target["sample_id"])
        for reference in references[target_id]:
            reference_id = str(reference["sample_id"])
            variant_id = f"{target_id}__ref_{reference_id}"
            try:
                reference_audio = Path(reference["audio_path"]).resolve()
                if not reference_audio.is_file():
                    raise FileNotFoundError(reference_audio)
                generated = provider.generate_voice_clone(
                    text=str(target["transcript"]),
                    ref_audio_path=reference_audio,
                    ref_text=str(reference["transcript"]),
                    language=LANGUAGE,
                )
                waveform = np.asarray(generated.audio, dtype=np.float32)
                provider_wav = audio_dir / f"{variant_id}.provider.wav"
                canonical_wav = audio_dir / f"{variant_id}.wav"
                sf.write(provider_wav, waveform, int(generated.sample_rate), subtype="FLOAT")
                subprocess.run(
                    [ffmpeg, "-y", "-v", "error", "-i", str(provider_wav), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(canonical_wav)],
                    check=True,
                    capture_output=True,
                    timeout=120,
                )
                results[variant_id] = {
                    "variant_id": variant_id,
                    "sample_id": target_id,
                    "target_sample_id": target_id,
                    "reference_sample_id": reference_id,
                    "speaker_id": str(target["speaker_id"]),
                    "paired_key": target["paired_key"],
                    "split": target["split"],
                    "target_transcript": target["transcript"],
                    "target_transcript_sha256": hashlib.sha256(str(target["transcript"]).encode("utf-8")).hexdigest(),
                    "reference_transcript": reference["transcript"],
                    "reference_transcript_sha256": hashlib.sha256(str(reference["transcript"]).encode("utf-8")).hexdigest(),
                    "reference_audio": str(reference_audio),
                    "reference_audio_sha256": sha256_file(reference_audio),
                    "provider": provider.name,
                    "model_id": provider.model_id,
                    "language": LANGUAGE,
                    "seed": SEED,
                    "provider_sample_rate_hz": int(generated.sample_rate),
                    "provider_audio": str(provider_wav),
                    "provider_audio_sha256": sha256_file(provider_wav),
                    "canonical_16k_audio": str(canonical_wav),
                    "canonical_audio_sha256": sha256_file(canonical_wav),
                    "duration_s": round(len(waveform) / int(generated.sample_rate), 4),
                    "status": "ok",
                }
                print(f"OK {variant_id}", flush=True)
            except Exception as exc:
                failures.append({"variant_id": variant_id, "sample_id": target_id, "reference_sample_id": reference_id, "error": str(exc)})
                print(f"FAIL {variant_id}: {exc}", flush=True)
    result = {
        "schema_version": 1,
        "manifest_type": "aishell1_multireference_tts_style_audit",
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "speakers": list(SPEAKERS),
        "sample_count": len(records),
        "references_per_target": refs_per_target,
        "reference_policy": "same_speaker_non_target_cyclic_references",
        "provider": provider.name,
        "model_id": provider.model_id,
        "language": LANGUAGE,
        "seed": SEED,
        "variants_total": len(records) * refs_per_target,
        "variants_ok": len(results),
        "variants_failed": len(failures),
        "complete": len(results) == len(records) * refs_per_target and not failures,
        "results": results,
        "failures": failures,
    }
    (outdir / "tts_meta.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--refs-per-target", type=int, default=3)
    args = parser.parse_args(argv)
    result = generate(args.manifest.resolve(), args.outdir.resolve(), args.refs_per_target)
    print(json.dumps({"complete": result["complete"], "variants_ok": result["variants_ok"], "variants_failed": result["variants_failed"]}, ensure_ascii=False))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
