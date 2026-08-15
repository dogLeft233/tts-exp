#!/usr/bin/env python3
"""Generate Qwen cloud voice-clone TTS for the prepared RAMC n25 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
SAMPLE_RATE = 16_000
MODEL = "qwen3-tts-vc-2026-01-22"
PROVIDER = "dashscope_vc"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def redact(text: str, secret: str) -> str:
    return text.replace(secret, "[REDACTED]") if secret else text


def canonicalize(source: Path, destination: Path) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    values, sample_rate = sf.read(destination, dtype="float32", always_2d=False)
    values = np.asarray(values)
    if values.ndim != 1 or int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"canonical output is not mono 16 kHz: {destination}")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"canonical output is empty or non-finite: {destination}")
    return {
        "canonical_16k_audio": str(destination.resolve()),
        "canonical_audio_sha256": sha256_file(destination),
        "sample_rate_hz": int(sample_rate),
        "samples": int(values.size),
        "duration_s": round(float(values.size / sample_rate), 6),
        "peak": float(np.max(np.abs(values))),
    }


def load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = list(payload.get("records", []))
    if payload.get("dataset") != "magicdata_ramc" or len(records) != 25:
        raise ValueError("expected the prepared RAMC n25 manifest")
    if int(payload.get("speaker_count", -1)) != 25:
        raise ValueError("RAMC cohort must contain 25 speakers")
    speakers = {str(row.get("speaker_id")) for row in records}
    if len(speakers) != 25 or "S0770" in speakers:
        raise ValueError("RAMC speaker set is invalid or contains forbidden S0770")
    for row in records:
        if not Path(str(row["audio_path"])).is_file():
            raise FileNotFoundError(row["audio_path"])
        if sha256_file(Path(str(row["audio_path"]))) != str(row["audio_sha256"]):
            raise ValueError(f"natural audio hash mismatch for {row['sample_id']}")
        if not str(row.get("transcript", "")).strip():
            raise ValueError(f"empty transcript for {row['sample_id']}")
    return records


def voice_id_for(provider: Any, registry_key: str, sample_id: str, reference: Path) -> tuple[str, str]:
    audio_sha = provider._audio_sha(reference)
    cached = provider._registry.get(registry_key)
    if cached and cached.get("audio_sha") == audio_sha:
        return str(cached["voice_id"]), str(cached.get("voice_name", "cached"))
    voice_name = f"ttsexpramc{sample_id}"
    voice_id = provider._register_voice(reference, voice_name)
    provider._registry[registry_key] = {
        "audio_sha": audio_sha,
        "voice_id": voice_id,
        "voice_name": voice_name,
    }
    provider._save_registry()
    return str(voice_id), voice_name


def generate(manifest_path: Path, outdir: Path, run_id: str) -> dict[str, Any]:
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {outdir}")
    records = load_manifest(manifest_path)
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required")

    sys.path.insert(0, str(SCRIPT_DIR))
    from tts.dashscope_vc import DashScopeQwen3VCProvider

    provider = DashScopeQwen3VCProvider(
        cfg={"dashscope_vc": {"target_model": MODEL, "voice_prefix": "ttsexpramc"}},
        env={"DASHSCOPE_API_KEY": api_key},
        run_id=run_id,
        repo_root=REPO,
    )
    audio_dir = outdir / "tts"
    results: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    manifest_hash = sha256_file(manifest_path)

    for index, row in enumerate(records, 1):
        sample_id = str(row["sample_id"])
        reference = Path(str(row["audio_path"])).resolve()
        try:
            reference_hash = sha256_file(reference)
            registry_key = f"ramc_{sample_id}"
            voice_id, voice_name = voice_id_for(provider, registry_key, sample_id, reference)
            audio, sample_rate = provider._synthesize(str(row["transcript"]), voice_id, "Chinese")
            waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
            if waveform.size == 0 or not np.isfinite(waveform).all():
                raise ValueError("provider output is empty or non-finite")
            provider_path = audio_dir / f"{sample_id}.provider.wav"
            canonical_path = audio_dir / f"{sample_id}.wav"
            provider_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(provider_path, waveform, int(sample_rate), subtype="FLOAT")
            canonical = canonicalize(provider_path, canonical_path)
            results[sample_id] = {
                "sample_id": sample_id,
                "paired_key": str(row["utterance_id"]),
                "utterance_id": row["utterance_id"],
                "dataset": row["dataset"],
                "conversation": row["conversation"],
                "speaker_id": row["speaker_id"],
                "dialect": row.get("dialect"),
                "selection_rule": row.get("selection_rule"),
                "split": "ramc_pilot25",
                "transcript": row["transcript"],
                "tts_input_transcript": row["transcript"],
                "reference_audio": str(reference),
                "reference_audio_sha256": reference_hash,
                "reference_role": "paired_natural_audio",
                "provider": PROVIDER,
                "model": MODEL,
                "voice_name": voice_name,
                "voice_id": voice_id,
                "provider_sample_rate_hz": int(sample_rate),
                "provider_audio": str(provider_path.resolve()),
                "provider_audio_sha256": sha256_file(provider_path),
                "provider_duration_s": round(float(waveform.size / sample_rate), 6),
                "source_audio_sha256": canonical["canonical_audio_sha256"],
                **canonical,
                "status": "ok",
            }
            print(f"OK {index}/25 sample={sample_id} speaker={row['speaker_id']}", flush=True)
        except Exception as exc:
            error = redact(f"{type(exc).__name__}: {exc}", api_key)
            failures.append({"sample_id": sample_id, "speaker_id": row["speaker_id"], "error": error})
            print(f"FAIL {sample_id}: {error}", flush=True)

    result = {
        "schema_version": 1,
        "manifest_type": "magicdata_ramc_qwen_cloud_tts_n25",
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": manifest_hash,
        "dataset": "magicdata_ramc",
        "provider": PROVIDER,
        "model": MODEL,
        "reference_policy": "paired_natural_audio_per_sample",
        "sample_count": len(records),
        "speaker_count": len({str(row['speaker_id']) for row in records}),
        "samples_ok": len(results),
        "samples_failed": len(failures),
        "complete": len(results) == 25 and not failures,
        "results": results,
        "failures": failures,
    }
    write_json(outdir / "tts_meta.json", result)
    print(json.dumps({"samples_ok": len(results), "failures": len(failures), "complete": result["complete"]}, ensure_ascii=False), flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    result = generate(args.manifest.resolve(), args.outdir.resolve(), args.run_id)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
