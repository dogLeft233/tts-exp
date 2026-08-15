#!/usr/bin/env python3
"""Generate Qwen DashScope voice-clone TTS for the frozen AISHELL-1 n25 cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
EXPECTED_SPEAKERS = ("S0765", "S0901", "S0906", "S0912", "S0913")
SAMPLE_RATE = 16_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def redact(text: str, secret: str) -> str:
    return text.replace(secret, "[REDACTED]") if secret else text


def canonicalize(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
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
    if values.ndim != 1 or int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"canonical output is not 16 kHz mono: {destination}")
    if not np.isfinite(values).all() or values.size == 0:
        raise ValueError(f"canonical output is non-finite or empty: {destination}")
    return {
        "canonical_16k_audio": str(destination),
        "canonical_audio_sha256": sha256_file(destination),
        "sample_rate_hz": int(sample_rate),
        "samples": int(values.size),
        "duration_s": round(float(values.size / sample_rate), 6),
        "peak": float(np.max(np.abs(values))),
    }


def load_cohort(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = list(payload.get("records", []))
    if payload.get("manifest_type") != "aishell1_mfa_linear_predefined_cohort":
        raise ValueError("unexpected cohort manifest type")
    if len(records) != 25:
        raise ValueError(f"expected 25 cohort records, found {len(records)}")
    speakers = {str(row.get("speaker_id")) for row in records}
    if speakers != set(EXPECTED_SPEAKERS):
        raise ValueError(f"unexpected speaker set: {sorted(speakers)}")
    if "S0770" in speakers:
        raise ValueError("heldout S0770 is forbidden")
    if any(sum(str(row.get("speaker_id")) == speaker for row in records) != 5 for speaker in EXPECTED_SPEAKERS):
        raise ValueError("expected five records per speaker")
    return records


def voice_id_for(provider: Any, sample_id: int, reference: Path) -> tuple[str, str]:
    audio_sha = provider._audio_sha(reference)
    cached = provider._registry.get(str(sample_id))
    if cached and cached.get("audio_sha") == audio_sha:
        return str(cached["voice_id"]), str(cached.get("voice_name", "cached"))
    voice_name = f"ttsexpcloud{sample_id}"
    voice_id = provider._register_voice(reference, voice_name)
    provider._registry[str(sample_id)] = {
        "audio_sha": audio_sha,
        "voice_id": voice_id,
        "voice_name": voice_name,
    }
    provider._save_registry()
    return str(voice_id), voice_name


def generate(cohort_path: Path, outdir: Path, run_id: str) -> dict[str, Any]:
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {outdir}")
    records = load_cohort(cohort_path)
    outdir.mkdir(parents=True, exist_ok=True)
    audio_dir = outdir / "tts"
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required")

    sys.path.insert(0, str(SCRIPT_DIR))
    from tts.dashscope_vc import DashScopeQwen3VCProvider

    provider = DashScopeQwen3VCProvider(
        cfg={
            "dashscope_vc": {
                "target_model": "qwen3-tts-vc-2026-01-22",
            }
        },
        env={"DASHSCOPE_API_KEY": api_key},
        run_id=run_id,
        repo_root=REPO,
    )
    results: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    cohort_hash = sha256_file(cohort_path)

    for index, row in enumerate(records, 1):
        sample_id = str(row["sample_id"])
        reference = Path(str(row["audio_path"])).resolve()
        try:
            if not reference.is_file():
                raise FileNotFoundError(reference)
            reference_hash = sha256_file(reference)
            voice_id, voice_name = voice_id_for(provider, int(sample_id), reference)
            audio, sample_rate = provider._synthesize(
                str(row["transcript"]), voice_id, "Chinese"
            )
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
                "paired_key": row["paired_key"],
                "speaker_id": row["speaker_id"],
                "split": row["split"],
                "transcript": row["transcript"],
                "reference_audio": str(reference),
                "reference_audio_sha256": reference_hash,
                "reference_role": "paired_natural_audio",
                "provider": provider.name,
                "model": "qwen3-tts-vc-2026-01-22",
                "voice_name": voice_name,
                "voice_id": voice_id,
                "provider_sample_rate_hz": int(sample_rate),
                "provider_audio": str(provider_path),
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
        "manifest_type": "aishell1_qwen_cloud_tts_n25",
        "cohort_manifest": str(cohort_path.resolve()),
        "cohort_manifest_sha256": cohort_hash,
        "provider": provider.name,
        "model": "qwen3-tts-vc-2026-01-22",
        "reference_policy": "paired_natural_audio_per_sample",
        "sample_count": len(records),
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
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    result = generate(args.cohort.resolve(), args.outdir.resolve(), args.run_id)
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
