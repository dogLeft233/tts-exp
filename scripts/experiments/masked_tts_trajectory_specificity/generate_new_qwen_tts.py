"""Generate paired English Qwen TTS audio for a new LRS3 cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

MODEL = "qwen3-tts-vc-2026-01-22"
PROVIDER = "dashscope_vc"
SAMPLE_RATE = 16_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def canonicalize(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(destination)],
        check=True,
    )
    values, sample_rate = sf.read(str(destination), dtype="float32", always_2d=False)
    values = np.asarray(values, dtype=np.float32)
    if int(sample_rate) != SAMPLE_RATE or values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"invalid canonical TTS audio: {destination}")
    return {
        "canonical_16k_audio": str(destination.resolve()),
        "canonical_audio_sha256": sha256_file(destination),
        "sample_rate_hz": int(sample_rate),
        "samples": int(values.size),
        "duration_s": round(float(values.size / sample_rate), 6),
        "peak": float(np.max(np.abs(values))),
    }


def redact(text: str, secret: str) -> str:
    return text.replace(secret, "[REDACTED]") if secret else text


def voice_id_for(provider: Any, sample_id: str, reference: Path) -> tuple[str, str]:
    audio_sha = provider._audio_sha(reference)
    cached = provider._registry.get(sample_id)
    if cached and cached.get("audio_sha") == audio_sha:
        return str(cached["voice_id"]), str(cached["voice_name"])
    voice_name = f"l3new{audio_sha[:11]}"
    voice_id = provider._register_voice(reference, voice_name)
    provider._registry[sample_id] = {"audio_sha": audio_sha, "voice_id": voice_id, "voice_name": voice_name}
    provider._save_registry()
    return str(voice_id), voice_name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    outdir = args.outdir.resolve()
    if outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = list(manifest.get("records", []))
    if manifest.get("status") != "complete" or len(records) != 16:
        raise ValueError("new cohort manifest is incomplete")
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required")

    import sys
    script_path = Path(__file__).resolve()
    sys.path.insert(0, str(script_path.parents[2]))
    from tts.dashscope_vc import DashScopeQwen3VCProvider

    repo_root = script_path.parents[3]
    provider = DashScopeQwen3VCProvider(
        cfg={"dashscope_vc": {"target_model": MODEL, "voice_prefix": "l3new"}},
        env={"DASHSCOPE_API_KEY": api_key},
        run_id=args.run_id,
        repo_root=repo_root,
    )
    provider = DashScopeQwen3VCProvider(
        cfg={"dashscope_vc": {"target_model": MODEL, "voice_prefix": "l3new"}},
        env={"DASHSCOPE_API_KEY": api_key},
        run_id=args.run_id,
        repo_root=repo_root,
    )
    existing_results: dict[str, dict[str, Any]] = {}
    meta_path = outdir / "tts_meta.json"
    if meta_path.is_file():
        try:
            existing_payload = json.loads(meta_path.read_text(encoding="utf-8"))
            existing_results = dict(existing_payload.get("results", {}))
        except (OSError, json.JSONDecodeError):
            existing_results = {}
    results: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    audio_dir = outdir / "tts"
    for index, record in enumerate(records, 1):
        sample_id = str(record["sample_id"])
        reference = Path(str(record["natural_audio"])).resolve()
        transcript = str(record["transcript"])
        cached = existing_results.get(sample_id)
        if (
            cached
            and cached.get("status") == "ok"
            and cached.get("source_group") == str(record["source_group"])
            and cached.get("transcript") == transcript
            and cached.get("reference_audio") == str(reference)
            and cached.get("reference_audio_sha256") == str(record["natural_audio_sha256"])
        ):
            canonical_path = Path(str(cached.get("canonical_16k_audio", "")))
            provider_path = Path(str(cached.get("provider_audio", "")))
            if (
                canonical_path.is_file()
                and provider_path.is_file()
                and cached.get("canonical_audio_sha256") == sha256_file(canonical_path)
                and cached.get("provider_audio_sha256") == sha256_file(provider_path)
            ):
                results[sample_id] = cached
                print(f"REUSE {index}/16 {sample_id}", flush=True)
                continue
        try:
            if not reference.is_file() or sha256_file(reference) != str(record["natural_audio_sha256"]):
                raise ValueError("natural reference audio hash mismatch")
            voice_id, voice_name = voice_id_for(provider, sample_id, reference)
            audio, sample_rate = provider._synthesize(transcript, voice_id, "English")
            waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
            if waveform.size == 0 or not np.isfinite(waveform).all():
                raise ValueError("provider output is empty or non-finite")
            provider_path = audio_dir / f"{sample_id}.provider.wav"
            canonical_path = audio_dir / f"{sample_id}.wav"
            provider_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(provider_path), waveform, int(sample_rate), subtype="PCM_16")
            canonical = canonicalize(provider_path, canonical_path)
            results[sample_id] = {
                "sample_id": sample_id,
                "source_group": str(record["source_group"]),
                "transcript": transcript,
                "reference_audio": str(reference),
                "reference_audio_sha256": sha256_file(reference),
                "reference_role": "paired_natural_audio",
                "provider": PROVIDER,
                "model": MODEL,
                "voice_id": voice_id,
                "provider_audio": str(provider_path.resolve()),
                "provider_audio_sha256": sha256_file(provider_path),
                "provider_sample_rate_hz": int(sample_rate),
                "provider_duration_s": round(float(waveform.size / int(sample_rate)), 6),
                **canonical,
                "status": "ok",
            }
            print(f"OK {index}/16 {sample_id}", flush=True)
        except Exception as exc:
            error = redact(f"{type(exc).__name__}: {exc}", api_key)
            failures.append({"sample_id": sample_id, "source_group": str(record["source_group"]), "error": error})
            print(f"FAIL {sample_id}: {error}", flush=True)

    payload = {
        "schema_version": 1,
        "manifest_type": "lrs3_masked_tts_new_confirmation_qwen_tts",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "provider": PROVIDER,
        "model": MODEL,
        "language": "English",
        "reference_policy": "paired_natural_audio_per_sample",
        "records_expected": len(records),
        "samples_ok": len(results),
        "samples_failed": len(failures),
        "complete": len(results) == len(records) and not failures,
        "results": results,
        "failures": failures,
    }
    write_json(outdir / "tts_meta.json", payload)
    print(json.dumps({"samples_ok": len(results), "samples_failed": len(failures), "complete": payload["complete"]}, ensure_ascii=False), flush=True)
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
