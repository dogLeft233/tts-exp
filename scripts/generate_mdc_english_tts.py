#!/usr/bin/env python3
"""Generate stable-ID English self-clone TTS outputs for the MDC run."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_mdc_english_pairs import (
    CANONICAL_CHANNELS,
    CANONICAL_SAMPLE_RATE,
    CANONICAL_SUBTYPE,
    EXPECTED_SAMPLE_IDS,
    sha256_file,
    validate_audio,
)
from utils import resolve_repo_path


def _load_pair_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"invalid pair manifest: {path}")
    return payload


def _load_provider(config_path: Path, run_id: str, repo: Path):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    tts_cfg = cfg.get("tts", {})
    from tts import get_tts_provider
    return get_tts_provider(
        tts_cfg,
        run_id=run_id,
        repo_root=repo,
        env=dict(os.environ),
    ), tts_cfg


def _canonicalize_tts(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.tmp.wav")
    if temp.exists():
        temp.unlink()
    import subprocess
    subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg", "-y", "-v", "error", "-i", str(source),
            "-ar", str(CANONICAL_SAMPLE_RATE), "-ac", str(CANONICAL_CHANNELS),
            "-c:a", "pcm_s16le", str(temp),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    temp.replace(destination)
    return validate_audio(destination)


def generate_tts(
    repo: Path,
    pair_manifest_path: Path,
    run_id: str,
    config_path: Path,
    provider: Any | None = None,
    provider_cfg: dict[str, Any] | None = None,
    sample_ids: list[str] | None = None,
    retries: int | None = None,
) -> dict[str, Any]:
    pair_manifest = _load_pair_manifest(pair_manifest_path)
    records = {str(row["sample_key"]): row for row in pair_manifest["records"]}
    pair_manifest_path = pair_manifest_path.resolve()
    selected = sample_ids or list(EXPECTED_SAMPLE_IDS)
    selected = [str(value) for value in selected]
    missing = sorted(set(selected) - set(records))
    if missing:
        raise ValueError(f"pair records missing: {missing}")
    if sample_ids is None and selected != EXPECTED_SAMPLE_IDS:
        raise ValueError("full MDC TTS run requires en_001..en_050")
    if provider is None:
        provider, provider_cfg = _load_provider(config_path, run_id, repo)
    provider_cfg = provider_cfg or {}
    tts_cfg = provider_cfg
    language = str(tts_cfg.get("language", "English"))
    retry_count = int(retries if retries is not None else tts_cfg.get("retry", 1))
    out_dir = repo / "runs" / run_id / "02_tts"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    failed: list[dict[str, Any]] = []
    started = time.monotonic()

    for sample_key in selected:
        row = records[sample_key]
        text = str(row.get("text", "")).strip()
        ref_audio = repo / row["audio_path"]
        if not text:
            failed.append({"sample_key": sample_key, "error": "empty text"})
            continue
        if not ref_audio.is_file():
            failed.append({"sample_key": sample_key, "error": f"missing reference: {ref_audio}"})
            continue
        error = None
        for attempt in range(retry_count + 1):
            try:
                generated = provider.generate_voice_clone(
                    text=text,
                    ref_audio_path=ref_audio,
                    ref_text=text,
                    language=language,
                )
                raw_path = out_dir / f"{sample_key}.provider.wav"
                sf.write(raw_path, np.asarray(generated.audio, dtype=np.float32), int(generated.sample_rate), subtype=CANONICAL_SUBTYPE)
                canonical_path = out_dir / f"{sample_key}.wav"
                qc = _canonicalize_tts(raw_path, canonical_path)
                results[sample_key] = {
                    "sample_key": sample_key,
                    "paired_key": sample_key,
                    "condition": "tts",
                    "utterance_id": f"mdc_tts/{sample_key}/tts",
                    "text": text,
                    "reference_audio": str(ref_audio.relative_to(repo)),
                    "reference_sha256": sha256_file(ref_audio),
                    "generated_audio": str(canonical_path.relative_to(repo)),
                    "generated_source_audio": str(raw_path.relative_to(repo)),
                    "provider": provider.name,
                    "language": language,
                    "sample_rate_hz": int(generated.sample_rate),
                    "model": generated.backend_meta.get("model") or generated.backend_meta.get("model_id"),
                    "backend_meta": generated.backend_meta,
                    "duration_s": qc["duration_s"],
                    "canonical_qc": qc,
                    "status": "ok",
                    "attempt": attempt + 1,
                }
                break
            except Exception as exc:  # provider boundary; preserve failure for rerun
                error = str(exc)
        else:
            failed.append({"sample_key": sample_key, "error": error})

    output = {
        "schema_version": 1,
        "run_id": run_id,
        "dataset": "mdc_tts",
        "language_code": "en",
        "provider": getattr(provider, "name", type(provider).__name__),
        "language": language,
        "seed": tts_cfg.get("seed", 42),
        "source_pair_manifest": str(
            pair_manifest_path.relative_to(repo)
            if pair_manifest_path.is_relative_to(repo)
            else pair_manifest_path
        ),
        "source_pair_manifest_sha256": sha256_file(pair_manifest_path),
        "sample_keys": selected,
        "samples_total": len(selected),
        "samples_ok": len(results),
        "samples_failed": len(failed),
        "complete": len(results) == len(selected) and not failed,
        "elapsed_s": round(time.monotonic() - started, 3),
        "results": results,
        "failed": failed,
    }
    output_path = out_dir / "tts_meta.json"
    temp = output_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(output_path)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pair-manifest", default="")
    parser.add_argument("--config", default="scripts/configs/r1_multiset.yaml")
    parser.add_argument("--sample-ids", default="")
    parser.add_argument("--retries", type=int, default=None)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent.parent
    pair_manifest = Path(args.pair_manifest) if args.pair_manifest else repo / "runs" / args.run_id / "00_pairs" / "pair_manifest.json"
    if not pair_manifest.is_absolute():
        pair_manifest = resolve_repo_path(repo, str(pair_manifest))
    sample_ids = [value.strip() for value in args.sample_ids.split(",") if value.strip()] or None
    result = generate_tts(
        repo=repo,
        pair_manifest_path=pair_manifest,
        run_id=args.run_id,
        config_path=resolve_repo_path(repo, args.config),
        sample_ids=sample_ids,
        retries=args.retries,
    )
    print(f"[mdc-en-tts] {result['samples_ok']}/{result['samples_total']} ok; complete={result['complete']}")
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
