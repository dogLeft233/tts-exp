#!/usr/bin/env python3
"""Generate SiliconFlow CosyVoice2 outputs for the existing LRS3 n=100 set.

The API provider uploads each natural reference voice once, caches the returned
voice URI in its run-local registry, and then synthesizes the paired transcript.
Raw provider audio and a 16 kHz mono PCM16 canonical copy are both retained.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf

from generate_lrs3_local_tts import (
    CANONICAL_SAMPLE_RATE,
    REPO_ROOT,
    as_mono_float,
    canonicalize,
    sha256_file,
    write_json,
    write_pcm16,
)
from tts.siliconflow import SiliconFlowProvider


DEFAULT_INPUT = REPO_ROOT / "runs/lrs3_local_tts_n100_20260921/00_inputs/cohort_n100.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/lrs3_local_tts_n100_20260921"
RUN_ID = "lrs3_siliconflow_cosyvoice2_n100_20260921"
MODEL = "FunAudioLLM/CosyVoice2-0.5B"
PROVIDER_SAMPLE_RATE = 24_000


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_records(path: Path, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or len(records) < limit:
        raise ValueError(f"input manifest has fewer than {limit} records: {path}")
    selected: list[dict[str, Any]] = []
    for record in records[:limit]:
        item = dict(record)
        reference = Path(str(item.get("resolved_reference_audio", "")))
        text = str(item.get("selected_text") or item.get("transcript") or "").strip()
        sample_id = str(item.get("sample_id", ""))
        if not sample_id or not text or not reference.is_file():
            raise ValueError(f"invalid LRS3 input record: {sample_id}")
        item["resolved_reference_audio"] = str(reference.resolve())
        item["selected_text"] = text
        item.setdefault("reference_audio_sha256", sha256_file(reference))
        selected.append(item)
    return selected


def make_provider() -> SiliconFlowProvider:
    # The key is read only from the process environment.  It is never put in
    # the config, manifest, registry, or generated files.
    return SiliconFlowProvider(
        {
            "provider": "siliconflow",
            "siliconflow": {
                "api_key_env": "SILICONFLOW_API_KEY",
                "model": MODEL,
                "response_format": "wav",
                "sample_rate": PROVIDER_SAMPLE_RATE,
                "speed": 1.0,
                "gain": 0.0,
            },
        },
        env=dict(os.environ),
        run_id=RUN_ID,
        repo_root=REPO_ROOT,
    )


def valid_existing_row(row: dict[str, Any]) -> bool:
    if row.get("status") != "ok":
        return False
    if not all(Path(str(row.get(field, ""))).is_file() for field in ("raw_audio", "canonical_audio")):
        return False
    qc = row.get("audio_qc", {})
    return (
        float(row.get("canonical_duration_s", 0.0)) >= 0.25
        and float(qc.get("peak", 0.0)) >= 0.005
        and float(qc.get("rms", 0.0)) >= 0.0005
    )


def synthesize_one(
    provider: SiliconFlowProvider,
    record: dict[str, Any],
    raw_path: Path,
    canonical_path: Path,
) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    reference = Path(record["resolved_reference_audio"])
    text = str(record["selected_text"])
    started = time.perf_counter()
    result = None
    provider_audio = None
    provider_sample_rate = None
    qc: dict[str, float] = {}
    last_qc_error = ""
    for attempt in range(1, 4):
        result = provider.generate_voice_clone(
            text=text,
            ref_audio_path=reference,
            ref_text=text,
            language="English",
            sample_id=sample_id,  # provider registry key; keeps retries cacheable
        )
        provider_audio = as_mono_float(result.audio)
        provider_sample_rate = int(result.sample_rate)
        peak = float(abs(provider_audio).max()) if provider_audio.size else 0.0
        rms = float((provider_audio.astype("float64") ** 2).mean() ** 0.5) if provider_audio.size else 0.0
        duration_s = float(provider_audio.size / provider_sample_rate) if provider_sample_rate else 0.0
        qc = {"peak": round(peak, 8), "rms": round(rms, 8), "duration_s": round(duration_s, 4)}
        if duration_s >= 0.25 and peak >= 0.005 and rms >= 0.0005:
            break
        last_qc_error = f"invalid audio QC on attempt {attempt}: {qc}"
        if attempt < 3:
            time.sleep(float(attempt))
    else:
        raise RuntimeError(last_qc_error)
    assert result is not None and provider_audio is not None and provider_sample_rate is not None
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(raw_path, provider_audio, provider_sample_rate, subtype="FLOAT", format="WAV")
    canonical_audio = canonicalize(provider_audio, provider_sample_rate)
    write_pcm16(canonical_path, canonical_audio)
    raw_info = sf.info(raw_path)
    canonical_info = sf.info(canonical_path)
    return {
        "sample_id": sample_id,
        "selection_index": int(record["selection_index"]),
        "reference_audio": str(reference.resolve()),
        "reference_audio_sha256": str(record["reference_audio_sha256"]),
        "transcript": text,
        "backend": "siliconflow",
        "model": MODEL,
        "backend_meta": dict(getattr(result, "backend_meta", {}) or {}),
        "generation_kwargs": {
            "model": MODEL,
            "response_format": "wav",
            "sample_rate": PROVIDER_SAMPLE_RATE,
            "speed": 1.0,
            "gain": 0.0,
        },
        "raw_audio": str(raw_path.resolve()),
        "raw_audio_sha256": sha256_file(raw_path),
        "raw_sample_rate_hz": int(raw_info.samplerate),
        "raw_channels": int(raw_info.channels),
        "raw_duration_s": round(float(raw_info.duration), 4),
        "canonical_audio": str(canonical_path.resolve()),
        "canonical_audio_sha256": sha256_file(canonical_path),
        "canonical_sample_rate_hz": int(canonical_info.samplerate),
        "canonical_channels": int(canonical_info.channels),
        "canonical_duration_s": round(float(canonical_info.duration), 4),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "audio_qc": qc,
        "provider_attempts": attempt,
        "status": "ok",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    input_manifest = resolve_path(args.input_manifest)
    output_root = resolve_path(args.output_root)
    records = load_records(input_manifest, args.limit)
    backend_dir = output_root / "03_siliconflow_cosyvoice2"
    raw_dir = backend_dir / "raw"
    canonical_dir = backend_dir / "canonical_16k"
    manifest_path = backend_dir / "tts_manifest.json"

    existing_payload: dict[str, Any] = {}
    if manifest_path.is_file() and not args.force:
        existing_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing_rows = {
        str(row.get("sample_id")): row
        for row in existing_payload.get("results", [])
        if isinstance(row, dict)
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "manifest_type": "lrs3_siliconflow_tts_comparison",
        "dataset": "lrs3",
        "language": "English",
        "backend": "siliconflow",
        "model": MODEL,
        "source_input_manifest": str(input_manifest.resolve()),
        "sample_selection": "same_first_n_records_as_local_qwen_and_indextts_run",
        "requested_sample_count": len(records),
        "canonical_sample_rate_hz": CANONICAL_SAMPLE_RATE,
        "generation": {"model": MODEL, "response_format": "wav", "sample_rate": PROVIDER_SAMPLE_RATE, "speed": 1.0, "gain": 0.0},
        "registry_path": str((REPO_ROOT / "runs" / RUN_ID / "02_tts" / "voice_registry_sf.json").resolve()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "results": [],
        "failures": [],
    }
    write_json(manifest_path, payload)
    print(f"selected {len(records)} records from {input_manifest}", flush=True)
    print(f"output: {backend_dir}", flush=True)
    provider = make_provider()
    print(f"provider ready: {MODEL}", flush=True)

    for position, record in enumerate(records, start=1):
        sample_id = str(record["sample_id"])
        raw_path = raw_dir / f"{sample_id}.wav"
        canonical_path = canonical_dir / f"{sample_id}.wav"
        old = existing_rows.get(sample_id)
        if old and not args.force and valid_existing_row(old):
            row = old
            print(f"{position}/{len(records)} {sample_id}: reused", flush=True)
        else:
            try:
                row = synthesize_one(provider, record, raw_path, canonical_path)
                print(
                    f"{position}/{len(records)} {sample_id}: "
                    f"{row['canonical_duration_s']:.2f}s ({row['elapsed_s']:.1f}s)",
                    flush=True,
                )
            except Exception as exc:  # keep the API batch resumable
                row = {
                    "sample_id": sample_id,
                    "selection_index": int(record["selection_index"]),
                    "reference_audio": str(record["resolved_reference_audio"]),
                    "transcript": str(record["selected_text"]),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                print(f"{position}/{len(records)} {sample_id}: FAILED {exc}", flush=True)
        existing_rows[sample_id] = row
        ordered = [existing_rows[str(item["sample_id"])] for item in records if str(item["sample_id"]) in existing_rows]
        payload["results"] = ordered
        payload["failures"] = [item for item in ordered if item.get("status") == "failed"]
        payload["completed_sample_count"] = sum(item.get("status") == "ok" for item in ordered)
        payload["failed_sample_count"] = len(payload["failures"])
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, payload)

    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    payload["status"] = "complete" if not payload["failures"] else "partial_failure"
    write_json(manifest_path, payload)
    print(
        f"finished: {payload['completed_sample_count']}/{len(records)} ok, "
        f"{payload['failed_sample_count']} failed",
        flush=True,
    )
    return 0 if not payload["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
