#!/usr/bin/env python3
"""Generate a reproducible local-TTS comparison set from an existing LRS3 cohort.

This script deliberately lives outside the fixed 22-sample bridge experiment.  It
uses the first N records of the already QC'ed 500-sample LRS3 cohort and writes
one backend-specific manifest, raw audio, and 16 kHz canonical audio.  Qwen and
IndexTTS are run in separate environments, so invoke this script once per backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COHORT = REPO_ROOT / "runs/lrs3_qwen_cloud_n500_20260817/00_manifest/manifest.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "runs/lrs3_local_tts_n100_20260921"
QWEN_MODEL = REPO_ROOT / "models/Qwen3-TTS-12Hz-0.6B-Base"
INDEX_ROOT = Path("/home/wjj/tts-audio/index-tts")
INDEX_MODEL = INDEX_ROOT / "checkpoints_2"
CANONICAL_SAMPLE_RATE = 16_000
QWEN_MAX_NEW_TOKENS = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_selection(cohort_path: Path, limit: int) -> list[dict[str, Any]]:
    payload = json.loads(cohort_path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"cohort has no records list: {cohort_path}")
    if len(records) < limit:
        raise ValueError(f"cohort has {len(records)} records, but {limit} requested")

    selected: list[dict[str, Any]] = []
    for index, original in enumerate(records[:limit]):
        if not isinstance(original, dict):
            raise ValueError(f"cohort record {index} is not an object")
        record = dict(original)
        sample_id = str(record.get("sample_id", ""))
        text = str(record.get("tts_transcript") or record.get("transcript") or "").strip()
        reference = resolve_repo_path(str(record.get("natural_audio_path", "")))
        if not sample_id or not text:
            raise ValueError(f"record {index} has no sample_id/text")
        if not reference.is_file():
            raise FileNotFoundError(f"reference audio missing for {sample_id}: {reference}")
        record["selection_index"] = index
        record["selected_text"] = text
        record["resolved_reference_audio"] = str(reference.resolve())
        record["reference_audio_sha256"] = sha256_file(reference)
        selected.append(record)
    return selected


@contextmanager
def isolated_seed(seed: int) -> Iterator[None]:
    """Make one synthesis call reproducible without changing the caller's RNG state."""

    numpy_state = np.random.get_state()
    python_state = random.getstate()
    try:
        np.random.seed(seed & 0xFFFFFFFF)
        random.seed(seed)
        try:
            import torch

            torch_state = torch.random.get_rng_state()
            cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            torch = None
            torch_state = None
            cuda_states = None
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        if torch is not None and torch_state is not None:
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)


def as_mono_float(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        raise ValueError("audio output is scalar")
    if array.ndim > 1:
        # soundfile uses (samples, channels); some model APIs use (channels, samples).
        channel_axis = 0 if array.shape[0] <= 8 and array.shape[0] < array.shape[-1] else 1
        array = array.mean(axis=channel_axis)
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scale = float(max(abs(info.min), info.max))
        array = array.astype(np.float32) / scale
    else:
        array = array.astype(np.float32, copy=False)
    if array.size == 0:
        raise ValueError("audio output is empty")
    if not np.isfinite(array).all():
        raise ValueError("audio output contains NaN or Inf")
    return np.clip(array, -1.0, 1.0)


def canonicalize(values: Any, sample_rate: int) -> np.ndarray:
    audio = as_mono_float(values)
    if int(sample_rate) != CANONICAL_SAMPLE_RATE:
        audio = resample_poly(audio, CANONICAL_SAMPLE_RATE, int(sample_rate)).astype(
            np.float32, copy=False
        )
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("canonical audio is empty or non-finite")
    return np.clip(audio, -1.0, 1.0)


def write_pcm16(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, CANONICAL_SAMPLE_RATE, subtype="PCM_16", format="WAV")


def load_qwen_provider() -> Any:
    if not QWEN_MODEL.is_dir():
        raise FileNotFoundError(f"local Qwen model not found: {QWEN_MODEL}")
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.tts.faster_qwen3 import FasterQwen3TTSProvider

    return FasterQwen3TTSProvider(
        {
            "provider": "faster_qwen3",
            "language": "English",
            "faster_qwen3": {
                "model_id": str(QWEN_MODEL),
                "clone_mode": "icl",
                "x_vector_only": False,
                "append_silence": True,
                "strict_backend": True,
                "max_new_tokens": QWEN_MAX_NEW_TOKENS,
            },
        },
        run_id="lrs3_local_tts_n100_20260921",
        repo_root=REPO_ROOT,
    )


def load_index_model() -> Any:
    if not INDEX_MODEL.is_dir():
        raise FileNotFoundError(f"local IndexTTS-2 checkpoint not found: {INDEX_MODEL}")
    if str(INDEX_ROOT) not in sys.path:
        sys.path.insert(0, str(INDEX_ROOT))
    # IndexTTS resolves several auxiliary assets relative to its repository.
    os.chdir(INDEX_ROOT)
    from indextts.infer_v2 import IndexTTS2

    return IndexTTS2(
        cfg_path=str(INDEX_MODEL / "config.yaml"),
        model_dir=str(INDEX_MODEL),
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_qwen_emo=False,
    )


def existing_result_is_valid(result: dict[str, Any], backend_dir: Path) -> bool:
    if result.get("status") != "ok":
        return False
    for field in ("raw_audio", "canonical_audio"):
        path = resolve_repo_path(str(result.get(field, "")))
        if not path.is_file():
            return False
    return True


def synthesize_one(
    backend: str,
    engine: Any,
    record: dict[str, Any],
    raw_path: Path,
    canonical_path: Path,
    seed: int,
) -> dict[str, Any]:
    reference = Path(record["resolved_reference_audio"])
    text = str(record["selected_text"])
    started = time.perf_counter()

    with isolated_seed(seed):
        if backend == "qwen":
            result = engine.generate_voice_clone(
                text=text,
                ref_audio_path=reference,
                ref_text=text,
                language="English",
            )
            provider_audio = as_mono_float(result.audio)
            provider_sample_rate = int(result.sample_rate)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(raw_path, provider_audio, provider_sample_rate, subtype="FLOAT", format="WAV")
            backend_meta = dict(getattr(result, "backend_meta", {}) or {})
        else:
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            output = engine.infer(
                spk_audio_prompt=str(reference),
                text=text,
                output_path=str(raw_path),
                verbose=False,
            )
            if output is not None and not raw_path.is_file():
                # Some IndexTTS versions return an array without writing it.
                if isinstance(output, tuple) and len(output) == 2:
                    returned_audio, returned_sr = output
                    sf.write(raw_path, as_mono_float(returned_audio), int(returned_sr), format="WAV")
                else:
                    raise RuntimeError("IndexTTS returned audio but did not create output_path")
            if not raw_path.is_file():
                raise RuntimeError("IndexTTS did not create output_path")
            provider_audio, provider_sample_rate = sf.read(raw_path, always_2d=False)
            provider_audio = as_mono_float(provider_audio)
            provider_sample_rate = int(provider_sample_rate)
            backend_meta = {"backend": "indextts2", "model_dir": str(INDEX_MODEL)}

    canonical_audio = canonicalize(provider_audio, provider_sample_rate)
    write_pcm16(canonical_path, canonical_audio)
    raw_info = sf.info(raw_path)
    canonical_info = sf.info(canonical_path)
    elapsed = time.perf_counter() - started
    return {
        "sample_id": str(record["sample_id"]),
        "selection_index": int(record["selection_index"]),
        "reference_audio": str(reference.resolve()),
        "reference_audio_sha256": str(record["reference_audio_sha256"]),
        "transcript": text,
        "transcript_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "seed": int(seed),
        "backend": backend,
        "backend_meta": backend_meta,
        "generation_kwargs": (
            {"max_new_tokens": QWEN_MAX_NEW_TOKENS, "append_silence": True}
            if backend == "qwen"
            else {"use_fp16": False, "use_qwen_emo": False}
        ),
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
        "elapsed_s": round(elapsed, 3),
        "status": "ok",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qwen", "index"), required=True)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true", help="regenerate even when a valid row exists")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    cohort_path = resolve_repo_path(args.cohort)
    output_root = resolve_repo_path(args.output_root)
    records = load_selection(cohort_path, args.limit)

    input_dir = output_root / "00_inputs"
    write_json(
        input_dir / f"cohort_n{args.limit}.json",
        {
            "schema_version": 1,
            "source_cohort": str(cohort_path.resolve()),
            "selection_policy": "first_n_records_in_existing_qc_cohort_order",
            "sample_count": len(records),
            "records": records,
        },
    )

    backend_dir = output_root / ("01_qwen_local" if args.backend == "qwen" else "02_index_tts2")
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
        "manifest_type": "lrs3_local_tts_comparison",
        "dataset": "lrs3",
        "language": "English",
        "backend": args.backend,
        "model": (
            "Qwen3-TTS-12Hz-0.6B-Base/faster_qwen3"
            if args.backend == "qwen"
            else "IndexTTS-2/checkpoints_2"
        ),
        "checkpoint": str((QWEN_MODEL if args.backend == "qwen" else INDEX_MODEL).resolve()),
        "source_cohort": str(cohort_path.resolve()),
        "source_cohort_selection": "first_n_records_in_existing_qc_cohort_order",
        "requested_sample_count": len(records),
        "canonical_sample_rate_hz": CANONICAL_SAMPLE_RATE,
        "generation": (
            {"max_new_tokens": QWEN_MAX_NEW_TOKENS, "append_silence": True}
            if args.backend == "qwen"
            else {"use_fp16": False, "use_cuda_kernel": False, "use_deepspeed": False, "use_qwen_emo": False}
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "results": [],
        "failures": [],
    }
    write_json(manifest_path, payload)

    print(f"[{args.backend}] selected {len(records)} records from {cohort_path}", flush=True)
    print(f"[{args.backend}] output: {backend_dir}", flush=True)
    print(f"[{args.backend}] loading model...", flush=True)
    engine = load_qwen_provider() if args.backend == "qwen" else load_index_model()
    print(f"[{args.backend}] model loaded", flush=True)

    for position, record in enumerate(records, start=1):
        sample_id = str(record["sample_id"])
        raw_path = raw_dir / f"{sample_id}.wav"
        canonical_path = canonical_dir / f"{sample_id}.wav"
        old = existing_rows.get(sample_id)
        if old and not args.force and existing_result_is_valid(old, backend_dir):
            row = old
            print(f"[{args.backend}] {position}/{len(records)} {sample_id}: reused", flush=True)
        else:
            seed = 20260921 + int(record["selection_index"])
            try:
                row = synthesize_one(args.backend, engine, record, raw_path, canonical_path, seed)
                print(
                    f"[{args.backend}] {position}/{len(records)} {sample_id}: "
                    f"{row['canonical_duration_s']:.2f}s ({row['elapsed_s']:.1f}s)",
                    flush=True,
                )
            except Exception as exc:  # keep the batch resumable if one clip fails
                row = {
                    "sample_id": sample_id,
                    "selection_index": int(record["selection_index"]),
                    "reference_audio": str(record["resolved_reference_audio"]),
                    "transcript": str(record["selected_text"]),
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                print(f"[{args.backend}] {position}/{len(records)} {sample_id}: FAILED {exc}", flush=True)

        existing_rows[sample_id] = row
        ordered_results = [existing_rows[str(item["sample_id"])] for item in records if str(item["sample_id"]) in existing_rows]
        payload["results"] = ordered_results
        payload["failures"] = [item for item in ordered_results if item.get("status") == "failed"]
        payload["completed_sample_count"] = sum(item.get("status") == "ok" for item in ordered_results)
        payload["failed_sample_count"] = len(payload["failures"])
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest_path, payload)

    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    payload["status"] = "complete" if not payload["failures"] else "partial_failure"
    write_json(manifest_path, payload)
    print(
        f"[{args.backend}] finished: {payload['completed_sample_count']}/{len(records)} ok, "
        f"{payload['failed_sample_count']} failed",
        flush=True,
    )
    return 0 if not payload["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
