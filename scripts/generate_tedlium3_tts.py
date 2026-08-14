#!/usr/bin/env python3
"""Generate TED-LIUM 3 self-clone TTS outputs with faster_qwen3 (ICL mode).

Streams TED-LIUM 3 parquet shards one at a time (audio bytes are written to
disk immediately and released, never accumulated in memory), cleans <unk>
tokens from transcripts, and runs faster_qwen3 self-clone TTS
(ref_text == text) for training-augmentation pairs (natural/tts).

Supports:
  --limit N       process first N selected rows (smoke)
  --all           process every eligible row in the corpus
  resume:         natural wavs and TTS wavs that already exist are skipped

Run inside ~/.venvs/qwen3 (local V100) or the ditto conda env (server).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils import resolve_repo_path  # noqa: E402

PARQUET_DIR_DEFAULT = "data/tedlium3_30pct/data"
TARGET_SAMPLE_RATE = 16_000


def clean_transcript(text: str) -> str:
    """Lowercase, collapse whitespace, drop <unk> tokens and NA markers."""
    cleaned = str(text).replace("<unk>", "").replace("<NA>", "")
    return " ".join(cleaned.lower().split())


def extract_natural(
    shard_dir: Path,
    natural_dir: Path,
    *,
    limit: int,
    seed: int,
) -> list[dict]:
    """Stream shards -> write natural wavs; return metadata rows (no bytes kept).

    One shard at a time: per-shard rows are written to disk and released before
    the next shard is read, so peak memory stays at ~one shard (~0.5GB).
    """
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards in {shard_dir}")
    natural_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    selected: list[dict] = []
    for shard in shards:
        if limit and len(selected) >= limit:
            break
        table = pq.read_table(shard, columns=["audio", "text", "speaker_id", "gender", "id"])
        audio_col = table.column("audio").combine_chunks()
        bytes_list = audio_col.field("bytes").to_pylist()
        text_list = table.column("text").to_pylist()
        spk_list = table.column("speaker_id").to_pylist()
        id_list = table.column("id").to_pylist()
        shard_rows: list[dict] = []
        for audio_bytes, text, spk, uid in zip(bytes_list, text_list, spk_list, id_list):
            if not audio_bytes or len(audio_bytes) < TARGET_SAMPLE_RATE:
                continue
            transcript = clean_transcript(str(text))
            if not transcript:
                continue
            shard_rows.append(
                {"id": str(uid), "speaker_id": str(spk), "text": transcript, "audio_bytes": audio_bytes}
            )
        rng.shuffle(shard_rows)
        for row in shard_rows:
            if limit and len(selected) >= limit:
                break
            out = natural_dir / f"{row['id']}.wav"
            if not out.exists():
                with open(out, "wb") as handle:
                    handle.write(row["audio_bytes"])
            selected.append(
                {
                    "id": row["id"],
                    "speaker_id": row["speaker_id"],
                    "text": row["text"],
                    "natural_path": str(out),
                }
            )
        # shard_rows released here; only lightweight metadata is retained
    return selected


def generate_tts(provider, rows: list[dict], tts_dir: Path, language: str, retries: int) -> tuple[dict, list[dict]]:
    tts_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    failed: list[dict] = []
    for row in rows:
        uid = row["id"]
        ref_audio = Path(row["natural_path"])
        out_path = tts_dir / f"{uid}.wav"
        if out_path.exists():
            results[uid] = {"id": uid, "status": "cached", "path": str(out_path)}
            continue
        error = None
        for attempt in range(retries + 1):
            try:
                generated = provider.generate_voice_clone(
                    text=row["text"],
                    ref_audio_path=ref_audio,
                    ref_text=row["text"],
                    language=language,
                )
                sf.write(out_path, np.asarray(generated.audio, dtype=np.float32), int(generated.sample_rate))
                results[uid] = {
                    "id": uid,
                    "status": "ok",
                    "path": str(out_path),
                    "duration_s": round(len(generated.audio) / int(generated.sample_rate), 3),
                    "sample_rate": int(generated.sample_rate),
                    "attempt": attempt + 1,
                }
                break
            except Exception as exc:
                error = str(exc)
        else:
            failed.append({"id": uid, "error": error})
    return results, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="tedlium3_tts")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="")
    parser.add_argument("--language", default="English")
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    if args.all and args.limit:
        parser.error("--all and --limit are mutually exclusive")
    limit = None if args.all else (args.limit or None)

    repo = Path(__file__).resolve().parent.parent
    cfg = {}
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    tts_cfg = cfg.get("tts", {})
    language = tts_cfg.get("language", args.language)

    shard_dir = resolve_repo_path(repo, PARQUET_DIR_DEFAULT)
    run_dir = repo / "runs" / args.run_id
    natural_dir = run_dir / "00_natural"
    tts_dir = run_dir / "02_tts"

    t0 = time.monotonic()
    rows = extract_natural(shard_dir, natural_dir, limit=limit or 0, seed=args.seed)
    if not rows:
        print("[tedlium3-tts] no rows extracted")
        return 1
    mode = "all" if limit is None else f"limit={limit}"
    print(f"[tedlium3-tts] {mode}: {len(rows)} rows selected in {time.monotonic()-t0:.0f}s")
    print(f"[tedlium3-tts] first: {rows[0]['id']}")
    print(f"[tedlium3-tts] natural audio -> {natural_dir}")

    from tts import get_tts_provider

    provider = get_tts_provider(
        tts_cfg,
        run_id=args.run_id,
        repo_root=repo,
        env=dict(os.environ),
    )
    print(f"[tedlium3-tts] provider: {provider.name}")

    t0 = time.monotonic()
    results, failed = generate_tts(provider, rows, tts_dir, language, args.retries)
    elapsed = time.monotonic() - t0
    done = sum(1 for value in results.values() if value.get("status") in {"ok", "cached"})
    print(f"[tedlium3-tts] done={done} failed={len(failed)} elapsed={elapsed:.0f}s")

    output = {
        "schema_version": 1,
        "run_id": args.run_id,
        "dataset": "tedlium3_30pct",
        "language": language,
        "provider": provider.name,
        "seed": args.seed,
        "mode": mode,
        "samples_total": len(rows),
        "samples_ok": done,
        "samples_failed": len(failed),
        "elapsed_s": round(elapsed, 3),
        "results": results,
        "failed": failed,
    }
    meta_path = tts_dir / "tts_meta.json"
    meta_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[tedlium3-tts] meta -> {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
