#!/usr/bin/env python3
"""Run paired StableAvatar inference for the TFG evaluation inputs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--natural-dir", required=True, type=Path)
    parser.add_argument("--tts-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--teacache-threshold", type=float, default=0.1)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed-base", type=int, default=20260730)
    parser.add_argument("--only", nargs="*", type=int)
    return parser.parse_args()


def write_status(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.resolve()
    os.chdir(model_dir)
    sys.path.insert(0, str(model_dir))

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    import importlib

    core = importlib.import_module("core")
    generate_video = core.generate_video
    load_pipeline = core.load_pipeline

    image_dir = args.image_dir.resolve()
    natural_dir = args.natural_dir.resolve()
    tts_dir = args.tts_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "batch_status.json"

    sample_ids = args.only or list(range(1, 14))
    cases = [(condition, sample_id) for condition in ("natural_raw", "tts_raw") for sample_id in sample_ids]
    records: list[dict] = []
    pipeline = None

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Cases: {len(cases)}; steps={args.steps}; frames={args.frames}", flush=True)

    for condition, sample_id in cases:
        final_path = output_dir / condition / f"{sample_id}.mp4"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{sample_id}.png"
        audio_dir = natural_dir if condition == "natural_raw" else tts_dir
        audio_path = audio_dir / f"{sample_id}.wav"
        record = {
            "condition": condition,
            "sample_id": sample_id,
            "image": str(image_path),
            "audio": str(audio_path),
            "output": str(final_path),
            "seed": args.seed_base + sample_id,
        }
        started = time.time()

        if final_path.is_file() and final_path.stat().st_size > 0:
            record.update({"status": "skipped_existing", "elapsed_s": 0.0})
            records.append(record)
            write_status(status_path, records)
            print(f"[{condition} {sample_id}] existing output, skip", flush=True)
            continue

        if not image_path.is_file() or not audio_path.is_file():
            record.update({"status": "failed", "error": "missing input"})
            records.append(record)
            write_status(status_path, records)
            print(f"[{condition} {sample_id}] missing input", flush=True)
            continue

        work_dir = output_dir / ".work" / condition / str(sample_id)
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            if pipeline is None:
                print("Loading StableAvatar pipeline...", flush=True)
                pipeline = load_pipeline()
                print("Pipeline loaded", flush=True)

            generated_path, actual_seed = generate_video(
                pipeline=pipeline,
                image_path=str(image_path),
                audio_path=str(audio_path),
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                seed=args.seed_base + sample_id,
                fps=25,
                clip_sample_n_frames=args.frames,
                overlap_window_length=5,
                teacache_threshold=args.teacache_threshold,
                output_dir=str(work_dir),
            )
            shutil.move(generated_path, final_path)
            raw_video = Path(generated_path.replace("_audio.mp4", ".mp4"))
            if raw_video.exists():
                raw_video.unlink()
            shutil.rmtree(work_dir, ignore_errors=True)
            record.update({
                "status": "ok",
                "seed": actual_seed,
                "bytes": final_path.stat().st_size,
                "elapsed_s": round(time.time() - started, 2),
            })
            print(f"[{condition} {sample_id}] ok {record['elapsed_s']}s -> {final_path}", flush=True)
        except Exception as exc:  # Keep the batch running and report all failures.
            record.update({
                "status": "failed",
                "error": repr(exc),
                "elapsed_s": round(time.time() - started, 2),
            })
            print(f"[{condition} {sample_id}] FAILED: {exc!r}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        records.append(record)
        write_status(status_path, records)

    ok = sum(record["status"] in {"ok", "skipped_existing"} for record in records)
    failed = len(records) - ok
    print(f"Finished: {ok} ok, {failed} failed", flush=True)
    return 0 if failed == 0 and len(records) == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
