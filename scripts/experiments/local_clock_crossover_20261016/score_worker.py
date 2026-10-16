#!/usr/bin/env python3
"""Official SyncNet audio embeddings and fixed-support distances for pilot WAVs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite score artifact: {path}")
    with path.open("xb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)


def score_one(engine: Any, job: dict[str, Any]) -> dict[str, Any]:
    from scripts.experiments.tts_native_gain_attribution import config

    wav = Path(job["wav_path"])
    visual_path = Path(job["visual_path"])
    if not wav.is_file() or sha256_file(wav) != job["wav_sha256"]:
        raise RuntimeError("pilot WAV missing or hash mismatch")
    if not visual_path.is_file() or sha256_file(visual_path) != job["visual_sha256"]:
        raise RuntimeError("cached visual feature missing or hash mismatch")
    support = np.asarray(job["support"], dtype=np.int64)
    if support.ndim != 1 or support.size < 1 or np.any(np.diff(support) <= 0):
        raise RuntimeError("frozen q support is invalid")
    primary = int(job["primary_index"])
    if not 0 <= primary < config.LAG_COUNT:
        raise RuntimeError("frozen primary lag index is invalid")
    visual = np.load(visual_path, allow_pickle=False)
    audio, audio_meta = engine.extract_audio(wav)
    matrix = engine.distance_matrix(visual, audio)
    if matrix.ndim != 2 or matrix.shape[1] != config.LAG_COUNT or not np.isfinite(matrix).all():
        raise RuntimeError(f"official distance matrix has invalid geometry: {matrix.shape}")
    if int(np.max(support)) >= matrix.shape[0] or int(np.max(support)) >= visual.shape[0]:
        raise RuntimeError("frozen q support is not present in real features; no padding admitted")
    if int(np.min(support)) < config.VSHIFT or int(np.max(support)) + config.VSHIFT >= audio.shape[0]:
        raise RuntimeError("frozen q support includes a lag without a real audio feature")
    rows = matrix[support]
    curve = np.mean(rows, axis=0, dtype=np.float64)
    min_index = int(np.argmin(curve))
    dmin = float(curve[min_index])
    background = float(np.median(curve))
    metrics = {
        "support_count": int(support.size),
        "support_rows": [int(x) for x in support],
        "curve": [float(x) for x in curve],
        "C": float(background - dmin),
        "B_median": background,
        "Dmin": dmin,
        "D0": float(curve[config.VSHIFT]),
        "argmin_index": min_index,
        "offset_audio_minus_video_frames": int(config.VSHIFT - min_index),
        "primary_index_frozen_from_report44": primary,
        "primary_lag_frames": int(primary - config.VSHIFT),
        "primary_fixedlag_distance": float(np.mean(rows[:, primary], dtype=np.float64)),
        "all_lag_columns_real_features": True,
    }
    out = Path(job["receipt_path"]).parent
    a_path, d_path, curve_path = Path(job["audio_feature_path"]), Path(job["distance_path"]), Path(job["curve_path"])
    save_npy(a_path, audio.astype(np.float32, copy=False))
    save_npy(d_path, matrix)
    save_npy(curve_path, curve)
    return {
        "status": "PASS", "id": str(job["id"]), "speaker": str(job["speaker"]),
        "reference_role": str(job["reference_role"]), "wav": {"path": str(wav.resolve()), "sha256": sha256_file(wav)},
        "visual": {"path": str(visual_path.resolve()), "sha256": sha256_file(visual_path), "shape": [int(x) for x in visual.shape]},
        "audio_features": {"path": str(a_path.resolve()), "sha256": sha256_file(a_path), "shape": [int(x) for x in audio.shape], "dtype": str(audio.dtype), "metadata": audio_meta},
        "distance_matrix": {"path": str(d_path.resolve()), "sha256": sha256_file(d_path), "shape": [int(x) for x in matrix.shape], "dtype": str(matrix.dtype)},
        "curve_artifact": {"path": str(curve_path.resolve()), "sha256": sha256_file(curve_path)},
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True, type=Path)
    args = parser.parse_args()
    from scripts.experiments.tts_native_gain_attribution import config
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    job_doc = json.loads(args.jobs.read_text(encoding="utf-8"))
    jobs = job_doc.get("jobs")
    if not isinstance(jobs, list):
        raise RuntimeError("jobs file must contain a jobs array")
    receipt_path = args.jobs.with_name(args.jobs.stem.replace("_jobs", "_worker_result") + ".json")
    results: list[dict[str, Any]] = []
    lock = Path(config.GPU_LOCK)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock_start = time.monotonic()
    with lock.open("a+b") as lock_handle:
        while True:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - lock_start > 590:
                    json_write(receipt_path, {"status": "GPU_LOCK_TIMEOUT", "pid": os.getpid(), "jobs_file_sha256": sha256_file(args.jobs), "results": results})
                    return 2
                time.sleep(1)
        engine = None
        try:
            engine = SyncNetEngine(model_path=config.SYNCNET_MODEL, device="cuda")
            runtime = {name: __import__("importlib.metadata", fromlist=["version"]).version(name) for name in ("numpy", "python_speech_features", "torch")}
            syncnet_source = REPO / "scripts/experiments/tts_native_gain_attribution/syncnet.py"
            environment = {"pid": os.getpid(), "runtime": runtime, "model_path": str(config.SYNCNET_MODEL), "model_sha256": engine.model_hash, "engine_code_hash": engine.code_hash, "syncnet_source_path": str(syncnet_source), "syncnet_source_sha256": sha256_file(syncnet_source), "gpu_lock_path": str(lock.resolve()), "jobs_expected": len(jobs)}
            json_write(receipt_path, {"status": "RUNNING", "environment": environment, "results": results})
            for job in jobs:
                began = time.monotonic()
                try:
                    result = score_one(engine, job)
                    result["elapsed_seconds"] = time.monotonic() - began
                    json_write(Path(job["receipt_path"]), result)
                except Exception as exc:
                    result = {"status": "FAILED", "id": job.get("id"), "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.monotonic() - began}
                    json_write(Path(job["receipt_path"]), result)
                results.append(result)
                json_write(receipt_path, {"status": "RUNNING", "environment": environment, "results": results})
        except Exception as exc:
            json_write(receipt_path, {"status": "FAILED", "pid": os.getpid(), "error": f"{type(exc).__name__}: {exc}", "results": results})
            return 1
        finally:
            if engine is not None:
                engine.close()
        failed = sum(row.get("status") != "PASS" for row in results)
        json_write(receipt_path, {"status": "PASS" if failed == 0 and len(results) == len(jobs) else "PARTIAL", "pid": os.getpid(), "environment": environment, "job_count": len(jobs), "passed_count": len(results) - failed, "failed_count": failed, "results": results})
    return 0 if failed == 0 and len(results) == len(jobs) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SCORE_WORKER_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
