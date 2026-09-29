#!/usr/bin/env python3
"""Pinned SyncNet scorer for the conditional Wav2Lip branch.

This process is intentionally separate from the orchestration process: the
repository's SyncNet dependencies and CUDA runtime live in a dedicated
environment.  It writes raw embeddings and the complete distance matrix so a
later audit can reconstruct the common-window C score without rerunning the
model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%s.tmp" % (path.name, os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def _curve(matrix: np.ndarray, support: Sequence[int], *, vshift: int) -> Dict[str, Any]:
    rows = np.asarray(list(support), dtype=np.int64)
    if rows.size < 1 or np.any(np.diff(rows) <= 0):
        raise ValueError("SyncNet support is empty or not strictly increasing")
    curve = matrix[rows].astype(np.float32).mean(axis=0, dtype=np.float32).astype(np.float64)
    index = int(np.argmin(curve))
    distance = float(curve[index])
    background = float(np.median(curve))
    return {
        "support_rows": [int(value) for value in rows],
        "support_count": int(rows.size),
        "curve": [float(value) for value in curve],
        "min_index": index,
        "official_offset": int(vshift - index),
        "sync_d": distance,
        "background_b": background,
        "sync_c": background - distance,
        "d0": float(curve[vshift]),
        "boundary_best": bool(index in (0, len(curve) - 1)),
    }


def score(video: Path, audio: Path, model: Path, output: Path, *, device: str = "cuda") -> Dict[str, Any]:
    from scripts.experiments.tts_native_gain_attribution import config
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    engine = SyncNetEngine(model_path=model, device=device)
    try:
        visual, visual_meta = engine.extract_visual(video)
        audial, audio_meta = engine.extract_audio(audio)
        matrix = SyncNetEngine.distance_matrix(visual, audial)
    finally:
        engine.close()
    row_count = int(matrix.shape[0])
    support = list(range(int(config.VSHIFT), row_count - int(config.VSHIFT)))
    result: Dict[str, Any] = {
        "status": "COMPLETE" if len(support) >= 1 else "UNMEASURABLE",
        "video": str(video.resolve()),
        "audio": str(audio.resolve()),
        "video_sha256": _sha256(video),
        "audio_sha256": _sha256(audio),
        "model": str(model.resolve()),
        "model_sha256": _sha256(model),
        "visual_meta": visual_meta,
        "audio_meta": audio_meta,
        "feature_count": {"visual": int(len(visual)), "audio": int(len(audial)), "matrix": row_count},
        "vshift": int(config.VSHIFT),
        "lag_count": int(config.LAG_COUNT),
        "min_required_rows": 50,
    }
    matrix_path = output.with_suffix(".distance.npy")
    visual_path = output.with_suffix(".visual.npy")
    audio_path = output.with_suffix(".audio.npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, matrix.astype(np.float64), allow_pickle=False)
    np.save(visual_path, visual.astype(np.float32), allow_pickle=False)
    np.save(audio_path, audial.astype(np.float32), allow_pickle=False)
    result.update({"distance_matrix": str(matrix_path.resolve()), "distance_matrix_sha256": _sha256(matrix_path), "visual_features": str(visual_path.resolve()), "audio_features": str(audio_path.resolve()), "support": support})
    if support:
        result["metrics"] = _curve(matrix, support, vshift=int(config.VSHIFT))
        result["measurable_for_main_sync_c"] = len(support) >= 50
    else:
        result["measurable_for_main_sync_c"] = False
    _write_json(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        result = score(args.video, args.audio, args.model, args.output, device=args.device)
        print(json.dumps({"status": result["status"], "output": str(args.output)}, ensure_ascii=False), flush=True)
        return 0 if result["status"] == "COMPLETE" else 2
    except Exception as exc:
        error = {"status": "ERROR", "error_type": type(exc).__name__, "reason": str(exc)}
        _write_json(args.output, error)
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
