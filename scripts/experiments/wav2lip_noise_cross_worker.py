"""SyncNet feature worker for ``wav2lip_noise_cross_v1``.

The worker runs in the dedicated SyncNet environment.  A single model
instance serves every row in a plan, and each unique video/audio condition is
forwarded once before the five crossed matrices are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from scripts.experiments.tts_native_gain_attribution.common import file_sha256  # noqa: E402
from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine  # noqa: E402


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = _canonical_hash(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return body


def _save_array(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, np.asarray(value), allow_pickle=False)
    temporary.replace(path)
    array = np.asarray(value)
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "shape": [int(item) for item in array.shape], "dtype": str(array.dtype)}


def _unique_paths(mapping: dict[str, str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for key, raw in mapping.items():
        path = Path(raw).resolve()
        if not path.is_file():
            raise RuntimeError(f"worker input is missing: {path}")
        result[str(key)] = path
    return result


def process_row(engine: SyncNetEngine, row: dict[str, Any], model_hash: str) -> dict[str, Any]:
    output_dir = Path(str(row["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = _unique_paths({str(k): str(v) for k, v in dict(row["videos"]).items()})
    audios = _unique_paths({str(k): str(v) for k, v in dict(row["audios"]).items()})
    visual_meta: dict[str, Any] = {}
    audio_meta: dict[str, Any] = {}
    visual_arrays: dict[str, np.ndarray] = {}
    audio_arrays: dict[str, np.ndarray] = {}
    for key, path in videos.items():
        array, metadata = engine.extract_visual(path)
        visual_arrays[key] = np.asarray(array, dtype=np.float32)
        visual_meta[key] = {**metadata, "input": str(path), "input_sha256": file_sha256(path), "feature": _save_array(output_dir / f"visual__{key}.npy", visual_arrays[key])}
    for key, path in audios.items():
        array, metadata = engine.extract_audio(path)
        audio_arrays[key] = np.asarray(array, dtype=np.float32)
        audio_meta[key] = {**metadata, "input": str(path), "input_sha256": file_sha256(path), "feature": _save_array(output_dir / f"audio__{key}.npy", audio_arrays[key])}
    cells: list[dict[str, Any]] = []
    for cell in row.get("cells", []):
        key = str(cell["key"])
        video_key = str(cell["video"])
        audio_key = str(cell["audio"])
        if video_key not in visual_arrays or audio_key not in audio_arrays:
            raise RuntimeError(f"cell {key} refers to an unextracted condition")
        matrix = engine.distance_matrix(visual_arrays[video_key], audio_arrays[audio_key])
        matrix_info = _save_array(output_dir / f"matrix__{key}.npy", matrix)
        cells.append({
            "key": key,
            "video": video_key,
            "audio": audio_key,
            "video_input": visual_meta[video_key]["input"],
            "video_input_sha256": visual_meta[video_key]["input_sha256"],
            "audio_input": audio_meta[audio_key]["input"],
            "audio_input_sha256": audio_meta[audio_key]["input_sha256"],
            "visual_feature": visual_meta[video_key]["feature"],
            "audio_feature": audio_meta[audio_key]["feature"],
            "matrix": matrix_info,
        })
    return {
        "schema_version": 1,
        "sample_id": int(row["sample_id"]),
        "source": str(row.get("source", "")),
        "kind": str(row.get("kind", "main")),
        "status": "complete",
        "model_sha256": model_hash,
        "visual": visual_meta,
        "audio": audio_meta,
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()
    os.environ["FFMPEG"] = str(args.ffmpeg)
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SyncNet scoring requested CUDA, but CUDA is unavailable")
    torch.manual_seed(42)
    np.random.seed(42)
    if args.device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    engine = SyncNetEngine(model_path=args.model.resolve(), batch_size=max(1, int(args.batch_size)), device=args.device)
    model_hash = file_sha256(args.model)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        for row in plan.get("rows", []):
            try:
                rows.append(process_row(engine, dict(row), model_hash))
            except Exception as exc:  # keep independent source failures in the ledger
                failures.append({"sample_id": row.get("sample_id"), "source": row.get("source"), "kind": row.get("kind", "main"), "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        engine.close()
    payload = {
        "schema_version": 1,
        "protocol_id": "wav2lip_noise_cross_v1",
        "status": "complete" if not failures else "partial",
        "device": args.device,
        "model_sha256": model_hash,
        "rows": rows,
        "failures": failures,
    }
    _write_json(args.result, payload)
    print(json.dumps({"status": payload["status"], "rows": len(rows), "failures": len(failures)}), flush=True)
    return 0 if rows or not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
