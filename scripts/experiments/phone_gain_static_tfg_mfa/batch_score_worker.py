"""Batch SyncNet worker for one sample; accepts generated videos only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.static_image_bridge.score_worker import (  # noqa: E402
    audio_embedding,
    read_video,
    sha256_file,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _visual_embedding(cropped: np.ndarray, scorer: Any, torch: Any, *, device: str, batch_size: int) -> tuple[Any, int]:
    count = int(cropped.shape[0] - 4)
    if count <= 0:
        raise ValueError("video has insufficient SyncNet visual windows")
    batches = []
    for start in range(0, count, max(1, int(batch_size))):
        stop = min(count, start + max(1, int(batch_size)))
        sequences = np.stack([cropped[index:index + 5] for index in range(start, stop)], axis=0)
        tensor = torch.from_numpy(np.transpose(sequences, (0, 4, 1, 2, 3))).float().to(device)
        with torch.no_grad():
            batches.append(scorer.__S__.forward_lip(tensor).detach().cpu())
    return torch.cat(batches, dim=0), count


def _write_video_scores(*, video_arm: str, video_path: Path, cropped: np.ndarray, crop_hashes: list[str], video_meta: dict[str, Any], visual: Any, visual_count: int, audio_features: dict[str, Any], audio_meta: dict[str, dict[str, int]], model: Path, output_dir: Path, vshift: int, torch: Any) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_path = output_dir / f"{video_arm}__visual.npy"
    np.save(visual_path, visual.numpy().astype(np.float64), allow_pickle=False)
    rows = []
    for audio_arm, aud in audio_features.items():
        audio_count = int(audio_meta[audio_arm]["feature_count"])
        audio_samples = int(audio_meta[audio_arm]["sample_count"])
        rows_for_audio = min(int(visual_count), audio_count)
        matrix = np.full((rows_for_audio, 2 * int(vshift) + 1), np.nan, dtype=np.float64)
        for column, lag in enumerate(range(-int(vshift), int(vshift) + 1)):
            start = max(0, -lag)
            stop = min(rows_for_audio, audio_count - lag)
            if stop <= start:
                continue
            left = visual[start:stop].float()
            right = aud[start + lag:stop + lag].float()
            matrix[start:stop, column] = torch.nn.functional.pairwise_distance(left, right).numpy()
        matrix_path = output_dir / f"{video_arm}__{audio_arm}__matrix.npy"
        np.save(matrix_path, matrix, allow_pickle=False)
        row = {
            "schema_version": 1,
            "status": "complete",
            "video_arm": video_arm,
            "audio_arm": str(audio_arm),
            "video": str(video_path.resolve()),
            "video_sha256": sha256_file(video_path),
            "audio": str(audio_meta[audio_arm]["path"]),
            "audio_sha256": str(audio_meta[audio_arm]["sha256"]),
            "device": "cuda",
            "model_sha256": sha256_file(model),
            "vshift": int(vshift),
            "matrix": str(matrix_path.resolve()),
            "matrix_sha256": sha256_file(matrix_path),
            "matrix_shape": [int(value) for value in matrix.shape],
            "finite_mask": np.isfinite(matrix).all(axis=1).astype(np.uint8).tolist(),
            "visual": str(visual_path.resolve()),
            "visual_sha256": sha256_file(visual_path),
            "visual_count": int(visual_count),
            "audio_feature_count": audio_count,
            "audio_sample_count": audio_samples,
            "video_meta": video_meta,
            "score_crop_hashes": crop_hashes,
        }
        result_path = output_dir / f"{video_arm}__{audio_arm}__worker.json"
        _write_json(result_path, row)
        row["worker_result"] = str(result_path.resolve())
        row["worker_result_sha256"] = sha256_file(result_path)
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-json", type=Path, required=True)
    parser.add_argument("--audio-json", type=Path, required=True)
    parser.add_argument("--score-box", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--vshift", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--request-sha256", default=None)
    args = parser.parse_args()

    import torch
    from SyncNetInstance import SyncNetInstance

    if not torch.cuda.is_available():
        raise RuntimeError("SyncNet scoring requires CUDA; CPU fallback is forbidden")
    videos = json.loads(args.videos_json.read_text(encoding="utf-8"))
    audios = json.loads(args.audio_json.read_text(encoding="utf-8"))
    score_box = json.loads(args.score_box.read_text(encoding="utf-8"))
    if not isinstance(videos, dict) or not videos or not isinstance(audios, dict) or not audios:
        raise ValueError("videos-json and audio-json must contain mappings")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    for path in videos.values():
        if not Path(str(path)).is_file():
            raise FileNotFoundError(path)
    for path in audios.values():
        if not Path(str(path)).is_file():
            raise FileNotFoundError(path)
    if not isinstance(score_box, dict) or not isinstance(score_box.get("box"), list) or len(score_box["box"]) != 4:
        raise ValueError("score-box must contain a four-value box")

    device = "cuda"
    scorer = SyncNetInstance(device=device)
    scorer.loadParameters(str(args.model))
    scorer.eval()
    audio_features: dict[str, Any] = {}
    audio_meta: dict[str, dict[str, int | str]] = {}
    for audio_arm, raw_path in audios.items():
        audio_path = Path(str(raw_path)).resolve()
        features, feature_count, sample_count = audio_embedding(audio_path, scorer, device, torch)
        audio_features[str(audio_arm)] = features
        audio_meta[str(audio_arm)] = {"path": str(audio_path), "sha256": sha256_file(audio_path), "feature_count": int(feature_count), "sample_count": int(sample_count)}

    rows = []
    for video_arm, raw_path in videos.items():
        video_path = Path(str(raw_path)).resolve()
        cropped, crop_hashes, video_meta = read_video(video_path, score_box)
        visual, visual_count = _visual_embedding(cropped, scorer, torch, device=device, batch_size=int(args.batch_size))
        rows.extend(_write_video_scores(video_arm=str(video_arm), video_path=video_path, cropped=cropped, crop_hashes=crop_hashes, video_meta=video_meta, visual=visual, visual_count=visual_count, audio_features=audio_features, audio_meta=audio_meta, model=args.model, output_dir=args.output_root / str(video_arm), vshift=int(args.vshift), torch=torch))
    _write_json(args.output_root / "batch_manifest.json", {"status": "PASS", "video_arms": sorted(str(key) for key in videos), "audio_arms": sorted(str(key) for key in audios), "rows": len(rows), "model_sha256": sha256_file(args.model), "request_sha256": args.request_sha256})
    print(json.dumps({"status": "PASS", "video_count": len(videos), "audio_count": len(audios), "cells": len(rows)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
