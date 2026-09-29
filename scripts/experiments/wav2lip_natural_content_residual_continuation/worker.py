from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import file_sha256


def _configure_determinism() -> dict[str, Any]:
    import torch

    torch.manual_seed(config.GENERATION_SEED)
    np.random.seed(config.GENERATION_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing silent CPU fallback")
    return {"torch_seed": config.GENERATION_SEED, "numpy_seed": config.GENERATION_SEED, "deterministic_algorithms": True, "cudnn_benchmark": False, "cudnn_deterministic": True, "tf32": False, "device": "cuda", "device_name": torch.cuda.get_device_name(0), "device_count": torch.cuda.device_count()}


def _chunk_mels(mel: np.ndarray) -> list[np.ndarray]:
    value = np.asarray(mel, dtype=np.float32)
    if value.shape != (config.MEL_BINS, config.MEL_FRAMES) or not np.isfinite(value).all():
        raise RuntimeError(f"mel does not match frozen shape: {value.shape}")
    chunks: list[np.ndarray] = []
    index = 0
    while True:
        start = int(index * 80.0 / config.FPS)
        if start + 16 > value.shape[1]:
            chunks.append(value[:, value.shape[1] - 16 :])
            break
        chunks.append(value[:, start : start + 16])
        index += 1
    if len(chunks) != config.WAV2LIP_FRAMES:
        raise RuntimeError(f"frozen mel chunk count changed: {len(chunks)}")
    return chunks


def _render_one(model: Any, static_face: np.ndarray, mel: np.ndarray) -> list[np.ndarray]:
    from scripts.experiments.wav2lip_face_roi_replacement.generation_worker import render_arm

    frames = [np.ascontiguousarray(static_face.copy()) for _ in range(config.WAV2LIP_FRAMES)]
    boxes = [[0, config.CROP_SIZE, 0, config.CROP_SIZE] for _ in range(config.WAV2LIP_FRAMES)]
    return render_arm(model, frames, boxes, _chunk_mels(mel), config.WAV2LIP_BATCH_SIZE, "cuda")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU-only continuation Wav2Lip worker")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=config.WAV2LIP_CHECKPOINT)
    parser.add_argument("--ffmpeg", type=Path, default=config.FFMPEG)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)

    import torch

    runtime = _configure_determinism()
    if not args.checkpoint.is_file() or file_sha256(args.checkpoint) != config.WAV2LIP_CHECKPOINT_SHA256:
        raise RuntimeError("Wav2Lip checkpoint is missing or has changed")
    from scripts.experiments.wav2lip_face_roi_replacement.generation_worker import encode_video, load_model

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or not isinstance(plan.get("rows"), list):
        raise RuntimeError("generation plan is malformed")
    model = load_model(args.checkpoint, "cuda")
    rows: list[dict[str, Any]] = []
    for plan_row in plan["rows"]:
        sample_id = str(plan_row["sample_id"])
        static_path = Path(str(plan_row["static_face"]))
        static_face = np.asarray(np.load(static_path, allow_pickle=False))
        if static_face.shape != (config.CROP_SIZE, config.CROP_SIZE, 3) or static_face.dtype != np.uint8:
            raise RuntimeError(f"static face is not uint8 224x224: {sample_id}")
        for arm, mel_path_value in dict(plan_row["arms"]).items():
            mel_path = Path(str(mel_path_value))
            output = Path(str(plan_row["outputs"][arm]))
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                raise RuntimeError(f"generation output already exists: {output}")
            generated = _render_one(model, static_face, np.asarray(np.load(mel_path, allow_pickle=False), dtype=np.float32))
            encoded = encode_video(generated, output, args.ffmpeg)
            rows.append({"sample_id": sample_id, "arm": str(arm), "output": str(output.resolve()), "output_sha256": file_sha256(output), "frame_count": len(generated), "static_face": str(static_path.resolve()), "static_face_sha256": file_sha256(static_path), "mel": str(mel_path.resolve()), "mel_sha256": file_sha256(mel_path), "raw_frame_sha256": encoded["raw_frame_sha256"], "codec": "ffv1", "device": "cuda", "batch_size": config.WAV2LIP_BATCH_SIZE})
            print(f"GENERATED {sample_id} {arm}", flush=True)
    payload = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "stage_id": "gpu_generation", "status": "complete", "runtime": runtime, "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_sha256(args.checkpoint), "rows": rows}
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
