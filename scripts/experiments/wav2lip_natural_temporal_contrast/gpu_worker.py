from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import config
from .common import file_sha256, read_json, write_json
from .transform import chunk_mels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GPU worker for the natural temporal contrast probe")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback")
    torch.manual_seed(config.GENERATION_SEED)
    np.random.seed(config.GENERATION_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from scripts.experiments.wav2lip_face_roi_replacement.generation_worker import encode_video, load_model, render_arm

    plan = read_json(args.plan)
    model = load_model(config.WAV2LIP_CHECKPOINT, "cuda")
    rows: list[dict] = []
    for item in plan.get("rows", []):
        sample_id = str(item["sample_id"])
        static = np.asarray(np.load(Path(str(item["static_face"])), allow_pickle=False))
        if static.shape != (224, 224, 3) or static.dtype != np.uint8:
            raise RuntimeError(f"invalid static face: {sample_id}")
        frames = [np.ascontiguousarray(static.copy()) for _ in range(config.FRAME_COUNT)]
        boxes = [[0, 224, 0, 224] for _ in frames]
        for arm, mel_value in dict(item["arms"]).items():
            output = Path(str(item["outputs"][arm]))
            if output.exists():
                raise RuntimeError(f"output exists: {output}")
            mel = np.asarray(np.load(Path(str(mel_value)), allow_pickle=False), dtype=np.float32)
            generated = render_arm(model, frames, boxes, chunk_mels(mel), config.WAV2LIP_BATCH_SIZE, "cuda")
            encoded = encode_video(generated, output, config.FFMPEG)
            rows.append({"sample_id": sample_id, "arm": str(arm), "video_only": str(output.resolve()), "video_only_sha256": file_sha256(output), "frame_count": len(generated), "raw_frame_sha256": encoded["raw_frame_sha256"], "static_face": str(Path(str(item["static_face"])).resolve()), "static_face_sha256": file_sha256(Path(str(item["static_face"]))), "mel": str(Path(str(mel_value)).resolve()), "mel_sha256": file_sha256(Path(str(mel_value))), "device": "cuda", "batch_size": config.WAV2LIP_BATCH_SIZE})
            print(f"GENERATED {sample_id} {arm}", flush=True)
    write_json(args.result, {"schema_version": 1, "protocol_id": "wav2lip_natural_temporal_contrast", "status": "complete", "rows": rows, "runtime": {"device": "cuda", "device_name": torch.cuda.get_device_name(0), "seed": config.GENERATION_SEED, "deterministic": True, "tf32": False}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
