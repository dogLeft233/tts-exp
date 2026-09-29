from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import config
from .common import file_sha256, load_self_hashed, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    torch.manual_seed(config.GENERATION_SEED)
    np.random.seed(config.GENERATION_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from scripts.experiments.wav2lip_face_roi_replacement.generation_worker import (
        encode_video,
        load_model,
        render_arm,
    )

    plan = load_self_hashed(args.plan)
    model = load_model(config.WAV2LIP_CHECKPOINT, "cuda")
    rows = []
    for item in plan.get("rows", []):
        sid = str(item["sample_id"])
        face_path = Path(str(item["static_face"]))
        face = np.asarray(np.load(face_path, allow_pickle=False))
        if face.shape != (224, 224, 3) or face.dtype != np.uint8:
            raise RuntimeError(f"bad static face: {sid}")
        frames = [np.ascontiguousarray(face.copy()) for _ in range(config.FRAME_COUNT)]
        boxes = [[0, 224, 0, 224] for _ in frames]
        for arm, mel_path in dict(item["arms"]).items():
            output = Path(str(item["outputs"][arm]))
            mel = np.asarray(
                np.load(Path(str(mel_path)), allow_pickle=False), dtype=np.float32
            )
            from scripts.experiments.masked_tts_tfg_probe.direct_mel import chunk_mels

            chunks = chunk_mels(mel, config.FPS)
            if len(chunks) != config.FRAME_COUNT:
                raise RuntimeError(f"chunk count changed: {sid}")
            generated = render_arm(
                model, frames, boxes, chunks, config.WAV2LIP_BATCH_SIZE, "cuda"
            )
            encoded = encode_video(generated, output, config.FFMPEG)
            rows.append(
                {
                    "sample_id": sid,
                    "arm": str(arm),
                    "video_only": str(output.resolve()),
                    "video_only_sha256": file_sha256(output),
                    "frame_count": len(generated),
                    "raw_frame_sha256": encoded["raw_frame_sha256"],
                    "static_face": str(face_path.resolve()),
                    "static_face_sha256": file_sha256(face_path),
                    "mel": str(Path(str(mel_path)).resolve()),
                    "mel_sha256": file_sha256(Path(str(mel_path))),
                }
            )
            print(f"GENERATED {sid} {arm}", flush=True)
    write_json(
        args.result,
        {
            "schema_version": 1,
            "protocol_id": "wav2lip_reference_conditioning_interaction",
            "status": "complete",
            "rows": rows,
            "runtime": {
                "device": "cuda",
                "seed": config.GENERATION_SEED,
                "deterministic": True,
                "tf32": False,
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
