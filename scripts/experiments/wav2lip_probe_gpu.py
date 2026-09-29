from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from scripts.experiments import wav2lip_probe_runtime as rt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback")
    if int(torch.cuda.mem_get_info()[0]) < 5 * 1024**3:
        raise RuntimeError("less than 5 GiB of free GPU memory")
    torch.manual_seed(20260909)
    np.random.seed(20260909)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from scripts.experiments.wav2lip_face_roi_replacement.generation_worker import encode_video, file_sha256, load_model, render_arm

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    model = load_model(args.checkpoint, "cuda")
    rows = []
    for item in plan["rows"]:
        static_path = Path(str(item["static_face"]))
        static_face = np.asarray(np.load(static_path, allow_pickle=False))
        if static_face.shape != (224, 224, 3) or static_face.dtype != np.uint8:
            raise RuntimeError(f"invalid static face: {static_path}")
        frames = [np.ascontiguousarray(static_face.copy()) for _ in range(93)]
        boxes = [[0, 224, 0, 224] for _ in range(93)]
        for arm, mel_path in item["arms"].items():
            output = Path(str(item["outputs"][arm]))
            partial = output.with_name(f".{output.name}.{os.getpid()}.partial")
            partial.unlink(missing_ok=True)
            mel = np.asarray(np.load(Path(str(mel_path)), allow_pickle=False), dtype=np.float32)
            chunks = []
            index = 0
            while True:
                start = int(index * 80.0 / 25.0)
                if start + 16 > mel.shape[1]:
                    chunks.append(mel[:, -16:])
                    break
                chunks.append(mel[:, start : start + 16])
                index += 1
            generated = render_arm(model, frames, boxes, chunks, 4, "cuda")
            encoded = encode_video(generated, partial, args.ffmpeg)
            rt.mux_video_audio(partial, Path(str(item["source_audio"])), args.ffmpeg)
            partial.replace(output)
            rows.append({"sample_id": item["sample_id"], "arm": arm, "output": str(output.resolve()), "output_sha256": file_sha256(output), "source_audio": item["source_audio"], "static_face": str(static_path.resolve()), "static_face_sha256": file_sha256(static_path), "mel": str(Path(str(mel_path)).resolve()), "mel_sha256": file_sha256(Path(str(mel_path))), "frame_count": len(generated), "device": "cuda", "batch_size": 4, "raw_frame_sha256": encoded["raw_frame_sha256"]})
    args.result.parent.mkdir(parents=True, exist_ok=True)
    rt.write_json(args.result, {"schema_version": 1, "protocol_id": args.protocol_id, "status": "complete", "runtime": {"device": "cuda", "seed": 20260909, "deterministic": True, "tf32": False, "checkpoint_sha256": file_sha256(args.checkpoint)}, "rows": rows})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
