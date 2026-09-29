from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2

WAV2LIP_ROOT = Path(__file__).resolve().parents[3] / "third_party/Wav2Lip"
sys.path.insert(0, str(WAV2LIP_ROOT))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()

    import torch
    from face_detection.detection.sfd import FaceDetector

    requests = json.loads(args.requests.read_text(encoding="utf-8"))
    if not isinstance(requests, list):
        raise TypeError("requests must be a list")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("static image face detection requires CUDA")
    detector = FaceDetector(device=device, verbose=False)
    records = []
    for request in requests:
        path = Path(str(request["image"])).resolve()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode reference image: {path}")
        detections = detector.detect_from_image(image)
        candidates = []
        for detection in detections:
            values = [float(value) for value in detection]
            if len(values) != 5:
                continue
            candidates.append({"x1": values[0], "y1": values[1], "x2": values[2], "y2": values[3], "score": values[4]})
        eligible = [item for item in candidates if item["score"] >= float(args.threshold)]
        eligible.sort(key=lambda item: (-item["score"], item["x1"], item["y1"]))
        if not eligible:
            raise RuntimeError(f"no face reaches threshold {args.threshold}: {path}")
        records.append({
            "sample_id": str(request["sample_id"]),
            "image": str(path),
            "image_sha256": sha256_bytes(image[..., ::-1].copy().tobytes()),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "detections": candidates,
            "selected": eligible[0],
            "threshold": float(args.threshold),
            "device": device,
        })
    payload = {"schema_version": 1, "status": "complete", "threshold": float(args.threshold), "device": device, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
