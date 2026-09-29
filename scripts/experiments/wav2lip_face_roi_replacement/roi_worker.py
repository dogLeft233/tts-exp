from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
ROOT = Path(__file__).resolve().parents[3] / "third_party/Wav2Lip"
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from face_detection import FaceAlignment, LandmarksType


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--boxes", required=True)
    parser.add_argument("--overlays", required=True)
    parser.add_argument("--ffprobe", required=False)
    args = parser.parse_args()
    video_path = Path(args.video)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open source video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise RuntimeError(f"invalid source frame: {video_path}")
            frames.append(np.ascontiguousarray(frame))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"source video has no frames: {video_path}")
    height, width = frames[0].shape[:2]
    if any(frame.shape != frames[0].shape for frame in frames):
        raise RuntimeError("source frame dimensions change")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector = FaceAlignment(LandmarksType._2D, flip_input=False, device=device, face_detector="sfd", verbose=False)
    raw_boxes: list[list[int]] = []
    boxes: list[list[int]] = []
    for index, frame in enumerate(frames):
        predictions = detector.get_detections_for_batch(np.asarray([frame]))
        rect = predictions[0] if predictions else None
        if rect is None:
            raise RuntimeError(f"face not detected at frame {index}")
        x1, y1, x2, y2 = (int(value) for value in rect)
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError(f"empty raw face box at frame {index}: {rect}")
        top = max(0, y1)
        bottom = min(height, y2 + 10)
        left = max(0, x1)
        right = min(width, x2)
        if not (0 <= top < bottom <= height and 0 <= left < right <= width):
            raise RuntimeError(f"empty padded face box at frame {index}: {(top, bottom, left, right)}")
        if top == 0 and bottom == height and left == 0 and right == width:
            raise RuntimeError(f"full-frame face box at frame {index}")
        raw_boxes.append([x1, y1, x2, y2])
        boxes.append([top, bottom, left, right])
    overlays = {}
    overlay_dir = Path(args.overlays)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    indices = {"head": 0, "middle": len(frames) // 2, "tail": len(frames) - 1}
    for label, index in indices.items():
        image = frames[index].copy()
        top, bottom, left, right = boxes[index]
        cv2.rectangle(image, (left, top), (right - 1, bottom - 1), (0, 255, 0), 2)
        path = overlay_dir / f"{label}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"cannot write overlay: {path}")
        overlays[label] = str(path.resolve())
    detector_files = [
        ROOT / "face_detection/api.py",
        ROOT / "face_detection/models.py",
        ROOT / "face_detection/detection/sfd/sfd_detector.py",
        ROOT / "face_detection/detection/sfd/net_s3fd.py",
        ROOT / "face_detection/detection/sfd/s3fd.pth",
    ]
    detector_bindings = {str(path.relative_to(ROOT)): {"path": str(path.resolve()), "sha256": file_sha256(path)} for path in detector_files if path.is_file()}
    body = {
        "schema_version": 1,
        "stage_id": "roi_boxes",
        "protocol_id": "wav2lip_face_roi_replacement",
        "source_video": str(video_path.resolve()),
        "source_video_sha256": file_sha256(video_path),
        "frame_count": len(frames),
        "width": width,
        "height": height,
        "raw_box_order": ["left", "top", "right", "bottom"],
        "box_order": ["top", "bottom", "left", "right"],
        "raw_boxes": raw_boxes,
        "boxes": boxes,
        "box_track_sha256": canonical_hash(boxes),
        "overlays": overlays,
        "detector": {"name": "FaceAlignment SFD", "flip_input": False, "pads": [0, 10, 0, 0], "resize_factor": 1, "rotate": False, "nosmooth": True, "batch_size": 1, "device": device, "bindings": detector_bindings},
    }
    body["artifact_sha256"] = canonical_hash({key: value for key, value in body.items()})
    output = Path(args.boxes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
