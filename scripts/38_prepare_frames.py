#!/usr/bin/env python3
"""38_prepare_frames.py - Quality-filtered master reference frames for TFG.

For each video in the manifest, sample candidate frames, filter by face
quality (yaw / face size / sharpness), and write the best frame at ORIGINAL
resolution - no upscaling, no AI super-resolution, no forced 512x512.

Output: runs/<run_id>/00_frames/{i}.png + frame_manifest.json

Detectors (pluggable via extract_metrics_for_frame):
  - "stub"        : caller-provided metrics (tests / remote override)
  - "haar"        : OpenCV Haar cascade (face area + sharpness; yaw unknown -> 0)
  - "mediapipe"   : MediaPipe FaceMesh (real yaw / mouth / eye metrics)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from frame_quality import FaceMetrics, compute_blur_score, select_best_frame

N_CANDIDATES = 12
MAX_YAW_DEG = 30.0
MIN_FACE_AREA_PX = 20000.0


def first_acceptable_frame(
    candidates: list[tuple[int, float, np.ndarray]],
    extract_metrics,
    max_yaw_deg: float = MAX_YAW_DEG,
    min_face_area_px: float = MIN_FACE_AREA_PX,
) -> tuple[int, FaceMetrics] | None:
    """Prefer the first frame that passes the quality gate.

    Empirically (HDTF paired test) Ditto generates better lip sync from the
    video's first frame than from an arbitrary mid-video quality-selected
    frame: mid-video frames carry speaking-state mouth/eye pose that the
    model treats as the initial motion state.
    """
    for idx, (_, _, frame) in enumerate(candidates):
        m = extract_metrics(frame)
        if m is None:
            continue
        if abs(m.yaw_deg) <= max_yaw_deg and m.face_area_px >= min_face_area_px:
            return idx, m
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_metrics_for_frame(frame) -> FaceMetrics | None:
    """Default detector: OpenCV Haar cascade.

    Haar boxes carry no pose estimate, so yaw is reported as 0.0 (no
    penalty). The detector name records this limitation; remote runs may
    swap in a landmark-based detector via this hook.
    """
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    boxes = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
    x, y, w, h = boxes[0]
    return FaceMetrics(
        yaw_deg=0.0,
        face_area_px=int(w * h),
        blur_score=compute_blur_score(gray),
    )


def _make_detector(detector_name: str):
    """Return the extract_metrics callable for the requested detector."""
    if detector_name in ("haar", "stub"):
        return extract_metrics_for_frame
    if detector_name == "mediapipe":
        from frame_detectors import mediapipe_extract_metrics

        return mediapipe_extract_metrics
    raise ValueError(f"unknown detector: {detector_name}")


def _sample_candidates(video_path: Path, n: int) -> list[tuple[int, float, np.ndarray]]:
    """Uniformly sample n frames as (index, timestamp_s, bgr_frame)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = total / fps if total > 0 else 0.0
    sampled: list[tuple[int, float, np.ndarray]] = []
    try:
        if total <= 0:
            return sampled
        positions = np.linspace(0.0, 0.98 * duration, n)
        for timestamp_s in positions:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame_idx = int(round(timestamp_s * fps))
            sampled.append((frame_idx, float(timestamp_s), frame))
    finally:
        cap.release()
    return sampled


def prepare_frames(
    repo: Path,
    run_id: str,
    video_manifest_path: Path,
    count: int,
    detector_name: str,
    n_candidates: int = N_CANDIDATES,
    frame_strategy: str = "first",
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict:
    payload = json.loads(video_manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if len(records) < count:
        raise ValueError(f"video manifest has fewer than {count} candidate records")

    out_dir = repo / "runs" / run_id / "00_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    extract_metrics = _make_detector(detector_name)
    frame_strategy = frame_strategy or "first"

    prepared: list[dict] = []
    rejected: list[dict] = []
    for record in records:
        if len(prepared) >= count:
            break
        sample_id = len(prepared) + 1
        video_path = Path(record["video_local_path"])
        if not video_path.is_absolute():
            video_path = repo / video_path
        if not video_path.exists():
            rejected.append(
                {"attempted_sample_id": sample_id, "stem": record["stem"], "error": "video missing"}
            )
            continue
        try:
            candidates = _sample_candidates(video_path, n_candidates)
            if not candidates:
                raise ValueError("no decodable frames")

            if frame_strategy == "first":
                selected = first_acceptable_frame(candidates, extract_metrics)
                if selected is None:
                    selected = select_best_frame(
                        candidates,
                        lambda c: extract_metrics(c[2]),
                        max_yaw_deg=MAX_YAW_DEG,
                        min_face_area_px=MIN_FACE_AREA_PX,
                    )
            else:
                selected = select_best_frame(
                    candidates,
                    lambda c: extract_metrics(c[2]),
                    max_yaw_deg=MAX_YAW_DEG,
                    min_face_area_px=MIN_FACE_AREA_PX,
                )
            if selected is None:
                raise ValueError("no acceptable frame (no face / profile / too small)")

            frame_idx, timestamp_s, best_frame = candidates[selected[0]]
            metrics = selected[1]
            destination = out_dir / f"{sample_id}.png"
            if destination.exists() and not overwrite:
                import cv2

                cached = cv2.imread(str(destination))
                if cached is None:
                    raise ValueError(f"corrupt cached frame: {destination}")
                h, w = cached.shape[:2]
                write_frame = False
            else:
                import cv2

                cv2.imwrite(str(destination), best_frame)
                h, w = best_frame.shape[:2]
                write_frame = True
            prepared.append(
                {
                    "sample_id": sample_id,
                    "stem": record["stem"],
                    "detector": detector_name,
                    "source_video": str(video_path),
                    "frame_idx": frame_idx,
                    "timestamp_s": round(timestamp_s, 4),
                    "sha256": sha256_file(destination),
                    "source_resolution": [w, h],
                    "output_resolution": [w, h],
                    "yaw_deg": round(metrics.yaw_deg, 3),
                    "face_area_px": int(metrics.face_area_px),
                    "blur_score": round(metrics.blur_score, 3),
                    "mouth_open_ratio": (
                        round(metrics.mouth_open_ratio, 4)
                        if metrics.mouth_open_ratio is not None
                        else None
                    ),
                    "eye_open": (
                        round(metrics.eye_open, 4) if metrics.eye_open is not None else None
                    ),
                    "cached": not write_frame,
                }
            )
            action = "Cached" if not write_frame else "Prepared"
            print(f"{action} sample {sample_id}: {record['stem']} frame {frame_idx}")
        except Exception as exc:
            rejected.append(
                {
                    "attempted_sample_id": sample_id,
                    "stem": record["stem"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"Rejected candidate {record['stem']}: {rejected[-1]['error']}")

    manifest = {
        "schema_version": 1,
        "detector": detector_name,
        "n_candidates": n_candidates,
        "max_yaw_deg": MAX_YAW_DEG,
        "min_face_area_px": MIN_FACE_AREA_PX,
        "samples_total": min(len(prepared) + len(rejected), count),
        "samples_ok": len(prepared),
        "rejected_candidates": rejected,
        "records": prepared,
    }
    (out_dir / "frame_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--video-manifest", required=True, type=Path)
    parser.add_argument("--count", type=int, default=13)
    parser.add_argument("--detector", default="haar", choices=["haar", "mediapipe", "stub"])
    parser.add_argument("--frame-strategy", default="first", choices=["first", "quality"])
    parser.add_argument("--n-candidates", type=int, default=N_CANDIDATES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    result = prepare_frames(
        repo=repo,
        run_id=args.run_id,
        video_manifest_path=args.video_manifest,
        count=args.count,
        detector_name=args.detector,
        n_candidates=args.n_candidates,
        frame_strategy=args.frame_strategy,
        overwrite=args.overwrite,
    )
    if result["samples_ok"] < args.count:
        print(
            f"[prepare_frames] prepared {result['samples_ok']}/{args.count}; "
            f"rejected: {json.dumps(result['rejected_candidates'], ensure_ascii=False)}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
