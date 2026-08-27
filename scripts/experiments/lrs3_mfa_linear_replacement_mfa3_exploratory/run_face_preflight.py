from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "third_party/Wav2Lip"))
import face_detection

from .run_strict_replacement import (
    STAGE_ID as PARENT_STAGE_ID,
    _assert_gpu_ready,
    build_protocol_manifest,
    file_sha256,
)
from .protocol import (
    CANONICAL_CANDIDATE_PATH,
    CANONICAL_STAGE00_PATH,
    PROTOCOL_ID,
    write_json,
)

STAGE_ID = "03_face_preflight_retry1"
FACE_DET_BATCH_SIZE = 4
MIN_FACE_READY_RECORDS = 24


def _read_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open face video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"face video contains no frames: {path}")
    return frames


def _detect_all(detector: Any, frames: list[np.ndarray]) -> list[int]:
    batch_size = FACE_DET_BATCH_SIZE
    while True:
        try:
            detections: list[Any] = []
            for start in range(0, len(frames), batch_size):
                detections.extend(detector.get_detections_for_batch(np.asarray(frames[start:start + batch_size])))
            return [index for index, detection in enumerate(detections) if detection is None]
        except RuntimeError:
            if batch_size == 1:
                raise
            batch_size //= 2


def run(stage00_path: Path, candidate_path: Path, output_dir: Path) -> dict[str, Any]:
    stage00_path = stage00_path.resolve()
    candidate_path = candidate_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty face preflight output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        protocol = build_protocol_manifest(stage00_path, candidate_path)
        _assert_gpu_ready()
        detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False,
            device="cuda",
        )
        results: list[dict[str, Any]] = []
        for record in protocol["cohort"]["records"]:
            face = Path(str(record["face_video"]))
            frames = _read_frames(face)
            failed_frames = _detect_all(detector, frames)
            results.append({
                "sample_id": str(record["sample_id"]),
                "source_group": str(record["source_group"]),
                "face_video": str(face),
                "face_video_sha256": str(record["face_video_sha256"]),
                "frame_count": len(frames),
                "failed_frame_indices": failed_frames,
                "face_ready": not failed_frames,
            })
        del detector
        ready_ids = [row["sample_id"] for row in results if row["face_ready"]]
        failed = [row for row in results if not row["face_ready"]]
        manifest = {
            "schema_version": 1,
            "manifest_type": "lrs3_mfa3_exploratory_wav2lip_face_preflight",
            "protocol_id": PROTOCOL_ID,
            "stage_id": STAGE_ID,
            "status": "complete" if len(ready_ids) >= MIN_FACE_READY_RECORDS else "blocked",
            "engineering_decision": "GO" if len(ready_ids) >= MIN_FACE_READY_RECORDS else "BLOCKED",
            "scientific_decision": "not_available",
            "parent_stage_id": PARENT_STAGE_ID,
            "parent_stage_manifest_sha256": file_sha256(output_dir.parent / "03_strict_replacement_retry3/protocol_manifest.json") if (output_dir.parent / "03_strict_replacement_retry3/protocol_manifest.json").is_file() else None,
            "parents": {
                "stage00": {"path": str(stage00_path), "sha256": file_sha256(stage00_path)},
                "candidate": {"path": str(candidate_path), "sha256": file_sha256(candidate_path)},
            },
            "selection": {
                "rule": "retain all canonical candidate records whose frozen Wav2Lip S3FD detector finds a face in every source-video frame",
                "score_based_selection": False,
                "syncnet_used_for_selection": False,
                "audio_or_candidate_quality_used_for_selection": False,
                "source_order_preserved": True,
            },
            "input_record_count": len(results),
            "face_ready_record_count": len(ready_ids),
            "face_failed_record_count": len(failed),
            "face_ready_sample_ids": ready_ids,
            "results": results,
            "media_access": {
                "fit_media_opened": True,
                "internal_dev_media_opened": False,
                "validation_media_opened": False,
                "test_media_opened": False,
            },
        }
        write_json(output_dir / "face_preflight_manifest.json", manifest)
        summary = {
            "schema_version": 1,
            "stage_id": STAGE_ID,
            "protocol_id": PROTOCOL_ID,
            "status": manifest["status"],
            "engineering_decision": manifest["engineering_decision"],
            "scientific_decision": "not_available",
            "input_record_count": len(results),
            "face_ready_record_count": len(ready_ids),
            "face_failed_record_count": len(failed),
            "face_failed_sample_ids": [row["sample_id"] for row in failed],
            "score_based_selection": False,
            "syncnet_used_for_selection": False,
            "sealed_media_opened": False,
        }
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "decision.json", {
            "stage_id": STAGE_ID,
            "engineering_decision": manifest["engineering_decision"],
            "scientific_decision": "not_available",
            "next_allowed_stage": "03_strict_replacement_face_ready" if manifest["engineering_decision"] == "GO" else None,
            "reason": "deterministic frozen Wav2Lip face preflight completed before strict replacement selection",
        })
        return summary
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "stage_id": STAGE_ID,
            "protocol_id": PROTOCOL_ID,
            "status": "blocked",
            "engineering_decision": "BLOCKED",
            "scientific_decision": "not_available",
            "error": str(exc),
        }
        write_json(output_dir / "failure.json", payload)
        write_json(output_dir / "summary.json", {
            "schema_version": 1,
            "stage_id": STAGE_ID,
            "protocol_id": PROTOCOL_ID,
            "status": "blocked",
            "engineering_decision": "BLOCKED",
            "scientific_decision": "not_available",
            "sealed_media_opened": False,
        })
        write_json(output_dir / "decision.json", {
            "stage_id": STAGE_ID,
            "engineering_decision": "BLOCKED",
            "scientific_decision": "not_available",
            "next_allowed_stage": None,
            "reason": str(exc),
        })
        return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage00", type=Path, default=CANONICAL_STAGE00_PATH)
    parser.add_argument("--candidate", type=Path, default=CANONICAL_CANDIDATE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.stage00, args.candidate, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["engineering_decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
