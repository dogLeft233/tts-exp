from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.protocol import (
    PROTOCOL_ID,
    file_sha256,
    write_json,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.run_face_preflight import (
    FACE_DET_BATCH_SIZE,
    _assert_gpu_ready,
    _detect_all,
    _read_frames,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.run_visual_expanded_render import (
    ALIGNMENT_PATH,
    CANDIDATE_PATH,
    STAGE00_PATH,
    _build_render_manifest,
    _validate_inputs,
)

REPO = Path(__file__).resolve().parents[3]
STAGE_ID = "03_visual_expanded_face_preflight_retry1"
OUTPUT_PATH = (
    REPO
    / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
    / STAGE_ID
)
MIN_FACE_READY_RECORDS = 57
MIN_GROUP_COUNT = 23


def run(
    stage00_path: Path = STAGE00_PATH,
    candidate_path: Path = CANDIDATE_PATH,
    alignment_path: Path = ALIGNMENT_PATH,
    output_dir: Path = OUTPUT_PATH,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty face preflight output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        stage00_path = stage00_path.resolve()
        candidate_path = candidate_path.resolve()
        alignment_path = alignment_path.resolve()
        stage00, candidate, _ = _validate_inputs(stage00_path, candidate_path, alignment_path)
        render_protocol = _build_render_manifest(
            stage00_path,
            candidate_path,
            alignment_path,
            stage00,
            candidate,
        )
        gpu_gate = _assert_gpu_ready()
        import sys

        wav2lip_root = REPO / "third_party/Wav2Lip"
        if str(wav2lip_root) not in sys.path:
            sys.path.insert(0, str(wav2lip_root))
        import face_detection

        detector = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False,
            device="cuda",
        )
        results: list[dict[str, Any]] = []
        try:
            for record in render_protocol["cohort"]["records"]:
                sample_id = str(record["sample_id"])
                face = Path(str(record["face_video"]))
                frames = _read_frames(face)
                failed_frames = _detect_all(detector, frames)
                results.append(
                    {
                        "sample_id": sample_id,
                        "source_group": str(record["source_group"]),
                        "face_video": str(face),
                        "face_video_sha256": str(record["face_video_sha256"]),
                        "frame_count": len(frames),
                        "failed_frame_indices": failed_frames,
                        "face_ready": not failed_frames,
                    }
                )
        finally:
            del detector
        ready_ids = [row["sample_id"] for row in results if row["face_ready"]]
        ready_groups = {row["source_group"] for row in results if row["face_ready"]}
        face_ready = len(ready_ids) >= MIN_FACE_READY_RECORDS and len(ready_groups) == MIN_GROUP_COUNT
        manifest = {
            "schema_version": 1,
            "manifest_type": "lrs3_mfa3_visual_expanded_face_preflight",
            "protocol_id": PROTOCOL_ID,
            "stage_id": STAGE_ID,
            "status": "complete" if face_ready else "blocked",
            "engineering_decision": "GO" if face_ready else "BLOCKED",
            "scientific_decision": "not_available",
            "parents": {
                "stage00": {"path": str(stage00_path), "sha256": file_sha256(stage00_path)},
                "candidate": {"path": str(candidate_path), "sha256": file_sha256(candidate_path)},
                "alignment": {"path": str(alignment_path), "sha256": file_sha256(alignment_path)},
            },
            "input_record_count": len(results),
            "face_ready_record_count": len(ready_ids),
            "face_failed_record_count": len(results) - len(ready_ids),
            "face_ready_sample_ids": ready_ids,
            "face_ready_group_count": len(ready_groups),
            "face_det_batch_size": FACE_DET_BATCH_SIZE,
            "results": results,
            "selection": {
                "rule": "retain all records whose frozen Wav2Lip S3FD detector finds a face in every source-video frame",
                "score_based_selection": False,
                "visual_metric_used_for_selection": False,
                "syncnet_used_for_selection": False,
                "audio_quality_used_for_selection": False,
                "source_order_preserved": True,
            },
            "gpu_gate": gpu_gate,
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
            "face_failed_record_count": len(results) - len(ready_ids),
            "face_ready_group_count": len(ready_groups),
            "minimum_face_ready_record_count": MIN_FACE_READY_RECORDS,
            "minimum_group_count": MIN_GROUP_COUNT,
            "score_based_selection": False,
            "syncnet_used_for_selection": False,
            "sealed_media_opened": False,
        }
        write_json(output_dir / "summary.json", summary)
        write_json(
            output_dir / "decision.json",
            {
                **summary,
                "next_allowed_stage": "03_visual_expanded_wav2lip_render_retry2" if face_ready else None,
                "reason": "frozen all-frame Wav2Lip face preflight completed before render selection",
            },
        )
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
        write_json(output_dir / "summary.json", payload)
        write_json(output_dir / "decision.json", {**payload, "next_allowed_stage": None})
        return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage00", type=Path, default=STAGE00_PATH)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE_PATH)
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    result = run(args.stage00, args.candidate, args.alignment, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["engineering_decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
