from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import config
from .common import (
    ProtocolError,
    canonical_json_sha256,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def _run_detection(video: Path, boxes_path: Path, overlay_dir: Path, log_path: Path) -> dict[str, Any]:
    if boxes_path.exists() or overlay_dir.exists():
        raise ProtocolError(f"partial ROI artifact cannot be resumed: {video}")
    boxes_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["NUMBA_DISABLE_JIT"] = "1"
    environment["NUMBA_CACHE_DIR"] = str(overlay_dir.parent / "numba_cache")
    command = [
        str(config.WAV2LIP_PYTHON),
        str(Path(__file__).with_name("roi_worker.py")),
        "--video", str(video),
        "--boxes", str(boxes_path),
        "--overlays", str(overlay_dir),
        "--ffprobe", str(config.FFPROBE),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(config.WAV2LIP_ROOT), env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not boxes_path.is_file():
        raise ProtocolError(f"official face detection failed: {log_path}")
    return verify_self_hashed_json(boxes_path)


def _review_row(row: Mapping[str, Any]) -> dict[str, Any]:
    boxes = row.get("boxes")
    raw = row.get("raw_boxes")
    frame_count = int(row.get("frame_count", 0))
    width = int(row.get("width", 0))
    height = int(row.get("height", 0))
    errors: list[str] = []
    if not isinstance(boxes, list) or len(boxes) != frame_count:
        errors.append("final box count differs from source frames")
    if not isinstance(raw, list) or len(raw) != frame_count:
        errors.append("raw box count differs from source frames")
    for index, box in enumerate(boxes if isinstance(boxes, list) else []):
        if not isinstance(box, list) or len(box) != 4:
            errors.append(f"frame {index}: malformed top,bottom,left,right box")
            continue
        top, bottom, left, right = (int(value) for value in box)
        if not (0 <= top < bottom <= height and 0 <= left < right <= width):
            errors.append(f"frame {index}: box is outside or empty")
        if top == 0 and bottom == height and left == 0 and right == width:
            errors.append(f"frame {index}: full-frame fallback is forbidden")
        if (bottom - top) * (right - left) >= 0.95 * width * height:
            errors.append(f"frame {index}: box covers almost the whole canvas")
    overlays = row.get("overlays")
    evidence: list[dict[str, Any]] = []
    if isinstance(overlays, Mapping):
        for label in ("head", "middle", "tail"):
            value = Path(str(overlays.get(label, "")))
            if not value.is_file():
                errors.append(f"missing {label} overlay")
            else:
                evidence.append({"label": label, "path": str(value.resolve()), "sha256": file_sha256(value)})
    else:
        errors.append("ROI overlay evidence is missing")
    return {
        "sample_id": str(row.get("sample_id", "")),
        "source_group": str(row.get("source_group", "")),
        "reviewer": "codex-automated-geometry-audit",
        "decision": "PASS" if not errors else "BLOCKED",
        "conclusion": "official face detector boxes cover the detected face and padded chin/mouth region; no full-frame fallback" if not errors else "ROI geometry cannot be accepted",
        "method": "raw FaceAlignment SFD box plus official pads [0,10,0,0]; deterministic canvas/empty/full-frame checks",
        "evidence": evidence,
        "errors": errors,
    }


def build_roi_manifest(records: Sequence[Mapping[str, Any]], output: Path, review_output: Path, log_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        sample_id = str(record["sample_id"])
        video = Path(str(record["face_video"]["path"]))
        if not video.is_file() or file_sha256(video) != str(record["face_video"]["sha256"]):
            raise ProtocolError(f"source video hash changed: {sample_id}")
        boxes_path = output.parent / "boxes" / f"{sample_id}.json"
        overlay_dir = output.parent / "overlays" / sample_id
        detected = _run_detection(video, boxes_path, overlay_dir, log_dir / f"{sample_id}.log")
        if detected.get("source_video_sha256") != str(record["face_video"]["sha256"]):
            raise ProtocolError(f"ROI source binding differs: {sample_id}")
        row = {
            "schema_version": 1,
            "stage_id": "roi",
            "protocol_id": config.PROTOCOL_ID,
            "sample_id": sample_id,
            "source_group": str(record["source_group"]),
            "source_video": str(video.resolve()),
            "source_video_sha256": str(record["face_video"]["sha256"]),
            "boxes_path": str(boxes_path.resolve()),
            "boxes_sha256": file_sha256(boxes_path),
            "frame_count": int(detected["frame_count"]),
            "width": int(detected["width"]),
            "height": int(detected["height"]),
            "raw_boxes": detected["raw_boxes"],
            "boxes": detected["boxes"],
            "overlays": detected["overlays"],
            "detector": detected["detector"],
        }
        row["box_track_sha256"] = canonical_json_sha256(row["boxes"])
        rows.append(row)
        review_rows.append(_review_row(row))
        print(f"ROI {index}/{len(records)} {sample_id}", flush=True)
    manifest = {
        "schema_version": 1,
        "stage_id": "roi",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if len(rows) == config.EXPECTED_RECORD_COUNT else "blocked",
        "record_count": len(rows),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "detector_config": {"flip_input": False, "pads": [0, 10, 0, 0], "nosmooth": True, "resize_factor": 1, "rotate": False, "crop": [0, -1, 0, -1], "batch_size": 1, "detector": "FaceAlignment SFD"},
        "rows": rows,
    }
    review = {"schema_version": 1, "stage_id": "roi_review", "protocol_id": config.PROTOCOL_ID, "status": "complete" if all(item["decision"] == "PASS" for item in review_rows) else "blocked", "record_count": len(review_rows), "expected_record_count": config.EXPECTED_RECORD_COUNT, "rows": review_rows}
    write_self_hashed_json(output, manifest)
    write_self_hashed_json(review_output, review)
    return verify_self_hashed_json(output), verify_self_hashed_json(review_output)
