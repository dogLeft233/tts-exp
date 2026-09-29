from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import config
from .common import ProtocolError, bytes_sha256, file_sha256, run_logged


def read_first_frame(video: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ProtocolError(f"cannot open source video: {video}")
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ProtocolError(f"source video has no decodable frame 0: {video}")
    return np.ascontiguousarray(frame)


def write_png(path: Path, frame: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if existing is None or not np.array_equal(existing, frame):
            raise ProtocolError(f"existing static reference differs: {path}")
    else:
        if not cv2.imwrite(str(path), frame):
            raise ProtocolError(f"failed to write static reference: {path}")
    decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if decoded is None or not np.array_equal(decoded, frame):
        raise ProtocolError(f"PNG round-trip changed reference pixels: {path}")
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return {
        "path": str(path.resolve()),
        "container_sha256": file_sha256(path),
        "rgb_pixel_sha256": bytes_sha256(np.ascontiguousarray(rgb).tobytes()),
        "bgr_pixel_sha256": bytes_sha256(np.ascontiguousarray(decoded).tobytes()),
        "width": int(decoded.shape[1]),
        "height": int(decoded.shape[0]),
        "dtype": str(decoded.dtype),
        "source_frame_index": 0,
    }


def select_detection(detections: list[Mapping[str, Any]], threshold: float = config.FACE_DET_THRESHOLD) -> dict[str, float]:
    eligible = [item for item in detections if float(item["score"]) >= threshold]
    if not eligible:
        raise ProtocolError(f"no face reaches the frozen detector threshold {threshold}")
    eligible.sort(key=lambda item: (-float(item["score"]), float(item["x1"]), float(item["y1"])))
    return {key: float(eligible[0][key]) for key in ("x1", "y1", "x2", "y2", "score")}


def generation_box(selected: Mapping[str, Any], width: int, height: int) -> list[int]:
    x1 = max(0, math.floor(float(selected["x1"])))
    y1 = max(0, math.floor(float(selected["y1"])))
    x2 = min(width, math.ceil(float(selected["x2"])))
    y2 = min(height, math.ceil(float(selected["y2"]) + config.GENERATION_BOTTOM_PAD))
    if x2 <= x1 or y2 <= y1:
        raise ProtocolError("selected face box is empty after fixed bottom padding")
    return [x1, y1, x2, y2]


def score_box(selected: Mapping[str, Any]) -> dict[str, Any]:
    width = float(selected["x2"]) - float(selected["x1"])
    height = float(selected["y2"]) - float(selected["y1"])
    side = max(1, round(1.5 * max(width, height)))
    center_x = (float(selected["x1"]) + float(selected["x2"])) / 2.0
    center_y = (float(selected["y1"]) + float(selected["y2"])) / 2.0
    left = math.floor(center_x - side / 2.0)
    top = math.floor(center_y - side / 2.0)
    return {
        "order": "x1,y1,x2,y2; unbounded square with zero padding",
        "center": [center_x, center_y],
        "side": side,
        "box": [left, top, left + side, top + side],
        "scale_from_detection": 1.5,
    }


def crop_zero_padded(frame: np.ndarray, box: Mapping[str, Any], output_size: int = 224) -> np.ndarray:
    x1, y1, x2, y2 = [int(value) for value in box["box"]]
    side = int(box["side"])
    if x2 - x1 != side or y2 - y1 != side or side <= 0:
        raise ProtocolError("invalid square score box")
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    src_x1, src_y1 = max(0, x1), max(0, y1)
    src_x2, src_y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if src_x2 > src_x1 and src_y2 > src_y1:
        canvas[src_y1 - y1:src_y2 - y1, src_x1 - x1:src_x2 - x1] = frame[src_y1:src_y2, src_x1:src_x2]
    return cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def _detection_request(paths: config.RunPaths, items: list[dict[str, Any]]) -> Path:
    request_path = paths.image_dir / "detection_requests.json"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return request_path


def _run_detection(paths: config.RunPaths, requests: Path) -> dict[str, Any]:
    output = paths.image_dir / "detections.json"
    if output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and len(payload.get("records", [])) == config.EXPECTED_RECORD_COUNT:
            return payload
    log = paths.logs_dir / "face_detection.log"
    command = [str(config.WAV2LIP_PYTHON), str(Path(__file__).with_name("detect_worker.py")), "--requests", str(requests), "--output", str(output), "--threshold", str(config.FACE_DET_THRESHOLD)]
    run_logged(command, config.REPO, log)
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or len(payload.get("records", [])) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("face detection output is incomplete")
    return payload


def prepare_images(paths: config.RunPaths, inputs: Mapping[str, Any]) -> dict[str, Any]:
    paths.image_dir.mkdir(parents=True, exist_ok=True)
    requests: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    for item in inputs["records"]:
        sid = str(item["sample_id"])
        source_video = Path(str(item["face_video"]["path"]))
        frame = read_first_frame(source_video)
        image_path = paths.image_dir / f"{sid}__frame0.png"
        image_meta = write_png(image_path, frame)
        requests.append({"sample_id": sid, "image": str(image_path.resolve()), "image_sha256": image_meta["rgb_pixel_sha256"]})
        copy = dict(item)
        copy["static_reference"] = {"image": image_meta, "source_video_sha256": item["face_video"]["container_sha256"], "source_frame_index": 0}
        enriched.append(copy)

    detection_payload = _run_detection(paths, _detection_request(paths, requests))
    detection_by_id = {str(row["sample_id"]): row for row in detection_payload["records"]}
    if len(detection_by_id) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("face detection keyed join is incomplete")
    final_records: list[dict[str, Any]] = []
    for item in enriched:
        sid = str(item["sample_id"])
        detection = detection_by_id.get(sid)
        if detection is None:
            raise ProtocolError(f"face detection missing: {sid}")
        selected = select_detection(detection["detections"])
        image_path = Path(str(item["static_reference"]["image"]["path"]))
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ProtocolError(f"static reference disappeared: {sid}")
        generated = generation_box(selected, int(frame.shape[1]), int(frame.shape[0]))
        scored = score_box(selected)
        crop = crop_zero_padded(frame, scored)
        crop_path = paths.image_dir / f"{sid}__score_crop.png"
        crop_meta = write_png(crop_path, crop)
        static = dict(item["static_reference"])
        static.update({
            "detection": {"threshold": config.FACE_DET_THRESHOLD, "all": detection["detections"], "selected": selected},
            "generation_box_xyxy": generated,
            "score_box": scored,
            "score_crop": crop_meta,
            "input_mode": "one_png_only",
            "dynamic_source_frames_forbidden": True,
        })
        copy = dict(item)
        copy["static_reference"] = static
        final_records.append(copy)
    return {**dict(inputs), "records": final_records, "static_reference_contract": "source video frame 0 decoded once; all arms consume the same PNG and fixed boxes"}


def validate_static_inputs(inputs: Mapping[str, Any]) -> None:
    if len(inputs.get("records", [])) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("static input record count is not 22")
    for item in inputs["records"]:
        static = item.get("static_reference")
        if not isinstance(static, Mapping) or static.get("source_frame_index") != 0 or static.get("input_mode") != "one_png_only":
            raise ProtocolError(f"static reference contract missing: {item.get('sample_id')}")
        image = Path(str(static["image"]["path"]))
        if file_sha256(image) != str(static["image"]["container_sha256"]):
            raise ProtocolError(f"static PNG hash changed: {image}")
        if not Path(str(static["score_crop"]["path"])).is_file():
            raise ProtocolError(f"fixed score crop is missing: {item.get('sample_id')}")
        generation = static.get("generation_box_xyxy")
        if not isinstance(generation, list) or len(generation) != 4:
            raise ProtocolError(f"generation box is invalid: {item.get('sample_id')}")
