"""CPU MediaPipe visual audit for the frozen fresh-source pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import (
    GROUP_ID_RE,
    REPO_ROOT,
    VISUAL_METHOD,
    file_sha256,
    inspect_clip,
    load_self_hashed,
    write_json,
)

MIN_VALID_FRACTION = 0.95
MIN_EYE_DISTANCE_PX = 40.0
MIN_FRAMES = 140
FRAME_COUNT = 140
FPS = 25.0
MODEL_RELATIVE_PATH = Path("checkpoints/mediapipe/face_landmarker.task")
DATA_ROOT_RELATIVE = Path("data/dataset_samples/lrs3/pretrain")


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _review_index(review_path: Path | None) -> dict[str, dict[str, Any]]:
    if review_path is None or not review_path.is_file():
        return {}
    value = json.loads(review_path.read_text(encoding="utf-8"))
    raw = value.get("reviews", value) if isinstance(value, dict) else {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): item for key, item in raw.items() if isinstance(item, dict)}


def _review_for(index: dict[str, dict[str, Any]], clip: dict[str, Any]) -> dict[str, Any]:
    keys = [str(clip.get("mp4_sha256", "")), str(clip.get("clip_id", "")), str(clip.get("mp4_path", ""))]
    for key in keys:
        if key and key in index:
            return dict(index[key])
    item = clip.get("input_review")
    return dict(item) if isinstance(item, dict) else {}


def _save_preview(video_path: Path, preview_path: Path) -> str | None:
    """Save a small first-frame preview and return its SHA, if decodable."""

    try:
        import cv2

        capture = cv2.VideoCapture(str(video_path))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            return None
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(preview_path), frame):
            return None
        return file_sha256(preview_path)
    except (ImportError, OSError, ValueError):
        return None


def _points_from_sequence(sequence: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if hasattr(sequence, "landmarks"):
        points = np.asarray(sequence.landmarks, dtype=np.float64)
        valid = np.asarray(sequence.valid, dtype=bool)
        timestamps = np.asarray(getattr(sequence, "timestamps_s", np.arange(points.shape[0], dtype=np.float64) / FPS), dtype=np.float64)
        metadata = dict(getattr(sequence, "metadata", {}) or {})
    elif isinstance(sequence, dict):
        points = np.asarray(sequence.get("landmarks"), dtype=np.float64)
        valid = np.asarray(sequence.get("valid"), dtype=bool)
        timestamps = np.asarray(sequence.get("timestamps_s", np.arange(points.shape[0], dtype=np.float64) / FPS), dtype=np.float64)
        metadata = dict(sequence.get("metadata", {}) or {})
    else:
        raise ValueError("landmarker did not return a VisualSequence-compatible object")
    if points.ndim != 3 or points.shape[0] < FRAME_COUNT or points.shape[1] < 264 or points.shape[2] != 2:
        raise ValueError("visual audit needs at least 140 frames of 2-D landmarks")
    if valid.ndim != 1 or valid.shape[0] != points.shape[0]:
        raise ValueError("visual audit valid array shape mismatch")
    if timestamps.shape != (points.shape[0],) or not np.isfinite(timestamps).all() or (timestamps.size > 1 and not np.all(np.diff(timestamps) > 0)):
        raise ValueError("visual audit timestamp array is invalid")
    return points, valid, timestamps, metadata


def _face_measurements(points: np.ndarray, valid: np.ndarray, width: int, height: int) -> dict[str, Any]:
    first = np.asarray(points[0], dtype=np.float64)
    finite = np.isfinite(first).all(axis=1)
    first_frame_face = bool(valid[0] and finite[33] and finite[263])
    eye_distance = 0.0
    face_box: list[float] | None = None
    if first_frame_face:
        left = first[33]
        right = first[263]
        # MediaPipe Face Landmarker returns normalized coordinates.  Keep the
        # conversion explicit in the artifact so a validator can recompute it.
        scale = np.array([float(width), float(height)], dtype=np.float64)
        distance = np.linalg.norm((right - left) * scale)
        eye_distance = float(distance) if np.isfinite(distance) else 0.0
        cloud = first[finite] * scale
        if cloud.size:
            x0, y0 = cloud.min(axis=0)
            x1, y1 = cloud.max(axis=0)
            face_box = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
    return {
        "first_frame_face": first_frame_face,
        "eye_distance_px": eye_distance,
        "face_box": face_box,
    }


def _write_review(run_root: Path, group: str, clip: dict[str, Any], review: dict[str, Any], preview_path: Path | None) -> tuple[dict[str, Any], Path]:
    path_value = review.get("path")
    if path_value:
        path = Path(str(path_value)).resolve()
        if path.is_file():
            return review, path
    path = run_root / "acquisition" / "input_reviews" / f"{group}_{clip['clip_id']}.json"
    payload = {
        "schema_version": 1,
        "source_group": group,
        "clip_id": str(clip["clip_id"]),
        "clip_sha256": clip.get("mp4_sha256"),
        "status": str(review.get("status", "PENDING")),
        "reviewer": str(review.get("reviewer", "unattended_visual_audit")),
        "single_speaker": review.get("single_speaker"),
        "no_dubbing": review.get("no_dubbing"),
        "mouth_visible": review.get("mouth_visible"),
        "transcript_complete": review.get("transcript_complete"),
        "evidence_path": str(preview_path.resolve()) if preview_path else None,
        "evidence_sha256": file_sha256(preview_path) if preview_path and preview_path.is_file() else None,
        "rationale": str(review.get("rationale", "No independent input review supplied; admission remains pending.")),
    }
    return write_json(path, payload), path


def _save_arrays(path: Path, points: np.ndarray, valid: np.ndarray, timestamps: np.ndarray, metadata: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, landmarks=points, valid=valid, timestamps_s=timestamps, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_))
    return file_sha256(path)


def _audit_clip(run_root: Path, group: str, clip: dict[str, Any], model_path: Path, review_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = Path(str(clip.get("mp4_path", ""))).resolve()
    base = {
        "source_group": group,
        "clip_id": str(clip.get("clip_id", source.stem)),
        "clip_path": str(source),
        "clip_sha256": clip.get("mp4_sha256"),
        "status": "FAIL",
    }
    if not source.is_file():
        return dict(base, status="ERROR", reason="missing clip")
    try:
        actual_sha = file_sha256(source)
        if clip.get("mp4_sha256") != actual_sha:
            return dict(base, status="ERROR", reason="pool clip SHA mismatch", actual_clip_sha256=actual_sha)
        media = inspect_clip(source)
        base.update({"duration_s": media.get("duration_s"), "decoded_pcm_samples": media.get("decoded_pcm_samples"), "pts_ok": media.get("pts_ok"), "width": media.get("width"), "height": media.get("height")})
        preview_path = run_root / "acquisition" / "previews" / f"{group}_{clip['clip_id']}.jpg"
        preview_sha = _save_preview(source, preview_path)
        review, review_path = _write_review(run_root, group, clip, _review_for(review_index, clip), preview_path if preview_sha else None)
        base.update({"preview_path": str(preview_path.resolve()) if preview_sha else None, "preview_sha256": preview_sha, "input_review_path": str(review_path.resolve()), "input_review_sha256": file_sha256(review_path), "input_review_status": review.get("status", "PENDING"), "reviewer": review.get("reviewer")})
        if not (media.get("has_video") and media.get("has_audio") and media.get("pts_ok") and np.isfinite(float(media.get("duration_s", float("nan")))) and 6.0 <= float(media.get("duration_s", 0.0)) <= 10.0 and 96000 <= int(media.get("decoded_pcm_samples", 0)) <= 160000):
            return dict(base, reason="media_duration_pts_or_audio_gate_failed")
        from scripts.experiments.lrs3_tts_visual_advantage.video_features import (
            extract_video_features,
        )

        sequence = extract_video_features(source, model_path)
        points, valid, timestamps, metadata = _points_from_sequence(sequence)
        width = int(media.get("width") or metadata.get("width") or 0)
        height = int(media.get("height") or metadata.get("height") or 0)
        measurements = _face_measurements(points, valid, width, height)
        first_valid = valid[:FRAME_COUNT]
        valid_count = int(first_valid.sum())
        valid_fraction = float(valid_count / FRAME_COUNT)
        mouth_visible = bool(review.get("mouth_visible") is True and review.get("status") == "PASS")
        array_metadata = {
            "schema_version": 1,
            "method": VISUAL_METHOD,
            "source_group": group,
            "clip_id": str(clip["clip_id"]),
            "clip_sha256": actual_sha,
            "model_path": str(model_path.resolve()),
            "model_sha256": file_sha256(model_path),
            "frame_count": FRAME_COUNT,
            "fps": FPS,
            "zero_pts": True,
            "width": width,
            "height": height,
            "metadata": metadata,
        }
        array_path = run_root / "acquisition" / "visual_arrays" / f"{group}_{clip['clip_id']}.npz"
        array_sha = _save_arrays(array_path, points[:FRAME_COUNT], valid[:FRAME_COUNT], timestamps[:FRAME_COUNT], array_metadata)
        base.update({"landmark_array_path": str(array_path.resolve()), "landmark_array_sha256": array_sha, "valid_array_sha256": hashlib.sha256(valid[:FRAME_COUNT].astype(np.bool_).tobytes()).hexdigest(), "frames_covered": valid_count, "valid_fraction": valid_fraction, "mouth_visible": mouth_visible, **measurements, "model_sha256": file_sha256(model_path), "model_path": str(model_path.resolve())})
        gate_failures = []
        if valid_fraction < MIN_VALID_FRACTION:
            gate_failures.append("valid_fraction")
        if not measurements["first_frame_face"]:
            gate_failures.append("first_frame_face")
        if measurements["eye_distance_px"] < MIN_EYE_DISTANCE_PX:
            gate_failures.append("eye_distance_px")
        if not mouth_visible:
            gate_failures.append("input_review_mouth_visible")
        if gate_failures:
            return dict(base, reason=";".join(gate_failures))
        return dict(base, status="PASS", reason=None, audit_verified=True)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, ImportError, AttributeError, TypeError) as exc:
        return dict(base, status="ERROR", reason=f"{type(exc).__name__}: {exc}")


def audit_pool(run_root: Path, pool_path: Path | None = None, *, model_path: Path | None = None, review_path: Path | None = None) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    try:
        import cv2

        cv2.setNumThreads(2)
    except ImportError:
        pass
    run_root = Path(run_root).resolve()
    acquisition = run_root / "acquisition"
    pool_file = (pool_path or acquisition / "pool.json").resolve()
    plan_file = acquisition / "plan.json"
    plan = load_self_hashed(plan_file)
    pool = load_self_hashed(pool_file)
    model_value = model_path or Path(str((plan.get("visual") or {}).get("model_path", REPO_ROOT / MODEL_RELATIVE_PATH)))
    model_path = model_value.resolve()
    model_sha = file_sha256(model_path) if model_path.is_file() else None
    review_index = _review_index(review_path)
    rows: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    if pool.get("source_plan_sha256") != file_sha256(plan_file):
        errors.append("pool is not bound to the current source plan")
    if pool.get("status") != "READY":
        errors.append(f"source pool status is {pool.get('status')}")
    if not model_path.is_file():
        errors.append(f"visual model asset is missing: {model_path}")
    elif model_sha is None:
        errors.append("visual model SHA cannot be computed")
    if errors:
        payload = {"schema_version": 1, "status": "BLOCKED_VISUAL_ASSET" if not model_path.is_file() else "BLOCKED_SOURCE_POOL", "method": VISUAL_METHOD, "model_path": str(model_path), "model_sha256": model_sha, "thresholds": {"min_valid_fraction": MIN_VALID_FRACTION, "min_eye_distance_px": MIN_EYE_DISTANCE_PX, "min_frames": MIN_FRAMES}, "source_plan_sha256": file_sha256(plan_file), "pool_sha256": file_sha256(pool_file), "groups": {}, "errors": errors}
        result = write_json(acquisition / "visual_audit.json", payload)
        (acquisition / "clip_audit.jsonl").write_text(json.dumps({"status": "BLOCKED", "reasons": errors}, ensure_ascii=False) + "\n", encoding="utf-8")
        return result
    for group_entry in pool.get("groups", []):
        group = str(group_entry.get("source_group"))
        if not GROUP_ID_RE.fullmatch(group):
            continue
        selected: dict[str, Any] | None = None
        for clip in sorted(group_entry.get("clips", []), key=lambda item: str(item.get("clip_id")))[: int(pool.get("clips_per_group", 8))]:
            result = _audit_clip(run_root, group, clip, model_path, review_index)
            rows.append(result)
            if result.get("status") == "PASS" and selected is None:
                selected = result
        if selected is not None:
            groups[group] = selected
    audit_status = "READY" if groups else "BLOCKED_VISUAL_SCREENING"
    _write_lines = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    (acquisition / "clip_audit.jsonl").write_text(_write_lines, encoding="utf-8")
    payload = {
        "schema_version": 1,
        "status": audit_status,
        "method": VISUAL_METHOD,
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "options": {"running_mode": "IMAGE", "num_faces": 1, "frame_count": FRAME_COUNT, "fps": FPS, "zero_pts": True},
        "thresholds": {"min_valid_fraction": MIN_VALID_FRACTION, "min_eye_distance_px": MIN_EYE_DISTANCE_PX, "min_frames": MIN_FRAMES},
        "audit_mode": "unblock_v1",
        "source_plan_sha256": file_sha256(plan_file),
        "pool_sha256": file_sha256(pool_file),
        "clip_audit_path": str((acquisition / "clip_audit.jsonl").resolve()),
        "clip_audit_sha256": file_sha256(acquisition / "clip_audit.jsonl"),
        "groups": groups,
        "attempted_clip_count": len(rows),
        "pass_group_count": len(groups),
    }
    return write_json(acquisition / "visual_audit.json", payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit_pool(args.run_root, args.pool, model_path=args.model, review_path=args.review)
    except (OSError, RuntimeError, ValueError, ImportError, KeyError, TypeError, AttributeError) as exc:
        print(f"visual audit status=BLOCKED error={type(exc).__name__}: {exc}")
        return 2
    print(f"visual audit status={result.get('status')} pass_groups={result.get('pass_group_count', 0)}")
    return 0 if result.get("status") == "READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
