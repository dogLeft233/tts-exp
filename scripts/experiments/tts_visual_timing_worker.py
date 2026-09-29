#!/usr/bin/env python3
"""Python-3.8-compatible visual/media worker for ``tts_visual_timing``.

The worker deliberately does not import the older visual feature module.  It
records raw MediaPipe output (including invalid frames) and performs only
media operations.  All scientific scoring stays in the pure metrics module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import wave
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%s.tmp" % (path.name, os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def _probe(path: Path) -> Dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("ffprobe failed: %s" % result.stderr.decode("utf-8", errors="replace")[-1000:])
    return json.loads(result.stdout.decode("utf-8"))


def _probe_frame_pts(path: Path) -> List[Dict[str, Any]]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=pts,best_effort_timestamp,pts_time,best_effort_timestamp_time",
            "-of", "json", str(path),
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("ffprobe frame timestamps failed")
    payload = json.loads(result.stdout.decode("utf-8"))
    return [dict(item) for item in payload.get("frames", []) if isinstance(item, dict)]


def _open_landmarker(asset: Path):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(asset.resolve())),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=2,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


def extract(
    video: Path,
    landmarker_asset: Path,
    output_npz: Path,
    output_json: Optional[Path] = None,
    preview_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Extract all 478 normalized landmarks and the original frame validity."""

    if not video.is_file() or not landmarker_asset.is_file():
        raise FileNotFoundError(str(video if not video.is_file() else landmarker_asset))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError("cannot open video: %s" % video)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("invalid video metadata: %s" % video)
    mp, landmarker = _open_landmarker(landmarker_asset)
    landmarks: List[np.ndarray] = []
    valid: List[bool] = []
    multiple_face_frames: List[int] = []
    preview_frames: Dict[str, np.ndarray] = {}
    decode_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[:2] != (height, width) or frame.shape[2] != 3:
                raise RuntimeError("malformed decoded frame %d" % decode_count)
            # Keep three decoded RGB-independent previews for auditability.  We
            # retain only three frames in memory and never use them for the
            # metric itself.
            if decode_count == 0:
                preview_frames["first"] = frame.copy()
            if declared > 0 and decode_count == declared // 2:
                preview_frames["middle"] = frame.copy()
            preview_frames["last"] = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            faces = list(result.face_landmarks or [])
            if len(faces) > 1:
                multiple_face_frames.append(decode_count)
            if len(faces) == 1 and len(faces[0]) == 478:
                landmarks.append(np.asarray([(float(point.x), float(point.y)) for point in faces[0]], dtype=np.float64))
                valid.append(True)
            else:
                landmarks.append(np.full((478, 2), np.nan, dtype=np.float64))
                valid.append(False)
            decode_count += 1
    finally:
        capture.release()
        landmarker.close()
    if not landmarks:
        raise RuntimeError("video has no decodable frames: %s" % video)
    pts = _probe_frame_pts(video)
    timestamps = np.arange(len(landmarks), dtype=np.float64) / fps
    if len(pts) == len(landmarks):
        raw_times: List[Optional[float]] = []
        for item in pts:
            value = item.get("best_effort_timestamp_time", item.get("pts_time"))
            try:
                raw_times.append(float(value))
            except (TypeError, ValueError):
                raw_times.append(None)
        if all(value is not None and math.isfinite(float(value)) for value in raw_times):
            timestamps = np.asarray([float(value) for value in raw_times], dtype=np.float64)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_npz.with_name(".%s.%s.tmp.npz" % (output_npz.name, os.getpid()))
    np.savez_compressed(
        str(temporary),
        landmarks=np.stack(landmarks, axis=0),
        valid=np.asarray(valid, dtype=np.bool_),
        timestamps_s=timestamps,
    )
    os.replace(str(temporary), str(output_npz))
    previews: List[Dict[str, Any]] = []
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        for name in ("first", "middle", "last"):
            image = preview_frames.get(name)
            if image is None:
                continue
            path = preview_dir / (name + ".png")
            if not cv2.imwrite(str(path), image):
                raise RuntimeError("cannot write preview: %s" % path)
            previews.append({"name": name, "path": str(path.resolve()), "sha256": sha256_file(path)})
    metadata: Dict[str, Any] = {
        "status": "MULTIPLE_FACES" if multiple_face_frames else "COMPLETE",
        "video": str(video.resolve()),
        "video_sha256": sha256_file(video),
        "landmarker_asset": str(landmarker_asset.resolve()),
        "landmarker_sha256": sha256_file(landmarker_asset),
        "fps": fps,
        "width": width,
        "height": height,
        "declared_frame_count": declared,
        "decoded_frame_count": len(landmarks),
        "valid_fraction": float(np.mean(valid)),
        "multiple_face_frames": multiple_face_frames,
        "pts_count": len(pts),
        "pts_source": "ffprobe_best_effort_timestamp_time" if len(pts) == len(landmarks) else "frame_index_over_fps",
        "pts_qc": {
            "count_matches_decoded": len(pts) == len(landmarks),
            "strictly_increasing": bool(len(timestamps) < 2 or np.all(np.diff(timestamps) > 0)),
            "max_interval_deviation_ms": float(np.max(np.abs(np.diff(timestamps) - (1.0 / fps))) * 1000.0) if len(timestamps) > 1 else 0.0,
        },
        "failure_stats": {"invalid_frames": int(np.sum(~np.asarray(valid, dtype=bool))), "multiple_face_frames": len(multiple_face_frames)},
        "previews": previews,
        "npz_sha256": sha256_file(output_npz),
    }
    if output_json is not None:
        write_json(output_json, metadata)
    return metadata


def _read_frames(path: Path) -> Tuple[List[np.ndarray], float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("cannot open video: %s" % path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(np.ascontiguousarray(frame))
    finally:
        capture.release()
    if not frames or fps <= 0:
        raise RuntimeError("source video is empty or has invalid fps")
    return frames, fps, width, height


def _encode_ffv1(frames: Sequence[np.ndarray], fps: float, output: Path) -> Dict[str, Any]:
    if not frames:
        raise ValueError("cannot encode empty frame sequence")
    height, width = frames[0].shape[:2]
    raw = b"".join(np.ascontiguousarray(frame, dtype=np.uint8).tobytes() for frame in frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(".%s.%s.partial%s" % (output.stem, os.getpid(), output.suffix))
    command = [
        "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", "%dx%d" % (width, height), "-r", "%.12g" % fps, "-i", "pipe:0",
        "-an", "-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "bgr0", "-f", "matroska", str(temporary),
    ]
    result = subprocess.run(command, input=raw, capture_output=True, check=False)
    if result.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        raise RuntimeError("FFV1 encode failed: %s" % result.stderr.decode("utf-8", errors="replace")[-1000:])
    os.replace(str(temporary), str(output))
    return {"output": str(output.resolve()), "sha256": sha256_file(output), "frame_count": len(frames), "width": width, "height": height, "fps": fps, "command": command}


def retime_video(
    video: Path,
    output: Path,
    *,
    delta_s: float = 0.0,
    frozen: bool = False,
    protocol: str = "tts_visual_timing_v1",
) -> Dict[str, Any]:
    """Create a silent FFV1 control with the exact registered frame map."""

    from tts_visual_timing_metrics import local_warp_indices

    frames, fps, width, height = _read_frames(video)
    if abs(float(fps) - 25.0) > 1e-6:
        raise RuntimeError("protocol requires 25 fps, got %.9g" % fps)
    if frozen:
        middle = len(frames) // 2
        mapping = [middle for _ in frames]
        mapping_meta: Dict[str, Any] = {"mode": "FROZEN", "middle_index": middle, "frame_count": len(frames), "fps": fps}
    elif abs(float(delta_s)) < 1e-15:
        mapping = list(range(len(frames)))
        mapping_meta = {"mode": "IDENTITY", "frame_count": len(frames), "fps": fps}
    else:
        mapping, mapping_meta = local_warp_indices(len(frames), fps, float(delta_s), protocol=protocol)
        mapping_meta["mode"] = "LOCAL_WARP"
    selected = [frames[index] for index in mapping]
    result = _encode_ffv1(selected, fps, output)
    result.update({"source": str(video.resolve()), "source_sha256": sha256_file(video), "mapping": mapping, "mapping_meta": mapping_meta, "protocol": protocol, "has_audio": False})
    return result


def _read_pcm16(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        width = handle.getsampwidth()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if channels != 1 or rate != 16000 or width != 2:
        raise RuntimeError("audio must be mono 16 kHz PCM16")
    pcm = np.frombuffer(raw, dtype="<i2").copy()
    return pcm, {"channels": channels, "sample_rate": rate, "sample_width": width, "sample_count": int(len(pcm)), "pcm_sha256": hashlib.sha256(pcm.astype("<i2", copy=False).tobytes()).hexdigest(), "peak": int(np.max(np.abs(pcm))) if len(pcm) else 0, "saturated_samples": int(np.sum(np.abs(pcm) >= 32767))}


def _write_timemap(path: Path, source_knots: Sequence[int], target_knots: Sequence[int]) -> None:
    if len(source_knots) != len(target_knots) or len(source_knots) < 2:
        raise ValueError("time map needs at least two paired knots")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["%d %d" % (int(source), int(target)) for source, target in zip(source_knots, target_knots)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stretch_audio(source: Path, output: Path, *, rubberband: Path, source_knots: Sequence[int], target_knots: Sequence[int], time_ratio: float) -> Dict[str, Any]:
    """Run the fixed Rubber Band time-map protocol and perform PCM QC."""

    if not rubberband.is_file() or not os.access(str(rubberband), os.X_OK):
        raise FileNotFoundError(str(rubberband))
    source_pcm, source_meta = _read_pcm16(source)
    if not math.isfinite(float(time_ratio)) or time_ratio <= 0:
        raise ValueError("time ratio must be positive")
    with tempfile.TemporaryDirectory(prefix="tts_visual_timing_", dir=str(output.parent if output.parent.exists() else Path("/tmp"))) as temp:
        timemap = Path(temp) / "timemap.txt"
        _write_timemap(timemap, source_knots, target_knots)
        command = [str(rubberband), "-3", "--pitch", "0", "--ignore-clipping", "--timemap", str(timemap), "--time", "%.12g" % float(time_ratio), str(source), str(output)]
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode != 0 or not output.is_file():
            raise RuntimeError("Rubber Band failed: %s" % result.stderr.decode("utf-8", errors="replace")[-2000:])
        combined_log = (result.stdout + b"\n" + result.stderr).decode("utf-8", errors="replace")
        if re.search(r"(?:automatic|auto(?:matic)?)[^\n]{0,80}gain|\bclipping\b|\bclipped\b", combined_log, flags=re.IGNORECASE):
            raise RuntimeError("Rubber Band reported clipping or automatic gain handling")
    output_pcm, output_meta = _read_pcm16(output)
    if output_meta["saturated_samples"]:
        raise RuntimeError("Rubber Band output contains saturated PCM samples")
    if not np.isfinite(output_pcm.astype(np.float64)).all():
        raise RuntimeError("Rubber Band output contains non-finite samples")
    expected = int(target_knots[-1])
    length_delta = int(len(output_pcm) - expected)
    if abs(length_delta) > 1:
        raise RuntimeError("Rubber Band output length differs by more than one sample")
    # A one-sample excess is the only permitted correction.  Never pad or
    # truncate a larger discrepancy, and record the correction explicitly.
    tail_trimmed = 0
    if length_delta == 1:
        with wave.open(str(output), "wb") as handle:
            handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000)
            handle.writeframes(np.asarray(output_pcm[:-1], dtype="<i2").tobytes())
        output_pcm, output_meta = _read_pcm16(output)
        tail_trimmed = 1
        length_delta = int(len(output_pcm) - expected)
    return {"status": "COMPLETE", "source": str(source.resolve()), "source_sha256": sha256_file(source), "output": str(output.resolve()), "output_sha256": sha256_file(output), "source_meta": source_meta, "output_meta": output_meta, "length_delta_samples": length_delta, "tail_trimmed_samples": tail_trimmed, "command": command, "stdout": result.stdout.decode("utf-8", errors="replace"), "stderr": result.stderr.decode("utf-8", errors="replace")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("extract", "retime_video", "stretch_audio"), required=True)
    parser.add_argument("--video")
    parser.add_argument("--landmarker")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata")
    parser.add_argument("--delta-ms", type=float, default=0.0)
    parser.add_argument("--protocol", choices=("tts_visual_timing_v1", "tts_visual_timing_v2"), default="tts_visual_timing_v1")
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--audio")
    parser.add_argument("--rubberband")
    parser.add_argument("--source-knots", default="")
    parser.add_argument("--target-knots", default="")
    parser.add_argument("--time-ratio", type=float, default=1.0)
    parser.add_argument("--preview-dir")
    args = parser.parse_args(argv)
    try:
        if args.mode == "extract":
            if not args.video or not args.landmarker:
                raise ValueError("extract requires --video and --landmarker")
            result = extract(Path(args.video), Path(args.landmarker), Path(args.output), Path(args.metadata) if args.metadata else None, Path(args.preview_dir) if args.preview_dir else None)
        elif args.mode == "retime_video":
            if not args.video:
                raise ValueError("retime_video requires --video")
            result = retime_video(Path(args.video), Path(args.output), delta_s=float(args.delta_ms) / 1000.0, frozen=bool(args.frozen), protocol=args.protocol)
        else:
            if not args.audio or not args.rubberband:
                raise ValueError("stretch_audio requires --audio and --rubberband")
            source_knots = [int(item) for item in args.source_knots.split(",") if item.strip()]
            target_knots = [int(item) for item in args.target_knots.split(",") if item.strip()]
            result = stretch_audio(Path(args.audio), Path(args.output), rubberband=Path(args.rubberband), source_knots=source_knots, target_knots=target_knots, time_ratio=float(args.time_ratio))
        if args.metadata and args.mode != "extract":
            write_json(Path(args.metadata), result)
        else:
            print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        error = {"status": "ERROR", "error_type": type(exc).__name__, "reason": str(exc)}
        if args.metadata:
            write_json(Path(args.metadata), error)
        else:
            print(json.dumps(error, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
