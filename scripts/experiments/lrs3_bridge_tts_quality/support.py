"""Freeze common video/crop support before rendering and scoring."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .audio import read_pcm16
from .common import (
    ProtocolError,
    canonical_json_sha256,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .gpu import gpu_lease, release_torch_memory


def _ffprobe(path: Path) -> dict[str, Any]:
    if not config.FFPROBE.is_file():
        raise ProtocolError(f"ffprobe is missing: {config.FFPROBE}")
    result = subprocess.run(
        [str(config.FFPROBE), "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ProtocolError(f"ffprobe failed for {path}: {result.stderr[-1000:]}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"ffprobe returned invalid JSON: {path}") from exc
    return payload


def probe_video(path: str | Path) -> dict[str, Any]:
    """Return decoded frame count and geometry used by the fixed support."""

    target = Path(path)
    payload = _ffprobe(target)
    streams = [row for row in payload.get("streams", []) if row.get("codec_type") == "video"]
    if len(streams) != 1:
        raise ProtocolError(f"expected one video stream: {target}")
    stream = streams[0]
    frame_value = stream.get("nb_read_frames", stream.get("nb_frames"))
    try:
        frame_count = int(frame_value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"video frame count is unavailable: {target}") from exc
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise ProtocolError(f"video geometry/frame count is invalid: {target}")
    rate = str(stream.get("r_frame_rate", ""))
    return {
        "path": str(target.resolve()),
        "file_sha256": file_sha256(target),
        "decoded_frame_count": frame_count,
        "width": width,
        "height": height,
        "r_frame_rate": rate,
        "fps": config.FPS,
        "full_frame_box": [0, height, 0, width],
    }


def common_support_length(natural_samples: int, source_frames: int) -> int:
    if natural_samples <= 0 or source_frames <= 0:
        raise ProtocolError("support inputs must be positive")
    frames = min(int(source_frames), int(natural_samples) // config.SAMPLES_PER_FRAME)
    if frames < config.SYNCNET_MIN_TRACK:
        raise ProtocolError(f"common support is shorter than {config.SYNCNET_MIN_TRACK} frames")
    return frames


def _as_frame_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64).reshape(-1)
    if array.size == 0:
        raise ProtocolError("face track has no frames")
    return array


def choose_track(tracks: Sequence[Mapping[str, Any]], required_frames: int) -> tuple[int, Mapping[str, Any]]:
    """Select one original-video track by fixed coverage/order only."""

    candidates: list[tuple[int, int, Mapping[str, Any]]] = []
    for index, item in enumerate(tracks):
        track = item.get("track") if isinstance(item, Mapping) else None
        proc = item.get("proc_track") if isinstance(item, Mapping) else None
        if not isinstance(track, Mapping) or not isinstance(proc, Mapping):
            continue
        frames = _as_frame_array(track.get("frame"))
        if frames[0] > 0 or frames[-1] < required_frames - 1:
            continue
        expected = np.arange(int(frames[0]), int(frames[-1]) + 1, dtype=np.int64)
        if not np.array_equal(frames, expected) or frames.size < required_frames:
            continue
        try:
            x = np.asarray(proc["x"], dtype=np.float64).reshape(-1)
            y = np.asarray(proc["y"], dtype=np.float64).reshape(-1)
            s = np.asarray(proc["s"], dtype=np.float64).reshape(-1)
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("selected track has malformed processed coordinates") from exc
        if min(x.size, y.size, s.size) < required_frames or not all(np.isfinite(value[:required_frames]).all() for value in (x, y, s)):
            continue
        if np.any(s[:required_frames] <= 0):
            continue
        # Longest coverage wins; index is the stable tie breaker.  No score
        # or generated arm is consulted here.
        candidates.append((int(frames.size), index, item))
    if not candidates:
        raise ProtocolError("original face video has no deterministic track covering frame 0/support")
    candidates.sort(key=lambda value: (-value[0], value[1]))
    return candidates[0][1], candidates[0][2]


def _run_support_detector(face_video: Path, sample_id: str, work_dir: Path, log_path: Path) -> Mapping[str, Any]:
    command = [
        str(config.SYNCNET_PYTHON),
        "run_pipeline.py",
        "--videofile", str(face_video),
        "--reference", sample_id,
        "--data_dir", str(work_dir),
        "--min_track", str(config.SYNCNET_MIN_TRACK),
        "--frame_rate", str(config.FPS),
        "--overwrite",
    ]
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(config.FFMPEG.parent.resolve()), env.get("PATH", "")])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(config.SYNCNET_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise ProtocolError(f"face support detector failed for {sample_id}: exit {result.returncode}")
    tracks_path = work_dir / "pywork" / sample_id / "tracks.pckl"
    if not tracks_path.is_file():
        raise ProtocolError(f"face support detector did not write tracks: {sample_id}")
    with tracks_path.open("rb") as handle:
        tracks = pickle.load(handle)
    if not isinstance(tracks, list):
        raise ProtocolError(f"face support track file is malformed: {sample_id}")
    return {"command": command, "log": str(log_path.resolve()), "tracks": tracks, "tracks_path": str(tracks_path.resolve())}


def _serialize_track(track: Mapping[str, Any], required_frames: int) -> dict[str, Any]:
    source_track = track["track"]
    proc = track["proc_track"]
    frames = _as_frame_array(source_track["frame"])
    bbox = np.asarray(source_track["bbox"], dtype=np.float64)
    if bbox.ndim != 2 or bbox.shape[0] != frames.size or bbox.shape[1] != 4:
        raise ProtocolError("face track bbox shape is invalid")
    return {
        "frame": frames[:required_frames].tolist(),
        "bbox": bbox[:required_frames].tolist(),
        "proc_track": {
            key: np.asarray(proc[key], dtype=np.float64).reshape(-1)[:required_frames].tolist()
            for key in ("x", "y", "s")
        },
    }


def freeze_support(run_root: Path, cohort: Mapping[str, Any]) -> dict[str, Any]:
    """Detect tracks on each original face video exactly once, serially."""

    paths = config.RunPaths(run_root)
    paths.bridge.mkdir(parents=True, exist_ok=True)
    support_dir = paths.bridge / "support"
    support_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    gpu_held = False
    try:
        with gpu_lease("syncnet_support"):
            gpu_held = True
            for record in cohort.get("records", []):
                sample_id = str(record["sample_id"])
                output = support_dir / f"{sample_id}.json"
                try:
                    natural, _natural_meta = read_pcm16(Path(str(record["natural_audio"]["path"])))
                    video = Path(str(record["face_video"]["path"]))
                    video_meta = probe_video(video)
                    frames = common_support_length(natural.size, int(video_meta["decoded_frame_count"]))
                    if output.is_file():
                        prior = verify_self_hashed_json(output)
                        if (
                            prior.get("sample_id") == sample_id
                            and prior.get("face_video_sha256") == video_meta["file_sha256"]
                            and prior.get("support_frames") == frames
                        ):
                            rows.append(prior)
                            continue
                        raise ProtocolError(f"existing support identity changed: {sample_id}")
                    detector = _run_support_detector(video, sample_id, paths.bridge / "support_work" / sample_id, paths.bridge / "logs" / f"support_{sample_id}.log")
                    track_index, track = choose_track(detector["tracks"], frames)
                    serialized = _serialize_track(track, frames)
                    row = {
                        "schema_version": 1,
                        "stage_id": "03_bridge",
                        "protocol_id": config.PROTOCOL_ID,
                        "sample_id": sample_id,
                        "source_group": str(record["source_group"]),
                        "face_video": str(video.resolve()),
                        "face_video_sha256": video_meta["file_sha256"],
                        "natural_audio_sha256": str(record["natural_audio"]["sha256"]),
                        "natural_sample_count": int(natural.size),
                        "source_decoded_frame_count": int(video_meta["decoded_frame_count"]),
                        "support_frame_start": 0,
                        "support_frames": frames,
                        "support_audio_samples": frames * config.SAMPLES_PER_FRAME,
                        "fps": config.FPS,
                        "track_index": track_index,
                        "track": serialized,
                        "crop_scale": 0.40,
                        "crop_size": [224, 224],
                        "detector": {key: value for key, value in detector.items() if key != "tracks"},
                        "full_frame_geometry": video_meta,
                        "selection_policy": "original face video; starts at frame 0; longest covering track; index tie-break",
                    }
                    write_self_hashed_json(output, row)
                    rows.append(row)
                    print(f"SUPPORT {len(rows)}/{config.EXPECTED_RECORD_COUNT} {sample_id}", flush=True)
                except Exception as exc:  # noqa: BLE001 - preserve each failed support record
                    failures.append({"sample_id": sample_id, "error_type": type(exc).__name__, "error": str(exc)})
    except Exception as exc:  # noqa: BLE001 - preserve stage failure in manifest
        failures.append({"sample_id": None, "error_type": type(exc).__name__, "error": str(exc), "gpu_stage": True})
    finally:
        if gpu_held:
            release_torch_memory()
    complete = not failures and len(rows) == config.EXPECTED_RECORD_COUNT
    payload = {
        "schema_version": 1,
        "stage_id": "03_bridge",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if complete else "blocked",
        "record_count": len(rows),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "support_frames_formula": "min(source_decoded_frame_count, floor(natural_samples/640))",
        "rows": rows,
        "failures": failures,
    }
    write_self_hashed_json(paths.bridge / "geometry.json", payload)
    if not complete:
        raise ProtocolError(f"common support incomplete: {len(rows)}/{config.EXPECTED_RECORD_COUNT}")
    return payload


def _load_proc_track(support: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proc = support.get("track", {}).get("proc_track")
    if not isinstance(proc, Mapping):
        raise ProtocolError("support has no processed track")
    values = tuple(np.asarray(proc[key], dtype=np.float64).reshape(-1) for key in ("x", "y", "s"))
    required = int(support["support_frames"])
    if any(value.size != required for value in values) or not all(np.isfinite(value).all() for value in values):
        raise ProtocolError("processed track does not match frozen support")
    return values  # type: ignore[return-value]


def crop_video_with_frozen_track(video: Path, support: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Apply one frozen SyncNet crop to one rendered video."""

    import cv2

    if file_sha256(Path(str(support["face_video"]))) != str(support["face_video_sha256"]):
        raise ProtocolError("frozen support source face video changed")
    frames_required = int(support["support_frames"])
    x_values, y_values, s_values = _load_proc_track(support)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ProtocolError(f"cannot open rendered video: {video}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"XVID"), config.FPS, (224, 224))
    if not writer.isOpened():
        capture.release()
        raise ProtocolError(f"cannot create frozen crop: {output}")
    count = 0
    expected_width = int(support.get("full_frame_geometry", {}).get("width", 0))
    expected_height = int(support.get("full_frame_geometry", {}).get("height", 0))
    try:
        while count < frames_required:
            ok, image = capture.read()
            if not ok:
                raise ProtocolError(f"rendered video ends before frozen support: {video}")
            if expected_width and expected_height and (image.shape[1], image.shape[0]) != (expected_width, expected_height):
                raise ProtocolError(f"rendered frame geometry differs from frozen full-frame geometry: {video}")
            bs = float(s_values[count])
            bsi = int(bs * (1.0 + 2.0 * 0.40))
            if bsi <= 0:
                raise ProtocolError("frozen crop scale is invalid")
            padded = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), "constant", constant_values=(110, 110))
            my = float(y_values[count]) + bsi
            mx = float(x_values[count]) + bsi
            top = int(my - bs)
            bottom = int(my + bs * (1.0 + 2.0 * 0.40))
            left = int(mx - bs * (1.0 + 0.40))
            right = int(mx + bs * (1.0 + 0.40))
            face = padded[top:bottom, left:right]
            if face.size == 0:
                raise ProtocolError(f"frozen crop is empty at frame {count}: {video}")
            writer.write(cv2.resize(face, (224, 224)))
            count += 1
    finally:
        capture.release()
        writer.release()
    temporary.replace(output)
    return {
        "path": str(output.resolve()),
        "sha256": file_sha256(output),
        "source_video_sha256": file_sha256(video),
        "support_sha256": canonical_json_sha256(dict(support)),
        "frame_count": frames_required,
        "fps": config.FPS,
        "crop_scale": 0.40,
        "crop_size": [224, 224],
    }


__all__ = ["choose_track", "common_support_length", "crop_video_with_frozen_track", "freeze_support", "probe_video"]
