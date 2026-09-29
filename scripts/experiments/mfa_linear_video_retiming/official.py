from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .common import ProtocolError, file_sha256, write_json


_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_C_RE = re.compile(rf"Confidence:\s*({_FLOAT})")
_D_RE = re.compile(rf"Min dist:\s*({_FLOAT})")
_OFFSET_RE = re.compile(r"AV offset:\s*(-?\d+)")


def _run_logged(command: Sequence[str], *, cwd: Path, env: Mapping[str, str], log: Path, timeout: int) -> dict[str, Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        with log.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(list(command), ensure_ascii=False) + "\n")
            handle.flush()
            result = subprocess.run(list(command), cwd=str(cwd), env=dict(env), stdout=handle,
                                    stderr=subprocess.STDOUT, timeout=int(timeout), check=False)
            handle.write(f"\n[returncode] {result.returncode}\n")
        return {"returncode": int(result.returncode), "elapsed_seconds": time.monotonic() - started}
    except subprocess.TimeoutExpired:
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[TIMEOUT] after {timeout} seconds\n")
        return {"returncode": 124, "elapsed_seconds": time.monotonic() - started, "timed_out": True}


def parse_official_log(path: str | Path) -> dict[str, float | int]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    c_values, d_values, offsets = _C_RE.findall(text), _D_RE.findall(text), _OFFSET_RE.findall(text)
    if len(c_values) != 1 or len(d_values) != 1 or len(offsets) != 1:
        raise ProtocolError(f"OFFICIAL_SCORE_PARSE_ERROR: expected one C, D, offset; got {len(c_values)}/{len(d_values)}/{len(offsets)}")
    result: dict[str, float | int] = {"sync_c": float(c_values[0]), "sync_d": float(d_values[0]), "offset": int(offsets[0])}
    if not np.isfinite(result["sync_c"]) or not np.isfinite(result["sync_d"]):
        raise ProtocolError("OFFICIAL_SCORE_NONFINITE")
    if abs(int(result["offset"])) == 15:
        raise ProtocolError("OFFICIAL_BOUNDARY_PEAK")
    return result


def _decode_video(ffprobe: Path, path: Path) -> tuple[np.ndarray, list[float]]:
    probe = subprocess.run([str(ffprobe), "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)],
                           capture_output=True, check=False)
    if probe.returncode != 0:
        raise ProtocolError(f"ffprobe PTS inspection failed: {path}")
    payload = json.loads(probe.stdout.decode("utf-8"))
    pts = [float(row["best_effort_timestamp_time"]) for row in payload.get("frames", [])]
    cap = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(np.ascontiguousarray(frame))
    cap.release()
    if not frames or len(pts) != len(frames):
        raise ProtocolError(f"video decode/PTS frame count mismatch: {path}")
    return np.stack(frames), pts


def _verify_mux_video(video: Path, muxed: Path, ffprobe: Path) -> dict[str, Any]:
    source, source_pts = _decode_video(ffprobe, video)
    actual, actual_pts = _decode_video(ffprobe, muxed)
    if source.shape != actual.shape or not np.array_equal(source, actual):
        raise ProtocolError("MUX_VIDEO_PIXEL_MISMATCH")
    if len(source_pts) != len(actual_pts) or not np.allclose(source_pts, actual_pts, atol=1e-7, rtol=0):
        raise ProtocolError("MUX_VIDEO_PTS_MISMATCH")
    return {
        "frame_count": int(source.shape[0]), "width": int(source.shape[2]), "height": int(source.shape[1]),
        "source_frame_sha256": hashlib.sha256(source.tobytes()).hexdigest(),
        "muxed_frame_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        "pts_sha256": hashlib.sha256(np.asarray(source_pts, dtype="<f8").tobytes()).hexdigest(),
        "pixels_identical": True, "pts_identical": True,
    }


def _matrix_from_activesd(path: Path) -> np.ndarray:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
    except Exception as exc:
        raise ProtocolError(f"ACTIVESD_UNREADABLE: {path}") from exc
    found: list[np.ndarray] = []

    def visit(value: Any) -> None:
        try:
            array = np.asarray(value)
        except Exception:
            array = np.asarray([], dtype=np.float32)
        if array.ndim == 2 and 31 in array.shape and min(array.shape) > 0 and np.issubdtype(array.dtype, np.number):
            found.append(np.asarray(array, dtype=np.float32))
            return
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(payload)
    candidates = [item for item in found if item.shape[0] == 31 or item.shape[1] == 31]
    unique: dict[tuple[tuple[int, int], str], np.ndarray] = {}
    for item in candidates:
        unique[(item.shape, hashlib.sha256(item.tobytes()).hexdigest())] = item
    if len(unique) != 1:
        raise ProtocolError(f"ACTIVESD_SHAPE_ERROR: expected one 31xN matrix, found {[item.shape for item in unique.values()]}")
    matrix = next(iter(unique.values()))
    if matrix.shape[0] != 31:
        matrix = matrix.T
    if matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ProtocolError("ACTIVESD_NONFINITE_OR_EMPTY")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def recompute_official_curve(matrix: np.ndarray, *, columns: int | None = None) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != 31 or not np.isfinite(values).all():
        raise ProtocolError("official distance matrix must be finite [31,N]")
    support = values.shape[1] if columns is None else int(columns)
    if support < 1 or support > values.shape[1]:
        raise ProtocolError("official curve support is invalid")
    curve = np.mean(values[:, :support], axis=1, dtype=np.float32).astype(np.float64)
    minimum = float(np.min(curve))
    index = int(np.flatnonzero(curve == minimum)[0])
    background = float(np.median(curve))
    return {
        "curve": curve.tolist(), "sync_c": background - minimum, "sync_d": minimum,
        "background_b": background, "d0": float(curve[15]), "offset": 15 - index,
        "best_lag": index - 15, "support_columns": support,
        "matrix_shape": [31, int(support)],
    }


def _find_single(patterns: Sequence[Path], label: str) -> Path:
    items = sorted({path.resolve() for root in patterns for path in root.rglob("*") if path.is_file()})
    if len(items) != 1:
        raise ProtocolError(f"OFFICIAL_{label}_COUNT: expected one, found {len(items)}")
    return items[0]


def _pipeline_crop_files(data_dir: Path, reference: str) -> list[Path]:
    """Return the face-track crops emitted by this SyncNet checkout.

    ``run_pipeline.py`` writes tracked crops below ``pycrop/<reference>``.
    The similarly named ``crop_dir`` tree belongs to another pipeline layout
    and is not created by the pinned SyncNet version in this repository.
    """

    crop_dirs = sorted((data_dir / "pycrop").glob(reference))
    return sorted(path for folder in crop_dirs for path in folder.glob("*.avi") if path.is_file())


def score_official_cell(
    config: Mapping[str, Any], cell: Mapping[str, Any], *, run_dir: str | Path,
    resume: bool = True, max_elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    repo = Path(config["repo_root"]).resolve()
    root = Path(run_dir).resolve()
    official_root = root / "06_official"
    sample_id, portrait_id = str(cell["sample_id"]), str(cell["portrait_id"])
    video_arm, audio_role = str(cell["video_arm"]), str(cell["audio_role"])
    key = f"p{portrait_id}_s{sample_id}_v{video_arm}_a{audio_role}"
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    cell_dir = official_root / "cells" / key
    result_path = cell_dir / "result.json"
    video, audio = Path(str(cell["video"])).resolve(), Path(str(cell["audio"])).resolve()
    for path in (video, audio):
        if not path.is_file():
            raise ProtocolError(f"OFFICIAL_INPUT_MISSING: {path}")
    expected = {"video_sha256": file_sha256(video), "audio_sha256": file_sha256(audio),
                "cell_key": key, "syncnet_model_sha256": file_sha256(config["paths"]["syncnet_model"])}
    if resume and result_path.is_file():
        from .common import verify_json
        saved = verify_json(result_path, self_hash=True)
        if all(saved.get(name) == value for name, value in expected.items()) and saved.get("status") == "PASS":
            return saved
        raise ProtocolError(f"OFFICIAL_RESUME_FINGERPRINT_MISMATCH: {key}")

    cell_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = Path(config["paths"]["ffmpeg"]).resolve()
    ffprobe = Path(config["paths"]["ffprobe"]).resolve()
    helper = __import__("scripts.experiments.phone_gain_static_tfg_mfa.official_score", fromlist=["strict_mux"])
    muxed = cell_dir / "cell.mkv"
    mux = helper.strict_mux(video=video, audio=audio, output=muxed, ffmpeg=ffmpeg, ffprobe=ffprobe,
                            log_path=cell_dir / "mux.log")
    mux_pixels = _verify_mux_video(video, muxed, ffprobe)
    sync_root = Path(config["paths"]["syncnet_root"]).resolve()
    data_dir = cell_dir / "pipeline"
    reference = f"mfa_linear_video_retiming_{key}"
    official_cfg = config["official"]
    env = dict(os.environ)
    env["PATH"] = str(ffmpeg.parent) + os.pathsep + env.get("PATH", "")
    found_ffmpeg = shutil.which("ffmpeg", path=env["PATH"])
    if not found_ffmpeg or Path(found_ffmpeg).resolve() != ffmpeg:
        raise ProtocolError("OFFICIAL_FFMPEG_PATH_MISMATCH")
    commands = [
        ([str(Path(config["paths"]["syncnet_python"]).absolute()), str(sync_root / "run_pipeline.py"),
          "--videofile", str(muxed), "--reference", reference, "--data_dir", str(data_dir),
          "--min_track", str(official_cfg["min_track"]), "--overwrite"],
         cell_dir / "pipeline.log", int(official_cfg["pipeline_timeout_seconds"])),
        ([str(Path(config["paths"]["syncnet_python"]).absolute()), str(sync_root / "run_syncnet.py"),
          "--videofile", str(muxed), "--reference", reference, "--data_dir", str(data_dir),
          "--initial_model", str(Path(config["paths"]["syncnet_model"]).resolve()),
          "--vshift", "15", "--batch_size", str(official_cfg["batch_size"])],
         cell_dir / "syncnet.log", int(official_cfg["syncnet_timeout_seconds"])),
    ]
    command_receipts = []
    for command, log, timeout in commands:
        effective_timeout = timeout
        budget_limited_timeout = False
        if max_elapsed_seconds is not None:
            remaining = float(max_elapsed_seconds) - (time.monotonic() - started)
            if remaining < 1.0:
                raise ProtocolError("BUDGET_LIMITED:TOTAL_ACTIVE_GPU_BUDGET_EXHAUSTED")
            effective_timeout = min(timeout, max(1, int(remaining)))
            budget_limited_timeout = effective_timeout < timeout or remaining <= timeout
        run = _run_logged(command, cwd=sync_root, env=env, log=log, timeout=effective_timeout)
        command_receipts.append({"argv": command, "log": str(log.resolve()), **run})
        if run.get("timed_out") and budget_limited_timeout:
            raise ProtocolError("BUDGET_LIMITED:TOTAL_ACTIVE_GPU_BUDGET_EXHAUSTED")
        if run["returncode"] != 0:
            raise ProtocolError(f"OFFICIAL_COMMAND_FAILED:{log.name}:{run['returncode']}")

    crop_files = _pipeline_crop_files(data_dir, reference)
    if len(crop_files) == 0:
        raise ProtocolError("OFFICIAL_ZERO_TRACK")
    if len(crop_files) != 1:
        raise ProtocolError(f"OFFICIAL_MULTIPLE_TRACKS:{len(crop_files)}")
    crop = crop_files[0]
    capture = cv2.VideoCapture(str(crop))
    crop_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    crop_fps = float(capture.get(cv2.CAP_PROP_FPS))
    crop_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    crop_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if crop_count <= 0 or abs(crop_fps - 25.0) > 0.1:
        raise ProtocolError("OFFICIAL_CROP_MEDIA_INVALID")
    activesd_paths = list(data_dir.rglob("activesd.pckl"))
    if len(activesd_paths) != 1:
        raise ProtocolError(f"OFFICIAL_ACTIVESD_COUNT:{len(activesd_paths)}")
    matrix = _matrix_from_activesd(activesd_paths[0])
    recomputed = recompute_official_curve(matrix)
    parsed = parse_official_log(cell_dir / "syncnet.log")
    if abs(float(parsed["sync_c"]) - float(recomputed["sync_c"])) > 0.0011 or abs(float(parsed["sync_d"]) - float(recomputed["sync_d"])) > 0.0011:
        raise ProtocolError("OFFICIAL_LOG_MATRIX_PARITY_FAILED")
    if int(parsed["offset"]) != int(recomputed["offset"]):
        raise ProtocolError("OFFICIAL_LOG_MATRIX_OFFSET_MISMATCH")
    track_paths = list(data_dir.rglob("tracks.pckl"))
    if len(track_paths) > 1:
        raise ProtocolError("OFFICIAL_TRACK_PICKLE_AMBIGUOUS")
    result = {
        "schema_version": 1, "status": "PASS", "score_kind": "official_pipeline", **expected,
        "sample_id": sample_id, "paired_key": str(cell["paired_key"]), "speaker_id": str(cell["speaker_id"]),
        "portrait_id": portrait_id, "video_arm": video_arm, "audio_role": audio_role,
        "video": str(video), "audio": str(audio), "mux": mux, "mux_pixel_pts_audit": mux_pixels,
        "reference": reference, "min_track": int(official_cfg["min_track"]),
        "ffmpeg_resolved_by_official_pipeline": str(Path(found_ffmpeg).resolve()),
        "commands": command_receipts,
        "crop": {"path": str(crop.resolve()), "sha256": file_sha256(crop), "frame_count": crop_count,
                 "source_time_start_seconds": 0.0, "source_time_end_seconds": crop_count / crop_fps,
                 "fps": crop_fps, "width": crop_width, "height": crop_height},
        "tracks_pickle": ({"path": str(track_paths[0].resolve()), "sha256": file_sha256(track_paths[0])} if track_paths else None),
        "activesd": {"path": str(activesd_paths[0].resolve()), "sha256": file_sha256(activesd_paths[0]),
                     "shape": [31, int(matrix.shape[1])], "distance_matrix": matrix.tolist()},
        "official_sync_c": float(parsed["sync_c"]), "official_sync_d": float(parsed["sync_d"]),
        "official_offset": int(parsed["offset"]), "recomputed_official_curve": recomputed,
        "search_sync_c": cell.get("search_sync_c"), "search_sync_d": cell.get("search_sync_d"),
        "search_d0": cell.get("search_d0"), "search_offset": cell.get("search_offset"),
    }
    return write_json(result_path, result, self_hash=True)


def add_common_support(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matrices = [np.asarray(row["activesd"]["distance_matrix"], dtype=np.float32) for row in rows]
    if not matrices:
        raise ProtocolError("no official matrices to compare")
    widths = [matrix.shape[1] for matrix in matrices]
    common = min(widths)
    curves = [recompute_official_curve(matrix, columns=common) for matrix in matrices]
    return {"status": "MATCHED" if len(set(widths)) == 1 else "SUPPORT_MISMATCH",
            "support_columns_by_cell": widths, "common_support_columns": common,
            "common_support_curves": curves}
