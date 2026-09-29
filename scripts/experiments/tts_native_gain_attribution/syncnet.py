"""Frozen-video preparation and official SyncNet V2 feature/matrix export."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .audio import read_pcm16_wav, write_pcm16_wav
from .common import (
    ProtocolError,
    canonical_sha256,
    copy_atomic,
    executable_path,
    file_sha256,
    gpu_lease,
    read_self_hashed_json,
    run_command,
    run_monitored,
    write_self_hashed_json,
)

if str(config.SYNCNET_ROOT) not in sys.path:
    sys.path.insert(0, str(config.SYNCNET_ROOT))


def _npy_atomic(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, np.asarray(value), allow_pickle=False)
    temporary.replace(path)
    return file_sha256(path)


def _json_hash(value: Any) -> str:
    return canonical_sha256(value)


def scoring_environment() -> dict[str, Any]:
    """Record libraries that affect JPEG decode or SyncNet tensor kernels."""

    import cv2
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def scoring_environment_hash() -> str:
    return canonical_sha256(scoring_environment())


def video_signature(path: str | Path) -> dict[str, Any]:
    """Hash decoded pixels in order, without materialising a frame directory."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ProtocolError("opencv is required for SyncNet video decoding") from exc
    target = Path(path)
    capture = cv2.VideoCapture(str(target))
    digest = hashlib.sha256()
    frame_count = 0
    shape: list[int] | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_count == 0:
                shape = [int(item) for item in frame.shape]
            if shape != [int(item) for item in frame.shape]:
                raise ProtocolError(f"video frame shape changes within {target}")
            digest.update(np.ascontiguousarray(frame).tobytes())
            frame_count += 1
    finally:
        capture.release()
    if frame_count < 5 or shape is None:
        raise ProtocolError(f"video has too few decodable frames: {target}")
    probe = read_json_probe(target)
    video_stream = next((row for row in probe.get("streams", []) if isinstance(row, Mapping) and row.get("codec_type") == "video"), {})
    first_frame = first_video_frame_timestamps(target)
    pts_payload = {
        "time_base": video_stream.get("time_base"),
        "start_time": video_stream.get("start_time"),
        "first_frame_pts": first_frame.get("pts"),
        "first_frame_pts_time": first_frame.get("pts_time"),
        "first_frame_best_effort_timestamp": first_frame.get("best_effort_timestamp"),
        "first_frame_best_effort_timestamp_time": first_frame.get("best_effort_timestamp_time"),
        "frame_count": frame_count,
        "fps": video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"),
    }
    return {
        "path": str(target.resolve()),
        "file_sha256": file_sha256(target),
        "pixel_sha256": digest.hexdigest(),
        "frame_count": frame_count,
        "frame_shape": shape,
        "pts": pts_payload,
        "pts_sha256": _json_hash(pts_payload),
        "video_hash": _json_hash({"pixel_sha256": digest.hexdigest(), "pts_sha256": _json_hash(pts_payload)}),
    }


def read_json_probe(path: Path) -> dict[str, Any]:
    from .common import ffprobe_json

    return ffprobe_json(path)


def first_video_frame_timestamps(path: Path) -> dict[str, Any]:
    """Read the first decoded video-frame timestamps, rather than stream hints."""

    executable = executable_path(config.FFPROBE, "ffprobe")
    result = run_command(
        (
            str(executable),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-read_intervals",
            "%+#1",
            "-show_entries",
            "frame=best_effort_timestamp,best_effort_timestamp_time,pts,pts_time",
            "-of",
            "json",
            str(path),
        )
    )
    try:
        value = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"ffprobe frame timestamp output is malformed: {path}") from exc
    frames = value.get("frames") if isinstance(value, Mapping) else None
    if not isinstance(frames, list) or not frames or not isinstance(frames[0], Mapping):
        raise ProtocolError(f"ffprobe returned no first video frame timestamp: {path}")
    return dict(frames[0])


def _load_syncnet_classes() -> Any:
    from SyncNetInstance import SyncNetInstance

    return SyncNetInstance


class SyncNetEngine:
    """One serial official SyncNet model instance for a whole stage."""

    def __init__(self, *, model_path: Path = config.SYNCNET_MODEL, batch_size: int = config.SYNCNET_BATCH_SIZE, device: str = "cuda") -> None:
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ProtocolError("RESOURCE_WAIT: CUDA is unavailable for SyncNet")
        if not model_path.is_file():
            raise ProtocolError(f"SyncNet checkpoint is missing: {model_path}")
        SyncNetInstance = _load_syncnet_classes()
        self.device = device
        self.batch_size = int(batch_size)
        self.model_path = model_path.resolve()
        self.model = SyncNetInstance(device=device)
        self.model.loadParameters(str(self.model_path))
        self.model.eval()
        # The reference wrapper exposes evaluate/extract_feature, while the
        # two modality forwards live on its registered ``S`` submodule.
        self.network = self.model.__S__
        self.network.eval()
        self.model_hash = file_sha256(self.model_path)
        self.code_hash = canonical_sha256({
            "SyncNetInstance.py": file_sha256(config.SYNCNET_INSTANCE),
            "SyncNetModel.py": file_sha256(config.SYNCNET_DEFINITION),
            "worker_module": file_sha256(Path(__file__)),
        })
        self._torch = torch

    def close(self) -> None:
        self.network = None  # type: ignore[assignment]
        self.model = None  # type: ignore[assignment]
        if self.device.startswith("cuda"):
            self._torch.cuda.empty_cache()

    def _visual_batch(self, windows: Sequence[np.ndarray]) -> np.ndarray:
        if not windows:
            raise ProtocolError("empty visual batch")
        stacked = np.stack(windows, axis=0)
        if stacked.ndim != 5 or stacked.shape[1] != 5 or stacked.shape[4] != 3:
            raise ProtocolError(f"unexpected visual window shape: {stacked.shape}")
        return np.transpose(stacked, (0, 4, 1, 2, 3)).astype(np.float32, copy=False)

    @staticmethod
    def _stream_mjpeg(video_path: str | Path) -> Any:
        """Yield the historical worker's JPEG-decoded frames without disk files."""

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise ProtocolError("opencv is required for SyncNet") from exc
        executable = executable_path(config.FFMPEG, "ffmpeg")
        command = [
            str(executable),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-threads",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            raise ProtocolError(f"cannot start ffmpeg visual decoder: {video_path}") from exc
        if process.stdout is None or process.stderr is None:  # pragma: no cover
            process.kill()
            raise ProtocolError("ffmpeg visual decoder did not expose pipes")
        buffer = bytearray()
        try:
            while True:
                chunk = process.stdout.read(1 << 20)
                if not chunk:
                    break
                buffer.extend(chunk)
                while True:
                    start = buffer.find(b"\xff\xd8")
                    if start < 0:
                        # Retain a possible marker prefix split across reads.
                        del buffer[:-1]
                        break
                    if start:
                        del buffer[:start]
                    end = buffer.find(b"\xff\xd9", 2)
                    if end < 0:
                        break
                    encoded = np.frombuffer(bytes(buffer[: end + 2]), dtype=np.uint8)
                    del buffer[: end + 2]
                    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if frame is None:
                        process.kill()
                        raise ProtocolError(f"ffmpeg emitted an undecodable JPEG frame: {video_path}")
                    yield frame
            stderr = process.stderr.read().decode("utf-8", errors="replace")
            return_code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.kill()
            process.wait()
            raise
        if return_code != 0:
            raise ProtocolError(f"ffmpeg visual decode failed for {video_path}: {stderr[-1000:]}")
        if buffer:
            raise ProtocolError(f"ffmpeg emitted an incomplete JPEG stream: {video_path}")

    def extract_visual(self, video_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
        """Run ``forward_lip`` on historical JPEG-decoded frames in a stream."""

        outputs: list[np.ndarray] = []
        buffer: list[np.ndarray] = []
        frame_count = 0
        with self._torch.inference_mode():
            for frame in self._stream_mjpeg(video_path):
                frame_count += 1
                buffer.append(frame)
                if len(buffer) >= self.batch_size + 4:
                    windows = [buffer[index : index + 5] for index in range(self.batch_size)]
                    tensor = self._torch.from_numpy(self._visual_batch(windows)).to(self.device)
                    output = self.network.forward_lip(tensor).detach().cpu().numpy().astype(np.float32)
                    outputs.append(output)
                    buffer = buffer[self.batch_size :]
            if len(buffer) >= 5:
                windows = [buffer[index : index + 5] for index in range(len(buffer) - 4)]
                tensor = self._torch.from_numpy(self._visual_batch(windows)).to(self.device)
                outputs.append(self.network.forward_lip(tensor).detach().cpu().numpy().astype(np.float32))
        if not outputs:
            raise ProtocolError(f"SyncNet visual extraction produced no windows: {video_path}")
        features = np.concatenate(outputs, axis=0)
        if features.ndim != 2 or features.shape[1] != config.EMBEDDING_DIM or not np.isfinite(features).all():
            raise ProtocolError(f"invalid visual feature shape: {features.shape}")
        return features, {"frame_count": frame_count, "window_count": int(features.shape[0]), "window_length_frames": 5, "forward": "forward_lip", "device": self.device, "visual_decode_mode": config.SYNCNET_VISUAL_DECODE_MODE, "frames_streamed": True, "frame_files_written": False}

    def _audio_windows(self, pcm: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        try:
            import python_speech_features
        except ImportError as exc:  # pragma: no cover
            raise ProtocolError("python_speech_features is required for SyncNet") from exc
        if pcm.dtype != np.int16 or pcm.ndim != 1 or pcm.size < config.SAMPLES_PER_FRAME * 6:
            raise ProtocolError("audio feature input violates PCM16 contract")
        mfcc = np.asarray(python_speech_features.mfcc(pcm, config.SAMPLE_RATE), dtype=np.float64)
        if mfcc.ndim != 2 or mfcc.shape[1] != 13 or not np.isfinite(mfcc).all():
            raise ProtocolError(f"invalid MFCC shape: {mfcc.shape}")
        count_by_clock = int(pcm.size // 640) - 5
        count_by_mfcc = int((mfcc.shape[0] - 20) // 4 + 1)
        count = min(count_by_clock, count_by_mfcc)
        if count < 1:
            raise ProtocolError(f"audio has no valid SyncNet windows: {pcm.size} samples, {mfcc.shape[0]} MFCC frames")
        windows = np.stack([mfcc[index * 4 : index * 4 + 20].T for index in range(count)], axis=0)
        return windows.astype(np.float32), {
            "sample_count": int(pcm.size),
            "mfcc_shape": [int(item) for item in mfcc.shape],
            "mfcc_layout": "[13,frames] after official transpose",
            "mfcc_library": "python_speech_features.mfcc default parameters",
            "window_count": count,
            "window_length_mfcc": 20,
            "window_stride_mfcc": 4,
            "forward": "forward_aud",
        }

    def extract_audio(self, wav_path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
        pcm = read_pcm16_wav(wav_path)
        windows, metadata = self._audio_windows(pcm)
        outputs: list[np.ndarray] = []
        with self._torch.inference_mode():
            for start in range(0, windows.shape[0], self.batch_size):
                batch = self._torch.from_numpy(windows[start : start + self.batch_size])[:, None, :, :].to(self.device)
                outputs.append(self.network.forward_aud(batch).detach().cpu().numpy().astype(np.float32))
        features = np.concatenate(outputs, axis=0)
        if features.ndim != 2 or features.shape[1] != config.EMBEDDING_DIM or not np.isfinite(features).all():
            raise ProtocolError(f"invalid audio feature shape: {features.shape}")
        metadata["feature_count"] = int(features.shape[0])
        metadata["pcm_sha256"] = hashlib.sha256(np.asarray(pcm, dtype="<i2").tobytes()).hexdigest()
        return features, metadata

    @staticmethod
    def distance_matrix(video_features: np.ndarray, audio_features: np.ndarray) -> np.ndarray:
        """Reproduce official ``calc_pdist`` with float32 and eps=1e-6."""

        import torch

        visual = np.asarray(video_features, dtype=np.float32)
        audio = np.asarray(audio_features, dtype=np.float32)
        if visual.ndim != 2 or audio.ndim != 2 or visual.shape[1] != config.EMBEDDING_DIM or audio.shape[1] != config.EMBEDDING_DIM:
            raise ProtocolError(f"embedding dimensions are invalid: {visual.shape}/{audio.shape}")
        row_count = min(visual.shape[0], audio.shape[0])
        if row_count < 1:
            raise ProtocolError("no common embedding rows")
        visual_tensor = torch.from_numpy(visual[:row_count])
        audio_tensor = torch.from_numpy(audio[:row_count])
        padded = torch.nn.functional.pad(audio_tensor, (0, 0, config.VSHIFT, config.VSHIFT))
        rows: list[np.ndarray] = []
        with torch.inference_mode():
            for index in range(row_count):
                left = visual_tensor[index : index + 1].expand(config.LAG_COUNT, -1)
                right = padded[index : index + config.LAG_COUNT]
                rows.append(torch.nn.functional.pairwise_distance(left, right, eps=1e-6).numpy().astype(np.float64))
        matrix = np.stack(rows, axis=0)
        if matrix.shape != (row_count, config.LAG_COUNT) or not np.isfinite(matrix).all():
            raise ProtocolError(f"distance matrix is invalid: {matrix.shape}")
        return matrix

    @staticmethod
    def unit_features(features: np.ndarray) -> np.ndarray:
        value = np.asarray(features, dtype=np.float32)
        norms = np.linalg.norm(value, axis=1)
        if np.any(norms <= 0.0) or not np.isfinite(norms).all():
            raise ProtocolError("unit-norm diagnostic encountered a zero feature")
        return value / norms[:, None]


def _selection_from_pipeline(work_dir: Path, reference: str, destination: Path) -> dict[str, Any]:
    track_path = work_dir / "pywork" / reference / "tracks.pckl"
    crop_dir = work_dir / "pycrop" / reference
    if not track_path.is_file():
        raise ProtocolError(f"official face tracks are missing: {track_path}")
    try:
        with track_path.open("rb") as handle:
            tracks = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError) as exc:
        raise ProtocolError(f"official face tracks cannot be read: {track_path}") from exc
    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(tracks if isinstance(tracks, list) else []):
        if not isinstance(item, Mapping) or not isinstance(item.get("track"), Mapping):
            continue
        frame = np.asarray(item["track"].get("frame"), dtype=np.int64)
        crop = crop_dir / f"{index:05d}.avi"
        if frame.size <= config.OFFICIAL_PIPELINE["min_track"] or not crop.is_file():
            continue
        bbox = np.asarray(item["track"].get("bbox"), dtype=np.float64)
        candidates.append({
            "track_index": index,
            "crop_path": str(crop),
            "frame_count": int(frame.size),
            "start_frame": int(frame[0]),
            "end_frame": int(frame[-1]),
            "frame_indices": [int(value) for value in frame.tolist()],
            "bbox_sha256": hashlib.sha256(np.ascontiguousarray(bbox).tobytes()).hexdigest(),
            "crop_sha256": file_sha256(crop),
        })
    if not candidates:
        raise ProtocolError(f"official pipeline found no valid face track: {reference}")
    candidates.sort(key=lambda row: (-int(row["frame_count"]), int(row["start_frame"]), Path(str(row["crop_path"])).name))
    selected = candidates[0]
    copy_atomic(Path(str(selected["crop_path"])), destination)
    evidence = {
        "schema_version": 1,
        "selection_rule": "longest continuous track, earliest start frame, crop filename lexicographic",
        "official_pipeline_config": dict(config.OFFICIAL_PIPELINE),
        "reference": reference,
        "candidates": candidates,
        "selected": {**selected, "crop_path": str(destination), "crop_sha256": file_sha256(destination)},
    }
    write_self_hashed_json(destination.with_name("selection.json"), evidence)
    return evidence


def _make_real_crop(paths: config.RunPaths, row: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    sample_id = int(row["sample_id"])
    final_crop = paths.fixed_video / "crops" / "R" / f"{sample_id}.avi"
    selection_path = final_crop.with_name(f"{sample_id}.selection.json")
    if final_crop.is_file() and selection_path.is_file():
        try:
            return final_crop, read_self_hashed_json(selection_path)
        except ProtocolError:
            final_crop.unlink(missing_ok=True)
            selection_path.unlink(missing_ok=True)
    reference = f"tts_native_gain_real_{sample_id}"
    work_dir = paths.fixed_video / "_official" / str(sample_id)
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(config.SYNCNET_PYTHON),
        str(config.SYNCNET_PIPELINE),
        "--videofile",
        str(row["real_video"]),
        "--reference",
        reference,
        "--data_dir",
        str(work_dir),
        "--facedet_scale",
        str(config.OFFICIAL_PIPELINE["facedet_scale"]),
        "--crop_scale",
        str(config.OFFICIAL_PIPELINE["crop_scale"]),
        "--min_track",
        str(config.OFFICIAL_PIPELINE["min_track"]),
        "--frame_rate",
        str(config.OFFICIAL_PIPELINE["frame_rate"]),
        "--num_failed_det",
        str(config.OFFICIAL_PIPELINE["num_failed_det"]),
        "--min_face_size",
        str(config.OFFICIAL_PIPELINE["min_face_size"]),
        "--overwrite",
    ]
    run_monitored(command, cwd=config.SYNCNET_ROOT, log_path=work_dir / "pipeline.log", interval_seconds=30.0)
    final_crop.parent.mkdir(parents=True, exist_ok=True)
    selection = _selection_from_pipeline(work_dir, reference, final_crop)
    # The helper writes an intermediate selection file beside the crop.  Move
    # the evidence to an ID-specific name before the temporary pipeline tree
    # is removed; otherwise successive IDs would overwrite one another.
    intermediate_selection = final_crop.with_name("selection.json")
    write_self_hashed_json(selection_path, selection)
    intermediate_selection.unlink(missing_ok=True)
    shutil.rmtree(work_dir, ignore_errors=True)
    return final_crop, selection


def fixed_video_stage(paths: config.RunPaths, assets: Mapping[str, Any]) -> dict[str, Any]:
    root = paths.fixed_video
    manifest_path = root / "manifest.json"
    current_code_hash = package_code_hash()
    if manifest_path.is_file():
        try:
            cached = read_self_hashed_json(manifest_path)
            if cached.get("status") == "COMPLETE" and cached.get("code_hash") == current_code_hash and len(cached.get("videos", [])) == len(config.SAMPLE_IDS) * len(config.VIDEO_TYPES) and all(Path(str(row.get("selection", ""))).name == f"{int(row.get('sample_id'))}.selection.json" and Path(str(row.get("selection", ""))).is_file() and row.get("pts", {}).get("first_frame_pts") is not None for row in cached.get("videos", [])):
                return cached
        except ProtocolError:
            pass
    root.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    videos: list[dict[str, Any]] = []
    for row in assets.get("records", []):
        sample_id = int(row["sample_id"])
        for video_type, source in (("V_N", "N"), ("V_T", "T"), ("R", "R")):
            try:
                if video_type in {"V_N", "V_T"}:
                    source_row = row["sources"][source]
                    source_path = Path(str(source_row["crop"]["path"]))
                    selection = read_self_hashed_json(source_row["crop_selection"]["path"])
                else:
                    # Face tracking is the only new GPU operation required by
                    # A.  It is guarded by the shared lease and never chosen
                    # using a SyncNet score.
                    cached_real_crop = paths.fixed_video / "crops" / "R" / f"{sample_id}.avi"
                    cached_real_selection = cached_real_crop.with_name(f"{sample_id}.selection.json")
                    if cached_real_crop.is_file() and cached_real_selection.is_file():
                        source_path, selection = _make_real_crop(paths, row)
                    else:
                        with gpu_lease(estimated_persistent=128 << 20, estimated_temp=config.CELL_TEMP_BUDGET_BYTES):
                            source_path, selection = _make_real_crop(paths, row)
                destination = root / "crops" / video_type / f"{sample_id}.avi"
                if source_path.resolve() != destination.resolve():
                    copy_atomic(source_path, destination)
                signature = video_signature(destination)
                selection_copy = destination.with_name(f"{sample_id}.selection.json")
                if not selection_copy.is_file():
                    write_self_hashed_json(selection_copy, dict(selection))
                selection_hash = file_sha256(selection_copy)
                videos.append({
                    "id": sample_id,
                    "sample_id": sample_id,
                    "source_group": str(row["source_group"]),
                    "video_type": video_type,
                    "source": source,
                    "path": str(destination),
                    "file_sha256": file_sha256(destination),
                    "video_hash": signature["video_hash"],
                    "pixel_sha256": signature["pixel_sha256"],
                    "pts_sha256": signature["pts_sha256"],
                    "frame_count": signature["frame_count"],
                    "pts": signature["pts"],
                    "roi_hash": canonical_sha256({"selection_sha256": selection_hash, "pixel_sha256": signature["pixel_sha256"]}),
                    "selection": str(selection_copy),
                    "selection_sha256": selection_hash,
                    "status": "COMPLETE",
                })
            except (OSError, ProtocolError, KeyError, TypeError, ValueError) as exc:
                failures.append({"sample_id": sample_id, "video_type": video_type, "status": "INPUT_INVALID", "error_type": "FIXED_VIDEO_FAILED", "reason": str(exc)})
                videos.append({"id": sample_id, "sample_id": sample_id, "source_group": str(row["source_group"]), "video_type": video_type, "source": source, "path": None, "video_hash": None, "pts_sha256": None, "roi_hash": None, "status": "INPUT_INVALID", "error_type": "FIXED_VIDEO_FAILED", "reason": str(exc)})
    expected = len(config.SAMPLE_IDS) * len(config.VIDEO_TYPES)
    result = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "stage": "02_fixed_video",
        "status": "COMPLETE" if len(videos) == expected and not failures and all(row.get("status") == "COMPLETE" for row in videos) else "INCOMPLETE",
        "expected_video_count": expected,
        "video_count": len(videos),
        "videos": videos,
        "failures": failures,
        "no_score_based_track_choice": True,
        "streaming_decode": True,
        "code_hash": current_code_hash,
    }
    return write_self_hashed_json(manifest_path, result)


def _audio_record_map(audio_manifest: Mapping[str, Any]) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    result: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in audio_manifest.get("records", []):
        result[(int(row["sample_id"]), str(row["source"]), str(row["condition"]))] = row
    return result


def _video_record_map(fixed_manifest: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    return {(int(row["sample_id"]), str(row["video_type"])): row for row in fixed_manifest.get("videos", [])}


def _save_feature(path: Path, array: np.ndarray, metadata: Mapping[str, Any]) -> dict[str, Any]:
    sha = _npy_atomic(path, array)
    payload = {"path": str(path), "sha256": sha, "shape": [int(item) for item in array.shape], "dtype": str(array.dtype), **dict(metadata)}
    write_self_hashed_json(path.with_suffix(".json"), payload)
    return payload


def _load_cached_feature(path: Path, *, expected_sha: str | None = None, expected_decode_mode: str | None = None, expected_code_hash: str | None = None, expected_environment_hash: str | None = None) -> tuple[np.ndarray, dict[str, Any]] | None:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = read_self_hashed_json(metadata_path)
        if expected_sha is not None and str(metadata.get("video_hash", metadata.get("pcm_hash", ""))) != expected_sha:
            return None
        if expected_decode_mode is not None and str(metadata.get("visual_decode_mode")) != expected_decode_mode:
            return None
        if expected_code_hash is not None and str(metadata.get("code_hash")) != expected_code_hash:
            return None
        if expected_environment_hash is not None and str(metadata.get("environment_hash")) != expected_environment_hash:
            return None
        if file_sha256(path) != str(metadata.get("sha256")):
            return None
        value = np.load(path, allow_pickle=False)
        if value.ndim != 2 or value.shape[1] != config.EMBEDDING_DIM or not np.isfinite(value).all():
            return None
        return np.asarray(value, dtype=np.float32), metadata
    except (OSError, ValueError, ProtocolError):
        return None


def _shift_pcm(values: np.ndarray, shift_samples: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.int16)
    if array.ndim != 1 or array.size == 0:
        raise ProtocolError("cannot shift an empty PCM array")
    result = np.zeros_like(array)
    if shift_samples >= 0:
        if shift_samples < array.size:
            result[shift_samples:] = array[: array.size - shift_samples]
    else:
        amount = -int(shift_samples)
        if amount < array.size:
            result[: array.size - amount] = array[amount:]
    return result


def _support_hash(rows: Sequence[int]) -> str:
    return canonical_sha256({"support": [int(item) for item in rows], "rule": "common INTERIOR after vshift"})


def _v15_replay_summary(assets: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare the new ORIGINAL cells with the frozen v15 matrices."""

    cell_map = {(int(row["id"]), str(row["video_type"]), str(row["eval_condition"])): row for row in cells}
    asset_map = {int(row["sample_id"]): row for row in assets.get("records", [])}
    comparisons: list[dict[str, Any]] = []
    tolerance = 1e-4
    for sample_id in config.SAMPLE_IDS:
        asset = asset_map.get(sample_id, {})
        for source, video_type in (("N", "V_N"), ("T", "V_T")):
            historical = Path(str(asset.get("sources", {}).get(source, {}).get("matrix", {}).get("path", "")))
            current_row = cell_map.get((sample_id, video_type, "ORIGINAL"))
            if current_row is None or not historical.is_file():
                comparisons.append({"sample_id": sample_id, "source": source, "status": "MISSING"})
                continue
            old = np.asarray(np.load(historical, allow_pickle=False), dtype=np.float64)
            new = np.asarray(np.load(str(current_row["matrix_path"]), allow_pickle=False), dtype=np.float64)
            if old.shape != new.shape:
                comparisons.append({"sample_id": sample_id, "source": source, "status": "SHAPE_MISMATCH", "historical_shape": [int(x) for x in old.shape], "current_shape": [int(x) for x in new.shape]})
                continue
            maximum = float(np.max(np.abs(old - new)))
            comparisons.append({"sample_id": sample_id, "source": source, "status": "PASS" if maximum <= tolerance else "MISMATCH", "max_abs_error": maximum, "historical_shape": [int(x) for x in old.shape]})
    maximum = max((float(row.get("max_abs_error", float("inf"))) for row in comparisons), default=float("inf"))
    return {"status": "PASS" if len(comparisons) == len(config.SAMPLE_IDS) * 2 and all(row.get("status") == "PASS" for row in comparisons) else "FAIL", "tolerance_max_abs": tolerance, "comparison_count": len(comparisons), "max_abs_error": maximum, "comparisons": comparisons, "historical_worker_matrices_are_context_only": True}


def _control_result(baseline: np.ndarray, candidate: np.ndarray, *, shift_name: str, support: Sequence[int]) -> dict[str, Any]:
    from .analysis import curve_metrics

    base = curve_metrics(baseline, support)
    current = curve_metrics(candidate, support)
    if shift_name == "IDENTITY":
        max_error = float(np.max(np.abs(baseline[np.asarray(support)] - candidate[np.asarray(support)])))
        status = "IDENTITY_PASS" if max_error <= 1e-5 and int(base["official_offset"]) == int(current["official_offset"]) else "CONTROL_FAILED"
        return {"status": status, "max_abs_matrix_error": max_error, "baseline": base, "candidate": current}
    expected = -5 if shift_name == "PLUS_200MS" else 5
    delta = int(current["official_offset"]) - int(base["official_offset"])
    baseline_column = int(base["min_index"])
    base_distance = float(np.mean(baseline[np.asarray(support), baseline_column]))
    shifted_distance = float(np.mean(candidate[np.asarray(support), baseline_column]))
    # ``official_offset = VSHIFT - min_index``.  A requested offset delta of
    # -5 therefore means the expected minimum column moves by +5.
    expected_min_index_delta = -expected
    boundary = bool(base["min_index"] in (0, config.LAG_COUNT - 1) or int(base["min_index"]) + expected_min_index_delta < 0 or int(base["min_index"]) + expected_min_index_delta >= config.LAG_COUNT)
    if boundary:
        status = "CONTROL_UNINFORMATIVE"
    elif abs(delta - expected) <= 1 and shifted_distance > base_distance:
        status = "DELAY_DETECTED"
    else:
        status = "CONTROL_FAILED"
    return {"status": status, "expected_offset_delta": expected, "observed_offset_delta": delta, "baseline_column": baseline_column, "baseline_column_distance": base_distance, "shifted_column_distance": shifted_distance, "baseline": base, "candidate": current}


def score_a_stage(paths: config.RunPaths, assets: Mapping[str, Any], audio_manifest: Mapping[str, Any], fixed_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Run A with one visual forward per fixed crop and one audio forward per WAV."""

    root = paths.fixed_video
    manifest_path = root / "a_manifest.json"
    current_code_hash = package_code_hash()
    current_environment = scoring_environment()
    current_environment_hash = canonical_sha256(current_environment)
    if manifest_path.is_file():
        try:
            cached = read_self_hashed_json(manifest_path)
            if cached.get("status") == "COMPLETE" and cached.get("visual_decode_mode") == config.SYNCNET_VISUAL_DECODE_MODE and cached.get("code_hash") == current_code_hash and cached.get("execution_environment_hash") == current_environment_hash and cached.get("audio_feature_aliases_complete") is True and len(cached.get("cells", [])) == config.EXPECTED_A_SCIENCE and len(cached.get("controls", [])) == config.EXPECTED_A_CONTROLS:
                return cached
        except ProtocolError:
            pass
    root.mkdir(parents=True, exist_ok=True)
    feature_root = root / "features"
    matrix_root = root / "matrices"
    cells: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    videos = _video_record_map(fixed_manifest)
    audios = _audio_record_map(audio_manifest)
    source_for_type = {"V_N": "N", "V_T": "T", "R": "R"}
    visual_features: dict[tuple[int, str], np.ndarray] = {}
    audio_features: dict[tuple[int, str, str], np.ndarray] = {}
    matrices: dict[tuple[int, str, str], np.ndarray] = {}
    visual_meta: dict[tuple[int, str], Mapping[str, Any]] = {}
    audio_meta: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    model_forward_counts = {"visual": 0, "audio": 0, "shifted_audio": 0}
    cache_counts = {"visual": 0, "audio": 0, "audio_alias": 0}

    cached_feature_count = len(list((feature_root / "visual").glob("*/*.npy"))) + len(list((feature_root / "audio").glob("*/*/*.npy")))
    # On a resumed stage only missing matrices/controls are persistent output;
    # charging the full first-run feature budget would incorrectly prevent a
    # valid cache completion on a nearly-full but still safe disk.
    persistent_estimate = 32 << 20 if cached_feature_count >= 180 else 384 << 20
    with gpu_lease(estimated_persistent=persistent_estimate, estimated_temp=config.SYNCNET_CELL_TEMP_BUDGET_BYTES):
        engine = SyncNetEngine()
        try:
            for sample_id in config.SAMPLE_IDS:
                for video_type in config.VIDEO_TYPES:
                    video_row = videos.get((sample_id, video_type))
                    if not video_row or video_row.get("status") != "COMPLETE":
                        failures.append({"sample_id": sample_id, "video_type": video_type, "error_type": "VIDEO_MISSING", "reason": "fixed crop unavailable"})
                        continue
                    path = Path(str(video_row["path"]))
                    cached_feature = _load_cached_feature(feature_root / "visual" / video_type / f"{sample_id}.npy", expected_sha=str(video_row["video_hash"]), expected_decode_mode=config.SYNCNET_VISUAL_DECODE_MODE, expected_code_hash=current_code_hash, expected_environment_hash=current_environment_hash)
                    if cached_feature is None:
                        vf, vm = engine.extract_visual(path)
                        model_forward_counts["visual"] += 1
                        visual_meta[(sample_id, video_type)] = _save_feature(feature_root / "visual" / video_type / f"{sample_id}.npy", vf, {"video_hash": video_row["video_hash"], "video_path": str(path), "code_hash": current_code_hash, "environment_hash": current_environment_hash, **vm})
                    else:
                        cache_counts["visual"] += 1
                        vf, vm = cached_feature
                        visual_meta[(sample_id, video_type)] = vm
                    visual_features[(sample_id, video_type)] = vf
                for source in config.SOURCES:
                    for condition in config.AUDIO_CONDITIONS:
                        ar = audios.get((sample_id, source, condition))
                        if not ar:
                            failures.append({"sample_id": sample_id, "source": source, "condition": condition, "error_type": "AUDIO_MISSING", "reason": "audio manifest row unavailable"})
                            continue
                        # A0 and ORIGINAL are separate logical conditions but
                        # may share actual PCM/features under the common g rule.
                        feature_path = feature_root / "audio" / source / str(sample_id) / f"{condition}.npy"
                        prior = next((key for key, value in audio_features.items() if key[0] == sample_id and key[1] == source and audios[key]["pcm_sha256"] == ar["pcm_sha256"]), None)
                        if prior is not None:
                            cache_counts["audio_alias"] += 1
                            audio_features[(sample_id, source, condition)] = audio_features[prior]
                            prior_meta = dict(audio_meta[prior])
                            prior_path = Path(str(prior_meta["path"]))
                            if prior_path.resolve() != feature_path.resolve():
                                copy_atomic(prior_path, feature_path)
                            alias_meta = {**prior_meta, "path": str(feature_path), "sha256": file_sha256(feature_path), "reused_from": list(prior)}
                            write_self_hashed_json(feature_path.with_suffix(".json"), alias_meta)
                            audio_meta[(sample_id, source, condition)] = alias_meta
                            continue
                        cached_feature = _load_cached_feature(feature_path, expected_sha=str(ar["pcm_sha256"]), expected_code_hash=current_code_hash, expected_environment_hash=current_environment_hash)
                        if cached_feature is not None:
                            cache_counts["audio"] += 1
                            af, am = cached_feature
                        else:
                            af, am = engine.extract_audio(Path(str(ar["path"])))
                            model_forward_counts["audio"] += 1
                        audio_features[(sample_id, source, condition)] = af
                        audio_meta[(sample_id, source, condition)] = am if cached_feature is not None else _save_feature(feature_path, af, {"audio_path": str(ar["path"]), "pcm_hash": ar["pcm_sha256"], "code_hash": current_code_hash, "environment_hash": current_environment_hash, **am})
        finally:
            engine.close()

    # Matrix calculations are CPU-only after the model forwards.  This keeps
    # GPU residency short and makes the 180 logical cells cheap to materialise.
    for sample_id in config.SAMPLE_IDS:
        for video_type in config.VIDEO_TYPES:
            source = source_for_type[video_type]
            vf = visual_features.get((sample_id, video_type))
            if vf is None:
                continue
            condition_matrices: dict[str, np.ndarray] = {}
            for condition in config.AUDIO_CONDITIONS:
                af = audio_features.get((sample_id, source, condition))
                if af is not None:
                    condition_matrices[condition] = SyncNetEngine.distance_matrix(vf, af)
                    matrices[(sample_id, video_type, condition)] = condition_matrices[condition]
            if len(condition_matrices) != len(config.AUDIO_CONDITIONS):
                continue
            common_count = min(value.shape[0] for value in condition_matrices.values())
            support = list(range(config.VSHIFT, common_count - config.VSHIFT))
            support_status = "PASS" if len(support) >= config.MIN_INTERIOR_ROWS else "INCOMPLETE"
            support_hash = _support_hash(support)
            base_key: dict[str, str] = {}
            for condition in config.AUDIO_CONDITIONS:
                matrix = condition_matrices[condition]
                path = matrix_root / video_type / str(sample_id) / f"{condition}.npy"
                reused_from = None
                if condition == "ORIGINAL" and np.array_equal(matrix, condition_matrices["A0"]):
                    reused_from = f"A:{sample_id}:{video_type}:A0"
                    a0_path = matrix_root / video_type / str(sample_id) / "A0.npy"
                    # A previous run may have left a stale ORIGINAL file from
                    # a different decoder.  Materialise the alias from the
                    # current A0 matrix before recording its hash.
                    if a0_path.is_file():
                        copy_atomic(a0_path, path)
                        matrix_hash = file_sha256(path)
                    else:
                        matrix_hash = _npy_atomic(path, matrix)
                else:
                    matrix_hash = _npy_atomic(path, matrix)
                base_key[condition] = str(path)
                row = {
                    "stage": "A",
                    "id": sample_id,
                    "source_group": next(str(item["source_group"]) for item in assets["records"] if int(item["sample_id"]) == sample_id),
                    "source": source,
                    "video_type": video_type,
                    "driver_condition": "FIXED_VIDEO",
                    "eval_condition": condition,
                    "seed": None,
                    "video_hash": videos[(sample_id, video_type)]["video_hash"],
                    "pcm_hash": audios[(sample_id, source, condition)]["pcm_sha256"],
                    "roi_hash": videos[(sample_id, video_type)]["roi_hash"],
                    "support_hash": support_hash if support_status == "PASS" else None,
                    "support_rows": support,
                    "model_hash": engine_hash_from_files(),
                    "code_hash": package_code_hash(),
                    "matrix_path": str(path),
                    "matrix_hash": matrix_hash,
                    "matrix_shape": [int(item) for item in matrix.shape],
                    "reused_from": reused_from,
                    "status": "COMPLETE" if support_status == "PASS" else "INCOMPLETE",
                }
                cells.append(row)
            if support_status != "PASS":
                failures.append({"sample_id": sample_id, "video_type": video_type, "error_type": "SUPPORT_TOO_SHORT", "reason": f"common INTERIOR rows={len(support)}"})

    # Controls use freshly shifted PCM and freshly recomputed audio features.
    shifted_features: dict[tuple[int, str, str], np.ndarray] = {}
    with gpu_lease(estimated_persistent=8 << 20, estimated_temp=config.SYNCNET_CELL_TEMP_BUDGET_BYTES):
        engine = SyncNetEngine()
        try:
            for sample_id in config.CONTROL_IDS:
                for source in config.SOURCES:
                    base_audio = read_pcm16_wav(Path(str(audios[(sample_id, source, "A0")]["path"])))
                    for shift_name, shift_samples in (("PLUS_200MS", 3200), ("MINUS_200MS", -3200)):
                        shifted = _shift_pcm(base_audio, shift_samples)
                        shifted_path = root / "controls" / str(sample_id) / source / f"{shift_name}.wav"
                        write_pcm16_wav(shifted_path, shifted)
                        af, am = engine.extract_audio(shifted_path)
                        model_forward_counts["shifted_audio"] += 1
                        shifted_features[(sample_id, source, shift_name)] = af
                        _save_feature(feature_root / "controls" / source / str(sample_id) / f"{shift_name}.npy", af, {"pcm_hash": hashlib.sha256(np.asarray(shifted, dtype="<i2").tobytes()).hexdigest(), "shift_samples": shift_samples, "code_hash": current_code_hash, "environment_hash": current_environment_hash, **am})
        finally:
            engine.close()
    for sample_id in config.CONTROL_IDS:
        for video_type in config.VIDEO_TYPES:
            source = source_for_type[video_type]
            vf = visual_features.get((sample_id, video_type))
            if vf is None:
                continue
            base = matrices.get((sample_id, video_type, "A0"))
            if base is None:
                continue
            max_count = min(base.shape[0], *(shifted_features[(sample_id, source, name)].shape[0] for name in ("PLUS_200MS", "MINUS_200MS")))
            support = list(range(config.VSHIFT + 5, max_count - config.VSHIFT - 5))
            support_hash = _support_hash(support) if len(support) >= config.MIN_INTERIOR_ROWS else None
            for shift_name in config.CONTROL_SHIFTS:
                if shift_name == "IDENTITY":
                    matrix = base
                    pcm_hash = audios[(sample_id, source, "A0")]["pcm_sha256"]
                    reused_from = f"A:{sample_id}:{video_type}:A0"
                else:
                    matrix = SyncNetEngine.distance_matrix(vf, shifted_features[(sample_id, source, shift_name)])
                    pcm_hash = hashlib.sha256(np.asarray(_shift_pcm(read_pcm16_wav(Path(str(audios[(sample_id, source, "A0")]["path"]))), 3200 if shift_name == "PLUS_200MS" else -3200), dtype="<i2").tobytes()).hexdigest()
                    reused_from = None
                path = matrix_root / "controls" / video_type / str(sample_id) / f"{shift_name}.npy"
                matrix_hash = _npy_atomic(path, matrix)
                status_payload = {"status": "CONTROL_UNINFORMATIVE" if len(support) < config.MIN_INTERIOR_ROWS else "CONTROL_FAILED", "reason": "common control support below 25"} if len(support) < config.MIN_INTERIOR_ROWS else _control_result(base, matrix, shift_name=shift_name, support=support)
                controls.append({
                    "stage": "A_CONTROL",
                    "id": sample_id,
                    "source_group": next(str(item["source_group"]) for item in assets["records"] if int(item["sample_id"]) == sample_id),
                    "source": source,
                    "video_type": video_type,
                    "driver_condition": "FIXED_VIDEO",
                    "eval_condition": shift_name,
                    "seed": None,
                    "video_hash": videos[(sample_id, video_type)]["video_hash"],
                    "pcm_hash": pcm_hash,
                    "roi_hash": videos[(sample_id, video_type)]["roi_hash"],
                    "support_hash": support_hash,
                    "support_rows": support,
                    "model_hash": engine_hash_from_files(),
                    "code_hash": package_code_hash(),
                    "matrix_path": str(path),
                    "matrix_hash": matrix_hash,
                    "matrix_shape": [int(item) for item in matrix.shape],
                    "reused_from": reused_from,
                    "control": status_payload,
                    "status": "COMPLETE" if status_payload["status"] in {"IDENTITY_PASS", "DELAY_DETECTED", "CONTROL_UNINFORMATIVE", "CONTROL_FAILED"} else "INCOMPLETE",
                })

    replay = _v15_replay_summary(assets, cells)
    if replay.get("status") != "PASS":
        failures.append({"error_type": "V15_REPLAY_MISMATCH", "reason": "new ORIGINAL matrices exceed the frozen replay tolerance", "replay": replay})
    result = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "stage": "02_fixed_video_scoring",
        "status": "COMPLETE" if len(cells) == config.EXPECTED_A_SCIENCE and len(controls) == config.EXPECTED_A_CONTROLS and not failures and all(row["status"] == "COMPLETE" for row in cells + controls) else "INCOMPLETE",
        "expected_science_cell_count": config.EXPECTED_A_SCIENCE,
        "expected_control_cell_count": config.EXPECTED_A_CONTROLS,
        "science_cell_count": len(cells),
        "control_cell_count": len(controls),
        "cells": cells,
        "controls": controls,
        "failures": failures,
        "feature_forward_counts": {"visual": len(visual_features), "audio": len(audio_features), "shifted_audio": len(shifted_features)},
        "model_forward_counts": model_forward_counts,
        "feature_cache_counts": cache_counts,
        "visual_decode_mode": config.SYNCNET_VISUAL_DECODE_MODE,
        "visual_frames_streamed": True,
        "visual_frame_files_written": False,
        "audio_feature_aliases_complete": True,
        "execution_environment": current_environment,
        "execution_environment_hash": current_environment_hash,
        "v15_replay": replay,
        "distance_matrices_are_cpu_post_forward": True,
        "model_hash": engine_hash_from_files(),
        "code_hash": package_code_hash(),
    }
    return write_self_hashed_json(manifest_path, result)


def engine_hash_from_files() -> str:
    return canonical_sha256({"model": file_sha256(config.SYNCNET_MODEL), "definition": file_sha256(config.SYNCNET_DEFINITION), "instance": file_sha256(config.SYNCNET_INSTANCE)})


def package_code_hash() -> str:
    # Feature/matrix caches depend on the extraction contract, not on the
    # report, validator, or perception modules.  Keeping this stage-local
    # prevents an editorial or audit-only edit from needlessly launching 36
    # visual and 144 audio forwards again.
    relevant = (config.REPO / "scripts/experiments/tts_native_gain_attribution/config.py", config.REPO / "scripts/experiments/tts_native_gain_attribution/common.py", config.REPO / "scripts/experiments/tts_native_gain_attribution/audio.py", Path(__file__))
    return canonical_sha256({str(path.name): file_sha256(path) for path in relevant if path.is_file()})
