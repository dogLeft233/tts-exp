"""Reference preparation and variable-length TFG adapters."""

from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .protocol import PROTOCOL_ID, ProtocolError, file_sha256, read_json, validate_receipt, write_json

_MEDIAPIPE_DETECTORS: dict[str, Any] = {}


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _load_first_frame(path: Path) -> np.ndarray:
    """Read exactly one external visual frame; never expose a source timeline."""
    if any("lrs3" in part.lower() for part in path.resolve().parts):
        raise ProtocolError(f"LRS3 visual input is forbidden: {path}")
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg"}:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ProtocolError(f"cannot read external visual image: {path}")
    else:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ProtocolError(f"cannot open external visual video: {path}")
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise ProtocolError(f"external visual video has no first frame: {path}")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ProtocolError(f"invalid external visual frame: {path}")
    return np.ascontiguousarray(frame)


def _haar_box(frame: np.ndarray) -> list[int] | None:
    if not hasattr(cv2, "CascadeClassifier"):
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, width, height = max(faces, key=lambda value: int(value[2]) * int(value[3]))
    return [int(y), int(y + height), int(x), int(x + width)]


def _mediapipe_box(frame: np.ndarray, model_path: Path | None) -> list[int] | None:
    if model_path is None or not model_path.is_file():
        raise ProtocolError(f"MediaPipe face detector model is missing: {model_path}")
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:  # pragma: no cover - environment-specific deployment failure
        raise ProtocolError("detector=mediapipe requires the mediapipe package") from exc
    key = str(model_path.resolve())
    detector = _MEDIAPIPE_DETECTORS.get(key)
    if detector is None:
        options = vision.FaceDetectorOptions(
            base_options=python.BaseOptions(model_asset_path=key),
            min_detection_confidence=0.5,
        )
        detector = vision.FaceDetector.create_from_options(options)
        _MEDIAPIPE_DETECTORS[key] = detector
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(image)
    detections = list(result.detections or [])
    if not detections:
        return None
    height, width = frame.shape[:2]
    candidates: list[list[int]] = []
    for detection in detections:
        box = detection.bounding_box
        left = max(0, int(box.origin_x))
        top = max(0, int(box.origin_y))
        right = min(width, int(box.origin_x + box.width))
        bottom = min(height, int(box.origin_y + box.height))
        if left < right and top < bottom:
            candidates.append([top, bottom, left, right])
    return max(candidates, key=lambda value: (value[1] - value[0]) * (value[3] - value[2])) if candidates else None


def prepare_reference(
    visual_path: Path,
    output_dir: Path,
    sample_id: str,
    *,
    detector: str = "haar",
    cohort_source_video: str | None = None,
    detector_model: str | None = None,
) -> dict[str, Any]:
    """Prepare one frozen frame from an external non-LRS3 visual asset."""
    frame = _load_first_frame(visual_path.resolve())
    height, width = frame.shape[:2]
    if (height, width) == (224, 224):
        box = [0, 224, 0, 224]
        strategy = "full_224_external_first_frame"
    elif detector == "haar":
        box = _haar_box(frame)
        if box is None:
            raise ProtocolError(f"no face detected in external first frame: {visual_path}")
        strategy = "external_first_frame_haar"
    elif detector == "mediapipe":
        model_path = Path(detector_model).resolve() if detector_model else None
        box = _mediapipe_box(frame, model_path)
        if box is None:
            raise ProtocolError(f"no face detected in external first frame: {visual_path}")
        strategy = "external_first_frame_mediapipe"
    else:
        raise ProtocolError(f"unsupported reference detector: {detector}")
    top, bottom, left, right = box
    if not (0 <= top < bottom <= height and 0 <= left < right <= width):
        raise ProtocolError(f"invalid reference box {box} for {visual_path}")
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        raise ProtocolError(f"empty reference crop: {visual_path}")
    crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = output_dir / "reference.png"
    crop_path = output_dir / "crop.png"
    if not cv2.imwrite(str(reference_path), frame):
        raise ProtocolError(f"failed to write reference frame: {reference_path}")
    if not cv2.imwrite(str(crop_path), crop):
        raise ProtocolError(f"failed to write reference crop: {crop_path}")
    receipt = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "sample_id": sample_id,
        "status": "ready",
        "visual_source": str(visual_path.resolve()),
        "visual_source_sha256": file_sha256(visual_path),
        "visual_source_kind": "image" if visual_path.suffix.lower() in {".png", ".jpg", ".jpeg"} else "video",
        "visual_source_frame_policy": "first_frame_only",
        "visual_source_used_for_generation": "one_frame_repeated_for_each_mel_chunk",
        "cohort_source_video": cohort_source_video,
        "frame_idx": 0,
        "frame_count": 1,
        "frame_shape": list(frame.shape),
        "strategy": strategy,
        "detector": detector,
        "detector_model": str(Path(detector_model).resolve()) if detector_model else None,
        "detector_model_sha256": file_sha256(Path(detector_model).resolve()) if detector_model else None,
        "box_top_bottom_left_right": box,
        "reference": str(reference_path.resolve()),
        "reference_sha256": file_sha256(reference_path),
        "crop": str(crop_path.resolve()),
        "crop_sha256": file_sha256(crop_path),
        "crop_shape": list(crop.shape),
        "crop_interpolation": "cv2.INTER_AREA",
        "frame_hash": _hash_array(frame),
    }
    write_json(output_dir / "receipt.json", receipt)
    return receipt


def prepare_references(
    repo_root: Path,
    run_dir: Path,
    audit_rows: Sequence[Mapping[str, Any]],
    *,
    detector: str = "haar",
    detector_model: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for audit in audit_rows:
        sid = str(audit["sample_id"])
        target = run_dir / "02_reference" / sid
        existing = target / "receipt.json"
        if existing.is_file():
            try:
                receipt = read_json(existing)
                if receipt.get("visual_source_sha256") == audit.get("visual_source_sha256_actual") and receipt.get("status") in {"ready", "fallback_ready"}:
                    rows.append(receipt)
                    continue
            except (OSError, ValueError):
                pass
        try:
            receipt = prepare_reference(
                Path(str(audit["visual_source"])),
                target,
                sid,
                detector=detector,
                detector_model=detector_model,
                cohort_source_video=str(audit.get("source_video", "")),
            )
        except Exception as exc:  # record eligibility; do not renumber IDs
            receipt = {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "sample_id": sid,
                "source_group": str(audit.get("source_group", "")),
                "status": "ineligible",
                "reason": f"{type(exc).__name__}:{exc}",
                "visual_source": str(audit.get("visual_source", "")),
                "visual_source_sha256": audit.get("visual_source_sha256_actual"),
                "cohort_source_video": str(audit.get("source_video", "")),
            }
            write_json(target / "receipt.json", receipt)
        receipt["source_group"] = str(audit.get("source_group", ""))
        rows.append(receipt)
    write_json(run_dir / "02_reference/reference_manifest.json", {"schema_version": 1, "protocol_id": PROTOCOL_ID, "rows": rows})
    return rows


def _wav2lip_worker() -> Any:
    return importlib.import_module("scripts.experiments.wav2lip_face_roi_replacement.generation_worker")


def _mux_pcm(video_only: Path, audio: Path, output: Path, ffmpeg: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    command = [
        str(ffmpeg), "-y", "-v", "error", "-i", str(video_only), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-f", "matroska", str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        raise ProtocolError(f"PCM mux failed for {output}: {(result.stderr or '')[-1000:]}")
    temporary.replace(output)
    return {"command": command, "command_sha256": hashlib.sha256("\0".join(command).encode()).hexdigest()}


def render_wav2lip(
    cell: Mapping[str, Any],
    output_root: Path,
    cfg: Mapping[str, Any],
    *,
    model: Any | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Render one variable-length cell with canonical audio muxed losslessly."""
    worker = _wav2lip_worker()
    sample_id = str(cell["sample_id"])
    arm = str(cell["arm"])
    started = time.monotonic()
    output = output_root / "03_video" / str(cell["tfg"]) / sample_id / f"{arm}.mkv"
    receipt_path = output.with_name(f"{arm}.receipt.json")
    if receipt_path.is_file() and output.is_file():
        receipt = read_json(receipt_path)
        try:
            validate_receipt(receipt, cell)
            if receipt.get("output_sha256") == file_sha256(output):
                return receipt
        except ProtocolError:
            pass
    audio_path = Path(str(cell["feature"]["arms"][arm]["audio"])).resolve()
    reference = cell["reference"]
    visual_source = Path(str(reference["visual_source"])).resolve()
    if any("lrs3" in part.lower() for part in visual_source.parts):
        raise ProtocolError(f"LRS3 visual input is forbidden: {visual_source}")
    actual_visual_sha = file_sha256(visual_source)
    expected_visual_sha = str(reference.get("visual_source_sha256", ""))
    if not expected_visual_sha or actual_visual_sha != expected_visual_sha:
        raise ProtocolError(f"external visual source hash changed since plan: {visual_source}")
    reference_frame_path = Path(str(reference["reference"])).resolve()
    reference_frame = cv2.imread(str(reference_frame_path), cv2.IMREAD_COLOR)
    if reference_frame is None:
        raise ProtocolError(f"cannot read frozen external reference frame: {reference_frame_path}")
    boxes = [list(map(int, reference["box_top_bottom_left_right"]))]
    chunks, mel_info = worker.mel_chunks(audio_path)
    if not chunks:
        raise ProtocolError(f"no mel chunks for {cell['cell_key']}")
    # Wav2Lip requires one image/box per audio chunk.  Every arm receives the
    # same single frame from an external non-LRS3 asset; no source-video motion
    # is ever loaded into the renderer.
    frame_batch = [reference_frame.copy() for _ in chunks]
    box_batch = [boxes[0] for _ in chunks]
    if model is None:
        checkpoint = Path(str(cfg["wav2lip_checkpoint"])).resolve()
        model = worker.load_model(checkpoint, device)
    generated = worker.render_arm(model, frame_batch, box_batch, chunks, int(cfg.get("wav2lip_batch_size", 4)), device)
    video_only = output.with_name(f".{arm}.video_only.mkv")
    worker.encode_video(generated, video_only, Path(str(cfg["ffmpeg"])))
    mux = _mux_pcm(video_only, audio_path, output, Path(str(cfg["ffmpeg"])))
    video_only.unlink(missing_ok=True)
    receipt = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "cell_key": cell["cell_key"],
        "protocol_hash": cell.get("protocol_hash"),
        "block_id": cell["block_id"],
        "sample_id": sample_id,
        "source_group": cell["source_group"],
        "tfg": cell["tfg"],
        "arm": arm,
        "status": "complete",
        "attempt": int(cell.get("attempt", 1)),
        "seed": int(cell["seed"]),
        "audio": str(audio_path),
        "audio_sha256": file_sha256(audio_path),
        "audio_pcm_sha256": cell["feature"]["arms"][arm].get("pcm_sha256"),
        "visual_source": str(visual_source),
        "visual_source_sha256": actual_visual_sha,
        "visual_source_frame_policy": "first_frame_only",
        "visual_source_used_for_generation": "one_frame_repeated_for_each_mel_chunk",
        "cohort_source_video": reference.get("cohort_source_video"),
        "reference_frame_sha256": reference.get("reference_sha256"),
        "reference_crop_sha256": reference.get("crop_sha256"),
        "box": boxes[0],
        "checkpoint": str(Path(str(cfg["wav2lip_checkpoint"])).resolve()),
        "checkpoint_sha256": file_sha256(Path(str(cfg["wav2lip_checkpoint"])).resolve()),
        "device": device,
        "frame_count": len(generated),
        "generated_frames": len(generated),
        "scoreable_frames": None,
        "tail_excluded_ms": None,
        "mel_chunk_count": len(chunks),
        "mel": mel_info,
        "fixed_reference_frame_repeated": True,
        "tail_frames_reused_for_generation": 0,
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "mux": mux,
        "elapsed_seconds": float(time.monotonic() - started),
        "completed_at_epoch": time.time(),
    }
    write_json(receipt_path, receipt)
    return receipt


def render_ditto(cell: Mapping[str, Any], output_root: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Optional Ditto adapter; requires an explicit offline Ditto config."""
    started = time.monotonic()
    ditto_cfg = cfg.get("ditto")
    if not isinstance(ditto_cfg, Mapping):
        raise ProtocolError("Ditto replication requested without explicit cfg")
    adapter = importlib.import_module("scripts.03_ditto")
    # Ditto consumes a still reference image; the source video remains only
    # the provenance anchor for the frozen crop/reference receipt.
    source = Path(str(cell["reference"]["crop"])).resolve()
    audio = Path(str(cell["feature"]["arms"][cell["arm"]]["audio"])).resolve()
    output = output_root / "03_video" / "ditto" / str(cell["sample_id"]) / f"{cell['arm']}.mp4"
    mode = str(ditto_cfg.get("mode", "pytorch_fallback"))
    ok = adapter.run_ditto(Path(str(output_root.parents[1])), str(output_root.name), dict(ditto_cfg), audio, source, output, mode, retries=1, seed=int(cell["seed"]))
    if not ok or not output.is_file():
        raise ProtocolError(f"Ditto did not produce {output}")
    receipt = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "protocol_hash": cell.get("protocol_hash"), "cell_key": cell["cell_key"], "sample_id": cell["sample_id"], "source_group": cell["source_group"], "tfg": "ditto", "arm": cell["arm"], "status": "complete", "output": str(output.resolve()), "output_sha256": file_sha256(output), "audio": str(audio), "audio_sha256": file_sha256(audio), "seed": int(cell["seed"]), "elapsed_seconds": float(time.monotonic() - started)}
    write_json(output.with_name(f"{cell['arm']}.receipt.json"), receipt)
    return receipt


__all__ = ["prepare_reference", "prepare_references", "render_wav2lip", "render_ditto"]
