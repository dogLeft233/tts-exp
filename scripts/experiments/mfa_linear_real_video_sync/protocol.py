"""Input isolation, immutable locks, and fixed-coordinate protocol helpers."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.io import wavfile

import cv2

from .config import (
    CURVE_SIZE,
    FORBIDDEN_FIELD_TOKENS,
    MODEL_INPUT_FIELDS,
    P1_MFA_PARENT_SHA256,
    P1_SAMPLE_ID,
    SAMPLE_RATE,
    SEGMENT_FRAMES,
    SEGMENT_SAMPLES,
    TARGET_GAP_MIN,
    TRAINING_RECORD_FIELDS,
    VIDEO_FPS,
    VSHIFT,
)


class ProtocolError(ValueError):
    """Raised when an input cannot satisfy the frozen protocol."""


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    descriptor = f"{value.dtype}|{value.shape}|".encode()
    return sha256_bytes(descriptor + value.tobytes())


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_once(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically create JSON and refuse all overwrite/retry behavior."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise FileExistsError(f"create-once artifact already exists: {target}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def save_torch_once(path: str | Path, payload: Any) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise FileExistsError(f"create-once checkpoint already exists: {target}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return validate_artifact(target)


def load_json_resume_cell(path: str | Path, *, expected_sha256: str) -> Any:
    target = Path(path)
    validate_artifact(target, expected_sha256)
    try:
        payload = read_json(target)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ProtocolError(f"resume cell is corrupt: {target}") from error
    if not isinstance(payload, Mapping) or payload.get("status") not in {"complete", "GO"}:
        raise ProtocolError(f"resume cell is partial or stale: {target}")
    return payload


def validate_artifact(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    digest = sha256_file(target)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ProtocolError(f"artifact hash mismatch: {target}")
    return {"path": str(target.resolve()), "sha256": digest, "bytes": target.stat().st_size}


def validate_training_record_fields(record: Mapping[str, Any]) -> None:
    fields = set(record)
    unknown = fields - TRAINING_RECORD_FIELDS
    missing = TRAINING_RECORD_FIELDS - fields
    lowered = {field.lower() for field in fields}
    leaked = sorted(
        field
        for field in lowered
        if any(token in field for token in FORBIDDEN_FIELD_TOKENS)
    )
    if leaked:
        raise ProtocolError(f"natural/conditioning/input-isolation leakage: {leaked}")
    if unknown:
        raise ProtocolError(f"forbidden or unknown training-record fields: {sorted(unknown)}")
    if missing:
        raise ProtocolError(f"training record is missing fields: {sorted(missing)}")


def model_inputs(record: Mapping[str, Any]) -> dict[str, Any]:
    validate_training_record_fields(record)
    return {field: record[field] for field in MODEL_INPUT_FIELDS}


def validate_waveform(
    waveform: np.ndarray | torch.Tensor,
    *,
    sample_rate: int = SAMPLE_RATE,
    expected_samples: int = SEGMENT_SAMPLES,
) -> None:
    if sample_rate != SAMPLE_RATE:
        raise ProtocolError(f"sample rate must be exactly {SAMPLE_RATE}")
    if isinstance(waveform, torch.Tensor):
        if waveform.ndim == 3:
            valid_shape = waveform.shape[0] == 1 and waveform.shape[1] == 1
            sample_count = waveform.shape[-1]
        elif waveform.ndim == 1:
            valid_shape = True
            sample_count = waveform.numel()
        else:
            valid_shape = False
            sample_count = waveform.shape[-1] if waveform.ndim else 0
        finite = bool(torch.isfinite(waveform.detach()).all().item())
    else:
        values = np.asarray(waveform)
        valid_shape = values.ndim == 1
        sample_count = values.size if valid_shape else 0
        finite = bool(np.issubdtype(values.dtype, np.number) and np.isfinite(values).all())
    if not valid_shape:
        raise ProtocolError("waveform must be one mono sequence (or [1,1,N] tensor)")
    if sample_count != expected_samples:
        raise ProtocolError(f"waveform must contain exactly {expected_samples} samples, got {sample_count}")
    if not finite:
        raise ProtocolError("waveform must contain only finite values")


def load_mfa_linear_waveform(
    path: str | Path,
    *,
    expected_sha256: str,
    start_sample: int = 0,
    sample_count: int = SEGMENT_SAMPLES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load the locked float32 MFA parent without repair or conversion."""
    source = Path(path).resolve()
    locked = validate_artifact(source, expected_sha256)
    sample_rate, values = wavfile.read(source)
    if int(sample_rate) != SAMPLE_RATE:
        raise ProtocolError(f"MFA-linear sample rate must be {SAMPLE_RATE}, got {sample_rate}")
    if values.dtype != np.float32 or values.ndim != 1:
        raise ProtocolError(f"MFA-linear parent must be mono pcm_f32le, got {values.dtype} {values.shape}")
    if not np.isfinite(values).all() or np.max(np.abs(values)) > 1.0:
        raise ProtocolError("MFA-linear parent must be finite normalized waveform")
    end = start_sample + sample_count
    if start_sample < 0 or end > values.size:
        raise ProtocolError("requested exact MFA-linear segment is outside the parent")
    segment = np.ascontiguousarray(values[start_sample:end])
    validate_waveform(segment, expected_samples=sample_count)
    scaled = np.clip(segment.astype(np.float64) * 32768.0, -32768.0, 32767.0)
    rounded_pcm = np.round(scaled).astype("<i2")
    return segment, {
        **locked,
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "encoding": "pcm_f32le",
        "source_samples": int(values.size),
        "sample_interval": [int(start_sample), int(end)],
        "segment_float32_sha256": sha256_bytes(segment.tobytes()),
        "step0_rounded_pcm16_sha256": sha256_bytes(rounded_pcm.tobytes()),
    }


def load_pcm16_waveform(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    start_sample: int = 0,
    sample_count: int = SEGMENT_SAMPLES,
) -> tuple[np.ndarray, dict[str, Any]]:
    source = Path(path).resolve()
    locked = validate_artifact(source, expected_sha256)
    sample_rate, pcm = wavfile.read(source)
    if int(sample_rate) != SAMPLE_RATE:
        raise ProtocolError(f"audio sample rate must be {SAMPLE_RATE}, got {sample_rate}")
    if pcm.dtype != np.int16:
        raise ProtocolError(f"audio must be signed PCM16, got {pcm.dtype}")
    if pcm.ndim != 1:
        raise ProtocolError("audio must be mono")
    end = start_sample + sample_count
    if start_sample < 0 or end > pcm.size:
        raise ProtocolError("requested exact segment is outside the source waveform")
    segment = pcm[start_sample:end]
    if segment.size != sample_count:
        raise ProtocolError("exact segment length mismatch")
    waveform = segment.astype(np.float32) / 32768.0
    validate_waveform(waveform, expected_samples=sample_count)
    return waveform, {
        **locked,
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "encoding": "signed_pcm16",
        "source_samples": int(pcm.size),
        "sample_interval": [int(start_sample), int(end)],
        "segment_pcm_sha256": sha256_bytes(segment.tobytes()),
    }


def validate_protocol_metadata(metadata: Mapping[str, Any], *, sample_id: str, source_group: str) -> None:
    expected = {
        "sample_id": sample_id,
        "source_group": source_group,
        "protocol_split": "train",
        "fps": VIDEO_FPS,
        "frame_start": 0,
        "frame_end": SEGMENT_FRAMES,
        "sample_start": 0,
        "sample_end": SEGMENT_SAMPLES,
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ProtocolError(f"input lock metadata mismatch: {canonical_json(mismatches)}")


def calibrate_target_offset(curve: Sequence[float] | np.ndarray | torch.Tensor, *, vshift: int = VSHIFT) -> dict[str, Any]:
    values = torch.as_tensor(curve, dtype=torch.float64).detach().cpu()
    if values.ndim != 1 or values.numel() != 2 * vshift + 1:
        raise ProtocolError(f"natural curve must contain exactly {2 * vshift + 1} values")
    if not torch.isfinite(values).all():
        raise ProtocolError("natural curve is non-finite")
    ordered, indices = torch.sort(values)
    gap = float(ordered[1] - ordered[0])
    ambiguity_limit = TARGET_GAP_MIN + 8 * np.finfo(np.float64).eps * max(1.0, abs(float(ordered[0])))
    if not math.isfinite(gap) or gap <= ambiguity_limit:
        raise ProtocolError(f"natural target offset is ambiguous: gap={gap} <= {TARGET_GAP_MIN}")
    target_index = int(indices[0])
    target_offset = target_index - vshift
    return {
        "curve": [float(item) for item in values],
        "curve_sha256": sha256_tensor(values),
        "target_index": target_index,
        "target_offset": target_offset,
        "official_av_offset": -target_offset,
        "best_second_gap": gap,
        "vshift": vshift,
        "curve_index_contract": "target_offset + vshift",
        "official_offset_contract": "-target_offset",
        "detached": True,
    }


def validate_target_offset_artifact(artifact: Mapping[str, Any]) -> None:
    recomputed = calibrate_target_offset(artifact.get("curve", []), vshift=int(artifact.get("vshift", VSHIFT)))
    for key in ("target_index", "target_offset", "official_av_offset", "curve_sha256"):
        if artifact.get(key) != recomputed[key]:
            raise ProtocolError(f"stored natural target artifact violates {key}")


def read_fixed_video_frames(
    path: str | Path,
    *,
    start_frame: int = 0,
    frame_count: int = SEGMENT_FRAMES,
) -> np.ndarray:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or abs(fps - VIDEO_FPS) > 0.01:
        capture.release()
        raise ProtocolError(f"video must be exactly {VIDEO_FPS} fps, got {fps}")
    try:
        for _ in range(start_frame):
            ok, _ = capture.read()
            if not ok:
                raise ProtocolError("video ends before the locked start frame")
        frames = []
        for _ in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise ProtocolError("video ends before the locked frame interval")
            frames.append(frame)
    finally:
        capture.release()
    values = np.stack(frames)
    if values.dtype != np.uint8 or values.ndim != 4 or values.shape[-1] != 3:
        raise ProtocolError("decoded video must yield BGR uint8 frames")
    return values


def materialize_ffv1_once(path: str | Path, frames: np.ndarray) -> dict[str, Any]:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"create-once frozen video already exists: {target}")
    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[0] != SEGMENT_FRAMES or values.shape[-1] != 3 or values.dtype != np.uint8:
        raise ProtocolError("frozen visual input must be 96 BGR uint8 frames")
    height, width = values.shape[1:3]
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(VIDEO_FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "ffv1",
        "-level",
        "3",
        str(target),
    ]
    completed = subprocess.run(command, input=values.tobytes(), check=False, capture_output=True)
    if completed.returncode != 0 or not target.is_file():
        target.unlink(missing_ok=True)
        raise RuntimeError(f"FFV1 materialization failed: {completed.stderr.decode(errors='replace').strip()}")
    decoded = read_fixed_video_frames(target)
    source_hashes = frame_hashes(values)
    decoded_hashes = frame_hashes(decoded)
    if source_hashes != decoded_hashes:
        target.unlink(missing_ok=True)
        raise ProtocolError("FFV1 round trip changed decoded BGR frames")
    return {
        "path": str(target.resolve()),
        "sha256": sha256_file(target),
        "fps": VIDEO_FPS,
        "frame_interval": [0, SEGMENT_FRAMES],
        "frame_count": SEGMENT_FRAMES,
        "width": width,
        "height": height,
        "codec": "ffv1",
        "decoded_bgr_frame_sha256": decoded_hashes,
    }


def extract_official_bgr_frames(path: str | Path, output_dir: str | Path) -> np.ndarray:
    """Extract the exact JPEG/BGR frames used by the official scorer."""
    source = Path(path).resolve()
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError(f"create-once official frame directory already exists: {target}")
    target.mkdir(parents=True)
    completed = subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-nostdin", "-i", str(source),
            "-threads", "1", "-f", "image2", str(target / "%06d.jpg"),
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        for item in target.glob("*"):
            item.unlink(missing_ok=True)
        target.rmdir()
        raise RuntimeError(f"official JPEG frame extraction failed: {completed.stderr.decode(errors='replace').strip()}")
    paths = sorted(target.glob("*.jpg"))
    if len(paths) != SEGMENT_FRAMES:
        raise ProtocolError(f"official extraction produced {len(paths)} frames, expected {SEGMENT_FRAMES}")
    frames = []
    for frame_path in paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ProtocolError(f"official extraction yielded unreadable frame: {frame_path}")
        frames.append(frame)
    values = np.stack(frames)
    if values.shape[1:] != (224, 224, 3) or values.dtype != np.uint8:
        raise ProtocolError(f"official BGR frames have unexpected shape: {values.shape}")
    return values


def frame_hashes(frames: np.ndarray) -> list[str]:
    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[0] != SEGMENT_FRAMES or values.shape[-1] != 3 or values.dtype != np.uint8:
        raise ProtocolError("decoded visual segment must be 96 BGR uint8 frames")
    return [sha256_bytes(np.ascontiguousarray(frame).tobytes()) for frame in values]


def validate_visual_lock(lock: Mapping[str, Any], decoded_frames: np.ndarray) -> None:
    hashes = frame_hashes(decoded_frames)
    if lock.get("fps") != VIDEO_FPS or lock.get("frame_interval") != [0, SEGMENT_FRAMES]:
        raise ProtocolError("visual lock timebase mismatch")
    if lock.get("decoded_bgr_frame_sha256") != hashes:
        raise ProtocolError("VISUAL_COORDINATE_MISMATCH: decoded frame hashes changed")


def p1_metadata_template() -> dict[str, Any]:
    return {
        "sample_id": P1_SAMPLE_ID,
        "source_group": "6ORDQFh0Byw",
        "protocol_split": "train",
        "fps": VIDEO_FPS,
        "frame_start": 0,
        "frame_end": SEGMENT_FRAMES,
        "sample_start": 0,
        "sample_end": SEGMENT_SAMPLES,
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "expected_mfa_parent_sha256": P1_MFA_PARENT_SHA256,
    }
