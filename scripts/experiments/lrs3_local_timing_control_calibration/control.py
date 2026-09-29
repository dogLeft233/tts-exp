from __future__ import annotations

import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    CalibrationError,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)


def read_pcm16(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        sample_count = handle.getnframes()
        payload = handle.readframes(sample_count)
    if channels != config.PCM_CHANNELS or sample_width != config.PCM_SAMPLE_WIDTH or sample_rate != config.SAMPLE_RATE:
        raise CalibrationError(f"audio format violates PCM16/16k mono contract: {path}")
    values = np.frombuffer(payload, dtype="<i2").copy()
    if values.size != sample_count or values.size < config.MIN_AUDIO_SAMPLES:
        raise CalibrationError(f"audio sample count is invalid: {path}")
    return values, {
        "sample_count": int(values.size),
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "sample_width": int(sample_width),
        "pcm_sha256": file_sha256(path),
    }


def pcm16_to_float(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype != np.int16 or values.ndim != 1:
        raise ValueError("PCM input must be one-dimensional int16")
    return values.astype(np.float64) / 32768.0


def write_pcm16(path: Path, values: np.ndarray) -> str:
    values = np.asarray(values)
    if values.dtype != np.int16 or values.ndim != 1 or values.size == 0:
        raise ValueError("PCM output must be a non-empty one-dimensional int16 array")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(config.PCM_CHANNELS)
        handle.setsampwidth(config.PCM_SAMPLE_WIDTH)
        handle.setframerate(config.SAMPLE_RATE)
        handle.writeframes(np.asarray(values, dtype="<i2").tobytes())
    temporary.replace(path)
    return file_sha256(path)


def waveform_qc(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise CalibrationError("waveform is empty or non-finite")
    peak = float(np.abs(values).max())
    return {
        "sample_count": int(values.size),
        "rms": float(np.sqrt(np.mean(values * values))),
        "peak": peak,
        "dc_offset": float(np.mean(values)),
        "finite": True,
        "clipped_sample_count": int(np.sum(np.abs(values) >= 1.0)),
        "clipped": bool(peak >= 1.0),
    }


def local_swap_pcm16(values: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    values = np.asarray(values)
    if values.dtype != np.int16 or values.ndim != 1:
        raise ValueError("PCM input must be one-dimensional int16")
    length = int(values.size)
    if length < config.MIN_AUDIO_SAMPLES:
        raise CalibrationError("LOCAL_SWAP input is shorter than the registered minimum")
    b1 = length // 4
    b2 = length // 2
    b3 = (3 * length) // 4
    swapped = np.concatenate((values[:b1], values[b2:b3], values[b1:b2], values[b3:])).astype("<i2")
    if swapped.size != values.size:
        raise CalibrationError("LOCAL_SWAP changed sample count")
    return swapped, {"length": length, "b1": b1, "b2": b2, "b3": b3}


def smooth_warp_pcm16(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(values)
    if values.dtype != np.int16 or values.ndim != 1:
        raise ValueError("PCM input must be one-dimensional int16")
    length = int(values.size)
    amplitude = float(config.SMOOTH_WARP_AMPLITUDE_SAMPLES)
    if length < config.SMOOTH_WARP_MIN_SAMPLES or length - 1 <= 2.0 * np.pi * amplitude:
        raise CalibrationError("LOCAL_WARP_120 input is too short for a strictly monotone map")
    indices = np.arange(length, dtype=np.float64)
    source = indices + amplitude * np.sin(2.0 * np.pi * indices / float(length - 1))
    source[0] = 0.0
    source[-1] = float(length - 1)
    if not np.all(np.diff(source) > 0.0) or float(source.min()) < 0.0 or float(source.max()) > float(length - 1):
        raise CalibrationError("LOCAL_WARP_120 map is not strictly monotone or leaves the input range")
    warped_float = np.interp(source, indices, values.astype(np.float64))
    warped = np.rint(warped_float).astype("<i2")
    if warped.size != values.size or warped[0] != values[0] or warped[-1] != values[-1]:
        raise CalibrationError("LOCAL_WARP_120 changed length or endpoint samples")
    displacement = source - indices
    steps = np.diff(source)
    return warped, {
        "length": length,
        "amplitude_samples": config.SMOOTH_WARP_AMPLITUDE_SAMPLES,
        "amplitude_ms": 120,
        "max_positive_displacement_samples": float(displacement.max()),
        "max_negative_displacement_samples": float(displacement.min()),
        "min_mapping_step": float(steps.min()),
        "max_mapping_step": float(steps.max()),
        "mapping_monotone": True,
        "mapping_in_range": True,
        "interpolation": "linear",
        "rounding": "nearest_ties_to_even",
        "float_dtype": "float64",
        "operation": "local_nonconstant_time_map",
    }


def construct_control(values: np.ndarray, branch: str) -> tuple[np.ndarray, dict[str, Any]]:
    if branch == config.REPAIR_BRANCH:
        output, metadata = local_swap_pcm16(values)
        return output, {"arm": config.LOCAL_SWAP_ARM, "construction": "registered_local_swap", **metadata}
    if branch == config.SMOOTH_BRANCH:
        output, metadata = smooth_warp_pcm16(values)
        return output, {"arm": config.SMOOTH_WARP_ARM, "construction": "registered_smooth_local_warp_120", **metadata}
    raise ValueError(f"unknown calibration branch: {branch}")


def construct_arm(values: np.ndarray, arm: str, branch: str) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(values)
    if values.dtype != np.int16 or values.ndim != 1:
        raise ValueError("PCM input must be one-dimensional int16")
    if arm in config.BASE_ARMS:
        return values.astype("<i2", copy=True), {"construction": "natural_pcm16_copy", "pcm_identity": "byte_identical_to_natural"}
    if arm == config.control_arm_for_branch(branch):
        return construct_control(values, branch)
    raise CalibrationError(f"unknown arm for branch {branch}: {arm}")


def _arm_row(
    record: Mapping[str, Any],
    arm: str,
    branch: str,
    output_stage: Path,
    protocol_sha256: str,
) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    natural_path = Path(str(record["natural_audio"]["path"]))
    natural, natural_meta = read_pcm16(natural_path)
    if natural_meta["pcm_sha256"] != record["natural_audio"]["sha256"]:
        raise CalibrationError(f"natural audio binding changed: {sample_id}")
    output = output_stage / "audio" / arm / f"{sample_id}.wav"
    sidecar = output.with_suffix(".json")
    if output.is_file() and sidecar.is_file():
        prior = verify_self_hashed_json(sidecar)
        if (
            prior.get("protocol_id") != config.PROTOCOL_ID
            or prior.get("protocol_sha256") != protocol_sha256
            or prior.get("sample_id") != sample_id
            or prior.get("arm") != arm
            or prior.get("branch") != branch
            or prior.get("natural_audio_sha256") != natural_meta["pcm_sha256"]
            or prior.get("output_sha256") != file_sha256(output)
        ):
            raise CalibrationError(f"existing audio identity changed: {sample_id}/{arm}")
        return prior
    if output.exists() or sidecar.exists():
        raise CalibrationError(f"partial audio cell cannot be resumed: {sample_id}/{arm}")
    values, construction = construct_arm(natural, arm, branch)
    if arm in config.BASE_ARMS and values.tobytes() != natural.tobytes():
        raise CalibrationError(f"natural PCM bytes changed: {sample_id}/{arm}")
    output_sha256 = write_pcm16(output, values)
    generated, output_meta = read_pcm16(output)
    if generated.size != natural.size or output_meta["pcm_sha256"] != output_sha256:
        raise CalibrationError(f"audio output format validation failed: {sample_id}/{arm}")
    row = {
        "schema_version": 1,
        "stage_id": "02_audio",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_sha256": protocol_sha256,
        "sample_id": sample_id,
        "source_group": str(record["source_group"]),
        "branch": branch,
        "arm": arm,
        "natural_audio": str(natural_path),
        "natural_audio_sha256": natural_meta["pcm_sha256"],
        "output": str(output),
        "output_sha256": output_sha256,
        "sample_count": int(generated.size),
        "format": output_meta,
        "construction": construction,
        "waveform_qc": waveform_qc(pcm16_to_float(generated)),
    }
    write_self_hashed_json(sidecar, row)
    return row


def materialize_audio(cohort: Mapping[str, Any], protocol: Mapping[str, Any], output_stage: Path) -> dict[str, Any]:
    branch = str(protocol.get("branch", ""))
    arms = config.arms_for_branch(branch)
    if cohort.get("status") != "complete" or len(cohort.get("records", [])) != config.EXPECTED_RECORD_COUNT:
        raise CalibrationError("cohort is incomplete")
    if protocol.get("status") != "locked" or protocol.get("branch") not in config.BRANCHES:
        raise CalibrationError("protocol is not locked for audio materialization")
    protocol_sha256 = file_sha256(Path(str(protocol["_path"]))) if "_path" in protocol else str(protocol.get("artifact_sha256", ""))
    if not protocol_sha256:
        raise CalibrationError("protocol hash is unavailable")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, record in enumerate(cohort["records"], 1):
        arm_rows: list[dict[str, Any]] = []
        for arm in arms:
            try:
                arm_rows.append(_arm_row(record, arm, branch, output_stage, protocol_sha256))
            except (OSError, CalibrationError, RuntimeError, TypeError, ValueError, FloatingPointError) as exc:
                failures.append({"sample_id": str(record["sample_id"]), "arm": arm, "error_type": type(exc).__name__, "error": str(exc)})
        if len(arm_rows) == len(arms):
            rows.append({"sample_id": str(record["sample_id"]), "source_group": str(record["source_group"]), "arms": arm_rows})
        print(f"AUDIO {index}/{config.EXPECTED_RECORD_COUNT} {record['sample_id']}", flush=True)
    cell_count = sum(len(row["arms"]) for row in rows)
    complete = not failures and len(rows) == config.EXPECTED_RECORD_COUNT and cell_count == config.expected_video_count(branch)
    manifest = {
        "schema_version": 1,
        "stage_id": "02_audio",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if complete else "blocked",
        "branch": branch,
        "control_arm": config.control_arm_for_branch(branch),
        "arms": list(arms),
        "record_count": len(rows),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "audio_cell_count": cell_count,
        "expected_audio_cell_count": config.expected_video_count(branch),
        "rows": rows,
        "failures": failures,
        "protocol_sha256": protocol_sha256,
        "score_read_count": 0,
        "frozen_before_scoring": True,
    }
    write_self_hashed_json(output_stage / "audio_manifest.json", manifest)
    if not complete:
        raise CalibrationError(f"audio materialization incomplete: {cell_count}/{config.expected_video_count(branch)}")
    return manifest
