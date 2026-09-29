"""Stage 05: serial, paired Wav2Lip rendering."""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .audio import read_pcm16, write_pcm16
from .common import (
    ProtocolError,
    canonical_json_sha256,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .gpu import gpu_lease
from .support import probe_video


def shared_face_geometry(video: Path) -> dict[str, Any]:
    metadata = probe_video(video)
    return {
        "mode": "full_frame",
        "box_order": ["top", "bottom", "left", "right"],
        "box": metadata["full_frame_box"],
        "width": metadata["width"],
        "height": metadata["height"],
        "fps": config.FPS,
        "source_video_sha256": metadata["file_sha256"],
        "source_decoded_frame_count": metadata["decoded_frame_count"],
    }


def wav2lip_command(face: Path, audio: Path, output: Path, geometry: Mapping[str, Any]) -> list[str]:
    top, bottom, left, right = (str(int(value)) for value in geometry["box"])
    return [
        str(config.WAV2LIP_PYTHON),
        str(config.WAV2LIP_INFERENCE),
        "--checkpoint_path", str(config.WAV2LIP_CHECKPOINT),
        "--face", str(face),
        "--audio", str(audio),
        "--outfile", str(output),
        "--box", top, bottom, left, right,
        "--nosmooth",
        "--face_det_batch_size", "16",
        "--wav2lip_batch_size", "16",
    ]


def _runtime_bindings() -> dict[str, str]:
    for path in (config.WAV2LIP_PYTHON, config.WAV2LIP_INFERENCE, config.WAV2LIP_CHECKPOINT):
        if not path.is_file():
            raise ProtocolError(f"Wav2Lip runtime binding is missing: {path}")
    checkpoint_hash = file_sha256(config.WAV2LIP_CHECKPOINT)
    if checkpoint_hash != config.WAV2LIP_CHECKPOINT_SHA256:
        raise ProtocolError("Wav2Lip checkpoint hash differs from the frozen setup")
    return {
        "python_sha256": file_sha256(config.WAV2LIP_PYTHON),
        "inference_sha256": file_sha256(config.WAV2LIP_INFERENCE),
        "checkpoint_sha256": checkpoint_hash,
    }


def _prepare_driver_audio(source: Path, output_stage: Path, arm: str, sample_id: str) -> dict[str, Any]:
    """Create the fixed Wav2Lip right-context copy for one bridge arm.

    Wav2Lip's mel chunker needs a small amount of audio after the last frozen
    scoring frame. This copy is used only by the video generator. Scoring
    later reads the original bridge file and its exact frozen prefix.
    """

    source = source.resolve()
    if not source.is_file():
        raise ProtocolError(f"bridge driver audio is missing: {source}")
    source_values, source_meta = read_pcm16(source)
    output = output_stage / "driver_audio" / arm / f"{sample_id}.wav"
    sidecar = output.with_suffix(".json")
    source_sha256 = source_meta["file_sha256"]
    expected = {
        "protocol_id": config.PROTOCOL_ID,
        "sample_id": sample_id,
        "arm": arm,
        "source_audio": str(source),
        "source_audio_sha256": source_sha256,
        "source_sample_count": int(source_values.size),
        "driver_tail_pad_samples": config.WAV2LIP_TAIL_PAD_SAMPLES,
        "driver_tail_pad_value": 0,
        "driver_audio_policy": "fixed_zero_right_context_for_frozen_support_only; score_original_prefix",
    }
    if output.is_file() and sidecar.is_file():
        prior = verify_self_hashed_json(sidecar)
        driver_values, driver_meta = read_pcm16(output)
        if (
            all(prior.get(key) == value for key, value in expected.items())
            and prior.get("driver_audio") == str(output.resolve())
            and prior.get("driver_audio_sha256") == driver_meta["file_sha256"]
            and prior.get("driver_sample_count") == int(driver_values.size)
            and driver_values.size == source_values.size + config.WAV2LIP_TAIL_PAD_SAMPLES
            and np.array_equal(driver_values[: source_values.size], source_values)
            and np.all(driver_values[source_values.size :] == 0)
        ):
            return prior
        raise ProtocolError(f"existing driver audio identity changed: {sample_id}/{arm}")
    if output.exists() or sidecar.exists():
        raise ProtocolError(f"partial driver audio cannot be resumed: {sample_id}/{arm}")
    padded = np.concatenate(
        [source_values, np.zeros(config.WAV2LIP_TAIL_PAD_SAMPLES, dtype=np.int16)]
    )
    write_pcm16(output, padded)
    driver_values, driver_meta = read_pcm16(output)
    if (
        driver_values.size != source_values.size + config.WAV2LIP_TAIL_PAD_SAMPLES
        or not np.array_equal(driver_values[: source_values.size], source_values)
        or not np.all(driver_values[source_values.size :] == 0)
    ):
        raise ProtocolError(f"driver audio right-context construction failed: {sample_id}/{arm}")
    row = {
        "schema_version": 1,
        "stage_id": "05_videos",
        **expected,
        "driver_audio": str(output.resolve()),
        "driver_audio_sha256": driver_meta["file_sha256"],
        "driver_decoded_pcm_sha256": driver_meta["decoded_pcm_sha256"],
        "driver_sample_count": int(driver_values.size),
    }
    write_self_hashed_json(sidecar, row)
    return row


def _render_one(
    record: Mapping[str, Any],
    bridge_row: Mapping[str, Any],
    arm: str,
    repeat: int,
    repeat_seed: int,
    geometry: Mapping[str, Any],
    output_stage: Path,
    bindings: Mapping[str, str],
    driver_audio_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    face = Path(str(record["face_video"]["path"]))
    arm_by_name = {str(row["arm"]): row for row in bridge_row.get("arms", [])}
    if set(arm_by_name) != set(config.ARMS):
        raise ProtocolError(f"bridge arm coverage is incomplete: {sample_id}")
    source_audio = Path(str(arm_by_name[arm]["output"])).resolve()
    if driver_audio_row is None:
        audio = source_audio
        driver_tail_pad_samples = 0
    else:
        if (
            str(driver_audio_row.get("source_audio", "")) != str(source_audio)
            or str(driver_audio_row.get("source_audio_sha256", "")) != file_sha256(source_audio)
            or str(driver_audio_row.get("arm", "")) != arm
            or str(driver_audio_row.get("sample_id", "")) != sample_id
        ):
            raise ProtocolError(f"driver audio source binding differs: {sample_id}/{arm}")
        audio = Path(str(driver_audio_row.get("driver_audio", ""))).resolve()
        driver_tail_pad_samples = int(driver_audio_row.get("driver_tail_pad_samples", -1))
        if not audio.is_file() or file_sha256(audio) != str(driver_audio_row.get("driver_audio_sha256", "")):
            raise ProtocolError(f"driver audio binding is missing or changed: {sample_id}/{arm}")
        if driver_tail_pad_samples != config.WAV2LIP_TAIL_PAD_SAMPLES:
            raise ProtocolError(f"driver audio tail padding differs: {sample_id}/{arm}")
    source_audio_sha256 = file_sha256(source_audio)
    audio_sha256 = file_sha256(audio)
    geometry_hash = canonical_json_sha256(dict(geometry))
    output = output_stage / "videos" / sample_id / f"repeat_{repeat}" / arm / "video.mp4"
    sidecar = output.with_suffix(".json")
    if output.is_file() and sidecar.is_file():
        prior = verify_self_hashed_json(sidecar)
        if (
            prior.get("protocol_id") == config.PROTOCOL_ID
            and prior.get("sample_id") == sample_id
            and prior.get("arm") == arm
            and prior.get("render_repeat") == repeat
            and prior.get("repeat_seed") == repeat_seed
            and prior.get("face_sha256") == file_sha256(face)
            and prior.get("audio_sha256") == audio_sha256
            and prior.get("source_audio") == str(source_audio)
            and prior.get("source_audio_sha256") == source_audio_sha256
            and prior.get("driver_tail_pad_samples") == driver_tail_pad_samples
            and prior.get("geometry_sha256") == geometry_hash
            and prior.get("output_sha256") == file_sha256(output)
            and prior.get("checkpoint_sha256") == config.WAV2LIP_CHECKPOINT_SHA256
        ):
            return prior
        raise ProtocolError(f"existing render identity changed: {sample_id}/{repeat}/{arm}")
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    if output.exists() or sidecar.exists() or temporary.exists():
        raise ProtocolError(f"partial render cannot be resumed: {sample_id}/{repeat}/{arm}")
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_stage / "work" / sample_id / f"repeat_{repeat}" / arm
    log_path = output_stage / "logs" / sample_id / f"repeat_{repeat}_{arm}.log"
    work_dir.mkdir(parents=True, exist_ok=True)
    # The vendored Wav2Lip script writes temp/result.avi relative to cwd.
    # Keep that hard-coded path isolated per render cell.
    (work_dir / "temp").mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = wav2lip_command(face, audio, temporary, geometry)
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(repeat_seed)
    environment["TTS_EXP_RENDER_SEED"] = str(repeat_seed)
    started = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(work_dir), env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise ProtocolError(f"Wav2Lip render failed: {sample_id}/{repeat}/{arm}, exit={result.returncode}")
    temporary.replace(output)
    return {
        "schema_version": 1,
        "stage_id": "05_videos",
        "protocol_id": config.PROTOCOL_ID,
        "sample_id": sample_id,
        "source_group": str(record["source_group"]),
        "arm": arm,
        "render_repeat": repeat,
        "repeat_seed": repeat_seed,
        "invocation_id": f"{sample_id}/{repeat}/{arm}/{int(started * 1_000_000)}",
        "face": str(face.resolve()),
        "face_sha256": file_sha256(face),
        "audio": str(audio.resolve()),
        "audio_sha256": audio_sha256,
        "source_audio": str(source_audio),
        "source_audio_sha256": source_audio_sha256,
        "driver_tail_pad_samples": driver_tail_pad_samples,
        "driver_tail_pad_value": 0,
        "driver_audio_policy": "fixed_zero_right_context_for_frozen_support_only; score_original_prefix",
        "geometry": dict(geometry),
        "geometry_sha256": geometry_hash,
        "output": str(output.resolve()),
        "output_sha256": file_sha256(output),
        "command": command,
        "command_sha256": canonical_json_sha256(command),
        "runtime_bindings": bindings,
        "checkpoint_sha256": bindings["checkpoint_sha256"],
        "work_dir": str(work_dir.resolve()),
        "log": str(log_path.resolve()),
        "process_independent": True,
    }


def _load_bridge(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = verify_self_hashed_json(path)
    if payload.get("status") != "complete" or payload.get("record_count") != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("bridge audio manifest is incomplete")
    result: dict[str, Mapping[str, Any]] = {}
    for row in payload.get("rows", []):
        sample_id = str(row["sample_id"])
        if sample_id in result:
            raise ProtocolError(f"duplicate bridge row: {sample_id}")
        result[sample_id] = row
    if len(result) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("bridge audio coverage is incomplete")
    return result


def _load_execution_order(path: Path) -> list[dict[str, Any]]:
    payload = verify_self_hashed_json(path)
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != config.EXPECTED_VIDEO_COUNT:
        raise ProtocolError("render execution order is not the registered 176-item schedule")
    return [dict(item) for item in items]


def run_render_stage(run_root: Path, cohort: Mapping[str, Any]) -> dict[str, Any]:
    paths = config.RunPaths(run_root)
    bridge_by_id = _load_bridge(paths.bridge / "audio_manifest.json")
    order = _load_execution_order(paths.protocol / "execution_order.json")
    record_by_id = {str(row["sample_id"]): row for row in cohort.get("records", [])}
    if len(record_by_id) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("cohort is incomplete before render")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    driver_audio_rows: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        bindings = _runtime_bindings()
        geometry_by_id = {
            str(record["sample_id"]): shared_face_geometry(Path(str(record["face_video"]["path"])))
            for record in cohort.get("records", [])
        }
        for record in cohort.get("records", []):
            sample_id = str(record["sample_id"])
            arm_by_name = {str(row["arm"]): row for row in bridge_by_id[sample_id].get("arms", [])}
            if set(arm_by_name) != set(config.ARMS):
                raise ProtocolError(f"bridge arm coverage is incomplete: {sample_id}")
            for arm in config.ARMS:
                driver_audio_rows[(sample_id, arm)] = _prepare_driver_audio(
                    Path(str(arm_by_name[arm]["output"])), paths.videos, arm, sample_id
                )
        write_self_hashed_json(
            paths.videos / "driver_audio_manifest.json",
            {
                "schema_version": 1,
                "stage_id": "05_videos",
                "protocol_id": config.PROTOCOL_ID,
                "status": "complete",
                "record_count": config.EXPECTED_RECORD_COUNT,
                "driver_audio_count": len(driver_audio_rows),
                "expected_driver_audio_count": config.EXPECTED_RECORD_COUNT * len(config.ARMS),
                "tail_pad_samples": config.WAV2LIP_TAIL_PAD_SAMPLES,
                "rows": [driver_audio_rows[key] for key in sorted(driver_audio_rows)],
            },
        )
        with gpu_lease("wav2lip_render"):
            for ordinal, item in enumerate(order, 1):
                sample_id = str(item.get("sample_id", ""))
                arm = str(item.get("video_arm", ""))
                repeat = int(item.get("render_repeat", -1))
                if sample_id not in record_by_id or arm not in config.ARMS or repeat not in config.REPEATS:
                    failures.append({"ordinal": ordinal, "item": item, "error": "execution item is outside frozen matrix"})
                    continue
                record = record_by_id[sample_id]
                try:
                    geometry = geometry_by_id[sample_id]
                    rendered = _render_one(
                        record,
                        bridge_by_id[sample_id],
                        arm,
                        repeat,
                        {0: 20260913, 1: 20260914}[repeat],
                        geometry,
                        paths.videos,
                        bindings,
                        driver_audio_row=driver_audio_rows[(sample_id, arm)],
                    )
                    rows.append(rendered)
                    print(f"RENDER {ordinal}/{config.EXPECTED_VIDEO_COUNT} {sample_id} r={repeat} {arm}", flush=True)
                except Exception as exc:  # noqa: BLE001 - preserve each failed matrix item
                    failures.append({"ordinal": ordinal, "sample_id": sample_id, "render_repeat": repeat, "video_arm": arm, "error_type": type(exc).__name__, "error": str(exc)})
    except Exception as exc:  # noqa: BLE001 - preserve stage failure in manifest
        failures.append({"ordinal": None, "error_type": type(exc).__name__, "error": str(exc), "gpu_stage": True})
    identities = {(str(row["sample_id"]), int(row["render_repeat"]), str(row["arm"])) for row in rows}
    complete = not failures and len(rows) == config.EXPECTED_VIDEO_COUNT and len(identities) == config.EXPECTED_VIDEO_COUNT
    manifest = {
        "schema_version": 1,
        "stage_id": "05_videos",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if complete else "blocked",
        "record_count": len({str(row["sample_id"]) for row in rows}),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "video_count": len(rows),
        "expected_video_count": config.EXPECTED_VIDEO_COUNT,
        "arms": list(config.ARMS),
        "repeats": list(config.REPEATS),
        "schedule_sha256": file_sha256(paths.protocol / "execution_order.json"),
        "analysis_lock_sha256": file_sha256(paths.bridge / "analysis_lock.json") if (paths.bridge / "analysis_lock.json").is_file() else None,
        "runtime_bindings": bindings if "bindings" in locals() else None,
        "driver_audio_manifest_sha256": file_sha256(paths.videos / "driver_audio_manifest.json") if (paths.videos / "driver_audio_manifest.json").is_file() else None,
        "driver_tail_pad_samples": config.WAV2LIP_TAIL_PAD_SAMPLES,
        "rows": rows,
        "failures": failures,
    }
    write_self_hashed_json(paths.videos / "videos_manifest.json", manifest)
    write_self_hashed_json(paths.videos / "failures.json", {"schema_version": 1, "stage_id": "05_videos", "protocol_id": config.PROTOCOL_ID, "status": manifest["status"], "failures": failures})
    if not complete:
        raise ProtocolError(f"video rendering incomplete: {len(rows)}/{config.EXPECTED_VIDEO_COUNT}")
    return manifest


__all__ = ["_prepare_driver_audio", "run_render_stage", "shared_face_geometry", "wav2lip_command"]
