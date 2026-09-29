from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
from PIL import Image

from scripts.experiments.static_image_bridge.render_worker import encode_ffv1_stream

from .common import ProtocolError, canonical_json_sha256, file_sha256, read_json, write_json


def image_rgb_sha256(path: str | Path) -> str:
    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def read_video_frames(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ProtocolError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.dtype != np.uint8 or frame.shape != (height, width, 3):
                raise ProtocolError(f"decoded frame shape/dtype changed in {path}")
            frames.append(frame)
    finally:
        capture.release()
    if not frames or abs(fps - 25.0) > 0.01:
        raise ProtocolError(f"video must contain frames at 25 fps: {path} frames={len(frames)} fps={fps}")
    return np.stack(frames, axis=0), {"frame_count": len(frames), "fps": fps, "width": width, "height": height}


def _run_environment(config: Mapping[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    ffmpeg_dir = str(Path(config["paths"]["ffmpeg"]).resolve().parent)
    env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
    return env


def build_render_argv(
    config: Mapping[str, Any],
    *,
    image: str | Path,
    image_sha256: str,
    audio: str | Path,
    box_xyxy: Sequence[int],
    output: str | Path,
    receipt: str | Path,
) -> list[str]:
    paths = config["paths"]
    if len(box_xyxy) != 4:
        raise ProtocolError("Wav2Lip box must be XYXY")
    return [
        str(Path(paths["wav2lip_python"]).absolute()),
        str(Path(paths["render_worker"]).resolve()),
        "--image", str(Path(image).resolve()),
        "--image-rgb-sha256", image_sha256,
        "--audio", str(Path(audio).resolve()),
        "--box", *[str(int(value)) for value in box_xyxy],
        "--checkpoint", str(Path(paths["wav2lip_checkpoint"]).resolve()),
        "--ffmpeg", str(Path(paths["ffmpeg"]).resolve()),
        "--outfile", str(Path(output).resolve()),
        "--result", str(Path(receipt).resolve()),
        "--batch-size", str(int(config["models"]["wav2lip_batch_size"])),
        "--seed", str(int(config["models"]["seed"])),
        "--device", str(config["models"]["device"]),
    ]


def canonicalize_tail(
    raw_video: str | Path,
    output_video: str | Path,
    *,
    target_frame_count: int,
    expected_raw_count: int | None = None,
    ffmpeg: str | Path,
    max_repair_frames: int = 5,
) -> dict[str, Any]:
    source = Path(raw_video).resolve()
    output = Path(output_video).resolve()
    frames, metadata = read_video_frames(source)
    raw_count = int(frames.shape[0])
    if expected_raw_count is not None and raw_count != int(expected_raw_count):
        raise ProtocolError(f"raw render receipt frame count differs from decoded count: {raw_count} != {expected_raw_count}")
    delta = int(target_frame_count) - raw_count
    if abs(delta) > int(max_repair_frames):
        raise ProtocolError(f"MEDIA_LENGTH_MISMATCH: raw={raw_count}, target={target_frame_count}, diff={delta}")
    if delta < 0:
        canonical = frames[: int(target_frame_count)].copy()
        action = "right_crop"
    elif delta > 0:
        pad = np.repeat(frames[-1:], delta, axis=0)
        canonical = np.concatenate((frames, pad), axis=0)
        action = "repeat_last_frame"
    else:
        canonical = frames
        action = "identity"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        output.unlink()
    encoded = encode_ffv1_stream(
        canonical,
        width=int(metadata["width"]),
        height=int(metadata["height"]),
        output=output,
        ffmpeg=Path(ffmpeg).resolve(),
    )
    decoded, decoded_meta = read_video_frames(output)
    if decoded.shape != canonical.shape or not np.array_equal(decoded, canonical):
        raise ProtocolError("canonical FFV1 file did not preserve the normalized pixel frames")
    return {
        "raw_video": str(source),
        "raw_video_sha256": file_sha256(source),
        "raw_frame_count": raw_count,
        "target_frame_count": int(target_frame_count),
        "canonical_frame_count": int(decoded.shape[0]),
        "tail_action": action,
        "tail_repair_frames": abs(delta),
        "raw_decoded_frames_sha256": hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest(),
        "canonical_decoded_frames_sha256": hashlib.sha256(np.ascontiguousarray(decoded).tobytes()).hexdigest(),
        "canonical_video": str(output),
        "canonical_video_sha256": file_sha256(output),
        "fps": decoded_meta["fps"],
        "width": decoded_meta["width"],
        "height": decoded_meta["height"],
        "encoder": encoded,
    }


def _fingerprint_matches(receipt_path: Path, fingerprint: str) -> dict[str, Any] | None:
    try:
        payload = read_json(receipt_path)
        body = dict(payload)
        recorded = body.pop("artifact_sha256", None)
        if recorded != canonical_json_sha256(body) or body.get("render_fingerprint") != fingerprint:
            return None
        video = Path(str(body.get("canonical_video", "")))
        if not video.is_file() or file_sha256(video) != body.get("canonical_video_sha256"):
            return None
        raw_receipt = Path(str(body.get("worker_receipt", "")))
        if not raw_receipt.is_file() or file_sha256(raw_receipt) != body.get("worker_receipt_sha256"):
            return None
        return body
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _assert_static_outside(frames: np.ndarray, image_bgr: np.ndarray, box: Sequence[int]) -> None:
    x1, y1, x2, y2 = [int(value) for value in box]
    outside = np.ones(image_bgr.shape[:2], dtype=bool)
    outside[y1:y2, x1:x2] = False
    if frames.shape[1:] != image_bgr.shape:
        raise ProtocolError("render dimensions differ from static portrait")
    for index, frame in enumerate(frames):
        if not np.array_equal(frame[outside], image_bgr[outside]):
            raise ProtocolError(f"static portrait pixels changed outside the generation box at frame {index}")


def _verify_worker_receipt(
    receipt: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    image: Path,
    image_sha: str,
    audio: Path,
    audio_sha: str,
    box: Sequence[int],
    output: Path,
) -> None:
    if receipt.get("status") != "complete" or receipt.get("input_mode") != "one_png_only":
        raise ProtocolError("Wav2Lip worker did not certify one static PNG input")
    if receipt.get("image_rgb_sha256") != image_sha or Path(str(receipt.get("image", ""))).resolve() != image.resolve():
        raise ProtocolError("Wav2Lip worker receipt has a different portrait binding")
    if receipt.get("audio_sha256") != audio_sha or Path(str(receipt.get("audio", ""))).resolve() != audio.resolve():
        raise ProtocolError("Wav2Lip worker receipt has a different N/M audio role")
    if receipt.get("box_xyxy") != [int(value) for value in box]:
        raise ProtocolError("Wav2Lip worker did not receive the frozen XYXY generation box")
    count = int(receipt.get("frames_rendered", 0))
    if count < 1 or len(receipt.get("source_frame_indices", [])) != count or set(receipt["source_frame_indices"]) != {0}:
        raise ProtocolError("Wav2Lip worker receipt does not prove one-PNG static sourcing")
    paths = config["paths"]
    if receipt.get("output_codec") != "ffv1" or abs(float(receipt.get("fps", -1)) - 25.0) > 0.01:
        raise ProtocolError("Wav2Lip worker output must be FFV1 at 25 fps")
    if Path(str(receipt.get("output", ""))).resolve() != output.resolve() or file_sha256(output) != receipt.get("output_sha256"):
        raise ProtocolError("Wav2Lip worker output path/hash mismatch")
    if Path(str(receipt.get("checkpoint", ""))).resolve() != Path(paths["wav2lip_checkpoint"]).resolve():
        raise ProtocolError("Wav2Lip loaded a different checkpoint")
    if receipt.get("checkpoint_sha256") != file_sha256(paths["wav2lip_checkpoint"]):
        raise ProtocolError("Wav2Lip checkpoint receipt SHA mismatch")
    configured_python = Path(paths["wav2lip_python"]).absolute()
    expected_prefix = configured_python.parent.parent.resolve()
    if (Path(str(receipt.get("python_executable", ""))).absolute() != configured_python
            or Path(str(receipt.get("python_realpath", ""))).resolve() != configured_python.resolve()
            or Path(str(receipt.get("python_prefix", ""))).resolve() != expected_prefix):
        raise ProtocolError("Wav2Lip was not run in the frozen Python environment")
    root = Path(paths["wav2lip_root"]).resolve()
    for key in ("audio_module", "models_module", "wav2lip_class_source"):
        module_path = Path(str(receipt.get(key, ""))).resolve()
        if root not in module_path.parents:
            raise ProtocolError(f"Wav2Lip imported {key} outside the frozen source tree: {module_path}")
        hash_key = f"{key}_sha256"
        if file_sha256(module_path) != receipt.get(hash_key):
            raise ProtocolError(f"Wav2Lip source receipt hash mismatch: {module_path}")
    if not receipt.get("loaded_parameter_sha256"):
        raise ProtocolError("Wav2Lip worker did not bind the loaded model parameters")


def _render_one(
    config: Mapping[str, Any],
    *,
    record: Mapping[str, Any],
    portrait: Mapping[str, Any],
    audio_role: str,
    run_dir: Path,
    budget_deadline: float | None = None,
) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    portrait_id = str(portrait["portrait_id"])
    audio_row = record["audio"]["natural" if audio_role == "N" else "mfa_linear"]
    audio_path = Path(audio_row["path"]).resolve()
    image_path = Path(portrait["path"]).resolve()
    image_sha = str(portrait["rgb_pixel_sha256"])
    image_hash = image_rgb_sha256(image_path)
    if image_hash != image_sha:
        raise ProtocolError(f"portrait pixel hash changed before render: {portrait_id}")
    audio_sha = file_sha256(audio_path)
    if audio_sha != audio_row["sha256"]:
        raise ProtocolError(f"audio changed before render: {sample_id}/{audio_role}")
    box = [int(value) for value in portrait["generation_box_xyxy"]]
    raw_dir = run_dir / "01_generation" / "raw" / portrait_id / sample_id
    canonical_dir = run_dir / "01_generation" / "canonical" / portrait_id / sample_id
    raw_video = raw_dir / f"{audio_role}.mkv"
    worker_receipt = raw_dir / f"{audio_role}.worker.json"
    canonical_video = canonical_dir / f"{audio_role}.mkv"
    receipt_path = canonical_dir / f"{audio_role}.receipt.json"
    render_binding = {
        "sample_id": sample_id,
        "paired_key": record["paired_key"],
        "speaker_id": record["speaker_id"],
        "portrait_id": portrait_id,
        "audio_role": audio_role,
        "audio_sha256": audio_sha,
        "image_rgb_sha256": image_sha,
        "generation_box_xyxy": box,
        "checkpoint_sha256": file_sha256(config["paths"]["wav2lip_checkpoint"]),
        "render_worker_sha256": file_sha256(config["paths"]["render_worker"]),
        "seed": int(config["models"]["seed"]),
        "device": config["models"]["device"],
    }
    fingerprint = canonical_json_sha256(render_binding)
    cached = _fingerprint_matches(receipt_path, fingerprint)
    if cached is not None:
        return cached

    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_video.parent.mkdir(parents=True, exist_ok=True)
    command = build_render_argv(
        config,
        image=image_path,
        image_sha256=image_sha,
        audio=audio_path,
        box_xyxy=box,
        output=raw_video,
        receipt=worker_receipt,
    )
    log_path = raw_dir / f"{audio_role}.wav2lip.log"
    configured_timeout = int(config.get("timeouts", {}).get("wav2lip_seconds", 900))
    timeout = configured_timeout
    budget_limited = False
    if budget_deadline is not None:
        remaining = budget_deadline - time.monotonic()
        if remaining < 1.0:
            raise ProtocolError("BUDGET_LIMITED:TOTAL_ACTIVE_GPU_BUDGET_EXHAUSTED")
        timeout = min(configured_timeout, max(1, int(remaining)))
        budget_limited = timeout < configured_timeout or remaining <= configured_timeout
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(command, ensure_ascii=False) + "\n")
        try:
            result = subprocess.run(
                command,
                cwd=str(Path(config["repo_root"]).resolve()),
                env=_run_environment(config),
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            if budget_limited:
                raise ProtocolError("BUDGET_LIMITED:TOTAL_ACTIVE_GPU_BUDGET_EXHAUSTED") from exc
            raise
        handle.write(f"\n[returncode] {result.returncode}\n")
    if result.returncode != 0 or not raw_video.is_file() or not worker_receipt.is_file():
        raise ProtocolError(f"Wav2Lip render failed for {sample_id}/{portrait_id}/{audio_role}; see {log_path}")
    raw_payload = read_json(worker_receipt)
    _verify_worker_receipt(
        raw_payload,
        config=config,
        image=image_path,
        image_sha=image_sha,
        audio=audio_path,
        audio_sha=audio_sha,
        box=box,
        output=raw_video,
    )
    target_frames = (int(record["audio"]["sample_count"]) + 639) // 640
    canonical = canonicalize_tail(
        raw_video,
        canonical_video,
        target_frame_count=target_frames,
        expected_raw_count=int(raw_payload["frames_rendered"]),
        ffmpeg=config["paths"]["ffmpeg"],
        max_repair_frames=int(config["media"]["tail_repair_max_frames"]),
    )
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    frames, _ = read_video_frames(canonical_video)
    if image_bgr is None:
        raise ProtocolError(f"cannot decode portrait PNG: {image_path}")
    _assert_static_outside(frames, image_bgr, box)
    receipt = {
        **render_binding,
        "render_fingerprint": fingerprint,
        "render_command": command,
        "render_log": str(log_path.resolve()),
        "worker_receipt": str(worker_receipt.resolve()),
        "worker_receipt_sha256": file_sha256(worker_receipt),
        "worker_checkpoint_sha256": raw_payload["checkpoint_sha256"],
        "loaded_parameter_sha256": raw_payload["loaded_parameter_sha256"],
        "worker_python": raw_payload["python_executable"],
        "worker_python_realpath": raw_payload["python_realpath"],
        "worker_python_prefix": raw_payload["python_prefix"],
        "worker_python_version": raw_payload["python_version"],
        "torch_version": raw_payload["torch_version"],
        "numpy_version": raw_payload["numpy_version"],
        "opencv_version": raw_payload["opencv_version"],
        "source_modules": {
            key: {"path": raw_payload[key], "sha256": raw_payload[f"{key}_sha256"]}
            for key in ("audio_module", "models_module", "wav2lip_class_source")
        },
        "raw_video": canonical["raw_video"],
        "raw_video_sha256": canonical["raw_video_sha256"],
        "canonical_video": canonical["canonical_video"],
        "canonical_video_sha256": canonical["canonical_video_sha256"],
        "raw_frame_count": canonical["raw_frame_count"],
        "canonical_frame_count": canonical["canonical_frame_count"],
        "tail_action": canonical["tail_action"],
        "tail_repair_frames": canonical["tail_repair_frames"],
        "valid_frame_count": min(target_frames, canonical["raw_frame_count"]),
        "frames_sha256": canonical["canonical_decoded_frames_sha256"],
        "fps": canonical["fps"],
        "width": canonical["width"],
        "height": canonical["height"],
        "outside_generation_box_identity": True,
        "worker_receipt_payload": raw_payload,
    }
    return write_json(receipt_path, receipt, self_hash=True)


def render_baselines(
    config: Mapping[str, Any],
    frozen_inputs: Mapping[str, Any],
    run_dir: str | Path,
    *,
    budget_deadline: float | None = None,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    portrait_rows = frozen_inputs["portrait_bindings"]["portraits"]
    records: list[dict[str, Any]] = []
    for record in frozen_inputs["records"]:
        sample_outputs: dict[str, Any] = {}
        for portrait_id in config["portraits"]:
            portrait = portrait_rows[str(portrait_id)]
            arm_receipts = {
                role: _render_one(config, record=record, portrait=portrait, audio_role=role, run_dir=root,
                                  budget_deadline=budget_deadline)
                for role in ("N", "M")
            }
            if arm_receipts["N"]["canonical_frame_count"] != arm_receipts["M"]["canonical_frame_count"]:
                raise ProtocolError(f"N/M Wav2Lip frame counts differ for {record['sample_id']}/portrait{portrait_id}")
            if arm_receipts["N"]["width"] != arm_receipts["M"]["width"] or arm_receipts["N"]["height"] != arm_receipts["M"]["height"]:
                raise ProtocolError(f"N/M video dimensions differ for {record['sample_id']}/portrait{portrait_id}")
            n_frames, _ = read_video_frames(arm_receipts["N"]["canonical_video"])
            m_frames, _ = read_video_frames(arm_receipts["M"]["canonical_video"])
            box = portrait["generation_box_xyxy"]
            image_bgr = cv2.imread(str(portrait["path"]), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ProtocolError(f"cannot decode portrait {portrait_id}")
            _assert_static_outside(n_frames, image_bgr, box)
            _assert_static_outside(m_frames, image_bgr, box)
            outside = np.ones(image_bgr.shape[:2], dtype=bool)
            x1, y1, x2, y2 = box
            outside[y1:y2, x1:x2] = False
            if not np.array_equal(n_frames[:, outside], m_frames[:, outside]):
                raise ProtocolError(f"N/M Wav2Lip frames differ outside the generation box for {record['sample_id']}/portrait{portrait_id}")
            valid_count = min(int(arm_receipts["N"]["valid_frame_count"]), int(arm_receipts["M"]["valid_frame_count"]))
            sample_outputs[str(portrait_id)] = {
                "portrait_id": str(portrait_id),
                "frame_count": int(n_frames.shape[0]),
                "valid_frame_count": valid_count,
                "width": int(n_frames.shape[2]),
                "height": int(n_frames.shape[1]),
                "N": arm_receipts["N"],
                "M": arm_receipts["M"],
            }
        lengths = {(row["frame_count"], row["valid_frame_count"]) for row in sample_outputs.values()}
        if len(lengths) != 1:
            raise ProtocolError(f"portrait transfer requires identical N/M frame support for sample {record['sample_id']}: {lengths}")
        records.append({
            "sample_id": str(record["sample_id"]),
            "paired_key": record["paired_key"],
            "speaker_id": record["speaker_id"],
            "transcript": record["transcript"],
            "portraits": sample_outputs,
            "frame_count": next(iter(lengths))[0],
            "valid_frame_count": next(iter(lengths))[1],
        })
    payload = {
        "schema_version": 1,
        "stage": "generate",
        "status": "COMPLETE",
        "sample_count": len(records),
        "portrait_ids": [str(value) for value in config["portraits"]],
        "audio_roles": {"N": "natural", "M": "mfa_linear"},
        "records": records,
    }
    write_json(root / "01_generation" / "manifest.json", payload, self_hash=True)
    return payload
