"""Run the ``ditto_timing_rate_v1`` temporal-identifiability experiment.

The runner is deliberately thin: frozen parent assets are audited, crop media
are decoded with the existing SyncNet contract, waveform/frame retiming is
performed in memory, and the frozen SyncNet is run again.  Statistical rules
live in :mod:`ditto_timing_rate_metrics`; the companion checker repeats them
without importing that module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.experiments import ditto_timing_rate_metrics as metrics
from scripts.experiments.tts_native_gain_attribution import config as sync_config
from scripts.experiments.tts_native_gain_attribution.audio import write_pcm16_wav
from scripts.experiments.tts_native_gain_attribution.common import (
    ProtocolError,
    decode_media_pcm16,
    executable_path,
    ffprobe_json,
    file_sha256,
    read_json,
    run_command,
    write_json,
    write_self_hashed_json,
)
from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

PARENT_RUN = REPO / "runs/vsr_ditto50_linkage_v1"
PARENT_MANIFEST = PARENT_RUN / "manifest.json"
PARENT_SYNC = PARENT_RUN / "syncnet_records.json"
PARENT_VALIDATION = PARENT_RUN / "validation.json"
MODEL_PATH = REPO / "third_party/syncnet_python/data/syncnet_v2.model"
PROTOCOL_ID = "ditto_timing_rate_v1"
COHORT_ID = "ditto50_s0765"
EXPECTED_MANIFEST_SHA = "b388453fbfe378058e39ed5ea33ed53e6e36b363175869f764c7539777bfefd6"
EXPECTED_SYNC_SHA = ""  # parent sync records are bound by their individual crop/model receipts
EXPECTED_MODEL_SHA = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
SMOKE_IDS = (1, 3, 12)
CONDITIONS = ("natural", "tts")
ARMS = ("N", "T")
MODES = ("O", "SHORT", "LONG")
MODE_NAMES = {"O": "native", "SHORT": "short", "LONG": "long"}
FPS = 25
SAMPLE_RATE = 16_000
SAMPLES_PER_FRAME = 640
TAIL_GUARD_S = 0.080
RATE_MAX_CORRECTION = 1280
BOOTSTRAP_SEED = metrics.BOOTSTRAP_SEED


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return _sha_bytes(array.tobytes())


def _load(path: Path) -> Any:
    return read_json(path)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _write(path: Path, value: Any, *, hashed: bool = False) -> dict[str, Any]:
    payload = _json_value(value)
    if hashed:
        return write_self_hashed_json(path, payload)
    write_json(path, payload)
    return payload


def _save_npy(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, np.asarray(value), allow_pickle=False)
    temporary.replace(path)
    return file_sha256(path)


def _load_npy(path: Path) -> np.ndarray:
    try:
        value = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"cannot read feature array: {path}") from exc
    if not isinstance(value, np.ndarray) or value.ndim != 2 or not np.isfinite(value).all():
        raise ProtocolError(f"invalid feature array: {path}")
    return value


def _parent_manifest() -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    if not PARENT_MANIFEST.is_file() or not PARENT_SYNC.is_file():
        raise ProtocolError("parent Ditto-50 run is missing")
    actual = file_sha256(PARENT_MANIFEST)
    if actual != EXPECTED_MANIFEST_SHA:
        raise ProtocolError(f"parent manifest hash changed: {actual} != {EXPECTED_MANIFEST_SHA}")
    manifest = _load(PARENT_MANIFEST)
    sync_rows = _load(PARENT_SYNC).get("records", [])
    by_key: dict[int, dict[str, Any]] = {}
    for row in sync_rows:
        if not isinstance(row, dict) or row.get("status") != "COMPLETE":
            continue
        sample_id = int(row["id"])
        by_key.setdefault(sample_id, {})[str(row["condition"])] = row
    return manifest, by_key


def _fraction_float(value: Any) -> float | None:
    try:
        if value in (None, "N/A", "0/0"):
            return None
        if isinstance(value, str) and "/" in value:
            return float(Fraction(value))
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _crop_probe(path: Path) -> dict[str, Any]:
    probe = ffprobe_json(path)
    streams = [row for row in probe.get("streams", []) if isinstance(row, dict)]
    video = next((row for row in streams if row.get("codec_type") == "video"), None)
    audio = [row for row in streams if row.get("codec_type") == "audio"]
    if video is None or len(audio) != 1:
        raise ProtocolError(f"crop needs one video and one audio stream: {path}")
    fps = _fraction_float(video.get("avg_frame_rate") or video.get("r_frame_rate"))
    start_v = _fraction_float(video.get("start_time")) or 0.0
    start_a = _fraction_float(audio[0].get("start_time")) or 0.0
    first_pts: dict[str, Any] = {}
    try:
        executable = executable_path(sync_config.FFPROBE, "ffprobe")
        result = run_command((str(executable), "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#3", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)))
        first_pts = json.loads(result.stdout.decode("utf-8"))
    except Exception as exc:  # metadata is retained as a diagnostic; decode remains authoritative
        first_pts = {"error": str(exc)}
    frames = list(SyncNetEngine._stream_mjpeg(path))
    if not frames:
        raise ProtocolError(f"crop has no decodable frames: {path}")
    shapes = {tuple(int(x) for x in frame.shape) for frame in frames}
    if shapes != {(224, 224, 3)}:
        raise ProtocolError(f"crop frame shape is not 224x224 BGR: {path}: {sorted(shapes)}")
    pcm = decode_media_pcm16(path)
    frame_count = len(frames)
    audio_samples = int(pcm.size)
    L = min(frame_count, audio_samples // SAMPLES_PER_FRAME)
    if L < 6:
        raise ProtocolError(f"crop is too short for SyncNet windows: {path}")
    tail_trim_s = max((frame_count - L) / FPS, (audio_samples - L * SAMPLES_PER_FRAME) / SAMPLE_RATE)
    first_frame_pts = first_pts.get("frames", [{}])[0] if isinstance(first_pts, dict) and first_pts.get("frames") else {}
    timestamps = first_pts.get("frames", []) if isinstance(first_pts, dict) else []
    step_ok = True
    steps: list[float] = []
    for left, right in pairwise(timestamps):
        a = _fraction_float(left.get("best_effort_timestamp_time"))
        b = _fraction_float(right.get("best_effort_timestamp_time"))
        if a is not None and b is not None:
            steps.append(b - a)
    if steps:
        step_ok = all(abs(step - 1.0 / FPS) <= 1e-3 for step in steps)
    clock_status = "PASS"
    if fps is None or abs(fps - FPS) > 1e-6 or abs(start_v - start_a) > 1.0 / SAMPLE_RATE or not step_ok:
        clock_status = "UNSUPPORTED_CROP_CLOCK"
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256(path),
        "frame_count": frame_count,
        "audio_samples": audio_samples,
        "fps": fps,
        "width": int(video.get("width", 0)),
        "height": int(video.get("height", 0)),
        "video_start_time": start_v,
        "audio_start_time": start_a,
        "first_frame_pts": first_frame_pts,
        "frame_step_probe": steps,
        "L": int(L),
        "F": int(L - 5),
        "duration_s": float(L / FPS),
        "tail_trim_s": float(tail_trim_s),
        "tail_trim_status": "EXCESS_TAIL_TRIM" if tail_trim_s > TAIL_GUARD_S else "PASS",
        "clock_status": clock_status,
        "pcm_sha256": _sha_array(pcm[: L * SAMPLES_PER_FRAME]),
    }


def audit_inputs(run_dir: Path, *, smoke: bool = False) -> dict[str, Any]:
    manifest, sync_by_id = _parent_manifest()
    if manifest.get("cohort_id") != COHORT_ID:
        raise ProtocolError(f"unexpected cohort: {manifest.get('cohort_id')}")
    source_rows = []
    wanted = set(SMOKE_IDS if smoke else manifest.get("run_sample_ids", []))
    for row in manifest.get("records", []):
        if not isinstance(row, dict) or row.get("eligibility") != "eligible":
            continue
        sample_id = int(row["id"])
        if sample_id not in wanted:
            continue
        sync_rows = sync_by_id.get(sample_id, {})
        conditions: dict[str, Any] = {}
        reasons: list[str] = []
        for condition in CONDITIONS:
            receipt = sync_rows.get(condition)
            if not receipt:
                reasons.append(f"missing_{condition}_receipt")
                continue
            crop = Path(str(receipt.get("crop", ""))).resolve()
            if not crop.is_file():
                reasons.append(f"missing_{condition}_crop")
                continue
            for binding_name in ("input", "track"):
                bound_path = Path(str(receipt.get(binding_name, ""))).resolve()
                expected_sha = receipt.get(f"{binding_name}_sha256")
                if not bound_path.is_file():
                    reasons.append(f"missing_{condition}_{binding_name}")
                elif not isinstance(expected_sha, str) or file_sha256(bound_path) != expected_sha:
                    reasons.append(f"{condition}_{binding_name}_hash")
            if receipt.get("model_sha256") != EXPECTED_MODEL_SHA:
                reasons.append(f"{condition}_model_hash")
            if receipt.get("crop_sha256") != file_sha256(crop):
                reasons.append(f"{condition}_crop_hash")
            try:
                probe = _crop_probe(crop)
            except ProtocolError as exc:
                reasons.append(f"{condition}_probe:{exc}")
                continue
            conditions[condition] = {
                "crop": str(crop),
                "crop_sha256": receipt.get("crop_sha256"),
                "track": receipt.get("track"),
                "track_sha256": receipt.get("track_sha256"),
                "parent_score": {key: receipt.get(key) for key in ("sync_c", "sync_d", "av_offset")},
                "probe": probe,
            }
        source_rows.append({
            "id": sample_id,
            "cohort_id": row.get("cohort_id"),
            "paired_key": row.get("paired_key"),
            "wid": row.get("wid"),
            "speaker": row.get("speaker"),
            "normalized_text": row.get("normalized_text"),
            "eligibility": row.get("eligibility"),
            "conditions": conditions,
            "audit_reasons": reasons,
            "audit_status": "PASS" if not reasons and len(conditions) == 2 else "INCOMPLETE",
        })
    source_rows.sort(key=lambda row: int(row["id"]))
    audit = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "cohort_id": COHORT_ID,
        "status": "PASS" if all(row["audit_status"] == "PASS" for row in source_rows) else "INCOMPLETE",
        "smoke": bool(smoke),
        "candidate_count": len(source_rows),
        "source_eligible_count": len(source_rows),
        "ids": [row["id"] for row in source_rows],
        "source_manifest": {"path": str(PARENT_MANIFEST), "sha256": file_sha256(PARENT_MANIFEST)},
        "source_sync_records": {"path": str(PARENT_SYNC), "sha256": file_sha256(PARENT_SYNC)},
        "parent_validation": _load(PARENT_VALIDATION) if PARENT_VALIDATION.is_file() else None,
        "records": source_rows,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ffmpeg": str(executable_path(sync_config.FFMPEG, "ffmpeg")),
            "ffprobe": str(executable_path(sync_config.FFPROBE, "ffprobe")),
            "model": str(MODEL_PATH),
            "model_sha256": file_sha256(MODEL_PATH),
            "protocol_device": "deferred_to_extract",
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _write(run_dir / "inputs.json", audit, hashed=True)
    return audit


def _load_inputs(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "inputs.json"
    if not path.is_file():
        raise ProtocolError(f"inputs.json is missing: {path}")
    return _load(path)


def _pair_records(inputs: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in inputs.get("records", []) if row.get("audit_status") == "PASS"]


def prepare_media(run_dir: Path) -> dict[str, Any]:
    inputs = _load_inputs(run_dir)
    records: list[dict[str, Any]] = []
    for row in _pair_records(inputs):
        n = row["conditions"]["natural"]["probe"]
        t = row["conditions"]["tts"]["probe"]
        Ln, Lt = int(n["L"]), int(t["L"])
        targets = {"SHORT": min(Ln, Lt), "LONG": max(Ln, Lt)}
        cells: dict[str, Any] = {}
        for arm, condition, L in (("N", "natural", Ln), ("T", "tts", Lt)):
            for mode in MODES:
                if mode == "O":
                    M = L
                    rate = 1.0
                else:
                    M = int(targets[mode])
                    rate = float(L / M)
                cell = {
                    "id": int(row["id"]),
                    "arm": arm,
                    "condition": condition,
                    "mode": mode,
                    "source_L": L,
                    "target_M": M,
                    "source_F": max(0, L - 5),
                    "target_F": max(0, M - 5),
                    "rate": rate,
                    "rate_decimal": format(rate, ".17g"),
                    "status": "PASS" if M >= 6 and 0.5 <= rate <= 2.0 else "RATE_OUT_OF_PROTOCOL",
                    "source_crop": row["conditions"][condition]["crop"],
                    "source_crop_sha256": row["conditions"][condition]["crop_sha256"],
                    "tail_trim_status": row["conditions"][condition]["probe"]["tail_trim_status"],
                    "clock_status": row["conditions"][condition]["probe"]["clock_status"],
                }
                if cell["tail_trim_status"] != "PASS" or cell["clock_status"] != "PASS":
                    cell["status"] = "UNSUPPORTED_CROP_CLOCK" if cell["clock_status"] != "PASS" else "EXCESS_TAIL_TRIM"
                cells[f"{arm}_{mode}"] = cell
        records.append({"id": int(row["id"]), "paired_key": row["paired_key"], "L_N": Ln, "L_T": Lt, "M_SHORT": targets["SHORT"], "M_LONG": targets["LONG"], "cells": cells})
    media = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if records else "INCOMPLETE",
        "records": records,
        "rules": {
            "canonical": "L=min(decoded_crop_frames,floor(decoded_pcm_samples/640)); F=L-5",
            "video_map": "u=j*r; linear BGR; round-to-even; endpoint hold",
            "audio_map": "ffmpeg atempo; mono 16k PCM16; tail-only correction <=1280 samples",
        },
    }
    _write(run_dir / "media.json", media, hashed=True)
    _write(run_dir / "protocol.json", {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "cohort_id": COHORT_ID,
        "smoke": bool(inputs.get("smoke")),
        "device": None,
        "batch_size": 20,
        "K_MAX": metrics.K_MAX,
        "lags": list(metrics.LAGS),
        "deltas": list(metrics.DELTAS),
        "guard": {"left_frames": metrics.GUARD_LEFT, "right_frames": metrics.GUARD_RIGHT},
        "phase_count": metrics.PHASE_COUNT,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": metrics.BOOTSTRAP_DRAWS,
        "source_manifest_sha256": file_sha256(PARENT_MANIFEST),
        "model_sha256": file_sha256(MODEL_PATH),
    }, hashed=True)
    return media


def _decode_canonical(crop: Path, expected_L: int) -> tuple[np.ndarray, np.ndarray]:
    frames = list(SyncNetEngine._stream_mjpeg(crop))
    pcm = decode_media_pcm16(crop)
    L = min(len(frames), int(pcm.size // SAMPLES_PER_FRAME))
    if L != int(expected_L):
        raise ProtocolError(f"canonical length changed for {crop}: {L} != {expected_L}")
    frames_array = np.stack(frames[:L], axis=0).astype(np.uint8, copy=False)
    pcm_array = np.asarray(pcm[: L * SAMPLES_PER_FRAME], dtype="<i2").copy()
    return frames_array, pcm_array


def retime_frames(frames: np.ndarray, target_M: int) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(frames)
    if source.ndim != 4 or source.shape[1:] != (224, 224, 3) or source.dtype != np.uint8:
        raise ProtocolError(f"frames violate canonical BGR contract: {source.shape}/{source.dtype}")
    L = int(source.shape[0])
    M = int(target_M)
    if L < 1 or M < 1:
        raise ProtocolError("retime needs positive frame counts")
    rate = float(L / M)
    output = np.empty((M, 224, 224, 3), dtype=np.uint8)
    hold_count = 0
    for j in range(M):
        u = float(j) * rate
        lo = math.floor(u)
        if lo >= L - 1:
            output[j] = source[-1]
            hold_count += 1
            continue
        hi = min(lo + 1, L - 1)
        w = u - lo
        value = (1.0 - w) * source[lo].astype(np.float64) + w * source[hi].astype(np.float64)
        output[j] = np.clip(np.rint(value), 0.0, 255.0).astype(np.uint8)
    return output, {"status": "PASS", "source_L": L, "target_M": M, "rate": rate, "map": "u=j*r, floor/next, round-to-even", "endpoint_hold_count": hold_count, "input_sha256": _sha_array(source), "output_sha256": _sha_array(output)}


def _write_temp_wav(directory: Path, pcm: np.ndarray) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="atempo_", suffix=".wav", dir=directory, delete=False) as handle:
        path = Path(handle.name)
    write_pcm16_wav(path, np.asarray(pcm, dtype=np.int16))
    return path


def retime_audio(pcm: np.ndarray, target_samples: int, rate: float, scratch: Path) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(pcm)
    if source.dtype != np.int16 or source.ndim != 1 or source.size < 1:
        raise ProtocolError("audio violates PCM16 contract")
    target = int(target_samples)
    if target < 1 or not 0.5 <= float(rate) <= 2.0:
        raise ProtocolError("audio target/rate is outside protocol")
    if abs(float(rate) - 1.0) < 1e-15:
        output = source.copy()
        raw_count = int(output.size)
        command: list[str] = []
    else:
        source_path = _write_temp_wav(scratch, source)
        executable = executable_path(sync_config.FFMPEG, "ffmpeg")
        rate_text = format(float(rate), ".17g")
        command = [str(executable), "-y", "-v", "error", "-i", str(source_path), "-filter:a", f"atempo={rate_text}", "-ac", "1", "-ar", str(SAMPLE_RATE), "-acodec", "pcm_s16le", "-f", "s16le", "pipe:1"]
        try:
            result = run_command(command)
            payload = result.stdout
            if len(payload) % 2:
                raise ProtocolError("atempo returned odd PCM byte count")
            output = np.frombuffer(payload, dtype="<i2").copy()
            raw_count = int(output.size)
        finally:
            source_path.unlink(missing_ok=True)
    correction = target - int(output.size)
    if abs(correction) > RATE_MAX_CORRECTION:
        raise ProtocolError(f"RATE_LENGTH_QC_FAILED: correction={correction}")
    if correction > 0:
        output = np.pad(output, (0, correction), mode="constant")
    elif correction < 0:
        output = output[:target]
    if output.size != target:
        raise ProtocolError("audio retime did not reach target length")
    return np.asarray(output, dtype=np.int16), {
        "source_samples": int(source.size),
        "target_samples": target,
        "rate": float(rate),
        "rate_decimal": format(float(rate), ".17g"),
        "command": command,
        "raw_output_samples": raw_count,
        "tail_correction_samples": correction,
        "input_pcm_sha256": _sha_array(source),
        "output_pcm_sha256": _sha_array(output),
        "status": "PASS",
    }


def _forward_visual(engine: SyncNetEngine, frames: np.ndarray) -> np.ndarray:
    torch = engine._torch
    outputs: list[np.ndarray] = []
    if frames.shape[0] < 5:
        raise ProtocolError("visual input has fewer than five frames")
    with torch.inference_mode():
        for start in range(0, frames.shape[0] - 4, engine.batch_size):
            windows = [frames[index : index + 5] for index in range(start, min(frames.shape[0] - 4, start + engine.batch_size))]
            tensor = torch.from_numpy(engine._visual_batch(windows)).to(engine.device)
            outputs.append(engine.network.forward_lip(tensor).detach().cpu().numpy().astype(np.float32))
    value = np.concatenate(outputs, axis=0)
    if value.ndim != 2 or value.shape[1] != sync_config.EMBEDDING_DIM or not np.isfinite(value).all():
        raise ProtocolError(f"invalid visual feature shape: {value.shape}")
    return value


def _forward_audio(engine: SyncNetEngine, pcm: np.ndarray) -> np.ndarray:
    torch = engine._torch
    windows, _ = engine._audio_windows(np.asarray(pcm, dtype=np.int16))
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, windows.shape[0], engine.batch_size):
            batch = torch.from_numpy(windows[start : start + engine.batch_size])[:, None, :, :].to(engine.device)
            outputs.append(engine.network.forward_aud(batch).detach().cpu().numpy().astype(np.float32))
    value = np.concatenate(outputs, axis=0)
    if value.ndim != 2 or value.shape[1] != sync_config.EMBEDDING_DIM or not np.isfinite(value).all():
        raise ProtocolError(f"invalid audio feature shape: {value.shape}")
    return value


def _cell_dir(run_dir: Path, sample_id: int, arm: str, mode: str) -> Path:
    return run_dir / "features" / str(sample_id) / arm / mode


def _receipt_path(run_dir: Path, sample_id: int, arm: str, mode: str) -> Path:
    return run_dir / "receipts" / str(sample_id) / f"{arm}_{mode}.json"


def _extract_one(engine: SyncNetEngine, run_dir: Path, cell: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
    sample_id, arm, mode = int(cell["id"]), str(cell["arm"]), str(cell["mode"])
    receipt_path = _receipt_path(run_dir, sample_id, arm, mode)
    if resume and receipt_path.is_file():
        try:
            old = _load(receipt_path)
            if old.get("status") == "COMPLETE":
                return old
        except Exception:
            pass
    out_dir = _cell_dir(run_dir, sample_id, arm, mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "id": sample_id,
        "arm": arm,
        "mode": mode,
        "source_crop": cell["source_crop"],
        "source_crop_sha256": cell["source_crop_sha256"],
        "source_L": int(cell["source_L"]),
        "target_M": int(cell["target_M"]),
        "rate": float(cell["rate"]),
        "device": engine.device,
        "model_sha256": engine.model_hash,
        "status": "FAILED",
    }
    try:
        frames, pcm = _decode_canonical(Path(cell["source_crop"]), int(cell["source_L"]))
        scratch = run_dir / "scratch"
        if mode == "O":
            transformed_frames = frames
            transformed_pcm = pcm
            frame_receipt = {"status": "IDENTITY", "input_sha256": _sha_array(frames), "output_sha256": _sha_array(frames), "source_L": int(frames.shape[0]), "target_M": int(frames.shape[0]), "rate": 1.0}
            audio_receipt = {"status": "IDENTITY", "input_pcm_sha256": _sha_array(pcm), "output_pcm_sha256": _sha_array(pcm), "source_samples": int(pcm.size), "target_samples": int(pcm.size), "rate": 1.0, "tail_correction_samples": 0, "command": []}
        else:
            transformed_frames, frame_receipt = retime_frames(frames, int(cell["target_M"]))
            transformed_pcm, audio_receipt = retime_audio(pcm, int(cell["target_M"]) * SAMPLES_PER_FRAME, float(cell["rate"]), scratch)
        expected_F = int(cell["target_F"])
        visual = _forward_visual(engine, transformed_frames)[:expected_F]
        audio = _forward_audio(engine, transformed_pcm)[:expected_F]
        if visual.shape[0] != expected_F or audio.shape[0] != expected_F:
            raise ProtocolError(f"feature count mismatch: {visual.shape}/{audio.shape}, expected {expected_F}")
        official = engine.distance_matrix(visual, audio)
        visual_path = out_dir / "v.npy"
        audio_path = out_dir / "a.npy"
        official_path = out_dir / "official.npy"
        result.update({
            "status": "COMPLETE",
            "feature_shape": [int(expected_F), int(visual.shape[1])],
            "visual_path": str(visual_path),
            "audio_path": str(audio_path),
            "official_path": str(official_path),
            "visual_sha256": _save_npy(visual_path, visual),
            "audio_sha256": _save_npy(audio_path, audio),
            "official_sha256": _save_npy(official_path, official),
            "visual_input_sha256": frame_receipt.get("output_sha256"),
            "audio_input_sha256": audio_receipt.get("output_pcm_sha256"),
            "frame_transform": frame_receipt,
            "audio_transform": audio_receipt,
            "alias_of": None,
            "forward_count": 2,
        })
        _write(receipt_path, result, hashed=True)
        return result
    except Exception as exc:
        result["status"] = "FAILED"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        _write(receipt_path, result, hashed=True)
        return result


def extract_features(run_dir: Path, *, device: str = "cuda", resume: bool = False) -> dict[str, Any]:
    media = _load(run_dir / "media.json")
    protocol = _load(run_dir / "protocol.json")
    protocol["device"] = device
    protocol["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": None,
        "torch_cuda": None,
        "model_sha256": file_sha256(MODEL_PATH),
        "syncnet_module_sha256": file_sha256(Path(__file__).parent / "tts_native_gain_attribution" / "syncnet.py"),
    }
    engine = SyncNetEngine(model_path=MODEL_PATH, batch_size=20, device=device)
    try:
        import torch

        protocol["environment"]["torch"] = torch.__version__
        protocol["environment"]["torch_cuda"] = torch.version.cuda
        results: list[dict[str, Any]] = []
        for pair in media.get("records", []):
            cells = pair.get("cells", {})
            ordered = [cells[f"{arm}_{mode}"] for arm in ARMS for mode in MODES]
            # Native features are always materialised first.  A rate=1 matched
            # cell may then use an exact feature alias, but the receipt remains
            # explicit about the logical cell and forward count.
            for cell in ordered:
                if cell.get("status") not in ("PASS", "EXCESS_TAIL_TRIM"):
                    results.append({"id": cell["id"], "arm": cell["arm"], "mode": cell["mode"], "status": cell.get("status"), "reason": "media_audit"})
                    continue
                if cell["mode"] != "O" and abs(float(cell["rate"]) - 1.0) < 1e-15:
                    source_receipt_path = _receipt_path(run_dir, int(cell["id"]), str(cell["arm"]), "O")
                    if source_receipt_path.is_file():
                        source_receipt = _load(source_receipt_path)
                    else:
                        source_receipt = None
                    if source_receipt and source_receipt.get("status") == "COMPLETE":
                        out_dir = _cell_dir(run_dir, int(cell["id"]), str(cell["arm"]), str(cell["mode"]))
                        out_dir.mkdir(parents=True, exist_ok=True)
                        result = dict(source_receipt)
                        result.update({"id": int(cell["id"]), "arm": str(cell["arm"]), "mode": str(cell["mode"]), "source_L": int(cell["source_L"]), "target_M": int(cell["target_M"]), "rate": 1.0, "alias_of": str(source_receipt_path), "forward_count": 0})
                        for key, name in (("visual_path", "v.npy"), ("audio_path", "a.npy"), ("official_path", "official.npy")):
                            source_path = Path(source_receipt[key])
                            target_path = out_dir / name
                            shutil.copy2(source_path, target_path)
                            result[key] = str(target_path)
                            result[key.replace("_path", "_sha256")] = file_sha256(target_path)
                        result["frame_transform"] = {"status": "IDENTITY_ALIAS", "alias_source": str(source_receipt_path)}
                        result["audio_transform"] = {"status": "IDENTITY_ALIAS", "alias_source": str(source_receipt_path)}
                        _write(_receipt_path(run_dir, int(cell["id"]), str(cell["arm"]), str(cell["mode"])), result, hashed=True)
                        results.append(result)
                        continue
                results.append(_extract_one(engine, run_dir, cell, resume=resume))
        protocol["model_hash"] = engine.model_hash
        protocol["code_hash"] = engine.code_hash
        _write(run_dir / "protocol.json", protocol, hashed=True)
        _write(run_dir / "extract_summary.json", {"protocol_id": PROTOCOL_ID, "status": "PASS", "cells": results, "complete_count": sum(row.get("status") == "COMPLETE" for row in results), "cell_count": len(results)}, hashed=True)
        return {"status": "PASS", "cells": results}
    finally:
        engine.close()


def _read_cell(run_dir: Path, sample_id: int, arm: str, mode: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    receipt = _load(_receipt_path(run_dir, sample_id, arm, mode))
    if receipt.get("status") != "COMPLETE":
        raise ProtocolError(f"cell is not complete: {sample_id}/{arm}/{mode}")
    for key in ("visual_path", "audio_path", "official_path"):
        path = Path(receipt[key])
        if not path.is_file() or receipt.get(key.replace("_path", "_sha256")) != file_sha256(path):
            raise ProtocolError(f"cell artifact hash mismatch: {receipt[key]}")
    return receipt, _load_npy(Path(receipt["visual_path"])), _load_npy(Path(receipt["audio_path"])), _load_npy(Path(receipt["official_path"]))


def _cell_analysis(run_dir: Path, pair: dict[str, Any], arm: str, mode: str, phase_records: list[dict[str, Any]], *, unit: bool = False) -> dict[str, Any]:
    receipt, visual, audio, official = _read_cell(run_dir, int(pair["id"]), arm, mode)
    F = int(visual.shape[0])
    support = metrics.support_for_halves(F)
    if not support["complete"]:
        raise ProtocolError(f"support insufficient for {pair['id']}/{arm}/{mode}")
    calibrations = {
        "H0": metrics.calibrate_lag(visual, audio, support["halves"]["H1"]["rows"]),
        "H1": metrics.calibrate_lag(visual, audio, support["halves"]["H0"]["rows"]),
    }
    if unit:
        visual_value = visual / np.linalg.norm(visual, axis=1, keepdims=True)
        audio_value = audio / np.linalg.norm(audio, axis=1, keepdims=True)
    else:
        visual_value, audio_value = visual, audio
    ranked = metrics.rank_cell(visual_value, audio_value, phase_records, f"{arm}_{mode}", calibrations)
    return {
        "arm": arm,
        "mode": mode,
        "F": F,
        "support": support,
        "calibration": calibrations,
        "rank": ranked,
        "rank_metric": "unit" if unit else "raw",
        "c_interior": metrics.official_c_interior(official),
        "receipt": {"visual_sha256": receipt.get("visual_sha256"), "audio_sha256": receipt.get("audio_sha256"), "official_sha256": receipt.get("official_sha256")},
    }


def _summary(values: list[float], *, primary: bool = False) -> dict[str, Any]:
    return metrics.bootstrap_summary(values, primary=primary, seed=BOOTSTRAP_SEED, draws=metrics.BOOTSTRAP_DRAWS)


def analyze_run(run_dir: Path) -> dict[str, Any]:
    media = _load(run_dir / "media.json")
    pair_metrics: list[dict[str, Any]] = []
    for pair in media.get("records", []):
        sample_id = int(pair["id"])
        cell_data: dict[str, dict[str, Any]] = {}
        reasons: list[str] = []
        native_only: dict[str, Any] = {"status": "INCOMPLETE", "reasons": []}
        try:
            native_counts: dict[str, int] = {}
            for arm in ARMS:
                receipt, visual, _, _ = _read_cell(run_dir, sample_id, arm, "O")
                native_counts[f"{arm}_O"] = int(visual.shape[0])
                if receipt.get("status") != "COMPLETE":
                    raise ProtocolError(f"native receipt is incomplete: {arm}_O")
            native_phases = metrics.build_common_phases(native_counts)
            if not native_phases["complete"]:
                native_only["reasons"] = ["native_common_phase_support"]
            else:
                native_cells = {f"{arm}_O": _cell_analysis(run_dir, pair, arm, "O", native_phases["records"]) for arm in ARMS}
                native_only = {
                    "status": "COMPLETE",
                    "cells": native_cells,
                    "H_native": float(native_cells["T_O"]["rank"]["R"] - native_cells["N_O"]["rank"]["R"]),
                    "delta_C_interior": float(native_cells["T_O"]["c_interior"].get("C", 0.0) - native_cells["N_O"]["c_interior"].get("C", 0.0)),
                    "phase_counts": native_phases["distinct_index_counts"],
                    "reasons": [],
                }
        except Exception as exc:
            native_only["reasons"] = [f"native_analysis:{type(exc).__name__}:{exc}"]
        try:
            frame_counts: dict[str, int] = {}
            for arm in ARMS:
                for mode in MODES:
                    receipt, visual, _, _ = _read_cell(run_dir, sample_id, arm, mode)
                    frame_counts[f"{arm}_{mode}"] = int(visual.shape[0])
                    if receipt.get("audio_transform", {}).get("status") not in ("PASS", "IDENTITY", "IDENTITY_ALIAS"):
                        reasons.append(f"{arm}_{mode}_transform")
            phase_bundle = metrics.build_common_phases(frame_counts)
            if not phase_bundle["complete"]:
                reasons.append("common_phase_support")
            phase_records = phase_bundle["records"]
            if not reasons:
                for arm in ARMS:
                    for mode in MODES:
                        cell_data[f"{arm}_{mode}"] = _cell_analysis(run_dir, pair, arm, mode, phase_records)
        except Exception as exc:
            reasons.append(f"analysis:{type(exc).__name__}:{exc}")
        pair_metrics.append({
            "id": sample_id,
            "paired_key": pair.get("paired_key"),
            "L_N": int(pair.get("L_N", 0)),
            "L_T": int(pair.get("L_T", 0)),
            "M_SHORT": int(pair.get("M_SHORT", 0)),
            "M_LONG": int(pair.get("M_LONG", 0)),
            "cells": cell_data,
            "native_only": native_only,
            "status": "COMPLETE" if not reasons and len(cell_data) == 6 else "INCOMPLETE",
            "reasons": reasons,
        })

    complete = [row for row in pair_metrics if row["status"] == "COMPLETE"]
    native_only_rows = [row for row in pair_metrics if row.get("native_only", {}).get("status") == "COMPLETE"]
    for row in complete:
        cells = row["cells"]
        row["H_native"] = float(cells["T_O"]["rank"]["R"] - cells["N_O"]["rank"]["R"])
        row["H_rate"] = float(0.5 * ((cells["T_SHORT"]["rank"]["R"] - cells["N_SHORT"]["rank"]["R"]) + (cells["T_LONG"]["rank"]["R"] - cells["N_LONG"]["rank"]["R"])))
        row["H_short"] = float(cells["T_SHORT"]["rank"]["R"] - cells["N_SHORT"]["rank"]["R"])
        row["H_long"] = float(cells["T_LONG"]["rank"]["R"] - cells["N_LONG"]["rank"]["R"])
        row["H_zero"] = float(0.5 * ((cells["T_O"]["rank"]["R_zero"] - cells["N_O"]["rank"]["R_zero"]) + (cells["T_SHORT"]["rank"]["R_zero"] - cells["N_SHORT"]["rank"]["R_zero"])))
        row["delta_C_interior"] = float(cells["T_O"]["c_interior"].get("C", 0.0) - cells["N_O"]["c_interior"].get("C", 0.0))
        row["mean_R"] = {key: float(cells[key]["rank"]["R"]) for key in cells}
        row["boundary_cells"] = [key for key, value in cells.items() if any(value["calibration"][half].get("boundary") for half in ("H0", "H1"))]

    ids = [int(row["id"]) for row in complete]
    values = {name: [float(row[name]) for row in complete] for name in ("H_native", "H_rate", "H_short", "H_long", "H_zero", "delta_C_interior")}
    primary_native = _summary(values["H_native"], primary=True)
    primary_rate = _summary(values["H_rate"], primary=True)
    summaries = {name: _summary(value) for name, value in values.items()}
    native_c = _summary(values["delta_C_interior"])
    boundary_count = sum(bool(row.get("boundary_cells")) for row in complete)
    all_r_values = {key: [float(row["mean_R"][key]) for row in complete] for key in ("N_O", "N_SHORT", "N_LONG", "T_O", "T_SHORT", "T_LONG")}
    chance = {key: metrics.chance_status(value, seed=BOOTSTRAP_SEED) for key, value in all_r_values.items()}
    transform_ok = True
    for media_pair in media.get("records", []):
        for arm in ARMS:
            for mode in MODES:
                receipt_path = _receipt_path(run_dir, int(media_pair["id"]), arm, mode)
                try:
                    receipt = _load(receipt_path)
                except Exception:
                    transform_ok = False
                    continue
                if (
                    receipt.get("status") != "COMPLETE"
                    or receipt.get("frame_transform", {}).get("status") not in ("IDENTITY", "PASS", "IDENTITY_ALIAS")
                    or receipt.get("audio_transform", {}).get("status") not in ("IDENTITY", "PASS", "IDENTITY_ALIAS")
                ):
                    transform_ok = False
    statuses = {
        "engineering_status": "PASS" if (run_dir / "extract_summary.json").is_file() else "INCOMPLETE",
        "support_status": "SUFFICIENT" if len(complete) >= 30 else "INSUFFICIENT_PAIRS",
        "calibration_status": "CALIBRATION_RANGE_LIMITED" if complete and boundary_count / len(complete) > 0.10 else "PASS",
        "rate_control_status": "PASS" if transform_ok else "RATE_CONTROL_FAILED",
        "native_gain_status": "CONFIRMED" if native_c.get("status") == "COMPLETE" and native_c["ci95"][0] > 0.0 else "UNCONFIRMED",
        "H_native_status": metrics.status_from_ci(primary_native),
        "H_rate_status": metrics.status_from_ci(primary_rate),
        "temporal_validity": "ALL_ABOVE_CHANCE" if chance and all(item["status"] == "ALL_ABOVE_CHANCE" for item in chance.values()) else "INCONCLUSIVE",
    }
    if statuses["engineering_status"] != "PASS":
        mechanism_status = "TECHNICAL_INVALID"
    elif statuses["support_status"] != "SUFFICIENT":
        mechanism_status = "INSUFFICIENT_PAIRS"
    elif statuses["calibration_status"] != "PASS":
        mechanism_status = "CALIBRATION_RANGE_LIMITED"
    elif statuses["native_gain_status"] != "CONFIRMED":
        mechanism_status = "NATIVE_GAIN_UNCONFIRMED"
    elif statuses["H_native_status"] != "POSITIVE":
        mechanism_status = "NO_CONFIRMED_NATIVE_LOCAL_ADVANTAGE"
    elif statuses["rate_control_status"] != "PASS":
        mechanism_status = "RATE_CONTROL_FAILED"
    elif statuses["temporal_validity"] != "ALL_ABOVE_CHANCE":
        mechanism_status = "TEMPORAL_MEASUREMENT_INCONCLUSIVE"
    elif statuses["H_native_status"] == "POSITIVE" and statuses["H_rate_status"] == "POSITIVE" and summaries["H_short"]["mean"] > 0 and summaries["H_long"]["mean"] > 0:
        mechanism_status = "LOCAL_ADVANTAGE_SURVIVES_GLOBAL_CONTROLS"
    elif statuses["H_native_status"] == "POSITIVE" and statuses["H_rate_status"] != "POSITIVE":
        mechanism_status = "RATE_TEST_INCONCLUSIVE"
    elif statuses["H_rate_status"] == "POSITIVE" and (summaries["H_short"]["mean"] <= 0 or summaries["H_long"]["mean"] <= 0):
        mechanism_status = "TARGET_RATE_DEPENDENT"
    else:
        mechanism_status = "INCONCLUSIVE"
    analysis = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "candidate_count": len(pair_metrics),
        "complete_count": len(complete),
        "ids": ids,
        "excluded_ids": [int(row["id"]) for row in pair_metrics if row["status"] != "COMPLETE"],
        "excluded_reasons": {str(int(row["id"])): row.get("reasons", []) for row in pair_metrics if row["status"] != "COMPLETE"},
        "native_only_count": len(native_only_rows),
        "native_only_ids": [int(row["id"]) for row in native_only_rows],
        "native_only_summary": _summary([float(row["native_only"]["H_native"]) for row in native_only_rows]),
        "statuses": {**statuses, "mechanism_status": mechanism_status},
        "primary": {"H_native": primary_native, "H_rate": primary_rate},
        "summaries": summaries,
        "native_c_interior": native_c,
        "chance": chance,
        "boundary_pair_count": boundary_count,
        "pair_metrics_path": str(run_dir / "pair_metrics.json"),
        "limitations": ["single speaker S0765", "historical Ditto cohort", "global duration control is not phoneme-level", "retiming artifacts remain a competing explanation"],
    }
    _write(run_dir / "pair_metrics.json", {"schema_version": 1, "protocol_id": PROTOCOL_ID, "pairs": pair_metrics, "J": ids, "J_A": [int(row["id"]) for row in native_only_rows]}, hashed=True)
    np.savez(run_dir / "row_metrics.npz", ids=np.asarray(ids, dtype=np.int64), H_native=np.asarray(values["H_native"], dtype=np.float64), H_rate=np.asarray(values["H_rate"], dtype=np.float64), H_short=np.asarray(values["H_short"], dtype=np.float64), H_long=np.asarray(values["H_long"], dtype=np.float64), delta_C_interior=np.asarray(values["delta_C_interior"], dtype=np.float64))
    _write(run_dir / "analysis.json", analysis, hashed=True)
    _write_report(run_dir, analysis)
    _plot_curves(run_dir, complete)
    return analysis


def _write_report(run_dir: Path, analysis: dict[str, Any]) -> None:
    primary = analysis["primary"]
    lines = [
        f"# Ditto timing/rate experiment ({PROTOCOL_ID})",
        "",
        f"- complete pairs J: {analysis['complete_count']}",
        f"- native-only pairs J_A: {analysis['native_only_count']}",
        f"- excluded IDs: {analysis['excluded_ids']}",
        f"- exclusion reasons: {analysis['excluded_reasons']}",
        f"- mechanism status: `{analysis['statuses']['mechanism_status']}`",
        f"- H_native mean/CI: {primary['H_native'].get('mean')} / {primary['H_native'].get('ci95')}",
        f"- H_rate mean/CI: {primary['H_rate'].get('mean')} / {primary['H_rate'].get('ci95')}",
        f"- native C_interior status: {analysis['statuses']['native_gain_status']}",
        "",
        "该报告只描述冻结 SyncNet 在输入窗口隔离、共同相对进度和整句变速控制下的响应，不能作为嘴型真值或因果证明。",
    ]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_curves(run_dir: Path, pairs: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = ["O", "SHORT", "LONG"]
    n_values = []
    t_values = []
    for mode in labels:
        n_values.append([row["cells"][f"N_{mode}"]["rank"]["R"] for row in pairs])
        t_values.append([row["cells"][f"T_{mode}"]["rank"]["R"] for row in pairs])
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(3)
    means_n = [float(np.mean(v)) if v else np.nan for v in n_values]
    means_t = [float(np.mean(v)) if v else np.nan for v in t_values]
    ax.plot(x, means_n, marker="o", label="natural")
    ax.plot(x, means_t, marker="o", label="TTS")
    ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x, labels)
    ax.set_ylabel("local rank score")
    ax.set_title(f"Ditto timing/rate, n={len(pairs)}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "timing_curves.png", dpi=140)
    plt.close(fig)


def run_validation(run_dir: Path) -> dict[str, Any]:
    from scripts.experiments.check_ditto_timing_rate import validate_run

    return validate_run(run_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "prepare", "extract", "analyze", "validate", "all"), default="all")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    try:
        stages = [args.stage] if args.stage != "all" else ["audit", "prepare", "extract", "analyze", "validate"]
        result: Any = None
        for stage in stages:
            if stage == "audit":
                result = audit_inputs(run_dir, smoke=bool(args.smoke))
            elif stage == "prepare":
                result = prepare_media(run_dir)
            elif stage == "extract":
                result = extract_features(run_dir, device=args.device, resume=args.resume)
            elif stage == "analyze":
                result = analyze_run(run_dir)
            elif stage == "validate":
                result = run_validation(run_dir)
        if isinstance(result, dict):
            print(json.dumps({"stage": args.stage, "status": result.get("status", "PASS"), "run_dir": str(run_dir), "mechanism_status": result.get("statuses", {}).get("mechanism_status") if isinstance(result.get("statuses"), dict) else None}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
