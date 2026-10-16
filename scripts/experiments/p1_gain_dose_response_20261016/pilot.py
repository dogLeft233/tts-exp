from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
WAV2LIP = REPO / "third_party/Wav2Lip"
WAV2LIP_PY = Path("/home/wjj/.venvs/wav2lip/bin/python")
SYNCNET_PY = Path("/home/wjj/.venvs/syncnet/bin/python")
SYNCNET_MODEL = REPO / "third_party/syncnet_python/data/syncnet_v2.model"
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
FFPROBE = Path("/home/wjj/miniconda3/bin/ffprobe")
FACE = REPO / "data/data/image/3.png"
WAV2LIP_CHECKPOINT = WAV2LIP / "checkpoints/wav2lip_gan.pth"
WAV2LIP_INFERENCE = WAV2LIP / "inference.py"
SEEDED_INFERENCE_WRAPPER = REPO / "scripts/experiments/p1_gain_dose_response_20261016/wav2lip_seeded_inference.py"
WAV2LIP_CHECKPOINT_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
WAV2LIP_INFERENCE_SHA256 = "42cfc8d3060921e6714c8e08243e5508d2e637d29cd0be78760492c5d26380a9"
FACE_SHA256 = "f425463ba279fea88ac807d02bf048fd2f891d4efae1c8903eae3afc22d216c9"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
TARGET_RMS = 0.0376838172675746
SAMPLE_RATE = 16_000
FPS = 25
LAG_RADIUS = 15
SCORE_BOX_XYXY = (33, 18, 462, 447)
RAW_SEED = 2026101601
REPEAT2_SEED = 2026101602
EXPECTED_CONDITIONS = ("RAW_R1", "RAW_R2", "FIXED_R1", "FIXED_R2")


class PilotError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, payload: dict[str, Any], *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    if replace:
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        with temporary.open("xb") as stream:
            stream.write(raw)
        os.replace(temporary, path)
    else:
        with path.open("xb") as stream:
            stream.write(raw)


def process_start_ticks(pid: int) -> str | None:
    try:
        record = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        tail = record[record.rfind(")") + 2 :].split()
        return tail[19]
    except (OSError, IndexError):
        return None


def load_packet(packet_path: Path, source_manifest_path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    packet_raw = packet_path.read_bytes()
    packet = json.loads(packet_raw.decode("utf-8"))
    packet_sha = hashlib.sha256(packet_raw).hexdigest()
    manifest_raw = source_manifest_path.read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    actual_manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    expected_manifest_sha = packet.get("source_manifest_sha256")
    if actual_manifest_sha != expected_manifest_sha:
        raise PilotError("source manifest file hash does not match the reviewed packet")
    payload_hash = canonical_sha256({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    if payload_hash != manifest.get("manifest_sha256"):
        raise PilotError("source manifest internal fingerprint mismatch")
    if packet.get("source_manifest_payload_sha256") != manifest.get("manifest_sha256"):
        raise PilotError("source manifest payload fingerprint differs from reviewed packet")
    script_hashes = packet.get("source_code_sha256", {})
    for relative, expected in script_hashes.items():
        path = REPO / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise PilotError(f"source code pin mismatch: {relative}")
    return packet, packet_sha, manifest


def read_pcm16(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    with wave.open(str(path), "rb") as stream:
        info = {
            "channels": stream.getnchannels(),
            "sample_width_bytes": stream.getsampwidth(),
            "sample_rate_hz": stream.getframerate(),
            "frames": stream.getnframes(),
        }
        payload = stream.readframes(stream.getnframes())
    if info["channels"] != 1 or info["sample_width_bytes"] != 2 or info["sample_rate_hz"] != SAMPLE_RATE:
        raise PilotError(f"natural source is not mono PCM16/16 kHz: {path}: {info}")
    values = np.frombuffer(payload, dtype="<i2").copy()
    if values.size != info["frames"]:
        raise PilotError(f"source frame count mismatch: {path}")
    return values, info


def read_float_wav(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    raw = path.read_bytes()
    if len(raw) < 44 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE" or raw[12:16] != b"fmt ":
        raise PilotError(f"generated float WAV header is invalid: {path}")
    fmt_size, format_tag, channels, rate, byte_rate, block_align, bits = struct.unpack_from("<IHHIIHH", raw, 16)
    data_offset = 20 + fmt_size
    if raw[data_offset : data_offset + 4] != b"data":
        raise PilotError(f"generated float WAV data chunk is not in the frozen position: {path}")
    data_size = struct.unpack_from("<I", raw, data_offset + 4)[0]
    payload = raw[data_offset + 8 : data_offset + 8 + data_size]
    if (format_tag, channels, rate, byte_rate, block_align, bits) != (3, 1, SAMPLE_RATE, 64_000, 4, 32):
        raise PilotError(f"generated audio is not mono IEEE-float32/16 kHz: {path}")
    if len(payload) != data_size or data_size % 4:
        raise PilotError(f"generated float WAV payload length is invalid: {path}")
    samples = np.frombuffer(payload, dtype="<f4").copy()
    if fmt_size < 16 or samples.size < 1 or not np.isfinite(samples).all():
        raise PilotError(f"generated float WAV contains invalid samples: {path}")
    return samples, {"sample_rate_hz": rate, "channels": channels, "frames": int(samples.size), "format_tag": format_tag, "bits_per_sample": bits}


def write_float_wav(path: Path, samples: np.ndarray) -> None:
    value = np.asarray(samples, dtype="<f4")
    if value.ndim != 1 or value.size < 1 or not np.isfinite(value).all():
        raise PilotError("scaled audio must be finite nonempty mono float32")
    payload = np.ascontiguousarray(value).tobytes()
    fmt = struct.pack("<IHHIIHH", 16, 3, 1, SAMPLE_RATE, SAMPLE_RATE * 4, 4, 32)
    body = b"WAVEfmt " + fmt + b"data" + struct.pack("<I", len(payload)) + payload
    with path.open("xb") as stream:
        stream.write(b"RIFF" + struct.pack("<I", len(body)) + body)


def waveform_stats(samples: np.ndarray) -> dict[str, float]:
    value = np.asarray(samples, dtype=np.float64)
    rms = float(np.sqrt(np.mean(value * value, dtype=np.float64)))
    peak = float(np.max(np.abs(value)))
    return {
        "rms": rms,
        "rms_dbfs": float(20.0 * np.log10(rms)) if rms > 0 else float("-inf"),
        "peak_abs": peak,
        "frames": int(value.size),
        "duration_s": float(value.size / SAMPLE_RATE),
    }


def scaled_float32(source: np.ndarray, gain: float) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(source, dtype=np.float32)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all() or not math.isfinite(gain) or gain <= 0:
        raise PilotError("scale input/gain is invalid")
    y = (x.astype(np.float64) * np.float64(gain)).astype(np.float32)
    if not np.isfinite(y).all() or float(np.max(np.abs(y))) >= 1.0:
        raise PilotError("uniform gain would clip or create a nonfinite waveform")
    expected = x.astype(np.float64) * np.float64(gain)
    residual = y.astype(np.float64) - expected
    rel_l2 = float(np.linalg.norm(residual) / max(np.linalg.norm(expected), 1e-30))
    inverse = y.astype(np.float64) / np.float64(gain)
    inverse_rel_l2 = float(np.linalg.norm(inverse - x.astype(np.float64)) / max(np.linalg.norm(x), 1e-30))
    spectrum_x = np.abs(np.fft.rfft(x.astype(np.float64)))
    spectrum_y = np.abs(np.fft.rfft(y.astype(np.float64)))
    shape_x = spectrum_x / max(float(spectrum_x.sum()), 1e-30)
    shape_y = spectrum_y / max(float(spectrum_y.sum()), 1e-30)
    shape_error = float(np.max(np.abs(shape_y - shape_x)))
    if rel_l2 > 1e-7 or inverse_rel_l2 > 1e-7 or shape_error > 1e-7:
        raise PilotError("float32 scaling failed the exact-scalar numerical tolerance")
    return y, {
        "gain": float(gain),
        "gain_db": float(20.0 * np.log10(gain)),
        "relative_l2_residual": rel_l2,
        "inverse_relative_l2_residual": inverse_rel_l2,
        "normalized_spectrum_shape_max_abs_error": shape_error,
        "clock_preserved": True,
        "frame_count_preserved": True,
        "no_clipping": True,
        "finite": True,
        "input_stats": waveform_stats(x),
        "output_stats": waveform_stats(y),
    }


def dose_id(db: float) -> str:
    if db == 0:
        return "RAW"
    text = f"{abs(db):g}".replace(".", "P")
    return f"DOSE_{'P' if db > 0 else 'M'}{text}"


def condition_plan(dose_db: float) -> list[dict[str, Any]]:
    if not math.isfinite(dose_db) or dose_db <= 0:
        raise PilotError("locked symmetric dose must be positive and finite")
    dose_rows = [
        {"condition_id": "RAW_R1", "kind": "RAW", "repeat": 1, "audio_variant": "RAW", "relative_dose_db": 0.0, "seed": RAW_SEED, "dose_curve_zero_alias": True},
        {"condition_id": "RAW_R2", "kind": "RAW", "repeat": 2, "audio_variant": "RAW", "relative_dose_db": 0.0, "seed": REPEAT2_SEED, "dose_curve_zero_alias": False},
        {"condition_id": "FIXED_R1", "kind": "FIXED", "repeat": 1, "audio_variant": "FIXED", "relative_dose_db": None, "seed": RAW_SEED, "dose_curve_zero_alias": False},
        {"condition_id": "FIXED_R2", "kind": "FIXED", "repeat": 2, "audio_variant": "FIXED", "relative_dose_db": None, "seed": REPEAT2_SEED, "dose_curve_zero_alias": False},
    ]
    for level in (-dose_db, -dose_db / 2.0, dose_db / 2.0, dose_db):
        dose_rows.append({"condition_id": dose_id(level), "kind": "DOSE", "repeat": 1, "audio_variant": dose_id(level), "relative_dose_db": float(level), "seed": RAW_SEED, "dose_curve_zero_alias": False})
    return dose_rows


def prepare_audio(packet_path: Path, source_manifest_path: Path, source_audio_root: Path, output_root: Path) -> None:
    packet, packet_sha, source_manifest = load_packet(packet_path, source_manifest_path)
    expected_dose = packet.get("locked_symmetric_dose_db")
    if expected_dose != source_manifest.get("uniform_dose_safety", {}).get("selected_max_symmetric_dose_db"):
        raise PilotError("packet dose differs from pre-QA cohort result")
    if expected_dose is None:
        raise PilotError("the frozen roster has no safe uniform positive dose")
    if output_root.exists():
        raise FileExistsError(f"run output root already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    variant_root = output_root / "audio_variants"
    variant_root.mkdir()
    speakers: list[dict[str, Any]] = []
    plan = condition_plan(float(expected_dose))
    for source_row in source_manifest["cohort"]:
        sid = source_row["speaker"]
        input_path = source_audio_root / sid / source_row["filename"]
        if sha256_file(input_path) != source_row["audio_sha256"]:
            raise PilotError(f"uploaded source PCM hash mismatch for {sid}")
        pcm, info = read_pcm16(input_path)
        x = pcm.astype(np.float32) / np.float32(32768.0)
        raw_stats = waveform_stats(x)
        if info["frames"] != source_row["audio_audit"]["frames"]:
            raise PilotError(f"uploaded source PCM frame count changed for {sid}")
        sid_root = variant_root / sid
        sid_root.mkdir()
        rms = raw_stats["rms"]
        if rms <= 0:
            raise PilotError(f"source PCM is silent for {sid}")
        gains: dict[str, float] = {"FIXED": TARGET_RMS / rms}
        levels = source_manifest["uniform_dose_safety"]["levels_db"]
        for level in levels:
            if level != 0:
                gains[dose_id(float(level))] = 10.0 ** (float(level) / 20.0)
        variants: dict[str, dict[str, Any]] = {
            "RAW": {"path": str(input_path.resolve()), "sha256": source_row["audio_sha256"], "source_pcm_identity": True, "stats": raw_stats, "relative_dose_db": 0.0}
        }
        for variant_id, gain in gains.items():
            samples, numerical = scaled_float32(x, gain)
            path = sid_root / f"{variant_id}.wav"
            write_float_wav(path, samples)
            reread, header = read_float_wav(path)
            if not np.array_equal(reread, samples) or header["frames"] != info["frames"]:
                raise PilotError(f"scaled waveform roundtrip changed values/clock for {sid}/{variant_id}")
            stats = waveform_stats(reread)
            if variant_id == "FIXED":
                relative_db = float(20.0 * np.log10(stats["rms"] / raw_stats["rms"]))
            else:
                relative_db = float(numerical["gain_db"])
            variants[variant_id] = {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "pcm_sha256": hashlib.sha256(np.ascontiguousarray(reread, dtype="<f4").tobytes()).hexdigest(),
                "float32_wav": True,
                "sample_rate_hz": header["sample_rate_hz"],
                "frame_count": header["frames"],
                "audio_rms": stats["rms"],
                "audio_rms_dbfs": stats["rms_dbfs"],
                "relative_dose_db": relative_db,
                "numerical_verification": numerical,
            }
        speaker_plan = []
        for item in plan:
            row = dict(item)
            variant = variants[row["audio_variant"]]
            if row["kind"] == "FIXED":
                row["relative_dose_db"] = variant["relative_dose_db"]
            row.update({"audio_path": variant["path"], "audio_sha256": variant["sha256"], "audio_rms_dbfs": variant["stats"]["rms_dbfs"] if "stats" in variant else variant["audio_rms_dbfs"]})
            speaker_plan.append(row)
        speakers.append({
            "speaker_id": sid,
            "source_audio_path": str(input_path.resolve()),
            "source_sha256": source_row["audio_sha256"],
            "sample_rate_hz": SAMPLE_RATE,
            "frame_count": info["frames"],
            "duration_s": raw_stats["duration_s"],
            "source_rms": raw_stats["rms"],
            "source_rms_dbfs": raw_stats["rms_dbfs"],
            "conditions": speaker_plan,
            "audio_variants": variants,
        })
    manifest = {
        "schema_version": 1,
        "status": "AUDIO_VARIANTS_READY",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": packet_sha,
        "source_manifest_sha256": packet["source_manifest_sha256"],
        "source_manifest_payload_sha256": source_manifest["manifest_sha256"],
        "locked_symmetric_dose_db": float(expected_dose),
        "dose_levels_db": source_manifest["uniform_dose_safety"]["levels_db"],
        "condition_plan": plan,
        "speaker_count": len(speakers),
        "variant_generation": "whole-waveform positive scalar only; FLOAT32 WAV; no filtering, compressor, limiter, resampling, or clipping",
        "speakers": speakers,
        "renderer_forwards_performed": False,
        "scoring_forwards_performed": False,
    }
    manifest_path = output_root / "audio_variant_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "protocol_sha256": packet_sha, "manifest": str(manifest_path), "speakers": len(speakers), "variants_per_speaker": 6}, ensure_ascii=False), flush=True)


def expected_pins(packet: dict[str, Any]) -> None:
    assets = {
        "wav2lip_checkpoint": (WAV2LIP_CHECKPOINT, WAV2LIP_CHECKPOINT_SHA256),
        "wav2lip_inference": (WAV2LIP_INFERENCE, WAV2LIP_INFERENCE_SHA256),
        "face": (FACE, FACE_SHA256),
        "syncnet_model": (SYNCNET_MODEL, SYNCNET_MODEL_SHA256),
    }
    for name, (path, digest) in assets.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise PilotError(f"{name} source pin mismatch: {path}")
        if packet.get("asset_sha256", {}).get(name) != digest:
            raise PilotError(f"reviewed packet does not bind {name}")


def prepare_render_attempt_dirs(video: Path, log: Path, work: Path) -> None:
    video.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=False)
    (work / "temp").mkdir()


def render(packet_path: Path, source_manifest_path: Path, output_root: Path, *, speaker_limit: int | None = None, continue_existing: bool = False) -> None:
    packet, packet_sha, source_manifest = load_packet(packet_path, source_manifest_path)
    expected_pins(packet)
    audio_manifest_path = output_root / "audio_variant_manifest.json"
    audio_manifest = json.loads(audio_manifest_path.read_text(encoding="utf-8"))
    if audio_manifest.get("protocol_sha256") != packet_sha or audio_manifest.get("source_manifest_sha256") != packet["source_manifest_sha256"]:
        raise PilotError("audio variants are not bound to the current reviewed packet")
    conditions = audio_manifest["condition_plan"]
    if len(conditions) != 8:
        raise PilotError("frozen render plan must contain exactly eight unique video conditions")
    if speaker_limit is not None and not 1 <= speaker_limit <= len(audio_manifest["speakers"]):
        raise PilotError("speaker limit must select at least one planned speaker")
    planned = len(audio_manifest["speakers"]) * len(conditions)
    receipt_path = output_root / "render_receipt.json"
    if receipt_path.exists():
        if not continue_existing or speaker_limit is not None:
            raise FileExistsError("render receipt already exists; use --continue to add only unattempted planned rows")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("protocol_sha256") != packet_sha or receipt.get("source_manifest_sha256") != packet["source_manifest_sha256"] or receipt.get("planned_video_count") != planned:
            raise PilotError("existing render receipt is not bound to the reviewed plan")
    else:
        if continue_existing:
            raise FileNotFoundError("--continue requested without an existing render receipt")
        receipt = {
            "schema_version": 1,
            "status": "RUNNING",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": packet_sha,
            "source_manifest_sha256": packet["source_manifest_sha256"],
            "parent_pid": os.getpid(),
            "parent_start_ticks": process_start_ticks(os.getpid()),
            "planned_video_count": planned,
            "attempted_video_count": 0,
            "conditions": conditions,
            "attempts": [],
            "renderer": {"python": str(WAV2LIP_PY), "inference": str(WAV2LIP_INFERENCE), "checkpoint": str(WAV2LIP_CHECKPOINT), "face": str(FACE), "face_sha256": FACE_SHA256, "fps_hz": FPS},
        }
    video_root = output_root / "videos"
    logs_root = output_root / "logs/wav2lip"
    work_root = output_root / "wav2lip_work"
    video_root.mkdir(exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(exist_ok=True)
    if not receipt_path.exists():
        write_json(receipt_path, receipt)
    attempted_keys = [(row.get("speaker_id"), row.get("condition_id")) for row in receipt["attempts"]]
    if len(attempted_keys) != len(set(attempted_keys)):
        raise PilotError("render receipt contains duplicate speaker/condition attempts")
    attempted = set(attempted_keys)
    speaker_rows = audio_manifest["speakers"] if speaker_limit is None else audio_manifest["speakers"][:speaker_limit]
    for speaker_row in speaker_rows:
        sid = speaker_row["speaker_id"]
        for condition in conditions:
            condition_id = condition["condition_id"]
            if (sid, condition_id) in attempted:
                continue
            speaker_condition = condition_row(speaker_row, condition_id)
            audio = Path(speaker_condition["audio_path"])
            if not audio.is_file() or sha256_file(audio) != speaker_condition["audio_sha256"]:
                attempt = {"speaker_id": sid, "condition_id": condition_id, "status": "INPUT_AUDIO_MISSING_OR_CHANGED", "returncode": None, "pid": None, "start_ticks": None}
                receipt["attempts"].append(attempt)
                receipt["attempted_video_count"] += 1
                write_json(receipt_path, receipt, replace=True)
                continue
            video = video_root / sid / f"{condition_id}.mp4"
            log = logs_root / sid / f"{condition_id}.log"
            work = work_root / sid / condition_id
            if video.exists() or log.exists() or work.exists():
                raise FileExistsError(f"refusing to overwrite render attempt: {sid}/{condition_id}")
            prepare_render_attempt_dirs(video, log, work)
            command = [
                str(WAV2LIP_PY), str(SEEDED_INFERENCE_WRAPPER), str(WAV2LIP_INFERENCE),
                "--checkpoint_path", str(WAV2LIP_CHECKPOINT),
                "--face", str(FACE),
                "--audio", str(audio),
                "--outfile", str(video),
                "--face_det_batch_size", "4",
                "--wav2lip_batch_size", "4",
                "--nosmooth",
            ]
            env = dict(os.environ)
            env["PYTHONHASHSEED"] = str(condition["seed"])
            env["P1_RENDER_SEED"] = str(condition["seed"])
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            started = datetime.now(timezone.utc).isoformat()
            with log.open("xb") as log_stream:
                child = subprocess.Popen(command, cwd=work, env=env, stdout=log_stream, stderr=subprocess.STDOUT, start_new_session=True)
                ticks = process_start_ticks(child.pid)
                returncode = child.wait()
            seed_config = None
            seed_config_prefix = "P1_RENDER_SEED_CONFIG="
            with log.open("r", encoding="utf-8", errors="replace") as log_stream:
                for line in log_stream:
                    if line.startswith(seed_config_prefix):
                        if seed_config is not None:
                            seed_config = None
                            break
                        try:
                            seed_config = json.loads(line[len(seed_config_prefix) :])
                        except json.JSONDecodeError:
                            seed_config = None
                            break
            expected_seed_config = {
                "seed": condition["seed"],
                "pythonhashseed": str(condition["seed"]),
                "cublas_workspace_config": ":4096:8",
                "python_random_seed": condition["seed"],
                "numpy_seed": condition["seed"],
                "torch_seed": condition["seed"],
                "cuda_seed_all": condition["seed"],
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
            }
            seed_config_valid = isinstance(seed_config, dict) and all(seed_config.get(key) == value for key, value in expected_seed_config.items())
            complete = returncode == 0 and video.is_file() and video.stat().st_size > 0 and seed_config_valid
            attempt = {
                "speaker_id": sid,
                "condition_id": condition_id,
                "kind": condition["kind"],
                "repeat": condition["repeat"],
                "seed": condition["seed"],
                "relative_dose_db": speaker_condition["relative_dose_db"],
                "audio_rms_dbfs": speaker_condition["audio_rms_dbfs"],
                "audio_path": str(audio.resolve()),
                "audio_sha256": sha256_file(audio),
                "video_path": str(video.resolve()),
                "video_sha256": sha256_file(video) if video.is_file() and video.stat().st_size > 0 else None,
                "video_bytes": video.stat().st_size if video.is_file() else 0,
                "command": command,
                "seed_runtime": seed_config,
                "seed_runtime_valid": seed_config_valid,
                "seed_wrapper_sha256": sha256_file(SEEDED_INFERENCE_WRAPPER),
                "cwd": str(work.resolve()),
                "stdout_stderr_log": str(log.resolve()),
                "log_sha256": sha256_file(log),
                "started_utc": started,
                "pid": child.pid,
                "start_ticks": ticks,
                "returncode": returncode,
                "status": "RENDERED" if complete else "RENDER_FAILED",
            }
            receipt["attempts"].append(attempt)
            attempted.add((sid, condition_id))
            receipt["attempted_video_count"] += 1
            receipt["status"] = "RUNNING"
            write_json(receipt_path, receipt, replace=True)
            print("RENDER", sid, condition_id, attempt["status"], returncode, flush=True)
    failures = [row for row in receipt["attempts"] if row.get("status") != "RENDERED"]
    if len(receipt["attempts"]) == planned and not failures:
        receipt["status"] = "COMPLETE"
    elif speaker_limit is not None and len(receipt["attempts"]) == speaker_limit * len(conditions) and not failures:
        receipt["status"] = "SMOKE_COMPLETE"
    else:
        receipt["status"] = "PARTIAL"
    receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["failure_count"] = len(failures)
    write_json(receipt_path, receipt, replace=True)
    print(json.dumps({"status": receipt["status"], "planned": planned, "attempted": len(receipt["attempts"]), "failed": len(failures), "receipt": str(receipt_path)}, ensure_ascii=False), flush=True)


def condition_row(speaker_row: dict[str, Any], condition_id: str) -> dict[str, Any]:
    matches = [item for item in speaker_row["conditions"] if item["condition_id"] == condition_id]
    if len(matches) != 1:
        raise PilotError(f"condition mapping is missing or duplicated: {speaker_row['speaker_id']}/{condition_id}")
    return matches[0]


def empty_condition_result(speaker_row: dict[str, Any], condition_id: str) -> dict[str, Any]:
    source_condition = condition_row(speaker_row, condition_id)
    variant = speaker_row["audio_variants"][source_condition["audio_variant"]]
    return {
        **source_condition,
        "status": "NOT_ATTEMPTED",
        "missing_reason": "planned row has not been processed",
        "audio_rms_dbfs": variant.get("audio_rms_dbfs", variant.get("stats", {}).get("rms_dbfs")),
        "relative_dose_db": source_condition.get("relative_dose_db"),
        "audio_variant_sha256": variant["sha256"],
        "video_sha256": None,
        "feature_sha256": None,
        "visual_sha256": None,
        "distance_path": None,
        "distance_sha256": None,
        "distance_shape": None,
        "curve31": None,
        "lags": list(range(-LAG_RADIUS, LAG_RADIUS + 1)),
        "q_support": [],
        "C": None,
        "B": None,
        "Dmin": None,
        "D0": None,
        "anchor": None,
        "curve31_full": None,
        "q_support_full": [],
        "C_full": None,
        "B_full": None,
        "Dmin_full": None,
        "D0_full": None,
        "anchor_full": None,
        "motion": {"status": "NOT_MEASURED_UNQUALIFIED"},
        "encoder": {"status": "NOT_MEASURED"},
    }


def empty_speaker_result(source_row: dict[str, Any], audio_row: dict[str, Any], conditions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "speaker_id": source_row["speaker"],
        "source_sha256": source_row["audio_sha256"],
        "source_text_sha256": source_row.get("source_text_sha256"),
        "processing_status": "NOT_STARTED",
        "support": {"status": "NOT_SCORED", "primary_guard20_q_indices": [], "full_q_indices": [], "lag_values": list(range(-LAG_RADIUS, LAG_RADIUS + 1))},
        "conditions": [empty_condition_result(audio_row, item["condition_id"]) for item in conditions],
    }


def shared_support(audio_count: int, visual_counts: dict[str, int]) -> dict[str, Any]:
    lag_values = list(range(-LAG_RADIUS, LAG_RADIUS + 1))
    if len(visual_counts) != 8 or audio_count < 1 or any(count < 1 for count in visual_counts.values()):
        return {"status": "INCOMPLETE_EIGHT_CELL_SET", "primary_guard20_q_indices": [], "primary_guard20_query_count": 0, "full_q_indices": [], "full_query_count": 0, "lag_values": lag_values, "audio_window_count": int(audio_count), "visual_window_counts": visual_counts, "no_padding_verified": False, "shared_by_all_eight_conditions": False}
    limit = min([audio_count, *visual_counts.values()])
    primary = list(range(20, limit - 20))
    full_limit = min(min(visual_counts.values()), audio_count - 15)
    full = list(range(15, full_limit))
    valid = bool(primary and full) and all(
        0 <= q + lag < audio_count and all(q < count for count in visual_counts.values())
        for q in primary + full
        for lag in lag_values
    )
    status = "PASS" if valid else "INSUFFICIENT_OR_INVALID_SUPPORT"
    return {
        "status": status,
        "query_limit_L": int(limit),
        "primary_guard20_q_indices": primary if valid else [],
        "primary_guard20_query_count": len(primary) if valid else 0,
        "full_q_indices": full if valid else [],
        "full_query_count": len(full) if valid else 0,
        "lag_values": lag_values,
        "audio_window_count": int(audio_count),
        "visual_window_counts": {key: int(value) for key, value in visual_counts.items()},
        "no_padding_verified": bool(valid),
        "shared_by_all_eight_conditions": True,
    }


def encode_audio(packet_path: Path, source_manifest_path: Path, output_root: Path, *, device: str = "cuda") -> None:
    packet, packet_sha, source_manifest = load_packet(packet_path, source_manifest_path)
    expected_pins(packet)
    audio_manifest = json.loads((output_root / "audio_variant_manifest.json").read_text(encoding="utf-8"))
    if audio_manifest.get("protocol_sha256") != packet_sha:
        raise PilotError("audio variants are not bound to the current reviewed packet")
    output_path = output_root / "encoder_manifest.json"
    if output_path.exists():
        raise FileExistsError("encoder manifest already exists; refusing duplicate encoder diagnostics")
    from scripts.experiments.portable_source_axis_20260930.wav2lip import encode_latents, load_model, mel_chunks

    import torch

    torch.set_num_threads(2)
    model, model_metadata = load_model(WAV2LIP, WAV2LIP_CHECKPOINT, device=device)
    array_root = output_root / "encoder"
    array_root.mkdir(exist_ok=False)
    speakers: list[dict[str, Any]] = []
    try:
        for speaker_row in audio_manifest["speakers"]:
            sid = speaker_row["speaker_id"]
            variants = speaker_row["audio_variants"]
            latent_records: dict[str, dict[str, Any]] = {}
            sid_arrays = array_root / sid
            sid_arrays.mkdir()
            for variant_id, variant in variants.items():
                wav_path = Path(variant["path"])
                if sha256_file(wav_path) != variant["sha256"]:
                    raise PilotError(f"audio variant changed before Wav2Lip encoder: {sid}/{variant_id}")
                if variant_id == "RAW":
                    pcm, _ = read_pcm16(wav_path)
                    waveform = pcm.astype(np.float32) / np.float32(32768.0)
                else:
                    waveform, _ = read_float_wav(wav_path)
                mel, chunks, starts, frontend = mel_chunks(waveform, WAV2LIP)
                latents = encode_latents(model, chunks, device=device, batch_size=32)
                if len(starts) != latents.shape[0]:
                    raise PilotError(f"Wav2Lip mel and encoder windows disagree for {sid}/{variant_id}")
                mel_path = sid_arrays / f"{variant_id}_mel.npy"
                chunks_path = sid_arrays / f"{variant_id}_mel_chunks.npy"
                latent_path = sid_arrays / f"{variant_id}_encoder.npy"
                for path, array in ((mel_path, mel), (chunks_path, chunks), (latent_path, latents)):
                    with path.open("xb") as stream:
                        np.save(stream, np.asarray(array), allow_pickle=False)
                latent_records[variant_id] = {
                    "status": "ENCODED",
                    "audio_path": variant["path"],
                    "audio_sha256": variant["sha256"],
                    "mel_path": str(mel_path.resolve()),
                    "mel_sha256": sha256_file(mel_path),
                    "mel_shape": list(mel.shape),
                    "mel_chunks_path": str(chunks_path.resolve()),
                    "mel_chunks_sha256": sha256_file(chunks_path),
                    "mel_chunks_shape": list(chunks.shape),
                    "mel_starts": starts,
                    "encoder_path": str(latent_path.resolve()),
                    "encoder_sha256": sha256_file(latent_path),
                    "encoder_shape": list(latents.shape),
                    "encoder_array_sha256": hashlib.sha256(np.ascontiguousarray(latents).tobytes()).hexdigest(),
                    "frontend": frontend,
                    "forward_performed": True,
                }
            reference = np.load(latent_records["RAW"]["encoder_path"], allow_pickle=False)
            for variant_id, record in latent_records.items():
                current = np.load(record["encoder_path"], allow_pickle=False)
                if current.shape != reference.shape:
                    record["relative_summary"] = {"status": "WINDOW_SHAPE_MISMATCH"}
                    continue
                ref_norm = np.linalg.norm(reference.astype(np.float64), axis=1)
                cur_norm = np.linalg.norm(current.astype(np.float64), axis=1)
                denom = np.maximum(ref_norm * cur_norm, 1e-30)
                cosine = np.sum(reference.astype(np.float64) * current.astype(np.float64), axis=1) / denom
                angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                norm_ratio = cur_norm / np.maximum(ref_norm, 1e-30)
                record["relative_summary"] = {
                    "status": "MEASURED",
                    "reference_variant": "RAW",
                    "window_count": int(current.shape[0]),
                    "mean_cosine": float(np.mean(cosine)),
                    "median_cosine": float(np.median(cosine)),
                    "mean_angle_degrees": float(np.mean(angle)),
                    "mean_norm_ratio": float(np.mean(norm_ratio)),
                    "median_norm_ratio": float(np.median(norm_ratio)),
                    "window_cosine_sha256": hashlib.sha256(np.ascontiguousarray(cosine, dtype="<f8").tobytes()).hexdigest(),
                }
            speakers.append({"speaker_id": sid, "conditions": latent_records})
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    result = {
        "schema_version": 1,
        "status": "COMPLETE" if len(speakers) == len(audio_manifest["speakers"]) else "PARTIAL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": packet_sha,
        "source_manifest_sha256": packet["source_manifest_sha256"],
        "parent_pid": os.getpid(),
        "parent_start_ticks": process_start_ticks(os.getpid()),
        "wav2lip_model": model_metadata,
        "code_bindings": {
            "portable_wav2lip_helper": sha256_file(REPO / "scripts/experiments/portable_source_axis_20260930/wav2lip.py"),
            "portable_ops": sha256_file(REPO / "scripts/experiments/portable_source_axis_20260930/ops.py"),
            "wav2lip_audio": sha256_file(WAV2LIP / "audio.py"),
            "wav2lip_model_definition": sha256_file(WAV2LIP / "models/wav2lip.py"),
        },
        "speakers": speakers,
    }
    write_json(output_path, result)
    print(json.dumps({"status": result["status"], "manifest": str(output_path), "speakers": len(speakers)}, ensure_ascii=False), flush=True)


def run_logged(command: list[str], log_path: Path, *, cwd: Path = REPO) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise FileExistsError(f"refusing to overwrite log: {log_path}")
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("xb") as stream:
        child = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        ticks = process_start_ticks(child.pid)
        returncode = child.wait()
    return {"command": command, "started_utc": started, "pid": child.pid, "start_ticks": ticks, "returncode": returncode, "log_path": str(log_path.resolve()), "log_sha256": sha256_file(log_path)}


def probe_video(path: Path, output_dir: Path) -> dict[str, Any]:
    command = [str(FFPROBE), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,start_time,duration,nb_frames", "-of", "json", str(path)]
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "ffprobe.json"
    completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
    log_path.write_text(completed.stdout + "\nSTDERR\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise PilotError(f"ffprobe failed for rendered video: {path}")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise PilotError(f"video does not have exactly one selected video stream: {path}")
    stream = streams[0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if rate not in {"25/1", "25/1.0"}:
        raise PilotError(f"rendered video is not 25 fps: {path}: {rate}")
    return {"path": str(path.resolve()), "command": command, "returncode": completed.returncode, "metadata": stream, "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(), "stderr": completed.stderr}


def extract_visual_features(video_path: Path, out_dir: Path, engine: Any) -> tuple[np.ndarray, dict[str, Any]]:
    import cv2
    from scripts.experiments.wav2lip_roi_peak_recheck.worker import _crop_visual_frame

    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite SyncNet frame/features directory: {out_dir}")
    frame_dir = out_dir / "jpeg_frames"
    frame_dir.mkdir(parents=True, exist_ok=False)
    pattern = frame_dir / "%06d.jpg"
    command = [str(FFMPEG), "-nostdin", "-y", "-loglevel", "error", "-i", str(video_path), "-threads", "1", "-f", "image2", str(pattern)]
    log_path = out_dir / "ffmpeg_frames.log"
    process_meta = run_logged(command, log_path)
    if process_meta["returncode"] != 0:
        raise PilotError(f"ffmpeg frame extraction failed: {video_path}")
    paths = sorted(frame_dir.glob("*.jpg"))
    if not paths or [path.name for path in paths] != [f"{index:06d}.jpg" for index in range(1, len(paths) + 1)]:
        raise PilotError(f"decoded JPEG frames are empty or noncontiguous: {video_path}")
    x1, y1, x2, y2 = SCORE_BOX_XYXY
    box_worker = (y1, y2, x1, x2)
    frames: list[np.ndarray] = []
    decoded_shape: tuple[int, ...] | None = None
    for frame_path in paths:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None or image.dtype != np.uint8:
            raise PilotError(f"cv2 cannot decode frame: {frame_path}")
        if decoded_shape is None:
            decoded_shape = image.shape
        elif image.shape != decoded_shape:
            raise PilotError(f"decoded frame dimensions changed in {video_path}")
        frames.append(_crop_visual_frame(image, box_worker))
    outputs: list[np.ndarray] = []
    with engine._torch.inference_mode():
        for start in range(0, len(frames) - 4, engine.batch_size):
            stop = min(len(frames) - 4, start + engine.batch_size)
            windows = [frames[index : index + 5] for index in range(start, stop)]
            batch = engine._torch.from_numpy(engine._visual_batch(windows)).to(engine.device)
            outputs.append(engine.network.forward_lip(batch).detach().cpu().numpy().astype(np.float32))
    if not outputs:
        raise PilotError(f"video has fewer than five frames: {video_path}")
    visual = np.concatenate(outputs, axis=0)
    if visual.ndim != 2 or visual.shape[1] != 1024 or not np.isfinite(visual).all():
        raise PilotError(f"SyncNet visual feature shape is invalid: {visual.shape}")
    visual_path = out_dir / "visual.npy"
    with visual_path.open("xb") as stream:
        np.save(stream, visual, allow_pickle=False)
    metadata = {
        "status": "EXTRACTED",
        "video_sha256": sha256_file(video_path),
        "visual_path": str(visual_path.resolve()),
        "visual_sha256": sha256_file(visual_path),
        "visual_array_sha256": hashlib.sha256(np.ascontiguousarray(visual).tobytes()).hexdigest(),
        "visual_shape": list(visual.shape),
        "frame_count": len(frames),
        "decoded_frame_shape": list(decoded_shape or ()),
        "crop_box_xyxy": list(SCORE_BOX_XYXY),
        "crop_box_worker_order": list(box_worker),
        "crop_size": [224, 224],
        "resize": "cv2.INTER_AREA",
        "frame_jpeg_sha256": [sha256_file(path) for path in paths],
        "frame_extract_process": process_meta,
        "frame_command": command,
    }
    return visual, metadata


def curve_metrics(curve: np.ndarray, anchor_index: int) -> dict[str, Any]:
    value = np.asarray(curve, dtype=np.float64)
    if value.shape != (31,) or not np.isfinite(value).all():
        raise PilotError("distance curve must contain 31 finite values")
    best = int(np.argmin(value))
    median = float(np.median(value))
    minimum = float(np.min(value))
    zero = float(value[LAG_RADIUS])
    anchor = float(value[anchor_index])
    return {
        "B": median,
        "Dmin": minimum,
        "D0": zero,
        "C": median - minimum,
        "bestlag": best - LAG_RADIUS,
        "argmin_index": best,
        "offset_at_lag_boundary": best in (0, 30),
        "search_gain": zero - minimum,
        "anchor_D": anchor,
        "anchor_C": median - anchor,
        "anchor_lag_index": int(anchor_index),
        "anchor_lag_frames": int(anchor_index - LAG_RADIUS),
    }


def score(packet_path: Path, source_manifest_path: Path, output_root: Path, *, device: str = "cuda", speaker_limit: int | None = None, continue_existing: bool = False) -> None:
    packet, packet_sha, source_manifest = load_packet(packet_path, source_manifest_path)
    expected_pins(packet)
    if speaker_limit is not None and (continue_existing or speaker_limit < 1):
        raise PilotError("a score smoke selects a positive speaker limit and cannot be a continuation")
    audio_manifest = json.loads((output_root / "audio_variant_manifest.json").read_text(encoding="utf-8"))
    render_receipt = json.loads((output_root / "render_receipt.json").read_text(encoding="utf-8"))
    if audio_manifest.get("protocol_sha256") != packet_sha or render_receipt.get("protocol_sha256") != packet_sha:
        raise PilotError("audio/render inputs are not bound to this reviewed packet")
    expected_render_status = "SMOKE_COMPLETE" if speaker_limit is not None else "COMPLETE"
    if render_receipt.get("status") != expected_render_status and not (continue_existing and render_receipt.get("status") == "COMPLETE"):
        raise PilotError(f"render stage is not ready for this scoring stage: {render_receipt.get('status')}")
    if sha256_file(SYNCNET_MODEL) != SYNCNET_MODEL_SHA256:
        raise PilotError("SyncNet checkpoint hash mismatch")
    import torch
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    torch.set_num_threads(2)
    engine = SyncNetEngine(model_path=SYNCNET_MODEL, batch_size=32, device=device)
    render_by = {(row.get("speaker_id"), row.get("condition_id")): row for row in render_receipt["attempts"]}
    encoder_manifest_path = output_root / "encoder_manifest.json"
    encoder_manifest = json.loads(encoder_manifest_path.read_text(encoding="utf-8")) if encoder_manifest_path.is_file() else None
    encoder_by = {row["speaker_id"]: row["conditions"] for row in encoder_manifest["speakers"]} if encoder_manifest else {}
    results_path = output_root / "score_results.json"
    if results_path.exists():
        if not continue_existing or speaker_limit is not None:
            raise FileExistsError("score results already exist; use --continue to score only unprocessed speakers")
        result = json.loads(results_path.read_text(encoding="utf-8"))
        if result.get("protocol_sha256") != packet_sha or result.get("source_manifest_sha256") != packet["source_manifest_sha256"]:
            raise PilotError("existing score results are not bound to the reviewed plan")
        expected_speakers = [row["speaker_id"] for row in audio_manifest["speakers"]]
        if result.get("planned_speakers") != expected_speakers or len(result.get("speakers", [])) != len(expected_speakers):
            raise PilotError("existing score results do not preserve the frozen speaker denominator")
    else:
        if continue_existing:
            raise FileNotFoundError("--continue requested without existing score results")
        result = {
        "schema_version": 1,
        "status": "RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": packet_sha,
        "source_manifest_sha256": packet["source_manifest_sha256"],
        "syncnet_model_sha256": SYNCNET_MODEL_SHA256,
        "syncnet_code_hash": engine.code_hash,
        "engine_metadata": {"engine": "SyncNetEngine", "model_sha256": SYNCNET_MODEL_SHA256, "code_hash": engine.code_hash, "device": str(device), "batch_size": 32},
        "parent_pid": os.getpid(),
        "parent_start_ticks": process_start_ticks(os.getpid()),
        "planned_speakers": [row["speaker_id"] for row in audio_manifest["speakers"]],
        "planned_conditions": audio_manifest["condition_plan"],
        "planned_cell_count": len(audio_manifest["speakers"]) * len(audio_manifest["condition_plan"]),
        "scoring_contract": {"geometry": "raw SyncNet embeddings", "primary_support": "common q=[20,L-20); L=min(audio window count, each of 8 video window counts); all 31 lags real, no padding", "full_support": "common q=[15,min(min video window count,audio window count-15)); all 31 lags real, no padding", "lag_frames": list(range(-15, 16)), "visual_route": "ffmpeg JPEG frames -> cv2 BGR -> frozen score ROI -> _crop_visual_frame INTER_AREA 224x224 -> SyncNetEngine._visual_batch/forward_lip", "audio_route": "SyncNetEngine.extract_audio on one unchanged natural PCM16 source per speaker, reused for all eight videos"},
        "speakers": [empty_speaker_result(source, audio, audio_manifest["condition_plan"]) for source, audio in zip(source_manifest["cohort"], audio_manifest["speakers"], strict=True)],
        }
        write_json(results_path, result)
    result["parent_pid"] = os.getpid()
    result["parent_start_ticks"] = process_start_ticks(os.getpid())
    speaker_rows = list(zip(source_manifest["cohort"], audio_manifest["speakers"], strict=True))
    if speaker_limit is not None and not 1 <= speaker_limit <= len(speaker_rows):
        raise PilotError("speaker limit must select at least one planned speaker")
    if speaker_limit is not None:
        speaker_rows = speaker_rows[:speaker_limit]
    result_by_sid = {row["speaker_id"]: row for row in result["speakers"]}
    for row in result["speakers"]:
        if row.get("processing_status") == "RUNNING":
            row["processing_status"] = "FAILED"
            row["error"] = "previous scoring process ended while this speaker was marked RUNNING; this speaker is not retried"
            for condition_row_value in row["conditions"]:
                if condition_row_value.get("status") == "NOT_ATTEMPTED":
                    condition_row_value.update({"status": "SCORE_INTERRUPTED", "missing_reason": row["error"]})
    completed_sids = {sid for sid, row in result_by_sid.items() if row.get("processing_status") in {"COMPLETE", "FAILED"}}
    if continue_existing:
        write_json(results_path, result, replace=True)
    current_sid: str | None = None
    try:
        for source_row, audio_row in speaker_rows:
            sid = source_row["speaker"]
            if sid != audio_row["speaker_id"]:
                raise PilotError("source and audio manifest cohort order differs")
            if sid in completed_sids:
                continue
            current_sid = sid
            speaker_record = result_by_sid[sid]
            speaker_record["processing_status"] = "RUNNING"
            speaker_record.pop("error", None)
            write_json(results_path, result, replace=True)
            source_audio = Path(audio_row["source_audio_path"])
            if sha256_file(source_audio) != source_row["audio_sha256"]:
                raise PilotError(f"natural scoring PCM changed for {sid}")
            audio_features, audio_meta = engine.extract_audio(source_audio)
            audio_feature_path = output_root / "audio_features" / sid / "natural.npy"
            audio_feature_path.parent.mkdir(parents=True, exist_ok=True)
            with audio_feature_path.open("xb") as stream:
                np.save(stream, audio_features, allow_pickle=False)
            audio_feature_hash = sha256_file(audio_feature_path)
            visual_by: dict[str, np.ndarray] = {}
            visual_meta: dict[str, dict[str, Any]] = {}
            preliminary: list[dict[str, Any]] = []
            for condition in audio_manifest["condition_plan"]:
                condition_id = condition["condition_id"]
                render_record = render_by.get((sid, condition_id))
                speaker_condition = condition_row(audio_row, condition_id)
                row = dict(speaker_condition)
                variant = audio_row["audio_variants"][condition["audio_variant"]]
                row.update({
                    "status": "UNSCORED",
                    "missing_reason": None,
                    "video_path": None,
                    "video_sha256": None,
                    "feature_path": None,
                    "feature_sha256": None,
                    "audio_path": variant["path"],
                    "audio_variant_sha256": variant["sha256"],
                    "audio_feature_path": str(audio_feature_path.resolve()),
                    "audio_feature_sha256": audio_feature_hash,
                    "audio_rms_dbfs": variant.get("audio_rms_dbfs", variant.get("stats", {}).get("rms_dbfs")),
                    "relative_dose_db": speaker_condition.get("relative_dose_db"),
                    "audio_variant_sha256": variant["sha256"],
                    "engine_metadata": {"engine": "SyncNetEngine", "model_sha256": SYNCNET_MODEL_SHA256, "code_hash": engine.code_hash, "device": str(device), "batch_size": 32, "audio_extraction": audio_meta},
                    "curve31": None,
                    "lags": list(range(-LAG_RADIUS, LAG_RADIUS + 1)),
                    "lag_frame_convention": "column j compares visual feature q with audio feature q+(j-15); lag 0 is column 15",
                    "q_support": [],
                    "distance_path": None,
                    "distance_sha256": None,
                    "distance_shape": None,
                    "C": None,
                    "B": None,
                    "Dmin": None,
                    "D0": None,
                    "bestlag": None,
                    "anchor": None,
                    "curve31_full": None,
                    "q_support_full": [],
                    "C_full": None,
                    "B_full": None,
                    "Dmin_full": None,
                    "D0_full": None,
                    "anchor_full": None,
                    "motion": {"status": "NOT_MEASURED_UNQUALIFIED"},
                    "encoder": encoder_by.get(sid, {}).get(condition["audio_variant"], {}).get("relative_summary", {"status": "NOT_MEASURED"}),
                    "provenance": {"source_sha256": source_row["audio_sha256"], "audio_variant_sha256": variant["sha256"], "audio_feature_sha256": audio_feature_hash},
                })
                if render_record is None:
                    row.update({"status": "RENDER_ATTEMPT_MISSING", "missing_reason": "no render attempt row"})
                    preliminary.append(row)
                    continue
                if render_record.get("status") != "RENDERED" or not render_record.get("video_sha256"):
                    row.update({"status": "RENDER_FAILED", "missing_reason": render_record.get("status", "failed or missing output"), "render_returncode": render_record.get("returncode")})
                    preliminary.append(row)
                    continue
                video = Path(render_record["video_path"])
                if not video.is_file() or sha256_file(video) != render_record["video_sha256"]:
                    row.update({"status": "VIDEO_CHANGED", "missing_reason": "rendered video missing or hash mismatch"})
                    preliminary.append(row)
                    continue
                probe_dir = output_root / "ffprobe" / sid / condition_id
                extract_dir = output_root / "syncnet_visual" / sid / condition_id
                try:
                    probe = probe_video(video, probe_dir)
                    visual, metadata = extract_visual_features(video, extract_dir, engine)
                    metadata["probe"] = probe
                    visual_by[condition_id] = visual
                    visual_meta[condition_id] = metadata
                    row.update({"status": "FEATURED", "video_path": str(video.resolve()), "video_sha256": render_record["video_sha256"], "feature_path": metadata["visual_path"], "feature_sha256": metadata["visual_sha256"], "visual_sha256": metadata["visual_sha256"], "visual_window_count": int(visual.shape[0]), "visual_metadata": metadata})
                except Exception as exc:
                    row.update({"status": "VISUAL_FEATURE_FAILED", "missing_reason": f"{type(exc).__name__}: {exc}", "video_path": str(video.resolve()), "video_sha256": render_record["video_sha256"]})
                preliminary.append(row)
            all_ready = len(visual_by) == len(audio_manifest["condition_plan"])
            support: dict[str, Any]
            if all_ready:
                visual_counts = {key: int(value.shape[0]) for key, value in visual_by.items()}
                support = shared_support(int(audio_features.shape[0]), visual_counts)
                primary_q = support["primary_guard20_q_indices"]
                full_q = support["full_q_indices"]
                if support["status"] == "PASS":
                    matrices: dict[str, np.ndarray] = {}
                    curves: dict[str, np.ndarray] = {}
                    for row in preliminary:
                        cid = row["condition_id"]
                        matrix = engine.distance_matrix(visual_by[cid], audio_features)
                        if matrix.shape[0] <= (primary_q[-1] if primary_q else 0) or matrix.shape[1] != 31:
                            raise PilotError(f"SyncNet distance matrix cannot cover frozen q support: {sid}/{cid}")
                        # The row-wise q mapping is the official SyncNet mapping.  Restrict it to real 31-lag context.
                        primary = np.ascontiguousarray(matrix[primary_q], dtype=np.float64)
                        full = np.ascontiguousarray(matrix[full_q], dtype=np.float64)
                        if primary.shape != (len(primary_q), 31) or full.shape != (len(full_q), 31):
                            raise PilotError(f"distance crop shape differs from frozen q support: {sid}/{cid}")
                        matrices[cid] = primary
                        curves[cid] = primary.mean(axis=0, dtype=np.float64)
                        primary_path = output_root / "distances" / sid / f"{cid}_guard20.npy"
                        full_path = output_root / "distances" / sid / f"{cid}_full.npy"
                        primary_path.parent.mkdir(parents=True, exist_ok=True)
                        with primary_path.open("xb") as stream:
                            np.save(stream, primary, allow_pickle=False)
                        with full_path.open("xb") as stream:
                            np.save(stream, full, allow_pickle=False)
                        row.update({
                            "status": "SCORED",
                            "q_support": primary_q,
                            "distance_path": str(primary_path.resolve()),
                            "distance_sha256": sha256_file(primary_path),
                            "distance_shape": list(primary.shape),
                            "curve31": curves[cid].tolist(),
                            "distance_path_full": str(full_path.resolve()),
                            "distance_sha256_full": sha256_file(full_path),
                            "distance_shape_full": list(full.shape),
                            "q_support_full": full_q,
                        })
                        metrics = curve_metrics(curves[cid], 15)
                        row.update({key: metrics[key] for key in ("C", "B", "Dmin", "D0", "bestlag", "argmin_index", "offset_at_lag_boundary")})
                        row.update({"search_gain": metrics["search_gain"], "anchor_lag_frames": None})
                        full_curve = full.mean(axis=0, dtype=np.float64)
                        full_metrics = curve_metrics(full_curve, 15)
                        row.update({"curve31_full": full_curve.tolist(), "C_full": full_metrics["C"], "B_full": full_metrics["B"], "Dmin_full": full_metrics["Dmin"], "D0_full": full_metrics["D0"], "bestlag_full": full_metrics["bestlag"], "search_gain_full": full_metrics["search_gain"]})
                    raw_curve = curves["RAW_R1"]
                    anchor_index = int(np.argmin(raw_curve))
                    for row in preliminary:
                        primary_metrics = curve_metrics(np.asarray(row["curve31"], dtype=np.float64), anchor_index)
                        full_metrics = curve_metrics(np.asarray(row["curve31_full"], dtype=np.float64), anchor_index)
                        row["anchor"] = {"D": primary_metrics["anchor_D"], "C": primary_metrics["anchor_C"], "bestlag": primary_metrics["bestlag"], "search_gain": primary_metrics["search_gain"], "fixed_anchor_lag_index": anchor_index, "fixed_anchor_lag_frames": anchor_index - LAG_RADIUS, "reference_condition": "RAW_R1"}
                        row["anchor_lag_frames"] = anchor_index - LAG_RADIUS
                        row["anchor_full"] = {"D": full_metrics["anchor_D"], "C": full_metrics["anchor_C"], "bestlag": full_metrics["bestlag"], "search_gain": full_metrics["search_gain"], "fixed_anchor_lag_index": anchor_index, "fixed_anchor_lag_frames": anchor_index - LAG_RADIUS, "reference_condition": "RAW_R1"}
                else:
                    for row in preliminary:
                        row.update({"status": "COMMON_SUPPORT_INVALID", "missing_reason": "primary/full shared real-lag q support is empty or contains padded contexts"})
            else:
                all_lengths: dict[str, int | None] = {item["condition_id"]: int(visual_by[item["condition_id"]].shape[0]) if item["condition_id"] in visual_by else None for item in audio_manifest["condition_plan"]}
                support = {"status": "INCOMPLETE_EIGHT_CELL_SET", "primary_guard20_q_indices": [], "primary_guard20_query_count": 0, "full_q_indices": [], "full_query_count": 0, "lag_values": list(range(-LAG_RADIUS, LAG_RADIUS + 1)), "audio_window_count": int(audio_features.shape[0]), "visual_window_counts": all_lengths, "no_padding_verified": False, "shared_by_all_eight_conditions": False}
                for row in preliminary:
                    if row["status"] == "FEATURED":
                        row.update({"status": "NO_COMMON_EIGHT_CELL_SUPPORT", "missing_reason": "at least one planned condition lacks visual features"})
            speaker_record.update({
                "audio_feature_path": str(audio_feature_path.resolve()),
                "audio_feature_sha256": audio_feature_hash,
                "audio_feature_shape": list(audio_features.shape),
                "audio_feature_metadata": audio_meta,
                "support": support,
                "conditions": preliminary,
                "processing_status": "COMPLETE",
            })
            write_json(results_path, result, replace=True)
            print("SCORED_SPEAKER", sid, support["status"], support.get("primary_guard20_query_count", 0), flush=True)
            current_sid = None
    except Exception as exc:
        if current_sid is not None:
            failed = result_by_sid[current_sid]
            failed["processing_status"] = "FAILED"
            failed["error"] = f"{type(exc).__name__}: {exc}"
            failed["support"] = {"status": "SCORE_FAILED", "primary_guard20_q_indices": [], "full_q_indices": [], "lag_values": list(range(-LAG_RADIUS, LAG_RADIUS + 1))}
            for condition_row_value in failed["conditions"]:
                if condition_row_value.get("status") in {"NOT_ATTEMPTED", "FEATURED", "UNSCORED"}:
                    condition_row_value.update({"status": "SCORE_FAILED", "missing_reason": failed["error"]})
        result["status"] = "PARTIAL"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(results_path, result, replace=True)
        rows = [condition for speaker in result["speakers"] for condition in speaker["conditions"]]
        partial_success = sum(row.get("status") == "SCORED" for row in rows)
        partial_receipt = {"schema_version": 1, "status": "PARTIAL", "protocol_sha256": packet_sha, "score_results_path": str(results_path.resolve()), "score_results_sha256": sha256_file(results_path), "planned_cell_count": result["planned_cell_count"], "scored_cell_count": partial_success, "failed_speaker_id": current_sid, "parent_pid": os.getpid(), "parent_start_ticks": process_start_ticks(os.getpid()), "syncnet_model_sha256": SYNCNET_MODEL_SHA256}
        receipt_path = output_root / "score_receipt.json"
        write_json(receipt_path, partial_receipt, replace=receipt_path.exists())
        raise
    finally:
        engine.close()
    rows = [condition for speaker in result["speakers"] for condition in speaker["conditions"]]
    success = sum(row.get("status") == "SCORED" for row in rows)
    speakers_complete = all(row.get("processing_status") == "COMPLETE" for row in result["speakers"])
    result["status"] = "COMPLETE" if speakers_complete and success == result["planned_cell_count"] else "PARTIAL"
    result["scored_cell_count"] = success
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(results_path, result, replace=True)
    receipt = {"schema_version": 1, "status": result["status"], "protocol_sha256": packet_sha, "score_results_path": str(results_path.resolve()), "score_results_sha256": sha256_file(results_path), "planned_cell_count": result["planned_cell_count"], "scored_cell_count": success, "parent_pid": os.getpid(), "parent_start_ticks": process_start_ticks(os.getpid()), "syncnet_model_sha256": SYNCNET_MODEL_SHA256}
    receipt_path = output_root / "score_receipt.json"
    write_json(receipt_path, receipt, replace=receipt_path.exists())
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen P1 scalar-gain audio prep, Wav2Lip generation, encoder diagnostic and official SyncNet scoring.")
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("prepare-audio", "render", "encode", "score"):
        item = sub.add_parser(name)
        item.add_argument("--packet", type=Path, required=True)
        item.add_argument("--source-manifest", type=Path, required=True)
        item.add_argument("--output-root", type=Path, required=True)
        if name == "prepare-audio":
            item.add_argument("--source-audio-root", type=Path, required=True)
        if name in {"encode", "score"}:
            item.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
        if name in {"render", "score"}:
            item.add_argument("--speaker-limit", type=int)
            item.add_argument("--continue", dest="continue_existing", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare-audio":
        prepare_audio(args.packet, args.source_manifest, args.source_audio_root, args.output_root)
    elif args.stage == "render":
        render(args.packet, args.source_manifest, args.output_root, speaker_limit=args.speaker_limit, continue_existing=args.continue_existing)
    elif args.stage == "encode":
        encode_audio(args.packet, args.source_manifest, args.output_root, device=args.device)
    elif args.stage == "score":
        score(args.packet, args.source_manifest, args.output_root, device=args.device, speaker_limit=args.speaker_limit, continue_existing=args.continue_existing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
