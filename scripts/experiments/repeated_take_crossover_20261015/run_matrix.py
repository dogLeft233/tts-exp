"""Render and score a frozen repeated-take 4x4 Wav2Lip/SyncNet matrix.

Run this on the qualified remote host. The input JSON has this shape:
``{"fixed_face":{"path":...,"sha256":...,"score_box_xyxy":[x1,y1,x2,y2]},
"speakers":[{"speaker":"S0003","takes":[{"role":"N1","path":...,"sha256":...}, ...]}]}``.
An N1/N2-only input config can be used with ``prepare-natural`` while TTS is
pending; a later full ``prepare`` adds T1/T2 and remaining speakers safely.
The primary scores use each selected source WAV at its native timeline after
mono/16 kHz/RMS canonicalization. No trimming, padding, warping, or SyncNet
best-lag alignment is performed. All 16 cells within a speaker use one common
query interval that has real audio support at every lag -15..+15.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
RUN_ROOT_PARTS = ("runs", "repeated_take_crossover_20261015_attempt01")
WAV2LIP = REPO / "third_party/Wav2Lip"
WAV2LIP_PY = Path("/home/wjj/.venvs/wav2lip/bin/python")
SYNCNET_MODEL = REPO / "third_party/syncnet_python/data/syncnet_v2.model"
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
FFPROBE = Path("/home/wjj/miniconda3/bin/ffprobe")
WAV2LIP_CHECKPOINT_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
WAV2LIP_INFERENCE_SHA256 = "42cfc8d3060921e6714c8e08243e5508d2e637d29cd0be78760492c5d26380a9"
TARGET_RMS = 0.0376838172675746
SAMPLE_RATE = 16000
FPS = 25
LAG_RADIUS = 15
ROLES = ("N1", "N2", "T1", "T2")
ROLE_CLASS = {"N1": "N", "N2": "N", "T1": "T", "T2": "T"}


class MatrixError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise MatrixError(f"refusing to overwrite existing artifact: {path}")
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _run(command: list[str], log: Path | None = None, *, cwd: Path = REPO) -> None:
    if log is None:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        if log.exists():
            raise MatrixError(f"refusing to overwrite command log: {log}")
        with log.open("x", encoding="utf-8") as handle:
            result = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode:
        raise MatrixError(f"command failed with status {result.returncode}; see {log or command[0]}")


def load_config(path: Path, run_dir: Path, *, allow_natural_only: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    face = cfg.get("fixed_face")
    if not isinstance(face, dict):
        raise MatrixError("protocol lacks fixed_face record")
    face_path = Path(str(face.get("path", ""))).resolve(strict=True)
    if sha256_file(face_path) != face.get("sha256"):
        raise MatrixError("fixed face hash mismatch")
    xyxy = face.get("score_box_xyxy")
    if not isinstance(xyxy, list) or len(xyxy) != 4:
        raise MatrixError("fixed score box must be [x1,y1,x2,y2]")
    x1, y1, x2, y2 = (int(x) for x in xyxy)
    if not (0 <= x1 < x2 and 0 <= y1 < y2):
        raise MatrixError("fixed score box is invalid")
    speakers = cfg.get("speakers")
    if not isinstance(speakers, list) or not speakers:
        raise MatrixError("protocol has no speakers")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for speaker_row in speakers:
        speaker = str(speaker_row.get("speaker", ""))
        if not re.fullmatch(r"S[0-9]{4}", speaker) or speaker in seen:
            raise MatrixError("speaker IDs are invalid or repeated")
        seen.add(speaker)
        items = speaker_row.get("takes")
        by_role: dict[str, dict[str, Any]] = {}
        allowed_counts = {2, 4} if allow_natural_only else {4}
        if not isinstance(items, list) or len(items) not in allowed_counts:
            raise MatrixError(f"{speaker} has an invalid number of audio takes")
        for item in items:
            role = str(item.get("role", ""))
            if role not in ROLES or role in by_role:
                raise MatrixError(f"{speaker} has invalid/duplicate role")
            audio = Path(str(item.get("path", ""))).resolve(strict=True)
            actual_sha = sha256_file(audio)
            if actual_sha != item.get("sha256"):
                raise MatrixError(f"{speaker}/{role} source hash mismatch")
            by_role[role] = {"role": role, "source_path": str(audio), "source_sha256": actual_sha,
                             "source_kind": "natural" if role.startswith("N") else "tts"}
        required_roles = {"N1", "N2"} if allow_natural_only else set(ROLES)
        if not required_roles <= set(by_role) or (not allow_natural_only and set(by_role) != set(ROLES)):
            raise MatrixError(f"{speaker} lacks a required role")
        rows.append({"speaker": speaker, "takes": by_role})
    run_dir_resolved = run_dir.resolve()
    expected_run_dir = (REPO.joinpath(*RUN_ROOT_PARTS)).resolve()
    if run_dir_resolved != expected_run_dir:
        raise MatrixError("run directory is outside the authorized repeated_take_crossover run")
    if face_path == run_dir_resolved or face_path.is_relative_to(run_dir_resolved):
        raise MatrixError("fixed face must be an existing pinned source asset")
    return cfg, rows


def _read_pcm16_wav(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        pcm = handle.readframes(frames)
    if channels != 1 or width != 2 or sample_rate != SAMPLE_RATE or frames < 1:
        raise MatrixError(f"canonical WAV violates mono PCM16/16k contract: {path}")
    values = np.frombuffer(pcm, dtype="<i2").copy()
    if values.size != frames:
        raise MatrixError(f"canonical WAV frame count mismatch: {path}")
    return values, {"sample_rate_hz": sample_rate, "channels": channels,
                    "sample_width_bytes": width, "samples": frames,
                    "duration_s": frames / sample_rate,
                    "decoded_pcm_sha256": hashlib.sha256(pcm).hexdigest()}


def canonicalize_audio(source: Path, destination: Path, work_dir: Path) -> dict[str, Any]:
    if destination.exists():
        raise MatrixError(f"refusing to overwrite canonical audio: {destination}")
    work_dir.mkdir(parents=True, exist_ok=False)
    unscaled = work_dir / "unscaled_mono_16k_float.wav"
    command = [str(FFMPEG), "-nostdin", "-v", "error", "-n", "-i", str(source),
               "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
               "-c:a", "pcm_f32le", str(unscaled)]
    _run(command, work_dir / "ffmpeg.log")
    import soundfile as sf
    values, sample_rate = sf.read(str(unscaled), dtype="float32", always_2d=False)
    values = np.asarray(values, dtype=np.float32)
    if int(sample_rate) != SAMPLE_RATE or values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise MatrixError("resampled source audio is invalid")
    pre_rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
    if not math.isfinite(pre_rms) or pre_rms <= 0:
        raise MatrixError("source audio RMS is zero or non-finite")
    gain = TARGET_RMS / pre_rms
    predicted_peak = float(np.max(np.abs(values.astype(np.float64))) * gain)
    if predicted_peak >= 1.0:
        raise MatrixError("fixed-RMS normalization would clip; no audio artifact accepted")
    normalized = (values * np.float32(gain)).astype(np.float32, copy=False)
    if normalized.size != values.size or not np.isfinite(normalized).all():
        raise MatrixError("normalization altered the sample count or produced non-finite samples")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), normalized, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    pcm, pcm_meta = _read_pcm16_wav(destination)
    observed_rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)) / 32768.0)
    if pcm_meta["samples"] != values.size:
        raise MatrixError("PCM16 writing altered the sample count")
    return {
        "source_path": str(source), "source_sha256": sha256_file(source),
        "unscaled_16k_float_path": str(unscaled), "unscaled_16k_float_sha256": sha256_file(unscaled),
        "canonical_path": str(destination.resolve()), "canonical_sha256": sha256_file(destination),
        "canonical_pcm_sha256": pcm_meta["decoded_pcm_sha256"],
        "sample_rate_hz": SAMPLE_RATE, "channels": 1, "sample_width_bytes": 2,
        "samples": pcm_meta["samples"], "duration_s": pcm_meta["duration_s"],
        "rms_before_gain": pre_rms, "gain": gain, "target_rms": TARGET_RMS,
        "rms_after_pcm16": observed_rms, "predicted_peak_after_gain": predicted_peak,
        "peak_after_pcm16": float(np.max(np.abs(pcm.astype(np.float64) / 32768.0))),
        "trim_pad_warp": False,
        "canonicalization": "ffmpeg decode/downmix/resample to mono 16 kHz float WAV; scalar RMS gain; PCM16 quantization",
        "ffmpeg_command": command,
    }


def prepare(config: dict[str, Any], rows: list[dict[str, Any]], run_dir: Path) -> None:
    manifest_path = run_dir / "audio_manifest.json"
    prior: dict[tuple[str, str], dict[str, Any]] = {}
    prior_config_hashes: list[str] = []
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("target_rms") != TARGET_RMS or old.get("sample_rate_hz") != SAMPLE_RATE:
            raise MatrixError("existing audio manifest uses a different canonicalization contract")
        prior_config_hashes = list(old.get("input_config_sha256s", []))
        for item in old.get("records", []):
            key = (item.get("speaker"), item.get("role"))
            if key in prior:
                raise MatrixError("existing audio manifest has duplicate speaker/role entries")
            prior[key] = item
    config_sha = sha256_file(Path(config["_path"]))
    config_hashes = sorted(set(prior_config_hashes + [config_sha]))
    for row in rows:
        speaker = row["speaker"]
        for role in ROLES:
            if role not in row["takes"]:
                continue
            take = row["takes"][role]
            key = (speaker, role)
            existing = prior.get(key)
            if existing is not None:
                if existing.get("source_sha256") != take["source_sha256"]:
                    raise MatrixError(f"prepared source changed for {speaker}/{role}")
                canonical = Path(existing["canonical_path"])
                if not canonical.is_file() or sha256_file(canonical) != existing.get("canonical_sha256"):
                    raise MatrixError(f"prepared canonical audio changed for {speaker}/{role}")
                continue
            destination = run_dir / "audio" / speaker / f"{role}_fixedrms_16k_pcm16.wav"
            work = run_dir / "audio_work" / speaker / role
            meta = canonicalize_audio(Path(take["source_path"]), destination, work)
            record = {"speaker": speaker, "role": role, **take, **meta}
            prior[key] = record
            print("CANONICAL", speaker, role, meta["samples"], meta["rms_after_pcm16"], flush=True)
            write_json(manifest_path, {
                "schema_version": 1, "status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(),
                "input_config_sha256s": config_hashes, "target_rms": TARGET_RMS,
                "sample_rate_hz": SAMPLE_RATE, "trim_pad_warp": False,
                "records": [prior[k] for k in sorted(prior, key=lambda x: (x[0], ROLES.index(x[1])))],
            }, replace=manifest_path.exists())
    records = [prior[k] for k in sorted(prior, key=lambda x: (x[0], ROLES.index(x[1])))]
    write_json(manifest_path, {
        "schema_version": 1, "status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_config_sha256s": config_hashes, "target_rms": TARGET_RMS,
        "sample_rate_hz": SAMPLE_RATE, "trim_pad_warp": False, "records": records,
    }, replace=manifest_path.exists())


def render(config: dict[str, Any], rows: list[dict[str, Any]], run_dir: Path,
           speakers: set[str]) -> None:
    audio_manifest = json.loads((run_dir / "audio_manifest.json").read_text(encoding="utf-8"))
    by_audio = {(r["speaker"], r["role"]): r for r in audio_manifest["records"]}
    ckpt = WAV2LIP / "checkpoints/wav2lip_gan.pth"
    inference = WAV2LIP / "inference.py"
    if sha256_file(ckpt) != WAV2LIP_CHECKPOINT_SHA256 or sha256_file(inference) != WAV2LIP_INFERENCE_SHA256:
        raise MatrixError("Wav2Lip code/checkpoint pin mismatch")
    face = config["fixed_face"]
    manifest_path = run_dir / "video_manifest.json"
    config_sha = sha256_file(Path(config["_path"]))
    records: list[dict[str, Any]] = []
    prior: dict[tuple[str, str], dict[str, Any]] = {}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (old.get("fixed_face", {}).get("sha256") != face.get("sha256") or
                old.get("wav2lip_checkpoint_sha256") != WAV2LIP_CHECKPOINT_SHA256 or
                old.get("wav2lip_inference_sha256") != WAV2LIP_INFERENCE_SHA256 or
                old.get("config_sha256") != config_sha):
            raise MatrixError("existing video manifest does not match the frozen render contract")
        for item in old.get("records", []):
            key = (item.get("speaker"), item.get("video_role"))
            if key in prior:
                raise MatrixError("existing video manifest has duplicate speaker/role entries")
            video_path = Path(item["video_path"])
            if not video_path.is_file() or sha256_file(video_path) != item.get("video_sha256"):
                raise MatrixError(f"existing rendered video changed for {key[0]}/{key[1]}")
            prior[key] = item
    for row in rows:
        speaker = row["speaker"]
        if speaker not in speakers:
            continue
        for role in ROLES:
            source = by_audio[(speaker, role)]
            key = (speaker, role)
            existing = prior.get(key)
            if existing is not None:
                if existing.get("source_audio_sha256") != source.get("canonical_sha256"):
                    raise MatrixError(f"existing video was rendered from different audio: {speaker}/{role}")
                continue
            video = run_dir / "video" / speaker / f"V_{role}.mp4"
            if video.exists():
                raise MatrixError(f"refusing to overwrite rendered video: {video}")
            video.parent.mkdir(parents=True, exist_ok=True)
            work = run_dir / "wav2lip_work" / speaker / role
            work.mkdir(parents=True, exist_ok=False)
            (work / "temp").mkdir()
            command = [str(WAV2LIP_PY), str(inference), "--checkpoint_path", str(ckpt),
                       "--face", str(Path(face["path"]).resolve()), "--audio", source["canonical_path"],
                       "--outfile", str(video.resolve()), "--face_det_batch_size", "4",
                       "--wav2lip_batch_size", "4", "--nosmooth"]
            log = run_dir / "logs" / f"{speaker}_{role}_wav2lip.log"
            _run(command, log=log, cwd=work)
            if not video.is_file() or video.stat().st_size <= 0:
                raise MatrixError(f"Wav2Lip did not create a nonempty video: {video}")
            item = {"speaker": speaker, "video_role": role, "audio_role": role,
                            "video_path": str(video.resolve()), "video_sha256": sha256_file(video),
                            "video_bytes": video.stat().st_size, "source_audio_path": source["canonical_path"],
                            "source_audio_sha256": source["canonical_sha256"],
                            "face_path": face["path"], "face_sha256": face["sha256"],
                            "command": command, "log_path": str(log.resolve()), "log_sha256": sha256_file(log)}
            prior[key] = item
            records = [prior[k] for k in sorted(prior)]
            write_json(manifest_path, {
                "schema_version": 1, "status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(),
                "config_sha256": config_sha, "fixed_face": face,
                "wav2lip_checkpoint_sha256": WAV2LIP_CHECKPOINT_SHA256,
                "wav2lip_inference_sha256": WAV2LIP_INFERENCE_SHA256, "records": records,
            }, replace=manifest_path.exists())
            print("RENDERED", speaker, role, video.stat().st_size, flush=True)
    records = [prior[k] for k in sorted(prior)]
    write_json(manifest_path, {
        "schema_version": 1, "status": "prepared", "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha, "fixed_face": face,
        "wav2lip_checkpoint_sha256": WAV2LIP_CHECKPOINT_SHA256,
        "wav2lip_inference_sha256": WAV2LIP_INFERENCE_SHA256, "records": records,
    }, replace=manifest_path.exists())


def _probe_video(path: Path) -> dict[str, Any]:
    command = [str(FFPROBE), "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height,avg_frame_rate,r_frame_rate,start_time,duration,nb_frames",
               "-of", "json", str(path)]
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
    if result.returncode:
        raise MatrixError("ffprobe failed for Wav2Lip output")
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise MatrixError("Wav2Lip output does not have one selected video stream")
    stream = streams[0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if rate not in {"25/1", "25/1.0"}:
        raise MatrixError(f"Wav2Lip output frame rate is not 25 fps: {rate}")
    return {key: stream.get(key) for key in ("width", "height", "avg_frame_rate", "r_frame_rate", "start_time", "duration", "nb_frames")}


def _extract_fixed_frames(video: Path, frame_dir: Path, box_xyxy: list[int]) -> tuple[list[np.ndarray], dict[str, Any]]:
    import cv2
    from scripts.experiments.wav2lip_roi_peak_recheck.worker import _crop_visual_frame

    if frame_dir.exists():
        raise MatrixError(f"refusing to overwrite decoded frames: {frame_dir}")
    frame_dir.mkdir(parents=True, exist_ok=False)
    pattern = frame_dir / "%06d.jpg"
    command = [str(FFMPEG), "-nostdin", "-y", "-loglevel", "error", "-i", str(video),
               "-threads", "1", "-f", "image2", str(pattern)]
    # -y only targets the fresh, isolated frame directory.
    _run(command)
    paths = sorted(frame_dir.glob("*.jpg"))
    if not paths or [p.name for p in paths] != [f"{i:06d}.jpg" for i in range(1, len(paths) + 1)]:
        raise MatrixError("decoded video frames are empty or non-contiguous")
    x1, y1, x2, y2 = map(int, box_xyxy)
    box_worker = (y1, y2, x1, x2)
    frames: list[np.ndarray] = []
    decoded_shape: tuple[int, ...] | None = None
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise MatrixError(f"could not decode frame: {path.name}")
        if decoded_shape is None:
            decoded_shape = image.shape
        elif image.shape != decoded_shape:
            raise MatrixError("video frame geometry changes")
        frames.append(_crop_visual_frame(image, box_worker))
    return frames, {"frame_count": len(frames), "decoded_frame_shape": list(decoded_shape or ()),
                    "crop_box_xyxy": [x1, y1, x2, y2], "crop_box_worker_order": list(box_worker),
                    "crop_size": [224, 224], "resize": "cv2.INTER_AREA", "ffmpeg_command": command,
                    "frame_sha256": [sha256_file(path) for path in paths]}


def _visual_features(engine: Any, frames: list[np.ndarray]) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with engine._torch.inference_mode():
        for start in range(0, len(frames) - 4, engine.batch_size):
            stop = min(len(frames) - 4, start + engine.batch_size)
            windows = [frames[index:index + 5] for index in range(start, stop)]
            batch = engine._torch.from_numpy(engine._visual_batch(windows)).to(engine.device)
            features = engine.network.forward_lip(batch).detach().cpu().numpy().astype(np.float32)
            outputs.append(features)
    if not outputs:
        raise MatrixError("video has fewer than five frames")
    result = np.concatenate(outputs, axis=0)
    if result.ndim != 2 or result.shape[1] != 1024 or not np.isfinite(result).all():
        raise MatrixError("SyncNet visual embedding is invalid")
    return result


def score(config: dict[str, Any], rows: list[dict[str, Any]], run_dir: Path,
          speakers: set[str]) -> None:
    import torch
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine

    if sha256_file(SYNCNET_MODEL) != SYNCNET_MODEL_SHA256:
        raise MatrixError("SyncNet model pin mismatch")
    video_manifest = json.loads((run_dir / "video_manifest.json").read_text(encoding="utf-8"))
    audio_manifest = json.loads((run_dir / "audio_manifest.json").read_text(encoding="utf-8"))
    audio_by = {(r["speaker"], r["role"]): r for r in audio_manifest["records"]}
    video_by = {(r["speaker"], r["video_role"]): r for r in video_manifest["records"]}
    config_sha = sha256_file(Path(config["_path"]))
    matrix_path = run_dir / "matrix_scores.json"
    prior_results: dict[str, dict[str, Any]] = {}
    if matrix_path.exists():
        old_matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        if old_matrix.get("config_sha256") != config_sha or old_matrix.get("syncnet_model_sha256") != SYNCNET_MODEL_SHA256:
            raise MatrixError("existing score manifest does not match the frozen scoring contract")
        for item in old_matrix.get("speaker_results", []):
            sid = item.get("speaker")
            if sid in prior_results:
                raise MatrixError("existing score manifest has duplicate speaker records")
            prior_results[sid] = item
    torch.set_num_threads(2)
    engine = SyncNetEngine(model_path=SYNCNET_MODEL, batch_size=32, device="cuda")
    results: list[dict[str, Any]] = []
    try:
        for row in rows:
            speaker = row["speaker"]
            if speaker not in speakers:
                continue
            score_path = run_dir / "scores" / f"{speaker}.json"
            if score_path.exists():
                saved = json.loads(score_path.read_text(encoding="utf-8"))
                if saved.get("speaker") != speaker:
                    raise MatrixError(f"saved speaker score identity mismatch: {speaker}")
                for role in ROLES:
                    vrec = video_by[(speaker, role)]
                    arec = audio_by[(speaker, role)]
                    vf = saved.get("features", {}).get("visual", {}).get(role, {})
                    af = saved.get("features", {}).get("audio", {}).get(role, {})
                    if vf.get("video_sha256") != vrec.get("video_sha256"):
                        raise MatrixError(f"saved visual feature provenance mismatch: {speaker}/{role}")
                    if af.get("extract", {}).get("pcm_sha256") != arec.get("canonical_pcm_sha256"):
                        raise MatrixError(f"saved audio feature provenance mismatch: {speaker}/{role}")
                prior_results[speaker] = saved
                print("SCORE_REUSE", speaker, flush=True)
                continue
            visual_features: dict[str, np.ndarray] = {}
            audio_features: dict[str, np.ndarray] = {}
            feature_meta: dict[str, Any] = {"visual": {}, "audio": {}}
            for role in ROLES:
                audio_record = audio_by[(speaker, role)]
                source = Path(audio_record["canonical_path"])
                if sha256_file(source) != audio_record["canonical_sha256"]:
                    raise MatrixError(f"canonical audio changed after freeze: {speaker}/{role}")
                audio_features[role], ameta = engine.extract_audio(source)
                audio_path = run_dir / "features" / speaker / f"audio_{role}.npy"
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                if audio_path.exists():
                    raise MatrixError(f"refusing to overwrite audio features: {audio_path}")
                np.save(audio_path, audio_features[role], allow_pickle=False)
                feature_meta["audio"][role] = {"path": str(audio_path.resolve()), "sha256": sha256_file(audio_path),
                                                "shape": list(audio_features[role].shape), "extract": ameta}
                video_record = video_by[(speaker, role)]
                video_path = Path(video_record["video_path"])
                if sha256_file(video_path) != video_record["video_sha256"]:
                    raise MatrixError(f"Wav2Lip video changed after freeze: {speaker}/{role}")
                probe = _probe_video(video_path)
                frames, vmeta = _extract_fixed_frames(video_path, run_dir / "decoded_frames" / speaker / role,
                                                      config["fixed_face"]["score_box_xyxy"])
                visual_features[role] = _visual_features(engine, frames)
                visual_path = run_dir / "features" / speaker / f"visual_{role}.npy"
                if visual_path.exists():
                    raise MatrixError(f"refusing to overwrite visual features: {visual_path}")
                np.save(visual_path, visual_features[role], allow_pickle=False)
                feature_meta["visual"][role] = {"path": str(visual_path.resolve()), "sha256": sha256_file(visual_path),
                                                 "shape": list(visual_features[role].shape),
                                                 "video_sha256": video_record["video_sha256"],
                                                 "probe": probe, "extract": vmeta}
            all_rows = [min(visual_features[v].shape[0], audio_features[a].shape[0])
                        for v in ROLES for a in ROLES]
            support_start = LAG_RADIUS
            support_end = min(all_rows) - LAG_RADIUS
            if support_end <= support_start:
                raise MatrixError(f"no full 31-lag common support for {speaker}")
            cells: dict[str, dict[str, Any]] = {}
            c_values: dict[tuple[str, str], float] = {}
            for video_role in ROLES:
                visual = visual_features[video_role]
                for audio_role in ROLES:
                    audio = audio_features[audio_role]
                    full = SyncNetEngine.distance_matrix(visual, audio)
                    common = np.ascontiguousarray(full[support_start:support_end], dtype=np.float64)
                    if common.shape != (support_end - support_start, 31):
                        raise MatrixError("cell does not cover the frozen full-lag support")
                    curve = common.mean(axis=0, dtype=np.float64)
                    argmin = int(np.argmin(curve))
                    b_median, dmin, d0 = float(np.median(curve)), float(curve.min()), float(curve[LAG_RADIUS])
                    c_best = b_median - dmin
                    distance_path = run_dir / "distances" / speaker / f"V_{video_role}__A_{audio_role}.npy"
                    distance_path.parent.mkdir(parents=True, exist_ok=True)
                    if distance_path.exists():
                        raise MatrixError(f"refusing to overwrite distance array: {distance_path}")
                    np.save(distance_path, common, allow_pickle=False)
                    cells[f"V_{video_role}_A_{audio_role}"] = {
                        "video_role": video_role, "audio_role": audio_role,
                        "distance_path": str(distance_path.resolve()), "distance_sha256": sha256_file(distance_path),
                        "distance_shape": list(common.shape), "common_window_count": int(common.shape[0]),
                        "lag_frames": list(range(-LAG_RADIUS, LAG_RADIUS + 1)),
                        "full_mean_distance_curve": curve.tolist(), "B_median": b_median,
                        "Dmin": dmin, "D0": d0, "C": c_best,
                        "offset_audio_minus_video_frames": LAG_RADIUS - argmin,
                        "argmin_index": argmin, "offset_at_lag_boundary": argmin in (0, 30),
                        "C_3dp": round(c_best, 3), "Dmin_3dp": round(dmin, 3),
                        "D0_3dp": round(d0, 3),
                    }
                    c_values[(video_role, audio_role)] = c_best
            per_driver = []
            for video_role in ROLES:
                same_alt = "N2" if video_role == "N1" else "N1" if video_role == "N2" else "T2" if video_role == "T1" else "T1"
                opposite = [role for role in ROLES if ROLE_CLASS[role] != ROLE_CLASS[video_role]]
                matched = c_values[(video_role, video_role)]
                within_penalty = matched - c_values[(video_role, same_alt)]
                cross_mean = float(np.mean([c_values[(video_role, role)] for role in opposite]))
                cross_penalty = matched - cross_mean
                per_driver.append({
                    "video_role": video_role, "matched_audio_role": video_role,
                    "same_class_other_take_role": same_alt, "opposite_class_audio_roles": opposite,
                    "matched_minus_other_same_class_C": within_penalty,
                    "matched_minus_cross_class_mean_C": cross_penalty,
                    "cross_minus_within_penalty_C_descriptive": cross_penalty - within_penalty,
                })
            support = {
                "query_start_window_inclusive": support_start,
                "query_end_window_exclusive": support_end,
                "query_window_count": support_end - support_start,
                "lag_radius_frames": LAG_RADIUS,
                "query_anchor_start_s": support_start / FPS,
                "query_anchor_end_s_exclusive": support_end / FPS,
                "visual_window_convention": "five frames [q,q+4], SyncNet row q; frame rate 25 Hz",
                "audio_window_convention": "MFCC frames [4q,4q+19], q-th SyncNet row, source PCM timeline",
                "source_timeline": "raw native recording timeline after mono/16 kHz/RMS canonicalization; no local alignment",
                "speech_content_coverage": "not estimated; report duration and support fractions only",
                "common_support_fraction_of_each": {
                    role: {"audio": (support_end - support_start) / audio_features[role].shape[0],
                           "video": (support_end - support_start) / visual_features[role].shape[0]}
                    for role in ROLES
                },
                "feature_window_counts": {"audio": {r: int(audio_features[r].shape[0]) for r in ROLES},
                                           "video": {r: int(visual_features[r].shape[0]) for r in ROLES}},
            }
            result = {"speaker": speaker, "support": support, "cells": cells,
                      "per_driver_contrasts": per_driver, "features": feature_meta}
            results.append(result)
            write_json(score_path, result)
            prior_results[speaker] = result
            print("SCORED", speaker, "support", support["query_window_count"],
                  "self_C", {r: round(c_values[(r, r)], 3) for r in ROLES}, flush=True)
    finally:
        engine.close()
    all_results = [prior_results[sid] for sid in sorted(prior_results)]
    expected_ids = {row["speaker"] for row in rows}
    status = "complete" if expected_ids <= set(prior_results) else "partial"
    summary = _speaker_equal_summary(all_results)
    write_json(matrix_path, {
        "schema_version": 1, "status": status, "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha,
        "syncnet_model_sha256": SYNCNET_MODEL_SHA256, "lag_support": list(range(-15, 16)),
        "metric_definitions": {
            "B_median": "median of the mean distance curve over all 31 lags",
            "Dmin": "minimum of the mean distance curve over all 31 lags",
            "D0": "mean distance at lag zero",
            "C": "B_median - Dmin; higher means a sharper best-lag minimum",
            "offset_audio_minus_video_frames": "15 - argmin(curve); range -15..+15",
            "cross_minus_within_penalty_C_descriptive": "matched-minus-cross-class-mean C penalty minus matched-minus-same-class-other-take C penalty; descriptive only",
        },
        "primary_clock": "unwarped raw native time; full common valid support, no SyncNet-best-lag realignment",
        "speaker_equal_summary": summary,
        "speaker_results": all_results,
    }, replace=matrix_path.exists())


def _speaker_equal_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive source-level contrasts with ordinary equal-speaker bootstrap CIs."""
    contrast_rows: list[dict[str, Any]] = []
    for result in results:
        cells = result["cells"]
        c = {(v, a): float(cells[f"V_{v}_A_{a}"]["C"]) for v in ROLES for a in ROLES}
        matched_n = (c[("N1", "N1")] + c[("N2", "N2")]) / 2
        within_n = (c[("N1", "N2")] + c[("N2", "N1")]) / 2
        cross_n = np.mean([c[(v, a)] for v in ("N1", "N2") for a in ("T1", "T2")])
        matched_t = (c[("T1", "T1")] + c[("T2", "T2")]) / 2
        within_t = (c[("T1", "T2")] + c[("T2", "T1")]) / 2
        cross_t = np.mean([c[(v, a)] for v in ("T1", "T2") for a in ("N1", "N2")])
        contrast_rows.append({
            "speaker": result["speaker"],
            "matched_N_C": float(matched_n), "within_N_C": float(within_n), "cross_N_C": float(cross_n),
            "within_N_loss_C": float(matched_n - within_n), "cross_N_loss_C": float(matched_n - cross_n),
            "within_minus_cross_N_C_descriptive": float(within_n - cross_n),
            "matched_T_C": float(matched_t), "within_T_C": float(within_t), "cross_T_C": float(cross_t),
            "within_T_loss_C": float(matched_t - within_t), "cross_T_loss_C": float(matched_t - cross_t),
            "within_minus_cross_T_C_descriptive": float(within_t - cross_t),
            "native_self_gap_T_minus_N_C": float(matched_t - matched_n),
        })
    metrics = [key for key in contrast_rows[0] if key != "speaker"] if contrast_rows else []
    rng = np.random.Generator(np.random.PCG64(20261015))
    n = len(contrast_rows)
    if n:
        sample_indices = rng.integers(0, n, size=(2000, n))
    summaries: dict[str, Any] = {}
    for metric in metrics:
        values = np.asarray([row[metric] for row in contrast_rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise MatrixError(f"non-finite source contrast: {metric}")
        if n >= 2:
            draws = values[sample_indices].mean(axis=1)
            ci = [float(x) for x in np.quantile(draws, [0.025, 0.975])]
        else:
            ci = None
        summaries[metric] = {
            "mean": float(values.mean()), "ci95_descriptive": ci,
            "positive_sources": int(np.sum(values > 0)), "source_count": n,
        }
    return {
        "unit": "speaker/source, equal weight",
        "bootstrap": {"draws": 2000, "seed": 20261015, "interval": "ordinary percentile 95%, descriptive"},
        "rows": contrast_rows,
        "summary": summaries,
        "interpretation": "within_minus_cross is descriptive and does not identify a universal source preference; native_self_gap is the matched T minus matched N C contrast",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare-natural", "prepare", "render", "score"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--speaker-ids", default="", help="comma-separated frozen speaker subset")
    args = parser.parse_args()
    config_path, run_dir = args.config.resolve(strict=True), args.run_dir.resolve()
    config, rows = load_config(config_path, run_dir, allow_natural_only=args.phase == "prepare-natural")
    run_dir.mkdir(parents=True, exist_ok=True)
    config["_path"] = str(config_path)
    speakers = {row["speaker"] for row in rows}
    selected = {item.strip() for item in args.speaker_ids.split(",") if item.strip()} if args.speaker_ids else speakers
    if not selected or not selected <= speakers:
        raise SystemExit("speaker selection must be a nonempty subset of the frozen protocol")
    if args.phase in {"prepare-natural", "prepare"}:
        # Preparation has no GPU forward and must cover exactly the selected cohort.
        rows = [row for row in rows if row["speaker"] in selected]
        prepare(config, rows, run_dir)
    elif args.phase == "render":
        render(config, rows, run_dir, selected)
    else:
        score(config, rows, run_dir, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
