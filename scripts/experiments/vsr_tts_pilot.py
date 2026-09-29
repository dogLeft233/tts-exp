"""Reproducible visual-speech content pilot for the AVTR-1 TTS comparison.

The script intentionally has four explicit stages.  ``audit`` freezes the
correct AVTR-1 text binding, ``extract`` runs the already-installed Chinese
visual-only AVSR model, ``analyze`` computes target-vs-decoy content margins,
and ``validate`` independently recomputes CTC and paired statistics.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# The runner is both importable as ``scripts.experiments...`` and executable
# directly from the repository root.  In the latter case Python starts with
# ``scripts/experiments`` on sys.path, so add the repository root before the
# package import.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments.vsr_tts_metrics import (
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    adjacent_repeated_character_count,
    build_decoys,
    content_margin,
    ctc_nll,
    decision_status,
    greedy_cer,
    length_normalized_nll,
    make_views,
    normalize_text,
    paired_summary,
    token_ids,
)


AVSR_ROOT = ROOT / "third_party" / "AVSR"
DEFAULT_RUN = ROOT / "runs" / "vsr_tts_content_v1"
SAMPLE_IDS = tuple(range(1, 14))
CONDITIONS = ("natural", "tts")
VIEWS = ("native", "frozen", "reversed", "matched", "matched_frozen")
MODEL_REL = Path("benchmarks/CMLR/models/CMLR_V_WER8.0/model.pth")
MODEL_JSON_REL = Path("benchmarks/CMLR/models/CMLR_V_WER8.0/model.json")
LM_REL = Path("benchmarks/CMLR/language_models/lm_zh/model.pth")
LM_JSON_REL = Path("benchmarks/CMLR/language_models/lm_zh/model.json")
PROTOCOL_ID = "vsr_tts_content_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically and reject accidental NaN/Infinity values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_command(arguments: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(list(arguments), check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _ffprobe(path: Path, *, frames: bool = False) -> Dict[str, Any]:
    entries = "stream=index,codec_type,codec_name,r_frame_rate,avg_frame_rate,nb_frames,start_time,duration:format=duration"
    arguments = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json"]
    if frames:
        arguments = [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_entries", "frame=best_effort_timestamp_time,pkt_duration_time",
            "-of", "json",
        ]
    else:
        arguments.extend(["-show_entries", entries])
    arguments.append(str(path))
    result = run_command(arguments)
    return json.loads(result.stdout)


def _stream_summary(path: Path) -> Dict[str, Any]:
    payload = _ffprobe(path)
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = [item for item in streams if item.get("codec_type") == "audio"]
    if video is None:
        raise ValueError("no video stream: {}".format(path))
    rate = str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "")
    fps = None
    if "/" in rate:
        numerator, denominator = rate.split("/", 1)
        if float(denominator) != 0:
            fps = float(numerator) / float(denominator)
    frame_count = video.get("nb_frames")
    return {
        "video_codec": video.get("codec_name"),
        "audio_stream_count": len(audio),
        "fps": fps,
        "frame_count": int(frame_count) if frame_count not in (None, "N/A", "") else None,
        "start_time": float(video.get("start_time", 0.0) or 0.0),
        "duration_s": float(video.get("duration") or payload.get("format", {}).get("duration") or 0.0),
        "width": video.get("width"),
        "height": video.get("height"),
    }


def _frame_digest(path: Path) -> Tuple[str, List[float]]:
    result = run_command(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "framemd5", "-"])
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip() and not line.startswith("#")]
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    timestamps = _ffprobe(path, frames=True).get("frames", [])
    parsed = [float(item["best_effort_timestamp_time"]) for item in timestamps if "best_effort_timestamp_time" in item]
    return digest, parsed


def _make_video_only_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_command(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-map", "0:v:0", "-c:v", "copy", "-an", str(destination)])


def parse_r2_readme(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pattern = re.compile(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(.*?)\s*\|\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        sample_id, wid, duration, transcript = match.groups()
        if not sample_id.isdigit() or not wid.startswith("BAC"):
            continue
        rows.append({
            "id": int(sample_id),
            "wid": wid.strip(),
            "documented_duration_s": float(duration.rstrip("s ")),
            "raw_text": transcript.strip(),
        })
    rows.sort(key=lambda row: row["id"])
    if [row["id"] for row in rows] != list(SAMPLE_IDS):
        raise ValueError("R2 README does not contain exactly sample IDs 1..13")
    return rows


def _load_model_char_list(model_json: Path) -> List[str]:
    payload = read_json(model_json)
    if not isinstance(payload, list) or len(payload) < 3 or not isinstance(payload[2], dict):
        raise ValueError("unexpected AVSR model JSON layout")
    char_list = [str(item) for item in payload[2].get("char_list", [])]
    if len(char_list) != 3363 or char_list[0] != "<blank>" or "<eos>" not in char_list:
        raise ValueError("unexpected CMLR character list")
    return char_list


def _git_commit(path: Path) -> Optional[str]:
    result = run_command(["git", "-C", str(path), "rev-parse", "HEAD"], check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _environment_payload(run_dir: Path) -> Dict[str, Any]:
    model_paths = {
        "model": AVSR_ROOT / MODEL_REL,
        "model_json": AVSR_ROOT / MODEL_JSON_REL,
        "lm": AVSR_ROOT / LM_REL,
        "lm_json": AVSR_ROOT / LM_JSON_REL,
    }
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "python": sys.version,
        "platform": platform.platform(),
        "avsr_root": str(AVSR_ROOT.resolve()),
        "avsr_commit": _git_commit(AVSR_ROOT),
        "model_paths": {key: str(value.resolve()) for key, value in model_paths.items()},
        "model_sha256": {key: sha256_file(value) for key, value in model_paths.items() if value.is_file()},
        "seed": BOOTSTRAP_SEED,
        "protocol_id": PROTOCOL_ID,
        "code_sha256": {
            "runner": sha256_file(Path(__file__)),
            "metrics": sha256_file(Path(__file__).with_name("vsr_tts_metrics.py")),
            "avsr_infer": sha256_file(AVSR_ROOT / "infer.py"),
            "avsr_pipeline": sha256_file(AVSR_ROOT / "pipelines" / "pipeline.py"),
            "avsr_data_module": sha256_file(AVSR_ROOT / "pipelines" / "data" / "data_module.py"),
            "avsr_video_process": sha256_file(AVSR_ROOT / "pipelines" / "detectors" / "mediapipe" / "video_process.py"),
        },
    }
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        payload["torch_error"] = repr(exc)
    write_json(run_dir / "environment.json", payload)
    return payload


def _make_config(run_dir: Path) -> Path:
    source = AVSR_ROOT / "configs" / "CMLR_V_WER8.0.ini"
    config = configparser.ConfigParser()
    if not config.read(str(source)):
        raise FileNotFoundError(str(source))
    config["model"]["model_path"] = str((AVSR_ROOT / MODEL_REL).resolve())
    config["model"]["model_conf"] = str((AVSR_ROOT / MODEL_JSON_REL).resolve())
    config["model"]["rnnlm"] = str((AVSR_ROOT / LM_REL).resolve())
    config["model"]["rnnlm_conf"] = str((AVSR_ROOT / LM_JSON_REL).resolve())
    destination = run_dir / "environment" / "CMLR_V_WER8.0.ini"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        config.write(handle)
    return destination


def _verify_frozen_inputs(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    """Reject resume when any audited source file or model weight changed."""

    for row in manifest.get("records", []):
        for key, expected in row.get("sha256", {}).items():
            if expected is None:
                continue
            path_key = {
                "natural": "natural_video",
                "tts": "tts_video",
                "natural_audio": "natural_audio",
                "face": "face",
            }.get(key)
            if path_key is None:
                continue
            path = Path(str(row[path_key]))
            if not path.is_file() or sha256_file(path) != str(expected):
                raise ValueError("refusing resume: frozen input changed: {}".format(path))
    for document in manifest.get("source_documents", {}).values():
        if not document:
            continue
        path = Path(str(document["path"]))
        if not path.is_file() or sha256_file(path) != str(document["sha256"]):
            raise ValueError("refusing resume: source document changed: {}".format(path))
    environment_path = run_dir / "environment.json"
    if environment_path.is_file():
        environment = read_json(environment_path)
        for key, path_value in environment.get("model_paths", {}).items():
            expected = environment.get("model_sha256", {}).get(key)
            path = Path(str(path_value))
            if expected and (not path.is_file() or sha256_file(path) != str(expected)):
                raise ValueError("refusing resume: model binding changed: {}".format(path))


def audit_inputs(run_dir: Path, *, smoke: bool = False, resume: bool = False) -> Dict[str, Any]:
    """Freeze AVTR-1 source binding and candidate texts."""

    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise ValueError("manifest exists; pass --resume or use a new run directory")
        existing = read_json(manifest_path)
        if bool(existing.get("smoke")) != bool(smoke):
            raise ValueError("refusing resume: frozen smoke/full scope changed")
        _verify_frozen_inputs(run_dir, existing)
        return existing
    readme = ROOT / "runs" / "r2_assets" / "README.md"
    transcript_path = ROOT / "runs" / "r2_assets" / "transcript.json"
    char_list = _load_model_char_list(AVSR_ROOT / MODEL_JSON_REL)
    readme_rows = parse_r2_readme(readme)
    transcript_payload = read_json(transcript_path) if transcript_path.is_file() else {}
    transcript_results = transcript_payload.get("results", {}) if isinstance(transcript_payload, dict) else {}
    records: List[Dict[str, Any]] = []
    for row in readme_rows:
        sample_id = int(row["id"])
        raw_text = str(row["raw_text"])
        normalized = normalize_text(raw_text)
        reasons: List[str] = []
        transcript_row = transcript_results.get(str(sample_id))
        transcript_match = transcript_row is None or str(transcript_row.get("text", "")) == raw_text
        if not transcript_match:
            reasons.append("TRANSCRIPT_JSON_CONFLICT")
        wid_match = re.search(r"S(\d+)", row["wid"])
        speaker = "S" + wid_match.group(1) if wid_match else None
        if speaker is None:
            reasons.append("MISSING_SPEAKER_IN_WID")
        try:
            target_token_ids = token_ids(normalized, char_list)
        except Exception as exc:
            target_token_ids = None
            reasons.append("TARGET_OOV:{}".format(str(exc)))
        natural_video = ROOT / "results" / "avtr1" / "natural_raw" / (str(sample_id) + ".mp4")
        tts_video = ROOT / "results" / "avtr1" / "tts_raw" / (str(sample_id) + ".mp4")
        natural_audio = ROOT / "data" / "data" / "audio" / (str(sample_id) + ".wav")
        face = ROOT / "data" / "data" / "image" / (str(sample_id) + ".png")
        paths = {"natural": natural_video, "tts": tts_video, "natural_audio": natural_audio, "face": face}
        hashes: Dict[str, Optional[str]] = {}
        probes: Dict[str, Any] = {}
        for name, path in paths.items():
            if not path.is_file():
                reasons.append("MISSING_{}".format(name.upper()))
                hashes[name] = None
                continue
            hashes[name] = sha256_file(path)
            if name in {"natural", "tts"}:
                try:
                    probes[name] = _stream_summary(path)
                    if probes[name].get("fps") is None or abs(float(probes[name]["fps"]) - 25.0) > 1e-3:
                        reasons.append("FPS_NOT_25_{}".format(name.upper()))
                except Exception as exc:
                    reasons.append("VIDEO_PROBE_{}_{}".format(name.upper(), type(exc).__name__))
            elif name == "natural_audio":
                try:
                    probes[name] = _stream_summary(path)
                except Exception:
                    # WAV has no video stream; a simple ffprobe format check is enough.
                    try:
                        raw = _ffprobe(path)
                        probes[name] = {"duration_s": float(raw.get("format", {}).get("duration") or 0.0)}
                    except Exception as exc:
                        reasons.append("AUDIO_PROBE_{}".format(type(exc).__name__))
        if len(reasons) == 0 and target_token_ids is not None:
            eligibility = "eligible"
        else:
            eligibility = "blocked"
        records.append({
            "id": sample_id,
            "wid": row["wid"],
            "speaker": speaker,
            "raw_text": raw_text,
            "normalized_text": normalized,
            "target_token_ids": target_token_ids,
            "natural_video": str(natural_video.resolve()),
            "tts_video": str(tts_video.resolve()),
            "natural_audio": str(natural_audio.resolve()),
            "face": str(face.resolve()),
            "sha256": hashes,
            "probe": probes,
            "documented_duration_s": row["documented_duration_s"],
            "transcript_json_match": bool(transcript_match),
            "eligibility": eligibility,
            "reasons": reasons,
        })

    decoy_input = [
        {"id": row["id"], "normalized_text": row["normalized_text"], "token_ids": row["target_token_ids"]}
        for row in records
    ]
    decoys = build_decoys(decoy_input, count=5)
    for row in records:
        row["decoys"] = decoys.get(str(row["id"]), [])
        if len(row["decoys"]) < 5:
            row["eligibility"] = "blocked"
            row["reasons"].append("INSUFFICIENT_OOV_FREE_DECOYS")

    environment = _environment_payload(run_dir)
    config_path = _make_config(run_dir)
    environment["config_path"] = str(config_path.resolve())
    environment["config_sha256"] = sha256_file(config_path)
    write_json(run_dir / "environment.json", environment)
    all_hashes = []
    for row in records:
        all_hashes.append(json.dumps(row["sha256"], sort_keys=True))
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "audited",
        "candidate_count": len(SAMPLE_IDS),
        "smoke": bool(smoke),
        "run_sample_ids": [1] if smoke else list(SAMPLE_IDS),
        "seed": BOOTSTRAP_SEED,
        "source_documents": {
            "r2_readme": {"path": str(readme.resolve()), "sha256": sha256_file(readme)},
            "transcript_json": {"path": str(transcript_path.resolve()), "sha256": sha256_file(transcript_path)} if transcript_path.is_file() else None,
            "avtr_deployment": {"path": str((ROOT / "basic-memory" / "docs" / "deployment" / "avtr-1.md").resolve()), "sha256": sha256_file(ROOT / "basic-memory" / "docs" / "deployment" / "avtr-1.md")} if (ROOT / "basic-memory" / "docs" / "deployment" / "avtr-1.md").is_file() else None,
        },
        "provenance_level": "historical_documented",
        "config_path": str(config_path.resolve()),
        "char_list_sha256": sha256_text("\n".join(char_list)),
        "decoy_rule": "five other complete sentences sorted by absolute character length then numeric sample ID, before VSR outputs",
        "records": records,
        "source_fingerprint": sha256_text("|".join(all_hashes) + "|" + str(bool(smoke))),
        "old_invalid_pilot": [
            "/tmp/avsr_avtr1_pilot_20260918.json",
            "/tmp/avsr_avtr1_intermediate_20260918.json",
        ],
        "old_invalid_reason": "INVALID_REFERENCE_BINDING: AVTR-1 video IDs were paired with a different AISHELL100 transcript table",
    }
    write_json(manifest_path, manifest)
    return manifest


def load_vsr(run_dir: Path, device: str = "cuda:0") -> Any:
    """Load the official AVSR pipeline exactly once for a run."""

    import torch

    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; rerun explicitly with --device cpu")
    if str(AVSR_ROOT) not in sys.path:
        sys.path.insert(0, str(AVSR_ROOT))
    from pipelines.pipeline import InferencePipeline

    config_path = run_dir / "environment" / "CMLR_V_WER8.0.ini"
    if not config_path.is_file():
        raise FileNotFoundError(str(config_path))
    pipeline = InferencePipeline(str(config_path), detector="mediapipe", face_track=True, device=requested)
    pipeline.model.model.eval()
    return pipeline


def _set_determinism(seed: int = BOOTSTRAP_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def _face_count_qc(video_path: Path) -> Dict[str, Any]:
    """Count MediaPipe detections without changing the official crop path."""

    import mediapipe as mp
    import torchvision

    frames = torchvision.io.read_video(str(video_path), pts_unit="sec")[0].numpy()
    face_detection = mp.solutions.face_detection
    full = face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)
    short = face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)
    counts: List[int] = []
    for frame in frames:
        result = full.process(frame)
        if not result.detections:
            result = short.process(frame)
        counts.append(len(result.detections) if result.detections else 0)
    return {
        "frame_count": len(counts),
        "face_count_min": min(counts) if counts else 0,
        "face_count_max": max(counts) if counts else 0,
        "multi_face_frame_count": sum(value > 1 for value in counts),
        "zero_face_frame_count": sum(value == 0 for value in counts),
        "counts": counts,
    }


def _max_false_run(values: Sequence[bool]) -> int:
    current = 0
    maximum = 0
    for value in values:
        current = current + 1 if not value else 0
        maximum = max(maximum, current)
    return maximum


def _tensor_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _save_preview(x: np.ndarray, destination: Path) -> None:
    import cv2

    destination.parent.mkdir(parents=True, exist_ok=True)
    for label, index in (("first", 0), ("middle", len(x[0]) // 2), ("last", len(x[0]) - 1)):
        frame = np.asarray(x[0, index], dtype=np.float32)
        frame = np.clip((frame * 0.165 + 0.421) * 255.0, 0.0, 255.0).astype(np.uint8)
        cv2.imwrite(str(destination / (label + ".png")), frame)


def extract_views(pipeline: Any, row: Mapping[str, Any], condition: str, run_dir: Path) -> Dict[str, Any]:
    """Strip audio, run the official detector/crop, and return native x + QC."""

    import torch

    source = Path(str(row[condition + "_video"]))
    temporary = run_dir / "_tmp_video_only" / str(row["id"]) / (condition + ".mp4")
    if temporary.exists():
        temporary.unlink()
    _make_video_only_copy(source, temporary)
    try:
        source_digest, source_pts = _frame_digest(source)
        silent_digest, silent_pts = _frame_digest(temporary)
        source_summary = _stream_summary(source)
        silent_summary = _stream_summary(temporary)
        video_frames = pipeline.dataloader.load_video(str(temporary))
        landmarks = pipeline.process_landmarks(str(temporary), None)
        raw_valid = [landmark is not None for landmark in landmarks] if landmarks is not None else []
        if landmarks is None or not raw_valid:
            raise RuntimeError("MediaPipe returned no landmarks")
        video = pipeline.dataloader.video_process(video_frames, list(landmarks))
        if video is None:
            raise RuntimeError("official VideoProcess returned no crop")
        tensor = pipeline.dataloader.video_transform(torch.tensor(video))
        x = tensor.detach().cpu().numpy().astype(np.float32)
        if x.ndim != 4 or x.shape[0] != 1 or x.shape[2:] != (88, 88):
            raise RuntimeError("unexpected official mouth tensor shape: {}".format(tuple(x.shape)))
        if x.shape[1] != len(video_frames):
            raise RuntimeError("mouth tensor/frame count mismatch")
        face_qc = _face_count_qc(temporary)
        detection_fraction = float(sum(raw_valid) / len(raw_valid))
        qc = {
            "status": "COMPLETE",
            "source": str(source.resolve()),
            "video_only": str(temporary.resolve()),
            "source_frame_digest": source_digest,
            "video_only_frame_digest": silent_digest,
            "frame_digest_equal": source_digest == silent_digest,
            "source_pts": source_pts,
            "video_only_pts": silent_pts,
            "pts_equal": len(source_pts) == len(silent_pts) and all(abs(a - b) <= 1e-6 for a, b in zip(source_pts, silent_pts)),
            "source_stream": source_summary,
            "video_only_stream": silent_summary,
            "raw_landmark_valid_fraction": detection_fraction,
            "raw_landmark_missing_max_run": _max_false_run(raw_valid),
            "face_qc": face_qc,
            "single_face_qc": bool(face_qc["multi_face_frame_count"] == 0),
            "detection_qc": bool(detection_fraction >= 0.95 and _max_false_run(raw_valid) <= 5),
            "tensor_shape": list(x.shape),
            "tensor_sha256": _tensor_sha256(x),
            "duration_s": float(source_summary.get("duration_s") or 0.0),
        }
        qc["eligible_for_pair"] = bool(
            qc["frame_digest_equal"] and qc["pts_equal"] and qc["single_face_qc"] and qc["detection_qc"]
        )
        preview_dir = run_dir / "features" / str(row["id"]) / condition / "previews"
        _save_preview(x, preview_dir)
        return {"x": x, "qc": qc}
    finally:
        if temporary.exists():
            temporary.unlink()
        try:
            parent = temporary.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass


def _forward_view(pipeline: Any, view: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    tensor = torch.from_numpy(np.asarray(view, dtype=np.float32)).to(torch.device(device))
    with torch.no_grad():
        encoder = pipeline.model.model.encode(tensor)
        logp = pipeline.model.model.ctc.log_softmax(encoder.unsqueeze(0)).squeeze(0)
    encoder_np = encoder.detach().cpu().numpy().astype(np.float32)
    logp_np = logp.detach().cpu().numpy().astype(np.float32)
    if encoder_np.ndim != 2 or logp_np.ndim != 2 or not np.isfinite(logp_np).all():
        raise RuntimeError("non-finite or malformed VSR output")
    return encoder_np, logp_np


def _beam_text(pipeline: Any, native_x: np.ndarray, device: str) -> Optional[str]:
    import torch

    try:
        tensor = torch.from_numpy(native_x).to(torch.device(device))
        with torch.no_grad():
            return str(pipeline.model.infer(tensor))
    except Exception:
        return None


def _extract_one_pair(pipeline: Any, row: Mapping[str, Any], run_dir: Path, device: str, resume: bool) -> None:
    sample_id = str(row["id"])
    existing = [run_dir / "features" / sample_id / condition / (view + ".npz") for condition in CONDITIONS for view in VIEWS]
    qc_paths = [run_dir / "features" / sample_id / (condition + ".json") for condition in CONDITIONS]
    if resume and all(path.is_file() for path in existing + qc_paths):
        return
    prepared: Dict[str, Dict[str, Any]] = {}
    for condition in CONDITIONS:
        prepared[condition] = extract_views(pipeline, row, condition, run_dir)
    n_length = int(prepared["natural"]["x"].shape[1])
    t_length = int(prepared["tts"]["x"].shape[1])
    matched_length = int(math.floor((n_length + t_length) / 2.0 + 0.5))
    for condition in CONDITIONS:
        condition_dir = run_dir / "features" / sample_id / condition
        condition_dir.mkdir(parents=True, exist_ok=True)
        qc = dict(prepared[condition]["qc"])
        qc["matched_length"] = matched_length
        views = make_views(prepared[condition]["x"], matched_length=matched_length)
        for view_name in VIEWS:
            encoder, logp = _forward_view(pipeline, views[view_name], device)
            destination = condition_dir / (view_name + ".npz")
            payload = {"encoder": encoder, "logp": logp}
            if view_name == "native":
                payload["native_x"] = prepared[condition]["x"]
            np.savez_compressed(destination, **payload)
            if view_name == "native":
                beam = _beam_text(pipeline, prepared[condition]["x"], device)
                qc["beam_text"] = beam
                qc["native_repeat_max_abs"] = None
                qc["native_repeat_encoder_max_abs"] = None
                try:
                    repeat_encoder, repeat_logp = _forward_view(pipeline, views["native"], device)
                    qc["native_repeat_encoder_max_abs"] = float(np.max(np.abs(repeat_encoder - encoder)))
                    qc["native_repeat_max_abs"] = float(np.max(np.abs(repeat_logp - logp)))
                except Exception as exc:
                    qc["native_repeat_error"] = repr(exc)
        write_json(run_dir / "features" / sample_id / (condition + ".json"), qc)


def _load_feature(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _view_metrics(logp: np.ndarray, target: Sequence[int], decoys: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    target_loss = length_normalized_nll(logp, target)
    decoy_losses = []
    named_losses: Dict[str, float] = {"target": float(target_loss)}
    for decoy in decoys:
        loss = length_normalized_nll(logp, decoy["token_ids"])
        decoy_losses.append(loss)
        named_losses[str(decoy["id"])] = float(loss)
    margin = content_margin(target_loss, decoy_losses)
    candidate_values = [target_loss] + decoy_losses
    rank = 1 + sum(loss < target_loss for loss in decoy_losses)
    strict_top1 = sum(loss == target_loss for loss in decoy_losses) == 0 and rank == 1
    return {
        "status": "COMPLETE",
        "target_loss": float(target_loss),
        "decoy_losses": named_losses,
        "M": float(margin),
        "target_rank": int(rank),
        "strict_target_top1": bool(strict_top1),
        "candidate_count": len(candidate_values),
        "greedy_cer": float(greedy_cer(logp, target)),
        "blank_probability": float(np.mean(np.exp(logp[:, 0]))),
        "argmax_blank_fraction": float(np.mean(np.argmax(logp, axis=1) == 0)),
        "entropy": float(np.mean(-np.sum(np.exp(logp) * logp, axis=1))),
    }


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def analyze_run(run_dir: Path) -> Dict[str, Any]:
    manifest = read_json(run_dir / "manifest.json")
    all_records: List[Dict[str, Any]] = []
    row_by_id = {str(row["id"]): row for row in manifest["records"]}
    run_ids = [int(value) for value in manifest.get("run_sample_ids", SAMPLE_IDS)]
    for sample_id in run_ids:
        row = row_by_id[str(sample_id)]
        for condition in CONDITIONS:
            qc_path = run_dir / "features" / str(sample_id) / (condition + ".json")
            qc = read_json(qc_path) if qc_path.is_file() else {"status": "MISSING"}
            for view in VIEWS:
                feature_path = run_dir / "features" / str(sample_id) / condition / (view + ".npz")
                record: Dict[str, Any] = {"id": sample_id, "condition": condition, "view": view, "status": "FAILED", "feature": str(feature_path.resolve())}
                try:
                    if not feature_path.is_file():
                        raise FileNotFoundError(str(feature_path))
                    feature = _load_feature(feature_path)
                    metrics = _view_metrics(feature["logp"], row["target_token_ids"], row["decoys"])
                    record.update(metrics)
                    record["qc_eligible"] = bool(qc.get("eligible_for_pair", False))
                    record["frame_digest_equal"] = bool(qc.get("frame_digest_equal", False))
                    record["pts_equal"] = bool(qc.get("pts_equal", False))
                    record["native_repeat_max_abs"] = _safe_float(qc.get("native_repeat_max_abs"))
                    if view == "native":
                        beam_text = qc.get("beam_text")
                        beam_cer = None
                        if beam_text:
                            try:
                                char_list = _load_model_char_list(AVSR_ROOT / MODEL_JSON_REL)
                                beam_ids = token_ids(str(beam_text), char_list)
                                beam_cer = float(
                                    __import__("scripts.experiments.vsr_tts_metrics", fromlist=["levenshtein_distance"]).levenshtein_distance(
                                        row["target_token_ids"], beam_ids
                                    )
                                    / float(len(row["target_token_ids"]))
                                )
                            except Exception:
                                beam_cer = None
                        record["beam_text"] = beam_text
                        record["beam_cer"] = beam_cer
                        record["duration_s"] = _safe_float(qc.get("duration_s"))
                        record["detection_fraction"] = _safe_float(qc.get("raw_landmark_valid_fraction"))
                    record["status"] = "COMPLETE"
                except Exception as exc:
                    record["error"] = "{}: {}".format(type(exc).__name__, str(exc))
                all_records.append(record)
    records_path = run_dir / "records.jsonl"
    records_path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in all_records), encoding="utf-8")
    complete = [row for row in all_records if row.get("status") == "COMPLETE"]
    by_key = {(str(row["id"]), row["condition"], row["view"]): row for row in complete}
    pair_rows: List[Dict[str, Any]] = []
    for sample_id in run_ids:
        n_native = by_key.get((str(sample_id), "natural", "native"))
        n_frozen = by_key.get((str(sample_id), "natural", "frozen"))
        n_matched = by_key.get((str(sample_id), "natural", "matched"))
        n_matched_frozen = by_key.get((str(sample_id), "natural", "matched_frozen"))
        n_reversed = by_key.get((str(sample_id), "natural", "reversed"))
        t_native = by_key.get((str(sample_id), "tts", "native"))
        t_frozen = by_key.get((str(sample_id), "tts", "frozen"))
        t_matched = by_key.get((str(sample_id), "tts", "matched"))
        t_matched_frozen = by_key.get((str(sample_id), "tts", "matched_frozen"))
        t_reversed = by_key.get((str(sample_id), "tts", "reversed"))
        if all(item is not None for item in (n_native, n_frozen, n_matched, n_matched_frozen, n_reversed, t_native, t_frozen, t_matched, t_matched_frozen, t_reversed)):
            pair_rows.append({
                "id": sample_id,
                "g": float((t_native["M"] - t_frozen["M"]) - (n_native["M"] - n_frozen["M"])),
                "b": float(t_native["M"] - n_native["M"]),
                "gmatched": float((t_matched["M"] - t_matched_frozen["M"]) - (n_matched["M"] - n_matched_frozen["M"])),
                "r": float((t_native["M"] - t_reversed["M"])),
                "natural_q": float(n_native["M"] - n_frozen["M"]),
                "tts_q": float(t_native["M"] - t_frozen["M"]),
            })

    calibration: Dict[str, Any] = {"status": "UNCALIBRATED", "arms": {}}
    for condition in CONDITIONS:
        native = [row for row in complete if row["condition"] == condition and row["view"] == "native"]
        q_values = {str(item["id"]): next((pair["natural_q"] if condition == "natural" else pair["tts_q"] for pair in pair_rows if int(pair["id"]) == int(item["id"])), None) for item in native}
        q_values = {key: value for key, value in q_values.items() if value is not None}
        top1_fraction = float(sum(bool(item.get("strict_target_top1")) for item in native) / len(native)) if native else 0.0
        q_array = np.asarray(list(q_values.values()), dtype=np.float64)
        calibration["arms"][condition] = {
            "native_count": len(native),
            "strict_native_top1_fraction": top1_fraction,
            "q_count": len(q_values),
            "q_positive_fraction": float(np.mean(q_array > 0.0)) if len(q_array) else 0.0,
            "q_mean": float(np.mean(q_array)) if len(q_array) else None,
        }
    calibration_ok = all(
        details["strict_native_top1_fraction"] >= 0.60
        and details["q_positive_fraction"] >= 0.75
        and details["q_mean"] is not None
        and details["q_mean"] > 0.0
        for details in calibration["arms"].values()
    )
    calibration["status"] = "CALIBRATED_ON_COHORT" if calibration_ok else "INCONCLUSIVE_VSR_VALIDITY"
    paired = paired_summary(pair_rows, min_pairs=10)
    technical_ok = bool(pair_rows) and all(
        bool(row.get("qc_eligible")) for row in complete if row["view"] == "native"
    )
    historical_sync = None
    sync_path = ROOT / "results" / "avtr1" / "04_eval" / "eval_meta.json"
    if sync_path.is_file():
        sync = read_json(sync_path)
        values = sync.get("results", {})
        natural_scores = [float(item["sync_c"]) for key, item in values.items() if key.startswith("natural_raw:")]
        tts_scores = [float(item["sync_c"]) for key, item in values.items() if key.startswith("tts_raw:")]
        historical_sync = {
            "status": "DESCRIPTIVE_HISTORICAL_ONLY",
            "natural_mean": float(np.mean(natural_scores)) if natural_scores else None,
            "tts_mean": float(np.mean(tts_scores)) if tts_scores else None,
            "delta": float(np.mean(tts_scores) - np.mean(natural_scores)) if natural_scores and tts_scores else None,
            "path": str(sync_path.resolve()),
        }
    analysis = {
        "schema_version": 1,
        "status": "complete",
        "run_sample_ids": run_ids,
        "denominators": {"candidate_count": 13, "paired_complete": len(pair_rows), "records_complete": len(complete), "records_total": len(all_records)},
        "calibration": calibration,
        "paired": paired,
        "pair_rows": pair_rows,
        "engineering": {"technical_ok": technical_ok, "independent_validation": "PENDING"},
        "provisional_decision": decision_status(engineering_ok=technical_ok, calibration=calibration, paired=paired),
        "historical_syncnet": historical_sync,
        "invalid_old_pilot": manifest.get("old_invalid_reason"),
    }
    write_json(run_dir / "analysis.json", analysis)
    _write_report(run_dir, analysis)
    return analysis


def _independent_ctc_nll(logp: np.ndarray, target: Sequence[int], blank: int = 0) -> float:
    values = np.asarray(logp, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("invalid independent logp")
    target_values = [int(value) for value in target]
    required = len(target_values) + adjacent_repeated_character_count(target_values)
    if not target_values or len(values) < required:
        raise ValueError("unreachable independent target")
    expanded: List[int] = [int(blank)]
    for value in target_values:
        expanded.extend([value, int(blank)])
    states = len(expanded)
    alpha = np.full(states, -np.inf, dtype=np.float64)
    alpha[0] = values[0, int(blank)]
    alpha[1] = values[0, target_values[0]]
    for time_index in range(1, len(values)):
        next_alpha = np.full(states, -np.inf, dtype=np.float64)
        for state, label in enumerate(expanded):
            candidates = [alpha[state]]
            if state > 0:
                candidates.append(alpha[state - 1])
            if state > 1 and label != int(blank) and label != expanded[state - 2]:
                candidates.append(alpha[state - 2])
            next_alpha[state] = np.logaddexp.reduce(np.asarray(candidates, dtype=np.float64)) + values[time_index, label]
        alpha = next_alpha
    log_probability = np.logaddexp(alpha[-1], alpha[-2])
    result = float(-log_probability / float(len(target_values)))
    if not math.isfinite(result):
        raise ValueError("independent CTC result is non-finite")
    return result


def _independent_bootstrap(rows: Sequence[Mapping[str, Any]], metric: str) -> Dict[str, Any]:
    values = {str(row["id"]): float(row[metric]) for row in rows}
    ordered = sorted(values)
    array = np.asarray([values[key] for key in ordered], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)), dtype=np.int64)
    means = array[indices].mean(axis=1, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "positive_fraction": float(np.mean(array > 0.0)),
        "values": {key: float(values[key]) for key in ordered},
        "count": len(array),
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
    }


def _write_report(run_dir: Path, analysis: Mapping[str, Any]) -> None:
    calibration = analysis.get("calibration", {})
    paired = analysis.get("paired", {})
    lines = [
        "# VSR 视觉内容辨识与 TTS 配对验证",
        "",
        "本报告只解释冻结的 CMLR visual-only VSR 对 AVTR-1 历史视频的内容支持；不把 VSR 分数当成人类评价或毫秒同步真值。",
        "",
        "- 样本分母：13 个固定候选，实际配对完成：{}".format(paired.get("pair_count", 0)),
        "- 测量校准：{}".format(calibration.get("status")),
        "- 当前判读：{}".format(analysis.get("decision", analysis.get("provisional_decision"))),
        "- 旧 pilot：INVALID_REFERENCE_BINDING（跨数据集同名 ID 文本绑定），旧 JSON 不作为证据。",
        "",
        "## 配对指标",
        "",
        "| metric | mean | 95% CI | positive fraction |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("g", "b", "gmatched", "r"):
        details = paired.get("metrics", {}).get(metric, {})
        if details.get("status") == "COMPLETE":
            lines.append("| {} | {:.6f} | [{:.6f}, {:.6f}] | {:.3f} |".format(metric, details["mean"], details["ci95"][0], details["ci95"][1], details["positive_fraction"]))
        else:
            lines.append("| {} | — | — | — |".format(metric))
    lines.extend(["", "SyncNet 旧结果只作为描述性历史背景，未用于筛选样本或重新评分。", ""])
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_smoke_status(run_dir: Path, manifest: Mapping[str, Any], stage: str) -> None:
    if not bool(manifest.get("smoke")):
        return
    feature_files = list((run_dir / "features").glob("*/*/*.npz")) if (run_dir / "features").is_dir() else []
    write_json(
        run_dir / "smoke.json",
        {
            "schema_version": 1,
            "status": "COMPLETE" if len(feature_files) == 10 else "PARTIAL",
            "stage": stage,
            "smoke_sample_ids": manifest.get("run_sample_ids", [1]),
            "feature_file_count": len(feature_files),
            "required_feature_file_count": 10,
            "formal_denominator": 13,
            "formal_conclusion_allowed": False,
        },
    )


def validate_run(run_dir: Path) -> Dict[str, Any]:
    """Independently recompute all saved CTC cells and pair statistics."""

    manifest = read_json(run_dir / "manifest.json")
    analysis = read_json(run_dir / "analysis.json")
    records = _load_jsonl(run_dir / "records.jsonl")
    row_by_id = {str(row["id"]): row for row in manifest["records"]}
    checked = 0
    max_ctc_error = 0.0
    max_logp_row_error = 0.0
    failures: List[str] = []
    for record in records:
        if record.get("status") != "COMPLETE":
            continue
        feature_path = Path(record["feature"])
        feature = _load_feature(feature_path)
        row_error = float(np.max(np.abs(np.exp(feature["logp"].astype(np.float64)).sum(axis=1) - 1.0)))
        max_logp_row_error = max(max_logp_row_error, row_error)
        if row_error > 1e-5:
            failures.append("LOGP_NORMALIZATION:{}:{}".format(record["id"], record["view"]))
        source = row_by_id[str(record["id"])]
        candidates = [("target", source["target_token_ids"])] + [(str(item["id"]), item["token_ids"]) for item in source["decoys"]]
        producer_losses = dict(record.get("decoy_losses", {}))
        for key, target in candidates:
            independent = _independent_ctc_nll(feature["logp"], target)
            produced = float(producer_losses[key])
            error = abs(independent - produced)
            max_ctc_error = max(max_ctc_error, error)
            checked += 1
            if error > 1e-5:
                failures.append("CTC:{}:{}:{}".format(record["id"], record["view"], key))
    expected_pair = analysis.get("pair_rows", [])
    stats_error = 0.0
    for metric in ("g", "b", "gmatched", "r"):
        independent = _independent_bootstrap(expected_pair, metric) if expected_pair else None
        produced = analysis.get("paired", {}).get("metrics", {}).get(metric, {})
        if independent is None or produced.get("status") != "COMPLETE":
            continue
        for key in ("mean", "median", "positive_fraction"):
            stats_error = max(stats_error, abs(float(independent[key]) - float(produced[key])))
        for left, right in zip(independent["ci95"], produced["ci95"]):
            stats_error = max(stats_error, abs(float(left) - float(right)))
        if independent["values"] != produced.get("values"):
            failures.append("PAIR_VALUES:{}".format(metric))
    for sample_id in manifest.get("run_sample_ids", SAMPLE_IDS):
        for condition in CONDITIONS:
            qc_path = run_dir / "features" / str(sample_id) / (condition + ".json")
            if not qc_path.is_file():
                failures.append("MISSING_QC:{}:{}".format(sample_id, condition))
                continue
            qc = read_json(qc_path)
            if not qc.get("frame_digest_equal") or not qc.get("pts_equal"):
                failures.append("AUDIO_ISOLATION:{}:{}".format(sample_id, condition))
            repeat = qc.get("native_repeat_max_abs")
            repeat_encoder = qc.get("native_repeat_encoder_max_abs")
            if repeat is None or repeat_encoder is None:
                failures.append("MISSING_REPEAT_CHECK:{}:{}".format(sample_id, condition))
            elif float(repeat) > 1e-5 or float(repeat_encoder) > 1e-5:
                failures.append("NONDETERMINISTIC:{}:{}".format(sample_id, condition))
    status = "PASS" if not failures and max_ctc_error <= 1e-5 and max_logp_row_error <= 1e-5 and stats_error <= 1e-8 else "FAIL"
    validation = {
        "schema_version": 1,
        "status": status,
        "independent_dp": True,
        "checked_ctc_cells": checked,
        "max_ctc_abs_error": max_ctc_error,
        "max_logp_row_probability_error": max_logp_row_error,
        "statistics_max_abs_error": stats_error,
        "tolerances": {"logp": 1e-5, "ctc_abs": 1e-5, "statistics_abs": 1e-8, "repeat_encoder_abs": 1e-5},
        "failures": failures,
    }
    write_json(run_dir / "validation.json", validation)
    engineering = dict(analysis.get("engineering", {}))
    engineering["independent_validation"] = status
    engineering_ok = bool(engineering.get("technical_ok")) and status == "PASS"
    analysis["engineering"] = engineering
    analysis["decision"] = decision_status(engineering_ok=engineering_ok, calibration=analysis.get("calibration", {}), paired=analysis.get("paired", {}))
    write_json(run_dir / "analysis.json", analysis)
    _write_report(run_dir, analysis)
    return validation


def _ensure_manifest(run_dir: Path, smoke: bool, resume: bool) -> Dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if bool(manifest.get("smoke")) != bool(smoke):
            raise ValueError("run scope mismatch: smoke/full cannot share a frozen run")
        _verify_frozen_inputs(run_dir, manifest)
        return manifest
    return audit_inputs(run_dir, smoke=smoke, resume=resume)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--stage", choices=("audit", "extract", "analyze", "validate", "all"), default="audit")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume and args.stage in {"audit", "all"}:
        raise ValueError("refusing non-empty output directory without --resume: {}".format(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    _set_determinism()
    if args.stage in {"audit", "all"}:
        manifest = audit_inputs(run_dir, smoke=args.smoke, resume=args.resume)
    else:
        manifest = _ensure_manifest(run_dir, args.smoke, args.resume)
    if args.stage == "audit":
        print(json.dumps({"status": manifest.get("status"), "run_sample_ids": manifest.get("run_sample_ids")}, ensure_ascii=False))
        return 0
    if args.stage in {"extract", "all"}:
        pipeline = load_vsr(run_dir, device=args.device)
        row_by_id = {str(row["id"]): row for row in manifest["records"]}
        for sample_id in manifest.get("run_sample_ids", SAMPLE_IDS):
            row = row_by_id[str(sample_id)]
            if row.get("eligibility") != "eligible":
                continue
            _extract_one_pair(pipeline, row, run_dir, args.device, args.resume)
        _write_smoke_status(run_dir, manifest, "extract")
    if args.stage in {"analyze", "all"}:
        analysis = analyze_run(run_dir)
    else:
        analysis = read_json(run_dir / "analysis.json") if (run_dir / "analysis.json").is_file() else None
    if analysis is not None:
        _write_smoke_status(run_dir, manifest, "analyze")
    if args.stage in {"validate", "all"}:
        validation = validate_run(run_dir)
    else:
        validation = None
    if validation is not None:
        _write_smoke_status(run_dir, manifest, "validate")
    result = {"stage": args.stage, "run_dir": str(run_dir), "analysis_decision": analysis.get("decision", analysis.get("provisional_decision")) if analysis else None, "validation": validation.get("status") if validation else None}
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
