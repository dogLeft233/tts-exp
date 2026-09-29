"""Ditto-50 VSR/Sync-C linkage experiment.

The runner is intentionally a thin, staged wrapper around the already
validated CMLR visual-only VSR code and the repository's SyncNet scripts.  It
freezes the Ditto-50 source binding before inference, keeps the VSR and SyncNet
validity sets separate, and joins them only by the audited cohort id.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import math
import os
import pickle
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import vsr_tts_metrics as legacy_metrics
from scripts.experiments.vsr_tts_pilot import (
    AVSR_ROOT,
    BOOTSTRAP_SEED,
    CONDITIONS,
    MODEL_JSON_REL,
    MODEL_REL,
    VIEWS,
    _extract_one_pair,
    _forward_view,
    _independent_ctc_nll,
    _load_feature,
    _load_model_char_list,
    _make_config,
    _set_determinism,
    _stream_summary,
    _view_metrics,
    load_vsr,
)
from scripts.experiments.vsr_ditto50_metrics import (
    CORRELATION_BOOTSTRAP_SEED,
    MIN_JOINT_PAIRS,
    PERMUTATION_SEED,
    association_status,
    bootstrap_mean,
    build_pair_rows,
    freeze_decoys,
    linkage_statistics,
    sync_status,
    visual_status,
)


DITTO_ROOT = ROOT / "runs" / "two_stage_hubert_aishell1_20260810" / "ditto_videos"
PAIRED_MANIFEST = ROOT / "results" / "rhythm_style_500" / "source_manifests" / "aishell1_test_400_paired_audio.json"
NATURAL_MANIFEST = ROOT / "results" / "rhythm_style_500" / "source_manifests" / "aishell1_test_400_natural.json"
TTS_META = ROOT / "results" / "rhythm_style_500" / "aishell1_test_400" / "audit" / "tts_meta.json"
TRANSCRIPT_DIR = ROOT / "results" / "rhythm_style_500" / "aishell1_test_400" / "transcripts"
SYNCNET_ROOT = ROOT / "third_party" / "syncnet_python"
SYNCNET_MODEL = SYNCNET_ROOT / "data" / "syncnet_v2.model"
DEFAULT_RUN = ROOT / "runs" / "vsr_ditto50_linkage_v1"
COHORT_ID = "ditto50_s0765"
PROTOCOL_ID = "vsr_ditto50_linkage_v1"
LEGACY_ROOT = "/mnt/e/Documents/tts-audio/tts-exp/"
SAMPLE_IDS = tuple(range(1, 51))
SYNCNET_PYTHON_DEFAULT = "/home/wjj/.venvs/syncnet/bin/python"
SYNCNET_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
SYNCNET_PATTERN = re.compile(
    r"Confidence:\s*([^\n]+).*?Min\s+dist:\s*([^\n]+).*?AV\s+offset:\s*([^\n]+)",
    re.IGNORECASE | re.DOTALL,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, allow_nan=False) + "\n")


def _resolve_source(raw: Any) -> Path:
    value = str(raw)
    if value.startswith(LEGACY_ROOT):
        candidate = ROOT / value[len(LEGACY_ROOT):]
    elif Path(value).is_absolute():
        candidate = Path(value)
    else:
        candidate = ROOT / value
    return candidate.resolve()


def _read_wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.getnframes()
    if rate <= 0:
        raise ValueError("invalid WAV sample rate: {}".format(path))
    return float(frames) / float(rate)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _load_char_list() -> List[str]:
    return _load_model_char_list(AVSR_ROOT / MODEL_JSON_REL)


def _ffprobe(path: Path) -> Dict[str, Any]:
    command = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return json.loads(result.stdout)


def _audit_video(path: Path) -> Dict[str, Any]:
    payload = _ffprobe(path)
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if video is None:
        raise ValueError("missing video stream: {}".format(path))
    rate = str(video.get("avg_frame_rate") or video.get("r_frame_rate") or "")
    fps = None
    if "/" in rate:
        numerator, denominator = rate.split("/", 1)
        if float(denominator) != 0:
            fps = float(numerator) / float(denominator)
    frame_count = video.get("nb_frames")
    if frame_count in (None, "N/A", ""):
        count_command = ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)]
        count_payload = json.loads(subprocess.run(count_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout)
        frame_count = (count_payload.get("streams") or [{}])[0].get("nb_read_frames")
    duration = float(video.get("duration") or payload.get("format", {}).get("duration") or 0.0)
    audio_duration = float(audio.get("duration") or 0.0) if audio else None
    return {
        "video_codec": video.get("codec_name"),
        "audio_codec": audio.get("codec_name") if audio else None,
        "audio_stream_count": len([item for item in streams if item.get("codec_type") == "audio"]),
        "fps": fps,
        "frame_count": int(frame_count) if frame_count not in (None, "N/A", "") else None,
        "video_start_time": float(video.get("start_time") or 0.0),
        "audio_start_time": float(audio.get("start_time") or 0.0) if audio else None,
        "duration_s": duration,
        "audio_duration_s": audio_duration,
        "width": video.get("width"),
        "height": video.get("height"),
    }


def _environment(run_dir: Path, config_path: Path) -> Dict[str, Any]:
    model_paths = {
        "vsr_model": AVSR_ROOT / MODEL_REL,
        "vsr_model_json": AVSR_ROOT / MODEL_JSON_REL,
        "syncnet_model": SYNCNET_MODEL,
        "config": config_path,
    }
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "avsr_root": str(AVSR_ROOT.resolve()),
        "syncnet_root": str(SYNCNET_ROOT.resolve()),
        "model_paths": {key: str(value.resolve()) for key, value in model_paths.items()},
        "model_sha256": {key: sha256_file(value) for key, value in model_paths.items() if value.is_file()},
        "code_sha256": {
            "runner": sha256_file(Path(__file__)),
            "metrics": sha256_file(Path(__file__).with_name("vsr_ditto50_metrics.py")),
            "legacy_vsr": sha256_file(Path(__file__).with_name("vsr_tts_pilot.py")),
        },
        "seed": BOOTSTRAP_SEED,
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


def _source_document(path: Path) -> Optional[Dict[str, str]]:
    if not path.is_file():
        return None
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _wid_from_paired_key(value: str) -> str:
    match = re.search(r"(BAC\d+S\d+W\d+)$", str(value))
    if match is None:
        raise ValueError("cannot extract WID from paired_key: {}".format(value))
    return match.group(1)


def _audit_record(mapping: Mapping[str, Any], paired: Mapping[str, Any], natural: Mapping[str, Any], tts: Mapping[str, Any], char_list: Sequence[str]) -> Dict[str, Any]:
    sid = int(mapping["id"])
    reasons: List[str] = []
    paired_key = str(mapping.get("paired_key", ""))
    wid = _wid_from_paired_key(paired_key)
    if wid != str(paired.get("source_utterance_id")):
        reasons.append("WID_MISMATCH")
    if str(paired.get("speaker_id")) != str(mapping.get("speaker")):
        reasons.append("SPEAKER_MISMATCH")
    texts = [str(paired.get("transcript", "")), str(natural.get("transcript", "")), str(tts.get("text", ""))]
    transcript_path = TRANSCRIPT_DIR / ("{:04d}.txt".format(int(paired.get("expansion_id", sid))))
    if transcript_path.is_file():
        texts.append(_text_file(transcript_path))
    else:
        reasons.append("MISSING_TRANSCRIPT_FILE")
    normalized = [legacy_metrics.normalize_text(text) for text in texts]
    if not normalized or any(not value for value in normalized) or len(set(normalized)) != 1:
        reasons.append("TEXT_CONFLICT")
    natural_audio = _resolve_source(mapping.get("natural"))
    tts_audio = _resolve_source(mapping.get("tts"))
    natural_video = DITTO_ROOT / "natural_raw" / (str(sid) + ".mp4")
    tts_video = DITTO_ROOT / "tts_raw" / (str(sid) + ".mp4")
    paths = {"natural_audio": natural_audio, "tts_audio": tts_audio, "natural_video": natural_video, "tts_video": tts_video}
    hashes: Dict[str, Optional[str]] = {}
    for name, path in paths.items():
        if not path.is_file():
            reasons.append("MISSING_{}".format(name.upper()))
            hashes[name] = None
        else:
            hashes[name] = sha256_file(path)
    expansion_id = int(paired.get("expansion_id", sid))
    if expansion_id != sid:
        reasons.append("EXPANSION_ID_MAPPING_MISMATCH")
    if int(tts.get("sample_id", expansion_id)) != expansion_id:
        reasons.append("TTS_META_ID_MISMATCH")
    if hashes.get("natural_audio") != paired.get("natural_sha256") or hashes.get("natural_audio") != natural.get("natural_sha256"):
        reasons.append("NATURAL_AUDIO_HASH_MISMATCH")
    if hashes.get("tts_audio") != paired.get("tts_sha256") or hashes.get("tts_audio") != tts.get("tts_sha256_remote"):
        reasons.append("TTS_AUDIO_HASH_MISMATCH")
    try:
        target_tokens = legacy_metrics.token_ids(normalized[0], char_list)
    except Exception as exc:
        target_tokens = None
        reasons.append("TARGET_OOV:{}".format(str(exc)))
    probes: Dict[str, Any] = {}
    for name in ("natural_video", "tts_video"):
        if hashes.get(name) is None:
            continue
        try:
            probe = _audit_video(paths[name])
            probes[name] = probe
            if probe.get("audio_stream_count", 0) < 1:
                reasons.append("MISSING_AUDIO_STREAM_{}".format(name.upper()))
            if probe.get("fps") is None or abs(float(probe["fps"]) - 25.0) > 1e-3:
                reasons.append("FPS_NOT_25_{}".format(name.upper()))
            if probe.get("frame_count") is None or int(probe["frame_count"]) < 5:
                reasons.append("INVALID_FRAME_COUNT_{}".format(name.upper()))
        except Exception as exc:
            reasons.append("VIDEO_PROBE_{}_{}".format(name.upper(), type(exc).__name__))
    for audio_key, video_key in (("natural_audio", "natural_video"), ("tts_audio", "tts_video")):
        if hashes.get(audio_key) is None or video_key not in probes:
            continue
        try:
            audio_duration = _read_wav_duration(paths[audio_key])
            video_probe = probes[video_key]
            if abs(float(video_probe.get("audio_duration_s") or 0.0) - audio_duration) > 0.12:
                reasons.append("AUDIO_VIDEO_SOURCE_DURATION_MISMATCH_{}".format(video_key.upper()))
            if abs(float(video_probe.get("duration_s") or 0.0) - float(video_probe.get("audio_duration_s") or 0.0)) > 0.12:
                reasons.append("VIDEO_AUDIO_DURATION_MISMATCH_{}".format(video_key.upper()))
            if abs(float(video_probe.get("video_start_time") or 0.0) - float(video_probe.get("audio_start_time") or 0.0)) > 0.04:
                reasons.append("VIDEO_AUDIO_START_MISMATCH_{}".format(video_key.upper()))
        except Exception as exc:
            reasons.append("AUDIO_PROBE_{}_{}".format(audio_key.upper(), type(exc).__name__))
    return {
        "id": sid,
        "cohort_id": COHORT_ID,
        "paired_key": paired_key,
        "wid": wid,
        "speaker": str(mapping.get("speaker")),
        "expansion_id": expansion_id,
        "raw_text": texts[0],
        "normalized_text": normalized[0] if normalized else "",
        "target_token_ids": target_tokens,
        "natural_audio": str(natural_audio),
        "tts_audio": str(tts_audio),
        "natural_video": str(natural_video.resolve()),
        "tts_video": str(tts_video.resolve()),
        "sha256": hashes,
        "probe": probes,
        "source_rows": {"paired": dict(paired), "natural": dict(natural), "tts_meta": dict(tts)},
        "audio_provenance": "hash_verified",
        "video_generation_provenance": "historical_documented",
        "eligibility": "eligible" if not reasons and target_tokens is not None else "blocked",
        "reasons": reasons,
    }


def _verify_frozen_inputs(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    for row in manifest.get("records", []):
        for key, expected in row.get("sha256", {}).items():
            if expected is None:
                continue
            path = Path(str(row.get(key)))
            if not path.is_file() or sha256_file(path) != str(expected):
                raise ValueError("frozen input changed: {}".format(path))
    for item in manifest.get("source_documents", {}).values():
        if item is None:
            continue
        path = Path(str(item["path"]))
        if not path.is_file() or sha256_file(path) != str(item["sha256"]):
            raise ValueError("source document changed: {}".format(path))


def audit_ditto50(run_dir: Path, smoke: bool = False, resume: bool = False) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        if not resume:
            raise ValueError("manifest exists; pass --resume")
        manifest = read_json(manifest_path)
        if bool(manifest.get("smoke")) != bool(smoke):
            raise ValueError("smoke/full scope mismatch")
        _verify_frozen_inputs(run_dir, manifest)
        return manifest
    mapping = read_json(DITTO_ROOT / "sample_mapping.json")
    paired_payload = read_json(PAIRED_MANIFEST)
    natural_payload = read_json(NATURAL_MANIFEST)
    tts_payload = read_json(TTS_META)
    paired_rows = {str(row["source_utterance_id"]): row for row in paired_payload.get("records", [])}
    natural_rows = {str(row["source_utterance_id"]): row for row in natural_payload.get("records", [])}
    tts_rows = {int(row["sample_id"]): row for row in tts_payload.get("results", [])}
    char_list = _load_char_list()
    records: List[Dict[str, Any]] = []
    for item in sorted(mapping, key=lambda row: int(row["id"])):
        sid = int(item["id"])
        wid = _wid_from_paired_key(item["paired_key"])
        paired = paired_rows.get(wid, {})
        natural = natural_rows.get(wid, {})
        tts = tts_rows.get(int(paired.get("expansion_id", sid)), {})
        if not paired:
            item_copy = dict(item)
            item_copy["paired_key"] = item["paired_key"]
        record = _audit_record(item, paired, natural, tts, char_list)
        records.append(record)
    valid_text_records = [row for row in records if row["eligibility"] == "eligible" and row.get("target_token_ids") is not None]
    decoys = freeze_decoys(valid_text_records, count=5)
    for row in records:
        row["decoys"] = decoys.get(str(row["id"]), [])
        if len(row["decoys"]) < 5 and row["eligibility"] == "eligible":
            row["eligibility"] = "blocked"
            row["reasons"].append("INSUFFICIENT_OOV_FREE_DECOYS")
    config_path = _make_config(run_dir)
    _environment(run_dir, config_path)
    docs = {
        "sample_mapping": _source_document(DITTO_ROOT / "sample_mapping.json"),
        "paired_manifest": _source_document(PAIRED_MANIFEST),
        "natural_manifest": _source_document(NATURAL_MANIFEST),
        "tts_meta": _source_document(TTS_META),
        "vsr_model_json": _source_document(AVSR_ROOT / MODEL_JSON_REL),
        "syncnet_model": _source_document(SYNCNET_MODEL),
    }
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "cohort_id": COHORT_ID,
        "status": "audited",
        "candidate_count": 50,
        "smoke": bool(smoke),
        "run_sample_ids": [1, 2, 3] if smoke else list(SAMPLE_IDS),
        "seed": BOOTSTRAP_SEED,
        "source_documents": docs,
        "audio_provenance": "hash_verified",
        "video_generation_provenance": "historical_documented",
        "thresholds": {"min_joint_pairs": MIN_JOINT_PAIRS, "detection_fraction": 0.95, "max_missing_run": 5, "track_coverage": 0.95},
        "decoy_rule": "unique normalized source texts, smallest numeric id retained, five nearest lengths then numeric id",
        "records": records,
        "source_fingerprint": canonical_hash(records),
    }
    write_json(manifest_path, manifest)
    return manifest


def _feature_receipt(run_dir: Path, row: Mapping[str, Any]) -> Path:
    return run_dir / "receipts" / "features" / (str(int(row["id"])) + ".json")


def _feature_receipt_valid(run_dir: Path, row: Mapping[str, Any]) -> bool:
    receipt_path = _feature_receipt(run_dir, row)
    if not receipt_path.is_file():
        return False
    try:
        receipt = read_json(receipt_path)
        if receipt.get("status") != "COMPLETE" or receipt.get("protocol_id") != PROTOCOL_ID:
            return False
        for key, expected in row.get("sha256", {}).items():
            if expected is not None and receipt.get("input_sha256", {}).get(key) != expected:
                return False
        for path_value, expected in receipt.get("outputs", {}).items():
            path = Path(path_value)
            if not path.is_file() or sha256_file(path) != expected:
                return False
        return True
    except Exception:
        return False


def _write_feature_receipt(run_dir: Path, row: Mapping[str, Any]) -> None:
    sid = str(int(row["id"]))
    outputs: Dict[str, str] = {}
    for path in sorted((run_dir / "features" / sid).glob("**/*")):
        if path.is_file():
            outputs[str(path.resolve())] = sha256_file(path)
    write_json(_feature_receipt(run_dir, row), {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "id": int(row["id"]),
        "input_sha256": dict(row.get("sha256", {})),
        "outputs": outputs,
    })


def _append_failure(run_dir: Path, stage: str, row_id: Any, error: Exception) -> None:
    append_jsonl(run_dir / "failures.jsonl", {"stage": stage, "id": row_id, "error": "{}: {}".format(type(error).__name__, str(error))})


def extract_ditto50(run_dir: Path, manifest: Mapping[str, Any], device: str = "cuda:0", resume: bool = False) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    _verify_frozen_inputs(run_dir, manifest)
    _set_determinism(BOOTSTRAP_SEED)
    pipeline = load_vsr(run_dir, device=device)
    row_by_id = {int(row["id"]): row for row in manifest.get("records", [])}
    completed = 0
    skipped = 0
    failed = 0
    for sid in manifest.get("run_sample_ids", []):
        row = row_by_id[int(sid)]
        if row.get("eligibility") != "eligible":
            _append_failure(run_dir, "extract", sid, ValueError("source eligibility: {}".format(row.get("reasons"))))
            failed += 1
            continue
        if resume and _feature_receipt_valid(run_dir, row):
            skipped += 1
            continue
        feature_dir = run_dir / "features" / str(int(sid))
        if feature_dir.exists():
            shutil.rmtree(str(feature_dir))
        try:
            _extract_one_pair(pipeline, row, run_dir, device, resume=False)
            qc_paths = [run_dir / "features" / str(int(sid)) / (condition + ".json") for condition in CONDITIONS]
            feature_paths = [run_dir / "features" / str(int(sid)) / condition / (view + ".npz") for condition in CONDITIONS for view in VIEWS]
            if not all(path.is_file() for path in qc_paths + feature_paths):
                raise RuntimeError("incomplete VSR feature set")
            _write_feature_receipt(run_dir, row)
            completed += 1
        except Exception as exc:
            failed += 1
            _append_failure(run_dir, "extract", sid, exc)
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    summary = {"schema_version": 1, "status": "COMPLETE_WITH_EXCLUSIONS" if failed else "COMPLETE", "expected": len(manifest.get("run_sample_ids", [])), "completed": completed, "skipped": skipped, "failed": failed}
    write_json(run_dir / "extract_summary.json", summary)
    return summary


def _parse_sync_value(value: str, kind: str) -> Any:
    value = value.strip().split()[0]
    if kind == "offset":
        if not re.fullmatch(r"[-+]?\d+", value):
            raise ValueError("invalid signed offset: {}".format(value))
        return int(value)
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite SyncNet score")
    return parsed


def parse_syncnet_output(stdout: str) -> Dict[str, Any]:
    """Parse exactly one complete SyncNet Confidence/Min dist/offset group."""

    confidence_matches = re.findall(r"Confidence:\s*(" + SYNCNET_FLOAT + r")", stdout, re.IGNORECASE)
    distance_matches = re.findall(r"Min\s+dist:\s*(" + SYNCNET_FLOAT + r")", stdout, re.IGNORECASE)
    offset_matches = re.findall(r"AV\s+offset:\s*([-+]?\d+)", stdout, re.IGNORECASE)
    if len(confidence_matches) != 1 or len(distance_matches) != 1 or len(offset_matches) != 1:
        raise ValueError("expected exactly one complete SyncNet score group; got C={}, D={}, offset={}".format(len(confidence_matches), len(distance_matches), len(offset_matches)))
    return {
        "sync_c": _parse_sync_value(confidence_matches[0], "score"),
        "sync_d": _parse_sync_value(distance_matches[0], "score"),
        "av_offset": _parse_sync_value(offset_matches[0], "offset"),
    }


def _track_frame_array(track: Mapping[str, Any]) -> np.ndarray:
    # run_pipeline.py stores each crop as {"track": {"frame": ...},
    # "proc_track": ...}; small fixtures and older forks may store the inner
    # track directly.  Accept both while keeping selection independent of the
    # score.
    payload = track.get("track", track)
    values = np.asarray(payload.get("frame", []), dtype=np.int64).reshape(-1)
    if len(values) == 0:
        raise ValueError("empty SyncNet track")
    return values


def select_longest_track(tracks: Sequence[Mapping[str, Any]]) -> Tuple[int, Mapping[str, Any]]:
    if not tracks:
        raise ValueError("no SyncNet face tracks")
    candidates = [(len(_track_frame_array(track)), index, track) for index, track in enumerate(tracks)]
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return int(candidates[0][1]), candidates[0][2]


def _run_subprocess(command: Sequence[str], cwd: Path, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(list(command), cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout, check=False)


def _sync_receipt_valid(run_dir: Path, row: Mapping[str, Any], condition: str) -> bool:
    score_path = run_dir / "syncnet" / str(int(row["id"])) / condition / "score.json"
    if not score_path.is_file():
        return False
    try:
        score = read_json(score_path)
        return score.get("status") == "COMPLETE" and score.get("input_sha256") == row.get("sha256", {}).get(condition + "_video")
    except Exception:
        return False


def _clean_sync_scratch(data_dir: Path, keep_crop: Path) -> None:
    for name in ("pyframes", "pyavi", "pytmp"):
        path = data_dir / name
        if path.exists():
            shutil.rmtree(str(path))
    crop_root = data_dir / "pycrop"
    if crop_root.exists():
        for path in crop_root.glob("**/*"):
            if path.is_file() and path.resolve() != keep_crop.resolve():
                try:
                    path.unlink()
                except OSError:
                    pass


def _run_syncnet_cell(run_dir: Path, row: Mapping[str, Any], condition: str, syncnet_python: str, resume: bool) -> Dict[str, Any]:
    sid = int(row["id"])
    output_dir = run_dir / "syncnet" / str(sid) / condition
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "score.json"
    input_path = Path(str(row[condition + "_video"]))
    expected_input_hash = row.get("sha256", {}).get(condition + "_video")
    if resume and score_path.is_file():
        cached = read_json(score_path)
        if cached.get("status") == "COMPLETE" and cached.get("input_sha256") == expected_input_hash:
            return cached
    data_dir = output_dir / "pipeline_data"
    if data_dir.exists():
        shutil.rmtree(str(data_dir))
    data_dir.mkdir(parents=True, exist_ok=True)
    reference = "ditto50_{}_{}".format(condition, sid)
    pipeline_command = [syncnet_python, str(SYNCNET_ROOT / "run_pipeline.py"), "--videofile", str(input_path), "--reference", reference, "--data_dir", str(data_dir), "--min_track", "50", "--frame_rate", "25", "--facedet_scale", "0.25", "--crop_scale", "0.40", "--num_failed_det", "25", "--min_face_size", "100", "--overwrite"]
    started = time.time()
    pipeline_result = _run_subprocess(pipeline_command, SYNCNET_ROOT, timeout=600)
    (output_dir / "pipeline.stdout.log").write_text(pipeline_result.stdout, encoding="utf-8")
    (output_dir / "pipeline.stderr.log").write_text(pipeline_result.stderr, encoding="utf-8")
    if pipeline_result.returncode != 0:
        score = {"schema_version": 1, "status": "FAILED", "id": sid, "condition": condition, "reason": "RUN_PIPELINE_FAILED", "returncode": pipeline_result.returncode, "input_sha256": expected_input_hash}
        write_json(score_path, score)
        return score
    tracks_path = data_dir / "pywork" / reference / "tracks.pckl"
    try:
        with tracks_path.open("rb") as handle:
            tracks = pickle.load(handle)
        selected_index, selected = select_longest_track(tracks)
        frames = _track_frame_array(selected)
        source_frames = int(_audit_video(input_path).get("frame_count") or 0)
        contiguous = bool(len(frames) > 0 and np.all(np.diff(frames) == 1))
        coverage = float(len(frames) / source_frames) if source_frames else 0.0
        if not contiguous or coverage < 0.95:
            raise ValueError("TRACK_COVERAGE_FAILED contiguous={} coverage={:.6f}".format(contiguous, coverage))
        crop_path = data_dir / "pycrop" / reference / ("{:05d}.avi".format(selected_index))
        if not crop_path.is_file():
            raise FileNotFoundError(str(crop_path))
        crop_saved = output_dir / "crop.avi"
        shutil.copyfile(str(crop_path), str(crop_saved))
        track_payload = {"selected_index": selected_index, "frame": [int(value) for value in frames], "track_count": len(tracks), "source_frame_count": source_frames, "coverage": coverage, "contiguous": contiguous}
        write_json(output_dir / "track.json", track_payload)
        demo_reference = "demo_{}_{}".format(condition, sid)
        demo_tmp = output_dir / "demo_tmp"
        demo_command = [syncnet_python, str(SYNCNET_ROOT / "demo_syncnet.py"), "--videofile", str(crop_saved), "--initial_model", str(SYNCNET_MODEL), "--vshift", "15", "--batch_size", "20", "--tmp_dir", str(demo_tmp), "--reference", demo_reference]
        demo_result = _run_subprocess(demo_command, SYNCNET_ROOT, timeout=600)
        (output_dir / "demo.stdout.log").write_text(demo_result.stdout, encoding="utf-8")
        (output_dir / "demo.stderr.log").write_text(demo_result.stderr, encoding="utf-8")
        combined = demo_result.stdout + "\n" + demo_result.stderr
        if demo_result.returncode != 0:
            raise RuntimeError("demo_syncnet returncode {}".format(demo_result.returncode))
        parsed = parse_syncnet_output(combined)
        score = {
            "schema_version": 1,
            "status": "COMPLETE",
            "id": sid,
            "condition": condition,
            "sync_c": parsed["sync_c"],
            "sync_d": parsed["sync_d"],
            "av_offset": parsed["av_offset"],
            "input": str(input_path.resolve()),
            "input_sha256": expected_input_hash,
            "model": str(SYNCNET_MODEL.resolve()),
            "model_sha256": sha256_file(SYNCNET_MODEL),
            "track": str((output_dir / "track.json").resolve()),
            "track_sha256": sha256_file(output_dir / "track.json"),
            "crop": str(crop_saved.resolve()),
            "crop_sha256": sha256_file(crop_saved),
            "track_qc": True,
            "track_coverage": coverage,
            "selected_track_index": selected_index,
            "commands": {"pipeline": pipeline_command, "demo": demo_command},
            "elapsed_s": float(time.time() - started),
        }
        write_json(score_path, score)
        _clean_sync_scratch(data_dir, crop_path)
        return score
    except Exception as exc:
        score = {"schema_version": 1, "status": "FAILED", "id": sid, "condition": condition, "reason": "{}: {}".format(type(exc).__name__, str(exc)), "input_sha256": expected_input_hash, "track_qc": False}
        write_json(score_path, score)
        return score


def score_syncnet(run_dir: Path, manifest: Mapping[str, Any], syncnet_python: str = SYNCNET_PYTHON_DEFAULT, resume: bool = False) -> Dict[str, Any]:
    _verify_frozen_inputs(run_dir, manifest)
    records: List[Dict[str, Any]] = []
    for sid in manifest.get("run_sample_ids", []):
        row = next(item for item in manifest["records"] if int(item["id"]) == int(sid))
        if row.get("eligibility") != "eligible":
            continue
        for condition in CONDITIONS:
            result = _run_syncnet_cell(run_dir, row, condition, syncnet_python, resume)
            records.append(result)
    eligible_count = sum(1 for row in manifest.get("records", []) if int(row["id"]) in {int(value) for value in manifest.get("run_sample_ids", [])} and row.get("eligibility") == "eligible")
    expected_cells = eligible_count * 2
    write_json(run_dir / "syncnet_records.json", {"schema_version": 1, "records": records, "expected_cells": expected_cells})
    complete = sum(item.get("status") == "COMPLETE" for item in records)
    failed = len(records) - complete
    summary = {"schema_version": 1, "status": "COMPLETE_WITH_EXCLUSIONS" if failed else "COMPLETE", "expected": expected_cells, "completed": complete, "failed": failed, "source_excluded": len(manifest.get("run_sample_ids", [])) - eligible_count}
    write_json(run_dir / "syncnet_summary.json", summary)
    return summary


def _build_view_rows(run_dir: Path, manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    row_by_id = {int(row["id"]): row for row in manifest.get("records", [])}
    for sid in manifest.get("run_sample_ids", []):
        source = row_by_id[int(sid)]
        for condition in CONDITIONS:
            qc_path = run_dir / "features" / str(int(sid)) / (condition + ".json")
            qc = read_json(qc_path) if qc_path.is_file() else {}
            qc_eligible = bool(qc.get("eligible_for_pair", False)) and _finite(qc.get("native_repeat_max_abs")) and _finite(qc.get("native_repeat_encoder_max_abs")) and float(qc.get("native_repeat_max_abs")) <= 1e-5 and float(qc.get("native_repeat_encoder_max_abs")) <= 1e-5
            for view in VIEWS:
                feature_path = run_dir / "features" / str(int(sid)) / condition / (view + ".npz")
                record: Dict[str, Any] = {"id": int(sid), "condition": condition, "view": view, "feature": str(feature_path.resolve()), "qc_eligible": qc_eligible, "status": "FAILED"}
                try:
                    if source.get("eligibility") != "eligible":
                        raise ValueError("source audit blocked: {}".format(source.get("reasons")))
                    feature = _load_feature(feature_path)
                    values = _view_metrics(feature["logp"], source["target_token_ids"], source["decoys"])
                    record.update(values)
                    record["source_sha256"] = dict(source.get("sha256", {}))
                    record["duration_s"] = qc.get("duration_s")
                    record["detection_fraction"] = qc.get("raw_landmark_valid_fraction")
                    record["native_repeat_max_abs"] = qc.get("native_repeat_max_abs")
                    record["native_repeat_encoder_max_abs"] = qc.get("native_repeat_encoder_max_abs")
                except Exception as exc:
                    record["error"] = "{}: {}".format(type(exc).__name__, str(exc))
                rows.append(record)
    return rows


def _load_sync_rows(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "syncnet_records.json"
    if not path.is_file():
        return []
    payload = read_json(path)
    return [dict(item) for item in payload.get("records", [])]


def _calibration(view_rows: Sequence[Mapping[str, Any]], pair_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ids = {int(row["id"]) for row in pair_rows}
    result: Dict[str, Any] = {"status": "UNCALIBRATED", "joint_count": len(ids), "arms": {}}
    for condition in CONDITIONS:
        native = [row for row in view_rows if int(row["id"]) in ids and row.get("condition") == condition and row.get("view") == "native" and row.get("status") == "COMPLETE" and row.get("qc_eligible") is True]
        q_values: List[float] = []
        for item in native:
            frozen = next((row for row in view_rows if int(row["id"]) == int(item["id"]) and row.get("condition") == condition and row.get("view") == "frozen" and row.get("status") == "COMPLETE" and row.get("qc_eligible") is True), None)
            if frozen is not None:
                q_values.append(float(item["M"]) - float(frozen["M"]))
        top1 = [bool(row.get("strict_target_top1")) for row in native]
        result["arms"][condition] = {
            "native_count": len(native),
            "strict_native_top1_count": int(sum(top1)),
            "strict_native_top1_fraction": float(np.mean(top1)) if top1 else 0.0,
            "q_count": len(q_values),
            "q_positive_count": int(sum(value > 0.0 for value in q_values)),
            "q_positive_fraction": float(np.mean(np.asarray(q_values) > 0.0)) if q_values else 0.0,
            "q_mean": float(np.mean(q_values)) if q_values else None,
        }
    okay = all(
        details["strict_native_top1_fraction"] >= 0.60 and details["q_positive_fraction"] >= 0.75 and details["q_mean"] is not None and details["q_mean"] > 0.0
        for details in result["arms"].values()
    ) and len(ids) >= MIN_JOINT_PAIRS
    result["status"] = "CALIBRATED_ON_JOINT_COHORT" if okay else "INCONCLUSIVE_VSR_VALIDITY"
    return result


def _draw_scatter(run_dir: Path, pair_rows: Sequence[Mapping[str, Any]], linkage: Mapping[str, Any]) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [float(row["g"]) for row in pair_rows]
        ys = [float(row["delta_c"]) for row in pair_rows]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(xs, ys, s=28, alpha=0.8)
        for row in pair_rows:
            ax.annotate(str(row["id"]), (float(row["g"]), float(row["delta_c"])), fontsize=6, alpha=0.7)
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.axvline(0.0, color="black", linewidth=0.7)
        rho = linkage.get("primary_association", {}).get("spearman", {}).get("rho")
        ax.set_xlabel("VSR dynamic gain G")
        ax.set_ylabel("Sync-C gain ΔC")
        ax.set_title("Ditto-50 linkage (n={}, Spearman={})".format(len(pair_rows), "null" if rho is None else "{:.3f}".format(float(rho))))
        fig.tight_layout()
        destination = run_dir / "linkage_scatter.png"
        fig.savefig(str(destination), dpi=160)
        plt.close(fig)
        return str(destination.resolve())
    except Exception as exc:
        write_json(run_dir / "scatter_error.json", {"error": repr(exc)})
        return None


def _summary_value(linkage: Mapping[str, Any], metric: str) -> Mapping[str, Any]:
    return linkage.get("metrics", {}).get(metric, {})


def _run_sync_status(manifest: Mapping[str, Any], linkage: Mapping[str, Any]) -> str:
    """Keep smoke output technical-only; classify Sync-C only for formal runs."""

    if bool(manifest.get("smoke")):
        return "SMOKE_TECHNICAL_ONLY"
    return sync_status(linkage.get("metrics", {}).get("delta_c", {}))


def _write_report(run_dir: Path, analysis: Mapping[str, Any]) -> None:
    linkage = analysis.get("linkage", {})
    lines = [
        "# Ditto-50 VSR 与 Sync-C 增益关联验证",
        "",
        "本报告使用同一批 Ditto-50 自然/TTS 视频，VSR 只读视频帧，SyncNet 使用原始 MP4 内嵌音轨。结果是单说话人 S0765 的历史 cohort 上的探索性测量，不是因果检验。",
        "",
        "- 分母：candidate={}，source eligible={}，V={}，S={}，J={}".format(analysis.get("candidate_count"), analysis.get("source_eligible_count"), analysis.get("visual_count"), analysis.get("sync_count"), analysis.get("joint_count")),
        "- engineering_status：{}".format(analysis.get("engineering_status")),
        "- calibration_status：{}".format(analysis.get("calibration_status")),
        "- sync_status：{}".format(analysis.get("sync_status")),
        "- visual_status：{}".format(analysis.get("visual_status")),
        "- association_status：{}".format(analysis.get("association_status")),
        "",
        "| metric | n | mean | 95% CI | positive / zero / negative |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in ("g", "b", "gmatched", "r_natural", "r_tts", "delta_c", "delta_d", "delta_cer"):
        item = _summary_value(linkage, metric)
        if item.get("status") == "COMPLETE":
            lines.append("| {} | {} | {:.6f} | [{:.6f}, {:.6f}] | {} / {} / {} |".format(metric, item["count"], item["mean"], item["ci95"][0], item["ci95"][1], item["positive_count"], item["zero_count"], item["negative_count"]))
        else:
            lines.append("| {} | {} | — | — | — |".format(metric, item.get("count", 0)))
    assoc = linkage.get("primary_association", {}).get("spearman", {})
    lines.extend(["", "主要关联：Spearman(G, ΔC) = {}，置换 p = {}，相关 bootstrap CI = {}".format(assoc.get("rho"), assoc.get("p_two_sided"), linkage.get("primary_association", {}).get("bootstrap", {}).get("ci95")), ""])
    table = linkage.get("contingency", {})
    lines.extend(["| | ΔC>0 | ΔC≤0 |", "|---|---:|---:|", "| G>0 | {} | {} |".format(table.get("rows", {}).get("G>0", {}).get("deltaC>0", 0), table.get("rows", {}).get("G>0", {}).get("deltaC<=0", 0)), "| G≤0 | {} | {} |".format(table.get("rows", {}).get("G<=0", {}).get("deltaC>0", 0), table.get("rows", {}).get("G<=0", {}).get("deltaC<=0", 0)), "", "交集同时为正的样本数：{}；独立假设下期望交集：{}".format(table.get("both_positive_count"), table.get("expected_both_under_independence")), ""])
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_ditto50(run_dir: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    view_rows = _build_view_rows(run_dir, manifest)
    records_path = run_dir / "records.jsonl"
    records_path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in view_rows), encoding="utf-8")
    sync_rows = _load_sync_rows(run_dir)
    run_sample_ids = {int(value) for value in manifest.get("run_sample_ids", [])}
    source_ids = {int(row["id"]) for row in manifest.get("records", []) if int(row["id"]) in run_sample_ids and row.get("eligibility") == "eligible"}
    by_key = {(int(row["id"]), str(row["condition"]), str(row["view"])): row for row in view_rows}
    visual_ids = set()
    for sid in source_ids:
        if all((sid, condition, view) in by_key and by_key[(sid, condition, view)].get("status") == "COMPLETE" and by_key[(sid, condition, view)].get("qc_eligible") is True for condition in CONDITIONS for view in VIEWS):
            visual_ids.add(sid)
    sync_by_key = {(int(row.get("id", -1)), str(row.get("condition"))): row for row in sync_rows}
    sync_ids = {sid for sid in source_ids if all(sync_by_key.get((sid, condition), {}).get("status") == "COMPLETE" and sync_by_key.get((sid, condition), {}).get("track_qc") is True for condition in CONDITIONS)}
    pair_rows = build_pair_rows(view_rows, sync_rows)
    pair_rows = [row for row in pair_rows if int(row["id"]) in visual_ids and int(row["id"]) in sync_ids]
    write_json(run_dir / "pair_rows.json", {"schema_version": 1, "ids": [int(row["id"]) for row in pair_rows], "rows": pair_rows})
    linkage = linkage_statistics(pair_rows)
    calibration = _calibration(view_rows, pair_rows)
    engineering = "PENDING_INDEPENDENT_VALIDATION"
    g_summary = linkage.get("metrics", {}).get("g", {})
    b_summary = linkage.get("metrics", {}).get("b", {})
    gm_summary = linkage.get("metrics", {}).get("gmatched", {})
    visual = visual_status(engineering_status=engineering, calibration_status=calibration["status"], joint_count=len(pair_rows), g_summary=g_summary, b_summary=b_summary, matched_summary=gm_summary)
    assoc = linkage.get("primary_association", {}).get("spearman", {})
    association = association_status(engineering, len(pair_rows), calibration["status"], {"rho": assoc.get("rho"), "p_two_sided": assoc.get("p_two_sided"), "ci95": linkage.get("primary_association", {}).get("bootstrap", {}).get("ci95")})
    paired_sync_status = _run_sync_status(manifest, linkage)
    analysis = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "candidate_count": len(manifest.get("records", [])),
        "source_eligible_count": len(source_ids),
        "visual_count": len(visual_ids),
        "sync_count": len(sync_ids),
        "joint_count": len(pair_rows),
        "visual_ids": sorted(visual_ids),
        "sync_ids": sorted(sync_ids),
        "joint_ids": sorted(int(row["id"]) for row in pair_rows),
        "calibration": calibration,
        "linkage": linkage,
        "engineering_status": engineering,
        "calibration_status": calibration["status"],
        "sync_status": paired_sync_status,
        "visual_status": visual,
        "association_status": association,
        "scope": {"cohort_id": COHORT_ID, "speaker": "S0765", "historical_validation_cohort": True, "causal": False},
    }
    _draw_scatter(run_dir, pair_rows, linkage)
    write_json(run_dir / "analysis.json", analysis)
    _write_report(run_dir, analysis)
    return analysis


def _independent_rank(values: Sequence[float]) -> List[float]:
    ordered = sorted((float(value), index) for index, value in enumerate(values))
    output = [0.0] * len(ordered)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][0] == ordered[cursor][0]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for index in range(cursor, end):
            output[ordered[index][1]] = average
        cursor = end
    return output


def _independent_spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x = np.asarray(_independent_rank(xs), dtype=np.float64)
    y = np.asarray(_independent_rank(ys), dtype=np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    return None if denominator <= 1e-15 else float(np.dot(x, y) / denominator)


def _independent_mean_summary(values: Sequence[float], metric_seed: int = BOOTSTRAP_SEED) -> Dict[str, Any]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if len(array) == 0:
        return {"status": "NOT_ESTIMABLE", "count": 0}
    rng = np.random.default_rng(metric_seed)
    indices = rng.integers(0, len(array), size=(20_000, len(array)), dtype=np.int64)
    means = array[indices].mean(axis=1, dtype=np.float64)
    try:
        low = float(np.quantile(means, 0.025, method="linear"))
        high = float(np.quantile(means, 0.975, method="linear"))
    except TypeError:
        low = float(np.quantile(means, 0.025, interpolation="linear"))
        high = float(np.quantile(means, 0.975, interpolation="linear"))
    return {
        "status": "COMPLETE",
        "count": len(array),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci95": [low, high],
        "positive_count": int(np.sum(array > 0.0)),
        "zero_count": int(np.sum(array == 0.0)),
        "negative_count": int(np.sum(array < 0.0)),
        "positive_fraction": float(np.mean(array > 0.0)),
        "values": [float(value) for value in array],
    }


def _independent_linkage_statistics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ordered = sorted(rows, key=lambda row: int(row["id"]))
    metrics = ("g", "b", "gmatched", "r_natural", "r_tts", "delta_c", "delta_d", "delta_cer")
    summaries = {metric: _independent_mean_summary([float(row[metric]) for row in ordered]) for metric in metrics}
    g = [float(row["g"]) for row in ordered]
    c = [float(row["delta_c"]) for row in ordered]
    rho = _independent_spearman(g, c)
    if rho is None:
        association = {"status": "NOT_ESTIMABLE", "rho": None, "ci95": None, "p_two_sided": None}
    else:
        rng = np.random.default_rng(PERMUTATION_SEED)
        exceed = 0
        for _ in range(20_000):
            value = _independent_spearman(g, rng.permutation(np.asarray(c)).tolist())
            if value is not None and abs(value) >= abs(rho) - 1e-12:
                exceed += 1
        rng = np.random.default_rng(CORRELATION_BOOTSTRAP_SEED)
        bootstrap_values: List[float] = []
        for _ in range(20_000):
            indices = rng.integers(0, len(g), size=len(g), dtype=np.int64)
            value = _independent_spearman([g[index] for index in indices], [c[index] for index in indices])
            if value is not None:
                bootstrap_values.append(float(value))
        try:
            ci = [float(np.quantile(np.asarray(bootstrap_values), 0.025, method="linear")), float(np.quantile(np.asarray(bootstrap_values), 0.975, method="linear"))]
        except TypeError:
            ci = [float(np.quantile(np.asarray(bootstrap_values), 0.025, interpolation="linear")), float(np.quantile(np.asarray(bootstrap_values), 0.975, interpolation="linear"))]
        association = {"status": "COMPLETE", "rho": float(rho), "p_two_sided": float((1 + exceed) / 20_001.0), "ci95": ci}
    g_positive = [float(row["g"]) > 0.0 for row in ordered]
    c_positive = [float(row["delta_c"]) > 0.0 for row in ordered]
    g_count = int(sum(g_positive))
    c_count = int(sum(c_positive))
    both_count = int(sum(left and right for left, right in zip(g_positive, c_positive)))
    n = len(ordered)
    contingency = {
        "n": n,
        "g_positive_count": g_count,
        "delta_c_positive_count": c_count,
        "both_positive_count": both_count,
        "only_vsr_positive_count": int(sum(left and not right for left, right in zip(g_positive, c_positive))),
        "only_sync_positive_count": int(sum((not left) and right for left, right in zip(g_positive, c_positive))),
        "both_nonpositive_count": int(sum((not left) and (not right) for left, right in zip(g_positive, c_positive))),
        "expected_both_under_independence": float(n * g_count / n * c_count / n) if n else None,
    }
    return {"count": len(ordered), "metrics": summaries, "association": association, "contingency": contingency}


def _rebuild_view_rows_independent(run_dir: Path, manifest: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], int, float, List[str]]:
    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    checked_cells = 0
    max_loss_error = 0.0
    source_by_id = {int(row["id"]): row for row in manifest.get("records", [])}
    producer_rows: Dict[Tuple[int, str, str], Mapping[str, Any]] = {}
    records_path = run_dir / "records.jsonl"
    if records_path.is_file():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                producer_rows[(int(item["id"]), str(item["condition"]), str(item["view"]))] = item
    for sid in manifest.get("run_sample_ids", []):
        source = source_by_id[int(sid)]
        for condition in CONDITIONS:
            qc_path = run_dir / "features" / str(int(sid)) / (condition + ".json")
            qc = read_json(qc_path) if qc_path.is_file() else {}
            qc_eligible = bool(qc.get("eligible_for_pair", False)) and _finite(qc.get("native_repeat_max_abs")) and _finite(qc.get("native_repeat_encoder_max_abs")) and float(qc.get("native_repeat_max_abs")) <= 1e-5 and float(qc.get("native_repeat_encoder_max_abs")) <= 1e-5
            for view in VIEWS:
                feature_path = run_dir / "features" / str(int(sid)) / condition / (view + ".npz")
                row: Dict[str, Any] = {"id": int(sid), "condition": condition, "view": view, "feature": str(feature_path.resolve()), "qc_eligible": qc_eligible, "status": "FAILED"}
                try:
                    feature = _load_feature(feature_path)
                    logp = np.asarray(feature["logp"], dtype=np.float64)
                    target_loss = _independent_ctc_nll(logp, source["target_token_ids"])
                    decoy_losses: Dict[str, float] = {}
                    for decoy in source["decoys"]:
                        decoy_losses[str(decoy["id"])] = _independent_ctc_nll(logp, decoy["token_ids"])
                    margin = float(np.mean(list(decoy_losses.values())) - target_loss)
                    rank = 1 + sum(value < target_loss for value in decoy_losses.values())
                    row.update({"status": "COMPLETE", "target_loss": float(target_loss), "decoy_losses": {"target": float(target_loss), **decoy_losses}, "M": margin, "target_rank": rank, "strict_target_top1": bool(rank == 1 and all(value != target_loss for value in decoy_losses.values())), "greedy_cer": float(legacy_metrics.greedy_cer(logp, source["target_token_ids"]))})
                    producer = producer_rows.get((int(sid), condition, view))
                    if producer is not None:
                        for key, expected in row["decoy_losses"].items():
                            actual = producer.get("decoy_losses", {}).get(key)
                            if actual is None:
                                failures.append("MISSING_PRODUCER_LOSS:{}:{}:{}".format(sid, condition, view))
                            else:
                                error = abs(float(actual) - float(expected))
                                max_loss_error = max(max_loss_error, error)
                                checked_cells += 1
                                if error > 1e-5:
                                    failures.append("CTC:{}:{}:{}:{}".format(sid, condition, view, key))
                    probability_error = float(np.max(np.abs(np.exp(logp).sum(axis=1) - 1.0)))
                    if probability_error > 1e-5:
                        failures.append("LOGP_NORMALIZATION:{}:{}:{}".format(sid, condition, view))
                except Exception as exc:
                    row["error"] = "{}: {}".format(type(exc).__name__, str(exc))
                rows.append(row)
    return rows, checked_cells, max_loss_error, failures


def _independent_pair_rows(view_rows: Sequence[Mapping[str, Any]], sync_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {(int(row["id"]), str(row["condition"]), str(row["view"])): row for row in view_rows if row.get("status") == "COMPLETE" and row.get("qc_eligible") is True}
    sync_by_key = {(int(row["id"]), str(row["condition"])): row for row in sync_rows if row.get("status") == "COMPLETE" and row.get("track_qc") is True}
    output: List[Dict[str, Any]] = []
    for sid in sorted({int(row["id"]) for row in view_rows}):
        if any((sid, condition, view) not in by_key for condition in CONDITIONS for view in VIEWS) or any((sid, condition) not in sync_by_key for condition in CONDITIONS):
            continue
        n = {view: by_key[(sid, "natural", view)] for view in VIEWS}
        t = {view: by_key[(sid, "tts", view)] for view in VIEWS}
        ns, ts = sync_by_key[(sid, "natural")], sync_by_key[(sid, "tts")]
        output.append({"id": sid, "g": (t["native"]["M"] - t["frozen"]["M"]) - (n["native"]["M"] - n["frozen"]["M"]), "b": t["native"]["M"] - n["native"]["M"], "gmatched": (t["matched"]["M"] - t["matched_frozen"]["M"]) - (n["matched"]["M"] - n["matched_frozen"]["M"]), "r_natural": n["native"]["M"] - n["reversed"]["M"], "r_tts": t["native"]["M"] - t["reversed"]["M"], "delta_c": float(ts["sync_c"] - ns["sync_c"]), "delta_d": float(ns["sync_d"] - ts["sync_d"]), "delta_cer": n["native"].get("greedy_cer", 0.0) - t["native"].get("greedy_cer", 0.0), "natural_q": n["native"]["M"] - n["frozen"]["M"], "tts_q": t["native"]["M"] - t["frozen"]["M"]})
    return output


def _compare_float(left: Any, right: Any, tolerance: float) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return left == right


def validate_ditto50(run_dir: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    failures: List[str] = []
    _verify_frozen_inputs(run_dir, manifest)
    independent_views, checked_cells, max_loss_error, loss_failures = _rebuild_view_rows_independent(run_dir, manifest)
    failures.extend(loss_failures)
    sync_rows = _load_sync_rows(run_dir)
    eligible_ids = {
        int(row["id"])
        for row in manifest.get("records", [])
        if int(row["id"]) in {int(value) for value in manifest.get("run_sample_ids", [])} and row.get("eligibility") == "eligible"
    }
    expected_sync_keys = {(sid, condition) for sid in eligible_ids for condition in CONDITIONS}
    actual_sync_keys = {(int(row.get("id", -1)), str(row.get("condition"))) for row in sync_rows}
    for sid, condition in sorted(expected_sync_keys - actual_sync_keys):
        failures.append("MISSING_SYNC_CELL:{}:{}".format(sid, condition))
    for row in sync_rows:
        if (int(row.get("id", -1)), str(row.get("condition"))) in expected_sync_keys and row.get("status") != "COMPLETE":
            failures.append("SYNC_CELL_NOT_COMPLETE:{}:{}".format(row.get("id"), row.get("condition")))
    for score in sync_rows:
        if score.get("status") != "COMPLETE":
            continue
        stdout_path = run_dir / "syncnet" / str(int(score["id"])) / str(score["condition"]) / "demo.stdout.log"
        stderr_path = run_dir / "syncnet" / str(int(score["id"])) / str(score["condition"]) / "demo.stderr.log"
        try:
            parsed = parse_syncnet_output(stdout_path.read_text(encoding="utf-8") + "\n" + stderr_path.read_text(encoding="utf-8"))
            if not _compare_float(parsed["sync_c"], score.get("sync_c"), 1e-8) or not _compare_float(parsed["sync_d"], score.get("sync_d"), 1e-8) or parsed["av_offset"] != score.get("av_offset"):
                failures.append("SYNC_LOG_SCORE_MISMATCH:{}:{}".format(score.get("id"), score.get("condition")))
        except Exception as exc:
            failures.append("SYNC_LOG_PARSE:{}:{}:{}".format(score.get("id"), score.get("condition"), type(exc).__name__))
        input_path = Path(str(score.get("input", "")))
        if not input_path.is_file() or sha256_file(input_path) != score.get("input_sha256"):
            failures.append("SYNC_INPUT_HASH:{}:{}".format(score.get("id"), score.get("condition")))
        for key in ("track", "crop"):
            path = Path(str(score.get(key, "")))
            expected = score.get(key + "_sha256")
            if not path.is_file() or not expected or sha256_file(path) != expected:
                failures.append("SYNC_ARTIFACT_HASH:{}:{}:{}".format(score.get("id"), score.get("condition"), key))
    independent_pairs = _independent_pair_rows(independent_views, sync_rows)
    pair_path = run_dir / "pair_rows.json"
    producer_pairs = read_json(pair_path).get("rows", []) if pair_path.is_file() else []
    if [int(row["id"]) for row in independent_pairs] != [int(row["id"]) for row in producer_pairs]:
        failures.append("PAIR_ID_SET_MISMATCH")
    producer_by_id = {int(row["id"]): row for row in producer_pairs}
    for row in independent_pairs:
        producer = producer_by_id.get(int(row["id"]))
        if producer is None:
            continue
        for key in ("g", "b", "gmatched", "r_natural", "r_tts", "delta_c", "delta_d", "delta_cer"):
            if not _compare_float(row.get(key), producer.get(key), 1e-8):
                failures.append("PAIR_VALUE_MISMATCH:{}:{}".format(row["id"], key))
    independent_view_by_key = {(int(row["id"]), str(row["condition"]), str(row["view"])): row for row in independent_views}
    for sid in sorted(eligible_ids):
        for condition in CONDITIONS:
            for view in VIEWS:
                key = (sid, condition, view)
                item = independent_view_by_key.get(key)
                if item is None:
                    failures.append("MISSING_VIEW_CELL:{}:{}:{}".format(sid, condition, view))
                elif item.get("status") != "COMPLETE":
                    failures.append("VIEW_CELL_NOT_COMPLETE:{}:{}:{}".format(sid, condition, view))
    independent_linkage = _independent_linkage_statistics(independent_pairs)
    analysis_path = run_dir / "analysis.json"
    analysis = read_json(analysis_path) if analysis_path.is_file() else {}
    producer_linkage = analysis.get("linkage", {})
    statistics_max_error = 0.0
    for metric, expected in independent_linkage.get("metrics", {}).items():
        actual = producer_linkage.get("metrics", {}).get(metric, {})
        for key in ("count", "mean", "median", "positive_fraction"):
            if key in expected:
                try:
                    statistics_max_error = max(statistics_max_error, abs(float(expected.get(key)) - float(actual.get(key))))
                except (TypeError, ValueError):
                    pass
            if key in expected and not _compare_float(expected.get(key), actual.get(key), 1e-8):
                failures.append("STAT_MISMATCH:{}:{}".format(metric, key))
        for index, value in enumerate(expected.get("ci95", [])):
            if index < len(actual.get("ci95", [])):
                statistics_max_error = max(statistics_max_error, abs(float(value) - float(actual["ci95"][index])))
            if index >= len(actual.get("ci95", [])) or not _compare_float(value, actual["ci95"][index], 1e-8):
                failures.append("STAT_CI_MISMATCH:{}:{}".format(metric, index))
    expected_assoc = independent_linkage.get("association", {})
    actual_assoc = producer_linkage.get("primary_association", {}).get("spearman", {})
    for key in ("rho", "p_two_sided"):
        if expected_assoc.get(key) is not None:
            statistics_max_error = max(statistics_max_error, abs(float(expected_assoc.get(key)) - float(actual_assoc.get(key))))
        if expected_assoc.get(key) is not None and not _compare_float(expected_assoc.get(key), actual_assoc.get(key), 1e-8):
            failures.append("ASSOCIATION_MISMATCH:{}".format(key))
    expected_contingency = independent_linkage.get("contingency", {})
    actual_contingency = producer_linkage.get("contingency", {})
    for key in ("n", "g_positive_count", "delta_c_positive_count", "both_positive_count", "only_vsr_positive_count", "only_sync_positive_count", "both_nonpositive_count"):
        if expected_contingency.get(key) != actual_contingency.get(key):
            failures.append("CONTINGENCY_MISMATCH:{}".format(key))
    if expected_contingency.get("expected_both_under_independence") is not None:
        if not _compare_float(expected_contingency.get("expected_both_under_independence"), actual_contingency.get("expected_both_under_independence"), 1e-8):
            failures.append("CONTINGENCY_MISMATCH:expected_both_under_independence")
    for sid in manifest.get("run_sample_ids", []):
        source_for_qc = next(item for item in manifest["records"] if int(item["id"]) == int(sid))
        if source_for_qc.get("eligibility") != "eligible":
            continue
        for condition in CONDITIONS:
            qc_path = run_dir / "features" / str(int(sid)) / (condition + ".json")
            if not qc_path.is_file():
                failures.append("MISSING_QC:{}:{}".format(sid, condition))
                continue
            qc = read_json(qc_path)
            if not qc.get("frame_digest_equal") or not qc.get("pts_equal"):
                failures.append("AUDIO_ISOLATION:{}:{}".format(sid, condition))
            if qc.get("native_repeat_max_abs") is None or qc.get("native_repeat_encoder_max_abs") is None:
                failures.append("MISSING_REPEAT_CHECK:{}:{}".format(sid, condition))
            elif float(qc["native_repeat_max_abs"]) > 1e-5 or float(qc["native_repeat_encoder_max_abs"]) > 1e-5:
                failures.append("NONDETERMINISTIC:{}:{}".format(sid, condition))
            if not _feature_receipt_valid(run_dir, source_for_qc):
                failures.append("FEATURE_RECEIPT:{}".format(sid))
    independent_calibration = _calibration(independent_views, independent_pairs)
    preliminary_engineering = "PASS" if not failures and max_loss_error <= 1e-5 and statistics_max_error <= 1e-8 else "FAIL"
    independent_metrics = independent_linkage.get("metrics", {})
    expected_sync_status = _run_sync_status(manifest, {"metrics": independent_metrics})
    expected_visual_status = visual_status(
        engineering_status=preliminary_engineering,
        calibration_status=independent_calibration["status"],
        joint_count=len(independent_pairs),
        g_summary=independent_metrics.get("g", {}),
        b_summary=independent_metrics.get("b", {}),
        matched_summary=independent_metrics.get("gmatched", {}),
    )
    independent_association = independent_linkage.get("association", {})
    expected_association_status = association_status(
        preliminary_engineering,
        len(independent_pairs),
        independent_calibration["status"],
        {"rho": independent_association.get("rho"), "p_two_sided": independent_association.get("p_two_sided"), "ci95": independent_association.get("ci95")},
    )
    expected_statuses = {
        "calibration_status": independent_calibration["status"],
        "sync_status": expected_sync_status,
        "visual_status": expected_visual_status,
        "association_status": expected_association_status,
    }
    # analyze writes a provisional engineering status before this independent
    # pass.  In that state its visual label is intentionally technical-only;
    # after a prior validation, all status fields must agree exactly.
    if analysis.get("engineering_status") != "PENDING_INDEPENDENT_VALIDATION":
        for key, expected in expected_statuses.items():
            if analysis.get(key) != expected:
                failures.append("STATUS_MISMATCH:{}".format(key))
    producer_calibration = analysis.get("calibration", {})
    for condition in CONDITIONS:
        expected_arm = independent_calibration.get("arms", {}).get(condition, {})
        actual_arm = producer_calibration.get("arms", {}).get(condition, {})
        for key in ("native_count", "strict_native_top1_count", "q_count", "q_positive_count"):
            if expected_arm.get(key) != actual_arm.get(key):
                failures.append("CALIBRATION_MISMATCH:{}:{}".format(condition, key))
        for key in ("strict_native_top1_fraction", "q_positive_fraction", "q_mean"):
            if expected_arm.get(key) is None:
                if actual_arm.get(key) is not None:
                    failures.append("CALIBRATION_MISMATCH:{}:{}".format(condition, key))
            elif not _compare_float(expected_arm.get(key), actual_arm.get(key), 1e-8):
                failures.append("CALIBRATION_MISMATCH:{}:{}".format(condition, key))
    expected_views = len(manifest.get("run_sample_ids", [])) * 10
    completed_views = sum(row.get("status") == "COMPLETE" for row in independent_views)
    failed_views = max(0, expected_views - completed_views)
    sync_manifest = read_json(run_dir / "syncnet_records.json") if (run_dir / "syncnet_records.json").is_file() else {}
    expected_sync_cells = int(sync_manifest.get("expected_cells", len([row for row in manifest.get("records", []) if row.get("eligibility") == "eligible" and int(row["id"]) in {int(value) for value in manifest.get("run_sample_ids", [])}]) * 2))
    completed_sync_cells = sum(row.get("status") == "COMPLETE" for row in sync_rows)
    failed_sync_cells = max(0, expected_sync_cells - completed_sync_cells)
    source_excluded = sum(1 for row in manifest.get("records", []) if int(row["id"]) in {int(value) for value in manifest.get("run_sample_ids", [])} and row.get("eligibility") != "eligible")
    status = "PASS" if not failures and max_loss_error <= 1e-5 and statistics_max_error <= 1e-8 else "FAIL"
    validation = {"schema_version": 1, "status": status, "checked_ctc_cells": checked_cells, "max_ctc_abs_error": max_loss_error, "statistics_max_abs_error": statistics_max_error, "tolerances": {"logp": 1e-5, "ctc_abs": 1e-5, "statistics_abs": 1e-8}, "failures": failures, "expected": {"views": expected_views, "sync_cells": expected_sync_cells}, "completed": {"views": completed_views, "sync_cells": completed_sync_cells}, "failed": {"views": failed_views, "sync_cells": failed_sync_cells}, "source_excluded": source_excluded, "independently_checked": len(independent_pairs), "independently_checked_counts": {"pairs": len(independent_pairs), "ctc_cells": checked_cells, "sync_cells": completed_sync_cells}}
    write_json(run_dir / "validation.json", validation)
    analysis["engineering_status"] = status
    analysis["sync_status"] = _run_sync_status(manifest, analysis.get("linkage", {}))
    analysis["visual_status"] = visual_status(engineering_status=status, calibration_status=analysis.get("calibration_status", "INCONCLUSIVE_VSR_VALIDITY"), joint_count=int(analysis.get("joint_count", 0)), g_summary=analysis.get("linkage", {}).get("metrics", {}).get("g", {}), b_summary=analysis.get("linkage", {}).get("metrics", {}).get("b", {}), matched_summary=analysis.get("linkage", {}).get("metrics", {}).get("gmatched", {}))
    assoc = analysis.get("linkage", {}).get("primary_association", {}).get("spearman", {})
    ci = analysis.get("linkage", {}).get("primary_association", {}).get("bootstrap", {}).get("ci95")
    analysis["association_status"] = association_status(status, int(analysis.get("joint_count", 0)), analysis.get("calibration_status", "INCONCLUSIVE_VSR_VALIDITY"), {"rho": assoc.get("rho"), "p_two_sided": assoc.get("p_two_sided"), "ci95": ci})
    write_json(run_dir / "analysis.json", analysis)
    _write_report(run_dir, analysis)
    return validation


def verify_inputs(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    """Public alias used by downstream handoffs and tests."""

    _verify_frozen_inputs(run_dir, manifest)


def _ensure_manifest(run_dir: Path, smoke: bool, resume: bool) -> Dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if bool(manifest.get("smoke")) != bool(smoke):
            raise ValueError("smoke/full scope mismatch")
        _verify_frozen_inputs(run_dir, manifest)
        return manifest
    return audit_ditto50(run_dir, smoke=smoke, resume=resume)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--stage", choices=("audit", "extract", "syncnet", "analyze", "validate", "all"), default="audit")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--syncnet-python", default=SYNCNET_PYTHON_DEFAULT)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume and args.stage in ("audit", "all"):
        raise ValueError("refusing non-empty output directory without --resume: {}".format(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    _set_determinism(BOOTSTRAP_SEED)
    manifest = _ensure_manifest(run_dir, args.smoke, args.resume)
    if args.stage == "audit":
        print(json.dumps({"status": manifest.get("status"), "candidate_count": manifest.get("candidate_count"), "run_sample_ids": manifest.get("run_sample_ids")}, ensure_ascii=False))
        return 0
    if args.stage in ("extract", "all"):
        extract_ditto50(run_dir, manifest, device=args.device, resume=args.resume)
    if args.stage in ("syncnet", "all"):
        score_syncnet(run_dir, manifest, syncnet_python=args.syncnet_python, resume=args.resume)
    if args.stage in ("analyze", "all"):
        analyze_ditto50(run_dir, manifest)
    if args.stage in ("validate", "all"):
        if not (run_dir / "analysis.json").is_file():
            analyze_ditto50(run_dir, manifest)
        validation = validate_ditto50(run_dir, manifest)
        print(json.dumps({"status": validation.get("status"), "failures": len(validation.get("failures", []))}, ensure_ascii=False))
        return 0 if validation.get("status") == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
