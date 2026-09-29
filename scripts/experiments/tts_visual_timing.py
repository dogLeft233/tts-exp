#!/usr/bin/env python3
"""Runner for ``tts_visual_timing_v1``.

The runner is intentionally a thin, auditable state machine.  It never
modifies parent runs and it keeps execution state separate from scientific
state.  Heavy visual/media work is delegated to the Python-3.8 worker; the
measurement and decisions are pure functions in ``tts_visual_timing_metrics``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from tts_visual_timing_metrics import (
    FPS,
    aperture_events,
    audio_time_map,
    bootstrap_mean,
    continuous_support_blocks,
    decide,
    event_distance,
    evaluate_calibration_record,
    group_summary,
    map_time,
    normalized_to_pixels,
)


REPO = Path(__file__).resolve().parents[2]
# The runner is normally invoked as ``python scripts/experiments/...``.  In
# that form Python puts the script directory on ``sys.path`` rather than the
# repository root, while the TextGrid/MFA helpers are imported as the
# namespace package ``scripts.experiments``.  Bind the root explicitly so
# audit metadata and the gated generation path do not turn into a spurious
# ``No module named 'scripts'`` failure.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DEFAULT_RUN = REPO / "runs/tts_visual_timing_v1"
SMOKE_RUN = REPO / "runs/tts_visual_timing_smoke"
PROTOCOL = "tts_visual_timing_v1"
PROTOCOLS: Dict[str, Dict[str, Any]] = {
    "tts_visual_timing_v1": {
        "warp_knots": [[0.0, 0.0], [0.20, 0.0], [0.35, 1.0], [0.65, 1.0], [0.80, 0.0], [1.0, 0.0]],
        "reference_window": [0.35, 0.65],
        "version": 1,
    },
    "tts_visual_timing_v2": {
        "warp_knots": [[0.0, 0.0], [0.10, 0.0], [0.20, 1.0], [0.80, 1.0], [0.90, 0.0], [1.0, 0.0]],
        "reference_window": [0.20, 0.80],
        "reference_safety_s": 0.400,
        "support_gap_tolerance_s": 1e-6,
        "version": 2,
    },
}
SEED = 20260920
BOOTSTRAP_DRAWS = 20_000
PRIMARY_ALPHA = 0.0166666666666667
MAIN_IDS = tuple(range(151, 163))
CALIBRATION_IDS = ("lrs3_6ul2TSvUDog_00007", "lrs3_6wk4dkYSrV0_00006", "lrs3_73jPh0eRPSY_00008", "lrs3_6qqqVwM6bMM_00007")
INPUTS = REPO / "runs/tts_time_instance_20260917_v1/inputs.json"
CALIBRATION_PROTOCOL = REPO / "runs/lrs3_real_video_local_timing_20260905_tail_v2/protocol.json"
ASSETS = REPO / "runs/tts_native_gain_attribution_implementation_20260915_v1/00_audit/assets.json"
B_AUDIO_MANIFEST = REPO / "runs/tts_time_instance_20260917_v1/B/audio_manifest.json"
MECHANISM_SUMMARY = REPO / "runs/lrs3_tts_gain_mechanism_review_v15/03_analysis/summary.json"
VSR_RUN = REPO / "runs/vsr_ditto50_linkage_v1"
LANDMARKER = REPO / "runs/lrs3_tts_visual_advantage_20260824/01_metric_parity_retry7/snapshots/mediapipe_landmarker.task"
LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
WAV2LIP_CHECKPOINT = REPO / "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth"
SYNCNET_MODEL = REPO / "third_party/syncnet_python/data/syncnet_v2.model"
WORKER = Path(__file__).with_name("tts_visual_timing_worker.py")
FIXTURES = Path(__file__).with_name("tts_visual_timing_fixtures.py")
CHECKER = Path(__file__).with_name("check_tts_visual_timing.py")
RENDER_WORKER = REPO / "scripts/experiments/static_image_bridge/render_worker.py"
SYNCNET_WORKER = Path(__file__).with_name("tts_visual_timing_syncnet_worker.py")
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
VISUAL_PYTHON = Path("/home/wjj/miniconda3/envs/autoavsr/bin/python")
WAV2LIP_PYTHON = Path("/home/wjj/.venvs/wav2lip/bin/python")
SYNCNET_PYTHON = Path("/home/wjj/.venvs/syncnet/bin/python")
MFA = Path("/home/wjj/.local/bin/mfa")
MFA_DICTIONARY = "english_us_mfa"
MFA_ACOUSTIC = "english_mfa"


def _protocol_config(protocol: str) -> Dict[str, Any]:
    if protocol not in PROTOCOLS:
        raise ValueError("unknown timing protocol: %s" % protocol)
    config = dict(PROTOCOLS[protocol])
    config["protocol"] = protocol
    config["seed"] = SEED
    config["bootstrap_draws"] = BOOTSTRAP_DRAWS
    config["primary_alpha"] = PRIMARY_ALPHA
    config["main_ids"] = list(MAIN_IDS)
    config["calibration_ids"] = list(CALIBRATION_IDS)
    return config


def _code_bindings() -> List[Dict[str, Any]]:
    code_files = [Path(__file__), Path(__file__).with_name("tts_visual_timing_metrics.py"), WORKER, SYNCNET_WORKER, FIXTURES, CHECKER]
    return [{"path": str(item.resolve()), "sha256": sha256_file(item) if item.is_file() else None} for item in code_files]


def _identity_hash(protocol: str, *, smoke: bool) -> str:
    return _json_hash({"protocol": _protocol_config(protocol), "smoke": bool(smoke), "code": _code_bindings()})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%s.tmp" % (path.name, os.getpid()))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".%s.%s.tmp" % (path.name, os.getpid()))
    temporary.write_text(value, encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _path_binding(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), "exists": resolved.is_file(), "sha256": sha256_file(resolved) if resolved.is_file() else None}


def _run_command(command: Sequence[str], *, cwd: Optional[Path] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
    result = subprocess.run([str(item) for item in command], cwd=str(cwd) if cwd else None, capture_output=True, check=False, timeout=timeout)
    return {"command": [str(item) for item in command], "returncode": int(result.returncode), "stdout": result.stdout.decode("utf-8", errors="replace"), "stderr": result.stderr.decode("utf-8", errors="replace")}


def _probe(path: Path) -> Dict[str, Any]:
    result = _run_command(("ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)))
    if result["returncode"] != 0:
        raise RuntimeError("ffprobe failed for %s" % path)
    return read_json_from_text(result["stdout"])


def read_json_from_text(value: str) -> Any:
    return json.loads(value, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)))


def _transcript_text_from_source(raw: str) -> str:
    """Extract the transcript field from an LRS3 text/LAB sidecar.

    These sidecars intentionally contain ``TEXT:`` and ``CONF:`` metadata;
    hashing the complete file and comparing its whole body to the transcript
    would falsely report every valid input as a binding failure.
    """

    match = re.search(r"^\s*TEXT:\s*(.*?)\s*$", raw, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1) if match else raw


def _first_frame_pts(path: Path) -> Dict[str, Any]:
    result = _run_command(("ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1", "-show_entries", "frame=pts,best_effort_timestamp,pts_time,best_effort_timestamp_time", "-of", "json", str(path)))
    if result["returncode"] != 0:
        return {}
    payload = read_json_from_text(result["stdout"])
    frames = payload.get("frames", []) if isinstance(payload, Mapping) else []
    return dict(frames[0]) if frames and isinstance(frames[0], Mapping) else {}


def _wav_meta(path: Path) -> Dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        channels, rate, width, count = handle.getnchannels(), handle.getframerate(), handle.getsampwidth(), handle.getnframes()
        raw = handle.readframes(count)
    pcm_hash = sha256_bytes(np.frombuffer(raw, dtype="<i2").astype("<i2", copy=False).tobytes()) if width == 2 else sha256_bytes(raw)
    return {"path": str(path.resolve()), "file_sha256": sha256_file(path), "channels": channels, "sample_rate": rate, "sample_width": width, "sample_count": count, "pcm_sha256": pcm_hash}


def _stream_meta(path: Path) -> Dict[str, Any]:
    payload = _probe(path)
    streams = payload.get("streams", []) if isinstance(payload, Mapping) else []
    video = next((dict(row) for row in streams if row.get("codec_type") == "video"), None)
    audio = next((dict(row) for row in streams if row.get("codec_type") == "audio"), None)
    return {"path": str(path.resolve()), "file_sha256": sha256_file(path), "video": video, "audio": audio, "first_video_frame": _first_frame_pts(path), "format": payload.get("format", {})}


def _load_parent_inputs() -> Tuple[Dict[str, Any], Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[str, Any]]:
    inputs = read_json(INPUTS)
    assets_payload = read_json(ASSETS)
    calibration_payload = read_json(CALIBRATION_PROTOCOL)
    mechanism = read_json(MECHANISM_SUMMARY) if MECHANISM_SUMMARY.is_file() else {}
    input_records = {int(row["sample_id"]): dict(row) for row in inputs.get("records", [])}
    asset_records = {int(row["sample_id"]): dict(row) for row in assets_payload.get("records", [])}
    calibration_records = {str(row["sample_id"]): dict(row) for row in calibration_payload.get("records", [])}
    delta_c = {int(row["sample_id"]): dict(row) for row in mechanism.get("paired", []) if isinstance(row, Mapping) and "sample_id" in row}
    # The fourth value is kept as a small typed bundle for the runner; it is
    # not written back to any parent run.
    return inputs, input_records, asset_records, {"calibration": calibration_records, "delta_c": delta_c}  # type: ignore[return-value]


def _calibration_rows() -> List[Dict[str, Any]]:
    payload = read_json(CALIBRATION_PROTOCOL)
    by_id = {str(row.get("sample_id")): dict(row) for row in payload.get("records", [])}
    return [by_id[item] for item in CALIBRATION_IDS if item in by_id]


def _main_record(row: Mapping[str, Any], asset: Mapping[str, Any]) -> Dict[str, Any]:
    sid = int(row["sample_id"])
    sources = asset.get("sources", {})
    result: Dict[str, Any] = {"sample_id": sid, "source_group": str(row["source_group"]), "transcript": row.get("transcript"), "transcript_source": str(row.get("transcript_source", "")), "transcript_source_sha256": row.get("transcript_source_sha256"), "real_video": str(row["real_video"]), "N_audio": str(row["N_audio"]), "T_audio": str(row["T_audio"]), "N_pcm_sha256": row.get("N_pcm_sha256"), "T_pcm_sha256": row.get("T_pcm_sha256"), "portrait": str(row["portrait"]), "videos": {}, "crops": {}, "textgrids": {"N": str(REPO / ("runs/tts_time_instance_20260917_v1/A/A_textgrids/%d_N_audio.TextGrid" % sid)), "T": str(REPO / ("runs/tts_time_instance_20260917_v1/A/A_textgrids/%d_T_audio.TextGrid" % sid))}}
    for arm, key in (("R", "R"), ("N", "N"), ("T", "T")):
        item = sources.get(key, {})
        media = item.get("media", {}) if isinstance(item, Mapping) else {}
        result["videos"][arm] = str(media.get("path", row["real_video"] if arm == "R" else ""))
        if arm in ("N", "T"):
            result["crops"][arm] = {"media": str(item.get("crop", {}).get("path", "")), "selection": str(item.get("crop_selection", {}).get("path", ""))}
    return result


def _clock_binding(record: Dict[str, Any], asset: Mapping[str, Any]) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {"status": "CLOCK_BINDING_UNRESOLVED", "reason": "NOT_CHECKED", "arms": {}}
    try:
        real_probe = _stream_meta(Path(record["videos"]["R"]))
        real_video = real_probe.get("video") or {}
        real_audio = real_probe.get("audio") or {}
        real_start = float(real_probe.get("first_video_frame", {}).get("best_effort_timestamp_time", real_probe.get("first_video_frame", {}).get("pts_time", 0.0)))
        if abs(real_start) > 1e-6 or str(real_video.get("time_base", "")) == "":
            raise ValueError("real video first PTS is not proven at zero")
        evidence["arms"]["R"] = {"video_first_pts_s": real_start, "audio_start_time": real_audio.get("start_time"), "source": "single_container_real_video"}
        for arm, pcm_key in (("N", "N_pcm_sha256"), ("T", "T_pcm_sha256")):
            crop_info = record["crops"][arm]
            selection = read_json(Path(crop_info["selection"]))
            selected = selection.get("selected", {})
            indices = [int(item) for item in selected.get("frame_indices", [])]
            if not indices or int(selected.get("start_frame", -1)) != indices[0] or indices[0] != 0:
                raise ValueError("crop selection has no auditable zero source-frame origin")
            crop_meta = _stream_meta(Path(crop_info["media"]))
            video = crop_meta.get("video") or {}
            audio = crop_meta.get("audio") or {}
            crop_audio_start = float(audio.get("start_time", 0.0) or 0.0)
            crop_video_start = float(crop_meta.get("first_video_frame", {}).get("best_effort_timestamp_time", crop_meta.get("first_video_frame", {}).get("pts_time", 0.0)) or 0.0)
            parent_item = asset.get("sources", {}).get(arm, {})
            pcm_match = parent_item.get("worker_input_pcm_sha256") == record.get("%s_pcm_sha256" % arm)
            if abs(crop_audio_start) > 1e-6 or abs(crop_video_start) > 1e-6 or not pcm_match or str(video.get("time_base", "")) == "":
                raise ValueError("crop stream start or PCM binding is not proven")
            evidence["arms"][arm] = {"selection_start_frame": indices[0], "crop_video_first_pts_s": crop_video_start, "crop_audio_start_time": crop_audio_start, "pcm_binding": pcm_match, "frame_rate": video.get("r_frame_rate"), "time_base": video.get("time_base")}
        evidence["status"] = "RESOLVED"
        evidence["reason"] = "ZERO_SOURCE_FRAME_AND_AUDIO_PTS_BOUND"
    except Exception as exc:
        evidence["reason"] = str(exc)
    return evidence


def audit_inputs(
    run_dir: Path,
    *,
    smoke: bool = False,
    resume: bool = False,
    protocol: str = PROTOCOL,
) -> Dict[str, Any]:
    """Freeze all inputs and prove the available audio/video clock origins."""

    if resume and (run_dir / "manifest.json").is_file():
        existing = read_json(run_dir / "manifest.json")
        if existing.get("protocol") != protocol or existing.get("identity_hash") != _identity_hash(protocol, smoke=smoke):
            raise RuntimeError("CACHE_IDENTITY_MISMATCH")
        return existing
    run_dir.mkdir(parents=True, exist_ok=True)
    # A fresh audit is a new execution ledger.  Resume keeps already completed
    # cells so an interrupted run can fill only the missing work.
    if not resume:
        (run_dir / "cells.jsonl").unlink(missing_ok=True)
    inputs, input_records, assets, extra = _load_parent_inputs()
    delta_c = extra.get("delta_c", {}) if isinstance(extra, Mapping) else {}
    selected_ids = [151] if smoke else list(MAIN_IDS)
    main_records: List[Dict[str, Any]] = []
    failures: List[str] = []
    for sid in selected_ids:
        row = input_records.get(int(sid))
        asset = assets.get(int(sid))
        if row is None or asset is None:
            failures.append("missing parent record %s" % sid)
            continue
        record = _main_record(row, asset)
        record["input_hashes"] = {}
        for key, path_value in [("R_video", record["videos"]["R"]), ("N_video", record["videos"]["N"]), ("T_video", record["videos"]["T"]), ("N_audio", record["N_audio"]), ("T_audio", record["T_audio"]), ("portrait", record["portrait"]), ("transcript_source", record["transcript_source"]), ("N_textgrid", record["textgrids"]["N"]), ("T_textgrid", record["textgrids"]["T"])]:
            path = Path(path_value)
            binding = _path_binding(path)
            record["input_hashes"][key] = binding
            if not binding["exists"]:
                failures.append("%s:%s missing" % (sid, key))
        record["clock_binding"] = _clock_binding(record, asset)
        if record["clock_binding"]["status"] != "RESOLVED":
            failures.append("%s:CLOCK_BINDING_UNRESOLVED" % sid)
        record["delta_c"] = delta_c.get(int(sid), {}).get("delta_c")
        try:
            record["audio_metadata"] = {"N": _wav_meta(Path(record["N_audio"])), "T": _wav_meta(Path(record["T_audio"]))}
            record["video_metadata"] = {arm: _stream_meta(Path(record["videos"][arm])) for arm in ("R", "N", "T")}
            transcript = str(record.get("transcript", ""))
            normalized = " ".join(transcript.upper().split())
            source_path = Path(record["transcript_source"])
            source_text = source_path.read_text(encoding="utf-8")
            source_transcript = _transcript_text_from_source(source_text)
            source_normalized = " ".join(source_transcript.upper().split())
            record["text"] = {"source": str(source_path.resolve()), "source_byte_sha256": sha256_file(source_path), "declared_source_sha256": record.get("transcript_source_sha256"), "transcript_sha256": sha256_bytes(transcript.encode("utf-8")), "normalized_sha256": sha256_bytes(normalized.encode("utf-8")), "transcript_normalized": normalized, "source_text_field": source_transcript, "source_normalized": source_normalized, "source_matches_transcript": source_normalized == normalized}
            if record.get("transcript_source_sha256") and record["text"]["source_byte_sha256"] != record.get("transcript_source_sha256"):
                raise ValueError("transcript source byte hash mismatch")
            if source_normalized != normalized:
                raise ValueError("transcript source content mismatch")
            record["textgrid_timing"] = {}
            for arm in ("N", "T"):
                tokens = _parse_textgrid(Path(record["textgrids"][arm]))
                record["textgrid_timing"][arm] = {"token_count": len(tokens), "tokens": _safe(tokens), "words": sorted({int(item["word_index"]) for item in tokens if item.get("word_index") is not None})}
        except Exception as exc:
            record["metadata_error"] = str(exc)
            failures.append("%s:metadata:%s" % (sid, exc))
        main_records.append(record)
    calibration = _calibration_rows()
    if smoke:
        calibration = calibration[:1]
    calibration_records: List[Dict[str, Any]] = []
    main_groups = {item["source_group"] for item in main_records}
    for row in calibration:
        item = {"sample_id": str(row["sample_id"]), "source_group": str(row.get("source_group", "")), "face_video": str(row["face_video"]["path"]), "natural_audio": str(row["natural_audio"]["path"]), "source_video_timeline": row.get("source_video_timeline", {})}
        item["input_hashes"] = {key: _path_binding(Path(value)) for key, value in (("face_video", item["face_video"]), ("natural_audio", item["natural_audio"]))}
        if not item["input_hashes"]["face_video"]["exists"] or not item["input_hashes"]["natural_audio"]["exists"]:
            failures.append("calibration:%s:missing" % item["sample_id"])
        if item["source_group"] in main_groups:
            failures.append("calibration:%s:source_group_overlap" % item["sample_id"])
        calibration_records.append(item)
    code_files = [Path(__file__), Path(__file__).with_name("tts_visual_timing_metrics.py"), WORKER, SYNCNET_WORKER, FIXTURES, CHECKER]
    for file_path in code_files:
        if not file_path.is_file():
            failures.append("missing_code:%s" % file_path)
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": protocol,
        "status": "COMPLETE" if not failures else "PARTIAL",
        "execution": "COMPLETE" if not failures else "PARTIAL",
        "science": "EXPLORATORY_ONLY",
        "smoke": bool(smoke),
        "fixed_main_denominator": len(MAIN_IDS),
        "selected_main_ids": selected_ids,
        "selected_calibration_ids": [item["sample_id"] for item in calibration_records],
        "main_records": main_records,
        "calibration_records": calibration_records,
        "bindings": {"inputs": _path_binding(INPUTS), "calibration_protocol": _path_binding(CALIBRATION_PROTOCOL), "assets": _path_binding(ASSETS), "mechanism_summary": _path_binding(MECHANISM_SUMMARY), "landmarker": _path_binding(LANDMARKER), "wav2lip_checkpoint": _path_binding(WAV2LIP_CHECKPOINT), "syncnet_model": _path_binding(SYNCNET_MODEL), "generation_audio_manifest": _path_binding(B_AUDIO_MANIFEST), "generation_render_worker": _path_binding(RENDER_WORKER), "mfa_executable": _path_binding(MFA), "ffmpeg": _path_binding(FFMPEG), "code": [{"path": str(item.resolve()), "sha256": sha256_file(item) if item.is_file() else None} for item in code_files]},
        "protocol_config": _protocol_config(protocol),
        "failures": failures,
        "config_hash": _json_hash({"protocol": protocol, "seed": SEED, "main_ids": selected_ids, "calibration_ids": [item["sample_id"] for item in calibration_records], "protocol_config": _protocol_config(protocol)}),
    }
    manifest["identity_hash"] = _identity_hash(protocol, smoke=smoke)
    write_json(run_dir / "manifest.json", _safe(manifest))
    write_json(run_dir / "protocol.json", _safe({"protocol": protocol, "config": _protocol_config(protocol), "identity_hash": manifest["identity_hash"], "config_hash": manifest["config_hash"], "parent_bindings": manifest["bindings"], "code": _code_bindings()}))
    # Record audit outcomes as first-class cells.  This keeps missing metadata,
    # clock failures, and source-binding failures distinguishable from a later
    # scientific gate stop.
    manifest_artifact = str((run_dir / "manifest.json").resolve())
    for record in main_records:
        sid = str(record.get("sample_id"))
        reasons = [item for item in failures if item.startswith(sid + ":")]
        status = "FAILED" if reasons else "COMPLETE"
        _append_cell(run_dir, {
            "stage": "audit", "id": sid, "source_group": record.get("source_group"),
            "arm": "INPUTS", "input_hash": _json_hash(record.get("input_hashes", {})),
            "config_hash": manifest["config_hash"], "status": status,
            "reason": "; ".join(reasons) if reasons else None,
            "artifacts": [manifest_artifact],
        })
    for record in calibration_records:
        sid = str(record.get("sample_id"))
        reasons = [item for item in failures if item.startswith("calibration:" + sid + ":")]
        status = "FAILED" if reasons else "COMPLETE"
        _append_cell(run_dir, {
            "stage": "audit", "id": sid, "source_group": record.get("source_group"),
            "arm": "INPUTS", "input_hash": _json_hash(record.get("input_hashes", {})),
            "config_hash": manifest["config_hash"], "status": status,
            "reason": "; ".join(reasons) if reasons else None,
            "artifacts": [manifest_artifact],
        })
    environment = {"python": platform.python_version(), "platform": platform.platform(), "visual_python": _run_command((str(VISUAL_PYTHON), "-c", "import sys; print(sys.version)")), "wav2lip_python": _run_command((str(WAV2LIP_PYTHON), "-c", "import sys; print(sys.version)")) if WAV2LIP_PYTHON.is_file() else {"status": "MISSING"}, "syncnet_python": _run_command((str(SYNCNET_PYTHON), "-c", "import sys; print(sys.version)")) if SYNCNET_PYTHON.is_file() else {"status": "MISSING"}, "mfa": _run_command((str(MFA), "version")) if MFA.is_file() else {"status": "MISSING"}, "landmarker_sha256_expected": LANDMARKER_SHA256, "landmarker_sha256_actual": sha256_file(LANDMARKER) if LANDMARKER.is_file() else None}
    write_json(run_dir / "environment.json", _safe(environment))
    return manifest


def diagnose(run_dir: Path, manifest: Mapping[str, Any], *, protocol: Optional[str] = None) -> Dict[str, Any]:
    """Run the fixed v2 read-only support diagnosis before visual comparison."""

    protocol = protocol or str(manifest.get("protocol", PROTOCOL))
    if protocol != "tts_visual_timing_v2":
        result = {"protocol": protocol, "status": "NOT_APPLICABLE", "reason": "DIAGNOSIS_IS_V2_REPAIR_STAGE"}
        write_json(run_dir / "diagnosis.json", result)
        return result
    calibration_rows: List[Dict[str, Any]] = []
    old_event_dir = REPO / "runs/tts_visual_timing_v1/events"
    for row in manifest.get("calibration_records", []):
        sid = str(row.get("sample_id"))
        event_path = old_event_dir / ("calibration_%s_REAL.json" % sid)
        events: List[Dict[str, Any]] = []
        if event_path.is_file():
            try:
                payload = read_json(event_path)
                events = [dict(item) for item in payload.get("events", [])]
            except Exception:
                events = []
        frame_count = None
        source = Path(str(row.get("face_video", "")))
        try:
            metadata = _stream_meta(source)
            video = metadata.get("video") or {}
            rate = str(video.get("r_frame_rate", "25/1"))
            numerator, denominator = rate.split("/", 1)
            fps = float(numerator) / float(denominator)
            duration = float((metadata.get("format") or {}).get("duration", 0.0) or 0.0)
            frame_count = int(round(duration * fps)) if duration else None
        except Exception:
            fps = FPS
        interval = _v2_source_interval(frame_count, fps) if frame_count else None
        calibration_rows.append({"sample_id": sid, "event_file": str(event_path), "event_count": len(events), "reference_events_in_v2_interval": sum(interval is not None and interval[0] <= float(event.get("time_s", -1.0)) < interval[1] for event in events), "v1_event_count": len(events), "v2_source_interval_s": None if interval is None else [float(interval[0]), float(interval[1])]})
    mapping_rows: List[Dict[str, Any]] = []
    for record in manifest.get("main_records", []):
        sid = int(record["sample_id"])
        item: Dict[str, Any] = {"sample_id": sid, "source_group": record.get("source_group"), "arms": {}}
        for arm in ("N", "T"):
            path = Path(str(record.get("textgrids", {}).get(arm, "")))
            try:
                tokens = _parse_textgrid(path)
                usable = [token for token in tokens if bool(token.get("label")) and not bool(token.get("silence")) and not bool(token.get("unknown")) and float(token.get("end_s", 0.0)) > float(token.get("start_s", 0.0))]
                missing_word = [token for token in usable if token.get("word_index") is None or not token.get("word_normalized")]
                silence_count = sum(not bool(token.get("label")) or bool(token.get("silence")) or bool(token.get("unknown")) for token in tokens)
                item["arms"][arm] = {"path": str(path), "token_count": len(tokens), "usable_phone_count": len(usable), "silence_or_unknown_count": int(silence_count), "usable_without_word_count": len(missing_word), "word_labels": [str(token.get("word_normalized")) for token in usable if token.get("word_normalized")]}
            except Exception as exc:
                item["arms"][arm] = {"path": str(path), "status": "ERROR", "reason": "%s: %s" % (type(exc).__name__, exc)}
        try:
            n_tokens = _parse_textgrid(Path(record["textgrids"]["N"]))
            t_tokens = _parse_textgrid(Path(record["textgrids"]["T"]))
            mapping = audio_time_map(n_tokens, t_tokens)
            blocks = mapping.get("support_blocks", [])
            item["mapping"] = {"status": mapping.get("status"), "reason": mapping.get("reason"), "n_speech_coverage": mapping.get("n_speech_coverage"), "t_speech_coverage": mapping.get("t_speech_coverage"), "matched_phone_count": mapping.get("matched_phone_count"), "matched_word_count": mapping.get("matched_word_count"), "continuous_block_count": len(blocks), "continuous_block_lengths_s": [{"n": float(block["n_end_s"] - block["n_start_s"]), "t": float(block["t_end_s"] - block["t_start_s"]), "segment_count": len(block.get("segment_indices", []))} for block in blocks]}
        except Exception as exc:
            item["mapping"] = {"status": "ERROR", "reason": "%s: %s" % (type(exc).__name__, exc)}
        mapping_rows.append(item)
    result = {"protocol": protocol, "status": "COMPLETE", "science": "NOT_TESTED", "manifest_identity_hash": manifest.get("identity_hash"), "calibration": calibration_rows, "main_textgrid_mapping": mapping_rows, "rules": {"word_mapping_usable_only": True, "coverage_min": 0.80, "support_gap_tolerance_s": 1e-6, "neighborhood_s": 0.240}}
    write_json(run_dir / "diagnosis.json", _safe(result))
    return result


def _append_cell(run_dir: Path, row: Mapping[str, Any]) -> None:
    path = run_dir / "cells.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_safe(dict(row)), ensure_ascii=False, allow_nan=False)
    key = (str(row.get("stage")), str(row.get("id")), str(row.get("arm")))
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    replaced = False
    updated: List[str] = []
    for line in lines:
        try:
            current = json.loads(line)
            current_key = (str(current.get("stage")), str(current.get("id")), str(current.get("arm")))
        except (TypeError, ValueError):
            current_key = ("", "", "")
        if current_key == key:
            if not replaced:
                updated.append(encoded)
                replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.append(encoded)
    write_text(path, "\n".join(updated) + "\n")


def _run_extract(run_dir: Path, tag: str, video: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    feature_path = run_dir / "features" / (tag + ".npz")
    meta_path = run_dir / "features" / (tag + ".json")
    preview_dir = run_dir / "previews" / tag
    if feature_path.is_file() and meta_path.is_file():
        metadata = read_json(meta_path)
        previews_ok = all(Path(str(item.get("path", ""))).is_file() for item in metadata.get("previews", [])) and len(metadata.get("previews", [])) >= 3
        if metadata.get("video_sha256") == sha256_file(video) and metadata.get("landmarker_sha256") == (sha256_file(LANDMARKER) if LANDMARKER.is_file() else None) and previews_ok:
            with np.load(feature_path, allow_pickle=False) as data:
                return metadata, {"landmarks": np.asarray(data["landmarks"]), "valid": np.asarray(data["valid"], dtype=bool), "timestamps_s": np.asarray(data["timestamps_s"], dtype=np.float64)}
    command = (str(VISUAL_PYTHON), str(WORKER), "--mode", "extract", "--video", str(video), "--landmarker", str(LANDMARKER), "--output", str(feature_path), "--metadata", str(meta_path), "--preview-dir", str(preview_dir))
    result = _run_command(command, timeout=3600.0)
    if result["returncode"] != 0 or not feature_path.is_file() or not meta_path.is_file():
        raise RuntimeError("visual worker failed for %s: %s" % (video, result["stderr"][-1000:]))
    metadata = read_json(meta_path)
    with np.load(feature_path, allow_pickle=False) as data:
        arrays = {"landmarks": np.asarray(data["landmarks"]), "valid": np.asarray(data["valid"], dtype=bool), "timestamps_s": np.asarray(data["timestamps_s"], dtype=np.float64)}
    return metadata, arrays


def _score_features(run_dir: Path, tag: str, metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    result = aperture_events(arrays["landmarks"], arrays["valid"], arrays["timestamps_s"], width=float(metadata["width"]), height=float(metadata["height"]), coordinate_system="normalized")
    if metadata.get("status") == "MULTIPLE_FACES":
        result.update(status="DETECTION_FAILURE", reason="MULTIPLE_FACES_IN_VIDEO")
    event_path = run_dir / "events" / (tag + ".json")
    write_json(event_path, _safe(result))
    return result


def _control_mapping_from_result(result: Mapping[str, Any]) -> List[int]:
    mapping = result.get("mapping")
    if not isinstance(mapping, list) or not mapping:
        raise ValueError("worker did not return a frame mapping")
    return [int(item) for item in mapping]


def _run_retime(
    run_dir: Path,
    tag: str,
    source: Path,
    *,
    delta_ms: float = 0.0,
    frozen: bool = False,
    protocol: str = PROTOCOL,
) -> Dict[str, Any]:
    output = run_dir / "controls" / (tag + ".mkv")
    result_path = run_dir / "controls" / (tag + ".json")
    if output.is_file() and result_path.is_file():
        value = read_json(result_path)
        if value.get("source_sha256") == sha256_file(source) and value.get("worker_sha256") == sha256_file(WORKER) and value.get("protocol", "tts_visual_timing_v1") == protocol and float(value.get("delta_s", value.get("mapping_meta", {}).get("delta_s", 0.0)) or 0.0) == float(delta_ms) / 1000.0 and bool(value.get("mapping_meta", {}).get("mode") == "FROZEN") == bool(frozen):
            return value
    command = [str(VISUAL_PYTHON), str(WORKER), "--mode", "retime_video", "--video", str(source), "--output", str(output), "--metadata", str(result_path), "--protocol", protocol]
    if frozen:
        command.append("--frozen")
    else:
        command.extend(["--delta-ms", "%.12g" % float(delta_ms)])
    result = _run_command(command, timeout=3600.0)
    if result["returncode"] != 0 or not result_path.is_file():
        raise RuntimeError("retime worker failed for %s: %s" % (source, result["stderr"][-1000:]))
    value = read_json(result_path)
    value["protocol"] = protocol
    value["delta_s"] = None if frozen else float(delta_ms) / 1000.0
    value["worker_sha256"] = sha256_file(WORKER)
    write_json(result_path, value)
    return value


def _events_in(result: Mapping[str, Any], left: float = -math.inf, right: float = math.inf) -> List[Dict[str, Any]]:
    return [dict(item) for item in result.get("events", []) if left <= float(item.get("time_s", -1.0)) < right]


def _expected_event_time(source_index: float, mapping: Sequence[int], fps: float, *, allow_nearest: bool = True) -> Optional[float]:
    source_value = float(source_index)
    if not math.isfinite(source_value):
        return None
    exact_index = int(math.floor(source_value))
    positions = [index for index, value in enumerate(mapping) if int(value) == exact_index and abs(source_value - exact_index) <= 1e-9]
    if positions:
        return float(0.5 * (positions[0] + positions[-1]) / fps)
    left = int(math.floor(source_value)); right = int(math.ceil(source_value))
    if left != right and 0 <= left < right < len(mapping):
        left_positions = [index for index, value in enumerate(mapping) if int(value) == left]
        right_positions = [index for index, value in enumerate(mapping) if int(value) == right]
        if left_positions and right_positions:
            left_time = 0.5 * (left_positions[0] + left_positions[-1]) / fps
            right_time = 0.5 * (right_positions[0] + right_positions[-1]) / fps
            return float(left_time + (source_value - left) * (right_time - left_time))
    if not allow_nearest or not mapping:
        return None
    nearest = min(range(len(mapping)), key=lambda index: abs(int(mapping[index]) - source_value))
    return float(nearest / fps)


def _matched_recovery(expected: Sequence[Any], observed: Sequence[Any], tolerance_s: float = 0.040) -> Dict[str, Any]:
    if not expected:
        return {"recovery_rate": None, "matched_count": 0, "median_abs_error_ms": None, "median_signed_shift_ms": None, "distance_ms": None}
    distance = event_distance(expected, observed) if observed or expected else None
    matches = distance.get("matches", []) if distance else []
    good = [item for item in matches if float(item["error_s"]) <= tolerance_s]
    signed: List[float] = []
    for item in good:
        ref_time = float(expected[int(item["reference_index"])]["time_s"] if isinstance(expected[int(item["reference_index"])], Mapping) else expected[int(item["reference_index"])])
        pred_time = float(observed[int(item["prediction_index"])]["time_s"] if isinstance(observed[int(item["prediction_index"])], Mapping) else observed[int(item["prediction_index"])])
        signed.append(pred_time - ref_time)
    return {"recovery_rate": float(len(good) / len(expected)), "matched_count": len(good), "median_abs_error_ms": float(np.median([float(item["error_s"]) for item in good]) * 1000.0) if good else None, "median_signed_shift_ms": float(np.median(signed) * 1000.0) if signed else None, "distance_ms": float(distance["distance_ms"]) if distance else None}


def _v2_source_interval(frame_count: int, fps: float = FPS) -> Tuple[float, float]:
    length = float(frame_count) / float(fps)
    return (0.20 * length + 0.400, 0.80 * length - 0.400)


def _v2_supported_events(events: Sequence[Mapping[str, Any]], mapping: Sequence[int], source_interval: Tuple[float, float]) -> List[Dict[str, Any]]:
    left, right = map(float, source_interval)
    result: List[Dict[str, Any]] = []
    for event in events:
        position = float(event.get("event_position_frames", event.get("index")))
        lo = int(math.floor(position)); hi = int(math.ceil(position))
        if lo < 0 or hi >= len(mapping):
            continue
        source_position = float(mapping[lo]) if lo == hi else float(mapping[lo]) + (position - lo) * float(int(mapping[hi]) - int(mapping[lo]))
        if left <= source_position / FPS < right:
            result.append(dict(event))
    return result


def _calibrate_v2(run_dir: Path, manifest: Mapping[str, Any], *, smoke: bool = False) -> Dict[str, Any]:
    rows = list(manifest.get("calibration_records", []))
    output: Dict[str, Any] = {"protocol": "tts_visual_timing_v2", "execution": "COMPLETE", "science": "NOT_TESTED" if smoke else "NO_CLEAR_SUPPORT", "records": [], "controls": [], "gate": {}}
    if not rows:
        output.update(execution="PARTIAL", science="INSUFFICIENT_SUPPORT", reason="NO_CALIBRATION_RECORDS")
        write_json(run_dir / "calibration.json", output)
        return output
    control_specs = (("REPEAT", 0.0, False), ("LOCAL_+80", 80.0, False), ("LOCAL_-80", -80.0, False), ("LOCAL_+160", 160.0, False), ("LOCAL_-160", -160.0, False), ("FROZEN", None, True))
    for row in rows:
        sid = str(row["sample_id"])
        source = Path(row["face_video"])
        record: Dict[str, Any] = {"sample_id": sid, "source_group": row.get("source_group"), "status": "ERROR", "reason": None, "arms": {}, "input_hash": sha256_file(source) if source.is_file() else None, "protocol": "tts_visual_timing_v2"}
        try:
            real_meta, real_arrays = _run_extract(run_dir, "calibration_%s_REAL" % sid, source)
            real_metric = _score_features(run_dir, "calibration_%s_REAL" % sid, real_meta, real_arrays)
            record["arms"]["REAL"] = real_metric
            _append_cell(run_dir, {"stage": "calibrate", "id": sid, "source_group": row.get("source_group"), "arm": "REAL", "input_hash": record.get("input_hash"), "config_hash": _json_hash({"protocol": "tts_visual_timing_v2", "control": "REAL"}), "status": "COMPLETE", "reason": None, "artifacts": [str((run_dir / "events" / ("calibration_%s_REAL.json" % sid)).resolve())]})
            source_interval = _v2_source_interval(len(real_arrays["timestamps_s"]), FPS)
            reference_events = [dict(event) for event in real_metric.get("events", []) if source_interval[0] <= float(event["time_s"]) < source_interval[1]]
            record["source_interval_s"] = [float(source_interval[0]), float(source_interval[1])]
            record["reference_events"] = reference_events
            controls: Dict[str, Dict[str, Any]] = {}
            for arm, delta_ms, frozen in control_specs:
                worker_result = _run_retime(run_dir, "calibration_%s_%s" % (sid, arm), source, delta_ms=0.0 if delta_ms is None else delta_ms, frozen=frozen, protocol="tts_visual_timing_v2")
                control_path = Path(worker_result["output"])
                meta, arrays = _run_extract(run_dir, "calibration_%s_%s" % (sid, arm), control_path)
                metric = _score_features(run_dir, "calibration_%s_%s" % (sid, arm), meta, arrays)
                metric["mapping"] = _control_mapping_from_result(worker_result)
                metric["delta_s"] = None if delta_ms is None else float(delta_ms) / 1000.0
                metric["protocol"] = "tts_visual_timing_v2"
                controls[arm] = metric
                record["arms"][arm] = metric
                _append_cell(run_dir, {"stage": "calibrate", "id": sid, "source_group": row.get("source_group"), "arm": arm, "input_hash": worker_result.get("source_sha256"), "config_hash": _json_hash({"protocol": "tts_visual_timing_v2", "delta_ms": delta_ms, "frozen": frozen, "worker_sha256": worker_result.get("worker_sha256")}), "status": "COMPLETE", "reason": None, "artifacts": [str(control_path), str((run_dir / "events" / ("calibration_%s_%s.json" % (sid, arm))).resolve())]})
            evaluated_controls = {
                arm: {"events": metric.get("events", []), "mapping": metric.get("mapping", []), "status": metric.get("status"), "delta_s": metric.get("delta_s")}
                for arm, metric in controls.items()
            }
            evaluation = evaluate_calibration_record(reference_events, evaluated_controls, source_interval=source_interval, fps=FPS)
            record.update({"summary": evaluation, "checks": evaluation.get("checks", {}), "passed": evaluation.get("passed", False), "status": evaluation.get("status"), "reason": evaluation.get("reason")})
            # Preserve the actual per-arm evidence while keeping the compact
            # summary as the independent decision input.
            for arm, summary in evaluation.get("arms", {}).items():
                record["arms"].setdefault(arm, {}).update({"calibration": summary})
        except Exception as exc:
            record["status"] = "ERROR"
            record["reason"] = "%s: %s" % (type(exc).__name__, exc)
            output["execution"] = "PARTIAL"
        # REAL is also the record-level cell, replacing the provisional REAL
        # row without creating a second key in cells.jsonl.
        _append_cell(run_dir, {"stage": "calibrate", "id": sid, "source_group": row.get("source_group"), "arm": "REAL", "input_hash": record.get("input_hash"), "config_hash": _json_hash({"protocol": "tts_visual_timing_v2", "control": "REAL", "record_decision": True}), "status": "COMPLETE" if record.get("status") == "PASS" else "FAILED", "reason": record.get("reason"), "artifacts": [str((run_dir / "events" / ("calibration_%s_REAL.json" % sid)).resolve())]})
        output["records"].append(record)
    passed = sum(bool(item.get("passed")) for item in output["records"])
    required = 99 if smoke else 3
    output["gate"] = {"passed_records": passed, "record_count": len(output["records"]), "required_passed": required, "status": "VISUAL_TIMING_CALIBRATED" if passed >= required else "NOT_CALIBRATED"}
    output["science"] = "NOT_TESTED" if smoke else ("SUPPORTED" if passed >= 3 else "NO_CLEAR_SUPPORT")
    write_json(run_dir / "calibration.json", _safe(output))
    return output


def calibrate(run_dir: Path, manifest: Mapping[str, Any], *, smoke: bool = False, protocol: Optional[str] = None) -> Dict[str, Any]:
    protocol = protocol or str(manifest.get("protocol", PROTOCOL))
    if protocol == "tts_visual_timing_v2":
        return _calibrate_v2(run_dir, manifest, smoke=smoke)
    rows = list(manifest.get("calibration_records", []))
    output: Dict[str, Any] = {"protocol": PROTOCOL, "execution": "COMPLETE", "science": "NOT_TESTED" if smoke else "NO_CLEAR_SUPPORT", "records": [], "controls": [], "gate": {}}
    if not rows:
        output.update(execution="PARTIAL", science="INSUFFICIENT_SUPPORT", reason="NO_CALIBRATION_RECORDS")
        write_json(run_dir / "calibration.json", output)
        return output
    for row in rows:
        sid = str(row["sample_id"])
        source = Path(row["face_video"])
        record: Dict[str, Any] = {"sample_id": sid, "source_group": row.get("source_group"), "status": "FAILED", "reason": None, "arms": {}, "input_hash": sha256_file(source) if source.is_file() else None}
        try:
            real_meta, real_arrays = _run_extract(run_dir, "calibration_%s_REAL" % sid, source)
            real_metric = _score_features(run_dir, "calibration_%s_REAL" % sid, real_meta, real_arrays)
            record["arms"]["REAL"] = real_metric
            _append_cell(run_dir, {"stage": "calibrate", "id": sid, "source_group": row.get("source_group"), "arm": "REAL", "input_hash": record.get("input_hash"), "config_hash": _json_hash({"metric": "aperture_events", "control": "REAL"}), "status": "COMPLETE", "reason": None, "artifacts": [str(run_dir / "events" / ("calibration_%s_REAL.json" % sid))]})
            repeat_worker = _run_retime(run_dir, "calibration_%s_REPEAT" % sid, source, delta_ms=0.0)
            control_specs = [("REPEAT", repeat_worker, 0.0, False), ("LOCAL_+80", _run_retime(run_dir, "calibration_%s_LOCAL_P80" % sid, source, delta_ms=80.0), 80.0, False), ("LOCAL_-80", _run_retime(run_dir, "calibration_%s_LOCAL_M80" % sid, source, delta_ms=-80.0), -80.0, False), ("LOCAL_+160", _run_retime(run_dir, "calibration_%s_LOCAL_P160" % sid, source, delta_ms=160.0), 160.0, False), ("LOCAL_-160", _run_retime(run_dir, "calibration_%s_LOCAL_M160" % sid, source, delta_ms=-160.0), -160.0, False), ("FROZEN", _run_retime(run_dir, "calibration_%s_FROZEN" % sid, source, frozen=True), None, True)]
            controls: Dict[str, Dict[str, Any]] = {}
            for arm, worker_result, delta_ms, frozen in control_specs:
                control_path = Path(worker_result["output"])
                meta, arrays = _run_extract(run_dir, "calibration_%s_%s" % (sid, arm), control_path)
                metric = _score_features(run_dir, "calibration_%s_%s" % (sid, arm), meta, arrays)
                mapping = _control_mapping_from_result(worker_result)
                metric["mapping"] = mapping
                metric["delta_ms"] = delta_ms
                controls[arm] = metric
                record["arms"][arm] = metric
                _append_cell(run_dir, {"stage": "calibrate", "id": sid, "source_group": row.get("source_group"), "arm": arm, "input_hash": worker_result.get("source_sha256"), "config_hash": _json_hash({"delta_ms": delta_ms, "frozen": frozen}), "status": "COMPLETE", "reason": None, "artifacts": [str(control_path), str(run_dir / "events" / ("calibration_%s_%s.json" % (sid, arm)))]})
            source_frames = len(real_arrays["timestamps_s"])
            duration = source_frames / FPS
            lower, upper = 0.35 * duration + 0.24, 0.65 * duration - 0.24
            reference_events = _events_in(real_metric, lower, upper)
            if len(reference_events) < 2 or real_metric.get("status") != "MEASURABLE":
                raise RuntimeError("calibration reference has fewer than two measurable central events")
            summary: Dict[str, Any] = {"central_bounds_s": [lower, upper], "reference_events": reference_events, "arms": {}}
            for arm, metric in controls.items():
                mapping = metric["mapping"]
                expected = [dict(item, time_s=_expected_event_time(int(item["index"]), mapping, FPS)) for item in reference_events]
                observed = _events_in(metric, lower, upper)
                recovery = _matched_recovery(expected, observed)
                distance_to_real = event_distance(reference_events, observed) if observed or reference_events else None
                summary["arms"][arm] = {"expected_events": expected, "observed_events": observed, "recovery": recovery, "distance_to_real_ms": float(distance_to_real["distance_ms"]) if distance_to_real else None, "status": metric.get("status")}
            record["summary"] = summary
            repeat_ok = summary["arms"]["REPEAT"]["distance_to_real_ms"] is not None and summary["arms"]["REPEAT"]["distance_to_real_ms"] <= 20.0
            warp_ok = all(summary["arms"][arm]["recovery"]["recovery_rate"] is not None and summary["arms"][arm]["recovery"]["recovery_rate"] >= 0.8 and summary["arms"][arm]["recovery"]["median_abs_error_ms"] is not None and summary["arms"][arm]["recovery"]["median_abs_error_ms"] <= 40.0 for arm in ("LOCAL_+80", "LOCAL_-80", "LOCAL_+160", "LOCAL_-160"))
            sign_ok = True
            for arm, expected_sign in (("LOCAL_+80", -1), ("LOCAL_+160", -1), ("LOCAL_-80", 1), ("LOCAL_-160", 1)):
                expected_times = summary["arms"][arm]["expected_events"]
                shifts = [float(expected_times[index]["time_s"]) - float(reference_events[index]["time_s"]) for index in range(min(len(expected_times), len(reference_events)))]
                shift = float(np.median(shifts) * 1000.0) if shifts else None
                sign_ok = sign_ok and shift is not None and (shift * expected_sign > 0)
            order_ok = all(summary["arms"][large]["distance_to_real_ms"] is not None and summary["arms"][small]["distance_to_real_ms"] is not None and summary["arms"][large]["distance_to_real_ms"] > summary["arms"][small]["distance_to_real_ms"] > summary["arms"]["REPEAT"]["distance_to_real_ms"] for large, small in (("LOCAL_+160", "LOCAL_+80"), ("LOCAL_-160", "LOCAL_-80")))
            frozen_ok = controls["FROZEN"].get("status") in ("LOW_MOTION", "DETECTION_FAILURE") and not controls["FROZEN"].get("events")
            record["checks"] = {"repeat_e_le_20ms": repeat_ok, "warp_recovery_and_error": warp_ok, "sign_correct": sign_ok, "monotone_error": order_ok, "frozen_not_perfect": frozen_ok}
            record["passed"] = bool(repeat_ok and warp_ok and sign_ok and order_ok and frozen_ok)
            record["status"] = "PASS" if record["passed"] else "FAIL"
            record["reason"] = None if record["passed"] else "CALIBRATION_GATE_FAILED"
        except Exception as exc:
            record["status"] = "FAILED"
            record["reason"] = str(exc)
            output["execution"] = "PARTIAL"
        # Reuse the REAL cell as the record-level outcome.  The seven planned
        # control cells remain exactly one row each, while a central-event or
        # monotonicity failure is visible in the ledger instead of looking like
        # a successful calibration merely because all videos decoded.
        _append_cell(run_dir, {
            "stage": "calibrate", "id": sid, "source_group": row.get("source_group"),
            "arm": "REAL", "input_hash": record.get("input_hash"),
            "config_hash": _json_hash({"metric": "aperture_events", "control": "REAL", "record_decision": True}),
            "status": "COMPLETE" if record.get("status") == "PASS" else "FAILED",
            "reason": record.get("reason"),
            "artifacts": [str((run_dir / "events" / ("calibration_%s_REAL.json" % sid)).resolve())],
        })
        output["records"].append(record)
    passed = sum(bool(row.get("passed")) for row in output["records"])
    required = 3 if not smoke else 99
    output["gate"] = {"passed_records": passed, "record_count": len(output["records"]), "required_passed": required, "status": "VISUAL_TIMING_CALIBRATED" if passed >= required else "NOT_CALIBRATED"}
    output["science"] = "NOT_TESTED" if smoke else ("SUPPORTED" if passed >= 3 else "NO_CLEAR_SUPPORT")
    write_json(run_dir / "calibration.json", _safe(output))
    return output


def audit_vsr(run_dir: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Recompute the old Ditto-50 summary from saved rows only."""

    output: Dict[str, Any] = {"protocol": "vsr_ditto50_linkage_v1", "execution": "COMPLETE", "science": "EXPLORATORY_ONLY", "source": str((VSR_RUN / "pair_rows.json").resolve())}
    pair_path = VSR_RUN / "pair_rows.json"
    analysis_path = VSR_RUN / "analysis.json"
    if not pair_path.is_file():
        output.update(execution="PARTIAL", science="NOT_TESTED", status="CACHE_NOT_RECOMPUTABLE", reason="PAIR_ROWS_MISSING")
        write_json(run_dir / "vsr_audit.json", output)
        _append_cell(run_dir, {"stage": "vsr-audit", "id": "vsr_ditto50", "source_group": "ditto50", "arm": "SUMMARY", "input_hash": _path_binding(pair_path).get("sha256"), "config_hash": _json_hash({"protocol": output["protocol"]}), "status": "FAILED", "reason": output["reason"], "artifacts": [str((run_dir / "vsr_audit.json").resolve())]})
        return output
    payload = read_json(pair_path)
    rows = [dict(item) for item in payload.get("rows", [])]
    values = {key: [float(row[key]) for row in rows if row.get(key) is not None] for key in ("g", "gmatched", "delta_c")}
    strict = {"natural": [bool(row.get("natural_strict_target_top1", row.get("natural_top1", False))) for row in rows], "tts": [bool(row.get("tts_strict_target_top1", row.get("tts_top1", False))) for row in rows]}
    def spearman(left: Sequence[float], right: Sequence[float]) -> Optional[float]:
        if len(left) < 2 or len(left) != len(right):
            return None
        x = np.asarray(left, dtype=np.float64)
        y = np.asarray(right, dtype=np.float64)
        rx = np.argsort(np.argsort(x)).astype(np.float64)
        ry = np.argsort(np.argsort(y)).astype(np.float64)
        denominator = np.linalg.norm(rx - rx.mean()) * np.linalg.norm(ry - ry.mean())
        return float(np.dot(rx - rx.mean(), ry - ry.mean()) / denominator) if denominator else None
    old_analysis = read_json(analysis_path) if analysis_path.is_file() else {}
    output.update({"status": "COMPLETE", "pair_count": len(rows), "metrics": {key: {"mean": float(np.mean(item)) if item else None, "median": float(np.median(item)) if item else None, "positive_count": int(sum(value > 0 for value in item))} for key, item in values.items()}, "strict_top1": {arm: {"count": len(item), "positive_count": int(sum(item)), "fraction": float(np.mean(item)) if item else None} for arm, item in strict.items()}, "association": {"spearman_g_delta_c": spearman(values["g"], values["delta_c"])}, "parent_calibration_status": old_analysis.get("calibration_status"), "interpretation": "旧支线仅说明内容辨识存在正迹象；它不是动作时间指标，不能产生英文VSR分数。"})
    write_json(run_dir / "vsr_audit.json", _safe(output))
    _append_cell(run_dir, {"stage": "vsr-audit", "id": "vsr_ditto50", "source_group": "ditto50", "arm": "SUMMARY", "input_hash": _path_binding(pair_path).get("sha256"), "config_hash": _json_hash({"protocol": output["protocol"], "formula": "parent_saved_rows"}), "status": "COMPLETE", "reason": None, "artifacts": [str((run_dir / "vsr_audit.json").resolve())]})
    return output


def _parse_textgrid(path: Path) -> List[Dict[str, Any]]:
    from scripts.experiments.tts_time_instance import parse_textgrid

    return [dict(item) for item in parse_textgrid(path)]


def _aggregate_segment_distances(
    reference: Sequence[Mapping[str, Any]],
    prediction: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    direction: str,
    *,
    support_intervals: Optional[Sequence[Sequence[float]]] = None,
) -> Tuple[Optional[float], int, List[Dict[str, Any]]]:
    # v1's phone-local aperture is retained for historical reruns.  v2 passes
    # frozen intervals made from continuous support blocks and never subtracts
    # 240 ms independently from every phone.
    if support_intervals is None:
        total_cost = 0.0
        denominator = 0
        details: List[Dict[str, Any]] = []
        for segment in segments:
            n_left = float(segment["n_start_s"]) + 0.240
            n_right = float(segment["n_end_s"]) - 0.240
            refs = [item for item in reference if n_left <= float(item["time_s"]) < n_right]
            if not refs:
                continue
            if direction == "identity":
                preds = [item for item in prediction if float(segment["n_start_s"]) <= float(item["time_s"]) < float(segment["n_end_s"])]
            else:
                preds = []
                for item in prediction:
                    mapped = map_time(float(item["time_s"]), segments, direction="t_to_n")
                    if mapped is not None and float(segment["n_start_s"]) <= mapped < float(segment["n_end_s"]):
                        preds.append(dict(item, time_s=mapped))
            distance = event_distance(refs, preds)
            total_cost += float(distance["distance_s"]) * int(distance["reference_count"] + distance["prediction_count"])
            denominator += int(distance["reference_count"] + distance["prediction_count"])
            details.append({"segment": dict(segment), "distance": distance})
        if denominator == 0:
            return None, 0, details
        return float(total_cost / denominator), denominator, details

    total_cost = 0.0
    denominator = 0
    details: List[Dict[str, Any]] = []
    for interval in support_intervals:
        if len(interval) != 2:
            raise ValueError("support interval must contain two endpoints")
        n_left, n_right = float(interval[0]), float(interval[1])
        if not n_right > n_left:
            continue
        refs = [item for item in reference if n_left <= float(item["time_s"]) < n_right]
        if direction == "identity":
            preds = [dict(item) for item in prediction if n_left <= float(item["time_s"]) < n_right]
        else:
            preds = []
            for item in prediction:
                mapped = map_time(float(item["time_s"]), segments, direction="t_to_n")
                if mapped is not None and n_left <= mapped < n_right:
                    preds.append(dict(item, time_s=mapped))
        if not refs and not preds:
            continue
        if not refs and preds:
            # There is no reference in this frozen support interval.  It is
            # not a scoreable interval and must not manufacture a denominator.
            continue
        distance = event_distance(refs, preds)
        total_cost += float(distance["distance_s"]) * int(distance["reference_count"] + distance["prediction_count"])
        denominator += int(distance["reference_count"] + distance["prediction_count"])
        details.append({"support_interval": [n_left, n_right], "distance": distance})
    if denominator == 0:
        return None, 0, details
    return float(total_cost / denominator), denominator, details


def _extract_and_score(run_dir: Path, tag: str, video: Path) -> Dict[str, Any]:
    metadata, arrays = _run_extract(run_dir, tag, video)
    metric = _score_features(run_dir, tag, metadata, arrays)
    metric["metadata"] = metadata
    return metric


def _spearman_rows(rows: Sequence[Mapping[str, Any]], left_key: str, right_key: str) -> Optional[float]:
    usable = [(float(row[left_key]), float(row[right_key])) for row in rows if row.get(left_key) is not None and row.get(right_key) is not None and math.isfinite(float(row[left_key])) and math.isfinite(float(row[right_key]))]
    if len(usable) < 3:
        return None
    left = np.asarray([item[0] for item in usable], dtype=np.float64)
    right = np.asarray([item[1] for item in usable], dtype=np.float64)
    rl = np.argsort(np.argsort(left)).astype(np.float64)
    rr = np.argsort(np.argsort(right)).astype(np.float64)
    denom = np.linalg.norm(rl - rl.mean()) * np.linalg.norm(rr - rr.mean())
    return float(np.dot(rl - rl.mean(), rr - rr.mean()) / denom) if denom else None


def _permutation_spearman(rows: Sequence[Mapping[str, Any]], left_key: str, right_key: str, *, draws: int = BOOTSTRAP_DRAWS) -> Dict[str, Any]:
    usable = sorted([(str(row.get("source_group")), float(row[left_key]), float(row[right_key])) for row in rows if row.get(left_key) is not None and row.get(right_key) is not None], key=lambda item: item[0])
    if len(usable) < 3:
        return {"status": "NOT_ESTIMABLE", "observed": None, "p_two_sided": None, "draws": int(draws), "seed": SEED}
    left = np.asarray([item[1] for item in usable], dtype=np.float64)
    right = np.asarray([item[2] for item in usable], dtype=np.float64)
    observed = _spearman_rows([{left_key: x, right_key: y} for x, y in zip(left, right)], left_key, right_key)
    if observed is None:
        return {"status": "NOT_ESTIMABLE", "observed": None, "p_two_sided": None, "draws": int(draws), "seed": SEED}
    rng = np.random.Generator(np.random.PCG64(SEED))
    extreme = 0
    for _ in range(int(draws)):
        permutation = rng.permutation(right)
        value = _spearman_rows([{left_key: x, right_key: y} for x, y in zip(left, permutation)], left_key, right_key)
        if value is not None and abs(value) >= abs(float(observed)):
            extreme += 1
    return {"status": "COMPLETE", "observed": float(observed), "p_two_sided": float((extreme + 1) / (int(draws) + 1)), "draws": int(draws), "seed": SEED, "group_labels": [item[0] for item in usable]}


def _freeze_native_support(
    reference_metric: Mapping[str, Any],
    mapping: Mapping[str, Any],
    *,
    neighborhood_s: float = 0.240,
) -> Dict[str, Any]:
    """Freeze event support on the N clock before looking at N/T predictions."""

    reference_events = [dict(item) for item in reference_metric.get("events", [])]
    segments = list(mapping.get("segments", []))
    blocks = list(mapping.get("support_blocks", []))
    intervals: List[List[float]] = []
    accepted: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for event_index, event in enumerate(reference_events):
        center = float(event["time_s"])
        selected_block = None
        for block_index, block in enumerate(blocks):
            if float(block["n_start_s"]) <= center - neighborhood_s and center + neighborhood_s <= float(block["n_end_s"]):
                selected_block = (block_index, block)
                break
        if selected_block is None:
            excluded.append({"reference_index": event_index, "time_s": center, "reason": "NEIGHBORHOOD_NOT_INSIDE_CONTINUOUS_BLOCK"})
            continue
        block_index, block = selected_block
        left, right = center - neighborhood_s, center + neighborhood_s
        intervals.append([left, right])
        accepted.append({"reference_index": event_index, "time_s": center, "support_interval": [left, right], "block_index": block_index, "segment_indices": list(block.get("segment_indices", []))})
    # Merge touching/overlapping event neighborhoods while retaining a source
    # list for audit.  Reference event ownership is already unique.
    merged: List[List[float]] = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1] + 1e-12:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return {"reference_events": reference_events, "support_blocks": blocks, "support_intervals": merged, "accepted_events": accepted, "excluded_events": excluded, "segments": segments, "neighborhood_s": neighborhood_s}


def compare_native(
    run_dir: Path,
    manifest: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    smoke: bool = False,
    protocol: Optional[str] = None,
) -> Dict[str, Any]:
    protocol = protocol or str(manifest.get("protocol", PROTOCOL))
    is_v2 = protocol == "tts_visual_timing_v2"
    output: Dict[str, Any] = {"protocol": protocol, "execution": "SKIPPED_BY_GATE", "science": "NOT_TESTED", "records": [], "gate": {}}
    if calibration.get("science") != "SUPPORTED":
        output["reason"] = "VISUAL_TIMING_CALIBRATION_GATE_NOT_PASSED"
        for record in manifest.get("main_records", []):
            output["records"].append({"sample_id": record.get("sample_id"), "source_group": record.get("source_group"), "status": "SKIPPED_BY_GATE", "reason": output["reason"]})
        _append_main_stage_cells(run_dir, manifest, stage="native", arms=("RNT",), status="SKIPPED_BY_GATE", reason=output["reason"], config={"metric": "visual_event_time"})
        write_json(run_dir / "native.json", output)
        return output
    output["execution"] = "COMPLETE"
    for record in manifest.get("main_records", []):
        sid, group = int(record["sample_id"]), str(record["source_group"])
        item: Dict[str, Any] = {"sample_id": sid, "source_group": group, "status": "FAILED", "reason": None, "delta_c": record.get("delta_c")}
        try:
            r_metric = _extract_and_score(run_dir, "native_%d_R" % sid, Path(record["videos"]["R"]))
            n_metric = _extract_and_score(run_dir, "native_%d_N" % sid, Path(record["videos"]["N"]))
            t_metric = _extract_and_score(run_dir, "native_%d_T" % sid, Path(record["videos"]["T"]))
            if r_metric.get("status") != "MEASURABLE":
                raise RuntimeError("REFERENCE_%s" % r_metric.get("status"))
            if not is_v2 and any(metric.get("status") != "MEASURABLE" for metric in (n_metric, t_metric)):
                raise RuntimeError("one of R/N/T visual event sequences is not measurable")
            for label, metric in (("N", n_metric), ("T", t_metric)):
                if metric.get("status") == "DETECTION_FAILURE":
                    raise RuntimeError("%s_DETECTION_FAILURE" % label)
            n_tokens = _parse_textgrid(Path(record["textgrids"]["N"]))
            t_tokens = _parse_textgrid(Path(record["textgrids"]["T"]))
            mapping = audio_time_map(n_tokens, t_tokens)
            if mapping.get("status") != "COMPLETE":
                raise RuntimeError(str(mapping.get("reason")))
            reference = list(r_metric.get("events", []))
            n_events = list(n_metric.get("events", []))
            t_events = list(t_metric.get("events", []))
            support = _freeze_native_support(r_metric, mapping) if is_v2 else None
            support_intervals = support.get("support_intervals", []) if support else None
            if is_v2:
                write_json(run_dir / "support" / ("native_%d.json" % sid), _safe(support))
                if not support_intervals:
                    raise RuntimeError("NO_SUPPORTED_REFERENCE_EVENTS")
            e_n, denom_n, details_n = _aggregate_segment_distances(reference, n_events, mapping["segments"], "identity", support_intervals=support_intervals)
            e_t, denom_t, details_t = _aggregate_segment_distances(reference, t_events, mapping["segments"], "t_to_n", support_intervals=support_intervals)
            if e_n is None or e_t is None:
                raise RuntimeError("no supported reference events")
            item.update({"status": "COMPLETE", "reference_event_count": int(sum(len(detail["distance"]["matches"]) + detail["distance"]["missing_reference_count"] for detail in details_n)), "support_denominator": denom_n, "E_N_s": e_n, "E_T_s": e_t, "E_N_native_ms": float(e_n * 1000.0), "E_T_natural_clock_ms": float(e_t * 1000.0), "B_V_ms": float((e_n - e_t) * 1000.0), "mapping": mapping, "support": support, "segment_details_N": details_n, "segment_details_T": details_t, "event_counts": {"R": len(reference), "N": len(n_events), "T": len(t_events)}, "prediction_status": {"R": r_metric.get("status"), "N": n_metric.get("status"), "T": t_metric.get("status")}})
            _append_cell(run_dir, {"stage": "native", "id": sid, "source_group": group, "arm": "RNT", "input_hash": _json_hash(record.get("input_hashes", {})), "config_hash": _json_hash({"metric": "aperture_events", "mapping": "ordinal_word_phone_lcs"}), "status": "COMPLETE", "reason": None, "artifacts": [str(run_dir / "events" / ("native_%d_R.json" % sid)), str(run_dir / "events" / ("native_%d_N.json" % sid)), str(run_dir / "events" / ("native_%d_T.json" % sid))]})
        except Exception as exc:
            item["reason"] = str(exc)
            output["execution"] = "PARTIAL"
        output["records"].append(item)
        _append_cell(run_dir, {"stage": "native", "id": str(sid), "source_group": group, "arm": "RNT", "input_hash": _json_hash(record.get("input_hashes", {})), "config_hash": _json_hash({"metric": "aperture_events", "mapping": "ordinal_word_phone_lcs"}), "status": "COMPLETE" if item.get("status") == "COMPLETE" else "FAILED", "reason": item.get("reason"), "artifacts": [str((run_dir / "native.json").resolve())]})
    valid = [row for row in output["records"] if row.get("status") == "COMPLETE" and row.get("B_V_ms") is not None]
    values = {str(row["source_group"]): float(row["B_V_ms"]) for row in valid}
    delta_values = {str(row["source_group"]): float(row["delta_c"]) for row in valid if row.get("delta_c") is not None}
    if smoke:
        output["science"] = "NOT_TESTED"
        output["gate"] = {"status": "SMOKE_ONLY", "valid_records": len(valid)}
    else:
        bv_decision = decide(values, threshold=0.0, minimum_mean=20.0, min_groups=8, draws=BOOTSTRAP_DRAWS, alpha=PRIMARY_ALPHA)
        dc_bootstrap = bootstrap_mean(delta_values, draws=BOOTSTRAP_DRAWS, alpha=0.05)
        output["decision"] = {"BV": bv_decision, "delta_c_95": dc_bootstrap}
        positive_sources = sum(value > 0 for value in values.values())
        output["gate"] = {"status": "VISUAL_NATIVE_SUPPORT" if bv_decision.get("science") == "SUPPORTED" and positive_sources >= math.ceil(2.0 * len(values) / 3.0) and sum(int(row.get("reference_event_count", 0)) for row in valid) >= 16 else "NO_CLEAR_SUPPORT", "valid_source_count": len(values), "reference_event_count": sum(int(row.get("reference_event_count", 0)) for row in valid), "positive_source_count": positive_sources}
        output["science"] = "SUPPORTED" if output["gate"]["status"] == "VISUAL_NATIVE_SUPPORT" else ("INSUFFICIENT_SUPPORT" if len(values) < 8 else "NO_CLEAR_SUPPORT")
    output["association"] = {"spearman_BV_delta_c": _permutation_spearman(valid, "B_V_ms", "delta_c"), "positive_sign_intersection": {"both_positive": int(sum(float(row["B_V_ms"]) > 0 and float(row["delta_c"]) > 0 for row in valid)), "bv_positive": int(sum(float(row["B_V_ms"]) > 0 for row in valid)), "delta_c_positive": int(sum(float(row["delta_c"]) > 0 for row in valid)), "valid_count": len(valid)}}
    write_json(run_dir / "native.json", _safe(output))
    return output


def _half_up(value: float) -> int:
    if not math.isfinite(float(value)):
        raise ValueError("non-finite knot")
    return int(math.floor(float(value) + 0.5))


def build_audio_knots(
    n_tokens: Sequence[Mapping[str, Any]],
    t_tokens: Sequence[Mapping[str, Any]],
    *,
    sample_rate: int = 16000,
    n_duration_samples: Optional[int] = None,
    t_duration_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Build Rubber Band's ``source=T -> target=N`` sample-frame map.

    The MFA map only covers matched phone intervals.  The protocol requires
    the complete audio endpoints as explicit knots, so the caller supplies
    the measured PCM lengths.  When omitted (unit tests and diagnostics), the
    largest matched endpoint is used; generation always supplies both lengths.
    """
    mapping = audio_time_map(n_tokens, t_tokens)
    if mapping.get("status") != "COMPLETE":
        return {"status": "UNSUPPORTED", "reason": mapping.get("reason"), "mapping": mapping}
    source_duration = int(t_duration_samples) if t_duration_samples is not None else max((_half_up(float(item["t_end_s"]) * sample_rate) for item in mapping["segments"]), default=0)
    target_duration = int(n_duration_samples) if n_duration_samples is not None else max((_half_up(float(item["n_end_s"]) * sample_rate) for item in mapping["segments"]), default=0)
    if source_duration <= 0 or target_duration <= 0:
        return {"status": "UNSUPPORTED", "reason": "AUDIO_DURATION_UNAVAILABLE", "mapping": mapping}
    source: List[int] = [0]
    target: List[int] = [0]
    for segment in mapping["segments"]:
        source.extend((_half_up(float(segment["t_start_s"]) * sample_rate), _half_up(float(segment["t_end_s"]) * sample_rate)))
        target.extend((_half_up(float(segment["n_start_s"]) * sample_rate), _half_up(float(segment["n_end_s"]) * sample_rate)))
    # The final knot is mandatory even when the last phone is unmatched.
    source.append(source_duration)
    target.append(target_duration)
    # Retain only monotonic knots.  Equal input coordinates are accepted only
    # when their target also agrees.
    pairs: List[Tuple[int, int]] = []
    for left, right in zip(source, target):
        if pairs and left == pairs[-1][0]:
            if right != pairs[-1][1]:
                return {"status": "UNSUPPORTED", "reason": "CONFLICTING_DUPLICATE_SOURCE_KNOT", "mapping": mapping}
            continue
        pairs.append((left, right))
    for (left0, right0), (left1, right1) in zip(pairs, pairs[1:]):
        if left1 <= left0 or right1 <= right0:
            return {"status": "UNSUPPORTED", "reason": "NON_MONOTONE_KNOTS", "mapping": mapping}
        ratio = (right1 - right0) / float(left1 - left0)
        if ratio < 0.5 or ratio > 2.0:
            return {"status": "UNSUPPORTED", "reason": "LOCAL_STRETCH_OUTSIDE_0_5_2_0", "mapping": mapping}
    return {"status": "COMPLETE", "source_knots": [item[0] for item in pairs], "target_knots": [item[1] for item in pairs], "mapping": mapping}


def _identity_audio_knots(mapping: Mapping[str, Any], side: str, duration_samples: int, *, sample_rate: int = 16000) -> Tuple[List[int], List[int]]:
    """Build identity knots at the same phone-boundary support as T_NAT."""

    if side not in ("n", "t"):
        raise ValueError("identity side must be n or t")
    key_start = side + "_start_s"; key_end = side + "_end_s"
    values = [0, int(duration_samples)]
    for segment in mapping.get("segments", []):
        values.extend((_half_up(float(segment[key_start]) * sample_rate), _half_up(float(segment[key_end]) * sample_rate)))
    knots = sorted(set(int(value) for value in values))
    if not knots or knots[0] != 0 or knots[-1] != int(duration_samples) or any(right <= left for left, right in zip(knots, knots[1:])):
        raise ValueError("identity knots are not strictly monotone")
    return knots, list(knots)


def _find_rubberband(run_dir: Path) -> Optional[Path]:
    for candidate in (Path("rubberband"), Path("/usr/bin/rubberband"), Path("/home/wjj/miniconda3/bin/rubberband"), run_dir / "tools/rubberband/bin/rubberband"):
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return candidate.resolve()
    return None


def _install_rubberband(run_dir: Path) -> Optional[Path]:
    prefix = run_dir / "tools" / "rubberband"
    command = ("/home/wjj/miniconda3/bin/conda", "create", "-y", "-p", str(prefix), "-c", "conda-forge", "rubberband")
    result = _run_command(command, timeout=1800.0)
    if result["returncode"] != 0:
        return None
    candidate = prefix / "bin" / "rubberband"
    return candidate if candidate.is_file() else None


def _prepare_rubberband(run_dir: Path) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Resolve the CLI only after the science gates have admitted generation."""

    candidate = _find_rubberband(run_dir) or _install_rubberband(run_dir)
    if candidate is None:
        receipt = {"status": "DEPENDENCY_BLOCKED", "reason": "RUBBERBAND_CLI_NOT_FOUND"}
        write_json(run_dir / "generation" / "rubberband.json", receipt)
        return None, receipt
    version = _run_command((str(candidate), "--version"), timeout=60.0)
    help_result = _run_command((str(candidate), "--full-help"), timeout=60.0)
    help_text = help_result["stdout"] + "\n" + help_result["stderr"]
    receipt = {"status": "COMPLETE", "path": str(candidate.resolve()), "sha256": sha256_file(candidate), "version": version, "full_help": help_result, "timemap_supported": "--timemap" in help_text}
    if version["returncode"] != 0 or help_result["returncode"] != 0 or not receipt["timemap_supported"]:
        receipt["status"] = "DEPENDENCY_BLOCKED"
        receipt["reason"] = "RUBBERBAND_TIMEMAP_OR_HELP_UNAVAILABLE"
        write_json(run_dir / "generation" / "rubberband.json", receipt)
        return None, receipt
    write_json(run_dir / "generation" / "rubberband.json", receipt)
    return candidate, receipt


def _generation_skip(manifest: Mapping[str, Any], reason: str, *, protocol: Optional[str] = None) -> Dict[str, Any]:
    return {"protocol": protocol or str(manifest.get("protocol", PROTOCOL)), "execution": "SKIPPED_BY_GATE", "science": "NOT_TESTED", "reason": reason, "records": [{"sample_id": item.get("sample_id"), "source_group": item.get("source_group"), "status": "SKIPPED_BY_GATE", "reason": reason} for item in manifest.get("main_records", [])], "gate": {"status": "NOT_ENTERED"}}


def _append_main_stage_cells(run_dir: Path, manifest: Mapping[str, Any], *, stage: str, arms: Sequence[str], status: str, reason: Optional[str], config: Mapping[str, Any]) -> None:
    """Write explicit rows for a gated or dependency-blocked main stage."""

    for record in manifest.get("main_records", []):
        sid = str(record.get("sample_id"))
        for arm in arms:
            _append_cell(run_dir, {
                "stage": stage, "id": sid, "source_group": record.get("source_group"),
                "arm": arm, "input_hash": _json_hash(record.get("input_hashes", {})),
                "config_hash": _json_hash(dict(config, arm=arm)), "status": status,
                "reason": reason, "artifacts": [],
            })


def _wav_rms_db(path: Path) -> Optional[float]:
    try:
        pcm, meta = _read_pcm16(path)
    except Exception:
        return None
    if not len(pcm):
        return None
    rms = float(np.sqrt(np.mean(np.square(pcm.astype(np.float64)))))
    return float(20.0 * math.log10(max(rms, 1e-12) / 32768.0))


def _read_pcm16(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    with wave.open(str(path), "rb") as handle:
        channels, rate, width, count = handle.getnchannels(), handle.getframerate(), handle.getsampwidth(), handle.getnframes()
        raw = handle.readframes(count)
    if channels != 1 or rate != 16000 or width != 2:
        raise ValueError("audio must be mono 16 kHz PCM16")
    values = np.frombuffer(raw, dtype="<i2").copy()
    return values, {"sample_count": int(len(values)), "sample_rate": rate, "channels": channels, "sample_width": width}


def _portrait_rgb_hash(path: Path) -> str:
    import cv2

    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("cannot read portrait %s" % path)
    return sha256_bytes(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes())


def _render_wav2lip(run_dir: Path, record: Mapping[str, Any], arm: str, audio: Path, *, repeat: bool = False) -> Dict[str, Any]:
    """Render one fixed-box Wav2Lip arm and verify its receipt.

    This is intentionally a local invocation of ``render_worker.py`` rather
    than a call to the historical three-arm loop.  It makes the five-arm
    protocol explicit and keeps the old run immutable.
    """

    sid = int(record["sample_id"])
    branch = "repeat" if repeat else "main"
    output = run_dir / "generation" / "videos" / branch / str(sid) / (arm + ".mkv")
    receipt_path = output.with_suffix(".worker.json")
    portrait = Path(str(record["portrait"]))
    box = [int(value) for value in record["generation_box"]]
    image_hash = _portrait_rgb_hash(portrait)
    if output.is_file() and receipt_path.is_file():
        cached = read_json(receipt_path)
        if cached.get("status") == "complete" and cached.get("audio_sha256") == sha256_file(audio) and cached.get("image_rgb_sha256") == image_hash and cached.get("box_xyxy") == box and cached.get("seed") == 42:
            return cached
    log = run_dir / "generation" / "logs" / ("render_%s_%s%s.log" % (sid, arm, "_repeat" if repeat else ""))
    command = [str(WAV2LIP_PYTHON), str(RENDER_WORKER), "--image", str(portrait), "--image-rgb-sha256", image_hash, "--audio", str(audio), "--box", *[str(value) for value in box], "--checkpoint", str(WAV2LIP_CHECKPOINT), "--ffmpeg", str(FFMPEG), "--outfile", str(output), "--result", str(receipt_path), "--batch-size", "4", "--seed", "42"]
    log.parent.mkdir(parents=True, exist_ok=True)
    result = _run_command(command, cwd=REPO, timeout=3600.0)
    log.write_text(result["stdout"] + "\n" + result["stderr"], encoding="utf-8")
    if result["returncode"] != 0 or not output.is_file() or not receipt_path.is_file():
        raise RuntimeError("Wav2Lip render failed for %s/%s: %s" % (sid, arm, result["stderr"][-1000:]))
    receipt = read_json(receipt_path)
    if receipt.get("status") != "complete":
        raise RuntimeError("Wav2Lip receipt is not complete for %s/%s" % (sid, arm))
    receipt.update({"sample_id": sid, "source_group": record.get("source_group"), "arm": arm, "repeat": bool(repeat), "command": command, "log": str(log.resolve()), "output_sha256": sha256_file(output), "audio_sha256": sha256_file(audio), "checkpoint_sha256": sha256_file(WAV2LIP_CHECKPOINT)})
    write_json(receipt_path, receipt)
    return receipt


def _run_stretch_audio(run_dir: Path, sid: int, arm: str, source: Path, rubberband: Path, source_knots: Sequence[int], target_knots: Sequence[int]) -> Dict[str, Any]:
    output = run_dir / "generation" / "audio" / ("%d_%s.wav" % (sid, arm))
    receipt_path = output.with_suffix(".json")
    if output.is_file() and receipt_path.is_file():
        cached = read_json(receipt_path)
        if cached.get("status") == "COMPLETE" and cached.get("source_sha256") == sha256_file(source) and cached.get("output_sha256") == sha256_file(output):
            return cached
    output.parent.mkdir(parents=True, exist_ok=True)
    ratio = float(target_knots[-1]) / float(source_knots[-1])
    command = [str(VISUAL_PYTHON), str(WORKER), "--mode", "stretch_audio", "--audio", str(source), "--output", str(output), "--metadata", str(receipt_path), "--rubberband", str(rubberband), "--source-knots", ",".join(str(int(value)) for value in source_knots), "--target-knots", ",".join(str(int(value)) for value in target_knots), "--time-ratio", "%.12g" % ratio]
    result = _run_command(command, cwd=REPO, timeout=1800.0)
    if result["returncode"] != 0 or not output.is_file() or not receipt_path.is_file():
        raise RuntimeError("Rubber Band failed for %d/%s: %s" % (sid, arm, result["stderr"][-1000:]))
    receipt = read_json(receipt_path)
    if receipt.get("status") != "COMPLETE":
        raise RuntimeError("Rubber Band receipt is not complete for %d/%s" % (sid, arm))
    return receipt


def _syncnet_score(run_dir: Path, sid: int, arm: str, video: Path, audio: Path) -> Dict[str, Any]:
    output = run_dir / "generation" / "syncnet" / ("%d_%s.json" % (sid, arm))
    if output.is_file():
        cached = read_json(output)
        if cached.get("status") == "COMPLETE" and cached.get("video_sha256") == sha256_file(video) and cached.get("audio_sha256") == sha256_file(audio) and cached.get("model_sha256") == sha256_file(SYNCNET_MODEL):
            return cached
    command = [str(SYNCNET_PYTHON), str(SYNCNET_WORKER), "--video", str(video), "--audio", str(audio), "--model", str(SYNCNET_MODEL), "--output", str(output)]
    result = _run_command(command, cwd=REPO, timeout=3600.0)
    if result["returncode"] != 0 or not output.is_file():
        raise RuntimeError("SyncNet scorer failed for %d/%s: %s" % (sid, arm, result["stderr"][-1000:]))
    return read_json(output)


def _score_c_on_support(score: Mapping[str, Any], support_count: int) -> Optional[float]:
    try:
        matrix = np.load(str(score["distance_matrix"]), allow_pickle=False)
        support = list(range(15, min(int(matrix.shape[0]), int(support_count)) - 15))
        if len(support) < 50:
            return None
        curve = matrix[support].astype(np.float32).mean(axis=0, dtype=np.float32).astype(np.float64)
        return float(np.median(curve) - np.min(curve))
    except Exception:
        return None


def _ci_within(summary: Mapping[str, Any], lower: float, upper: float) -> bool:
    ci = summary.get("ci", [None, None])
    return isinstance(ci, list) and len(ci) == 2 and ci[0] is not None and ci[1] is not None and float(ci[0]) >= lower and float(ci[1]) <= upper


def evaluate_wav2lip_baseline_gate(baseline: Mapping[str, Any], *, mfa_qc_status: str) -> Dict[str, Any]:
    """Apply the fixed baseline gate without inspecting any media artifacts."""

    reasons: List[str] = []
    native_delta = baseline.get("native_delta_c", {})
    bv_w = baseline.get("BV_W", {})
    valid_source_count = int(baseline.get("valid_source_count", 0) or 0)
    native_ci = native_delta.get("ci", [None, None]) if isinstance(native_delta, Mapping) else [None, None]
    bv_ci = bv_w.get("ci", [None, None]) if isinstance(bv_w, Mapping) else [None, None]
    if valid_source_count < 8:
        reasons.append("VALID_SOURCE_COUNT_LT_8")
    if not isinstance(native_ci, list) or len(native_ci) != 2 or native_ci[0] is None or float(native_ci[0]) <= 0:
        reasons.append("NATIVE_DELTA_CI_NOT_ABOVE_ZERO")
    if not isinstance(bv_ci, list) or len(bv_ci) != 2 or bv_ci[0] is None or float(bv_ci[0]) <= 0:
        reasons.append("BV_W_CI_NOT_ABOVE_ZERO")
    for key, lower, upper, label in (
        ("identity_n_delta_c", -0.20, 0.20, "IDENTITY_N_C_NOT_EQUIVALENT"),
        ("identity_t_delta_c", -0.20, 0.20, "IDENTITY_T_C_NOT_EQUIVALENT"),
        ("identity_n_delta_E_ms", -20.0, 20.0, "IDENTITY_N_E_NOT_EQUIVALENT"),
        ("identity_t_delta_E_ms", -20.0, 20.0, "IDENTITY_T_E_NOT_EQUIVALENT"),
    ):
        if not _ci_within(baseline.get(key, {}), lower, upper):
            reasons.append(label)
    if mfa_qc_status != "COMPLETE":
        reasons.append("MFA_QC_%s" % mfa_qc_status)
    return {"status": "WAV2LIP_BASELINE_CONFIRMED" if not reasons else "WAV2LIP_BASELINE_UNCONFIRMED", "reasons": reasons, "mfa_qc_status": mfa_qc_status, "valid_source_count": valid_source_count}


def evaluate_replacement_gate(
    rv_decision: Mapping[str, Any],
    rc_decision: Mapping[str, Any],
    rv_values: Mapping[str, float],
    rc_values: Mapping[str, float],
    *,
    mfa_qc_status: str,
    rv_id_values: Optional[Mapping[str, float]] = None,
    rc_id_values: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Apply the pre-registered replacement gate to already computed summaries."""

    reasons: List[str] = []
    if len(rv_values) < 8 or len(rc_values) < 8:
        reasons.append("VALID_SOURCE_COUNT_LT_8")
    if rv_decision.get("science") != "SUPPORTED" or float(rv_decision.get("mean", -math.inf)) < 20.0:
        reasons.append("R_V_NOT_SUPPORTED")
    if rc_decision.get("science") != "SUPPORTED":
        reasons.append("R_C_NOT_SUPPORTED")
    required_rv_positive = int(math.ceil(2.0 * len(rv_values) / 3.0)) if rv_values else 1
    required_rc_positive = int(math.ceil(2.0 * len(rc_values) / 3.0)) if rc_values else 1
    if sum(float(value) > 0 for value in rv_values.values()) < required_rv_positive:
        reasons.append("R_V_POSITIVE_FRACTION_LOW")
    if sum(float(value) > 0 for value in rc_values.values()) < required_rc_positive:
        reasons.append("R_C_POSITIVE_FRACTION_LOW")
    if mfa_qc_status != "COMPLETE":
        reasons.append("MFA_QC_%s" % mfa_qc_status)
    # The two N_ID comparisons are a directional manipulation check.  They are
    # deliberately a separate condition from the corrected-N primary gates.
    for label, values in (("R_V_ID", rv_id_values), ("R_C_ID", rc_id_values)):
        if values is not None and (not values or sum(float(value) > 0 for value in values.values()) < int(math.ceil(2.0 * len(values) / 3.0))):
            reasons.append("%s_DIRECTION_LOW" % label)
    return {"status": "REPLACEMENT_SUPPORT" if not reasons else "NO_CLEAR_SUPPORT", "reasons": reasons, "valid_source_count": min(len(rv_values), len(rc_values))}


def _generation_context(manifest: Mapping[str, Any]) -> Dict[int, Dict[str, Any]]:
    if not B_AUDIO_MANIFEST.is_file():
        raise RuntimeError("B audio manifest is missing: %s" % B_AUDIO_MANIFEST)
    payload = read_json(B_AUDIO_MANIFEST)
    by_id = {int(item["sample_id"]): dict(item) for item in payload.get("records", [])}
    result: Dict[int, Dict[str, Any]] = {}
    for row in manifest.get("main_records", []):
        sid = int(row["sample_id"])
        item = by_id.get(sid)
        if item is None or str(item.get("source_group")) != str(row.get("source_group")):
            raise RuntimeError("generation audio manifest mismatch for %d" % sid)
        portrait = Path(str(row["portrait"]))
        if item.get("portrait_sha256") and str(item["portrait_sha256"]) != sha256_file(portrait):
            raise RuntimeError("portrait hash mismatch for %d" % sid)
        box = item.get("box_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            raise RuntimeError("missing generation box for %d" % sid)
        result[sid] = dict(row, generation_box=[int(value) for value in box])
    return result


def _generation_mfa_qc(run_dir: Path, records: Mapping[int, Mapping[str, Any]], audio_rows: Sequence[Tuple[int, str, Path]]) -> Dict[str, Any]:
    """Re-align generated audio with the pinned MFA executable.

    This is an intervention check under the same aligner, not an independent
    speech truth.  A failed/blocked alignment is carried into the generation
    receipt and never silently treated as an identity pass.
    """

    try:
        from scripts.experiments.tts_time_instance import parse_textgrid, run_mfa

        mfa_records = []
        for sid, arm, audio in audio_rows:
            source = records[sid]
            mfa_records.append({"sample_id": "%d_%s" % (sid, arm), "transcript": source.get("transcript", ""), "audio": str(audio)})
        manifest = run_mfa(mfa_records, ["audio"], run_dir / "generation" / "mfa", prefix="generated")
        if manifest.get("status") != "COMPLETE":
            return {"status": "MFA_BLOCKED", "manifest": manifest}
        grids = Path(str(manifest["textgrids"]))
        checks: List[Dict[str, Any]] = []
        for sid, arm, _audio in audio_rows:
            generated_grid = grids / ("%d_%s_audio.TextGrid" % (sid, arm))
            if not generated_grid.is_file():
                checks.append({"sample_id": sid, "arm": arm, "status": "MISSING"})
                continue
            source = records[sid]
            target_arm = "N" if arm in ("N_ID", "T_NAT") else "T"
            target_tokens = _parse_textgrid(Path(source["textgrids"][target_arm]))
            observed_tokens = [dict(item) for item in parse_textgrid(generated_grid)]
            mapping = audio_time_map(target_tokens, observed_tokens)
            boundary_errors_ms: List[float] = []
            if mapping.get("status") == "COMPLETE":
                for segment in mapping.get("segments", []):
                    boundary_errors_ms.extend((abs(float(segment["n_start_s"]) - float(segment["t_start_s"])) * 1000.0, abs(float(segment["n_end_s"]) - float(segment["t_end_s"])) * 1000.0))
            median_error = float(np.median(boundary_errors_ms)) if boundary_errors_ms else None
            p90_error = float(np.percentile(boundary_errors_ms, 90)) if boundary_errors_ms else None
            median_limit, p90_limit = ((40.0, 80.0) if arm == "T_NAT" else (20.0, 40.0))
            qc_ok = mapping.get("status") == "COMPLETE" and median_error is not None and p90_error is not None and median_error <= median_limit and p90_error <= p90_limit
            checks.append({"sample_id": sid, "arm": arm, "status": "COMPLETE" if qc_ok else ("BOUNDARY_ERROR" if mapping.get("status") == "COMPLETE" else "INSUFFICIENT_COVERAGE"), "mapping": mapping, "boundary_error_ms": {"median": median_error, "p90": p90_error, "median_limit": median_limit, "p90_limit": p90_limit}})
        return {"status": "COMPLETE" if checks and all(item.get("status") == "COMPLETE" for item in checks) else "MFA_QC_FAILED", "manifest": manifest, "checks": checks}
    except Exception as exc:
        return {"status": "MFA_BLOCKED", "reason": "%s: %s" % (type(exc).__name__, exc)}


def run_generation(run_dir: Path, manifest: Mapping[str, Any], calibration: Mapping[str, Any], native: Mapping[str, Any], *, smoke: bool = False, protocol: Optional[str] = None) -> Dict[str, Any]:
    """Implement the Rubber Band/Wav2Lip branch; execute only after both gates."""

    protocol = protocol or str(manifest.get("protocol", PROTOCOL))
    generation_arms = ("N", "T", "N_ID", "T_ID", "T_NAT")
    if calibration.get("science") != "SUPPORTED":
        result = _generation_skip(manifest, "VISUAL_TIMING_CALIBRATION_GATE_NOT_PASSED", protocol=protocol)
        _append_main_stage_cells(run_dir, manifest, stage="generation", arms=generation_arms, status="SKIPPED_BY_GATE", reason=result["reason"], config={"protocol": "wav2lip_time_map"})
        write_json(run_dir / "generation.json", result)
        return result
    if native.get("gate", {}).get("status") != "VISUAL_NATIVE_SUPPORT" or not native.get("decision", {}).get("delta_c_95", {}).get("ci", [None, 0])[0] > 0:
        result = _generation_skip(manifest, "NATIVE_BV_OR_DELTA_C_GATE_NOT_PASSED", protocol=protocol)
        _append_main_stage_cells(run_dir, manifest, stage="generation", arms=generation_arms, status="SKIPPED_BY_GATE", reason=result["reason"], config={"protocol": "wav2lip_time_map"})
        write_json(run_dir / "generation.json", result)
        return result
    rubberband, rubberband_receipt = _prepare_rubberband(run_dir)
    if rubberband is None:
        result = _generation_skip(manifest, str(rubberband_receipt.get("reason", "DEPENDENCY_BLOCKED_RUBBERBAND_TIMEMAP_UNAVAILABLE")), protocol=protocol)
        result["execution"] = "DEPENDENCY_BLOCKED"
        for item in result["records"]:
            item["status"] = "DEPENDENCY_BLOCKED"
        _append_main_stage_cells(run_dir, manifest, stage="generation", arms=generation_arms, status="DEPENDENCY_BLOCKED", reason=result["reason"], config={"protocol": "wav2lip_time_map"})
        write_json(run_dir / "generation.json", result)
        return result
    try:
        context = _generation_context(manifest)
    except Exception as exc:
        result = _generation_skip(manifest, "GENERATION_INPUT_BINDING_FAILED:%s" % exc, protocol=protocol)
        result["execution"] = "PARTIAL"
        for item in result["records"]:
            item["status"] = "FAILED"
        _append_main_stage_cells(run_dir, manifest, stage="generation", arms=generation_arms, status="FAILED", reason=result["reason"], config={"protocol": "wav2lip_time_map"})
        write_json(run_dir / "generation.json", result)
        return result

    generated: List[Dict[str, Any]] = []
    for sid, record in sorted(context.items()):
        row: Dict[str, Any] = {"sample_id": sid, "source_group": record.get("source_group"), "status": "FAILED", "arms": {}, "checks": {}}
        try:
            n_tokens = _parse_textgrid(Path(record["textgrids"]["N"]))
            t_tokens = _parse_textgrid(Path(record["textgrids"]["T"]))
            n_count = _read_pcm16(Path(record["N_audio"]))[0].size
            t_count = _read_pcm16(Path(record["T_audio"]))[0].size
            audio_map_for_identity = build_audio_knots(n_tokens, t_tokens, n_duration_samples=n_count, t_duration_samples=t_count)
            if audio_map_for_identity.get("status") != "COMPLETE":
                raise RuntimeError("identity knot construction failed for %d: %s" % (sid, audio_map_for_identity.get("reason")))
            n_identity_source, n_identity_target = _identity_audio_knots(audio_map_for_identity["mapping"], "n", n_count)
            t_identity_source, t_identity_target = _identity_audio_knots(audio_map_for_identity["mapping"], "t", t_count)
            identity_n = _run_stretch_audio(run_dir, sid, "N_ID", Path(record["N_audio"]), rubberband, n_identity_source, n_identity_target)
            identity_t = _run_stretch_audio(run_dir, sid, "T_ID", Path(record["T_audio"]), rubberband, t_identity_source, t_identity_target)
            row["arms"]["N_ID"] = {"audio": identity_n.get("output"), "receipt": identity_n}
            row["arms"]["T_ID"] = {"audio": identity_t.get("output"), "receipt": identity_t}
            for arm, audio_path in (("N", Path(record["N_audio"])), ("T", Path(record["T_audio"])), ("N_ID", Path(identity_n["output"])), ("T_ID", Path(identity_t["output"]))):
                rms_source = _wav_rms_db(Path(record["N_audio"] if arm.startswith("N") else record["T_audio"]))
                rms_output = _wav_rms_db(audio_path)
                if rms_source is None or rms_output is None or (arm in ("N_ID", "T_ID") and abs(rms_output - rms_source) > 0.5):
                    raise RuntimeError("identity RMS equivalence failed for %d/%s" % (sid, arm))
                render = _render_wav2lip(run_dir, record, arm, audio_path)
                row["arms"][arm] = {**row["arms"].get(arm, {}), "audio": str(audio_path.resolve()), "video": render.get("output"), "render": render}
                row["checks"][arm + "_rms_db"] = {"source": rms_source, "output": rms_output, "difference_db": None if rms_source is None or rms_output is None else rms_output - rms_source}
                row["checks"][arm + "_render"] = "COMPLETE"
            row["status"] = "COMPLETE"
        except Exception as exc:
            row["reason"] = "%s: %s" % (type(exc).__name__, exc)
        generated.append(row)
        for arm in ("N", "T", "N_ID", "T_ID"):
            _append_cell(run_dir, {"stage": "generation", "id": str(sid), "source_group": record.get("source_group"), "arm": arm, "input_hash": _json_hash(record.get("input_hashes", {})), "config_hash": _json_hash({"protocol": "wav2lip_time_map", "arm": arm}), "status": "COMPLETE" if row.get("status") == "COMPLETE" and arm in row.get("arms", {}) else "FAILED", "reason": None if row.get("status") == "COMPLETE" and arm in row.get("arms", {}) else row.get("reason", "ARM_NOT_COMPLETE"), "artifacts": [str((run_dir / "generation.json").resolve())]})

    complete = [item for item in generated if item.get("status") == "COMPLETE"]
    generated_audio_rows = [(int(item["sample_id"]), arm, Path(str(payload["audio"]))) for item in complete for arm, payload in item.get("arms", {}).items() if arm in ("N_ID", "T_ID")]
    mfa_qc = _generation_mfa_qc(run_dir, context, generated_audio_rows) if generated_audio_rows else {"status": "NOT_RUN"}
    baseline: Dict[str, Any] = {"status": "NOT_ESTIMABLE", "records": []}
    if len(complete) == len(generated):
        # Score all four baseline arms with the exact audio used to render the
        # corresponding video.  A separate worker persists matrices and
        # exposes the fixed 50-row Sync-C measurability gate.
        for row in complete:
            sid = int(row["sample_id"])
            scored: Dict[str, Any] = {}
            for arm in ("N", "T", "N_ID", "T_ID"):
                scored[arm] = _syncnet_score(run_dir, sid, arm, Path(row["arms"][arm]["video"]), Path(row["arms"][arm]["audio"]))
            row["syncnet"] = scored
        for row in complete:
            metrics = row["syncnet"]
            common_rows = min(int(metrics[arm]["feature_count"]["matrix"]) for arm in ("N", "T", "N_ID", "T_ID"))
            row["sync_c"] = {arm: _score_c_on_support(metrics[arm], common_rows) for arm in ("N", "T", "N_ID", "T_ID")}
            row["checks"]["sync_common_rows"] = common_rows
            row["checks"]["sync_time_support"] = {arm: {"matrix_rows": int(metrics[arm]["feature_count"]["matrix"]), "first_pts_s": metrics[arm].get("visual_meta", {}).get("first_pts_s"), "fps": metrics[arm].get("visual_meta", {}).get("fps")} for arm in ("N", "T", "N_ID", "T_ID")}
            # Reuse only the frozen native mapping and independently extract
            # the newly rendered visual streams.  No SyncNet lag is used to
            # align these events.
            native_row = next((item for item in native.get("records", []) if str(item.get("source_group")) == str(row.get("source_group")) and item.get("status") == "COMPLETE"), None)
            if native_row is not None:
                record = context[int(row["sample_id"])]
                ref_metric = _extract_and_score(run_dir, "generation_%d_R" % int(row["sample_id"]), Path(record["videos"]["R"]))
                arm_metrics = {arm: _extract_and_score(run_dir, "generation_%d_%s" % (int(row["sample_id"]), arm), Path(row["arms"][arm]["video"])) for arm in ("N", "T", "N_ID", "T_ID")}
                mapping_segments = native_row.get("mapping", {}).get("segments", [])
                ref_events = list(ref_metric.get("events", []))
                visual_e: Dict[str, Optional[float]] = {}
                for arm in ("N", "T", "N_ID", "T_ID"):
                    direction = "identity" if arm.startswith("N") else "t_to_n"
                    value, _denominator, _details = _aggregate_segment_distances(ref_events, list(arm_metrics[arm].get("events", [])), mapping_segments, direction)
                    visual_e[arm] = None if value is None else float(value * 1000.0)
                row["visual_E_ms"] = visual_e
        native_c = {str(item["source_group"]): float(item["sync_c"]["T"] - item["sync_c"]["N"]) for item in complete if item.get("sync_c", {}).get("N") is not None and item.get("sync_c", {}).get("T") is not None}
        identity_n_c = {str(item["source_group"]): float(item["sync_c"]["N_ID"] - item["sync_c"]["N"]) for item in complete if item.get("sync_c", {}).get("N_ID") is not None and item.get("sync_c", {}).get("N") is not None}
        identity_t_c = {str(item["source_group"]): float(item["sync_c"]["T_ID"] - item["sync_c"]["T"]) for item in complete if item.get("sync_c", {}).get("T_ID") is not None and item.get("sync_c", {}).get("T") is not None}
        bv_w = {str(item["source_group"]): float(item["visual_E_ms"]["N"] - item["visual_E_ms"]["T"]) for item in complete if item.get("visual_E_ms", {}).get("N") is not None and item.get("visual_E_ms", {}).get("T") is not None}
        identity_n_e = {str(item["source_group"]): float(item["visual_E_ms"]["N_ID"] - item["visual_E_ms"]["N"]) for item in complete if item.get("visual_E_ms", {}).get("N_ID") is not None and item.get("visual_E_ms", {}).get("N") is not None}
        identity_t_e = {str(item["source_group"]): float(item["visual_E_ms"]["T_ID"] - item["visual_E_ms"]["T"]) for item in complete if item.get("visual_E_ms", {}).get("T_ID") is not None and item.get("visual_E_ms", {}).get("T") is not None}
        baseline = {"status": "COMPLETE", "records": complete, "native_delta_c": bootstrap_mean(native_c, draws=BOOTSTRAP_DRAWS, alpha=0.05), "BV_W": bootstrap_mean(bv_w, draws=BOOTSTRAP_DRAWS, alpha=0.05), "identity_n_delta_c": bootstrap_mean(identity_n_c, draws=BOOTSTRAP_DRAWS, alpha=0.05), "identity_t_delta_c": bootstrap_mean(identity_t_c, draws=BOOTSTRAP_DRAWS, alpha=0.05), "identity_n_delta_E_ms": bootstrap_mean(identity_n_e, draws=BOOTSTRAP_DRAWS, alpha=0.05), "identity_t_delta_E_ms": bootstrap_mean(identity_t_e, draws=BOOTSTRAP_DRAWS, alpha=0.05), "valid_source_count": len(native_c)}
        baseline["gate"] = evaluate_wav2lip_baseline_gate(baseline, mfa_qc_status=str(mfa_qc.get("status")))
    else:
        baseline["gate"] = {"status": "WAV2LIP_BASELINE_UNCONFIRMED", "reason": "NOT_ALL_FOUR_BASELINE_ARMS_COMPLETE"}

    replacement: Dict[str, Any] = {"status": "NOT_TESTED", "gate": {"status": "NOT_ENTERED"}}
    if baseline.get("gate", {}).get("status") == "WAV2LIP_BASELINE_CONFIRMED":
        replacement_rows: List[Dict[str, Any]] = []
        tnat_audio_rows: List[Tuple[int, str, Path]] = []
        rv_values: Dict[str, float] = {}
        rc_values: Dict[str, float] = {}
        rv_id_values: Dict[str, float] = {}
        rc_id_values: Dict[str, float] = {}
        for row in complete:
            sid = int(row["sample_id"]); record = context[sid]
            n_tokens = _parse_textgrid(Path(record["textgrids"]["N"])); t_tokens = _parse_textgrid(Path(record["textgrids"]["T"]))
            n_count = _read_pcm16(Path(record["N_audio"]))[0].size; t_count = _read_pcm16(Path(record["T_audio"]))[0].size
            knot_result = build_audio_knots(n_tokens, t_tokens, n_duration_samples=n_count, t_duration_samples=t_count)
            if knot_result.get("status") != "COMPLETE":
                raise RuntimeError("T_NAT knot construction failed for %d: %s" % (sid, knot_result.get("reason")))
            t_nat = _run_stretch_audio(run_dir, sid, "T_NAT", Path(record["T_audio"]), rubberband, knot_result["source_knots"], knot_result["target_knots"])
            render = _render_wav2lip(run_dir, record, "T_NAT", Path(t_nat["output"]))
            tnat_audio_rows.append((sid, "T_NAT", Path(t_nat["output"])))
            if sid == 151:
                render["repeat"] = _render_wav2lip(run_dir, record, "T_NAT", Path(t_nat["output"]), repeat=True)
            score_nat = _syncnet_score(run_dir, sid, "T_NAT_against_N", Path(render["output"]), Path(record["N_audio"]))
            baseline_row = next(item for item in complete if int(item["sample_id"]) == sid)
            score_n = baseline_row["syncnet"]["N"]
            score_nid = baseline_row["syncnet"]["N_ID"]
            common_rows = min(int(score_nat["feature_count"]["matrix"]), int(score_n["feature_count"]["matrix"]))
            common_id_rows = min(int(score_nat["feature_count"]["matrix"]), int(score_nid["feature_count"]["matrix"]))
            c_nat = _score_c_on_support(score_nat, common_rows)
            c_n = _score_c_on_support(score_n, common_rows)
            c_nid = _score_c_on_support(score_nid, common_id_rows)
            r_c = None if c_nat is None or c_n is None else float(c_nat - c_n)
            r_c_id = None if c_nat is None or c_nid is None else float(c_nat - c_nid)

            # T_NAT already uses the N sample clock.  Its visual events are
            # therefore compared by identity; only the historical T arm needs
            # the inverse T→N phone map.
            visual_r = _extract_and_score(run_dir, "generation_%d_R" % sid, Path(record["videos"]["R"]))
            visual_n = _extract_and_score(run_dir, "generation_%d_N" % sid, Path(baseline_row["arms"]["N"]["video"]))
            visual_nid = _extract_and_score(run_dir, "generation_%d_N_ID" % sid, Path(baseline_row["arms"]["N_ID"]["video"]))
            visual_tnat = _extract_and_score(run_dir, "generation_%d_T_NAT" % sid, Path(render["output"]))
            native_row = next((item for item in native.get("records", []) if str(item.get("source_group")) == str(record.get("source_group")) and item.get("status") == "COMPLETE"), None)
            e_n = e_nid = e_tnat = None
            if native_row is not None:
                segments = native_row.get("mapping", {}).get("segments", [])
                reference_events = list(visual_r.get("events", []))
                e_n, _den_n, _detail_n = _aggregate_segment_distances(reference_events, list(visual_n.get("events", [])), segments, "identity")
                e_nid, _den_nid, _detail_nid = _aggregate_segment_distances(reference_events, list(visual_nid.get("events", [])), segments, "identity")
                e_tnat, _den_tnat, _detail_tnat = _aggregate_segment_distances(reference_events, list(visual_tnat.get("events", [])), segments, "identity")
            r_v = None if e_n is None or e_tnat is None else float((e_n - e_tnat) * 1000.0)
            r_v_id = None if e_nid is None or e_tnat is None else float((e_nid - e_tnat) * 1000.0)
            if r_c is not None:
                rc_values[str(record["source_group"])] = r_c
            if r_c_id is not None:
                rc_id_values[str(record["source_group"])] = r_c_id
            if r_v is not None:
                rv_values[str(record["source_group"])] = r_v
            if r_v_id is not None:
                rv_id_values[str(record["source_group"])] = r_v_id
            replacement_rows.append({"sample_id": sid, "source_group": record["source_group"], "T_NAT": render, "syncnet": score_nat, "visual_E_ms": {"N": None if e_n is None else float(e_n * 1000.0), "N_ID": None if e_nid is None else float(e_nid * 1000.0), "T_NAT": None if e_tnat is None else float(e_tnat * 1000.0)}, "R_V_ms": r_v, "R_V_vs_N_ID_ms": r_v_id, "R_C": r_c, "R_C_vs_N_ID": r_c_id, "knots": knot_result})
        tnat_qc = _generation_mfa_qc(run_dir, context, tnat_audio_rows) if tnat_audio_rows else {"status": "NOT_RUN"}
        rv_decision = decide(rv_values, threshold=0.0, minimum_mean=20.0, min_groups=8, draws=BOOTSTRAP_DRAWS, alpha=PRIMARY_ALPHA)
        rc_decision = decide(rc_values, threshold=0.0, minimum_mean=0.0, min_groups=8, draws=BOOTSTRAP_DRAWS, alpha=PRIMARY_ALPHA)
        gate = evaluate_replacement_gate(rv_decision, rc_decision, rv_values, rc_values, mfa_qc_status=str(tnat_qc.get("status")), rv_id_values=rv_id_values, rc_id_values=rc_id_values)
        replacement = {"status": "COMPLETE", "records": replacement_rows, "mfa_qc": tnat_qc, "R_V": rv_decision, "R_C": rc_decision, "R_V_vs_N_ID": rv_id_values, "R_C_vs_N_ID": rc_id_values, "gate": gate}
    # Independent repeat renders are required for every arm that actually ran
    # for the first source.  They are never copied from the main output and
    # are retained even when a later baseline gate fails.
    repeat_row = next((item for item in generated if int(item["sample_id"]) == 151 and item.get("status") == "COMPLETE"), None)
    if repeat_row is not None:
        repeat_record = context[151]
        for arm, arm_payload in list(repeat_row.get("arms", {}).items()):
            try:
                repeat_audio = Path(str(arm_payload["audio"]))
                arm_payload.setdefault("repeat", {})["render"] = _render_wav2lip(run_dir, repeat_record, arm, repeat_audio, repeat=True)
            except Exception as exc:
                arm_payload.setdefault("repeat", {})["status"] = "FAILED"
                arm_payload["repeat"]["reason"] = "%s: %s" % (type(exc).__name__, exc)
    if baseline.get("gate", {}).get("status") == "WAV2LIP_BASELINE_CONFIRMED":
        replacement_status = replacement.get("status") == "COMPLETE"
        replacement_reason = None if replacement_status else replacement.get("gate", {}).get("status", "REPLACEMENT_NOT_RUN")
        for record in manifest.get("main_records", []):
            _append_cell(run_dir, {"stage": "generation", "id": str(record.get("sample_id")), "source_group": record.get("source_group"), "arm": "T_NAT", "input_hash": _json_hash(record.get("input_hashes", {})), "config_hash": _json_hash({"protocol": "wav2lip_time_map", "arm": "T_NAT"}), "status": "COMPLETE" if replacement_status else "FAILED", "reason": replacement_reason, "artifacts": [str((run_dir / "generation.json").resolve())]})
    else:
        _append_main_stage_cells(run_dir, manifest, stage="generation", arms=("T_NAT",), status="SKIPPED_BY_GATE", reason="WAV2LIP_BASELINE_UNCONFIRMED", config={"protocol": "wav2lip_time_map"})
    result = {"protocol": protocol, "execution": "COMPLETE" if len(complete) == len(generated) else "PARTIAL", "science": "SUPPORTED" if replacement.get("gate", {}).get("status") == "REPLACEMENT_SUPPORT" else "NO_CLEAR_SUPPORT", "rubberband": str(rubberband), "rubberband_receipt": rubberband_receipt, "records": generated, "mfa_qc": mfa_qc, "baseline": baseline, "replacement": replacement, "gate": baseline.get("gate", {})}
    write_json(run_dir / "generation.json", _safe(result))
    return result


def _make_figures(run_dir: Path, calibration: Mapping[str, Any], native: Mapping[str, Any], generation: Optional[Mapping[str, Any]] = None) -> List[str]:
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return paths
    records = calibration.get("records", [])
    if records:
        fig, ax = plt.subplots(figsize=(7, 4))
        for record in records:
            summary = record.get("summary", {})
            for arm, item in summary.get("arms", {}).items():
                shifts = [float(value["time_s"]) - float(summary.get("reference_events", [])[index]["time_s"]) for index, value in enumerate(item.get("observed_events", [])[: len(summary.get("reference_events", []))])]
                if shifts:
                    ax.plot(range(len(shifts)), np.asarray(shifts) * 1000.0, marker=".", label="%s:%s" % (record.get("sample_id"), arm))
        ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set(xlabel="event index", ylabel="observed - reference (ms)", title="visual timing calibration")
        if ax.lines:
            ax.legend(fontsize=6, ncol=2)
        path = figures / "calibration_event_offsets.png"
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
        paths.append(str(path))
    rows = [row for row in native.get("records", []) if row.get("status") == "COMPLETE" and row.get("delta_c") is not None and row.get("B_V_ms") is not None]
    if rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter([float(row["delta_c"]) for row in rows], [float(row["B_V_ms"]) for row in rows])
        for row in rows:
            ax.annotate(str(row["sample_id"]), (float(row["delta_c"]), float(row["B_V_ms"])), fontsize=6)
        ax.axhline(0.0, color="black", linewidth=0.5); ax.axvline(0.0, color="black", linewidth=0.5)
        ax.set(xlabel="historical ΔSync-C", ylabel="B_V (ms)", title="visual timing and Sync-C")
        path = figures / "bv_delta_c.png"
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
        paths.append(str(path))
    # When the native stage is admitted, retain an event-level view rather
    # than only a scalar scatter.  T events are put on the frozen N clock using
    # the already recorded MFA map; no visual or SyncNet alignment is done.
    native_rows = [row for row in native.get("records", []) if row.get("status") == "COMPLETE"]
    if native_rows:
        fig, axes = plt.subplots(max(1, len(native_rows)), 1, figsize=(8, max(2.5, 1.8 * len(native_rows))), squeeze=False)
        for axis, row in zip(axes[:, 0], native_rows):
            sid = int(row["sample_id"])
            axis.set_title("sample %d; missing/extra remain visible in event counts" % sid, fontsize=8)
            for arm, marker, y in (("R", "o", 2.0), ("N", "x", 1.0)):
                event_path = run_dir / "events" / ("native_%d_%s.json" % (sid, arm))
                events = read_json(event_path).get("events", []) if event_path.is_file() else []
                axis.scatter([float(item["time_s"]) for item in events], [y] * len(events), marker=marker, label=arm, s=18)
            event_path = run_dir / "events" / ("native_%d_T.json" % sid)
            events = read_json(event_path).get("events", []) if event_path.is_file() else []
            mapped = [map_time(float(item["time_s"]), row.get("mapping", {}).get("segments", []), direction="t_to_n") for item in events]
            mapped = [float(item) for item in mapped if item is not None]
            axis.scatter(mapped, [0.0] * len(mapped), marker="^", label="T→N", s=18)
            axis.set_yticks([0.0, 1.0, 2.0]); axis.set_yticklabels(["T→N", "N", "R"]); axis.grid(axis="x", alpha=0.25)
            if axis is axes[0, 0]: axis.legend(fontsize=7, ncol=3)
        axes[-1, 0].set_xlabel("time on natural clock (s)")
        path = figures / "native_event_timelines.png"
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
        paths.append(str(path))
    # Generation is conditional.  If it runs, preserve a compact five-arm
    # visual/Sync-C panel in addition to the raw per-arm receipts.
    generation_rows = [row for row in (generation or {}).get("records", []) if row.get("status") == "COMPLETE" and row.get("syncnet")]
    if generation_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        labels = [str(row.get("sample_id")) for row in generation_rows]
        for arm in ("N", "T", "N_ID", "T_ID"):
            values = [row.get("visual_E_ms", {}).get(arm) for row in generation_rows]
            axes[0].plot(labels, [np.nan if value is None else float(value) for value in values], marker=".", label=arm)
        axes[0].set_title("Wav2Lip visual event error (ms)"); axes[0].set_ylabel("E"); axes[0].legend(fontsize=7)
        for arm in ("N", "T", "N_ID", "T_ID"):
            values = [row.get("sync_c", {}).get(arm) for row in generation_rows]
            axes[1].plot(labels, [np.nan if value is None else float(value) for value in values], marker=".", label=arm)
        axes[1].set_title("Wav2Lip common-window Sync-C"); axes[1].legend(fontsize=7)
        path = figures / "generation_five_arm_metrics.png"
        fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
        paths.append(str(path))
    return paths


def report(run_dir: Path, manifest: Mapping[str, Any], vsr: Mapping[str, Any], calibration: Mapping[str, Any], native: Mapping[str, Any], generation: Mapping[str, Any], *, smoke: bool = False, protocol: Optional[str] = None) -> Dict[str, Any]:
    protocol = protocol or str(manifest.get("protocol", PROTOCOL))
    figures = _make_figures(run_dir, calibration, native, generation)
    diagnosis = read_json(run_dir / "diagnosis.json") if (run_dir / "diagnosis.json").is_file() else {"status": "NOT_RUN"}
    analysis = {"protocol": protocol, "smoke": bool(smoke), "diagnosis": {"status": diagnosis.get("status"), "science": diagnosis.get("science")}, "stages": {"audit": {"execution": manifest.get("execution"), "science": manifest.get("science")}, "vsr_audit": {"execution": vsr.get("execution"), "science": vsr.get("science")}, "calibration": {"execution": calibration.get("execution"), "science": calibration.get("science"), "gate": calibration.get("gate")}, "native": {"execution": native.get("execution"), "science": native.get("science"), "gate": native.get("gate")}, "generation": {"execution": generation.get("execution"), "science": generation.get("science"), "gate": generation.get("gate")}}, "figures": figures}
    write_json(run_dir / "analysis.json", _safe(analysis))
    lines = ["# TTS 嘴部动作时间校准与自然时间轴迁移实验", "", "协议: `%s`" % protocol, "", "## 固定分母与状态", "", "- 主分析固定分母: %d；本次运行: %s。" % (len(MAIN_IDS), "smoke" if smoke else "正式队列"), "- audit: execution=%s, science=%s。" % (manifest.get("execution"), manifest.get("science")), "- diagnose: status=%s。" % diagnosis.get("status"), "- 旧中文 VSR 只读复算: execution=%s, science=%s。" % (vsr.get("execution"), vsr.get("science")), "- 纯视觉校准: execution=%s, science=%s, gate=%s。" % (calibration.get("execution"), calibration.get("science"), calibration.get("gate", {}).get("status")), "- 历史原生视觉时间比较: execution=%s, science=%s, gate=%s。" % (native.get("execution"), native.get("science"), native.get("gate", {}).get("status")), "- Wav2Lip/Rubber Band 迁移: execution=%s, science=%s, gate=%s。" % (generation.get("execution"), generation.get("science"), generation.get("gate", {}).get("status")), "", "## 解释边界", "", "本协议的事件是纯视觉开合低谷，不能命名为特定音素，也不是嘴型真值。校准失败、时钟无法绑定、依赖阻塞和科学阴性分别记录；没有把任何失败阶段写成阳性。", ""]
    if native.get("decision"):
        bv = native["decision"].get("BV", {})
        lines.extend(["## 主要数值", "", "- B_V 分组均值: %s ms；98.333%% CI: %s。" % (bv.get("mean"), bv.get("ci")), "- reference 事件数和可测 source 数: %s / %s。" % (native.get("gate", {}).get("reference_event_count"), native.get("gate", {}).get("valid_source_count")), ""])
    if vsr.get("status") == "COMPLETE":
        metrics = vsr.get("metrics", {})
        lines.extend(["## 既有 Ditto-50 VSR 只读复算", "", "- pair_count=%s；G 均值=%s；Gmatched 均值=%s；历史 ΔSync-C 均值=%s（正向 %s/%s）。" % (vsr.get("pair_count"), metrics.get("g", {}).get("mean"), metrics.get("gmatched", {}).get("mean"), metrics.get("delta_c", {}).get("mean"), metrics.get("delta_c", {}).get("positive_count"), vsr.get("pair_count")), "- 该结果仍为 `EXPLORATORY_ONLY`，是内容辨识支线，不是视觉动作时间真值，也不产生英文 VSR 分数。", ""])
    lines.extend(["## 固定队列与停止原因", "", "- 校准记录: %d 条，规则要求通过 %d 条；实际通过 %d 条。" % (len(calibration.get("records", [])), int(calibration.get("gate", {}).get("required_passed", 0)), int(calibration.get("gate", {}).get("passed_records", 0)))])
    for row in calibration.get("records", []):
        lines.append("- calibration %s: %s%s" % (row.get("sample_id"), row.get("status"), ("；" + str(row.get("reason")) if row.get("reason") else "")))
    if native.get("reason"):
        lines.append("- native: %s" % native.get("reason"))
    if generation.get("reason"):
        lines.append("- generation: %s" % generation.get("reason"))
    lines.append("")
    if manifest.get("failures"):
        lines.extend(["## 输入/依赖问题", "", *["- %s" % item for item in manifest["failures"][:50]], ""])
    write_text(run_dir / "report.md", "\n".join(lines))
    return analysis


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), default=PROTOCOL)
    parser.add_argument("--stage", choices=("all", "audit", "diagnose", "vsr-audit", "calibrate", "native", "generation", "report", "validate"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    default_run = (REPO / "runs/tts_visual_timing_v2_smoke") if args.protocol == "tts_visual_timing_v2" and args.smoke else ((REPO / "runs/tts_visual_timing_v2") if args.protocol == "tts_visual_timing_v2" else (SMOKE_RUN if args.smoke else DEFAULT_RUN))
    run_dir = (args.run_dir or default_run).resolve()
    smoke = bool(args.smoke or run_dir.name.endswith("_smoke"))
    try:
        manifest = audit_inputs(run_dir, smoke=smoke, resume=args.resume, protocol=args.protocol) if args.stage in ("all", "audit", "diagnose", "vsr-audit", "calibrate", "native", "generation", "report") else read_json(run_dir / "manifest.json")
        if args.stage == "audit":
            return 0
        if args.protocol == "tts_visual_timing_v2" and args.stage in ("all", "diagnose", "vsr-audit", "calibrate", "native", "generation", "report"):
            diagnosis = diagnose(run_dir, manifest, protocol=args.protocol)
            if args.stage == "diagnose":
                return 0
        elif args.stage == "diagnose":
            diagnose(run_dir, manifest, protocol=args.protocol)
            return 0
        vsr = audit_vsr(run_dir, manifest) if args.stage in ("all", "vsr-audit", "calibrate", "native", "generation", "report") else read_json(run_dir / "vsr_audit.json")
        if args.stage == "vsr-audit":
            return 0
        calibration = calibrate(run_dir, manifest, smoke=smoke, protocol=args.protocol) if args.stage in ("all", "calibrate", "native", "generation", "report") else read_json(run_dir / "calibration.json")
        if args.stage == "calibrate":
            return 0
        native = compare_native(run_dir, manifest, calibration, smoke=smoke, protocol=args.protocol) if args.stage in ("all", "native", "generation", "report") else read_json(run_dir / "native.json")
        if args.stage == "native":
            return 0
        generation = run_generation(run_dir, manifest, calibration, native, smoke=smoke, protocol=args.protocol) if args.stage in ("all", "generation", "report") else read_json(run_dir / "generation.json")
        if args.stage == "generation":
            return 0
        if args.stage == "validate":
            validation = _run_command((sys.executable, str(Path(__file__).with_name("check_tts_visual_timing.py")), "--run-dir", str(run_dir)), cwd=REPO, timeout=3600.0)
            if validation["stdout"]:
                print(validation["stdout"], end="")
            if validation["returncode"] != 0:
                print(validation["stderr"], file=sys.stderr, end="")
                return 2
            (run_dir / "fatal_error.json").unlink(missing_ok=True)
            return 0
        report(run_dir, manifest, vsr, calibration, native, generation, smoke=smoke, protocol=args.protocol)
        if args.stage == "report":
            (run_dir / "fatal_error.json").unlink(missing_ok=True)
            return 0
        validation = _run_command((sys.executable, str(Path(__file__).with_name("check_tts_visual_timing.py")), "--run-dir", str(run_dir)), cwd=REPO, timeout=3600.0)
        if validation["returncode"] != 0:
            print(validation["stdout"], end="")
            print(validation["stderr"], file=sys.stderr, end="")
            return 2
        (run_dir / "fatal_error.json").unlink(missing_ok=True)
        return 0
    except Exception as exc:
        write_json(run_dir / "fatal_error.json", {"status": "ERROR", "type": type(exc).__name__, "reason": str(exc)})
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
