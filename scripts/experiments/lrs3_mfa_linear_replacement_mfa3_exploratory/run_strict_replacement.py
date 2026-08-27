from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.experiments.lrs3_mfa_linear_replacement.strict_mux import (
    file_sha256,
    mux_and_verify,
)

from .protocol import (
    CANONICAL_CANDIDATE_PATH,
    CANONICAL_CANDIDATE_STAGE_ID,
    CANONICAL_FACE_PREFLIGHT_PATH,
    CANONICAL_SYNCNET_PREFLIGHT_PATH,
    CANONICAL_SYNCNET_PREFLIGHT_SHA256,
    CANONICAL_FACE_READY_RENDER_PARENT,
    CANONICAL_STAGE00_ID,
    CANONICAL_STAGE00_PATH,
    PROTOCOL_ID,
    REPO,
    canonical_sha256,
    load_json,
    validate_stage00,
    write_json,
)

STAGE_ID = "03_strict_replacement_face_ready_retry7"
SAMPLE_RATE = 16_000
MIN_TRACK = 50
WAV2LIP_ROOT = REPO / "third_party/Wav2Lip"
WAV2LIP_PY = Path.home() / ".venvs/wav2lip/bin/python"
WAV2LIP_INFERENCE = WAV2LIP_ROOT / "inference.py"
WAV2LIP_CHECKPOINT = WAV2LIP_ROOT / "checkpoints/wav2lip_gan.pth"
SYNCNET_ROOT = REPO / "third_party/syncnet_python"
SYNCNET_PY = Path.home() / ".venvs/syncnet/bin/python"
SYNCNET_PIPELINE = SYNCNET_ROOT / "run_pipeline.py"
SYNCNET_SCORE = SYNCNET_ROOT / "run_syncnet.py"
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"
CELL_SPECS = {
    "G_N_E_N": {"video_arm": "natural", "audio_arm": "natural", "role": "authoritative_baseline"},
    "G_M_E_N": {"video_arm": "candidate", "audio_arm": "natural", "role": "authoritative_target"},
    "G_N_E_M": {"video_arm": "natural", "audio_arm": "candidate", "role": "scorer_audio_domain_diagnostic"},
    "G_M_E_M": {"video_arm": "candidate", "audio_arm": "candidate", "role": "native_pair_diagnostic"},
}
PRIMARY_CELLS = ("G_N_E_N", "G_M_E_N")


def _tool_binding(name: str) -> dict[str, str]:
    resolved = shutil.which(name)
    if not resolved:
        raise FileNotFoundError(name)
    path = Path(resolved).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def _asset(path: Path) -> dict[str, str]:
    path = path.absolute()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": file_sha256(path)}


def _bound_path(value: str | Path) -> Path:
    path = Path(str(value))
    return (path if path.is_absolute() else REPO / path).resolve()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    write_json(path, value)


def _run_logged(command: Sequence[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(
            [str(value) for value in command],
            cwd=str(cwd),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(result.returncode)


def _assert_gpu_ready() -> dict[str, Any]:
    hour = time.localtime().tm_hour
    if hour >= 23 or hour < 8:
        raise RuntimeError("GPU use is forbidden during the registered 23:00-08:00 window")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed with exit {result.returncode}")
    gpus: list[dict[str, int]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError("nvidia-smi GPU output is malformed")
        index, utilization, memory_used, memory_total = (int(float(field)) for field in fields)
        gpus.append({
            "index": index,
            "utilization_gpu_percent": utilization,
            "memory_used_mib": memory_used,
            "memory_total_mib": memory_total,
        })
    if not gpus:
        raise RuntimeError("nvidia-smi returned no GPUs")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if processes.returncode != 0:
        raise RuntimeError(f"nvidia-smi compute-process query failed with exit {processes.returncode}")
    active = [line.strip() for line in processes.stdout.splitlines() if line.strip()]
    if active:
        raise RuntimeError(f"GPU has active compute processes: {active}")
    return {
        "checked": True,
        "gpus": gpus,
        "active_compute_processes": [],
        "nonzero_utilization_without_compute_process": any(row["utilization_gpu_percent"] != 0 for row in gpus),
        "local_hour": hour,
    }


def _parse_syncnet(log_path: Path) -> dict[str, float | int]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    confidence = re.search(r"Confidence:\s+([-+0-9.eE]+)", text)
    distance = re.search(r"Min dist:\s+([-+0-9.eE]+)", text)
    offset = re.search(r"AV offset:\s+(-?\d+)", text)
    if confidence is None or distance is None:
        raise ValueError(f"SyncNet score fields missing: {log_path}")
    score: dict[str, float | int] = {
        "sync_c": float(confidence.group(1)),
        "sync_d": float(distance.group(1)),
        "av_offset": int(offset.group(1)) if offset else 0,
    }
    if not all(math.isfinite(float(score[key])) for key in ("sync_c", "sync_d", "av_offset")):
        raise ValueError(f"SyncNet score is non-finite: {log_path}")
    return score


def _source_bindings(
    stage00: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    selected_ids: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parents = stage00.get("parents")
    if not isinstance(parents, Mapping) or not isinstance(parents.get("source_manifest"), Mapping):
        raise ValueError("Stage00 source manifest binding is missing")
    source_path = Path(str(parents["source_manifest"]["path"])).resolve()
    if file_sha256(source_path) != str(parents["source_manifest"]["sha256"]):
        raise ValueError("canonical source manifest hash mismatch")
    source = load_json(source_path)
    source_by_id = {str(row["sample_id"]): row for row in source.get("records", [])}
    stage_records = {str(row["sample_id"]): row for row in stage00["cohort"]["records"]}
    results = list(candidate.get("results", []))
    if candidate.get("record_count") != len(results) or len(results) != 146:
        raise ValueError("candidate manifest does not contain exactly 146 results")
    expected_ids = [str(row["sample_id"]) for row in stage00["cohort"]["records"] if str(row["sample_id"]) in {str(item.get("sample_id")) for item in results}]
    actual_ids = [str(item.get("sample_id", "")) for item in results]
    if actual_ids != expected_ids or len(actual_ids) != len(set(actual_ids)):
        raise ValueError("candidate IDs are not an ordered Stage00 subset")
    if selected_ids is not None and (not selected_ids or not selected_ids <= set(actual_ids)):
        raise ValueError("selected face-ready IDs are not a non-empty candidate subset")
    records: list[dict[str, Any]] = []
    for item in results:
        sample_id = str(item["sample_id"])
        if selected_ids is not None and sample_id not in selected_ids:
            continue
        stage_row = stage_records[sample_id]
        source_row = source_by_id.get(sample_id)
        if source_row is None:
            raise ValueError(f"source manifest record missing: {sample_id}")
        if str(source_row.get("source_group")) != str(stage_row["source_group"]):
            raise ValueError(f"source group mismatch: {sample_id}")
        face = _bound_path(str(source_row["video_local_path"]))
        natural = Path(str(stage_row["natural_audio"])).resolve()
        candidate_audio = Path(str(item["candidate_audio"])).resolve()
        if not face.is_file() or file_sha256(face) != str(source_row["video_sha256"]):
            raise ValueError(f"face video hash mismatch: {sample_id}")
        if not natural.is_file() or file_sha256(natural) != str(stage_row["natural_audio_sha256"]):
            raise ValueError(f"natural audio hash mismatch: {sample_id}")
        if not candidate_audio.is_file() or file_sha256(candidate_audio) != str(item["candidate_audio_sha256"]):
            raise ValueError(f"candidate audio hash mismatch: {sample_id}")
        if int(item["natural_samples"]) != int(stage_row["natural_audio_samples"]) or int(item["candidate_samples"]) != int(stage_row["natural_audio_samples"]):
            raise ValueError(f"candidate length identity mismatch: {sample_id}")
        mapping = item.get("mapping", {})
        if int(mapping.get("speech_fallback_frames", -1)) != 0:
            raise ValueError(f"candidate contains speech fallback frames: {sample_id}")
        records.append({
            "sample_id": sample_id,
            "source_group": str(stage_row["source_group"]),
            "transcript": str(stage_row["transcript"]),
            "face_video": str(face),
            "face_video_sha256": str(source_row["video_sha256"]),
            "natural_audio": str(natural),
            "natural_audio_sha256": str(stage_row["natural_audio_sha256"]),
            "candidate_audio": str(candidate_audio),
            "candidate_audio_sha256": str(item["candidate_audio_sha256"]),
            "natural_samples": int(stage_row["natural_audio_samples"]),
            "candidate_mapping": mapping,
        })
    return {"path": str(source_path), "sha256": file_sha256(source_path)}, {
        "record_count": len(records),
        "ordered_sample_ids_sha256": canonical_sha256([row["sample_id"] for row in records]),
        "records": records,
    }


def build_protocol_manifest(
    stage00_path: Path,
    candidate_path: Path,
    face_preflight_path: Path | None = None,
    syncnet_preflight_path: Path | None = None,
) -> dict[str, Any]:
    if stage00_path.resolve() != CANONICAL_STAGE00_PATH.resolve():
        raise ValueError("Stage03 requires the canonical exploratory Stage00 path")
    if candidate_path.resolve() != CANONICAL_CANDIDATE_PATH.resolve():
        raise ValueError("Stage03 requires the canonical retry4 candidate path")
    stage00 = load_json(stage00_path)
    validate_stage00(stage00_path, stage00, expected_stage_id=CANONICAL_STAGE00_ID, expected_path=CANONICAL_STAGE00_PATH)
    candidate = load_json(candidate_path)
    if candidate.get("protocol_id") != PROTOCOL_ID or candidate.get("stage_id") != CANONICAL_CANDIDATE_STAGE_ID or candidate.get("status") != "complete":
        raise ValueError("candidate manifest is not the canonical complete retry4 artifact")
    if candidate.get("assets") != stage00.get("assets") or candidate.get("candidate_contract") != stage00.get("candidate_contract"):
        raise ValueError("candidate contract/assets do not match Stage00")
    face_binding = None
    syncnet_binding = None
    selected_ids = None
    if syncnet_preflight_path is not None and face_preflight_path is None:
        raise ValueError("SyncNet preflight requires the canonical face preflight")
    if face_preflight_path is not None:
        if face_preflight_path.resolve() != CANONICAL_FACE_PREFLIGHT_PATH.resolve():
            raise ValueError("Stage03 requires the canonical face preflight path")
        face_preflight = load_json(face_preflight_path)
        if face_preflight.get("protocol_id") != PROTOCOL_ID or face_preflight.get("stage_id") != "03_face_preflight_retry1" or face_preflight.get("status") != "complete" or face_preflight.get("engineering_decision") != "GO":
            raise ValueError("face preflight is not complete GO")
        face_parents = face_preflight.get("parents", {})
        for key, path in (("stage00", stage00_path), ("candidate", candidate_path)):
            binding = face_parents.get(key)
            if not isinstance(binding, Mapping) or Path(str(binding.get("path"))).resolve() != path.resolve() or str(binding.get("sha256")) != file_sha256(path):
                raise ValueError(f"face preflight {key} parent mismatch")
        face_ready_ids = [str(value) for value in face_preflight.get("face_ready_sample_ids", [])]
        selected_ids = set(face_ready_ids)
        if len(selected_ids) < 24 or len(selected_ids) != len(face_ready_ids):
            raise ValueError("face-ready cohort is smaller than the minimum exploratory size or contains duplicates")
        face_binding = {"path": str(face_preflight_path.resolve()), "sha256": file_sha256(face_preflight_path)}
        if syncnet_preflight_path is not None:
            if syncnet_preflight_path.resolve() != CANONICAL_SYNCNET_PREFLIGHT_PATH.resolve() or file_sha256(syncnet_preflight_path) != CANONICAL_SYNCNET_PREFLIGHT_SHA256:
                raise ValueError("Stage03 requires the canonical immutable SyncNet preflight artifact")
            syncnet_preflight = load_json(syncnet_preflight_path)
            if syncnet_preflight.get("protocol_id") != PROTOCOL_ID or syncnet_preflight.get("stage_id") != "03_syncnet_preflight_retry1" or syncnet_preflight.get("status") != "complete" or syncnet_preflight.get("engineering_decision") != "GO":
                raise ValueError("SyncNet preflight is not complete GO")
            sync_selection = syncnet_preflight.get("selection", {})
            if sync_selection.get("min_track") != MIN_TRACK or sync_selection.get("score_based_selection") is not False or sync_selection.get("syncnet_used_for_selection") is not False:
                raise ValueError("SyncNet preflight selection contract mismatch")
            sync_parent = syncnet_preflight.get("parents", {}).get("face_preflight")
            if not isinstance(sync_parent, Mapping) or Path(str(sync_parent.get("path"))).resolve() != face_preflight_path.resolve() or str(sync_parent.get("sha256")) != file_sha256(face_preflight_path):
                raise ValueError("SyncNet preflight face parent mismatch")
            sync_ids = [str(value) for value in syncnet_preflight.get("selected_sample_ids", [])]
            expected_sync_ids = [sample_id for sample_id in face_ready_ids if sample_id in set(sync_ids)]
            if syncnet_preflight.get("input_record_count") != len(face_ready_ids) or syncnet_preflight.get("selected_record_count") != 133 or syncnet_preflight.get("excluded_record_count") != 2 or len(sync_ids) != 133 or len(sync_ids) != len(set(sync_ids)) or sync_ids != expected_sync_ids or not set(sync_ids) <= selected_ids:
                raise ValueError("SyncNet preflight counts or IDs do not match the canonical structural subset")
            if syncnet_preflight.get("selected_sample_ids_sha256") != canonical_sha256(sync_ids):
                raise ValueError("SyncNet preflight selected ID hash mismatch")
            selected_ids = set(sync_ids)
            syncnet_binding = {"path": str(syncnet_preflight_path.resolve()), "sha256": file_sha256(syncnet_preflight_path)}
    source_binding, cohort = _source_bindings(stage00, candidate, selected_ids=selected_ids)
    assets = {
        "wav2lip_python": _asset(WAV2LIP_PY),
        "wav2lip_inference": _asset(WAV2LIP_INFERENCE),
        "wav2lip_checkpoint": _asset(WAV2LIP_CHECKPOINT),
        "syncnet_python": _asset(SYNCNET_PY),
        "syncnet_pipeline": _asset(SYNCNET_PIPELINE),
        "syncnet_score": _asset(SYNCNET_SCORE),
        "syncnet_model": _asset(SYNCNET_MODEL),
        "ffmpeg": _tool_binding("ffmpeg"),
        "ffprobe": _tool_binding("ffprobe"),
    }
    return {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa3_exploratory_strict_replacement_protocol",
        "protocol_id": PROTOCOL_ID,
        "stage_id": STAGE_ID,
        "status": "locked",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "parents": {
            "stage00": {"path": str(stage00_path.resolve()), "sha256": file_sha256(stage00_path)},
            "candidate": {"path": str(candidate_path.resolve()), "sha256": file_sha256(candidate_path)},
            "source_manifest": source_binding,
            "face_preflight": face_binding,
            "syncnet_preflight": syncnet_binding,
        },
        "selection": {
            "face_preflight_used": face_preflight_path is not None,
            "syncnet_preflight_used": syncnet_preflight_path is not None,
            "score_based_selection": False,
            "syncnet_used_for_selection": False,
            "source_order_preserved": True,
        },
        "assets": assets,
        "cohort": cohort,
        "cells": CELL_SPECS,
        "primary_cells": list(PRIMARY_CELLS),
        "diagnostic_cells": ["G_N_E_M", "G_M_E_M"],
        "syncnet": {"min_track": MIN_TRACK, "cache_reuse": False},
        "media_access": {
            "fit_media_opened": False,
            "internal_dev_media_opened": False,
            "validation_media_opened": False,
            "test_media_opened": False,
        },
    }


def _render(manifest: Mapping[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in manifest["cohort"]["records"]:
        sample_id = str(row["sample_id"])
        for arm, audio_key in (("natural", "natural_audio"), ("candidate", "candidate_audio")):
            _assert_gpu_ready()
            video = output_dir / "renders" / arm / f"{sample_id}.mp4"
            work = output_dir / "wav2lip_work" / arm / sample_id
            video.parent.mkdir(parents=True, exist_ok=True)
            (work / "temp").mkdir(parents=True, exist_ok=True)
            command = [
                str(WAV2LIP_PY), str(WAV2LIP_INFERENCE),
                "--checkpoint_path", str(WAV2LIP_CHECKPOINT),
                "--face", str(row["face_video"]), "--audio", str(row[audio_key]),
                "--outfile", str(video), "--face_det_batch_size", "4",
                "--wav2lip_batch_size", "4", "--nosmooth",
            ]
            log = output_dir / "logs" / "wav2lip" / arm / f"{sample_id}.log"
            return_code = _run_logged(command, cwd=work, log_path=log)
            if return_code != 0 or not video.is_file():
                raise RuntimeError(f"Wav2Lip failed for {arm}/{sample_id}: exit {return_code}")
            results.append({
                "sample_id": sample_id,
                "arm": arm,
                "video": str(video),
                "video_sha256": file_sha256(video),
                "command": command,
                "log": str(log),
            })
    return results


def _reuse_renders(manifest: Mapping[str, Any], parent_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    parent_dir = parent_dir.resolve()
    if parent_dir != CANONICAL_FACE_READY_RENDER_PARENT.resolve():
        raise ValueError("render reuse requires the canonical face-ready retry1 directory")
    manifest_path = parent_dir / "render_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    parent = load_json(manifest_path)
    parent_record_count = int(parent.get("record_count", -1))
    parent_render_count = int(parent.get("render_count", -1))
    renders = list(parent.get("renders", []))
    if parent.get("stage_id") != "03_strict_replacement_face_ready_retry1" or parent_record_count != 135 or parent_render_count != 270 or len(renders) != 270:
        raise ValueError("face-ready retry1 render manifest is incomplete or mismatched")
    face_binding = manifest.get("parents", {}).get("face_preflight")
    if not isinstance(face_binding, Mapping) or Path(str(face_binding.get("path"))).resolve() != CANONICAL_FACE_PREFLIGHT_PATH.resolve() or str(face_binding.get("sha256")) != file_sha256(CANONICAL_FACE_PREFLIGHT_PATH):
        raise ValueError("reused render parent is not bound to the canonical face preflight")
    face_preflight = load_json(CANONICAL_FACE_PREFLIGHT_PATH)
    face_ready_ids = [str(value) for value in face_preflight.get("face_ready_sample_ids", [])]
    if len(face_ready_ids) != 135 or len(face_ready_ids) != len(set(face_ready_ids)):
        raise ValueError("canonical face preflight does not contain exactly 135 unique IDs")
    parent_expected_keys = {
        (sample_id, arm)
        for sample_id in face_ready_ids
        for arm in ("natural", "candidate")
    }
    actual_keys = {(str(row.get("sample_id")), str(row.get("arm"))) for row in renders}
    if actual_keys != parent_expected_keys:
        raise ValueError("reused render IDs/arms do not match the complete canonical face-ready cohort")
    expected_keys = {
        (str(row["sample_id"]), arm)
        for row in manifest["cohort"]["records"]
        for arm in ("natural", "candidate")
    }
    if not expected_keys <= actual_keys:
        raise ValueError("reused render IDs/arms do not cover the frozen cohort")
    selected_renders: list[dict[str, Any]] = []
    for row in renders:
        key = (str(row.get("sample_id")), str(row.get("arm")))
        if key not in expected_keys:
            continue
        video = Path(str(row["video"])).resolve()
        if not video.is_file() or file_sha256(video) != str(row["video_sha256"]):
            raise ValueError(f"reused render hash mismatch: {video}")
        selected_renders.append(dict(row))
    if len(selected_renders) != len(expected_keys):
        raise ValueError("reused render coverage contains duplicate cohort keys")
    return selected_renders, {"path": str(manifest_path), "sha256": file_sha256(manifest_path)}


def _mux(manifest: Mapping[str, Any], output_dir: Path, renders: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    render_by_key = {(str(row["sample_id"]), str(row["arm"])): row for row in renders}
    results: list[dict[str, Any]] = []
    ffmpeg = str(manifest["assets"]["ffmpeg"]["path"])
    ffprobe = str(manifest["assets"]["ffprobe"]["path"])
    for row in manifest["cohort"]["records"]:
        sample_id = str(row["sample_id"])
        for cell, spec in CELL_SPECS.items():
            source_video = render_by_key[(sample_id, spec["video_arm"])]
            audio = row["natural_audio"] if spec["audio_arm"] == "natural" else row["candidate_audio"]
            output = output_dir / "mux" / cell / f"{sample_id}.mkv"
            output.parent.mkdir(parents=True, exist_ok=True)
            verification = mux_and_verify(
                source_video=str(source_video["video"]),
                expected_audio=audio,
                output_path=output,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
            )
            results.append({
                "sample_id": sample_id,
                "cell": cell,
                "role": spec["role"],
                "video_arm": spec["video_arm"],
                "audio_arm": spec["audio_arm"],
                "video": source_video["video"],
                "video_sha256": source_video["video_sha256"],
                "audio": audio,
                "audio_sha256": row[f"{spec['audio_arm']}_audio_sha256"],
                "muxed_file": str(output),
                "verification": verification,
            })
    return results


def _score(manifest: Mapping[str, Any], output_dir: Path, muxes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    syncnet_python = str(manifest["assets"]["syncnet_python"]["path"])
    syncnet_root = SYNCNET_ROOT.resolve()
    syncnet_model = str(manifest["assets"]["syncnet_model"]["path"])
    scores: list[dict[str, Any]] = []
    for row in muxes:
        sample_id = str(row["sample_id"])
        cell = str(row["cell"])
        reference = f"lrs3_mfa3_exploratory_{sample_id}_{cell}"
        sync_dir = output_dir / "syncnet" / cell / sample_id
        sync_dir.mkdir(parents=True, exist_ok=True)
        pipeline_log = output_dir / "logs" / "syncnet" / cell / f"{sample_id}.pipeline.log"
        _assert_gpu_ready()
        pipeline_rc = _run_logged([
            syncnet_python, str(SYNCNET_PIPELINE), "--videofile", str(row["muxed_file"]),
            "--reference", reference, "--data_dir", str(sync_dir), "--min_track", str(MIN_TRACK), "--overwrite",
        ], cwd=syncnet_root, log_path=pipeline_log)
        if pipeline_rc != 0:
            raise RuntimeError(f"SyncNet pipeline failed for {cell}/{sample_id}: exit {pipeline_rc}")
        score_log = output_dir / "logs" / "syncnet" / cell / f"{sample_id}.score.log"
        _assert_gpu_ready()
        score_rc = _run_logged([
            syncnet_python, str(SYNCNET_SCORE), "--videofile", str(row["muxed_file"]),
            "--reference", reference, "--data_dir", str(sync_dir), "--initial_model", syncnet_model,
        ], cwd=syncnet_root, log_path=score_log)
        if score_rc != 0:
            raise RuntimeError(f"SyncNet score failed for {cell}/{sample_id}: exit {score_rc}")
        parsed = _parse_syncnet(score_log)
        scores.append({**row, "cache_reused": False, "syncnet_model_sha256": manifest["assets"]["syncnet_model"]["sha256"], "min_track": MIN_TRACK, **parsed, "pipeline_log": str(pipeline_log), "score_log": str(score_log)})
    return scores


def _failure(output_dir: Path, error: str, *, manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "stage_id": STAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "error": error,
        "partial_render_count": len(list((output_dir / "renders").glob("*/*.mp4"))),
        "partial_mux_count": len(list((output_dir / "mux").glob("*/*.mkv"))),
        "partial_score_log_count": len(list((output_dir / "logs" / "syncnet").glob("*/*.score.log"))),
    }
    protocol_manifest = output_dir / "protocol_manifest.json"
    if manifest is not None and protocol_manifest.is_file():
        payload["protocol_manifest_sha256"] = file_sha256(protocol_manifest)
    _write_json(output_dir / "failure.json", payload)
    summary = {
        "schema_version": 1,
        "stage_id": STAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "render_count": payload["partial_render_count"],
        "mux_count": payload["partial_mux_count"],
        "score_count": payload["partial_score_log_count"],
        "sealed_media_opened": False,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "decision.json", {
        "stage_id": STAGE_ID,
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "next_allowed_stage": None,
        "reason": error,
    })
    return summary


def run(
    stage00_path: Path,
    candidate_path: Path,
    output_dir: Path,
    *,
    face_preflight_path: Path | None = None,
    syncnet_preflight_path: Path | None = None,
    render_parent_path: Path | None = None,
) -> dict[str, Any]:
    stage00_path = stage00_path.resolve()
    candidate_path = candidate_path.resolve()
    output_dir = output_dir.resolve()
    if face_preflight_path is not None:
        face_preflight_path = face_preflight_path.resolve()
    if syncnet_preflight_path is not None:
        syncnet_preflight_path = syncnet_preflight_path.resolve()
    if render_parent_path is not None:
        render_parent_path = render_parent_path.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty exploratory Stage03 output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] | None = None
    try:
        manifest = build_protocol_manifest(stage00_path, candidate_path, face_preflight_path, syncnet_preflight_path)
        record_count = int(manifest["cohort"]["record_count"])
        if render_parent_path is None:
            renders = _render(manifest, output_dir)
            render_parent_binding = None
        else:
            renders, render_parent_binding = _reuse_renders(manifest, render_parent_path)
        manifest["parents"]["render_parent"] = render_parent_binding
        _write_json(output_dir / "protocol_manifest.json", manifest)
        (output_dir / "protocol_manifest.sha256").write_text(file_sha256(output_dir / "protocol_manifest.json") + "\n", encoding="utf-8")
        _write_json(output_dir / "render_manifest.json", {"stage_id": STAGE_ID, "record_count": record_count, "render_count": len(renders), "renders": renders, "reused_parent": render_parent_binding})
        muxes = _mux(manifest, output_dir, renders)
        _write_json(output_dir / "mux_manifest.json", {"stage_id": STAGE_ID, "record_count": record_count, "mux_count": len(muxes), "muxes": muxes})
        scores = _score(manifest, output_dir, muxes)
        result = {
            **manifest,
            "status": "complete",
            "engineering_decision": "GO",
            "scientific_decision": "not_available",
            "renders": renders,
            "muxes": muxes,
            "scores": scores,
        }
        _write_json(output_dir / "replacement_manifest.json", result)
        summary = {
            "schema_version": 1,
            "stage_id": STAGE_ID,
            "protocol_id": PROTOCOL_ID,
            "status": "complete",
            "engineering_decision": "GO",
            "scientific_decision": "not_available",
            "record_count": record_count,
            "render_count": len(renders),
            "mux_count": len(muxes),
            "score_count": len(scores),
            "expected_render_count": record_count * 2,
            "expected_mux_count": record_count * 4,
            "expected_score_count": record_count * 4,
            "primary_cells_complete": list(PRIMARY_CELLS),
            "diagnostic_cells_complete": ["G_N_E_M", "G_M_E_M"],
            "sealed_media_opened": False,
        }
        _write_json(output_dir / "summary.json", summary)
        _write_json(output_dir / "decision.json", {
            "stage_id": STAGE_ID,
            "engineering_decision": "GO",
            "scientific_decision": "not_available",
            "next_allowed_stage": "04_statistics_exploratory",
            "reason": f"complete {record_count}-record four-cell strict replacement matrix with fresh official SyncNet scores",
        })
        return summary
    except Exception as exc:
        return _failure(output_dir, str(exc), manifest=manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage00", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--face-preflight", type=Path)
    parser.add_argument("--syncnet-preflight", type=Path)
    parser.add_argument("--render-parent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.stage00,
        args.candidate,
        args.output,
        face_preflight_path=args.face_preflight,
        syncnet_preflight_path=args.syncnet_preflight,
        render_parent_path=args.render_parent,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["engineering_decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
