from __future__ import annotations

import hashlib
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.experiments.lrs3_mfa_linear_replacement.candidate_audio import (
    candidate_from_features,
    canonical_pcm_s16le,
    validate_wavlm_interface,
)
from scripts.wavlm_knn_vc_adapter import WavLMKNNVCAdapter

from . import config
from .dtw import DTWError, build_hard_dtw_mapping
from .protocol import (
    ProtocolError,
    load_json,
    sha256_file,
    write_json,
    write_json_once,
)


def gpu_gate() -> dict[str, Any]:
    hour = time.localtime().tm_hour
    if hour < 8 or hour >= 23:
        raise RuntimeError("GPU use is forbidden during the registered 23:00-08:00 window")
    query = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    result = subprocess.run(query, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed with exit {result.returncode}")
    gpus = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError("nvidia-smi GPU output is malformed")
        index, utilization, memory_used, memory_total = (int(float(field)) for field in fields)
        if utilization != 0:
            raise RuntimeError(f"GPU {index} is occupied: utilization={utilization}")
        gpus.append({"index": index, "utilization_gpu_percent": utilization, "memory_used_mib": memory_used, "memory_total_mib": memory_total})
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
    return {"checked": True, "local_hour": hour, "gpus": gpus, "active_compute_processes": []}


def _read_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != 16000 or handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ProtocolError(f"audio is not canonical 16 kHz mono PCM16: {path}")
        values = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32767.0
    if values.size == 0 or not np.isfinite(values).all():
        raise ProtocolError(f"audio is empty or non-finite: {path}")
    return values


def _write_pcm16(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm)


def _features_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.float32)
    return hashlib.sha256(array.astype("<f4", copy=False).tobytes()).hexdigest()


def _validate_stage00(stage00_dir: Path) -> dict[str, Any]:
    manifest_path = stage00_dir / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("protocol_id") != config.PROTOCOL_ID or manifest.get("status") != "complete":
        raise ProtocolError("Stage 00 protocol identity is invalid")
    if manifest.get("cohort", {}).get("ordered_sample_ids_sha256") != config.EXPECTED_COHORT_HASH:
        raise ProtocolError("Stage 00 cohort hash is invalid")
    if int(manifest.get("cohort", {}).get("record_count", -1)) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("Stage 00 cohort count is invalid")
    if manifest.get("media_access", {}).get("fit_features_created") is not False or manifest.get("media_access", {}).get("fit_scores_created") is not False:
        raise ProtocolError("Stage 00 claims feature or score access")
    return manifest


def _load_existing_cell(cell_path: Path, audio_path: Path, trace_path: Path, expected_sample_id: str) -> dict[str, Any] | None:
    if not cell_path.exists() and not audio_path.exists() and not trace_path.exists():
        return None
    if not cell_path.is_file() or not audio_path.is_file() or not trace_path.is_file():
        raise ProtocolError(f"partial candidate cell cannot be resumed: {expected_sample_id}")
    row = load_json(cell_path)
    if str(row.get("sample_id")) != expected_sample_id or row.get("candidate_audio") != str(audio_path) or row.get("trace") != str(trace_path):
        raise ProtocolError(f"candidate cell identity changed: {expected_sample_id}")
    if row.get("candidate_audio_sha256") != sha256_file(audio_path) or row.get("trace_sha256") != sha256_file(trace_path):
        raise ProtocolError(f"candidate cell hash changed: {expected_sample_id}")
    return row


def _generate_cell(row: dict[str, Any], adapter: WavLMKNNVCAdapter, interface: dict[str, Any], output: Path) -> dict[str, Any]:
    sample_id = str(row["sample_id"])
    audio_path = output / "audio" / f"{sample_id}.wav"
    trace_path = output / "traces" / f"{sample_id}.json"
    cell_path = output / "cells" / f"{sample_id}.json"
    natural_values = _read_pcm16(Path(row["natural_audio"]["path"]))
    tts_values = _read_pcm16(Path(row["tts_audio"]["path"]))
    if natural_values.size != int(row["natural_samples"]):
        raise ProtocolError(f"natural sample count changed: {sample_id}")
    natural_features = adapter.extract(torch.from_numpy(natural_values).unsqueeze(0)).cpu().numpy()
    tts_features = adapter.extract(torch.from_numpy(tts_values).unsqueeze(0)).cpu().numpy()
    mapping, diagnostics, trace = build_hard_dtw_mapping(
        natural_features,
        tts_features,
        row["natural_tokens"],
        row["tts_tokens"],
        band_ratio=config.BAND_RATIO,
        frame_ownership_policy=config.DTW_FRAME_OWNERSHIP_POLICY,
    )
    candidate, candidate_metadata = candidate_from_features(
        natural_features=natural_features,
        tts_features=tts_features,
        mapping=mapping,
        natural_audio_samples=natural_values.size,
        vocode=lambda conditioning: adapter.vocode(torch.from_numpy(conditioning)).cpu().numpy(),
    )
    pcm, pcm_qc = canonical_pcm_s16le(candidate, natural_values.size)
    _write_pcm16(audio_path, pcm)
    trace_payload = {
        **trace,
        "sample_id": sample_id,
        "source_group": row["source_group"],
        "natural_feature_sha256": _features_sha256(natural_features),
        "tts_feature_sha256": _features_sha256(tts_features),
        "natural_audio_sha256": row["natural_audio"]["sha256"],
        "tts_audio_sha256": row["tts_audio"]["sha256"],
        "natural_values_in_conditioning": False,
    }
    write_json(trace_path, trace_payload)
    cell = {
        "schema_version": 1,
        "sample_id": sample_id,
        "source_group": row["source_group"],
        "natural_audio_sha256": row["natural_audio"]["sha256"],
        "tts_audio_sha256": row["tts_audio"]["sha256"],
        "natural_textgrid_sha256": row["natural_textgrid"]["sha256"],
        "tts_textgrid_sha256": row["tts_textgrid"]["sha256"],
        "candidate_audio": str(audio_path),
        "candidate_audio_sha256": sha256_file(audio_path),
        "candidate_samples": int(candidate.size),
        "natural_samples": int(natural_values.size),
        "trace": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "mapping": diagnostics,
        "candidate": candidate_metadata,
        "pcm_qc": pcm_qc,
        "model_interface": interface,
        "natural_values_in_conditioning": False,
    }
    write_json_once(cell_path, cell)
    return cell


def run_candidates(stage00_dir: str | Path = config.STAGE00, output_dir: str | Path = config.STAGE01, *, local_knn_vc: str | Path = config.KNN_VC_SOURCE) -> dict[str, Any]:
    stage00_root = Path(stage00_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "candidate_manifest.json").exists():
        raise FileExistsError(f"candidate stage is already finalized: {output}")
    stage00 = _validate_stage00(stage00_root)
    rows = list(stage00["cohort"]["records"])
    gate = gpu_gate()
    adapter = WavLMKNNVCAdapter.load_pretrained(device="cuda", source=Path(local_knn_vc).resolve(), revision=config.KNN_VC_REVISION)
    interface = validate_wavlm_interface(
        adapter.metadata(),
        revision=config.KNN_VC_REVISION,
        wavlm_checkpoint_sha256=str(stage00["assets"]["wavlm_checkpoint"]["sha256"]),
        vocoder_checkpoint_sha256=str(stage00["assets"]["vocoder_checkpoint"]["sha256"]),
    )
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        sample_id = str(row["sample_id"])
        audio_path = output / "audio" / f"{sample_id}.wav"
        trace_path = output / "traces" / f"{sample_id}.json"
        cell_path = output / "cells" / f"{sample_id}.json"
        existing = _load_existing_cell(cell_path, audio_path, trace_path, sample_id)
        if existing is not None:
            results.append(existing)
            print(f"RESUME {index}/{len(rows)} {sample_id}", flush=True)
            continue
        try:
            cell = _generate_cell(row, adapter, interface, output)
        except (DTWError, FloatingPointError, OSError, RuntimeError, ValueError) as exc:
            failures.append({"sample_id": sample_id, "source_group": row["source_group"], "error_type": type(exc).__name__, "error": str(exc)})
            print(f"FAIL {index}/{len(rows)} {sample_id}: {exc}", flush=True)
            continue
        results.append(cell)
        print(f"OK {index}/{len(rows)} {sample_id} frames={cell['mapping']['natural_frame_count']} displacement={cell['mapping']['max_coordinate_displacement']:.3f}", flush=True)
    if failures:
        failure = {
            "schema_version": 1,
            "stage_id": "01_candidates",
            "protocol_id": config.PROTOCOL_ID,
            "status": "blocked",
            "engineering_decision": "BLOCKED",
            "scientific_decision": "not_available",
            "record_count": len(results),
            "expected_record_count": len(rows),
            "failures": failures,
            "completed_sample_ids": [str(row["sample_id"]) for row in results],
            "gpu_gate": gate,
        }
        write_json_once(output / "failure_ledger.json", failure)
        summary = {
            "schema_version": 1,
            "stage_id": "01_candidates",
            "protocol_id": config.PROTOCOL_ID,
            "status": "blocked",
            "engineering_decision": "BLOCKED",
            "scientific_decision": "not_available",
            "record_count": len(results),
            "expected_record_count": len(rows),
            "failure_count": len(failures),
            "candidate_generation_started": True,
            "gpu_used": True,
            "media_access": {
                "fit_media_opened": True,
                "fit_features_created": True,
                "fit_scores_created": False,
                "validation_media_opened": False,
                "test_media_opened": False,
            },
        }
        write_json_once(output / "summary.json", summary)
        write_json_once(output / "decision.json", {
            "schema_version": 1,
            "stage_id": "01_candidates",
            "protocol_id": config.PROTOCOL_ID,
            "engineering_decision": "BLOCKED",
            "scientific_decision": "not_available",
            "next_allowed_stage": None,
            "reason": "at least one frozen cohort record cannot produce a complete same-phone hard-DTW mapping",
        })
        return summary
    if [str(row["sample_id"]) for row in results] != [str(row["sample_id"]) for row in rows]:
        raise ProtocolError("candidate results are not in frozen cohort order")
    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa_dtw_comparison_candidates",
        "stage_id": "01_candidates",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "record_count": len(results),
        "cohort_sha256": stage00["cohort"]["ordered_sample_ids_sha256"],
        "gpu_gate": gate,
        "candidate_contract": stage00["candidate_contract"],
        "dtw_contract": stage00["dtw_contract"],
        "results": results,
    }
    write_json_once(output / "candidate_manifest.json", manifest)
    summary = {
        "schema_version": 1,
        "stage_id": "01_candidates",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "record_count": len(results),
        "candidate_generation_started": True,
        "gpu_used": True,
        "media_access": {
            "fit_media_opened": True,
            "fit_features_created": True,
            "fit_scores_created": False,
            "validation_media_opened": False,
            "test_media_opened": False,
        },
    }
    write_json_once(output / "summary.json", summary)
    write_json_once(output / "decision.json", {
        "schema_version": 1,
        "stage_id": "01_candidates",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "next_allowed_stage": "02_diagonal",
        "reason": "all 133 DTW candidates satisfy the common frozen WavLM/HiFi-GAN/PCM contract",
    })
    return summary
