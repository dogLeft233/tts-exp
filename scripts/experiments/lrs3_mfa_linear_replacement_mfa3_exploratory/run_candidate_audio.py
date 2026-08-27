from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.experiments.lrs3_mfa_linear_replacement.candidate_audio import (
    canonical_pcm_s16le,
    candidate_from_features,
    validate_wavlm_interface,
)
from scripts.experiments.lrs3_mfa_linear_replacement.mfa_alignment import (
    build_frame_mapping,
    trace_rows,
)
from scripts.experiments.lrs3_mfa_linear_replacement.run_stage01 import (
    file_sha256,
    write_json,
)
from scripts.wavlm_knn_vc_adapter import WavLMKNNVCAdapter
from .protocol import (
    CANONICAL_STAGE00_ID,
    CANONICAL_STAGE00_PATH,
    CANDIDATE_STAGE_ID,
    KNN_VC_REVISION,
    MFA_EXECUTABLE,
    MFA_ROOT_DIR,
    PROTOCOL_ID,
    VOCODER_CHECKPOINT,
    WAVLM_CHECKPOINT,
    load_json,
    validate_stage00,
)

STAGE_ID = CANDIDATE_STAGE_ID
SAMPLE_RATE = 16_000


def _assert_exploratory_gpu_ready() -> dict[str, Any]:
    hour = time.localtime().tm_hour
    if hour >= 23 or hour < 8:
        raise RuntimeError("GPU use is forbidden during the registered 23:00-08:00 window")
    query = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    result = subprocess.run(query, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed with exit {result.returncode}")
    gpus: list[dict[str, int]] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError("nvidia-smi GPU output is malformed")
        index, utilization, memory_used, memory_total = (int(float(field)) for field in fields)
        if utilization != 0:
            raise RuntimeError(f"GPU {index} is occupied: utilization={utilization}")
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
    active_processes = [line.strip() for line in processes.stdout.splitlines() if line.strip()]
    if active_processes:
        raise RuntimeError(f"GPU has active compute processes: {active_processes}")
    return {"checked": True, "gpus": gpus, "active_compute_processes": [], "local_hour": hour}


def _read_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"audio is not canonical 16 kHz mono PCM16: {path}")
        values = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32767.0
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"audio is empty or non-finite: {path}")
    return values


def _write_pcm16(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)


def _failure(output_dir: Path, error: str, *, gpu_gate: dict[str, Any] | None = None, partial_audio: list[str] | None = None) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "stage_id": STAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "error": error,
        "candidate_generation_started": gpu_gate is not None,
        "partial_audio_paths": partial_audio or [],
        "partial_audio_count": len(partial_audio or []),
        "gpu_gate": gpu_gate,
    }
    write_json(output_dir / "candidate_failure.json", payload)
    summary = {
        "schema_version": 1,
        "stage_id": STAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "blocked",
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "record_count": 0,
        "expected_record_count": None,
        "candidate_generation_started": gpu_gate is not None,
        "gpu_used": gpu_gate is not None,
        "media_access": {
            "fit_media_opened": True,
            "internal_dev_media_opened": False,
            "validation_media_opened": False,
            "test_media_opened": False,
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "decision.json", {
        "stage_id": STAGE_ID,
        "engineering_decision": "BLOCKED",
        "scientific_decision": "not_available",
        "next_allowed_stage": None,
        "reason": error,
    })
    return summary


def _validate_candidate_assets(stage00: dict[str, Any]) -> None:
    assets = stage00.get("assets", {})
    for key, path in (("wavlm", WAVLM_CHECKPOINT), ("vocoder", VOCODER_CHECKPOINT)):
        binding = assets.get(key)
        if not isinstance(binding, dict):
            raise ValueError(f"Stage00 candidate asset binding missing: {key}")
        if Path(str(binding.get("path"))).resolve() != path.resolve() or file_sha256(path) != binding.get("sha256"):
            raise ValueError(f"Stage00 candidate asset binding mismatch: {key}")


def run(stage00_path: Path, alignment_path: Path, output_dir: Path, *, local_knn_vc: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty exploratory candidate output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_gate: dict[str, Any] | None = None
    try:
        stage00 = load_json(stage00_path)
        validate_stage00(
            stage00_path,
            stage00,
            expected_stage_id=CANONICAL_STAGE00_ID,
            expected_path=CANONICAL_STAGE00_PATH,
        )
        _validate_candidate_assets(stage00)
        alignment = load_json(alignment_path)
        if alignment.get("protocol_id") != PROTOCOL_ID or alignment.get("clean_record_count", 0) < 24:
            raise ValueError("alignment screen does not contain at least 24 clean records")
        records = list(alignment.get("records", []))
        stage00_by_id = {str(row["sample_id"]): row for row in stage00["cohort"]["records"]}
        if any(str(row["sample_id"]) not in stage00_by_id for row in records):
            raise ValueError("alignment clean records are not a subset of Stage00 pool")
        if alignment.get("clean_sample_ids_sha256") != hashlib.sha256(
            json.dumps([str(row["sample_id"]) for row in records], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest():
            raise ValueError("alignment clean-record hash mismatch")
        gpu_gate = _assert_exploratory_gpu_ready()
        adapter = WavLMKNNVCAdapter.load_pretrained(
            device="cuda",
            source=local_knn_vc,
            revision=KNN_VC_REVISION,
        )
        validate_wavlm_interface(
            adapter.metadata(),
            revision=KNN_VC_REVISION,
            wavlm_checkpoint_sha256=file_sha256(WAVLM_CHECKPOINT),
            vocoder_checkpoint_sha256=file_sha256(VOCODER_CHECKPOINT),
        )
    except Exception as exc:
        return _failure(output_dir, str(exc), gpu_gate=gpu_gate)

    results: list[dict[str, Any]] = []
    try:
        for row in records:
            sample_id = str(row["sample_id"])
            source = stage00_by_id[sample_id]
            natural_values = _read_pcm16(Path(str(source["natural_audio"])))
            tts_values = _read_pcm16(Path(str(source["tts_audio"])))
            natural_features = adapter.extract(torch.from_numpy(natural_values).unsqueeze(0)).cpu().numpy()
            tts_features = adapter.extract(torch.from_numpy(tts_values).unsqueeze(0)).cpu().numpy()
            mapping, mapping_stats = build_frame_mapping(
                natural_features.shape[0],
                tts_features.shape[0],
                row["natural_tokens"],
                row["tts_tokens"],
            )
            candidate, metadata = candidate_from_features(
                natural_features=natural_features,
                tts_features=tts_features,
                mapping=mapping,
                natural_audio_samples=natural_values.size,
                vocode=lambda conditioning: adapter.vocode(torch.from_numpy(conditioning)).cpu().numpy(),
            )
            pcm, pcm_qc = canonical_pcm_s16le(candidate, natural_values.size)
            audio_path = output_dir / "audio" / f"{sample_id}.wav"
            _write_pcm16(audio_path, pcm)
            trace_path = output_dir / "traces" / f"{sample_id}.json"
            write_json(trace_path, {
                "schema_version": 1,
                "sample_id": sample_id,
                "mapping_stats": mapping_stats,
                "frames": trace_rows(mapping),
            })
            results.append({
                "sample_id": sample_id,
                "source_group": source["source_group"],
                "natural_audio_sha256": source["natural_audio_sha256"],
                "tts_audio_sha256": source["tts_audio_sha256"],
                "natural_textgrid_sha256": row["natural_textgrid_sha256"],
                "tts_textgrid_sha256": row["tts_textgrid_sha256"],
                "trace_sha256": file_sha256(trace_path),
                "candidate_audio": str(audio_path),
                "candidate_audio_sha256": file_sha256(audio_path),
                "candidate_samples": int(candidate.size),
                "natural_samples": int(natural_values.size),
                "mapping": mapping_stats,
                "candidate": metadata,
                "pcm_qc": pcm_qc,
                "model_interface": adapter.metadata(),
            })
    except Exception as exc:
        partial = sorted(str(path) for path in (output_dir / "audio").glob("*.wav"))
        return _failure(output_dir, str(exc), gpu_gate=gpu_gate, partial_audio=partial)

    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa3_exploratory_candidate_audio",
        "stage_id": STAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "record_count": len(results),
        "results": results,
        "gpu_gate": gpu_gate,
        "candidate_contract": stage00["candidate_contract"],
        "assets": stage00["assets"],
    }
    write_json(output_dir / "candidate_manifest.json", manifest)
    summary = {
        "schema_version": 1,
        "stage_id": STAGE_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "record_count": len(results),
        "candidate_generation_started": True,
        "gpu_used": True,
        "media_access": {
            "fit_media_opened": True,
            "internal_dev_media_opened": False,
            "validation_media_opened": False,
            "test_media_opened": False,
        },
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "decision.json", {
        "stage_id": STAGE_ID,
        "engineering_decision": "GO",
        "scientific_decision": "not_available",
        "next_allowed_stage": "03_wav2lip_strict_replacement_exploratory",
        "reason": "all clean MFA3 screen records generated exact-length candidate audio",
    })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage00", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-knn-vc", type=Path, default=Path.home() / ".cache/torch/hub/bshall_knn-vc_c616845c4e309e24d5927f15adbdf277a3d65358")
    args = parser.parse_args()
    result = run(args.stage00.resolve(), args.alignment.resolve(), args.output.resolve(), local_knn_vc=args.local_knn_vc.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["engineering_decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
