from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .audio import pcm16_to_float, read_pcm16, waveform_qc
from .common import canonical_json_sha256, file_sha256, write_self_hashed_json
from .protocol import ProtocolError


def wav2lip_mels(waveform: np.ndarray, wav2lip_root: Path = config.WAV2LIP_ROOT) -> np.ndarray:
    root = str(wav2lip_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import audio

    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if values.size < config.MIN_AUDIO_SAMPLES or not np.isfinite(values).all():
        raise ProtocolError("Wav2Lip mel input is empty or non-finite")
    result = np.asarray(audio.melspectrogram(values), dtype=np.float64)
    if result.ndim != 2 or result.shape[0] != 80 or not np.isfinite(result).all():
        raise ProtocolError(f"unexpected Wav2Lip mel shape: {result.shape}")
    return result


def directional_metrics(natural_mel: np.ndarray, mfa_mel: np.ndarray, candidate_mel: np.ndarray) -> dict[str, float]:
    natural_mel = np.asarray(natural_mel, dtype=np.float64)
    mfa_mel = np.asarray(mfa_mel, dtype=np.float64)
    candidate_mel = np.asarray(candidate_mel, dtype=np.float64)
    if natural_mel.shape != mfa_mel.shape or natural_mel.shape != candidate_mel.shape:
        raise ProtocolError("Wav2Lip mel shapes differ")
    direction = (mfa_mel - natural_mel).reshape(-1)
    movement = (candidate_mel - natural_mel).reshape(-1)
    denominator = max(float(np.dot(direction, direction)), 1e-12)
    progress = float(np.dot(movement, direction) / denominator)
    residual = movement - progress * direction
    return {
        "progress": progress,
        "orthogonal_ratio": float(np.linalg.norm(residual) / max(float(np.linalg.norm(direction)), 1e-12)),
        "mel_mae_to_natural": float(np.mean(np.abs(candidate_mel - natural_mel))),
        "mel_mae_to_mfa_linear": float(np.mean(np.abs(candidate_mel - mfa_mel))),
        "natural_to_mfa_linear_mel_l2": float(np.linalg.norm(direction)),
    }


def _audio_rows(audio_manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if audio_manifest.get("status") != "complete":
        raise ProtocolError("Stage 01 audio manifest is incomplete")
    result: dict[str, Mapping[str, Any]] = {}
    for row in audio_manifest.get("rows", []):
        sample_id = str(row.get("sample_id", ""))
        if sample_id in result or len(row.get("arms", [])) != len(config.ARMS):
            raise ProtocolError(f"audio row identity is incomplete: {sample_id}")
        result[sample_id] = row
    if len(result) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("audio row count is incomplete")
    return result


def _arm_rows(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for arm_row in row.get("arms", []):
        arm = str(arm_row.get("arm", ""))
        if arm not in config.ARMS or arm in result:
            raise ProtocolError(f"audio arm identity is invalid: {row.get('sample_id')}/{arm}")
        result[arm] = arm_row
    if tuple(result) != config.ARMS:
        raise ProtocolError(f"audio arm order is invalid: {row.get('sample_id')}")
    return result


def run_stage01_diagnostics(cohort: Mapping[str, Any], audio_manifest: Mapping[str, Any], output_stage: Path) -> dict[str, Any]:
    if config.STAGES["03_scores"].exists() and any(config.STAGES["03_scores"].iterdir()):
        raise ProtocolError("audio diagnostics must be frozen before SyncNet scores are read")
    audio_by_id = _audio_rows(audio_manifest)
    rows: list[dict[str, Any]] = []
    diagnostics_config = {
        "wav2lip_root": str(config.WAV2LIP_ROOT.resolve()),
        "mel": "official Wav2Lip audio.melspectrogram",
        "direction": "dot(mel(X)-mel(N), mel(M)-mel(N))/max(dot(d,d),1e-12)",
        "orthogonal": "norm(u-progress*d)/max(norm(d),1e-12)",
        "score_read_count": 0,
    }
    metric_config_sha256 = canonical_json_sha256(diagnostics_config)
    for index, record in enumerate(cohort.get("records", []), 1):
        sample_id = str(record["sample_id"])
        audio_row = audio_by_id.get(sample_id)
        if audio_row is None:
            raise ProtocolError(f"missing audio row: {sample_id}")
        arms = _arm_rows(audio_row)
        natural_int16, natural_meta = read_pcm16(Path(str(record["natural_audio"]["path"])))
        mfa_int16, mfa_meta = read_pcm16(Path(str(record["mfa_linear_audio"]["path"])))
        if natural_meta["pcm_sha256"] != record["natural_audio"]["sha256"] or mfa_meta["pcm_sha256"] != record["mfa_linear_audio"]["sha256"]:
            raise ProtocolError(f"Stage 01 source audio binding changed: {sample_id}")
        if natural_int16.size != mfa_int16.size or natural_int16.size != int(record["natural_sample_count"]):
            raise ProtocolError(f"Stage 01 source lengths differ: {sample_id}")
        natural_mel = wav2lip_mels(pcm16_to_float(natural_int16))
        mfa_mel = wav2lip_mels(pcm16_to_float(mfa_int16))
        direction = (mfa_mel - natural_mel).reshape(-1)
        if float(np.dot(direction, direction)) <= 1e-12:
            raise ProtocolError(f"natural/MFA-linear mel direction is degenerate: {sample_id}")
        arm_diagnostics: dict[str, Any] = {}
        for arm in config.ARMS:
            arm_row = arms[arm]
            path = Path(str(arm_row["output"]))
            candidate_int16, candidate_meta = read_pcm16(path)
            if candidate_meta["pcm_sha256"] != arm_row["output_sha256"] or candidate_int16.size != natural_int16.size:
                raise ProtocolError(f"candidate binding changed: {sample_id}/{arm}")
            candidate_mel = wav2lip_mels(pcm16_to_float(candidate_int16))
            arm_info: dict[str, Any] = {
                "arm": arm,
                "output": str(path),
                "output_sha256": arm_row["output_sha256"],
                "mel_shape": list(candidate_mel.shape),
                "waveform_qc": waveform_qc(pcm16_to_float(candidate_int16)),
                "mel_mae_to_natural": float(np.mean(np.abs(candidate_mel - natural_mel))),
                "mel_mae_to_mfa_linear": float(np.mean(np.abs(candidate_mel - mfa_mel))),
            }
            if arm == config.BRIDGE_ARM:
                arm_info.update(directional_metrics(natural_mel, mfa_mel, candidate_mel))
            arm_diagnostics[arm] = arm_info
        rows.append(
            {
                "sample_id": sample_id,
                "source_group": str(record["source_group"]),
                "natural_audio_sha256": natural_meta["pcm_sha256"],
                "mfa_linear_audio_sha256": mfa_meta["pcm_sha256"],
                "natural_sample_count": int(natural_int16.size),
                "natural_mel_shape": list(natural_mel.shape),
                "mfa_linear_mel_shape": list(mfa_mel.shape),
                "natural_to_mfa_linear_mel_mae": float(np.mean(np.abs(mfa_mel - natural_mel))),
                "arms": arm_diagnostics,
                "metric_config_sha256": metric_config_sha256,
            }
        )
        print(f"DIAGNOSTIC {index}/{config.EXPECTED_RECORD_COUNT} {sample_id}", flush=True)
    complete = len(rows) == config.EXPECTED_RECORD_COUNT
    result = {
        "schema_version": 1,
        "stage_id": "01_audio",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete" if complete else "blocked",
        "record_count": len(rows),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "arms": list(config.ARMS),
        "rows": rows,
        "metric_config": diagnostics_config,
        "metric_config_sha256": metric_config_sha256,
        "audio_manifest_sha256": file_sha256(config.STAGES["01_audio"] / "audio_manifest.json"),
        "frozen_before_scoring": True,
        "score_read_count": 0,
    }
    write_self_hashed_json(output_stage / "diagnostics.json", result)
    write_self_hashed_json(
        output_stage / "diagnostics_media_access.json",
        {
            "schema_version": 1,
            "stage_id": "01_audio",
            "protocol_id": config.PROTOCOL_ID,
            "score_read_count": 0,
            "sealed_media_accessed": False,
        },
    )
    if not complete:
        raise ProtocolError("Stage 01 diagnostics are incomplete")
    return result
