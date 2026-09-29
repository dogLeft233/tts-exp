"""Parent binding and diagnostic candidate materialization."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..mfa_linear_real_video_sync.model import build_waveform_model, module_state_sha256
from ..mfa_linear_real_video_sync.protocol import load_mfa_linear_waveform, write_json_once
from ..mfa_linear_real_video_sync.syncnet_loss import log_mel_trust_components, waveform_qc
from ..mfa_linear_sync_existing_data.protocol import ExistingDataProtocolError
from ..mfa_linear_sync_existing_data.validate import validate_manifest
from .config import (
    DIAGNOSTIC_PARENT_INVALID,
    DIAGNOSTIC_MEL_LIMIT,
    EVAL_RECORD_COUNT,
    TRAIN_RECORD_COUNT,
    DiagnosticConfig,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ExistingDataProtocolError(f"artifact must be an object: {path}")
    return payload


def validate_parent(parent_root: str | Path) -> dict[str, Any]:
    root = Path(parent_root).resolve()
    try:
        manifest_path = root / "01_data_lock/manifest.json"
        history_path = root / "02_training/history.json"
        checkpoint_path = root / "02_training/step100/model.pt"
        validation_path = root / "validation.json"
        decision_path = root / "decision.json"
        manifest_validation = validate_manifest(manifest_path)
        history = read_object(history_path)
        validation = read_object(validation_path)
        decision = read_object(decision_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ExistingDataProtocolError(f"{DIAGNOSTIC_PARENT_INVALID}: missing or corrupt parent") from error
    if history.get("status") != "complete" or history.get("steps") != 100:
        raise ExistingDataProtocolError(f"{DIAGNOSTIC_PARENT_INVALID}: training history is not complete")
    if validation.get("status") != "valid" or validation.get("artifact_graph_valid") is not True:
        raise ExistingDataProtocolError(f"{DIAGNOSTIC_PARENT_INVALID}: parent validation is not valid")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ExistingDataProtocolError(f"{DIAGNOSTIC_PARENT_INVALID}: step-100 state is missing")
    model = build_waveform_model(seed=20_260_903, device="cpu")
    model.load_state_dict(payload["state_dict"])
    model_hash = module_state_sha256(model)
    if payload.get("model_sha256") != model_hash or history.get("final_model_sha256") != model_hash:
        raise ExistingDataProtocolError(f"{DIAGNOSTIC_PARENT_INVALID}: step-100 model hash mismatch")
    manifest = read_object(manifest_path)
    if len(manifest.get("training", [])) != TRAIN_RECORD_COUNT or len(manifest.get("evaluation", [])) != EVAL_RECORD_COUNT:
        raise ExistingDataProtocolError(f"{DIAGNOSTIC_PARENT_INVALID}: parent denominator mismatch")
    if decision.get("config", {}).get("train_record_count") != TRAIN_RECORD_COUNT:
        raise ExistingDataProtocolError(f"{DIAGNOSTIC_PARENT_INVALID}: parent configuration mismatch")
    return {
        "status": "complete",
        "parent_root": str(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "history_path": str(history_path),
        "history_sha256": sha256_file(history_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "step100_model_sha256": model_hash,
        "validation_path": str(validation_path),
        "validation_sha256": sha256_file(validation_path),
        "decision_path": str(decision_path),
        "decision_sha256": sha256_file(decision_path),
        "training_record_count": TRAIN_RECORD_COUNT,
        "evaluation_record_count": EVAL_RECORD_COUNT,
        "manifest_validation_sha256": manifest_validation["manifest_sha256"],
    }


def load_adapter(parent_lock: Mapping[str, Any], *, device: str | torch.device) -> torch.nn.Module:
    checkpoint = Path(str(parent_lock["checkpoint_path"]))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_waveform_model(seed=20_260_903, device=device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = _FrozenAdapter(model)
    if module_state_sha256(adapter) != str(parent_lock["step100_model_sha256"]):
        raise ExistingDataProtocolError(f"{DIAGNOSTIC_PARENT_INVALID}: loaded adapter hash mismatch")
    return adapter


class _FrozenAdapter(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._model = model
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if not isinstance(waveform, torch.Tensor) or tuple(waveform.shape) != (1, 1, 61_440):
            raise ExistingDataProtocolError("NATURAL_OR_SIDE_CHANNEL_LEAKAGE: exact waveform only")
        if not torch.is_floating_point(waveform):
            raise ExistingDataProtocolError("adapter waveform must be floating point")
        return self._model(waveform)

    def state_dict(self, *args: Any, **kwargs: Any) -> Mapping[str, torch.Tensor]:
        return self._model.state_dict(*args, **kwargs)


def materialize_candidate(
    row: Mapping[str, Any],
    adapter: torch.nn.Module,
    output_root: str | Path,
    *,
    mel_limit: float = DIAGNOSTIC_MEL_LIMIT,
) -> dict[str, Any]:
    sid = str(row["sample_id"])
    mfa, _ = load_mfa_linear_waveform(row["mfa_audio"], expected_sha256=str(row["mfa_audio_sha256"]))
    device = next(adapter.parameters()).device
    source = torch.from_numpy(np.asarray(mfa, dtype=np.float32)).to(device)[None, None]
    with torch.inference_mode():
        candidate = adapter(source)
    qc = waveform_qc(candidate, source, residual_bound=0.05)
    qc["mel_distance"] = float(log_mel_trust_components(candidate, source)["distance"].cpu())
    qc["original_mel_limit"] = 0.10
    qc["diagnostic_mel_limit"] = mel_limit
    qc["diagnostic_tolerance_pass"] = qc["mel_distance"] <= mel_limit
    qc["unchanged_checks_pass"] = bool(
        qc.get("finite") and qc.get("exact_shape") and qc.get("residual_bound_pass")
        and qc.get("pcm_saturation_pass")
    )
    path = Path(output_root).resolve() / "02_candidates" / sid / "candidate.wav"
    if not qc["unchanged_checks_pass"] or not qc["diagnostic_tolerance_pass"]:
        return {
            "status": "engineering_invalid",
            "sample_id": sid,
            "source_mfa_sha256": row["mfa_audio_sha256"],
            "qc": qc,
            "candidate_path": None,
        }
    from ..mfa_linear_real_video_sync.evaluate import write_pcm16_wav_once
    lock = write_pcm16_wav_once(path, candidate.detach().cpu())
    result = {
        "status": "complete",
        "sample_id": sid,
        "source_mfa_sha256": row["mfa_audio_sha256"],
        "candidate_wav_sha256": sha256_file(path),
        "candidate_path": str(path.resolve()),
        "model_input": ["mfa_linear_tts_waveform"],
        "natural_in_adapter": False,
        "qc": qc,
        "pcm_lock": lock,
    }
    write_json_once(path.parent / "candidate.json", result)
    return result
