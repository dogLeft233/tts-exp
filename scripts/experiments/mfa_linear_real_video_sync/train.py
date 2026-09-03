"""Fixed-step P1/P2 optimizer for the TTS-only waveform prototype."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .config import MODEL_RESIDUAL_SCALE, P1_STEPS, P2_STEPS, PrototypeConfig, Q
from .model import (
    assert_identity,
    assert_syncnet_frozen,
    build_waveform_model,
    module_state_sha256,
    trainable_parameters,
)
from .protocol import validate_training_record_fields
from .syncnet_loss import (
    audio_embeddings,
    log_mel_trust_components,
    official_syncnet_distance_curve,
    target_margin_loss_components,
    waveform_qc,
)


@dataclass(frozen=True)
class PreparedRecord:
    sample_id: str
    mfa_linear_tts_waveform: Tensor
    visual_embedding: Tensor
    target_offset: int
    pristine_curve: Tensor

    def training_payload(self) -> dict[str, Any]:
        payload = {
            "mfa_linear_tts_waveform": self.mfa_linear_tts_waveform,
            "visual_embedding": self.visual_embedding,
            "target_offset": self.target_offset,
            "pristine_curve": self.pristine_curve,
        }
        validate_training_record_fields(payload)
        return payload

    def validate(self) -> None:
        payload = self.training_payload()
        waveform = payload["mfa_linear_tts_waveform"]
        visual = payload["visual_embedding"]
        pristine = payload["pristine_curve"]
        if waveform.ndim != 3 or waveform.shape[:2] != (1, 1):
            raise ValueError("MFA-linear waveform must have shape [1,1,N]")
        if visual.ndim != 2 or visual.shape[0] != 91 or visual.requires_grad:
            raise ValueError("visual embedding must be detached [91,D]")
        if pristine.shape != (31,) or pristine.requires_grad:
            raise ValueError("pristine curve must be detached with shape [31]")
        if not -15 <= int(self.target_offset) <= 15:
            raise ValueError("target offset must be in [-15,+15]")
        for value in (waveform, visual, pristine):
            if not torch.isfinite(value.detach()).all():
                raise FloatingPointError("prepared record contains non-finite tensors")


def configure_determinism(config: PrototypeConfig) -> None:
    config.validate()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def compute_record_loss(model: nn.Module, syncnet: nn.Module, record: PreparedRecord) -> dict[str, Any]:
    record.validate()
    assert_syncnet_frozen(syncnet)
    source = record.mfa_linear_tts_waveform
    candidate = model(source)
    if candidate.shape != source.shape or not torch.isfinite(candidate.detach()).all():
        raise RuntimeError("NONFINITE_TRAINING: candidate shape/value invalid")
    embedding = audio_embeddings(syncnet, candidate, video_frame_count=96)
    if not embedding.requires_grad:
        raise RuntimeError("NO_CANDIDATE_GRADIENT: candidate audio embedding is detached")
    curve = official_syncnet_distance_curve(embedding, record.visual_embedding)
    sync = target_margin_loss_components(
        curve,
        pristine_curve=record.pristine_curve,
        target_offset=record.target_offset,
    )
    trust = log_mel_trust_components(candidate, source)
    total = sync["total"] + trust["total"]
    if not torch.isfinite(total.detach()):
        raise RuntimeError("NONFINITE_TRAINING: total loss is non-finite")
    qc = waveform_qc(candidate, source, residual_bound=MODEL_RESIDUAL_SCALE)
    return {
        "total": total,
        "sync": sync["total"],
        "target_violation": sync["target_violation"],
        "ranking_violation": sync["ranking_violation"],
        "trust": trust["total"],
        "mel_distance": trust["distance"],
        "curve": curve,
        "candidate": candidate,
        "audio_embedding": embedding,
        "qc": qc,
    }


def evaluate_record(model: nn.Module, syncnet: nn.Module, record: PreparedRecord) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        source = record.mfa_linear_tts_waveform
        candidate = model(source)
        embedding = audio_embeddings(syncnet, candidate, video_frame_count=96)
        curve = official_syncnet_distance_curve(embedding, record.visual_embedding)
        sync = target_margin_loss_components(
            curve,
            pristine_curve=record.pristine_curve,
            target_offset=record.target_offset,
        )
        trust = log_mel_trust_components(candidate, source)
        qc = waveform_qc(candidate, source, residual_bound=MODEL_RESIDUAL_SCALE)
    return {
        "sample_id": record.sample_id,
        "total_loss": float((sync["total"] + trust["total"]).cpu()),
        "sync_loss": float(sync["total"].cpu()),
        "trust_loss": float(trust["total"].cpu()),
        "mel_distance": float(trust["distance"].cpu()),
        "curve": [float(value) for value in curve.cpu()],
        "candidate": candidate.detach(),
        "qc": qc,
    }


def _history_row(step: int, record: PreparedRecord, losses: Mapping[str, Any], gradient_norm: float) -> dict[str, Any]:
    return {
        "step": step,
        "sample_id": record.sample_id,
        "total_loss": float(losses["total"].detach().cpu()),
        "sync_loss": float(losses["sync"].detach().cpu()),
        "target_violation": float(losses["target_violation"].detach().cpu()),
        "ranking_violation": float(losses["ranking_violation"].detach().cpu()),
        "trust_loss": float(losses["trust"].detach().cpu()),
        "mel_distance": float(losses["mel_distance"].detach().cpu()),
        "gradient_norm_before_clip": gradient_norm,
        "qc": dict(losses["qc"]),
    }


def train_fixed_steps(
    records: Sequence[PreparedRecord],
    syncnet: nn.Module,
    *,
    stage: str,
    config: PrototypeConfig = PrototypeConfig(),
) -> dict[str, Any]:
    config.validate()
    if stage not in {"P1_ONE_RECORD", "P2_SHARED_FOUR"}:
        raise ValueError("stage must be P1_ONE_RECORD or P2_SHARED_FOUR")
    expected_records = 1 if stage == "P1_ONE_RECORD" else 4
    expected_steps = P1_STEPS if stage == "P1_ONE_RECORD" else P2_STEPS
    if len(records) != expected_records:
        raise ValueError(f"{stage} requires exactly {expected_records} records")
    if config.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_CUDA_UNAVAILABLE")
    configure_determinism(config)
    device = torch.device("cuda")
    for record in records:
        record.validate()
    if any(
        tensor.device.type != "cuda"
        for record in records
        for tensor in (record.mfa_linear_tts_waveform, record.visual_embedding)
    ):
        raise ValueError("all prepared tensors must already be on CUDA")
    syncnet.to(device)
    assert_syncnet_frozen(syncnet)
    frozen_before = module_state_sha256(syncnet)
    model = build_waveform_model(seed=config.seed, device=device)
    model.train()
    for record in records:
        assert_identity(model, record.mfa_linear_tts_waveform)
    initial_model_hash = module_state_sha256(model)
    initial_model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    step0 = [evaluate_record(model, syncnet, record) for record in records]
    for baseline, record in zip(step0, records, strict=True):
        error = max(abs(a - b) for a, b in zip(baseline["curve"], [float(v) for v in record.pristine_curve.cpu()], strict=True))
        if error > Q:
            raise RuntimeError(f"IDENTITY_FAILURE: step-0 curve differs from locked pristine by {error}")
    model.train()
    optimizer = torch.optim.AdamW(
        trainable_parameters(model, syncnet),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )
    history = []
    for step in range(1, expected_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        rows = []
        for record in records:
            losses = compute_record_loss(model, syncnet, record)
            (losses["total"] / len(records)).backward()
            rows.append((record, losses))
        gradient = torch.nn.utils.clip_grad_norm_(
            trainable_parameters(model, syncnet), config.gradient_clip_norm
        )
        gradient_norm = float(gradient.detach().cpu())
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise RuntimeError("NO_CANDIDATE_GRADIENT: aggregate model gradient is invalid")
        optimizer.step()
        for record, losses in rows:
            history.append(_history_row(step, record, losses, gradient_norm))
    model.eval()
    final = [evaluate_record(model, syncnet, record) for record in records]
    frozen_after = module_state_sha256(syncnet)
    if frozen_after != frozen_before:
        raise RuntimeError("FROZEN_STATE_MUTATION: SyncNet state changed")
    return {
        "stage": stage,
        "steps": expected_steps,
        "record_order": [record.sample_id for record in records],
        "config": config.to_dict(),
        "initial_model_sha256": initial_model_hash,
        "initial_model_state": initial_model_state,
        "final_model_sha256": module_state_sha256(model),
        "syncnet_sha256_before": frozen_before,
        "syncnet_sha256_after": frozen_after,
        "history": history,
        "step0": step0,
        "final": final,
        "model": model,
        "optimizer": optimizer,
    }
