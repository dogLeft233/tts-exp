"""Fresh fixed-step trainer for the 200-record scale comparison."""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch import nn

from ..mfa_linear_real_video_sync.config import PrototypeConfig
from ..mfa_linear_real_video_sync.model import (
    assert_identity,
    assert_syncnet_frozen,
    build_waveform_model,
    module_state_sha256,
    trainable_parameters,
)
from ..mfa_linear_real_video_sync.train import (
    PreparedRecord,
    _history_row,
    compute_record_loss,
    configure_determinism,
    evaluate_record,
)
from .config import MIN_SUCCESS_RECORDS, TRAIN_RECORD_COUNT, TRAIN_STEPS, ScaleConfig


def train_scale(
    records: Sequence[PreparedRecord],
    syncnet: nn.Module,
    *,
    config: ScaleConfig = ScaleConfig(),
) -> dict[str, Any]:
    config.validate()
    if len(records) != TRAIN_RECORD_COUNT:
        raise ValueError(f"scale training requires exactly {TRAIN_RECORD_COUNT} records")
    prototype = PrototypeConfig()
    configure_determinism(prototype)
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_CUDA_UNAVAILABLE")
    device = torch.device("cuda")
    syncnet.to(device)
    assert_syncnet_frozen(syncnet)
    for record in records:
        record.validate()
        if any(
            tensor.device.type != "cuda"
            for tensor in (record.mfa_linear_tts_waveform, record.visual_embedding)
        ):
            raise ValueError("all prepared tensors must already be on CUDA")
    model = build_waveform_model(seed=config.seed, device=device)
    model.train()
    for record in records:
        assert_identity(model, record.mfa_linear_tts_waveform)
    initial_hash = module_state_sha256(model)
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    step0 = [evaluate_record(model, syncnet, record) for record in records]
    for baseline, record in zip(step0, records, strict=True):
        error = max(
            abs(a - b)
            for a, b in zip(
                baseline["curve"],
                [float(value) for value in record.pristine_curve.cpu()],
                strict=True,
            )
        )
        if error > config.q:
            raise RuntimeError(f"IDENTITY_FAILURE: step-0 curve differs by {error}")
    optimizer = torch.optim.AdamW(
        trainable_parameters(model, syncnet),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.epsilon,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, Any]] = []
    frozen_before = module_state_sha256(syncnet)
    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        step_rows: list[tuple[PreparedRecord, dict[str, Any]]] = []
        for record in records:
            losses = compute_record_loss(model, syncnet, record)
            (losses["total"] / TRAIN_RECORD_COUNT).backward()
            step_rows.append((record, losses))
        gradient = torch.nn.utils.clip_grad_norm_(
            trainable_parameters(model, syncnet), config.gradient_clip_norm
        )
        gradient_norm = float(gradient.detach().cpu())
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise RuntimeError("NO_CANDIDATE_GRADIENT: aggregate model gradient is invalid")
        optimizer.step()
        for record, losses in step_rows:
            history.append(_history_row(step, record, losses, gradient_norm))
            del losses
    model.eval()
    final = [evaluate_record(model, syncnet, record) for record in records]
    frozen_after = module_state_sha256(syncnet)
    if frozen_after != frozen_before:
        raise RuntimeError("FROZEN_STATE_MUTATION: SyncNet state changed")
    return {
        "stage": "SCALE_SHARED_200",
        "steps": TRAIN_STEPS,
        "record_order": [record.sample_id for record in records],
        "config": config.to_dict(),
        "initial_model_sha256": initial_hash,
        "initial_model_state": initial_state,
        "final_model_sha256": module_state_sha256(model),
        "syncnet_sha256_before": frozen_before,
        "syncnet_sha256_after": frozen_after,
        "history": history,
        "step0": step0,
        "final": final,
        "model": model,
        "optimizer": optimizer,
        "minimum_success_records": MIN_SUCCESS_RECORDS,
    }
