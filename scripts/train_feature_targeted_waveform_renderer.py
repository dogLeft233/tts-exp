#!/usr/bin/env python3
"""Train the frozen Stage-1-targeted waveform renderer on train/valid only.

The default loss realizes a detached Stage-1 HuBERT layer-6 goal while
preserving natural layer-2 content, short-time energy, and a bounded waveform
residual.  A diagnostic oracle mode can instead use masked, MFA-aligned raw-TTS
layer-6 features as the realization supervision target.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import AdamW

from feature_targeted_waveform_renderer import (
    DEFAULT_STAGE2_CHANNELS,
    DEFAULT_STAGE2_CONDITIONING_DIM,
    DEFAULT_STAGE2_DILATIONS,
    DEFAULT_STAGE2_RESIDUAL_SCALE,
    FeatureTargetedWaveformRenderer,
    FrozenStage1Artifact,
    load_frozen_stage1,
)
from prepare_two_stage_hubert_manifest import file_sha256
from train_tts_aligned_feature_puller import (
    _load_stage1_hubert,
    _read_stage1_wave,
    extract_hubert_layers,
    load_stage1_records,
)
from hubert_feature_alignment import align_tts_features_to_natural
from tts_aligned_feature_puller import validate_hubert_interface
from waveform_hubert_enhancer import (
    DEFAULT_FRAME_STRIDE,
    DEFAULT_HIDDEN_SIZE,
    TARGET_SAMPLE_RATE,
    EncoderInterface,
    _finite_tensor,
)

SCHEMA_VERSION = 1
EXPERIMENT_TYPE = "feature_targeted_waveform_renderer"
CHECKPOINT_SELECTION_METRIC = "valid_realize_combined"
CHECKPOINT_SELECTION_POLICY = (
    "lowest finite valid realize_cosine plus realize_smooth_l1 among checkpoints "
    "with zero clipped output samples; ties select the earlier epoch"
)
TARGET_MODES = ("stage1_goal", "aligned_raw_tts_oracle")


def _validate_target_mode(target_mode: str) -> str:
    normalized = str(target_mode)
    if normalized not in TARGET_MODES:
        raise ValueError(f"target_mode must be one of {TARGET_MODES}")
    return normalized


@dataclass(frozen=True)
class Stage2LossWeights:
    realize_cosine: float = 1.0
    realize_smooth_l1: float = 1.0
    content_cosine: float = 0.25
    content_smooth_l1: float = 0.25
    energy: float = 0.10
    residual_l1: float = 0.005
    residual_smoothness: float = 0.005
    anti_clipping: float = 0.10
    smooth_l1_beta: float = 1.0

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in values.values()):
            raise ValueError("Stage 2 loss weights must be finite and non-negative")
        if self.smooth_l1_beta <= 0:
            raise ValueError("smooth_l1_beta must be positive")

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class TargetFirstSchedule:
    """Pre-registered warmup that delays content and energy preservation."""

    warmup_epochs: int = 1

    def __post_init__(self) -> None:
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be non-negative")

    def active_weights(self, weights: Stage2LossWeights, epoch: int) -> dict[str, float]:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        active = weights.as_dict()
        active.pop("smooth_l1_beta")
        if epoch < self.warmup_epochs:
            active["content_cosine"] = 0.0
            active["content_smooth_l1"] = 0.0
            active["energy"] = 0.0
        return active

    def as_dict(self) -> dict[str, int | str]:
        return {
            "warmup_epochs": int(self.warmup_epochs),
            "policy": "realize_and_residual_constraints_first_then_fixed_content_energy",
        }


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _finite_scalar(value: Tensor, name: str) -> Tensor:
    if value.ndim != 0 or not torch.isfinite(value):
        raise FloatingPointError(f"{name} must be a finite scalar")
    return value


def _masked_mean(value: Tensor, mask: Tensor, name: str) -> Tensor:
    if tuple(value.shape) != tuple(mask.shape):
        raise ValueError(f"{name} must have shape matching its frame mask")
    if not bool(mask.any()):
        raise ValueError(f"{name} requires at least one real frame")
    return value[mask].mean()


def _feature_terms(
    enhanced: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    smooth_l1_beta: float,
    name: str,
) -> tuple[Tensor, Tensor]:
    if enhanced.shape != target.shape or enhanced.ndim != 3 or enhanced.shape[-1] != DEFAULT_HIDDEN_SIZE:
        raise ValueError(f"{name} feature tensors must have matching [B,F,{DEFAULT_HIDDEN_SIZE}] shapes")
    if mask.ndim != 2 or tuple(mask.shape) != tuple(enhanced.shape[:2]) or mask.dtype != torch.bool:
        raise ValueError(f"{name} mask must be Boolean [B,F]")
    target = target.detach()
    cosine = _masked_mean(
        1.0 - F.cosine_similarity(enhanced, target, dim=-1, eps=1e-8),
        mask,
        f"{name} cosine",
    )
    smooth_l1 = _masked_mean(
        F.smooth_l1_loss(enhanced, target, beta=smooth_l1_beta, reduction="none").mean(dim=-1),
        mask,
        f"{name} Smooth-L1",
    )
    return _finite_scalar(cosine, f"{name} cosine"), _finite_scalar(smooth_l1, f"{name} Smooth-L1")


def _energy_envelope(waveform: Tensor) -> Tensor:
    if waveform.ndim != 3 or waveform.shape[1] != 1:
        raise ValueError("energy envelope requires waveform shape [B,1,N]")
    return torch.sqrt(
        F.avg_pool1d(waveform.square(), kernel_size=DEFAULT_FRAME_STRIDE, stride=DEFAULT_FRAME_STRIDE // 2, ceil_mode=True).clamp_min(1e-8)
    )


def compute_stage2_loss(
    output: Mapping[str, Tensor],
    natural_waveform: Tensor,
    *,
    active_weights: Mapping[str, float],
    smooth_l1_beta: float,
    realize_target: Tensor | None = None,
    realize_mask: Tensor | None = None,
    target_mode: str = "stage1_goal",
) -> tuple[Tensor, dict[str, Tensor], dict[str, int | float | str]]:
    """Compute realization and natural-preservation terms for one target mode."""
    target_mode = _validate_target_mode(target_mode)
    required = {
        "waveform",
        "residual",
        "natural_layer2",
        "natural_layer6",
        "stage1_goal_layer6",
        "enhanced_layer2",
        "enhanced_layer6",
        "padding_mask",
    }
    missing = required.difference(output)
    if missing:
        raise ValueError(f"Stage 2 output is missing fields: {sorted(missing)}")
    enhanced = output["waveform"]
    residual = output["residual"]
    if enhanced.shape != natural_waveform.shape or residual.shape != natural_waveform.shape:
        raise ValueError("natural waveform, enhanced waveform, and residual must share shape")
    _finite_tensor(enhanced, "enhanced waveform")
    _finite_tensor(residual, "waveform residual")
    padding_mask = output["padding_mask"]
    if realize_target is None:
        if realize_mask is not None:
            raise ValueError("realize_mask requires an explicit realize_target")
        realize_target = output["stage1_goal_layer6"]
        realization_mask = padding_mask
        target_mode = "stage1_goal"
    else:
        if realize_mask is None:
            raise ValueError("an explicit realize_target requires realize_mask")
        realization_mask = realize_mask
    _finite_tensor(realize_target, f"{target_mode} realization target")
    realize_cosine, realize_smooth_l1 = _feature_terms(
        output["enhanced_layer6"],
        realize_target,
        realization_mask,
        smooth_l1_beta=smooth_l1_beta,
        name="realize",
    )
    content_cosine, content_smooth_l1 = _feature_terms(
        output["enhanced_layer2"],
        output["natural_layer2"],
        padding_mask,
        smooth_l1_beta=smooth_l1_beta,
        name="content",
    )
    energy = torch.mean(torch.abs(_energy_envelope(enhanced) - _energy_envelope(natural_waveform).detach()))
    residual_l1 = torch.mean(torch.abs(residual))
    residual_smoothness = torch.mean(torch.abs(residual[..., 1:] - residual[..., :-1]))
    anti_clipping = torch.mean(F.relu(torch.abs(enhanced) - 0.98).square())
    terms = {
        "realize_cosine": realize_cosine,
        "realize_smooth_l1": realize_smooth_l1,
        "content_cosine": content_cosine,
        "content_smooth_l1": content_smooth_l1,
        "energy": _finite_scalar(energy, "energy preservation"),
        "residual_l1": _finite_scalar(residual_l1, "residual L1"),
        "residual_smoothness": _finite_scalar(residual_smoothness, "residual smoothness"),
        "anti_clipping": _finite_scalar(anti_clipping, "anti-clipping"),
    }
    if set(active_weights) != set(terms):
        raise ValueError("active Stage 2 weights must match the exact loss-term whitelist")
    total = sum(
        (float(active_weights[name]) * value for name, value in terms.items()),
        enhanced.new_zeros(()),
    )
    terms["total"] = _finite_scalar(total, "Stage 2 total loss")
    diagnostics: dict[str, int | float | str] = {
        "real_frame_count": int(padding_mask.sum().item()),
    }
    if target_mode == "stage1_goal" and realization_mask is padding_mask:
        diagnostics["stage1_target_frame_count"] = int(realization_mask.sum().item())
    else:
        diagnostics.update(
            {
                "target_mode": target_mode,
                "realize_target_frame_count": int(realization_mask.sum().item()),
            }
        )
    return terms["total"], terms, diagnostics


def _audio_stats(audio: Tensor) -> dict[str, float | int | bool]:
    _finite_tensor(audio, "audio diagnostics")
    values = audio.detach().float().reshape(-1)
    return {
        "sample_count": int(values.numel()),
        "rms": float(torch.sqrt(torch.mean(values.square())).cpu()),
        "mean_abs": float(values.abs().mean().cpu()),
        "peak": float(values.abs().max().cpu()),
        "clipped_sample_count": int((values.abs() >= 1.0).sum().cpu()),
        "finite": True,
    }


def _residual_stats(residual: Tensor) -> dict[str, float | bool]:
    _finite_tensor(residual, "residual diagnostics")
    values = residual.detach().float()
    return {
        "rms": float(torch.sqrt(torch.mean(values.square())).cpu()),
        "mean_abs": float(values.abs().mean().cpu()),
        "peak": float(values.abs().max().cpu()),
        "finite": True,
    }


def _proximity_to_target(
    enhanced: Tensor,
    natural: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    prefix: str,
) -> dict[str, float | bool]:
    natural_cosine, natural_smooth = _feature_terms(
        natural.detach(), target.detach(), mask, smooth_l1_beta=1.0, name=f"natural_{prefix}"
    )
    enhanced_cosine, enhanced_smooth = _feature_terms(
        enhanced.detach(), target.detach(), mask, smooth_l1_beta=1.0, name=f"enhanced_{prefix}"
    )
    natural_combined = float((natural_cosine + natural_smooth).cpu())
    enhanced_combined = float((enhanced_cosine + enhanced_smooth).cpu())
    return {
        f"natural_to_{prefix}_combined": natural_combined,
        f"enhanced_to_{prefix}_combined": enhanced_combined,
        "absolute_improvement": natural_combined - enhanced_combined,
        "relative_improvement": (natural_combined - enhanced_combined) / max(natural_combined, 1e-12),
        "enhanced_closer_than_natural": enhanced_combined < natural_combined,
        "matched_frame_count": int(mask.sum().item()),
        "coverage": float(mask.sum().item() / max(1, mask.numel())),
    }


def _proximity_to_stage1_goal(output: Mapping[str, Tensor]) -> dict[str, float | bool]:
    mask = output["padding_mask"]
    return _proximity_to_target(
        output["enhanced_layer6"],
        output["natural_layer6"],
        output["stage1_goal_layer6"],
        mask,
        prefix="stage1_goal",
    )


def _gradient_norm_and_cosine(
    renderer: FeatureTargetedWaveformRenderer,
    record: Mapping[str, Any],
    *,
    epoch: int,
    weights: Stage2LossWeights,
    schedule: TargetFirstSchedule,
    read_wave: Callable[..., Tensor],
    device: torch.device,
) -> dict[str, float | bool]:
    """Measure objective compatibility on one fixed train item without stepping."""
    renderer.train()
    natural = read_wave(record["natural_path"]).to(device)
    target_mode = _validate_target_mode(str(record.get("target_mode", "stage1_goal")))
    realize_target = record.get("realize_target_layer6")
    realize_mask = record.get("realize_target_mask")
    output = renderer(
        natural,
        natural_layer2=record["natural_layer2"],
        natural_layer6=record["natural_layer6"],
        delta_layer6=record["stage1_delta_layer6"],
        padding_mask=record["padding_mask"],
    )
    _, terms, _ = compute_stage2_loss(
        output,
        natural,
        active_weights=schedule.active_weights(weights, epoch),
        smooth_l1_beta=weights.smooth_l1_beta,
        realize_target=realize_target,
        realize_mask=realize_mask,
        target_mode=target_mode,
    )
    realize = terms["realize_cosine"] * weights.realize_cosine + terms["realize_smooth_l1"] * weights.realize_smooth_l1
    active = schedule.active_weights(weights, epoch)
    preservation = sum(
        (terms[name] * active[name] for name in active if name not in {"realize_cosine", "realize_smooth_l1"}),
        natural.new_zeros(()),
    )
    parameters = [parameter for parameter in renderer.parameters() if parameter.requires_grad]
    realize_grads = torch.autograd.grad(realize, parameters, retain_graph=True, allow_unused=True)
    preservation_grads = torch.autograd.grad(preservation, parameters, allow_unused=True)
    realize_vector = torch.cat([
        (gradient.detach().reshape(-1) if gradient is not None else torch.zeros_like(parameter).reshape(-1))
        for gradient, parameter in zip(realize_grads, parameters)
    ])
    preservation_vector = torch.cat([
        (gradient.detach().reshape(-1) if gradient is not None else torch.zeros_like(parameter).reshape(-1))
        for gradient, parameter in zip(preservation_grads, parameters)
    ])
    realize_norm = float(torch.linalg.vector_norm(realize_vector).cpu())
    preservation_norm = float(torch.linalg.vector_norm(preservation_vector).cpu())
    both_nonzero = realize_norm > 0.0 and preservation_norm > 0.0
    cosine = float(F.cosine_similarity(realize_vector, preservation_vector, dim=0, eps=1e-12).cpu()) if both_nonzero else 0.0
    if not all(math.isfinite(value) for value in (realize_norm, preservation_norm, cosine)):
        raise FloatingPointError("Stage 2 calibration gradient diagnostic is non-finite")
    return {
        "realize_gradient_norm": realize_norm,
        "preservation_gradient_norm": preservation_norm,
        "cosine": cosine,
        "both_nonzero": both_nonzero,
    }


def _source_hashes(item: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("natural_sha256", "tts_sha256", "matched_span_metadata_sha256"):
        value = str(item.get(name) or "")
        if not value:
            raise ValueError(f"Stage 2 record requires {name}")
        result[name] = value
    return result


def _validate_stage1_provenance(
    checkpoint_path: str | Path,
    stage1_artifact: FrozenStage1Artifact,
    manifest_path: str | Path,
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    path = Path(checkpoint_path).resolve()
    if stage1_artifact.checkpoint_path != str(path):
        raise ValueError("frozen Stage 1 artifact path does not match requested checkpoint")
    if stage1_artifact.checkpoint_sha256 != file_sha256(path):
        raise ValueError("frozen Stage 1 checkpoint hash changed after loading")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Stage 1 checkpoint must be an object")
    manifest = Path(manifest_path).resolve()
    if payload.get("manifest_sha256") != file_sha256(manifest):
        raise ValueError("Stage 1 checkpoint was selected from a different stage manifest")
    identity = payload.get("split_identity")
    if not isinstance(identity, Mapping) or set(identity) != {"train", "valid"}:
        raise ValueError("Stage 1 checkpoint split identity is incompatible")
    for split in ("train", "valid"):
        expected = identity[split]
        if not isinstance(expected, Mapping):
            raise ValueError("Stage 1 split identity is malformed")
        keys = [str(item["paired_key"]) for item in groups[split]]
        if list(expected.get("paired_keys", [])) != keys:
            raise ValueError(f"Stage 1 {split} paired keys do not match Stage 2 manifest")
        expected_hashes = expected.get("source_hashes")
        if not isinstance(expected_hashes, Mapping):
            raise ValueError("Stage 1 source hashes are missing")
        for item in groups[split]:
            key = str(item["paired_key"])
            if expected_hashes.get(key) != _source_hashes(item):
                raise ValueError(f"Stage 1 source identity mismatch for {key}")


def prepare_record(
    item: Mapping[str, Any],
    *,
    renderer: FeatureTargetedWaveformRenderer,
    device: torch.device,
    read_wave: Callable[..., Tensor] = _read_stage1_wave,
    target_mode: str = "stage1_goal",
) -> dict[str, Any]:
    """Cache natural conditioning and the selected detached realization target."""
    target_mode = _validate_target_mode(target_mode)
    split = str(item.get("split") or "")
    if split not in {"train", "valid"}:
        raise ValueError(f"Stage 2 cannot prepare split {split!r}")
    natural_path = Path(str(item.get("natural_path") or ""))
    if not natural_path.is_file() or file_sha256(natural_path) != str(item.get("natural_sha256") or ""):
        raise ValueError("natural source hash mismatch")
    natural = read_wave(natural_path).to(device)
    natural_layer2, natural_layer6, delta_layer6 = renderer.natural_features(natural)
    padding_mask = torch.ones(natural_layer6.shape[:2], dtype=torch.bool, device=device)
    goal = renderer.stage1.goal_layer6(natural_layer6, delta_layer6)
    record: dict[str, Any] = {
        "paired_key": str(item["paired_key"]),
        "split": split,
        "speaker_group": str(item["speaker_group"]),
        "natural_path": str(natural_path),
        "natural_layer2": natural_layer2,
        "natural_layer6": natural_layer6,
        "stage1_delta_layer6": delta_layer6,
        "stage1_goal_layer6": goal.detach(),
        "padding_mask": padding_mask,
        "hashes": _source_hashes(item),
        "target_mode": target_mode,
        "encoder_calls": {"natural_l2_l6_and_stage1": 1, "raw_tts": 0},
    }
    if target_mode == "aligned_raw_tts_oracle":
        tts = read_wave(item["tts_path"], allow_24k_resample=True).to(device)
        selected_layer = int(renderer.interface["selected_layer"])
        tts_features = extract_hubert_layers(
            renderer.encoder,
            tts,
            layers=(selected_layer,),
            interface=renderer.interface,
        )
        aligned_tts, target_mask, alignment = align_tts_features_to_natural(
            natural_layer6,
            tts_features[selected_layer],
            item["matched_spans"],
            frame_stride_samples=int(renderer.interface["frame_stride_samples"]),
            sample_rate=int(renderer.interface["sample_rate"]),
            min_frames=1,
        )
        if target_mask.ndim == 1:
            target_mask = target_mask.unsqueeze(0)
        if target_mask.dtype != torch.bool or not bool(target_mask.any()):
            raise ValueError("oracle raw-TTS alignment produced no target frames")
        if tuple(aligned_tts.shape) != tuple(natural_layer6.shape):
            raise ValueError("oracle raw-TTS target does not match natural L6 shape")
        record.update(
            {
                "realize_target_layer6": aligned_tts.detach(),
                "realize_target_mask": target_mask.detach(),
                "realize_target_alignment": alignment,
                "encoder_calls": {"natural_l2_l6_and_stage1": 1, "raw_tts": 1},
            }
        )
    return record


def prepare_records(
    records: Sequence[Mapping[str, Any]],
    *,
    renderer: FeatureTargetedWaveformRenderer,
    device: torch.device,
    read_wave: Callable[..., Tensor] = _read_stage1_wave,
    target_mode: str = "stage1_goal",
) -> list[dict[str, Any]]:
    return [
        prepare_record(
            item,
            renderer=renderer,
            device=device,
            read_wave=read_wave,
            target_mode=target_mode,
        )
        for item in records
    ]


def run_epoch(
    renderer: FeatureTargetedWaveformRenderer,
    records: Sequence[Mapping[str, Any]],
    *,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    weights: Stage2LossWeights,
    schedule: TargetFirstSchedule,
    device: torch.device,
    read_wave: Callable[..., Tensor] = _read_stage1_wave,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    training = optimizer is not None
    renderer.train(training)
    active = schedule.active_weights(weights, epoch)
    totals = {name: 0.0 for name in (*active, "total")}
    rows: list[dict[str, Any]] = []
    for record in records:
        natural = read_wave(record["natural_path"]).to(device)
        target_mode = _validate_target_mode(str(record.get("target_mode", "stage1_goal")))
        realize_target = record.get("realize_target_layer6")
        realize_mask = record.get("realize_target_mask")
        if target_mode == "aligned_raw_tts_oracle" and (realize_target is None or realize_mask is None):
            raise ValueError("oracle Stage 2 record is missing its masked raw-TTS target")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = renderer(
                natural,
                natural_layer2=record["natural_layer2"],
                natural_layer6=record["natural_layer6"],
                delta_layer6=record["stage1_delta_layer6"],
                padding_mask=record["padding_mask"],
            )
            loss, terms, loss_diagnostics = compute_stage2_loss(
                output,
                natural,
                active_weights=active,
                smooth_l1_beta=weights.smooth_l1_beta,
                realize_target=realize_target,
                realize_mask=realize_mask,
                target_mode=target_mode,
            )
            if training:
                loss.backward()
                gradients = [
                    parameter.grad.detach().float().norm()
                    for parameter in renderer.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                if not gradients:
                    raise FloatingPointError("Stage 2 renderer received no gradients")
                gradient_norm = float(torch.stack(gradients).norm().cpu())
                if not math.isfinite(gradient_norm):
                    raise FloatingPointError("Stage 2 gradient norm is non-finite")
                optimizer.step()
            else:
                gradient_norm = 0.0
        residual_peak = float(output["residual"].detach().abs().max().cpu())
        if residual_peak > renderer.waveform_model.residual_scale + 1e-5:
            raise FloatingPointError("Stage 2 residual exceeded its serialized bound")
        enhanced_stats = _audio_stats(output["waveform"])
        row = {
            "paired_key": record["paired_key"],
            "split": record["split"],
            "target_mode": target_mode,
            "losses": {name: float(value.detach().cpu()) for name, value in terms.items()},
            "loss_diagnostics": loss_diagnostics,
            "active_weights": dict(active),
            "natural": _audio_stats(natural),
            "enhanced": enhanced_stats,
            "residual": _residual_stats(output["residual"]),
            "stage1_goal_proximity": _proximity_to_stage1_goal(output),
            "gradient_norm": gradient_norm,
            "encoder_calls": {"natural_l2_l6_and_stage1": 0, "enhanced_l2_l6": 1, "raw_tts": 0},
            "prepared_encoder_calls": dict(record["encoder_calls"]),
            "hashes": dict(record["hashes"]),
            "eligible": bool(enhanced_stats["finite"] and enhanced_stats["clipped_sample_count"] == 0),
        }
        if target_mode == "aligned_raw_tts_oracle":
            row["realize_target_proximity"] = _proximity_to_target(
                output["enhanced_layer6"],
                output["natural_layer6"],
                realize_target,
                realize_mask,
                prefix="aligned_raw_tts",
            )
            row["realize_target_alignment"] = record["realize_target_alignment"]
        rows.append(row)
        for name, value in terms.items():
            totals[name] += float(value.detach().cpu())
    if not rows:
        raise ValueError("Stage 2 epoch requires at least one record")
    means = {name: value / len(rows) for name, value in totals.items()}
    means[CHECKPOINT_SELECTION_METRIC] = means["realize_cosine"] + means["realize_smooth_l1"]
    means["eligible_fraction"] = float(np.mean([bool(row["eligible"]) for row in rows]))
    return means, rows


def _split_identity(prepared: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    if set(prepared) != {"train", "valid"}:
        raise ValueError("Stage 2 checkpoint must contain only train and valid identities")
    return {
        split: {
            "paired_keys": [str(row["paired_key"]) for row in rows],
            "source_hashes": {str(row["paired_key"]): dict(row["hashes"]) for row in rows},
        }
        for split, rows in prepared.items()
    }


def _trainable_state_dict(renderer: FeatureTargetedWaveformRenderer) -> dict[str, Tensor]:
    names = {name for name, parameter in renderer.named_parameters() if parameter.requires_grad}
    return {
        name: value.detach().cpu().clone()
        for name, value in renderer.state_dict().items()
        if name in names
    }


def make_checkpoint(
    renderer: FeatureTargetedWaveformRenderer,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    history: Sequence[Mapping[str, Any]],
    prepared: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest_path: str | Path,
    weights: Stage2LossWeights,
    schedule: TargetFirstSchedule,
    config: TrainConfig,
    target_mode: str = "stage1_goal",
) -> dict[str, Any]:
    target_mode = _validate_target_mode(target_mode)
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    state = _trainable_state_dict(renderer)
    if not state or any(not torch.isfinite(value).all() for value in state.values()):
        raise FloatingPointError("Stage 2 trainable state is absent or non-finite")
    manifest = Path(manifest_path).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_type": EXPERIMENT_TYPE,
        "epoch": int(epoch),
        "state_dict": state,
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": renderer.model_config(),
        "encoder_interface": dict(renderer.interface),
        "stage1_checkpoint": renderer.stage1_artifact.as_dict(),
        "manifest_path": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "split_identity": _split_identity(prepared),
        "target_mode": target_mode,
        "target_contract": {
            "layer": int(renderer.interface["selected_layer"]),
            "alignment": "matched_phone_spans_only" if target_mode == "aligned_raw_tts_oracle" else "frozen_stage1_goal",
            "mask_policy": "unmatched_frames_excluded" if target_mode == "aligned_raw_tts_oracle" else "all_real_frames",
        },
        "loss_weights": weights.as_dict(),
        "loss_schedule": schedule.as_dict(),
        "train_config": config.as_dict(),
        "selection": {
            "metric": CHECKPOINT_SELECTION_METRIC,
            "policy": CHECKPOINT_SELECTION_POLICY,
            "eligible_splits": ["valid"],
            "heldout_excluded": True,
        },
        "history": [dict(row) for row in history],
    }


def validate_checkpoint(payload: Mapping[str, Any], renderer: FeatureTargetedWaveformRenderer) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported Stage 2 checkpoint schema")
    if payload.get("experiment_type") != EXPERIMENT_TYPE:
        raise ValueError("checkpoint is not a frozen Stage-1-targeted renderer")
    if payload.get("encoder_interface") != renderer.interface:
        raise ValueError("Stage 2 checkpoint HuBERT interface does not match renderer")
    if payload.get("model_config") != renderer.model_config():
        raise ValueError("Stage 2 checkpoint model config does not match renderer")
    if payload.get("stage1_checkpoint") != renderer.stage1_artifact.as_dict():
        raise ValueError("Stage 2 checkpoint frozen Stage 1 provenance does not match renderer")
    if set(payload.get("split_identity", {})) != {"train", "valid"}:
        raise ValueError("Stage 2 checkpoint must retain train/valid identities only")
    target_mode = _validate_target_mode(str(payload.get("target_mode", "stage1_goal")))
    target_contract = payload.get("target_contract")
    if target_contract is not None:
        if not isinstance(target_contract, Mapping):
            raise ValueError("Stage 2 checkpoint target contract is malformed")
        expected_alignment = "matched_phone_spans_only" if target_mode == "aligned_raw_tts_oracle" else "frozen_stage1_goal"
        expected_mask = "unmatched_frames_excluded" if target_mode == "aligned_raw_tts_oracle" else "all_real_frames"
        if (
            int(target_contract.get("layer", -1)) != int(renderer.interface["selected_layer"])
            or target_contract.get("alignment") != expected_alignment
            or target_contract.get("mask_policy") != expected_mask
        ):
            raise ValueError("Stage 2 checkpoint target contract is incompatible")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping) or selection.get("metric") != CHECKPOINT_SELECTION_METRIC:
        raise ValueError("Stage 2 checkpoint selection policy is incompatible")
    state = payload.get("state_dict")
    expected = _trainable_state_dict(renderer)
    if not isinstance(state, Mapping) or set(state) != set(expected):
        raise ValueError("Stage 2 checkpoint state keys are incompatible")
    for name, value in state.items():
        if not isinstance(value, Tensor) or value.shape != expected[name].shape or not torch.isfinite(value).all():
            raise ValueError(f"invalid Stage 2 checkpoint tensor: {name}")


def save_checkpoint(path: str | Path, payload: Mapping[str, Any], renderer: FeatureTargetedWaveformRenderer) -> None:
    validate_checkpoint(payload, renderer)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(output)


def load_checkpoint(
    path: str | Path,
    renderer: FeatureTargetedWaveformRenderer,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Stage 2 checkpoint payload must be an object")
    validate_checkpoint(payload, renderer)
    state = dict(renderer.state_dict())
    state.update(payload["state_dict"])
    renderer.load_state_dict(state, strict=True)
    return dict(payload)


def train_stage2(
    manifest_path: str | Path,
    stage1_checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    config: TrainConfig = TrainConfig(),
    weights: Stage2LossWeights = Stage2LossWeights(),
    schedule: TargetFirstSchedule = TargetFirstSchedule(),
    encoder: nn.Module | None = None,
    interface: EncoderInterface | Mapping[str, Any] | None = None,
    read_wave: Callable[..., Tensor] = _read_stage1_wave,
    conditioning_dim: int = DEFAULT_STAGE2_CONDITIONING_DIM,
    channels: int = DEFAULT_STAGE2_CHANNELS,
    dilations: Sequence[int] = DEFAULT_STAGE2_DILATIONS,
    residual_scale: float = DEFAULT_STAGE2_RESIDUAL_SCALE,
    target_mode: str = "stage1_goal",
) -> dict[str, Any]:
    """Train Stage 2 on train/valid only with a fixed verified realization target."""
    target_mode = _validate_target_mode(target_mode)
    _set_seed(config.seed)
    device = torch.device(config.device)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Stage 2 output directory must be fresh: {output}")
    groups, _ = load_stage1_records(manifest_path)
    stage1, artifact = load_frozen_stage1(stage1_checkpoint_path, map_location="cpu")
    stage1 = stage1.to(device)
    if encoder is None:
        encoder, loaded_interface = _load_stage1_hubert(device=device)
        interface_payload: Mapping[str, Any] = loaded_interface.as_dict()
    else:
        if interface is None:
            raise ValueError("an explicit HuBERT interface is required with an injected encoder")
        interface_payload = interface.as_dict() if isinstance(interface, EncoderInterface) else interface
        encoder.to(device)
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    interface_payload = validate_hubert_interface(interface_payload)
    _validate_stage1_provenance(stage1_checkpoint_path, artifact, manifest_path, groups)
    renderer = FeatureTargetedWaveformRenderer(
        encoder,
        stage1.to(device),
        artifact,
        interface=interface_payload,
        conditioning_dim=conditioning_dim,
        channels=channels,
        dilations=dilations,
        residual_scale=residual_scale,
    ).to(device)
    prepared = {
        split: prepare_records(
            rows,
            renderer=renderer,
            device=device,
            read_wave=read_wave,
            target_mode=target_mode,
        )
        for split, rows in groups.items()
    }
    optimizer = AdamW(
        [parameter for parameter in renderer.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_metric: float | None = None
    for epoch in range(config.epochs):
        train_metrics, train_rows = run_epoch(
            renderer,
            prepared["train"],
            optimizer=optimizer,
            epoch=epoch,
            weights=weights,
            schedule=schedule,
            device=device,
            read_wave=read_wave,
        )
        with torch.no_grad():
            valid_metrics, valid_rows = run_epoch(
                renderer,
                prepared["valid"],
                optimizer=None,
                epoch=epoch,
                weights=weights,
                schedule=schedule,
                device=device,
                read_wave=read_wave,
            )
        calibration = _gradient_norm_and_cosine(
            renderer,
            prepared["train"][0],
            epoch=epoch,
            weights=weights,
            schedule=schedule,
            read_wave=read_wave,
            device=device,
        )
        entry = {
            "epoch": epoch,
            "train": train_metrics,
            "valid": valid_metrics,
            "calibration_gradient_alignment": calibration,
            "train_diagnostics": train_rows,
            "valid_diagnostics": valid_rows,
        }
        history.append(entry)
        checkpoint = make_checkpoint(
            renderer,
            optimizer,
            epoch=epoch,
            history=history,
            prepared=prepared,
            manifest_path=manifest_path,
            weights=weights,
            schedule=schedule,
            config=config,
            target_mode=target_mode,
        )
        save_checkpoint(output / "last.pt", checkpoint, renderer)
        metric = valid_metrics[CHECKPOINT_SELECTION_METRIC]
        eligible = math.isfinite(metric) and valid_metrics["eligible_fraction"] == 1.0
        if eligible and (best_metric is None or metric < best_metric):
            save_checkpoint(output / "best.pt", checkpoint, renderer)
            best_metric = metric
    if best_metric is None:
        raise FloatingPointError("no eligible finite Stage 2 validation checkpoint")
    return {
        "output_dir": str(output),
        "epochs": config.epochs,
        "target_mode": target_mode,
        "best_metric": best_metric,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--target-mode", choices=TARGET_MODES, default="stage1_goal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = train_stage2(
        args.manifest,
        args.stage1_checkpoint,
        args.output_dir,
        config=TrainConfig(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
        ),
        schedule=TargetFirstSchedule(warmup_epochs=args.warmup_epochs),
        target_mode=args.target_mode,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


__all__ = [
    "CHECKPOINT_SELECTION_METRIC",
    "CHECKPOINT_SELECTION_POLICY",
    "EXPERIMENT_TYPE",
    "SCHEMA_VERSION",
    "Stage2LossWeights",
    "TARGET_MODES",
    "TargetFirstSchedule",
    "TrainConfig",
    "compute_stage2_loss",
    "load_checkpoint",
    "make_checkpoint",
    "prepare_record",
    "prepare_records",
    "run_epoch",
    "save_checkpoint",
    "train_stage2",
    "validate_checkpoint",
]


if __name__ == "__main__":
    main()
