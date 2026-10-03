"""Differentiable SyncNet-guided optimization boundary for the shared head.

The concrete Wav2Lip input adapter and SyncNet audiovisual crop adapter are
injected by the experiment runner. Frozen model parameters stay frozen, while
autograd remains enabled through generated frames back to enhanced audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: Exception | None = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass(frozen=True)
class TrainStepInfo:
    sync_score_mean: float
    identity_penalty: float
    loss: float
    gradient_l2: float
    audio_gradient_l2: float
    parameter_update_l2: float
    batch_size: int


def freeze_model_parameters(*models: Any) -> None:
    """Freeze weights and select eval mode without disabling input gradients."""

    if torch is None:
        raise RuntimeError("training support requires PyTorch") from _TORCH_IMPORT_ERROR
    for model in models:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)


def sync_guided_train_step(
    *,
    enhancement_stack: Any,
    optimizer: Any,
    natural_audio,
    video_context: Any,
    render_video: Callable[[Any, Any], Any],
    sync_score: Callable[..., Any],
    use_global_rms: bool = False,
    use_local_envelope: bool = False,
    use_spectral_head: bool = True,
    local_window_ms: float = 32.0,
    local_ratio: float = 0.8,
    local_min_gain_db: float = -6.0,
    local_max_gain_db: float = 6.0,
    post_rms_match: str = "none",
    target_rms: float = 0.0376838172675746,
    anchor_lag_frames: int = 0,
    peak_ceiling: float | None = None,
    rms_stage_peak_ceiling: float | None = None,
    identity_weight: float = 0.02,
    spectral_smoothness_weight: float = 0.002,
) -> TrainStepInfo:
    """Run one fit-batch step against frozen Wav2Lip and natural-audio SyncNet.

    ``render_video(context, enhanced_audio)`` must call the frozen Wav2Lip
    generator with gradients enabled and return generated frames. The
    ``sync_score(generated_frames, natural_audio)`` adapter must score those
    frames against the untouched natural PCM. Its result is a finite tensor
    with higher values representing better synchronization.

    No gradient is requested for generator/SyncNet parameters. A gradient
    path from the SyncNet score to the enhanced waveform is explicitly checked
    before the optimizer step.
    """

    if torch is None:
        raise RuntimeError("training support requires PyTorch") from _TORCH_IMPORT_ERROR
    if identity_weight < 0 or spectral_smoothness_weight < 0:
        raise ValueError("regularization weights must be non-negative")
    if natural_audio.ndim not in (1, 2, 3):
        raise ValueError("natural_audio must be mono [N], [B,N], or [B,1,N]")
    enhancement_stack.train()
    optimizer.zero_grad(set_to_none=True)
    enhanced, log_mask = enhancement_stack.forward_with_mask(
        natural_audio,
        use_global_rms=use_global_rms,
        use_local_envelope=use_local_envelope,
        use_spectral_head=use_spectral_head,
        local_window_ms=local_window_ms,
        local_ratio=local_ratio,
        local_min_gain_db=local_min_gain_db,
        local_max_gain_db=local_max_gain_db,
        post_rms_match=post_rms_match,
        target_rms=target_rms,
        peak_ceiling=peak_ceiling,
        rms_stage_peak_ceiling=rms_stage_peak_ceiling,
    )
    generated_video = render_video(video_context, enhanced)
    # The concrete adapter needs the same frozen source-track/frame clock used
    # by the renderer in order to crop natural PCM exactly as SyncNet does.
    scores = sync_score(
        generated_video,
        natural_audio.detach(),
        context=video_context,
        anchor_lag_frames=anchor_lag_frames,
    )
    if not torch.is_tensor(scores) or scores.numel() == 0 or not bool(torch.isfinite(scores).all().item()):
        raise RuntimeError("SyncNet adapter must return non-empty finite differentiable scores")
    score_mean = scores.mean()
    if not score_mean.requires_grad:
        raise RuntimeError("SyncNet score is detached from generated video/audio")
    audio_gradient = torch.autograd.grad(
        score_mean,
        enhanced,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if audio_gradient is None:
        raise RuntimeError("no gradient path from natural-audio SyncNet score to enhanced audio")
    audio_gradient_l2 = torch.linalg.vector_norm(audio_gradient.detach().to(torch.float64))
    if not bool(torch.isfinite(audio_gradient_l2).item()) or float(audio_gradient_l2.item()) <= 0.0:
        raise RuntimeError("natural-audio SyncNet gradient is zero or non-finite")
    audio_gradient_value = float(audio_gradient_l2.item())

    natural = natural_audio.to(dtype=enhanced.dtype)
    if natural.ndim == 3:
        natural = natural[:, 0, :]
    elif natural.ndim == 1:
        natural = natural.unsqueeze(0)
    candidate = enhanced[:, 0, :] if enhanced.ndim == 3 else enhanced
    if candidate.ndim == 1:
        candidate = candidate.unsqueeze(0)
    if natural.shape != candidate.shape:
        raise RuntimeError("natural and enhanced audio shapes differ in training step")
    identity_penalty = torch.mean((candidate - natural) ** 2)
    if log_mask.numel() and log_mask.shape[-1] > 1 and log_mask.shape[-2] > 1:
        mask_smoothness = torch.mean(torch.abs(log_mask[..., 1:, :] - log_mask[..., :-1, :]))
        mask_smoothness = mask_smoothness + torch.mean(
            torch.abs(log_mask[..., :, 1:] - log_mask[..., :, :-1])
        )
    else:
        mask_smoothness = score_mean.new_zeros(())
    loss = (
        -score_mean
        + identity_weight * identity_penalty
        + spectral_smoothness_weight * mask_smoothness
    )
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("training objective became non-finite")
    loss.backward()

    grad_sq = torch.zeros((), dtype=torch.float64, device=loss.device)
    has_trainable_gradient = False
    for parameter in enhancement_stack.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            has_trainable_gradient = True
            grad_sq = grad_sq + parameter.grad.detach().to(torch.float64).square().sum()
    if not has_trainable_gradient:
        raise RuntimeError("enhancement head received no trainable gradients")
    gradient_l2 = float(torch.sqrt(grad_sq).item())
    if not torch.isfinite(grad_sq).item() or gradient_l2 <= 0.0:
        raise RuntimeError("enhancement gradient norm is zero or non-finite")
    parameters_before = [
        parameter.detach().clone()
        for parameter in enhancement_stack.parameters()
        if parameter.requires_grad
    ]
    if not parameters_before:
        raise RuntimeError("enhancement head has no trainable parameters")
    optimizer.step()
    update_sq = torch.zeros((), dtype=torch.float64, device=loss.device)
    trainable_after = [parameter for parameter in enhancement_stack.parameters() if parameter.requires_grad]
    if len(trainable_after) != len(parameters_before):
        raise RuntimeError("trainable parameter set changed during optimizer step")
    for before, after in zip(parameters_before, trainable_after):
        update_sq = update_sq + (after.detach() - before).to(torch.float64).square().sum()
    parameter_update_l2 = float(torch.sqrt(update_sq).item())
    if not torch.isfinite(update_sq).item() or parameter_update_l2 <= 0.0:
        raise RuntimeError("optimizer produced no finite nonzero head parameter update")
    return TrainStepInfo(
        sync_score_mean=float(score_mean.detach().item()),
        identity_penalty=float(identity_penalty.detach().item()),
        loss=float(loss.detach().item()),
        gradient_l2=gradient_l2,
        audio_gradient_l2=audio_gradient_value,
        parameter_update_l2=parameter_update_l2,
        batch_size=int(candidate.shape[0]),
    )
