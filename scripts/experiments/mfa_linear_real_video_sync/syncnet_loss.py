"""Differentiable, official-aligned SyncNet V2 frontend and objective."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import (
    CURVE_SIZE,
    MEL_TRUST_LIMIT,
    Q,
    SAMPLE_RATE,
    SATURATION_FRACTION_MAX,
    SEGMENT_FRAMES,
    SYNCNET_MFCC_PER_VIDEO_FRAME,
    SYNCNET_MFCC_WINDOW,
    SYNCNET_WINDOW_FRAMES,
    VSHIFT,
)

SYNCNET_FRAME_LENGTH = 400
SYNCNET_FRAME_STEP = 160
SYNCNET_NFFT = 512
SYNCNET_NUM_FILTERS = 26
SYNCNET_NUM_CEPS = 13
SYNCNET_PCM_SCALE = 32768.0
SYNCNET_LOG_EPS = np.finfo(np.float64).eps
WAV2LIP_N_FFT = 800
WAV2LIP_HOP = 200
WAV2LIP_WIN = 800
WAV2LIP_MELS = 80


def _require_finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise FloatingPointError(f"{name} is non-finite")


def pcm16_straight_through(waveform: Tensor) -> Tensor:
    """Return integer-valued PCM16 floats with a live unsaturated gradient."""
    if not torch.is_floating_point(waveform):
        raise TypeError("waveform must be floating point")
    _require_finite(waveform, "waveform")
    scaled = torch.clamp(waveform * SYNCNET_PCM_SCALE, -32768.0, 32767.0)
    quantized = torch.round(scaled)
    return scaled + (quantized - scaled).detach()


def pcm16_forward_numpy(waveform: Tensor) -> np.ndarray:
    pcm = pcm16_straight_through(waveform).detach().cpu().numpy()
    return pcm.astype(np.int16)


def _dct_matrix(input_size: int, output_size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    n = torch.arange(input_size, device=device, dtype=dtype)
    k = torch.arange(output_size, device=device, dtype=dtype).unsqueeze(1)
    matrix = torch.cos(math.pi / input_size * (n + 0.5) * k)
    matrix[0] *= math.sqrt(1.0 / input_size)
    if output_size > 1:
        matrix[1:] *= math.sqrt(2.0 / input_size)
    return matrix


def _framesig(waveform: Tensor) -> Tensor:
    sample_count = waveform.size(-1)
    if sample_count < 1:
        raise ValueError("waveform must contain samples")
    frame_count = 1 if sample_count <= SYNCNET_FRAME_LENGTH else 1 + math.ceil(
        (sample_count - SYNCNET_FRAME_LENGTH) / SYNCNET_FRAME_STEP
    )
    padded_count = (frame_count - 1) * SYNCNET_FRAME_STEP + SYNCNET_FRAME_LENGTH
    padded = F.pad(waveform, (0, padded_count - sample_count))
    return padded.unfold(-1, SYNCNET_FRAME_LENGTH, SYNCNET_FRAME_STEP)


def _syncnet_filterbank(*, device: torch.device, dtype: torch.dtype) -> Tensor:
    from python_speech_features.base import get_filterbanks

    filters = get_filterbanks(
        nfilt=SYNCNET_NUM_FILTERS,
        nfft=SYNCNET_NFFT,
        samplerate=SAMPLE_RATE,
        lowfreq=0,
        highfreq=None,
    )
    return torch.as_tensor(filters, device=device, dtype=dtype)


def _cepstral_lifter(cepstral: Tensor, lifter: int = 22) -> Tensor:
    if lifter <= 0:
        return cepstral
    index = torch.arange(cepstral.size(-1), device=cepstral.device, dtype=cepstral.dtype)
    scale = 1.0 + (lifter / 2.0) * torch.sin(math.pi * index / lifter)
    return cepstral * scale


def syncnet_v2_mfcc(pcm_waveform: Tensor, *, sample_rate: int = SAMPLE_RATE, preemphasis: float = 0.97) -> Tensor:
    """Match python_speech_features.mfcc and return ``[B,13,F]``."""
    if pcm_waveform.ndim == 1:
        pcm_waveform = pcm_waveform.unsqueeze(0)
    if pcm_waveform.ndim == 3 and pcm_waveform.size(1) == 1:
        pcm_waveform = pcm_waveform[:, 0]
    if pcm_waveform.ndim != 2:
        raise ValueError(f"waveform must have shape [B,N], got {tuple(pcm_waveform.shape)}")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"SyncNet V2 expects {SAMPLE_RATE} Hz audio")
    if not torch.is_floating_point(pcm_waveform):
        pcm_waveform = pcm_waveform.float()
    _require_finite(pcm_waveform, "PCM waveform")
    output_dtype = pcm_waveform.dtype
    waveform = pcm_waveform.to(torch.float64)
    emphasized = torch.cat(
        (waveform[..., :1], waveform[..., 1:] - preemphasis * waveform[..., :-1]), dim=-1
    )
    frames = _framesig(emphasized)
    spectrum = torch.fft.rfft(frames, n=SYNCNET_NFFT, dim=-1)
    power = spectrum.abs().square() / SYNCNET_NFFT
    energy = power.sum(dim=-1).clamp_min(SYNCNET_LOG_EPS)
    filter_energy = torch.einsum(
        "bfn,mn->bfm",
        power,
        _syncnet_filterbank(device=waveform.device, dtype=waveform.dtype),
    ).clamp_min(SYNCNET_LOG_EPS)
    dct = _dct_matrix(
        SYNCNET_NUM_FILTERS,
        SYNCNET_NUM_CEPS,
        device=waveform.device,
        dtype=waveform.dtype,
    )
    cepstral = torch.einsum("bfm,cm->bfc", filter_energy.log(), dct)
    cepstral = _cepstral_lifter(cepstral)
    mfcc = torch.cat((energy.log().unsqueeze(-1), cepstral[..., 1:]), dim=-1)
    result = mfcc.transpose(1, 2).to(output_dtype)
    _require_finite(result, "MFCC")
    return result


def syncnet_audio_windows(
    mfcc: Tensor,
    *,
    video_frame_count: int,
    audio_sample_count: int,
) -> Tensor:
    if mfcc.ndim != 3 or mfcc.size(1) != SYNCNET_NUM_CEPS:
        raise ValueError(f"mfcc must have shape [B,13,F], got {tuple(mfcc.shape)}")
    coordinate_count = min(video_frame_count, audio_sample_count // (SAMPLE_RATE // 25))
    window_count = coordinate_count - SYNCNET_WINDOW_FRAMES
    if window_count < 1:
        raise ValueError("need at least six aligned audio/video frames")
    starts = torch.arange(window_count, device=mfcc.device) * SYNCNET_MFCC_PER_VIDEO_FRAME
    indices = starts[:, None] + torch.arange(SYNCNET_MFCC_WINDOW, device=mfcc.device)[None]
    if int(indices[-1, -1]) >= mfcc.size(-1):
        raise ValueError("MFCC does not cover the requested audio/video coordinates")
    return mfcc[:, :, indices].permute(0, 2, 1, 3).reshape(
        -1, 1, SYNCNET_NUM_CEPS, SYNCNET_MFCC_WINDOW
    )


def syncnet_video_windows(frames: Tensor) -> Tensor:
    """Build official ``[W,3,5,H,W]`` BGR/0..255 visual windows."""
    if frames.ndim == 4 and frames.shape[-1] == 3:
        frames = frames.permute(3, 0, 1, 2).unsqueeze(0)
    elif frames.ndim == 4 and frames.shape[1] == 3:
        frames = frames.permute(1, 0, 2, 3).unsqueeze(0)
    if frames.ndim != 5 or frames.shape[0] != 1 or frames.shape[1] != 3:
        raise ValueError("frames must be [F,H,W,3], [F,3,H,W], or [1,3,F,H,W]")
    frame_count = frames.shape[2]
    window_count = frame_count - SYNCNET_WINDOW_FRAMES
    if window_count < 1:
        raise ValueError("need at least six video frames")
    windows = torch.stack(
        [frames[:, :, start : start + SYNCNET_WINDOW_FRAMES] for start in range(window_count)],
        dim=1,
    )
    return windows.reshape(-1, 3, SYNCNET_WINDOW_FRAMES, frames.shape[-2], frames.shape[-1]).float()


def audio_embeddings(syncnet: nn.Module, candidate: Tensor, *, video_frame_count: int = SEGMENT_FRAMES) -> Tensor:
    pcm = pcm16_straight_through(candidate)
    mfcc = syncnet_v2_mfcc(pcm)
    windows = syncnet_audio_windows(
        mfcc,
        video_frame_count=video_frame_count,
        audio_sample_count=candidate.shape[-1],
    )
    embedding = syncnet.forward_aud(windows)
    _require_finite(embedding, "candidate audio embedding")
    return embedding


def cached_visual_embeddings(syncnet: nn.Module, frames: Tensor, *, batch_size: int = 32) -> Tensor:
    syncnet.eval()
    windows = syncnet_video_windows(frames)
    outputs = []
    with torch.inference_mode():
        for start in range(0, windows.shape[0], batch_size):
            outputs.append(syncnet.forward_lip(windows[start : start + batch_size]))
        embedding = torch.cat(outputs, dim=0)
    _require_finite(embedding, "visual embedding")
    return embedding.detach()


def official_syncnet_distance_curve(
    audio_embedding: Tensor,
    visual_embedding: Tensor,
    *,
    vshift: int = VSHIFT,
    detach_visual: bool = True,
) -> Tensor:
    if audio_embedding.ndim != 2 or visual_embedding.ndim != 2:
        raise ValueError("embeddings must have shape [windows,dimensions]")
    if audio_embedding.shape != visual_embedding.shape:
        raise ValueError("audio and visual embedding shapes must match")
    if vshift < 0:
        raise ValueError("vshift must be non-negative")
    padded_audio = F.pad(audio_embedding, (0, 0, vshift, vshift))
    candidates = padded_audio.unfold(0, 2 * vshift + 1, 1).permute(0, 2, 1)
    visual = visual_embedding.detach() if detach_visual else visual_embedding
    curve = F.pairwise_distance(visual.unsqueeze(1), candidates).mean(dim=0)
    _require_finite(curve, "SyncNet curve")
    return curve


def target_margin_loss_components(
    candidate_curve: Tensor,
    *,
    pristine_curve: Tensor,
    target_offset: int,
    vshift: int = VSHIFT,
    q: float = Q,
) -> dict[str, Tensor]:
    expected = 2 * vshift + 1
    if vshift < 1 or candidate_curve.ndim != 1 or candidate_curve.numel() != expected:
        raise ValueError("candidate curve must contain every searched offset")
    if pristine_curve.ndim != 1 or pristine_curve.numel() != expected:
        raise ValueError("pristine curve must contain every searched offset")
    if not -vshift <= target_offset <= vshift:
        raise ValueError("target offset is outside the searched range")
    if not math.isfinite(q) or q <= 0:
        raise ValueError("q must be finite and positive")
    _require_finite(candidate_curve, "candidate curve")
    pristine = pristine_curve.detach().to(candidate_curve)
    _require_finite(pristine, "pristine curve")
    d0 = pristine.min()
    c0 = pristine.median() - d0
    delta_d = torch.maximum(d0.new_tensor(3 * q), d0 * 0.01)
    delta_c = torch.maximum(c0.new_tensor(3 * q), c0 * 0.01)
    if not bool((d0 > delta_d).item()) or not bool((c0 > 0).item()):
        raise ValueError("pristine curve lacks a valid positive D/C scale")
    d_goal = (d0 - delta_d).detach()
    c_goal = (c0 + delta_c).detach()
    s_d = d0.clamp_min(1e-3).detach()
    s_c = c0.clamp_min(1e-3).detach()
    target_index = target_offset + vshift
    target = candidate_curve[target_index]
    target_violation = F.relu((target - d_goal) / s_d)
    mask = torch.ones(expected, dtype=torch.bool, device=candidate_curve.device)
    mask[target_index] = False
    ranking_violations = F.relu((c_goal - (candidate_curve[mask] - target)) / s_c)
    ranking_violation = ranking_violations.mean()
    total = target_violation + ranking_violation
    _require_finite(total, "target-margin loss")
    return {
        "total": total,
        "sync": total,
        "curve": candidate_curve,
        "pristine_curve": pristine,
        "d0": d0.detach(),
        "c0": c0.detach(),
        "delta_d": delta_d.detach(),
        "delta_c": delta_c.detach(),
        "d_goal": d_goal,
        "c_goal": c_goal,
        "s_d": s_d,
        "s_c": s_c,
        "target_distance": target,
        "target_violation": target_violation,
        "ranking_violations": ranking_violations,
        "ranking_violation": ranking_violation,
    }


def _wav2lip_mel_basis(*, device: torch.device, dtype: torch.dtype) -> Tensor:
    import librosa.filters

    basis = librosa.filters.mel(
        sr=SAMPLE_RATE,
        n_fft=WAV2LIP_N_FFT,
        n_mels=WAV2LIP_MELS,
        fmin=55,
        fmax=7600,
    )
    return torch.as_tensor(basis, device=device, dtype=dtype)


def wav2lip_normalized_log_mel(waveform: Tensor) -> Tensor:
    if waveform.ndim == 3 and waveform.size(1) == 1:
        waveform = waveform[:, 0]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [B,N] or [B,1,N]")
    _require_finite(waveform, "mel waveform")
    waveform = waveform.float()
    emphasized = torch.cat(
        (waveform[..., :1], waveform[..., 1:] - 0.97 * waveform[..., :-1]), dim=-1
    )
    window = torch.hann_window(WAV2LIP_WIN, device=waveform.device, dtype=waveform.dtype)
    spectrum = torch.stft(
        emphasized,
        n_fft=WAV2LIP_N_FFT,
        hop_length=WAV2LIP_HOP,
        win_length=WAV2LIP_WIN,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
    ).abs()
    mel = torch.einsum(
        "mf,bft->bmt", _wav2lip_mel_basis(device=waveform.device, dtype=waveform.dtype), spectrum
    )
    min_level = math.exp(-100 / 20 * math.log(10))
    mel_db = 20 * torch.log10(mel.clamp_min(min_level)) - 20
    normalized = (8 * ((mel_db + 100) / 100) - 4).clamp(-4, 4)
    _require_finite(normalized, "normalized log-mel")
    return normalized


def log_mel_trust_violation(distance: Tensor, *, limit: float = MEL_TRUST_LIMIT) -> Tensor:
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("trust limit must be finite and positive")
    _require_finite(distance, "log-mel distance")
    return F.relu((distance - limit) / limit)


def log_mel_trust_components(
    candidate: Tensor,
    mfa_linear: Tensor,
    *,
    limit: float = MEL_TRUST_LIMIT,
) -> dict[str, Tensor]:
    if candidate.shape != mfa_linear.shape:
        raise ValueError("candidate and MFA-linear waveform shapes must match")
    distance = F.l1_loss(
        wav2lip_normalized_log_mel(candidate),
        wav2lip_normalized_log_mel(mfa_linear).detach(),
    )
    violation = log_mel_trust_violation(distance, limit=limit)
    return {"total": violation, "trust": violation, "distance": distance, "limit": distance.new_tensor(limit)}


def waveform_qc(
    candidate: Tensor,
    mfa_linear: Tensor,
    *,
    residual_bound: float,
    saturation_fraction_max: float = SATURATION_FRACTION_MAX,
) -> dict[str, Any]:
    if candidate.shape != mfa_linear.shape:
        raise ValueError("candidate output length or shape changed")
    _require_finite(candidate, "candidate waveform")
    _require_finite(mfa_linear, "MFA-linear waveform")
    residual = candidate - mfa_linear
    peak = float(residual.detach().abs().max().cpu())
    if peak > residual_bound + 1e-6:
        raise ValueError("candidate residual exceeds configured bound")
    scaled = candidate.detach() * SYNCNET_PCM_SCALE
    saturated = (scaled < -32768.0) | (scaled > 32767.0)
    count = int(saturated.sum().cpu())
    fraction = count / candidate.numel()
    return {
        "finite": True,
        "exact_shape": True,
        "residual_peak": peak,
        "residual_rms": float(residual.detach().square().mean().sqrt().cpu()),
        "input_peak": float(mfa_linear.detach().abs().max().cpu()),
        "candidate_peak": float(candidate.detach().abs().max().cpu()),
        "input_rms": float(mfa_linear.detach().square().mean().sqrt().cpu()),
        "candidate_rms": float(candidate.detach().square().mean().sqrt().cpu()),
        "pcm_saturation_count": count,
        "pcm_saturation_fraction": fraction,
        "pcm_saturation_pass": fraction <= saturation_fraction_max,
        "residual_bound_pass": peak <= residual_bound + 1e-6,
    }


__all__ = [
    "audio_embeddings",
    "cached_visual_embeddings",
    "log_mel_trust_components",
    "log_mel_trust_violation",
    "official_syncnet_distance_curve",
    "pcm16_forward_numpy",
    "pcm16_straight_through",
    "syncnet_audio_windows",
    "syncnet_v2_mfcc",
    "syncnet_video_windows",
    "target_margin_loss_components",
    "waveform_qc",
    "wav2lip_normalized_log_mel",
]
