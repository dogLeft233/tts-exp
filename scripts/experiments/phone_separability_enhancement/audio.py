"""Differentiable, exact-length band-gain renderer with hard clock guards."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _torch():
    import torch

    return torch


def _float_waveform(pcm: np.ndarray | Any):
    torch = _torch()
    if isinstance(pcm, torch.Tensor):
        value = pcm
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 1:
            raise ValueError("waveform must be one-dimensional")
        return value.to(dtype=torch.float32)
    value = np.asarray(pcm)
    if value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("PCM must be a non-empty finite one-dimensional array")
    if value.dtype == np.int16:
        return torch.from_numpy(value.astype(np.float32) / 32768.0)
    return torch.as_tensor(value, dtype=torch.float32)


def _sample_interval(begin: float, end: float, sample_rate: int, count: int) -> tuple[int, int]:
    left = max(0, min(count, int(round(begin * sample_rate))))
    right = max(left, min(count, int(round(end * sample_rate))))
    return left, right


def make_protected_mask(tokens: Sequence[Mapping[str, Any]], sample_count: int, *, sample_rate: int = 16_000, edge_guard_s: float = 0.010, taper_s: float = 0.005) -> dict[str, np.ndarray]:
    """Create a [0, 1] edit mask and an immutable hard-protection mask."""

    if sample_count <= 0 or sample_rate <= 0:
        raise ValueError("sample_count and sample_rate must be positive")
    mask = np.zeros(int(sample_count), dtype=np.float32)
    protected = np.ones(int(sample_count), dtype=bool)
    taper = max(0, int(round(float(taper_s) * sample_rate)))
    guard = max(0, int(round(float(edge_guard_s) * sample_rate)))
    for token in tokens:
        speech = bool(token.get("speech", not token.get("silence", False)))
        left, right = _sample_interval(float(token["start_s"]), float(token["end_s"]), sample_rate, sample_count)
        if right <= left or not speech:
            continue
        editable_left = min(right, left + guard)
        editable_right = max(left, right - guard)
        protected[left:editable_left] = True
        protected[editable_right:right] = True
        if editable_right <= editable_left:
            continue
        protected[editable_left:editable_right] = False
        mask[editable_left:editable_right] = 1.0
        if taper > 0:
            ramp = min(taper, editable_right - editable_left)
            if ramp > 0:
                mask[editable_left:editable_left + ramp] *= np.linspace(0.0, 1.0, ramp, endpoint=False, dtype=np.float32)
                mask[editable_right - ramp:editable_right] *= np.linspace(1.0, 0.0, ramp, endpoint=False, dtype=np.float32)
    # Re-apply hard guards after taper. This is intentional: no non-zero taper
    # may leak into silence, utterance edges, or per-phone edge guards.
    mask[protected] = 0.0
    return {"mask": mask, "protected": protected}


def _node_frequency_weights(n_freq: int, n_bands: int, sample_rate: int, device: Any, dtype: Any):
    torch = _torch()
    freqs = torch.linspace(0.0, sample_rate / 2.0, n_freq, device=device, dtype=dtype)
    mel_min = 2595.0 * torch.log10(torch.tensor(1.0, device=device, dtype=dtype))
    mel_max = 2595.0 * torch.log10(torch.tensor(1.0 + (sample_rate / 2.0) / 700.0, device=device, dtype=dtype))
    nodes_mel = torch.linspace(mel_min, mel_max, n_bands, device=device, dtype=dtype)
    freq_mel = 2595.0 * torch.log10(1.0 + freqs / 700.0)
    right = torch.searchsorted(nodes_mel, freq_mel, right=True).clamp(1, n_bands - 1)
    left = right - 1
    denominator = (nodes_mel[right] - nodes_mel[left]).clamp_min(1e-8)
    fraction = ((freq_mel - nodes_mel[left]) / denominator).clamp(0.0, 1.0)
    weights = torch.zeros((n_freq, n_bands), device=device, dtype=dtype)
    weights.scatter_(1, left[:, None], (1.0 - fraction)[:, None])
    weights.scatter_add_(1, right[:, None], fraction[:, None])
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _smooth_time(value: Any):
    torch = _torch()
    kernel = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], device=value.device, dtype=value.dtype) / 9.0
    kernel = kernel.view(1, 1, -1).expand(value.shape[1], 1, -1)
    padded = torch.nn.functional.pad(value, (2, 2), mode="replicate")
    return torch.nn.functional.conv1d(padded, kernel, groups=value.shape[1])


def occurrence_parameter_layout(tokens: Sequence[Mapping[str, Any]], *, dynamic: bool, sample_rate: int = 16_000, node_ms: float = 80.0) -> list[dict[str, Any]]:
    layout: list[dict[str, Any]] = []
    cursor = 0
    for index, token in enumerate(tokens):
        if not bool(token.get("speech", not token.get("silence", False))):
            continue
        duration = max(0.0, float(token["end_s"]) - float(token["start_s"]))
        nodes = max(1, int(math.ceil(duration * 1000.0 / float(node_ms)))) if dynamic else 1
        layout.append({"token_index": int(index), "label": str(token.get("label", "")), "start_s": float(token["start_s"]), "end_s": float(token["end_s"]), "nodes": nodes, "start": cursor, "end": cursor + nodes * 24})
        cursor += nodes * 24
    return layout


def parameter_count(layout: Sequence[Mapping[str, Any]]) -> int:
    return int(max((int(row["end"]) for row in layout), default=0))


def gain_field_from_parameters(parameters: Any, tokens: Sequence[Mapping[str, Any]], frame_times: Any, *, dynamic: bool, sample_rate: int = 16_000, node_ms: float = 80.0, max_gain_db: float = 6.0):
    torch = _torch()
    if parameters.ndim == 1:
        parameters = parameters.unsqueeze(0)
    if parameters.ndim != 2 or parameters.shape[0] != 1:
        raise ValueError("parameters must have shape [1, D]")
    times = frame_times if isinstance(frame_times, torch.Tensor) else torch.as_tensor(frame_times, device=parameters.device, dtype=parameters.dtype)
    layout = occurrence_parameter_layout(tokens, dynamic=dynamic, sample_rate=sample_rate, node_ms=node_ms)
    field = torch.zeros((1, 24, int(times.numel())), device=parameters.device, dtype=parameters.dtype)
    for spec in layout:
        span = parameters[:, int(spec["start"]):int(spec["end"])].reshape(1, int(spec["nodes"]), 24)
        centers = torch.linspace(float(spec["start_s"]), float(spec["end_s"]), int(spec["nodes"]), device=parameters.device, dtype=parameters.dtype)
        positions = torch.searchsorted(centers, times).clamp(1, int(spec["nodes"]) - 1) if int(spec["nodes"]) > 1 else torch.zeros_like(times, dtype=torch.long)
        if int(spec["nodes"]) == 1:
            values = span[:, 0, :].unsqueeze(-1).expand(1, 24, times.numel())
            inside = (times >= float(spec["start_s"])) & (times < float(spec["end_s"]))
        else:
            left = positions - 1
            right = positions
            frac = ((times - centers[left]) / (centers[right] - centers[left]).clamp_min(1e-8)).clamp(0.0, 1.0)
            values = span[:, left, :].transpose(1, 2) * (1.0 - frac).view(1, 1, -1) + span[:, right, :].transpose(1, 2) * frac.view(1, 1, -1)
            inside = (times >= float(spec["start_s"])) & (times < float(spec["end_s"]))
        field = torch.where(inside.view(1, 1, -1), values, field)
    return float(max_gain_db) * torch.tanh(field)


class BandGainRenderer:
    """Phase-preserving STFT magnitude gain with differentiable projection."""

    def __init__(self, *, sample_rate: int = 16_000, n_fft: int = 512, win_length: int = 512, hop_length: int = 128, n_bands: int = 24, max_gain_db: float = 6.0, floor: float = 1e-8) -> None:
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_bands = int(n_bands)
        self.max_gain_db = float(max_gain_db)
        self.floor = float(floor)

    def band_features(self, waveform: Any) -> tuple[Any, Any]:
        torch = _torch()
        x = _float_waveform(waveform)
        if x.numel() <= 256:
            raise ValueError("SHORT_AUDIO")
        window = torch.hann_window(self.win_length, device=x.device, dtype=x.dtype, periodic=True)
        spectrum = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=window, center=True, pad_mode="reflect", return_complex=True)
        power = spectrum.abs().square()
        weights = _node_frequency_weights(spectrum.shape[0], self.n_bands, self.sample_rate, x.device, x.dtype)
        node_power = torch.einsum("fb,ft->bt", weights, power)
        return torch.log(node_power.clamp_min(self.floor)).unsqueeze(0), spectrum

    def render(self, waveform: Any, gain_db: Any, mask: Any, *, source_pcm: np.ndarray | None = None) -> tuple[Any, dict[str, Any]]:
        torch = _torch()
        x = _float_waveform(waveform)
        if x.numel() <= 256:
            raise ValueError("SHORT_AUDIO")
        if gain_db.ndim == 2:
            gain_db = gain_db.unsqueeze(0)
        if gain_db.ndim != 3 or gain_db.shape[0] != 1 or gain_db.shape[1] != self.n_bands:
            raise ValueError(f"gain_db must have shape [1,{self.n_bands},frames]")
        window = torch.hann_window(self.win_length, device=x.device, dtype=x.dtype, periodic=True)
        spectrum = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=window, center=True, pad_mode="reflect", return_complex=True)
        frames = spectrum.shape[-1]
        if gain_db.shape[-1] != frames:
            gain_db = torch.nn.functional.interpolate(gain_db, size=frames, mode="linear", align_corners=True)
        gain_db = _smooth_time(gain_db.clamp(-self.max_gain_db, self.max_gain_db))
        weights = _node_frequency_weights(spectrum.shape[0], self.n_bands, self.sample_rate, x.device, x.dtype)
        frequency_gain = torch.einsum("fb,bt->ft", weights, gain_db[0])
        modified = spectrum * torch.pow(torch.tensor(10.0, device=x.device, dtype=x.dtype), frequency_gain / 20.0)
        reconstructed = torch.istft(modified, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=window, center=True, length=x.numel())
        edit_mask = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask, device=x.device, dtype=x.dtype)
        edit_mask = edit_mask.reshape(-1).to(dtype=x.dtype)
        if edit_mask.numel() != x.numel():
            raise ValueError("mask and waveform lengths differ")
        residual = edit_mask * (reconstructed - x)
        source_energy = (edit_mask * x).square().sum().clamp_min(1e-8)
        residual_ratio = (residual.square().sum() / source_energy).clamp_min(0.0)
        alpha_energy = torch.minimum(torch.ones_like(residual_ratio), torch.sqrt(torch.tensor(0.01, device=x.device, dtype=x.dtype) / residual_ratio.clamp_min(1e-12)))
        positive = residual > 0
        negative = residual < 0
        peak_pos = torch.where(positive, (1.0 - x) / residual.clamp_min(1e-12), torch.full_like(residual, float("inf"))).min()
        peak_neg = torch.where(negative, (-1.0 - x) / residual.clamp_max(-1e-12), torch.full_like(residual, float("inf"))).min()
        alpha_peak = torch.clamp(torch.minimum(peak_pos, peak_neg), 0.0, 1.0)
        alpha = torch.minimum(alpha_energy, alpha_peak)
        output = x + alpha * residual
        output = torch.where(edit_mask > 0, output, x)
        rms_source = torch.sqrt((edit_mask * x).square().mean().clamp_min(1e-12))
        rms_output = torch.sqrt((edit_mask * output).square().mean().clamp_min(1e-12))
        snr = 10.0 * torch.log10(source_energy / residual.square().sum().clamp_min(1e-12))
        metadata = {
            "sample_count": int(x.numel()),
            "frames": int(frames),
            "residual_energy_ratio_before": float(residual_ratio.detach().cpu()),
            "residual_energy_ratio_after": float((alpha * residual).square().sum().detach().cpu() / source_energy.detach().cpu().clamp(min=1e-8)),
            "alpha_energy": float(alpha_energy.detach().cpu()),
            "alpha_peak": float(alpha_peak.detach().cpu()),
            "alpha": float(alpha.detach().cpu()),
            "snr_db": float(snr.detach().cpu()),
            "rms_change_db": float((20.0 * torch.log10(rms_output / rms_source)).detach().cpu()),
            "gain_db_max": float(gain_db.detach().abs().max().cpu()),
            "gain_db_rms": float(torch.sqrt(gain_db.square().mean()).detach().cpu()),
            "mask_fraction": float((edit_mask > 0).float().mean().detach().cpu()),
            "effective_edit": bool(float((alpha * residual).abs().max().detach().cpu()) > 1.0 / 32768.0),
        }
        return output, metadata


def export_pcm(waveform: Any, source_pcm: np.ndarray, protected: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    torch = _torch()
    value = waveform.detach().cpu().numpy() if isinstance(waveform, torch.Tensor) else np.asarray(waveform)
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    source = np.asarray(source_pcm, dtype=np.int16).reshape(-1)
    if value.size != source.size:
        raise ValueError("waveform and source PCM lengths differ")
    quantized = np.rint(np.clip(value, -1.0, 32767.0 / 32768.0) * 32768.0).astype(np.int64)
    quantized = np.clip(quantized, -32768, 32767).astype(np.int16)
    if protected is not None:
        hard = np.asarray(protected, dtype=bool)
        if hard.shape != source.shape:
            raise ValueError("protected mask shape differs")
        quantized[hard] = source[hard]
    metadata = {
        "sample_count": int(quantized.size),
        "pcm_sha256": hashlib.sha256(quantized.astype("<i2", copy=False).tobytes()).hexdigest(),
        "changed_sample_count": int(np.count_nonzero(quantized != source)),
        "max_abs_pcm_delta": int(np.max(np.abs(quantized.astype(np.int32) - source.astype(np.int32)))) if quantized.size else 0,
        "new_saturated_samples": int(np.count_nonzero((np.abs(quantized) >= 32767) & (np.abs(source) < 32767))),
    }
    return quantized, metadata


def validate_distortion(source_pcm: np.ndarray, candidate_pcm: np.ndarray, mask: np.ndarray, *, min_snr_db: float = 20.0, max_rms_change_db: float = 1.0) -> dict[str, Any]:
    source = np.asarray(source_pcm, dtype=np.float64)
    candidate = np.asarray(candidate_pcm, dtype=np.float64)
    edit = np.asarray(mask, dtype=np.float64) > 0
    if source.shape != candidate.shape or source.shape != edit.shape:
        raise ValueError("source, candidate, mask shape mismatch")
    residual = (candidate - source)[edit]
    reference = source[edit]
    ratio = float(np.sum(residual * residual) / max(np.sum(reference * reference), 1e-8)) if residual.size else 0.0
    snr = float(10.0 * math.log10(max(np.sum(reference * reference), 1e-8) / max(np.sum(residual * residual), 1e-8))) if residual.size else float("inf")
    rms_source = float(np.sqrt(np.mean(reference * reference))) if reference.size else 0.0
    rms_candidate = float(np.sqrt(np.mean(candidate[edit] * candidate[edit]))) if reference.size else 0.0
    rms_change = float(20.0 * math.log10(max(rms_candidate, 1e-12) / max(rms_source, 1e-12))) if reference.size else 0.0
    return {"residual_energy_ratio": ratio, "snr_db": snr, "rms_change_db": rms_change, "new_saturated_samples": int(np.count_nonzero((np.abs(candidate) >= 32767) & (np.abs(source) < 32767))), "pass": bool(snr >= min_snr_db and abs(rms_change) <= max_rms_change_db and np.all(np.isfinite(candidate)))}


def build_gain_control(source_pcm: np.ndarray, candidate_pcm: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Match the realized residual energy with a mask-only constant-residual control."""

    source = np.asarray(source_pcm, dtype=np.int16)
    candidate = np.asarray(candidate_pcm, dtype=np.int16)
    edit = np.asarray(mask, dtype=np.float64) > 0
    if source.shape != candidate.shape or source.shape != edit.shape:
        raise ValueError("gain-control input shapes differ")
    residual = candidate.astype(np.float64) - source.astype(np.float64)
    target = float(np.sum((residual[edit]) ** 2))
    base = source.astype(np.float64)
    # The control is a constant residual direction inside the same editable
    # mask; its scale is solved analytically and then peak-projected.
    direction = np.sign(np.where(edit, base, 0.0))
    direction[~edit] = 0.0
    denom = float(np.sum(direction[edit] ** 2))
    scale = math.sqrt(target / max(denom, 1e-8))
    control_float = base + scale * direction
    safe = np.ones_like(control_float)
    positive = direction > 0
    negative = direction < 0
    if np.any(positive):
        safe[positive] = np.minimum(safe[positive], (32767.0 - base[positive]) / np.maximum(scale * direction[positive], 1e-12))
    if np.any(negative):
        safe[negative] = np.minimum(safe[negative], (-32768.0 - base[negative]) / np.minimum(scale * direction[negative], -1e-12))
    control_float = base + max(0.0, min(1.0, float(np.min(safe)))) * scale * direction
    output, metadata = export_pcm(control_float / 32768.0, source, protected=~edit)
    actual = float(np.sum((output.astype(np.float64) - source.astype(np.float64))[edit] ** 2))
    metadata.update({"target_residual_energy": target, "actual_residual_energy": actual, "relative_energy_error": abs(actual - target) / max(target, 1.0), "algorithm": "constant_masked_gain_control"})
    return output, metadata


__all__ = [
    "BandGainRenderer",
    "build_gain_control",
    "export_pcm",
    "gain_field_from_parameters",
    "make_protected_mask",
    "occurrence_parameter_layout",
    "parameter_count",
    "validate_distortion",
]
