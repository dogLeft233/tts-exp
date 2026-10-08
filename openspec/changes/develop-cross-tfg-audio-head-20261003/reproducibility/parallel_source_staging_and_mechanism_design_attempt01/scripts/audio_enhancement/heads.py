"""Shared, identity-initialized phase-preserving spectral residual head."""

from __future__ import annotations

import math

try:
    import torch
    from torch import nn
except ImportError as exc:  # keep audio I/O/baselines usable without PyTorch
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR: Exception | None = exc
else:
    _TORCH_IMPORT_ERROR = None


def _torch_local_envelope_compression(
    x,
    *,
    window_ms: float = 32.0,
    ratio: float = 0.8,
    min_gain_db: float = -6.0,
    max_gain_db: float = 6.0,
    envelope_floor: float = 1e-5,
    sample_rate_hz: int = 16_000,
):
    """Differentiable mirror of the fixed 16 kHz local RMS compressor."""

    if torch is None:
        raise RuntimeError("local envelope compression requires PyTorch") from _TORCH_IMPORT_ERROR
    if x.ndim != 2 or x.shape[-1] == 0:
        raise ValueError("local envelope expects [batch,samples]")
    if sample_rate_hz != 16_000 or not 20.0 <= window_ms <= 50.0:
        raise ValueError("local envelope is fixed to 16 kHz and 20--50 ms windows")
    if not math.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("local envelope ratio must be finite and in (0,1]")
    if not (math.isfinite(min_gain_db) and math.isfinite(max_gain_db)):
        raise ValueError("local envelope gain bounds must be finite")
    if min_gain_db > 0 or max_gain_db < 0 or min_gain_db > max_gain_db:
        raise ValueError("local envelope gain bounds must include 0 dB and be ordered")
    if not math.isfinite(envelope_floor) or envelope_floor <= 0:
        raise ValueError("envelope_floor must be finite and positive")

    n_window = max(1, int(round(sample_rate_hz * window_ms / 1000.0)))
    n_window = min(n_window, max(1, (x.shape[-1] // 2) * 2 + 1))
    left = n_window // 2 if n_window % 2 else n_window // 2 - 1
    right = n_window - 1 - left
    power = x.square().unsqueeze(1)
    padded = torch.nn.functional.pad(power, (left, right), mode="replicate")
    kernel = torch.full((1, 1, n_window), 1.0 / n_window, dtype=x.dtype, device=x.device)
    local_power = torch.nn.functional.conv1d(padded, kernel).squeeze(1)
    if local_power.shape != x.shape:
        raise RuntimeError("local RMS smoother changed sample count")
    envelope = torch.sqrt(local_power.clamp_min(envelope_floor**2))
    global_rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True).clamp_min(0.0))
    raw_db = 20.0 * (ratio - 1.0) * torch.log10(envelope / global_rms.clamp_min(envelope_floor))
    gain_db = raw_db.clamp(min_gain_db, max_gain_db)
    gain = torch.pow(torch.tensor(10.0, dtype=x.dtype, device=x.device), gain_db / 20.0)
    gain = torch.where(global_rms <= 1e-12, torch.ones_like(gain), gain)
    return x * gain


if nn is not None:

    class SpectralResidualHead(nn.Module):
        """Predict a bounded real magnitude mask; retain each input STFT phase.

        For STFT ``X`` and real positive mask ``m``, the returned waveform is
        ``x + iSTFT((m - 1) * X)``.  The last projection is zero-initialized,
        making the initial head exactly identity while keeping the gradient
        path through the generated waveform available for frozen TFG models.
        """

        def __init__(
            self,
            *,
            n_fft: int = 512,
            hop_length: int = 128,
            hidden_channels: int = 24,
            max_residual_db: float = 3.0,
            level_normalize: bool = True,
            frequency_center_mask: bool = False,
            frame_energy_normalize: bool = False,
        ) -> None:
            super().__init__()
            if n_fft < 64 or n_fft % 2 or hop_length <= 0 or hop_length > n_fft:
                raise ValueError("invalid STFT dimensions")
            if hidden_channels < 4:
                raise ValueError("hidden_channels must be at least 4")
            if not (0.0 < max_residual_db <= 12.0):
                raise ValueError("max_residual_db must be in (0, 12]")
            self.n_fft = int(n_fft)
            self.hop_length = int(hop_length)
            self.max_residual_db = float(max_residual_db)
            self.level_normalize = bool(level_normalize)
            self.frequency_center_mask = bool(frequency_center_mask)
            self.frame_energy_normalize = bool(frame_energy_normalize)
            self.register_buffer("window", torch.hann_window(self.n_fft, periodic=True))
            self.mask_net = nn.Sequential(
                nn.Conv2d(1, hidden_channels, kernel_size=(5, 5), padding=(2, 2)),
                nn.GELU(),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(3, 9), padding=(1, 4)),
                nn.GELU(),
                nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            )
            final = self.mask_net[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

        @staticmethod
        def _shape_input(waveform):
            if waveform.ndim == 1:
                return waveform.unsqueeze(0), "mono"
            if waveform.ndim == 2:
                return waveform, "batch"
            if waveform.ndim == 3 and waveform.shape[1] == 1:
                return waveform[:, 0, :], "batch_channel"
            raise ValueError("waveform must be [N], [B,N], or [B,1,N]")

        @staticmethod
        def _restore_shape(waveform, shape: str):
            if shape == "mono":
                return waveform[0]
            if shape == "batch_channel":
                return waveform[:, None, :]
            return waveform

        def forward_with_mask(self, waveform):
            if waveform.is_complex():
                raise ValueError("waveform input must be real-valued")
            x, original_shape = self._shape_input(waveform)
            if x.shape[-1] == 0:
                raise ValueError("empty waveforms are not supported")
            x = x.to(dtype=torch.float32).contiguous()
            if not bool(torch.isfinite(x).all().item()):
                raise ValueError("waveform contains NaN or infinite samples")
            spectrum = torch.stft(
                x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window.to(device=x.device, dtype=x.dtype),
                center=True,
                pad_mode="constant",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
            magnitude = spectrum.abs()
            log_magnitude = torch.log(magnitude.clamp_min(1e-7)).unsqueeze(1)
            if self.level_normalize:
                log_magnitude = log_magnitude - log_magnitude.mean(dim=(-2, -1), keepdim=True)
            raw = self.mask_net(log_magnitude).squeeze(1)
            if self.frequency_center_mask:
                raw = raw - raw.mean(dim=1, keepdim=True)
            bound_ln = self.max_residual_db * (0.11512925464970229)  # ln(10)/20
            log_mask = bound_ln * torch.tanh(raw)
            mask = torch.exp(log_mask)
            if self.frame_energy_normalize:
                input_energy = magnitude.square().sum(dim=1, keepdim=True)
                masked_energy = (mask * magnitude).square().sum(dim=1, keepdim=True)
                frame_scale = torch.sqrt(
                    input_energy.clamp_min(1e-24) / masked_energy.clamp_min(1e-24)
                )
                normalized_mask = mask * frame_scale
                normalized_log_mask = torch.log(normalized_mask.clamp_min(1e-12))
                if bool((normalized_log_mask.abs() > bound_ln + 1e-6).any().item()):
                    raise RuntimeError(
                        "frame-energy normalization would exceed the declared final mask bound"
                    )
                mask = normalized_mask
                log_mask = normalized_log_mask
            delta_spectrum = (mask - 1.0).to(spectrum.dtype) * spectrum
            residual = torch.istft(
                delta_spectrum,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window.to(device=x.device, dtype=x.dtype),
                center=True,
                normalized=False,
                onesided=True,
                length=x.shape[-1],
            )
            output = x + residual
            if output.shape != x.shape or not bool(torch.isfinite(output).all().item()):
                raise RuntimeError("spectral residual head violated output shape/finite contract")
            return self._restore_shape(output, original_shape), log_mask

        def forward(self, waveform):
            return self.forward_with_mask(waveform)[0]


    class EnhancementStack(nn.Module):
        """Separately switchable RMS, local-envelope, and spectral stages."""

        def __init__(self, head: SpectralResidualHead | None = None) -> None:
            super().__init__()
            self.head = head if head is not None else SpectralResidualHead()

        def forward_with_mask(
            self,
            waveform,
            *,
            use_global_rms: bool = False,
            use_local_envelope: bool = False,
            use_spectral_head: bool = True,
            local_window_ms: float = 32.0,
            local_ratio: float = 0.8,
            local_min_gain_db: float = -6.0,
            local_max_gain_db: float = 6.0,
            post_rms_match: str = "none",
            target_rms: float = 0.0376838172675746,
            peak_ceiling: float | None = None,
            rms_stage_peak_ceiling: float | None = None,
            return_diagnostics: bool = False,
        ):
            shape = waveform.shape
            if waveform.ndim == 1:
                x = waveform.unsqueeze(0)
                layout = "mono"
            elif waveform.ndim == 2:
                x = waveform
                layout = "batch"
            elif waveform.ndim == 3 and waveform.shape[1] == 1:
                x = waveform[:, 0, :]
                layout = "channel"
            else:
                raise ValueError("waveform must be [N], [B,N], or [B,1,N]")
            if x.shape[-1] == 0:
                raise ValueError("empty waveforms are not supported")
            x = x.to(torch.float32)
            if not bool(torch.isfinite(x).all().item()):
                raise ValueError("waveform contains NaN or infinite samples")
            if post_rms_match not in {"none", "input", "target"}:
                raise ValueError("post_rms_match must be 'none', 'input', or 'target'")
            if peak_ceiling is not None and (peak_ceiling <= 0 or not math.isfinite(float(peak_ceiling))):
                raise ValueError("peak_ceiling must be positive and finite or None")
            if rms_stage_peak_ceiling is not None and (
                rms_stage_peak_ceiling <= 0 or not math.isfinite(float(rms_stage_peak_ceiling))
            ):
                raise ValueError("rms_stage_peak_ceiling must be positive and finite or None")
            input_rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True).clamp_min(0.0))
            if use_global_rms:
                if target_rms <= 0 or not math.isfinite(float(target_rms)):
                    raise ValueError("target_rms must be positive and finite")
                rms = torch.sqrt(torch.mean(x.square(), dim=-1, keepdim=True).clamp_min(0.0))
                gain = target_rms / rms.clamp_min(1e-12)
                if rms_stage_peak_ceiling is not None:
                    peak = x.abs().amax(dim=-1, keepdim=True)
                    gain = torch.minimum(gain, rms_stage_peak_ceiling / peak.clamp_min(1e-12))
                gain = torch.where(rms <= 1e-12, torch.ones_like(gain), gain)
                x = x * gain
            after_global_rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True).clamp_min(0.0))
            if use_local_envelope:
                x = _torch_local_envelope_compression(
                    x,
                    window_ms=local_window_ms,
                    ratio=local_ratio,
                    min_gain_db=local_min_gain_db,
                    max_gain_db=local_max_gain_db,
                )
            after_local_rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True).clamp_min(0.0))
            if use_spectral_head:
                y, log_mask = self.head.forward_with_mask(x)
            else:
                y = x
                frame_count = 1 + shape[-1] // self.head.hop_length
                log_mask = torch.zeros(
                    x.shape[0],
                    self.head.n_fft // 2 + 1,
                    frame_count,
                    device=x.device,
                    dtype=x.dtype,
                )
            after_head_rms = torch.sqrt(y.square().mean(dim=-1, keepdim=True).clamp_min(0.0))
            if post_rms_match != "none":
                if target_rms <= 0 or not math.isfinite(float(target_rms)):
                    raise ValueError("target_rms must be positive and finite")
                requested_rms = (
                    input_rms
                    if post_rms_match == "input"
                    else torch.full_like(input_rms, float(target_rms))
                )
                match_gain = requested_rms / after_head_rms.clamp_min(1e-12)
                match_gain = torch.where(
                    (requested_rms <= 1e-12) | (after_head_rms <= 1e-12),
                    torch.ones_like(match_gain),
                    match_gain,
                )
                y = y * match_gain
            after_rms_match = torch.sqrt(y.square().mean(dim=-1, keepdim=True).clamp_min(0.0))
            final_peak_gain = torch.ones_like(after_rms_match)
            if peak_ceiling is not None:
                peak = y.abs().amax(dim=-1, keepdim=True)
                final_peak_gain = torch.minimum(
                    torch.ones_like(peak),
                    float(peak_ceiling) / peak.clamp_min(1e-12),
                )
                y = y * final_peak_gain
            output_rms = torch.sqrt(y.square().mean(dim=-1, keepdim=True).clamp_min(0.0))
            expected_internal_shape = (shape[-1],) if layout == "mono" else (shape[0], shape[-1])
            actual_internal_shape = tuple(y.shape)
            if layout == "mono" and len(actual_internal_shape) == 2 and actual_internal_shape[0] == 1:
                y = y[0]
                actual_internal_shape = tuple(y.shape)
            if actual_internal_shape != expected_internal_shape:
                raise RuntimeError("enhancement stack changed waveform shape")
            if layout == "channel":
                y = y[:, None, :]
            if return_diagnostics:
                diagnostics = {
                    "input_rms": input_rms.detach(),
                    "after_global_rms": after_global_rms.detach(),
                    "after_local_rms": after_local_rms.detach(),
                    "after_head_rms": after_head_rms.detach(),
                    "after_rms_match_rms": after_rms_match.detach(),
                    "output_rms": output_rms.detach(),
                    "final_peak_gain": final_peak_gain.detach(),
                    "post_rms_match": post_rms_match,
                }
                return y, log_mask, diagnostics
            return y, log_mask

        def forward(self, waveform, **kwargs):
            return self.forward_with_mask(waveform, **kwargs)[0]

else:

    class SpectralResidualHead:  # pragma: no cover - depends on optional PyTorch
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("SpectralResidualHead requires PyTorch") from _TORCH_IMPORT_ERROR

    class EnhancementStack:  # pragma: no cover - depends on optional PyTorch
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("EnhancementStack requires PyTorch") from _TORCH_IMPORT_ERROR
