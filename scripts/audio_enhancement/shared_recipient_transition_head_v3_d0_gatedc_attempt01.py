"""Preparation-only shared recipient-conditioned transition head with GateDC control.

This file defines a future audio transform; importing it creates no models,
loads no data, and performs no signal processing.  All transform choices are
required constructor arguments.  The only trainable values are nine
reference-coded band contrasts.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


SCHEMA = "cross_tfg_shared_recipient_transition_head_v3_d0_gate_dc_attempt01"
PREPARATION_ONLY = True
TRAINABLE_SCALAR_COUNT = 9
BAND_COUNT = 4
TEMPORAL_COMPONENTS = ("static", "positive_flux", "negative_flux")

# Integration references only. This module does not import these files.
REFERENCE_SOURCE_PINS = {
    "shared_recipient_transition_head_v2.py": "664938a3aabd15b0a5e091cf88f8dc7ebae6916c545d1c0624ca837cb37e85bc",
    "shared_recipient_transition_head_v1.py": "25098dfcdcea733968010a132d24ff120dc300d75b7c18f0004edfffea127abd",
    "wav2lip_fullcontext_fit_runtime_v4_attempt01.py": "dd751132bd3d8dec5abce288c2f765df78d9b7ef9150cdfb0ccff52ba72ee458",
    "wav2lip_frontend.py": "3480087daecf8dc295f4a3b20703f393bada2690e1f5ce7839ba1dc4fbdec8ef",
    "baselines.py": "7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2",
    "phase_clock_diagnostics_v1.py": "25ed6f204dea7e5c149f7c897dfccfebe972ad0ef30eda1a2b1dbdc59ae216c5",
}


def _require_finite_scalar(name: str, value: float, *, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _pad_1d(values: Tensor, width: int, mode: str) -> Tensor:
    if width == 0:
        return values
    if mode not in {"reflect", "replicate"}:
        raise ValueError("padding mode must be explicitly set to reflect or replicate")
    if mode == "reflect" and values.shape[-1] <= width:
        raise ValueError("reflect padding needs more frames than its padding width")
    return F.pad(values, (width, width), mode=mode)


def _smooth_gate(values: Tensor, width: int, mode: str) -> Tensor:
    if width == 1:
        return values
    half = width // 2
    padded = _pad_1d(values.reshape(1, 1, -1), half, mode)
    return F.avg_pool1d(padded, kernel_size=width, stride=1).reshape(-1)


def _smooth_mask_axis(values: Tensor, width: int, axis: str, mode: str) -> Tensor:
    if width == 1:
        return values
    half = width // 2
    if axis == "frequency":
        if mode == "reflect" and values.shape[0] <= half:
            raise ValueError("frequency reflect padding exceeds the available bins")
        pad = (0, 0, half, half)
        kernel = (width, 1)
    elif axis == "time":
        if mode == "reflect" and values.shape[1] <= half:
            raise ValueError("time reflect padding exceeds the available frames")
        pad = (half, half, 0, 0)
        kernel = (1, width)
    else:
        raise ValueError("axis must be frequency or time")
    if mode not in {"reflect", "replicate"}:
        raise ValueError("mask padding mode must be reflect or replicate")
    padded = F.pad(values[None, None, :, :], pad, mode=mode)
    return F.avg_pool2d(padded, kernel_size=kernel, stride=1)[0, 0]


class SharedRecipientTransitionHeadV3(nn.Module):
    """A source-conditioned, four-band smooth log-magnitude residual head.

    ``band_contrasts_db`` has shape [3, 3]: three temporal components by
    three non-reference bands.  A fourth, explicit reference band has fixed
    coefficient zero. These are nine stored trainable scalars; effective
    design rank depends on observed band support and the actual gate sequences.
    There is no trainable waveform-wide gain outside this parameterization.
    """

    def __init__(
        self,
        *,
        sample_rate_hz: int,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window_kind: str,
        window_periodic: bool,
        stft_center: bool,
        stft_pad_mode: str,
        stft_normalized: bool,
        stft_onesided: bool,
        stft_return_complex: bool,
        band_edges_hz: Sequence[float],
        reference_band_index: int,
        flux_denominator_floor: float,
        activity_floor: float,
        flux_normalization_scale: float,
        positive_flux_quantile: float,
        negative_flux_quantile: float,
        gate_temperature: float,
        gate_smoothing_frames: int,
        gate_smoothing_pad_mode: str,
        mask_smoothing_frequency_bins: int,
        mask_smoothing_time_frames: int,
        mask_smoothing_pad_mode: str,
        frequency_common_mode_policy: str,
        max_abs_mask_db: float,
        target_rms: float,
        rms_epsilon: float,
        peak_ceiling: float | None,
        baseline_match_atol: float,
        video_fps_num: int,
        video_fps_den: int,
        frame_power_floor: float,
    ) -> None:
        super().__init__()
        if sample_rate_hz <= 0 or n_fft <= 0 or n_fft % 2 != 0:
            raise ValueError("sample rate and positive even n_fft are required")
        if hop_length <= 0 or not 0 < win_length <= n_fft:
            raise ValueError("hop_length and win_length are outside the configured STFT")
        if window_kind != "hann":
            raise ValueError("this source revision supports only an explicitly selected Hann window")
        if not isinstance(window_periodic, bool):
            raise ValueError("window_periodic must be an explicit boolean")
        if not isinstance(stft_center, bool):
            raise ValueError("stft_center must be an explicit boolean")
        if stft_center is not True:
            raise ValueError("this Hann-window source revision requires centered STFT/ISTFT")
        if stft_normalized is not False or stft_onesided is not True or stft_return_complex is not True:
            raise ValueError("this source revision requires unnormalized, one-sided complex STFT")
        if stft_pad_mode not in {"reflect", "replicate", "constant"}:
            raise ValueError("unsupported explicit STFT padding mode")
        if len(band_edges_hz) != BAND_COUNT + 1:
            raise ValueError("exactly five edges are required for four frequency bands")
        edges = tuple(_require_finite_scalar("band edge", edge) for edge in band_edges_hz)
        nyquist = sample_rate_hz / 2.0
        if edges[0] != 0.0 or not math.isclose(edges[-1], nyquist, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("band edges must span exactly zero through Nyquist")
        if any(left >= right for left, right in zip(edges, edges[1:])):
            raise ValueError("band edges must be strictly increasing")
        if not 0 <= reference_band_index < BAND_COUNT:
            raise ValueError("reference_band_index must identify one of the four bands")

        self.flux_denominator_floor = _require_finite_scalar(
            "flux_denominator_floor", flux_denominator_floor, positive=True
        )
        self.activity_floor = _require_finite_scalar("activity_floor", activity_floor, positive=True)
        self.flux_normalization_scale = _require_finite_scalar(
            "flux_normalization_scale", flux_normalization_scale, positive=True
        )
        self.positive_flux_quantile = _require_finite_scalar(
            "positive_flux_quantile", positive_flux_quantile
        )
        self.negative_flux_quantile = _require_finite_scalar(
            "negative_flux_quantile", negative_flux_quantile
        )
        if not 0.0 <= self.positive_flux_quantile <= 1.0:
            raise ValueError("positive_flux_quantile must be within [0,1]")
        if not 0.0 <= self.negative_flux_quantile <= 1.0:
            raise ValueError("negative_flux_quantile must be within [0,1]")
        self.gate_temperature = _require_finite_scalar("gate_temperature", gate_temperature, positive=True)
        self.gate_smoothing_frames = self._odd_width("gate_smoothing_frames", gate_smoothing_frames)
        self.mask_smoothing_frequency_bins = self._odd_width(
            "mask_smoothing_frequency_bins", mask_smoothing_frequency_bins
        )
        self.mask_smoothing_time_frames = self._odd_width(
            "mask_smoothing_time_frames", mask_smoothing_time_frames
        )
        for name, mode in (
            ("gate_smoothing_pad_mode", gate_smoothing_pad_mode),
            ("mask_smoothing_pad_mode", mask_smoothing_pad_mode),
        ):
            if mode not in {"reflect", "replicate"}:
                raise ValueError(f"{name} must be explicitly set to reflect or replicate")
        if frequency_common_mode_policy not in {"retain_reference_coded", "remove_per_frame"}:
            raise ValueError("frequency common-mode policy must be retain_reference_coded or remove_per_frame")
        self.frequency_common_mode_policy = frequency_common_mode_policy
        self.max_abs_mask_db = _require_finite_scalar("max_abs_mask_db", max_abs_mask_db, positive=True)
        self.target_rms = _require_finite_scalar("target_rms", target_rms, positive=True)
        self.rms_epsilon = _require_finite_scalar("rms_epsilon", rms_epsilon, positive=True)
        if peak_ceiling is not None:
            peak_ceiling = _require_finite_scalar("peak_ceiling", peak_ceiling, positive=True)
        self.peak_ceiling = peak_ceiling
        self.baseline_match_atol = _require_finite_scalar("baseline_match_atol", baseline_match_atol)
        if self.baseline_match_atol < 0.0:
            raise ValueError("baseline_match_atol must be nonnegative")
        if video_fps_num <= 0 or video_fps_den <= 0:
            raise ValueError("video frame rate numerator and denominator must be positive")
        samples_per_frame_numerator = sample_rate_hz * video_fps_den
        if samples_per_frame_numerator % video_fps_num:
            raise ValueError("video frame boundaries must land on integer source samples")
        self.samples_per_video_frame = samples_per_frame_numerator // video_fps_num
        self.frame_power_floor = _require_finite_scalar("frame_power_floor", frame_power_floor, positive=True)

        self.sample_rate_hz = int(sample_rate_hz)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.window_kind = window_kind
        self.window_periodic = window_periodic
        self.stft_center = stft_center
        self.stft_pad_mode = stft_pad_mode
        self.band_edges_hz = edges
        self.reference_band_index = int(reference_band_index)
        self.gate_smoothing_pad_mode = gate_smoothing_pad_mode
        self.mask_smoothing_pad_mode = mask_smoothing_pad_mode
        self.video_fps_num = int(video_fps_num)
        self.video_fps_den = int(video_fps_den)

        frequencies = torch.arange(n_fft // 2 + 1, dtype=torch.float32) * (
            float(sample_rate_hz) / float(n_fft)
        )
        basis = torch.zeros((BAND_COUNT, frequencies.numel()), dtype=torch.float32)
        for band_index, (lower, upper) in enumerate(zip(edges, edges[1:])):
            if band_index == BAND_COUNT - 1:
                selected = (frequencies >= lower) & (frequencies <= upper)
            else:
                selected = (frequencies >= lower) & (frequencies < upper)
            basis[band_index, selected] = 1.0
        if not bool(torch.all(basis.sum(dim=0) == 1.0).item()):
            raise ValueError("band edges must assign every one-sided frequency bin exactly once")
        band_bin_counts = basis.sum(dim=1).to(dtype=torch.int64)
        if bool(torch.any(band_bin_counts == 0).item()):
            raise ValueError("all four configured frequency bands must contain at least one STFT bin")
        self.register_buffer("frequency_hz", frequencies)
        self.register_buffer("band_basis", basis)
        self.register_buffer("band_bin_counts", band_bin_counts)
        self.register_buffer(
            "stft_window",
            torch.hann_window(win_length, periodic=window_periodic, dtype=torch.float32),
        )

        # Reference coding removes one redundant common offset per temporal
        # component. There are exactly nine trainable scalars and no hidden gain.
        self.band_contrasts_db = nn.Parameter(torch.zeros((len(TEMPORAL_COMPONENTS), 3)))

    @staticmethod
    def _odd_width(name: str, value: int) -> int:
        if not isinstance(value, int) or value <= 0 or value % 2 != 1:
            raise ValueError(f"{name} must be a positive odd integer")
        return value

    def _four_band_weights(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        free = self.band_contrasts_db.to(device=device, dtype=dtype)
        reference = torch.zeros((free.shape[0], 1), dtype=dtype, device=device)
        return torch.cat(
            (free[:, : self.reference_band_index], reference, free[:, self.reference_band_index :]),
            dim=1,
        )

    def _source_clock_flux(self, source_spectrum: Tensor) -> dict[str, Tensor]:
        magnitude = source_spectrum.abs()
        frame_total = magnitude.sum(dim=0)
        active_frame_mask = frame_total > self.activity_floor
        if magnitude.shape[1] == 1:
            positive = magnitude.new_zeros((1,))
            negative = magnitude.new_zeros((1,))
            previous_frame_total = frame_total.new_zeros((1,))
            denominator = frame_total.new_zeros((1,))
            valid_transition_mask = torch.zeros((1,), dtype=torch.bool, device=magnitude.device)
        else:
            changes = magnitude[:, 1:] - magnitude[:, :-1]
            previous_frame_total = torch.cat((frame_total.new_zeros((1,)), frame_total[:-1]))
            denominator = torch.cat(
                (frame_total.new_zeros((1,)), frame_total[:-1].clamp_min(self.flux_denominator_floor))
            )
            positive_changes = torch.relu(changes).sum(dim=0) / denominator[1:]
            negative_changes = torch.relu(-changes).sum(dim=0) / denominator[1:]
            positive_changes = positive_changes / self.flux_normalization_scale
            negative_changes = negative_changes / self.flux_normalization_scale
            invalid_first = magnitude.new_zeros((1,))
            positive = torch.cat((invalid_first, positive_changes))
            negative = torch.cat((invalid_first, negative_changes))
            valid_pair = active_frame_mask[:-1] & active_frame_mask[1:]
            # Transition k compares frames k and k+1. Align the previous-frame
            # denominator test to that same T-1 transition axis.
            valid_pair = valid_pair & (
                previous_frame_total[1:] > self.flux_denominator_floor
            )
            valid_transition_mask = torch.cat(
                (torch.zeros((1,), dtype=torch.bool, device=magnitude.device), valid_pair)
            )
            # Invalid positions carry finite placeholders only. The validity
            # mask is authoritative for quantiles, gates, and later reporting.
            positive = torch.where(valid_transition_mask, positive, 0.0)
            negative = torch.where(valid_transition_mask, negative, 0.0)
        return {
            "magnitude": magnitude,
            "frame_magnitude_sum": frame_total,
            "active_frame_mask": active_frame_mask,
            "previous_frame_magnitude_sum": previous_frame_total,
            "flux_denominator": denominator,
            "positive_relative_flux": positive,
            "negative_relative_flux": negative,
            "valid_transition_mask": valid_transition_mask,
        }

    def _make_gate(self, flux: Tensor, valid: Tensor, quantile: float) -> tuple[Tensor, Tensor, int, bool]:
        selected = flux[valid]
        if selected.numel() == 0:
            zero = flux.new_zeros(())
            gate = flux.new_zeros(flux.shape)
            return gate, zero, 0, False
        threshold = torch.quantile(selected, quantile)
        raw_gate = torch.sigmoid((flux - threshold) / self.gate_temperature)
        raw_gate = raw_gate * valid.to(dtype=flux.dtype)
        smoothed_gate = _smooth_gate(
            raw_gate, self.gate_smoothing_frames, self.gate_smoothing_pad_mode
        )
        smoothed_gate = smoothed_gate * valid.to(dtype=flux.dtype)
        return smoothed_gate, threshold, int(selected.numel()), True

    def _gate_dc_control(self, gate: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        valid_float = valid.to(dtype=gate.dtype)
        count = valid_float.sum().clamp_min(1.0)
        dc = (gate * valid_float).sum() / count
        return dc * valid_float, dc

    def _rms_peak_baseline(self, samples: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        rms_square = samples.square().mean()
        safe_rms = torch.sqrt(rms_square.clamp_min(self.rms_epsilon * self.rms_epsilon))
        active = rms_square > self.rms_epsilon * self.rms_epsilon
        requested = self.target_rms / safe_rms
        rms_gain = torch.where(active, requested, torch.ones_like(requested))
        rms_scaled = samples * rms_gain
        peak = rms_scaled.abs().amax()
        if self.peak_ceiling is None:
            peak_gain = torch.ones_like(peak)
            peak_limited = torch.zeros((), dtype=torch.bool, device=samples.device)
        else:
            peak_active = peak > self.peak_ceiling
            safe_peak = peak.clamp_min(self.rms_epsilon)
            requested_peak_gain = self.peak_ceiling / safe_peak
            peak_gain = torch.where(peak_active, requested_peak_gain, torch.ones_like(requested_peak_gain))
            peak_limited = peak_active
        result = rms_scaled * peak_gain
        info = {
            "input_rms": torch.sqrt(rms_square),
            "rms_gain": rms_gain,
            "peak_before_uniform_cap": peak,
            "uniform_peak_gain": peak_gain,
            "peak_limited": peak_limited,
            "output_rms": torch.sqrt(result.square().mean()),
            "output_peak": result.abs().amax(),
        }
        return result, info

    def _frame_power(self, samples: Tensor) -> tuple[Tensor, Tensor]:
        complete_frames = samples.numel() // self.samples_per_video_frame
        used_samples = complete_frames * self.samples_per_video_frame
        if complete_frames == 0:
            empty = samples.new_zeros((0,))
            return empty, torch.zeros((0,), dtype=torch.bool, device=samples.device)
        frames = samples[:used_samples].reshape(complete_frames, self.samples_per_video_frame)
        power = frames.square().mean(dim=1)
        valid = power > self.frame_power_floor
        return power, valid

    def forward(
        self,
        original_pcm_float32: Tensor,
        canonical_rms_baseline_pcm_float32: Tensor,
        *,
        gate_mode: str = "dynamic",
    ) -> dict[str, Any]:
        """Transform one mono utterance against its sealed RMS baseline.

        The caller must bind the supplied baseline to this exact original PCM,
        target RMS and peak policy. This method checks shape, dtype and finite
        values; source-file/hash lineage is the caller's responsibility.
        """

        if gate_mode not in {"dynamic", "gate_dc"}:
            raise ValueError("gate_mode must be dynamic or gate_dc")

        source = original_pcm_float32
        baseline = canonical_rms_baseline_pcm_float32
        if source.ndim != 1 or source.numel() == 0 or source.dtype != torch.float32:
            raise ValueError("original PCM must be nonempty mono float32")
        if baseline.shape != source.shape or baseline.dtype != torch.float32:
            raise ValueError("canonical RMS baseline must be same-length mono float32")
        if baseline.device != source.device:
            raise ValueError("source and canonical RMS baseline must be on the same device")
        if not bool(torch.isfinite(source).all().item()) or not bool(torch.isfinite(baseline).all().item()):
            raise ValueError("source and canonical RMS baseline must be finite")
        if self.stft_center and self.stft_pad_mode == "reflect" and source.numel() <= self.n_fft // 2:
            raise ValueError("source is too short for the configured centered reflect-padded STFT")
        if not self.stft_center and source.numel() < self.n_fft:
            raise ValueError("source is shorter than n_fft for the configured uncentered STFT")

        window = self.stft_window.to(device=source.device, dtype=source.dtype)
        source_spectrum = torch.stft(
            source,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.stft_center,
            pad_mode=self.stft_pad_mode,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        flux = self._source_clock_flux(source_spectrum)
        positive_gate, positive_threshold, positive_count_scalar, positive_has_activity = self._make_gate(
            flux["positive_relative_flux"], flux["valid_transition_mask"], self.positive_flux_quantile
        )
        negative_gate, negative_threshold, negative_count_scalar, negative_has_activity = self._make_gate(
            flux["negative_relative_flux"], flux["valid_transition_mask"], self.negative_flux_quantile
        )
        positive_gate_dynamic = positive_gate
        negative_gate_dynamic = negative_gate
        positive_gate_dc, positive_gate_dc_mean = self._gate_dc_control(
            positive_gate_dynamic, flux["valid_transition_mask"]
        )
        negative_gate_dc, negative_gate_dc_mean = self._gate_dc_control(
            negative_gate_dynamic, flux["valid_transition_mask"]
        )
        if gate_mode == "gate_dc":
            positive_gate, negative_gate = positive_gate_dc, negative_gate_dc

        weights = self._four_band_weights(source.dtype, source.device)
        basis = self.band_basis.to(device=source.device, dtype=source.dtype)
        band_masks = basis.transpose(0, 1) @ weights.transpose(0, 1)
        static_mask = band_masks[:, 0:1]
        positive_mask = band_masks[:, 1:2] * positive_gate[None, :]
        negative_mask = band_masks[:, 2:3] * negative_gate[None, :]
        raw_mask_db = static_mask + positive_mask + negative_mask
        smoothed_frequency = _smooth_mask_axis(
            raw_mask_db,
            self.mask_smoothing_frequency_bins,
            "frequency",
            self.mask_smoothing_pad_mode,
        )
        smoothed_mask_db = _smooth_mask_axis(
            smoothed_frequency,
            self.mask_smoothing_time_frames,
            "time",
            self.mask_smoothing_pad_mode,
        )
        common_mode_before_policy = smoothed_mask_db.mean(dim=0, keepdim=True)
        if self.frequency_common_mode_policy == "remove_per_frame":
            policy_mask_db = smoothed_mask_db - common_mode_before_policy
        else:
            policy_mask_db = smoothed_mask_db

        max_abs_before_projection = policy_mask_db.abs().amax()
        normalized_extent = max_abs_before_projection / self.max_abs_mask_db
        safe_extent = normalized_extent.clamp_min(torch.finfo(source.dtype).tiny)
        projection_scale = torch.where(
            normalized_extent > 0.0,
            torch.tanh(normalized_extent) / safe_extent,
            torch.ones_like(normalized_extent),
        )
        projected_mask_db = policy_mask_db * projection_scale
        magnitude_gain = torch.pow(source.new_tensor(10.0), projected_mask_db / 20.0)
        delta_spectrum = source_spectrum * (magnitude_gain - 1.0)
        delta_pcm = torch.istft(
            delta_spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.stft_center,
            normalized=False,
            onesided=True,
            length=source.numel(),
            return_complex=False,
        )
        raw_candidate_pcm = source + delta_pcm
        normalized_candidate_pcm, candidate_rms_info = self._rms_peak_baseline(raw_candidate_pcm)
        normalized_source_pcm, source_rms_info = self._rms_peak_baseline(source)
        baseline_match_error = (baseline - normalized_source_pcm).abs().amax()
        if baseline_match_error.item() > self.baseline_match_atol:
            raise ValueError(
                "canonical RMS baseline differs from the configured source/RMS/peak policy "
                "by more than baseline_match_atol"
            )

        # No zero-parameter bypass: at zero contrasts delta_pcm is exactly zero,
        # so the parenthesized residual is zero while gradients still flow
        # through normalized_candidate_pcm for nonzero perturbations.
        output_pcm = baseline + (normalized_candidate_pcm - normalized_source_pcm)
        if output_pcm.shape != source.shape or not bool(torch.isfinite(output_pcm).all().item()):
            raise FloatingPointError("head output violated finite exact-length PCM contract")
        final_output_rms = torch.sqrt(output_pcm.square().mean())
        final_output_peak = output_pcm.abs().amax()
        final_output_rms_error = final_output_rms - self.target_rms
        final_peak_excess = (
            None
            if self.peak_ceiling is None
            else torch.relu(final_output_peak - self.peak_ceiling)
        )

        source_power, frame_power_valid = self._frame_power(source)
        candidate_power, _ = self._frame_power(raw_candidate_pcm)
        baseline_power, _ = self._frame_power(baseline)
        output_power, _ = self._frame_power(output_pcm)
        power_ratio_valid = frame_power_valid & (source_power > self.frame_power_floor)
        candidate_power_ratio = torch.where(
            power_ratio_valid,
            candidate_power / source_power.clamp_min(self.frame_power_floor),
            0.0,
        )
        output_power_ratio = torch.where(
            power_ratio_valid,
            output_power / source_power.clamp_min(self.frame_power_floor),
            0.0,
        )
        stft_frame_centers_samples = torch.arange(
            source_spectrum.shape[1], device=source.device, dtype=torch.int64
        ) * self.hop_length
        if not self.stft_center:
            stft_frame_centers_samples = stft_frame_centers_samples + self.n_fft // 2
        video_frame_centers_samples = (
            torch.arange(source_power.numel(), device=source.device, dtype=torch.int64)
            * self.samples_per_video_frame
            + self.samples_per_video_frame // 2
        )
        parameter_values = self.band_contrasts_db
        return {
            "schema": SCHEMA,
            "output_pcm_float32": output_pcm,
            "canonical_rms_baseline_pcm_float32": baseline,
            "source_sample_count": int(source.numel()),
            "source_stft": source_spectrum,
            "constructed_stft_delta": delta_spectrum,
            "raw_candidate_pcm_before_rms": raw_candidate_pcm,
            "delta_pcm_before_rms": delta_pcm,
            "raw_log_magnitude_mask_db": raw_mask_db,
            "smoothed_log_magnitude_mask_db": smoothed_mask_db,
            "frequency_common_mode_before_policy_db": common_mode_before_policy,
            "policy_mask_db_before_smooth_bound": policy_mask_db,
            "projected_log_magnitude_mask_db": projected_mask_db,
            "magnitude_gain": magnitude_gain,
            "band_basis": basis,
            "band_bin_counts": self.band_bin_counts,
            "reference_coded_band_weights_db": weights,
            "learned_band_contrasts_db": parameter_values,
            "positive_relative_flux": flux["positive_relative_flux"],
            "negative_relative_flux": flux["negative_relative_flux"],
            "active_frame_mask": flux["active_frame_mask"],
            "valid_transition_mask": flux["valid_transition_mask"],
            "previous_frame_magnitude_sum": flux["previous_frame_magnitude_sum"],
            "flux_denominator": flux["flux_denominator"],
            "gate_mode": gate_mode,
            "positive_flux_gate": positive_gate,
            "negative_flux_gate": negative_gate,
            "positive_flux_gate_dynamic": positive_gate_dynamic,
            "negative_flux_gate_dynamic": negative_gate_dynamic,
            "positive_flux_gate_dc_mean": positive_gate_dc_mean,
            "negative_flux_gate_dc_mean": negative_gate_dc_mean,
            "positive_flux_threshold": positive_threshold,
            "negative_flux_threshold": negative_threshold,
            "positive_flux_valid_transition_count": positive_count_scalar,
            "negative_flux_valid_transition_count": negative_count_scalar,
            "positive_flux_has_valid_support": bool(positive_has_activity),
            "negative_flux_has_valid_support": bool(negative_has_activity),
            "smooth_projection_scale": projection_scale,
            "preprojection_max_abs_mask_db": max_abs_before_projection,
            "postprojection_max_abs_mask_db": projected_mask_db.abs().amax(),
            "stft_frame_centers_source_samples": stft_frame_centers_samples,
            "video_frame_centers_source_samples": video_frame_centers_samples,
            "physical_frame_power": {
                "source_original_pcm": source_power,
                "candidate_before_rms": candidate_power,
                "canonical_rms_baseline": baseline_power,
                "output_pcm": output_power,
                "candidate_to_source_ratio": candidate_power_ratio,
                "output_to_source_ratio": output_power_ratio,
                "valid_source_power_mask": power_ratio_valid,
                "trailing_unframed_sample_count": int(
                    source.numel() % self.samples_per_video_frame
                ),
            },
            "normalization": {
                "candidate": candidate_rms_info,
                "source_rms_baseline_internal": source_rms_info,
                "canonical_baseline_input_rms": torch.sqrt(baseline.square().mean()),
                "canonical_baseline_max_abs": baseline.abs().amax(),
                "canonical_baseline_max_abs_match_error": baseline_match_error,
                "declared_target_rms": source.new_tensor(self.target_rms),
                "declared_peak_ceiling": self.peak_ceiling,
                "baseline_match_atol": source.new_tensor(self.baseline_match_atol),
                "final_output_rms": final_output_rms,
                "final_output_rms_error_to_target": final_output_rms_error,
                "final_output_peak": final_output_peak,
                "final_output_peak_ceiling_excess": final_peak_excess,
                "normalization_residual_added_to_canonical_baseline": (
                    normalized_candidate_pcm - normalized_source_pcm
                ),
            },
            "configuration": self.configuration(),
        }

    def configuration(self) -> Mapping[str, Any]:
        """Return every nonlearned behavior setting for a caller's sealed plan."""

        return {
            "schema": SCHEMA,
            "sample_rate_hz": self.sample_rate_hz,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "window_kind": self.window_kind,
            "window_periodic": self.window_periodic,
            "stft_center": self.stft_center,
            "stft_pad_mode": self.stft_pad_mode,
            "stft_normalized": False,
            "stft_onesided": True,
            "stft_return_complex": True,
            "band_edges_hz": list(self.band_edges_hz),
            "reference_band_index": self.reference_band_index,
            "band_bin_counts": self.band_bin_counts.detach().cpu().tolist(),
            "flux_denominator_floor": self.flux_denominator_floor,
            "activity_floor": self.activity_floor,
            "flux_normalization_scale": self.flux_normalization_scale,
            "positive_flux_quantile": self.positive_flux_quantile,
            "negative_flux_quantile": self.negative_flux_quantile,
            "gate_temperature": self.gate_temperature,
            "gate_smoothing_frames": self.gate_smoothing_frames,
            "gate_smoothing_pad_mode": self.gate_smoothing_pad_mode,
            "mask_smoothing_frequency_bins": self.mask_smoothing_frequency_bins,
            "mask_smoothing_time_frames": self.mask_smoothing_time_frames,
            "mask_smoothing_pad_mode": self.mask_smoothing_pad_mode,
            "frequency_common_mode_policy": self.frequency_common_mode_policy,
            "max_abs_mask_db": self.max_abs_mask_db,
            "mask_projection": "global_smooth_radial_tanh",
            "target_rms": self.target_rms,
            "rms_epsilon": self.rms_epsilon,
            "peak_ceiling": self.peak_ceiling,
            "baseline_match_atol": self.baseline_match_atol,
            "video_fps_num": self.video_fps_num,
            "video_fps_den": self.video_fps_den,
            "samples_per_video_frame": self.samples_per_video_frame,
            "frame_power_floor": self.frame_power_floor,
            "stored_trainable_scalar_count": TRAINABLE_SCALAR_COUNT,
            "reference_source_pins": dict(REFERENCE_SOURCE_PINS),
        }


__all__ = [
    "BAND_COUNT",
    "PREPARATION_ONLY",
    "REFERENCE_SOURCE_PINS",
    "SCHEMA",
    "SharedRecipientTransitionHeadV3",
    "TEMPORAL_COMPONENTS",
    "TRAINABLE_SCALAR_COUNT",
]
