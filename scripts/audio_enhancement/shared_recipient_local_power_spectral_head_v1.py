"""Source-conditioned local-envelope plus spectral-shape candidate head.

Preparation-only signal construction. Importing this module does not create a
model, read audio, or run a transform. It reuses the explicitly configured V3
STFT, gate, clock, and canonical-RMS helpers, while adding two bounded local
envelope coordinates. It has no frontend/backend identifier input and no
optimizer, renderer, or scorer dependency.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .shared_recipient_transition_head_v3 import (
    PREPARATION_ONLY as V3_PREPARATION_ONLY,
    REFERENCE_SOURCE_PINS as V3_REFERENCE_SOURCE_PINS,
    SCHEMA as V3_SCHEMA,
    SharedRecipientTransitionHeadV3,
    TEMPORAL_COMPONENTS,
    _smooth_mask_axis,
)


SCHEMA = "cross_tfg_shared_recipient_local_power_spectral_head_v1"
PREPARATION_ONLY = True
TRAINABLE_SCALAR_COUNT = 11
SHAPE_SCALAR_COUNT = 9
ENVELOPE_SCALAR_COUNT = 2
COMMON_DOSE_RULE = "kappa=min(1,max_abs_mask_db/(2*A+B)); A=max_abs(policy_shape_mask_db); B=max_abs(centered_raw_envelope_db)"
ONE_SIDED_POWER_RULE = "real_onesided_parseval_bins_dc_and_nyquist_1_interior_2"
ENVELOPE_CENTER_RULE = "subtract_arithmetic_mean_on_valid_source_transitions_then_zero_invalid"

REFERENCE_SOURCE_PINS = {
    **dict(V3_REFERENCE_SOURCE_PINS),
    "shared_recipient_transition_head_v3.py": "77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87",
}

_V3_INIT_KEYS = (
    "sample_rate_hz",
    "n_fft",
    "hop_length",
    "win_length",
    "window_kind",
    "window_periodic",
    "stft_center",
    "stft_pad_mode",
    "stft_normalized",
    "stft_onesided",
    "stft_return_complex",
    "band_edges_hz",
    "reference_band_index",
    "flux_denominator_floor",
    "activity_floor",
    "flux_normalization_scale",
    "positive_flux_quantile",
    "negative_flux_quantile",
    "gate_temperature",
    "gate_smoothing_frames",
    "gate_smoothing_pad_mode",
    "mask_smoothing_frequency_bins",
    "mask_smoothing_time_frames",
    "mask_smoothing_pad_mode",
    "frequency_common_mode_policy",
    "max_abs_mask_db",
    "target_rms",
    "rms_epsilon",
    "peak_ceiling",
    "baseline_match_atol",
    "video_fps_num",
    "video_fps_den",
    "frame_power_floor",
)


def _positive_finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


class SharedRecipientLocalPowerSpectralHeadV1(SharedRecipientTransitionHeadV3):
    """An 11-scalar source-only head with RMS, envelope, shape, and full cells.

    The first nine coordinates use the V3 reference-coded four-band basis and
    its three source-clock temporal components. Their effective design rank
    depends on the observed source. The last two coordinates control a centered
    scalar envelope driven by the same positive/negative source-relative-flux
    gates. The global per-utterance ``kappa`` is shared by all active cells.
    """

    def __init__(
        self,
        *,
        v3_configuration: Mapping[str, Any],
        shape_coordinate_scale_db: float,
        envelope_coordinate_scale_db: float,
        stft_power_activity_floor: float,
        one_sided_power_bin_policy: str,
        envelope_centering_policy: str,
    ) -> None:
        if not V3_PREPARATION_ONLY:
            raise RuntimeError("the referenced V3 module is not marked preparation-only")
        if not isinstance(v3_configuration, Mapping):
            raise TypeError("v3_configuration must be a sealed mapping of explicit V3 settings")
        if v3_configuration.get("schema") != V3_SCHEMA:
            raise ValueError("v3_configuration schema does not match the pinned V3 head")
        missing = [key for key in _V3_INIT_KEYS if key not in v3_configuration]
        if missing:
            raise ValueError(f"v3_configuration is missing explicit settings: {missing}")
        if int(v3_configuration.get("stored_trainable_scalar_count", -1)) != SHAPE_SCALAR_COUNT:
            raise ValueError("the referenced V3 configuration must carry exactly nine shape scalars")
        reference_pins = v3_configuration.get("reference_source_pins")
        if not isinstance(reference_pins, Mapping):
            raise ValueError("V3 reference source pins are required")
        mismatched_pins = {
            path: {"expected": expected, "actual": reference_pins.get(path)}
            for path, expected in V3_REFERENCE_SOURCE_PINS.items()
            if reference_pins.get(path) != expected
        }
        if mismatched_pins:
            raise ValueError(f"V3 source pins do not match this source-only extension: {mismatched_pins}")

        v3_init = {key: v3_configuration[key] for key in _V3_INIT_KEYS}
        if float(v3_init["max_abs_mask_db"]) != 3.0:
            raise ValueError("the common total-dose bound is explicitly fixed at 3 dB")
        if v3_init["peak_ceiling"] is not None:
            raise ValueError("this candidate preserves the canonical no-peak-cap policy")

        if one_sided_power_bin_policy != ONE_SIDED_POWER_RULE:
            raise ValueError("one_sided_power_bin_policy must explicitly select the Parseval bin weights")
        if envelope_centering_policy != ENVELOPE_CENTER_RULE:
            raise ValueError("envelope_centering_policy must explicitly select valid-transition centering")
        self.shape_coordinate_scale_db = _positive_finite(
            "shape_coordinate_scale_db", shape_coordinate_scale_db
        )
        self.envelope_coordinate_scale_db = _positive_finite(
            "envelope_coordinate_scale_db", envelope_coordinate_scale_db
        )
        self.stft_power_activity_floor = _positive_finite(
            "stft_power_activity_floor", stft_power_activity_floor
        )
        self.one_sided_power_bin_policy = one_sided_power_bin_policy
        self.envelope_centering_policy = envelope_centering_policy
        self._sealed_v3_configuration = dict(v3_configuration)

        super().__init__(**v3_init)

        # Replace V3's nine direct-dB parameters with nine bounded coordinates.
        # The shape stays nine-dimensional; the two envelope coordinates make
        # eleven total. No branch bypasses the residual path at zero.
        del self.band_contrasts_db
        self.shape_coordinates = nn.Parameter(torch.zeros((len(TEMPORAL_COMPONENTS), 3)))
        self.envelope_coordinates = nn.Parameter(torch.zeros((ENVELOPE_SCALAR_COUNT,)))

    def _bounded_shape_weights(self, dtype: torch.dtype, device: torch.device) -> Tensor:
        free = self.shape_coordinate_scale_db * torch.tanh(
            self.shape_coordinates.to(device=device, dtype=dtype)
        )
        reference = torch.zeros((free.shape[0], 1), dtype=dtype, device=device)
        return torch.cat(
            (free[:, : self.reference_band_index], reference, free[:, self.reference_band_index :]),
            dim=1,
        )

    def _one_sided_power_weights(self, bins: int, dtype: torch.dtype, device: torch.device) -> Tensor:
        expected_bins = self.n_fft // 2 + 1
        if bins != expected_bins:
            raise ValueError("source STFT frequency dimension does not match the configured n_fft")
        weights = torch.full((bins,), 2.0, dtype=dtype, device=device)
        weights[0] = 1.0
        weights[-1] = 1.0
        return weights

    def _construct_cell(
        self,
        source: Tensor,
        baseline: Tensor,
        normalized_source: Tensor,
        source_spectrum: Tensor,
        window: Tensor,
        mask_db: Tensor,
        bin_weights: Tensor,
        source_stft_power: Tensor,
        stft_power_support: Tensor,
    ) -> dict[str, Any]:
        magnitude_gain = torch.pow(source.new_tensor(10.0), mask_db / 20.0)
        constructed_stft = source_spectrum * magnitude_gain
        constructed_stft_delta = source_spectrum * (magnitude_gain - 1.0)
        delta_pcm = torch.istft(
            constructed_stft_delta,
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
        raw_pcm = source + delta_pcm
        normalized_pcm, normalization = self._rms_peak_baseline(raw_pcm)
        # Preserve a gradient path at zero: the residual vanishes exactly, but
        # the candidate branch remains in the graph for nonzero coordinates.
        output_pcm = baseline + (normalized_pcm - normalized_source)
        if output_pcm.shape != source.shape or not bool(torch.isfinite(output_pcm).all().item()):
            raise FloatingPointError("candidate cell violated finite exact-length float32 output")
        frame_power, frame_valid = self._frame_power(output_pcm)
        source_frame_power, _ = self._frame_power(source)
        ratio_valid = frame_valid & (source_frame_power > self.frame_power_floor)
        ratio = torch.where(
            ratio_valid,
            frame_power / source_frame_power.clamp_min(self.frame_power_floor),
            torch.zeros_like(frame_power),
        )
        return {
            "output_pcm_float32": output_pcm,
            "raw_candidate_pcm_before_rms": raw_pcm,
            "delta_pcm_before_rms": delta_pcm,
            "constructed_stft": constructed_stft,
            "constructed_stft_delta": constructed_stft_delta,
            "log_magnitude_mask_db": mask_db,
            "magnitude_gain": magnitude_gain,
            "one_sided_weighted_stft_coefficient_power_by_frame": (
                constructed_stft.abs().square() * bin_weights[:, None]
            ).sum(dim=0),
            "one_sided_weighted_stft_power_ratio_to_source": torch.where(
                stft_power_support,
                (constructed_stft.abs().square() * bin_weights[:, None]).sum(dim=0)
                / source_stft_power.clamp_min(torch.finfo(source.dtype).tiny),
                torch.zeros_like(source_stft_power),
            ),
            "stft_power_support_mask": stft_power_support,
            "waveform_40ms_power": frame_power,
            "waveform_40ms_power_ratio_to_source": ratio,
            "waveform_40ms_valid_source_power_mask": ratio_valid,
            "normalization": normalization,
            "output_sample_count": int(output_pcm.numel()),
            "output_rms": torch.sqrt(output_pcm.square().mean()),
            "output_peak": output_pcm.abs().amax(),
        }

    def forward(
        self,
        original_pcm_float32: Tensor,
        canonical_rms_baseline_pcm_float32: Tensor,
    ) -> dict[str, Any]:
        source = original_pcm_float32
        baseline = canonical_rms_baseline_pcm_float32
        if source.ndim != 1 or source.numel() == 0 or source.dtype != torch.float32:
            raise ValueError("original PCM must be nonempty mono float32")
        if baseline.shape != source.shape or baseline.dtype != torch.float32:
            raise ValueError("canonical RMS baseline must be same-length mono float32")
        if baseline.device != source.device:
            raise ValueError("source and canonical RMS baseline must share a device")
        if not bool(torch.isfinite(source).all().item()) or not bool(torch.isfinite(baseline).all().item()):
            raise ValueError("source and canonical RMS baseline must be finite")
        if source.numel() <= self.n_fft // 2 and self.stft_center and self.stft_pad_mode == "reflect":
            raise ValueError("source is too short for centered reflect-padded STFT")

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
        baseline_spectrum = torch.stft(
            baseline,
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
        positive_gate, positive_threshold, positive_count, positive_supported = self._make_gate(
            flux["positive_relative_flux"], flux["valid_transition_mask"], self.positive_flux_quantile
        )
        negative_gate, negative_threshold, negative_count, negative_supported = self._make_gate(
            flux["negative_relative_flux"], flux["valid_transition_mask"], self.negative_flux_quantile
        )

        basis = self.band_basis.to(device=source.device, dtype=source.dtype)
        weights_db = self._bounded_shape_weights(source.dtype, source.device)
        band_masks_db = basis.transpose(0, 1) @ weights_db.transpose(0, 1)
        raw_shape_db = (
            band_masks_db[:, 0:1]
            + band_masks_db[:, 1:2] * positive_gate[None, :]
            + band_masks_db[:, 2:3] * negative_gate[None, :]
        )
        smoothed_shape_db = _smooth_mask_axis(
            raw_shape_db,
            self.mask_smoothing_frequency_bins,
            "frequency",
            self.mask_smoothing_pad_mode,
        )
        smoothed_shape_db = _smooth_mask_axis(
            smoothed_shape_db,
            self.mask_smoothing_time_frames,
            "time",
            self.mask_smoothing_pad_mode,
        )
        frequency_common_mode_db = smoothed_shape_db.mean(dim=0, keepdim=True)
        if self.frequency_common_mode_policy == "remove_per_frame":
            policy_shape_db = smoothed_shape_db - frequency_common_mode_db
        else:
            policy_shape_db = smoothed_shape_db

        envelope_coefficients_db = self.envelope_coordinate_scale_db * torch.tanh(
            self.envelope_coordinates.to(device=source.device, dtype=source.dtype)
        )
        raw_envelope_db = (
            envelope_coefficients_db[0] * positive_gate
            + envelope_coefficients_db[1] * negative_gate
        )
        valid_transition = flux["valid_transition_mask"]
        valid_float = valid_transition.to(dtype=source.dtype)
        valid_count = valid_float.sum()
        envelope_mean_db = (raw_envelope_db * valid_float).sum() / valid_count.clamp_min(1.0)
        centered_envelope_db = torch.where(
            valid_transition,
            raw_envelope_db - envelope_mean_db,
            torch.zeros_like(raw_envelope_db),
        )

        shape_extent_db = policy_shape_db.abs().amax()
        envelope_extent_db = centered_envelope_db.abs().amax()
        total_extent_db = 2.0 * shape_extent_db + envelope_extent_db
        safe_extent_db = total_extent_db.clamp_min(torch.finfo(source.dtype).tiny)
        dose_scale = torch.minimum(
            source.new_ones(()), source.new_tensor(self.max_abs_mask_db) / safe_extent_db
        )
        dose_scale = torch.where(total_extent_db > 0.0, dose_scale, source.new_ones(()))

        bin_weights = self._one_sided_power_weights(
            source_spectrum.shape[0], source.dtype, source.device
        )
        source_bin_power = source_spectrum.abs().square()
        source_stft_power = (source_bin_power * bin_weights[:, None]).sum(dim=0)
        uncorrected_shape_gain = torch.pow(
            source.new_tensor(10.0), (dose_scale * policy_shape_db) / 20.0
        )
        masked_stft_power = (
            source_bin_power
            * bin_weights[:, None]
            * uncorrected_shape_gain.square()
        ).sum(dim=0)
        stft_power_support = (
            torch.isfinite(source_stft_power)
            & torch.isfinite(masked_stft_power)
            & (source_stft_power > self.stft_power_activity_floor)
            & (masked_stft_power > 0.0)
        )
        safe_source_power = source_stft_power.clamp_min(torch.finfo(source.dtype).tiny)
        safe_masked_power = masked_stft_power.clamp_min(torch.finfo(source.dtype).tiny)
        power_compensation_db = 10.0 * torch.log10(safe_source_power / safe_masked_power)
        power_compensation_db = torch.where(
            stft_power_support, power_compensation_db, torch.zeros_like(power_compensation_db)
        )

        shape_cell_mask_db = dose_scale * policy_shape_db + power_compensation_db[None, :]
        envelope_cell_mask_db = dose_scale * centered_envelope_db[None, :].expand_as(policy_shape_db)
        full_cell_mask_db = shape_cell_mask_db + envelope_cell_mask_db
        maximum_constructed_mask_db = torch.maximum(
            envelope_cell_mask_db.abs().amax(),
            torch.maximum(shape_cell_mask_db.abs().amax(), full_cell_mask_db.abs().amax()),
        )
        if maximum_constructed_mask_db.item() > self.max_abs_mask_db + 1e-5:
            raise FloatingPointError("common-kappa mask construction exceeded its declared total dB bound")
        normalized_source, source_normalization = self._rms_peak_baseline(source)
        baseline_match_error = (baseline - normalized_source).abs().amax()
        if baseline_match_error.item() > self.baseline_match_atol:
            raise ValueError("canonical RMS baseline does not match the configured source/RMS policy")

        zero_mask = torch.zeros_like(policy_shape_db)
        source_frame_power, source_frame_valid = self._frame_power(source)
        baseline_frame_power, baseline_frame_valid = self._frame_power(baseline)
        rms_frame_ratio_valid = source_frame_valid & (source_frame_power > self.frame_power_floor)
        rms_frame_ratio = torch.where(
            rms_frame_ratio_valid,
            baseline_frame_power / source_frame_power.clamp_min(self.frame_power_floor),
            torch.zeros_like(source_frame_power),
        )
        cells = {
            "rms_reference": {
                "output_pcm_float32": baseline,
                "raw_candidate_pcm_before_rms": source,
                "delta_pcm_before_rms": torch.zeros_like(source),
                "constructed_stft": source_spectrum,
                "constructed_stft_delta": torch.zeros_like(source_spectrum),
                "post_rms_output_stft": baseline_spectrum,
                "log_magnitude_mask_db": zero_mask,
                "magnitude_gain": torch.ones_like(policy_shape_db),
                "one_sided_weighted_stft_coefficient_power_by_frame": (
                    source_spectrum.abs().square() * bin_weights[:, None]
                ).sum(dim=0),
                "one_sided_weighted_stft_power_ratio_to_source": torch.where(
                    stft_power_support,
                    source_stft_power / source_stft_power.clamp_min(torch.finfo(source.dtype).tiny),
                    torch.zeros_like(source_stft_power),
                ),
                "stft_power_support_mask": stft_power_support,
                "waveform_40ms_power": baseline_frame_power,
                "waveform_40ms_power_ratio_to_source": rms_frame_ratio,
                "waveform_40ms_valid_source_power_mask": rms_frame_ratio_valid,
                "normalization": source_normalization,
                "output_sample_count": int(baseline.numel()),
                "output_rms": torch.sqrt(baseline.square().mean()),
                "output_peak": baseline.abs().amax(),
            },
            "envelope_only": self._construct_cell(
                source, baseline, normalized_source, source_spectrum, window, envelope_cell_mask_db,
                bin_weights, source_stft_power, stft_power_support,
            ),
            "shape_only": self._construct_cell(
                source, baseline, normalized_source, source_spectrum, window, shape_cell_mask_db,
                bin_weights, source_stft_power, stft_power_support,
            ),
            "full": self._construct_cell(
                source, baseline, normalized_source, source_spectrum, window, full_cell_mask_db,
                bin_weights, source_stft_power, stft_power_support,
            ),
        }
        for cell_id, cell in cells.items():
            if cell["output_sample_count"] != int(source.numel()):
                raise RuntimeError(f"{cell_id} changed the source sample count")

        stft_frame_centers_samples = (
            torch.arange(source_spectrum.shape[1], device=source.device, dtype=torch.int64)
            * self.hop_length
        )
        if not self.stft_center:
            stft_frame_centers_samples = stft_frame_centers_samples + self.n_fft // 2
        samples_per_video_frame = self.samples_per_video_frame
        video_frame_centers_samples = (
            torch.arange(source.numel() // samples_per_video_frame, device=source.device, dtype=torch.int64)
            * samples_per_video_frame
            + samples_per_video_frame // 2
        )
        output_pcm = cells["full"]["output_pcm_float32"]
        return {
            "schema": SCHEMA,
            "output_pcm_float32": output_pcm,
            "source_sample_count": int(source.numel()),
            "cells": cells,
            "source_stft": source_spectrum,
            "constructed_stft_delta": cells["full"]["constructed_stft_delta"],
            "raw_log_magnitude_shape_mask_db": raw_shape_db,
            "smoothed_shape_mask_db": smoothed_shape_db,
            "frequency_common_mode_before_policy_db": frequency_common_mode_db,
            "policy_shape_mask_db": policy_shape_db,
            "raw_envelope_mask_db": raw_envelope_db,
            "centered_envelope_mask_db": centered_envelope_db,
            "dose_scale_kappa": dose_scale,
            "dose_bound_db": source.new_tensor(self.max_abs_mask_db),
            "preprojection_shape_extent_db": shape_extent_db,
            "preprojection_envelope_extent_db": envelope_extent_db,
            "power_compensation_db": power_compensation_db,
            "stft_source_weighted_coefficient_power": source_stft_power,
            "stft_masked_weighted_coefficient_power_before_compensation": masked_stft_power,
            "stft_power_support_mask": stft_power_support,
            "one_sided_power_bin_weights": bin_weights,
            "band_basis": basis,
            "band_bin_counts": self.band_bin_counts,
            "reference_coded_band_weights_db": weights_db,
            "bounded_shape_coefficients_db": self.shape_coordinate_scale_db
            * torch.tanh(self.shape_coordinates),
            "bounded_envelope_coefficients_db": envelope_coefficients_db,
            "positive_relative_flux": flux["positive_relative_flux"],
            "negative_relative_flux": flux["negative_relative_flux"],
            "active_frame_mask": flux["active_frame_mask"],
            "valid_transition_mask": valid_transition,
            "positive_flux_gate": positive_gate,
            "negative_flux_gate": negative_gate,
            "positive_flux_threshold": positive_threshold,
            "negative_flux_threshold": negative_threshold,
            "positive_flux_valid_transition_count": positive_count,
            "negative_flux_valid_transition_count": negative_count,
            "positive_flux_has_valid_support": bool(positive_supported),
            "negative_flux_has_valid_support": bool(negative_supported),
            "envelope_valid_transition_mean_db": envelope_mean_db,
            "stft_frame_centers_source_samples": stft_frame_centers_samples,
            "video_frame_centers_source_samples": video_frame_centers_samples,
            "trailing_unframed_sample_count": int(source.numel() % samples_per_video_frame),
            "normalization": {
                "canonical_baseline_input_rms": torch.sqrt(baseline.square().mean()),
                "canonical_baseline_max_abs": baseline.abs().amax(),
                "canonical_baseline_max_abs_match_error": baseline_match_error,
                "declared_target_rms": source.new_tensor(self.target_rms),
                "peak_ceiling": None,
                "final_full_output_rms": cells["full"]["output_rms"],
                "final_full_output_peak": cells["full"]["output_peak"],
                "final_full_output_rms_error_to_target": cells["full"]["output_rms"]
                - self.target_rms,
            },
            "configuration": self.configuration(),
        }

    def configuration(self) -> Mapping[str, Any]:
        config = dict(super().configuration())
        config.update(
            {
                "schema": SCHEMA,
                "mask_projection": COMMON_DOSE_RULE,
                "one_sided_power_bin_policy": self.one_sided_power_bin_policy,
                "envelope_centering_policy": self.envelope_centering_policy,
                "shape_coordinate_scale_db": self.shape_coordinate_scale_db,
                "envelope_coordinate_scale_db": self.envelope_coordinate_scale_db,
                "stft_power_activity_floor": self.stft_power_activity_floor,
                "peak_ceiling": None,
                "stored_trainable_scalar_count": TRAINABLE_SCALAR_COUNT,
                "shape_scalar_count": SHAPE_SCALAR_COUNT,
                "envelope_scalar_count": ENVELOPE_SCALAR_COUNT,
                "reference_source_pins": dict(REFERENCE_SOURCE_PINS),
                "v3_reference_source_pins": dict(V3_REFERENCE_SOURCE_PINS),
            }
        )
        return config


__all__ = [
    "COMMON_DOSE_RULE",
    "ENVELOPE_CENTER_RULE",
    "ENVELOPE_SCALAR_COUNT",
    "ONE_SIDED_POWER_RULE",
    "PREPARATION_ONLY",
    "REFERENCE_SOURCE_PINS",
    "SCHEMA",
    "SHAPE_SCALAR_COUNT",
    "SharedRecipientLocalPowerSpectralHeadV1",
    "TRAINABLE_SCALAR_COUNT",
]
