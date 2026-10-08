"""Direct D0 and unit-direction Fit40 objective for the V3 transition head.

Preparation-only source. It delegates only native rendering/scoring to the
existing V3/V4 bridge; it creates no model, optimizer, or data access.
"""

from __future__ import annotations

from typing import Any, Mapping

from scripts.audio_enhancement import v3_v4_native_runtime_adapter_attempt03 as _native


SCHEMA = "cross_tfg_v3_v4_native_runtime_adapter_attempt04_direct_d0_direction"
PREPARATION_ONLY = True
DB_TO_NATURAL_LOG_GAIN = _native.DB_TO_NATURAL_LOG_GAIN
D0_DENOMINATOR_EPSILON = 1e-8
COSINE_DISTANCE_DENOMINATOR_FLOOR = 0.05
CBEST_RELATIVE_DENOMINATOR_FLOOR = 0.05

# Frozen before any new Fit40 execution. The Cbest safeguard coefficient is
# 1.0; the normalized worsening hinge is squared.
OBJECTIVE_WEIGHTS = {
    "relative_D0_improvement": 1.0,
    "relative_zero_lag_cosine_improvement": 1.0,
    "Cbest_nonworsening_guard": 1.0,
    "waveform_relative_mse_to_rms_reference": 0.02,
    "mask_total_variation": 0.002,
}


def score_with_v4_native_path(
    *,
    v4_runtime: Any,
    v4_adapter: Any,
    context: Any,
    geometry_record: Mapping[str, Any],
    enhanced_audio: Any,
) -> dict[str, Any]:
    """Use the unchanged, source-pinned V4 native rendering/scoring path."""

    return _native.score_with_v4_native_path(
        v4_runtime=v4_runtime,
        v4_adapter=v4_adapter,
        context=context,
        geometry_record=geometry_record,
        enhanced_audio=enhanced_audio,
    )


def v4_objective_for_v3_candidate(
    *,
    torch: Any,
    functional: Any,
    v4_runtime: Any,
    v4_adapter: Any,
    context: Any,
    geometry_record: Mapping[str, Any],
    v3_result: Mapping[str, Any],
    rms_curve: Any,
    rms_visual_embeddings: Any,
    objective_weights: Mapping[str, float],
) -> dict[str, Any]:
    """Reward relative zero-lag D0/cosine gains with a Cbest guard.

    The dynamic gate is the training objective. GateDC is a paired, post-fit
    control and is intentionally not optimized by this adapter.
    """

    if set(objective_weights) != set(OBJECTIVE_WEIGHTS):
        raise ValueError("objective weight keys differ from the frozen attempt04 contract")
    for name, expected in OBJECTIVE_WEIGHTS.items():
        if float(objective_weights[name]) != expected:
            raise ValueError(f"objective coefficient differs from frozen attempt04 value: {name}")

    candidate_audio = v3_result["output_pcm_float32"]
    score = score_with_v4_native_path(
        v4_runtime=v4_runtime,
        v4_adapter=v4_adapter,
        context=context,
        geometry_record=geometry_record,
        enhanced_audio=candidate_audio,
    )
    curve = score["curve_float32"]
    visual = score["visual_embeddings_float32"]
    audio_center = score["audio_center_embeddings_float32"].detach()
    reference_curve = rms_curve.detach().to(device=curve.device, dtype=torch.float32)
    reference_visual = rms_visual_embeddings.detach().to(device=visual.device, dtype=torch.float32)

    reference_metrics = v4_runtime._curve_metrics(torch, reference_curve)
    candidate_metrics = v4_runtime._curve_metrics(torch, curve)
    reference_norm, _, reference_cos = v4_runtime._embedding_metrics(
        torch, functional, reference_visual, audio_center
    )
    candidate_norm, _, candidate_cos = v4_runtime._embedding_metrics(
        torch, functional, visual, audio_center
    )

    d0_reference = reference_metrics["D0_zero_lag_distance"].detach()
    d0_candidate = candidate_metrics["D0_zero_lag_distance"]
    relative_d0_improvement = (d0_reference - d0_candidate) / d0_reference.clamp_min(
        D0_DENOMINATOR_EPSILON
    )

    cosine_reference = reference_cos.mean().detach()
    cosine_candidate = candidate_cos.mean()
    cosine_distance_scale = (1.0 - cosine_reference).clamp_min(
        COSINE_DISTANCE_DENOMINATOR_FLOOR
    )
    relative_cosine_improvement = (cosine_candidate - cosine_reference) / cosine_distance_scale

    cbest_reference = reference_metrics["Cbest_B_minus_Dmin"].detach()
    cbest_candidate = candidate_metrics["Cbest_B_minus_Dmin"]
    cbest_scale = cbest_reference.abs().clamp_min(CBEST_RELATIVE_DENOMINATOR_FLOOR)
    relative_cbest_worsening = torch.relu((cbest_reference - cbest_candidate) / cbest_scale)
    cbest_guard_penalty = (
        float(objective_weights["Cbest_nonworsening_guard"])
        * relative_cbest_worsening.square()
    )

    reward = (
        float(objective_weights["relative_D0_improvement"]) * relative_d0_improvement
        + float(objective_weights["relative_zero_lag_cosine_improvement"])
        * relative_cosine_improvement
    )
    baseline_audio = v3_result["canonical_rms_baseline_pcm_float32"].detach()
    if candidate_audio.shape != baseline_audio.shape:
        raise ValueError("V3 waveform and canonical RMS reference lengths differ")
    waveform_mse = torch.mean((candidate_audio - baseline_audio).square())
    waveform_energy = baseline_audio.square().mean().clamp_min(1e-8)
    relative_waveform_mse = waveform_mse / waveform_energy

    mask_db = v3_result["projected_log_magnitude_mask_db"]
    mask_tv_ln_gain = v4_runtime._mask_tv(torch, mask_db * DB_TO_NATURAL_LOG_GAIN)
    loss = (
        -reward
        + cbest_guard_penalty
        + float(objective_weights["waveform_relative_mse_to_rms_reference"])
        * relative_waveform_mse
        + float(objective_weights["mask_total_variation"]) * mask_tv_ln_gain
    )

    metrics = {
        "reward": reward,
        "loss": loss,
        "relative_D0_improvement": relative_d0_improvement,
        "relative_zero_lag_cosine_improvement": relative_cosine_improvement,
        "cosine_distance_denominator": cosine_distance_scale,
        "relative_Cbest_worsening": relative_cbest_worsening,
        "Cbest_nonworsening_guard_penalty": cbest_guard_penalty,
        "waveform_relative_mse_to_rms_reference": relative_waveform_mse,
        "mask_total_variation_ln_gain": mask_tv_ln_gain,
        "mask_total_variation_db_equivalent": mask_tv_ln_gain / DB_TO_NATURAL_LOG_GAIN,
        "B_RMS": reference_metrics["B_median_distance"],
        "Dmin_RMS": reference_metrics["Dmin_best_lag_distance"],
        "D0_RMS": d0_reference,
        "Cbest_RMS": cbest_reference,
        "Canchor_RMS": reference_metrics["Canchor_zero_B_minus_D0"],
        "far_RMS": reference_metrics["far_lag_contrast_abs4plus"],
        "B_candidate": candidate_metrics["B_median_distance"],
        "Dmin_candidate": candidate_metrics["Dmin_best_lag_distance"],
        "D0_candidate": d0_candidate,
        "Cbest_candidate": cbest_candidate,
        "Canchor_candidate": candidate_metrics["Canchor_zero_B_minus_D0"],
        "far_candidate": candidate_metrics["far_lag_contrast_abs4plus"],
        "mean_cosine_RMS": cosine_reference,
        "mean_cosine_candidate": cosine_candidate,
        "mean_vnorm_RMS": reference_norm.mean(),
        "mean_vnorm_candidate": candidate_norm.mean(),
        "visual_embedding_log_norm_ratio_squared": torch.mean(
            torch.log(candidate_norm / reference_norm.detach()).square()
        ),
    }
    for name, value in metrics.items():
        if not torch.isfinite(value).all().item():
            raise FloatingPointError(f"non-finite direct-D0 objective diagnostic: {name}")
    return {"score": score, "metrics": metrics, "loss": loss}


__all__ = [
    "SCHEMA",
    "PREPARATION_ONLY",
    "OBJECTIVE_WEIGHTS",
    "D0_DENOMINATOR_EPSILON",
    "COSINE_DISTANCE_DENOMINATOR_FLOOR",
    "CBEST_RELATIVE_DENOMINATOR_FLOOR",
    "score_with_v4_native_path",
    "v4_objective_for_v3_candidate",
]
