"""V3-to-V4 bridge with V4's frozen natural-log mask TV units.

This is a source-only engineering adapter. It loads no weights, creates no
optimizer, and preserves the pinned native renderer, 31-lag reduction,
objective coefficients, and source/query clock.
"""

from __future__ import annotations

from typing import Any, Mapping


SCHEMA = "cross_tfg_v3_v4_native_runtime_adapter_attempt03"
PREPARATION_ONLY = True
DB_TO_NATURAL_LOG_GAIN = 0.11512925464970229  # ln(10) / 20; pinned heads.py convention


def score_with_v4_native_path(
    *,
    v4_runtime: Any,
    v4_adapter: Any,
    context: Any,
    geometry_record: Mapping[str, Any],
    enhanced_audio: Any,
) -> dict[str, Any]:
    rendered_crops = v4_runtime._render_native_video(
        v4_adapter, context, enhanced_audio, geometry_record
    )
    curve = v4_adapter.sync_distance_curve(rendered_crops, context=context)
    visual = v4_adapter.last_visual_embeddings
    audio_center = v4_adapter.last_audio_center_embeddings
    if visual is None or audio_center is None:
        raise RuntimeError("V4 native scorer did not expose its 27 query embeddings")
    if tuple(curve.shape) != (31,) or tuple(visual.shape) != (27, 1024):
        raise RuntimeError("V4 native scorer returned an unexpected curve or query shape")
    return {
        "curve_float32": curve,
        "visual_embeddings_float32": visual,
        "audio_center_embeddings_float32": audio_center,
        "rendered_crop_shape": tuple(int(value) for value in rendered_crops.shape),
    }


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

    delta_cbest = (
        candidate_metrics["Cbest_B_minus_Dmin"]
        - reference_metrics["Cbest_B_minus_Dmin"].detach()
    )
    delta_far = (
        candidate_metrics["far_lag_contrast_abs4plus"]
        - reference_metrics["far_lag_contrast_abs4plus"].detach()
    )
    d0_violation = torch.relu(
        candidate_metrics["D0_zero_lag_distance"]
        - reference_metrics["D0_zero_lag_distance"].detach()
    )
    canchor_violation = torch.relu(
        reference_metrics["Canchor_zero_B_minus_D0"].detach()
        - candidate_metrics["Canchor_zero_B_minus_D0"]
    )
    d0_penalty = float(objective_weights["D0_nonworsening_guard"]) * d0_violation.square()
    canchor_penalty = (
        float(objective_weights["Canchor_nonworsening_guard"])
        * canchor_violation.square()
    )
    reward = (
        float(objective_weights["relative_Cbest_improvement"]) * delta_cbest
        + float(objective_weights["far_lag_contrast_improvement"]) * delta_far
        - d0_penalty
        - canchor_penalty
    )
    norm_penalty = torch.mean(torch.log(candidate_norm / reference_norm.detach()).square())

    baseline_audio = v3_result["canonical_rms_baseline_pcm_float32"].detach()
    if candidate_audio.shape != baseline_audio.shape:
        raise ValueError("V3 waveform and V4 canonical RMS reference lengths differ")
    waveform_mse = torch.mean((candidate_audio - baseline_audio).square())
    waveform_energy = baseline_audio.square().mean().clamp_min(1e-8)
    relative_waveform_mse = waveform_mse / waveform_energy

    mask_db = v3_result["projected_log_magnitude_mask_db"]
    mask_ln_gain = mask_db * DB_TO_NATURAL_LOG_GAIN
    mask_tv_ln_gain = v4_runtime._mask_tv(torch, mask_ln_gain)
    loss = (
        -reward
        + float(objective_weights["visual_embedding_log_norm_ratio_squared"]) * norm_penalty
        + float(objective_weights["waveform_relative_mse_to_rms_reference"]) * relative_waveform_mse
        + float(objective_weights["mask_total_variation"]) * mask_tv_ln_gain
    )
    metrics = {
        "reward": reward,
        "loss": loss,
        "delta_Cbest": delta_cbest,
        "delta_far_lag_contrast_abs4plus": delta_far,
        "D0_guard_violation": d0_violation,
        "Canchor_guard_violation": canchor_violation,
        "D0_guard_penalty": d0_penalty,
        "Canchor_guard_penalty": canchor_penalty,
        "visual_embedding_log_norm_ratio_squared": norm_penalty,
        "waveform_relative_mse_to_rms_reference": relative_waveform_mse,
        "mask_total_variation_ln_gain": mask_tv_ln_gain,
        "mask_total_variation_db_equivalent": mask_tv_ln_gain / DB_TO_NATURAL_LOG_GAIN,
        "B_RMS": reference_metrics["B_median_distance"],
        "Dmin_RMS": reference_metrics["Dmin_best_lag_distance"],
        "D0_RMS": reference_metrics["D0_zero_lag_distance"],
        "Cbest_RMS": reference_metrics["Cbest_B_minus_Dmin"],
        "Canchor_RMS": reference_metrics["Canchor_zero_B_minus_D0"],
        "far_RMS": reference_metrics["far_lag_contrast_abs4plus"],
        "B_candidate": candidate_metrics["B_median_distance"],
        "Dmin_candidate": candidate_metrics["Dmin_best_lag_distance"],
        "D0_candidate": candidate_metrics["D0_zero_lag_distance"],
        "Cbest_candidate": candidate_metrics["Cbest_B_minus_Dmin"],
        "Canchor_candidate": candidate_metrics["Canchor_zero_B_minus_D0"],
        "far_candidate": candidate_metrics["far_lag_contrast_abs4plus"],
        "mean_cosine_RMS": reference_cos.mean(),
        "mean_cosine_candidate": candidate_cos.mean(),
        "mean_vnorm_RMS": reference_norm.mean(),
        "mean_vnorm_candidate": candidate_norm.mean(),
    }
    for name, value in metrics.items():
        if not torch.isfinite(value).all().item():
            raise FloatingPointError(f"non-finite V4 objective diagnostic: {name}")
    return {"score": score, "metrics": metrics, "loss": loss}


__all__ = [
    "SCHEMA",
    "DB_TO_NATURAL_LOG_GAIN",
    "score_with_v4_native_path",
    "v4_objective_for_v3_candidate",
]
