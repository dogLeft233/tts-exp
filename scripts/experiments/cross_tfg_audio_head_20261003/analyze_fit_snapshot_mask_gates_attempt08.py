"""Produce source-pinned, score-free V3 mask and gate diagnostics for one Fit point.

This utility validates a real frozen SPSA point and its existing shared
application receipts, then runs the exact pinned snapshot head once on the
point's original PCM on CPU. It records source-clock gates, four-band mask and
linear-gain means, flux covariance, temporal DC/centered decompositions, and
the existing single global projection. It does not score, render video,
optimize, select, or write transformed audio.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

HELPER05_REL = Path("scripts/audio_enhancement/fit40_snapshot_spsa_application_attempt05.py")
HELPER05_BYTES = 56593
HELPER05_SHA256 = "3a33cf6f148de36d0433074c2efadb98167880ed5b7f1f66138a797a047afc5b"
ANALYZER_SCHEMA = "cross_tfg_fit_snapshot_mask_gate_diagnostic_v2"
REQUEST_SCHEMA = "cross_tfg_fit_snapshot_mask_gate_analysis_request_v2"
CENTER_STEPS = (5, 10, 15, 20)
MODEL_ORDER = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
CAPTURE_KEYS = (
    "raw_log_magnitude_mask_db",
    "smoothed_log_magnitude_mask_db",
    "frequency_common_mode_before_policy_db",
    "policy_mask_db_before_smooth_bound",
    "projected_log_magnitude_mask_db",
    "band_basis",
    "reference_coded_band_weights_db",
    "positive_relative_flux",
    "negative_relative_flux",
    "active_frame_mask",
    "valid_transition_mask",
    "positive_flux_gate",
    "negative_flux_gate",
    "positive_flux_threshold",
    "negative_flux_threshold",
    "positive_flux_valid_transition_count",
    "negative_flux_valid_transition_count",
    "positive_flux_has_valid_support",
    "negative_flux_has_valid_support",
    "smooth_projection_scale",
    "preprojection_max_abs_mask_db",
    "postprojection_max_abs_mask_db",
    "stft_frame_centers_source_samples",
)


class DiagnosticError(RuntimeError):
    """An input is not the exact pinned Fit point or forward contract."""


def _validate_center_fit_roster(family: Mapping[str, Any], point_plan: Mapping[str, Any],
                                method_id: str, method_config_sha: str,
                                binding_sha: str) -> Tuple[str, ...]:
    """Require the center point's real three-model rows to cover frozen Fit40."""
    family_models = [item.get("model_id") for item in family.get("models", [])
                     if isinstance(item, Mapping)]
    point_models = [item.get("model_id") for item in point_plan.get("models", [])
                    if isinstance(item, Mapping)]
    if family_models != list(MODEL_ORDER) or point_models != list(MODEL_ORDER):
        raise DiagnosticError("family/center model order differs from the frozen three-model roster")

    family_identity: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    family_samples: Dict[str, set] = {}
    for model_id in MODEL_ORDER:
        model_rows = [row for row in family.get("rows", [])
                      if isinstance(row, Mapping) and row.get("model_id") == model_id
                      and row.get("method_id") == "identity" and row.get("split") == "fit"]
        ids = [row.get("sample_id") for row in model_rows]
        if len(model_rows) != 40 or len(set(ids)) != 40 or not all(isinstance(x, str) for x in ids):
            raise DiagnosticError("frozen family does not contain exactly 40 unique Fit identity rows for %s" % model_id)
        family_samples[model_id] = set(ids)
        for row in model_rows:
            family_identity[(model_id, str(row["sample_id"]))] = row
    expected = family_samples[MODEL_ORDER[0]]
    if any(family_samples[model_id] != expected for model_id in MODEL_ORDER[1:]):
        raise DiagnosticError("frozen family models do not share one exact 40-clip Fit roster")

    center_rows = [row for row in point_plan.get("rows", [])
                   if isinstance(row, Mapping) and row.get("method_id") == method_id]
    if len(center_rows) != 120 or len({row.get("row_id") for row in center_rows}) != 120:
        raise DiagnosticError("center point must contain exactly 120 unique rows across three models")
    seen_by_model: Dict[str, set] = {}
    for model_id in MODEL_ORDER:
        model_rows = [row for row in center_rows if row.get("model_id") == model_id]
        ids = [row.get("sample_id") for row in model_rows]
        if len(model_rows) != 40 or len(set(ids)) != 40 or set(ids) != expected:
            raise DiagnosticError("center point rows do not cover the actual 40-clip roster for %s" % model_id)
        seen_by_model[model_id] = set(ids)
        for row in model_rows:
            if (row.get("method_config_sha256") != method_config_sha
                    or row.get("spsa_point_binding_sha256") != binding_sha):
                raise DiagnosticError("center row does not bind the exact serialized method/config")
            parent = family_identity.get((model_id, str(row.get("sample_id"))))
            if parent is None:
                raise DiagnosticError("center row has no corresponding frozen Fit identity parent")
            for key in ("speaker_id", "seed", "split", "natural_pcm", "score_audio",
                        "source_face", "source_video", "source_media_kind"):
                if row.get(key) != parent.get(key):
                    raise DiagnosticError("center row changed frozen identity/source field %s" % key)
            if row.get("score_audio") != row.get("natural_pcm"):
                raise DiagnosticError("center row score audio is not the exact natural PCM reference")
    if any(seen_by_model[model_id] != expected for model_id in MODEL_ORDER):
        raise DiagnosticError("center models do not share the actual 40-clip Fit roster")
    return tuple(row["sample_id"] for row in family.get("rows", [])
                 if isinstance(row, Mapping) and row.get("model_id") == MODEL_ORDER[0]
                 and row.get("method_id") == "identity" and row.get("split") == "fit")


def _validate_final_center_selection(request: Mapping[str, Any], family: Mapping[str, Any],
                                     family_pin: Mapping[str, Any], method: Mapping[str, Any],
                                     config: Mapping[str, Any], theta: list,
                                     method_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """Bind diagnostics to the actual final selector result, including fallback status."""
    receipt, _, pin = _read_pinned_json(
        request.get("final_selector_receipt"), "final-center selector receipt", "receipt_sha256"
    )
    final = receipt.get("final_selection")
    if (receipt.get("schema") != "fit40_final_center_selector_receipt_v1"
            or receipt.get("status") != "COMPLETE_FIT_ONLY_FINAL_CENTER_SELECTION_NOT_FINAL_DEV"
            or receipt.get("metric_backend") != "torch_cpu"
            or receipt.get("final_dev_or_confirm_admission_performed") is not False
            or not isinstance(final, Mapping)
            or receipt.get("selection_status") != final.get("status")
            or final.get("metric_backend") != "torch_cpu"):
        raise DiagnosticError("final-center selector receipt is not a complete canonical torch_cpu result")
    if (receipt.get("composed_family_plan_sha256") != family_pin.get("sha256")
            or receipt.get("composed_family_plan_self_sha256") != family.get("plan_sha256")
            or receipt.get("scientific_input_pins_sha256") != family.get("scientific_input_pins_sha256")):
        raise DiagnosticError("final selector receipt belongs to another frozen family/input binding")
    specs = receipt.get("center_specs")
    if not isinstance(specs, list) or len(specs) != len(CENTER_STEPS):
        raise DiagnosticError("final selector receipt does not contain all four center specs")
    by_step = {item.get("source_step"): item for item in specs
               if isinstance(item, Mapping) and type(item.get("source_step")) is int}
    if set(by_step) != set(CENTER_STEPS) or len(by_step) != len(CENTER_STEPS):
        raise DiagnosticError("final selector center specs are not exactly k=5/10/15/20")

    status = final.get("status")
    if status == "ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV":
        selected_method_id = final.get("selected_method_id")
        if not isinstance(selected_method_id, str) or final.get("refinement_seed_method_id") is not None:
            raise DiagnosticError("admitted selector result lacks its unique selected method")
        classification = "ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV_MECHANISM_DIAGNOSTIC_ONLY"
    elif status == "NOT_ADMITTED":
        selected_method_id = final.get("refinement_seed_method_id")
        if not isinstance(selected_method_id, str) or final.get("selected_method_id") is not None:
            raise DiagnosticError("NOT_ADMITTED result lacks its unique provisional refinement seed")
        classification = "PROVISIONAL_REFINEMENT_SEED_MECHANISM_DIAGNOSTIC_ONLY"
    else:
        raise DiagnosticError("final selector status is not an admissible complete Fit outcome")
    if selected_method_id != method_id:
        raise DiagnosticError("requested center is not the receipt-selected center/refinement seed")

    source_step = config.get("source_step")
    spec = by_step.get(source_step)
    if (method.get("kind") != "fit40_spsa_center"
            or type(source_step) is not int
            or not isinstance(spec, Mapping) or spec.get("point_kind") != "spsa_center"
            or spec.get("method_id") != method_id or spec.get("theta") != theta
            or spec.get("artifact_sha256") != config.get("artifact_sha256")
            or spec.get("binding_sha256") != config.get("binding_sha256")):
        raise DiagnosticError("requested point config differs from the selected full-center spec")
    return receipt, pin, classification


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    return _sha256(_canonical_bytes({key: item for key, item in value.items() if key != field}))


def _file_pin(path: Path) -> Dict[str, Any]:
    if path.is_symlink():
        raise DiagnosticError("symlink input is not allowed: %s" % path)
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise DiagnosticError("input is not a regular file: %s" % path)
    raw = resolved.read_bytes()
    return {"path": str(resolved), "bytes": len(raw), "sha256": _sha256(raw)}


def _read_pinned_json(pin: Mapping[str, Any], label: str,
                      self_field: Optional[str] = None) -> Tuple[Dict[str, Any], bytes, Dict[str, Any]]:
    if not isinstance(pin, Mapping):
        raise DiagnosticError("%s pin is missing" % label)
    path_text = pin.get("path")
    if not isinstance(path_text, str) or not Path(path_text).is_absolute():
        raise DiagnosticError("%s path must be absolute" % label)
    path = Path(path_text)
    actual = _file_pin(path)
    raw = path.read_bytes()
    if actual["bytes"] != pin.get("bytes") or actual["sha256"] != pin.get("sha256"):
        raise DiagnosticError("%s raw bytes differ from the supplied pin" % label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DiagnosticError("%s is not UTF-8 JSON" % label) from exc
    if not isinstance(value, dict):
        raise DiagnosticError("%s root is not a JSON object" % label)
    if self_field is not None:
        expected = value.get(self_field)
        if (not isinstance(expected, str) or _self_hash(value, self_field) != expected
                or pin.get("self_sha256") != expected):
            raise DiagnosticError("%s canonical self hash differs" % label)
    return value, raw, actual


def _load_pinned_module(name: str, path: Path, expected: Mapping[str, Any]) -> Any:
    actual = _file_pin(path)
    if (actual["bytes"], actual["sha256"]) != (expected.get("bytes"), expected.get("sha256")):
        raise DiagnosticError("%s source differs from its exact pin" % name)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DiagnosticError("cannot load pinned module %s" % name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    loaded = Path(str(module.__file__)).resolve(strict=True)
    if loaded != path.resolve(strict=True):
        raise DiagnosticError("%s was imported from an unexpected path" % name)
    return module


def _as_numpy(value: Any, torch: Any) -> Any:
    import numpy as np

    if isinstance(value, torch.Tensor):
        return np.ascontiguousarray(value.detach().cpu().numpy())
    return np.asarray(value)


def _capture_forward(transform: Any, torch: Any, samples: Any) -> Tuple[Any, Dict[str, Any]]:
    captured: Dict[str, Any] = {}
    call_count = [0]

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        if not isinstance(output, dict):
            raise DiagnosticError("pinned V3 head did not return its diagnostics dictionary")
        call_count[0] += 1
        captured.clear()
        for key in CAPTURE_KEYS:
            if key not in output:
                raise DiagnosticError("pinned head output omitted %s" % key)
            value = output[key]
            if isinstance(value, torch.Tensor):
                captured[key] = value.detach().cpu().numpy().copy()
            elif type(value) in (bool, int, float):
                captured[key] = value
            else:
                raise DiagnosticError("unexpected diagnostic value type for %s" % key)

    hook = transform.head.register_forward_hook(capture)
    try:
        output_samples, transform_info = transform.transform(samples)
    finally:
        hook.remove()
    if call_count[0] != 1:
        raise DiagnosticError("expected exactly one pinned head forward, observed %d" % call_count[0])
    return output_samples, captured


def _component_decomposition(head: Any, head_module: Any,
                             output: Mapping[str, Any], torch: Any) -> Dict[str, Any]:
    """Reconstruct linear components and apply the observed single global scale."""
    import numpy as np

    basis = head.band_basis.detach().to(device="cpu", dtype=torch.float32)
    weights = torch.as_tensor(output["reference_coded_band_weights_db"], dtype=torch.float32)
    positive_gate = torch.as_tensor(output["positive_flux_gate"], dtype=torch.float32)
    negative_gate = torch.as_tensor(output["negative_flux_gate"], dtype=torch.float32)
    if tuple(basis.shape) != (4, head.n_fft // 2 + 1) or tuple(weights.shape) != (3, 4):
        raise DiagnosticError("four-band basis or [3,4] reference-coded weights changed shape")
    time_count = int(positive_gate.shape[0])
    if int(negative_gate.shape[0]) != time_count or time_count < 1:
        raise DiagnosticError("positive and negative gate clock lengths differ")
    basis_t = basis.transpose(0, 1)
    static = (basis_t @ weights[0].reshape(4, 1)).expand(-1, time_count)
    positive = (basis_t @ weights[1].reshape(4, 1)) * positive_gate.reshape(1, -1)
    negative = (basis_t @ weights[2].reshape(4, 1)) * negative_gate.reshape(1, -1)
    raw_components = torch.stack((static, positive, negative), dim=0)
    raw_sum = raw_components.sum(dim=0)
    raw_actual = torch.as_tensor(output["raw_log_magnitude_mask_db"], dtype=torch.float32)
    if not torch.allclose(raw_sum, raw_actual, rtol=1e-6, atol=1e-6):
        raise DiagnosticError("reference-coded component sum does not reconstruct the actual raw mask")

    smoothed = []
    for component in raw_components:
        along_frequency = head_module._smooth_mask_axis(
            component, head.mask_smoothing_frequency_bins, "frequency", head.mask_smoothing_pad_mode
        )
        along_time = head_module._smooth_mask_axis(
            along_frequency, head.mask_smoothing_time_frames, "time", head.mask_smoothing_pad_mode
        )
        if head.frequency_common_mode_policy == "remove_per_frame":
            along_time = along_time - along_time.mean(dim=0, keepdim=True)
        smoothed.append(along_time)
    policy_components = torch.stack(smoothed, dim=0)
    policy_sum = policy_components.sum(dim=0)
    policy_actual = torch.as_tensor(output["policy_mask_db_before_smooth_bound"], dtype=torch.float32)
    if not torch.allclose(policy_sum, policy_actual, rtol=1e-5, atol=2e-6):
        raise DiagnosticError("linear component decomposition does not reconstruct the actual preprojection mask")

    scale = float(output["smooth_projection_scale"])
    projected_components = policy_components * scale
    projected_sum = projected_components.sum(dim=0)
    projected_actual = torch.as_tensor(output["projected_log_magnitude_mask_db"], dtype=torch.float32)
    if not torch.allclose(projected_sum, projected_actual, rtol=1e-5, atol=2e-6):
        raise DiagnosticError("single observed global scale does not reconstruct the actual projected mask")

    bin_counts = basis.sum(dim=1)
    band_weights = basis / bin_counts.reshape(4, 1)

    def per_band(mask: Any) -> Any:
        return band_weights @ mask

    raw_band_per_frame = per_band(raw_actual)
    smoothed_actual = torch.as_tensor(output["smoothed_log_magnitude_mask_db"], dtype=torch.float32)
    common_mode = torch.as_tensor(output["frequency_common_mode_before_policy_db"], dtype=torch.float32)
    if smoothed_actual.shape != raw_actual.shape or common_mode.shape != (1, time_count):
        raise DiagnosticError("smoothed/common-mode masks do not match the [frequency, frame] clock")
    expected_policy = (smoothed_actual - common_mode
                       if head.frequency_common_mode_policy == "remove_per_frame"
                       else smoothed_actual)
    if not torch.allclose(expected_policy, policy_actual, rtol=1e-6, atol=1e-6):
        raise DiagnosticError("preprojection policy mask does not match the declared common-mode rule")
    smoothed_band_per_frame = per_band(smoothed_actual)
    common_mode_band_per_frame = per_band(common_mode.expand_as(smoothed_actual))
    pre_band_per_frame = per_band(policy_actual)
    post_band_per_frame = per_band(projected_actual)
    raw_gain = torch.pow(torch.tensor(10.0, dtype=torch.float32), raw_actual / 20.0)
    smoothed_gain = torch.pow(torch.tensor(10.0, dtype=torch.float32), smoothed_actual / 20.0)
    pre_gain = torch.pow(torch.tensor(10.0, dtype=torch.float32), policy_actual / 20.0)
    post_gain = torch.pow(torch.tensor(10.0, dtype=torch.float32), projected_actual / 20.0)
    raw_band_gain_by_frame = per_band(raw_gain)
    smoothed_band_gain_by_frame = per_band(smoothed_gain)
    pre_band_gain_by_frame = per_band(pre_gain)
    post_band_gain_by_frame = per_band(post_gain)
    component_post_band_per_frame = torch.stack(
        [per_band(projected_components[index]) for index in range(3)], dim=0
    )

    def time_dc_and_centered(values: Any) -> Tuple[Any, Any, float]:
        dc = values.mean(dim=-1, keepdim=True)
        centered = values - dc
        reconstruction_error = float((dc + centered - values).abs().amax().item())
        return dc, centered, reconstruction_error

    pre_dc, pre_centered, pre_dc_error = time_dc_and_centered(policy_actual)
    post_dc, post_centered, post_dc_error = time_dc_and_centered(projected_actual)
    pre_dc_reconstruction = pre_dc.expand_as(policy_actual)
    post_dc_reconstruction = post_dc.expand_as(projected_actual)

    def band_temporal_summary(mask_curve: Any, gain_curve: Any,
                              valid_mask: Any, pos_flux: Any,
                              neg_flux: Any) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"per_band": []}
        valid_np = np.asarray(valid_mask, dtype=np.bool_)
        pos_np = np.asarray(pos_flux, dtype=np.float64)
        neg_np = np.asarray(neg_flux, dtype=np.float64)

        def pair_stats(x: Any, y: Any) -> Dict[str, Any]:
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            if x.shape != y.shape or x.ndim != 1 or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise DiagnosticError("flux relation arrays must be finite, aligned vectors")
            if x.size == 0:
                return {"n": 0, "covariance_population": None, "correlation": None}
            xc = x - x.mean(dtype=np.float64)
            yc = y - y.mean(dtype=np.float64)
            covariance = float(np.mean(xc * yc, dtype=np.float64))
            denom = math.sqrt(float(np.mean(xc * xc, dtype=np.float64)
                                    * np.mean(yc * yc, dtype=np.float64)))
            return {"n": int(x.size), "covariance_population": covariance,
                    "correlation": covariance / denom if denom > 0.0 else None}

        mask_np = np.asarray(mask_curve, dtype=np.float64)
        gain_np = np.asarray(gain_curve, dtype=np.float64)
        for band_index in range(mask_np.shape[0]):
            mask_series = mask_np[band_index]
            gain_series = gain_np[band_index]
            if valid_np.shape != mask_series.shape or pos_np.shape != mask_series.shape or neg_np.shape != mask_series.shape:
                raise DiagnosticError("band, flux, and source-clock frame counts differ")
            mask_valid = mask_series[valid_np]
            gain_valid = gain_series[valid_np]
            pos_valid = pos_np[valid_np]
            neg_valid = neg_np[valid_np]
            summary["per_band"].append({
                "band_index": band_index,
                "mask_db_mean_all_frames": float(mask_series.mean(dtype=np.float64)),
                "mask_db_mean_valid_transitions": float(mask_valid.mean(dtype=np.float64)) if mask_valid.size else None,
                "mask_db_variance_valid_transitions": float(mask_valid.var(dtype=np.float64)) if mask_valid.size else None,
                "linear_amplitude_gain_mean_all_frames": float(gain_series.mean(dtype=np.float64)),
                "linear_amplitude_gain_temporal_dc_all_frames": float(gain_series.mean(dtype=np.float64)),
                "linear_amplitude_gain_centered_temporal_variance_all_frames": float(
                    np.mean((gain_series - gain_series.mean(dtype=np.float64)) ** 2, dtype=np.float64)),
                "linear_amplitude_gain_mean_valid_transitions": float(gain_valid.mean(dtype=np.float64)) if gain_valid.size else None,
                "linear_amplitude_gain_centered_temporal_variance_valid_transitions": (
                    float(np.mean((gain_valid - gain_valid.mean(dtype=np.float64)) ** 2, dtype=np.float64))
                    if gain_valid.size else None),
                "valid_transition_count": int(valid_np.sum()),
                "mask_db_vs_positive_relative_flux": pair_stats(mask_valid, pos_valid),
                "mask_db_vs_negative_relative_flux": pair_stats(mask_valid, neg_valid),
                "gain_vs_positive_relative_flux": pair_stats(gain_valid, pos_valid),
                "gain_vs_negative_relative_flux": pair_stats(gain_valid, neg_valid),
            })
        return summary

    valid_mask = output["valid_transition_mask"]
    pos_flux = output["positive_relative_flux"]
    neg_flux = output["negative_relative_flux"]
    band_summaries = {
        "raw_before_smoothing": band_temporal_summary(
            raw_band_per_frame.numpy(), raw_band_gain_by_frame.numpy(), valid_mask, pos_flux, neg_flux),
        "smoothed_before_common_mode": band_temporal_summary(
            smoothed_band_per_frame.numpy(), smoothed_band_gain_by_frame.numpy(), valid_mask, pos_flux, neg_flux),
        "common_mode_component": band_temporal_summary(
            common_mode_band_per_frame.numpy(),
            per_band(torch.pow(torch.tensor(10.0, dtype=torch.float32),
                               common_mode.expand_as(smoothed_actual) / 20.0)).numpy(),
            valid_mask, pos_flux, neg_flux),
        "preprojection_after_common_mode_policy": band_temporal_summary(
            pre_band_per_frame.numpy(), pre_band_gain_by_frame.numpy(), valid_mask, pos_flux, neg_flux),
        "postprojection": band_temporal_summary(
            post_band_per_frame.numpy(), post_band_gain_by_frame.numpy(), valid_mask, pos_flux, neg_flux),
    }
    # Positive/negative flux are polarity proxies. These comparisons expose
    # asymmetry in the actual gates and component contribution; they do not
    # infer phonetic onset/offset events beyond the head's source-clock flux.
    component_polarity: Dict[str, Any] = {}
    for band_index in range(4):
        positive_series = component_post_band_per_frame[1, band_index].numpy().astype(np.float64)
        negative_series = component_post_band_per_frame[2, band_index].numpy().astype(np.float64)
        valid_np = np.asarray(valid_mask, dtype=np.bool_)
        p = positive_series[valid_np]
        n = negative_series[valid_np]
        component_polarity[str(band_index)] = {
            "positive_component_mean_db_valid": float(p.mean(dtype=np.float64)) if p.size else None,
            "negative_component_mean_db_valid": float(n.mean(dtype=np.float64)) if n.size else None,
            "positive_minus_negative_mean_db_valid": float(p.mean(dtype=np.float64) - n.mean(dtype=np.float64)) if p.size else None,
            "positive_component_abs_sum_db_valid": float(np.abs(p).sum(dtype=np.float64)),
            "negative_component_abs_sum_db_valid": float(np.abs(n).sum(dtype=np.float64)),
            "valid_transition_count": int(valid_np.sum()),
        }
    arrays = {
        "band_basis_4_by_frequency": basis.numpy(),
        "band_bin_counts": bin_counts.to(dtype=torch.int64).numpy(),
        "reference_coded_band_weights_db_3_by_4": weights.numpy(),
        "raw_mask_components_db_3_by_frequency_by_frame": raw_components.numpy(),
        "preprojection_policy_mask_components_db_3_by_frequency_by_frame": policy_components.numpy(),
        "projected_mask_components_db_3_by_frequency_by_frame": projected_components.numpy(),
        "raw_log_magnitude_mask_db": raw_actual.numpy(),
        "smoothed_log_magnitude_mask_db": output["smoothed_log_magnitude_mask_db"],
        "frequency_common_mode_before_policy_db": output["frequency_common_mode_before_policy_db"],
        "preprojection_policy_dc_by_frequency_db": pre_dc.numpy(),
        "preprojection_policy_centered_by_frequency_db": pre_centered.numpy(),
        "preprojection_policy_dc_reconstruction_db": pre_dc_reconstruction.numpy(),
        "preprojection_policy_dc_plus_centered_db": (pre_dc_reconstruction + pre_centered).numpy(),
        "postprojection_dc_by_frequency_db": post_dc.numpy(),
        "postprojection_centered_by_frequency_db": post_centered.numpy(),
        "postprojection_dc_reconstruction_db": post_dc_reconstruction.numpy(),
        "postprojection_dc_plus_centered_db": (post_dc_reconstruction + post_centered).numpy(),
        "policy_mask_db_before_projection": policy_actual.numpy(),
        "projected_log_magnitude_mask_db": projected_actual.numpy(),
        "four_band_preprojection_mean_db_by_frame": pre_band_per_frame.numpy(),
        "four_band_postprojection_mean_db_by_frame": post_band_per_frame.numpy(),
        "four_band_projected_component_mean_db_by_frame": component_post_band_per_frame.numpy(),
        "four_band_raw_mean_gain_by_frame": raw_band_gain_by_frame.numpy(),
        "four_band_smoothed_mean_gain_by_frame": smoothed_band_gain_by_frame.numpy(),
        "four_band_preprojection_mean_gain_by_frame": pre_band_gain_by_frame.numpy(),
        "four_band_postprojection_mean_gain_by_frame": post_band_gain_by_frame.numpy(),
        "four_band_preprojection_mean_gain_centered_by_frame": (
            pre_band_gain_by_frame - pre_band_gain_by_frame.mean(dim=1, keepdim=True)).numpy(),
        "four_band_postprojection_mean_gain_centered_by_frame": (
            post_band_gain_by_frame - post_band_gain_by_frame.mean(dim=1, keepdim=True)).numpy(),
        "four_band_positive_component_mean_db_by_frame": component_post_band_per_frame[1].numpy(),
        "four_band_negative_component_mean_db_by_frame": component_post_band_per_frame[2].numpy(),
        "four_band_preprojection_mean_db": pre_band_per_frame.mean(dim=1).numpy(),
        "four_band_postprojection_mean_db": post_band_per_frame.mean(dim=1).numpy(),
    }
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise DiagnosticError("nonfinite diagnostic array: %s" % key)

    actual_premax = float(torch.as_tensor(output["preprojection_max_abs_mask_db"]).item())
    max_limit = float(head.max_abs_mask_db)
    extent = actual_premax / max_limit
    expected_scale = math.tanh(extent) / extent if extent > 0.0 else 1.0
    observed_postmax = float(torch.as_tensor(output["postprojection_max_abs_mask_db"]).item())
    recomputed_postmax = float(projected_actual.abs().amax().item())
    return {
        "arrays": arrays,
        "checks": {
            "raw_component_sum_max_abs_error": float((raw_sum - raw_actual).abs().amax().item()),
            "preprojection_component_sum_max_abs_error": float((policy_sum - policy_actual).abs().amax().item()),
            "projected_component_sum_max_abs_error": float((projected_sum - projected_actual).abs().amax().item()),
            "preprojection_dc_plus_centered_max_abs_error": pre_dc_error,
            "postprojection_dc_plus_centered_max_abs_error": post_dc_error,
            "global_projection_scale": scale,
            "global_projection_scale_formula": "tanh(preprojection_max_abs_db / max_abs_mask_db) / (preprojection_max_abs_db / max_abs_mask_db); 1 at zero extent",
            "global_projection_scale_formula_abs_error": abs(scale - expected_scale),
            "preprojection_max_abs_mask_db": actual_premax,
            "postprojection_max_abs_mask_db": observed_postmax,
            "postprojection_max_abs_mask_recomputed_db": recomputed_postmax,
            "postprojection_max_abs_mask_abs_error_db": abs(observed_postmax - recomputed_postmax),
            "projection_bound_db": max_limit,
            "projection_within_declared_bound": observed_postmax <= max_limit + 1e-5,
            "component_decomposition_uses_observed_single_global_scale": True,
        },
        "band_statistics": band_summaries,
        "polarity_component_asymmetry": component_polarity,
    }


def _gate_diagnostics(output: Mapping[str, Any]) -> Dict[str, Any]:
    import numpy as np

    arrays: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    valid = np.asarray(output["valid_transition_mask"], dtype=np.bool_)
    for sign in ("positive", "negative"):
        gate = np.asarray(output[sign + "_flux_gate"], dtype=np.float64)
        flux = np.asarray(output[sign + "_relative_flux"], dtype=np.float64)
        if gate.ndim != 1 or flux.shape != gate.shape or valid.shape != gate.shape:
            raise DiagnosticError("%s gate/flux/validity traces have inconsistent clocks" % sign)
        if not np.isfinite(gate).all() or not np.isfinite(flux).all():
            raise DiagnosticError("%s gate/flux trace is nonfinite" % sign)
        all_frame_dc = float(gate.mean(dtype=np.float64))
        valid_dc = float(gate[valid].mean(dtype=np.float64)) if bool(valid.any()) else None
        arrays[sign + "_relative_flux"] = flux.astype(np.float32)
        arrays[sign + "_gate"] = gate.astype(np.float32)
        arrays[sign + "_gate_centered_all_frames"] = (gate - all_frame_dc).astype(np.float32)
        arrays[sign + "_gate_centered_valid_transitions"] = np.where(
            valid, gate - (valid_dc if valid_dc is not None else 0.0), 0.0
        ).astype(np.float32)
        summary[sign] = {
            "threshold": float(output[sign + "_flux_threshold"]),
            "has_valid_support": bool(output[sign + "_flux_has_valid_support"]),
            "valid_transition_count_reported_by_head": int(output[sign + "_flux_valid_transition_count"]),
            "valid_transition_count_recomputed": int(valid.sum()),
            "gate_dc_mean_all_stft_frames": all_frame_dc,
            "gate_mean_valid_transitions": valid_dc,
            "gate_all_frame_mean_square_fluctuation": float(np.mean((gate - all_frame_dc) ** 2)),
            "valid_transition_mean_square_fluctuation": (
                float(np.mean((gate[valid] - valid_dc) ** 2)) if valid_dc is not None else None
            ),
        }
    if summary["positive"]["valid_transition_count_reported_by_head"] != int(valid.sum()):
        raise DiagnosticError("positive valid transition count differs from the shared mask")
    if summary["negative"]["valid_transition_count_reported_by_head"] != int(valid.sum()):
        raise DiagnosticError("negative valid transition count differs from the shared mask")
    positive_gate = np.asarray(output["positive_flux_gate"], dtype=np.float64)[valid]
    negative_gate = np.asarray(output["negative_flux_gate"], dtype=np.float64)[valid]
    positive_flux = np.asarray(output["positive_relative_flux"], dtype=np.float64)[valid]
    negative_flux = np.asarray(output["negative_relative_flux"], dtype=np.float64)[valid]
    summary["polarity_asymmetry_on_valid_source_transitions"] = {
        "n": int(valid.sum()),
        "positive_gate_mean": float(positive_gate.mean(dtype=np.float64)) if positive_gate.size else None,
        "negative_gate_mean": float(negative_gate.mean(dtype=np.float64)) if negative_gate.size else None,
        "positive_minus_negative_gate_mean": (
            float(positive_gate.mean(dtype=np.float64) - negative_gate.mean(dtype=np.float64))
            if positive_gate.size else None),
        "positive_relative_flux_mean": float(positive_flux.mean(dtype=np.float64)) if positive_flux.size else None,
        "negative_relative_flux_mean": float(negative_flux.mean(dtype=np.float64)) if negative_flux.size else None,
        "positive_minus_negative_flux_mean": (
            float(positive_flux.mean(dtype=np.float64) - negative_flux.mean(dtype=np.float64))
            if positive_flux.size else None),
        "interpretation": "positive/negative source-clock flux polarity comparison; not phonetic event labeling",
    }
    arrays["active_frame_mask"] = np.asarray(output["active_frame_mask"], dtype=np.bool_)
    arrays["valid_transition_mask"] = valid
    arrays["stft_frame_centers_source_samples"] = np.asarray(
        output["stft_frame_centers_source_samples"], dtype=np.int64
    )
    return {"arrays": arrays, "summary": summary}


def run_analysis(request_path: Path, output_dir: Path) -> Dict[str, Any]:
    import numpy as np
    import torch

    if output_dir.exists() or output_dir.is_symlink():
        raise DiagnosticError("output directory must be fresh: %s" % output_dir)
    if output_dir.parent.is_symlink() or not output_dir.parent.is_dir():
        raise DiagnosticError("output parent must be an existing real directory")
    req_pin = _file_pin(request_path)
    if req_pin["bytes"] > 128 * 1024:
        raise DiagnosticError("analysis request exceeds the local safety limit")
    request_raw = request_path.read_bytes()
    request = json.loads(request_raw.decode("utf-8"))
    if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA:
        raise DiagnosticError("unexpected analysis request schema")
    if request.get("status") != "READ_ONLY_DIAGNOSTIC_ONLY":
        raise DiagnosticError("request must declare READ_ONLY_DIAGNOSTIC_ONLY")
    if request.get("request_sha256") != _self_hash(request, "request_sha256"):
        raise DiagnosticError("analysis request canonical self hash is invalid")

    root_text = request.get("project_root")
    if not isinstance(root_text, str) or not Path(root_text).is_absolute():
        raise DiagnosticError("project_root must be absolute")
    project_root = Path(root_text)
    if project_root.is_symlink() or not project_root.is_dir():
        raise DiagnosticError("project_root must be a real directory")
    project_root = project_root.resolve(strict=True)

    family, family_raw, family_pin = _read_pinned_json(
        request.get("family_plan"), "family plan", "plan_sha256"
    )
    point_plan, point_raw, point_pin = _read_pinned_json(
        request.get("point_plan"), "point plan", "plan_sha256"
    )
    if family.get("schema_version") != "cross-tfg-audio-eval-v1" or family.get("status") != "FROZEN":
        raise DiagnosticError("family plan is not the expected frozen Stage-A producer")
    if point_plan.get("schema_version") != "cross-tfg-audio-eval-v1" or point_plan.get("status") != "FROZEN":
        raise DiagnosticError("point plan is not the expected frozen Stage-A producer")
    if point_plan.get("scientific_input_pins_sha256") != family.get("scientific_input_pins_sha256"):
        raise DiagnosticError("point and family plans do not bind the same scientific input pins")

    model_id = request.get("model_id")
    sample_id = request.get("sample_id")
    method_id = request.get("method_id")
    if model_id not in MODEL_ORDER or not all(isinstance(x, str) and x for x in (sample_id, method_id)):
        raise DiagnosticError("model/sample/method identity is incomplete")
    methods = [item for item in point_plan.get("methods", [])
               if isinstance(item, Mapping) and item.get("method_id") == method_id]
    if len(methods) != 1:
        raise DiagnosticError("selected point method is not unique")
    method = methods[0]
    config = method.get("config")
    if not isinstance(config, Mapping):
        raise DiagnosticError("selected method has no serialized config")
    method_config_sha = _sha256(_canonical_bytes(config))
    if method_config_sha != method.get("config_sha256"):
        raise DiagnosticError("selected method config hash is invalid")
    theta = config.get("theta")
    if (not isinstance(theta, list) or len(theta) != 9
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)) for x in theta)):
        raise DiagnosticError("selected point theta must be nine finite serialized values")
    if _sha256(_canonical_bytes(theta)) != config.get("theta_sha256"):
        raise DiagnosticError("selected point theta canonical SHA is invalid")
    final_selection_receipt = None
    final_selection_pin = None
    selection_classification = "PROPOSAL_POINT_MECHANISM_DIAGNOSTIC_ONLY"
    if (method.get("kind") == "fit40_spsa_proposal"
            and config.get("method") == "fit40_spsa_proposal"
            and config.get("schema") == "fit40_spsa_proposal_transform_v1"):
        sample_ids = config.get("sample_ids")
        if (not isinstance(sample_ids, list) or len(sample_ids) != 8
                or len(set(sample_ids)) != 8 or sample_id not in sample_ids):
            raise DiagnosticError("proposal sample must belong to its exact ordered eight-clip subset")
    elif (method.get("kind") == "fit40_spsa_center"
          and config.get("method") == "fit40_spsa_center"
          and config.get("schema") == "fit40_spsa_center_transform_v1"
          and config.get("point_kind") == "spsa_center"):
        if type(config.get("source_step")) is not int or config.get("source_step") not in CENTER_STEPS:
            raise DiagnosticError("center source step is outside the frozen k=5/10/15/20 centers")
        center_roster = _validate_center_fit_roster(
            family, point_plan, method_id, method_config_sha, config.get("binding_sha256")
        )
        if sample_id not in center_roster:
            raise DiagnosticError("requested sample is outside the actual 40-row center roster")
        final_selection_receipt, final_selection_pin, selection_classification = (
            _validate_final_center_selection(request, family, family_pin, method, config, theta, method_id)
        )
    else:
        raise DiagnosticError("point method is neither a frozen SPSA proposal nor a final-center method")

    parent_rows = [row for row in family.get("rows", [])
                   if isinstance(row, Mapping) and row.get("model_id") == model_id
                   and row.get("sample_id") == sample_id and row.get("method_id") == "identity"]
    point_rows = [row for row in point_plan.get("rows", [])
                  if isinstance(row, Mapping) and row.get("model_id") == model_id
                  and row.get("sample_id") == sample_id and row.get("method_id") == method_id]
    if len(parent_rows) != 1 or len(point_rows) != 1:
        raise DiagnosticError("family identity row or point row is not unique")
    parent_row = parent_rows[0]
    point_row = point_rows[0]
    if (parent_row.get("split") != "fit" or parent_row.get("score_audio") != parent_row.get("natural_pcm")
            or point_row.get("split") != "fit" or point_row.get("score_audio") != point_row.get("natural_pcm")
            or point_row.get("method_config_sha256") != method_config_sha):
        raise DiagnosticError("selected point changed Fit identity or natural-score-audio lineage")
    # A frozen point-plan row is a blueprint; the actual producer leaves its
    # spsa_binding and generator_condition_input null until application. Bind
    # the row here through the immutable method binding, then validate the
    # realized theta/output from the actual shared and v05 receipts below.
    if (point_row.get("spsa_point_binding_sha256") != config.get("binding_sha256")
            or point_row.get("method_id") != method_id
            or point_row.get("method_config_sha256") != method_config_sha):
        raise DiagnosticError("point-plan blueprint does not bind the exact selected method point")

    shared_receipt, shared_raw, shared_pin = _read_pinned_json(
        request.get("shared_application_receipt"), "shared application receipt", "self_sha256"
    )
    origin_receipt, origin_raw, origin_pin = _read_pinned_json(
        request.get("origin_application_receipt"), "v05 origin application receipt", "self_sha256"
    )
    if (shared_receipt.get("schema") != "cross_tfg_fit_snapshot_shared_application_v1"
            or shared_receipt.get("status") != "PASS_FIT_SNAPSHOT_SHARED_APPLICATION_ONLY"
            or shared_receipt.get("sample_id") != sample_id):
        raise DiagnosticError("existing shared application receipt has the wrong producer identity")
    binding = shared_receipt.get("method_binding")
    if (not isinstance(binding, Mapping) or binding.get("method_id") != method_id
            or binding.get("method_config_sha256") != method_config_sha
            or binding.get("theta_sha256") != config.get("theta_sha256")
            or binding.get("theta") != theta
            or binding.get("source_snapshot_step") != config.get("source_snapshot_step")):
        raise DiagnosticError("shared application receipt is not bound to the selected point")
    if (shared_receipt.get("point_plan", {}).get("bytes") != point_pin["bytes"]
            or shared_receipt.get("point_plan", {}).get("sha256") != point_pin["sha256"]):
        raise DiagnosticError("shared application receipt references another point plan")
    origin_ref = shared_receipt.get("origin_v05_application_receipt")
    if (not isinstance(origin_ref, Mapping) or origin_ref.get("bytes") != origin_pin["bytes"]
            or origin_ref.get("sha256") != origin_pin["sha256"]
            or origin_ref.get("self_sha256") != origin_receipt.get("self_sha256")):
        raise DiagnosticError("shared receipt does not bind the supplied immutable v05 origin")
    if (origin_receipt.get("schema") != "cross_tfg_fit_snapshot_generator_drive_result_v1"
            or origin_receipt.get("status") != "PASS_FIT_SNAPSHOT_GENERATOR_DRIVE_ONLY"
            or origin_receipt.get("method_id") != method_id
            or origin_receipt.get("theta_sha256") != config.get("theta_sha256")
            or origin_receipt.get("sample_id") != sample_id):
        raise DiagnosticError("v05 origin receipt is not the exact selected point application")

    natural = parent_row.get("natural_pcm")
    if not isinstance(natural, Mapping):
        raise DiagnosticError("identity parent lacks natural PCM metadata")
    sample_pin = family.get("scientific_input_pins", {}).get("natural_pcm_" + sample_id)
    if not isinstance(sample_pin, Mapping):
        raise DiagnosticError("family plan has no exact sample natural PCM pin")
    natural_path = project_root.joinpath(*Path(sample_pin.get("path", "")).parts)
    if (natural_path.is_symlink() or str(natural_path.resolve(strict=True)) != natural.get("path")
            or not natural_path.is_file()):
        raise DiagnosticError("identity source path differs from the exact plan-root path")
    natural_pin_actual = _file_pin(natural_path)
    if (natural_pin_actual["bytes"] != sample_pin.get("bytes")
            or natural_pin_actual["sha256"] != sample_pin.get("sha256")
            or natural_pin_actual["sha256"] != natural.get("file_sha256")):
        raise DiagnosticError("natural source file differs from the exact sample plan pin")

    helper_path = project_root / HELPER05_REL
    helper05_pin = _file_pin(helper_path)
    if (helper05_pin["bytes"], helper05_pin["sha256"]) != (HELPER05_BYTES, HELPER05_SHA256):
        raise DiagnosticError("frozen helper05 source differs from its exact raw pin")
    helper = _load_pinned_module("fit40_snapshot_spsa_application_attempt05_mask_diag", helper_path,
                                  {"bytes": HELPER05_BYTES, "sha256": HELPER05_SHA256})
    point_context = helper.validate_snapshot_point(
        project_root=project_root, plan=point_plan, method=method, parent_row=parent_row
    )
    if (point_context["config_sha256"] != method_config_sha
            or point_context["config"].get("theta_sha256") != config.get("theta_sha256")):
        raise DiagnosticError("frozen snapshot validator returned another method or theta")
    extractor = _load_pinned_module("fit40_training_snapshots_attempt02_mask_diag",
                                    point_context["extractor_path"], point_context["extractor_pin"])
    transform = extractor.FitSnapshotTransform(
        snapshot=point_context["snapshot"],
        head_module_path=point_context["head_path"],
        head_module_sha256=point_context["head_pin"]["sha256"],
        baselines_path=point_context["baseline_path"],
        device="cpu",
    )
    with torch.no_grad():
        transform.head.band_contrasts_db.copy_(torch.as_tensor(theta, dtype=torch.float32, device="cpu").reshape(3, 3))
    transform.head.eval()
    # FitSnapshotTransform loads the head from this exact pin but keeps its
    # dynamically created module local. Load the same byte-pinned source under
    # a diagnostic name to reuse its smoothing helper without changing the
    # frozen transform or relying on sys.modules registration.
    head_module = _load_pinned_module(
        "shared_recipient_transition_head_v3_mask_gate_diag",
        point_context["head_path"], point_context["head_pin"]
    )

    sys.path.insert(0, str(project_root))
    audio_io_path = point_context["audio_io_path"]
    audio_io = _load_pinned_module("fit40_snapshot_audio_io_mask_diag", audio_io_path,
                                  point_context["audio_io_pin"])
    samples, input_info = audio_io.read_canonical_wav(natural_path, target_rate=16000)
    if (input_info.file_sha256 != natural.get("file_sha256")
            or input_info.pcm_float32_sha256 != natural.get("pcm_sha256")
            or input_info.frames != natural.get("sample_count")
            or input_info.sample_rate != 16000 or input_info.source_sample_rate != 16000
            or input_info.source_channels != 1):
        raise DiagnosticError("canonical source PCM does not reproduce the frozen source clock")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        raise DiagnosticError("fresh CPU process could not establish one interop thread") from exc
    output_samples, captured = _capture_forward(transform, torch, samples)
    decomposition = _component_decomposition(transform.head, head_module, captured, torch)
    gates = _gate_diagnostics(captured)

    output_info = shared_receipt.get("output_generator_drive_wav")
    if not isinstance(output_info, Mapping):
        raise DiagnosticError("shared receipt omits its generated waveform pin")
    shared_wav_path = Path(str(output_info.get("path", "")))
    actual_wav_pin = _file_pin(shared_wav_path)
    if (actual_wav_pin["bytes"] != output_info.get("bytes")
            or actual_wav_pin["sha256"] != output_info.get("sha256")):
        raise DiagnosticError("existing shared generator WAV differs from its receipt")
    existing_samples, existing_info = audio_io.read_canonical_wav(shared_wav_path, target_rate=16000)
    if (existing_info.pcm_float32_sha256 != output_info.get("pcm_float32_sha256")
            or existing_info.frames != input_info.frames or existing_info.sample_rate != 16000):
        raise DiagnosticError("existing shared waveform does not reproduce its file/PCM/source-clock pins")
    output_array = np.ascontiguousarray(output_samples, dtype=np.float32)
    existing_array = np.ascontiguousarray(existing_samples, dtype=np.float32)
    if output_array.shape != existing_array.shape:
        raise DiagnosticError("local CPU diagnostic and actual application have different sample counts")
    output_pcm_sha = _sha256(output_array.astype("<f4", copy=False).tobytes())
    max_abs_output_delta = float(np.max(np.abs(output_array.astype(np.float64) - existing_array.astype(np.float64))))
    projection_evidence = origin_receipt.get("projection_evidence")
    if not isinstance(projection_evidence, Mapping):
        raise DiagnosticError("actual v05 origin receipt omits projection evidence")
    for name in ("preprojection_max_abs_mask_db", "postprojection_max_abs_mask_db", "projection_scale"):
        if name not in projection_evidence:
            raise DiagnosticError("actual v05 projection evidence lacks %s" % name)
    checks = decomposition["checks"]
    origin_projection_delta = {
        "preprojection_max_abs_mask_db": abs(float(checks["preprojection_max_abs_mask_db"])
                                               - float(projection_evidence["preprojection_max_abs_mask_db"])),
        "postprojection_max_abs_mask_db": abs(float(checks["postprojection_max_abs_mask_db"])
                                                - float(projection_evidence["postprojection_max_abs_mask_db"])),
        "projection_scale": abs(float(checks["global_projection_scale"])
                                 - float(projection_evidence["projection_scale"])),
    }
    if max(origin_projection_delta.values()) > 1e-5:
        raise DiagnosticError("replayed projection evidence differs from the actual pinned origin by >1e-5")

    gate_arrays = gates["arrays"]
    mask_arrays = decomposition["arrays"]
    gate_arrays.update(mask_arrays)
    arrays_buf = io.BytesIO()
    np.savez_compressed(arrays_buf, **gate_arrays)
    arrays_raw = arrays_buf.getvalue()
    if not arrays_raw:
        raise DiagnosticError("diagnostic arrays archive is empty")

    output_dir.mkdir(mode=0o700)
    arrays_path = output_dir / "mask_gate_arrays_attempt01.npz"
    with arrays_path.open("xb") as handle:
        handle.write(arrays_raw)
        handle.flush()
        os.fsync(handle.fileno())
    arrays_pin = _file_pin(arrays_path)
    runtime = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "device": "cpu",
        "intraop_threads": int(torch.get_num_threads()),
        "interop_threads": int(torch.get_num_interop_threads()),
        "optimizer_created_or_stepped": False,
        "video_render_or_score_performed": False,
        "cuda_used": False,
    }
    head_pin = point_context["head_pin"]
    snapshot = point_context["snapshot"]
    packet: Dict[str, Any] = {
        "schema": ANALYZER_SCHEMA,
        "status": "PASS_SOURCE_PINNED_MASK_AND_GATE_DIAGNOSTIC_ONLY",
        "scope": "one receipt-bound Fit point/sample; CPU head diagnostic only; no score, video, optimizer, or selector execution",
        "request": {**req_pin, "self_sha256": request.get("request_sha256")},
        "family_plan": family_pin,
        "point_plan": point_pin,
        "method": {
            "method_id": method_id,
            "kind": method.get("kind"),
            "point_kind": config.get("point_kind", "spsa_proposal"),
            "method_config_sha256": method_config_sha,
            "proposal_id": config.get("proposal_id"),
            "pair_id": config.get("pair_id"),
            "iteration": config.get("iteration"),
            "side": config.get("side"),
            "source_snapshot_step": config.get("source_snapshot_step"),
            "source_seed_method_id": config.get("source_seed_method_id"),
            "snapshot_artifact_sha256": config.get("artifact_sha256"),
            "source_step": config.get("source_step"),
            "theta": list(theta),
            "theta_sha256": config.get("theta_sha256"),
            "selection_claim": selection_classification,
        },
        "final_selection": ({
            "receipt": final_selection_pin,
            "selection_status": final_selection_receipt.get("selection_status"),
            "selected_method_id": final_selection_receipt.get("final_selection", {}).get("selected_method_id"),
            "refinement_seed_method_id": final_selection_receipt.get("final_selection", {}).get("refinement_seed_method_id"),
            "fit_candidate_admitted": final_selection_receipt.get("selection_status") == "ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV",
            "portable_or_final_dev_claimed": False,
            "classification": selection_classification,
        } if final_selection_receipt is not None else {
            "receipt": None,
            "classification": selection_classification,
            "portable_or_final_dev_claimed": False,
        }),
        "sample": {"sample_id": sample_id, "model_id": model_id,
                   "parent_identity_row_id": parent_row["row_id"],
                   "point_row_id": point_row["row_id"]},
        "source_audio": {
            "file": natural_pin_actual,
            "pcm_float32_sha256": input_info.pcm_float32_sha256,
            "sample_count": int(input_info.frames),
            "sample_rate_hz": int(input_info.sample_rate),
            "channels": 1,
            "source_encoding": input_info.source_encoding,
        },
        "pinned_sources": {
            "helper05": helper05_pin,
            "snapshot_extractor": point_context["extractor_pin"],
            "head_implementation": head_pin,
            "head_constructor_sha256": snapshot.get("head_constructor_sha256"),
            "snapshot_record": point_context["record_pin"],
            "snapshot_index": point_context["index_pin"],
            "audio_io": point_context["audio_io_pin"],
            "baselines": point_context["baseline_pin"],
        },
        "existing_application": {
            "shared_receipt": shared_pin,
            "origin_v05_receipt": origin_pin,
            "generator_drive_wav": {**actual_wav_pin,
                                     "pcm_float32_sha256": existing_info.pcm_float32_sha256,
                                     "sample_count": int(existing_info.frames)},
            "replayed_output_pcm_float32_sha256": output_pcm_sha,
            "replayed_vs_actual_output_max_abs_delta": max_abs_output_delta,
            "projection_evidence_absolute_differences_vs_origin": origin_projection_delta,
        },
        "forward": {
            "head_schema": captured.get("schema", snapshot.get("head_schema")),
            "stft_frame_count": int(captured["positive_flux_gate"].shape[0]),
            "source_clock_preserved": int(output_array.size) == int(input_info.frames),
            "positive_gate": gates["summary"]["positive"],
            "negative_gate": gates["summary"]["negative"],
            "polarity_asymmetry_on_valid_source_transitions": gates["summary"]["polarity_asymmetry_on_valid_source_transitions"],
            "mask_decomposition": checks,
            "four_band_statistics_pre_common_mode_policy_postprojection": decomposition["band_statistics"],
            "polarity_component_asymmetry": decomposition["polarity_component_asymmetry"],
            "four_band_effective_mask_mean_db_preprojection": mask_arrays["four_band_preprojection_mean_db"].tolist(),
            "four_band_effective_mask_mean_db_postprojection": mask_arrays["four_band_postprojection_mean_db"].tolist(),
            "four_band_postprojection_gain_summary_by_band": decomposition["band_statistics"]["postprojection"]["per_band"],
            "source_clock": {
                "sample_rate_hz": int(input_info.sample_rate),
                "sample_count": int(input_info.frames),
                "duration_seconds_rational": {"numerator": int(input_info.frames), "denominator": int(input_info.sample_rate)},
                "stft_frame_count": int(captured["stft_frame_centers_source_samples"].shape[0]),
                "stft_frame_center_sample_indices": [int(x) for x in captured["stft_frame_centers_source_samples"].tolist()],
                "stft_frame_center_seconds_rational": [
                    {"numerator": int(x), "denominator": int(input_info.sample_rate)}
                    for x in captured["stft_frame_centers_source_samples"].tolist()
                ],
                "valid_transition_count": int(np.asarray(captured["valid_transition_mask"], dtype=np.bool_).sum()),
            },
            "preprojection_policy": transform.head.frequency_common_mode_policy,
            "preprojection_component_order": ["static", "positive_flux", "negative_flux"],
            "fourth_band_role": "fixed_reference_band; three learned contrasts per temporal component",
            "global_projection_interpretation": "one observed smooth radial tanh scale multiplies the full summed policy mask and its additive components",
            "static_only_limit": (
                "Static-only is not a pure timing ablation: positive/negative gate-weighted terms have observed nonzero temporal means, so dropping them would remove that DC contribution and may change the summed preprojection maximum; the head would then recompute one global radial tanh scale for the changed full mask. This diagnostic did not run a static-only counterfactual."
            ),
            "polarity_limit": "positive/negative flux are source-clock onset/offset polarity proxies, not independently labeled phonetic onsets or offsets",
        },
        "diagnostic_arrays": arrays_pin,
        "runtime": runtime,
        "tests_and_claim_limits": {
            "mask_gate_diagnostic_only": True,
            "does_not_claim_static_only_is_a_pure_timing_ablation": True,
            "does_not_claim_native_video_or_syncnet_effect": True,
            "does_not_claim_final_theta_or_admission": True,
        },
    }
    packet["self_sha256"] = _self_hash(packet, "self_sha256")
    receipt_path = output_dir / "mask_gate_diagnostic_receipt_attempt01.json"
    receipt_raw = (json.dumps(packet, sort_keys=True, indent=2, ensure_ascii=False,
                              allow_nan=False) + "\n").encode("utf-8")
    with receipt_path.open("xb") as handle:
        handle.write(receipt_raw)
        handle.flush()
        os.fsync(handle.fileno())
    return {"receipt_path": str(receipt_path.resolve(strict=True)),
            "receipt_bytes": len(receipt_raw), "receipt_sha256": _sha256(receipt_raw),
            "receipt_self_sha256": packet["self_sha256"],
            "arrays_path": str(arrays_path.resolve(strict=True)),
            "arrays_bytes": arrays_pin["bytes"], "arrays_sha256": arrays_pin["sha256"]}


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_analysis(args.request, args.output_dir)
    except Exception as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": "PASS_SOURCE_PINNED_MASK_AND_GATE_DIAGNOSTIC_ONLY", **result},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
