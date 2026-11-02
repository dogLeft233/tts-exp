"""Produce source-pinned, score-free V3 mask and gate diagnostics for one Fit point.

This utility validates a real frozen SPSA point and its existing shared
application receipts, then runs the exact pinned snapshot head once on the
point's original PCM on CPU. It records the head's source-clock gates, the
four-band mask before and after the existing global projection, and an
additive component decomposition. It does not score, render video, optimize,
select, or write transformed audio.
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

HELPER07_REL = Path("scripts/audio_enhancement/fit40_snapshot_spsa_application_attempt07.py")
HELPER07_BYTES = 66046
HELPER07_SHA256 = "989b28c6239c2770bee6799afa85f0b0f0b48ec39bdd584cf68c035c464f1ec4"
HELPER05_REL = Path("scripts/audio_enhancement/fit40_snapshot_spsa_application_attempt05.py")
HELPER05_BYTES = 56593
HELPER05_SHA256 = "3a33cf6f148de36d0433074c2efadb98167880ed5b7f1f66138a797a047afc5b"
ANALYZER_SCHEMA = "cross_tfg_fit_snapshot_mask_gate_diagnostic_v1"
REQUEST_SCHEMA = "cross_tfg_fit_snapshot_mask_gate_analysis_request_v1"
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

    pre_band_per_frame = per_band(policy_actual)
    post_band_per_frame = per_band(projected_actual)
    component_post_band_per_frame = torch.stack(
        [per_band(projected_components[index]) for index in range(3)], dim=0
    )
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
        "policy_mask_db_before_projection": policy_actual.numpy(),
        "projected_log_magnitude_mask_db": projected_actual.numpy(),
        "four_band_preprojection_mean_db_by_frame": pre_band_per_frame.numpy(),
        "four_band_postprojection_mean_db_by_frame": post_band_per_frame.numpy(),
        "four_band_projected_component_mean_db_by_frame": component_post_band_per_frame.numpy(),
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
    if sample_id not in config.get("sample_ids", []):
        raise DiagnosticError("sample is outside this actual proposal point's declared subset")

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

    helper_path = project_root / HELPER07_REL
    helper = _load_pinned_module("fit40_snapshot_spsa_application_attempt07_mask_diag", helper_path,
                                  {"bytes": HELPER07_BYTES, "sha256": HELPER07_SHA256})
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
    head_module = sys.modules.get("fit40_snapshot_head")
    if head_module is None or Path(str(head_module.__file__)).resolve(strict=True) != point_context["head_path"].resolve(strict=True):
        raise DiagnosticError("pinned head module was not loaded from the snapshot-index source")

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
        "scope": "one actual Fit SPSA point/sample; CPU head forward only; no score, video, optimizer, or selection",
        "request": {**req_pin, "self_sha256": request.get("request_sha256")},
        "family_plan": family_pin,
        "point_plan": point_pin,
        "method": {
            "method_id": method_id,
            "kind": method.get("kind"),
            "method_config_sha256": method_config_sha,
            "proposal_id": config.get("proposal_id"),
            "pair_id": config.get("pair_id"),
            "iteration": config.get("iteration"),
            "side": config.get("side"),
            "source_snapshot_step": config.get("source_snapshot_step"),
            "source_seed_method_id": config.get("source_seed_method_id"),
            "snapshot_artifact_sha256": config.get("artifact_sha256"),
            "theta": list(theta),
            "theta_sha256": config.get("theta_sha256"),
            "selection_claim": "NONE_DIAGNOSTIC_ONLY",
        },
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
            "helper07": _file_pin(helper_path),
            "helper05": point_context["helper05_pin"],
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
            "mask_decomposition": checks,
            "four_band_effective_mask_mean_db_preprojection": mask_arrays["four_band_preprojection_mean_db"].tolist(),
            "four_band_effective_mask_mean_db_postprojection": mask_arrays["four_band_postprojection_mean_db"].tolist(),
            "preprojection_policy": transform.head.frequency_common_mode_policy,
            "preprojection_component_order": ["static", "positive_flux", "negative_flux"],
            "fourth_band_role": "fixed_reference_band; three learned contrasts per temporal component",
            "global_projection_interpretation": "one observed smooth radial tanh scale multiplies the full summed policy mask and its additive components",
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
