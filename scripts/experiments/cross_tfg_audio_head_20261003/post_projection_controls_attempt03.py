"""Source-only definitions for frozen-theta post-projection Fit40 controls.

This module does not load a checkpoint, create an optimizer, score/render a
video, or execute any control at import time.  A future caller must supply the
already-loaded, frozen head plus source/baseline tensors and actual completion
lineage.  The helper replays only the existing head forward to retrieve its
actual globally projected mask, then applies controls to that mask without
reprojection or gradients.

The circular rolls deliberately wrap 64 STFT frames (512 ms at the pinned
16 kHz / 128-sample hop).  Per-bin mask summaries and mean comparisons reduce
the original float32 mask values in float64 and report tolerances rather than
claiming bit-exact means.  Both the analytically constructed masked-STFT phase
and the phase after reanalysis of the synthesized PCM are summarized.  Roll
seams, their approximate frame neighborhood, and reflected-STFT edge
interaction are recorded, along with the actual valid-transition and active-
frame support masks and mode-specific gate sequences. No frames or score
queries are removed, and these controls are not pure timing interventions or
causal evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA = "cross_tfg_fit40_post_projection_control_diagnostic_v2"
HEAD_SCHEMA = "cross_tfg_shared_recipient_transition_head_v3_d0_gate_dc_attempt01"
RUNNER_SCHEMA = "cross_tfg_fit40_native_transition_head_v3_d0_direction_gatedc_runner_attempt02"
COMPLETION_STATUS = "COMPLETE_FIT40_DIRECT_D0_DIRECTION_GATE_DC_120_STEPS"
CONTROL_NAMES = ("post_proj_mean", "post_proj_roll_plus64", "post_proj_roll_minus64")
EXPECTED_SAMPLE_RATE_HZ = 16_000
EXPECTED_N_FFT = 512
EXPECTED_HOP_LENGTH = 128
EXPECTED_WIN_LENGTH = 512
EXPECTED_FPS_NUM = 25
EXPECTED_FPS_DEN = 1
EXPECTED_MASK_LIMIT_DB = 3.0
ROLL_MS = 512
MEAN_CONTROL_ABS_TOL_DB = 1e-6
ROLL_MEAN_ABS_TOL_DB = 1e-10


class ControlError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pin_shape(pin: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(pin, dict) or set(pin) != {"path", "bytes", "sha256"}:
        raise ControlError(f"{label} must be an exact path/bytes/sha256 pin")
    if not isinstance(pin["bytes"], int) or pin["bytes"] <= 0:
        raise ControlError(f"{label} byte count is invalid")
    digest = pin["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ControlError(f"{label} SHA-256 is not lowercase hexadecimal")
    return {"path": str(pin["path"]), "bytes": pin["bytes"], "sha256": digest}


def _tensor_f32_bytes(value: Any, torch: Any) -> bytes:
    array = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return array.astype("<f4", copy=False).tobytes(order="C")


def _parameter_payload(head: Any, torch: Any) -> dict[str, Any]:
    values = head.band_contrasts_db.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    values = values.astype("<f4", copy=False)
    if list(values.shape) != [3, 3]:
        raise ControlError("frozen head parameter must remain exactly [3,3]")
    return {
        "name": "band_contrasts_db",
        "shape": [3, 3],
        "dtype": "float32",
        "temporal_components": ["static", "positive_flux", "negative_flux"],
        "non_reference_band_order": [0, 1, 2],
        "values": values.tolist(),
    }


def _parameter_sha(head: Any, torch: Any) -> str:
    return _sha(_canonical(_parameter_payload(head, torch)))


def _decoded_object(raw: bytes, expected: Mapping[str, Any], *, label: str) -> None:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControlError(f"{label} raw bytes are not UTF-8 JSON") from exc
    if not isinstance(value, dict) or dict(expected) != value:
        raise ControlError(f"{label} decoded object differs from its pinned raw bytes")


def _assert_raw_pin(raw: bytes, pin: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    normalized = _pin_shape(dict(pin), label=label)
    if len(raw) != normalized["bytes"] or _sha(raw) != normalized["sha256"]:
        raise ControlError(f"{label} actual bytes do not match the supplied whole-file pin")
    return normalized


def _require_pin_equal(left: Any, right: Any, *, label: str) -> None:
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ControlError(f"{label} must be represented by two complete pins")
    if _pin_shape(left, label=f"{label}.left") != _pin_shape(right, label=f"{label}.right"):
        raise ControlError(f"{label} pins disagree")


def _completion_artifact_map(completion: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": completion.get("checkpoint"),
        "training_record": completion.get("training_record"),
        "runtime_source_pins": completion.get("runtime_source_pins"),
        "baseline_arrays": completion.get("baseline_arrays"),
        "zero_gate_arrays": completion.get("zero_gate_arrays"),
        "strict_backward_qualification": completion.get("strict_backward_qualification"),
        "training_score_arrays": completion.get("training_score_arrays"),
        "training_steps": completion.get("training_steps"),
        "fit40_gate_dc_control": completion.get("fit40_gate_dc_control"),
    }


def _validate_fit_lineage(
    *,
    actual_completion: Mapping[str, Any],
    actual_completion_raw: bytes,
    actual_checkpoint: Mapping[str, Any],
    actual_checkpoint_raw: bytes,
    actual_training_record: Mapping[str, Any],
    actual_training_record_raw: bytes,
    actual_runtime_source_pins: Mapping[str, Any],
    actual_runtime_source_pins_raw: bytes,
    actual_native_geometry_manifest: Mapping[str, Any],
    actual_native_geometry_manifest_raw: bytes,
    actual_head_config: Mapping[str, Any],
    actual_head_config_raw: bytes,
    actual_head_constructor: Mapping[str, Any],
    expected_checkpoint_parameter_sha256: str,
    source_lineage_pins: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Cross-bind actual completion files before any diagnostic replay."""
    required = {
        "completion", "checkpoint", "training_record", "runtime_source_pins",
        "head_config", "head_implementation", "runner", "baseline_arrays",
        "zero_gate_arrays", "strict_backward_qualification", "training_score_arrays",
        "training_steps", "fit40_gate_dc_control", "fit_source_manifest",
        "native_geometry_manifest", "parent_plan",
    }
    if not isinstance(source_lineage_pins, Mapping) or not required.issubset(source_lineage_pins):
        missing = sorted(required - set(source_lineage_pins))
        raise ControlError(f"source/runtime/checkpoint lineage pins are incomplete: {missing}")
    pins = {
        str(key): _pin_shape(dict(value), label=f"source_lineage_pins.{key}")
        for key, value in source_lineage_pins.items()
    }
    for label, raw, obj in (
        ("completion", actual_completion_raw, actual_completion),
        ("checkpoint", actual_checkpoint_raw, actual_checkpoint),
        ("training_record", actual_training_record_raw, actual_training_record),
        ("runtime_source_pins", actual_runtime_source_pins_raw, actual_runtime_source_pins),
        ("native_geometry_manifest", actual_native_geometry_manifest_raw, actual_native_geometry_manifest),
        ("head_config", actual_head_config_raw, actual_head_config),
    ):
        _assert_raw_pin(raw, pins[label], label=f"source_lineage_pins.{label}")
        _decoded_object(raw, obj, label=label)

    for name, actual_pin in _completion_artifact_map(actual_completion).items():
        if name not in pins:
            raise ControlError(f"completion artifact pin is not carried in source_lineage_pins: {name}")
        _require_pin_equal(actual_pin, pins[name], label=f"completion.{name}")

    if actual_completion.get("training_record_raw_sha256") != pins["training_record"]["sha256"]:
        raise ControlError("completion raw training-record hash differs from the downloaded training record")
    if actual_training_record.get("checkpoint_sha256") != pins["checkpoint"]["sha256"]:
        raise ControlError("training record does not bind the downloaded checkpoint bytes")
    if actual_training_record.get("runtime_source_pins_sha256") != pins["runtime_source_pins"]["sha256"]:
        raise ControlError("training record does not bind the downloaded runtime-source-pins bytes")
    if actual_training_record.get("training_steps_jsonl_sha256") != pins["training_steps"]["sha256"]:
        raise ControlError("training record does not bind the 120-step ledger")
    if actual_training_record.get("training_score_arrays_sha256") != pins["training_score_arrays"]["sha256"]:
        raise ControlError("training record does not bind its training score arrays")
    if actual_training_record.get("fit_source_manifest_sha256") != pins["fit_source_manifest"]["sha256"]:
        raise ControlError("training record does not bind the Fit40 source manifest")
    if actual_training_record.get("native_geometry_manifest_sha256") != pins["native_geometry_manifest"]["sha256"]:
        raise ControlError("training record does not bind the actual native geometry manifest")
    if actual_training_record.get("parent_plan_raw_sha256") != pins["parent_plan"]["sha256"]:
        raise ControlError("training record does not bind the frozen parent plan")
    geometry_payload = {
        key: value for key, value in actual_native_geometry_manifest.items() if key != "payload_sha256"
    }
    if (
        actual_native_geometry_manifest.get("schema") != "cross_tfg_native_full512_fit_geometry_manifest_v1"
        or not isinstance(actual_native_geometry_manifest.get("rows"), list)
        or len(actual_native_geometry_manifest["rows"]) != 40
        or actual_native_geometry_manifest.get("payload_sha256") != _sha(_canonical(geometry_payload))
    ):
        raise ControlError("actual Fit40 geometry manifest schema, denominator, or canonical digest is invalid")

    if actual_completion.get("schema") != RUNNER_SCHEMA or actual_completion.get("status") != COMPLETION_STATUS:
        raise ControlError("completion is not the actual full Fit40 D0/GateDC 120-step terminal")
    if (
        actual_completion.get("completed_optimizer_steps") != 120
        or actual_completion.get("fit_rows") != 40
        or actual_completion.get("fit_speakers") != 20
        or actual_completion.get("zero_gate_rows_passed") != 40
    ):
        raise ControlError("completion does not bind the exact 40-row/20-speaker/120-step denominator")
    for key in ("selection_record_created", "development_audio_opened", "confirmation_audio_opened"):
        if actual_completion.get(key) is not False:
            raise ControlError(f"Fit-only completion has an unexpected {key} value")

    if actual_training_record.get("schema") != "cross_tfg_v3_d0_direction_gatedc_training_record_v1":
        raise ControlError("training record is not the pinned D0/GateDC record schema")
    if (
        actual_training_record.get("status") != "COMPLETE"
        or actual_training_record.get("completed_optimizer_steps") != 120
        or actual_training_record.get("planned_optimizer_steps") != 120
        or actual_training_record.get("fit_row_count") != 40
        or actual_training_record.get("fit_speaker_count") != 20
        or actual_training_record.get("selection_status") != "NOT_SELECTED"
        or actual_training_record.get("selection_record_created") is not False
        or actual_training_record.get("development_audio_opened") is not False
        or actual_training_record.get("confirmation_audio_opened") is not False
    ):
        raise ControlError("training record does not preserve the frozen Fit-only 120-step scope")
    gate_dc_record = actual_training_record.get("gate_dc_control")
    if not isinstance(gate_dc_record, dict) or (
        gate_dc_record.get("status") != "COMPLETE_FIT40_SCORE_ONLY_SAME_THETA"
        or gate_dc_record.get("branch_rows") != 80
        or gate_dc_record.get("optimizer_updates") != 0
    ):
        raise ControlError("training record lacks the complete same-theta, score-only GateDC ledger")
    _require_pin_equal(gate_dc_record.get("artifact"), pins["fit40_gate_dc_control"], label="training_record.fit40_gate_dc_control")

    if actual_checkpoint.get("schema") != "cross_tfg_v3_d0_direction_gatedc_parameter_checkpoint_v1":
        raise ControlError("checkpoint is not the pinned D0/GateDC parameter checkpoint")
    if actual_checkpoint.get("head_schema") != HEAD_SCHEMA:
        raise ControlError("checkpoint head schema differs from the pinned D0 head")
    if actual_checkpoint.get("selection_status") != "UNSELECTED_FIT_CHECKPOINT_SELECTION_RECORD_NOT_CREATED":
        raise ControlError("checkpoint is not explicitly unselected")
    checkpoint_parameter = actual_checkpoint.get("parameter")
    if not isinstance(checkpoint_parameter, dict) or set(checkpoint_parameter) != {
        "name", "shape", "dtype", "temporal_components", "non_reference_band_order", "values"
    }:
        raise ControlError("checkpoint parameter is malformed")
    if _sha(_canonical(checkpoint_parameter)) != expected_checkpoint_parameter_sha256:
        raise ControlError("checkpoint parameter digest differs from the actual run binding")
    implementation = actual_checkpoint.get("implementation")
    _require_pin_equal(implementation, pins["head_implementation"], label="checkpoint.implementation")
    if actual_training_record.get("head_implementation_sha256") != pins["head_implementation"]["sha256"]:
        raise ControlError("training record does not bind the checkpoint head implementation")
    if actual_checkpoint.get("constructor_sha256") != actual_training_record.get("constructor_sha256"):
        raise ControlError("checkpoint and training record constructor hashes disagree")
    if actual_head_config.get("schema") != "cross_tfg_shared_recipient_transition_head_v3_cpu_qualification_config_v1":
        raise ControlError("head constructor config schema differs from the exact runner input")
    config_constructor = actual_head_config.get("head_constructor")
    if not isinstance(config_constructor, dict) or dict(config_constructor) != dict(actual_head_constructor):
        raise ControlError("supplied constructor differs from the actual pinned head-config file")
    if _sha(_canonical(dict(actual_head_constructor))) != actual_checkpoint.get("constructor_sha256"):
        raise ControlError("pinned head constructor does not match checkpoint constructor digest")

    bindings = actual_checkpoint.get("training_artifact_bindings")
    if not isinstance(bindings, dict):
        raise ControlError("checkpoint lacks the exact training-artifact bindings")
    for field, pin_name in (
        ("training_steps_jsonl_sha256", "training_steps"),
        ("training_score_arrays_sha256", "training_score_arrays"),
        ("fit40_gate_dc_control_sha256", "fit40_gate_dc_control"),
        ("runtime_source_pins_sha256", "runtime_source_pins"),
        ("fit_source_manifest_sha256", "fit_source_manifest"),
        ("native_geometry_manifest_sha256", "native_geometry_manifest"),
        ("parent_plan_raw_sha256", "parent_plan"),
    ):
        if bindings.get(field) != pins[pin_name]["sha256"]:
            raise ControlError(f"checkpoint training artifact binding differs: {field}")
    if bindings.get("completed_optimizer_steps") != 120:
        raise ControlError("checkpoint does not bind exactly 120 completed updates")

    modules = actual_runtime_source_pins.get("training_modules")
    if not isinstance(modules, dict):
        raise ControlError("runtime-source-pins object lacks training_modules")
    _require_pin_equal(modules.get("v3_head"), pins["head_implementation"], label="runtime_source_pins.training_modules.v3_head")
    _require_pin_equal(actual_runtime_source_pins.get("runner"), pins["runner"], label="runtime_source_pins.runner")
    _require_pin_equal(actual_runtime_source_pins.get("head_config"), pins["head_config"], label="runtime_source_pins.head_config")

    backward = actual_training_record.get("strict_backward_qualification")
    backward_result = backward.get("result") if isinstance(backward, dict) else None
    full_objective = backward_result.get("full_objective_backward") if isinstance(backward_result, dict) else None
    if (
        not isinstance(backward, dict)
        or backward.get("status") != "PASS_FULL_OBJECTIVE_BACKWARD_BEFORE_OPTIMIZER"
        or backward.get("v3_head_adam_created_before_qualification") is not False
        or not isinstance(full_objective, dict)
        or full_objective.get("gradient_shape") != [3, 3]
        or not isinstance(full_objective.get("gradient_nonzero_scalar_count"), int)
        or full_objective.get("gradient_nonzero_scalar_count", 0) <= 0
    ):
        raise ControlError("Fit120 lacks its actual finite/nonzero full-objective backward qualification")
    _require_pin_equal(
        backward.get("artifact"), pins["strict_backward_qualification"],
        label="training_record.strict_backward_qualification",
    )
    if _canonical(actual_runtime_source_pins.get("gpu_identity")) != _canonical(actual_completion.get("gpu_identity")):
        raise ControlError("runtime-source-pins GPU identity differs from the completion receipt")
    return pins


def _assert_pinned_head(head: Any, expected_constructor: Mapping[str, Any]) -> dict[str, Any]:
    config = head.configuration()
    expected = {
        "schema": HEAD_SCHEMA,
        "sample_rate_hz": EXPECTED_SAMPLE_RATE_HZ,
        "n_fft": EXPECTED_N_FFT,
        "hop_length": EXPECTED_HOP_LENGTH,
        "win_length": EXPECTED_WIN_LENGTH,
        "window_kind": "hann",
        "window_periodic": True,
        "stft_center": True,
        "stft_pad_mode": "reflect",
        "stft_normalized": False,
        "stft_onesided": True,
        "stft_return_complex": True,
        "max_abs_mask_db": EXPECTED_MASK_LIMIT_DB,
        "video_fps_num": EXPECTED_FPS_NUM,
        "video_fps_den": EXPECTED_FPS_DEN,
        "frequency_common_mode_policy": "remove_per_frame",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ControlError(f"head configuration differs from frozen value {key}")
    for key, value in expected_constructor.items():
        if key not in config or _canonical(config[key]) != _canonical(value):
            raise ControlError(f"loaded head does not match the actual training constructor: {key}")
    frames_num = EXPECTED_SAMPLE_RATE_HZ * ROLL_MS
    frames_den = 1000 * EXPECTED_HOP_LENGTH
    if frames_num % frames_den:
        raise ControlError("512 ms is not an integer number of pinned STFT frames")
    shift = frames_num // frames_den
    if shift != 64:
        raise ControlError("pinned 512 ms control must roll by exactly 64 STFT frames")
    return config


def _tensor_summary(value: Any) -> dict[str, float]:
    return {
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "max_abs": float(value.abs().max().item()),
    }


def _per_bin_stats(mask: Any, torch: Any) -> list[dict[str, float]]:
    # Input masks remain the producer's float32 values. Statistics are
    # reduced in float64 to make the stated per-bin comparison independent of
    # float32 reduction order; this does not claim bit-exact equality.
    mask64 = mask.to(dtype=torch.float64)
    means = mask64.mean(dim=1).detach().cpu().tolist()
    minima = mask.min(dim=1).values.detach().cpu().tolist()
    maxima = mask.max(dim=1).values.detach().cpu().tolist()
    stds = mask64.std(dim=1, unbiased=False).detach().cpu().tolist()
    return [
        {
            "bin_index": index,
            "mean_db_float64_reduction": float(means[index]),
            "min_db": float(minima[index]),
            "max_db": float(maxima[index]),
            "std_population_db_float64_reduction": float(stds[index]),
        }
        for index in range(mask.shape[0])
    ]


def _per_bin_mean_comparison(
    reference: Any, candidate: Any, torch: Any, *, tolerance_db: float
) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[0] != EXPECTED_N_FFT // 2 + 1:
        raise ControlError("per-bin mean comparison requires matching [257,time] masks")
    reference_mean = reference.to(dtype=torch.float64).mean(dim=1)
    candidate_mean = candidate.to(dtype=torch.float64).mean(dim=1)
    errors = (candidate_mean - reference_mean).abs()
    errors_list = [float(value) for value in errors.detach().cpu().tolist()]
    return {
        "reduction_dtype": "float64",
        "frequency_bin_count": int(reference.shape[0]),
        "per_bin_abs_error_db": errors_list,
        "max_abs_error_db": float(errors.max().item()),
        "mean_abs_error_db": float(errors.mean().item()),
        "all_bins_within_tolerance": bool((errors <= tolerance_db).all().item()),
        "bins_within_tolerance": int((errors <= tolerance_db).sum().item()),
        "absolute_tolerance_db": tolerance_db,
        "bit_exact_mean_claim": False,
    }


def _phase_delta_summary(actual: Any, reference: Any, torch: Any) -> dict[str, Any]:
    if actual.shape != reference.shape or not torch.is_complex(actual) or not torch.is_complex(reference):
        raise ControlError("phase comparison requires matching complex STFT tensors")
    magnitude_floor = torch.finfo(reference.real.dtype).tiny
    valid = (actual.abs() > magnitude_floor) & (reference.abs() > magnitude_floor)
    count = int(valid.sum().item())
    if not count:
        return {
            "comparison": "principal_angle(actual * conjugate(reference))",
            "valid_complex_bin_count": 0,
            "invalid_or_zero_complex_bin_count": int(valid.numel()),
            "mean_signed_rad_float64": None,
            "mean_abs_rad_float64": None,
            "p95_abs_rad_float64": None,
            "max_abs_rad_float64": None,
        }
    actual64 = actual[valid].to(dtype=torch.complex128)
    reference64 = reference[valid].to(dtype=torch.complex128)
    signed = torch.angle(actual64 * torch.conj(reference64)).to(dtype=torch.float64)
    absolute = signed.abs()
    ordered = torch.sort(absolute).values
    p95_index = max(0, min(count - 1, math.ceil(0.95 * count) - 1))
    return {
        "comparison": "principal_angle(actual * conjugate(reference))",
        "valid_complex_bin_count": count,
        "invalid_or_zero_complex_bin_count": int(valid.numel() - count),
        "mean_signed_rad_float64": float(signed.mean().item()),
        "mean_abs_rad_float64": float(absolute.mean().item()),
        "p95_abs_rad_float64": float(ordered[p95_index].item()),
        "max_abs_rad_float64": float(absolute.max().item()),
    }


def _roll_seam_contract(control_name: str, frames: int) -> dict[str, Any]:
    if control_name.endswith("plus64"):
        destination_start, destination_end = 0, 64
        source_start, source_end = frames - 64, frames
        join_before = 64
        signed_frames = 64
        description = "first 64 output mask frames receive the source mask's last 64 frames"
    elif control_name.endswith("minus64"):
        destination_start, destination_end = frames - 64, frames
        source_start, source_end = 0, 64
        join_before = frames - 64
        signed_frames = -64
        description = "last 64 output mask frames receive the source mask's first 64 frames"
    else:
        return {
            "is_circular_roll": False,
            "signed_frames": 0,
            "signed_milliseconds": 0,
            "frame_count": frames,
            "seam_frame_count": 0,
            "wrap_destination_frame_range_half_open": None,
            "wrap_source_frame_range_half_open": None,
            "join_boundary_before_output_frame_index": None,
            "join_boundary_sample_index": None,
            "join_neighborhood_frame_range_half_open_approximate": None,
            "wrap_description": None,
            "causal_timing_claim": False,
            "scoring_frames_or_queries_removed": False,
            "limitation": "mean-mask control uses all original utterance frames and retains the full source clock; it is a distribution control, not an utterance crop",
        }
    neighborhood_start = max(0, join_before - 2)
    neighborhood_end = min(frames, join_before + 2)
    return {
        "is_circular_roll": True,
        "signed_frames": signed_frames,
        "signed_milliseconds": ROLL_MS if signed_frames > 0 else -ROLL_MS,
        "frame_count": frames,
        "seam_frame_count": 64,
        "wrap_destination_frame_range_half_open": [destination_start, destination_end],
        "wrap_source_frame_range_half_open": [source_start, source_end],
        "join_boundary_before_output_frame_index": join_before,
        "join_boundary_sample_index": join_before * EXPECTED_HOP_LENGTH,
        "join_neighborhood_frame_range_half_open_approximate": [neighborhood_start, neighborhood_end],
        "join_neighborhood_note": "Approximate nearby STFT centers within two 128-sample hops of the mask join; actual ISTFT overlap, centered reflect padding, and whole-signal RMS/baseline correction affect the realized waveform beyond this neighborhood.",
        "wrap_description": description,
        "causal_timing_claim": False,
        "scoring_frames_or_queries_removed": False,
        "limitation": "circular wrap creates a mask seam; centered reflect-padded STFT, ISTFT overlap, and whole-signal RMS/baseline correction affect the realized waveform. The seam neighborhood is descriptive, not a mask for excluding frames or score queries.",
    }


def _transition_support_summary(original: Mapping[str, Any], head: Any, torch: Any) -> dict[str, Any]:
    active_tensor = original.get("active_frame_mask")
    valid_tensor = original.get("valid_transition_mask")
    positive_flux = original.get("positive_relative_flux")
    negative_flux = original.get("negative_relative_flux")
    positive_gate = original.get("positive_flux_gate")
    negative_gate = original.get("negative_flux_gate")
    frames = int(original.get("source_stft").shape[1]) if original.get("source_stft") is not None else 0
    vectors = {
        "active_frame_mask": active_tensor,
        "valid_transition_mask": valid_tensor,
        "positive_relative_flux": positive_flux,
        "negative_relative_flux": negative_flux,
        "positive_flux_gate": positive_gate,
        "negative_flux_gate": negative_gate,
        "previous_frame_magnitude_sum": original.get("previous_frame_magnitude_sum"),
    }
    for label, vector in vectors.items():
        if vector is None or vector.ndim != 1 or int(vector.numel()) != frames:
            raise ControlError(f"actual head transition-support vector has invalid frame alignment: {label}")
    if active_tensor.dtype != torch.bool or valid_tensor.dtype != torch.bool:
        raise ControlError("actual active/valid support masks must be boolean tensors")

    active = [bool(value) for value in active_tensor.detach().cpu().tolist()]
    valid = [bool(value) for value in valid_tensor.detach().cpu().tolist()]
    previous = [float(value) for value in vectors["previous_frame_magnitude_sum"].detach().to(dtype=torch.float64).cpu().tolist()]
    reason_counts = {
        "first_frame_has_no_previous_transition": 0,
        "inactive_or_quiet_adjacent_frame": 0,
        "previous_magnitude_below_flux_denominator_floor": 0,
        "unclassified_invalid": 0,
        "valid_transition": 0,
    }
    reason_by_frame: list[str] = []
    floor = float(head.flux_denominator_floor)
    for index, is_valid in enumerate(valid):
        if is_valid:
            reason = "valid_transition"
        elif index == 0:
            reason = "first_frame_has_no_previous_transition"
        elif not (active[index - 1] and active[index]):
            reason = "inactive_or_quiet_adjacent_frame"
        elif previous[index] <= floor:
            reason = "previous_magnitude_below_flux_denominator_floor"
        else:
            reason = "unclassified_invalid"
        reason_by_frame.append(reason)
        reason_counts[reason] += 1

    def float_vector(value: Any) -> list[float]:
        return [float(item) for item in value.detach().to(dtype=torch.float64).cpu().tolist()]

    valid_indices = [index for index, flag in enumerate(valid) if flag]
    expected_valid = [False] + [
        active[index - 1] and active[index] and previous[index] > floor
        for index in range(1, frames)
    ] if frames else []
    if valid != expected_valid:
        raise ControlError("head valid-transition mask does not match active-pair and previous-denominator rules")
    return {
        "frame_index_axis": "source_stft_frame_index; frame centers are separately pinned in the STFT section",
        "active_frame_mask": active,
        "valid_transition_mask": valid,
        "valid_transition_mask_canonical_sha256": _sha(_canonical(valid)),
        "valid_transition_true_frame_indices": valid_indices,
        "valid_transition_count": len(valid_indices),
        "active_frame_count": sum(active),
        "invalid_transition_reason_by_frame": reason_by_frame,
        "invalid_transition_reason_counts": reason_counts,
        "positive_relative_flux_full_with_invalid_zero_placeholders": float_vector(positive_flux),
        "negative_relative_flux_full_with_invalid_zero_placeholders": float_vector(negative_flux),
        "positive_flux_gate_sequence_actual_mode": float_vector(positive_gate),
        "negative_flux_gate_sequence_actual_mode": float_vector(negative_gate),
        "gate_dc_support_note": "GateDC averages the dynamic gate only over valid transitions and reapplies the valid mask. The first frame and quiet/invalid transitions are therefore unsupported acoustic transitions; later static-mask terms and mask smoothing can still produce a nonzero post-projection mask there.",
        "zero_flux_values_on_invalid_frames_are_placeholders": True,
        "flux_quantification_scope": "Only valid_transition_mask=true entries describe acoustic transitions; invalid entries are finite zero placeholders, not measured zero transitions.",
        "source_stft_frame_count": frames,
        "full_source_frame_support_retained": True,
    }


def _frame_rms(samples: Any, samples_per_frame: int, torch: Any) -> dict[str, Any]:
    complete = samples.numel() // samples_per_frame
    used = complete * samples_per_frame
    if complete:
        rms = samples[:used].reshape(complete, samples_per_frame).square().mean(dim=1).sqrt()
        rms_cpu = rms.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().astype("<f4", copy=False)
        return {
            "frame_count": int(complete),
            "trailing_unframed_samples": int(samples.numel() - used),
            "frame_rms_float32_sha256": _sha(rms_cpu.tobytes()),
            "frame_rms_min": float(rms.min().item()),
            "frame_rms_max": float(rms.max().item()),
            "frame_rms_mean": float(rms.mean().item()),
        }
    return {"frame_count": 0, "trailing_unframed_samples": int(samples.numel()), "frame_rms_float32_sha256": _sha(b"")}


def _complex64_sha(value: Any, torch: Any) -> str:
    if not torch.is_complex(value):
        raise ControlError("STFT diagnostic input must be complex")
    pair = torch.view_as_real(value.detach().to(device="cpu")).contiguous().numpy()
    return _sha(pair.astype("<f4", copy=False).tobytes(order="C"))


def _reconstruct_from_projected_mask(
    head: Any,
    source: Any,
    baseline: Any,
    source_spectrum: Any,
    mask_db: Any,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    window = head.stft_window.to(device=source.device, dtype=source.dtype)
    gain = torch.pow(source.new_tensor(10.0), mask_db / 20.0)
    delta_spectrum = source_spectrum * (gain - 1.0)
    delta_pcm = torch.istft(
        delta_spectrum,
        n_fft=head.n_fft,
        hop_length=head.hop_length,
        win_length=head.win_length,
        window=window,
        center=head.stft_center,
        normalized=False,
        onesided=True,
        length=source.numel(),
        return_complex=False,
    )
    raw_candidate = source + delta_pcm
    normalized_candidate, candidate_norm = head._rms_peak_baseline(raw_candidate)
    normalized_source, source_norm = head._rms_peak_baseline(source)
    baseline_match_error = (baseline - normalized_source).abs().amax()
    if float(baseline_match_error.item()) > float(head.baseline_match_atol):
        raise ControlError("supplied canonical RMS baseline fails the head's frozen match tolerance")
    output = baseline + (normalized_candidate - normalized_source)
    if output.shape != source.shape or not bool(torch.isfinite(output).all().item()):
        raise ControlError("post-projection reconstruction violated full-length finite PCM")
    return output, {
        "candidate_rms_before_target_normalization": float(candidate_norm["input_rms"].item()),
        "candidate_rms_after_target_normalization": float(candidate_norm["output_rms"].item()),
        "source_rms_after_target_normalization": float(source_norm["output_rms"].item()),
        "baseline_match_max_abs_error": float(baseline_match_error.item()),
        "target_rms": float(head.target_rms),
        "peak_ceiling": head.peak_ceiling,
        "candidate_peak_limited": bool(candidate_norm["peak_limited"].item()),
    }


def _realized_stft_summary(head: Any, output: Any, source_spectrum: Any, mask_db: Any, torch: Any) -> dict[str, Any]:
    output_spectrum = torch.stft(
        output,
        n_fft=head.n_fft,
        hop_length=head.hop_length,
        win_length=head.win_length,
        window=head.stft_window.to(device=output.device, dtype=output.dtype),
        center=head.stft_center,
        pad_mode=head.stft_pad_mode,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    if output_spectrum.shape != source_spectrum.shape:
        raise ControlError("reconstructed output STFT changed frequency/frame dimensions")
    expected_gain = torch.pow(output.new_tensor(10.0), mask_db / 20.0)
    # This is the analytically constructed masked STFT. A positive real gain
    # changes magnitude only; the subsequent ISTFT, residual RMS normalization,
    # baseline addition, and reanalysis can produce a different realized phase.
    constructed_masked_spectrum = source_spectrum * expected_gain
    source_abs = source_spectrum.abs()
    output_abs = output_spectrum.abs()
    valid = source_abs > torch.finfo(output.dtype).tiny
    realized_db = torch.zeros_like(source_abs)
    realized_db[valid] = 20.0 * torch.log10(output_abs[valid] / source_abs[valid])
    requested_db = mask_db
    db_error = realized_db[valid] - requested_db[valid]
    expected_abs = source_abs * expected_gain
    absolute_error = (output_abs - expected_abs).abs()
    relative_error = absolute_error / expected_abs.clamp_min(torch.finfo(output.dtype).tiny)
    return {
        "shape_frequency_by_time": list(output_spectrum.shape),
        "constructed_masked_stft_complex64_sha256": _complex64_sha(constructed_masked_spectrum, torch),
        "realized_output_stft_complex64_sha256": _complex64_sha(output_spectrum, torch),
        "source_stft_complex64_sha256": _complex64_sha(source_spectrum, torch),
        "phase_semantics": {
            "constructed_mask_times_source_phase": "positive real gain is applied directly to source STFT; phase should be unchanged except floating-point representation",
            "realized_output_phase": "computed by reanalyzing the synthesized, residual-RMS-normalized full-length output PCM; not assumed equal to constructed mask phase",
            "constructed_vs_source": _phase_delta_summary(constructed_masked_spectrum, source_spectrum, torch),
            "realized_reanalysis_vs_constructed": _phase_delta_summary(output_spectrum, constructed_masked_spectrum, torch),
            "realized_reanalysis_vs_source": _phase_delta_summary(output_spectrum, source_spectrum, torch),
        },
        "realized_output_stft_abs": _tensor_summary(output_abs),
        "requested_mask_db_on_nonzero_source_bins": _tensor_summary(requested_db[valid]) if bool(valid.any().item()) else None,
        "realized_log_magnitude_change_db_on_nonzero_source_bins": _tensor_summary(realized_db[valid]) if bool(valid.any().item()) else None,
        "realized_minus_requested_mask_db": _tensor_summary(db_error) if db_error.numel() else None,
        "magnitude_difference_from_unscaled_projected_stft": _tensor_summary(absolute_error),
        "relative_magnitude_difference_from_unscaled_projected_stft": _tensor_summary(relative_error),
        "zero_source_magnitude_bin_count": int((~valid).sum().item()),
    }


def _control_masks(projected_mask: Any, torch: Any) -> dict[str, Any]:
    if projected_mask.ndim != 2 or list(projected_mask.shape[:1]) != [EXPECTED_N_FFT // 2 + 1]:
        raise ControlError("projected mask must have exactly 257 one-sided frequency bins")
    frames = int(projected_mask.shape[1])
    if frames <= 64:
        raise ControlError("utterance has too few STFT frames for the frozen ±64-frame roll")
    # Keep the mask tensor in its original dtype, but accumulate its full-time
    # frequency-bin mean in float64 before casting the resulting control back.
    per_bin_mean = projected_mask.to(dtype=torch.float64).mean(dim=1, keepdim=True).to(dtype=projected_mask.dtype)
    return {
        "post_proj_mean": per_bin_mean.expand_as(projected_mask),
        "post_proj_roll_plus64": torch.roll(projected_mask, shifts=64, dims=1),
        "post_proj_roll_minus64": torch.roll(projected_mask, shifts=-64, dims=1),
    }


def run_fit40_post_projection_controls(
    head: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    actual_completion_pin: Mapping[str, Any],
    actual_completion_raw: bytes,
    actual_checkpoint: Mapping[str, Any],
    actual_checkpoint_raw: bytes,
    checkpoint_parameter: Mapping[str, Any],
    expected_checkpoint_parameter_sha256: str,
    actual_training_record_raw: bytes,
    expected_row_ids: Sequence[str],
    actual_runtime_source_pins_raw: bytes,
    actual_head_config: Mapping[str, Any],
    actual_head_config_raw: bytes,
    actual_head_constructor: Mapping[str, Any],
    actual_runtime_identity: Mapping[str, Any],
    expected_original_gate_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    actual_gate_ledger_raw: bytes,
    actual_completion: Mapping[str, Any],
    actual_training_record: Mapping[str, Any],
    actual_runtime_source_pins: Mapping[str, Any],
    actual_native_geometry_manifest: Mapping[str, Any],
    actual_native_geometry_manifest_raw: bytes,
    source_lineage_pins: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build no-update controls from actual post-projection masks for 40 Fit rows.

    Each ``rows`` entry supplies ``row_id``, ``sample_id``, ``speaker_id``,
    ``source_pcm_float32`` and ``canonical_rms_baseline_pcm_float32`` tensors,
    actual ``source_file_pin`` and ``baseline_file_pin`` objects, and the
    canonical source/baseline PCM hashes and sample count. The caller also
    supplies the exact raw completion/checkpoint/training/runtime/config bytes
    and their actual whole-file pins. ``expected_original_gate_rows``
    maps ``(row_id, gate_mode)`` to the frozen runner's 80-row GateDC ledger,
    including its ``candidate_waveform_float32_sha256``.

    Returns small JSON-compatible records and the actual control waveforms as
    an in-memory mapping. Callers must persist the receipt/audio atomically
    under a fresh directory and bind it to the actual terminal lineage; this
    function performs no file writes and no scoring.
    """
    import torch

    config = _assert_pinned_head(head, actual_head_constructor)
    if len(rows) != 40 or len(expected_row_ids) != 40 or len(set(expected_row_ids)) != 40:
        raise ControlError("post-projection controls require the exact ordered 40 Fit rows")
    observed_ids = [str(item.get("row_id")) for item in rows]
    expected_ids = [str(value) for value in expected_row_ids]
    if observed_ids != expected_ids:
        raise ControlError("post-projection inputs do not match the frozen Fit40 row order")
    runtime_fit_rows = actual_runtime_source_pins.get("fit_rows")
    geometry_rows = actual_native_geometry_manifest.get("rows")
    if (
        not isinstance(runtime_fit_rows, list) or len(runtime_fit_rows) != 40
        or not isinstance(geometry_rows, list) or len(geometry_rows) != 40
    ):
        raise ControlError("actual runtime/geometry artifacts do not contain exactly 40 rows")
    runtime_by_id = {str(item.get("row_id")): item for item in runtime_fit_rows if isinstance(item, dict)}
    geometry_by_id = {str(item.get("row_id")): item for item in geometry_rows if isinstance(item, dict)}
    if (
        len(runtime_by_id) != 40 or len(geometry_by_id) != 40
        or set(runtime_by_id) != set(expected_ids) or set(geometry_by_id) != set(expected_ids)
        or [str(item.get("row_id")) for item in runtime_fit_rows] != expected_ids
    ):
        raise ControlError("actual runtime/geometry row identities/order differ from Fit40")
    actual_pins = _validate_fit_lineage(
        actual_completion=actual_completion,
        actual_completion_raw=actual_completion_raw,
        actual_checkpoint=actual_checkpoint,
        actual_checkpoint_raw=actual_checkpoint_raw,
        actual_training_record=actual_training_record,
        actual_training_record_raw=actual_training_record_raw,
        actual_runtime_source_pins=actual_runtime_source_pins,
        actual_runtime_source_pins_raw=actual_runtime_source_pins_raw,
        actual_native_geometry_manifest=actual_native_geometry_manifest,
        actual_native_geometry_manifest_raw=actual_native_geometry_manifest_raw,
        actual_head_config=actual_head_config,
        actual_head_config_raw=actual_head_config_raw,
        actual_head_constructor=actual_head_constructor,
        expected_checkpoint_parameter_sha256=expected_checkpoint_parameter_sha256,
        source_lineage_pins=source_lineage_pins,
    )
    if dict(checkpoint_parameter) != dict(actual_checkpoint.get("parameter", {})):
        raise ControlError("loaded checkpoint parameter differs from its actual checkpoint JSON")
    _assert_raw_pin(actual_completion_raw, actual_completion_pin, label="actual_completion_pin")
    _require_pin_equal(actual_completion_pin, actual_pins["completion"], label="actual completion pin")
    if _canonical(dict(actual_runtime_identity)) != _canonical(actual_completion.get("gpu_identity")):
        raise ControlError("post-projection runtime GPU identity differs from the actual training completion")
    runtime_device = str(actual_runtime_identity.get("runtime_device", ""))
    if str(head.band_contrasts_db.device) != runtime_device:
        raise ControlError("loaded head is not on the exact CUDA device used by Fit40")
    if not bool(torch.cuda.is_initialized()):
        raise ControlError("the actual Fit40 CUDA device is not initialized in this replay process")
    if torch.get_num_threads() != int(actual_runtime_identity.get("torch_num_threads", -1)):
        raise ControlError("PyTorch intra-op thread count differs from the actual Fit40 runtime")
    if torch.get_num_interop_threads() != int(actual_runtime_identity.get("torch_num_interop_threads", -1)):
        raise ControlError("PyTorch inter-op thread count differs from the actual Fit40 runtime")
    _assert_raw_pin(actual_gate_ledger_raw, actual_pins["fit40_gate_dc_control"], label="actual GateDC ledger")
    parsed_gate_rows = []
    for line_number, line in enumerate(actual_gate_ledger_raw.splitlines(), start=1):
        if not line:
            raise ControlError(f"GateDC ledger contains an empty line at {line_number}")
        try:
            entry = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ControlError(f"GateDC ledger is not valid JSONL at line {line_number}") from exc
        if not isinstance(entry, dict):
            raise ControlError(f"GateDC ledger line {line_number} is not an object")
        parsed_gate_rows.append(entry)
    if len(parsed_gate_rows) != 80:
        raise ControlError("actual GateDC ledger does not contain exactly 80 branch rows")
    for (row_id, mode), entry in expected_original_gate_rows.items():
        if entry not in parsed_gate_rows:
            raise ControlError(f"GateDC lookup row is not present in the raw ledger: {row_id}/{mode}")
    if set(expected_original_gate_rows) != {(row_id, mode) for row_id in expected_ids for mode in ("dynamic", "gate_dc")}:
        raise ControlError("actual runner ledger must contain exactly dynamic and GateDC records for all 40 rows")
    actual_theta_payload = _parameter_payload(head, torch)
    if dict(checkpoint_parameter) != actual_theta_payload:
        raise ControlError("loaded frozen head parameter differs from the actual checkpoint parameter")
    theta_sha = _sha(_canonical(actual_theta_payload))
    if theta_sha != expected_checkpoint_parameter_sha256:
        raise ControlError("checkpoint parameter canonical SHA differs from its actual completion binding")
    theta_before = theta_sha
    records: list[dict[str, Any]] = []
    outputs: dict[str, Any] = {}
    support_by_row_branch: dict[str, dict[str, Any]] = {}
    fps_num, fps_den = EXPECTED_FPS_NUM, EXPECTED_FPS_DEN
    samples_per_video_frame = EXPECTED_SAMPLE_RATE_HZ * fps_den // fps_num
    shift_frames = (EXPECTED_SAMPLE_RATE_HZ * ROLL_MS) // (1000 * EXPECTED_HOP_LENGTH)

    with torch.no_grad():
        for row in rows:
            row_id = str(row["row_id"])
            source = row["source_pcm_float32"]
            baseline = row["canonical_rms_baseline_pcm_float32"]
            if source.ndim != 1 or baseline.shape != source.shape or source.dtype != torch.float32 or baseline.dtype != torch.float32 or source.device != baseline.device:
                raise ControlError(f"source/baseline tensor contract differs for {row_id}")
            sample_count = int(source.numel())
            if row.get("source_sample_count") != sample_count or sample_count <= 0:
                raise ControlError(f"source sample count differs from actual row metadata: {row_id}")
            source_sha = _sha(_tensor_f32_bytes(source, torch))
            baseline_sha = _sha(_tensor_f32_bytes(baseline, torch))
            source_file_pin = _pin_shape(row.get("source_file_pin"), label=f"rows[{row_id}].source_file_pin")
            baseline_file_pin = _pin_shape(row.get("baseline_file_pin"), label=f"rows[{row_id}].baseline_file_pin")
            _assert_raw_pin(row.get("source_file_raw_bytes", b""), source_file_pin, label=f"source WAV {row_id}")
            _assert_raw_pin(row.get("baseline_file_raw_bytes", b""), baseline_file_pin, label=f"baseline WAV {row_id}")
            runtime_row = runtime_by_id[row_id]
            natural = runtime_row.get("natural_pcm")
            geometry_row = geometry_by_id[row_id]
            baseline_record = geometry_row.get("rms_fixed_baseline")
            if not isinstance(natural, dict) or not isinstance(baseline_record, dict):
                raise ControlError(f"actual runtime/geometry row lineage is incomplete: {row_id}")
            _require_pin_equal(
                source_file_pin,
                {"path": natural.get("path"), "bytes": source_file_pin["bytes"], "sha256": natural.get("file_sha256")},
                label=f"runtime source file {row_id}",
            )
            _require_pin_equal(
                baseline_file_pin, baseline_record.get("rms_input_float32_wav"),
                label=f"geometry baseline file {row_id}",
            )
            if source_sha != str(row.get("source_pcm_float32_sha256", "")).lower():
                raise ControlError(f"original source PCM hash mismatch: {row_id}")
            if baseline_sha != str(row.get("canonical_rms_baseline_pcm_float32_sha256", "")).lower():
                raise ControlError(f"canonical RMS baseline PCM hash mismatch: {row_id}")
            if (
                row.get("sample_id") != runtime_row.get("sample_id")
                or row.get("speaker_id") != runtime_row.get("speaker_id")
                or row.get("source_sample_count") != natural.get("sample_count")
                or source_sha != natural.get("pcm_sha256")
                or baseline_sha != baseline_record.get("output_float32_pcm_sha256")
                or baseline_record.get("method_id") != "rms_fixed"
                or baseline_record.get("peak_ceiling") is not None
                or baseline_record.get("audio_time_warped") is not False
            ):
                raise ControlError(f"source PCM or canonical baseline differs from actual Fit40 lineage: {row_id}")

            for mode in ("dynamic", "gate_dc"):
                original = head(source, baseline, gate_mode=mode)
                if original.get("gate_mode") != mode or int(original.get("source_sample_count", -1)) != sample_count:
                    raise ControlError(f"original head replay returned wrong mode/clock: {row_id}/{mode}")
                mask = original.get("projected_log_magnitude_mask_db")
                source_spectrum = original.get("source_stft")
                if mask is None or source_spectrum is None or list(mask.shape) != [257, source_spectrum.shape[1]]:
                    raise ControlError(f"actual projected mask/STFT dimensions are invalid: {row_id}/{mode}")
                if not bool(torch.isfinite(mask).all().item()) or not bool(torch.isfinite(source_spectrum.real).all().item()) or not bool(torch.isfinite(source_spectrum.imag).all().item()):
                    raise ControlError(f"actual projected mask/STFT contains nonfinite values: {row_id}/{mode}")
                original_max = float(mask.abs().max().item())
                if original_max > EXPECTED_MASK_LIMIT_DB + 1e-6:
                    raise ControlError(f"actual post-projection mask exceeds the frozen 3 dB radial bound: {row_id}/{mode}")
                gate_record = expected_original_gate_rows[(row_id, mode)]
                if (
                    gate_record.get("row_id") != row_id
                    or gate_record.get("gate_mode") != mode
                    or gate_record.get("same_frozen_parameter_sha256") != theta_sha
                    or gate_record.get("source_pcm_float32_sha256") != source_sha
                    or gate_record.get("source_sample_count") != sample_count
                    or gate_record.get("gate_dc_control_is_training_objective") is not False
                ):
                    raise ControlError(f"runner GateDC ledger row identity mismatch: {row_id}/{mode}")
                expected_output_sha = gate_record.get("candidate_waveform_float32_sha256")
                if not isinstance(expected_output_sha, str) or len(expected_output_sha) != 64:
                    raise ControlError(f"runner lacks its original waveform hash: {row_id}/{mode}")
                original_output = original.get("output_pcm_float32")
                if _sha(_tensor_f32_bytes(original_output, torch)) != expected_output_sha:
                    raise ControlError(f"frozen-theta original output replay differs from actual Fit40 record: {row_id}/{mode}")

                support_key = f"{row_id}::{mode}"
                support_by_row_branch[support_key] = _transition_support_summary(original, head, torch)

                base_rebuilt, base_rebuild_norm = _reconstruct_from_projected_mask(head, source, baseline, source_spectrum, mask, torch)
                if not torch.equal(base_rebuilt, original_output):
                    raise ControlError(f"post-projection formula does not exactly reconstruct the original branch: {row_id}/{mode}")

                control_masks = _control_masks(mask, torch)
                original_sorted = torch.sort(mask, dim=1).values
                original_bin_stats = _per_bin_stats(mask, torch)
                for control_name in CONTROL_NAMES:
                    control_mask = control_masks[control_name]
                    if not bool(torch.isfinite(control_mask).all().item()):
                        raise ControlError(f"post-projection control mask is nonfinite: {row_id}/{mode}/{control_name}")
                    control_max = float(control_mask.abs().max().item())
                    if control_max > EXPECTED_MASK_LIMIT_DB + 1e-6:
                        raise ControlError(f"post-projection control exceeds the unchanged 3 dB bound: {row_id}/{mode}/{control_name}")
                    if control_name.startswith("post_proj_roll_"):
                        control_sorted = torch.sort(control_mask, dim=1).values
                        distribution_equal_per_bin = [
                            bool(value)
                            for value in (original_sorted == control_sorted).all(dim=1).detach().cpu().tolist()
                        ]
                        dist_equal = all(distribution_equal_per_bin)
                        mean_comparison = _per_bin_mean_comparison(
                            mask, control_mask, torch, tolerance_db=ROLL_MEAN_ABS_TOL_DB
                        )
                        mean_error = mean_comparison["max_abs_error_db"]
                        if not dist_equal or not mean_comparison["all_bins_within_tolerance"]:
                            raise ControlError(f"circular roll did not preserve every per-bin mask distribution: {row_id}/{mode}/{control_name}")
                        exact_mean_bin_count = sum(error == 0.0 for error in mean_comparison["per_bin_abs_error_db"])
                    else:
                        per_bin_target = mask.to(dtype=torch.float64).mean(dim=1, keepdim=True).to(dtype=mask.dtype)
                        exact_broadcast = torch.equal(control_mask, per_bin_target.expand_as(mask))
                        mean_comparison = _per_bin_mean_comparison(
                            mask, control_mask, torch, tolerance_db=MEAN_CONTROL_ABS_TOL_DB
                        )
                        mean_error = mean_comparison["max_abs_error_db"]
                        if not exact_broadcast or not mean_comparison["all_bins_within_tolerance"]:
                            raise ControlError(f"post-projection mean control differs from exact per-bin full-time mean: {row_id}")
                        dist_equal = False
                        distribution_equal_per_bin = None
                        exact_mean_bin_count = sum(error == 0.0 for error in mean_comparison["per_bin_abs_error_db"])

                    seam_contract = _roll_seam_contract(control_name, int(mask.shape[1]))

                    output, norm = _reconstruct_from_projected_mask(head, source, baseline, source_spectrum, control_mask, torch)
                    output_bytes = _tensor_f32_bytes(output, torch)
                    output_sha = _sha(output_bytes)
                    output_spectrum_summary = _realized_stft_summary(head, output, source_spectrum, control_mask, torch)
                    rms = float(torch.sqrt(output.square().mean()).item())
                    source_rms = float(torch.sqrt(source.square().mean()).item())
                    baseline_rms = float(torch.sqrt(baseline.square().mean()).item())
                    stft_frames = int(source_spectrum.shape[1])
                    if stft_frames != sample_count // EXPECTED_HOP_LENGTH + 1:
                        raise ControlError(f"centered STFT frame count differs from source clock: {row_id}")
                    control_id = f"{row_id}::{mode}::{control_name}"
                    outputs[control_id] = output.detach()
                    records.append({
                        "schema": SCHEMA,
                        "row_id": row_id,
                        "sample_id": str(row.get("sample_id")),
                        "speaker_id": str(row.get("speaker_id")),
                        "base_gate_mode": mode,
                        "control": control_name,
                        "transition_support_key": support_key,
                        "same_frozen_parameter_sha256": theta_sha,
                        "input_source_pcm_float32_sha256": source_sha,
                        "source_audio_file_pin": source_file_pin,
                        "canonical_rms_baseline_pcm_float32_sha256": baseline_sha,
                        "canonical_rms_baseline_file_pin": baseline_file_pin,
                        "original_gate_record_waveform_sha256": expected_output_sha,
                        "replayed_original_waveform_sha256": _sha(_tensor_f32_bytes(original_output, torch)),
                        "base_projection_reconstruction_exact": True,
                        "source_sample_rate_hz": EXPECTED_SAMPLE_RATE_HZ,
                        "source_sample_count": sample_count,
                        "source_clock_video_fps": {"numerator": fps_num, "denominator": fps_den},
                        "source_clock_video_frame_count_floor": sample_count // samples_per_video_frame,
                        "source_clock_trailing_unframed_samples": sample_count % samples_per_video_frame,
                        "stft": {
                            "n_fft": head.n_fft,
                            "hop_length": head.hop_length,
                            "win_length": head.win_length,
                            "window_kind": head.window_kind,
                            "window_periodic": head.window_periodic,
                            "center": head.stft_center,
                            "pad_mode": head.stft_pad_mode,
                            "one_sided_bin_count": int(mask.shape[0]),
                            "frame_count": stft_frames,
                            "frame_center_first_sample": 0,
                            "frame_center_last_sample": (stft_frames - 1) * head.hop_length,
                            "source_stft_complex64_sha256": _complex64_sha(source_spectrum, torch),
                            **output_spectrum_summary,
                        },
                        "post_projection_mask": {
                            "projection_scale_from_original": float(original["smooth_projection_scale"].item()),
                            "global_max_abs_db_original": original_max,
                        "global_max_abs_db_control": control_max,
                        "frozen_bound_db": EXPECTED_MASK_LIMIT_DB,
                        "original_smooth_global_radial_projection_scale": float(original["smooth_projection_scale"].item()),
                        "gate_dc_is_a_functional_ablation_not_a_pure_timing_control": True,
                        "reprojection_applied": False,
                            "gradient_enabled": False,
                            "frequency_bin_count": int(mask.shape[0]),
                            "time_frame_count": int(mask.shape[1]),
                            "per_bin_statistics_original": original_bin_stats,
                            "per_bin_statistics_control": _per_bin_stats(control_mask, torch),
                            "per_bin_mean_comparison_float64": mean_comparison,
                            "distribution_exactly_preserved_all_bins": bool(dist_equal),
                            "distribution_exactly_preserved_bin_count": (
                                sum(distribution_equal_per_bin) if distribution_equal_per_bin is not None else None
                            ),
                            "sorted_value_multiset_equal_per_bin": distribution_equal_per_bin,
                            "mean_close_all_bins": mean_comparison["all_bins_within_tolerance"],
                            "mean_close_bin_count": mean_comparison["bins_within_tolerance"],
                            "exact_mean_bin_count": exact_mean_bin_count,
                            "max_per_bin_mean_abs_error_db": mean_error,
                            "mean_reduction_dtype": "float64",
                            "mean_bit_exact_claim": False,
                            "mask_float32_sha256": _sha(_tensor_f32_bytes(control_mask, torch)),
                        },
                        "roll_contract": seam_contract,
                        "pcm": {
                            "output_pcm_float32_sha256": output_sha,
                            "output_sample_count": int(output.numel()),
                            "output_exact_full_source_length": output.numel() == sample_count,
                            "source_rms": source_rms,
                            "canonical_baseline_rms": baseline_rms,
                            "control_output_rms": rms,
                            "target_rms": float(head.target_rms),
                            "output_rms_error_to_target": rms - float(head.target_rms),
                            "output_peak_abs": float(output.abs().max().item()),
                            "frame_rms": _frame_rms(output, samples_per_video_frame, torch),
                            **norm,
                        },
                        "input_pins": actual_pins,
                        "scoring_or_rendering_performed": False,
                        "full_fit40_source_rows_preserved": True,
                        "score_queries_filtered_for_seam": False,
                        "optimizer_created_or_stepped": False,
                        "head_source_modified": False,
                        "checkpoint_selected": False,
                    })

    if _parameter_sha(head, torch) != theta_before:
        raise ControlError("head parameters changed during no-update post-projection diagnostics")
    if len(records) != 40 * 2 * len(CONTROL_NAMES) or len(outputs) != len(records):
        raise ControlError("post-projection control denominator is incomplete")
    return {
        "schema": SCHEMA,
        "status": "BUILT_NO_UPDATE_POST_PROJECTION_CONTROLS_PENDING_READBACK",
        "qualification_status": "DIAGNOSTIC_ONLY_NOT_A_QUALIFICATION_RESULT",
        "row_count": 40,
        "base_branches": ["dynamic", "gate_dc"],
        "controls_per_branch": list(CONTROL_NAMES),
        "record_count": len(records),
        "output_count": len(outputs),
        "frozen_parameter_sha256": theta_sha,
        "head_configuration_sha256": _sha(_canonical(config)),
        "input_lineage_pins": actual_pins,
        "no_reprojection": True,
        "no_gradients": True,
        "no_optimizer": True,
        "no_render_or_score": True,
        "full_fit40_source_rows_preserved": True,
        "score_query_filtering_for_seams": False,
        "records": records,
        "transition_support_by_row_branch": support_by_row_branch,
        "outputs_float32_tensors": outputs,
        "interpretation_limit": "post-projection controls distinguish temporal distribution from the actual global radial projection; GateDC remains a functional ablation because the global projection couples all mask cells. Circular rolls have wrap seams and centered-STFT edge interaction; no timing-only or causal claim is made.",
    }
