"""Source-only definitions for frozen-theta post-projection Fit40 controls.

This module does not load a checkpoint, create an optimizer, score/render a
video, or execute any control at import time.  A future caller must supply the
already-loaded, frozen head plus source/baseline tensors and actual completion
lineage.  The helper replays only the existing head forward to retrieve its
actual globally projected mask, then applies controls to that mask without
reprojection or gradients.

The circular rolls deliberately wrap 64 STFT frames (512 ms at the pinned
16 kHz / 128-sample hop).  Their wrap seam and reflected-STFT edge interaction
are recorded; these controls are not pure timing interventions and do not
establish causality before later controlled analysis.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA = "cross_tfg_fit40_post_projection_control_diagnostic_v1"
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
    # One row for each of the 257 one-sided bins; std uses population
    # normalization so its definition is explicit and stable.
    return [
        {
            "bin_index": index,
            "mean_db": float(mask[index].mean().item()),
            "min_db": float(mask[index].min().item()),
            "max_db": float(mask[index].max().item()),
            "std_population_db": float(mask[index].std(unbiased=False).item()),
        }
        for index in range(mask.shape[0])
    ]


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
        "realized_output_stft_complex64_sha256": _complex64_sha(output_spectrum, torch),
        "source_stft_complex64_sha256": _complex64_sha(source_spectrum, torch),
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
    per_bin_mean = projected_mask.mean(dim=1, keepdim=True)
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
                        dist_equal = torch.equal(original_sorted, torch.sort(control_mask, dim=1).values)
                        original_means = mask.mean(dim=1)
                        control_means = control_mask.mean(dim=1)
                        mean_error = float((original_means - control_means).abs().max().item())
                        if not dist_equal or mean_error > 1e-7:
                            raise ControlError(f"circular roll did not preserve every per-bin mask distribution: {row_id}/{mode}/{control_name}")
                        bin_equal_count = int((original_means == control_means).sum().item())
                        wrap = "first_64_frames_receive_original_tail" if control_name.endswith("plus64") else "last_64_frames_receive_original_head"
                    else:
                        per_bin_target = mask.mean(dim=1, keepdim=True)
                        exact_broadcast = torch.equal(control_mask, per_bin_target.expand_as(mask))
                        control_means = control_mask.mean(dim=1)
                        mean_error = float((mask.mean(dim=1) - control_means).abs().max().item())
                        if not exact_broadcast or mean_error > 1e-7:
                            raise ControlError(f"post-projection mean control differs from exact per-bin full-time mean: {row_id}")
                        dist_equal = False
                        bin_equal_count = int((mask.mean(dim=1) == control_means).sum().item())
                        wrap = None

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
                            "distribution_exactly_preserved_all_bins": bool(dist_equal),
                            "distribution_exactly_preserved_bin_count": int(mask.shape[0]) if dist_equal else None,
                            "mean_close_all_bins": mean_error <= 1e-7,
                            "mean_close_bin_count": bin_equal_count,
                            "max_per_bin_mean_abs_error_db": mean_error,
                            "mask_float32_sha256": _sha(_tensor_f32_bytes(control_mask, torch)),
                        },
                        "roll_contract": {
                            "is_circular_roll": control_name.startswith("post_proj_roll_"),
                            "signed_frames": 64 if control_name.endswith("plus64") else -64 if control_name.endswith("minus64") else 0,
                            "signed_milliseconds": ROLL_MS if control_name.endswith("plus64") else -ROLL_MS if control_name.endswith("minus64") else 0,
                            "wrap_seam": wrap,
                            "seam_frame_count": 64 if wrap else 0,
                            "causal_timing_claim": False,
                            "limitation": "circular wrap creates a seam; centered reflect-padded STFT interacts with utterance edges; this is a post-projection functional control, not a pure timing intervention",
                        },
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
        "records": records,
        "outputs_float32_tensors": outputs,
        "interpretation_limit": "post-projection controls distinguish temporal distribution from the actual global radial projection; GateDC remains a functional ablation because the global projection couples all mask cells. Circular rolls have wrap seams and centered-STFT edge interaction; no timing-only or causal claim is made.",
    }
