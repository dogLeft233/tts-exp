"""Bounded CPU engineering check of the V3 head's zero-parameter gradient.

This worker uses only the pinned synthetic waveform in its qualification
config. It does not read fit audio, persist waveform samples, initialize a
frontend/model, train/select parameters, generate video, or score SyncNet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Mapping


WORKER_SCHEMA = "cross_tfg_v3_zero_theta_gradient_supplement_worker_v1"
RESULT_SCHEMA = "cross_tfg_v3_zero_theta_gradient_supplement_result_v1"
MANIFEST_SCHEMA = "cross_tfg_v3_zero_theta_gradient_supplement_manifest_v1"


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _write_new(path: Path, value: bytes) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _pin_sources(repo_root: Path, config: Mapping[str, Any], worker_path: Path) -> dict[str, dict[str, Any]]:
    actual: dict[str, dict[str, Any]] = {}
    pins = config["source_pins"]
    for name, pin in pins.items():
        rel = Path(str(pin["path"]))
        path = (repo_root / rel).resolve(strict=True)
        try:
            path.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(f"source pin escapes repository: {name}") from exc
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"pinned source is not a regular non-symlink file: {name}")
        digest, size = _sha256_file(path)
        if digest != pin["sha256"] or size != int(pin["bytes"]):
            raise ValueError(f"pinned source mismatch: {name}")
        actual[name] = {"path": str(rel.as_posix()), "sha256": digest, "bytes": size}
    worker_pin = pins["zero_theta_worker"]
    if worker_path.resolve(strict=True) != (repo_root / worker_pin["path"]).resolve(strict=True):
        raise ValueError("executed worker path differs from its configured source pin")
    return actual


def _finite_json(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, (float, int)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _probe_waveform(probe: Mapping[str, Any], torch: Any) -> Any:
    sample_rate = int(probe["sample_rate_hz"])
    count = int(probe["sample_count"])
    time_axis = torch.arange(count, dtype=torch.float32, device="cpu") / float(sample_rate)
    waveform = torch.zeros(count, dtype=torch.float32, device="cpu")
    for component in probe["components"]:
        phase = float(component["envelope_phase_radians"])
        envelope = float(component["envelope_floor"]) + float(component["envelope_depth"]) * (
            0.5 + 0.5 * torch.sin(2.0 * math.pi * float(component["envelope_hz"]) * time_axis + phase)
        )
        step = component["amplitude_step"]
        if step is not None:
            envelope = envelope * torch.where(
                time_axis >= float(step["time_seconds"]),
                torch.full_like(time_axis, float(step["after_multiplier"])),
                torch.full_like(time_axis, float(step["before_multiplier"])),
            )
        carrier = torch.sin(
            2.0 * math.pi * float(component["carrier_hz"]) * time_axis
            + float(component["carrier_phase_radians"])
        )
        waveform = waveform + float(component["amplitude"]) * envelope * carrier
    return waveform.contiguous()


def _probe_weights(probe: Mapping[str, Any], count: int, torch: Any) -> tuple[Any, Any]:
    sample_rate = int(probe["sample_rate_hz"])
    time_axis = torch.arange(count, dtype=torch.float32, device="cpu") / float(sample_rate)
    objective = probe["objective"]
    linear_weight = torch.zeros_like(time_axis)
    for term in objective["linear_terms"]:
        linear_weight = linear_weight + float(term["coefficient"]) * torch.sin(
            2.0 * math.pi * float(term["frequency_hz"]) * time_axis + float(term["phase_radians"])
        )
    quadratic_weight = torch.full_like(time_axis, float(objective["quadratic_weight_offset"]))
    for term in objective["quadratic_weight_terms"]:
        quadratic_weight = quadratic_weight + float(term["coefficient"]) * torch.cos(
            2.0 * math.pi * float(term["frequency_hz"]) * time_axis + float(term["phase_radians"])
        )
    return linear_weight.contiguous(), quadratic_weight.contiguous()


def _set_theta(head: Any, values: Any, torch: Any) -> None:
    with torch.no_grad():
        head.band_contrasts_db.copy_(torch.as_tensor(values, dtype=torch.float32, device="cpu"))


def _loss(head: Any, source: Any, baseline: Any, linear_weight: Any, quadratic_weight: Any, scale: float) -> tuple[Any, Any]:
    result = head(source, baseline)
    residual = result["output_pcm_float32"] - baseline
    objective = float(scale) * ((residual * linear_weight).mean() + (residual.square() * quadratic_weight).mean())
    return objective, result


def _matrix_rank_summary(matrix: Any, numpy: Any, tolerance: float) -> dict[str, Any]:
    value = numpy.asarray(matrix, dtype=numpy.float64)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        return {"rank": 0, "singular_values": [], "condition_number": None, "row_count": int(value.shape[0]) if value.ndim == 2 else 0}
    singular = numpy.linalg.svd(value, compute_uv=False)
    rank = int(numpy.sum(singular > float(tolerance)))
    condition = float(singular[0] / singular[-1]) if singular[-1] > float(tolerance) else None
    return {
        "rank": rank,
        "singular_values": singular.tolist(),
        "condition_number": condition,
        "row_count": int(value.shape[0]),
        "column_count": int(value.shape[1]),
    }


def _run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve(strict=True)
    config_path = Path(args.config).resolve(strict=True)
    worker_path = Path(__file__).resolve(strict=True)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    if config.get("schema") != "cross_tfg_v3_zero_theta_gradient_supplement_config_v1":
        raise ValueError("unsupported supplement config schema")
    expected_cfg_path = (repo_root / config["config_relative_path"]).resolve(strict=True)
    if config_path != expected_cfg_path:
        raise ValueError("config file path does not match the declared repository-relative path")
    sources = _pin_sources(repo_root, config, worker_path)
    runtime = config["runtime"]
    if list(sys.version_info[:3]) != runtime["python_version"]:
        raise RuntimeError("Python version differs from the frozen engineering runtime")
    if Path(sys.executable).resolve() != Path(runtime["python_executable"]).resolve():
        raise RuntimeError("interpreter path differs from the frozen engineering runtime")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE must be 1")
    for variable, expected in runtime["environment"].items():
        if os.environ.get(variable) != expected:
            raise RuntimeError(f"required CPU runtime environment mismatch: {variable}")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise RuntimeError(f"single-thread CPU bound missing: {variable}")

    import numpy as np

    if np.__version__ != runtime["numpy_version"]:
        raise RuntimeError("NumPy version differs from the frozen engineering runtime")
    sys.path.insert(0, str(repo_root))
    import torch
    from scripts.audio_enhancement.baselines import apply_global_rms
    from scripts.audio_enhancement.shared_recipient_transition_head_v3 import (
        BAND_COUNT,
        SCHEMA as HEAD_SCHEMA,
        TRAINABLE_SCALAR_COUNT,
        SharedRecipientTransitionHeadV3,
    )

    if torch.__version__ != runtime["torch_version"]:
        raise RuntimeError("Torch version differs from the frozen engineering runtime")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized; this worker is CPU-only")
    torch.set_num_threads(int(runtime["torch_intraop_threads"]))
    torch.set_num_interop_threads(int(runtime["torch_interop_threads"]))
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Torch CPU thread bound did not take effect")
    if HEAD_SCHEMA != config["head_schema"] or TRAINABLE_SCALAR_COUNT != 9 or BAND_COUNT != 4:
        raise RuntimeError("V3 head schema or scalar/band count mismatch")

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).absolute()
    out_dir = out_dir.resolve(strict=False)
    expected_out = (repo_root / config["output_relative_path"]).resolve(strict=False)
    if out_dir != expected_out:
        raise ValueError("output path differs from the declared fresh output path")
    try:
        out_dir.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("output path must remain inside the repository") from exc
    if out_dir.exists():
        raise FileExistsError("output directory already exists; refusing overwrite")
    out_dir.mkdir(parents=True, exist_ok=False)

    qcfg_path = (repo_root / config["qualification_config_relative_path"]).resolve(strict=True)
    qcfg = json.loads(qcfg_path.read_text(encoding="utf-8"))
    if qcfg.get("purpose") != "engineering_numerical_qualification_only":
        raise ValueError("source qualification config purpose mismatch")
    if qcfg["head"]["schema"] != HEAD_SCHEMA:
        raise ValueError("source qualification config V3 schema mismatch")
    ctor = dict(qcfg["head_constructor"])
    head = SharedRecipientTransitionHeadV3(**ctor).to(device="cpu").eval()
    params = list(head.parameters())
    if len(params) != 1 or params[0].numel() != 9 or params[0].dtype != torch.float32:
        raise RuntimeError("V3 parameter tensor does not match the frozen nine-scalar contract")
    if head.band_contrasts_db.shape != (3, 3):
        raise RuntimeError("V3 coefficient matrix must have shape [3,3]")

    probe = qcfg["synthetic_probe"]
    if int(probe["sample_count"]) > int(config["resource_limits"]["max_synthetic_sample_count"]):
        raise MemoryError("synthetic probe exceeds its frozen sample-count bound")
    source = _probe_waveform(probe, torch)
    source_np = source.detach().cpu().numpy().astype(np.float32, copy=False)
    baseline_np, baseline_info = apply_global_rms(
        source_np,
        float(qcfg["numpy_rms_baseline"]["target_rms"]),
        peak_ceiling=qcfg["numpy_rms_baseline"]["peak_ceiling"],
        silence_epsilon=float(qcfg["numpy_rms_baseline"]["silence_epsilon"]),
    )
    baseline_np = np.ascontiguousarray(baseline_np, dtype=np.float32)
    source_tensor = torch.from_numpy(np.ascontiguousarray(source_np, dtype=np.float32))
    baseline_tensor = torch.from_numpy(baseline_np)
    linear_weight, quadratic_weight = _probe_weights(probe, int(source.numel()), torch)
    objective_scale = float(probe["objective"]["loss_scale"])
    zero_theta = np.zeros((3, 3), dtype=np.float32)
    _set_theta(head, zero_theta, torch)
    head.zero_grad(set_to_none=True)
    objective, zero_result = _loss(head, source_tensor, baseline_tensor, linear_weight, quadratic_weight, objective_scale)
    objective.backward()
    if head.band_contrasts_db.grad is None:
        raise RuntimeError("zero-theta objective did not retain an autograd gradient")
    gradient = head.band_contrasts_db.grad.detach().cpu().numpy().astype(np.float64)
    zero_output = zero_result["output_pcm_float32"].detach().cpu().numpy().astype(np.float32, copy=False)
    baseline_bytes = np.ascontiguousarray(baseline_np.astype("<f4", copy=False)).tobytes()
    output_bytes = np.ascontiguousarray(zero_output.astype("<f4", copy=False)).tobytes()
    zero_equal = output_bytes == baseline_bytes
    zero_max_abs_error = float(np.max(np.abs(zero_output - baseline_np)))
    zero_mask = zero_result["raw_log_magnitude_mask_db"].detach().cpu().numpy().astype(np.float32)
    projected_mask = zero_result["projected_log_magnitude_mask_db"].detach().cpu().numpy().astype(np.float32)
    smoothed_mask = zero_result["smoothed_log_magnitude_mask_db"].detach().cpu().numpy().astype(np.float32)
    positive_flux = zero_result["positive_relative_flux"].detach().cpu().numpy().astype(np.float32)
    negative_flux = zero_result["negative_relative_flux"].detach().cpu().numpy().astype(np.float32)
    positive_gate = zero_result["positive_flux_gate"].detach().cpu().numpy().astype(np.float32)
    negative_gate = zero_result["negative_flux_gate"].detach().cpu().numpy().astype(np.float32)
    valid_transition = zero_result["valid_transition_mask"].detach().cpu().numpy().astype(np.uint8)
    active_frame = zero_result["active_frame_mask"].detach().cpu().numpy().astype(np.uint8)
    magnitude = zero_result["source_stft"].abs().detach().cpu().numpy()
    activity_cells = magnitude > np.float32(ctor["activity_floor"])
    basis = zero_result["band_basis"].detach().cpu().numpy().astype(np.float32)
    temporal_support = valid_transition.astype(np.bool_)
    temporal_matrix = np.column_stack((
        np.ones(int(temporal_support.sum()), dtype=np.float64),
        positive_gate[temporal_support].astype(np.float64),
        negative_gate[temporal_support].astype(np.float64),
    ))
    temporal_rank = _matrix_rank_summary(temporal_matrix, np, float(qcfg["qualification_checks"]["matrix_rank_tolerance"]))
    band_rank = _matrix_rank_summary(basis.T, np, float(qcfg["qualification_checks"]["matrix_rank_tolerance"]))

    finite_difference = probe["finite_difference"]
    coordinate_step = float(finite_difference["coordinate_step_db"])
    abs_tolerance = float(finite_difference["absolute_tolerance"])
    rel_tolerance = float(finite_difference["relative_tolerance"])
    coordinate_fd = np.empty((3, 3), dtype=np.float64)
    for flat_index in range(9):
        plus = zero_theta.copy().reshape(-1)
        minus = zero_theta.copy().reshape(-1)
        plus[flat_index] += np.float32(coordinate_step)
        minus[flat_index] -= np.float32(coordinate_step)
        _set_theta(head, plus.reshape(3, 3), torch)
        with torch.no_grad():
            plus_loss, _ = _loss(head, source_tensor, baseline_tensor, linear_weight, quadratic_weight, objective_scale)
        _set_theta(head, minus.reshape(3, 3), torch)
        with torch.no_grad():
            minus_loss, _ = _loss(head, source_tensor, baseline_tensor, linear_weight, quadratic_weight, objective_scale)
        coordinate_fd.reshape(-1)[flat_index] = float((plus_loss.item() - minus_loss.item()) / (2.0 * coordinate_step))
    coordinate_error = np.abs(coordinate_fd - gradient)
    coordinate_allowed = abs_tolerance + rel_tolerance * np.maximum(np.abs(coordinate_fd), np.abs(gradient))
    coordinate_agreement = coordinate_error <= coordinate_allowed

    raw_direction = np.asarray(finite_difference["direction_matrix"], dtype=np.float64)
    direction_norm = float(np.linalg.norm(raw_direction))
    if not math.isfinite(direction_norm) or direction_norm == 0.0:
        raise ValueError("frozen directional finite-difference probe is not finite and nonzero")
    direction = raw_direction / direction_norm
    directional_step = float(finite_difference["directional_step_db"])
    _set_theta(head, (direction * directional_step).astype(np.float32), torch)
    with torch.no_grad():
        plus_loss, _ = _loss(head, source_tensor, baseline_tensor, linear_weight, quadratic_weight, objective_scale)
    _set_theta(head, (-direction * directional_step).astype(np.float32), torch)
    with torch.no_grad():
        minus_loss, _ = _loss(head, source_tensor, baseline_tensor, linear_weight, quadratic_weight, objective_scale)
    directional_fd = float((plus_loss.item() - minus_loss.item()) / (2.0 * directional_step))
    directional_autograd = float(np.sum(gradient * direction))
    directional_error = abs(directional_fd - directional_autograd)
    directional_allowed = abs_tolerance + rel_tolerance * max(abs(directional_fd), abs(directional_autograd))
    directional_agreement = directional_error <= directional_allowed
    _set_theta(head, zero_theta, torch)

    silence_count = int(probe["silence_sample_count"])
    silence_source = np.zeros(silence_count, dtype=np.float32)
    silence_baseline, _ = apply_global_rms(
        silence_source,
        float(qcfg["numpy_rms_baseline"]["target_rms"]),
        peak_ceiling=qcfg["numpy_rms_baseline"]["peak_ceiling"],
        silence_epsilon=float(qcfg["numpy_rms_baseline"]["silence_epsilon"]),
    )
    with torch.no_grad():
        silence_result = head(torch.from_numpy(silence_source), torch.from_numpy(np.ascontiguousarray(silence_baseline, dtype=np.float32)))
    silence_output = silence_result["output_pcm_float32"].detach().cpu().numpy().astype(np.float32, copy=False)
    silence_equal = bool(np.array_equal(silence_output, silence_baseline))

    all_arrays = {
        "theta_zero_f32": zero_theta,
        "autograd_gradient_f64": gradient,
        "coordinate_finite_difference_f64": coordinate_fd,
        "coordinate_absolute_error_f64": coordinate_error,
        "coordinate_allowed_error_f64": coordinate_allowed,
        "coordinate_agreement_u8": coordinate_agreement.astype(np.uint8),
        "direction_matrix_normalized_f64": direction,
        "valid_transition_mask_u8": valid_transition,
        "active_frame_mask_u8": active_frame,
        "activity_cells_u8": activity_cells.astype(np.uint8),
        "positive_relative_flux_f32": positive_flux,
        "negative_relative_flux_f32": negative_flux,
        "positive_flux_gate_f32": positive_gate,
        "negative_flux_gate_f32": negative_gate,
        "raw_log_magnitude_mask_db_f32": zero_mask,
        "smoothed_log_magnitude_mask_db_f32": smoothed_mask,
        "projected_log_magnitude_mask_db_f32": projected_mask,
        "band_basis_f32": basis,
        "source_frame_power_f32": zero_result["physical_frame_power"]["source_original_pcm"].detach().cpu().numpy().astype(np.float32),
        "baseline_frame_power_f32": zero_result["physical_frame_power"]["canonical_rms_baseline"].detach().cpu().numpy().astype(np.float32),
        "output_frame_power_f32": zero_result["physical_frame_power"]["output_pcm"].detach().cpu().numpy().astype(np.float32),
        "stft_frame_centers_source_samples_i64": zero_result["stft_frame_centers_source_samples"].detach().cpu().numpy().astype(np.int64),
        "video_frame_centers_source_samples_i64": zero_result["video_frame_centers_source_samples"].detach().cpu().numpy().astype(np.int64),
        "directional_finite_difference_f64": np.asarray([directional_fd], dtype=np.float64),
        "directional_autograd_f64": np.asarray([directional_autograd], dtype=np.float64),
        "directional_absolute_error_f64": np.asarray([directional_error], dtype=np.float64),
        "directional_allowed_error_f64": np.asarray([directional_allowed], dtype=np.float64),
        "directional_agreement_u8": np.asarray([directional_agreement], dtype=np.uint8),
    }
    for name, array in all_arrays.items():
        if not np.isfinite(array).all():
            raise FloatingPointError(f"supplement diagnostic array is non-finite: {name}")
    uncompressed_array_bytes = sum(int(array.nbytes) for array in all_arrays.values())
    if uncompressed_array_bytes > int(config["resource_limits"]["max_uncompressed_array_bytes"]):
        raise MemoryError("supplement arrays exceeded their frozen uncompressed byte bound")
    npz_path = out_dir / "zero_theta_diagnostic_arrays.npz"
    with npz_path.open("xb") as stream:
        np.savez_compressed(stream, **all_arrays)
        stream.flush()
        os.fsync(stream.fileno())

    gradient_finite = bool(np.isfinite(gradient).all())
    hard_checks = {
        "zero_theta_output_bitwise_equal_to_numpy_rms_baseline": zero_equal,
        "zero_theta_output_exact_sample_count": int(zero_output.size) == int(source_tensor.numel()),
        "zero_theta_output_finite": bool(np.isfinite(zero_output).all()),
        "autograd_gradient_all_finite": gradient_finite,
        "raw_mask_finite_and_zero": bool(np.isfinite(zero_mask).all() and np.count_nonzero(zero_mask) == 0),
        "projected_mask_finite_and_zero": bool(np.isfinite(projected_mask).all() and np.count_nonzero(projected_mask) == 0),
        "silence_output_bitwise_equal_to_canonical_silence_baseline": silence_equal,
        "silence_output_exactly_zero": int(np.count_nonzero(silence_output)) == 0,
        "coordinate_finite_differences_all_finite": bool(np.isfinite(coordinate_fd).all()),
        "directional_finite_difference_finite": bool(math.isfinite(directional_fd)),
    }
    if not all(hard_checks.values()):
        status = "COMPLETE_WITH_HARD_CHECK_FAILURE"
    elif bool(np.all(coordinate_agreement)) and directional_agreement:
        status = "COMPLETE_ENGINEERING_CHECKS_PASS"
    else:
        status = "COMPLETE_WITH_FINITE_DIFFERENCE_DISCREPANCY"
    result = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "scope": config["scope"],
        "qualification_config_sha256": _sha256_bytes(config_bytes),
        "pinned_sources": sources,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_initialized": bool(torch.cuda.is_initialized()),
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "probe": {
            "probe_id": probe["probe_id"],
            "sample_count": int(source_tensor.numel()),
            "objective_value_at_zero": float(objective.detach().item()),
            "baseline_pcm_float32_sha256": _sha256_bytes(baseline_bytes),
            "output_pcm_float32_sha256": _sha256_bytes(output_bytes),
            "zero_output_max_abs_error": zero_max_abs_error,
            "numpy_rms_baseline_info": {
                "input_rms": baseline_info.input_rms,
                "output_rms": baseline_info.output_rms,
                "applied_gain": baseline_info.applied_gain,
                "peak_limited": baseline_info.peak_limited,
            },
            "gradient_matrix_f64": _finite_json(gradient.tolist()),
            "gradient_finite": gradient_finite,
            "gradient_nonzero_scalar_count": int(np.count_nonzero(gradient)),
            "coordinate_fd_matrix_f64": _finite_json(coordinate_fd.tolist()),
            "coordinate_agreement_count": int(np.count_nonzero(coordinate_agreement)),
            "coordinate_agreement_total": int(coordinate_agreement.size),
            "coordinate_step_db": coordinate_step,
            "direction_matrix_normalized_f64": direction.tolist(),
            "directional_autograd": directional_autograd,
            "directional_finite_difference": directional_fd,
            "directional_absolute_error": directional_error,
            "directional_allowed_error": directional_allowed,
            "directional_agreement": bool(directional_agreement),
            "finite_difference_tolerance_source": "qualification_config.synthetic_probe.finite_difference",
            "temporal_design": temporal_rank,
            "frequency_band_design": band_rank,
            "valid_transition_count": int(np.sum(valid_transition)),
            "active_source_cell_count": int(np.sum(activity_cells)),
            "frequency_bin_count": int(basis.shape[1]),
            "frequency_band_bin_counts": [int(v) for v in basis.sum(axis=1)],
            "zero_mask_max_abs_db": float(np.max(np.abs(zero_mask))),
            "projected_mask_max_abs_db": float(np.max(np.abs(projected_mask))),
            "silence_baseline_sha256": _sha256_bytes(np.ascontiguousarray(silence_baseline.astype("<f4", copy=False)).tobytes()),
            "silence_output_sha256": _sha256_bytes(np.ascontiguousarray(silence_output.astype("<f4", copy=False)).tobytes()),
            "silence_output_exactly_zero": int(np.count_nonzero(silence_output)) == 0,
        },
        "hard_checks": hard_checks,
        "interpretation_boundary": "Finite differences check numerical consistency only. A zero or near-zero derivative is retained as measured. Positive-minus-negative relative magnitude flux also encodes local total-magnitude change; this is not a video, articulation-only, or efficacy result.",
    }
    result_bytes = _canonical_json(result)
    if len(result_bytes) > int(config["resource_limits"]["max_json_bytes"]):
        raise MemoryError("supplement JSON exceeded its frozen byte bound")
    _write_new(out_dir / "result.json", result_bytes)
    matrix_sha, matrix_size = _sha256_file(npz_path)
    if matrix_size > int(config["resource_limits"]["max_matrix_archive_bytes"]):
        raise MemoryError("supplement NPZ exceeded its frozen compressed byte bound")
    result_path = out_dir / "result.json"
    result_sha, result_size = _sha256_file(result_path)
    max_output = int(config["resource_limits"]["max_total_output_bytes"])
    total_bytes = matrix_size + result_size
    if total_bytes > max_output:
        raise MemoryError("supplement output exceeded its frozen byte bound")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": status,
        "output_files": [
            {"path": "zero_theta_diagnostic_arrays.npz", "bytes": matrix_size, "sha256": matrix_sha},
            {"path": "result.json", "bytes": result_size, "sha256": result_sha},
        ],
        "total_output_bytes": total_bytes,
        "max_output_bytes": max_output,
        "uncompressed_array_bytes": uncompressed_array_bytes,
        "max_uncompressed_array_bytes": int(config["resource_limits"]["max_uncompressed_array_bytes"]),
        "configuration_sha256": _sha256_bytes(config_bytes),
        "source_pins": sources,
    }
    _write_new(out_dir / "result_manifest.json", _canonical_json(manifest))
    return 0 if status == "COMPLETE_ENGINEERING_CHECKS_PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
