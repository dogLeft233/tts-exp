"""CPU-only engineering qualification for the unfrozen V3 waveform head.

This worker is deliberately separate from training, frontend inference,
generation, and scoring. It reads only the 40 sealed historical fit WAVs named
by the pinned fit40 transfer manifest. It never writes audio. Importing this
file has no side effects; execution requires every path as an explicit CLI
argument and writes only bounded JSON/NPZ diagnostics to a new output folder.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


WORKER_SCHEMA = "cross_tfg_shared_recipient_transition_head_v3_cpu_qualification_worker_v1"
RESULT_SCHEMA = "cross_tfg_shared_recipient_transition_head_v3_cpu_qualification_result_v1"
SUMMARY_SCHEMA = "cross_tfg_shared_recipient_transition_head_v3_cpu_qualification_summary_v1"
MANIFEST_SCHEMA = "cross_tfg_shared_recipient_transition_head_v3_cpu_qualification_manifest_v1"
HEX = frozenset("0123456789abcdef")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _f32le_sha256(values: Any) -> str:
    import numpy as np

    arr = np.asarray(values, dtype=np.dtype("<f4"))
    if arr.ndim != 1:
        raise ValueError("PCM digest requires a one-dimensional float32 array")
    return _sha256_bytes(np.ascontiguousarray(arr).tobytes())


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> tuple[str, int]:
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return _sha256_bytes(payload), len(payload)


def _append_jsonl_durable(
    stream: Any, value: Mapping[str, Any], *, max_bytes: int, current_bytes: int
) -> tuple[int, str]:
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    if current_bytes + len(payload) > max_bytes:
        raise MemoryError("qualification JSONL would exceed configured byte bound")
    stream.write(payload)
    stream.flush()
    os.fsync(stream.fileno())
    return len(payload), _sha256_bytes(payload)


def _safe_relative_path(value: str) -> PurePosixPath:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"unsafe transfer-relative path: {value!r}")
    return candidate


def _verify_file_pin(repo_root: Path, name: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    relative = _safe_relative_path(str(entry["path"]))
    expected_sha = str(entry["sha256"])
    expected_bytes = int(entry["bytes"])
    if len(expected_sha) != 64 or any(char not in HEX for char in expected_sha):
        raise ValueError(f"invalid SHA-256 in source pin {name}")
    path = repo_root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"source pin is absent or symlinked: {relative.as_posix()}")
    actual_sha, actual_bytes = _sha256_file(path)
    if actual_sha != expected_sha or actual_bytes != expected_bytes:
        raise ValueError(f"source pin mismatch: {name}")
    return {"path": relative.as_posix(), "sha256": actual_sha, "bytes": actual_bytes}


def _verify_runtime_environment(expected: Mapping[str, Any]) -> dict[str, Any]:
    if list(sys.version_info[:3]) != list(expected["python_version"]):
        raise RuntimeError(f"Python version mismatch: observed {list(sys.version_info[:3])}")
    # Deliberately compare the literal executable path. Resolving a venv path
    # could silently replace it with its base interpreter.
    if sys.executable != expected["python_executable"]:
        raise RuntimeError("literal Python executable path mismatch")
    for variable, value in expected["environment"].items():
        if os.environ.get(variable) != value:
            raise RuntimeError(f"runtime environment mismatch: {variable}")

    import numpy as np
    import torch

    if np.__version__ != expected["numpy_version"]:
        raise RuntimeError(f"NumPy version mismatch: observed {np.__version__}")
    if torch.__version__ != expected["torch_version"]:
        raise RuntimeError(f"Torch version mismatch: observed {torch.__version__}")
    torch.set_num_threads(int(expected["torch_intraop_threads"]))
    torch.set_num_interop_threads(int(expected["torch_interop_threads"]))
    if torch.get_num_threads() != int(expected["torch_intraop_threads"]):
        raise RuntimeError("Torch intra-op thread count mismatch")
    if torch.get_num_interop_threads() != int(expected["torch_interop_threads"]):
        raise RuntimeError("Torch inter-op thread count mismatch")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized in a CPU-only qualification worker")
    return {
        "python_version": list(sys.version_info[:3]),
        "python_executable_literal": sys.executable,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_initialized": torch.cuda.is_initialized(),
        "platform": platform.platform(),
    }


def _load_manifest(source_root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_rel = _safe_relative_path(str(config["source_bundle"]["manifest_relative_path"]))
    manifest_path = source_root.joinpath(*manifest_rel.parts)
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError("fit40 transfer manifest is missing or symlinked")
    manifest_sha, manifest_bytes = _sha256_file(manifest_path)
    pin = config["source_bundle"]["manifest_pin"]
    if manifest_sha != pin["sha256"] or manifest_bytes != int(pin["bytes"]):
        raise ValueError("fit40 transfer manifest raw pin mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != config["source_bundle"]["schema"]:
        raise ValueError("fit40 transfer manifest schema mismatch")
    if manifest.get("status") != config["source_bundle"]["required_status"]:
        raise ValueError("fit40 transfer manifest status mismatch")
    scope = manifest.get("scope", {})
    if any(bool(scope.get(key)) for key in ("dev_audio_read", "confirm_audio_read", "model_forward", "render", "score", "candidate_transform_executed")):
        raise ValueError("fit40 source manifest violates fit-only/no-effect scope")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != int(config["source_bundle"]["expected_row_count"]):
        raise ValueError("fit40 source manifest does not have the exact declared row count")
    row_ids = [str(row.get("row_id", "")) for row in rows]
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    speaker_ids = [str(row.get("speaker_id", "")) for row in rows]
    canonical_js = [int(row.get("canonical_j", -1)) for row in rows]
    if len(set(row_ids)) != len(rows) or len(set(sample_ids)) != len(rows):
        raise ValueError("fit40 row/sample identifiers are not unique")
    if len(set(speaker_ids)) != int(config["source_bundle"]["expected_speaker_count"]):
        raise ValueError("fit40 speaker count differs from the frozen roster")
    if sorted(canonical_js.count(j) for j in (0, 15)) != [len(rows) // 2, len(rows) // 2]:
        raise ValueError("fit40 rows do not contain the exact j=0/j=15 endpoint split")
    expected_speakers = set(config["source_bundle"]["expected_speakers"])
    if set(speaker_ids) != expected_speakers:
        raise ValueError("fit40 speaker IDs differ from the frozen ADR013 fit roster")
    for row in rows:
        if int(row["sample_rate_hz"]) != int(config["audio_contract"]["sample_rate_hz"]):
            raise ValueError(f"sample-rate mismatch in manifest row {row['row_id']}")
        if int(row["channels"]) != int(config["audio_contract"]["channels"]):
            raise ValueError(f"channel-count mismatch in manifest row {row['row_id']}")
        if row["source_encoding"] != config["audio_contract"]["source_encoding"]:
            raise ValueError(f"encoding mismatch in manifest row {row['row_id']}")
        if int(row["sample_count"]) > int(config["resource_limits"]["max_audio_samples"]):
            raise ValueError(f"fit WAV exceeds sample-count bound: {row['row_id']}")
        if int(row["transfer_file_bytes"]) > int(config["resource_limits"]["max_single_wav_bytes"]):
            raise ValueError(f"fit WAV exceeds file-byte bound: {row['row_id']}")
    return manifest, {"path": manifest_rel.as_posix(), "sha256": manifest_sha, "bytes": manifest_bytes}


def _load_fit_audio(source_root: Path, row: Mapping[str, Any], config: Mapping[str, Any], read_canonical_wav: Any, pcm_float32_sha256: Any) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    relative = _safe_relative_path(str(row["transfer_relative_path"]))
    path = source_root.joinpath(*relative.parts)
    root_abs = source_root.resolve(strict=True)
    path_abs = path.resolve(strict=True) if path.exists() else path.absolute()
    if path.is_symlink() or not path.is_file() or os.path.commonpath((str(root_abs), str(path_abs))) != str(root_abs):
        raise FileNotFoundError(f"fit WAV is missing, symlinked, or escapes source root: {relative.as_posix()}")
    expected_file_sha = str(row["transfer_file_sha256"])
    actual_file_sha, actual_bytes = _sha256_file(path)
    if actual_file_sha != expected_file_sha or actual_bytes != int(row["transfer_file_bytes"]):
        raise ValueError(f"fit WAV raw file pin mismatch: {row['row_id']}")
    samples, info = read_canonical_wav(path, target_rate=int(config["audio_contract"]["sample_rate_hz"]))
    pcm_sha = pcm_float32_sha256(samples)
    checks = {
        "file_sha256": info.file_sha256 == expected_file_sha,
        "file_bytes": actual_bytes == int(row["transfer_file_bytes"]),
        "canonical_pcm_sha256": pcm_sha == row["canonical_float32_pcm_sha256"],
        "sample_count": int(samples.size) == int(row["sample_count"]),
        "sample_rate": info.sample_rate == int(config["audio_contract"]["sample_rate_hz"]),
        "source_sample_rate": info.source_sample_rate == int(config["audio_contract"]["sample_rate_hz"]),
        "source_channels": info.source_channels == int(config["audio_contract"]["channels"]),
        "source_encoding": info.source_encoding == config["audio_contract"]["source_encoding"],
        "resampler_none": info.resampler == "none",
        "finite": bool(np.isfinite(samples).all()),
    }
    if not all(checks.values()):
        raise ValueError(f"canonical audio contract mismatch in {row['row_id']}: {checks}")
    return np.ascontiguousarray(samples, dtype=np.float32), {
        "file_sha256": actual_file_sha,
        "file_bytes": actual_bytes,
        "canonical_float32_pcm_sha256": pcm_sha,
        "sample_count": int(samples.size),
        "canonical_decode_checks": checks,
    }


def _as_np(value: Any, dtype: Any | None = None) -> Any:
    array = value.detach().cpu().numpy()
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def _temporal_diagnostics(
    result: Mapping[str, Any], torch: Any, rank_tolerance: float, gate_difference_threshold: float
) -> dict[str, Any]:
    positive = result["positive_flux_gate"].detach()
    negative = result["negative_flux_gate"].detach()
    valid = result["valid_transition_mask"].detach().to(dtype=torch.bool)
    active = result["active_frame_mask"].detach().to(dtype=torch.bool)
    difference = (positive - negative).abs()
    if bool(valid.any().item()):
        design = torch.stack((torch.ones_like(positive), positive, negative), dim=1)[valid]
        temporal_rank = int(torch.linalg.matrix_rank(design, tol=rank_tolerance).item())
        max_gate_difference = float(difference[valid].amax().item())
        mean_gate_difference = float(difference[valid].mean().item())
    else:
        temporal_rank = 0
        max_gate_difference = 0.0
        mean_gate_difference = 0.0
    return {
        "stft_frame_count": int(positive.numel()),
        "active_frame_count": int(active.sum().item()),
        "valid_transition_count": int(valid.sum().item()),
        "first_transition_invalid": bool(not valid[0].item()) if valid.numel() else None,
        "first_positive_flux_placeholder": float(result["positive_relative_flux"][0].item()) if positive.numel() else None,
        "first_negative_flux_placeholder": float(result["negative_relative_flux"][0].item()) if negative.numel() else None,
        "first_positive_gate": float(positive[0].item()) if positive.numel() else None,
        "first_negative_gate": float(negative[0].item()) if negative.numel() else None,
        "positive_negative_gate_max_abs_difference_on_valid_support": max_gate_difference,
        "positive_negative_gate_mean_abs_difference_on_valid_support": mean_gate_difference,
        "positive_negative_gate_arrays_distinct_on_valid_support": max_gate_difference > 0.0,
        "positive_negative_gate_arrays_distinct_above_diagnostic_threshold": (
            max_gate_difference > gate_difference_threshold
        ),
        "gate_difference_diagnostic_threshold": gate_difference_threshold,
        "temporal_design_columns": ["intercept", "positive_flux_gate", "negative_flux_gate"],
        "temporal_design_rank": temporal_rank,
        "temporal_design_rank_tolerance": rank_tolerance,
    }


def _fit_row_arrays(result: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    flux = {
        "raw_mask_db_f32": _as_np(result["raw_log_magnitude_mask_db"], np.float32),
        "projected_mask_db_f32": _as_np(result["projected_log_magnitude_mask_db"], np.float32),
        "positive_relative_flux_f32": _as_np(result["positive_relative_flux"], np.float32),
        "negative_relative_flux_f32": _as_np(result["negative_relative_flux"], np.float32),
        "positive_gate_f32": _as_np(result["positive_flux_gate"], np.float32),
        "negative_gate_f32": _as_np(result["negative_flux_gate"], np.float32),
        "active_frame_mask_u8": _as_np(result["active_frame_mask"], np.uint8),
        "valid_transition_mask_u8": _as_np(result["valid_transition_mask"], np.uint8),
        "frame_magnitude_sum_f32": _as_np(result["source_stft"].abs().sum(dim=0), np.float32),
        "previous_frame_magnitude_sum_f32": _as_np(result["previous_frame_magnitude_sum"], np.float32),
        "flux_denominator_f32": _as_np(result["flux_denominator"], np.float32),
        "stft_frame_centers_source_samples_i64": _as_np(
            result["stft_frame_centers_source_samples"], np.int64
        ),
    }
    power = result["physical_frame_power"]
    flux.update({
        "source_frame_power_f32": _as_np(power["source_original_pcm"], np.float32),
        "candidate_frame_power_f32": _as_np(power["candidate_before_rms"], np.float32),
        "baseline_frame_power_f32": _as_np(power["canonical_rms_baseline"], np.float32),
        "output_frame_power_f32": _as_np(power["output_pcm"], np.float32),
        "valid_source_frame_power_u8": _as_np(power["valid_source_power_mask"], np.uint8),
        "video_frame_centers_source_samples_i64": _as_np(
            result["video_frame_centers_source_samples"], np.int64
        ),
    })
    return flux


def _validate_array_bank_size(arrays: Mapping[str, Any], max_uncompressed_bytes: int) -> int:
    total = 0
    for key, array in arrays.items():
        if getattr(array, "dtype", None) is None or array.dtype.hasobject:
            raise TypeError(f"object or invalid array prohibited in matrix bank: {key}")
        total += int(array.nbytes)
    if total > max_uncompressed_bytes:
        raise MemoryError(f"matrix bank exceeds configured uncompressed byte bound: {total}")
    return total


def _make_probe_waveform(probe: Mapping[str, Any], torch: Any) -> Any:
    sample_rate = int(probe["sample_rate_hz"])
    count = int(probe["sample_count"])
    time_axis = torch.arange(count, dtype=torch.float32) / float(sample_rate)
    waveform = torch.zeros(count, dtype=torch.float32)
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


def _probe_weight(probe: Mapping[str, Any], count: int, torch: Any) -> tuple[Any, Any]:
    sample_rate = int(probe["sample_rate_hz"])
    time_axis = torch.arange(count, dtype=torch.float32) / float(sample_rate)
    objective = probe["objective"]
    weight = torch.zeros_like(time_axis)
    for term in objective["linear_terms"]:
        weight = weight + float(term["coefficient"]) * torch.sin(
            2.0 * math.pi * float(term["frequency_hz"]) * time_axis + float(term["phase_radians"])
        )
    quadratic_weight = torch.full_like(time_axis, float(objective["quadratic_weight_offset"]))
    for term in objective["quadratic_weight_terms"]:
        quadratic_weight = quadratic_weight + float(term["coefficient"]) * torch.cos(
            2.0 * math.pi * float(term["frequency_hz"]) * time_axis + float(term["phase_radians"])
        )
    return weight.contiguous(), quadratic_weight.contiguous()


def _set_parameters(head: Any, values: Any, torch: Any) -> None:
    with torch.no_grad():
        head.band_contrasts_db.copy_(torch.as_tensor(values, dtype=torch.float32, device="cpu"))


def _probe_loss(head: Any, source: Any, baseline: Any, weight: Any, quadratic_weight: Any, scale: float, torch: Any) -> tuple[Any, Any]:
    result = head(source, baseline)
    residual = result["output_pcm_float32"] - baseline
    loss = float(scale) * ((residual * weight).mean() + (residual.square() * quadratic_weight).mean())
    return loss, result


def _gradient_diagnostics(head: Any, baseline_fn: Any, config: Mapping[str, Any], torch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    probe = config["synthetic_probe"]
    source = _make_probe_waveform(probe, torch)
    baseline_np, baseline_info = baseline_fn(
        source.detach().cpu().numpy(),
        float(config["head_constructor"]["target_rms"]),
        peak_ceiling=config["head_constructor"]["peak_ceiling"],
        silence_epsilon=float(config["numpy_rms_baseline"]["silence_epsilon"]),
    )
    baseline = torch.from_numpy(np.ascontiguousarray(baseline_np, dtype=np.float32))
    weight, quadratic_weight = _probe_weight(probe, int(source.numel()), torch)
    coefficients = np.asarray(probe["diagnostic_parameter_matrix_db"], dtype=np.float32)
    _set_parameters(head, coefficients, torch)

    head.zero_grad(set_to_none=True)
    loss, nonzero_result = _probe_loss(
        head,
        source,
        baseline,
        weight,
        quadratic_weight,
        float(probe["objective"]["loss_scale"]),
        torch,
    )
    loss.backward()
    gradient = head.band_contrasts_db.grad.detach().cpu().numpy().astype(np.float64)
    base = head.band_contrasts_db.detach().cpu().numpy().astype(np.float32).copy()
    epsilon = float(probe["finite_difference"]["coordinate_step_db"])
    abs_tol = float(probe["finite_difference"]["absolute_tolerance"])
    rel_tol = float(probe["finite_difference"]["relative_tolerance"])
    coordinate_rows = []

    def scalar_loss(values: Any) -> float:
        _set_parameters(head, values, torch)
        with torch.no_grad():
            value, _ = _probe_loss(
                head,
                source,
                baseline,
                weight,
                quadratic_weight,
                float(probe["objective"]["loss_scale"]),
                torch,
            )
        return float(value.item())

    for flat_index in range(9):
        plus = base.copy().reshape(-1)
        minus = base.copy().reshape(-1)
        plus[flat_index] += epsilon
        minus[flat_index] -= epsilon
        fd = (scalar_loss(plus.reshape(3, 3)) - scalar_loss(minus.reshape(3, 3))) / (2.0 * epsilon)
        auto = float(gradient.reshape(-1)[flat_index])
        error = abs(fd - auto)
        allowed = abs_tol + rel_tol * max(abs(fd), abs(auto))
        coordinate_rows.append({
            "flat_parameter_index": flat_index,
            "temporal_component_index": flat_index // 3,
            "nonreference_band_index": flat_index % 3,
            "autograd": auto,
            "central_finite_difference": fd,
            "absolute_error": error,
            "allowed_error": allowed,
            "agreement": bool(error <= allowed),
        })

    direction = np.asarray(probe["finite_difference"]["direction_matrix"], dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    if not math.isfinite(direction_norm) or direction_norm == 0.0:
        raise ValueError("finite-difference direction must be finite and nonzero")
    direction = direction / direction_norm
    directional_step = float(probe["finite_difference"]["directional_step_db"])
    plus_direction = base.astype(np.float64) + directional_step * direction
    minus_direction = base.astype(np.float64) - directional_step * direction
    fd_direction = (
        scalar_loss(plus_direction.astype(np.float32)) - scalar_loss(minus_direction.astype(np.float32))
    ) / (2.0 * directional_step)
    auto_direction = float(np.sum(gradient * direction))
    directional_error = abs(fd_direction - auto_direction)
    directional_allowed = abs_tol + rel_tol * max(abs(fd_direction), abs(auto_direction))
    _set_parameters(head, coefficients, torch)
    probe_temporal = _temporal_diagnostics(
        nonzero_result,
        torch,
        float(config["qualification_checks"]["matrix_rank_tolerance"]),
        float(config["qualification_checks"]["signed_gate_distinctness_diagnostic_threshold"]),
    )

    probe_result = {
        "probe_id": probe["probe_id"],
        "sample_count": int(source.numel()),
        "baseline_pcm_sha256": _f32le_sha256(baseline_np),
        "numpy_baseline_info": {
            "input_rms": baseline_info.input_rms,
            "output_rms": baseline_info.output_rms,
            "applied_gain": baseline_info.applied_gain,
            "peak_limited": baseline_info.peak_limited,
        },
        "loss": float(loss.detach().item()),
        "gradient_matrix": gradient.tolist(),
        "gradient_all_finite": bool(np.isfinite(gradient).all()),
        "gradient_nonzero_scalar_count": int(np.count_nonzero(gradient)),
        "coordinate_finite_difference": coordinate_rows,
        "all_coordinate_agreement": all(row["agreement"] for row in coordinate_rows),
        "normalized_direction_matrix": direction.tolist(),
        "directional_autograd": auto_direction,
        "directional_central_finite_difference": fd_direction,
        "directional_absolute_error": directional_error,
        "directional_allowed_error": directional_allowed,
        "directional_agreement": bool(directional_error <= directional_allowed),
        "temporal_diagnostics": probe_temporal,
        "nonzero_parameter_output_checks": {
            "exact_source_sample_count": int(nonzero_result["output_pcm_float32"].numel()) == int(source.numel()),
            "output_finite": bool(torch.isfinite(nonzero_result["output_pcm_float32"]).all().item()),
            "first_transition_unsupported_with_zero_placeholders": bool(
                probe_temporal["first_transition_invalid"]
                and probe_temporal["first_positive_flux_placeholder"] == 0.0
                and probe_temporal["first_negative_flux_placeholder"] == 0.0
                and probe_temporal["first_positive_gate"] == 0.0
                and probe_temporal["first_negative_gate"] == 0.0
            ),
            "max_abs_projected_mask_db": float(nonzero_result["postprojection_max_abs_mask_db"].item()),
            "declared_mask_bound_db": float(config["head_constructor"]["max_abs_mask_db"]),
            "mask_within_bound": bool(
                nonzero_result["postprojection_max_abs_mask_db"].item()
                <= float(config["head_constructor"]["max_abs_mask_db"])
                + float(config["qualification_checks"]["mask_bound_tolerance_db"])
            ),
            "final_output_rms": float(nonzero_result["normalization"]["final_output_rms"].item()),
            "final_output_rms_error_to_target": float(
                nonzero_result["normalization"]["final_output_rms_error_to_target"].item()
            ),
            "final_output_peak": float(nonzero_result["normalization"]["final_output_peak"].item()),
            "final_peak_ceiling_excess": None,
        },
    }
    probe_arrays = {
        "synthetic_source_float32": source.detach().cpu().numpy().astype(np.float32, copy=False),
        "synthetic_baseline_float32": baseline_np.astype(np.float32, copy=False),
        "synthetic_loss_weight_float32": weight.detach().cpu().numpy().astype(np.float32, copy=False),
        "synthetic_quadratic_weight_float32": quadratic_weight.detach().cpu().numpy().astype(np.float32, copy=False),
        "synthetic_raw_mask_db_f32": _as_np(nonzero_result["raw_log_magnitude_mask_db"], np.float32),
        "synthetic_projected_mask_db_f32": _as_np(nonzero_result["projected_log_magnitude_mask_db"], np.float32),
        "synthetic_positive_flux_f32": _as_np(nonzero_result["positive_relative_flux"], np.float32),
        "synthetic_negative_flux_f32": _as_np(nonzero_result["negative_relative_flux"], np.float32),
        "synthetic_positive_gate_f32": _as_np(nonzero_result["positive_flux_gate"], np.float32),
        "synthetic_negative_gate_f32": _as_np(nonzero_result["negative_flux_gate"], np.float32),
        "synthetic_frame_magnitude_sum_f32": _as_np(
            nonzero_result["source_stft"].abs().sum(dim=0), np.float32
        ),
        "synthetic_previous_frame_magnitude_sum_f32": _as_np(
            nonzero_result["previous_frame_magnitude_sum"], np.float32
        ),
        "synthetic_flux_denominator_f32": _as_np(nonzero_result["flux_denominator"], np.float32),
        "synthetic_valid_transition_mask_u8": _as_np(nonzero_result["valid_transition_mask"], np.uint8),
        "synthetic_active_frame_mask_u8": _as_np(nonzero_result["active_frame_mask"], np.uint8),
        "synthetic_parameters_db_f32": coefficients,
        "synthetic_autograd_matrix_f64": gradient,
        "synthetic_coordinate_fd_f64": np.asarray(
            [row["central_finite_difference"] for row in coordinate_rows], dtype=np.float64
        ).reshape(3, 3),
        "synthetic_direction_matrix_f64": direction,
        "synthetic_band_basis_f32": _as_np(nonzero_result["band_basis"], np.float32),
        "synthetic_source_stft_real_f32": _as_np(nonzero_result["source_stft"].real, np.float32),
        "synthetic_source_stft_imag_f32": _as_np(nonzero_result["source_stft"].imag, np.float32),
    }
    return probe_result, probe_arrays


def _run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).absolute()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    if config.get("schema") != "cross_tfg_shared_recipient_transition_head_v3_cpu_qualification_config_v1":
        raise ValueError("qualification config schema mismatch")
    config_sha = _sha256_bytes(config_bytes)
    config["_runtime_config_sha256"] = config_sha
    repo_root = Path(args.repo_root).absolute()
    source_root = Path(args.source_root).absolute()
    output_dir = Path(args.output_dir).absolute()
    if output_dir.exists():
        raise FileExistsError(f"qualification output directory already exists: {output_dir}")

    source_pins = {}
    for name, entry in config["source_pins"].items():
        source_pins[name] = _verify_file_pin(repo_root, name, entry)
    runtime = _verify_runtime_environment(config["runtime"])

    # Imports occur only after literal interpreter/environment/source gates.
    sys.path.insert(0, str(repo_root))
    import numpy as np
    import torch
    from scripts.audio_enhancement.audio_io import pcm_float32_sha256, read_canonical_wav
    from scripts.audio_enhancement.baselines import apply_global_rms
    from scripts.audio_enhancement.shared_recipient_transition_head_v3 import (
        BAND_COUNT,
        SCHEMA as HEAD_SCHEMA,
        TRAINABLE_SCALAR_COUNT,
        SharedRecipientTransitionHeadV3,
    )

    manifest, manifest_pin = _load_manifest(source_root, config)
    if HEAD_SCHEMA != config["head"]["schema"] or TRAINABLE_SCALAR_COUNT != 9 or BAND_COUNT != 4:
        raise RuntimeError("V3 head schema/count mismatch")
    if not bool(torch.backends.mkl.is_available()) and config["runtime"].get("require_mkl", False):
        raise RuntimeError("configured MKL runtime is unavailable")
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA unexpectedly initialized after import")

    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    partial_path = output_dir / "results.partial.jsonl"
    result_records = []
    raw_masks = []
    projected_masks = []
    row_frame_offsets = [0]
    vector_fields = (
        "positive_relative_flux_f32",
        "negative_relative_flux_f32",
        "positive_gate_f32",
        "negative_gate_f32",
        "active_frame_mask_u8",
        "valid_transition_mask_u8",
        "frame_magnitude_sum_f32",
        "previous_frame_magnitude_sum_f32",
        "flux_denominator_f32",
        "stft_frame_centers_source_samples_i64",
    )
    fit_vectors = {field: [] for field in vector_fields}
    power_fields = (
        "source_frame_power_f32",
        "candidate_frame_power_f32",
        "baseline_frame_power_f32",
        "output_frame_power_f32",
        "valid_source_frame_power_u8",
        "video_frame_centers_source_samples_i64",
    )
    fit_power = {field: [] for field in power_fields}
    row_video_frame_offsets = [0]
    records_bytes = 0
    head = SharedRecipientTransitionHeadV3(**config["head_constructor"])
    head.to(device="cpu", dtype=torch.float32)
    head.eval()
    parameter_probe = config["synthetic_probe"]["diagnostic_parameter_matrix_db"]
    array_bank_bytes = 0

    try:
        with partial_path.open("xb") as journal:
            for ordinal, row in enumerate(manifest["rows"]):
                source_np, source_receipt = _load_fit_audio(
                    source_root, row, config, read_canonical_wav, pcm_float32_sha256
                )
                baseline_np, rms_info = apply_global_rms(
                    source_np,
                    float(config["head_constructor"]["target_rms"]),
                    peak_ceiling=config["head_constructor"]["peak_ceiling"],
                    silence_epsilon=float(config["numpy_rms_baseline"]["silence_epsilon"]),
                )
                baseline_np = np.ascontiguousarray(baseline_np, dtype=np.float32)
                source_tensor = torch.from_numpy(source_np)
                baseline_tensor = torch.from_numpy(baseline_np)

                _set_parameters(head, np.zeros((3, 3), dtype=np.float32), torch)
                with torch.no_grad():
                    zero_result = head(source_tensor, baseline_tensor)
                zero_output_np = _as_np(zero_result["output_pcm_float32"], np.dtype("<f4"))
                baseline_le = baseline_np.astype(np.dtype("<f4"), copy=False)
                zero_bytes = np.ascontiguousarray(zero_output_np, dtype=np.dtype("<f4")).tobytes()
                baseline_bytes = np.ascontiguousarray(baseline_le, dtype=np.dtype("<f4")).tobytes()
                bitwise_equal = zero_bytes == baseline_bytes
                zero_record = {
                    "exact_sample_count": int(zero_output_np.size) == int(source_np.size),
                    "finite": bool(np.isfinite(zero_output_np).all()),
                    "bitwise_equal_to_numpy_rms_baseline": bitwise_equal,
                    "output_pcm_float32_sha256": _f32le_sha256(zero_output_np),
                    "baseline_pcm_float32_sha256": _f32le_sha256(baseline_np),
                    "max_abs_difference": float(np.max(np.abs(zero_output_np - baseline_le))),
                }

                _set_parameters(head, parameter_probe, torch)
                with torch.no_grad():
                    result = head(source_tensor, baseline_tensor)
                arrays = _fit_row_arrays(result)
                raw_mask = arrays.pop("raw_mask_db_f32")
                projected_mask = arrays.pop("projected_mask_db_f32")
                raw_masks.append(raw_mask)
                projected_masks.append(projected_mask)
                frame_count = int(raw_mask.shape[1])
                if projected_mask.shape != raw_mask.shape:
                    raise ValueError(f"mask shape mismatch for {row['row_id']}")
                row_frame_offsets.append(row_frame_offsets[-1] + frame_count)
                for field in vector_fields:
                    fit_vectors[field].append(arrays[field])
                    array_bank_bytes += int(arrays[field].nbytes)
                for field in power_fields:
                    fit_power[field].append(arrays[field])
                    array_bank_bytes += int(arrays[field].nbytes)
                array_bank_bytes += int(raw_mask.nbytes) + int(projected_mask.nbytes)
                if array_bank_bytes > int(config["resource_limits"]["max_uncompressed_matrix_bytes"]):
                    raise MemoryError("fit matrix bank exceeded configured uncompressed byte bound")

                normalization = result["normalization"]
                post_bound = float(result["postprojection_max_abs_mask_db"].item())
                output_np = _as_np(result["output_pcm_float32"], np.float32)
                base_peak = float(np.max(np.abs(baseline_np)))
                output_peak = float(np.max(np.abs(output_np)))
                target_rms = float(config["head_constructor"]["target_rms"])
                rms_error = float(normalization["final_output_rms_error_to_target"].item())
                mask_bound_ok = post_bound <= float(config["head_constructor"]["max_abs_mask_db"]) + float(
                    config["qualification_checks"]["mask_bound_tolerance_db"]
                )
                temporal = _temporal_diagnostics(
                    result,
                    torch,
                    float(config["qualification_checks"]["matrix_rank_tolerance"]),
                    float(config["qualification_checks"]["signed_gate_distinctness_diagnostic_threshold"]),
                )
                output_record = {
                    "source_sample_count": int(source_np.size),
                    "output_sample_count": int(output_np.size),
                    "output_pcm_float32_sha256": _f32le_sha256(output_np),
                    "output_finite": bool(np.isfinite(output_np).all()),
                    "final_output_rms": float(normalization["final_output_rms"].item()),
                    "final_output_rms_error_to_target": rms_error,
                    "final_output_rms_error_to_target_within_engineering_tolerance": abs(rms_error) <= float(
                        config["qualification_checks"]["final_rms_abs_tolerance"]
                    ),
                    "declared_target_rms": target_rms,
                    "final_output_peak": output_peak,
                    "baseline_peak": base_peak,
                    "output_peak_delta_to_baseline": output_peak - base_peak,
                    "declared_peak_ceiling": config["head_constructor"]["peak_ceiling"],
                    "final_peak_ceiling_excess": None,
                    "preprojection_max_abs_mask_db": float(result["preprojection_max_abs_mask_db"].item()),
                    "postprojection_max_abs_mask_db": post_bound,
                    "declared_mask_bound_db": float(config["head_constructor"]["max_abs_mask_db"]),
                    "mask_within_bound": mask_bound_ok,
                    "numpy_baseline": {
                        "input_rms": float(rms_info.input_rms),
                        "target_rms": float(rms_info.target_rms),
                        "applied_gain": float(rms_info.applied_gain),
                        "output_rms": float(rms_info.output_rms),
                        "peak_limited": bool(rms_info.peak_limited),
                    },
                    "torch_internal_baseline_match_max_abs": float(
                        normalization["canonical_baseline_max_abs_match_error"].item()
                    ),
                    "first_transition_invalid": temporal["first_transition_invalid"],
                    "valid_transition_count": temporal["valid_transition_count"],
                    "temporal_design_rank": temporal["temporal_design_rank"],
                    "positive_negative_gate_max_abs_difference": temporal[
                        "positive_negative_gate_max_abs_difference_on_valid_support"
                    ],
                    "temporal_diagnostics": temporal,
                    "matrix_frame_offset_start": row_frame_offsets[-2],
                    "matrix_frame_offset_stop": row_frame_offsets[-1],
                    "video_frame_offset_start": row_video_frame_offsets[-1],
                }
                video_frames = int(arrays["source_frame_power_f32"].size)
                row_video_frame_offsets.append(row_video_frame_offsets[-1] + video_frames)
                output_record["video_frame_offset_stop"] = row_video_frame_offsets[-1]
                record = {
                    "schema": RESULT_SCHEMA,
                    "ordinal": ordinal,
                    "row_id": row["row_id"],
                    "sample_id": row["sample_id"],
                    "speaker_id": row["speaker_id"],
                    "canonical_j": int(row["canonical_j"]),
                    "sample_rate_hz": int(config["audio_contract"]["sample_rate_hz"]),
                    "source_audio": source_receipt,
                    "zero_parameter_check": zero_record,
                    "diagnostic_nonzero_parameter_check": output_record,
                    "configuration_sha256": None,
                    "source_manifest_sha256": None,
                }
                record["configuration_sha256"] = config["_runtime_config_sha256"]
                record["source_manifest_sha256"] = manifest_pin["sha256"]
                written, _ = _append_jsonl_durable(
                    journal,
                    record,
                    max_bytes=int(config["resource_limits"]["max_jsonl_bytes"]),
                    current_bytes=records_bytes,
                )
                records_bytes += written
                result_records.append(record)
                del source_tensor, baseline_tensor, source_np, baseline_np, result, zero_result, output_np

        probe_result, probe_arrays = _gradient_diagnostics(head, apply_global_rms, config, torch)
        silence_count = int(config["synthetic_probe"]["silence_sample_count"])
        silence_np = np.zeros(silence_count, dtype=np.float32)
        silence_baseline_np, silence_rms = apply_global_rms(
            silence_np,
            float(config["head_constructor"]["target_rms"]),
            peak_ceiling=config["head_constructor"]["peak_ceiling"],
            silence_epsilon=float(config["numpy_rms_baseline"]["silence_epsilon"]),
        )
        silence_source_tensor = torch.from_numpy(silence_np)
        silence_baseline_tensor = torch.from_numpy(np.ascontiguousarray(silence_baseline_np, dtype=np.float32))
        with torch.no_grad():
            silence_result = head(silence_source_tensor, silence_baseline_tensor)
        silence_output = _as_np(silence_result["output_pcm_float32"], np.float32)
        silence_record = {
            "sample_count": silence_count,
            "input_exactly_zero": bool(np.count_nonzero(silence_np) == 0),
            "numpy_output_exactly_zero": bool(np.count_nonzero(silence_baseline_np) == 0),
            "head_output_exactly_equal_baseline": bool(np.array_equal(silence_output, silence_baseline_np)),
            "head_output_exactly_zero": bool(np.count_nonzero(silence_output) == 0),
            "output_finite": bool(np.isfinite(silence_output).all()),
            "valid_transition_count": int(silence_result["valid_transition_mask"].sum().item()),
            "temporal_design_rank": _temporal_diagnostics(
                silence_result,
                torch,
                float(config["qualification_checks"]["matrix_rank_tolerance"]),
                float(config["qualification_checks"]["signed_gate_distinctness_diagnostic_threshold"]),
            )["temporal_design_rank"],
            "numpy_rms_applied_gain": float(silence_rms.applied_gain),
            "numpy_peak_limited": bool(silence_rms.peak_limited),
        }

        matrix_arrays = {
            "fit_row_ids_utf8": np.asarray([row["row_id"] for row in manifest["rows"]], dtype="U64"),
            "fit_sample_ids_utf8": np.asarray([row["sample_id"] for row in manifest["rows"]], dtype="U32"),
            "fit_speaker_ids_utf8": np.asarray([row["speaker_id"] for row in manifest["rows"]], dtype="U16"),
            "fit_canonical_j_i64": np.asarray([row["canonical_j"] for row in manifest["rows"]], dtype=np.int64),
            "fit_stft_frame_offsets_i64": np.asarray(row_frame_offsets, dtype=np.int64),
            "fit_video_frame_offsets_i64": np.asarray(row_video_frame_offsets, dtype=np.int64),
            "fit_raw_mask_db_f32": np.concatenate(raw_masks, axis=1).astype(np.float32, copy=False),
            "fit_projected_mask_db_f32": np.concatenate(projected_masks, axis=1).astype(np.float32, copy=False),
            "fit_band_basis_f32": _as_np(head.band_basis, np.float32),
            "fit_probe_parameters_db_f32": np.asarray(parameter_probe, dtype=np.float32),
        }
        for field, parts in fit_vectors.items():
            matrix_arrays[f"fit_{field}"] = np.concatenate(parts).astype(
                np.uint8 if field.endswith("_u8") else (np.int64 if field.endswith("_i64") else np.float32), copy=False
            )
        for field, parts in fit_power.items():
            matrix_arrays[f"fit_{field}"] = np.concatenate(parts).astype(
                np.uint8 if field.endswith("_u8") else (np.int64 if field.endswith("_i64") else np.float32), copy=False
            )
        matrix_arrays.update(probe_arrays)
        matrix_arrays["fit_flux_normalization_scale_f32"] = np.asarray(
            [config["head_constructor"]["flux_normalization_scale"]], dtype=np.float32
        )
        matrix_arrays["fit_flux_denominator_floor_f32"] = np.asarray(
            [config["head_constructor"]["flux_denominator_floor"]], dtype=np.float32
        )
        matrix_arrays["fit_activity_floor_f32"] = np.asarray(
            [config["head_constructor"]["activity_floor"]], dtype=np.float32
        )

        uncompressed_bytes = _validate_array_bank_size(
            matrix_arrays, int(config["resource_limits"]["max_uncompressed_matrix_bytes"])
        )
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **matrix_arrays)
        npz_bytes = buffer.getvalue()
        if len(npz_bytes) > int(config["resource_limits"]["max_matrix_archive_bytes"]):
            raise MemoryError("compressed qualification matrix bank exceeded configured byte bound")
        if records_bytes + len(npz_bytes) + int(config["resource_limits"]["summary_manifest_reserve_bytes"]) > int(
            config["resource_limits"]["max_total_output_bytes"]
        ):
            raise MemoryError("qualification output would exceed configured total byte bound")
        matrix_path = output_dir / "qualification_matrices.npz"
        with matrix_path.open("xb") as stream:
            stream.write(npz_bytes)
            stream.flush()
            os.fsync(stream.fileno())

        partial_path.replace(output_dir / "results.jsonl")
        zero_all = all(record["zero_parameter_check"]["bitwise_equal_to_numpy_rms_baseline"] for record in result_records)
        first_transition_all = all(
            record["diagnostic_nonzero_parameter_check"]["first_transition_invalid"] is True
            and record["diagnostic_nonzero_parameter_check"]["temporal_diagnostics"]["first_positive_flux_placeholder"] == 0.0
            and record["diagnostic_nonzero_parameter_check"]["temporal_diagnostics"]["first_negative_flux_placeholder"] == 0.0
            and record["diagnostic_nonzero_parameter_check"]["temporal_diagnostics"]["first_positive_gate"] == 0.0
            and record["diagnostic_nonzero_parameter_check"]["temporal_diagnostics"]["first_negative_gate"] == 0.0
            for record in result_records
        )
        exact_lengths_all = all(
            record["diagnostic_nonzero_parameter_check"]["source_sample_count"]
            == record["diagnostic_nonzero_parameter_check"]["output_sample_count"]
            for record in result_records
        )
        masks_all = all(record["diagnostic_nonzero_parameter_check"]["mask_within_bound"] for record in result_records)
        rms_all = all(
            record["diagnostic_nonzero_parameter_check"]["final_output_rms_error_to_target_within_engineering_tolerance"]
            for record in result_records
        )
        coordinate_all = bool(probe_result["all_coordinate_agreement"])
        direction_ok = bool(probe_result["directional_agreement"])
        silence_ok = bool(silence_record["head_output_exactly_equal_baseline"] and silence_record["head_output_exactly_zero"])
        synthetic_first_transition_ok = bool(
            probe_result["nonzero_parameter_output_checks"]["first_transition_unsupported_with_zero_placeholders"]
        )
        hard_checks = {
            "all_40_zero_parameter_outputs_bitwise_equal_numpy_rms_baseline": zero_all,
            "all_40_first_transition_entries_unsupported_with_zero_placeholders": first_transition_all,
            "all_40_nonzero_outputs_exact_length": exact_lengths_all,
            "all_40_nonzero_outputs_finite": all(record["diagnostic_nonzero_parameter_check"]["output_finite"] for record in result_records),
            "all_40_projected_masks_within_declared_bound": masks_all,
            "all_40_final_rms_errors_within_engineering_tolerance": rms_all,
            "synthetic_silence_exact_zero_and_baseline": silence_ok,
            "synthetic_first_transition_unsupported_with_zero_placeholders": synthetic_first_transition_ok,
            "synthetic_parameter_gradients_finite": probe_result["gradient_all_finite"],
            "all_nine_coordinate_finite_differences_agree": coordinate_all,
            "directional_finite_difference_agrees": direction_ok,
        }
        summary = {
            "schema": SUMMARY_SCHEMA,
            "worker_schema": WORKER_SCHEMA,
            "qualification_status": "COMPLETE_ENGINEERING_DIAGNOSTICS",
            "candidate_selection_status": "NOT_A_CANDIDATE_OR_TRAINING_FREEZE",
            "configuration_sha256": config_sha,
            "source_manifest_sha256": manifest_pin["sha256"],
            "head_schema": HEAD_SCHEMA,
            "source_pins": source_pins,
            "runtime": runtime,
            "head_constructor": dict(config["head_constructor"]),
            "qualification_checks": dict(config["qualification_checks"]),
            "engineering_probe_specification": dict(config["synthetic_probe"]),
            "scope": dict(config["scope"]),
            "planned_fit_rows": len(manifest["rows"]),
            "completed_fit_rows": len(result_records),
            "speaker_count": len({record["speaker_id"] for record in result_records}),
            "zero_parameter_bitwise_equal_count": sum(
                bool(record["zero_parameter_check"]["bitwise_equal_to_numpy_rms_baseline"])
                for record in result_records
            ),
            "valid_transition_count_distribution": {
                "min": min(record["diagnostic_nonzero_parameter_check"]["valid_transition_count"] for record in result_records),
                "max": max(record["diagnostic_nonzero_parameter_check"]["valid_transition_count"] for record in result_records),
            },
            "temporal_design_rank_counts": {
                str(rank): sum(record["diagnostic_nonzero_parameter_check"]["temporal_design_rank"] == rank for record in result_records)
                for rank in range(4)
            },
            "synthetic_nonuniform_probe": probe_result,
            "synthetic_silence_probe": silence_record,
            "diagnostic_checks": hard_checks,
            "engineering_diagnostic_checks_all_pass": all(hard_checks.values()),
            "signed_gate_distinct_fit_row_count": sum(
                record["diagnostic_nonzero_parameter_check"]["temporal_diagnostics"]["positive_negative_gate_arrays_distinct_on_valid_support"]
                for record in result_records
            ),
            "signed_gate_distinct_synthetic_probe": probe_result["temporal_diagnostics"]["positive_negative_gate_arrays_distinct_on_valid_support"],
            "array_bank_uncompressed_bytes": uncompressed_bytes,
            "matrix_archive_bytes": len(npz_bytes),
            "matrix_archive_sha256": _sha256_bytes(npz_bytes),
            "jsonl_bytes": records_bytes,
            "max_observed_audio_sample_count": max(int(row["sample_count"]) for row in manifest["rows"]),
            "input_audio_in_memory_policy": "one source row at a time; no input/output waveform persisted",
            "elapsed_seconds": time.time() - started,
            "peak_rss_platform_units": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "peak_rss_units": "KiB on Linux; platform-specific otherwise",
        }
        summary_sha, summary_bytes = _write_json_atomic(output_dir / "summary.json", summary)
        file_entries = []
        for name in ("results.jsonl", "qualification_matrices.npz", "summary.json"):
            sha, size = _sha256_file(output_dir / name)
            file_entries.append({"path": name, "sha256": sha, "bytes": size})
        if sum(entry["bytes"] for entry in file_entries) > int(config["resource_limits"]["max_total_output_bytes"]):
            raise MemoryError("qualification outputs exceed configured total byte bound")
        terminal = {
            "schema": MANIFEST_SCHEMA,
            "status": "COMPLETE",
            "qualification_status": summary["qualification_status"],
            "engineering_diagnostic_checks_all_pass": summary["engineering_diagnostic_checks_all_pass"],
            "configuration_sha256": config_sha,
            "source_manifest_sha256": manifest_pin["sha256"],
            "files": file_entries,
            "completed_fit_rows": len(result_records),
            "scope": dict(config["scope"]),
        }
        terminal_sha, terminal_bytes = _write_json_atomic(output_dir / "result_manifest.json", terminal)
        if sum(entry["bytes"] for entry in file_entries) + terminal_bytes > int(config["resource_limits"]["max_total_output_bytes"]):
            raise MemoryError("terminal manifest pushes qualification outputs over configured total byte bound")
        print(json.dumps({
            "status": "COMPLETE",
            "output_dir": str(output_dir),
            "fit_rows": len(result_records),
            "engineering_diagnostic_checks_all_pass": summary["engineering_diagnostic_checks_all_pass"],
            "summary_sha256": summary_sha,
            "result_manifest_sha256": terminal_sha,
            "result_manifest_bytes": terminal_bytes,
        }, sort_keys=True))
        return 0 if summary["engineering_diagnostic_checks_all_pass"] else 2
    except Exception as exc:
        failure = {
            "schema": MANIFEST_SCHEMA,
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "configuration_sha256": config_sha,
            "source_manifest_sha256": manifest_pin["sha256"],
            "completed_journal_rows": len(result_records),
            "scope": dict(config["scope"]),
        }
        if output_dir.exists():
            _write_json_atomic(output_dir / "failure.json", failure)
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--source-root", required=True, help="directory containing the pinned manifest and source_wav members")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True, help="must be a fresh path")
    args = parser.parse_args()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
