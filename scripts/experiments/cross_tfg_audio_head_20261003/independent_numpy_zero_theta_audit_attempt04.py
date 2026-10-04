"""NumPy-only independent audit for the V3 zero-theta gradient supplement.

The consumer loads only sealed diagnostic archives, synthesizes the frozen
analytic probe in NumPy, and recomputes the V3 signal path through the pinned
independent NumPy consumer. It never imports the V3 producer/head or Torch and
never writes waveform samples.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


CONSUMER_SCHEMA = "cross_tfg_v3_zero_theta_numpy_independent_consumer_v4"
RESULT_SCHEMA = "cross_tfg_v3_zero_theta_numpy_independent_result_v4"
MANIFEST_SCHEMA = "cross_tfg_v3_zero_theta_numpy_independent_manifest_v4"
HEX = frozenset("0123456789abcdef")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> tuple[str, int]:
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


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _write_new(path: Path, content: bytes) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def _repo_file(repo_root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or not rel.parts or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError(f"unsafe repository-relative path: {relative}")
    current = repo_root
    if current.is_symlink():
        raise ValueError("repository root may not be a symlink")
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink in pinned repository source path: {relative}")
    if not current.is_file():
        raise FileNotFoundError(f"pinned source is absent or not regular: {relative}")
    return current


def _verify_source_pins(repo_root: Path, pins: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for name, pin in pins.items():
        expected_sha = str(pin["sha256"])
        expected_size = int(pin["bytes"])
        if len(expected_sha) != 64 or any(char not in HEX for char in expected_sha):
            raise ValueError(f"invalid SHA-256 syntax for source {name}")
        path = _repo_file(repo_root, str(pin["path"]))
        actual_sha, actual_size = _sha_file(path)
        observed[name] = {"path": str(pin["path"]), "bytes": actual_size, "sha256": actual_sha}
        if actual_sha != expected_sha or actual_size != expected_size:
            raise ValueError(f"source pin mismatch: {name}")
    return observed


def _inside_repo(repo_root: Path, value: str, *, must_exist: bool = True) -> Path:
    candidate = Path(value)
    path = candidate.resolve(strict=must_exist) if candidate.is_absolute() else (repo_root / candidate).resolve(strict=must_exist)
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("input/output path must be under the declared repository root") from exc
    if must_exist and (path.is_symlink() or not path.is_dir()):
        raise ValueError("input artifact path must be a regular directory without a symlink")
    return path


def _inside_repo_file(repo_root: Path, value: str) -> Path:
    candidate = Path(value)
    path = candidate.resolve(strict=True) if candidate.is_absolute() else (repo_root / candidate).resolve(strict=True)
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("config path must be under the declared repository root") from exc
    if path.is_symlink() or not path.is_file():
        raise ValueError("config path must be a regular file without a symlink")
    return path


def _verify_manifest_files(root: Path, manifest: Mapping[str, Any], file_key: str, max_file_bytes: int) -> dict[str, dict[str, Any]]:
    files = manifest.get(file_key)
    if not isinstance(files, list) or not files:
        raise ValueError(f"manifest has no {file_key} list")
    observed: dict[str, dict[str, Any]] = {}
    for entry in files:
        rel = str(entry["path"])
        pure = PurePosixPath(rel)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise ValueError(f"unsafe manifest member: {rel}")
        path = root.joinpath(*pure.parts)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"manifest member absent or unsafe: {rel}")
        digest, size = _sha_file(path)
        if size > max_file_bytes:
            raise MemoryError(f"manifest member exceeds configured input cap: {rel}")
        if digest != entry.get("sha256") or size != int(entry.get("bytes", -1)):
            raise ValueError(f"manifest member seal mismatch: {rel}")
        if rel in observed:
            raise ValueError(f"duplicate manifest member: {rel}")
        observed[rel] = {"bytes": size, "sha256": digest}
    return observed


def _load_npz(path: Path, *, max_compressed: int, max_uncompressed: int, numpy: Any) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("NPZ input absent or unsafe")
    if path.stat().st_size > max_compressed:
        raise MemoryError("NPZ input exceeds configured compressed-byte limit")
    with zipfile.ZipFile(path, "r") as archive:
        members = archive.infolist()
        if sum(int(member.file_size) for member in members) > max_uncompressed:
            raise MemoryError("NPZ input exceeds configured uncompressed-byte limit")
    with numpy.load(path, allow_pickle=False) as archive:
        return {key: numpy.asarray(archive[key]).copy() for key in archive.files}


def _load_pinned_consumer(repo_root: Path, relative_path: str, numpy: Any) -> Any:
    path = _repo_file(repo_root, relative_path)
    if "torch" in sys.modules:
        raise RuntimeError("Torch was imported before loading the NumPy consumer")
    spec = importlib.util.spec_from_file_location("pinned_v3_numpy_consumer", path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load the pinned NumPy consumer module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if "torch" in sys.modules:
        raise RuntimeError("pinned independent NumPy consumer imported Torch")
    for required in ("_forward_numpy", "_numpy_rms_baseline", "_objective", "_f32le_sha", "_compare_array", "_scalar_compare"):
        if not callable(getattr(module, required, None)):
            raise RuntimeError(f"pinned NumPy consumer is missing required helper {required}")
    return module


def _synthesize_waveform(probe: Mapping[str, Any], numpy: Any) -> Any:
    f32 = numpy.float32
    sample_rate = int(probe["sample_rate_hz"])
    time_axis = numpy.arange(int(probe["sample_count"]), dtype=numpy.float32) / f32(sample_rate)
    waveform = numpy.zeros(time_axis.shape, dtype=numpy.float32)
    for component in probe["components"]:
        env_phase = f32(component["envelope_phase_radians"])
        env_angular_frequency = f32(2.0 * math.pi * float(component["envelope_hz"]))
        env_arg = f32(env_angular_frequency * time_axis + env_phase)
        envelope = f32(component["envelope_floor"]) + f32(component["envelope_depth"]) * (
            f32(0.5) + f32(0.5) * numpy.sin(env_arg).astype(numpy.float32, copy=False)
        )
        step = component["amplitude_step"]
        if step is not None:
            multiplier = numpy.where(
                time_axis >= f32(step["time_seconds"]),
                f32(step["after_multiplier"]),
                f32(step["before_multiplier"]),
            ).astype(numpy.float32, copy=False)
            envelope = f32(envelope * multiplier)
        carrier_angular_frequency = f32(2.0 * math.pi * float(component["carrier_hz"]))
        carrier_arg = f32(carrier_angular_frequency * time_axis + f32(component["carrier_phase_radians"]))
        carrier = numpy.sin(carrier_arg).astype(numpy.float32, copy=False)
        waveform = f32(waveform + f32(f32(component["amplitude"]) * envelope * carrier))
    return numpy.ascontiguousarray(waveform, dtype=numpy.float32)


def _synthesize_weights(probe: Mapping[str, Any], count: int, numpy: Any) -> tuple[Any, Any]:
    f32 = numpy.float32
    sr = f32(probe["sample_rate_hz"])
    time_axis = numpy.arange(count, dtype=numpy.float32) / sr
    objective = probe["objective"]
    linear = numpy.zeros(count, dtype=numpy.float32)
    for term in objective["linear_terms"]:
        angular_frequency = f32(2.0 * math.pi * float(term["frequency_hz"]))
        angle = f32(angular_frequency * time_axis + f32(term["phase_radians"]))
        linear = f32(linear + f32(f32(term["coefficient"]) * numpy.sin(angle).astype(numpy.float32, copy=False)))
    quadratic = numpy.full(count, f32(objective["quadratic_weight_offset"]), dtype=numpy.float32)
    for term in objective["quadratic_weight_terms"]:
        angular_frequency = f32(2.0 * math.pi * float(term["frequency_hz"]))
        angle = f32(angular_frequency * time_axis + f32(term["phase_radians"]))
        quadratic = f32(quadratic + f32(f32(term["coefficient"]) * numpy.cos(angle).astype(numpy.float32, copy=False)))
    return numpy.ascontiguousarray(linear), numpy.ascontiguousarray(quadratic)


def _array_compare(actual: Any, expected: Any, tolerances: Mapping[str, Any], kind: str, numpy: Any) -> dict[str, Any]:
    a = numpy.asarray(actual)
    e = numpy.asarray(expected)
    if a.shape != e.shape:
        return {"shape_match": False, "actual_shape": list(a.shape), "expected_shape": list(e.shape), "within_tolerance": False}
    if a.dtype.kind in "fc" and (not numpy.isfinite(a).all() or not numpy.isfinite(e).all()):
        return {"shape_match": True, "finite": False, "within_tolerance": False}
    if kind == "exact":
        equal = bool(numpy.array_equal(a, e))
        return {"shape_match": True, "exact_equal": equal, "max_abs_error": 0.0 if equal else float(numpy.max(numpy.abs(a.astype(numpy.float64) - e.astype(numpy.float64)))), "within_tolerance": equal}
    tol = tolerances[kind]
    error = numpy.abs(a.astype(numpy.float64) - e.astype(numpy.float64))
    allowed = float(tol["atol"]) + float(tol["rtol"]) * numpy.abs(e.astype(numpy.float64))
    return {
        "shape_match": True,
        "finite": True,
        "max_abs_error": float(numpy.max(error)) if error.size else 0.0,
        "rms_error": float(numpy.sqrt(numpy.mean(error * error))) if error.size else 0.0,
        "max_allowed_error": float(numpy.max(allowed)) if allowed.size else float(tol["atol"]),
        "within_tolerance": bool(numpy.all(error <= allowed)),
    }


def _finite_json(value: Any, numpy: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _finite_json(item, numpy) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item, numpy) for item in value]
    if isinstance(value, (numpy.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (numpy.integer, int)):
        return int(value)
    if isinstance(value, (numpy.bool_, bool)):
        return bool(value)
    return value


def _run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve(strict=True)
    config_path = _inside_repo_file(repo_root, args.config)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    if config.get("schema") != "cross_tfg_v3_zero_theta_numpy_independent_config_v4":
        raise ValueError("unsupported independent zero-theta config schema")
    if config_path != (repo_root / config["config_relative_path"]).resolve(strict=True):
        raise ValueError("independent config path does not match its declared relative path")
    source_pins = _verify_source_pins(repo_root, config["source_pins"])
    runtime = config["runtime"]
    if list(sys.version_info[:3]) != runtime["python_version"]:
        raise RuntimeError("Python version differs from the pinned independent NumPy runtime")
    if Path(sys.executable).resolve() != Path(runtime["python_executable"]).resolve():
        raise RuntimeError("interpreter path differs from the pinned independent NumPy runtime")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE must equal 1")
    for key, value in runtime["environment"].items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"CPU-only NumPy runtime environment mismatch: {key}")

    import numpy as np

    if np.__version__ != runtime["numpy_version"]:
        raise RuntimeError("NumPy version differs from the frozen audit runtime")
    if "torch" in sys.modules:
        raise RuntimeError("Torch was imported before the NumPy-only audit")
    independent = _load_pinned_consumer(repo_root, config["independent_signal_path"], np)
    if "torch" in sys.modules:
        raise RuntimeError("Torch was imported while loading the independent signal implementation")

    cpu_root = _inside_repo(repo_root, args.cpu02_output_dir)
    supplement_root = _inside_repo(repo_root, args.zero_supplement_output_dir)
    output_root = _inside_repo(repo_root, args.output_dir, must_exist=False)
    if output_root.exists():
        raise FileExistsError("independent audit output exists; refusing overwrite")

    max_input_file = int(config["resource_limits"]["max_input_file_bytes"])
    cpu_manifest_path = cpu_root / "result_manifest.json"
    zero_manifest_path = supplement_root / "result_manifest.json"
    cpu_manifest_sha, cpu_manifest_size = _sha_file(cpu_manifest_path)
    zero_manifest_sha, zero_manifest_size = _sha_file(zero_manifest_path)
    cpu_manifest = json.loads(cpu_manifest_path.read_text(encoding="utf-8"))
    zero_manifest = json.loads(zero_manifest_path.read_text(encoding="utf-8"))
    if cpu_manifest.get("status") != "COMPLETE" or cpu_manifest.get("schema") != "cross_tfg_shared_recipient_transition_head_v3_cpu_qualification_manifest_v1":
        raise ValueError("CPU02 qualification bundle is not a sealed COMPLETE artifact")
    accepted_zero_statuses = {"COMPLETE_ENGINEERING_CHECKS_PASS", "COMPLETE_WITH_HARD_CHECK_FAILURE", "COMPLETE_WITH_FINITE_DIFFERENCE_DISCREPANCY"}
    if zero_manifest.get("schema") != "cross_tfg_v3_zero_theta_gradient_supplement_manifest_v1":
        raise ValueError("zero-theta supplement manifest schema mismatch")
    if zero_manifest.get("status") not in accepted_zero_statuses:
        raise ValueError("zero-theta manifest status is not a recognized terminal result")
    cpu_files = _verify_manifest_files(cpu_root, cpu_manifest, "files", max_input_file)
    zero_files = _verify_manifest_files(supplement_root, zero_manifest, "output_files", max_input_file)
    expected_cpu_names = {"results.jsonl", "qualification_matrices.npz", "summary.json"}
    expected_zero_names = {"zero_theta_diagnostic_arrays.npz", "result.json"}
    if set(cpu_files) != expected_cpu_names or set(zero_files) != expected_zero_names:
        raise ValueError("sealed input bundle file roster differs from the declared output contract")
    if int(cpu_manifest.get("completed_fit_rows", -1)) != int(config["expected_cpu02_fit_rows"]):
        raise ValueError("CPU02 fit row count differs from the pinned qualification config")
    if zero_manifest.get("configuration_sha256") != source_pins["zero_supplement_config"]["sha256"]:
        raise ValueError("zero supplement manifest was produced from a different config")
    if zero_manifest.get("source_pins", {}).get("qualification_config", {}).get("sha256") != source_pins["qualification_config02"]["sha256"]:
        raise ValueError("zero supplement manifest does not bind the expected CPU02 config")

    summary_path = cpu_root / "summary.json"
    cpu_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    qcfg_path = _repo_file(repo_root, config["source_pins"]["qualification_config02"]["path"])
    qcfg_bytes = qcfg_path.read_bytes()
    qcfg = json.loads(qcfg_bytes.decode("utf-8"))
    if _sha_bytes(qcfg_bytes) != source_pins["qualification_config02"]["sha256"]:
        raise ValueError("frozen attempt02 config hash mismatch")
    if cpu_manifest.get("configuration_sha256") != source_pins["qualification_config02"]["sha256"]:
        raise ValueError("CPU02 artifact manifest config binding mismatch")
    if cpu_summary.get("configuration_sha256") != source_pins["qualification_config02"]["sha256"]:
        raise ValueError("CPU02 artifact summary config binding mismatch")
    zero_qcfg_path = _repo_file(repo_root, config["source_pins"]["zero_supplement_config"]["path"])
    zero_qcfg_bytes = zero_qcfg_path.read_bytes()
    zero_qcfg = json.loads(zero_qcfg_bytes.decode("utf-8"))
    if _sha_bytes(zero_qcfg_bytes) != source_pins["zero_supplement_config"]["sha256"]:
        raise ValueError("frozen zero-theta supplement config hash mismatch")
    scope = cpu_summary.get("scope", {})
    if not scope.get("historical_fit_only") or scope.get("development_audio_read") is not False or scope.get("confirmation_audio_read") is not False:
        raise ValueError("CPU02 source scope is not the expected fit-only closed scope")

    max_compressed = int(config["resource_limits"]["max_input_npz_bytes"])
    max_uncompressed = int(config["resource_limits"]["max_input_npz_uncompressed_bytes"])
    cpu_arrays = _load_npz(cpu_root / "qualification_matrices.npz", max_compressed=max_compressed, max_uncompressed=max_uncompressed, numpy=np)
    zero_arrays = _load_npz(supplement_root / "zero_theta_diagnostic_arrays.npz", max_compressed=max_compressed, max_uncompressed=max_uncompressed, numpy=np)
    zero_result = json.loads((supplement_root / "result.json").read_text(encoding="utf-8"))
    if zero_result.get("schema") != "cross_tfg_v3_zero_theta_gradient_supplement_result_v1":
        raise ValueError("zero-theta result schema mismatch")
    if zero_result.get("status") != zero_manifest.get("status"):
        raise ValueError("zero-theta result and manifest terminal statuses differ")
    if zero_result.get("status") not in accepted_zero_statuses:
        raise ValueError("zero-theta result status is not a recognized terminal result")
    if zero_result.get("scope") != zero_qcfg.get("scope"):
        raise ValueError("zero-theta result scope differs from its pinned config")
    zero_worker_sources = _verify_source_pins(repo_root, zero_qcfg.get("source_pins", {}))
    if zero_result.get("pinned_sources") != zero_worker_sources:
        raise ValueError("zero-theta result source observations differ from the pinned worker config")
    if zero_manifest.get("source_pins") != zero_worker_sources:
        raise ValueError("zero-theta manifest source observations differ from the pinned worker config")
    if zero_result.get("qualification_config_sha256") != source_pins["zero_supplement_config"]["sha256"]:
        raise ValueError("zero-theta result does not bind the expected supplement config")
    if zero_result.get("pinned_sources", {}).get("qualification_config", {}).get("sha256") != source_pins["qualification_config02"]["sha256"]:
        raise ValueError("zero-theta result does not bind frozen CPU02 config")

    required_cpu = {
        "synthetic_source_float32",
        "synthetic_baseline_float32",
        "synthetic_loss_weight_float32",
        "synthetic_quadratic_weight_float32",
    }
    required_zero = {
        "activity_cells_u8",
        "autograd_gradient_f64",
        "coordinate_finite_difference_f64",
        "coordinate_absolute_error_f64",
        "coordinate_allowed_error_f64",
        "coordinate_agreement_u8",
        "direction_matrix_normalized_f64",
        "valid_transition_mask_u8",
        "active_frame_mask_u8",
        "activity_cells_u8",
        "positive_relative_flux_f32",
        "negative_relative_flux_f32",
        "positive_flux_gate_f32",
        "negative_flux_gate_f32",
        "raw_log_magnitude_mask_db_f32",
        "smoothed_log_magnitude_mask_db_f32",
        "projected_log_magnitude_mask_db_f32",
        "band_basis_f32",
        "source_frame_power_f32",
        "baseline_frame_power_f32",
        "output_frame_power_f32",
        "stft_frame_centers_source_samples_i64",
        "video_frame_centers_source_samples_i64",
        "directional_finite_difference_f64",
        "directional_autograd_f64",
        "directional_absolute_error_f64",
        "directional_allowed_error_f64",
        "directional_agreement_u8",
    }
    if not required_cpu.issubset(cpu_arrays) or not required_zero.issubset(zero_arrays):
        raise ValueError("one of the sealed input NPZs is missing required independent-audit arrays")

    source_saved = np.asarray(cpu_arrays["synthetic_source_float32"], dtype=np.float32)
    baseline_saved = np.asarray(cpu_arrays["synthetic_baseline_float32"], dtype=np.float32)
    if source_saved.ndim != 1 or source_saved.size != int(qcfg["synthetic_probe"]["sample_count"]):
        raise ValueError("CPU02 synthetic source shape does not match its frozen config")
    if source_saved.size > int(config["resource_limits"]["max_probe_samples"]):
        raise MemoryError("synthetic source exceeds the independent audit sample cap")
    if not np.isfinite(source_saved).all() or not np.isfinite(baseline_saved).all():
        raise FloatingPointError("CPU02 synthetic waveform or baseline contains nonfinite values")
    source_analytic = _synthesize_waveform(qcfg["synthetic_probe"], np)
    if source_analytic.shape != source_saved.shape:
        raise ValueError("independent analytic source shape differs from CPU02 saved source")
    analytic_diff = source_analytic.astype(np.float64) - source_saved.astype(np.float64)
    analytic_source_comparison = {
        "shape_match": True,
        "exact_sample_equal": bool(np.array_equal(source_analytic, source_saved)),
        "numpy_generated_f32le_sha256": independent._f32le_sha(source_analytic, np),
        "cpu02_saved_source_f32le_sha256": independent._f32le_sha(source_saved, np),
        "max_abs_difference": float(np.max(np.abs(analytic_diff))),
        "rms_difference": float(np.sqrt(np.mean(analytic_diff * analytic_diff))),
        "difference_count_nonzero": int(np.count_nonzero(analytic_diff)),
        "acceptance_threshold": None,
        "interpretation": "Recorded as a synthesis cross-check; no post hoc tolerance is imposed.",
    }
    baseline_np, baseline_info = independent._numpy_rms_baseline(
        source_saved,
        float(qcfg["numpy_rms_baseline"]["target_rms"]),
        qcfg["numpy_rms_baseline"]["peak_ceiling"],
        float(qcfg["numpy_rms_baseline"]["silence_epsilon"]),
        np,
    )
    baseline_np = np.ascontiguousarray(baseline_np, dtype=np.float32)
    analytic_baseline, analytic_baseline_info = independent._numpy_rms_baseline(
        source_analytic,
        float(qcfg["numpy_rms_baseline"]["target_rms"]),
        qcfg["numpy_rms_baseline"]["peak_ceiling"],
        float(qcfg["numpy_rms_baseline"]["silence_epsilon"]),
        np,
    )
    analytic_baseline = np.ascontiguousarray(analytic_baseline, dtype=np.float32)
    baseline_hash = independent._f32le_sha(baseline_np, np)
    saved_baseline_hash = independent._f32le_sha(baseline_saved, np)
    analytic_baseline_hash = independent._f32le_sha(analytic_baseline, np)
    expected_baseline_hash = zero_result["probe"]["baseline_pcm_float32_sha256"]
    cpu_probe_baseline_hash = cpu_summary["synthetic_nonuniform_probe"]["baseline_pcm_sha256"]
    baseline_comparison = {
        "independent_numpy_baseline_hash": baseline_hash,
        "cpu02_saved_baseline_hash": saved_baseline_hash,
        "zero_supplement_baseline_hash": expected_baseline_hash,
        "cpu02_summary_baseline_hash": cpu_probe_baseline_hash,
        "independently_synthesized_source_baseline_hash": analytic_baseline_hash,
        "recomputed_array_equals_cpu02_saved_baseline": bool(np.array_equal(baseline_np, baseline_saved)),
        "analytic_source_baseline_equals_cpu02_saved_baseline": bool(np.array_equal(analytic_baseline, baseline_saved)),
        "hash_matches_zero_supplement": baseline_hash == expected_baseline_hash,
        "hash_matches_cpu02_summary": baseline_hash == cpu_probe_baseline_hash,
        "hash_matches_cpu02_saved_baseline": baseline_hash == saved_baseline_hash,
        "baseline_info": baseline_info,
    }

    zero_theta = np.zeros((3, 3), dtype=np.float32)
    recomputed = independent._forward_numpy(source_saved, baseline_np, zero_theta, qcfg, np)
    recomputed_zero_output_hash = independent._f32le_sha(recomputed["output_pcm_f32"], np)
    recomputed_zero_output_max_abs = float(np.max(np.abs(recomputed["output_pcm_f32"] - baseline_np)))
    analytic_zero = independent._forward_numpy(source_analytic, analytic_baseline, zero_theta, qcfg, np)
    analytic_zero_output_hash = independent._f32le_sha(analytic_zero["output_pcm_f32"], np)
    analytic_zero_output_max_abs = float(np.max(np.abs(analytic_zero["output_pcm_f32"] - analytic_baseline)))
    tolerances = config["comparison_tolerances"]
    array_pairs = (
        ("valid_transition_mask_u8", "valid_transition_mask_u8", "exact"),
        ("active_frame_mask_u8", "active_frame_mask_u8", "exact"),
        ("positive_relative_flux_f32", "positive_relative_flux_f32", "flux"),
        ("negative_relative_flux_f32", "negative_relative_flux_f32", "flux"),
        ("positive_gate_f32", "positive_flux_gate_f32", "gate"),
        ("negative_gate_f32", "negative_flux_gate_f32", "gate"),
        ("raw_mask_db_f32", "raw_log_magnitude_mask_db_f32", "mask"),
        ("smoothed_mask_db_f32", "smoothed_log_magnitude_mask_db_f32", "mask"),
        ("projected_mask_db_f32", "projected_log_magnitude_mask_db_f32", "mask"),
        ("band_basis_f32", "band_basis_f32", "exact"),
        ("stft_frame_centers_source_samples_i64", "stft_frame_centers_source_samples_i64", "exact"),
        ("physical.source_frame_power_f32", "source_frame_power_f32", "power"),
        ("physical.baseline_frame_power_f32", "baseline_frame_power_f32", "power"),
        ("physical.output_frame_power_f32", "output_frame_power_f32", "power"),
        ("physical.video_frame_centers_source_samples_i64", "video_frame_centers_source_samples_i64", "exact"),
    )
    zero_array_checks: dict[str, Any] = {}
    independent_npz_arrays: dict[str, Any] = {
        "theta_zero_f32": zero_theta,
        "recomputed_positive_relative_flux_f32": recomputed["positive_relative_flux_f32"],
        "recomputed_negative_relative_flux_f32": recomputed["negative_relative_flux_f32"],
        "recomputed_positive_gate_f32": recomputed["positive_gate_f32"],
        "recomputed_negative_gate_f32": recomputed["negative_gate_f32"],
        "recomputed_raw_mask_db_f32": recomputed["raw_mask_db_f32"],
        "recomputed_smoothed_mask_db_f32": recomputed["smoothed_mask_db_f32"],
        "recomputed_projected_mask_db_f32": recomputed["projected_mask_db_f32"],
        "recomputed_valid_transition_mask_u8": recomputed["valid_transition_mask_u8"].astype(np.uint8),
        "recomputed_active_frame_mask_u8": recomputed["active_frame_mask_u8"].astype(np.uint8),
        "recomputed_activity_cells_u8": (recomputed["magnitude"] > np.float32(qcfg["head_constructor"]["activity_floor"])).astype(np.uint8),
        "worker_activity_cells_u8": np.asarray(zero_arrays["activity_cells_u8"], dtype=np.uint8),
        "recomputed_band_basis_f32": recomputed["band_basis_f32"],
        "recomputed_source_frame_power_f32": recomputed["physical"]["source_frame_power_f32"],
        "recomputed_baseline_frame_power_f32": recomputed["physical"]["baseline_frame_power_f32"],
        "recomputed_output_frame_power_f32": recomputed["physical"]["output_frame_power_f32"],
        "recomputed_stft_frame_centers_source_samples_i64": recomputed["stft_frame_centers_source_samples_i64"],
        "recomputed_video_frame_centers_source_samples_i64": recomputed["physical"]["video_frame_centers_source_samples_i64"],
    }
    for local_key, zero_key, tolerance_name in array_pairs:
        actual = recomputed
        for part in local_key.split("."):
            actual = actual[part]
        kind = tolerance_name
        if tolerance_name == "exact":
            comparison = _array_compare(actual, zero_arrays[zero_key], tolerances, "exact", np)
        else:
            comparison = _array_compare(actual, zero_arrays[zero_key], tolerances, kind, np)
        zero_array_checks[zero_key] = comparison
    recomputed_activity = (recomputed["magnitude"] > np.float32(qcfg["head_constructor"]["activity_floor"])).astype(np.uint8)
    zero_array_checks["activity_cells_u8"] = _array_compare(
        recomputed_activity, zero_arrays["activity_cells_u8"], tolerances, "exact", np
    )
    zero_array_checks["activity_cells_u8_source_support"] = {
        "shape": list(recomputed_activity.shape),
        "active_cell_count": int(recomputed_activity.sum()),
        "active_frequency_bin_count": int(np.any(recomputed_activity, axis=1).sum()),
        "valid_transition_count": int(recomputed["valid_transition_mask_u8"].sum()),
        "worker_saved_array": True,
    }

    zero_output_hash = zero_result["probe"]["output_pcm_float32_sha256"]
    zero_output_max_abs = float(zero_result["probe"]["zero_output_max_abs_error"])
    baseline_hash_match = baseline_hash == expected_baseline_hash == cpu_probe_baseline_hash
    zero_baseline_check = {
        "cpu02_saved_baseline_array_equals_recomputed_baseline": bool(np.array_equal(baseline_saved, baseline_np)),
        "independent_baseline_hash_matches_cpu02_and_zero_supplement": baseline_hash_match,
        "zero_supplement_output_hash_equals_baseline_hash": zero_output_hash == baseline_hash,
        "independent_numpy_zero_output_hash": recomputed_zero_output_hash,
        "independent_numpy_zero_output_hash_equals_baseline": recomputed_zero_output_hash == baseline_hash,
        "independent_numpy_zero_output_exactly_equals_baseline": bool(np.array_equal(recomputed["output_pcm_f32"], baseline_np)),
        "independent_numpy_zero_output_max_abs_error": recomputed_zero_output_max_abs,
        "analytic_source_numpy_zero_output_hash": analytic_zero_output_hash,
        "analytic_source_numpy_zero_output_hash_equals_analytic_baseline": analytic_zero_output_hash == analytic_baseline_hash,
        "analytic_source_numpy_zero_output_max_abs_error": analytic_zero_output_max_abs,
        "zero_supplement_reported_max_abs_error": zero_output_max_abs,
        "zero_supplement_reported_output_equal": bool(zero_result["hard_checks"]["zero_theta_output_bitwise_equal_to_numpy_rms_baseline"]),
        "independent_zero_theta_output_is_baseline_by_v3_residual_contract": True,
    }

    weights, quadratic = _synthesize_weights(qcfg["synthetic_probe"], int(source_saved.size), np)
    weight_saved = np.asarray(cpu_arrays["synthetic_loss_weight_float32"], dtype=np.float32)
    quadratic_saved = np.asarray(cpu_arrays["synthetic_quadratic_weight_float32"], dtype=np.float32)
    objective_weight_checks = {
        "independent_linear_weight_vs_cpu02": _array_compare(weights, weight_saved, tolerances, "fft", np),
        "independent_quadratic_weight_vs_cpu02": _array_compare(quadratic, quadratic_saved, tolerances, "fft", np),
    }
    zero_objective_analytic, _ = independent._objective(source_analytic, analytic_baseline, weights, quadratic, zero_theta, qcfg, np)
    zero_objective_saved_source, _ = independent._objective(source_saved, baseline_np, weights, quadratic, zero_theta, qcfg, np)
    objective_value_check = independent._scalar_compare(
        zero_objective_analytic,
        float(zero_result["probe"]["objective_value_at_zero"]),
        float(tolerances["scalar"]["atol"]),
        float(tolerances["scalar"]["rtol"]),
    )
    saved_source_objective_value_check = independent._scalar_compare(
        zero_objective_saved_source,
        float(zero_result["probe"]["objective_value_at_zero"]),
        float(tolerances["scalar"]["atol"]),
        float(tolerances["scalar"]["rtol"]),
    )

    measured_gradient = np.asarray(zero_arrays["autograd_gradient_f64"], dtype=np.float64)
    result_gradient = np.asarray(zero_result["probe"]["gradient_matrix_f64"], dtype=np.float64)
    gradient_result_check = _array_compare(measured_gradient, result_gradient, tolerances, "scalar", np)
    fd = qcfg["synthetic_probe"]["finite_difference"]
    coordinate_step = float(fd["coordinate_step_db"])
    def _coordinate_fd(signal: Any, base: Any) -> Any:
        output = np.empty((3, 3), dtype=np.float64)
        for flat_index in range(9):
            plus = zero_theta.copy().reshape(-1)
            minus = zero_theta.copy().reshape(-1)
            plus[flat_index] += np.float32(coordinate_step)
            minus[flat_index] -= np.float32(coordinate_step)
            plus_loss, _ = independent._objective(signal, base, weights, quadratic, plus.reshape(3, 3), qcfg, np)
            minus_loss, _ = independent._objective(signal, base, weights, quadratic, minus.reshape(3, 3), qcfg, np)
            output.reshape(-1)[flat_index] = (plus_loss - minus_loss) / (2.0 * coordinate_step)
        return output

    numpy_fd = _coordinate_fd(source_analytic, analytic_baseline)
    saved_source_numpy_fd = _coordinate_fd(source_saved, baseline_np)
    saved_fd = np.asarray(zero_arrays["coordinate_finite_difference_f64"], dtype=np.float64)
    agreement_tolerances = config["comparison_tolerances"]["finite_difference"]
    coordinate_error = np.abs(numpy_fd - measured_gradient)
    coordinate_allowed = float(agreement_tolerances["atol"]) + float(agreement_tolerances["rtol"]) * np.maximum(np.abs(numpy_fd), np.abs(measured_gradient))
    coordinate_rows = []
    for row in range(3):
        for col in range(3):
            coordinate_rows.append({
                "temporal_component_index": row,
                "nonreference_band_index": col,
                "numpy_fd": float(numpy_fd[row, col]),
                "saved_source_numpy_fd": float(saved_source_numpy_fd[row, col]),
                "worker_saved_fd": float(saved_fd[row, col]),
                "torch_autograd": float(measured_gradient[row, col]),
                "numpy_minus_autograd": float(numpy_fd[row, col] - measured_gradient[row, col]),
                "allowed_error": float(coordinate_allowed[row, col]),
                "numpy_fd_vs_autograd_within_tolerance": bool(coordinate_error[row, col] <= coordinate_allowed[row, col]),
                "saved_source_numpy_fd_vs_autograd": float(saved_source_numpy_fd[row, col] - measured_gradient[row, col]),
                "analytic_vs_saved_source_fd_difference": float(numpy_fd[row, col] - saved_source_numpy_fd[row, col]),
                "numpy_fd_vs_worker_fd": _array_compare(np.asarray([numpy_fd[row, col]]), np.asarray([saved_fd[row, col]]), tolerances, "finite_difference", np),
            })
    raw_direction = np.asarray(fd["direction_matrix"], dtype=np.float64)
    direction = raw_direction / float(np.linalg.norm(raw_direction))
    direction_saved = np.asarray(zero_arrays["direction_matrix_normalized_f64"], dtype=np.float64)
    step = float(fd["directional_step_db"])
    plus_loss, _ = independent._objective(source_analytic, analytic_baseline, weights, quadratic, (step * direction).astype(np.float32), qcfg, np)
    minus_loss, _ = independent._objective(source_analytic, analytic_baseline, weights, quadratic, (-step * direction).astype(np.float32), qcfg, np)
    directional_numpy_fd = float((plus_loss - minus_loss) / (2.0 * step))
    plus_saved_loss, _ = independent._objective(source_saved, baseline_np, weights, quadratic, (step * direction).astype(np.float32), qcfg, np)
    minus_saved_loss, _ = independent._objective(source_saved, baseline_np, weights, quadratic, (-step * direction).astype(np.float32), qcfg, np)
    saved_source_directional_fd = float((plus_saved_loss - minus_saved_loss) / (2.0 * step))
    directional_autograd = float(np.sum(measured_gradient * direction))
    directional_error = abs(directional_numpy_fd - directional_autograd)
    directional_allowed = float(agreement_tolerances["atol"]) + float(agreement_tolerances["rtol"]) * max(abs(directional_numpy_fd), abs(directional_autograd))
    saved_directional_error = abs(saved_source_directional_fd - directional_autograd)
    saved_directional_allowed = float(agreement_tolerances["atol"]) + float(agreement_tolerances["rtol"]) * max(abs(saved_source_directional_fd), abs(directional_autograd))
    directional = {
        "normalized_direction_matrix": direction.tolist(),
        "worker_saved_direction_matrix": direction_saved.tolist(),
        "direction_matrix_vs_worker": _array_compare(direction, direction_saved, tolerances, "finite_difference", np),
        "numpy_directional_fd": directional_numpy_fd,
        "saved_source_numpy_directional_fd": saved_source_directional_fd,
        "worker_saved_directional_fd": float(zero_arrays["directional_finite_difference_f64"].reshape(-1)[0]),
        "torch_autograd_directional_derivative": directional_autograd,
        "worker_saved_autograd_directional_derivative": float(zero_arrays["directional_autograd_f64"].reshape(-1)[0]),
        "absolute_error_numpy_fd_vs_torch_autograd": directional_error,
        "allowed_error": directional_allowed,
        "numpy_fd_vs_torch_autograd_within_tolerance": bool(directional_error <= directional_allowed),
        "saved_source_numpy_vs_torch_autograd_absolute_error": saved_directional_error,
        "saved_source_numpy_vs_torch_autograd_allowed_error": saved_directional_allowed,
        "saved_source_numpy_vs_torch_autograd_within_tolerance": bool(saved_directional_error <= saved_directional_allowed),
        "numpy_directional_fd_vs_worker_fd": _array_compare(
            np.asarray([directional_numpy_fd]),
            zero_arrays["directional_finite_difference_f64"].reshape(-1),
            tolerances,
            "finite_difference",
            np,
        ),
    }

    independent_npz = {
        **independent_npz_arrays,
        "independent_autograd_comparison_gradient_f64": measured_gradient,
        "independent_numpy_coordinate_fd_f64": numpy_fd,
        "saved_source_numpy_coordinate_fd_f64": saved_source_numpy_fd,
        "worker_coordinate_fd_f64": saved_fd,
        "coordinate_fd_minus_autograd_f64": numpy_fd - measured_gradient,
        "direction_matrix_normalized_f64": direction,
        "directional_numpy_fd_f64": np.asarray([directional_numpy_fd], dtype=np.float64),
        "saved_source_directional_numpy_fd_f64": np.asarray([saved_source_directional_fd], dtype=np.float64),
        "directional_torch_autograd_f64": np.asarray([directional_autograd], dtype=np.float64),
    }
    for name, array in independent_npz.items():
        if not np.isfinite(np.asarray(array)).all():
            raise FloatingPointError(f"independent output diagnostic is nonfinite: {name}")
    output_root.mkdir(parents=True, exist_ok=False)
    arrays_path = output_root / "independent_zero_theta_arrays.npz"
    with arrays_path.open("xb") as stream:
        np.savez_compressed(stream, **independent_npz)
        stream.flush()
        os.fsync(stream.fileno())
    arrays_sha, arrays_size = _sha_file(arrays_path)
    if arrays_size > int(config["resource_limits"]["max_output_npz_bytes"]):
        raise MemoryError("independent arrays exceeded configured compressed output bound")

    all_support_checks = all(bool(value.get("within_tolerance", True)) for value in zero_array_checks.values() if isinstance(value, dict))
    result = {
        "schema": RESULT_SCHEMA,
        "status": "COMPLETE_WITH_MEASURED_COMPARISONS",
        "scope": config["scope"],
        "configuration_sha256": _sha_bytes(config_bytes),
        "source_pins": source_pins,
        "input_bundles": {
            "cpu02_manifest_raw_sha256": cpu_manifest_sha,
            "cpu02_manifest_bytes": cpu_manifest_size,
            "zero_supplement_manifest_raw_sha256": zero_manifest_sha,
            "zero_supplement_manifest_bytes": zero_manifest_size,
            "cpu02_result_manifest_files": cpu_files,
            "zero_supplement_manifest_files": zero_files,
        },
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "torch_imported": "torch" in sys.modules},
        "analytic_source_synthesis": analytic_source_comparison,
        "baseline_hash_comparison": baseline_comparison,
        "zero_theta_baseline_check": zero_baseline_check,
        "objective_weight_checks": objective_weight_checks,
        "objective_at_zero": {
            "independent_numpy_analytic_source": zero_objective_analytic,
            "independent_numpy_cpu02_saved_source": zero_objective_saved_source,
            "worker_saved": float(zero_result["probe"]["objective_value_at_zero"]),
            "analytic_source_vs_worker_comparison": objective_value_check,
            "saved_source_vs_worker_comparison": saved_source_objective_value_check,
        },
        "zero_array_checks": zero_array_checks,
        "all_saved_gate_mask_power_comparisons_within_declared_tolerances": all_support_checks,
        "gradient": {
            "torch_autograd_matrix_f64": measured_gradient.tolist(),
            "finite": bool(np.isfinite(measured_gradient).all()),
            "nonzero_scalar_count": int(np.count_nonzero(measured_gradient)),
            "worker_result_vs_npz": gradient_result_check,
            "coordinate_finite_difference_checks": coordinate_rows,
            "coordinate_numpy_fd_vs_autograd_agreement_count": int(sum(row["numpy_fd_vs_autograd_within_tolerance"] for row in coordinate_rows)),
            "coordinate_count": len(coordinate_rows),
            "directional": directional,
            "tolerances": agreement_tolerances,
        },
        "interpretation_boundary": "This is an independent NumPy reconstruction of one synthetic engineering probe. Differences are retained as measurements; this is not a training, frontend, video, speech-quality, or efficacy result. The saved source waveform is read in memory for the exact CPU02 comparison and is not copied into the audit output.",
    }
    result_bytes = _canonical_json(result)
    if len(result_bytes) > int(config["resource_limits"]["max_json_bytes"]):
        raise MemoryError("independent result JSON exceeded configured byte bound")
    result_path = output_root / "independent_result.json"
    _write_new(result_path, result_bytes)
    result_sha, result_size = _sha_file(result_path)
    total_size = arrays_size + result_size
    if total_size > int(config["resource_limits"]["max_total_output_bytes"]):
        raise MemoryError("independent audit output exceeded configured total-byte bound")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": result["status"],
        "files": [
            {"path": "independent_zero_theta_arrays.npz", "bytes": arrays_size, "sha256": arrays_sha},
            {"path": "independent_result.json", "bytes": result_size, "sha256": result_sha},
        ],
        "total_bytes": total_size,
        "max_total_bytes": int(config["resource_limits"]["max_total_output_bytes"]),
        "configuration_sha256": _sha_bytes(config_bytes),
        "source_pins": source_pins,
        "input_manifest_sha256": {
            "cpu02": cpu_manifest_sha,
            "zero_supplement": zero_manifest_sha,
        },
    }
    manifest_bytes = _canonical_json(manifest)
    if len(manifest_bytes) > int(config["resource_limits"]["max_json_bytes"]):
        raise MemoryError("independent manifest exceeded configured JSON byte bound")
    _write_new(output_root / "result_manifest.json", manifest_bytes)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cpu02-output-dir", required=True)
    parser.add_argument("--zero-supplement-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return _run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
