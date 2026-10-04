#!/usr/bin/env python3
"""Source-only float32 cancellation diagnostic for the frozen V3 fit40 probe.

Consumes sealed magnitude/flux arrays only. It never reads WAV/PCM, imports the
head or Torch, runs a model, or decides whether a candidate is useful.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path, PurePosixPath
from typing import Any


def _sha_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return digest.hexdigest(), size


def _safe_file(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or not rel.parts or any(part in ("", ".", "..") for part in rel.parts):
        raise ValueError("unsafe relative input path")
    candidate = root.joinpath(*rel.parts)
    if candidate.is_symlink():
        raise ValueError("symlinked pinned input is not allowed")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink() or not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError("input is not a regular file beneath its pinned root")
    return resolved


def _verified_file(path: Path, expected: dict[str, Any], label: str) -> dict[str, Any]:
    sha, size = _sha_file(path)
    if size != int(expected["bytes"]) or sha != expected["sha256"]:
        raise ValueError(f"pinned {label} size/hash mismatch")
    return {"path": str(path), "bytes": size, "sha256": sha}


def _read_json(path: Path, maximum: int) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    if len(data) > maximum:
        raise ValueError("JSON input exceeds configured byte bound")
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value, data


def _inside(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts):
        raise ValueError("unsafe manifest member path")
    result = root.joinpath(*rel.parts)
    if result.is_symlink():
        raise ValueError("symlinked manifest member is not allowed")
    resolved = result.resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file() or not resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError("manifest member is not a regular file beneath its bundle")
    return resolved


def _verify_bundle(root: Path, pin: dict[str, Any], maximum_file: int) -> tuple[dict[str, Any], dict[str, Path]]:
    manifest_path = _inside(root, pin["manifest_name"])
    manifest_sha, manifest_size = _sha_file(manifest_path)
    if manifest_sha != pin["manifest_sha256"] or manifest_size != int(pin["manifest_bytes"]):
        raise ValueError(f"{pin['name']} manifest raw binding mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files: dict[str, Path] = {}
    declared: dict[str, dict[str, Any]] = {}
    for item in manifest["files"]:
        rel = item["path"]
        if rel in declared:
            raise ValueError(f"duplicate {pin['name']} manifest member")
        if int(item["bytes"]) > maximum_file:
            raise ValueError(f"{pin['name']} manifest member exceeds configured file bound")
        declared[rel] = item
    expected = pin["members"]
    if set(declared) != set(expected):
        raise ValueError(f"{pin['name']} manifest member roster mismatch")
    for rel, member_pin in expected.items():
        if declared[rel] != {"bytes": int(member_pin["bytes"]), "path": rel, "sha256": member_pin["sha256"]}:
            raise ValueError(f"{pin['name']} manifest's declared member pin mismatch")
        path = _inside(root, rel)
        _verified_file(path, member_pin, f"{pin['name']} member {rel}")
        files[rel] = path
    return manifest, files


def _write_json(path: Path, value: dict[str, Any]) -> tuple[str, int]:
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _u_bound(abs_exact: Any, u: float, eta_half: float) -> Any:
    """One correctly-rounded binary32 op bound, including gradual underflow."""
    import numpy as np
    return u * np.abs(np.asarray(abs_exact, dtype=np.float64)) + eta_half


def _sum_bound(term_abs_sum: Any, n: int, u: float, eta: float) -> Any:
    """Conservative n-term reduction bound for any addition tree with n-1 adds."""
    import numpy as np
    gamma = ((n - 1) * u) / (1.0 - (n - 1) * u)
    # n*eta dominates the propagated half-ulp absolute allowance under
    # gradual underflow; the gamma term is the standard relative bound.
    return gamma * np.asarray(term_abs_sum, dtype=np.float64) + n * eta


def _quantiles(values: Any, np: Any) -> dict[str, float | None]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {"count": 0, "min": None, "p05": None, "median": None, "p95": None, "max": None}
    return {
        "count": int(v.size),
        "min": float(np.min(v)),
        "p05": float(np.quantile(v, 0.05)),
        "median": float(np.quantile(v, 0.5)),
        "p95": float(np.quantile(v, 0.95)),
        "max": float(np.max(v)),
    }


def _run(args: argparse.Namespace) -> int:
    import numpy as np

    repo = Path(args.repo_root).resolve(strict=True)
    cfg_path = Path(args.config).resolve(strict=True)
    cfg_sha, cfg_size = _sha_file(cfg_path)
    if cfg_sha != args.expected_config_sha256 or cfg_size != int(args.expected_config_bytes):
        raise ValueError("diagnostic config raw pin mismatch")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    runtime = cfg["runtime"]
    if list(sys.version_info[:3]) != runtime["python_version"] or platform.python_implementation() != runtime["implementation"]:
        raise RuntimeError("Python runtime differs from the pinned contract")
    if np.__version__ != runtime["numpy_version"]:
        raise RuntimeError("NumPy runtime differs from the pinned contract")
    if "torch" in sys.modules:
        raise RuntimeError("Torch was imported into the source-only NumPy process")
    if os.environ.get("PYTHONNOUSERSITE") not in ("1", "true", "True"):
        raise RuntimeError("PYTHONNOUSERSITE must be enabled")
    for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(env_name) != "1":
            raise RuntimeError(f"{env_name} must equal 1")

    for name, pin in cfg["source_pins"].items():
        source = _safe_file(repo, pin["path"])
        _verified_file(source, pin, name)

    cpu_root = Path(args.cpu_output_dir).resolve(strict=True)
    audit_root = Path(args.audit_output_dir).resolve(strict=True)
    expected_cpu_root = (repo / cfg["bundles"]["cpu02"]["root_relative_path"]).resolve(strict=True)
    expected_audit_root = (repo / cfg["bundles"]["numpy03"]["root_relative_path"]).resolve(strict=True)
    if cpu_root != expected_cpu_root or audit_root != expected_audit_root:
        raise ValueError("input bundle directories differ from the config's exact readback paths")
    cpu_manifest, cpu_files = _verify_bundle(cpu_root, cfg["bundles"]["cpu02"], int(cfg["limits"]["max_input_member_bytes"]))
    audit_manifest, audit_files = _verify_bundle(audit_root, cfg["bundles"]["numpy03"], int(cfg["limits"]["max_input_member_bytes"]))
    if cpu_manifest.get("status") != "COMPLETE" or int(cpu_manifest.get("completed_fit_rows", -1)) != int(cfg["expected_rows"]):
        raise ValueError("CPU02 artifact is not the declared complete fit40 engineering output")
    if audit_manifest.get("status") != "COMPLETE_WITH_NUMERIC_MISMATCHES" or int(audit_manifest.get("rows", -1)) != int(cfg["expected_rows"]):
        raise ValueError("NumPy03 artifact does not match the preserved numeric-mismatch input")
    if cpu_manifest.get("configuration_sha256") != cfg["source_pins"]["cpu02_config"]["sha256"]:
        raise ValueError("CPU02 output does not bind its pinned config")
    if cpu_manifest.get("source_manifest_sha256") != cfg["source_pins"]["fit40_source_manifest"]["sha256"]:
        raise ValueError("CPU02 output does not bind the pinned fit40 source manifest")
    if audit_manifest.get("consumer_config_sha256") != cfg["source_pins"]["numpy03_config"]["sha256"]:
        raise ValueError("NumPy03 output does not bind its pinned config")
    if audit_manifest.get("qualification_config_sha256") != cfg["source_pins"]["cpu02_config"]["sha256"]:
        raise ValueError("NumPy03 output does not bind the pinned CPU02 config")
    if audit_manifest.get("qualification_result_manifest_sha256") != cfg["bundles"]["cpu02"]["manifest_sha256"]:
        raise ValueError("NumPy03 result does not bind the exact CPU02 source manifest")
    if audit_manifest.get("fit40_manifest_sha256") != cfg["source_pins"]["fit40_source_manifest"]["sha256"]:
        raise ValueError("NumPy03 fit40 lineage differs from the pinned transfer manifest")
    total_input_bytes = sum(int(pin["bytes"]) for pin in cfg["source_pins"].values())
    for bundle in cfg["bundles"].values():
        total_input_bytes += int(bundle["manifest_bytes"])
        total_input_bytes += sum(int(member["bytes"]) for member in bundle["members"].values())
    if total_input_bytes > int(cfg["limits"]["max_total_input_bytes"]):
        raise ValueError("pinned input bundle exceeds configured aggregate byte bound")

    with np.load(cpu_files["qualification_matrices.npz"], allow_pickle=False) as z:
        cpu = {key: z[key] for key in z.files}
    with np.load(audit_files["independent_numpy_matrices.npz"], allow_pickle=False) as z:
        independent = {key: z[key] for key in z.files}
    cpu_records = [json.loads(line) for line in cpu_files["results.jsonl"].read_text(encoding="utf-8").splitlines() if line.strip()]
    audit_records = [json.loads(line) for line in audit_files["audit_results.jsonl"].read_text(encoding="utf-8").splitlines() if line.strip()]
    required_cpu = {
        "fit_row_ids_utf8", "fit_sample_ids_utf8", "fit_speaker_ids_utf8", "fit_canonical_j_i64",
        "fit_stft_frame_offsets_i64", "fit_positive_relative_flux_f32", "fit_negative_relative_flux_f32",
        "fit_active_frame_mask_u8", "fit_valid_transition_mask_u8", "fit_frame_magnitude_sum_f32",
        "fit_previous_frame_magnitude_sum_f32", "fit_flux_denominator_f32", "fit_stft_frame_centers_source_samples_i64",
    }
    required_numpy = {
        "fit_row_ids_utf8", "fit_sample_ids_utf8", "fit_speaker_ids_utf8", "fit_canonical_j_i64",
        "fit_stft_frame_offsets_i64", "fit_source_magnitude_f32", "fit_recomputed_positive_relative_flux_f32",
        "fit_recomputed_negative_relative_flux_f32", "fit_recomputed_valid_transition_mask_u8",
        "fit_recomputed_frame_magnitude_sum_f32", "fit_recomputed_previous_frame_magnitude_sum_f32",
    }
    if not required_cpu.issubset(cpu) or not required_numpy.issubset(independent):
        raise ValueError("a required frozen matrix field is absent")
    for key in ("fit_row_ids_utf8", "fit_sample_ids_utf8", "fit_speaker_ids_utf8", "fit_canonical_j_i64", "fit_stft_frame_offsets_i64"):
        if not np.array_equal(cpu[key], independent[key]):
            raise ValueError(f"CPU02/NumPy03 row lineage mismatch at {key}")
    ids = [str(item) for item in independent["fit_row_ids_utf8"]]
    if len(ids) != int(cfg["expected_rows"]) or len(set(ids)) != len(ids):
        raise ValueError("fit row IDs are not unique or do not match the frozen row count")
    if len(cpu_records) != len(ids) or len(audit_records) != len(ids):
        raise ValueError("row-level CPU/NumPy JSONL rosters do not match the sealed matrices")
    if [str(row.get("row_id")) for row in cpu_records] != ids or [str(row.get("row_id")) for row in audit_records] != ids:
        raise ValueError("row-level CPU/NumPy JSONL ordering differs from the sealed matrices")
    prior_audit_boolean_failures: dict[str, int] = {}
    for record in audit_records:
        checks = record.get("measured_numeric_checks")
        if not isinstance(checks, dict):
            raise ValueError("preserved NumPy03 row lacks its measured numeric check map")
        for name, value in checks.items():
            if isinstance(value, bool) and not value:
                prior_audit_boolean_failures[name] = prior_audit_boolean_failures.get(name, 0) + 1
    offsets = np.asarray(independent["fit_stft_frame_offsets_i64"], dtype=np.int64)
    magnitude = np.asarray(independent["fit_source_magnitude_f32"], dtype=np.float32)
    if offsets.shape != (len(ids) + 1,) or offsets[0] != 0 or offsets[-1] != magnitude.shape[1] or np.any(np.diff(offsets) < 2):
        raise ValueError("frozen STFT offsets do not describe one supported transition per row")
    if magnitude.shape[0] != int(cfg["spectrum_bins"]) or not np.all(np.isfinite(magnitude)) or np.any(magnitude < 0):
        raise ValueError("frozen magnitude matrix shape or finite/nonnegative contract mismatch")

    qcfg_path = _safe_file(repo, cfg["source_pins"]["cpu02_config"]["path"])
    qcfg = json.loads(qcfg_path.read_text(encoding="utf-8"))
    options = qcfg["head_constructor"]
    scale = np.float32(options["flux_normalization_scale"])
    floor = np.float32(options["flux_denominator_floor"])
    activity_floor = np.float32(options["activity_floor"])
    if (not np.isfinite(scale) or scale <= 0 or not np.isfinite(floor) or floor <= 0
            or not np.isfinite(activity_floor) or activity_floor < 0):
        raise ValueError("pinned flux scale/floor/activity floor is outside its finite contract")

    u = 2.0 ** -24
    eta = 2.0 ** -149
    eta_half = 2.0 ** -150
    bins = magnitude.shape[0]
    sum_gamma = ((bins - 1) * u) / (1.0 - (bins - 1) * u)

    derived: dict[str, list[Any]] = {key: [] for key in (
        "valid_transition_mask_u8", "cpu_valid_transition_mask_u8", "numpy_valid_transition_mask_u8",
        "frame_index_current_i64", "frame_center_current_source_samples_i64",
        "total_current_f32", "total_previous_f32", "frame_total_ratio_f32",
        "positive_numerator_f32", "negative_numerator_f32", "direct_difference_sum_f32",
        "rounded_totals_difference_f32", "positive_relative_flux_f32", "negative_relative_flux_f32",
        "lhs_positive_minus_negative_f32", "rhs_rounded_total_delta_f32", "identity_residual_f32",
        "final_residual_subtraction_rounding_bound_f64", "direct_difference_sum_rounding_bound_f64",
        "separate_totals_delta_rounding_bound_f64", "residual_absolute_rounding_bound_f64",
        "ratio_abs_total_delta_over_pos_neg_f32",
        "numpy_saved_positive_flux_f32", "numpy_saved_negative_flux_f32",
        "cpu_saved_positive_flux_f32", "cpu_saved_negative_flux_f32",
        "pos_minus_neg_sum64", "direct_diff_sum64", "original_magnitude_delta64",
        "total_current64", "total_previous64", "exact_total_delta64",
        "float64_identity_residual64", "float64_diff_vs_exact_total_delta64",
        "numpy_saved_positive_flux_abs_error_f32", "numpy_saved_negative_flux_abs_error_f32",
        "cpu_saved_positive_flux_abs_error_f32", "cpu_saved_negative_flux_abs_error_f32",
    )}
    row_summaries: list[dict[str, Any]] = []
    transition_offsets = [0]
    validity_mismatch_count = 0
    total_sum_max_abs_error_numpy = 0.0
    total_sum_max_abs_error_cpu = 0.0
    for row_index, row_id in enumerate(ids):
        start, stop = int(offsets[row_index]), int(offsets[row_index + 1])
        row_mag = magnitude[:, start:stop]
        row_frames = row_mag.shape[1]
        total = np.sum(row_mag, axis=0, dtype=np.float32)
        active = total > activity_floor
        valid_full = np.zeros(row_frames, dtype=np.uint8)
        valid_full[1:] = (active[:-1] & active[1:] & (total[:-1] > floor)).astype(np.uint8)
        valid = valid_full[1:]
        cpu_valid = np.asarray(cpu["fit_valid_transition_mask_u8"][start + 1:stop], dtype=np.uint8)
        numpy_valid = np.asarray(independent["fit_recomputed_valid_transition_mask_u8"][start + 1:stop], dtype=np.uint8)
        validity_mismatch_count += int(np.count_nonzero(valid != cpu_valid) + np.count_nonzero(valid != numpy_valid))

        diff = np.subtract(row_mag[:, 1:], row_mag[:, :-1], dtype=np.float32)
        positive_terms = np.maximum(diff, np.float32(0.0))
        negative_terms = np.maximum(-diff, np.float32(0.0))
        pos_num = np.sum(positive_terms, axis=0, dtype=np.float32)
        neg_num = np.sum(negative_terms, axis=0, dtype=np.float32)
        direct_num = np.sum(diff, axis=0, dtype=np.float32)
        total_delta = np.subtract(total[1:], total[:-1], dtype=np.float32)
        denominator = np.maximum(total[:-1], floor)
        pos_div1 = np.divide(pos_num, denominator, dtype=np.float32)
        neg_div1 = np.divide(neg_num, denominator, dtype=np.float32)
        pos_rel = np.divide(pos_div1, scale, dtype=np.float32)
        neg_rel = np.divide(neg_div1, scale, dtype=np.float32)
        lhs = np.subtract(pos_rel, neg_rel, dtype=np.float32)
        rhs_div1 = np.divide(total_delta, denominator, dtype=np.float32)
        rhs = np.divide(rhs_div1, scale, dtype=np.float32)
        residual = np.subtract(lhs, rhs, dtype=np.float32)
        ratio = np.divide(total[1:], np.maximum(total[:-1], np.finfo(np.float32).tiny), dtype=np.float32)
        positive_negative_sum = np.add(pos_num, neg_num, dtype=np.float32)
        cancellation_ratio = np.divide(np.abs(direct_num), positive_negative_sum, out=np.zeros_like(pos_num), where=positive_negative_sum > 0, dtype=np.float32)

        f32_stages = {
            "input_magnitude": row_mag,
            "frame_magnitude_sum": total,
            "bin_difference": diff,
            "positive_terms": positive_terms,
            "negative_terms": negative_terms,
            "positive_reduction": pos_num,
            "negative_reduction": neg_num,
            "direct_difference_reduction": direct_num,
            "separate_total_subtraction": total_delta,
            "denominator": denominator,
            "positive_divide_by_denominator": pos_div1,
            "negative_divide_by_denominator": neg_div1,
            "positive_relative_flux": pos_rel,
            "negative_relative_flux": neg_rel,
            "relative_flux_subtraction": lhs,
            "total_delta_divide_by_denominator": rhs_div1,
            "right_relative_flux": rhs,
            "final_identity_subtraction": residual,
            "frame_total_ratio": ratio,
            "positive_plus_negative_reduction": positive_negative_sum,
            "cancellation_ratio": cancellation_ratio,
        }
        nonfinite_by_stage = {
            name: int(np.size(value) - np.count_nonzero(np.isfinite(value)))
            for name, value in f32_stages.items()
        }
        actual_f32_finite = all(count == 0 for count in nonfinite_by_stage.values())
        if not actual_f32_finite:
            raise FloatingPointError("actual binary32 intermediate is nonfinite; overflow/invalid arithmetic violates bound assumptions")

        # Recompute from the original saved float32 magnitudes after promoting
        # each operand, rather than promoting the already-rounded float32 diff.
        # These are float64 reductions, not a reanalysis of source PCM.
        magnitude64 = row_mag.astype(np.float64)
        delta64 = magnitude64[:, 1:] - magnitude64[:, :-1]
        pos64 = np.sum(np.maximum(delta64, 0.0), axis=0, dtype=np.float64)
        neg64 = np.sum(np.maximum(-delta64, 0.0), axis=0, dtype=np.float64)
        direct64 = np.sum(delta64, axis=0, dtype=np.float64)
        total_current64 = np.sum(magnitude64[:, 1:], axis=0, dtype=np.float64)
        total_previous64 = np.sum(magnitude64[:, :-1], axis=0, dtype=np.float64)
        exact_total_delta64 = total_current64 - total_previous64
        identity64 = (pos64 - neg64) - direct64
        diff_vs_total64 = direct64 - exact_total_delta64

        diff_error_bound = np.sum(u * np.abs(row_mag[:, 1:].astype(np.float64) - row_mag[:, :-1].astype(np.float64)) + eta_half, axis=0, dtype=np.float64)
        pos_sum_bound = sum_gamma * np.sum(np.abs(positive_terms.astype(np.float64)), axis=0, dtype=np.float64) + bins * eta
        neg_sum_bound = sum_gamma * np.sum(np.abs(negative_terms.astype(np.float64)), axis=0, dtype=np.float64) + bins * eta
        direct_sum_bound = sum_gamma * np.sum(np.abs(diff.astype(np.float64)), axis=0, dtype=np.float64) + bins * eta
        direct_difference_bound = diff_error_bound + direct_sum_bound
        curr_sum_bound = sum_gamma * np.sum(np.abs(row_mag[:, 1:].astype(np.float64)), axis=0, dtype=np.float64) + bins * eta
        prev_sum_bound = sum_gamma * np.sum(np.abs(row_mag[:, :-1].astype(np.float64)), axis=0, dtype=np.float64) + bins * eta
        rounded_total_delta_exact = total[1:].astype(np.float64) - total[:-1].astype(np.float64)
        right_num_bound = curr_sum_bound + prev_sum_bound + u * np.abs(rounded_total_delta_exact) + eta_half

        denom64 = denominator.astype(np.float64)
        scale64 = float(scale)
        pos_div1_exact = pos_num.astype(np.float64) / denom64
        neg_div1_exact = neg_num.astype(np.float64) / denom64
        rhs_div1_exact = total_delta.astype(np.float64) / denom64
        pos_path_bound = (pos_sum_bound + diff_error_bound) / (denom64 * scale64)
        pos_path_bound += (u * np.abs(pos_div1_exact) + eta_half) / scale64
        pos_div2_exact = pos_div1.astype(np.float64) / scale64
        pos_path_bound += u * np.abs(pos_div2_exact) + eta_half
        neg_path_bound = (neg_sum_bound + diff_error_bound) / (denom64 * scale64)
        neg_path_bound += (u * np.abs(neg_div1_exact) + eta_half) / scale64
        neg_div2_exact = neg_div1.astype(np.float64) / scale64
        neg_path_bound += u * np.abs(neg_div2_exact) + eta_half
        left_sub_exact = pos_rel.astype(np.float64) - neg_rel.astype(np.float64)
        left_rel_bound = pos_path_bound + neg_path_bound + u * np.abs(left_sub_exact) + eta_half
        right_rel_bound = right_num_bound / (denom64 * scale64)
        right_rel_bound += (u * np.abs(rhs_div1_exact) + eta_half) / scale64
        rhs_div2_exact = rhs_div1.astype(np.float64) / scale64
        right_rel_bound += u * np.abs(rhs_div2_exact) + eta_half
        # residual itself is another rounded binary32 subtraction. Include
        # its rounding relative to the exact difference of the stored f32 L/R.
        final_residual_sub_exact = lhs.astype(np.float64) - rhs.astype(np.float64)
        final_residual_sub_bound = u * np.abs(final_residual_sub_exact) + eta_half
        separate_totals_delta_bound = right_num_bound
        bound = left_rel_bound + right_rel_bound + final_residual_sub_bound
        f64_bound_stages = {
            "float64_reference_identity": identity64,
            "float64_reference_delta_difference": diff_vs_total64,
            "direct_difference_sum_bound": direct_difference_bound,
            "separate_totals_delta_bound": separate_totals_delta_bound,
            "positive_relative_path_bound": pos_path_bound,
            "negative_relative_path_bound": neg_path_bound,
            "left_relative_path_bound": left_rel_bound,
            "right_relative_path_bound": right_rel_bound,
            "final_residual_subtraction_bound": final_residual_sub_bound,
            "full_residual_bound": bound,
        }
        nonfinite_f64_by_stage = {
            name: int(np.size(value) - np.count_nonzero(np.isfinite(value)))
            for name, value in f64_bound_stages.items()
        }
        actual_f64_finite = all(count == 0 for count in nonfinite_f64_by_stage.values())
        if not actual_f64_finite:
            raise FloatingPointError("nonfinite float64 identity or propagated arithmetic bound")

        numpy_pos = np.asarray(independent["fit_recomputed_positive_relative_flux_f32"][start + 1:stop], dtype=np.float32)
        numpy_neg = np.asarray(independent["fit_recomputed_negative_relative_flux_f32"][start + 1:stop], dtype=np.float32)
        cpu_pos = np.asarray(cpu["fit_positive_relative_flux_f32"][start + 1:stop], dtype=np.float32)
        cpu_neg = np.asarray(cpu["fit_negative_relative_flux_f32"][start + 1:stop], dtype=np.float32)
        numpy_total = np.asarray(independent["fit_recomputed_frame_magnitude_sum_f32"][start:stop], dtype=np.float32)
        cpu_total = np.asarray(cpu["fit_frame_magnitude_sum_f32"][start:stop], dtype=np.float32)
        total_sum_max_abs_error_numpy = max(total_sum_max_abs_error_numpy, float(np.max(np.abs(total - numpy_total))))
        total_sum_max_abs_error_cpu = max(total_sum_max_abs_error_cpu, float(np.max(np.abs(total - cpu_total))))

        arrays = {
            "valid_transition_mask_u8": valid,
            "cpu_valid_transition_mask_u8": cpu_valid,
            "numpy_valid_transition_mask_u8": numpy_valid,
            "frame_index_current_i64": np.arange(1, row_frames, dtype=np.int64),
            "frame_center_current_source_samples_i64": np.asarray(cpu["fit_stft_frame_centers_source_samples_i64"][start + 1:stop], dtype=np.int64),
            "total_current_f32": total[1:], "total_previous_f32": total[:-1], "frame_total_ratio_f32": ratio,
            "positive_numerator_f32": pos_num, "negative_numerator_f32": neg_num,
            "direct_difference_sum_f32": direct_num, "rounded_totals_difference_f32": total_delta,
            "positive_relative_flux_f32": pos_rel, "negative_relative_flux_f32": neg_rel,
            "lhs_positive_minus_negative_f32": lhs, "rhs_rounded_total_delta_f32": rhs,
            "identity_residual_f32": residual,
            "final_residual_subtraction_rounding_bound_f64": final_residual_sub_bound,
            "direct_difference_sum_rounding_bound_f64": direct_difference_bound,
            "separate_totals_delta_rounding_bound_f64": separate_totals_delta_bound,
            "residual_absolute_rounding_bound_f64": bound,
            "ratio_abs_total_delta_over_pos_neg_f32": cancellation_ratio,
            "numpy_saved_positive_flux_f32": numpy_pos, "numpy_saved_negative_flux_f32": numpy_neg,
            "cpu_saved_positive_flux_f32": cpu_pos, "cpu_saved_negative_flux_f32": cpu_neg,
            "pos_minus_neg_sum64": pos64 - neg64, "direct_diff_sum64": direct64,
            "original_magnitude_delta64": direct64,
            "total_current64": total_current64, "total_previous64": total_previous64,
            "exact_total_delta64": exact_total_delta64, "float64_identity_residual64": identity64,
            "float64_diff_vs_exact_total_delta64": diff_vs_total64,
            "numpy_saved_positive_flux_abs_error_f32": np.abs(numpy_pos - pos_rel),
            "numpy_saved_negative_flux_abs_error_f32": np.abs(numpy_neg - neg_rel),
            "cpu_saved_positive_flux_abs_error_f32": np.abs(cpu_pos - pos_rel),
            "cpu_saved_negative_flux_abs_error_f32": np.abs(cpu_neg - neg_rel),
        }
        for key, value in arrays.items():
            derived[key].append(np.asarray(value))
        transition_offsets.append(transition_offsets[-1] + row_frames - 1)

        valid_index = np.flatnonzero(valid.astype(bool))
        if valid_index.size:
            resid_valid = np.abs(residual[valid_index].astype(np.float64))
            local_arg = int(valid_index[int(np.argmax(resid_valid))])
            valid_bound = bound[valid_index]
            row_summary = {
                "row_id": row_id,
                "sample_id": str(independent["fit_sample_ids_utf8"][row_index]),
                "speaker_id": str(independent["fit_speaker_ids_utf8"][row_index]),
                "canonical_j": int(independent["fit_canonical_j_i64"][row_index]),
                "transition_count": int(row_frames - 1),
                "valid_transition_count": int(valid_index.size),
                "all_binary32_intermediates_finite_no_overflow": actual_f32_finite,
                "nonfinite_binary32_counts_by_stage": nonfinite_by_stage,
                "all_float64_references_and_bounds_finite": actual_f64_finite,
                "nonfinite_float64_counts_by_stage": nonfinite_f64_by_stage,
                "valid_mask_exact_match_cpu02": bool(np.array_equal(valid, cpu_valid)),
                "valid_mask_exact_match_numpy03": bool(np.array_equal(valid, numpy_valid)),
                "max_abs_identity_residual_f32": float(resid_valid.max()),
                "max_residual_frame_index_current": local_arg + 1,
                "max_residual_frame_center_source_sample": int(arrays["frame_center_current_source_samples_i64"][local_arg]),
                "signed_residual_at_max": float(residual[local_arg]),
                "rounding_bound_at_max": float(bound[local_arg]),
                "valid_residual_bound_exceedance_count": int(np.count_nonzero(resid_valid > valid_bound)),
                "max_abs_numerator_identity_residual_f32": float(np.max(np.abs((pos_num[valid_index] - neg_num[valid_index]) - direct_num[valid_index]))),
                "max_abs_rounded_total_delta_vs_direct_sum_f32": float(np.max(np.abs(total_delta[valid_index] - direct_num[valid_index]))),
                "max_direct_sum_rounding_bound_f64": float(np.max(direct_difference_bound[valid_index])),
                "max_separate_totals_delta_rounding_bound_f64": float(np.max(separate_totals_delta_bound[valid_index])),
                "max_final_residual_subtraction_bound_f64": float(np.max(final_residual_sub_bound[valid_index])),
                "float64_identity_residual_max": float(np.max(np.abs(identity64[valid_index]))),
                "float64_direct_diff_vs_sum_difference_max": float(np.max(np.abs(diff_vs_total64[valid_index]))),
                "valid_frame_total_current_over_previous": _quantiles(ratio[valid_index], np),
                "abs_direct_change_over_positive_plus_negative": _quantiles(cancellation_ratio[valid_index], np),
                "saved_numpy_positive_flux_abs_error_max": float(np.max(arrays["numpy_saved_positive_flux_abs_error_f32"][valid_index])),
                "saved_numpy_negative_flux_abs_error_max": float(np.max(arrays["numpy_saved_negative_flux_abs_error_f32"][valid_index])),
                "saved_cpu_positive_flux_abs_error_max": float(np.max(arrays["cpu_saved_positive_flux_abs_error_f32"][valid_index])),
                "saved_cpu_negative_flux_abs_error_max": float(np.max(arrays["cpu_saved_negative_flux_abs_error_f32"][valid_index])),
            }
        else:
            row_summary = {
                "row_id": row_id,
                "sample_id": str(independent["fit_sample_ids_utf8"][row_index]),
                "speaker_id": str(independent["fit_speaker_ids_utf8"][row_index]),
                "canonical_j": int(independent["fit_canonical_j_i64"][row_index]),
                "transition_count": int(row_frames - 1),
                "valid_transition_count": 0,
                "all_binary32_intermediates_finite_no_overflow": actual_f32_finite,
                "nonfinite_binary32_counts_by_stage": nonfinite_by_stage,
                "all_float64_references_and_bounds_finite": actual_f64_finite,
                "nonfinite_float64_counts_by_stage": nonfinite_f64_by_stage,
                "valid_mask_exact_match_cpu02": bool(np.array_equal(valid, cpu_valid)),
                "valid_mask_exact_match_numpy03": bool(np.array_equal(valid, numpy_valid)),
                "max_abs_identity_residual_f32": None,
                "max_residual_frame_index_current": None,
                "max_residual_frame_center_source_sample": None,
                "signed_residual_at_max": None,
                "rounding_bound_at_max": None,
                "valid_residual_bound_exceedance_count": 0,
                "max_abs_numerator_identity_residual_f32": None,
                "max_abs_rounded_total_delta_vs_direct_sum_f32": None,
                "max_direct_sum_rounding_bound_f64": None,
                "max_separate_totals_delta_rounding_bound_f64": None,
                "max_final_residual_subtraction_bound_f64": None,
                "float64_identity_residual_max": None,
                "float64_direct_diff_vs_sum_difference_max": None,
                "valid_frame_total_current_over_previous": _quantiles([], np),
                "abs_direct_change_over_positive_plus_negative": _quantiles([], np),
                "saved_numpy_positive_flux_abs_error_max": None,
                "saved_numpy_negative_flux_abs_error_max": None,
                "saved_cpu_positive_flux_abs_error_max": None,
                "saved_cpu_negative_flux_abs_error_max": None,
            }
        row_summaries.append(row_summary)

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError("diagnostic output directory already exists")
    output.mkdir(parents=True)
    packed = {key: np.concatenate(values) for key, values in derived.items()}
    packed["row_transition_offsets_i64"] = np.asarray(transition_offsets, dtype=np.int64)
    npz_path = output / "flux_rounding_arrays.npz"
    np.savez_compressed(npz_path, **packed)
    if npz_path.stat().st_size > int(cfg["limits"]["max_npz_bytes"]):
        raise ValueError("derived array archive exceeds configured byte bound")
    jsonl_path = output / "row_diagnostics.jsonl"
    with jsonl_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in row_summaries:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "schema": cfg["output_schema"],
        "status": "COMPLETE_DIAGNOSTIC",
        "interpretation": "The residuals diagnose float32 operation-order effects in the positive-minus-negative relative-flux identity. The bound includes the final rounded residual subtraction; separate numerator-path bounds are retained. Float64 identities promote the original saved float32 magnitude operands before subtraction. This is not a head-quality or transfer result.",
        "scope": cfg["scope"],
        "configuration_sha256": cfg_sha,
        "configuration_bytes": cfg_size,
        "source_pins": cfg["source_pins"],
        "input_manifest_pins": {
            "cpu02": {"sha256": cfg["bundles"]["cpu02"]["manifest_sha256"], "status": cpu_manifest["status"]},
            "numpy03": {"sha256": cfg["bundles"]["numpy03"]["manifest_sha256"], "status": audit_manifest["status"]},
        },
        "row_count": len(row_summaries),
        "transition_count": int(packed["valid_transition_mask_u8"].size),
        "valid_transition_count": int(np.sum(packed["valid_transition_mask_u8"], dtype=np.int64)),
        "valid_mask_mismatch_count_vs_cpu_and_numpy": validity_mismatch_count,
        "preserved_numpy03_boolean_failures_by_check": prior_audit_boolean_failures,
        "verified_input_bytes_including_pinned_sources_and_bundle_members": total_input_bytes,
        "float32_bound_model": {
            "unit_roundoff": u,
            "smallest_subnormal": eta,
            "sum_terms": bins,
            "sum_gamma_n_minus_1": sum_gamma,
            "sum_bound": "gamma_(n-1) * sum(abs(terms)) + n * eta, conservative for any n-term addition tree with at most n-1 rounded additions under binary32 round-to-nearest and gradual underflow",
            "operation_bound": "u * abs(exact_operation_result) + eta/2 for each rounded binary32 subtraction/division; propagated through input-bin subtraction, the positive/negative or separate-total reductions, numerator subtraction, denominator division, scale division, and the final rounded L-minus-R subtraction",
            "assumptions": ["finite inputs", "runtime-checked finite binary32 intermediates (therefore no observed overflow/invalid operation)", "IEEE-754 binary32 round-to-nearest", "gradual underflow", "each 257-term reduction uses no more than 256 additions"],
        },
        "cross_runtime_frame_magnitude_sum_max_abs_error": {
            "numpy03": total_sum_max_abs_error_numpy,
            "cpu02": total_sum_max_abs_error_cpu,
        },
        "all_valid_residuals_within_derived_bound": bool(np.all(np.abs(packed["identity_residual_f32"][packed["valid_transition_mask_u8"].astype(bool)].astype(np.float64)) <= packed["residual_absolute_rounding_bound_f64"][packed["valid_transition_mask_u8"].astype(bool)])),
        "all_binary32_intermediates_finite_no_overflow": all(row["all_binary32_intermediates_finite_no_overflow"] for row in row_summaries),
        "all_float64_references_and_bounds_finite": all(row["all_float64_references_and_bounds_finite"] for row in row_summaries),
        "max_valid_abs_identity_residual_f32": float(np.max(np.abs(packed["identity_residual_f32"][packed["valid_transition_mask_u8"].astype(bool)]))),
        "max_valid_derived_rounding_bound_f64": float(np.max(packed["residual_absolute_rounding_bound_f64"][packed["valid_transition_mask_u8"].astype(bool)])),
        "input_files": {
            "cpu02_manifest": cfg["bundles"]["cpu02"]["manifest_sha256"],
            "numpy03_manifest": cfg["bundles"]["numpy03"]["manifest_sha256"],
        },
        "outputs": {
            "flux_rounding_arrays.npz": {"bytes": npz_path.stat().st_size, "sha256": _sha_file(npz_path)[0]},
            "row_diagnostics.jsonl": {"bytes": jsonl_path.stat().st_size, "sha256": _sha_file(jsonl_path)[0]},
        },
        "rows": row_summaries,
    }
    summary_sha, summary_size = _write_json(output / "summary.json", summary)
    files = []
    for name in ("flux_rounding_arrays.npz", "row_diagnostics.jsonl", "summary.json"):
        digest, size = _sha_file(output / name)
        files.append({"path": name, "bytes": size, "sha256": digest})
    manifest = {
        "schema": cfg["manifest_schema"],
        "status": "COMPLETE_DIAGNOSTIC",
        "configuration_sha256": cfg_sha,
        "cpu02_manifest_sha256": cfg["bundles"]["cpu02"]["manifest_sha256"],
        "numpy03_manifest_sha256": cfg["bundles"]["numpy03"]["manifest_sha256"],
        "row_count": len(row_summaries),
        "files": files,
    }
    manifest_sha, manifest_size = _write_json(output / "result_manifest.json", manifest)
    if sum(item["bytes"] for item in files) + manifest_size > int(cfg["limits"]["max_total_output_bytes"]):
        raise ValueError("diagnostic output exceeds configured total-byte bound")
    print(json.dumps({"status": manifest["status"], "rows": len(row_summaries), "manifest_sha256": manifest_sha, "manifest_bytes": manifest_size, "summary_sha256": summary_sha, "summary_bytes": summary_size}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-config-bytes", required=True, type=int)
    parser.add_argument("--cpu-output-dir", required=True)
    parser.add_argument("--audit-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return _run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
