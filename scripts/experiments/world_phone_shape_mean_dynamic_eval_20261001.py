#!/usr/bin/env python3
"""Create the frozen five-cell WORLD spectral-shape EVAL waveforms.

This producer reads only sealed physical EVAL inputs and a completed, independently
qualified CAL factorial. It writes WORLD parameters and waveforms; it never loads
video, SyncNet, effect summaries, or model code.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FIXED_RMS = 0.0376838172675746
SAMPLE_RATE = 16_000
FRAME_PERIOD_MS = 5.0
SHAPE_EPSILON = 1e-12
PARAMETER_ABS_TOL = 1e-12
SOFTMAX_ABS_TOL = 1e-12
SOFTMAX_REL_TOL = 1e-10
SOFTMAX_RELATIVE_DENOMINATOR_FLOOR = 1e-12
SP_POWER_REL_TOL = 1e-10
WAVE_BRIDGE_MAX_ABS_TOL = 1e-6
FIXED_RMS_ABS_TOL = 1e-9

PHYSICAL_MANIFEST_RAW_SHA256 = "27cb2cdf2246920c0667dd5bcc5d744253b3c6b7d407270c1254ffaf986aa3b3"
PHYSICAL_MANIFEST_PAYLOAD_SHA256 = "eb77432936c3454a241eb2505a966ffa0d05aa5be8514451cc2da66aa7ed9de8"
PHYSICAL_PLAN_RAW_SHA256 = "f92dbcfc71ecb775d98c3cd33ec19b4dd99d206c27d155eba8ae9bc51204861e"
PHYSICAL_PLAN_PAYLOAD_SHA256 = "bfae127eed35c459d1f6a2fff8feebac4f4b5d855b1f26611141138410f22b24"
PHYSICAL_QA_RAW_SHA256 = "e78e11587ace920003163d04010670aa4d273f80f5b7df657d0b44eb143fd39d"
SOURCE_INPUT_MANIFEST_RAW_SHA256 = "6b0ce3252cf776a86cf84b19ab53afd936526216e4ddb30dc587a72a66409b4b"
SOURCE_INPUT_MANIFEST_PAYLOAD_SHA256 = "b18f83ad07a660e6c0bdef3b8780c59b7b12497370a79916eb95c7dfe07303bf"
TTS_PAIR_MANIFEST_RAW_SHA256 = "2aea4750a02125629d82f85b4842932465a202dd03eec96864d096d8856070d2"
TTS_PAIR_MANIFEST_PAYLOAD_SHA256 = "e0b6b2c69db11703cc505a8d4627287d307825609ecfb65383807587ed0ac70f"
WORLD_SOURCE_SHA256 = "7a6661afced892037d5f32fa3760e8d0906b77f40bb06f5f3ce53a66a04729b2"
EXECUTION_BINDINGS_SHA256 = "cde66fa11f140866fc87f07b7a39ca68ddefc403ff7839b239987549e9f7e47d"
PHONE_TABLE_RAW_SHA256 = "1673092554bfc03dbc385136c3a66c08821499c0ec6c25602d385d1d0d7cf16f"
PHONE_TABLE_PAYLOAD_SHA256 = "586d86fedd159ba1c654586bf6892d87b36b1ed8d88c666217d648d230fb2259"
PHONE_INVENTORY_RAW_SHA256 = "62eeb63462c6e2d378482d1cdc7876d2e05666bac0f54c32d2a08f5e114db2c2"
PHONE_INVENTORY_PAYLOAD_SHA256 = "54a51ff1a9af891271b0eaed797dc0bb7508a338ef7e38f922b8f9292fa40f5d"
CAL_WORLD_MANIFEST_RAW_SHA256 = "dc23a765794b0dd6e735be26955dbcb39e4ce06dff4fbe1be9f1c6b926ab7efa"
CAL_WORLD_MANIFEST_PAYLOAD_SHA256 = "f3b8045245edcf21eb30c944f61318f8ac3538669152d64f68c882543d75a1f"
CAL_WORLD_PLAN_RAW_SHA256 = "594592fe09a763634b3f757220096b0f08c038923aec418816442fc99b0879e7"
CAL_WORLD_PLAN_PAYLOAD_SHA256 = "85b94f1f7ccdb2a7c7dbceadf72d4e0f298019c64122514df9249a2d9442c65a"
CAL_WORLD_SPEC_SHA256 = "01456c701f63863342820d5a3a87f98e7e3c93fa534aee79d977a3ca26f096d9"
CAL_ROOT_FREEZE_RAW_SHA256 = "14c8c1244e176bb9e2172dfeba0f8dcd78e67c56f4645ec0cc180d4e08dffa20"
CAL_ROOT_FREEZE_PAYLOAD_SHA256 = "892f537848c06887f8072c9ec3a1ddbbcba8dd2eca5d9389377cab3b0f8b0ac3"
EVAL_DESIGN_SHA256 = "e17d257e43a78af6d7a291641ade6825b61c430ba51b8052383cab54dbea31fb"
METRIC_ADDENDUM_SHA256 = "9a6d9c14ad6d067911bd3e754d11e974787afe77a7b4bc05e278d92046ad33e6"
EXPECTED_INPUT_IDENTITY_SHA256 = "deb85e134904879d662ffce58990c04992420a25f0796184e907729f99af1f27"
DIRECTIONS = {
    "tts_to_natural": {"recipient": "natural", "donor": "tts"},
    "natural_to_tts": {"recipient": "tts", "donor": "natural"},
}
CONDITIONS = ("natural", "tts")
CELLS = ("q00", "q10", "q01", "q11", "SELF_GRID")
PARAM_KEYS = ("f0", "time", "sp", "ap", "frequency_hz")
FEATURE_KEYS = (*PARAM_KEYS, "fft_size")
AUDIT_KEYS = {
    "mapped_frames", "mapped_f0", "mapped_ap", "mapped_shape", "donor_time",
    "donor_frame", "donor_voiced", "joint_voiced", "sp_power_error_by_frame",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(value: dict[str, Any], excluded: str = "manifest_sha256") -> str:
    payload = {key: item for key, item in value.items() if key != excluded}
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: Any, np: Any) -> str:
    value = np.asarray(array)
    if value.dtype.hasobject:
        raise TypeError("object arrays are not allowed in scientific artifacts")
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def sha256_required(path_value: str | Path, expected: str, role: str) -> tuple[Path, dict[str, Any]]:
    path = Path(path_value).resolve(strict=True)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{role} raw SHA-256 mismatch: expected {expected}, got {actual}")
    return path, {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def load_npz(path: Path, np: Any, expected_keys: set[str] | None = None) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        keys = list(archive.files)
        if expected_keys is not None and set(keys) != expected_keys:
            raise ValueError(f"NPZ key inventory mismatch at {path}: expected={sorted(expected_keys)}, actual={sorted(keys)}")
        return {key: archive[key].copy() for key in keys}


def _float32_pcm_sha(audio: Any, np: Any) -> str:
    data = np.asarray(audio, dtype="<f4").reshape(-1)
    return hashlib.sha256(data.tobytes(order="C")).hexdigest()


def _read_float_wav(path: Path, meta: dict[str, Any], sf: Any, np: Any, role: str) -> tuple[Any, dict[str, Any]]:
    if sha256_file(path) != meta.get("file_sha256"):
        raise ValueError(f"{role} WAV file hash differs from frozen manifest")
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if int(rate) != SAMPLE_RATE or data.ndim != 2 or data.shape[1] != 1:
        raise ValueError(f"{role} must decode as mono {SAMPLE_RATE} Hz float WAV")
    audio = np.asarray(data[:, 0], dtype=np.float32)
    if len(audio) != int(meta.get("samples", -1)) or int(meta.get("sample_rate_hz", -1)) != SAMPLE_RATE:
        raise ValueError(f"{role} decoded sample count/rate differs from frozen metadata")
    if not np.isfinite(audio).all():
        raise ValueError(f"{role} contains nonfinite PCM")
    pcm_sha = _float32_pcm_sha(audio, np)
    if pcm_sha != meta.get("float32_pcm_sha256"):
        raise ValueError(f"{role} decoded PCM hash differs from frozen manifest")
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return audio, {"path": str(path), "file_sha256": meta["file_sha256"], "float32_pcm_sha256": pcm_sha, "samples": len(audio), "sample_rate_hz": int(rate), "rms": rms, "subtype": str(meta.get("subtype", "unknown"))}


def _load_feature(meta: dict[str, Any], np: Any, input_ledger: dict[str, dict[str, Any]], role: str) -> dict[str, Any]:
    path, entry = sha256_required(meta["path"], str(meta["file_sha256"]), role)
    input_ledger[str(path)] = {**entry, "role": role}
    feature = load_npz(path, np, set(FEATURE_KEYS))
    feature = {key: np.asarray(value) for key, value in feature.items()}
    feature["fft_size"] = int(np.asarray(feature["fft_size"]).item())
    for name in PARAM_KEYS:
        if not np.isfinite(feature[name]).all():
            raise ValueError(f"{role} array {name} contains nonfinite values")
    if np.any(feature["sp"] < 0.0) or np.any(feature["ap"] < -1e-6) or np.any(feature["ap"] > 1.0 + 1e-6):
        raise ValueError(f"{role} SP/AP values are outside their allowed domains")
    if feature["time"].ndim != 1 or feature["f0"].shape != feature["time"].shape:
        raise ValueError(f"{role} F0/time dimensions disagree")
    if feature["sp"].ndim != 2 or feature["ap"].shape != feature["sp"].shape or feature["sp"].shape[0] != len(feature["time"]):
        raise ValueError(f"{role} SP/AP/time dimensions disagree")
    if feature["frequency_hz"].shape != (feature["sp"].shape[1],) or feature["fft_size"] != 2 * (feature["sp"].shape[1] - 1):
        raise ValueError(f"{role} frequency grid/FFT dimensions disagree")
    if feature["time"].size > 1 and np.any(np.diff(feature["time"].astype(np.float64)) <= 0.0):
        raise ValueError(f"{role} WORLD time centers are not strictly increasing")
    shapes = meta.get("feature_shapes")
    if not isinstance(shapes, dict) or any(list(feature[key].shape) != shapes.get(key) for key in ("f0", "time", "sp", "ap", "frequency_hz")):
        raise ValueError(f"{role} arrays differ from frozen feature shapes")
    if int(meta.get("fft_size", -1)) != feature["fft_size"]:
        raise ValueError(f"{role} FFT size differs from frozen feature metadata")
    return feature


def _load_parameter(meta: dict[str, Any], np: Any, input_ledger: dict[str, dict[str, Any]], role: str) -> tuple[dict[str, Any], Path]:
    path, entry = sha256_required(meta["path"], str(meta["file_sha256"]), role)
    input_ledger[str(path)] = {**entry, "role": role}
    params = load_npz(path, np, set(PARAM_KEYS))
    if any(not np.isfinite(np.asarray(params[key])).all() for key in PARAM_KEYS):
        raise ValueError(f"{role} has nonfinite parameter arrays")
    if np.any(np.asarray(params["sp"]) < 0.0) or np.any(np.asarray(params["ap"]) < -1e-6) or np.any(np.asarray(params["ap"]) > 1.0 + 1e-6):
        raise ValueError(f"{role} SP/AP values are outside their allowed domains")
    if int(meta.get("fft_size", -1)) != 2 * (np.asarray(params["sp"]).shape[1] - 1):
        raise ValueError(f"{role} FFT metadata differs from parameter shape")
    if np.asarray(params["frequency_hz"]).shape != (np.asarray(params["sp"]).shape[1],):
        raise ValueError(f"{role} frequency grid shape differs")
    return {key: np.asarray(value) for key, value in params.items()}, path


def _require_equal(actual: Any, expected: Any, np: Any, role: str, *, tolerance: float = 0.0) -> dict[str, Any]:
    left, right = np.asarray(actual), np.asarray(expected)
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError(f"{role} shape/dtype mismatch: {left.shape}/{left.dtype} != {right.shape}/{right.dtype}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"{role} contains nonfinite values")
    error = np.abs(left.astype(np.float64) - right.astype(np.float64))
    max_abs = float(np.max(error, initial=0.0))
    bit_exact = bool(np.array_equal(left, right))
    if max_abs > tolerance:
        raise ValueError(f"{role} exceeds tolerance {tolerance}: max_abs={max_abs}")
    return {"bit_exact": bit_exact, "max_abs_error": max_abs, "tolerance": tolerance, "shape": list(left.shape), "dtype": left.dtype.str}


def _softmax_endpoint_audit(actual: Any, endpoint: Any, np: Any, role: str) -> dict[str, Any]:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(endpoint, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError(f"{role} shape/nonfinite endpoint comparison failed")
    error = np.abs(left - right)
    relative_denominator = np.maximum(np.abs(right), SOFTMAX_RELATIVE_DENOMINATOR_FLOOR)
    relative_error = error / relative_denominator
    allowed = SOFTMAX_ABS_TOL + SOFTMAX_REL_TOL * np.abs(right)
    max_abs = float(np.max(error, initial=0.0))
    max_rel = float(np.max(relative_error, initial=0.0))
    violations = int(np.count_nonzero(error > allowed))
    if violations:
        raise ValueError(
            f"{role} exceeds mixed softmax tolerance: violations={violations}, "
            f"max_abs={max_abs}, max_relative={max_rel}"
        )
    return {
        "passed": True,
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
        "absolute_tolerance": SOFTMAX_ABS_TOL,
        "relative_tolerance": SOFTMAX_REL_TOL,
        "relative_denominator_floor": SOFTMAX_RELATIVE_DENOMINATOR_FLOOR,
        "reference_shape": list(right.shape),
        "violating_elements": violations,
    }


def _epsilon_floor_audit(shape: Any, np: Any) -> dict[str, Any]:
    values = np.asarray(shape, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("epsilon-floor audit requires finite nonnegative 2D shapes")
    added_mass = np.sum(np.maximum(SHAPE_EPSILON - values, 0.0), axis=1, dtype=np.float64)
    upper_bound = values.shape[1] * SHAPE_EPSILON
    observed_max = float(np.max(added_mass, initial=0.0))
    if observed_max > upper_bound + np.finfo(np.float64).eps:
        raise ValueError("observed epsilon-floor added mass exceeds its frequency-bin bound")
    return {
        "row_count": int(values.shape[0]),
        "frequency_bin_count": int(values.shape[1]),
        "observed_max_added_mass": observed_max,
        "observed_mean_added_mass": float(np.mean(added_mass, dtype=np.float64)) if len(added_mass) else 0.0,
        "rows_with_floored_bins": int(np.count_nonzero(np.any(values < SHAPE_EPSILON, axis=1))),
        "upper_bound_per_row": upper_bound,
        "mass_by_row": added_mass.tolist(),
    }


def _stable_softmax(log_rows: Any, np: Any) -> Any:
    values = np.asarray(log_rows, dtype=np.float64)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    denom = np.sum(exponent, axis=1, keepdims=True, dtype=np.float64)
    if np.any(~np.isfinite(exponent)) or np.any(~np.isfinite(denom)) or np.any(denom <= 0.0):
        raise ValueError("softmax produced invalid spectral rows")
    return exponent / denom


def _interp_rows(values: Any, times: Any, query: float, np: Any) -> Any:
    """Linear interpolation with the original WORLD helper's endpoint holding."""
    rows = np.asarray(values, dtype=np.float64)
    axis = np.asarray(times, dtype=np.float64)
    if rows.ndim != 2 or axis.ndim != 1 or rows.shape[0] != len(axis) or len(axis) == 0:
        raise ValueError("log-SP interpolation requires nonempty, aligned time rows")
    right = int(np.searchsorted(axis, query, side="left"))
    if right <= 0:
        return rows[0].copy()
    if right >= len(axis):
        return rows[-1].copy()
    left = right - 1
    span = float(axis[right] - axis[left])
    alpha = 0.0 if span <= 0.0 else min(1.0, max(0.0, (query - float(axis[left])) / span))
    return rows[left] * (1.0 - alpha) + rows[right] * alpha


def _array_entry(path: Path, file_sha256: str, array_key: str, array: Any, np: Any) -> dict[str, Any]:
    value = np.asarray(array)
    return {"path": str(path), "file_sha256": file_sha256, "key": array_key, "array_sha256": array_sha256(value, np), "dtype": value.dtype.str, "shape": list(value.shape)}


def _read_control_files(args: argparse.Namespace, np: Any) -> dict[str, Any]:
    input_paths = {
        "physical_manifest": args.physical_manifest,
        "physical_plan": args.physical_plan,
        "physical_qa": args.physical_qa,
        "source_input_manifest": args.input_manifest,
        "tts_pair_manifest": args.tts_pair_manifest,
        "world_source": args.world_source,
        "execution_bindings": args.execution_bindings,
        "phone_table": args.phone_table,
        "phone_inventory": args.phone_inventory,
        "calibration_world_manifest": args.calibration_manifest,
        "calibration_world_plan": args.calibration_plan,
        "calibration_world_spec": args.calibration_spec,
        "calibration_qa": args.calibration_qa,
        "calibration_freeze": args.calibration_freeze,
        "evaluation_design": args.eval_design,
        "metric_addendum": args.metric_addendum,
        "independent_qa_source": args.independent_qa_source,
    }
    plan_path = args.analysis_plan.resolve(strict=True)
    analysis_plan = load_json(plan_path)
    if analysis_plan.get("manifest_sha256") != fingerprint(analysis_plan):
        raise ValueError("EVAL analysis plan embedded payload fingerprint mismatch")
    if analysis_plan.get("manifest_type") != "world_phone_shape_mean_dynamic_evaluation_plan":
        raise ValueError("analysis plan is not the new evaluation shape-factor plan")
    if analysis_plan.get("status") != "FROZEN_FOR_EVALUATION_WAVEFORM_PRODUCTION":
        raise ValueError("EVAL waveform production is not enabled by a frozen plan")
    if analysis_plan.get("scope") != "evaluation_only" or analysis_plan.get("split") != "evaluation":
        raise ValueError("analysis plan scope/split is not EVAL-only")
    if analysis_plan.get("evaluation_outcomes_read") is not False or analysis_plan.get("evaluation_effects_read") is not False or analysis_plan.get("model_forward_performed") is not False:
        raise ValueError("analysis plan records forbidden outcome/effect/model access")
    actual_source_sha = sha256_file(Path(__file__).resolve())
    if analysis_plan.get("analysis_source_sha256") != actual_source_sha:
        raise ValueError("analysis plan does not bind this exact producer source")
    bindings = analysis_plan.get("input_bindings")
    if not isinstance(bindings, dict):
        raise ValueError("analysis plan input_bindings must be an object")
    if set(bindings) != set(input_paths):
        raise ValueError(f"analysis plan input role inventory mismatch: {sorted(bindings)}")

    fixed_bindings = {
        "physical_manifest": (PHYSICAL_MANIFEST_RAW_SHA256, PHYSICAL_MANIFEST_PAYLOAD_SHA256, "manifest_sha256"),
        "physical_plan": (PHYSICAL_PLAN_RAW_SHA256, PHYSICAL_PLAN_PAYLOAD_SHA256, "manifest_sha256"),
        "physical_qa": (PHYSICAL_QA_RAW_SHA256, None, None),
        "source_input_manifest": (SOURCE_INPUT_MANIFEST_RAW_SHA256, SOURCE_INPUT_MANIFEST_PAYLOAD_SHA256, "manifest_sha256"),
        "tts_pair_manifest": (TTS_PAIR_MANIFEST_RAW_SHA256, TTS_PAIR_MANIFEST_PAYLOAD_SHA256, "manifest_sha256"),
        "world_source": (WORLD_SOURCE_SHA256, None, None),
        "execution_bindings": (EXECUTION_BINDINGS_SHA256, None, None),
        "phone_table": (PHONE_TABLE_RAW_SHA256, PHONE_TABLE_PAYLOAD_SHA256, "manifest_sha256"),
        "phone_inventory": (PHONE_INVENTORY_RAW_SHA256, PHONE_INVENTORY_PAYLOAD_SHA256, "inventory_sha256"),
        "calibration_world_manifest": (CAL_WORLD_MANIFEST_RAW_SHA256, CAL_WORLD_MANIFEST_PAYLOAD_SHA256, "manifest_sha256"),
        "calibration_world_plan": (CAL_WORLD_PLAN_RAW_SHA256, CAL_WORLD_PLAN_PAYLOAD_SHA256, "manifest_sha256"),
        "calibration_world_spec": (CAL_WORLD_SPEC_SHA256, None, None),
        "calibration_freeze": (CAL_ROOT_FREEZE_RAW_SHA256, CAL_ROOT_FREEZE_PAYLOAD_SHA256, "manifest_sha256"),
        "evaluation_design": (EVAL_DESIGN_SHA256, None, None),
        "metric_addendum": (METRIC_ADDENDUM_SHA256, None, None),
    }
    file_entries: dict[str, dict[str, Any]] = {}
    resolved: dict[str, Path] = {}
    documents: dict[str, dict[str, Any]] = {}
    for role, path_arg in input_paths.items():
        binding = bindings[role]
        if not isinstance(binding, dict) or not isinstance(binding.get("raw_sha256"), str):
            raise ValueError(f"analysis plan has no raw SHA binding for {role}")
        expected_raw = binding["raw_sha256"]
        expected_payload = binding.get("payload_sha256")
        payload_field = binding.get("payload_field")
        if role in fixed_bindings:
            fixed_raw, fixed_payload, fixed_field = fixed_bindings[role]
            if expected_raw != fixed_raw or expected_payload != fixed_payload or payload_field != fixed_field:
                raise ValueError(f"frozen input binding differs from the executed reference for {role}")
        path, entry = sha256_required(path_arg, expected_raw, role)
        if binding.get("path") is not None and Path(str(binding["path"])).resolve() != path:
            raise ValueError(f"analysis plan path binding differs for {role}")
        file_entries[role] = entry
        resolved[role] = path
        if payload_field is not None or expected_payload is not None:
            if not isinstance(payload_field, str) or not isinstance(expected_payload, str):
                raise ValueError(f"payload binding is incomplete for {role}")
            document = load_json(path)
            actual_payload = fingerprint(document, excluded=payload_field)
            if document.get(payload_field) != expected_payload or actual_payload != expected_payload:
                raise ValueError(f"embedded/canonical payload identity differs for {role}")
            documents[role] = document
        elif path.suffix.lower() == ".json":
            documents[role] = load_json(path)

    qa_source = resolved["independent_qa_source"]
    if analysis_plan.get("independent_qa_source_sha256") != file_entries["independent_qa_source"]["sha256"]:
        raise ValueError("analysis plan does not bind this exact independent QA source")
    if analysis_plan.get("independent_qa_source_path") is not None and Path(str(analysis_plan["independent_qa_source_path"])).resolve() != qa_source:
        raise ValueError("analysis plan independent QA source path differs from the supplied source")

    # CAL QA and its independent root freeze are run-dependent, but both exact
    # identities must be recorded in the reviewed EVAL plan before this gate.
    cal_qa = documents.get("calibration_qa")
    cal_freeze = documents.get("calibration_freeze")
    if cal_qa is None or cal_qa.get("status") != "PASS":
        raise ValueError("shape-factor CAL independent QA is not PASS")
    for key, expected in (("direction_rows_verified", 40), ("cell_waveforms_verified", 200),
                          ("input_files_verified", 320), ("output_files_verified", 280)):
        if cal_qa.get(key) != expected:
            raise ValueError(f"shape-factor CAL independent QA {key} differs")
    for key in ("evaluation_inputs_read", "evaluation_outcomes_read", "effects_read", "model_forward_performed"):
        if cal_qa.get(key) is not False:
            raise ValueError(f"shape-factor CAL independent QA records forbidden access: {key}")
    if cal_freeze is None or cal_freeze.get("status") != "FROZEN":
        raise ValueError("shape-factor CAL output/QA has not been root-frozen")
    cal_manifest = documents["calibration_world_manifest"]
    if cal_manifest.get("status") != "CAL_FACTORIAL_WAVEFORMS_READY_FOR_ROOT_REVIEW" or cal_manifest.get("split") != "calibration":
        raise ValueError("shape-factor CAL producer manifest is incomplete or wrong-split")
    if cal_manifest.get("evaluation_waveforms_read") is not False or cal_manifest.get("evaluation_outcomes_read") is not False or cal_manifest.get("effects_read") is not False or cal_manifest.get("model_forward_performed") is not False:
        raise ValueError("shape-factor CAL producer records forbidden EVAL/model access")
    cal_plan = documents["calibration_world_plan"]
    if cal_plan.get("scope") != "calibration_only" or cal_plan.get("evaluation_waveforms_read") is not False or cal_plan.get("evaluation_outcomes_read") is not False:
        raise ValueError("shape-factor CAL plan is not CAL-only")
    if cal_freeze.get("calibration_producer_manifest_raw_sha256") != bindings["calibration_world_manifest"]["raw_sha256"]:
        raise ValueError("root CAL freeze does not bind the exact factorial CAL report")
    if cal_freeze.get("calibration_independent_qa_raw_sha256") != bindings["calibration_qa"]["raw_sha256"]:
        raise ValueError("root CAL freeze does not bind the exact independent QA report")

    physical = documents["physical_manifest"]
    physical_plan = documents["physical_plan"]
    physical_qa = documents["physical_qa"]
    if physical.get("manifest_type") != "world_phone_transfer_evaluation_producer" or physical.get("split") != "evaluation":
        raise ValueError("parent WORLD manifest is not the sealed physical EVAL view")
    if physical.get("manifest_sha256") != PHYSICAL_MANIFEST_PAYLOAD_SHA256 or fingerprint(physical) != PHYSICAL_MANIFEST_PAYLOAD_SHA256:
        raise ValueError("parent physical EVAL manifest payload differs")
    if physical.get("status") != "EVALUATION_WAVEFORMS_READY" or physical.get("split") != "evaluation":
        raise ValueError("parent physical EVAL waveforms are incomplete")
    if physical.get("evaluation_outcomes_read") is not False or physical.get("evaluation_waveforms_read") is not True:
        raise ValueError("parent physical EVAL access markers differ from the frozen input")
    if physical.get("world_transfer_plan_sha256") != PHYSICAL_PLAN_PAYLOAD_SHA256:
        raise ValueError("parent physical manifest does not bind the frozen physical plan")
    if str(Path(str(physical.get("world_transfer_plan_path", ""))).resolve()) != str(resolved["physical_plan"]):
        raise ValueError("parent physical manifest plan path differs from the supplied physical plan")
    if physical_plan.get("manifest_sha256") != PHYSICAL_PLAN_PAYLOAD_SHA256 or fingerprint(physical_plan) != PHYSICAL_PLAN_PAYLOAD_SHA256:
        raise ValueError("parent physical EVAL plan payload differs")
    if physical_plan.get("manifest_type") != "world_phone_transfer_plan" or physical_plan.get("status") != "SEALED_BEFORE_ENDPOINT_FORWARD":
        raise ValueError("parent physical plan does not describe the sealed EVAL overlay run")
    if physical_qa.get("status") != "PASS" or physical_qa.get("failures") != []:
        raise ValueError("independent physical EVAL waveform QA is not PASS")
    qa_inputs = physical_qa.get("inputs", {})
    for qa_key, role in (("physical_manifest", "physical_manifest"), ("physical_plan", "physical_plan")):
        if qa_inputs.get(qa_key, {}).get("sha256") != bindings[role]["raw_sha256"]:
            raise ValueError(f"physical QA report does not bind {role}")
    coverage = physical_qa.get("coverage", {})
    expected_coverage = {
        "sample_pairs": 80, "recipient_contexts": 160, "completed_recipient_contexts": 160,
        "original_mfa_qualified_pairs": 79, "physical_overlay_qualified_pairs": 80,
        "expected_physical_wavs": 1920, "expected_npz_artifacts": 1920,
    }
    if any(coverage.get(key) != value for key, value in expected_coverage.items()):
        raise ValueError("independent physical EVAL QA coverage differs from the frozen cohort")
    if physical_qa.get("classification") != "independent EVAL waveform/parameter-dose/local-energy QA; no SyncNet or visual scores read":
        raise ValueError("physical QA report does not carry the frozen waveform-only scope")
    for key in ("evaluation_effects_read", "evaluation_outcomes_read", "effects_read", "model_forward_performed"):
        if key in physical_qa and physical_qa[key] is not False:
            raise ValueError(f"physical QA report records forbidden effect/outcome/model access: {key}")

    phone_table = documents["phone_table"]
    phone_inventory = documents["phone_inventory"]
    if phone_table.get("status") != "FROZEN_FROM_MODEL_INVENTORY_AND_CAL_ONLY_LABELS":
        raise ValueError("phone normalization table is not frozen from CAL")
    if phone_inventory.get("status") != "FROZEN_CAL_ONLY_PHONE_INVENTORY" or phone_inventory.get("selection_split") != "calibration":
        raise ValueError("phone inventory is not the frozen CAL-only inventory")
    if phone_inventory.get("evaluation_textgrids_or_endpoints_read") is not False:
        raise ValueError("phone inventory records EVAL access")
    expected_labels = phone_inventory.get("selected_phone_labels")
    if expected_labels != documents["calibration_world_plan"].get("inferential_phone_labels"):
        raise ValueError("inference labels differ from the frozen CAL-only inventory")
    if analysis_plan.get("inferential_phone_labels") != expected_labels:
        raise ValueError("EVAL plan changes the frozen 16-label family")
    if analysis_plan.get("ordered_evaluation_ids") != [row.get("id") for row in physical.get("records", [])]:
        raise ValueError("EVAL plan order is not the exact sealed physical-manifest order")
    if analysis_plan.get("input_identity_sha256") != EXPECTED_INPUT_IDENTITY_SHA256:
        raise ValueError("EVAL plan input identity digest differs from the frozen 100-pair source identity")
    if analysis_plan.get("fixed_rms") != FIXED_RMS or analysis_plan.get("sample_rate_hz") != SAMPLE_RATE or analysis_plan.get("frame_period_ms") != FRAME_PERIOD_MS:
        raise ValueError("EVAL plan fixed RMS or WORLD clock differs")
    if analysis_plan.get("directions") != list(DIRECTIONS) or analysis_plan.get("cells") != list(CELLS):
        raise ValueError("EVAL plan direction/cell order differs from the frozen source")
    if analysis_plan.get("intended_pair_count") != 80 or analysis_plan.get("intended_speaker_count") != 40 or analysis_plan.get("intended_direction_rows") != 160 or analysis_plan.get("intended_cell_waveforms") != 800:
        raise ValueError("EVAL plan cohort/output count differs")
    if analysis_plan.get("q00_q11_softmax_abs_tolerance") != SOFTMAX_ABS_TOL or analysis_plan.get("q00_q11_softmax_rel_tolerance") != SOFTMAX_REL_TOL or analysis_plan.get("sp_row_power_rel_tolerance") != SP_POWER_REL_TOL or analysis_plan.get("waveform_bridge_max_abs_tolerance") != WAVE_BRIDGE_MAX_ABS_TOL:
        raise ValueError("EVAL plan numeric gates differ from the CAL protocol")

    # A separate root freeze must explicitly release physical waveform writing
    # and bind this exact source and plan. It is not a model/effect report.
    evaluation_freeze_path = args.evaluation_freeze.resolve(strict=True)
    evaluation_freeze = load_json(evaluation_freeze_path)
    if evaluation_freeze.get("status") != "FROZEN" or evaluation_freeze.get("evaluation_waveform_production_authorized") is not True:
        raise ValueError("new EVAL physical plan has not been explicitly frozen for waveform production")
    if evaluation_freeze.get("analysis_plan_sha256") != analysis_plan.get("manifest_sha256"):
        raise ValueError("new EVAL freeze does not bind the analysis plan payload")
    if evaluation_freeze.get("analysis_plan_raw_sha256") != sha256_file(plan_path):
        raise ValueError("new EVAL freeze does not bind the analysis plan bytes")
    if evaluation_freeze.get("analysis_source_sha256") != actual_source_sha:
        raise ValueError("new EVAL freeze does not bind this source")
    if evaluation_freeze.get("evaluation_effects_read") is not False or evaluation_freeze.get("evaluation_outcomes_read") is not False:
        raise ValueError("new EVAL freeze records forbidden effect/outcome inspection")
    file_entries["evaluation_freeze"] = {"path": str(evaluation_freeze_path), "sha256": sha256_file(evaluation_freeze_path), "bytes": evaluation_freeze_path.stat().st_size}
    documents["evaluation_freeze"] = evaluation_freeze
    resolved["evaluation_freeze"] = evaluation_freeze_path
    return {
        "manifest": physical,
        "phone_table": phone_table,
        "phone_inventory": phone_inventory,
        "calibration_manifest": cal_manifest,
        "calibration_plan": cal_plan,
        "calibration_qa": cal_qa,
        "calibration_freeze": cal_freeze,
        "physical_qa": physical_qa,
        "analysis_plan": analysis_plan,
        "documents": documents,
        "resolved": resolved,
        "control_file_inventory": file_entries,
    }


def _import_frozen_world_source(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("frozen_world_producer_for_shape_factorization", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the frozen WORLD producer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for symbol in ("normalized_sp", "transfer_parameters", "synthesize_fixed", "write_float_wav"):
        if not callable(getattr(module, symbol, None)):
            raise RuntimeError(f"frozen WORLD producer is missing {symbol}")
    return module


def _compare_transfer(saved: dict[str, Any], computed: dict[str, Any], np: Any, role: str) -> dict[str, Any]:
    if set(saved) != AUDIT_KEYS or set(computed) != AUDIT_KEYS:
        raise ValueError(f"{role} transfer audit key inventory differs")
    comparisons = {}
    for key in sorted(AUDIT_KEYS):
        left, right = np.asarray(saved[key]), np.asarray(computed[key])
        if left.dtype != right.dtype or left.shape != right.shape:
            raise ValueError(f"{role}/{key} transfer audit dtype/shape mismatch")
        if left.dtype.kind == "f":
            equal = bool(np.array_equal(left, right, equal_nan=True))
        else:
            equal = bool(np.array_equal(left, right))
        if not equal:
            delta = np.abs(left.astype(np.float64) - right.astype(np.float64)) if left.dtype.kind in "fiu" else None
            max_abs = float(np.max(delta, initial=0.0)) if delta is not None else None
            raise ValueError(f"{role}/{key} differs from recomputed frozen transfer audit; max_abs={max_abs}")
        comparisons[key] = {"bit_exact": True, "dtype": left.dtype.str, "shape": list(left.shape), "array_sha256": array_sha256(left, np)}
    return comparisons


def _read_source_wav(meta: dict[str, Any], sf: Any, np: Any, input_ledger: dict[str, dict[str, Any]], role: str) -> tuple[Any, dict[str, Any]]:
    # The physical WORLD producer analyzes the staged audio path. Bind that
    # consumed file and retain the original source reference independently.
    # The source and staged copies are required to be byte-identical by the
    # parent manifest; do not try to open a Windows-origin source path remotely.
    source_sha = str(meta["source_audio_sha256"])
    staged_sha = str(meta["staged_audio_sha256"])
    if source_sha != staged_sha:
        raise ValueError(f"{role} staged audio bytes differ from the frozen source file SHA")
    path, entry = sha256_required(meta["staged_audio_path"], staged_sha, role)
    input_ledger[str(path)] = {**entry, "role": role}
    pcm, rate = sf.read(str(path), dtype="float32", always_2d=True)
    pcm16, rate16 = sf.read(str(path), dtype="int16", always_2d=True)
    if int(rate) != SAMPLE_RATE or int(rate16) != SAMPLE_RATE or pcm.ndim != 2 or pcm.shape[1] != 1 or pcm16.shape != pcm.shape:
        raise ValueError(f"{role} must decode as mono {SAMPLE_RATE} Hz with matching PCM16/float32 clocks")
    audio = np.asarray(pcm[:, 0], dtype=np.float32)
    pcm16_sha = hashlib.sha256(np.asarray(pcm16, dtype="<i2").tobytes(order="C")).hexdigest()
    float_sha = _float32_pcm_sha(audio, np)
    if len(audio) != int(meta.get("raw_decoded_samples", -1)) or int(meta.get("sample_rate_hz", -1)) != SAMPLE_RATE:
        raise ValueError(f"{role} decoded sample count/rate differs from frozen metadata")
    if float_sha != meta.get("raw_float32_pcm_sha256") or pcm16_sha != meta.get("raw_decoded_pcm16_sha256"):
        raise ValueError(f"{role} decoded PCM identity differs from frozen metadata")
    if not np.isfinite(audio).all():
        raise ValueError(f"{role} contains nonfinite PCM")
    return audio, {"path": str(path), "file_sha256": entry["sha256"], "original_source_path": str(meta["source_audio_path"]),
                   "original_source_file_sha256": source_sha, "staged_audio_path": str(path),
                   "staged_audio_file_sha256": entry["sha256"], "float32_pcm_sha256": float_sha,
                   "pcm16_sha256": pcm16_sha, "samples": int(len(audio)), "sample_rate_hz": int(rate),
                   "channels": 1, "raw_rms": float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))}


def _read_inputs(args: argparse.Namespace, np: Any, sf: Any) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    controls = _read_control_files(args, np)
    world = _import_frozen_world_source(controls["resolved"]["world_source"])
    manifest = controls["manifest"]
    ordered_ids = controls["analysis_plan"]["ordered_evaluation_ids"]
    source_manifest = controls["documents"]["source_input_manifest"]
    tts_manifest = controls["documents"]["tts_pair_manifest"]
    if source_manifest.get("status") != "complete" or source_manifest.get("frozen_before_tts") is not True or source_manifest.get("records_expected") != 100:
        raise ValueError("source input manifest is not the complete frozen 100-pair manifest")
    if tts_manifest.get("status") != "complete" or tts_manifest.get("records_expected") != 100 or tts_manifest.get("split_counts") != {"calibration": 20, "evaluation": 80}:
        raise ValueError("TTS pair manifest is not the complete frozen 20/80 cohort")
    source_rows = source_manifest.get("records")
    tts_rows = tts_manifest.get("records")
    if not isinstance(source_rows, list) or len(source_rows) != 100 or not isinstance(tts_rows, list) or len(tts_rows) != 100:
        raise ValueError("source/TTS pair manifest row counts differ from the frozen 100-pair cohort")
    identity_objects = []
    source_by_id: dict[str, dict[str, Any]] = {}
    tts_by_id: dict[str, dict[str, Any]] = {}
    for source_row, tts_row in zip(source_rows, tts_rows):
        identity = {key: source_row[key] for key in ("id", "speaker", "text", "split")}
        identity["ref_sha256"] = source_row["ref_sha256"]
        identity_objects.append(identity)
        sample_id = str(source_row["id"])
        if sample_id in source_by_id or sample_id != str(tts_row.get("id")):
            raise ValueError("source/TTS pair manifest IDs are duplicate or out of order")
        if tts_row.get("input_identity") != identity or tts_row.get("input_identity_sha256") != hashlib.sha256(canonical_json(identity)).hexdigest():
            raise ValueError(f"{sample_id} TTS input identity differs from the frozen source row")
        if tts_row.get("natural_raw", {}).get("sha256") != source_row.get("ref_sha256"):
            raise ValueError(f"{sample_id} natural reference hash differs between input and TTS pair manifests")
        if tts_row.get("qwen_canonical_16k", {}).get("sha256") is None:
            raise ValueError(f"{sample_id} frozen TTS manifest lacks canonical 16 kHz audio identity")
        source_by_id[sample_id] = source_row
        tts_by_id[sample_id] = tts_row
    identity_sha = hashlib.sha256(canonical_json(identity_objects)).hexdigest()
    if identity_sha != EXPECTED_INPUT_IDENTITY_SHA256 or tts_manifest.get("input_identity_sha256") != identity_sha:
        raise ValueError("100-pair source/TTS input identity digest differs from the frozen cohort")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 80:
        raise ValueError("sealed physical EVAL manifest must contain exactly 80 records")
    if [str(row.get("id")) for row in records] != ordered_ids or len(set(ordered_ids)) != 80:
        raise ValueError("physical EVAL record order differs from the frozen 80-row plan")
    speaker_ids: dict[str, set[str]] = {}
    for row in records:
        speaker_ids.setdefault(str(row.get("speaker")), set()).add(str(row.get("id")))
    if len(speaker_ids) != 40 or any(len(ids) != 2 for ids in speaker_ids.values()):
        raise ValueError("physical EVAL records do not contain the 40-speaker roster")
    ledger: dict[str, dict[str, Any]] = {}
    pairs: list[dict[str, Any]] = []
    for ordered_index, record in enumerate(records):
        sample_id = str(record["id"])
        if record.get("ordered_index") != ordered_index or record.get("split") != "evaluation":
            raise ValueError(f"{sample_id} is out of order or not in the EVAL split")
        source_row = source_by_id.get(sample_id)
        tts_row = tts_by_id.get(sample_id)
        if source_row is None or tts_row is None or source_row.get("split") != "evaluation":
            raise ValueError(f"{sample_id} is not an EVAL row in the frozen source/TTS manifests")
        identity = {key: source_row[key] for key in ("id", "speaker", "text", "split")}
        identity["ref_sha256"] = source_row["ref_sha256"]
        identity_sha = hashlib.sha256(canonical_json(identity)).hexdigest()
        if record.get("source_identity_sha256") != identity_sha or tts_row.get("input_identity_sha256") != identity_sha:
            raise ValueError(f"{sample_id} source identity digest differs across the physical and source/TTS manifests")
        if record.get("speaker") != source_row.get("speaker") or record.get("speaker") != tts_row.get("speaker"):
            raise ValueError(f"{sample_id} speaker differs across physical/source/TTS manifests")
        condition_records = record.get("conditions")
        if not isinstance(condition_records, dict):
            condition_records = {condition: record.get(condition) for condition in CONDITIONS}
        sides: dict[str, Any] = {}
        for condition in CONDITIONS:
            info = condition_records.get(condition)
            if not isinstance(info, dict) or info.get("condition_qualified") is not True:
                raise ValueError(f"{sample_id}/{condition} is not qualified in the frozen physical input")
            if info.get("source_identity_sha256") != record.get("source_identity_sha256"):
                raise ValueError(f"{sample_id}/{condition} source identity differs from its pair")
            expected_source_sha = source_row["ref_sha256"] if condition == "natural" else tts_row["qwen_canonical_16k"]["sha256"]
            if info.get("source_audio_sha256") != expected_source_sha:
                raise ValueError(f"{sample_id}/{condition} physical source hash differs from frozen input/TTS manifest")
            raw_audio, raw_meta = _read_source_wav(info, sf, np, ledger, f"{sample_id}/{condition}/A_RAW_source")
            feature = _load_feature(info["world_features"], np, ledger, f"{sample_id}/{condition}/WORLD_features")
            fixed_path, fixed_entry = sha256_required(info["fixed_audio"]["path"], str(info["fixed_audio"]["file_sha256"]), f"{sample_id}/{condition}/fixed_audio")
            ledger[str(fixed_path)] = {**fixed_entry, "role": f"{sample_id}/{condition}/fixed_audio"}
            fixed_audio, fixed_meta = _read_float_wav(fixed_path, info["fixed_audio"], sf, np, f"{sample_id}/{condition}/fixed_audio")
            fixed_rms = float(np.sqrt(np.mean(fixed_audio.astype(np.float64) ** 2)))
            if abs(fixed_rms - FIXED_RMS) > FIXED_RMS_ABS_TOL:
                raise ValueError(f"{sample_id}/{condition} fixed audio RMS differs from the frozen target")
            if len(raw_audio) != len(fixed_audio):
                raise ValueError(f"{sample_id}/{condition} RAW and fixed source lengths differ")
            frame_steps = np.diff(feature["time"].astype(np.float64))
            if frame_steps.size and not np.allclose(frame_steps, FRAME_PERIOD_MS / 1000.0, rtol=0.0, atol=1e-7):
                raise ValueError(f"{sample_id}/{condition} WORLD frame clock is not the frozen 5 ms grid")
            if feature["time"].size and (float(feature["time"][0]) < -1e-12 or float(feature["time"][-1]) > len(fixed_audio) / SAMPLE_RATE + FRAME_PERIOD_MS / 1000.0 + 1e-9):
                raise ValueError(f"{sample_id}/{condition} WORLD feature centers fall outside the audio clock")
            sides[condition] = {"feature": feature, "audio": fixed_audio, "audio_meta": fixed_meta,
                                "raw_audio": raw_audio, "raw_audio_meta": raw_meta, "info": info}
        if not np.array_equal(sides["natural"]["feature"]["frequency_hz"], sides["tts"]["feature"]["frequency_hz"]):
            raise ValueError(f"{sample_id} N/T frequency grids differ")
        directions: dict[str, Any] = {}
        if record.get("pair_qualified") is not True or record.get("physical_alignment_pair_qualified") is not True:
            raise ValueError(f"{sample_id} is not physically qualified in the frozen EVAL view")
        for direction_name, rule in DIRECTIONS.items():
            direction_meta = record["directions"][direction_name]
            if direction_meta.get("recipient_condition") != rule["recipient"] or direction_meta.get("donor_condition") != rule["donor"]:
                raise ValueError(f"{sample_id}/{direction_name} recipient/donor identity mismatch")
            rec_condition, don_condition = rule["recipient"], rule["donor"]
            rec_feature = sides[rec_condition]["feature"]
            don_feature = sides[don_condition]["feature"]
            inputs: dict[str, Any] = {}
            for role, field in (("phone_map", "phone_map"), ("transfer_audit", "parameter_transfer_audit")):
                meta = direction_meta[field]
                path, entry = sha256_required(meta["path"], str(meta["file_sha256"]), f"{sample_id}/{direction_name}/{role}")
                ledger[str(path)] = {**entry, "role": f"{sample_id}/{direction_name}/{role}"}
                inputs[f"{role}_path"] = path
                inputs[f"{role}_meta"] = meta
                inputs[role] = load_json(path) if role == "phone_map" else load_npz(path, np, AUDIT_KEYS)
            phone_map = inputs["phone_map"]
            if canonical_json(phone_map) != canonical_json(direction_meta.get("mapping")):
                raise ValueError(f"{sample_id}/{direction_name} embedded phone map differs from the sealed map file")
            if phone_map.get("split") != "evaluation" or phone_map.get("id") != sample_id or phone_map.get("speaker") != str(record["speaker"]):
                raise ValueError(f"{sample_id}/{direction_name} phone-map identity/split mismatch")
            if phone_map.get("pair_qualified") is not True or phone_map.get("recipient_condition") != rec_condition or phone_map.get("donor_condition") != don_condition:
                raise ValueError(f"{sample_id}/{direction_name} phone map is not qualified for this direction")
            if phone_map.get("recipient_source_pcm_sha256") != condition_records[rec_condition]["source_audio_sha256"] or phone_map.get("donor_source_pcm_sha256") != condition_records[don_condition]["source_audio_sha256"]:
                raise ValueError(f"{sample_id}/{direction_name} phone map is bound to different source audio")
            if int(direction_meta["phone_map"].get("matched_phone_count", -1)) != len(phone_map.get("matched_phones", [])) or not phone_map.get("matched_phones"):
                raise ValueError(f"{sample_id}/{direction_name} matched phone count is invalid")
            computed_arms, computed_transfer = world.transfer_parameters(rec_feature, don_feature, phone_map, np)
            audit_checks = _compare_transfer(inputs["transfer_audit"], computed_transfer["detail"], np, f"{sample_id}/{direction_name}")
            inputs["recomputed_transfer"] = computed_transfer
            inputs["audit_checks"] = audit_checks
            inputs["parameters"] = {}
            inputs["old_waves"] = {}
            for arm in ("RT", "SP"):
                arm_meta = direction_meta["arms"][arm]
                if arm_meta.get("status") != "ok":
                    raise ValueError(f"{sample_id}/{direction_name}/{arm} original physical arm is not successful")
                params_meta = arm_meta["parameters"]
                params, params_path = _load_parameter(params_meta, np, ledger, f"{sample_id}/{direction_name}/{arm}/parameters")
                expected_params = computed_arms[arm]
                shape_checks = {}
                for key in ("f0", "sp", "ap"):
                    shape_checks[key] = _require_equal(params[key], expected_params[key], np, f"{sample_id}/{direction_name}/{arm}/{key}")
                shape_checks["time"] = _require_equal(params["time"], rec_feature["time"], np, f"{sample_id}/{direction_name}/{arm}/time")
                shape_checks["frequency_hz"] = _require_equal(params["frequency_hz"], rec_feature["frequency_hz"], np, f"{sample_id}/{direction_name}/{arm}/frequency_hz")
                inputs["parameters"][arm] = {"arrays": params, "meta": params_meta, "path": params_path, "checks": shape_checks}
                wave_path, wave_entry = sha256_required(arm_meta["path"], str(arm_meta["file_sha256"]), f"{sample_id}/{direction_name}/{arm}/waveform")
                ledger[str(wave_path)] = {**wave_entry, "role": f"{sample_id}/{direction_name}/{arm}/waveform"}
                wave, wave_meta = _read_float_wav(wave_path, arm_meta, sf, np, f"{sample_id}/{direction_name}/{arm}/waveform")
                wave_rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
                if len(wave) != len(sides[rec_condition]["audio"]) or abs(wave_rms - FIXED_RMS) > FIXED_RMS_ABS_TOL:
                    raise ValueError(f"{sample_id}/{direction_name}/{arm} old waveform clock or RMS differs")
                inputs["old_waves"][arm] = {"audio": wave, "meta": wave_meta, "path": wave_path}
            rt_params = inputs["parameters"]["RT"]["arrays"]
            sp_params = inputs["parameters"]["SP"]["arrays"]
            for key in ("f0", "time", "ap", "frequency_hz"):
                _require_equal(sp_params[key], rt_params[key], np, f"{sample_id}/{direction_name}/SP_vs_RT/{key}")
            directions[direction_name] = {
                "meta": direction_meta, "inputs": inputs,
                "recipient_condition": rec_condition, "donor_condition": don_condition,
            }
        pairs.append({
            "id": sample_id, "speaker": str(record["speaker"]), "ordered_index": ordered_index,
            "source_identity_sha256": record["source_identity_sha256"],
            "mfa_pair_qualified": bool(record.get("mfa_pair_qualified")),
            "physical_alignment_pair_qualified": bool(record.get("physical_alignment_pair_qualified")),
            "physical_alignment_overlay_used": bool(record.get("physical_alignment_overlay_used")),
            "physical_overlay_record_sha256": record.get("physical_overlay_record_sha256"),
            "sides": sides, "directions": directions,
        })
    expected_input_files = 80 * 18
    if len(pairs) != 80 or len(ledger) != expected_input_files:
        raise ValueError(f"EVAL input inventory incomplete: pairs={len(pairs)}, files={len(ledger)}, expected=80/{expected_input_files}")
    return controls, pairs, ledger


def _make_token_layout(phone_map: dict[str, Any], times: Any, mapped: Any, duration: float, np: Any) -> tuple[list[dict[str, Any]], Any]:
    world_times = np.asarray(times, dtype=np.float64)
    mapped_frames = np.asarray(mapped, dtype=bool)
    if mapped_frames.shape != world_times.shape:
        raise ValueError("transfer mapped-frame mask does not match recipient time clock")
    phones = phone_map.get("matched_phones", [])
    recipient_indices = [int(item["recipient_phone_index"]) for item in phones]
    donor_indices = [int(item["donor_phone_index"]) for item in phones]
    if len(set(recipient_indices)) != len(recipient_indices) or len(set(donor_indices)) != len(donor_indices):
        raise ValueError("matched phone map repeats a recipient or donor token index")
    claim_count = np.zeros(len(world_times), dtype=np.int16)
    raw_masks = []
    for phone in phones:
        if phone.get("recipient_normalized_label") != phone.get("donor_normalized_label"):
            raise ValueError("matched LCS pair has nonidentical normalized labels")
        r0, r1 = map(float, phone["recipient_interval_s"])
        if not (math.isfinite(r0) and math.isfinite(r1) and r1 > r0):
            raise ValueError("matched recipient token has an invalid interval")
        mask = (world_times >= r0) & (world_times < r1)
        if abs(r1 - duration) <= 1e-9:
            mask |= np.abs(world_times - r1) <= 1e-9
        claim_count += mask.astype(np.int16)
        raw_masks.append(mask)
    ambiguous = claim_count > 1
    tokens = []
    assigned = np.zeros(len(world_times), dtype=bool)
    for token_index, (phone, interval_mask) in enumerate(zip(phones, raw_masks)):
        token_map_mask = interval_mask & mapped_frames
        eligible_mask = token_map_mask & ~ambiguous
        if np.any(assigned & eligible_mask):
            raise ValueError("a recipient WORLD frame was assigned to more than one phone token")
        assigned |= eligible_mask
        tokens.append({
            "token_index": token_index,
            "phone": phone,
            "interval_mask": interval_mask,
            "mapped_mask": token_map_mask,
            "eligible_mask": eligible_mask,
            "recipient_interval_frame_count": int(np.count_nonzero(interval_mask)),
            "mapped_frame_count": int(np.count_nonzero(token_map_mask)),
            "ambiguous_frame_count": int(np.count_nonzero(token_map_mask & ambiguous)),
            "eligible_frame_count": int(np.count_nonzero(eligible_mask)),
        })
    mapped_unclaimed = mapped_frames & (claim_count == 0)
    if np.any(mapped_unclaimed):
        raise ValueError("frozen transfer audit maps frames outside every matched recipient phone")
    return tokens, ambiguous


def _mean(values: Any, np: Any) -> float | None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.mean(arr)) if arr.size else None


def _token_adjacent_metric(shape: Any, indices: Any, np: Any) -> dict[str, Any]:
    values = np.asarray(shape, dtype=np.float64)
    frame_ids = np.asarray(indices, dtype=np.int64)
    if frame_ids.size < 2:
        return {"mean_abs_adjacent_delta_log_shape": None, "consecutive_pair_count": 0}
    pair_left = frame_ids[:-1][np.diff(frame_ids) == 1]
    if pair_left.size == 0:
        return {"mean_abs_adjacent_delta_log_shape": None, "consecutive_pair_count": 0}
    log_shape = np.log(np.maximum(values, SHAPE_EPSILON))
    delta = np.abs(log_shape[pair_left + 1] - log_shape[pair_left])
    return {"mean_abs_adjacent_delta_log_shape": float(np.mean(delta, dtype=np.float64)), "consecutive_pair_count": int(len(pair_left)), "p50_binwise_abs_delta": float(np.median(delta))}


def _self_grid_shape(
    own_log_shape: Any,
    rec_times: Any,
    don_times: Any,
    phone: dict[str, Any],
    eligible_indices: Any,
    donor_duration: float,
    np: Any,
) -> tuple[Any | None, dict[str, Any]]:
    r0, r1 = map(float, phone["recipient_interval_s"])
    d0, d1 = map(float, phone["donor_interval_s"])
    rtimes_all = np.asarray(rec_times, dtype=np.float64)
    dtimes_all = np.asarray(don_times, dtype=np.float64)
    indices = np.asarray(eligible_indices, dtype=np.int64)
    donor_mask = (dtimes_all >= d0) & (dtimes_all < d1)
    if abs(d1 - donor_duration) <= 1e-9:
        donor_mask |= np.abs(dtimes_all - d1) <= 1e-9
    donor_grid = dtimes_all[donor_mask]
    if indices.size == 0 or donor_grid.size == 0:
        return None, {"status": "no_common_frame_domain", "recipient_frame_count": int(indices.size), "donor_grid_center_count": int(donor_grid.size)}
    rec_grid = rtimes_all[indices]
    if rec_grid.size > 1 and np.any(np.diff(rec_grid) <= 0.0):
        raise ValueError("SELF_GRID recipient centers are not strictly increasing")
    if donor_grid.size > 1 and np.any(np.diff(donor_grid) <= 0.0):
        raise ValueError("SELF_GRID donor centers are not strictly increasing")
    own_rows = np.asarray(own_log_shape, dtype=np.float64)[indices]
    intermediate = np.empty((len(donor_grid), own_rows.shape[1]), dtype=np.float64)
    for j, donor_time in enumerate(donor_grid):
        phase = min(1.0, max(0.0, (float(donor_time) - d0) / (d1 - d0)))
        recipient_query = r0 + phase * (r1 - r0)
        intermediate[j] = _interp_rows(own_rows, rec_grid, recipient_query, np)
    back = np.empty_like(own_rows)
    for j, recipient_time in enumerate(rec_grid):
        phase = min(1.0, max(0.0, (float(recipient_time) - r0) / (r1 - r0)))
        donor_query = d0 + phase * (d1 - d0)
        back[j] = _interp_rows(intermediate, donor_grid, donor_query, np)
    return _stable_softmax(back, np), {
        "status": "ok",
        "recipient_frame_count": int(indices.size),
        "donor_grid_center_count": int(donor_grid.size),
        "operation": "recipient_log_shape_to_donor_5ms_centers_then_back_to_recipient_centers; linear interpolation; endpoint holding",
        "recipient_frame_indices": indices.tolist(),
    }


def _factorize_direction(
    pair: dict[str, Any], direction_name: str, direction: dict[str, Any],
    world: Any, np: Any, sf: Any, output_root: Path,
) -> dict[str, Any]:
    sample_id = pair["id"]
    recipient_condition = direction["recipient_condition"]
    donor_condition = direction["donor_condition"]
    recipient = pair["sides"][recipient_condition]
    donor = pair["sides"][donor_condition]
    inputs = direction["inputs"]
    rec_feature = recipient["feature"]
    don_feature = donor["feature"]
    npz_meta = inputs["parameters"]
    rt_params = npz_meta["RT"]["arrays"]
    sp_params = npz_meta["SP"]["arrays"]
    rec_sp = np.asarray(rt_params["sp"], dtype=np.float64)
    rec_power = np.sum(rec_sp, axis=1, dtype=np.float64)
    if not np.isfinite(rec_power).all() or np.any(rec_power <= 0.0):
        raise ValueError(f"{sample_id}/{direction_name} recipient SP has a nonpositive/nonfinite row power")
    own_shape = world.normalized_sp(rec_sp, np)
    if not np.isfinite(own_shape).all() or np.any(own_shape < 0.0):
        raise ValueError(f"{sample_id}/{direction_name} normalized recipient SP is invalid")
    mapped_shape = np.asarray(inputs["transfer_audit"]["mapped_shape"], dtype=np.float64)
    mapped_frames = np.asarray(inputs["transfer_audit"]["mapped_frames"], dtype=bool)
    if mapped_shape.shape != own_shape.shape or mapped_frames.shape != rec_feature["time"].shape:
        raise ValueError(f"{sample_id}/{direction_name} mapped shape/frame clock mismatch")
    if np.any(mapped_shape[mapped_frames] < 0.0) or not np.isfinite(mapped_shape).all():
        raise ValueError(f"{sample_id}/{direction_name} mapped shape contains invalid values")
    mapped_row_sums = np.sum(mapped_shape[mapped_frames], axis=1, dtype=np.float64)
    if mapped_row_sums.size and float(np.max(np.abs(mapped_row_sums - 1.0))) > PARAMETER_ABS_TOL:
        raise ValueError(f"{sample_id}/{direction_name} saved donor mapped shapes are not normalized")
    times = np.asarray(rt_params["time"], dtype=np.float64)
    duration = len(recipient["audio"]) / SAMPLE_RATE
    donor_duration = len(donor["audio"]) / SAMPLE_RATE
    tokens, ambiguous_mask = _make_token_layout(inputs["phone_map"], times, mapped_frames, duration, np)
    own_log = np.log(np.maximum(own_shape, SHAPE_EPSILON))
    donor_log = np.log(np.maximum(mapped_shape, SHAPE_EPSILON))
    recipient_floor_all = _epsilon_floor_audit(own_shape, np)
    mapped_donor_floor_all = _epsilon_floor_audit(mapped_shape[mapped_frames], np)

    cell_shapes = {name: own_shape.copy() for name in CELLS}
    sp_endpoint_shape = np.asarray(sp_params["sp"], dtype=np.float64) / rec_power[:, None]
    cell_shapes["SELF_GRID"] = own_shape.copy()
    eligible_frame_mask = np.zeros(mapped_frames.shape, dtype=bool)
    for token in tokens:
        eligible_frame_mask |= np.asarray(token["eligible_mask"], dtype=bool)
    means_arrays: dict[str, Any] = {}
    token_rows: list[dict[str, Any]] = []
    q00_formula_audits: list[dict[str, Any]] = []
    q11_mapped_formula_audits: list[dict[str, Any]] = []
    q11_endpoint_formula_audits: list[dict[str, Any]] = []
    q00_full_endpoint_audits: list[dict[str, Any]] = []
    q11_full_endpoint_audits: list[dict[str, Any]] = []
    epsilon_floor_audits: list[dict[str, Any]] = []
    self_grid_rows = 0

    for token in tokens:
        phone = token["phone"]
        idx = np.flatnonzero(token["eligible_mask"])
        token_id = int(token["token_index"])
        token_prefix = f"token_{token_id:04d}"
        row: dict[str, Any] = {
            "token_index": token_id,
            "recipient_phone_index": int(phone["recipient_phone_index"]),
            "donor_phone_index": int(phone["donor_phone_index"]),
            "word_index": phone.get("word_index"),
            "word_label": phone.get("word_label"),
            "phone_label": str(phone["recipient_normalized_label"]),
            "recipient_raw_label": str(phone["recipient_raw_label"]),
            "donor_raw_label": str(phone["donor_raw_label"]),
            "phone_in_frozen16_inventory": str(phone["recipient_normalized_label"]) in pair["phone_labels"],
            "recipient_interval_s": list(map(float, phone["recipient_interval_s"])),
            "donor_interval_s": list(map(float, phone["donor_interval_s"])),
            "recipient_interval_frame_count": token["recipient_interval_frame_count"],
            "mapped_frame_count": token["mapped_frame_count"],
            "ambiguous_frame_count": token["ambiguous_frame_count"],
            "eligible_frame_count": token["eligible_frame_count"],
            "mean_domain": "unique common mapped 5ms recipient WORLD centers in this exact token only",
        }
        if idx.size == 0:
            row.update({"status": "no_eligible_frames", "mu_recipient_array_key": None, "mu_donor_array_key": None, "frame_indices_array_key": None})
            token_rows.append(row)
            continue
        mu_r = np.mean(own_log[idx], axis=0, dtype=np.float64)
        mu_d = np.mean(donor_log[idx], axis=0, dtype=np.float64)
        rho_r = own_log[idx] - mu_r[None, :]
        rho_d = donor_log[idx] - mu_d[None, :]
        l_recon_r = mu_r[None, :] + rho_r
        l_recon_d = mu_d[None, :] + rho_d
        if float(np.max(np.abs(l_recon_r - own_log[idx]))) > PARAMETER_ABS_TOL or float(np.max(np.abs(l_recon_d - donor_log[idx]))) > PARAMETER_ABS_TOL:
            raise ValueError(f"{sample_id}/{direction_name}/{token_prefix} mean/residual reconstruction failed")
        expected_q00 = _stable_softmax(mu_r[None, :] + rho_r, np)
        expected_q10 = _stable_softmax(mu_d[None, :] + rho_r, np)
        expected_q01 = _stable_softmax(mu_r[None, :] + rho_d, np)
        expected_q11 = _stable_softmax(mu_d[None, :] + rho_d, np)
        q00_match = _softmax_endpoint_audit(expected_q00, own_shape[idx], np, f"{sample_id}/{direction_name}/{token_prefix}/q00")
        q11_mapped_match = _softmax_endpoint_audit(expected_q11, mapped_shape[idx], np, f"{sample_id}/{direction_name}/{token_prefix}/q11_mapped")
        q11_endpoint_match = _softmax_endpoint_audit(expected_q11, sp_endpoint_shape[idx], np, f"{sample_id}/{direction_name}/{token_prefix}/q11_saved_SP")
        q00_formula_audits.append(q00_match)
        q11_mapped_formula_audits.append(q11_mapped_match)
        q11_endpoint_formula_audits.append(q11_endpoint_match)
        own_floor = _epsilon_floor_audit(own_shape[idx], np)
        donor_floor = _epsilon_floor_audit(mapped_shape[idx], np)
        epsilon_floor_audits.append({"token_index": token_id, "recipient": own_floor, "mapped_donor": donor_floor})

        # Keep formula-generated rows until the full-grid endpoint bridge is
        # checked. Endpoint arrays are copied only after that gate passes.
        cell_shapes["q00"][idx] = expected_q00
        cell_shapes["q10"][idx] = expected_q10
        cell_shapes["q01"][idx] = expected_q01
        cell_shapes["q11"][idx] = expected_q11

        component_keys = {
            "mu_recipient_array_key": f"{token_prefix}_mu_recipient",
            "mu_donor_array_key": f"{token_prefix}_mu_donor",
            "frame_indices_array_key": f"{token_prefix}_frame_indices",
        }
        means_arrays[component_keys["mu_recipient_array_key"]] = mu_r
        means_arrays[component_keys["mu_donor_array_key"]] = mu_d
        means_arrays[component_keys["frame_indices_array_key"]] = idx.astype(np.int32)

        self_shape, self_meta = _self_grid_shape(own_log, times, don_feature["time"], phone, idx, donor_duration, np)
        if self_shape is not None:
            cell_shapes["SELF_GRID"][idx] = self_shape
            self_grid_rows += len(idx)
        dynamic = {
            "q00": _token_adjacent_metric(cell_shapes["q00"], idx, np),
            "q10": _token_adjacent_metric(cell_shapes["q10"], idx, np),
            "q01": _token_adjacent_metric(cell_shapes["q01"], idx, np),
            "q11": _token_adjacent_metric(cell_shapes["q11"], idx, np),
        }
        dynamic["SELF_GRID"] = _token_adjacent_metric(cell_shapes["SELF_GRID"], idx, np)
        row.update({
            "status": "ok",
            "mu_recipient_array_key": component_keys["mu_recipient_array_key"],
            "mu_donor_array_key": component_keys["mu_donor_array_key"],
            "frame_indices_array_key": component_keys["frame_indices_array_key"],
            "mean_log_shape_gap_mean_absolute_per_bin": float(np.mean(np.abs(mu_d - mu_r))),
            "recipient_log_residual_rms": float(np.sqrt(np.mean(rho_r * rho_r))),
            "donor_log_residual_rms": float(np.sqrt(np.mean(rho_d * rho_d))),
            "mean_residual_reconstruction_max_abs": max(float(np.max(np.abs(l_recon_r - own_log[idx]))), float(np.max(np.abs(l_recon_d - donor_log[idx])))),
            "q00_softmax_ownshape": q00_match,
            "q11_softmax_donormapped": q11_mapped_match,
            "q11_softmax_saved_SP_endpoint": q11_endpoint_match,
            "epsilon_floor_mass": {"recipient": own_floor, "mapped_donor": donor_floor},
            "self_grid": self_meta,
            "cell_logshape_dynamics": dynamic,
        })
        token_rows.append(row)

    # Every arm uses recipient shape outside the exact common eligible domain.
    # Ambiguous mapped frames are not inherited from the old SP endpoint; the
    # whole-grid endpoint gate below must account for them.
    outside_eligible_mask = ~eligible_frame_mask
    for name in CELLS:
        cell_shapes[name][outside_eligible_mask] = own_shape[outside_eligible_mask]
        if not np.array_equal(cell_shapes[name][outside_eligible_mask], own_shape[outside_eligible_mask]):
            raise ValueError(f"{sample_id}/{direction_name}/{name} changed recipient shape outside eligible token frames")
    ambiguous_mapped_mask = ambiguous_mask & mapped_frames
    ambiguous_exact_own = bool(np.array_equal(sp_endpoint_shape[ambiguous_mapped_mask], own_shape[ambiguous_mapped_mask]))
    if np.any(ambiguous_mapped_mask) and not ambiguous_exact_own:
        raise ValueError(f"{sample_id}/{direction_name} original SP endpoint differs from own shape on mapped ambiguous frames")
    q00_full_endpoint_audits.append(_softmax_endpoint_audit(
        cell_shapes["q00"], own_shape, np, f"{sample_id}/{direction_name}/q00_full_grid_RT"
    ))
    q11_full_endpoint_audits.append(_softmax_endpoint_audit(
        cell_shapes["q11"], sp_endpoint_shape, np, f"{sample_id}/{direction_name}/q11_full_grid_SP"
    ))

    # Verify all shape rows, SP power, endpoints and unchanged source factors.
    cell_sp: dict[str, Any] = {}
    power_audits: dict[str, Any] = {}
    for cell in CELLS:
        shape = np.asarray(cell_shapes[cell], dtype=np.float64)
        if shape.shape != own_shape.shape or not np.isfinite(shape).all() or np.any(shape < 0.0):
            raise ValueError(f"{sample_id}/{direction_name}/{cell} shape contains invalid values")
        row_sums = np.sum(shape, axis=1, dtype=np.float64)
        if float(np.max(np.abs(row_sums - 1.0), initial=0.0)) > PARAMETER_ABS_TOL:
            raise ValueError(f"{sample_id}/{direction_name}/{cell} normalized shape row sums failed")
        sp = shape * rec_power[:, None]
        power_error = np.abs(np.sum(sp, axis=1, dtype=np.float64) - rec_power)
        relative_error = power_error / np.maximum(np.abs(rec_power), 1e-30)
        max_abs = float(np.max(power_error, initial=0.0))
        max_rel = float(np.max(relative_error, initial=0.0))
        if max_rel > SP_POWER_REL_TOL:
            raise ValueError(f"{sample_id}/{direction_name}/{cell} recipient SP row power was not preserved")
        cell_sp[cell] = sp
        power_audits[cell] = {"max_abs_power_error": max_abs, "max_relative_power_error": max_rel, "relative_tolerance": SP_POWER_REL_TOL, "frame_count": int(len(sp))}

    # Store exact source endpoint SP arrays only after the softmax formula
    # checks pass; the floating-point formula itself is not called bit-exact.
    cell_sp["q00"] = np.asarray(rt_params["sp"]).copy()
    cell_sp["q11"] = np.asarray(sp_params["sp"]).copy()

    parameters = {
        "f0": np.asarray(rt_params["f0"]).copy(),
        "time": np.asarray(rt_params["time"]).copy(),
        "ap": np.asarray(rt_params["ap"]).copy(),
        "frequency_hz": np.asarray(rt_params["frequency_hz"]).copy(),
        "fft_size": np.asarray(int(npz_meta["RT"]["meta"]["fft_size"]), dtype=np.int32),
    }
    unchanged_parameter_checks = {
        name: _require_equal(parameters[name], rt_params[name], np, f"{sample_id}/{direction_name}/{name}_recipient_copy")
        for name in ("f0", "ap", "time", "frequency_hz")
    }
    for cell in CELLS:
        parameters[f"sp_{cell}"] = cell_sp[cell]
    endpoint_array_copy_checks = {
        "q00_to_RT_SP": _require_equal(parameters["sp_q00"], rt_params["sp"], np, f"{sample_id}/{direction_name}/q00_exact_RT_SP_copy"),
        "q11_to_SP_SP": _require_equal(parameters["sp_q11"], sp_params["sp"], np, f"{sample_id}/{direction_name}/q11_exact_SP_SP_copy"),
    }

    relative_dir = Path("artifacts") / "evaluation" / sample_id / direction_name
    component_path = output_root / relative_dir / "phone_components.npz"
    parameter_path = output_root / relative_dir / "factorized_parameters.npz"
    waveform_dir = output_root / relative_dir / "audio"
    component_path.parent.mkdir(parents=True, exist_ok=True)
    parameter_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(component_path, **means_arrays)
    np.savez_compressed(parameter_path, **parameters)
    components_sha = sha256_file(component_path)
    params_sha = sha256_file(parameter_path)
    component_array_refs = {key: _array_entry(component_path, components_sha, key, value, np) for key, value in means_arrays.items()}
    parameter_array_refs = {key: _array_entry(parameter_path, params_sha, key, value, np) for key, value in parameters.items()}

    recipient_samples = len(recipient["audio"])
    cells: dict[str, Any] = {}
    for cell in CELLS:
        params = {
            "f0": parameters["f0"],
            "time": parameters["time"],
            "sp": parameters[f"sp_{cell}"],
            "ap": parameters["ap"],
        }
        wave, synth = world.synthesize_fixed(_load_pyworld(), params, recipient_samples, np)
        if wave is None or synth.get("status") != "ok":
            raise ValueError(f"{sample_id}/{direction_name}/{cell} WORLD synthesis failed: {synth}")
        if len(wave) != recipient_samples:
            raise ValueError(f"{sample_id}/{direction_name}/{cell} waveform length differs from recipient")
        measured_rms = float(np.sqrt(np.mean(np.asarray(wave, dtype=np.float64) ** 2)))
        if abs(measured_rms - FIXED_RMS) > FIXED_RMS_ABS_TOL:
            raise ValueError(f"{sample_id}/{direction_name}/{cell} fixed RMS error {measured_rms - FIXED_RMS}")
        wav_path = waveform_dir / f"{cell}.wav"
        wav_meta = world.write_float_wav(wav_path, wave, sf, np)
        cell_row: dict[str, Any] = {
            "status": "ok",
            "parameter_npz": {"path": str(parameter_path), "file_sha256": params_sha, "sp_key": f"sp_{cell}", "sp_array": parameter_array_refs[f"sp_{cell}"]},
            "waveform": wav_meta,
            "synthesis": synth,
            "power_preservation": power_audits[cell],
        }
        if cell in ("q00", "q11"):
            old_arm = "RT" if cell == "q00" else "SP"
            old_wave = inputs["old_waves"][old_arm]["audio"]
            diff = np.asarray(wave, dtype=np.float64) - np.asarray(old_wave, dtype=np.float64)
            bridge_max = float(np.max(np.abs(diff), initial=0.0))
            bridge_rms = float(np.sqrt(np.mean(diff * diff)))
            bridge_bit_identical = bool(np.array_equal(np.asarray(wave, dtype=np.float32), np.asarray(old_wave, dtype=np.float32)))
            if bridge_max > WAVE_BRIDGE_MAX_ABS_TOL:
                raise ValueError(f"{sample_id}/{direction_name}/{cell} waveform bridge to original {old_arm} exceeds tolerance")
            repeated, repeat_meta = world.synthesize_fixed(_load_pyworld(), params, recipient_samples, np)
            if repeated is None or not np.array_equal(np.asarray(repeated, dtype=np.float32), np.asarray(wave, dtype=np.float32)):
                raise ValueError(f"{sample_id}/{direction_name}/{cell} repeated synthesis is not bit-identical")
            cell_row["endpoint_bridge"] = {
                "existing_arm": old_arm,
                "existing_wave_path": str(inputs["old_waves"][old_arm]["path"]),
                "existing_wave_file_sha256": inputs["old_waves"][old_arm]["meta"]["file_sha256"],
                "bit_identical": bridge_bit_identical,
                "max_abs_error": bridge_max,
                "rms_error": bridge_rms,
                "max_abs_tolerance": WAVE_BRIDGE_MAX_ABS_TOL,
                "repeat_bit_identical": True,
                "repeat_pcm_sha256": _float32_pcm_sha(repeated, np),
                "repeat_synthesis": repeat_meta,
            }
        cells[cell] = cell_row

    aggregate_floor = {
        "recipient_all_world_frames": recipient_floor_all,
        "mapped_donor_all_common_mapped_frames": mapped_donor_floor_all,
        "recipient_observed_max_added_mass": max((item["recipient"]["observed_max_added_mass"] for item in epsilon_floor_audits), default=0.0),
        "mapped_donor_observed_max_added_mass": max((item["mapped_donor"]["observed_max_added_mass"] for item in epsilon_floor_audits), default=0.0),
        "theoretical_max_added_mass_per_row": int(rec_feature["sp"].shape[1]) * SHAPE_EPSILON,
        "recipient_rows_with_floored_bins": sum(item["recipient"]["rows_with_floored_bins"] for item in epsilon_floor_audits),
        "mapped_donor_rows_with_floored_bins": sum(item["mapped_donor"]["rows_with_floored_bins"] for item in epsilon_floor_audits),
    }

    token_counts = Counter(str(row["phone_label"]) for row in token_rows)
    pooled_token_metrics = {}
    for label in sorted(token_counts):
        rows = [row for row in token_rows if row["phone_label"] == label and row.get("status") == "ok"]
        pooled_token_metrics[label] = {
            "matched_token_count": int(token_counts[label]),
            "eligible_token_count": len(rows),
            "frozen16_inference_member": label in pair["phone_labels"],
            "direction": direction_name,
            "cells": {
                cell: {
                    "token_unweighted_mean_adjacent_logshape_delta": _mean([
                        row["cell_logshape_dynamics"][cell]["mean_abs_adjacent_delta_log_shape"]
                        for row in rows if row["cell_logshape_dynamics"][cell]["mean_abs_adjacent_delta_log_shape"] is not None
                    ], np),
                    "token_count_with_adjacent_pairs": sum(
                        row["cell_logshape_dynamics"][cell]["consecutive_pair_count"] > 0 for row in rows
                    ),
                }
                for cell in CELLS
            },
        }

    return {
        "id": sample_id,
        "speaker": pair["speaker"],
        "ordered_index": pair["ordered_index"],
        "source_identity_sha256": pair["source_identity_sha256"],
        "mfa_pair_qualified_original": pair["mfa_pair_qualified"],
        "physical_alignment_pair_qualified": pair["physical_alignment_pair_qualified"],
        "physical_alignment_overlay_used": pair["physical_alignment_overlay_used"],
        "physical_overlay_record_sha256": pair["physical_overlay_record_sha256"],
        "direction": direction_name,
        "recipient_condition": recipient_condition,
        "donor_condition": donor_condition,
        "scoring_audio": {
            "A_RAW": recipient["raw_audio_meta"],
            "A_RT": inputs["old_waves"]["RT"]["meta"],
            "A_RT_arm": "RT",
            "fixed_audio_role": "recipient fixed audio used only to compute WORLD parameters",
        },
        "condition_audio_refs": {
            condition: {
                "source_raw": pair["sides"][condition]["raw_audio_meta"],
                "fixed_world_input": pair["sides"][condition]["audio_meta"],
                "world_features": pair["sides"][condition]["info"]["world_features"],
            }
            for condition in CONDITIONS
        },
        "recipient_fixed_audio_sha256": recipient["audio_meta"]["file_sha256"],
        "donor_fixed_audio_sha256": donor["audio_meta"]["file_sha256"],
        "recipient_world_feature_sha256": recipient["info"]["world_features"]["file_sha256"],
        "donor_world_feature_sha256": donor["info"]["world_features"]["file_sha256"],
        "phone_map": {"path": str(inputs["phone_map_path"]), "file_sha256": sha256_file(inputs["phone_map_path"]), "matched_phone_count": len(tokens)},
        "transfer_audit": {"path": str(inputs["transfer_audit_path"]), "file_sha256": sha256_file(inputs["transfer_audit_path"]), "arrays": inputs["audit_checks"]},
        "original_parameter_endpoints": {
            arm: {"path": str(npz_meta[arm]["path"]), "file_sha256": npz_meta[arm]["meta"]["file_sha256"], "checks": npz_meta[arm]["checks"]}
            for arm in ("RT", "SP")
        },
        "sample_count": recipient_samples,
        "world_frame_count": int(len(times)),
        "mapped_frame_count": int(np.count_nonzero(mapped_frames)),
        "ambiguous_mapped_frame_count": int(np.count_nonzero(ambiguous_mask & mapped_frames)),
        "ambiguous_mapped_frames_exactly_own_in_original_SP_endpoint": ambiguous_exact_own,
        "outside_eligible_frames_exactly_own_in_all_cells": True,
        "eligible_factorization_frame_count": int(sum(token["eligible_frame_count"] for token in tokens)),
        "f0_ap_time_identity": "recipient arrays copied unchanged for all five cells",
        "components_npz": {"path": str(component_path), "file_sha256": components_sha, "arrays": component_array_refs},
        "parameters_npz": {"path": str(parameter_path), "file_sha256": params_sha, "arrays": parameter_array_refs},
        "cells": cells,
        "token_rows": token_rows,
        "token_label_summary": pooled_token_metrics,
        "softmax_formula_endpoint_audits": {
            "q00": {
                "comparison_count": len(q00_formula_audits),
                "max_abs_error": max((item["max_abs_error"] for item in q00_formula_audits), default=0.0),
                "max_relative_error": max((item["max_relative_error"] for item in q00_formula_audits), default=0.0),
                "passed": True,
            },
            "q11_mapped_shape": {
                "comparison_count": len(q11_mapped_formula_audits),
                "max_abs_error": max((item["max_abs_error"] for item in q11_mapped_formula_audits), default=0.0),
                "max_relative_error": max((item["max_relative_error"] for item in q11_mapped_formula_audits), default=0.0),
                "passed": True,
            },
            "q11_saved_SP_endpoint": {
                "comparison_count": len(q11_endpoint_formula_audits),
                "max_abs_error": max((item["max_abs_error"] for item in q11_endpoint_formula_audits), default=0.0),
                "max_relative_error": max((item["max_relative_error"] for item in q11_endpoint_formula_audits), default=0.0),
                "passed": True,
            },
            "q00_full_grid_RT_endpoint": q00_full_endpoint_audits[0],
            "q11_full_grid_SP_endpoint": q11_full_endpoint_audits[0],
            "absolute_tolerance": SOFTMAX_ABS_TOL,
            "relative_tolerance": SOFTMAX_REL_TOL,
            "relative_denominator_floor": SOFTMAX_RELATIVE_DENOMINATOR_FLOOR,
            "formula_bit_exact_claim": False,
        },
        "epsilon_floor_mass_audit": {
            "formula": "sum(max(epsilon - normalized_shape, 0)) per row",
            "epsilon": SHAPE_EPSILON,
            **aggregate_floor,
            "token_rows": epsilon_floor_audits,
        },
        "unchanged_recipient_factor_checks": unchanged_parameter_checks,
        "stored_endpoint_sp_array_copy_checks": endpoint_array_copy_checks,
        "self_grid_eligible_frame_count": self_grid_rows,
        "sp_power_preservation": power_audits,
        "status": "PASS",
    }


_PYWORLD: Any | None = None


def _load_pyworld() -> Any:
    global _PYWORLD
    if _PYWORLD is None:
        import pyworld
        _PYWORLD = pyworld
    return _PYWORLD


def _runtime_identity(manifest: dict[str, Any], np: Any, sf: Any) -> dict[str, Any]:
    frozen = manifest.get("world_runtime", {})
    module = _load_pyworld()
    pyworld_version = importlib.metadata.version("pyworld")
    numpy_version = importlib.metadata.version("numpy")
    python_version = ".".join(map(str, sys.version_info[:3]))
    package_path = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
    package_sha = sha256_file(package_path)
    expected_package_path = Path(str(frozen.get("pyworld_package_path", ""))).resolve(strict=True)
    if pyworld_version != "0.3.5" or pyworld_version != frozen.get("pyworld_version"):
        raise RuntimeError(f"pyworld version mismatch: actual={pyworld_version}, frozen={frozen.get('pyworld_version')}")
    if numpy_version != frozen.get("numpy_version") or python_version != frozen.get("python_version"):
        raise RuntimeError("Python/NumPy runtime differs from the executed frozen WORLD environment")
    if package_path != expected_package_path or package_sha != frozen.get("pyworld_package_sha256"):
        raise RuntimeError("pyworld package path/hash differs from the executed frozen WORLD environment")
    extension_module = getattr(module, "pyworld", None)
    if extension_module is None or not getattr(extension_module, "__file__", None):
        raise RuntimeError("pyworld extension module path is unavailable")
    extension_path = Path(str(extension_module.__file__)).resolve(strict=True)
    extension_sha = sha256_file(extension_path)
    if extension_path != Path(str(frozen["pyworld_extension_path"])).resolve(strict=True) or extension_sha != frozen.get("pyworld_extension_sha256"):
        raise RuntimeError("pyworld extension path/hash differs from the executed frozen WORLD environment")
    if frozen.get("sample_rate_hz") != SAMPLE_RATE or frozen.get("frame_period_ms") != FRAME_PERIOD_MS:
        raise RuntimeError("frozen WORLD runtime sample rate/frame period differs from the analysis")
    return {
        "python_version": python_version,
        "numpy_version": numpy_version,
        "soundfile_version": importlib.metadata.version("soundfile"),
        "pyworld_version": pyworld_version,
        "pyworld_package_path": str(package_path),
        "pyworld_package_sha256": package_sha,
        "pyworld_extension_path": str(extension_path),
        "pyworld_extension_sha256": extension_sha,
        "sample_rate_hz": SAMPLE_RATE,
        "frame_period_ms": FRAME_PERIOD_MS,
    }


def _output_file_inventory(output_root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(p for p in output_root.rglob("*") if p.is_file() and p.name not in {"output_file_inventory.json", "evaluation_manifest.json"}):
        rows.append({"path": str(path.relative_to(output_root).as_posix()), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {"schema_version": 1, "manifest_type": "world_phone_shape_factorial_eval_output_file_inventory", "files": rows}
    payload["manifest_sha256"] = fingerprint(payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    controls, pairs, input_ledger = _read_inputs(args, np, sf)
    runtime_identity = _runtime_identity(controls["manifest"], np, sf)
    plan = controls["analysis_plan"]
    plan_path = controls["resolved"]["analysis_plan"]
    source_path = Path(__file__).resolve()
    if args.preflight_only:
        return {
            "status": "PREFLIGHT_PASS",
            "scope": "evaluation_physical_waveform_production_only",
            "split": "evaluation",
            "record_count": len(pairs),
            "speaker_count": len({pair["speaker"] for pair in pairs}),
            "direction_count": len(pairs) * len(DIRECTIONS),
            "expected_cell_count": len(pairs) * len(DIRECTIONS) * len(CELLS),
            "input_file_count_verified": len(input_ledger),
            "ordered_ids": [pair["id"] for pair in pairs],
            "directions": list(DIRECTIONS),
            "cells": list(CELLS),
            "evaluation_waveforms_read": True,
            "evaluation_outcomes_read": False,
            "evaluation_effects_read": False,
            "effects_read": False,
            "model_forward_performed": False,
            "physical_manifest_raw_sha256": PHYSICAL_MANIFEST_RAW_SHA256,
            "physical_manifest_payload_sha256": PHYSICAL_MANIFEST_PAYLOAD_SHA256,
            "physical_plan_raw_sha256": PHYSICAL_PLAN_RAW_SHA256,
            "physical_plan_payload_sha256": PHYSICAL_PLAN_PAYLOAD_SHA256,
            "physical_qa_raw_sha256": PHYSICAL_QA_RAW_SHA256,
            "analysis_plan_raw_sha256": sha256_file(plan_path),
            "analysis_plan_payload_sha256": plan["manifest_sha256"],
            "analysis_source_sha256": sha256_file(source_path),
            "independent_qa_source_sha256": sha256_file(controls["resolved"]["independent_qa_source"]),
            "evaluation_freeze_sha256": sha256_file(controls["resolved"]["evaluation_freeze"]),
            "input_inventory": sorted(input_ledger.values(), key=lambda item: (item["role"], item["path"])),
            "control_file_inventory": sorted(controls["control_file_inventory"].values(), key=lambda item: item["path"]),
            "runtime": runtime_identity,
        }

    output_root = args.out.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse an existing output directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    world = _import_frozen_world_source(controls["resolved"]["world_source"])
    phone_labels = set(controls["phone_inventory"]["selected_phone_labels"])
    for pair in pairs:
        pair["phone_labels"] = phone_labels

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    intended = len(pairs) * len(DIRECTIONS)
    for pair in pairs:
        for direction_name in DIRECTIONS:
            try:
                row = _factorize_direction(pair, direction_name, pair["directions"][direction_name], world, np, sf, output_root)
            except Exception as exc:
                row = {
                    "id": pair["id"], "speaker": pair["speaker"], "ordered_index": pair["ordered_index"],
                    "source_identity_sha256": pair["source_identity_sha256"], "direction": direction_name,
                    "recipient_condition": DIRECTIONS[direction_name]["recipient"],
                    "donor_condition": DIRECTIONS[direction_name]["donor"],
                    "mfa_pair_qualified_original": pair["mfa_pair_qualified"],
                    "physical_alignment_pair_qualified": pair["physical_alignment_pair_qualified"],
                    "physical_alignment_overlay_used": pair["physical_alignment_overlay_used"],
                    "physical_overlay_record_sha256": pair["physical_overlay_record_sha256"],
                    "scoring_audio": None,
                    "status": "failed", "failure": f"{type(exc).__name__}:{exc}",
                }
                failures.append({"id": pair["id"], "direction": direction_name, "failure": row["failure"]})
            rows.append(row)
            print(f"EVAL shape-factor {len(rows)}/{intended} {pair['id']} {direction_name} {row['status']}", flush=True)

    complete = sum(row.get("status") == "PASS" for row in rows)
    complete_cells = sum(sum(cell.get("status") == "ok" for cell in row.get("cells", {}).values()) for row in rows)
    output_inventory = _output_file_inventory(output_root)
    output_rows = output_inventory.get("files", [])
    output_wav_count = sum(str(item.get("path", "")).lower().endswith(".wav") for item in output_rows)
    output_npz_count = sum(str(item.get("path", "")).lower().endswith(".npz") for item in output_rows)
    unique_rows = {(str(row.get("id")), str(row.get("direction"))) for row in rows}
    row_identity_complete = len(rows) == intended and len(unique_rows) == intended
    output_inventory_complete = len(output_rows) == 1120 and output_wav_count == 800 and output_npz_count == 320
    inventory_path = output_root / "output_file_inventory.json"
    write_json(inventory_path, output_inventory)
    plan_payload = plan["manifest_sha256"]
    eval_freeze = controls["documents"]["evaluation_freeze"]
    cal_freeze = controls["calibration_freeze"]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "manifest_type": "world_phone_shape_mean_dynamic_factorial_evaluation",
        "status": "EVAL_FACTORIAL_WAVEFORMS_READY_FOR_INDEPENDENT_REVIEW" if complete == intended and complete_cells == intended * len(CELLS) and not failures and row_identity_complete and output_inventory_complete else "INCOMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "evaluation_physical_waveform_production_only",
        "split": "evaluation",
        "evaluation_waveforms_read": True,
        "evaluation_outcomes_read": False,
        "evaluation_effects_read": False,
        "effects_read": False,
        "model_forward_performed": False,
        "analysis_spec_path": str(args.eval_design.resolve()),
        "analysis_spec_raw_sha256": EVAL_DESIGN_SHA256,
        "metric_addendum_path": str(args.metric_addendum.resolve()),
        "metric_addendum_raw_sha256": METRIC_ADDENDUM_SHA256,
        "analysis_plan_path": str(plan_path),
        "analysis_plan_raw_sha256": sha256_file(plan_path),
        "analysis_plan_payload_sha256": plan_payload,
        "analysis_source_path": str(source_path),
        "analysis_source_sha256": sha256_file(source_path),
        "independent_qa_source_path": str(controls["resolved"]["independent_qa_source"]),
        "independent_qa_source_sha256": sha256_file(controls["resolved"]["independent_qa_source"]),
        "evaluation_root_freeze_path": str(controls["resolved"]["evaluation_freeze"]),
        "evaluation_root_freeze_sha256": sha256_file(controls["resolved"]["evaluation_freeze"]),
        "calibration_root_freeze_path": str(controls["resolved"]["calibration_freeze"]),
        "calibration_root_freeze_sha256": sha256_file(controls["resolved"]["calibration_freeze"]),
        "input_bindings": controls["control_file_inventory"],
        "input_file_count_verified": len(input_ledger),
        "input_inventory": sorted(input_ledger.values(), key=lambda item: (item["role"], item["path"])),
        "output_file_inventory_path": str(inventory_path),
        "output_file_inventory_sha256": output_inventory["manifest_sha256"],
        "runtime": {
            "python": sys.version,
            **runtime_identity,
            "fixed_world_source_sha256": WORLD_SOURCE_SHA256,
            "fixed_rms": FIXED_RMS,
            "sample_rate_hz": SAMPLE_RATE,
            "frame_period_ms": FRAME_PERIOD_MS,
        },
        "factorization": {
            "epsilon": SHAPE_EPSILON,
            "coordinate": "log normalized SP shape; per-frequency-bin exact-token mean and centered residual; row softmax",
            "cells": list(CELLS),
            "directions": list(DIRECTIONS),
            "q00_q11_softmax_formula_tolerance": {
                "elementwise_rule": "abs(actual-endpoint) <= 1e-12 + 1e-10*abs(endpoint)",
                "relative_denominator_floor": SOFTMAX_RELATIVE_DENOMINATOR_FLOOR,
                "formula_bit_exact_claim": False,
            },
            "epsilon_floor_mass": "sum(max(epsilon-normalized_shape,0)) per row; report observed mass and frequency_bin_count*epsilon bound",
            "q00": "recipient phone mean + recipient within-phone residual; formula bridge to recipient normalized SP then exact RT endpoint array copy",
            "q10": "donor-mapped phone mean + recipient within-phone residual",
            "q01": "recipient phone mean + donor-mapped within-phone residual",
            "q11": "donor-mapped phone mean + donor-mapped within-phone residual; formula bridge to saved SP endpoint then exact endpoint array copy",
            "self_grid": "recipient-only log shape interpolated to donor 5ms centers and back; separate two-pass grid-cycle control, not a pure interpolation proof",
            "token_mean_domain": "same unique mapped recipient 5ms frames for each exact phone token; no pooling across repeated occurrences",
            "ambiguous_frames": "excluded from all token means and recipient-owned in all five cells; original q11 SP endpoint must match recipient shape exactly on mapped ambiguous frames",
            "unmapped_frames": "recipient shape in every cell",
            "other_parameters": "recipient F0, AP, time centers, frequency grid, sample clock, and global RMS stay fixed",
            "row_power": "multiply each normalized shape by recipient SP row power",
            "rms_policy": "frozen fixed global RMS after synthesis and exact right trim",
            "inference_labels": controls["phone_inventory"]["selected_phone_labels"],
            "all_matched_labels_retained": True,
        },
        "intended_pairs": 80,
        "intended_speakers": 40,
        "ordered_evaluation_ids": list(plan["ordered_evaluation_ids"]),
        "input_identity_sha256": EXPECTED_INPUT_IDENTITY_SHA256,
        "intended_direction_rows": intended,
        "complete_direction_rows": complete,
        "intended_cell_waveforms": intended * len(CELLS),
        "complete_cell_waveforms": complete_cells,
        "direction_row_identity_complete": row_identity_complete,
        "output_file_count": len(output_rows),
        "output_wav_count": output_wav_count,
        "output_npz_count": output_npz_count,
        "output_inventory_complete": output_inventory_complete,
        "failure_count": len(failures),
        "failures": failures,
        "records": rows,
    }
    payload["manifest_sha256"] = fingerprint(payload)
    out_path = output_root / "evaluation_manifest.json"
    write_json(out_path, payload)
    return {
        "status": payload["status"], "direction_rows": len(rows), "complete_direction_rows": complete,
        "cell_waveforms": complete_cells, "failures": len(failures),
        "manifest_sha256": payload["manifest_sha256"], "manifest_path": str(out_path),
        "output_inventory_sha256": output_inventory["manifest_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-manifest", required=True, type=Path)
    parser.add_argument("--physical-plan", required=True, type=Path)
    parser.add_argument("--physical-qa", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--tts-pair-manifest", required=True, type=Path)
    parser.add_argument("--world-source", required=True, type=Path)
    parser.add_argument("--execution-bindings", required=True, type=Path)
    parser.add_argument("--phone-table", required=True, type=Path)
    parser.add_argument("--phone-inventory", required=True, type=Path)
    parser.add_argument("--calibration-manifest", required=True, type=Path)
    parser.add_argument("--calibration-plan", required=True, type=Path)
    parser.add_argument("--calibration-spec", required=True, type=Path)
    parser.add_argument("--calibration-qa", required=True, type=Path)
    parser.add_argument("--calibration-freeze", required=True, type=Path)
    parser.add_argument("--eval-design", required=True, type=Path)
    parser.add_argument("--metric-addendum", required=True, type=Path)
    parser.add_argument("--independent-qa-source", required=True, type=Path)
    parser.add_argument("--evaluation-freeze", required=True, type=Path)
    parser.add_argument("--analysis-plan", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.preflight_only and args.out is None:
        parser.error("--out is required unless --preflight-only is used")
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
    if args.preflight_only:
        return 0 if result.get("status") == "PREFLIGHT_PASS" else 2
    return 0 if result.get("status") == "EVAL_FACTORIAL_WAVEFORMS_READY_FOR_INDEPENDENT_REVIEW" and result.get("complete_direction_rows") == 160 and result.get("cell_waveforms") == 800 and result.get("failures") == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
