#!/usr/bin/env python3
"""CAL-only WORLD factorial separating phone mean shape from local dynamics.

The input is the already frozen 20-pair CAL WORLD run. The producer refuses
non-CAL manifests and has no EVAL path argument. It does not import a scorer,
video model, or endpoint. It writes only CAL parameter arrays, float WAVs, and
engineering QA needed to review a later, separately frozen EVAL protocol.
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

WORLD_MANIFEST_RAW_SHA256 = "297d3c138fc2de75e86f32db43f7d42f94476e3275c9ef9b72235c1880af07cc"
WORLD_MANIFEST_PAYLOAD_SHA256 = "75b47f39a7a462753877db6f03e41758335011764b279061d546c3e812cec29c"
WORLD_PLAN_RAW_SHA256 = "2c5d26b09c0bd3fea4812c16612387ec5c96308082c9da1e2a823ee1f6c9d8df"
WORLD_PLAN_PAYLOAD_SHA256 = "e444a0fc6b322b8e73589ce23969a0142642df76527703cab69f6f8cf9f71760"
WORLD_SOURCE_SHA256 = "7a6661afced892037d5f32fa3760e8d0906b77f40bb06f5f3ce53a66a04729b2"
EXECUTION_BINDINGS_SHA256 = "cde66fa11f140866fc87f07b7a39ca68ddefc403ff7839b239987549e9f7e47d"
PHONE_TABLE_RAW_SHA256 = "1673092554bfc03dbc385136c3a66c08821499c0ec6c25602d385d1d0d7cf16f"
PHONE_TABLE_PAYLOAD_SHA256 = "586d86fedd159ba1c654586bf6892d87b36b1ed8d88c666217d648d230fb2259"
CAL_QC_RAW_SHA256 = "a7a93f3bf591b85ff337f4fb295a058563697885548d3f2b0ffc4270c5fe6016"
CAL_QC_PAYLOAD_SHA256 = "ee73b11fa317e4f486933825112d01793605eae72dd6294ef9e08d085a04fe35"
CAL_AP_QC_RAW_SHA256 = "5ff39cfd82817f56eb9cc348e2d8481432973a09db7fd9a41ed9a280d91301c5"
CAL_AP_QC_PAYLOAD_SHA256 = "262cbb739a198aefd14490b392a0fc4a3b97fd0d4f0f9b6ad15d1dc8eb25cc07"
CAL_FREEZE_RAW_SHA256 = "0a5b694975efeb07b9d1e9c0beb10f0142bb4203ba4a026d7ff965477991ab9d"
PHONE_INVENTORY_RAW_SHA256 = "62eeb63462c6e2d378482d1cdc7876d2e05666bac0f54c32d2a08f5e114db2c2"
PHONE_INVENTORY_PAYLOAD_SHA256 = "54a51ff1a9af891271b0eaed797dc0bb7508a338ef7e38f922b8f9292fa40f5d"
ANALYSIS_SPEC_SHA256 = "01456c701f63863342820d5a3a87f98e7e3c93fa534aee79d977a3ca26f096d9"
ANALYSIS_PLAN_RAW_SHA256 = "594592fe09a763634b3f757220096b0f08c038923aec418816442fc99b0879e7"
ANALYSIS_PLAN_PAYLOAD_SHA256 = "85b94f1f7ccdb2a7c7dbceadf72d4e0f298019c64122514df9249a2d9442c65a"

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


def _read_control_files(args: argparse.Namespace, np: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_paths = {
        "world_manifest": (args.world_manifest, WORLD_MANIFEST_RAW_SHA256),
        "world_plan": (args.world_plan, WORLD_PLAN_RAW_SHA256),
        "world_source": (args.world_source, WORLD_SOURCE_SHA256),
        "execution_bindings": (args.execution_bindings, EXECUTION_BINDINGS_SHA256),
        "phone_table": (args.phone_table, PHONE_TABLE_RAW_SHA256),
        "calibration_qc": (args.calibration_qc, CAL_QC_RAW_SHA256),
        "calibration_ap_qc": (args.calibration_ap_qc, CAL_AP_QC_RAW_SHA256),
        "calibration_freeze": (args.calibration_freeze, CAL_FREEZE_RAW_SHA256),
        "phone_inventory": (args.phone_inventory, PHONE_INVENTORY_RAW_SHA256),
        "analysis_spec": (args.spec, ANALYSIS_SPEC_SHA256),
        "analysis_plan": (args.analysis_plan, ANALYSIS_PLAN_RAW_SHA256),
    }
    file_entries: dict[str, dict[str, Any]] = {}
    resolved: dict[str, Path] = {}
    for role, (path_arg, expected) in raw_paths.items():
        path, entry = sha256_required(path_arg, expected, role)
        file_entries[role] = entry
        resolved[role] = path
    manifest = load_json(resolved["world_manifest"])
    # The frozen WORLD plan file contains both CAL and EVAL condition rows.
    # Its exact raw digest and expected payload digest are bound here without
    # deserializing or enumerating those rows; only the CAL producer manifest
    # is opened as an input dataset.
    phone_table = load_json(resolved["phone_table"])
    cal_qc = load_json(resolved["calibration_qc"])
    ap_qc = load_json(resolved["calibration_ap_qc"])
    freeze = load_json(resolved["calibration_freeze"])
    phone_inventory = load_json(resolved["phone_inventory"])
    analysis_plan = load_json(resolved["analysis_plan"])

    if manifest.get("manifest_sha256") != WORLD_MANIFEST_PAYLOAD_SHA256 or fingerprint(manifest) != WORLD_MANIFEST_PAYLOAD_SHA256:
        raise ValueError("frozen WORLD CAL manifest raw/payload identity failed")
    if manifest.get("split") != "calibration" or manifest.get("manifest_type") != "world_phone_transfer_calibration_producer":
        raise ValueError("input WORLD manifest is not the frozen calibration manifest")
    if manifest.get("evaluation_outcomes_read") is not False or manifest.get("evaluation_waveforms_read") is not False:
        raise ValueError("source WORLD CAL manifest has an evaluation-read marker")
    if manifest.get("world_transfer_plan_sha256") != WORLD_PLAN_PAYLOAD_SHA256:
        raise ValueError("WORLD CAL manifest does not bind the frozen WORLD plan")
    if str(Path(str(manifest.get("world_transfer_plan_path", ""))).resolve()) != str(resolved["world_plan"]):
        raise ValueError("WORLD CAL manifest plan path differs from supplied frozen plan")
    if manifest.get("phone_normalization_table_sha256") != PHONE_TABLE_PAYLOAD_SHA256:
        raise ValueError("WORLD CAL manifest does not bind the frozen phone table payload")
    if Path(str(manifest.get("phone_normalization_table_path", ""))).name != resolved["phone_table"].name:
        raise ValueError("WORLD CAL manifest phone-table path basename differs from supplied phone table")
    if manifest.get("phone_normalizer_source_sha256") != WORLD_SOURCE_SHA256:
        raise ValueError("WORLD CAL manifest phone normalizer source differs from the executed source")
    if manifest.get("fixed_rms") != FIXED_RMS:
        raise ValueError("WORLD CAL manifest fixed RMS differs from the frozen target")
    if manifest.get("status") != "CAL_WAVEFORMS_READY_FOR_INDEPENDENT_REVIEW":
        raise ValueError("WORLD CAL manifest is not the completed calibration producer output")
    counts = manifest.get("sample_counts", {})
    if manifest.get("failures") != [] or counts.get("intended") != 20 or counts.get("mfa_pair_qualified") != 20:
        raise ValueError("WORLD CAL manifest is not a complete 20-pair input")
    if phone_table.get("manifest_sha256") != PHONE_TABLE_PAYLOAD_SHA256 or fingerprint(phone_table) != PHONE_TABLE_PAYLOAD_SHA256:
        raise ValueError("phone table raw/payload identity failed")
    if phone_inventory.get("inventory_sha256") != PHONE_INVENTORY_PAYLOAD_SHA256 or fingerprint(phone_inventory, excluded="inventory_sha256") != PHONE_INVENTORY_PAYLOAD_SHA256:
        raise ValueError("frozen CAL phone inventory raw/payload identity failed")
    if phone_inventory.get("status") != "FROZEN_CAL_ONLY_PHONE_INVENTORY" or phone_inventory.get("selection_split") != "calibration":
        raise ValueError("phone inventory is not the frozen CAL-only inventory")
    if phone_inventory.get("evaluation_textgrids_or_endpoints_read") is not False:
        raise ValueError("phone inventory records an evaluation read")
    if phone_inventory.get("selected_phone_labels") != analysis_plan.get("inferential_phone_labels"):
        raise ValueError("analysis plan's fixed phone labels differ from the frozen 16-label inventory")
    if cal_qc.get("manifest_sha256") != CAL_QC_PAYLOAD_SHA256 or fingerprint(cal_qc) != CAL_QC_PAYLOAD_SHA256:
        raise ValueError("CAL waveform-QC raw/payload identity failed")
    if cal_qc.get("status") != "READY_FOR_INDEPENDENT_REVIEW" or cal_qc.get("evaluation_waveforms_read") is not False or cal_qc.get("evaluation_effects_read") is not False:
        raise ValueError("CAL waveform-QC status/access markers differ from frozen QC")
    if cal_qc.get("world_transfer_plan_sha256") != WORLD_PLAN_PAYLOAD_SHA256:
        raise ValueError("CAL waveform QC does not bind the frozen WORLD plan")
    if ap_qc.get("manifest_sha256") != CAL_AP_QC_PAYLOAD_SHA256:
        raise ValueError("prior CAL acoustic descriptor payload identity failed")
    if fingerprint(ap_qc) != CAL_AP_QC_PAYLOAD_SHA256 or ap_qc.get("split") != "calibration":
        raise ValueError("prior CAL acoustic descriptor content/split mismatch")
    if ap_qc.get("evaluation_waveforms_read") is not False or ap_qc.get("evaluation_outcomes_read") is not False:
        raise ValueError("prior CAL acoustic descriptor records an evaluation read")
    if freeze.get("status") != "FROZEN":
        raise ValueError("root CAL waveform freeze is not frozen")
    required_freeze = {
        "calibration_producer_manifest_sha256": WORLD_MANIFEST_PAYLOAD_SHA256,
        "calibration_producer_manifest_raw_sha256": WORLD_MANIFEST_RAW_SHA256,
        "world_transfer_plan_sha256": WORLD_PLAN_PAYLOAD_SHA256,
        "world_transfer_plan_raw_sha256": WORLD_PLAN_RAW_SHA256,
        "calibration_qc_sha256": CAL_QC_PAYLOAD_SHA256,
        "calibration_qc_raw_sha256": CAL_QC_RAW_SHA256,
        "calibration_executed_source_sha256": WORLD_SOURCE_SHA256,
        "calibration_readonly_ap_energy_sidecar_raw_sha256": CAL_AP_QC_RAW_SHA256,
    }
    for key, expected in required_freeze.items():
        if freeze.get(key) != expected:
            raise ValueError(f"root CAL freeze binding mismatch for {key}")
    evidence = freeze.get("evidence_review", {})
    if evidence.get("world_evaluation_waveforms_produced") is not False or evidence.get("world_evaluation_effects_read") is not False:
        raise ValueError("root freeze evidence does not certify CAL-only timing of the physical run")
    if freeze.get("qc_pre_qc_producer_payload_sha256") != cal_qc.get("producer_manifest_sha256"):
        raise ValueError("root CAL freeze and waveform-QC producer payload linkage differ")
    if analysis_plan.get("manifest_sha256") != ANALYSIS_PLAN_PAYLOAD_SHA256 or fingerprint(analysis_plan) != ANALYSIS_PLAN_PAYLOAD_SHA256:
        raise ValueError("new analysis plan raw/payload identity failed")
    if analysis_plan.get("spec_sha256") != ANALYSIS_SPEC_SHA256:
        raise ValueError("new analysis plan does not bind the protocol spec")
    expected_bindings = {
        "world_manifest_raw_sha256": WORLD_MANIFEST_RAW_SHA256,
        "world_manifest_payload_sha256": WORLD_MANIFEST_PAYLOAD_SHA256,
        "world_plan_raw_sha256": WORLD_PLAN_RAW_SHA256,
        "world_plan_payload_sha256": WORLD_PLAN_PAYLOAD_SHA256,
        "world_source_sha256": WORLD_SOURCE_SHA256,
        "execution_bindings_sha256": EXECUTION_BINDINGS_SHA256,
        "phone_table_raw_sha256": PHONE_TABLE_RAW_SHA256,
        "phone_table_payload_sha256": PHONE_TABLE_PAYLOAD_SHA256,
        "calibration_qc_raw_sha256": CAL_QC_RAW_SHA256,
        "calibration_qc_payload_sha256": CAL_QC_PAYLOAD_SHA256,
        "calibration_ap_qc_raw_sha256": CAL_AP_QC_RAW_SHA256,
        "calibration_ap_qc_payload_sha256": CAL_AP_QC_PAYLOAD_SHA256,
        "calibration_freeze_raw_sha256": CAL_FREEZE_RAW_SHA256,
        "phone_inventory_raw_sha256": PHONE_INVENTORY_RAW_SHA256,
        "phone_inventory_payload_sha256": PHONE_INVENTORY_PAYLOAD_SHA256,
    }
    if analysis_plan.get("input_bindings") != expected_bindings:
        raise ValueError("new analysis plan input bindings differ from the frozen CAL assets")
    if analysis_plan.get("scope") != "calibration_only" or analysis_plan.get("evaluation_waveforms_read") is not False or analysis_plan.get("evaluation_outcomes_read") is not False:
        raise ValueError("new analysis plan is not explicitly CAL-only")
    expected_gates = {
        "q00_q11_softmax_formula_abs_tolerance": SOFTMAX_ABS_TOL,
        "q00_q11_softmax_formula_relative_tolerance": SOFTMAX_REL_TOL,
        "q00_q11_softmax_formula_relative_denominator_floor": SOFTMAX_RELATIVE_DENOMINATOR_FLOOR,
        "sp_row_power_relative_error_max": SP_POWER_REL_TOL,
        "q00_q11_waveform_bridge_abs_max": WAVE_BRIDGE_MAX_ABS_TOL,
        "fixed_rms_absolute_error_max": FIXED_RMS_ABS_TOL,
    }
    gates = analysis_plan.get("gates", {})
    if any(gates.get(key) != value for key, value in expected_gates.items()):
        raise ValueError("new analysis plan numeric gates differ from the frozen source")
    if gates.get("q00_q11_endpoint_storage") != "exact copy of sealed RT/SP arrays only after full-grid softmax-formula bridge passes; formula itself is not bit-exact":
        raise ValueError("new analysis plan endpoint-storage semantics differ")
    if gates.get("q00_q11_outside_eligible_endpoint_bridge") != "compare full-grid endpoint shape including recipient-shape rows outside eligible tokens; mapped ambiguous frames must be exactly recipient shape or absent; fail rather than copy an endpoint when this bridge fails":
        raise ValueError("new analysis plan outside-eligible endpoint bridge semantics differ")
    if gates.get("epsilon_floor_mass_upper_bound") != "frequency_bin_count * 1e-12 per row" or gates.get("epsilon_floor_mass_observed") != "record per-row added mass and observed maximum for recipient and mapped-donor normalized SP":
        raise ValueError("new analysis plan epsilon-floor mass bound differs")
    if gates.get("q00_q11_repeat_synthesis") != "bit-identical" or gates.get("right_trim_only") is not True or gates.get("padding") is not False or gates.get("clip_or_time_warp") is not False or gates.get("required_complete_direction_rows") != 40:
        raise ValueError("new analysis plan synthesis/clock/complete-row gates differ")
    expected_cells = {
        "q00": "softmax(muR + rhoR) on eligible frames and recipient shape elsewhere; full-grid RT bridge then exact endpoint storage",
        "q10": "softmax(muD + rhoR)",
        "q01": "softmax(muR + rhoD)",
        "q11": "softmax(muD + rhoD) on eligible frames and recipient shape elsewhere; full-grid SP bridge then exact endpoint storage",
        "SELF_GRID": "recipient log normalized shape sampled on donor original 5 ms centers by normalized token phase and mapped back to recipient centers with linear interpolation and endpoint holding",
    }
    if analysis_plan.get("cells") != expected_cells or analysis_plan.get("all_matched_labels_retained") is not True:
        raise ValueError("new analysis plan cell formulas/all-label retention differ")
    expected_ambiguous_policy = "exclude from every token mean and keep at own-recipient shape in all five cells; mapped ambiguous frames must exactly match own shape in the original SP endpoint or the q11 full-grid bridge fails"
    if analysis_plan.get("phone_domain", {}).get("ambiguous_frames") != expected_ambiguous_policy:
        raise ValueError("new analysis plan ambiguous-frame policy differs")
    if analysis_plan.get("cells_per_direction") != list(CELLS) or analysis_plan.get("direction_rows") != 40 or analysis_plan.get("pair_count") != 20 or analysis_plan.get("speaker_count") != 10:
        raise ValueError("new analysis plan cohort/cell counts differ from the frozen design")
    if analysis_plan.get("target_rms") != FIXED_RMS or analysis_plan.get("sample_rate_hz") != SAMPLE_RATE or analysis_plan.get("world_frame_period_ms") != FRAME_PERIOD_MS:
        raise ValueError("new analysis plan fixed RMS or clock differs from source constants")
    if analysis_plan.get("model_forward_performed") is not False or analysis_plan.get("status") != "SEALED_FOR_CAL_ONLY_CPU_PREFLIGHT_AND_WAVEFORM_QA":
        raise ValueError("new analysis plan status/model-forward markers are invalid")
    constraints = analysis_plan.get("analysis_constraints", {})
    if constraints.get("no_evaluation_effect_or_endpoint_reads") is not True or constraints.get("calibration_RT_SP_reference_parameters_and_audio_are_read") is not True:
        raise ValueError("analysis plan does not distinguish CAL reference reads from prohibited EVAL reads")
    if analysis_plan.get("plan_read_policy") != "Hash-pin the full WORLD plan file without deserializing its combined CAL/EVAL condition list; open only the CAL producer manifest as a dataset.":
        raise ValueError("analysis plan's EVAL-isolation plan-read policy differs from the producer")
    records = manifest.get("records")
    expected_ids = analysis_plan.get("ordered_calibration_ids")
    if not isinstance(records, list) or len(records) != 20 or not isinstance(expected_ids, list):
        raise ValueError("frozen CAL record count or planned ID inventory is invalid")
    ids = [str(row.get("id")) for row in records]
    if ids != expected_ids or len(set(ids)) != 20:
        raise ValueError("ordered CAL record IDs differ from the frozen analysis plan")
    if len({str(row.get("speaker")) for row in records}) != 10:
        raise ValueError("frozen CAL cohort does not contain the expected 10 speakers")
    for record in records:
        if record.get("split") != "calibration" or record.get("pair_qualified") is not True:
            raise ValueError(f"unqualified/non-CAL row in frozen manifest: {record.get('id')}")
    input_bindings_out = {role: entry for role, entry in file_entries.items()}
    return {
        "manifest": manifest,
        "phone_table": phone_table,
        "cal_qc": cal_qc,
        "ap_qc": ap_qc,
        "freeze": freeze,
        "phone_inventory": phone_inventory,
        "analysis_plan": analysis_plan,
        "resolved": resolved,
        "control_file_inventory": input_bindings_out,
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


def _read_inputs(args: argparse.Namespace, np: Any, sf: Any) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    controls = _read_control_files(args, np)
    world = _import_frozen_world_source(controls["resolved"]["world_source"])
    ledger: dict[str, dict[str, Any]] = {}
    pairs: list[dict[str, Any]] = []
    for record in controls["manifest"]["records"]:
        sample_id = str(record["id"])
        sides: dict[str, Any] = {}
        for condition in CONDITIONS:
            info = record[condition]
            feature = _load_feature(info["world_features"], np, ledger, f"{sample_id}/{condition}/WORLD_features")
            audio_path, audio_entry = sha256_required(info["fixed_audio"]["path"], str(info["fixed_audio"]["file_sha256"]), f"{sample_id}/{condition}/fixed_audio")
            ledger[str(audio_path)] = {**audio_entry, "role": f"{sample_id}/{condition}/fixed_audio"}
            audio, audio_meta = _read_float_wav(audio_path, info["fixed_audio"], sf, np, f"{sample_id}/{condition}/fixed_audio")
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            if abs(rms - FIXED_RMS) > FIXED_RMS_ABS_TOL:
                raise ValueError(f"{sample_id}/{condition} fixed audio RMS differs from the frozen target")
            frame_steps = np.diff(feature["time"].astype(np.float64))
            if frame_steps.size and not np.allclose(frame_steps, FRAME_PERIOD_MS / 1000.0, rtol=0.0, atol=1e-7):
                raise ValueError(f"{sample_id}/{condition} WORLD frame clock is not the frozen 5 ms grid")
            if feature["time"].size and (float(feature["time"][0]) < -1e-12 or float(feature["time"][-1]) > len(audio) / SAMPLE_RATE + FRAME_PERIOD_MS / 1000.0 + 1e-9):
                raise ValueError(f"{sample_id}/{condition} WORLD time centers fall outside the decoded audio clock")
            sides[condition] = {"feature": feature, "audio": audio, "audio_meta": audio_meta, "info": info}

        if not np.array_equal(sides["natural"]["feature"]["frequency_hz"], sides["tts"]["feature"]["frequency_hz"]):
            raise ValueError(f"{sample_id} N/T frequency grids differ")
        directions: dict[str, Any] = {}
        for direction_name, rule in DIRECTIONS.items():
            direction = record["directions"][direction_name]
            if direction.get("recipient_condition") != rule["recipient"] or direction.get("donor_condition") != rule["donor"]:
                raise ValueError(f"{sample_id}/{direction_name} direction identity mismatch")
            direction_inputs: dict[str, Any] = {}
            for role, meta in (("phone_map", direction["phone_map"]), ("transfer_audit", direction["parameter_transfer_audit"])):
                path, entry = sha256_required(meta["path"], str(meta["file_sha256"]), f"{sample_id}/{direction_name}/{role}")
                ledger[str(path)] = {**entry, "role": f"{sample_id}/{direction_name}/{role}"}
                direction_inputs[role] = load_json(path) if role == "phone_map" else load_npz(path, np, AUDIT_KEYS)
                direction_inputs[f"{role}_meta"] = meta
                direction_inputs[f"{role}_path"] = path
            rec_condition, don_condition = rule["recipient"], rule["donor"]
            recipient, donor = sides[rec_condition]["feature"], sides[don_condition]["feature"]
            if int(direction["phone_map"].get("matched_phone_count", -1)) != len(direction_inputs["phone_map"].get("matched_phones", [])):
                raise ValueError(f"{sample_id}/{direction_name} matched phone count mismatch")
            phone_map = direction_inputs["phone_map"]
            if phone_map.get("split") != "calibration" or phone_map.get("id") != sample_id or phone_map.get("speaker") != str(record["speaker"]):
                raise ValueError(f"{sample_id}/{direction_name} phone-map identity mismatch")
            if phone_map.get("recipient_condition") != rec_condition or phone_map.get("donor_condition") != don_condition:
                raise ValueError(f"{sample_id}/{direction_name} phone-map direction mismatch")
            if phone_map.get("recipient_source_pcm_sha256") != record[rec_condition]["source_audio_sha256"] or phone_map.get("donor_source_pcm_sha256") != record[don_condition]["source_audio_sha256"]:
                raise ValueError(f"{sample_id}/{direction_name} phone map is bound to other input PCM")

            computed_arms, computed_transfer = world.transfer_parameters(recipient, donor, phone_map, np)
            audit_checks = _compare_transfer(direction_inputs["transfer_audit"], computed_transfer["detail"], np, f"{sample_id}/{direction_name}")
            direction_inputs["recomputed_transfer"] = computed_transfer
            direction_inputs["audit_checks"] = audit_checks
            direction_inputs["parameters"] = {}
            direction_inputs["old_waves"] = {}
            for arm in ("RT", "SP"):
                arm_meta = direction["arms"][arm]
                if arm_meta.get("status") != "ok":
                    raise ValueError(f"{sample_id}/{direction_name}/{arm} original arm is not successful")
                params_meta = arm_meta["parameters"]
                params, params_path = _load_parameter(params_meta, np, ledger, f"{sample_id}/{direction_name}/{arm}/parameters")
                expected_params = computed_arms[arm]
                shape_checks = {}
                for key in ("f0", "sp", "ap"):
                    shape_checks[key] = _require_equal(params[key], expected_params[key], np, f"{sample_id}/{direction_name}/{arm}/{key}")
                shape_checks["time"] = _require_equal(params["time"], recipient["time"], np, f"{sample_id}/{direction_name}/{arm}/time")
                shape_checks["frequency_hz"] = _require_equal(params["frequency_hz"], recipient["frequency_hz"], np, f"{sample_id}/{direction_name}/{arm}/frequency_hz")
                direction_inputs["parameters"][arm] = {"arrays": params, "meta": params_meta, "path": params_path, "checks": shape_checks}
                wave_path, wave_file = sha256_required(arm_meta["path"], str(arm_meta["file_sha256"]), f"{sample_id}/{direction_name}/{arm}/waveform")
                ledger[str(wave_path)] = {**wave_file, "role": f"{sample_id}/{direction_name}/{arm}/waveform"}
                wave, wave_meta = _read_float_wav(wave_path, arm_meta, sf, np, f"{sample_id}/{direction_name}/{arm}/waveform")
                wave_rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
                if len(wave) != len(sides[rec_condition]["audio"]) or abs(wave_rms - FIXED_RMS) > FIXED_RMS_ABS_TOL:
                    raise ValueError(f"{sample_id}/{direction_name}/{arm} old waveform clock or RMS differs")
                direction_inputs["old_waves"][arm] = {"audio": wave, "meta": wave_meta, "path": wave_path}
            rt_params = direction_inputs["parameters"]["RT"]["arrays"]
            sp_params = direction_inputs["parameters"]["SP"]["arrays"]
            for key in ("f0", "time", "ap", "frequency_hz"):
                _require_equal(sp_params[key], rt_params[key], np, f"{sample_id}/{direction_name}/SP_vs_RT/{key}")
            directions[direction_name] = {"meta": direction, "inputs": direction_inputs, "recipient_condition": rec_condition, "donor_condition": don_condition}
        pairs.append({"id": sample_id, "speaker": str(record["speaker"]), "source_identity_sha256": record["source_identity_sha256"], "sides": sides, "directions": directions})
    if len(pairs) != 20 or len(directions) != 2 or len(ledger) != 320:
        raise ValueError(f"CAL input inventory incomplete: pairs={len(pairs)}, artifacts={len(ledger)}, expected=20/320")
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

    relative_dir = Path("artifacts") / "calibration" / sample_id / direction_name
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
        "source_identity_sha256": pair["source_identity_sha256"],
        "direction": direction_name,
        "recipient_condition": recipient_condition,
        "donor_condition": donor_condition,
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
    for path in sorted(p for p in output_root.rglob("*") if p.is_file() and p.name not in {"output_file_inventory.json", "calibration_manifest.json"}):
        rows.append({"path": str(path.relative_to(output_root).as_posix()), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {"schema_version": 1, "manifest_type": "world_phone_shape_factorial_cal_output_file_inventory", "files": rows}
    payload["manifest_sha256"] = fingerprint(payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    controls, pairs, input_ledger = _read_inputs(args, np, sf)
    runtime_identity = _runtime_identity(controls["manifest"], np, sf)
    plan = controls["analysis_plan"]
    if args.preflight_only:
        return {
            "status": "PREFLIGHT_PASS",
            "scope": "calibration_only",
            "split": "calibration",
            "record_count": len(pairs),
            "speaker_count": len({pair["speaker"] for pair in pairs}),
            "direction_count": len(pairs) * len(DIRECTIONS),
            "expected_cell_count": len(pairs) * len(DIRECTIONS) * len(CELLS),
            "input_file_count_verified": len(input_ledger),
            "ordered_ids": [pair["id"] for pair in pairs],
            "directions": list(DIRECTIONS),
            "cells": list(CELLS),
            "evaluation_waveforms_read": False,
            "evaluation_outcomes_read": False,
            "effects_read": False,
            "model_forward_performed": False,
            "world_manifest_raw_sha256": WORLD_MANIFEST_RAW_SHA256,
            "world_manifest_payload_sha256": WORLD_MANIFEST_PAYLOAD_SHA256,
            "analysis_plan_raw_sha256": ANALYSIS_PLAN_RAW_SHA256,
            "analysis_plan_payload_sha256": ANALYSIS_PLAN_PAYLOAD_SHA256,
            "analysis_spec_sha256": ANALYSIS_SPEC_SHA256,
            "calibration_freeze_raw_sha256": CAL_FREEZE_RAW_SHA256,
            "input_inventory": sorted(input_ledger.values(), key=lambda item: (item["role"], item["path"])),
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

    rows = []
    failures = []
    intended = len(pairs) * len(DIRECTIONS)
    for pair in pairs:
        for direction_name in DIRECTIONS:
            try:
                row = _factorize_direction(pair, direction_name, pair["directions"][direction_name], world, np, sf, output_root)
            except Exception as exc:
                row = {
                    "id": pair["id"], "speaker": pair["speaker"], "direction": direction_name,
                    "recipient_condition": DIRECTIONS[direction_name]["recipient"],
                    "donor_condition": DIRECTIONS[direction_name]["donor"],
                    "status": "failed", "failure": f"{type(exc).__name__}:{exc}",
                }
                failures.append({"id": pair["id"], "direction": direction_name, "failure": row["failure"]})
            rows.append(row)
            print(f"CAL shape-factor {len(rows)}/{intended} {pair['id']} {direction_name} {row['status']}", flush=True)

    complete = sum(row.get("status") == "PASS" for row in rows)
    output_inventory = _output_file_inventory(output_root)
    inventory_path = output_root / "output_file_inventory.json"
    write_json(inventory_path, output_inventory)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "manifest_type": "world_phone_shape_mean_dynamic_factorial_calibration",
        "status": "CAL_FACTORIAL_WAVEFORMS_READY_FOR_ROOT_REVIEW" if complete == intended and not failures else "INCOMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "calibration_only",
        "split": "calibration",
        "evaluation_waveforms_read": False,
        "evaluation_outcomes_read": False,
        "effects_read": False,
        "model_forward_performed": False,
        "analysis_spec_path": str(args.spec.resolve()),
        "analysis_spec_raw_sha256": ANALYSIS_SPEC_SHA256,
        "analysis_plan_path": str(args.analysis_plan.resolve()),
        "analysis_plan_raw_sha256": ANALYSIS_PLAN_RAW_SHA256,
        "analysis_plan_payload_sha256": ANALYSIS_PLAN_PAYLOAD_SHA256,
        "analysis_source_path": str(Path(__file__).resolve()),
        "analysis_source_sha256": sha256_file(Path(__file__).resolve()),
        "input_bindings": controls["control_file_inventory"],
        "calibration_root_freeze_status": controls["freeze"]["status"],
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
            "coordinate": "log normalized SP shape; per-frequency-bin token mean and centered residual; row softmax",
            "q00_q11_softmax_formula_tolerance": {
                "elementwise_rule": "abs(actual-endpoint) <= 1e-12 + 1e-10*abs(endpoint)",
                "relative_denominator_floor": SOFTMAX_RELATIVE_DENOMINATOR_FLOOR,
                "formula_bit_exact_claim": False,
            },
            "epsilon_floor_mass": "sum(max(epsilon-normalized_shape,0)) per row; report observed mass and frequency_bin_count*epsilon bound",
            "q00": "recipient phone mean + recipient within-phone residual",
            "q10": "donor-mapped phone mean + recipient within-phone residual",
            "q01": "recipient phone mean + donor-mapped within-phone residual",
            "q11": "donor-mapped phone mean + donor-mapped within-phone residual",
            "self_grid": "recipient-only log shape interpolated to donor 5ms centers and back; separate grid-cycle control, not a donor arm or pure interpolation proof",
            "token_mean_domain": "same unique mapped recipient 5ms frames for the exact phone-token pair; no pooling by label or across repeated tokens",
            "ambiguous_frames": "excluded from token means and kept at recipient shape in all five cells; original q11 SP endpoint must agree exactly there if mapped, and the full-grid bridge must pass before endpoint copy",
            "unmapped_frames": "recipient shape in every cell",
            "other_parameters": "recipient F0, AP, time centers and frequency grid unchanged bit-for-bit",
            "row_power": "multiply every cell's normalized shape by recipient SP row sum",
            "rms_policy": "frozen fixed global RMS after synthesis and exact right trim",
            "inference_labels": controls["phone_inventory"]["selected_phone_labels"],
            "all_matched_labels_retained": True,
        },
        "intended_direction_rows": intended,
        "complete_direction_rows": complete,
        "failure_count": len(failures),
        "failures": failures,
        "records": rows,
    }
    payload["manifest_sha256"] = fingerprint(payload)
    out_path = output_root / "calibration_manifest.json"
    write_json(out_path, payload)
    return {"status": payload["status"], "direction_rows": len(rows), "complete_direction_rows": complete, "failures": len(failures), "manifest_sha256": payload["manifest_sha256"], "manifest_path": str(out_path), "output_inventory_sha256": output_inventory["manifest_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-manifest", required=True, type=Path)
    parser.add_argument("--world-plan", required=True, type=Path)
    parser.add_argument("--world-source", required=True, type=Path)
    parser.add_argument("--execution-bindings", required=True, type=Path)
    parser.add_argument("--phone-table", required=True, type=Path)
    parser.add_argument("--calibration-qc", required=True, type=Path)
    parser.add_argument("--calibration-ap-qc", required=True, type=Path)
    parser.add_argument("--calibration-freeze", required=True, type=Path)
    parser.add_argument("--phone-inventory", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--analysis-plan", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.preflight_only and args.out is None:
        parser.error("--out is required unless --preflight-only is used")
    result = run(args)
    if args.preflight_only:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0 if result["status"] == "CAL_FACTORIAL_WAVEFORMS_READY_FOR_ROOT_REVIEW" and result["complete_direction_rows"] == 40 and result["failures"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
