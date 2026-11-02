#!/usr/bin/env python3
"""CAL geometry and fresh decoder runner for the RMS latent factorization.

The input endpoints are the exact Wav2Lip audio-encoder latents saved by the
frozen mel partition run (q00=RAW and q11=RMS-fixed). This program does not
recompute the mel frontend or call the audio encoder for EVAL. The mel runner
is imported only for sealed runtime, rendering, SyncNet and score utilities.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SPEC_SHA256 = "3eaf572afd83682e5dc9d09dc325313bc2e16b969183fc8f89d391424b861e23"
MEL_SPEC_SHA256 = "abeced0f6f50a94fe49fc085af40358071dacc5091c7c796c26c5edafbe63ee8"
TARGET_RMS = 0.0376838172675746
CONDITIONS = ("q00", "q10", "q01", "q11")
EXPECTED_CAL_FILES = 20
EXPECTED_EVAL_ROWS = 80
EPS = np.finfo(np.float64).tiny


class FactorizationError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _payload_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical({key: item for key, item in value.items() if key != "payload_sha256"})).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FactorizationError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha_required(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FactorizationError(f"missing {label}: {path}")
    actual = _sha_file(path)
    if actual != expected:
        raise FactorizationError(f"{label} SHA mismatch: {path} ({actual} != {expected})")


def _load_mel_runner(path: Path):
    source = path.resolve(strict=True)
    if str(source.parent) not in sys.path:
        sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("mel_partition_runner_bound", source)
    if spec is None or spec.loader is None:
        raise FactorizationError(f"cannot load mel runner utility module: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _geometry(z0: np.ndarray, z1: np.ndarray) -> dict[str, np.ndarray]:
    if z0.dtype != np.float32 or z1.dtype != np.float32 or z0.ndim != 2 or z1.shape != z0.shape or z0.shape[1] != 512:
        raise FactorizationError(f"latent endpoints must be equal float32 [frames,512]: {z0.shape}/{z0.dtype} {z1.shape}/{z1.dtype}")
    if not np.isfinite(z0).all() or not np.isfinite(z1).all():
        raise FactorizationError("latent endpoint has nonfinite values")
    a = z0.astype(np.float64)
    b = z1.astype(np.float64)
    r0 = np.linalg.norm(a, axis=1)
    r1 = np.linalg.norm(b, axis=1)
    if not np.isfinite(r0).all() or not np.isfinite(r1).all() or np.any(r0 <= 0) or np.any(r1 <= 0):
        raise FactorizationError("latent endpoints have nonfinite or nonpositive frame radii")
    unit0 = a / r0[:, None]
    unit1 = b / r1[:, None]
    cosine = np.sum(unit0 * unit1, axis=1, dtype=np.float64)
    q10 = (b * (r0 / r1)[:, None]).astype(np.float32)
    q01 = (a * (r1 / r0)[:, None]).astype(np.float32)
    q00 = z0.copy()
    q11 = z1.copy()
    r_q10 = np.linalg.norm(q10.astype(np.float64), axis=1)
    r_q01 = np.linalg.norm(q01.astype(np.float64), axis=1)
    rel10 = np.abs(r_q10 - r0) / np.maximum(r0, EPS)
    rel01 = np.abs(r_q01 - r1) / np.maximum(r1, EPS)
    interaction = b - q10.astype(np.float64) - q01.astype(np.float64) + a
    analytic_interaction = (r1 - r0)[:, None] * (unit1 - unit0)
    return {
        "q00": q00,
        "q10": q10,
        "q01": q01,
        "q11": q11,
        "r0_float64": r0,
        "r1_float64": r1,
        "radius_ratio_r1_over_r0_float64": r1 / r0,
        "cosine_float64": cosine,
        "direction_delta_l2_float64": np.linalg.norm(unit1 - unit0, axis=1),
        "radius_delta_float64": r1 - r0,
        "q10_delta_from_q00_l2_float64": np.linalg.norm(q10.astype(np.float64) - a, axis=1),
        "q01_delta_from_q00_l2_float64": np.linalg.norm(q01.astype(np.float64) - a, axis=1),
        "interaction_l2_float64": np.linalg.norm(interaction, axis=1),
        "analytic_interaction_l2_float64": np.linalg.norm(analytic_interaction, axis=1),
        "q10_mixed_relative_radius_error_float64": rel10,
        "q01_mixed_relative_radius_error_float64": rel01,
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    return {"p05": float(np.quantile(v, 0.05)), "p50": float(np.quantile(v, 0.50)), "p95": float(np.quantile(v, 0.95)), "mean": float(np.mean(v, dtype=np.float64)), "max_abs": float(np.max(np.abs(v), initial=0.0))}


def calibration_qa(args: argparse.Namespace) -> None:
    source_dir = args.calibration_dir.resolve(strict=True)
    spec_path = args.spec.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FactorizationError(f"output already exists; use a fresh attempt path: {output}")
    _sha_required(spec_path, SPEC_SHA256, "frozen RMS factorization spec")
    files = sorted(source_dir.glob("*.npz"))
    if len(files) != EXPECTED_CAL_FILES:
        raise FactorizationError(f"expected 20 original CAL feature NPZs, found {len(files)}")
    attempts_path = source_dir.parent / "calibration_attempts.json"
    summary_path = source_dir.parent / "calibration_summary.json"
    attempts = _read_json(attempts_path) if attempts_path.is_file() else {}
    successful_attempt = next((item for item in attempts.get("attempts", []) if item.get("attempt") == 3), None)
    if successful_attempt is not None and (successful_attempt.get("status") != "succeeded" or successful_attempt.get("rows_completed") != EXPECTED_CAL_FILES):
        raise FactorizationError("original calibration attempt 3 is not recorded as a complete 20-row run")
    output.mkdir(parents=True)
    clip_rows: list[dict[str, Any]] = []
    speaker_rows: dict[str, list[dict[str, Any]]] = {}
    all_arrays: dict[str, np.ndarray] = {}
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            required = {"natural_z_raw", "natural_z_fixed"}
            if not required.issubset(archive.files):
                raise FactorizationError(f"CAL feature keys missing in {path.name}: {sorted(required-set(archive.files))}")
            z0 = np.asarray(archive["natural_z_raw"])
            z1 = np.asarray(archive["natural_z_fixed"])
            archive_keys = list(archive.files)
        geometry = _geometry(z0, z1)
        if np.max(geometry["q10_mixed_relative_radius_error_float64"], initial=0.0) > 1e-6 or np.max(geometry["q01_mixed_relative_radius_error_float64"], initial=0.0) > 1e-6:
            raise FactorizationError(f"float32 mixed-cell radius error exceeds the frozen 1e-6 bound: {path.name}")
        if not np.array_equal(geometry["q00"], z0) or not np.array_equal(geometry["q11"], z1):
            raise FactorizationError(f"factorization endpoint copy identity failed: {path.name}")
        analytic_error = geometry["interaction_l2_float64"] - geometry["analytic_interaction_l2_float64"]
        stem = path.stem
        arrays_path = output / "arrays" / f"{stem}_geometry.npz"
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"z0_q00_raw_float32": z0, "z1_q11_fixed_float32": z1, **geometry}
        np.savez_compressed(arrays_path, **payload)
        array_info = {name: {"shape": list(value.shape), "dtype": str(value.dtype), "array_sha256": _sha_array(value)} for name, value in payload.items()}
        row = {
            "id": stem,
            "speaker": stem.split("_", 1)[0],
            "status": "qualified",
            "source_file": str(path.resolve()),
            "source_file_sha256": _sha_file(path),
            "source_archive_keys": archive_keys,
            "endpoint_arrays": {
                "natural_z_raw": {"shape": list(z0.shape), "dtype": str(z0.dtype), "array_sha256": _sha_array(z0)},
                "natural_z_fixed": {"shape": list(z1.shape), "dtype": str(z1.dtype), "array_sha256": _sha_array(z1)},
            },
            "frames": int(z0.shape[0]),
            "geometry_arrays": {"path": str(arrays_path.resolve()), "file_sha256": _sha_file(arrays_path), "arrays": array_info},
            "r0": _quantiles(geometry["r0_float64"]),
            "r1": _quantiles(geometry["r1_float64"]),
            "r1_over_r0": _quantiles(geometry["radius_ratio_r1_over_r0_float64"]),
            "cosine": _quantiles(geometry["cosine_float64"]),
            "direction_delta_l2": _quantiles(geometry["direction_delta_l2_float64"]),
            "radius_delta": _quantiles(geometry["radius_delta_float64"]),
            "q10_delta_from_q00_l2": _quantiles(geometry["q10_delta_from_q00_l2_float64"]),
            "q01_delta_from_q00_l2": _quantiles(geometry["q01_delta_from_q00_l2_float64"]),
            "latent_interaction_l2": _quantiles(geometry["interaction_l2_float64"]),
            "analytic_interaction_identity_max_abs_error": float(np.max(np.abs(analytic_error), initial=0.0)),
            "q10_mixed_relative_radius_error_max": float(np.max(geometry["q10_mixed_relative_radius_error_float64"], initial=0.0)),
            "q01_mixed_relative_radius_error_max": float(np.max(geometry["q01_mixed_relative_radius_error_float64"], initial=0.0)),
            "all_endpoints_finite": True,
            "all_frame_radii_positive": True,
            "q00_q11_endpoints_exact_copy": True,
            "model_forward_performed": False,
            "effects_read": False,
        }
        clip_rows.append(row)
        speaker_rows.setdefault(row["speaker"], []).append(row)
        print(f"qualified {stem} frames={z0.shape[0]} ratio_median={row['r1_over_r0']['p50']:.6f} cos_median={row['cosine']['p50']:.6f}", flush=True)
    def clip_equal(field: str, metric: str) -> dict[str, float]:
        values = [row[field][metric] for row in clip_rows]
        return _quantiles(np.asarray(values, dtype=np.float64))
    speaker_means: dict[str, Any] = {}
    for field in ("r1_over_r0", "cosine", "direction_delta_l2", "radius_delta", "q10_delta_from_q00_l2", "q01_delta_from_q00_l2", "latent_interaction_l2"):
        speaker_means[field] = _quantiles(np.asarray([np.mean([row[field]["p50"] for row in rows], dtype=np.float64) for rows in speaker_rows.values()], dtype=np.float64))
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "confirm-rms-latent-factorization-20261001",
        "status": "PASS",
        "created_at_utc": _utc_now(),
        "spec": {"path": str(spec_path), "sha256": _sha_file(spec_path)},
        "producer_code": {"path": str(Path(__file__).resolve()), "sha256": _sha_file(Path(__file__))},
        "calibration_feature_root": str(source_dir),
        "calibration_attempt_record": {"path": str(attempts_path.resolve()), "raw_sha256": _sha_file(attempts_path)} if attempts_path.is_file() else None,
        "calibration_summary_record": {"path": str(summary_path.resolve()), "raw_sha256": _sha_file(summary_path)} if summary_path.is_file() else None,
        "source_encoder_code_sha256": successful_attempt.get("code_sha256", {}) if successful_attempt else None,
        "calibration_input_file_count": len(files),
        "calibration_clip_count": len(clip_rows),
        "calibration_speaker_count": len(speaker_rows),
        "target_rms": TARGET_RMS,
        "factorization": {
            "z0": "natural_z_raw exact saved float32 encoder output",
            "z1": "natural_z_fixed exact saved float32 encoder output",
            "q10": "float32(z1 * r0/r1), fixed direction and raw radius",
            "q01": "float32(z0 * r1/r0), raw direction and fixed radius",
            "radii_computed_in": "float64",
            "mixed_relative_norm_error_limit": 1e-6,
            "zero_radius_policy": "row retained as failed; no replacement direction",
        },
        "calibration_summary": {
            "clip_level_equal_weight": {field: {"p50_of_clip_medians": float(np.median([row[field]["p50"] for row in clip_rows])), "mean_of_clip_medians": float(np.mean([row[field]["p50"] for row in clip_rows]))} for field in speaker_means},
            "speaker_level_equal_weight": speaker_means,
            "max_q10_relative_radius_error": max(row["q10_mixed_relative_radius_error_max"] for row in clip_rows),
            "max_q01_relative_radius_error": max(row["q01_mixed_relative_radius_error_max"] for row in clip_rows),
            "max_analytic_interaction_identity_error": max(row["analytic_interaction_identity_max_abs_error"] for row in clip_rows),
        },
        "rows": clip_rows,
        "model_forward_performed": False,
        "effects_read": False,
    }
    report["payload_sha256"] = _payload_sha(report)
    _write_json(output / "calibration_geometry_qa.json", report)
    print(json.dumps({"status": "PASS", "clips": len(clip_rows), "speakers": len(speaker_rows), "report": str(output / "calibration_geometry_qa.json"), "report_sha256": _sha_file(output / "calibration_geometry_qa.json"), "model_forward_performed": False, "effects_read": False}, ensure_ascii=False, sort_keys=True), flush=True)


def _jsonl_read(path: Path) -> list[dict[str, Any]]:
    result = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise FactorizationError(f"JSONL line {line_no} is not an object: {path}")
        result.append(item)
    return result


def _array_bridge(reference_path: str | Path, current: np.ndarray) -> dict[str, Any]:
    path = Path(reference_path).resolve(strict=True)
    reference = np.load(path, allow_pickle=False)
    if reference.shape != current.shape:
        return {"reference_path": str(path), "reference_file_sha256": _sha_file(path), "shape_match": False, "reference_shape": list(reference.shape), "current_shape": list(current.shape)}
    delta = current.astype(np.float64) - reference.astype(np.float64)
    return {
        "reference_path": str(path),
        "reference_file_sha256": _sha_file(path),
        "reference_array_sha256": _sha_array(reference),
        "current_array_sha256": _sha_array(current),
        "shape_match": True,
        "dtype_match": str(reference.dtype) == str(current.dtype),
        "bit_exact": bool(np.array_equal(reference, current)),
        "max_abs_difference": float(np.max(np.abs(delta), initial=0.0)),
        "mean_abs_difference": float(np.mean(np.abs(delta), dtype=np.float64)),
    }


def _video_pixel_bridge(reference_path: str | Path, current_path: str | Path) -> dict[str, Any]:
    import cv2

    reference_path, current_path = Path(reference_path).resolve(strict=True), Path(current_path).resolve(strict=True)
    left, right = cv2.VideoCapture(str(reference_path)), cv2.VideoCapture(str(current_path))
    if not left.isOpened() or not right.isOpened():
        left.release()
        right.release()
        raise FactorizationError("cannot decode baseline/current bridge video")
    compared_frames = 0
    equal_pixels = total_pixels = abs_sum = 0
    max_abs = 0
    shape_match = True
    try:
        while True:
            ok_l, frame_l = left.read()
            ok_r, frame_r = right.read()
            if not ok_l or not ok_r:
                break
            compared_frames += 1
            if frame_l.shape != frame_r.shape:
                shape_match = False
                continue
            diff = np.abs(frame_l.astype(np.int16) - frame_r.astype(np.int16))
            equal_pixels += int(np.count_nonzero(diff == 0))
            total_pixels += int(diff.size)
            abs_sum += int(diff.sum(dtype=np.int64))
            max_abs = max(max_abs, int(diff.max(initial=0)))
        left_count = int(left.get(cv2.CAP_PROP_FRAME_COUNT))
        right_count = int(right.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        left.release()
        right.release()
    return {
        "reference_video_path": str(reference_path),
        "reference_video_file_sha256": _sha_file(reference_path),
        "current_video_path": str(current_path),
        "current_video_file_sha256": _sha_file(current_path),
        "reference_reported_frames": left_count,
        "current_reported_frames": right_count,
        "compared_frames": compared_frames,
        "frame_count_match": left_count == right_count == compared_frames,
        "frame_shape_match": shape_match,
        "pixel_count_compared": total_pixels,
        "pixel_exact_fraction": (equal_pixels / total_pixels) if total_pixels else None,
        "pixel_mean_abs_difference": (abs_sum / total_pixels) if total_pixels else None,
        "pixel_max_abs_difference": max_abs if total_pixels else None,
    }


def _load_latent(meta: Mapping[str, Any], condition: str, sample_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(str(meta.get("path", ""))).resolve(strict=True)
    _sha_required(path, str(meta.get("file_sha256", "")), f"mel {condition} latent file {sample_id}")
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float32 or value.ndim != 2 or value.shape[1] != 512 or not np.isfinite(value).all():
        raise FactorizationError(f"mel {condition} latent violates float32 [chunks,512] finite contract: {sample_id} {value.shape}/{value.dtype}")
    if list(value.shape) != meta.get("shape") or str(value.dtype) != meta.get("dtype") or _sha_array(value) != meta.get("array_sha256"):
        raise FactorizationError(f"mel {condition} latent metadata mismatch: {sample_id}")
    return value, {"path": str(path), "file_sha256": _sha_file(path), "array_sha256": _sha_array(value), "shape": list(value.shape), "dtype": str(value.dtype)}


def prepare_eval(args: argparse.Namespace) -> None:
    """Seal all 80 latent factorization rows and assets before fresh decoding."""
    mel = _load_mel_runner(args.mel_runner.resolve(strict=True))
    mel_control = args.mel_control.resolve(strict=True)
    cal_qa_path = args.calibration_qa.resolve(strict=True)
    spec_path = args.spec.resolve(strict=True)
    output = args.factor_dir.resolve()
    outdir = args.outdir.resolve()
    control = args.control_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FactorizationError(f"factor output directory is not empty: {output}")
    if control.exists() and any(control.iterdir()):
        raise FactorizationError(f"control directory is not empty: {control}")
    if output == control or output in control.parents or control in output.parents:
        raise FactorizationError("factor output and control paths must be disjoint")
    if outdir.exists() and any(outdir.iterdir()):
        raise FactorizationError(f"fresh decoder output directory must be absent or empty: {outdir}")
    if any(left == right or left in right.parents or right in left.parents for left, right in ((output, outdir), (control, outdir))):
        raise FactorizationError("factor, decoder output and control paths must be disjoint")
    _sha_required(spec_path, SPEC_SHA256, "frozen RMS factorization spec")
    cal_qa = _read_json(cal_qa_path)
    if cal_qa.get("payload_sha256") != _payload_sha(cal_qa):
        raise FactorizationError("CAL geometry QA payload fingerprint mismatch")
    if cal_qa.get("status") != "PASS" or cal_qa.get("model_forward_performed") is not False or cal_qa.get("effects_read") is not False:
        raise FactorizationError("CAL geometry QA is not a PASS no-forward/no-effects artifact")
    if cal_qa.get("spec", {}).get("sha256") != SPEC_SHA256 or cal_qa.get("calibration_clip_count") != EXPECTED_CAL_FILES:
        raise FactorizationError("CAL geometry QA does not match the frozen 20-clip factorization protocol")
    cal_source = cal_qa.get("producer_code")
    if not isinstance(cal_source, Mapping) or cal_source.get("sha256") != _sha_file(Path(__file__)):
        raise FactorizationError("CAL geometry QA was not produced by this exact RMS runner source")
    plan_path = mel_control / "execution_plan.json"
    mel_run_path = mel_control / "run_manifest.json"
    results_path = mel_control / "execution_results.jsonl"
    plan = _read_json(plan_path)
    manifest = _read_json(mel_run_path)
    if plan.get("payload_sha256") != mel._payload_sha(plan) or manifest.get("payload_sha256") != mel._payload_sha(manifest):
        raise FactorizationError("mel execution plan or final run manifest payload fingerprint mismatch")
    if plan.get("spec", {}).get("sha256") != MEL_SPEC_SHA256:
        raise FactorizationError("mel source run is not bound to the frozen mel partition specification")
    _sha_required(Path(str(plan["spec"]["path"])), MEL_SPEC_SHA256, "frozen mel partition spec")
    if plan.get("code", {}).get("sha256") != mel._sha256_file(mel.__file__):
        raise FactorizationError("mel runner source differs from its frozen plan")
    if manifest.get("status") != "complete" or manifest.get("planned_rows") != EXPECTED_EVAL_ROWS or manifest.get("planned_cells") != 320:
        raise FactorizationError("mel run is not a complete 80x4 run")
    if manifest.get("model_forward_performed") is not True or manifest.get("effects_read") is not False:
        raise FactorizationError("mel run does not have the required executed/no-effects status")
    if Path(str(manifest.get("execution_plan_path", ""))).resolve() != plan_path.resolve():
        raise FactorizationError("mel run manifest is not attached to the supplied preflight control")
    if manifest.get("execution_plan_raw_sha256") != _sha_file(plan_path) or manifest.get("execution_plan_payload_sha256") != plan.get("payload_sha256"):
        raise FactorizationError("mel run manifest does not bind the supplied frozen execution plan")
    if Path(str(manifest.get("rows_path", ""))).resolve() != results_path.resolve():
        raise FactorizationError("mel run manifest rows path differs from the supplied result file")
    _sha_required(Path(manifest["execution_runtime_path"]), manifest["execution_runtime_raw_sha256"], "mel runtime record")
    _sha_required(results_path, manifest["rows_raw_sha256"], "mel execution result rows")
    results = _jsonl_read(results_path)
    samples = plan.get("planned_samples")
    if not isinstance(samples, list) or len(samples) != EXPECTED_EVAL_ROWS or len(results) != EXPECTED_EVAL_ROWS:
        raise FactorizationError("mel run does not retain all 80 planned sample rows")
    rows_by_id = {str(row.get("id")): row for row in results}
    if len(rows_by_id) != EXPECTED_EVAL_ROWS:
        raise FactorizationError("mel run results contain duplicate or missing sample IDs")
    output.mkdir(parents=True, exist_ok=True)
    factor_rows = []
    source_root = Path(str(manifest["artifact_root"]))
    for sample in samples:
        sample_id = str(sample["id"])
        result = rows_by_id.get(sample_id)
        base_row: dict[str, Any] = {"id": sample_id, "speaker": sample["speaker"], "split": "evaluation", "status": "failed", "failure": None, "source_audio": {key: sample.get(key) for key in ("source_wav_path", "source_wav_file_sha256", "source_pcm_array_sha256", "source_waveform_array_sha256", "sample_count", "sample_rate_hz", "fixed_pcm_path", "fixed_pcm_file_sha256", "fixed_pcm_array_sha256", "fixed_pcm_target_rms")}}
        try:
            if result is None or result.get("status") != "complete":
                raise FactorizationError("source mel run row is not complete")
            if sample.get("fixed_pcm_target_rms") != TARGET_RMS:
                raise FactorizationError("source mel row is not bound to the frozen fixed RMS target")
            raw_pcm = mel._read_raw_pcm(Path(sample["source_wav_path"]), int(sample["sample_count"]), sample["source_wav_file_sha256"], sample["source_pcm_array_sha256"])
            fixed_path = Path(sample["fixed_pcm_path"])
            _sha_required(fixed_path, sample["fixed_pcm_file_sha256"], f"RMS-fixed PCM {sample_id}")
            fixed_pcm = np.load(fixed_path, allow_pickle=False)
            if fixed_pcm.dtype != np.float32 or fixed_pcm.ndim != 1 or len(fixed_pcm) != len(raw_pcm) or not np.isfinite(fixed_pcm).all() or _sha_array(fixed_pcm) != sample.get("fixed_pcm_array_sha256"):
                raise FactorizationError("RMS-fixed float32 PCM fails the frozen source length/hash contract")
            cells = result.get("cells")
            if not isinstance(cells, Mapping):
                raise FactorizationError("source mel run row has no cell records")
            latents = {}
            latent_meta = {}
            for condition in ("q00", "q11"):
                cell = cells.get(condition)
                if not isinstance(cell, Mapping) or cell.get("status") != "generated_scored" or not isinstance(cell.get("latents"), Mapping):
                    raise FactorizationError(f"source mel endpoint cell is incomplete: {condition}")
                value, meta = _load_latent(cell["latents"], condition, sample_id)
                if value.shape[0] != int(sample.get("chunk_count", -1)):
                    raise FactorizationError(f"mel endpoint latent length differs from its mel chunk count: {condition}")
                if cell["latents"].get("input_chunk_array_sha256") != sample.get("mel_arrays", {}).get(f"chunks_{condition}", {}).get("array_sha256"):
                    raise FactorizationError(f"mel endpoint latent is not bound to the corresponding frozen mel chunk array: {condition}")
                if cell["latents"].get("input_chunk_file_sha256") != sample.get("mel_file_sha256"):
                    raise FactorizationError(f"mel endpoint latent is not bound to the frozen mel NPZ file: {condition}")
                latents[condition] = value
                meta["encoder_forward_events"] = cell["latents"].get("encoder_forward_events", [])
                meta["input_chunk_array_sha256"] = cell["latents"].get("input_chunk_array_sha256")
                meta["input_chunk_file_sha256"] = cell["latents"].get("input_chunk_file_sha256")
                latent_meta[condition] = meta
            geometry = _geometry(latents["q00"], latents["q11"])
            if np.max(geometry["q10_mixed_relative_radius_error_float64"], initial=0.0) > 1e-6:
                raise FactorizationError("q10 mixed-cell relative radius error exceeds frozen 1e-6 limit")
            if np.max(geometry["q01_mixed_relative_radius_error_float64"], initial=0.0) > 1e-6:
                raise FactorizationError("q01 mixed-cell relative radius error exceeds frozen 1e-6 limit")
            if not np.array_equal(geometry["q00"], latents["q00"]) or not np.array_equal(geometry["q11"], latents["q11"]):
                raise FactorizationError("q00/q11 exact saved mel endpoint identity failed")
            factor_path = output / "factors" / f"{sample_id}.npz"
            factor_path.parent.mkdir(parents=True, exist_ok=True)
            mixed_payload = {key: geometry[key] for key in geometry if key not in ("q00", "q11")}
            np.savez_compressed(factor_path, **mixed_payload)
            factor_arrays = {key: {"shape": list(value.shape), "dtype": str(value.dtype), "array_sha256": _sha_array(value)} for key, value in mixed_payload.items()}
            base_row.update({
                "status": "prepared",
                "fixed_pcm_actual_rms": float(np.sqrt(np.mean(np.square(fixed_pcm.astype(np.float64)), dtype=np.float64))),
                "latent_frames": int(latents["q00"].shape[0]),
                "endpoint_latents": latent_meta,
                "baseline_endpoint_outputs": {
                    condition: {"video": cells[condition].get("video"), "V": cells[condition].get("V")}
                    for condition in ("q00", "q11")
                },
                "mixed_latents": {
                    "path": str(factor_path.resolve()),
                    "file_sha256": _sha_file(factor_path),
                    "arrays": factor_arrays,
                    "q10_array_sha256": factor_arrays["q10"]["array_sha256"],
                    "q01_array_sha256": factor_arrays["q01"]["array_sha256"],
                },
                "r0_p50": float(np.median(geometry["r0_float64"])),
                "r1_p50": float(np.median(geometry["r1_float64"])),
                "r1_over_r0_p50": float(np.median(geometry["radius_ratio_r1_over_r0_float64"])),
                "cosine_p50": float(np.median(geometry["cosine_float64"])),
                "q10_mixed_relative_radius_error_max": float(np.max(geometry["q10_mixed_relative_radius_error_float64"], initial=0.0)),
                "q01_mixed_relative_radius_error_max": float(np.max(geometry["q01_mixed_relative_radius_error_float64"], initial=0.0)),
                "model_forward_performed": False,
                "effects_read": False,
            })
        except Exception as exc:
            base_row["failure"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        factor_rows.append(base_row)
        print(f"{base_row['status']} {sample_id} frames={base_row.get('latent_frames', 0)}", flush=True)
    mel_module_sha = _sha_file(Path(mel.__file__))
    if mel_module_sha != plan.get("code", {}).get("sha256"):
        raise FactorizationError("mel runner utility module changed while preparing factorization")
    factor_rows_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in factor_rows).encode("utf-8")
    factor_rows_path = control / "factorization_rows.jsonl"
    control.mkdir(parents=True, exist_ok=True)
    factor_rows_path.write_bytes(factor_rows_bytes)
    local_assets, normalized_assets = mel._asset_registry(Path(str(plan["execution_assets"]["registry_path"])))
    if normalized_assets["registry_raw_sha256"] != plan.get("execution_assets_raw_sha256"):
        raise FactorizationError("mel execution asset registry changed after source run")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    runtime = mel._runtime_metadata()
    if not runtime.get("cuda_available"):
        raise FactorizationError("EVAL factorization runner preflight requires CUDA-enabled runner host")
    factor_plan: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": "confirm-rms-latent-factorization-20261001",
        "consumer_id": "tts-rms-latent-factorization-gpu-20261001",
        "status": "frozen_pre_forward",
        "created_at_utc": _utc_now(),
        "spec": {"path": str(spec_path), "sha256": _sha_file(spec_path)},
        "mel_spec_sha256": MEL_SPEC_SHA256,
        "code": {"path": str(Path(__file__).resolve()), "sha256": _sha_file(Path(__file__))},
        "mel_runner": {"path": str(Path(mel.__file__).resolve()), "sha256": mel_module_sha},
        "calibration_geometry_qa": {"path": str(cal_qa_path), "raw_sha256": _sha_file(cal_qa_path), "payload_sha256": cal_qa.get("payload_sha256"), "status": cal_qa.get("status")},
        "factor_output_root": str(output),
        "output_dir": str(outdir),
        "control_dir": str(control),
        "mel_source_run": {
            "plan_path": str(plan_path), "plan_raw_sha256": _sha_file(plan_path), "plan_payload_sha256": plan.get("payload_sha256"),
            "run_manifest_path": str(mel_run_path), "run_manifest_raw_sha256": _sha_file(mel_run_path), "run_manifest_payload_sha256": manifest.get("payload_sha256"),
            "results_path": str(results_path), "results_raw_sha256": _sha_file(results_path), "artifact_root": str(source_root),
        },
        "target_rms": TARGET_RMS,
        "input_identity_sha256": plan.get("input_identity_sha256"),
        "execution_assets": normalized_assets,
        "execution_assets_raw_sha256": normalized_assets["registry_raw_sha256"],
        "runtime": runtime,
        "factorization_definition": {
            "q00": "exact mel-run q00 audio-encoder float32 latent z0",
            "q10": "float32(float64(z1) * r0/r1), fixed direction and raw radius",
            "q01": "float32(float64(z0) * r1/r0), raw direction and fixed radius",
            "q11": "exact mel-run q11 audio-encoder float32 latent z1",
            "radii": "float64 rowwise L2 norm over 512 dimensions",
            "mixed_relative_radius_error_limit": 1e-6,
            "zero_radius": "retain planned row as failure; no replacement",
        },
        "forward_flag_policy": {
            "torch_use_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cudnn_allow_tf32": False,
            "matmul_allow_tf32": False,
            "torch_numpy_seed": 20260930,
            "cublas_workspace_config": ":4096:8",
        },
        "planned_samples": factor_rows,
        "planned_sample_count": len(factor_rows),
        "planned_decoder_rows": len(factor_rows) * 4,
        "model_forward_performed": False,
        "effects_read": False,
    }
    factor_plan["rows_path"] = str(factor_rows_path.resolve())
    factor_plan["rows_raw_sha256"] = _sha_file(factor_rows_path)
    factor_plan["payload_sha256"] = _payload_sha(factor_plan)
    _write_json(control / "execution_plan.json", factor_plan)
    shutil.copyfile(Path(__file__).resolve(), control / "producer_snapshot.py")
    prepared = sum(row["status"] == "prepared" for row in factor_rows)
    summary = {
        "status": "PASS" if prepared == EXPECTED_EVAL_ROWS else "INCOMPLETE",
        "plan_path": str((control / "execution_plan.json").resolve()),
        "plan_raw_sha256": _sha_file(control / "execution_plan.json"),
        "plan_payload_sha256": factor_plan["payload_sha256"],
        "rows_path": str(factor_rows_path.resolve()),
        "rows_raw_sha256": factor_plan["rows_raw_sha256"],
        "prepared_rows": prepared,
        "planned_rows": EXPECTED_EVAL_ROWS,
        "planned_decoder_rows": len(factor_rows) * 4,
        "factor_output_root": str(output),
        "model_forward_performed": False,
        "effects_read": False,
    }
    _write_json(control / "preflight_report.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    if summary["status"] != "PASS":
        raise FactorizationError(f"EVAL factorization preflight is incomplete: prepared {prepared}/{EXPECTED_EVAL_ROWS} rows; report retained at {control / 'preflight_report.json'}")


def run(args: argparse.Namespace) -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise FactorizationError("CUBLAS_WORKSPACE_CONFIG=:4096:8 must be set by the launcher before CUDA initialization")
    plan_path = args.plan.resolve(strict=True)
    plan = _read_json(plan_path)
    if plan.get("payload_sha256") != _payload_sha(plan) or plan.get("code", {}).get("sha256") != _sha_file(Path(__file__)):
        raise FactorizationError("factorization plan or runner source seal mismatch")
    _sha_required(Path(plan["spec"]["path"]), SPEC_SHA256, "frozen RMS factorization spec")
    source_run = plan["mel_source_run"]
    _sha_required(Path(source_run["plan_path"]), source_run["plan_raw_sha256"], "mel source plan")
    _sha_required(Path(source_run["run_manifest_path"]), source_run["run_manifest_raw_sha256"], "mel source run manifest")
    _sha_required(Path(source_run["results_path"]), source_run["results_raw_sha256"], "mel source result rows")
    mel_spec_record = _read_json(Path(source_run["plan_path"])).get("spec", {})
    if mel_spec_record.get("sha256") != MEL_SPEC_SHA256:
        raise FactorizationError("bound mel source plan no longer names the frozen mel specification")
    _sha_required(Path(str(mel_spec_record.get("path", ""))), MEL_SPEC_SHA256, "frozen mel partition spec")
    mel = _load_mel_runner(Path(plan["mel_runner"]["path"]))
    if _sha_file(Path(mel.__file__)) != plan["mel_runner"]["sha256"]:
        raise FactorizationError("bound mel runner utility source changed after preflight")
    if _sha_file(Path(plan["rows_path"])) != plan["rows_raw_sha256"]:
        raise FactorizationError("factorization rows changed after preflight")
    if _sha_file(Path(plan["calibration_geometry_qa"]["path"])) != plan["calibration_geometry_qa"]["raw_sha256"]:
        raise FactorizationError("CAL geometry QA changed after preflight")
    assets, normalized_assets = mel._asset_registry(Path(str(plan["execution_assets"]["registry_path"])))
    if normalized_assets["registry_raw_sha256"] != plan["execution_assets_raw_sha256"]:
        raise FactorizationError("execution asset registry changed after preflight")
    runtime = mel._runtime_metadata()
    if runtime != plan["runtime"]:
        raise FactorizationError("runtime changed after factorization preflight")
    factor_output = Path(plan["factor_output_root"]).resolve(strict=True)
    outdir = Path(plan["output_dir"]).resolve()
    if factor_output != Path(args.factor_dir).resolve() or outdir != Path(args.outdir).resolve():
        raise FactorizationError("runtime output paths differ from the frozen factorization plan")
    control = plan_path.parent.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        raise FactorizationError(f"fresh decoder output directory must be empty: {outdir}")
    if control != plan_path.parent.resolve():
        raise FactorizationError("execution plan is outside its own control directory")
    if (control / "execution_results.jsonl").exists():
        raise FactorizationError("execution results already exist; use a fresh attempt")
    if (control / "execution_events.jsonl").exists():
        raise FactorizationError("execution events already exist; use a fresh attempt")
    outdir.mkdir(parents=True, exist_ok=True)
    import cv2
    import torch

    forward_flags = mel._configure_inference_flags(torch)
    # The frozen portable helper root is carried by the original mel plan.
    mel_plan = _read_json(Path(plan["mel_source_run"]["plan_path"]))
    portable_root = Path(mel_plan["portable_code_root"])
    helper = mel._wav2lip_import(Path(assets["wav2lip"]["root"]), portable_root)
    face = assets["primary_face"]
    face_frame = cv2.imread(face["path"], cv2.IMREAD_COLOR)
    if face_frame is None or face_frame.shape != (512, 512, 3):
        raise FactorizationError("frozen primary face image does not decode to 512x512")
    face_pixel_sha = hashlib.sha256(np.ascontiguousarray(face_frame).tobytes()).hexdigest()
    if face_pixel_sha != face.get("decoded_bgr_pixel_sha256"):
        raise FactorizationError("frozen face decoded pixels differ from asset registry")
    model, model_info = helper.load_model(assets["wav2lip"]["root"], assets["wav2lip"]["checkpoint_path"], device="cuda")
    if model_info.get("checkpoint_sha256") != assets["wav2lip"]["sha256"]:
        raise FactorizationError("loaded Wav2Lip checkpoint differs from sealed asset registry")
    events_path = control / "execution_events.jsonl"
    sync = mel._SyncNet(Path(assets["syncnet"]["root"]), Path(assets["syncnet"]["checkpoint_path"]), device="cuda", batch_size=20, events_path=events_path, ffmpeg=Path(assets["ffmpeg"]["path"]))
    _write_json(control / "execution_runtime.json", {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "runtime": runtime,
        "torch_flags_after_deterministic_override_before_forward": forward_flags,
        "deterministic_override": {"applied": True, "seed": 20260930, "cublas_workspace_config": ":4096:8"},
        "wav2lip_model": model_info,
        "syncnet": {"checkpoint_sha256": sync.checkpoint_sha256, "loaded_state_sha256": sync.loaded_state_sha256, "batch_size": 20},
        "face_registry_raw_sha256": assets["face_registry_raw_sha256"],
        "primary_face_file_sha256": face["sha256"],
        "primary_face_decoded_pixel_sha256": face_pixel_sha,
        "model_forward_performed_at_creation": False,
        "audio_encoder_forward_performed": False,
        "effects_read": False,
    })
    face_skips = mel._cache_face_skips(model, helper, face_frame, face["generation_box_xyxy"], torch=torch, device="cuda", events_path=events_path, sample_id="__shared_primary_face__")
    results_path = control / "execution_results.jsonl"
    results_path.touch()
    completed = failures = 0
    for row in plan["planned_samples"]:
        sample_id = row["id"]
        result: dict[str, Any] = {"id": sample_id, "speaker": row["speaker"], "split": "evaluation", "status": "running", "cells": {}, "model_forward_performed": True, "audio_encoder_forward_performed": False, "effects_read": False}
        try:
            if row["status"] != "prepared":
                raise FactorizationError("planned latent geometry row was not prepared")
            source = row["source_audio"]
            pcm = mel._read_raw_pcm(Path(source["source_wav_path"]), int(source["sample_count"]), source["source_wav_file_sha256"], source["source_pcm_array_sha256"])
            a_features, a_meta, a_events = sync.audio(pcm, sample_id=sample_id)
            a_record = mel._save_npy(outdir / "features" / sample_id / "A_raw.npy", a_features)
            a_record.update({"feature_meta": a_meta, "forward_events": a_events})
            result["shared_A"] = a_record
            mixed_path = Path(row["mixed_latents"]["path"])
            _sha_required(mixed_path, row["mixed_latents"]["file_sha256"], f"mixed latent archive {sample_id}")
            with np.load(mixed_path, allow_pickle=False) as archive:
                mixed = {name: np.asarray(archive[name]) for name in archive.files}
            z_sources = {}
            for condition in ("q00", "q11"):
                meta = row["endpoint_latents"][condition]
                path = Path(meta["path"])
                _sha_required(path, meta["file_sha256"], f"{condition} endpoint latent {sample_id}")
                z = np.load(path, allow_pickle=False)
                if z.dtype != np.float32 or list(z.shape) != meta["shape"] or mel._array_sha256(z) != meta["array_sha256"]:
                    raise FactorizationError(f"{condition} exact endpoint latent changed: {sample_id}")
                z_sources[condition] = z
            for condition in CONDITIONS:
                if condition == "q00":
                    latent = z_sources["q00"]
                    latent_meta = row["endpoint_latents"]["q00"]
                elif condition == "q11":
                    latent = z_sources["q11"]
                    latent_meta = row["endpoint_latents"]["q11"]
                else:
                    latent = mixed[condition]
                    expected_array = row["mixed_latents"]["arrays"][condition]
                    if latent.dtype != np.float32 or list(latent.shape) != expected_array["shape"] or mel._array_sha256(latent) != expected_array["array_sha256"]:
                        raise FactorizationError(f"mixed latent array changed: {sample_id}/{condition}")
                    latent_meta = {"path": str(mixed_path), "file_sha256": row["mixed_latents"]["file_sha256"], "array_sha256": expected_array["array_sha256"], "shape": expected_array["shape"], "dtype": expected_array["dtype"], "archive_key": condition}
                if len(latent) != row["latent_frames"] or not np.isfinite(latent).all():
                    raise FactorizationError(f"latent frame clock/finite check failed: {sample_id}/{condition}")
                video_path = outdir / "videos" / sample_id / f"{condition}.mp4"
                video_meta = mel._decode_to_video(model, helper, latent, face_skips, face_frame, face["generation_box_xyxy"], video_path, Path(assets["ffmpeg"]["path"]), torch=torch, device="cuda", batch_size=64, events_path=events_path, sample_id=sample_id, condition=condition)
                crop_path = outdir / "scored_video" / sample_id / f"{condition}.mkv"
                crop_meta = mel._crop_video(helper, video_path, face["score_box"], crop_path, Path(assets["ffmpeg"]["path"]))
                expected_frame_count = int(len(latent))
                for label, metadata in (("decoded video", video_meta), ("score crop", crop_meta)):
                    if int(metadata.get("frame_count", -1)) != expected_frame_count:
                        raise FactorizationError(f"{label} frame count differs from exact latent frame clock for {sample_id}/{condition}: {metadata.get('frame_count')} != {expected_frame_count}")
                v, v_meta, v_events = sync.visual(crop_path, sample_id=sample_id, condition=condition)
                if int(v_meta.get("frame_count", -1)) != expected_frame_count or int(v_meta.get("feature_count", -1)) != len(v) or len(v) != expected_frame_count - 5:
                    raise FactorizationError(f"SyncNet V frame/window count differs from exact decoded frame clock for {sample_id}/{condition}: {v_meta.get('frame_count')}/{len(v)} vs {expected_frame_count}")
                v_record = mel._save_npy(outdir / "features" / sample_id / f"V_{condition}.npy", v)
                v_record.update({"feature_meta": v_meta, "forward_events": v_events})
                bridge = None
                if condition in ("q00", "q11"):
                    baseline = row["baseline_endpoint_outputs"][condition]
                    baseline_video = baseline["video"]
                    baseline_v = baseline["V"]
                    _sha_required(Path(baseline_video["path"]), baseline_video["sha256"], f"mel baseline {condition} video {sample_id}")
                    _sha_required(Path(baseline_v["path"]), baseline_v["file_sha256"], f"mel baseline {condition} V feature {sample_id}")
                    bridge = {
                        "video_pixel_stream": _video_pixel_bridge(baseline_video["path"], video_path),
                        "syncnet_v_array": _array_bridge(baseline_v["path"], v),
                        "source_video_file_sha256": baseline_video["sha256"],
                        "source_v_array_sha256": baseline_v["array_sha256"],
                    }
                result["cells"][condition] = {
                    "status": "generated_scored",
                    "latent_input": latent_meta,
                    "video": video_meta,
                    "scored_video": crop_meta,
                    "V": v_record,
                    "frame_clock_check": {
                        "latent_frames": expected_frame_count,
                        "decoded_video_frames": int(video_meta["frame_count"]),
                        "score_crop_frames": int(crop_meta["frame_count"]),
                        "syncnet_v_decoded_frames": int(v_meta["frame_count"]),
                        "syncnet_v_feature_windows": int(len(v)),
                        "expected_syncnet_v_feature_windows": expected_frame_count - 5,
                        "passed": True,
                    },
                    "baseline_bridge": bridge,
                }
            v_features = {condition: np.load(Path(result["cells"][condition]["V"]["path"]), allow_pickle=False) for condition in CONDITIONS}
            result["scores"] = mel._score_clip(outdir, sample_id, v_features, a_features, torch)
            result["status"] = "complete"
            completed += 1
        except Exception as exc:
            result["status"] = "failed"
            result["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
            failures += 1
        _append_jsonl(results_path, result)
        print(json.dumps({"id": sample_id, "status": result["status"], "completed_rows": completed, "planned_rows": EXPECTED_EVAL_ROWS}, ensure_ascii=False, sort_keys=True), flush=True)
    rows = _jsonl_read(results_path)
    if len(rows) != EXPECTED_EVAL_ROWS:
        raise FactorizationError(f"run did not retain all 80 intended rows: {len(rows)}")
    artifact_ledger = {path.relative_to(outdir).as_posix(): _sha_file(path) for path in sorted(p for p in outdir.rglob("*") if p.is_file())}
    run_manifest = {
        "schema_version": 1,
        "experiment_id": plan["experiment_id"],
        "consumer_id": plan["consumer_id"],
        "status": "complete" if all(row.get("status") == "complete" for row in rows) else "complete_with_failures",
        "finished_at_utc": _utc_now(),
        "execution_plan_path": str(plan_path),
        "execution_plan_raw_sha256": _sha_file(plan_path),
        "execution_plan_payload_sha256": plan["payload_sha256"],
        "preflight_report_path": str((control / "preflight_report.json").resolve()),
        "preflight_report_sha256": _sha_file(control / "preflight_report.json"),
        "runner_code_sha256": plan["code"]["sha256"],
        "mel_runner_sha256": plan["mel_runner"]["sha256"],
        "execution_runtime_path": str((control / "execution_runtime.json").resolve()),
        "execution_runtime_raw_sha256": _sha_file(control / "execution_runtime.json"),
        "calibration_geometry_qa": plan["calibration_geometry_qa"],
        "mel_source_run": plan["mel_source_run"],
        "execution_assets_raw_sha256": plan["execution_assets_raw_sha256"],
        "runtime": plan["runtime"],
        "runtime_record": {"model": model_info, "syncnet_checkpoint_sha256": sync.checkpoint_sha256, "syncnet_state_sha256": sync.loaded_state_sha256, "face_pixel_sha256": face_pixel_sha, "cudnn_benchmark": False},
        "planned_rows": EXPECTED_EVAL_ROWS,
        "completed_rows": completed,
        "planned_cells": EXPECTED_EVAL_ROWS * 4,
        "failed_rows": failures,
        "rows_path": str(results_path.resolve()),
        "rows_raw_sha256": _sha_file(results_path),
        "execution_events_path": str(events_path.resolve()),
        "execution_events_raw_sha256": _sha_file(events_path),
        "execution_event_count": sum(1 for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()),
        "artifact_root": str(outdir.resolve()),
        "artifact_count": len(artifact_ledger),
        "artifact_ledger": artifact_ledger,
        "model_forward_performed": True,
        "audio_encoder_forward_performed": False,
        "mel_frontend_recomputed": False,
        "effects_read": False,
    }
    run_manifest["payload_sha256"] = _payload_sha(run_manifest)
    _write_json(control / "run_manifest.json", run_manifest)
    print(json.dumps({"status": run_manifest["status"], "rows": completed, "cells": EXPECTED_EVAL_ROWS * 4, "failures": failures, "run_manifest": str((control / "run_manifest.json").resolve()), "run_manifest_sha256": _sha_file(control / "run_manifest.json"), "artifact_count": len(artifact_ledger)}, ensure_ascii=False, sort_keys=True), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cal = commands.add_parser("cal-qa", help="compute no-forward CAL20 latent geometry from original feature NPZs")
    cal.add_argument("--calibration-dir", required=True, type=Path)
    cal.add_argument("--spec", required=True, type=Path)
    cal.add_argument("--output", required=True, type=Path)
    cal.set_defaults(func=calibration_qa)
    prep = commands.add_parser("prepare-eval", help="validate mel q00/q11, factorize latents, and freeze no-forward GPU plan")
    prep.add_argument("--mel-runner", required=True, type=Path)
    prep.add_argument("--mel-control", required=True, type=Path)
    prep.add_argument("--calibration-qa", required=True, type=Path)
    prep.add_argument("--spec", required=True, type=Path)
    prep.add_argument("--factor-dir", required=True, type=Path)
    prep.add_argument("--outdir", required=True, type=Path)
    prep.add_argument("--control-dir", required=True, type=Path)
    prep.set_defaults(func=prepare_eval)
    execute = commands.add_parser("run", help="freshly decode and score the four sealed latent factor cells")
    execute.add_argument("--plan", required=True, type=Path)
    execute.add_argument("--factor-dir", required=True, type=Path)
    execute.add_argument("--outdir", required=True, type=Path)
    execute.set_defaults(func=run)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:2000]}, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
