#!/usr/bin/env python3
"""Summarize three-method visual geometry on a shared tracked-frame mask.

This is a read-only consumer for the derived visual/speech-motion join. It
rechecks the joined plan/result/visual seals, uses the intersection of all
methods' valid frames (and adjacent transitions for velocity), and reports
descriptive per-context, per-speaker, per-model, and equal-model summaries.
It does not render, rescore SyncNet, retime, infer quality from larger motion,
or calculate confidence intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PLAN_SCHEMA = "cross-tfg-audio-eval-v1"
VISUAL_SCHEMA = "cross-tfg-visual-motion-guard-v1"
OUTPUT_SCHEMA = "cross-tfg-common-support-geometry-summary-v1"
GROUP_FIELDS = ("split", "sample_id", "speaker_id", "model_id", "face_id", "seed")
FRAME_ARRAYS = {
    "frame_index_int32": (np.dtype("int32"), "frame"),
    "timestamp_num_int64": (np.dtype("int64"), "rational_numerator"),
    "timestamp_den_int64": (np.dtype("int64"), "rational_denominator"),
    "tracking_valid_bool": (np.dtype("bool"), "valid_mask"),
    "mouth_aperture_eye_normalized_float64": (np.dtype("float64"), "ipd"),
    "mouth_width_eye_normalized_float64": (np.dtype("float64"), "ipd"),
    "mouth_height_eye_normalized_float64": (np.dtype("float64"), "ipd"),
    "mouth_area_eye_normalized_squared_float64": (np.dtype("float64"), "ipd_squared"),
    "aperture_velocity_eye_normalized_per_s_float64": (np.dtype("float64"), "ipd_per_s"),
    "aperture_velocity_valid_bool": (np.dtype("bool"), "transition_valid_mask"),
    "aperture_acceleration_eye_normalized_per_s2_float64": (np.dtype("float64"), "ipd_per_s2"),
    "aperture_acceleration_valid_bool": (np.dtype("bool"), "second_transition_valid_mask"),
    "aperture_jerk_eye_normalized_per_s3_float64": (np.dtype("float64"), "ipd_per_s3"),
    "aperture_jerk_valid_bool": (np.dtype("bool"), "third_transition_valid_mask"),
}
METHOD_ORDER = ("identity", "rms_fixed")
METRICS = (
    "aperture_mean_ipd", "aperture_median_ipd", "aperture_p05_ipd", "aperture_p95_ipd",
    "aperture_sd_ipd", "mouth_width_median_ipd", "mouth_height_median_ipd",
    "mouth_area_median_ipd2", "aperture_velocity_mean_ipd_per_s",
    "aperture_velocity_mean_abs_ipd_per_s", "aperture_velocity_rms_ipd_per_s",
    "aperture_velocity_p05_ipd_per_s", "aperture_velocity_p95_ipd_per_s",
    "aperture_velocity_closing_fraction", "aperture_acceleration_abs_median_ipd_per_s2",
    "aperture_jerk_abs_median_ipd_per_s3",
)


class SummaryError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_sha(value: Mapping[str, Any], key: str) -> str:
    return sha_bytes(canonical({name: item for name, item in value.items() if name != key}))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SummaryError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise SummaryError(f"expected JSON object at {path}:{line_no}")
                rows.append(value)
    return rows


def group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in GROUP_FIELDS)


def _finite_mean(values: np.ndarray) -> float | None:
    return float(np.mean(values, dtype=np.float64)) if values.size else None


def _finite_median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if values.size else None


def _finite_quantile(values: np.ndarray, q: float) -> float | None:
    return float(np.quantile(values, q)) if values.size else None


def _geometry_metrics(arrays_by_method: Mapping[str, Mapping[str, np.ndarray]]) -> tuple[dict[str, Any], dict[str, Any]]:
    methods = list(arrays_by_method)
    frame_count = len(arrays_by_method[methods[0]]["frame_index_int32"])
    common_frames = np.ones(frame_count, dtype=bool)
    for arrays in arrays_by_method.values():
        common_frames &= arrays["tracking_valid_bool"]
    transition_count = max(0, frame_count - 1)
    common_velocity = np.ones(transition_count, dtype=bool)
    for arrays in arrays_by_method.values():
        common_velocity &= arrays["aperture_velocity_valid_bool"]
    acceleration_count = max(0, frame_count - 2)
    common_acceleration = np.ones(acceleration_count, dtype=bool)
    for arrays in arrays_by_method.values():
        common_acceleration &= arrays["aperture_acceleration_valid_bool"]
    jerk_count = max(0, frame_count - 3)
    common_jerk = np.ones(jerk_count, dtype=bool)
    for arrays in arrays_by_method.values():
        common_jerk &= arrays["aperture_jerk_valid_bool"]

    support = {
        "render_frames": frame_count,
        "common_valid_frames": int(common_frames.sum()),
        "common_valid_frame_fraction": float(common_frames.mean()) if frame_count else 0.0,
        "common_valid_velocity_transitions": int(common_velocity.sum()),
        "common_valid_acceleration_transitions": int(common_acceleration.sum()),
        "common_valid_jerk_transitions": int(common_jerk.sum()),
        "common_frame_mask_sha256": sha_bytes(np.ascontiguousarray(common_frames, dtype=np.uint8).tobytes()),
        "common_velocity_mask_sha256": sha_bytes(np.ascontiguousarray(common_velocity, dtype=np.uint8).tobytes()),
        "common_acceleration_mask_sha256": sha_bytes(np.ascontiguousarray(common_acceleration, dtype=np.uint8).tobytes()),
        "common_jerk_mask_sha256": sha_bytes(np.ascontiguousarray(common_jerk, dtype=np.uint8).tobytes()),
    }
    values: dict[str, dict[str, float | None]] = {}
    for method, arrays in arrays_by_method.items():
        aperture = arrays["mouth_aperture_eye_normalized_float64"][common_frames]
        width = arrays["mouth_width_eye_normalized_float64"][common_frames]
        height = arrays["mouth_height_eye_normalized_float64"][common_frames]
        area = arrays["mouth_area_eye_normalized_squared_float64"][common_frames]
        velocity = arrays["aperture_velocity_eye_normalized_per_s_float64"][common_velocity]
        acceleration = arrays["aperture_acceleration_eye_normalized_per_s2_float64"][common_acceleration]
        jerk = arrays["aperture_jerk_eye_normalized_per_s3_float64"][common_jerk]
        values[method] = {
            "aperture_mean_ipd": _finite_mean(aperture),
            "aperture_median_ipd": _finite_median(aperture),
            "aperture_p05_ipd": _finite_quantile(aperture, 0.05),
            "aperture_p95_ipd": _finite_quantile(aperture, 0.95),
            "aperture_sd_ipd": float(np.std(aperture, dtype=np.float64)) if aperture.size else None,
            "mouth_width_median_ipd": _finite_median(width),
            "mouth_height_median_ipd": _finite_median(height),
            "mouth_area_median_ipd2": _finite_median(area),
            "aperture_velocity_mean_ipd_per_s": _finite_mean(velocity),
            "aperture_velocity_mean_abs_ipd_per_s": _finite_mean(np.abs(velocity)),
            "aperture_velocity_rms_ipd_per_s": float(np.sqrt(np.mean(velocity * velocity, dtype=np.float64))) if velocity.size else None,
            "aperture_velocity_p05_ipd_per_s": _finite_quantile(velocity, 0.05),
            "aperture_velocity_p95_ipd_per_s": _finite_quantile(velocity, 0.95),
            "aperture_velocity_closing_fraction": float(np.mean(velocity < 0)) if velocity.size else None,
            "aperture_acceleration_abs_median_ipd_per_s2": _finite_median(np.abs(acceleration)),
            "aperture_jerk_abs_median_ipd_per_s3": _finite_median(np.abs(jerk)),
        }
    return support, values


def _load_visual_arrays(visual_root: Path, visual_row: Mapping[str, Any], expected_fps: Fraction) -> dict[str, np.ndarray]:
    ref = visual_row.get("visual_npz")
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise SummaryError(f"{visual_row.get('row_id')}: visual NPZ reference is missing")
    path_ref = Path(ref["path"])
    if path_ref.is_absolute() or ".." in path_ref.parts:
        raise SummaryError(f"{visual_row.get('row_id')}: visual NPZ path escapes the visual output root")
    path = (visual_root / path_ref).resolve(strict=True)
    if sha_file(path) != ref.get("file_sha256"):
        raise SummaryError(f"{visual_row.get('row_id')}: visual NPZ raw hash mismatch")
    metadata = ref.get("arrays")
    if not isinstance(metadata, dict):
        raise SummaryError(f"{visual_row.get('row_id')}: visual NPZ array seals are missing")
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in FRAME_ARRAYS}
    except Exception as exc:
        raise SummaryError(f"{visual_row.get('row_id')}: cannot load visual arrays: {type(exc).__name__}") from exc
    frames = arrays["frame_index_int32"]
    if frames.ndim != 1 or not np.array_equal(frames, np.arange(len(frames), dtype=np.int32)):
        raise SummaryError(f"{visual_row.get('row_id')}: frame indices are not a contiguous source-clock sequence")
    n = len(frames)
    expected_shapes = {
        "frame_index_int32": (n,), "timestamp_num_int64": (n,), "timestamp_den_int64": (n,),
        "tracking_valid_bool": (n,), "mouth_aperture_eye_normalized_float64": (n,),
        "mouth_width_eye_normalized_float64": (n,), "mouth_height_eye_normalized_float64": (n,),
        "mouth_area_eye_normalized_squared_float64": (n,),
        "aperture_velocity_eye_normalized_per_s_float64": (max(0, n - 1),),
        "aperture_velocity_valid_bool": (max(0, n - 1),),
        "aperture_acceleration_eye_normalized_per_s2_float64": (max(0, n - 2),),
        "aperture_acceleration_valid_bool": (max(0, n - 2),),
        "aperture_jerk_eye_normalized_per_s3_float64": (max(0, n - 3),),
        "aperture_jerk_valid_bool": (max(0, n - 3),),
    }
    for name, array in arrays.items():
        expected_dtype = FRAME_ARRAYS[name][0]
        item = metadata.get(name)
        if array.dtype != expected_dtype or array.shape != expected_shapes[name] or not isinstance(item, dict):
            raise SummaryError(f"{visual_row.get('row_id')}: {name} dtype/shape/seal mismatch")
        if item.get("dtype") not in (array.dtype.str, array.dtype.name) or item.get("shape") != list(array.shape):
            raise SummaryError(f"{visual_row.get('row_id')}: {name} declared dtype/shape mismatch")
        if item.get("array_sha256") != sha_bytes(np.ascontiguousarray(array).tobytes(order="C")):
            raise SummaryError(f"{visual_row.get('row_id')}: {name} array digest mismatch")
    nums, dens = arrays["timestamp_num_int64"], arrays["timestamp_den_int64"]
    if np.any(dens <= 0):
        raise SummaryError(f"{visual_row.get('row_id')}: timestamp denominator must be positive")
    for index, (num, den) in enumerate(zip(nums.tolist(), dens.tolist(), strict=True)):
        if Fraction(int(num), int(den)) != Fraction(index, 1) / expected_fps:
            raise SummaryError(f"{visual_row.get('row_id')}: timestamp is not on the expected rational source clock")
    for name, array in arrays.items():
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise SummaryError(f"{visual_row.get('row_id')}: {name} contains nonfinite values")
    return arrays


def _speaker_equal_summary(
    records: Sequence[Mapping[str, Any]],
    method_ids: Sequence[str],
    model_ids: Sequence[str],
    expected_speakers_by_model: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    values_by_method_model_speaker: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    delta_by_method_model_speaker: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row.get("status") != "PASS":
            continue
        model, speaker = str(row["model_id"]), str(row["speaker_id"])
        for method in method_ids:
            for metric, value in row.get("metrics_by_method", {}).get(method, {}).items():
                if metric in METRICS and value is not None:
                    values_by_method_model_speaker[(method, model, speaker)][metric].append(float(value))
            if method == "identity":
                continue
            for metric, value in row.get("deltas_vs_identity", {}).get(method, {}).items():
                if metric in METRICS and value is not None:
                    delta_by_method_model_speaker[(method, model, speaker)][metric].append(float(value))

    model_rows: list[dict[str, Any]] = []
    for model in model_ids:
        expected_speakers = list(expected_speakers_by_model.get(model, []))
        for method in method_ids:
            speaker_values: dict[str, dict[str, float]] = {}
            context_counts: dict[str, dict[str, int]] = {}
            for speaker in expected_speakers:
                metric_values = values_by_method_model_speaker.get((method, model, speaker), {})
                means = {metric: float(np.mean(metric_values[metric], dtype=np.float64)) for metric in METRICS if metric_values.get(metric)}
                if means:
                    speaker_values[speaker] = means
                    context_counts[speaker] = {metric: len(metric_values.get(metric, [])) for metric in METRICS}
            summary_values = {
                metric: float(np.mean([speaker_values[s][metric] for s in speaker_values if metric in speaker_values[s]], dtype=np.float64))
                if any(metric in speaker_values[s] for s in speaker_values) else None
                for metric in METRICS
            }
            model_rows.append({
                "model_id": model, "method_id": method, "expected_speakers": len(expected_speakers),
                "speakers_with_geometry": len(speaker_values), "contexts_by_speaker_by_metric": context_counts,
                "equal_speaker_mean_of_context_means": summary_values,
            })
            if method == "identity":
                continue
            delta_speakers: dict[str, dict[str, float]] = {}
            delta_context_counts: dict[str, dict[str, int]] = {}
            for speaker in expected_speakers:
                metric_values = delta_by_method_model_speaker.get((method, model, speaker), {})
                means = {metric: float(np.mean(metric_values[metric], dtype=np.float64)) for metric in METRICS if metric_values.get(metric)}
                if means:
                    delta_speakers[speaker] = means
                    delta_context_counts[speaker] = {metric: len(metric_values.get(metric, [])) for metric in METRICS}
            delta_means = {
                metric: float(np.mean([delta_speakers[s][metric] for s in delta_speakers if metric in delta_speakers[s]], dtype=np.float64))
                if any(metric in delta_speakers[s] for s in delta_speakers) else None
                for metric in METRICS
            }
            model_rows.append({
                "model_id": model, "method_id": method, "contrast": "paired_delta_vs_identity",
                "expected_speakers": len(expected_speakers), "speakers_with_geometry": len(delta_speakers),
                "contexts_by_speaker_by_metric": delta_context_counts,
                "equal_speaker_mean_of_context_mean_deltas": delta_means,
            })

    equal_model: list[dict[str, Any]] = []
    for method in method_ids:
        for contrast in ("absolute", "paired_delta_vs_identity"):
            if contrast == "paired_delta_vs_identity" and method == "identity":
                continue
            by_model: dict[str, dict[str, float | None]] = {}
            for row in model_rows:
                if row["method_id"] != method:
                    continue
                is_delta = row.get("contrast") == "paired_delta_vs_identity"
                if is_delta != (contrast == "paired_delta_vs_identity"):
                    continue
                by_model[str(row["model_id"])] = row["equal_speaker_mean_of_context_mean_deltas" if is_delta else "equal_speaker_mean_of_context_means"]
            means = {}
            for metric in METRICS:
                members = [by_model[model][metric] for model in model_ids if model in by_model and by_model[model].get(metric) is not None]
                means[metric] = float(np.mean(members, dtype=np.float64)) if members else None
            equal_model.append({
                "method_id": method, "contrast": contrast,
                "expected_models": len(model_ids), "models_with_values_by_metric": {
                    metric: sum(model in by_model and by_model[model].get(metric) is not None for model in model_ids)
                    for metric in METRICS
                },
                "equal_model_mean_of_equal_speaker_means": means,
            })
    return {"model_method_speaker_equal": model_rows, "equal_model": equal_model}


def run(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan).resolve(strict=True)
    results_path = Path(args.results).resolve(strict=True)
    result_manifest_path = Path(args.result_manifest).resolve(strict=True)
    visual_root = Path(args.visual_root).resolve(strict=True)
    out = Path(args.out).resolve()
    if out.exists():
        raise SummaryError(f"fresh output directory already exists: {out}")
    plan, result_manifest = read_json(plan_path), read_json(result_manifest_path)
    plan_sha, results_sha, result_manifest_sha = sha_file(plan_path), sha_file(results_path), sha_file(result_manifest_path)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "FROZEN" or plan.get("plan_sha256") != payload_sha(plan, "plan_sha256"):
        raise SummaryError("joined plan schema/fingerprint is invalid")
    if result_manifest.get("schema_version") != PLAN_SCHEMA or result_manifest.get("run_id") != plan.get("run_id") or result_manifest.get("manifest_sha256") != payload_sha(result_manifest, "manifest_sha256"):
        raise SummaryError("joined result manifest schema/fingerprint is invalid")
    if (result_manifest.get("plan_raw_sha256") != plan_sha or result_manifest.get("plan_sha256") != plan.get("plan_sha256") or
            result_manifest.get("results_jsonl_sha256") != results_sha):
        raise SummaryError("joined result manifest does not bind the requested plan/results")
    visual_manifest_path, visual_audit_path = visual_root / "visual_motion_manifest.json", visual_root / "audit.json"
    visual_results_path = visual_root / "visual_motion_results.jsonl"
    visual_manifest, visual_audit = read_json(visual_manifest_path), read_json(visual_audit_path)
    visual_manifest_sha, visual_audit_sha, visual_results_sha = sha_file(visual_manifest_path), sha_file(visual_audit_path), sha_file(visual_results_path)
    if visual_manifest.get("schema_version") != VISUAL_SCHEMA or visual_manifest.get("manifest_sha256") != payload_sha(visual_manifest, "manifest_sha256"):
        raise SummaryError("joined visual-motion manifest schema/fingerprint is invalid")
    if visual_audit.get("status") != visual_manifest.get("status"):
        raise SummaryError("joined visual-motion audit/manifest statuses differ")
    if (visual_audit.get("plan_raw_sha256") != plan_sha or visual_audit.get("results_raw_sha256") != results_sha or
            visual_audit.get("result_manifest_raw_sha256") != result_manifest_sha):
        raise SummaryError("joined visual-motion audit does not bind the exact source plan/results")
    files = visual_manifest.get("files")
    if not isinstance(files, list):
        raise SummaryError("visual manifest has no file inventory")
    inventory = {str(item.get("path")): item for item in files if isinstance(item, dict)}
    if len(inventory) != len(files):
        raise SummaryError("visual manifest has duplicate/malformed inventory paths")
    for rel, item in inventory.items():
        relative = Path(rel)
        if relative.is_absolute() or ".." in relative.parts:
            raise SummaryError(f"visual inventory path escapes its root: {rel}")
        path = (visual_root / relative).resolve(strict=True)
        if path.stat().st_size != item.get("bytes") or sha_file(path) != item.get("sha256"):
            raise SummaryError(f"visual output inventory mismatch: {rel}")
    for rel, path in (("audit.json", visual_audit_path), ("visual_motion_results.jsonl", visual_results_path)):
        item = inventory.get(rel)
        if not isinstance(item, dict) or item.get("bytes") != path.stat().st_size or item.get("sha256") != sha_file(path):
            raise SummaryError(f"visual manifest does not seal {rel}")

    planned = plan.get("rows")
    result_rows, visual_rows = read_jsonl(results_path), read_jsonl(visual_results_path)
    if not isinstance(planned, list) or not (len(planned) == len(result_rows) == len(visual_rows)):
        raise SummaryError("joined plan/result/visual row counts differ")
    method_ids = [str(item.get("method_id")) for item in plan.get("methods", []) if isinstance(item, dict)]
    if len(method_ids) != 3 or len(set(method_ids)) != 3 or "identity" not in method_ids or "rms_fixed" not in method_ids:
        raise SummaryError(f"expected identity, RMS, and one candidate method; found {method_ids}")
    candidate_method = next(method for method in method_ids if method not in METHOD_ORDER)
    models = [str(item.get("model_id")) for item in plan.get("models", []) if isinstance(item, dict)]
    if len(models) != 3 or len(set(models)) != 3:
        raise SummaryError(f"expected exactly three unique frontends; found {models}")
    syncnet = plan.get("protocol", {}).get("syncnet", {})
    fps = Fraction(int(syncnet.get("fps_num", 25)), int(syncnet.get("fps_den", 1)))
    if fps != Fraction(25, 1):
        raise SummaryError(f"expected the frozen 25 fps source clock, got {fps}")
    for index, (p, r, v) in enumerate(zip(planned, result_rows, visual_rows, strict=True)):
        for field in ("row_id", "split", "sample_id", "speaker_id", "model_id", "face_id", "seed", "method_id", "method_config_sha256", "checkpoint_sha256"):
            if p.get(field) != r.get(field) or p.get(field) != v.get(field):
                raise SummaryError(f"row identity mismatch at joined row {index}: {field}")
        if r.get("status") not in ("PASS", "FAILED", "INCOMPLETE") or v.get("status") not in ("PASS", "FAILED", "INCOMPLETE"):
            raise SummaryError(f"invalid producer/visual state at {p.get('row_id')}")
        if r.get("status") == "PASS":
            expected_pcm = p.get("natural_pcm", {}).get("pcm_sha256")
            if (r.get("natural_pcm_sha256") != expected_pcm or r.get("score_audio_sha256") != expected_pcm or
                    r.get("score", {}).get("score_audio_pcm_sha256") != expected_pcm):
                raise SummaryError(f"SyncNet did not score original natural PCM at {p.get('row_id')}")
            if v.get("source_face_sha256") != p.get("source_face", {}).get("file_sha256"):
                raise SummaryError(f"visual row is not bound to the planned face at {p.get('row_id')}")
            if v.get("video", {}).get("file_sha256") != r.get("render", {}).get("silent_video_sha256"):
                raise SummaryError(f"visual row is not bound to the exact rendered video at {p.get('row_id')}")

    result_by_id = {str(row["row_id"]): row for row in result_rows}
    visual_by_id = {str(row["row_id"]): row for row in visual_rows}
    planned_by_group: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in planned:
        key = group_key(row)
        method = str(row["method_id"])
        if method in planned_by_group[key]:
            raise SummaryError(f"duplicate method in paired context: {key}/{method}")
        planned_by_group[key][method] = row
    expected_groups = len(planned_by_group)
    if expected_groups != 180 or any(set(rows) != set(method_ids) for rows in planned_by_group.values()):
        raise SummaryError(f"joined context/method denominator is not complete: {expected_groups}")

    source_status = Counter(str(row.get("status")) for row in result_rows)
    visual_status = Counter(str(row.get("status")) for row in visual_rows)
    context_records: list[dict[str, Any]] = []
    failure_reasons: Counter[str] = Counter()
    for key, rows_by_method in planned_by_group.items():
        per_result = [result_by_id[str(rows_by_method[method]["row_id"])] for method in method_ids]
        per_visual = [visual_by_id[str(rows_by_method[method]["row_id"])] for method in method_ids]
        states = [(row.get("status"), vis.get("status")) for row, vis in zip(per_result, per_visual, strict=True)]
        if any(result_status != "PASS" or visual_state != "PASS" for result_status, visual_state in states):
            reason = "SOURCE_OR_VISUAL_NONPASS"
            failure_reasons[reason] += 1
            context_records.append({"context": list(key), "status": "INCOMPLETE", "reason": reason,
                                    "method_states": {method: {"result": result_by_id[str(rows_by_method[method]["row_id"])].get("status"),
                                                                "visual": visual_by_id[str(rows_by_method[method]["row_id"])].get("status")}
                                                       for method in method_ids}})
            continue
        natural_rows = [rows_by_method[m].get("natural_pcm", {}) for m in method_ids]
        score_rows = [rows_by_method[m].get("score_audio", {}) for m in method_ids]
        natural_sig = {tuple(ref.get(field) for field in ("file_sha256", "pcm_sha256", "sample_count", "sample_rate_hz", "channels", "subtype")) for ref in natural_rows}
        score_sig = {tuple(ref.get(field) for field in ("file_sha256", "pcm_sha256", "sample_count", "sample_rate_hz", "channels", "subtype")) for ref in score_rows}
        face_hashes = {rows_by_method[m].get("source_face", {}).get("file_sha256") for m in method_ids}
        video_hashes = {rows_by_method[m].get("source_video", {}).get("file_sha256") for m in method_ids}
        if len(natural_sig) != 1 or natural_sig != score_sig or len(face_hashes) != 1 or len(video_hashes) != 1:
            raise SummaryError(f"paired methods differ in natural PCM, score audio, portrait, or source video: {key}")
        loaded = {method: _load_visual_arrays(visual_root, visual_by_id[str(rows_by_method[method]["row_id"])], fps)
                  for method in method_ids}
        frame_sequences = {tuple(a["frame_index_int32"].tolist()) for a in loaded.values()}
        timestamp_sequences = {
            tuple(Fraction(int(num), int(den)) for num, den in zip(a["timestamp_num_int64"].tolist(),
                                                                    a["timestamp_den_int64"].tolist(), strict=True))
            for a in loaded.values()
        }
        if len(frame_sequences) != 1 or len(timestamp_sequences) != 1:
            raise SummaryError(f"paired methods do not share an exact rational frame clock: {key}")
        support, values = _geometry_metrics(loaded)
        if support["common_valid_frames"] == 0:
            reason = "NO_COMMON_VALID_TRACKED_FRAME"
            failure_reasons[reason] += 1
            context_records.append({"context": list(key), "status": "INCOMPLETE", "reason": reason,
                                    "support": support, "metrics_by_method": values})
            continue
        deltas = {}
        for method in method_ids:
            if method == "identity":
                continue
            deltas[method] = {
                metric: (None if values[method][metric] is None or values["identity"][metric] is None
                         else float(values[method][metric] - values["identity"][metric]))
                for metric in METRICS
            }
        context_records.append({"context": list(key), "status": "PASS", "common_support": support,
                                "metrics_by_method": values, "deltas_vs_identity": deltas,
                                "interpretation": "descriptive common-tracking-frame geometry; larger aperture or movement is not a quality criterion"})

    expected_speakers_by_model: dict[str, list[str]] = {}
    for model in models:
        expected_speakers_by_model[model] = sorted({str(row["speaker_id"]) for row in planned if str(row["model_id"]) == model})
    weighted = _speaker_equal_summary(context_records, method_ids, models, expected_speakers_by_model)

    out.mkdir(parents=True)
    contexts_path = out / "common_support_geometry_by_context.jsonl"
    with contexts_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in context_records:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
    summary = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "PASS" if len(context_records) == expected_groups and not failure_reasons else "INCOMPLETE",
        "source_run_id": plan.get("run_id"),
        "source_plan_raw_sha256": plan_sha, "source_results_raw_sha256": results_sha,
        "source_result_manifest_raw_sha256": result_manifest_sha,
        "source_visual_manifest_raw_sha256": visual_manifest_sha,
        "source_visual_audit_raw_sha256": visual_audit_sha,
        "source_visual_results_raw_sha256": visual_results_sha,
        "planned_rows": len(planned), "result_status_counts": dict(source_status),
        "visual_status_counts": dict(visual_status), "planned_context_groups": expected_groups,
        "complete_common_support_context_groups": sum(row.get("status") == "PASS" for row in context_records),
        "incomplete_context_groups": sum(row.get("status") != "PASS" for row in context_records),
        "incomplete_reasons": dict(failure_reasons), "expected_models": models,
        "expected_speakers_by_model": expected_speakers_by_model,
        "method_ids": method_ids, "candidate_method_id": candidate_method,
        "frame_rate": {"numerator": fps.numerator, "denominator": fps.denominator},
        "speaker_equal_model_summary": weighted,
        "aggregation": "Per-context metrics use the exact intersection of all three methods' valid visual frames/transitions. Context means are averaged within speaker, speakers equally within model, then models equally. No CI or inferential test.",
        "scope": {"development_only": True, "confirmation_accessed": False,
                  "rendering_or_syncnet_rescore": False, "effects_selected_from_visual_metrics": False},
    }
    summary_path = out / "common_support_geometry_summary.json"
    summary_path.write_bytes(canonical(summary) + b"\n")
    files = []
    for path in (contexts_path, summary_path):
        files.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha_file(path)})
    manifest = {"schema_version": OUTPUT_SCHEMA, "status": summary["status"], "run_id": plan.get("run_id"),
                "planned_rows": len(planned), "context_groups": expected_groups, "files": files}
    manifest["manifest_sha256"] = payload_sha(manifest, "manifest_sha256")
    manifest_path = out / "manifest.json"
    manifest_path.write_bytes(canonical(manifest) + b"\n")
    print(json.dumps({"status": summary["status"], "planned_rows": len(planned),
                      "complete_context_groups": summary["complete_common_support_context_groups"],
                      "incomplete_context_groups": summary["incomplete_context_groups"],
                      "summary_raw_sha256": sha_file(summary_path), "manifest_raw_sha256": sha_file(manifest_path)},
                     sort_keys=True, separators=(",", ":")))
    return 0 if summary["status"] == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--result-manifest", required=True)
    parser.add_argument("--visual-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"common_support_geometry_failed:{type(exc).__name__}:{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
