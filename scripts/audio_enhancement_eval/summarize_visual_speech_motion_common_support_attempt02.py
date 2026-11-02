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
    expected_samples_by_model_speaker: Mapping[tuple[str, str], Sequence[str]],
    expected_seeds_by_model: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    by_unit: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    for row in records:
        unit = (str(row["model_id"]), str(row["speaker_id"]), str(row["sample_id"]), int(row["seed"]))
        if unit in by_unit:
            raise SummaryError(f"duplicate model/speaker/clip/seed context: {unit}")
        by_unit[unit] = row

    def _method_context_value(
        method: str, model: str, speaker: str, sample: str, seed: int, metric: str, *, delta: bool
    ) -> float | None:
        context = by_unit.get((model, speaker, sample, seed))
        if context is None or context.get("status") != "PASS":
            return None
        source = context.get("deltas_vs_identity", {}) if delta else context.get("metrics_by_method", {})
        value = source.get(method, {}).get(metric)
        return None if value is None else float(value)

    model_rows: list[dict[str, Any]] = []
    model_metric_values: dict[tuple[str, str, str], float | None] = {}
    model_metric_deltas: dict[tuple[str, str, str], float | None] = {}
    for row in records:
        if not all(row.get(field) is not None for field in ("split", "sample_id", "speaker_id", "model_id", "face_id", "seed")):
            raise SummaryError("context row lacks explicit grouping fields for seed→clip→speaker aggregation")
    for model in model_ids:
        expected_speakers = list(expected_speakers_by_model.get(model, []))
        expected_seeds = [int(seed) for seed in expected_seeds_by_model.get(model, [])]
        if not expected_speakers or not expected_seeds:
            raise SummaryError(f"model has no frozen speaker/seed roster: {model}")
        for method in method_ids:
            for is_delta in (False, True):
                if is_delta and method == "identity":
                    continue
                speaker_metrics: dict[str, dict[str, float]] = defaultdict(dict)
                seed_cell_counts: dict[str, dict[str, int]] = defaultdict(dict)
                clip_counts: dict[str, dict[str, int]] = defaultdict(dict)
                for metric in METRICS:
                    complete_speaker_values: list[float] = []
                    expected_seed_cells = 0
                    observed_seed_cells = 0
                    complete_clips = 0
                    expected_clips = 0
                    complete_speakers = 0
                    for speaker in expected_speakers:
                        samples = list(expected_samples_by_model_speaker.get((model, speaker), []))
                        if not samples:
                            raise SummaryError(f"speaker has no planned clips: {model}/{speaker}")
                        clip_means: list[float] = []
                        speaker_complete = True
                        speaker_observed_clips = 0
                        for sample in samples:
                            values = [
                                _method_context_value(method, model, speaker, sample, seed, metric, delta=is_delta)
                                for seed in expected_seeds
                            ]
                            observed = [value for value in values if value is not None]
                            expected_seed_cells += len(expected_seeds)
                            observed_seed_cells += len(observed)
                            expected_clips += 1
                            if len(observed) == len(expected_seeds):
                                clip_means.append(float(np.mean(observed, dtype=np.float64)))
                                complete_clips += 1
                                speaker_observed_clips += 1
                            else:
                                speaker_complete = False
                        clip_counts[speaker][metric] = speaker_observed_clips
                        seed_cell_counts[speaker][metric] = sum(
                            _method_context_value(method, model, speaker, sample, seed, metric, delta=is_delta) is not None
                            for sample in samples for seed in expected_seeds
                        )
                        if speaker_complete and len(clip_means) == len(samples):
                            speaker_metrics[speaker][metric] = float(np.mean(clip_means, dtype=np.float64))
                            complete_speaker_values.append(speaker_metrics[speaker][metric])
                            complete_speakers += 1
                    strict_mean = (float(np.mean(complete_speaker_values, dtype=np.float64))
                                   if complete_speakers == len(expected_speakers) else None)
                    available_mean = (float(np.mean(complete_speaker_values, dtype=np.float64))
                                      if complete_speaker_values else None)
                    target = model_metric_deltas if is_delta else model_metric_values
                    target[(method, model, metric)] = strict_mean
                    if is_delta:
                        summary_key = "equal_speaker_mean_of_clip_mean_deltas"
                    else:
                        summary_key = "equal_speaker_mean_of_clip_means"
                    model_rows.append({
                        "model_id": model, "method_id": method,
                        **({"contrast": "paired_delta_vs_identity"} if is_delta else {}),
                        "metric": metric,
                        "expected_speakers": len(expected_speakers),
                        "complete_speakers": complete_speakers,
                        "missing_speakers": len(expected_speakers) - complete_speakers,
                        "expected_clips": expected_clips,
                        "complete_clips": complete_clips,
                        "missing_clips": expected_clips - complete_clips,
                        "expected_seed_cells": expected_seed_cells,
                        "observed_seed_cells": observed_seed_cells,
                        "missing_seed_cells": expected_seed_cells - observed_seed_cells,
                        "expected_seeds_per_clip": len(expected_seeds),
                        summary_key: strict_mean,
                        "available_complete_speaker_mean": available_mean,
                        "aggregation_order": "seed_mean_within_clip_then_equal_clip_mean_within_speaker_then_equal_speaker_mean_within_model",
                    })

    equal_model: list[dict[str, Any]] = []
    for method in method_ids:
        for contrast in ("absolute", "paired_delta_vs_identity"):
            if contrast == "paired_delta_vs_identity" and method == "identity":
                continue
            for metric in METRICS:
                source = model_metric_deltas if contrast == "paired_delta_vs_identity" else model_metric_values
                available = [source.get((method, model, metric)) for model in model_ids]
                members = [float(value) for value in available if value is not None]
                strict = float(np.mean(members, dtype=np.float64)) if len(members) == len(model_ids) else None
                equal_model.append({
                    "method_id": method, "contrast": contrast, "metric": metric,
                    "expected_models": len(model_ids), "complete_models": len(members),
                    "missing_models": len(model_ids) - len(members),
                    "equal_model_mean_of_complete_equal_speaker_means": strict,
                    "available_equal_model_mean": float(np.mean(members, dtype=np.float64)) if members else None,
                    "aggregation_order": "seed_mean_within_clip_then_equal_clip_mean_within_speaker_then_equal_speaker_mean_within_model_then_equal_model_mean",
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
        group_fields = dict(zip(GROUP_FIELDS, key, strict=True))
        per_result = [result_by_id[str(rows_by_method[method]["row_id"])] for method in method_ids]
        per_visual = [visual_by_id[str(rows_by_method[method]["row_id"])] for method in method_ids]
        states = [(row.get("status"), vis.get("status")) for row, vis in zip(per_result, per_visual, strict=True)]
        if any(result_status != "PASS" or visual_state != "PASS" for result_status, visual_state in states):
            reason = "SOURCE_OR_VISUAL_NONPASS"
            failure_reasons[reason] += 1
            context_records.append({**group_fields, "context": list(key), "status": "INCOMPLETE", "reason": reason,
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
            context_records.append({**group_fields, "context": list(key), "status": "INCOMPLETE", "reason": reason,
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
        context_records.append({**group_fields, "context": list(key), "status": "PASS", "common_support": support,
                                "metrics_by_method": values, "deltas_vs_identity": deltas,
                                "interpretation": "descriptive common-tracking-frame geometry; larger aperture or movement is not a quality criterion"})

    expected_speakers_by_model: dict[str, list[str]] = {}
    expected_samples_by_model_speaker: dict[tuple[str, str], list[str]] = {}
    expected_seeds_by_model: dict[str, list[int]] = {}
    for model in models:
        expected_speakers_by_model[model] = sorted({str(row["speaker_id"]) for row in planned if str(row["model_id"]) == model})
        expected_seeds_by_model[model] = sorted({int(row["seed"]) for row in planned if str(row["model_id"]) == model})
        for speaker in expected_speakers_by_model[model]:
            clips = sorted({str(row["sample_id"]) for row in planned
                            if str(row["model_id"]) == model and str(row["speaker_id"]) == speaker})
            expected_samples_by_model_speaker[(model, speaker)] = clips
            if len(clips) != 2:
                raise SummaryError(f"frozen visual plan requires exactly two clips per speaker: {model}/{speaker}")
        if len(expected_seeds_by_model[model]) != 3:
            raise SummaryError(f"frozen visual plan requires exactly three planned seeds per model: {model}")
    weighted = _speaker_equal_summary(context_records, method_ids, models, expected_speakers_by_model,
                                      expected_samples_by_model_speaker, expected_seeds_by_model)

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
        "expected_clips_by_model_speaker": {
            f"{model}/{speaker}": len(samples)
            for (model, speaker), samples in expected_samples_by_model_speaker.items()
        },
        "expected_seeds_by_model": expected_seeds_by_model,
        "method_ids": method_ids, "candidate_method_id": candidate_method,
        "frame_rate": {"numerator": fps.numerator, "denominator": fps.denominator},
        "speaker_equal_model_summary": weighted,
        "aggregation": "Per-context metrics use the exact intersection of all three methods' valid visual frames/transitions. For each metric, seed values are averaged within clip only when all three planned seeds are present; clip means are averaged within speaker only when both planned clips are complete; complete speaker means are averaged within model only when all expected speakers are complete; model means are averaged equally only when all three model estimates are complete. Missing seeds/clips/speakers/models remain explicit. No CI or inferential test.",
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
