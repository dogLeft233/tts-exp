#!/usr/bin/env python3
"""Independent, descriptive mouth-motion sidecar for cross-TFG renders.

This tool reads rendered video only. It does not score SyncNet, alter video or
audio, select endpoints, or treat larger mouth motion as an improvement. It
extracts eye-normalized mouth geometry on the render's declared frame clock,
then reports paired descriptive changes against the planned identity method.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "cross-tfg-visual-motion-guard-v1"
FEATURE_SOURCE_REL = Path("scripts/experiments/lrs3_tts_visual_advantage/video_features.py")
FEATURE_SOURCE_SHA256 = "2ce7006030e7f93b139cbbc8f5152c9b39719bea2fe7fbf16fbf18110cce5ff4"
LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
IDENTITY_FIELDS = (
    "row_id", "split", "sample_id", "speaker_id", "model_id", "audio_frontend",
    "model_role", "face_id", "seed", "method_id", "method_config_sha256", "checkpoint_sha256",
)
SUMMARY_METRICS = (
    "aperture_median_ipd", "aperture_p05_ipd", "aperture_p95_ipd", "aperture_range_p95_p05_ipd",
    "mouth_width_median_ipd", "mouth_height_median_ipd", "mouth_area_median_ipd2",
    "aperture_velocity_mean_ipd_per_s", "aperture_velocity_rms_ipd_per_s",
    "aperture_velocity_p05_ipd_per_s", "aperture_velocity_p95_ipd_per_s",
    "closing_transition_fraction", "closing_speed_median_ipd_per_s",
    "aperture_acceleration_abs_median_ipd_per_s2", "aperture_jerk_abs_median_ipd_per_s3",
)


class GuardError(RuntimeError):
    pass


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha(value: Mapping[str, Any], self_field: str) -> str:
    return _sha256_bytes(_canonical_json({key: item for key, item in value.items() if key != self_field}))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GuardError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise GuardError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _resolve(root: Path, value: Any, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise GuardError(f"{where} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require_sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise GuardError(f"{where} must be a lowercase SHA-256")
    return value


def _load_feature_module(path: Path) -> Any:
    if _sha256_file(path) != FEATURE_SOURCE_SHA256:
        raise GuardError("pinned visual feature source SHA mismatch")
    spec = importlib.util.spec_from_file_location("cross_tfg_pinned_video_features", path)
    if spec is None or spec.loader is None:
        raise GuardError("cannot load the pinned video feature module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _clock_sidecar(path: Path, expected_sha: str, frame_count: int, fps_num: int, fps_den: int) -> list[Fraction]:
    raw = path.read_bytes()
    if _sha256_bytes(raw) != expected_sha:
        raise GuardError(f"render timestamp sidecar SHA mismatch: {path}")
    value = json.loads(raw.decode("utf-8"))
    frames = value.get("frames") if isinstance(value, dict) else value
    if not isinstance(frames, list) or len(frames) != frame_count:
        raise GuardError(f"render timestamp count mismatch: {path}")
    times: list[Fraction] = []
    for index, item in enumerate(frames):
        if not isinstance(item, dict) or item.get("frame_index") != index:
            raise GuardError(f"render timestamp indices are not contiguous: {path}:{index}")
        numerator, denominator = item.get("timestamp_num"), item.get("timestamp_den")
        if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
            raise GuardError(f"invalid rational render timestamp: {path}:{index}")
        stamp = Fraction(numerator, denominator)
        expected = Fraction(index * fps_den, fps_num)
        if stamp != expected:
            raise GuardError(f"render timestamp disagrees with frozen FPS at frame {index}: {path}")
        times.append(stamp)
    return times


def _quantile(values: np.ndarray, q: float) -> float | None:
    return float(np.quantile(values, q)) if values.size else None


def _trajectory_summary(sequence: Any, times: Sequence[Fraction]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    valid = np.asarray(sequence.valid, dtype=bool)
    aperture = np.asarray(sequence.mouth_open, dtype=np.float64)
    width = np.asarray(sequence.mouth_width, dtype=np.float64)
    height = np.asarray(sequence.mouth_height, dtype=np.float64)
    area = np.asarray(sequence.mouth_area, dtype=np.float64)
    if any(array.shape != valid.shape for array in (aperture, width, height, area)) or len(times) != len(valid):
        raise GuardError("mouth feature arrays do not share the frozen render frame clock")
    frame_count = len(valid)
    dt = np.diff(np.asarray([float(value) for value in times], dtype=np.float64))
    adjacent_valid = valid[:-1] & valid[1:] if frame_count > 1 else np.zeros(0, dtype=bool)
    velocity = np.zeros(max(0, frame_count - 1), dtype=np.float64)
    velocity[adjacent_valid] = np.diff(aperture)[adjacent_valid] / dt[adjacent_valid]
    velocity_valid = adjacent_valid & np.isfinite(velocity)
    triplet_valid = velocity_valid[:-1] & velocity_valid[1:] if velocity.size > 1 else np.zeros(0, dtype=bool)
    acceleration = np.zeros(max(0, velocity.size - 1), dtype=np.float64)
    if acceleration.size:
        dt_mid = 0.5 * (dt[:-1] + dt[1:])
        acceleration[triplet_valid] = np.diff(velocity)[triplet_valid] / dt_mid[triplet_valid]
    acceleration_valid = triplet_valid & np.isfinite(acceleration)
    quad_valid = acceleration_valid[:-1] & acceleration_valid[1:] if acceleration.size > 1 else np.zeros(0, dtype=bool)
    jerk = np.zeros(max(0, acceleration.size - 1), dtype=np.float64)
    if jerk.size:
        dt_accel = 0.5 * (dt[:-2] + 2 * dt[1:-1] + dt[2:]) / 2.0
        jerk[quad_valid] = np.diff(acceleration)[quad_valid] / dt_accel[quad_valid]
    jerk_valid = quad_valid & np.isfinite(jerk)

    selected_aperture = aperture[valid]
    selected_width = width[valid]
    selected_height = height[valid]
    selected_area = area[valid]
    selected_velocity = velocity[velocity_valid]
    closing_velocity = -selected_velocity[selected_velocity < 0]
    metric_values: dict[str, float | int | None] = {
        "valid_frame_count": int(valid.sum()),
        "render_frame_count": frame_count,
        "tracking_valid_fraction": float(valid.mean()) if frame_count else 0.0,
        "aperture_median_ipd": _quantile(selected_aperture, 0.50),
        "aperture_p05_ipd": _quantile(selected_aperture, 0.05),
        "aperture_p95_ipd": _quantile(selected_aperture, 0.95),
        "aperture_range_p95_p05_ipd": (float(np.quantile(selected_aperture, 0.95) - np.quantile(selected_aperture, 0.05)) if selected_aperture.size else None),
        "mouth_width_median_ipd": _quantile(selected_width, 0.50),
        "mouth_height_median_ipd": _quantile(selected_height, 0.50),
        "mouth_area_median_ipd2": _quantile(selected_area, 0.50),
        "aperture_velocity_transition_count": int(velocity_valid.sum()),
        "aperture_velocity_mean_ipd_per_s": float(selected_velocity.mean()) if selected_velocity.size else None,
        "aperture_velocity_rms_ipd_per_s": float(np.sqrt(np.mean(selected_velocity ** 2))) if selected_velocity.size else None,
        "aperture_velocity_p05_ipd_per_s": _quantile(selected_velocity, 0.05),
        "aperture_velocity_p95_ipd_per_s": _quantile(selected_velocity, 0.95),
        "closing_transition_fraction": float(np.mean(selected_velocity < 0)) if selected_velocity.size else None,
        "closing_speed_median_ipd_per_s": _quantile(closing_velocity, 0.50),
        "aperture_acceleration_transition_count": int(acceleration_valid.sum()),
        "aperture_acceleration_abs_median_ipd_per_s2": _quantile(np.abs(acceleration[acceleration_valid]), 0.50),
        "aperture_jerk_transition_count": int(jerk_valid.sum()),
        "aperture_jerk_abs_median_ipd_per_s3": _quantile(np.abs(jerk[jerk_valid]), 0.50),
        "interpretation": "descriptive movement and tracking guard; larger aperture or velocity is not inherently better",
    }
    arrays = {
        "frame_index_int32": np.arange(frame_count, dtype=np.int32),
        "timestamp_num_int64": np.asarray([value.numerator for value in times], dtype=np.int64),
        "timestamp_den_int64": np.asarray([value.denominator for value in times], dtype=np.int64),
        "tracking_valid_bool": valid.astype(np.bool_, copy=False),
        "mouth_aperture_eye_normalized_float64": aperture,
        "mouth_width_eye_normalized_float64": width,
        "mouth_height_eye_normalized_float64": height,
        "mouth_area_eye_normalized_squared_float64": area,
        "aperture_velocity_eye_normalized_per_s_float64": velocity,
        "aperture_velocity_valid_bool": velocity_valid.astype(np.bool_, copy=False),
        "aperture_acceleration_eye_normalized_per_s2_float64": acceleration,
        "aperture_acceleration_valid_bool": acceleration_valid.astype(np.bool_, copy=False),
        "aperture_jerk_eye_normalized_per_s3_float64": jerk,
        "aperture_jerk_valid_bool": jerk_valid.astype(np.bool_, copy=False),
    }
    return metric_values, arrays


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return {
        "path": path.name,
        "file_sha256": _sha256_file(path),
        "arrays": {
            key: {
                "dtype": np.asarray(value).dtype.str,
                "shape": list(np.asarray(value).shape),
                "array_sha256": _sha256_bytes(np.ascontiguousarray(value).tobytes(order="C")),
            }
            for key, value in arrays.items()
        },
    }


def _paired_geometry_deltas(
    plan_rows: Sequence[Mapping[str, Any]],
    result_by_id: Mapping[str, Mapping[str, Any]],
    visual_by_id: Mapping[str, Mapping[str, Any]],
    method_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    identities = [str(method["method_id"]) for method in method_by_id.values() if method.get("kind") == "identity"]
    if len(identities) != 1:
        return [{"status": "UNPAIRED_NO_UNIQUE_IDENTITY_METHOD", "identity_method_ids": identities}]
    identity_id = identities[0]
    grouped: dict[tuple[Any, ...], dict[str, str]] = defaultdict(dict)
    for row in plan_rows:
        key = tuple(row.get(field) for field in ("split", "sample_id", "speaker_id", "model_id", "face_id", "seed"))
        grouped[key][str(row["method_id"])] = str(row["row_id"])
    output: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        identity_row_id = rows.get(identity_id)
        identity = visual_by_id.get(identity_row_id or "")
        for method_id, row_id in rows.items():
            candidate = visual_by_id.get(row_id)
            if method_id == identity_id:
                continue
            record: dict[str, Any] = {
                "split": key[0], "sample_id": key[1], "speaker_id": key[2], "model_id": key[3],
                "face_id": key[4], "seed": key[5], "method_id": method_id,
                "identity_method_id": identity_id, "candidate_row_id": row_id,
                "identity_row_id": identity_row_id,
            }
            if not identity or not candidate or identity.get("status") != "PASS" or candidate.get("status") != "PASS":
                record.update({"status": "UNPAIRED_MISSING_OR_FAILED_VISUAL_ROW", "deltas_vs_identity": None})
            elif identity["clock_signature"] != candidate["clock_signature"]:
                record.update({"status": "UNPAIRED_CLOCK_MISMATCH", "query_clock_equal": False, "deltas_vs_identity": None})
            else:
                left = identity["metrics"]
                right = candidate["metrics"]
                record.update({
                    "status": "PASS",
                    "query_clock_equal": True,
                    "deltas_vs_identity": {
                        metric: (None if left.get(metric) is None or right.get(metric) is None else float(right[metric]) - float(left[metric]))
                        for metric in SUMMARY_METRICS
                    },
                })
            output.append(record)
    return output


def run(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    if out.exists():
        raise GuardError(f"output directory already exists: {out}")
    out.mkdir(parents=True)
    plan_path = Path(args.plan).resolve(strict=True)
    results_path = Path(args.results).resolve(strict=True)
    manifest_path = Path(args.result_manifest).resolve(strict=True)
    artifact_root = Path(args.artifact_root).resolve(strict=True)
    plan = _load_json(plan_path)
    results_manifest = _load_json(manifest_path)
    plan_raw_sha = _sha256_file(plan_path)
    results_raw_sha = _sha256_file(results_path)
    manifest_raw_sha = _sha256_file(manifest_path)
    if plan.get("schema_version") != "cross-tfg-audio-eval-v1" or plan.get("status") != "FROZEN":
        raise GuardError("plan is not a frozen cross-TFG audio plan")
    if plan.get("plan_sha256") != _payload_sha(plan, "plan_sha256"):
        raise GuardError("plan payload SHA mismatch")
    if results_manifest.get("schema_version") != "cross-tfg-audio-eval-v1" or results_manifest.get("run_id") != plan.get("run_id"):
        raise GuardError("result manifest schema/run_id differs from the frozen plan")
    if results_manifest.get("manifest_sha256") != _payload_sha(results_manifest, "manifest_sha256"):
        raise GuardError("result-manifest payload SHA mismatch")
    if results_manifest.get("plan_raw_sha256") != plan_raw_sha or results_manifest.get("plan_sha256") != plan.get("plan_sha256"):
        raise GuardError("result manifest does not bind this frozen plan")
    if results_manifest.get("results_jsonl_sha256") != results_raw_sha:
        raise GuardError("result manifest does not bind the exact results JSONL")
    planned_rows = plan.get("rows")
    results_rows = _load_jsonl(results_path)
    if not isinstance(planned_rows, list) or len(planned_rows) != len(results_rows):
        raise GuardError("plan and result row counts differ")
    if results_manifest.get("planned_rows") != len(planned_rows) or results_manifest.get("result_rows") != len(results_rows):
        raise GuardError("manifest row totals differ from the frozen row lists")
    plan_ids = [row.get("row_id") for row in planned_rows if isinstance(row, dict)]
    result_ids = [row.get("row_id") for row in results_rows]
    if len(plan_ids) != len(planned_rows) or len(set(plan_ids)) != len(plan_ids) or len(set(result_ids)) != len(result_ids):
        raise GuardError("plan/result row identities must be unique")
    expected_status_counts = Counter(str(row.get("status")) for row in results_rows)
    if results_manifest.get("status_counts") != {key: int(expected_status_counts.get(key, 0)) for key in ("PASS", "FAILED", "INCOMPLETE")}:
        raise GuardError("result-manifest status counts disagree with result rows")
    if results_manifest.get("counts_by_model_method") is None:
        raise GuardError("result manifest lacks per-model/method row accounting")
    for index, (planned, result) in enumerate(zip(planned_rows, results_rows, strict=True)):
        if result.get("row_id") != planned.get("row_id"):
            raise GuardError(f"result rows are not in frozen plan order at index {index}")
        for field in IDENTITY_FIELDS:
            if result.get(field) != planned.get(field):
                raise GuardError(f"identity mismatch at row {index}: {field}")
        for name, result_field in (("source_face", "source_face_sha256"), ("source_video", "source_video_sha256")):
            ref = planned.get(name)
            if not isinstance(ref, dict) or not isinstance(ref.get("file_sha256"), str) or result.get(result_field) != ref.get("file_sha256"):
                raise GuardError(f"result does not bind planned {name} at row {index}")
        if result.get("status") not in ("PASS", "FAILED", "INCOMPLETE"):
            raise GuardError(f"invalid result status at row {index}")
        if result.get("status") != "PASS" and not isinstance(result.get("failure_reason"), str):
            raise GuardError(f"non-PASS result lacks failure_reason at row {index}")

    feature_path = Path(args.video_features).resolve(strict=True)
    asset_path = Path(args.landmarker_asset).resolve(strict=True)
    if _sha256_file(feature_path) != FEATURE_SOURCE_SHA256 or _sha256_file(asset_path) != LANDMARKER_SHA256:
        raise GuardError("pinned MediaPipe helper/asset SHA mismatch")
    feature_module = _load_feature_module(feature_path)
    import cv2  # type: ignore[import-not-found]
    import mediapipe as mp  # type: ignore[import-not-found]

    try:
        _mp, landmarker = feature_module.create_landmarker(asset_path)
        handle = (_mp, landmarker)
        file_cache: dict[str, str] = {}
        status_counts: Counter[str] = Counter()
        visual_by_id: dict[str, dict[str, Any]] = {}
        output_rows: list[dict[str, Any]] = []
        for planned, result in zip(planned_rows, results_rows, strict=True):
            row_id = str(planned["row_id"])
            status = result.get("status")
            base: dict[str, Any] = {field: planned.get(field) for field in IDENTITY_FIELDS}
            base["producer_status"] = status
            if status != "PASS":
                reason = result.get("failure_reason")
                base.update({"status": "SOURCE_ROW_NOT_PASS", "failure_reason": reason})
                status_counts["SOURCE_ROW_NOT_PASS"] += 1
                output_rows.append(base)
                visual_by_id[row_id] = base
                continue

            render = result.get("render")
            if not isinstance(render, dict):
                raise GuardError(f"PASS row has no render object: {row_id}")
            video_path = _resolve(artifact_root, render.get("silent_video_path"), f"{row_id}.render.silent_video_path")
            expected_video_sha = _require_sha(render.get("silent_video_sha256"), f"{row_id}.render.silent_video_sha256")
            cache_key = str(video_path)
            actual_video_sha = file_cache.get(cache_key)
            if actual_video_sha is None:
                if not video_path.is_file():
                    raise GuardError(f"missing silent rendered video: {video_path}")
                actual_video_sha = _sha256_file(video_path)
                file_cache[cache_key] = actual_video_sha
            if actual_video_sha != expected_video_sha:
                raise GuardError(f"rendered video SHA mismatch for {row_id}")
            expected_frames = render.get("frame_count")
            fps_num, fps_den = render.get("fps_num"), render.get("fps_den")
            if type(expected_frames) is not int or expected_frames <= 0 or type(fps_num) is not int or type(fps_den) is not int or fps_num <= 0 or fps_den <= 0:
                raise GuardError(f"invalid frame count/FPS metadata for {row_id}")
            timestamp_path = _resolve(artifact_root, render.get("timestamps_path"), f"{row_id}.render.timestamps_path")
            timestamp_sha = _require_sha(render.get("timestamps_sha256"), f"{row_id}.render.timestamps_sha256")
            exact_times = _clock_sidecar(timestamp_path, timestamp_sha, expected_frames, fps_num, fps_den)
            sequence = feature_module.extract_video_features(video_path, asset_path, landmarker_handle=handle)
            if sequence.metadata.get("decoded_frame_count") != expected_frames or len(sequence.valid) != expected_frames:
                raise GuardError(f"MediaPipe frame count differs from sealed render frame count for {row_id}")
            if not np.allclose(sequence.timestamps_s, np.asarray([float(value) for value in exact_times]), rtol=0.0, atol=1e-5):
                raise GuardError(f"MediaPipe decoder clock differs from frozen render timestamp sidecar for {row_id}")
            metrics, trajectory_arrays = _trajectory_summary(sequence, exact_times)
            stem = hashlib.sha256(row_id.encode("utf-8")).hexdigest()[:20]
            npz_path = out / "trajectories" / f"{stem}.npz"
            npz_ref = _write_npz(npz_path, trajectory_arrays)
            npz_ref["path"] = npz_path.relative_to(out).as_posix()
            clock_signature = _sha256_bytes(_canonical_json({
                "fps_num": fps_num, "fps_den": fps_den,
                "frame_count": expected_frames,
                "timestamps": [[value.numerator, value.denominator] for value in exact_times],
            }))
            record = {
                **base,
                "status": "PASS",
                "source_face_sha256": planned["source_face"]["file_sha256"],
                "video": {"path": str(video_path), "file_sha256": actual_video_sha, "kind": "silent_render"},
                "render_clock": {"frame_count": expected_frames, "fps_num": fps_num, "fps_den": fps_den, "timestamps_path": str(timestamp_path), "timestamps_sha256": timestamp_sha, "clock_signature": clock_signature},
                "feature_source_sha256": FEATURE_SOURCE_SHA256,
                "landmarker_asset_sha256": LANDMARKER_SHA256,
                "visual_npz": npz_ref,
                "metrics": metrics,
            }
            status_counts["PASS"] += 1
            visual_by_id[row_id] = {**record, "clock_signature": clock_signature}
            output_rows.append(record)
        landmarker.close()
    finally:
        if "landmarker" in locals():
            try:
                landmarker.close()
            except Exception:
                pass

    methods = plan.get("methods", [])
    method_by_id = {str(item["method_id"]): item for item in methods if isinstance(item, dict) and isinstance(item.get("method_id"), str)}
    paired = _paired_geometry_deltas(planned_rows, {str(row["row_id"]): row for row in results_rows}, visual_by_id, method_by_id)
    with (out / "visual_motion_results.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    with (out / "paired_geometry_deltas.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in paired:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")

    status = "PASS" if status_counts["PASS"] == len(planned_rows) else "INCOMPLETE"
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "run_id": plan.get("run_id"),
        "plan_path": str(plan_path), "plan_raw_sha256": plan_raw_sha, "plan_payload_sha256": plan["plan_sha256"],
        "results_path": str(results_path), "results_raw_sha256": results_raw_sha,
        "result_manifest_path": str(manifest_path), "result_manifest_raw_sha256": manifest_raw_sha,
        "result_manifest_payload_sha256": results_manifest["manifest_sha256"],
        "feature_source_path": str(feature_path), "feature_source_sha256": FEATURE_SOURCE_SHA256,
        "landmarker_asset_path": str(asset_path), "landmarker_asset_sha256": LANDMARKER_SHA256,
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "opencv": cv2.__version__, "mediapipe": mp.__version__},
        "planned_rows": len(planned_rows), "result_rows": len(results_rows), "visual_status_counts": dict(status_counts),
        "accounted_for_planned_rows": sum(status_counts.values()),
        "paired_geometry_row_count": len(paired), "paired_geometry_pass_count": sum(row.get("status") == "PASS" for row in paired),
        "metrics_definition": "eye-distance-normalized mouth geometry; finite differences use the sealed rational render clock; descriptive motion/tracking guard only",
        "larger_is_not_better": True,
        "effects_selected_from_visual_metrics": False,
        "outputs": ["visual_motion_results.jsonl", "paired_geometry_deltas.jsonl", "trajectories/"],
    }
    files = []
    for path in sorted(item for item in out.rglob("*") if item.is_file()):
        files.append({"path": path.relative_to(out).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    (out / "audit.json").write_bytes(_canonical_json(audit) + b"\n")
    files.append({"path": "audit.json", "bytes": (out / "audit.json").stat().st_size, "sha256": _sha256_file(out / "audit.json")})
    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "run_id": plan.get("run_id"),
        "files": files,
    }
    output_manifest["manifest_sha256"] = _payload_sha(output_manifest, "manifest_sha256")
    (out / "visual_motion_manifest.json").write_bytes(_canonical_json(output_manifest) + b"\n")
    return 0 if status == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--result-manifest", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--video-features", type=Path, default=repo / FEATURE_SOURCE_REL)
    parser.add_argument("--landmarker-asset", type=Path, default=repo / "checkpoints/mediapipe/face_landmarker.task")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        out = Path(args.out).resolve()
        if out.exists():
            error = {"schema_version": SCHEMA_VERSION, "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "argv": sys.argv}
            (out / "audit.json").write_bytes(_canonical_json(error) + b"\n")
        print(f"visual_motion_guard_failed:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
