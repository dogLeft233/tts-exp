#!/usr/bin/env python3
"""Strictly pair descriptive visual-motion outputs across frozen audio runs.

This comparison is a post-render descriptive guard. It does not score SyncNet,
select a candidate, or interpret larger mouth motion as better quality. Pairing
requires the same original PCM, portrait, model, seed, and exact rational frame
clock. Missing or failed rows remain in the reported planned denominators.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "cross-tfg-visual-motion-cross-run-comparison-v1"
IDENTITY_FIELDS = (
    "split", "sample_id", "speaker_id", "model_id", "face_id", "seed",
)
PAIR_METRICS = (
    "tracking_valid_fraction",
    "aperture_median_ipd",
    "aperture_range_p95_p05_ipd",
    "mouth_width_median_ipd",
    "mouth_height_median_ipd",
    "mouth_area_median_ipd2",
    "aperture_velocity_mean_ipd_per_s",
    "aperture_velocity_rms_ipd_per_s",
    "aperture_velocity_p95_ipd_per_s",
    "closing_transition_fraction",
    "aperture_acceleration_abs_median_ipd_per_s2",
    "aperture_jerk_abs_median_ipd_per_s3",
)
TRAJECTORY_ARRAYS = (
    "frame_index_int32", "timestamp_num_int64", "timestamp_den_int64",
    "tracking_valid_bool", "mouth_aperture_eye_normalized_float64",
    "mouth_width_eye_normalized_float64", "mouth_height_eye_normalized_float64",
    "mouth_area_eye_normalized_squared_float64",
)
CONTROLS = ("identity", "rms_fixed")
CANDIDATE_METHOD = "spectral_head_fit_v4_native_attempt01"


class ComparisonError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha(value: Mapping[str, Any], field: str) -> str:
    return _sha_bytes(_canonical({key: item for key, item in value.items() if key != field}))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ComparisonError(f"expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ComparisonError(f"expected JSON object at {path}:{number}")
            rows.append(row)
    return rows


def _read_run(label: str, plan_path: Path, results_path: Path, manifest_path: Path, visual_dir: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    results_path = results_path.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    visual_dir = visual_dir.resolve(strict=True)
    plan, manifest = _load_json(plan_path), _load_json(manifest_path)
    results = _load_jsonl(results_path)
    visual_audit = _load_json(visual_dir / "audit.json")
    visual_manifest = _load_json(visual_dir / "visual_motion_manifest.json")
    visual_rows = _load_jsonl(visual_dir / "visual_motion_results.jsonl")
    plan_raw, results_raw, manifest_raw = _sha_file(plan_path), _sha_file(results_path), _sha_file(manifest_path)
    if plan.get("schema_version") != "cross-tfg-audio-eval-v1" or plan.get("status") != "FROZEN":
        raise ComparisonError(f"{label}: plan is not frozen cross-TFG audio evaluation data")
    if plan.get("plan_sha256") != _payload_sha(plan, "plan_sha256"):
        raise ComparisonError(f"{label}: plan payload SHA mismatch")
    if manifest.get("schema_version") != "cross-tfg-audio-eval-v1" or manifest.get("run_id") != plan.get("run_id"):
        raise ComparisonError(f"{label}: result manifest run/schema mismatch")
    if manifest.get("manifest_sha256") != _payload_sha(manifest, "manifest_sha256"):
        raise ComparisonError(f"{label}: result manifest payload SHA mismatch")
    if manifest.get("plan_raw_sha256") != plan_raw or manifest.get("plan_sha256") != plan.get("plan_sha256"):
        raise ComparisonError(f"{label}: result manifest does not bind exact plan")
    if manifest.get("results_jsonl_sha256") != results_raw:
        raise ComparisonError(f"{label}: result manifest does not bind exact result ledger")
    planned = plan.get("rows")
    if not isinstance(planned, list) or len(planned) != len(results) or len(results) != len(visual_rows):
        raise ComparisonError(f"{label}: planned/result/visual row counts differ")
    if visual_audit.get("schema_version") != "cross-tfg-visual-motion-guard-v1":
        raise ComparisonError(f"{label}: unsupported visual audit schema")
    if (visual_manifest.get("schema_version") != "cross-tfg-visual-motion-guard-v1" or
            visual_manifest.get("manifest_sha256") != _payload_sha(visual_manifest, "manifest_sha256") or
            visual_manifest.get("run_id") != plan.get("run_id")):
        raise ComparisonError(f"{label}: visual output manifest schema/fingerprint/run mismatch")
    for field, expected in (("plan_raw_sha256", plan_raw), ("results_raw_sha256", results_raw),
                            ("result_manifest_raw_sha256", manifest_raw)):
        if visual_audit.get(field) != expected:
            raise ComparisonError(f"{label}: visual guard audit does not bind {field}")
    if visual_audit.get("planned_rows") != len(planned) or visual_audit.get("result_rows") != len(results):
        raise ComparisonError(f"{label}: visual guard audit row totals mismatch")
    visual_status_counts = Counter(str(row.get("status")) for row in visual_rows)
    if visual_audit.get("visual_status_counts") != dict(visual_status_counts):
        raise ComparisonError(f"{label}: visual audit status counts differ from row ledger")
    listed_files = visual_manifest.get("files")
    if not isinstance(listed_files, list):
        raise ComparisonError(f"{label}: visual output manifest lacks a file inventory")
    seen_visual_paths: set[str] = set()
    for item in listed_files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ComparisonError(f"{label}: malformed visual output inventory item")
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts or rel.as_posix() in seen_visual_paths:
            raise ComparisonError(f"{label}: unsafe or duplicate visual output path")
        seen_visual_paths.add(rel.as_posix())
        path = (visual_dir / rel).resolve(strict=True)
        if path.stat().st_size != item.get("bytes") or _sha_file(path) != item.get("sha256"):
            raise ComparisonError(f"{label}: visual output inventory mismatch for {rel.as_posix()}")
    visual_results_item = next((item for item in listed_files if isinstance(item, dict) and item.get("path") == "visual_motion_results.jsonl"), None)
    visual_results_path = visual_dir / "visual_motion_results.jsonl"
    if (not isinstance(visual_results_item, dict) or visual_results_item.get("bytes") != visual_results_path.stat().st_size or
            visual_results_item.get("sha256") != _sha_file(visual_results_path)):
        raise ComparisonError(f"{label}: visual results ledger is not sealed by its output manifest")
    result_by_id = {str(row.get("row_id")): row for row in results}
    visual_by_id = {str(row.get("row_id")): row for row in visual_rows}
    if len(result_by_id) != len(results) or len(visual_by_id) != len(visual_rows):
        raise ComparisonError(f"{label}: duplicate result or visual row IDs")
    for index, (p, r, v) in enumerate(zip(planned, results, visual_rows, strict=True)):
        if p.get("row_id") != r.get("row_id") or p.get("row_id") != v.get("row_id"):
            raise ComparisonError(f"{label}: row order/identity mismatch at {index}")
        for field in IDENTITY_FIELDS + ("method_id", "method_config_sha256", "checkpoint_sha256"):
            if p.get(field) != r.get(field) or p.get(field) != v.get(field):
                raise ComparisonError(f"{label}: {field} mismatch at row {index}")
        natural = p.get("natural_pcm")
        score_audio = p.get("score_audio")
        if not isinstance(natural, dict) or not isinstance(score_audio, dict):
            raise ComparisonError(f"{label}: missing natural/score audio reference at row {index}")
        if natural.get("file_sha256") != score_audio.get("file_sha256") or natural.get("pcm_sha256") != score_audio.get("pcm_sha256"):
            raise ComparisonError(f"{label}: SyncNet score audio is not the original natural PCM at row {index}")
        if r.get("status") == "PASS":
            natural_pcm_sha = natural.get("pcm_sha256")
            score_pcm_sha = r.get("score_audio_sha256")
            nested_score_sha = r.get("score", {}).get("score_audio_pcm_sha256")
            result_natural_sha = r.get("natural_pcm_sha256")
            if score_pcm_sha != natural_pcm_sha or nested_score_sha != natural_pcm_sha or result_natural_sha != natural_pcm_sha:
                raise ComparisonError(f"{label}: PASS score is not sealed to the natural PCM at row {index}")
        if v.get("source_face_sha256") != p.get("source_face", {}).get("file_sha256"):
            raise ComparisonError(f"{label}: visual row source-face pin mismatch at row {index}")
        if v.get("producer_status") != r.get("status"):
            raise ComparisonError(f"{label}: visual row producer status mismatch at row {index}")
        if r.get("status") == "PASS" and v.get("status") == "PASS":
            render = r.get("render", {})
            if v.get("video", {}).get("file_sha256") != render.get("silent_video_sha256"):
                raise ComparisonError(f"{label}: visual row video hash mismatch at row {index}")
            clock = v.get("render_clock", {})
            if (clock.get("frame_count") != render.get("frame_count") or
                    clock.get("fps_num") != render.get("fps_num") or clock.get("fps_den") != render.get("fps_den") or
                    clock.get("timestamps_sha256") != render.get("timestamps_sha256")):
                raise ComparisonError(f"{label}: visual clock does not match result render clock at row {index}")
    return {
        "label": label, "plan": plan, "planned_rows": planned, "results": results,
        "result_by_id": result_by_id, "visual_by_id": visual_by_id,
        "visual_dir": visual_dir,
        "binding": {"run_id": plan.get("run_id"), "plan_path": str(plan_path), "plan_raw_sha256": plan_raw,
                    "results_path": str(results_path), "results_raw_sha256": results_raw,
                    "result_manifest_path": str(manifest_path), "result_manifest_raw_sha256": manifest_raw,
                    "visual_dir": str(visual_dir), "visual_audit_status": visual_audit.get("status")},
    }


def _key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in IDENTITY_FIELDS)


def _load_trajectory(visual_dir: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    ref = row.get("visual_npz")
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise ComparisonError(f"visual row has no trajectory archive: {row.get('row_id')}")
    path = (visual_dir / ref["path"]).resolve(strict=True)
    if visual_dir.resolve() not in path.parents or _sha_file(path) != ref.get("file_sha256"):
        raise ComparisonError(f"visual trajectory file binding mismatch: {row.get('row_id')}")
    arrays: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in TRAJECTORY_ARRAYS:
            if name not in archive.files:
                raise ComparisonError(f"visual trajectory lacks {name}: {row.get('row_id')}")
            value = np.asarray(archive[name])
            descriptor = ref.get("arrays", {}).get(name)
            if (not isinstance(descriptor, dict) or descriptor.get("dtype") != value.dtype.str or
                    descriptor.get("shape") != list(value.shape) or
                    descriptor.get("array_sha256") != _sha_bytes(np.ascontiguousarray(value).tobytes(order="C"))):
                raise ComparisonError(f"visual trajectory array seal mismatch: {row.get('row_id')}:{name}")
            arrays[name] = value
    n = len(arrays["frame_index_int32"])
    if any(arrays[name].shape != (n,) for name in TRAJECTORY_ARRAYS):
        raise ComparisonError(f"visual trajectory array shape mismatch: {row.get('row_id')}")
    if not np.array_equal(arrays["frame_index_int32"], np.arange(n, dtype=np.int32)):
        raise ComparisonError(f"visual trajectory frame index mismatch: {row.get('row_id')}")
    if n != row.get("render_clock", {}).get("frame_count"):
        raise ComparisonError(f"visual trajectory length differs from row clock: {row.get('row_id')}")
    clock = row.get("render_clock", {})
    fps_num, fps_den = clock.get("fps_num"), clock.get("fps_den")
    nums, dens = arrays["timestamp_num_int64"], arrays["timestamp_den_int64"]
    if type(fps_num) is not int or type(fps_den) is not int or fps_num <= 0 or fps_den <= 0 or np.any(dens <= 0):
        raise ComparisonError(f"invalid rational render clock: {row.get('row_id')}")
    for index, (num, den) in enumerate(zip(nums.tolist(), dens.tolist(), strict=True)):
        if Fraction(int(num), int(den)) != Fraction(index * fps_den, fps_num):
            raise ComparisonError(f"trajectory timestamp differs from its frozen frame clock: {row.get('row_id')}[{index}]")
    return arrays


def _summary_stats(values: Any) -> dict[str, float | None]:
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"median": None, "mean": None, "rms": None, "p95": None}
    return {"median": float(np.median(values)), "mean": float(np.mean(values)),
            "rms": float(np.sqrt(np.mean(values * values))), "p95": float(np.quantile(values, 0.95))}


def _common_support_deltas(candidate_visual: Mapping[str, Any], control_visual: Mapping[str, Any], fps_num: int, fps_den: int) -> tuple[dict[str, float | None], dict[str, Any]]:
    import numpy as np

    c = candidate_visual["_arrays"]
    b = control_visual["_arrays"]
    cvalid = np.asarray(c["tracking_valid_bool"], dtype=bool)
    bvalid = np.asarray(b["tracking_valid_bool"], dtype=bool)
    n = min(cvalid.size, bvalid.size)
    common = cvalid[:n] & bvalid[:n]
    frame_idx = np.flatnonzero(common)
    outputs: dict[str, float | None] = {}
    feature_map = {
        "aperture_median_ipd": "mouth_aperture_eye_normalized_float64",
        "mouth_width_median_ipd": "mouth_width_eye_normalized_float64",
        "mouth_height_median_ipd": "mouth_height_eye_normalized_float64",
        "mouth_area_median_ipd2": "mouth_area_eye_normalized_squared_float64",
    }
    c_ap = np.asarray(c["mouth_aperture_eye_normalized_float64"], dtype=np.float64)[:n]
    b_ap = np.asarray(b["mouth_aperture_eye_normalized_float64"], dtype=np.float64)[:n]
    for metric, field in feature_map.items():
        left = np.asarray(c[field], dtype=np.float64)[:n][common]
        right = np.asarray(b[field], dtype=np.float64)[:n][common]
        if metric == "aperture_median_ipd":
            left_value = float(np.median(left)) if left.size else None
            right_value = float(np.median(right)) if right.size else None
        else:
            left_value = float(np.median(left)) if left.size else None
            right_value = float(np.median(right)) if right.size else None
        outputs[metric] = None if left_value is None or right_value is None else left_value - right_value
    c_ap_common, b_ap_common = c_ap[common], b_ap[common]
    c_range = float(np.quantile(c_ap_common, .95) - np.quantile(c_ap_common, .05)) if c_ap_common.size else None
    b_range = float(np.quantile(b_ap_common, .95) - np.quantile(b_ap_common, .05)) if b_ap_common.size else None
    outputs["aperture_range_p95_p05_ipd"] = None if c_range is None or b_range is None else c_range - b_range

    transition = common[:-1] & common[1:] if n > 1 else np.zeros(0, dtype=bool)
    dt = float(Fraction(fps_den, fps_num))
    c_vel_full = np.diff(c_ap) / dt if n > 1 else np.zeros(0, dtype=np.float64)
    b_vel_full = np.diff(b_ap) / dt if n > 1 else np.zeros(0, dtype=np.float64)
    c_vel, b_vel = c_vel_full[transition], b_vel_full[transition]
    c_vs, b_vs = _summary_stats(c_vel), _summary_stats(b_vel)
    outputs["aperture_velocity_mean_ipd_per_s"] = None if c_vs["mean"] is None or b_vs["mean"] is None else c_vs["mean"] - b_vs["mean"]
    outputs["aperture_velocity_rms_ipd_per_s"] = None if c_vs["rms"] is None or b_vs["rms"] is None else c_vs["rms"] - b_vs["rms"]
    outputs["aperture_velocity_p95_ipd_per_s"] = None if c_vs["p95"] is None or b_vs["p95"] is None else c_vs["p95"] - b_vs["p95"]
    c_close = float(np.mean(c_vel < 0)) if c_vel.size else None
    b_close = float(np.mean(b_vel < 0)) if b_vel.size else None
    outputs["closing_transition_fraction"] = None if c_close is None or b_close is None else c_close - b_close

    # Higher derivatives use only consecutive transitions with shared tracked frames.
    triplet = common[:-2] & common[1:-1] & common[2:] if n > 2 else np.zeros(0, dtype=bool)
    c_acc_full = np.diff(c_vel_full) / dt if c_vel_full.size > 1 else np.zeros(0, dtype=np.float64)
    b_acc_full = np.diff(b_vel_full) / dt if b_vel_full.size > 1 else np.zeros(0, dtype=np.float64)
    c_acc, b_acc = c_acc_full[triplet], b_acc_full[triplet]
    acc_stats_c = _summary_stats(np.abs(c_acc))
    acc_stats_b = _summary_stats(np.abs(b_acc))
    outputs["aperture_acceleration_abs_median_ipd_per_s2"] = None if acc_stats_c["median"] is None or acc_stats_b["median"] is None else acc_stats_c["median"] - acc_stats_b["median"]
    quadruplet = common[:-3] & common[1:-2] & common[2:-1] & common[3:] if n > 3 else np.zeros(0, dtype=bool)
    c_jerk_full = np.diff(c_acc_full) / dt if c_acc_full.size > 1 else np.zeros(0, dtype=np.float64)
    b_jerk_full = np.diff(b_acc_full) / dt if b_acc_full.size > 1 else np.zeros(0, dtype=np.float64)
    c_jerk, b_jerk = c_jerk_full[quadruplet], b_jerk_full[quadruplet]
    jerk_stats_c = _summary_stats(np.abs(c_jerk))
    jerk_stats_b = _summary_stats(np.abs(b_jerk))
    outputs["aperture_jerk_abs_median_ipd_per_s3"] = None if jerk_stats_c["median"] is None or jerk_stats_b["median"] is None else jerk_stats_c["median"] - jerk_stats_b["median"]
    return outputs, {
        "common_valid_frame_count": int(common.sum()), "compared_frame_count": int(n),
        "candidate_valid_frame_count": int(cvalid[:n].sum()), "control_valid_frame_count": int(bvalid[:n].sum()),
        "candidate_valid_frame_fraction": float(cvalid.mean()) if cvalid.size else 0.0,
        "control_valid_frame_fraction": float(bvalid.mean()) if bvalid.size else 0.0,
        "common_valid_fraction": float(common.mean()) if n else 0.0,
        "common_transition_count": int(transition.sum()),
        "candidate_valid_frame_count_full": int(cvalid.sum()), "control_valid_frame_count_full": int(bvalid.sum()),
    }


def run(args: argparse.Namespace) -> int:
    candidate = _read_run("candidate", Path(args.candidate_plan), Path(args.candidate_results),
                          Path(args.candidate_manifest), Path(args.candidate_visual_dir))
    baseline_set_path = Path(args.baseline_set).resolve(strict=True)
    baseline_set = _load_json(baseline_set_path)
    if baseline_set.get("schema_version") != "cross-tfg-visual-baseline-set-v1":
        raise ComparisonError("baseline set schema mismatch")
    if baseline_set.get("manifest_sha256") != _payload_sha(baseline_set, "manifest_sha256"):
        raise ComparisonError("baseline set payload SHA mismatch")
    entries = baseline_set.get("runs")
    if not isinstance(entries, list) or not entries:
        raise ComparisonError("baseline set must list at least one baseline run")
    baselines = []
    seen_models: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ComparisonError(f"baseline set entry {index} is malformed")
        run = _read_run(f"baseline_{index}", Path(entry["plan"]), Path(entry["results"]),
                        Path(entry["result_manifest"]), Path(entry["visual_dir"]))
        expected_model = entry.get("model_id")
        found_models = {str(row.get("model_id")) for row in run["planned_rows"]}
        if found_models != {expected_model}:
            raise ComparisonError(f"baseline set entry {index} model roster differs from {expected_model!r}: {sorted(found_models)}")
        methods = {str(item.get("method_id")) for item in run["plan"].get("methods", []) if isinstance(item, dict)}
        if not set(CONTROLS).issubset(methods):
            raise ComparisonError(f"baseline set entry {index} lacks identity/rms_fixed controls")
        if expected_model in seen_models:
            raise ComparisonError(f"baseline model is listed more than once: {expected_model}")
        seen_models.add(str(expected_model))
        run["expected_model_id"] = expected_model
        baselines.append(run)
    cand_methods = {str(m.get("method_id")) for m in candidate["plan"].get("methods", []) if isinstance(m, dict)}
    base_methods = {str(m.get("method_id")) for m in baseline["plan"].get("methods", []) if isinstance(m, dict)}
    if cand_methods != {CANDIDATE_METHOD}:
        raise ComparisonError(f"candidate plan methods differ: {sorted(cand_methods)}")
    if not set(CONTROLS).issubset(base_methods):
        raise ComparisonError(f"baseline plan lacks required controls: {sorted(set(CONTROLS) - base_methods)}")

    candidate_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    baseline_by_key_method: dict[tuple[tuple[Any, ...], str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for p in candidate["planned_rows"]:
        k = _key(p)
        if k in candidate_by_key:
            raise ComparisonError(f"duplicate candidate context key: {k}")
        candidate_by_key[k] = p
    for baseline_run in baselines:
        for p in baseline_run["planned_rows"]:
            key = (_key(p), str(p.get("method_id")))
            if key in baseline_by_key_method:
                raise ComparisonError(f"duplicate baseline context/method key: {key}")
            baseline_by_key_method[key] = (baseline_run, p)

    comparisons: list[dict[str, Any]] = []
    for k, cp in candidate_by_key.items():
        c_result = candidate["result_by_id"].get(str(cp["row_id"]), {})
        c_visual = candidate["visual_by_id"].get(str(cp["row_id"]), {})
        for control_method in CONTROLS:
            baseline_binding = baseline_by_key_method.get((k, control_method))
            baseline_run, bp = baseline_binding if baseline_binding is not None else (None, None)
            record: dict[str, Any] = {
                **dict(zip(IDENTITY_FIELDS, k, strict=True)),
                "candidate_method_id": CANDIDATE_METHOD,
                "control_method_id": control_method,
                "candidate_row_id": cp.get("row_id"),
                "control_row_id": bp.get("row_id") if bp else None,
                "status": "UNPAIRED_MISSING_CONTROL_PLAN_ROW",
                "deltas_candidate_minus_control": None,
            }
            if bp is not None:
                assert baseline_run is not None
                b_result = baseline_run["result_by_id"].get(str(bp["row_id"]), {})
                b_visual = baseline_run["visual_by_id"].get(str(bp["row_id"]), {})
                c_nat, b_nat = cp.get("natural_pcm", {}), bp.get("natural_pcm", {})
                c_score, b_score = cp.get("score_audio", {}), bp.get("score_audio", {})
                bindings_equal = (
                    c_nat.get("file_sha256") == b_nat.get("file_sha256") and
                    c_nat.get("pcm_sha256") == b_nat.get("pcm_sha256") and
                    c_nat.get("sample_count") == b_nat.get("sample_count") and
                    c_score.get("file_sha256") == b_score.get("file_sha256") and
                    cp.get("source_face", {}).get("file_sha256") == bp.get("source_face", {}).get("file_sha256") and
                    cp.get("source_video", {}).get("file_sha256") == bp.get("source_video", {}).get("file_sha256")
                )
                record["original_pcm_and_source_bindings_equal"] = bool(bindings_equal)
                record["candidate_status"] = c_result.get("status")
                record["control_status"] = b_result.get("status")
                record["candidate_visual_status"] = c_visual.get("status")
                record["control_visual_status"] = b_visual.get("status")
                if not bindings_equal:
                    record["status"] = "UNPAIRED_SOURCE_BINDING_MISMATCH"
                elif c_result.get("status") != "PASS" or b_result.get("status") != "PASS":
                    record["status"] = "UNPAIRED_RENDER_ROW_NOT_PASS"
                elif c_visual.get("status") != "PASS" or b_visual.get("status") != "PASS":
                    record["status"] = "UNPAIRED_VISUAL_ROW_NOT_PASS"
                elif c_visual.get("render_clock", {}).get("clock_signature") != b_visual.get("render_clock", {}).get("clock_signature"):
                    record["status"] = "UNPAIRED_RATIONAL_CLOCK_MISMATCH"
                elif (c_visual.get("render_clock", {}).get("frame_count") != b_visual.get("render_clock", {}).get("frame_count") or
                      c_visual.get("render_clock", {}).get("fps_num") != b_visual.get("render_clock", {}).get("fps_num") or
                      c_visual.get("render_clock", {}).get("fps_den") != b_visual.get("render_clock", {}).get("fps_den")):
                    record["status"] = "UNPAIRED_RATIONAL_CLOCK_MISMATCH"
                else:
                    if "_arrays" not in c_visual:
                        c_visual["_arrays"] = _load_trajectory(candidate["visual_dir"], c_visual)
                    if "_arrays" not in b_visual:
                        b_visual["_arrays"] = _load_trajectory(baseline_run["visual_dir"], b_visual)
                    deltas, support = _common_support_deltas(
                        c_visual, b_visual,
                        int(c_visual["render_clock"]["fps_num"]),
                        int(c_visual["render_clock"]["fps_den"]),
                    )
                    deltas["tracking_valid_fraction"] = float(c_visual["metrics"]["tracking_valid_fraction"]) - float(b_visual["metrics"]["tracking_valid_fraction"])
                    record.update({"status": "PAIRED_DESCRIPTIVE", "deltas_candidate_minus_control": deltas,
                                   "clock_signature": c_visual["render_clock"]["clock_signature"],
                                   "common_valid_support": support})
            comparisons.append(record)

    model_summary: dict[str, Any] = {}
    speaker_rows: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    expected_by_speaker: Counter[tuple[str, str, str]] = Counter()
    paired_by_speaker: Counter[tuple[str, str, str]] = Counter()
    for row in comparisons:
        group = (str(row["model_id"]), str(row["control_method_id"]), str(row["speaker_id"]))
        expected_by_speaker[group] += 1
        if row["status"] != "PAIRED_DESCRIPTIVE":
            continue
        paired_by_speaker[group] += 1
        for metric, value in row["deltas_candidate_minus_control"].items():
            if value is not None:
                speaker_rows[group][metric].append(float(value))
    models = sorted({str(row.get("model_id")) for row in candidate["planned_rows"]})
    speakers = sorted({str(row.get("speaker_id")) for row in candidate["planned_rows"]})
    for model in models:
        model_summary[model] = {}
        for control_method in CONTROLS:
            subset = [r for r in comparisons if r["model_id"] == model and r["control_method_id"] == control_method]
            paired_speaker = {
                speaker: speaker_rows.get((model, control_method, speaker), {})
                for speaker in speakers
            }
            complete_speakers = [speaker for speaker in speakers
                                 if expected_by_speaker[(model, control_method, speaker)] > 0
                                 and paired_by_speaker[(model, control_method, speaker)] == expected_by_speaker[(model, control_method, speaker)]]
            metric_summary = {}
            for metric in PAIR_METRICS:
                speaker_means = [sum(paired_speaker[speaker][metric]) / len(paired_speaker[speaker][metric])
                                 for speaker in complete_speakers
                                 if metric in paired_speaker[speaker] and paired_speaker[speaker][metric]]
                metric_summary[metric] = {
                    "equal_speaker_mean_delta": (sum(speaker_means) / len(speaker_means)) if speaker_means else None,
                    "fully_paired_speaker_count": len(speaker_means),
                    "planned_speaker_count": len(speakers),
                    "all_planned_speakers_fully_paired": len(speaker_means) == len(speakers),
                }
            model_summary[model][control_method] = {
                "planned_candidate_contexts": len(candidate["planned_rows"]) // max(1, len(models)),
                "paired_context_count": sum(r["status"] == "PAIRED_DESCRIPTIVE" for r in subset),
                "planned_contexts_by_speaker": {speaker: expected_by_speaker[(model, control_method, speaker)] for speaker in speakers},
                "paired_contexts_by_speaker": {speaker: paired_by_speaker[(model, control_method, speaker)] for speaker in speakers},
                "fully_paired_speakers": complete_speakers,
                "unpaired_status_counts": dict(Counter(r["status"] for r in subset if r["status"] != "PAIRED_DESCRIPTIVE")),
                "metric_deltas": metric_summary,
            }
    equal_model_summary = {}
    for control_method in CONTROLS:
        equal_model_summary[control_method] = {}
        for metric in PAIR_METRICS:
            values = [model_summary[model][control_method]["metric_deltas"][metric]["equal_speaker_mean_delta"] for model in models]
            selected = [float(value) for value in values if value is not None]
            equal_model_summary[control_method][metric] = {
                "equal_model_mean_delta": sum(selected) / len(selected) if selected else None,
                "models_with_estimable_pairs": len(selected),
                "per_model_deltas": {model: model_summary[model][control_method]["metric_deltas"][metric]["equal_speaker_mean_delta"] for model in models},
            }

    out = Path(args.out).resolve()
    if out.exists():
        raise ComparisonError(f"output path already exists: {out}")
    out.mkdir(parents=True)
    with (out / "paired_visual_rows.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in comparisons:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    summary = {
        "schema_version": SCHEMA,
        "status": "DESCRIPTIVE_COMPLETE" if all(r["status"] == "PAIRED_DESCRIPTIVE" for r in comparisons) else "DESCRIPTIVE_WITH_UNPAIRED_ROWS",
        "bindings": {"candidate": candidate["binding"], "baseline_set_path": str(baseline_set_path),
                      "baseline_set_raw_sha256": _sha_file(baseline_set_path),
                      "baseline_runs": [run["binding"] for run in baselines]},
        "candidate_method_id": CANDIDATE_METHOD,
        "control_method_ids": list(CONTROLS),
        "candidate_planned_contexts": len(candidate["planned_rows"]),
        "comparison_planned_pairs": len(comparisons),
        "paired_status_counts": dict(Counter(row["status"] for row in comparisons)),
        "models": model_summary,
        "equal_model_descriptive_summary": equal_model_summary,
        "aggregation": "paired row deltas; mean within speaker across planned clip/seed contexts; then equal-weight speakers per model; then equal-weight models",
        "interpretation": "descriptive geometry and tracking guard only; larger aperture, area, or velocity is not inherently better; no confidence intervals, causal claims, or candidate selection",
    }
    (out / "summary.json").write_bytes(_canonical(summary) + b"\n")
    return 0 if summary["status"] == "DESCRIPTIVE_COMPLETE" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in ("candidate",):
        parser.add_argument(f"--{prefix}-plan", required=True)
        parser.add_argument(f"--{prefix}-results", required=True)
        parser.add_argument(f"--{prefix}-manifest", required=True)
        parser.add_argument(f"--{prefix}-visual-dir", required=True)
    parser.add_argument("--baseline-set", required=True, help="sealed JSON manifest naming one baseline run per model")
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(f"visual_motion_comparison_failed:{type(exc).__name__}:{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
