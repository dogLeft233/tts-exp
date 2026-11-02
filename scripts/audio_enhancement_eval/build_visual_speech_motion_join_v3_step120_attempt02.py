#!/usr/bin/env python3
"""Build a read-only step-120 candidate/identity/RMS visual join.

The builder joins already rendered identity, RMS, and one learned candidate
rows from separate frozen runs. It copies the sealed result rows and visual
trajectory arrays; it does not render, score, resample, or alter any source run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "cross-tfg-visual-speech-motion-join-v1"
PLAN_SCHEMA = "cross-tfg-audio-eval-v1"
VISUAL_SCHEMA = "cross-tfg-visual-motion-guard-v1"
IDENTITY = ("row_id", "split", "sample_id", "speaker_id", "model_id", "audio_frontend", "model_role",
            "face_id", "seed", "method_id", "method_config_sha256", "checkpoint_sha256")
GROUP_KEY = ("split", "sample_id", "speaker_id", "model_id", "face_id", "seed")
REQUIRED_METHODS = ("identity", "rms_fixed", "shared_recipient_transition_head_v3_fit40_step120")
MODEL_BINDING_FIELDS = (
    "model_id", "audio_frontend", "model_sha256", "checkpoint_sha256", "code_sha256",
    "preprocessing_sha256", "generator_audio_preprocessing_sha256", "score_model_sha256",
    "fps_num", "fps_den",
)
VISUAL_ARRAYS = (
    "frame_index_int32", "timestamp_num_int64", "timestamp_den_int64", "tracking_valid_bool",
    "mouth_aperture_eye_normalized_float64", "mouth_width_eye_normalized_float64",
    "mouth_height_eye_normalized_float64", "mouth_area_eye_normalized_squared_float64",
    "aperture_velocity_eye_normalized_per_s_float64", "aperture_velocity_valid_bool",
)
EXPECTED_FEATURE_SOURCE_SHA256 = "2ce7006030e7f93b139cbbc8f5152c9b39719bea2fe7fbf16fbf18110cce5ff4"
EXPECTED_LANDMARKER_ASSET_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
RUNTIME_KEYS = ("python", "numpy", "opencv", "mediapipe")


class JoinBuildError(RuntimeError):
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


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise JoinBuildError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise JoinBuildError(f"expected JSON object at {path}:{line_no}")
                rows.append(row)
    return rows


def read_source(entry: Mapping[str, Any]) -> dict[str, Any]:
    label = str(entry.get("role"))
    plan_path = Path(str(entry["plan"])).resolve(strict=True)
    results_path = Path(str(entry["results"])).resolve(strict=True)
    manifest_path = Path(str(entry["result_manifest"])).resolve(strict=True)
    visual_dir = Path(str(entry["visual_dir"])).resolve(strict=True)
    paths = {"plan": plan_path, "results": results_path, "result_manifest": manifest_path}
    for key, path in paths.items():
        expected = entry.get(f"{key}_sha256")
        actual = sha_file(path)
        if not isinstance(expected, str) or actual != expected:
            raise JoinBuildError(f"{label}: exact {key} raw hash mismatch")
    plan, manifest = read_json(plan_path), read_json(manifest_path)
    results = read_jsonl(results_path)
    plan_sha, results_sha, manifest_sha = (sha_file(plan_path), sha_file(results_path), sha_file(manifest_path))
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "FROZEN" or plan.get("plan_sha256") != payload_sha(plan, "plan_sha256"):
        raise JoinBuildError(f"{label}: source plan schema/fingerprint is invalid")
    if manifest.get("schema_version") != PLAN_SCHEMA or manifest.get("run_id") != plan.get("run_id") or manifest.get("manifest_sha256") != payload_sha(manifest, "manifest_sha256"):
        raise JoinBuildError(f"{label}: source result manifest schema/fingerprint is invalid")
    if manifest.get("plan_raw_sha256") != plan_sha or manifest.get("plan_sha256") != plan.get("plan_sha256") or manifest.get("results_jsonl_sha256") != results_sha:
        raise JoinBuildError(f"{label}: result manifest does not bind source plan/results")
    planned = plan.get("rows")
    if not isinstance(planned, list) or len(planned) != len(results) or [r.get("row_id") for r in planned] != [r.get("row_id") for r in results]:
        raise JoinBuildError(f"{label}: frozen row/result order mismatch")
    for p, r in zip(planned, results, strict=True):
        if any(p.get(field) != r.get(field) for field in IDENTITY):
            raise JoinBuildError(f"{label}: result identity differs from plan: {p.get('row_id')}")

    allowed_result_statuses = {"PASS", "FAILED", "INCOMPLETE"}
    unknown_status_rows = [row for row in results
                           if type(row.get("status")) is not str or row.get("status") not in allowed_result_statuses]
    if unknown_status_rows:
        details = [(str(row.get("row_id")), repr(row.get("status"))) for row in unknown_status_rows[:10]]
        raise JoinBuildError(f"CONTROL_FAILED_UNRECOGNIZED_RESULT_STATUS:{label}:{details}")
    status_counts = Counter(row["status"] for row in results)
    expected_status_counts = {state: int(status_counts.get(state, 0)) for state in ("PASS", "FAILED", "INCOMPLETE")}
    manifest_counts = manifest.get("status_counts")
    if (type(manifest.get("planned_rows")) is not int or manifest.get("planned_rows") != len(planned)
            or type(manifest.get("result_rows")) is not int or manifest.get("result_rows") != len(results)
            or not isinstance(manifest_counts, dict)
            or set(manifest_counts) != set(expected_status_counts)
            or any(type(value) is not int for value in manifest_counts.values())
            or manifest_counts != expected_status_counts):
        raise JoinBuildError(f"{label}: result-manifest row/status totals differ from exact planned/result rows")
    expected_terminal_status = "PASS" if status_counts["FAILED"] == 0 and status_counts["INCOMPLETE"] == 0 else "INCOMPLETE"
    if manifest.get("status") != expected_terminal_status:
        raise JoinBuildError(f"{label}: result-manifest terminal status conflicts with exact result ledger")

    visual_manifest_path = visual_dir / "visual_motion_manifest.json"
    visual_audit_path = visual_dir / "audit.json"
    visual_results_path = visual_dir / "visual_motion_results.jsonl"
    expected_visual_sha = entry.get("visual_manifest_sha256")
    if not isinstance(expected_visual_sha, str) or sha_file(visual_manifest_path) != expected_visual_sha:
        raise JoinBuildError(f"{label}: visual output manifest raw hash mismatch")
    visual_manifest = read_json(visual_manifest_path)
    visual_audit = read_json(visual_audit_path)
    visual_rows = read_jsonl(visual_results_path)
    if visual_manifest.get("schema_version") != VISUAL_SCHEMA or visual_manifest.get("manifest_sha256") != payload_sha(visual_manifest, "manifest_sha256"):
        raise JoinBuildError(f"{label}: visual output manifest is invalid")
    if visual_audit.get("status") != visual_manifest.get("status") or visual_audit.get("run_id") != plan.get("run_id"):
        raise JoinBuildError(f"{label}: visual audit/manifest/source run mismatch")
    if (visual_audit.get("plan_raw_sha256") != plan_sha or visual_audit.get("results_raw_sha256") != results_sha or
            visual_audit.get("result_manifest_raw_sha256") != manifest_sha):
        raise JoinBuildError(f"{label}: visual guard does not bind exact frozen source files")
    if len(visual_rows) != len(planned) or [r.get("row_id") for r in visual_rows] != [r.get("row_id") for r in planned]:
        raise JoinBuildError(f"{label}: visual guard rows differ from frozen source rows")
    visual_items = {str(item.get("path")): item for item in visual_manifest.get("files", []) if isinstance(item, dict)}
    if len(visual_items) != len(visual_manifest.get("files", [])):
        raise JoinBuildError(f"{label}: visual inventory has duplicate/malformed paths")
    for rel, item in visual_items.items():
        relative = Path(rel)
        if relative.is_absolute() or ".." in relative.parts:
            raise JoinBuildError(f"{label}: visual inventory path escapes its root: {rel}")
        path = (visual_dir / relative).resolve(strict=True)
        if path.stat().st_size != item.get("bytes") or sha_file(path) != item.get("sha256"):
            raise JoinBuildError(f"{label}: visual inventory member mismatch: {rel}")
    for path_key, path in (("audit.json", visual_audit_path), ("visual_motion_results.jsonl", visual_results_path)):
        item = visual_items.get(path_key)
        if not isinstance(item, dict) or item.get("bytes") != path.stat().st_size or item.get("sha256") != sha_file(path):
            raise JoinBuildError(f"{label}: visual output file is not sealed: {path_key}")
    for p, r, v in zip(planned, results, visual_rows, strict=True):
        if any(v.get(field) != p.get(field) for field in IDENTITY):
            raise JoinBuildError(f"{label}: visual row identity differs from plan: {p.get('row_id')}")
        if v.get("producer_status") != r.get("status"):
            raise JoinBuildError(f"{label}: visual row producer status mismatch: {p.get('row_id')}")

    model_id = entry.get("model_id")
    planned_models = {str(row.get("model_id")) for row in planned}
    if label == "candidate" and model_id is None:
        if not planned_models:
            raise JoinBuildError(f"{label}: candidate source contains no model rows")
    elif planned_models != {model_id}:
        raise JoinBuildError(f"{label}: expected one model {model_id!r}, found {sorted(planned_models)}")
    return {"entry": dict(entry), "plan": plan, "planned": planned, "results": results,
            "visual_rows": visual_rows, "plan_path": plan_path, "results_path": results_path,
            "manifest_path": manifest_path, "visual_dir": visual_dir,
            "visual_manifest": visual_manifest, "visual_audit": visual_audit,
            "source_pins": {"plan_raw_sha256": plan_sha, "results_raw_sha256": results_sha,
                            "result_manifest_raw_sha256": manifest_sha,
                            "visual_manifest_raw_sha256": expected_visual_sha,
                            "visual_results_raw_sha256": sha_file(visual_results_path)}}


def group_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in GROUP_KEY)


def run(args: argparse.Namespace) -> int:
    import numpy as np

    input_path = Path(args.inputs).resolve(strict=True)
    inputs = read_json(input_path)
    if inputs.get("schema_version") != "cross-tfg-visual-speech-motion-input-v1" or inputs.get("manifest_sha256") != payload_sha(inputs, "manifest_sha256"):
        raise JoinBuildError("input manifest schema/fingerprint mismatch")
    entries = inputs.get("sources")
    if not isinstance(entries, list) or len(entries) != 4:
        raise JoinBuildError("expected exactly one step-120 candidate and three frontend baseline sources")
    context_count = inputs.get("contexts_per_model", 60)
    if type(context_count) is not int or context_count != 60:
        raise JoinBuildError("CONTROL_FAILED_FIXED_CONTEXT_DENOMINATOR: contexts_per_model must be exactly 60")
    sources = [read_source(entry) for entry in entries]
    source_visual_runtime = [{
        "role": source["entry"].get("role"), "model_id": source["entry"].get("model_id"),
        "runtime": source["visual_audit"].get("runtime"),
        "feature_source_sha256": source["visual_audit"].get("feature_source_sha256"),
        "landmarker_asset_sha256": source["visual_audit"].get("landmarker_asset_sha256"),
    } for source in sources]
    for item in source_visual_runtime:
        runtime = item.get("runtime")
        if (item.get("feature_source_sha256") != EXPECTED_FEATURE_SOURCE_SHA256 or
                item.get("landmarker_asset_sha256") != EXPECTED_LANDMARKER_ASSET_SHA256 or
                not isinstance(runtime, dict) or any(not isinstance(runtime.get(key), str) or not runtime[key]
                                                     for key in RUNTIME_KEYS)):
            raise JoinBuildError("each source must bind the qualified feature code, asset, and complete runtime")
    runtime_pins = {tuple(item["runtime"][key] for key in RUNTIME_KEYS) for item in source_visual_runtime}
    feature_pins = {(item["feature_source_sha256"], item["landmarker_asset_sha256"]) for item in source_visual_runtime}
    if len(feature_pins) != 1 or feature_pins != {(EXPECTED_FEATURE_SOURCE_SHA256, EXPECTED_LANDMARKER_ASSET_SHA256)}:
        raise JoinBuildError("source visual rows use different or unexpected feature code/landmark assets")
    if len(runtime_pins) != 1:
        raise JoinBuildError("source visual rows use different qualified runtimes")
    candidates = [source for source in sources if source["entry"].get("role") == "candidate"]
    baselines = [source for source in sources if source["entry"].get("role") == "baseline"]
    if len(candidates) != 1 or len(baselines) != 3:
        raise JoinBuildError("input manifest roles must be candidate + three baselines")
    candidate = candidates[0]
    if not (len(candidate["planned"]) == len(candidate["results"]) == len(candidate["visual_rows"]) == 180):
        raise JoinBuildError("CONTROL_FAILED_FIXED_CANDIDATE_DENOMINATOR: expected 180 candidate plan/result/visual rows")
    c_methods = {str(m.get("method_id")) for m in candidate["plan"].get("methods", []) if isinstance(m, dict)}
    if c_methods != {REQUIRED_METHODS[-1]}:
        raise JoinBuildError(f"candidate is not the frozen fixed-step-120 method: {sorted(c_methods)}")
    model_baseline: dict[str, dict[str, Any]] = {}
    for source in baselines:
        model_id = str(source["entry"].get("model_id"))
        if not (len(source["planned"]) == len(source["results"]) == len(source["visual_rows"]) == 120):
            raise JoinBuildError(f"CONTROL_FAILED_FIXED_BASELINE_DENOMINATOR: {model_id} must have 120 plan/result/visual rows")
        source_method_counts = Counter(str(row.get("method_id")) for row in source["planned"])
        if source_method_counts != Counter({"identity": 60, "rms_fixed": 60}):
            raise JoinBuildError(f"CONTROL_FAILED_FIXED_BASELINE_METHOD_COUNTS: {model_id}: {dict(source_method_counts)}")
        if model_id in model_baseline:
            raise JoinBuildError(f"duplicate baseline frontend: {model_id}")
        method_ids = {str(m.get("method_id")) for m in source["plan"].get("methods", []) if isinstance(m, dict)}
        if not {"identity", "rms_fixed"}.issubset(method_ids):
            raise JoinBuildError(f"baseline lacks identity/RMS: {model_id}")
        model_baseline[model_id] = source
    models = {str(row.get("model_id")) for row in candidate["planned"]}
    if len(models) != 3:
        raise JoinBuildError(f"CONTROL_FAILED_FIXED_MODEL_DENOMINATOR: expected three candidate models, found {sorted(models)}")
    if models != set(model_baseline):
        raise JoinBuildError(f"candidate/baseline frontend rosters differ: {sorted(models)} != {sorted(model_baseline)}")

    source_by_role_model_method: dict[tuple[str, str, str], dict[tuple[Any, ...], tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]] = {}
    for model_id, source in model_baseline.items():
        for method in ("identity", "rms_fixed"):
            source_by_role_model_method[("baseline", model_id, method)] = {}
        plan_by_id = {str(row["row_id"]): row for row in source["planned"]}
        results_by_id = {str(row["row_id"]): row for row in source["results"]}
        visuals_by_id = {str(row["row_id"]): row for row in source["visual_rows"]}
        for row in source["planned"]:
            method = str(row.get("method_id"))
            if method not in ("identity", "rms_fixed"):
                continue
            key = group_key(row)
            table = source_by_role_model_method[("baseline", model_id, method)]
            if key in table:
                raise JoinBuildError(f"duplicate baseline context/method: {model_id}/{method}/{key}")
            row_id = str(row["row_id"])
            table[key] = (plan_by_id[row_id], results_by_id[row_id], visuals_by_id[row_id])

    candidate_table: dict[tuple[Any, ...], tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = {}
    c_by_id = {str(row["row_id"]): row for row in candidate["results"]}
    cv_by_id = {str(row["row_id"]): row for row in candidate["visual_rows"]}
    for row in candidate["planned"]:
        key = group_key(row)
        if key in candidate_table:
            raise JoinBuildError(f"duplicate candidate context: {key}")
        rid = str(row["row_id"])
        candidate_table[key] = (row, c_by_id[rid], cv_by_id[rid])
    expected_context_count = 60
    if len(candidate_table) != 180:
        raise JoinBuildError(f"CONTROL_FAILED_FIXED_CANDIDATE_CONTEXT_DENOMINATOR: expected 180 unique contexts, found {len(candidate_table)}")
    candidate_contexts_by_model = {
        model_id: {key for key in candidate_table if str(key[3]) == model_id}
        for model_id in models
    }
    if any(len(candidate_contexts_by_model[model_id]) != expected_context_count for model_id in models):
        counts = {model_id: len(candidate_contexts_by_model[model_id]) for model_id in sorted(models)}
        raise JoinBuildError(f"CONTROL_FAILED_FIXED_CONTEXTS_PER_MODEL: expected 60 each, found {counts}")
    for model_id in sorted(models):
        for method in ("identity", "rms_fixed"):
            actual = set(source_by_role_model_method[("baseline", model_id, method)])
            expected = candidate_contexts_by_model[model_id]
            if actual != expected:
                raise JoinBuildError(
                    f"baseline contexts do not exactly match step-120 candidate for {model_id}/{method}: "
                    f"missing={len(expected - actual)} extra={len(actual - expected)}"
                )

    method_source: dict[str, dict[str, Any]] = {}
    for source in baselines:
        for method in source["plan"].get("methods", []):
            if isinstance(method, dict) and method.get("method_id") in ("identity", "rms_fixed"):
                method_source.setdefault(str(method["method_id"]), method)
                if canonical(method_source[str(method["method_id"])]) != canonical(method):
                    raise JoinBuildError(f"baseline method definition differs across frontends: {method.get('method_id')}")
    candidate_method = next(m for m in candidate["plan"]["methods"] if m.get("method_id") == REQUIRED_METHODS[-1])
    method_source[REQUIRED_METHODS[-1]] = candidate_method
    if set(method_source) != set(REQUIRED_METHODS):
        raise JoinBuildError("joined method definitions are incomplete")

    model_source: dict[str, dict[str, Any]] = {}
    for source in baselines:
        for model in source["plan"].get("models", []):
            model_source[str(model["model_id"])] = model
    for model in candidate["plan"].get("models", []):
        model_id = str(model["model_id"])
        if model_id in model_source and any(model_source[model_id].get(field) != model.get(field) for field in MODEL_BINDING_FIELDS):
            raise JoinBuildError(f"candidate/baseline model identity or scoring/frontend pins differ for {model_id}")
        model_source.setdefault(model_id, model)
    if set(model_source) != models:
        raise JoinBuildError("joined model registry differs from expected frontends")

    joined_plan_rows: list[dict[str, Any]] = []
    joined_results: list[dict[str, Any]] = []
    joined_visual: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {"sources": []}
    for source in sources:
        provenance["sources"].append({"role": source["entry"].get("role"), "model_id": source["entry"].get("model_id"),
                                      "source_run_id": source["plan"].get("run_id"), **source["source_pins"]})
    contexts = list(candidate_table.keys())
    for key in contexts:
        model_id = str(key[3])
        per_method = {
            "identity": source_by_role_model_method[("baseline", model_id, "identity")].get(key),
            "rms_fixed": source_by_role_model_method[("baseline", model_id, "rms_fixed")].get(key),
            REQUIRED_METHODS[-1]: candidate_table.get(key),
        }
        if any(value is None for value in per_method.values()):
            missing = [method for method, value in per_method.items() if value is None]
            raise JoinBuildError(f"missing planned rows for context {key}: {missing}")
        refs = [per_method[method] for method in REQUIRED_METHODS]
        planned = [item[0] for item in refs]
        results = [item[1] for item in refs]
        visuals = [copy.deepcopy(item[2]) for item in refs]
        natural_refs = [row.get("natural_pcm", {}) for row in planned]
        score_refs = [row.get("score_audio", {}) for row in planned]
        ref_fields = ("file_sha256", "pcm_sha256", "sample_count", "sample_rate_hz", "channels", "subtype")
        natural_signatures = {tuple(ref.get(field) for field in ref_fields) for ref in natural_refs}
        score_signatures = {tuple(ref.get(field) for field in ref_fields) for ref in score_refs}
        face_hashes = {row.get("source_face", {}).get("file_sha256") for row in planned}
        video_hashes = {row.get("source_video", {}).get("file_sha256") for row in planned}
        if len(natural_signatures) != 1 or natural_signatures != score_signatures or len(face_hashes) != 1 or len(video_hashes) != 1:
            raise JoinBuildError(f"methods do not share natural PCM/score audio/portrait/source video: {key}")
        for planned_row, row in zip(planned, results, strict=True):
            if row.get("status") == "PASS":
                score = row.get("score", {})
                natural_sha = planned_row.get("natural_pcm", {}).get("pcm_sha256")
                if (row.get("natural_pcm_sha256") != natural_sha or row.get("score_audio_sha256") != natural_sha or
                        score.get("score_audio_pcm_sha256") != natural_sha):
                    raise JoinBuildError(f"SyncNet audio is not the shared natural PCM: {row.get('row_id')}")
        visual_clocks = [visual.get("render_clock", {}) for visual in visuals]
        all_visual_pass = all(visual.get("status") == "PASS" for visual in visuals)
        clock_signatures = {clock.get("clock_signature") for clock in visual_clocks}
        clock_shapes = {(clock.get("frame_count"), clock.get("fps_num"), clock.get("fps_den")) for clock in visual_clocks}
        if all_visual_pass and (len(clock_signatures) != 1 or len(clock_shapes) != 1):
            raise JoinBuildError(f"methods do not share the same rational visual clock: {key}")
        for planned_row, result_row, visual_row in zip(planned, results, visuals, strict=True):
            if visual_row.get("status") == "PASS":
                original_ref = visual_row.get("visual_npz", {}).get("path")
                source_root = next(source["visual_dir"] for source in sources
                                   if visual_row["row_id"] in {str(r["row_id"]) for r in source["visual_rows"]})
                original_path = Path(str(original_ref))
                if not original_path.is_absolute():
                    original_path = source_root / original_path
                original_path = original_path.resolve(strict=True)
                if sha_file(original_path) != visual_row["visual_npz"].get("file_sha256"):
                    raise JoinBuildError(f"trajectory source SHA mismatch: {visual_row.get('row_id')}")
            joined_plan_rows.append(planned_row)
            joined_results.append(result_row)
            joined_visual.append(visual_row)

    joined_method_counts = Counter(str(row.get("method_id")) for row in joined_plan_rows)
    if not (len(joined_plan_rows) == len(joined_results) == len(joined_visual) == 540):
        raise JoinBuildError("CONTROL_FAILED_FIXED_JOIN_DENOMINATOR: expected 540 planned/result/visual rows")
    if joined_method_counts != Counter({method: 180 for method in REQUIRED_METHODS}):
        raise JoinBuildError(f"CONTROL_FAILED_FIXED_JOIN_METHOD_COUNTS: {dict(joined_method_counts)}")
    if len({str(row.get("row_id")) for row in joined_plan_rows}) != 540:
        raise JoinBuildError("CONTROL_FAILED_DUPLICATE_JOIN_ROW_ID: expected 540 unique joined row IDs")

    out = Path(args.out).resolve()
    if out.exists():
        raise JoinBuildError(f"fresh output directory already exists: {out}")
    out.mkdir(parents=True)
    (out / "visual_motion").mkdir()
    (out / "visual_motion" / "trajectories").mkdir()

    derived_plan = copy.deepcopy(candidate["plan"])
    derived_plan["run_id"] = "cross_tfg_audio_head_20261003_step120_v3_visual_speech_join_attempt01"
    derived_plan["methods"] = [method_source[method] for method in REQUIRED_METHODS]
    derived_plan["models"] = [model_source[model_id] for model_id in sorted(models)]
    derived_plan["rows"] = joined_plan_rows
    derived_plan["source_visual_speech_join"] = {
        "schema_version": SCHEMA,
        "aggregation_input_only": True,
        "source_runs": provenance["sources"],
        "selected_methods": list(REQUIRED_METHODS),
        "planned_rows": len(joined_plan_rows),
        "rows_per_method": {method: sum(row.get("method_id") == method for row in joined_plan_rows) for method in REQUIRED_METHODS},
        "rule": "copy exact original identity/rms_fixed/step120-candidate rows with matching model/sample/speaker/face/seed; no rerender or SyncNet rescore",
    }
    derived_plan.pop("plan_sha256", None)
    derived_plan["plan_sha256"] = payload_sha(derived_plan, "plan_sha256")
    plan_out = out / "plan.json"
    plan_out.write_bytes(canonical(derived_plan) + b"\n")
    results_out = out / "results.jsonl"
    with results_out.open("x", encoding="utf-8", newline="\n") as stream:
        for row in joined_results:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
    status_counts = Counter(str(row.get("status")) for row in joined_results)
    failure_reasons = Counter(
        str(row.get("failure_reason", "UNSPECIFIED_NONPASS"))
        for row in joined_results if row.get("status") != "PASS"
    )
    counts_by_model_method = []
    for model_id in sorted(models):
        for method_id in REQUIRED_METHODS:
            members = [row for row in joined_results if row.get("model_id") == model_id and row.get("method_id") == method_id]
            counts_by_model_method.append({"model_id": model_id, "method_id": method_id, "planned_rows": len(members),
                                           "result_rows": len(members), "status_counts": {state: sum(row.get("status") == state for row in members) for state in ("PASS", "FAILED", "INCOMPLETE")}})
    result_manifest = {
        "schema_version": PLAN_SCHEMA, "run_id": derived_plan["run_id"],
        "status": "PASS" if status_counts["FAILED"] == 0 and status_counts["INCOMPLETE"] == 0 else "INCOMPLETE",
        "plan_raw_sha256": sha_file(plan_out), "plan_sha256": derived_plan["plan_sha256"],
        "results_jsonl_sha256": sha_file(results_out), "planned_rows": len(joined_plan_rows),
        "result_rows": len(joined_results),
        "status_counts": {state: int(status_counts[state]) for state in ("PASS", "FAILED", "INCOMPLETE")},
        "counts_by_model_method": counts_by_model_method,
        "failure_reasons": [{"reason": reason, "count": count} for reason, count in sorted(failure_reasons.items())],
        "source_visual_speech_join": derived_plan["source_visual_speech_join"],
    }
    result_manifest["manifest_sha256"] = payload_sha(result_manifest, "manifest_sha256")
    manifest_out = out / "result_manifest.json"
    manifest_out.write_bytes(canonical(result_manifest) + b"\n")

    visual_out = out / "visual_motion"
    visual_jsonl = visual_out / "visual_motion_results.jsonl"
    visual_status_counts = Counter(str(row.get("status")) for row in joined_visual)
    file_inventory = []
    for planned_row, visual_row in zip(joined_plan_rows, joined_visual, strict=True):
        if visual_row.get("status") != "PASS":
            continue
        ref = visual_row["visual_npz"]
        source = next(source for source in sources if visual_row["row_id"] in {str(r["row_id"]) for r in source["visual_rows"]})
        source_path = Path(str(ref["path"]))
        if not source_path.is_absolute():
            source_path = source["visual_dir"] / source_path
        source_path = source_path.resolve(strict=True)
        safe_stem = hashlib.sha256(str(visual_row["row_id"]).encode("utf-8")).hexdigest()[:24]
        relative = Path("trajectories") / f"{safe_stem}.npz"
        destination = visual_out / relative
        shutil.copyfile(source_path, destination)
        if sha_file(destination) != ref.get("file_sha256"):
            raise JoinBuildError(f"copied trajectory differs from its sealed source: {visual_row.get('row_id')}")
        visual_row["visual_npz"]["path"] = relative.as_posix()
        file_inventory.append({"path": relative.as_posix(), "bytes": destination.stat().st_size, "sha256": sha_file(destination)})
    with visual_jsonl.open("x", encoding="utf-8", newline="\n") as stream:
        for row in joined_visual:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
    file_inventory.append({"path": "visual_motion_results.jsonl", "bytes": visual_jsonl.stat().st_size, "sha256": sha_file(visual_jsonl)})
    visual_status = "PASS" if visual_status_counts["PASS"] == len(joined_visual) else "INCOMPLETE"
    visual_audit = {
        "schema_version": VISUAL_SCHEMA, "status": visual_status, "run_id": derived_plan["run_id"],
        "plan_path": str(plan_out), "plan_raw_sha256": sha_file(plan_out), "plan_payload_sha256": derived_plan["plan_sha256"],
        "results_path": str(results_out), "results_raw_sha256": sha_file(results_out),
        "result_manifest_path": str(manifest_out), "result_manifest_raw_sha256": sha_file(manifest_out),
        "result_manifest_payload_sha256": result_manifest["manifest_sha256"],
        "planned_rows": len(joined_plan_rows), "result_rows": len(joined_results),
        "visual_status_counts": dict(visual_status_counts), "accounted_for_planned_rows": len(joined_visual),
        "feature_source_sha256": EXPECTED_FEATURE_SOURCE_SHA256,
        "landmarker_asset_sha256": EXPECTED_LANDMARKER_ASSET_SHA256,
        "source_visual_runtime": source_visual_runtime, "effects_selected_from_visual_metrics": False,
        "interpretation": "derived row join of independently sealed visual trajectories; natural PCM motion association input only",
    }
    visual_audit_path = visual_out / "audit.json"
    visual_audit_path.write_bytes(canonical(visual_audit) + b"\n")
    file_inventory.append({"path": "audit.json", "bytes": visual_audit_path.stat().st_size, "sha256": sha_file(visual_audit_path)})
    visual_manifest = {"schema_version": VISUAL_SCHEMA, "status": visual_status, "run_id": derived_plan["run_id"], "files": file_inventory}
    visual_manifest["manifest_sha256"] = payload_sha(visual_manifest, "manifest_sha256")
    visual_manifest_path = visual_out / "visual_motion_manifest.json"
    visual_manifest_path.write_bytes(canonical(visual_manifest) + b"\n")

    receipt = {
        "schema_version": SCHEMA, "status": "DERIVED_INPUT_CREATED",
        "input_manifest_path": str(input_path), "input_manifest_raw_sha256": sha_file(input_path),
        "output_directory": str(out), "plan_raw_sha256": sha_file(plan_out),
        "results_raw_sha256": sha_file(results_out), "result_manifest_raw_sha256": sha_file(manifest_out),
        "visual_manifest_raw_sha256": sha_file(visual_manifest_path), "planned_rows": len(joined_plan_rows),
        "status_counts": {state: int(status_counts[state]) for state in ("PASS", "FAILED", "INCOMPLETE")},
        "visual_status_counts": dict(visual_status_counts), "copied_trajectory_count": len(file_inventory) - 2,
        "source_run_count": len(sources), "confirmation_accessed": False,
        "rendering_or_syncnet_rescore": False,
        "next_step": "run the frozen speech-motion association on this derived plan/results/visual input with original natural PCM artifacts",
    }
    (out / "join_receipt.json").write_bytes(canonical(receipt) + b"\n")
    return 0 if result_manifest["status"] == "PASS" and visual_status == "PASS" else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        return run(args)
    except JoinBuildError as exc:
        print(f"CONTROL_FAILED:{exc}")
        return 2
    except Exception as exc:
        print(f"visual_speech_motion_join_failed:{type(exc).__name__}:{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
