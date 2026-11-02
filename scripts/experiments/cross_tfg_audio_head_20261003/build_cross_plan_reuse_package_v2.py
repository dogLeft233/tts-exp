#!/usr/bin/env python3
"""Build a candidate-only dev plan and a derived, cross-plan joint analysis package.

The source StageB run is read-only. Candidate artifacts are freshly generated.
The later joint package keeps source result/media files at their original paths,
binds exact source JSONL-line hashes, and rebuilds five-method support summaries.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA = "cross-tfg-audio-eval-v1"
SOURCE_METHODS = ("identity", "rms_fixed", "rms_localcompress", "spectral_head_fit24")
IDENTITY_FIELDS = (
    "row_id", "split", "sample_id", "speaker_id", "model_id", "audio_frontend",
    "model_role", "face_id", "seed", "method_id", "method_config_sha256", "checkpoint_sha256",
)
PASS_ARTIFACT_PATHS = (
    ("generator_pcm",),
    ("render", "generated_video_path", "generated_video_sha256"),
    ("render", "muxed_video_path", "muxed_video_sha256"),
    ("render", "silent_video_path", "silent_video_sha256"),
    ("render", "selected_track_crop", "path", "file_sha256"),
    ("render", "selected_track_crop", "frame_map_path", "frame_map_sha256"),
    ("render", "timestamps_path", "timestamps_sha256"),
    ("render", "crop_roi_path", "crop_roi_sha256"),
    ("score", "primary_pcm16_window", "path", "file_sha256"),
    ("score", "primary_syncnet_input", "path", "file_sha256"),
    ("score", "raw_matrix_npz", "path", "file_sha256"),
)


class ReusePlanError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _seal(obj: Mapping[str, Any], field: str) -> str:
    body = dict(obj)
    body.pop(field, None)
    return _sha(_canonical(body))


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReusePlanError(f"cannot parse {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReusePlanError(f"{label} must be a JSON object")
    return value


def _get_path(obj: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    value: Any = obj
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _artifact_reuse_audit(result: Mapping[str, Any]) -> dict[str, Any]:
    """Check actual artifact refs for a PASS result; failures remain planned."""
    issues: list[str] = []
    checked: list[dict[str, Any]] = []
    cache: dict[str, tuple[bool, str | None]] = {}
    for keys in PASS_ARTIFACT_PATHS:
        value = _get_path(result, keys)
        if keys == ("generator_pcm",):
            if not isinstance(value, Mapping):
                issues.append("missing generator_pcm ref")
                continue
            path, expected = value.get("path"), value.get("file_sha256")
        else:
            parent = _get_path(result, keys[:-2])
            path = parent.get(keys[-2]) if isinstance(parent, Mapping) else None
            expected = parent.get(keys[-1]) if isinstance(parent, Mapping) else None
        if not isinstance(path, str) or not isinstance(expected, str) or len(expected) != 64:
            issues.append(f"missing path/file SHA ref at {'.'.join(keys)}")
            continue
        if path not in cache:
            p = Path(path)
            if not p.is_file():
                cache[path] = (False, None)
            else:
                actual = _sha_file(p)
                cache[path] = (actual == expected, actual)
        ok, actual = cache[path]
        checked.append({"path": path, "expected_sha256": expected, "actual_sha256": actual, "ok": ok})
        if not ok:
            issues.append(f"artifact hash/path check failed at {'.'.join(keys)}")
    score = result.get("score") if isinstance(result.get("score"), Mapping) else {}
    mux = result.get("mux_audio_readback") if isinstance(result.get("mux_audio_readback"), Mapping) else {}
    if score.get("status") != "PASS":
        issues.append("score status is not PASS")
    if mux.get("status") != "PASS" or mux.get("matches_natural_canonical_pcm") is not True or mux.get("video_stream_identity_verified") is not True:
        issues.append("natural PCM mux/video-stream identity did not PASS")
    if result.get("score_audio_sha256") != result.get("natural_pcm_sha256"):
        issues.append("score audio hash is not the natural PCM hash")
    return {"eligible_for_reuse": not issues, "issues": issues, "checked_artifacts": checked}


def _load_run(plan_path: Path, results_path: Path, manifest_path: Path, repeat_path: Path, label: str) -> dict[str, Any]:
    plan = _load_json(plan_path, f"{label} plan")
    manifest = _load_json(manifest_path, f"{label} result manifest")
    plan_raw_sha, results_raw_sha, repeat_raw_sha = _sha_file(plan_path), _sha_file(results_path), _sha_file(repeat_path)
    if plan.get("schema_version") != SCHEMA or plan.get("status") != "FROZEN":
        raise ReusePlanError(f"{label} plan is not a frozen {SCHEMA} plan")
    if plan.get("plan_sha256") != _seal(plan, "plan_sha256"):
        raise ReusePlanError(f"{label} embedded plan seal mismatch")
    if manifest.get("schema_version") != SCHEMA or manifest.get("run_id") != plan.get("run_id"):
        raise ReusePlanError(f"{label} manifest schema/run id mismatch")
    if manifest.get("manifest_sha256") != _seal(manifest, "manifest_sha256"):
        raise ReusePlanError(f"{label} manifest payload seal mismatch")
    expected_manifest_bindings = {
        "plan_sha256": plan.get("plan_sha256"), "plan_raw_sha256": plan_raw_sha,
        "results_jsonl_sha256": results_raw_sha, "generation_repeat_qc_jsonl_sha256": repeat_raw_sha,
    }
    for key, expected in expected_manifest_bindings.items():
        if manifest.get(key) != expected:
            raise ReusePlanError(f"{label} manifest does not bind {key}")
    planned_rows = plan.get("rows")
    if not isinstance(planned_rows, list):
        raise ReusePlanError(f"{label} plan rows are malformed")
    results, line_sha = [], {}
    with results_path.open("rb") as stream:
        for line_no, raw_line in enumerate(stream, 1):
            if not raw_line.strip():
                continue
            try:
                result = json.loads(raw_line)
            except Exception as exc:
                raise ReusePlanError(f"{label} result JSONL line {line_no} is invalid: {exc}") from exc
            if not isinstance(result, dict):
                raise ReusePlanError(f"{label} result JSONL line {line_no} is not an object")
            if len(results) >= len(planned_rows) or result.get("row_id") != planned_rows[len(results)].get("row_id"):
                raise ReusePlanError(f"{label} result order/id mismatch at line {line_no}")
            results.append(result)
            line_sha[result["row_id"]] = _sha(raw_line)
    if len(results) != len(planned_rows):
        raise ReusePlanError(f"{label} row count differs from the plan")
    counts = Counter(row.get("status") for row in results)
    for field in ("planned_rows", "result_rows"):
        if manifest.get(field) != len(planned_rows):
            raise ReusePlanError(f"{label} manifest {field} mismatch")
    if manifest.get("status_counts") != {s: int(counts.get(s, 0)) for s in ("PASS", "FAILED", "INCOMPLETE")}:
        raise ReusePlanError(f"{label} manifest status counts mismatch")
    repeat_rows = [json.loads(line) for line in repeat_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(repeat_rows) != len(plan.get("generation_repeat_qc", [])) or manifest.get("generation_repeat_qc_rows") != len(repeat_rows):
        raise ReusePlanError(f"{label} repeat-QC row count mismatch")
    return {
        "plan": plan, "manifest": manifest, "results": results, "result_by_id": {r["row_id"]: r for r in results},
        "line_sha_by_id": line_sha, "repeat_rows": repeat_rows, "plan_path": plan_path,
        "results_path": results_path, "manifest_path": manifest_path, "repeat_path": repeat_path,
        "plan_raw_sha256": plan_raw_sha, "results_raw_sha256": results_raw_sha,
        "manifest_raw_sha256": _sha_file(manifest_path), "repeat_raw_sha256": repeat_raw_sha,
    }


def _context(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(k) for k in ("split", "sample_id", "speaker_id", "model_id", "face_id", "seed"))


def _candidate_method(spec_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = _load_json(spec_path, "candidate method spec")
    if spec.get("schema_version") != "cross_tfg_candidate_method_spec_v1":
        raise ReusePlanError("candidate method spec schema mismatch")
    if spec.get("fit_scope") != "historical_fit_only" or spec.get("fit_data_splits") != ["fit"]:
        raise ReusePlanError("candidate fit provenance must explicitly be historical fit-only")
    method = spec.get("method")
    if not isinstance(method, dict) or not isinstance(method.get("config"), dict):
        raise ReusePlanError("candidate spec requires a complete method record with config")
    if method.get("method_id") in SOURCE_METHODS:
        raise ReusePlanError("candidate method id must be new and distinct from all four source methods")
    if method.get("config_sha256") != _sha(_canonical(method["config"])):
        raise ReusePlanError("candidate method config SHA mismatch")
    for name in ("fit_plan", "fit_report", "checkpoint"):
        ref = spec.get(name)
        if not isinstance(ref, dict) or not isinstance(ref.get("path"), str) or not isinstance(ref.get("file_sha256"), str):
            raise ReusePlanError(f"candidate spec is missing {name} path/file_sha256")
        p = Path(ref["path"])
        if not p.is_file() or _sha_file(p) != ref["file_sha256"]:
            raise ReusePlanError(f"candidate {name} file is missing or has wrong SHA256")
    ckpt = spec["checkpoint"]
    if method.get("checkpoint_sha256") != ckpt["file_sha256"] or method["config"].get("checkpoint_sha256") != ckpt["file_sha256"]:
        raise ReusePlanError("candidate method checkpoint pins disagree with checkpoint file")
    if method["config"].get("checkpoint_path") != ckpt["path"]:
        raise ReusePlanError("candidate method checkpoint path differs from the supplied checkpoint ref")
    if method["config"].get("fit_plan_raw_sha256") != spec["fit_plan"].get("file_sha256"):
        raise ReusePlanError("candidate method does not bind the supplied fit plan")
    if method["config"].get("fit_report_sha256") != spec["fit_report"].get("file_sha256"):
        raise ReusePlanError("candidate method does not bind the supplied fit report")
    return spec, method


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def _write_plan_bundle(out: Path, plan: dict[str, Any], rows: list[dict[str, Any]], manifest: dict[str, Any],
                       sidecars: Mapping[str, bytes] | None = None) -> None:
    if out.exists():
        raise FileExistsError(f"refusing existing output directory: {out}")
    out.mkdir(parents=True)
    plan_bytes = (json.dumps(plan, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    rows_bytes = b"".join(_canonical(row) + b"\n" for row in rows)
    manifest["plan_raw_sha256"] = _sha(plan_bytes)
    manifest["planned_rows_sha256"] = _sha(rows_bytes)
    manifest["planned_rows"] = len(rows)
    manifest["method_counts"] = {m["method_id"]: sum(r["method_id"] == m["method_id"] for r in rows) for m in plan["methods"]}
    manifest["manifest_sha256"] = _seal(manifest, "manifest_sha256")
    _write_exclusive(out / "plan.json", plan_bytes)
    _write_exclusive(out / "planned_rows.jsonl", rows_bytes)
    for name, raw in (sidecars or {}).items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ReusePlanError(f"unsafe sidecar filename: {name}")
        _write_exclusive(out / name, raw)
    _write_exclusive(out / "plan_manifest.json", (json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode())


def build_candidate_plan(args: argparse.Namespace) -> int:
    source = _load_run(args.source_plan, args.source_results, args.source_manifest, args.source_repeat_qc, "source StageB")
    source_plan = source["plan"]
    methods = [m.get("method_id") for m in source_plan.get("methods", [])]
    if tuple(methods) != SOURCE_METHODS or len(source_plan.get("rows", [])) != 240:
        raise ReusePlanError(f"source plan method/row inventory differs from frozen StageB source: {methods}/{len(source_plan.get('rows', []))}")
    if any(row.get("split") != "dev" for row in source_plan["rows"]):
        raise ReusePlanError("candidate-only source plan must be development-only")
    spec, method = _candidate_method(args.candidate_method_spec)
    if method["method_id"] in methods:
        raise ReusePlanError("candidate method id already exists in source plan")
    if source_plan.get("generation_seed_family", {}).get("dev") != [20261003, 20261004, 20261005]:
        raise ReusePlanError("source dev generation-seed family differs from frozen three-seed family")
    by_context: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    context_order: list[tuple[Any, ...]] = []
    for planned in source_plan["rows"]:
        key = _context(planned)
        if key not in by_context:
            context_order.append(key)
        by_context[key][planned["method_id"]] = planned
    if len(context_order) != 60:
        raise ReusePlanError(f"expected 20 clips × 3 seeds = 60 contexts, got {len(context_order)}")
    row_id_to_result = source["result_by_id"]
    candidate_rows: list[dict[str, Any]] = []
    source_reuse_audit: list[dict[str, Any]] = []
    for key in context_order:
        context_methods = by_context[key]
        if set(context_methods) != set(SOURCE_METHODS):
            raise ReusePlanError(f"source method rows are incomplete for context {key}")
        identity_ref = context_methods["identity"]
        static_fields = ("natural_pcm", "score_audio", "source_face", "source_video", "audio_frontend", "model_id", "model_role", "face_id", "seed")
        for method_id in SOURCE_METHODS:
            row = context_methods[method_id]
            for field in static_fields:
                if row.get(field) != identity_ref.get(field):
                    raise ReusePlanError(f"source context differs in immutable {field} across methods: {key}/{method_id}")
            settings = row.get("generation_settings", {})
            identity_settings = identity_ref.get("generation_settings", {})
            for field in ("face_mode", "fps_num", "fps_den", "generator", "input_audio", "pre_mel_peak_normalization", "seed"):
                if settings.get(field) != identity_settings.get(field):
                    raise ReusePlanError(f"source context differs in frozen generation setting {field}: {key}/{method_id}")
        identity = context_methods["identity"]
        candidate = copy.deepcopy(identity)
        candidate["row_id"] = identity["row_id"].rsplit("_identity", 1)[0] + "_" + method["method_id"]
        candidate["method_id"] = method["method_id"]
        candidate["method_config_sha256"] = method["config_sha256"]
        candidate["checkpoint_sha256"] = method.get("checkpoint_sha256")
        candidate["generation_settings"] = {
            **candidate.get("generation_settings", {}),
            "audio_transform": method.get("config", {}).get("method"),
            "fit_report_sha256": spec["fit_report"]["file_sha256"],
            "head_checkpoint_sha256": spec["checkpoint"]["file_sha256"],
        }
        reuse_rows = []
        for method_id in SOURCE_METHODS:
            src_plan = context_methods[method_id]
            src_result = row_id_to_result[src_plan["row_id"]]
            for field in IDENTITY_FIELDS:
                if src_result.get(field) != src_plan.get(field):
                    raise ReusePlanError(f"source row/result identity mismatch {src_plan['row_id']}.{field}")
            status = src_result.get("status")
            audit = _artifact_reuse_audit(src_result) if status == "PASS" else {
                "eligible_for_reuse": False,
                "issues": [f"source result status is {status}"],
                "checked_artifacts": [],
            }
            ref = {
                "method_id": method_id,
                "source_row_id": src_plan["row_id"],
                "source_result_line_sha256": source["line_sha_by_id"][src_plan["row_id"]],
                "source_result_status": status,
                "eligible_for_reuse": audit["eligible_for_reuse"],
                "reuse_audit_sha256": _sha(_canonical(audit)),
            }
            reuse_rows.append(ref)
            source_reuse_audit.append({"context_key": list(key), **ref, "reuse_audit": audit})
        candidate["cross_plan_reuse"] = {
            "schema_version": "cross_tfg_cross_plan_reuse_row_v1",
            "source_run_id": source_plan["run_id"],
            "source_plan_raw_sha256": source["plan_raw_sha256"],
            "source_plan_payload_sha256": source_plan["plan_sha256"],
            "source_result_manifest_raw_sha256": source["manifest_raw_sha256"],
            "source_result_manifest_payload_sha256": source["manifest"]["manifest_sha256"],
            "source_results_jsonl_sha256": source["results_raw_sha256"],
            "source_repeat_qc_raw_sha256": source["repeat_raw_sha256"],
            "context_key": {field: val for field, val in zip(("split", "sample_id", "speaker_id", "model_id", "face_id", "seed"), key)},
            "reused_rows": reuse_rows,
            "reuse_policy": "reuse_only_exact_hash_bound_PASS_rows; preserve all source plan denominators and files",
        }
        candidate_rows.append(candidate)
    plan = copy.deepcopy(source_plan)
    plan["run_id"] = args.run_id
    plan["methods"] = [copy.deepcopy(method)]
    plan["rows"] = candidate_rows
    plan["generation_repeat_qc"] = []
    plan["contrasts"] = []
    plan["analysis"] = copy.deepcopy(plan.get("analysis", {}))
    plan["analysis"]["contrasts"] = []
    plan["analysis"]["summary_endpoints"] = []
    plan["analysis"]["execution_only_candidate_subrun"] = True
    plan["analysis"]["efficacy_requires_joint_analysis"] = True
    plan["config_sha256"] = _sha(_canonical({
        "source_plan_raw_sha256": source["plan_raw_sha256"],
        "candidate_method_config_sha256": method["config_sha256"],
        "candidate_checkpoint_sha256": method["checkpoint_sha256"],
        "run_id": args.run_id,
    }))
    plan["execution"] = {
        **plan.get("execution", {}),
        "stage": "C_candidate_only_generation_and_scoring",
        "scope": "dev-only 60-row new candidate; baseline controls reused by hash-bound cross-plan lineage; no confirmation audio/effects read",
        "planned_rows": 60,
        "source_run_id": source_plan["run_id"],
        "source_plan_raw_sha256": source["plan_raw_sha256"],
        "source_result_manifest_raw_sha256": source["manifest_raw_sha256"],
        "source_results_jsonl_sha256": source["results_raw_sha256"],
        "source_repeat_qc_raw_sha256": source["repeat_raw_sha256"],
        "candidate_method_spec_raw_sha256": _sha_file(args.candidate_method_spec),
        "candidate_checkpoint_sha256": method["checkpoint_sha256"],
        "candidate_fit_report_sha256": spec["fit_report"]["file_sha256"],
        "reuse_policy": "actual final PASS rows only after exact plan/manifest/line/artifact hash verification; failed planned rows stay in combined denominator",
        "analysis_scope": "candidate-only subrun is not an efficacy result; use joint five-method package",
    }
    audit_payload = {"schema_version":"cross_tfg_source_reuse_audit_v1", "source_run_id":source_plan["run_id"],
        "source_plan_raw_sha256":source["plan_raw_sha256"], "source_result_manifest_raw_sha256":source["manifest_raw_sha256"],
        "source_results_jsonl_sha256":source["results_raw_sha256"], "rows":source_reuse_audit}
    audit_bytes=(json.dumps(audit_payload,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    plan["cross_plan_reuse"] = {
        "schema_version": "cross_tfg_cross_plan_reuse_v1",
        "source_run_id": source_plan["run_id"],
        "source_plan_raw_sha256": source["plan_raw_sha256"],
        "source_result_manifest_raw_sha256": source["manifest_raw_sha256"],
        "source_results_jsonl_sha256": source["results_raw_sha256"],
        "source_repeat_qc_raw_sha256": source["repeat_raw_sha256"],
        "baseline_methods": list(SOURCE_METHODS),
        "candidate_method_id": method["method_id"],
        "source_reuse_audit_path": "reuse_audit.json",
        "source_reuse_audit_raw_sha256": _sha(audit_bytes),
        "source_reuse_audit_rows": len(source_reuse_audit),
        "source_four_method_estimand_is_separate": True,
    }
    plan["status"] = "FROZEN"
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = _seal(plan, "plan_sha256")
    manifest = {
        "schema": "cross_tfg_candidate_only_plan_bundle_v1", "run_id": args.run_id,
        "plan_sha256": plan["plan_sha256"], "source_run_id": source_plan["run_id"],
        "source_plan_raw_sha256": source["plan_raw_sha256"],
        "source_result_manifest_raw_sha256": source["manifest_raw_sha256"],
        "source_results_jsonl_sha256": source["results_raw_sha256"],
        "source_repeat_qc_raw_sha256": source["repeat_raw_sha256"],
        "candidate_method_id": method["method_id"], "candidate_checkpoint_sha256": method["checkpoint_sha256"],
        "candidate_fit_report_sha256": spec["fit_report"]["file_sha256"],
        "confirm_audio_read": False, "planned_rows": 60,
        "source_reuse_audit_raw_sha256": _sha(audit_bytes),
    }
    _write_plan_bundle(args.output_dir, plan, candidate_rows, manifest, {"reuse_audit.json":audit_bytes})
    print(json.dumps({"status":"FROZEN", "run_id":args.run_id, "planned_rows":60,
        "method_id":method["method_id"], "plan_sha256":plan["plan_sha256"],
        "plan_raw_sha256":_sha_file(args.output_dir/"plan.json"),
        "source_run_id":source_plan["run_id"], "eligible_source_rows":sum(x["eligible_for_reuse"] for x in source_reuse_audit),
        "source_rows":len(source_reuse_audit), "confirm_audio_read":False},sort_keys=True))
    return 0


def _load_runner(path: Path):
    if not path.is_file():
        raise ReusePlanError(f"pinned source runner not found: {path}")
    spec = importlib.util.spec_from_file_location("cross_tfg_pinned_stageb_runner", path)
    if spec is None or spec.loader is None:
        raise ReusePlanError("cannot import pinned StageB runner for common-support reconstruction")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result_matrix_ref(result: Mapping[str, Any]) -> tuple[Path, str]:
    ref = _get_path(result, ("score", "raw_matrix_npz"))
    if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str) or not isinstance(ref.get("file_sha256"), str):
        raise ReusePlanError(f"missing raw_matrix_npz file ref for {result.get('row_id')}")
    path = Path(ref["path"])
    if not path.is_file() or _sha_file(path) != ref["file_sha256"]:
        raise ReusePlanError(f"raw_matrix_npz SHA mismatch for {result.get('row_id')}")
    return path, ref["file_sha256"]


def _contrast_policy(policy_path: Path, source_plan: Mapping[str, Any], candidate_method_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policy = _load_json(policy_path, "joint analysis policy")
    if policy.get("schema_version") != "cross_tfg_joint_analysis_policy_v1" or policy.get("status") != "FROZEN":
        raise ReusePlanError("joint analysis policy must be frozen before constructing the joint package")
    contrasts = policy.get("contrasts")
    if not isinstance(contrasts, list) or not contrasts:
        raise ReusePlanError("joint analysis policy must declare contrasts")
    expected_pairs = {
        (candidate_method_id, "identity", "main"),
        (candidate_method_id, "rms_fixed", "main"),
        (candidate_method_id, "rms_localcompress", "main"),
        (candidate_method_id, "spectral_head_fit24", "diagnostic"),
    }
    actual_pairs = {(x.get("method_a"), x.get("method_b"), x.get("role")) for x in contrasts if isinstance(x, dict)}
    if actual_pairs != expected_pairs or len(contrasts) != len(expected_pairs):
        raise ReusePlanError(f"joint contrasts must be the four frozen candidate-vs-control pairs; got {actual_pairs}")
    if policy.get("bootstrap") != source_plan.get("analysis", {}).get("bootstrap"):
        raise ReusePlanError("joint policy must inherit the source plan's exact PCG bootstrap draws/settings")
    if policy.get("aggregation") != source_plan.get("analysis", {}).get("aggregation"):
        raise ReusePlanError("joint policy must inherit the source plan's exact seed/sample/speaker aggregation")
    if policy.get("supports") != source_plan.get("supports"):
        raise ReusePlanError("joint policy support definitions differ from the frozen source plan")
    endpoints = policy.get("summary_endpoints")
    if not isinstance(endpoints, list):
        raise ReusePlanError("joint policy must freeze summary_endpoints")
    expected = {(c["contrast_id"], s, "Cbest") for c in contrasts for s in ("full", "guard20")}
    actual = {(e.get("contrast_id"), e.get("support_id"), e.get("metric")) for e in endpoints if isinstance(e, dict)}
    if actual != expected or len(endpoints) != len(expected):
        raise ReusePlanError("joint policy endpoints must cover each declared contrast × FULL/GUARD20 × Cbest exactly once")
    for e in endpoints:
        if e.get("split") != "dev" or e.get("exploratory") is not True or e.get("audio_estimand") != "native_roi_original_pcm16" or e.get("roi_estimand") != "native_official":
            raise ReusePlanError("joint development endpoints must be exploratory dev/native official ROI/original PCM16")
    return copy.deepcopy(contrasts), copy.deepcopy(endpoints)


def build_joint_package(args: argparse.Namespace) -> int:
    source = _load_run(args.source_plan, args.source_results, args.source_manifest, args.source_repeat_qc, "source StageB")
    candidate = _load_run(args.candidate_plan, args.candidate_results, args.candidate_manifest, args.candidate_repeat_qc, "candidate-only")
    source_plan, candidate_plan = source["plan"], candidate["plan"]
    lineage = candidate_plan.get("cross_plan_reuse")
    expected_lineage = {
        "source_run_id": source_plan.get("run_id"), "source_plan_raw_sha256": source["plan_raw_sha256"],
        "source_result_manifest_raw_sha256": source["manifest_raw_sha256"],
        "source_results_jsonl_sha256": source["results_raw_sha256"], "source_repeat_qc_raw_sha256": source["repeat_raw_sha256"],
    }
    if not isinstance(lineage, dict) or any(lineage.get(k) != v for k, v in expected_lineage.items()):
        raise ReusePlanError("candidate plan cross-plan source bindings do not match finalized source run")
    if candidate_plan.get("generation_repeat_qc"):
        raise ReusePlanError("candidate-only run must not generate a duplicate model-level repeat QC")
    if candidate_plan.get("methods") is None or len(candidate_plan["methods"]) != 1:
        raise ReusePlanError("candidate-only plan must have exactly one new method")
    candidate_method = candidate_plan["methods"][0]
    candidate_method_id = candidate_method["method_id"]
    if candidate_method_id in SOURCE_METHODS:
        raise ReusePlanError("candidate-only method id overlaps source method roster")
    if len(candidate_plan.get("rows", [])) != 60 or len(candidate["results"]) != 60:
        raise ReusePlanError("candidate-only plan and final result ledger must each contain 60 rows")
    contrasts, endpoints = _contrast_policy(args.analysis_policy, source_plan, candidate_method_id)

    # Rebuild context order with all five methods adjacent. Source files remain untouched.
    source_plan_by_id = {r["row_id"]: r for r in source_plan["rows"]}
    source_result_by_id = source["result_by_id"]
    candidate_plan_by_context = {_context(r): r for r in candidate_plan["rows"]}
    candidate_result_by_id = candidate["result_by_id"]
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    context_order: list[tuple[Any, ...]] = []
    for row in source_plan["rows"]:
        key = _context(row)
        if key not in grouped:
            context_order.append(key)
        grouped[key][row["method_id"]] = row
    if set(candidate_plan_by_context) != set(context_order) or len(candidate_plan_by_context) != 60:
        raise ReusePlanError("candidate-only context set differs from the exact frozen 60 source contexts")
    joint_plan_rows: list[dict[str, Any]] = []
    joint_results: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    for key in context_order:
        base_context = grouped[key]
        if set(base_context) != set(SOURCE_METHODS):
            raise ReusePlanError(f"source context method set mismatch for {key}")
        cand_plan_row = candidate_plan_by_context[key]
        if cand_plan_row.get("method_id") != candidate_method_id:
            raise ReusePlanError(f"candidate method id mismatch for {key}")
        reuse_refs = (cand_plan_row.get("cross_plan_reuse") or {}).get("reused_rows", [])
        reuse_by_method = {x.get("method_id"): x for x in reuse_refs if isinstance(x, dict)}
        if set(reuse_by_method) != set(SOURCE_METHODS):
            raise ReusePlanError(f"candidate row lacks one exact cross-plan row reference per source method: {key}")
        context_rows = [base_context[m] for m in SOURCE_METHODS] + [cand_plan_row]
        joint_plan_rows.extend(copy.deepcopy(context_rows))
        for plan_row in context_rows:
            is_candidate = plan_row["method_id"] == candidate_method_id
            if is_candidate:
                result = copy.deepcopy(candidate_result_by_id[plan_row["row_id"]])
                source_line_sha = candidate["line_sha_by_id"][plan_row["row_id"]]
                source_run = candidate_plan["run_id"]
            else:
                result = copy.deepcopy(source_result_by_id[plan_row["row_id"]])
                source_line_sha = source["line_sha_by_id"][plan_row["row_id"]]
                source_run = source_plan["run_id"]
                ref = reuse_by_method[plan_row["method_id"]]
                if ref.get("source_row_id") != plan_row["row_id"] or ref.get("source_result_line_sha256") != source_line_sha:
                    raise ReusePlanError(f"candidate plan's source row/line hash binding differs for {plan_row['row_id']}")
                if result.get("status") == "PASS" and ref.get("eligible_for_reuse") is not True:
                    result["status"] = "INCOMPLETE"
                    result["failure_reason"] = "source result was PASS but its cross-plan artifact reuse audit failed"
                    result.setdefault("score", {})["status"] = "INCOMPLETE"
                elif result.get("status") == "PASS":
                    current_audit = _artifact_reuse_audit(result)
                    if current_audit.get("eligible_for_reuse") is not True:
                        result["status"] = "INCOMPLETE"
                        result["failure_reason"] = "source artifacts no longer pass the cross-plan reuse audit"
                        result.setdefault("score", {})["status"] = "INCOMPLETE"
            result["cross_plan_provenance"] = {
                "source_run_id": source_run,
                "source_result_line_sha256": source_line_sha,
                "derived_joint_run_id": args.run_id,
                "source_artifacts_remain_authoritative": True,
            }
            if result.get("status") == "PASS":
                try:
                    matrix_path, _ = _result_matrix_ref(result)
                    with np.load(matrix_path, allow_pickle=False) as z:
                        result["_internal_arrays"] = {name: np.asarray(z[name]).copy() for name in (
                            "query_frame_index", "query_timestamp_num", "query_timestamp_den", "track_frame_index",
                            "valid_query_mask", "distance_by_query_lag", "lag_offsets_frames")}
                except Exception as exc:
                    result["status"] = "INCOMPLETE"
                    result["failure_reason"] = f"cross-plan raw matrix reuse verification failed: {type(exc).__name__}: {exc}"
                    result.setdefault("score", {})["status"] = "INCOMPLETE"
            lineage_rows.append({"row_id": plan_row["row_id"], "method_id": plan_row["method_id"],
                "source_run_id": source_run, "source_result_line_sha256": source_line_sha,
                "reuse_status": result.get("status")})
            joint_results.append(result)

    joint_plan = copy.deepcopy(source_plan)
    joint_plan["run_id"] = args.run_id
    joint_plan["methods"] = copy.deepcopy(source_plan["methods"]) + [copy.deepcopy(candidate_method)]
    joint_plan["rows"] = joint_plan_rows
    joint_plan["contrasts"] = contrasts
    joint_plan["analysis"] = copy.deepcopy(source_plan["analysis"])
    joint_plan["analysis"]["contrasts"] = contrasts
    joint_plan["analysis"]["summary_endpoints"] = endpoints
    joint_plan["analysis"]["source_four_method_estimand_run_id"] = source_plan["run_id"]
    joint_plan["analysis"]["source_four_method_estimand_separate"] = True
    joint_plan["execution"] = {
        **source_plan.get("execution", {}),
        "stage": "D_joint_analysis_cross_plan_reuse",
        "scope": "dev-only derived five-method joint analysis; source four-method StageB estimand remains separate; no confirmation audio/effects read",
        "planned_rows": 300,
        "source_run_id": source_plan["run_id"],
        "candidate_run_id": candidate_plan["run_id"],
        "source_plan_raw_sha256": source["plan_raw_sha256"],
        "source_result_manifest_raw_sha256": source["manifest_raw_sha256"],
        "source_results_jsonl_sha256": source["results_raw_sha256"],
        "source_repeat_qc_raw_sha256": source["repeat_raw_sha256"],
        "candidate_plan_raw_sha256": candidate["plan_raw_sha256"],
        "candidate_result_manifest_raw_sha256": candidate["manifest_raw_sha256"],
        "candidate_results_jsonl_sha256": candidate["results_raw_sha256"],
        "candidate_repeat_qc_raw_sha256": candidate["repeat_raw_sha256"],
        "reuse_policy": "only exact source PASS rows with valid artifact and source-line hashes contribute; all planned failed rows remain in the 300-row ledger",
        "support_policy": "recompute FULL and GUARD20 independently over the five-method common valid query intersection for each sample/model/face/seed",
        "joint_result_lineage_sha256": _sha(_canonical(lineage_rows)),
    }
    lineage_payload={
        "schema_version": "cross_tfg_joint_lineage_v1",
        "provenance_kind": "DERIVED_JOINT_ANALYSIS_CROSS_PLAN_REUSE",
        "source_run_id": source_plan["run_id"],
        "candidate_run_id": candidate_plan["run_id"],
        "rows": lineage_rows,
    }
    lineage_bytes=(json.dumps(lineage_payload,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    joint_plan["cross_plan_lineage"] = {
        "schema_version": "cross_tfg_joint_lineage_v1",
        "provenance_kind": "DERIVED_JOINT_ANALYSIS_CROSS_PLAN_REUSE",
        "source_four_method_estimand_run_id": source_plan["run_id"],
        "source_plan_raw_sha256": source["plan_raw_sha256"],
        "source_manifest_raw_sha256": source["manifest_raw_sha256"],
        "source_results_raw_sha256": source["results_raw_sha256"],
        "source_repeat_qc_raw_sha256": source["repeat_raw_sha256"],
        "candidate_plan_raw_sha256": candidate["plan_raw_sha256"],
        "candidate_manifest_raw_sha256": candidate["manifest_raw_sha256"],
        "candidate_results_raw_sha256": candidate["results_raw_sha256"],
        "lineage_sidecar_path": "cross_plan_lineage.json",
        "lineage_sidecar_raw_sha256": _sha(lineage_bytes),
        "lineage_rows": len(lineage_rows),
        "source_npz_and_media_refs_remain_authoritative": True,
    }
    joint_plan["config_sha256"] = _sha(_canonical({
        "source_plan_raw_sha256": source["plan_raw_sha256"],
        "candidate_plan_raw_sha256": candidate["plan_raw_sha256"],
        "analysis_policy_raw_sha256": _sha_file(args.analysis_policy),
        "joint_run_id": args.run_id,
    }))
    joint_plan["status"] = "FROZEN"
    joint_plan.pop("plan_sha256", None)
    joint_plan["plan_sha256"] = _seal(joint_plan, "plan_sha256")

    runner = _load_runner(args.pinned_runner)
    runner._finish_supports(joint_plan, joint_results)
    # The writer strips only private arrays; cross-plan provenance stays visible.
    status_counts = Counter(r.get("status") for r in joint_results)
    method_counts = []
    methods_by_id = [m["method_id"] for m in joint_plan["methods"]]
    for model in joint_plan["models"]:
        for method_id in methods_by_id:
            subset = [r for r in joint_results if r.get("model_id") == model["model_id"] and r.get("method_id") == method_id]
            method_counts.append({"model_id":model["model_id"],"method_id":method_id,"planned_rows":len(subset),"result_rows":len(subset),
                "status_counts":{s:int(sum(r.get("status")==s for r in subset)) for s in ("PASS","FAILED","INCOMPLETE")}})
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    plan_bytes=(json.dumps(joint_plan,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    result_bytes=b"".join(_canonical({k:v for k,v in row.items() if not k.startswith("_internal_")})+b"\n" for row in joint_results)
    repeat_bytes=source["repeat_path"].read_bytes()
    _write_exclusive(args.output_dir/"plan.json",plan_bytes)
    _write_exclusive(args.output_dir/"results.jsonl",result_bytes)
    _write_exclusive(args.output_dir/"generation_repeat_qc.jsonl",repeat_bytes)
    repeat_record=source["repeat_rows"][0] if source["repeat_rows"] else None
    result_manifest={
        "schema_version":SCHEMA,"run_id":args.run_id,
        "status":"PASS" if status_counts["FAILED"]==0 and status_counts["INCOMPLETE"]==0 and (repeat_record is None or repeat_record.get("status") in ("PASS","NONDETERMINISTIC")) else "INCOMPLETE",
        "plan_sha256":joint_plan["plan_sha256"],"plan_raw_sha256":_sha(plan_bytes),
        "results_jsonl_sha256":_sha(result_bytes),"generation_repeat_qc_jsonl_sha256":_sha(repeat_bytes),
        "planned_rows":len(joint_plan_rows),"result_rows":len(joint_results),
        "generation_repeat_qc_rows":len(joint_plan.get("generation_repeat_qc",[])),
        "status_counts":{s:int(status_counts.get(s,0)) for s in ("PASS","FAILED","INCOMPLETE")},
        "counts_by_model_method":method_counts,
        "failure_reasons":[{"row_id":r["row_id"],"reason":r.get("failure_reason")} for r in joint_results if r.get("status")!="PASS"],
        "repeat_qc_status":repeat_record.get("status") if repeat_record else "NOT_RUN_SUBRUN",
        "provenance_kind":"DERIVED_JOINT_ANALYSIS_CROSS_PLAN_REUSE",
        "source_plan_raw_sha256":source["plan_raw_sha256"],"source_result_manifest_raw_sha256":source["manifest_raw_sha256"],
        "source_results_jsonl_sha256":source["results_raw_sha256"],"source_repeat_qc_raw_sha256":source["repeat_raw_sha256"],
        "candidate_plan_raw_sha256":candidate["plan_raw_sha256"],"candidate_result_manifest_raw_sha256":candidate["manifest_raw_sha256"],
        "candidate_results_jsonl_sha256":candidate["results_raw_sha256"],
        "joint_row_lineage_sha256":_sha(_canonical(lineage_rows)),
    }
    result_manifest["manifest_sha256"]=_seal(result_manifest,"manifest_sha256")
    _write_exclusive(args.output_dir/"result_manifest.json",(json.dumps(result_manifest,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode())
    plan_manifest={"schema":"cross_tfg_joint_analysis_bundle_v1","run_id":args.run_id,
        "plan_sha256":joint_plan["plan_sha256"],"plan_raw_sha256":_sha(plan_bytes),
        "results_jsonl_sha256":_sha(result_bytes),"result_manifest_sha256":_sha_file(args.output_dir/"result_manifest.json"),
        "source_run_id":source_plan["run_id"],"candidate_run_id":candidate_plan["run_id"],
        "provenance_kind":"DERIVED_JOINT_ANALYSIS_CROSS_PLAN_REUSE","planned_rows":len(joint_plan_rows),
        "status_counts":result_manifest["status_counts"],"confirm_audio_read":False}
    plan_manifest["manifest_sha256"]=_seal(plan_manifest,"manifest_sha256")
    _write_exclusive(args.output_dir/"cross_plan_lineage.json",lineage_bytes)
    _write_exclusive(args.output_dir/"joint_manifest.json",(json.dumps(plan_manifest,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode())
    print(json.dumps({"status":result_manifest["status"],"run_id":args.run_id,"planned_rows":300,
        "status_counts":result_manifest["status_counts"],"method_counts":{m:sum(r["method_id"]==m for r in joint_plan_rows) for m in methods_by_id},
        "plan_sha256":joint_plan["plan_sha256"],"results_jsonl_sha256":_sha(result_bytes),
        "result_manifest_sha256":_sha_file(args.output_dir/"result_manifest.json"),
        "provenance_kind":"DERIVED_JOINT_ANALYSIS_CROSS_PLAN_REUSE","confirm_audio_read":False},sort_keys=True))
    return 0 if result_manifest["status"]=="PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    candidate=sub.add_parser("candidate-plan",help="seal a new 60-row candidate-only dev plan")
    for name in ("source-plan","source-results","source-manifest","source-repeat-qc","candidate-method-spec","output-dir"):
        candidate.add_argument("--"+name,type=Path,required=True)
    candidate.add_argument("--run-id",required=True)
    candidate.set_defaults(func=build_candidate_plan)
    joint=sub.add_parser("joint-package",help="build a derived 300-row five-method analysis package")
    for name in ("source-plan","source-results","source-manifest","source-repeat-qc","candidate-plan","candidate-results","candidate-manifest","candidate-repeat-qc","analysis-policy","pinned-runner","output-dir"):
        joint.add_argument("--"+name,type=Path,required=True)
    joint.add_argument("--run-id",required=True)
    joint.set_defaults(func=build_joint_package)
    return parser


def main(argv: list[str] | None=None) -> int:
    args=build_parser().parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(json.dumps({"status":"FAILED","error_type":type(exc).__name__,"error":str(exc)},sort_keys=True),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
