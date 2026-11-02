#!/usr/bin/env python3
"""Analyze Dev20 with one frozen, selector-admitted Fit SPSA center.

This additive entry point keeps the attempt06 cohort, thresholds, 540-row
denominator, score consumer, visual gate, ASR gate, endpoint reduction, and
speaker bootstrap. It accepts only the separate selected-center plan and its
pre-effect freeze. Results and visual/ASR effects are not opened until the
freeze, selection artifact, application manifest, and all waveform receipts
have passed their pinned source checks.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

BUILDER_REL = Path("scripts/experiments/cross_tfg_audio_head_20261003/build_future_dev20_selected_spsa_plan_attempt03.py")
BUILDER_BYTES = 65301
BUILDER_RAW_SHA256 = "e8a17138aed8d0d87bd38bec82faa4dd0598806f7fb7f841f0d9717e63c6a6bd"
LEGACY_ANALYZER_REL = Path("scripts/experiments/cross_tfg_audio_head_20261003/analyze_future_dev20_admission_attempt01.py")
LEGACY_ANALYZER_BYTES = 63521
LEGACY_ANALYZER_RAW_SHA256 = "88603111675fe75053f1079abde29a9a8c38d3727f6fba93a603e932d97f8f24"
POLICY_REL = Path("openspec/changes/develop-cross-tfg-audio-head-20261003/future-dev20-admission-policy-attempt01.json")
POLICY_RAW_SHA256 = "f215ecff1dc9908eebfc3195a96d76b041bea47ca0fa7df5d64b6481031398ed"
PROTOCOL_REL = Path(".codex/scratch/dev20_future_cohort_protocol_attempt06/future_dev20_three_method_protocol_attempt06.json")
PROTOCOL_RAW_SHA256 = "73ac37eee2c8c728a7123fa85834dc53104a4b809cbc017593bd86939816eb17"
PORTABLE_REL = Path("scripts/audio_enhancement/portable_selected_fit_center_attempt05.py")
PORTABLE_RAW_SHA256 = "29bc744d250924221373a9244a735f3f7c3b2285d1b0d97357e6eda7a965f987"
PLAN_BINDING_SCHEMA = "cross_tfg_future_dev20_selected_spsa_center_plan_binding_v1"
OVERLAY_SCHEMA = "cross_tfg_future_dev20_selected_spsa_center_protocol_v1"
FREEZE_SCHEMA = "cross_tfg_future_dev20_selected_spsa_center_pre_effect_freeze_v1"
FREEZE_STATUS = "FROZEN_BEFORE_DEV20_SCORE_ACCESS"
SELECTION_STATUS = "ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV"
CONDITION_ID = "selected_spsa_center"
METHOD_KIND = "fit40_spsa_center"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[A-Za-z0-9._/-]{7,128}$")
UTC_RE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z$")


class SelectedAdmissionError(RuntimeError):
    pass


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_source(root: Path, relative: Path, expected_bytes: int, expected_sha: str) -> tuple[Path, bytes]:
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise SelectedAdmissionError(f"pinned source is not a regular non-symlink file: {relative}")
    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    if len(raw) != resolved.stat().st_size or (len(raw), _sha(raw)) != (expected_bytes, expected_sha):
        raise SelectedAdmissionError(f"pinned source bytes/SHA differ: {relative}")
    return resolved, raw


def _load_source_module(root: Path, relative: Path, module_name: str,
                        expected_bytes: int, expected_sha: str) -> Any:
    path, raw = _read_source(root, relative, expected_bytes, expected_sha)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SelectedAdmissionError(f"cannot load pinned source module: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    actual = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
    if actual != path or path.read_bytes() != raw:
        raise SelectedAdmissionError(f"loaded pinned source changed/origin differs: {relative}")
    return module


def _load_json(legacy: Any, path: Path, label: str) -> tuple[dict[str, Any], bytes, Path]:
    try:
        doc, raw, resolved = legacy.read_json(path, max_bytes=128 * 1024 * 1024)
    except Exception as exc:
        raise SelectedAdmissionError(f"{label} read failed: {type(exc).__name__}: {exc}") from exc
    return doc, raw, resolved


def _same_ref(actual: Mapping[str, Any], expected: Mapping[str, Any], *, self_field: str | None = None) -> bool:
    keys = ["path", "bytes", "sha256"] + (["self_sha256"] if self_field else [])
    return all(actual.get(key) == expected.get(key) for key in keys)


def _verify_pre_effect_binding(root: Path, legacy: Any, builder: Any, plan_path: Path,
                               freeze_path: Path) -> dict[str, Any]:
    plan, plan_raw, resolved_plan = _load_json(legacy, plan_path, "selected-center plan")
    if (plan.get("schema_version") != legacy.EXPECTED_EVAL_SCHEMA
            or plan.get("status") != "FROZEN"
            or plan.get("plan_sha256") != legacy.payload_sha(plan, "plan_sha256")):
        raise SelectedAdmissionError("selected-center plan schema/status/canonical self-hash is invalid")
    binding = plan.get("dev20_selected_spsa_center")
    if (not isinstance(binding, dict) or binding.get("schema") != PLAN_BINDING_SCHEMA
            or binding.get("selection_status") != SELECTION_STATUS
            or binding.get("checkpoint_sha256") is not None
            or binding.get("freeze_required_before_any_effect") is not True
            or binding.get("score_effects_read") is not False
            or binding.get("confirm_data_or_effects_accessed") is not False):
        raise SelectedAdmissionError("plan does not carry the additive selected-center pre-effect binding")

    artifact_ref = binding.get("selected_artifact")
    app_ref = binding.get("application_manifest")
    base_ref = binding.get("identity_rms_base_plan")
    overlay_ref = binding.get("protocol_overlay")
    if any(not isinstance(ref, dict) for ref in (artifact_ref, app_ref, base_ref, overlay_ref)):
        raise SelectedAdmissionError("plan binding omits a required selected artifact/application/base/overlay ref")
    overlay_path = Path(str(overlay_ref.get("path", "")))
    overlay, overlay_raw, resolved_overlay = _load_json(legacy, overlay_path, "selected-center protocol overlay")
    if (not _same_ref({"path": str(resolved_overlay), "bytes": len(overlay_raw), "sha256": _sha(overlay_raw),
                       "self_sha256": overlay.get("self_sha256")}, overlay_ref, self_field="self_sha256")
            or overlay.get("schema") != OVERLAY_SCHEMA
            or overlay.get("status") != "FROZEN_SELECTED_CENTER_BEFORE_DEV20_RENDER_OR_SCORE"
            or overlay.get("selection_status") != SELECTION_STATUS
            or overlay.get("self_sha256") != legacy.payload_sha(overlay, "self_sha256")
            or overlay.get("score_effects_read") is not False
            or overlay.get("confirm_data_or_effects_accessed") is not False
            or overlay.get("native_tfg_qualification_claim") is not False
            or overlay.get("final_dev_or_confirm_admission_performed") is not False):
        raise SelectedAdmissionError("selected-center overlay raw/self/source status does not match its plan ref")
    if (overlay.get("selected_spsa_center") != binding.get("selected_center")
            or overlay.get("application_manifest", {}).get("sha256") != app_ref.get("sha256")
            or overlay.get("portable_artifact", {}).get("sha256") != artifact_ref.get("sha256")
            or overlay.get("identity_rms_base_plan", {}).get("sha256") != base_ref.get("sha256")):
        raise SelectedAdmissionError("overlay and plan binding disagree on selected artifact/application/base plan")

    source_protocol, protocol_raw, policy, policy_raw = builder._verify_source_protocol(root)
    if (overlay.get("source_protocol_attempt06", {}).get("sha256") != PROTOCOL_RAW_SHA256
            or overlay.get("source_protocol_attempt06", {}).get("bytes") != len(protocol_raw)
            or overlay.get("source_policy_attempt01", {}).get("sha256") != POLICY_RAW_SHA256
            or overlay.get("source_policy_attempt01", {}).get("bytes") != len(policy_raw)
            or plan.get("cohort_and_source") != source_protocol.get("cohort_and_source")):
        raise SelectedAdmissionError("overlay/plan no longer bind the frozen attempt06 cohort and policy")

    selected = binding.get("selected_center")
    if not isinstance(selected, dict):
        raise SelectedAdmissionError("plan selected-center descriptor is missing")
    artifact_path = Path(str(artifact_ref.get("path", "")))
    portable = builder._load_portable(root)
    artifact, artifact_raw, artifact_selected, _evidence = builder._validate_selected_artifact(
        root, portable, artifact_path)
    if (not _same_ref({"path": str(artifact_path.resolve(strict=True)), "bytes": len(artifact_raw),
                       "sha256": _sha(artifact_raw)}, artifact_ref)
            or artifact.get("artifact_sha256") != artifact_ref.get("artifact_sha256")
            or artifact_selected != selected
            or artifact.get("artifact_sha256") != selected.get("artifact_sha256")):
        raise SelectedAdmissionError("portable selected artifact differs from the planned raw/self pins")
    source_rows, source_by_id = builder._source_rows(source_protocol)
    base_path = Path(str(base_ref.get("path", "")))
    base_plan, base_raw, _base_rows = builder._validate_base_plan(root, base_path, source_protocol, source_by_id)
    if (len(base_raw) != base_ref.get("bytes") or _sha(base_raw) != base_ref.get("sha256")
            or base_plan.get("plan_sha256") != base_ref.get("self_sha256")):
        raise SelectedAdmissionError("identity/RMS source plan raw/self pins changed after freeze")
    app_path = Path(str(app_ref.get("path", "")))
    app_manifest, app_raw, _applications = builder._validate_application_manifest(
        root, app_path, source_protocol, source_by_id, portable,
        artifact_path, artifact, artifact_raw, selected)
    if (len(app_raw) != app_ref.get("bytes") or _sha(app_raw) != app_ref.get("sha256")
            or app_manifest.get("self_sha256") != app_ref.get("self_sha256")):
        raise SelectedAdmissionError("application manifest/receipts/output WAV pins changed after freeze")

    method_id, method = legacy.discover_candidate_method(plan)
    config = method.get("config") if isinstance(method, dict) else None
    if (method_id != selected.get("method_id") or method.get("kind") != METHOD_KIND
            or method.get("checkpoint_sha256") is not None or not isinstance(config, dict)
            or config.get("method") != METHOD_KIND or config.get("schema") != "fit40_spsa_center_transform_v1"
            or config.get("point_kind") != "spsa_center" or config.get("source_step") not in (5, 10, 15, 20)
            or config.get("theta_sha256") != selected.get("theta_sha256")
            or config.get("artifact_sha256") != selected.get("artifact_sha256")
            or config.get("binding_sha256") != selected.get("binding_sha256")
            or config.get("input_pins_sha256") != selected.get("input_pins_sha256")
            or method.get("config_sha256") != legacy.sha_bytes(legacy.canonical_json(config))
            or method.get("config_sha256") != selected.get("config_sha256")):
        raise SelectedAdmissionError("candidate method is not the exact admitted selected SPSA center")
    if len(plan.get("rows", [])) != 540:
        raise SelectedAdmissionError("selected-center plan is not the complete frozen 540-row denominator")

    freeze, freeze_raw, resolved_freeze = _load_json(legacy, freeze_path, "pre-effect freeze receipt")
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("status") != FREEZE_STATUS
            or freeze.get("selection_status") != SELECTION_STATUS
            or freeze.get("self_sha256") != legacy.payload_sha(freeze, "self_sha256")
            or freeze.get("score_effects_read") is not False
            or freeze.get("confirm_data_or_effects_accessed") is not False
            or freeze.get("dev_or_confirm_admission_performed") is not False
            or freeze.get("point_kind") != "spsa_center"
            or freeze.get("source_step") != selected.get("source_step")
            or freeze.get("candidate_method_id") != selected.get("method_id")
            or freeze.get("method_config_sha256") != method.get("config_sha256")
            or freeze.get("checkpoint_sha256") is not None
            or freeze.get("theta_sha256") != selected.get("theta_sha256")
            or freeze.get("artifact_sha256") != selected.get("artifact_sha256")
            or freeze.get("binding_sha256") != selected.get("binding_sha256")
            or freeze.get("input_pins_sha256") != selected.get("input_pins_sha256")
            or not isinstance(freeze.get("frozen_commit"), str)
            or not COMMIT_RE.fullmatch(freeze["frozen_commit"])
            or not isinstance(freeze.get("frozen_at_utc"), str)
            or not UTC_RE.fullmatch(freeze["frozen_at_utc"])):
        raise SelectedAdmissionError("pre-effect freeze receipt is invalid or refers to another selected center")
    plan_pin = {"path": str(resolved_plan), "bytes": len(plan_raw), "sha256": _sha(plan_raw),
                "self_sha256": plan.get("plan_sha256")}
    overlay_pin = {"path": str(resolved_overlay), "bytes": len(overlay_raw), "sha256": _sha(overlay_raw),
                   "self_sha256": overlay.get("self_sha256")}
    if (freeze.get("plan") != plan_pin or freeze.get("protocol_overlay") != overlay_pin
            or freeze.get("selected_artifact") != artifact_ref
            or freeze.get("application_manifest") != app_ref
            or freeze.get("source_protocol_attempt06") != binding.get("source_protocol_attempt06")
            or freeze.get("source_policy_attempt01") != binding.get("source_policy_attempt01")):
        raise SelectedAdmissionError("freeze receipt does not bind the exact plan/overlay/artifact/application inputs")
    builder_path, builder_raw = _read_source(root, BUILDER_REL, BUILDER_BYTES, BUILDER_RAW_SHA256)
    builder_ref = freeze.get("builder_source")
    if builder_ref != {"path": BUILDER_REL.as_posix(), "bytes": len(builder_raw), "sha256": _sha(builder_raw)}:
        raise SelectedAdmissionError("freeze receipt does not bind the exact additive plan-builder source")

    # The historical verifier labels the third condition with the old DirectD0
    # name. Normalize only this in-memory design row so its unchanged roster
    # validator can check the exact same selected method and 27-job matrix.
    compatibility_protocol = copy.deepcopy(overlay)
    design = compatibility_protocol.get("primary_design", {})
    jobs = design.get("run_jobs", [])
    if not isinstance(jobs, list) or len(jobs) != 27:
        raise SelectedAdmissionError("selected-center overlay does not carry all 27 native jobs")
    for job in jobs:
        if job.get("condition_id") == CONDITION_ID and job.get("method_id") == method_id:
            job["condition_id"] = "direct_d0_step120"
    source_map = legacy.verify_protocol_rows(compatibility_protocol, plan)
    legacy.validate_native_analysis_protocol(plan)
    return {
        "plan": plan, "plan_raw": plan_raw, "plan_path": resolved_plan,
        "overlay": overlay, "overlay_raw": overlay_raw, "overlay_path": resolved_overlay,
        "freeze": freeze, "freeze_raw": freeze_raw, "freeze_path": resolved_freeze,
        "policy": policy, "source_protocol": source_protocol, "source_map": source_map,
        "candidate_id": method_id, "candidate_method": method,
        "builder_source": {"path": BUILDER_REL.as_posix(), "bytes": len(builder_raw), "sha256": _sha(builder_raw)},
        "legacy_analyzer_source": {"path": LEGACY_ANALYZER_REL.as_posix(),
                                   "bytes": LEGACY_ANALYZER_BYTES,
                                   "sha256": LEGACY_ANALYZER_RAW_SHA256},
        "portable_source": {"path": PORTABLE_REL.as_posix(), "bytes": builder.PORTABLE_BYTES,
                            "sha256": PORTABLE_RAW_SHA256},
        "selected_artifact": {"path": str(artifact_path.resolve(strict=True)), "bytes": len(artifact_raw),
                              "sha256": _sha(artifact_raw), "artifact_sha256": artifact.get("artifact_sha256")},
        "application_manifest": {"path": str(app_path.resolve(strict=True)), "bytes": len(app_raw),
                                 "sha256": _sha(app_raw), "self_sha256": app_manifest.get("self_sha256")},
        "base_plan": {"path": str(base_path.resolve(strict=True)), "bytes": len(base_raw),
                      "sha256": _sha(base_raw), "self_sha256": base_plan.get("plan_sha256")},
        "overlay_compatibility": "condition label normalized only in memory for the pinned attempt01 roster helper; result identities remain selected method ID",
    }


def _verify_selected_asr(legacy: Any, args: argparse.Namespace, context: Mapping[str, Any],
                         result_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_calls_path = args.asr_candidate_calls.resolve(strict=True)
    original_loader = legacy.load_jsonl

    def selected_condition_loader(path: Path, *, max_bytes: int = 1 << 34):
        rows, raw, malformed = original_loader(path, max_bytes=max_bytes)
        try:
            same_file = Path(path).resolve(strict=True) == candidate_calls_path
        except OSError:
            same_file = False
        if not same_file:
            return rows, raw, malformed
        adapted: list[dict[str, Any]] = []
        for row in rows:
            if row.get("condition") != CONDITION_ID:
                raise SelectedAdmissionError("candidate ASR ledger condition does not name selected_spsa_center")
            copy_row = dict(row)
            copy_row["condition"] = "direct_d0_step120"
            adapted.append(copy_row)
        return adapted, raw, malformed

    legacy.load_jsonl = selected_condition_loader
    try:
        evidence = legacy.verify_asr(
            context["source_protocol"], context["source_map"], context["plan"], result_rows,
            context["candidate_id"], args.asr_baseline_summary, args.asr_baseline_calls,
            args.asr_baseline_payload_manifest, args.asr_candidate_summary,
            args.asr_candidate_calls, args.asr_candidate_payload_manifest,
            args.asr_baseline_payload_root, args.asr_candidate_payload_root,
        )
    finally:
        legacy.load_jsonl = original_loader
    evidence["condition_adapter"] = {
        "input_condition": CONDITION_ID,
        "pinned_analyzer_argument_condition": "direct_d0_step120",
        "adaptation": "in-memory validation compatibility only; source ledger bytes/SHA remain unchanged",
    }
    return evidence


def run(args: argparse.Namespace) -> int:
    out_dir = args.out.resolve(strict=False)
    if out_dir.exists() or out_dir.is_symlink():
        raise SelectedAdmissionError(f"output already exists; use a fresh directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    root = args.root.resolve(strict=True)
    report: dict[str, Any] = {
        "schema": "cross_tfg_future_dev20_selected_spsa_admission_attempt02_v1",
        "status": "INCOMPLETE_NOT_ADMITTED", "policy_status": "FROZEN_PROSPECTIVE_NOT_RUN",
        "input_pins": {}, "row_denominator": {}, "gates": {},
        "score_effects_read": False, "confirm_data_or_effects_accessed": False,
        "failures": [],
        "notes": ["Selected Fit SPSA center is a waveform candidate only; this analyzer never claims a checkpoint or Confirm result."],
    }
    bootstrap_index = None
    try:
        legacy = _load_source_module(root, LEGACY_ANALYZER_REL, "_pinned_future_dev20_analyzer_attempt01",
                                     LEGACY_ANALYZER_BYTES, LEGACY_ANALYZER_RAW_SHA256)
        builder = _load_source_module(root, BUILDER_REL, "_pinned_future_dev20_selected_center_builder_attempt02",
                                      BUILDER_BYTES, BUILDER_RAW_SHA256)
        context = _verify_pre_effect_binding(root, legacy, builder, args.plan, args.pre_effect_freeze)
        policy = context["policy"]
        report["policy_status"] = policy["status"]
        report["input_pins"].update({
            "policy": legacy._file_ref(root / POLICY_REL, (root / POLICY_REL).read_bytes()),
            "source_protocol": legacy.source_pin(root, PROTOCOL_REL, PROTOCOL_RAW_SHA256),
            "legacy_analyzer": context["legacy_analyzer_source"],
            "selected_center_builder": context["builder_source"],
            "portable_selected_center_wrapper": context["portable_source"],
            "selected_artifact": context["selected_artifact"],
            "application_manifest": context["application_manifest"],
            "identity_rms_base_plan": context["base_plan"],
            "protocol_overlay": legacy._file_ref(context["overlay_path"], context["overlay_raw"]),
            "pre_effect_freeze": legacy._file_ref(context["freeze_path"], context["freeze_raw"]),
        })
        report["gates"]["fixed_selected_center_pre_effect_freeze"] = {
            "status": "PASS", "selection_status": SELECTION_STATUS,
            "point_kind": "spsa_center", "source_step": context["freeze"].get("source_step"),
            "candidate_method_id": context["candidate_id"], "checkpoint_sha256": None,
            "candidate_selection_after_effects": False,
            "confirm_data_or_effects_accessed": False,
        }
        report["gates"]["frozen_cohort_and_native_plan"] = {
            "status": "PASS", "planned_rows": 540, "source_clips": 20,
            "speakers": 10, "models": 3, "seeds": 3,
            "conditions": ["identity", "rms_fixed", CONDITION_ID],
            "native_lags": 31, "anchor_column": 15, "supports": ["full", "guard20"],
        }

        # Effect-bearing score/result files are opened only after the full
        # selector, portable application, plan, and freeze gate above passes.
        plan, plan_raw, plan_path = legacy.read_json(args.plan, max_bytes=128 * 1024 * 1024)
        if (plan_path != context["plan_path"] or plan_raw != context["plan_raw"]
                or plan.get("plan_sha256") != context["plan"].get("plan_sha256")):
            raise SelectedAdmissionError("plan changed after pre-effect freeze validation; result access is blocked")
        results, results_raw, malformed = legacy.load_jsonl(args.results, max_bytes=1 << 34)
        report["score_effects_read"] = True
        result_manifest, manifest_raw, manifest_path = legacy.read_json(args.result_manifest, max_bytes=16 * 1024 * 1024)
        if malformed:
            raise SelectedAdmissionError("results ledger contains malformed lines; preserve and reject the full denominator")
        if (result_manifest.get("schema_version") != legacy.EXPECTED_EVAL_SCHEMA
                or result_manifest.get("run_id") != plan.get("run_id")
                or result_manifest.get("manifest_sha256") != legacy.payload_sha(result_manifest, "manifest_sha256")
                or result_manifest.get("plan_raw_sha256") != legacy.sha_bytes(plan_raw)
                or result_manifest.get("results_jsonl_sha256") != legacy.sha_bytes(results_raw)):
            raise SelectedAdmissionError("result manifest does not bind the exact selected-center plan/results bytes")
        counts = Counter(str(row.get("status")) for row in results)
        report["input_pins"].update({
            "plan": legacy._file_ref(plan_path, plan_raw),
            "results": legacy._file_ref(args.results, results_raw),
            "result_manifest": legacy._file_ref(manifest_path, manifest_raw),
        })
        report["row_denominator"] = {
            "planned_rows": len(plan.get("rows", [])), "parsed_result_rows": len(results),
            "malformed_jsonl_lines": 0,
            "status_counts": {name: int(counts.get(name, 0)) for name in ("PASS", "FAILED", "INCOMPLETE", "OTHER")},
            "all_failures_retained": True,
        }
        coverage = legacy.validate_complete_rows(plan, results, context["source_map"], context["candidate_id"])
        report["gates"]["complete_540_rows"] = {"status": "PASS", **coverage}
        if (counts.get("PASS", 0) != 540 or counts.get("FAILED", 0) != 0 or counts.get("INCOMPLETE", 0) != 0
                or result_manifest.get("planned_rows") != 540 or result_manifest.get("result_rows") != 540
                or result_manifest.get("status") != "PASS"):
            raise SelectedAdmissionError("all 540 renderer/score rows must PASS without failures or denominator loss")

        asr = _verify_selected_asr(legacy, args, context, results)
        report["gates"]["reference_free_asr"] = asr
        report["input_pins"]["asr_baseline_summary"] = asr["baseline_summary"]
        report["input_pins"]["asr_baseline_calls"] = asr["baseline_calls"]
        report["input_pins"]["asr_baseline_payload_manifest"] = asr["baseline_payload_manifest"]
        report["input_pins"]["asr_candidate_summary"] = asr["candidate_summary"]
        report["input_pins"]["asr_candidate_calls"] = asr["candidate_calls"]
        report["input_pins"]["asr_candidate_payload_manifest"] = asr["candidate_payload_manifest"]
        if asr.get("status") != "PASS":
            raise SelectedAdmissionError("selected-center reference-free ASR consistency gates failed")

        result_by_id = {str(row["row_id"]): row for row in results}
        visual = legacy.verify_visual(args.visual_manifest, plan, result_manifest,
                                      legacy.sha_bytes(plan_raw), legacy.sha_bytes(manifest_raw),
                                      plan["rows"], result_by_id)
        report["gates"]["independent_visual_tracking"] = visual
        report["input_pins"]["visual_manifest"] = visual["manifest"]
        if visual.get("status") != "PASS":
            raise SelectedAdmissionError("independent visual tracking gate failed")

        consumer = legacy.load_pinned_consumer(root)
        namespace = argparse.Namespace(
            plan=str(args.plan), results=str(args.results), result_manifest=str(args.result_manifest),
            generation_repeat_qc=str(args.generation_repeat_qc), artifact_root=str(args.artifact_root),
            ffmpeg=args.ffmpeg, ffprobe=args.ffprobe,
        )
        state = consumer._load_and_validate_inputs(namespace)
        state["supports"] = {item["support_id"]: item for item in state["plan"]["supports"]}
        rebuilt = consumer._rebuild_all_supports(state)
        report["input_pins"]["generation_repeat_qc"] = legacy._file_ref(
            args.generation_repeat_qc, legacy.read_regular(args.generation_repeat_qc)[1])
        if state["status_counts"] != {"PASS": 540, "FAILED": 0, "INCOMPLETE": 0}:
            raise SelectedAdmissionError("pinned independent consumer did not validate all 540 PASS rows")
        context_records, speaker_values = legacy.summarize_supports(plan, rebuilt, context["candidate_id"])
        report["gates"]["independent_native_pcm_clock_support_consumer"] = {
            "status": "PASS", "consumer_source_sha256": legacy.CONSUMER_RAW_SHA256,
            "result_rows": 540, "paired_contexts": len(context_records),
            "all_original_pcm_mux_clock_native_matrix_checks": True,
            "support_contexts": context_records,
        }
        endpoints, bootstrap, bootstrap_index = legacy.make_endpoints(speaker_values)
        admission = legacy.point_gate_summary(endpoints)
        candidate_method = context["candidate_method"]
        report["candidate_method"] = {
            "method_id": context["candidate_id"],
            "kind": METHOD_KIND,
            "point_kind": "spsa_center",
            "source_step": context["freeze"]["source_step"],
            "config_sha256": candidate_method.get("config_sha256"),
            "checkpoint_sha256": None,
            "theta_sha256": context["freeze"]["theta_sha256"],
            "artifact_sha256": context["freeze"]["artifact_sha256"],
            "binding_sha256": context["freeze"]["binding_sha256"],
        }
        report["endpoints"] = endpoints
        report["bootstrap"] = bootstrap
        report["point_admission"] = admission
        report["status"] = "PASS_DEVELOPMENT_ADMISSION" if admission["status"] == "PASS" else "FAIL_POINT_ADMISSION"
    except Exception as exc:
        report["failures"].append({"type": type(exc).__name__, "message": str(exc)})
        report["status"] = "INCOMPLETE_NOT_ADMITTED"
    legacy_for_emit = locals().get("legacy")
    if legacy_for_emit is not None:
        legacy_for_emit._emit_report(out_dir, report, bootstrap_index)
    else:
        raw = json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
        report["self_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in report.items() if k != "self_sha256"},
                                                           sort_keys=True, separators=(",", ":"),
                                                           ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        raw = json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
        (out_dir / "admission_report.json").write_bytes(raw)
    print(json.dumps({"status": report["status"], "out": str(out_dir),
                      "planned_rows": report.get("row_denominator", {}).get("planned_rows"),
                      "failure_count": len(report["failures"])}, sort_keys=True))
    return 0 if report["status"] == "PASS_DEVELOPMENT_ADMISSION" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pre-effect-freeze", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--result-manifest", type=Path, required=True)
    parser.add_argument("--generation-repeat-qc", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--visual-manifest", type=Path, required=True)
    parser.add_argument("--asr-baseline-summary", type=Path, required=True)
    parser.add_argument("--asr-baseline-calls", type=Path, required=True)
    parser.add_argument("--asr-baseline-payload-manifest", type=Path, required=True)
    parser.add_argument("--asr-baseline-payload-root", type=Path, required=True)
    parser.add_argument("--asr-candidate-summary", type=Path, required=True)
    parser.add_argument("--asr-candidate-calls", type=Path, required=True)
    parser.add_argument("--asr-candidate-payload-manifest", type=Path, required=True)
    parser.add_argument("--asr-candidate-payload-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
