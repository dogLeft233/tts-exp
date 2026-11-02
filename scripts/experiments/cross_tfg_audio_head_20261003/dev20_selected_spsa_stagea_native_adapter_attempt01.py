#!/usr/bin/env python3
"""Run six selected-SPSA-center rows through the frozen native Stage-A core.

This is an additive native execution adapter for the already-frozen Dev20
selected-center plan. It verifies the plan/freeze/application lineage, admits
only one exact six-row slice from the 60-row per-model candidate roster, and
uses the package11 Stage-A ``_render_and_score`` path. The private method copy
selects its existing generator-only precomputed-waveform branch. Stage-A still
receives the original natural PCM for delivery, muxing, source clock, and
SyncNet scoring. This module does not apply the selected transform, select a
center, read Dev/Confirm effects, or compute admission statistics.

The caller must run it under the model's already-pinned lexical runtime and
provide the actual package11 modules/runtime binding. ``render_block`` is
deliberately row-level: a complete call represents exactly six of 60 selected
rows for one model, while all manifests retain the parent 540-row denominator
and the selected 180-row overall/60-row-per-model denominators.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


BUILDER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/build_future_dev20_selected_spsa_plan_attempt04.py"
BUILDER_BYTES = 74982
BUILDER_SHA256 = "36a84af3da6c92c54dc4a8456d50fc804cbbaa6d828dfc10250bb05073554b7c"
ANALYZER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/analyze_future_dev20_selected_spsa_admission_attempt03.py"
ANALYZER_BYTES = 32444
ANALYZER_SHA256 = "5435efe54cd15ea4ded5477b9b76b567daa2b88d5b1c21c86e21e6d6e1a58a25"
PORTABLE_REL = "scripts/audio_enhancement/portable_selected_fit_center_attempt05.py"
PORTABLE_BYTES = 58811
PORTABLE_SHA256 = "29bc744d250924221373a9244a735f3f7c3b2285d1b0d97357e6eda7a965f987"
PORTABLE_APPLY_PIN = {
    "path": "scripts/audio_enhancement/portable_fit_snapshot_apply_attempt03.py",
    "bytes": 32995,
    "sha256": "2e8f0db6ef37a673f7fcadd7ed26a2375eeb94c3a3534e0b0c706bd4338c5fa4",
}
FINAL_INGEST_PIN = {
    "path": "scripts/experiments/cross_tfg_audio_head_20261003/fit_only_final_center_stagea_ingest_attempt05.py",
    "bytes": 106167,
    "sha256": "3384ebe67ed99569af6307c88746a0b16329897dda9935678677dcf80147e8ea",
}
FINAL_INGEST_DEPENDENCY_SOURCE_PINS = {
    "selector": {
        "path": "scripts/experiments/cross_tfg_audio_head_20261003/fit_only_spsa_selection_attempt06.py",
        "bytes": 66767,
        "sha256": "dda610b985026be7f9328c479358e96c983a582c511964e8c6e72a6d398692e2",
        "module_name": "scripts.experiments.cross_tfg_audio_head_20261003.fit_only_spsa_selection_attempt06",
    },
    "initial_ingest": {
        "path": "scripts/experiments/cross_tfg_audio_head_20261003/fit_only_stagea_ingest_attempt03.py",
        "bytes": 53184,
        "sha256": "22cd7f983b22babcae0d9c120f4720cb84a218ec4d2b84551d5e1668d23aeb95",
        "module_name": "scripts.experiments.cross_tfg_audio_head_20261003.fit_only_stagea_ingest_attempt03",
    },
    "packet_builder": {
        "path": "scripts/experiments/cross_tfg_audio_head_20261003/prepare_fit40_spsa_stagea_packets_attempt05.py",
        "bytes": 47748,
        "sha256": "cd5e2b80ab0bdd4f599d863468c687724898b2e6baad451a5acfe2123648bdae",
        "module_name": "scripts.experiments.cross_tfg_audio_head_20261003.prepare_fit40_spsa_stagea_packets_attempt05",
    },
}
PROTOCOL_REL = ".codex/scratch/dev20_future_cohort_protocol_attempt06/future_dev20_three_method_protocol_attempt06.json"
PROTOCOL_SHA256 = "73ac37eee2c8c728a7123fa85834dc53104a4b809cbc017593bd86939816eb17"
POLICY_REL = "openspec/changes/develop-cross-tfg-audio-head-20261003/future-dev20-admission-policy-attempt01.json"
POLICY_SHA256 = "f215ecff1dc9908eebfc3195a96d76b041bea47ca0fa7df5d64b6481031398ed"

PLAN_SCHEMA = "cross-tfg-audio-eval-v1"
PLAN_BINDING_SCHEMA = "cross_tfg_future_dev20_selected_spsa_center_plan_binding_v1"
FREEZE_SCHEMA = "cross_tfg_future_dev20_selected_spsa_center_pre_effect_freeze_v1"
FREEZE_STATUS = "FROZEN_BEFORE_DEV20_SCORE_ACCESS"
OVERLAY_SCHEMA = "cross_tfg_future_dev20_selected_spsa_center_protocol_v1"
APP_MANIFEST_SCHEMA = "cross_tfg_dev20_selected_spsa_center_application_manifest_v1"
APP_RECEIPT_SCHEMA = "cross_tfg_v3_portable_selected_center_wav_application_v1"
APP_RECEIPT_STATUS = "PASS_SELECTED_FIT_CENTER_APPLICATION_ONLY_NOT_FINAL_DEV"
SELECTION_STATUS = "ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV"
METHOD_KIND = "fit40_spsa_center"
MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
SEEDS = (20261003, 20261004, 20261005)
EXPECTED_CONDITIONS = ("identity", "rms_fixed", "selected_spsa_center")
BLOCK_ROWS = 6
RESERVE_BYTES = 5 * 1024 * 1024 * 1024

_PATCH_LOCK = threading.Lock()


class Dev20AdapterError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _payload_sha(doc: Mapping[str, Any], field: str) -> str:
    return _sha(_canonical({key: value for key, value in doc.items() if key != field}))


def _require_real_path(path: Path, label: str, *, exists: bool = True) -> None:
    if not path.is_absolute():
        raise Dev20AdapterError(label + " must be absolute")
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise Dev20AdapterError(label + " traverses a symlink: " + str(cursor))
    if exists and not path.exists():
        raise Dev20AdapterError(label + " does not exist: " + str(path))


def _read_regular(path: Path, label: str, limit: int = 128 * 1024 * 1024) -> bytes:
    _require_real_path(path, label)
    if not path.is_file():
        raise Dev20AdapterError(label + " is not a regular file")
    info = path.lstat()
    if info.st_size <= 0 or info.st_size > limit:
        raise Dev20AdapterError(label + " has an invalid bounded size")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise Dev20AdapterError(label + " changed during read")
    return raw


def _read_json(path: Path, label: str, limit: int = 128 * 1024 * 1024) -> Tuple[Dict[str, Any], bytes]:
    raw = _read_regular(path, label, limit)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Dev20AdapterError(label + " is not UTF-8 JSON: " + str(exc)) from exc
    if not isinstance(value, dict):
        raise Dev20AdapterError(label + " root must be an object")
    return value, raw


def _check_ref(ref: Mapping[str, Any], *, label: str, self_field: Optional[str] = None,
               max_bytes: int = 128 * 1024 * 1024) -> Tuple[Dict[str, Any], bytes, Path]:
    if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str):
        raise Dev20AdapterError(label + " file reference is missing")
    path = Path(str(ref["path"]))
    raw = _read_regular(path, label, max_bytes)
    if len(raw) != ref.get("bytes") or _sha(raw) != ref.get("sha256"):
        raise Dev20AdapterError(label + " raw bytes differ from its frozen reference")
    if self_field is None:
        return {}, raw, path.resolve(strict=True)
    doc, parsed_raw = _read_json(path, label, max_bytes)
    if parsed_raw != raw or doc.get(self_field) != _payload_sha(doc, self_field):
        raise Dev20AdapterError(label + " canonical self-hash is invalid")
    if ref.get("self_sha256") != doc.get(self_field):
        raise Dev20AdapterError(label + " self pin differs from its frozen reference")
    return doc, raw, path.resolve(strict=True)


def _require_ref_path(ref: Mapping[str, Any], expected_path: Path, label: str) -> None:
    try:
        actual = Path(str(ref.get("path", ""))).resolve(strict=True)
        expected = expected_path.resolve(strict=True)
    except OSError as exc:
        raise Dev20AdapterError(label + " path cannot be resolved: " + str(exc)) from exc
    if actual != expected:
        raise Dev20AdapterError(label + " path differs from the frozen reference")


def _same_file_pin(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str,
                   include_self: bool = False) -> None:
    fields = ("path", "bytes", "sha256") + (("self_sha256",) if include_self else ())
    if any(actual.get(key) != expected.get(key) for key in fields):
        raise Dev20AdapterError(label + " pin differs")


def _module_matches(module: Any, ref: Mapping[str, Any], label: str) -> Path:
    if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str):
        raise Dev20AdapterError(label + " source reference is absent")
    path = Path(str(getattr(module, "__file__", "")))
    raw = _read_regular(path, label + " source", 8 * 1024 * 1024)
    if (path.resolve(strict=True) != Path(str(ref["path"])).resolve(strict=True)
            or ("bytes" in ref and len(raw) != ref.get("bytes"))
            or _sha(raw) != ref.get("sha256", ref.get("raw_sha256"))):
        raise Dev20AdapterError(label + " loaded source differs from the exact plan pin")
    return path.resolve(strict=True)


def _inside(path: Path, parent: Path, label: str) -> None:
    try:
        path.resolve(strict=True).relative_to(parent.resolve(strict=True))
    except (ValueError, OSError) as exc:
        raise Dev20AdapterError(label + " escapes its pinned project root") from exc


def _source_model_plan_refs(plan: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    entries = plan.get("execution", {}).get("source_model_plans")
    if not isinstance(entries, list):
        raise Dev20AdapterError("plan omits its actual three-model source-plan table")
    output: Dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.get("model_id") not in MODELS:
            raise Dev20AdapterError("source-model plan table has an unknown/malformed model entry")
        model_id = str(entry["model_id"])
        if model_id in output:
            raise Dev20AdapterError("source-model plan table repeats a model")
        output[model_id] = entry
    if tuple(sorted(output)) != tuple(sorted(MODELS)):
        raise Dev20AdapterError("source-model plan table is not the exact three-model set")
    return output


def _verify_plan_and_freeze(root: Path, plan_path: Path, freeze_path: Path) -> Dict[str, Any]:
    root = root.resolve(strict=True)
    plan_path = plan_path.resolve(strict=True)
    freeze_path = freeze_path.resolve(strict=True)
    plan, plan_raw = _read_json(plan_path, "frozen selected-center Dev20 plan")
    freeze, freeze_raw = _read_json(freeze_path, "pre-effect freeze receipt")
    if (plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "FROZEN"
            or plan.get("plan_sha256") != _payload_sha(plan, "plan_sha256")):
        raise Dev20AdapterError("Dev20 plan schema/status/canonical self-hash is invalid")
    binding = plan.get("dev20_selected_spsa_center")
    if (not isinstance(binding, dict) or binding.get("schema") != PLAN_BINDING_SCHEMA
            or binding.get("selection_status") != SELECTION_STATUS
            or binding.get("checkpoint_sha256") is not None
            or binding.get("freeze_required_before_any_effect") is not True
            or binding.get("score_effects_read") is not False
            or binding.get("confirm_data_or_effects_accessed") is not False):
        raise Dev20AdapterError("selected center plan is not bound to its pre-effect freeze")
    selected = binding.get("selected_center")
    if (not isinstance(selected, dict) or selected.get("point_kind") != "spsa_center"
            or selected.get("source_step") not in (5, 10, 15, 20)):
        raise Dev20AdapterError("plan lacks an admitted k5/k10/k15/k20 selected center")
    methods = plan.get("methods")
    if not isinstance(methods, list) or len(methods) != 3:
        raise Dev20AdapterError("Dev20 plan must retain exactly identity, RMS, and selected-center methods")
    matches = [item for item in methods if isinstance(item, Mapping)
               and item.get("method_id") == selected.get("method_id")]
    if len(matches) != 1:
        raise Dev20AdapterError("selected-center method is absent or duplicated")
    method = dict(matches[0])
    config = method.get("config")
    if (method.get("kind") != METHOD_KIND or method.get("checkpoint_sha256") is not None
            or not isinstance(config, dict) or config.get("method") != METHOD_KIND
            or config.get("schema") != "fit40_spsa_center_transform_v1"
            or config.get("point_kind") != "spsa_center"
            or config.get("method_id") != selected.get("method_id")
            or config.get("source_step") != selected.get("source_step")
            or config.get("theta_sha256") != selected.get("theta_sha256")
            or config.get("artifact_sha256") != selected.get("artifact_sha256")
            or config.get("binding_sha256") != selected.get("binding_sha256")
            or config.get("input_pins_sha256") != selected.get("input_pins_sha256")
            or method.get("config_sha256") != selected.get("config_sha256")
            or method.get("config_sha256") != _sha(_canonical(config))):
        raise Dev20AdapterError("selected-center method/config/checkpoint identity differs")
    rows = plan.get("rows")
    execution = plan.get("execution")
    if (not isinstance(rows, list) or len(rows) != 540
            or not isinstance(execution, dict) or execution.get("planned_rows") != 540
            or execution.get("candidate_selection") != "fixed_before_dev20_effect_access"
            or execution.get("dev_or_confirm_access") is not False
            or execution.get("score_effects_read") is not False):
        raise Dev20AdapterError("plan no longer preserves the complete frozen 540-row denominator")

    freeze_pin = {"path": str(freeze_path), "bytes": len(freeze_raw), "sha256": _sha(freeze_raw),
                  "self_sha256": freeze.get("self_sha256")}
    plan_pin = {"path": str(plan_path), "bytes": len(plan_raw), "sha256": _sha(plan_raw),
                "self_sha256": plan.get("plan_sha256")}
    if (freeze.get("schema") != FREEZE_SCHEMA or freeze.get("status") != FREEZE_STATUS
            or freeze.get("selection_status") != SELECTION_STATUS
            or freeze.get("self_sha256") != _payload_sha(freeze, "self_sha256")
            or freeze.get("score_effects_read") is not False
            or freeze.get("confirm_data_or_effects_accessed") is not False
            or freeze.get("dev_or_confirm_admission_performed") is not False
            or freeze.get("checkpoint_sha256") is not None
            or freeze.get("point_kind") != "spsa_center"
            or freeze.get("source_step") != selected.get("source_step")
            or freeze.get("candidate_method_id") != selected.get("method_id")
            or freeze.get("method_config_sha256") != method.get("config_sha256")
            or freeze.get("theta_sha256") != selected.get("theta_sha256")
            or freeze.get("artifact_sha256") != selected.get("artifact_sha256")
            or freeze.get("binding_sha256") != selected.get("binding_sha256")
            or freeze.get("input_pins_sha256") != selected.get("input_pins_sha256")
            or freeze.get("plan") != plan_pin
            or freeze.get("exact_row_reconstruction") != binding.get("exact_row_reconstruction")):
        raise Dev20AdapterError("pre-effect freeze does not bind the exact selected-center plan")
    builder_pin = {"path": BUILDER_REL, "bytes": BUILDER_BYTES, "sha256": BUILDER_SHA256}
    builder_ref = freeze.get("builder_source")
    if not isinstance(builder_ref, Mapping) or any(builder_ref.get(key) != value for key, value in builder_pin.items()):
        raise Dev20AdapterError("freeze does not bind exact Dev20 builder04 source")
    builder_raw = _read_regular(root / BUILDER_REL, "frozen Dev20 builder04", 4 * 1024 * 1024)
    if len(builder_raw) != BUILDER_BYTES or _sha(builder_raw) != BUILDER_SHA256:
        raise Dev20AdapterError("local Dev20 builder04 source differs from the frozen source pin")
    analyzer_raw = _read_regular(root / ANALYZER_REL, "frozen Dev20 analyzer03", 4 * 1024 * 1024)
    if len(analyzer_raw) != ANALYZER_BYTES or _sha(analyzer_raw) != ANALYZER_SHA256:
        raise Dev20AdapterError("local Dev20 analyzer03 source differs from the accepted source pin")
    portable_raw = _read_regular(root / PORTABLE_REL, "portable selected-center wrapper05", 4 * 1024 * 1024)
    if len(portable_raw) != PORTABLE_BYTES or _sha(portable_raw) != PORTABLE_SHA256:
        raise Dev20AdapterError("portable selected-center wrapper05 source differs from its accepted pin")
    for pin, label in ((PORTABLE_APPLY_PIN, "portable waveform application"),
                       (FINAL_INGEST_PIN, "final-center ingest")):
        source_raw = _read_regular(root / pin["path"], label + " source", 8 * 1024 * 1024)
        if len(source_raw) != pin["bytes"] or _sha(source_raw) != pin["sha256"]:
            raise Dev20AdapterError(label + " source differs from the portable selected-center lineage pin")
    for role, pin in FINAL_INGEST_DEPENDENCY_SOURCE_PINS.items():
        source_raw = _read_regular(root / pin["path"], "final-center ingest " + role + " source", 8 * 1024 * 1024)
        if len(source_raw) != pin["bytes"] or _sha(source_raw) != pin["sha256"]:
            raise Dev20AdapterError("final-center ingest " + role + " source differs from its pinned dependency")

    source_protocol_ref = binding.get("source_protocol_attempt06")
    source_policy_ref = binding.get("source_policy_attempt01")
    freeze_protocol_ref = freeze.get("source_protocol_attempt06")
    freeze_policy_ref = freeze.get("source_policy_attempt01")
    source_protocol_doc: Optional[Dict[str, Any]] = None
    for source_rel, expected_sha, ref, freeze_ref, label in (
            (PROTOCOL_REL, PROTOCOL_SHA256, source_protocol_ref, freeze_protocol_ref, "frozen 20-source protocol"),
            (POLICY_REL, POLICY_SHA256, source_policy_ref, freeze_policy_ref, "frozen Dev20 admission policy")):
        source_path = root / source_rel
        source_doc, source_raw = _read_json(source_path, label, 8 * 1024 * 1024)
        if (not isinstance(ref, Mapping) or not isinstance(freeze_ref, Mapping)
                or freeze_ref != dict(ref) or len(source_raw) != ref.get("bytes")
                or _sha(source_raw) != expected_sha or ref.get("sha256") != expected_sha
                or source_doc.get("self_sha256") != ref.get("self_sha256")
                or source_doc.get("self_sha256") != _payload_sha(source_doc, "self_sha256")):
            raise Dev20AdapterError(label + " raw/self source pin differs from the selected plan/freeze")
        if source_rel == PROTOCOL_REL:
            source_protocol_doc = source_doc
    for key in ("protocol_overlay", "selected_artifact", "application_manifest", "identity_rms_base_plan"):
        ref = binding.get(key)
        freeze_ref = freeze.get(key)
        if not isinstance(ref, Mapping) or not isinstance(freeze_ref, Mapping):
            raise Dev20AdapterError("plan/freeze omits " + key)
        _same_file_pin(freeze_ref, ref, "freeze " + key, include_self="self_sha256" in ref)

    overlay_ref = binding["protocol_overlay"]
    overlay, overlay_raw, _overlay_path = _check_ref(
        overlay_ref, label="selected-center protocol overlay", self_field="self_sha256")
    if (overlay.get("schema") != OVERLAY_SCHEMA
            or overlay.get("selection_status") != SELECTION_STATUS
            or overlay.get("selected_spsa_center") != selected
            or overlay.get("score_effects_read") is not False
            or overlay.get("confirm_data_or_effects_accessed") is not False
            or overlay.get("native_tfg_qualification_claim") is not False
            or overlay.get("final_dev_or_confirm_admission_performed") is not False
            or freeze.get("protocol_overlay") != {
                "path": str(Path(str(overlay_ref["path"])).resolve(strict=True)),
                "bytes": len(overlay_raw), "sha256": _sha(overlay_raw),
                "self_sha256": overlay.get("self_sha256")}):
        raise Dev20AdapterError("selected-center protocol overlay differs from plan/freeze")
    if (overlay.get("source_protocol_attempt06") != source_protocol_ref
            or overlay.get("source_policy_attempt01") != source_policy_ref):
        raise Dev20AdapterError("selected-center overlay changed the frozen protocol or policy refs")

    artifact_ref = binding["selected_artifact"]
    artifact, artifact_raw, _artifact_path = _check_ref(artifact_ref, label="portable selected-center artifact")
    if (artifact.get("artifact_sha256") != selected.get("artifact_sha256")
            or artifact.get("artifact_sha256") != _payload_sha(artifact, "artifact_sha256")
            or artifact.get("selection_status") != SELECTION_STATUS
            or artifact.get("selected_center") != selected
            or artifact.get("final_dev_or_confirm_admission_performed") is not False
            or artifact.get("native_tfg_qualification_claim") is not False):
        raise Dev20AdapterError("portable selected-center artifact raw/canonical identity differs")
    wrapper_ref = artifact.get("wrapper_source_pin")
    if (not isinstance(wrapper_ref, Mapping)
            or wrapper_ref.get("path") != PORTABLE_REL
            or wrapper_ref.get("bytes") != PORTABLE_BYTES
            or wrapper_ref.get("sha256") != PORTABLE_SHA256):
        raise Dev20AdapterError("portable artifact does not cite the exact selected-center wrapper05 source")
    if (artifact.get("portable_apply_source_pin") != PORTABLE_APPLY_PIN
            or artifact.get("final_ingest_source_pin") != FINAL_INGEST_PIN
            or artifact.get("final_ingest_dependency_source_pins") != FINAL_INGEST_DEPENDENCY_SOURCE_PINS
            or not isinstance(artifact.get("source_code_pins"), Mapping)):
        raise Dev20AdapterError("portable selected artifact application/ingest source closure differs")
    if freeze.get("selected_artifact") != dict(artifact_ref):
        raise Dev20AdapterError("freeze selected-artifact reference differs")

    app_ref = binding["application_manifest"]
    app, app_raw, app_path = _check_ref(app_ref, label="selected-center application manifest", self_field="self_sha256")
    if (app.get("schema") != APP_MANIFEST_SCHEMA
            or app.get("status") != "COMPLETE_SELECTED_SPSA_CENTER_WAVEFORM_APPLICATION_ONLY_NOT_FINAL_DEV"
            or app.get("selection_status") != SELECTION_STATUS
            or app.get("selected_center") != selected
            or app.get("theta_sha256") != selected.get("theta_sha256")
            or app.get("native_tfg_qualification_claim") is not False
            or app.get("final_dev_or_confirm_admission_performed") is not False
            or freeze.get("application_manifest") != dict(app_ref)):
        raise Dev20AdapterError("selected-center application manifest differs from plan/freeze")

    base_ref = binding["identity_rms_base_plan"]
    base_plan, base_raw, _base_path = _check_ref(base_ref, label="identity/RMS base plan", self_field="plan_sha256")
    if (base_plan.get("schema_version") != PLAN_SCHEMA or base_plan.get("status") != "FROZEN"
            or not isinstance(base_plan.get("rows"), list)):
        raise Dev20AdapterError("identity/RMS source plan is not the exact frozen base plan")
    base_by_key: Dict[Tuple[str, str, int, str], Mapping[str, Any]] = {}
    for row in base_plan["rows"]:
        if not isinstance(row, Mapping):
            raise Dev20AdapterError("identity/RMS base plan contains a non-object row")
        key = (str(row.get("sample_id")), str(row.get("model_id")), int(row.get("seed", -1)),
               str(row.get("method_id")))
        if key in base_by_key:
            raise Dev20AdapterError("identity/RMS base plan repeats a source/model/seed/method key")
        base_by_key[key] = row
    if len(base_by_key) != len(base_plan["rows"]):
        raise Dev20AdapterError("identity/RMS base plan key map differs")

    source_cohort = plan.get("cohort_and_source", {})
    if (not isinstance(source_protocol_doc, Mapping)
            or source_cohort != source_protocol_doc.get("cohort_and_source")):
        raise Dev20AdapterError("selected plan cohort/source differs from the exact frozen protocol")
    source_rows = source_cohort.get("source_rows") if isinstance(source_cohort, Mapping) else None
    if not isinstance(source_rows, list) or len(source_rows) != 20:
        raise Dev20AdapterError("plan does not retain the frozen 20-source roster")
    source_by_id = {str(row.get("sample_id")): row for row in source_rows if isinstance(row, Mapping)}
    if len(source_by_id) != 20:
        raise Dev20AdapterError("frozen source roster has duplicate or malformed sample IDs")

    artifact_evidence = artifact.get("evidence")
    if (not isinstance(artifact_evidence, Mapping)
            or set(artifact_evidence) != {"final_center_index", "selector_receipt", "center_terminal",
                                          "point_plan", "seed_application_artifact"}):
        raise Dev20AdapterError("portable selected artifact does not retain the exact five-file selection evidence closure")
    expected_evidence_pins = {}
    for evidence_name, evidence_ref in artifact_evidence.items():
        if not isinstance(evidence_ref, Mapping):
            raise Dev20AdapterError("portable selected artifact evidence ref is malformed: " + str(evidence_name))
        expected_evidence_pins[evidence_name] = {
            name: value for name, value in evidence_ref.items() if name != "raw_utf8"
        }
    app_rows = app.get("rows")
    if not isinstance(app_rows, list) or len(app_rows) != 20:
        raise Dev20AdapterError("application manifest does not preserve exactly 20 selected-center WAV receipts")
    app_by_sample: Dict[str, Dict[str, Any]] = {}
    for app_row in app_rows:
        if not isinstance(app_row, Mapping):
            raise Dev20AdapterError("application manifest has a non-object row")
        sample_id = app_row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in app_by_sample:
            raise Dev20AdapterError("application manifest sample roster is malformed/duplicated")
        source = source_by_id.get(sample_id)
        if not isinstance(source, Mapping):
            raise Dev20AdapterError("application manifest names a source outside the frozen protocol")
        receipt_ref = app_row.get("receipt")
        receipt, receipt_raw, receipt_path = _check_ref(
            receipt_ref, label="application receipt " + sample_id, self_field="self_sha256", max_bytes=8 * 1024 * 1024)
        if (receipt.get("schema") != APP_RECEIPT_SCHEMA or receipt.get("status") != APP_RECEIPT_STATUS
                or receipt.get("selection_status") != SELECTION_STATUS
                or receipt.get("selected_center") != selected
                or receipt.get("native_tfg_qualification_claim") is not False
                or receipt.get("final_dev_or_confirm_admission_performed") is not False):
            raise Dev20AdapterError("application receipt status/selection differs for " + sample_id)
        expected_artifact_receipt_ref = {
            "path": str(Path(str(artifact_ref["path"])).resolve(strict=True)),
            "bytes": artifact_ref["bytes"],
            "sha256": artifact_ref["sha256"],
            "artifact_sha256": artifact.get("artifact_sha256"),
        }
        selected_theta = selected.get("theta")
        if (not isinstance(selected_theta, list) or len(selected_theta) != 9
                or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                       or not math.isfinite(float(value)) for value in selected_theta)):
            raise Dev20AdapterError("selected-center artifact theta is not nine finite values")
        try:
            theta_f32 = [struct.unpack("<f", struct.pack("<f", float(value)))[0]
                         for value in selected_theta]
            theta_rows = [theta_f32[offset:offset + 3] for offset in range(0, 9, 3)]
            theta_f32_sha256 = _sha(struct.pack("<9f", *theta_f32))
        except (OverflowError, struct.error) as exc:
            raise Dev20AdapterError("selected-center theta cannot be represented as float32") from exc
        rms = receipt.get("rms")
        projection = receipt.get("projection")
        projection_peak = projection.get("postprojection_max_abs_mask_db") if isinstance(projection, Mapping) else None
        expected_seed_lineage = {
            "method_id": selected.get("source_seed_method_id"),
            "source_snapshot_step": selected.get("source_snapshot_step"),
            "source_snapshot_artifact_sha256": selected.get("source_snapshot_artifact_sha256"),
        }
        if (receipt.get("selected_artifact") != expected_artifact_receipt_ref
                or receipt.get("theta") != theta_rows
                or receipt.get("theta_little_endian_f32_sha256") != theta_f32_sha256
                or receipt.get("seed_lineage") != expected_seed_lineage
                or receipt.get("evidence_pins") != expected_evidence_pins
                or receipt.get("portable_apply_source_pin") != artifact.get("portable_apply_source_pin")
                or receipt.get("wrapper_source_pin") != artifact.get("wrapper_source_pin")
                or receipt.get("final_ingest_source_pin") != artifact.get("final_ingest_source_pin")
                or receipt.get("final_ingest_dependency_source_pins") != artifact.get("final_ingest_dependency_source_pins")
                or receipt.get("runtime_source_pins") != artifact.get("source_code_pins")
                or not isinstance(rms, Mapping) or rms.get("peak_limiter_applied") is not False
                or not isinstance(rms.get("output_rms_error"), (int, float))
                or isinstance(rms.get("output_rms_error"), bool)
                or not math.isfinite(float(rms["output_rms_error"]))
                or float(rms["output_rms_error"]) < 0.0
                or not isinstance(projection, Mapping) or projection.get("within_bound") is not True
                or projection.get("bound_db") != 3.0
                or projection.get("projection_is_global_scale") is not True
                or not isinstance(projection_peak, (int, float)) or isinstance(projection_peak, bool)
                or not math.isfinite(float(projection_peak)) or float(projection_peak) > 3.0 + 1e-5):
            raise Dev20AdapterError("application receipt theta/RMS/projection/evidence differs for " + sample_id)
        output_wav = receipt.get("output_wav")
        input_wav = receipt.get("input_wav")
        if (not isinstance(output_wav, Mapping) or not isinstance(input_wav, Mapping)
                or app_row.get("output_wav") != dict(output_wav)
                or app_row.get("source_wav_sha256") != source.get("source_wav_sha256")
                or app_row.get("source_pcm_sha256") != source.get("canonical_f32_sha256")
                or app_row.get("sample_count_N") != source.get("sample_count_N")
                or input_wav.get("sha256") != source.get("source_wav_sha256")
                or input_wav.get("pcm_float32_sha256") != source.get("canonical_f32_sha256")
                or input_wav.get("sample_count") != source.get("sample_count_N")
                or input_wav.get("sample_rate_hz") != 16000 or input_wav.get("channels") != 1):
            raise Dev20AdapterError("application manifest/receipt source or output bindings differ for " + sample_id)
        input_path = Path(str(input_wav.get("path", "")))
        input_raw = _read_regular(input_path, "portable application natural WAV " + sample_id, 16 * 1024 * 1024)
        if (len(input_raw) != input_wav.get("bytes")
                or _sha(input_raw) != source.get("source_wav_sha256")):
            raise Dev20AdapterError("application receipt natural WAV bytes differ from frozen source for " + sample_id)
        output_path = Path(str(output_wav.get("path", "")))
        output_raw = _read_regular(output_path, "selected waveform " + sample_id, 16 * 1024 * 1024)
        if (len(output_raw) != output_wav.get("bytes") or _sha(output_raw) != output_wav.get("sha256")
                or output_wav.get("encoding") != "FLOAT_32"
                or output_wav.get("sample_rate_hz") != 16000 or output_wav.get("channels") != 1
                or type(output_wav.get("sample_count")) is not int or output_wav.get("sample_count") <= 0):
            raise Dev20AdapterError("selected waveform raw/file/geometry pin differs for " + sample_id)
        app_by_sample[sample_id] = {
            "sample_id": sample_id,
            "receipt_ref": {"path": str(receipt_path), "bytes": len(receipt_raw),
                            "sha256": _sha(receipt_raw), "self_sha256": receipt["self_sha256"]},
            "receipt": receipt,
            "output_wav": dict(output_wav),
        }
    if len(app_by_sample) != 20:
        raise Dev20AdapterError("application manifest source denominator is not 20")

    if len(source_by_id) != 20 or set(source_by_id) != set(app_by_sample):
        raise Dev20AdapterError("source and application rosters do not match exactly")

    condition_counts: Dict[str, int] = {method_id: 0 for method_id in ("identity", "rms_fixed", selected["method_id"])}
    selected_by_model: Dict[str, List[Dict[str, Any]]] = {model_id: [] for model_id in MODELS}
    ids = []
    for row in rows:
        if not isinstance(row, dict):
            raise Dev20AdapterError("plan contains a non-object planned row")
        method_id = str(row.get("method_id"))
        if method_id not in condition_counts or row.get("model_id") not in MODELS or row.get("seed") not in SEEDS:
            raise Dev20AdapterError("plan row is outside the frozen three-condition/model/seed roster")
        condition_counts[method_id] += 1
        ids.append(row.get("row_id"))
        source = source_by_id.get(str(row.get("sample_id")))
        if source is None or row.get("speaker_id") != source.get("speaker_id"):
            raise Dev20AdapterError("plan row speaker/source identity differs")
        natural = row.get("natural_pcm")
        if (not isinstance(natural, Mapping)
                or natural.get("file_sha256") != source.get("source_wav_sha256")
                or natural.get("pcm_sha256") != source.get("canonical_f32_sha256")
                or natural.get("sample_count") != source.get("sample_count_N")
                or natural.get("sample_rate_hz") != 16000
                or natural.get("channels") != 1):
            raise Dev20AdapterError("plan row natural PCM differs from the exact frozen 20-source protocol")
        application = app_by_sample[str(row.get("sample_id"))]
        application_input = application["receipt"].get("input_wav")
        if (not isinstance(application_input, Mapping)
                or application_input.get("sha256") != natural.get("file_sha256")
                or application_input.get("pcm_float32_sha256") != natural.get("pcm_sha256")
                or application_input.get("sample_count") != natural.get("sample_count")):
            raise Dev20AdapterError("portable application receipt does not use this exact planned natural PCM source")
        if method_id != selected.get("method_id"):
            base = base_by_key.get((str(row.get("sample_id")), str(row.get("model_id")),
                                    int(row.get("seed", -1)), method_id))
            if not isinstance(base, Mapping) or _canonical(dict(row)) != _canonical(dict(base)):
                raise Dev20AdapterError("identity/RMS row differs from exact base-plan row")
            continue
        model_id = str(row["model_id"])
        app_item = app_by_sample.get(str(row.get("sample_id")))
        base_identity = base_by_key.get((str(row.get("sample_id")), model_id, int(row.get("seed")), "identity"))
        if not isinstance(app_item, Mapping) or not isinstance(base_identity, Mapping):
            raise Dev20AdapterError("selected-center row lacks its application or identity parent")
        receipt_ref = app_item["receipt_ref"]
        output = app_item["output_wav"]
        expected_input = {
            "path": output["path"], "bytes": output["bytes"], "sha256": output["sha256"],
            "file_sha256": output["sha256"], "pcm_sha256": output["pcm_float32_sha256"],
            "sample_count": output["sample_count"], "sample_rate_hz": output["sample_rate_hz"],
            "channels": output["channels"], "encoding": output["encoding"],
            "application_receipt_raw_sha256": receipt_ref["sha256"],
            "application_row_receipt_sha256": receipt_ref["self_sha256"],
        }
        binding_row = row.get("spsa_binding")
        if (row.get("checkpoint_sha256") is not None
                or row.get("score_audio") != row.get("natural_pcm")
                or row.get("method_config_sha256") != method.get("config_sha256")
                or row.get("generation_settings", {}).get("input_audio") != "precomputed_selected_spsa_center_waveform"
                or row.get("generator_condition_input") != expected_input
                or not isinstance(binding_row, Mapping)
                or binding_row.get("schema") != "cross_tfg_dev20_selected_spsa_center_row_binding_v1"
                or binding_row.get("point_kind") != "spsa_center"
                or binding_row.get("source_step") != selected.get("source_step")
                or binding_row.get("method_id") != selected.get("method_id")
                or binding_row.get("method_config_sha256") != method.get("config_sha256")
                or binding_row.get("theta_sha256") != selected.get("theta_sha256")
                or binding_row.get("artifact_sha256") != selected.get("artifact_sha256")
                or binding_row.get("binding_sha256") != selected.get("binding_sha256")
                or binding_row.get("input_pins_sha256") != selected.get("input_pins_sha256")
                or binding_row.get("application_receipt") != receipt_ref
                or binding_row.get("generator_condition_input") != expected_input
                or binding_row.get("score_audio_is_natural_pcm") is not True
                or row.get("spsa_point_binding_sha256") != selected.get("binding_sha256")):
            raise Dev20AdapterError("selected-center plan row/application/source binding differs for " + str(row.get("row_id")))
        selected_by_model[model_id].append(row)
    if (condition_counts != {"identity": 180, "rms_fixed": 180, selected["method_id"]: 180}
            or len(set(ids)) != 540 or any(len(selected_by_model[item]) != 60 for item in MODELS)):
        raise Dev20AdapterError("frozen plan does not preserve 180 condition / 60-per-model candidate denominators")
    for model_id, model_rows in selected_by_model.items():
        per_sample: Dict[str, List[Mapping[str, Any]]] = {}
        for row in model_rows:
            per_sample.setdefault(str(row["sample_id"]), []).append(row)
        if (set(per_sample) != set(source_by_id)
                or any(len(items) != 3 or {int(item["seed"]) for item in items} != set(SEEDS)
                       for items in per_sample.values())):
            raise Dev20AdapterError("selected model roster is not exact 20-source × 3-seed")
    if binding.get("exact_row_reconstruction", {}).get("status") != "PASS_EXACT_RECONSTRUCTED_540_ROWS":
        raise Dev20AdapterError("plan lacks the builder04 full ordered-row reconstruction PASS")
    if binding.get("exact_row_reconstruction", {}).get("ordered_rows_sha256") != _sha(_canonical(rows)):
        raise Dev20AdapterError("builder04 ordered-row reconstruction pin differs from the current 540 rows")
    if freeze.get("exact_row_reconstruction") != binding.get("exact_row_reconstruction"):
        raise Dev20AdapterError("freeze exact-row reconstruction differs from plan")
    if (len(selected_by_model) != 3 or any(len(rows_for_model) != 60 for rows_for_model in selected_by_model.values())):
        raise Dev20AdapterError("candidate model denominator is not exactly 60 each")
    return {
        "root": root, "plan": plan, "plan_raw": plan_raw, "plan_path": plan_path,
        "plan_pin": plan_pin, "freeze": freeze, "freeze_raw": freeze_raw,
        "freeze_path": freeze_path, "freeze_pin": freeze_pin, "selected": selected,
        "method": method, "selected_by_model": selected_by_model, "source_by_id": source_by_id,
        "application_by_sample": app_by_sample, "base_plan": base_plan,
        "base_plan_raw": base_raw, "base_by_key": base_by_key,
        "overlay": overlay, "overlay_raw": overlay_raw, "application_manifest": app,
        "application_manifest_raw": app_raw, "artifact": artifact, "artifact_raw": artifact_raw,
    }


def _validate_arrays(arrays: Mapping[str, Any], expected_queries: Optional[int] = None) -> Dict[str, Any]:
    try:
        import numpy as np
    except Exception as exc:
        raise Dev20AdapterError("native lexical runtime has no NumPy") from exc
    expected_keys = {
        "query_frame_index", "query_timestamp_num", "query_timestamp_den",
        "track_frame_index", "valid_query_mask", "distance_by_query_lag", "lag_offsets_frames",
    }
    if not isinstance(arrays, Mapping) or set(arrays) != expected_keys:
        raise Dev20AdapterError("Stage-A did not return the exact seven official score arrays")
    frame = np.asarray(arrays["query_frame_index"])
    timestamp_num = np.asarray(arrays["query_timestamp_num"])
    timestamp_den = np.asarray(arrays["query_timestamp_den"])
    track = np.asarray(arrays["track_frame_index"])
    valid = np.asarray(arrays["valid_query_mask"])
    matrix = np.asarray(arrays["distance_by_query_lag"])
    lag = np.asarray(arrays["lag_offsets_frames"])
    count = int(frame.shape[0]) if frame.ndim == 1 else -1
    if (count <= 0 or (expected_queries is not None and count != expected_queries)
            or timestamp_num.shape != (count,) or timestamp_den.shape != (count,)
            or track.shape != (count,) or valid.shape != (count,) or matrix.shape != (count, 31)
            or lag.shape != (31,)
            or frame.dtype.kind != "i" or frame.dtype.itemsize != 4
            or timestamp_num.dtype.kind != "i" or timestamp_num.dtype.itemsize != 8
            or timestamp_den.dtype.kind != "i" or timestamp_den.dtype.itemsize != 8
            or track.dtype.kind != "i" or track.dtype.itemsize != 4
            or valid.dtype.kind != "b" or matrix.dtype.kind != "f" or matrix.dtype.itemsize != 4
            or lag.dtype.kind != "i" or lag.dtype.itemsize != 4
            or lag.tolist() != list(range(-15, 16))):
        raise Dev20AdapterError("official Stage-A array dtypes/shapes/31-lag axis differ")
    if not np.isfinite(matrix).all() or np.any(timestamp_den <= 0):
        raise Dev20AdapterError("official score arrays contain non-finite values or invalid clock denominators")
    return {"query_count": count, "distance_shape": list(matrix.shape),
            "valid_query_count": int(np.count_nonzero(valid)), "lag_offsets_frames": lag.tolist()}


def _validate_npz_ref(result: Mapping[str, Any], row_root: Path, arrays: Mapping[str, Any]) -> Dict[str, Any]:
    score = result.get("score")
    ref = score.get("raw_matrix_npz") if isinstance(score, Mapping) else None
    if not isinstance(ref, Mapping) or not isinstance(ref.get("path"), str):
        raise Dev20AdapterError("Stage-A row lacks its raw official NPZ reference")
    path = Path(str(ref["path"]))
    raw = _read_regular(path, "official raw score NPZ", 256 * 1024 * 1024)
    try:
        path.resolve(strict=True).relative_to(row_root.resolve(strict=True))
    except (ValueError, OSError) as exc:
        raise Dev20AdapterError("official raw score NPZ escapes the exact candidate row output") from exc
    if ref.get("file_sha256") != _sha(raw):
        raise Dev20AdapterError("official raw score NPZ bytes differ from Stage-A result metadata")
    metadata = ref.get("arrays")
    if not isinstance(metadata, Mapping) or set(metadata) != set(arrays):
        raise Dev20AdapterError("official raw NPZ array table differs from seven returned arrays")
    for name, array in arrays.items():
        try:
            import numpy as np
            actual_sha = _sha(np.ascontiguousarray(array).tobytes(order="C"))
        except Exception as exc:
            raise Dev20AdapterError("cannot hash official score arrays") from exc
        item = metadata.get(name)
        if (not isinstance(item, Mapping) or item.get("dtype") != array.dtype.str
                or item.get("shape") != list(array.shape) or item.get("array_sha256") != actual_sha):
            raise Dev20AdapterError("official NPZ array pin differs for " + name)
    return {"path": str(path.resolve(strict=True)), "bytes": len(raw), "sha256": _sha(raw),
            "array_count": len(arrays)}


class Dev20SelectedStageAAdapter:
    """Bind one exact backend runtime and execute six selected rows at a time.

    ``runtime`` is a model-specific mapping containing the already-loaded
    pinned ``core`` and ``args`` plus ``project_root``, ``runtime_binding``
    (JSON mapping), ``runtime_binding_raw``, ``runtime_binding_path``, and
    ``resource_helper``. Muse additionally supplies ``overlay`` and
    ``cross_frontend``. Ditto supplies ``overlay`` and the prevalidated
    ``expected_source_and_weights`` returned by the pinned Ditto source/weight
    validator. Modules are checked against the frozen source-model execution
    references before any native call.
    """

    def __init__(self, verified: Mapping[str, Any], runtime: Mapping[str, Any]):
        self.context = dict(verified)
        self.plan = verified["plan"]
        self.freeze = verified["freeze"]
        self.selected = verified["selected"]
        self.method = verified["method"]
        self.runtime = dict(runtime)
        adapter_path = Path(__file__).resolve(strict=True)
        adapter_raw = _read_regular(adapter_path, "Dev20 native adapter source", 8 * 1024 * 1024)
        self.adapter_source_pin = {"path": str(adapter_path), "bytes": len(adapter_raw),
                                  "sha256": _sha(adapter_raw)}
        self.model_id = str(self.runtime.get("model_id", ""))
        if self.model_id not in MODELS:
            raise Dev20AdapterError("runtime model ID is outside the frozen three-model family")
        self.rows = list(verified["selected_by_model"][self.model_id])
        if len(self.rows) != 60:
            raise Dev20AdapterError("selected-center model subset is not exactly 60 rows")
        models = self.plan.get("models", [])
        matches = [item for item in models if isinstance(item, Mapping) and item.get("model_id") == self.model_id]
        if len(matches) != 1:
            raise Dev20AdapterError("selected plan does not contain exactly one backend model descriptor")
        self.model = dict(matches[0])
        self.core = self.runtime.get("core")
        self.overlay = self.runtime.get("overlay")
        self.cross = self.runtime.get("cross_frontend")
        self.args = self.runtime.get("args")
        self.project_root = Path(str(self.runtime.get("project_root", "")))
        _require_real_path(self.project_root, "native project root")
        self.project_root = self.project_root.resolve(strict=True)
        if Path(str(getattr(self.args, "project_root", ""))).resolve(strict=True) != self.project_root:
            raise Dev20AdapterError("Stage-A args.project_root differs from the pinned native source root")
        self.runtime_ref = self.plan.get("execution", {}).get("runtime_bindings_by_model", {}).get(self.model_id)
        if not isinstance(self.runtime_ref, Mapping):
            raise Dev20AdapterError("selected plan omits exact runtime binding reference")
        self.runtime_path = Path(str(self.runtime.get("runtime_binding_path", "")))
        self.runtime_raw = bytes(self.runtime.get("runtime_binding_raw", b""))
        self.binding = self.runtime.get("runtime_binding")
        if not isinstance(self.binding, Mapping):
            raise Dev20AdapterError("backend runtime binding is not an object")
        _require_real_path(self.runtime_path, "backend runtime binding")
        try:
            pinned_runtime_path = Path(str(self.runtime_ref.get("path", ""))).resolve(strict=True)
            actual_runtime_path = self.runtime_path.resolve(strict=True)
            runtime_doc = json.loads(self.runtime_raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Dev20AdapterError("runtime binding raw JSON/path cannot be verified") from exc
        if (not isinstance(runtime_doc, dict)
                or actual_runtime_path != pinned_runtime_path
                or dict(self.binding) != runtime_doc
                or not self.runtime_path.is_absolute() or self.runtime_path.is_symlink()
                or not self.runtime_path.is_file()
                or self.runtime_path.read_bytes() != self.runtime_raw
                or (self.runtime_ref.get("bytes") is not None
                    and len(self.runtime_raw) != self.runtime_ref.get("bytes"))
                or _sha(self.runtime_raw) != self.runtime_ref.get("raw_sha256")):
            raise Dev20AdapterError("runtime binding path/body differs from exact plan pin")
        try:
            self.runtime_path.resolve(strict=True).relative_to(self.project_root)
        except ValueError as exc:
            raise Dev20AdapterError("runtime binding escapes exact native project root") from exc
        if (self.binding.get("model_id") != self.model_id
                or self.binding.get("scientific_input_pins_sha256") != self.plan.get("scientific_input_pins_sha256")):
            raise Dev20AdapterError("runtime binding model/scientific source pins differ from selected plan")
        self.resource_helper = self.runtime.get("resource_helper")
        resources = self.binding.get("resources", {})
        helper_ref = resources.get("resource_helper") if isinstance(resources, Mapping) else None
        if not isinstance(helper_ref, Mapping):
            raise Dev20AdapterError("runtime binding lacks its exact live resource helper reference")
        _module_matches(self.resource_helper, helper_ref, "live resource helper")
        self.source_plan_refs = _source_model_plan_refs(self.plan)
        source_ref = self.source_plan_refs[self.model_id].get("plan")
        self.source_plan, self.source_plan_raw, self.source_plan_path = _check_ref(
            source_ref, label=self.model_id + " frozen source plan", self_field="plan_sha256")
        if (self.source_plan.get("status") != "FROZEN"
                or self.source_plan.get("models") != [self.model]
                or len(self.source_plan.get("rows", [])) != 120
                or self.plan.get("execution", {}).get("source_model_execution", {}).get(self.model_id)
                   != self.source_plan.get("execution")):
            raise Dev20AdapterError("model source plan descriptor/execution differs from composed selected plan")
        if self.source_plan.get("run_id") != self.binding.get("run_id"):
            raise Dev20AdapterError("model runtime binding does not belong to the exact source model plan")
        self.source_protocol = self.plan.get("execution", {}).get("source_model_protocols", {}).get(self.model_id)
        if not isinstance(self.source_protocol, Mapping):
            raise Dev20AdapterError("selected plan omits model-specific source scoring protocol")
        if isinstance(self.source_plan.get("protocol"), Mapping):
            source_generator = self.source_plan["protocol"].get("generator")
            selected_generator = self.source_protocol.get("generator")
            if source_generator is not None and selected_generator != source_generator:
                raise Dev20AdapterError("selected plan changed the source model's per-model generator protocol")
        self.generator = self.source_protocol.get("generator")
        if not isinstance(self.generator, Mapping):
            raise Dev20AdapterError("model-specific source protocol has no native generator entry")
        self.source_execution = self.plan.get("execution", {}).get("source_model_execution", {}).get(self.model_id)
        if not isinstance(self.source_execution, Mapping):
            raise Dev20AdapterError("selected plan omits model-specific source execution pins")
        self._validate_native_modules()
        self.identity_by_key = verified["base_by_key"]
        self._verify_candidate_identity_rows()

    def _validate_native_modules(self) -> None:
        core_ref = self.source_execution.get("stagea_runner")
        core_path = _module_matches(self.core, core_ref, "package11 Stage-A core")
        _inside(core_path, self.project_root, "Stage-A core")
        if self.model_id == "musetalk_fixed_portrait_attempt16":
            overlay_ref = self.source_execution.get("musetalk_block_overlay")
            cross_ref = self.source_execution.get("cross_frontend_runner")
            _inside(_module_matches(self.overlay, overlay_ref, "MuseTalk native overlay"),
                    self.project_root, "MuseTalk overlay")
            _inside(_module_matches(self.cross, cross_ref, "MuseTalk cross frontend"),
                    self.project_root, "MuseTalk cross frontend")
        elif self.model_id == "ditto_hubert_fixed_portrait_attempt18":
            overlay_ref = self.source_execution.get("ditto_block_overlay")
            _inside(_module_matches(self.overlay, overlay_ref, "Ditto native overlay"),
                    self.project_root, "Ditto overlay")
            if not isinstance(self.runtime.get("expected_source_and_weights"), Mapping):
                raise Dev20AdapterError("Ditto runtime lacks source/weight evidence from its pinned validator")
        if (self.model_id != "wav2lip_gan_bac9a81e"
                and (self.overlay is None or not callable(getattr(self.overlay, "_capacity_observation", None)))):
            raise Dev20AdapterError("native overlay lacks its actual per-row capacity observer")

    def _verify_candidate_identity_rows(self) -> None:
        for row in self.rows:
            base = self.identity_by_key.get((str(row.get("sample_id")), self.model_id,
                                             int(row.get("seed", -1)), "identity"))
            if not isinstance(base, Mapping):
                raise Dev20AdapterError("selected row has no exact base identity parent")
            fields = ("sample_id", "speaker_id", "split", "model_id", "face_id", "seed", "natural_pcm",
                      "score_audio", "source_face", "source_video", "source_media_kind",
                      "source_video_sha256", "source_face_sha256")
            if any(row.get(field) != base.get(field) for field in fields):
                raise Dev20AdapterError("selected row changed frozen identity/source/clock fields")
            if row.get("score_audio") != row.get("natural_pcm"):
                raise Dev20AdapterError("selected row SyncNet score input is not the original natural PCM")

    def _validate_generator_input(self, row: Mapping[str, Any]) -> Dict[str, Any]:
        app_item = self.context["application_by_sample"].get(str(row.get("sample_id")))
        if not isinstance(app_item, Mapping):
            raise Dev20AdapterError("selected candidate has no frozen application receipt")
        expected = row.get("generator_condition_input")
        output = app_item.get("output_wav")
        receipt_ref = app_item.get("receipt_ref")
        if not isinstance(expected, Mapping) or not isinstance(output, Mapping) or not isinstance(receipt_ref, Mapping):
            raise Dev20AdapterError("selected row application lineage is incomplete")
        receipt, raw, receipt_path = _check_ref(
            receipt_ref, label="candidate application receipt", self_field="self_sha256", max_bytes=8 * 1024 * 1024)
        if (receipt_path != Path(str(receipt_ref["path"])).resolve(strict=True)
                or receipt.get("selected_center") != self.selected
                or receipt.get("output_wav") != output):
            raise Dev20AdapterError("application receipt identity/output differs from frozen manifest")
        current = {
            "path": output["path"], "bytes": output["bytes"], "sha256": output["sha256"],
            "file_sha256": output["sha256"], "pcm_sha256": output["pcm_float32_sha256"],
            "sample_count": output["sample_count"], "sample_rate_hz": output["sample_rate_hz"],
            "channels": output["channels"], "encoding": output["encoding"],
            "application_receipt_raw_sha256": _sha(raw),
            "application_row_receipt_sha256": receipt["self_sha256"],
        }
        if dict(expected) != current:
            raise Dev20AdapterError("row generator_condition_input differs from exact application receipt")
        if (receipt.get("clock", {}).get("preserved") is not True
                or receipt.get("clock", {}).get("input_samples") != row.get("natural_pcm", {}).get("sample_count")
                or receipt.get("clock", {}).get("output_samples") != row.get("natural_pcm", {}).get("sample_count")
                or output.get("sample_count") != row.get("natural_pcm", {}).get("sample_count")
                or output.get("sample_rate_hz") != 16000 or output.get("channels") != 1
                or output.get("encoding") != "FLOAT_32"):
            raise Dev20AdapterError("selected WAV no longer preserves original sample clock/N")
        output_path = Path(str(output.get("path", "")))
        output_raw = _read_regular(output_path, "selected candidate generator WAV", 16 * 1024 * 1024)
        if len(output_raw) != output.get("bytes") or _sha(output_raw) != output.get("sha256"):
            raise Dev20AdapterError("selected candidate generator WAV raw bytes changed")
        if row.get("score_audio") != row.get("natural_pcm"):
            raise Dev20AdapterError("candidate score audio diverged from natural source")
        return {"receipt": receipt, "receipt_ref": dict(receipt_ref),
                "output_wav": dict(output), "generator_condition_input": current}

    def _expected_frames(self, row: Mapping[str, Any]) -> Optional[int]:
        if self.model_id == "wav2lip_gan_bac9a81e":
            binding = row.get("wav2lip_native_frame_binding", {})
            value = binding.get("expected_inference_frame_count") if isinstance(binding, Mapping) else None
        elif self.model_id == "musetalk_fixed_portrait_attempt16":
            binding = row.get("musetalk_native_frame_binding", {})
            value = binding.get("expected_frame_count") if isinstance(binding, Mapping) else None
            n = binding.get("sample_count") if isinstance(binding, Mapping) else None
            if type(n) is not int or value != (n * 25) // 16000:
                raise Dev20AdapterError("Muse native frame binding differs from floor(N*25/16000)")
        else:
            binding = row.get("ditto_native_frame_binding", {})
            value = binding.get("expected_frame_count") if isinstance(binding, Mapping) else None
            n = binding.get("sample_count") if isinstance(binding, Mapping) else None
            if type(n) is not int or value != int(math.ceil(n * 25.0 / 16000.0)):
                raise Dev20AdapterError("Ditto native frame binding differs from ceil(N*25/16000)")
        if type(value) is not int or value <= 0:
            raise Dev20AdapterError("selected row lacks a positive source-bound native frame count")
        return value

    def _capacity_gate(self, remaining: Sequence[Mapping[str, Any]], output_root: Path) -> Dict[str, Any]:
        if self.model_id == "musetalk_fixed_portrait_attempt16":
            remaining_plan = copy.deepcopy(self.plan)
            remaining_plan["rows"] = [copy.deepcopy(dict(row)) for row in remaining]
            gate = self.overlay._capacity_observation(
                remaining_plan, dict(self.binding), self.resource_helper, output_root, 0, whole_stage=False)
            if not isinstance(gate, Mapping) or gate.get("status") != "PASS":
                raise Dev20AdapterError("live MuseTalk candidate-scope capacity gate failed")
            return dict(gate)
        if self.model_id == "ditto_hubert_fixed_portrait_attempt18":
            remaining_plan = copy.deepcopy(self.plan)
            remaining_plan["rows"] = [copy.deepcopy(dict(row)) for row in remaining]
            gate = self.overlay._capacity_observation(
                remaining_plan, self.binding, self.resource_helper, output_root, 0, whole_stage=False)
            if not isinstance(gate, Mapping) or gate.get("status") != "PASS":
                raise Dev20AdapterError("live Ditto candidate-scope capacity gate failed")
            return dict(gate)
        return self._w2l_candidate_capacity_gate(remaining, output_root)

    def _w2l_candidate_capacity_gate(self, remaining: Sequence[Mapping[str, Any]],
                                     output_root: Path) -> Dict[str, Any]:
        """Use the pinned W2L live observer with an exact 60-row candidate estimate."""
        resources = self.binding.get("resources", {})
        basis = resources.get("capacity_basis", {}) if isinstance(resources, Mapping) else {}
        pilot_map = basis.get("per_backend_pilot_row_output_bytes", {}) if isinstance(basis, Mapping) else {}
        pilot_bytes = pilot_map.get(self.model_id) if isinstance(pilot_map, Mapping) else None
        slack = basis.get("capacity_scenario", {}).get("explicit_slack_multiplier") if isinstance(basis, Mapping) else None
        reserve = resources.get("output_reserve_bytes") if isinstance(resources, Mapping) else None
        if (basis.get("status") != "MEASURED_W2L_ONLY_PILOT_WITH_EXPLICIT_FIT240_SCENARIO_AND_LIVE_GATES"
                or basis.get("media_bytes_estimated") is not True or type(pilot_bytes) is not int or pilot_bytes <= 0
                or type(slack) not in (int, float) or slack < 1 or type(reserve) is not int
                or reserve != RESERVE_BYTES or not remaining or len(remaining) > 60):
            raise Dev20AdapterError("W2L binding lacks the pinned pilot/slack/reserve for a 60-row selected scope")
        per_row = int(math.ceil(pilot_bytes * float(slack)))
        required = per_row * len(remaining) + reserve
        snapshot = self.resource_helper.resource_snapshot(output_root, {
            "min_gpu_free_mib": 0, "min_ram_available_bytes": 0,
            "min_shm_free_bytes": 0, "min_output_free_bytes": 0,
        })
        gpu_uuid = resources.get("gpu_uuid")
        gpu = next((item for item in snapshot.get("gpus", []) if item.get("uuid") == gpu_uuid), None)
        ram = snapshot.get("ram", {})
        shm = snapshot.get("shm", {})
        disk = snapshot.get("output_disk", {})
        cgroup = ram.get("cgroup", {})
        expected_cgroup = resources.get("cgroup", {})
        reasons = []
        if gpu is None or int(gpu.get("memory_free_mib", -1)) < int(resources.get("min_gpu_free_mib", 0)):
            reasons.append("bound V100 free memory is below its pinned minimum")
        if int(ram.get("available_bytes", -1)) < int(resources.get("min_effective_ram_available_bytes", 0)):
            reasons.append("effective RAM is below its pinned minimum")
        if int(shm.get("available_bytes", -1)) < required:
            reasons.append("/dev/shm is below remaining selected candidate estimate plus reserve")
        if (disk.get("target") != resources.get("output_mount_target")
                or int(disk.get("available_bytes", -1)) < required):
            reasons.append("output mount/capacity differs from the pinned selected candidate requirement")
        actual_identity = cgroup.get("identity", {}) if isinstance(cgroup, Mapping) else {}
        expected_identity = expected_cgroup.get("identity", {}) if isinstance(expected_cgroup, Mapping) else {}
        if (cgroup.get("version") != expected_cgroup.get("version")
                or actual_identity.get("mount_target") != expected_identity.get("mount_target")
                or actual_identity.get("mount_root") != expected_identity.get("mount_root")):
            reasons.append("stable cgroup identity differs from the pinned W2L runtime")
        if cgroup.get("finite_limit") is True:
            available = cgroup.get("available_bytes")
            if type(available) is not int or available < int(resources.get("min_effective_ram_available_bytes", 0)):
                reasons.append("finite cgroup memory is below the pinned minimum")
        elif cgroup.get("finite_limit") is not False or cgroup.get("available_bytes") is not None:
            reasons.append("live unlimited cgroup fields are malformed")
        if reasons:
            raise Dev20AdapterError("W2L selected-candidate live capacity gate failed: " + "; ".join(reasons))
        return {
            "schema": "cross_tfg_dev20_selected_candidate_capacity_gate_v1",
            "status": "PASS", "model_id": self.model_id,
            "scope": "remaining exact selected-center rows for this model; estimate, not hard bound",
            "remaining_rows": len(remaining), "remaining_candidate_rows_model": 60,
            "candidate_rows_overall": 180, "parent_planned_rows": 540,
            "pilot_row_bytes": pilot_bytes, "explicit_slack_multiplier": float(slack),
            "estimated_bytes": per_row * len(remaining), "reserve_bytes": reserve,
            "required_bytes_with_reserve": required,
            "observed_gpu_free_mib": gpu.get("memory_free_mib") if gpu else None,
            "observed_ram_available_bytes": ram.get("available_bytes"),
            "observed_shm_available_bytes": shm.get("available_bytes"),
            "observed_output_available_bytes": disk.get("available_bytes"),
            "snapshot_sha256": _sha(_canonical(snapshot)),
        }

    def render_block(self, *, block_start: int, output_root: Union[str, Path]) -> Dict[str, Any]:
        """Render one aligned six-row slice and retain every planned row result.

        ``block_start`` is relative to this model's exact ordered 60 candidate
        rows. Any row failure is retained as FAILED and the unattempted rows
        become explicit INCOMPLETE denominator records; this call never retries.
        """
        if type(block_start) is not int or block_start < 0 or block_start % BLOCK_ROWS or block_start + BLOCK_ROWS > 60:
            raise Dev20AdapterError("candidate block must be an aligned six-row slice inside the exact 60-row model roster")
        candidate_rows = self.rows[block_start:block_start + BLOCK_ROWS]
        if len(candidate_rows) != BLOCK_ROWS:
            raise Dev20AdapterError("selected candidate block is not exactly six rows")
        output = Path(output_root)
        _require_real_path(output, "candidate block output root", exists=False)
        if output.exists() or output.is_symlink() or not output.parent.is_dir():
            raise Dev20AdapterError("candidate block output root must be absent under an existing real parent")
        try:
            output.resolve(strict=False).relative_to(Path(str(self.plan.get("execution", {}).get("output_root", ""))).resolve(strict=False))
        except (ValueError, OSError) as exc:
            raise Dev20AdapterError("candidate block output escapes the frozen Dev20 output root") from exc

        # Read every row's exact application receipt and original PCM before
        # creating output directories or invoking a renderer.
        row_inputs: List[Dict[str, Any]] = []
        natural_cache: Dict[Tuple[str, str, str], Tuple[Path, Any, Dict[str, Any]]] = {}
        for row in candidate_rows:
            app_item = self._validate_generator_input(row)
            natural_ref = row.get("natural_pcm")
            cache_key = (str(natural_ref.get("path")), str(natural_ref.get("file_sha256")),
                         str(natural_ref.get("pcm_sha256")))
            if cache_key not in natural_cache:
                natural_cache[cache_key] = self.core._reference(natural_ref)
            natural_path, natural_samples, actual_ref = natural_cache[cache_key]
            if (row.get("score_audio") != row.get("natural_pcm")
                    or actual_ref.get("pcm_sha256") != natural_ref.get("pcm_sha256")
                    or actual_ref.get("sample_count") != natural_ref.get("sample_count")):
                raise Dev20AdapterError("original natural PCM/clock binding changed before render")
            self._expected_frames(row)
            row_inputs.append({"row": row, "application": app_item, "natural_path": natural_path,
                               "natural_samples": natural_samples, "natural_ref": actual_ref})

        if not _PATCH_LOCK.acquire(False):
            raise Dev20AdapterError("another Dev20 native Stage-A block is active in this process")
        try:
            output.mkdir(mode=0o700)
            ledger_path = output / "results.jsonl"
            results_stream = ledger_path.open("xb")
            started = time.time()
            row_results: List[Dict[str, Any]] = []
            resource_gates: List[Dict[str, Any]] = []
            completed = 0
            failure_seen = False
            strict_receipt_dir = output / "strict_receipts"
            if self.model_id == "musetalk_fixed_portrait_attempt16":
                strict_receipt_dir.mkdir(mode=0o700)
            env_keys = {"CROSS_TFG_CAPACITY_STAGE", "CROSS_TFG_RUNTIME_BINDING_PATH",
                        "CROSS_TFG_RUNTIME_BINDING_RAW_SHA256", "CROSS_TFG_DETERMINISM_RECEIPT_DIR",
                        "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE"}
            saved_env = {key: os.environ.get(key) for key in env_keys}
            core = self.core
            original_generator = core._run_generator
            original_frame_clock = core._frame_clock
            frame_counts: Dict[str, int] = {}
            processed = {"count": 0}
            muse_prior: Dict[str, set] = {}
            os.environ["CROSS_TFG_CAPACITY_STAGE"] = {
                "wav2lip_gan_bac9a81e": "fit40_w2l240",
                "musetalk_fixed_portrait_attempt16": "fit40_musetalk240",
                "ditto_hubert_fixed_portrait_attempt18": "fit40_ditto240",
            }[self.model_id]
            os.environ["CROSS_TFG_RUNTIME_BINDING_PATH"] = str(self.runtime_path.resolve(strict=True))
            os.environ["CROSS_TFG_RUNTIME_BINDING_RAW_SHA256"] = _sha(self.runtime_raw)
            os.environ["PYTHONNOUSERSITE"] = "1"
            os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
            if self.model_id == "musetalk_fixed_portrait_attempt16":
                os.environ["CROSS_TFG_DETERMINISM_RECEIPT_DIR"] = str(strict_receipt_dir.resolve(strict=True))

            if self.model_id == "musetalk_fixed_portrait_attempt16":
                def muse_generator(**kwargs: Any) -> Dict[str, Any]:
                    row = kwargs.get("row")
                    if not isinstance(row, Mapping) or row.get("model_id") != self.model_id:
                        raise Dev20AdapterError("MuseTalk native generator received a different model row")
                    index = processed["count"]
                    prior = {path.name for path in strict_receipt_dir.iterdir()
                             if path.is_file() and not path.is_symlink()}
                    if len(prior) != index * 2:
                        raise Dev20AdapterError("Muse strict receipts do not preserve two-per-completed-row")
                    execution = self.overlay._validate_runtime_evidence(
                        core, self.cross, **kwargs, generator=self.generator,
                        prior_receipt_names=prior)
                    expected = self._expected_frames(row)
                    frame_counts[str(Path(kwargs["work"]).resolve(strict=True))] = int(expected)
                    processed["count"] += 1
                    return dict(execution)

                def muse_frame_clock(path: Path, ffprobe: str, fps_num: int, fps_den: int):
                    frames, stream = original_frame_clock(path, ffprobe, fps_num, fps_den)
                    row_root = str(Path(path).resolve(strict=False).parent.parent)
                    expected = frame_counts.get(row_root)
                    if expected is not None and len(frames) != expected:
                        raise Dev20AdapterError("Muse decoded frame count differs from its frozen native clock")
                    return frames, stream
                core._run_generator = muse_generator
                core._frame_clock = muse_frame_clock

            ditto_expected_by_row_root: Dict[str, int] = {}
            if self.model_id == "ditto_hubert_fixed_portrait_attempt18":
                def ditto_generator(**kwargs: Any) -> Dict[str, Any]:
                    row = kwargs.get("row")
                    if not isinstance(row, Mapping) or row.get("model_id") != self.model_id:
                        raise Dev20AdapterError("Ditto native generator received a different model row")
                    return self.overlay._run_ditto_generator(
                        core=core, generator=self.generator,
                        expected_source_and_weights=self.runtime["expected_source_and_weights"],
                        expected_frames_by_row_root=ditto_expected_by_row_root,
                        processed_rows=processed, **kwargs)

                def ditto_frame_clock(path: Path, ffprobe: str, fps_num: int, fps_den: int):
                    frames, stream = original_frame_clock(path, ffprobe, fps_num, fps_den)
                    row_root = str(Path(path).resolve(strict=False).parent.parent)
                    expected = ditto_expected_by_row_root.get(row_root)
                    if expected is not None:
                        if len(frames) != expected:
                            raise Dev20AdapterError("Ditto decoded frame count differs from its frozen native clock")
                        for index, frame in enumerate(frames):
                            if (frame.get("frame_index") != index
                                    or not self.overlay.timestamp_matches_frame(frame, index, 25, 1)):
                                raise Dev20AdapterError("Ditto frame PTS differs from the pinned native 25 fps clock")
                    return frames, stream
                core._run_generator = ditto_generator
                core._frame_clock = ditto_frame_clock

            try:
                for index, item in enumerate(row_inputs):
                    row = item["row"]
                    row_id = str(row["row_id"])
                    row_root = output / row_id
                    natural_ref = item["natural_ref"]
                    result = None
                    try:
                        remaining = candidate_rows[index:]
                        gate = self._capacity_gate(remaining, output)
                        resource_gates.append({"row_id": row_id, "gate": gate})
                        if self.model_id == "ditto_hubert_fixed_portrait_attempt18":
                            ditto_expected_by_row_root[str(row_root.resolve(strict=False))] = int(self._expected_frames(row))
                        method_copy = copy.deepcopy(self.method)
                        method_copy["config"] = copy.deepcopy(self.method["config"])
                        method_copy["config"]["method"] = "precomputed_fit_snapshot_waveform"
                        if method_copy.get("config_sha256") != self.method.get("config_sha256"):
                            raise Dev20AdapterError("private generator-only method marker changed the frozen method pin")
                        result = core._render_and_score(
                            row, self.model, method_copy, item["natural_path"], item["natural_samples"],
                            natural_ref, row_root, self.args, qc_repeat=False)
                        transform = result.get("transform_diagnostics") if isinstance(result, Mapping) else None
                        app_receipt = item["application"]["receipt"]
                        app_ref = item["application"]["receipt_ref"]
                        candidate_wav = item["application"]["output_wav"]
                        if (not isinstance(result, dict) or result.get("status") != "PASS"
                                or result.get("score_audio_sha256") != row["natural_pcm"].get("pcm_sha256")
                                or result.get("natural_pcm_sha256") != row["natural_pcm"].get("pcm_sha256")
                                or result.get("generator_pcm_sha256") != row["generator_condition_input"].get("pcm_sha256")
                                or not isinstance(transform, Mapping)
                                or transform.get("operation") != "use_exact_precomputed_fit_snapshot_waveform_as_wav2lip_input"
                                or transform.get("input_file_sha256") != candidate_wav.get("sha256")
                                or transform.get("input_pcm_float32_sha256") != candidate_wav.get("pcm_float32_sha256")
                                or transform.get("application_receipt_raw_sha256") != app_ref.get("sha256")
                                or transform.get("application_row_receipt_sha256") != app_receipt.get("self_sha256")):
                            raise Dev20AdapterError("Stage-A result differs from selected waveform/natural score pins")
                        delivery = result.get("mux_audio_readback")
                        if (not isinstance(delivery, Mapping) or delivery.get("status") != "PASS"
                                or delivery.get("decoded_float32_pcm_sha256") != row["natural_pcm"].get("pcm_sha256")
                                or delivery.get("sample_count") != row["natural_pcm"].get("sample_count")):
                            raise Dev20AdapterError("native mux did not preserve the original natural PCM and N")
                        arrays = result.get("_internal_arrays")
                        summary = _validate_arrays(arrays)
                        npz_ref = _validate_npz_ref(result, row_root, arrays)
                        public = {key: value for key, value in result.items() if not key.startswith("_internal_")}
                        public["generator_condition_input"] = copy.deepcopy(row["generator_condition_input"])
                        public["selected_spsa_center_binding"] = copy.deepcopy(row["spsa_binding"])
                        public["selected_spsa_center_binding_sha256"] = row["spsa_point_binding_sha256"]
                        base_parent = self.identity_by_key[(str(row["sample_id"]), self.model_id,
                                                           int(row["seed"]), "identity")]
                        public["parent_identity_row_id"] = base_parent["row_id"]
                        public["native_row_evidence"] = {
                            "schema": "cross_tfg_dev20_selected_spsa_native_row_evidence_v1",
                            "native_adapter_source": self.adapter_source_pin,
                            "point_kind": "spsa_center", "source_step": self.selected["source_step"],
                            "plan_sha256": self.context["plan_pin"]["sha256"],
                            "plan_self_sha256": self.plan["plan_sha256"],
                            "pre_effect_freeze_sha256": self.context["freeze_pin"]["sha256"],
                            "pre_effect_freeze_self_sha256": self.freeze["self_sha256"],
                            "application_receipt": item["application"]["receipt_ref"],
                            "official_arrays": summary, "raw_npz": npz_ref,
                            "candidate_scope": {"parent_planned_rows": 540,
                                                "candidate_rows_overall": 180,
                                                "candidate_rows_per_model": 60},
                            "score_audio_is_natural_pcm": True,
                        }
                        row_results.append(public)
                        completed += 1
                    except Exception as exc:
                        failure_seen = True
                        if isinstance(result, dict):
                            failed = {key: value for key, value in result.items() if not key.startswith("_internal_")}
                        else:
                            failed = core._identity_base(row, natural_ref)
                        failed["status"] = "FAILED"
                        failed["failure_reason"] = type(exc).__name__ + ": " + str(exc)
                        failed["selected_spsa_center_binding"] = copy.deepcopy(row.get("spsa_binding"))
                        failed["selected_spsa_center_binding_sha256"] = row.get("spsa_point_binding_sha256")
                        failed["parent_identity_row_id"] = self.identity_by_key[(str(row["sample_id"]), self.model_id,
                                                                                  int(row["seed"]), "identity")]["row_id"]
                        failed["native_row_evidence"] = {
                            "schema": "cross_tfg_dev20_selected_spsa_native_row_evidence_v1",
                            "native_adapter_source": self.adapter_source_pin,
                            "point_kind": "spsa_center", "source_step": self.selected["source_step"],
                            "plan_sha256": self.context["plan_pin"]["sha256"],
                            "pre_effect_freeze_sha256": self.context["freeze_pin"]["sha256"],
                            "candidate_scope": {"parent_planned_rows": 540,
                                                "candidate_rows_overall": 180,
                                                "candidate_rows_per_model": 60},
                        }
                        row_results.append(failed)
                    for record in row_results[-1:]:
                        encoded = (json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                        results_stream.write(encoded)
                        results_stream.flush()
                        os.fsync(results_stream.fileno())
                    if failure_seen:
                        for pending in candidate_rows[index + 1:]:
                            pending_natural = pending["natural_pcm"]
                            parent_id = self.identity_by_key[(str(pending["sample_id"]), self.model_id,
                                                             int(pending["seed"]), "identity")]["row_id"]
                            incomplete = core._identity_base(pending, pending_natural)
                            incomplete["status"] = "INCOMPLETE"
                            incomplete["failure_reason"] = "not attempted after a prior selected-center row failure"
                            incomplete["selected_spsa_center_binding"] = copy.deepcopy(pending.get("spsa_binding"))
                            incomplete["selected_spsa_center_binding_sha256"] = pending.get("spsa_point_binding_sha256")
                            incomplete["parent_identity_row_id"] = parent_id
                            incomplete["native_row_evidence"] = {
                                "schema": "cross_tfg_dev20_selected_spsa_native_row_evidence_v1",
                                "native_adapter_source": self.adapter_source_pin,
                                "point_kind": "spsa_center", "source_step": self.selected["source_step"],
                                "plan_sha256": self.context["plan_pin"]["sha256"],
                                "pre_effect_freeze_sha256": self.context["freeze_pin"]["sha256"],
                                "candidate_scope": {"parent_planned_rows": 540,
                                                    "candidate_rows_overall": 180,
                                                    "candidate_rows_per_model": 60},
                            }
                            row_results.append(incomplete)
                            results_stream.write((json.dumps(incomplete, sort_keys=True, ensure_ascii=False,
                                                            allow_nan=False) + "\n").encode("utf-8"))
                            results_stream.flush()
                            os.fsync(results_stream.fileno())
                        break
                results_stream.close()
            finally:
                core._run_generator = original_generator
                core._frame_clock = original_frame_clock
                for key, previous in saved_env.items():
                    if previous is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = previous

            ledger_raw = _read_regular(ledger_path, "six-row selected candidate ledger", 64 * 1024 * 1024)
            status_counts = {status: sum(row.get("status") == status for row in row_results)
                             for status in ("PASS", "FAILED", "INCOMPLETE")}
            status = "BLOCK_COMPLETE" if status_counts["PASS"] == BLOCK_ROWS else "BLOCK_COMPLETE_WITH_FAILURES"
            manifest = {
                "schema": "cross_tfg_dev20_selected_spsa_native_block_manifest_v1",
                "status": status, "run_id": self.plan.get("run_id"), "model_id": self.model_id,
                "method_id": self.selected["method_id"], "point_kind": "spsa_center",
                "source_step": self.selected["source_step"], "checkpoint_sha256": None,
                "parent_plan": self.context["plan_pin"],
                "pre_effect_freeze": self.context["freeze_pin"],
                "parent_planned_rows": 540, "condition_rows": 180,
                "candidate_rows_overall": 180, "candidate_rows_per_model": 60,
                "block_start_index_within_model_candidate_rows": block_start,
                "block_index_within_model_candidate_rows": block_start // BLOCK_ROWS,
                "block_rows": BLOCK_ROWS, "result_rows": len(row_results),
                "status_counts": status_counts,
                "results_jsonl": {"path": str(ledger_path.resolve(strict=True)),
                                  "bytes": len(ledger_raw), "sha256": _sha(ledger_raw)},
                "resource_gate_count": len(resource_gates), "resource_gates": resource_gates,
                "completed_rows": completed, "started_unix": started, "ended_unix": time.time(),
                "native_runtime": {"python_executable": os.path.realpath(os.sys.executable),
                                   "stagea_core": str(Path(str(self.core.__file__)).resolve(strict=True)),
                                   "native_adapter_source": self.adapter_source_pin,
                                   "runtime_binding": {"path": str(self.runtime_path.resolve(strict=True)),
                                                       "bytes": len(self.runtime_raw),
                                                       "sha256": _sha(self.runtime_raw)}},
                "support_scope": "row-local raw arrays only; no common FULL/GUARD20 reduction or admission",
            }
            manifest["self_sha256"] = _payload_sha(manifest, "self_sha256")
            manifest_raw = (json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n").encode("utf-8")
            manifest_path = output / "result_manifest.json"
            with manifest_path.open("xb") as handle:
                handle.write(manifest_raw)
                handle.flush()
                os.fsync(handle.fileno())
            run_status = {
                "schema": "cross_tfg_dev20_selected_spsa_native_block_status_v1",
                "status": status, "run_id": self.plan.get("run_id"), "model_id": self.model_id,
                "method_id": self.selected["method_id"], "planned_rows": 540,
                "candidate_rows_per_model": 60, "block_start_index": block_start,
                "block_index": block_start // BLOCK_ROWS, "block_rows": BLOCK_ROWS,
                "completed_rows": completed, "result_rows": len(row_results),
                "status_counts": status_counts, "result_manifest_sha256": _sha(manifest_raw),
                "parent_plan_sha256": self.context["plan_pin"]["sha256"],
                "pre_effect_freeze_sha256": self.context["freeze_pin"]["sha256"],
            }
            run_status_path = output / "run_status.json"
            with run_status_path.open("xb") as handle:
                handle.write((json.dumps(run_status, sort_keys=True, ensure_ascii=False, indent=2,
                                         allow_nan=False) + "\n").encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            return {"status": status, "model_id": self.model_id, "block_start_index": block_start,
                    "block_index": block_start // BLOCK_ROWS, "planned_rows": 540,
                    "candidate_rows_overall": 180, "candidate_rows_per_model": 60,
                    "result_rows": len(row_results), "status_counts": status_counts,
                    "output_root": str(output.resolve(strict=True)),
                    "results_jsonl": str(ledger_path.resolve(strict=True)),
                    "result_manifest": str(manifest_path.resolve(strict=True)),
                    "run_status": str(run_status_path.resolve(strict=True))}
        finally:
            _PATCH_LOCK.release()


def load_verified_context(root: Union[str, Path], plan_path: Union[str, Path],
                          pre_effect_freeze_path: Union[str, Path]) -> Dict[str, Any]:
    """Verify a frozen plan/freeze and its application lineage without effects.

    The accepted coordinator must run its full analyzer03 pre-effect verifier
    before constructing this context. This runtime check independently binds
    the exact plan, freeze, builder04, policy/protocol, selected artifact,
    application manifest/receipts, 540-row hash, and each candidate input.
    """
    return _verify_plan_and_freeze(Path(root), Path(plan_path), Path(pre_effect_freeze_path))


__all__ = ["Dev20AdapterError", "Dev20SelectedStageAAdapter", "load_verified_context"]
