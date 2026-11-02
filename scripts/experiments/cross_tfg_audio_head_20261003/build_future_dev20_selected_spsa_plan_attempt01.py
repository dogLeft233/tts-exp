#!/usr/bin/env python3
"""Build a late-bound 540-row Dev20 plan for an admitted SPSA center.

This additive builder leaves the historical DirectD0/step-120 protocol intact.
It requires a freshly qualified, two-condition identity/RMS Stage-A plan for the
same Dev20 sources, a portable selected-center artifact, and the complete set
of 20 verified waveform-application receipts.  It performs no rendering,
scoring, selection, or optimizer work.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

PROTOCOL_REL = Path(".codex/scratch/dev20_future_cohort_protocol_attempt06/future_dev20_three_method_protocol_attempt06.json")
PROTOCOL_RAW_SHA256 = "73ac37eee2c8c728a7123fa85834dc53104a4b809cbc017593bd86939816eb17"
POLICY_REL = Path("openspec/changes/develop-cross-tfg-audio-head-20261003/future-dev20-admission-policy-attempt01.json")
POLICY_RAW_SHA256 = "f215ecff1dc9908eebfc3195a96d76b041bea47ca0fa7df5d64b6481031398ed"
PORTABLE_REL = Path("scripts/audio_enhancement/portable_selected_fit_center_attempt05.py")
PORTABLE_RAW_SHA256 = "29bc744d250924221373a9244a735f3f7c3b2285d1b0d97357e6eda7a965f987"
PORTABLE_BYTES = 58811
SCHEMA = "cross_tfg_future_dev20_selected_spsa_center_protocol_v1"
PLAN_BINDING_SCHEMA = "cross_tfg_future_dev20_selected_spsa_center_plan_binding_v1"
APP_MANIFEST_SCHEMA = "cross_tfg_dev20_selected_spsa_center_application_manifest_v1"
APP_MANIFEST_STATUS = "COMPLETE_SELECTED_SPSA_CENTER_WAVEFORM_APPLICATION_ONLY_NOT_FINAL_DEV"
APP_RECEIPT_SCHEMA = "cross_tfg_v3_portable_selected_center_wav_application_v1"
APP_RECEIPT_STATUS = "PASS_SELECTED_FIT_CENTER_APPLICATION_ONLY_NOT_FINAL_DEV"
SELECTION_STATUS = "ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV"
METHOD_KIND = "fit40_spsa_center"
CONDITION_ID = "selected_spsa_center"
MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
SEEDS = (20261003, 20261004, 20261005)
SPEAKERS = tuple(f"S{i:04d}" for i in range(102, 112))
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


class BuildError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical_bytes({key: item for key, item in value.items() if key != field}))


def read_raw(path: Path, label: str, max_bytes: int = 128 * 1024 * 1024) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise BuildError(f"{label} must be an existing regular non-symlink file: {path}")
    info = path.stat(follow_symlinks=False)
    if info.st_size <= 0 or info.st_size > max_bytes:
        raise BuildError(f"{label} has invalid size {info.st_size}: {path}")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise BuildError(f"{label} changed during read: {path}")
    return raw


def read_json(path: Path, label: str, max_bytes: int = 128 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    raw = read_raw(path, label, max_bytes)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise BuildError(f"{label} JSON root is not an object")
    return value, raw


def verify_self(doc: Mapping[str, Any], field: str, label: str) -> None:
    if doc.get(field) != payload_sha(doc, field):
        raise BuildError(f"{label} payload self-hash is invalid")


def make_ref(path: Path, raw: bytes, self_sha: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve(strict=True)), "bytes": len(raw), "sha256": sha_bytes(raw),
    }
    if self_sha is not None:
        result["self_sha256"] = self_sha
    return result


def _load_portable(root: Path) -> Any:
    path = root / PORTABLE_REL
    raw = read_raw(path, "portable selected-center wrapper", 2 * 1024 * 1024)
    if len(raw) != PORTABLE_BYTES or sha_bytes(raw) != PORTABLE_RAW_SHA256:
        raise BuildError("portable selected-center wrapper does not match the exact attempt05 pin")
    spec = importlib.util.spec_from_file_location("_pinned_dev20_selected_center_portable05", path)
    if spec is None or spec.loader is None:
        raise BuildError("cannot load exact pinned portable selected-center wrapper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._load_bound_modules(root)
    if module._wrapper_source_pin(root).get("sha256") != PORTABLE_RAW_SHA256:
        raise BuildError("loaded portable wrapper does not report its exact relative source pin")
    return module


def _verify_source_protocol(root: Path) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    protocol, protocol_raw = read_json(root / PROTOCOL_REL, "historical cohort protocol")
    if sha_bytes(protocol_raw) != PROTOCOL_RAW_SHA256:
        raise BuildError("historical attempt06 protocol raw bytes differ from its immutable pin")
    if protocol.get("schema") != "cross_tfg_future_dev20_three_method_protocol_attempt06_v1" or protocol.get("status") != "FROZEN_PROSPECTIVE_NOT_RUN":
        raise BuildError("historical attempt06 protocol schema/status mismatch")
    verify_self(protocol, "self_sha256", "historical cohort protocol")
    policy, policy_raw = read_json(root / POLICY_REL, "frozen Dev20 admission policy")
    if sha_bytes(policy_raw) != POLICY_RAW_SHA256:
        raise BuildError("frozen Dev20 policy raw bytes differ from its immutable pin")
    if policy.get("status") != "FROZEN_PROSPECTIVE_NOT_RUN":
        raise BuildError("frozen Dev20 policy is not prospective")
    verify_self(policy, "self_sha256", "frozen Dev20 policy")
    if policy.get("cohort", {}).get("protocol_raw_sha256") != PROTOCOL_RAW_SHA256:
        raise BuildError("policy does not identify the exact historical cohort protocol")
    return protocol, protocol_raw, policy, policy_raw


def _source_rows(protocol: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    cohort = protocol.get("cohort_and_source")
    design = protocol.get("primary_design")
    rows = cohort.get("source_rows") if isinstance(cohort, Mapping) else None
    if not isinstance(rows, list) or len(rows) != 20 or not isinstance(design, Mapping):
        raise BuildError("attempt06 protocol does not contain the exact 20-source/primary-design roster")
    if design.get("total_planned_rows") != 540 or tuple(design.get("renderer_seeds", [])) != SEEDS:
        raise BuildError("attempt06 source design denominator or seed order differs")
    by_id: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise BuildError("protocol source row is not an object")
        sid, speaker = raw.get("sample_id"), raw.get("speaker_id")
        if not isinstance(sid, str) or not sid or sid in by_id or speaker not in SPEAKERS:
            raise BuildError("protocol source sample/speaker identity is missing or duplicated")
        if (type(raw.get("sample_count_N")) is not int or raw["sample_count_N"] <= 0
                or raw.get("sample_rate_hz") != 16000
                or not SHA_RE.fullmatch(str(raw.get("canonical_f32_sha256", "")))
                or not SHA_RE.fullmatch(str(raw.get("source_wav_sha256", "")))):
            raise BuildError(f"protocol source PCM identity is incomplete: {sid}")
        by_id[sid] = raw
        counts[speaker] = counts.get(speaker, 0) + 1
    if len(by_id) != 20 or set(counts) != set(SPEAKERS) or any(v != 2 for v in counts.values()):
        raise BuildError("source roster must be exactly two clips for each frozen speaker")
    if cohort.get("speaker_counts") != counts or cohort.get("selected_speakers") != list(SPEAKERS):
        raise BuildError("protocol speaker roster summary differs from the 20 exact source rows")
    return copy.deepcopy(rows), by_id


def _validate_selected_artifact(portable: Any, path: Path) -> tuple[dict[str, Any], bytes, dict[str, Any], dict[str, Any]]:
    artifact, raw, _selected = portable._load_selected_artifact(path)
    if (artifact.get("selection_status") != SELECTION_STATUS
            or artifact.get("final_dev_or_confirm_admission_performed") is not False
            or artifact.get("native_tfg_qualification_claim") is not False):
        raise BuildError("portable artifact is not only an admitted Fit-center application artifact")
    selected = artifact.get("selected_center")
    if (not isinstance(selected, dict) or selected.get("point_kind") != "spsa_center"
            or selected.get("source_step") not in (5, 10, 15, 20)
            or not isinstance(selected.get("method_id"), str)
            or not SHA_RE.fullmatch(str(selected.get("theta_sha256", "")))
            or not SHA_RE.fullmatch(str(selected.get("artifact_sha256", "")))
            or not SHA_RE.fullmatch(str(selected.get("binding_sha256", "")))
            or not SHA_RE.fullmatch(str(selected.get("input_pins_sha256", "")))):
        raise BuildError("portable artifact lacks the actual admitted SPSA-center identity")
    evidence = artifact.get("evidence")
    if not isinstance(evidence, dict):
        raise BuildError("portable selected artifact lacks embedded selector/index/terminal/plan evidence")
    center_spec = portable._selected_center_from_evidence(
        *_selected_evidence_args(portable, evidence))
    if center_spec != selected:
        raise BuildError("portable selected-center summary differs from its embedded exact evidence")
    return artifact, raw, selected, evidence


def _selected_evidence_args(portable: Any, evidence: Mapping[str, Any]) -> tuple[Any, ...]:
    docs: dict[str, dict[str, Any]] = {}
    for key, self_field in (("final_center_index", "index_sha256"), ("selector_receipt", "receipt_sha256"),
                            ("center_terminal", "terminal_sha256"), ("point_plan", "plan_sha256"),
                            ("seed_application_artifact", "artifact_sha256")):
        docs[key], _raw = portable._verify_embedded_ref(evidence.get(key), key, self_field)
    return (
        docs["final_center_index"], docs["selector_receipt"], docs["center_terminal"],
        docs["point_plan"], docs["seed_application_artifact"],
    )


def _validate_base_plan(root: Path, path: Path, protocol: Mapping[str, Any],
                        source_by_id: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], bytes, dict[tuple[str, str, int, str], dict[str, Any]]]:
    plan, raw = read_json(path, "identity/RMS Stage-A plan")
    if (plan.get("schema_version") != "cross-tfg-audio-eval-v1" or plan.get("status") != "FROZEN"
            or plan.get("plan_sha256") != payload_sha(plan, "plan_sha256")):
        raise BuildError("identity/RMS source plan is not a canonical frozen Stage-A plan")
    design = protocol["primary_design"]
    models = plan.get("models")
    models_by_id = {str(item.get("model_id")): item for item in models if isinstance(item, dict)} if isinstance(models, list) else {}
    protocol_models = {str(item.get("model_id")): item for item in design.get("models", []) if isinstance(item, dict)}
    if set(models_by_id) != set(MODELS) or set(protocol_models) != set(MODELS):
        raise BuildError("identity/RMS plan and frozen protocol do not bind exactly the three model descriptors")
    for model_id in MODELS:
        if models_by_id[model_id].get("audio_frontend") != protocol_models[model_id].get("audio_frontend"):
            raise BuildError(f"model frontend differs from frozen protocol: {model_id}")
    methods = plan.get("methods")
    if not isinstance(methods, list) or len(methods) != 2:
        raise BuildError("base plan must contain exactly identity and fixed-RMS methods")
    method_by_id = {str(m.get("method_id")): m for m in methods if isinstance(m, dict)}
    if set(method_by_id) != {"identity", "rms_fixed"}:
        raise BuildError("base method table is not identity plus rms_fixed")
    for method_id, kind in (("identity", "identity"), ("rms_fixed", "rms_fixed")):
        method = method_by_id[method_id]
        if (method.get("kind") != kind or method.get("checkpoint_sha256") is not None
                or method.get("config_sha256") != sha_bytes(canonical_bytes(method.get("config")))):
            raise BuildError(f"base method config is not pinned correctly: {method_id}")
    rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) != 360:
        raise BuildError("fresh two-condition Stage-A base plan must contain exactly 360 rows")
    row_by_key: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise BuildError("base plan contains a non-object row")
        sid = row.get("sample_id")
        source = source_by_id.get(str(sid))
        key = (str(sid), str(row.get("model_id")), row.get("seed"), str(row.get("method_id")))
        if key in row_by_key or source is None or key[1] not in MODELS or type(key[2]) is not int or key[2] not in SEEDS:
            raise BuildError(f"base plan row key is duplicate or outside the frozen 20×3×3 family: {key}")
        method = method_by_id.get(key[3])
        natural = row.get("natural_pcm")
        if (method is None or row.get("split") != "dev" or row.get("speaker_id") != source.get("speaker_id")
                or row.get("face_id") != design.get("face_id") or row.get("audio_frontend") != models_by_id[key[1]].get("audio_frontend")
                or row.get("method_config_sha256") != method.get("config_sha256") or row.get("checkpoint_sha256") is not None
                or not isinstance(natural, dict) or natural.get("file_sha256") != source.get("source_wav_sha256")
                or natural.get("pcm_sha256") != source.get("canonical_f32_sha256")
                or natural.get("sample_count") != source.get("sample_count_N")
                or natural.get("sample_rate_hz") != 16000 or natural.get("channels") != 1
                or row.get("score_audio") != natural):
            raise BuildError(f"base identity/RMS row does not preserve exact original source/score PCM: {key}")
        if key[3] == "identity" and row.get("method_config_sha256") != method_by_id["identity"].get("config_sha256"):
            raise BuildError("identity row points to a different method config")
        row_by_key[key] = row
    expected = {(sid, model, seed, method) for sid in source_by_id for model in MODELS for seed in SEEDS for method in ("identity", "rms_fixed")}
    if set(row_by_key) != expected:
        raise BuildError("base plan is not the exact 20×3×3×2 identity/RMS denominator")
    supports = {x.get("support_id"): x for x in plan.get("supports", []) if isinstance(x, dict)}
    if (set(supports) != {"full", "guard20"} or supports["full"].get("trim_each_end") != 0
            or supports["guard20"].get("trim_each_end") != 20
            or supports["full"].get("minimum_queries") != 1
            or supports["guard20"].get("minimum_queries") != 41):
        raise BuildError("base plan support definitions differ from frozen FULL/GUARD20")
    syncnet = plan.get("protocol", {}).get("syncnet", {})
    if (syncnet.get("lag_offsets_frames") != list(range(-15, 16))
            or syncnet.get("anchor_column") != 15 or syncnet.get("zero_lag_column") != 15
            or plan.get("protocol", {}).get("roi", {}).get("mode") != "native_official"):
        raise BuildError("base native score protocol must retain official ROI, 31 lags and anchor column 15")
    _verify_model_source_closure(plan, root, models_by_id)
    return plan, raw, row_by_key


def _verify_model_source_closure(plan: Mapping[str, Any], root: Path, models_by_id: Mapping[str, Mapping[str, Any]]) -> None:
    execution = plan.get("execution")
    if not isinstance(execution, dict):
        raise BuildError("fresh base plan lacks its model execution closure")
    entries = execution.get("source_model_plans")
    if not isinstance(entries, list) or len(entries) != 3:
        raise BuildError("base execution must bind three independently pinned model plans")
    by_model = {str(item.get("model_id")): item for item in entries if isinstance(item, dict)}
    if set(by_model) != set(MODELS):
        raise BuildError("base model source plan table is not the exact three-model set")
    runtime_refs = execution.get("runtime_bindings_by_model")
    per_model_exec = execution.get("source_model_execution")
    protocols = execution.get("source_model_protocols")
    if any(not isinstance(value, dict) or set(value) != set(MODELS) for value in (runtime_refs, per_model_exec, protocols)):
        raise BuildError("base plan lacks complete per-model runtime, execution and protocol maps")
    for model_id in MODELS:
        entry = by_model[model_id]
        for role in ("plan", "planned_rows"):
            ref = entry.get(role)
            if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
                raise BuildError(f"base source plan lacks exact {role} reference for {model_id}")
            path = Path(ref["path"])
            raw = read_raw(path, f"{model_id} {role}", 128 * 1024 * 1024)
            if len(raw) != ref.get("bytes") or sha_bytes(raw) != ref.get("sha256"):
                raise BuildError(f"base source {role} bytes differ from bound reference: {model_id}")
            if role == "plan":
                source_plan, _ = read_json(path, f"{model_id} model plan", 128 * 1024 * 1024)
                if (source_plan.get("schema_version") != "cross-tfg-audio-eval-v1"
                        or source_plan.get("status") != "FROZEN"
                        or source_plan.get("plan_sha256") != payload_sha(source_plan, "plan_sha256")
                        or source_plan.get("models") != [models_by_id[model_id]]
                        or source_plan.get("rows") is None or len(source_plan["rows"]) != 120
                        or source_plan.get("methods") != plan.get("methods")):
                    raise BuildError(f"model source plan not exact 120-row identity/RMS plan: {model_id}")
                if entry.get("plan_run_id") != source_plan.get("run_id"):
                    raise BuildError(f"model source plan run ID differs: {model_id}")
            else:
                try:
                    parsed = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise BuildError(f"model planned rows are invalid JSONL: {model_id}") from exc
                if len(parsed) != 120:
                    raise BuildError(f"model planned-row denominator is not 120: {model_id}")
                source_plan_path = Path(str(entry["plan"]["path"]))
                source_plan, _ = read_json(source_plan_path, f"{model_id} model plan", 128 * 1024 * 1024)
                if parsed != source_plan.get("rows"):
                    raise BuildError(f"model planned-row ledger differs from source plan rows: {model_id}")
        bind = runtime_refs.get(model_id)
        if not isinstance(bind, dict) or not isinstance(bind.get("path"), str):
            raise BuildError(f"runtime binding ref missing for {model_id}")
        runtime_raw = read_raw(Path(bind["path"]), f"{model_id} runtime binding", 16 * 1024 * 1024)
        if (bind.get("bytes") is not None and len(runtime_raw) != bind.get("bytes")
                or sha_bytes(runtime_raw) != bind.get("raw_sha256")):
            raise BuildError(f"runtime binding raw bytes differ for {model_id}")
        model_exec = per_model_exec[model_id]
        qualification = model_exec.get("runtime_qualification") if isinstance(model_exec, dict) else None
        if not isinstance(qualification, dict) or qualification.get("status", "").startswith("PASS") is not True:
            raise BuildError(f"model lacks a positive source-side runtime qualification receipt: {model_id}")
        qual_path, qual_sha = qualification.get("path"), qualification.get("raw_sha256")
        if not isinstance(qual_path, str) or not isinstance(qual_sha, str):
            raise BuildError(f"runtime qualification receipt lacks raw path/SHA: {model_id}")
        if sha_bytes(read_raw(Path(qual_path), f"{model_id} runtime qualification", 16 * 1024 * 1024)) != qual_sha:
            raise BuildError(f"runtime qualification source receipt differs from its pin: {model_id}")
        if (model_exec.get("candidate_selection") not in ("not performed", "NOT_PERFORMED")
                or model_exec.get("dev_or_confirm_access") is not False):
            raise BuildError(f"source model execution closure claims prior candidate selection/effects: {model_id}")


def _validate_application_manifest(path: Path, protocol: Mapping[str, Any], source_by_id: Mapping[str, Mapping[str, Any]],
                                   portable: Any, artifact_path: Path, artifact: Mapping[str, Any], artifact_raw: bytes,
                                   selected: Mapping[str, Any]) -> tuple[dict[str, Any], bytes, dict[str, dict[str, Any]]]:
    manifest, raw = read_json(path, "selected-center application manifest", 16 * 1024 * 1024)
    if (manifest.get("schema") != APP_MANIFEST_SCHEMA or manifest.get("status") != APP_MANIFEST_STATUS
            or manifest.get("selection_status") != SELECTION_STATUS
            or manifest.get("native_tfg_qualification_claim") is not False
            or manifest.get("final_dev_or_confirm_admission_performed") is not False):
        raise BuildError("application manifest is not the exact pre-Dev selected-center application contract")
    verify_self(manifest, "self_sha256", "application manifest")
    artifact_pin = manifest.get("selected_artifact")
    actual_artifact_ref = make_ref(artifact_path, artifact_raw, artifact.get("artifact_sha256"))
    if artifact_pin != {**actual_artifact_ref, "artifact_sha256": artifact.get("artifact_sha256")}:
        raise BuildError("application manifest does not bind the exact portable selected-center artifact")
    if manifest.get("selected_center") != selected:
        raise BuildError("application manifest center identity differs from portable artifact")
    if manifest.get("source_protocol_raw_sha256") != PROTOCOL_RAW_SHA256:
        raise BuildError("application manifest does not bind the exact 20-source protocol")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 20:
        raise BuildError("application manifest must retain every one of the 20 source clips")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise BuildError("application manifest contains a non-object sample")
        sid = row.get("sample_id")
        source = source_by_id.get(str(sid))
        if sid in by_id or source is None:
            raise BuildError(f"application manifest has duplicate or unexpected sample {sid}")
        receipt_ref = row.get("receipt")
        if not isinstance(receipt_ref, dict) or not isinstance(receipt_ref.get("path"), str):
            raise BuildError(f"application receipt ref is missing for {sid}")
        receipt_path = Path(receipt_ref["path"])
        receipt, receipt_raw = read_json(receipt_path, f"application receipt {sid}", 8 * 1024 * 1024)
        verify_self(receipt, "self_sha256", f"application receipt {sid}")
        if (receipt_ref.get("bytes") != len(receipt_raw) or receipt_ref.get("sha256") != sha_bytes(receipt_raw)
                or receipt_ref.get("self_sha256") != receipt.get("self_sha256")
                or receipt.get("schema") != APP_RECEIPT_SCHEMA or receipt.get("status") != APP_RECEIPT_STATUS
                or receipt.get("selection_status") != SELECTION_STATUS
                or receipt.get("final_dev_or_confirm_admission_performed") is not False
                or receipt.get("native_tfg_qualification_claim") is not False):
            raise BuildError(f"application receipt raw/self/schema/status mismatch for {sid}")
        selected_artifact_ref = receipt.get("selected_artifact")
        if selected_artifact_ref != {**actual_artifact_ref, "artifact_sha256": artifact.get("artifact_sha256")}:
            raise BuildError(f"application receipt does not bind the selected artifact for {sid}")
        if receipt.get("selected_center") != selected:
            raise BuildError(f"application receipt selected center differs for {sid}")
        input_wav, output_wav, clock = receipt.get("input_wav"), receipt.get("output_wav"), receipt.get("clock")
        if (not isinstance(input_wav, dict) or not isinstance(output_wav, dict) or not isinstance(clock, dict)
                or input_wav.get("file_sha256") != source.get("source_wav_sha256")
                or input_wav.get("pcm_float32_sha256") != source.get("canonical_f32_sha256")
                or input_wav.get("sample_count") != source.get("sample_count_N")
                or input_wav.get("sample_rate_hz") != 16000 or input_wav.get("channels") != 1
                or output_wav.get("encoding") != "FLOAT_32"
                or output_wav.get("sample_count") != source.get("sample_count_N")
                or output_wav.get("sample_rate_hz") != 16000 or output_wav.get("channels") != 1
                or output_wav.get("pcm_float32_sha256") is None
                or clock.get("preserved") is not True
                or clock.get("input_samples") != source.get("sample_count_N")
                or clock.get("output_samples") != source.get("sample_count_N")):
            raise BuildError(f"application receipt does not preserve source PCM/N/clock for {sid}")
        rms, projection = receipt.get("rms"), receipt.get("projection")
        if (not isinstance(rms, dict) or rms.get("peak_limiter_applied") is not False
                or not isinstance(rms.get("output_rms_error"), (int, float))
                or rms.get("output_rms_error") > 1e-6
                or not isinstance(projection, dict) or projection.get("within_bound") is not True
                or projection.get("bound_db") != 3.0):
            raise BuildError(f"application RMS/projection gate is not PASS for {sid}")
        output_path = Path(str(output_wav.get("path", "")))
        output_raw = read_raw(output_path, f"applied FLOAT_32 WAV {sid}", 8 * 1024 * 1024)
        if (len(output_raw) != output_wav.get("bytes") or sha_bytes(output_raw) != output_wav.get("sha256")
                or row.get("output_wav") != output_wav
                or row.get("source_pcm_sha256") != source.get("canonical_f32_sha256")
                or row.get("source_wav_sha256") != source.get("source_wav_sha256")
                or row.get("sample_count_N") != source.get("sample_count_N")):
            raise BuildError(f"application output full bytes/row mapping differ for {sid}")
        if (receipt.get("theta_little_endian_f32_sha256") is None
                or selected.get("theta_sha256") != manifest.get("theta_sha256", selected.get("theta_sha256"))):
            raise BuildError(f"application manifest/receipt lacks exact selected theta identity for {sid}")
        by_id[sid] = {
            "sample_id": sid,
            "source_wav_sha256": source["source_wav_sha256"],
            "source_pcm_sha256": source["canonical_f32_sha256"],
            "sample_count_N": source["sample_count_N"],
            "output_wav": copy.deepcopy(output_wav),
            "receipt_ref": make_ref(receipt_path, receipt_raw, receipt["self_sha256"]),
        }
    if set(by_id) != set(source_by_id):
        raise BuildError("application manifest does not cover the complete frozen source roster")
    return manifest, raw, by_id


def _validate_method_config(selected: Mapping[str, Any], evidence: Mapping[str, Any], portable: Any) -> dict[str, Any]:
    point_plan, _raw = portable._verify_embedded_ref(evidence["point_plan"], "point plan", "plan_sha256")
    method_matches = [m for m in point_plan.get("methods", []) if isinstance(m, dict) and m.get("method_id") == selected.get("method_id")]
    if len(method_matches) != 1:
        raise BuildError("selected point plan does not retain one exact center method")
    method = copy.deepcopy(method_matches[0])
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
            or method.get("config_sha256") != sha_bytes(canonical_bytes(config))):
        raise BuildError("selected center method config does not preserve its Fit selector identity")
    return method


def _condition_overlay(protocol: Mapping[str, Any], base_plan_path: Path, base_plan_raw: bytes,
                       policy_raw: bytes, selected: Mapping[str, Any], artifact_path: Path, artifact_raw: bytes,
                       app_path: Path, app_raw: bytes, app_manifest: Mapping[str, Any], run_id: str,
                       output_root: Path, model_plan: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    design = copy.deepcopy(protocol["primary_design"])
    design["conditions"] = [
        {"condition_id": "identity", "method_id": "identity", "role": "original-natural-audio generator and score anchor", "training": "none"},
        {"condition_id": "rms_fixed", "method_id": "rms_fixed", "role": "fixed-RMS baseline", "training": "none"},
        {"condition_id": CONDITION_ID, "method_id": selected["method_id"],
         "role": "one preselected complete-Fit SPSA center applied as the generator-only waveform",
         "training": "no new training; checkpoint_sha256 remains null"},
    ]
    design["candidate_checkpoint"] = {
        "bytes": None, "method_id": selected["method_id"], "sha256": None,
        "selection_status": SELECTION_STATUS, "point_kind": "spsa_center",
        "source_step": selected["source_step"], "checkpoint_sha256": None,
        "theta_sha256": selected["theta_sha256"], "artifact_sha256": selected["artifact_sha256"],
        "binding_sha256": selected["binding_sha256"], "input_pins_sha256": selected["input_pins_sha256"],
        "candidate_is_frozen_before_any_future_dev20_effect_is_read": True,
    }
    source_rows = protocol["cohort_and_source"]["source_rows"]
    jobs = []
    for old in protocol["primary_design"]["run_jobs"]:
        old_condition = old["condition_id"]
        condition = CONDITION_ID if old_condition == "direct_d0_step120" else old_condition
        method_id = selected["method_id"] if condition == CONDITION_ID else condition
        jobs.append({
            "condition_id": condition, "model_id": old["model_id"], "seed": old["seed"],
            "run_id": f"{run_id}_{condition}_{old['model_id']}_s{old['seed']}",
            "output_root": str(output_root / condition / old["model_id"] / f"seed_{old['seed']}"),
            "method_id": method_id,
        })
    design["run_jobs"] = jobs
    design["planned_row_formula"] = "20 frozen source rows × 3 conditions × 3 native models × 3 seeds = 540"
    design["conditions"][-1]["application_manifest_sha256"] = sha_bytes(app_raw)
    policy_candidate = policy.get("cohort", {}).get("conditions", [])
    old_thresholds = {
        "asr_thresholds": copy.deepcopy(policy.get("asr_gate", {}).get("thresholds")),
        "primary_thresholds": copy.deepcopy(policy.get("primary_acceptance", policy.get("primary_admission", {}))),
        "bootstrap": copy.deepcopy(policy.get("simultaneous_descriptive_family", {}).get("bootstrap")),
    }
    overlay = {
        "schema": SCHEMA,
        "status": "FROZEN_SELECTED_CENTER_BEFORE_DEV20_RENDER_OR_SCORE",
        "selection_status": SELECTION_STATUS,
        "native_tfg_qualification_claim": False,
        "final_dev_or_confirm_admission_performed": False,
        "score_effects_read": False,
        "confirm_data_or_effects_accessed": False,
        "source_protocol_attempt06": {
            "path": PROTOCOL_REL.as_posix(), "bytes": len(canonical_bytes(protocol)),
            "sha256": PROTOCOL_RAW_SHA256, "self_sha256": protocol.get("self_sha256"),
        },
        "source_policy_attempt01": {"path": POLICY_REL.as_posix(), "bytes": len(policy_raw), "sha256": POLICY_RAW_SHA256,
                                    "self_sha256": policy.get("self_sha256")},
        "source_condition_before_overlay": copy.deepcopy(policy_candidate),
        "frozen_thresholds_and_family": old_thresholds,
        "cohort_and_source": copy.deepcopy(protocol["cohort_and_source"]),
        "primary_design": design,
        "selected_spsa_center": copy.deepcopy(dict(selected)),
        "portable_artifact": make_ref(artifact_path, artifact_raw, None) | {"artifact_sha256": selected["artifact_sha256"]},
        "application_manifest": make_ref(app_path, app_raw, app_manifest.get("self_sha256")),
        "identity_rms_base_plan": make_ref(base_plan_path, base_plan_raw, model_plan.get("plan_sha256")),
        "generator_condition_route": {
            "scope": "generator_condition_only",
            "source": "same portable selected-center FLOAT_32 waveform for every model and seed per sample",
            "score_audio": "exact original natural PCM for every condition",
            "sample_rate_hz": 16000, "channels": 1, "sample_count_N_preserved": True,
            "source_clock_preserved": True, "checkpoint_sha256": None,
        },
        "required_freeze_before_effect_access": [
            "selector receipt and final-center index raw/self pins",
            "selected center terminal and point plan raw/self pins",
            "portable selected artifact and all 20 application receipt/output pins",
            "identity/RMS plan and per-model native runtime source closure",
            "complete 540-row plan raw/self pin, commit and UTC freeze time",
        ],
    }
    overlay["self_sha256"] = payload_sha(overlay, "self_sha256")
    return overlay


def _make_candidate_row(identity: Mapping[str, Any], selected: Mapping[str, Any], method: Mapping[str, Any],
                        application: Mapping[str, Any], app_ref: Mapping[str, Any], evidence_refs: Mapping[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(dict(identity))
    row["method_id"] = selected["method_id"]
    row["method_config_sha256"] = method["config_sha256"]
    row["checkpoint_sha256"] = None
    row["row_id"] = "_".join((str(row["split"]), str(row["speaker_id"]), str(row["sample_id"]),
                              str(row["model_id"]), str(row["face_id"]), str(row["seed"]),
                              str(selected["method_id"])))
    row["generation_settings"] = copy.deepcopy(row.get("generation_settings", {}))
    row["generation_settings"]["input_audio"] = "precomputed_selected_spsa_center_waveform"
    output = copy.deepcopy(application["output_wav"])
    row["generator_condition_input"] = {
        "path": output["path"], "bytes": output["bytes"], "sha256": output["sha256"],
        "file_sha256": output["sha256"], "pcm_sha256": output["pcm_float32_sha256"],
        "sample_count": output["sample_count"], "sample_rate_hz": output["sample_rate_hz"],
        "channels": output["channels"], "encoding": output["encoding"],
        "application_receipt_raw_sha256": app_ref["sha256"],
        "application_row_receipt_sha256": app_ref["self_sha256"],
    }
    row["spsa_point_binding_sha256"] = selected["binding_sha256"]
    row["spsa_binding"] = {
        "schema": "cross_tfg_dev20_selected_spsa_center_row_binding_v1",
        "point_kind": "spsa_center", "source_step": selected["source_step"],
        "method_id": selected["method_id"], "method_config_sha256": method["config_sha256"],
        "theta_sha256": selected["theta_sha256"], "artifact_sha256": selected["artifact_sha256"],
        "binding_sha256": selected["binding_sha256"], "input_pins_sha256": selected["input_pins_sha256"],
        "application_receipt": copy.deepcopy(dict(app_ref)),
        "selection_evidence": copy.deepcopy(dict(evidence_refs)),
        "generator_condition_input": copy.deepcopy(row["generator_condition_input"]),
        "score_audio_is_natural_pcm": row.get("score_audio") == row.get("natural_pcm"),
    }
    return row


def build_plan(*, root: Path, base_plan_path: Path, artifact_path: Path, application_manifest_path: Path,
               run_id: str, output_root: Path, out_plan: Path, out_protocol: Path) -> dict[str, Any]:
    if not RUN_ID_RE.fullmatch(run_id):
        raise BuildError("run_id must be a short stable slug")
    for path, label in ((out_plan, "plan output"), (out_protocol, "protocol overlay output")):
        if path.exists() or path.is_symlink() or path.parent.is_symlink() or not path.parent.is_dir():
            raise BuildError(f"{label} must be absent under an existing real parent: {path}")
    if out_plan.resolve(strict=False) == out_protocol.resolve(strict=False):
        raise BuildError("plan and protocol overlay outputs must be distinct files")
    if output_root.exists() or output_root.is_symlink():
        raise BuildError("future Dev20 output root must be fresh and absent")
    protocol, protocol_raw, policy, policy_raw = _verify_source_protocol(root)
    source_rows, source_by_id = _source_rows(protocol)
    portable = _load_portable(root)
    artifact, artifact_raw, selected, evidence = _validate_selected_artifact(portable, artifact_path)
    method = _validate_method_config(selected, evidence, portable)
    manifest, manifest_raw, application_by_id = _validate_application_manifest(
        application_manifest_path, protocol, source_by_id, portable, artifact_path, artifact, artifact_raw, selected)
    base_plan, base_raw, base_by_key = _validate_base_plan(root, base_plan_path, protocol, source_by_id)
    if base_plan_path.resolve(strict=True) in (out_plan.resolve(strict=False), out_protocol.resolve(strict=False)):
        raise BuildError("output aliases an input plan")

    overlay = _condition_overlay(protocol, base_plan_path, base_raw, policy_raw, selected,
                                 artifact_path, artifact_raw, application_manifest_path, manifest_raw,
                                 manifest, run_id, output_root, base_plan, policy)
    overlay_raw = json.dumps(overlay, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    overlay_pin = {"path": str(out_protocol.resolve(strict=False)), "bytes": len(overlay_raw),
                   "sha256": sha_bytes(overlay_raw), "self_sha256": overlay["self_sha256"]}
    evidence_refs = {
        key: {k: v for k, v in ref.items() if k != "raw_utf8"}
        for key, ref in evidence.items() if isinstance(ref, dict)
    }
    planned_rows = []
    for job in overlay["primary_design"]["run_jobs"]:
        cond, model_id, seed = job["condition_id"], job["model_id"], job["seed"]
        for source in source_rows:
            sid = source["sample_id"]
            if cond == CONDITION_ID:
                identity = base_by_key[(sid, model_id, seed, "identity")]
                row = _make_candidate_row(identity, selected, method, application_by_id[sid],
                                          application_by_id[sid]["receipt_ref"], evidence_refs)
            else:
                row = copy.deepcopy(base_by_key[(sid, model_id, seed, cond)])
            planned_rows.append(row)
    if len(planned_rows) != 540 or len({row.get("row_id") for row in planned_rows}) != 540:
        raise BuildError("constructed plan does not preserve the complete unique 540-row denominator")
    method_table = [copy.deepcopy(m) for m in base_plan["methods"]]
    method_table.append(copy.deepcopy(method))
    plan = copy.deepcopy(base_plan)
    plan["run_id"] = run_id
    plan["methods"] = method_table
    plan["rows"] = planned_rows
    plan["cohort_and_source"] = copy.deepcopy(protocol["cohort_and_source"])
    plan["dev20_selected_spsa_center"] = {
        "schema": PLAN_BINDING_SCHEMA,
        "condition_id": CONDITION_ID,
        "selection_status": SELECTION_STATUS,
        "score_effects_read": False,
        "confirm_data_or_effects_accessed": False,
        "source_protocol_attempt06": {"bytes": len(protocol_raw), "sha256": sha_bytes(protocol_raw),
                                       "self_sha256": protocol["self_sha256"]},
        "source_policy_attempt01": {"bytes": len(policy_raw), "sha256": sha_bytes(policy_raw),
                                     "self_sha256": policy["self_sha256"]},
        "identity_rms_base_plan": make_ref(base_plan_path, base_raw, base_plan["plan_sha256"]),
        "protocol_overlay": overlay_pin,
        "selected_center": copy.deepcopy(selected),
        "selected_artifact": make_ref(artifact_path, artifact_raw) | {"artifact_sha256": selected["artifact_sha256"]},
        "application_manifest": make_ref(application_manifest_path, manifest_raw, manifest["self_sha256"]),
        "checkpoint_sha256": None,
        "freeze_required_before_any_effect": True,
    }
    execution = plan.get("execution")
    if not isinstance(execution, dict):
        raise BuildError("base plan lacks execution object")
    execution["planned_rows"] = 540
    execution["run_jobs"] = copy.deepcopy(overlay["primary_design"]["run_jobs"])
    execution["output_root"] = str(output_root.resolve(strict=False))
    execution["candidate_selection"] = "fixed_before_dev20_effect_access"
    execution["dev_or_confirm_access"] = False
    execution["score_effects_read"] = False
    plan["plan_source_scope"] = "selected-spsa-center additive Dev20 plan; inherited identity/RMS and 20-source cohort are byte-bound"
    plan["status"] = "FROZEN"
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = payload_sha(plan, "plan_sha256")
    plan_raw = json.dumps(plan, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    # Recheck the exact source inputs after all derivation and before either create-only write.
    if read_raw(artifact_path, "selected artifact recheck") != artifact_raw:
        raise BuildError("portable selected-center artifact changed during plan derivation")
    if read_raw(application_manifest_path, "application manifest recheck", 16 * 1024 * 1024) != manifest_raw:
        raise BuildError("application manifest changed during plan derivation")
    if read_raw(base_plan_path, "base plan recheck") != base_raw:
        raise BuildError("identity/RMS base plan changed during plan derivation")
    _write_exclusive(out_protocol, overlay_raw, "protocol overlay")
    _write_exclusive(out_plan, plan_raw, "Dev20 plan")
    return {
        "status": "PLAN_FROZEN_BEFORE_DEV20_EFFECT_ACCESS",
        "selection_status": SELECTION_STATUS,
        "plan": {"path": str(out_plan.resolve(strict=True)), "bytes": len(plan_raw),
                 "sha256": sha_bytes(plan_raw), "self_sha256": plan["plan_sha256"]},
        "protocol_overlay": {"path": str(out_protocol.resolve(strict=True)), "bytes": len(overlay_raw),
                             "sha256": sha_bytes(overlay_raw), "self_sha256": overlay["self_sha256"]},
        "planned_rows": 540, "rows_per_condition": 180, "rows_per_model_condition": 60,
        "candidate_method_id": selected["method_id"], "source_step": selected["source_step"],
        "checkpoint_sha256": None, "score_effects_read": False, "confirm_data_or_effects_accessed": False,
    }


def _write_exclusive(path: Path, raw: bytes, label: str) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
    except FileExistsError as exc:
        raise BuildError(f"{label} output already exists: {path}") from exc


def create_pre_effect_freeze(*, root: Path, plan_path: Path, protocol_path: Path,
                             frozen_commit: str, frozen_at_utc: str, out_path: Path) -> dict[str, Any]:
    if out_path.exists() or out_path.is_symlink() or out_path.parent.is_symlink() or not out_path.parent.is_dir():
        raise BuildError("freeze receipt output must be absent under an existing real parent")
    plan, plan_raw = read_json(plan_path, "selected-center Dev20 plan")
    overlay, overlay_raw = read_json(protocol_path, "selected-center protocol overlay")
    verify_self(plan, "plan_sha256", "Dev20 plan")
    verify_self(overlay, "self_sha256", "Dev20 protocol overlay")
    binding = plan.get("dev20_selected_spsa_center")
    if (plan.get("schema_version") != "cross-tfg-audio-eval-v1" or plan.get("status") != "FROZEN"
            or not isinstance(binding, dict) or binding.get("schema") != PLAN_BINDING_SCHEMA
            or binding.get("checkpoint_sha256") is not None or len(plan.get("rows", [])) != 540
            or overlay.get("schema") != SCHEMA or overlay.get("selection_status") != SELECTION_STATUS
            or overlay.get("score_effects_read") is not False
            or overlay.get("confirm_data_or_effects_accessed") is not False
            or binding.get("protocol_overlay", {}).get("sha256") != sha_bytes(overlay_raw)):
        raise BuildError("plan/overlay cannot be frozen as a complete selected-center pre-effect candidate")
    if (not isinstance(frozen_commit, str) or not frozen_commit
            or not re.fullmatch(r"[A-Za-z0-9._/-]{7,128}", frozen_commit)
            or not isinstance(frozen_at_utc, str)
            or not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z", frozen_at_utc)):
        raise BuildError("freeze requires the actual frozen commit and UTC timestamp")
    receipt = {
        "schema": "cross_tfg_future_dev20_selected_spsa_center_pre_effect_freeze_v1",
        "status": "FROZEN_BEFORE_DEV20_SCORE_ACCESS",
        "selection_status": SELECTION_STATUS,
        "candidate_method_id": binding["selected_center"]["method_id"],
        "method_config_sha256": next(m["config_sha256"] for m in plan["methods"]
                                      if m.get("method_id") == binding["selected_center"]["method_id"]),
        "checkpoint_sha256": None,
        "point_kind": "spsa_center", "source_step": binding["selected_center"]["source_step"],
        "theta_sha256": binding["selected_center"]["theta_sha256"],
        "artifact_sha256": binding["selected_center"]["artifact_sha256"],
        "binding_sha256": binding["selected_center"]["binding_sha256"],
        "input_pins_sha256": binding["selected_center"]["input_pins_sha256"],
        "plan": make_ref(plan_path, plan_raw, plan["plan_sha256"]),
        "protocol_overlay": make_ref(protocol_path, overlay_raw, overlay["self_sha256"]),
        "selected_artifact": binding["selected_artifact"],
        "application_manifest": binding["application_manifest"],
        "source_protocol_attempt06": binding["source_protocol_attempt06"],
        "source_policy_attempt01": binding["source_policy_attempt01"],
        "frozen_commit": frozen_commit, "frozen_at_utc": frozen_at_utc,
        "score_effects_read": False, "confirm_data_or_effects_accessed": False,
        "dev_or_confirm_admission_performed": False,
    }
    receipt["self_sha256"] = payload_sha(receipt, "self_sha256")
    raw = json.dumps(receipt, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    _write_exclusive(out_path, raw, "pre-effect freeze receipt")
    return {"status": receipt["status"], "path": str(out_path.resolve(strict=True)),
            "bytes": len(raw), "sha256": sha_bytes(raw), "self_sha256": receipt["self_sha256"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--base-identity-rms-plan", type=Path, required=True)
    build.add_argument("--selected-artifact", type=Path, required=True)
    build.add_argument("--application-manifest", type=Path, required=True)
    build.add_argument("--run-id", required=True)
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--out-plan", type=Path, required=True)
    build.add_argument("--out-protocol-overlay", type=Path, required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument("--plan", type=Path, required=True)
    freeze.add_argument("--protocol-overlay", type=Path, required=True)
    freeze.add_argument("--frozen-commit", required=True)
    freeze.add_argument("--frozen-at-utc", required=True)
    freeze.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_plan(root=args.root.resolve(strict=True), base_plan_path=args.base_identity_rms_plan,
                                artifact_path=args.selected_artifact, application_manifest_path=args.application_manifest,
                                run_id=args.run_id, output_root=args.output_root,
                                out_plan=args.out_plan, out_protocol=args.out_protocol_overlay)
        else:
            result = create_pre_effect_freeze(root=args.root.resolve(strict=True), plan_path=args.plan,
                                              protocol_path=args.protocol_overlay,
                                              frozen_commit=args.frozen_commit,
                                              frozen_at_utc=args.frozen_at_utc, out_path=args.out)
    except Exception as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
