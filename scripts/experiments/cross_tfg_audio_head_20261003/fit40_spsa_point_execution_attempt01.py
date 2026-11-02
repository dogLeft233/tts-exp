#!/usr/bin/env python3
"""Run one Fit SPSA point on one native backend and seal complete row evidence.

This is the per-child execution layer only.  The parent dispatcher must first
verify the complete initial 720-row family and exact initial selector receipt,
then initialize/replay the controller and materialize a one-method point plan.
This module never asks/tells SPSA, selects, updates a center, or touches
Dev/Confirm.  One invocation handles one backend and block_index 0: 8 ordered
rows for a proposal or 40 ordered rows for a center.  Failures retain one
ledger record for every planned row and never trigger replay.

The model-specific caller constructs its already-pinned adapter, then calls
``execute_point_block``.  A separate parent process calls
``write_child_execution_receipt`` after wait() returns; the child status never
claims its own numeric parent-captured exit code.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


FAMILY_SCHEMA = "cross-tfg-audio-eval-v1"
POINT_STATUS_SCHEMA = "fit40_spsa_point_run_status_v1"
POINT_MANIFEST_SCHEMA = "fit40_spsa_point_result_manifest_v1"
CHILD_RECEIPT_SCHEMA = "fit40_spsa_point_child_execution_receipt_v1"
EXECUTION_PACKAGE_SCHEMA = "fit40_spsa_point_execution_package_manifest_v1"
INITIAL_RECEIPT_SCHEMA = "fit40_initial_family_selector_receipt_v1"
INITIAL_INGEST_SCHEMA = "fit40_stagea_selector_ingest_v1"
INITIAL_SELECTION_STATUSES = {
    "ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV",
    "NOT_ADMITTED",
}
MODEL_ORDER = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
SNAPSHOT_METHOD_IDS = {
    "fit40_native_transition_head_v3_attempt11_step030",
    "fit40_native_transition_head_v3_attempt11_step060",
    "fit40_native_transition_head_v3_attempt11_step090",
    "fit40_native_transition_head_v3_attempt11_step120",
}
ARRAY_DTYPES = {
    "query_frame_index": "<i4",
    "query_timestamp_num": "<i8",
    "query_timestamp_den": "<i8",
    "track_frame_index": "<i4",
    "valid_query_mask": "|b1",
    "distance_by_query_lag": "<f4",
    "lag_offsets_frames": "<i4",
}
ARRAY_NAMES = frozenset(ARRAY_DTYPES)


class PointExecutionError(RuntimeError):
    """Invalid point execution lineage, evidence, or output state."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    return _sha(_canonical({key: item for key, item in value.items() if key != field}))


def _file_pin(path: Path, *, self_field: Optional[str] = None) -> Dict[str, Any]:
    if path.is_symlink():
        raise PointExecutionError("refusing to pin a symlink: " + str(path))
    try:
        info = path.lstat()
    except OSError as exc:
        raise PointExecutionError("required output/input file is absent: " + str(path)) from exc
    if not stat.S_ISREG(info.st_mode):
        raise PointExecutionError("required output/input is not a regular file: " + str(path))
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise PointExecutionError("file size changed while pinning: " + str(path))
    result = {"path": str(path.resolve(strict=True)), "bytes": len(raw), "sha256": _sha(raw)}
    if self_field is not None:
        packet = _decode_object(raw, str(path))
        expected = packet.get(self_field)
        if not isinstance(expected, str) or expected != _self_hash(packet, self_field):
            raise PointExecutionError("invalid canonical self hash: " + str(path))
        result["self_sha256"] = expected
    return result


def _decode_object(raw: bytes, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PointExecutionError(label + " is not UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PointExecutionError(label + " JSON root is not an object")
    return value


def _verify_pin_object(pin: Mapping[str, Any], label: str,
                       *, packet_self_field: Optional[str] = None) -> Tuple[Path, bytes, Dict[str, Any]]:
    expected_fields = {"path", "bytes", "sha256"}
    if packet_self_field is not None:
        expected_fields.add("self_sha256")
    if not isinstance(pin, Mapping) or set(pin) != expected_fields:
        raise PointExecutionError(label + " pin has unexpected or missing fields")
    name = pin.get("path")
    if not isinstance(name, str) or not Path(name).is_absolute():
        raise PointExecutionError(label + " pin path must be absolute")
    path = Path(name)
    if path.is_symlink():
        raise PointExecutionError(label + " pin resolves through a symlink")
    try:
        info = path.lstat()
    except OSError as exc:
        raise PointExecutionError(label + " pin file is absent") from exc
    if not stat.S_ISREG(info.st_mode):
        raise PointExecutionError(label + " pin is not a regular file")
    raw = path.read_bytes()
    if (len(raw) != info.st_size or len(raw) != pin.get("bytes")
            or _sha(raw) != pin.get("sha256")):
        raise PointExecutionError(label + " raw size/SHA differs from its pin")
    packet = _decode_object(raw, label) if packet_self_field is not None else {}
    if packet_self_field is not None:
        if (packet.get(packet_self_field) != _self_hash(packet, packet_self_field)
                or packet.get(packet_self_field) != pin.get("self_sha256")):
            raise PointExecutionError(label + " canonical self hash differs from its pin")
    actual = {"path": str(path.resolve(strict=True)), "bytes": len(raw), "sha256": _sha(raw)}
    if packet_self_field is not None:
        actual["self_sha256"] = packet[packet_self_field]
    return path.resolve(strict=True), raw, {"pin": actual, "packet": packet}


def _verify_link(link: Mapping[str, Any], label: str, packet_self_field: str) -> Dict[str, Any]:
    _path, _raw, loaded = _verify_pin_object(link, label, packet_self_field=packet_self_field)
    return loaded["pin"]


def _same_file_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("path", "bytes", "sha256"))


def _same_raw_self_pin(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (isinstance(left, Mapping) and isinstance(right, Mapping)
            and dict(left) == dict(right)
            and set(left) == {"path", "bytes", "sha256", "self_sha256"})


def _validate_initial_selector_receipt(
    receipt_pin: Mapping[str, Any], family_pin: Mapping[str, Any], family: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    _receipt_path, _receipt_raw, loaded = _verify_pin_object(
        receipt_pin, "initial 720-row selector receipt", packet_self_field="receipt_sha256")
    receipt = loaded["packet"]
    if (receipt.get("schema") != INITIAL_RECEIPT_SCHEMA
            or receipt.get("status") != "INITIAL_720_SELECTOR_COMPLETE"
            or receipt.get("metric_backend") != "torch_cpu"):
        raise PointExecutionError("initial selector receipt lacks the canonical completed 720-row status")
    selection = receipt.get("selection")
    if not isinstance(selection, Mapping):
        raise PointExecutionError("initial selector receipt omits its serialized actual selection")
    selection_status = selection.get("status")
    if (selection_status not in INITIAL_SELECTION_STATUSES
            or receipt.get("selection_status") != selection_status
            or selection.get("metric_backend") != "torch_cpu"):
        raise PointExecutionError("initial selection is not an exact torch_cpu terminal outcome")
    accounting = selection.get("accounting")
    if (not isinstance(accounting, Mapping)
            or accounting.get("unique_official_score_rows") != 720
            or accounting.get("observed_unique_rows") != 720
            or accounting.get("failed_score_rows") != 0
            or accounting.get("missing_score_rows") != 0
            or accounting.get("unexpected_score_rows") != 0):
        raise PointExecutionError("initial selector evidence does not account for the complete clean 720 rows")
    selection_errors = selection.get("errors", [])
    if selection_errors:
        raise PointExecutionError("initial selector contains errors and cannot seed SPSA")
    seed = receipt.get("spsa_seed_method_id")
    if (not isinstance(seed, str) or seed not in SNAPSHOT_METHOD_IDS
            or (selection_status == "ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV"
                and selection.get("selected_method_id") != seed)
            or (selection_status == "NOT_ADMITTED"
                and (selection.get("selected_method_id") is not None
                     or selection.get("refinement_seed_method_id") != seed))):
        raise PointExecutionError("initial SPSA seed does not match the actual admitted/fallback selection")
    if receipt.get("scientific_input_pins_sha256") != family.get("scientific_input_pins_sha256"):
        raise PointExecutionError("initial selector receipt input pins differ from the composed family")
    family_link = receipt.get("composed_family_plan")
    if not isinstance(family_link, Mapping) or dict(family_link) != family_pin:
        raise PointExecutionError("initial selector receipt does not bind this exact composed family plan")
    ingest_link = receipt.get("initial_ingest_index")
    if not isinstance(ingest_link, Mapping):
        raise PointExecutionError("initial selector receipt omits the exact Stage-A ingest index")
    _index_path, _index_raw, index_loaded = _verify_pin_object(
        ingest_link, "initial Stage-A ingest index", packet_self_field="index_sha256")
    ingest = index_loaded["packet"]
    if (ingest.get("schema") != INITIAL_INGEST_SCHEMA
            or receipt.get("initial_selector_index_sha256") != index_loaded["pin"]["sha256"]
            or receipt.get("initial_selector_index_self_sha256") != ingest.get("index_sha256")):
        raise PointExecutionError("initial ingest index link or producer schema differs from the receipt")
    models = ingest.get("models")
    if (not isinstance(models, list) or len(models) != 3
            or [entry.get("model_id") if isinstance(entry, Mapping) else None for entry in models] != list(MODEL_ORDER)):
        raise PointExecutionError("initial ingest index does not preserve the exact three-model order")
    seed_method_id = receipt.get("spsa_seed_method_id")
    return dict(loaded["pin"]), {"receipt": dict(loaded["pin"]), "seed_method_id": seed_method_id,
                                 "selection_status": selection_status}


def _validate_execution_package_manifest(
    package_pin: Mapping[str, Any], family_pin: Mapping[str, Any], model_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    _path, _raw, loaded = _verify_pin_object(
        package_pin, "point execution package manifest", packet_self_field="manifest_sha256")
    package = loaded["packet"]
    if (package.get("schema") != EXECUTION_PACKAGE_SCHEMA or package.get("status") != "FROZEN"
            or package.get("composed_family_plan_pin") != family_pin):
        raise PointExecutionError("execution package manifest is not frozen to this exact family plan")
    source_path = Path(__file__).resolve(strict=True)
    runner_actual = _file_pin(source_path)
    runner_ref = package.get("point_runner_pin")
    helper_ref = package.get("shared_application_helper_pin")
    if (not isinstance(runner_ref, Mapping) or not _same_file_identity(runner_ref, runner_actual)
            or not isinstance(helper_ref, Mapping)):
        raise PointExecutionError("execution package runner/helper pin is absent or differs from source")
    _verify_emitted_pin(helper_ref, "shared application helper")
    entries = package.get("execution_sources_by_model")
    if (not isinstance(entries, Mapping) or set(entries) != set(MODEL_ORDER)
            or model_id not in entries):
        raise PointExecutionError("execution package does not pin the exact three model source closures")
    model_entry = entries[model_id]
    required = {"launcher_pin", "runner_pin", "adapter_pin", "runtime_binding_pin", "source_pins"}
    if not isinstance(model_entry, Mapping) or set(model_entry) != required:
        raise PointExecutionError("model execution package entry has unexpected/missing pin roles")
    for role in ("launcher_pin", "runner_pin", "adapter_pin", "runtime_binding_pin"):
        pin = model_entry.get(role)
        if not isinstance(pin, Mapping):
            raise PointExecutionError("execution package omits model source role: " + role)
        _verify_emitted_pin(pin, "package " + role)
    source_pins = model_entry.get("source_pins")
    if not isinstance(source_pins, Mapping) or not source_pins:
        raise PointExecutionError("execution package model source closure is empty")
    for role, pin in source_pins.items():
        if not isinstance(role, str) or not role or not isinstance(pin, Mapping):
            raise PointExecutionError("execution package source closure entry is malformed")
        _verify_emitted_pin(pin, "package source " + role)
    return dict(loaded["pin"]), dict(package), dict(model_entry)


def _verify_plan_pin(pin: Mapping[str, Any], label: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    _path, _raw, loaded = _verify_pin_object(pin, label, packet_self_field="plan_sha256")
    plan = loaded["packet"]
    if (plan.get("schema_version") != FAMILY_SCHEMA
            or plan.get("status") != "FROZEN"):
        raise PointExecutionError(label + " is not a frozen self-hashed Stage-A plan")
    return dict(loaded["pin"]), plan


def _stable_parent_identity(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return tuple(row.get(key) for key in (
        "split", "sample_id", "speaker_id", "model_id", "face_id", "seed",
        "source_media_kind", "natural_pcm", "source_face", "source_video", "face_box",
    ))


def validate_point_scope(
    *, family_plan_pin: Mapping[str, Any], point_plan_pin: Mapping[str, Any],
    initial_selector_receipt_pin: Mapping[str, Any], execution_package_manifest_pin: Mapping[str, Any],
    model_id: str,
) -> Dict[str, Any]:
    """Verify all preconditions before model loading or output creation."""
    family_pin, family = _verify_plan_pin(family_plan_pin, "720-row composed family plan")
    point_pin, point = _verify_plan_pin(point_plan_pin, "derived one-method point plan")
    if model_id not in MODEL_ORDER:
        raise PointExecutionError("model_id is outside the exact frozen three-model order")
    package_pin, package, package_entry = _validate_execution_package_manifest(
        execution_package_manifest_pin, family_pin, model_id)
    execution = family.get("execution")
    if (not isinstance(execution, Mapping) or execution.get("planned_rows") != 720
            or execution.get("model_ids") != list(MODEL_ORDER)
            or execution.get("candidate_selection") != "not performed"
            or execution.get("dev_or_confirm_access") is not False):
        raise PointExecutionError("composed family does not carry the exact complete Fit-only 720 scope")
    family_rows = family.get("rows")
    family_models = family.get("models")
    family_methods = family.get("methods")
    if (not isinstance(family_rows, list) or len(family_rows) != 720
            or not isinstance(family_models, list)
            or [item.get("model_id") if isinstance(item, Mapping) else None for item in family_models]
            != list(MODEL_ORDER)
            or not isinstance(family_methods, list) or len(family_methods) != 6):
        raise PointExecutionError("composed family rows do not preserve all 720 planned cells")
    family_method_ids = [item.get("method_id") if isinstance(item, Mapping) else None
                         for item in family_methods]
    expected_method_ids = ["identity", "rms_fixed", *sorted(SNAPSHOT_METHOD_IDS)]
    if len(set(family_method_ids)) != 6 or set(family_method_ids) != set(expected_method_ids):
        raise PointExecutionError("composed family does not preserve the exact six source methods")
    family_samples = {str(row.get("sample_id")) for row in family_rows if isinstance(row, Mapping)}
    family_contexts = set()
    for row in family_rows:
        if (not isinstance(row, Mapping) or row.get("split") != "fit"
                or row.get("model_id") not in MODEL_ORDER
                or row.get("method_id") not in family_method_ids
                or row.get("score_audio") != row.get("natural_pcm")):
            raise PointExecutionError("family contains a non-Fit row or changed original scoring audio")
        key = (str(row.get("sample_id")), row.get("model_id"), row.get("method_id"))
        if key in family_contexts:
            raise PointExecutionError("family repeats a sample/model/method context")
        family_contexts.add(key)
    expected_family_contexts = {
        (sample, model, method_id)
        for sample in family_samples for model in MODEL_ORDER for method_id in family_method_ids
    }
    if len(family_samples) != 40 or family_contexts != expected_family_contexts:
        raise PointExecutionError("family does not contain exactly 40×3×6 Fit contexts")
    point_execution = point.get("execution")
    parent_ref = point_execution.get("spsa_execution_parent_plan") if isinstance(point_execution, Mapping) else None
    if (not isinstance(parent_ref, Mapping)
            or not _same_file_identity(parent_ref, family_pin)
            or point.get("scientific_input_pins_sha256") != family.get("scientific_input_pins_sha256")):
        raise PointExecutionError("point plan parent or scientific input identity differs from the frozen family")
    methods = point.get("methods")
    rows = point.get("rows")
    point_models = point.get("models")
    if (not isinstance(methods, list) or len(methods) != 1 or not isinstance(rows, list)
            or not isinstance(point_models, list)
            or [item.get("model_id") if isinstance(item, Mapping) else None for item in point_models]
            != list(MODEL_ORDER)
            or not isinstance(point_execution, Mapping)
            or point_execution.get("planned_rows") != len(rows)
            or point.get("generation_repeat_qc") != []):
        raise PointExecutionError("point plan must be one exact derived method with its full row denominator")
    method = methods[0]
    if not isinstance(method, Mapping):
        raise PointExecutionError("point plan method descriptor is not an object")
    config = method.get("config")
    config_sha = method.get("config_sha256")
    if (not isinstance(config, Mapping) or not isinstance(config_sha, str)
            or _sha(_canonical(config)) != config_sha):
        raise PointExecutionError("point method config hash does not match its frozen method")
    kind = config.get("method")
    if kind == "fit40_spsa_proposal":
        point_kind = "spsa_proposal"
        sample_count = 8
        point_id = config.get("proposal_id")
        scope = point_execution.get("spsa_execution_scope", {})
        required_scope_kind = "spsa_pair_proposal"
    elif kind == "fit40_spsa_center":
        point_kind = "spsa_center"
        sample_count = 40
        point_id = method.get("method_id")
        scope = point_execution.get("spsa_execution_scope", {})
        required_scope_kind = "spsa_full_center"
    else:
        raise PointExecutionError("point method is outside the frozen proposal/center transform schemas")
    if (not isinstance(scope, Mapping) or scope.get("scope_kind") != required_scope_kind
            or scope.get("method_id") != method.get("method_id")
            or scope.get("input_pins_sha256") != family.get("scientific_input_pins_sha256")
            or scope.get("run_id") != point.get("run_id")):
        raise PointExecutionError("point execution scope differs from the frozen proposal/center method")
    if type(scope.get("iteration")) is not int:
        raise PointExecutionError("point execution scope lacks an integer iteration/source step")
    if point_kind == "spsa_proposal":
        if (config.get("side") not in {"plus", "minus"}
                or config.get("proposal_id") != point_id
                or config.get("pair_id") != scope.get("pair_id")
                or config.get("proposal_id") != scope.get("proposal_id")
                or config.get("side") != scope.get("side")
                or config.get("iteration") != scope.get("iteration")):
            raise PointExecutionError("proposal row identity differs from the controller-bound pair scope")
    elif (config.get("point_kind") != point_kind
          or config.get("source_step") not in {5, 10, 15, 20}
          or config.get("source_step") != scope.get("source_step")
          or config.get("source_step") != scope.get("iteration")):
        raise PointExecutionError("center source step differs from its exact center scope")
    expected_total = sample_count * len(MODEL_ORDER)
    if len(rows) != expected_total:
        raise PointExecutionError("point plan does not preserve its exact 24/120 all-model denominator")
    family_ids = {str(row.get("row_id")) for row in family_rows if isinstance(row, Mapping)}
    if len(family_ids) != 720:
        raise PointExecutionError("family row IDs are not unique across all 720 planned cells")
    identity_by_context = {}
    for row in family_rows:
        if isinstance(row, Mapping) and row.get("method_id") == "identity":
            key = (row.get("sample_id"), row.get("model_id"))
            if key in identity_by_context:
                raise PointExecutionError("family repeats an identity parent for a model/sample context")
            identity_by_context[key] = row
    model_rows = [row for row in rows if isinstance(row, Mapping) and row.get("model_id") == model_id]
    all_point_samples = list(dict.fromkeys(str(row.get("sample_id")) for row in rows))
    if len(all_point_samples) != sample_count:
        raise PointExecutionError("point plan sample roster differs from the exact 8/40 denominator")
    expected_context_order = [(sample, backend) for sample in all_point_samples
                              for backend in MODEL_ORDER]
    observed_context_order = [(row.get("sample_id"), row.get("model_id"))
                              for row in rows if isinstance(row, Mapping)]
    if (len(model_rows) != sample_count
            or observed_context_order != expected_context_order):
        raise PointExecutionError("point model slice differs from exact ordered 8/40 row denominator")
    seen_samples = set()
    for row in model_rows:
        sample_id = row.get("sample_id")
        if (row.get("split") != "fit" or row.get("method_id") != method.get("method_id")
                or row.get("method_config_sha256") != config_sha
                or row.get("spsa_point_binding_sha256") != config.get("binding_sha256")
                or row.get("seed") != 20261003):
            raise PointExecutionError("point model row differs from the exact method/seed/split contract")
        if sample_id in seen_samples:
            raise PointExecutionError("point model subset repeats a sample ID")
        seen_samples.add(sample_id)
        parent = identity_by_context.get((sample_id, model_id))
        if not isinstance(parent, Mapping) or _stable_parent_identity(row) != _stable_parent_identity(parent):
            raise PointExecutionError("point row changed frozen identity/source/clock metadata")
        if row.get("score_audio") != row.get("natural_pcm"):
            raise PointExecutionError("point row must keep scoring audio equal to original natural PCM")
    if point_kind == "spsa_proposal":
        speaker_counts = {}
        for sample in all_point_samples:
            identity = identity_by_context.get((sample, model_id))
            if not isinstance(identity, Mapping):
                raise PointExecutionError("proposal sample is absent from the initial family")
            speaker_counts[str(identity.get("speaker_id"))] = speaker_counts.get(str(identity.get("speaker_id")), 0) + 1
        if len(speaker_counts) != 4 or set(speaker_counts.values()) != {2}:
            raise PointExecutionError("proposal scope must be the frozen two-clips-from-four-speakers batch")
    elif set(all_point_samples) != {str(row.get("sample_id")) for row in family_rows
                                    if isinstance(row, Mapping) and row.get("split") == "fit"}:
        raise PointExecutionError("center scope must preserve the complete 40-sample Fit roster")
    ingest_pin, gate_summary = _validate_initial_selector_receipt(
        initial_selector_receipt_pin, family_pin, family)
    runtime_ref = execution.get("runtime_bindings_by_model", {}).get(model_id)
    runtime_pin = package_entry.get("runtime_binding_pin")
    if not isinstance(runtime_ref, Mapping) or not isinstance(runtime_pin, Mapping):
        raise PointExecutionError("family/package lacks the exact model runtime binding")
    runtime_sha = runtime_ref.get("sha256", runtime_ref.get("raw_sha256"))
    if (runtime_ref.get("path") != runtime_pin.get("path")
            or runtime_ref.get("bytes") != runtime_pin.get("bytes")
            or runtime_sha != runtime_pin.get("sha256")):
        raise PointExecutionError("execution package runtime pin differs from the frozen family runtime binding")
    return {
        "family_plan": family,
        "family_plan_pin": family_pin,
        "point_plan": point,
        "point_plan_pin": point_pin,
        "method": method,
        "model_id": model_id,
        "point_kind": point_kind,
        "point_id": point_id,
        "planned_model_rows": model_rows,
        "all_point_rows": rows,
        "sample_ids": all_point_samples,
        "iteration": scope.get("iteration"),
        "proposal_id": config.get("proposal_id"),
        "pair_id": config.get("pair_id"),
        "side": config.get("side"),
        "source_step": config.get("source_step"),
        "initial_ingest_index_pin": ingest_pin,
        "initial_selector": gate_summary,
        "execution_package_manifest_pin": package_pin,
        "shared_application_helper_pin": package["shared_application_helper_pin"],
        "execution_package_manifest": package_entry,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_utc(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise PointExecutionError(label + " must be an RFC3339 UTC timestamp")
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise PointExecutionError(label + " is not an RFC3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise PointExecutionError(label + " must carry an explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def _proc_start_ticks(pid: int) -> int:
    """Read Linux proc field 22 for this still-live child, without env access."""
    raw = Path("/proc").joinpath(str(pid), "stat").read_text(encoding="ascii")
    close = raw.rfind(")")
    fields = raw[close + 2:].split()
    if close < 0 or len(fields) <= 19:
        raise PointExecutionError("cannot read actual child start ticks from /proc")
    ticks = int(fields[19])  # field 22; text after comm starts at field 3
    if ticks <= 0:
        raise PointExecutionError("child process start ticks are not positive")
    return ticks


def _write_exclusive(path: Path, raw: bytes) -> Dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise PointExecutionError("refusing to overwrite point evidence: " + str(path))
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return _file_pin(path)


def _write_self_json(path: Path, value: Mapping[str, Any], field: str) -> Dict[str, Any]:
    packet = dict(value)
    packet[field] = _self_hash(packet, field)
    raw = (json.dumps(packet, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + "\n").encode("utf-8")
    pin = _write_exclusive(path, raw)
    pin["self_sha256"] = packet[field]
    return pin


def _jsonl_pin(path: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    raw = b"".join(_canonical(row) + b"\n" for row in rows)
    return _write_exclusive(path, raw)


def _validate_arrays(arrays: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    try:
        import numpy as np
    except Exception as exc:
        raise PointExecutionError("NumPy is required to seal official score arrays") from exc
    if not isinstance(arrays, Mapping) or set(arrays) != ARRAY_NAMES:
        raise PointExecutionError("official score arrays differ from the exact seven-array contract")
    normalized = {}
    matrix_shape = None
    for name in sorted(ARRAY_NAMES):
        value = np.asarray(arrays[name])
        if value.dtype.str != ARRAY_DTYPES[name] or value.ndim != (2 if name == "distance_by_query_lag" else 1):
            raise PointExecutionError("official score array dtype/rank differs: " + name)
        if name == "distance_by_query_lag":
            if value.shape[1:] != (31,) or not np.isfinite(value).all():
                raise PointExecutionError("official raw SyncNet matrix must be finite [N,31]")
            matrix_shape = value.shape
        normalized[name] = np.ascontiguousarray(value)
    count = matrix_shape[0] if matrix_shape is not None else -1
    if count <= 0 or any(value.shape[0] != count for key, value in normalized.items() if key != "lag_offsets_frames"):
        raise PointExecutionError("official NPZ arrays do not share one nonempty query count")
    if normalized["lag_offsets_frames"].tolist() != list(range(-15, 16)):
        raise PointExecutionError("official lag axis is not exactly -15..15")
    if (np.any(normalized["query_timestamp_den"] <= 0)
            or not normalized["valid_query_mask"].all()):
        raise PointExecutionError("official source-clock or decoded-support arrays are invalid")
    meta = {}
    for name, value in normalized.items():
        meta[name] = {"dtype": value.dtype.str, "shape": list(value.shape),
                      "array_sha256": _sha(value.tobytes(order="C"))}
    return meta


def _validate_npz_result(record: Mapping[str, Any], arrays: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        import numpy as np
    except Exception as exc:
        raise PointExecutionError("NumPy is required to read official score NPZ") from exc
    meta = _validate_arrays(arrays)
    score = record.get("score")
    raw_ref = score.get("raw_matrix_npz") if isinstance(score, Mapping) else None
    if (not isinstance(raw_ref, Mapping) or set(raw_ref) != {"path", "file_sha256", "arrays"}
            or not isinstance(raw_ref.get("path"), str) or not Path(raw_ref["path"]).is_absolute()
            or raw_ref.get("arrays") != meta):
        raise PointExecutionError("passing record lacks the exact producer NPZ metadata contract")
    path = Path(raw_ref["path"])
    if path.is_symlink() or not path.is_file():
        raise PointExecutionError("official raw score NPZ is absent or symlinked")
    raw = path.read_bytes()
    if _sha(raw) != raw_ref.get("file_sha256"):
        raise PointExecutionError("official raw score NPZ bytes differ from the Stage-A record")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != ARRAY_NAMES:
                raise PointExecutionError("NPZ members differ from the seven-array official contract")
            for key in ARRAY_NAMES:
                actual = np.asarray(archive[key])
                expected = np.asarray(arrays[key])
                if actual.dtype.str != ARRAY_DTYPES[key] or not np.array_equal(actual, expected):
                    raise PointExecutionError("actual NPZ array differs from adapter-returned array: " + key)
    except PointExecutionError:
        raise
    except Exception as exc:
        raise PointExecutionError("cannot read actual official NPZ: " + type(exc).__name__ + ": " + str(exc)) from exc
    info = path.stat()
    return {"path": str(path.resolve(strict=True)), "bytes": info.st_size,
            "sha256": raw_ref["file_sha256"], "array_refs": meta}


def _failed_row(plan_row: Mapping[str, Any], reason: str, status: str) -> Dict[str, Any]:
    natural = plan_row.get("natural_pcm")
    face = plan_row.get("source_face")
    video = plan_row.get("source_video")
    return {
        **{key: plan_row.get(key) for key in (
            "row_id", "split", "sample_id", "speaker_id", "model_id", "face_id",
            "seed", "method_id", "method_config_sha256")},
        "schema_version": "cross_tfg_audio_head_stagea_v1",
        "status": status,
        "failure_reason": str(reason)[:2000],
        "natural_pcm_sha256": natural.get("pcm_sha256") if isinstance(natural, Mapping) else None,
        "score_audio_sha256": natural.get("pcm_sha256") if isinstance(natural, Mapping) else None,
        "source_media_kind": plan_row.get("source_media_kind", "video"),
        "source_video_sha256": video.get("file_sha256") if isinstance(video, Mapping) else None,
        "source_face_sha256": face.get("file_sha256") if isinstance(face, Mapping) else None,
    }


def _success_item(item: Mapping[str, Any], planned_row: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not isinstance(item, Mapping) or not isinstance(item.get("row"), Mapping):
        raise PointExecutionError("adapter success item omits a public Stage-A row")
    record = dict(item["row"])
    if (record.get("status") != "PASS" or record.get("row_id") != planned_row.get("row_id")
            or record.get("model_id") != planned_row.get("model_id")
            or record.get("method_id") != planned_row.get("method_id")):
        raise PointExecutionError("adapter success row differs from its exact planned identity")
    arrays = item.get("official_arrays")
    npz_pin = _validate_npz_result(record, arrays)
    public = {key: value for key, value in record.items() if not key.startswith("_internal_")}
    return public, {"row_id": str(record["row_id"]), "status": "PASS", "raw_matrix_npz": npz_pin}


def _source_pin_table(adapter: Any, context: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    getter = getattr(adapter, "validation_source_pins", None)
    values = getter() if callable(getter) else None
    if not isinstance(values, Mapping) or not values:
        raise PointExecutionError("adapter did not provide its exact validation source pins")
    output = {}
    for role, pin in values.items():
        if not isinstance(role, str) or not role or not isinstance(pin, Mapping):
            raise PointExecutionError("adapter source pin table has malformed entries")
        checked = _file_pin(Path(str(pin.get("path", ""))))
        if (checked["bytes"] != pin.get("bytes") or checked["sha256"] != pin.get("sha256")):
            raise PointExecutionError("adapter source file changed after its runtime validation: " + role)
        output[role] = checked
    declared = [context["family_plan_pin"], context["point_plan_pin"],
                context["shared_application_helper_pin"],
                context["execution_package_manifest"]["launcher_pin"],
                context["execution_package_manifest"]["runner_pin"],
                context["execution_package_manifest"]["adapter_pin"],
                context["execution_package_manifest"]["runtime_binding_pin"],
                *context["execution_package_manifest"]["source_pins"].values()]
    declared.append({key: context["execution_package_manifest_pin"][key]
                     for key in ("path", "bytes", "sha256")})
    declared.append(_file_pin(Path(__file__).resolve(strict=True)))
    allowed = {(str(pin.get("path")), pin.get("bytes"), pin.get("sha256"))
               for pin in declared if isinstance(pin, Mapping)}
    for role, pin in output.items():
        if (pin["path"], pin["bytes"], pin["sha256"]) not in allowed:
            raise PointExecutionError("adapter validation source is absent from the frozen execution package: " + role)
    return output


def execute_point_block(
    *, adapter_factory: Any, family_plan_pin: Mapping[str, Any], point_plan_pin: Mapping[str, Any],
    initial_selector_receipt_pin: Mapping[str, Any], execution_package_manifest_pin: Mapping[str, Any],
    model_id: str,
    output_root: Union[str, Path], start_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """Render/score one exact per-model point; preserve every planned row.

    ``output_root`` must not exist.  Adapter/render failures are converted to
    FAILED/INCOMPLETE ledger rows, after which terminal manifest/status are
    still sealed and the returned ``exit_code`` is nonzero.  Invalid source
    gates fail before creating the output directory or calling the adapter.
    """
    context = validate_point_scope(
        family_plan_pin=family_plan_pin, point_plan_pin=point_plan_pin,
        initial_selector_receipt_pin=initial_selector_receipt_pin,
        execution_package_manifest_pin=execution_package_manifest_pin, model_id=model_id)
    if not callable(adapter_factory):
        raise PointExecutionError("adapter_factory must be callable and is invoked only after all 720 gates")
    source_pins = context["execution_package_manifest"]["source_pins"]
    method = context["method"]
    planned = context["planned_model_rows"]
    run_id = str(context["point_plan"].get("run_id"))
    output = Path(output_root)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise PointExecutionError("point output root must be a fresh absolute path")
    output.mkdir(parents=True, exist_ok=False)
    planned_pin = _jsonl_pin(output / "planned_rows.jsonl", planned)
    row_output = output / "row_outputs"
    row_output.mkdir()
    actual_pid = os.getpid()
    actual_ticks = _proc_start_ticks(actual_pid)
    started = start_utc or _utc_now()
    _parse_utc(started, "point child started_utc")
    records: List[Dict[str, Any]] = []
    row_manifest: List[Dict[str, Any]] = []
    adapter_validation_pins: Dict[str, Dict[str, Any]] = {}
    failure_reason = None
    try:
        # Instantiate the model adapter only after all complete-family gates,
        # but inside the denominator-preserving failure boundary.
        adapter = adapter_factory(context)
        adapter_validation_pins = _source_pin_table(adapter, context)
        adapter_pin = context["execution_package_manifest"]["adapter_pin"]
        if not any(_same_file_identity(pin, adapter_pin)
                   for pin in adapter_validation_pins.values()):
            raise PointExecutionError("adapter did not expose its exact frozen adapter source pin")
        rendered = adapter.render_subset(candidate_rows=planned, method=method, output_root=row_output)
        if not isinstance(rendered, Sequence) or isinstance(rendered, (str, bytes)) or len(rendered) != len(planned):
            raise PointExecutionError("adapter returned a row count different from the planned model denominator")
        for item, plan_row in zip(rendered, planned):
            record, manifest_item = _success_item(item, plan_row)
            records.append(record)
            row_manifest.append(manifest_item)
    except Exception as exc:
        failure_reason = type(exc).__name__ + ": " + str(exc)
        completed = getattr(exc, "completed", [])
        if not isinstance(completed, Sequence) or isinstance(completed, (str, bytes)):
            completed = []
        if len(completed) > len(planned):
            completed = []
            failure_reason = "MalformedAdapterFailure: completed prefix exceeds planned denominator"
        for idx, (item, plan_row) in enumerate(zip(completed, planned)):
            try:
                record, manifest_item = _success_item(item, plan_row)
            except Exception as nested:
                completed = []
                records = []
                row_manifest = []
                failure_reason += "; malformed completed prefix: " + type(nested).__name__ + ": " + str(nested)
                break
            records.append(record)
            row_manifest.append(manifest_item)
        offset = len(records)
        failed_row_id = getattr(exc, "failed_row_id", None)
        failed_index = next((i for i, row in enumerate(planned[offset:], offset)
                             if row.get("row_id") == failed_row_id), None)
        if failed_index is None:
            failed_index = offset if offset < len(planned) else None
        if failed_index is not None:
            for index in range(offset, len(planned)):
                plan_row = planned[index]
                if index == failed_index:
                    failed_result = getattr(exc, "failed_result", None)
                    detail = failure_reason
                    if isinstance(failed_result, Mapping) and failed_result.get("failure_reason"):
                        detail += "; row: " + str(failed_result.get("failure_reason"))
                    records.append(_failed_row(plan_row, detail, "FAILED"))
                    row_manifest.append({"row_id": plan_row["row_id"], "status": "FAILED", "raw_matrix_npz": None})
                else:
                    records.append(_failed_row(plan_row, "not attempted after prior point-row failure", "INCOMPLETE"))
                    row_manifest.append({"row_id": plan_row["row_id"], "status": "INCOMPLETE", "raw_matrix_npz": None})
        else:
            # The adapter returned all rows, but a downstream integrity check
            # failed. Retain a complete failure denominator without replay.
            records = [_failed_row(row, failure_reason, "FAILED") for row in planned]
            row_manifest = [{"row_id": row["row_id"], "status": "FAILED", "raw_matrix_npz": None}
                            for row in planned]
    if len(records) != len(planned) or len(row_manifest) != len(planned):
        raise PointExecutionError("internal row accounting failed to preserve the complete point denominator")
    if [item.get("row_id") for item in records] != [row.get("row_id") for row in planned]:
        raise PointExecutionError("terminal ledger row order differs from the frozen planned-row order")
    results_pin = _jsonl_pin(output / "results.jsonl", records)
    counts = {status: sum(row.get("status") == status for row in records)
              for status in ("PASS", "FAILED", "INCOMPLETE")}
    terminal_status = "POINT_BLOCK_COMPLETE" if counts == {"PASS": len(planned), "FAILED": 0, "INCOMPLETE": 0} else "POINT_BLOCK_TERMINAL_WITH_FAILURES"
    source_manifest_rows = []
    for row, detail in zip(records, row_manifest):
        source_manifest_rows.append({
            "row_id": row.get("row_id"), "status": row.get("status"),
            "raw_matrix_npz": detail.get("raw_matrix_npz"),
        })
    manifest_value = {
        "schema": POINT_MANIFEST_SCHEMA,
        "status": terminal_status,
        "run_id": run_id,
        "point_kind": context["point_kind"], "point_id": context["point_id"],
        "method_id": method.get("method_id"),
        "proposal_id": context.get("proposal_id"), "pair_id": context.get("pair_id"),
        "iteration": context.get("iteration"), "side": context.get("side"),
        "source_step": context.get("source_step"), "model_id": model_id,
        "block_index": 0,
        "family_plan_pin": context["family_plan_pin"],
        "point_plan_pin": context["point_plan_pin"],
        "initial_selector_receipt_pin": context["initial_selector"]["receipt"],
        "execution_package_manifest_pin": context["execution_package_manifest_pin"],
        "initial_ingest_index_pin": context["initial_ingest_index_pin"],
        "planned_rows_pin": planned_pin, "ordered_row_ids": [row["row_id"] for row in planned],
        "planned_rows": len(planned), "result_rows": len(records), "status_counts": counts,
        "results_jsonl_pin": results_pin, "row_results": source_manifest_rows,
        "source_pins": source_pins, "adapter_validation_pins": adapter_validation_pins,
        "started_utc": started,
        "terminalized_utc": _utc_now(), "worker_error": failure_reason,
    }
    manifest_pin = _write_self_json(output / "result_manifest.json", manifest_value, "manifest_sha256")
    status_value = {
        "schema": POINT_STATUS_SCHEMA,
        "status": terminal_status,
        "run_id": run_id,
        "point_kind": context["point_kind"], "point_id": context["point_id"],
        "method_id": method.get("method_id"),
        "proposal_id": context.get("proposal_id"), "pair_id": context.get("pair_id"),
        "iteration": context.get("iteration"), "side": context.get("side"),
        "source_step": context.get("source_step"), "model_id": model_id,
        "block_index": 0,
        "family_plan_pin": context["family_plan_pin"],
        "point_plan_pin": context["point_plan_pin"],
        "initial_selector_receipt_pin": context["initial_selector"]["receipt"],
        "execution_package_manifest_pin": context["execution_package_manifest_pin"],
        "initial_ingest_index_pin": context["initial_ingest_index_pin"],
        "planned_rows_pin": planned_pin, "ordered_row_ids": [row["row_id"] for row in planned],
        "planned_rows": len(planned), "completed_rows": len(records),
        "status_counts": counts, "results_jsonl_pin": results_pin,
        "result_manifest_pin": manifest_pin,
        "child_pid": actual_pid, "child_start_ticks": actual_ticks,
        "started_utc": started, "ended_utc": _utc_now(),
        "source_pins": source_pins, "adapter_validation_pins": adapter_validation_pins,
        "worker_error": failure_reason,
    }
    status_pin = _write_self_json(output / "run_status.json", status_value, "status_sha256")
    return {
        "status": terminal_status,
        "exit_code": 0 if terminal_status == "POINT_BLOCK_COMPLETE" else 2,
        "run_status": status_value, "run_status_pin": status_pin,
        "result_manifest": manifest_value, "result_manifest_pin": manifest_pin,
        "results_jsonl_pin": results_pin, "planned_rows_pin": planned_pin,
        "output_root": str(output.resolve(strict=True)),
    }


def _verify_emitted_pin(pin: Mapping[str, Any], label: str,
                        *, packet_self_field: Optional[str] = None) -> Dict[str, Any]:
    _path, _raw, loaded = _verify_pin_object(pin, label, packet_self_field=packet_self_field)
    return {**loaded["pin"], "packet": loaded["packet"]}


def write_child_execution_receipt(
    *, receipt_path: Union[str, Path], context: Mapping[str, Any],
    process: Mapping[str, Any], launcher_pin: Mapping[str, Any], adapter_pin: Mapping[str, Any],
    runner_pin: Mapping[str, Any], runtime_binding_pin: Mapping[str, Any],
    source_pins: Mapping[str, Mapping[str, Any]],
    execution_package_manifest_pin: Mapping[str, Any],
    run_status_pin: Mapping[str, Any], result_manifest_pin: Mapping[str, Any],
    results_jsonl_pin: Mapping[str, Any],
) -> Dict[str, Any]:
    """Seal parent-captured Popen evidence after child exit; never spawns/replays."""
    required_process = {"child_pid", "child_start_ticks", "child_exit_code", "started_utc", "ended_utc"}
    if not isinstance(process, Mapping) or not required_process.issubset(process):
        raise PointExecutionError("parent process evidence lacks actual pid/ticks/numeric rc/times")
    if (type(process.get("child_pid")) is not int or process["child_pid"] <= 0
            or type(process.get("child_start_ticks")) is not int or process["child_start_ticks"] <= 0
            or type(process.get("child_exit_code")) is not int):
        raise PointExecutionError("parent process evidence has invalid pid/start ticks/exit code types")
    process_started = _parse_utc(process.get("started_utc"), "parent process started_utc")
    process_ended = _parse_utc(process.get("ended_utc"), "parent process ended_utc")
    if process_ended < process_started:
        raise PointExecutionError("parent process end time precedes its start time")
    status = _verify_emitted_pin(run_status_pin, "point run status", packet_self_field="status_sha256")
    manifest = _verify_emitted_pin(result_manifest_pin, "point result manifest", packet_self_field="manifest_sha256")
    ledger = _verify_emitted_pin(results_jsonl_pin, "point results ledger")
    package = _verify_emitted_pin(
        execution_package_manifest_pin, "point execution package manifest",
        packet_self_field="manifest_sha256")
    package_packet = package["packet"]
    status_packet = status["packet"]
    manifest_packet = manifest["packet"]
    manifest_actual_pin = {key: manifest[key] for key in ("path", "bytes", "sha256", "self_sha256")}
    ledger_actual_pin = {key: ledger[key] for key in ("path", "bytes", "sha256")}
    model_entry = package_packet.get("execution_sources_by_model", {}).get(status_packet.get("model_id"))
    if not isinstance(model_entry, Mapping):
        raise PointExecutionError("execution package has no source closure for the child model")
    expected_roles = {
        "launcher_pin": launcher_pin,
        "runner_pin": runner_pin,
        "adapter_pin": adapter_pin,
        "runtime_binding_pin": runtime_binding_pin,
    }
    for role, observed in expected_roles.items():
        if model_entry.get(role) != dict(observed):
            raise PointExecutionError("parent receipt role differs from the frozen package manifest: " + role)
    if model_entry.get("source_pins") != dict(source_pins):
        raise PointExecutionError("parent source-pin closure differs from the frozen package manifest")
    point_runner_pin = package_packet.get("point_runner_pin")
    helper_pin = package_packet.get("shared_application_helper_pin")
    if (not isinstance(point_runner_pin, Mapping) or not isinstance(helper_pin, Mapping)
            or model_entry.get("runner_pin") != dict(runner_pin)):
        raise PointExecutionError("parent runner/helper identity differs from the frozen package manifest")
    package_actual_pin = {key: package[key] for key in ("path", "bytes", "sha256", "self_sha256")}
    shared_fields = ("status", "run_id", "point_kind", "point_id", "method_id",
                     "proposal_id", "pair_id", "iteration", "side", "source_step",
                     "model_id", "block_index", "family_plan_pin", "point_plan_pin",
                     "initial_selector_receipt_pin", "initial_ingest_index_pin",
                     "execution_package_manifest_pin", "planned_rows_pin", "ordered_row_ids",
                     "planned_rows", "results_jsonl_pin", "source_pins", "adapter_validation_pins")
    if any(status_packet.get(key) != manifest_packet.get(key) for key in shared_fields):
        raise PointExecutionError("status and manifest differ on the shared point identity or evidence pins")
    if (status_packet.get("schema") != POINT_STATUS_SCHEMA
            or manifest_packet.get("schema") != POINT_MANIFEST_SCHEMA
            or not _same_raw_self_pin(status_packet.get("execution_package_manifest_pin", {}), package_actual_pin)
            or not _same_raw_self_pin(manifest_packet.get("execution_package_manifest_pin", {}), package_actual_pin)
            or status_packet.get("source_pins") != model_entry.get("source_pins")
            or manifest_packet.get("source_pins") != model_entry.get("source_pins")):
        raise PointExecutionError("status/manifest package pin or schema differs from the verified execution package")
    if (dict(execution_package_manifest_pin) != package_actual_pin
            or status_packet.get("child_pid") != process["child_pid"]
            or status_packet.get("child_start_ticks") != process["child_start_ticks"]
            or status_packet.get("status") != manifest_packet.get("status")
            or not _same_raw_self_pin(status_packet.get("result_manifest_pin", {}), manifest_actual_pin)
            or not _same_file_identity(status_packet.get("results_jsonl_pin", {}), results_jsonl_pin)
            or dict(manifest_packet.get("results_jsonl_pin", {})) != ledger_actual_pin
            or status_packet.get("block_index") != 0
            or manifest_packet.get("block_index") != 0):
        raise PointExecutionError("child outputs do not agree on exact terminal identity and raw/self pins")
    child_started = _parse_utc(status_packet.get("started_utc"), "child status started_utc")
    child_ended = _parse_utc(status_packet.get("ended_utc"), "child status ended_utc")
    manifest_started = _parse_utc(manifest_packet.get("started_utc"), "result manifest started_utc")
    manifest_ended = _parse_utc(manifest_packet.get("terminalized_utc"), "result manifest terminalized_utc")
    if (manifest_started != child_started or process_started > child_started
            or child_ended < manifest_ended or process_ended < child_ended):
        raise PointExecutionError("parent/child/manifest UTC timestamps do not form a valid execution interval")
    try:
        _planned_path, planned_raw, _planned = _verify_pin_object(
            status_packet.get("planned_rows_pin", {}), "planned point rows")
        planned_rows = [_decode_object(line, "planned point row")
                        for line in planned_raw.splitlines() if line]
        ledger_rows = [_decode_object(line, "point result row")
                       for line in Path(ledger["path"]).read_bytes().splitlines() if line]
    except Exception as exc:
        if isinstance(exc, PointExecutionError):
            raise
        raise PointExecutionError("cannot read point planned/result JSONL: " + str(exc)) from exc
    if (len(planned_rows) != status_packet.get("planned_rows")
            or [row.get("row_id") for row in planned_rows] != status_packet.get("ordered_row_ids")
            or len(ledger_rows) != len(planned_rows)
            or [row.get("row_id") for row in ledger_rows] != status_packet.get("ordered_row_ids")):
        raise PointExecutionError("planned/result ledgers differ from the complete ordered denominator")
    allowed_sources = {
        (str(pin.get("path")), pin.get("bytes"), pin.get("sha256"))
        for pin in (list(model_entry.get("source_pins", {}).values())
                    + [model_entry.get("launcher_pin"), model_entry.get("runner_pin"),
                       model_entry.get("adapter_pin"), model_entry.get("runtime_binding_pin"),
                       package_packet.get("point_runner_pin"),
                       package_packet.get("shared_application_helper_pin")])
        if isinstance(pin, Mapping)
    }
    adapter_validation_pins = status_packet.get("adapter_validation_pins")
    if not isinstance(adapter_validation_pins, Mapping):
        raise PointExecutionError("status omits its adapter validation source identities")
    for role, pin in adapter_validation_pins.items():
        if (not isinstance(role, str) or not isinstance(pin, Mapping)
                or (str(pin.get("path")), pin.get("bytes"), pin.get("sha256")) not in allowed_sources):
            raise PointExecutionError("adapter validation source is outside the frozen execution package")
        _verify_emitted_pin(pin, "adapter validation source " + role)
    actual_counts = {state: sum(row.get("status") == state for row in ledger_rows)
                     for state in ("PASS", "FAILED", "INCOMPLETE")}
    manifest_row_results = manifest_packet.get("row_results")
    if (status_packet.get("status_counts") != actual_counts
            or manifest_packet.get("status_counts") != actual_counts
            or status_packet.get("completed_rows") != len(ledger_rows)
            or manifest_packet.get("result_rows") != len(ledger_rows)
            or manifest_packet.get("planned_rows") != len(planned_rows)
            or not isinstance(manifest_row_results, list)
            or len(manifest_row_results) != len(ledger_rows)):
        raise PointExecutionError("status/manifest counts do not match the actual planned/result records")
    if ((actual_counts == {"PASS": len(planned_rows), "FAILED": 0, "INCOMPLETE": 0}
         and process["child_exit_code"] != 0)
            or (actual_counts != {"PASS": len(planned_rows), "FAILED": 0, "INCOMPLETE": 0}
                and process["child_exit_code"] == 0)):
        raise PointExecutionError("parent-captured exit code contradicts the terminal row outcomes")
    for row, row_ref in zip(ledger_rows, manifest_row_results):
        if row_ref.get("row_id") != row.get("row_id") or row_ref.get("status") != row.get("status"):
            raise PointExecutionError("manifest row result differs from its ordered result-ledger row")
        score = row.get("score")
        raw_ref = score.get("raw_matrix_npz") if isinstance(score, Mapping) else None
        npz_pin = row_ref.get("raw_matrix_npz")
        if row.get("status") == "PASS":
            if (not isinstance(raw_ref, Mapping) or not isinstance(npz_pin, Mapping)
                    or raw_ref.get("path") != npz_pin.get("path")
                    or raw_ref.get("file_sha256") != npz_pin.get("sha256")
                    or raw_ref.get("arrays") != npz_pin.get("array_refs")):
                raise PointExecutionError("manifest NPZ pin/array references differ from its actual result row")
            npz_actual = _verify_emitted_pin(
                {key: npz_pin.get(key) for key in ("path", "bytes", "sha256")},
                "official score NPZ " + str(row.get("row_id")))
            if npz_actual.get("bytes") != npz_pin.get("bytes"):
                raise PointExecutionError("official score NPZ byte count changed after child completion")
            try:
                import numpy as np
                with np.load(npz_pin["path"], allow_pickle=False) as archive:
                    if set(archive.files) != ARRAY_NAMES:
                        raise PointExecutionError("official score NPZ members changed after child completion")
                    arrays = {name: np.asarray(archive[name]) for name in ARRAY_NAMES}
                if _validate_arrays(arrays) != npz_pin.get("array_refs"):
                    raise PointExecutionError("official score NPZ array bytes changed after child completion")
            except PointExecutionError:
                raise
            except Exception as exc:
                raise PointExecutionError("cannot reread official score NPZ: " + str(exc)) from exc
        elif npz_pin is not None:
            raise PointExecutionError("failed or incomplete rows must not claim an official NPZ")
    source_table = {}
    for role, pin in source_pins.items():
        if not isinstance(role, str) or not isinstance(pin, Mapping):
            raise PointExecutionError("child source-pin map is malformed")
        source_table[role] = _verify_emitted_pin(pin, "source " + role)
        source_table[role].pop("packet", None)
    for label, pin in (("launcher", launcher_pin), ("runner", runner_pin),
                       ("adapter", adapter_pin),
                       ("runtime binding", runtime_binding_pin)):
        _verify_emitted_pin(pin, label + " source")
    receipt = {
        "schema": CHILD_RECEIPT_SCHEMA, "status": "CHILD_EXITED",
        "run_id": status_packet.get("run_id"),
        "point_kind": status_packet.get("point_kind"), "point_id": status_packet.get("point_id"),
        "method_id": status_packet.get("method_id"),
        "proposal_id": status_packet.get("proposal_id"), "pair_id": status_packet.get("pair_id"),
        "iteration": status_packet.get("iteration"), "side": status_packet.get("side"),
        "source_step": status_packet.get("source_step"), "model_id": status_packet.get("model_id"),
        "block_index": 0,
        "family_plan_pin": status_packet.get("family_plan_pin"),
        "point_plan_pin": status_packet.get("point_plan_pin"),
        "execution_package_manifest_pin": {key: package[key] for key in ("path", "bytes", "sha256", "self_sha256")},
        "planned_rows_pin": status_packet.get("planned_rows_pin"),
        "ordered_row_ids": status_packet.get("ordered_row_ids"),
        "child_pid": process["child_pid"], "child_start_ticks": process["child_start_ticks"],
        "child_exit_code": process["child_exit_code"],
        "started_utc": process["started_utc"], "ended_utc": process["ended_utc"],
        "launcher_pin": dict(launcher_pin), "runner_pin": dict(runner_pin),
        "adapter_pin": dict(adapter_pin),
        "runtime_binding_pin": dict(runtime_binding_pin), "source_pins": source_table,
        "point_runner_pin": dict(point_runner_pin),
        "shared_application_helper_pin": dict(helper_pin),
        "run_status_pin": {key: status[key] for key in ("path", "bytes", "sha256", "self_sha256")},
        "result_manifest_pin": {key: manifest[key] for key in ("path", "bytes", "sha256", "self_sha256")},
        "results_jsonl_pin": {key: ledger[key] for key in ("path", "bytes", "sha256")},
    }
    pin = Path(receipt_path)
    if not pin.is_absolute():
        raise PointExecutionError("child execution receipt path must be absolute")
    return _write_self_json(pin, receipt, "receipt_sha256")


__all__ = [
    "PointExecutionError", "validate_point_scope", "execute_point_block",
    "write_child_execution_receipt", "POINT_STATUS_SCHEMA", "POINT_MANIFEST_SCHEMA",
    "CHILD_RECEIPT_SCHEMA",
]
