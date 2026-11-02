"""Fail-closed final SPSA-center Stage-A family ingestion.

This reader is deliberately separate from ``fit_only_stagea_ingest_attempt03``:
the latter freezes the initial four-snapshot family and its six-method plans.
This module consumes that complete initial family plus actual controller/journal
and 24/120-row SPSA point evidence. It never creates scores or controller state.
The frozen initial selector is replayed first to verify the initialization; the
final-center selector is called only after all 40 paired proposals, all 20
replayed ask/tell transitions, and all four complete centers have been verified
from pinned files.

No partial family can produce selection evidence. All rows must be official
Stage-A original-PCM rows with seven-array ``[N,31]`` SyncNet payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.experiments.cross_tfg_audio_head_20261003 import (
    fit_only_spsa_selection_attempt06 as selector,
    fit_only_stagea_ingest_attempt03 as initial_ingest,
    prepare_fit40_spsa_stagea_packets_attempt05 as packet_builder,
)


SCHEMA = "fit40_final_center_stagea_ingest_v1"
INITIAL_SELECTOR_RECEIPT_SCHEMA = "fit40_initial_family_selector_receipt_v1"
CONTROLLER_JOURNAL_SCHEMA = "fit40_spsa_controller_journal_v1"
COMPOSED_PLAN_SCHEMA = "cross-tfg-audio-eval-v1"
JOURNAL_ENTRIES = 20
PROPOSAL_POINT_COUNT = 40
CENTER_STEPS = (5, 10, 15, 20)
CENTER_COUNT = 4
FINAL_ROW_COUNT = 720
INITIAL_ROW_COUNT = 720
STATUS_READY = "READY_FOR_COMPLETE_FINAL_CENTER_INGEST_NOT_SELECTED"
TERMINAL_BLOCK = "BLOCK_COMPLETE"
ALLOWED_INITIAL_SELECTION = {"ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV", "NOT_ADMITTED"}


class FinalCenterIngestError(ValueError):
    """Pinned final-center evidence is incomplete, malformed, or inconsistent."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _hash_object(value: Mapping[str, Any], field: str, label: str) -> str:
    body = dict(value)
    declared = body.pop(field, None)
    actual = _sha(_canonical(body))
    if declared != actual:
        raise FinalCenterIngestError(f"{label} {field} mismatch")
    return actual


def _plain(value: Any) -> Any:
    """Canonical JSON-compatible projection for frozen controller dataclasses."""
    from dataclasses import fields, is_dataclass

    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return _plain(value.item())
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise FinalCenterIngestError(f"controller state contains non-JSON value {type(value).__name__}")


def controller_state(controller: selector.SPSAController) -> dict[str, Any]:
    """Return the complete replayable ask/tell state, including RNG/history."""
    pending = controller._pending
    return {
        "schema": "fit40_spsa_controller_state_v1",
        "iteration": controller.iteration,
        "theta": list(controller.theta),
        "theta_initial": list(controller.theta_initial),
        "input_pins_sha256": controller.input_pins_sha256,
        "candidate_artifact_sha256": controller.candidate_artifact_sha256,
        "metric_backend": controller.metric_backend,
        "identity_reference_method_id": controller.identity_reference_method_id,
        "rms_reference_method_id": controller.rms_reference_method_id,
        "rng_bit_generator": controller._rng.bit_generator.__class__.__name__,
        "rng_state": _plain(controller._rng.bit_generator.state),
        "pending_pair": _plain(pending),
        "halted_reason": controller._halted_reason,
        "history": _plain(controller.history),
    }


@dataclass(frozen=True)
class FinalCenterIngest:
    """Verified final selector result and the evidence accounting that produced it."""

    initial_selection: Any
    final_selection: selector.SnapshotSelection
    initial_ingest_index_sha256: str
    initial_selector_receipt_sha256: str
    initial_selector_receipt_self_sha256: str
    controller_journal_sha256: str
    controller_journal_self_sha256: str
    composed_family_plan_sha256: str
    composed_family_plan_self_sha256: str
    execution_package_manifest_sha256: str
    execution_package_manifest_self_sha256: str
    scientific_input_pins_sha256: str
    verified_initial_rows: int
    verified_proposal_rows: int
    verified_center_rows: int
    verified_final_rows: int
    verified_candidate_npz_count: int
    center_specs: tuple[selector.SnapshotSpec, ...]


def _load_json_pin(files: Mapping[str, initial_ingest._PinnedFile], file_id: Any,
                   role: str, label: str) -> tuple[initial_ingest._PinnedFile, dict[str, Any], bytes]:
    pin = initial_ingest._required_file(files, file_id, role)
    raw = initial_ingest._read_pinned(pin)
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise FinalCenterIngestError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise FinalCenterIngestError(f"{label} must be a JSON object")
    return pin, value, raw


def _verify_receipt_link(receipt: Mapping[str, Any], *, index_pin: initial_ingest._PinnedFile,
                         index_self_sha256: str) -> None:
    if receipt.get("schema") != INITIAL_SELECTOR_RECEIPT_SCHEMA:
        raise FinalCenterIngestError("initial selector receipt schema differs from the declared contract")
    if receipt.get("status") != "INITIAL_720_SELECTOR_COMPLETE":
        raise FinalCenterIngestError("initial selector receipt is not terminal complete")
    if receipt.get("metric_backend") != "torch_cpu":
        raise FinalCenterIngestError("initial selector receipt must record canonical torch_cpu metrics")
    link = receipt.get("initial_ingest_index")
    expected = {
        "path": index_pin.declared_path,
        "bytes": index_pin.byte_count,
        "sha256": index_pin.sha256,
        "self_sha256": index_self_sha256,
    }
    if link != expected:
        raise FinalCenterIngestError("initial selector receipt does not bind the exact initial ingest index")


def _verify_initial_preselector(index: Mapping[str, Any], files: Mapping[str, initial_ingest._PinnedFile]) -> None:
    """Reject incomplete/failed initial families before calling consumer03's selector."""
    if index.get("schema") != initial_ingest.SCHEMA:
        raise FinalCenterIngestError("initial ingest index has the wrong schema")
    _hash_object(index, "index_sha256", "initial ingest index")
    if index.get("status") != "READY_FOR_LOCAL_INGEST_NOT_SELECTION":
        raise FinalCenterIngestError("initial ingest index is not the sealed preselection packet")
    model_entries = index.get("models")
    if not isinstance(model_entries, list) or len(model_entries) != 3:
        raise FinalCenterIngestError("initial family must contain exactly three model plans")
    all_ids: set[str] = set()
    all_records: list[Mapping[str, Any]] = []
    all_npz_refs: set[str] = set()
    model_ids: list[str] = []
    for entry in model_entries:
        try:
            (model_id, plan, _plan_raw, planned_rows, _method_map, records,
             errors, _used) = initial_ingest._parse_model_family(entry, files)
        except Exception as exc:
            raise FinalCenterIngestError(f"initial model family failed before selector: {exc}") from exc
        model_ids.append(model_id)
        if errors:
            raise FinalCenterIngestError(f"initial model {model_id} has ledger errors: {errors[:3]}")
        if len(planned_rows) != initial_ingest.ROWS_PER_MODEL or len(records) != initial_ingest.ROWS_PER_MODEL:
            raise FinalCenterIngestError(f"initial model {model_id} is not a complete 240-row plan/ledger")
        planned_ids = [row.get("row_id") for row in planned_rows]
        record_ids = [row.get("row_id") for row in records]
        if (len(set(planned_ids)) != initial_ingest.ROWS_PER_MODEL
                or set(record_ids) != set(planned_ids)
                or len(record_ids) != len(planned_ids)):
            raise FinalCenterIngestError(f"initial model {model_id} has missing/duplicate observed rows")
        if any(record.get("status") != "PASS" for record in records):
            raise FinalCenterIngestError(f"initial model {model_id} contains FAILED/INCOMPLETE rows")
        for row_id in planned_ids:
            if row_id in all_ids:
                raise FinalCenterIngestError(f"initial family repeats row ID {row_id}")
            all_ids.add(row_id)
        all_records.extend(records)
        # Ensure that every passing initial row has a declared, unique raw NPZ
        # before the initial consumer is allowed to invoke selector06.
        for record in records:
            score = record.get("score")
            raw_ref = score.get("raw_matrix_npz") if isinstance(score, Mapping) else None
            path = raw_ref.get("path") if isinstance(raw_ref, Mapping) else None
            if not isinstance(path, str) or not path:
                raise FinalCenterIngestError(f"initial row {record.get('row_id')} lacks official NPZ reference")
            if path in all_npz_refs:
                raise FinalCenterIngestError(f"initial raw NPZ path is reused: {path}")
            all_npz_refs.add(path)
    if model_ids != list(index.get("model_ids", model_ids)):
        raise FinalCenterIngestError("initial model order differs from the index model roster")
    if len(all_ids) != INITIAL_ROW_COUNT or index.get("expected_row_count") != INITIAL_ROW_COUNT:
        raise FinalCenterIngestError("initial full family denominator is not exactly 720")
    if len(all_records) != INITIAL_ROW_COUNT or len(all_npz_refs) != INITIAL_ROW_COUNT:
        raise FinalCenterIngestError("initial family rows or NPZ references are incomplete")


def _verify_composed_family_plan(
    plan: Mapping[str, Any], raw: bytes, plan_pin: initial_ingest._PinnedFile,
    initial_index: Mapping[str, Any], initial_files: Mapping[str, initial_ingest._PinnedFile],
    initial_family: initial_ingest.FamilySelection,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    if (plan.get("schema_version") != COMPOSED_PLAN_SCHEMA or plan.get("status") != "FROZEN"
            or _hash_object(plan, "plan_sha256", "composed family plan") != plan.get("plan_sha256")):
        raise FinalCenterIngestError("composed family plan is not an exact frozen self-hashed Stage-A plan")
    if (len(plan.get("rows", [])) != FINAL_ROW_COUNT
            or len(plan.get("methods", [])) != 6
            or plan.get("scientific_input_pins_sha256") != initial_family.family.input_pins_sha256
            or plan.get("scientific_input_pins") != initial_family.family.input_pins):
        raise FinalCenterIngestError("composed family plan does not preserve the actual initial 720/input pins")
    model_ids = tuple(initial_family.family.roster.model_ids)
    if tuple(model.get("model_id") for model in plan.get("models", [])) != model_ids:
        raise FinalCenterIngestError("composed family model roster/order differs from initial selector family")
    execution = plan.get("execution")
    if not isinstance(execution, Mapping) or execution.get("dev_or_confirm_access") is not False:
        raise FinalCenterIngestError("composed family plan does not explicitly deny Dev/Confirm access")
    source_plans = execution.get("source_model_plans")
    if not isinstance(source_plans, list) or [item.get("model_id") for item in source_plans] != list(model_ids):
        raise FinalCenterIngestError("composed family plan lacks ordered source-model plan bindings")
    initial_entries = {entry.get("model_id"): entry for entry in initial_index.get("models", [])}
    initial_by_model: dict[str, tuple[Mapping[str, Any], list[dict[str, Any]]]] = {}
    for item in initial_entries.values():
        model_id = item.get("model_id")
        plan_file = initial_ingest._required_file(initial_files, item.get("plan_file_id"), "plan")
        rows_file = initial_ingest._required_file(initial_files, item.get("planned_rows_file_id"), "planned_rows")
        p_raw = initial_ingest._read_pinned(plan_file)
        r_raw = initial_ingest._read_pinned(rows_file)
        p_value = json.loads(p_raw.decode("utf-8"))
        rows = initial_ingest._parse_jsonl(r_raw, f"initial rows {model_id}")
        initial_by_model[model_id] = (p_value, rows)
    for source in source_plans:
        model_id = source["model_id"]
        source_plan_pin = source.get("plan")
        source_rows_pin = source.get("planned_rows")
        if not isinstance(source_plan_pin, Mapping) or not isinstance(source_rows_pin, Mapping):
            raise FinalCenterIngestError(f"composed plan source binding malformed for {model_id}")
        original_plan, original_rows = initial_by_model[model_id]
        original_entry = initial_entries[model_id]
        original_plan_pin = initial_ingest._required_file(initial_files, original_entry.get("plan_file_id"), "plan")
        original_rows_pin = initial_ingest._required_file(initial_files, original_entry.get("planned_rows_file_id"), "planned_rows")
        if (source_plan_pin.get("sha256") != original_plan_pin.sha256
                or source_plan_pin.get("bytes") != original_plan_pin.byte_count
                or source_plan_pin.get("path") != original_plan_pin.declared_path
                or source_rows_pin.get("sha256") != original_rows_pin.sha256
                or source_rows_pin.get("bytes") != original_rows_pin.byte_count
                or source_rows_pin.get("path") != original_rows_pin.declared_path):
            raise FinalCenterIngestError(f"composed plan source bytes/path differ from actual initial plan: {model_id}")
        descriptor = next((x for x in plan["models"] if x.get("model_id") == model_id), None)
        if descriptor != original_plan.get("models", [None])[0]:
            raise FinalCenterIngestError(f"composed model descriptor differs from its actual source plan: {model_id}")
        projected_rows = [row for row in plan["rows"] if row.get("model_id") == model_id]
        source_rows_by_id = {row.get("row_id"): row for row in original_rows}
        if (len(projected_rows) != initial_ingest.ROWS_PER_MODEL
                or {row.get("row_id"): row for row in projected_rows} != source_rows_by_id):
            raise FinalCenterIngestError(f"composed model rows differ from initial source plan: {model_id}")
    _verify_composed_methods_and_protocol(
        plan,
        {model_id: (source_plan, source_rows)
         for model_id, (source_plan, source_rows) in initial_by_model.items()},
    )
    # The source plans are already individually checked by the initial ingest;
    # keep their method maps and exact context rows for point-plan validation.
    method_maps: dict[str, Mapping[str, Any]] = {}
    rows_by_context: dict[str, Mapping[str, Any]] = {}
    for model_id, (source_plan, source_rows) in initial_by_model.items():
        method_maps[model_id] = {item["method_id"]: item for item in source_plan["methods"]}
        for row in source_rows:
            key = f"{row['sample_id']}|{model_id}|{row['method_id']}"
            rows_by_context[key] = row
    return dict(plan), method_maps, rows_by_context


def _point_method_spec(method: Mapping[str, Any], *, expected_kind: str,
                       expected_id: str, input_sha: str,
                       expected_seed_id: str, expected_seed_step: int,
                       expected_seed_theta: Sequence[float],
                       expected_seed_artifact: str) -> selector.SnapshotSpec:
    if method.get("kind") != expected_kind or method.get("checkpoint_sha256") is not None:
        raise FinalCenterIngestError("point method kind/checkpoint identity differs from frozen SPSA producer")
    config = method.get("config")
    if not isinstance(config, Mapping) or method.get("config_sha256") != _sha(_canonical(config)):
        raise FinalCenterIngestError("point method config is malformed or not canonically hashed")
    if expected_kind == "fit40_spsa_center":
        if (config.get("method") != "fit40_spsa_center"
                or config.get("schema") != "fit40_spsa_center_transform_v1"
                or config.get("point_kind") != "spsa_center"
                or config.get("method_id") != expected_id):
            raise FinalCenterIngestError("center method config/schema/point ID differs")
        step = config.get("source_step")
    elif expected_kind == "fit40_spsa_proposal":
        if (config.get("method") != "fit40_spsa_proposal"
                or config.get("schema") != "fit40_spsa_proposal_transform_v1"
                or config.get("method_id") != expected_id):
            raise FinalCenterIngestError("proposal method config/schema/point ID differs")
        step = config.get("iteration")
    else:
        raise FinalCenterIngestError("unsupported SPSA method kind")
    try:
        theta = tuple(float(x) for x in config.get("theta", []))
    except Exception as exc:
        raise FinalCenterIngestError("point theta is not numeric") from exc
    if len(theta) != 9 or not all(math.isfinite(x) for x in theta):
        raise FinalCenterIngestError("point theta must contain nine finite values")
    theta_sha = _sha(_canonical(list(theta)))
    artifact = config.get("artifact_sha256")
    binding = config.get("binding_sha256")
    if (config.get("theta_sha256") != theta_sha
            or config.get("input_pins_sha256") != input_sha
            or config.get("source_seed_method_id") != expected_seed_id
            or config.get("source_snapshot_step") != expected_seed_step
            or config.get("source_theta") != list(expected_seed_theta)
            or config.get("source_snapshot_artifact_sha256") != expected_seed_artifact
            or artifact != expected_seed_artifact):
        raise FinalCenterIngestError("point theta or persisted seed lineage differs from actual initialization")
    if expected_kind == "fit40_spsa_center":
        if step not in CENTER_STEPS or method.get("method_id") != expected_id:
            raise FinalCenterIngestError("center step/method is not one of the four predeclared k values")
        expected_binding = selector.full_fit_candidate_binding_sha256(
            method_id=expected_id, artifact_sha256=artifact,
            input_pins_sha256=input_sha, source_step=step, theta=theta,
            point_kind="spsa_center",
        )
        point_kind = "spsa_center"
    else:
        if type(step) is not int or not 0 <= step < JOURNAL_ENTRIES:
            raise FinalCenterIngestError("proposal iteration is outside the 20-pair journal")
        side = config.get("side")
        if side not in {"plus", "minus"}:
            raise FinalCenterIngestError("proposal side is not plus/minus")
        expected_binding = selector._sha256(selector._canonical({
            "schema": "fit40_spsa_proposal_binding_v1",
            "proposal_id": config.get("proposal_id"),
            "pair_id": config.get("pair_id"),
            "side": side,
            "iteration": step,
            "method_id": expected_id,
            "theta_sha256": theta_sha,
            "artifact_sha256": artifact,
            "input_pins_sha256": input_sha,
            "batch_id": config.get("batch_id"),
            "renderer_seed": selector.RENDERER_SEED,
            "sample_ids": config.get("sample_ids"),
        }))
        point_kind = "snapshot"
    if binding != expected_binding:
        raise FinalCenterIngestError("point binding SHA differs from the frozen selector binding function")
    source_step = step if point_kind == "spsa_center" else 0
    return selector.SnapshotSpec(expected_id, artifact, binding, source_step, theta, point_kind)


def _verify_plan_projection(point_plan: Mapping[str, Any], base_plan: Mapping[str, Any],
                            point_raw: bytes, composed_pin: initial_ingest._PinnedFile,
                            method: Mapping[str, Any], expected_samples: Sequence[str],
                            rows_by_context: Mapping[str, Mapping[str, Any]],
                            expected_scope: Mapping[str, Any]) -> list[dict[str, Any]]:
    if (point_plan.get("schema_version") != COMPOSED_PLAN_SCHEMA
            or point_plan.get("status") != "FROZEN"):
        raise FinalCenterIngestError("SPSA point plan is not frozen Stage-A schema")
    _hash_object(point_plan, "plan_sha256", "SPSA point plan")
    mutable = {"run_id", "methods", "rows", "generation_repeat_qc", "execution", "analysis", "plan_sha256"}
    for key, value in base_plan.items():
        if key not in mutable and point_plan.get(key) != value:
            raise FinalCenterIngestError(f"SPSA point plan changed inherited family field {key}")
    for key in set(point_plan) - set(base_plan) - mutable:
        raise FinalCenterIngestError(f"SPSA point plan added unreviewed family field {key}")
    if point_plan.get("models") != base_plan.get("models"):
        raise FinalCenterIngestError("SPSA point plan changed model descriptors")
    point_execution = point_plan.get("execution")
    base_execution = base_plan.get("execution")
    if not isinstance(point_execution, Mapping) or not isinstance(base_execution, Mapping):
        raise FinalCenterIngestError("SPSA point plan lacks execution lineage")
    parent_pin = point_execution.get("spsa_execution_parent_plan")
    expected_parent = {"path": composed_pin.declared_path, "bytes": composed_pin.byte_count,
                       "sha256": composed_pin.sha256}
    if parent_pin != expected_parent:
        raise FinalCenterIngestError("SPSA point plan does not bind the exact composed family plan")
    mutable_execution = {"spsa_execution_parent_plan", "spsa_execution_scope", "runtime_readiness", "planned_rows"}
    if {k: v for k, v in point_execution.items() if k not in mutable_execution} != {
            k: v for k, v in base_execution.items() if k not in mutable_execution}:
        raise FinalCenterIngestError("SPSA point plan changed inherited runtime/score execution protocol")
    if point_plan.get("methods") != [dict(method)] or point_plan.get("generation_repeat_qc") != []:
        raise FinalCenterIngestError("SPSA point plan does not contain exactly the expected method/no-repeat rows")
    scope = point_execution.get("spsa_execution_scope")
    if not isinstance(scope, Mapping) or dict(scope) != dict(expected_scope):
        raise FinalCenterIngestError("SPSA point plan execution scope is not bound to its exact method")
    if point_execution.get("runtime_readiness") != "REQUIRED_SEPARATE_SPSA_BINDING_BEFORE_EXECUTION":
        raise FinalCenterIngestError("SPSA point plan runtime readiness marker differs from producer")
    if point_execution.get("planned_rows") != len(expected_samples) * 3:
        raise FinalCenterIngestError("SPSA point plan denominator differs from 8×3 or 40×3")
    actual_rows = point_plan.get("rows")
    if not isinstance(actual_rows, list) or len(actual_rows) != len(expected_samples) * 3:
        raise FinalCenterIngestError("SPSA point plan row count differs from its exact scope")
    expected_rows: list[dict[str, Any]] = []
    model_ids = [item["model_id"] for item in base_plan["models"]]
    method_id = method["method_id"]
    for sample_id in expected_samples:
        for model_id in model_ids:
            identity = rows_by_context.get(f"{sample_id}|{model_id}|identity")
            if not isinstance(identity, Mapping):
                raise FinalCenterIngestError(f"family identity row missing for {sample_id}/{model_id}")
            expected = dict(identity)
            expected["row_id"] = packet_builder._row_id(identity, method_id)
            expected["method_id"] = method_id
            expected["method_config_sha256"] = method["config_sha256"]
            expected["checkpoint_sha256"] = None
            expected["spsa_point_binding_sha256"] = method["config"]["binding_sha256"]
            expected["score_audio"] = dict(expected["natural_pcm"])
            expected_rows.append(expected)
    if actual_rows != expected_rows:
        raise FinalCenterIngestError("SPSA point planned rows are not the exact source identity projection")
    return [dict(row) for row in actual_rows]


def _status_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in ("PASS", "FAILED", "INCOMPLETE")}
    for record in records:
        status = record.get("status")
        if status not in counts:
            raise FinalCenterIngestError(f"unsupported terminal result status {status!r}")
        counts[status] += 1
    return counts


def _validate_utc_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise FinalCenterIngestError(f"{label} must be an RFC3339 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise FinalCenterIngestError(f"{label} is not a valid RFC3339 UTC timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise FinalCenterIngestError(f"{label} is not explicitly UTC")
    return parsed


def _verify_external_file_pin(
    value: Any, files: Mapping[str, initial_ingest._PinnedFile], label: str,
    *, require_self_sha256: bool = False,
) -> dict[str, Any]:
    """Require a provenance pin to resolve to exactly one already indexed file."""
    if not isinstance(value, Mapping):
        raise FinalCenterIngestError(f"{label} must be a file pin object")
    required = {"path", "bytes", "sha256"}
    allowed = required | ({"self_sha256"} if require_self_sha256 else set())
    if set(value) != allowed:
        raise FinalCenterIngestError(f"{label} has unexpected/missing pin fields")
    if (not isinstance(value.get("path"), str) or not value["path"]
            or type(value.get("bytes")) is not int or value["bytes"] < 0
            or not initial_ingest._is_sha(value.get("sha256"))):
        raise FinalCenterIngestError(f"{label} has invalid path/size/raw SHA")
    if require_self_sha256 and (
            not initial_ingest._is_sha(value.get("self_sha256"))):
        raise FinalCenterIngestError(f"{label} has invalid embedded self SHA")
    matches = [pin for pin in files.values()
               if pin.declared_path == value["path"] and pin.byte_count == value["bytes"]
               and pin.sha256 == value["sha256"]]
    if not matches:
        raise FinalCenterIngestError(f"{label} does not resolve to an exact file-table entry")
    for match in matches:
        initial_ingest._read_pinned(match)
    return dict(value)


def _validate_execution_package_manifest(
    package: Mapping[str, Any], package_pin: initial_ingest._PinnedFile,
    family_plan: Mapping[str, Any], family_pin: initial_ingest._PinnedFile,
    family_self_sha256: str, files: Mapping[str, initial_ingest._PinnedFile],
) -> dict[str, Any]:
    if package.get("schema") != "fit40_spsa_point_execution_package_manifest_v1" or package.get("status") != "FROZEN":
        raise FinalCenterIngestError("point execution package manifest schema/status differs")
    package_self = _hash_object(package, "manifest_sha256", "point execution package manifest")
    if not isinstance(package.get("package_id"), str) or not package["package_id"]:
        raise FinalCenterIngestError("point execution package lacks its immutable package ID")
    expected_family = _pin_link(family_pin, family_self_sha256)
    if package.get("composed_family_plan_pin") != expected_family:
        raise FinalCenterIngestError("point execution package does not bind the exact composed family plan")
    family_models = [entry.get("model_id") for entry in family_plan.get("models", [])]
    by_model = package.get("execution_sources_by_model")
    if not isinstance(by_model, Mapping) or set(by_model) != set(family_models):
        raise FinalCenterIngestError("point execution package model source order differs from the frozen family")
    shared_helper = _verify_external_file_pin(
        package.get("shared_application_helper_pin"), files, "shared application helper pin")
    point_runner = _verify_external_file_pin(
        package.get("point_runner_pin"), files, "native point runner pin")
    expected_fields = {
        "launcher_pin", "runner_pin", "adapter_pin", "runtime_binding_pin", "source_pins",
    }
    checked_models: dict[str, dict[str, Any]] = {}
    runtime_bindings = family_plan.get("execution", {}).get("runtime_bindings_by_model")
    for model_id in family_models:
        model_sources = by_model.get(model_id)
        if not isinstance(model_sources, Mapping) or set(model_sources) != expected_fields:
            raise FinalCenterIngestError(f"point execution source closure fields differ for {model_id}")
        checked = {
            field: _verify_external_file_pin(
                model_sources.get(field), files, f"{model_id} execution package {field}")
            for field in ("launcher_pin", "runner_pin", "adapter_pin", "runtime_binding_pin")
        }
        source_pins = model_sources.get("source_pins")
        if not isinstance(source_pins, Mapping) or not source_pins:
            raise FinalCenterIngestError(f"point execution package lacks source pins for {model_id}")
        checked_source_pins: dict[str, dict[str, Any]] = {}
        for role, source_pin in source_pins.items():
            if not isinstance(role, str) or not role:
                raise FinalCenterIngestError(f"point execution source pin role is malformed for {model_id}")
            checked_source_pins[role] = _verify_external_file_pin(
                source_pin, files, f"{model_id} execution package source {role}")
        checked["source_pins"] = checked_source_pins
        binding_map = runtime_bindings.get(model_id) if isinstance(runtime_bindings, Mapping) else None
        runtime = checked["runtime_binding_pin"]
        expected_digest = ((binding_map.get("raw_sha256") or binding_map.get("sha256"))
                           if isinstance(binding_map, Mapping) else None)
        if (not isinstance(binding_map, Mapping)
                or runtime.get("path") != binding_map.get("path")
                or runtime.get("sha256") != expected_digest
                or ("bytes" in binding_map and runtime.get("bytes") != binding_map["bytes"])):
            raise FinalCenterIngestError(f"execution package runtime pin differs from frozen model runtime: {model_id}")
        checked_models[model_id] = checked
    if package_pin.sha256 != _sha(initial_ingest._read_pinned(package_pin)):
        raise FinalCenterIngestError("execution package raw pin changed during validation")
    return {
        "package_pin": _pin_link(package_pin, package_self),
        "shared_application_helper_pin": shared_helper,
        "point_runner_pin": point_runner,
        "execution_sources_by_model": checked_models,
    }


def _point_index_maps(index: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[int, Mapping[str, Any]]]:
    proposals = index.get("proposal_points")
    centers = index.get("center_points")
    if not isinstance(proposals, list) or len(proposals) != PROPOSAL_POINT_COUNT:
        raise FinalCenterIngestError("final index must contain exactly forty proposal point entries")
    if not isinstance(centers, list) or len(centers) != CENTER_COUNT:
        raise FinalCenterIngestError("final index must contain exactly four full-center entries")
    by_method: dict[str, Mapping[str, Any]] = {}
    for entry in proposals:
        if not isinstance(entry, Mapping):
            raise FinalCenterIngestError("proposal point entry is not an object")
        method_id = entry.get("method_id")
        if not isinstance(method_id, str) or not method_id or method_id in by_method:
            raise FinalCenterIngestError("proposal method IDs are missing or duplicated")
        if entry.get("point_id") != method_id or entry.get("point_kind") != "spsa_proposal":
            raise FinalCenterIngestError("proposal point identity/kind is malformed")
        by_method[method_id] = entry
    center_by_step: dict[int, Mapping[str, Any]] = {}
    for entry in centers:
        if not isinstance(entry, Mapping):
            raise FinalCenterIngestError("center point entry is not an object")
        step = entry.get("source_step")
        if type(step) is not int or step not in CENTER_STEPS or step in center_by_step:
            raise FinalCenterIngestError("center steps are missing, duplicated, or outside 5/10/15/20")
        method_id = entry.get("method_id")
        if (entry.get("point_id") != method_id
                or entry.get("point_kind") != "spsa_center"
                or method_id != f"fit40_d0_spsa_center_k{step:02d}"):
            raise FinalCenterIngestError("center point ID/kind differs from its predeclared step")
        center_by_step[step] = entry
    if tuple(sorted(center_by_step)) != CENTER_STEPS:
        raise FinalCenterIngestError("center entries do not cover all four predeclared steps")
    return by_method, center_by_step


def _validate_final_index_header(index: Mapping[str, Any]) -> None:
    if index.get("schema") != SCHEMA:
        raise FinalCenterIngestError("final-center index schema differs")
    _hash_object(index, "index_sha256", "final-center index")
    if index.get("status") != STATUS_READY:
        raise FinalCenterIngestError("final-center index is not in the preselection ready state")
    if (index.get("metric_backend") != "torch_cpu"
            or index.get("selection_status") != "NOT_SELECTED"):
        raise FinalCenterIngestError("final-center index must be unselected and bind torch_cpu metrics")
    exact_counts = {
        "expected_initial_row_count": INITIAL_ROW_COUNT,
        "expected_final_row_count": FINAL_ROW_COUNT,
        "expected_proposal_pair_count": JOURNAL_ENTRIES,
        "expected_proposal_point_count": PROPOSAL_POINT_COUNT,
        "expected_center_count": CENTER_COUNT,
        "expected_center_row_count": CENTER_COUNT * 40 * 3,
        "expected_proposal_row_count": PROPOSAL_POINT_COUNT * 8 * 3,
    }
    for key, expected in exact_counts.items():
        if type(index.get(key)) is not int or index[key] != expected:
            raise FinalCenterIngestError(f"final-center index {key} does not retain the frozen denominator")
    files = index.get("files")
    if not isinstance(files, list) or not files:
        raise FinalCenterIngestError("final-center index lacks a non-empty exact file table")
    if not isinstance(index.get("execution_package_manifest_file_id"), str):
        raise FinalCenterIngestError("final-center index lacks its independently sealed execution package manifest")
    _point_index_maps(index)


def _index_file(index: Mapping[str, Any], files: Mapping[str, initial_ingest._PinnedFile],
                field: str, role: str, label: str) -> initial_ingest._PinnedFile:
    pin = initial_ingest._required_file(files, index.get(field), role)
    return pin


def _validate_point_blocks(
    entry: Mapping[str, Any], point_plan: Mapping[str, Any], point_pin: initial_ingest._PinnedFile,
    planned_rows: list[dict[str, Any]], planned_rows_pin: initial_ingest._PinnedFile,
    family_pin: initial_ingest._PinnedFile, family_self_sha256: str,
    files: Mapping[str, initial_ingest._PinnedFile],
    expected_method: Mapping[str, Any], input_sha: str,
    npz_by_declared: Mapping[str, initial_ingest._PinnedFile],
    used_npz_ids: set[str], execution_package: Mapping[str, Any],
) -> tuple[list[selector.FitScoreRow], list[dict[str, Any]]]:
    """Read one point's three native model executions under the point-runner contract."""
    converted: list[selector.FitScoreRow] = []
    block_pins: list[dict[str, Any]] = []
    blocks = entry.get("blocks")
    model_ids = [model.get("model_id") for model in point_plan.get("models", [])]
    if not isinstance(blocks, list) or len(blocks) != len(model_ids) or len(model_ids) != 3:
        raise FinalCenterIngestError("each point must contain exactly one native execution block per model")
    if [block.get("model_id") if isinstance(block, Mapping) else None for block in blocks] != model_ids:
        raise FinalCenterIngestError("point execution blocks do not preserve the frozen model order")
    if len({block.get("model_id") for block in blocks if isinstance(block, Mapping)}) != 3:
        raise FinalCenterIngestError("point execution blocks repeat a model")
    point_self = point_plan.get("plan_sha256")
    family_self = family_self_sha256
    expected_point_pin = {
        "path": point_pin.declared_path, "bytes": point_pin.byte_count,
        "sha256": point_pin.sha256, "self_sha256": point_self,
    }
    expected_family_pin = {
        "path": family_pin.declared_path, "bytes": family_pin.byte_count,
        "sha256": family_pin.sha256, "self_sha256": family_self,
    }
    package_pin = execution_package.get("package_pin")
    if not isinstance(package_pin, Mapping):
        raise FinalCenterIngestError("point execution package was not validated")
    expected_point_kind = entry.get("point_kind")
    expected_row_ids_all = [row["row_id"] for row in planned_rows]
    per_model_rows = {
        model_id: [row for row in planned_rows if row.get("model_id") == model_id]
        for model_id in model_ids
    }
    method_id = expected_method["method_id"]
    converted_by_id: dict[str, selector.FitScoreRow] = {}
    for block in blocks:
        model_id = block["model_id"]
        model_rows = per_model_rows[model_id]
        model_row_ids = [row["row_id"] for row in model_rows]
        if block.get("block_index") != 0 or block.get("row_ids") != model_row_ids:
            raise FinalCenterIngestError(f"{model_id} native execution must be block zero over its exact row subset")
        model_rows_pin = initial_ingest._required_file(
            files, block.get("planned_rows_file_id"), "model_planned_rows")
        model_rows_raw = initial_ingest._read_pinned(model_rows_pin)
        decoded_model_rows = initial_ingest._parse_jsonl(model_rows_raw, f"{model_id} point model rows")
        if decoded_model_rows != model_rows:
            raise FinalCenterIngestError(f"{model_id} model planned_rows file is not the exact point-plan projection")
        expected_rows_link = {"path": model_rows_pin.declared_path,
                              "bytes": model_rows_pin.byte_count,
                              "sha256": model_rows_pin.sha256}
        descriptor = initial_ingest._model_descriptor({"models": [
            next(item for item in point_plan["models"] if item.get("model_id") == model_id)
        ]}, model_id)
        sync_protocol = initial_ingest._score_protocol(point_plan)
        status_pin = initial_ingest._required_file(files, block.get("run_status_file_id"), "run_status")
        manifest_pin = initial_ingest._required_file(files, block.get("result_manifest_file_id"), "result_manifest")
        ledger_pin = initial_ingest._required_file(files, block.get("results_file_id"), "results_jsonl")
        child_pin = initial_ingest._required_file(
            files, block.get("child_execution_receipt_file_id"), "child_execution_receipt")
        status_raw = initial_ingest._read_pinned(status_pin)
        manifest_raw = initial_ingest._read_pinned(manifest_pin)
        ledger_raw = initial_ingest._read_pinned(ledger_pin)
        child_raw = initial_ingest._read_pinned(child_pin)
        try:
            status = json.loads(status_raw.decode("utf-8"))
            manifest = json.loads(manifest_raw.decode("utf-8"))
            child = json.loads(child_raw.decode("utf-8"))
        except Exception as exc:
            raise FinalCenterIngestError(f"{model_id} point execution JSON is malformed: {exc}") from exc
        if not all(isinstance(value, Mapping) for value in (status, manifest, child)):
            raise FinalCenterIngestError(f"{model_id} point execution files must contain JSON objects")
        records = initial_ingest._parse_jsonl(ledger_raw, f"{model_id} point results")
        status_self = _hash_object(status, "status_sha256", f"{model_id} point run status")
        manifest_self = _hash_object(manifest, "manifest_sha256", f"{model_id} point result manifest")
        child_self = _hash_object(child, "receipt_sha256", f"{model_id} point child receipt")
        results_link = {"path": ledger_pin.declared_path, "bytes": ledger_pin.byte_count,
                        "sha256": ledger_pin.sha256}
        status_link = {"path": status_pin.declared_path, "bytes": status_pin.byte_count,
                       "sha256": status_pin.sha256, "self_sha256": status_self}
        manifest_link = {"path": manifest_pin.declared_path, "bytes": manifest_pin.byte_count,
                         "sha256": manifest_pin.sha256, "self_sha256": manifest_self}
        model_execution_sources = execution_package["execution_sources_by_model"].get(model_id)
        if not isinstance(model_execution_sources, Mapping):
            raise FinalCenterIngestError(f"execution package lacks the model source closure: {model_id}")
        expected_execution_pins = {
            "execution_package_manifest_pin": dict(package_pin),
            "point_runner_pin": execution_package["point_runner_pin"],
            "shared_application_helper_pin": execution_package["shared_application_helper_pin"],
            **dict(model_execution_sources),
        }
        expected_identity = {
            "run_id": entry.get("run_id"), "point_id": entry.get("point_id"),
            "method_id": method_id, "point_kind": expected_point_kind,
            "model_id": model_id, "block_index": 0,
            "family_plan_pin": expected_family_pin, "point_plan_pin": expected_point_pin,
            "planned_rows_pin": expected_rows_link, "row_ids": model_row_ids,
            "planned_rows": len(model_rows), "completed_rows": len(model_rows),
            "failed_rows": 0, "incomplete_rows": 0,
            "status_counts": {"PASS": len(model_rows), "FAILED": 0, "INCOMPLETE": 0},
        }
        if expected_point_kind == "spsa_proposal":
            for key in ("proposal_id", "pair_id", "iteration", "side"):
                expected_identity[key] = entry.get(key)
        else:
            expected_identity["source_step"] = entry.get("source_step")
        if (status.get("schema") != "fit40_spsa_point_run_status_v1"
                or status.get("status") != "COMPLETE"
                or manifest.get("schema") != "fit40_spsa_point_result_manifest_v1"
                or manifest.get("status") != "COMPLETE"
                or child.get("schema") != "fit40_spsa_point_child_execution_receipt_v1"
                or child.get("status") != "CHILD_EXITED"):
            raise FinalCenterIngestError(f"{model_id} point execution is not a complete successful terminal run")
        for label, envelope in (("run status", status), ("result manifest", manifest)):
            for key, expected_value in expected_identity.items():
                if envelope.get(key) != expected_value:
                    raise FinalCenterIngestError(f"{model_id} {label} {key} differs from the exact point roster")
        child_identity_keys = (
            "run_id", "point_id", "method_id", "point_kind", "model_id", "block_index",
            "family_plan_pin", "point_plan_pin", "planned_rows_pin", "row_ids",
        )
        if expected_point_kind == "spsa_proposal":
            child_identity_keys += ("proposal_id", "pair_id", "iteration", "side")
        else:
            child_identity_keys += ("source_step",)
        for key in child_identity_keys:
            if child.get(key) != expected_identity[key]:
                raise FinalCenterIngestError(f"{model_id} child receipt {key} differs from point identity")
        if (status.get("status_sha256") != status_self
                or status.get("result_manifest_pin") != manifest_link
                or status.get("results_jsonl_pin") != results_link
                or status.get("execution_package_manifest_pin") != dict(package_pin)
                or manifest.get("results_jsonl_pin") != results_link
                or manifest.get("planned_rows_pin") != expected_rows_link
                or manifest.get("execution_package_manifest_pin") != dict(package_pin)
                or manifest.get("manifest_sha256") != manifest_self
                or child.get("run_status_pin") != status_link
                or child.get("result_manifest_pin") != manifest_link
                or child.get("results_jsonl_pin") != results_link
                or child.get("execution_package_manifest_pin") != dict(package_pin)):
            raise FinalCenterIngestError(f"{model_id} status/result-manifest file lineage differs from exact pins")
        pid, ticks = status.get("child_pid"), status.get("child_start_ticks")
        child_pid, child_ticks = child.get("child_pid"), child.get("child_start_ticks")
        started = _validate_utc_timestamp(status.get("started_utc"), f"{model_id} status start")
        ended = _validate_utc_timestamp(status.get("ended_utc"), f"{model_id} status end")
        child_started = _validate_utc_timestamp(child.get("started_utc"), f"{model_id} child start")
        child_ended = _validate_utc_timestamp(child.get("ended_utc"), f"{model_id} child end")
        if (type(pid) is not int or pid <= 0 or type(ticks) is not int or ticks <= 0
                or (child_pid, child_ticks) != (pid, ticks)
                or type(child.get("child_exit_code")) is not int or child["child_exit_code"] != 0
                or child_started != started or child_ended != ended or ended < started):
            raise FinalCenterIngestError(f"{model_id} terminal status lacks valid process/time lineage")
        # The separately pinned execution package is the source of truth for the
        # point runner and model runtime. The index block and parent-captured child
        # repeat its closure; neither may self-assert code after review.
        for field, expected_exec_pin in expected_execution_pins.items():
            _verify_external_file_pin(expected_exec_pin, files, f"{model_id} packaged {field}")
            if block.get(field) != expected_exec_pin or child.get(field) != expected_exec_pin:
                raise FinalCenterIngestError(
                    f"{model_id} block/child {field} differs from the frozen execution package")
        if len(records) != len(model_rows) or [item.get("row_id") for item in records] != model_row_ids:
            raise FinalCenterIngestError(f"{model_id} results ledger order/count differs from the exact point rows")
        counts = _status_counts(records)
        if counts != expected_identity["status_counts"]:
            raise FinalCenterIngestError(f"{model_id} point contains failed or incomplete rows")
        npz_map = manifest.get("raw_score_npz_by_row")
        if not isinstance(npz_map, Mapping) or set(npz_map) != set(model_row_ids):
            raise FinalCenterIngestError(f"{model_id} result manifest does not pin one raw NPZ per row")
        for record, plan_row in zip(records, model_rows):
            initial_ingest._validate_record_against_plan(record, plan_row, descriptor, sync_protocol)
            score = record.get("score")
            raw_ref = score.get("raw_matrix_npz") if isinstance(score, Mapping) else None
            if not isinstance(raw_ref, Mapping):
                raise FinalCenterIngestError(f"result row {record.get('row_id')} lacks raw score NPZ metadata")
            declared = raw_ref.get("path")
            npz_pin = npz_by_declared.get(declared)
            manifest_npz_pin = npz_map.get(record["row_id"])
            expected_npz_link = {
                "path": npz_pin.declared_path, "bytes": npz_pin.byte_count,
                "sha256": npz_pin.sha256,
                "array_refs": raw_ref.get("arrays"),
            } if npz_pin is not None else None
            manifest_row = npz_map.get(record["row_id"])
            if (npz_pin is None or npz_pin.file_id in used_npz_ids
                    or not isinstance(manifest_row, Mapping)
                    or manifest_row.get("row_id") != record["row_id"]
                    or manifest_row.get("status") != "PASS"
                    or manifest_row.get("raw_matrix_npz") != expected_npz_link
                    or raw_ref.get("file_sha256") != npz_pin.sha256):
                raise FinalCenterIngestError(f"no unique exact NPZ file/array pin for result {record.get('row_id')}")
            arrays = initial_ingest._load_arrays(npz_pin, raw_ref)
            converted_row = initial_ingest.official_stagea_score_row(
                record, arrays, artifact_sha256=expected_method["config"]["artifact_sha256"],
                binding_sha256=expected_method["config"]["binding_sha256"],
                input_pins_sha256=input_sha, expected_anchor_column=15,
            )
            if record["row_id"] in converted_by_id:
                raise FinalCenterIngestError(f"point result repeats row ID {record['row_id']}")
            converted_by_id[record["row_id"]] = converted_row
            used_npz_ids.add(npz_pin.file_id)
        block_pins.append({
            "model_id": model_id, "block_index": 0, "row_ids": model_row_ids,
            "planned_rows_file_id": model_rows_pin.file_id,
            "child_execution_receipt_file_id": child_pin.file_id,
            "child_execution_receipt_sha256": child_pin.sha256,
            "child_execution_receipt_self_sha256": child_self,
            "run_status_file_id": status_pin.file_id,
            "run_status_sha256": status_pin.sha256,
            "result_manifest_file_id": manifest_pin.file_id,
            "result_manifest_sha256": manifest_pin.sha256,
            "results_file_id": ledger_pin.file_id,
            "results_sha256": ledger_pin.sha256,
            "result_manifest_self_sha256": manifest_self,
        })
    if set(converted_by_id) != set(expected_row_ids_all) or len(converted_by_id) != len(planned_rows):
        raise FinalCenterIngestError("point result rows do not exactly cover the point plan")
    converted = [converted_by_id[row_id] for row_id in expected_row_ids_all]
    return converted, block_pins


def _point_rows(entry: Mapping[str, Any], *, expected_kind: str, expected_id: str,
                expected_samples: Sequence[str], base_plan: Mapping[str, Any],
                composed_pin: initial_ingest._PinnedFile, composed_self_sha256: str,
                files: Mapping[str, initial_ingest._PinnedFile],
                method: Mapping[str, Any], rows_by_context: Mapping[str, Mapping[str, Any]],
                input_sha: str, npz_by_declared: Mapping[str, initial_ingest._PinnedFile],
                used_npz_ids: set[str], expected_scope: Mapping[str, Any],
                execution_package: Mapping[str, Any],
                ) -> tuple[list[selector.FitScoreRow], dict[str, Any]]:
    plan_pin, point_plan, point_raw = _load_json_pin(files, entry.get("plan_file_id"), "plan", "point plan")
    planned_pin = initial_ingest._required_file(files, entry.get("planned_rows_file_id"), "planned_rows")
    planned_rows = initial_ingest._parse_jsonl(initial_ingest._read_pinned(planned_pin), "point planned rows")
    if point_plan.get("run_id") != entry.get("run_id"):
        raise FinalCenterIngestError("point entry run ID differs from its plan")
    if point_plan.get("plan_sha256") != _hash_object(point_plan, "plan_sha256", "point plan"):
        raise FinalCenterIngestError("point plan payload hash changed")
    if point_plan.get("plan_sha256") != _sha(_canonical({k: v for k, v in point_plan.items() if k != "plan_sha256"})):
        raise FinalCenterIngestError("point plan self hash mismatch")
    planned = _verify_plan_projection(point_plan, base_plan, point_raw, composed_pin, method,
                                      expected_samples, rows_by_context, expected_scope)
    if planned_rows != planned or point_plan.get("rows") != planned:
        raise FinalCenterIngestError("point plan JSONL is not byte-decoded canonical plan row content")
    config = method["config"]
    expected_point_kind = "spsa_center" if expected_kind == "fit40_spsa_center" else "spsa_proposal"
    if (entry.get("point_id") != expected_id
            or entry.get("method_id") != expected_id
            or config.get("method_id") != expected_id
            or entry.get("point_kind") != expected_point_kind):
        raise FinalCenterIngestError("point entry identity differs from the actual controller point")
    if expected_kind == "fit40_spsa_proposal":
        if (entry.get("proposal_id") != config.get("proposal_id")
                or entry.get("pair_id") != config.get("pair_id")
                or entry.get("iteration") != config.get("iteration")
                or entry.get("side") != config.get("side")):
            raise FinalCenterIngestError("proposal entry IDs differ from its actual ask pair")
    else:
        if entry.get("source_step") != config.get("source_step"):
            raise FinalCenterIngestError("center entry step differs from the tell-derived center")
    converted, block_pins = _validate_point_blocks(
        entry, point_plan, plan_pin, planned, planned_pin, composed_pin,
        composed_self_sha256, files, method, input_sha,
        npz_by_declared, used_npz_ids, execution_package,
    )
    return converted, {
        "point_id": expected_id,
        "method_id": expected_id,
        "plan_file_id": plan_pin.file_id,
        "plan_raw_sha256": plan_pin.sha256,
        "planned_rows_file_id": planned_pin.file_id,
        "planned_rows_sha256": planned_pin.sha256,
        "row_count": len(planned),
        "block_pins": block_pins,
    }


def _initial_family(index_path: Path, index_pin: initial_ingest._PinnedFile,
                    initial_receipt: Mapping[str, Any],
                    initial_receipt_self_sha256: str) -> initial_ingest.FamilySelection:
    initial_index_raw = initial_ingest._read_pinned(index_pin)
    try:
        initial_index = json.loads(initial_index_raw.decode("utf-8"))
    except Exception as exc:
        raise FinalCenterIngestError(f"initial ingest index is not JSON: {exc}") from exc
    if not isinstance(initial_index, Mapping):
        raise FinalCenterIngestError("initial ingest index is not an object")
    nested_files = initial_ingest._file_table(initial_index)
    _verify_initial_preselector(initial_index, nested_files)
    initial_index_self = _hash_object(initial_index, "index_sha256", "initial ingest index")
    _verify_receipt_link(initial_receipt, index_pin=index_pin,
                         index_self_sha256=initial_index_self)
    receipt_self = _hash_object(initial_receipt, "receipt_sha256", "initial selector receipt")
    if (initial_receipt.get("initial_selector_index_sha256") != _sha(initial_index_raw)
            or initial_receipt.get("initial_selector_index_self_sha256") != initial_index_self
            or initial_receipt.get("selection_status") not in ALLOWED_INITIAL_SELECTION
            or initial_receipt.get("metric_backend") != "torch_cpu"):
        raise FinalCenterIngestError("initial selector receipt links/status/backend are inconsistent")
    family_selection = initial_ingest.ingest_family(index_pin.local_path, metric_backend="torch_cpu")
    family = family_selection.family
    if (not family.complete_transport or family_selection.selection is None
            or family_selection.status not in ALLOWED_INITIAL_SELECTION
            or family_selection.spsa_seed_method_id is None
            or family.producer_failure_rows or family.input_artifact_failures
            or family.integrity_errors):
        raise FinalCenterIngestError("actual initial selector family is not complete and fully passing")
    if (initial_receipt.get("selection") != _plain(family_selection.selection)
            or initial_receipt.get("selection_status") != family_selection.status
            or initial_receipt.get("spsa_seed_method_id") != family_selection.spsa_seed_method_id
            or initial_receipt.get("scientific_input_pins_sha256") != family.input_pins_sha256):
        raise FinalCenterIngestError("initial selector receipt differs from actual full-family re-evaluation")
    if _sha(initial_index_raw) != _sha(initial_ingest._read_pinned(index_pin)):
        raise FinalCenterIngestError("initial ingest index bytes changed during verification")
    if initial_receipt_self_sha256 != receipt_self:
        raise FinalCenterIngestError("initial selector receipt self hash differs from its outer pin")
    return family_selection


def _pin_link(pin: initial_ingest._PinnedFile, self_sha256: str) -> dict[str, Any]:
    return {"path": pin.declared_path, "bytes": pin.byte_count,
            "sha256": pin.sha256, "self_sha256": self_sha256}


def _verify_composed_methods_and_protocol(
    composed: Mapping[str, Any],
    source_plans: Mapping[str, tuple[Mapping[str, Any], list[dict[str, Any]]]],
) -> None:
    first_id = next(iter(source_plans))
    first_plan = source_plans[first_id][0]
    if composed.get("methods") != first_plan.get("methods"):
        raise FinalCenterIngestError("composed family method roster differs from its pinned W2L source plan")
    for model_id, (source_plan, _rows) in source_plans.items():
        if source_plan.get("methods") != first_plan.get("methods"):
            raise FinalCenterIngestError(f"source model method roster differs for {model_id}")
        if source_plan.get("protocol") != first_plan.get("protocol"):
            raise FinalCenterIngestError(f"source model score protocol differs for {model_id}")
        protocol_binding = composed.get("execution", {}).get("source_model_protocols", {}).get(model_id)
        if protocol_binding != source_plan.get("protocol"):
            raise FinalCenterIngestError(f"composed model protocol is not the exact source protocol for {model_id}")
        generator = source_plan.get("protocol", {}).get("generator")
        if composed.get("protocol", {}).get("generators_by_model", {}).get(model_id) != generator:
            raise FinalCenterIngestError(f"composed generator protocol differs from source model {model_id}")


def _controller_links(controller_journal: Mapping[str, Any], *, initial_index_pin: initial_ingest._PinnedFile,
                      initial_index_self: str, receipt_pin: initial_ingest._PinnedFile,
                      receipt_self: str, composed_pin: initial_ingest._PinnedFile,
                      composed_self: str) -> None:
    expected = {
        "initial_ingest_index": _pin_link(initial_index_pin, initial_index_self),
        "initial_selector_receipt": _pin_link(receipt_pin, receipt_self),
        "composed_family_plan": _pin_link(composed_pin, composed_self),
    }
    if any(controller_journal.get(key) != value for key, value in expected.items()):
        raise FinalCenterIngestError("controller journal source links differ from actual raw/payload pins")


def _evaluation(controller: selector.SPSAController, point_rows: Sequence[selector.FitScoreRow],
                family_rows: Sequence[selector.FitScoreRow], proposal: selector.SPSAProposal
                ) -> selector.ProposalEvaluation:
    sample_ids = set(proposal.sample_ids)
    refs = {controller.identity_reference_method_id, controller.rms_reference_method_id}
    reused = [row for row in family_rows if row.method_id in refs and row.sample_id in sample_ids]
    candidate = [row for row in point_rows if row.method_id == proposal.method_id]
    expected = len(sample_ids) * len(controller.roster.model_ids)
    if (len(reused) != expected * 2 or len(candidate) != expected
            or {row.sample_id for row in candidate} != sample_ids
            or {row.model_id for row in candidate} != set(controller.roster.model_ids)):
        raise FinalCenterIngestError("proposal evaluation lacks exact paired 8-clip/3-model rows or baseline reuse")
    return selector.ProposalEvaluation(
        proposal_id=proposal.proposal_id, pair_id=proposal.pair_id, side=proposal.side,
        method_id=proposal.method_id, theta=proposal.theta, theta_sha256=proposal.theta_sha256,
        artifact_sha256=proposal.artifact_sha256, binding_sha256=proposal.binding_sha256,
        input_pins_sha256=proposal.input_pins_sha256, batch_id=proposal.batch_id,
        rows=tuple((*reused, *candidate)),
    )


def _replay_controller_journal(
    journal: Mapping[str, Any], family_selection: initial_ingest.FamilySelection,
    initial_index_pin: initial_ingest._PinnedFile, initial_index_self: str,
    receipt_pin: initial_ingest._PinnedFile, receipt_self: str,
    composed_pin: initial_ingest._PinnedFile, composed_self: str,
    composed: Mapping[str, Any], files: Mapping[str, initial_ingest._PinnedFile],
    proposal_entries: Sequence[Mapping[str, Any]],
    npz_by_declared: Mapping[str, initial_ingest._PinnedFile], used_npz_ids: set[str],
    execution_package: Mapping[str, Any],
) -> tuple[selector.SPSAController, dict[str, list[selector.FitScoreRow]], list[dict[str, Any]]]:
    if (journal.get("schema") != CONTROLLER_JOURNAL_SCHEMA
            or journal.get("status") != "COMPLETE_20_ASK_TELL_PAIRS"):
        raise FinalCenterIngestError("controller journal is not a terminal 20-pair journal")
    _hash_object(journal, "journal_sha256", "controller journal")
    _controller_links(journal, initial_index_pin=initial_index_pin,
                      initial_index_self=initial_index_self, receipt_pin=receipt_pin,
                      receipt_self=receipt_self, composed_pin=composed_pin,
                      composed_self=composed_self)
    family = family_selection.family
    initial_family_pin_sha = family.input_pins_sha256
    methods = {method["method_id"]: method for method in composed["methods"]}
    seed_id = family_selection.spsa_seed_method_id
    seed_method = methods.get(seed_id)
    if not isinstance(seed_method, Mapping):
        raise FinalCenterIngestError("controller seed method is absent from the exact composed family")
    lineage = packet_builder._seed_snapshot_lineage(family_selection, seed_method)
    seed_spec = next((spec for spec in family.candidates if spec.method_id == seed_id), None)
    if seed_spec is None:
        raise FinalCenterIngestError("controller seed lacks a verified initial snapshot spec")
    controller = initial_ingest.initialize_spsa(family_selection, metric_backend="torch_cpu")
    if (journal.get("metric_backend") != "torch_cpu"
            or journal.get("scientific_input_pins_sha256") != initial_family_pin_sha
            or journal.get("seed_method_id") != seed_id
            or journal.get("initial_theta") != list(seed_spec.theta)
            or journal.get("candidate_artifact_sha256") != seed_spec.artifact_sha256
            or journal.get("rng_seed") != selector.SPSA_RNG_SEED
            or journal.get("initial_controller_state") != controller_state(controller)):
        raise FinalCenterIngestError("journal initialization does not exactly replay from the complete initial selector")
    entries = journal.get("entries")
    if not isinstance(entries, list) or len(entries) != JOURNAL_ENTRIES:
        raise FinalCenterIngestError("controller journal must contain exactly twenty ordered entries")
    points_by_method = {entry.get("method_id"): entry for entry in proposal_entries}
    if len(points_by_method) != PROPOSAL_POINT_COUNT or None in points_by_method:
        raise FinalCenterIngestError("proposal point index does not contain forty unique methods")
    expected_method_ids: set[str] = set()
    point_rows_by_method: dict[str, list[selector.FitScoreRow]] = {}
    row_receipts: list[dict[str, Any]] = []
    previous = None
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or entry.get("entry_index") != index:
            raise FinalCenterIngestError("journal entries are not in strict 0..19 order")
        if entry.get("previous_entry_sha256") != previous:
            raise FinalCenterIngestError(f"journal hash chain breaks before entry {index}")
        entry_sha = _hash_object(entry, "entry_sha256", f"journal entry {index}")
        pair = controller.ask()
        if entry.get("iteration") != index or entry.get("pair_id") != pair.pair_id:
            raise FinalCenterIngestError(f"journal pair identity differs from deterministic ask at {index}")
        if entry.get("ask_pair") != _plain(pair):
            raise FinalCenterIngestError(f"serialized ask pair differs from actual controller at {index}")
        point_ids = {"plus": pair.plus.method_id, "minus": pair.minus.method_id}
        expected_method_ids.update(point_ids.values())
        if (entry.get("plus_point_id") != point_ids["plus"]
                or entry.get("minus_point_id") != point_ids["minus"]):
            raise FinalCenterIngestError(f"journal point IDs differ from actual pair at {index}")
        candidate_evaluations = {}
        for side, proposal in (("plus", pair.plus), ("minus", pair.minus)):
            point_entry = points_by_method.get(proposal.method_id)
            if point_entry is None:
                raise FinalCenterIngestError(f"journal proposal plan is missing: {proposal.method_id}")
            expected_method = packet_builder._method_for_proposal(proposal, seed_method,
                                                                  initial_family_pin_sha, lineage)
            if point_entry.get("point_id") != proposal.method_id:
                raise FinalCenterIngestError("point entry ID differs from ask-generated method ID")
            expected_scope = {
                "scope_kind": "spsa_pair_proposal", "iteration": index,
                "pair_id": pair.pair_id, "proposal_id": proposal.proposal_id,
                "side": side, "batch_id": pair.batch_id,
                "sample_ids": list(pair.sample_ids), "speaker_ids": list(pair.speakers),
                "source_seed_method_id": seed_id, "method_id": proposal.method_id,
                "artifact_sha256": proposal.artifact_sha256,
                "binding_sha256": proposal.binding_sha256,
                "input_pins_sha256": initial_family_pin_sha,
                "run_id": f"fit40_spsa_k{index + 1:02d}_{side}_{proposal.proposal_id[:12]}",
            }
            point_rows, row_receipt = _point_rows(
                point_entry, expected_kind="fit40_spsa_proposal", expected_id=proposal.method_id,
                expected_samples=pair.sample_ids, base_plan=composed, composed_pin=composed_pin,
                composed_self_sha256=composed_self,
                files=files, method=expected_method, rows_by_context=_composed_identity_rows(composed),
                input_sha=initial_family_pin_sha, npz_by_declared=npz_by_declared,
                used_npz_ids=used_npz_ids, expected_scope=expected_scope,
                execution_package=execution_package,
            )
            point_rows_by_method[proposal.method_id] = point_rows
            candidate_evaluations[side] = _evaluation(controller, point_rows, family.rows, proposal)
            row_receipt.update({"iteration": index, "side": side, "proposal_id": proposal.proposal_id})
            row_receipts.append(row_receipt)
        tell = controller.tell(candidate_evaluations["plus"], candidate_evaluations["minus"])
        if entry.get("tell_result") != _plain(tell):
            raise FinalCenterIngestError(f"journal tell result differs from canonical controller at {index}")
        state = controller_state(controller)
        if entry.get("controller_state_after") != state:
            raise FinalCenterIngestError(f"journal full controller state differs after tell {index}")
        if tell.status != "UPDATED" or tell.reason is not None:
            raise FinalCenterIngestError(f"controller did not update from complete paired evidence at {index}")
        previous = entry_sha
    if set(points_by_method) != expected_method_ids:
        raise FinalCenterIngestError("proposal point roster contains an unjournaled or missing method")
    if controller.iteration != JOURNAL_ENTRIES or controller._pending is not None or controller.halted:
        raise FinalCenterIngestError("replayed controller did not finish exactly twenty clean ask/tell pairs")
    return controller, point_rows_by_method, row_receipts


def _composed_identity_rows(composed: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = composed.get("rows")
    if not isinstance(rows, list):
        raise FinalCenterIngestError("composed family lacks rows")
    out: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("method_id") != "identity":
            continue
        key = f"{row.get('sample_id')}|{row.get('model_id')}|identity"
        if key in out:
            raise FinalCenterIngestError("composed family repeats an identity context row")
        out[key] = row
    if len(out) != 120:
        raise FinalCenterIngestError("composed family does not bind all 120 identity source rows")
    return out


def _method_row_contexts(composed: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    # Kept separate from _composed_identity_rows to make the expected seed
    # source method explicit; point rows themselves are cloned from identity.
    return _composed_identity_rows(composed)


def ingest_final_center_family(index_path: str | Path, *, metric_backend: str = "torch_cpu") -> FinalCenterIngest:
    """Verify the complete initial family, real ask/tell journal, and final centers.

    The initial selector is replayed to verify the initialized family. The
    final-center selector is invoked only after every raw plan, full score row,
    NPZ, journal transition, and 24/120-row point family has passed its pin and
    denominator checks. This function never initializes or advances SPSA.
    """
    if metric_backend != "torch_cpu":
        raise FinalCenterIngestError("final-center admission requires the canonical torch_cpu backend")
    index_path = Path(index_path)
    if index_path.is_symlink() or not index_path.is_file():
        raise FinalCenterIngestError("final-center index must be a non-symlink regular file")
    try:
        index_raw = index_path.read_bytes()
        index = json.loads(index_raw.decode("utf-8"))
    except Exception as exc:
        raise FinalCenterIngestError(f"final-center index is unreadable JSON: {exc}") from exc
    if not isinstance(index, Mapping):
        raise FinalCenterIngestError("final-center index must be a JSON object")
    _validate_final_index_header(index)
    files = initial_ingest._file_table(index)

    initial_index_pin = _index_file(index, files, "initial_ingest_index_file_id",
                                    "initial_ingest_index", "initial ingest index")
    initial_index_pin, initial_index, initial_index_raw = _load_json_pin(
        files, initial_index_pin.file_id, "initial_ingest_index", "initial ingest index")
    initial_files = initial_ingest._file_table(initial_index)
    initial_receipt_pin, initial_receipt, _ = _load_json_pin(
        files, index.get("initial_selector_receipt_file_id"),
        "initial_selector_receipt", "initial selector receipt")
    initial_receipt_self = _hash_object(initial_receipt, "receipt_sha256", "initial selector receipt")
    family_selection = _initial_family(
        Path(initial_index_pin.local_path), initial_index_pin, initial_receipt, initial_receipt_self)
    family = family_selection.family
    initial_index_self = _hash_object(initial_index, "index_sha256", "initial ingest index")
    if (index.get("scientific_input_pins_sha256") != family.input_pins_sha256
            or tuple(index.get("model_ids", [])) != tuple(family.roster.model_ids)
            or tuple(index.get("sample_ids", [])) != tuple(family.roster.sample_ids)):
        raise FinalCenterIngestError("final-center index roster/input digest differs from actual initial family")
    if _sha(initial_index_raw) != initial_index_pin.sha256:
        raise FinalCenterIngestError("initial ingest index raw pin changed during read")

    composed_pin, composed, composed_raw = _load_json_pin(
        files, index.get("composed_family_plan_file_id"),
        "composed_family_plan", "composed family plan")
    composed_self = _hash_object(composed, "plan_sha256", "composed family plan")
    composed, source_method_maps, _source_rows_by_context = _verify_composed_family_plan(
        composed, composed_raw, composed_pin, initial_index, initial_files, family_selection)
    composed_self = _hash_object(composed, "plan_sha256", "composed family plan")
    if initial_receipt.get("composed_family_plan") != _pin_link(composed_pin, composed_self):
        raise FinalCenterIngestError("initial selector receipt does not bind the exact composed family plan")
    package_pin, execution_package_doc, _package_raw = _load_json_pin(
        files, index.get("execution_package_manifest_file_id"),
        "execution_package_manifest", "point execution package manifest")
    execution_package = _validate_execution_package_manifest(
        execution_package_doc, package_pin, composed, composed_pin, composed_self, files)
    source_plans = {}
    for model_entry in initial_index["models"]:
        model_id = model_entry["model_id"]
        source_plan_pin = initial_ingest._required_file(initial_files, model_entry["plan_file_id"], "plan")
        source_rows_pin = initial_ingest._required_file(
            initial_files, model_entry["planned_rows_file_id"], "planned_rows")
        source_plan = json.loads(initial_ingest._read_pinned(source_plan_pin).decode("utf-8"))
        source_rows = initial_ingest._parse_jsonl(
            initial_ingest._read_pinned(source_rows_pin), f"source rows {model_id}")
        source_plans[model_id] = (source_plan, source_rows)
    _verify_composed_methods_and_protocol(composed, source_plans)
    methods = {item["method_id"]: item for item in composed["methods"]}
    seed_id = family_selection.spsa_seed_method_id
    seed_method = methods.get(seed_id)
    if not isinstance(seed_method, Mapping):
        raise FinalCenterIngestError("actual initialized seed method is not in the composed family")
    seed_spec = next((spec for spec in family.candidates if spec.method_id == seed_id), None)
    if seed_spec is None:
        raise FinalCenterIngestError("actual initial selector seed lacks a bound snapshot spec")
    lineage = packet_builder._seed_snapshot_lineage(family_selection, seed_method)

    npz_by_declared = _raw_npz_table(files)
    expected_candidate_npz_count = PROPOSAL_POINT_COUNT * 8 * 3 + CENTER_COUNT * 40 * 3
    if len(npz_by_declared) != expected_candidate_npz_count:
        raise FinalCenterIngestError("final index raw NPZ denominator is not 960 proposal + 480 center arrays")
    used_npz_ids: set[str] = set()

    journal_pin, journal, _journal_raw = _load_json_pin(
        files, index.get("controller_journal_file_id"), "controller_journal", "controller journal")
    journal_self = _hash_object(journal, "journal_sha256", "controller journal")
    controller, _proposal_rows, proposal_receipts = _replay_controller_journal(
        journal, family_selection, initial_index_pin, initial_index_self,
        initial_receipt_pin, initial_receipt_self, composed_pin, composed_self,
        composed, files, index["proposal_points"],
        npz_by_declared, used_npz_ids, execution_package,
    )
    proposal_entries, center_entries = _point_index_maps(index)
    # _replay_controller_journal verifies all proposals against deterministic
    # ask/tell output. Parse each point once more only through the returned
    # row receipts, then ensure every indexed proposal was included.
    if len(proposal_receipts) != PROPOSAL_POINT_COUNT:
        raise FinalCenterIngestError("ask/tell replay did not validate all forty proposal point ledgers")

    center_rows: list[selector.FitScoreRow] = []
    center_specs: list[selector.SnapshotSpec] = []
    center_receipts: list[dict[str, Any]] = []
    identity_rows = _composed_identity_rows(composed)
    for step in CENTER_STEPS:
        point_entry = center_entries[step]
        tell = controller.history[step - 1]
        center_spec = tell.full_center_spec
        if (not tell.full_center_score_due or center_spec is None
                or center_spec.source_step != step or center_spec.point_kind != "spsa_center"):
            raise FinalCenterIngestError(f"controller history has no actual tell-derived center at k={step}")
        expected_method = packet_builder._method_for_center(center_spec, seed_method,
                                                             family.input_pins_sha256, lineage)
        reconstructed_spec = _point_method_spec(
            expected_method, expected_kind="fit40_spsa_center",
            expected_id=center_spec.method_id, input_sha=family.input_pins_sha256,
            expected_seed_id=seed_id, expected_seed_step=seed_spec.source_step,
            expected_seed_theta=seed_spec.theta,
            expected_seed_artifact=seed_spec.artifact_sha256,
        )
        if reconstructed_spec != center_spec:
            raise FinalCenterIngestError(f"center point at k={step} differs from actual controller tell state")
        samples = family.roster.sample_ids
        scope = {
            "scope_kind": "spsa_full_center", "iteration": step,
            "method_id": center_spec.method_id, "point_kind": "spsa_center",
            "source_seed_method_id": seed_id, "source_step": step,
            "artifact_sha256": center_spec.artifact_sha256,
            "binding_sha256": center_spec.binding_sha256,
            "input_pins_sha256": family.input_pins_sha256,
            "sample_ids": list(samples), "speaker_ids": list(family.roster.speaker_ids),
            "run_id": f"fit40_spsa_center_k{step:02d}_{center_spec.binding_sha256[:12]}",
        }
        rows, point_receipt = _point_rows(
            point_entry, expected_kind="fit40_spsa_center", expected_id=center_spec.method_id,
            expected_samples=samples, base_plan=composed, composed_pin=composed_pin,
            composed_self_sha256=composed_self, files=files, method=expected_method,
            rows_by_context=identity_rows, input_sha=family.input_pins_sha256,
            npz_by_declared=npz_by_declared, used_npz_ids=used_npz_ids,
            expected_scope=scope, execution_package=execution_package,
        )
        center_specs.append(center_spec)
        center_rows.extend(rows)
        center_receipts.append(point_receipt)
    if tuple(spec.source_step for spec in center_specs) != CENTER_STEPS:
        raise FinalCenterIngestError("actual center point specs do not preserve the frozen k=5/10/15/20 order")
    if (len(used_npz_ids) != expected_candidate_npz_count
            or {pin.file_id for pin in npz_by_declared.values()} != used_npz_ids):
        raise FinalCenterIngestError("final proposal/center NPZ files are missing, duplicated, or unreferenced")

    reference_ids = {item.method_id for item in family.reference_specs}
    reference_rows = [row for row in family.rows if row.method_id in reference_ids]
    if (len(reference_rows) != 240 or any(row.status != "PASS" for row in reference_rows)
            or len(center_rows) != 480):
        raise FinalCenterIngestError("final selector requires 240 exact shared references and 480 passing centers")
    final_rows = [*reference_rows, *center_rows]
    final_ids = [row.row_id for row in final_rows]
    expected_method_ids = {*reference_ids, *(spec.method_id for spec in center_specs)}
    if (len(final_rows) != FINAL_ROW_COUNT or len(set(final_ids)) != FINAL_ROW_COUNT
            or {row.method_id for row in final_rows} != expected_method_ids):
        raise FinalCenterIngestError("final selector evidence is not the unique complete 240+480 score family")
    try:
        final_selection = selector.select_snapshot_family(
            final_rows, roster=family.roster, reference_specs=family.reference_specs,
            candidates=center_specs, expected_input_pins_sha256=family.input_pins_sha256,
            metric_backend="torch_cpu",
        )
    except Exception as exc:
        raise FinalCenterIngestError(f"canonical final-center selector raised: {type(exc).__name__}: {exc}") from exc
    accounting = final_selection.accounting
    if (accounting.get("unique_official_score_rows") != FINAL_ROW_COUNT
            or accounting.get("observed_unique_rows") != FINAL_ROW_COUNT
            or accounting.get("failed_score_rows") != 0
            or accounting.get("missing_score_rows") != 0
            or final_selection.metric_backend != "torch_cpu"):
        raise FinalCenterIngestError("final-center selector did not retain complete canonical 720-row accounting")
    if final_selection.metric_backend == "numpy_diagnostic" or final_selection.status == "NUMPY_DIAGNOSTIC_ONLY_NOT_ADMITTED":
        raise FinalCenterIngestError("diagnostic NumPy metrics cannot produce final-center selection evidence")

    final_index_pin = initial_ingest._PinnedFile(
        "final_center_index", "final_center_index", str(index_path.resolve()),
        str(index_path.resolve()), len(index_raw), _sha(index_raw),
    )
    return FinalCenterIngest(
        initial_selection=family_selection.selection,
        final_selection=final_selection,
        initial_ingest_index_sha256=initial_index_pin.sha256,
        initial_selector_receipt_sha256=initial_receipt_pin.sha256,
        initial_selector_receipt_self_sha256=initial_receipt_self,
        controller_journal_sha256=journal_pin.sha256,
        controller_journal_self_sha256=journal_self,
        composed_family_plan_sha256=composed_pin.sha256,
        composed_family_plan_self_sha256=composed_self,
        execution_package_manifest_sha256=package_pin.sha256,
        execution_package_manifest_self_sha256=execution_package_doc["manifest_sha256"],
        scientific_input_pins_sha256=family.input_pins_sha256,
        verified_initial_rows=INITIAL_ROW_COUNT,
        verified_proposal_rows=PROPOSAL_POINT_COUNT * 8 * 3,
        verified_center_rows=len(center_rows),
        verified_final_rows=len(final_rows),
        verified_candidate_npz_count=len(used_npz_ids),
        center_specs=tuple(center_specs),
    )


def _raw_npz_table(files: Mapping[str, initial_ingest._PinnedFile]) -> dict[str, initial_ingest._PinnedFile]:
    out: dict[str, initial_ingest._PinnedFile] = {}
    for pin in files.values():
        if pin.role != "raw_matrix_npz":
            continue
        if pin.declared_path in out:
            raise FinalCenterIngestError(f"duplicate exact raw NPZ file path mapping: {pin.declared_path}")
        out[pin.declared_path] = pin
    return out


def selection_evidence_document(result: FinalCenterIngest) -> dict[str, Any]:
    """Serialize a verified result as Fit-only selector evidence."""
    status = ("COMPLETE_FIT_ONLY_FINAL_CENTER_SELECTION_NOT_FINAL_DEV"
              if result.final_selection.status in {"ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV", "NOT_ADMITTED"}
              else "COMPLETE_FIT_ONLY_FINAL_CENTER_CONTROL_FAILED")
    document: dict[str, Any] = {
        "schema": "fit40_final_center_selector_receipt_v1",
        "status": status,
        "selection_status": result.final_selection.status,
        "metric_backend": result.final_selection.metric_backend,
        "final_dev_or_confirm_admission_performed": False,
        "initial_selection": _plain(result.initial_selection),
        "final_selection": _plain(result.final_selection),
        "initial_ingest_index_sha256": result.initial_ingest_index_sha256,
        "initial_selector_receipt_sha256": result.initial_selector_receipt_sha256,
        "initial_selector_receipt_self_sha256": result.initial_selector_receipt_self_sha256,
        "controller_journal_sha256": result.controller_journal_sha256,
        "controller_journal_self_sha256": result.controller_journal_self_sha256,
        "composed_family_plan_sha256": result.composed_family_plan_sha256,
        "composed_family_plan_self_sha256": result.composed_family_plan_self_sha256,
        "execution_package_manifest_sha256": result.execution_package_manifest_sha256,
        "execution_package_manifest_self_sha256": result.execution_package_manifest_self_sha256,
        "scientific_input_pins_sha256": result.scientific_input_pins_sha256,
        "accounting": {
            "initial_rows_verified": result.verified_initial_rows,
            "proposal_rows_verified": result.verified_proposal_rows,
            "center_rows_verified": result.verified_center_rows,
            "final_selection_rows": result.verified_final_rows,
            "unique_candidate_npz_files_verified": result.verified_candidate_npz_count,
        },
        "center_specs": [_plain(spec) for spec in result.center_specs],
    }
    document["receipt_sha256"] = _sha(_canonical(document))
    return document


def write_selection_evidence_exclusive(path: str | Path, result: FinalCenterIngest) -> dict[str, Any]:
    """Create, never replace, the final selector receipt after successful ingest."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite final-center evidence: {out}")
    document = selection_evidence_document(result)
    raw = (json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")
    with out.open("xb") as stream:
        stream.write(raw)
        stream.flush()
    return {"path": str(out.resolve()), "bytes": len(raw), "sha256": _sha(raw),
            "self_sha256": document["receipt_sha256"]}


__all__ = [
    "CENTER_STEPS", "CONTROLLER_JOURNAL_SCHEMA", "FinalCenterIngest",
    "FinalCenterIngestError", "SCHEMA", "controller_state",
    "ingest_final_center_family", "selection_evidence_document",
    "write_selection_evidence_exclusive",
]
