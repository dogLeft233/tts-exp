#!/usr/bin/env python3
"""Create and execute the four tell-derived full Fit SPSA centers.

This module never starts a new SPSA scientific run or mutates the optimizer.
It replays a complete durable session in memory, accepts only a sealed
controller journal, then prepares one 120-row center plan and can serially
run the frozen 40-row/model child path.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import hashlib
import os
import subprocess
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.experiments.cross_tfg_audio_head_20261003 import (  # noqa: E402
    fit40_durable_spsa_session_attempt04 as durable,
    fit40_spsa_point_dispatcher_attempt09 as dispatcher_source,
    fit_only_final_center_stagea_ingest_attempt05 as final_ingest,
    fit_only_spsa_selection_attempt06 as selector,
    prepare_fit40_spsa_stagea_packets_attempt05 as packet_builder,
)

MODEL_ORDER = tuple(dispatcher_source.MODEL_ORDER)
CENTER_STEPS = (5, 10, 15, 20)
TERMINAL_SCHEMA = "fit40_spsa_center_dispatch_terminal_v1"
INTENT_SCHEMA = "fit40_spsa_center_dispatch_intent_v1"


class CenterDispatchError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    return _sha(_canonical({key: item for key, item in value.items() if key != field}))


def _read_json(path: Path, label: str, max_bytes: int = dispatcher_source.MAX_JSON_BYTES) -> Tuple[dict, bytes]:
    _resolved, raw = dispatcher_source._safe_file(str(path), label, max_bytes=max_bytes)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CenterDispatchError(label + " is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CenterDispatchError(label + " JSON root must be an object")
    return value, raw


def _plain(value: Any) -> Any:
    return durable._plain(value)


def center_spec_from_verified_history(session: Any, journal: Mapping[str, Any],
                                      step: int) -> selector.SnapshotSpec:
    """Return the precise SnapshotSpec emitted by the replayed tell at step."""
    if type(step) is not int or step not in CENTER_STEPS:
        raise CenterDispatchError("center step must be 5, 10, 15, or 20")
    if (journal.get("schema") != durable.JOURNAL_SCHEMA
            or journal.get("status") != "COMPLETE_20_ASK_TELL_PAIRS"
            or journal.get("metric_backend") != "torch_cpu"
            or not isinstance(journal.get("entries"), list)
            or len(journal["entries"]) != 20
            or journal.get("journal_sha256") != _self_hash(journal, "journal_sha256")):
        raise CenterDispatchError("center requires a self-valid complete canonical journal")
    controller = getattr(session, "controller", None)
    history = getattr(controller, "history", None)
    if (not isinstance(history, list) or len(history) != 20
            or getattr(controller, "iteration", None) != 20
            or getattr(controller, "_pending", None) is not None
            or getattr(controller, "halted", True)):
        raise CenterDispatchError("session controller is not a clean replay of twenty updates")
    tell = history[step - 1]
    center = getattr(tell, "full_center_spec", None)
    if (getattr(tell, "status", None) != "UPDATED"
            or getattr(tell, "iteration", None) != step
            or getattr(tell, "full_center_score_due", None) is not True
            or not isinstance(center, selector.SnapshotSpec)
            or center.point_kind != "spsa_center"
            or center.source_step != step
            or center.method_id != "fit40_d0_spsa_center_k%02d" % step):
        raise CenterDispatchError("requested center was not emitted by the corresponding tell")
    tell_doc = journal["entries"][step - 1].get("tell_result")
    if (not isinstance(tell_doc, Mapping) or tell_doc.get("status") != "UPDATED"
            or tell_doc.get("iteration") != step
            or tell_doc.get("full_center_spec") != _plain(center)):
        raise CenterDispatchError("sealed journal center differs from the replayed tell object")
    return center


def verify_sealed_session_journal(session: Any, journal_path: str | Path) -> Tuple[dict, dict]:
    path = Path(journal_path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise CenterDispatchError("controller journal must be an absolute regular file")
    journal, raw = _read_json(path, "controller journal", 64 * 1024 * 1024)
    if (journal.get("journal_sha256") != _self_hash(journal, "journal_sha256")
            or journal != session.controller_journal_document()
            or journal.get("status") != "COMPLETE_20_ASK_TELL_PAIRS"):
        raise CenterDispatchError("disk journal is not the exact complete replayed session")
    return journal, {"path": str(path.resolve(strict=True)), "bytes": len(raw),
                     "sha256": _sha(raw), "self_sha256": journal["journal_sha256"]}


def prepare_tell_center_packet(*, session: Any, journal_path: str | Path,
                               source_step: int, output_dir: str | Path) -> dict:
    journal, journal_pin = verify_sealed_session_journal(session, journal_path)
    center = center_spec_from_verified_history(session, journal, source_step)
    inputs = getattr(session, "inputs", None)
    selection = getattr(inputs, "family_selection", None)
    composed_pin = getattr(inputs, "composed_pin", None)
    if selection is None or composed_pin is None:
        raise CenterDispatchError("session lacks verified family selection or composed plan pin")
    packet = packet_builder.prepare_center_packet(
        base_plan_path=composed_pin.declared_path,
        expected_base_plan_sha256=composed_pin.sha256,
        family_selection=selection, center=center, output_dir=output_dir)
    if packet.get("planned_row_total") != 120 or packet.get("scope", {}).get("source_step") != source_step:
        raise CenterDispatchError("composer did not emit the full 120-row tell-derived center")
    return {"center_spec": center, "journal_pin": journal_pin, "packet": packet}


def validate_center_plan_shape(center: selector.SnapshotSpec, scope: Mapping[str, Any],
                               plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    """Pure packet-shape gate, testable without model/runtime initialization."""
    if (not isinstance(center, selector.SnapshotSpec) or center.point_kind != "spsa_center"
            or center.source_step not in CENTER_STEPS):
        raise CenterDispatchError("plan requires an actual predeclared center spec")
    if (scope.get("scope_kind") != "spsa_full_center" or scope.get("point_kind") != "spsa_center"
            or scope.get("source_step") != center.source_step or scope.get("method_id") != center.method_id
            or scope.get("artifact_sha256") != center.artifact_sha256
            or scope.get("binding_sha256") != center.binding_sha256):
        raise CenterDispatchError("center scope differs from its tell-generated point")
    if (plan.get("run_id") != scope.get("run_id") or plan.get("rows") != list(rows)
            or len(rows) != 120):
        raise CenterDispatchError("center plan does not retain its exact 120-row denominator")
    methods = plan.get("methods")
    if not isinstance(methods, list) or len(methods) != 1:
        raise CenterDispatchError("center plan must have exactly one method")
    method = methods[0]
    config = method.get("config") if isinstance(method, Mapping) else None
    if (method.get("method_id") != center.method_id or not isinstance(config, Mapping)
            or config.get("method") != "fit40_spsa_center"
            or config.get("schema") != "fit40_spsa_center_transform_v1"
            or config.get("point_kind") != "spsa_center"
            or config.get("source_step") != center.source_step
            or config.get("theta") != list(center.theta)
            or config.get("artifact_sha256") != center.artifact_sha256
            or config.get("binding_sha256") != center.binding_sha256):
        raise CenterDispatchError("center config differs from its actual tell theta/artifact/binding")
    models = plan.get("models")
    if not isinstance(models, list) or [item.get("model_id") for item in models] != list(MODEL_ORDER):
        raise CenterDispatchError("center plan changes the frozen model order")
    samples = list(dict.fromkeys(str(row.get("sample_id")) for row in rows))
    expected_contexts = [(sample, model) for sample in samples for model in MODEL_ORDER]
    observed_contexts = [(row.get("sample_id"), row.get("model_id")) for row in rows]
    if (len(samples) != 40 or observed_contexts != expected_contexts
            or any(row.get("split") != "fit" or row.get("method_id") != center.method_id
                   or row.get("score_audio") != row.get("natural_pcm") for row in rows)):
        raise CenterDispatchError("center rows alter Fit roster/order or original scoring audio")


def resume_verified_session(session_dir: str | Path) -> Any:
    root = Path(session_dir)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise CenterDispatchError("session root must be an existing absolute real directory")
    try:
        session = durable.DurableSPSASession.resume(session_dir=root)
    except Exception as exc:
        raise CenterDispatchError("complete initial/session replay failed: " + str(exc)) from exc
    if session.controller_journal_document().get("status") != "COMPLETE_20_ASK_TELL_PAIRS":
        raise CenterDispatchError("session does not contain a complete 20-pair controller history")
    return session


def _session_pins(session: Any) -> Tuple[dict, dict, dict]:
    inputs = session.inputs
    return (
        {"path": inputs.composed_pin.declared_path, "bytes": inputs.composed_pin.byte_count,
         "sha256": inputs.composed_pin.sha256, "self_sha256": inputs.composed_self},
        {"path": inputs.initial_receipt_pin.declared_path, "bytes": inputs.initial_receipt_pin.byte_count,
         "sha256": inputs.initial_receipt_pin.sha256, "self_sha256": inputs.initial_receipt_self},
        {"path": inputs.package_pin.declared_path, "bytes": inputs.package_pin.byte_count,
         "sha256": inputs.package_pin.sha256, "self_sha256": inputs.package_self},
    )


class _CenterIdentity:
    """Minimal compatibility carrier for the frozen serial block launcher."""

    def __init__(self, center: selector.SnapshotSpec) -> None:
        self.method_id = center.method_id
        # The point04 launcher uses these fields only in its parent launch-intent
        # record. Child status/manifest identity is reconstructed from the plan.
        self.proposal_id = center.method_id
        self.pair_id = None
        self.iteration = center.source_step
        self.side = "center"


class _CenterBlockDispatcher(dispatcher_source.PointBlockDispatcher):
    """Reuse candidate preparation and frozen child plumbing for each center."""

    def _collect_block(self, proposal: Any, side_plan: Any, model_id: str,
                       block_root: Path, *, extra_pins: Optional[Mapping[str, Mapping[str, Any]]] = None) -> dict:
        entry = self.model_entries[model_id]
        output = block_root / "stagea_output"
        status_path = output / "run_status.json"
        manifest_path = output / "result_manifest.json"
        ledger_path = output / "results.jsonl"
        receipt_path = block_root / "child_execution_receipt.json"
        status_actual, _status_raw, status = dispatcher_source._read_json_pin(
            dispatcher_source.file_pin(status_path, self_field="status_sha256"),
            "center run status", "status_sha256")
        manifest_actual, _manifest_raw, manifest = dispatcher_source._read_json_pin(
            dispatcher_source.file_pin(manifest_path, self_field="manifest_sha256"),
            "center result manifest", "manifest_sha256")
        ledger_actual = dispatcher_source.file_pin(ledger_path)
        records = dispatcher_source._parse_jsonl(Path(ledger_actual["path"]).read_bytes(), "center results")
        receipt_actual, _receipt_raw, child = dispatcher_source._read_json_pin(
            dispatcher_source.file_pin(receipt_path, self_field="receipt_sha256"),
            "center child receipt", "receipt_sha256")
        process_started_path = block_root / "process_started.json"
        process_started_actual, _process_started_raw, process_started = dispatcher_source._read_json_pin(
            dispatcher_source.file_pin(process_started_path, self_field="started_sha256"),
            "center parent start", "started_sha256")
        process_terminal_path = block_root / "process_terminal.json"
        process_terminal_actual, _process_terminal_raw, process_terminal = dispatcher_source._read_json_pin(
            dispatcher_source.file_pin(process_terminal_path, self_field="process_terminal_sha256"),
            "center parent terminal", "process_terminal_sha256")
        request_actual, _request_raw, request = dispatcher_source._read_json_pin(
            dispatcher_source.file_pin(block_root / "block_request.json", self_field="request_sha256"),
            "center model block request", "request_sha256")
        planned_rows = [dict(row) for row in side_plan.planned_rows if row.get("model_id") == model_id]
        row_ids = [row["row_id"] for row in planned_rows]
        candidate_rows_pin = request.get("candidate_rows_pin")
        if not isinstance(candidate_rows_pin, Mapping):
            raise CenterDispatchError("center model block request omits shared candidate rows")
        candidate_actual, _candidate_raw, candidate_packet = dispatcher_source._read_json_pin(
            candidate_rows_pin, "center shared candidate rows", "candidate_rows_sha256")
        candidate_scope = self.runner.validate_point_scope(
            family_plan_pin=self.family_pin, point_plan_pin=side_plan.plan_pin,
            initial_selector_receipt_pin=self.initial_receipt_pin,
            execution_package_manifest_pin=self.package_pin, model_id=model_id)
        self.runner._validate_shared_candidate_rows(candidate_scope, candidate_actual)
        if (candidate_packet.get("model_id") != model_id
                or candidate_packet.get("point_kind") != "spsa_center"
                or candidate_packet.get("point_id") != proposal.method_id
                or candidate_packet.get("method_id") != proposal.method_id
                or candidate_packet.get("source_step") != proposal.iteration
                or candidate_packet.get("proposal_id") is not None
                or candidate_packet.get("pair_id") is not None
                or candidate_packet.get("side") is not None
                or candidate_packet.get("iteration") is not None
                or [row.get("row_id") for row in candidate_packet.get("rows", [])] != row_ids):
            raise CenterDispatchError("center candidate sidecar differs from its exact 40-row point/model scope")
        counts = {name: sum(row.get("status") == name for row in records)
                  for name in ("PASS", "FAILED", "INCOMPLETE")}
        context = {
            "run_id": side_plan.plan["run_id"], "point_kind": "spsa_center",
            "point_id": proposal.method_id, "method_id": proposal.method_id,
            "source_step": proposal.iteration, "model_id": model_id, "block_index": 0,
            "family_plan_pin": self.family_pin, "point_plan_pin": side_plan.plan_pin,
            "initial_selector_receipt_pin": self.initial_receipt_pin,
            "initial_ingest_index_pin": self.initial_receipt.get("initial_ingest_index"),
            "execution_package_manifest_pin": self.package_pin,
            "candidate_rows_pin": candidate_actual,
            "row_ids": row_ids, "planned_rows": 40,
        }
        if (not all(isinstance(value, Mapping) for value in (status, manifest, child))
                or len(records) != 40 or [row.get("row_id") for row in records] != row_ids
                or status.get("schema") != self.runner.POINT_STATUS_SCHEMA
                or manifest.get("schema") != self.runner.POINT_MANIFEST_SCHEMA
                or child.get("schema") != self.runner.CHILD_RECEIPT_SCHEMA):
            raise CenterDispatchError("model center output does not preserve the exact forty-row terminal schema")
        # The immutable point04 status/manifest carry the full runner context.
        # The actual parent-captured child receipt intentionally has a narrower
        # contract: it binds the planned-rows *file pin* and ordered row IDs,
        # but does not duplicate the numeric denominator or selector/ingest
        # pins. Keep those fields checked on status/manifest and verify the
        # child receipt's own documented identity below.
        for document in (status, manifest):
            for key, expected in context.items():
                if document.get(key) != expected:
                    raise CenterDispatchError("center child lineage differs at " + key)
        child_identity = {
            "run_id": context["run_id"], "point_kind": "spsa_center",
            "point_id": proposal.method_id, "method_id": proposal.method_id,
            "model_id": model_id, "block_index": 0,
            "family_plan_pin": self.family_pin, "point_plan_pin": side_plan.plan_pin,
            "execution_package_manifest_pin": self.package_pin,
            "candidate_rows_pin": candidate_actual,
            "row_ids": row_ids,
        }
        for key, expected in child_identity.items():
            if child.get(key) != expected:
                raise CenterDispatchError("center child receipt identity differs at " + key)
        if (len(child.get("row_ids", [])) != 40
                or child.get("planned_rows_pin") != status.get("planned_rows_pin")
                or child.get("proposal_id") is not None or child.get("pair_id") is not None
                or child.get("iteration") != proposal.iteration
                or child.get("side") is not None
                or child.get("source_step") != proposal.iteration
                or ("planned_rows" in child and child.get("planned_rows") != 40)):
            raise CenterDispatchError("center child receipt does not bind the exact 40-row center identity")
        for document in (status, manifest, child):
            if (document.get("proposal_id") is not None or document.get("pair_id") is not None
                    or document.get("iteration") != proposal.iteration
                    or document.get("side") is not None):
                raise CenterDispatchError("center terminal was relabeled with proposal identity")
        status_self = status.get("status_sha256")
        manifest_self = manifest.get("manifest_sha256")
        receipt_self = child.get("receipt_sha256")
        if (status_self != _self_hash(status, "status_sha256")
                or manifest_self != _self_hash(manifest, "manifest_sha256")
                or receipt_self != _self_hash(child, "receipt_sha256")):
            raise CenterDispatchError("center child terminal self hash mismatch")
        status_link = {**status_actual}
        manifest_link = {**manifest_actual}
        ledger_link = dict(ledger_actual)
        expected_status_counts = counts
        if (status.get("status") != manifest.get("status")
                or status.get("status_counts") != expected_status_counts
                or manifest.get("status_counts") != expected_status_counts
                or status.get("row_ids") != row_ids or manifest.get("row_ids") != row_ids
                or status.get("result_manifest_pin") != manifest_link
                or status.get("results_jsonl_pin") != ledger_link
                or manifest.get("results_jsonl_pin") != ledger_link
                or child.get("run_status_pin") != status_link
                or child.get("result_manifest_pin") != manifest_link
                or child.get("results_jsonl_pin") != ledger_link):
            raise CenterDispatchError("center status, manifest, receipt, or ledger links disagree")
        start_utc = dispatcher_source._parse_utc_timestamp(
            process_started.get("started_utc"), "center parent start UTC")
        end_utc = dispatcher_source._parse_utc_timestamp(
            process_terminal.get("ended_utc"), "center parent end UTC")
        child_start = dispatcher_source._parse_utc_timestamp(status.get("started_utc"), "center child start UTC")
        child_end = dispatcher_source._parse_utc_timestamp(status.get("ended_utc"), "center child end UTC")
        terminalized = dispatcher_source._parse_utc_timestamp(
            manifest.get("terminalized_utc"), "center terminalized UTC")
        if (process_started.get("schema") != dispatcher_source.PROCESS_START_SCHEMA
                or process_started.get("status") != "STARTED"
                or process_started.get("model_id") != model_id
                or process_started.get("request_pin") != request_actual
                or process_terminal.get("schema") != dispatcher_source.PROCESS_END_SCHEMA
                or process_terminal.get("status") != "TERMINAL"
                or process_terminal.get("process_started_pin") != process_started_actual
                or process_terminal.get("pid") != process_started.get("pid")
                or process_terminal.get("start_ticks") != process_started.get("start_ticks")
                or process_terminal.get("timed_out") is not False
                or type(process_terminal.get("child_exit_code")) is not int
                or (child.get("child_pid"), child.get("child_start_ticks")) !=
                   (process_terminal.get("pid"), process_terminal.get("start_ticks"))
                or child.get("child_exit_code") != process_terminal.get("child_exit_code")
                or child.get("started_utc") != process_started.get("started_utc")
                or child.get("ended_utc") != process_terminal.get("ended_utc")
                or status.get("completed_rows") != counts["PASS"] + counts["FAILED"]
                or manifest.get("completed_rows") != counts["PASS"] + counts["FAILED"]
                or status.get("failed_rows") != counts["FAILED"]
                or manifest.get("failed_rows") != counts["FAILED"]
                or status.get("incomplete_rows") != counts["INCOMPLETE"]
                or manifest.get("incomplete_rows") != counts["INCOMPLETE"]
                or end_utc < start_utc or child_start < start_utc or child_end > end_utc
                or child_end < terminalized):
            raise CenterDispatchError("center process receipt does not contain the immutable child interval")
        child_closure = {
            "launcher_pin": entry["launcher_pin"], "runner_pin": entry["runner_pin"],
            "adapter_pin": entry["adapter_pin"], "runtime_binding_pin": entry["runtime_binding_pin"],
            "source_pins": entry["source_pins"], "point_runner_pin": self.point_runner_pin,
            "shared_application_helper_pin": self.helper_pin,
            "execution_package_manifest_pin": self.package_pin,
        }
        if any(child.get(field) != expected for field, expected in child_closure.items()):
            raise CenterDispatchError("center child receipt source closure differs from package")
        for document in (status, manifest):
            if (document.get("source_pins") != entry["source_pins"]
                    or document.get("execution_package_manifest_pin") != self.package_pin):
                raise CenterDispatchError("center child status/manifest source closure differs from package")
        adapter_validation = status.get("adapter_validation_pins")
        if not isinstance(adapter_validation, Mapping):
            raise CenterDispatchError("center model omitted adapter validation source pins")
        for role, pin in adapter_validation.items():
            actual, _raw, _unused = dispatcher_source._read_pin(pin, "center adapter source " + str(role))
            if any(actual.get(key) != pin.get(key) for key in ("path", "bytes", "sha256")):
                raise CenterDispatchError("center adapter source changed after execution: " + str(role))

        files: dict[str, dict] = {}
        candidate_rows_entry = dispatcher_source._add_file(
            files, Path(candidate_actual["path"]), "shared_candidate_rows",
            self_field="candidate_rows_sha256")
        if (candidate_rows_entry["bytes"] != candidate_actual["bytes"]
                or candidate_rows_entry["sha256"] != candidate_actual["sha256"]):
            raise CenterDispatchError("center candidate sidecar changed before parent collection")
        shared_output_file_ids_by_sample: dict[str, str] = {}
        shared_application_file_ids_by_sample: dict[str, str] = {}
        origin_application_file_ids_by_sample: dict[str, str] = {}
        projection_file_ids_by_row: dict[str, str] = {}
        for candidate in candidate_packet["rows"]:
            binding = candidate.get("spsa_binding") if isinstance(candidate, Mapping) else None
            shared_pin = binding.get("shared_application_receipt") if isinstance(binding, Mapping) else None
            projection_pin = binding.get("model_projection_receipt") if isinstance(binding, Mapping) else None
            if not isinstance(shared_pin, Mapping) or not isinstance(projection_pin, Mapping):
                raise CenterDispatchError("center candidate lacks shared/projection receipts")
            shared_actual, _shared_raw, shared_packet = dispatcher_source._read_json_pin(
                shared_pin, "center shared application receipt", "self_sha256")
            projection_actual, _projection_raw, _projection_packet = dispatcher_source._read_json_pin(
                projection_pin, "center model projection receipt", "self_sha256")
            sample_id = str(candidate.get("sample_id"))
            row_id = str(candidate.get("row_id"))
            output_ref = shared_packet.get("output_generator_drive_wav")
            origin_pin = shared_packet.get("origin_v05_application_receipt")
            if not isinstance(output_ref, Mapping) or not isinstance(origin_pin, Mapping):
                raise CenterDispatchError("center shared receipt lacks its exact output or v05 origin")
            shared_entry = dispatcher_source._add_file(
                files, Path(shared_actual["path"]), "shared_application_receipt", self_field="self_sha256")
            origin_actual, _origin_raw, _origin_packet = dispatcher_source._read_json_pin(
                origin_pin, "center shared v05 origin receipt", "self_sha256")
            origin_entry = dispatcher_source._add_file(
                files, Path(origin_actual["path"]), "shared_origin_v05_application_receipt", self_field="self_sha256")
            output_entry = dispatcher_source._add_file(
                files, Path(str(output_ref.get("path", ""))), "shared_generator_drive_wav")
            projection_entry = dispatcher_source._add_file(
                files, Path(projection_actual["path"]), "model_projection_receipt", self_field="self_sha256")
            if (shared_entry["bytes"] != shared_actual["bytes"]
                    or shared_entry["sha256"] != shared_actual["sha256"]
                    or origin_entry["bytes"] != origin_actual["bytes"]
                    or origin_entry["sha256"] != origin_actual["sha256"]
                    or projection_entry["bytes"] != projection_actual["bytes"]
                    or projection_entry["sha256"] != projection_actual["sha256"]
                    or output_entry["bytes"] != output_ref.get("bytes")
                    or output_entry["sha256"] != output_ref.get("sha256")):
                raise CenterDispatchError("center shared waveform/projection changed before parent collection")
            for mapping, key, value, label in (
                (shared_output_file_ids_by_sample, sample_id, output_entry["file_id"], "shared output"),
                (shared_application_file_ids_by_sample, sample_id, shared_entry["file_id"], "shared receipt"),
                (origin_application_file_ids_by_sample, sample_id, origin_entry["file_id"], "origin receipt"),
                (projection_file_ids_by_row, row_id, projection_entry["file_id"], "model projection"),
            ):
                if key in mapping and mapping[key] != value:
                    raise CenterDispatchError("one center sample/row maps to two " + label + " files")
                mapping[key] = value
        dispatcher_source._add_file(files, Path(self.family_pin["path"]), "composed_family_plan",
                                    self_field="plan_sha256")
        dispatcher_source._add_file(files, Path(self.initial_receipt_pin["path"]), "initial_selector_receipt",
                                    self_field="receipt_sha256")
        ingest_pin = self.initial_receipt.get("initial_ingest_index")
        if not isinstance(ingest_pin, Mapping):
            raise CenterDispatchError("initial selector receipt omits its exact initial ingest index")
        dispatcher_source._add_file(files, Path(ingest_pin["path"]), "initial_stagea_ingest_index",
                                    self_field="index_sha256")
        dispatcher_source._add_file(files, Path(self.package_pin["path"]), "execution_package_manifest",
                                    self_field="manifest_sha256")
        dispatcher_source._add_file(files, Path(self.point_runner_pin["path"]), "point_execution_runner")
        dispatcher_source._add_file(files, Path(self.helper_pin["path"]), "shared_application_helper")
        actual_dispatcher = dispatcher_source.file_pin(Path(__file__).resolve(strict=True))
        if actual_dispatcher != self.center_dispatcher_pin:
            raise CenterDispatchError("center dispatcher source changed during model execution")
        dispatcher_source._add_file(files, Path(self.center_dispatcher_pin["path"]), "center_dispatcher")
        for role, pin in (("common_model_child_launcher", entry["launcher_pin"]),
                          ("model_stagea_runner", entry["runner_pin"]),
                          ("model_spsa_adapter", entry["adapter_pin"])):
            dispatcher_source._add_file(files, Path(pin["path"]), role)
        dispatcher_source._add_file(files, Path(entry["runtime_binding_pin"]["path"]),
                                    "model_runtime_binding", self_field="binding_sha256")
        for role, pin in entry["source_pins"].items():
            dispatcher_source._add_file(files, Path(pin["path"]), "model_source:" + model_id + ":" + role,
                                        self_field="profile_sha256" if role == "launch_profile" else None)
        dispatcher_source._add_file(files, Path(side_plan.packet_index_pin["path"]), "center_packet_index",
                                    self_field="packet_sha256")
        dispatcher_source._add_file(files, Path(side_plan.plan_pin["path"]), "plan", self_field="plan_sha256")
        global_rows = dispatcher_source._add_file(files, Path(side_plan.planned_rows_pin["path"]), "planned_rows")
        model_rows_entry = dispatcher_source._add_file(
            files, output / "planned_rows.jsonl", "model_planned_rows")
        status_entry = dispatcher_source._add_file(files, status_path, "run_status", self_field="status_sha256")
        manifest_entry = dispatcher_source._add_file(files, manifest_path, "result_manifest",
                                                     self_field="manifest_sha256")
        ledger_entry = dispatcher_source._add_file(files, ledger_path, "results_jsonl")
        child_entry = dispatcher_source._add_file(files, receipt_path, "child_execution_receipt",
                                                  self_field="receipt_sha256")
        frame_map_ids: dict[str, str] = {}
        raw_npz_by_row = {}
        raw_npz_manifest = manifest.get("raw_score_npz_by_row")
        if not isinstance(raw_npz_manifest, Mapping) or set(raw_npz_manifest) != {
                record.get("row_id") for record in records if record.get("status") == "PASS"}:
            raise CenterDispatchError("center manifest NPZ map differs from successful result rows")
        for record in records:
            if record.get("status") != "PASS":
                continue
            raw_ref = record.get("score", {}).get("raw_matrix_npz")
            if not isinstance(raw_ref, Mapping):
                raise CenterDispatchError("passing center result lacks official seven-array NPZ pin")
            npz_entry = dispatcher_source._add_file(files, Path(raw_ref["path"]), "raw_matrix_npz")
            # Stage-A result rows pin NPZ raw SHA in `score.raw_matrix_npz`,
            # while the point04 manifest carries the measured byte count.
            # Do not require a non-existent `bytes` field on the score row.
            if (npz_entry["declared_path"] != raw_ref.get("path")
                    or npz_entry["sha256"] != raw_ref.get("file_sha256")):
                raise CenterDispatchError("center NPZ changed after the child terminal receipt")
            expected_manifest_npz = {
                "row_id": record["row_id"], "status": "PASS",
                "raw_matrix_npz": {"path": npz_entry["declared_path"], "bytes": npz_entry["bytes"],
                                   "sha256": npz_entry["sha256"], "array_refs": raw_ref.get("arrays")},
            }
            if raw_npz_manifest.get(record["row_id"]) != expected_manifest_npz:
                raise CenterDispatchError("center manifest/raw result NPZ link mismatch")
            raw_npz_by_row[record["row_id"]] = {
                "path": npz_entry["declared_path"], "bytes": npz_entry["bytes"],
                "sha256": npz_entry["sha256"], "array_refs": raw_ref.get("arrays"),
            }
            frame_map_ids.update(dispatcher_source._add_official_source_frame_map(files, record))
        for role, pin in adapter_validation.items():
            dispatcher_source._add_file(files, Path(pin["path"]),
                                        "adapter_validation:" + model_id + ":" + str(role))
        for name, role, self_field in (
            ("block_request.json", "model_block_request", "request_sha256"),
            ("stdout.log", "model_block_stdout", None), ("stderr.log", "model_block_stderr", None),
            ("model_launch_intent.json", "model_block_launch_intent", "intent_sha256"),
            ("process_started.json", "model_process_started", "started_sha256"),
            ("process_terminal.json", "model_process_terminal", "process_terminal_sha256"),
        ):
            path = block_root / name
            if path.is_file() and not path.is_symlink():
                dispatcher_source._safe_file(str(path), name,
                    max_bytes=dispatcher_source.MAX_LOG_BYTES if path.suffix == ".log" else dispatcher_source.MAX_JSON_BYTES)
                dispatcher_source._add_file(files, path, role, self_field=self_field)
        status_counts = dict(counts)
        terminal_status = status.get("status")
        expected_terminal = ("COMPLETE" if counts == {"PASS": 40, "FAILED": 0, "INCOMPLETE": 0}
                             else "TERMINAL_WITH_FAILURES")
        if terminal_status != expected_terminal:
            raise CenterDispatchError("center terminal status differs from its complete forty-row counts")
        block = {
            "model_id": model_id, "block_index": 0, "row_ids": row_ids,
            "candidate_rows_file_id": candidate_rows_entry["file_id"],
            "candidate_rows_pin": candidate_actual,
            "planned_rows_file_id": model_rows_entry["file_id"],
            "run_status_file_id": status_entry["file_id"],
            "result_manifest_file_id": manifest_entry["file_id"],
            "results_file_id": ledger_entry["file_id"],
            "child_execution_receipt_file_id": child_entry["file_id"],
            "launcher_pin": entry["launcher_pin"], "runner_pin": entry["runner_pin"],
            "adapter_pin": entry["adapter_pin"], "runtime_binding_pin": entry["runtime_binding_pin"],
            "source_pins": entry["source_pins"],
            "execution_package_manifest_pin": self.package_pin,
            "point_runner_pin": self.point_runner_pin,
            "shared_application_helper_pin": self.helper_pin,
            "official_source_frame_map_file_ids": frame_map_ids,
            "shared_output_file_ids_by_sample": shared_output_file_ids_by_sample,
            "shared_application_file_ids_by_sample": shared_application_file_ids_by_sample,
            "origin_v05_application_file_ids_by_sample": origin_application_file_ids_by_sample,
            "model_projection_file_ids_by_row": projection_file_ids_by_row,
            "terminal_status": terminal_status, "status_counts": status_counts,
        }
        if extra_pins:
            for name, pin in extra_pins.items():
                role = {"request": "model_block_request", "process_started": "model_process_started",
                        "process_terminal": "model_process_terminal", "child_receipt": "child_execution_receipt"}[name]
                self_field = {"request": "request_sha256", "process_started": "started_sha256",
                              "process_terminal": "process_terminal_sha256",
                              "child_receipt": "receipt_sha256"}[name]
                dispatcher_source._add_file(files, Path(pin["path"]), role, self_field=self_field)
        return {"block": block, "files": files, "records": records,
                "raw_npz_by_row": raw_npz_by_row, "frame_map_ids": frame_map_ids,
                "global_rows_file_id": global_rows["file_id"]}


class FitCenterDispatcher(_CenterBlockDispatcher):
    """Create-only, serialized full-center execution over the frozen point04 child."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.center_dispatcher_pin = dispatcher_source.file_pin(Path(__file__).resolve(strict=True))

    def bind_center_packet(self, center: selector.SnapshotSpec, packet_index_path: str | Path) -> Any:
        index_path = Path(packet_index_path)
        index_pin = dispatcher_source.file_pin(index_path, self_field="packet_sha256")
        index = dispatcher_source._json_object(index_path.read_bytes(), "center packet index")
        if (index.get("schema") != packet_builder.SCHEMA
                or index.get("status") != "PLANS_ONLY_REQUIRES_BOUND_RUNTIME_AND_EXECUTION"
                or index.get("execution_started") is not False
                or index.get("model_calls_started") is not False
                or index.get("planned_row_total") != 120
                or index.get("parent_plan") != {key: self.family_pin[key] for key in ("path", "bytes", "sha256")}):
            raise CenterDispatchError("center packet is not an untouched 120-row derivative of the exact family")
        scope = index.get("scope")
        if (not isinstance(scope, Mapping) or scope.get("point_kind") != "spsa_center"
                or scope.get("source_step") != center.source_step
                or scope.get("method_id") != center.method_id
                or scope.get("artifact_sha256") != center.artifact_sha256
                or scope.get("binding_sha256") != center.binding_sha256
                or scope.get("input_pins_sha256") != self.family.get("scientific_input_pins_sha256")):
            raise CenterDispatchError("center packet scope differs from the actual tell and family pins")
        entries = index.get("plans")
        if not isinstance(entries, list) or len(entries) != 1 or entries[0].get("label") != "center":
            raise CenterDispatchError("center packet must contain exactly one center plan")
        entry = entries[0]
        plan_pin = dict(entry["plan"])
        plan_pin["self_sha256"] = entry.get("plan_self_sha256")
        plan_actual, _plan_raw, plan = dispatcher_source._read_json_pin(
            plan_pin, "center plan", "plan_sha256")
        rows_actual, rows_raw, _unused = dispatcher_source._read_pin(
            entry["planned_rows"], "center planned rows", max_bytes=64 * 1024 * 1024)
        rows = dispatcher_source._parse_jsonl(rows_raw, "center planned rows")
        validate_center_plan_shape(center, scope, plan, rows)
        if (plan_actual["path"] != str(Path(plan_pin["path"]).resolve(strict=True))
                or rows_actual["path"] != str(Path(entry["planned_rows"]["path"]).resolve(strict=True))
                or entry.get("planned_row_count") != 120 or entry.get("run_id") != plan.get("run_id")):
            raise CenterDispatchError("center plan index file pins or declared denominator differ")
        side = dispatcher_source._PlanSide(center.method_id, "center", plan_actual,
                                           rows_actual, index_pin, plan, tuple(rows))
        self._center = center
        self._side = side
        self._packet_index_path = index_path.resolve(strict=True)
        return side

    def _center_paths(self, center: selector.SnapshotSpec) -> tuple[Path, Path, Path]:
        leaf = center.method_id
        if not leaf.startswith("fit40_d0_spsa_center_k") or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in leaf):
            raise CenterDispatchError("center method ID is not a safe frozen tell-derived leaf")
        root = self.dispatch_root / "centers"
        return root / leaf, root / leaf / "dispatch_intent.json", root / leaf / "center_terminal.json"

    def readback_center(self, center: selector.SnapshotSpec,
                        journal_pin: Mapping[str, Any]) -> dict:
        """Rebuild only from immutable child proofs; never relaunch a present leaf."""
        point_dir, intent_path, terminal_path = self._center_paths(center)
        centers_root = point_dir.parent
        if centers_root.is_symlink() or (centers_root.exists() and not centers_root.is_dir()):
            return {"disposition": "UNKNOWN", "reason": "center output parent is unsafe"}
        if not point_dir.exists() and not point_dir.is_symlink():
            return {"disposition": "ABSENT"}
        if point_dir.is_symlink() or not point_dir.is_dir() or intent_path.is_symlink() or not intent_path.is_file():
            return {"disposition": "UNKNOWN", "reason": "center leaf is present without a safe dispatch intent"}
        try:
            intent_pin, _intent_raw, intent = dispatcher_source._read_json_pin(
                dispatcher_source.file_pin(intent_path, self_field="intent_sha256"),
                "center dispatch intent", "intent_sha256")
            expected = {
                "schema": INTENT_SCHEMA, "status": "CENTER_DISPATCH_INTENT",
                "point_id": center.method_id, "method_id": center.method_id,
                "point_kind": "spsa_center", "source_step": center.source_step,
                "point_plan_pin": dict(self._side.plan_pin),
                "planned_rows_pin": {key: self._side.planned_rows_pin[key]
                                     for key in ("path", "bytes", "sha256")},
                "packet_index_pin": dict(self._side.packet_index_pin),
                "family_plan_pin": dict(self.family_pin),
                "initial_selector_receipt_pin": dict(self.initial_receipt_pin),
                "execution_package_manifest_pin": dict(self.package_pin),
                "controller_journal_pin": dict(journal_pin),
                "center_dispatcher_pin": dict(self.center_dispatcher_pin),
            }
            if (set(intent) != set(expected) | {"created_utc", "intent_sha256"}
                    or any(intent.get(key) != value for key, value in expected.items())):
                raise CenterDispatchError("center dispatch intent differs from the tell/plan/package binding")
            dispatcher_source._parse_utc_timestamp(intent.get("created_utc"), "center intent created_utc")
            fake = _CenterIdentity(center)
            gathered = []
            for model_id in MODEL_ORDER:
                block_root = point_dir / "blocks" / model_id
                receipt_path = block_root / "child_execution_receipt.json"
                if (block_root.is_symlink() or not block_root.is_dir()
                        or receipt_path.is_symlink() or not receipt_path.is_file()):
                    return {"disposition": "UNKNOWN", "reason": "center has a partial or unproven model block"}
                gathered.append(self._collect_block(fake, self._side, model_id, block_root))
            rebuilt = self._assemble_center_terminal(center, gathered, journal_pin)
            if terminal_path.exists() or terminal_path.is_symlink():
                if terminal_path.is_symlink() or not terminal_path.is_file():
                    raise CenterDispatchError("center terminal path is unsafe")
                actual, _terminal_raw, terminal = dispatcher_source._read_json_pin(
                    dispatcher_source.file_pin(terminal_path, self_field="terminal_sha256"),
                    "center dispatcher terminal", "terminal_sha256")
                if {key: value for key, value in terminal.items() if key != "terminal_sha256"} != rebuilt:
                    raise CenterDispatchError("persisted center terminal differs from immutable child proofs")
                return {"disposition": "TERMINAL", "terminal": terminal,
                        "terminal_pin": actual}
            terminal_pin = dispatcher_source._write_self_json(
                terminal_path, rebuilt, "terminal_sha256")
            terminal = {**rebuilt, "terminal_sha256": terminal_pin["self_sha256"]}
            return {"disposition": "TERMINAL", "terminal": terminal,
                    "terminal_pin": terminal_pin, "reconstructed": True}
        except Exception as exc:
            return {"disposition": "UNKNOWN", "reason": type(exc).__name__ + ": " + str(exc)}

    def execute_center(self, center: selector.SnapshotSpec, *, journal_pin: Mapping[str, Any]) -> dict:
        if getattr(self, "_center", None) != center or getattr(self, "_side", None) is None:
            raise CenterDispatchError("bind the exact center composer packet before execution")
        point_dir, intent_path, terminal_path = self._center_paths(center)
        centers_root = point_dir.parent
        if centers_root.exists() and (centers_root.is_symlink() or not centers_root.is_dir()):
            raise CenterDispatchError("center output parent path is unsafe")
        if dispatcher_source.file_pin(Path(__file__).resolve(strict=True)) != self.center_dispatcher_pin:
            raise CenterDispatchError("center dispatcher source pin changed before execution")
        prior = self.readback_center(center, journal_pin)
        if prior.get("disposition") == "TERMINAL":
            terminal = prior["terminal"]
            return {**terminal, "terminal_pin": prior["terminal_pin"],
                    "reconstructed": prior.get("reconstructed", False)}
        if prior.get("disposition") != "ABSENT":
            raise CenterDispatchError("center output is present or has an unknown prior outcome; never rerun")
        contexts: dict[str, Mapping[str, Any]] = {}
        for model_id in MODEL_ORDER:
            context = self.runner.validate_point_scope(
                family_plan_pin=self.family_pin, point_plan_pin=self._side.plan_pin,
                initial_selector_receipt_pin=self.initial_receipt_pin,
                execution_package_manifest_pin=self.package_pin, model_id=model_id)
            profile, _ = self._read_profile(model_id)
            entry = self.model_entries[model_id]
            dispatcher_source._materialize_environment(
                profile, request={"strict_receipt_dir": str(point_dir / "blocks" / model_id / "strict_receipts")},
                family_plan=self.family, runtime_pin=entry["runtime_binding_pin"])
            contexts[model_id] = context
        if not centers_root.exists():
            dispatcher_source._safe_mkdir(centers_root)
        dispatcher_source._safe_mkdir(point_dir)
        dispatcher_source._write_self_json(intent_path, {
            "schema": INTENT_SCHEMA, "status": "CENTER_DISPATCH_INTENT",
            "point_id": center.method_id, "method_id": center.method_id,
            "point_kind": "spsa_center", "source_step": center.source_step,
            "point_plan_pin": dict(self._side.plan_pin),
            "planned_rows_pin": {key: self._side.planned_rows_pin[key] for key in ("path", "bytes", "sha256")},
            "packet_index_pin": dict(self._side.packet_index_pin),
            "family_plan_pin": dict(self.family_pin),
            "initial_selector_receipt_pin": dict(self.initial_receipt_pin),
            "execution_package_manifest_pin": dict(self.package_pin),
            "controller_journal_pin": dict(journal_pin),
            "center_dispatcher_pin": dict(self.center_dispatcher_pin),
            "created_utc": dispatcher_source._utc_now(),
        }, "intent_sha256")
        prepared = self.prepare_shared_candidate_rows(
            point_plan_pin=self._side.plan_pin,
            planned_rows=self._side.planned_rows,
            output_root=point_dir / "shared_candidate_application",
            point_kind="spsa_center", point_id=center.method_id,
            method_id=center.method_id, source_step=center.source_step)
        fake = _CenterIdentity(center)
        gathered = []
        for model_id in MODEL_ORDER:
            model_dir = point_dir / "blocks" / model_id
            if not model_dir.parent.exists():
                dispatcher_source._safe_mkdir(model_dir.parent)
            dispatcher_source._safe_mkdir(model_dir)
            gathered.append(super()._run_block(
                fake, self._side, model_id, model_dir, contexts[model_id],
                prepared["candidate_rows_pins_by_model"][model_id]))
        terminal_body = self._assemble_center_terminal(center, gathered, journal_pin)
        terminal_pin = dispatcher_source._write_self_json(
            terminal_path, terminal_body, "terminal_sha256")
        return {**terminal_body, "terminal_pin": terminal_pin}

    def _assemble_center_terminal(self, center: selector.SnapshotSpec,
                                  gathered: Sequence[Mapping[str, Any]],
                                  journal_pin: Mapping[str, Any]) -> dict:
        if len(gathered) != 3 or [item["block"]["model_id"] for item in gathered] != list(MODEL_ORDER):
            raise CenterDispatchError("center requires the exact ordered three-model terminal set")
        merged: dict[str, dict] = {}
        raw_npz: dict[str, dict] = {}
        frame_map_ids: dict[str, str] = {}
        for item in gathered:
            block = item["block"]
            for fid, entry in item["files"].items():
                prior = merged.get(fid)
                if prior is not None and prior != entry:
                    raise CenterDispatchError("center file ID collision")
                if prior is None:
                    for old in merged.values():
                        if (old["role"], old["declared_path"]) == (entry["role"], entry["declared_path"]):
                            raise CenterDispatchError("center file table repeats a role/path")
                    merged[fid] = dict(entry)
            if set(raw_npz).intersection(item["raw_npz_by_row"]):
                raise CenterDispatchError("center repeats a candidate NPZ row")
            raw_npz.update(item["raw_npz_by_row"])
            if set(frame_map_ids).intersection(item["frame_map_ids"]):
                raise CenterDispatchError("center repeats a candidate official frame map")
            frame_map_ids.update(item["frame_map_ids"])
        plan_entry = dispatcher_source._add_file(
            merged, Path(self._side.plan_pin["path"]), "plan", self_field="plan_sha256")
        rows_entry = dispatcher_source._add_file(
            merged, Path(self._side.planned_rows_pin["path"]), "planned_rows")
        packet_entry = dispatcher_source._add_file(
            merged, self._packet_index_path, "center_packet_index", self_field="packet_sha256")
        journal_path = Path(str(journal_pin.get("path", "")))
        actual_journal = dispatcher_source.file_pin(journal_path, self_field="journal_sha256")
        if actual_journal != dict(journal_pin):
            raise CenterDispatchError("controller journal changed since the tell-derived center was verified")
        journal_entry = dispatcher_source._add_file(merged, journal_path, "controller_journal",
                                                    self_field="journal_sha256")
        counts = {name: sum(item["block"]["status_counts"][name] for item in gathered)
                  for name in ("PASS", "FAILED", "INCOMPLETE")}
        if sum(counts.values()) != 120 or len(raw_npz) != counts["PASS"] or set(frame_map_ids) != set(raw_npz):
            raise CenterDispatchError("center aggregates do not preserve full 120-row/frame-map denominators")
        status = ("COMPLETE" if counts == {"PASS": 120, "FAILED": 0, "INCOMPLETE": 0}
                  else ("INCOMPLETE" if counts["INCOMPLETE"] else "COMPLETE_WITH_FAILURES"))
        ingest = self.initial_receipt.get("initial_ingest_index")
        if not isinstance(ingest, Mapping):
            raise CenterDispatchError("initial selector receipt lacks its initial ingest index pin")
        point_entry = {
            "run_id": self._side.plan["run_id"], "point_kind": "spsa_center",
            "point_id": center.method_id, "method_id": center.method_id,
            "source_step": center.source_step,
            "plan_file_id": plan_entry["file_id"], "planned_rows_file_id": rows_entry["file_id"],
            "packet_index_file_id": packet_entry["file_id"],
            "family_plan_pin": self.family_pin,
            "initial_selector_receipt_pin": self.initial_receipt_pin,
            "initial_ingest_index_pin": ingest,
            "execution_package_manifest_pin": self.package_pin,
            "controller_journal_pin": dict(journal_pin),
            "center_dispatcher_pin": dict(self.center_dispatcher_pin),
            "center_dispatcher_file_id": next(
                item["file_id"] for item in merged.values() if item["role"] == "center_dispatcher"),
            "point_runner_pin": self.point_runner_pin,
            "shared_application_helper_pin": self.helper_pin,
            "official_source_frame_map_file_ids": frame_map_ids,
            "blocks": [dict(item["block"]) for item in gathered],
        }
        return {
            "schema": TERMINAL_SCHEMA, "status": status,
            "point_id": center.method_id, "method_id": center.method_id,
            "point_kind": "spsa_center", "source_step": center.source_step,
            "controller_journal_pin": dict(journal_pin),
            "center_dispatcher_pin": dict(self.center_dispatcher_pin),
            "point_entry": point_entry,
            "file_entries": sorted(merged.values(), key=lambda value: value["file_id"]),
            "raw_npz_by_row": raw_npz, "candidate_rows": len(raw_npz),
            "planned_candidate_rows": 120,
        }


def prepare_center_runtime_context(*, session_dir: str | Path, journal_path: str | Path,
                                   source_step: int, packet_dir: str | Path | None = None,
                                   packet_index_path: str | Path | None = None,
                                   family_plan_pin: Mapping[str, Any],
                                   initial_selector_receipt_pin: Mapping[str, Any],
                                   execution_package_manifest_pin: Mapping[str, Any],
                                   dispatch_root: str | Path) -> Tuple[Any, dict, FitCenterDispatcher]:
    """Replay a complete session, build one center plan, and source-gate all models."""
    session = resume_verified_session(session_dir)
    journal, journal_pin = verify_sealed_session_journal(session, journal_path)
    center = center_spec_from_verified_history(session, journal, source_step)
    session_composed_pin, session_receipt_pin, session_package_pin = _session_pins(session)
    if (dict(family_plan_pin) != session_composed_pin
            or dict(initial_selector_receipt_pin) != session_receipt_pin
            or dict(execution_package_manifest_pin) != session_package_pin):
        raise CenterDispatchError("caller family/receipt/package pins differ from replayed complete session")
    dispatcher = FitCenterDispatcher(
        family_plan_pin=family_plan_pin,
        initial_selector_receipt_pin=initial_selector_receipt_pin,
        execution_package_manifest_pin=execution_package_manifest_pin,
        dispatch_root=dispatch_root,
        point_outcome_factory=lambda **kwargs: kwargs,
    )
    if (packet_dir is None) == (packet_index_path is None):
        raise CenterDispatchError("supply exactly one fresh packet directory or existing packet index")
    if packet_index_path is not None:
        selected_packet_index = Path(packet_index_path)
        packet = {"packet_index_file": {
            "path": str(selected_packet_index.resolve(strict=True)),
            **dispatcher_source.file_pin(selected_packet_index, self_field="packet_sha256"),
        }}
        prepared = {"packet": packet, "center_spec": center, "journal_pin": journal_pin}
    else:
        prepared = prepare_tell_center_packet(
            session=session, journal_path=journal_path,
            source_step=source_step, output_dir=packet_dir)
    side = dispatcher.bind_center_packet(center, prepared["packet"]["packet_index_file"]["path"])
    for model_id in MODEL_ORDER:
        dispatcher.runner.validate_point_scope(
            family_plan_pin=dispatcher.family_pin, point_plan_pin=side.plan_pin,
            initial_selector_receipt_pin=dispatcher.initial_receipt_pin,
            execution_package_manifest_pin=dispatcher.package_pin, model_id=model_id)
        profile, _ = dispatcher._read_profile(model_id)
        dispatcher_source._materialize_environment(
            profile, request={"strict_receipt_dir": str(Path(dispatch_root) / "preflight" / model_id)},
            family_plan=dispatcher.family, runtime_pin=dispatcher.model_entries[model_id]["runtime_binding_pin"])
    return session, {"center": center, "journal": journal, "journal_pin": journal_pin,
                     "packet": prepared["packet"], "side": side}, dispatcher


def _merge_file_entries(target: dict[str, dict], entries: Sequence[Mapping[str, Any]]) -> None:
    by_role_path = {(item.get("role"), item.get("declared_path")): item
                    for item in target.values()}
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise CenterDispatchError("final index source file table contains a malformed entry")
        item = dict(raw)
        file_id = item.get("file_id")
        if not isinstance(file_id, str) or not file_id:
            raise CenterDispatchError("final index source file table has no file ID")
        old = target.get(file_id)
        if old is not None:
            if old != item:
                raise CenterDispatchError("final index file ID collides with different evidence")
            continue
        key = (item.get("role"), item.get("declared_path"))
        prior = by_role_path.get(key)
        if prior is not None:
            if any(prior.get(name) != item.get(name) for name in ("bytes", "sha256", "local_path")):
                raise CenterDispatchError("same final index role/path has inconsistent raw bytes")
            # The consumer addresses files by role/path as well as IDs. Keep a
            # single canonical entry when two producers used different ID rules.
            continue
        target[file_id] = item
        by_role_path[key] = item


def build_unselected_final_center_index(*, session: Any, journal_path: str | Path,
                                        center_terminal_paths: Mapping[int, str | Path],
                                        output_path: str | Path) -> dict:
    """Assemble the full frozen consumer input without running its selector.

    This is intentionally a transport/index builder. It refuses any absent,
    partial, failed, or unverified point and always writes selection_status as
    NOT_SELECTED. Consumer04 performs the eventual full selector replay.
    """
    journal, journal_pin = verify_sealed_session_journal(session, journal_path)
    if set(center_terminal_paths) != set(CENTER_STEPS):
        raise CenterDispatchError("final index requires exactly four tell centers at 5/10/15/20")
    if (session.controller.iteration != 20 or session.controller._pending is not None
            or len(session.point_entries) != 40):
        raise CenterDispatchError("final index requires the complete replayed twenty-pair session")
    proposal_points: list[dict] = []
    merged_files: dict[str, dict] = {}
    seen_proposal_ids = set()
    seen_candidate_rows = set()
    expected_methods = set()
    for journal_entry in journal["entries"]:
        expected_methods.add(journal_entry["plus_point_id"])
        expected_methods.add(journal_entry["minus_point_id"])
    if len(expected_methods) != 40:
        raise CenterDispatchError("sealed journal does not enumerate forty unique proposal methods")
    for key, record in session.point_entries.items():
        doc = record.get("point_doc") if isinstance(record, Mapping) else None
        event = record.get("event") if isinstance(record, Mapping) else None
        if (not isinstance(doc, Mapping) or doc.get("terminal_status") != "COMPLETE"
                or not isinstance(event, Mapping) or event.get("terminal_status") != "COMPLETE"):
            raise CenterDispatchError("proposal point lacks a complete replay-verified terminal record")
        point_entry = doc.get("point_entry")
        if (not isinstance(point_entry, Mapping) or point_entry.get("point_kind") != "spsa_proposal"
                or point_entry.get("method_id") not in expected_methods
                or point_entry.get("method_id") in seen_proposal_ids
                or point_entry.get("proposal_id") != key[0]
                or point_entry.get("side") != key[1]
                or len(doc.get("raw_npz_by_row", {})) != 24):
            raise CenterDispatchError("proposal entry differs from its exact durable ask/pair or 24-row NPZ denominator")
        seen_proposal_ids.add(point_entry["method_id"])
        rows = point_entry.get("official_source_frame_map_file_ids")
        if not isinstance(rows, Mapping) or len(rows) != 24:
            raise CenterDispatchError("proposal entry lacks 24 official source frame-map references")
        if seen_candidate_rows.intersection(doc["raw_npz_by_row"]):
            raise CenterDispatchError("candidate proposal row or NPZ identity is reused across asks")
        seen_candidate_rows.update(doc["raw_npz_by_row"])
        _merge_file_entries(merged_files, doc.get("file_entries", []))
        proposal_points.append(dict(point_entry))
    if seen_proposal_ids != expected_methods or len(seen_candidate_rows) != 960:
        raise CenterDispatchError("proposal terminal set does not exactly cover all 960 planned candidate rows")
    # Preserve journal order, plus before minus, exactly as deterministic ask/tell replay emitted it.
    order: dict[str, int] = {}
    for entry_index, journal_entry in enumerate(journal["entries"]):
        order[journal_entry["plus_point_id"]] = 2 * entry_index
        order[journal_entry["minus_point_id"]] = 2 * entry_index + 1
    proposal_points.sort(key=lambda item: order[item["method_id"]])

    center_points = []
    center_row_ids = set()
    for step in CENTER_STEPS:
        path = Path(center_terminal_paths[step])
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise CenterDispatchError("center terminal must be an absolute regular file")
        terminal, raw = _read_json(path, "center dispatch terminal")
        if (terminal.get("schema") != TERMINAL_SCHEMA
                or terminal.get("terminal_sha256") != _self_hash(terminal, "terminal_sha256")
                or terminal.get("status") != "COMPLETE"
                or terminal.get("point_kind") != "spsa_center"
                or terminal.get("source_step") != step
                or terminal.get("planned_candidate_rows") != 120
                or terminal.get("candidate_rows") != 120):
            raise CenterDispatchError("center terminal is not a complete full-120 tell-derived block")
        point_entry = terminal.get("point_entry")
        expected = center_spec_from_verified_history(
            session, journal, step)
        if (not isinstance(point_entry, Mapping) or point_entry.get("point_id") != expected.method_id
                or point_entry.get("method_id") != expected.method_id
                or point_entry.get("source_step") != step
                or point_entry.get("point_kind") != "spsa_center"
                or point_entry.get("plan_file_id") is None
                or len(point_entry.get("official_source_frame_map_file_ids", {})) != 120
                or len(terminal.get("raw_npz_by_row", {})) != 120):
            raise CenterDispatchError("center terminal point identity/120-row NPZ/frame-map closure differs")
        if center_row_ids.intersection(terminal["raw_npz_by_row"]):
            raise CenterDispatchError("center row NPZ identity is reused between tell steps")
        center_row_ids.update(terminal["raw_npz_by_row"])
        _merge_file_entries(merged_files, terminal.get("file_entries", []))
        # Keep the exact terminal sidecar as additional provenance. The frozen
        # consumer independently validates every row and child receipt.
        terminal_file = dispatcher_source._add_file(
            merged_files, path, "center_dispatch_terminal", self_field="terminal_sha256")
        point_entry = dict(point_entry)
        point_entry["dispatch_terminal_file_id"] = terminal_file["file_id"]
        center_points.append(point_entry)
    if len(center_row_ids) != 480:
        raise CenterDispatchError("four center points do not cover the exact 480 candidate rows")

    inputs = session.inputs
    durable_files = [durable._file_entry(pin) for pin in inputs.files.values()]
    _merge_file_entries(merged_files, durable_files)
    journal_entry = dispatcher_source._add_file(
        merged_files, Path(journal_pin["path"]), "controller_journal", self_field="journal_sha256")
    _merge_file_entries(merged_files, [journal_entry])
    roles = {(item["role"], item["declared_path"]): item["file_id"] for item in merged_files.values()}
    required_ids = {
        "initial_ingest_index_file_id": roles.get(("initial_ingest_index", inputs.initial_index_pin.declared_path)),
        "initial_selector_receipt_file_id": roles.get(("initial_selector_receipt", inputs.initial_receipt_pin.declared_path)),
        "composed_family_plan_file_id": roles.get(("composed_family_plan", inputs.composed_pin.declared_path)),
        "execution_package_manifest_file_id": roles.get(("execution_package_manifest", inputs.package_pin.declared_path)),
        "controller_journal_file_id": journal_entry["file_id"],
    }
    if any(not value for value in required_ids.values()):
        raise CenterDispatchError("final index cannot resolve one of its pinned top-level evidence files")
    family = inputs.family_selection.family
    body = {
        "schema": final_ingest.SCHEMA,
        "status": final_ingest.STATUS_READY,
        "metric_backend": "torch_cpu", "selection_status": "NOT_SELECTED",
        "expected_initial_row_count": 720, "expected_final_row_count": 720,
        "expected_proposal_pair_count": 20, "expected_proposal_point_count": 40,
        "expected_center_count": 4, "expected_center_row_count": 480,
        "expected_proposal_row_count": 960,
        "scientific_input_pins_sha256": family.input_pins_sha256,
        "model_ids": list(family.roster.model_ids),
        "sample_ids": list(family.roster.sample_ids),
        **required_ids,
        "proposal_points": proposal_points,
        "center_points": center_points,
        "files": sorted(merged_files.values(), key=lambda item: item["file_id"]),
    }
    body["index_sha256"] = _self_hash(body, "index_sha256")
    target = Path(output_path)
    if not target.is_absolute() or target.exists() or target.is_symlink():
        raise CenterDispatchError("final index output must be an absent absolute path")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write((json.dumps(body, sort_keys=True, indent=2, ensure_ascii=False,
                                 allow_nan=False) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(target.resolve(strict=True)), "bytes": target.stat().st_size,
            "sha256": _sha(target.read_bytes()), "self_sha256": body["index_sha256"],
            "status": body["status"], "selection_status": body["selection_status"],
            "proposal_rows": 960, "center_rows": 480}


__all__ = [
    "CENTER_STEPS", "CenterDispatchError", "FitCenterDispatcher",
    "build_unselected_final_center_index",
    "center_spec_from_verified_history", "prepare_center_runtime_context",
    "prepare_tell_center_packet", "resume_verified_session", "validate_center_plan_shape",
    "verify_sealed_session_journal",
]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "execute", "build-index"))
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--source-step", type=int, choices=CENTER_STEPS)
    parser.add_argument("--packet-dir")
    parser.add_argument("--packet-index")
    parser.add_argument("--dispatch-root")
    parser.add_argument("--output")
    parser.add_argument("--center-terminal", action="append", default=[], metavar="STEP=PATH")
    args = parser.parse_args(argv)
    try:
        session = resume_verified_session(args.session_dir)
        journal, journal_pin = verify_sealed_session_journal(session, args.journal)
        if args.mode in {"prepare", "execute"}:
            if (args.source_step not in CENTER_STEPS or not args.dispatch_root
                    or ((args.packet_dir is None) == (args.packet_index is None))):
                raise CenterDispatchError(
                    "prepare/execute requires --source-step, --dispatch-root, and exactly one packet source")
            if args.mode == "prepare" and args.packet_index is not None:
                raise CenterDispatchError("prepare mode creates a fresh packet and cannot consume --packet-index")
            pins = _session_pins(session)
            family_pin, receipt_pin, package_pin = pins
            # prepare_center_runtime_context replays and compares the same
            # session once more at this public boundary; no caller-supplied
            # scientific ID or theta is accepted.
            replayed, context, dispatcher = prepare_center_runtime_context(
                session_dir=args.session_dir, journal_path=args.journal,
                source_step=args.source_step, packet_dir=args.packet_dir,
                packet_index_path=args.packet_index,
                family_plan_pin=family_pin, initial_selector_receipt_pin=receipt_pin,
                execution_package_manifest_pin=package_pin, dispatch_root=args.dispatch_root)
            center = context["center"]
            packet_index = context["packet"]["packet_index_file"]["path"]
            report = {
                "status": "PREFLIGHT_PASS_NO_MODEL_CALLS",
                "point_kind": "spsa_center", "source_step": center.source_step,
                "point_id": center.method_id, "packet_index_path": packet_index,
                "planned_rows": 120, "model_ids": list(MODEL_ORDER),
                "journal_pin": context["journal_pin"], "execution_started": False,
            }
            if args.mode == "execute":
                terminal = dispatcher.execute_center(center, journal_pin=context["journal_pin"])
                report.update({"status": terminal["status"],
                               "terminal_path": terminal["terminal_pin"]["path"],
                               "terminal_pin": terminal["terminal_pin"],
                               "candidate_rows": terminal["candidate_rows"],
                               "planned_candidate_rows": terminal["planned_candidate_rows"],
                               "execution_started": True})
            print(json.dumps(report, sort_keys=True, ensure_ascii=False))
            return 0 if report["status"] in {"PREFLIGHT_PASS_NO_MODEL_CALLS", "COMPLETE"} else 2
        if args.output is None:
            raise CenterDispatchError("build-index requires --output")
        centers = {}
        for raw in args.center_terminal:
            if "=" not in raw:
                raise CenterDispatchError("each --center-terminal must be STEP=PATH")
            raw_step, raw_path = raw.split("=", 1)
            try:
                step = int(raw_step)
            except ValueError as exc:
                raise CenterDispatchError("center terminal step must be an integer") from exc
            if step in centers or step not in CENTER_STEPS or not raw_path:
                raise CenterDispatchError("center terminal steps must uniquely cover 5/10/15/20")
            centers[step] = raw_path
        result = build_unselected_final_center_index(
            session=session, journal_path=args.journal,
            center_terminal_paths=centers, output_path=args.output)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "error": type(exc).__name__ + ": " + str(exc)},
                         sort_keys=True, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
