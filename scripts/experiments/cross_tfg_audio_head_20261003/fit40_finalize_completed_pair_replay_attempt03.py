"""Seal a recovered completion record for an already-persisted SPSA tell.

This finalizer is read-only with respect to the durable session and model
outputs. It accepts only the exact seven-event replay prefix, recomputes the
session from its journal and pinned official score evidence, validates the
READY lineage and the prior wrapper-failure capture, then writes one
create-only recovered-completion receipt. It never asks, tells, dispatches,
renders, or scores.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


EXPECTED_DRIVER_SHA256 = "c0e940f198e2e38660e9d21142b5718f86cfd4b85257d650ccf172bfc05e467f"
EXPECTED_DRIVER_BYTES = 48242
EXPECTED_SESSION_SHA256 = "10e50009a12e7cf6f4286e4d30a9cf36668a6425d6b4b82247b837887f9edd75"
EXPECTED_SESSION_BYTES = 92810
EXPECTED_PAIR_ID = "e5fa43c586f7665835da25c62d2e0acb3260f9662a2412baf913e2a688d73fc3"
EXPECTED_EVENT_TYPES = [
    "SESSION_INITIALIZED", "PAIR_ASKED", "POINT_DISPATCH_INTENT", "POINT_TERMINAL",
    "POINT_DISPATCH_INTENT", "POINT_TERMINAL", "PAIR_TOLD",
]
EXPECTED_DRIVER_ERROR = (
    "AttributeError: 'SPSATellResult' object has no attribute 'pair_id'"
)
COMPLETION_SCHEMA = "fit40_completed_pair_replay_prefix_recovery_v1"
COMPLETION_STATUS = "COMPLETE_PAIR_REPLAY_PREFIX_RECOVERED_POST_TELL"


class RecoveryError(RuntimeError):
    pass


def _raw_pin(path: str | Path, *, self_field: str | None = None) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise RecoveryError(f"expected a non-symlink regular file: {target}")
    resolved = target.resolve(strict=True)
    raw = resolved.read_bytes()
    pin: dict[str, Any] = {
        "path": str(resolved),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "self_sha256": None,
    }
    if self_field is not None:
        value = json.loads(raw.decode("utf-8"))
        actual_self = value.get(self_field)
        body = dict(value)
        body.pop(self_field, None)
        expected_self = hashlib.sha256(_canonical(body)).hexdigest()
        if actual_self != expected_self:
            raise RecoveryError(f"invalid embedded {self_field}: {resolved}")
        pin["self_sha256"] = actual_self
    return pin


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _read_json(path: str | Path, label: str) -> tuple[dict[str, Any], bytes]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise RecoveryError(f"{label} must be a non-symlink regular file: {target}")
    raw = target.read_bytes()
    value = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} must be a JSON object")
    return value, raw


def _load_module(path: Path, name: str, expected_bytes: int, expected_sha256: str):
    path = path.resolve(strict=True)
    raw = path.read_bytes()
    if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RecoveryError(f"durable-session source differs from the reviewed bytes: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RecoveryError(f"cannot load pinned durable-session module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve(strict=True) != path:
        raise RecoveryError("loaded durable-session source origin changed")
    return module


def _read_driver_failure(capture_dir: Path, driver_pin: Mapping[str, Any], replay_module) -> dict[str, Any]:
    if capture_dir.is_symlink() or not capture_dir.is_dir():
        raise RecoveryError("original driver capture directory is missing or unsafe")
    stdout_pin = _raw_pin(capture_dir / "stdout.txt")
    stderr_pin = _raw_pin(capture_dir / "stderr.txt")
    transport_pin = _raw_pin(capture_dir / "ssh_return.json")
    if (stdout_pin["bytes"] != 0
            or stdout_pin["sha256"] != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            or stderr_pin["bytes"] != 140
            or stderr_pin["sha256"] != "34c7327e46209f7ff522f07086f44d3988ff7859f1ae64cfe100a0e7613b3740"
            or transport_pin["bytes"] != 335
            or transport_pin["sha256"] != "72587f171720edc4cded4fbd58458f8a3c917cf6eaa00d4a8ba8ca31c6be8729"):
        raise RecoveryError("original replay04 process capture differs from the observed rc2 bytes")
    stdout_raw = (capture_dir / "stdout.txt").read_bytes()
    stderr_raw = (capture_dir / "stderr.txt").read_bytes()
    transport, _transport_raw = _read_json(capture_dir / "ssh_return.json", "replay04 transport capture")
    lines = stderr_raw.decode("utf-8").splitlines()
    if (stdout_raw != b"" or len(lines) != 2 or lines[-1] != "REMOTE_DRIVER_EXIT_CODE=2"
            or transport.get("ssh_return_code") != 2):
        raise RecoveryError("driver04 failure capture is not the exact post-tell rc2 observation")
    try:
        failure = json.loads(lines[0])
    except Exception as exc:
        raise RecoveryError("driver04 stderr does not contain its JSON failure record") from exc
    if (failure.get("status") != "PAIR_REPLAY_FAILED"
            or failure.get("error") != EXPECTED_DRIVER_ERROR
            or driver_pin.get("sha256") != EXPECTED_DRIVER_SHA256
            or driver_pin.get("bytes") != EXPECTED_DRIVER_BYTES):
        raise RecoveryError("driver04 failure and exact reviewed source pin do not match")
    return {
        "status": "PAIR_REPLAY_FAILED_AFTER_DURABLE_TELL",
        "driver_source_pin": dict(driver_pin),
        "driver_return_code": 2,
        "ssh_return_code": 2,
        "failure_error": EXPECTED_DRIVER_ERROR,
        "staged_failure_capture_directory": str(capture_dir.resolve(strict=True)),
        "stdout_pin": stdout_pin,
        "stderr_pin": stderr_pin,
        "ssh_return_pin": transport_pin,
        "started_utc": transport.get("started_utc"),
        "finished_utc": transport.get("finished_utc"),
        "replay_driver_module_processes_started": 1,
    }


def _seal_once(path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise RecoveryError("completion parent must exist as a non-symlink directory")
    sealed = dict(document)
    sealed["receipt_sha256"] = hashlib.sha256(_canonical(sealed)).hexdigest()
    raw = (json.dumps(sealed, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(str(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    pin = _raw_pin(path, self_field="receipt_sha256")
    if pin["bytes"] != len(raw) or pin["sha256"] != hashlib.sha256(raw).hexdigest():
        raise RecoveryError("create-only recovered completion readback mismatch")
    return pin


def run(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise RecoveryError("recovered-completion finalizer requires CUDA hidden")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    completion_path = Path(args.completion_path)
    if completion_path.exists() or completion_path.is_symlink():
        raise RecoveryError("recovered completion target must be absent; never overwrite")

    project_root = Path(args.project_root).resolve(strict=True)
    driver_source = Path(args.driver_source).resolve(strict=True)
    driver_pin = _raw_pin(driver_source)
    if (driver_pin["bytes"] != EXPECTED_DRIVER_BYTES
            or driver_pin["sha256"] != EXPECTED_DRIVER_SHA256):
        raise RecoveryError("replay driver source differs from the reviewed attempt04 bytes")
    replay_spec = importlib.util.spec_from_file_location("fit40_replay_driver04_for_finalization", driver_source)
    if replay_spec is None or replay_spec.loader is None:
        raise RecoveryError("cannot load reviewed replay driver helpers")
    replay_module = importlib.util.module_from_spec(replay_spec)
    sys.modules[replay_spec.name] = replay_module
    replay_spec.loader.exec_module(replay_module)
    if Path(replay_module.__file__).resolve(strict=True) != driver_source:
        raise RecoveryError("replay driver helper origin changed")

    replay_module.configure_project_imports(project_root)
    old_source = Path(args.old_session_source).resolve(strict=True)
    new_source = Path(args.new_session_source).resolve(strict=True)
    old_module = _load_module(
        old_source, "fit40_old_durable_session05_for_finalization",
        90811, "03acef33a1d38e874104a663f1a6d1e16a811190b111ebeb52febf6882fa2412")
    new_module = _load_module(
        new_source, "fit40_new_durable_session08_for_finalization",
        EXPECTED_SESSION_BYTES, EXPECTED_SESSION_SHA256)
    old_session = old_module.DurableSPSASession.resume(session_dir=args.old_session_dir)
    new_session = new_module.DurableSPSASession.resume(session_dir=args.new_session_dir)
    # The replay driver helper intentionally accepts only the pre-replay
    # event0/event1 state.  This finalizer instead checks that same original
    # ask/state pair and then proves the already-persisted event2..event6
    # completion without calling ask_pair() or tell_pair().
    if [event.get("event_type") for event in old_session.events] != replay_module.EVENT_TYPES_OLD:
        raise RecoveryError("old session is not exactly the preserved six-event pair-01 UNKNOWN prefix")
    if [event.get("event_type") for event in new_session.events] != EXPECTED_EVENT_TYPES:
        raise RecoveryError("fresh session event chain is not the exact 0–6 completed replay prefix")
    old_ask = old_session.events[1]
    fresh_ask = new_session.events[1]
    if (old_ask.get("ask_pair") != fresh_ask.get("ask_pair")
            or old_ask.get("controller_state_after") != fresh_ask.get("controller_state_after")
            or old_session.events[0].get("controller_state")
            != new_session.events[0].get("controller_state")):
        raise RecoveryError("fresh replay ask/controller/PCG64 state differs from the original ask")
    pair = old_session.pending_pair
    if pair is None or new_session.pending_pair is not None:
        raise RecoveryError("old pending ask or already-completed fresh session state is missing")
    if (pair.iteration != 0 or pair.pair_id != EXPECTED_PAIR_ID
            or old_ask.get("pair_id") != pair.pair_id
            or new_module._plain(pair) != fresh_ask.get("ask_pair")):
        raise RecoveryError("fresh deterministic pair is not the exact original pair-01 ask")
    if pair.pair_id != EXPECTED_PAIR_ID:
        raise RecoveryError("durably told pair differs from exact pair-01")
    if (len(old_session.events) != 6 or new_session.controller.iteration != 1
            or new_session.pending_pair is not None or new_session.halted):
        raise RecoveryError("old/fresh session controller state differs from the recovered first tell")

    lineage_path = Path(args.lineage_path)
    lineage, _lineage_raw = replay_module.read_json(lineage_path, "READY replay lineage")
    lineage_pin = replay_module.raw_pin(lineage_path, "receipt_sha256")
    if (lineage.get("schema") != "fit40_completed_pair_replay_lineage_v1"
            or lineage.get("status") != "READY_TO_IMPORT_VERIFIED_COMPLETE_PAIR_TERMINALS"
            or lineage.get("pair_id") != pair.pair_id
            or lineage.get("ask_pair") != new_module._plain(pair)
            or lineage.get("ask_pair_sha256") != hashlib.sha256(
                replay_module.canonical(new_module._plain(pair))).hexdigest()):
        raise RecoveryError("READY lineage does not bind the exact existing pair and ask")

    expected_old_module = replay_module._module_source_pin(old_session, "durable_session_source")
    expected_new_module = replay_module._module_source_pin(new_session, "durable_session_source")
    if (expected_old_module["sha256"] != "03acef33a1d38e874104a663f1a6d1e16a811190b111ebeb52febf6882fa2412"
            or expected_new_module["sha256"] != EXPECTED_SESSION_SHA256
            or lineage.get("source_pins", {}).get("old_durable_session_source") != expected_old_module
            or lineage.get("source_pins", {}).get("replay_durable_session_source") != expected_new_module
            or lineage.get("source_pins", {}).get("replay_driver_source") != driver_pin):
        raise RecoveryError("READY lineage source closure differs from exact loaded sources")

    replay_module._same_receipt_except_session_source(old_session.inputs.receipt,
                                                       new_session.inputs.receipt)
    for key in ("scientific_input_pins", "scientific_input_pins_sha256", "spsa_seed_method_id",
                "initial_selection_status", "initial_theta", "source_roster", "reference_specs",
                "source_snapshot_lineage", "composed_family_plan", "execution_package_manifest",
                "initial_ingest_index"):
        if old_session.manifest.get(key) != new_session.manifest.get(key):
            raise RecoveryError(f"fresh session changed a scientific input field: {key}")
    expected_lineage_links = {
        "fresh_initial_selector_receipt": replay_module.verify_link(
            new_session.manifest["initial_selector_receipt"], "fresh initial-selector receipt"),
        "composed_family_plan": replay_module.verify_link(
            new_session.manifest["composed_family_plan"], "fresh composed family plan"),
        "execution_package_manifest": replay_module.verify_link(
            new_session.manifest["execution_package_manifest"], "fresh execution package"),
        "initial_ingest_index": replay_module.verify_link(
            new_session.manifest["initial_ingest_index"], "fresh initial ingest index"),
    }
    if any(lineage.get(key) != pin for key, pin in expected_lineage_links.items()):
        raise RecoveryError("READY lineage receipt/package/family/index links differ from the resumed session")

    pair_dir = Path(args.old_pair_dir).resolve(strict=True)
    old_packet_pin = replay_module._verify_outer_packet(pair_dir, pair, old_session)
    outer_failure = replay_module._capture_process_failure(pair_dir, pair, old_session)
    point_summaries: dict[str, Any] = {}
    candidate_npz_paths: set[str] = set()
    candidate_map_paths: set[str] = set()
    fresh_terminal_pins: dict[str, Any] = {}
    for side, proposal in (("plus", pair.plus), ("minus", pair.minus)):
        key = (proposal.proposal_id, side)
        old_record = old_session.point_entries.get(key)
        new_record = new_session.point_entries.get(key)
        if (not isinstance(old_record, Mapping) or not isinstance(new_record, Mapping)
                or old_record.get("event", {}).get("terminal_status") != "COMPLETE"
                or new_record.get("event", {}).get("terminal_status") != "COMPLETE"
                or old_record.get("point_doc") != new_record.get("point_doc")):
            raise RecoveryError(f"{side} point terminal is absent or differs from the immutable original")
        summary = replay_module._point_file_links(
            new_module, old_record, proposal, old_session,
            candidate_npz_paths, candidate_map_paths)
        if lineage.get("point_terminals", {}).get(side) != summary:
            raise RecoveryError(f"READY lineage does not match the reread {side} point evidence")
        point_summaries[side] = summary
        evaluation = new_session._load_official_evaluation(proposal, new_record["point_doc"])
        if (not new_session._validate_evaluation_matches(evaluation, proposal)
                or len([row for row in evaluation.rows if row.method_id == proposal.method_id]) != 24):
            raise RecoveryError(f"{side} official evaluation no longer matches the exact proposal")
        new_session._validate_evaluation_rows(evaluation, proposal)
        derivations = new_session.native_support_derivations.get(key)
        candidate_rows = {row.row_id for row in evaluation.rows if row.method_id == proposal.method_id}
        if not isinstance(derivations, Mapping) or set(derivations) != candidate_rows or len(candidate_rows) != 24:
            raise RecoveryError(f"{side} native support derivations do not cover its 24 official rows")
        journal_eval = new_session.events[6].get("journal_entry", {}).get(f"{side}_evaluation", {})
        persisted = journal_eval.get("candidate_point_evidence", {}).get(
            "native_support_derivations_by_row", {})
        if dict(persisted) != dict(derivations):
            raise RecoveryError(f"PAIR_TOLD {side} native derivation record differs from reread evidence")
        fresh_terminal_pins[side] = dict(new_record["point_pin"])

    if len(candidate_npz_paths) != 48 or len(candidate_map_paths) != 48:
        raise RecoveryError("reread completed pair does not contain 48 unique NPZs and frame maps")
    if lineage.get("official_candidate_evidence_counts") != {
            "candidate_rows": 48, "official_raw_npz": 48,
            "official_source_frame_maps": 48, "model_child_receipts_with_numeric_rc0": 6}:
        raise RecoveryError("READY lineage candidate evidence counts differ")

    tell_event = new_session.events[6]
    tell_body = tell_event.get("journal_entry")
    tell_result = tell_event.get("tell_result")
    if (tell_event.get("event_type") != "PAIR_TOLD" or tell_event.get("pair_id") != pair.pair_id
            or tell_event.get("iteration") != 0 or not isinstance(tell_body, Mapping)
            or not isinstance(tell_result, Mapping) or tell_result.get("status") != "UPDATED"
            or tell_body.get("tell_result") != tell_result
            or tell_body.get("pair_id") != pair.pair_id):
        raise RecoveryError("persisted event6 is not the exact already-written UPDATED tell")
    entry_field, entry_self = replay_module.embedded_self(tell_body)
    if entry_field != "entry_sha256" or tell_body.get("entry_sha256") != entry_self:
        raise RecoveryError("persisted PAIR_TOLD journal entry self hash is invalid")
    driver_failure = _read_driver_failure(Path(args.driver_capture_dir), driver_pin, replay_module)
    finalizer_pin = _raw_pin(Path(__file__).resolve(strict=True))

    old_events = [replay_module._event_link(old_session, i) for i in range(len(old_session.events))]
    new_events = [replay_module._event_link(new_session, i) for i in range(len(new_session.events))]
    declarations = lineage.get("replay_adoption_intents")
    if not isinstance(declarations, list) or len(declarations) != 2:
        raise RecoveryError("READY lineage does not bind both readback-only side imports")
    adopted = replay_module._bind_fresh_replay_events(new_session, declarations, pair, new_module)
    if len(adopted) != 2 or any(
            item.get("fresh_intent_semantics") != "readback_adoption_only_no_child_or_model_process"
            for item in adopted):
        raise RecoveryError("fresh side intents are not explicitly readback-only")

    source_pins = {
        "old_durable_session_source": expected_old_module,
        "replay_durable_session_source": expected_new_module,
        "replay_driver_source": driver_pin,
        "completion_recovery_finalizer_source": finalizer_pin,
    }
    completion = {
        "schema": COMPLETION_SCHEMA,
        "status": COMPLETION_STATUS,
        "continuation_kind": "existing_pair_terminal_replay_no_native_dispatch",
        "pair_id": pair.pair_id,
        "iteration": 0,
        "ask_pair": new_module._plain(pair),
        "tell_result": dict(tell_result),
        "controller_state_after": tell_event["controller_state_after"],
        "tell_journal_entry": dict(tell_body),
        "new_session_manifest": replay_module.raw_pin(
            Path(args.new_session_dir) / "session_manifest.json", "manifest_sha256"),
        "new_session_events": new_events,
        "tell_event": new_events[6],
        "point_entries": fresh_terminal_pins,
        "replay_adoption_intents": adopted,
        "old_terminal_provenance": {
            "old_session_manifest": lineage["old_session_manifest"],
            "old_session_events": old_events,
            "old_point_entries": lineage["old_point_entries"],
            "old_outer_pair_process_failure": outer_failure,
            "original_pair_packet_index": old_packet_pin,
        },
        "initial_selector_receipt": lineage["fresh_initial_selector_receipt"],
        "composed_family_plan": lineage["composed_family_plan"],
        "execution_package_manifest": lineage["execution_package_manifest"],
        "initial_ingest_index": lineage["initial_ingest_index"],
        "lineage_receipt": replay_module.verify_link(lineage_pin, "READY replay lineage receipt"),
        "native_work_for_replayed_prefix": {
            "new_model_processes_started": 0,
            "existing_candidate_rows_reused_once": 48,
            "official_npz_revalidated": 48,
            "overall_pair_child_exit_code": None,
        },
        "recovery_provenance": {
            "recovery_schema": "fit40_completed_pair_replay_post_tell_recovery_v1",
            "recovery_status": "RECOVERED_FROM_PERSISTED_UPDATED_TELL_AFTER_WRAPPER_EXIT",
            "original_driver_failure": driver_failure,
            "replay04_source_pin": driver_pin,
            "completion_recovery_finalizer_source_pin": finalizer_pin,
            "fresh_session_event_count_before_finalization": 7,
            "persisted_tell_event_index": 6,
            "persisted_tell_status": "UPDATED",
            "tell_rerun": False,
            "ask_rerun": False,
            "dispatch_or_model_process_started_by_finalizer": False,
            "official_rows_revalidated_by_finalizer": 48,
            "overall_pair_child_exit_code": None,
        },
        "source_pins": source_pins,
    }
    completion_pin = _seal_once(completion_path, completion)
    return {
        "status": COMPLETION_STATUS,
        "completion_receipt": completion_pin,
        "lineage_receipt": replay_module.verify_link(lineage_pin, "READY replay lineage receipt"),
        "pair_id": pair.pair_id,
        "tell_status": tell_result["status"],
        "controller_iteration": new_session.controller.iteration,
        "session_event_count": len(new_session.events),
        "new_model_processes_started": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--driver-source", required=True, type=Path)
    parser.add_argument("--old-session-source", required=True, type=Path)
    parser.add_argument("--new-session-source", required=True, type=Path)
    parser.add_argument("--old-session-dir", required=True, type=Path)
    parser.add_argument("--new-session-dir", required=True, type=Path)
    parser.add_argument("--old-pair-dir", required=True, type=Path)
    parser.add_argument("--lineage-path", required=True, type=Path)
    parser.add_argument("--driver-capture-dir", required=True, type=Path)
    parser.add_argument("--completion-path", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(args)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "RECOVERED_COMPLETION_FAILED",
                          "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
