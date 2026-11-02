"""Import one already-complete SPSA pair into a corrected durable session.

This is a one-shot control/data repair for the immutable pair-01 outputs. It
revalidates the old terminal records and official score files, then returns
those records through the normal durable-session point outcome API. It never
imports the native point dispatcher or launches a model process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


LINEAGE_SCHEMA = "fit40_completed_pair_replay_lineage_v1"
COMPLETION_SCHEMA = "fit40_completed_pair_replay_prefix_v1"
EVENT_TYPES_OLD = [
    "SESSION_INITIALIZED", "PAIR_ASKED", "POINT_DISPATCH_INTENT",
    "POINT_TERMINAL", "POINT_DISPATCH_INTENT", "POINT_TERMINAL",
]
EVENT_TYPES_FRESH_PREFIX = [
    "SESSION_INITIALIZED", "PAIR_ASKED", "POINT_DISPATCH_INTENT",
    "POINT_TERMINAL", "POINT_DISPATCH_INTENT", "POINT_TERMINAL", "PAIR_TOLD",
]


class ReplayError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def embedded_self(doc: Mapping[str, Any]) -> tuple[str, str]:
    matches = []
    for key, value in doc.items():
        if not key.endswith("_sha256") or not isinstance(value, str) or len(value) != 64:
            continue
        body = dict(doc)
        body.pop(key)
        if sha(canonical(body)) == value:
            matches.append((key, value))
    if len(matches) != 1:
        raise ReplayError("JSON artifact does not have exactly one verifiable canonical self hash")
    return matches[0]


def read_json(path: str | Path, label: str) -> tuple[dict[str, Any], bytes]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ReplayError(f"{label} must be a non-symlink regular file: {target}")
    raw = target.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ReplayError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayError(f"{label} must be a JSON object")
    return value, raw


def raw_pin(path: str | Path, self_field: str | None = None) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ReplayError(f"cannot pin non-regular/symlink source: {target}")
    resolved = target.resolve(strict=True)
    raw = resolved.read_bytes()
    result: dict[str, Any] = {
        "path": str(resolved), "bytes": len(raw), "sha256": sha(raw),
        "self_sha256": None,
    }
    if self_field is not None:
        value, _ = read_json(resolved, "self-hashed JSON")
        actual_field, actual_self = embedded_self(value)
        if actual_field != self_field or value.get(self_field) != actual_self:
            raise ReplayError(f"{resolved} has the wrong embedded self field")
        result["self_sha256"] = actual_self
    else:
        try:
            value, _ = read_json(resolved, "optional-self JSON")
        except ReplayError:
            value = None
        if isinstance(value, Mapping):
            try:
                _field, result["self_sha256"] = embedded_self(value)
            except ReplayError:
                pass
    return result


def verify_link(link: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(link, Mapping) or set(link) != {"path", "bytes", "sha256", "self_sha256"}:
        raise ReplayError(f"{label} link does not have the exact four-field pin schema")
    observed = raw_pin(link["path"])
    if any(observed[key] != link[key] for key in ("path", "bytes", "sha256")):
        raise ReplayError(f"{label} raw file pin changed")
    if link["self_sha256"] is not None:
        value, _ = read_json(link["path"], label)
        _field, self_value = embedded_self(value)
        if self_value != link["self_sha256"]:
            raise ReplayError(f"{label} canonical self pin changed")
        observed["self_sha256"] = self_value
    return observed


def load_module(path: Path, name: str):
    path = path.resolve(strict=True)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ReplayError(f"cannot load pinned durable-session source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve(strict=True) != path:
        raise ReplayError(f"loaded durable-session origin differs from its pinned path: {path}")
    return module


def configure_project_imports(project_root: Path) -> list[str]:
    root = project_root.resolve(strict=True)
    root_scripts = root / "scripts"
    if root_scripts.is_symlink() or not root_scripts.is_dir():
        raise ReplayError("staged queue scripts root is absent or unsafe")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    scripts = importlib.import_module("scripts")
    if getattr(scripts, "__file__", None) is not None:
        raise ReplayError("queue scripts import is not the measured namespace package")
    paths = []
    for value in getattr(scripts, "__path__", ()):
        path = Path(str(value))
        if path.is_symlink() or not path.is_dir():
            raise ReplayError("scripts namespace contains an unsafe path")
        resolved = path.resolve(strict=True)
        if resolved != root_scripts.resolve(strict=True):
            raise ReplayError("scripts namespace contains an unpinned import root")
        paths.append(str(resolved))
    if paths != [str(root_scripts.resolve(strict=True))]:
        raise ReplayError("scripts namespace does not resolve solely to the pinned queue source")
    return paths


def _module_source_pin(session, role: str) -> dict[str, Any]:
    entries = [item for item in session.manifest.get("source_file_table", [])
               if isinstance(item, Mapping) and item.get("role") == role]
    if len(entries) != 1:
        raise ReplayError(f"session manifest does not have one {role} source pin")
    item = entries[0]
    observed = raw_pin(item.get("local_path"))
    if (item.get("bytes") != observed["bytes"] or item.get("sha256") != observed["sha256"]
            or Path(item.get("local_path", "")).resolve(strict=True)
            != Path(item.get("declared_path", "")).resolve(strict=True)):
        raise ReplayError(f"session source table changed for {role}")
    return {"path": item["declared_path"], "bytes": observed["bytes"],
            "sha256": observed["sha256"], "self_sha256": None}


def _same_receipt_except_session_source(old_receipt: Mapping[str, Any],
                                        new_receipt: Mapping[str, Any]) -> None:
    old = dict(old_receipt)
    new = dict(new_receipt)
    old.pop("receipt_sha256", None)
    new.pop("receipt_sha256", None)
    old_sources = dict(old.get("source_code_pins", {}))
    new_sources = dict(new.get("source_code_pins", {}))
    if "durable_session" not in old_sources or "durable_session" not in new_sources:
        raise ReplayError("initial selector receipts omit the durable-session source provenance")
    old_sources.pop("durable_session")
    new_sources.pop("durable_session")
    old["source_code_pins"] = old_sources
    new["source_code_pins"] = new_sources
    if old != new:
        raise ReplayError("fresh initial-selector receipt changed a scientific selection/input field")


def _event_link(session, index: int) -> dict[str, Any]:
    path = session.root / "events" / f"event_{index:06d}.json"
    return raw_pin(path, "event_sha256")


def _event_body(session, index: int) -> dict[str, Any]:
    path = session.root / "events" / f"event_{index:06d}.json"
    return read_json(path, f"session event {index}")[0]


def _identity_state(session_module, old_session, new_session) -> tuple[Any, dict[str, Any]]:
    if [event.get("event_type") for event in old_session.events] != EVENT_TYPES_OLD:
        raise ReplayError("old session is not exactly the preserved six-event pair-01 UNKNOWN prefix")
    if [event.get("event_type") for event in new_session.events] != [
            "SESSION_INITIALIZED", "PAIR_ASKED"]:
        raise ReplayError("fresh session must still be an event0/event1-only prefix before replay")
    old_ask = old_session.events[1]
    new_ask = new_session.events[1]
    if (old_ask.get("ask_pair") != new_ask.get("ask_pair")
            or old_ask.get("controller_state_after") != new_ask.get("controller_state_after")
            or old_session.events[0].get("controller_state")
            != new_session.events[0].get("controller_state")):
        raise ReplayError("fresh replay pair/controller/PCG64 state differs from the original ask")
    if old_session.pending_pair is None:
        raise ReplayError("old session lost the pair-01 pending controller state")
    pair = old_session.pending_pair
    fresh_pair = new_session.pending_pair
    if fresh_pair is not None and session_module._plain(fresh_pair) != new_ask.get("ask_pair"):
        raise ReplayError("fresh pending pair differs from its persisted deterministic ask")
    if fresh_pair is None:
        if (not new_session.events or new_session.events[-1].get("event_type") != "PAIR_TOLD"
                or new_session.controller.iteration != 1):
            raise ReplayError("fresh controller has neither the pending pair nor its one completed tell")
    if (pair.iteration != 0 or pair.pair_id != old_ask.get("pair_id")
            or session_module._plain(pair) != new_ask.get("ask_pair")):
        raise ReplayError("fresh deterministic pair is not the exact original pair-01 ask")
    return pair, new_ask


def _verify_process_absent(pid: Any, ticks: Any, label: str) -> dict[str, Any]:
    if type(pid) is not int or pid <= 0 or type(ticks) is not int or ticks <= 0:
        raise ReplayError(f"{label} process identity is malformed")
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return {"pid": pid, "start_ticks": ticks, "identity_status": "ABSENT"}
    raw = stat_path.read_text(encoding="utf-8")
    close = raw.rfind(")")
    if close < 0:
        raise ReplayError(f"cannot parse current proc identity for {label}")
    fields = raw[close + 2:].split()
    if len(fields) <= 19:
        raise ReplayError(f"proc identity fields are incomplete for {label}")
    current_ticks = int(fields[19])
    if current_ticks == ticks:
        raise ReplayError(f"old {label} process is still running; do not replay")
    return {"pid": pid, "start_ticks": ticks, "identity_status": "PID_REUSED_DIFFERENT_START_TICKS",
            "current_start_ticks": current_ticks}


def _capture_process_failure(pair_dir: Path, pair: Any, old_session) -> dict[str, Any]:
    receipt_path = pair_dir / "process_receipt.json"
    receipt, _ = read_json(receipt_path, "old outer pair process receipt")
    self_field, self_value = embedded_self(receipt)
    if (self_field != "receipt_sha256" or receipt.get("schema") != "fit40_spsa_pair_process_receipt_v1"
            or receipt.get("status") != "UNKNOWN" or receipt.get("child_exit_code") != 2
            or receipt.get("relaunch_prohibited") is not True
            or receipt.get("pair_id") != pair.pair_id or receipt.get("iteration") != pair.iteration):
        raise ReplayError("old pair process receipt is not the pinned UNKNOWN/rc2 failure being repaired")
    state = receipt.get("session_readback")
    if (not isinstance(state, Mapping) or state.get("event_count") != 6
            or state.get("matching_pair_tell") is not None
            or state.get("controller_iteration") != 0
            or state.get("pending_pair", {}).get("pair_id") != pair.pair_id):
        raise ReplayError("old process receipt does not bind the untold six-event session prefix")
    child_stdout = raw_pin(pair_dir / "child.stdout")
    child_stderr = raw_pin(pair_dir / "child.stderr")
    process_start = read_json(pair_dir / "process_start.json", "old pair process start")[0]
    launch_intent = read_json(pair_dir / "launch_intent.json", "old pair launch intent")[0]
    if (receipt.get("stdout_pin") != {key: child_stdout[key] for key in ("path", "bytes", "sha256")}
            or receipt.get("stderr_pin") != {key: child_stderr[key] for key in ("path", "bytes", "sha256")}
            or receipt.get("process_start_pin", {}).get("sha256")
            != raw_pin(pair_dir / "process_start.json")["sha256"]
            or receipt.get("intent_pin", {}).get("sha256")
            != raw_pin(pair_dir / "launch_intent.json")["sha256"]):
        raise ReplayError("old process receipt stdout/stderr/start/intent pins differ from captured bytes")
    pids = [
        _verify_process_absent(receipt.get("child_pid"), receipt.get("child_start_ticks"), "pair child"),
    ]
    start_pid = process_start.get("child_pid")
    start_ticks = process_start.get("child_start_ticks")
    if (start_pid, start_ticks) != (receipt.get("child_pid"), receipt.get("child_start_ticks")):
        raise ReplayError("pair process-start identity differs from the captured pair-child receipt")
    coordinator_dir = pair_dir.parents[1]
    coordinator_manifest_path = coordinator_dir / "coordinator_manifest.json"
    coordinator_intent_path = coordinator_dir / "coordinator_start_intent.json"
    coordinator_start_path = coordinator_dir / "coordinator_start_receipt.json"
    coordinator_manifest, _ = read_json(coordinator_manifest_path, "outer coordinator manifest")
    manifest_field, manifest_self = embedded_self(coordinator_manifest)
    coordinator_intent, _ = read_json(coordinator_intent_path, "outer coordinator start intent")
    intent_field, intent_self = embedded_self(coordinator_intent)
    coordinator_start, _ = read_json(coordinator_start_path, "outer coordinator start receipt")
    coordinator_start_field, coordinator_start_self = embedded_self(coordinator_start)
    if (manifest_field != "coordinator_sha256" or intent_field != "intent_sha256"
            or coordinator_start_field != "receipt_sha256"
            or coordinator_start.get("coordinator_id") != receipt.get("coordinator_id")):
        raise ReplayError("outer coordinator start receipt does not bind the failed pair coordinator")
    if (coordinator_manifest.get("session_dir") != str(old_session.root)
            or coordinator_manifest.get("session_id") != old_session.manifest.get("session_id")
            or coordinator_manifest.get("coordinator_id") != receipt.get("coordinator_id")
            or coordinator_intent.get("coordinator_manifest_pin", {}).get("sha256")
            != raw_pin(coordinator_manifest_path)["sha256"]):
        raise ReplayError("outer coordinator manifest/intent do not bind the preserved old session")
    pids.append(_verify_process_absent(coordinator_start.get("child_pid"),
                                       coordinator_start.get("child_start_ticks"),
                                       "outer coordinator"))
    return {
        "outer_process_receipt": raw_pin(receipt_path, "receipt_sha256"),
        "outer_launch_intent": raw_pin(pair_dir / "launch_intent.json"),
        "outer_process_start": raw_pin(pair_dir / "process_start.json"),
        "outer_coordinator_manifest": raw_pin(coordinator_manifest_path, "coordinator_sha256"),
        "outer_coordinator_start_intent": raw_pin(coordinator_intent_path, "intent_sha256"),
        "outer_coordinator_start_receipt": raw_pin(coordinator_start_path, "receipt_sha256"),
        "outer_child_stdout": child_stdout,
        "outer_child_stderr": child_stderr,
        "outer_status": "UNKNOWN",
        "outer_child_exit_code": 2,
        "old_process_identity_readback": pids,
        "old_event_count": len(old_session.events),
        "old_pair_tell": None,
        "relaunch_prohibited": True,
    }


def _point_file_links(session_module, record: Mapping[str, Any], proposal: Any,
                      session, cross_npz_paths: set[str], cross_map_paths: set[str]) -> dict[str, Any]:
    doc = record.get("point_doc")
    if not isinstance(doc, Mapping) or doc.get("terminal_status") != "COMPLETE":
        raise ReplayError(f"old {proposal.side} point is not a COMPLETE terminal")
    point_entry = doc.get("point_entry")
    if not isinstance(point_entry, Mapping):
        raise ReplayError("old point terminal has no point-entry mapping")
    point_files = session.inputs.files
    # The point-specific file table is independently validated and merged below.
    point_files = session_module.initial_ingest._file_table({"files": doc.get("file_entries", [])})
    npz_by_declared = {pin.declared_path: pin for pin in point_files.values()
                       if pin.role == "raw_matrix_npz"}
    maps_by_id = point_entry.get("official_source_frame_map_file_ids")
    raw_npz_by_row = doc.get("raw_npz_by_row")
    candidate_ids = set(raw_npz_by_row) if isinstance(raw_npz_by_row, Mapping) else set()
    if not isinstance(maps_by_id, Mapping) or set(maps_by_id) != candidate_ids or len(candidate_ids) != 24:
        raise ReplayError("complete point must bind exactly 24 official NPZ and frame-map rows")
    evidence = []
    for row_id in sorted(candidate_ids):
        npz_ref = raw_npz_by_row[row_id]
        npz_pin = npz_by_declared.get(npz_ref.get("path"))
        if (npz_pin is None or npz_pin.byte_count != npz_ref.get("bytes")
                or npz_pin.sha256 != npz_ref.get("sha256")):
            raise ReplayError(f"raw NPZ does not resolve for candidate row {row_id}")
        frame_id = maps_by_id[row_id]
        frame_pin = point_files.get(frame_id)
        if (frame_pin is None or frame_pin.role != "official_source_frame_map"
                or frame_pin.pin_name != row_id):
            raise ReplayError(f"official source frame map does not resolve for candidate row {row_id}")
        npz_local = str(Path(npz_pin.local_path).resolve(strict=True))
        map_local = str(Path(frame_pin.local_path).resolve(strict=True))
        if npz_local in cross_npz_paths or map_local in cross_map_paths:
            raise ReplayError("old plus/minus point evidence reuses a candidate NPZ or frame map")
        cross_npz_paths.add(npz_local)
        cross_map_paths.add(map_local)
        evidence.append({
            "row_id": row_id,
            "raw_matrix_npz": {
                "declared_path": npz_pin.declared_path, "local_path": npz_pin.local_path,
                "bytes": npz_pin.byte_count, "sha256": npz_pin.sha256,
                "array_refs": dict(npz_ref["array_refs"]),
            },
            "official_source_frame_map": {
                "file_id": frame_pin.file_id, "pin_name": frame_pin.pin_name,
                "declared_path": frame_pin.declared_path, "local_path": frame_pin.local_path,
                "bytes": frame_pin.byte_count, "sha256": frame_pin.sha256,
            },
        })
    child_receipts = []
    expected_models = list(session.inputs.family_selection.family.roster.model_ids)
    blocks = point_entry.get("blocks")
    if not isinstance(blocks, list) or [b.get("model_id") for b in blocks] != expected_models:
        raise ReplayError("point terminal model blocks changed order")
    for block in blocks:
        if (block.get("terminal_status") != "COMPLETE"
                or block.get("status_counts") != {"FAILED": 0, "INCOMPLETE": 0, "PASS": 8}
                or len(block.get("row_ids", [])) != 8):
            raise ReplayError("old terminal block is not complete with its exact eight-row denominator")
        pin = point_files.get(block.get("child_execution_receipt_file_id"))
        if pin is None or pin.role != "child_execution_receipt":
            raise ReplayError("terminal block child receipt is not present in its pinned file table")
        child, _raw = read_json(pin.local_path, "native model child receipt")
        field, self_value = embedded_self(child)
        if (field != "receipt_sha256" or child.get("schema") != "fit40_spsa_point_child_execution_receipt_v1"
                or type(child.get("child_exit_code")) is not int or child["child_exit_code"] != 0
                or self_value != child.get("receipt_sha256")):
            raise ReplayError("old model child receipt does not prove successful numeric exit")
        child_receipts.append({"model_id": block["model_id"],
                               "pin": {"path": pin.declared_path, "bytes": pin.byte_count,
                                       "sha256": pin.sha256, "self_sha256": self_value},
                               "child_exit_code": 0})
    return {
        "side": proposal.side,
        "proposal_id": proposal.proposal_id,
        "point_entry": dict(point_entry),
        "point_entry_pin": dict(record["point_pin"]),
        "candidate_row_count": len(candidate_ids),
        "official_raw_npz_count": len(evidence),
        "official_source_frame_map_count": len(evidence),
        "candidate_evidence": evidence,
        "model_child_receipts": child_receipts,
    }


def _verify_outer_packet(pair_dir: Path, pair: Any, old_session) -> dict[str, Any]:
    packet_path = pair_dir / "packet" / "packet_index.json"
    packet, _ = read_json(packet_path, "original pair packet index")
    field, packet_self = embedded_self(packet)
    if (field != "packet_sha256" or packet.get("schema") != "fit40_spsa_stagea_execution_packet_v1"
            or packet.get("planned_row_total") != 48 or packet_self != packet.get("packet_sha256")
            or packet.get("scope", {}).get("pair_id") != pair.pair_id):
        raise ReplayError("original packet does not bind the exact 48-row deterministic pair")
    expected = {"plus": pair.plus, "minus": pair.minus}
    for item in packet.get("plans", []):
        side = item.get("label")
        proposal = expected.get(side)
        if proposal is None:
            raise ReplayError("original packet contains an unexpected proposal side")
        if item.get("run_id") != old_session.point_entries[(proposal.proposal_id, side)]["point_doc"]["point_entry"].get("run_id"):
            raise ReplayError("packet run ID differs from actual complete terminal point")
        plan_path = item.get("plan", {}).get("path")
        plan_link = raw_pin(plan_path, "plan_sha256")
        if (plan_link["bytes"] != item["plan"]["bytes"]
                or plan_link["sha256"] != item["plan"]["sha256"]
                or plan_link["self_sha256"] != item["plan_self_sha256"]):
            raise ReplayError("original pair packet plan pin changed")
    if {item.get("label") for item in packet.get("plans", [])} != {"plus", "minus"}:
        raise ReplayError("original pair packet does not bind exactly plus and minus plans")
    return raw_pin(packet_path, "packet_sha256")


def _replay_adoption_declarations(session_module, pair: Any,
                                  old_event_bodies: list[Mapping[str, Any]],
                                  old_events: list[Mapping[str, Any]],
                                  point_summaries: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(old_event_bodies) != 6 or len(old_events) != 6:
        raise ReplayError("old session must provide exactly six pinned pair-prefix events")
    declarations = []
    for side, proposal, old_intent_index, old_terminal_index, fresh_intent_index, fresh_terminal_index in (
            ("plus", pair.plus, 2, 3, 2, 3),
            ("minus", pair.minus, 4, 5, 4, 5)):
        old_intent = old_event_bodies[old_intent_index]
        old_terminal = old_event_bodies[old_terminal_index]
        if (old_intent.get("event_type") != "POINT_DISPATCH_INTENT"
                or old_intent.get("proposal_id") != proposal.proposal_id
                or old_intent.get("pair_id") != pair.pair_id
                or old_intent.get("side") != side
                or old_intent.get("proposal") != session_module._plain(proposal)
                or old_terminal.get("event_type") != "POINT_TERMINAL"
                or old_terminal.get("proposal_id") != proposal.proposal_id
                or old_terminal.get("pair_id") != pair.pair_id
                or old_terminal.get("side") != side
                or old_terminal.get("terminal_status") != "COMPLETE"):
            raise ReplayError(f"old {side} intent/terminal events do not bind this exact proposal")
        summary = point_summaries[side]
        if (old_terminal.get("point_entry") != summary.get("point_entry")
                or old_terminal.get("point_entry_pin") != summary.get("point_entry_pin")):
            raise ReplayError(f"old {side} terminal event and point document disagree")
        declarations.append({
            "side": side,
            "proposal_id": proposal.proposal_id,
            "pair_id": pair.pair_id,
            "iteration": proposal.iteration,
            "fresh_session_dispatch_intent_event_index": fresh_intent_index,
            "fresh_session_point_terminal_event_index": fresh_terminal_index,
            "fresh_intent_semantics": "readback_adoption_only_no_child_or_model_process",
            "old_session_dispatch_intent_event": dict(old_events[old_intent_index]),
            "old_session_point_terminal_event": dict(old_events[old_terminal_index]),
            "old_point_entry_pin": dict(summary["point_entry_pin"]),
            "old_point_entry": dict(summary["point_entry"]),
            "old_model_child_receipts": [dict(item) for item in summary["model_child_receipts"]],
        })
    return declarations


def _bind_fresh_replay_events(session, declarations: list[Mapping[str, Any]],
                              pair: Any, session_module) -> list[dict[str, Any]]:
    bound = []
    for declaration in declarations:
        side = declaration["side"]
        proposal = pair.plus if side == "plus" else pair.minus
        intent_index = declaration["fresh_session_dispatch_intent_event_index"]
        terminal_index = declaration["fresh_session_point_terminal_event_index"]
        intent = _event_body(session, intent_index)
        terminal = _event_body(session, terminal_index)
        if (intent.get("event_type") != "POINT_DISPATCH_INTENT"
                or intent.get("proposal_id") != proposal.proposal_id
                or intent.get("point_id") != proposal.method_id
                or intent.get("pair_id") != pair.pair_id
                or intent.get("iteration") != proposal.iteration
                or intent.get("side") != side
                or intent.get("proposal") != session_module._plain(proposal)
                or terminal.get("event_type") != "POINT_TERMINAL"
                or terminal.get("proposal_id") != proposal.proposal_id
                or terminal.get("point_id") != proposal.method_id
                or terminal.get("pair_id") != pair.pair_id
                or terminal.get("iteration") != proposal.iteration
                or terminal.get("side") != side
                or terminal.get("terminal_status") != "COMPLETE"
                or terminal.get("point_entry") != declaration["old_point_entry"]):
            raise ReplayError(f"fresh {side} intent/terminal does not import the exact old terminal")
        bound.append({
            **dict(declaration),
            "fresh_session_dispatch_intent_event": _event_link(session, intent_index),
            "fresh_session_point_terminal_event": _event_link(session, terminal_index),
            "fresh_point_entry_pin": dict(session.point_entries[(proposal.proposal_id, side)]["point_pin"]),
        })
    return bound


def _seal_once(path: Path, document: Mapping[str, Any], self_field: str) -> dict[str, Any]:
    path = path.absolute()
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ReplayError("receipt parent must already exist as a non-symlink directory")
    sealed = dict(document)
    sealed[self_field] = sha(canonical(sealed))
    raw = (json.dumps(sealed, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + "\n").encode("utf-8")
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        actual, actual_raw = read_json(path, "existing create-only receipt")
        if actual_raw != raw:
            raise ReplayError(f"refusing to replace a non-identical receipt: {path}")
        return raw_pin(path, self_field)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(str(path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    pin = raw_pin(path, self_field)
    if pin["bytes"] != len(raw) or pin["sha256"] != sha(raw):
        raise ReplayError("create-only receipt readback differs from emitted bytes")
    return pin


class ExistingTerminalExecutor:
    """Adapter whose execute callback only imports a prevalidated old terminal."""

    def __init__(self, session_module, fresh_session, outcomes: Mapping[tuple[str, str], Any]):
        self.session_module = session_module
        self.fresh_session = fresh_session
        self.outcomes = dict(outcomes)

    def _terminal(self, proposal):
        outcome = self.outcomes.get((proposal.proposal_id, proposal.side))
        if outcome is None or outcome.point_entry is None:
            raise ReplayError("no prevalidated old terminal exists for the exact pending proposal")
        return outcome

    def readback(self, proposal):
        key = (proposal.proposal_id, proposal.side)
        if key not in self.outcomes:
            raise ReplayError("fresh session requested a proposal outside the old pair")
        if key in self.fresh_session.dispatch_intents:
            return self._terminal(proposal)
        return self.session_module.PointOutcome(disposition="ABSENT")

    def execute(self, proposal):
        # This is reached only after the fresh durable session wrote its own
        # POINT_DISPATCH_INTENT. It returns old verified bytes; no child,
        # renderer, scorer, selector, or dispatcher is invoked here.
        return self._terminal(proposal)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise ReplayError("replay process must run with CUDA hidden")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for receipt_path, label in ((Path(args.lineage_path), "lineage"),
                                (Path(args.completion_path), "completion")):
        if receipt_path.parent.is_symlink() or not receipt_path.parent.is_dir():
            raise ReplayError(f"{label} receipt parent must already exist as a non-symlink directory")
        if receipt_path.exists() or receipt_path.is_symlink():
            raise ReplayError(f"one-shot {label} receipt target must be absent before replay")
    project_root = Path(args.project_root).resolve(strict=True)
    namespace_paths = configure_project_imports(project_root)
    old_source = Path(args.old_session_source).resolve(strict=True)
    new_source = Path(args.new_session_source).resolve(strict=True)
    old_module = load_module(old_source, "fit40_replay_old_session_attempt05")
    new_module = load_module(new_source, "fit40_replay_new_session_attempt06")
    if (Path(old_module.__file__).resolve(strict=True) != old_source
            or Path(new_module.__file__).resolve(strict=True) != new_source):
        raise ReplayError("durable session modules were not loaded from the explicit source files")

    old_session = old_module.DurableSPSASession.resume(session_dir=args.old_session_dir)
    fresh_session = new_module.DurableSPSASession.resume(session_dir=args.new_session_dir)
    pair, fresh_ask = _identity_state(new_module, old_session, fresh_session)
    if old_session.manifest["initial_selector_receipt"]["path"] == fresh_session.manifest["initial_selector_receipt"]["path"]:
        raise ReplayError("replay must bind distinct immutable initial-receipt02 and receipt03 paths")
    for key in ("scientific_input_pins", "scientific_input_pins_sha256", "spsa_seed_method_id",
                "initial_selection_status", "initial_theta", "source_roster",
                "reference_specs", "source_snapshot_lineage", "composed_family_plan",
                "execution_package_manifest", "initial_ingest_index"):
        if old_session.manifest.get(key) != fresh_session.manifest.get(key):
            raise ReplayError(f"fresh session changed frozen scientific/session input field: {key}")
    _same_receipt_except_session_source(old_session.inputs.receipt, fresh_session.inputs.receipt)

    pair_dir = Path(args.old_pair_dir).resolve(strict=True)
    old_packet_pin = _verify_outer_packet(pair_dir, pair, old_session)
    outer_failure = _capture_process_failure(pair_dir, pair, old_session)
    outcomes = {}
    point_summaries = {}
    candidate_npz_paths: set[str] = set()
    candidate_map_paths: set[str] = set()
    new_evaluations = {}
    for proposal in (pair.plus, pair.minus):
        key = (proposal.proposal_id, proposal.side)
        record = old_session.point_entries.get(key)
        if not isinstance(record, Mapping) or record.get("event", {}).get("terminal_status") != "COMPLETE":
            raise ReplayError(f"old {proposal.side} point is not an actual COMPLETE terminal")
        doc = record.get("point_doc")
        # Revalidate the exact old point under the corrected consumer before
        # any fresh-session event is appended.
        if key not in fresh_session.point_entries:
            fresh_session._validate_point_document(proposal, doc)
        else:
            if fresh_session.point_entries[key].get("point_doc") != doc:
                raise ReplayError("fresh session already contains a different point document")
        fresh_session._merge_point_files(doc["file_entries"])
        point_summaries[proposal.side] = _point_file_links(
            new_module, record, proposal, old_session, candidate_npz_paths, candidate_map_paths)
        evaluation = fresh_session._load_official_evaluation(proposal, doc)
        if (not fresh_session._validate_evaluation_matches(evaluation, proposal)
                or len([row for row in evaluation.rows if row.method_id == proposal.method_id]) != 24):
            raise ReplayError("official score evaluation does not reconstruct the exact 24 candidate rows")
        fresh_session._validate_evaluation_rows(evaluation, proposal)
        new_evaluations[proposal.side] = evaluation
        outcomes[key] = new_module.PointOutcome(
            disposition="TERMINAL", terminal_status="COMPLETE",
            point_entry=dict(doc["point_entry"]),
            files=tuple(dict(item) for item in doc["file_entries"]),
            raw_npz_by_row={key: dict(value) for key, value in doc["raw_npz_by_row"].items()},
            reason="verified existing complete native terminal imported into corrected session",
        )

    # The pure preparation path must not import the native executor/renderer.
    if any("point_execution" in name or "stagea_adapter" in name for name in sys.modules):
        raise ReplayError("native point executor/adapter unexpectedly loaded during data preflight")
    if len(candidate_npz_paths) != 48 or len(candidate_map_paths) != 48:
        raise ReplayError("old pair does not provide 48 unique NPZs and 48 unique source frame maps")

    old_receipt = old_session.inputs.receipt
    new_receipt = fresh_session.inputs.receipt
    source_links = {
        "old_durable_session_source": _module_source_pin(old_session, "durable_session_source"),
        "replay_durable_session_source": _module_source_pin(fresh_session, "durable_session_source"),
        "replay_driver_source": raw_pin(Path(__file__).resolve(strict=True)),
    }
    old_events = [_event_link(old_session, index) for index in range(len(old_session.events))]
    new_events = [_event_link(fresh_session, index) for index in range(min(2, len(fresh_session.events)))]
    old_event_bodies = [_event_body(old_session, index) for index in range(len(old_session.events))]
    if (len(old_event_bodies) != 6 or old_event_bodies[-1].get("event_type") != "POINT_TERMINAL"
            or fresh_ask.get("event_type") != "PAIR_ASKED"):
        raise ReplayError("old or fresh session event chain is not the expected pair prefix")
    replay_adoption_declarations = _replay_adoption_declarations(
        new_module, pair, old_event_bodies, old_events, point_summaries)

    lineage = {
        "schema": LINEAGE_SCHEMA,
        "status": "READY_TO_IMPORT_VERIFIED_COMPLETE_PAIR_TERMINALS",
        "continuation_kind": "existing_pair_terminal_replay_no_native_dispatch",
        "pair_id": pair.pair_id,
        "iteration": pair.iteration,
        "ask_pair": new_module._plain(pair),
        "ask_pair_sha256": sha(canonical(new_module._plain(pair))),
        "old_controller_state_after_ask": old_session.events[1]["controller_state_after"],
        "fresh_controller_state_after_ask": fresh_ask["controller_state_after"],
        "old_session_manifest": raw_pin(Path(args.old_session_dir) / "session_manifest.json", "manifest_sha256"),
        "old_session_events": old_events,
        "old_point_entries": {
            side: dict(point_summaries[side]["point_entry_pin"]) for side in ("plus", "minus")
        },
        "replay_adoption_intents": replay_adoption_declarations,
        "old_initial_selector_receipt": verify_link(old_session.manifest["initial_selector_receipt"],
                                                     "old initial selector receipt"),
        "fresh_initial_selector_receipt": verify_link(fresh_session.manifest["initial_selector_receipt"],
                                                       "fresh initial selector receipt"),
        "composed_family_plan": verify_link(fresh_session.manifest["composed_family_plan"],
                                             "composed family plan"),
        "execution_package_manifest": verify_link(fresh_session.manifest["execution_package_manifest"],
                                                   "execution package manifest"),
        "initial_ingest_index": verify_link(fresh_session.manifest["initial_ingest_index"],
                                             "initial ingest index"),
        "old_outer_pair_process_failure": outer_failure,
        "original_pair_packet_index": old_packet_pin,
        "point_terminals": point_summaries,
        "official_candidate_evidence_counts": {
            "candidate_rows": 48, "official_raw_npz": 48,
            "official_source_frame_maps": 48,
            "model_child_receipts_with_numeric_rc0": 6,
        },
        "session_lineage": {
            "old_session_id": old_session.manifest["session_id"],
            "fresh_session_id": fresh_session.manifest["session_id"],
            "old_event_count_before_replay": len(old_session.events),
            "fresh_event_count_at_lineage_creation": 2,
            "fresh_session_manifest": raw_pin(Path(args.new_session_dir) / "session_manifest.json",
                                               "manifest_sha256"),
            "fresh_session_initial_events": new_events,
        },
        "source_pins": source_links,
        "runtime_effects": {
            "native_dispatcher_imported": False,
            "new_model_processes_started": 0,
            "new_audio_or_video_rendered": False,
            "new_score_npz_generated": False,
            "existing_official_npz_revalidated": 48,
        },
    }
    lineage_path = Path(args.lineage_path)
    lineage_pin = _seal_once(lineage_path, lineage, "receipt_sha256")

    # Re-read the READY receipt immediately before the normal durable-session
    # API can append intents or terminal entries.
    stored_lineage, _ = read_json(lineage_path, "pair replay lineage receipt")
    if (embedded_self(stored_lineage) != ("receipt_sha256", lineage_pin["self_sha256"])
            or stored_lineage.get("status") != "READY_TO_IMPORT_VERIFIED_COMPLETE_PAIR_TERMINALS"
            or stored_lineage.get("ask_pair") != new_module._plain(pair)):
        raise ReplayError("create-only replay lineage failed immediate readback")

    # This is deliberately one-shot. The preflight above accepts only a fresh
    # event0/event1 session; a partially persisted replay must be inspected and
    # handled as a separate recovery, never resumed through this driver.
    executor = ExistingTerminalExecutor(new_module, fresh_session, outcomes)
    tell = fresh_session.run_pending_pair(executor)
    if tell.pair_id != pair.pair_id:
        raise ReplayError("fresh session tell does not match the one original pair")
    if not fresh_session.events or fresh_session.events[-1].get("event_type") != "PAIR_TOLD":
        raise ReplayError("durable session returned without a PAIR_TOLD event")
    tell_event = fresh_session.events[-1]

    if (tell_event.get("pair_id") != pair.pair_id or tell_event.get("iteration") != 0
            or tell_event.get("tell_result", tell_event.get("journal_entry", {}).get("tell_result", {})).get("pair_id")
            not in (None, pair.pair_id)):
        raise ReplayError("persisted tell event does not bind the exact replayed pair")
    if [event.get("event_type") for event in fresh_session.events] != EVENT_TYPES_FRESH_PREFIX:
        raise ReplayError("fresh replay did not produce the exact seven-event first-pair prefix")
    if (fresh_session.controller.iteration != 1 or fresh_session.pending_pair is not None
            or fresh_session.halted):
        raise ReplayError("replayed pair did not leave the expected iteration-1 live controller state")
    tell_body = tell_event.get("journal_entry")
    if not isinstance(tell_body, Mapping) or tell_body.get("pair_id") != pair.pair_id:
        raise ReplayError("PAIR_TOLD event omits the exact self-hashed journal entry")
    tell_result = tell_event.get("tell_result")
    if (not isinstance(tell_result, Mapping) or tell_result.get("status") != "UPDATED"
            or tell_body.get("tell_result") != tell_result):
        raise ReplayError("completed replay prefix did not produce the exact UPDATED tell record")
    replay_adoption_intents = _bind_fresh_replay_events(
        fresh_session, replay_adoption_declarations, pair, new_module)
    fresh_terminal_pins = {}
    for side, proposal in (("plus", pair.plus), ("minus", pair.minus)):
        record = fresh_session.point_entries.get((proposal.proposal_id, side))
        if not isinstance(record, Mapping) or record["event"].get("terminal_status") != "COMPLETE":
            raise ReplayError(f"fresh replay session does not have a complete {side} terminal")
        fresh_terminal_pins[side] = dict(record["point_pin"])

    completion = {
        "schema": COMPLETION_SCHEMA,
        "status": "COMPLETE_PAIR_REPLAY_PREFIX",
        "continuation_kind": "existing_pair_terminal_replay_no_native_dispatch",
        "pair_id": pair.pair_id,
        "iteration": 0,
        "ask_pair": new_module._plain(pair),
        "tell_result": dict(tell_result),
        "controller_state_after": tell_event["controller_state_after"],
        "tell_journal_entry": dict(tell_body),
        "new_session_manifest": raw_pin(Path(args.new_session_dir) / "session_manifest.json",
                                         "manifest_sha256"),
        "new_session_events": [_event_link(fresh_session, index)
                                for index in range(len(fresh_session.events))],
        "tell_event": _event_link(fresh_session, 6),
        "point_entries": fresh_terminal_pins,
        "replay_adoption_intents": replay_adoption_intents,
        "lineage_receipt": verify_link(lineage_pin, "READY replay lineage receipt"),
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
        "source_pins": source_links,
        "native_work_for_replayed_prefix": {
            "new_model_processes_started": 0,
            "existing_candidate_rows_reused_once": 48,
            "official_npz_revalidated": 48,
            "overall_pair_child_exit_code": None,
        },
    }
    completion_pin = _seal_once(Path(args.completion_path), completion, "receipt_sha256")
    return {
        "status": completion["status"],
        "lineage_receipt": lineage_pin,
        "completion_receipt": completion_pin,
        "pair_id": pair.pair_id,
        "tell_result": completion["tell_result"],
        "controller_iteration": fresh_session.controller.iteration,
        "pending_pair": fresh_session.pending_pair is not None,
        "session_event_count": len(fresh_session.events),
        "native_processes_started": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--old-session-source", required=True, type=Path)
    parser.add_argument("--new-session-source", required=True, type=Path)
    parser.add_argument("--old-session-dir", required=True, type=Path)
    parser.add_argument("--new-session-dir", required=True, type=Path)
    parser.add_argument("--old-pair-dir", required=True, type=Path)
    parser.add_argument("--lineage-path", required=True, type=Path)
    parser.add_argument("--completion-path", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run(args)
        print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "PAIR_REPLAY_FAILED", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
