#!/usr/bin/env python3
"""Durably supervise exactly one SPSA pair per child process, up to 20 pairs.

This is orchestration only.  It does not create the initial selector receipt or
session, modify the selector, or evaluate a proposal itself.  Every pair launch
has a create-only intent and process-start receipt before waiting for its child.
An intent without a terminal receipt is read back and becomes UNKNOWN; it is
never replayed automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence


PACKAGE = "scripts.experiments.cross_tfg_audio_head_20261003."
BOOTSTRAP_REL = "scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_bootstrap_cli_attempt14.py"
SESSION_REL = "scripts/experiments/cross_tfg_audio_head_20261003/fit40_durable_spsa_session_attempt04.py"
BOOTSTRAP_PIN = (28_519, "ca684649dd68ddc5390c441f77e70b41adbc7767d59d7122dffc8e0ede3f2408")
SESSION_PIN = (89_950, "9e7c6fe51f8426196770f9171a61c37c6444690f902f83466b1e0d5a703a6709")
DISPATCH_MOUNT_ROOT = Path("/dev/shm")
DISPATCH_ROOT_NAME = "fit40_joint_spsa_execution_recovery_attempt03"
RECOVERY_LINEAGE_SCHEMA = "fit40_durable_spsa_partial_native_recovery_lineage_v1"
RECOVERY_LINEAGE_STATUS = "VERIFIED_PARTIAL_NATIVE_W2L_BLOCK_ADOPTION"
# Filled only after the create-only lineage receipt has been read back.
RECOVERY_LINEAGE_PIN = (0, "", "")
RECOVERY_ADOPTION_SCHEMA = "fit40_spsa_partial_native_adoption_v1"
RECOVERY_ADOPTION_PIN = (0, "", "")
OLD_SESSION_ID = "fit40_spsa_recovery05_package11"
FRESH_SESSION_ID = "fit40_spsa_recovery06_package11"
OLD_PAIR_ID = "e5fa43c586f7665835da25c62d2e0acb3260f9662a2412baf913e2a688d73fc3"
OLD_PLUS_PROPOSAL_ID = "e869a1bb8f667ee867ecfa7f8cc5a5c4cca7d1a218c7b8c3438df52bbc9cd9c8"
OLD_MINUS_PROPOSAL_ID = "dab011a03e16c1b912c47f227e16443a8aef13afbeefd72391c734a0749178c8"
OLD_ROOT_QC_SELF_SHA256 = "9f36290560334280d71571c50409a08327d89e8c79ae0446faca99a20677e9ce"
COORDINATOR_SCHEMA = "fit40_spsa_twenty_pair_coordinator_v2"
START_INTENT_SCHEMA = "fit40_spsa_coordinator_start_intent_v2"
START_RECEIPT_SCHEMA = "fit40_spsa_coordinator_start_receipt_v2"
PAIR_INTENT_SCHEMA = "fit40_spsa_pair_launch_intent_v2"
PAIR_START_SCHEMA = "fit40_spsa_pair_process_start_v1"
PAIR_RECEIPT_SCHEMA = "fit40_spsa_pair_process_receipt_v1"
PAIR_READBACK_SCHEMA = "fit40_spsa_pair_unknown_readback_v1"
TERMINAL_STATUSES = {"COMPLETE_UPDATED", "NO_UPDATE", "UNKNOWN"}
MAX_PAIRS = 20
SESSION_MANIFEST_SCHEMA = "fit40_durable_spsa_session_manifest_v2"


class CoordinatorError(RuntimeError):
    """Invalid source, durable state, or child execution evidence."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return _sha(_canonical(body))


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(field, None)
    result[field] = _sha(_canonical(result))
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _regular_bytes(path: Path, label: str, *, allow_symlink: bool = False) -> bytes:
    if not path.is_absolute() or (path.is_symlink() and not allow_symlink) or not path.is_file():
        raise CoordinatorError("%s must be an absolute regular file: %s" % (label, path))
    target = path.resolve(strict=True) if allow_symlink else path
    mode = target.stat(follow_symlinks=False).st_mode
    if not stat.S_ISREG(mode):
        raise CoordinatorError("%s is not a regular file: %s" % (label, path))
    return target.read_bytes()


def _file_pin(path: Path, *, allow_symlink: bool = False) -> dict[str, Any]:
    raw = _regular_bytes(path, "pinned file", allow_symlink=allow_symlink)
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": len(raw), "sha256": _sha(raw)}


def _self_file_pin(path: Path, label: str, field: str) -> dict[str, Any]:
    _value, _raw, pin = _read_self_json(path, label, field)
    return pin


def _read_self_json(path: Path, label: str, field: str) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    raw = _regular_bytes(path, label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise CoordinatorError("%s is not UTF-8 JSON" % label) from exc
    if not isinstance(value, dict) or value.get(field) != _self_hash(value, field):
        raise CoordinatorError("%s self hash is missing or invalid" % label)
    pin = {"path": str(path.resolve(strict=True)), "bytes": len(raw),
           "sha256": _sha(raw), "self_sha256": value[field]}
    return value, raw, pin


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise CoordinatorError("refusing to overwrite durable artifact: %s" % path) from exc
    try:
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_dir(path.parent)
    except Exception:
        raise


def _fsync_dir(path: Path) -> None:
    if os.name != "posix":
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_exclusive(path: Path, value: Mapping[str, Any], field: str) -> dict[str, Any]:
    document = _seal(value, field)
    raw = (json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + "\n").encode("utf-8")
    _write_exclusive(path, raw)
    return {"path": str(path.resolve(strict=True)), "bytes": len(raw),
            "sha256": _sha(raw), "self_sha256": document[field]}


def _source_closure(repo_root: Path) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for relative, expected in ((BOOTSTRAP_REL, BOOTSTRAP_PIN), (SESSION_REL, SESSION_PIN)):
        path = repo_root / relative
        pin = _file_pin(path)
        if (pin["bytes"], pin["sha256"]) != expected:
            raise CoordinatorError("frozen coordinator dependency changed: %s" % relative)
        observed[relative] = pin
    return observed


def _validate_recovery_lineage(path_value: str | Path, *, session_root: Path,
                               session_manifest: Mapping[str, Any],
                               session_manifest_pin: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the immutable W2L partial block and deterministic fresh-session lineage."""
    path = Path(path_value)
    lineage, _raw, pin = _read_self_json(path, "partial-native recovery lineage", "lineage_sha256")
    if (lineage.get("schema") != RECOVERY_LINEAGE_SCHEMA
            or lineage.get("status") != RECOVERY_LINEAGE_STATUS
            or (pin.get("bytes"), pin.get("sha256"), pin.get("self_sha256")) != RECOVERY_LINEAGE_PIN):
        raise CoordinatorError("partial-native recovery lineage schema/status/pin is invalid")
    prior = lineage.get("prior_attempt")
    fresh = lineage.get("fresh_session")
    files = prior.get("source_file_pins") if isinstance(prior, Mapping) else None
    adoption_pin = lineage.get("partial_native_adoption_receipt_pin")
    if (not isinstance(prior, Mapping) or not isinstance(fresh, Mapping)
            or not isinstance(files, Mapping) or not isinstance(adoption_pin, Mapping)):
        raise CoordinatorError("partial-native lineage omits the prior/fresh session or adoption pins")

    job_control = Path("/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/job_control")
    old_session_root = job_control / "fit40_durable_spsa_session_attempt05"
    old_pair_dir = job_control / "fit40_spsa_twenty_pair_coordinator_attempt07" / "pairs" / "pair_01"
    old_packet_dir = old_pair_dir / "packet"
    old_dispatch_root = Path("/dev/shm/fit40_joint_spsa_execution_attempt07/pairs/pair_01/dispatch_root")
    old_point_root = old_dispatch_root / "points" / (OLD_PLUS_PROPOSAL_ID + "_plus")
    old_block_root = old_point_root / "blocks" / "wav2lip_gan_bac9a81e"
    expected_paths = {
        "old_session_manifest": old_session_root / "session_manifest.json",
        "old_event0": old_session_root / "events" / "event_000000.json",
        "old_event1": old_session_root / "events" / "event_000001.json",
        "old_event2": old_session_root / "events" / "event_000002.json",
        "old_pair_launch_intent": old_pair_dir / "launch_intent.json",
        "old_pair_process_receipt": old_pair_dir / "process_receipt.json",
        "old_packet_index": old_packet_dir / "packet_index.json",
        "old_plus_plan": old_packet_dir / "plus" / "plan.json",
        "old_plus_planned_rows": old_packet_dir / "plus" / "planned_rows.jsonl",
        "old_dispatch_intent": old_point_root / "dispatch_intent.json",
        "old_candidate_w2l": old_point_root / "shared_candidate_application" / "candidate_rows" / "wav2lip_gan_bac9a81e.json",
        "old_candidate_muse": old_point_root / "shared_candidate_application" / "candidate_rows" / "musetalk_fixed_portrait_attempt16.json",
        "old_candidate_ditto": old_point_root / "shared_candidate_application" / "candidate_rows" / "ditto_hubert_fixed_portrait_attempt18.json",
        "old_block_request": old_block_root / "block_request.json",
        "old_process_started": old_block_root / "process_started.json",
        "old_process_terminal": old_block_root / "process_terminal.json",
        "old_child_receipt": old_block_root / "child_execution_receipt.json",
        "old_run_status": old_block_root / "stagea_output" / "run_status.json",
        "old_result_manifest": old_block_root / "stagea_output" / "result_manifest.json",
        "old_results_jsonl": old_block_root / "stagea_output" / "results.jsonl",
        "fresh_session_manifest": session_root / "session_manifest.json",
        "fresh_event0": session_root / "events" / "event_000000.json",
    }
    self_fields = {
        "old_session_manifest": "manifest_sha256", "old_event0": "event_sha256",
        "old_event1": "event_sha256", "old_event2": "event_sha256",
        "old_pair_launch_intent": "intent_sha256",
        "old_pair_process_receipt": "receipt_sha256",
        "old_packet_index": "packet_sha256", "old_plus_plan": "plan_sha256",
        "old_dispatch_intent": "intent_sha256", "old_candidate_w2l": "candidate_rows_sha256",
        "old_candidate_muse": "candidate_rows_sha256", "old_candidate_ditto": "candidate_rows_sha256",
        "old_block_request": "request_sha256", "old_process_started": "started_sha256",
        "old_process_terminal": "process_terminal_sha256", "old_child_receipt": "receipt_sha256",
        "old_run_status": "status_sha256", "old_result_manifest": "manifest_sha256",
        "fresh_session_manifest": "manifest_sha256", "fresh_event0": "event_sha256",
    }
    required_roles = set(expected_paths)
    if set(files) != required_roles:
        raise CoordinatorError("partial-native lineage source-file role set differs from the fixed contract")
    loaded: dict[str, dict[str, Any]] = {}
    for role, expected_path in expected_paths.items():
        ref = files.get(role)
        if not isinstance(ref, Mapping) or Path(str(ref.get("path", ""))) != expected_path:
            raise CoordinatorError("partial-native source path differs for " + role)
        field = self_fields.get(role)
        if field is None:
            actual = _file_pin(expected_path)
            value = None
        else:
            value, _source_raw, actual = _read_self_json(expected_path, role, field)
        if dict(ref) != actual:
            raise CoordinatorError("partial-native source bytes/self pin changed for " + role)
        loaded[role] = value if value is not None else {}

    old_manifest = loaded["old_session_manifest"]
    old_event0, old_event1, old_event2 = (loaded["old_event0"], loaded["old_event1"], loaded["old_event2"])
    pair_intent, pair_receipt = loaded["old_pair_launch_intent"], loaded["old_pair_process_receipt"]
    packet_index, plus_plan = loaded["old_packet_index"], loaded["old_plus_plan"]
    old_dispatch_intent = loaded["old_dispatch_intent"]
    block_request = loaded["old_block_request"]
    block_started = loaded["old_process_started"]
    block_terminal = loaded["old_process_terminal"]
    block_child = loaded["old_child_receipt"]
    block_status = loaded["old_run_status"]
    block_manifest = loaded["old_result_manifest"]

    if (prior.get("session_id") != OLD_SESSION_ID
            or prior.get("session_root") != str(old_session_root)
            or old_manifest.get("session_id") != OLD_SESSION_ID
            or old_manifest.get("status") != "OPEN"
            or old_event0.get("event_index") != 0 or old_event0.get("event_type") != "SESSION_INITIALIZED"
            or old_event1.get("event_index") != 1 or old_event1.get("event_type") != "PAIR_ASKED"
            or old_event2.get("event_index") != 2 or old_event2.get("event_type") != "POINT_DISPATCH_INTENT"
            or old_event1.get("previous_event_sha256") != old_event0.get("event_sha256")
            or old_event2.get("previous_event_sha256") != old_event1.get("event_sha256")
            or old_event0.get("session_id") != OLD_SESSION_ID
            or old_event1.get("session_id") != OLD_SESSION_ID
            or old_event2.get("session_id") != OLD_SESSION_ID
            or old_event0.get("selection_status") != "NOT_ADMITTED"):
        raise CoordinatorError("prior durable session event chain differs from the immutable partial-native attempt")
    old_ask = old_event1.get("ask_pair")
    plus = old_ask.get("plus") if isinstance(old_ask, Mapping) else None
    minus = old_ask.get("minus") if isinstance(old_ask, Mapping) else None
    if (not isinstance(plus, Mapping) or not isinstance(minus, Mapping)
            or old_ask.get("pair_id") != OLD_PAIR_ID
            or plus.get("proposal_id") != OLD_PLUS_PROPOSAL_ID
            or minus.get("proposal_id") != OLD_MINUS_PROPOSAL_ID
            or old_event1.get("pair_id") != OLD_PAIR_ID
            or old_event2.get("proposal_id") != OLD_PLUS_PROPOSAL_ID
            or old_event2.get("point_id") != plus.get("method_id")
            or old_event2.get("proposal") != plus):
        raise CoordinatorError("prior ask and plus-side point intent are not the exact original pair")
    if (pair_intent.get("pair_id") != OLD_PAIR_ID
            or pair_intent.get("session_id") != OLD_SESSION_ID
            or pair_intent.get("plus_proposal_id") != OLD_PLUS_PROPOSAL_ID
            or pair_intent.get("minus_proposal_id") != OLD_MINUS_PROPOSAL_ID
            or pair_receipt.get("schema") != PAIR_RECEIPT_SCHEMA
            or pair_receipt.get("status") != "UNKNOWN"
            or pair_receipt.get("child_exit_code") != 2
            or pair_receipt.get("pair_id") != OLD_PAIR_ID
            or pair_receipt.get("session_readback", {}).get("event_count") != 3
            or pair_receipt.get("session_readback", {}).get("matching_pair_tell") is not None
            or pair_receipt.get("relaunch_prohibited") is not True):
        raise CoordinatorError("prior outer UNKNOWN process receipt differs from the captured failure")
    expected_pair_intent_pin = {**_file_pin(expected_paths["old_pair_launch_intent"]),
                                "self_sha256": pair_intent.get("intent_sha256")}
    if pair_receipt.get("intent_pin") != expected_pair_intent_pin:
        raise CoordinatorError("prior outer process receipt does not bind the exact old launch intent")
    if (packet_index.get("execution_started") is not False
            or packet_index.get("model_calls_started") is not False
            or packet_index.get("scope", {}).get("pair_id") != OLD_PAIR_ID
            or packet_index.get("plans", [{}])[0].get("label") != "plus"
            or plus_plan.get("methods", [{}])[0].get("config", {}).get("proposal_id") != OLD_PLUS_PROPOSAL_ID
            or plus_plan.get("methods", [{}])[0].get("config", {}).get("method_id") != plus.get("method_id")):
        raise CoordinatorError("old plus packet is not the exact source point for adoption")
    if (old_dispatch_intent.get("schema") != "fit40_spsa_point_dispatch_intent_v1"
            or old_dispatch_intent.get("status") != "DISPATCH_INTENT"
            or old_dispatch_intent.get("proposal_id") != OLD_PLUS_PROPOSAL_ID
            or old_dispatch_intent.get("pair_id") != OLD_PAIR_ID
            or old_dispatch_intent.get("side") != "plus"
            or old_dispatch_intent.get("point_plan_pin", {}).get("sha256") != files["old_plus_plan"].get("sha256")):
        raise CoordinatorError("old plus dispatch intent does not bind the actual point plan")
    if (block_request.get("schema") != "fit40_spsa_model_block_request_v1"
            or block_request.get("status") != "READY"
            or block_request.get("model_id") != "wav2lip_gan_bac9a81e"
            or block_request.get("block_root") != str(old_block_root.resolve(strict=True))
            or block_request.get("point_plan_pin") != files["old_plus_plan"]
            or block_request.get("candidate_rows_pin") != files["old_candidate_w2l"]
            or block_request.get("family_plan_pin") != old_manifest.get("composed_family_plan")
            or block_request.get("initial_selector_receipt_pin") != old_manifest.get("initial_selector_receipt")
            or block_request.get("execution_package_manifest_pin") != old_manifest.get("execution_package_manifest")
            or block_started.get("status") != "STARTED"
            or block_started.get("schema") != "fit40_spsa_model_process_started_v1"
            or block_started.get("model_id") != "wav2lip_gan_bac9a81e"
            or block_started.get("request_pin") != files["old_block_request"]
            or block_started.get("pid") != 720_290
            or block_started.get("start_ticks") != 8_455_777
            or block_terminal.get("process_started_pin") != files.get("old_process_started")
            or block_terminal.get("status") != "TERMINAL"
            or block_terminal.get("pid") != block_started.get("pid")
            or block_terminal.get("start_ticks") != block_started.get("start_ticks")
            or block_terminal.get("timed_out") is not False
            or block_terminal.get("child_exit_code") != 0
            or block_child.get("child_exit_code") != 0
            or block_child.get("model_id") != "wav2lip_gan_bac9a81e"
            or block_child.get("method_id") != plus.get("method_id")
            or block_child.get("point_id") != plus.get("method_id")
            or block_child.get("proposal_id") != OLD_PLUS_PROPOSAL_ID
            or block_child.get("pair_id") != OLD_PAIR_ID
            or block_child.get("side") != "plus"
            or block_child.get("iteration") != 0
            or block_child.get("point_plan_pin") != files["old_plus_plan"]
            or block_child.get("candidate_rows_pin") != files["old_candidate_w2l"]
            or block_status.get("schema") != "fit40_spsa_point_run_status_v1"
            or block_status.get("model_id") != "wav2lip_gan_bac9a81e"
            or block_status.get("method_id") != plus.get("method_id")
            or block_status.get("point_id") != plus.get("method_id")
            or block_status.get("proposal_id") != OLD_PLUS_PROPOSAL_ID
            or block_status.get("pair_id") != OLD_PAIR_ID
            or block_status.get("side") != "plus"
            or block_status.get("iteration") != 0
            or block_status.get("point_plan_pin") != files["old_plus_plan"]
            or block_status.get("candidate_rows_pin") != files["old_candidate_w2l"]
            or block_status.get("status") != "COMPLETE"
            or block_status.get("planned_rows_pin") != block_manifest.get("planned_rows_pin")
            or block_status.get("results_jsonl_pin") != block_manifest.get("results_jsonl_pin")
            or block_status.get("result_manifest_pin") != files["old_result_manifest"]
            or block_status.get("planned_rows") != 8
            or block_status.get("completed_rows") != 8
            or block_status.get("failed_rows") != 0
            or block_status.get("incomplete_rows") != 0
            or block_status.get("status_counts") != {"PASS": 8, "FAILED": 0, "INCOMPLETE": 0}
            or block_manifest.get("schema") != "fit40_spsa_point_result_manifest_v1"
            or block_manifest.get("model_id") != "wav2lip_gan_bac9a81e"
            or block_manifest.get("method_id") != plus.get("method_id")
            or block_manifest.get("point_id") != plus.get("method_id")
            or block_manifest.get("proposal_id") != OLD_PLUS_PROPOSAL_ID
            or block_manifest.get("pair_id") != OLD_PAIR_ID
            or block_manifest.get("side") != "plus"
            or block_manifest.get("iteration") != 0
            or block_manifest.get("point_plan_pin") != files["old_plus_plan"]
            or block_manifest.get("candidate_rows_pin") != files["old_candidate_w2l"]
            or block_manifest.get("status") != "COMPLETE"
            or block_manifest.get("results_jsonl_pin") != files["old_results_jsonl"]
            or block_manifest.get("planned_rows") != 8
            or block_manifest.get("completed_rows") != 8
            or block_manifest.get("failed_rows") != 0
            or block_manifest.get("incomplete_rows") != 0
            or block_manifest.get("status_counts") != {"PASS": 8, "FAILED": 0, "INCOMPLETE": 0}):
        raise CoordinatorError("adopted W2L block is not the exact terminal eight-row success")
    root_review = lineage.get("root_partial_native_review")
    if (not isinstance(root_review, Mapping)
            or root_review.get("status") != "PASS_ROOT_ACTUAL_PARTIAL_NATIVE_W2L8_SOURCE_AND_ARRAYS"
            or root_review.get("receipt_self_sha256") != OLD_ROOT_QC_SELF_SHA256
            or root_review.get("official_pass_rows") != 8
            or root_review.get("official_seven_array_checks") != 56
            or root_review.get("tell_events") != 0):
        raise CoordinatorError("lineage does not bind the actual independent W2L8 readback qualification")
    processes = prior.get("process_absence")
    expected_processes = {(720_156, 8_452_781), (720_199, 8_453_743), (720_290, 8_455_777)}
    if (not isinstance(processes, list)
            or {(item.get("pid"), item.get("start_ticks")) for item in processes
                if isinstance(item, Mapping)} != expected_processes
            or any(item.get("matching_process_present") is not False for item in processes
                   if isinstance(item, Mapping))
            or len(processes) != len(expected_processes)
            or any(_pid_matches(item["pid"], item["start_ticks"]) for item in processes)):
        raise CoordinatorError("prior UNKNOWN/coordinator/W2L processes are not proven absent")

    adoption, _adoption_raw, actual_adoption_pin = _read_self_json(
        Path(str(adoption_pin.get("path", ""))), "partial-native adoption receipt", "adoption_sha256")
    if (dict(adoption_pin) != actual_adoption_pin
            or (actual_adoption_pin.get("bytes"), actual_adoption_pin.get("sha256"),
                actual_adoption_pin.get("self_sha256")) != RECOVERY_ADOPTION_PIN
            or adoption.get("schema") != RECOVERY_ADOPTION_SCHEMA
            or adoption.get("status") != "CLASSIFIED_PARTIAL_NATIVE_OUTPUT"
            or adoption.get("proposal_identity") != {
                key: plus[key] for key in ("proposal_id", "pair_id", "side", "iteration", "method_id",
                                            "theta", "theta_sha256", "artifact_sha256", "binding_sha256",
                                            "input_pins_sha256", "batch_id", "sample_ids", "direction", "c_k")
            }
            or adoption.get("family_plan_pin") != old_manifest.get("composed_family_plan")
            or adoption.get("initial_selector_receipt_pin") != old_manifest.get("initial_selector_receipt")
            or adoption.get("execution_package_manifest_pin") != old_manifest.get("execution_package_manifest")
            or adoption.get("packet_index_pin") != files["old_packet_index"]
            or adoption.get("point_plan_pin") != files["old_plus_plan"]
            or adoption.get("planned_rows_pin") != files["old_plus_planned_rows"]
            or adoption.get("adopted_model_id") != "wav2lip_gan_bac9a81e"
            or adoption.get("adopted_block_root") != str(old_block_root.resolve(strict=True))):
        raise CoordinatorError("partial-native adoption receipt does not bind the reviewed source block")

    fresh_root = session_root.resolve(strict=True)
    if (not session_root.is_dir() or session_root.is_symlink()
            or fresh.get("session_root") != str(fresh_root)
            or fresh.get("session_id") != FRESH_SESSION_ID
            or session_manifest.get("schema") != SESSION_MANIFEST_SCHEMA
            or session_manifest.get("status") != "OPEN"
            or session_manifest.get("session_id") != FRESH_SESSION_ID
            or session_manifest.get("initial_selection_status") != old_manifest.get("initial_selection_status")
            or session_manifest.get("composed_family_plan") != old_manifest.get("composed_family_plan")
            or session_manifest.get("initial_selector_receipt") != old_manifest.get("initial_selector_receipt")
            or session_manifest.get("execution_package_manifest") != old_manifest.get("execution_package_manifest")
            or session_manifest.get("source_snapshot_lineage") != old_manifest.get("source_snapshot_lineage")
            or session_manifest.get("scientific_input_pins_sha256") != old_manifest.get("scientific_input_pins_sha256")
            or dict(fresh.get("session_manifest_pin", {})) != dict(session_manifest_pin)):
        raise CoordinatorError("fresh session does not preserve the exact prior scientific inputs and package")
    fresh_events_dir = session_root / "events"
    if fresh_events_dir.is_symlink() or not fresh_events_dir.is_dir():
        raise CoordinatorError("fresh session event directory is missing or unsafe")
    fresh_events = sorted(fresh_events_dir.glob("event_*.json"))
    if len(fresh_events) != 1 or fresh_events[0].name != "event_000000.json" or fresh_events[0].is_symlink():
        raise CoordinatorError("fresh recovery session must have event0 only before the first ask")
    fresh_event0 = loaded["fresh_event0"]
    old_state = old_event0.get("controller_state")
    fresh_state = fresh_event0.get("controller_state")
    if (fresh_event0.get("event_index") != 0
            or fresh_event0.get("event_type") != "SESSION_INITIALIZED"
            or fresh_event0.get("session_id") != FRESH_SESSION_ID
            or fresh_event0.get("selection_status") != old_event0.get("selection_status")
            or fresh_state != old_state
            or fresh_state.get("iteration") != 0
            or fresh_state.get("pending_pair") is not None
            or fresh_state.get("history") != []
            or fresh_state.get("halted_reason") is not None
            or fresh.get("event0_pin") != files["fresh_event0"]):
        raise CoordinatorError("fresh event0 is not the exact unchanged controller/RNG state")
    if (lineage.get("controller_equivalence") != {
            "old_event0_controller_state_sha256": _sha(_canonical(old_state)),
            "fresh_event0_controller_state_sha256": _sha(_canonical(fresh_state)),
            "old_event1_ask_pair_sha256": _sha(_canonical(old_event1.get("ask_pair"))),
            "old_event1_controller_state_after_sha256": _sha(_canonical(old_event1.get("controller_state_after"))),
            "old_event2_proposal_sha256": _sha(_canonical(old_event2.get("proposal"))),
            "fresh_event0_equals_old_event0": True,
            "fresh_first_ask_must_equal_old_event1": True,
            "fresh_first_ask_state_after_must_equal_old_event1": True,
            "old_unknown_intent_is_never_replayed": True,
        }):
        raise CoordinatorError("partial-native lineage's deterministic controller-state statement is invalid")
    return pin


def _read_recovery_payload(lineage_pin: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read the already-sealed adoption payload through its exact lineage pin."""
    lineage_path = Path(str(lineage_pin.get("path", "")))
    lineage, _raw, observed_lineage_pin = _read_self_json(
        lineage_path, "partial-native recovery lineage", "lineage_sha256")
    if (observed_lineage_pin != dict(lineage_pin)
            or (observed_lineage_pin.get("bytes"), observed_lineage_pin.get("sha256"),
                observed_lineage_pin.get("self_sha256")) != RECOVERY_LINEAGE_PIN):
        raise CoordinatorError("partial-native recovery lineage pin changed")
    adoption_pin = lineage.get("partial_native_adoption_receipt_pin")
    if not isinstance(adoption_pin, Mapping):
        raise CoordinatorError("recovery lineage omits its adoption receipt pin")
    adoption, _adoption_raw, observed_adoption_pin = _read_self_json(
        Path(str(adoption_pin.get("path", ""))), "partial-native adoption receipt", "adoption_sha256")
    if (observed_adoption_pin != dict(adoption_pin)
            or (observed_adoption_pin.get("bytes"), observed_adoption_pin.get("sha256"),
                observed_adoption_pin.get("self_sha256")) != RECOVERY_ADOPTION_PIN
            or adoption.get("schema") != RECOVERY_ADOPTION_SCHEMA
            or adoption.get("status") != "CLASSIFIED_PARTIAL_NATIVE_OUTPUT"):
        raise CoordinatorError("partial-native adoption receipt pin or status changed")
    return lineage, adoption, dict(adoption_pin)

def _proc_start_ticks(pid: int, proc_root: Path = Path("/proc")) -> int | None:
    """Read Linux process starttime (stat field 22); None means unobservable."""
    stat_path = proc_root / str(pid) / "stat"
    try:
        raw = stat_path.read_text(encoding="ascii")
        # comm is parenthesized and may contain spaces or ')'; split after its final ')'.
        tail = raw[raw.rfind(")") + 2:].split()
        value = int(tail[19])  # fields after comm begin at field 3; field 22 is index 19.
        return value if value >= 0 else None
    except (OSError, ValueError, IndexError):
        return None


def _pid_matches(pid: Any, ticks: Any, proc_root: Path = Path("/proc")) -> bool:
    return (type(pid) is int and type(ticks) is int
            and _proc_start_ticks(pid, proc_root=proc_root) == ticks)


def _safe_directory(path: Path, *, create: bool) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise CoordinatorError("coordinator directory must be absolute and non-symlinked")
    if create:
        if path.exists():
            raise CoordinatorError("coordinator directory must be absent: %s" % path)
        path.mkdir(parents=False, exist_ok=False, mode=0o700)
        _fsync_dir(path.parent)
    elif not path.is_dir():
        raise CoordinatorError("coordinator directory is missing: %s" % path)
    return path.resolve(strict=True)


def _validate_dispatch_storage_root(value: str | Path, *, create: bool) -> Path:
    if not isinstance(value, (str, Path)):
        raise CoordinatorError("dispatch storage root must be a path")
    path = Path(value)
    if DISPATCH_MOUNT_ROOT.is_symlink() or not DISPATCH_MOUNT_ROOT.is_dir():
        raise CoordinatorError("dispatch mount root is missing or unsafe")
    expected_parent = DISPATCH_MOUNT_ROOT.resolve(strict=True)
    expected = expected_parent / DISPATCH_ROOT_NAME
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise CoordinatorError("dispatch storage root parent is missing or unsafe") from exc
    if (not path.is_absolute() or path.is_symlink()
            or path != expected or parent != expected_parent):
        raise CoordinatorError("dispatch storage root must be the dedicated /dev/shm execution namespace")
    if create:
        if path.exists():
            raise CoordinatorError("dispatch storage root must be absent for a fresh coordinator")
        path.mkdir(mode=0o700, exist_ok=False)
        _fsync_dir(path.parent)
        (path / "pairs").mkdir(mode=0o700, exist_ok=False)
        _fsync_dir(path)
    else:
        if path.is_symlink() or not path.is_dir():
            raise CoordinatorError("sealed dispatch storage root is missing or unsafe")
        pairs = path / "pairs"
        if pairs.is_symlink() or not pairs.is_dir():
            raise CoordinatorError("sealed dispatch storage pair namespace is missing or unsafe")
    resolved = path.resolve(strict=True)
    if resolved != expected:
        raise CoordinatorError("dispatch storage root resolves outside its exact namespace")
    return resolved


def _python_pin(executable: Path) -> dict[str, Any]:
    if not executable.is_absolute() or not executable.exists() or not executable.is_file():
        raise CoordinatorError("Python executable must be an absolute existing file")
    target = executable.resolve(strict=True)
    raw = target.read_bytes()
    return {"invoked_path": str(executable), "resolved_path": str(target),
            "bytes": len(raw), "sha256": _sha(raw)}


def _module_plain(value: Any) -> Any:
    # Reuse the session's canonical serializer when loaded; this fallback is for tests.
    if hasattr(value, "__dataclass_fields__"):
        return {key: _module_plain(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(key): _module_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_module_plain(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _module_plain(value.item())
        except Exception:
            pass
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise CoordinatorError("proposal contains a non-canonical value: %s" % type(value).__name__)


def _read_child_json(raw: bytes) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _tell_for_pair(session: Any, pair_id: str) -> dict[str, Any] | None:
    matches = [event for event in session.events
               if event.get("event_type") == "PAIR_TOLD" and event.get("pair_id") == pair_id]
    if len(matches) > 1:
        raise CoordinatorError("session replay has duplicate PAIR_TOLD events for a pair")
    return dict(matches[0]) if matches else None


def _readback_summary(session: Any, pair_id: str) -> dict[str, Any]:
    tell = _tell_for_pair(session, pair_id)
    pair = session.pending_pair
    pending = None
    if pair is not None:
        pending = {"pair_id": pair.pair_id, "iteration": pair.iteration,
                   "plus_proposal_id": pair.plus.proposal_id,
                   "minus_proposal_id": pair.minus.proposal_id}
    return {"session_id": session.manifest.get("session_id"),
            "controller_iteration": session.controller.iteration,
            "controller_halted": bool(session.halted),
            "pending_pair": pending,
            "event_count": len(session.events),
            "last_event_sha256": (session.events[-1].get("event_sha256") if session.events else None),
            "matching_pair_tell": tell}


class TwentyPairCoordinator:
    """Append-only pair supervisor. Child/session behavior is injectable for CPU tests."""

    def __init__(self, root: Path, manifest: Mapping[str, Any]):
        self.root = root
        self.manifest = dict(manifest)

    @classmethod
    def create(cls, *, coordinator_dir: str | Path, session_dir: str | Path,
               repo_root: str | Path, python_executable: str | Path,
               dispatch_root: str | Path, recovery_lineage_receipt: str | Path) -> "TwentyPairCoordinator":
        repo = Path(repo_root).resolve(strict=True)
        session_input = Path(session_dir)
        if session_input.is_symlink():
            raise CoordinatorError("session root must not be a symlink")
        session_root = session_input.resolve(strict=True)
        if not session_root.is_dir():
            raise CoordinatorError("session root must be a real directory")
        source_closure = _source_closure(repo)
        bootstrap = repo / BOOTSTRAP_REL
        runtime = _python_pin(Path(python_executable))
        session_manifest_path = session_root / "session_manifest.json"
        session_manifest, _raw, session_manifest_pin = _read_self_json(
            session_manifest_path, "durable session manifest", "manifest_sha256")
        if (session_manifest.get("schema") != SESSION_MANIFEST_SCHEMA
                or session_manifest.get("status") != "OPEN"):
            raise CoordinatorError("session manifest is not the sealed open SPSA session")
        recovery_lineage_pin = _validate_recovery_lineage(
            recovery_lineage_receipt, session_root=session_root,
            session_manifest=session_manifest, session_manifest_pin=session_manifest_pin)
        lineage, _lineage_raw, observed_lineage_pin = _read_self_json(
            Path(str(recovery_lineage_pin["path"])), "partial-native recovery lineage", "lineage_sha256")
        if observed_lineage_pin != recovery_lineage_pin:
            raise CoordinatorError("recovery lineage changed between validation and coordinator creation")
        adoption_pin = lineage.get("partial_native_adoption_receipt_pin")
        if not isinstance(adoption_pin, Mapping):
            raise CoordinatorError("recovery lineage omits the immutable adoption receipt")
        adoption, _adoption_raw, observed_adoption_pin = _read_self_json(
            Path(str(adoption_pin.get("path", ""))), "partial-native adoption receipt", "adoption_sha256")
        if (observed_adoption_pin != dict(adoption_pin)
                or adoption.get("schema") != RECOVERY_ADOPTION_SCHEMA
                or adoption.get("status") != "CLASSIFIED_PARTIAL_NATIVE_OUTPUT"):
            raise CoordinatorError("adoption receipt changed before coordinator creation")
        dispatch_storage_root = _validate_dispatch_storage_root(dispatch_root, create=True)
        root = _safe_directory(Path(coordinator_dir), create=True)
        (root / "pairs").mkdir(mode=0o700)
        (root / "process_logs").mkdir(mode=0o700)
        _fsync_dir(root)
        manifest = _seal({
            "schema": COORDINATOR_SCHEMA,
            "status": "FROZEN_SOURCE_PINNED_COORDINATOR",
            "coordinator_id": "fit40_twenty_pair_" + uuid.uuid4().hex,
            "created_utc": _utc_now(),
            "repo_root": str(repo),
            "session_dir": str(session_root),
            "dispatch_storage_root": str(dispatch_storage_root),
            "session_id": session_manifest.get("session_id"),
            "session_manifest_pin": session_manifest_pin,
            "pre_native_recovery_lineage_pin": recovery_lineage_pin,
            "partial_native_adoption_receipt_pin": dict(adoption_pin),
            "bootstrap_pin": source_closure[BOOTSTRAP_REL],
            "session_module_pin": source_closure[SESSION_REL],
            "coordinator_source_pin": _file_pin(Path(__file__).resolve(strict=True)),
            "python_runtime": runtime,
            "max_pairs": MAX_PAIRS,
            "launch_mode": "one_bootstrap_run_pair_child_per_pair",
            "science_run_started_at_manifest_seal": False,
        }, "coordinator_sha256")
        _write_json_exclusive(root / "coordinator_manifest.json", manifest, "coordinator_sha256")
        return cls(root, manifest)

    @classmethod
    def open(cls, coordinator_dir: str | Path) -> "TwentyPairCoordinator":
        root = _safe_directory(Path(coordinator_dir), create=False)
        value, _raw, _pin = _read_self_json(root / "coordinator_manifest.json",
                                           "coordinator manifest", "coordinator_sha256")
        if (value.get("schema") != COORDINATOR_SCHEMA
                or value.get("status") != "FROZEN_SOURCE_PINNED_COORDINATOR"
                or value.get("science_run_started_at_manifest_seal") is not False
                or value.get("max_pairs") != MAX_PAIRS):
            raise CoordinatorError("coordinator manifest schema/status/limit mismatch")
        observed_dispatch_root = _validate_dispatch_storage_root(
            value.get("dispatch_storage_root", ""), create=False)
        if str(observed_dispatch_root) != value.get("dispatch_storage_root"):
            raise CoordinatorError("coordinator manifest dispatch storage root is not canonical")
        return cls(root, value)

    def _verify_runtime_sources(self) -> tuple[Path, Path, Path]:
        repo = Path(self.manifest["repo_root"])
        closure = _source_closure(repo)
        if (closure[BOOTSTRAP_REL] != self.manifest.get("bootstrap_pin")
                or closure[SESSION_REL] != self.manifest.get("session_module_pin")
                or _file_pin(Path(__file__).resolve(strict=True)) != self.manifest.get("coordinator_source_pin")):
            raise CoordinatorError("source closure changed after coordinator creation")
        recovery_lineage_pin = self.manifest.get("pre_native_recovery_lineage_pin")
        if (not isinstance(recovery_lineage_pin, Mapping)
                or _self_file_pin(Path(str(recovery_lineage_pin.get("path", ""))),
                                  "pre-native recovery lineage", "lineage_sha256")
                != dict(recovery_lineage_pin)):
            raise CoordinatorError("pre-native recovery lineage changed after coordinator creation")
        _lineage, _adoption, adoption_pin = _read_recovery_payload(recovery_lineage_pin)
        if adoption_pin != self.manifest.get("partial_native_adoption_receipt_pin"):
            raise CoordinatorError("partial-native adoption receipt changed after coordinator creation")
        runtime = self.manifest.get("python_runtime")
        if not isinstance(runtime, Mapping):
            raise CoordinatorError("coordinator runtime pin is missing")
        executable = Path(str(runtime.get("invoked_path", "")))
        if _python_pin(executable) != dict(runtime):
            raise CoordinatorError("coordinator Python runtime changed")
        session = Path(self.manifest["session_dir"])
        session_value, _raw, session_pin = _read_self_json(
            session / "session_manifest.json", "durable session manifest", "manifest_sha256")
        if (session_value.get("session_id") != self.manifest.get("session_id")
                or session_pin != self.manifest.get("session_manifest_pin")):
            raise CoordinatorError("durable session manifest changed")
        dispatch_root = _validate_dispatch_storage_root(
            self.manifest.get("dispatch_storage_root", ""), create=False)
        if str(dispatch_root) != self.manifest.get("dispatch_storage_root"):
            raise CoordinatorError("sealed dispatch storage root changed")
        return repo, session, executable

    def _pair_dir(self, index: int) -> Path:
        return self.root / "pairs" / ("pair_%02d" % index)

    def _recovery_payload(self) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        recovery_lineage_pin = self.manifest.get("pre_native_recovery_lineage_pin")
        if not isinstance(recovery_lineage_pin, Mapping):
            raise CoordinatorError("coordinator manifest omits partial-native recovery lineage")
        lineage, adoption, adoption_pin = _read_recovery_payload(recovery_lineage_pin)
        if adoption_pin != self.manifest.get("partial_native_adoption_receipt_pin"):
            raise CoordinatorError("coordinator recovery adoption pin differs from the lineage")
        return lineage, adoption, adoption_pin

    def _validate_first_recovery_ask(self, pair: Any, session: Any) -> None:
        """Require the fresh session's first durable ask to replay the old ask exactly."""
        lineage, _adoption, _adoption_pin = self._recovery_payload()
        roles = lineage.get("prior_attempt", {}).get("source_file_pins", {})
        old_event_pin = roles.get("old_event1") if isinstance(roles, Mapping) else None
        if not isinstance(old_event_pin, Mapping):
            raise CoordinatorError("recovery lineage omits the original durable PAIR_ASKED event")
        old_event, _raw, observed = _read_self_json(
            Path(str(old_event_pin.get("path", ""))), "prior PAIR_ASKED event", "event_sha256")
        if dict(old_event_pin) != observed:
            raise CoordinatorError("prior PAIR_ASKED event differs from the recovery lineage pin")
        events = getattr(session, "events", None)
        if not isinstance(events, list) or len(events) != 2:
            raise CoordinatorError("fresh session must contain only initialization and its first ask")
        fresh_event0, fresh_event1 = events
        old_ask = old_event.get("ask_pair")
        fresh_ask = fresh_event1.get("ask_pair") if isinstance(fresh_event1, Mapping) else None
        if (not isinstance(old_ask, Mapping)
                or fresh_event0.get("event_index") != 0
                or fresh_event0.get("event_type") != "SESSION_INITIALIZED"
                or fresh_event1.get("event_index") != 1
                or fresh_event1.get("event_type") != "PAIR_ASKED"
                or fresh_event1.get("previous_event_sha256") != fresh_event0.get("event_sha256")
                or fresh_event1.get("session_id") != FRESH_SESSION_ID
                or fresh_event1.get("pair_id") != OLD_PAIR_ID
                or pair.pair_id != OLD_PAIR_ID
                or pair.iteration != 0
                or _canonical(_module_plain(pair)) != _canonical(old_ask)
                or _canonical(fresh_ask) != _canonical(old_ask)
                or _canonical(fresh_event1.get("controller_state_after"))
                   != _canonical(old_event.get("controller_state_after"))
                or session.pending_pair != pair
                or session.controller.iteration != 0):
            raise CoordinatorError("fresh first ask or post-ask PCG64 state differs from the immutable old ask")

    def _existing_prefix(self) -> tuple[int, str | None]:
        pair_root = self.root / "pairs"
        if pair_root.is_symlink() or not pair_root.is_dir():
            raise CoordinatorError("pair state directory is missing or unsafe")
        names = sorted(path.name for path in pair_root.iterdir())
        if any(not re.fullmatch(r"pair_[0-9]{2}", name) for name in names):
            raise CoordinatorError("unexpected entry in pair state directory")
        updated = 0
        unresolved: str | None = None
        for index, name in enumerate(names, 1):
            if name != "pair_%02d" % index:
                raise CoordinatorError("pair records are not a contiguous prefix")
            directory = pair_root / name
            if directory.is_symlink() or not directory.is_dir():
                raise CoordinatorError("pair record is not a real directory")
            intent_path, receipt_path = directory / "launch_intent.json", directory / "process_receipt.json"
            if not intent_path.is_file() or intent_path.is_symlink():
                raise CoordinatorError("pair directory has no valid durable intent")
            intent, _raw, _pin = _read_self_json(intent_path, "pair launch intent", "intent_sha256")
            if intent.get("pair_index") != index or intent.get("schema") != PAIR_INTENT_SCHEMA:
                raise CoordinatorError("pair launch intent identity mismatch")
            expected_dispatch_base = _validate_dispatch_storage_root(
                self.manifest.get("dispatch_storage_root", ""), create=False)
            expected_dispatch = (expected_dispatch_base / "pairs" / name / "dispatch_root")
            _lineage, adoption, adoption_pin = self._recovery_payload()
            proposal_pair = intent.get("proposal_pair")
            plus_intent = proposal_pair.get("plus") if isinstance(proposal_pair, Mapping) else None
            adoption_applies = (
                index == 1 and isinstance(plus_intent, Mapping)
                and proposal_pair.get("pair_id") == OLD_PAIR_ID
                and plus_intent.get("proposal_id") == OLD_PLUS_PROPOSAL_ID
                and _canonical(plus_intent) == _canonical(adoption.get("proposal_identity")))
            if index == 1 and not adoption_applies:
                raise CoordinatorError("first pair is not the exact pair authorized for partial-native adoption")
            expected_packet = (Path(str(adoption["packet_index_pin"]["path"])).parent
                               if adoption_applies else directory / "packet")
            expected_argv = [str(self.manifest["python_runtime"]["invoked_path"]),
                             str(Path(self.manifest["repo_root"]) / BOOTSTRAP_REL),
                             "--repo-root", self.manifest["repo_root"], "run-pair",
                             "--session-dir", self.manifest["session_dir"],
                             "--packet-dir", str(expected_packet),
                             "--dispatch-root", str(expected_dispatch)]
            if adoption_applies:
                expected_argv.extend(["--partial-native-adoption-receipt", adoption_pin["path"]])
            if (intent.get("dispatch_storage_root") != str(expected_dispatch_base)
                    or intent.get("dispatch_root") != str(expected_dispatch)
                    or intent.get("packet_dir") != str(expected_packet)
                    or intent.get("partial_native_adoption_receipt_pin")
                       != (adoption_pin if adoption_applies else None)
                    or intent.get("pre_native_recovery_lineage_pin")
                       != self.manifest.get("pre_native_recovery_lineage_pin")
                    or intent.get("argv") != expected_argv):
                raise CoordinatorError("pair intent dispatch storage path differs from the sealed root/index")
            if not receipt_path.exists():
                readback_path = directory / "unknown_readback.json"
                if readback_path.exists():
                    readback, _raw, _pin = _read_self_json(
                        readback_path, "unknown pair readback", "readback_sha256")
                    if (readback.get("schema") != PAIR_READBACK_SCHEMA
                            or readback.get("status") != "UNKNOWN"
                            or readback.get("relaunch_prohibited") is not True):
                        raise CoordinatorError("existing unknown readback is malformed")
                unresolved = name
                break
            if expected_dispatch.is_symlink() or not expected_dispatch.is_dir():
                raise CoordinatorError("terminal pair receipt has no real dispatch output root")
            receipt, _raw, _pin = _read_self_json(receipt_path, "pair process receipt", "receipt_sha256")
            expected_intent_pin = {**_file_pin(intent_path), "self_sha256": intent["intent_sha256"]}
            if receipt.get("intent_pin") != expected_intent_pin:
                raise CoordinatorError("pair process receipt does not bind its exact launch intent")
            if (receipt.get("schema") != PAIR_RECEIPT_SCHEMA
                    or receipt.get("pair_index") != index
                    or receipt.get("pair_id") != intent.get("pair_id")
                    or receipt.get("iteration") != intent.get("iteration")
                    or type(receipt.get("child_exit_code")) not in (int, type(None))):
                raise CoordinatorError("pair process receipt schema or identity mismatch")
            start_pin = receipt.get("process_start_pin")
            if start_pin is not None:
                expected_start_path = (directory / "process_start.json").resolve(strict=True)
                if Path(str(start_pin.get("path", ""))).resolve(strict=True) != expected_start_path:
                    raise CoordinatorError("process start receipt is outside its immutable pair leaf")
                start, _start_raw, observed_start_pin = _read_self_json(
                    expected_start_path, "pair process start receipt", "start_sha256")
                if (observed_start_pin != dict(start_pin)
                        or start.get("schema") != PAIR_START_SCHEMA
                        or start.get("child_pid") != receipt.get("child_pid")
                        or start.get("child_start_ticks") != receipt.get("child_start_ticks")):
                    raise CoordinatorError("process receipt does not bind its parent-captured PID/start ticks")
            captured_raw: dict[str, bytes] = {}
            for role, leaf in (("stdout_pin", "child.stdout"), ("stderr_pin", "child.stderr")):
                output_pin = receipt.get(role)
                if not isinstance(output_pin, Mapping):
                    raise CoordinatorError("pair process receipt omits captured %s" % role)
                expected_output = directory / leaf
                if output_pin.get("path") is None:
                    if (receipt.get("status") != "UNKNOWN"
                            or expected_output.exists() or expected_output.is_symlink()
                            or output_pin.get("bytes") != 0
                            or output_pin.get("sha256") != _sha(b"")):
                        raise CoordinatorError("missing child output is only valid for an empty UNKNOWN receipt")
                    output_raw = b""
                else:
                    output_path = expected_output.resolve(strict=True)
                    if Path(str(output_pin.get("path", ""))).resolve(strict=True) != output_path:
                        raise CoordinatorError("captured child output is outside its immutable pair leaf")
                    output_raw = _regular_bytes(output_path, "captured child output")
                if (len(output_raw) != output_pin.get("bytes")
                        or _sha(output_raw) != output_pin.get("sha256")):
                    raise CoordinatorError("captured child output bytes changed")
                captured_raw[role] = output_raw
            parsed_child = _read_child_json(captured_raw["stdout_pin"])
            if parsed_child != receipt.get("child_result"):
                raise CoordinatorError("stored child result differs from its captured stdout bytes")
            if receipt.get("status") == "COMPLETE_UPDATED":
                if (type(receipt.get("child_exit_code")) is not int
                        or receipt["child_exit_code"] != 0
                        or type(receipt.get("child_pid")) is not int
                        or type(receipt.get("child_start_ticks")) is not int
                        or start_pin is None
                        or not isinstance(parsed_child, Mapping)
                        or parsed_child.get("status") != "PAIR_TOLD_UPDATED"
                        or parsed_child.get("tell_result", {}).get("status") != "UPDATED"):
                    raise CoordinatorError("UPDATED receipt lacks numeric child rc 0")
                updated += 1
            elif receipt.get("status") == "NO_UPDATE":
                if (receipt.get("child_exit_code") != 2 or start_pin is None
                        or type(receipt.get("child_pid")) is not int
                        or type(receipt.get("child_start_ticks")) is not int
                        or not isinstance(parsed_child, Mapping)
                        or parsed_child.get("status") != "PAIR_TOLD_NOT_UPDATED"
                        or parsed_child.get("tell_result", {}).get("status") != "NO_UPDATE"):
                    raise CoordinatorError("NO_UPDATE receipt lacks its exact terminal child evidence")
                return updated, name
            else:
                return updated, name
        return updated, unresolved

    def start_detached(self, *, python_executable: str | Path | None = None) -> dict[str, Any]:
        """Create start intent, launch this coordinator in a new session, then pin its PID."""
        if (self.root / "coordinator_start_intent.json").exists():
            raise CoordinatorError("detached start intent already exists; refusing duplicate start")
        repo, session, default_python = self._verify_runtime_sources()
        python = Path(python_executable).resolve(strict=True) if python_executable else default_python
        runtime = _python_pin(python)
        if runtime != self.manifest["python_runtime"]:
            raise CoordinatorError("detached Python runtime differs from the sealed coordinator runtime")
        argv = [str(python), str(Path(__file__).resolve(strict=True)), "run",
                "--repo-root", str(repo), "--session-dir", str(session),
                "--coordinator-dir", str(self.root),
                "--dispatch-root", str(self.manifest["dispatch_storage_root"])]
        source_pin = _file_pin(Path(__file__).resolve(strict=True))
        manifest_pin = _self_file_pin(self.root / "coordinator_manifest.json",
                                      "coordinator manifest", "coordinator_sha256")
        intent = _seal({"schema": START_INTENT_SCHEMA, "coordinator_id": self.manifest["coordinator_id"],
                        "created_utc": _utc_now(), "argv": argv,
                        "coordinator_source_pin": source_pin,
                        "coordinator_manifest_pin": manifest_pin,
                        "python_runtime": runtime,
                        "dispatch_storage_root": self.manifest["dispatch_storage_root"]},
                       "intent_sha256")
        intent_pin = _write_json_exclusive(self.root / "coordinator_start_intent.json", intent,
                                           "intent_sha256")
        stdout_path, stderr_path = self.root / "process_logs" / "coordinator.stdout", self.root / "process_logs" / "coordinator.stderr"
        for path in (stdout_path, stderr_path):
            if path.exists() or path.is_symlink():
                raise CoordinatorError("detached log target already exists")
        started = _utc_now()
        try:
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                process = subprocess.Popen(argv, cwd=str(repo), stdin=subprocess.DEVNULL,
                                           stdout=stdout, stderr=stderr, close_fds=True,
                                           start_new_session=True)
                ticks = _proc_start_ticks(process.pid)
        except Exception as exc:
            receipt = {"schema": START_RECEIPT_SCHEMA, "status": "NOT_STARTED",
                       "coordinator_id": self.manifest["coordinator_id"],
                       "intent_pin": intent_pin, "started_utc": started,
                       "argv": argv, "coordinator_source_pin": source_pin,
                       "coordinator_manifest_pin": manifest_pin,
                       "python_runtime": runtime,
                       "dispatch_storage_root": self.manifest["dispatch_storage_root"],
                       "child_pid": None, "child_start_ticks": None,
                       "error_type": type(exc).__name__, "error_text": str(exc)}
            _write_json_exclusive(self.root / "coordinator_start_receipt.json", receipt,
                                  "receipt_sha256")
            raise CoordinatorError("detached coordinator process did not start") from exc
        receipt = {"schema": START_RECEIPT_SCHEMA,
                   "status": "STARTED" if ticks is not None else "STARTED_TICKS_UNAVAILABLE",
                   "coordinator_id": self.manifest["coordinator_id"],
                   "intent_pin": intent_pin, "started_utc": started,
                   "argv": argv, "coordinator_source_pin": source_pin,
                   "coordinator_manifest_pin": manifest_pin,
                   "python_runtime": runtime,
                   "dispatch_storage_root": self.manifest["dispatch_storage_root"],
                   "child_pid": process.pid, "child_start_ticks": ticks,
                   "stdout_path": str(stdout_path.resolve(strict=True)),
                   "stderr_path": str(stderr_path.resolve(strict=True))}
        _write_json_exclusive(self.root / "coordinator_start_receipt.json", receipt,
                              "receipt_sha256")
        return receipt

    def _load_session(self, session_loader: Callable[[], Any] | None = None) -> Any:
        if session_loader is not None:
            return session_loader()
        repo, session_dir, _python = self._verify_runtime_sources()
        bootstrap_path = repo / BOOTSTRAP_REL
        import importlib.util
        spec = importlib.util.spec_from_file_location("fit40_pinned_bootstrap_for_coordinator", bootstrap_path)
        if spec is None or spec.loader is None:
            raise CoordinatorError("cannot load the pinned bootstrap CLI")
        bootstrap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bootstrap)
        modules = bootstrap._load_frozen_modules(repo)
        return modules["session"].DurableSPSASession.resume(session_dir=session_dir)

    def _make_pair_intent(self, *, index: int, pair: Any, session: Any,
                          repo: Path, session_dir: Path) -> dict[str, Any]:
        directory = self._pair_dir(index)
        if directory.exists() or directory.is_symlink():
            raise CoordinatorError("pair output leaf already exists")
        directory.mkdir(mode=0o700)
        _fsync_dir(directory)
        dispatch_base = Path(self.manifest["dispatch_storage_root"])
        dispatch = dispatch_base / "pairs" / ("pair_%02d" % index) / "dispatch_root"
        proposal_pair = {"pair_id": pair.pair_id, "iteration": pair.iteration,
                         "plus": _module_plain(pair.plus), "minus": _module_plain(pair.minus)}
        if session.pending_pair != pair:
            raise CoordinatorError("pair is not the exact durable pending ask")
        _lineage, adoption, adoption_pin = self._recovery_payload()
        adoption_applies = (
            index == 1 and pair.pair_id == OLD_PAIR_ID
            and pair.plus.proposal_id == OLD_PLUS_PROPOSAL_ID
            and _canonical(proposal_pair["plus"]) == _canonical(adoption.get("proposal_identity")))
        if index == 1 and not adoption_applies:
            raise CoordinatorError("first pair does not exactly match the adopted partial-native plus proposal")
        packet = (Path(str(adoption["packet_index_pin"]["path"])).parent
                  if adoption_applies else directory / "packet")
        argv = [str(self.manifest["python_runtime"]["invoked_path"]),
                str(repo / BOOTSTRAP_REL), "--repo-root", str(repo), "run-pair",
                "--session-dir", str(session_dir), "--packet-dir", str(packet),
                "--dispatch-root", str(dispatch)]
        if adoption_applies:
            argv.extend(["--partial-native-adoption-receipt", adoption_pin["path"]])
        intent = {"schema": PAIR_INTENT_SCHEMA, "pair_index": index,
                  "coordinator_id": self.manifest["coordinator_id"],
                  "session_id": session.manifest.get("session_id"),
                  "iteration": pair.iteration, "pair_id": pair.pair_id,
                  "plus_proposal_id": pair.plus.proposal_id,
                  "minus_proposal_id": pair.minus.proposal_id,
                  "proposal_pair": proposal_pair,
                  "pair_sha256": _sha(_canonical(proposal_pair)),
                  "session_manifest_pin": _file_pin(session_dir / "session_manifest.json"),
                  "bootstrap_pin": _file_pin(repo / BOOTSTRAP_REL),
                  "coordinator_manifest_pin": _file_pin(self.root / "coordinator_manifest.json"),
                  "pre_native_recovery_lineage_pin": self.manifest["pre_native_recovery_lineage_pin"],
                  "partial_native_adoption_receipt_pin": adoption_pin if adoption_applies else None,
                  "argv": argv, "packet_dir": str(packet),
                  "dispatch_storage_root": str(dispatch_base),
                  "dispatch_root": str(dispatch),
                  "created_utc": _utc_now()}
        return _seal(intent, "intent_sha256")

    def _prepare_pair_dispatch_root(self, index: int) -> Path:
        base = _validate_dispatch_storage_root(
            self.manifest.get("dispatch_storage_root", ""), create=False)
        pairs_root = base / "pairs"
        if pairs_root.is_symlink() or not pairs_root.is_dir():
            raise CoordinatorError("dispatch pair namespace is missing or unsafe")
        pair_dir = pairs_root / ("pair_%02d" % index)
        if pair_dir.exists() or pair_dir.is_symlink():
            raise CoordinatorError("pair dispatch output leaf already exists")
        pair_dir.mkdir(mode=0o700, exist_ok=False)
        dispatch = pair_dir / "dispatch_root"
        dispatch.mkdir(mode=0o700, exist_ok=False)
        _fsync_dir(pair_dir)
        _fsync_dir(pairs_root)
        return dispatch.resolve(strict=True)

    def _child_command(self, argv: Sequence[str], directory: Path) -> dict[str, Any]:
        stdout_path, stderr_path = directory / "child.stdout", directory / "child.stderr"
        started = _utc_now()
        pid: int | None = None
        ticks: int | None = None
        return_code: int | None = None
        launch_pin: dict[str, Any] | None = None
        exception: dict[str, str] | None = None
        try:
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                process = subprocess.Popen(list(argv), cwd=self.manifest["repo_root"],
                                           stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                           close_fds=True, start_new_session=True)
                pid = process.pid
                ticks = _proc_start_ticks(pid)
                launch = _seal({"schema": PAIR_START_SCHEMA, "status": "STARTED",
                                "coordinator_id": self.manifest["coordinator_id"],
                                "child_pid": pid, "child_start_ticks": ticks,
                                "started_utc": started, "argv": list(argv)}, "start_sha256")
                launch_pin = _write_json_exclusive(directory / "process_start.json", launch,
                                                   "start_sha256")
                return_code = process.wait()
        except Exception as exc:
            exception = {"type": type(exc).__name__, "message": str(exc)}
        if pid is not None and ticks is not None and _pid_matches(pid, ticks):
            return {"pid": pid, "start_ticks": ticks, "return_code": None,
                    "started_utc": started, "ended_utc": None, "launch_pin": launch_pin,
                    "stdout_path": str(stdout_path.resolve()) if stdout_path.exists() else None,
                    "stdout_bytes": None, "stdout_sha256": None,
                    "stderr_path": str(stderr_path.resolve()) if stderr_path.exists() else None,
                    "stderr_bytes": None, "stderr_sha256": None,
                    "stdout_json": None, "exception": exception, "running": True}
        ended = _utc_now()
        stdout_raw = stdout_path.read_bytes() if stdout_path.exists() and not stdout_path.is_symlink() else b""
        stderr_raw = stderr_path.read_bytes() if stderr_path.exists() and not stderr_path.is_symlink() else b""
        return {"pid": pid, "start_ticks": ticks, "return_code": return_code,
                "started_utc": started, "ended_utc": ended,
                "launch_pin": launch_pin,
                "stdout_path": str(stdout_path.resolve()) if stdout_path.exists() else None,
                "stdout_bytes": len(stdout_raw), "stdout_sha256": _sha(stdout_raw),
                "stderr_path": str(stderr_path.resolve()) if stderr_path.exists() else None,
                "stderr_bytes": len(stderr_raw), "stderr_sha256": _sha(stderr_raw),
                "stdout_json": _read_child_json(stdout_raw), "exception": exception,
                "running": False}

    def _unknown_from_existing_intent(self, index: int, session: Any) -> dict[str, Any]:
        directory = self._pair_dir(index)
        intent, _raw, intent_pin = _read_self_json(directory / "launch_intent.json",
                                                    "pair launch intent", "intent_sha256")
        start_path = directory / "process_start.json"
        if start_path.exists():
            start, _raw, start_pin = _read_self_json(start_path, "pair process start", "start_sha256")
            if (start.get("schema") != PAIR_START_SCHEMA
                    or start.get("status") != "STARTED"
                    or start.get("coordinator_id") != self.manifest["coordinator_id"]):
                raise CoordinatorError("existing process-start receipt identity is invalid")
            if start.get("child_pid") is not None and _pid_matches(
                    start.get("child_pid"), start.get("child_start_ticks")):
                return {"status": "RUNNING", "pair_index": index,
                        "pair_id": intent.get("pair_id"), "child_pid": start["child_pid"],
                        "child_start_ticks": start.get("child_start_ticks"),
                        "process_start_pin": start_pin, "relaunch_prohibited": True}
        readback = _readback_summary(session, str(intent.get("pair_id")))
        receipt_path = directory / "unknown_readback.json"
        body = {"schema": PAIR_READBACK_SCHEMA, "status": "UNKNOWN",
                "coordinator_id": self.manifest["coordinator_id"],
                "pair_index": index, "pair_id": intent.get("pair_id"),
                "intent_pin": intent_pin, "readback_utc": _utc_now(),
                "child_exit_code": None, "process_exit_unknown": True,
                "session_readback": readback, "relaunch_prohibited": True}
        pin = _write_json_exclusive(receipt_path, body, "readback_sha256")
        return {"status": "UNKNOWN", "pair_index": index,
                "pair_id": intent.get("pair_id"), "readback_pin": pin,
                "relaunch_prohibited": True}

    def run(self, *, session_loader: Callable[[], Any] | None = None,
            child_runner: Callable[[Sequence[str], Path], Mapping[str, Any]] | None = None,
            max_pairs: int = MAX_PAIRS) -> dict[str, Any]:
        if max_pairs != MAX_PAIRS:
            raise CoordinatorError("coordinator requires the full predeclared 20-pair limit")
        repo, session_dir, _python = self._verify_runtime_sources()
        session = self._load_session(session_loader)
        completed, unresolved = self._existing_prefix()
        if unresolved is not None:
            index = int(unresolved[-2:])
            if (self._pair_dir(index) / "process_receipt.json").exists():
                receipt, _raw, _pin = _read_self_json(self._pair_dir(index) / "process_receipt.json",
                                                      "pair process receipt", "receipt_sha256")
                return {"status": receipt.get("status"), "completed_pairs": completed,
                        "stop_pair_index": index, "reason": "prior terminal outcome is not UPDATED"}
            existing_readback = self._pair_dir(index) / "unknown_readback.json"
            if existing_readback.exists():
                readback, _raw, pin = _read_self_json(existing_readback,
                                                      "unknown pair readback", "readback_sha256")
                return {"status": readback.get("status"), "completed_pairs": completed,
                        "stop_pair_index": index, "readback_pin": pin,
                        "reason": "prior unresolved intent is permanently non-dispatchable"}
            result = self._unknown_from_existing_intent(index, session)
            return {"status": result["status"], "completed_pairs": completed,
                    "stop_pair_index": index, "readback": result}
        if session.controller.iteration != completed or session.halted or session.pending_pair is not None:
            raise CoordinatorError("session replay state does not match the exact completed coordinator prefix")
        # Cross-check every previously written COMPLETE receipt against the actual event-chain replay.
        tells = {event.get("pair_id"): event for event in session.events
                 if event.get("event_type") == "PAIR_TOLD"}
        for index in range(1, completed + 1):
            receipt, _raw, _pin = _read_self_json(self._pair_dir(index) / "process_receipt.json",
                                                   "pair process receipt", "receipt_sha256")
            event = tells.get(receipt.get("pair_id"))
            if (receipt.get("status") != "COMPLETE_UPDATED" or event is None
                    or event.get("pair_id") != receipt.get("pair_id")
                    or event.get("iteration") != receipt.get("iteration")
                    or event.get("tell_result", {}).get("status") != "UPDATED"):
                raise CoordinatorError("prior process receipt does not match deterministic session replay")
        results = []
        launcher = child_runner or self._child_command
        for index in range(completed + 1, MAX_PAIRS + 1):
            session = self._load_session(session_loader)
            if session.halted or session.controller.iteration != index - 1:
                raise CoordinatorError("session halted or iteration drifted before next pair")
            pair = session.pending_pair or session.ask_pair()
            if pair.iteration != index - 1:
                raise CoordinatorError("deterministic controller returned an unexpected pair iteration")
            if index == 1:
                self._validate_first_recovery_ask(pair, session)
            directory = self._pair_dir(index)
            intent = self._make_pair_intent(index=index, pair=pair, session=session,
                                            repo=repo, session_dir=session_dir)
            intent_pin = _write_json_exclusive(directory / "launch_intent.json", intent,
                                               "intent_sha256")
            created_dispatch = self._prepare_pair_dispatch_root(index)
            if str(created_dispatch) != intent.get("dispatch_root"):
                raise CoordinatorError("created pair dispatch root differs from its durable intent")
            observed = dict(launcher(intent["argv"], directory))
            if observed.get("running") is True:
                return {"status": "RUNNING", "completed_pairs": index - 1,
                        "stop_pair_index": index, "child_pid": observed.get("pid"),
                        "child_start_ticks": observed.get("start_ticks"),
                        "reason": "child process is still alive; no duplicate launch"}
            session_after = self._load_session(session_loader)
            readback = _readback_summary(session_after, pair.pair_id)
            child_json = observed.get("stdout_json")
            tell = child_json.get("tell_result") if isinstance(child_json, Mapping) else None
            matching_tell = readback.get("matching_pair_tell")
            exact_updated = (
                type(observed.get("return_code")) is int
                and observed.get("return_code") == 0
                and type(observed.get("pid")) is int
                and type(observed.get("start_ticks")) is int
                and isinstance(child_json, Mapping)
                and child_json.get("status") == "PAIR_TOLD_UPDATED"
                and child_json.get("pair_id") == pair.pair_id
                and child_json.get("iteration") == pair.iteration
                and isinstance(tell, Mapping) and tell.get("status") == "UPDATED"
                and isinstance(matching_tell, Mapping)
                and matching_tell.get("iteration") == pair.iteration
                and matching_tell.get("tell_result", {}).get("status") == "UPDATED"
                and session_after.controller.iteration == index
                and session_after.pending_pair is None and not session_after.halted)
            no_update = (
                type(observed.get("return_code")) is int
                and observed.get("return_code") == 2
                and isinstance(child_json, Mapping)
                and child_json.get("status") == "PAIR_TOLD_NOT_UPDATED"
                and child_json.get("pair_id") == pair.pair_id
                and isinstance(tell, Mapping) and tell.get("status") == "NO_UPDATE"
                and isinstance(matching_tell, Mapping)
                and matching_tell.get("tell_result", {}).get("status") == "NO_UPDATE")
            status = "COMPLETE_UPDATED" if exact_updated else "NO_UPDATE" if no_update else "UNKNOWN"
            body = {"schema": PAIR_RECEIPT_SCHEMA, "status": status,
                    "coordinator_id": self.manifest["coordinator_id"],
                    "pair_index": index, "iteration": pair.iteration, "pair_id": pair.pair_id,
                    "plus_proposal_id": pair.plus.proposal_id,
                    "minus_proposal_id": pair.minus.proposal_id,
                    "intent_pin": intent_pin,
                    "process_start_pin": observed.get("launch_pin"),
                    "child_pid": observed.get("pid"),
                    "child_start_ticks": observed.get("start_ticks"),
                    "child_exit_code": observed.get("return_code"),
                    "started_utc": observed.get("started_utc"),
                    "ended_utc": observed.get("ended_utc"),
                    "argv": intent["argv"],
                    "stdout_pin": {"path": observed.get("stdout_path"),
                                   "bytes": observed.get("stdout_bytes"),
                                   "sha256": observed.get("stdout_sha256")},
                    "stderr_pin": {"path": observed.get("stderr_path"),
                                   "bytes": observed.get("stderr_bytes"),
                                   "sha256": observed.get("stderr_sha256")},
                    "child_result": child_json,
                    "session_readback": readback,
                    "exception": observed.get("exception"),
                    "relaunch_prohibited": True}
            receipt_pin = _write_json_exclusive(directory / "process_receipt.json", body,
                                                "receipt_sha256")
            results.append({"status": status, "pair_index": index, "pair_id": pair.pair_id,
                            "receipt_pin": receipt_pin})
            if status != "COMPLETE_UPDATED":
                return {"status": status, "completed_pairs": index - 1,
                        "stop_pair_index": index, "pairs": results}
        summary_path = self.root / "complete_summary.json"
        if summary_path.exists():
            summary, _raw, pin = _read_self_json(summary_path, "coordinator complete summary", "summary_sha256")
            if summary.get("status") != "COMPLETE_20_UPDATED" or summary.get("completed_pairs") != MAX_PAIRS:
                raise CoordinatorError("existing coordinator summary is not a valid complete record")
            return {"status": summary["status"], "completed_pairs": MAX_PAIRS,
                    "summary_pin": pin, "pairs": results}
        summary = {"schema": "fit40_spsa_twenty_pair_coordinator_summary_v1",
                   "status": "COMPLETE_20_UPDATED", "coordinator_id": self.manifest["coordinator_id"],
                   "completed_pairs": MAX_PAIRS,
                   "pair_receipt_pins": [item["receipt_pin"] for item in results],
                   "completed_utc": _utc_now()}
        summary_pin = _write_json_exclusive(self.root / "complete_summary.json", summary,
                                            "summary_sha256")
        return {"status": "COMPLETE_20_UPDATED", "completed_pairs": MAX_PAIRS,
                "summary_pin": summary_pin, "pairs": results}


def _start_detached(args: argparse.Namespace) -> int:
    coordinator = TwentyPairCoordinator.create(
        coordinator_dir=args.coordinator_dir, session_dir=args.session_dir,
        repo_root=args.repo_root, python_executable=args.python,
        dispatch_root=args.dispatch_root,
        recovery_lineage_receipt=args.recovery_lineage_receipt)
    receipt = coordinator.start_detached()
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0 if receipt.get("status") == "STARTED" else 2


def _run_cli(args: argparse.Namespace) -> int:
    coordinator = TwentyPairCoordinator.open(args.coordinator_dir)
    if (Path(args.repo_root).resolve(strict=True) != Path(coordinator.manifest["repo_root"])
            or Path(args.session_dir).resolve(strict=True) != Path(coordinator.manifest["session_dir"])
            or str(_validate_dispatch_storage_root(args.dispatch_root, create=False))
            != coordinator.manifest.get("dispatch_storage_root")):
        raise CoordinatorError("run arguments differ from the sealed coordinator/session/dispatch roots")
    coordinator._verify_runtime_sources()
    # A detached child must match the parent-observed PID/start-tick receipt.
    start_path = coordinator.root / "coordinator_start_receipt.json"
    start_intent_path = coordinator.root / "coordinator_start_intent.json"
    deadline = time.monotonic() + 10.0
    while not start_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not start_path.is_file() or not start_intent_path.is_file():
        raise CoordinatorError("run requires a prior detached start receipt")
    start, _raw, _pin = _read_self_json(start_path, "coordinator start receipt", "receipt_sha256")
    intent, _raw, intent_pin = _read_self_json(
        start_intent_path, "coordinator start intent", "intent_sha256")
    manifest_pin = _self_file_pin(coordinator.root / "coordinator_manifest.json",
                                  "coordinator manifest", "coordinator_sha256")
    source_pin = _file_pin(Path(__file__).resolve(strict=True))
    runtime = coordinator.manifest.get("python_runtime")
    if not isinstance(runtime, Mapping):
        raise CoordinatorError("sealed coordinator Python runtime is missing")
    actual_runtime = _python_pin(Path(str(runtime.get("invoked_path", ""))))
    expected_argv = [str(actual_runtime["invoked_path"]),
                     str(Path(__file__).resolve(strict=True)), "run",
                     "--repo-root", str(Path(args.repo_root).resolve(strict=True)),
                     "--session-dir", str(Path(args.session_dir).resolve(strict=True)),
                     "--coordinator-dir", str(coordinator.root),
                     "--dispatch-root", str(coordinator.manifest["dispatch_storage_root"])]
    if (intent.get("schema") != START_INTENT_SCHEMA
            or intent.get("coordinator_id") != coordinator.manifest.get("coordinator_id")
            or intent.get("coordinator_id") != start.get("coordinator_id")
            or intent.get("coordinator_manifest_pin") != manifest_pin
            or intent.get("coordinator_source_pin") != source_pin
            or intent.get("python_runtime") != dict(runtime)
            or intent.get("dispatch_storage_root") != coordinator.manifest.get("dispatch_storage_root")
            or _python_pin(Path(str(runtime.get("invoked_path", "")))) != dict(runtime)
            or intent.get("argv") != expected_argv
            or start.get("schema") != START_RECEIPT_SCHEMA
            or start.get("status") != "STARTED"
            or start.get("coordinator_id") != intent.get("coordinator_id")
            or start.get("intent_pin") != intent_pin
            or start.get("coordinator_manifest_pin") != manifest_pin
            or start.get("coordinator_source_pin") != source_pin
            or start.get("python_runtime") != dict(runtime)
            or start.get("dispatch_storage_root") != coordinator.manifest.get("dispatch_storage_root")
            or start.get("argv") != expected_argv
            or type(start.get("child_pid")) is not int
            or type(start.get("child_start_ticks")) is not int
            or start.get("child_pid") != os.getpid()
            or _proc_start_ticks(os.getpid()) != start.get("child_start_ticks")):
        raise CoordinatorError("run process does not match the parent-captured detached start receipt")
    result = coordinator.run()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result.get("status") == "COMPLETE_20_UPDATED" else 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("start-detached", help="create durable state and spawn the 20-pair coordinator")
    start.add_argument("--repo-root", type=Path, required=True)
    start.add_argument("--session-dir", type=Path, required=True)
    start.add_argument("--coordinator-dir", type=Path, required=True)
    start.add_argument("--dispatch-root", type=Path, required=True,
                       help="fresh /dev/shm root for pair model outputs; control files stay in coordinator-dir")
    start.add_argument("--recovery-lineage-receipt", type=Path, required=True,
                       help="self-hashed receipt binding the fresh session to verified pre-native failures")
    start.add_argument("--python", type=Path, default=Path(sys.executable))
    run = sub.add_parser("run", help="continue only the parent-captured detached coordinator process")
    run.add_argument("--repo-root", type=Path, required=True)
    run.add_argument("--session-dir", type=Path, required=True)
    run.add_argument("--coordinator-dir", type=Path, required=True)
    run.add_argument("--dispatch-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "start-detached":
            return _start_detached(args)
        if args.command == "run":
            return _run_cli(args)
        raise CoordinatorError("unsupported command")
    except Exception as exc:
        print("coordinator failed closed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CoordinatorError", "TwentyPairCoordinator", "main"]
