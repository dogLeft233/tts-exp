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
BOOTSTRAP_REL = "scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_bootstrap_cli_attempt17.py"
SESSION_REL = "scripts/experiments/cross_tfg_audio_head_20261003/fit40_durable_spsa_session_attempt08.py"
BOOTSTRAP_PIN = (29_245, "284dcabf2f7a032805fa3321ffdedea21deee167e77000e4075d382d4f3c3840")
SESSION_PIN = (92_810, "10e50009a12e7cf6f4286e4d30a9cf36668a6425d6b4b82247b837887f9edd75")
DISPATCH_MOUNT_ROOT = Path("/dev/shm")
DISPATCH_ROOT_NAME = "fit40_joint_spsa_execution_clean_rerun_attempt03"
COORDINATOR_SCHEMA = "fit40_spsa_twenty_pair_coordinator_v4"
START_INTENT_SCHEMA = "fit40_spsa_coordinator_start_intent_v2"
START_RECEIPT_SCHEMA = "fit40_spsa_coordinator_start_receipt_v2"
PAIR_INTENT_SCHEMA = "fit40_spsa_pair_launch_intent_v2"
PAIR_START_SCHEMA = "fit40_spsa_pair_process_start_v1"
PAIR_RECEIPT_SCHEMA = "fit40_spsa_pair_process_receipt_v1"
PAIR_READBACK_SCHEMA = "fit40_spsa_pair_unknown_readback_v1"
TERMINAL_STATUSES = {"COMPLETE_UPDATED", "NO_UPDATE", "UNKNOWN"}
MAX_PAIRS = 20
SESSION_MANIFEST_SCHEMA = "fit40_durable_spsa_session_manifest_v2"
REPLAY_PREFIX_SCHEMA = "fit40_completed_pair_replay_prefix_recovery_v1"
REPLAY_PREFIX_STATUS = "COMPLETE_PAIR_REPLAY_PREFIX_RECOVERED_POST_TELL"
REPLAY_LINEAGE_SCHEMA = "fit40_completed_pair_replay_lineage_v1"
REPLAY_LINEAGE_STATUS = "READY_TO_IMPORT_VERIFIED_COMPLETE_PAIR_TERMINALS"
REPLAY_CONTINUATION_KIND = "existing_pair_terminal_replay_no_native_dispatch"
REPLAY_PREFIX_COUNT = 1
REPLAY_DRIVER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/fit40_replay_completed_pair_attempt04.py"
REPLAY_DRIVER_PIN = (48_242, "c0e940f198e2e38660e9d21142b5718f86cfd4b85257d650ccf172bfc05e467f")
RECOVERY_FINALIZER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/fit40_finalize_completed_pair_replay_attempt03.py"
RECOVERY_FINALIZER_PIN = (24_166, "b02fa9a91a571bc8bcb2e58ad4f3cbf2d88834044766a84b896ccf990a6028cd")
REPLAY_DRIVER_FAILURE = "AttributeError: 'SPSATellResult' object has no attribute 'pair_id'"
REPLAY_FAILURE_CAPTURE_PINS = {
    "stdout_pin": (0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    "stderr_pin": (140, "34c7327e46209f7ff522f07086f44d3988ff7859f1ae64cfe100a0e7613b3740"),
    "ssh_return_pin": (335, "72587f171720edc4cded4fbd58458f8a3c917cf6eaa00d4a8ba8ca31c6be8729"),
}
RECOVERY_RECEIPT_LEAF = "fit40_completed_pair_replay_completion_attempt05.json"
RECOVERY_CAPTURE_LEAF = "fit40_completed_pair_replay_attempt04_failure_capture"
PRIOR_EVENT1_PATH = Path(
    "/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/"
    "job_control/fit40_durable_spsa_session_attempt07/events/event_000001.json"
)
PRIOR_EVENT1_BYTES = 10_014
PRIOR_EVENT1_SHA256 = "9036c1566c3dbc5fa61d9061ae2982671a52b801d7524bd19ad1849b83f4dd58"
PRIOR_EVENT1_SELF_SHA256 = "ecc635d4fe8368bf70fdb2e452b5a75862a5bb47d885999f72668d1be5ad655c"


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


def _expected_recovery_receipt_path(session_root: Path) -> Path:
    return (session_root.parent / RECOVERY_RECEIPT_LEAF).resolve(strict=True)


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


def _validate_clean_first_ask(session: Any, pair: Any) -> None:
    """Require a fresh session's first durable ask to reproduce prior event1 exactly."""
    prior, prior_raw, prior_pin = _read_self_json(
        PRIOR_EVENT1_PATH, "prior immutable PAIR_ASKED event", "event_sha256")
    if (len(prior_raw) != PRIOR_EVENT1_BYTES
            or _sha(prior_raw) != PRIOR_EVENT1_SHA256
            or prior.get("event_sha256") != PRIOR_EVENT1_SELF_SHA256
            or prior_pin.get("self_sha256") != PRIOR_EVENT1_SELF_SHA256
            or prior.get("event_type") != "PAIR_ASKED"
            or prior.get("event_index") != 1):
        raise CoordinatorError("prior first-ask event changed from its approved raw/self pin")
    events = getattr(session, "events", None)
    if not isinstance(events, list) or len(events) != 2:
        raise CoordinatorError("fresh session must contain only initialization and its first ask")
    current = events[1]
    if (not isinstance(current, Mapping)
            or events[0].get("event_type") != "SESSION_INITIALIZED"
            or current.get("event_type") != "PAIR_ASKED"
            or current.get("event_index") != 1
            or current.get("pair_id") != pair.pair_id
            or current.get("iteration") != pair.iteration
            or _canonical(_module_plain(pair)) != _canonical(prior.get("ask_pair"))
            or _canonical(current.get("ask_pair")) != _canonical(prior.get("ask_pair"))
            or _canonical(current.get("controller_state_after"))
               != _canonical(prior.get("controller_state_after"))):
        raise CoordinatorError("fresh first ask or post-ask controller/RNG state differs from prior event1")

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


def _load_pinned_session(repo_root: Path, session_dir: Path) -> Any:
    """Load and replay only the session module selected by the frozen bootstrap."""
    _source_closure(repo_root)
    bootstrap_path = repo_root / BOOTSTRAP_REL
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fit40_pinned_bootstrap_for_coordinator21", bootstrap_path)
    if spec is None or spec.loader is None:
        raise CoordinatorError("cannot load the pinned bootstrap CLI")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    modules = bootstrap._load_frozen_modules(repo_root)
    expected_session_path = (repo_root / SESSION_REL).resolve(strict=True)
    if Path(modules["session"].__file__).resolve(strict=True) != expected_session_path:
        raise CoordinatorError("loaded durable-session module origin differs from the pinned session source")
    return modules["session"].DurableSPSASession.resume(session_dir=session_dir)


def _read_verified_link(link: Any, label: str, self_field: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(link, Mapping) or set(link) != {"path", "bytes", "sha256", "self_sha256"}:
        raise CoordinatorError("%s link has an unexpected shape" % label)
    path = Path(str(link.get("path", "")))
    value, _raw, observed = _read_self_json(path, label, self_field)
    if observed != dict(link):
        raise CoordinatorError("%s raw/self pin does not match its linked bytes" % label)
    return value, observed


def _verify_replay_source_pins(source_pins: Any, *, lineage_source_pins: Any,
                               session: Any, old_session_manifest: Mapping[str, Any],
                               repo_root: Path) -> None:
    """Bind the recovered receipt to its READY lineage, failed driver, and finalizer."""
    lineage_roles = {"old_durable_session_source", "replay_durable_session_source",
                     "replay_driver_source"}
    finalizer_role = "completion_recovery_finalizer_source"
    receipt_roles = lineage_roles | {finalizer_role}
    if (not isinstance(source_pins, Mapping) or set(source_pins) != receipt_roles
            or not isinstance(lineage_source_pins, Mapping)
            or set(lineage_source_pins) != lineage_roles
            or any(source_pins.get(role) != lineage_source_pins.get(role)
                   for role in lineage_roles)):
        raise CoordinatorError("recovered completion source pins differ from the READY lineage")

    observed: dict[str, dict[str, Any]] = {}
    for role in sorted(receipt_roles):
        pin = source_pins[role]
        if (not isinstance(pin, Mapping)
                or set(pin) != {"path", "bytes", "sha256", "self_sha256"}
                or pin.get("self_sha256") is not None
                or type(pin.get("bytes")) is not int
                or pin.get("bytes") < 1):
            raise CoordinatorError("replay source pin has an unexpected raw-file shape: %s" % role)
        path = Path(str(pin.get("path", "")))
        if not path.is_absolute():
            raise CoordinatorError("replay source pin is not absolute: %s" % role)
        observed[role] = {**_file_pin(path), "self_sha256": None}
        if observed[role] != dict(pin):
            raise CoordinatorError("replay source pin changed on disk: %s" % role)

    driver_path = (repo_root / REPLAY_DRIVER_REL).resolve(strict=True)
    driver_pin = {**_file_pin(driver_path), "self_sha256": None}
    if ((driver_pin["bytes"], driver_pin["sha256"]) != REPLAY_DRIVER_PIN
            or observed["replay_driver_source"] != driver_pin):
        raise CoordinatorError("replay completion was not produced by the pinned replay driver")

    finalizer_path = (repo_root / RECOVERY_FINALIZER_REL).resolve(strict=True)
    finalizer_pin = {**_file_pin(finalizer_path), "self_sha256": None}
    if ((finalizer_pin["bytes"], finalizer_pin["sha256"]) != RECOVERY_FINALIZER_PIN
            or observed[finalizer_role] != finalizer_pin):
        raise CoordinatorError("recovered completion was not produced by the pinned finalizer")

    def session_source_pin(manifest: Mapping[str, Any], label: str) -> dict[str, Any]:
        entries = [entry for entry in manifest.get("source_file_table", [])
                   if isinstance(entry, Mapping)
                   and entry.get("role") == "durable_session_source"]
        if len(entries) != 1:
            raise CoordinatorError("%s manifest does not contain one durable-session source" % label)
        entry = entries[0]
        path = Path(str(entry.get("declared_path", "")))
        if not path.is_absolute():
            raise CoordinatorError("%s durable-session source path is not absolute" % label)
        return {"path": str(path.resolve(strict=True)), "bytes": entry.get("bytes"),
                "sha256": entry.get("sha256"), "self_sha256": None}

    expected_new = session_source_pin(session.manifest, "replay")
    expected_old = session_source_pin(old_session_manifest, "old")
    if (observed["replay_durable_session_source"] != expected_new
            or observed["old_durable_session_source"] != expected_old):
        raise CoordinatorError("replay source pins differ from their old/fresh session manifests")


def _verify_recovery_provenance(value: Any, source_pins: Mapping[str, Any],
                                expected_capture_dir: Path) -> None:
    if not isinstance(value, Mapping):
        raise CoordinatorError("recovered completion omits its recovery provenance")
    if set(value) != {
            "recovery_schema", "recovery_status", "original_driver_failure",
            "replay04_source_pin", "completion_recovery_finalizer_source_pin",
            "fresh_session_event_count_before_finalization", "persisted_tell_event_index",
            "persisted_tell_status", "tell_rerun", "ask_rerun",
            "dispatch_or_model_process_started_by_finalizer",
            "official_rows_revalidated_by_finalizer", "overall_pair_child_exit_code"}:
        raise CoordinatorError("recovered completion provenance has an unexpected field set")
    expected = {
        "recovery_schema": "fit40_completed_pair_replay_post_tell_recovery_v1",
        "recovery_status": "RECOVERED_FROM_PERSISTED_UPDATED_TELL_AFTER_WRAPPER_EXIT",
        "fresh_session_event_count_before_finalization": 7,
        "persisted_tell_event_index": 6,
        "persisted_tell_status": "UPDATED",
        "tell_rerun": False,
        "ask_rerun": False,
        "dispatch_or_model_process_started_by_finalizer": False,
        "official_rows_revalidated_by_finalizer": 48,
        "overall_pair_child_exit_code": None,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise CoordinatorError("recovered completion provenance does not prove a read-only post-tell seal")
    if (type(value.get("fresh_session_event_count_before_finalization")) is not int
            or type(value.get("persisted_tell_event_index")) is not int
            or type(value.get("official_rows_revalidated_by_finalizer")) is not int
            or value.get("tell_rerun") is not False
            or value.get("ask_rerun") is not False
            or value.get("dispatch_or_model_process_started_by_finalizer") is not False
            or value.get("overall_pair_child_exit_code") is not None):
        raise CoordinatorError("recovered completion provenance has invalid value types")
    if (value.get("completion_recovery_finalizer_source_pin")
            != source_pins.get("completion_recovery_finalizer_source")
            or value.get("replay04_source_pin") != source_pins.get("replay_driver_source")):
        raise CoordinatorError("recovery provenance source pins differ from the receipt source table")

    failure = value.get("original_driver_failure")
    if (not isinstance(failure, Mapping)
            or set(failure) != {
                "status", "driver_source_pin", "driver_return_code", "ssh_return_code",
                "failure_error", "staged_failure_capture_directory", "stdout_pin", "stderr_pin",
                "ssh_return_pin", "started_utc", "finished_utc",
                "replay_driver_module_processes_started"}
            or failure.get("status") != "PAIR_REPLAY_FAILED_AFTER_DURABLE_TELL"
            or failure.get("driver_source_pin") != source_pins.get("replay_driver_source")
            or type(failure.get("driver_return_code")) is not int
            or failure.get("driver_return_code") != 2
            or type(failure.get("ssh_return_code")) is not int
            or failure.get("ssh_return_code") != 2
            or failure.get("failure_error") != REPLAY_DRIVER_FAILURE
            or type(failure.get("replay_driver_module_processes_started")) is not int
            or failure.get("replay_driver_module_processes_started") != 1):
        raise CoordinatorError("recovery provenance does not preserve the original post-tell rc2")
    capture_dir = Path(str(failure.get("staged_failure_capture_directory", "")))
    if (not capture_dir.is_absolute() or capture_dir.is_symlink() or not capture_dir.is_dir()
            or capture_dir.resolve(strict=True) != expected_capture_dir.resolve(strict=True)):
        raise CoordinatorError("original replay04 capture directory is missing or unsafe")
    expected_capture_files = {
        "stdout_pin": "stdout.txt",
        "stderr_pin": "stderr.txt",
        "ssh_return_pin": "ssh_return.json",
    }
    observed_raw: dict[str, bytes] = {}
    for role, leaf in expected_capture_files.items():
        pin = failure.get(role)
        expected_bytes, expected_sha256 = REPLAY_FAILURE_CAPTURE_PINS[role]
        if (not isinstance(pin, Mapping)
                or set(pin) != {"path", "bytes", "sha256", "self_sha256"}
                or pin.get("self_sha256") is not None
                or pin.get("bytes") != expected_bytes
                or pin.get("sha256") != expected_sha256):
            raise CoordinatorError("original replay04 %s does not match its captured bytes" % role)
        expected_path = (capture_dir / leaf).resolve(strict=True)
        if Path(str(pin.get("path", ""))).resolve(strict=True) != expected_path:
            raise CoordinatorError("original replay04 capture pin is outside its exact capture leaf")
        observed_pin = {**_file_pin(expected_path), "self_sha256": None}
        if observed_pin != dict(pin):
            raise CoordinatorError("original replay04 capture file changed: %s" % role)
        observed_raw[role] = _regular_bytes(expected_path, "original replay04 %s" % role)

    if observed_raw["stdout_pin"] != b"":
        raise CoordinatorError("original replay04 stdout is not the captured empty file")
    stderr_lines = observed_raw["stderr_pin"].decode("utf-8").splitlines()
    if (len(stderr_lines) != 2 or stderr_lines[-1] != "REMOTE_DRIVER_EXIT_CODE=2"):
        raise CoordinatorError("original replay04 stderr does not preserve the observed rc2")
    try:
        failure_record = json.loads(stderr_lines[0])
        transport = json.loads(observed_raw["ssh_return_pin"].decode("utf-8-sig"))
    except Exception as exc:
        raise CoordinatorError("original replay04 failure capture is not valid JSON") from exc
    if (failure_record.get("error") != REPLAY_DRIVER_FAILURE
            or failure_record.get("status") != "PAIR_REPLAY_FAILED"
            or transport.get("ssh_return_code") != 2):
        raise CoordinatorError("original replay04 failure capture disagrees with recovery receipt")


def _verify_replay_prefix_receipt(
    receipt_path: Path, *, session: Any, session_manifest_pin: Mapping[str, Any],
    repo_root: Path,
    expected_receipt_pin: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the recovered UPDATED tell against session09's full event replay.

    The prefix is a durable session tell plus an explicit post-tell recovery
    record; it is never treated as a synthetic process/child receipt.
    """
    receipt, _raw, receipt_pin = _read_self_json(
        receipt_path, "completed-pair replay-prefix receipt", "receipt_sha256")
    if expected_receipt_pin is not None and receipt_pin != dict(expected_receipt_pin):
        raise CoordinatorError("replay-prefix receipt changed after coordinator creation")
    if (receipt.get("schema") != REPLAY_PREFIX_SCHEMA
            or receipt.get("status") != REPLAY_PREFIX_STATUS
            or receipt.get("continuation_kind") != REPLAY_CONTINUATION_KIND
            or receipt.get("iteration") != 0
            or not isinstance(receipt.get("pair_id"), str)
            or not isinstance(receipt.get("ask_pair"), Mapping)
            or not isinstance(receipt.get("tell_result"), Mapping)
            or receipt.get("tell_result", {}).get("status") != "UPDATED"
            or not isinstance(receipt.get("controller_state_after"), Mapping)
            or not isinstance(receipt.get("old_terminal_provenance"), Mapping)
            or not isinstance(receipt.get("recovery_provenance"), Mapping)
            or receipt.get("native_work_for_replayed_prefix") != {
                "new_model_processes_started": 0,
                "existing_candidate_rows_reused_once": 48,
                "official_npz_revalidated": 48,
                "overall_pair_child_exit_code": None,
            }):
        raise CoordinatorError("replay-prefix receipt schema or first UPDATED tell identity is invalid")
    if not isinstance(receipt.get("source_pins"), Mapping):
        raise CoordinatorError("recovered completion has no source-pin table")
    expected_capture_dir = receipt_path.parent / RECOVERY_CAPTURE_LEAF
    _verify_recovery_provenance(receipt["recovery_provenance"], receipt["source_pins"],
                                expected_capture_dir)

    lineage, _lineage_pin = _read_verified_link(
        receipt.get("lineage_receipt"), "completed-pair READY lineage receipt", "receipt_sha256")
    old_provenance = receipt["old_terminal_provenance"]
    if (lineage.get("schema") != REPLAY_LINEAGE_SCHEMA
            or lineage.get("status") != REPLAY_LINEAGE_STATUS
            or lineage.get("continuation_kind") != REPLAY_CONTINUATION_KIND
            or lineage.get("pair_id") != receipt.get("pair_id")
            or lineage.get("iteration") != 0
            or lineage.get("ask_pair") != receipt.get("ask_pair")
            or lineage.get("ask_pair_sha256") != _sha(_canonical(receipt.get("ask_pair")))
            or receipt.get("replay_adoption_intents") is None
            or lineage.get("replay_adoption_intents") is None
            or lineage.get("fresh_initial_selector_receipt") != receipt.get("initial_selector_receipt")
            or lineage.get("composed_family_plan") != receipt.get("composed_family_plan")
            or lineage.get("execution_package_manifest") != receipt.get("execution_package_manifest")
            or lineage.get("initial_ingest_index") != receipt.get("initial_ingest_index")
            or old_provenance != {
                "old_session_manifest": lineage.get("old_session_manifest"),
                "old_session_events": lineage.get("old_session_events"),
                "old_point_entries": lineage.get("old_point_entries"),
                "old_outer_pair_process_failure": lineage.get("old_outer_pair_process_failure"),
                "original_pair_packet_index": lineage.get("original_pair_packet_index"),
            }):
        raise CoordinatorError("completion receipt differs from its verified READY lineage")

    session_manifest_links = {
        "initial_selector_receipt": "initial_selector_receipt",
        "composed_family_plan": "composed_family_plan",
        "execution_package_manifest": "execution_package_manifest",
        "initial_ingest_index": "initial_ingest_index",
    }
    if any(receipt.get(receipt_key) != session.manifest.get(manifest_key)
           for receipt_key, manifest_key in session_manifest_links.items()):
        raise CoordinatorError("recovered completion links differ from the replayed session manifest")

    failure = old_provenance.get("old_outer_pair_process_failure")
    if (not isinstance(failure, Mapping)
            or failure.get("outer_status") != "UNKNOWN"
            or failure.get("outer_child_exit_code") != 2
            or failure.get("old_event_count") != 6
            or failure.get("old_pair_tell") is not None
            or failure.get("relaunch_prohibited") is not True):
        raise CoordinatorError("replay lineage does not preserve the original UNKNOWN/rc2 failure")

    if (lineage.get("official_candidate_evidence_counts") != {
            "candidate_rows": 48, "official_raw_npz": 48,
            "official_source_frame_maps": 48,
            "model_child_receipts_with_numeric_rc0": 6,
        }
            or lineage.get("runtime_effects") != {
                "native_dispatcher_imported": False,
                "new_model_processes_started": 0,
                "new_audio_or_video_rendered": False,
                "new_score_npz_generated": False,
                "existing_official_npz_revalidated": 48,
            }):
        raise CoordinatorError("replay lineage candidate or no-dispatch evidence is incomplete")

    new_manifest, manifest_pin = _read_verified_link(
        receipt.get("new_session_manifest"), "replay session manifest", "manifest_sha256")
    session_root = Path(session.root).resolve(strict=True)
    expected_manifest_path = (session_root / "session_manifest.json").resolve(strict=True)
    if Path(manifest_pin["path"]).resolve(strict=True) != expected_manifest_path:
        raise CoordinatorError("replay receipt manifest is outside the continued session")
    actual_manifest, _manifest_raw, actual_manifest_pin = _read_self_json(
        expected_manifest_path, "continued session manifest", "manifest_sha256")
    if (actual_manifest_pin != dict(session_manifest_pin)
            or actual_manifest_pin != manifest_pin
            or actual_manifest.get("session_id") != session.manifest.get("session_id")):
        raise CoordinatorError("replay receipt does not bind this exact durable session")

    lineage_session = lineage.get("session_lineage")
    if (not isinstance(lineage_session, Mapping)
            or lineage_session.get("fresh_session_id") != actual_manifest.get("session_id")
            or lineage_session.get("fresh_session_manifest") != receipt.get("new_session_manifest")
            or lineage_session.get("old_event_count_before_replay") != 6
            or lineage_session.get("fresh_event_count_at_lineage_creation") != 2):
        raise CoordinatorError("READY lineage does not bind the old and fresh session identities")

    for receipt_key, manifest_key in (
        ("initial_selector_receipt", "initial_selector_receipt"),
        ("composed_family_plan", "composed_family_plan"),
        ("execution_package_manifest", "execution_package_manifest"),
    ):
        if receipt.get(receipt_key) != session.manifest.get(manifest_key):
            raise CoordinatorError("replay prefix changed session %s lineage" % manifest_key)

    event_links = receipt.get("new_session_events")
    if not isinstance(event_links, list) or len(event_links) != 7 or len(session.events) < 7:
        raise CoordinatorError("replay prefix must bind the exact first seven one-pair session events")
    observed_events = []
    for index, link in enumerate(event_links):
        event, event_pin = _read_verified_link(
            link, "replay session event %d" % index, "event_sha256")
        expected_path = (session_root / "events" / ("event_%06d.json" % index)).resolve(strict=True)
        if Path(event_pin["path"]).resolve(strict=True) != expected_path:
            raise CoordinatorError("replay event link is outside its canonical session leaf")
        if event != session.events[index]:
            raise CoordinatorError("replay event differs from deterministic session replay")
        if event.get("event_index") != index or event.get("session_id") != session.manifest.get("session_id"):
            raise CoordinatorError("replay event identity or ordering is invalid")
        observed_events.append(event)
    expected_types = ["SESSION_INITIALIZED", "PAIR_ASKED", "POINT_DISPATCH_INTENT",
                      "POINT_TERMINAL", "POINT_DISPATCH_INTENT", "POINT_TERMINAL", "PAIR_TOLD"]
    if [event.get("event_type") for event in observed_events] != expected_types:
        raise CoordinatorError("replay session event sequence is not one complete plus/minus pair")
    if lineage_session.get("fresh_session_initial_events") != event_links[:2]:
        raise CoordinatorError("READY lineage initial-event pins differ from the replayed session")
    ask_event, tell_event = observed_events[1], observed_events[6]
    ask_pair = receipt["ask_pair"]
    pair_id = receipt["pair_id"]
    prior_ask, prior_ask_raw, prior_ask_pin = _read_self_json(
        PRIOR_EVENT1_PATH, "prior immutable first PAIR_ASKED event", "event_sha256")
    if (len(prior_ask_raw) != PRIOR_EVENT1_BYTES
            or _sha(prior_ask_raw) != PRIOR_EVENT1_SHA256
            or prior_ask_pin.get("self_sha256") != PRIOR_EVENT1_SELF_SHA256
            or prior_ask.get("event_type") != "PAIR_ASKED"
            or prior_ask.get("event_index") != 1):
        raise CoordinatorError("prior deterministic first ask changed from its approved pin")
    old_manifest, _old_manifest_pin = _read_verified_link(
        old_provenance.get("old_session_manifest"), "old durable-session manifest", "manifest_sha256")
    old_events = old_provenance.get("old_session_events")
    old_session_root = Path(str(old_provenance["old_session_manifest"]["path"])).parent.resolve(strict=True)
    expected_old_types = ["SESSION_INITIALIZED", "PAIR_ASKED", "POINT_DISPATCH_INTENT",
                          "POINT_TERMINAL", "POINT_DISPATCH_INTENT", "POINT_TERMINAL"]
    observed_old_events = []
    for index, link in enumerate(old_events):
        old_event, old_event_pin = _read_verified_link(
            link, "old session event %d" % index, "event_sha256")
        expected_old_path = (old_session_root / "events" / ("event_%06d.json" % index)).resolve(strict=True)
        if (Path(old_event_pin["path"]).resolve(strict=True) != expected_old_path
                or old_event.get("event_index") != index
                or old_event.get("session_id") != old_manifest.get("session_id")
                or old_event.get("event_type") != expected_old_types[index]):
            raise CoordinatorError("old replay source event chain is not the exact six-event prefix")
        observed_old_events.append(old_event)
    old_ask_event = observed_old_events[1]
    if (old_events[1] != prior_ask_pin
            or old_ask_event.get("ask_pair") != prior_ask.get("ask_pair")
            or old_ask_event.get("controller_state_after") != prior_ask.get("controller_state_after")
            or prior_ask.get("ask_pair") != ask_pair):
        raise CoordinatorError("old session event1 is not the pinned deterministic first ask")
    if (ask_pair != ask_event.get("ask_pair")
            or ask_pair.get("pair_id") != pair_id
            or ask_pair.get("iteration") != 0
            or ask_event.get("pair_id") != pair_id
            or ask_event.get("iteration") != 0
            or ask_event.get("ask_pair") != prior_ask.get("ask_pair")
            or ask_event.get("controller_state_after") != prior_ask.get("controller_state_after")
            or lineage.get("old_controller_state_after_ask") != prior_ask.get("controller_state_after")
            or lineage.get("fresh_controller_state_after_ask") != ask_event.get("controller_state_after")
            or tell_event.get("pair_id") != pair_id
            or tell_event.get("iteration") != 0
            or tell_event.get("tell_result") != receipt.get("tell_result")
            or tell_event.get("tell_result", {}).get("status") != "UPDATED"
            or tell_event.get("controller_state_after") != receipt.get("controller_state_after")
            or receipt.get("controller_state_after", {}).get("iteration") != 1
            or receipt.get("controller_state_after", {}).get("pending_pair") is not None
            or receipt.get("controller_state_after", {}).get("halted_reason") is not None):
        raise CoordinatorError("replayed tell does not match the first completed pair/controller state")
    tell_link = receipt.get("tell_event")
    if (tell_link != event_links[6]
            or receipt.get("tell_journal_entry") != tell_event.get("journal_entry")):
        raise CoordinatorError("replay receipt tell-event pin differs from its ordered event list")

    _verify_replay_source_pins(
        receipt.get("source_pins"), lineage_source_pins=lineage.get("source_pins"),
        session=session, old_session_manifest=old_manifest, repo_root=repo_root)

    old_points = old_provenance.get("old_point_entries")
    lineage_terminals = lineage.get("point_terminals")
    lineage_adoptions = lineage.get("replay_adoption_intents")
    completion_adoptions = receipt.get("replay_adoption_intents")
    points = receipt.get("point_entries")
    if (not isinstance(old_events, list) or len(old_events) != 6
            or not isinstance(old_points, Mapping) or set(old_points) != {"plus", "minus"}
            or not isinstance(lineage_terminals, Mapping)
            or set(lineage_terminals) != {"plus", "minus"}
            or not isinstance(points, Mapping) or set(points) != {"plus", "minus"}
            or not isinstance(lineage_adoptions, list) or len(lineage_adoptions) != 2
            or not isinstance(completion_adoptions, list) or len(completion_adoptions) != 2):
        raise CoordinatorError("replay adoption lineage does not contain the exact two-side prefix")
    fresh_adoption_fields = {
        "fresh_session_dispatch_intent_event",
        "fresh_session_point_terminal_event",
        "fresh_point_entry_pin",
    }
    for side, intent_index, terminal_index in (("plus", 2, 3), ("minus", 4, 5)):
        declaration_matches = [item for item in lineage_adoptions
                               if isinstance(item, Mapping) and item.get("side") == side]
        completion_matches = [item for item in completion_adoptions
                              if isinstance(item, Mapping) and item.get("side") == side]
        if len(declaration_matches) != 1 or len(completion_matches) != 1:
            raise CoordinatorError("replay adoption intent is missing or duplicated for %s" % side)
        declaration, adopted = declaration_matches[0], completion_matches[0]
        adopted_base = {key: value for key, value in adopted.items()
                        if key not in fresh_adoption_fields}
        old_point_pin = old_points[side]
        terminal_summary = lineage_terminals[side]
        child_receipts = declaration.get("old_model_child_receipts")
        model_ids = list(session.inputs.family_selection.family.roster.model_ids)
        proposal = ask_pair.get(side)
        if not isinstance(proposal, Mapping):
            raise CoordinatorError("replay ask omits its %s proposal" % side)
        if (adopted_base != dict(declaration)
                or declaration.get("pair_id") != pair_id
                or declaration.get("proposal_id") != proposal.get("proposal_id")
                or declaration.get("iteration") != 0
                or declaration.get("fresh_session_dispatch_intent_event_index") != intent_index
                or declaration.get("fresh_session_point_terminal_event_index") != terminal_index
                or declaration.get("fresh_intent_semantics")
                   != "readback_adoption_only_no_child_or_model_process"
                or declaration.get("old_session_dispatch_intent_event") != old_events[intent_index]
                or declaration.get("old_session_point_terminal_event") != old_events[terminal_index]
                or declaration.get("old_point_entry_pin") != old_point_pin
                or declaration.get("old_point_entry") != terminal_summary.get("point_entry")
                or declaration.get("old_model_child_receipts") != terminal_summary.get("model_child_receipts")
                or terminal_summary.get("side") != side
                or terminal_summary.get("proposal_id") != proposal.get("proposal_id")
                or terminal_summary.get("candidate_row_count") != 24
                or terminal_summary.get("official_raw_npz_count") != 24
                or terminal_summary.get("official_source_frame_map_count") != 24
                or not isinstance(terminal_summary.get("candidate_evidence"), list)
                or len(terminal_summary.get("candidate_evidence", [])) != 24
                or adopted.get("fresh_session_dispatch_intent_event") != event_links[intent_index]
                or adopted.get("fresh_session_point_terminal_event") != event_links[terminal_index]
                or adopted.get("fresh_point_entry_pin") != points[side]
                or not isinstance(child_receipts, list) or len(child_receipts) != 3
                or [item.get("model_id") if isinstance(item, Mapping) else None
                    for item in child_receipts] != model_ids):
            raise CoordinatorError("replay adoption receipt does not bind exact old/new %s events" % side)
        old_intent, observed_old_intent_pin = _read_verified_link(
            old_events[intent_index], "old %s dispatch-intent event" % side, "event_sha256")
        old_terminal, observed_old_terminal_pin = _read_verified_link(
            old_events[terminal_index], "old %s point-terminal event" % side, "event_sha256")
        if (observed_old_intent_pin != declaration["old_session_dispatch_intent_event"]
                or observed_old_terminal_pin != declaration["old_session_point_terminal_event"]
                or old_intent.get("event_type") != "POINT_DISPATCH_INTENT"
                or old_intent.get("proposal") != proposal
                or old_intent.get("proposal_id") != proposal.get("proposal_id")
                or old_intent.get("pair_id") != pair_id
                or old_intent.get("side") != side
                or old_terminal.get("event_type") != "POINT_TERMINAL"
                or old_terminal.get("proposal_id") != proposal.get("proposal_id")
                or old_terminal.get("pair_id") != pair_id
                or old_terminal.get("side") != side
                or old_terminal.get("terminal_status") != "COMPLETE"
                or old_terminal.get("point_entry") != declaration["old_point_entry"]
                or old_terminal.get("point_entry_pin") != old_point_pin):
            raise CoordinatorError("old %s event bodies do not bind the deterministic ask and point pin" % side)
        fresh_intent = observed_events[intent_index]
        fresh_terminal = observed_events[terminal_index]
        if (fresh_intent.get("event_type") != "POINT_DISPATCH_INTENT"
                or fresh_intent.get("proposal_id") != proposal.get("proposal_id")
                or fresh_intent.get("point_id") != proposal.get("method_id")
                or fresh_intent.get("pair_id") != pair_id
                or fresh_intent.get("iteration") != 0
                or fresh_intent.get("side") != side
                or fresh_intent.get("proposal") != proposal
                or fresh_terminal.get("event_type") != "POINT_TERMINAL"
                or fresh_terminal.get("proposal_id") != proposal.get("proposal_id")
                or fresh_terminal.get("point_id") != proposal.get("method_id")
                or fresh_terminal.get("pair_id") != pair_id
                or fresh_terminal.get("iteration") != 0
                or fresh_terminal.get("side") != side
                or fresh_terminal.get("terminal_status") != "COMPLETE"
                or fresh_terminal.get("point_entry") != declaration.get("old_point_entry")
                or fresh_terminal.get("point_entry_pin") != points[side]):
            raise CoordinatorError("fresh %s events do not record the exact readback adoption" % side)
        if (not isinstance(child_receipts, list) or len(child_receipts) != 3
                or any(not isinstance(item, Mapping) or item.get("child_exit_code") != 0
                       for item in child_receipts)):
            raise CoordinatorError("replayed %s adoption omits the three original rc0 model receipts" % side)

    proposals = ask_pair.get("plus"), ask_pair.get("minus")
    for side, proposal in zip(("plus", "minus"), proposals):
        if not isinstance(proposal, Mapping):
            raise CoordinatorError("replay ask is missing its %s proposal" % side)
        point, point_pin = _read_verified_link(
            points[side], "replay %s point entry" % side, "point_entry_sha256")
        key = (proposal.get("proposal_id"), side)
        record = session.point_entries.get(key)
        if (record is None or point_pin != record.get("point_pin")
                or point != record.get("point_doc")
                or record.get("event", {}).get("terminal_status") != "COMPLETE"):
            raise CoordinatorError("replay %s point proof differs from the session terminal readback" % side)
    return receipt_pin


class TwentyPairCoordinator:
    """Append-only pair supervisor. Child/session behavior is injectable for CPU tests."""

    def __init__(self, root: Path, manifest: Mapping[str, Any]):
        self.root = root
        self.manifest = dict(manifest)

    @classmethod
    def create(cls, *, coordinator_dir: str | Path, session_dir: str | Path,
               repo_root: str | Path, python_executable: str | Path,
               dispatch_root: str | Path,
               replay_prefix_receipt_path: str | Path) -> "TwentyPairCoordinator":
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
        replay_path = Path(replay_prefix_receipt_path)
        if replay_path.resolve(strict=True) != _expected_recovery_receipt_path(session_root):
            raise CoordinatorError("recovered prefix receipt is not at its reserved attempt05 leaf")
        replay, _replay_raw, replay_pin = _read_self_json(
            replay_path, "completed-pair replay-prefix receipt", "receipt_sha256")
        session = _load_pinned_session(repo, session_root)
        _verify_replay_prefix_receipt(
            replay_path, session=session, session_manifest_pin=session_manifest_pin,
            repo_root=repo,
            expected_receipt_pin=replay_pin)
        if (len(session.events) != 7 or session.controller.iteration != 1
                or session.pending_pair is not None or session.halted):
            raise CoordinatorError("fresh coordinator must start from the one-tell session snapshot")
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
            "initial_completed_prefix": REPLAY_PREFIX_COUNT,
            "replay_prefix_receipt_pin": replay_pin,
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
                or value.get("max_pairs") != MAX_PAIRS
                or value.get("initial_completed_prefix") != REPLAY_PREFIX_COUNT
                or not isinstance(value.get("replay_prefix_receipt_pin"), Mapping)):
            raise CoordinatorError("coordinator manifest schema/status/limit mismatch")
        _value, _raw, replay_pin = _read_self_json(
            Path(str(value["replay_prefix_receipt_pin"].get("path", ""))),
            "completed-pair replay-prefix receipt", "receipt_sha256")
        session_root = Path(str(value.get("session_dir", ""))).resolve(strict=True)
        if (replay_pin != value["replay_prefix_receipt_pin"]
                or Path(replay_pin["path"]).resolve(strict=True)
                != _expected_recovery_receipt_path(session_root)):
            raise CoordinatorError("coordinator replay-prefix receipt pin changed")
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
        _receipt, _receipt_raw, replay_pin = _read_self_json(
            Path(str(self.manifest["replay_prefix_receipt_pin"]["path"])),
            "completed-pair replay-prefix receipt", "receipt_sha256")
        if (replay_pin != self.manifest.get("replay_prefix_receipt_pin")
                or Path(replay_pin["path"]).resolve(strict=True)
                != _expected_recovery_receipt_path(session)):
            raise CoordinatorError("replay-prefix receipt changed after coordinator creation")
        dispatch_root = _validate_dispatch_storage_root(
            self.manifest.get("dispatch_storage_root", ""), create=False)
        if str(dispatch_root) != self.manifest.get("dispatch_storage_root"):
            raise CoordinatorError("sealed dispatch storage root changed")
        return repo, session, executable

    def _pair_dir(self, index: int) -> Path:
        return self.root / "pairs" / ("pair_%02d" % index)

    def _actual_pair_receipt_pins(self) -> list[dict[str, Any]]:
        pins: list[dict[str, Any]] = []
        for index in range(REPLAY_PREFIX_COUNT + 1, MAX_PAIRS + 1):
            path = self._pair_dir(index) / "process_receipt.json"
            receipt, _raw, pin = _read_self_json(path, "native pair process receipt", "receipt_sha256")
            if (receipt.get("pair_index") != index
                    or receipt.get("status") != "COMPLETE_UPDATED"
                    or type(receipt.get("child_exit_code")) is not int
                    or receipt.get("child_exit_code") != 0):
                raise CoordinatorError("completed execution prefix contains a non-UPDATED or unproven native pair")
            pins.append(pin)
        if len(pins) != MAX_PAIRS - REPLAY_PREFIX_COUNT:
            raise CoordinatorError("final native receipt list does not cover exactly the 19 launched pairs")
        return pins

    def _existing_prefix(self) -> tuple[int, str | None]:
        pair_root = self.root / "pairs"
        if pair_root.is_symlink() or not pair_root.is_dir():
            raise CoordinatorError("pair state directory is missing or unsafe")
        names = sorted(path.name for path in pair_root.iterdir())
        if any(not re.fullmatch(r"pair_[0-9]{2}", name) for name in names):
            raise CoordinatorError("unexpected entry in pair state directory")
        replayed_prefix = self.manifest.get("initial_completed_prefix")
        if replayed_prefix != REPLAY_PREFIX_COUNT:
            raise CoordinatorError("coordinator does not have its required typed replay prefix")
        if "pair_01" in names:
            raise CoordinatorError("pair_01 must remain replay evidence, never a synthetic native child receipt")
        updated = REPLAY_PREFIX_COUNT
        unresolved: str | None = None
        for ordinal, name in enumerate(names, 1):
            index = REPLAY_PREFIX_COUNT + ordinal
            if name != "pair_%02d" % index:
                raise CoordinatorError("pair records are not a contiguous prefix")
            if index > MAX_PAIRS:
                raise CoordinatorError("coordinator contains more than the predeclared 20 pair slots")
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
            expected_packet = directory / "packet"
            expected_argv = [str(self.manifest["python_runtime"]["invoked_path"]),
                             str(Path(self.manifest["repo_root"]) / BOOTSTRAP_REL),
                             "--repo-root", self.manifest["repo_root"], "run-pair",
                             "--session-dir", self.manifest["session_dir"],
                             "--packet-dir", str(expected_packet),
                             "--dispatch-root", str(expected_dispatch)]
            if (intent.get("dispatch_storage_root") != str(expected_dispatch_base)
                    or intent.get("dispatch_root") != str(expected_dispatch)
                    or intent.get("packet_dir") != str(expected_packet)
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
        packet = directory / "packet"
        dispatch_base = Path(self.manifest["dispatch_storage_root"])
        dispatch = dispatch_base / "pairs" / ("pair_%02d" % index) / "dispatch_root"
        proposal_pair = {"pair_id": pair.pair_id, "iteration": pair.iteration,
                         "plus": _module_plain(pair.plus), "minus": _module_plain(pair.minus)}
        if session.pending_pair != pair:
            raise CoordinatorError("pair is not the exact durable pending ask")
        argv = [str(self.manifest["python_runtime"]["invoked_path"]),
                str(repo / BOOTSTRAP_REL), "--repo-root", str(repo), "run-pair",
                "--session-dir", str(session_dir), "--packet-dir", str(packet),
                "--dispatch-root", str(dispatch)]
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
        _verify_replay_prefix_receipt(
            Path(str(self.manifest["replay_prefix_receipt_pin"]["path"])),
            session=session, session_manifest_pin=self.manifest["session_manifest_pin"],
            repo_root=repo,
            expected_receipt_pin=self.manifest["replay_prefix_receipt_pin"])
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
        for index in range(REPLAY_PREFIX_COUNT + 1, completed + 1):
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
            if index == 1:
                _validate_clean_first_ask(session, pair)
            if pair.iteration != index - 1:
                raise CoordinatorError("deterministic controller returned an unexpected pair iteration")
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
            actual_pins = self._actual_pair_receipt_pins()
            if (summary.get("schema") != "fit40_spsa_twenty_pair_coordinator_summary_v2"
                    or summary.get("status") != "COMPLETE_20_UPDATED"
                    or summary.get("completed_pairs") != MAX_PAIRS
                    or summary.get("replayed_prefix_pairs") != REPLAY_PREFIX_COUNT
                    or summary.get("launched_pairs") != MAX_PAIRS - REPLAY_PREFIX_COUNT
                    or summary.get("replay_prefix_receipt_pin")
                       != self.manifest["replay_prefix_receipt_pin"]
                    or summary.get("pair_receipt_pins") != actual_pins):
                raise CoordinatorError("existing coordinator summary is not a valid complete record")
            return {"status": summary["status"], "completed_pairs": MAX_PAIRS,
                    "replayed_prefix_pairs": REPLAY_PREFIX_COUNT,
                    "launched_pairs": MAX_PAIRS - REPLAY_PREFIX_COUNT,
                    "replay_prefix_receipt_pin": self.manifest["replay_prefix_receipt_pin"],
                    "pair_receipt_pins": actual_pins,
                    "summary_pin": pin, "pairs": results}
        pair_receipt_pins = self._actual_pair_receipt_pins()
        summary = {"schema": "fit40_spsa_twenty_pair_coordinator_summary_v2",
                   "status": "COMPLETE_20_UPDATED", "coordinator_id": self.manifest["coordinator_id"],
                   "completed_pairs": MAX_PAIRS,
                   "replayed_prefix_pairs": REPLAY_PREFIX_COUNT,
                   "launched_pairs": MAX_PAIRS - REPLAY_PREFIX_COUNT,
                   "replay_prefix_receipt_pin": self.manifest["replay_prefix_receipt_pin"],
                   "pair_receipt_pins": pair_receipt_pins,
                   "completed_utc": _utc_now()}
        summary_pin = _write_json_exclusive(self.root / "complete_summary.json", summary,
                                            "summary_sha256")
        return {"status": "COMPLETE_20_UPDATED", "completed_pairs": MAX_PAIRS,
                "replayed_prefix_pairs": REPLAY_PREFIX_COUNT,
                "launched_pairs": MAX_PAIRS - REPLAY_PREFIX_COUNT,
                "replay_prefix_receipt_pin": self.manifest["replay_prefix_receipt_pin"],
                "pair_receipt_pins": pair_receipt_pins,
                "summary_pin": summary_pin, "pairs": results}


def _start_detached(args: argparse.Namespace) -> int:
    coordinator = TwentyPairCoordinator.create(
        coordinator_dir=args.coordinator_dir, session_dir=args.session_dir,
        repo_root=args.repo_root, python_executable=args.python,
        dispatch_root=args.dispatch_root,
        replay_prefix_receipt_path=args.replay_prefix_receipt)
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
    start.add_argument("--replay-prefix-receipt", type=Path, required=True,
                       help="self-pinned actual one-pair replay receipt; not a synthetic child process receipt")
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
