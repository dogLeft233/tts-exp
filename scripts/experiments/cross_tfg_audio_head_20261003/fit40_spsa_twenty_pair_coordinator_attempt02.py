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
BOOTSTRAP_REL = "scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_bootstrap_cli_attempt02.py"
SESSION_REL = "scripts/experiments/cross_tfg_audio_head_20261003/fit40_durable_spsa_session_attempt03.py"
BOOTSTRAP_PIN = (27_865, "a52a5866f9bcbed39c26392b0fde44619f5506cdf213170b43d949026af63025")
SESSION_PIN = (89_950, "64af31bf2fcdbd5324cc6f11611ec9c90f5c81413fdc7aeed978445d143f77fd")
COORDINATOR_SCHEMA = "fit40_spsa_twenty_pair_coordinator_v1"
START_INTENT_SCHEMA = "fit40_spsa_coordinator_start_intent_v1"
START_RECEIPT_SCHEMA = "fit40_spsa_coordinator_start_receipt_v1"
PAIR_INTENT_SCHEMA = "fit40_spsa_pair_launch_intent_v1"
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
               repo_root: str | Path, python_executable: str | Path) -> "TwentyPairCoordinator":
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
            "session_id": session_manifest.get("session_id"),
            "session_manifest_pin": session_manifest_pin,
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
        return repo, session, executable

    def _pair_dir(self, index: int) -> Path:
        return self.root / "pairs" / ("pair_%02d" % index)

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
                "--coordinator-dir", str(self.root)]
        source_pin = _file_pin(Path(__file__).resolve(strict=True))
        manifest_pin = _self_file_pin(self.root / "coordinator_manifest.json",
                                      "coordinator manifest", "coordinator_sha256")
        intent = _seal({"schema": START_INTENT_SCHEMA, "coordinator_id": self.manifest["coordinator_id"],
                        "created_utc": _utc_now(), "argv": argv,
                        "coordinator_source_pin": source_pin,
                        "coordinator_manifest_pin": manifest_pin,
                        "python_runtime": runtime}, "intent_sha256")
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
        dispatch = directory / "dispatch_root"
        dispatch.mkdir(mode=0o700)
        _fsync_dir(directory)
        packet = directory / "packet"
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
                  "argv": argv, "packet_dir": str(packet), "dispatch_root": str(dispatch),
                  "created_utc": _utc_now()}
        return _seal(intent, "intent_sha256")

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
            directory = self._pair_dir(index)
            intent = self._make_pair_intent(index=index, pair=pair, session=session,
                                            repo=repo, session_dir=session_dir)
            intent_pin = _write_json_exclusive(directory / "launch_intent.json", intent,
                                               "intent_sha256")
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
        repo_root=args.repo_root, python_executable=args.python)
    receipt = coordinator.start_detached()
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0 if receipt.get("status") == "STARTED" else 2


def _run_cli(args: argparse.Namespace) -> int:
    coordinator = TwentyPairCoordinator.open(args.coordinator_dir)
    if (Path(args.repo_root).resolve(strict=True) != Path(coordinator.manifest["repo_root"])
            or Path(args.session_dir).resolve(strict=True) != Path(coordinator.manifest["session_dir"])):
        raise CoordinatorError("run arguments differ from the sealed coordinator/session roots")
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
                     "--coordinator-dir", str(coordinator.root)]
    if (intent.get("schema") != START_INTENT_SCHEMA
            or intent.get("coordinator_id") != coordinator.manifest.get("coordinator_id")
            or intent.get("coordinator_id") != start.get("coordinator_id")
            or intent.get("coordinator_manifest_pin") != manifest_pin
            or intent.get("coordinator_source_pin") != source_pin
            or intent.get("python_runtime") != dict(runtime)
            or _python_pin(Path(str(runtime.get("invoked_path", "")))) != dict(runtime)
            or intent.get("argv") != expected_argv
            or start.get("schema") != START_RECEIPT_SCHEMA
            or start.get("status") != "STARTED"
            or start.get("coordinator_id") != intent.get("coordinator_id")
            or start.get("intent_pin") != intent_pin
            or start.get("coordinator_manifest_pin") != manifest_pin
            or start.get("coordinator_source_pin") != source_pin
            or start.get("python_runtime") != dict(runtime)
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
    start.add_argument("--python", type=Path, default=Path(sys.executable))
    run = sub.add_parser("run", help="continue only the parent-captured detached coordinator process")
    run.add_argument("--repo-root", type=Path, required=True)
    run.add_argument("--session-dir", type=Path, required=True)
    run.add_argument("--coordinator-dir", type=Path, required=True)
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
