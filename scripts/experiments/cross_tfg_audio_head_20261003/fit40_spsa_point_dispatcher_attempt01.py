#!/usr/bin/env python3
"""Parent-side launcher for one durable, three-model Fit SPSA point.

This module owns source-pinned launch profiles, the immutable execution-package
manifest, and serial native model-block dispatch.  The durable SPSA session
owns every ask/tell and calls this module through ``readback``/``execute``.
There is no controller initialization, selection, Dev/Confirm access, or retry
in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence


PROFILE_SCHEMA = "fit40_spsa_model_launch_profile_v1"
PACKAGE_SCHEMA = "fit40_spsa_point_execution_package_manifest_v1"
REQUEST_SCHEMA = "fit40_spsa_model_block_request_v1"
DISPATCH_INTENT_SCHEMA = "fit40_spsa_point_dispatch_intent_v1"
MODEL_INTENT_SCHEMA = "fit40_spsa_model_block_launch_intent_v1"
PROCESS_START_SCHEMA = "fit40_spsa_model_process_started_v1"
PROCESS_END_SCHEMA = "fit40_spsa_model_process_terminal_v1"
POINT_TERMINAL_SCHEMA = "fit40_spsa_point_dispatch_terminal_v1"
MODEL_ORDER = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
PROFILE_REQUIRED = {
    "schema", "status", "model_id", "interpreter", "runtime_binding_pin",
    "project_root", "python_path_entries", "environment", "adapter_args",
    "timeouts", "profile_sha256",
}
PACKAGE_ENTRY_REQUIRED = {
    "launcher_pin", "runner_pin", "adapter_pin", "runtime_binding_pin", "source_pins",
}
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_LOG_BYTES = 64 * 1024 * 1024


class DispatcherError(RuntimeError):
    """Invalid immutable source, point, process, or terminal evidence."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _self_sha(value: Mapping[str, Any], field: str) -> str:
    return _sha(_canonical({key: item for key, item in value.items() if key != field}))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _safe_file(path_value: Any, label: str, *, max_bytes: int = MAX_JSON_BYTES) -> tuple[Path, bytes]:
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise DispatcherError(label + " path must be absolute")
    path = Path(path_value)
    try:
        info = os.lstat(str(path))
    except OSError as exc:
        raise DispatcherError(label + " is absent") from exc
    if not stat.S_ISREG(info.st_mode) or info.st_size < 0 or info.st_size > max_bytes:
        raise DispatcherError(label + " is not a bounded regular file")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise DispatcherError(label + " changed size during read")
    return path.resolve(strict=True), raw


def file_pin(path: str | Path, *, self_field: str | None = None) -> dict[str, Any]:
    target = Path(path)
    if not target.is_absolute():
        raise DispatcherError("source pin path must be absolute")
    resolved, raw = _safe_file(str(target), "pinned file", max_bytes=2**63 - 1)
    result: dict[str, Any] = {"path": str(resolved), "bytes": len(raw), "sha256": _sha(raw)}
    if self_field is not None:
        value = _json_object(raw, str(target))
        value_self = value.get(self_field)
        if not isinstance(value_self, str) or value_self != _self_sha(value, self_field):
            raise DispatcherError("self-hash differs for " + str(target))
        result["self_sha256"] = value_self
    return result


def _read_pin(pin: Mapping[str, Any], label: str, *, self_field: str | None = None,
              exact_self_pin: bool = False, max_bytes: int = MAX_JSON_BYTES) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    if not isinstance(pin, Mapping):
        raise DispatcherError(label + " pin is not an object")
    required = {"path", "bytes", "sha256"}
    if exact_self_pin:
        required.add("self_sha256")
    if set(pin) != required:
        raise DispatcherError(label + " pin fields differ from its declared contract")
    path, raw = _safe_file(pin.get("path"), label, max_bytes=max_bytes)
    if type(pin.get("bytes")) is not int or pin["bytes"] != len(raw) or pin.get("sha256") != _sha(raw):
        raise DispatcherError(label + " raw size/SHA differs from its pin")
    value: dict[str, Any] = {}
    actual = {"path": str(path), "bytes": len(raw), "sha256": _sha(raw)}
    if self_field:
        value = _json_object(raw, label)
        embedded = value.get(self_field)
        if not isinstance(embedded, str) or embedded != _self_sha(value, self_field):
            raise DispatcherError(label + " embedded canonical self hash differs")
        if pin.get("self_sha256") is not None and pin["self_sha256"] != embedded:
            raise DispatcherError(label + " declared canonical self hash differs")
        if exact_self_pin and pin["self_sha256"] != embedded:
            raise DispatcherError(label + " self hash differs from the exact pin")
        actual["self_sha256"] = embedded
    return actual, raw, value


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DispatcherError(label + " is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise DispatcherError(label + " JSON root must be an object")
    return value


def _write_exclusive(path: str | Path, raw: bytes) -> dict[str, Any]:
    target = Path(path)
    if not target.is_absolute() or target.exists() or target.is_symlink():
        raise DispatcherError("refusing non-absolute or pre-existing output: " + str(target))
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise DispatcherError("output parent must be an existing real directory: " + str(target.parent))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(str(target), flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)
    try:
        dfd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass
    return file_pin(target)


def _write_self_json(path: str | Path, body: Mapping[str, Any], field: str) -> dict[str, Any]:
    packet = {key: value for key, value in body.items() if key != field}
    packet[field] = _self_sha(packet, field)
    raw = (json.dumps(packet, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + "\n").encode("utf-8")
    return {**_write_exclusive(path, raw), "self_sha256": packet[field]}


def _raw_pin(pin: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(pin, Mapping) or not {"path", "bytes", "sha256"}.issubset(pin):
        raise DispatcherError(label + " raw file pin is malformed")
    return {key: pin[key] for key in ("path", "bytes", "sha256")}


def seal_launch_profile(
    path: str | Path, *, model_id: str, interpreter: Mapping[str, Any],
    runtime_binding_pin: Mapping[str, Any], project_root: str | Path,
    python_path_entries: Sequence[str | Path], environment: Mapping[str, str],
    adapter_args: Mapping[str, Any], timeout_seconds: int,
    terminate_grace_seconds: float = 10.0,
) -> dict[str, Any]:
    """Create one immutable runtime profile from explicit, measured values.

    This helper refuses to derive Python identity or environment settings. The
    caller supplies the exact remote probe output and clean child environment.
    """
    if model_id not in MODEL_ORDER:
        raise DispatcherError("profile model is outside the frozen three-model order")
    expected_interpreter_keys = {"invocation_path", "resolved_path", "prefix", "python_version"}
    if not isinstance(interpreter, Mapping) or set(interpreter) != expected_interpreter_keys:
        raise DispatcherError("interpreter identity must contain the four exact measured fields")
    for key in ("invocation_path", "resolved_path", "prefix"):
        value = interpreter.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise DispatcherError("interpreter identity contains a non-absolute " + key)
    version = interpreter.get("python_version")
    if not isinstance(version, str) or not version.count(".") == 2:
        raise DispatcherError("interpreter identity needs a measured major.minor.micro version")
    # The invocation path may intentionally be a venv symlink. Its resolved
    # identity is pinned separately and checked again inside the child.
    if os.path.realpath(str(interpreter["invocation_path"])) != str(interpreter["resolved_path"]):
        raise DispatcherError("interpreter invocation does not resolve to its declared binary")
    if not Path(str(interpreter["resolved_path"])).is_file() or Path(str(interpreter["resolved_path"])).is_symlink():
        raise DispatcherError("resolved interpreter must be an existing regular file")
    if not Path(str(interpreter["prefix"])).is_dir():
        raise DispatcherError("interpreter prefix is absent")
    runtime_actual, _runtime_raw, runtime = _read_pin(
        runtime_binding_pin, "runtime binding", self_field="binding_sha256", max_bytes=4 * 1024 * 1024)
    runtime_raw_pin = _raw_pin(runtime_actual, "runtime binding")
    if not isinstance(runtime.get("binding_sha256"), str):
        raise DispatcherError("runtime binding lacks the canonical binding hash")
    project = Path(project_root)
    if not project.is_absolute() or project.is_symlink() or not project.is_dir():
        raise DispatcherError("model project root must be an existing absolute real directory")
    roots = []
    for item in python_path_entries:
        root = Path(item)
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise DispatcherError("Python import roots must be existing absolute real directories")
        roots.append(str(root.resolve(strict=True)))
    if not roots or len(roots) != len(set(roots)):
        raise DispatcherError("profile import roots must be nonempty and unique")
    if (not isinstance(environment, Mapping) or not environment
            or any(not isinstance(k, str) or not k or not isinstance(v, str) or "\x00" in v
                   for k, v in environment.items())):
        raise DispatcherError("profile environment must be a nonempty explicit string mapping")
    if not isinstance(adapter_args, Mapping):
        raise DispatcherError("profile adapter arguments must be an object")
    if type(timeout_seconds) is not int or not 60 <= timeout_seconds <= 24 * 60 * 60:
        raise DispatcherError("model block timeout must be between 60 seconds and one day")
    if not isinstance(terminate_grace_seconds, (int, float)) or not 0 < terminate_grace_seconds <= 60:
        raise DispatcherError("termination grace must be a positive value no greater than 60 seconds")
    body = {
        "schema": PROFILE_SCHEMA, "status": "FROZEN", "model_id": model_id,
        "interpreter": dict(interpreter), "runtime_binding_pin": runtime_raw_pin,
        "project_root": str(project.resolve(strict=True)), "python_path_entries": roots,
        "environment": dict(environment), "adapter_args": dict(adapter_args),
        "timeouts": {"child_seconds": timeout_seconds,
                     "terminate_grace_seconds": float(terminate_grace_seconds)},
    }
    return _write_self_json(path, body, "profile_sha256")


def seal_execution_package_manifest(
    path: str | Path, *, package_id: str, family_plan_pin: Mapping[str, Any],
    point_runner_pin: Mapping[str, Any], shared_application_helper_pin: Mapping[str, Any],
    execution_sources_by_model: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Write a create-only package whose inputs and interpreter profiles are all pinned."""
    family_actual, _family_raw, family = _read_pin(
        family_plan_pin, "composed family plan", self_field="plan_sha256", exact_self_pin=True)
    if family.get("schema_version") != "cross-tfg-audio-eval-v1" or family.get("status") != "FROZEN":
        raise DispatcherError("execution package family plan is not a frozen composed plan")
    if not isinstance(package_id, str) or not package_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in package_id):
        raise DispatcherError("package_id must be a path-safe nonempty token")
    runner_actual, _runner_raw, _runner = _read_pin(point_runner_pin, "point execution runner")
    helper_actual, _helper_raw, _helper = _read_pin(
        shared_application_helper_pin, "shared waveform helper")
    if set(execution_sources_by_model) != set(MODEL_ORDER):
        raise DispatcherError("execution package must contain the exact three-model order")
    normalized: dict[str, dict[str, Any]] = {}
    for model_id in MODEL_ORDER:
        entry = execution_sources_by_model[model_id]
        if not isinstance(entry, Mapping) or set(entry) != PACKAGE_ENTRY_REQUIRED:
            raise DispatcherError("model package entry has unexpected or missing fields: " + model_id)
        roles: dict[str, dict[str, Any]] = {}
        for role in ("launcher_pin", "runner_pin", "adapter_pin", "runtime_binding_pin"):
            pin = _raw_pin(entry[role], model_id + " " + role)
            actual, _raw, _value = _read_pin(pin, model_id + " " + role)
            roles[role] = actual
        source_pins = entry.get("source_pins")
        if not isinstance(source_pins, Mapping) or not source_pins or "launch_profile" not in source_pins:
            raise DispatcherError("model package must pin a frozen launch profile and source closure: " + model_id)
        checked_sources: dict[str, dict[str, Any]] = {}
        for role, raw_pin in source_pins.items():
            if not isinstance(role, str) or not role:
                raise DispatcherError("source closure role is malformed")
            pin = _raw_pin(raw_pin, model_id + " source " + role)
            actual, _raw, value = _read_pin(
                pin, model_id + " source " + role,
                self_field="profile_sha256" if role == "launch_profile" else None)
            if role == "launch_profile":
                if (value.get("schema") != PROFILE_SCHEMA or value.get("status") != "FROZEN"
                        or value.get("model_id") != model_id
                        or value.get("runtime_binding_pin") != roles["runtime_binding_pin"]):
                    raise DispatcherError("launch profile differs from its model/runtime package: " + model_id)
            checked_sources[role] = {key: actual[key] for key in ("path", "bytes", "sha256")}
        runtime_ref = family.get("execution", {}).get("runtime_bindings_by_model", {}).get(model_id)
        runtime_pin = roles["runtime_binding_pin"]
        if (not isinstance(runtime_ref, Mapping)
                or runtime_ref.get("path") != runtime_pin["path"]
                or runtime_ref.get("bytes") != runtime_pin["bytes"]
                or runtime_ref.get("sha256", runtime_ref.get("raw_sha256")) != runtime_pin["sha256"]):
            raise DispatcherError("model runtime pin differs from the actual composed family binding: " + model_id)
        normalized[model_id] = {
            **roles,
            "source_pins": checked_sources,
        }
    body = {
        "schema": PACKAGE_SCHEMA, "status": "FROZEN", "package_id": package_id,
        "composed_family_plan_pin": family_actual,
        "point_runner_pin": {key: runner_actual[key] for key in ("path", "bytes", "sha256")},
        "shared_application_helper_pin": {key: helper_actual[key] for key in ("path", "bytes", "sha256")},
        "execution_sources_by_model": normalized,
    }
    return _write_self_json(path, body, "manifest_sha256")


def _materialize_environment(profile: Mapping[str, Any], *, request: Mapping[str, Any],
                              family_plan: Mapping[str, Any], runtime_pin: Mapping[str, Any]) -> dict[str, str]:
    source_exec = family_plan.get("execution", {}).get("source_model_execution", {})
    model_id = profile.get("model_id")
    model_exec = source_exec.get(model_id) if isinstance(source_exec, Mapping) else None
    if not isinstance(model_exec, Mapping) or not isinstance(model_exec.get("capacity_stage"), str):
        raise DispatcherError("family plan does not expose the model's actual capacity stage")
    substitutions = {
        "strict_receipt_dir": str(request.get("strict_receipt_dir", "")),
        "runtime_binding_path": str(runtime_pin["path"]),
        "runtime_binding_sha256": str(runtime_pin["sha256"]),
        "capacity_stage": str(model_exec["capacity_stage"]),
    }
    environment = profile.get("environment")
    if not isinstance(environment, Mapping) or not environment:
        raise DispatcherError("launch profile has no exact child environment")
    output: dict[str, str] = {}
    for key, value in environment.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise DispatcherError("launch environment keys and values must be strings")
        try:
            output[key] = value.format_map(substitutions)
        except (KeyError, ValueError) as exc:
            raise DispatcherError("launch environment contains an unsupported substitution") from exc
    if len(output) != len(environment) or any("\x00" in value for value in output.values()):
        raise DispatcherError("launch environment is malformed")
    return output


def _proposal_identity(proposal: Any) -> dict[str, Any]:
    required = ("proposal_id", "pair_id", "side", "iteration", "method_id", "theta",
                "theta_sha256", "artifact_sha256", "binding_sha256", "input_pins_sha256",
                "batch_id", "sample_ids", "direction", "c_k")
    if any(not hasattr(proposal, key) for key in required):
        raise DispatcherError("executor input is not the frozen SPSAProposal object")
    identity = {key: getattr(proposal, key) for key in required}
    if (identity["side"] not in ("plus", "minus") or type(identity["iteration"]) is not int
            or not isinstance(identity["proposal_id"], str) or not identity["proposal_id"]
            or not isinstance(identity["method_id"], str) or not identity["method_id"]
            or not isinstance(identity["pair_id"], str) or not identity["pair_id"]):
        raise DispatcherError("SPSA proposal identity is malformed")
    if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in identity["proposal_id"]):
        raise DispatcherError("proposal ID is not path-safe")
    return identity


def _file_entry(path: Path, role: str, *, self_field: str | None = None) -> dict[str, Any]:
    pin = file_pin(path, self_field=self_field)
    file_id = "pt-" + _sha((role + "\0" + pin["path"] + "\0" + pin["sha256"]).encode("utf-8"))[:24]
    return {"file_id": file_id, "role": role, "declared_path": pin["path"],
            "local_path": pin["path"], "bytes": pin["bytes"], "sha256": pin["sha256"]}


def _add_file(entries: dict[str, dict[str, Any]], path: Path, role: str,
              *, self_field: str | None = None) -> dict[str, Any]:
    item = _file_entry(path, role, self_field=self_field)
    prior = entries.get(item["file_id"])
    if prior is not None and prior != item:
        raise DispatcherError("file ID collision with different file metadata")
    for old in entries.values():
        if old["role"] == role and old["declared_path"] == item["declared_path"] and old != item:
            raise DispatcherError("one role/path resolves to conflicting bytes")
    entries[item["file_id"]] = item
    return item


def _load_module(pin: Mapping[str, Any], label: str, module_name: str) -> Any:
    actual, _raw, _value = _read_pin(pin, label)
    source = Path(actual["path"])
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_path = getattr(existing, "__file__", None)
        if not isinstance(existing_path, str) or Path(existing_path).resolve(strict=True) != source:
            raise DispatcherError(label + " module name is bound to another source")
        return existing
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise DispatcherError("cannot import the pinned " + label)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if Path(str(module.__file__)).resolve(strict=True) != source:
        raise DispatcherError(label + " import origin differs from its pin")
    return module


def _read_json_pin(pin: Mapping[str, Any], label: str, self_field: str) -> tuple[dict[str, Any], dict[str, Any]]:
    actual, _raw, value = _read_pin(pin, label, self_field=self_field, exact_self_pin=True)
    return actual, value


def _parse_jsonl(raw: bytes, label: str) -> list[dict[str, Any]]:
    rows = []
    for index, line in enumerate(raw.splitlines()):
        if not line:
            continue
        value = _json_object(line, f"{label} line {index + 1}")
        rows.append(value)
    return rows


def _proc_start_ticks(pid: int) -> int:
    if os.name != "posix":
        raise DispatcherError("native child start ticks require the Linux /proc process table")
    raw = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    close = raw.rfind(")")
    if close < 0:
        raise DispatcherError("child /proc stat lacks a process-name delimiter")
    fields = raw[close + 2:].split()
    if len(fields) <= 19:
        raise DispatcherError("child /proc stat omits start ticks")
    value = int(fields[19])
    if value <= 0:
        raise DispatcherError("child start ticks are not positive")
    return value


def _safe_mkdir(path: Path, *, exist_ok: bool = False) -> None:
    if path.exists() or path.is_symlink():
        if exist_ok and path.is_dir() and not path.is_symlink():
            return
        raise DispatcherError("refusing an existing output directory: " + str(path))
    path.mkdir(parents=False, exist_ok=False, mode=0o700)


@dataclass(frozen=True)
class _PlanSide:
    proposal_id: str
    side: str
    plan_pin: Mapping[str, Any]
    planned_rows_pin: Mapping[str, Any]
    packet_index_pin: Mapping[str, Any]
    plan: Mapping[str, Any]
    planned_rows: tuple[Mapping[str, Any], ...]


class PointBlockDispatcher:
    """A serial native-model executor implementing the durable session API."""

    def __init__(
        self, *, family_plan_pin: Mapping[str, Any], initial_selector_receipt_pin: Mapping[str, Any],
        execution_package_manifest_pin: Mapping[str, Any], dispatch_root: str | Path,
        point_outcome_factory: Callable[..., Any],
    ) -> None:
        self.family_pin, _family_raw, self.family = _read_json_pin(
            family_plan_pin, "composed family plan", "plan_sha256")
        self.initial_receipt_pin, _receipt_raw, self.initial_receipt = _read_json_pin(
            initial_selector_receipt_pin, "initial selector receipt", "receipt_sha256")
        self.package_pin, _package_raw, self.package = _read_json_pin(
            execution_package_manifest_pin, "execution package manifest", "manifest_sha256")
        if (self.package.get("schema") != PACKAGE_SCHEMA or self.package.get("status") != "FROZEN"
                or self.package.get("composed_family_plan_pin") != self.family_pin):
            raise DispatcherError("execution package is not frozen to the supplied family plan")
        self.model_entries = self.package.get("execution_sources_by_model")
        if not isinstance(self.model_entries, Mapping) or list(self.model_entries) != list(MODEL_ORDER):
            raise DispatcherError("execution package model order differs from the exact three-model order")
        self.point_runner_pin = self.package.get("point_runner_pin")
        if not isinstance(self.point_runner_pin, Mapping):
            raise DispatcherError("execution package omits its point runner")
        self.runner = _load_module(self.point_runner_pin, "point execution runner", "fit40_dispatch_point_runner")
        actual_helper, _helper_raw, _helper = _read_pin(
            self.package.get("shared_application_helper_pin"), "shared waveform helper")
        self.helper_pin = actual_helper
        root = Path(dispatch_root)
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise DispatcherError("dispatch root must be a pre-created absolute real directory")
        self.dispatch_root = root.resolve(strict=True)
        self.point_outcome_factory = point_outcome_factory
        self._side_plans: dict[tuple[str, str], _PlanSide] = {}

    def bind_pair_packet(self, pair: Any, packet_index_path: str | Path) -> None:
        """Bind a composer05 packet to the exact outstanding proposal pair."""
        proposals = (pair.plus, pair.minus)
        identities = [_proposal_identity(proposal) for proposal in proposals]
        if (identities[0]["pair_id"] != identities[1]["pair_id"]
                or identities[0]["iteration"] != identities[1]["iteration"]
                or identities[0]["side"] != "plus" or identities[1]["side"] != "minus"):
            raise DispatcherError("proposal pair is not ordered plus then minus")
        index_path = Path(packet_index_path)
        index_pin = file_pin(index_path, self_field="packet_sha256")
        index = _json_object(index_path.read_bytes(), "proposal packet index")
        if (index.get("schema") != "fit40_spsa_stagea_execution_packet_v1"
                or index.get("status") != "PLANS_ONLY_REQUIRES_BOUND_RUNTIME_AND_EXECUTION"
                or index.get("execution_started") is not False
                or index.get("model_calls_started") is not False
                or index.get("parent_plan") != {k: self.family_pin[k] for k in ("path", "bytes", "sha256")}):
            raise DispatcherError("composer packet is not an untouched plan-only derivative of the frozen family")
        scope = index.get("scope")
        if (not isinstance(scope, Mapping) or scope.get("pair_id") != identities[0]["pair_id"]
                or scope.get("iteration") != identities[0]["iteration"]
                or scope.get("sample_ids") != list(identities[0]["sample_ids"])
                or scope.get("batch_id") != identities[0]["batch_id"]):
            raise DispatcherError("composer packet scope differs from the actual pending pair")
        plans = index.get("plans")
        if not isinstance(plans, list) or len(plans) != 2:
            raise DispatcherError("composer packet must contain exactly plus/minus plans")
        by_side = {entry.get("label"): entry for entry in plans if isinstance(entry, Mapping)}
        if set(by_side) != {"plus", "minus"}:
            raise DispatcherError("composer packet sides differ from the pending pair")
        for proposal, identity in zip(proposals, identities):
            entry = by_side[identity["side"]]
            plan_pin_raw = entry.get("plan")
            rows_pin_raw = entry.get("planned_rows")
            plan_self = entry.get("plan_self_sha256")
            if not isinstance(plan_pin_raw, Mapping) or not isinstance(rows_pin_raw, Mapping):
                raise DispatcherError("composer packet plan or row pin is malformed")
            plan_pin = dict(plan_pin_raw)
            plan_pin["self_sha256"] = plan_self
            plan_actual, _plan_raw, plan = _read_json_pin(plan_pin, "proposal point plan", "plan_sha256")
            rows_actual, rows_raw, _ = _read_pin(rows_pin_raw, "proposal planned rows", max_bytes=64 * 1024 * 1024)
            rows = _parse_jsonl(rows_raw, "proposal planned rows")
            config_methods = plan.get("methods")
            config_method = (config_methods[0] if isinstance(config_methods, list) and len(config_methods) == 1
                             else None)
            config = config_method.get("config") if isinstance(config_method, Mapping) else None
            expected = {
                "method": "fit40_spsa_proposal", "proposal_id": identity["proposal_id"],
                "pair_id": identity["pair_id"], "side": identity["side"],
                "iteration": identity["iteration"], "method_id": identity["method_id"],
                "theta": list(identity["theta"]), "theta_sha256": identity["theta_sha256"],
                "artifact_sha256": identity["artifact_sha256"],
                "binding_sha256": identity["binding_sha256"],
                "input_pins_sha256": identity["input_pins_sha256"],
                "batch_id": identity["batch_id"], "sample_ids": list(identity["sample_ids"]),
                "direction": list(identity["direction"]), "c_k": identity["c_k"],
            }
            if (not isinstance(config, Mapping)
                    or any(config.get(key) != value for key, value in expected.items())
                    or entry.get("run_id") != plan.get("run_id")
                    or plan.get("rows") != rows
                    or len(rows) != 24
                    or len(rows) != entry.get("planned_row_count")):
                raise DispatcherError("composer plan does not preserve the actual proposal theta/24-row denominator")
            if (plan_actual["path"] != str(Path(str(plan_pin_raw["path"])).resolve(strict=True))
                    or rows_actual["path"] != str(Path(str(rows_pin_raw["path"])).resolve(strict=True))):
                raise DispatcherError("proposal plan packet paths are not canonical")
            self._side_plans[(identity["proposal_id"], identity["side"])] = _PlanSide(
                identity["proposal_id"], identity["side"], plan_actual, rows_actual,
                index_pin, plan, tuple(rows))

    def _paths(self, proposal: Any) -> tuple[Path, Path, Path]:
        identity = _proposal_identity(proposal)
        point_dir = self.dispatch_root / "points" / (identity["proposal_id"] + "_" + identity["side"])
        return point_dir, point_dir / "dispatch_intent.json", point_dir / "point_terminal.json"

    def readback(self, proposal: Any) -> Any:
        identity = _proposal_identity(proposal)
        point_dir, intent_path, terminal_path = self._paths(proposal)
        if not point_dir.exists() and not point_dir.is_symlink():
            return self.point_outcome_factory(disposition="ABSENT")
        if point_dir.is_symlink() or not point_dir.is_dir():
            return self.point_outcome_factory(disposition="UNKNOWN", reason="proposal output path is unsafe")
        if terminal_path.is_file() and not terminal_path.is_symlink():
            try:
                terminal = self._read_terminal(terminal_path, proposal)
                return self._outcome_from_terminal(terminal)
            except Exception as exc:
                return self.point_outcome_factory(disposition="UNKNOWN", reason=str(exc))
        # A full set of immutable child proofs can be reconstructed after a
        # parent crash, but any partial launch remains unknown and is never rerun.
        if not intent_path.is_file() or intent_path.is_symlink():
            return self.point_outcome_factory(disposition="UNKNOWN", reason="proposal directory exists without a valid dispatch intent")
        try:
            self._read_intent(intent_path, identity)
            side_plan = self._require_side_plan(identity)
            gathered = []
            for model_id in MODEL_ORDER:
                block_root = point_dir / "blocks" / model_id
                receipt_path = block_root / "child_execution_receipt.json"
                if not receipt_path.is_file() or receipt_path.is_symlink():
                    return self.point_outcome_factory(disposition="UNKNOWN", reason="proposal has a missing model terminal proof")
                gathered.append(self._collect_block(proposal, side_plan, model_id, block_root))
            terminal = self._assemble_terminal(proposal, side_plan, gathered)
            return self._outcome_from_terminal(terminal)
        except Exception as exc:
            return self.point_outcome_factory(disposition="UNKNOWN", reason=str(exc))

    def execute(self, proposal: Any) -> Any:
        identity = _proposal_identity(proposal)
        existing = self.readback(proposal)
        if existing.disposition != "ABSENT":
            raise DispatcherError("dispatcher refuses to rerun a present or unknown proposal")
        side_plan = self._require_side_plan(identity)
        point_dir, intent_path, terminal_path = self._paths(proposal)
        points_root = self.dispatch_root / "points"
        if not points_root.exists():
            _safe_mkdir(points_root)
        elif points_root.is_symlink() or not points_root.is_dir():
            raise DispatcherError("dispatch points root is unsafe")
        _safe_mkdir(point_dir)
        _write_self_json(intent_path, {
            "schema": DISPATCH_INTENT_SCHEMA, "status": "DISPATCH_INTENT",
            "proposal_id": identity["proposal_id"], "pair_id": identity["pair_id"],
            "iteration": identity["iteration"], "side": identity["side"],
            "point_plan_pin": dict(side_plan.plan_pin),
            "planned_rows_pin": dict(side_plan.planned_rows_pin),
            "packet_index_pin": dict(side_plan.packet_index_pin),
            "execution_package_manifest_pin": dict(self.package_pin),
            "created_utc": _utc_now(),
        }, "intent_sha256")
        gathered = []
        # Sequential launch is deliberate: each native backend owns the one V100.
        for model_id in MODEL_ORDER:
            block_root = point_dir / "blocks" / model_id
            _safe_mkdir(block_root.parent, exist_ok=True) if block_root.parent.exists() else _safe_mkdir(block_root.parent)
            _safe_mkdir(block_root)
            context = self.runner.validate_point_scope(
                family_plan_pin=self.family_pin,
                point_plan_pin=side_plan.plan_pin,
                initial_selector_receipt_pin=self.initial_receipt_pin,
                execution_package_manifest_pin=self.package_pin,
                model_id=model_id,
            )
            gathered.append(self._run_block(proposal, side_plan, model_id, block_root, context))
        terminal = self._assemble_terminal(proposal, side_plan, gathered)
        _write_self_json(terminal_path, terminal, "terminal_sha256")
        return self._outcome_from_terminal(terminal)

    def _require_side_plan(self, identity: Mapping[str, Any]) -> _PlanSide:
        side = self._side_plans.get((identity["proposal_id"], identity["side"]))
        if side is None:
            raise DispatcherError("exact composer packet must be bound before durable session execution")
        method = side.plan.get("methods", [{}])[0]
        config = method.get("config", {}) if isinstance(method, Mapping) else {}
        if (config.get("proposal_id") != identity["proposal_id"]
                or config.get("pair_id") != identity["pair_id"]
                or config.get("side") != identity["side"]
                or config.get("method_id") != identity["method_id"]):
            raise DispatcherError("bound packet no longer matches the supplied SPSA proposal")
        return side

    def _read_intent(self, path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
        actual = file_pin(path, self_field="intent_sha256")
        _actual, _raw, value = _read_json_pin(actual, "point dispatch intent", "intent_sha256")
        for key in ("proposal_id", "pair_id", "iteration", "side"):
            if value.get(key) != identity[key]:
                raise DispatcherError("dispatch intent differs from the persisted SPSA proposal")
        return value

    def _read_profile(self, model_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        entry = self.model_entries[model_id]
        profile_pin = entry["source_pins"]["launch_profile"]
        actual, _raw, profile = _read_pin(
            profile_pin, model_id + " launch profile", self_field="profile_sha256")
        if (profile.get("schema") != PROFILE_SCHEMA or profile.get("status") != "FROZEN"
                or profile.get("model_id") != model_id
                or profile.get("runtime_binding_pin") != entry["runtime_binding_pin"]):
            raise DispatcherError("launch profile identity differs from the frozen package entry")
        return profile, actual

    def _run_block(self, proposal: Any, side_plan: _PlanSide, model_id: str,
                   block_root: Path, context: Mapping[str, Any]) -> dict[str, Any]:
        entry = self.model_entries[model_id]
        profile, profile_actual = self._read_profile(model_id)
        output_root = block_root / "stagea_output"
        strict_dir = block_root / "strict_receipts"
        request_path = block_root / "block_request.json"
        request_body = {
            "schema": REQUEST_SCHEMA, "status": "READY", "model_id": model_id,
            "block_root": str(block_root.resolve(strict=True)),
            "output_root": str(output_root), "strict_receipt_dir": str(strict_dir),
            "family_plan_pin": dict(self.family_pin),
            "point_plan_pin": dict(side_plan.plan_pin),
            "initial_selector_receipt_pin": dict(self.initial_receipt_pin),
            "execution_package_manifest_pin": dict(self.package_pin),
            "point_runner_pin": dict(self.point_runner_pin),
        }
        request_pin = _write_self_json(request_path, request_body, "request_sha256")
        environment = _materialize_environment(
            profile, request=request_body, family_plan=self.family,
            runtime_pin=entry["runtime_binding_pin"])
        argv = [str(profile["interpreter"]["invocation_path"]),
                str(entry["launcher_pin"]["path"]), "--request", str(request_path)]
        intent_path = block_root / "model_launch_intent.json"
        _write_self_json(intent_path, {
            "schema": MODEL_INTENT_SCHEMA, "status": "MODEL_LAUNCH_INTENT",
            "proposal_id": getattr(proposal, "proposal_id"), "side": getattr(proposal, "side"),
            "model_id": model_id, "request_pin": request_pin,
            "launch_profile_pin": {key: profile_actual[key] for key in ("path", "bytes", "sha256")},
            "environment_sha256": _sha(_canonical(environment)),
            "argv": argv, "cwd": profile["project_root"], "created_utc": _utc_now(),
        }, "intent_sha256")
        stdout_path, stderr_path = block_root / "stdout.log", block_root / "stderr.log"
        if stdout_path.exists() or stderr_path.exists():
            raise DispatcherError("model launch log path already exists")
        started = _utc_now()
        with stdout_path.open("xb") as stdout_stream, stderr_path.open("xb") as stderr_stream:
            try:
                proc = subprocess.Popen(
                    argv, cwd=str(profile["project_root"]), env=environment,
                    stdin=subprocess.DEVNULL, stdout=stdout_stream, stderr=stderr_stream,
                    close_fds=True, start_new_session=True,
                )
            except Exception as exc:
                raise DispatcherError("native model Popen failed after a durable launch intent") from exc
            pid = int(proc.pid)
            try:
                ticks = _proc_start_ticks(pid)
            except Exception as exc:
                raise DispatcherError("child started but parent could not capture /proc start ticks") from exc
            process_started_pin = _write_self_json(block_root / "process_started.json", {
                "schema": PROCESS_START_SCHEMA, "status": "STARTED",
                "pid": pid, "start_ticks": ticks, "started_utc": started,
                "request_pin": request_pin, "model_id": model_id,
            }, "started_sha256")
            timeout = profile.get("timeouts", {}).get("child_seconds")
            grace = profile.get("timeouts", {}).get("terminate_grace_seconds")
            if type(timeout) is not int or not 60 <= timeout <= 24 * 60 * 60:
                raise DispatcherError("frozen profile child timeout is malformed")
            if not isinstance(grace, (int, float)) or not 0 < grace <= 60:
                raise DispatcherError("frozen profile termination grace is malformed")
            timed_out = False
            try:
                return_code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(pid, signal.SIGTERM)
                except OSError:
                    pass
                try:
                    return_code = proc.wait(timeout=float(grace))
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except OSError:
                        pass
                    return_code = proc.wait()
        ended = _utc_now()
        process_terminal = {
            "schema": PROCESS_END_SCHEMA, "status": "TERMINAL",
            "pid": pid, "start_ticks": ticks, "child_exit_code": int(return_code),
            "started_utc": started, "ended_utc": ended, "timed_out": timed_out,
            "process_started_pin": process_started_pin,
        }
        process_terminal_pin = _write_self_json(block_root / "process_terminal.json",
                                                process_terminal, "process_terminal_sha256")
        stdout_stream.flush()
        stderr_stream.flush()
        if timed_out:
            raise DispatcherError("model child exceeded its pinned timeout; output is preserved, outcome is unknown")
        if not output_root.is_dir() or output_root.is_symlink():
            raise DispatcherError("model child exited without its planned point output directory")
        status_path = output_root / "run_status.json"
        manifest_path = output_root / "result_manifest.json"
        ledger_path = output_root / "results.jsonl"
        if not all(path.is_file() and not path.is_symlink() for path in (status_path, manifest_path, ledger_path)):
            raise DispatcherError("model child exited without a complete terminal status/manifest/ledger")
        status_pin = file_pin(status_path, self_field="status_sha256")
        manifest_pin = file_pin(manifest_path, self_field="manifest_sha256")
        ledger_pin = file_pin(ledger_path)
        process = {
            "child_pid": pid, "child_start_ticks": ticks, "child_exit_code": int(return_code),
            "started_utc": started, "ended_utc": ended,
        }
        receipt_pin = self.runner.write_child_execution_receipt(
            receipt_path=block_root / "child_execution_receipt.json", context=context,
            process=process,
            launcher_pin=entry["launcher_pin"], adapter_pin=entry["adapter_pin"],
            runner_pin=entry["runner_pin"], runtime_binding_pin=entry["runtime_binding_pin"],
            source_pins=entry["source_pins"],
            execution_package_manifest_pin=self.package_pin,
            run_status_pin=status_pin, result_manifest_pin=manifest_pin,
            results_jsonl_pin=ledger_pin,
        )
        return self._collect_block(proposal, side_plan, model_id, block_root,
                                   extra_pins={"request": request_pin,
                                               "process_started": process_started_pin,
                                               "process_terminal": process_terminal_pin,
                                               "child_receipt": receipt_pin})

    def _collect_block(self, proposal: Any, side_plan: _PlanSide, model_id: str,
                       block_root: Path, *, extra_pins: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
        entry = self.model_entries[model_id]
        output = block_root / "stagea_output"
        status_path, manifest_path = output / "run_status.json", output / "result_manifest.json"
        ledger_path = output / "results.jsonl"
        receipt_path = block_root / "child_execution_receipt.json"
        status_actual, _status_raw, status = _read_json_pin(
            file_pin(status_path, self_field="status_sha256"), "child point status", "status_sha256")
        manifest_actual, _manifest_raw, manifest = _read_json_pin(
            file_pin(manifest_path, self_field="manifest_sha256"), "child point manifest", "manifest_sha256")
        ledger_pin = file_pin(ledger_path)
        ledger_raw = Path(ledger_pin["path"]).read_bytes()
        records = _parse_jsonl(ledger_raw, "child result ledger")
        child_actual, _child_raw, child = _read_json_pin(
            file_pin(receipt_path, self_field="receipt_sha256"), "parent-captured child receipt", "receipt_sha256")
        process_terminal_path = block_root / "process_terminal.json"
        process_terminal_actual, _process_raw, process_terminal = _read_json_pin(
            file_pin(process_terminal_path, self_field="process_terminal_sha256"),
            "parent process terminal", "process_terminal_sha256")
        if process_terminal.get("timed_out") is not False:
            raise DispatcherError("timed-out model child cannot be marked terminal")
        planned_model_rows = [row for row in side_plan.planned_rows if row.get("model_id") == model_id]
        row_ids = [row.get("row_id") for row in planned_model_rows]
        if len(row_ids) != 8 or [row.get("row_id") for row in records] != row_ids:
            raise DispatcherError("model ledger does not contain the exact ordered eight proposal rows")
        if [row.get("row_id") for row in records] != status.get("row_ids"):
            raise DispatcherError("child terminal status row IDs differ from its ledger")
        expected_identity = {
            "model_id": model_id, "block_index": 0,
            "run_id": side_plan.plan["run_id"], "point_kind": "spsa_proposal",
            "point_id": getattr(proposal, "method_id"), "method_id": getattr(proposal, "method_id"),
            "proposal_id": getattr(proposal, "proposal_id"), "pair_id": getattr(proposal, "pair_id"),
            "iteration": getattr(proposal, "iteration"), "side": getattr(proposal, "side"),
            "family_plan_pin": self.family_pin,
            "point_plan_pin": side_plan.plan_pin,
            "execution_package_manifest_pin": self.package_pin,
            "row_ids": row_ids,
        }
        for document, label in ((status, "status"), (manifest, "manifest"), (child, "child receipt")):
            for key, expected in expected_identity.items():
                if document.get(key) != expected:
                    raise DispatcherError(f"child {label} {key} differs from the exact point/package")
        status_counts = {name: sum(row.get("status") == name for row in records)
                         for name in ("PASS", "FAILED", "INCOMPLETE")}
        if (status.get("status_counts") != status_counts or manifest.get("status_counts") != status_counts
                or status.get("status") != manifest.get("status")
                or child.get("child_exit_code") != process_terminal.get("child_exit_code")
                or (child.get("child_pid"), child.get("child_start_ticks"))
                   != (process_terminal.get("pid"), process_terminal.get("start_ticks"))):
            raise DispatcherError("child status/manifest/receipt/process accounting differs")
        for field, expected in (
            ("launcher_pin", entry["launcher_pin"]), ("runner_pin", entry["runner_pin"]),
            ("adapter_pin", entry["adapter_pin"]), ("runtime_binding_pin", entry["runtime_binding_pin"]),
            ("source_pins", entry["source_pins"]),
            ("point_runner_pin", self.point_runner_pin),
            ("shared_application_helper_pin", self.helper_pin),
        ):
            if child.get(field) != expected:
                raise DispatcherError("child receipt does not match package closure: " + field)
        for document in (status, manifest):
            if document.get("source_pins") != entry["source_pins"]:
                raise DispatcherError("child status/manifest source closure differs from package")
        status_link = {**status_actual}
        manifest_link = {**manifest_actual}
        ledger_link = dict(ledger_pin)
        if (status.get("result_manifest_pin") != manifest_link
                or status.get("results_jsonl_pin") != ledger_link
                or manifest.get("results_jsonl_pin") != ledger_link
                or child.get("run_status_pin") != status_link
                or child.get("result_manifest_pin") != manifest_link
                or child.get("results_jsonl_pin") != ledger_link):
            raise DispatcherError("child status/manifest/receipt raw/self file links do not agree")
        if status_counts["PASS"] + status_counts["FAILED"] + status_counts["INCOMPLETE"] != 8:
            raise DispatcherError("child status counts do not retain all eight proposal rows")
        # All dynamically claimed source files are rehashed after process exit.
        adapter_validation = status.get("adapter_validation_pins")
        if not isinstance(adapter_validation, Mapping):
            raise DispatcherError("child omitted runtime adapter source validation pins")
        for role, pin in adapter_validation.items():
            actual, _raw, _unused = _read_pin(pin, "adapter validation source " + str(role))
            if not all(actual.get(key) == pin.get(key) for key in ("path", "bytes", "sha256")):
                raise DispatcherError("adapter source changed after terminal status: " + str(role))
        entry_files: dict[str, dict[str, Any]] = {}
        _add_file(entry_files, Path(str(side_plan.packet_index_pin["path"])), "proposal_packet_index")
        _add_file(entry_files, Path(str(side_plan.plan_pin["path"])), "plan", self_field="plan_sha256")
        _add_file(entry_files, Path(str(side_plan.planned_rows_pin["path"])), "planned_rows")
        planned_model_path = output / "planned_rows.jsonl"
        model_rows_entry = _add_file(entry_files, planned_model_path, "model_planned_rows")
        status_entry = _add_file(entry_files, status_path, "run_status", self_field="status_sha256")
        manifest_entry = _add_file(entry_files, manifest_path, "result_manifest", self_field="manifest_sha256")
        ledger_entry = _add_file(entry_files, ledger_path, "results_jsonl")
        child_entry = _add_file(entry_files, receipt_path, "child_execution_receipt", self_field="receipt_sha256")
        for row in records:
            if row.get("status") != "PASS":
                continue
            npz_ref = row.get("score", {}).get("raw_matrix_npz")
            if not isinstance(npz_ref, Mapping):
                raise DispatcherError("passing point row omits its raw official score NPZ")
            npz_path = Path(str(npz_ref.get("path", "")))
            npz_pin = _add_file(entry_files, npz_path, "raw_matrix_npz")
            if npz_pin["bytes"] != npz_ref.get("bytes") or npz_pin["sha256"] != npz_ref.get("file_sha256"):
                raise DispatcherError("point NPZ source changed after its child receipt")
        for path_name, role, self_field in (
            ("block_request.json", "model_block_request", "request_sha256"),
            ("stdout.log", "model_block_stdout", None),
            ("stderr.log", "model_block_stderr", None),
            ("model_launch_intent.json", "model_block_launch_intent", "intent_sha256"),
            ("process_started.json", "model_process_started", "started_sha256"),
            ("process_terminal.json", "model_process_terminal", "process_terminal_sha256"),
        ):
            path = block_root / path_name
            if path.is_file() and not path.is_symlink():
                max_bytes = MAX_LOG_BYTES if path.suffix == ".log" else MAX_JSON_BYTES
                _safe_file(str(path), path_name, max_bytes=max_bytes)
                _add_file(entry_files, path, role, self_field=self_field)
        block = {
            "model_id": model_id, "block_index": 0, "row_ids": row_ids,
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
            "terminal_status": status["status"],
            "status_counts": status_counts,
        }
        if extra_pins:
            for name, pin in extra_pins.items():
                path = Path(pin["path"])
                role = {"request": "model_block_request", "process_started": "model_process_started",
                        "process_terminal": "model_process_terminal", "child_receipt": "child_execution_receipt"}[name]
                self_field = {"request": "request_sha256", "process_started": "started_sha256",
                              "process_terminal": "process_terminal_sha256",
                              "child_receipt": "receipt_sha256"}[name]
                _add_file(entry_files, path, role, self_field=self_field)
        return {
            "block": block, "files": entry_files,
            "records": records,
            "raw_npz_by_row": {
                str(row["row_id"]): {
                    "path": row["score"]["raw_matrix_npz"]["path"],
                    "bytes": row["score"]["raw_matrix_npz"]["bytes"],
                    "sha256": row["score"]["raw_matrix_npz"]["file_sha256"],
                    "array_refs": row["score"]["raw_matrix_npz"]["arrays"],
                } for row in records if row.get("status") == "PASS"
            },
        }

    def _assemble_terminal(self, proposal: Any, side_plan: _PlanSide,
                           gathered: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        identity = _proposal_identity(proposal)
        if (len(gathered) != 3
                or [item.get("block", {}).get("model_id") for item in gathered] != list(MODEL_ORDER)):
            raise DispatcherError("terminal proposal must contain the exact ordered three model blocks")
        merged_files: dict[str, dict[str, Any]] = {}
        all_npz: dict[str, dict[str, Any]] = {}
        for item in gathered:
            for file_id, file_entry_value in item["files"].items():
                old = merged_files.get(file_id)
                if old is not None and old != file_entry_value:
                    raise DispatcherError("proposal file table has a conflicting file ID")
                for previous in merged_files.values():
                    if (previous["role"], previous["declared_path"]) == (
                            file_entry_value["role"], file_entry_value["declared_path"]):
                        raise DispatcherError("proposal file table repeats a role/declaration pair")
                merged_files[file_id] = dict(file_entry_value)
            overlap = set(all_npz).intersection(item["raw_npz_by_row"])
            if overlap:
                raise DispatcherError("proposal repeats a candidate row across model NPZ maps")
            all_npz.update(item["raw_npz_by_row"])
        global_plan_entry = _add_file(merged_files, Path(str(side_plan.plan_pin["path"])),
                                      "plan", self_field="plan_sha256")
        global_rows_entry = _add_file(merged_files, Path(str(side_plan.planned_rows_pin["path"])), "planned_rows")
        packet_index_entry = _add_file(merged_files, Path(str(side_plan.packet_index_pin["path"])),
                                       "proposal_packet_index", self_field="packet_sha256")
        status_values = [item["block"]["terminal_status"] for item in gathered]
        terminal_status = "COMPLETE" if status_values == ["COMPLETE"] * 3 else "TERMINAL_WITH_FAILURES"
        point_entry = {
            "run_id": side_plan.plan["run_id"], "point_kind": "spsa_proposal",
            "point_id": identity["method_id"], "method_id": identity["method_id"],
            "proposal_id": identity["proposal_id"], "pair_id": identity["pair_id"],
            "iteration": identity["iteration"], "side": identity["side"],
            "plan_file_id": global_plan_entry["file_id"],
            "planned_rows_file_id": global_rows_entry["file_id"],
            "packet_index_file_id": packet_index_entry["file_id"],
            "blocks": [dict(item["block"]) for item in gathered],
        }
        return {
            "schema": POINT_TERMINAL_SCHEMA, "status": terminal_status,
            "proposal_id": identity["proposal_id"], "pair_id": identity["pair_id"],
            "iteration": identity["iteration"], "side": identity["side"],
            "point_entry": point_entry,
            "file_entries": sorted(merged_files.values(), key=lambda value: value["file_id"]),
            "raw_npz_by_row": all_npz,
            "model_statuses": status_values,
            "candidate_rows": len(all_npz), "planned_candidate_rows": len(side_plan.plan["rows"]),
        }

    def _read_terminal(self, path: Path, proposal: Any) -> dict[str, Any]:
        actual = file_pin(path, self_field="terminal_sha256")
        _actual, _raw, terminal = _read_json_pin(actual, "dispatcher point terminal", "terminal_sha256")
        identity = _proposal_identity(proposal)
        for key in ("proposal_id", "pair_id", "iteration", "side"):
            if terminal.get(key) != identity[key]:
                raise DispatcherError("terminal proposal identity differs from the durable ask")
        if terminal.get("schema") != POINT_TERMINAL_SCHEMA:
            raise DispatcherError("dispatcher terminal schema differs")
        self._verify_terminal_files(terminal)
        return terminal

    def _verify_terminal_files(self, terminal: Mapping[str, Any]) -> None:
        entries = terminal.get("file_entries")
        if not isinstance(entries, list) or not entries:
            raise DispatcherError("terminal point file table is empty")
        seen_roles: set[tuple[str, str]] = set()
        seen_ids: set[str] = set()
        for item in entries:
            if not isinstance(item, Mapping):
                raise DispatcherError("terminal file table item is malformed")
            role, declared = item.get("role"), item.get("declared_path")
            if (not isinstance(role, str) or not isinstance(declared, str)
                    or item.get("file_id") in seen_ids or (role, declared) in seen_roles
                    or item.get("local_path") != declared):
                raise DispatcherError("terminal file table contains duplicate or invalid identities")
            actual = file_pin(declared)
            if (actual["bytes"], actual["sha256"]) != (item.get("bytes"), item.get("sha256")):
                raise DispatcherError("terminal file table raw bytes changed: " + declared)
            seen_ids.add(item["file_id"])
            seen_roles.add((role, declared))
        blocks = terminal.get("point_entry", {}).get("blocks")
        if not isinstance(blocks, list) or [item.get("model_id") for item in blocks] != list(MODEL_ORDER):
            raise DispatcherError("terminal point blocks differ from the exact model order")

    def _outcome_from_terminal(self, terminal: Mapping[str, Any]) -> Any:
        return self.point_outcome_factory(
            disposition="TERMINAL", terminal_status=terminal["status"],
            point_entry=terminal["point_entry"], files=tuple(terminal["file_entries"]),
            raw_npz_by_row=terminal["raw_npz_by_row"],
            reason=None if terminal["status"] == "COMPLETE" else "one or more complete model blocks recorded terminal row failures",
        )


__all__ = [
    "DispatcherError", "PointBlockDispatcher", "seal_launch_profile",
    "seal_execution_package_manifest", "file_pin", "MODEL_ORDER",
    "PROFILE_SCHEMA", "PACKAGE_SCHEMA", "POINT_TERMINAL_SCHEMA",
]
