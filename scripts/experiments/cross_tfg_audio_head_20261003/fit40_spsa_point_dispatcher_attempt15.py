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
import copy
import errno
import hashlib
import importlib
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
SHARED_CANDIDATE_ROWS_SCHEMA = "fit40_spsa_shared_candidate_rows_v1"
PARTIAL_NATIVE_ADOPTION_SCHEMA = "fit40_spsa_partial_native_adoption_v1"
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
MODEL_SOURCE_ROLES = {
    MODEL_ORDER[0]: {"launch_profile"},
    MODEL_ORDER[1]: {"launch_profile", "musetalk_block_overlay", "cross_frontend_runner", "resource_helper"},
    MODEL_ORDER[2]: {"launch_profile", "ditto_block_overlay", "resource_helper"},
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


def _parse_utc_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise DispatcherError(label + " is not an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise DispatcherError(label + " is not a valid ISO UTC timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise DispatcherError(label + " is not UTC")
    return parsed


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
    except OSError as exc:
        # Windows does not consistently expose directory handles to os.fsync.
        # On POSIX this durability boundary is mandatory and must fail closed.
        unsupported_windows_directory_fsync = {
            errno.EACCES, errno.EPERM, errno.EINVAL, errno.EBADF,
            getattr(errno, "ENOTSUP", errno.EINVAL),
        }
        if os.name != "nt" or exc.errno not in unsupported_windows_directory_fsync:
            raise
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
        if (not isinstance(source_pins, Mapping)
                or not MODEL_SOURCE_ROLES[model_id].issubset(source_pins)):
            raise DispatcherError("model package omits a required frozen source-closure role: " + model_id)
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
                        or set(value) != PROFILE_REQUIRED
                        or value.get("model_id") != model_id
                        or value.get("runtime_binding_pin") != roles["runtime_binding_pin"]):
                    raise DispatcherError("launch profile differs from its model/runtime package: " + model_id)
                interpreter = value.get("interpreter")
                if (not isinstance(interpreter, Mapping) or set(interpreter) != {
                        "invocation_path", "resolved_path", "prefix", "python_version"}
                        or not isinstance(value.get("timeouts"), Mapping)
                        or set(value["timeouts"]) != {"child_seconds", "terminate_grace_seconds"}
                        or not isinstance(value.get("environment"), Mapping)
                        or not isinstance(value.get("adapter_args"), Mapping)
                        or not isinstance(value.get("python_path_entries"), list)):
                    raise DispatcherError("launch profile shape differs from its frozen schema: " + model_id)
            checked_sources[role] = {key: actual[key] for key in ("path", "bytes", "sha256")}
        runtime_ref = family.get("execution", {}).get("runtime_bindings_by_model", {}).get(model_id)
        runtime_pin = roles["runtime_binding_pin"]
        if (not isinstance(runtime_ref, Mapping)
                or runtime_ref.get("path") != runtime_pin["path"]
                or ("bytes" in runtime_ref and (type(runtime_ref.get("bytes")) is not int
                    or runtime_ref["bytes"] != runtime_pin["bytes"]))
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


def _file_entry(path: Path, role: str, *, self_field: str | None = None,
                pin_name: str | None = None) -> dict[str, Any]:
    pin = file_pin(path, self_field=self_field)
    file_id = "pt-" + _sha((role + "\0" + pin["path"] + "\0" + pin["sha256"]).encode("utf-8"))[:24]
    item = {"file_id": file_id, "role": role, "declared_path": pin["path"],
            "local_path": pin["path"], "bytes": pin["bytes"], "sha256": pin["sha256"]}
    if pin_name is not None:
        if role != "official_source_frame_map" or not isinstance(pin_name, str) or not pin_name:
            raise DispatcherError("only a named official source frame map may carry pin_name")
        item["pin_name"] = pin_name
    return item


def _add_file(entries: dict[str, dict[str, Any]], path: Path, role: str,
              *, self_field: str | None = None,
              pin_name: str | None = None) -> dict[str, Any]:
    item = _file_entry(path, role, self_field=self_field, pin_name=pin_name)
    prior = entries.get(item["file_id"])
    if prior is not None and prior != item:
        raise DispatcherError("file ID collision with different file metadata")
    for old in entries.values():
        if old["role"] == role and old["declared_path"] == item["declared_path"] and old != item:
            raise DispatcherError("one role/path resolves to conflicting bytes")
    entries[item["file_id"]] = item
    return item


def _add_official_source_frame_map(entries: dict[str, dict[str, Any]],
                                   row: Mapping[str, Any]) -> dict[str, str]:
    """Pin one producer-emitted decoded source-clock frame map for a PASS row."""
    row_id = row.get("row_id")
    score = row.get("score")
    primary = score.get("primary_syncnet_input") if isinstance(score, Mapping) else None
    if (not isinstance(row_id, str) or not row_id or not isinstance(primary, Mapping)
            or not isinstance(primary.get("frame_map_path"), str)
            or not isinstance(primary.get("frame_map_sha256"), str)):
        raise DispatcherError("passing row omits its official source-clock frame-map path/SHA")
    frame_path = Path(primary["frame_map_path"])
    resolved, raw = _safe_file(primary["frame_map_path"], "official source frame map")
    if not frame_path.is_absolute() or str(resolved) != primary["frame_map_path"]:
        raise DispatcherError("official source frame-map path must be absolute and canonical")
    if _sha(raw) != primary["frame_map_sha256"]:
        raise DispatcherError("official source frame map differs from its score-row raw SHA")
    try:
        frame_map = _json_object(raw, "official source frame map")
    except DispatcherError:
        raise
    if (frame_map.get("schema_version") != "cross-tfg-track-frame-map-v2"
            or frame_map.get("fps_num") != 25 or frame_map.get("fps_den") != 1
            or not isinstance(frame_map.get("frames"), list)):
        raise DispatcherError("official source frame map has an unexpected schema or source clock")
    render = row.get("render")
    render_track = render.get("selected_track_crop") if isinstance(render, Mapping) else None
    render_has_track = isinstance(render, Mapping) and "selected_track_crop" in render
    if render_has_track and (
            not isinstance(render_track, Mapping)
            or render_track.get("frame_map_path") != primary["frame_map_path"]
            or render_track.get("frame_map_sha256") != primary["frame_map_sha256"]):
        raise DispatcherError("render and score frame-map paths/hashes do not agree for a PASS row")
    entry = _add_file(entries, resolved, "official_source_frame_map", pin_name=row_id)
    return {row_id: entry["file_id"]}


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


def _read_json_pin(pin: Mapping[str, Any], label: str, self_field: str) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    """Read a pinned self-hashed JSON file and return pin, raw bytes, value.

    All consumers use the raw bytes alongside the observed pin and decoded
    packet.  Keeping that three-value contract explicit prevents callers from
    silently depending on a shorter helper result.
    """
    actual, raw, value = _read_pin(pin, label, self_field=self_field, exact_self_pin=True)
    return actual, raw, value


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


def _terminate_and_reap(proc: subprocess.Popen[Any], pid: int, grace: float = 10.0) -> None:
    """Best-effort cleanup if parent evidence capture fails after Popen."""
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait()
        except OSError:
            pass
    # A parent can exit after TERM while a grandchild remains in its new
    # session.  Kill any surviving members of that exact process group.
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass


def _wait_for_native_child(proc: subprocess.Popen[Any], pid: int,
                           timeout: float, grace: float) -> tuple[int, bool]:
    if type(timeout) not in (int, float) or timeout <= 0 or type(grace) not in (int, float) or grace <= 0:
        raise DispatcherError("native process timeout/grace is malformed")
    try:
        return int(proc.wait(timeout=timeout)), False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            try:
                proc.terminate()
            except OSError:
                pass
        try:
            return_code = int(proc.wait(timeout=float(grace)))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                try:
                    proc.kill()
                except OSError:
                    pass
            return_code = int(proc.wait())
        # Do not leave child processes alive after the timed parent exits.
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass
        return return_code, True
    except BaseException:
        _terminate_and_reap(proc, pid, float(grace))
        raise


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
        partial_native_adoption_receipt_pin: Mapping[str, Any] | None = None,
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
        if not isinstance(self.model_entries, Mapping) or set(self.model_entries) != set(MODEL_ORDER):
            raise DispatcherError("execution package model order differs from the exact three-model order")
        self.point_runner_pin = self.package.get("point_runner_pin")
        if not isinstance(self.point_runner_pin, Mapping):
            raise DispatcherError("execution package omits its point runner")
        self.runner = _load_module(self.point_runner_pin, "point execution runner", "fit40_dispatch_point_runner")
        actual_helper, _helper_raw, _helper = _read_pin(
            self.package.get("shared_application_helper_pin"), "shared waveform helper")
        self.helper_pin = actual_helper
        self.helper = _load_module(
            self.package.get("shared_application_helper_pin"),
            "shared waveform helper", "fit40_dispatch_shared_application_helper")
        self.snapshot_project_root = self._resolve_snapshot_project_root()
        root = Path(dispatch_root)
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise DispatcherError("dispatch root must be a pre-created absolute real directory")
        self.dispatch_root = root.resolve(strict=True)
        self.point_outcome_factory = point_outcome_factory
        self._side_plans: dict[tuple[str, str], _PlanSide] = {}
        self.partial_native_adoption_pin: dict[str, Any] | None = None
        self.partial_native_adoption: dict[str, Any] | None = None
        if partial_native_adoption_receipt_pin is not None:
            (self.partial_native_adoption_pin, _adoption_raw,
             self.partial_native_adoption) = _read_json_pin(
                partial_native_adoption_receipt_pin, "partial-native adoption receipt",
                "adoption_sha256")
            if (self.partial_native_adoption.get("schema") != PARTIAL_NATIVE_ADOPTION_SCHEMA
                    or self.partial_native_adoption.get("status") != "CLASSIFIED_PARTIAL_NATIVE_OUTPUT"
                    or self.partial_native_adoption.get("execution_package_manifest_pin") != self.package_pin
                    or self.partial_native_adoption.get("family_plan_pin") != self.family_pin
                    or self.partial_native_adoption.get("initial_selector_receipt_pin") != self.initial_receipt_pin):
                raise DispatcherError("partial-native adoption receipt differs from the active immutable source/package")
            adopted_identity = self.partial_native_adoption.get("proposal_identity")
            if (not isinstance(adopted_identity, Mapping)
                    or set(adopted_identity) != {
                        "proposal_id", "pair_id", "side", "iteration", "method_id", "theta",
                        "theta_sha256", "artifact_sha256", "binding_sha256", "input_pins_sha256",
                        "batch_id", "sample_ids", "direction", "c_k"}
                    or adopted_identity.get("side") != "plus"
                    or type(adopted_identity.get("iteration")) is not int):
                raise DispatcherError("partial-native adoption proposal_identity is malformed")

    def _resolve_snapshot_project_root(self) -> Path:
        """Resolve the source-plan project root from the frozen W2L family entry.

        The helper05 source pin is checked at the derived root before it is
        returned. This deliberately avoids assuming that the model repository,
        package overlay, and staged natural-audio project share one root.
        """
        execution = self.family.get("execution")
        source_plans = execution.get("source_model_plans") if isinstance(execution, Mapping) else None
        if not isinstance(source_plans, list):
            raise DispatcherError("composed family omits its source model plan roster")
        matches = [item for item in source_plans
                   if isinstance(item, Mapping) and item.get("model_id") == MODEL_ORDER[0]]
        if len(matches) != 1:
            raise DispatcherError("composed family does not identify one frozen W2L source plan")
        plan_ref = matches[0].get("plan")
        if not isinstance(plan_ref, Mapping) or not isinstance(plan_ref.get("path"), str):
            raise DispatcherError("frozen W2L source-plan entry omits its exact path")
        plan_path = Path(plan_ref["path"])
        if not plan_path.is_absolute() or plan_path.is_symlink() or not plan_path.is_file():
            raise DispatcherError("frozen W2L source plan is absent or unsafe")
        plan_actual = file_pin(plan_path, self_field="plan_sha256")
        declared_sha = plan_ref.get("raw_sha256", plan_ref.get("sha256"))
        if (declared_sha != plan_actual["sha256"]
                or ("bytes" in plan_ref and
                    (type(plan_ref.get("bytes")) is not int
                     or plan_ref.get("bytes") != plan_actual["bytes"]))):
            raise DispatcherError("frozen W2L source-plan bytes differ from their family reference")
        root = plan_path.resolve(strict=True).parent
        # Keep the exact staged plan and its declared audio source together.
        # The plan is re-read as bytes here so later namespace binding derives
        # from the actual frozen source-plan declaration, not a guessed path.
        plan_raw = plan_path.read_bytes()
        if len(plan_raw) != plan_actual["bytes"] or _sha(plan_raw) != plan_actual["sha256"]:
            raise DispatcherError("frozen W2L source plan changed while resolving its import closure")
        source_plan = _json_object(plan_raw, "frozen W2L source plan")
        input_pins = source_plan.get("scientific_input_pins")
        audio_pin = input_pins.get("shared_source_audio_io.py") if isinstance(input_pins, Mapping) else None
        if not isinstance(audio_pin, Mapping):
            raise DispatcherError("frozen W2L source plan omits its shared audio_io pin")
        audio_rel = audio_pin.get("path")
        audio_bytes = audio_pin.get("bytes")
        audio_sha = audio_pin.get("sha256")
        if (audio_rel != "scripts/audio_enhancement/audio_io.py"
                or type(audio_bytes) is not int or audio_bytes <= 0
                or not isinstance(audio_sha, str) or len(audio_sha) != 64
                or any(ch not in "0123456789abcdef" for ch in audio_sha)):
            raise DispatcherError("frozen W2L audio_io source pin has an unexpected shape")
        self.snapshot_audio_io_pin = {
            "path": str((root / audio_rel).resolve(strict=False)),
            "bytes": audio_bytes,
            "sha256": audio_sha,
        }
        self.snapshot_plan_pin = dict(plan_actual)
        helper_rel = getattr(self.helper, "HELPER05_REL", None)
        helper_size = getattr(self.helper, "HELPER05_BYTES", None)
        helper_sha = getattr(self.helper, "HELPER05_SHA256", None)
        if (helper_rel is None or type(helper_size) is not int
                or not isinstance(helper_sha, str) or len(helper_sha) != 64):
            raise DispatcherError("shared helper07 does not expose its pinned helper05 closure")
        helper05_path = root.joinpath(*Path(*helper_rel.parts).parts)
        actual = file_pin(helper05_path)
        if (actual["bytes"], actual["sha256"]) != (helper_size, helper_sha):
            raise DispatcherError("W2L source project does not contain the exact helper05 source closure")
        return root

    def ensure_snapshot_audio_import_namespace(self) -> dict[str, Any]:
        """Expose only the frozen W2L source-plan package under queue ``scripts``.

        The queue checkout's ``scripts`` is an already-loaded namespace package.
        Inserting the W2L project into ``sys.path`` cannot extend that live
        namespace, so helper07 otherwise fails to import its pinned audio_io.
        This narrowly appends the source-plan ``scripts`` directory after
        validating its marker files and the plan-pinned audio_io bytes. It does
        not copy or rewrite source files and leaves helper05's exact-origin gate
        intact.
        """
        queue_root = Path(__file__).resolve(strict=True).parents[3]
        queue_scripts_path = queue_root / "scripts"
        source_scripts_path = self.snapshot_project_root / "scripts"
        if queue_scripts_path.is_symlink() or source_scripts_path.is_symlink():
            raise DispatcherError("queue/source-plan scripts path cannot be a symlink")
        queue_scripts = queue_scripts_path.resolve(strict=True)
        source_scripts = source_scripts_path.resolve(strict=True)
        source_audio = source_scripts / "audio_enhancement"
        if (queue_scripts == source_scripts or queue_scripts.is_symlink()
                or source_scripts.is_symlink() or not source_scripts.is_dir()
                or source_audio.is_symlink() or not source_audio.is_dir()):
            raise DispatcherError("queue/source-plan scripts roots are absent or unexpectedly shared")

        # These two marker files are part of the already-read W2L source closure.
        # The queue side must remain a namespace package (no initializer is
        # introduced as a side effect of making the staged source visible).
        queue_init = queue_scripts_path / "__init__.py"
        if queue_init.exists() or queue_init.is_symlink():
            raise DispatcherError("queue scripts root is no longer the expected namespace package")
        initializer_pins = (
            (source_scripts / "__init__.py", 71,
             "71e1f3380ab35a1bae4eb41392aba4e4547a426fefdb82af50fb2a1ff61583c5"),
            (source_audio / "__init__.py", 78,
             "6fcc0728dd2373c834dc2d8d35b1a6757a2052cb0b441224ac6e210810844587"),
        )
        for path, expected_bytes, expected_sha in initializer_pins:
            actual = file_pin(path)
            if (actual["bytes"], actual["sha256"]) != (expected_bytes, expected_sha):
                raise DispatcherError("W2L source-plan package initializer differs from its measured pin")

        audio_actual = file_pin(Path(self.snapshot_audio_io_pin["path"]))
        if any(audio_actual.get(key) != self.snapshot_audio_io_pin[key]
               for key in ("path", "bytes", "sha256")):
            raise DispatcherError("W2L source-plan audio_io file differs from its exact input pin")

        scripts_module = sys.modules.get("scripts")
        if scripts_module is None:
            try:
                scripts_module = importlib.import_module("scripts")
            except Exception as exc:
                raise DispatcherError("cannot load queue scripts namespace") from exc
        if getattr(scripts_module, "__file__", None) is not None:
            raise DispatcherError("loaded scripts root is a regular package, not the queue namespace")
        package_paths = getattr(scripts_module, "__path__", None)
        module_spec = getattr(scripts_module, "__spec__", None)
        spec_paths = getattr(module_spec, "submodule_search_locations", None)
        if package_paths is None or spec_paths is None:
            raise DispatcherError("loaded scripts root is not a namespace package")

        resolved_paths: list[Path] = []
        for value in package_paths:
            path = Path(str(value))
            if path.is_symlink() or not path.is_dir():
                raise DispatcherError("loaded queue scripts namespace has an unsafe search path")
            resolved = path.resolve(strict=True)
            if resolved not in (queue_scripts, source_scripts):
                raise DispatcherError("loaded scripts namespace contains an unpinned search root")
            if resolved not in resolved_paths:
                resolved_paths.append(resolved)
        # The same dispatcher prepares plus and minus in one process. The
        # first call starts from the queue namespace; subsequent calls may see
        # exactly the extension installed by the first call. Keep this ordered
        # two-root set closed so no unrelated import root can accumulate.
        if resolved_paths not in ([queue_scripts], [queue_scripts, source_scripts]):
            raise DispatcherError("loaded scripts namespace has an unexpected source-root sequence")

        expected_audio_path = Path(self.snapshot_audio_io_pin["path"]).resolve(strict=True)
        expected_package_init = (source_audio / "__init__.py").resolve(strict=True)
        for module_name, expected_path in (
            ("scripts.audio_enhancement", expected_package_init),
            ("scripts.audio_enhancement.audio_io", expected_audio_path),
        ):
            loaded = sys.modules.get(module_name)
            if loaded is None:
                continue
            loaded_path = getattr(loaded, "__file__", None)
            if not isinstance(loaded_path, str) or Path(loaded_path).resolve(strict=True) != expected_path:
                raise DispatcherError("preloaded shared audio module has an unpinned origin: " + module_name)

        if source_scripts not in resolved_paths:
            resolved_paths.append(source_scripts)
        namespace_paths = [str(path) for path in resolved_paths]
        scripts_module.__path__ = namespace_paths
        module_spec.submodule_search_locations = namespace_paths
        importlib.invalidate_caches()
        try:
            audio_package = importlib.import_module("scripts.audio_enhancement")
            audio_module = importlib.import_module("scripts.audio_enhancement.audio_io")
        except Exception as exc:
            raise DispatcherError("cannot import the source-plan-pinned shared audio package") from exc
        if (Path(getattr(audio_package, "__file__", "")).resolve(strict=True) != expected_package_init
                or Path(getattr(audio_module, "__file__", "")).resolve(strict=True) != expected_audio_path
                or getattr(scripts_module, "audio_enhancement", None) is not audio_package
                or getattr(audio_package, "audio_io", None) is not audio_module):
            raise DispatcherError("loaded shared audio package/module origins differ from source-plan pins")
        package_actual = file_pin(expected_package_init)
        module_actual = file_pin(expected_audio_path)
        if any(module_actual.get(key) != self.snapshot_audio_io_pin[key]
               for key in ("path", "bytes", "sha256")):
            raise DispatcherError("loaded shared audio_io module changed after namespace binding")
        return {
            "queue_scripts_root": str(queue_scripts),
            "source_plan_scripts_root": str(source_scripts),
            "scripts_search_path": namespace_paths,
            "scripts_initializer_absent": True,
            "audio_package_initializer": package_actual,
            "audio_io_source": module_actual,
            "source_plan_pin": dict(self.snapshot_plan_pin),
        }

    def prepare_shared_candidate_rows(
        self, *, point_plan_pin: Mapping[str, Any], planned_rows: Sequence[Mapping[str, Any]],
        output_root: str | Path, point_kind: str, point_id: str, method_id: str,
        proposal_id: str | None = None, pair_id: str | None = None,
        iteration: int | None = None, side: str | None = None,
        source_step: int | None = None,
    ) -> dict[str, Any]:
        """Apply one shared waveform per sample and seal three candidate sidecars.

        This is also the no-render production-path preparation API for a
        CPU-only preflight. Every path below ``output_root`` is create-only;
        the function never constructs adapters or invokes a renderer/scorer.
        """
        point_actual, _point_raw, point = _read_json_pin(
            point_plan_pin, "derived shared point plan", "plan_sha256")
        if (point.get("methods") is None or len(point.get("methods", [])) != 1
                or point.get("methods", [{}])[0].get("method_id") != method_id
                or point_kind not in {"spsa_proposal", "spsa_center"}
                or point_id != method_id):
            raise DispatcherError("shared candidate preparation point identity differs from the one-method plan")
        frozen_rows = point.get("rows")
        if (not isinstance(frozen_rows, list)
                or len(frozen_rows) != len(planned_rows)
                or _canonical(frozen_rows) != _canonical(list(planned_rows))):
            raise DispatcherError("shared candidate preparation rows differ from the exact point plan")
        method = point["methods"][0]
        contexts = {}
        for model_id in MODEL_ORDER:
            contexts[model_id] = self.runner.validate_point_scope(
                family_plan_pin=self.family_pin, point_plan_pin=point_actual,
                initial_selector_receipt_pin=self.initial_receipt_pin,
                execution_package_manifest_pin=self.package_pin, model_id=model_id)
            context = contexts[model_id]
            if (context.get("point_kind") != point_kind
                    or context.get("method_id") != method_id
                    or context.get("proposal_id") != proposal_id
                    or context.get("pair_id") != pair_id
                    or context.get("iteration") != iteration
                    or context.get("side") != side
                    or context.get("source_step") != source_step):
                raise DispatcherError("candidate preparation metadata differs from the exact validated point scope")
        self.ensure_snapshot_audio_import_namespace()
        target = Path(output_root)
        if not target.is_absolute() or target.exists() or target.is_symlink():
            raise DispatcherError("shared candidate output root must be a fresh absolute path")
        _safe_mkdir(target)
        shared_root = target / "shared"
        projections_root = target / "projections"
        candidates_root = target / "candidate_rows"
        _safe_mkdir(shared_root)
        _safe_mkdir(projections_root)
        _safe_mkdir(candidates_root)

        identity_by_sample_model: dict[tuple[str, str], Mapping[str, Any]] = {}
        for row in self.family.get("rows", []):
            if isinstance(row, Mapping) and row.get("method_id") == "identity":
                key = (str(row.get("sample_id")), str(row.get("model_id")))
                if key in identity_by_sample_model:
                    raise DispatcherError("family repeats a source identity row for a model/sample")
                identity_by_sample_model[key] = row
        sample_order = list(dict.fromkeys(str(row.get("sample_id")) for row in frozen_rows))
        candidates_by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in MODEL_ORDER}
        source_rows_by_model: dict[str, list[Mapping[str, Any]]] = {
            model: [row for row in frozen_rows if row.get("model_id") == model]
            for model in MODEL_ORDER
        }
        projection_pins_by_model_row: dict[str, dict[str, Mapping[str, Any]]] = {
            model: {} for model in MODEL_ORDER
        }
        shared_refs_by_sample: dict[str, dict[str, Mapping[str, Any]]] = {}
        for sample_id in sample_order:
            parents = {model: identity_by_sample_model.get((sample_id, model)) for model in MODEL_ORDER}
            if any(not isinstance(parents[model], Mapping) for model in MODEL_ORDER):
                raise DispatcherError("shared application lacks one exact identity parent per model")
            point_rows = {model: next((row for row in frozen_rows
                                      if str(row.get("sample_id")) == sample_id
                                      and row.get("model_id") == model), None)
                          for model in MODEL_ORDER}
            if any(not isinstance(point_rows[model], Mapping) for model in MODEL_ORDER):
                raise DispatcherError("point plan lacks one exact candidate row per model/sample")
            sample_key = _sha(sample_id.encode("utf-8"))[:24]
            sample_dir = shared_root / sample_key
            _safe_mkdir(sample_dir)
            wav_path = sample_dir / "generator_drive.wav"
            origin_path = sample_dir / "origin_v05_application_receipt.json"
            shared_path = sample_dir / "shared_application_receipt.json"
            generated = self.helper.apply_snapshot_theta_shared(
                project_root=self.snapshot_project_root,
                family_plan=self.family, family_plan_pin=self.family_pin,
                point_plan=point, point_plan_pin=point_actual, method=method,
                parent_rows_by_model=parents,
                output_wav_path=wav_path, origin_receipt_path=origin_path,
                shared_receipt_path=shared_path, origin_model_id=MODEL_ORDER[0], device="cpu")
            shared_refs_by_sample[sample_id] = {
                "output_generator_drive_wav": dict(generated["output_generator_drive_wav"]),
                "origin_v05_application_receipt": dict(generated["origin_application_receipt"]),
                "shared_application_receipt": dict(generated["shared_application_receipt"]),
            }
            for model_id in MODEL_ORDER:
                projection_dir = projections_root / model_id
                if not projection_dir.exists():
                    _safe_mkdir(projection_dir)
                projection_path = projection_dir / (sample_key + ".json")
                projected = self.helper.project_shared_application_to_model_row(
                    project_root=self.snapshot_project_root,
                    family_plan=self.family, family_plan_pin=self.family_pin,
                    point_plan=point, point_plan_pin=point_actual, method=method,
                    model_id=model_id, parent_rows_by_model=parents,
                    parent_row=parents[model_id], point_row=point_rows[model_id],
                    shared_application_receipt_pin=generated["shared_application_receipt"],
                    projection_receipt_path=projection_path)
                candidate = dict(projected["candidate_row"])
                candidate_base = {key: value for key, value in candidate.items()
                                  if key not in {"spsa_binding", "generator_condition_input"}}
                if _canonical(candidate_base) != _canonical(point_rows[model_id]):
                    raise DispatcherError("helper07 candidate projection changed a frozen point-row field")
                candidates_by_model[model_id].append(candidate)
                projection_pins_by_model_row[model_id][str(point_rows[model_id]["row_id"])] = dict(
                    projected["model_projection_receipt_pin"])

        candidate_pins_by_model = {}
        candidate_file_ids = {}
        for model_id in MODEL_ORDER:
            expected_rows = source_rows_by_model[model_id]
            candidates = candidates_by_model[model_id]
            if ([row.get("row_id") for row in candidates]
                    != [row.get("row_id") for row in expected_rows]):
                raise DispatcherError("shared candidate projection changed the ordered per-model row roster")
            packet = {
                "schema": SHARED_CANDIDATE_ROWS_SCHEMA,
                "status": "PASS_SHARED_APPLICATION_PROJECTIONS_READY",
                "run_id": point.get("run_id"), "point_kind": point_kind,
                "point_id": point_id, "method_id": method_id,
                "proposal_id": proposal_id, "pair_id": pair_id,
                "iteration": iteration, "side": side, "source_step": source_step,
                "model_id": model_id,
                "family_plan_pin": dict(self.family_pin),
                "point_plan_pin": dict(point_actual),
                "execution_package_manifest_pin": dict(self.package_pin),
                "rows": candidates,
            }
            sidecar_path = candidates_root / (model_id + ".json")
            candidate_pin = _write_self_json(
                sidecar_path, packet, "candidate_rows_sha256")
            candidate_pins_by_model[model_id] = candidate_pin
            candidate_file_ids[model_id] = str(sidecar_path)
        return {
            "candidate_rows_by_model": candidates_by_model,
            "candidate_rows_pins_by_model": candidate_pins_by_model,
            "candidate_rows_paths_by_model": candidate_file_ids,
            "shared_artifacts_by_sample": shared_refs_by_sample,
            "model_projection_receipt_pins_by_model_row": projection_pins_by_model_row,
            "sample_order": sample_order,
            "output_root": str(target.resolve(strict=True)),
        }

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

    def _adoption_applies(self, identity: Mapping[str, Any]) -> bool:
        adoption = self.partial_native_adoption
        if adoption is None:
            return False
        adopted_identity = adoption.get("proposal_identity")
        return (isinstance(adopted_identity, Mapping)
                and identity.get("side") == "plus"
                and _canonical(identity) == _canonical(adopted_identity))

    def _validate_partial_native_adoption(
        self, proposal: Any, side_plan: _PlanSide,
    ) -> dict[str, Any] | None:
        """Validate and reconstruct the one immutable W2L block being adopted.

        This path is read-only. It verifies the exact old packet and model
        block, then rechecks each shared candidate projection with helper07.
        The returned W2L result is passed to terminal assembly exactly once;
        no prior UNKNOWN dispatcher is invoked or replayed.
        """
        identity = _proposal_identity(proposal)
        if not self._adoption_applies(identity):
            return None
        adoption = self.partial_native_adoption
        adoption_pin = self.partial_native_adoption_pin
        if not isinstance(adoption, Mapping) or not isinstance(adoption_pin, Mapping):
            raise DispatcherError("partial-native adoption is not bound to a self-pinned receipt")
        expected_adoption_keys = {
            "schema", "status", "proposal_identity", "family_plan_pin",
            "initial_selector_receipt_pin", "execution_package_manifest_pin",
            "packet_index_pin", "point_plan_pin", "planned_rows_pin",
            "adopted_model_id", "adopted_block_root", "adopted_block_file_pins",
            "candidate_rows_pins_by_model", "adoption_sha256",
        }
        expected_identity = {
            "proposal_id": identity["proposal_id"], "method_id": identity["method_id"],
            "pair_id": identity["pair_id"], "iteration": identity["iteration"], "side": "plus",
        }
        if (set(adoption) != expected_adoption_keys
                or _canonical(adoption.get("proposal_identity")) != _canonical(identity)
                or any(adoption.get("proposal_identity", {}).get(key) != value
                       for key, value in expected_identity.items())
                or adoption.get("family_plan_pin") != self.family_pin
                or adoption.get("initial_selector_receipt_pin") != self.initial_receipt_pin
                or adoption.get("execution_package_manifest_pin") != self.package_pin
                or adoption.get("point_plan_pin") != dict(side_plan.plan_pin)
                or adoption.get("packet_index_pin") != dict(side_plan.packet_index_pin)
                or adoption.get("planned_rows_pin") != dict(side_plan.planned_rows_pin)):
            raise DispatcherError("partial-native adoption does not bind this exact plus plan/package/ask")
        model_block = adoption.get("adopted_model_id")
        block_root_value = adoption.get("adopted_block_root")
        captured_block_pins = adoption.get("adopted_block_file_pins")
        candidate_pins = adoption.get("candidate_rows_pins_by_model")
        if (model_block != MODEL_ORDER[0] or not isinstance(block_root_value, str)
                or not Path(block_root_value).is_absolute()
                or not isinstance(captured_block_pins, Mapping)
                or set(captured_block_pins) != {
                    "block_request", "process_started", "process_terminal", "child_receipt",
                    "run_status", "result_manifest", "results_jsonl"}
                or not isinstance(candidate_pins, Mapping)
                or set(candidate_pins) != set(MODEL_ORDER)):
            raise DispatcherError("partial-native adoption block/candidate references are malformed")
        block_root = Path(block_root_value)
        if (block_root.is_symlink() or not block_root.is_dir()
                or str(block_root.resolve(strict=True)) != block_root_value):
            raise DispatcherError("adopted W2L block root is absent or traverses a symlink")
        block_paths = {
            "block_request": (block_root / "block_request.json", "request_sha256"),
            "process_started": (block_root / "process_started.json", "started_sha256"),
            "process_terminal": (block_root / "process_terminal.json", "process_terminal_sha256"),
            "child_receipt": (block_root / "child_execution_receipt.json", "receipt_sha256"),
            "run_status": (block_root / "stagea_output" / "run_status.json", "status_sha256"),
            "result_manifest": (block_root / "stagea_output" / "result_manifest.json", "manifest_sha256"),
            "results_jsonl": (block_root / "stagea_output" / "results.jsonl", None),
        }
        for name, (path, self_field) in block_paths.items():
            expected = captured_block_pins[name]
            if not isinstance(expected, Mapping):
                raise DispatcherError("adoption receipt omits the captured W2L " + name + " pin")
            actual = file_pin(path, self_field=self_field)
            if dict(expected) != actual:
                raise DispatcherError("adopted W2L " + name + " no longer matches the immutable partial block")
        if not isinstance(candidate_pins.get(MODEL_ORDER[0]), Mapping):
            raise DispatcherError("adoption receipt omits the W2L candidate-row pin")

        # Re-run the ordinary immutable block verifier against the same plan,
        # package, sidecar, child receipt, ledger, NPZs, and source frame maps.
        adopted_w2l = self._collect_block(
            proposal, side_plan, MODEL_ORDER[0], block_root)
        if (adopted_w2l["block"].get("terminal_status") != "COMPLETE"
                or adopted_w2l["block"].get("status_counts") != {"PASS": 8, "FAILED": 0, "INCOMPLETE": 0}
                or adopted_w2l["block"].get("candidate_rows_pin") != candidate_pins[MODEL_ORDER[0]]):
            raise DispatcherError("adopted W2L block is not the exact complete eight-row candidate result")
        process_terminal_path = block_root / "process_terminal.json"
        child_receipt_path = block_root / "child_execution_receipt.json"
        process_terminal_pin = file_pin(
            process_terminal_path, self_field="process_terminal_sha256")
        child_receipt_pin = file_pin(child_receipt_path, self_field="receipt_sha256")
        _process_actual, _process_raw, process_terminal = _read_json_pin(
            process_terminal_pin, "adopted W2L process terminal", "process_terminal_sha256")
        _child_actual, _child_raw, child_receipt = _read_json_pin(
            child_receipt_pin, "adopted W2L child receipt", "receipt_sha256")
        if (type(process_terminal.get("child_exit_code")) is not int
                or process_terminal.get("child_exit_code") != 0
                or type(child_receipt.get("child_exit_code")) is not int
                or child_receipt.get("child_exit_code") != 0):
            raise DispatcherError("adopted W2L block must have matching successful child exit evidence")

        # Validate all three candidate sidecars against the same point rows and
        # validate every shared application/projection receipt. This proves the
        # later Muse/Ditto blocks will consume the already-rendered shared WAVs.
        candidate_rows_by_model: dict[str, list[dict[str, Any]]] = {}
        contexts: dict[str, Mapping[str, Any]] = {}
        point_rows_by_model: dict[str, dict[str, Mapping[str, Any]]] = {}
        parent_rows_by_sample: dict[str, dict[str, Mapping[str, Any]]] = {}
        for family_row in self.family.get("rows", []):
            if isinstance(family_row, Mapping) and family_row.get("method_id") == "identity":
                sample_id = str(family_row.get("sample_id"))
                parent_rows_by_sample.setdefault(sample_id, {})[str(family_row.get("model_id"))] = family_row
        for model_id in MODEL_ORDER:
            contexts[model_id] = self.runner.validate_point_scope(
                family_plan_pin=self.family_pin, point_plan_pin=side_plan.plan_pin,
                initial_selector_receipt_pin=self.initial_receipt_pin,
                execution_package_manifest_pin=self.package_pin, model_id=model_id)
            candidates, packet = self.runner._validate_shared_candidate_rows(
                contexts[model_id], candidate_pins[model_id])
            if (packet.get("model_id") != model_id
                    or packet.get("proposal_id") != identity["proposal_id"]
                    or packet.get("pair_id") != identity["pair_id"]
                    or packet.get("side") != "plus"
                    or packet.get("iteration") != identity["iteration"]):
                raise DispatcherError("adopted candidate sidecar identity differs from the persisted plus ask")
            candidate_rows_by_model[model_id] = candidates
            point_rows_by_model[model_id] = {
                str(row["row_id"]): row for row in contexts[model_id]["planned_model_rows"]
            }
        if any(len(candidate_rows_by_model[model]) != 8 for model in MODEL_ORDER):
            raise DispatcherError("adopted candidate sidecars do not preserve the exact 24-row point denominator")

        self.ensure_snapshot_audio_import_namespace()
        shared_by_sample: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
        for model_id in MODEL_ORDER:
            for candidate in candidate_rows_by_model[model_id]:
                sample_id = str(candidate.get("sample_id"))
                row_id = str(candidate.get("row_id"))
                parents = parent_rows_by_sample.get(sample_id)
                point_row = point_rows_by_model[model_id].get(row_id)
                if (not isinstance(parents, Mapping) or set(parents) != set(MODEL_ORDER)
                        or not isinstance(point_row, Mapping)):
                    raise DispatcherError("adopted projection has no exact three-model parent/point context")
                binding = candidate.get("spsa_binding")
                shared_pin = binding.get("shared_application_receipt") if isinstance(binding, Mapping) else None
                projection_pin = binding.get("model_projection_receipt") if isinstance(binding, Mapping) else None
                if not isinstance(shared_pin, Mapping) or not isinstance(projection_pin, Mapping):
                    raise DispatcherError("adopted candidate lacks shared/projection receipt pins")
                lineage = self.helper.validate_model_projection_lineage(
                    project_root=self.snapshot_project_root,
                    family_plan=self.family, family_plan_pin=self.family_pin,
                    point_plan=side_plan.plan, point_plan_pin=side_plan.plan_pin,
                    method=side_plan.plan["methods"][0], model_id=model_id,
                    parent_rows_by_model=parents, parent_row=parents[model_id],
                    point_row=point_row, candidate_row=candidate,
                    shared_application_receipt_pin=shared_pin,
                    model_projection_receipt_pin=projection_pin)
                output = lineage.get("output_generator_drive_wav")
                if not isinstance(output, Mapping):
                    raise DispatcherError("adopted projection validator omitted the shared output WAV pin")
                prior = shared_by_sample.get(sample_id)
                current = (dict(lineage["shared_application_receipt"]), dict(output))
                if prior is not None and prior != current:
                    raise DispatcherError("adopted model projections do not share one physical waveform per sample")
                shared_by_sample[sample_id] = current
        if set(shared_by_sample) != {str(row.get("sample_id")) for row in side_plan.planned_rows}:
            raise DispatcherError("adopted shared waveform map differs from the exact eight-sample proposal")
        return {
            "w2l_block": adopted_w2l,
            "candidate_rows_by_model": candidate_rows_by_model,
            "candidate_rows_pins_by_model": {model: dict(candidate_pins[model]) for model in MODEL_ORDER},
            "shared_application_receipt_pins_by_sample": {
                sample: dict(value[0]) for sample, value in shared_by_sample.items()},
            "shared_output_wav_pins_by_sample": {
                sample: dict(value[1]) for sample, value in shared_by_sample.items()},
        }

    def readback(self, proposal: Any) -> Any:
        identity = _proposal_identity(proposal)
        point_dir, intent_path, terminal_path = self._paths(proposal)
        points_root = self.dispatch_root / "points"
        if points_root.is_symlink() or (points_root.exists() and not points_root.is_dir()):
            return self.point_outcome_factory(disposition="UNKNOWN", reason="proposal parent directory is unsafe")
        try:
            side_plan = self._require_side_plan(identity)
            adoption = self._validate_partial_native_adoption(proposal, side_plan)
        except Exception as exc:
            return self.point_outcome_factory(disposition="UNKNOWN", reason=str(exc))
        if not point_dir.exists() and not point_dir.is_symlink():
            return self.point_outcome_factory(disposition="ABSENT")
        if point_dir.is_symlink() or not point_dir.is_dir():
            return self.point_outcome_factory(disposition="UNKNOWN", reason="proposal output path is unsafe")
        if not intent_path.is_file() or intent_path.is_symlink():
            return self.point_outcome_factory(disposition="UNKNOWN", reason="proposal directory exists without a valid dispatch intent")
        try:
            self._read_intent(intent_path, identity, side_plan)
        except Exception as exc:
            return self.point_outcome_factory(disposition="UNKNOWN", reason=str(exc))
        if terminal_path.exists() or terminal_path.is_symlink():
            if not terminal_path.is_file() or terminal_path.is_symlink():
                return self.point_outcome_factory(
                    disposition="UNKNOWN", reason="proposal terminal path is unsafe")
            try:
                terminal = self._read_terminal(terminal_path, proposal)
                gathered = []
                for model_id in MODEL_ORDER:
                    if adoption is not None and model_id == MODEL_ORDER[0]:
                        gathered.append(adoption["w2l_block"])
                    else:
                        block_root = point_dir / "blocks" / model_id
                        gathered.append(self._collect_block(proposal, side_plan, model_id, block_root))
                rebuilt = self._assemble_terminal(
                    proposal, side_plan, gathered,
                    partial_native_adoption_receipt_pin=self.partial_native_adoption_pin if adoption else None)
                if {key: value for key, value in terminal.items() if key != "terminal_sha256"} != rebuilt:
                    raise DispatcherError("persisted terminal differs from immutable child proofs")
                return self._outcome_from_terminal(terminal)
            except Exception as exc:
                return self.point_outcome_factory(disposition="UNKNOWN", reason=str(exc))
        # A full set of immutable child proofs can be reconstructed after a
        # parent crash, but any partial launch remains unknown and is never rerun.
        try:
            gathered = []
            for model_id in MODEL_ORDER:
                if adoption is not None and model_id == MODEL_ORDER[0]:
                    gathered.append(adoption["w2l_block"])
                else:
                    block_root = point_dir / "blocks" / model_id
                    receipt_path = block_root / "child_execution_receipt.json"
                    if not receipt_path.is_file() or receipt_path.is_symlink():
                        return self.point_outcome_factory(disposition="UNKNOWN", reason="proposal has a missing model terminal proof")
                    gathered.append(self._collect_block(proposal, side_plan, model_id, block_root))
            terminal = self._assemble_terminal(
                proposal, side_plan, gathered,
                partial_native_adoption_receipt_pin=self.partial_native_adoption_pin if adoption else None)
            return self._outcome_from_terminal(terminal)
        except Exception as exc:
            return self.point_outcome_factory(disposition="UNKNOWN", reason=str(exc))

    def execute(self, proposal: Any) -> Any:
        identity = _proposal_identity(proposal)
        existing = self.readback(proposal)
        if existing.disposition != "ABSENT":
            raise DispatcherError("dispatcher refuses to rerun a present or unknown proposal")
        side_plan = self._require_side_plan(identity)
        adopted = self._validate_partial_native_adoption(proposal, side_plan)
        point_dir, intent_path, terminal_path = self._paths(proposal)
        points_root = self.dispatch_root / "points"
        if points_root.exists() and (points_root.is_symlink() or not points_root.is_dir()):
            raise DispatcherError("dispatch points root is unsafe")
        if point_dir.exists() or point_dir.is_symlink():
            raise DispatcherError("point output leaf appeared after the ABSENT readback")
        # Complete all source/runtime/720-scope gates before making the durable
        # intent. A malformed static profile must not poison a proposal slot.
        contexts: dict[str, Mapping[str, Any]] = {}
        for model_id in MODEL_ORDER:
            context = self.runner.validate_point_scope(
                family_plan_pin=self.family_pin,
                point_plan_pin=side_plan.plan_pin,
                initial_selector_receipt_pin=self.initial_receipt_pin,
                execution_package_manifest_pin=self.package_pin,
                model_id=model_id,
            )
            profile, _profile_pin = self._read_profile(model_id)
            entry = self.model_entries[model_id]
            request_probe = {
                "strict_receipt_dir": str(point_dir / "blocks" / model_id / "strict_receipts"),
            }
            _materialize_environment(
                profile, request=request_probe, family_plan=self.family,
                runtime_pin=entry["runtime_binding_pin"])
            contexts[model_id] = context
        if not points_root.exists():
            _safe_mkdir(points_root)
        _safe_mkdir(point_dir)
        _write_self_json(intent_path, {
            "schema": DISPATCH_INTENT_SCHEMA, "status": "DISPATCH_INTENT",
            "proposal_id": identity["proposal_id"], "pair_id": identity["pair_id"],
            "iteration": identity["iteration"], "side": identity["side"],
            "point_plan_pin": dict(side_plan.plan_pin),
            "planned_rows_pin": dict(side_plan.planned_rows_pin),
            "packet_index_pin": dict(side_plan.packet_index_pin),
            "execution_package_manifest_pin": dict(self.package_pin),
            "partial_native_adoption_receipt_pin": (
                dict(self.partial_native_adoption_pin) if adopted is not None else None),
            "created_utc": _utc_now(),
        }, "intent_sha256")
        if adopted is not None:
            prepared = adopted
        else:
            prepared = self.prepare_shared_candidate_rows(
                point_plan_pin=side_plan.plan_pin,
                planned_rows=side_plan.planned_rows,
                output_root=point_dir / "shared_candidate_application",
                point_kind="spsa_proposal", point_id=identity["method_id"],
                method_id=identity["method_id"], proposal_id=identity["proposal_id"],
                pair_id=identity["pair_id"], iteration=identity["iteration"],
                side=identity["side"], source_step=None)
        gathered = []
        # Sequential launch is deliberate: each native backend owns the one V100.
        for model_id in MODEL_ORDER:
            if adopted is not None and model_id == MODEL_ORDER[0]:
                gathered.append(adopted["w2l_block"])
                continue
            block_root = point_dir / "blocks" / model_id
            _safe_mkdir(block_root.parent, exist_ok=True) if block_root.parent.exists() else _safe_mkdir(block_root.parent)
            _safe_mkdir(block_root)
            gathered.append(self._run_block(
                proposal, side_plan, model_id, block_root, contexts[model_id],
                prepared["candidate_rows_pins_by_model"][model_id]))
        terminal = self._assemble_terminal(
            proposal, side_plan, gathered,
            partial_native_adoption_receipt_pin=self.partial_native_adoption_pin if adopted else None)
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

    def _read_intent(self, path: Path, identity: Mapping[str, Any],
                     side_plan: _PlanSide) -> dict[str, Any]:
        actual = file_pin(path, self_field="intent_sha256")
        _actual, _raw, value = _read_json_pin(actual, "point dispatch intent", "intent_sha256")
        expected_keys = {
            "schema", "status", "proposal_id", "pair_id", "iteration", "side",
            "point_plan_pin", "planned_rows_pin", "packet_index_pin",
            "execution_package_manifest_pin", "partial_native_adoption_receipt_pin",
            "created_utc", "intent_sha256",
        }
        if set(value) != expected_keys:
            raise DispatcherError("dispatch intent fields differ from the exact immutable contract")
        if value.get("schema") != DISPATCH_INTENT_SCHEMA or value.get("status") != "DISPATCH_INTENT":
            raise DispatcherError("dispatch intent schema/status differs")
        for key in ("proposal_id", "pair_id", "iteration", "side"):
            if value.get(key) != identity[key]:
                raise DispatcherError("dispatch intent differs from the persisted SPSA proposal")
        if (value.get("point_plan_pin") != dict(side_plan.plan_pin)
                or value.get("planned_rows_pin") != {
                    key: side_plan.planned_rows_pin[key] for key in ("path", "bytes", "sha256")}
                or value.get("packet_index_pin") != dict(side_plan.packet_index_pin)
                or value.get("execution_package_manifest_pin") != dict(self.package_pin)
                or value.get("partial_native_adoption_receipt_pin") != (
                    dict(self.partial_native_adoption_pin)
                    if self._adoption_applies(identity) else None)):
            raise DispatcherError("dispatch intent is bound to a different point or package")
        _parse_utc_timestamp(value.get("created_utc"), "dispatch intent created_utc")
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
        interpreter = profile.get("interpreter")
        if not isinstance(interpreter, Mapping) or set(interpreter) != {
                "invocation_path", "resolved_path", "prefix", "python_version"}:
            raise DispatcherError("launch profile interpreter identity is malformed")
        invocation = interpreter.get("invocation_path")
        resolved = interpreter.get("resolved_path")
        prefix = interpreter.get("prefix")
        if (not isinstance(invocation, str) or not Path(invocation).is_absolute()
                or not isinstance(resolved, str) or not Path(resolved).is_absolute()
                or not isinstance(prefix, str) or not Path(prefix).is_absolute()
                or not Path(invocation).exists()
                or os.path.realpath(invocation) != resolved
                or Path(resolved).is_symlink() or not Path(resolved).is_file()
                or str(Path(resolved).resolve(strict=True)) != resolved
                or not Path(prefix).is_dir()
                or str(Path(prefix).resolve(strict=True)) != prefix):
            raise DispatcherError("launch profile Python invocation/resolution/prefix no longer matches")
        project = profile.get("project_root")
        roots = profile.get("python_path_entries")
        if (not isinstance(project, str) or not Path(project).is_absolute()
                or Path(project).is_symlink() or not Path(project).is_dir()
                or str(Path(project).resolve(strict=True)) != project
                or not isinstance(roots, list) or not roots
                or any(not isinstance(root, str) or not Path(root).is_absolute()
                       or Path(root).is_symlink() or not Path(root).is_dir()
                       or str(Path(root).resolve(strict=True)) != root for root in roots)
                or len(set(roots)) != len(roots)):
            raise DispatcherError("launch profile project/import roots are absent or unsafe")
        timeouts = profile.get("timeouts")
        adapter_args = profile.get("adapter_args")
        environment = profile.get("environment")
        child_timeout = timeouts.get("child_seconds") if isinstance(timeouts, Mapping) else None
        grace = timeouts.get("terminate_grace_seconds") if isinstance(timeouts, Mapping) else None
        if (set(profile) != PROFILE_REQUIRED
                or not isinstance(adapter_args, Mapping)
                or not isinstance(environment, Mapping) or not environment
                or any(not isinstance(key, str) or not key or not isinstance(value, str)
                       or "\x00" in value for key, value in environment.items())
                or not isinstance(timeouts, Mapping)
                or set(timeouts) != {"child_seconds", "terminate_grace_seconds"}
                or type(child_timeout) is not int or not 60 <= child_timeout <= 24 * 60 * 60
                or not isinstance(grace, (int, float)) or not 0 < grace <= 60):
            raise DispatcherError("launch profile settings differ from the frozen contract")
        return profile, actual

    def _run_block(self, proposal: Any, side_plan: _PlanSide, model_id: str,
                   block_root: Path, context: Mapping[str, Any],
                   candidate_rows_pin: Mapping[str, Any]) -> dict[str, Any]:
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
            "candidate_rows_pin": dict(candidate_rows_pin),
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
                _terminate_and_reap(proc, pid)
                raise DispatcherError("child started but parent could not capture /proc start ticks") from exc
            try:
                process_started_pin = _write_self_json(block_root / "process_started.json", {
                    "schema": PROCESS_START_SCHEMA, "status": "STARTED",
                    "pid": pid, "start_ticks": ticks, "started_utc": started,
                    "request_pin": request_pin, "model_id": model_id,
                }, "started_sha256")
            except Exception as exc:
                _terminate_and_reap(proc, pid)
                raise DispatcherError("child started but its parent start receipt could not be sealed") from exc
            timeout = profile.get("timeouts", {}).get("child_seconds")
            grace = profile.get("timeouts", {}).get("terminate_grace_seconds")
            if type(timeout) is not int or not 60 <= timeout <= 24 * 60 * 60:
                raise DispatcherError("frozen profile child timeout is malformed")
            if not isinstance(grace, (int, float)) or not 0 < grace <= 60:
                raise DispatcherError("frozen profile termination grace is malformed")
            return_code, timed_out = _wait_for_native_child(proc, pid, timeout, float(grace))
        ended = _utc_now()
        process_terminal = {
            "schema": PROCESS_END_SCHEMA, "status": "TERMINAL",
            "pid": pid, "start_ticks": ticks, "child_exit_code": int(return_code),
            "started_utc": started, "ended_utc": ended, "timed_out": timed_out,
            "process_started_pin": process_started_pin,
        }
        process_terminal_pin = _write_self_json(block_root / "process_terminal.json",
                                                process_terminal, "process_terminal_sha256")
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
        process_started_path = block_root / "process_started.json"
        process_started_actual, _process_started_raw, process_started = _read_json_pin(
            file_pin(process_started_path, self_field="started_sha256"),
            "parent process start receipt", "started_sha256")
        process_terminal_path = block_root / "process_terminal.json"
        process_terminal_actual, _process_raw, process_terminal = _read_json_pin(
            file_pin(process_terminal_path, self_field="process_terminal_sha256"),
            "parent process terminal", "process_terminal_sha256")
        request_actual, _request_raw, request = _read_json_pin(
            file_pin(block_root / "block_request.json", self_field="request_sha256"),
            "model block request", "request_sha256")
        candidate_rows_pin = request.get("candidate_rows_pin")
        if not isinstance(candidate_rows_pin, Mapping):
            raise DispatcherError("model block request omits the shared candidate-row sidecar pin")
        candidate_actual, _candidate_raw, candidate_packet = _read_json_pin(
            candidate_rows_pin, "shared candidate rows", "candidate_rows_sha256")
        scope = self.runner.validate_point_scope(
            family_plan_pin=self.family_pin, point_plan_pin=side_plan.plan_pin,
            initial_selector_receipt_pin=self.initial_receipt_pin,
            execution_package_manifest_pin=self.package_pin, model_id=model_id)
        self.runner._validate_shared_candidate_rows(scope, candidate_actual)
        if (candidate_packet.get("model_id") != model_id
                or candidate_packet.get("point_id") != getattr(proposal, "method_id")
                or candidate_packet.get("proposal_id") != getattr(proposal, "proposal_id")
                or candidate_packet.get("pair_id") != getattr(proposal, "pair_id")
                or candidate_packet.get("side") != getattr(proposal, "side")
                or candidate_packet.get("iteration") != getattr(proposal, "iteration")):
            raise DispatcherError("candidate sidecar identity differs from its actual point/model block")
        if (process_started.get("schema") != PROCESS_START_SCHEMA
                or process_started.get("status") != "STARTED"
                or process_started.get("model_id") != model_id
                or type(process_started.get("pid")) is not int or process_started["pid"] <= 0
                or type(process_started.get("start_ticks")) is not int or process_started["start_ticks"] <= 0
                or process_started.get("request_pin") != request_actual):
            raise DispatcherError("parent process start evidence is malformed or detached from its request")
        process_started_time = _parse_utc_timestamp(process_started.get("started_utc"),
                                                     "parent process started_utc")
        if (process_terminal.get("schema") != PROCESS_END_SCHEMA
                or process_terminal.get("status") != "TERMINAL"
                or process_terminal.get("process_started_pin") != process_started_actual
                or process_terminal.get("pid") != process_started.get("pid")
                or process_terminal.get("start_ticks") != process_started.get("start_ticks")
                or process_terminal.get("started_utc") != process_started.get("started_utc")
                or process_terminal.get("timed_out") is not False
                or type(process_terminal.get("child_exit_code")) is not int):
            raise DispatcherError("parent process terminal does not continue its exact start receipt")
        process_ended_time = _parse_utc_timestamp(process_terminal.get("ended_utc"),
                                                  "parent process ended_utc")
        if process_ended_time < process_started_time:
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
            "initial_selector_receipt_pin": self.initial_receipt_pin,
            "initial_ingest_index_pin": self.initial_receipt.get("initial_ingest_index"),
            "execution_package_manifest_pin": self.package_pin,
            "candidate_rows_pin": candidate_actual,
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
                   != (process_terminal.get("pid"), process_terminal.get("start_ticks"))
                or child.get("started_utc") != process_started.get("started_utc")
                or child.get("ended_utc") != process_terminal.get("ended_utc")):
            raise DispatcherError("child status/manifest/receipt/process accounting differs")
        if (status.get("candidate_rows_pin") != candidate_actual
                or manifest.get("candidate_rows_pin") != candidate_actual
                or child.get("candidate_rows_pin") != candidate_actual):
            raise DispatcherError("child status/manifest/receipt do not bind the exact candidate sidecar")
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
        candidate_rows_entry = _add_file(
            entry_files, Path(candidate_actual["path"]), "shared_candidate_rows",
            self_field="candidate_rows_sha256")
        if (candidate_rows_entry["bytes"] != candidate_actual["bytes"]
                or candidate_rows_entry["sha256"] != candidate_actual["sha256"]):
            raise DispatcherError("candidate sidecar changed while being added to the terminal file table")
        shared_output_file_ids_by_sample: dict[str, str] = {}
        shared_application_file_ids_by_sample: dict[str, str] = {}
        origin_application_file_ids_by_sample: dict[str, str] = {}
        projection_file_ids_by_row: dict[str, str] = {}
        for candidate in candidate_packet.get("rows", []):
            binding = candidate.get("spsa_binding") if isinstance(candidate, Mapping) else None
            shared_pin = binding.get("shared_application_receipt") if isinstance(binding, Mapping) else None
            projection_pin = binding.get("model_projection_receipt") if isinstance(binding, Mapping) else None
            if not isinstance(shared_pin, Mapping) or not isinstance(projection_pin, Mapping):
                raise DispatcherError("candidate sidecar row lacks the two helper07 projection receipts")
            shared_actual, _shared_raw, shared_packet = _read_json_pin(
                shared_pin, "shared application receipt", "self_sha256")
            projection_actual, _projection_raw, _projection_packet = _read_json_pin(
                projection_pin, "model projection receipt", "self_sha256")
            sample_id = str(candidate.get("sample_id"))
            row_id = str(candidate.get("row_id"))
            output_ref = shared_packet.get("output_generator_drive_wav")
            origin_pin = shared_packet.get("origin_v05_application_receipt")
            if not isinstance(output_ref, Mapping) or not isinstance(origin_pin, Mapping):
                raise DispatcherError("shared receipt omits its exact output or v05 origin receipt")
            shared_entry = _add_file(entry_files, Path(shared_actual["path"]),
                                     "shared_application_receipt", self_field="self_sha256")
            origin_actual, _origin_raw, _origin_packet = _read_json_pin(
                origin_pin, "shared application v05 origin receipt", "self_sha256")
            origin_entry = _add_file(entry_files, Path(origin_actual["path"]),
                                     "shared_origin_v05_application_receipt", self_field="self_sha256")
            output_path = Path(str(output_ref.get("path", "")))
            output_entry = _add_file(entry_files, output_path, "shared_generator_drive_wav")
            projection_entry = _add_file(entry_files, Path(projection_actual["path"]),
                                          "model_projection_receipt", self_field="self_sha256")
            if (shared_entry["bytes"] != shared_actual["bytes"]
                    or shared_entry["sha256"] != shared_actual["sha256"]
                    or origin_entry["bytes"] != origin_actual["bytes"]
                    or origin_entry["sha256"] != origin_actual["sha256"]
                    or projection_entry["bytes"] != projection_actual["bytes"]
                    or projection_entry["sha256"] != projection_actual["sha256"]
                    or output_entry["bytes"] != output_ref.get("bytes")
                    or output_entry["sha256"] != output_ref.get("sha256")):
                raise DispatcherError("shared waveform/projection bytes changed before parent collection")
            for mapping, key, value, label in (
                (shared_output_file_ids_by_sample, sample_id, output_entry["file_id"], "shared output"),
                (shared_application_file_ids_by_sample, sample_id, shared_entry["file_id"], "shared receipt"),
                (origin_application_file_ids_by_sample, sample_id, origin_entry["file_id"], "origin receipt"),
                (projection_file_ids_by_row, row_id, projection_entry["file_id"], "model projection"),
            ):
                if key in mapping and mapping[key] != value:
                    raise DispatcherError("one sample/row maps to two " + label + " files")
                mapping[key] = value
        _add_file(entry_files, Path(str(self.family_pin["path"])),
                  "composed_family_plan", self_field="plan_sha256")
        _add_file(entry_files, Path(str(self.initial_receipt_pin["path"])),
                  "initial_selector_receipt", self_field="receipt_sha256")
        ingest_pin = self.initial_receipt.get("initial_ingest_index")
        if not isinstance(ingest_pin, Mapping):
            raise DispatcherError("initial selector receipt omits its verified ingest index")
        _add_file(entry_files, Path(str(ingest_pin["path"])),
                  "initial_stagea_ingest_index", self_field="index_sha256")
        _add_file(entry_files, Path(str(self.package_pin["path"])),
                  "execution_package_manifest", self_field="manifest_sha256")
        _add_file(entry_files, Path(str(self.point_runner_pin["path"])), "point_execution_runner")
        _add_file(entry_files, Path(str(self.helper_pin["path"])), "shared_application_helper")
        _add_file(entry_files, Path(str(entry["launcher_pin"]["path"])), "common_model_child_launcher")
        _add_file(entry_files, Path(str(entry["runner_pin"]["path"])), "model_stagea_runner")
        _add_file(entry_files, Path(str(entry["adapter_pin"]["path"])), "model_spsa_adapter")
        _add_file(entry_files, Path(str(entry["runtime_binding_pin"]["path"])),
                  "model_runtime_binding", self_field="binding_sha256")
        for role, source_pin in entry["source_pins"].items():
            _add_file(entry_files, Path(str(source_pin["path"])),
                      "model_source:" + model_id + ":" + str(role),
                      self_field="profile_sha256" if role == "launch_profile" else None)
        _add_file(entry_files, Path(str(side_plan.packet_index_pin["path"])), "proposal_packet_index")
        _add_file(entry_files, Path(str(side_plan.plan_pin["path"])), "plan", self_field="plan_sha256")
        _add_file(entry_files, Path(str(side_plan.planned_rows_pin["path"])), "planned_rows")
        planned_model_path = output / "planned_rows.jsonl"
        model_rows_entry = _add_file(entry_files, planned_model_path, "model_planned_rows")
        status_entry = _add_file(entry_files, status_path, "run_status", self_field="status_sha256")
        manifest_entry = _add_file(entry_files, manifest_path, "result_manifest", self_field="manifest_sha256")
        ledger_entry = _add_file(entry_files, ledger_path, "results_jsonl")
        child_entry = _add_file(entry_files, receipt_path, "child_execution_receipt", self_field="receipt_sha256")
        frame_map_file_id_by_row: dict[str, str] = {}
        raw_npz_by_row: dict[str, dict[str, Any]] = {}
        for row in records:
            if row.get("status") != "PASS":
                continue
            npz_ref = row.get("score", {}).get("raw_matrix_npz")
            if not isinstance(npz_ref, Mapping):
                raise DispatcherError("passing point row omits its raw official score NPZ")
            npz_path_value = npz_ref.get("path")
            if not isinstance(npz_path_value, str) or not npz_path_value:
                raise DispatcherError("passing point row has no exact raw official score NPZ path")
            npz_path = Path(npz_path_value)
            npz_pin = _add_file(entry_files, npz_path, "raw_matrix_npz")
            declared_bytes = npz_ref.get("bytes")
            if (npz_pin["declared_path"] != npz_path_value
                    or ("bytes" in npz_ref and (type(declared_bytes) is not int
                                                  or declared_bytes != npz_pin["bytes"]))
                    or npz_pin["sha256"] != npz_ref.get("file_sha256")):
                raise DispatcherError("point NPZ source changed after its child receipt")
            row_id = str(row.get("row_id", ""))
            if row_id in frame_map_file_id_by_row:
                raise DispatcherError("point row repeats its source frame-map binding")
            array_refs = npz_ref.get("arrays")
            if not isinstance(array_refs, Mapping) or not array_refs:
                raise DispatcherError("passing point row omits official score NPZ array references")
            raw_npz_by_row[row_id] = {
                "path": npz_pin["declared_path"],
                "bytes": npz_pin["bytes"],
                "sha256": npz_pin["sha256"],
                "array_refs": array_refs,
            }
            frame_map_file_id_by_row.update(_add_official_source_frame_map(entry_files, row))
        for role, pin in adapter_validation.items():
            _add_file(entry_files, Path(str(pin["path"])),
                      "adapter_validation:" + model_id + ":" + str(role))
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
            "official_source_frame_map_file_ids": frame_map_file_id_by_row,
            "shared_output_file_ids_by_sample": shared_output_file_ids_by_sample,
            "shared_application_file_ids_by_sample": shared_application_file_ids_by_sample,
            "origin_v05_application_file_ids_by_sample": origin_application_file_ids_by_sample,
            "model_projection_file_ids_by_row": projection_file_ids_by_row,
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
            "raw_npz_by_row": raw_npz_by_row,
        }

    def _assemble_terminal(
        self, proposal: Any, side_plan: _PlanSide,
        gathered: Sequence[Mapping[str, Any]], *,
        partial_native_adoption_receipt_pin: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        identity = _proposal_identity(proposal)
        if (len(gathered) != 3
                or [item.get("block", {}).get("model_id") for item in gathered] != list(MODEL_ORDER)):
            raise DispatcherError("terminal proposal must contain the exact ordered three model blocks")
        merged_files: dict[str, dict[str, Any]] = {}
        all_npz: dict[str, dict[str, Any]] = {}
        all_frame_maps: dict[str, str] = {}
        candidate_rows_file_ids_by_model: dict[str, str] = {}
        shared_output_file_ids_by_sample: dict[str, str] = {}
        shared_application_file_ids_by_sample: dict[str, str] = {}
        origin_v05_application_file_ids_by_sample: dict[str, str] = {}
        model_projection_file_ids_by_row: dict[str, str] = {}
        for item in gathered:
            counts = item.get("block", {}).get("status_counts")
            if not isinstance(counts, Mapping) or set(counts) != {"PASS", "FAILED", "INCOMPLETE"}:
                raise DispatcherError("model block status counts are malformed")
            expected_model_status = (
                "COMPLETE" if counts == {"PASS": 8, "FAILED": 0, "INCOMPLETE": 0}
                else "TERMINAL_WITH_FAILURES"
            )
            if item["block"].get("terminal_status") != expected_model_status:
                raise DispatcherError("model block terminal status differs from its preserved denominator")
            for file_id, file_entry_value in item["files"].items():
                old = merged_files.get(file_id)
                if old is not None and old != file_entry_value:
                    raise DispatcherError("proposal file table has a conflicting file ID")
                if old == file_entry_value:
                    continue
                for previous in merged_files.values():
                    if (previous["role"], previous["declared_path"]) == (
                            file_entry_value["role"], file_entry_value["declared_path"]):
                        raise DispatcherError("proposal file table repeats a role/declaration pair")
                merged_files[file_id] = dict(file_entry_value)
            overlap = set(all_npz).intersection(item["raw_npz_by_row"])
            if overlap:
                raise DispatcherError("proposal repeats a candidate row across model NPZ maps")
            all_npz.update(item["raw_npz_by_row"])
            frame_maps = item["block"].get("official_source_frame_map_file_ids")
            if not isinstance(frame_maps, Mapping) or set(all_frame_maps).intersection(frame_maps):
                raise DispatcherError("model blocks contain malformed or duplicate source frame-map rows")
            all_frame_maps.update({str(row_id): str(file_id) for row_id, file_id in frame_maps.items()})
            model_id = str(item["block"]["model_id"])
            candidate_file_id = item["block"].get("candidate_rows_file_id")
            if not isinstance(candidate_file_id, str) or model_id in candidate_rows_file_ids_by_model:
                raise DispatcherError("terminal block omits or duplicates its candidate sidecar file")
            candidate_rows_file_ids_by_model[model_id] = candidate_file_id
            for field, aggregate in (
                ("shared_output_file_ids_by_sample", shared_output_file_ids_by_sample),
                ("shared_application_file_ids_by_sample", shared_application_file_ids_by_sample),
                ("origin_v05_application_file_ids_by_sample", origin_v05_application_file_ids_by_sample),
            ):
                values = item["block"].get(field)
                if not isinstance(values, Mapping):
                    raise DispatcherError("terminal block omits shared waveform provenance map: " + field)
                for key, value in values.items():
                    if key in aggregate and aggregate[key] != value:
                        raise DispatcherError("model blocks disagree on one physical shared waveform receipt")
                    aggregate[str(key)] = str(value)
            projection_ids = item["block"].get("model_projection_file_ids_by_row")
            if (not isinstance(projection_ids, Mapping)
                    or set(model_projection_file_ids_by_row).intersection(projection_ids)):
                raise DispatcherError("terminal blocks contain malformed or duplicated model projection maps")
            model_projection_file_ids_by_row.update(
                {str(row_id): str(file_id) for row_id, file_id in projection_ids.items()})
        global_plan_entry = _add_file(merged_files, Path(str(side_plan.plan_pin["path"])),
                                      "plan", self_field="plan_sha256")
        global_rows_entry = _add_file(merged_files, Path(str(side_plan.planned_rows_pin["path"])), "planned_rows")
        packet_index_entry = _add_file(merged_files, Path(str(side_plan.packet_index_pin["path"])),
                                       "proposal_packet_index", self_field="packet_sha256")
        adoption_entry = None
        if partial_native_adoption_receipt_pin is not None:
            adoption_entry = _add_file(
                merged_files, Path(str(partial_native_adoption_receipt_pin["path"])),
                "partial_native_adoption_receipt", self_field="adoption_sha256")
        ingest_pin = self.initial_receipt.get("initial_ingest_index")
        if not isinstance(ingest_pin, Mapping):
            raise DispatcherError("initial selector receipt omits its verified ingest index")
        status_values = [item["block"]["terminal_status"] for item in gathered]
        aggregate_counts = {
            name: sum(item["block"]["status_counts"][name] for item in gathered)
            for name in ("PASS", "FAILED", "INCOMPLETE")
        }
        if sum(aggregate_counts.values()) != len(side_plan.plan["rows"]):
            raise DispatcherError("terminal model blocks do not preserve the complete point denominator")
        if aggregate_counts == {"PASS": len(side_plan.plan["rows"]), "FAILED": 0, "INCOMPLETE": 0}:
            terminal_status = "COMPLETE"
        elif aggregate_counts["INCOMPLETE"]:
            terminal_status = "INCOMPLETE"
        else:
            terminal_status = "COMPLETE_WITH_FAILURES"
        if len(all_npz) != aggregate_counts["PASS"]:
            raise DispatcherError("point NPZ map differs from the complete passing-row count")
        if set(all_frame_maps) != set(all_npz):
            raise DispatcherError("point frame-map denominator differs from its PASS NPZ rows")
        expected_samples = {str(row.get("sample_id")) for row in side_plan.planned_rows}
        expected_candidate_rows = {str(row.get("row_id")) for row in side_plan.planned_rows}
        if (set(candidate_rows_file_ids_by_model) != set(MODEL_ORDER)
                or set(shared_output_file_ids_by_sample) != expected_samples
                or set(shared_application_file_ids_by_sample) != expected_samples
                or set(origin_v05_application_file_ids_by_sample) != expected_samples
                or set(model_projection_file_ids_by_row) != expected_candidate_rows):
            raise DispatcherError("shared application/projection file maps do not cover the exact point denominator")
        point_entry = {
            "run_id": side_plan.plan["run_id"], "point_kind": "spsa_proposal",
            "point_id": identity["method_id"], "method_id": identity["method_id"],
            "proposal_id": identity["proposal_id"], "pair_id": identity["pair_id"],
            "iteration": identity["iteration"], "side": identity["side"],
            "plan_file_id": global_plan_entry["file_id"],
            "planned_rows_file_id": global_rows_entry["file_id"],
            "packet_index_file_id": packet_index_entry["file_id"],
            "family_plan_pin": self.family_pin,
            "initial_selector_receipt_pin": self.initial_receipt_pin,
            "initial_ingest_index_pin": ingest_pin,
            "execution_package_manifest_pin": self.package_pin,
            "point_runner_pin": self.point_runner_pin,
            "shared_application_helper_pin": self.helper_pin,
            "partial_native_adoption_receipt_pin": (
                dict(partial_native_adoption_receipt_pin)
                if partial_native_adoption_receipt_pin is not None else None),
            "adopted_model_ids": [MODEL_ORDER[0]] if adoption_entry is not None else [],
            "candidate_rows_file_ids_by_model": candidate_rows_file_ids_by_model,
            "shared_output_file_ids_by_sample": shared_output_file_ids_by_sample,
            "shared_application_file_ids_by_sample": shared_application_file_ids_by_sample,
            "origin_v05_application_file_ids_by_sample": origin_v05_application_file_ids_by_sample,
            "model_projection_file_ids_by_row": model_projection_file_ids_by_row,
            "official_source_frame_map_file_ids": all_frame_maps,
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
        point_entry = terminal.get("point_entry")
        if (not isinstance(point_entry, Mapping)
                or point_entry.get("family_plan_pin") != self.family_pin
                or point_entry.get("initial_selector_receipt_pin") != self.initial_receipt_pin
                or point_entry.get("execution_package_manifest_pin") != self.package_pin
                or point_entry.get("partial_native_adoption_receipt_pin") != (
                    dict(self.partial_native_adoption_pin)
                    if self._adoption_applies(_proposal_identity(proposal)) else None)
                or point_entry.get("adopted_model_ids") != (
                    [MODEL_ORDER[0]] if self._adoption_applies(_proposal_identity(proposal)) else [])):
            raise DispatcherError("dispatcher terminal point entry differs from the immutable family/package")
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
            base_file_keys = {"file_id", "role", "declared_path", "local_path", "bytes", "sha256"}
            expected_file_keys = (base_file_keys | {"pin_name"}
                                  if item.get("role") == "official_source_frame_map" else base_file_keys)
            if set(item) != expected_file_keys:
                raise DispatcherError("terminal file entry fields differ from the frozen file-table schema")
            role, declared = item.get("role"), item.get("declared_path")
            if (not isinstance(role, str) or not isinstance(declared, str)
                    or item.get("file_id") in seen_ids or (role, declared) in seen_roles
                    or item.get("local_path") != declared
                    or type(item.get("bytes")) is not int or item["bytes"] < 0
                    or not isinstance(item.get("sha256"), str) or len(item["sha256"]) != 64):
                raise DispatcherError("terminal file table contains duplicate or invalid identities")
            expected_file_id = "pt-" + _sha((role + "\0" + declared + "\0" + item["sha256"]).encode("utf-8"))[:24]
            if item.get("file_id") != expected_file_id:
                raise DispatcherError("terminal file ID does not match its pinned role/path/SHA")
            actual = file_pin(declared)
            if (actual["path"] != declared
                    or (actual["bytes"], actual["sha256"]) != (item.get("bytes"), item.get("sha256"))):
                raise DispatcherError("terminal file table raw bytes changed: " + declared)
            if role == "official_source_frame_map" and (
                    not isinstance(item.get("pin_name"), str) or not item["pin_name"]):
                raise DispatcherError("official source frame-map entry omits its row_id pin_name")
            seen_ids.add(item["file_id"])
            seen_roles.add((role, declared))
        blocks = terminal.get("point_entry", {}).get("blocks")
        if not isinstance(blocks, list) or [item.get("model_id") for item in blocks] != list(MODEL_ORDER):
            raise DispatcherError("terminal point blocks differ from the exact model order")
        point_entry = terminal.get("point_entry", {})
        adoption_pin = point_entry.get("partial_native_adoption_receipt_pin")
        adopted_models = point_entry.get("adopted_model_ids")
        adoption_file_entries = [item for item in entries
                                 if item.get("role") == "partial_native_adoption_receipt"]
        if adoption_pin is None:
            if adopted_models != [] or adoption_file_entries:
                raise DispatcherError("terminal claims partial-native adoption without its receipt")
        else:
            if (not isinstance(adoption_pin, Mapping)
                    or adopted_models != [MODEL_ORDER[0]]
                    or len(adoption_file_entries) != 1
                    or adoption_file_entries[0].get("declared_path") != adoption_pin.get("path")
                    or adoption_file_entries[0].get("bytes") != adoption_pin.get("bytes")
                    or adoption_file_entries[0].get("sha256") != adoption_pin.get("sha256")):
                raise DispatcherError("terminal adoption receipt pin differs from its file table/model map")
        frame_map_ids = point_entry.get("official_source_frame_map_file_ids")
        npz_by_row = terminal.get("raw_npz_by_row")
        if (not isinstance(frame_map_ids, Mapping) or not isinstance(npz_by_row, Mapping)
                or set(frame_map_ids) != set(npz_by_row)):
            raise DispatcherError("official source frame maps do not cover exactly the passing NPZ rows")
        frame_map_entries = {item["file_id"]: item for item in entries
                             if item.get("role") == "official_source_frame_map"}
        files_by_id = {item["file_id"]: item for item in entries}
        candidate_ids = point_entry.get("candidate_rows_file_ids_by_model")
        shared_output_ids = point_entry.get("shared_output_file_ids_by_sample")
        shared_receipt_ids = point_entry.get("shared_application_file_ids_by_sample")
        origin_ids = point_entry.get("origin_v05_application_file_ids_by_sample")
        projection_ids = point_entry.get("model_projection_file_ids_by_row")
        if (not isinstance(candidate_ids, Mapping) or set(candidate_ids) != set(MODEL_ORDER)
                or not isinstance(shared_output_ids, Mapping)
                or not isinstance(shared_receipt_ids, Mapping)
                or not isinstance(origin_ids, Mapping)
                or not isinstance(projection_ids, Mapping)):
            raise DispatcherError("terminal point entry omits shared application/projection file maps")
        roles_by_id = {key: value.get("role") for key, value in files_by_id.items()}
        if (any(roles_by_id.get(file_id) != "shared_candidate_rows" for file_id in candidate_ids.values())
                or any(roles_by_id.get(file_id) != "shared_generator_drive_wav" for file_id in shared_output_ids.values())
                or any(roles_by_id.get(file_id) != "shared_application_receipt" for file_id in shared_receipt_ids.values())
                or any(roles_by_id.get(file_id) != "shared_origin_v05_application_receipt" for file_id in origin_ids.values())
                or any(roles_by_id.get(file_id) != "model_projection_receipt" for file_id in projection_ids.values())):
            raise DispatcherError("terminal shared-application file maps resolve to the wrong roles")
        if set(frame_map_ids.values()) != set(frame_map_entries):
            raise DispatcherError("official source frame-map files differ from the row binding map")
        for row_id, file_id in frame_map_ids.items():
            frame_map_entry = frame_map_entries.get(file_id)
            if frame_map_entry is None or frame_map_entry.get("pin_name") != row_id:
                raise DispatcherError("official source frame-map pin_name differs from its candidate row")
        if terminal.get("status") not in {"COMPLETE", "COMPLETE_WITH_FAILURES", "INCOMPLETE"}:
            raise DispatcherError("terminal point status is outside the durable session vocabulary")
        if (type(terminal.get("candidate_rows")) is not int
                or type(terminal.get("planned_candidate_rows")) is not int
                or terminal["planned_candidate_rows"] != 24
                or not 0 <= terminal["candidate_rows"] <= terminal["planned_candidate_rows"]):
            raise DispatcherError("terminal candidate/denominator counts are malformed")

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
