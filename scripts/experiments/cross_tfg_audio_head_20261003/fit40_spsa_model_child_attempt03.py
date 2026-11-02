#!/usr/bin/env python3
"""Run one package-pinned native SPSA point block.

The parent dispatcher owns process creation and the numeric child exit receipt.
This entry point validates the frozen family/point/package gates before it
loads a model adapter, then delegates one ordered per-model subset to the
frozen point runner. It does not ask or tell SPSA, select a candidate, or retry.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from typing import Any, Mapping


REQUEST_SCHEMA = "fit40_spsa_model_block_request_v1"
PROFILE_SCHEMA = "fit40_spsa_model_launch_profile_v1"
MODEL_ORDER = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
PROJECT_ROOT = Path("/home/wjj/tts-audio/tts-exp")
ARRAY_LIMIT = 16 * 1024 * 1024


class ChildLaunchError(RuntimeError):
    pass


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _self_sha(value: Mapping[str, Any], field: str, label: str) -> str:
    observed = value.get(field)
    body = {key: item for key, item in value.items() if key != field}
    expected = _sha(_canonical(body))
    if not isinstance(observed, str) or observed != expected:
        raise ChildLaunchError(f"{label} canonical self hash differs")
    return observed


def _read_regular(path_value: Any, label: str, *, max_bytes: int = ARRAY_LIMIT) -> tuple[Path, bytes]:
    if not isinstance(path_value, (str, os.PathLike)):
        raise ChildLaunchError(label + " path must be a string or path-like value")
    try:
        path = Path(path_value)
    except (TypeError, ValueError) as exc:
        raise ChildLaunchError(label + " path is malformed") from exc
    if not path.is_absolute():
        raise ChildLaunchError(label + " path must be absolute")
    if path.is_symlink() or not path.is_file():
        raise ChildLaunchError(label + " must be a non-symlink regular file")
    try:
        info = os.lstat(str(path))
    except OSError as exc:
        raise ChildLaunchError(label + " cannot be stat'ed safely") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ChildLaunchError(label + " must be a non-symlink regular file")
    size = info.st_size
    if size < 0 or size > max_bytes:
        raise ChildLaunchError(label + " exceeds its bounded byte limit")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ChildLaunchError(label + " changed size during read")
    return path.resolve(strict=True), raw


def _read_pin(pin: Any, label: str, *, self_field: str | None = None,
              max_bytes: int = ARRAY_LIMIT) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    if not isinstance(pin, Mapping) or not {"path", "bytes", "sha256"}.issubset(pin):
        raise ChildLaunchError(label + " pin is malformed")
    path, raw = _read_regular(pin["path"], label, max_bytes=max_bytes)
    if (type(pin.get("bytes")) is not int or pin["bytes"] != len(raw)
            or pin.get("sha256") != _sha(raw)):
        raise ChildLaunchError(label + " raw bytes/SHA differ from the declared pin")
    try:
        value = json.loads(raw.decode("utf-8")) if self_field else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChildLaunchError(label + " JSON is malformed") from exc
    if self_field:
        if not isinstance(value, dict):
            raise ChildLaunchError(label + " JSON root must be an object")
        self_sha = _self_sha(value, self_field, label)
        declared_self = pin.get("self_sha256")
        if declared_self is not None and declared_self != self_sha:
            raise ChildLaunchError(label + " self pin differs from its JSON")
    else:
        self_sha = None
    actual = {"path": str(path), "bytes": len(raw), "sha256": _sha(raw)}
    if self_sha is not None:
        actual["self_sha256"] = self_sha
    return actual, raw, value


def _load_module(pin: Mapping[str, Any], label: str, module_name: str) -> Any:
    actual, _raw, _value = _read_pin(pin, label)
    path = Path(actual["path"])
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_path = getattr(existing, "__file__", None)
        if not isinstance(existing_path, str) or Path(existing_path).resolve(strict=True) != path:
            raise ChildLaunchError(label + " module name is already bound to another source")
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ChildLaunchError("cannot import pinned " + label)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    module_path = getattr(module, "__file__", None)
    if not isinstance(module_path, str) or Path(module_path).resolve(strict=True) != path:
        raise ChildLaunchError(label + " imported from an unexpected source")
    return module


def _validate_request(path: str | Path) -> dict[str, Any]:
    _path, raw = _read_regular(path, "block request", max_bytes=2 * 1024 * 1024)
    try:
        request = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChildLaunchError("block request JSON is malformed") from exc
    if not isinstance(request, dict):
        raise ChildLaunchError("block request JSON root must be an object")
    _self_sha(request, "request_sha256", "block request")
    if request.get("schema") != REQUEST_SCHEMA or request.get("status") != "READY":
        raise ChildLaunchError("block request schema/status differs")
    model_id = request.get("model_id")
    if model_id not in MODEL_ORDER:
        raise ChildLaunchError("model ID is outside the exact frozen three-model order")
    return request


def _load_launch_profile(package_entry: Mapping[str, Any], model_id: str,
                         context: Mapping[str, Any], request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_pins = package_entry.get("source_pins")
    profile_pin = source_pins.get("launch_profile") if isinstance(source_pins, Mapping) else None
    actual, _raw, profile = _read_pin(profile_pin, "model launch profile", self_field="profile_sha256")
    if (profile.get("schema") != PROFILE_SCHEMA or profile.get("status") != "FROZEN"
            or profile.get("model_id") != model_id):
        raise ChildLaunchError("launch profile schema/status/model differs")
    interpreter = profile.get("interpreter")
    if not isinstance(interpreter, Mapping) or set(interpreter) != {
            "invocation_path", "resolved_path", "prefix", "python_version"}:
        raise ChildLaunchError("launch profile interpreter identity is malformed")
    python_path = interpreter.get("invocation_path")
    if (not isinstance(python_path, str) or not Path(python_path).is_absolute()
            or os.path.abspath(sys.executable) != os.path.abspath(python_path)
            or os.path.realpath(sys.executable) != interpreter.get("resolved_path")
            or os.path.realpath(sys.prefix) != interpreter.get("prefix")
            or ".".join(str(value) for value in sys.version_info[:3]) != interpreter.get("python_version")):
        raise ChildLaunchError("current interpreter invocation differs from the pinned model profile")
    runtime_pin = package_entry.get("runtime_binding_pin")
    if not isinstance(runtime_pin, Mapping):
        raise ChildLaunchError("execution package has no model runtime binding")
    expected_binding_identity = {key: runtime_pin.get(key) for key in ("path", "bytes", "sha256")}
    if profile.get("runtime_binding_pin") != expected_binding_identity:
        raise ChildLaunchError("launch profile is not bound to the package runtime binding")
    profile_project_root = profile.get("project_root")
    if (not isinstance(profile_project_root, str) or not Path(profile_project_root).is_absolute()
            or Path(profile_project_root).is_symlink() or not Path(profile_project_root).is_dir()):
        raise ChildLaunchError("launch profile project root is absent or unsafe")
    roots = profile.get("python_path_entries")
    if not isinstance(roots, list) or not roots or any(not isinstance(item, str) for item in roots):
        raise ChildLaunchError("launch profile must pin explicit import roots")
    for raw_root in roots:
        root = Path(raw_root)
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise ChildLaunchError("launch profile import root is absent or unsafe")
    if len(set(roots)) != len(roots):
        raise ChildLaunchError("launch profile repeats an import root")
    expected_environment = profile.get("environment")
    if not isinstance(expected_environment, Mapping):
        raise ChildLaunchError("launch profile omits its explicit child environment")
    source_exec = context["family_plan"]["execution"]["source_model_execution"][model_id]
    runtime_pin = package_entry["runtime_binding_pin"]
    substitutions = {
        "strict_receipt_dir": str(request.get("strict_receipt_dir", "")),
        "runtime_binding_path": str(runtime_pin["path"]),
        "runtime_binding_sha256": str(runtime_pin["sha256"]),
        "capacity_stage": str(source_exec["capacity_stage"]),
    }
    def expand(value: str) -> str:
        try:
            return value.format_map(substitutions)
        except (KeyError, ValueError) as exc:
            raise ChildLaunchError("launch profile contains an unknown environment substitution") from exc
    materialized_environment = {key: expand(value) for key, value in expected_environment.items()
                                if isinstance(key, str) and isinstance(value, str)}
    if len(materialized_environment) != len(expected_environment):
        raise ChildLaunchError("launch profile environment keys/values must be strings")
    for key, value in materialized_environment.items():
        if os.environ.get(key) != value:
            raise ChildLaunchError("child environment differs from the pinned profile: " + str(key))
    if set(os.environ) != set(materialized_environment):
        raise ChildLaunchError("child environment includes keys outside the exact pinned profile")
    sys.path[:0] = [str(Path(item).resolve(strict=True)) for item in roots]
    return dict(profile), actual


def _module_pin(package_entry: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    pins = package_entry.get("source_pins")
    pin = pins.get(role) if isinstance(pins, Mapping) else None
    if not isinstance(pin, Mapping):
        raise ChildLaunchError("frozen model source closure lacks role " + role)
    return pin


def _adapter_factory_factory(request: Mapping[str, Any], runner: Any,
                             package_entry: Mapping[str, Any], profile: Mapping[str, Any]):
    model_id = request["model_id"]

    def create(context: Mapping[str, Any]) -> Any:
        # All 720-row, selector, point, runtime and package checks run before
        # this callback. Model modules are imported only after those gates.
        core = _load_module(package_entry["runner_pin"], "Stage-A core", "fit40_pinned_stagea_core")
        adapter_module = _load_module(package_entry["adapter_pin"], "model adapter", "fit40_pinned_spsa_adapter")
        runtime_actual, runtime_raw, runtime = _read_pin(
            package_entry["runtime_binding_pin"], "model runtime binding", self_field="binding_sha256")
        args = dict(profile.get("adapter_args", {}))
        args["project_root"] = profile["project_root"]
        namespace = SimpleNamespace(**args)
        if model_id == MODEL_ORDER[0]:
            cls = getattr(adapter_module, "W2LSPSAStageAAdapter", None)
            if cls is None:
                raise ChildLaunchError("pinned W2L adapter class is absent")
            return cls(core=core, family_plan=context["family_plan"],
                       family_plan_pin=context["family_plan_pin"],
                       point_plan=context["point_plan"], point_plan_pin=context["point_plan_pin"],
                       args=namespace)
        if model_id == MODEL_ORDER[1]:
            overlay = _load_module(_module_pin(package_entry, "musetalk_block_overlay"),
                                   "Muse block overlay", "fit40_pinned_musetalk_overlay")
            cross = _load_module(_module_pin(package_entry, "cross_frontend_runner"),
                                 "Muse cross frontend", "fit40_pinned_cross_frontend")
            helper = _load_module(_module_pin(package_entry, "resource_helper"),
                                  "Muse resource helper", "fit40_pinned_muse_resource_helper")
            strict_dir = Path(str(request["strict_receipt_dir"]))
            block_root = Path(str(request["block_root"]))
            if (not block_root.is_absolute() or block_root.is_symlink() or not block_root.is_dir()
                    or strict_dir.parent.resolve(strict=True) != block_root.resolve(strict=True)
                    or strict_dir.exists() or strict_dir.is_symlink()):
                raise ChildLaunchError("Muse strict receipt directory must be a fresh direct child of its block root")
            strict_dir.mkdir(parents=True, exist_ok=False)
            binding_path = Path(str(package_entry["runtime_binding_pin"]["path"]))
            if (not binding_path.is_absolute() or binding_path.is_symlink()
                    or not binding_path.is_file()):
                raise ChildLaunchError("Muse runtime binding is not a pinned regular file")
            cls = getattr(adapter_module, "MuseSPSAStageAAdapter", None)
            if cls is None:
                raise ChildLaunchError("pinned Muse adapter class is absent")
            return cls(core=core, muse_overlay=overlay, cross_frontend=cross,
                       parent_plan=context["family_plan"], runtime_binding=runtime,
                       runtime_binding_raw=runtime_raw, runtime_binding_path=binding_path,
                       family_plan_path=context["family_plan_pin"]["path"],
                       point_plan=context["point_plan"], point_plan_path=context["point_plan_pin"]["path"],
                       point_plan_file_pin={key: context["point_plan_pin"][key]
                                            for key in ("path", "bytes", "sha256")},
                       point_plan_self_sha256=context["point_plan_pin"]["self_sha256"],
                       resource_helper=helper, args=namespace, strict_receipt_dir=strict_dir)
        if model_id == MODEL_ORDER[2]:
            overlay = _load_module(_module_pin(package_entry, "ditto_block_overlay"),
                                   "Ditto block overlay", "fit40_pinned_ditto_overlay")
            helper = _load_module(_module_pin(package_entry, "resource_helper"),
                                  "Ditto resource helper", "fit40_pinned_ditto_resource_helper")
            cls = getattr(adapter_module, "DittoSPSAStageAAdapter", None)
            if cls is None:
                raise ChildLaunchError("pinned Ditto adapter class is absent")
            return cls(core=core, ditto_overlay=overlay,
                       point_plan=context["point_plan"], point_plan_pin=context["point_plan_pin"],
                       resource_helper=helper, args=namespace)
        raise ChildLaunchError("model adapter dispatch fell outside the frozen roster")

    return create


def execute_request(request_path: str | Path) -> int:
    try:
        path = Path(request_path)
    except (TypeError, ValueError) as exc:
        raise ChildLaunchError("block request path is malformed") from exc
    if not path.is_absolute():
        raise ChildLaunchError("block request path must be absolute")
    if path.is_symlink() or not path.is_file():
        raise ChildLaunchError("block request must be an existing non-symlink file")
    request = _validate_request(path.resolve(strict=True))
    runner_actual, _raw, _unused = _read_pin(
        request.get("point_runner_pin"), "point runner")
    package_actual, _package_raw, package = _read_pin(
        request.get("execution_package_manifest_pin"), "execution package manifest",
        self_field="manifest_sha256")
    model_id = request["model_id"]
    package_entry = package.get("execution_sources_by_model", {}).get(model_id)
    if not isinstance(package_entry, Mapping):
        raise ChildLaunchError("execution package lacks the requested model")
    launcher_actual, _launcher_raw, _launcher_value = _read_pin(
        package_entry.get("launcher_pin"), "common model child launcher")
    source_path = Path(__file__).resolve(strict=True)
    source_raw = source_path.read_bytes()
    if (launcher_actual.get("path") != str(source_path)
            or launcher_actual.get("bytes") != len(source_raw)
            or launcher_actual.get("sha256") != _sha(source_raw)):
        raise ChildLaunchError("common child launcher source differs from its frozen package pin")
    block_root = Path(str(request.get("block_root", "")))
    output_root = Path(str(request.get("output_root", "")))
    if (not block_root.is_absolute() or block_root.is_symlink() or not block_root.is_dir()
            or output_root.parent.resolve(strict=True) != block_root.resolve(strict=True)
            or output_root.exists() or output_root.is_symlink()):
        raise ChildLaunchError("block/output roots are unsafe or already used")
    runner = _load_module(runner_actual, "point runner", "fit40_pinned_point_runner")
    # Validate the complete 720 gate and exact one-method projection before
    # adapter imports or render/scoring work.
    context = runner.validate_point_scope(
        family_plan_pin=request["family_plan_pin"], point_plan_pin=request["point_plan_pin"],
        initial_selector_receipt_pin=request["initial_selector_receipt_pin"],
        execution_package_manifest_pin=request["execution_package_manifest_pin"], model_id=model_id)
    if context["execution_package_manifest_pin"] != package_actual:
        raise ChildLaunchError("request package raw/self pin differs from validated point-runner context")
    profile, _profile_actual = _load_launch_profile(package_entry, model_id, context, request)
    factory = _adapter_factory_factory(request, runner, package_entry, profile)
    result = runner.execute_point_block(
        adapter_factory=factory,
        family_plan_pin=request["family_plan_pin"], point_plan_pin=request["point_plan_pin"],
        initial_selector_receipt_pin=request["initial_selector_receipt_pin"],
        execution_package_manifest_pin=request["execution_package_manifest_pin"],
        model_id=model_id, output_root=request["output_root"],
        start_utc=request.get("child_started_utc"))
    return result["exit_code"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    return execute_request(args.request)


if __name__ == "__main__":
    raise SystemExit(main())
