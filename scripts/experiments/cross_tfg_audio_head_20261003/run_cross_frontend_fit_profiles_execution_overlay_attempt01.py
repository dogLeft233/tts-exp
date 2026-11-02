#!/usr/bin/env python3
"""Operational provenance/control overlay for the frozen profile dispatcher.

The scientific transform, source rows, generator adapters and scoring core remain
the frozen implementations.  This wrapper adds a row-bound execution receipt
and gives each Ditto row a unique writable GPU-gate directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

DISPATCHER_SHA256 = "dacda9729db79ec96c73c19402866b78fb3d695b2560ca6f285ce4ce0c056660"
OVERLAY_SCHEMA = "cross_tfg_execution_overlay_v1"
ADAPTER_CONTRACT = "cross_tfg_silent_video_cli_v1"
_DISPATCHER: Any = None
_DISPATCHER_LOAD_STAGEA: Any = None
_DISPATCHER_LOAD_MODULE: Any = None
_STATE: dict[str, Any] = {}
_ACTIVE: dict[str, dict[str, Any]] = {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _load_dispatcher(path: Path, expected_sha256: str):
    global _DISPATCHER, _DISPATCHER_LOAD_STAGEA, _DISPATCHER_LOAD_MODULE
    path = path.resolve(strict=True)
    if len(expected_sha256) != 64 or sha256_file(path) != expected_sha256.lower():
        raise RuntimeError("frozen shared-profile dispatcher SHA mismatch")
    spec = importlib.util.spec_from_file_location("cross_tfg_frozen_shared_profile_dispatcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen shared-profile dispatcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _DISPATCHER = module
    _DISPATCHER_LOAD_STAGEA = module._load_stagea
    _DISPATCHER_LOAD_MODULE = module._load_module
    module._load_stagea = _dispatcher_load_stagea
    module._load_module = _dispatcher_load_module
    return module


def _entry_source_sha() -> str:
    return sha256_file(Path(__file__).resolve(strict=True))


def _row_identity(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (str(row.get("row_id", "")), str(row.get("model_id", "")),
            str(row.get("method_id", "")), int(row.get("seed", -1)))


def _make_binding(row: Mapping[str, Any], model: Mapping[str, Any], work: Path) -> dict[str, Any]:
    row_id, model_id, method_id, seed = _row_identity(row)
    if not row_id or not model_id or not method_id or seed < 0:
        raise RuntimeError("planned row lacks its execution-overlay identity")
    work = work.resolve()
    control_root = work / "control"
    generator = model.get("generator", {})
    is_adapter = isinstance(generator, Mapping) and generator.get("contract") == ADAPTER_CONTRACT
    ditto_control = control_root / "ditto_gpu_gate" if model_id.startswith("ditto_") else None
    env = {
        "CROSS_TFG_EXECUTION_OVERLAY_PATH": str(Path(__file__).resolve()),
        "CROSS_TFG_EXECUTION_OVERLAY_SHA256": _entry_source_sha(),
        "CROSS_TFG_EXECUTION_OVERLAY_MANIFEST_PATH": _STATE["manifest_path"],
        "CROSS_TFG_EXECUTION_OVERLAY_MANIFEST_SHA256": _STATE["manifest_sha256"],
        "CROSS_TFG_ROW_CONTROL_DIR": str(control_root),
    }
    if ditto_control is not None:
        env["CROSS_TFG_CONTROL_DIR"] = str(ditto_control)
    return {
        "schema": OVERLAY_SCHEMA,
        "run_id": _STATE["run_id"],
        "plan_path": _STATE["plan_path"],
        "plan_raw_sha256": _STATE["plan_sha256"],
        "entrypoint_path": str(Path(__file__).resolve()),
        "entrypoint_sha256": env["CROSS_TFG_EXECUTION_OVERLAY_SHA256"],
        "manifest_path": _STATE["manifest_path"],
        "manifest_sha256": _STATE["manifest_sha256"],
        "dispatcher_path": _STATE["dispatcher_path"],
        "dispatcher_sha256": DISPATCHER_SHA256,
        "row_id": row_id,
        "model_id": model_id,
        "method_id": method_id,
        "seed": seed,
        "row_work_dir": str(work),
        "effective_row_control_dir": str(control_root),
        "effective_model_control_dir": str(ditto_control) if ditto_control is not None else None,
        "adapter_contract": ADAPTER_CONTRACT if is_adapter else None,
        "environment_bindings": env,
        "scientific_input_override": False,
        "plan_or_method_mutation": False,
    }


def _write_binding(binding: Mapping[str, Any], work: Path) -> dict[str, Any]:
    control = work / "control"
    control.mkdir(parents=True, exist_ok=True)
    path = control / "execution_overlay.json"
    raw = canonical(dict(binding)) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {**dict(binding), "row_receipt_path": str(path.resolve()),
            "row_receipt_sha256": hashlib.sha256(raw).hexdigest(), "row_receipt_bytes": len(raw)}


def _recorded_row_path(row_id: str, *, failed: bool) -> Path:
    root = Path(_STATE["output_root"])
    candidate = root / "rows" / row_id
    if candidate.exists() or not failed:
        return candidate
    return root / "row_failures" / row_id


def _instrument_core(core: Any) -> Any:
    original_generator = core._run_generator
    original_render = core._render_and_score
    original_identity = core._identity_base

    def tagged_generator(**kwargs):
        row = kwargs["row"]
        model = kwargs["model"]
        work = Path(kwargs["work"])
        binding = _make_binding(row, model, work)
        previous = {key: os.environ.get(key) for key in binding["environment_bindings"]}
        try:
            os.environ.update(binding["environment_bindings"])
            result = original_generator(**kwargs)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        result["execution_overlay"] = binding
        return result

    def tagged_render(row, model, method, natural_path, samples, natural_ref, row_root, args, *, qc_repeat=False):
        binding = _make_binding(row, model, Path(row_root))
        _ACTIVE[str(row["row_id"])] = {"binding": binding, "work": Path(row_root)}
        result = original_render(row, model, method, natural_path, samples, natural_ref, row_root, args, qc_repeat=qc_repeat)
        recorded = _write_binding(binding, Path(row_root))
        result["execution_overlay"] = recorded
        _ACTIVE.pop(str(row["row_id"]), None)
        return result

    def tagged_identity(row, natural_ref):
        result = original_identity(row, natural_ref)
        row_id = str(row.get("row_id", ""))
        active = _ACTIVE.pop(row_id, None)
        if active is not None:
            binding, work = active["binding"], active["work"]
        else:
            model = _STATE["models"].get(str(row.get("model_id", "")), {})
            work = _recorded_row_path(row_id, failed=True)
            binding = _make_binding(row, model, work)
        result["execution_overlay"] = _write_binding(binding, work)
        return result

    core._run_generator = tagged_generator
    core._render_and_score = tagged_render
    core._identity_base = tagged_identity
    return core


def _load_stagea(path: Path):
    core = _DISPATCHER_LOAD_STAGEA(path)
    return _instrument_core(core)


def _dispatcher_load_stagea(path: Path):
    # Route through this module's patchable loader so the separately pinned GPU
    # admission wrapper can instrument core.run_logged before any GPU stage.
    return _load_stagea(path)


def _dispatcher_load_module(name: str, path: Path, expected_sha256: str):
    module, resolved = _DISPATCHER_LOAD_MODULE(name, path, expected_sha256)
    if name == "cross_tfg_pinned_cross_frontend_adapter_runner":
        original_adapter = module._adapter_generator

        def tagged_adapter(core: Any, **kwargs):
            row = kwargs["row"]
            model = kwargs["model"]
            work = Path(kwargs["work"])
            binding = _make_binding(row, model, work)
            copied_model = dict(model)
            copied_generator = dict(model.get("generator", {}))
            env = dict(copied_generator.get("env", {}))
            env.update(binding["environment_bindings"])
            copied_generator["env"] = env
            copied_model["generator"] = copied_generator
            result = original_adapter(core, **{**kwargs, "model": copied_model})
            result["execution_overlay"] = binding
            return result

        module._adapter_generator = tagged_adapter
    return module, resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--execution-overlay-manifest", required=True)
    parser.add_argument("--expected-execution-overlay-manifest-sha256", required=True)
    parser.add_argument("--execution-overlay-dispatcher", required=True)
    parser.add_argument("--execution-overlay-expected-plan-raw-sha256", required=True)
    own, remaining = parser.parse_known_args(argv)

    manifest_path = Path(own.execution_overlay_manifest).resolve(strict=True)
    manifest_sha = sha256_file(manifest_path)
    if manifest_sha != own.expected_execution_overlay_manifest_sha256.lower():
        raise RuntimeError("execution-overlay manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != OVERLAY_SCHEMA or manifest.get("entrypoint_sha256") != _entry_source_sha():
        raise RuntimeError("execution-overlay manifest does not bind this entrypoint")
    if manifest.get("dispatcher_sha256") != DISPATCHER_SHA256:
        raise RuntimeError("execution-overlay manifest dispatcher pin mismatch")
    if manifest.get("entrypoint_path") != str(Path(__file__).resolve()):
        raise RuntimeError("execution-overlay manifest entrypoint path mismatch")
    dispatcher_path = Path(own.execution_overlay_dispatcher).resolve(strict=True)
    if manifest.get("dispatcher_path") != str(dispatcher_path):
        raise RuntimeError("execution-overlay manifest dispatcher path mismatch")
    if manifest.get("ditto_control_dir_template") != "{output_root}/rows/{row_id}/control/ditto_gpu_gate":
        raise RuntimeError("execution-overlay manifest Ditto control-directory policy mismatch")
    if manifest.get("control_override_scope") != ["CROSS_TFG_CONTROL_DIR for Ditto rows only"]:
        raise RuntimeError("execution-overlay manifest operational override scope mismatch")

    core_args = argparse.ArgumentParser(add_help=False)
    core_args.add_argument("--plan", required=True)
    core_args.add_argument("--output-root", required=True)
    selected, _ = core_args.parse_known_args(remaining)
    plan_path = Path(selected.plan).resolve(strict=True)
    plan_sha = sha256_file(plan_path)
    if plan_sha != own.execution_overlay_expected_plan_raw_sha256.lower() or plan_sha != manifest.get("plan_raw_sha256"):
        raise RuntimeError("execution overlay is bound to a different frozen plan")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("run_id") != manifest.get("run_id"):
        raise RuntimeError("execution overlay plan run_id mismatch")
    if manifest.get("plan_path") != str(plan_path):
        raise RuntimeError("execution overlay plan path mismatch")
    output_root = Path(selected.output_root).resolve()
    _STATE.update({
        "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
        "dispatcher_path": str(dispatcher_path),
        "plan_path": str(plan_path), "plan_sha256": plan_sha, "run_id": plan["run_id"],
        "output_root": str(output_root), "models": {item["model_id"]: item for item in plan["models"]},
    })
    dispatcher = _load_dispatcher(Path(own.execution_overlay_dispatcher), DISPATCHER_SHA256)
    return int(dispatcher.main(remaining))


if __name__ == "__main__":
    raise SystemExit(main())
