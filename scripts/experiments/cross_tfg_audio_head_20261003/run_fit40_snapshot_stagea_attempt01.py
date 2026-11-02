#!/usr/bin/env python3
"""Score four persisted Fit40 snapshots through the frozen three-model Stage-A core.

This adapter leaves Stage-A's native rendering, original-PCM mux-back, and
SyncNet scoring code pinned and unchanged. It adds the persisted parameter
transform and a separate 720-row Fit capacity gate. Dispatch is rejected
unless the plan binds an actual all-three-backend qualification/readiness
receipt, snapshot index, runtime binding, and measured capacity receipt.
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

import numpy as np

from build_fit40_snapshot_stagea_plan_attempt01 import (
    BASE_METHOD_IDS,
    CAPACITY_STAGE,
    FRONTEND_ADAPTER_BYTES,
    FRONTEND_ADAPTER_REL,
    FRONTEND_ADAPTER_SHA256,
    REQUIRED_MODEL_IDS,
    RUNTIME_HELPER_BYTES,
    RUNTIME_HELPER_REL,
    RUNTIME_HELPER_SHA256,
    STAGEA_CORE_BYTES,
    STAGEA_CORE_REL,
    STAGEA_CORE_SHA256,
    _verify_backend_readiness,
    pin_file,
    read_pinned_json,
)
from fit40_training_snapshots_attempt01 import (
    FIT_PAIRS,
    FIT_SPEAKERS,
    PROFILES,
    FitSnapshotTransform,
    SnapshotError,
    canonical_bytes,
    sha256_bytes,
    sha256_file,
    verify_snapshot_index,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/run_fit40_snapshot_stagea_attempt01.py"
ADAPTER_CONTRACT = "cross_tfg_silent_video_cli_v1"
W2L_MODEL_ID = REQUIRED_MODEL_IDS[0]
METHOD_KIND = "fit40_parameter_snapshot"
CAPACITY_STAGE_ENV = "CROSS_TFG_FIT_SNAPSHOT_CAPACITY_STAGE"
BINDING_PATH_ENV = "CROSS_TFG_FIT_SNAPSHOT_RUNTIME_BINDING_PATH"
BINDING_SHA_ENV = "CROSS_TFG_FIT_SNAPSHOT_RUNTIME_BINDING_RAW_SHA256"


def _load_module(name: str, path: Path, expected_bytes: int, expected_sha256: str):
    pin = pin_file(path)
    if pin["bytes"] != expected_bytes or pin["sha256"] != expected_sha256:
        raise SnapshotError(f"pinned source mismatch: {path}")
    spec = importlib.util.spec_from_file_location(name, path.resolve(strict=True))
    if spec is None or spec.loader is None:
        raise SnapshotError(f"cannot load pinned module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, pin


def _load_runtime_binding(plan: Mapping[str, Any], expected_snapshot_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    ref = plan.get("execution", {}).get("fit_snapshot_runtime_binding")
    if not isinstance(ref, Mapping):
        raise SnapshotError("Fit snapshot runtime binding is not frozen in the plan")
    env_path = os.environ.get(BINDING_PATH_ENV, "")
    env_sha = os.environ.get(BINDING_SHA_ENV, "").lower()
    if not env_path or not env_sha or len(env_sha) != 64:
        raise SnapshotError("explicit Fit snapshot runtime-binding path and raw SHA are required in the environment")
    target = Path(env_path)
    if not target.is_absolute() or target.resolve(strict=True) != Path(str(ref.get("path"))).resolve(strict=True):
        raise SnapshotError("environment runtime-binding path differs from the frozen plan")
    actual = pin_file(target)
    if actual["bytes"] != ref.get("bytes") or actual["sha256"] != ref.get("sha256") or actual["sha256"] != env_sha:
        raise SnapshotError("runtime-binding raw bytes differ from both plan and caller pins")
    binding, raw = read_pinned_json(target, env_sha, "Fit snapshot runtime binding")
    if (binding.get("schema") != "cross_tfg_fit40_snapshot_runtime_binding_v1"
            or binding.get("status") != "PASS"
            or binding.get("binding_sha256") != sha256_bytes(canonical_bytes({k: v for k, v in binding.items() if k != "binding_sha256"}))
            or binding.get("planned_rows") != 720
            or binding.get("model_ids") != list(REQUIRED_MODEL_IDS)
            or binding.get("snapshot_set_index_raw_sha256") != expected_snapshot_sha256):
        raise SnapshotError("runtime binding self-hash or Fit snapshot plan scope is invalid")
    return binding, actual


def _validate_snapshot_plan(plan_path: Path, rows_path: Path, expected_plan_sha256: str,
                            project_root: Path, *, core_pin: Mapping[str, Any],
                            frontend_pin: Mapping[str, Any]) -> dict[str, Any]:
    if plan_path.is_symlink() or rows_path.is_symlink():
        raise SnapshotError("plan and planned rows may not be symlinks")
    plan, plan_raw = read_pinned_json(plan_path, expected_plan_sha256, "Fit snapshot plan")
    if (plan.get("schema_version") != "cross-tfg-audio-eval-v1" or plan.get("status") != "FROZEN"
            or plan.get("plan_sha256") != sha256_bytes(canonical_bytes({k: v for k, v in plan.items() if k != "plan_sha256"}))):
        raise SnapshotError("Fit snapshot plan schema/status/self-hash is invalid")
    if plan.get("execution", {}).get("stagea_core_sha256") != core_pin.get("sha256"):
        raise SnapshotError("plan does not bind the exact Stage-A core")
    if plan.get("execution", {}).get("cross_frontend_adapter_sha256") != frontend_pin.get("sha256"):
        raise SnapshotError("plan does not bind the exact native video adapter")
    if plan.get("execution", {}).get("fit_snapshot_runner_sha256") != sha256_file(Path(__file__)):
        raise SnapshotError("plan does not bind this exact Fit snapshot runner")
    if plan.get("execution", {}).get("scope") != "fit-only; no development or confirmation audio opened":
        raise SnapshotError("plan scope is not the frozen Fit-only scope")
    methods = plan.get("methods")
    models = plan.get("models")
    rows = plan.get("rows")
    if (not isinstance(methods, list) or not isinstance(models, list) or not isinstance(rows, list)
            or len(rows) != 720 or len(methods) != 6
            or [item.get("model_id") for item in models] != list(REQUIRED_MODEL_IDS)
            or [item.get("method_id") for item in methods[:2]] != list(BASE_METHOD_IDS)):
        raise SnapshotError("Fit snapshot plan must contain 40×3×6 rows over the exact model/method roster")
    if any(row.get("split") != "fit" or row.get("score_audio") != row.get("natural_pcm") for row in rows):
        raise SnapshotError("Fit rows must score the original natural PCM on the original source clock")
    if plan.get("generation_repeat_qc") != []:
        raise SnapshotError("this snapshot plan must not declare an unqualified repeat run")
    if plan.get("snapshot_native_endpoint", {}).get("status") != "UNBOUND_NATIVE_FIT_SCORES_NOT_RUN":
        raise SnapshotError("plan falsely claims the all-three-backend endpoint already exists")

    index_ref = plan.get("execution", {}).get("snapshot_set_index")
    if not isinstance(index_ref, Mapping):
        raise SnapshotError("plan does not pin its snapshot-set index")
    index_path = Path(str(index_ref.get("path", "")))
    if not index_path.is_absolute():
        raise SnapshotError("snapshot index path must be absolute for this execution host")
    index, snapshot_records = verify_snapshot_index(index_path, str(index_ref.get("raw_sha256", "")))
    if (index_ref.get("bytes") != index_path.stat().st_size
            or index.get("index_sha256") != index_ref.get("index_sha256")):
        raise SnapshotError("snapshot index byte size or canonical self-hash differs from plan")
    snapshot_method_ids = [entry["method_id"] for entry in index["snapshots"]]
    if [item.get("method_id") for item in methods[2:]] != snapshot_method_ids:
        raise SnapshotError("plan candidate methods differ from the pinned four snapshot records")

    method_by_id = {item["method_id"]: item for item in methods}
    for entry in index["snapshots"]:
        method = method_by_id[entry["method_id"]]
        config = method.get("config", {})
        if (method.get("kind") != "fit_parameter_snapshot"
                or config.get("method") != METHOD_KIND
                or config.get("snapshot_index_raw_sha256") != index_ref.get("raw_sha256")
                or config.get("snapshot_id") != entry.get("snapshot_id")
                or config.get("source_optimizer_step") != entry.get("source_optimizer_step")
                or config.get("snapshot_record_raw_sha256") != entry.get("sha256")
                or method.get("config_sha256") != sha256_bytes(canonical_bytes(config))):
            raise SnapshotError(f"snapshot method config is not bound to its exact record: {entry['method_id']}")

    rows_bytes = rows_path.read_bytes()
    planned_rows = [json.loads(line) for line in rows_bytes.splitlines() if line.strip()]
    if len(planned_rows) != 720 or planned_rows != rows:
        raise SnapshotError("planned_rows.jsonl does not equal the exact ordered frozen plan rows")
    if len({row.get("row_id") for row in planned_rows}) != 720:
        raise SnapshotError("Fit snapshot plan contains duplicate row IDs")
    row_ids_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in planned_rows:
        key = (str(row.get("sample_id")), str(row.get("model_id")), str(row.get("method_id")), str(row.get("seed")))
        if key in row_ids_by_key:
            raise SnapshotError("duplicate Fit context/model/method/seed tuple")
        row_ids_by_key[key] = row
    samples = {str(row["sample_id"]) for row in planned_rows}
    if len(samples) != 40 or {str(row["speaker_id"]) for row in planned_rows} != set(FIT_SPEAKERS):
        raise SnapshotError("Fit snapshot denominator does not contain all 40 rows and 20 speakers")
    seeds = {row.get("seed") for row in planned_rows}
    if len(seeds) != 1:
        raise SnapshotError("Fit snapshot methods must share one predeclared seed")
    if len(row_ids_by_key) != 720:
        raise SnapshotError("Fit snapshot row identity roster is incomplete")
    for sample_id in samples:
        for model_id in REQUIRED_MODEL_IDS:
            group = [row_ids_by_key[(sample_id, model_id, method_id, str(next(iter(seeds))))]
                     for method_id in [*BASE_METHOD_IDS, *snapshot_method_ids]]
            natural = group[0].get("natural_pcm")
            if any(row.get("natural_pcm") != natural or row.get("score_audio") != natural for row in group):
                raise SnapshotError("six-way Fit comparisons do not share exact original PCM/SyncNet audio")

    readiness_ref = plan.get("execution", {}).get("backend_readiness")
    if not isinstance(readiness_ref, Mapping):
        raise SnapshotError("plan does not pin all-three-backend readiness")
    readiness_path = Path(str(readiness_ref.get("path", "")))
    readiness, readiness_raw = read_pinned_json(readiness_path, str(readiness_ref.get("sha256", "")), "backend readiness")
    if len(readiness_raw) != readiness_ref.get("bytes"):
        raise SnapshotError("backend readiness byte size differs from plan")
    readiness_info = _verify_backend_readiness(readiness, readiness_raw, str(index_ref["raw_sha256"]))
    binding_ref = plan.get("execution", {}).get("fit_snapshot_runtime_binding")
    capacity_ref = plan.get("execution", {}).get("fit_snapshot_capacity_receipt")
    if (binding_ref != readiness_info["fit_runtime_binding"]
            or capacity_ref != readiness_info["capacity_receipt"]):
        raise SnapshotError("plan runtime/capacity refs differ from the reviewed readiness manifest")

    runtime_binding, actual_binding = _load_runtime_binding(plan, str(index_ref["raw_sha256"]))
    if (actual_binding != binding_ref or runtime_binding.get("capacity_receipt_raw_sha256") != capacity_ref.get("sha256")
            or runtime_binding.get("snapshot_method_ids") != snapshot_method_ids):
        raise SnapshotError("runtime binding differs from plan/readiness snapshot scope")
    runner_pin = pin_file(Path(__file__))
    return {
        "plan": plan,
        "rows": planned_rows,
        "snapshot_index": index,
        "snapshot_records": snapshot_records,
        "method_by_id": method_by_id,
        "backend_readiness": readiness,
        "backend_readiness_info": readiness_info,
        "runtime_binding": runtime_binding,
        "runtime_binding_pin": actual_binding,
        "plan_pin": {"path": str(plan_path.resolve()), "bytes": len(plan_raw), "sha256": sha256_bytes(plan_raw)},
        "planned_rows_pin": {"path": str(rows_path.resolve()), "bytes": len(rows_bytes), "sha256": sha256_bytes(rows_bytes)},
        "runner_pin": runner_pin,
        "snapshot_index_path": index_path,
        "project_root": project_root,
    }


def _row_resource_gate(core: Any, helper: Any, prepared: Mapping[str, Any],
                       plan: Mapping[str, Any], row: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    binding_path = Path(os.environ[BINDING_PATH_ENV]).resolve(strict=True)
    binding_raw = binding_path.read_bytes()
    binding_sha = sha256_bytes(binding_raw)
    if binding_sha != os.environ[BINDING_SHA_ENV].lower() or binding_sha != prepared["runtime_binding_pin"]["sha256"]:
        raise SnapshotError("runtime binding changed during the Stage-A run")
    resources = prepared["runtime_binding"].get("resources", {})
    basis = resources.get("capacity_basis", {})
    stage = os.environ.get(CAPACITY_STAGE_ENV, "")
    if stage != CAPACITY_STAGE or stage not in basis.get("stage_output_required_bytes", {}):
        raise SnapshotError("live capacity stage differs from the frozen 720-row Fit snapshot binding")
    per_backend = basis.get("per_backend_controls_row_output_bytes", {})
    model_id = str(row.get("model_id", ""))
    slack = float(basis.get("capacity_scenario", {}).get("explicit_slack_multiplier", 0.0))
    reserve = resources.get("output_reserve_bytes")
    if (model_id not in REQUIRED_MODEL_IDS or type(per_backend.get(model_id)) is not int
            or per_backend[model_id] <= 0 or type(reserve) is not int or reserve <= 0 or slack < 1):
        raise SnapshotError("live row resource allowance is missing for this backend")
    row_minimum = reserve + int(per_backend[model_id] * slack)
    snapshot = helper.resource_snapshot(output_root, {
        "min_gpu_free_mib": 0,
        "min_ram_available_bytes": 0,
        "min_shm_free_bytes": 0,
        "min_output_free_bytes": 0,
    })
    gpu_uuid = resources["gpu_uuid"]
    gpu = next((item for item in snapshot.get("gpus", []) if item.get("uuid") == gpu_uuid), None)
    reasons: list[str] = []
    if gpu is None:
        reasons.append("bound GPU UUID is absent from the live inventory")
    elif int(gpu.get("memory_free_mib", -1)) < resources["min_gpu_free_mib"]:
        reasons.append("live GPU free memory is below the bound minimum")
    ram = snapshot.get("ram", {})
    if int(ram.get("available_bytes", -1)) < resources["min_effective_ram_available_bytes"]:
        reasons.append("live effective RAM is below the bound minimum")
    shm = snapshot.get("shm", {})
    if int(shm.get("available_bytes", -1)) < max(resources["min_shm_free_bytes"], row_minimum):
        reasons.append("live shared memory is below the bound minimum")
    disk = snapshot.get("output_disk", {})
    if disk.get("target") != resources["output_mount_target"]:
        reasons.append("live output mount differs from the binding")
    if int(disk.get("available_bytes", -1)) < row_minimum:
        reasons.append("live output free space is below reserve plus this backend's row allowance")
    cgroup = ram.get("cgroup", {})
    expected_cgroup = resources.get("cgroup", {})
    if cgroup.get("version") != expected_cgroup.get("version"):
        reasons.append("live cgroup version differs from the binding")
    actual_identity = cgroup.get("identity", {})
    expected_identity = expected_cgroup.get("identity", {})
    for key in ("mount_target", "mount_root", "membership_path", "leaf_path", "finite_limit", "limit_bytes"):
        if actual_identity.get(key) != expected_identity.get(key):
            reasons.append(f"live cgroup identity differs at {key}")
    return {
        "schema": "cross_tfg_stagea_per_row_resource_gate_v1",
        "status": "FAILED_RESOURCE_GATE" if reasons else "PASS",
        "row_id": row.get("row_id"),
        "model_id": model_id,
        "runtime_binding_raw_sha256": binding_sha,
        "capacity_stage": stage,
        "required_output_bytes": row_minimum,
        "gpu_uuid": gpu_uuid,
        "gpu_free_mib": gpu.get("memory_free_mib") if gpu else None,
        "ram_available_bytes": ram.get("available_bytes"),
        "shm_available_bytes": shm.get("available_bytes"),
        "output_disk": disk,
        "cgroup_identity_sha256": sha256_bytes(canonical_bytes(actual_identity)),
        "failure_reasons": reasons,
    }


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stagea-core", type=Path, default=REPO_ROOT / STAGEA_CORE_REL)
    parser.add_argument("--cross-frontend-adapter-runner", type=Path, default=REPO_ROOT / FRONTEND_ADAPTER_REL)
    own, remaining = parser.parse_known_args(argv)
    core, core_pin = _load_module("fit40_snapshot_stagea_core", own.stagea_core, STAGEA_CORE_BYTES, STAGEA_CORE_SHA256)
    frontend, frontend_pin = _load_module("fit40_snapshot_native_video_adapter", own.cross_frontend_adapter_runner,
                                          FRONTEND_ADAPTER_BYTES, FRONTEND_ADAPTER_SHA256)
    if (own.stagea_core.resolve(strict=True) != (REPO_ROOT / STAGEA_CORE_REL).resolve(strict=True)
            or own.cross_frontend_adapter_runner.resolve(strict=True) != (REPO_ROOT / FRONTEND_ADAPTER_REL).resolve(strict=True)):
        raise SnapshotError("caller selected a non-frozen Stage-A core or native video adapter")
    stagea_args = core.build_parser().parse_args(remaining)
    project_root = Path(stagea_args.project_root).resolve(strict=True)
    helper_path = project_root / RUNTIME_HELPER_REL
    helper, helper_pin = _load_module("fit40_snapshot_runtime_resource_helper", helper_path,
                                      RUNTIME_HELPER_BYTES, RUNTIME_HELPER_SHA256)
    prepared = _validate_snapshot_plan(
        Path(stagea_args.plan), Path(stagea_args.planned_rows), stagea_args.expected_plan_raw_sha256,
        project_root, core_pin=core_pin, frontend_pin=frontend_pin,
    )
    readiness_sources = prepared["backend_readiness_info"]["source_pins"]
    if readiness_sources.get("runtime_resource_helper", {}).get("sha256") != helper_pin["sha256"]:
        raise SnapshotError("backend readiness does not bind this resource helper")

    original_transform = core._transform
    original_generator = core._run_generator
    transforms: dict[str, FitSnapshotTransform] = {}
    snapshot_records = prepared["snapshot_records"]
    binding = prepared["runtime_binding"]
    resources = binding.get("resources", {})
    target_device = resources.get("torch_device")
    if not isinstance(target_device, str) or not target_device:
        raise SnapshotError("runtime binding lacks the explicit Torch device for snapshot transforms")

    def transform(samples: np.ndarray, method: Mapping[str, Any]):
        config = method.get("config", {})
        if not isinstance(config, Mapping) or config.get("method") != METHOD_KIND:
            return original_transform(samples, method)
        method_id = str(method.get("method_id", ""))
        snapshot = snapshot_records.get(method_id)
        if snapshot is None:
            raise SnapshotError(f"plan requested an unpinned snapshot method: {method_id}")
        if method_id not in transforms:
            profile = PROFILES.get(str(snapshot.get("training_family")))
            if profile is None:
                raise SnapshotError("snapshot training family is unknown")
            head_path = project_root / Path(profile["head_module"]["relative"])
            transforms[method_id] = FitSnapshotTransform(
                snapshot=snapshot,
                head_module_path=head_path,
                head_module_sha256=profile["head_module"]["sha256"],
                device=target_device,
            )
        return transforms[method_id].transform(samples)

    def generator(**kwargs):
        row = kwargs.get("row", {})
        model = kwargs.get("model", {})
        if isinstance(row, Mapping) and row.get("model_id") == W2L_MODEL_ID:
            return frontend._wav2lip_native_generator(core, **kwargs)
        generator_cfg = model.get("generator", {}) if isinstance(model, Mapping) else {}
        if isinstance(generator_cfg, Mapping) and generator_cfg.get("contract") == ADAPTER_CONTRACT:
            return frontend._adapter_generator(core, **kwargs)
        raise SnapshotError(f"non-W2L native video adapter is missing for {row.get('model_id')}")

    def resource_gate(plan: Mapping[str, Any], row: Mapping[str, Any], args: argparse.Namespace,
                      output_root: Path) -> dict[str, Any]:
        return _row_resource_gate(core, helper, prepared, plan, row, output_root)

    core._transform = transform
    core._run_generator = generator
    core._row_resource_gate = resource_gate
    return core.run(stagea_args)


if __name__ == "__main__":
    raise SystemExit(run())
