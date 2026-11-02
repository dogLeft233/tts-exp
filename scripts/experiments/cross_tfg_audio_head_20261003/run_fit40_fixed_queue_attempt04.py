#!/usr/bin/env python3
"""Run the fixed Fit40 queue from a verified historical baseline on Linux."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiments.cross_tfg_audio_head_20261003.fit40_fixed_queue_controller_attempt03 import (
    FixedFitQueueController,
    MODEL_ORDER,
    OUTPUT_ROOT_OVERRIDES,
    QueueError,
    FrozenBundle,
    _read_regular_pin,
    canonical,
    linux_process_start_ticks,
    load_pinned_bundles,
    make_detached_main_launcher,
    make_w2l_runner_launcher,
    run_ready_blocks,
    sha256,
)
from scripts.experiments.cross_tfg_audio_head_20261003.run_fit40_fixed_queue_attempt03 import (
    initialize_history,
    verify_only,
)


W2L_HELPER_REL = Path(
    "scripts/experiments/cross_tfg_audio_head_20261003/"
    "v3_dev180_newhost_latebind_packet_attempt02.py"
)
W2L_HELPER_BYTES = 62830
W2L_HELPER_SHA256 = "406f95e4a262fb62bb02bc67249f3e6a7bf211b803a014622e042329d772fdf7"
QUEUE_GATE_SCHEMA = "fit40_fixed_queue_live_capacity_gate_attempt04_v1"
QUEUE_ABSENT_SCHEMA = "fit40_fixed_queue_absent_output_leaf_attempt04_v1"
MIN_THRESHOLDS = {
    "min_gpu_free_mib": 0,
    "min_ram_available_bytes": 0,
    "min_shm_free_bytes": 0,
    "min_output_free_bytes": 0,
}


def _load_json_pin(repo_root: Path, pin: Any, label: str) -> dict[str, Any]:
    raw = _read_regular_pin(repo_root, pin, label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QueueError(f"{label} is not UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise QueueError(f"{label} must be a JSON object")
    return value


def _resource_helper_ref(repo_root: Path, bundle: FrozenBundle) -> tuple[Path, int, str]:
    """Resolve the exact helper named by the frozen plan/binding."""
    if bundle.spec.model_id == MODEL_ORDER[0]:
        execution = bundle.plan.get("execution", {})
        binding_ref = execution.get("w2l_runtime_binding") if isinstance(execution, Mapping) else None
        if not isinstance(binding_ref, Mapping) or not isinstance(binding_ref.get("path"), str):
            raise QueueError("W2L plan has no runtime-binding path for its live resource probe")
        binding_path = Path(binding_ref["path"])
        project_root = binding_path.parent.parent
        return project_root / W2L_HELPER_REL, W2L_HELPER_BYTES, W2L_HELPER_SHA256
    execution = bundle.plan.get("execution", {})
    ref = execution.get("resource_helper") if isinstance(execution, Mapping) else None
    if not isinstance(ref, Mapping):
        binding_raw = _read_regular_pin(repo_root, bundle.spec.runtime_binding, "runtime binding")
        binding = json.loads(binding_raw.decode("utf-8"))
        resources = binding.get("resources", {})
        ref = resources.get("resource_helper") if isinstance(resources, Mapping) else None
    if not isinstance(ref, Mapping):
        raise QueueError("frozen model plan/binding has no resource-helper pin")
    path, size, digest = ref.get("path"), ref.get("bytes"), ref.get("sha256")
    if (not isinstance(path, str) or not Path(path).is_absolute()
            or type(size) is not int or size <= 0
            or not isinstance(digest, str) or len(digest) != 64):
        raise QueueError("frozen resource-helper path/bytes/SHA pin is malformed")
    return Path(path), size, digest


def _load_resource_helper(repo_root: Path, bundle: FrozenBundle) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    binding_raw = _read_regular_pin(repo_root, bundle.spec.runtime_binding, "runtime binding")
    binding = json.loads(binding_raw.decode("utf-8"))
    helper_path, helper_bytes, helper_sha = _resource_helper_ref(repo_root, bundle)
    if helper_path.is_symlink() or not helper_path.is_file():
        raise QueueError("pinned live resource helper is missing or symlinked")
    helper_raw = helper_path.read_bytes()
    if len(helper_raw) != helper_bytes or sha256(helper_raw) != helper_sha:
        raise QueueError("live resource helper differs from its frozen raw pin")
    spec = importlib.util.spec_from_file_location(
        f"_fit40_queue_resource_{bundle.spec.model_id}", helper_path
    )
    if spec is None or spec.loader is None:
        raise QueueError("cannot import pinned live resource helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "resource_snapshot", None)):
        raise QueueError("pinned resource helper does not export resource_snapshot")
    return module, binding, {"path": str(helper_path), "bytes": len(helper_raw), "sha256": helper_sha}


def _required_free_bytes(model_id: str, resources: Mapping[str, Any]) -> int:
    basis = resources.get("capacity_basis")
    if not isinstance(basis, Mapping):
        raise QueueError("runtime binding has no pinned capacity basis")
    if model_id == MODEL_ORDER[0]:
        scenario = basis.get("capacity_scenario")
        value = scenario.get("required_free_bytes") if isinstance(scenario, Mapping) else None
    elif model_id == MODEL_ORDER[1]:
        value = basis.get("required_free_bytes_initial")
    elif model_id == MODEL_ORDER[2]:
        value = basis.get("required_with_reserve_bytes")
    else:
        raise QueueError("resource gate received a model outside the frozen allowlist")
    reserve = resources.get("output_reserve_bytes")
    minimum = resources.get("min_shm_free_bytes", 0)
    if (type(value) is not int or value <= 0 or type(reserve) is not int or reserve <= 0
            or type(minimum) is not int or minimum < 0):
        raise QueueError("runtime binding capacity estimate/reserve fields are malformed")
    return max(value, minimum)


def _resource_gate_record(model_id: str, block_start_index: int, plan_sha256: str,
                          binding_sha256: str, helper_pin: Mapping[str, Any],
                          resources: Mapping[str, Any], snapshot: Mapping[str, Any],
                          required_bytes: int) -> dict[str, Any]:
    gpu_uuid = resources.get("gpu_uuid")
    gpu_name = resources.get("gpu_name")
    min_gpu = resources.get("min_gpu_free_mib")
    min_ram = resources.get("min_effective_ram_available_bytes")
    mount_target = resources.get("output_mount_target")
    if (not isinstance(gpu_uuid, str) or not isinstance(gpu_name, str)
            or type(min_gpu) is not int or type(min_ram) is not int
            or not isinstance(mount_target, str)):
        raise QueueError("frozen runtime binding resource thresholds are malformed")
    reasons: list[str] = []
    gpus = snapshot.get("gpus")
    gpu = next((item for item in gpus if item.get("uuid") == gpu_uuid), None) if isinstance(gpus, list) else None
    if gpu is None or gpu.get("name") != gpu_name:
        reasons.append("bound GPU UUID/name is absent from the live inventory")
    elif type(gpu.get("memory_free_mib")) is not int or gpu["memory_free_mib"] < min_gpu:
        reasons.append("live GPU free memory is below the frozen model threshold")
    ram = snapshot.get("ram")
    if not isinstance(ram, Mapping) or type(ram.get("available_bytes")) is not int:
        reasons.append("live effective RAM snapshot is malformed")
    elif ram["available_bytes"] < min_ram:
        reasons.append("live effective RAM is below the frozen model threshold")
    shm = snapshot.get("shm")
    if not isinstance(shm, Mapping) or type(shm.get("available_bytes")) is not int:
        reasons.append("live /dev/shm snapshot is malformed")
    elif shm["available_bytes"] < required_bytes:
        reasons.append("live /dev/shm free bytes are below the frozen estimate plus reserve")
    disk = snapshot.get("output_disk")
    if not isinstance(disk, Mapping) or disk.get("target") != mount_target:
        reasons.append("live output mount differs from the frozen mount target")
    elif type(disk.get("available_bytes")) is not int or disk["available_bytes"] < required_bytes:
        reasons.append("live output free bytes are below the frozen estimate plus reserve")
    expected_cgroup = resources.get("cgroup")
    cgroup = ram.get("cgroup") if isinstance(ram, Mapping) else None
    if not isinstance(expected_cgroup, Mapping) or not isinstance(cgroup, Mapping):
        reasons.append("live or frozen cgroup identity is malformed")
    else:
        actual_identity = cgroup.get("identity")
        expected_identity = expected_cgroup.get("identity")
        if (cgroup.get("version") != expected_cgroup.get("version")
                or not isinstance(actual_identity, Mapping)
                or not isinstance(expected_identity, Mapping)
                or actual_identity.get("mount_target") != expected_identity.get("mount_target")
                or actual_identity.get("mount_root") != expected_identity.get("mount_root")):
            reasons.append("live cgroup version or stable mount identity differs from the frozen host")
        elif cgroup.get("finite_limit") is True:
            available = cgroup.get("available_bytes")
            if type(available) is not int or available < min_ram:
                reasons.append("live finite cgroup memory is below the frozen RAM threshold")
        elif cgroup.get("finite_limit") is not False or cgroup.get("available_bytes") is not None:
            reasons.append("live unlimited cgroup memory fields are malformed")
    body: dict[str, Any] = {
        "schema": QUEUE_GATE_SCHEMA,
        "status": "PASS" if not reasons else "FAILED_RESOURCE_GATE",
        "model_id": model_id,
        "block_start_index": block_start_index,
        "plan_raw_sha256": plan_sha256,
        "runtime_binding_raw_sha256": binding_sha256,
        "resource_helper": dict(helper_pin),
        "capacity_basis_sha256": sha256(canonical(resources["capacity_basis"])),
        "required_remaining_bytes": required_bytes,
        "observed_unix_time": time.time(),
        "minimum_gpu_free_mib": min_gpu,
        "minimum_effective_ram_bytes": min_ram,
        "output_mount_target": mount_target,
        "observed": {
            "host": snapshot.get("host"),
            "gpu": dict(gpu) if isinstance(gpu, Mapping) else None,
            "ram_available_bytes": ram.get("available_bytes") if isinstance(ram, Mapping) else None,
            "cgroup": dict(ram["cgroup"]) if isinstance(ram, Mapping) and isinstance(ram.get("cgroup"), Mapping) else None,
            "shm_available_bytes": shm.get("available_bytes") if isinstance(shm, Mapping) else None,
            "output_disk": dict(disk) if isinstance(disk, Mapping) else None,
            "snapshot_sha256": sha256(canonical(snapshot)),
        },
        "failure_reasons": reasons,
    }
    body["evidence_raw_sha256"] = sha256(canonical(body))
    return body


def build_live_resource_gate(repo_root: Path, bundle: FrozenBundle, block_start_index: int,
                             output_leaf: Path) -> dict[str, Any]:
    """Call the pinned host resource snapshot helper and apply frozen thresholds."""
    if block_start_index < 0 or block_start_index % 6:
        raise QueueError("live resource gate block start is invalid")
    binding_raw = _read_regular_pin(repo_root, bundle.spec.runtime_binding, "runtime binding")
    binding = json.loads(binding_raw.decode("utf-8"))
    resources = binding.get("resources")
    if not isinstance(resources, Mapping):
        raise QueueError("runtime binding has no resources object")
    helper, _binding_again, helper_pin = _load_resource_helper(repo_root, bundle)
    snapshot = helper.resource_snapshot(output_leaf, MIN_THRESHOLDS)
    return _resource_gate_record(
        bundle.spec.model_id, block_start_index, bundle.spec.plan.sha256,
        bundle.spec.runtime_binding.sha256, helper_pin, resources, snapshot,
        _required_free_bytes(bundle.spec.model_id, resources),
    )


def build_absent_leaf_proof(model_id: str, block_start_index: int,
                            plan_sha256: str, output_base: Path) -> dict[str, Any] | None:
    if model_id not in OUTPUT_ROOT_OVERRIDES or block_start_index < 0 or block_start_index % 6:
        raise QueueError("absent-leaf request is outside the frozen model/block allowlist")
    if str(output_base) != OUTPUT_ROOT_OVERRIDES[model_id]:
        raise QueueError("output namespace differs from the fixed model allowlist")
    if output_base.is_symlink() or not output_base.is_dir():
        return None
    output_leaf = output_base / f"block_{block_start_index // 6:03d}_attempt01"
    if output_leaf.exists() or output_leaf.is_symlink():
        return None
    body: dict[str, Any] = {
        "schema": QUEUE_ABSENT_SCHEMA,
        "status": "PASS",
        "model_id": model_id,
        "block_start_index": block_start_index,
        "plan_raw_sha256": plan_sha256,
        "output_leaf": str(output_leaf),
        "output_base": str(output_base),
        "exists": False,
        "observed_unix_time": time.time(),
    }
    body["evidence_raw_sha256"] = sha256(canonical(body))
    return body


def run_queue(repo_root: Path, historical_index: Path, expected_index_sha256: str,
              state_dir: Path, receipt_dir: Path, *, max_blocks: int | None = None) -> dict[str, Any]:
    """Run ready models serially until the fixed family is complete or paused."""
    bundles = load_pinned_bundles(repo_root)
    controller = FixedFitQueueController(
        bundles, state_dir,
        historical_prefix_index=historical_index,
        historical_prefix_index_sha256=expected_index_sha256,
    )
    # Refuse implicit history creation: initialize-history is a separate,
    # create-only operation that must be reviewed before any model launch.
    controller._load_historical_import()
    owner_pid = os.getpid()
    owner_ticks = linux_process_start_ticks(owner_pid)
    receipt_dir.mkdir(parents=True, exist_ok=False)
    launchers = {}
    for bundle in bundles:
        source_pin = bundle.spec.launch_sources[0]
        if bundle.spec.model_id == MODEL_ORDER[0]:
            launchers[bundle.spec.model_id] = make_w2l_runner_launcher(receipt_dir)
        else:
            source_path = repo_root / source_pin.path
            launchers[bundle.spec.model_id] = make_detached_main_launcher(
                source_path, source_pin.sha256, receipt_dir
            )
    bundle_by_model = {bundle.spec.model_id: bundle for bundle in bundles}
    output_base_by_model = {model: Path(path) for model, path in OUTPUT_ROOT_OVERRIDES.items()}
    live_evidence: list[dict[str, Any]] = []

    def gates(model_id: str, start: int) -> Mapping[str, Any] | None:
        bundle = bundle_by_model[model_id]
        output_leaf = output_base_by_model[model_id] / f"block_{start // 6:03d}_attempt01"
        try:
            record = build_live_resource_gate(repo_root, bundle, start, output_leaf)
        except Exception as exc:
            record = {
                "status": "FAILED_RESOURCE_GATE", "model_id": model_id,
                "block_start_index": start, "plan_raw_sha256": bundle.spec.plan.sha256,
                "evidence_raw_sha256": sha256(f"{type(exc).__name__}: {exc}".encode("utf-8")),
                "failure_reasons": [f"{type(exc).__name__}: {exc}"],
            }
        live_evidence.append(record)
        return record

    def absent(model_id: str, start: int) -> Mapping[str, Any] | None:
        return build_absent_leaf_proof(model_id, start,
                                       bundle_by_model[model_id].spec.plan.sha256,
                                       output_base_by_model[model_id])

    terminal_events = run_ready_blocks(
        controller,
        owner_pid=owner_pid,
        owner_start_ticks=owner_ticks,
        available_model_ids=lambda: MODEL_ORDER,
        live_resource_gate=gates,
        absent_output_leaf=absent,
        launchers=launchers,
        max_blocks=max_blocks,
    )
    state = controller.snapshot()
    status = ("PASS_QUEUE_RUN_TERMINAL" if terminal_events else
              "PASS_QUEUE_COMPLETE_NO_WORK" if all(
                  state["next_start_by_model"][model] >= 240
                  or model in state["paused_models"] for model in MODEL_ORDER
              ) else "PASS_QUEUE_NO_READY_BLOCK")
    return {
        "schema": "fit40_fixed_queue_run_attempt04_v1",
        "status": status,
        "owner": {"pid": owner_pid, "start_ticks": owner_ticks},
        "terminal_events": terminal_events,
        "live_gate_observations": live_evidence,
        "queue_state": state,
        "launches_models": bool(terminal_events),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("verify-only", "initialize-history", "run"), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--historical-index", type=Path, required=True)
    parser.add_argument("--expected-index-sha256", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path)
    parser.add_argument("--max-blocks", type=int)
    args = parser.parse_args()
    if args.mode == "verify-only":
        result = verify_only(args.repo_root, args.historical_index,
                             args.expected_index_sha256, args.state_dir)
    elif args.mode == "initialize-history":
        result = initialize_history(args.repo_root, args.historical_index,
                                    args.expected_index_sha256, args.state_dir)
    else:
        if args.receipt_dir is None:
            parser.error("--receipt-dir is required with --mode run")
        if args.max_blocks is not None and args.max_blocks <= 0:
            parser.error("--max-blocks must be positive when supplied")
        result = run_queue(args.repo_root, args.historical_index,
                           args.expected_index_sha256, args.state_dir,
                           args.receipt_dir, max_blocks=args.max_blocks)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
