#!/usr/bin/env python3
"""Freeze a six-method, three-backend Fit40 Stage-A score plan.

Inputs are deliberately late-bound and required: a frozen 240-row Fit plan
containing only identity/RMS on all three native models, an extracted four-
snapshot index, and an independently prepared backend-readiness receipt with
actual qualification, runtime-binding, and 720-row capacity pins. This tool
does not infer model readiness from the W2L-only post-projection proxy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from fit40_training_snapshots_attempt01 import (
    FIT_PAIRS,
    FIT_SPEAKERS,
    REQUIRED_MODEL_IDS,
    SnapshotError,
    canonical_bytes,
    sha256_bytes,
    sha256_file,
    verify_snapshot_index,
)


STAGEA_CORE_REL = "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt06.py"
STAGEA_CORE_BYTES = 79475
STAGEA_CORE_SHA256 = "291f8b5cd48d8ebc63aafd814dcd37450eecba949fab98283cf20a27c6335483"
FRONTEND_ADAPTER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_dev_attempt09.py"
FRONTEND_ADAPTER_BYTES = 23249
FRONTEND_ADAPTER_SHA256 = "04adf397a2edc45c6e3bc1e67d2265ea1c6ad2fcdac3df268b290938e5bf74df"
RUNTIME_HELPER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/v3_dev180_newhost_latebind_packet_attempt02.py"
RUNTIME_HELPER_BYTES = 62830
RUNTIME_HELPER_SHA256 = "406f95e4a262fb62bb02bc67249f3e6a7bf211b803a014622e042329d772fdf7"
FIT_RUNNER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/run_fit40_snapshot_stagea_attempt01.py"
FIT_PLAN_BUILDER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/build_fit40_snapshot_stagea_plan_attempt01.py"
BASE_METHOD_IDS = ("identity", "rms_fixed")
CAPACITY_STAGE = "fit40_snapshots720"


def pin_file(path: Path, *, expected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if path.is_symlink():
        raise SnapshotError(f"refusing symlink input: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise SnapshotError(f"expected a regular file: {resolved}")
    pin = {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}
    if expected is not None and (pin["bytes"] != expected.get("bytes") or pin["sha256"] != expected.get("sha256")):
        raise SnapshotError(f"raw source pin mismatch: {resolved}")
    return pin


def read_pinned_json(path: Path, expected_raw_sha256: str, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink():
        raise SnapshotError(f"refusing symlink {label}: {path}")
    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    if sha256_bytes(raw) != expected_raw_sha256.lower():
        raise SnapshotError(f"{label} raw SHA-256 mismatch")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise SnapshotError(f"{label} must be a JSON object")
    return value, raw


def _read_ref(ref: Any, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(ref, Mapping):
        raise SnapshotError(f"{label} raw file pin is required")
    actual = pin_file(Path(str(ref.get("path", ""))))
    if actual["bytes"] != ref.get("bytes") or actual["sha256"] != ref.get("sha256"):
        raise SnapshotError(f"{label} raw bytes differ from its declared pin")
    value, _ = read_pinned_json(Path(actual["path"]), actual["sha256"], label)
    return value, actual


def _verify_backend_readiness(readiness: Mapping[str, Any], readiness_raw: bytes,
                              expected_snapshot_index_sha256: str) -> dict[str, Any]:
    if (readiness.get("schema") != "cross_tfg_fit40_snapshot_backend_readiness_v1"
            or readiness.get("status") != "PASS"
            or readiness.get("split") != "fit"
            or readiness.get("model_ids") != list(REQUIRED_MODEL_IDS)
            or readiness.get("capacity_stage") != CAPACITY_STAGE
            or readiness.get("planned_rows") != 720):
        raise SnapshotError("backend readiness is not an exact three-model 720-row Fit qualification")
    if readiness.get("readiness_sha256") != sha256_bytes(canonical_bytes({k: v for k, v in readiness.items() if k != "readiness_sha256"})):
        raise SnapshotError("backend readiness canonical self-hash is invalid")
    if readiness.get("snapshot_set_index_raw_sha256") != expected_snapshot_index_sha256:
        raise SnapshotError("backend readiness binds a different snapshot index")
    models = readiness.get("model_qualifications")
    if not isinstance(models, list) or [item.get("model_id") for item in models if isinstance(item, dict)] != list(REQUIRED_MODEL_IDS):
        raise SnapshotError("backend readiness must pin qualifications in the exact three-model order")
    model_refs = []
    for item, model_id in zip(models, REQUIRED_MODEL_IDS):
        if item.get("qualification_status") != "PASS" or item.get("runtime_binding_status") != "PASS":
            raise SnapshotError(f"native model qualification/binding is not PASS: {model_id}")
        qualification, qualification_pin = _read_ref(item.get("qualification"), f"{model_id} qualification")
        binding, binding_pin = _read_ref(item.get("runtime_binding"), f"{model_id} runtime binding")
        if qualification.get("model_id") != model_id or binding.get("model_id") != model_id:
            raise SnapshotError(f"native qualification/binding model identity differs: {model_id}")
        if not str(qualification.get("status", "")).startswith("PASS"):
            raise SnapshotError(f"native qualification file is not a PASS artifact: {model_id}")
        if binding.get("status") != "PASS":
            raise SnapshotError(f"native runtime binding file is not PASS: {model_id}")
        model_refs.append({"model_id": model_id, "qualification": qualification_pin, "runtime_binding": binding_pin})

    expected_source_files = {
        "stagea_core": (STAGEA_CORE_REL, STAGEA_CORE_BYTES, STAGEA_CORE_SHA256),
        "cross_frontend_adapter": (FRONTEND_ADAPTER_REL, FRONTEND_ADAPTER_BYTES, FRONTEND_ADAPTER_SHA256),
        "runtime_resource_helper": (RUNTIME_HELPER_REL, RUNTIME_HELPER_BYTES, RUNTIME_HELPER_SHA256),
        "fit_snapshot_runner": (FIT_RUNNER_REL, None, None),
        "fit_snapshot_plan_builder": (FIT_PLAN_BUILDER_REL, None, None),
    }
    fixed_source_pins = {}
    for name, (relative, expected_bytes, expected_sha) in expected_source_files.items():
        ref = readiness.get("source_pins", {}).get(name)
        if not isinstance(ref, Mapping):
            raise SnapshotError(f"backend readiness lacks source pin {name}")
        source_path = Path(str(ref.get("path", "")))
        actual = pin_file(source_path)
        if actual["bytes"] != ref.get("bytes") or actual["sha256"] != ref.get("sha256"):
            raise SnapshotError(f"backend readiness source pin mismatch: {name}")
        if expected_sha is not None and (actual["bytes"] != expected_bytes or actual["sha256"] != expected_sha):
            raise SnapshotError(f"frozen Stage-A source changed: {name}")
        if source_path.as_posix().replace("\\", "/").endswith("/" + relative):
            fixed_source_pins[name] = actual
        elif name in ("stagea_core", "cross_frontend_adapter", "runtime_resource_helper"):
            raise SnapshotError(f"readiness source path is not the fixed source for {name}")

    capacity, capacity_pin = _read_ref(readiness.get("capacity_receipt"), "Fit snapshot capacity receipt")
    if (capacity.get("schema") != "cross_tfg_fit40_snapshot_capacity_receipt_v1"
            or capacity.get("status") != "PASS"
            or capacity.get("planned_rows") != 720
            or capacity.get("model_ids") != list(REQUIRED_MODEL_IDS)
            or capacity.get("snapshot_method_ids") != readiness.get("snapshot_method_ids")):
        raise SnapshotError("capacity receipt does not bind all 720 planned Fit rows")
    binding, binding_pin = _read_ref(readiness.get("fit_runtime_binding"), "Fit snapshot runtime binding")
    if (binding.get("schema") != "cross_tfg_fit40_snapshot_runtime_binding_v1"
            or binding.get("status") != "PASS"
            or binding.get("binding_sha256") != sha256_bytes(canonical_bytes({k: v for k, v in binding.items() if k != "binding_sha256"}))
            or binding.get("planned_rows") != 720
            or binding.get("model_ids") != list(REQUIRED_MODEL_IDS)
            or binding.get("snapshot_method_ids") != readiness.get("snapshot_method_ids")
            or binding.get("capacity_receipt_raw_sha256") != capacity_pin["sha256"]
            or binding.get("snapshot_set_index_raw_sha256") != expected_snapshot_index_sha256):
        raise SnapshotError("runtime binding does not self-bind the 720-row Fit snapshot plan")
    resources = binding.get("resources")
    basis = resources.get("capacity_basis") if isinstance(resources, dict) else None
    if not isinstance(basis, dict):
        raise SnapshotError("Fit runtime binding lacks its measured capacity basis")
    if (basis.get("status") != "MEASURED_PILOT_WITH_EXPLICIT_SCENARIO_AND_LIVE_GATES"
            or basis.get("media_bytes_estimated") is not True
            or type(basis.get("stage_output_required_bytes", {}).get(CAPACITY_STAGE)) is not int
            or basis.get("stage_output_required_bytes", {}).get(CAPACITY_STAGE, 0) <= 0):
        raise SnapshotError("Fit runtime binding lacks an explicit measured 720-row output scenario")
    size_by_backend = basis.get("per_backend_controls_row_output_bytes")
    if not isinstance(size_by_backend, dict) or any(type(size_by_backend.get(model_id)) is not int or size_by_backend[model_id] <= 0 for model_id in REQUIRED_MODEL_IDS):
        raise SnapshotError("Fit runtime binding lacks per-backend row output allowances")
    slack = basis.get("capacity_scenario", {}).get("explicit_slack_multiplier")
    if isinstance(slack, bool) or not isinstance(slack, (int, float)) or slack < 1:
        raise SnapshotError("Fit runtime binding lacks its declared capacity slack multiplier")
    for key in ("gpu_uuid", "output_mount_target", "torch_device"):
        if not isinstance(resources.get(key), str) or not resources[key]:
            raise SnapshotError(f"Fit runtime binding lacks {key}")
    for key in ("min_gpu_free_mib", "min_effective_ram_available_bytes", "min_shm_free_bytes", "output_reserve_bytes"):
        if type(resources.get(key)) is not int or resources[key] < 0:
            raise SnapshotError(f"Fit runtime binding has invalid {key}")
    if not isinstance(resources.get("cgroup"), dict):
        raise SnapshotError("Fit runtime binding lacks the cgroup identity")
    return {
        "model_qualifications": model_refs,
        "capacity_receipt": capacity_pin,
        "fit_runtime_binding": binding_pin,
        "source_pins": fixed_source_pins,
        "binding_payload": binding,
    }


def _validate_base_fit_plan(base: Mapping[str, Any], snapshot_index: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if base.get("schema_version") != "cross-tfg-audio-eval-v1" or base.get("status") != "FROZEN":
        raise SnapshotError("base plan must be a frozen Stage-A compatible plan")
    rows = base.get("rows")
    methods = base.get("methods")
    models = base.get("models")
    if not isinstance(rows, list) or len(rows) != 240:
        raise SnapshotError("base Fit plan must contain exactly 240 identity/RMS rows")
    if not isinstance(methods, list) or [item.get("method_id") for item in methods] != list(BASE_METHOD_IDS):
        raise SnapshotError("base Fit plan must contain only identity and rms_fixed methods in order")
    if not isinstance(models, list) or [item.get("model_id") for item in models] != list(REQUIRED_MODEL_IDS):
        raise SnapshotError("base Fit plan must contain the exact three native model IDs in order")
    if any(row.get("split") != "fit" for row in rows):
        raise SnapshotError("base plan contains a non-Fit row")
    if any(row.get("score_audio") != row.get("natural_pcm") for row in rows):
        raise SnapshotError("base Fit score audio must remain the exact original natural PCM")
    method_ids = [entry.get("method_id") for entry in snapshot_index.get("snapshots", [])]
    if len(method_ids) != 4 or len(set(method_ids)) != 4:
        raise SnapshotError("snapshot set does not provide four unique source-step methods")

    source_rows = snapshot_index.get("fit_roster")
    if not isinstance(source_rows, list) or len(source_rows) != 40:
        raise SnapshotError("snapshot index lacks the exact 40-row Fit roster")
    source_by_sample = {row["sample_id"]: row for row in source_rows}
    if len(source_by_sample) != 40:
        raise SnapshotError("snapshot Fit roster has duplicate sample IDs")
    if [source_by_sample[row["sample_id"]]["speaker_id"] for row in source_rows] == []:
        raise SnapshotError("snapshot Fit source roster is empty")

    seen: dict[tuple[str, str, str], dict[str, Any]] = {}
    contexts: dict[str, dict[str, Any]] = {}
    for row in rows:
        model_id = row.get("model_id")
        method_id = row.get("method_id")
        sample_id = row.get("sample_id")
        speaker_id = row.get("speaker_id")
        if model_id not in REQUIRED_MODEL_IDS or method_id not in BASE_METHOD_IDS:
            raise SnapshotError("base Fit plan contains an unexpected method or model")
        expected_source = source_by_sample.get(sample_id)
        if expected_source is None or expected_source.get("speaker_id") != speaker_id:
            raise SnapshotError("base plan sample/speaker does not match the pinned Fit roster")
        natural = row.get("natural_pcm", {})
        expected_pcm = expected_source.get("natural_pcm", {})
        if (natural.get("sample_count") != expected_pcm.get("sample_count")
                or natural.get("sample_rate_hz") != expected_pcm.get("sample_rate_hz")
                or natural.get("channels") != expected_pcm.get("channels")
                or natural.get("file_sha256") != expected_pcm.get("file_sha256")
                or natural.get("pcm_sha256") != expected_pcm.get("pcm_float32_sha256")):
            raise SnapshotError("base plan natural PCM differs from the pinned Fit source")
        if row.get("score_audio") != natural or row.get("seed") is None or row.get("face_id") is None:
            raise SnapshotError("base Fit row lacks original-clock score PCM, seed, or face identity")
        key = (sample_id, model_id, method_id)
        if key in seen:
            raise SnapshotError("base Fit plan contains duplicate context/method rows")
        seen[key] = row
        context_key = f"{sample_id}\0{model_id}"
        context = contexts.setdefault(context_key, {"sample_id": sample_id, "model_id": model_id,
                                                      "speaker_id": speaker_id, "seed": row["seed"],
                                                      "face_id": row["face_id"], "identity": None, "rms": None})
        if context["speaker_id"] != speaker_id or context["seed"] != row["seed"] or context["face_id"] != row["face_id"]:
            raise SnapshotError("base Fit methods/models do not share the same speaker, seed, and face")
        context["identity" if method_id == "identity" else "rms"] = row
    expected_keys = {(sample_id, model_id, method_id) for sample_id in source_by_sample for model_id in REQUIRED_MODEL_IDS for method_id in BASE_METHOD_IDS}
    if set(seen) != expected_keys or len(contexts) != 120:
        raise SnapshotError("base plan does not cover all 40 clips × 3 models × 2 baseline methods")
    for context in contexts.values():
        identity, rms = context["identity"], context["rms"]
        if identity is None or rms is None:
            raise SnapshotError("base Fit context is missing identity or RMS")
        for key in ("source_face", "source_video", "generation_settings", "natural_pcm", "score_audio", "seed", "face_id", "model_role", "audio_frontend"):
            if identity.get(key) != rms.get(key):
                raise SnapshotError(f"identity/RMS baseline context differs at {key}")
    seeds = {row.get("seed") for row in rows}
    if len(seeds) != 1:
        raise SnapshotError("snapshot plan requires one shared predeclared scoring seed")
    return rows, {"source_by_sample": source_by_sample, "contexts": contexts, "method_ids": method_ids}


def build_fit_snapshot_plan(*, base_plan_path: Path, expected_base_plan_sha256: str,
                            snapshot_index_path: Path, expected_snapshot_index_sha256: str,
                            backend_readiness_path: Path, expected_backend_readiness_sha256: str,
                            run_id: str, output_dir: Path) -> dict[str, Any]:
    base, base_raw = read_pinned_json(base_plan_path, expected_base_plan_sha256, "base Fit plan")
    base_payload = {k: v for k, v in base.items() if k != "plan_sha256"}
    if base.get("plan_sha256") != sha256_bytes(canonical_bytes(base_payload)):
        raise SnapshotError("base Fit plan canonical self-hash is invalid")
    snapshot_index, snapshot_records = verify_snapshot_index(snapshot_index_path, expected_snapshot_index_sha256)
    readiness, readiness_raw = read_pinned_json(backend_readiness_path, expected_backend_readiness_sha256, "backend readiness")
    if readiness.get("readiness_sha256") != sha256_bytes(canonical_bytes({k: v for k, v in readiness.items() if k != "readiness_sha256"})):
        raise SnapshotError("backend readiness canonical self-hash is invalid")
    readiness_info = _verify_backend_readiness(readiness, readiness_raw, sha256_bytes(snapshot_index_path.read_bytes()))
    base_rows, roster = _validate_base_fit_plan(base, snapshot_index)
    if readiness.get("snapshot_method_ids") != roster["method_ids"]:
        raise SnapshotError("backend readiness snapshot method order differs")
    if not run_id or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in run_id):
        raise SnapshotError("run_id must be explicit and path-safe")
    if output_dir.is_symlink():
        raise SnapshotError("plan output directory may not be a symlink")
    output = output_dir.resolve(strict=False)
    if output.exists():
        raise FileExistsError(f"plan output directory must be absent: {output}")

    base_methods = {item["method_id"]: item for item in base["methods"]}
    methods = [dict(base_methods[method_id]) for method_id in BASE_METHOD_IDS]
    snapshot_steps = []
    for snapshot_entry in snapshot_index["snapshots"]:
        method_id = snapshot_entry["method_id"]
        record = snapshot_records[method_id]
        config = {
            "method": "fit40_parameter_snapshot",
            "snapshot_index_path": str(snapshot_index_path.resolve(strict=True)),
            "snapshot_index_raw_sha256": sha256_bytes(snapshot_index_path.read_bytes()),
            "snapshot_id": record["snapshot_id"],
            "snapshot_record_path": str((snapshot_index_path.resolve(strict=True).parent / snapshot_entry["path"]).resolve(strict=True)),
            "snapshot_record_raw_sha256": snapshot_entry["sha256"],
            "training_family": record["training_family"],
            "source_optimizer_step": record["source_optimizer_step"],
        }
        method = {
            "method_id": method_id,
            "kind": "fit_parameter_snapshot",
            "config": config,
            "config_sha256": sha256_bytes(canonical_bytes(config)),
            "checkpoint_sha256": record["source_checkpoint"]["sha256"],
            "selection_status": "NOT_SELECTED",
        }
        methods.append(method)
        snapshot_steps.append(method_id)

    base_row_map = {(row["sample_id"], row["model_id"], row["method_id"]): row for row in base_rows}
    fit_rows: list[dict[str, Any]] = []
    for sample_id, source in roster["source_by_sample"].items():
        for model_id in REQUIRED_MODEL_IDS:
            identity = base_row_map[(sample_id, model_id, "identity")]
            for method in methods:
                method_id = method["method_id"]
                if method_id in BASE_METHOD_IDS:
                    row = dict(base_row_map[(sample_id, model_id, method_id)])
                else:
                    row = dict(identity)
                    row["method_id"] = method_id
                    row["method_config_sha256"] = method["config_sha256"]
                    row["checkpoint_sha256"] = method["checkpoint_sha256"]
                    row["row_id"] = "_".join((
                        "fit", str(row["speaker_id"]), str(row["sample_id"]), str(model_id),
                        str(row["face_id"]), str(row["seed"]), method_id,
                    ))
                row["score_audio"] = dict(row["natural_pcm"])
                fit_rows.append(row)
    if len(fit_rows) != 720 or len({row["row_id"] for row in fit_rows}) != 720:
        raise SnapshotError("constructed Fit snapshot plan is not exactly 720 unique rows")

    expected_methods = list(BASE_METHOD_IDS) + snapshot_steps
    if [method["method_id"] for method in methods] != expected_methods:
        raise SnapshotError("six-method order differs from the frozen plan contract")
    backend_pin = pin_file(backend_readiness_path, expected={"bytes": len(readiness_raw), "sha256": sha256_bytes(readiness_raw)})
    runtime_binding_pin = readiness_info["fit_runtime_binding"]
    capacity_pin = readiness_info["capacity_receipt"]
    stagea_core_pin = readiness_info["source_pins"]["stagea_core"]
    frontend_pin = readiness_info["source_pins"]["cross_frontend_adapter"]
    runner_path = Path(__file__).resolve().parent / "run_fit40_snapshot_stagea_attempt01.py"
    runner_pin = pin_file(runner_path)
    if readiness_info["source_pins"].get("fit_snapshot_runner", runner_pin) != runner_pin:
        raise SnapshotError("backend readiness does not pin this exact Fit snapshot runner")

    plan = dict(base)
    plan.update({
        "run_id": run_id,
        "status": "FROZEN",
        "models": list(base["models"]),
        "methods": methods,
        "rows": fit_rows,
        "generation_repeat_qc": [],
        "fit_speaker_ids": list(FIT_SPEAKERS),
        "dev_speaker_ids": [],
        "confirm_speaker_ids": [],
    })
    plan["analysis"] = {
        "contrasts": [],
        "summary_endpoints": [],
        "simultaneous_family_ids": [],
        "simultaneous_family_status": "NOT_DECLARED_SELECTION_NOT_PERFORMED",
        "selection_status": "NOT_SELECTED",
        "aggregation": {"future_consumer": "must use actual all-three-backend query masks and common source-clock query keys"},
    }
    plan["execution"] = dict(base.get("execution", {}))
    plan["execution"].update({
        "stage": "A",
        "scope": "fit-only; no development or confirmation audio opened",
        "planned_rows": 720,
        "primary_score": "native_roi_original_pcm16",
        "native_mp3_score": "diagnostic only",
        "fit_snapshot_runner_sha256": runner_pin["sha256"],
        "stagea_core_sha256": stagea_core_pin["sha256"],
        "cross_frontend_adapter_sha256": frontend_pin["sha256"],
        "snapshot_set_index": {
            "path": str(snapshot_index_path.resolve(strict=True)),
            "bytes": snapshot_index_path.stat().st_size,
            "raw_sha256": sha256_bytes(snapshot_index_path.read_bytes()),
            "index_sha256": snapshot_index["index_sha256"],
        },
        "backend_readiness": backend_pin,
        "fit_snapshot_runtime_binding": runtime_binding_pin,
        "fit_snapshot_capacity_receipt": capacity_pin,
        "base_fit_plan": {"path": str(base_plan_path.resolve(strict=True)), "bytes": len(base_raw),
                          "raw_sha256": sha256_bytes(base_raw)},
        "qualification_refs": readiness_info["model_qualifications"],
        "candidate_selection": "not performed; all planned rows remain in the denominator",
    })
    plan["snapshot_native_endpoint"] = snapshot_index["native_endpoint"]
    plan["plan_sha256"] = sha256_bytes(canonical_bytes({k: v for k, v in plan.items() if k != "plan_sha256"}))

    output.mkdir(parents=True, exist_ok=False)
    plan_raw = json.dumps(plan, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    rows_raw = b"".join(canonical_bytes(row) + b"\n" for row in fit_rows)
    with (output / "plan.json").open("xb") as handle:
        handle.write(plan_raw)
        handle.flush()
        os.fsync(handle.fileno())
    with (output / "planned_rows.jsonl").open("xb") as handle:
        handle.write(rows_raw)
        handle.flush()
        os.fsync(handle.fileno())
    build_receipt = {
        "schema": "cross_tfg_fit40_snapshot_stagea_plan_build_receipt_v1",
        "status": "FROZEN_NOT_DISPATCHED",
        "run_id": run_id,
        "plan": {"path": str((output / "plan.json").resolve()), "bytes": len(plan_raw), "raw_sha256": sha256_bytes(plan_raw), "plan_sha256": plan["plan_sha256"]},
        "planned_rows": {"path": str((output / "planned_rows.jsonl").resolve()), "bytes": len(rows_raw), "raw_sha256": sha256_bytes(rows_raw), "count": len(fit_rows)},
        "snapshot_set_index": plan["execution"]["snapshot_set_index"],
        "backend_readiness": backend_pin,
        "runtime_binding": runtime_binding_pin,
        "capacity_receipt": capacity_pin,
        "selection_status": "NOT_SELECTED",
    }
    receipt_raw = json.dumps(build_receipt, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with (output / "build_receipt.json").open("xb") as handle:
        handle.write(receipt_raw)
        handle.flush()
        os.fsync(handle.fileno())
    return {"status": build_receipt["status"], "output_dir": str(output),
            "plan_raw_sha256": sha256_bytes(plan_raw), "planned_rows_raw_sha256": sha256_bytes(rows_raw),
            "planned_rows": len(fit_rows), "plan_sha256": plan["plan_sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-fit-plan", type=Path, required=True)
    parser.add_argument("--expected-base-fit-plan-sha256", required=True)
    parser.add_argument("--snapshot-index", type=Path, required=True)
    parser.add_argument("--expected-snapshot-index-sha256", required=True)
    parser.add_argument("--backend-readiness", type=Path, required=True)
    parser.add_argument("--expected-backend-readiness-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_fit_snapshot_plan(
        base_plan_path=args.base_fit_plan,
        expected_base_plan_sha256=args.expected_base_fit_plan_sha256,
        snapshot_index_path=args.snapshot_index,
        expected_snapshot_index_sha256=args.expected_snapshot_index_sha256,
        backend_readiness_path=args.backend_readiness,
        expected_backend_readiness_sha256=args.expected_backend_readiness_sha256,
        run_id=args.run_id,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
