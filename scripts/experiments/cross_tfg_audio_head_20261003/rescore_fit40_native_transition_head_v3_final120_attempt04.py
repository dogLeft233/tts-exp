"""No-gradient final-step rescore for the frozen fit40 V3 transition head.

This runner is source-only until a separately reviewed packet late-binds a
completed fit directory and the actual fit40 source/geometry inputs. It creates
no V3 optimizer and performs no V3 updates. The qualified V2 loader constructs
its legacy optimizer as part of its API; the rescorer discards it without a
step. It reads no development/confirmation data and selects no checkpoint. The
training runner's per-step curves are pre-update observations; this runner
produces a separate common-step-120 score table.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA = "cross_tfg_fit40_native_transition_head_v3_final120_rescore_attempt04"
SOURCE_MANIFEST_SHA256 = "0894f33df8504f9db7f3074ea901bcd987de23d533c029431b9150325e0cd020"
PARENT_PLAN_SHA256 = "e491d0bb210fe78647f4845543a88c629d37cecc09a0823c287560fbb1e71a06"
PARENT_PLAN_PAYLOAD_SHA256 = "7200d93801ca0749b472937645d1e955e5b39c730f6a2d1a5e24d3f380f6dce9"
HEAD_CONFIG_SHA256 = "0abfbb9797a84254e6d0a6e72266862a079e5066f2fec394e0796b0c2664d2f5"
TRAIN_RUNNER_PATH = "scripts/experiments/cross_tfg_audio_head_20261003/run_fit40_native_transition_head_v3_attempt04.py"
TRAIN_RUNNER_SHA256 = "66e3a59e088a46a9681ccb07ce18111b9153c12eb59134edd7c8dd38e57dc1aa"
V2_RUNTIME_SHA256 = "d086c61cdb993bffa5f9ff7a55bf4b66b1b9127028f8b80eb3b5f1be74464d7b"
V4_RUNTIME_SHA256 = "dd751132bd3d8dec5abce288c2f765df78d9b7ef9150cdfb0ccff52ba72ee458"
V3_HEAD_SHA256 = "77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87"
ADAPTER_SHA256 = "4f4f6f496bd15e073268694673b8df5678404e672aafa625ff9a6abc135248aa"
AUDIO_IO_SHA256 = "7b951990343cf5d68f3f6046e77cd5c40ee736777d196de3723418bca5965004"
BASELINES_SHA256 = "7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2"
PRODUCER_SHA256 = "0ecfd1f3be25f6c56d76fa6fecd1ab5dd9b1868e03b5394f355939e74dc14df7"
ZERO_GATE_ATOL = 1e-5
TARGET_RMS = 0.0376838172675746
SAMPLES_PER_40MS_FRAME = 640
FIT_SEED = 20261003
OBJECTIVE_WEIGHTS = {
    "relative_Cbest_improvement": 1.0,
    "far_lag_contrast_improvement": 0.25,
    "D0_nonworsening_guard": 2.0,
    "Canchor_nonworsening_guard": 1.0,
    "visual_embedding_log_norm_ratio_squared": 0.1,
    "waveform_relative_mse_to_rms_reference": 0.02,
    "mask_total_variation": 0.002,
}


class RescoreError(RuntimeError):
    pass


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def canonical_sha(value: Any) -> str:
    return sha_bytes(canonical_bytes(value))


def read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink():
        raise RescoreError(f"refusing a symlink JSON input: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RescoreError(f"expected a regular JSON file: {resolved}")
    raw = resolved.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RescoreError(f"expected a JSON object: {resolved}")
    return value, raw


def pin(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise RescoreError(f"refusing a symlink artifact: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RescoreError(f"expected a regular artifact: {resolved}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha_file(resolved)}


def verify_pin_record(record: Any, *, base: Path, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
        raise RescoreError(f"malformed {label} file reference")
    path = Path(str(record["path"]))
    if path.is_symlink():
        raise RescoreError(f"{label} path is a symlink")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise RescoreError(f"{label} escapes its completed training output root") from exc
    actual = pin(resolved)
    if actual["bytes"] != int(record["bytes"]) or actual["sha256"] != str(record["sha256"]).lower():
        raise RescoreError(f"{label} raw file pin mismatch")
    return resolved


def atomic_json(path: Path, payload: dict[str, Any]) -> bytes:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        temp.unlink(missing_ok=True)
        raise FileExistsError(f"refusing to replace an existing output: {path}")
    os.replace(temp, path)
    return raw


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
    with path.open("ab") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def write_status(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    with temp.open("wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_jsonl(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise RescoreError(f"blank line in JSONL ledger {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RescoreError(f"non-object JSONL record {path}:{line_number}")
            rows.append(value)
    if len(rows) != expected_rows:
        raise RescoreError(f"{path.name} has {len(rows)} rows; expected {expected_rows}")
    return rows


def validate_training_bundle(training_root: Path) -> dict[str, Any]:
    root = training_root.resolve(strict=True)
    if training_root.is_symlink() or not root.is_dir():
        raise RescoreError("training output root must be an existing non-symlink directory")
    if (root / "selection_record.json").exists():
        raise RescoreError("post-fit rescore refuses a previously selected checkpoint")
    completion_path = root / "completion.json"
    completion, completion_raw = read_json(completion_path)
    if (
        completion.get("schema") != "cross_tfg_fit40_native_transition_head_v3_training_runner_attempt02"
        or completion.get("status") != "COMPLETE_FIT40_V3_120_STEPS"
        or int(completion.get("completed_optimizer_steps", -1)) != 120
        or int(completion.get("fit_rows", -1)) != 40
        or int(completion.get("fit_speakers", -1)) != 20
        or int(completion.get("zero_gate_rows_passed", -1)) != 40
        or completion.get("selection_record_created") is not False
        or completion.get("development_audio_opened") is not False
        or completion.get("confirmation_audio_opened") is not False
    ):
        raise RescoreError("training completion is not the exact unselected fit40/120-step terminal")
    training_record_path = verify_pin_record(completion.get("training_record"), base=root, label="training_record")
    checkpoint_path = verify_pin_record(completion.get("checkpoint"), base=root, label="checkpoint")
    runtime_path = verify_pin_record(completion.get("runtime_source_pins"), base=root, label="runtime_source_pins")
    baseline_arrays_path = verify_pin_record(completion.get("baseline_arrays"), base=root, label="baseline_arrays")
    zero_arrays_path = verify_pin_record(completion.get("zero_gate_arrays"), base=root, label="zero_gate_arrays")
    train_arrays_path = verify_pin_record(completion.get("training_score_arrays"), base=root, label="training_score_arrays")
    steps_path = verify_pin_record(completion.get("training_steps"), base=root, label="training_steps")
    status, _ = read_json(root / "status.json")
    if status.get("schema") != completion["schema"] or status.get("status") != completion["status"] or status.get("stage") != "complete":
        raise RescoreError("terminal status.json does not agree with completion.json")

    training_record, training_record_raw = read_json(training_record_path)
    checkpoint, checkpoint_raw = read_json(checkpoint_path)
    runtime_pins, runtime_raw = read_json(runtime_path)
    if str(runtime_pins.get("runner", {}).get("sha256", "")).lower() != TRAIN_RUNNER_SHA256:
        raise RescoreError("training runtime pins do not bind the reviewed fit runner source")
    if sha_bytes(training_record_raw) != str(completion.get("training_record_raw_sha256", "")).lower():
        raise RescoreError("training-record raw digest differs from completion")
    if (
        training_record.get("schema") != "cross_tfg_v3_candidate_training_record_v1"
        or training_record.get("status") != "COMPLETE"
        or int(training_record.get("completed_optimizer_steps", -1)) != 120
        or int(training_record.get("planned_optimizer_steps", -1)) != 120
        or int(training_record.get("fit_row_count", -1)) != 40
        or int(training_record.get("fit_speaker_count", -1)) != 20
        or training_record.get("zero_head_gate", {}).get("status") != "PASS"
        or int(training_record.get("zero_head_gate", {}).get("row_count", -1)) != 40
        or training_record.get("development_audio_opened") is not False
        or training_record.get("confirmation_audio_opened") is not False
        or training_record.get("selection_record_created") is not False
        or training_record.get("selection_status") != "NOT_SELECTED"
        or int(training_record.get("pass_count", -1)) != 3
        or int(training_record.get("rows_per_pass", -1)) != 40
        or int(training_record.get("fit_seed", -1)) != FIT_SEED
        or training_record.get("optimizer") != {"class": "Adam", "learning_rate": 0.1, "betas": [0.9, 0.999], "eps": 1e-8}
        or training_record.get("objective", {}).get("weights") != OBJECTIVE_WEIGHTS
        or training_record.get("zero_head_gate", {}).get("same_cuda_device_as_rms_baseline") is not True
    ):
        raise RescoreError("training_record does not certify an unselected completed fit-only checkpoint")
    if checkpoint.get("schema") != "cross_tfg_v3_parameter_checkpoint_v1" or checkpoint.get("selection_status") != "UNSELECTED_FIT_CHECKPOINT_SELECTION_RECORD_NOT_CREATED":
        raise RescoreError("checkpoint schema/selection state differs from the frozen V3 contract")
    if (
        checkpoint.get("head_schema") != "cross_tfg_shared_recipient_transition_head_v3"
        or str(checkpoint.get("implementation", {}).get("sha256", "")).lower() != V3_HEAD_SHA256
        or str(training_record.get("head_implementation_sha256", "")).lower() != V3_HEAD_SHA256
    ):
        raise RescoreError("training/checkpoint head implementation pin differs from frozen V3 bytes")
    if sha_bytes(runtime_raw) != checkpoint.get("training_artifact_bindings", {}).get("runtime_source_pins_sha256"):
        raise RescoreError("checkpoint does not bind the actual runtime source pins")
    if str(training_record.get("checkpoint_sha256", "")).lower() != sha_bytes(checkpoint_raw):
        raise RescoreError("training record does not bind the actual final checkpoint bytes")
    bindings = checkpoint.get("training_artifact_bindings", {})
    expected_artifact_shas = {
        "training_steps_jsonl_sha256": sha_file(steps_path),
        "training_score_arrays_sha256": sha_file(train_arrays_path),
        "fit_source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "parent_plan_raw_sha256": PARENT_PLAN_SHA256,
    }
    for key, expected in expected_artifact_shas.items():
        if str(bindings.get(key, "")).lower() != expected:
            raise RescoreError(f"checkpoint binding mismatch for {key}")
    record_artifact_shas = {
        "training_steps_jsonl_sha256": sha_file(steps_path),
        "training_score_arrays_sha256": sha_file(train_arrays_path),
        "runtime_source_pins_sha256": sha_bytes(runtime_raw),
    }
    for key, expected in record_artifact_shas.items():
        if str(training_record.get(key, "")).lower() != expected:
            raise RescoreError(f"training record artifact digest mismatch for {key}")
    if int(bindings.get("completed_optimizer_steps", -1)) != 120:
        raise RescoreError("checkpoint optimizer-step binding is not 120")
    theta = checkpoint.get("parameter", {}).get("values")
    if checkpoint.get("parameter", {}).get("name") != "band_contrasts_db" or checkpoint.get("parameter", {}).get("shape") != [3, 3] or checkpoint.get("parameter", {}).get("dtype") != "float32":
        raise RescoreError("checkpoint does not contain the exact 3x3 float32 V3 parameter")
    if not isinstance(theta, list) or len(theta) != 3 or any(not isinstance(row, list) or len(row) != 3 for row in theta):
        raise RescoreError("checkpoint V3 parameter matrix is malformed")
    flat_theta = [float(value) for row in theta for value in row]
    if not all(math.isfinite(value) for value in flat_theta):
        raise RescoreError("checkpoint V3 parameters contain non-finite values")
    if checkpoint.get("constructor_sha256") != training_record.get("constructor_sha256"):
        raise RescoreError("checkpoint and training record constructor digests disagree")
    if checkpoint.get("training_artifact_bindings", {}).get("native_geometry_manifest_sha256") != training_record.get("native_geometry_manifest_sha256"):
        raise RescoreError("checkpoint and training record geometry digests disagree")
    if checkpoint.get("training_artifact_bindings", {}).get("fit_source_manifest_sha256") != training_record.get("fit_source_manifest_sha256"):
        raise RescoreError("checkpoint and training record source digests disagree")
    records = load_jsonl(steps_path, 120)
    if [int(row.get("step", -1)) for row in records] != list(range(1, 121)):
        raise RescoreError("training step ledger is not exactly ordered 1..120")
    step_theta = records[-1].get("theta_after_update")
    if step_theta != theta:
        raise RescoreError("checkpoint theta is not the recorded post-update theta after step 120")
    if records[-1].get("candidate_waveform_float32_sha256") is None:
        raise RescoreError("step 120 lacks its pre-update waveform identity")

    return {
        "root": root,
        "completion": completion,
        "completion_raw_sha256": sha_bytes(completion_raw),
        "training_record": training_record,
        "training_record_raw_sha256": sha_bytes(training_record_raw),
        "checkpoint": checkpoint,
        "checkpoint_raw_sha256": sha_bytes(checkpoint_raw),
        "runtime_pins": runtime_pins,
        "runtime_raw_sha256": sha_bytes(runtime_raw),
        "baseline_arrays_path": baseline_arrays_path,
        "zero_arrays_path": zero_arrays_path,
        "train_arrays_path": train_arrays_path,
        "steps_path": steps_path,
        "step_records": records,
        "theta": theta,
        "artifacts": {
            "completion": pin(completion_path),
            "training_record": pin(training_record_path),
            "checkpoint": pin(checkpoint_path),
            "runtime_source_pins": pin(runtime_path),
            "baseline_arrays": pin(baseline_arrays_path),
            "zero_gate_arrays": pin(zero_arrays_path),
            "training_score_arrays": pin(train_arrays_path),
            "training_steps": pin(steps_path),
        },
    }


def validate_input_bindings(args, fit):
    parent_plan, parent_raw = read_json(args.parent_plan)
    source_manifest, source_raw = read_json(args.source_manifest)
    geometry, geometry_raw = read_json(args.geometry_manifest)
    head_config, head_config_raw = read_json(args.head_config)
    if sha_bytes(parent_raw) != PARENT_PLAN_SHA256 or sha_bytes(source_raw) != SOURCE_MANIFEST_SHA256 or sha_bytes(head_config_raw) != HEAD_CONFIG_SHA256:
        raise RescoreError("parent plan, fit roster or V3 config raw source pin mismatch")
    if sha_bytes(geometry_raw) != args.expected_geometry_manifest_sha256.lower():
        raise RescoreError("actual geometry manifest differs from its late-bound raw SHA")
    plan_payload = {key: value for key, value in parent_plan.items() if key != "plan_payload_sha256"}
    if parent_plan.get("plan_payload_sha256") != PARENT_PLAN_PAYLOAD_SHA256 or canonical_sha(plan_payload) != PARENT_PLAN_PAYLOAD_SHA256:
        raise RescoreError("exact parent-plan payload digest mismatch")
    runtime_pins = fit["runtime_pins"]
    expected_refs = {
        "parent_plan": (PARENT_PLAN_SHA256, PARENT_PLAN_PAYLOAD_SHA256),
        "fit_source_manifest": (SOURCE_MANIFEST_SHA256, None),
        "native_geometry_manifest": (args.expected_geometry_manifest_sha256.lower(), None),
        "head_config": (HEAD_CONFIG_SHA256, None),
    }
    for field, (raw_sha, payload_sha) in expected_refs.items():
        record = runtime_pins.get(field)
        if not isinstance(record, dict) or str(record.get("sha256", "")).lower() != raw_sha:
            raise RescoreError(f"training runtime pins do not bind {field}")
        if payload_sha is not None and str(record.get("payload_sha256", "")).lower() != payload_sha:
            raise RescoreError(f"training runtime pins do not bind {field} payload")
    if geometry.get("schema") != "cross_tfg_native_full512_fit_geometry_manifest_v1" or geometry.get("status") != "COMPLETE_FIT_ONLY_ENGINEERING_RMS_GEOMETRY":
        raise RescoreError("fit geometry is not the completed fit-only RMS manifest")
    geometry_payload = {key: value for key, value in geometry.items() if key != "payload_sha256"}
    if geometry.get("payload_sha256") != canonical_sha(geometry_payload):
        raise RescoreError("native geometry payload digest mismatch")
    source_pin = geometry.get("source_pins", {}).get("source_manifest", {})
    if str(source_pin.get("sha256", "")).lower() != SOURCE_MANIFEST_SHA256:
        raise RescoreError("geometry does not bind the exact fit40 source manifest")
    parent_pin = geometry.get("source_pins", {}).get("v2_runtime_plan", {})
    if str(parent_pin.get("sha256", "")).lower() != PARENT_PLAN_SHA256:
        raise RescoreError("geometry does not bind the exact V2 parent plan")
    checkpoint_geometry_sha = fit["checkpoint"].get("training_artifact_bindings", {}).get("native_geometry_manifest_sha256")
    if str(checkpoint_geometry_sha).lower() != args.expected_geometry_manifest_sha256.lower():
        raise RescoreError("actual geometry manifest differs from the trained checkpoint binding")
    training_geometry_sha = fit["training_record"].get("native_geometry_manifest_sha256")
    if str(training_geometry_sha).lower() != args.expected_geometry_manifest_sha256.lower():
        raise RescoreError("actual geometry manifest differs from the training record binding")
    if (
        geometry.get("fit_rows") != 40
        or geometry.get("method_id") != "rms_fixed"
        or int(geometry.get("source_clock_fps_num", -1)) != 25
        or int(geometry.get("source_clock_fps_den", -1)) != 1
        or geometry.get("audio_clock_policy") != "unchanged_original_natural_pcm_mfcc_queries"
        or geometry.get("baseline_control", {}).get("peak_ceiling") is not None
        or float(geometry.get("baseline_control", {}).get("target_rms", -1.0)) != TARGET_RMS
    ):
        raise RescoreError("fit geometry changed the frozen RMS/native clock contract")
    constructor = head_config.get("head_constructor")
    if not isinstance(constructor, dict) or canonical_sha(constructor) != fit["checkpoint"].get("constructor_sha256"):
        raise RescoreError("final checkpoint does not bind the exact frozen constructor config")
    if constructor.get("target_rms") != TARGET_RMS or constructor.get("peak_ceiling") is not None:
        raise RescoreError("final rescore must retain unbounded fixed-RMS construction")
    return parent_plan, source_manifest, geometry, head_config, parent_raw, source_raw, geometry_raw, head_config_raw


def tensor_np(tensor, torch, *, dtype=None):
    target_dtype = torch.float32 if dtype is None else dtype
    return tensor.detach().to(device="cpu", dtype=target_dtype).contiguous().numpy()


def finite_float(value, torch) -> float:
    result = float(value.detach().item() if hasattr(value, "detach") else value)
    if not math.isfinite(result):
        raise RescoreError("non-finite final rescore diagnostic")
    return result


def make_40ms_frame_record(power_arrays, valid_array, row_index: int, frame_count: int) -> dict[str, Any]:
    names = ("source_original", "candidate_before_rms", "canonical_rms_baseline", "final_output")
    summaries = {}
    valid_indices = valid_array[row_index, :frame_count]
    for column, name in enumerate(names):
        values = power_arrays[row_index, column, :frame_count]
        selected = values[valid_indices]
        if selected.size:
            summaries[name] = {
                "valid_frame_count": int(selected.size),
                "mean_power": float(selected.mean(dtype=__import__("numpy").float64)),
                "median_power": float(__import__("numpy").median(selected.astype("float64"))),
                "mean_db_relative_to_1": float(10.0 * math.log10(max(float(selected.mean(dtype=__import__("numpy").float64)), 1e-30))),
            }
        else:
            summaries[name] = {"valid_frame_count": 0, "mean_power": None, "median_power": None, "mean_db_relative_to_1": None}
    ratio_valid = valid_array[row_index, :frame_count]
    ratios = power_arrays[row_index, 4:6, :frame_count]
    for col, label in ((0, "candidate_before_rms_to_source"), (1, "final_output_to_source")):
        selected = ratios[col, ratio_valid]
        summaries[label] = {
            "valid_frame_count": int(selected.size),
            "mean_ratio": float(selected.mean(dtype="float64")) if selected.size else None,
            "median_ratio": float(__import__("numpy").median(selected.astype("float64"))) if selected.size else None,
        }
    return summaries


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output-dir", type=Path, required=True)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--geometry-manifest", type=Path, required=True)
    parser.add_argument("--expected-geometry-manifest-sha256", required=True)
    parser.add_argument("--head-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", required=True, help="explicit CUDA device; must match the frozen training GPU UUID")
    return parser.parse_args(argv)


def run(args) -> dict[str, Any]:
    output_dir = args.output_dir
    if output_dir.is_symlink():
        raise RescoreError("rescore output may not be a symlink")
    output = output_dir.resolve(strict=False)
    allowed = Path("/dev/shm/cross_tfg_audio_head_20261003").resolve(strict=False)
    try:
        output.relative_to(allowed)
    except ValueError as exc:
        raise RescoreError("rescore output must be inside the dedicated experiment SHM namespace") from exc
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise RescoreError(f"rescore output directory must be absent or empty: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    status_path = output / "status.json"
    status = {
        "schema": SCHEMA,
        "status": "RUNNING_TERMINAL_INPUT_VALIDATION",
        "stage": "terminal_input_validation",
        "started_unix_seconds": time.time(),
        "completed_rows": 0,
        "planned_rows": 40,
        "v3_optimizer_created": False,
        "optimizer_updates": 0,
        "v2_loader_optimizer_constructed_then_discarded": False,
        "development_audio_opened": False,
        "confirmation_audio_opened": False,
        "selection_record_created": False,
    }
    write_status(status_path, status)
    active_stage = "terminal_input_validation"
    completed_rows = 0
    v2_optimizer_discarded = False
    try:
        fit = validate_training_bundle(args.training_output_dir)
        inputs = validate_input_bindings(args, fit)
        parent_plan, source_manifest, geometry_manifest, head_config, parent_raw, source_raw, geometry_raw, head_config_raw = inputs
        training_root = fit["root"]
        geometry_rows = geometry_manifest.get("rows")
        if not isinstance(geometry_rows, list) or len(geometry_rows) != 40:
            raise RescoreError("geometry manifest does not contain all forty fit rows")
        geometry_by_id = {str(row.get("row_id")): row for row in geometry_rows if isinstance(row, dict)}
        if len(geometry_by_id) != 40:
            raise RescoreError("geometry row IDs are not unique")

        runner_path = REPO_ROOT / TRAIN_RUNNER_PATH
        if sha_file(runner_path) != TRAIN_RUNNER_SHA256:
            raise RescoreError("frozen fit runner source differs from the reviewed V3 training source")
        train_runner = importlib.import_module("scripts.experiments.cross_tfg_audio_head_20261003.run_fit40_native_transition_head_v3_attempt04")
        if Path(train_runner.__file__).resolve() != runner_path.resolve():
            raise RescoreError("imported fit runner path differs from the reviewed source path")
        prepared = train_runner._build_fit_rows(source_manifest, args.source_root, geometry_by_id)
        row_ids = [str(item["row"]["row_id"]) for item in prepared]
        expected_step_rows = row_ids * 3
        if len(row_ids) != 40 or [str(record.get("row_id")) for record in fit["step_records"]] != expected_step_rows:
            raise RescoreError("training ledger does not contain the exact three ordered passes of the fit40 roster")
        if fit["training_record"].get("zero_head_gate", {}).get("passed_rows") != row_ids:
            raise RescoreError("training record zero-head gate does not pass the exact forty-row source roster")
        if [int(record.get("pass_index_zero_based", -1)) for record in fit["step_records"]] != [p for p in range(3) for _ in row_ids]:
            raise RescoreError("training ledger pass indices are not three complete source-order passes")

        import numpy as np

        with np.load(fit["baseline_arrays_path"], allow_pickle=False) as archive:
            baseline_ids = archive["row_ids"].astype(str).tolist()
            baseline_curves = archive["rms_curves_float32"].astype("<f4", copy=True)
            baseline_visuals = archive["rms_visual_embeddings_float32"].astype("<f4", copy=True)
            baseline_queries = archive["query_frame_indices_int32"].astype("<i4", copy=True)
        if baseline_ids != row_ids or baseline_curves.shape != (40, 31) or baseline_visuals.shape != (40, 27, 1024) or baseline_queries.shape != (40, 27):
            raise RescoreError("saved RMS baseline arrays do not exactly match the forty rows and native query support")
        if not np.isfinite(baseline_curves).all() or not np.isfinite(baseline_visuals).all():
            raise RescoreError("saved baseline curves or visual embeddings contain non-finite values")
        baseline_records = load_jsonl(training_root / "rms_baselines.jsonl", 40)
        if [str(record.get("row_id")) for record in baseline_records] != row_ids:
            raise RescoreError("RMS baseline JSONL is not in the exact fit-row order")
        for i, record in enumerate(baseline_records):
            if not np.array_equal(np.asarray(record.get("curve31_float32"), dtype="<f4"), baseline_curves[i]):
                raise RescoreError(f"saved RMS curve JSONL differs from its array for {row_ids[i]}")

        with np.load(fit["zero_arrays_path"], allow_pickle=False) as archive:
            zero_ids = archive["row_ids"].astype(str).tolist()
            zero_curves = archive["zero_head_curves_float32"].astype("<f4", copy=True)
            zero_visuals = archive["zero_head_visual_embeddings_float32"].astype("<f4", copy=True)
            zero_rms_curves = archive["rms_curves_float32"].astype("<f4", copy=True)
            zero_rms_visuals = archive["rms_visual_embeddings_float32"].astype("<f4", copy=True)
        if (
            zero_ids != row_ids
            or zero_curves.shape != (40, 31)
            or zero_visuals.shape != (40, 27, 1024)
            or zero_rms_curves.shape != (40, 31)
            or zero_rms_visuals.shape != (40, 27, 1024)
            or not np.isfinite(zero_curves).all()
            or not np.isfinite(zero_visuals).all()
            or not np.array_equal(zero_rms_curves, baseline_curves)
            or not np.array_equal(zero_rms_visuals, baseline_visuals)
        ):
            raise RescoreError("saved RMS curve/embedding copies differ from the pinned training baseline arrays")
        zero_curve_max_abs_vs_rms = float(np.max(np.abs(zero_curves - zero_rms_curves)))
        zero_visual_max_abs_vs_rms = float(np.max(np.abs(zero_visuals - zero_rms_visuals)))
        if zero_curve_max_abs_vs_rms > ZERO_GATE_ATOL or zero_visual_max_abs_vs_rms > ZERO_GATE_ATOL:
            raise RescoreError("saved same-device zero-head curves/embeddings exceed the frozen 1e-5 baseline gate")

        with np.load(fit["train_arrays_path"], allow_pickle=False) as archive:
            train_curve_ids = archive["row_ids"].astype(str).tolist()
            train_steps = archive["step_indices_int32"].astype("<i4", copy=True)
            train_curves = archive["candidate_curves_float32"]
            train_visual = archive["candidate_visual_embeddings_float32"]
        if train_curve_ids != expected_step_rows or not np.array_equal(train_steps, np.arange(1, 121, dtype="<i4")) or train_curves.shape != (120, 31) or train_visual.shape != (120, 27, 1024) or not np.isfinite(train_curves).all() or not np.isfinite(train_visual).all():
            raise RescoreError("training score arrays do not match the ordered 120-step training ledger")

        # Verify the immutable runtime/config source pins before importing the model stack.
        module_expected = {
            "v2_runtime": V2_RUNTIME_SHA256,
            "v4_runtime": V4_RUNTIME_SHA256,
            "v3_head": V3_HEAD_SHA256,
            "v3_v4_adapter": ADAPTER_SHA256,
            "audio_io": AUDIO_IO_SHA256,
            "baselines": BASELINES_SHA256,
            "fit40_geometry_producer": PRODUCER_SHA256,
        }
        recorded_training_modules = fit["runtime_pins"].get("training_modules", {})
        for key, expected_hash in module_expected.items():
            record = recorded_training_modules.get(key)
            if not isinstance(record, dict) or str(record.get("sha256", "")).lower() != expected_hash:
                raise RescoreError(f"completed training runtime does not pin the reviewed module: {key}")
        local_sources = {
            "v2_runtime": REPO_ROOT / "scripts/audio_enhancement/wav2lip_fullcontext_fit_runtime_v2_attempt02.py",
            "v4_runtime": REPO_ROOT / "scripts/audio_enhancement/wav2lip_fullcontext_fit_runtime_v4_attempt01.py",
            "v3_head": REPO_ROOT / "scripts/audio_enhancement/shared_recipient_transition_head_v3.py",
            "v3_v4_adapter": REPO_ROOT / "scripts/audio_enhancement/v3_v4_native_runtime_adapter_attempt03.py",
            "audio_io": REPO_ROOT / "scripts/audio_enhancement/audio_io.py",
            "baselines": REPO_ROOT / "scripts/audio_enhancement/baselines.py",
            "fit40_geometry_producer": REPO_ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/build_fit40_natural_cache_rms_geometry_attempt04.py",
        }
        for key, path in local_sources.items():
            if not path.is_file() or sha_file(path) != module_expected[key]:
                raise RescoreError(f"local source differs from the qualified module pin: {key}")

        # Artifact/source pin and numerical-array validation are complete before the native model stack is imported.
        # NumPy was used above for the frozen-array checks; the outer reviewed launcher owns GPU admission.
        active_stage = "native_runtime_load"
        status.update({"status": "RUNNING_NATIVE_RUNTIME_LOAD", "stage": active_stage})
        write_status(status_path, status)
        import torch
        import torch.nn.functional as functional
        from scripts.audio_enhancement import baselines
        from scripts.audio_enhancement import shared_recipient_transition_head_v3 as v3_head
        from scripts.audio_enhancement import wav2lip_fullcontext_fit_runtime_v2_attempt02 as v2
        from scripts.audio_enhancement import wav2lip_fullcontext_fit_runtime_v4_attempt01 as v4
        from scripts.audio_enhancement import v3_v4_native_runtime_adapter_attempt03 as adapter_module

        if args.device != "cuda" and not (args.device.startswith("cuda:") and args.device[5:].isdigit()):
            raise RescoreError("an explicit CUDA device is required; CPU fallback is forbidden")
        for env_name, expected in (("CUBLAS_WORKSPACE_CONFIG", ":4096:8"), ("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("OPENBLAS_NUM_THREADS", "1"), ("PYTHONHASHSEED", "0")):
            if os.environ.get(env_name) != expected:
                raise RescoreError(f"runtime environment differs from frozen training binding: {env_name}")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(FIT_SEED)
        np.random.seed(FIT_SEED)
        torch.manual_seed(FIT_SEED)
        torch.cuda.manual_seed_all(FIT_SEED)
        if not torch.cuda.is_available():
            raise RescoreError("native final rescore requires CUDA")
        device = torch.device(args.device)
        if device.type != "cuda" or (device.index is not None and device.index >= torch.cuda.device_count()):
            raise RescoreError("requested CUDA device is unavailable")
        if device.index is not None and torch.cuda.current_device() != device.index:
            torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())
        if list(sys.version_info[:3]) != [3, 10, 12] or str(torch.__version__) != "2.5.1+cu124" or str(np.__version__) != "1.26.4":
            raise RescoreError("runtime Python/Torch/NumPy versions differ from the fit runtime")
        current_gpu = torch.cuda.get_device_properties(device.index)
        uuid = str(getattr(current_gpu, "uuid", "")).strip().lower()
        trained_gpu = fit["completion"].get("gpu_identity", {})
        producer_gpu = geometry_manifest.get("runtime_gpu_identity", {})
        if not uuid or uuid != str(trained_gpu.get("uuid", "")).strip().lower() or uuid != str(producer_gpu.get("uuid", "")).strip().lower():
            raise RescoreError("final checkpoint, geometry producer and current scorer GPU UUIDs differ")
        if (
            int(current_gpu.total_memory) != int(trained_gpu.get("total_memory_bytes", -1))
            or str(current_gpu.name) != str(trained_gpu.get("device_name", ""))
            or int(current_gpu.total_memory) != int(producer_gpu.get("torch_total_memory_bytes", -1))
        ):
            raise RescoreError("final scorer GPU identity differs from the training/geometry device")

        producer_profile = producer_gpu.get("runtime_profile")
        if not isinstance(producer_profile, dict):
            raise RescoreError("geometry producer GPU identity lacks its runtime profile")
        expected_gpu_identity = {
            "uuid": uuid,
            "device_index": int(device.index),
            "device_name": str(current_gpu.name),
            "memory_total_bytes": int(current_gpu.total_memory),
            "memory_total_mib": int(current_gpu.total_memory) // (1024 * 1024),
            "device_literal": args.device,
            "torch_device_name": str(current_gpu.name),
            "torch_total_memory_bytes": int(current_gpu.total_memory),
            "torch_compute_capability": [int(current_gpu.major), int(current_gpu.minor)],
        }
        for key, expected in expected_gpu_identity.items():
            actual = producer_gpu.get(key)
            if key == "uuid" and actual is not None:
                actual = str(actual).strip().lower()
            if canonical_bytes(actual) != canonical_bytes(expected):
                raise RescoreError(f"geometry producer and final scorer GPU identity differ: {key}")

        constructor = head_config["head_constructor"]
        cache_clone = output / "input_cache_clone"
        runtime_plan = copy.deepcopy(parent_plan)
        runtime_plan["runtime"]["full_context"]["audio_embedding_cache_root"] = str(cache_clone)
        runtime_plan["runtime"]["full_context"]["source_sample_count_total"] = sum(int(row["sample_count"]) for row in source_manifest["rows"])
        runtime_plan["fit_config"]["selected_rows"] = row_ids
        runtime_plan["fit_config"]["source_sample_count_total"] = runtime_plan["runtime"]["full_context"]["source_sample_count_total"]
        runtime_plan["runtime"]["fit_v4"] = {
            "schema": "cross_tfg_wav2lip_fullcontext_fit_runtime_v4_attempt01",
            "native_geometry_manifest_path": str(args.geometry_manifest.resolve(strict=True)),
            "native_geometry_manifest_sha256": args.expected_geometry_manifest_sha256.lower(),
        }
        active_stage = "audio_cache_clone"
        cache_clone_record = train_runner._clone_audio_cache(geometry_manifest, cache_clone)
        active_stage = "model_load_and_geometry"
        runtime = v2.create_runtime(plan=runtime_plan, fit_rows=[row["row"] for row in prepared], device=str(device), seed=FIT_SEED)
        if "optimizer" not in runtime or "enhancement_stack" not in runtime:
            raise RescoreError("qualified V2 loader did not return its legacy training objects for explicit discard")
        # create_runtime constructs a legacy V2 Adam object as part of its API.
        # Discard it without calling step(); the V3 scorer below creates no optimizer.
        legacy_optimizer = runtime.pop("optimizer")
        legacy_stack = runtime.pop("enhancement_stack")
        del legacy_optimizer, legacy_stack
        v2_optimizer_discarded = True
        runtime_pins = runtime["source_pins"]
        if not isinstance(runtime_pins, dict):
            raise RescoreError("V2 runtime did not provide its source pins")
        essential_runtime_keys = (
            "wav2lip_git_head", "wav2lip_sources", "wav2lip_checkpoint",
            "syncnet_git_head", "syncnet_sources", "syncnet_checkpoint",
            "official_frontend", "mel_spec", "python_speech_features",
            "torch_version", "numpy_version", "runtime_factory_path", "runtime_factory_sha256",
        )
        trained_runtime = fit["runtime_pins"].get("qualified_v2_runtime_source_pins", {})
        for key in essential_runtime_keys:
            if canonical_bytes(runtime_pins.get(key)) != canonical_bytes(trained_runtime.get(key)):
                raise RescoreError(f"current native scoring runtime differs from fit-time binding: {key}")
            if canonical_bytes(geometry_manifest.get("source_pins", {}).get("runtime_models_and_frontends", {}).get(key)) != canonical_bytes(runtime_pins.get(key)):
                raise RescoreError(f"geometry producer differs from current native scoring source: {key}")
        full_context_cfg = runtime_plan["runtime"]["full_context"]
        expected_batch_policy = {
            "video_frame_batch_size": int(full_context_cfg["video_frame_batch_size"]),
            "audio_embedding_batch_size": int(full_context_cfg["audio_embedding_batch_size"]),
            "syncnet_batch_size": int(full_context_cfg["syncnet_batch_size"]),
        }
        if (
            int(adapter.frame_batch_size) != expected_batch_policy["video_frame_batch_size"]
            or int(adapter.syncnet_batch_size) != expected_batch_policy["syncnet_batch_size"]
            or int(runtime_pins.get("full_context_audio_embedding_cache", {}).get("batch_size", -1)) != expected_batch_policy["audio_embedding_batch_size"]
        ):
            raise RescoreError("final scorer batch policy differs from the frozen parent plan")
        expected_runtime_profile = {
            "python_executable_literal": str(sys.executable),
            "python_version": ".".join(str(part) for part in sys.version_info[:3]),
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
            "seed": FIT_SEED,
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "batch_policy": {
                "wav2lip_native_frame_batch_size": expected_batch_policy["video_frame_batch_size"],
                "syncnet_audio_embedding_batch_size": expected_batch_policy["audio_embedding_batch_size"],
                "syncnet_visual_embedding_batch_size": expected_batch_policy["syncnet_batch_size"],
            },
            "frame_rate": {"numerator": 25, "denominator": 1},
            "source_sample_rate_hz": 16_000,
            "wav2lip_git_head": str(parent_plan["runtime"]["wav2lip"]["git_head"]),
            "wav2lip_checkpoint_sha256": str(parent_plan["runtime"]["wav2lip"]["checkpoint_sha256"]),
            "syncnet_git_head": str(parent_plan["runtime"]["syncnet"]["git_head"]),
            "syncnet_checkpoint_sha256": str(parent_plan["runtime"]["syncnet"]["checkpoint_sha256"]),
            "fixed_portrait_sha256": str(geometry_manifest.get("source_portrait", {}).get("sha256", "")).lower(),
        }
        for key, expected in expected_runtime_profile.items():
            if canonical_bytes(producer_profile.get(key)) != canonical_bytes(expected):
                raise RescoreError(f"fit geometry and final scorer runtime profiles differ: {key}")
        if sha_file(Path(v2.__file__).resolve()) != V2_RUNTIME_SHA256 or sha_file(Path(v4.__file__).resolve()) != V4_RUNTIME_SHA256 or sha_file(Path(v3_head.__file__).resolve()) != V3_HEAD_SHA256 or sha_file(Path(adapter_module.__file__).resolve()) != ADAPTER_SHA256:
            raise RescoreError("imported native modules differ from their frozen byte pins")
        adapter = runtime["adapter"]
        for model in (adapter.wav2lip_model, adapter.syncnet_model):
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        context_for_row = runtime["context_for_row"]
        geometry_runtime_cfg = runtime_plan["runtime"]["fit_v4"]
        _, native_geometries, native_geometry_records = v4._load_native_geometry_manifest(runtime_plan, [item["row"] for item in prepared], geometry_runtime_cfg)
        geometry_record_by_id = {str(record["row_id"]): record for record in native_geometry_records}
        if set(native_geometries) != set(row_ids) or set(geometry_record_by_id) != set(row_ids):
            raise RescoreError("native geometry loader did not return exactly the forty fit rows")

        head = v3_head.SharedRecipientTransitionHeadV3(**constructor).to(device=device)
        theta_tensor = torch.as_tensor(fit["theta"], dtype=torch.float32, device=device)
        if tuple(theta_tensor.shape) != (3, 3) or not torch.isfinite(theta_tensor).all().item():
            raise RescoreError("frozen checkpoint theta is not a finite 3x3 float32 matrix")
        with torch.no_grad():
            head.band_contrasts_db.copy_(theta_tensor)
        if not np.array_equal(tensor_np(head.band_contrasts_db, torch).astype("<f4"), np.asarray(fit["theta"], dtype="<f4")):
            raise RescoreError("loaded V3 theta differs from the frozen checkpoint values")
        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad_(False)

        max_frames = max(int(item["row"]["natural_pcm"]["sample_count"]) // SAMPLES_PER_40MS_FRAME for item in prepared)
        curves_out = np.empty((40, 31), dtype="<f4")
        visual_out = np.empty((40, 27, 1024), dtype="<f4")
        audio_center_out = np.empty((40, 27, 1024), dtype="<f4")
        query_indices_out = np.empty((40, 27), dtype="<i4")
        lag_offsets = np.arange(-15, 16, dtype="<i2")
        frame_power_out = np.zeros((40, 6, max_frames), dtype="<f4")
        frame_power_valid_out = np.zeros((40, max_frames), dtype=np.bool_)
        frame_centers_out = np.full((40, max_frames), -1, dtype="<i8")
        frame_counts_out = np.zeros((40,), dtype="<i4")
        trailing_out = np.zeros((40,), dtype="<i4")
        records: list[dict[str, Any]] = []
        ledger = output / "final120_rescore.jsonl"
        if ledger.exists():
            raise RescoreError("fresh final-rescore JSONL already exists")
        active_stage = "final120_no_grad_rescore"
        started = time.time()
        write_status(status_path, {**status, "status": "RUNNING_FROZEN_STEP120_NO_GRAD_RESCORE", "stage": active_stage})
        with torch.no_grad():
            for index, item in enumerate(prepared):
                row = item["row"]
                row_id = str(row["row_id"])
                context = context_for_row(row)
                context_cache = next(
                    value for value in geometry_manifest["rows"] if str(value["row_id"]) == row_id
                ).get("natural_audio_embedding_cache")
                cache_record = context.cache_record
                if not isinstance(context_cache, dict) or cache_record.get("npz_sha256") != context_cache.get("npz_sha256") or cache_record.get("metadata_sha256") != context_cache.get("metadata_sha256"):
                    raise RescoreError(f"natural-audio embedding cache changed for {row_id}")
                rms_audio_np, rms_record = train_runner._verify_baseline_wav(row, geometry_by_id[row_id], item["audio"], baselines)
                baseline_tensor = torch.from_numpy(np.ascontiguousarray(rms_audio_np)).to(device=device, dtype=torch.float32)
                source_tensor = torch.from_numpy(np.ascontiguousarray(item["audio"])).to(device=device, dtype=torch.float32)
                if source_tensor.numel() != context.expected_sample_count or baseline_tensor.numel() != context.expected_sample_count:
                    raise RescoreError(f"source/RMS/context sample count differs for {row_id}")
                v3_result = head(source_tensor, baseline_tensor)
                output_tensor = v3_result["output_pcm_float32"]
                if output_tensor.dtype != torch.float32 or output_tensor.ndim != 1 or output_tensor.numel() != source_tensor.numel() or not torch.isfinite(output_tensor).all().item():
                    raise RescoreError(f"frozen V3 output changed exact finite float32 sample contract for {row_id}")
                score = adapter_module.score_with_v4_native_path(
                    v4_runtime=v4,
                    v4_adapter=adapter,
                    context=context,
                    geometry_record=train_runner._row_geometry_record(native_geometries, row_id),
                    enhanced_audio=output_tensor,
                )
                curve = score["curve_float32"]
                visual = score["visual_embeddings_float32"]
                audio_center = score["audio_center_embeddings_float32"].detach()
                if tuple(curve.shape) != (31,) or tuple(visual.shape) != (27, 1024) or tuple(audio_center.shape) != (27, 1024):
                    raise RescoreError(f"final scorer support differs from 27 queries × 31 lags for {row_id}")
                curve_np = tensor_np(curve, torch).astype("<f4", copy=False)
                visual_np = tensor_np(visual, torch).astype("<f4", copy=False)
                audio_center_np = tensor_np(audio_center, torch).astype("<f4", copy=False)
                baseline_curve = baseline_curves[index]
                metrics = {key: finite_float(value, torch) for key, value in v4._curve_metrics(torch, curve).items()}
                metrics.update({f"RMS_{key}": finite_float(value, torch) for key, value in v4._curve_metrics(torch, torch.from_numpy(baseline_curve).to(device=device)).items()})
                argmin_index = int(torch.argmin(curve).item())
                baseline_argmin_index = int(np.argmin(baseline_curve))
                visual_norm, audio_norm, cosine = v4._embedding_metrics(torch, functional, visual, audio_center)
                candidate_pcm = tensor_np(output_tensor, torch).astype("<f4", copy=False)
                raw_candidate_pcm = tensor_np(v3_result["raw_candidate_pcm_before_rms"], torch).astype("<f4", copy=False)
                source_pcm = item["audio"].astype("<f4", copy=False)
                baseline_pcm = rms_audio_np.astype("<f4", copy=False)
                pcm_hashes = {
                    "source_wav_file_sha256": str(row["natural_pcm"]["file_sha256"]),
                    "source_canonical_float32_pcm_sha256": sha_bytes(source_pcm.tobytes()),
                    "source_pcm16_payload_sha256": str(row["natural_pcm"]["direct_source_pcm16_sha256"]),
                    "rms_baseline_wav_file_sha256": str(rms_record["file_sha256"]),
                    "rms_baseline_float32_pcm_sha256": sha_bytes(baseline_pcm.tobytes()),
                    "raw_candidate_before_rms_float32_pcm_sha256": sha_bytes(raw_candidate_pcm.tobytes()),
                    "final_candidate_float32_pcm_sha256": sha_bytes(candidate_pcm.tobytes()),
                }
                if pcm_hashes["source_canonical_float32_pcm_sha256"] != row["natural_pcm"]["pcm_sha256"]:
                    raise RescoreError(f"source PCM identity drifted during final rescore for {row_id}")

                power = v3_result["physical_frame_power"]
                power_names = ("source_original_pcm", "candidate_before_rms", "canonical_rms_baseline", "output_pcm")
                nframes = int(power["source_original_pcm"].numel())
                if nframes > max_frames or nframes != source_pcm.size // SAMPLES_PER_40MS_FRAME:
                    raise RescoreError(f"physical 40ms framing mismatch for {row_id}")
                for column, key in enumerate(power_names):
                    values = tensor_np(power[key], torch).astype("<f4", copy=False)
                    if values.shape != (nframes,) or not np.isfinite(values).all():
                        raise RescoreError(f"invalid 40ms power vector {key} for {row_id}")
                    frame_power_out[index, column, :nframes] = values
                for column, key in ((4, "candidate_to_source_ratio"), (5, "output_to_source_ratio")):
                    values = tensor_np(power[key], torch).astype("<f4", copy=False)
                    if values.shape != (nframes,) or not np.isfinite(values).all():
                        raise RescoreError(f"invalid 40ms power ratio {key} for {row_id}")
                    frame_power_out[index, column, :nframes] = values
                valid_power = power["valid_source_power_mask"].detach().to(device="cpu", dtype=torch.bool).numpy()
                if valid_power.shape != (nframes,):
                    raise RescoreError(f"40ms physical-power support mask has wrong shape for {row_id}")
                frame_power_valid_out[index, :nframes] = valid_power
                frame_centers_out[index, :nframes] = np.arange(nframes, dtype="<i8") * SAMPLES_PER_40MS_FRAME + SAMPLES_PER_40MS_FRAME // 2
                frame_counts_out[index] = nframes
                trailing_out[index] = int(power["trailing_unframed_sample_count"])
                frame_summary = make_40ms_frame_record(frame_power_out, frame_power_valid_out, index, nframes)

                geometry_record = geometry_by_id[row_id]
                geometry_digest = canonical_sha(geometry_record)
                query_start = int(context.query_start_frame)
                query_frames = np.arange(query_start, query_start + 27, dtype="<i4")
                if baseline_queries[index].tolist() != query_frames.tolist():
                    raise RescoreError(f"final audio query frame indices differ from saved RMS baseline for {row_id}")
                query_indices_out[index] = query_frames
                curves_out[index] = curve_np
                visual_out[index] = visual_np
                audio_center_out[index] = audio_center_np
                projected = finite_float(v3_result["postprojection_max_abs_mask_db"], torch)
                raw_mask = tensor_np(v3_result["raw_log_magnitude_mask_db"], torch)
                projected_mask = tensor_np(v3_result["projected_log_magnitude_mask_db"], torch)
                gates = {
                    "valid_transition_count": int(v3_result["positive_flux_valid_transition_count"]),
                    "positive_flux_threshold": finite_float(v3_result["positive_flux_threshold"], torch),
                    "negative_flux_threshold": finite_float(v3_result["negative_flux_threshold"], torch),
                    "positive_gate_mean_valid": float(tensor_np(v3_result["positive_flux_gate"], torch)[tensor_np(v3_result["valid_transition_mask"], torch, dtype=torch.bool)].mean()) if int(v3_result["positive_flux_valid_transition_count"]) else None,
                    "negative_gate_mean_valid": float(tensor_np(v3_result["negative_flux_gate"], torch)[tensor_np(v3_result["valid_transition_mask"], torch, dtype=torch.bool)].mean()) if int(v3_result["negative_flux_valid_transition_count"]) else None,
                    "raw_mask_max_abs_db": float(np.max(np.abs(raw_mask))),
                    "projected_mask_max_abs_db": projected,
                    "projection_scale": finite_float(v3_result["smooth_projection_scale"], torch),
                }
                normalization = v3_result["normalization"]
                rms_metric = lambda tensor: finite_float(torch.sqrt(tensor.square().mean()), torch)
                peak_metric = lambda tensor: finite_float(tensor.abs().amax(), torch)
                waveform = {
                    "sample_count": int(candidate_pcm.size),
                    "source_rms": rms_metric(source_tensor),
                    "source_peak_abs": peak_metric(source_tensor),
                    "canonical_rms_baseline_rms": rms_metric(baseline_tensor),
                    "canonical_rms_baseline_peak_abs": peak_metric(baseline_tensor),
                    "raw_candidate_pre_rms_rms": rms_metric(v3_result["raw_candidate_pcm_before_rms"]),
                    "raw_candidate_pre_rms_peak_abs": peak_metric(v3_result["raw_candidate_pcm_before_rms"]),
                    "final_output_rms": rms_metric(output_tensor),
                    "final_output_peak_abs": peak_metric(output_tensor),
                    "final_output_rms_error_from_target": rms_metric(output_tensor) - TARGET_RMS,
                    "declared_peak_ceiling": constructor.get("peak_ceiling"),
                    "peak_limiter_activated": bool(normalization["candidate"]["peak_limited"].item()),
                    "canonical_baseline_match_error_max_abs": finite_float(normalization["canonical_baseline_max_abs_match_error"], torch),
                }
                model_metrics = {
                    "curve31": metrics,
                    "candidate_argmin_lag_frame": int(argmin_index - 15),
                    "candidate_argmin_first_tie_index": argmin_index,
                    "rms_argmin_lag_frame": int(baseline_argmin_index - 15),
                    "rms_argmin_first_tie_index": baseline_argmin_index,
                    "delta_Cbest_vs_RMS": metrics["Cbest_B_minus_Dmin"] - metrics["RMS_Cbest_B_minus_Dmin"],
                    "delta_D0_vs_RMS": metrics["D0_zero_lag_distance"] - metrics["RMS_D0_zero_lag_distance"],
                    "delta_Canchor_vs_RMS": metrics["Canchor_zero_B_minus_D0"] - metrics["RMS_Canchor_zero_B_minus_D0"],
                    "delta_far_vs_RMS": metrics["far_lag_contrast_abs4plus"] - metrics["RMS_far_lag_contrast_abs4plus"],
                    "mean_visual_embedding_l2": finite_float(visual_norm.mean(), torch),
                    "mean_natural_audio_embedding_l2": finite_float(audio_norm.mean(), torch),
                    "mean_zero_lag_unit_cosine": finite_float(cosine.mean(), torch),
                }
                record = {
                    "sequence_index": index,
                    "row_id": row_id,
                    "sample_id": row["sample_id"],
                    "speaker_id": row["speaker_id"],
                    "canonical_j": int(row["canonical_j"]),
                    "split": "fit",
                    "checkpoint_sha256": fit["artifacts"]["checkpoint"]["sha256"],
                    "geometry_record_sha256": geometry_digest,
                    "source_clock": {
                        "fps_num": 25,
                        "fps_den": 1,
                        "audio_query_start_frame": query_start,
                        "audio_query_frames_27": query_frames.tolist(),
                        "lag_offsets_frames": list(range(-15, 16)),
                        "mapped_prediction_source_frames_32": list(geometry_record["prediction_source_clock_frames"]),
                        "output_source_clock_frames_32": list(geometry_record["output_source_clock_frames"]),
                        "track_local_frames_32": list(geometry_record["track_local_frame_indices"]),
                        "clock_shift_applied": False,
                    },
                    "pcm_hashes": pcm_hashes,
                    "waveform": waveform,
                    "curve_metrics": model_metrics,
                    "embedding_metrics": {
                        "visual_shape": [27, 1024],
                        "audio_center_shape": [27, 1024],
                        "visual_norm_mean": finite_float(visual_norm.mean(), torch),
                        "audio_norm_mean": finite_float(audio_norm.mean(), torch),
                        "cosine_mean": finite_float(cosine.mean(), torch),
                    },
                    "source_conditioned_gate_and_mask_summary": gates,
                    "physical_40ms_frame_power": {
                        "definition": "unweighted mean-square sample power on original 16 kHz source-clock bins of 640 samples; last partial bin is excluded and counted",
                        "full_frame_count": nframes,
                        "trailing_unframed_samples": trailing_out[index],
                        "valid_power_frame_count": int(valid_power.sum()),
                        "summary": frame_summary,
                    },
                    "natural_audio_cache_sha256": cache_record.get("npz_sha256"),
                    "natural_audio_cache_metadata_sha256": cache_record.get("metadata_sha256"),
                    "status": "PASS_FINAL_STEP120_FROZEN_NO_GRAD",
                }
                records.append(record)
                append_jsonl(ledger, record)
                completed_rows = index + 1
                write_status(status_path, {
                    **status,
                    "status": "RUNNING_FROZEN_STEP120_NO_GRAD_RESCORE",
                    "stage": active_stage,
                    "completed_rows": completed_rows,
                    "planned_rows": 40,
                    "checkpoint_sha256": fit["artifacts"]["checkpoint"]["sha256"],
                    "v3_optimizer_created": False,
                    "optimizer_updates": 0,
                    "v2_loader_optimizer_constructed_then_discarded": v2_optimizer_discarded,
                    "last_row_id": row_id,
                })

        if len(records) != 40:
            raise RescoreError("frozen checkpoint rescore did not complete all forty fit rows")
        arrays_path = output / "final120_rescore_arrays.npz"
        with arrays_path.open("xb") as handle:
            np.savez_compressed(
                handle,
                row_ids=np.asarray(row_ids, dtype="U96"),
                lag_offsets_frames_int16=lag_offsets,
                candidate_curves_float32=curves_out,
                candidate_visual_embeddings_float32=visual_out,
                original_audio_center_embeddings_float32=audio_center_out,
                query_frame_indices_int32=query_indices_out,
                frame_power_float32=frame_power_out,
                frame_power_kind=np.asarray(["source_original", "candidate_before_rms", "canonical_rms_baseline", "final_output", "candidate_to_source_ratio", "output_to_source_ratio"], dtype="U40"),
                frame_power_valid_mask_bool=frame_power_valid_out,
                frame_center_source_samples_int64=frame_centers_out,
                frame_count_int32=frame_counts_out,
                trailing_unframed_sample_count_int32=trailing_out,
            )
            handle.flush()
            os.fsync(handle.fileno())
        rows_raw_sha = sha_file(ledger)
        arrays_pin = pin(arrays_path)
        ledger_pin = pin(ledger)
        mean_deltas = {
            metric: float(np.mean([record["curve_metrics"][f"delta_{metric}_vs_RMS"] for record in records], dtype="float64"))
            for metric in ("Cbest", "D0", "Canchor", "far")
        }
        source_pins = {
            "rescore_runner": pin(Path(__file__).resolve()),
            "training_runner": {"path": TRAIN_RUNNER_PATH, "bytes": 72032, "sha256": TRAIN_RUNNER_SHA256},
            "training_completion": fit["artifacts"]["completion"],
            "training_record": fit["artifacts"]["training_record"],
            "checkpoint": fit["artifacts"]["checkpoint"],
            "training_artifacts": fit["artifacts"],
            "training_runtime_source_pins": fit["artifacts"]["runtime_source_pins"],
            "source_manifest": {"path": str(args.source_manifest.resolve(strict=True)), "bytes": len(source_raw), "sha256": sha_bytes(source_raw)},
            "parent_plan": {"path": str(args.parent_plan.resolve(strict=True)), "bytes": len(parent_raw), "sha256": sha_bytes(parent_raw), "payload_sha256": PARENT_PLAN_PAYLOAD_SHA256},
            "native_geometry_manifest": {"path": str(args.geometry_manifest.resolve(strict=True)), "bytes": len(geometry_raw), "sha256": sha_bytes(geometry_raw), "payload_sha256": geometry_manifest["payload_sha256"]},
            "head_config": {"path": str(args.head_config.resolve(strict=True)), "bytes": len(head_config_raw), "sha256": sha_bytes(head_config_raw)},
            "native_model_runtime": runtime_pins,
            "audio_cache_clone": cache_clone_record,
            "gpu_identity": {
                "uuid": uuid,
                "cuda_device_index": int(device.index),
                "device_name": str(current_gpu.name),
                "total_memory_bytes": int(current_gpu.total_memory),
                "training_uuid_match": True,
                "geometry_producer_uuid_match": True,
            },
        }
        completion = {
            "schema": SCHEMA,
            "status": "COMPLETE_FIT_ONLY_UNSELECTED_FINAL120_RESCORE",
            "fit_rows_planned": 40,
            "fit_rows_scored": 40,
            "fit_speakers": 20,
            "checkpoint_step": 120,
            "checkpoint_is_unselected": True,
            "training_step120_curve_is_preupdate_observation": True,
            "final_rescore_uses_postupdate_checkpoint_theta": True,
            "same_device_zero_head_gate_array_audit": {
                "status": "PASS",
                "curve31_max_abs_vs_rms": zero_curve_max_abs_vs_rms,
                "visual_embedding_max_abs_vs_rms": zero_visual_max_abs_vs_rms,
                "frozen_max_abs_tolerance": ZERO_GATE_ATOL,
                "waveform_zero_head_parity": "bit_equal_as_recorded_by_training_gate",
            },
            "v3_optimizer_created": False,
            "optimizer_updates": 0,
            "v2_loader_optimizer_constructed_then_discarded": v2_optimizer_discarded,
            "selection_record_created": False,
            "development_audio_opened": False,
            "confirmation_audio_opened": False,
            "checkpoint_parameter": fit["checkpoint"]["parameter"],
            "checkpoint_sha256": fit["artifacts"]["checkpoint"]["sha256"],
            "rows_jsonl": ledger_pin,
            "arrays_npz": arrays_pin,
            "rows_jsonl_raw_sha256": rows_raw_sha,
            "fit_mean_delta_vs_native_rms": mean_deltas,
            "source_pins": source_pins,
            "gpu_identity": source_pins["gpu_identity"],
            "elapsed_seconds": time.time() - started,
        }
        completion["payload_sha256"] = canonical_sha({key: value for key, value in completion.items() if key != "payload_sha256"})
        completion_raw = atomic_json(output / "final120_rescore_manifest.json", completion)
        write_status(status_path, {**completion, "status_file_stage": "complete"})
        print(json.dumps({"schema": SCHEMA, "status": completion["status"], "rows": 40, "manifest_sha256": sha_bytes(completion_raw)}, ensure_ascii=False, sort_keys=True))
        return completion
    except Exception as exc:
        failure = {
            "schema": SCHEMA,
            "status": "FAILED_FINAL120_FIT_ONLY_RESCORE",
            "stage": active_stage,
            "completed_rows": completed_rows,
            "exception_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=25),
            "v3_optimizer_created": False,
            "optimizer_updates": 0,
            "v2_loader_optimizer_constructed_then_discarded": v2_optimizer_discarded,
            "selection_record_created": False,
            "development_audio_opened": False,
            "confirmation_audio_opened": False,
            "failed_unix_seconds": time.time(),
        }
        try:
            if not (output / "failure.json").exists():
                atomic_json(output / "failure.json", failure)
            write_status(status_path, {**status, **failure, "status": "FAILED"})
        finally:
            raise


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        run(args)
        return 0
    except Exception as exc:
        print(json.dumps({"schema": SCHEMA, "status": "FAILED", "error": str(exc)}, sort_keys=True, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
