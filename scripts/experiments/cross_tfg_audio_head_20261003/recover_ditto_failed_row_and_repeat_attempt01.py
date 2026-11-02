#!/usr/bin/env python3
"""Retry the single ENOSPC Ditto row and perform a real fresh repeat.

The frozen 240-row plan and its original run are read-only. Every PASS row is
reused only after validating its source result line, every bound file, and all
seven raw-score arrays. The failed planned row and the same-seed identity
repeat are freshly rendered and rescored under a new SHM-backed gate path.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PLAN_RAW_SHA256 = "170080a0c1608cbf34f8353b1da656f5d9c56de850fdd3f50f4eff4a967e43a0"
PLAN_PAYLOAD_SHA256 = "a5c5f4ba36ec665b0a42ec11a1084f4e6f5d8a04a36b5b4906e8c55b750c0bc3"
ROWS_RAW_SHA256 = "f2fa4c0b56b5a4e91e22ef3997bbf491065148027e88324e42c9ae25832e5635"
OLD_RESULTS_RAW_SHA256 = "a4789da0c102d5bfbac43fea0b8979b40ea91498c8bdd6a95a4bce98746498a9"
OLD_MANIFEST_RAW_SHA256 = "6f170c5ada451a5e2a50f999d6edadb193dfd99c3f616063e880a7d24d7b5fca"
STAGEA_CORE_SHA256 = "30c894ea31541cdde76a5a915bf635a36838a1cdb11f761c334c8986e71a01bc"
FRONTEND_RUNNER_SHA256 = "7f78a14485eebaa20e7afd893a46b2ead26db5544194d988d7d9890d6a43ebbc"
DITTO_SHIM_SHA256 = "848c7ddd5905eb262935b87197ac13c0f5ed1e7cb760ed091cf8b0e4fc8e1a67"
EXPECTED_FAILED_ROW = "dev_S0050_BAC009S0050W0122_ditto_hubert_fixed_portrait_attempt18_primary_face_static_portrait_v1_20261005_spectral_head_fit_v3_zero_lag_unitcos"
REPEAT_REFERENCE_ROW = "dev_S0005_BAC009S0005W0121_ditto_hubert_fixed_portrait_attempt18_primary_face_static_portrait_v1_20261003_identity"
GPU_MIN_FREE_MIB = 18000
GPU_WAIT_SECONDS = 3600
GPU_POLL_SECONDS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_pinned_module(name: str, path: Path, expected_sha256: str):
    path = path.resolve(strict=True)
    actual = sha256_file(path)
    if len(expected_sha256) != 64 or actual != expected_sha256:
        raise RuntimeError(f"pinned source mismatch for {path}: {actual}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import pinned source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_refs(value: Any, cache: dict[str, str], row_id: str) -> int:
    checked = 0
    if isinstance(value, Mapping):
        path_text = value.get("path")
        file_hash = value.get("file_sha256")
        if isinstance(path_text, str) and isinstance(file_hash, str):
            path = Path(path_text)
            if not path.is_file():
                raise RuntimeError(f"PASS row {row_id} references missing file {path}")
            key = str(path)
            actual = cache.get(key)
            if actual is None:
                actual = sha256_file(path)
                cache[key] = actual
            if len(file_hash) != 64 or actual != file_hash.lower():
                raise RuntimeError(f"PASS row {row_id} file hash mismatch: {path}")
            if value.get("bytes") is not None and int(value["bytes"]) != path.stat().st_size:
                raise RuntimeError(f"PASS row {row_id} byte count mismatch: {path}")
            checked += 1
        for nested in value.values():
            checked += verify_refs(nested, cache, row_id)
    elif isinstance(value, list):
        for nested in value:
            checked += verify_refs(nested, cache, row_id)
    return checked


def load_score_arrays(result: Mapping[str, Any], core: Any) -> dict[str, np.ndarray]:
    row_id = str(result.get("row_id"))
    score = result.get("score")
    if not isinstance(score, Mapping) or not isinstance(score.get("raw_matrix_npz"), Mapping):
        raise RuntimeError(f"PASS row {row_id} has no raw_matrix_npz")
    ref = score["raw_matrix_npz"]
    path = Path(str(ref.get("path", "")))
    if not path.is_file() or sha256_file(path) != ref.get("file_sha256"):
        raise RuntimeError(f"PASS row {row_id} NPZ file hash mismatch")
    metadata = ref.get("arrays")
    if not isinstance(metadata, Mapping) or set(metadata) != set(core.ARRAY_DTYPES):
        raise RuntimeError(f"PASS row {row_id} NPZ array map mismatch")
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(core.ARRAY_DTYPES):
            raise RuntimeError(f"PASS row {row_id} NPZ key set mismatch")
        for key, expected_dtype in core.ARRAY_DTYPES.items():
            array = np.ascontiguousarray(loaded[key])
            info = metadata[key]
            if array.dtype.str != expected_dtype.str or array.dtype.str != info.get("dtype"):
                raise RuntimeError(f"PASS row {row_id} NPZ dtype mismatch for {key}")
            if list(array.shape) != info.get("shape"):
                raise RuntimeError(f"PASS row {row_id} NPZ shape mismatch for {key}")
            if sha256_bytes(array.tobytes(order="C")) != info.get("array_sha256"):
                raise RuntimeError(f"PASS row {row_id} NPZ array hash mismatch for {key}")
            arrays[key] = array
    count = arrays["query_frame_index"].shape[0]
    if arrays["distance_by_query_lag"].shape != (count, 31):
        raise RuntimeError(f"PASS row {row_id} distance matrix must have N x 31 shape")
    for key in ("query_timestamp_num", "query_timestamp_den", "track_frame_index", "valid_query_mask"):
        if arrays[key].shape != (count,):
            raise RuntimeError(f"PASS row {row_id} query map length mismatch for {key}")
    if arrays["lag_offsets_frames"].shape != (31,) or not np.array_equal(
        arrays["lag_offsets_frames"], np.arange(-15, 16, dtype=arrays["lag_offsets_frames"].dtype)
    ):
        raise RuntimeError(f"PASS row {row_id} lag axis mismatch")
    return arrays


def gpu_admission(control_dir: Path, stage: str, row_id: str) -> dict[str, Any]:
    started = time.monotonic()
    deadline = started + GPU_WAIT_SECONDS
    last_event = 0.0
    while True:
        now = time.monotonic()
        try:
            query = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False, timeout=10,
            )
            if query.returncode != 0:
                raise RuntimeError(query.stderr.decode("utf-8", "replace")[-1000:])
            free_mib = int(query.stdout.decode("ascii", "strict").splitlines()[0].strip())
            probe_error = None
        except Exception as exc:
            free_mib = None
            probe_error = repr(exc)
        event = {
            "schema": "ditto_recovery_gpu_admission_v1", "stage": stage,
            "row_id": row_id, "gpu_free_mib": free_mib,
            "minimum_free_mib": GPU_MIN_FREE_MIB,
            "elapsed_seconds": round(now - started, 3), "timestamp_unix": time.time(),
            "probe_error": probe_error,
        }
        if free_mib is not None and free_mib >= GPU_MIN_FREE_MIB:
            event["status"] = "ADMITTED"
            append_json_line(control_dir / "gpu_admission_events.jsonl", event)
            write_json(control_dir / "gpu_admission_current.json", event)
            return event
        if now >= deadline:
            event["status"] = "WAIT_EXPIRED"
            append_json_line(control_dir / "gpu_admission_events.jsonl", event)
            write_json(control_dir / "gpu_admission_current.json", event)
            raise RuntimeError(f"GPU admission timed out for {stage}/{row_id}")
        if now - last_event >= 30:
            event["status"] = "WAITING"
            append_json_line(control_dir / "gpu_admission_events.jsonl", event)
            write_json(control_dir / "gpu_admission_current.json", event)
            last_event = now
        time.sleep(GPU_POLL_SECONDS)


def append_json_line(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def public_result(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_internal_")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--reuse-run", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--expected-plan-raw-sha256", default=PLAN_RAW_SHA256)
    parser.add_argument("--expected-rows-raw-sha256", default=ROWS_RAW_SHA256)
    parser.add_argument("--expected-results-raw-sha256", default=OLD_RESULTS_RAW_SHA256)
    parser.add_argument("--expected-manifest-raw-sha256", default=OLD_MANIFEST_RAW_SHA256)
    own = parser.parse_args()

    task = Path(own.task_root).resolve(strict=True)
    code = task / "code"
    plan_path = task / "plan/plan.json"
    rows_path = task / "plan/planned_rows.jsonl"
    old_out = Path(own.reuse_run).resolve(strict=True)
    out = Path(own.output_root).resolve()
    control = Path(own.control_root).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"recovery output root is non-empty: {out}")
    if control.exists() and any(control.iterdir()):
        raise FileExistsError(f"recovery control root is non-empty: {control}")
    if out == old_out or old_out in out.parents or out in old_out.parents:
        raise RuntimeError("recovery output must be separate from the frozen original run")
    out.mkdir(parents=True, exist_ok=True)
    control.mkdir(parents=True, exist_ok=True)
    gate_control = control / "ditto_adapter_gate"
    gate_control.mkdir()
    (control / "tmp").mkdir()
    (control / "cache").mkdir()
    for variable in ("TMPDIR", "TMP", "TEMP"):
        os.environ[variable] = str(control / "tmp")
    os.environ["XDG_CACHE_HOME"] = str(control / "cache")

    plan_raw = plan_path.read_bytes()
    rows_raw = rows_path.read_bytes()
    if sha256_bytes(plan_raw) != own.expected_plan_raw_sha256 or sha256_bytes(rows_raw) != own.expected_rows_raw_sha256:
        raise RuntimeError("frozen plan/rows raw SHA differs from recovery pins")
    plan = json.loads(plan_raw.decode("utf-8"))
    rows = [json.loads(line) for line in rows_raw.decode("utf-8").splitlines() if line.strip()]
    old_results_path = old_out / "results.jsonl"
    old_manifest_path = old_out / "result_manifest.json"
    old_repeat_path = old_out / "generation_repeat_qc.jsonl"
    for path, expected in (
        (old_results_path, own.expected_results_raw_sha256),
        (old_manifest_path, own.expected_manifest_raw_sha256),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"frozen source result hash mismatch: {path}")
    if plan.get("status") != "FROZEN" or plan.get("plan_sha256") != PLAN_PAYLOAD_SHA256:
        raise RuntimeError("frozen Ditto plan payload does not match the reviewed plan")
    if len(rows) != 240 or [r["row_id"] for r in rows] != [r["row_id"] for r in plan["rows"]]:
        raise RuntimeError("frozen Ditto plan is not the exact 240-row ledger order")
    if not old_repeat_path.is_file():
        raise RuntimeError("original repeat-QC ledger is missing")

    core_path = code / "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt02.py"
    wrapper_path = code / "run_cross_frontend_dev_attempt03.py"
    shim_bundle_path = code / "ditto_five_arg_shim_attempt03.py"
    if sha256_file(core_path) != STAGEA_CORE_SHA256 or sha256_file(wrapper_path) != FRONTEND_RUNNER_SHA256 or sha256_file(shim_bundle_path) != DITTO_SHIM_SHA256:
        raise RuntimeError("frozen Stage-A/adapter source pins changed")
    sys.path.insert(0, str(code))
    wrapper = load_pinned_module("ditto_recovery_frontend_wrapper", wrapper_path, FRONTEND_RUNNER_SHA256)
    core, _ = wrapper._load_core(core_path, STAGEA_CORE_SHA256)

    model = plan["models"][0]
    original_generator = model["generator"]
    shim_path = Path(str(original_generator.get("adapter_path", ""))).resolve(strict=True)
    if model["model_id"] != "ditto_hubert_fixed_portrait_attempt18":
        raise RuntimeError("unexpected frozen Ditto model id")
    if sha256_file(shim_path) != DITTO_SHIM_SHA256 or original_generator.get("adapter_sha256") != DITTO_SHIM_SHA256:
        raise RuntimeError("frozen Ditto adapter reference differs from this bundle")
    model_for_render = copy.deepcopy(model)
    model_for_render["generator"]["env"]["CROSS_TFG_CONTROL_DIR"] = str(gate_control)

    old_results = [json.loads(line) for line in old_results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    old_manifest = read_json(old_manifest_path)
    manifest_payload = {key: value for key, value in old_manifest.items() if key != "manifest_sha256"}
    if old_manifest.get("manifest_sha256") != sha256_bytes(core.canonical(manifest_payload)):
        raise RuntimeError("original result manifest canonical seal is invalid")
    if old_manifest.get("plan_raw_sha256") != own.expected_plan_raw_sha256 or old_manifest.get("results_jsonl_sha256") != own.expected_results_raw_sha256:
        raise RuntimeError("original result manifest does not bind the expected plan/results")
    if old_manifest.get("status_counts") != {"PASS": 239, "FAILED": 1, "INCOMPLETE": 0}:
        raise RuntimeError("original Ditto ledger is not the diagnosed 239 PASS / 1 FAILED run")
    old_repeat = [json.loads(line) for line in old_repeat_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(old_repeat) != 1 or old_repeat[0].get("status") != "FAILED":
        raise RuntimeError("original repeat QC is not the diagnosed single failed generation attempt")
    if old_manifest.get("generation_repeat_qc_jsonl_sha256") != sha256_file(old_repeat_path):
        raise RuntimeError("original manifest does not bind the failed repeat-QC bytes")
    comparison = old_repeat[0].get("comparison")
    if not isinstance(comparison, Mapping) or any(value is not None for value in comparison.values()):
        raise RuntimeError("original failed repeat must have null comparisons; it cannot establish determinism")
    if plan.get("confirm_speaker_ids") or plan.get("generation_seed_family", {}).get("confirm"):
        raise RuntimeError("the frozen Ditto development plan unexpectedly includes confirmation data")

    if len(old_results) != 240 or [r.get("row_id") for r in old_results] != [r["row_id"] for r in rows]:
        raise RuntimeError("original results do not exactly preserve the frozen planned order")
    plan_by_id = {row["row_id"]: row for row in rows}
    reused: list[dict[str, Any]] = []
    retry_rows: list[dict[str, Any]] = []
    refs_cache: dict[str, str] = {}
    ref_count = 0
    for result in old_results:
        row_id = result["row_id"]
        planned = plan_by_id[row_id]
        for field in core.IDENTITY_FIELDS:
            if result.get(field) != planned.get(field):
                raise RuntimeError(f"old Ditto row identity mismatch: {row_id} / {field}")
        if result.get("status") == "PASS":
            ref_count += verify_refs(result, refs_cache, row_id)
            internal = copy.deepcopy(result)
            internal["_internal_arrays"] = load_score_arrays(result, core)
            internal["_internal_video_path"] = str(result["render"].get("silent_video_path", ""))
            internal["_internal_matrix_sha256"] = result["score"]["raw_matrix_npz"]["arrays"]["distance_by_query_lag"]["array_sha256"]
            internal["_internal_crop_policy_sha256"] = result["render"].get("crop_policy_sha256")
            reused.append(internal)
        elif result.get("status") == "FAILED":
            if row_id != EXPECTED_FAILED_ROW:
                raise RuntimeError(f"unexpected failed planned row: {row_id}")
            if "no space left on device" not in str(result.get("failure_reason", "")).lower():
                raise RuntimeError("the single original row failure is not the diagnosed ENOSPC condition")
            retry_rows.append(planned)
        else:
            raise RuntimeError(f"unexpected original status {result.get('status')} in {row_id}")
    if len(reused) != 239 or len(retry_rows) != 1 or retry_rows[0]["row_id"] != EXPECTED_FAILED_ROW:
        raise RuntimeError("recovery requires exactly 239 reusable PASS rows and the one diagnosed ENOSPC row")

    reference_row = plan_by_id.get(REPEAT_REFERENCE_ROW)
    baseline = next((item for item in reused if item["row_id"] == REPEAT_REFERENCE_ROW), None)
    if reference_row is None or baseline is None or reference_row["method_id"] != "identity":
        raise RuntimeError("same-seed identity repeat baseline is not among the verified PASS rows")

    old_line_hashes = {
        json.loads(line.decode("utf-8"))["row_id"]: sha256_bytes(line)
        for line in old_results_path.read_bytes().splitlines() if line.strip()
    }
    audit = {
        "schema": "ditto_stageb_single_row_recovery_audit_v1",
        "frozen_plan_raw_sha256": own.expected_plan_raw_sha256,
        "frozen_plan_payload_sha256": PLAN_PAYLOAD_SHA256,
        "frozen_rows_raw_sha256": own.expected_rows_raw_sha256,
        "source_results_raw_sha256": own.expected_results_raw_sha256,
        "source_manifest_raw_sha256": own.expected_manifest_raw_sha256,
        "source_repeat_qc_raw_sha256": sha256_file(old_repeat_path),
        "source_run_path": str(old_out),
        "reused_pass_rows": len(reused),
        "reused_pass_row_ids": [row["row_id"] for row in reused],
        "reused_result_line_sha256": {result["row_id"]: old_line_hashes[result["row_id"]] for result in reused},
        "retry_row_ids": [EXPECTED_FAILED_ROW],
        "original_repeat_failed_status": old_repeat[0].get("status"),
        "repeat_reference_row_id": REPEAT_REFERENCE_ROW,
        "verified_bound_file_references": ref_count,
        "verified_unique_files": len(refs_cache),
        "verified_unique_file_bytes": sum(Path(path).stat().st_size for path in refs_cache),
        "all_reused_score_npz_and_seven_arrays_verified": True,
        "fresh_adapter_control_path": str(gate_control),
        "recovery_output_root": str(out),
        "status": "PASS_PREFLIGHT_AUDIT",
    }
    audit_path = control / "reuse_preflight_audit.json"
    if audit_path.exists():
        raise FileExistsError(audit_path)
    write_json(audit_path, audit)
    write_json(control / "run_status.json", {
        "schema": "ditto_stageb_single_row_recovery_status_v1", "status": "PREFLIGHT_PASS",
        "run_id": plan["run_id"], "pid": os.getpid(), "planned_rows": 240,
        "reused_pass_rows": 239, "retry_rows": 1, "repeat_rows": 1,
        "reuse_preflight_audit_path": str(audit_path), "reuse_preflight_audit_sha256": sha256_file(audit_path),
        "output_root": str(out), "control_root": str(control),
    })

    args = core.build_parser().parse_args([
        "--plan", str(plan_path), "--planned-rows", str(rows_path),
        "--expected-plan-raw-sha256", own.expected_plan_raw_sha256,
        "--output-root", str(out), "--project-root", str(code),
        "--wav2lip-python", str(original_generator["python"]),
        "--wav2lip-root", "/root/autodl-tmp/wav2lip-syncnet/Wav2Lip",
        "--generator-child", str(shim_path),
        "--syncnet-root", "/root/autodl-tmp/wav2lip-syncnet/syncnet_python",
        "--syncnet-python", "/root/autodl-tmp/wav2lip-syncnet/envs/syncnet/bin/python",
        "--syncnet-checkpoint", "/root/autodl-tmp/wav2lip-syncnet/syncnet_python/data/syncnet_v2.model",
        "--syncnet-evaluator", str(code / "scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py"),
        "--ffmpeg", "/usr/bin/ffmpeg", "--ffprobe", "/usr/bin/ffprobe",
        "--ffmpeg-sha256", plan["protocol"]["syncnet"]["primary_audio_input"]["ffmpeg_sha256"],
        "--pcm16-lineage-sha256", plan["execution"]["primary_audio_lineage_code_sha256"],
        "--generator-timeout", "3600", "--syncnet-timeout", "1800", "--ffmpeg-timeout", "300",
    ])
    args.project_root = str(code)

    original_run_logged = core.run_logged
    target_row_id = EXPECTED_FAILED_ROW
    active_row = {"row_id": target_row_id}

    def admitted_run_logged(argv: Sequence[str], **kwargs: Any) -> dict[str, Any]:
        words = [str(item) for item in argv]
        command_script = words[1] if len(words) > 1 else ""
        if command_script == str(shim_path) or command_script.endswith("/run_pipeline.py") or command_script.endswith("/stagea_syncnet_evaluator.py"):
            stage = "ditto_renderer" if command_script == str(shim_path) else ("syncnet_tracker" if command_script.endswith("/run_pipeline.py") else "syncnet_evaluator")
            admission = gpu_admission(control, stage, str(active_row["row_id"]))
            kwargs_env = kwargs.get("env")
            if kwargs_env is not None:
                updated_env = dict(kwargs_env)
                updated_env["CROSS_TFG_GPU_ADMISSION_STATUS"] = admission["status"]
                kwargs["env"] = updated_env
        return original_run_logged(argv, **kwargs)

    core.run_logged = admitted_run_logged
    try:
        result_by_id: dict[str, dict[str, Any]] = {result["row_id"]: result for result in reused}
        for row in retry_rows:
            active_row["row_id"] = row["row_id"]
            status_path = control / "run_status.json"
            write_json(status_path, {
                "schema": "ditto_stageb_single_row_recovery_status_v1", "status": "RUNNING_RETRY_ROW",
                "run_id": plan["run_id"], "pid": os.getpid(), "planned_rows": 240,
                "reused_pass_rows": len(reused), "completed_retry_rows": 0,
                "current_row_id": row["row_id"], "output_root": str(out),
            })
            from scripts.audio_enhancement.audio_io import read_canonical_wav
            natural_path = Path(row["natural_pcm"]["path"]).resolve(strict=True)
            natural_samples, audio_info = read_canonical_wav(natural_path)
            natural_ref = {
                "path": str(natural_path), "file_sha256": audio_info.file_sha256,
                "pcm_sha256": audio_info.pcm_float32_sha256, "sample_rate_hz": 16000,
                "channels": 1, "sample_count": audio_info.frames, "subtype": "IEEE_FLOAT_32_LE",
            }
            if natural_ref["file_sha256"] != row["natural_pcm"]["file_sha256"] or natural_ref["pcm_sha256"] != row["natural_pcm"]["pcm_sha256"]:
                raise RuntimeError("failed-row natural canonical audio no longer matches the frozen plan")
            method = next(item for item in plan["methods"] if item["method_id"] == row["method_id"])
            try:
                result = core._render_and_score(
                    row, model_for_render, method, natural_path, natural_samples, natural_ref,
                    out / "rows" / row["row_id"], args, qc_repeat=False,
                )
            except Exception as exc:
                result = core._identity_base(row, natural_ref)
                result["failure_reason"] = f"{type(exc).__name__}: {exc}"
                result["failure_traceback"] = traceback.format_exc()
                core.write_json(out / "row_failures" / row["row_id"] / "failure.json", {
                    "row_id": row["row_id"], "error": str(exc),
                    "traceback": result["failure_traceback"], "unix_time": time.time(),
                })
            result_by_id[row["row_id"]] = result
            write_json(control / "run_status.json", {
                "schema": "ditto_stageb_single_row_recovery_status_v1", "status": "RETRY_ROW_COMPLETE",
                "run_id": plan["run_id"], "pid": os.getpid(), "planned_rows": 240,
                "reused_pass_rows": len(reused), "completed_retry_rows": 1,
                "retry_row_status": result["status"], "current_row_id": row["row_id"],
                "output_root": str(out),
            })

        if len(result_by_id) != 240:
            raise RuntimeError("recovery result map does not preserve the frozen 240 planned rows")
        ordered = [result_by_id[row["row_id"]] for row in rows]
        if any("_internal_arrays" not in result for result in ordered if result["status"] == "PASS"):
            raise RuntimeError("a PASS row is missing its verified raw score arrays")

        active_row["row_id"] = REPEAT_REFERENCE_ROW
        baseline = result_by_id[REPEAT_REFERENCE_ROW]
        repeat_row = plan_by_id[REPEAT_REFERENCE_ROW]
        repeat_natural_path = Path(repeat_row["natural_pcm"]["path"]).resolve(strict=True)
        repeat_samples, repeat_info = read_canonical_wav(repeat_natural_path)
        repeat_natural_ref = {
            "path": str(repeat_natural_path), "file_sha256": repeat_info.file_sha256,
            "pcm_sha256": repeat_info.pcm_float32_sha256, "sample_rate_hz": 16000,
            "channels": 1, "sample_count": repeat_info.frames, "subtype": "IEEE_FLOAT_32_LE",
        }
        if repeat_natural_ref["file_sha256"] != repeat_row["natural_pcm"]["file_sha256"] or repeat_natural_ref["pcm_sha256"] != repeat_row["natural_pcm"]["pcm_sha256"]:
            raise RuntimeError("repeat baseline natural audio no longer matches the frozen plan")
        write_json(control / "run_status.json", {
            "schema": "ditto_stageb_single_row_recovery_status_v1", "status": "RUNNING_FRESH_REPEAT",
            "run_id": plan["run_id"], "pid": os.getpid(), "planned_rows": 240,
            "reused_pass_rows": len(reused), "completed_retry_rows": 1,
            "retry_row_status": result_by_id[EXPECTED_FAILED_ROW]["status"],
            "repeat_reference_row_id": REPEAT_REFERENCE_ROW, "output_root": str(out),
        })
        try:
            repeat_record = core._repeat_record(
                plan, baseline, repeat_natural_path, repeat_samples, repeat_natural_ref,
                model_for_render, out, args,
            )
        except Exception as exc:
            repeat_record = {
                "run_id": plan["run_id"], "model_id": plan["generation_repeat_qc"][0]["model_id"],
                "input_sample_id": plan["generation_repeat_qc"][0]["input_sample_id"],
                "face_id": plan["generation_repeat_qc"][0]["face_id"],
                "seed": plan["generation_repeat_qc"][0]["seed"],
                "deterministic": plan["generation_repeat_qc"][0]["deterministic"],
                "followup_protocol": plan["generation_repeat_qc"][0]["followup_protocol"],
                "status": "FAILED", "attempts": [{"status": "FAILED", "failure_reason": f"{type(exc).__name__}: {exc}"}],
                "comparison": {"exact_video": False, "exact_timestamps": False, "exact_query_map": False,
                               "exact_distance": False, "max_frame_absdiff": None,
                               "max_distance_absdiff": None, "abs_delta_Cbest": None},
            }

        core._finish_supports(plan, ordered)
        results_path = out / "results.jsonl"
        with results_path.open("xb") as stream:
            for result in ordered:
                stream.write(core.canonical(public_result(result)) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        repeat_path = out / "generation_repeat_qc.jsonl"
        with repeat_path.open("xb") as stream:
            stream.write(core.canonical(repeat_record) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())

        counts = Counter(result["status"] for result in ordered)
        model_ids = list(dict.fromkeys(row["model_id"] for row in rows))
        method_ids = list(dict.fromkeys(row["method_id"] for row in rows))
        by_method = []
        for model_id in model_ids:
            for method_id in method_ids:
                subset = [result for result in ordered if result["model_id"] == model_id and result["method_id"] == method_id]
                by_method.append({
                    "model_id": model_id, "method_id": method_id, "planned_rows": len(subset),
                    "result_rows": len(subset), "status_counts": {
                        state: sum(result["status"] == state for result in subset)
                        for state in ("PASS", "FAILED", "INCOMPLETE")
                    },
                })
        status = "PASS" if counts["FAILED"] == 0 and repeat_record["status"] in ("PASS", "NONDETERMINISTIC") else "INCOMPLETE"
        manifest = {
            "schema_version": "cross-tfg-audio-eval-v1", "run_id": plan["run_id"], "status": status,
            "plan_sha256": plan["plan_sha256"], "plan_raw_sha256": sha256_file(plan_path),
            "results_jsonl_sha256": sha256_file(results_path),
            "generation_repeat_qc_jsonl_sha256": sha256_file(repeat_path),
            "planned_rows": len(rows), "result_rows": len(ordered),
            "generation_repeat_qc_rows": len(plan.get("generation_repeat_qc", [])),
            "status_counts": {state: int(counts.get(state, 0)) for state in ("PASS", "FAILED", "INCOMPLETE")},
            "counts_by_model_method": by_method,
            "failure_reasons": [{"row_id": item["row_id"], "reason": item.get("failure_reason")}
                                for item in ordered if item["status"] != "PASS"],
            "repeat_qc_status": repeat_record["status"],
        }
        manifest["manifest_sha256"] = sha256_bytes(core.canonical(manifest))
        manifest_path = out / "result_manifest.json"
        core.write_json(manifest_path, manifest)
        write_json(control / "run_status.json", {
            "schema": "ditto_stageb_single_row_recovery_status_v1",
            "status": "COMPLETE" if status == "PASS" else "COMPLETE_WITH_FAILURES",
            "run_id": plan["run_id"], "pid": os.getpid(), "planned_rows": 240,
            "result_rows": len(ordered), "status_counts": manifest["status_counts"],
            "retry_row_id": EXPECTED_FAILED_ROW,
            "retry_row_status": result_by_id[EXPECTED_FAILED_ROW]["status"],
            "repeat_qc_status": repeat_record["status"],
            "result_manifest_path": str(manifest_path), "result_manifest_sha256": sha256_file(manifest_path),
            "output_root": str(out), "control_root": str(control),
        })
        print(json.dumps({"status": status, "planned_rows": len(rows), "status_counts": manifest["status_counts"],
                          "retry_row_status": result_by_id[EXPECTED_FAILED_ROW]["status"],
                          "repeat_qc_status": repeat_record["status"], "output_root": str(out),
                          "manifest_sha256": sha256_file(manifest_path)}, sort_keys=True))
        return 0 if status == "PASS" else 2
    finally:
        core.run_logged = original_run_logged


if __name__ == "__main__":
    raise SystemExit(main())
