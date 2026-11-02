#!/usr/bin/env python3
"""Recover only ENOSPC rows from the frozen shared-frequency fit run.

Previously passed rows are reused only after verifying the original plan,
ordered ledger, result manifest, every file reference and all seven raw score
arrays. Failed planned rows are re-rendered by the unchanged Stage-A core.
The core then recomputes common supports across the complete frozen 360-row
denominator.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CORE_SHA256 = "d12c863ffcd5eacde7b0d09b0abc097429d0318ccea3b059d022d5b2e52e2a94"
ADAPTER_RUNNER_SHA256 = "7f78a14485eebaa20e7afd893a46b2ead26db5544194d988d7d9890d6a43ebbc"
PROFILE_SOURCE_SHA256 = "fd7303f513fdbada80da50e5dd395e9441142f1658544765d7523f1618f55253"
OLD_PLAN_RAW_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
OLD_ROWS_RAW_SHA256 = "2105cd4182757bbc73a5339c7fa7ed49d1027dcac4159c4b84483ba5404c795b"
OLD_RESULTS_RAW_SHA256 = "186525a14cc67c6936fd92ff7855f9dcb8dbe41840ac70cc5b2a8ea04a60cd4b"
OLD_MANIFEST_RAW_SHA256 = "339b98d184a08b3b31145ac2150743e95cd3afd5ae08acc4a95ad050b80e3dfd"
PROFILE_METHOD = "shared_frequency_profile"
ADAPTER_CONTRACT = "cross_tfg_silent_video_cli_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_pinned_module(name: str, path: Path, expected_sha256: str):
    path = path.resolve(strict=True)
    actual = sha256_file(path)
    if len(expected_sha256) != 64 or actual != expected_sha256:
        raise RuntimeError(f"source SHA mismatch for {path}: {actual}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, path


def verify_old_pass_file_refs(value: Any, cache: dict[str, str], row_id: str) -> int:
    checked = 0
    if isinstance(value, Mapping):
        path_value = value.get("path")
        file_sha = value.get("file_sha256")
        if isinstance(path_value, str) and isinstance(file_sha, str):
            path = Path(path_value)
            if not path.is_file():
                raise RuntimeError(f"PASS row {row_id} missing bound file: {path}")
            actual = cache.get(str(path))
            if actual is None:
                actual = sha256_file(path)
                cache[str(path)] = actual
            if len(file_sha) != 64 or actual != file_sha.lower():
                raise RuntimeError(f"PASS row {row_id} file SHA mismatch: {path}")
            size = value.get("bytes")
            if size is not None and int(size) != path.stat().st_size:
                raise RuntimeError(f"PASS row {row_id} file byte-count mismatch: {path}")
            checked += 1
        for nested in value.values():
            checked += verify_old_pass_file_refs(nested, cache, row_id)
    elif isinstance(value, list):
        for nested in value:
            checked += verify_old_pass_file_refs(nested, cache, row_id)
    return checked


def load_verified_score_arrays(result: Mapping[str, Any], core: Any) -> dict[str, np.ndarray]:
    score = result.get("score")
    if not isinstance(score, Mapping) or not isinstance(score.get("raw_matrix_npz"), Mapping):
        raise RuntimeError(f"PASS row {result.get('row_id')} has no raw_matrix_npz reference")
    ref = score["raw_matrix_npz"]
    path = Path(str(ref.get("path", "")))
    if not path.is_file() or sha256_file(path) != ref.get("file_sha256"):
        raise RuntimeError(f"raw score NPZ file SHA mismatch for {result.get('row_id')}")
    metadata = ref.get("arrays")
    if not isinstance(metadata, Mapping) or set(metadata) != set(core.ARRAY_DTYPES):
        raise RuntimeError(f"raw score NPZ array metadata mismatch for {result.get('row_id')}")
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(core.ARRAY_DTYPES):
            raise RuntimeError(f"raw score NPZ keys mismatch for {result.get('row_id')}")
        for key, expected_dtype in core.ARRAY_DTYPES.items():
            array = np.ascontiguousarray(loaded[key])
            info = metadata[key]
            if array.dtype.str != expected_dtype.str:
                raise RuntimeError(f"raw score NPZ dtype mismatch ({key}) for {result.get('row_id')}")
            if list(array.shape) != info.get("shape") or array.dtype.str != info.get("dtype"):
                raise RuntimeError(f"raw score NPZ shape/dtype seal mismatch ({key}) for {result.get('row_id')}")
            actual_array_sha = sha256_bytes(array.tobytes(order="C"))
            if actual_array_sha != info.get("array_sha256"):
                raise RuntimeError(f"raw score NPZ array SHA mismatch ({key}) for {result.get('row_id')}")
            arrays[key] = array
    n = arrays["query_frame_index"].shape[0]
    if arrays["distance_by_query_lag"].shape != (n, 31):
        raise RuntimeError(f"raw score matrix must be N x 31 for {result.get('row_id')}")
    for key in ("query_timestamp_num", "query_timestamp_den", "track_frame_index", "valid_query_mask"):
        if arrays[key].shape != (n,):
            raise RuntimeError(f"raw score query-map length mismatch ({key}) for {result.get('row_id')}")
    if arrays["lag_offsets_frames"].shape != (31,) or not np.array_equal(arrays["lag_offsets_frames"], np.arange(-15, 16, dtype=arrays["lag_offsets_frames"].dtype)):
        raise RuntimeError(f"raw score lag axis mismatch for {result.get('row_id')}")
    return arrays


def read_and_verify_reuse(args: argparse.Namespace, core: Any) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    plan_path = Path(args.plan).resolve(strict=True)
    rows_path = Path(args.planned_rows).resolve(strict=True)
    reuse_root = Path(args.reuse_run).resolve(strict=True)
    old_results_path = reuse_root / "results.jsonl"
    old_manifest_path = reuse_root / "result_manifest.json"
    for path, expected in (
        (plan_path, OLD_PLAN_RAW_SHA256),
        (rows_path, OLD_ROWS_RAW_SHA256),
        (old_results_path, OLD_RESULTS_RAW_SHA256),
        (old_manifest_path, OLD_MANIFEST_RAW_SHA256),
    ):
        actual = sha256_file(path)
        if len(expected) != 64 or actual != expected:
            raise RuntimeError(f"frozen recovery input SHA mismatch for {path}: {actual}")

    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw.decode("utf-8"))
    if plan.get("status") != "FROZEN" or plan.get("plan_sha256") != sha256_bytes(core.canonical({k: v for k, v in plan.items() if k != "plan_sha256"})):
        raise RuntimeError("frozen shared-profile plan payload seal is invalid")
    planned_rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(planned_rows) != 360 or [r["row_id"] for r in planned_rows] != [r["row_id"] for r in plan["rows"]]:
        raise RuntimeError("ordered frozen plan rows are not exactly the planned 360")
    old_results = [json.loads(line) for line in old_results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(old_results) != len(planned_rows) or [r.get("row_id") for r in old_results] != [r["row_id"] for r in planned_rows]:
        raise RuntimeError("old result ledger does not match frozen row order")
    manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    expected_payload = sha256_bytes(core.canonical({k: v for k, v in manifest.items() if k != "manifest_sha256"}))
    if manifest.get("manifest_sha256") != expected_payload:
        raise RuntimeError("old result manifest canonical payload seal is invalid")
    if (manifest.get("planned_rows"), manifest.get("result_rows"), manifest.get("status_counts")) != (360, 360, {"PASS": 174, "FAILED": 186, "INCOMPLETE": 0}):
        raise RuntimeError("old manifest counts do not match the diagnosed 174 PASS / 186 ENOSPC failures")
    if manifest.get("plan_raw_sha256") != OLD_PLAN_RAW_SHA256 or manifest.get("results_jsonl_sha256") != OLD_RESULTS_RAW_SHA256:
        raise RuntimeError("old manifest does not bind the exact frozen plan/results bytes")

    plan_by_id = {r["row_id"]: r for r in planned_rows}
    reused: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    checked_paths: dict[str, str] = {}
    total_checked_refs = 0
    for result in old_results:
        row_id = result["row_id"]
        planned = plan_by_id[row_id]
        for field in core.IDENTITY_FIELDS:
            if result.get(field) != planned.get(field):
                raise RuntimeError(f"old row identity differs from frozen plan: {row_id} ({field})")
        if result.get("status") == "PASS":
            total_checked_refs += verify_old_pass_file_refs(result, checked_paths, row_id)
            arrays = load_verified_score_arrays(result, core)
            reusable = copy.deepcopy(result)
            reusable["_internal_arrays"] = arrays
            reusable["_internal_video_path"] = str(result["render"].get("silent_video_path", ""))
            reusable["_internal_matrix_sha256"] = result["score"]["raw_matrix_npz"]["arrays"]["distance_by_query_lag"]["array_sha256"]
            reused[row_id] = reusable
        elif result.get("status") == "FAILED":
            reason = str(result.get("failure_reason", ""))
            if "no space left on device" not in reason.lower():
                raise RuntimeError(f"old FAILED row is not the diagnosed ENOSPC class: {row_id}: {reason}")
            failures.append(row_id)
        else:
            raise RuntimeError(f"unexpected old row status for recovery: {row_id}: {result.get('status')}")
    if len(reused) != 174 or len(failures) != 186 or len(reused) + len(failures) != 360:
        raise RuntimeError("reuse/retry row counts do not preserve the frozen 360-row denominator")
    return reused, {
        "schema": "cross_tfg_shared_frequency_fit_recovery_reuse_audit_v1",
        "plan_raw_sha256": OLD_PLAN_RAW_SHA256,
        "planned_rows_raw_sha256": OLD_ROWS_RAW_SHA256,
        "old_results_raw_sha256": OLD_RESULTS_RAW_SHA256,
        "old_result_manifest_raw_sha256": OLD_MANIFEST_RAW_SHA256,
        "old_run_path": str(reuse_root),
        "old_pass_reused": len(reused),
        "enospc_rows_retried": len(failures),
        "planned_denominator": 360,
        "file_refs_sha_verified": total_checked_refs,
        "unique_files_sha_verified": len(checked_paths),
        "unique_files_bytes_verified": sum(Path(path).stat().st_size for path in checked_paths),
        "old_pass_matrix_npz_verified": len(reused),
        "retry_row_ids": failures,
        "status": "PASS_REUSE_INPUT_AUDIT",
    }


def _load_profile_module(profile_source: Path):
    profile_source = profile_source.resolve(strict=True)
    if sha256_file(profile_source) != PROFILE_SOURCE_SHA256:
        raise RuntimeError("shared frequency transform source SHA differs from the frozen source")
    repo_root = str(profile_source.parent.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from scripts.audio_enhancement import shared_frequency_profile_v1
    return shared_frequency_profile_v1


def profile_transform(samples: np.ndarray, method: Mapping[str, Any], profile_module: Any) -> tuple[np.ndarray, dict[str, Any]]:
    config = method.get("config")
    if not isinstance(config, Mapping) or config.get("method") != PROFILE_METHOD:
        raise ValueError("profile transform requires the frozen shared_frequency_profile configuration")
    output, diagnostics = profile_module.transform_shared_profile(
        samples, str(config.get("profile_id", "")),
        sample_rate_hz=int(config.get("sample_rate_hz", 16000)),
        target_rms=float(config["target_rms"]),
        peak_ceiling=float(config["peak_ceiling"]),
    )
    output = np.ascontiguousarray(output, dtype=np.float32)
    if output.shape != samples.shape or not np.isfinite(output).all():
        raise RuntimeError("shared profile did not preserve finite float32 sample count")
    info = dict(diagnostics)
    info.update({"operation": "pinned_shared_frequency_profile_v1", "profile_source_sha256": PROFILE_SOURCE_SHA256,
                 "profile_id": str(config.get("profile_id", ""))})
    return output, info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stagea-core", required=True)
    parser.add_argument("--adapter-runner", required=True)
    parser.add_argument("--profile-source", required=True)
    parser.add_argument("--reuse-run", required=True)
    parser.add_argument("--ditto-control-dir", required=True)
    own, remaining = parser.parse_known_args(argv)

    control_dir = Path(own.ditto_control_dir).resolve()
    control_dir.mkdir(parents=True, exist_ok=False)
    temp_root = control_dir / "tmp"
    cache_root = control_dir / "cache"
    temp_root.mkdir()
    cache_root.mkdir()
    for name in ("TMPDIR", "TMP", "TEMP"):
        os.environ[name] = str(temp_root)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    os.environ["MPLCONFIGDIR"] = str(cache_root / "matplotlib")

    core, core_path = load_pinned_module("cross_tfg_recovery_stagea_core", Path(own.stagea_core), CORE_SHA256)
    adapter_runner, adapter_path = load_pinned_module("cross_tfg_recovery_adapter_runner", Path(own.adapter_runner), ADAPTER_RUNNER_SHA256)
    profile_module = _load_profile_module(Path(own.profile_source))
    stagea_args = core.build_parser().parse_args(remaining)
    if str(Path(stagea_args.output_root).resolve()) == str(Path(own.reuse_run).resolve()):
        raise RuntimeError("recovery output must be a new root, never the failed source run")
    if Path(stagea_args.output_root).exists() and any(Path(stagea_args.output_root).iterdir()):
        raise RuntimeError("recovery output root is not empty")

    reused, audit = read_and_verify_reuse(stagea_args, core)
    audit.update({
        "new_recovery_run_path": str(Path(stagea_args.output_root).resolve()),
        "stagea_core_path": str(Path(own.stagea_core).resolve()),
        "stagea_core_sha256": CORE_SHA256,
        "adapter_runner_path": str(Path(own.adapter_runner).resolve()),
        "adapter_runner_sha256": ADAPTER_RUNNER_SHA256,
        "profile_source_path": str(Path(own.profile_source).resolve()),
        "profile_source_sha256": PROFILE_SOURCE_SHA256,
        "ditto_control_override_path": str(control_dir / "ditto_gpu_gate"),
        "temp_root": str(temp_root),
        "tempfile_root": __import__("tempfile").gettempdir(),
        "phase": "PREFLIGHT_VERIFIED_BEFORE_GENERATION",
    })
    (control_dir / "recovery_preflight_audit.json").write_text(
        json.dumps(audit, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    original_transform = core._transform
    original_render_and_score = core._render_and_score

    def transform(samples: np.ndarray, method: Mapping[str, Any]):
        config = method.get("config", {})
        if isinstance(config, Mapping) and config.get("method") == PROFILE_METHOD:
            return profile_transform(samples, method, profile_module)
        return original_transform(samples, method)

    def generator(**kwargs):
        model = kwargs["model"]
        model_generator = model.get("generator", {})
        if isinstance(model_generator, Mapping) and model_generator.get("contract") == ADAPTER_CONTRACT:
            runtime_model = model
            if kwargs["row"].get("model_id") == "ditto_hubert_fixed_portrait_attempt18":
                runtime_model = copy.deepcopy(model)
                runtime_env = runtime_model["generator"].setdefault("env", {})
                runtime_env["CROSS_TFG_CONTROL_DIR"] = str(control_dir / "ditto_gpu_gate")
                Path(runtime_env["CROSS_TFG_CONTROL_DIR"]).mkdir(parents=True, exist_ok=True)
            return adapter_runner._adapter_generator(core, **{**kwargs, "model": runtime_model})
        return original_run_generator(**kwargs)

    original_run_generator = core._run_generator

    def render_and_score(row: Mapping[str, Any], model: Mapping[str, Any], method: Mapping[str, Any] | None,
                         natural_path: Path, natural_samples: np.ndarray, natural_ref: Mapping[str, Any],
                         row_root: Path, args: argparse.Namespace, *, qc_repeat: bool = False) -> dict[str, Any]:
        if qc_repeat:
            # Repeat QC is a fresh render by definition; never reuse the
            # already-scored identity result as its own repeat.
            return original_render_and_score(row, model, method, natural_path, natural_samples, natural_ref, row_root, args, qc_repeat=True)
        prior = reused.get(str(row["row_id"]))
        if prior is not None:
            if prior.get("status") != "PASS" or method is None:
                raise RuntimeError("recovery reuse attempted for a non-PASS/nonplanned row")
            return copy.deepcopy(prior)
        if str(row["row_id"]) not in audit["retry_row_ids"]:
            raise RuntimeError(f"new rendering refused for an unlisted row: {row['row_id']}")
        return original_render_and_score(row, model, method, natural_path, natural_samples, natural_ref, row_root, args, qc_repeat=qc_repeat)

    core._transform = transform
    core._run_generator = generator
    core._render_and_score = render_and_score
    return_code = core.run(stagea_args)
    audit.update({
        "new_recovery_run_path": str(Path(stagea_args.output_root).resolve()),
        "stagea_core_path": str(core_path), "stagea_core_sha256": sha256_file(core_path),
        "adapter_runner_path": str(adapter_path), "adapter_runner_sha256": sha256_file(adapter_path),
        "profile_source_path": str(Path(own.profile_source).resolve()), "profile_source_sha256": PROFILE_SOURCE_SHA256,
        "ditto_control_override_path": str(control_dir / "ditto_gpu_gate"),
        "temp_root": str(temp_root), "tempfile_root": __import__("tempfile").gettempdir(),
        "core_return_code": int(return_code),
    })
    audit_path = Path(stagea_args.output_root) / "recovery_reuse_audit.json"
    audit_path.write_text(json.dumps(audit, sort_keys=True, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
