from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .assets import freeze_inputs
from .common import (
    SYNCNET_REQUEST_EXEC_CODE,
    ProtocolError,
    canonical_json_sha256,
    file_sha256,
    read_json,
    verify_json,
    write_json,
)
from .config import build_run_fingerprint, load_config, validate_run_id
from .generation import render_baselines
from .retime import build_map, mirror_map, render_map, render_nearest


STAGES = ("audit", "render", "calibrate", "search", "seal", "transfer", "official", "check", "report")


def _run_dir(config: Mapping[str, Any], run_id: str) -> Path:
    return (Path(config["paths"]["output_root"]).resolve() / validate_run_id(run_id)).resolve()


def _git_commit(repo: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _runtime(config: Mapping[str, Any]) -> dict[str, Any]:
    code = "import importlib, json, platform, sys; names=['torch','numpy','cv2','scipy','python_speech_features','librosa']; out={'python':sys.executable,'python_version':platform.python_version(),'packages':{}};\nfor name in names:\n try:\n  m=importlib.import_module(name); out['packages'][name]=str(getattr(m,'__version__','unknown'))\n except Exception as e: out['packages'][name]=type(e).__name__+': '+str(e)\ntry:\n import torch; out['cuda_available']=torch.cuda.is_available(); out['cuda_devices']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]\nexcept Exception as e: out['cuda_error']=type(e).__name__+': '+str(e)\nprint(json.dumps(out))"
    values: dict[str, Any] = {}
    for key in ("wav2lip_python", "syncnet_python"):
        executable = Path(config["paths"][key]).absolute()
        result = subprocess.run([str(executable), "-c", code], cwd=config["repo_root"], capture_output=True, text=True, check=False)
        if result.returncode:
            raise ProtocolError(f"ENVIRONMENT_PROBE_FAILED:{key}:{result.stderr[-500:]}")
        try:
            values[key] = json.loads(result.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise ProtocolError(f"ENVIRONMENT_PROBE_INVALID:{key}") from exc
        required = {"torch", "numpy", "cv2", "scipy"}
        required.add("librosa" if key == "wav2lip_python" else "python_speech_features")
        missing = {name: values[key].get("packages", {}).get(name) for name in required
                   if str(values[key].get("packages", {}).get(name, "")).startswith(("ModuleNotFoundError", "ImportError"))}
        if missing:
            raise ProtocolError(f"ENVIRONMENT_PACKAGES_MISSING:{key}:{missing}")
        if str(config["models"].get("device", "cuda")).startswith("cuda") and values[key].get("cuda_available") is not True:
            raise ProtocolError(f"ENVIRONMENT_CUDA_UNAVAILABLE:{key}")
    executables = {}
    for key in ("ffmpeg", "ffprobe", "wav2lip_python", "syncnet_python"):
        path = Path(config["paths"][key])
        invocation_path = path.absolute() if key in {"wav2lip_python", "syncnet_python"} else path.resolve()
        executables[key] = {"path": str(invocation_path), "resolved_path": str(path.resolve()),
                            "sha256": file_sha256(path), "version": _version(invocation_path)}
    return {"host_python": sys.version, "host_platform": platform.platform(), "git_commit": _git_commit(Path(config["repo_root"])),
            "worker_environments": values, "executables": executables}


def _version(path: Path) -> str:
    args = [str(path), "-version"] if path.name in {"ffmpeg", "ffprobe"} else [str(path), "--version"]
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return (result.stdout or result.stderr).splitlines()[0] if result.stdout or result.stderr else f"returncode={result.returncode}"


def _estimate_output_bytes(config: Mapping[str, Any], frozen: Mapping[str, Any], *, smoke: bool) -> int:
    record_count = len(frozen["records"])
    portrait_count = 1 if smoke else 3
    # FFV1 is lossless; a conservative 0.25 raw-pixel estimate plus the full
    # official mux matrix leaves a 1.2 safety reserve without assuming RGB raw size.
    total_frames = sum(int(row["audio"]["sample_count"]) + 639 for row in frozen["records"])
    frames = total_frames // 640
    video_count = record_count * (2 * portrait_count + 3 + 2 * (portrait_count - 1))
    mux_count = record_count * (8 + 6 * (portrait_count - 1))
    width = max(int(row["width"]) for row in frozen["portrait_bindings"]["portraits"].values())
    height = max(int(row["height"]) for row in frozen["portrait_bindings"]["portraits"].values())
    audio_bytes = sum(int(row["audio"]["sample_count"]) * 2 for row in frozen["records"]) * mux_count // max(1, record_count)
    return int(video_count * frames / max(1, record_count) * width * height * 3 * 0.25 + audio_bytes)


def _available_ram_bytes(meminfo_path: Path = Path("/proc/meminfo")) -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:
        for line in meminfo_path.read_text(encoding="ascii").splitlines():
            fields = line.split()
            if fields and fields[0] == "MemAvailable:" and len(fields) >= 2:
                scale = 1024 if len(fields) >= 3 and fields[2].lower() == "kb" else 1
                return int(fields[1]) * scale
    except (OSError, ValueError):
        pass
    return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


def _resource_preflight(config: Mapping[str, Any], run_dir: Path, *, require_gpu: bool) -> dict[str, Any]:
    budget = config["budget"]
    usage = shutil.disk_usage(run_dir.parent)
    available_ram = _available_ram_bytes()
    result: dict[str, Any] = {"disk_free_bytes": int(usage.free), "available_ram_bytes": available_ram,
                              "minimum_disk_bytes": int(budget["minimum_free_disk_bytes"]),
                              "minimum_ram_bytes": int(budget["minimum_available_ram_bytes"]),
                              "status": "READY"}
    if usage.free < int(budget["minimum_free_disk_bytes"]):
        result.update(status="RESOURCE_WAIT", reason="DISK_BELOW_MINIMUM")
        return result
    if available_ram < int(budget["minimum_available_ram_bytes"]):
        result.update(status="RESOURCE_WAIT", reason="RAM_BELOW_MINIMUM")
        return result
    if require_gpu:
        nvidia = shutil.which("nvidia-smi")
        if not nvidia:
            result.update(status="RESOURCE_WAIT", reason="NVIDIA_SMI_UNAVAILABLE")
            return result
        query = subprocess.run([nvidia, "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, check=False)
        if query.returncode:
            result.update(status="RESOURCE_WAIT", reason="GPU_QUERY_FAILED")
            return result
        gpu_rows = [row.strip().split(",") for row in query.stdout.splitlines() if row.strip()]
        selected = next((row for row in gpu_rows if int(row[0].strip()) == 0), None)
        result["gpu_rows"] = gpu_rows
        if selected is None:
            result.update(status="RESOURCE_WAIT", reason="GPU_NOT_FOUND")
            return result
        if int(selected[1].strip()) < int(budget["minimum_free_gpu_mib"]):
            result.update(status="RESOURCE_WAIT", reason="GPU_LOW_FREE_MEMORY")
            return result
        processes = subprocess.run([nvidia, "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, check=False)
        if processes.returncode:
            result.update(status="RESOURCE_WAIT", reason="GPU_PROCESS_QUERY_FAILED")
            return result
        pids = [line.strip() for line in processes.stdout.splitlines() if line.strip()]
        result["compute_processes"] = pids
        if pids:
            result.update(status="RESOURCE_WAIT", reason="EXTERNAL_COMPUTE_PROCESS_PRESENT")
            return result
        if int(selected[2].strip()) > 10:
            result.update(status="RESOURCE_WAIT", reason="GPU_HIGH_UTILIZATION")
            return result
    return result


def _wait_for_resource_preflight(
    config: Mapping[str, Any], run_dir: Path, *, require_gpu: bool,
    max_wait_seconds: float = 30.0, poll_interval_seconds: float = 1.0,
) -> dict[str, Any]:
    """Wait briefly for utilization left by our prior cell to settle.

    A live compute process, low memory, or any other resource failure remains
    a typed RESOURCE_WAIT. Only a high utilization sample with no compute
    process is polled, and the wait is bounded; external work is never stopped.
    """
    started = time.monotonic()
    result = _resource_preflight(config, run_dir, require_gpu=require_gpu)
    polls = 0
    while result.get("status") != "READY" and result.get("reason") == "GPU_HIGH_UTILIZATION":
        remaining = max_wait_seconds - (time.monotonic() - started)
        if remaining <= 0:
            break
        time.sleep(min(poll_interval_seconds, remaining))
        polls += 1
        result = _resource_preflight(config, run_dir, require_gpu=require_gpu)
    result["cooldown_wait_seconds"] = round(time.monotonic() - started, 3)
    result["cooldown_poll_count"] = polls
    return result


def _request(config: Mapping[str, Any], frozen: Mapping[str, Any], generated: Mapping[str, Any],
             run_dir: Path, record: Mapping[str, Any], portrait_id: str, mode: str,
             run_fingerprint: str) -> dict[str, Any]:
    image = frozen["portrait_bindings"]["portraits"][portrait_id]
    portrait_generation = generated["portraits"][portrait_id]
    m_receipt = portrait_generation["M"]
    m_video = Path(m_receipt["canonical_video"]).resolve()
    request_core = {"run_fingerprint": run_fingerprint, "sample_id": str(record["sample_id"]),
                    "paired_key": str(record["paired_key"]), "speaker_id": str(record["speaker_id"]),
                    "portrait_id": portrait_id, "M_video_sha256": file_sha256(m_video),
                    "N_audio_sha256": str(record["audio"]["natural"]["sha256"]),
                    "score_box_xyxy": image["score_box"]["box"], "valid_frame_count": int(portrait_generation["valid_frame_count"]),
                    "syncnet_model_sha256": file_sha256(config["paths"]["syncnet_model"]),
                    "legacy_score_worker_sha256": file_sha256(config["paths"]["legacy_score_worker"])}
    request_fingerprint = canonical_json_sha256(request_core)
    state_base = run_dir / "03_search" / str(record["sample_id"])
    request = {
        "mode": ("search_v2" if mode == "search" and str(config.get("protocol", "")).endswith("_v2") else mode), "repo_root": str(Path(config["repo_root"]).resolve()),
        "syncnet_root": config["paths"]["syncnet_root"], "syncnet_model": config["paths"]["syncnet_model"],
        "expected_syncnet_sha256": config["models"]["syncnet_model_sha256"],
        "legacy_score_worker": config["paths"]["legacy_score_worker"],
        "sample_id": str(record["sample_id"]), "paired_key": str(record["paired_key"]),
        "speaker_id": str(record["speaker_id"]), "portrait_id": portrait_id,
        "video_m": str(m_video), "video_n": portrait_generation["N"]["canonical_video"],
        "audio_n": record["audio"]["natural"]["path"], "audio_m": record["audio"]["mfa_linear"]["path"],
        "score_box_xyxy": image["score_box"]["box"],
        "frame_count": int(portrait_generation["frame_count"]),
        "valid_frame_count": int(portrait_generation["valid_frame_count"]),
        "identity_map": build_map(int(portrait_generation["frame_count"]), int(portrait_generation["valid_frame_count"])),
        "request_fingerprint": request_fingerprint,
        "state_path": str((state_base / "state.json").resolve()),
        "result_path": str((state_base / "result.json").resolve()),
        "work_dir": str((state_base / "work").resolve()),
        "embedding_dir": str((state_base / "embeddings").resolve()),
        "batch_size": int(config["models"]["syncnet_batch_size"]),
        "search": dict(config["search"]),
    }
    return request


def _syncnet_request(config: Mapping[str, Any], request: Mapping[str, Any], *, log: Path,
                     timeout_seconds: float | None = None) -> dict[str, Any]:
    request_path = Path(str(request["state_path"])).parent / f"{request['mode']}.request.json"
    write_json(request_path, request)
    command = [str(Path(config["paths"]["syncnet_python"]).absolute()), "-c", SYNCNET_REQUEST_EXEC_CODE, str(request_path)]
    log.parent.mkdir(parents=True, exist_ok=True)
    configured_timeout = int(config.get("timeouts", {}).get("syncnet_seconds", 2400))
    timeout = configured_timeout if timeout_seconds is None else min(configured_timeout, int(timeout_seconds))
    if timeout < 1:
        raise ProtocolError("BUDGET_LIMITED:TOTAL_ACTIVE_GPU_BUDGET_EXHAUSTED")
    budget_limited_timeout = timeout_seconds is not None and timeout < configured_timeout
    with log.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(command, ensure_ascii=False) + "\n")
        try:
            result = subprocess.run(command, cwd=str(Path(config["repo_root"]).resolve()), stdout=handle,
                                    stderr=subprocess.STDOUT, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            handle.write(f"\n[TIMEOUT] after {timeout} seconds\n")
            if budget_limited_timeout:
                raise ProtocolError("BUDGET_LIMITED:TOTAL_ACTIVE_GPU_BUDGET_EXHAUSTED") from exc
            raise ProtocolError(f"SYNCNET_WORKER_TIMEOUT:{request['mode']}:{request.get('sample_id')}") from exc
        handle.write(f"\n[returncode] {result.returncode}\n")
    if result.returncode:
        raise ProtocolError(f"SYNCNET_WORKER_FAILED:{request['mode']}:{request.get('sample_id')}")
    output = Path(str(request["result_path"]))
    return verify_json(output, self_hash=True)


def _protocol_for(config: Mapping[str, Any], run_id: str, *, smoke: bool, frozen: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    frozen_sha = canonical_json_sha256({key: value for key, value in frozen.items() if key != "artifact_sha256"})
    config_view = deepcopy(dict(config))
    config_view["sample_ids"] = list(frozen["sample_ids"])
    config_view["portraits"] = ["3"] if smoke else list(config_view["portraits"])
    _base_fingerprint, bindings = build_run_fingerprint(config_view, frozen_sha)
    bindings["runtime"] = dict(runtime)
    fingerprint = canonical_json_sha256(bindings)
    return {"schema_version": 1, "protocol": config["protocol"], "protocol_version": config["protocol"],
            "run_id": run_id, "run_fingerprint": fingerprint, "run_bindings": bindings,
            "engineering_only": bool(smoke), "scientific_status": "NOT_RUN",
            "sample_ids": list(frozen["sample_ids"]), "portrait_ids": ["3"] if smoke else list(config["portraits"]),
            "frozen_inputs_sha256": frozen.get("artifact_sha256"), "runtime": dict(runtime),
            "budget": {"total_gpu_seconds": int(config["budget"]["total_gpu_seconds"]), "active_gpu_seconds": 0.0,
                       "resource_wait_seconds": 0.0}, "stage_status": {}}


def stage_audit(config: Mapping[str, Any], run_dir: Path, *, run_id: str, smoke: bool) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = run_dir / "00_protocol" / "protocol.json"
    frozen_path = run_dir / "00_protocol" / "frozen_inputs.json"
    sample_ids = ["1"] if smoke else list(config["sample_ids"])
    cohort_dir = run_dir / "00_cohort" if config["protocol"].endswith("_v2") else None
    if cohort_dir is not None:
        from .cohort_v2 import SAMPLE_IDS
        cohort = verify_json(cohort_dir / "cohort_v2.json", self_hash=True)
        summary = verify_json(cohort_dir / "mfa_summary_v2.json", self_hash=True)
        if cohort.get("sample_ids") != list(SAMPLE_IDS) or summary.get("status") != "COMPLETE":
            raise ProtocolError("INPUT_INCOMPLETE:MFA64 cohort preparation required before audit")
    if protocol_path.is_file() and frozen_path.is_file():
        protocol = verify_json(protocol_path, self_hash=True)
        frozen = verify_json(frozen_path, self_hash=True)
        current_runtime = _runtime(config)
        current_frozen = freeze_inputs(config, sample_ids=sample_ids, cohort_dir=cohort_dir)
        if canonical_json_sha256({k: v for k, v in frozen.items() if k != "artifact_sha256"}) != canonical_json_sha256(current_frozen):
            raise ProtocolError("FROZEN_INPUTS_CHANGED_DURING_AUDIT_RESUME")
        current = _protocol_for(config, run_id, smoke=smoke, frozen=frozen, runtime=current_runtime)
        if current["run_fingerprint"] != protocol.get("run_fingerprint"):
            raise ProtocolError("RUN_FINGERPRINT_MISMATCH: current source/assets/config changed")
        if list(frozen.get("sample_ids", [])) != sample_ids:
            raise ProtocolError("RUN_COHORT_MISMATCH")
        return {"protocol": protocol, "frozen_inputs": frozen, "resumed": True}
    partial_paths = list(run_dir.rglob("*")) if run_dir.exists() else []
    allowed_partial_paths = {run_dir / "00_protocol", frozen_path}
    unexpected_partial_paths = [path for path in partial_paths if path not in allowed_partial_paths
                                and (cohort_dir is None or path != cohort_dir and cohort_dir not in path.parents)]
    if protocol_path.exists() or (not frozen_path.is_file() and any(path != cohort_dir for path in run_dir.iterdir())) or unexpected_partial_paths:
        raise ProtocolError("NONEMPTY_RUN_WITHOUT_VALID_PROTOCOL")
    if frozen_path.is_file():
        frozen = verify_json(frozen_path, self_hash=True)
        current_frozen = freeze_inputs(config, sample_ids=sample_ids, cohort_dir=cohort_dir)
        frozen_core = dict(frozen)
        frozen_core.pop("artifact_sha256", None)
        if canonical_json_sha256(frozen_core) != canonical_json_sha256(current_frozen):
            raise ProtocolError("FROZEN_INPUTS_CHANGED_DURING_AUDIT_RESUME")
    else:
        frozen = freeze_inputs(config, sample_ids=sample_ids, cohort_dir=cohort_dir)
        frozen = write_json(frozen_path, frozen, self_hash=True)
    runtime = _runtime(config)
    protocol = _protocol_for(config, run_id, smoke=smoke, frozen=frozen, runtime=runtime)
    if cohort_dir is not None:
        ledger_path = cohort_dir / "gpu_budget.json"
        ledger = verify_json(ledger_path, self_hash=True)
        protocol["budget"]["active_gpu_seconds"] = float(ledger["active_gpu_seconds"])
        protocol["stage_active_gpu_seconds"] = {"cohort": float(ledger["active_gpu_seconds"])}
        protocol["cohort_budget_sha256"] = file_sha256(ledger_path)
    resources = _resource_preflight(config, run_dir, require_gpu=False)
    estimated = _estimate_output_bytes(config, frozen, smoke=smoke)
    resources["estimated_output_bytes"] = estimated
    resources["required_output_bytes_with_safety"] = int(estimated * float(config["budget"]["output_safety_multiplier"]))
    if resources["disk_free_bytes"] < resources["required_output_bytes_with_safety"]:
        resources.update(status="RESOURCE_WAIT", reason="INSUFFICIENT_ESTIMATED_OUTPUT_SPACE")
    protocol["preflight"] = resources
    protocol["stage_status"]["audit"] = "PASS" if resources["status"] == "READY" else resources["status"]
    write_json(protocol_path, protocol, self_hash=True)
    write_json(run_dir / "00_protocol" / "resource_preflight.json", resources, self_hash=True)
    return {"protocol": protocol, "frozen_inputs": frozen, "resumed": False}


def _check_budget(protocol: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    active = float(protocol.get("budget", {}).get("active_gpu_seconds", 0.0))
    if active >= float(config["budget"]["total_gpu_seconds"]):
        raise ProtocolError("TOTAL_ACTIVE_GPU_BUDGET_EXHAUSTED")


def _remaining_gpu_seconds(protocol: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    total = float(config["budget"]["total_gpu_seconds"])
    active = float(protocol.get("budget", {}).get("active_gpu_seconds", 0.0))
    return max(0.0, total - active)


def _timed_stage(protocol_path: Path, protocol: dict[str, Any], stage: str, operation: Any) -> Any:
    started = time.monotonic()
    try:
        return operation()
    finally:
        elapsed = time.monotonic() - started
        protocol.setdefault("budget", {})["active_gpu_seconds"] = float(protocol.get("budget", {}).get("active_gpu_seconds", 0.0)) + elapsed
        protocol.setdefault("stage_active_gpu_seconds", {})[stage] = float(protocol.get("stage_active_gpu_seconds", {}).get(stage, 0.0)) + elapsed
        write_json(protocol_path, protocol, self_hash=True)


def stage_render(config: dict[str, Any], run_dir: Path, frozen: Mapping[str, Any], protocol: dict[str, Any], *, smoke: bool) -> dict[str, Any]:
    path = run_dir / "01_generation" / "manifest.json"
    if path.is_file():
        old = verify_json(path, self_hash=True)
        if old.get("status") == "COMPLETE":
            return old
    _check_budget(protocol, config)
    resource = _resource_preflight(config, run_dir, require_gpu=True)
    protocol.setdefault("stage_status", {})["render_preflight"] = resource["status"]
    if resource["status"] != "READY":
        write_json(run_dir / "00_protocol" / "render_resource_wait.json", resource, self_hash=True)
        raise ProtocolError(f"RESOURCE_WAIT:{resource.get('reason')}")
    config = deepcopy(config)
    config["portraits"] = ["3"] if smoke else list(config["portraits"])
    remaining = _remaining_gpu_seconds(protocol, config)
    if remaining < 1:
        raise ProtocolError("BUDGET_LIMITED:TOTAL_ACTIVE_GPU_BUDGET_EXHAUSTED")
    config.setdefault("timeouts", {})["wav2lip_seconds"] = min(600, max(1, int(remaining)))
    deadline = time.monotonic() + remaining
    result = _timed_stage(run_dir / "00_protocol" / "protocol.json", protocol, "render",
                          lambda: render_baselines(config, frozen, run_dir, budget_deadline=deadline))
    return result


def _generation(config: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    return verify_json(run_dir / "01_generation" / "manifest.json", self_hash=True)


def stage_calibrate_search(config: Mapping[str, Any], run_dir: Path, frozen: Mapping[str, Any], protocol: dict[str, Any], *, mode: str) -> dict[str, Any]:
    generated = _generation(config, run_dir)
    for record in frozen["records"]:
        generated_record = next(row for row in generated["records"] if str(row["sample_id"]) == str(record["sample_id"]))
        request = _request(config, {**frozen, "portrait_bindings": frozen["portrait_bindings"]}, generated_record,
                           run_dir, record, "3", mode, str(protocol["run_fingerprint"]))
        state_path = Path(request["state_path"])
        if mode == "calibrate" and state_path.is_file():
            try:
                prior = verify_json(state_path, self_hash=True)
                if prior.get("request_fingerprint") == request["request_fingerprint"] and prior.get("status") in {"CALIBRATED", "SEARCHING", "BUDGET_LIMITED", "SEARCH_COMPLETE"}:
                    continue
            except Exception:
                pass
        _check_budget(protocol, config)
        resource = _resource_preflight(config, run_dir, require_gpu=True)
        if resource["status"] != "READY":
            raise ProtocolError(f"RESOURCE_WAIT:{resource.get('reason')}")
        remaining = _remaining_gpu_seconds(protocol, config)
        if remaining < 1:
            raise ProtocolError("BUDGET_LIMITED:TOTAL_ACTIVE_GPU_BUDGET_EXHAUSTED")
        timeout = min(int(config.get("timeouts", {}).get("syncnet_seconds", 2400)), max(1, int(remaining)))
        _timed_stage(run_dir / "00_protocol" / "protocol.json", protocol, mode,
                     lambda req=request, limit=timeout: _syncnet_request(config, req,
                         log=run_dir / "02_calibration" / f"{mode}_{req['sample_id']}.log", timeout_seconds=limit))
        worker_result = verify_json(request["result_path"], self_hash=True)
        if mode == "calibrate" and worker_result.get("status") != "CALIBRATED":
            raise ProtocolError(f"CALIBRATION_FAILED:{record['sample_id']}:{worker_result.get('status')}")
        allowed_search = ({"ALIGNMENT_GAIN", "ALIGNMENT_AND_C_GAIN", "NO_ACCEPTABLE_WARP", "BUDGET_LIMITED"}
                          if config["protocol"].endswith("_v2") else {"ACCEPTED_WARP", "NO_ACCEPTABLE_WARP", "BUDGET_LIMITED"})
        if mode == "search" and worker_result.get("status") not in allowed_search:
            raise ProtocolError(f"SEARCH_FAILED:{record['sample_id']}:{worker_result.get('status')}")
    return {"status": "COMPLETE", "mode": mode, "sample_count": len(frozen["records"])}


def _search_results(run_dir: Path, frozen: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in frozen["records"]:
        path = run_dir / "03_search" / str(record["sample_id"]) / "result.json"
        rows.append(verify_json(path, self_hash=True))
    return rows


def _save_ffv1(config: Mapping[str, Any], frames: np.ndarray, path: Path, expected: np.ndarray) -> dict[str, Any]:
    worker = __import__("scripts.experiments.static_image_bridge.render_worker", fromlist=["encode_ffv1_stream"])
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = worker.encode_ffv1_stream(frames, width=int(frames.shape[2]), height=int(frames.shape[1]),
                                        output=path, ffmpeg=Path(config["paths"]["ffmpeg"]).resolve())
    from .scorer import read_video_frames
    decoded, metadata = read_video_frames(path)
    if not np.array_equal(decoded, expected):
        raise ProtocolError(f"SEALED_FFV1_PIXEL_MISMATCH:{path}")
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "frame_count": int(decoded.shape[0]),
            "frames_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(), "encoded": encoded,
            "video_metadata": metadata}


def stage_seal(config: Mapping[str, Any], run_dir: Path, frozen: Mapping[str, Any], *, smoke: bool) -> dict[str, Any]:
    output = run_dir / "04_sealed" / "manifest.json"
    if output.is_file():
        sealed = verify_json(output, self_hash=True)
        if sealed.get("status") == "SEALED":
            return sealed
    generated = _generation(config, run_dir)
    results = _search_results(run_dir, frozen)
    result_by_sample = {str(row["sample_id"]): row for row in results}
    generated_by_sample = {str(row["sample_id"]): row for row in generated["records"]}
    records: list[dict[str, Any]] = []
    for record in frozen["records"]:
        sid = str(record["sample_id"])
        gen = generated_by_sample[sid]
        search = result_by_sample[sid]
        portrait3 = gen["portraits"]["3"]
        source, _ = __import__("scripts.experiments.mfa_linear_video_retiming.scorer", fromlist=["read_video_frames"]).read_video_frames(portrait3["M"]["canonical_video"])
        selected = search["selected"]
        if config["protocol"].endswith("_v2") and selected.get("score_kind") != "true_fixed_crop":
            raise ProtocolError(f"V2_SEAL_REQUIRES_TRUE_FIXED_CROP:{sid}")
        mapping = dict(selected["map"])
        retimed, interpolation_audit = render_map(source, mapping)
        nearest = render_nearest(source, mapping)
        identity = build_map(source.shape[0], int(portrait3["valid_frame_count"]))
        global_map = build_map(source.shape[0], int(portrait3["valid_frame_count"]))
        global_control = search.get("global_control") or {}
        if global_control.get("map"):
            global_map = dict(global_control["map"])
        mirror: dict[str, Any] | None
        mirror_frames: np.ndarray | None
        try:
            mirror = mirror_map(mapping)
            mirror_frames, _ = render_map(source, mirror)
        except ProtocolError:
            mirror, mirror_frames = None, None
        base = run_dir / "04_sealed" / "videos" / "3" / sid
        video_rows = {
            "N": {"path": portrait3["N"]["canonical_video"], "sha256": file_sha256(portrait3["N"]["canonical_video"]), "map": None},
            "M": {"path": portrait3["M"]["canonical_video"], "sha256": file_sha256(portrait3["M"]["canonical_video"]), "map": identity},
            "R": _save_ffv1(config, retimed, base / "R.mkv", retimed),
            "GLOBAL": _save_ffv1(config, render_map(source, global_map)[0], base / "GLOBAL.mkv", render_map(source, global_map)[0]),
            "NEAREST": _save_ffv1(config, nearest, base / "NEAREST.mkv", nearest),
            "MIRROR": (None if mirror_frames is None else _save_ffv1(config, mirror_frames, base / "MIRROR.mkv", mirror_frames)),
        }
        records.append({"sample_id": sid, "paired_key": record["paired_key"], "speaker_id": record["speaker_id"],
                        "transcript": record["transcript"], "portrait_id": "3", "frame_count": int(source.shape[0]),
                        "valid_frame_count": int(portrait3["valid_frame_count"]), "source_M": portrait3["M"],
                        "source_N": portrait3["N"], "search_result_path": str((run_dir / "03_search" / sid / "result.json").resolve()),
                        "selected_map": mapping, "map_sha256": mapping["map_sha256"],
                        "interpolation_audit": interpolation_audit, "selected_metrics": selected["metrics"],
                        "selected_score_kind": selected.get("score_kind", "true_fixed_crop"),
                        "baseline_metrics": search["baseline"], "search_status": search["status"],
                        "global_control": global_control, "videos": video_rows})
    payload = {"schema_version": 1, "status": "SEALED", "sealed_before_official": True,
               "engineering_only": bool(smoke), "sample_count": len(records), "records": records,
               "model_sha256": {"wav2lip": file_sha256(config["paths"]["wav2lip_checkpoint"]),
                                "syncnet": file_sha256(config["paths"]["syncnet_model"])},
               "run_fingerprint": verify_json(run_dir / "00_protocol" / "protocol.json", self_hash=True)["run_fingerprint"]}
    write_json(output, payload, self_hash=True)
    return verify_json(output, self_hash=True)


def stage_transfer(config: Mapping[str, Any], run_dir: Path, frozen: Mapping[str, Any], *, smoke: bool) -> dict[str, Any]:
    path = run_dir / "05_transfer" / "manifest.json"
    if path.is_file():
        prior = verify_json(path, self_hash=True)
        if prior.get("status") == "COMPLETE":
            return prior
    sealed = verify_json(run_dir / "04_sealed" / "manifest.json", self_hash=True)
    generated = _generation(config, run_dir)
    generated_by = {str(row["sample_id"]): row for row in generated["records"]}
    records = []
    for sealed_record in sealed["records"]:
        sid = str(sealed_record["sample_id"])
        source_map = sealed_record["selected_map"]
        target: dict[str, Any] = {}
        for portrait_id in (() if smoke else ("6", "9")):
            portrait = generated_by[sid]["portraits"][portrait_id]
            if int(portrait["frame_count"]) != int(source_map["frame_count"]):
                raise ProtocolError(f"PORTRAIT_TRANSFER_FRAME_COUNT_MISMATCH:{sid}:{portrait_id}")
            if int(portrait["valid_frame_count"]) != int(source_map["valid_frame_count"]):
                raise ProtocolError(f"PORTRAIT_TRANSFER_VALID_SUPPORT_MISMATCH:{sid}:{portrait_id}")
            transferred = build_map(int(portrait["frame_count"]), int(portrait["valid_frame_count"]),
                                    source_map["knot_deltas"], positions=source_map["knot_positions"])
            if transferred["q"] != source_map["q"]:
                raise ProtocolError("PORTRAIT_TRANSFER_Q_CHANGED")
            original, _ = __import__("scripts.experiments.mfa_linear_video_retiming.scorer", fromlist=["read_video_frames"]).read_video_frames(portrait["M"]["canonical_video"])
            output_frames, audit = render_map(original, transferred)
            video = _save_ffv1(config, output_frames, run_dir / "05_transfer" / "videos" / portrait_id / sid / "R.mkv", output_frames)
            target[portrait_id] = {"portrait_id": portrait_id, "M": portrait["M"], "N": portrait["N"],
                                   "map": transferred, "R": video, "interpolation_audit": audit,
                                   "score_box_xyxy": frozen["portrait_bindings"]["portraits"][portrait_id]["score_box"]["box"]}
        records.append({"sample_id": sid, "paired_key": sealed_record["paired_key"], "portraits": target})
    result = {"schema_version": 1, "status": "COMPLETE", "same_q_as_portrait3": True,
              "portrait_ids": [] if smoke else ["6", "9"], "records": records}
    return write_json(path, result, self_hash=True)


def _official_cells(run_dir: Path, frozen: Mapping[str, Any], sealed: Mapping[str, Any], transfer: Mapping[str, Any], *, smoke: bool) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    transfer_by = {str(row["sample_id"]): row for row in transfer["records"]}
    for row in sealed["records"]:
        sid = str(row["sample_id"])
        source_record = next(item for item in frozen["records"] if str(item["sample_id"]) == sid)
        audio_n = Path(source_record["audio"]["natural"]["path"])
        audio_m = Path(source_record["audio"]["mfa_linear"]["path"])
        for arm in ("N", "M", "R", "GLOBAL", "NEAREST", "MIRROR"):
            video_row = row["videos"].get(arm)
            if video_row is None:
                if arm == "MIRROR":
                    cells.append({"sample_id": sid, "paired_key": row["paired_key"], "speaker_id": row["speaker_id"],
                                  "portrait_id": "3", "video_arm": arm, "audio_role": "N",
                                  "audio": str(audio_n), "video": None, "status": "UNAVAILABLE",
                                  "reason": "MIRROR_MAP_INFEASIBLE"})
                    continue
                raise ProtocolError(f"SEALED_VIDEO_MISSING:{sid}:{arm}")
            for audio_role, audio in (("N", audio_n),):
                metrics = row["selected_metrics"] if arm == "R" else row["baseline_metrics"] if arm == "M" else {}
                cells.append({"sample_id": sid, "paired_key": row["paired_key"], "speaker_id": row["speaker_id"],
                              "portrait_id": "3", "video_arm": arm, "audio_role": audio_role,
                              "video": video_row["path"], "audio": str(audio),
                              "search_sync_c": metrics.get("sync_c"), "search_sync_d": metrics.get("sync_d"),
                              "search_d0": metrics.get("d0"), "search_offset": metrics.get("offset")})
        for arm, video_row in (("M", row["videos"]["M"]), ("R", row["videos"]["R"])):
            metrics = row["baseline_metrics"] if arm == "M" else row["selected_metrics"]
            cells.append({"sample_id": sid, "paired_key": row["paired_key"], "speaker_id": row["speaker_id"],
                          "portrait_id": "3", "video_arm": arm, "audio_role": "M",
                          "video": video_row["path"], "audio": str(audio_m),
                          "search_sync_c": None, "search_sync_d": None, "search_d0": None, "search_offset": None,
                          "diagnostic_search_metrics": metrics})
        if not smoke:
            for portrait_id in ("6", "9"):
                portrait = transfer_by[sid]["portraits"][portrait_id]
                for arm, video in (("N", portrait["N"]["canonical_video"]), ("M", portrait["M"]["canonical_video"]), ("R", portrait["R"]["path"])):
                    cells.append({"sample_id": sid, "paired_key": row["paired_key"], "speaker_id": row["speaker_id"],
                                  "portrait_id": portrait_id, "video_arm": arm, "audio_role": "N", "video": video,
                                  "audio": str(audio_n), "search_sync_c": None, "search_sync_d": None,
                                  "search_d0": None, "search_offset": None})
    expected_formal = 14 * len(frozen["records"])
    if not smoke and len(cells) != expected_formal:
        raise ProtocolError(f"OFFICIAL_CELL_DENOMINATOR_MISMATCH:{len(cells)} != {expected_formal}")
    if smoke and len(cells) != 8:
        raise ProtocolError(f"SMOKE_OFFICIAL_CELL_DENOMINATOR_MISMATCH:{len(cells)} != 8")
    return cells


def _is_ordered_prefix(prior: Sequence[str], expected: Sequence[str]) -> bool:
    return len(prior) <= len(expected) and list(prior) == list(expected[:len(prior)])


def stage_official(config: Mapping[str, Any], run_dir: Path, frozen: Mapping[str, Any], *, smoke: bool,
                   protocol: dict[str, Any] | None = None) -> dict[str, Any]:
    path = run_dir / "06_official" / "manifest.json"
    sealed = verify_json(run_dir / "04_sealed" / "manifest.json", self_hash=True)
    transfer = verify_json(run_dir / "05_transfer" / "manifest.json", self_hash=True)
    cells = _official_cells(run_dir, frozen, sealed, transfer, smoke=smoke)
    old_rows: list[dict[str, Any]] = []
    if path.is_file():
        old = verify_json(path, self_hash=True)
        old_rows = [dict(row) for row in old.get("rows", [])]
        expected_order = [f"p{row['portrait_id']}_s{row['sample_id']}_v{row['video_arm']}_a{row['audio_role']}" for row in cells]
        prior_order = [str(row.get("cell_key")) for row in old_rows]
        if not _is_ordered_prefix(prior_order, expected_order):
            raise ProtocolError("OFFICIAL_MANIFEST_CELL_ORDER_CHANGED")
    from .official import score_official_cell
    by_key = {str(row.get("cell_key")): row for row in old_rows}
    rows = []

    def persist(status: str) -> None:
        write_json(path, {"schema_version": 1, "status": status,
                          "sealed_manifest_sha256": file_sha256(run_dir / "04_sealed" / "manifest.json"),
                          "rows": rows}, self_hash=True)

    for cell in cells:
        key = f"p{cell['portrait_id']}_s{cell['sample_id']}_v{cell['video_arm']}_a{cell['audio_role']}"
        if cell.get("status") == "UNAVAILABLE":
            rows.append({**cell, "cell_key": key})
            persist("RUNNING")
            continue
        prior = by_key.get(key)
        if prior and prior.get("status") == "PASS":
            result_path = Path(prior["result_path"])
            if result_path.is_file():
                current = verify_json(result_path, self_hash=True)
                identity_matches = (
                    str(Path(str(prior.get("video", ""))).resolve()) == str(Path(cell["video"]).resolve())
                    and str(Path(str(prior.get("audio", ""))).resolve()) == str(Path(cell["audio"]).resolve())
                    and current.get("cell_key") == key
                    and str(Path(str(current.get("video", ""))).resolve()) == str(Path(cell["video"]).resolve())
                    and str(Path(str(current.get("audio", ""))).resolve()) == str(Path(cell["audio"]).resolve())
                    and current.get("syncnet_model_sha256") == file_sha256(config["paths"]["syncnet_model"])
                )
                if (identity_matches and current.get("video_sha256") == file_sha256(cell["video"])
                        and current.get("audio_sha256") == file_sha256(cell["audio"])):
                    rows.append({**prior, "cell_key": key})
                    persist("RUNNING")
                    continue
        remaining: float | None = None
        resource: dict[str, Any] | None = None
        if protocol is not None:
            _check_budget(protocol, config)
            resource = _wait_for_resource_preflight(config, run_dir, require_gpu=True)
            write_json(run_dir / "06_official" / "resource_preflight" / f"{key}.json", resource, self_hash=True)
            if resource["status"] != "READY":
                persist("RESOURCE_WAIT")
                return {"status": "RESOURCE_WAIT", "reason": resource.get("reason"),
                        "resource_preflight": resource, "row_count": len(rows),
                        "expected_row_count": len(cells), "rows": rows}
            remaining = _remaining_gpu_seconds(protocol, config)
            if remaining < 1.0:
                persist("BUDGET_LIMITED")
                return {"status": "BUDGET_LIMITED", "reason": "TOTAL_ACTIVE_GPU_BUDGET_EXHAUSTED",
                        "row_count": len(rows), "expected_row_count": len(cells), "rows": rows}
        try:
            score_operation = lambda: score_official_cell(config, cell, run_dir=run_dir, resume=False,
                                                          max_elapsed_seconds=remaining)
            result = (_timed_stage(run_dir / "00_protocol" / "protocol.json", protocol,
                                   "official", score_operation)
                      if protocol is not None else score_operation())
        except ProtocolError as exc:
            if str(exc).startswith("BUDGET_LIMITED"):
                persist("BUDGET_LIMITED")
                return {"status": "BUDGET_LIMITED", "reason": str(exc),
                        "row_count": len(rows), "expected_row_count": len(cells), "rows": rows}
            raise
        rows.append({"cell_key": key, "result_path": str((run_dir / "06_official" / "cells" / key / "result.json").resolve()),
                     "status": result["status"], "sample_id": cell["sample_id"], "paired_key": cell["paired_key"],
                     "speaker_id": cell["speaker_id"],
                     "portrait_id": cell["portrait_id"], "video_arm": cell["video_arm"], "audio_role": cell["audio_role"],
                     "video": cell["video"], "audio": cell["audio"], "video_sha256": result["video_sha256"],
                     "audio_sha256": result["audio_sha256"], "official_sync_c": result["official_sync_c"],
                     "official_sync_d": result["official_sync_d"], "official_d0": result["recomputed_official_curve"]["d0"],
                     "official_offset": result["official_offset"], "activesd_path": result["activesd"]["path"],
                     "activesd_sha256": result["activesd"]["sha256"], "crop_frame_count": result["crop"]["frame_count"],
                     "official_support_columns": result["recomputed_official_curve"]["support_columns"],
                     "resource_preflight": resource,
                     "search_sync_c": cell.get("search_sync_c"), "search_sync_d": cell.get("search_sync_d"),
                     "search_d0": cell.get("search_d0"), "search_offset": cell.get("search_offset")})
        persist("RUNNING")
    # The fixed primary contrasts require matching official support. Common-support
    # diagnostics are kept separately and never replace the official log metrics.
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "PASS":
            continue
        grouped.setdefault((str(row["sample_id"]), str(row["portrait_id"]), str(row["audio_role"])), {})[str(row["video_arm"])] = row
    comparisons = []
    from .official import recompute_official_curve
    for (sid, portrait, audio_role), arms in sorted(grouped.items()):
        if "M" not in arms or "R" not in arms:
            continue
        m, r = arms["M"], arms["R"]
        matched = int(m["official_support_columns"]) == int(r["official_support_columns"])
        # The persisted official worker matrix is the source for paired support checks.
        cell_result_m = verify_json(run_dir / "06_official" / "cells" / m["cell_key"] / "result.json", self_hash=True)
        cell_result_r = verify_json(run_dir / "06_official" / "cells" / r["cell_key"] / "result.json", self_hash=True)
        curve_m = recompute_official_curve(np.asarray(cell_result_m["activesd"]["distance_matrix"], dtype=np.float32), columns=min(m["official_support_columns"], r["official_support_columns"]))
        curve_r = recompute_official_curve(np.asarray(cell_result_r["activesd"]["distance_matrix"], dtype=np.float32), columns=min(m["official_support_columns"], r["official_support_columns"]))
        comparisons.append({"sample_id": sid, "portrait_id": portrait, "audio_role": audio_role,
                            "status": "MATCHED" if matched else "SUPPORT_MISMATCH",
                            "support_m": int(m["official_support_columns"]), "support_r": int(r["official_support_columns"]),
                            "official_delta_sync_c": float(r["official_sync_c"] - m["official_sync_c"]) if matched else None,
                            "official_delta_d0": float(r["official_d0"] - m["official_d0"]) if matched else None,
                            "common_support_delta_sync_c": float(curve_r["sync_c"] - curve_m["sync_c"]),
                            "common_support_delta_d0": float(curve_r["d0"] - curve_m["d0"]),
                            "common_support_columns": int(min(m["official_support_columns"], r["official_support_columns"]))})
    result = {"schema_version": 1, "status": "COMPLETE" if all(row.get("status") in {"PASS", "UNAVAILABLE"} for row in rows) and len(rows) == len(cells) else "INCOMPLETE",
              "sealed_manifest_sha256": file_sha256(run_dir / "04_sealed" / "manifest.json"), "official_score_prefix": "official_",
              "row_count": len(rows), "expected_row_count": len(cells), "rows": rows, "matched_support_comparisons": comparisons,
              "same_weight_independent_validation": False,
              "interpretation": "Official chain uses the same frozen SyncNet weights as search and is not an independent-model validation."}
    return write_json(path, result, self_hash=True)


def _load_run_context(config_path: Path, run_id: str, smoke: bool) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    run_dir = _run_dir(config, run_id)
    audit = stage_audit(config, run_dir, run_id=run_id, smoke=smoke)
    protocol = audit["protocol"]
    if bool(protocol["engineering_only"]) != bool(smoke):
        raise ProtocolError("SMOKE_FORMAL_RUN_MODE_MISMATCH")
    if protocol.get("stage_status", {}).get("audit") == "RESOURCE_WAIT":
        raise ProtocolError(f"RESOURCE_WAIT:{protocol.get('preflight', {}).get('reason')}")
    return config, run_dir, audit["frozen_inputs"], protocol


def run_stage(config_path: Path, run_id: str, stage: str, *, smoke: bool = False) -> int:
    try:
        config, run_dir, frozen, protocol = _load_run_context(config_path, run_id, smoke)
    except ProtocolError as exc:
        if str(exc).startswith("RESOURCE_WAIT:"):
            print(f"audit RESOURCE_WAIT: {exc}", file=sys.stderr)
            return 2
        raise
    protocol_path = run_dir / "00_protocol" / "protocol.json"
    try:
        result: Mapping[str, Any] | None = None
        if stage == "audit":
            pass
        elif stage == "render":
            result = stage_render(config, run_dir, frozen, protocol, smoke=smoke)
        elif stage in {"calibrate", "search"}:
            result = stage_calibrate_search(config, run_dir, frozen, protocol, mode=stage)
        elif stage == "seal":
            result = stage_seal(config, run_dir, frozen, smoke=smoke)
        elif stage == "transfer":
            result = stage_transfer(config, run_dir, frozen, smoke=smoke)
        elif stage == "official":
            result = stage_official(config, run_dir, frozen, smoke=smoke, protocol=protocol)
        elif stage == "check":
            from .check import check_run
            _check_budget(protocol, config)
            resource = _wait_for_resource_preflight(config, run_dir, require_gpu=True)
            resource_attempts: list[dict[str, Any]] = [{"stage": "check_start", **resource}]
            resource_path = run_dir / "07_check" / "resource_preflight.json"
            write_json(resource_path, {"status": "RUNNING", "attempts": resource_attempts}, self_hash=True)
            if resource["status"] != "READY":
                raise ProtocolError(f"RESOURCE_WAIT:{resource.get('reason')}")
            remaining = _remaining_gpu_seconds(protocol, config)
            forward_count = 2 * len(frozen["records"])
            if remaining < forward_count + 1:
                raise ProtocolError("BUDGET_LIMITED:TOTAL_ACTIVE_GPU_BUDGET_EXHAUSTED")
            deadline = time.monotonic() + remaining
            configured_timeout = int(config["official"]["syncnet_timeout_seconds"])

            def fresh_timeout_for_remaining(calls_left: int) -> int:
                current_resource = _wait_for_resource_preflight(config, run_dir, require_gpu=True)
                resource_attempts.append({"stage": "fresh_forward", "forward_index": len(resource_attempts),
                                          **current_resource})
                write_json(resource_path, {"status": "RUNNING", "attempts": resource_attempts}, self_hash=True)
                if current_resource["status"] != "READY":
                    raise ProtocolError(f"RESOURCE_WAIT:{current_resource.get('reason')}")
                seconds_left = deadline - time.monotonic()
                timeout = min(configured_timeout, int(seconds_left / max(1, calls_left + 1)))
                if timeout < 1:
                    raise ProtocolError("BUDGET_LIMITED:TOTAL_ACTIVE_GPU_BUDGET_EXHAUSTED")
                return timeout

            result = _timed_stage(protocol_path, protocol, "check", lambda: check_run(
                run_dir, config_path=config_path, force_fresh=True,
                timeout_for_remaining=fresh_timeout_for_remaining))
            if result.get("status") != "PASS":
                raise ProtocolError(f"CHECK_FAILED:{result.get('error', result.get('status'))}")
            write_json(resource_path, {"status": "PASS", "attempts": resource_attempts}, self_hash=True)
            result.setdefault("checks", {})["resource_preflight"] = resource_attempts
            write_json(run_dir / "07_check" / "validation.json", result, self_hash=True)
        elif stage == "report":
            from .report import write_report
            result = write_report(run_dir)
            from .check import check_run
            checked = check_run(run_dir, config_path=config_path, force_fresh=False)
            if checked.get("status") != "PASS":
                raise ProtocolError(f"REPORT_CHECK_FAILED:{checked.get('error', checked.get('status'))}")
        else:
            raise ProtocolError(f"unknown stage: {stage}")
        if result is not None and result.get("status") in {"RESOURCE_WAIT", "BUDGET_LIMITED"}:
            raise ProtocolError(f"{result['status']}:{result.get('reason', 'stage returned a resumable partial result')}")
        if stage == "official" and result is not None and result.get("status") != "COMPLETE":
            raise ProtocolError(f"OFFICIAL_INCOMPLETE:{result.get('status')}")
        protocol.setdefault("stage_status", {})[stage] = "PASS"
        write_json(protocol_path, protocol, self_hash=True)
        return 0
    except Exception as exc:
        message = str(exc)
        if message.startswith("RESOURCE_WAIT"):
            status = "RESOURCE_WAIT"
        elif message.startswith("BUDGET_LIMITED") or "TOTAL_ACTIVE_GPU_BUDGET_EXHAUSTED" in message:
            status = "BUDGET_LIMITED"
        elif message.startswith("CALIBRATION_FAILED"):
            status = "CALIBRATION_FAILED"
        elif message.startswith("CHECK_FAILED") or message.startswith("REPORT_CHECK_FAILED"):
            status = "CHECK_FAILED"
        elif message.startswith("OFFICIAL_INCOMPLETE"):
            status = "INCOMPLETE"
        else:
            status = "FAILED"
        protocol.setdefault("stage_status", {})[stage] = status
        protocol.setdefault("stage_errors", {})[stage] = f"{type(exc).__name__}: {exc}"
        write_json(protocol_path, protocol, self_hash=True)
        print(f"{stage} {status}: {exc}", file=sys.stderr)
        return 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MFA-linear fixed-cohort constrained video retiming")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--smoke", action="store_true", help="sample 1 / portrait 3 engineering-only run")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    requested = STAGES if args.stage == "all" else (args.stage,)
    for stage in requested:
        code = run_stage(args.config.resolve(), args.run_id, stage, smoke=args.smoke)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
