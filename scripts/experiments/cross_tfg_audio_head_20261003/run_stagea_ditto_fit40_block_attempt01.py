#!/usr/bin/env python3
"""Run one six-method Ditto block from the frozen Fit40 Stage-A plan.

This overlay imports the byte-pinned W2L Fit block executor and replaces only
the backend-specific renderer and live resource gates. The original PCM
delivery, official SyncNet tracking/scoring, decoded-frame support, method
waveforms, and six-row common-support calculation stay in the pinned executor.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True

MODEL_ID = "ditto_hubert_fixed_portrait_attempt18"
FRONTEND = "ditto_hubert_onnx_cpu_torch_cuda_static_portrait_attempt18"
RENDERER_RESULT_PROFILE = "ditto_hubert_onnx_cpu_renderer_torch_cuda"
DITTO_COMMIT = "c3e47eee2e626500017a0556b470d6d4182f85e8"
DITTO_RUNNER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_ditto_fit40_block_attempt01.py"
W2L_CORE_REL = "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_w2l_block_attempt02.py"
W2L_CORE_BYTES = 96_537
W2L_CORE_SHA256 = "6969c6e952d46b6753d279b0e2447aa71c7b8e92967cd42ee57dc8122da5616e"
RESOURCE_HELPER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/v3_dev180_newhost_latebind_packet_attempt02.py"
RESOURCE_HELPER_BYTES = 62_830
RESOURCE_HELPER_SHA256 = "406f95e4a262fb62bb02bc67249f3e6a7bf211b803a014622e042329d772fdf7"
RENDERER_REL = "job_control/ditto_v100_identity_repeat_attempt02/render_silent_video_v100_attempt02.py"
RENDERER_BYTES = 40_010
RENDERER_SHA256 = "7b37381777ad8cabf3665a7ab636295bc779b4a1fde5fa15dd485360772a53e4"
DITTO_DEPLOY_ROOT = "/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01"
DITTO_SOURCE_ROOT = DITTO_DEPLOY_ROOT + "/model_sources/ditto-talkinghead"
DITTO_WEIGHTS_ROOT = DITTO_DEPLOY_ROOT + "/model_assets/ditto_v04_hubert"
DITTO_CONFIG = DITTO_WEIGHTS_ROOT + "/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
DITTO_PYTHON = "/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01/bin/python"
BOOTSTRAP_PATH = DITTO_DEPLOY_ROOT + "/job_control/ditto_v100_identity_repeat_attempt02/sitecustomize.py"
BOOTSTRAP_BYTES = 9_525
BOOTSTRAP_SHA256 = "3099c4b2e1c650f7aa46c31145cc2c88ccda9f085288ab039c6ffed85007ab65"
IMPORT_REPORT_PATH = DITTO_DEPLOY_ROOT + "/job_control/ditto_v100_cpu_import_probe_attempt01/ditto_v100_import_cython_receipt_attempt01.json"
IMPORT_REPORT_BYTES = 35_066
IMPORT_REPORT_SHA256 = "2b1ba33d34808813f21059b0b743be30f5b55e1c224c1e22408b7c3a3b83df1e"
IMPORT_REPORT_SELF_SHA256 = "3c46031cab73234111b8d31c0b3df5f7bea358d57ef755107cbf5a73c2e06f74"
BLEND_EXTENSION_PATH = DITTO_DEPLOY_ROOT + "/job_control/ditto_v100_cpu_import_probe_attempt01/cython_build_attempt01/pyximport/lib.linux-x86_64-cpython-310/core/utils/blend/blend.cpython-310-x86_64-linux-gnu.so"
BLEND_EXTENSION_BYTES = 89_176
BLEND_EXTENSION_SHA256 = "5bb164bf4a3f82b9e83318fbc4a0a2912cff5b1c476ec9a4ab3375ff49640709"
GPU_UUID = "GPU-d12b3309-c299-da83-064d-a3cd619bd560"
GPU_NAME = "Tesla V100-SXM2-16GB"
TORCH_VERSION = "2.5.1+cu121"
CUDA_VERSION = "12.1"
PYTHON_VERSION = "3.10.20"
NUMPY_VERSION = "1.26.4"
ORT_VERSION = "1.20.1"
SAMPLE_RATE = 16_000
FPS_NUM = 25
FPS_DEN = 1
RESERVE_BYTES = 5_368_709_120
MIN_GPU_FREE_MIB = 10_000
MIN_RAM_AVAILABLE_BYTES = 8_589_934_592
MODEL_ASSET_SHA256 = "dc29eacae64c53a741e58a28e94081c5b1d0a04714f0b478fc7114c6a50a8478"
SCORE_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
FACE_SHA256 = "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6"
FACE_BYTES = 253_479
RUNTIME_BINDING_SCHEMA = "cross_tfg_fit40_ditto_runtime_binding_v1"
STAGE_GATE_SCHEMA = "cross_tfg_fit40_ditto_stage_resource_gate_v1"
ROW_GATE_SCHEMA = "cross_tfg_fit40_ditto_per_row_resource_gate_v1"
PROJECTION_SCHEMA = "cross_tfg_fit40_ditto_parent_projection_v1"
BOOTSTRAP_RECEIPT_SCHEMA = "cross_tfg_renderer_strict_determinism_bootstrap_receipt_v1"

STRICT_POLICY = {
    "pythonhashseed": "20261003", "cublas_workspace_config": ":4096:8",
    "omp_num_threads": 1, "mkl_num_threads": 1, "openblas_num_threads": 1,
    "numexpr_num_threads": 1, "torch_num_threads": 1,
    "torch_num_interop_threads": 1, "torch_use_deterministic_algorithms": True,
    "torch_cudnn_deterministic": True, "torch_cudnn_benchmark": False,
    "torch_allow_tf32": False,
}
STRICT_ENV = {
    "PYTHONHASHSEED": "20261003", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular(path: Path, limit: int = 32 * 1024 * 1024) -> bytes:
    if path.is_symlink():
        raise RuntimeError(f"refusing symlinked input: {path}")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size < 0 or before.st_size > limit:
        raise RuntimeError(f"input is not a bounded regular file: {path}")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    after = path.stat(follow_symlinks=False)
    if (len(raw) != before.st_size or len(raw) > limit
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise RuntimeError(f"input changed while being read: {path}")
    return raw


def expected_native_frame_count(sample_count: int, sample_rate_hz: int = SAMPLE_RATE,
                                fps_num: int = FPS_NUM, fps_den: int = FPS_DEN) -> int:
    """Ditto's native fixed-rate output contract: ceil(N * fps / sample_rate)."""
    if (type(sample_count) is not int or sample_count <= 0 or type(sample_rate_hz) is not int
            or sample_rate_hz <= 0 or type(fps_num) is not int or fps_num <= 0
            or type(fps_den) is not int or fps_den <= 0):
        raise ValueError("native-frame inputs must be positive integers")
    numerator = sample_count * fps_num
    denominator = sample_rate_hz * fps_den
    return (numerator + denominator - 1) // denominator


def validate_fit_parent_rows(rows: Sequence[Mapping[str, Any]], methods: Sequence[Mapping[str, Any]]) -> None:
    """Require the full frozen Fit40 roster and 6 contiguous rows per sample."""
    if len(rows) != 240 or len(methods) != 6:
        raise ValueError("parent must remain the complete 40-context × 6-method Fit plan")
    method_ids = [method.get("method_id") for method in methods]
    if len(set(method_ids)) != 6 or any(not isinstance(item, str) for item in method_ids):
        raise ValueError("the six frozen method IDs must be unique strings")
    sample_ids = {row.get("sample_id") for row in rows}
    speaker_ids = {row.get("speaker_id") for row in rows}
    if len(sample_ids) != 40 or len(speaker_ids) != 20:
        raise ValueError("parent roster must retain exactly 40 clips and 20 speakers")
    if any(row.get("split") != "fit" or row.get("source_media_kind") != "static_portrait"
           or row.get("seed") != 20261003 for row in rows):
        raise ValueError("parent rows must remain Fit-only static portrait rows with frozen seed")
    for start in range(0, len(rows), 6):
        block = rows[start:start + 6]
        if len(block) != 6 or len({row.get("sample_id") for row in block}) != 1:
            raise ValueError("each contiguous six-row block must be one complete clip context")
        if [row.get("method_id") for row in block] != method_ids:
            raise ValueError("each context must preserve the frozen six-method order")
        if len({row.get("speaker_id") for row in block}) != 1:
            raise ValueError("all six methods in a context must share its frozen speaker")
        if len({row.get("seed") for row in block}) != 1:
            raise ValueError("all six methods in a context must share its frozen seed")
    counts = {method_id: sum(row.get("method_id") == method_id for row in rows)
              for method_id in method_ids}
    if set(counts.values()) != {40}:
        raise ValueError("each frozen method must retain all 40 planned clips")


def project_fit_rows(parent_rows: Sequence[Mapping[str, Any]], methods: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project W2L rows to Ditto while preserving all source/method fields."""
    validate_fit_parent_rows(parent_rows, methods)
    projected: list[dict[str, Any]] = []
    for original in parent_rows:
        row = copy.deepcopy(dict(original))
        old_id = str(row["row_id"])
        natural = row.get("natural_pcm")
        if not isinstance(natural, Mapping):
            raise ValueError(f"row {old_id} lacks its original PCM binding")
        sample_count = natural.get("sample_count")
        if (natural.get("sample_rate_hz") != SAMPLE_RATE or natural.get("channels") != 1
                or row.get("score_audio") != natural):
            raise ValueError(f"row {old_id} changed the exact mono 16 kHz source/score PCM binding")
        face = row.get("source_face", {})
        if (face.get("file_sha256") != FACE_SHA256 or not isinstance(face.get("path"), str)):
            raise ValueError(f"row {old_id} is not bound to the frozen shared portrait")
        row["model_id"] = MODEL_ID
        row["audio_frontend"] = FRONTEND
        row["row_id"] = (f"fit_{row['speaker_id']}_{row['sample_id']}_{MODEL_ID}_"
                         f"{row['face_id']}_{row['seed']}_{row['method_id']}")
        settings = dict(row.get("generation_settings", {}))
        settings.update({"generator": "ditto_v100_renderer_attempt02",
                         "face_mode": "static_image",
                         "input_audio": "enhanced_generator_condition_only",
                         "fps_num": FPS_NUM, "fps_den": FPS_DEN,
                         "seed": int(row["seed"])})
        row["generation_settings"] = settings
        row.pop("wav2lip_native_frame_binding", None)
        row["ditto_native_frame_binding"] = {
            "fps_num": FPS_NUM, "fps_den": FPS_DEN,
            "sample_rate_hz": SAMPLE_RATE, "sample_count": sample_count,
            "expected_frame_count": expected_native_frame_count(sample_count),
            "frame_count_source": "ceil(generator_condition_sample_count*25/16000); verified by native renderer and decoded output PTS",
            "verified_against_generator_condition_audio": True,
            "verification_required_at_runtime": True,
        }
        projected.append(row)
    if len({row["row_id"] for row in projected}) != len(projected):
        raise ValueError("projected Ditto row IDs are not unique")
    return projected


def validate_renderer_input_report(row: Mapping[str, Any], *, audio_ref: Mapping[str, Any],
                                   report: Mapping[str, Any], profile: str = RENDERER_RESULT_PROFILE) -> None:
    """Check report identity, profile, full PCM input, and native ceil frame output."""
    expected_frames = expected_native_frame_count(int(audio_ref.get("sample_count", 0)))
    binding = row.get("ditto_native_frame_binding", {})
    audio = report.get("audio", {})
    video = report.get("silent_video", {})
    probe = video.get("probe", {}) if isinstance(video, Mapping) else {}
    if (report.get("status") != "PASS_ENGINEERING_ONLY"
            or report.get("model") != "ditto"
            or report.get("profile") != profile
            or report.get("row_id") != row.get("row_id")
            or report.get("seed") != row.get("seed")
            or binding.get("sample_count") != audio_ref.get("sample_count")
            or binding.get("expected_frame_count") != expected_frames
            or audio.get("file_sha256") != audio_ref.get("file_sha256")
            or audio.get("pcm_float32_sha256") != audio_ref.get("pcm_sha256")
            or audio.get("sample_count") != audio_ref.get("sample_count")
            or audio.get("sample_rate_hz") != SAMPLE_RATE
            or audio.get("channels") != 1
            or video.get("file_sha256") is None
            or video.get("probe", {}).get("frame_count") != expected_frames
            or video.get("probe", {}).get("decoded_frame_count") != expected_frames
            or video.get("probe", {}).get("fps") != "25/1"
            or video.get("probe", {}).get("decoded_pts_exact_25fps") is not True):
        raise RuntimeError("Ditto renderer report differs from the exact row, profile, PCM, or ceil-frame contract")


def validate_block(plan: Mapping[str, Any], block_start: int, block_count: int = 6) -> list[Mapping[str, Any]]:
    rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) != 240:
        raise ValueError("Ditto plan must keep all 240 rows in its parent denominator")
    if (type(block_start) is not int or type(block_count) is not int or block_count != 6
            or block_start < 0 or block_start % 6 or block_start + block_count > 240):
        raise ValueError("one invocation must select an in-range contiguous six-method context")
    block = rows[block_start:block_start + block_count]
    if (len(block) != 6 or len({row.get("sample_id") for row in block}) != 1
            or [row.get("method_id") for row in block] != [item.get("method_id") for item in plan.get("methods", [])]):
        raise ValueError("selected rows are not exactly the frozen six-method context")
    return block


def _verify_self(value: Mapping[str, Any], field: str, label: str) -> None:
    if value.get(field) != _sha_bytes(_canonical({key: item for key, item in value.items() if key != field})):
        raise RuntimeError(f"{label} canonical self-hash mismatch")


def _load_module(name: str, path: Path, expected_bytes: int, expected_sha256: str) -> Any:
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != expected_bytes
            or _sha_file(path) != expected_sha256):
        raise RuntimeError(f"pinned source is absent or changed: {path}")
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load pinned source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path, limit: int = 32 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, limit)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value, raw


def _load_jsonl(path: Path, limit: int = 32 * 1024 * 1024) -> tuple[list[dict[str, Any]], bytes]:
    raw = _read_regular(path, limit)
    values = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if any(not isinstance(value, dict) for value in values):
        raise RuntimeError(f"expected JSON objects in {path}")
    return values, raw


def _normalize_file_map(value: Any) -> dict[str, tuple[int | None, str]]:
    result: dict[str, tuple[int | None, str]] = {}
    if not isinstance(value, Mapping):
        return result
    for key, item in value.items():
        if isinstance(item, str):
            result[str(key)] = (None, item)
        elif isinstance(item, Mapping) and isinstance(item.get("sha256"), str):
            result[str(key)] = (item.get("bytes") if type(item.get("bytes")) is int else None,
                                item["sha256"])
    return result


def _renderer_static_pins(renderer_path: Path) -> tuple[dict[str, str], dict[str, tuple[int, str]]]:
    """Read the pinned renderer's literal source/weight maps without importing it."""
    tree = ast.parse(_read_regular(renderer_path, 2 * 1024 * 1024).decode("utf-8"), filename=str(renderer_path))
    assignments: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            for target in targets:
                if isinstance(target, ast.Name) and target.id in {"DITTO_SOURCE_PINS", "DITTO_WEIGHT_PINS"}:
                    assignments[target.id] = ast.literal_eval(value)
    source = assignments.get("DITTO_SOURCE_PINS")
    weights = assignments.get("DITTO_WEIGHT_PINS")
    if not isinstance(source, dict) or not isinstance(weights, dict):
        raise RuntimeError("pinned Ditto renderer lacks literal source/weight pin maps")
    return source, {str(key): (int(pair[0]), str(pair[1])) for key, pair in weights.items()}


def _validate_model_and_binding(plan: Mapping[str, Any], binding: Mapping[str, Any],
                                binding_raw: bytes, binding_path: Path,
                                parent_plan: Mapping[str, Any], parent_plan_raw: bytes,
                                parent_plan_path: Path, parent_rows_raw: bytes,
                                stagea_core_path: Path, stagea_core_sha: str,
                                overlay_path: Path, overlay_sha: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if (binding.get("schema") != RUNTIME_BINDING_SCHEMA or binding.get("status") != "PASS"
            or binding.get("model_id") != MODEL_ID or binding.get("audio_frontend") != FRONTEND
            or binding.get("run_id") != plan.get("run_id")
            or binding.get("gpu_uuid") != GPU_UUID):
        raise RuntimeError("Ditto runtime binding schema/status/model/run/GPU identity is invalid")
    _verify_self(binding, "binding_sha256", "Ditto runtime binding")
    projection = plan.get("ditto_fit_projection", {})
    binding_ref = plan.get("execution", {}).get("ditto_runtime_binding", {})
    if (binding_ref.get("path") != str(binding_path)
            or binding_ref.get("raw_sha256") != _sha_bytes(binding_raw)
            or projection.get("parent_plan_raw_sha256") != _sha_bytes(parent_plan_raw)
            or projection.get("parent_plan_path") != str(parent_plan_path)
            or projection.get("parent_rows_raw_sha256") != _sha_bytes(parent_rows_raw)
            or projection.get("parent_plan_payload_sha256") != parent_plan.get("plan_sha256")
            or projection.get("schema") != PROJECTION_SCHEMA):
        raise RuntimeError("Ditto plan does not bind the exact frozen parent and runtime binding")
    if (plan.get("status") != "FROZEN"
            or plan.get("plan_sha256") != _sha_bytes(_canonical({key: value for key, value in plan.items() if key != "plan_sha256"}))):
        raise RuntimeError("Ditto projected plan self-hash/status is invalid")
    execution = plan.get("execution", {})
    if (execution.get("planned_rows") != 240 or execution.get("model_ids") != [MODEL_ID]
            or execution.get("capacity_stage") != "fit40_ditto240"
            or execution.get("scientific_input_pins_sha256") != parent_plan.get("scientific_input_pins_sha256")
            or execution.get("gpu_uuid") != GPU_UUID
            or execution.get("dev_or_confirm_access") is not False
            or execution.get("candidate_selection") != "not performed"):
        raise RuntimeError("Ditto execution identity differs from the frozen Fit-only 240-row contract")
    core_ref = execution.get("stagea_runner", {})
    overlay_ref = execution.get("ditto_block_overlay", {})
    if (core_ref.get("path") != str(stagea_core_path)
            or core_ref.get("sha256") != stagea_core_sha
            or execution.get("stagea_runner_sha256") != stagea_core_sha
            or overlay_ref.get("path") != str(overlay_path)
            or overlay_ref.get("sha256") != overlay_sha):
        raise RuntimeError("plan source closure does not bind the exact W2L core and Ditto overlay")
    qualification = binding.get("runtime_qualification", {})
    if (execution.get("runtime_qualification", {}).get("path") != qualification.get("path")
            or execution.get("runtime_qualification", {}).get("raw_sha256") != qualification.get("raw_sha256")):
        raise RuntimeError("plan qualification reference differs from the actual Ditto binding")
    model = plan.get("models", [{}])[0]
    generator = model.get("generator", {})
    if (model.get("model_id") != MODEL_ID or model.get("audio_frontend") != FRONTEND
            or model.get("score_model_sha256") != parent_plan.get("models", [{}])[0].get("score_model_sha256")
            or model.get("model_sha256") != MODEL_ASSET_SHA256
            or model.get("checkpoint_sha256") != MODEL_ASSET_SHA256
            or model.get("git_head") != DITTO_COMMIT
            or generator.get("renderer_path") != DITTO_DEPLOY_ROOT + "/" + RENDERER_REL
            or generator.get("renderer_sha256") != RENDERER_SHA256
            or generator.get("renderer_report_profile") != RENDERER_RESULT_PROFILE
            or generator.get("python") != DITTO_PYTHON
            or generator.get("source_root") != DITTO_SOURCE_ROOT
            or generator.get("weights_root") != DITTO_WEIGHTS_ROOT
            or generator.get("config_path") != DITTO_CONFIG
            or generator.get("strict_bootstrap_path") != BOOTSTRAP_PATH
            or generator.get("strict_bootstrap_sha256") != BOOTSTRAP_SHA256
            or generator.get("contract") != "cross_tfg_silent_video_cli_v1"):
        raise RuntimeError("Ditto model descriptor differs from the pinned V100 native profile")
    runtime = binding.get("runtime", {})
    if (runtime.get("python") != DITTO_PYTHON or runtime.get("python_version") != PYTHON_VERSION
            or runtime.get("torch") != TORCH_VERSION or runtime.get("cuda") != CUDA_VERSION
            or runtime.get("onnxruntime") != ORT_VERSION or runtime.get("device") != GPU_NAME
            or runtime.get("compute_capability") != [7, 0]
            or runtime.get("gpu_uuid") != GPU_UUID
            or runtime.get("onnxruntime_providers") != ["CPUExecutionProvider"]):
        raise RuntimeError("Ditto runtime binding differs from the measured V100/PyTorch/CPU-ONNX profile")
    expected_refs = {
        "bootstrap": (generator.get("strict_bootstrap_path"), BOOTSTRAP_BYTES, BOOTSTRAP_SHA256),
        "import_report": (generator.get("import_report_path"), IMPORT_REPORT_BYTES, IMPORT_REPORT_SHA256),
        "blend_extension": (generator.get("blend_extension_path"), BLEND_EXTENSION_BYTES, BLEND_EXTENSION_SHA256),
    }
    for label, (path_text, size, digest) in expected_refs.items():
        path = Path(str(path_text))
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size or _sha_file(path) != digest:
            raise RuntimeError(f"pinned Ditto {label} source/runtime artifact is absent or changed")
    renderer_path = Path(str(generator["renderer_path"]))
    if renderer_path.is_symlink() or not renderer_path.is_file() or renderer_path.stat().st_size != RENDERER_BYTES or _sha_file(renderer_path) != RENDERER_SHA256:
        raise RuntimeError("pinned Ditto V100 renderer source is absent or changed")
    source_pins, weight_pins = _renderer_static_pins(renderer_path)
    described_sources = _normalize_file_map(model.get("source_sha256"))
    if {name: value[1] for name, value in described_sources.items()} != source_pins:
        raise RuntimeError("Ditto model descriptor source pin map differs from the renderer's literal source pins")
    model_assets = model.get("model_assets", {})
    described_weights: dict[str, tuple[int | None, str]] = {}
    if isinstance(model_assets, Mapping):
        def walk(value: Any) -> None:
            if isinstance(value, Mapping):
                rel = value.get("relative_path")
                digest = value.get("sha256")
                if isinstance(rel, str) and isinstance(digest, str):
                    described_weights[rel] = (value.get("bytes") if type(value.get("bytes")) is int else None, digest)
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(model_assets)
    for rel, expected in weight_pins.items():
        if rel == "ditto_cfg/v0.4_hubert_cfg_pytorch.pkl":
            # The plan's config may be represented separately from the model asset roster.
            observed = described_weights.get(rel)
            if observed is None:
                cfg_ref = generator.get("configuration", {})
                if (cfg_ref.get("path") == DITTO_CONFIG and cfg_ref.get("bytes") == expected[0]
                        and cfg_ref.get("sha256") == expected[1]):
                    described_weights[rel] = expected
                    continue
        if described_weights.get(rel) != expected:
            raise RuntimeError(f"Ditto plan is missing exact model weight/config pin: {rel}")
    if set(described_weights) != set(weight_pins):
        raise RuntimeError("Ditto model asset roster contains missing or unpinned extra files")
    return dict(model), dict(generator), Path(str(generator["renderer_path"]))


def _verify_parent_and_projection(plan: Mapping[str, Any], parent_plan: Mapping[str, Any],
                                  parent_rows: Sequence[Mapping[str, Any]], *,
                                  parent_plan_raw: bytes, parent_rows_raw: bytes) -> None:
    if (parent_plan.get("status") != "FROZEN"
            or parent_plan.get("plan_sha256") != _sha_bytes(_canonical({k: v for k, v in parent_plan.items() if k != "plan_sha256"}))):
        raise RuntimeError("frozen parent W2L Fit plan self-hash/status is invalid")
    if parent_rows != parent_plan.get("rows"):
        raise RuntimeError("external frozen W2L rows differ from the embedded ordered 240-row plan")
    methods = parent_plan.get("methods", [])
    validate_fit_parent_rows(parent_rows, methods)
    expected = project_fit_rows(parent_rows, methods)
    if plan.get("rows") != expected or plan.get("methods") != parent_plan.get("methods"):
        raise RuntimeError("Ditto plan changed a frozen Fit source/method row outside the allowed projection")
    if plan.get("scientific_input_pins_sha256") != parent_plan.get("scientific_input_pins_sha256"):
        raise RuntimeError("Ditto plan changed the frozen Fit scientific input pin identity")
    expected_protocol = parent_plan.get("protocol", {})
    protocol = plan.get("protocol", {})
    for key in ("audio", "mux", "roi", "syncnet"):
        if protocol.get(key) != expected_protocol.get(key):
            raise RuntimeError(f"Ditto projection changed frozen Stage-A protocol section {key}")
    source = plan.get("ditto_fit_projection", {})
    if (source.get("parent_plan_raw_sha256") != _sha_bytes(parent_plan_raw)
            or source.get("parent_rows_raw_sha256") != _sha_bytes(parent_rows_raw)
            or source.get("parent_plan_payload_sha256") != parent_plan.get("plan_sha256")
            or source.get("status") != "FROZEN_PARENT_PROJECTED_TO_DITTO_FIT240"):
        raise RuntimeError("Ditto plan projection lineage does not cite the exact frozen parent files")


def _load_binding(path_text: str, expected_raw_sha256: str, plan: Mapping[str, Any]) -> tuple[dict[str, Any], bytes, Path]:
    path = Path(path_text)
    if not path.is_absolute():
        raise RuntimeError("Ditto runtime binding path must be absolute")
    path = path.resolve(strict=True)
    raw = _read_regular(path, 8 * 1024 * 1024)
    expected = plan.get("execution", {}).get("ditto_runtime_binding", {}).get("raw_sha256")
    if _sha_bytes(raw) != expected_raw_sha256 or expected_raw_sha256 != expected:
        raise RuntimeError("Ditto runtime binding raw SHA differs from CLI and frozen plan")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Ditto runtime binding is not a JSON object")
    return value, raw, path


def _load_resource_helper(project_root: Path, binding: Mapping[str, Any]) -> Any:
    reference = binding.get("resources", {}).get("resource_helper", {})
    path = project_root / RESOURCE_HELPER_REL
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != RESOURCE_HELPER_BYTES
            or _sha_file(path) != RESOURCE_HELPER_SHA256
            or reference.get("path") != str(path) or reference.get("bytes") != RESOURCE_HELPER_BYTES
            or reference.get("sha256") != RESOURCE_HELPER_SHA256):
        raise RuntimeError("pinned live V100 resource helper is absent or differs from runtime binding")
    spec = importlib.util.spec_from_file_location("ditto_fit40_live_resource_helper", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pinned Ditto live resource helper")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    return helper


def _capacity_observation(plan: Mapping[str, Any], binding: Mapping[str, Any], helper: Any,
                          output_root: Path, first_remaining_row: int, *, whole_stage: bool) -> dict[str, Any]:
    resources = binding.get("resources", {})
    basis = resources.get("capacity_basis", {})
    if (resources.get("gpu_uuid") != GPU_UUID or resources.get("gpu_name") != GPU_NAME
            or resources.get("output_mount_target") != "/dev/shm"
            or resources.get("output_reserve_bytes") != RESERVE_BYTES
            or resources.get("min_gpu_free_mib") != MIN_GPU_FREE_MIB
            or resources.get("min_effective_ram_available_bytes") != MIN_RAM_AVAILABLE_BYTES
            or basis.get("status") != "MEASURED_DITTO_V100_ENGINEERING_ROW_SCALED_ESTIMATE_WITH_SLACK"
            or basis.get("media_bytes_estimated") is not True
            or basis.get("source_rows") != 40 or basis.get("methods_per_row") != 6):
        raise RuntimeError("Ditto runtime binding has no measured V100 240-row capacity basis")
    pilot_bytes = basis.get("pilot_row_readback_bytes")
    pilot_n = basis.get("pilot_sample_count")
    slack = basis.get("explicit_slack_multiplier")
    if (type(pilot_bytes) is not int or pilot_bytes <= 0 or type(pilot_n) is not int or pilot_n <= 0
            or type(slack) not in (int, float) or slack < 1):
        raise RuntimeError("Ditto capacity basis pilot readback or slack is invalid")
    if first_remaining_row < 0 or first_remaining_row >= 240:
        raise RuntimeError("invalid first remaining Fit40 row index")
    remaining = plan["rows"][first_remaining_row:]
    estimate = sum(math.ceil(pilot_bytes * int(row["natural_pcm"]["sample_count"]) / pilot_n)
                   for row in remaining) * float(slack)
    estimate = int(math.ceil(estimate))
    required = estimate + RESERVE_BYTES
    snapshot = helper.resource_snapshot(output_root, {
        "min_gpu_free_mib": 0, "min_ram_available_bytes": 0,
        "min_shm_free_bytes": 0, "min_output_free_bytes": 0,
    })
    gpu = next((item for item in snapshot.get("gpus", []) if item.get("uuid") == GPU_UUID), None)
    ram = snapshot.get("ram", {})
    shm = snapshot.get("shm", {})
    disk = snapshot.get("output_disk", {})
    reasons: list[str] = []
    if gpu is None or gpu.get("name") != GPU_NAME:
        reasons.append("bound Tesla V100 UUID/name is absent from the live GPU inventory")
    elif int(gpu.get("memory_free_mib", -1)) < MIN_GPU_FREE_MIB:
        reasons.append("live V100 free memory is below the pinned minimum")
    if int(ram.get("available_bytes", -1)) < MIN_RAM_AVAILABLE_BYTES:
        reasons.append("live effective RAM is below the pinned minimum")
    if int(shm.get("available_bytes", -1)) < required:
        reasons.append("live /dev/shm bytes are below the remaining Ditto estimate plus reserve")
    if (disk.get("target") != "/dev/shm" or int(disk.get("available_bytes", -1)) < required):
        reasons.append("live output filesystem differs from /dev/shm or is below estimate plus reserve")
    cgroup = ram.get("cgroup", {})
    expected_cgroup = resources.get("cgroup", {})
    actual_identity = cgroup.get("identity", {}) if isinstance(cgroup, Mapping) else {}
    expected_identity = expected_cgroup.get("identity", {}) if isinstance(expected_cgroup, Mapping) else {}
    if (cgroup.get("version") != expected_cgroup.get("version")
            or actual_identity.get("mount_target") != expected_identity.get("mount_target")
            or actual_identity.get("mount_root") != expected_identity.get("mount_root")):
        reasons.append("live cgroup version or stable mount identity differs from qualification")
    if cgroup.get("finite_limit") is True:
        available = cgroup.get("available_bytes")
        if type(available) is not int or available < MIN_RAM_AVAILABLE_BYTES:
            reasons.append("live finite cgroup memory is below the pinned minimum")
    elif cgroup.get("finite_limit") is not False or cgroup.get("available_bytes") is not None:
        reasons.append("live cgroup availability fields are malformed")
    return {
        "status": "PASS" if not reasons else "FAILED_RESOURCE_GATE",
        "whole_stage_remaining_roster": whole_stage,
        "first_remaining_row_index": first_remaining_row,
        "remaining_rows": len(remaining),
        "remaining_media_estimate_bytes": estimate,
        "required_bytes_with_reserve": required,
        "reserve_bytes": RESERVE_BYTES,
        "pilot_row_readback_bytes": pilot_bytes,
        "pilot_sample_count": pilot_n,
        "explicit_slack_multiplier": float(slack),
        "estimate_classification": "measured Ditto V100 engineering row readback scaled by original sample count and explicit slack; estimate only",
        "observed": {
            "gpu_uuid": GPU_UUID,
            "gpu_name": gpu.get("name") if gpu else None,
            "gpu_free_mib": gpu.get("memory_free_mib") if gpu else None,
            "effective_ram_available_bytes": ram.get("available_bytes"),
            "dev_shm_available_bytes": shm.get("available_bytes"),
            "output_mount": disk.get("target"),
            "output_available_bytes": disk.get("available_bytes"),
            "cgroup_version": cgroup.get("version"),
            "cgroup_identity": actual_identity,
            "cgroup_available_bytes": cgroup.get("available_bytes"),
            "cgroup_finite_limit": cgroup.get("finite_limit"),
        },
        "snapshot_sha256": _sha_bytes(_canonical(snapshot)),
        "failure_reasons": reasons,
        "observed_unix_time": time.time(),
    }


def _clean_common_env() -> dict[str, str]:
    allowed = {"HOME", "USER", "LOGNAME", "PATH", "LANG", "PYTHONNOUSERSITE"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    if "LD_LIBRARY_PATH" in os.environ:
        env["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    return env


def _renderer_env(row_id: str, row_root: Path, renderer: Path) -> dict[str, str]:
    env = _clean_common_env()
    ditto_root = Path(DITTO_SOURCE_ROOT)
    env.update(STRICT_ENV)
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join((str(Path(BOOTSTRAP_PATH).parent), str(ditto_root))),
        "CROSS_TFG_STRICT_STARTUP_POLICY": "1",
        "CROSS_TFG_RENDER_SEED": "20261003",
        "CROSS_TFG_ROW_ID": row_id,
        "CROSS_TFG_MODEL_ID": MODEL_ID,
        "CROSS_TFG_AUDIO_FRONTEND": "ditto",
        "CROSS_TFG_BACKEND_ENTRY": str(renderer.resolve()),
        "CROSS_TFG_DETERMINISM_RECEIPT_DIR": str((row_root / "runtime_determinism_receipts").resolve()),
        "CROSS_TFG_CONTROL_DIR": str(row_root.resolve()),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    return env


def _validate_strict_receipt(receipt_path: Path, row: Mapping[str, Any], renderer: Path) -> dict[str, Any]:
    raw = _read_regular(receipt_path, 65_536)
    record = json.loads(raw.decode("utf-8"))
    _verify_self(record, "receipt_sha256", str(receipt_path))
    if (record.get("schema") != BOOTSTRAP_RECEIPT_SCHEMA or record.get("status") != "PASS"
            or record.get("row_id") != row.get("row_id") or record.get("model_id") != MODEL_ID
            or record.get("audio_frontend") != "ditto" or record.get("seed") != row.get("seed")
            or record.get("python_executable") != DITTO_PYTHON
            or record.get("expected_backend_entry") != str(renderer.resolve())
            or record.get("entry_matches_backend") is not True
            or record.get("bootstrap_source_sha256") != BOOTSTRAP_SHA256
            or record.get("policy") != STRICT_POLICY):
        raise RuntimeError("Ditto strict startup receipt differs from the exact V100 row policy")
    return {"path": str(receipt_path.resolve()), "bytes": len(raw),
            "raw_sha256": _sha_bytes(raw), "receipt_sha256": record["receipt_sha256"],
            "pid": record.get("pid"), "entry_path": record.get("entry_path"),
            "python_executable": record.get("python_executable")}


def _run_ditto_generator(*, core: Any, row: Mapping[str, Any], model: Mapping[str, Any],
                         audio_path: Path, output_video: Path, work: Path,
                         args: argparse.Namespace, generator: Mapping[str, Any],
                         expected_frames_by_row_root: dict[str, int],
                         processed_rows: dict[str, int]) -> dict[str, Any]:
    if row.get("model_id") != MODEL_ID or model.get("model_id") != MODEL_ID:
        raise RuntimeError("Ditto generator received an unexpected model identity")
    renderer = Path(str(generator["renderer_path"]))
    audio_ref = core.read_json(audio_path.with_name(audio_path.name + ".lineage.json"))
    frame_binding = row.get("ditto_native_frame_binding", {})
    sample_count = audio_ref.get("sample_count")
    expected_frames = expected_native_frame_count(sample_count)
    if (frame_binding.get("sample_count") != sample_count
            or frame_binding.get("expected_frame_count") != expected_frames
            or frame_binding.get("fps_num") != FPS_NUM or frame_binding.get("fps_den") != FPS_DEN
            or audio_ref.get("sample_rate_hz") != SAMPLE_RATE or audio_ref.get("channels") != 1):
        raise RuntimeError("Ditto native frame binding or generator-condition PCM differs from ceil(N*25/16000)")
    if (row.get("source_face", {}).get("file_sha256") != FACE_SHA256
            or Path(str(row.get("source_face", {}).get("path", ""))).stat().st_size != FACE_BYTES
            or _sha_file(Path(str(row.get("source_face", {}).get("path", "")))) != FACE_SHA256):
        raise RuntimeError("Ditto row does not use the exact fixed primary portrait file")
    receipt_dir = work / "runtime_determinism_receipts"
    receipt_dir.mkdir(exist_ok=True)
    prior = {entry.name for entry in receipt_dir.iterdir()}
    if len(prior) != processed_rows["count"]:
        raise RuntimeError("Ditto strict receipt directory has an unexpected prior receipt count")
    report_path = output_video.with_suffix(".render.json")
    argv = [
        DITTO_PYTHON, str(renderer), "--model", "ditto", "--audio", str(audio_path),
        "--face", str(row["source_face"]["path"]), "--outfile", str(output_video),
        "--seed", str(row["seed"]), "--fps", "25", "--source-root", DITTO_SOURCE_ROOT,
        "--weights-root", DITTO_WEIGHTS_ROOT, "--config", DITTO_CONFIG,
        "--ffmpeg", str(args.ffmpeg), "--ffprobe", str(args.ffprobe),
        "--report", str(report_path), "--expected-audio-bytes", str(audio_ref["bytes"]),
        "--expected-audio-sha256", str(audio_ref["file_sha256"]),
        "--expected-audio-pcm-sha256", str(audio_ref["pcm_sha256"]),
        "--expected-audio-sample-count", str(sample_count),
        "--expected-frame-count", str(expected_frames), "--row-id", str(row["row_id"]),
    ]
    child_cwd = work / "generator_cwd"
    child_cwd.mkdir(exist_ok=False)
    stdout_path, stderr_path = work / "logs/generator.stdout.log", work / "logs/generator.stderr.log"
    execution = core.run_logged(argv, cwd=child_cwd, env=_renderer_env(str(row["row_id"]), work, renderer),
                                stdout_path=stdout_path, stderr_path=stderr_path,
                                timeout=args.generator_timeout)
    if execution.get("returncode") != 0 or not output_video.is_file() or output_video.stat().st_size == 0:
        raise RuntimeError(f"pinned Ditto renderer failed ({execution.get('returncode')})")
    if report_path.is_symlink() or not report_path.is_file():
        raise RuntimeError("pinned Ditto renderer omitted its create-only report")
    report, report_raw = _load_json(report_path, 4 * 1024 * 1024)
    validate_renderer_input_report(row, audio_ref=audio_ref, report=report,
                                   profile=generator["renderer_report_profile"])
    silent = report.get("silent_video", {})
    if (silent.get("path") != str(output_video.resolve(strict=True))
            or silent.get("bytes") != output_video.stat().st_size
            or silent.get("file_sha256") != _sha_file(output_video)):
        raise RuntimeError("Ditto renderer report does not bind the exact silent output bytes")
    current = {entry.name for entry in receipt_dir.iterdir()}
    added = current - prior
    if len(added) != 1:
        raise RuntimeError("each Ditto renderer process must create exactly one strict startup receipt")
    name = next(iter(added))
    if not name.endswith(".json") or not name[:-5].isdigit():
        raise RuntimeError("unexpected name in the row-scoped Ditto strict receipt directory")
    strict_receipt = _validate_strict_receipt(receipt_dir / name, row, renderer)
    report_receipt = report.get("strict_startup", {})
    if (report_receipt.get("raw_sha256") != strict_receipt["raw_sha256"]
            or report_receipt.get("receipt_sha256") != strict_receipt["receipt_sha256"]
            or report_receipt.get("path") != strict_receipt["path"]):
        raise RuntimeError("Ditto report does not cite the exact strict startup receipt bytes")
    processed_rows["count"] += 1
    expected_frames_by_row_root[str(work.resolve())] = expected_frames
    execution.update({
        "renderer_report": {"path": str(report_path.resolve()), "bytes": len(report_raw),
                            "raw_sha256": _sha_bytes(report_raw), "status": report["status"],
                            "profile": report["profile"]},
        "strict_startup_receipt": strict_receipt,
        "generator_input_audio": {"path": audio_ref.get("path"),
                                  "file_sha256": audio_ref.get("file_sha256"),
                                  "pcm_float32_sha256": audio_ref.get("pcm_sha256"),
                                  "sample_count": sample_count,
                                  "sample_rate_hz": SAMPLE_RATE, "channels": 1},
        "expected_frame_count": expected_frames,
        "native_renderer_status": report.get("status"),
        "native_renderer_profile": report.get("profile"),
        "native_render_report": report,
    })
    return execution


def _load_frozen_parent(path_text: str, expected_sha: str, rows_path_text: str,
                        expected_rows_sha: str) -> tuple[dict[str, Any], bytes, Path, list[dict[str, Any]], bytes, Path]:
    plan_path = Path(path_text).resolve(strict=True)
    rows_path = Path(rows_path_text).resolve(strict=True)
    plan_raw = _read_regular(plan_path)
    rows, rows_raw = _load_jsonl(rows_path)
    if _sha_bytes(plan_raw) != expected_sha or _sha_bytes(rows_raw) != expected_rows_sha:
        raise RuntimeError("frozen parent plan/rows raw SHA differs from CLI pin")
    parent = json.loads(plan_raw.decode("utf-8"))
    if not isinstance(parent, dict):
        raise RuntimeError("frozen parent plan is not an object")
    validate_fit_parent_rows(rows, parent.get("methods", []))
    return parent, plan_raw, plan_path, rows, rows_raw, rows_path


def _run(args: argparse.Namespace, own: argparse.Namespace) -> int:
    core_path = Path(own.stagea_core).resolve(strict=True)
    overlay_path = Path(__file__).resolve(strict=True)
    overlay_sha = _sha_file(overlay_path)
    if overlay_sha != own.expected_ditto_overlay_sha256:
        raise RuntimeError("Ditto adapter source differs from the launcher-bound source pin")
    core = _load_module("ditto_fit40_w2l_block_attempt02", core_path,
                        W2L_CORE_BYTES, own.expected_stagea_core_sha256)
    if own.expected_stagea_core_sha256 != W2L_CORE_SHA256:
        raise RuntimeError("W2L Fit block core does not match its frozen raw source pin")
    parent, parent_raw, parent_path, parent_rows, parent_rows_raw, parent_rows_path = _load_frozen_parent(
        own.parent_plan, own.expected_parent_plan_raw_sha256,
        own.parent_rows, own.expected_parent_rows_raw_sha256)
    stage_args = core.build_parser().parse_args(own.core_argv)
    plan_path = Path(stage_args.plan).resolve(strict=True)
    plan, plan_raw = _load_json(plan_path)
    if _sha_bytes(plan_raw) != stage_args.expected_plan_raw_sha256:
        raise RuntimeError("Ditto Stage-A plan raw SHA differs from caller")
    binding, binding_raw, binding_path = _load_binding(
        own.ditto_runtime_binding, own.expected_ditto_runtime_binding_raw_sha256, plan)
    planned_rows, planned_rows_raw = _load_jsonl(Path(stage_args.planned_rows).resolve(strict=True))
    if planned_rows != plan.get("rows"):
        raise RuntimeError("external Ditto planned rows differ from projected plan")
    _verify_parent_and_projection(plan, parent, parent_rows,
                                  parent_plan_raw=parent_raw, parent_rows_raw=parent_rows_raw)
    model, generator, renderer = _validate_model_and_binding(
        plan, binding, binding_raw, binding_path, parent, parent_raw, parent_path,
        parent_rows_raw, core_path, own.expected_stagea_core_sha256,
        overlay_path, overlay_sha)
    expected_rows = project_fit_rows(parent_rows, parent.get("methods", []))
    if planned_rows != expected_rows:
        raise RuntimeError("Ditto planned_rows.jsonl differs from the exact parent row projection")
    validate_block(plan, stage_args.block_start, stage_args.block_count)
    project_root = Path(stage_args.project_root).resolve(strict=True)
    if (Path(stage_args.syncnet_root).resolve(strict=True) != Path("/home/wjj/tts-audio/tts-exp/third_party/syncnet_python")
            or Path(stage_args.syncnet_checkpoint).is_symlink()
            or _sha_file(Path(stage_args.syncnet_checkpoint)) != SCORE_MODEL_SHA256
            or Path(stage_args.ffmpeg).is_symlink()
            or not Path(stage_args.ffmpeg).is_file()
            or _sha_file(Path(stage_args.ffmpeg)) != plan["protocol"]["syncnet"]["primary_audio_input"]["ffmpeg_sha256"]):
        raise RuntimeError("official SyncNet/checkpoint or FFmpeg differs from the frozen source-clock protocol")
    helper = _load_resource_helper(project_root, binding)

    # Stage-A parent is kept free of row-scoped native renderer identity. The
    # strict policy is installed only in each isolated Ditto child environment.
    for key in list(os.environ):
        if key.startswith("CROSS_TFG_"):
            os.environ.pop(key, None)
    out = Path(stage_args.output_root).resolve(strict=False)
    os.environ["CROSS_TFG_CAPACITY_STAGE"] = "fit40_ditto240"
    os.environ["CROSS_TFG_RUNTIME_BINDING_PATH"] = str(binding_path)
    os.environ["CROSS_TFG_RUNTIME_BINDING_RAW_SHA256"] = _sha_bytes(binding_raw)
    os.environ["CROSS_TFG_DETERMINISM_RECEIPT_DIR"] = str((out / "runtime_determinism_receipts").resolve(strict=False))
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    original_frame_clock = core._frame_clock
    expected_frames_by_row_root: dict[str, int] = {}
    processed_rows = {"count": 0}

    def _ditto_generator(**kwargs: Any) -> dict[str, Any]:
        return _run_ditto_generator(core=core, generator=generator,
                                    expected_frames_by_row_root=expected_frames_by_row_root,
                                    processed_rows=processed_rows, **kwargs)

    def _ditto_frame_clock(path: Path, ffprobe: str, fps_num: int, fps_den: int):
        frames, stream = original_frame_clock(path, ffprobe, fps_num, fps_den)
        row_root = str(Path(path).resolve(strict=False).parent.parent)
        expected = expected_frames_by_row_root.get(row_root)
        if expected is not None:
            if len(frames) != expected:
                raise RuntimeError("decoded Ditto silent-video frame count differs from exact ceil(N*25/16000)")
            for index, frame in enumerate(frames):
                if (frame.get("frame_index") != index
                        or frame.get("timestamp_num") != index or frame.get("timestamp_den") != FPS_NUM):
                    if not (index == 0 and frame.get("timestamp_num") == 0 and frame.get("timestamp_den") == 1):
                        raise RuntimeError("decoded Ditto output PTS differs from the exact 25 fps source clock")
        return frames, stream

    def _resource_gate(plan_value: Mapping[str, Any], output_root: Path,
                       first_index: int, whole_stage: bool) -> dict[str, Any]:
        observation = _capacity_observation(plan_value, binding, helper, output_root,
                                            first_index, whole_stage=whole_stage)
        return {
            "schema": STAGE_GATE_SCHEMA if whole_stage else ROW_GATE_SCHEMA,
            "run_id": plan_value.get("run_id"), "model_id": MODEL_ID,
            "audio_frontend": FRONTEND,
            "scientific_input_pins_sha256": plan_value.get("scientific_input_pins_sha256"),
            "runtime_binding_raw_sha256": _sha_bytes(binding_raw),
            **observation,
        }

    def _stage_gate(plan_value: Mapping[str, Any], stage_args_value: argparse.Namespace,
                    output_root: Path) -> dict[str, Any]:
        return _resource_gate(plan_value, output_root,
                              first_index=int(stage_args_value.block_start), whole_stage=True)

    def _row_gate(plan_value: Mapping[str, Any], row: Mapping[str, Any],
                  stage_args_value: argparse.Namespace, output_root: Path) -> dict[str, Any]:
        try:
            first_index = next(i for i, item in enumerate(plan_value["rows"])
                               if item.get("row_id") == row.get("row_id"))
        except StopIteration as exc:
            raise RuntimeError("Ditto row is absent from the full ordered 240-row parent projection") from exc
        return _resource_gate(plan_value, output_root, first_index, whole_stage=False)

    original_write_json = core.write_json
    def _write_json_with_ditto_scope(path: Path, value: Any) -> None:
        if Path(path).name == "result_manifest.json" and isinstance(value, dict):
            value["parent_scope"] = (
                "one six-row Ditto block from the frozen 240-row Fit plan; "
                "not full-backend or three-model family completion"
            )
            value["backend_id"] = MODEL_ID
            value["backend_frontend"] = FRONTEND
            value["parent_plan_raw_sha256"] = _sha_bytes(parent_raw)
            value["parent_plan_payload_sha256"] = parent.get("plan_sha256")
            value["block_overlay_sha256"] = overlay_sha
            value["manifest_sha256"] = _sha_bytes(_canonical({k: v for k, v in value.items()
                                                                 if k != "manifest_sha256"}))
        original_write_json(path, value)

    core.STAGE_RESOURCE_GATE_SCHEMA = STAGE_GATE_SCHEMA
    core.ROW_RESOURCE_GATE_SCHEMA = ROW_GATE_SCHEMA
    core._stage_resource_preflight = _stage_gate
    core._row_resource_gate = _row_gate
    core._run_generator = _ditto_generator
    core._frame_clock = _ditto_frame_clock
    core.write_json = _write_json_with_ditto_scope
    return core.run(stage_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stagea-core", required=True)
    parser.add_argument("--expected-stagea-core-sha256", required=True)
    parser.add_argument("--expected-ditto-overlay-sha256", required=True)
    parser.add_argument("--parent-plan", required=True)
    parser.add_argument("--expected-parent-plan-raw-sha256", required=True)
    parser.add_argument("--parent-rows", required=True)
    parser.add_argument("--expected-parent-rows-raw-sha256", required=True)
    parser.add_argument("--ditto-runtime-binding", required=True)
    parser.add_argument("--expected-ditto-runtime-binding-raw-sha256", required=True)
    parser.add_argument("core_argv", nargs=argparse.REMAINDER,
                        help="remaining arguments are passed to the pinned W2L Fit block executor")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    own = parser.parse_args(argv)
    if own.core_argv and own.core_argv[0] == "--":
        own.core_argv = own.core_argv[1:]
    if not own.core_argv:
        parser.error("pinned W2L core arguments are required after the adapter arguments")
    core_argv = list(own.core_argv)
    # parse_core_argv is intentionally performed only after the exact core source
    # has been loaded. It does not import or invoke the Ditto renderer.
    core_path = Path(own.stagea_core).resolve(strict=True)
    if (core_path.is_symlink() or not core_path.is_file()
            or core_path.stat().st_size != W2L_CORE_BYTES
            or _sha_file(core_path) != own.expected_stagea_core_sha256
            or own.expected_stagea_core_sha256 != W2L_CORE_SHA256):
        raise RuntimeError("W2L Stage-A scoring core does not match the reviewed immutable source pin")
    spec = importlib.util.spec_from_file_location("ditto_fit40_core_parser", str(core_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the pinned Stage-A parser")
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    stage_args = core.build_parser().parse_args(core_argv)
    # Reparse only our flags from the original argv; the argparse remainder has
    # already been separated, preserving exact option/value boundaries.
    own.core_argv = core_argv
    return _run(stage_args, own)


if __name__ == "__main__":
    raise SystemExit(main())
