#!/usr/bin/env python3
"""Build a Fit40 Ditto 240-row projection from pinned producer evidence.

This is a metadata-only plan builder.  It does not import a model, initialize
CUDA, render media, score SyncNet, access Dev/Confirm data, or select a method.
The emitted capacity estimate is an estimate from measured repeat output bytes;
the Stage-A executor still performs live resource gates before work starts and
before each row.
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
import stat
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
SCRATCH = ROOT / ".codex/scratch/v100_ditto_fit40_plan_attempt04"
PARENT_DIR = ROOT / ".codex/scratch/v100_fit_stagea_formal_w2l_fit_slice_attempt08/project_payload_attempt05"
PARENT_PLAN_DEFAULT = PARENT_DIR / "fit40_w2l240_plan.json"
PARENT_ROWS_DEFAULT = PARENT_DIR / "planned_rows.jsonl"
PARENT_BINDING_DEFAULT = PARENT_DIR / "control/w2l_runtime_binding.json"
RENDER_RECEIPT_DEFAULT = ROOT / ".codex/scratch/v100_ditto_native_receipt_readback_attempt01/run_receipt_attempt02.json"
SCORE_RECEIPT_DEFAULT = ROOT / ".codex/scratch/v100_ditto_identity_repeat_score_only_attempt04/readback_attempt01/ditto_v100_identity_repeat_score_only_attempt04/score_only_run_receipt_attempt04.json"
REPEAT_QC_DEFAULT = ROOT / ".codex/scratch/root_ditto_actual_repeat_readback_qc_attempt01.json"
RESOURCE_OBSERVATION_DEFAULT = ROOT / ".codex/scratch/v100_ditto_fit40_plan_attempt01/resource_observation_attempt01.json"
RENDERER_DEFAULT = ROOT / ".codex/scratch/v100_ditto_identity_repeat_attempt02/render_silent_video_v100_attempt02.py"
STAGEA_CORE_DEFAULT = PARENT_DIR / "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_w2l_block_attempt02.py"
OVERLAY_DEFAULT = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_ditto_fit40_block_attempt05.py"
RESOURCE_HELPER_DEFAULT = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/v3_dev180_newhost_latebind_packet_attempt02.py"
PCM16_HELPER_DEFAULT = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/stagea_pcm16_lineage.py"
SYNCNET_EVALUATOR_DEFAULT = ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py"

PARENT_PLAN_BYTES = 902_628
PARENT_PLAN_SHA256 = "25f49cf6e87800da57fc797210b5d0c4bb6bb3396af04f9f9e9ff1aec0146832"
PARENT_ROWS_BYTES = 678_368
PARENT_ROWS_SHA256 = "54a69d99b34cf70b02f58b4a4bae032bcf5446fb9777d7a5594f6cc45ffeffbe"
PARENT_BINDING_BYTES = 2_884
PARENT_BINDING_SHA256 = "a6638095c8ca05ffc6c48fe3d1017e9fd2f3d5da01b3279c01ca97b3e0a67d58"
RENDER_RECEIPT_BYTES = 64_682
RENDER_RECEIPT_SHA256 = "d94d85ee6ed2642b94d285455ff402c2d6e26c3807038d2a069fd14fafb76a7b"
SCORE_RECEIPT_BYTES = 278_067
SCORE_RECEIPT_SHA256 = "d2a869edf0faa9a95bc1b286bad7aad2f418f7b3bad7e7972f2115f5d79c9441"
REPEAT_QC_BYTES = 1_968
REPEAT_QC_SHA256 = "11a5e054fcb0893a494a8d75d098947b5e87e8fb9111f1db6dd10b3fd7350d50"
RESOURCE_OBSERVATION_BYTES = 2_327
RESOURCE_OBSERVATION_SHA256 = "012ad0fc0fad2e36555e6a9c03f5e52deffe5d2402e53062018bc000ec1a8e6f"

PARENT_RUNTIME_BINDING_SCHEMA = "cross_tfg_fit40_w2l_runtime_binding_v1"
RUNTIME_BINDING_SCHEMA = "cross_tfg_fit40_ditto_runtime_binding_v1"
PROJECTION_SCHEMA = "cross_tfg_fit40_ditto_parent_projection_v1"
RESOURCE_HELPER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/v3_dev180_newhost_latebind_packet_attempt02.py"
PROJECT_ROOT_REMOTE = "/home/wjj/tts-audio/tts-exp"
STAGE_ROOT_REMOTE = "/dev/shm/fit40_ditto240_stage_attempt04"
STAGE_PROJECT_REMOTE = STAGE_ROOT_REMOTE + "/project"
PARENT_PLAN_REMOTE = "/dev/shm/fit40_w2l240_stage_attempt05/project/fit40_w2l240_plan.json"
PARENT_ROWS_REMOTE = "/dev/shm/fit40_w2l240_stage_attempt05/project/planned_rows.jsonl"
RUN_ID = "fit40_ditto240_20261009_attempt04"

MODEL_ID = "ditto_hubert_fixed_portrait_attempt18"
FRONTEND = "ditto_hubert_onnx_cpu_torch_cuda_static_portrait_attempt18"
RENDERER_RESULT_PROFILE = "ditto_hubert_onnx_cpu_renderer_torch_cuda"
DITTO_COMMIT = "c3e47eee2e626500017a0556b470d6d4182f85e8"
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
SCORE_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
SYNCNET_ROOT = PROJECT_ROOT_REMOTE + "/third_party/syncnet_python"
SYNCNET_CHECKPOINT = SYNCNET_ROOT + "/data/syncnet_v2.model"
SYNCNET_EVALUATOR_REL = "scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py"
SYNCNET_EVALUATOR_BYTES = 6_281
SYNCNET_EVALUATOR_SHA256 = "b15762e1837fd28eaba8fe9e261ffb35b164246da555cb3a7e3bbc7c645f818b"
PCM16_HELPER_BYTES = 4_528
PCM16_HELPER_SHA256 = "f634bad9c287a2f7f8cd8ffa62674bdd9e97dbf4b0f00dbf44725e4ca4ca36a0"
SYNCNET_PYTHON = "/home/wjj/.venvs/w2l_syncnet_v100_attempt01/bin/python"
SYNCNET_PYTHON_LINK_TARGET = "/home/wjj/miniconda3/envs/autoavsr/bin/python"
SYNCNET_PYTHON_RESOLVED = "/home/wjj/miniconda3/envs/autoavsr/bin/python3.8"
SYNCNET_PYTHON_RESOLVED_BYTES = 15_160_256
SYNCNET_PYTHON_RESOLVED_SHA256 = "bfedc6cf63e5e438ab400aa18b92fd295e38506ecc079b8b898ac4a0a9163010"
SYNCNET_PYTHON_PREFIX = "/home/wjj/.venvs/w2l_syncnet_v100_attempt01"
SYNCNET_PYTHON_BASE_PREFIX = "/home/wjj/miniconda3/envs/autoavsr"
SYNCNET_PYTHON_VERSION = [3, 8, 20]
FFMPEG_PATH = "/usr/bin/ffmpeg"
FFMPEG_BYTES = 342_488
FFMPEG_SHA256 = "ed16af623947494a72e284b6eb8ff225f2da22b38b5d5069c2fd4b4ba3384e41"
FFPROBE_PATH = "/usr/bin/ffprobe"
FFPROBE_BYTES = 191_888
FFPROBE_SHA256 = "272f6ebc634a63d9c8b4ca68e964119d980f25154e5aa2c35e5487da48e9a58f"
FACE_SHA256 = "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6"
FACE_BYTES = 253_479
MODEL_ASSET_SHA256 = "dc29eacae64c53a741e58a28e94081c5b1d0a04714f0b478fc7114c6a50a8478"
RESERVE_BYTES = 5_368_709_120
MIN_GPU_FREE_MIB = 10_000
MIN_RAM_AVAILABLE_BYTES = 8_589_934_592
SLACK_MULTIPLIER = 3.0
PILOT_SAMPLE_COUNT = 92_225
EXPECTED_PILOT_BYTES = 2_630_393
REMOTE_RENDER_RECEIPT = "/dev/shm/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/ditto_v100_identity_repeat_attempt02/run_receipt_attempt02.json"
REMOTE_SCORE_RECEIPT = "/dev/shm/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/ditto_v100_identity_repeat_score_only_attempt04/score_only_run_receipt_attempt04.json"
ROOT_REPEAT_QC_LOCAL = ".codex/scratch/root_ditto_actual_repeat_readback_qc_attempt01.json"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_regular(path: Path, *, max_bytes: int, expected_bytes: int | None = None,
                 expected_sha256: str | None = None) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"expected a regular non-symlink input: {path}")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
        raise ValueError(f"input is not regular or exceeds its byte limit: {path}")
    if expected_bytes is not None and before.st_size != expected_bytes:
        raise ValueError(f"input byte count differs from its sealed pin: {path}")
    raw = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or len(raw) != before.st_size):
        raise ValueError(f"input changed while being read: {path}")
    digest = sha_bytes(raw)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"input SHA-256 differs from its sealed pin: {path}")
    return raw


def load_json(path: Path, *, expected_bytes: int, expected_sha256: str) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path, max_bytes=16 * 1024 * 1024,
                       expected_bytes=expected_bytes, expected_sha256=expected_sha256)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value, raw


def verify_self(value: Mapping[str, Any], field: str, label: str) -> None:
    expected = value.get(field)
    actual = sha_bytes(canonical({key: item for key, item in value.items() if key != field}))
    if expected != actual:
        raise ValueError(f"{label} canonical self-hash does not verify")


def load_module(path: Path, name: str, expected_bytes: int, expected_sha256: str) -> Any:
    source = read_regular(path, max_bytes=2 * 1024 * 1024,
                          expected_bytes=expected_bytes, expected_sha256=expected_sha256)
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load exact source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path, *, expected_bytes: int, expected_sha256: str) -> tuple[list[dict[str, Any]], bytes]:
    raw = read_regular(path, max_bytes=4 * 1024 * 1024,
                       expected_bytes=expected_bytes, expected_sha256=expected_sha256)
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("parent planned_rows.jsonl contains a non-object row")
    return rows, raw


def literal_renderer_pins(renderer_path: Path) -> tuple[dict[str, str], dict[str, tuple[int, str]]]:
    tree = ast.parse(read_regular(renderer_path, max_bytes=2 * 1024 * 1024,
                                 expected_bytes=40_010,
                                 expected_sha256="7b37381777ad8cabf3665a7ab636295bc779b4a1fde5fa15dd485360772a53e4").decode("utf-8"),
                     filename=str(renderer_path))
    assignments: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in {"DITTO_SOURCE_PINS", "DITTO_WEIGHT_PINS"}:
                    assignments[target.id] = ast.literal_eval(node.value)
    source = assignments.get("DITTO_SOURCE_PINS")
    weights = assignments.get("DITTO_WEIGHT_PINS")
    if not isinstance(source, dict) or not isinstance(weights, dict):
        raise ValueError("renderer source lacks literal source/weight maps")
    return source, {str(key): (int(value[0]), str(value[1])) for key, value in weights.items()}


def validate_actual_repeat_evidence(render_receipt: Mapping[str, Any], score_receipt: Mapping[str, Any],
                                    repeat_qc: Mapping[str, Any], render_raw: bytes,
                                    score_raw: bytes) -> dict[str, Any]:
    """Keep the failed overall renderer attempt failed while accepting its actual render rows."""
    if (render_receipt.get("schema") != "ditto_v100_identity_repeat_run_receipt_attempt02_v1"
            or render_receipt.get("status") != "FAILED_ENGINEERING_ONLY_REPEAT"
            or render_receipt.get("scientific_claim") is not False
            or render_receipt.get("selection_eligible") is not False
            or render_receipt.get("training_or_api") is not False):
        raise ValueError("attempt02 renderer receipt does not match its actual failed engineering-only status")
    render_rows = render_receipt.get("rows")
    if not isinstance(render_rows, list) or len(render_rows) != 2:
        raise ValueError("attempt02 renderer receipt must contain exactly two preserved rows")
    source_and_weights = None
    render_row_evidence = []
    for row in render_rows:
        report = row.get("renderer_report_payload", {})
        execution = row.get("execution", {})
        expected_audio = render_receipt.get("row_audio", {})
        expected_face = render_receipt.get("face", {})
        if (row.get("status") != "FAILED_ENGINEERING_ONLY"
                or not isinstance(report, Mapping)
                or report.get("status") != "PASS_ENGINEERING_ONLY"
                or report.get("model") != "ditto"
                or report.get("profile") != "ditto_hubert_onnx_cpu_renderer_torch_cuda"
                or report.get("row_id") != row.get("row_id")
                or not isinstance(report.get("seed"), Mapping)
                or report.get("seed", {}).get("seed") != render_receipt.get("seed")
                or report.get("seed", {}).get("python_numpy_torch_seeded") is not True
                or report.get("seed", {}).get("cuda_available") is not True
                or report.get("seed", {}).get("cudnn_deterministic") is not True
                or report.get("seed", {}).get("cudnn_benchmark") is not False
                or report.get("seed", {}).get("tf32") is not False
                or execution.get("returncode") != 0 or execution.get("timed_out") is not False
                or report.get("audio", {}).get("file_sha256") != expected_audio.get("file", {}).get("sha256")
                or report.get("audio", {}).get("pcm_float32_sha256") != expected_audio.get("pcm_float32_sha256")
                or report.get("audio", {}).get("sample_count") != expected_audio.get("sample_count")
                or not isinstance(report.get("face"), Mapping)
                or report.get("face", {}).get("path") != expected_face.get("path")
                or report.get("face", {}).get("bytes") != expected_face.get("bytes")
                or report.get("face", {}).get("file_sha256") != expected_face.get("sha256")
                or row.get("silent_video", {}).get("sha256") != report.get("silent_video", {}).get("file_sha256")):
            raise ValueError("attempt02 row lacks a successful native render bound to exact PCM/face inputs")
        observed_source = report.get("source_and_weights")
        if not isinstance(observed_source, Mapping):
            raise ValueError("render report lacks its exact source/weight identity maps")
        if source_and_weights is None:
            source_and_weights = copy.deepcopy(dict(observed_source))
        elif source_and_weights != observed_source:
            raise ValueError("two attempt02 render rows do not share the same source/weight identity")
        render_row_evidence.append({
            "row_id": row["row_id"],
            "row_status_preserved": row["status"],
            "execution_returncode": execution["returncode"],
            "renderer_report_status": report["status"],
            "renderer_report_ref": row.get("renderer_report"),
            "silent_video_ref": row.get("silent_video"),
        })
    if render_rows[0].get("silent_video", {}).get("sha256") != render_rows[1].get("silent_video", {}).get("sha256"):
        raise ValueError("attempt02 renderer repeats did not produce the same pinned silent video")
    if (score_receipt.get("schema") != "ditto_v100_identity_repeat_score_only_attempt04_v1"
            or score_receipt.get("status") != "PASS_ENGINEERING_ONLY_EXACT_REPEAT"
            or score_receipt.get("scientific_claim") is not False
            or score_receipt.get("selection_eligible") is not False
            or score_receipt.get("renderer_invoked") is not False
            or score_receipt.get("training_or_api") is not False
            or score_receipt.get("parent_attempt02_receipt") != {
                "bytes": len(render_raw), "path": REMOTE_RENDER_RECEIPT, "sha256": sha_bytes(render_raw)}):
        raise ValueError("score-only receipt does not bind the exact preserved attempt02 receipt")
    score_rows = score_receipt.get("rows")
    comparison = score_receipt.get("repeat_comparison", {})
    if (not isinstance(score_rows, list) or len(score_rows) != 2
            or any(row.get("status") != "PASS_ENGINEERING_ONLY" for row in score_rows)
            or comparison.get("expected_rows") != 2
            or comparison.get("row_statuses") != ["PASS_ENGINEERING_ONLY", "PASS_ENGINEERING_ONLY"]
            or comparison.get("all_score_and_query_arrays_exact_equal") is not True
            or comparison.get("same_preserved_silent_video_sha256") is not True
            or comparison.get("actual_query_count_repeat01") != 139
            or comparison.get("actual_query_count_repeat02") != 139
            or comparison.get("distance_matrix_max_abs_difference") != 0.0):
        raise ValueError("score-only receipt failed its actual complete-array exact-repeat gate")
    score_input = score_receipt.get("input_audio", {})
    face_input = score_receipt.get("input_face", {})
    parent_audio = render_receipt.get("row_audio", {})
    parent_face = render_receipt.get("face", {})
    if (score_input.get("file") != parent_audio.get("file")
            or score_input.get("canonical_pcm_f32_sha256") != parent_audio.get("pcm_float32_sha256")
            or score_input.get("sample_count") != parent_audio.get("sample_count")
            or score_input.get("sample_rate_hz") != 16_000
            or face_input != parent_face):
        raise ValueError("score-only rerun does not bind the exact original audio/face from attempt02")
    if (repeat_qc.get("schema") != "root_ditto_actual_repeat_readback_qc_v1"
            or repeat_qc.get("status") != "PASS_ENGINEERING_REPEAT_ORIGINAL_PCM_AND_DECODED_SUPPORT"
            or repeat_qc.get("run_receipt_raw_sha256") != sha_bytes(score_raw)
            or repeat_qc.get("all_seven_arrays_and_video_elementary_exact_repeat") is not True):
        raise ValueError("independent root repeat-QC does not bind the exact score-only receipt")
    qc_rows = repeat_qc.get("rows")
    if not isinstance(qc_rows, list) or len(qc_rows) != 2:
        raise ValueError("independent root QC must cover both repeat rows")
    for qc_row, score_row in zip(qc_rows, score_rows):
        payload = score_row.get("receipt_payload", {})
        score_arrays = payload.get("score_arrays", {}).get("arrays", {})
        matrix_sha = score_arrays.get("distance_by_query_lag", {}).get("array_bytes_sha256")
        if (qc_row.get("row_id") != score_row.get("row_id")
                or qc_row.get("matrix_shape") != [139, 31]
                or qc_row.get("matrix_sha256") != matrix_sha
                or qc_row.get("decoded_native_frames") != 145
                or qc_row.get("five_frame_supported_queries") != 139
                or qc_row.get("native_video_exact_pts") is not True
                or qc_row.get("full_original_f32_mux_readback_N") != score_input.get("sample_count")
                or qc_row.get("original_pcm16_slice_and_scoring_mux_readback") is not True):
            raise ValueError("independent root QC row differs from its actual score receipt fields")
    assert isinstance(source_and_weights, dict)
    return {
        "parent_render_receipt": {"path": REMOTE_RENDER_RECEIPT, "bytes": len(render_raw),
                                  "raw_sha256": sha_bytes(render_raw), "status_preserved": render_receipt["status"]},
        "score_only_receipt": {"path": REMOTE_SCORE_RECEIPT, "bytes": len(score_raw),
                               "raw_sha256": sha_bytes(score_raw), "status": score_receipt["status"]},
        "independent_root_repeat_qc": {"path": ROOT_REPEAT_QC_LOCAL,
                                       "raw_sha256": REPEAT_QC_SHA256,
                                       "self_sha256": repeat_qc.get("self_sha256"),
                                       "status": repeat_qc["status"]},
        "renderer_rows": render_row_evidence,
        "actual_repeat_rows": [
            {"row_id": row.get("row_id"), "status": row.get("status"),
             "receipt": row.get("receipt"),
             "query_count": row.get("receipt_payload", {}).get("query_support", {}).get("actual_query_count"),
             "matrix_shape": row.get("receipt_payload", {}).get("score_arrays", {}).get("arrays", {}).get("distance_by_query_lag", {}).get("shape"),
             "matrix_sha256": row.get("receipt_payload", {}).get("score_arrays", {}).get("arrays", {}).get("distance_by_query_lag", {}).get("array_bytes_sha256")}
            for row in score_rows
        ],
        "source_and_weights": source_and_weights,
        "qualification_scope": "V100 engineering repeat and official SyncNet evidence only; not a Fit efficacy result, full backend completion, or selection qualification",
    }


def validate_resource_observation(record: Mapping[str, Any]) -> dict[str, Any]:
    if (record.get("schema") != "v100_ditto_fit40_actual_resource_observation_v1"
            or record.get("status") != "READ_ONLY_OBSERVED"):
        raise ValueError("resource observation schema/status is invalid")
    verify_self(record, "self_sha256", "resource observation")
    obs = record.get("observation", {})
    if not isinstance(obs, Mapping):
        raise ValueError("resource observation body is malformed")
    gpu = next((item for item in obs.get("gpus", []) if item.get("uuid") == GPU_UUID), None)
    ram, shm, disk = obs.get("ram", {}), obs.get("shm", {}), obs.get("output_disk", {})
    cgroup = ram.get("cgroup", {})
    identity = cgroup.get("identity", {})
    if (gpu is None or gpu.get("name") != GPU_NAME or gpu.get("memory_free_mib", 0) < MIN_GPU_FREE_MIB
            or ram.get("available_bytes", 0) < MIN_RAM_AVAILABLE_BYTES
            or shm.get("available_bytes", 0) <= 0
            or disk.get("target") != "/dev/shm"
            or identity.get("mount_target") != "/sys/fs/cgroup"
            or identity.get("mount_root") != "/"
            or cgroup.get("version") != "v2"
            or cgroup.get("finite_limit") is not False
            or cgroup.get("available_bytes") is not None):
        raise ValueError("measured V100, /dev/shm, host RAM, or cgroup identity does not meet the pinned resource contract")
    pilots = obs.get("pilot_rows")
    if (not isinstance(pilots, list) or len(pilots) != 2
            or any(item.get("bytes") <= 0 or item.get("files") <= 0 for item in pilots)):
        raise ValueError("resource measurement lacks the two actual row-tree size observations")
    if max(item["bytes"] for item in pilots) != EXPECTED_PILOT_BYTES:
        raise ValueError("measured repeat row bytes differ from the exact pilot pin")
    return copy.deepcopy(dict(obs))


def estimate_remaining_bytes(rows: Sequence[Mapping[str, Any]], pilot_bytes: int,
                              pilot_sample_count: int, slack_multiplier: float = SLACK_MULTIPLIER) -> int:
    if (type(pilot_bytes) is not int or pilot_bytes <= 0
            or type(pilot_sample_count) is not int or pilot_sample_count <= 0
            or type(slack_multiplier) not in (int, float) or slack_multiplier < 1):
        raise ValueError("capacity estimation inputs must be positive and the slack at least one")
    scaled = 0
    for row in rows:
        count = row.get("natural_pcm", {}).get("sample_count")
        if type(count) is not int or count <= 0:
            raise ValueError("planned row lacks a positive original source sample count")
        scaled += math.ceil(pilot_bytes * count / pilot_sample_count)
    return int(math.ceil(scaled * float(slack_multiplier)))


def build_runtime_binding(*, run_id: str, scientific_input_pins_sha256: str,
                          qualification_evidence: Mapping[str, Any], source_and_weights: Mapping[str, Any],
                          capacity_resources: Mapping[str, Any], output_path_remote: str,
                          overlay_bytes: int, overlay_sha256: str) -> dict[str, Any]:
    runtime_qualification = qualification_evidence["score_only_receipt"]
    binding: dict[str, Any] = {
        "schema": RUNTIME_BINDING_SCHEMA,
        "status": "PASS",
        "binding_scope": "exact Ditto V100 source/runtime and engineering-repeat binding; Fit40 execution remains not started",
        "model_id": MODEL_ID,
        "audio_frontend": FRONTEND,
        "run_id": run_id,
        "gpu_uuid": GPU_UUID,
        "scientific_input_pins_sha256": scientific_input_pins_sha256,
        "runtime": {
            "python": DITTO_PYTHON, "python_version": PYTHON_VERSION,
            "torch": TORCH_VERSION, "cuda": CUDA_VERSION,
            "onnxruntime": ORT_VERSION, "device": GPU_NAME,
            "compute_capability": [7, 0], "gpu_uuid": GPU_UUID,
            "onnxruntime_providers": ["CPUExecutionProvider"],
        },
        "runtime_qualification": {
            "path": runtime_qualification["path"],
            "raw_sha256": runtime_qualification["raw_sha256"],
            "status": runtime_qualification["status"],
            "scope": qualification_evidence["qualification_scope"],
            "selection_eligible": False,
        },
        "qualification_evidence": copy.deepcopy(dict(qualification_evidence)),
        "source_and_weights": copy.deepcopy(dict(source_and_weights)),
        "source_closure": {
            "ditto_block_overlay": {"path": STAGE_PROJECT_REMOTE + "/scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_ditto_fit40_block_attempt05.py",
                                    "bytes": overlay_bytes, "sha256": overlay_sha256, "staged_create_only": True},
            "stagea_core": {"path": STAGE_PROJECT_REMOTE + "/scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_w2l_block_attempt02.py",
                            "bytes": 96_537,
                            "sha256": "6969c6e952d46b6753d279b0e2447aa71c7b8e92967cd42ee57dc8122da5616e",
                            "staged_create_only": True},
            "resource_helper": {"path": PROJECT_ROOT_REMOTE + "/" + RESOURCE_HELPER_REL,
                                "bytes": 62_830,
                                "sha256": "406f95e4a262fb62bb02bc67249f3e6a7bf211b803a014622e042329d772fdf7",
                                "observed_missing_remote_at_source_readback": True,
                                "staged_create_only_required": True},
            "pcm16_lineage_helper": {"path": PROJECT_ROOT_REMOTE + "/scripts/experiments/cross_tfg_audio_head_20261003/stagea_pcm16_lineage.py",
                                     "bytes": 4_528,
                                     "sha256": "f634bad9c287a2f7f8cd8ffa62674bdd9e97dbf4b0f00dbf44725e4ca4ca36a0"},
            "syncnet_evaluator": {"path": PROJECT_ROOT_REMOTE + "/" + SYNCNET_EVALUATOR_REL,
                                  "bytes": SYNCNET_EVALUATOR_BYTES,
                                  "sha256": SYNCNET_EVALUATOR_SHA256},
        },
        "resources": copy.deepcopy(dict(capacity_resources)),
    }
    binding["binding_sha256"] = sha_bytes(canonical(binding))
    if output_path_remote != STAGE_PROJECT_REMOTE + "/control/ditto_runtime_binding.json":
        raise ValueError("runtime binding path differs from its fresh stage namespace")
    return binding


def build_plan(parent: Mapping[str, Any], parent_raw: bytes, parent_rows: Sequence[Mapping[str, Any]],
               parent_rows_raw: bytes, parent_binding: Mapping[str, Any], parent_binding_raw: bytes,
               evidence: Mapping[str, Any], resource_observation: Mapping[str, Any],
               adapter_module: Any, *, stagea_core_bytes: int, stagea_core_sha256: str,
               overlay_bytes: int, overlay_sha256: str, resource_helper_bytes: int,
               resource_helper_sha256: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if (parent.get("schema_version") != "cross-tfg-audio-eval-v1" or parent.get("status") != "FROZEN"
            or parent.get("run_id") != "fit40_w2l240_20261009_attempt05"
            or parent.get("plan_sha256") != sha_bytes(canonical({k: v for k, v in parent.items() if k != "plan_sha256"}))
            or parent.get("rows") != list(parent_rows)):
        raise ValueError("actual W2L parent plan is not the exact frozen self-hashed 240-row plan")
    if (parent_binding.get("schema") != PARENT_RUNTIME_BINDING_SCHEMA or parent_binding.get("status") != "PASS"
            or parent_binding.get("run_id") != parent.get("run_id")
            or parent_binding.get("scientific_input_pins_sha256") != parent.get("scientific_input_pins_sha256")):
        raise ValueError("actual W2L parent runtime binding does not match the frozen plan")
    verify_self(parent_binding, "binding_sha256", "parent W2L runtime binding")
    if parent_binding.get("runtime_qualification_raw_sha256") != parent.get("execution", {}).get("runtime_qualification", {}).get("raw_sha256"):
        raise ValueError("W2L runtime binding qualification lineage differs from the frozen parent plan")
    methods = parent.get("methods")
    rows = adapter_module.project_fit_rows(parent_rows, methods)
    if len(rows) != 240:
        raise ValueError("Fit40 projection must retain all 240 rows")
    pilot_items = resource_observation["pilot_rows"]
    pilot_bytes = max(item["bytes"] for item in pilot_items)
    estimated = estimate_remaining_bytes(rows, pilot_bytes, PILOT_SAMPLE_COUNT, SLACK_MULTIPLIER)
    total_required = estimated + RESERVE_BYTES
    if resource_observation["shm"]["available_bytes"] < total_required:
        raise ValueError("measured /dev/shm free bytes do not cover estimate plus five-GiB reserve")
    if resource_observation["output_disk"]["target"] != "/dev/shm":
        raise ValueError("measured output mount is not /dev/shm")

    render_sources = evidence["source_and_weights"]
    source_files = render_sources["source_files"]
    weights = render_sources["weights"]
    score_preprocessing_sha256 = parent.get("protocol", {}).get("generator", {}).get("score_audio_preprocessing_sha256")
    if (not isinstance(score_preprocessing_sha256, str) or len(score_preprocessing_sha256) != 64
            or any(char not in "0123456789abcdef" for char in score_preprocessing_sha256)):
        raise ValueError("frozen parent lacks its exact SyncNet score preprocessing SHA")
    model = {
        "model_id": MODEL_ID,
        "audio_frontend": FRONTEND,
        "role": "candidate",
        "model_sha256": MODEL_ASSET_SHA256,
        "checkpoint_sha256": MODEL_ASSET_SHA256,
        "score_model_sha256": parent["models"][0]["score_model_sha256"],
        "preprocessing_sha256": score_preprocessing_sha256,
        "git_head": DITTO_COMMIT,
        "fps_num": 25,
        "fps_den": 1,
        "source_sha256": copy.deepcopy(source_files),
        "model_assets": {"source_files": copy.deepcopy(source_files), "weights": copy.deepcopy(weights)},
        "generator": {
            "contract": "cross_tfg_silent_video_cli_v1",
            "renderer_path": DITTO_DEPLOY_ROOT + "/job_control/ditto_v100_identity_repeat_attempt02/render_silent_video_v100_attempt02.py",
            "renderer_sha256": "7b37381777ad8cabf3665a7ab636295bc779b4a1fde5fa15dd485360772a53e4",
            "renderer_bytes": 40_010,
            "renderer_report_profile": RENDERER_RESULT_PROFILE,
            "python": DITTO_PYTHON,
            "source_root": DITTO_SOURCE_ROOT,
            "weights_root": DITTO_WEIGHTS_ROOT,
            "config_path": DITTO_CONFIG,
            "strict_bootstrap_path": BOOTSTRAP_PATH,
            "strict_bootstrap_sha256": BOOTSTRAP_SHA256,
            "strict_bootstrap_bytes": BOOTSTRAP_BYTES,
            "import_report_path": IMPORT_REPORT_PATH,
            "import_report_sha256": IMPORT_REPORT_SHA256,
            "import_report_bytes": IMPORT_REPORT_BYTES,
            "blend_extension_path": BLEND_EXTENSION_PATH,
            "blend_extension_sha256": BLEND_EXTENSION_SHA256,
            "blend_extension_bytes": BLEND_EXTENSION_BYTES,
            "generator_profile": RENDERER_RESULT_PROFILE,
            "renderer_invocation": "exact literal render_silent_video_v100_attempt02 CLI; fixed static face, seed 20261003, native 25 fps",
        },
    }
    parent_execution = parent.get("execution", {})
    qualification = evidence["score_only_receipt"]
    binding_remote = STAGE_PROJECT_REMOTE + "/control/ditto_runtime_binding.json"
    plan_remote = STAGE_PROJECT_REMOTE + "/plan/fit40_ditto240_plan.json"
    rows_remote = STAGE_PROJECT_REMOTE + "/plan/planned_rows.jsonl"
    core_remote = STAGE_PROJECT_REMOTE + "/scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_w2l_block_attempt02.py"
    overlay_remote = STAGE_PROJECT_REMOTE + "/scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_ditto_fit40_block_attempt05.py"
    capacity_basis = {
        "status": "MEASURED_DITTO_V100_ENGINEERING_ROW_SCALED_ESTIMATE_WITH_SLACK",
        "media_bytes_estimated": True,
        "source_rows": 40,
        "methods_per_row": 6,
        "pilot_row_readback_bytes": pilot_bytes,
        "pilot_sample_count": PILOT_SAMPLE_COUNT,
        "explicit_slack_multiplier": SLACK_MULTIPLIER,
        "pilot_measurement": {"actual_repeat_row_trees": copy.deepcopy(pilot_items),
                              "observed_bytes_are_all_regular_file_logical_sizes": True,
                              "max_row_bytes_used": pilot_bytes},
        "measured_estimate_bytes_for_all_240_rows": estimated,
        "required_with_reserve_bytes": total_required,
        "estimate_classification": "measured two-repeat Ditto output-tree bytes scaled by each original PCM N with 3x explicit slack; estimate only, subject to live stage and per-row gates",
        "resource_observation": {"path": str(RESOURCE_OBSERVATION_DEFAULT),
                                 "bytes": RESOURCE_OBSERVATION_BYTES,
                                 "raw_sha256": RESOURCE_OBSERVATION_SHA256,
                                 "self_sha256": "18598d0073bab7681fa8cba5670a531aced118fd1e27e60aaf2034a19c3f8718",
                                 "observed": {"unix_time": resource_observation["observed_unix_time"],
                                             "shm_available_bytes": resource_observation["shm"]["available_bytes"],
                                             "gpu_free_mib": next(g["memory_free_mib"] for g in resource_observation["gpus"] if g["uuid"] == GPU_UUID),
                                             "ram_available_bytes": resource_observation["ram"]["available_bytes"]}},
    }
    cgroup_observed = resource_observation["ram"]["cgroup"]
    resources = {
        "gpu_uuid": GPU_UUID,
        "gpu_name": GPU_NAME,
        "output_mount_target": "/dev/shm",
        "output_reserve_bytes": RESERVE_BYTES,
        "min_gpu_free_mib": MIN_GPU_FREE_MIB,
        "min_effective_ram_available_bytes": MIN_RAM_AVAILABLE_BYTES,
        "min_shm_free_bytes": total_required,
        "cgroup": {"version": cgroup_observed["version"],
                   "identity": {"mount_target": cgroup_observed["identity"]["mount_target"],
                                "mount_root": cgroup_observed["identity"]["mount_root"]},
                   "scope_specific_probe_is_observation_only": True},
        "capacity_basis": capacity_basis,
        "resource_helper": {"path": PROJECT_ROOT_REMOTE + "/" + RESOURCE_HELPER_REL,
                             "bytes": resource_helper_bytes,
                             "sha256": resource_helper_sha256},
        "initial_observation": {"gpu_free_mib": next(g["memory_free_mib"] for g in resource_observation["gpus"] if g["uuid"] == GPU_UUID),
                                "ram_available_bytes": resource_observation["ram"]["available_bytes"],
                                "shm_available_bytes": resource_observation["shm"]["available_bytes"],
                                "output_mount_available_bytes": resource_observation["output_disk"]["available_bytes"],
                                "is_not_a_substitute_for_live_preflight": True},
    }
    binding = build_runtime_binding(
        run_id=RUN_ID,
        scientific_input_pins_sha256=parent["scientific_input_pins_sha256"],
        qualification_evidence=evidence,
        source_and_weights=render_sources,
        capacity_resources=resources,
        output_path_remote=binding_remote,
        overlay_bytes=overlay_bytes,
        overlay_sha256=overlay_sha256,
    )
    binding_raw = json.dumps(binding, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    binding_raw_sha = sha_bytes(binding_raw)

    plan = copy.deepcopy(dict(parent))
    plan["run_id"] = RUN_ID
    plan["rows"] = rows
    plan["models"] = [model]
    plan["execution"] = {
        "stage": "A",
        "capacity_stage": "fit40_ditto240",
        "planned_rows": 240,
        "model_ids": [MODEL_ID],
        "gpu_uuid": GPU_UUID,
        "scientific_input_pins_sha256": parent["scientific_input_pins_sha256"],
        "primary_score": "native_roi_original_pcm16",
        "candidate_selection": "not performed",
        "dev_or_confirm_access": False,
        "scope": "Fit-only Ditto 240-row slice projected from the frozen W2L six-method parent plan; no training, selection, Dev/Confirm access, or full 720-row three-model completion claim",
        "block_policy": {"block_count": 40,
                         "rows_per_block": 6,
                         "block_order": "frozen W2L parent row order, one complete six-method context per block",
                         "block_terminal_status_is_not_full_family_completion": True},
        "output_root": STAGE_ROOT_REMOTE + "/output",
        "runtime_qualification": {"path": qualification["path"], "raw_sha256": qualification["raw_sha256"],
                                  "status": qualification["status"], "selection_eligible": False},
        "stagea_runner": {"path": core_remote, "bytes": stagea_core_bytes, "sha256": stagea_core_sha256},
        "stagea_runner_sha256": stagea_core_sha256,
        "ditto_block_overlay": {"path": overlay_remote, "bytes": overlay_bytes, "sha256": overlay_sha256},
        "ditto_runtime_binding": {"path": binding_remote, "raw_sha256": binding_raw_sha},
        "project_root": PROJECT_ROOT_REMOTE,
        "plan_path": plan_remote,
        "planned_rows_path": rows_remote,
        "parent_w2l_plan": {"path": PARENT_PLAN_REMOTE, "bytes": len(parent_raw), "raw_sha256": sha_bytes(parent_raw),
                             "payload_sha256": parent.get("plan_sha256")},
        "parent_w2l_planned_rows": {"path": PARENT_ROWS_REMOTE, "bytes": len(parent_rows_raw), "raw_sha256": sha_bytes(parent_rows_raw)},
        "parent_w2l_runtime_binding": {"path": str(parent_execution.get("w2l_runtime_binding", {}).get("path")),
                                        "raw_sha256": sha_bytes(parent_binding_raw), "status": parent_binding.get("status")},
        "capacity_basis": {"status": capacity_basis["status"],
                           "estimate_bytes": estimated,
                           "reserve_bytes": RESERVE_BYTES,
                           "explicit_slack_multiplier": SLACK_MULTIPLIER,
                           "estimate_is_not_a_hard_bound": True},
    }
    plan["ditto_fit_projection"] = {
        "schema": PROJECTION_SCHEMA,
        "status": "FROZEN_PARENT_PROJECTED_TO_DITTO_FIT240",
        "parent_run_id": parent["run_id"],
        "parent_plan_path": PARENT_PLAN_REMOTE,
        "parent_plan_raw_sha256": sha_bytes(parent_raw),
        "parent_plan_payload_sha256": parent["plan_sha256"],
        "parent_rows_path": PARENT_ROWS_REMOTE,
        "parent_rows_raw_sha256": sha_bytes(parent_rows_raw),
        "parent_runtime_binding_raw_sha256": sha_bytes(parent_binding_raw),
        "projection_operation": "replace only model/frontend/row ID and native frame binding; retain original 240 ordered Fit rows, all six methods/config hashes, natural PCM/score audio, portrait, sample/speaker/seed/split and protocol",
        "projected_rows": 240,
        "parent_scope": "W2L Fit40 240-row planned slice; backend block execution is independent and no parent rows or actual W2L results are relabeled as Ditto",
    }
    plan["status"] = "FROZEN"
    plan["plan_sha256"] = sha_bytes(canonical({key: value for key, value in plan.items() if key != "plan_sha256"}))
    return binding, plan, rows


def write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        raise


def build(args: argparse.Namespace) -> dict[str, Any]:
    parent, parent_raw = load_json(args.parent_plan, expected_bytes=PARENT_PLAN_BYTES, expected_sha256=PARENT_PLAN_SHA256)
    parent_rows, parent_rows_raw = load_jsonl(args.parent_rows, expected_bytes=PARENT_ROWS_BYTES, expected_sha256=PARENT_ROWS_SHA256)
    parent_binding, parent_binding_raw = load_json(args.parent_binding, expected_bytes=PARENT_BINDING_BYTES, expected_sha256=PARENT_BINDING_SHA256)
    render_receipt, render_raw = load_json(args.render_receipt, expected_bytes=RENDER_RECEIPT_BYTES, expected_sha256=RENDER_RECEIPT_SHA256)
    score_receipt, score_raw = load_json(args.score_receipt, expected_bytes=SCORE_RECEIPT_BYTES, expected_sha256=SCORE_RECEIPT_SHA256)
    repeat_qc, repeat_qc_raw = load_json(args.repeat_qc, expected_bytes=REPEAT_QC_BYTES, expected_sha256=REPEAT_QC_SHA256)
    resource_record, resource_raw = load_json(args.resource_observation, expected_bytes=RESOURCE_OBSERVATION_BYTES,
                                               expected_sha256=RESOURCE_OBSERVATION_SHA256)
    verify_self(repeat_qc, "self_sha256", "root independent Ditto repeat QC")
    if sha_bytes(repeat_qc_raw) != REPEAT_QC_SHA256 or sha_bytes(resource_raw) != RESOURCE_OBSERVATION_SHA256:
        raise ValueError("QC/resource raw input identity changed after the pinned read")
    resource_observation = validate_resource_observation(resource_record)
    evidence = validate_actual_repeat_evidence(render_receipt, score_receipt, repeat_qc, render_raw, score_raw)

    renderer_path = args.renderer
    source_pins, weight_pins = literal_renderer_pins(renderer_path)
    report_source_files = evidence["source_and_weights"].get("source_files", {})
    report_weights = evidence["source_and_weights"].get("weights", {})
    if set(report_source_files) != set(source_pins) or set(report_weights) != set(weight_pins):
        raise ValueError("actual renderer report source/weight roster differs from the renderer literal maps")
    for name, digest in source_pins.items():
        if report_source_files[name].get("sha256") != digest or type(report_source_files[name].get("bytes")) is not int:
            raise ValueError(f"actual renderer report source ref differs from the literal source hash: {name}")
    for name, (size, digest) in weight_pins.items():
        ref = report_weights[name]
        if ref.get("bytes") != size or ref.get("sha256") != digest:
            raise ValueError(f"actual renderer report model asset differs from its literal pin: {name}")

    adapter_bytes = len(read_regular(args.overlay, max_bytes=1_000_000))
    adapter_sha = sha_file(args.overlay)
    if (adapter_bytes != 68_998 or adapter_sha != "ba699de2be6edb4597d5336049798501f7ebf8a4b1118a6337c87f5d09d1c2db"):
        raise ValueError("Ditto adapter source changed from the source-reviewed attempt05")
    core_raw = read_regular(args.stagea_core, max_bytes=2 * 1024 * 1024,
                            expected_bytes=96_537,
                            expected_sha256="6969c6e952d46b6753d279b0e2447aa71c7b8e92967cd42ee57dc8122da5616e")
    helper_raw = read_regular(args.resource_helper, max_bytes=200_000,
                              expected_bytes=62_830,
                              expected_sha256="406f95e4a262fb62bb02bc67249f3e6a7bf211b803a014622e042329d772fdf7")
    pcm16_helper_raw = read_regular(args.pcm16_helper, max_bytes=100_000,
                                    expected_bytes=PCM16_HELPER_BYTES,
                                    expected_sha256=PCM16_HELPER_SHA256)
    evaluator_raw = read_regular(args.syncnet_evaluator, max_bytes=100_000,
                                 expected_bytes=SYNCNET_EVALUATOR_BYTES,
                                 expected_sha256=SYNCNET_EVALUATOR_SHA256)
    adapter_module = load_module(args.overlay, "ditto_fit40_plan_adapter_attempt05", adapter_bytes, adapter_sha)
    actual_renderer_contract_rows = []
    for render_row in render_receipt["rows"]:
        report = render_row["renderer_report_payload"]
        audio = report["audio"]
        sample_count = audio["sample_count"]
        validation_row = {
            "row_id": report["row_id"],
            "seed": report["seed"]["seed"],
            "ditto_native_frame_binding": {
                "sample_count": sample_count,
                "expected_frame_count": adapter_module.expected_native_frame_count(sample_count),
            },
        }
        adapter_module.validate_renderer_input_report(
            validation_row,
            audio_ref={"sample_count": sample_count,
                       "file_sha256": audio["file_sha256"],
                       "pcm_sha256": audio["pcm_float32_sha256"],
                       "bytes": audio["bytes"]},
            report=report,
            expected_source_and_weights=evidence["source_and_weights"],
            expected_face=report["face"],
        )
        actual_renderer_contract_rows.append({
            "row_id": render_row["row_id"],
            "status": "PASS_ACTUAL_RENDER_REPORT_CONTRACT",
            "actual_decoded_frames": report["silent_video"]["probe"]["decoded_frame_count"],
        })
    evidence["actual_stagea_renderer_report_contract"] = {
        "status": "PASS_ACTUAL_ENGINEERING_RENDER_REPORTS_VALIDATED_BY_ADAPTER04",
        "adapter_path": str(args.overlay),
        "adapter_bytes": adapter_bytes,
        "adapter_sha256": adapter_sha,
        "rows": actual_renderer_contract_rows,
        "scope": "two already-produced engineering repeat reports only; no Fit row was executed or scored",
    }
    binding, plan, rows = build_plan(
        parent, parent_raw, parent_rows, parent_rows_raw, parent_binding, parent_binding_raw,
        evidence, resource_observation, adapter_module,
        stagea_core_bytes=len(core_raw), stagea_core_sha256=sha_bytes(core_raw),
        overlay_bytes=adapter_bytes, overlay_sha256=adapter_sha,
        resource_helper_bytes=len(helper_raw), resource_helper_sha256=sha_bytes(helper_raw))
    execution = plan["execution"]
    binding_remote = execution["ditto_runtime_binding"]["path"]
    plan_remote = execution["plan_path"]
    rows_remote = execution["planned_rows_path"]
    core_remote = execution["stagea_runner"]["path"]
    overlay_remote = execution["ditto_block_overlay"]["path"]
    # Confirm the exact adapter-side validators accept the generated in-memory plan/binding lineage.
    parent_remote_path = Path(PARENT_PLAN_REMOTE)
    adapter_module._verify_parent_and_projection(plan, parent, parent_rows,
        parent_plan_raw=parent_raw, parent_rows_raw=parent_rows_raw)
    binding_raw = json.dumps(binding, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    if (binding.get("schema") != RUNTIME_BINDING_SCHEMA
            or binding.get("status") != "PASS"
            or binding.get("binding_sha256") != sha_bytes(canonical({k: v for k, v in binding.items() if k != "binding_sha256"}))
            or plan["execution"]["ditto_runtime_binding"].get("path") != STAGE_PROJECT_REMOTE + "/control/ditto_runtime_binding.json"
            or plan["execution"]["ditto_runtime_binding"].get("raw_sha256") != sha_bytes(binding_raw)
            or binding.get("source_closure", {}).get("ditto_block_overlay") != {
                "path": STAGE_PROJECT_REMOTE + "/scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_ditto_fit40_block_attempt05.py",
                "bytes": adapter_bytes, "sha256": adapter_sha,
                "staged_create_only": True}
            or binding.get("source_closure", {}).get("stagea_core", {}).get("sha256") != sha_bytes(core_raw)
            or binding.get("source_closure", {}).get("pcm16_lineage_helper") != {
                "path": PROJECT_ROOT_REMOTE + "/scripts/experiments/cross_tfg_audio_head_20261003/stagea_pcm16_lineage.py",
                "bytes": len(pcm16_helper_raw), "sha256": sha_bytes(pcm16_helper_raw)}
            or binding.get("source_closure", {}).get("syncnet_evaluator") != {
                "path": PROJECT_ROOT_REMOTE + "/" + SYNCNET_EVALUATOR_REL,
                "bytes": len(evaluator_raw), "sha256": sha_bytes(evaluator_raw)}):
        raise ValueError("generated metadata does not close over the actual renderer receipt and exact Stage-A sources")
    if plan["rows"] != rows or rows != adapter_module.project_fit_rows(parent_rows, parent["methods"]):
        raise ValueError("generated external row projection failed exact adapter round-trip")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    binding_path = output_dir / "ditto_runtime_binding_attempt04.json"
    plan_path = output_dir / "fit40_ditto240_plan_attempt04.json"
    rows_path = output_dir / "planned_rows_attempt04.jsonl"
    evidence_path = output_dir / "qualification_evidence_attempt04.json"
    evidence_raw = json.dumps(evidence, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    rows_raw = b"".join(canonical(row) + b"\n" for row in rows)
    plan_raw = json.dumps(plan, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    # Recompute raw binding/plan references from the exact pretty-printed output bytes.
    # Binding JSON raw hash is a plan reference; binding's own self hash is canonical.
    binding_output_raw = binding_raw
    expected_binding_hash = sha_bytes(binding_output_raw)
    if plan["execution"]["ditto_runtime_binding"]["raw_sha256"] != expected_binding_hash:
        raise ValueError("plan binding raw SHA does not bind the serialized runtime binding")
    write_exclusive(binding_path, binding_output_raw)
    write_exclusive(plan_path, plan_raw)
    write_exclusive(rows_path, rows_raw)
    write_exclusive(evidence_path, evidence_raw)
    outputs = {}
    for name, path in (("binding", binding_path), ("plan", plan_path), ("planned_rows", rows_path), ("qualification_evidence", evidence_path)):
        raw = path.read_bytes()
        outputs[name] = {"path": str(path), "bytes": len(raw), "raw_sha256": sha_bytes(raw)}
    receipt: dict[str, Any] = {
        "schema": "v100_ditto_fit40_plan_build_receipt_v1",
        "status": "PASS_METADATA_ONLY_FROZEN_PARENT_PROJECTED_NO_EXECUTION",
        "run_id": RUN_ID,
        "selection_eligible": False,
        "scientific_claim": False,
        "training_or_api": False,
        "execution_started": False,
        "parent_plan": {"path": str(args.parent_plan), "bytes": len(parent_raw), "raw_sha256": sha_bytes(parent_raw),
                        "payload_sha256": parent["plan_sha256"]},
        "parent_rows": {"path": str(args.parent_rows), "bytes": len(parent_rows_raw), "raw_sha256": sha_bytes(parent_rows_raw), "row_count": len(parent_rows)},
        "parent_binding": {"path": str(args.parent_binding), "bytes": len(parent_binding_raw), "raw_sha256": sha_bytes(parent_binding_raw), "binding_sha256": parent_binding["binding_sha256"]},
        "Ditto_engineering_qualification": evidence,
        "resource_snapshot": {"path": str(args.resource_observation), "bytes": len(resource_raw), "raw_sha256": sha_bytes(resource_raw),
                              "self_sha256": resource_record["self_sha256"],
                              "estimated_240_row_bytes": plan["execution"]["capacity_basis"]["estimate_bytes"],
                              "reserve_bytes": RESERVE_BYTES,
                              "measured_shm_free_bytes": resource_observation["shm"]["available_bytes"]},
        "source_pins": {
            "adapter": {"path": str(args.overlay), "bytes": adapter_bytes, "sha256": adapter_sha},
            "stagea_core": {"path": str(args.stagea_core), "bytes": len(core_raw), "sha256": sha_bytes(core_raw)},
            "resource_helper": {"path": str(args.resource_helper), "bytes": len(helper_raw), "sha256": sha_bytes(helper_raw)},
            "pcm16_lineage_helper": {"path": str(args.pcm16_helper), "bytes": len(pcm16_helper_raw), "sha256": sha_bytes(pcm16_helper_raw)},
            "syncnet_evaluator": {"path": str(args.syncnet_evaluator), "bytes": len(evaluator_raw), "sha256": sha_bytes(evaluator_raw)},
            "renderer": {"path": str(args.renderer), "bytes": 40_010,
                         "sha256": "7b37381777ad8cabf3665a7ab636295bc779b4a1fde5fa15dd485360772a53e4"},
        },
        "remote_stage_targets": {
            "stage_root": STAGE_ROOT_REMOTE,
            "project_root": STAGE_PROJECT_REMOTE,
            "output_root": STAGE_ROOT_REMOTE + "/output",
            "parent_plan": PARENT_PLAN_REMOTE,
            "parent_rows": PARENT_ROWS_REMOTE,
            "overlay": plan["execution"]["ditto_block_overlay"]["path"],
            "stagea_core": plan["execution"]["stagea_runner"]["path"],
            "plan": plan_remote,
            "planned_rows": rows_remote,
            "binding": binding_remote,
            "resource_helper_create_only_target": PROJECT_ROOT_REMOTE + "/" + RESOURCE_HELPER_REL,
        },
        "outputs": outputs,
    }
    receipt["self_sha256"] = sha_bytes(canonical(receipt))
    receipt_path = output_dir / "build_receipt_attempt04.json"
    write_exclusive(receipt_path, json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, default=PARENT_PLAN_DEFAULT)
    parser.add_argument("--parent-rows", type=Path, default=PARENT_ROWS_DEFAULT)
    parser.add_argument("--parent-binding", type=Path, default=PARENT_BINDING_DEFAULT)
    parser.add_argument("--render-receipt", type=Path, default=RENDER_RECEIPT_DEFAULT)
    parser.add_argument("--score-receipt", type=Path, default=SCORE_RECEIPT_DEFAULT)
    parser.add_argument("--repeat-qc", type=Path, default=REPEAT_QC_DEFAULT)
    parser.add_argument("--resource-observation", type=Path, default=RESOURCE_OBSERVATION_DEFAULT)
    parser.add_argument("--renderer", type=Path, default=RENDERER_DEFAULT)
    parser.add_argument("--stagea-core", type=Path, default=STAGEA_CORE_DEFAULT)
    parser.add_argument("--overlay", type=Path, default=OVERLAY_DEFAULT)
    parser.add_argument("--resource-helper", type=Path, default=RESOURCE_HELPER_DEFAULT)
    parser.add_argument("--pcm16-helper", type=Path, default=PCM16_HELPER_DEFAULT)
    parser.add_argument("--syncnet-evaluator", type=Path, default=SYNCNET_EVALUATOR_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=SCRATCH / "build_output_attempt04")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    receipt = build(build_parser().parse_args(argv))
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
