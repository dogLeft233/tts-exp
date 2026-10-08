#!/usr/bin/env python3
"""Strict-bootstrap CPU preflight for the frozen V3 candidate inputs.

This worker runs one old same-profile anchor and each distinct natural Dev
input exactly once in a single Python process. It does not render or score
video, access Confirm data, or select a candidate. Every generated artifact is
new and create-only under the task's SHM output root.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import pathlib
import shutil
import site
import stat
import sys
import time
from typing import Any


SCHEMA = "cross_tfg_v3_candidate_cpu_preflight_v1"
PREFLIGHT_ATTEMPT = "candidate_v3_cpu_preflight_attempt03"
RESULT_SCHEMA = "cross_tfg_v3_enhancement_cli_result_v1"
PLAN_RAW_SHA256 = "8615754443e653fc79203f108fa689205a77e7f6f234930b352204df08b98280"
PLAN_BYTES = 932236
PLAN_SELF_SHA256 = "fee2718c0c4b18e5acabf3ca7d7103344bfb3859f3d8a3b93862c1ad4ce7d612"
PLAN_MANIFEST_RAW_SHA256 = "d25985b666c59946a7acd1c948b834ea8628d0b9c8eef0442419818ea86c0192"
PLAN_MANIFEST_BYTES = 2513
PLAN_MANIFEST_SELF_SHA256 = "420d73cd5fb95692ded1c8bbd8cd82363ccb40169a064f21121be70f041909e5"
CONFIG03_RAW_SHA256 = "fd2ce3509c457e40e618818828f6040ab65641315d40efe8f48e5dc3c5c3d1df"
CONFIG03_BYTES = 5240
CONFIG02_RAW_SHA256 = "aba0dc744b76cfd9f73db547faa80508a4a46437432ef78feaf5b7bfb51f3666"
CONFIG02_BYTES = 5240
CLI03_RAW_SHA256 = "2cf5fc0f73d7c8d5aabbbc457672e487de2f8a88e683976bb58698325a80fdfd"
CLI03_BYTES = 43071
BOOTSTRAP_SHA256 = "3099c4b2e1c650f7aa46c31145cc2c88ccda9f085288ab039c6ffed85007ab65"
BOOTSTRAP_BYTES = 9525
OLD_CONFIG02_CLI_SHA256 = "7a560332ce86a82ffede4254b015cabf5f10176e6f601bfc172a331f7bb885e6"
OLD_CONFIG02_CLI_BYTES = 42653
EXPECTED_PYTHON = (3, 10, 12)
EXPECTED_TORCH = "2.5.1+cu124"
EXPECTED_NUMPY = "1.26.4"
EXPECTED_INTRA_THREADS = 1
EXPECTED_INTER_THREADS = 1
MIN_FREE_BYTES = 5 * 1024**3
MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024
EXPECTED_DEV_IDS = (
    "BAC009S0005W0121", "BAC009S0005W0123", "BAC009S0022W0122", "BAC009S0022W0123",
    "BAC009S0024W0121", "BAC009S0024W0122", "BAC009S0037W0121", "BAC009S0037W0122",
    "BAC009S0038W0121", "BAC009S0038W0122", "BAC009S0039W0121", "BAC009S0039W0123",
    "BAC009S0041W0121", "BAC009S0041W0122", "BAC009S0044W0121", "BAC009S0044W0122",
    "BAC009S0049W0121", "BAC009S0049W0122", "BAC009S0050W0121", "BAC009S0050W0122",
)
EXPECTED_MODELS = {
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
}
EXPECTED_METHOD = "shared_recipient_transition_head_v3_fit40_step120"
EXPECTED_SEEDS = (20261003, 20261004, 20261005)
EXPECTED_CHECKPOINT = "8d5eb607f77963291372b33245f13f9b9578ebf71657acf1b43243746c538eb8"
EXPECTED_TRAINING = "c2ed183ae5894bd3c42dc56c0242539f1f8290df1230d7b1aa4df8b7584a7e67"
EXPECTED_REFERENCE = {
    "sample_id": "BAC009S0052W0121",
    "row_id": "portable_fit_cli_smoke_attempt01_BAC009S0052W0121",
    "input_path": "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/v5_fit40_input_stage_attempt03/fit40_source_inputs/source_wav/BAC009S0052W0121.wav",
    "input_bytes": 125322,
    "input_file_sha256": "2d45f2b65dec29f77bcd56f05f883b16207a99e7848df3a06a203819a915b261",
    "input_pcm_sha256": "f4e611e9f816665ad7c4dce4980f461c4d9af4fe6fe9678c8f756bd03fa6b36c",
    "sample_count": 62639,
    "sample_rate_hz": 16000,
    "channels": 1,
    "old_output_bytes": 250600,
    "old_output_file_sha256": "9cf46db4c5caf7f058bda52b290ab5a79405fddf97ad541bfd9dc72624b6bfa0",
    "old_output_pcm_sha256": "ef72ce37b21878436a7f38888bff04a71446d71a4d2e7ebf85bb84fec6cd2f6c",
    "old_metadata_bytes": 12233,
    "old_metadata_sha256": "2aab5bfecf4a6d5ce6e916cc0fe1b3e453b216afc238e508904704a51999db76",
    "old_cli_sha256": OLD_CONFIG02_CLI_SHA256,
    "old_config_sha256": CONFIG02_RAW_SHA256,
}


class PreflightError(RuntimeError):
    pass


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def file_pin(path: pathlib.Path, *, expected_bytes: int | None = None, expected_sha: str | None = None, maximum: int = MAX_JSON_BYTES) -> tuple[bytes, dict[str, Any]]:
    if path.is_symlink():
        raise PreflightError(f"symlink is not accepted: {path}")
    try:
        before = path.lstat()
    except OSError as exc:
        raise PreflightError(f"cannot stat {path}: {exc}") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
        raise PreflightError(f"not a bounded regular file: {path}")
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise PreflightError(f"file changed while reading: {path}")
    digest = sha256(raw)
    if expected_bytes is not None and len(raw) != expected_bytes:
        raise PreflightError(f"byte count mismatch for {path}: {len(raw)}")
    if expected_sha is not None and digest != expected_sha:
        raise PreflightError(f"SHA-256 mismatch for {path}: {digest}")
    return raw, {"path": str(path), "bytes": len(raw), "sha256": digest}


def read_json(raw: bytes, label: str, *, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    if len(raw) > max_bytes:
        raise PreflightError(f"{label} exceeds JSON byte bound")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"{label} is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{label} must be an object")
    return value


def without_key_sha(obj: dict[str, Any], key: str) -> str:
    payload = dict(obj)
    expected = payload.pop(key, None)
    computed = sha256(canonical(payload))
    if expected != computed:
        raise PreflightError(f"{key} self-hash mismatch")
    return computed


def no_symlink_components(path: pathlib.Path) -> None:
    absolute = path.absolute()
    cursor = pathlib.Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor = cursor / component
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise PreflightError(f"symlink path component not accepted: {cursor}")


def create_exclusive(path: pathlib.Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    actual, pin = file_pin(path, expected_bytes=len(raw), expected_sha=sha256(raw), maximum=max(len(raw), 1))
    if actual != raw:
        raise PreflightError(f"create-only write verification failed: {path}")


def load_plan(plan_path: pathlib.Path, manifest_path: pathlib.Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    plan_raw, plan_pin = file_pin(plan_path, expected_bytes=PLAN_BYTES, expected_sha=PLAN_RAW_SHA256, maximum=2 * 1024 * 1024)
    plan = read_json(plan_raw, "candidate plan", max_bytes=2 * 1024 * 1024)
    without_key_sha(plan, "plan_sha256")
    if plan.get("plan_sha256") != PLAN_SELF_SHA256 or plan.get("status") != "FROZEN":
        raise PreflightError("candidate predecessor plan frozen identity differs")
    manifest_raw, manifest_pin = file_pin(manifest_path, expected_bytes=PLAN_MANIFEST_BYTES, expected_sha=PLAN_MANIFEST_RAW_SHA256)
    manifest = read_json(manifest_raw, "candidate plan manifest")
    without_key_sha(manifest, "manifest_sha256")
    if manifest.get("manifest_sha256") != PLAN_MANIFEST_SELF_SHA256 or manifest.get("status") != "FROZEN":
        raise PreflightError("candidate predecessor plan manifest frozen identity differs")
    if manifest.get("plan_raw_sha256") != PLAN_RAW_SHA256 or manifest.get("plan_sha256") != PLAN_SELF_SHA256:
        raise PreflightError("manifest does not bind exact predecessor plan raw/self hash")
    if manifest.get("candidate_config_raw_sha256") != CONFIG02_RAW_SHA256:
        raise PreflightError("predecessor plan manifest has unexpected CLI02 config pin")
    if manifest.get("planned_rows") != 180 or manifest.get("sample_count") != 20 or manifest.get("speaker_count") != 10:
        raise PreflightError("frozen candidate manifest scope differs from 180/20/10")
    if manifest.get("confirm_audio_read") is not False or manifest.get("confirm_effects_read") is not False:
        raise PreflightError("candidate predecessor manifest does not preserve Confirm isolation")
    if manifest.get("candidate_runtime_qualification_performed") is not False:
        raise PreflightError("predecessor plan was already labeled runtime qualified")
    rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) != 180:
        raise PreflightError("frozen candidate plan must contain exactly 180 rows")
    by_sample: dict[str, dict[str, Any]] = {}
    native_binding_by_sample: dict[str, dict[str, Any]] = {}
    frame_count_by_sample: dict[str, int] = {}
    row_ids: set[str] = set()
    row_cells: set[tuple[str, str, int, str]] = set()
    model_ids: set[str] = set()
    method_ids: set[str] = set()
    seeds: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise PreflightError("candidate plan contains a non-object row")
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or row_id in row_ids:
            raise PreflightError("candidate row IDs are missing or duplicated")
        row_ids.add(row_id)
        sample_id = row.get("sample_id")
        model_id = row.get("model_id")
        method_id = row.get("method_id")
        seed = row.get("seed")
        if row.get("split") != "dev" or row.get("model_role") != "candidate":
            raise PreflightError(f"row {row_id} is outside frozen dev candidate scope")
        if not isinstance(sample_id, str) or model_id not in EXPECTED_MODELS or method_id != EXPECTED_METHOD:
            raise PreflightError(f"row {row_id} has unexpected sample/model/method")
        if type(seed) is not int:
            raise PreflightError(f"row {row_id} seed is not an integer")
        cell = (sample_id, model_id, seed, method_id)
        if cell in row_cells:
            raise PreflightError(f"candidate plan duplicates matrix cell {cell}")
        row_cells.add(cell)
        model_ids.add(model_id)
        method_ids.add(method_id)
        seeds.add(seed)
        natural = row.get("natural_pcm")
        score = row.get("score_audio")
        if not isinstance(natural, dict) or not isinstance(score, dict) or natural != score:
            raise PreflightError(f"row {row_id} natural/score audio binding differs")
        if natural.get("sample_rate_hz") != 16000 or natural.get("channels") != 1:
            raise PreflightError(f"row {row_id} natural PCM is not mono 16 kHz")
        if natural.get("subtype") != "IEEE_FLOAT_32_LE":
            raise PreflightError(f"row {row_id} natural PCM subtype differs")
        if row.get("checkpoint_sha256") != EXPECTED_CHECKPOINT:
            raise PreflightError(f"row {row_id} checkpoint pin differs")
        gen = row.get("generation_settings")
        if not isinstance(gen, dict) or gen.get("v3_cli_config_sha256") != CONFIG02_RAW_SHA256:
            raise PreflightError(f"row {row_id} does not match frozen predecessor CLI02 config")
        if gen.get("training_record_sha256") != EXPECTED_TRAINING or gen.get("seed") != seed or gen.get("generator") != model_id:
            raise PreflightError(f"row {row_id} training record pin differs")
        binding = row.get("wav2lip_native_frame_binding")
        if model_id == "wav2lip_gan_bac9a81e":
            if not isinstance(binding, dict):
                raise PreflightError(f"Wav2Lip row {row_id} lacks native frame binding")
            if binding.get("sample_id") != sample_id or binding.get("sample_count") != natural.get("sample_count"):
                raise PreflightError(f"Wav2Lip row {row_id} frame binding sample count differs")
            if binding.get("natural_pcm_sha256") != natural.get("pcm_sha256"):
                raise PreflightError(f"Wav2Lip row {row_id} frame binding PCM identity differs")
            if binding.get("fps_num") != 25 or binding.get("fps_den") != 1:
                raise PreflightError(f"Wav2Lip row {row_id} native frame binding clock differs")
            frame_count = binding.get("expected_inference_frame_count")
            if type(frame_count) is not int or frame_count < 1:
                raise PreflightError(f"Wav2Lip row {row_id} native frame count is invalid")
            if sample_id in native_binding_by_sample and native_binding_by_sample[sample_id] != binding:
                raise PreflightError(f"Wav2Lip rows for {sample_id} disagree on the pinned native frame binding")
            if sample_id in frame_count_by_sample and frame_count_by_sample[sample_id] != frame_count:
                raise PreflightError(f"Wav2Lip rows for {sample_id} disagree on native emitted-frame count")
            native_binding_by_sample[sample_id] = binding
            frame_count_by_sample[sample_id] = frame_count
        if sample_id in by_sample and by_sample[sample_id] != natural:
            raise PreflightError(f"rows for {sample_id} disagree on natural PCM identity")
        by_sample[sample_id] = natural
    if len(row_ids) != 180 or model_ids != EXPECTED_MODELS or method_ids != {EXPECTED_METHOD}:
        raise PreflightError("candidate plan row/model/method scope is incomplete")
    if len(by_sample) != 20 or set(by_sample) != set(EXPECTED_DEV_IDS) or tuple(sorted(seeds)) != EXPECTED_SEEDS:
        raise PreflightError("candidate plan does not match exact frozen 20-sample/3-seed cohort")
    if set(native_binding_by_sample) != set(EXPECTED_DEV_IDS):
        raise PreflightError("candidate plan lacks one complete Wav2Lip native frame binding per Dev sample")
    expected_cells = {
        (sample_id, model_id, seed, EXPECTED_METHOD)
        for sample_id in EXPECTED_DEV_IDS
        for model_id in EXPECTED_MODELS
        for seed in seeds
    }
    if row_cells != expected_cells:
        raise PreflightError("candidate plan is not the exact 20-sample × 3-model × 3-seed matrix")
    if manifest.get("render_seed_family") != sorted(seeds):
        raise PreflightError("candidate plan manifest seed family differs from plan rows")
    return plan, manifest, rows, by_sample, native_binding_by_sample


def stage_candidate_provenance(config: dict[str, Any], repo_root: pathlib.Path, code_root: pathlib.Path) -> list[dict[str, Any]]:
    provenance = config.get("candidate_provenance")
    if not isinstance(provenance, dict) or set(provenance) != {"checkpoint", "training_record", "selection_record"}:
        raise PreflightError("CLI03 candidate provenance set differs")
    records: list[dict[str, Any]] = []
    for name in ("checkpoint", "training_record", "selection_record"):
        pin = provenance[name]
        if not isinstance(pin, dict) or set(pin) != {"path", "bytes", "sha256"}:
            raise PreflightError(f"candidate provenance pin {name} is malformed")
        rel = pathlib.PurePosixPath(pin["path"])
        if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
            raise PreflightError(f"candidate provenance path is unsafe: {name}")
        src = repo_root.joinpath(*rel.parts)
        dst = code_root.joinpath(*rel.parts)
        no_symlink_components(src)
        raw, source_pin = file_pin(src, expected_bytes=pin["bytes"], expected_sha=pin["sha256"], maximum=1 * 1024 * 1024)
        if dst.exists() or dst.is_symlink():
            raise PreflightError(f"isolated provenance target already exists: {dst}")
        create_exclusive(dst, raw)
        copied, copied_pin = file_pin(dst, expected_bytes=pin["bytes"], expected_sha=pin["sha256"], maximum=1 * 1024 * 1024)
        if copied != raw:
            raise PreflightError(f"isolated provenance copy changed: {name}")
        records.append({"name": name, "source": source_pin, "isolated_copy": copied_pin})
    return records


def load_reference_inputs(reference_wav_path: pathlib.Path, reference_metadata_path: pathlib.Path) -> tuple[dict[str, Any], bytes, bytes]:
    old_wav, old_wav_pin = file_pin(reference_wav_path, expected_bytes=EXPECTED_REFERENCE["old_output_bytes"], expected_sha=EXPECTED_REFERENCE["old_output_file_sha256"], maximum=1 * 1024 * 1024)
    old_meta, old_meta_pin = file_pin(reference_metadata_path, expected_bytes=EXPECTED_REFERENCE["old_metadata_bytes"], expected_sha=EXPECTED_REFERENCE["old_metadata_sha256"], maximum=256 * 1024)
    if len(old_wav) != EXPECTED_REFERENCE["old_output_bytes"] or sha256(old_wav) != EXPECTED_REFERENCE["old_output_file_sha256"]:
        raise PreflightError("old trained candidate reference WAV differs from independent readback pin")
    if len(old_meta) != EXPECTED_REFERENCE["old_metadata_bytes"] or sha256(old_meta) != EXPECTED_REFERENCE["old_metadata_sha256"]:
        raise PreflightError("old trained candidate reference metadata differs from independent readback pin")
    old_meta_obj = read_json(old_meta, "old trained candidate metadata")
    if old_meta_obj.get("schema") != RESULT_SCHEMA or old_meta_obj.get("status") != "PASS":
        raise PreflightError("old same-profile candidate reference metadata did not pass")
    if old_meta_obj.get("profile_role") != "METHOD_CANDIDATE":
        raise PreflightError("old reference profile is not the trained candidate profile")
    if old_meta_obj.get("config", {}).get("sha256") != CONFIG02_RAW_SHA256:
        raise PreflightError("old reference was not generated under CLI02 candidate config")
    if old_meta_obj.get("source_audio", {}).get("path") != EXPECTED_REFERENCE["input_path"]:
        raise PreflightError("old reference metadata source path differs")
    if old_meta_obj.get("source_audio", {}).get("file_sha256") != EXPECTED_REFERENCE["input_file_sha256"]:
        raise PreflightError("old reference metadata source file pin differs")
    if old_meta_obj.get("source_audio", {}).get("sample_count") != EXPECTED_REFERENCE["sample_count"] or old_meta_obj.get("source_audio", {}).get("sample_rate_hz") != 16000:
        raise PreflightError("old reference metadata source clock differs")
    canonical_input = old_meta_obj.get("canonical_input_pcm", {})
    if canonical_input.get("pcm_sha256") != EXPECTED_REFERENCE["input_pcm_sha256"] or canonical_input.get("sample_count") != EXPECTED_REFERENCE["sample_count"] or canonical_input.get("sample_rate_hz") != 16000 or canonical_input.get("channels") != 1:
        raise PreflightError("old reference metadata canonical source PCM identity differs")
    if old_meta_obj.get("enhanced_output", {}).get("pcm_float32_sha256") != EXPECTED_REFERENCE["old_output_pcm_sha256"]:
        raise PreflightError("old reference metadata output PCM pin differs")
    old_output = old_meta_obj.get("enhanced_output", {})
    if old_output.get("sample_count") != EXPECTED_REFERENCE["sample_count"] or old_output.get("sample_rate_hz") != 16000 or old_output.get("channels") != 1 or old_output.get("encoding") != "IEEE_FLOAT_32_LE":
        raise PreflightError("old reference metadata output clock/format differs")
    if old_meta_obj.get("source_pins", {}).get("cli", {}).get("sha256") != OLD_CONFIG02_CLI_SHA256:
        raise PreflightError("old candidate reference CLI source pin differs")
    old_parameter = old_meta_obj.get("head", {}).get("parameter")
    if not isinstance(old_parameter, dict) or not isinstance(old_parameter.get("values"), list):
        raise PreflightError("old candidate reference parameter shape is malformed")
    return {"output_wav": old_wav_pin, "metadata": old_meta_pin, "metadata_object": old_meta_obj}, old_wav, old_meta


def decode_float_wav_bytes(audio_io: Any, raw: bytes) -> tuple[Any, dict[str, Any]]:
    fmt, channels, rate, bits, block_align, payload = audio_io._parse_wave(raw)
    if (fmt, channels, rate, bits, block_align) != (3, 1, 16000, 32, 4):
        raise PreflightError("archived trained reference is not mono 16 kHz IEEE float32 WAV")
    decoded = audio_io._decode_wave_samples(fmt, channels, bits, block_align, payload)
    import numpy as np

    samples = np.ascontiguousarray(decoded[:, 0] if decoded.ndim == 2 else decoded, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise PreflightError("archived trained reference PCM is non-finite or malformed")
    return samples, {"sample_count": int(samples.size), "sample_rate_hz": rate, "channels": channels, "pcm_float32_sha256": audio_io.pcm_float32_sha256(samples)}


def observed_policy(bootstrap: Any, torch: Any, np: Any) -> dict[str, Any]:
    expected = bootstrap.POLICY
    obs = {
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "torch_use_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "torch_cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "torch_cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch_allow_tf32": bool(torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32),
    }
    if obs != expected:
        raise PreflightError("actual strict startup policy differs from pinned sitecustomize POLICY")
    if torch.cuda.is_initialized():
        raise PreflightError("CPU-only preflight unexpectedly initialized CUDA")
    if tuple(sys.version_info[:3]) != EXPECTED_PYTHON:
        raise PreflightError("actual Python version differs from the frozen V3 environment")
    if str(torch.__version__) != EXPECTED_TORCH or str(np.__version__) != EXPECTED_NUMPY:
        raise PreflightError("actual torch/NumPy versions differ from the frozen V3 environment")
    return obs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=pathlib.Path)
    parser.add_argument("--code-root", required=True, type=pathlib.Path)
    parser.add_argument("--plan", required=True, type=pathlib.Path)
    parser.add_argument("--plan-manifest", required=True, type=pathlib.Path)
    parser.add_argument("--config", required=True, type=pathlib.Path)
    parser.add_argument("--predecessor-config", required=True, type=pathlib.Path)
    parser.add_argument("--reference-wav", required=True, type=pathlib.Path)
    parser.add_argument("--reference-metadata", required=True, type=pathlib.Path)
    parser.add_argument("--output-root", required=True, type=pathlib.Path)
    parser.add_argument("--receipt", required=True, type=pathlib.Path)
    args = parser.parse_args()

    started = time.time()
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "preflight_attempt": PREFLIGHT_ATTEMPT,
        "status": "FAILED_CPU_PREFLIGHT",
        "scope": {
            "candidate_full180_started": False,
            "candidate_audio_rows_expected": 20,
            "candidate_audio_rows_completed": 0,
            "same_profile_reference_rows_expected": 1,
            "same_profile_reference_rows_completed": 0,
            "gpu_used": False,
            "cuda_initialized": None,
            "video_rendered": False,
            "scorer_called": False,
            "confirm_audio_read": False,
            "confirm_effects_read": False,
            "source_audio_modified": False,
            "candidate_selected": False,
        },
        "inputs": {},
        "staged_provenance": [],
        "runtime": {},
        "reference_comparison": {},
        "candidate_rows": [],
        "failure": None,
        "started_unix": started,
    }
    try:
        if os.environ.get("PYTHONNOUSERSITE") != "1" or site.ENABLE_USER_SITE is not False:
            raise PreflightError("PYTHONNOUSERSITE/user-site isolation is not active")
        if os.environ.get("CROSS_TFG_STRICT_STARTUP_POLICY") != "1":
            raise PreflightError("strict startup policy mode is absent")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise PreflightError("CUDA_VISIBLE_DEVICES is not empty")
        if any(os.environ.get(k) is not None for k in ("CROSS_TFG_ROW_ID", "CROSS_TFG_MODEL_ID", "CROSS_TFG_AUDIO_FRONTEND")):
            raise PreflightError("row-only renderer bindings leaked into the preflight process")
        if str(sys.executable) != "/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python":
            raise PreflightError(f"lexical Python invocation differs: {sys.executable}")
        if not args.output_root.is_dir() or args.output_root.is_symlink():
            raise PreflightError("fresh output root is absent or symlinked")
        if any(args.output_root.iterdir()):
            raise PreflightError("fresh CPU preflight output root is not empty")
        free_before = shutil.disk_usage(args.output_root).free
        if free_before < MIN_FREE_BYTES:
            raise PreflightError(f"SHM free space {free_before} is below 5 GiB reserve")

        plan, manifest, rows, inputs_by_sample, native_binding_by_sample = load_plan(args.plan, args.plan_manifest)
        config_raw, config_pin = file_pin(args.config, expected_bytes=CONFIG03_BYTES, expected_sha=CONFIG03_RAW_SHA256)
        config = read_json(config_raw, "CLI03 config", max_bytes=256 * 1024)
        old_config_raw, old_config_pin = file_pin(args.predecessor_config, expected_bytes=CONFIG02_BYTES, expected_sha=CONFIG02_RAW_SHA256)
        old_config = read_json(old_config_raw, "CLI02 candidate config", max_bytes=256 * 1024)
        old_projected = json.loads(json.dumps(old_config))
        new_projected = json.loads(json.dumps(config))
        old_projected["source_pins"]["cli"] = new_projected["source_pins"]["cli"]
        if old_projected != new_projected:
            raise PreflightError("CLI03 config differs from CLI02 config beyond the single CLI source pin")
        if config.get("profile_role") != "METHOD_CANDIDATE" or config.get("runtime", {}).get("device") != "cpu":
            raise PreflightError("CLI03 config is not the frozen CPU METHOD_CANDIDATE profile")
        if config.get("candidate_provenance", {}).get("checkpoint", {}).get("sha256") != EXPECTED_CHECKPOINT:
            raise PreflightError("candidate checkpoint SHA differs from the frozen plan")
        if config.get("candidate_provenance", {}).get("training_record", {}).get("sha256") != EXPECTED_TRAINING:
            raise PreflightError("candidate training record SHA differs from the frozen plan")
        if config.get("source_pins", {}).get("cli") != {"path": "scripts/audio_enhancement/enhance_v3_attempt03.py", "bytes": CLI03_BYTES, "sha256": CLI03_RAW_SHA256}:
            raise PreflightError("CLI03 config does not pin the exact attempt03 entrypoint")
        for name, pin in config["source_pins"].items():
            rel = pathlib.PurePosixPath(pin["path"])
            if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
                raise PreflightError(f"unsafe CLI source pin path: {name}")
            staged_path = args.code_root.joinpath(*rel.parts)
            file_pin(staged_path, expected_bytes=pin["bytes"], expected_sha=pin["sha256"], maximum=256 * 1024)
        provenance_copies = stage_candidate_provenance(config, args.repo_root, args.code_root)

        bootstrap_module = sys.modules.get("sitecustomize")
        if bootstrap_module is None:
            raise PreflightError("sitecustomize was not imported during Python startup")
        bootstrap_path = pathlib.Path(getattr(bootstrap_module, "__file__", "")).resolve(strict=True)
        bootstrap_raw, bootstrap_pin = file_pin(bootstrap_path, expected_bytes=BOOTSTRAP_BYTES, expected_sha=BOOTSTRAP_SHA256, maximum=64 * 1024)
        if tuple(pathlib.Path(p).resolve() for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)[:1] != (bootstrap_path.parent,):
            raise PreflightError("strict bootstrap directory is not first on PYTHONPATH")
        if not isinstance(getattr(bootstrap_module, "POLICY", None), dict):
            raise PreflightError("pinned sitecustomize has no POLICY contract")

        sys.dont_write_bytecode = True
        sys.path.insert(0, str(args.code_root.resolve(strict=True)))
        cli_path = args.code_root / "scripts/audio_enhancement/enhance_v3_attempt03.py"
        spec = importlib.util.spec_from_file_location("cross_tfg_preflight_cli03", cli_path)
        if spec is None or spec.loader is None:
            raise PreflightError("cannot load pinned CLI03 source")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        if cli.__file__ is None or pathlib.Path(cli.__file__).resolve(strict=True) != cli_path.resolve(strict=True):
            raise PreflightError("imported CLI module origin differs from staged attempt03 path")
        audio_io = importlib.import_module("scripts.audio_enhancement.audio_io")
        np = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        initial_policy = observed_policy(bootstrap_module, torch, np)
        runtime = {
            "python_argv0": sys.argv[0],
            "sys_executable": sys.executable,
            "sys_executable_resolved": str(pathlib.Path(sys.executable).resolve(strict=True)),
            "python_version": list(sys.version_info[:3]),
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
            "user_site_enabled": site.ENABLE_USER_SITE,
            "numpy_version": str(np.__version__),
            "torch_version": str(torch.__version__),
            "torch_cuda_initialized_before_rows": bool(torch.cuda.is_initialized()),
            "device": config["runtime"]["device"],
            "bootstrap": bootstrap_pin,
            "bootstrap_policy": initial_policy,
            "cli_origin": str(pathlib.Path(cli.__file__).resolve(strict=True)),
            "strict_process_count": 1,
            "rows_reused_one_torch_process": True,
        }
        if runtime["sys_executable_resolved"] != "/usr/bin/python3.10":
            raise PreflightError("resolved Python executable differs from the qualified runtime")
        if not torch.cuda.is_initialized():
            receipt["scope"]["cuda_initialized"] = False
        else:
            raise PreflightError("CUDA unexpectedly initialized before first row")

        old_reference, old_wav_bytes, old_metadata_bytes = load_reference_inputs(args.reference_wav, args.reference_metadata)
        old_metadata = old_reference["metadata_object"]
        if old_metadata.get("source_pins") != old_config.get("source_pins"):
            raise PreflightError("same-profile anchor source pins differ from predecessor CLI02 config")
        if old_metadata.get("head", {}).get("parameter") != config.get("parameter"):
            raise PreflightError("same-profile anchor parameter object differs from CLI03 config")
        old_candidate_provenance = old_metadata.get("head", {}).get("candidate_provenance", {})
        if old_candidate_provenance.get("checkpoint_sha256") != EXPECTED_CHECKPOINT or old_candidate_provenance.get("training_record_sha256") != EXPECTED_TRAINING:
            raise PreflightError("same-profile anchor provenance differs from frozen trained candidate")
        old_pins = old_candidate_provenance.get("pins", {})
        for name in ("checkpoint", "training_record", "selection_record"):
            if old_pins.get(name) != config.get("candidate_provenance", {}).get(name):
                raise PreflightError(f"same-profile anchor provenance pin differs for {name}")
        old_wave, old_wave_info = decode_float_wav_bytes(audio_io, old_wav_bytes)
        if old_wave_info["pcm_float32_sha256"] != EXPECTED_REFERENCE["old_output_pcm_sha256"]:
            raise PreflightError("old trained candidate reference PCM digest differs")
        old_runtime = old_metadata.get("runtime")
        if not isinstance(old_runtime, dict) or old_runtime.get("device") != "cpu":
            raise PreflightError("old same-profile CPU reference lacks recorded CPU runtime")
        for key in ("python_version", "numpy_version", "torch_version", "torch_interop_threads", "torch_intraop_threads"):
            if old_runtime.get(key) != config.get("runtime", {}).get(key):
                raise PreflightError(f"old same-profile anchor runtime differs for {key}")

        receipt["inputs"] = {
            "plan": {"path": str(args.plan), "bytes": PLAN_BYTES, "sha256": PLAN_RAW_SHA256, "plan_sha256": PLAN_SELF_SHA256, "run_id": plan.get("run_id"), "status": plan.get("status")},
            "plan_manifest": {"path": str(args.plan_manifest), "bytes": PLAN_MANIFEST_BYTES, "sha256": PLAN_MANIFEST_RAW_SHA256, "manifest_sha256": PLAN_MANIFEST_SELF_SHA256},
            "candidate_config_cli03": config_pin,
            "predecessor_config_cli02": old_config_pin,
            "cli02_to_cli03_config_change": {"only_changed_field": "source_pins.cli", "projection_equal_after_cli_pin_update": True},
            "reference_output_wav": old_reference["output_wav"],
            "reference_metadata": old_reference["metadata"],
            "reference_input": {"path": EXPECTED_REFERENCE["input_path"], "bytes": EXPECTED_REFERENCE["input_bytes"], "sha256": EXPECTED_REFERENCE["input_file_sha256"], "pcm_float32_sha256": EXPECTED_REFERENCE["input_pcm_sha256"], "sample_count": EXPECTED_REFERENCE["sample_count"], "sample_rate_hz": 16000},
            "dev_sample_count": len(inputs_by_sample),
            "dev_sample_ids": sorted(inputs_by_sample),
            "planned_rows": len(rows),
        }
        receipt["staged_provenance"] = provenance_copies
        receipt["runtime"] = runtime
        receipt["reference_comparison"] = {
            "sample_id": EXPECTED_REFERENCE["sample_id"],
            "old_cli_output_profile_role": old_metadata.get("profile_role"),
            "old_cli_source_sha256": old_metadata.get("source_pins", {}).get("cli", {}).get("sha256"),
            "old_config_sha256": old_metadata.get("config", {}).get("sha256"),
            "old_runtime": old_runtime,
            "new_runtime": {"python_version": runtime["python_version"], "numpy_version": runtime["numpy_version"], "torch_version": runtime["torch_version"], "device": runtime["device"], "torch_interop_threads": EXPECTED_INTER_THREADS, "torch_intraop_threads": EXPECTED_INTRA_THREADS, "strict_bootstrap_sha256": BOOTSTRAP_SHA256},
            "bootstrap_observability_in_old_reference": "not recorded in the archived CLI02 metadata",
            "exact_pcm_equality_is_gate": False,
            "comparison_purpose": "same-profile candidate CPU implementation/runtime reference; any PCM delta is recorded and interpreted with the runtime/bootstrap provenance",
        }

        # First run the exact same-profile reference input, then all twenty Dev
        # inputs in this same process. The CLI03 interop setter path is thereby
        # exercised after PyTorch work has already begun.
        tasks: list[tuple[str, str, dict[str, Any], int | None]] = [("same_profile_reference", EXPECTED_REFERENCE["sample_id"], EXPECTED_REFERENCE, None)]
        tasks.extend(
            ("candidate_dev", sample_id, pin, native_binding_by_sample[sample_id]["expected_inference_frame_count"])
            for sample_id, pin in sorted(inputs_by_sample.items())
        )
        for task_kind, sample_id, expected, native_expected_frame_count in tasks:
            input_path = pathlib.Path(expected.get("input_path") or expected.get("path"))
            no_symlink_components(input_path)
            _raw_input, input_file_pin = file_pin(input_path, expected_bytes=expected["input_bytes"] if "input_bytes" in expected else expected.get("file_bytes"), expected_sha=expected.get("input_file_sha256") or expected.get("file_sha256"), maximum=MAX_INPUT_BYTES)
            source_pcm, input_info = audio_io.read_canonical_wav(input_path, target_rate=16000)
            expected_pcm = expected.get("input_pcm_sha256") or expected.get("pcm_sha256")
            expected_count = expected.get("sample_count")
            if input_info.file_sha256 != input_file_pin["sha256"] or input_info.pcm_float32_sha256 != expected_pcm:
                raise PreflightError(f"input source/PCM pin mismatch for {sample_id}")
            if input_info.frames != expected_count or input_info.sample_rate != 16000 or input_info.source_channels != 1:
                raise PreflightError(f"input shape/clock differs for {sample_id}")
            if input_info.resampler != "none" or input_info.source_frames != expected_count or input_info.source_sample_rate != 16000:
                raise PreflightError(f"input requires resampling or retiming for {sample_id}")
            if source_pcm.dtype != np.float32 or source_pcm.ndim != 1 or source_pcm.size != expected_count or not np.isfinite(source_pcm).all():
                raise PreflightError(f"canonical source PCM shape/finite check failed for {sample_id}")
            source_rms = float(np.sqrt(np.mean(np.square(source_pcm.astype(np.float64)))))
            if not math.isfinite(source_rms):
                raise PreflightError(f"source RMS is not finite for {sample_id}")

            if task_kind == "same_profile_reference":
                out_dir = args.output_root / "same_profile_reference" / sample_id
            else:
                out_dir = args.output_root / "candidate_dev" / sample_id
            out_wav = out_dir / "enhanced.wav"
            out_json = out_dir / "cli_metadata.json"
            cli_meta = cli.run(input_path, out_wav, args.config, out_json)
            if cli_meta.get("schema") != RESULT_SCHEMA or cli_meta.get("status") != "PASS":
                raise PreflightError(f"CLI03 did not pass for {sample_id}")
            if cli_meta.get("profile_role") != "METHOD_CANDIDATE" or cli_meta.get("config", {}).get("sha256") != CONFIG03_RAW_SHA256:
                raise PreflightError(f"CLI03 metadata config/profile binding differs for {sample_id}")
            if cli_meta.get("source_audio", {}).get("file_sha256") != input_file_pin["sha256"]:
                raise PreflightError(f"CLI03 did not report the pinned original input for {sample_id}")
            source_meta = cli_meta.get("source_audio", {})
            canonical_meta = cli_meta.get("canonical_input_pcm", {})
            if (
                source_meta.get("sample_count") != expected_count
                or source_meta.get("sample_rate_hz") != 16000
                or source_meta.get("channels") != 1
                or canonical_meta.get("dtype") != "float32-le"
                or canonical_meta.get("channels") != 1
                or canonical_meta.get("sample_rate_hz") != 16000
                or canonical_meta.get("sample_count") != expected_count
                or canonical_meta.get("pcm_sha256") != expected_pcm
            ):
                raise PreflightError(f"CLI03 canonical source PCM/clock differs for {sample_id}")
            if cli_meta.get("head", {}).get("parameter") != config.get("parameter"):
                raise PreflightError(f"CLI03 output metadata parameter differs for {sample_id}")
            cli_provenance = cli_meta.get("head", {}).get("candidate_provenance", {})
            if cli_provenance.get("checkpoint_sha256") != EXPECTED_CHECKPOINT or cli_provenance.get("training_record_sha256") != EXPECTED_TRAINING:
                raise PreflightError(f"CLI03 output metadata candidate provenance differs for {sample_id}")
            if cli_provenance.get("pins") != config.get("candidate_provenance"):
                raise PreflightError(f"CLI03 output metadata provenance file pins differ for {sample_id}")
            output_meta = cli_meta.get("enhanced_output", {})
            if output_meta.get("sample_count") != expected_count or output_meta.get("sample_rate_hz") != 16000 or output_meta.get("channels") != 1:
                raise PreflightError(f"CLI03 output changed sample count/clock for {sample_id}")
            if output_meta.get("encoding") != "IEEE_FLOAT_32_LE" or output_meta.get("readback_bit_identical") is not True:
                raise PreflightError(f"CLI03 output WAV format/readback did not pass for {sample_id}")
            for key in ("rms", "peak"):
                if isinstance(output_meta.get(key), bool) or not isinstance(output_meta.get(key), (int, float)) or not math.isfinite(float(output_meta[key])):
                    raise PreflightError(f"CLI03 output {key} is not finite for {sample_id}")
            out_pcm, out_info = audio_io.read_canonical_wav(out_wav, target_rate=16000)
            _output_file_raw, output_file_pin = file_pin(out_wav, expected_sha=output_meta.get("file_sha256"), maximum=MAX_INPUT_BYTES)
            if out_info.pcm_float32_sha256 != output_meta.get("pcm_float32_sha256") or out_info.frames != expected_count:
                raise PreflightError(f"independent output WAV readback differs for {sample_id}")
            if not np.isfinite(out_pcm).all():
                raise PreflightError(f"enhanced PCM contains non-finite values for {sample_id}")
            metadata_raw, metadata_pin = file_pin(out_json, expected_sha=sha256(json.dumps(cli_meta, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"), maximum=MAX_JSON_BYTES)
            policy_after = observed_policy(bootstrap_module, torch, np)
            if policy_after != initial_policy or torch.cuda.is_initialized():
                raise PreflightError(f"strict process policy changed or CUDA initialized after {sample_id}")
            row_record = {
                "kind": task_kind,
                "sample_id": sample_id,
                "input": {**input_file_pin, "pcm_float32_sha256": input_info.pcm_float32_sha256, "sample_count": int(input_info.frames), "sample_rate_hz": int(input_info.sample_rate), "source_sample_rate_hz": int(input_info.source_sample_rate), "channels": int(input_info.source_channels), "source_frames": int(input_info.source_frames), "resampler": input_info.resampler, "rms": source_rms},
                "output_wav": output_file_pin,
                "cli_metadata": metadata_pin,
                "output_pcm_float32_sha256": out_info.pcm_float32_sha256,
                "output_sample_count": int(out_info.frames),
                "output_sample_rate_hz": int(out_info.sample_rate),
                "output_channels": 1,
                "output_rms": float(output_meta["rms"]),
                "output_peak": float(output_meta["peak"]),
                "native_expected_frame_count": native_expected_frame_count,
                "strict_policy_preserved": True,
                "cuda_initialized_after_row": False,
            }
            if task_kind == "same_profile_reference":
                old_wave, old_wave_info = decode_float_wav_bytes(audio_io, old_wav_bytes)
                if old_wave.size != out_pcm.size or old_wave_info["sample_rate_hz"] != out_info.sample_rate:
                    raise PreflightError("same-profile old/new reference output clocks or shapes differ")
                delta = out_pcm.astype(np.float64) - old_wave.astype(np.float64)
                row_record["reference_comparison"] = {
                    "old_pcm_float32_sha256": old_wave_info["pcm_float32_sha256"],
                    "new_pcm_float32_sha256": out_info.pcm_float32_sha256,
                    "exact_pcm_equal": bool(np.array_equal(out_pcm, old_wave)),
                    "max_abs_difference": float(np.max(np.abs(delta))) if delta.size else 0.0,
                    "rmse_difference": float(np.sqrt(np.mean(np.square(delta)))) if delta.size else 0.0,
                    "gating": False,
                    "interpretation": "recorded reference comparison; runtime bootstrap is not observed in old metadata and equality is not a preflight gate",
                }
                receipt["scope"]["same_profile_reference_rows_completed"] += 1
                receipt["reference_comparison"]["new_output_pcm_float32_sha256"] = out_info.pcm_float32_sha256
                receipt["reference_comparison"]["exact_pcm_equal"] = bool(np.array_equal(out_pcm, old_wave))
                receipt["reference_comparison"]["max_abs_difference"] = row_record["reference_comparison"]["max_abs_difference"]
                receipt["reference_comparison"]["rmse_difference"] = row_record["reference_comparison"]["rmse_difference"]
            else:
                receipt["scope"]["candidate_audio_rows_completed"] += 1
            receipt["candidate_rows"].append(row_record)

        if receipt["scope"]["candidate_audio_rows_completed"] != 20 or receipt["scope"]["same_profile_reference_rows_completed"] != 1:
            raise PreflightError("not all expected CPU preflight rows completed")
        free_after = shutil.disk_usage(args.output_root).free
        if free_after < MIN_FREE_BYTES:
            raise PreflightError(f"SHM free space {free_after} is below 5 GiB reserve after preflight")
        receipt["resource"] = {"filesystem_free_bytes_before": free_before, "filesystem_free_bytes_after": free_after, "reserve_bytes": MIN_FREE_BYTES}
        receipt["scope"]["cuda_initialized"] = bool(torch.cuda.is_initialized())
        receipt["status"] = "PASS_CPU_PREFLIGHT"
    except Exception as exc:
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc)[:1200]}
        if "resource" not in receipt and args.output_root.is_dir():
            try:
                receipt["resource"] = {"filesystem_free_bytes_at_failure": shutil.disk_usage(args.output_root).free, "reserve_bytes": MIN_FREE_BYTES}
            except OSError:
                pass
    receipt["finished_unix"] = time.time()
    receipt["receipt_sha256"] = sha256(canonical(receipt))
    raw_receipt = (json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    create_exclusive(args.receipt, raw_receipt)
    print(json.dumps({"schema": SCHEMA, "status": receipt["status"], "receipt_path": str(args.receipt), "receipt_bytes": len(raw_receipt), "receipt_sha256": sha256(raw_receipt), "receipt_self_sha256": receipt["receipt_sha256"], "candidate_rows_completed": receipt["scope"]["candidate_audio_rows_completed"], "same_profile_reference_rows_completed": receipt["scope"]["same_profile_reference_rows_completed"]}, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
