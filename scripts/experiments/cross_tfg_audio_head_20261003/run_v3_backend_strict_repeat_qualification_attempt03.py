#!/usr/bin/env python3
"""Qualify exact strict repeats for the three same-host video backends.

Engineering-only pilot on the already exposed QC06 natural_1/portrait input.
It does not create Dev/Confirm rows, score an audio candidate, call an API, or
select parameters. Each backend renders the identical waveform twice at the
same seed, then the existing official SyncNet pipeline/scorer is run on both
outputs. The complete media, score arrays, tracks, and crops remain in the
output tree. A PASS receipt is emitted only if every frozen equality gate is
exact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
BASE = "scripts/experiments/cross_tfg_audio_head_20261003"
RUN_ID = "cross_tfg_backend_strict_repeat_20261008_attempt03"
DEFAULT_OUTPUT = ROOT / "runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/backend_strict_repeat_attempt03"
QC06_SOURCE = Path("/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/inputs/pilot/natural_1.wav")
QC06_FACE = Path("/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/inputs/primary_face.png")
QC06_SOURCE_SHA256 = "263b19510c95c32db4874449aadefc9b37c008e4816c18028eaf8fef463828e5"
QC06_FACE_SHA256 = "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6"
QC06_MANIFEST_SHA256 = "f09cc4f339c43e96ea532cdaac4f9818fc42caec6ba7b212dcfe7ae4dc2c5436"
QC06_POLICY_SHA256 = "196f6b3654f0bf69eed74dbadd6f4948272d52edfcc5101945ba61044ea6f7cc"
STRICT_BOOTSTRAP_REL = BASE + "/runtime_determinism_attempt02/sitecustomize.py"
STRICT_BOOTSTRAP_SHA256 = "a2a56ad76a94f345e489bc186409958be201c2ad50bfe91c07a6f2902be59855"
STRICT_POLICY = {
    "pythonhashseed": "20261003", "cublas_workspace_config": ":4096:8",
    "omp_num_threads": "1", "mkl_num_threads": "1", "openblas_num_threads": "1",
    "numexpr_num_threads": "1", "torch_num_threads": 1, "torch_num_interop_threads": 1,
    "torch_use_deterministic_algorithms": True, "torch_cudnn_deterministic": True,
    "torch_cudnn_benchmark": False, "torch_allow_tf32": False,
}
GPU_UUID = "GPU-d09622d0-3535-e662-bb45-3df2b049fcea"
REPEAT_SEED = 20261003
MIN_GPU_FREE_MIB = 18000  # same explicit headroom gate required by the pinned Ditto adapter

W2L_ID = "wav2lip_gan_bac9a81e"
MUSE_ID = "musetalk_fixed_portrait_attempt16"
DITTO_ID = "ditto_hubert_fixed_portrait_attempt18"
MODELS = (W2L_ID, MUSE_ID, DITTO_ID)
FRONTENDS = {
    W2L_ID: "wav2lip_official_16k_float32_mel",
    MUSE_ID: "musetalk_whisper_fixed_portrait_haar_expanded_box_fp16",
    DITTO_ID: "ditto_hubert_onnx_cpu_torch_cuda_static_portrait_attempt18",
}

W2L_ROOT = Path("/root/autodl-tmp/wav2lip-syncnet/Wav2Lip")
W2L_PYTHON = Path("/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python")
W2L_CHECKPOINT = W2L_ROOT / "checkpoints/wav2lip_gan.pth"
W2L_HEAD = "bac9a81e63ecc153202353372e5724b83d9e6322"
W2L_DEPENDENCIES = {
    "audio.py": "71d5e927cac3d0b7103e7671a43b82b41b6a56398a092bbfa2e96d3041249c4c",
    "hparams.py": "4a4c31dc51ebd8140da18ac24841e582b2472389f7c65e7f0a6b4447fc0c71cf",
    "inference.py": "99662bfce2fdaf17c1cff8e71977ba02cb80c684058cd114c4e0648b12059fc6",
    "models/wav2lip.py": "a02b54288f8200332ee4f0ea0c8013bca542b94d8149e514ac25a742a7a70e66",
    "checkpoints/wav2lip_gan.pth": "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8",
}
W2L_ENTRY = W2L_ROOT / "inference.py"
MUSE_SHIM_REL = BASE + "/musetalk_five_arg_shim_attempt06.py"
MUSE_SHIM_SHA256 = "75f61c0ea97fabad43f7bd82536abceb84100feb6fd3bc80dd844797b52106f5"
MUSE_ENTRY = Path("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt16/code/render_silent_video_attempt09.py")
MUSE_ENTRY_SHA256 = "b75004340fbf90b937fb516c5d8ee013f193f763f5cf61bbfc7d28cb12ce88c8"
DITTO_SHIM_REL = BASE + "/ditto_five_arg_shim_attempt05.py"
DITTO_SHIM_SHA256 = "1d9691d297ab983203d3f5bf5d77034fd9ad38941147f4f62b166b6045557c5a"
DITTO_ENTRY = Path("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt18/code/render_silent_video_attempt07.py")
DITTO_ENTRY_SHA256 = "1415ee28206ba6afe70e32fb728b29fb3b49bb638bdc21d3c2a7e89c66131df0"

SYNCNET_ROOT = Path("/root/autodl-tmp/wav2lip-syncnet/syncnet_python")
SYNCNET_PYTHON = Path("/root/autodl-tmp/wav2lip-syncnet/envs/syncnet/bin/python")
SYNCNET_CHECKPOINT = SYNCNET_ROOT / "data/syncnet_v2.model"
SYNCNET_HEAD = "907c0b579c2e2d83f0eae1b2ac9e720cde4e5623"
SYNCNET_FILES = {
    "SyncNetInstance.py": "bd722bf844700d6a923dec2f10d7cacd65c0c6b22d09aa5d44d2509660e900d6",
    "SyncNetModel.py": "41b85aed0006546cb2d6dc9180c5bc3dac737ab80eff643022e397a349c30c22",
    "run_pipeline.py": "3654a3cf22c74cb3071fca6fb97ca9fb7dd1a5943fe864169ff8c0b57eb80acf",
    "run_syncnet.py": "c78fe9f75665aa9e7a2bb523e4e7e3734a857137ff70b29c88a7b4893c422ded",
}
SYNCNET_CHECKPOINT_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
FFMPEG = Path("/usr/bin/ffmpeg")
FFMPEG_SHA256 = "36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45"
DEPENDENCY_TEMPLATE = ROOT / "runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/runtime_binding_template_attempt03.json"
DEPENDENCY_TEMPLATE_SHA256 = "129163ed85effcf10fabc4166a5bc79239ff66a47ab37ca92ba540e9d2584fab"
RUNNER04_REL = BASE + "/run_v3_samehost_dev360_controls_attempt04.py"
RUNNER04_SHA256 = "f4a9b582caa1342de8c3208707174fe98bb2dbf93262658a84e03e555cf6b934"
QC_HELPER_REL = BASE + "/engineering_qc_oldpilot_attempt06.py"
QC_HELPER_SHA256 = "cf4d34832ee93bf5a4d2e4be349dad15469e423888c0a1c41bbbdf53f2a4726d"


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def write_json(path: Path, value: Any) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                     allow_nan=False).encode("utf-8") + b"\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return raw


def load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load pinned helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git_head(path: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False, timeout=30)
    if result.returncode:
        raise RuntimeError(f"git HEAD unavailable for {path}: {result.stderr.decode('utf-8', 'replace')[-1000:]}")
    return result.stdout.decode("ascii", "strict").strip()


def statvfs_record(path: Path) -> dict[str, Any]:
    probe = path
    while not probe.exists():
        if probe.parent == probe:
            raise FileNotFoundError(path)
        probe = probe.parent
    resolved = probe.resolve(strict=True)
    fs = os.statvfs(resolved)
    return {"requested_path": str(path), "existing_path": str(resolved),
            "device": int(resolved.stat().st_dev), "block_size": int(fs.f_frsize),
            "available_bytes": int(fs.f_bavail * fs.f_frsize),
            "total_bytes": int(fs.f_blocks * fs.f_frsize)}


def cgroup_memory_record() -> dict[str, Any]:
    result: dict[str, Any] = {"controller": "unavailable", "path": None,
                              "current_bytes": None, "limit_bytes": None,
                              "available_bytes": None, "identity": None}
    try:
        lines = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
        row = next((line for line in lines if line.startswith("0::")), None)
        if row is not None:
            cgroup_rel = row.split("::", 1)[1].lstrip("/")
            base = Path("/sys/fs/cgroup") / cgroup_rel
            current = int((base / "memory.current").read_text().strip())
            raw_limit = (base / "memory.max").read_text().strip()
            limit = None if raw_limit == "max" else int(raw_limit)
            result.update({"controller": "cgroup_v2", "path": str(base),
                           "current_bytes": current, "limit_bytes": limit,
                           "available_bytes": None if limit is None else max(0, limit - current),
                           "identity": {"controller": "cgroup_v2", "relative_path": cgroup_rel,
                                        "limit_bytes": limit, "inode": int(base.stat().st_ino)}})
            return result
        for line in lines:
            fields = line.split(":", 2)
            if len(fields) != 3 or "memory" not in fields[1].split(","):
                continue
            cgroup_rel = fields[2].lstrip("/")
            base = Path("/sys/fs/cgroup/memory") / cgroup_rel
            current = int((base / "memory.usage_in_bytes").read_text().strip())
            limit = int((base / "memory.limit_in_bytes").read_text().strip())
            result.update({"controller": "cgroup_v1", "path": str(base),
                           "current_bytes": current, "limit_bytes": limit,
                           "available_bytes": max(0, limit - current),
                           "identity": {"controller": "cgroup_v1", "relative_path": cgroup_rel,
                                        "limit_bytes": limit, "inode": int(base.stat().st_ino)}})
            return result
    except (OSError, ValueError, StopIteration):
        pass
    return result


def resource_snapshot(output_root: Path) -> dict[str, Any]:
    gpu_run = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.total,memory.free,driver_version",
         "--format=csv,noheader,nounits"], stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=15)
    if gpu_run.returncode:
        raise RuntimeError("nvidia-smi identity query failed: " + gpu_run.stderr.decode("utf-8", "replace")[-1000:])
    gpus = []
    for row in csv.reader(gpu_run.stdout.decode("utf-8", "strict").splitlines(), skipinitialspace=True):
        if len(row) != 4:
            raise RuntimeError(f"unexpected nvidia-smi identity row: {row!r}")
        gpus.append({"uuid": row[0].strip(), "memory_total_mib": int(row[1]),
                     "memory_free_mib": int(row[2]), "driver_version": row[3].strip()})
    host_available = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                host_available = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    cgroup = cgroup_memory_record()
    effective_ram = host_available
    if isinstance(cgroup.get("available_bytes"), int):
        effective_ram = cgroup["available_bytes"] if effective_ram is None else min(effective_ram, cgroup["available_bytes"])
    try:
        shm = statvfs_record(Path("/dev/shm"))
    except OSError:
        shm = None
    return {
        "observed_unix": time.time(), "host": platform.node(),
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "python_version": platform.python_version(), "gpus": gpus,
        "host_mem_available_bytes": host_available, "cgroup_memory": cgroup,
        "effective_ram_available_bytes": effective_ram,
        "shm": shm, "output_filesystem": statvfs_record(output_root),
    }


def verify_file(path: Path, expected_sha: str, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual = sha_file(path)
    if actual != expected_sha:
        raise RuntimeError(f"{label} SHA mismatch: {path}: {actual}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}


def hash_dependency(path_text: str, expected_sha: str, expected_bytes: int | None,
                    aliases: Mapping[str, str], label: str) -> dict[str, Any]:
    lexical = Path(os.path.abspath(path_text))
    if not lexical.is_absolute():
        raise ValueError(f"dependency path must be absolute: {path_text}")
    cursor = Path(lexical.anchor)
    chain: list[dict[str, str]] = []
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            expected_target = aliases.get(str(cursor))
            if expected_target is None:
                raise ValueError(f"dependency traverses an unapproved symlink: {cursor}")
            actual_target = cursor.resolve(strict=True)
            if actual_target != Path(expected_target).resolve(strict=True):
                raise ValueError(f"approved alias points to another target: {cursor}")
            chain.append({"path": str(cursor), "resolved_target": str(actual_target)})
        elif getattr(cursor, "is_junction", lambda: False)():
            raise ValueError(f"dependency traverses an unapproved junction: {cursor}")
    resolved = lexical.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"backend dependency is not a regular file: {path_text}")
    info = resolved.stat()
    if expected_bytes is not None and int(info.st_size) != expected_bytes:
        raise ValueError(f"backend dependency byte count differs from frozen pin: {path_text}")
    actual = sha_file(resolved)
    if actual != expected_sha:
        raise ValueError(f"backend dependency SHA differs from frozen pin: {path_text}: {actual}")
    return {"path": str(lexical), "resolved_path": str(resolved), "bytes": int(info.st_size),
            "sha256": actual, "approved_symlink_chain": chain}


def verify_dependency_template(requested_models: tuple[str, ...]) -> tuple[dict[str, Any], bytes, dict[str, Any], dict[str, list[dict[str, Any]]]]:
    template_path = DEPENDENCY_TEMPLATE
    raw = template_path.read_bytes()
    if sha_bytes(raw) != DEPENDENCY_TEMPLATE_SHA256:
        raise RuntimeError("frozen backend dependency template SHA mismatch")
    template = json.loads(raw.decode("utf-8"))
    if (not str(template.get("authority", "")).startswith("Metadata-only input template")
            or template.get("backend_source_policy", {}).get("dev_confirm_face_video_content_read") is not False):
        raise ValueError("dependency template is not the frozen source-only metadata template")
    if set(template.get("backends", {})) != set(MODELS):
        raise ValueError("dependency template does not contain exactly the three frozen backend identities")
    runner_path = ROOT / RUNNER04_REL
    verify_file(runner_path, RUNNER04_SHA256, "guarded launcher alias registry")
    runner = load_module(runner_path, "samehost_dev360_guard_for_repeat_qualification")
    aliases = runner.MUSE_ALLOWED_SYMLINKS
    observed: dict[str, list[dict[str, Any]]] = {}
    for model_id in requested_models:
        backend = template["backends"][model_id]
        deps = backend.get("dependencies")
        if not isinstance(deps, list) or not deps:
            raise ValueError(f"frozen dependency list is empty for {model_id}")
        current: list[dict[str, Any]] = []
        for dep in deps:
            if not isinstance(dep, dict) or not isinstance(dep.get("path"), str):
                raise ValueError(f"invalid dependency row for {model_id}")
            record = hash_dependency(dep["path"], str(dep.get("sha256", "")),
                                     dep.get("bytes") if type(dep.get("bytes")) is int else None,
                                     aliases if model_id == MUSE_ID else {},
                                     f"{model_id}:{dep.get('role')}")
            record.update({"role": dep.get("role"), "inventory_id": dep.get("inventory_id")})
            current.append(record)
        observed[model_id] = current
    return template, raw, {"path": str(template_path), "bytes": len(raw), "sha256": sha_bytes(raw)}, observed


def verify_runtime_files(template: Mapping[str, Any], observed_dependencies: Mapping[str, list[dict[str, Any]]],
                         requested_models: tuple[str, ...]) -> dict[str, Any]:
    project_files = {
        QC_HELPER_REL: QC_HELPER_SHA256,
        STRICT_BOOTSTRAP_REL: STRICT_BOOTSTRAP_SHA256,
        MUSE_SHIM_REL: MUSE_SHIM_SHA256,
        DITTO_SHIM_REL: DITTO_SHIM_SHA256,
    }
    files = {rel: verify_file(ROOT / rel, digest, rel) for rel, digest in project_files.items()}
    if sha_file(QC06_SOURCE) != QC06_SOURCE_SHA256 or sha_file(QC06_FACE) != QC06_FACE_SHA256:
        raise RuntimeError("actual exposed QC06 natural_1/portrait input pins do not match")
    w2l_head = git_head(W2L_ROOT)
    if w2l_head != W2L_HEAD:
        raise RuntimeError(f"Wav2Lip HEAD differs from frozen model identity: {w2l_head}")
    w2l_files = {rel: verify_file(W2L_ROOT / rel, expected, "Wav2Lip " + rel)
                 for rel, expected in W2L_DEPENDENCIES.items()}
    if sha_file(SYNCNET_CHECKPOINT) != SYNCNET_CHECKPOINT_SHA256:
        raise RuntimeError("SyncNet checkpoint SHA differs from QC06")
    syncnet_head = git_head(SYNCNET_ROOT)
    if syncnet_head != SYNCNET_HEAD:
        raise RuntimeError(f"SyncNet source HEAD differs from QC06: {syncnet_head}")
    syncnet_files = {rel: verify_file(SYNCNET_ROOT / rel, expected, "SyncNet " + rel)
                     for rel, expected in SYNCNET_FILES.items()}
    if sha_file(FFMPEG) != FFMPEG_SHA256:
        raise RuntimeError("FFmpeg executable differs from the actual Ditto/runtime pin")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise FileNotFoundError("ffprobe is unavailable on PATH")
    backend_identity: dict[str, Any] = {}
    for model_id in requested_models:
        backend = template["backends"][model_id]
        model_binding = backend.get("model_binding")
        if (not isinstance(model_binding, dict)
                or model_binding.get("model_id") != model_id
                or model_binding.get("audio_frontend") != FRONTENDS[model_id]):
            raise ValueError(f"frozen model binding identity/frontend differs for {model_id}")
        backend_identity[model_id] = {
            "model_binding": model_binding,
            "generator": {key: value for key, value in backend.get("generator", {}).items()
                          if key not in ("env", "CROSS_TFG_CONTROL_DIR")},
            "source_dependency_count": len(observed_dependencies[model_id]),
            "source_dependency_manifest_sha256": sha_bytes(canonical(observed_dependencies[model_id])),
            "source_model_dependencies": observed_dependencies[model_id],
        }
    return {"workspace_sources": files, "wav2lip_git_head": w2l_head,
            "wav2lip_dependencies": w2l_files, "syncnet_git_head": syncnet_head,
            "syncnet_dependencies": syncnet_files,
            "syncnet_checkpoint": {"path": str(SYNCNET_CHECKPOINT), "bytes": SYNCNET_CHECKPOINT.stat().st_size,
                                  "sha256": SYNCNET_CHECKPOINT_SHA256},
            "ffmpeg": {"path": str(FFMPEG), "bytes": FFMPEG.stat().st_size, "sha256": FFMPEG_SHA256},
            "ffprobe": {"path": str(Path(ffprobe).resolve(strict=True)),
                        "bytes": Path(ffprobe).stat().st_size, "sha256": sha_file(Path(ffprobe))},
            "backend_identity": backend_identity,
            "backend_dependencies": dict(observed_dependencies)}


def run_probe(argv: list[str], *, timeout: int = 60) -> str:
    proc = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f"runtime identity probe failed ({proc.returncode}): {proc.stderr.decode('utf-8','replace')[-1200:]}")
    return proc.stdout.decode("utf-8", "strict").strip()


def python_environment_identity(python: Path) -> dict[str, Any]:
    code = "import json,torch,numpy; import librosa; print(json.dumps({'python':__import__('sys').executable,'torch':torch.__version__,'numpy':numpy.__version__,'librosa':librosa.__version__},sort_keys=True))"
    raw = json.loads(run_probe([str(python), "-c", code]))
    return raw


def configure_parent_score_torch_once() -> Any:
    """Configure the in-process score reduction without re-setting inter-op."""
    import torch
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    return torch


def environment_for_row(model_id: str, row_id: str, receipt_dir: Path,
                        row_control_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "0", "PYTHONHASHSEED": "20261003",
        "TFG_DETERMINISM_SEED": str(REPEAT_SEED), "CROSS_TFG_RENDER_SEED": str(REPEAT_SEED),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CROSS_TFG_STRICT_STARTUP_POLICY": "1", "CROSS_TFG_ROW_ID": row_id,
        "CROSS_TFG_MODEL_ID": model_id, "CROSS_TFG_AUDIO_FRONTEND": FRONTENDS[model_id],
        "CROSS_TFG_BACKEND_ENTRY": str({W2L_ID: W2L_ENTRY, MUSE_ID: MUSE_ENTRY,
                                         DITTO_ID: DITTO_ENTRY}[model_id]),
        "CROSS_TFG_DETERMINISM_RECEIPT_DIR": str(receipt_dir.resolve()),
        "CROSS_TFG_CONTROL_DIR": str(row_control_dir.resolve()),
        "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    bootstrap_dir = (ROOT / STRICT_BOOTSTRAP_REL).parent
    if not (bootstrap_dir / "sitecustomize.py").is_file():
        raise FileNotFoundError(f"strict startup bootstrap directory is incomplete: {bootstrap_dir}")
    if model_id == W2L_ID:
        additional = [ROOT]
    elif model_id == MUSE_ID:
        additional = [
            Path("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt09/code/overlay_python310_attempt02"),
            Path("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt10/code/future_overlay_attempt02"),
            Path("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt16/code"),
        ]
    else:
        additional = [
            Path("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt13/code/overlays/setuptools_overlay_attempt12"),
            Path("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt13/code/overlays/cython_overlay_attempt12"),
            Path("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt13/code/overlays/filetype_overlay"),
            Path("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt10/code/future_overlay_attempt02"),
            Path("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt09/code/overlay_python310_attempt02"),
            Path("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_sources_attempt02/ditto-talkinghead"),
        ]
    env["PYTHONPATH"] = os.pathsep.join([str(bootstrap_dir), *[str(path) for path in additional]])
    return env


def strict_receipt_records(receipt_dir: Path, row_id: str, model_id: str,
                           expected_entry: Path, expected_python: Path) -> list[dict[str, Any]]:
    records = []
    expected_python_text = str(expected_python.resolve(strict=True))
    for path in sorted(receipt_dir.glob("*.json")):
        raw = path.read_bytes()
        record = json.loads(raw.decode("utf-8"))
        if (record.get("schema") != "cross_tfg_renderer_strict_determinism_bootstrap_receipt_v1"
                or record.get("status") != "PASS"
                or record.get("receipt_sha256") != payload_sha(record, "receipt_sha256")
                or record.get("bootstrap_source_sha256") != STRICT_BOOTSTRAP_SHA256
                or record.get("policy") != STRICT_POLICY or record.get("row_id") != row_id
                or record.get("model_id") != model_id or record.get("audio_frontend") != FRONTENDS[model_id]
                or record.get("seed") != REPEAT_SEED or record.get("python_executable") != expected_python_text
                or record.get("pid") != int(path.stem)):
            raise RuntimeError(f"strict startup receipt failed actual-process validation: {path}")
        entry_path = str(expected_entry.resolve(strict=True))
        if record.get("expected_backend_entry") != entry_path:
            raise RuntimeError(f"strict startup receipt expected another backend entry: {path}")
        if record.get("entry_matches_backend") is not (record.get("entry_path") == entry_path):
            raise RuntimeError(f"strict startup receipt entry equality flag is inconsistent: {path}")
        records.append({"path": str(path.resolve(strict=True)), "bytes": len(raw),
                        "sha256": sha_bytes(raw), "entry_path": record.get("entry_path"),
                        "entry_matches_backend": record["entry_matches_backend"],
                        "pid": record["pid"], "policy": record["policy"]})
    matching = [record for record in records if record["entry_matches_backend"] is True]
    if len(matching) != 1:
        raise RuntimeError(f"expected exactly one actual native renderer policy receipt for {row_id}; got {len(matching)}")
    return records


def render_one(model_id: str, row_id: str, audio_path: Path, face_path: Path,
               row_root: Path, runner: Any) -> dict[str, Any]:
    render_dir = row_root / "render"
    render_dir.mkdir(parents=True, exist_ok=False)
    receipt_dir = row_root / "strict_startup_receipts"
    receipt_dir.mkdir()
    control_dir = row_root / "control"
    control_dir.mkdir()
    child_cwd = row_root / "generator_cwd"
    child_cwd.mkdir()
    generated = render_dir / "generated_video.mp4"
    env = environment_for_row(model_id, row_id, receipt_dir, control_dir)
    if model_id == W2L_ID:
        argv = [str(W2L_PYTHON), str(W2L_ENTRY), "--checkpoint_path", str(W2L_CHECKPOINT),
                "--face", str(face_path), "--audio", str(audio_path), "--outfile", str(generated),
                "--static", "True", "--fps", "25", "--pads", "0", "10", "0", "0"]
        entry = W2L_ENTRY
    else:
        shim_rel, entry = ((MUSE_SHIM_REL, MUSE_ENTRY) if model_id == MUSE_ID
                           else (DITTO_SHIM_REL, DITTO_ENTRY))
        shim = ROOT / shim_rel
        argv = [str(W2L_PYTHON), str(shim), "--audio", str(audio_path),
                "--face", str(face_path), "--outfile", str(generated),
                "--seed", str(REPEAT_SEED), "--fps", "25"]
    before = resource_snapshot(row_root)
    gpu = next((item for item in before["gpus"] if item["uuid"] == GPU_UUID), None)
    if gpu is None or gpu["memory_free_mib"] < MIN_GPU_FREE_MIB:
        raise RuntimeError(f"strict repeat requires the pinned GPU with >= {MIN_GPU_FREE_MIB} MiB free: {before['gpus']}")
    run_record = runner.run(
        argv, cwd=child_cwd, log_stem=row_root / "logs" / "renderer",
        timeout=1800, env=env)
    if run_record["returncode"] != 0 or not generated.is_file() or generated.stat().st_size == 0:
        raise RuntimeError(f"backend renderer failed for {row_id} ({run_record['returncode']})")
    after = resource_snapshot(row_root)
    after_gpu = next((item for item in after["gpus"] if item["uuid"] == GPU_UUID), None)
    if after_gpu is None or before["cgroup_memory"].get("identity") != after["cgroup_memory"].get("identity"):
        raise RuntimeError(f"GPU/cgroup identity changed during renderer process: {row_id}")
    startup_receipts = strict_receipt_records(receipt_dir, row_id, model_id, entry, W2L_PYTHON)
    adapter_receipt_path = generated.with_suffix(".adapter_receipt.json")
    adapter_ref: dict[str, Any] | None = None
    if model_id != W2L_ID:
        if not adapter_receipt_path.is_file():
            raise RuntimeError(f"native adapter did not create its required receipt: {row_id}")
        adapter_raw = adapter_receipt_path.read_bytes()
        adapter_obj = json.loads(adapter_raw.decode("utf-8"))
        if (adapter_obj.get("status") not in ("PASS", "PASS_ENGINEERING_ONLY")
                or adapter_obj.get("row_id") != row_id
                or adapter_obj.get("model_id") != model_id
                or adapter_obj.get("seed") != REPEAT_SEED):
            raise RuntimeError(f"native adapter receipt does not bind its actual model row: {row_id}")
        adapter_ref = {"path": str(adapter_receipt_path.resolve()), "bytes": len(adapter_raw),
                       "sha256": sha_bytes(adapter_raw), "receipt": adapter_obj}
    return {
        "row_id": row_id, "model_id": model_id, "audio_frontend": FRONTENDS[model_id],
        "seed": REPEAT_SEED, "renderer_command": run_record,
        "generated_video_path": str(generated.resolve()),
        "generated_video_bytes": generated.stat().st_size,
        "generated_video_sha256": sha_file(generated),
        "generated_video_elementary_stream_sha256": runner.video_stream_sha256(generated),
        "source_audio_path": str(audio_path.resolve()), "source_audio_file_sha256": sha_file(audio_path),
        "source_face_path": str(face_path.resolve()), "source_face_sha256": sha_file(face_path),
        "native_adapter_receipt": adapter_ref,
        "strict_startup_process_receipts": startup_receipts,
        "resource_before": before, "resource_after": after,
    }


def mux_and_score(model_id: str, row: dict[str, Any], audio_ref: Mapping[str, Any],
                  row_root: Path, helper: Any, bootstrap: Path,
                  parent_torch: Any, syncnet_python_identity: Mapping[str, Any]) -> dict[str, Any]:
    generated = Path(row["generated_video_path"])
    silent = row_root / "silent_video.mkv"
    muxed = row_root / "original_pcm_f32_mux.mkv"
    strip = helper.run([str(FFMPEG), "-v", "error", "-nostdin", "-y", "-i", str(generated),
                        "-map", "0:v:0", "-c:v", "copy", "-an", "-map_metadata", "-1",
                        "-f", "matroska", str(silent)], cwd=row_root,
                       log_stem=row_root / "logs" / "strip_audio", timeout=300)
    if strip["returncode"] != 0 or not silent.is_file() or silent.stat().st_size == 0:
        raise RuntimeError(f"failed to create silent render for {row['row_id']}")
    mux = helper.run([str(FFMPEG), "-v", "error", "-nostdin", "-y", "-i", str(silent),
                      "-i", str(audio_ref["path"]), "-map", "0:v:0", "-map", "1:a:0",
                      "-c:v", "copy", "-c:a", "copy", "-map_metadata", "0", "-f", "matroska", str(muxed)],
                     cwd=row_root, log_stem=row_root / "logs" / "mux_original_pcm", timeout=300)
    if mux["returncode"] != 0 or not muxed.is_file() or muxed.stat().st_size == 0:
        raise RuntimeError(f"failed to restore original PCM for {row['row_id']}")
    decoded_pcm = helper.decode_f32(muxed)
    if (len(decoded_pcm) != int(audio_ref["sample_count"]) * 4
            or helper.sha256_bytes(decoded_pcm) != audio_ref["pcm_sha256"]):
        raise RuntimeError(f"muxed PCM differs from the shared QC06 waveform for {row['row_id']}")
    streams = helper.read_ffprobe(muxed).get("streams", [])
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if (len(video_streams) != 1 or len(audio_streams) != 1
            or audio_streams[0].get("codec_name") != "pcm_f32le"
            or int(audio_streams[0].get("sample_rate", 0)) != 16000
            or int(audio_streams[0].get("channels", 0)) != 1):
        raise RuntimeError(f"muxed SyncNet video does not contain exact mono PCM_F32LE 16 kHz audio: {row['row_id']}")
    arm_id = row["row_id"]
    # The QC06 scorer owns its separate one-shot Torch bootstrap. Strip any
    # renderer-only sitecustomize context inherited from the outer dispatcher
    # so the official SyncNet child does not set inter-op threads twice.
    renderer_keys = [key for key in os.environ if key.startswith("CROSS_TFG_")]
    renderer_keys.extend(key for key in ("PYTHONPATH", "TFG_DETERMINISM_SEED")
                         if key in os.environ and key not in renderer_keys)
    saved_env = {key: os.environ.pop(key) for key in renderer_keys}
    saved_path = os.environ.get("PATH")
    os.environ["PATH"] = str(FFMPEG.parent) + os.pathsep + (saved_path or "")
    try:
        fresh_score = helper.score_with_official_syncnet(
            muxed, row_root, SYNCNET_ROOT, SYNCNET_PYTHON, arm_id, bootstrap, REPEAT_SEED)
    finally:
        os.environ.update(saved_env)
        if saved_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = saved_path
    if (parent_torch.get_num_threads() != 1 or parent_torch.get_num_interop_threads() != 1
            or not parent_torch.are_deterministic_algorithms_enabled()):
        raise RuntimeError("SyncNet score reducer parent Torch policy changed during pilot")
    row.update({
        "silent_video_path": str(silent.resolve()), "silent_video_bytes": silent.stat().st_size,
        "silent_video_sha256": sha_file(silent),
        "muxed_video_path": str(muxed.resolve()), "muxed_video_bytes": muxed.stat().st_size,
        "muxed_video_sha256": sha_file(muxed), "muxed_video_elementary_stream_sha256": helper.video_stream_sha256(muxed),
        "muxed_video_stream": video_streams[0],
        "muxed_original_pcm_readback": {"sample_rate_hz": 16000, "channels": 1,
            "sample_count": int(audio_ref["sample_count"]), "pcm_float32_sha256": helper.sha256_bytes(decoded_pcm),
            "matches_qc06_input": True},
        "strip_command": strip, "mux_command": mux,
        "fresh_syncnet": fresh_score,
        "syncnet_python_identity": dict(syncnet_python_identity),
    })
    return row


def compare_repeats(model_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 2:
        raise ValueError(f"exactly two independent render rows required for {model_id}")
    left, right = rows
    a = left["fresh_syncnet"]["tracking_artifacts"]
    b = right["fresh_syncnet"]["tracking_artifacts"]
    a_tracks, b_tracks = left["fresh_syncnet"]["tracks"], right["fresh_syncnet"]["tracks"]
    score_npz_equal = left["fresh_syncnet"]["distance_npz_sha256"] == right["fresh_syncnet"]["distance_npz_sha256"]
    score_arrays_equal = [item["distance_by_query_lag_sha256"] for item in a_tracks] == [
        item["distance_by_query_lag_sha256"] for item in b_tracks]
    crop_equal = [item["sha256"] for item in a["crop_videos"]] == [
        item["sha256"] for item in b["crop_videos"]]
    box_equal = [item["bbox_sha256"] for item in a["numeric_tracks"]] == [
        item["bbox_sha256"] for item in b["numeric_tracks"]]
    generated_equal = left["generated_video_sha256"] == right["generated_video_sha256"]
    elementary_equal = (left["generated_video_elementary_stream_sha256"]
                        == right["generated_video_elementary_stream_sha256"])
    tracks_equal = a["tracks_pickle_sha256"] == b["tracks_pickle_sha256"]
    gate_values = {
        "generated_video_sha256_equal": generated_equal,
        "score_npz_sha256_equal": score_npz_equal,
        "crop_video_sha256_equal": crop_equal,
        "tracks_pickle_sha256_equal": tracks_equal,
        "face_bbox_arrays_sha256_equal": box_equal,
        "generated_video_elementary_stream_sha256_equal": elementary_equal,
        "syncnet_query_distance_arrays_sha256_equal": score_arrays_equal,
    }
    exact = all(gate_values.values())
    return {"status": "PASS_EXACT_MATCH" if exact else "NONDETERMINISTIC_DIFFERENCE",
            **gate_values,
            "repeat_row_ids": [row["row_id"] for row in rows],
            "generated_video_file_sha256": [left["generated_video_sha256"], right["generated_video_sha256"]],
            "generated_video_elementary_stream_sha256": [left["generated_video_elementary_stream_sha256"],
                                                            right["generated_video_elementary_stream_sha256"]],
            "score_npz_sha256": [left["fresh_syncnet"]["distance_npz_sha256"],
                                 right["fresh_syncnet"]["distance_npz_sha256"]],
            "tracks_pickle_sha256": [a["tracks_pickle_sha256"], b["tracks_pickle_sha256"]],
            "crop_video_sha256": [[item["sha256"] for item in a["crop_videos"]],
                                  [item["sha256"] for item in b["crop_videos"]]],
            "face_bbox_arrays_sha256": [[item["bbox_sha256"] for item in a["numeric_tracks"]],
                                         [item["bbox_sha256"] for item in b["numeric_tracks"]]],
            "syncnet_distance_array_sha256": [[item["distance_by_query_lag_sha256"] for item in a_tracks],
                                               [item["distance_by_query_lag_sha256"] for item in b_tracks]]}


def input_context(model_id: str, source_sha: str, face_sha: str,
                  audio_ref: Mapping[str, Any]) -> dict[str, Any]:
    context = {
        "scope": "engineering_qc06_exposed_input_only",
        "sample_id": "natural_1", "source_file_sha256": source_sha,
        "face_file_sha256": face_sha,
        "canonical_audio_file_sha256": audio_ref["file_sha256"],
        "canonical_audio_pcm_sha256": audio_ref["pcm_sha256"],
        "sample_rate_hz": 16000, "channels": 1,
        "sample_count": int(audio_ref["sample_count"]),
        "seed": REPEAT_SEED, "model_id": model_id,
        "strict_bootstrap_sha256": STRICT_BOOTSTRAP_SHA256,
        "determinism_policy_sha256": QC06_POLICY_SHA256,
        "syncnet_checkpoint_sha256": SYNCNET_CHECKPOINT_SHA256,
    }
    return {"context": context, "sha256": sha_bytes(canonical(context))}


def qualification_receipt(model_id: str, gate: Mapping[str, Any], rows: list[dict[str, Any]],
                           context: Mapping[str, Any], resources: Mapping[str, Any],
                           dependency_template: Mapping[str, Any], backend_identity: Mapping[str, Any],
                           output_root: Path) -> tuple[dict[str, Any], bytes]:
    status = "PASS_EXACT_MATCH" if gate.get("status") == "PASS_EXACT_MATCH" else "NONDETERMINISTIC_DIFFERENCE"
    receipt: dict[str, Any] = {
        "schema": "cross_tfg_backend_strict_determinism_qualification_v1",
        "status": status, "engineering_only": True,
        "selection_eligible": False, "dev_or_confirm_media_read": False,
        "run_id": RUN_ID, "model_id": model_id, "gpu_uuid": GPU_UUID,
        "determinism_policy_sha256": QC06_POLICY_SHA256,
        "policy_values": STRICT_POLICY,
        "strict_bootstrap_source_sha256": STRICT_BOOTSTRAP_SHA256,
        "qc06_engineering_manifest_sha256": QC06_MANIFEST_SHA256,
        "input_context": context["context"], "input_context_sha256": context["sha256"],
        "backend_runtime_identity": backend_identity,
        "backend_dependency_template": dependency_template,
        "resource_identity_before": resources,
        "repeat_rows": rows,
        "repeat_gate": dict(gate),
    }
    receipt["receipt_sha256"] = payload_sha(receipt, "receipt_sha256")
    raw = json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2,
                     allow_nan=False).encode("utf-8") + b"\n"
    return receipt, raw


def execute(args: argparse.Namespace) -> int:
    if not args.execute_strict_repeats:
        raise RuntimeError("rendering requires --execute-strict-repeats; source/readiness invocation does not generate media")
    requested_values = tuple(args.backends or MODELS)
    requested_models = tuple(model_id for model_id in MODELS if model_id in set(requested_values))
    if not requested_models or len(requested_models) != len(set(requested_values)):
        raise ValueError("requested backend subset is empty or contains an unknown/duplicate model")
    output_root = args.output_root.resolve()
    if output_root.exists() and (output_root.is_symlink() or not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError(f"refusing nonempty/unsafe pilot output path: {output_root}")
    helper_path = ROOT / QC_HELPER_REL
    helper = load_module(helper_path, "qc06_existing_scoring_and_media_helpers")
    template, template_raw, template_ref, observed_deps = verify_dependency_template(requested_models)
    identity = verify_runtime_files(template, observed_deps, requested_models)
    if str(Path(sys.executable).resolve(strict=True)) != str(W2L_PYTHON.resolve(strict=True)):
        raise RuntimeError("pilot supervisor must run in the exact pinned Wav2Lip Python environment")
    w2l_python_identity = python_environment_identity(W2L_PYTHON)
    syncnet_python_identity = json.loads(run_probe([str(SYNCNET_PYTHON), "-c",
        "import json,sys,torch,numpy; print(json.dumps({'python':sys.executable,'torch':torch.__version__,'numpy':numpy.__version__},sort_keys=True))"]))
    initial_resources = resource_snapshot(output_root)
    initial_gpu = next((gpu for gpu in initial_resources["gpus"] if gpu["uuid"] == GPU_UUID), None)
    if initial_gpu is None or initial_gpu["memory_free_mib"] < MIN_GPU_FREE_MIB:
        raise RuntimeError(f"pinned GPU must have >= {MIN_GPU_FREE_MIB} MiB free before any repeat: {initial_resources['gpus']}")
    output_root.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "schema": "cross_tfg_backend_strict_repeat_pilot_manifest_v1",
        "run_id": RUN_ID, "status": "RUNNING", "engineering_only": True,
        "selection_eligible": False, "dev_or_confirm_media_read": False,
        "planned_rows": 6, "requested_render_rows": 2 * len(requested_models), "completed_rows": 0,
        "row_ids": [f"strictrepeat_{model_id}_{rep:02d}" for model_id in MODELS for rep in (1, 2)],
        "requested_row_ids": [f"strictrepeat_{model_id}_{rep:02d}" for model_id in requested_models for rep in (1, 2)],
        "models": list(MODELS), "requested_models": list(requested_models), "seed": REPEAT_SEED,
        "policy_values": STRICT_POLICY, "determinism_policy_sha256": QC06_POLICY_SHA256,
        "qc06_engineering_manifest_sha256": QC06_MANIFEST_SHA256,
        "dependency_template": template_ref,
        "input_paths": {"source": str(QC06_SOURCE), "face": str(QC06_FACE)},
        "runtime_identity": identity, "wav2lip_python_identity": w2l_python_identity,
        "syncnet_python_identity": syncnet_python_identity,
        "resource_before": initial_resources, "rows": [], "backend_qualifications": {},
        "failures": [], "started_unix": time.time(),
    }
    write_json(output_root / "pilot_manifest.json", result)
    try:
        source = QC06_SOURCE.resolve(strict=True)
        face = QC06_FACE.resolve(strict=True)
        x, wavfile = helper.canonicalize_with_wav2lip_loader(source, W2L_ROOT)
        natural_ref = helper.write_float_wav(output_root / "input/natural_1_official_load_f32.wav", x, wavfile)
        natural_ref["sample_count"] = int(x.size)
        natural_ref["pcm_sha256"] = natural_ref["pcm_sha256"]
        import numpy as np
        canonical_samples = np.ascontiguousarray(x, dtype="<f4")
        if helper.sha256_bytes(canonical_samples.tobytes()) != natural_ref["pcm_sha256"]:
            raise RuntimeError("Wav2Lip canonical readback PCM digest differs from QC06 input context")
        result["canonical_qc06_audio"] = natural_ref
        source_sha, face_sha = sha_file(source), sha_file(face)
        rows_by_model: dict[str, list[dict[str, Any]]] = {model_id: [] for model_id in requested_models}
        bootstrap_path = output_root / "deterministic_torch_bootstrap_attempt06.py"
        bootstrap_raw = helper.SYNCNET_BOOTSTRAP_SOURCE.encode("utf-8")
        bootstrap_path.write_bytes(bootstrap_raw)
        if sha_bytes(bootstrap_raw) != "bc89c871396d34543fb8f83b87b5bc51e23b9d4e76796b0fda0fbc8c60933da7":
            raise RuntimeError("actual QC06 SyncNet deterministic bootstrap source differs")

        parent_torch = None
        for model_id in requested_models:
            model_root = output_root / "models" / model_id
            model_root.mkdir(parents=True, exist_ok=False)
            for repetition in (1, 2):
                row_id = f"strictrepeat_{model_id}_{repetition:02d}"
                row_root = model_root / f"repeat_{repetition:02d}"
                row_root.mkdir(parents=True, exist_ok=False)
                try:
                    row = render_one(model_id, row_id, Path(natural_ref["path"]), face,
                                     row_root, helper)
                    if parent_torch is None:
                        parent_torch = configure_parent_score_torch_once()
                    row = mux_and_score(model_id, row, natural_ref, row_root,
                                        helper, bootstrap_path, parent_torch,
                                        syncnet_python_identity)
                    row["status"] = "PASS"
                except Exception as exc:
                    row = {"row_id": row_id, "model_id": model_id,
                           "seed": REPEAT_SEED, "status": "FAILED",
                           "failure_reason": f"{type(exc).__name__}: {exc}",
                           "traceback": traceback.format_exc()}
                    result["failures"].append({"row_id": row_id, "reason": row["failure_reason"]})
                rows_by_model[model_id].append(row)
                result["rows"].append(row)
                result["completed_rows"] = len(result["rows"])
                write_json(output_root / f"pilot_manifest_progress_{len(result['rows']):02d}.json", result)

            if all(row.get("status") == "PASS" for row in rows_by_model[model_id]):
                gate = compare_repeats(model_id, rows_by_model[model_id])
                context = input_context(model_id, source_sha, face_sha, natural_ref)
                qual_root = output_root / "backend_qualifications"
                qual_root.mkdir(exist_ok=True)
                receipt, raw = qualification_receipt(
                    model_id, gate, rows_by_model[model_id], context,
                    initial_resources, template_ref,
                    identity["backend_identity"][model_id], output_root)
                receipt_path = qual_root / f"{model_id}.qualification.json"
                receipt_path.write_bytes(raw)
                result["backend_qualifications"][model_id] = {
                    "schema": receipt["schema"], "status": receipt["status"],
                    "model_id": model_id, "gpu_uuid": GPU_UUID,
                    "determinism_policy_sha256": QC06_POLICY_SHA256,
                    "policy_values": STRICT_POLICY,
                    "input_context_sha256": context["sha256"],
                    "repeat_gate": gate,
                    "receipt": {"path": str(receipt_path.resolve()),
                                "bytes": len(raw), "sha256": sha_bytes(raw)},
                }
            else:
                result["backend_qualifications"][model_id] = {
                    "schema": "cross_tfg_backend_strict_determinism_qualification_v1",
                    "status": "FAILED_RENDER_OR_SCORE", "model_id": model_id,
                    "completed_repeats": sum(row.get("status") == "PASS" for row in rows_by_model[model_id]),
                }
            write_json(output_root / f"pilot_manifest_backend_{MODELS.index(model_id)+1:02d}.json", result)
        for model_id in MODELS:
            if model_id not in requested_models:
                result["backend_qualifications"][model_id] = {
                    "schema": "cross_tfg_backend_strict_determinism_qualification_v1",
                    "status": "NOT_RUN", "model_id": model_id,
                    "reason": "backend_not_selected_in_this_engineering_dispatch",
                }
        requested_pass = all(result["backend_qualifications"].get(model_id, {}).get("status") == "PASS_EXACT_MATCH"
                             for model_id in requested_models)
        full_pass = len(requested_models) == len(MODELS) and requested_pass
        result["status"] = ("PASS_ALL_THREE_BACKENDS_EXACT_REPEAT" if full_pass else
                            "INCOMPLETE_BACKEND_SUBSET_WITH_REQUESTED_BACKENDS_PASS" if requested_pass else
                            "FAILED_OR_INCOMPLETE")
        result["resource_after"] = resource_snapshot(output_root)
        result["ended_unix"] = time.time()
        result["manifest_sha256"] = payload_sha(result, "manifest_sha256")
        write_json(output_root / "pilot_manifest_final.json", result)
        print(json.dumps({"status": result["status"], "output_root": str(output_root),
                          "backend_statuses": {key: value.get("status") for key, value in result["backend_qualifications"].items()}},
                         sort_keys=True))
        return 0 if requested_pass else 2
    except Exception as exc:
        result["status"] = "FAILED_OR_INCOMPLETE"
        result["failures"].append({"error_type": type(exc).__name__, "error": str(exc),
                                   "traceback": traceback.format_exc()})
        result["ended_unix"] = time.time()
        try:
            write_json(output_root / "pilot_manifest_failure.json", result)
        except FileExistsError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", dest="backends", action="append", choices=MODELS,
                        help="backend to include; repeat for a subset. The overall receipt remains incomplete until all three have separate PASS receipts.")
    parser.add_argument("--execute-strict-repeats", action="store_true",
                        help="explicitly render and score the six engineering-only exact-repeat rows")
    args = parser.parse_args(argv)
    if not args.execute_strict_repeats:
        selected = tuple(model_id for model_id in MODELS if model_id in set(args.backends or MODELS))
        print(json.dumps({"status": "SOURCE_READY_NO_EXECUTION", "run_id": RUN_ID,
                          "output_root": str(args.output_root), "planned_rows": 6,
                          "requested_render_rows": 2 * len(selected), "models": MODELS,
                          "requested_models": selected,
                          "input_scope": "QC06 exposed natural_1/portrait only"}, sort_keys=True))
        return 0
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
