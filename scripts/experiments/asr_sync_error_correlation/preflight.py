"""Environment checks and immutable local Wav2Vec2 deployment."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .asr_ctc import encode_reference, forced_align_reference, infer_waveform
from .config import ASR_MODEL_ID, SYNCNET_CHECKPOINT_SHA256, SYNCNET_ENV_PYTHON, SYNCNET_DIR
from .io import atomic_write_json, canonical_hash, file_sha256, read_json


def _probe(python: Path) -> dict[str, Any]:
    code = """import importlib, json, sys
mods = ['torch', 'torchaudio', 'transformers', 'huggingface_hub', 'numpy', 'scipy', 'soundfile', 'cv2', 'python_speech_features']
out = {'python': sys.executable, 'versions': {}}
for name in mods:
    try:
        mod = importlib.import_module(name)
        out['versions'][name] = getattr(mod, '__version__', 'unknown')
    except Exception as exc:
        out['versions'][name] = f'MISSING: {type(exc).__name__}: {exc}'
try:
    import torch
    out['cuda_available'] = bool(torch.cuda.is_available())
    out['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as exc:
    out['cuda_error'] = f'{type(exc).__name__}: {exc}'
print(json.dumps(out, sort_keys=True))
"""
    result = subprocess.run([str(python), "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        return {"python": str(python), "error": result.stderr[-500:]}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"python": str(python), "error": "dependency probe returned invalid JSON"}


def select_device(requested: str) -> tuple[str, dict[str, Any]]:
    import torch

    cuda = bool(torch.cuda.is_available())
    if requested == "cuda" and not cuda:
        raise RuntimeError("explicit CUDA requested but CUDA is unavailable")
    selected = "cuda" if requested == "auto" and cuda else requested
    if selected == "auto":
        selected = "cpu"
    gpu = torch.cuda.get_device_name(0) if cuda else None
    return selected, {
        "requested": requested,
        "selected": selected,
        "cuda_available": cuda,
        "gpu_name": gpu,
        "torch_version": getattr(torch, "__version__", "unknown"),
    }


def _matplotlib_smoke() -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=(1, 1))
    figure.canvas.draw()
    plt.close(figure)
    return {"version": matplotlib.__version__, "backend": str(matplotlib.get_backend())}


def check_environment(
    repo_root: Path,
    *,
    requested_device: str = "auto",
    asr_cache: Path | None = None,
    syncnet_python: Path | None = None,
    lock_path: Path | None = None,
    min_free_bytes: int = 2 * 1024**3,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    current_python = Path(sys.executable)
    sync_python = syncnet_python or (repo_root / SYNCNET_ENV_PYTHON)
    checkpoint = repo_root / SYNCNET_DIR / "data" / "syncnet_v2.model"
    selected_device, device_info = select_device(requested_device)
    usage = shutil.disk_usage(repo_root)
    checks: dict[str, bool] = {
        "main_python": current_python.is_file(),
        "syncnet_python": sync_python.is_file(),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "disk_space": usage.free >= min_free_bytes,
        "syncnet_checkpoint": checkpoint.is_file() and file_sha256(checkpoint) == SYNCNET_CHECKPOINT_SHA256,
    }
    main_probe = _probe(current_python) if checks["main_python"] else {"error": "main Python not found"}
    sync_probe = _probe(sync_python) if checks["syncnet_python"] else {"error": "SyncNet Python not found"}
    required_main = ("torch", "torchaudio", "transformers", "huggingface_hub", "numpy", "scipy", "soundfile")
    for package in required_main:
        value = str(main_probe.get("versions", {}).get(package, "MISSING"))
        checks[f"main_package_{package}"] = not value.startswith("MISSING")
    for package in ("torch", "numpy", "scipy", "cv2", "python_speech_features"):
        value = str(sync_probe.get("versions", {}).get(package, "MISSING"))
        checks[f"syncnet_package_{package}"] = not value.startswith("MISSING")
    try:
        plot_info = _matplotlib_smoke()
        checks["matplotlib_agg"] = True
    except Exception as exc:
        plot_info = {"error": f"{type(exc).__name__}: {exc}"}
        checks["matplotlib_agg"] = False
    if selected_device == "cuda":
        checks["cuda_smoke"] = bool(device_info["cuda_available"])
    cache = (asr_cache or repo_root / "checkpoints" / "huggingface").resolve()
    lock_path = (lock_path or repo_root / "00_preflight" / "asr_model_lock.json").resolve()
    try:
        lock = read_json(lock_path) if lock_path.is_file() else None
        checks["asr_model_lock"] = isinstance(lock, Mapping) and validate_model_lock(lock, cache)
    except (OSError, ValueError, TypeError):
        checks["asr_model_lock"] = False
    remediation = next((remediation_command(key) for key, value in checks.items() if not value), None)
    return {
        "status": "GO" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "selected_device": selected_device,
        "device": device_info,
        "main_python": main_probe,
        "syncnet_python": sync_probe,
        "matplotlib": plot_info,
        "disk_free_bytes": int(usage.free),
        "syncnet_checkpoint": str(checkpoint),
        "asr_cache": str(cache),
        "asr_model_lock": str(lock_path),
        "remediation": remediation,
    }


def remediation_command(check_name: str) -> str:
    if check_name.startswith("main_package_") or check_name == "matplotlib_agg":
        return "source /home/wjj/tts-audio/tts-exp/.venv/bin/activate && python -m pip install -r requirements-asr-sync-error-correlation.txt"
    if check_name == "syncnet_python" or check_name.startswith("syncnet_package_"):
        return "source ~/.venvs/syncnet/bin/activate && python -m pip install -r third_party/syncnet_python/requirements.txt"
    if check_name in ("ffmpeg", "ffprobe"):
        return "which ffmpeg ffprobe"
    if check_name == "syncnet_checkpoint":
        return "sha256sum third_party/syncnet_python/data/syncnet_v2.model"
    if check_name == "asr_model_lock":
        return "HF_ENDPOINT=https://hf-mirror.com python -m scripts.experiments.asr_sync_error_correlation.run --stage preflight --acquire-asr-model"
    if check_name == "disk_space":
        return "df -h ."
    if check_name == "cuda_smoke":
        return "python -c 'import torch; print(torch.cuda.is_available())'"
    return "python -m scripts.experiments.asr_sync_error_correlation.run --stage preflight"


def _snapshot_files(snapshot: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        relative = path.relative_to(snapshot).as_posix()
        files.append({"path": relative, "size": path.stat().st_size, "sha256": file_sha256(path)})
    return files


def acquire_asr_snapshot(
    *,
    cache_dir: Path,
    lock_path: Path,
    model_id: str = ASR_MODEL_ID,
) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(model_id, revision="main")
    revision = str(info.sha)
    if not revision or revision == "main":
        raise RuntimeError("Hugging Face did not return an immutable model revision")
    snapshot = Path(snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=str(cache_dir),
        local_files_only=False,
        allow_patterns=[
            "config.json", "feature_extractor_config.json", "preprocessor_config.json",
            "special_tokens_map.json", "tokenizer_config.json", "vocab.json", "pytorch_model.bin",
        ],
    )).resolve()
    files = _snapshot_files(snapshot)
    names = {row["path"] for row in files}
    if "config.json" not in names or not any(name.endswith((".bin", ".safetensors")) for name in names):
        raise RuntimeError("downloaded model snapshot is partial: config or weights missing")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    vocabulary_hash = canonical_hash(tokenizer.get_vocab())
    lock = {
        "schema_version": 1,
        "model_id": model_id,
        "revision": revision,
        "snapshot_path": str(snapshot),
        "files": files,
        "snapshot_hash": canonical_hash(files),
        "tokenizer_vocabulary_hash": vocabulary_hash,
        "acquisition": "huggingface_hub.snapshot_download",
    }
    atomic_write_json(lock_path, lock)
    if not validate_model_lock(lock, cache_dir):
        raise RuntimeError("new model lock failed immediate validation")
    return lock


def validate_model_lock(lock: Mapping[str, Any], cache_dir: Path | None = None) -> bool:
    if lock.get("model_id") != ASR_MODEL_ID:
        return False
    revision = str(lock.get("revision", ""))
    if not revision or revision == "main" or len(revision) < 20:
        return False
    snapshot = Path(str(lock.get("snapshot_path", "")))
    if not snapshot.is_dir() and cache_dir is not None:
        snapshot = Path(cache_dir) / snapshot
    files = lock.get("files")
    if not isinstance(files, list) or not files or lock.get("snapshot_hash") != canonical_hash(files) or not isinstance(lock.get("tokenizer_vocabulary_hash"), str):
        return False
    for row in files:
        if not isinstance(row, Mapping) or not row.get("path") or not row.get("sha256"):
            return False
        path = snapshot / str(row["path"])
        if not path.is_file() or path.stat().st_size != int(row["size"]) or file_sha256(path) != row["sha256"]:
            return False
    return True


def load_locked_processor_model(lock_path: Path, *, device: str) -> tuple[Any, Any, dict[str, Any]]:
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    lock = read_json(lock_path)
    snapshot = Path(str(lock["snapshot_path"]))
    if not validate_model_lock(lock):
        raise ValueError("ASR model lock is invalid or snapshot changed")
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(snapshot), local_files_only=True, torch_dtype="auto")
    model = model.to(device=device, dtype=__import__("torch").float32)
    model.eval()
    vocab_hash = canonical_hash(processor.tokenizer.get_vocab() if hasattr(processor, "tokenizer") else processor.get_vocab())
    if vocab_hash != lock.get("tokenizer_vocabulary_hash"):
        raise ValueError("tokenizer vocabulary hash differs from model lock")
    return processor, model, {
        "model_id": lock["model_id"],
        "revision": lock["revision"],
        "snapshot_path": str(snapshot),
        "snapshot_hash": lock["snapshot_hash"],
        "tokenizer_vocabulary_hash": vocab_hash,
        "device": device,
        "dtype": "float32",
        "local_files_only": True,
    }


def smoke_test_locked_model(processor: Any, model: Any, *, device: str, transcript: str | None = None) -> dict[str, Any]:
    import torch

    waveform = __import__("numpy").zeros(16000, dtype="float32")
    log_probs, metadata = infer_waveform(waveform, processor, model, device=device)
    ratio = getattr(model.config, "inputs_to_logits_ratio", None)
    if ratio is None or float(ratio) <= 0:
        raise ValueError("model has no positive inputs_to_logits_ratio")
    blank_id = int(getattr(model.config, "pad_token_id", 0))
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    forced = None
    if transcript:
        _, _, target_ids, delimiter_id = encode_reference(tokenizer, transcript, blank_id=blank_id)
        forced_sample = forced_align_reference(
            log_probs, tokenizer, transcript.split()[0], blank_id=blank_id,
            frame_stride_s=float(ratio) / 16000.0, audio_duration_s=1.0,
        )
        forced = {"target_token_count": int(target_ids.size), "delimiter_id": delimiter_id, "forced_alignment_word_count": len(forced_sample["words"])}
    return {**metadata, "frame_stride_s": float(ratio) / 16000.0, "forced_alignment": forced, "torch_version": torch.__version__}


def write_preflight(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(payload))
