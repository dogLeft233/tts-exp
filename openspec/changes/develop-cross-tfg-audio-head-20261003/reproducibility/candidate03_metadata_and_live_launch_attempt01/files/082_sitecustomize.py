"""Enforce and record the strict deterministic Torch policy before renderer imports.

This module is loaded by Python's normal site initialization when the pinned
directory is present on PYTHONPATH. It is not a qualification result by itself:
the guarded runner must verify one matching renderer-entry receipt for every
planned row, and a separate engineering repeat must still pass per backend.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path


POLICY = {
    "pythonhashseed": "20261003",
    "cublas_workspace_config": ":4096:8",
    "omp_num_threads": "1",
    "mkl_num_threads": "1",
    "openblas_num_threads": "1",
    "numexpr_num_threads": "1",
    "torch_num_threads": 1,
    "torch_num_interop_threads": 1,
    "torch_use_deterministic_algorithms": True,
    "torch_cudnn_deterministic": True,
    "torch_cudnn_benchmark": False,
    "torch_allow_tf32": False,
}
ENV_EXPECTED = {
    "PYTHONHASHSEED": POLICY["pythonhashseed"],
    "CUBLAS_WORKSPACE_CONFIG": POLICY["cublas_workspace_config"],
    "OMP_NUM_THREADS": POLICY["omp_num_threads"],
    "MKL_NUM_THREADS": POLICY["mkl_num_threads"],
    "OPENBLAS_NUM_THREADS": POLICY["openblas_num_threads"],
    "NUMEXPR_NUM_THREADS": POLICY["numexpr_num_threads"],
}
MODEL_ENTRY_PATHS = {
    "musetalk_fixed_portrait_attempt16": "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt16/code/render_silent_video_attempt09.py",
    "ditto_hubert_fixed_portrait_attempt18": "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt18/code/render_silent_video_attempt07.py",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _install_wav2lip_librosa_positional_compatibility() -> dict[str, str]:
    """Match the frozen frontend shim for Wav2Lip's historical positional call."""
    import inspect
    import librosa

    original = librosa.filters.mel
    original_signature = str(inspect.signature(original))
    if not getattr(original, "_cross_tfg_positional_compat", False):
        def compatible_mel(*args, **kwargs):
            positional = list(args)
            if positional and "sr" not in kwargs:
                kwargs["sr"] = positional.pop(0)
            if positional and "n_fft" not in kwargs:
                kwargs["n_fft"] = positional.pop(0)
            if positional:
                raise TypeError("unexpected positional arguments in librosa.filters.mel compatibility shim")
            return original(**kwargs)

        setattr(compatible_mel, "_cross_tfg_positional_compat", True)
        librosa.filters.mel = compatible_mel
    return {
        "id": "wav2lip_librosa_filters_mel_positional_sr_n_fft_to_keywords_v1",
        "librosa_version": str(getattr(librosa, "__version__", "unknown")),
        "original_signature": original_signature,
        "shim": "positional sr,n_fft mapped to keyword arguments; remaining defaults unchanged",
        "installed_before_renderer_import": True,
    }


def _apply_before_backend_imports() -> None:
    bad = {key: {"expected": value, "actual": os.environ.get(key)}
           for key, value in ENV_EXPECTED.items() if os.environ.get(key) != value}
    if bad:
        raise RuntimeError(f"strict deterministic environment is incomplete: {bad}")
    def option_value(name: str) -> str | None:
        try:
            index = sys.argv.index(name)
        except ValueError:
            return None
        return sys.argv[index + 1] if index + 1 < len(sys.argv) else None

    seed_text = (os.environ.get("CROSS_TFG_RENDER_SEED") or option_value("--seed")
                 or os.environ.get("TFG_DETERMINISM_SEED") or "20261003")
    if not seed_text.isdigit():
        raise RuntimeError("TFG_DETERMINISM_SEED must be a nonnegative decimal integer")
    seed = int(seed_text)

    row_id = os.environ.get("CROSS_TFG_ROW_ID")
    model_id = os.environ.get("CROSS_TFG_MODEL_ID")
    if not row_id and "--child" in sys.argv and option_value("--wav2lip-root-child"):
        output_text = option_value("--outfile")
        if output_text:
            output_path = Path(output_text)
            if len(output_path.parents) >= 2 and output_path.parent.name == "render":
                row_id = output_path.parent.parent.name
        model_id = model_id or "wav2lip_gan_bac9a81e"
    # Stage-A parent/evaluator Python processes inherit PYTHONPATH. Only renderer
    # and row-scoped shim processes apply the strict backend policy or write receipts.
    startup_policy = os.environ.get("CROSS_TFG_STRICT_STARTUP_POLICY") == "1"
    if (not row_id and not startup_policy) or (row_id and model_id not in {"wav2lip_gan_bac9a81e", *MODEL_ENTRY_PATHS}):
        return

    import numpy as np
    import torch

    torch.set_num_threads(POLICY["torch_num_threads"])
    torch.set_num_interop_threads(POLICY["torch_num_interop_threads"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = POLICY["torch_cudnn_deterministic"]
    torch.backends.cudnn.benchmark = POLICY["torch_cudnn_benchmark"]
    torch.backends.cudnn.allow_tf32 = POLICY["torch_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = POLICY["torch_allow_tf32"]
    torch.use_deterministic_algorithms(POLICY["torch_use_deterministic_algorithms"], warn_only=False)

    observed = {
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_cudnn_deterministic": torch.backends.cudnn.deterministic,
        "torch_cudnn_benchmark": torch.backends.cudnn.benchmark,
        "torch_allow_tf32": bool(torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32),
    }
    if observed != POLICY:
        raise RuntimeError(f"observed deterministic Torch/runtime policy differs: {observed!r}")
    # The candidate audio transform runs in one process before the per-row renderer
    # subprocesses. Its global strict-policy setup is required, but it is not a row receipt.
    if not row_id:
        return

    renderer_compatibility = (
        _install_wav2lip_librosa_positional_compatibility()
        if model_id == "wav2lip_gan_bac9a81e" else None
    )
    receipt_dir_text = os.environ.get("CROSS_TFG_DETERMINISM_RECEIPT_DIR")
    if not receipt_dir_text:
        raise RuntimeError("row-scoped renderer process lacks deterministic receipt directory")
    receipt_dir = Path(receipt_dir_text)
    if not receipt_dir.is_absolute() or receipt_dir.is_symlink() or not receipt_dir.is_dir():
        raise RuntimeError("deterministic receipt directory must be an existing absolute real directory")
    entry = Path(sys.argv[0])
    try:
        entry_path = str(entry.resolve(strict=True))
    except OSError:
        entry_path = str(entry.absolute())
    expected_entry = os.environ.get("CROSS_TFG_BACKEND_ENTRY")
    if not expected_entry and model_id == "wav2lip_gan_bac9a81e" and "--child" in sys.argv:
        expected_entry = entry_path
    if not expected_entry and model_id in MODEL_ENTRY_PATHS:
        expected_entry = MODEL_ENTRY_PATHS[model_id]
    record = {
        "schema": "cross_tfg_renderer_strict_determinism_bootstrap_receipt_v1",
        "status": "PASS",
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "row_id": row_id,
        "model_id": model_id,
        "audio_frontend": os.environ.get("CROSS_TFG_AUDIO_FRONTEND") or {
            "wav2lip_gan_bac9a81e": "w2l",
            "musetalk_fixed_portrait_attempt16": "muse",
            "ditto_hubert_fixed_portrait_attempt18": "ditto",
        }.get(model_id),
        "seed": seed,
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "entry_path": entry_path,
        "expected_backend_entry": expected_entry,
        "entry_matches_backend": bool(expected_entry and entry_path == str(Path(expected_entry).resolve(strict=True))),
        "bootstrap_source_sha256": _sha(Path(__file__).read_bytes()),
        "policy": observed,
        "policy_payload_sha256": _sha(_canonical(observed)),
        "renderer_compatibility": renderer_compatibility,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
    }
    record["receipt_sha256"] = _sha(_canonical(record))
    path = receipt_dir / f"{os.getpid()}.json"
    raw = (json.dumps(record, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


_apply_before_backend_imports()
