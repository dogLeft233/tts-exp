"""Pinned frozen Wav2Lip/SyncNet runtime for small historical fit-only runs.

The source-image and crop context is a fixed portrait proxy.  It is suitable
for checking a differentiable audio-to-mouth optimization path, not for
estimating native-video efficacy.  Candidate selection still requires fresh
official renders and development scoring.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .audio_io import read_canonical_wav
from .differentiable_adapter import Wav2LipSyncAdapter, Wav2LipSyncContext
from .heads import EnhancementStack, SpectralResidualHead
from .wav2lip_frontend import (
    Wav2LipTorchMel,
    check_frontend_parity,
    import_official_wav2lip_audio,
    official_chunk_starts,
    official_mel_basis,
)


WAV2LIP_GIT_HEAD = "bac9a81e63ecc153202353372e5724b83d9e6322"
SYNCNET_GIT_HEAD = "907c0b579c2e2d83f0eae1b2ac9e720cde4e5623"
EXPECTED_CONTEXT_SCHEMA = "cross_tfg_oldpilot_differentiable_context_v1"
SOFTMIN_TEMPERATURE = 0.1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _git_head(root: Path) -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.PIPE, text=True
    ).strip().lower()
    return value


def _verify_source_files(root: Path, pins: dict[str, str], role: str) -> dict[str, dict[str, Any]]:
    if not isinstance(pins, dict) or not pins:
        raise ValueError(f"{role} source SHA pins are missing")
    result: dict[str, dict[str, Any]] = {}
    for relative, expected in pins.items():
        path = (root / relative).resolve(strict=True)
        actual = sha256_file(path)
        if actual != str(expected).lower():
            raise ValueError(f"{role} source SHA mismatch: {relative} ({actual})")
        result[str(relative)] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual}
    return result


def _load_context_seed(runtime: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    config = runtime.get("context")
    if not isinstance(config, dict):
        raise ValueError("fit plan runtime.context must be an object")
    npz_path = Path(str(config["npz_path"])).resolve(strict=True)
    metadata_path = Path(str(config["metadata_path"])).resolve(strict=True)
    npz_sha = sha256_file(npz_path)
    metadata_sha = sha256_file(metadata_path)
    if npz_sha != str(config["npz_sha256"]).lower():
        raise ValueError("fixed portrait context NPZ differs from the frozen plan SHA")
    if metadata_sha != str(config["metadata_sha256"]).lower():
        raise ValueError("fixed portrait context metadata differs from the frozen plan SHA")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != EXPECTED_CONTEXT_SCHEMA or metadata.get("npz_file_sha256") != npz_sha:
        raise ValueError("fixed portrait context schema/NPZ binding mismatch")
    required = {
        "face_crops_bgr_u8": (np.dtype("uint8"), 4),
        "syncnet_crops_bgr_u8": (np.dtype("uint8"), 4),
        "patch_boxes_xyxy_i32": (np.dtype("int32"), 2),
        "frame_timestamps_seconds_f64": (np.dtype("float64"), 1),
        "syncnet_pcm16_crop_i16": (np.dtype("int16"), 1),
    }
    with np.load(npz_path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(required):
            raise ValueError("fixed portrait context NPZ key inventory mismatch")
        arrays = {name: np.ascontiguousarray(loaded[name]) for name in required}
    frame_count = int(metadata.get("frame_count", -1))
    expected_shapes = {
        "face_crops_bgr_u8": (frame_count, 3, 96, 96),
        "syncnet_crops_bgr_u8": (frame_count, 3, 224, 224),
        "patch_boxes_xyxy_i32": (frame_count, 4),
        "frame_timestamps_seconds_f64": (frame_count,),
    }
    actual_array_hashes: dict[str, str] = {}
    for name, (dtype, ndim) in required.items():
        array = arrays[name]
        if array.dtype != dtype or array.ndim != ndim:
            raise ValueError(f"fixed portrait context type/rank mismatch: {name}")
        if name in expected_shapes and array.shape != expected_shapes[name]:
            raise ValueError(f"fixed portrait context shape mismatch: {name}")
        if array.size == 0:
            raise ValueError(f"fixed portrait context array is empty: {name}")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"fixed portrait context has non-finite values: {name}")
        actual_array_hashes[name] = _array_sha(array)
    if metadata.get("array_sha256") != actual_array_hashes:
        raise ValueError("fixed portrait context array hashes differ from metadata")
    if frame_count < 6 or int(metadata.get("fps_num", 0)) != 25 or int(metadata.get("fps_den", 0)) != 1:
        raise ValueError("fixed portrait context must provide a 25 fps frame clock")
    # The context exporter freezes a single portrait crop for every source
    # frame.  Fail closed if that condition changed after export.
    face = arrays["face_crops_bgr_u8"]
    if not np.array_equal(face, np.broadcast_to(face[:1], face.shape)):
        raise ValueError("fit proxy requires the context's repeated fixed portrait face crop")
    return arrays, metadata, {
        "npz_path": str(npz_path),
        "npz_bytes": npz_path.stat().st_size,
        "npz_sha256": npz_sha,
        "metadata_path": str(metadata_path),
        "metadata_bytes": metadata_path.stat().st_size,
        "metadata_sha256": metadata_sha,
        "array_sha256": actual_array_hashes,
    }


def _load_models(runtime: dict[str, Any], device: Any):
    import torch

    w2l = runtime.get("wav2lip")
    sync = runtime.get("syncnet")
    if not isinstance(w2l, dict) or not isinstance(sync, dict):
        raise ValueError("fit plan must bind Wav2Lip and SyncNet runtime objects")
    w2l_root = Path(str(w2l["root"])).resolve(strict=True)
    w2l_checkpoint = Path(str(w2l["checkpoint"])).resolve(strict=True)
    sync_root = Path(str(sync["root"])).resolve(strict=True)
    sync_checkpoint = Path(str(sync["checkpoint"])).resolve(strict=True)
    if _git_head(w2l_root) != WAV2LIP_GIT_HEAD or w2l.get("git_head") != WAV2LIP_GIT_HEAD:
        raise ValueError("Wav2Lip Git HEAD differs from the frozen fit plan")
    if _git_head(sync_root) != SYNCNET_GIT_HEAD or sync.get("git_head") != SYNCNET_GIT_HEAD:
        raise ValueError("SyncNet Git HEAD differs from the frozen fit plan")
    w2l_sources = _verify_source_files(w2l_root, w2l.get("source_sha256"), "Wav2Lip")
    sync_sources = _verify_source_files(sync_root, sync.get("source_sha256"), "SyncNet")
    w2l_checkpoint_sha = sha256_file(w2l_checkpoint)
    sync_checkpoint_sha = sha256_file(sync_checkpoint)
    if w2l_checkpoint_sha != str(w2l.get("checkpoint_sha256", "")).lower():
        raise ValueError("Wav2Lip checkpoint SHA mismatch")
    if sync_checkpoint_sha != str(sync.get("checkpoint_sha256", "")).lower():
        raise ValueError("SyncNet checkpoint SHA mismatch")

    for path in (w2l_root, sync_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    psf_root = os.environ.get("CROSS_TFG_PSF_ROOT")
    if not psf_root:
        raise RuntimeError("fit runtime requires CROSS_TFG_PSF_ROOT to pin the pure-Python SyncNet MFCC dependency")
    resolved_psf_root = Path(psf_root).resolve(strict=True)
    if not resolved_psf_root.is_dir():
        raise RuntimeError("CROSS_TFG_PSF_ROOT must point to the pinned python_speech_features overlay root")
    if str(resolved_psf_root) not in sys.path:
        sys.path.insert(0, str(resolved_psf_root))
    audio_module, mel_spec, frontend_meta = import_official_wav2lip_audio(w2l_root)
    mel_basis = official_mel_basis(audio_module, mel_spec)
    mel_frontend = Wav2LipTorchMel(mel_basis, mel_spec).to(device)
    model_module = importlib.import_module("models.wav2lip")
    wav2lip_model = model_module.Wav2Lip()
    raw_checkpoint = torch.load(str(w2l_checkpoint), map_location="cpu", weights_only=False)
    state = raw_checkpoint.get("state_dict") if isinstance(raw_checkpoint, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError("Wav2Lip checkpoint has no non-empty state_dict")
    cleaned = {str(key).removeprefix("module."): value for key, value in state.items()}
    wav2lip_model.load_state_dict(cleaned, strict=True)
    wav2lip_model.to(device).eval()

    sync_module = importlib.import_module("SyncNetInstance")
    sync_instance = sync_module.SyncNetInstance(device=str(device))
    sync_instance.loadParameters(str(sync_checkpoint))
    syncnet_model = getattr(sync_instance, "__S__", None)
    if syncnet_model is None:
        raise RuntimeError("pinned SyncNet instance has no official SyncNet module")
    syncnet_model.to(device).eval()
    for model in (wav2lip_model, syncnet_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    try:
        from python_speech_features import mfcc
        import python_speech_features as psf_module
    except ImportError as exc:
        raise RuntimeError(
            "fit runtime needs python_speech_features; bind its pure-Python task overlay in CROSS_TFG_PSF_ROOT"
        ) from exc
    psf_file = Path(psf_module.__file__).resolve(strict=True)
    try:
        psf_file.relative_to(resolved_psf_root)
    except ValueError as exc:
        raise RuntimeError("python_speech_features was imported outside the pinned task-local overlay") from exc
    psf_package_root = psf_file.parent
    psf_source_files = []
    for path in sorted(psf_package_root.rglob("*.py")):
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(resolved_psf_root)
        except ValueError as exc:
            raise RuntimeError("python_speech_features source tree escaped its pinned overlay") from exc
        psf_source_files.append({"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)})
    if not psf_source_files:
        raise RuntimeError("python_speech_features overlay contains no Python source files")
    source_pins = {
        "wav2lip_git_head": WAV2LIP_GIT_HEAD,
        "wav2lip_sources": w2l_sources,
        "wav2lip_checkpoint": {"path": str(w2l_checkpoint), "bytes": w2l_checkpoint.stat().st_size, "sha256": w2l_checkpoint_sha},
        "syncnet_git_head": SYNCNET_GIT_HEAD,
        "syncnet_sources": sync_sources,
        "syncnet_checkpoint": {"path": str(sync_checkpoint), "bytes": sync_checkpoint.stat().st_size, "sha256": sync_checkpoint_sha},
        "official_frontend": frontend_meta,
        "mel_spec": mel_spec.__dict__,
        "python_speech_features": {
            "overlay_root": str(resolved_psf_root),
            "module_path": str(psf_file),
            "module_sha256": sha256_file(psf_file),
            "source_files": psf_source_files,
            "version": getattr(psf_module, "__version__", None),
        },
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "runtime_factory_path": str(Path(__file__).resolve()),
        "runtime_factory_sha256": sha256_file(Path(__file__).resolve()),
    }
    return wav2lip_model, syncnet_model, mel_frontend, audio_module, mfcc, source_pins


def create_runtime(*, plan: dict[str, Any], fit_rows: list[dict[str, Any]], device: str, seed: int) -> dict[str, Any]:
    """Factory consumed by ``scripts.audio_enhancement.fit_cli``."""

    import torch
    from torch.optim import Adam

    runtime_config = plan.get("runtime")
    if not isinstance(runtime_config, dict) or not runtime_config.get("source_context_is_training_proxy_only"):
        raise ValueError("fit runtime requires an explicitly proxy-only frozen runtime section")
    device_obj = torch.device(device)
    arrays, context_meta, context_pins = _load_context_seed(runtime_config)
    w2l_model, sync_model, mel_frontend, audio_module, mfcc, source_pins = _load_models(
        runtime_config, device_obj
    )
    first_row = fit_rows[0]
    first_audio, first_info = read_canonical_wav(Path(str(first_row["natural_pcm"]["path"])).resolve(strict=True))
    if first_info.file_sha256 != str(first_row["natural_pcm"]["file_sha256"]).lower():
        raise ValueError("fit frontend parity input differs from its frozen canonical WAV SHA")
    parity = check_frontend_parity(mel_frontend, audio_module, first_audio)
    adapter = Wav2LipSyncAdapter(
        wav2lip_model=w2l_model,
        syncnet_model=sync_model,
        mel_frontend=mel_frontend,
        frontend_parity=parity,
        frame_batch_size=4,
        syncnet_batch_size=16,
        psf_mfcc=mfcc,
    )

    base_face = arrays["face_crops_bgr_u8"][:1]
    base_sync_crop = arrays["syncnet_crops_bgr_u8"][:1]
    base_box = arrays["patch_boxes_xyxy_i32"][:1]
    context_cache: dict[str, Wav2LipSyncContext] = {}
    score_pcm16_pins: dict[str, dict[str, Any]] = {}
    source_pins["fit_score_pcm16_crops"] = score_pcm16_pins
    source_pins["fit_score_pcm16_policy"] = {
        "source": "direct PCM16 recovered from the frozen canonical fit-window WAV",
        "track_start_frame": 0,
        "sample_interval": "[0, frame_count * 640)",
        "reason": "SyncNet audio is cropped to the selected 25 fps track interval; the full fit waveform remains unchanged for Wav2Lip and sample-count checks",
        "codec_equivalence_claimed": False,
    }

    def context_for_row(row: dict[str, Any]) -> Wav2LipSyncContext:
        row_id = str(row["row_id"])
        if row_id in context_cache:
            return context_cache[row_id]
        natural = row["natural_pcm"]
        path = Path(str(natural["path"])).resolve(strict=True)
        audio, audio_info = read_canonical_wav(path)
        if (
            audio_info.file_sha256 != str(natural["file_sha256"]).lower()
            or audio_info.pcm_float32_sha256 != str(natural["pcm_sha256"]).lower()
            or audio_info.frames != int(natural["sample_count"])
        ):
            raise ValueError(f"fit proxy context audio binding mismatch: {row_id}")
        pcm16 = np.rint(
            np.clip(audio.astype(np.float64) * 32768.0, -32768.0, 32767.0)
        ).astype(np.int16)
        pcm16 = np.ascontiguousarray(pcm16)
        pcm16_sha = hashlib.sha256(pcm16.astype("<i2", copy=False).tobytes()).hexdigest()
        if natural.get("direct_source_pcm16_sha256") != pcm16_sha:
            raise ValueError(f"fit direct PCM16 interval hash mismatch: {row_id}")
        device_tensor = torch.from_numpy(audio).to(device=device_obj, dtype=torch.float32)
        with torch.no_grad():
            mel = mel_frontend(device_tensor)
        starts = official_chunk_starts(int(mel.shape[-1]), fps_num=25, fps_den=1)
        count = len(starts)
        if count < 6 or count > int(context_meta["frame_count"]):
            raise ValueError(f"fit window has unsupported official Wav2Lip chunk count: {row_id}: {count}")
        score_sample_count = count * 640
        if score_sample_count > pcm16.size:
            raise ValueError(f"fit PCM16 source does not cover the selected 25 fps context: {row_id}")
        score_pcm16 = np.ascontiguousarray(pcm16[:score_sample_count], dtype=np.int16)
        score_pcm16_sha = hashlib.sha256(score_pcm16.astype("<i2", copy=False).tobytes()).hexdigest()
        score_pcm16_pins[row_id] = {
            "source_pcm16_sha256": pcm16_sha,
            "score_pcm16_sha256": score_pcm16_sha,
            "score_sample_count": int(score_sample_count),
            "source_sample_count": int(pcm16.size),
            "track_start_frame": 0,
            "track_frame_count": int(count),
        }
        face_crops = np.ascontiguousarray(np.repeat(base_face, count, axis=0))
        sync_crops = np.ascontiguousarray(np.repeat(base_sync_crop, count, axis=0))
        boxes = np.ascontiguousarray(np.repeat(base_box, count, axis=0))
        timestamps = np.arange(count, dtype=np.float64) / 25.0
        context = Wav2LipSyncContext(
            face_crops_bgr=face_crops,
            syncnet_crops_bgr=sync_crops,
            patch_boxes_xyxy=boxes,
            mel_starts=starts,
            frame_timestamps_seconds=timestamps,
            fps_num=25,
            fps_den=1,
            track_start_frame=0,
            syncnet_pcm16_crop=score_pcm16,
            syncnet_pcm16_crop_sha256=score_pcm16_sha,
            expected_sample_count=int(audio.size),
            source_video_sha256=str(context_meta["source_video_sha256"]),
            face_crop_sha256=_array_sha(face_crops),
            syncnet_crop_sha256=_array_sha(sync_crops),
        )
        context.validate()
        context_cache[row_id] = context
        return context

    temperature = float(plan.get("fit_config", {}).get("proxy_loss", {}).get("temperature", SOFTMIN_TEMPERATURE))
    if not math.isclose(temperature, SOFTMIN_TEMPERATURE, rel_tol=0.0, abs_tol=0.0):
        raise ValueError("the initial fit loss temperature is frozen at 0.1")

    def median_minus_softmin(
        generated_video: Any,
        natural_audio: Any,
        *,
        context: Wav2LipSyncContext,
        anchor_lag_frames: int = 0,
        vshift: int = 15,
    ):
        del anchor_lag_frames
        curve = adapter.sync_distance_curve(generated_video, natural_audio, context=context, vshift=vshift)
        # Normalized log-mean-exp softmin; unlike a fixed-lag anchor it does
        # not assume one offset is best across the historical fit utterances.
        softmin = -temperature * (
            torch.logsumexp(-curve / temperature, dim=0) - math.log(curve.numel())
        )
        return curve.median() - softmin

    head = SpectralResidualHead(
        n_fft=512,
        hop_length=128,
        hidden_channels=24,
        max_residual_db=3.0,
        level_normalize=True,
    )
    stack = EnhancementStack(head).to(device_obj)
    fit_config = plan.get("fit_config")
    if not isinstance(fit_config, dict) or fit_config.get("optimizer") != "Adam":
        raise ValueError("fit_config.optimizer must be the frozen 'Adam' optimizer")
    learning_rate_value = fit_config.get("learning_rate")
    if isinstance(learning_rate_value, bool):
        raise ValueError("fit_config.learning_rate must be finite and positive")
    try:
        learning_rate = float(learning_rate_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("fit_config.learning_rate must be finite and positive") from exc
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("fit_config.learning_rate must be finite and positive")
    optimizer = Adam(stack.parameters(), lr=learning_rate)
    source_pins["portrait_context"] = context_pins
    source_pins["fit_runtime_loss"] = {
        "name": "median_lag_distance_minus_normalized_softmin",
        "temperature": temperature,
        "lag_count": 31,
        "formula": "median(D31) - (-tau*(logsumexp(-D31/tau)-log(31)))",
        "anchor_lag_frames_used": False,
    }
    source_pins["proxy_context_mode"] = runtime_config["context"]["training_context"]
    source_pins["seed"] = int(seed)
    return {
        "enhancement_stack": stack,
        "optimizer": optimizer,
        "adapter": adapter,
        "sync_score": median_minus_softmin,
        "score_config": {
            "score_proxy": "syncnet_31lag_median_minus_normalized_softmin_v1",
            "softmin_temperature": temperature,
            "lag_count": 31,
            "anchor_lag_frames_used": False,
        },
        "context_for_row": context_for_row,
        "source_pins": source_pins,
    }


__all__ = ["create_runtime"]
