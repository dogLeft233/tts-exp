#!/usr/bin/env python3
"""One-step, engineering-only gradient smoke for the shared audio head.

This loads the pinned Wav2Lip and SyncNet models, verifies the Torch mel
frontend against Wav2Lip's official NumPy frontend, and performs one optimizer
step on the historically exposed oldpilot.  It does not read dev/confirmation
scores and its differentiable image/crop path is explicitly a proxy for the
official AVI/JPEG pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from scripts.audio_enhancement.audio_io import read_canonical_wav
from scripts.audio_enhancement.differentiable_adapter import (
    Wav2LipSyncAdapter,
    Wav2LipSyncContext,
)
from scripts.audio_enhancement.heads import EnhancementStack, SpectralResidualHead
from scripts.audio_enhancement.training import (
    freeze_model_parameters,
    sync_guided_train_step,
)
from scripts.audio_enhancement.wav2lip_frontend import (
    Wav2LipTorchMel,
    check_frontend_parity,
    import_official_wav2lip_audio,
    official_chunk_starts,
    official_mel_basis,
)


WAV2LIP_GIT_HEAD = "bac9a81e63ecc153202353372e5724b83d9e6322"
SYNCNET_GIT_HEAD = "907c0b579c2e2d83f0eae1b2ac9e720cde4e5623"
WAV2LIP_SOURCE_SHA256 = {
    "audio.py": "71d5e927cac3d0b7103e7671a43b82b41b6a56398a092bbfa2e96d3041249c4c",
    "hparams.py": "4a4c31dc51ebd8140da18ac24841e582b2472389f7c65e7f0a6b4447fc0c71cf",
    "inference.py": "99662bfce2fdaf17c1cff8e71977ba02cb80c684058cd114c4e0648b12059fc6",
    "models/wav2lip.py": "a02b54288f8200332ee4f0ea0c8013bca542b94d8149e514ac25a742a7a70e66",
}
SYNCNET_SOURCE_SHA256 = {
    "SyncNetInstance.py": "bd722bf844700d6a923dec2f10d7cacd65c0c6b22d09aa5d44d2509660e900d6",
    "SyncNetModel.py": "41b85aed0006546cb2d6dc9180c5bc3dac737ab80eff643022e397a349c30c22",
    "run_pipeline.py": "3654a3cf22c74cb3071fca6fb97ca9fb7dd1a5943fe864169ff8c0b57eb80acf",
    "run_syncnet.py": "c78fe9f75665aa9e7a2bb523e4e7e3734a857137ff70b29c88a7b4893c422ded",
}
OLDPILOT_ORIGINAL_WAV_SHA256 = "263b19510c95c32db4874449aadefc9b37c008e4816c18028eaf8fef463828e5"
OLDPILOT_PRIMARY_FACE_SHA256 = "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6"
RMS_TARGET = 0.0376838172675746
ENGINEERING_ANCHOR_LAG = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return sha256_bytes(encoded)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write((json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def array_sha256(array: np.ndarray, dtype: str) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    return sha256_bytes(contiguous.tobytes())


def source_tree_hashes(root: Path, expected: dict[str, str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative, expected_sha in expected.items():
        path = (root / relative).resolve(strict=True)
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(f"pinned source hash mismatch: {relative} {actual_sha}")
        result[relative] = {"path": str(path), "bytes": path.stat().st_size, "sha256": actual_sha}
    return result


def git_head(root: Path) -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.PIPE, text=True
    ).strip().lower()
    return value


def _load_context(npz_path: Path, metadata_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, str]]:
    npz_raw_sha = sha256_file(npz_path)
    metadata_raw = metadata_path.read_bytes()
    metadata = json.loads(metadata_raw.decode("utf-8"))
    if metadata.get("schema") != "cross_tfg_oldpilot_differentiable_context_v1":
        raise ValueError("unexpected oldpilot context schema")
    declared_npz_sha = str(metadata.get("npz_file_sha256", "")).lower()
    if declared_npz_sha != npz_raw_sha:
        raise ValueError("context metadata does not bind the supplied NPZ file")
    required = {
        "face_crops_bgr_u8": (np.dtype("uint8"), 4),
        "syncnet_crops_bgr_u8": (np.dtype("uint8"), 4),
        "patch_boxes_xyxy_i32": (np.dtype("int32"), 2),
        "frame_timestamps_seconds_f64": (np.dtype("float64"), 1),
        "syncnet_pcm16_crop_i16": (np.dtype("int16"), 1),
    }
    with np.load(npz_path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(required):
            raise ValueError(f"context NPZ key inventory mismatch: {sorted(loaded.files)}")
        arrays = {name: np.ascontiguousarray(loaded[name]) for name in required}
    for name, (dtype, ndim) in required.items():
        array = arrays[name]
        if array.dtype != dtype or array.ndim != ndim:
            raise ValueError(f"context array dtype/rank mismatch for {name}")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"context array contains non-finite values: {name}")
    frame_count = int(metadata.get("frame_count", -1))
    if arrays["face_crops_bgr_u8"].shape != (frame_count, 3, 96, 96):
        raise ValueError("Wav2Lip face crop array shape differs from context metadata")
    if arrays["syncnet_crops_bgr_u8"].shape != (frame_count, 3, 224, 224):
        raise ValueError("SyncNet crop array shape differs from context metadata")
    if arrays["patch_boxes_xyxy_i32"].shape != (frame_count, 4):
        raise ValueError("patch box array shape differs from context metadata")
    if arrays["frame_timestamps_seconds_f64"].shape != (frame_count,):
        raise ValueError("timestamp array shape differs from context metadata")
    if frame_count <= 5 or int(metadata.get("fps_num", 0)) != 25 or int(metadata.get("fps_den", 0)) != 1:
        raise ValueError("oldpilot context must be a 25 fps multi-frame track")
    if arrays["syncnet_pcm16_crop_i16"].size != int(metadata.get("track_pcm16_sample_count", -1)):
        raise ValueError("exact SyncNet PCM16 crop sample count differs from metadata")
    hashes = {name: array_sha256(arrays[name], arrays[name].dtype.str) for name in arrays}
    declared_hashes = metadata.get("array_sha256")
    if not isinstance(declared_hashes, dict) or declared_hashes != hashes:
        raise ValueError("context array hashes do not match exact loaded arrays")
    return arrays, metadata, {"npz_raw_sha256": npz_raw_sha, "metadata_raw_sha256": sha256_bytes(metadata_raw)}


def _load_models_and_frontend(
    *, wav2lip_root: Path, wav2lip_checkpoint: Path,
    syncnet_root: Path, syncnet_checkpoint: Path, device: Any,
) -> tuple[Any, Any, Any, Any, Any, dict[str, Any], dict[str, Any]]:
    import torch

    w2l_head = git_head(wav2lip_root)
    sync_head = git_head(syncnet_root)
    if w2l_head != WAV2LIP_GIT_HEAD:
        raise ValueError(f"Wav2Lip git HEAD mismatch: {w2l_head}")
    if sync_head != SYNCNET_GIT_HEAD:
        raise ValueError(f"SyncNet git HEAD mismatch: {sync_head}")
    wav2lip_sources = source_tree_hashes(wav2lip_root, WAV2LIP_SOURCE_SHA256)
    syncnet_sources = source_tree_hashes(syncnet_root, SYNCNET_SOURCE_SHA256)

    # Import the exact repository modules only after the model source pins pass.
    if str(wav2lip_root) not in sys.path:
        sys.path.insert(0, str(wav2lip_root))
    audio_module, mel_spec, frontend_meta = import_official_wav2lip_audio(wav2lip_root)
    basis = official_mel_basis(audio_module, mel_spec)
    mel_frontend = Wav2LipTorchMel(basis, mel_spec).to(device)
    models_module = importlib.import_module("models.wav2lip")
    wav2lip_model = models_module.Wav2Lip()
    w2l_checkpoint = torch.load(str(wav2lip_checkpoint), map_location="cpu", weights_only=False)
    state = w2l_checkpoint.get("state_dict") if isinstance(w2l_checkpoint, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError("Wav2Lip checkpoint has no non-empty state_dict")
    cleaned = {key.removeprefix("module."): value for key, value in state.items()}
    wav2lip_model.load_state_dict(cleaned, strict=True)
    wav2lip_model.to(device).eval()

    if str(syncnet_root) not in sys.path:
        sys.path.insert(0, str(syncnet_root))
    syncnet_instance_module = importlib.import_module("SyncNetInstance")
    syncnet_instance = syncnet_instance_module.SyncNetInstance(device=str(device))
    syncnet_instance.loadParameters(str(syncnet_checkpoint))
    syncnet_model = getattr(syncnet_instance, "__S__", None)
    if syncnet_model is None:
        raise RuntimeError("pinned SyncNetInstance did not expose its official SyncNet module")
    syncnet_model.to(device).eval()
    for model in (wav2lip_model, syncnet_model):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    try:
        from python_speech_features import mfcc
    except ImportError as exc:
        raise RuntimeError("the Wav2Lip training environment needs python_speech_features for official SyncNet MFCC") from exc
    source_pins = {
        "wav2lip_git_head": w2l_head,
        "syncnet_git_head": sync_head,
        "wav2lip_sources": wav2lip_sources,
        "syncnet_sources": syncnet_sources,
        "wav2lip_checkpoint": {"path": str(wav2lip_checkpoint.resolve()), "bytes": wav2lip_checkpoint.stat().st_size, "sha256": sha256_file(wav2lip_checkpoint)},
        "syncnet_checkpoint": {"path": str(syncnet_checkpoint.resolve()), "bytes": syncnet_checkpoint.stat().st_size, "sha256": sha256_file(syncnet_checkpoint)},
        "official_frontend": frontend_meta,
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
    }
    return wav2lip_model, syncnet_model, mel_frontend, audio_module, mfcc, source_pins, {"wav2lip": w2l_checkpoint}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse an existing engineering output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "run_status.json"
    status = {
        "schema": "cross_tfg_one_step_gradient_smoke_status_v1",
        "status": "RUNNING",
        "engineering_only": True,
        "historically_exposed": True,
        "candidate_selection_authorized": False,
        "development_scores_read": False,
        "confirmation_scores_read": False,
        "optimizer_steps": 0,
    }
    atomic_json(status_path, status)
    try:
        wav2lip_root = args.wav2lip_root.resolve(strict=True)
        syncnet_root = args.syncnet_root.resolve(strict=True)
        wav2lip_checkpoint = args.wav2lip_checkpoint.resolve(strict=True)
        syncnet_checkpoint = args.syncnet_checkpoint.resolve(strict=True)
        original_source = args.original_source_wav.resolve(strict=True)
        canonical_audio_path = args.canonical_audio_wav.resolve(strict=True)
        face_path = args.primary_face.resolve(strict=True)
        context_npz = args.context_npz.resolve(strict=True)
        context_json = args.context_json.resolve(strict=True)
        if sha256_file(original_source) != OLDPILOT_ORIGINAL_WAV_SHA256:
            raise ValueError("oldpilot original natural WAV SHA mismatch")
        face_sha = sha256_file(face_path)
        if face_sha != OLDPILOT_PRIMARY_FACE_SHA256:
            raise ValueError("oldpilot primary face SHA mismatch")
        if sha256_file(wav2lip_checkpoint) != args.expected_wav2lip_checkpoint_sha256.lower():
            raise ValueError("Wav2Lip checkpoint differs from the explicit caller SHA")
        if sha256_file(syncnet_checkpoint) != args.expected_syncnet_checkpoint_sha256.lower():
            raise ValueError("SyncNet checkpoint differs from the explicit caller SHA")

        natural, natural_info = read_canonical_wav(canonical_audio_path)
        if (
            natural_info.sample_rate != 16_000
            or natural_info.source_sample_rate != 16_000
            or natural_info.source_channels != 1
            or natural_info.source_encoding != "FLOAT_32"
            or natural_info.frames != 92_225
        ):
            raise ValueError("oldpilot canonical float32 PCM does not match the known 92,225-sample contract")
        if natural_info.file_sha256 != args.expected_canonical_file_sha256.lower():
            raise ValueError("oldpilot canonical natural WAV file SHA differs from the explicit caller pin")
        if natural_info.pcm_float32_sha256 != args.expected_canonical_pcm_sha256.lower():
            raise ValueError("oldpilot canonical natural float32 PCM SHA differs from the explicit caller pin")
        arrays, context_meta, context_files = _load_context(context_npz, context_json)
        for field, expected in (
            ("original_source_sha256", OLDPILOT_ORIGINAL_WAV_SHA256),
            ("primary_face_sha256", face_sha),
            ("canonical_natural_pcm_sha256", natural_info.pcm_float32_sha256),
        ):
            if context_meta.get(field) != expected:
                raise ValueError(f"oldpilot context source binding mismatch: {field}")
        track_start = int(context_meta["track_start_frame"])
        full_frame_count = int(context_meta["frame_count"])
        training_frame_count = int(args.training_frame_count)
        if not 6 <= training_frame_count <= full_frame_count:
            raise ValueError("training frame prefix must be between six and the context frame count")
        timestamps = arrays["frame_timestamps_seconds_f64"][:training_frame_count]

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        wav2lip_model, syncnet_model, mel_frontend, audio_module, psf_mfcc, source_pins, _ = _load_models_and_frontend(
            wav2lip_root=wav2lip_root,
            wav2lip_checkpoint=wav2lip_checkpoint,
            syncnet_root=syncnet_root,
            syncnet_checkpoint=syncnet_checkpoint,
            device=device,
        )
        frontend_parity = check_frontend_parity(mel_frontend, audio_module, natural, atol=1e-4, rtol=1e-5)
        if frontend_parity.get("status") != "PASS":
            raise RuntimeError("Wav2Lip Torch mel frontend did not pass numerical parity")
        # Derive the official chunk grid from the actual pinned frontend output
        # length. Duration rounding is not a substitute for librosa's centered
        # STFT frame count and can shift the final repeated-tail window.
        with torch.no_grad():
            mel_frames = int(mel_frontend(torch.from_numpy(natural.copy()).to(device=device)).shape[-1])
        full_mel_starts = official_chunk_starts(mel_frames, fps_num=25, fps_den=1, width=16)
        if len(full_mel_starts) != full_frame_count:
            raise ValueError("official Wav2Lip mel chunk count differs from the frozen source video frame count")
        mel_starts = full_mel_starts[:training_frame_count]
        # The official selected SyncNet crop is two frames longer than this
        # historically exposed Wav2Lip MP4. Use only the PCM16 prefix supporting
        # the declared engineering query prefix and bind both full and used
        # crop hashes; this is not a claim that AVI audio equals direct PCM.
        used_pcm16_samples = training_frame_count * 640
        full_pcm16 = arrays["syncnet_pcm16_crop_i16"]
        if full_pcm16.size < used_pcm16_samples:
            raise ValueError("official SyncNet PCM16 crop does not cover the engineering frame prefix")
        used_pcm16 = np.ascontiguousarray(full_pcm16[:used_pcm16_samples], dtype=np.int16)
        used_pcm16_sha256 = array_sha256(used_pcm16, "<i2")
        adapter = Wav2LipSyncAdapter(
            wav2lip_model=wav2lip_model,
            syncnet_model=syncnet_model,
            mel_frontend=mel_frontend,
            frontend_parity=frontend_parity,
            frame_batch_size=args.frame_batch_size,
            syncnet_batch_size=args.syncnet_batch_size,
            psf_mfcc=psf_mfcc,
        )
        context = Wav2LipSyncContext(
            face_crops_bgr=arrays["face_crops_bgr_u8"][:training_frame_count],
            syncnet_crops_bgr=arrays["syncnet_crops_bgr_u8"][:training_frame_count],
            patch_boxes_xyxy=arrays["patch_boxes_xyxy_i32"][:training_frame_count],
            mel_starts=mel_starts,
            frame_timestamps_seconds=timestamps,
            mel_starts_are_prefix=training_frame_count < full_frame_count,
            fps_num=25,
            fps_den=1,
            track_start_frame=track_start,
            syncnet_pcm16_crop=used_pcm16,
            syncnet_pcm16_crop_sha256=used_pcm16_sha256,
            expected_sample_count=natural.size,
            source_video_sha256=context_meta["source_video_sha256"],
            face_crop_sha256=context_meta["face_crops_sha256"],
            syncnet_crop_sha256=context_meta["syncnet_crops_sha256"],
        )
        context.validate()

        head = SpectralResidualHead(
            n_fft=800,
            hop_length=200,
            hidden_channels=args.hidden_channels,
            max_residual_db=args.max_residual_db,
            level_normalize=True,
        ).to(device)
        stack = EnhancementStack(head).to(device)
        audio_tensor = torch.from_numpy(natural.copy()).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            identity_audio, identity_mask = stack.forward_with_mask(audio_tensor, use_spectral_head=True)
        identity_max_abs = float((identity_audio - audio_tensor).abs().max().item())
        if identity_audio.shape != audio_tensor.shape or identity_max_abs != 0.0:
            raise RuntimeError(f"zero-initialized spectral head is not exact identity: max abs={identity_max_abs}")
        if not bool(torch.isfinite(identity_mask).all().item()):
            raise FloatingPointError("initial spectral mask is non-finite")

        with torch.no_grad():
            baseline_frames = adapter.render_video(context, identity_audio)
            baseline_curve = adapter.sync_distance_curve(baseline_frames, audio_tensor, context=context, vshift=15)
            baseline_curve_cpu = baseline_curve.detach().cpu().numpy().astype(np.float64)
            baseline_cbest = float(np.median(baseline_curve_cpu) - np.min(baseline_curve_cpu))
            baseline_canchor = float(np.median(baseline_curve_cpu) - baseline_curve_cpu[15 + ENGINEERING_ANCHOR_LAG])
            baseline_best_lag = int(np.argmin(baseline_curve_cpu) - 15)

        optimizer = torch.optim.Adam(head.parameters(), lr=args.learning_rate)
        parameters_before = {
            name: value.detach().clone() for name, value in head.named_parameters()
        }
        step = sync_guided_train_step(
            enhancement_stack=stack,
            optimizer=optimizer,
            natural_audio=audio_tensor,
            video_context=context,
            render_video=adapter.render_video,
            sync_score=adapter.sync_score,
            use_global_rms=False,
            use_local_envelope=False,
            use_spectral_head=True,
            post_rms_match="none",
            anchor_lag_frames=ENGINEERING_ANCHOR_LAG,
            identity_weight=args.identity_weight,
            spectral_smoothness_weight=args.spectral_smoothness_weight,
        )
        per_parameter_gradients = []
        for name, parameter in head.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                per_parameter_gradients.append({"name": name, "present": False, "finite": False,
                                                "nonzero_elements": 0, "element_count": parameter.numel(),
                                                "l2": None})
                continue
            finite = bool(torch.isfinite(gradient).all().item())
            per_parameter_gradients.append({
                "name": name,
                "present": True,
                "finite": finite,
                "nonzero_elements": int(torch.count_nonzero(gradient).item()),
                "element_count": int(gradient.numel()),
                "l2": float(torch.linalg.vector_norm(gradient.detach().to(torch.float64)).item()),
            })
        head_after = {
            name: value.detach().clone() for name, value in head.named_parameters()
        }
        parameter_update_max_abs = max(
            float((head_after[name] - parameters_before[name]).abs().max().item())
            for name in head_after
        )
        if parameter_update_max_abs <= 0.0:
            raise RuntimeError("one optimizer step did not change any head parameter")
        with torch.no_grad():
            audio_after = stack(audio_tensor, use_spectral_head=True)
        if audio_after.shape != audio_tensor.shape or not bool(torch.isfinite(audio_after).all().item()):
            raise RuntimeError("updated audio head changed output shape or produced non-finite PCM")
        waveform_delta = float((audio_after - audio_tensor).abs().max().item())
        if waveform_delta <= 0.0:
            raise RuntimeError("updated head did not change the engineering waveform")

        source_root = Path(__file__).resolve().parents[3]
        package_sources = {
            rel: {"path": str((source_root / rel).resolve()), "bytes": (source_root / rel).stat().st_size,
                  "sha256": sha256_file((source_root / rel).resolve())}
            for rel in (
                "scripts/audio_enhancement/audio_io.py",
                "scripts/audio_enhancement/heads.py",
                "scripts/audio_enhancement/differentiable_adapter.py",
                "scripts/audio_enhancement/training.py",
                "scripts/audio_enhancement/wav2lip_frontend.py",
            )
        }
        head_state = {name: value.detach().cpu().contiguous() for name, value in head.state_dict().items()}
        checkpoint_path = output_dir / "head_after_one_step.pt"
        temporary_checkpoint = output_dir / f".{checkpoint_path.name}.{uuid.uuid4().hex}.tmp"
        torch.save(
            {
                "schema": "cross_tfg_audio_head_engineering_checkpoint_v1",
                "engineering_only": True,
                "historically_exposed": True,
                "architecture": {
                    "n_fft": 800,
                    "hop_length": 200,
                    "hidden_channels": args.hidden_channels,
                    "max_residual_db": args.max_residual_db,
                    "level_normalize": True,
                },
                "state_dict": head_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "learning_rate": args.learning_rate,
                "anchor_lag_frames": ENGINEERING_ANCHOR_LAG,
                "anchor_scope": "oldpilot engineering smoke only; not frozen for fit/dev/confirm",
                "package_sources": package_sources,
                "model_sources": source_pins,
                "input_context_files": context_files,
            },
            temporary_checkpoint,
        )
        with temporary_checkpoint.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_checkpoint, checkpoint_path)

        report = {
            "schema": "cross_tfg_audio_head_one_step_gradient_smoke_v1",
            "status": "PASS",
            "run_id": "cross_tfg_audio_head_oldpilot_gradient_smoke_20261003",
            "scope": "engineering-only oldpilot runtime/gradient path",
            "historically_exposed": True,
            "candidate_selection_authorized": False,
            "development_scores_read": False,
            "confirmation_scores_read": False,
            "official_video_score_claimed": False,
            "proxy_disclaimer": "Wav2Lip-to-SyncNet differentiable crop/paste path is a training proxy, not the official AVI/JPEG/codec scorer path.",
            "input": {
                "original_source_wav": {"path": str(original_source), "bytes": original_source.stat().st_size, "sha256": OLDPILOT_ORIGINAL_WAV_SHA256},
                "primary_face": {"path": str(face_path), "bytes": face_path.stat().st_size, "sha256": face_sha},
                "canonical_audio_wav": {"path": str(canonical_audio_path), "bytes": canonical_audio_path.stat().st_size,
                                        "file_sha256": natural_info.file_sha256, "pcm_sha256": natural_info.pcm_float32_sha256,
                                        "sample_rate_hz": natural_info.sample_rate, "channels": 1, "sample_count": natural_info.frames},
                "context_npz": {"path": str(context_npz), **context_files},
                "context_json": {"path": str(context_json), "bytes": context_json.stat().st_size},
                "frame_count": context.frame_count,
                "context_full_frame_count": full_frame_count,
                "engineering_frame_prefix_count": training_frame_count,
                "track_start_frame": track_start,
                "official_mel_chunk_starts_sha256": sha256_bytes(np.asarray(mel_starts, dtype="<i8").tobytes()),
                "syncnet_pcm16_crop_sha256": context_meta["array_sha256"]["syncnet_pcm16_crop_i16"],
                "syncnet_pcm16_crop_samples": int(full_pcm16.size),
                "syncnet_pcm16_engineering_prefix_sha256": context.syncnet_pcm16_crop_sha256,
                "syncnet_pcm16_engineering_prefix_samples": int(used_pcm16.size),
                "pcm16_derivation": context_meta.get("pcm16_derivation"),
                "pcm16_direct_float_slice_sha256": context_meta.get("pcm16_direct_float_slice_sha256"),
                "official_crop_vs_direct_pcm16_same": context_meta.get("pcm16_direct_float_slice_sha256") == context_meta["array_sha256"]["syncnet_pcm16_crop_i16"],
            },
            "frontend_parity": frontend_parity,
            "official_engineering_baseline_proxy": {
                "lag_curve_distance_float64": baseline_curve_cpu.tolist(),
                "Cbest_proxy": baseline_cbest,
                "Canchor_plus2_proxy": baseline_canchor,
                "best_lag_proxy_frames": baseline_best_lag,
                "official_oldpilot_Cbest_context": "separately measured by official rendered-video QC04; this report does not replace it",
            },
            "training_step": {
                "optimizer_steps": 1,
                "optimizer": "torch.optim.Adam",
                "learning_rate": args.learning_rate,
                "anchor_lag_frames": ENGINEERING_ANCHOR_LAG,
                "anchor_is_engineering_only": True,
                "loss_proxy": "negative C_anchor = D(anchor+2) - median(D[-15:+15]) plus identity and mask smoothness penalties",
                "head_identity_initial_max_abs": identity_max_abs,
                "audio_gradient_l2": step.audio_gradient_l2,
                "head_gradient_l2": step.gradient_l2,
                "parameter_update_l2": step.parameter_update_l2,
                "parameter_update_max_abs": parameter_update_max_abs,
                "enhanced_audio_max_abs_delta": waveform_delta,
                "proxy_Canchor_plus2_before_step": step.sync_score_mean,
                "loss": step.loss,
                "identity_penalty": step.identity_penalty,
                "aggregate_audio_and_head_gradients_finite_nonzero": bool(
                    step.audio_gradient_l2 > 0.0 and step.gradient_l2 > 0.0
                    and all(item["finite"] for item in per_parameter_gradients if item["present"])
                ),
                "gradient_tensor_count": len(per_parameter_gradients),
                "gradient_tensors_present": sum(1 for item in per_parameter_gradients if item["present"]),
                "gradient_tensors_finite": sum(1 for item in per_parameter_gradients if item["present"] and item["finite"]),
                "gradient_tensors_with_nonzero_elements": sum(1 for item in per_parameter_gradients if item["present"] and item["nonzero_elements"] > 0),
                "gradient_nonzero_elements_total": sum(item["nonzero_elements"] for item in per_parameter_gradients),
                "per_parameter_gradients": per_parameter_gradients,
                "frozen_wav2lip_and_syncnet_parameters": True,
            },
            "runtime": {**source_pins, "device": str(device), "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "seed": args.seed, "torch_threads": torch.get_num_threads(), "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)},
            "package_sources": package_sources,
            "checkpoint": {"path": str(checkpoint_path), "bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path)},
            "created_utc_epoch": __import__("time").time(),
        }
        report["manifest_sha256"] = canonical_sha(report)
        report_path = output_dir / "one_step_gradient_smoke.json"
        atomic_json(report_path, report)
        report["report_raw_sha256"] = sha256_file(report_path)
        status.update({
            "status": "PASS",
            "optimizer_steps": 1,
            "report_path": str(report_path),
            "report_raw_sha256": sha256_file(report_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
        })
        atomic_json(status_path, status)
        return report
    except Exception as exc:
        status.update({
            "status": "FAILED",
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
            "traceback": traceback.format_exc(),
        })
        atomic_json(status_path, status)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav2lip-root", type=Path, required=True)
    parser.add_argument("--wav2lip-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-wav2lip-checkpoint-sha256", required=True)
    parser.add_argument("--syncnet-root", type=Path, required=True)
    parser.add_argument("--syncnet-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-syncnet-checkpoint-sha256", required=True)
    parser.add_argument("--original-source-wav", type=Path, required=True)
    parser.add_argument("--canonical-audio-wav", type=Path, required=True)
    parser.add_argument("--expected-canonical-file-sha256", required=True)
    parser.add_argument("--expected-canonical-pcm-sha256", required=True)
    parser.add_argument("--primary-face", type=Path, required=True)
    parser.add_argument("--context-npz", type=Path, required=True)
    parser.add_argument("--context-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--syncnet-batch-size", type=int, default=32)
    parser.add_argument("--training-frame-count", type=int, default=32,
                        help="short engineering-only prefix to bound autograd memory; not a scientific support choice")
    parser.add_argument("--hidden-channels", type=int, default=24)
    parser.add_argument("--max-residual-db", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--identity-weight", type=float, default=0.02)
    parser.add_argument("--spectral-smoothness-weight", type=float, default=0.002)
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({
        "status": report["status"],
        "report_path": str(args.output_dir / "one_step_gradient_smoke.json"),
        "manifest_sha256": report["manifest_sha256"],
        "frontend_parity": report["frontend_parity"],
        "training_step": report["training_step"],
        "checkpoint": report["checkpoint"],
    }, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
