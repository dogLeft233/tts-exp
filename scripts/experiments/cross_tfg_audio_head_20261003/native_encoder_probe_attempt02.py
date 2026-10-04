#!/usr/bin/env python3
"""Native audio-encoder response probe for the frozen eight historical-fit clips.

This script measures learned audio representations only.  It never runs a
generator, video pipeline, SyncNet, or reads development/confirmation audio.
Each invocation is isolated to one pinned frontend/runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import time
import traceback
from typing import Any

import numpy as np


PLAN_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
PROFILE_SHA256 = "fd7303f513fdbada80da50e5dd395e9441142f1658544765d7523f1618f55253"
BASELINES_SHA256 = "7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2"
AUDIO_IO_SHA256 = "7b951990343cf5d68f3f6046e77cd5c40ee736777d196de3723418bca5965004"
TARGET_RMS = 0.0376838172675746
FPS = 25
SAMPLE_RATE = 16000
CONDITIONS = (
    "natural",
    "global_minus_6db",
    "global_plus_6db",
    "fixed_rms_0p0376838173",
    "local_r090_accent015_tilt_p025",
    "local_r090_flatten015_tilt_m025",
    "local_r090_accent_0.30_tilt0",
)
PROFILE_IDS = CONDITIONS[4:]
PROFILE_CONFIG_SHA256 = {
    "local_r090_accent015_tilt_p025": "19184db8a2fe48e2a6ed61002fe475609f39d719036001798de956bbec8cd60c",
    "local_r090_flatten015_tilt_m025": "11ef2f50de97e3fc8d0598721c374e0c0891719ecc50333adc53c4a58653cd51",
    "local_r090_accent_0.30_tilt0": "1e61cc4a9c5cc48a3e73b8e632444adcccb04ab705d8bf160f05dbf382435df2",
}
EXPECTED_MODELS = {
    "wav2lip": "wav2lip_gan_bac9a81e",
    "muse": "musetalk_fixed_portrait_attempt16",
    "ditto": "ditto_hubert_fixed_portrait_attempt18",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(samples: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(samples, dtype="<f4").tobytes()).hexdigest()


def require_file(path: Path, expected_sha: str | None, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing/nonregular {label}: {path}")
    observed = sha256_file(path)
    if expected_sha and observed != expected_sha:
        raise RuntimeError(f"SHA256 mismatch for {label}: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}


def load_plan(plan_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan_ref = require_file(plan_path, PLAN_SHA256, "frozen fit plan")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("run_id") != "cross_tfg_shared_frequency_profiles_fit_20261003_attempt03":
        raise RuntimeError("unexpected fit plan run id")
    if len(plan.get("rows", [])) != 360 or plan.get("execution", {}).get("scope", "").find("no dev/confirm") < 0:
        raise RuntimeError("fit plan scope/row count mismatch")
    rows = plan["rows"]
    plan_methods = {m["method_id"]: m for m in plan.get("methods", [])}
    for profile_id in PROFILE_IDS:
        method = plan_methods.get(profile_id)
        if not method or method.get("config_sha256") != PROFILE_CONFIG_SHA256[profile_id]:
            raise RuntimeError(f"selected profile config pin differs from fit plan: {profile_id}")
        cfg = method.get("config", {})
        if (
            cfg.get("method") != "shared_frequency_profile"
            or cfg.get("profile_id") != profile_id
            or cfg.get("profile_source_sha256") != PROFILE_SHA256
            or cfg.get("target_rms") != TARGET_RMS
            or cfg.get("peak_ceiling") != 0.98
        ):
            raise RuntimeError(f"selected profile parameters differ from fit plan: {profile_id}")
    rms_method = plan_methods.get("rms_fixed")
    if not rms_method or rms_method.get("config", {}).get("target_rms") != TARGET_RMS or rms_method.get("config", {}).get("peak_ceiling") is not None:
        raise RuntimeError("fixed RMS control differs from fit plan")
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("split") != "fit" or row.get("method_id") != "identity":
            continue
        if row.get("model_id") != EXPECTED_MODELS["wav2lip"]:
            continue
        sid = row["sample_id"]
        if sid in selected:
            raise RuntimeError(f"duplicate identity source row for {sid}")
        selected[sid] = row
    if len(selected) != 8:
        raise RuntimeError(f"expected eight unique identity source rows, found {len(selected)}")
    # Preserve the source-row order in the frozen plan.
    sample_ids = list(selected)
    speakers = sorted({selected[sid]["speaker_id"] for sid in sample_ids})
    if len(speakers) != 4 or not set(speakers).issubset(set(plan["fit_speaker_ids"])):
        raise RuntimeError("fit roster mismatch")
    # All three frontends must refer to the same original PCM source for every clip.
    for sid in sample_ids:
        refs = [
            r for r in rows
            if r.get("split") == "fit" and r.get("method_id") == "identity" and r.get("sample_id") == sid
        ]
        if {r.get("model_id") for r in refs} != set(EXPECTED_MODELS.values()) or len(refs) != 3:
            raise RuntimeError(f"identity model roster differs for {sid}")
        source_keys = {
            (r["natural_pcm"]["path"], r["natural_pcm"]["file_sha256"], r["natural_pcm"]["pcm_sha256"], r["natural_pcm"]["sample_count"])
            for r in refs
        }
        if len(source_keys) != 1:
            raise RuntimeError(f"original PCM binding differs by frontend for {sid}")
    plan["_raw_binding"] = plan_ref
    return plan, [selected[sid] for sid in sample_ids]


def load_transform_helpers(repo_root: Path):
    expected = {
        "scripts/audio_enhancement/shared_frequency_profile_v1.py": PROFILE_SHA256,
        "scripts/audio_enhancement/baselines.py": BASELINES_SHA256,
        "scripts/audio_enhancement/audio_io.py": AUDIO_IO_SHA256,
    }
    source_manifest = {}
    for rel, digest in expected.items():
        source_manifest[rel] = require_file(repo_root / rel, digest, rel)
    sys.path.insert(0, str(repo_root))
    from scripts.audio_enhancement.audio_io import read_canonical_wav, pcm_float32_sha256, write_float32_wav
    from scripts.audio_enhancement.baselines import apply_global_rms, FIXED_POSITIVE_CONTROL_RMS
    from scripts.audio_enhancement.shared_frequency_profile_v1 import transform_shared_profile
    if FIXED_POSITIVE_CONTROL_RMS != TARGET_RMS:
        raise RuntimeError("fixed-RMS target differs from the frozen plan")
    return read_canonical_wav, pcm_float32_sha256, write_float32_wav, apply_global_rms, transform_shared_profile, source_manifest


def build_condition_waveforms(
    waveform: np.ndarray,
    apply_global_rms,
    transform_shared_profile,
    pcm_hash,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    x = np.asarray(waveform, dtype=np.float32)
    scale_minus = np.float32(10.0 ** (-6.0 / 20.0))
    scale_plus = np.float32(10.0 ** (6.0 / 20.0))
    output: dict[str, np.ndarray] = {"natural": np.ascontiguousarray(x.copy())}
    diagnostics: dict[str, dict[str, Any]] = {}
    output["global_minus_6db"] = np.ascontiguousarray(x * scale_minus, dtype=np.float32)
    output["global_plus_6db"] = np.ascontiguousarray(x * scale_plus, dtype=np.float32)
    output["fixed_rms_0p0376838173"], rms_info = apply_global_rms(
        x, target_rms=TARGET_RMS, peak_ceiling=None
    )
    diagnostics["fixed_rms_0p0376838173"] = {
        "applied_gain": float(rms_info.applied_gain),
        "output_rms": float(rms_info.output_rms),
        "peak_limited": bool(rms_info.peak_limited),
    }
    for profile_id in PROFILE_IDS:
        transformed, details = transform_shared_profile(
            x,
            profile_id,
            sample_rate_hz=SAMPLE_RATE,
            target_rms=TARGET_RMS,
            peak_ceiling=0.98,
        )
        output[profile_id] = np.ascontiguousarray(transformed, dtype=np.float32)
        diagnostics[profile_id] = details
    for condition, values in output.items():
        if values.shape != x.shape or values.dtype != np.float32 or not np.isfinite(values).all():
            raise RuntimeError(f"invalid transformed waveform for {condition}")
        diagnostics.setdefault(condition, {})
        diagnostics[condition].update({
            "sample_count": int(values.size),
            "sample_rate_hz": SAMPLE_RATE,
            "pcm_float32_sha256": pcm_hash(values),
            "rms": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
            "peak_abs": float(np.max(np.abs(values.astype(np.float64)))),
        })
    return output, diagnostics


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator = np.einsum("ij,ij->i", a, b, optimize=True)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(numerator, denom, out=np.full_like(numerator, np.nan), where=denom > 1e-20)


def pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.size < 3 or np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _query_envelope(waveform: np.ndarray, query_count: int, fps: int) -> np.ndarray:
    """Return 40 ms source-clock log-RMS values at query starts i/fps."""
    values = np.asarray(waveform, dtype=np.float64)
    result = np.full(query_count, np.nan, dtype=np.float64)
    for i in range(query_count):
        start = i * SAMPLE_RATE // fps
        stop = min(int(values.size), (i + 1) * SAMPLE_RATE // fps)
        if stop > start:
            part = values[start:stop]
            result[i] = np.log(max(float(np.sqrt(np.mean(part * part))), 1e-12))
    return result


def feature_metrics(
    natural: np.ndarray,
    altered: np.ndarray,
    natural_waveform: np.ndarray,
    condition_waveform: np.ndarray,
    fps: int,
) -> dict[str, Any]:
    n = np.asarray(natural).reshape(natural.shape[0], -1).astype(np.float64)
    a = np.asarray(altered).reshape(altered.shape[0], -1).astype(np.float64)
    if n.shape != a.shape or n.shape[0] < 3:
        raise RuntimeError("native encoder output shape or time support changed across conditions")
    n_norm = np.linalg.norm(n, axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    raw_cos = cosine_rows(n, a)
    n_centered = n - n.mean(axis=0, keepdims=True)
    a_centered = a - a.mean(axis=0, keepdims=True)
    centered_cos = cosine_rows(n_centered, a_centered)
    n_delta, a_delta = np.diff(n, axis=0), np.diff(a, axis=0)
    n_delta_norm = np.linalg.norm(n_delta, axis=1)
    a_delta_norm = np.linalg.norm(a_delta, axis=1)
    n_delta_centered = n_delta - n_delta.mean(axis=0, keepdims=True)
    a_delta_centered = a_delta - a_delta.mean(axis=0, keepdims=True)

    # Original and altered envelopes are both sampled on the unchanged source
    # clock. This is an index-clock association, not a receptive-field claim.
    natural_log_rms = _query_envelope(natural_waveform, n.shape[0], fps)
    condition_log_rms = _query_envelope(condition_waveform, n.shape[0], fps)
    valid_natural_env = np.isfinite(natural_log_rms)
    valid_condition_env = np.isfinite(condition_log_rms)
    valid_natural_trans = valid_natural_env[1:] & valid_natural_env[:-1]
    valid_condition_trans = valid_condition_env[1:] & valid_condition_env[:-1]
    delta_natural_log_rms = np.diff(natural_log_rms)
    delta_condition_log_rms = np.diff(condition_log_rms)
    return {
        "query_count": int(n.shape[0]),
        "feature_dimension_after_flatten": int(n.shape[1]),
        "raw_frame_cosine_mean": float(np.nanmean(raw_cos)),
        "raw_frame_cosine_min": float(np.nanmin(raw_cos)),
        "feature_norm_ratio_mean": float(np.mean(a_norm / np.maximum(n_norm, 1e-20))),
        "feature_norm_ratio_median": float(np.median(a_norm / np.maximum(n_norm, 1e-20))),
        "query_time_centered_frame_cosine_mean": float(np.nanmean(centered_cos)),
        "query_time_centered_global_cosine": float(np.sum(n_centered * a_centered) / max(float(np.linalg.norm(n_centered) * np.linalg.norm(a_centered)), 1e-20)),
        "query_time_centered_relative_rms_change": float(np.linalg.norm(a_centered - n_centered) / max(float(np.linalg.norm(n_centered)), 1e-20)),
        "transition_delta_cosine_mean": float(np.nanmean(cosine_rows(n_delta, a_delta))),
        "transition_delta_norm_ratio_mean": float(np.mean(a_delta_norm / np.maximum(n_delta_norm, 1e-20))),
        "transition_delta_query_time_centered_cosine": float(np.sum(n_delta_centered * a_delta_centered) / max(float(np.linalg.norm(n_delta_centered) * np.linalg.norm(a_delta_centered)), 1e-20)),
        "transition_delta_relative_rms_change": float(np.linalg.norm(a_delta - n_delta) / max(float(np.linalg.norm(n_delta)), 1e-20)),
        "natural_log_rms_vs_natural_feature_norm_pearson": pearson(natural_log_rms[valid_natural_env], np.log(np.maximum(n_norm[valid_natural_env], 1e-20))),
        "natural_log_rms_vs_condition_feature_norm_pearson": pearson(natural_log_rms[valid_natural_env], np.log(np.maximum(a_norm[valid_natural_env], 1e-20))),
        "condition_log_rms_vs_condition_feature_norm_pearson": pearson(condition_log_rms[valid_condition_env], np.log(np.maximum(a_norm[valid_condition_env], 1e-20))),
        "natural_abs_log_rms_step_vs_natural_feature_transition_norm_pearson": pearson(np.abs(delta_natural_log_rms[valid_natural_trans]), n_delta_norm[valid_natural_trans]),
        "natural_abs_log_rms_step_vs_condition_feature_transition_norm_pearson": pearson(np.abs(delta_natural_log_rms[valid_natural_trans]), a_delta_norm[valid_natural_trans]),
        "condition_abs_log_rms_step_vs_condition_feature_transition_norm_pearson": pearson(np.abs(delta_condition_log_rms[valid_condition_trans]), a_delta_norm[valid_condition_trans]),
        "natural_envelope_valid_query_count": int(valid_natural_env.sum()),
        "condition_envelope_valid_query_count": int(valid_condition_env.sum()),
        "feature_centering_axis": "query_time_axis; flattened feature dimensions are not frequency-centered",
    }


def source_clock_map(frontend: str, query_count: int, sample_count: int, mel_frames: int | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for q in range(query_count):
        item: dict[str, Any] = {
            "query_index": q,
            "query_timestamp_num": q,
            "query_timestamp_den": FPS,
            "query_timestamp_seconds": q / FPS,
            "source_clock_frame_index": q,
            "source_sample_start": q * (SAMPLE_RATE // FPS),
            "source_sample_stop_exclusive": min(sample_count, (q + 1) * (SAMPLE_RATE // FPS)),
        }
        if frontend == "wav2lip":
            start = int(q * 80.0 / FPS)
            if mel_frames is None:
                raise RuntimeError("Wav2Lip mel frame count is missing")
            tail = start + 16 > mel_frames
            selected_start = mel_frames - 16 if tail else start
            item.update({
                "mel_start_index": int(selected_start),
                "mel_stop_exclusive": int(selected_start + 16),
                "uses_official_tail_window_reuse": bool(tail),
            })
        elif frontend == "muse":
            # Exact AudioProcessor.get_whisper_chunk selection: 50 Hz encoder
            # tokens, four left zeros, ten selected positions per 25 Hz query.
            audio_start = int(math.floor(q * 2.0))
            item.update({
                "whisper_feature_start_after_left_padding": audio_start,
                "whisper_feature_stop_exclusive_after_left_padding": audio_start + 10,
                "whisper_source_token_indices_50hz": list(range(audio_start - 4, audio_start + 6)),
                "whisper_left_padding_tokens": [x for x in range(audio_start - 4, audio_start + 6) if x < 0],
                "encoder_self_attention_support": "full 30-second Whisper segment; selected 10-token span is not a waveform receptive-field claim",
            })
        elif frontend == "ditto":
            chunk_start_frame = (q // 5) * 5
            padded_start = chunk_start_frame * (SAMPLE_RATE // FPS)
            left_padding_samples = 2000
            chunk_source_start = padded_start - left_padding_samples
            chunk_source_stop = chunk_source_start + 6480
            item.update({
                "wav2feat_chunk_start_frame": int(chunk_start_frame),
                "wav2feat_chunk_source_start_sample_unclipped": int(chunk_source_start),
                "wav2feat_chunk_source_stop_sample_unclipped": int(chunk_source_stop),
                "wav2feat_left_zero_pad_samples": max(0, -chunk_source_start),
                "wav2feat_right_zero_pad_samples": max(0, chunk_source_stop - sample_count),
                "wav2feat_chunk_length_samples": 6480,
                "wav2feat_chunk_valid_output_frames": 5,
            })
        rows.append(item)
    return {
        "fps_num": FPS,
        "fps_den": 1,
        "source_sample_rate_hz": SAMPLE_RATE,
        "sample_count": int(sample_count),
        "query_count": int(query_count),
        "queries": rows,
    }


def configure_torch(torch, *, require_cuda: bool, half: bool = False):
    if torch.__version__ != "2.5.1+cu124":
        raise RuntimeError(f"unexpected pinned frontend Torch version: {torch.__version__}")
    torch.set_num_threads(1)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("this pinned native frontend requires the qualified CUDA execution route")
    device = torch.device("cuda:0" if require_cuda else "cpu")
    dtype = torch.float16 if half else torch.float32
    return device, dtype


def init_wav2lip(plan: dict[str, Any], torch, repo_root: Path):
    model_plan = next(m for m in plan["models"] if m["model_id"] == EXPECTED_MODELS["wav2lip"])
    gen = model_plan["generator"]
    repo = Path(gen["repo"])
    checkpoint = Path(gen["checkpoint"])
    source_manifest = {}
    for rel, expected in model_plan["source_sha256"].items():
        source_manifest[rel] = require_file(repo / rel, expected, f"Wav2Lip source {rel}")
    checkpoint_manifest = require_file(checkpoint, gen["checkpoint_sha256"], "Wav2Lip checkpoint")
    sys.path.insert(0, str(repo))
    import audio as w2l_audio
    from models import Wav2Lip
    device, _ = configure_torch(torch, require_cuda=True, half=False)
    model = Wav2Lip()
    checkpoint_obj = torch.load(str(checkpoint), map_location=device)
    state = checkpoint_obj["state_dict"]
    remapped = {key.replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(remapped)
    model = model.to(device).eval()
    return model_plan, model, w2l_audio, device, {
        "source_commit": model_plan["git_head"],
        "source_files": source_manifest,
        "checkpoint": checkpoint_manifest,
    }


def run_wav2lip_encoder(wave: np.ndarray, temp_path: Path, *, model, w2l_audio, device, torch):
    # Match the pinned inference file path: write IEEE-float32, load with
    # audio.load_wav(16000), then audio.melspectrogram and official 16-column windows.
    loaded = w2l_audio.load_wav(str(temp_path), SAMPLE_RATE)
    if loaded.dtype != np.float32 or loaded.shape != wave.shape or canonical_sha(loaded) != canonical_sha(wave):
        raise RuntimeError("Wav2Lip audio loader changed canonical PCM samples or sample count")
    mel = w2l_audio.melspectrogram(loaded)
    if mel.ndim != 2 or mel.shape[0] != 80 or mel.shape[1] < 16 or not np.isfinite(mel).all():
        raise RuntimeError(f"unexpected Wav2Lip mel shape: {mel.shape}")
    chunks = []
    i = 0
    while True:
        start = int(i * 80.0 / FPS)
        if start + 16 > mel.shape[1]:
            chunks.append(mel[:, mel.shape[1] - 16:])
            break
        chunks.append(mel[:, start:start + 16])
        i += 1
    mel_batch = np.asarray(chunks, dtype=np.float32)
    mel_tensor = torch.from_numpy(mel_batch[:, None, :, :]).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        result = model.audio_encoder(mel_tensor)
    features = result.detach().float().reshape(result.shape[0], -1).cpu().numpy()
    return np.ascontiguousarray(features), source_clock_map("wav2lip", len(features), len(wave), mel.shape[1]), {
        "mel_shape": [int(x) for x in mel.shape],
        "official_mel_window_columns": 16,
        "official_mel_window_hop_rule": "int(i*80/fps); last 16 columns reused when next window exceeds mel support",
        "device": str(device),
        "model_eval": not model.training,
    }


def init_muse(plan: dict[str, Any], torch, transformers_module):
    model_plan = next(m for m in plan["models"] if m["model_id"] == EXPECTED_MODELS["muse"])
    source_root = Path(model_plan["generator"]["model_source_root"])
    workspace = Path(model_plan["generator"]["model_workspace"])
    whisper_root = workspace / "models" / "whisper-tiny"
    expected = model_plan["source_sha256"]["musetalk/utils/audio_processor.py"]
    source_manifest = {
        "musetalk/utils/audio_processor.py": require_file(
            source_root / "musetalk/utils/audio_processor.py", expected, "MuseTalk audio_processor.py"
        )
    }
    whisper_cfg = model_plan["model_assets"]["whisper"]
    for entry in whisper_cfg:
        require_file(whisper_root / entry["relative_path"], entry["sha256"], f"Muse Whisper {entry['relative_path']}")
    if transformers_module.__version__ != "4.39.2":
        raise RuntimeError(f"unexpected Transformers runtime: {transformers_module.__version__}")
    from transformers.models.whisper import feature_extraction_whisper
    whisper_extractor_source = Path(feature_extraction_whisper.__file__)
    source_manifest["transformers_whisper_feature_extractor.py"] = require_file(
        whisper_extractor_source,
        "579e4d965290295de49cd49cd1a72357b59083d6cdb5f70cd27d7d5cf3de50bf",
        "Transformers 4.39.2 Whisper feature extractor source",
    )
    sys.path.insert(0, str(source_root))
    from musetalk.utils.audio_processor import AudioProcessor
    processor = AudioProcessor(feature_extractor_path=str(whisper_root))
    device, dtype = configure_torch(torch, require_cuda=True, half=True)
    WhisperModel = transformers_module.WhisperModel
    whisper = WhisperModel.from_pretrained(str(whisper_root), local_files_only=True)
    whisper = whisper.to(device=device, dtype=dtype).eval()
    whisper.requires_grad_(False)
    return model_plan, processor, whisper, device, dtype, {
        "source_commit": model_plan["git_head"],
        "source_files": source_manifest,
        "native_qualification_report": require_file(
            Path(model_plan["generator"]["qualification_report_path"]),
            model_plan["generator"]["qualification_report_sha256"],
            "qualified MuseTalk native report",
        ),
        "whisper_assets": [
            require_file(whisper_root / entry["relative_path"], entry["sha256"], f"Muse Whisper {entry['relative_path']}")
            for entry in whisper_cfg
        ],
        "whisper_root": str(whisper_root),
    }


def run_muse_encoder(wave: np.ndarray, temp_path: Path, *, processor, whisper, device, dtype, torch, librosa_module):
    loaded, loaded_sr = librosa_module.load(str(temp_path), sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    if loaded_sr != SAMPLE_RATE or loaded.dtype != np.float32 or loaded.shape != wave.shape or canonical_sha(loaded) != canonical_sha(wave):
        raise RuntimeError("MuseTalk audio loader changed canonical PCM samples or sample count")
    whisper_inputs, decoded_length = processor.get_audio_feature(str(temp_path))
    if int(decoded_length) != int(wave.size):
        raise RuntimeError("MuseTalk librosa loader changed source sample count")
    with torch.inference_mode():
        result = processor.get_whisper_chunk(
            whisper_inputs,
            device,
            dtype,
            whisper,
            int(decoded_length),
            fps=FPS,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )
    features = result.detach().cpu().numpy()
    if features.ndim != 3 or features.shape[1:] != (50, 384):
        raise RuntimeError(f"unexpected native Muse conditioning shape: {features.shape}")
    return np.ascontiguousarray(features), source_clock_map("muse", len(features), len(wave)), {
        "whisper_input_feature_shapes": [[int(x) for x in v.shape] for v in whisper_inputs],
        "whisper_feature_dtype": str(dtype),
        "conditioning_shape": [int(x) for x in features.shape],
        "conditioning_layout": "query x selected_10_tokens x 5_hidden_states x 384_channels; flattened only for scalar comparison",
        "device": str(device),
        "model_eval": not whisper.training,
        "transformers_version": "4.39.2",
    }


def init_ditto(plan: dict[str, Any]):
    model_plan = next(m for m in plan["models"] if m["model_id"] == EXPECTED_MODELS["ditto"])
    gen = model_plan["generator"]
    source_root = Path(gen["ditto_source_root"])
    weights_root = Path(gen["ditto_weights_root"])
    cfg_path = Path(gen["ditto_config_path"])
    if not cfg_path.is_file():
        raise RuntimeError(f"missing Ditto config: {cfg_path}")
    # Verify the exact source and qualification pins recorded by the frozen
    # fit plan before importing or constructing the native feature extractor.
    pinned_model_sources = model_plan.get("source_sha256")
    if not isinstance(pinned_model_sources, dict) or not pinned_model_sources:
        raise RuntimeError("frozen Ditto plan does not contain source SHA pins")
    pinned_source_records = {}
    for rel, record in sorted(pinned_model_sources.items()):
        if not isinstance(record, dict) or not record.get("sha256"):
            raise RuntimeError(f"invalid frozen Ditto source SHA record: {rel}")
        pinned_source_records[rel] = require_file(source_root / rel, record["sha256"], f"frozen Ditto source {rel}")
    qualification_record = require_file(
        Path(gen["qualification_report_path"]), gen["qualification_report_sha256"], "qualified Ditto report"
    )
    import_report_record = require_file(
        Path(gen["import_report_path"]), gen["import_report_sha256"], "qualified Ditto import report"
    )
    sys.path.insert(0, str(source_root))
    from core.atomic_components.cfg import parse_cfg
    from core.atomic_components.wav2feat import Wav2Feat
    wav2feat_cfg = parse_cfg(str(cfg_path), str(weights_root))[6]
    extractor = Wav2Feat(**wav2feat_cfg)
    hubert = extractor.w2f.hubert
    if hubert.model_type != "onnx" or str(hubert.device).lower() != "cpu":
        raise RuntimeError(f"Ditto audio feature route differs from pinned CPU ONNX: {hubert.model_type}/{hubert.device}")
    files = (
        "core/atomic_components/cfg.py",
        "core/atomic_components/wav2feat.py",
        "core/aux_models/hubert_stream.py",
        "core/utils/load_model.py",
        "core/aux_models/modules/HubertStreamingONNX.py",
    )
    observed = {}
    for rel in files:
        candidate = source_root / rel
        if candidate.is_file():
            observed[rel] = {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": sha256_file(candidate)}
        else:
            raise RuntimeError(f"missing Ditto native feature source: {candidate}")
    onnx_path = Path(wav2feat_cfg["w2f_cfg"]["model_path"])
    if not onnx_path.is_file() or onnx_path.stat().st_size != 1460740880:
        raise RuntimeError("Ditto ONNX path/size differs from the pinned model asset")
    onnx_sha = sha256_file(onnx_path)
    expected_onnx = "04cd6c7c388aaf99cdea8d4599cefede9930e60d6051abd5a33c791aeb931697"
    if onnx_sha != expected_onnx:
        raise RuntimeError("Ditto ONNX SHA differs from frozen model asset")
    config_record = require_file(cfg_path, "8e4c9ecb8dbb5aa4d22cb3f9b2c6ceabc62499b04af5fe074826d5b1044c4146", "Ditto HuBERT config")
    runtime = {
        "frozen_plan_source_files": pinned_source_records,
        "qualified_report": qualification_record,
        "qualified_import_report": import_report_record,
        "observed_frontend_source_files": observed,
        "source_commit_expected": "c3e47eee2e626500017a0556b470d6d4182f85e8",
        "onnx_model": {"path": str(onnx_path), "bytes": onnx_path.stat().st_size, "sha256": onnx_sha},
        "config": config_record,
        "device": str(hubert.device),
        "model_type": str(hubert.model_type),
        "chunksize": [3, 5, 2],
    }
    return model_plan, extractor, runtime


def run_ditto_encoder(wave: np.ndarray, temp_path: Path, *, extractor, librosa_module):
    loaded, sr = librosa_module.load(str(temp_path), sr=SAMPLE_RATE)
    if sr != SAMPLE_RATE or loaded.shape != wave.shape or canonical_sha(loaded) != canonical_sha(wave):
        raise RuntimeError("Ditto librosa loader changed the exact canonical waveform")
    features = extractor.wav2feat(loaded, sr=SAMPLE_RATE, chunksize=(3, 5, 2))
    features = np.asarray(features)
    expected_frames = int(math.ceil(wave.size / (SAMPLE_RATE / FPS)))
    if features.shape != (expected_frames, 1024) or not np.isfinite(features).all():
        raise RuntimeError(f"unexpected Ditto native HuBERT feature shape: {features.shape}; expected {(expected_frames, 1024)}")
    return np.ascontiguousarray(features), source_clock_map("ditto", len(features), len(wave)), {
        "conditioning_shape": [int(x) for x in features.shape],
        "conditioning_dtype": str(features.dtype),
        "device": "cpu",
        "model_type": "onnx",
        "chunksize": [3, 5, 2],
        "chunk_length_samples": 6480,
        "chunk_start_step_samples": 3200,
        "left_speech_pad_samples": 2000,
        "right_speech_pad_samples": 6480,
        "model_eval": "ONNX Runtime inference session",
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "raw_frame_cosine_mean",
        "feature_norm_ratio_mean",
        "query_time_centered_frame_cosine_mean",
        "query_time_centered_global_cosine",
        "query_time_centered_relative_rms_change",
        "transition_delta_cosine_mean",
        "transition_delta_norm_ratio_mean",
        "transition_delta_query_time_centered_cosine",
        "transition_delta_relative_rms_change",
        "natural_log_rms_vs_natural_feature_norm_pearson",
        "natural_log_rms_vs_condition_feature_norm_pearson",
        "condition_log_rms_vs_condition_feature_norm_pearson",
        "natural_abs_log_rms_step_vs_natural_feature_transition_norm_pearson",
        "natural_abs_log_rms_step_vs_condition_feature_transition_norm_pearson",
        "condition_abs_log_rms_step_vs_condition_feature_transition_norm_pearson",
    ]
    result = {}
    for condition in CONDITIONS:
        subset = [r["metrics"] for r in rows if r["condition"] == condition and r["sample_id"] for _ in [0]]
        if len(subset) != 8:
            raise RuntimeError(f"aggregate does not have eight sample rows for {condition}")
        result[condition] = {}
        for field in fields:
            values = np.asarray([x[field] for x in subset if x.get(field) is not None], dtype=np.float64)
            result[condition][field] = {
                "n_valid_clips": int(values.size),
                "equal_clip_mean": float(values.mean()) if values.size else None,
                "minimum_clip": float(values.min()) if values.size else None,
                "maximum_clip": float(values.max()) if values.size else None,
            }
    return result


def execute(args: argparse.Namespace) -> dict[str, Any]:
    start_time = time.time()
    plan_path = Path(args.plan)
    repo_root = Path(args.repo_root)
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    tmp_dir = out_dir / "transient_float32_wavs"
    tmp_dir.mkdir()
    plan, source_rows = load_plan(plan_path)
    read_wav, pcm_hash, write_wav, apply_rms, transform_profile, transform_sources = load_transform_helpers(repo_root)
    if sys.version_info[:3] != (3, 10, 12):
        raise RuntimeError(f"expected pinned Python 3.10.12 runtime, got {sys.version_info[:3]}")
    if np.__version__ != "1.26.4":
        raise RuntimeError(f"expected pinned NumPy 1.26.4 runtime, got {np.__version__}")
    model_id = EXPECTED_MODELS[args.frontend]
    runtime: dict[str, Any] = {
        "python": sys.version,
        "platform": sys.platform,
        "numpy": np.__version__,
        "frontend": args.frontend,
        "model_id": model_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
    model = processor = whisper = wav2lipaudio = ditto_extractor = None
    torch = None
    frontend_detail = {}
    if args.frontend == "wav2lip":
        import torch as torch_mod
        torch = torch_mod
        model_plan, model, wav2lipaudio, device, frontend_detail = init_wav2lip(plan, torch, repo_root)
        runtime.update({"torch": torch.__version__, "device": str(device)})
    elif args.frontend == "muse":
        import torch as torch_mod
        import transformers as transformers_module
        torch = torch_mod
        import librosa
        model_plan, processor, whisper, device, dtype, frontend_detail = init_muse(plan, torch, transformers_module)
        runtime.update({"torch": torch.__version__, "transformers": transformers_module.__version__, "device": str(device), "dtype": str(dtype)})
    else:
        import librosa
        import torch as torch_mod
        if torch_mod.__version__ != "2.5.1+cu124":
            raise RuntimeError(f"unexpected Ditto runtime Torch version: {torch_mod.__version__}")
        ditto_extractor = None
        model_plan, ditto_extractor, detail = init_ditto(plan)
        frontend_detail.update(detail)
        runtime.update({"torch": torch_mod.__version__, "onnxruntime": importlib.metadata.version("onnxruntime"), "librosa": librosa.__version__, "device": "cpu"})
    runtime["transform_source_files"] = transform_sources

    if model_plan.get("model_id") != model_id:
        raise RuntimeError("model ID mismatch")
    rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        sid = source_row["sample_id"]
        speaker = source_row["speaker_id"]
        source_ref = source_row["natural_pcm"]
        source_path = Path(source_ref["path"])
        samples, read_info = read_wav(source_path, target_rate=SAMPLE_RATE)
        if read_info.file_sha256 != source_ref["file_sha256"] or read_info.pcm_float32_sha256 != source_ref["pcm_sha256"]:
            raise RuntimeError(f"original PCM source digest mismatch for {sid}")
        if samples.size != source_ref["sample_count"] or read_info.source_encoding != "PCM_16" or read_info.source_sample_rate != SAMPLE_RATE or read_info.source_channels != 1:
            raise RuntimeError(f"source PCM format differs from frozen plan for {sid}")
        waveforms, transform_info = build_condition_waveforms(samples, apply_rms, transform_profile, pcm_hash)
        native_features: dict[str, np.ndarray] = {}
        native_maps: dict[str, dict[str, Any]] = {}
        native_details: dict[str, dict[str, Any]] = {}
        for condition in CONDITIONS:
            wave = waveforms[condition]
            temp_path = tmp_dir / f"{sid}__{condition}.wav"
            write_info = write_wav(temp_path, wave, sample_rate=SAMPLE_RATE, peak_ceiling=None, overwrite=False)
            if write_info.pcm_float32_sha256 != transform_info[condition]["pcm_float32_sha256"]:
                raise RuntimeError(f"temporary float WAV changed transformed samples for {sid}/{condition}")
            try:
                if args.frontend == "wav2lip":
                    features, mapping, detail = run_wav2lip_encoder(
                        wave, temp_path, model=model, w2l_audio=wav2lipaudio, device=device, torch=torch
                    )
                elif args.frontend == "muse":
                    features, mapping, detail = run_muse_encoder(
                        wave, temp_path, processor=processor, whisper=whisper, device=device, dtype=dtype, torch=torch,
                        librosa_module=librosa,
                    )
                else:
                    import librosa
                    features, mapping, detail = run_ditto_encoder(
                        wave, temp_path, extractor=ditto_extractor, librosa_module=librosa
                    )
            finally:
                temp_path.unlink(missing_ok=True)
            if features.shape[0] != len(mapping["queries"]) or not np.isfinite(features).all():
                raise RuntimeError(f"native encoder returned invalid features for {sid}/{condition}")
            native_features[condition] = features
            native_maps[condition] = mapping
            native_details[condition] = detail
            if condition != "natural" and features.shape != native_features["natural"].shape:
                raise RuntimeError(f"encoder support changed by condition for {sid}/{condition}")
            if condition != "natural" and mapping != native_maps["natural"]:
                raise RuntimeError(f"source-clock/query support changed by condition for {sid}/{condition}")
            feature_path = out_dir / "features" / f"{sid}__{condition}.npz"
            feature_path.parent.mkdir(exist_ok=True)
            np.savez_compressed(
                feature_path,
                features=features,
                query_index=np.arange(features.shape[0], dtype=np.int32),
                query_timestamp_num=np.arange(features.shape[0], dtype=np.int32),
                query_timestamp_den=np.full(features.shape[0], FPS, dtype=np.int32),
            )
            entry = {
                "sample_id": sid,
                "speaker_id": speaker,
                "condition": condition,
                "status": "PASS",
                "source_pcm": {
                    "path": read_info.path,
                    "wav_sha256": read_info.file_sha256,
                    "pcm_float32_sha256": read_info.pcm_float32_sha256,
                    "sample_count": int(samples.size),
                    "sample_rate_hz": SAMPLE_RATE,
                    "encoding": read_info.source_encoding,
                },
                "condition_waveform": transform_info[condition],
                "feature_array": {
                    "path": str(feature_path),
                    "bytes": feature_path.stat().st_size,
                    "sha256": sha256_file(feature_path),
                    "shape": [int(x) for x in features.shape],
                    "dtype": str(features.dtype),
                },
                "clock_map": mapping,
                "native_frontend_detail": detail,
            }
            if condition == "natural":
                entry["metrics"] = feature_metrics(features, features, samples, wave, FPS)
            else:
                entry["metrics"] = feature_metrics(native_features["natural"], features, samples, wave, FPS)
            rows.append(entry)
        if len({json.dumps(native_maps[c], sort_keys=True) for c in CONDITIONS}) != 1:
            raise RuntimeError(f"query-clock map changed across transform conditions for {sid}")
        # Ensure no temporary enhanced waveform remains after this source row.
        if any(tmp_dir.glob(f"{sid}__*.wav")):
            raise RuntimeError(f"temporary transformed WAV cleanup failed for {sid}")

    shutil.rmtree(tmp_dir)
    if len(rows) != 8 * len(CONDITIONS):
        raise RuntimeError("the complete 8x7 source-condition roster was not produced")
    for sid in sorted({r["sample_id"] for r in rows}):
        if sum(r["sample_id"] == sid for r in rows) != len(CONDITIONS):
            raise RuntimeError(f"incomplete condition roster for {sid}")
    summary = {
        "schema": "cross_tfg_native_audio_encoder_probe_v1",
        "status": "COMPLETE",
        "plan": plan["_raw_binding"],
        "frontend": args.frontend,
        "model_id": model_id,
        "runtime": runtime,
        "scope": {
            "split": "historical_fit_only",
            "source_clip_count": len(source_rows),
            "planned_source_condition_units": 8 * len(CONDITIONS),
            "completed_source_condition_units": len(rows),
            "audio_encoder_forward": True,
            "generator_model_forward": False,
            "video_created_or_read": False,
            "syncnet_forward_or_scores_read": False,
            "onnx_runtime_loaded": args.frontend == "ditto",
            "development_audio_opened": False,
            "confirmation_audio_opened": False,
            "temporary_float32_waveforms_created_and_deleted": True,
            "persistent_transformed_waveforms": False,
            "audio_annotations_or_phoneme_alignments_loaded": False,
        },
        "condition_order": list(CONDITIONS),
        "condition_definitions": {
            "natural": "canonical original mono PCM16 decoded to float32, unchanged",
            "global_minus_6db": "float32 multiply by 10^(-6/20), no limiter",
            "global_plus_6db": "float32 multiply by 10^(+6/20), no limiter",
            "fixed_rms_0p0376838173": "plan fixed RMS scalar, no peak ceiling",
            PROFILE_IDS[0]: "frozen shared-frequency profile module, ratio 0.90 local compression, RMS target, peak ceiling 0.98",
            PROFILE_IDS[1]: "frozen shared-frequency profile module, ratio 0.90 local compression, RMS target, peak ceiling 0.98",
            PROFILE_IDS[2]: "frozen shared-frequency profile module, ratio 0.90 local compression, RMS target, peak ceiling 0.98",
        },
        "phase_note": "The three frozen shared-frequency profiles first apply local compression, then preserve the phase of that compressed waveform's STFT during the frequency mask. They are not strict original-waveform phase-preserving controls.",
        "runtime_source_details": frontend_detail,
        "row_count": len(rows),
        "rows": rows,
        "equal_clip_aggregates": aggregate_metrics(rows),
        "interpretation_limit": "This measures learned audio-encoder representation changes only. It does not establish mouth motion or synchronization effects. The 40 ms natural-envelope association is an index-clock diagnostic, not an encoder receptive-field claim. No phoneme annotations were loaded.",
        "elapsed_seconds": time.time() - start_time,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output_files = []
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
        output_files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    receipt = {
        "schema": "cross_tfg_native_audio_encoder_probe_receipt_v1",
        "status": "COMPLETE",
        "summary_path": str(summary_path),
        "summary_bytes": summary_path.stat().st_size,
        "summary_sha256": sha256_file(summary_path),
        "output_file_count": len(output_files),
        "outputs": output_files,
        "scope": summary["scope"],
    }
    receipt_path = out_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", choices=tuple(EXPECTED_MODELS), required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    try:
        receipt = execute(args)
        print(json.dumps({"status": receipt["status"], "summary_sha256": receipt["summary_sha256"], "outputs": receipt["output_file_count"]}, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}, sort_keys=True), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
