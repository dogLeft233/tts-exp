"""Frozen Wav2Lip-mel and phone-phase WavLM feature operations."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

from .config import (
    MEL_BINS,
    MEL_HOP_SAMPLES,
    MIN_TTS_FRAMES,
    NATURAL_SUPPORT_SAMPLES,
    SAMPLE_RATE,
    WAVLM_DIM,
    WINDOW_FRAMES,
)
from .protocol import canonical_json, sha256_file, sha256_text


def _wav2lip_audio_module():
    import sys

    vendor_dir = Path(__file__).resolve().parents[3] / "third_party" / "Wav2Lip"
    if str(vendor_dir) not in sys.path:
        sys.path.insert(0, str(vendor_dir))
    import audio

    return audio


def extract_natural_mel(audio_path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError(f"natural audio must be {SAMPLE_RATE} Hz: {audio_path}")
    waveform = np.asarray(waveform)
    if waveform.ndim != 1:
        raise ValueError(f"natural audio must be mono: {audio_path}")
    if waveform.shape[0] < NATURAL_SUPPORT_SAMPLES:
        raise ValueError(f"natural audio is shorter than frozen support: {audio_path}")
    waveform = np.ascontiguousarray(waveform[:NATURAL_SUPPORT_SAMPLES], dtype=np.float32)
    mel = np.asarray(_wav2lip_audio_module().melspectrogram(waveform), dtype=np.float32)
    if mel.ndim != 2 or mel.shape[0] != MEL_BINS or not np.isfinite(mel).all():
        raise ValueError(f"invalid Wav2Lip mel output for {audio_path}")
    return mel


def load_tts_feature(feature_path: Path) -> np.ndarray:
    values = np.asarray(np.load(feature_path, allow_pickle=False))
    if values.ndim != 2 or values.shape[1] != WAVLM_DIM or values.shape[0] < MIN_TTS_FRAMES:
        raise ValueError(f"TTS WavLM-L6 must have shape [N,{WAVLM_DIM}] with N >= {MIN_TTS_FRAMES}: {feature_path}")
    if values.dtype == object or not np.isfinite(values).all():
        raise ValueError(f"TTS WavLM-L6 is not finite numeric data: {feature_path}")
    return np.ascontiguousarray(values, dtype=np.float32)


def extract_selected_features(lock: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    natural: dict[str, np.ndarray] = {}
    tts: dict[str, np.ndarray] = {}
    for row in lock["records"]:
        sid = str(row["sample_id"])
        natural[sid] = extract_natural_mel(Path(str(row["paths"]["natural_audio"])))
        tts[sid] = load_tts_feature(Path(str(row["paths"]["tts_feature"])))
    return natural, tts


def fit_normalization(
    natural_mels: Mapping[str, np.ndarray],
    tts_features: Mapping[str, np.ndarray],
    train_ids: set[str],
) -> dict[str, Any]:
    if not train_ids:
        raise ValueError("normalization requires train IDs")
    natural_rows = np.concatenate([np.asarray(natural_mels[sid], dtype=np.float64).T for sid in sorted(train_ids)], axis=0)
    tts_rows = np.concatenate([np.asarray(tts_features[sid], dtype=np.float64) for sid in sorted(train_ids)], axis=0)
    mel_mean = natural_rows.mean(axis=0)
    mel_std = natural_rows.std(axis=0)
    tts_mean = tts_rows.mean(axis=0)
    tts_std = tts_rows.std(axis=0)
    mel_std = np.maximum(mel_std, 1e-6)
    tts_std = np.maximum(tts_std, 1e-6)
    result = {
        "schema_version": 1,
        "source": "frozen_train_groups_only",
        "train_ids": sorted(train_ids),
        "mel_mean": mel_mean.tolist(),
        "mel_std": mel_std.tolist(),
        "tts_mean": tts_mean.tolist(),
        "tts_std": tts_std.tolist(),
        "mel_std_floor": 1e-6,
        "tts_std_floor": 1e-6,
    }
    result["sha256"] = sha256_text(canonical_json(result))
    return result


def _stats_array(stats: Mapping[str, Any], key: str, length: int) -> np.ndarray:
    values = np.asarray(stats[key], dtype=np.float32)
    if values.shape != (length,) or not np.isfinite(values).all():
        raise ValueError(f"invalid normalization statistic {key}")
    return values


def standardize_mel(mel: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(mel, dtype=np.float32)
    mean = _stats_array(stats, "mel_mean", MEL_BINS)
    std = _stats_array(stats, "mel_std", MEL_BINS)
    return (values - mean[:, None]) / std[:, None]


def standardize_tts(values: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    features = np.asarray(values, dtype=np.float32)
    mean = _stats_array(stats, "tts_mean", WAVLM_DIM)
    std = _stats_array(stats, "tts_std", WAVLM_DIM)
    return (features - mean[None, :]) / std[None, :]


def phone_phase_linear(
    source: np.ndarray,
    source_start: int,
    source_end: int,
    destination_length: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values = np.asarray(source)
    if values.ndim != 2 or values.shape[1] != WAVLM_DIM:
        raise ValueError("source trajectory must have shape [N,1024]")
    if source_start < 0 or source_end > values.shape[0] or source_end - source_start < MIN_TTS_FRAMES:
        raise ValueError("source phone does not satisfy frozen WavLM support")
    if destination_length < 1:
        raise ValueError("destination phone must contain at least one frame")
    selected = np.asarray(values[source_start:source_end], dtype=np.float32)
    n = selected.shape[0]
    result = np.empty((destination_length, WAVLM_DIM), dtype=np.float32)
    provenance: list[dict[str, Any]] = []
    for i in range(destination_length):
        u = (i + 0.5) / destination_length
        unclipped = u * n - 0.5
        x = float(np.clip(unclipped, 0.0, n - 1.0))
        left = int(np.floor(x))
        right = min(left + 1, n - 1)
        alpha = float(x - left)
        result[i] = (1.0 - alpha) * selected[left] + alpha * selected[right]
        provenance.append({
            "destination_index": i,
            "destination_phase": u,
            "source_coordinate_unclipped": unclipped,
            "source_coordinate": x,
            "source_index_left": source_start + left,
            "source_index_right": source_start + right,
            "alpha": alpha,
            "clipped": bool(x != unclipped),
        })
    if not np.isfinite(result).all():
        raise FloatingPointError("phone phase interpolation produced non-finite data")
    return result, provenance


def build_example(
    mask: Mapping[str, Any],
    natural_mels: Mapping[str, np.ndarray],
    tts_features: Mapping[str, np.ndarray],
    stats: Mapping[str, Any],
    *,
    tts_mask: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray | dict[str, Any]]:
    sid = str(mask["sample_id"])
    mel = np.asarray(natural_mels[sid], dtype=np.float32)
    normalized = standardize_mel(mel, stats)
    window_start = int(mask["window_start_frame"])
    window = normalized[:, window_start:window_start + WINDOW_FRAMES].T.copy()
    target = window.copy()
    core_start, core_end = int(mask["core_start"]), int(mask["core_end"])
    mask_start, mask_end = int(mask["mask_start"]), int(mask["mask_end"])
    masked = window.copy()
    masked[mask_start:mask_end] = 0.0
    support = np.zeros((WINDOW_FRAMES, 1), dtype=np.float32)
    core = np.zeros((WINDOW_FRAMES, 1), dtype=np.float32)
    support[mask_start:mask_end, 0] = 1.0
    core[core_start:core_end, 0] = 1.0
    source_mask = mask if tts_mask is None else tts_mask
    source_id = str(source_mask["sample_id"])
    tts = standardize_tts(tts_features[source_id], stats)
    aligned_core, provenance = phone_phase_linear(
        tts,
        int(source_mask["tts_frame_start"]),
        int(source_mask["tts_frame_end"]),
        core_end - core_start,
    )
    aligned = np.zeros((WINDOW_FRAMES, WAVLM_DIM), dtype=np.float32)
    aligned[core_start:core_end] = aligned_core
    result: dict[str, np.ndarray | dict[str, Any]] = {
        "natural_mel": np.ascontiguousarray(masked, dtype=np.float32),
        "masked_support": support,
        "target_core": core,
        "tts_features": np.ascontiguousarray(aligned, dtype=np.float32),
        "target": np.ascontiguousarray(target, dtype=np.float32),
        "provenance": {"source_sample_id": source_id, "source_mask_sha256": source_mask["mask_sha256"], "frames": provenance},
    }
    for key in ("natural_mel", "masked_support", "target_core", "tts_features", "target"):
        value = np.asarray(result[key])
        if not np.isfinite(value).all():
            raise FloatingPointError(f"non-finite example field {key}")
    if np.any(aligned[np.asarray(core[:, 0]) == 0] != 0):
        raise ValueError("aligned TTS feature has nonzero values outside target core")
    return result


def feature_contract(lock: Mapping[str, Any]) -> dict[str, Any]:
    audio = _wav2lip_audio_module()
    import hparams as hparams_module

    module_paths = [Path(audio.__file__).resolve(), Path(hparams_module.__file__).resolve()]
    return {
        "wav2lip_audio": str(module_paths[0]),
        "wav2lip_audio_sha256": sha256_file(module_paths[0]),
        "wav2lip_hparams": str(module_paths[1]),
        "wav2lip_hparams_sha256": sha256_file(module_paths[1]),
        "sample_rate": SAMPLE_RATE,
        "natural_support_samples": NATURAL_SUPPORT_SAMPLES,
        "mel_bins": MEL_BINS,
        "mel_hop_samples": MEL_HOP_SAMPLES,
        "mel_frame_center_seconds": "m*200/16000",
        "wavlm_stride_samples": 320,
        "wavlm_dimension": WAVLM_DIM,
        "alignment": "phone_phase_linear_v1",
    }
