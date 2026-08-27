from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .mfa_alignment import FrameMapping, extend_final_token_for_feature_tail

SAMPLE_RATE = 16_000
HOP_SAMPLES = 320
FEATURE_DIM = 1024


def validate_wavlm_interface(
    interface: Mapping[str, Any],
    *,
    revision: str,
    wavlm_checkpoint_sha256: str,
    vocoder_checkpoint_sha256: str,
) -> dict[str, Any]:
    required = {
        "repository": "bshall/knn-vc",
        "revision": revision,
        "sample_rate": SAMPLE_RATE,
        "frame_stride_samples": HOP_SAMPLES,
        "selected_layer": 6,
        "feature_dim": FEATURE_DIM,
        "prematched_vocoder": True,
        "frozen": True,
        "loudness_normalization": False,
        "wavlm_checkpoint_sha256": wavlm_checkpoint_sha256,
        "vocoder_checkpoint_sha256": vocoder_checkpoint_sha256,
    }
    for key, expected in required.items():
        if interface.get(key) != expected:
            raise ValueError(f"WavLM/vocoder interface mismatch: {key}")
    return dict(required)


def exact_natural_length(values: np.ndarray, count: int) -> tuple[np.ndarray, dict[str, Any]]:
    if count <= 0:
        raise ValueError("target sample count must be positive")
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size > count:
        output = array[:count].copy()
        action = "right_crop"
    elif array.size < count:
        output = np.pad(array, (0, count - array.size)).astype(np.float32)
        action = "right_zero_pad"
    else:
        output = array.copy()
        action = "none"
    return output, {"raw_samples": int(array.size), "target_samples": int(count), "action": action, "adjustment_samples": int(array.size - count)}


def interpolate_conditioning(tts_features: np.ndarray, mapping: Sequence[FrameMapping]) -> np.ndarray:
    features = np.asarray(tts_features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] <= 0 or features.shape[1] <= 0:
        raise ValueError("TTS features must be a non-empty 2D array")
    if not mapping:
        raise ValueError("frame mapping is empty")
    natural_indices = [int(item.natural_frame_index) for item in mapping]
    if natural_indices != list(range(len(mapping))):
        raise ValueError("frame mapping must contain each natural frame exactly once in order")
    rows: list[np.ndarray] = []
    for item in mapping:
        if not 0 <= item.left_frame_index < features.shape[0] or not 0 <= item.right_frame_index < features.shape[0]:
            raise ValueError("mapping references a TTS frame outside feature array")
        alpha = float(item.interpolation_alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("mapping interpolation coefficient is invalid")
        rows.append(features[item.left_frame_index] + alpha * (features[item.right_frame_index] - features[item.left_frame_index]))
    if not rows:
        raise ValueError("frame mapping is empty")
    return np.stack(rows).astype(np.float32, copy=False)


def validate_raw_vocoder_length(values: np.ndarray, conditioning_frame_count: int, *, hop_samples: int = HOP_SAMPLES) -> dict[str, int]:
    array = np.asarray(values).reshape(-1)
    if conditioning_frame_count <= 0 or hop_samples <= 0:
        raise ValueError("conditioning frame count and hop must be positive")
    expected = conditioning_frame_count * hop_samples
    if array.size != expected:
        raise ValueError(f"raw vocoder length {array.size} does not match model contract {expected}")
    return {"conditioning_frames": int(conditioning_frame_count), "hop_samples": int(hop_samples), "raw_samples": int(array.size), "expected_samples": int(expected)}


def validate_waveform(values: np.ndarray, expected_samples: int, *, require_nonzero: bool = True) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size != expected_samples:
        raise ValueError(f"waveform sample count {array.size} != {expected_samples}")
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("waveform is empty or non-finite")
    if require_nonzero and not np.any(array != 0.0):
        raise ValueError("waveform is all zero")
    if np.any(np.abs(array) > 1.0 + 1e-6):
        raise ValueError("waveform contains out-of-range samples")
    clipped = int(np.count_nonzero(np.abs(array) >= 1.0))
    return {
        "samples": int(array.size),
        "peak": float(np.max(np.abs(array))),
        "rms": float(np.sqrt(np.mean(array * array))),
        "clipped_sample_count": clipped,
        "finite": True,
    }


def canonical_pcm_s16le(values: np.ndarray, expected_samples: int) -> tuple[bytes, dict[str, Any]]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    qc = validate_waveform(array, expected_samples)
    if qc["clipped_sample_count"]:
        raise ValueError("canonical PCM conversion refuses clipped samples")
    pcm = np.rint(array * 32767.0).astype("<i2", copy=False).tobytes()
    if len(pcm) != expected_samples * 2:
        raise ValueError("canonical PCM byte length mismatch")
    readback = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32767.0
    if readback.size != expected_samples:
        raise ValueError("canonical PCM readback sample count mismatch")
    return pcm, {**qc, "sample_rate_hz": SAMPLE_RATE, "channels": 1, "subtype": "PCM_16", "pcm_sha256": hashlib.sha256(pcm).hexdigest(), "pcm_bytes": len(pcm)}


def cosine_distance(output_features: np.ndarray, conditioning: np.ndarray) -> float:
    output = np.asarray(output_features, dtype=np.float64).reshape(-1)
    target = np.asarray(conditioning, dtype=np.float64).reshape(-1)
    if output.shape != target.shape or not np.isfinite(output).all() or not np.isfinite(target).all():
        raise ValueError("cosine-distance inputs must have equal finite shapes")
    output_norm = float(np.linalg.norm(output))
    target_norm = float(np.linalg.norm(target))
    if output_norm == 0.0 or target_norm == 0.0:
        raise ValueError("cosine-distance inputs must be non-zero")
    similarity = float(np.dot(output, target) / (output_norm * target_norm))
    return float(1.0 - max(-1.0, min(1.0, similarity)))


def candidate_from_features(
    *,
    natural_features: np.ndarray,
    tts_features: np.ndarray,
    mapping: Sequence[FrameMapping],
    natural_audio_samples: int,
    vocode: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    natural = np.asarray(natural_features)
    if natural.ndim != 2 or natural.shape[0] != len(mapping) or natural.shape[1] != FEATURE_DIM:
        raise ValueError("natural features must be [frames,1024] and match frame mapping count")
    tts_array = np.asarray(tts_features)
    if tts_array.ndim != 2 or tts_array.shape[1] != FEATURE_DIM:
        raise ValueError("TTS features must be [frames,1024]")
    conditioning = interpolate_conditioning(tts_array, mapping)
    if conditioning.shape != natural.shape:
        raise ValueError("conditioning shape does not match natural feature shape")
    raw = np.asarray(vocode(conditioning), dtype=np.float32).reshape(-1)
    raw_contract = validate_raw_vocoder_length(raw, conditioning.shape[0])
    output, length_adjustment = exact_natural_length(raw, natural_audio_samples)
    waveform_qc = validate_waveform(output, natural_audio_samples)
    return output, {
        "conditioning_frames": int(conditioning.shape[0]),
        "conditioning_dim": int(conditioning.shape[1]),
        "raw_vocoder": raw_contract,
        "length_adjustment": length_adjustment,
        "waveform_qc": waveform_qc,
        "conditioning_sha256": hashlib.sha256(conditioning.astype("<f4", copy=False).tobytes()).hexdigest(),
        "raw_waveform_sha256": hashlib.sha256(raw.astype("<f4", copy=False).tobytes()).hexdigest(),
        "canonical_waveform_sha256": hashlib.sha256(output.astype("<f4", copy=False).tobytes()).hexdigest(),
        "loudness_normalization": False,
    }


__all__ = [
    "SAMPLE_RATE",
    "HOP_SAMPLES",
    "exact_natural_length",
    "interpolate_conditioning",
    "validate_raw_vocoder_length",
    "validate_waveform",
    "canonical_pcm_s16le",
    "cosine_distance",
    "candidate_from_features",
    "extend_final_token_for_feature_tail",
]
