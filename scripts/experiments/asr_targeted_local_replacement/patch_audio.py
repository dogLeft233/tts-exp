from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import signal


SAMPLE_RATE = 16000


def read_pcm16(path: str | Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    values, sample_rate = sf.read(str(path), dtype="int16", always_2d=True)
    if int(sample_rate) != SAMPLE_RATE or values.shape[1] != 1:
        raise ValueError(f"audio must be mono 16 kHz: {path}")
    samples = np.asarray(values[:, 0], dtype=np.int16)
    if samples.size == 0:
        raise ValueError(f"audio is empty: {path}")
    return samples, int(sample_rate)


def pcm16_to_float(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples)
    if values.dtype != np.int16 or values.ndim != 1:
        raise ValueError("PCM input must be one-dimensional int16")
    return values.astype(np.float32) / np.float32(32768.0)


def float_to_pcm16(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("candidate audio must be non-empty and finite")
    if np.any(array < -1.0) or np.any(array > 1.0):
        raise ValueError("candidate audio clips before PCM write")
    quantized = np.rint(array * 32768.0)
    if np.any(quantized < -32768) or np.any(quantized > 32767):
        raise ValueError("candidate audio clips during PCM quantization")
    return quantized.astype(np.int16)


def write_pcm16(path: str | Path, values: np.ndarray, *, sample_rate: int = SAMPLE_RATE) -> dict[str, Any]:
    import soundfile as sf

    if int(sample_rate) != SAMPLE_RATE:
        raise ValueError("only 16 kHz PCM16 is allowed")
    samples = values if np.asarray(values).dtype == np.int16 else float_to_pcm16(values)
    samples = np.asarray(samples, dtype=np.int16).reshape(-1)
    if samples.size == 0:
        raise ValueError("cannot write empty PCM")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), samples, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    decoded, decoded_rate = read_pcm16(destination)
    if decoded_rate != SAMPLE_RATE or not np.array_equal(decoded, samples):
        raise ValueError("PCM writer failed exact int16 round-trip")
    return {"path": str(destination), "sample_rate_hz": SAMPLE_RATE, "channels": 1, "sample_count": int(samples.size)}


def _ramp_mask(length: int, ramp: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("destination interval is empty")
    mask = np.ones(length, dtype=np.float64)
    if ramp <= 0:
        return mask
    if ramp == 1:
        return mask
    phase = np.arange(ramp, dtype=np.float64) / float(ramp - 1)
    rising = 0.5 - 0.5 * np.cos(np.pi * phase)
    mask[:ramp] = rising
    mask[-ramp:] = rising[::-1]
    return mask


def resample_donor(donor: np.ndarray, destination_samples: int) -> np.ndarray:
    source = np.asarray(donor, dtype=np.float64).reshape(-1)
    if source.size == 0 or not np.all(np.isfinite(source)):
        raise ValueError("donor interval must be non-empty and finite")
    if int(destination_samples) <= 0:
        raise ValueError("destination sample count must be positive")
    result = np.asarray(signal.resample(source, int(destination_samples)), dtype=np.float64)
    if result.size != int(destination_samples) or not np.all(np.isfinite(result)):
        raise ValueError("resampler returned invalid donor")
    return result


def _validate_patch(patch: Mapping[str, Any], natural_count: int, tts_count: int) -> tuple[int, int, int, int]:
    d0, d1 = int(patch["destination_start_sample"]), int(patch["destination_end_sample"])
    s0, s1 = int(patch["donor_start_sample"]), int(patch["donor_end_sample"])
    if not 0 <= d0 < d1 <= natural_count:
        raise ValueError("destination interval is outside natural waveform")
    if not 0 <= s0 < s1 <= tts_count:
        raise ValueError("donor interval is outside TTS waveform")
    if int(patch.get("destination_samples", d1 - d0)) != d1 - d0 or int(patch.get("donor_samples", s1 - s0)) != s1 - s0:
        raise ValueError("patch sample counts do not match bounds")
    return d0, d1, s0, s1


def apply_patches(natural: np.ndarray, tts: np.ndarray, patches: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    natural_float = pcm16_to_float(np.asarray(natural, dtype=np.int16)) if np.asarray(natural).dtype == np.int16 else np.asarray(natural, dtype=np.float64).reshape(-1)
    tts_float = pcm16_to_float(np.asarray(tts, dtype=np.int16)) if np.asarray(tts).dtype == np.int16 else np.asarray(tts, dtype=np.float64).reshape(-1)
    if natural_float.size == 0 or tts_float.size == 0 or not np.all(np.isfinite(natural_float)) or not np.all(np.isfinite(tts_float)):
        raise ValueError("natural and TTS waveforms must be non-empty and finite")
    ordered = sorted((dict(patch) for patch in patches), key=lambda row: (int(row["destination_start_sample"]), int(row["destination_end_sample"]), str(row.get("block_id", ""))))
    result = np.array(natural_float, dtype=np.float64, copy=True)
    previous_end = -1
    changed_bounds: list[tuple[int, int]] = []
    for patch in ordered:
        d0, d1, s0, s1 = _validate_patch(patch, natural_float.size, tts_float.size)
        if d0 < previous_end:
            raise ValueError("patch destinations overlap")
        previous_end = d1
        donor = resample_donor(tts_float[s0:s1], d1 - d0)
        ramp = min(320, (d1 - d0) // 4)
        mask = _ramp_mask(d1 - d0, ramp)
        result[d0:d1] = (1.0 - mask) * result[d0:d1] + mask * donor
        patch["ramp_samples"] = int(ramp)
        patch["resampler"] = "scipy.signal.resample"
        patch["destination_samples"] = int(d1 - d0)
        patch["donor_samples"] = int(s1 - s0)
        changed = np.flatnonzero(result[d0:d1] != natural_float[d0:d1])
        if changed.size:
            changed_bounds.append((d0 + int(changed[0]), d0 + int(changed[-1]) + 1))
    if not np.all(np.isfinite(result)) or np.any(result < -1.0) or np.any(result > 1.0):
        raise ValueError("patched candidate is non-finite or clips")
    if ordered:
        mask = np.zeros(natural_float.size, dtype=bool)
        for patch in ordered:
            mask[int(patch["destination_start_sample"]):int(patch["destination_end_sample"])] = True
        if np.any(result[~mask] != natural_float[~mask]):
            raise ValueError("candidate changed samples outside destination mask")
    return result.astype(np.float64), {
        "patch_count": len(ordered),
        "patches": ordered,
        "changed_sample_count_prequantization": int(np.count_nonzero(result != natural_float)),
        "changed_sample_bounds": [[int(start), int(end)] for start, end in changed_bounds],
        "natural_sample_count": int(natural_float.size),
    }


def build_condition_audio(natural: np.ndarray, tts: np.ndarray, condition: str, patches: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    if condition == "natural":
        if patches:
            raise ValueError("natural identity condition cannot contain patches")
        result = pcm16_to_float(np.asarray(natural, dtype=np.int16)) if np.asarray(natural).dtype == np.int16 else np.asarray(natural, dtype=np.float64).reshape(-1)
        return result.astype(np.float64), {"condition": condition, "patch_count": 0, "identity": True, "natural_sample_count": int(result.size)}
    if condition not in ("asr_targeted", "target_control_0", "target_control_1"):
        raise ValueError(f"unregistered condition: {condition}")
    result, meta = apply_patches(natural, tts, patches)
    return result, {"condition": condition, "identity": False, **meta}


def validate_condition_output(path: str | Path, natural_path: str | Path, expected_count: int) -> dict[str, Any]:
    actual, rate = read_pcm16(path)
    natural, natural_rate = read_pcm16(natural_path)
    if rate != SAMPLE_RATE or natural_rate != SAMPLE_RATE or actual.size != int(expected_count) or actual.size != natural.size:
        raise ValueError("candidate PCM violates exact natural length contract")
    if not np.all(np.isfinite(actual.astype(np.float32))):
        raise ValueError("candidate PCM is non-finite")
    return {"sample_count": int(actual.size), "sample_rate_hz": rate, "channels": 1, "pcm16_sha256": __import__("hashlib").sha256(actual.tobytes()).hexdigest()}
