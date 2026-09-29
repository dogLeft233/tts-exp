from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_roi_retiming_oracle.common import read_pcm16_wav, write_pcm16_wav

from . import config
from .common import ExperimentError, bytes_sha256, file_sha256


def shift_pcm(pcm: bytes, direction: str, shift: int = config.AUDIO_SHIFT) -> bytes:
    """Apply the registered zero-padded integer sample shift."""
    if len(pcm) == 0 or len(pcm) % 2:
        raise ExperimentError("PCM must be non-empty and int16 aligned")
    if shift < 1:
        raise ExperimentError("shift must be positive")
    values = np.frombuffer(pcm, dtype="<i2")
    if direction == config.AUDIO_DELAY:
        shifted = np.concatenate((np.zeros(min(shift, values.size), dtype="<i2"), values[: max(0, values.size - shift)]))
    elif direction == config.AUDIO_ADVANCE:
        shifted = np.concatenate((values[min(shift, values.size) :], np.zeros(min(shift, values.size), dtype="<i2")))
    else:
        raise ExperimentError(f"unknown shift direction: {direction}")
    if shifted.size != values.size:
        raise ExperimentError(f"shift changed PCM length: {shifted.size} != {values.size}")
    return np.ascontiguousarray(shifted, dtype="<i2").tobytes()


def assert_shift_contract(natural: bytes, delayed: bytes, advanced: bytes, shift: int = config.AUDIO_SHIFT) -> None:
    if len(natural) != len(delayed) or len(natural) != len(advanced):
        raise ExperimentError("shifted PCM length differs from natural PCM")
    if delayed != shift_pcm(natural, config.AUDIO_DELAY, shift):
        raise ExperimentError("DELAY_200 PCM does not match the exact zero-padded contract")
    if advanced != shift_pcm(natural, config.AUDIO_ADVANCE, shift):
        raise ExperimentError("ADVANCE_200 PCM does not match the exact zero-padded contract")


def write_arm(path: Path, pcm: bytes, *, construction: str, source_pcm_sha256: str) -> dict[str, Any]:
    if path.exists():
        existing, _values, _params = read_pcm16_wav(path)
        if existing != pcm:
            raise ExperimentError(f"existing audio artifact differs: {path}")
    else:
        write_pcm16_wav(path, pcm)
    decoded, values, params = read_pcm16_wav(path)
    if decoded != pcm or values.size != len(pcm) // 2:
        raise ExperimentError(f"audio write/readback failed: {path}")
    return {
        "path": str(path.resolve()),
        "container_sha256": file_sha256(path),
        "decoded_pcm_sha256": bytes_sha256(decoded),
        "sample_count": int(values.size),
        "sample_rate": int(params["sample_rate"]),
        "channels": int(params["channels"]),
        "sample_width": int(params["sample_width"]),
        "construction": construction,
        "source_pcm_sha256": source_pcm_sha256,
        "length_preserved": True,
    }


def build_arms(natural_pcm: bytes, output_dir: Path, sample_id: str) -> dict[str, dict[str, Any]]:
    source_hash = bytes_sha256(natural_pcm)
    delayed = shift_pcm(natural_pcm, config.AUDIO_DELAY)
    advanced = shift_pcm(natural_pcm, config.AUDIO_ADVANCE)
    assert_shift_contract(natural_pcm, delayed, advanced)
    payloads = {
        config.AUDIO_N: (natural_pcm, "byte-identical untouched natural PCM16"),
        config.AUDIO_N_REPEAT: (natural_pcm, "independent N_REPEAT generation call using identical N PCM"),
        config.AUDIO_DELAY: (delayed, "exact DELAY_200: N[n-3200], zero outside range"),
        config.AUDIO_ADVANCE: (advanced, "exact ADVANCE_200: N[n+3200], zero outside range"),
    }
    result: dict[str, dict[str, Any]] = {}
    for arm, (pcm, construction) in payloads.items():
        result[arm] = write_arm(output_dir / f"{sample_id}__{arm}.wav", pcm, construction=construction, source_pcm_sha256=source_hash)
    if result[config.AUDIO_N]["decoded_pcm_sha256"] != result[config.AUDIO_N_REPEAT]["decoded_pcm_sha256"]:
        raise ExperimentError(f"N/N_REPEAT PCM mismatch: {sample_id}")
    return result
