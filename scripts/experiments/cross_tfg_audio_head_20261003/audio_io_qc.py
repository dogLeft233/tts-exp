#!/usr/bin/env python3
"""Actual local acoustic I/O QC for the cross-TFG enhancement boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import wave
from pathlib import Path

import numpy as np

from scripts.audio_enhancement.audio_io import read_canonical_wav, write_float32_wav
from scripts.audio_enhancement.baselines import (
    FIXED_POSITIVE_CONTROL_RMS,
    SpectralProfile,
    apply_flat_gain,
    apply_global_rms,
    apply_local_envelope_compression,
    apply_spectral_profile,
    waveform_rms,
)
from scripts.audio_enhancement.pipeline import transform


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_pcm16_stereo(path: Path, rate: int = 48_000, seconds: float = 0.5) -> None:
    count = int(rate * seconds)
    index = np.arange(count, dtype=np.float64) / rate
    left = 0.24 * np.sin(2.0 * math.pi * 440.0 * index)
    right = 0.14 * np.sin(2.0 * math.pi * 660.0 * index + 0.2)
    interleaved = np.stack((left, right), axis=1)
    pcm = np.rint(np.clip(interleaved, -1.0, 32767.0 / 32768.0) * 32768.0).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = out / "synthetic_stereo_48k_pcm16.wav"
    natural = out / "canonical_natural_16k_f32.wav"
    roundtrip = out / "identity_roundtrip_16k_f32.wav"
    write_pcm16_stereo(source)
    canonical, read_info = read_canonical_wav(source)
    if canonical.dtype != np.float32 or canonical.ndim != 1 or canonical.size != 8_000:
        raise RuntimeError("48 kHz stereo source did not canonicalize to 8,000 mono samples")
    natural_info = write_float32_wav(natural, canonical, sample_rate=16_000)
    natural_readback, natural_read_info = read_canonical_wav(natural)
    identity = transform(canonical, {"method": "identity"})
    identity_info = write_float32_wav(roundtrip, identity, sample_rate=16_000)
    if not np.array_equal(canonical, identity):
        raise RuntimeError("identity transform is not sample-bit exact")
    restored, restored_info = read_canonical_wav(roundtrip)
    if not np.array_equal(identity, restored):
        raise RuntimeError("float32 WAV readback changed identity samples")
    if not np.array_equal(canonical, natural_readback):
        raise RuntimeError("canonical natural reference WAV changed source PCM")

    rms_audio, rms_info = apply_global_rms(canonical)
    if abs(rms_info.output_rms - FIXED_POSITIVE_CONTROL_RMS) > 2e-8:
        raise RuntimeError("fixed RMS positive control missed its declared target")
    flat_audio, flat_info = apply_flat_gain(canonical, 3.0, max_abs_gain_db=6.0)
    local_audio, local_info = apply_local_envelope_compression(canonical)
    profile_audio = apply_spectral_profile(
        canonical,
        SpectralProfile(
            global_gain_db=1.0,
            band_gains_db=(-1.0, 0.0, 1.0),
            envelope_gains_db=(0.0, 1.0, 0.0, -1.0),
        ),
    )
    silence = np.zeros_like(canonical)
    silence_out = transform(silence, {"method": "local_compressor"})
    if not np.array_equal(silence, silence_out):
        raise RuntimeError("local envelope compressor changed exact silence")
    arrays = {"rms": rms_audio, "flat_gain": flat_audio, "local_compressor": local_audio, "profile": profile_audio}
    for name, array in arrays.items():
        if array.dtype != np.float32 or array.shape != canonical.shape or not np.isfinite(array).all():
            raise RuntimeError(f"{name} broke float32/length/finite contract")

    receipt = {
        "schema": "cross_tfg_audio_io_qc_v1",
        "status": "PASS",
        "scope": "synthetic 48 kHz stereo PCM16 canonicalization and local transform/file I/O only; no TFG/model/scoring data",
        "source": {
            "path": str(source),
            "file_sha256": sha256_file(source),
            "bytes": source.stat().st_size,
            "sample_rate_hz": read_info.source_sample_rate,
            "channels": read_info.source_channels,
            "source_frames": read_info.source_frames,
            "canonical_frames": int(canonical.size),
            "canonical_pcm_float32_sha256": read_info.pcm_float32_sha256,
            "resampler": read_info.resampler,
        },
        "natural_reference": {
            "path": str(natural),
            "file_sha256": natural_info.file_sha256,
            "pcm_float32_sha256": natural_info.pcm_float32_sha256,
            "readback_pcm_sha256": natural_read_info.pcm_float32_sha256,
        },
        "identity": {
            "path": str(roundtrip),
            "file_sha256": identity_info.file_sha256,
            "pcm_float32_sha256": identity_info.pcm_float32_sha256,
            "bit_exact_to_canonical": bool(np.array_equal(identity, canonical)),
            "file_readback_bit_exact": bool(np.array_equal(identity, restored)),
        },
        "fixed_rms": {
            "target_rms": FIXED_POSITIVE_CONTROL_RMS,
            "input_rms": rms_info.input_rms,
            "output_rms": rms_info.output_rms,
            "requested_gain": rms_info.requested_gain,
            "applied_gain": rms_info.applied_gain,
            "peak_limited": rms_info.peak_limited,
        },
        "flat_gain": {"gain_db": flat_info.gain_db, "input_rms": flat_info.input_rms,
                       "output_rms": flat_info.output_rms, "applied_gain": flat_info.applied_gain},
        "local_compressor": {"window_samples": local_info.envelope_window_samples,
                              "input_rms": local_info.input_rms, "output_rms": local_info.output_rms,
                              "min_gain_db": local_info.minimum_gain_db,
                              "max_gain_db": local_info.maximum_gain_db,
                              "silence_unchanged": local_info.silence_unchanged},
        "profile": {"output_rms": waveform_rms(profile_audio), "sample_count": int(profile_audio.size)},
        "silence_exact": bool(np.array_equal(silence, silence_out)),
        "all_outputs_same_length_finite_float32": True,
    }
    receipt_path = out / "audio_io_qc_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "receipt_path": str(receipt_path),
                      "receipt_sha256": sha256_file(receipt_path),
                      "canonical_frames": canonical.size}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
