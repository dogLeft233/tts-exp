"""Prepared r=.90 local-compression control for the expanded fit-only slate.

This module exposes a callable transform and its JSON-safe config.  Nothing in
this module is executed by import; callers must explicitly invoke the transform
after the shared540 gate and method freeze.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "cross_tfg_fit40_localcompress_r090_control_v1"
METHOD_ID = "localcompress_r090"
SAMPLE_RATE_HZ = 16_000
TARGET_RMS = 0.0376838172675746
LOCAL_WINDOW_MS = 32.0
LOCAL_RATIO = 0.90
LOCAL_MIN_GAIN_DB = -6.0
LOCAL_MAX_GAIN_DB = 6.0
ENVELOPE_FLOOR = 1e-5
EXPECTED_BASELINES_SHA256 = "7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2"
EXPECTED_SHARED_PROFILE_SHA256 = "fd7303f513fdbada80da50e5dd395e9441142f1658544765d7523f1618f55253"
EXPECTED_STAGEA_CORE_SHA256 = "d12c863ffcd5eacde7b0d09b0abc097429d0318ccea3b059d022d5b2e52e2a94"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def production_config() -> dict[str, Any]:
    """Return the Stage-A-compatible config, without running the method."""
    root = Path(__file__).resolve().parents[2]
    baseline_path = root / "scripts/audio_enhancement/baselines.py"
    profile_path = root / "scripts/audio_enhancement/shared_frequency_profile_v1.py"
    if _sha256(baseline_path) != EXPECTED_BASELINES_SHA256:
        raise RuntimeError("frozen baseline implementation source SHA changed")
    if _sha256(profile_path) != EXPECTED_SHARED_PROFILE_SHA256:
        raise RuntimeError("frozen r090 reference-profile source SHA changed")
    return {
        "schema": SCHEMA,
        "method_id": METHOD_ID,
        "selection_status": "PREPARED_NOT_FROZEN",
        "effect_execution_status": "BLOCKED_UNTIL_SHARED540_AND_METHOD_FREEZE",
        "kind": "bounded_local_compression_control",
        "config": {
            "method": "rms_then_localcompress",
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "pre_rms_target": TARGET_RMS,
            "post_rms_target": TARGET_RMS,
            "local_window_ms": LOCAL_WINDOW_MS,
            "local_ratio": LOCAL_RATIO,
            "local_min_gain_db": LOCAL_MIN_GAIN_DB,
            "local_max_gain_db": LOCAL_MAX_GAIN_DB,
            "envelope_floor": ENVELOPE_FLOOR,
            "peak_ceiling": None,
        },
        "stage_order": [
            "global RMS to fixed target, no peak cap",
            "centered local RMS envelope compression, ratio 0.90, gain clipped to ±6 dB",
            "global RMS to the same fixed target, no peak cap",
        ],
        "output_contract": {
            "dtype": "float32",
            "channels": 1,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "sample_count": "identical to input",
            "retiming_or_resampling": False,
            "spectral_mask": False,
            "peak_limiting": False,
        },
        "source_pins": {
            "baselines_path": "scripts/audio_enhancement/baselines.py",
            "baselines_raw_sha256": EXPECTED_BASELINES_SHA256,
            "shared_profile_reference_path": "scripts/audio_enhancement/shared_frequency_profile_v1.py",
            "shared_profile_reference_raw_sha256": EXPECTED_SHARED_PROFILE_SHA256,
            "stagea_transform_path": "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt03.py",
            "stagea_transform_raw_sha256": EXPECTED_STAGEA_CORE_SHA256,
        },
        "interpretation": (
            "A ratio-0.90 local-compression control, distinct from the existing ratio-0.80 "
            "localcompress arm and from the r090 spectral-mask profiles. This config is a "
            "production scaffold only; it is not a frozen method or an effect result."
        ),
    }


def apply_r090_localcompress_control(
    samples: np.ndarray, *, sample_rate_hz: int = SAMPLE_RATE_HZ
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the prepared r=.90 control when an approved plan invokes it.

    The two fixed-RMS stages match the existing Stage-A `rms_then_localcompress`
    operation; this function adds source-pin checks and the fixed ratio=.90.
    """
    if sample_rate_hz != SAMPLE_RATE_HZ:
        raise ValueError("r090 control requires canonical 16 kHz audio")
    x = np.asarray(samples)
    if x.dtype != np.float32 or x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("input must be non-empty finite mono float32 PCM")
    config_doc = production_config()
    root = Path(__file__).resolve().parents[2]
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.audio_enhancement.baselines import (
        apply_global_rms,
        apply_local_envelope_compression,
        waveform_rms,
    )

    before_rms = waveform_rms(x)
    pre, pre_info = apply_global_rms(x, TARGET_RMS, peak_ceiling=None)
    compressed, compressor_info = apply_local_envelope_compression(
        pre,
        sample_rate=SAMPLE_RATE_HZ,
        window_ms=LOCAL_WINDOW_MS,
        ratio=LOCAL_RATIO,
        min_gain_db=LOCAL_MIN_GAIN_DB,
        max_gain_db=LOCAL_MAX_GAIN_DB,
        envelope_floor=ENVELOPE_FLOOR,
    )
    output, post_info = apply_global_rms(compressed, TARGET_RMS, peak_ceiling=None)
    if output.shape != x.shape or output.dtype != np.float32 or not np.isfinite(output).all():
        raise RuntimeError("r090 control violated the finite, exact-length float32 contract")
    return np.ascontiguousarray(output), {
        "schema": SCHEMA,
        "method_id": METHOD_ID,
        "production_config": config_doc,
        "sample_count": int(x.size),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "input_rms": before_rms,
        "pre_rms": asdict(pre_info),
        "local_compressor": asdict(compressor_info),
        "post_rms": asdict(post_info),
        "output_rms": waveform_rms(output),
        "output_peak_abs": float(np.max(np.abs(output.astype(np.float64)))),
        "clock_preserved": True,
        "effect_status": "TRANSFORM_INVOKED_BY_CALLER",
    }
