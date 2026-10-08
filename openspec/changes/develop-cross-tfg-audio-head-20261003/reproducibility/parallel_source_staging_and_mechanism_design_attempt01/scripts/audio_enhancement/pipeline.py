"""Stable waveform transform callable for generation/evaluation producers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .audio_io import CANONICAL_SAMPLE_RATE
from .baselines import (
    SpectralProfile,
    apply_flat_gain,
    apply_global_rms,
    apply_local_envelope_compression,
    apply_spectral_profile,
)


def _canonical_input(waveform: np.ndarray) -> np.ndarray:
    x = np.asarray(waveform)
    if x.dtype != np.float32:
        raise TypeError("transform expects canonical float32 input")
    if x.ndim != 1 or x.size == 0:
        raise ValueError("transform expects a non-empty mono [samples] array")
    if not np.isfinite(x).all():
        raise ValueError("transform input contains NaN or infinite samples")
    return np.ascontiguousarray(x)


def transform(
    audio_16k_mono_f32: np.ndarray,
    config: Mapping[str, Any],
    *,
    head: Any | None = None,
) -> np.ndarray:
    """Apply a serializable method configuration to canonical input PCM.

    Supported ``method`` values:

    - ``identity``: exact sample-for-sample copy.
    - ``rms``: historical fixed target-RMS positive control; target may be
      explicitly overridden by a frozen run configuration.
    - ``flat_gain``: one declared utterance-wide gain, bounded independently
      from local-envelope and spectral controls.
    - ``profile``: bounded, phase-preserving low-dimensional magnitude profile.
    - ``local_compressor``: bounded, centered 20--50 ms envelope compressor.
    - ``spectral_head``: a supplied PyTorch head, optionally preceded by the
      independent global-RMS baseline.

    All paths return finite float32 mono audio with exactly the input sample
    count. Resampling belongs to :func:`read_canonical_wav`, not this callable.
    """

    if config.get("sample_rate_hz", CANONICAL_SAMPLE_RATE) != CANONICAL_SAMPLE_RATE:
        raise ValueError("the transform interface is frozen at 16 kHz")
    x = _canonical_input(audio_16k_mono_f32)
    method = str(config.get("method", "identity"))
    if method == "identity":
        output = x.copy()
    elif method == "rms":
        output, _ = apply_global_rms(
            x,
            target_rms=float(config.get("target_rms", 0.0376838172675746)),
            peak_ceiling=config.get("peak_ceiling"),
        )
    elif method == "flat_gain":
        output, _ = apply_flat_gain(
            x,
            gain_db=float(config.get("gain_db", 0.0)),
            max_abs_gain_db=float(config.get("max_abs_gain_db", 6.0)),
            peak_ceiling=config.get("peak_ceiling"),
        )
    elif method == "profile":
        profile_data = config.get("profile", {})
        if not isinstance(profile_data, Mapping):
            raise TypeError("profile config must be a mapping")
        profile = SpectralProfile(
            global_gain_db=float(profile_data.get("global_gain_db", 0.0)),
            band_gains_db=tuple(profile_data.get("band_gains_db", (0.0, 0.0, 0.0))),
            envelope_gains_db=tuple(profile_data.get("envelope_gains_db", (0.0,) * 8)),
            max_abs_gain_db=float(profile_data.get("max_abs_gain_db", 6.0)),
            max_total_gain_db=float(profile_data.get("max_total_gain_db", 6.0)),
        )
        output = apply_spectral_profile(x, profile)
    elif method == "local_compressor":
        output, _ = apply_local_envelope_compression(
            x,
            window_ms=float(config.get("window_ms", 32.0)),
            ratio=float(config.get("ratio", 0.8)),
            min_gain_db=float(config.get("min_gain_db", -6.0)),
            max_gain_db=float(config.get("max_gain_db", 6.0)),
            envelope_floor=float(config.get("envelope_floor", 1e-5)),
        )
    elif method == "spectral_head":
        if head is None:
            raise ValueError("spectral_head method requires a loaded head instance")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError("spectral_head inference requires PyTorch") from exc
        device = next(head.parameters()).device
        input_tensor = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        head.eval()
        from .heads import EnhancementStack

        stack = EnhancementStack(head).to(device)
        with torch.no_grad():
            output_tensor = stack(
                input_tensor,
                use_global_rms=bool(config.get("use_global_rms", False)),
                use_local_envelope=bool(config.get("use_local_envelope", False)),
                use_spectral_head=bool(config.get("use_spectral_head", True)),
                local_window_ms=float(config.get("local_window_ms", 32.0)),
                local_ratio=float(config.get("local_ratio", 0.8)),
                local_min_gain_db=float(config.get("local_min_gain_db", -6.0)),
                local_max_gain_db=float(config.get("local_max_gain_db", 6.0)),
                post_rms_match=str(config.get("post_rms_match", "none")),
                target_rms=float(config.get("target_rms", 0.0376838172675746)),
                rms_stage_peak_ceiling=config.get("rms_stage_peak_ceiling"),
                peak_ceiling=config.get("peak_ceiling"),
            )
        if isinstance(output_tensor, tuple):
            output_tensor = output_tensor[0]
        if output_tensor.ndim == 2 and output_tensor.shape[0] == 1:
            output_tensor = output_tensor[0]
        output = output_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    else:
        raise ValueError(f"unsupported enhancement method: {method}")

    result = np.asarray(output, dtype=np.float32)
    if result.ndim != 1 or result.shape != x.shape:
        raise RuntimeError("enhancement transform changed waveform shape or sample count")
    if not np.isfinite(result).all():
        raise RuntimeError("enhancement transform returned non-finite audio")
    return np.ascontiguousarray(result)
