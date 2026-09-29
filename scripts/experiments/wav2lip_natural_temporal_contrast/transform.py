from __future__ import annotations

from typing import Any

import numpy as np

from . import config
from .common import ProtocolError


def chunk_mels(mel: np.ndarray) -> list[np.ndarray]:
    value = np.asarray(mel, dtype=np.float32)
    if value.shape != (config.MEL_BINS, config.MEL_FRAMES) or not np.isfinite(value).all():
        raise ProtocolError(f"mel must be [80,308], got {value.shape}")
    chunks: list[np.ndarray] = []
    index = 0
    while True:
        start = int(index * 80.0 / config.FPS)
        if start + 16 > value.shape[1]:
            chunks.append(value[:, -16:])
            break
        chunks.append(value[:, start : start + 16])
        index += 1
    if len(chunks) != config.FRAME_COUNT:
        raise ProtocolError(f"frozen chunk count changed: {len(chunks)}")
    return chunks


def temporal_contrast(mel: np.ndarray) -> dict[str, Any]:
    value = np.asarray(mel, dtype=np.float64)
    if value.shape != (config.MEL_BINS, config.MEL_FRAMES) or not np.isfinite(value).all():
        raise ProtocolError(f"mel must be [80,308], got {value.shape}")
    kernel = np.asarray([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
    padded = np.pad(value, ((0, 0), (2, 2)), mode="reflect")
    smooth = sum(float(weight) * padded[:, offset : offset + value.shape[1]] for offset, weight in enumerate(kernel))
    residual = value - smooth
    residual[:, :2] = 0.0
    residual[:, -2:] = 0.0
    maximum = float(np.max(np.abs(residual)))
    amplitude = min(0.25, 0.5 / maximum) if maximum > 1e-12 else 0.0
    pre_smooth = value - amplitude * residual
    pre_sharp = value + amplitude * residual
    smooth_out = np.asarray(np.clip(pre_smooth, -4.0, 4.0), dtype=np.float32)
    sharp_out = np.asarray(np.clip(pre_sharp, -4.0, 4.0), dtype=np.float32)
    return {"N": np.asarray(value, dtype=np.float32), "SMOOTH": smooth_out, "SHARP": sharp_out, "residual": residual, "amplitude": amplitude, "preclip_l2": {"SMOOTH": float(np.linalg.norm(pre_smooth - value)), "SHARP": float(np.linalg.norm(pre_sharp - value))}, "postclip_l2": {"SMOOTH": float(np.linalg.norm(smooth_out.astype(np.float64) - value)), "SHARP": float(np.linalg.norm(sharp_out.astype(np.float64) - value))}, "clip_fraction": {"SMOOTH": float(np.mean((pre_smooth < -4.0) | (pre_smooth > 4.0))), "SHARP": float(np.mean((pre_sharp < -4.0) | (pre_sharp > 4.0)))} }
