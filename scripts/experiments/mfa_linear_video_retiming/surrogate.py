"""CPU-only SyncNet embedding surrogate for immutable retiming maps.

The five-frame visual window is anchored at its centre.  These values are
proposal scores only; they never substitute for a visual tower forward.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .common import ProtocolError
from .retime import validate_map


def interpolate_visual(visual: np.ndarray, mapping: Mapping[str, Any], rows: Sequence[int], *, anchor: str = "window_center") -> np.ndarray:
    validate_map(mapping)
    values = np.asarray(visual, dtype=np.float32)
    selected = np.asarray(rows, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != 1024 or selected.ndim != 1 or selected.size < 25:
        raise ProtocolError("PROXY_EMBEDDING_SHAPE_INVALID")
    if anchor not in {"window_center", "window_start"}:
        raise ProtocolError("PROXY_ANCHOR_INVALID")
    centre = 2 if anchor == "window_center" else 0
    q = np.asarray(mapping["q"], dtype=np.float64)
    if np.any(selected + centre >= q.size) or np.any(selected < 0):
        raise ProtocolError("PROXY_SUPPORT_OUT_OF_BOUNDS")
    source = q[selected + centre] - centre
    lo = np.floor(source).astype(np.int64)
    hi = np.ceil(source).astype(np.int64)
    if np.any(lo < 0) or np.any(hi >= values.shape[0]):
        raise ProtocolError("PROXY_SUPPORT_OUT_OF_BOUNDS")
    alpha = (source - lo).astype(np.float32)[:, None]
    return np.asarray((np.float32(1) - alpha) * values[lo] + alpha * values[hi], dtype=np.float32)


def score_proxy(visual: np.ndarray, audio: np.ndarray, mapping: Mapping[str, Any], rows: Sequence[int], *, anchor: str = "window_center", block_rows: int = 32) -> dict[str, Any]:
    selected = np.asarray(rows, dtype=np.int64)
    audio32 = np.asarray(audio, dtype=np.float32)
    if audio32.ndim != 2 or audio32.shape[1] != 1024 or block_rows < 1:
        raise ProtocolError("PROXY_AUDIO_SHAPE_INVALID")
    if np.any(selected - 15 < 0) or np.any(selected + 15 >= audio32.shape[0]):
        raise ProtocolError("PROXY_SUPPORT_OUT_OF_BOUNDS")
    proxy = interpolate_visual(visual, mapping, selected, anchor=anchor)
    lag = np.arange(-15, 16, dtype=np.int64)
    pieces: list[np.ndarray] = []
    for begin in range(0, selected.size, block_rows):
        end = min(begin + block_rows, selected.size)
        left = proxy[begin:end, None, :]
        right = audio32[selected[begin:end, None] + lag[None, :]]
        difference = np.asarray(left - right + np.float32(1e-6), dtype=np.float32)
        pieces.append(np.sqrt(np.sum(np.square(difference, dtype=np.float32), axis=2, dtype=np.float32), dtype=np.float32))
    distances = np.concatenate(pieces, axis=0)
    curve = np.mean(distances, axis=0, dtype=np.float32)
    minimum = float(np.min(curve))
    index = int(np.argmin(curve))
    return {"score_kind": "proxy_embedding", "anchor": anchor, "sync_c": float(np.median(curve) - minimum),
            "sync_d": minimum, "d0": float(curve[15]), "offset": 15 - index,
            "best_lag": index - 15, "curve": [float(x) for x in curve], "support_count": int(selected.size)}
