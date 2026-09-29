from __future__ import annotations

from typing import Any

import numpy as np

from scripts.experiments.wav2lip_probe_runtime import ProtocolError


def construct_gains(pcm: np.ndarray) -> dict[str, Any]:
    values = np.asarray(pcm)
    if values.ndim != 1 or values.dtype != np.int16:
        raise ProtocolError("gain input must be mono int16")
    peak = float(np.max(np.abs(values.astype(np.float64)))) if values.size else 0.0
    if peak == 0.0:
        return {"status": "INPUT_DEGENERATE", "reason": "silent_pcm", "peak": peak, "gain_plus": None, "gain_minus": None}
    gain_plus = min(float(10.0 ** (3.0 / 20.0)), float(0.98 * 32767.0 / peak))
    if gain_plus <= 1.0:
        return {"status": "INPUT_DEGENERATE", "reason": "no_amplification_headroom", "peak": peak, "gain_plus": gain_plus, "gain_minus": 1.0 / gain_plus}
    gain_minus = 1.0 / gain_plus
    plus = np.rint(gain_plus * values.astype(np.float64))
    minus = np.rint(gain_minus * values.astype(np.float64))
    for name, candidate in (("GAIN_PLUS", plus), ("GAIN_MINUS", minus)):
        if np.any(candidate < -32768.0) or np.any(candidate > 32767.0):
            raise ProtocolError(f"{name} would clip before int16 conversion")
    return {"status": "GO", "reason": None, "peak": peak, "gain_plus": gain_plus, "gain_minus": gain_minus, "plus": plus.astype(np.int16), "minus": minus.astype(np.int16), "plus_db": float(20.0 * np.log10(gain_plus)), "minus_db": float(20.0 * np.log10(gain_minus))}
