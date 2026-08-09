"""Tests for CPU-only enhancer evaluation metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from eval_pnp_audio_enhancer import evaluate_items


def test_evaluate_identity_reports_exact_length(tmp_path: Path) -> None:
    sr = 16_000
    natural = np.sin(np.linspace(0, 4 * np.pi, 1024, dtype=np.float32))
    target = natural.copy()
    natural_path = tmp_path / "natural.wav"
    target_path = tmp_path / "target.wav"
    sf.write(natural_path, natural, sr)
    sf.write(target_path, target, sr)
    result = evaluate_items(
        [{"paired_key": "x", "provider": "test", "natural_path": str(natural_path), "audio_path": str(target_path)}],
        chunk_seconds=0.02,
        stride_seconds=0.01,
    )
    assert result["item_count"] == 1
    assert result["valid_count"] == 1
    row = result["rows"][0]
    assert row["exact_length"] is True
    assert row["enhanced"]["finite"] is True
    assert row["enhanced_stft_loss"] == 0.0
    assert row["enhanced_log_mel_loss"] == 0.0
