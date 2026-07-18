import sys
from pathlib import Path
import importlib.util
import numpy as np

spec_obj_path = (
    Path(__file__).resolve().parent.parent / "scripts" / "29_oracle_fs_bert.py"
)
spec = importlib.util.spec_from_file_location("oracle_fs_bert", spec_obj_path)
mod = importlib.util.module_from_spec(spec)


def test_fs_dim_768():
    """Fs_cls and Fs_mean must be 768-dim, extracted from a fake BERT output."""
    spec.loader.exec_module(mod)
    import torch
    fake_last_hidden = torch.zeros(1, 5, 768)
    fake_last_hidden[0, 0] = 1.0  # CLS
    fake_last_hidden[0, 1:-1] = 2.0  # tokens
    from types import SimpleNamespace
    fake_outputs = SimpleNamespace(last_hidden_state=fake_last_hidden)
    Fs_cls, Fs_mean, meta = mod.extract_fs_from_outputs(fake_outputs)
    assert Fs_cls.shape == (768,)
    assert Fs_mean.shape == (768,)
    assert abs(Fs_cls[0] - 1.0) < 1e-5
    assert abs(Fs_mean[0] - 2.0) < 1e-5
    assert meta["token_count"] == 3


def test_fs_diagnostics_finite():
    spec.loader.exec_module(mod)
    fs = np.ones(768, dtype=np.float32) * 0.5
    fs_empty = np.zeros(768, dtype=np.float32)
    diag = mod.compute_fs_diagnostics(fs, fs_empty)
    assert "fs_norm" in diag
    assert "fs_empty_l1" in diag
    assert np.isfinite(diag["fs_norm"])
    assert np.isfinite(diag["fs_empty_l1"])


def test_cosine_similarity_simple():
    """cosine_similarity returns 1.0 for parallel vectors, 0.0 for orthogonal."""
    spec.loader.exec_module(mod)
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sim = mod.cosine_similarity(a, b)
    assert abs(sim - 1.0) < 1e-6
    sim2 = mod.cosine_similarity(a, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert abs(sim2 - 0.0) < 1e-6


def test_flag_degenerate_sigma_zero_and_empty_l1_thresholds():
    """flag_degenerate with all-identical ||Fs|| (σ=0) falls back to L1-only detection."""
    spec.loader.exec_module(mod)
    # All entries have identical ||Fs|| → σ=0 → only the l1<0.5 check fires
    entries = [
        {
            "sample_id": i,
            "condition": "natural" if i % 2 == 0 else "tts",
            "Fs_cls": [0.5] * 768,
            "fs_cls_norm": 14.0,  # arbitrary; identical across entries
            "fs_empty_l1": 0.2 if i == 0 else 5.0,  # only sample 0 has small L1 → degenerate
            "is_degenerate": False,
        }
        for i in range(4)
    ]
    result = mod.flag_degenerate(entries)
    assert result[0]["is_degenerate"] is True, "small fs_empty_l1 must trigger degeneracy"
    assert result[1]["is_degenerate"] is False
    assert result[2]["is_degenerate"] is False
    assert result[3]["is_degenerate"] is False
    # Verify the others are not affected by norm-based detection (σ=0 → no threshold)
    assert all(not r["is_degenerate"] for r in result[1:])


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
