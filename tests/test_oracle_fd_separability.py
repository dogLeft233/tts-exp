import sys
from pathlib import Path
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location(
    "oracle_fd_separability",
    Path(__file__).resolve().parent.parent / "scripts" / "31_oracle_fd_separability.py",
)
mod = importlib.util.module_from_spec(spec)


def test_construct_fd_no_fc_just_addition():
    """fc=None -> Fd = Fs broadcast + Fp (the Fd_zero tier)."""
    spec.loader.exec_module(mod)
    Fs = np.ones(768, dtype=np.float32) * 0.5
    Fp = np.zeros((10, 768), dtype=np.float32)
    Fd = mod.construct_fd(Fp, Fs, fc=None)
    assert Fd.shape == (10, 768)
    assert np.allclose(Fd, 0.5)


def test_construct_fd_with_fc_changes_output():
    """fc=random FC -> Fd differs from pure addition (orthogonal init rotates)."""
    spec.loader.exec_module(mod)
    Fs = np.ones(768, dtype=np.float32) * 0.5
    Fp = np.zeros((10, 768), dtype=np.float32)
    fc = mod.build_fc_orthogonal(hidden=768, seed=42)
    Fd = mod.construct_fd(Fp, Fs, fc=fc)
    assert Fd.shape == (10, 768)
    assert not np.allclose(Fd, 0.5)


def test_fc_deterministic_with_seed():
    """Same seed should produce identical FC output for given input."""
    spec.loader.exec_module(mod)
    Fs = np.ones(768, dtype=np.float32) * 0.5
    Fp = np.zeros((10, 768), dtype=np.float32)
    fc1 = mod.build_fc_orthogonal(hidden=768, seed=42)
    fc2 = mod.build_fc_orthogonal(hidden=768, seed=42)
    Fd1 = mod.construct_fd(Fp, Fs, fc=fc1)
    Fd2 = mod.construct_fd(Fp, Fs, fc=fc2)
    assert np.allclose(Fd1, Fd2), "FC with same seed must produce identical output"


def test_load_fp_at_layer_returns_correct_layer():
    """Loading a saved .npy and slicing to a specific layer returns the right shape."""
    import tempfile
    # Create a fake .npy with shape (4, T, 768) -- matches layers [0, 6, 11, 12]
    tmp = Path(tempfile.mkdtemp())
    fake_npy = tmp / "fake.npy"
    fake_data = np.stack([
        np.full((10, 768), 0.0, dtype=np.float32),    # layer 0
        np.full((10, 768), 6.0, dtype=np.float32),    # layer 6
        np.full((10, 768), 11.0, dtype=np.float32),   # layer 11
        np.full((10, 768), 12.0, dtype=np.float32),   # layer 12
    ], axis=0)  # shape (4, 10, 768)
    np.save(fake_npy, fake_data)
    try:
        spec.loader.exec_module(mod)
        # Default requested_layers = [0, 6, 11, 12]; layer 11 -> index 2
        Fp = mod.load_fp_at_layer(fake_npy, fp_layer=11)
        assert Fp.shape == (10, 768)
        assert np.allclose(Fp, 11.0), f"expected layer-11 values; got {Fp[0, 0]}"
        Fp0 = mod.load_fp_at_layer(fake_npy, fp_layer=0)
        assert np.allclose(Fp0, 0.0)
    finally:
        import shutil
        shutil.rmtree(tmp)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
