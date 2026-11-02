"""Scientific contracts for common-support lag diagnostic."""
import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location("cogenav_audit", Path(__file__).parents[1] / "scripts/experiments/cogenav_sync_audit_20261024.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_known_shift_and_common_support():
    x = np.eye(100, dtype=np.float32)
    a = np.zeros_like(x)
    a[4:] = x[:-4]
    curve, local = m.lag_curve(a, x)
    assert np.argmax(curve)-15 == 4
    assert local.shape == (31, 70)
    assert curve[19] == 1


def test_scale_invariance_and_relu():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(100, 32))
    np.testing.assert_allclose(m.lag_curve(a, a)[0], m.lag_curve(a*3, a*.1)[0], atol=1e-12)
    assert m.lag_curve(a, -a)[0][15] == 0


def test_reuse_requires_same_scientific_protocol():
    old = {"sources": 4, "audio": "canonical", "representation": "FP32"}
    assert m.compatible_protocol(old, dict(old, sources=20, reuse={"root": "pilot"}))
    assert not m.compatible_protocol(old, dict(old, audio="normalized_differently"))
    assert not m.compatible_protocol(old, dict(old, representation="FP16"))
