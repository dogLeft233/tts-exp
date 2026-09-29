import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'scripts/experiments/tts_acoustic_modulation_calibration.py'
spec = importlib.util.spec_from_file_location('acoustic_calibration', PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_synthetic_contracts():
    controls = m.synthetic_checks()
    assert len(controls) == 8
    for conditions in controls.values():
        assert conditions['identity']['correlation_lag_samples'] == 0


def test_shared_pair_headroom_preserves_relative_rms():
    t = np.arange(16003) / m.SR
    n = .85 * np.sin(2*np.pi*217*t) + .4*np.sin(2*np.pi*911*t)
    tts = .25 * np.sin(2*np.pi*191*t)
    before, after, g, _ = m.transform_pair(n, tts)
    assert 0 < g < 1
    for arm, source in [('N', n), ('T', tts)]:
        for cond in m.CONDITIONS:
            assert len(after[arm][cond]) == len(source)
            assert np.max(np.abs(after[arm][cond])) <= .98000000001
            assert abs(m.rms(before[arm][cond])/m.rms(source) - 1) < 1e-12
            assert abs(m.rms(after[arm][cond])/(g*m.rms(source)) - 1) < 1e-12
            np.testing.assert_allclose(after[arm][cond], before[arm][cond]*g, atol=0, rtol=0)


def test_residual_double_centering_and_gain_definition():
    rng = np.random.default_rng(915)
    x = rng.standard_normal(4001)*.1
    z = m.stft(x)
    r, _ = m.decompose(z)
    np.testing.assert_allclose(r.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(r.mean(1), 0, atol=1e-12)
    alpha = .8
    expected = m.istft(z * 10**(np.clip((alpha-1)*r, -6, 6)/20), len(x))
    expected *= m.rms(x)/m.rms(expected)
    actual, _ = m.transform(x, alpha)
    np.testing.assert_allclose(actual, expected, atol=0, rtol=0)


def test_gain_scale_invariance_and_zero_bins():
    x = np.random.default_rng(92).standard_normal(7021)*.1
    r, _ = m.decompose(m.stft(x))
    r2, _ = m.decompose(m.stft(x*.123))
    np.testing.assert_allclose(r, r2, atol=1e-12)
    before, after, g, _ = m.transform_pair(np.zeros(21), np.zeros(1))
    assert g == 1
    assert all(np.count_nonzero(x) == 0 for cells in after.values() for x in cells.values())
