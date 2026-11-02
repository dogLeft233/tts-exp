"""Signal-level regressions for diagnostic spectral probes."""
import sys
import unittest
from pathlib import Path

try:
    import numpy as np
    import scipy
    import soundfile
except ImportError:
    raise unittest.SkipTest('Requires pinned remote numerical environment')

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/experiments/phoneme_direction_20261026'))
from separate_encoder_decoder import bands
from probe_l6_postfilter import postfilter, fit_head, predict_head


class SpectralProbeTests(unittest.TestCase):
    def test_grid_and_zero_correction_identity(self):
        rng = np.random.default_rng(123)
        for n in [16001, 16320, 144384]:
            a = rng.normal(0, .02, n)
            y = bands(a)
            self.assertEqual(y.shape, ((n + 319) // 320, 15))
            b = postfilter(a, y, 1.)
            self.assertEqual(b.shape, a.shape)
            self.assertLess(float(abs(a - b).max()), 1e-10)

    def test_silence_and_bounded_extreme_prediction(self):
        a = np.zeros(16001)
        for value in [-200., 200.]:
            b = postfilter(a, np.full(bands(a).shape, value), 1.)
            self.assertTrue(np.isfinite(b).all())
            self.assertEqual(float(abs(b).max()), 0.)

    def test_ridge_recoverable_signal(self):
        rng = np.random.default_rng(987)
        h = rng.normal(size=(2000, 8)); w = rng.normal(size=(8, 15)); y = h @ w + 2.
        model = fit_head([h[:1500]], [y[:1500]])
        predicted = predict_head(h[1500:], model)
        self.assertLess(float(np.mean((predicted - y[1500:]) ** 2)), .1)


if __name__ == '__main__': unittest.main()
