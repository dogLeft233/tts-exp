"""Run in the pinned remote torch environment; tests the actual signal seam."""
import sys
import unittest
from pathlib import Path
try:
    import numpy as np
    import torch
except ImportError as exc:
    raise unittest.SkipTest('Signal tests require the pinned remote torch environment') from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/experiments/phoneme_direction_20261026'))
from probe_weak_segments import conditioning, encode


class ConditioningTest(unittest.TestCase):
    def test_peak_gain_reversible_without_waveform_bypass(self):
        a = np.linspace(-.3, .2, 12345, dtype='float32')
        b, gain = conditioning(a, 'peak95')
        self.assertAlmostEqual(float(abs(b).max()), .95, places=6)
        np.testing.assert_allclose(b / gain, a, atol=3e-8)

    def test_local_gain_finite_bounded_at_silence_and_edges(self):
        a = np.zeros(32001, dtype='float32'); a[8000:16000] = .1
        b, gain = conditioning(a, 'local_rms16')
        self.assertTrue(np.isfinite(gain).all())
        self.assertLessEqual(float(gain.max()), 10 ** 1.5)
        np.testing.assert_allclose(b / gain, a, atol=3e-8)
        self.assertTrue(np.all(b[a == 0] == 0))

    def test_chunk_encoder_preserves_hop_positions_and_final_frame(self):
        class LocalEncoder:
            def extract(self, x):
                # A valid RF400/hop320 local encoder exposes absolute starts.
                return x.unfold(0, 400, 320)[:, :1].repeat(1, 1024)
        for size in [16000, 32000, 32001, 80413]:
            a = np.arange(size, dtype='float32')
            plain = encode(LocalEncoder(), a, 'peak95')
            chunk = encode(LocalEncoder(), a, 'local_context')
            self.assertEqual(tuple(chunk.shape), ((size + 319) // 320, 1024))
            torch.testing.assert_close(plain, chunk, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
