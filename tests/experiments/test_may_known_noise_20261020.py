import unittest,importlib.util
from pathlib import Path
import numpy as np
SRC=Path(__file__).resolve().parents[2]/'scripts/experiments/may_known_noise_20261020/run.py'
spec=importlib.util.spec_from_file_location('known_noise',SRC);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class KnownNoiseControls(unittest.TestCase):
    def test_same_snr_and_noise_only_contrasts_are_distinct(self):
        s=np.array([.3,-.8,1.2],dtype=np.float32);n=np.array([.01,.02,-.01],dtype=np.float32)
        self.assertTrue(np.array_equal(audit.mixture(s,n,'B'),audit.mixture(s,n,'A')*.5))
        self.assertTrue(np.allclose(audit.mixture(s,n,'C')-s,n*.5,atol=1e-7))
        self.assertTrue(np.allclose(audit.mixture(s,n,'D')-s*.5,n,atol=1e-7))

    def test_noise_seed_is_deterministic_and_unit_rms(self):
        for kind in ['white','pink']:
            a=audit.noise_template(16000,kind,42);b=audit.noise_template(16000,kind,42)
            self.assertTrue(np.array_equal(a,b));self.assertAlmostEqual(float(np.sqrt(np.mean(a.astype(float)**2))),1,places=7)
            self.assertLess(abs(float(a.mean())),1e-7)

    def test_direction_metric_ignores_uniform_output_gain(self):
        m=np.array([[5,20,5,10,2],[2,8,2,4,1]],dtype=float)
        self.assertEqual(audit.response(m)['unit_noise_change'],0)
        self.assertGreater(audit.response(m)['relative_noise_change'],0)

    def test_mixing_never_clips_large_standardized_values(self):
        x=np.array([3,-3],dtype=np.float32);n=np.array([1,-1],dtype=np.float32)
        self.assertTrue(np.array_equal(audit.mixture(x,n,'A'),[4,-4]))
        with self.assertRaises(ValueError):audit.mixture(x,n[:1],'A')


if __name__=='__main__':unittest.main()
