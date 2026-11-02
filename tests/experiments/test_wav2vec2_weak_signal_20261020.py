"""Metric controls distinguish pure scale, selective attenuation and shape change."""
import importlib.util,unittest
from pathlib import Path
import numpy as np
SRC=Path(__file__).resolve().parents[2]/'scripts/experiments/wav2vec2_weak_signal_20261020/run.py'
spec=importlib.util.spec_from_file_location('weak_audit',SRC);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class SelectivityMetrics(unittest.TestCase):
    def test_uniform_half_is_not_selective_change(self):
        x=np.array([[1.,2.],[-1.,-2.],[2.,4.]])
        f,c=audit.pair_moments(x,x*.5);m=audit.metrics(f)
        self.assertEqual(m['gain_fit'],.5);self.assertEqual(m['norm_ratio'],.5)
        self.assertEqual(m['shape_residual'],0)
        self.assertTrue(np.allclose(audit.channel_stats(c,len(x))[5],.5))

    def test_channel_selectivity_survives_affine_offset(self):
        x=np.array([[-1.,-2.],[0.,0.],[1.,2.]])+10
        y=(x-10)*np.array([.7,.99])+20
        f,c=audit.pair_moments(x,y)
        self.assertTrue(np.allclose(audit.channel_stats(c,len(x))[5],[.7,.99]))

    def test_shape_rotation_does_not_mean_attenuation(self):
        x=np.array([[1.,0.],[0.,1.]])
        f,_=audit.pair_moments(x,x[:,::-1]);m=audit.metrics(f)
        self.assertEqual(m['norm_ratio'],1);self.assertGreater(m['shape_residual'],.9)

    def test_mismatched_support_rejected(self):
        with self.assertRaises(AssertionError):audit.pair_moments(np.ones((2,2)),np.ones((3,2)))


if __name__=='__main__':unittest.main()
