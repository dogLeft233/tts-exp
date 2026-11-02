import importlib.util
from pathlib import Path
import unittest
import numpy as np

P=Path(__file__).resolve().parents[2]/'scripts/experiments/may_noise_strength_20261020/run.py'
spec=importlib.util.spec_from_file_location('dose_scan',P);scan=importlib.util.module_from_spec(spec);spec.loader.exec_module(scan)

class Metrics(unittest.TestCase):
    def test_tiny_orthogonal_signal_survives_metric(self):
        x=np.array([[1.,0.],[1.,0.]]);y=x+np.array([[0.,1e-11],[0.,1e-11]])
        m=scan.summarize(scan.metric_frames(x,y),2)
        self.assertAlmostEqual(m['unit_delta']/1e-11,1.,places=12)
        self.assertEqual(m['changed_coordinate_fraction'],.5)

    def test_scale_invariance_and_exact_identity(self):
        r=np.random.default_rng(42);x=r.normal(size=(13,9));y=x+r.normal(size=x.shape)*.01
        a=scan.metric_frames(x,y);b=scan.metric_frames(.5*x,.5*y)
        np.testing.assert_array_equal(a[:,3],b[:,3])
        self.assertEqual(scan.summarize(scan.metric_frames(x,x),9)['unit_delta'],0)

    def test_input_rounding_erases_both_scales(self):
        s=np.array([1.,-2.,.3],np.float32);n=np.ones(3,np.float32)*1e-20
        v=scan.waveform_audit(s,n)
        self.assertEqual(v['input_changed_fraction'],0);self.assertEqual(v['half_changed_fraction'],0)
        self.assertTrue(v['exact_power_two_difference'])

    def test_sealed30_dose_and_increasing_snr(self):
        x=np.ones(13,np.float32);n=np.arange(13,dtype=np.float32)*.001
        _,a=scan.dose(x,n,30,'fp32');_,b=scan.dose(x,n,50,'fp32')
        np.testing.assert_array_equal(a,n);np.testing.assert_array_equal(b,n*np.float32(.1))

if __name__=='__main__':unittest.main()
