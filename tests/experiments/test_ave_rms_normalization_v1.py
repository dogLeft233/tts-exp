import importlib.util,unittest
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_multispeaker_full_20261017/ave_rms_normalization_v1.py'
s=importlib.util.spec_from_file_location('ave_rms',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class RMSNormalizationTests(unittest.TestCase):
 def test_scalar_gain_cancel_and_shape_preserved(self):
  x=np.random.default_rng(123).normal(.01,.12,20001)
  ys=[m.normalize(x*10**(g/20))[0] for g in [-6,0,6]]
  for y in ys:
   self.assertEqual(y.dtype,np.float32);self.assertEqual(y.shape,x.shape)
   self.assertLess(abs(np.sqrt(np.mean(y.astype(np.float64)**2))-.1),1e-8)
   self.assertTrue(np.allclose(y,x*(.1/np.sqrt(np.mean(x*x))),rtol=1e-7,atol=1e-8))
  self.assertTrue(np.allclose(ys[0],ys[2],rtol=1e-7,atol=1e-8))
 def test_no_clipping_and_no_centering(self):
  x=np.zeros(10000);x[2]=1;x[3]=.1
  y,r=m.normalize(x)
  self.assertGreater(y.max(),1);self.assertGreater(y.mean(),0)
  self.assertFalse(r['clipping']);self.assertFalse(r['mean_subtraction'])
 def test_reject_degenerate_nonfinite_and_channels(self):
  for x in [np.zeros(4),np.array([]),np.array([np.nan]),np.array([np.inf]),np.zeros((4,2))]:
   with self.assertRaises(ValueError):m.normalize(x)
  for target in [0,-1,np.inf,np.nan]:
   with self.assertRaises(ValueError):m.normalize(np.ones(4),target)
if __name__=='__main__':unittest.main()
