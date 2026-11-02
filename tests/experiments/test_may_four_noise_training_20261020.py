import importlib.util,unittest
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[2]/'scripts/experiments/may_four_noise_training_20261020/contrast.py'
s=importlib.util.spec_from_file_location('contrast',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Controls(unittest.TestCase):
 def test_identical_noise_and_binary_gain(self):
  rng=np.random.default_rng(42);s=rng.normal(size=10003).astype('float32');n=(rng.normal(size=len(s))*.03).astype('float32')
  for q in [None,30,90]:
   a=m.mixed_input(s,None if q is None else n,q,1.);b=m.mixed_input(s,None if q is None else n,q,.5)
   np.testing.assert_array_equal(b,a*np.float32(.5));np.testing.assert_array_equal(b-s*.5,(a-s)*.5)
 def test_pure_scale_effect_cancels_from_interaction(self):
  self.assertAlmostEqual(m.difference_in_differences(11.,10.,8.,7.),0)
  self.assertAlmostEqual(m.difference_in_differences(13.,10.,8.,7.),2)
 def test_reject_clock_or_unplanned_dose(self):
  with self.assertRaises(ValueError):m.mixed_input(np.zeros(10),np.zeros(9),30,.5)
  with self.assertRaises(ValueError):m.mixed_input(np.zeros(10),np.zeros(10),80,.5)
if __name__=='__main__':unittest.main()
