import sys,unittest
from pathlib import Path
import numpy as np
SRC=Path(__file__).resolve().parents[2]/'scripts/experiments/may_postnorm_scale_20261020'
sys.path.insert(0,str(SRC))
from scaling import postnorm_scale,expected_cells,validate_cells
class ScaleControls(unittest.TestCase):
    def test_adjustment_is_after_normalization(self):
        x=np.array([-.2,.1,.5,-.1,.04],dtype=np.float32)
        z=(x-x.mean())/np.sqrt(x.var()+1e-7)
        low=postnorm_scale(z,.5);high=postnorm_scale(z,2)
        self.assertTrue(np.array_equal(high,low*4));self.assertAlmostEqual(float(np.sqrt(np.mean(low**2)))/float(np.sqrt(np.mean(z**2))),.5,places=6)
        incorrectly_renormalized=(low-low.mean())/np.sqrt(low.var()+1e-7)
        self.assertGreater(float(np.max(abs(incorrectly_renormalized-low))),.1)
    def test_no_clipping_and_scale1_is_exact(self):
        z=np.array([-2,0,2],dtype=np.float32)
        self.assertTrue(np.array_equal(postnorm_scale(z,1),z));self.assertEqual(float(postnorm_scale(z,2).max()),4)
    def test_bad_scale_and_nonfinite_rejected(self):
        for z,s in [(np.array([1],dtype=np.float32),3),(np.array([np.inf],dtype=np.float32),.5),(np.ones(2,dtype=np.float64),1)]:
            with self.assertRaises(ValueError):postnorm_scale(z,s)
    def test_complete_pairing_no_duplicate_or_missing(self):
        rows=[dict(encoder=e,train_input_scale=t,inference_input_scale=q) for e,t,q in expected_cells()]
        self.assertTrue(validate_cells(rows))
        with self.assertRaises(ValueError):validate_cells(rows[:-1])
        with self.assertRaises(ValueError):validate_cells(rows[:-1]+rows[:1])
if __name__=='__main__':unittest.main()
