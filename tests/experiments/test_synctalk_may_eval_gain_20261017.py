import importlib.util
import itertools
import unittest
from pathlib import Path
import numpy as np

SCRIPT=Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_may_eval_gain_20261017/run.py'
spec=importlib.util.spec_from_file_location('may_eval_gain',SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class GainContracts(unittest.TestCase):
    def test_clipping_rejected(self):
        with self.assertRaises(ValueError):module.gain_pcm(np.array([20000,-20000],np.int16),6)

    def test_zero_identity_and_signed_rounding(self):
        original=np.array([-32768,-13,0,13,32767],np.int16)
        self.assertTrue(np.array_equal(module.gain_pcm(original,0),original))
        low=module.gain_pcm(np.array([-13,13],np.int16),-6)
        self.assertEqual(low.tolist(),[-7,7])

    def test_incomplete_or_duplicate_grid_rejected(self):
        rows=[dict(zip(['training_gain_db','driver_gain_db','evaluation_gain_db','clip_index'],key))
              for key in itertools.product([-6,0,6],[-6,0,6],[-6,0,6],range(4))]
        module.require_grid(rows)
        with self.assertRaises(ValueError):module.require_grid(rows[:-1])
        with self.assertRaises(ValueError):module.require_grid(rows[:-1]+rows[:1])


if __name__=='__main__':unittest.main()
