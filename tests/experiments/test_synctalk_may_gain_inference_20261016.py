import importlib.util
from pathlib import Path
import unittest
import numpy as np

path=Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_may_gain_inference_20261016/validate.py'
spec=importlib.util.spec_from_file_location('may_gain_validator',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ControlContracts(unittest.TestCase):
    def test_gained_scoring_pcm_is_rejected(self):
        original=np.array([5,-10,200,-400],dtype=np.int16)
        module.assert_pcm_equal(original.copy(),original)
        with self.assertRaisesRegex(AssertionError,'PCM content changed'):
            module.assert_pcm_equal(original*2,original)

    def test_shifted_original_pcm_is_rejected(self):
        original=np.array([5,-10,200,-400],dtype=np.int16)
        with self.assertRaisesRegex(AssertionError,'PCM content changed'):
            module.assert_pcm_equal(np.roll(original,1),original)

    def test_edge_score_cannot_replace_guarded_score(self):
        distances=np.full((80,31),10.,dtype=np.float32)
        distances[:20,15]=0;distances[-20:,15]=0
        self.assertEqual(module.score_curve(distances,0)['sync_c'],5.)
        self.assertEqual(module.score_curve(distances,20)['sync_c'],0.)


if __name__=='__main__':unittest.main()
