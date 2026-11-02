import importlib.util
from pathlib import Path
import unittest
import numpy as np

path=Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_may_gain_inference_20261016/analyze_hubert_may.py'
spec=importlib.util.spec_from_file_location('hubert_may_clock',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class SourceClockContract(unittest.TestCase):
    def test_two_hubert_rows_per_video_frame_and_reject_missing_support(self):
        x=np.arange(12).reshape(12,1)
        np.testing.assert_array_equal(module.paired_slice(x,2,3),x[4:10])
        with self.assertRaisesRegex(AssertionError,'Unsupported source-clock slice'):
            module.paired_slice(x[:9],2,3)


if __name__=='__main__':unittest.main()
