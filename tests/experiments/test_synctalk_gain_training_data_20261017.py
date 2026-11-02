"""Reject split leakage and silent validation/test aliasing."""
import ast
import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
path = ROOT / 'scripts/experiments/synctalk_gain_training_data_20261017/prepare.py'
spec = importlib.util.spec_from_file_location('synctalk_gain_data_prepare', path)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
vpath = ROOT / 'scripts/experiments/synctalk_gain_training_data_20261017/validate.py'
vspec = importlib.util.spec_from_file_location('synctalk_gain_data_validate', vpath)
validate = importlib.util.module_from_spec(vspec)
vspec.loader.exec_module(validate)


class SplitContracts(unittest.TestCase):
    def test_rejects_overlapping_audio_support_even_if_supervision_does_not_overlap(self):
        blocks = [dict(source_start_frame=0, source_end_frame_exclusive=100),
                  dict(source_start_frame=90, source_end_frame_exclusive=200)]
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            prepare.validate_split_ranges(blocks, 300, 25)

    def test_rejects_context_only_split(self):
        with self.assertRaisesRegex(ValueError, 'invalid'):
            prepare.validate_split_ranges([dict(source_start_frame=0,
                                                source_end_frame_exclusive=50)], 100, 25)

    def test_rejects_audio_shift_with_unchanged_length(self):
        pcm = np.array([10, 30, -20, 40, 0], dtype=np.int16)
        shifted = np.roll(pcm, 1).astype(np.float64) / 32768
        with self.assertRaisesRegex(ValueError, 'source-clock'):
            validate.verify_gain(shifted, pcm, 0)

    def test_rejects_wrong_gain_label(self):
        pcm = np.array([10, 30, -20, 40, 0], dtype=np.int16)
        with self.assertRaisesRegex(ValueError, 'pure gain'):
            validate.verify_gain(pcm.astype(np.float64) / 32768, pcm, 6)

    def test_rejects_wrong_split_audio_id(self):
        frame = dict(img_id=500, aud_id=500, transform_matrix=np.eye(4).tolist())
        with self.assertRaisesRegex(ValueError, 'audio clock'):
            validate.verify_clock([frame], {500: frame}, dict(source_start_frame=400))


if __name__ == '__main__':
    unittest.main()
