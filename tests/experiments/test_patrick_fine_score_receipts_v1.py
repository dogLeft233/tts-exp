import copy
import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[2] / 'scripts/experiments/synctalk_multispeaker_full_20261017/patrick_fine_score_receipts_v1.py'
SPEC = importlib.util.spec_from_file_location('fine_receipts', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FineReceiptTests(unittest.TestCase):
    def setUp(self):
        self.contract = dict(identity='PatrickLeahy', seed=42,
                             training_control_status='CONTROL_FAILED_UNEQUAL_ACTUAL_OPTIMIZER_UPDATES',
                             checkpoints={'-6': {'sha256': 'fine_minus6_state'}})
        self.receipt = dict(passed=True, rgb_and_pcm_readback_exact=True,
                            identity='PatrickLeahy', seed=42, train_gain_db=-6, driver_gain_db=0,
                            training_control_status=self.contract['training_control_status'],
                            frames=[dict(index=i) for i in range(1500)],
                            ema=dict(passed=True, full_state_finite=True, epoch=17,
                                     global_step=102000, ema_updates=102, ema_decay=.95,
                                     checkpoint_sha256='fine_minus6_state',
                                     independent_named_overlay_equals_torch_ema=True,
                                     raw_state_restored_exact=True, buffers_from_full_checkpoint=True),
                            **MODULE.fine_metadata())

    def test_fine17_descriptive_state_accepts_inherited_control_failure(self):
        MODULE.validate_render_receipt(self.receipt, self.contract, -6, 0)

    def test_coarse_or_unverified_checkpoint_is_rejected(self):
        for section, key, value in [(None, 'epoch', 10), (None, 'phase', 'coarse'),
                                    (None, 'parent_control_failed_retained', False),
                                    ('ema', 'checkpoint_sha256', 'coarse_state'),
                                    ('ema', 'ema_updates', 60),
                                    ('ema', 'raw_state_restored_exact', False)]:
            with self.subTest(section=section, key=key):
                changed = copy.deepcopy(self.receipt)
                (changed if section is None else changed[section])[key] = value
                with self.assertRaises(AssertionError):
                    MODULE.validate_render_receipt(changed, self.contract, -6, 0)

    def test_wrong_driver_or_reordered_frames_is_rejected(self):
        with self.assertRaises(AssertionError):
            MODULE.validate_render_receipt(self.receipt, self.contract, -6, 6)
        changed = copy.deepcopy(self.receipt)
        changed['frames'][1]['index'] = 0
        with self.assertRaises(AssertionError):
            MODULE.validate_render_receipt(changed, self.contract, -6, 0)


if __name__ == '__main__':
    unittest.main()
