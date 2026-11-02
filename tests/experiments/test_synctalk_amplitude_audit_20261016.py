"""Evidence-integrity regression: altered arrays must never pass verification."""
import hashlib
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

MODULE = Path(__file__).resolve().parents[2] / 'scripts/experiments/synctalk_amplitude_audit_20261016/validate.py'
spec = importlib.util.spec_from_file_location('synctalk_audit_validator', MODULE)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class EvidenceIntegrityTest(unittest.TestCase):
    def test_raw_array_readback_detects_corruption(self):
        with TemporaryDirectory() as directory:
            x = np.array([[.2, -.1], [.5, .3]], dtype=np.float32)
            p = Path(directory) / 'stage.npy'
            np.save(p, x)
            row = {'shape': [2,2], 'dtype': 'float32', 'sha256': hashlib.sha256(x.tobytes()).hexdigest(),
                   'rms': np.sqrt(np.mean(x.astype(float)**2)), 'peak': float(np.max(np.abs(x)))}
            np.testing.assert_array_equal(validator.verify_stage(row, p), x)
            damaged = x.copy()
            damaged[1,1] += .01
            np.save(p, damaged)
            with self.assertRaisesRegex(AssertionError, 'hash mismatch'):
                validator.verify_stage(row, p)
