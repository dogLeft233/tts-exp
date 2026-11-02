"""Guards for the independent parent audit, including rejection of shifted clocks."""
import importlib.util
from pathlib import Path
import unittest

P = Path(__file__).resolve().parents[1] / "scripts/experiments/syncnet_dataset40_independent_audit_20261025.py"
spec = importlib.util.spec_from_file_location("parent40_audit", P)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ParentAuditTests(unittest.TestCase):
    def test_preemphasis_boundary_and_shift_rejection(self):
        w = {"row": 1, "raw_samples_inclusive": [639, 4079],
             "mfcc_columns_halfopen": [4, 24], "video_frames_halfopen": [1, 6],
             "score_audio_center_s": 2359/16000}
        self.assertEqual(audit.verify_window(w, 1, 5000), 2359/16000)
        shifted = dict(w, score_audio_center_s=w["score_audio_center_s"]+.04)
        with self.assertRaises(AssertionError):
            audit.verify_window(shifted, 1, 5000)
        with self.assertRaises(AssertionError):
            audit.verify_window(w, 1, 4079)

    def test_padding_orientation_and_eps(self):
        import numpy as np
        a = np.array([[1., 2.], [3., 4.]], dtype=np.float32)
        v = np.array([[6., 5.], [4., 3.]], dtype=np.float32)
        got = audit.distance_numpy(a, v)
        for i in range(2):
            for k in range(31):
                index = i+k-15
                target = a[index] if 0 <= index < 2 else np.zeros(2, dtype=np.float32)
                expected = np.linalg.norm(v[i]-target+np.float32(1e-6))
                self.assertAlmostEqual(float(got[i, k]), float(expected), places=6)
        self.assertNotEqual(got[0, 14], got[0, 16])


if __name__ == "__main__":
    unittest.main()
