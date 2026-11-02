import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/experiments/synctalk_multispeaker_full_20261017'))
from coarse_preview_controller_v1 import order,validate_aggregate

class TestPreviewSchedule(unittest.TestCase):
    def test_baseline_first_and_every_video_once(self):
        self.assertEqual(order()[:3],[(0,0),(-6,0),(6,0)])
        self.assertEqual(len(order()),9);self.assertEqual(set(order()),{(a,b) for a in [-6,0,6] for b in [-6,0,6]})
    def test_partial_or_duplicate_grid_rejected(self):
        with self.assertRaises(AssertionError):validate_aggregate([])
        with self.assertRaises(AssertionError):validate_aggregate([dict(training_gain_db=0,driver_gain_db=0,evaluation_gain_db=0)]*27)

if __name__=='__main__':unittest.main()
