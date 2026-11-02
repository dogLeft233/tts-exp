import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_multispeaker_full_20261017'))
import numpy as np
from measure_video_changes import differences

class PixelDifference(unittest.TestCase):
    def test_unsigned_pixels_are_compared_without_wrap_and_count_pixels_once(self):
        first=np.zeros((2,2,3),dtype=np.uint8);second=first.copy();second[0,0]=[255,128,1]
        forward=differences(first,second);reverse=differences(second,first)
        self.assertEqual(forward,reverse);self.assertEqual(forward['absolute_sum'],384)
        self.assertEqual(forward['squared_sum'],255**2+128**2+1);self.assertEqual(forward['changed_pixels'],1)
        self.assertEqual(forward['pixels'],4);self.assertEqual(forward['channels'],12)
        self.assertEqual(differences(first,first)['squared_sum'],0)

if __name__=='__main__':unittest.main()
