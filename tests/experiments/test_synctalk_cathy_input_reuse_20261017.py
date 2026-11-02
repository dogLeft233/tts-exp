import copy,importlib.util,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_multispeaker_full_20261017/prepare_cathy_test_replacement.py'
spec=importlib.util.spec_from_file_location('cathy_reuse',p);mod=importlib.util.module_from_spec(spec)
import sys
sys.path.insert(0,str(p.parent));spec.loader.exec_module(mod)
class UnchangedMainControls(unittest.TestCase):
    def original(self):return dict(recordings=[dict(role='main',source={'file':'main.mp4','sha256':'source','seconds':295},pcm_sha256='pcm',frame_rgb_sequence_sha256='rgb',frames=7375,audio_samples=4720000,sha256='container-A'),dict(role='test',source={'file':'old_test.mp4'})])
    def test_new_test_and_container_id_do_not_change_training_input(self):
        original=self.original();candidate=copy.deepcopy(original);candidate['recordings'][0]['sha256']='container-B';candidate['recordings'][1]['source']['file']='new_test.mp4'
        mod.main_equivalent(original,candidate)
    def test_changed_PCM_blocks_training_reuse(self):
        original=self.original();candidate=copy.deepcopy(original);candidate['recordings'][0]['pcm_sha256']='altered'
        with self.assertRaises(AssertionError):mod.main_equivalent(original,candidate)
    def test_changed_original_recording_blocks_reuse(self):
        original=self.original();candidate=copy.deepcopy(original);candidate['recordings'][0]['source']['file']='other.mp4'
        with self.assertRaises(AssertionError):mod.main_equivalent(original,candidate)
if __name__=='__main__':unittest.main()
