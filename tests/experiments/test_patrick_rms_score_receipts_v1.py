import copy,importlib.util,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_multispeaker_full_20261017/patrick_rms_score_receipts_v1.py'
s=importlib.util.spec_from_file_location('rmsgate',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class ReceiptTests(unittest.TestCase):
 def fixture(self):
  c=dict(checkpoints={'0':dict(sha256='checkpoint')})
  r=dict(passed=True,rgb_and_pcm_readback_exact=True,experiment_version='patrick_ave_rms_coarse_v1',normalization_contract_sha256='contract',identity='PatrickLeahy',seed=42,phase='coarse',epoch=10,formal_fine17_model=False,train_gain_db=0,driver_gain_db=-6,ema=dict(passed=True,gain_db=0,checkpoint_sha256='checkpoint',ema_updates=60,epoch=10,global_step=60000),frames=[dict(index=i,source_audio_id=i+50) for i in range(1500)])
  return r,c
 def test_accept_exact(self):
  r,c=self.fixture();self.assertTrue(m.validate_render(r,c,'contract',0,-6))
 def test_reject_old_phase_gain_contract(self):
  for key,value in [('experiment_version','old'),('phase','fine'),('epoch',17),('normalization_contract_sha256','old'),('train_gain_db',6),('driver_gain_db',6)]:
   r,c=self.fixture();r[key]=value
   with self.assertRaises(AssertionError):m.validate_render(r,c,'contract',0,-6)
 def test_reject_checkpoint_ema_frame_order(self):
  for mutate in [lambda r:r['ema'].update(checkpoint_sha256='old'),lambda r:r['ema'].update(ema_updates=102),lambda r:r['frames'].reverse()]:
   r,c=self.fixture();mutate(r)
   with self.assertRaises(AssertionError):m.validate_render(r,c,'contract',0,-6)
if __name__=='__main__':unittest.main()
