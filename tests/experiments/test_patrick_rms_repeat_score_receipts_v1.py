"""Synthetic gate tests only; no fine/coarse experimental result is generated."""
import copy,importlib.util,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_multispeaker_full_20261017/patrick_rms_repeat_score_receipts_v1.py'
s=importlib.util.spec_from_file_location('repeat_score_gate',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class RepeatScoreGateTests(unittest.TestCase):
 def ready(self):
  receipts={}
  for gain in (-6,0,6):
   receipts[str(gain)]=dict(passed=True,identity='PatrickLeahy',seed=42,gain_db=gain,trial_version=m.VERSION,independent_repeat=True,old_fine_resumed=False,trial_contract_sha256='c'*64,normalization_contract_sha256='c'*64,phase='coarse',epoch=10,global_step=60000,epoch_attempts=6000,epoch_successful_updates=5999,epoch_amp_skips=1,original_full_checkpoint_restored=True,new_independent_initialization=False,cumulative_successful_updates=59990,phase_successful_updates=59990,phase_start_updates=0,checkpoint_sha256='a'*64,trace_sha256='b'*64,normalized_binding_sha256='d'*64,checkpoint_path=str(m.RUN/'training/PatrickLeahy/seed42'/f'gain_{gain:+d}'/'coarse/ep0010/attempt01/workspace/checkpoints/ngp_ep0010.pth'))
  return dict(passed=True,identity='PatrickLeahy',version=m.VERSION,contract_sha256='c'*64,no_training_job_active=True,new_27_grid_required=True,final_receipts=receipts,phase=dict(passed=False,status='CONTROL_FAILED_UNEQUAL_ACTUAL_OPTIMIZER_UPDATES'))
 def render(self):
  return dict(passed=True,rgb_and_pcm_readback_exact=True,experiment_version=m.VERSION,normalization_contract_sha256='c'*64,identity='PatrickLeahy',seed=42,phase='coarse',epoch=10,formal_fine17_model=False,train_gain_db=0,driver_gain_db=-6,ema=dict(passed=True,gain_db=0,checkpoint_sha256='a'*64,ema_updates=60,epoch=10,global_step=60000),frames=[dict(index=i,source_audio_id=i+50) for i in range(1500)])
 def test_accept_preserving_failed_control(self):
  r=self.ready();before=copy.deepcopy(r);self.assertTrue(m.validate_ready(r,'c'*64));self.assertEqual(before,r)
 def test_reject_old_or_partial_ready(self):
  for field,value in [('version','patrick_ave_rms_coarse_v1'),('no_training_job_active',False),('contract_sha256','x'*64)]:
   r=self.ready();r[field]=value
   with self.assertRaises(AssertionError):m.validate_ready(r,'c'*64)
  r=self.ready();del r['final_receipts']['6']
  with self.assertRaises(AssertionError):m.validate_ready(r,'c'*64)
 def test_reject_stale_phase_initialization_and_checkpoint(self):
  for field,value in [('epoch',3),('global_step',18000),('new_independent_initialization',True),('trial_version','patrick_ave_rms_coarse_v1'),('checkpoint_path','/tmp/ngp_ep0010.pth'),('cumulative_successful_updates',60001)]:
   r=self.ready();r['final_receipts']['0'][field]=value
   with self.assertRaises(AssertionError):m.validate_ready(r,'c'*64)
 def test_accept_exact_new_render(self):
  self.assertTrue(m.validate_render(self.render(),dict(checkpoints={'0':dict(sha256='a'*64)}),'c'*64,0,-6))
 def test_reject_old_video_wrong_ema_or_frames(self):
  cases=[lambda r:r.update(experiment_version='patrick_ave_rms_coarse_v1'),lambda r:r.update(driver_gain_db=6),lambda r:r['ema'].update(checkpoint_sha256='e'*64),lambda r:r['ema'].update(ema_updates=102),lambda r:r['frames'].reverse()]
  for change in cases:
   r=self.render();change(r)
   with self.assertRaises(AssertionError):m.validate_render(r,dict(checkpoints={'0':dict(sha256='a'*64)}),'c'*64,0,-6)
 def test_actual_live_training_is_ineligible(self):
  out=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_ave_rms_coarse_repeat_v1')
  import json
  a=json.loads((out/'active_controller.json').read_text());state=json.loads((out/a['receipt_relative']).read_text())
  with self.assertRaises((AssertionError,KeyError)):m.validate_ready(state,'c'*64)
  self.assertFalse((out/'coarse_ready_for_27_grid.json').exists())
if __name__=='__main__':unittest.main()
