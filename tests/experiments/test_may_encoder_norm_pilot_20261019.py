import importlib.util
from pathlib import Path
import unittest
P=Path(__file__).resolve().parents[2]/'scripts/experiments/may_encoder_norm_pilot_20261019/protocol.py'
s=importlib.util.spec_from_file_location('may_norm_protocol',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Controls(unittest.TestCase):
 def pair(self):
  x=dict(encoder='same',weight_sha256='w',config_sha256='c',pcm_sha256='p',generator_initial_sha256='i',steps=12000,frame_order_sha256='f',ray_schedule_sha256='r',source_ids=[4929,5679],precision='FP32',lr_policy='original prefix',feature_layout='paired50Hz',normalize=True)
  return x,{**x,'normalize':False}
 def test_rejects_encoder_or_weight_replacement(self):
  for key in ['encoder','weight_sha256','config_sha256']:
   a,b=self.pair();b[key]='different'
   with self.assertRaisesRegex(ValueError,key):m.validate_pair(a,b)
 def test_rejects_changed_update_clock_and_sampling(self):
  for key in ['steps','source_ids','ray_schedule_sha256','frame_order_sha256','generator_initial_sha256','precision','lr_policy','feature_layout','pcm_sha256']:
   a,b=self.pair();b[key]='changed'
   with self.assertRaisesRegex(ValueError,key):m.validate_pair(a,b)
 def test_valid_same_encoder_switch(self):
  self.assertTrue(m.validate_pair(*self.pair()))
 @unittest.skipUnless((m.DATA/'manifest.json').exists() and (m.SURVEY/'models/hubert_base/model_source.json').exists() and (m.SURVEY/'models/wav2vec2_base/model_source.json').exists(),'external scientific assets are not installed')
 def test_actual_frozen_budget_and_support(self):
  p=m.build();self.assertEqual(len(p['arms']),4);self.assertEqual(p['steps_per_model'],12000)
  self.assertEqual([c['id'] for c in p['test_clips']],['test_00','test_03'])
if __name__=='__main__':unittest.main()
