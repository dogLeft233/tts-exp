import importlib.util,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'scripts/experiments/synctalk_multispeaker_full_20261017/other_identities_controller_preview_v3.py'
s=importlib.util.spec_from_file_location('preview_controller',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Routing(unittest.TestCase):
 def test_existing_jobs_never_launch(self):
  for state in ['running','starting','completed']:self.assertEqual(m.job_route(False,state),'observe_existing')
  self.assertEqual(m.job_route(True,'completed'),'reuse_verified')
 def test_only_absent_job_launches(self):
  self.assertEqual(m.job_route(False,None),'launch_new')
  with self.assertRaises(AssertionError):m.job_route(False,'lost')
 def test_unequal_complete_phase_keeps_preview_boundary(self):
  phase={'identity':'PatrickLeahy','passed':False,'status':'CONTROL_FAILED_UNEQUAL_ACTUAL_OPTIMIZER_UPDATES'}
  receipts={g:{'epoch':10,'passed':True} for g in [0,-6,6]}
  self.assertTrue(m.preview_ready(phase,receipts));self.assertFalse(phase['passed'])
  receipts[6]['epoch']=9;self.assertFalse(m.preview_ready(phase,receipts))
if __name__=='__main__':unittest.main()
