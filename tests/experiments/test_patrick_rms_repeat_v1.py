"""Actual from-scratch repeat admission checks; no GPU recomputation."""
import hashlib,json,os,tarfile,unittest
from pathlib import Path
R=Path(os.environ.get('PATRICK_RMS_REPEAT','F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_ave_rms_coarse_repeat_v1'))
@unittest.skipUnless((R/'qualification/result.json').exists(),'actual repeat qualification not available')
class RepeatAdmission(unittest.TestCase):
 def test_same_complete_initial_and_no_recovery_policy(self):
  c=json.loads((R/'contract.json').read_text(encoding='utf-8'))
  self.assertEqual(c['initial_checkpoint_sha256'],'d6ce1f369bd2ee1695288c93f1602fcf181f0329313c710388e6b7c70071be5c')
  self.assertFalse(c['failed_state_warmstart']);self.assertFalse(c['density_cache_fallback'])
  self.assertEqual(c['normalization']['target_rms'],.1)
  for n,s in c['sources'].items():self.assertEqual(hashlib.sha256((R/'sources'/n).read_bytes()).hexdigest(),s)
 def test_actual_six_attempts_full_roundtrip_and_sampling(self):
  q=json.loads((R/'qualification/result.json').read_text(encoding='utf-8'))
  self.assertTrue(q['passed'] and q['initial_unchanged'] and q['actual_sampling_equal'])
  self.assertEqual(q['formal_attempts'],0);self.assertEqual(q['total_attempts'],6)
  fields=['frame_index','source_audio_id','lip_mode','pixel_index_sha256','actual_rays','rect','optimizer_lr']
  samples=[]
  with tarfile.open(R/'qualification/artifacts.tar.gz','r:gz') as t:
   for c in q['cases']:
    self.assertTrue(c['full_state_roundtrip_exact'])
    raw=t.extractfile(f"gain_{c['gain_db']:+d}/coarse/ep0001/attempt01/attempts.jsonl").read()
    self.assertEqual(hashlib.sha256(raw).hexdigest(),c['trace_sha256'])
    rows=[json.loads(x) for x in raw.splitlines()];self.assertEqual(len(rows),2)
    self.assertEqual(sum(x['successful_update'] for x in rows),c['actual_successful_updates'])
    samples.append([{k:x[k] for k in fields} for x in rows])
  self.assertEqual(samples[0],samples[1]);self.assertEqual(samples[0],samples[2])
 def test_fresh_every_archive_member_and_full_checkpoint_bytes(self):
  root=R/'qualification';rows=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
  expected={x['path']:x for x in rows if 'workspace/checkpoints/' not in x['path']};seen=set()
  with tarfile.open(root/'artifacts.tar.gz','r:gz') as t:
   for m in t:
    self.assertIn(m.name,expected);raw=t.extractfile(m).read();row=expected[m.name]
    self.assertEqual(len(raw),row['bytes']);self.assertEqual(hashlib.sha256(raw).hexdigest(),row['sha256']);seen.add(m.name)
  self.assertEqual(seen,set(expected))
  for row in rows:
   if 'workspace/checkpoints/' in row['path']:
    p=root/row['path'];self.assertEqual(p.stat().st_size,row['bytes']);self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),row['sha256'])
if __name__=='__main__':unittest.main()
