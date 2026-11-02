import ast,json,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'scripts/experiments/may_encoder_gain_norm_factorial_20261019'
OUT=Path('F:/Datasets/tts-exp/may_encoder_gain_norm_factorial_20261019')
class FactorialControls(unittest.TestCase):
 @unittest.skipUnless((OUT/'protocol.json').exists(),'external frozen protocol not installed')
 def test_complete_added_factorial_without_zero_gain_replay(self):
  p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
  self.assertEqual(len(p['arms']),8);self.assertEqual(len({a['id'] for a in p['arms']}),8)
  expected={(mid,n,g) for mid in ['hubert_base','wav2vec2_base'] for n in [True,False] for g in [-6,6]}
  self.assertEqual({(a['encoder'],a['norm'],a['train_gain_db']) for a in p['arms']},expected)
  self.assertEqual(p['new_formal_updates'],96000);self.assertEqual(p['steps_per_model'],12000)
  self.assertEqual(len(p['imported_zero_gain_arms']),4);self.assertEqual(p['test_gains_db'],[-6,0,6])
 def test_only_train_feature_gain_and_fixed_reference_change(self):
  old=(ROOT/'scripts/experiments/may_encoder_norm_pilot_20261019/train.py').read_text(encoding='utf-8')
  expected=old.replace("ds=dataset(arm['encoder'],'train',arm['norm']);","ds=dataset(arm['encoder'],'train',arm['norm'],arm['train_gain_db']);").replace("ref=reference.open() if key!='hubert_base_on' else None","ref=reference.open()")
  self.assertEqual((SRC/'train.py').read_text(encoding='utf-8'),expected)
  self.assertEqual((SRC/'runtime.py').read_bytes(),(ROOT/'scripts/experiments/may_encoder_norm_pilot_20261019/runtime.py').read_bytes())
 def test_tar_trailer_is_drained_before_wait(self):
  code='''import sys,tempfile,pathlib,hashlib,io,tarfile,base64
sys.path.insert(0,sys.argv[1]);import controller
payload=b'bounded scientific payload';archive=io.BytesIO()
with tarfile.open(fileobj=archive,mode='w') as tar:
 t=tarfile.TarInfo('member.bin');t.size=len(payload);tar.addfile(t,io.BytesIO(payload))
producer='import sys,base64;sys.stdout.buffer.write(base64.b64decode('+repr(base64.b64encode(archive.getvalue()).decode())+')+bytes(100000));sys.stdout.buffer.flush()'
controller.SSH=[sys.executable,'-c',producer]
controller.remote_json=lambda code:[dict(path='member.bin',bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest())]
with tempfile.TemporaryDirectory() as d:
 controller.copy_tree('unused',pathlib.Path(d));assert (pathlib.Path(d)/'member.bin').read_bytes()==payload
'''
  # The old implementation could block on an unread tail; this is a real bounded child pipe.
  r=subprocess.run([sys.executable,'-X','utf8','-c',code,str(SRC)],capture_output=True,timeout=15)
  self.assertEqual(r.returncode,0,r.stderr.decode(errors='replace'))
 def test_runtime_uses_original_precision_and_loss_paths(self):
  t=ast.parse((SRC/'train.py').read_text());names={n.name for n in t.body if isinstance(n,ast.FunctionDef)}
  self.assertIn('arm_run',names);self.assertIn('save_state',names)
  text=(SRC/'train.py').read_text();self.assertIn("loss.backward();finite(m,grad=True)",text);self.assertNotIn('autocast(',text);self.assertNotIn('clip_grad',text)
if __name__=='__main__':unittest.main()
