"""CPU-only tests; fixture receipts are not experimental results."""
import ast,copy,hashlib,importlib.util,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'scripts/experiments/synctalk_multispeaker_full_20261017/patrick_rms_repeat_controller_revision02.py'
OUT=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_ave_rms_coarse_repeat_v1')
spec=importlib.util.spec_from_file_location('metadata_handoff',SOURCE)
ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)

class MetadataHandoffTests(unittest.TestCase):
 def setUp(self):
  self.r=json.loads((OUT/'controller_revision02_metadata_handoff_v1/gain_+0_ep0001_derived_receipt.json').read_text())
 def test_only_exact_completed_prefix_can_continue(self):
  self.assertEqual(ctl.validate_prefix([self.r]),[(1,0)])
  for rows in ([dict(self.r,epoch=2)],[self.r,self.r],[dict(self.r,passed=False)]):
   with self.assertRaises(AssertionError):ctl.validate_prefix(rows)
  self.assertEqual(ctl.ORDER[1],(1,-6))
 def test_remote_checkpoint_rejects_escaped_namespace(self):
  for path in ('C:\\checkpoint.pth',ctl.RUN+'/training/../ngp_ep0001.pth','/tmp/ngp_ep0001.pth'):
   with self.assertRaises((AssertionError,ValueError)):ctl.remote_checkpoint(path)
 def test_frozen_wrapper_provenance_lookup_after_numeric_completion(self):
  # Execute only the frozen wrapper's function with a stub numerical epoch.
  # It reproduces the old missing alias failure and succeeds after adding that alias.
  tree=ast.parse((OUT/'sources/train_patrick_rms_repeat_epoch_v1.py').read_text())
  function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
  with tempfile.TemporaryDirectory() as temp:
   run=Path(temp)/'new';run.mkdir();data=Path(temp)/'same_input_dataset';data.mkdir()
   (run/'contract.json').write_text('{}');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
   (run/'qualification').mkdir();(run/'qualification/result.json').write_text(json.dumps(dict(passed=True,contract_sha256=sha(run/'contract.json'))))
   binding=data/'PatrickLeahy/train/gain_0/binding.json';binding.parent.mkdir(parents=True);binding.write_text('{"same_verified_binding":true}')
   receipt=run/'training/PatrickLeahy/seed42/gain_+0/coarse/ep0001/attempt01/epoch_receipt.json'
   def numeric_epoch(*args,**kwargs):
    receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(dict(completed_numeric_epoch=True)))
   stub=types.SimpleNamespace(run=numeric_epoch)
   ns=dict(RUN=run,install=lambda:{'parent_normalization_contract_sha256':'parent'},sha=sha,json=json)
   exec(compile(ast.Module(body=[function],type_ignores=[]),'<frozen-wrapper-function>','exec'),ns)
   args=types.SimpleNamespace(identity='PatrickLeahy',seed=42,phase='coarse',epoch=1,attempt=1,gain=0,checkpoint=None,resume=None,reference_trace=None)
   with patch.dict('sys.modules',{'train_other_identity_epoch_v1':stub}):
    with self.assertRaises(FileNotFoundError):ns['run'](args)
    self.assertTrue(json.loads(receipt.read_text())['completed_numeric_epoch'])
    # Windows symlink privileges are not required for this CPU fixture: exact copied
    # binding represents the already independently hashed Linux alias target.
    target=run/'datasets/PatrickLeahy/train/gain_0';target.mkdir(parents=True);(target/'binding.json').write_bytes(binding.read_bytes())
    ns['run'](args)
   result=json.loads(receipt.read_text());self.assertTrue(result['completed_numeric_epoch'])
   self.assertEqual(result['normalized_binding_sha256'],sha(binding));self.assertTrue(result['independent_repeat'])

if __name__=='__main__':unittest.main()
