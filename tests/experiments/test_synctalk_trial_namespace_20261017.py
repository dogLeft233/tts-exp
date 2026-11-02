import importlib.util,json,os,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
FOLDER=Path(os.environ.get('SYNCTALK_TRIAL_TEST_SOURCE',str(Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_multispeaker_full_20261017')))
def module(name):
    spec=importlib.util.spec_from_file_location(name,FOLDER/(name+'.py'));result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
class TrialNamespaceControls(unittest.TestCase):
    def test_unqualified_geometry_blocks_probe(self):
        probe=module('runtime_probe_trial')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'DebbieDingell_recording_roles_v2';p=root/'prepared/DebbieDingell/geometry_v2/geometry_qualification.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(dict(passed=False)))
            with self.assertRaises(AssertionError):probe.run(root,'capacity')
    def test_probe_namespace_and_identity_only(self):
        wrapper=module('runtime_probe_trial')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'DebbieDingell_recording_roles_v2';p=root/'prepared/DebbieDingell/geometry_v2/geometry_qualification.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(dict(passed=True)))
            calls=[];common=types.SimpleNamespace(ROOT=None,OFFICIAL=None);probe=types.SimpleNamespace(ROOT=root,run=calls.append)
            with patch.dict(sys.modules,{'runtime_common':common,'qualify_loading':probe}):wrapper.run(root,'loading')
            self.assertEqual(calls,['DebbieDingell']);self.assertEqual(common.ROOT,root);self.assertEqual(common.OFFICIAL,root/'official')
    def test_seal_rejects_external_symlink(self):
        seal=module('seal_trial_inputs')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'DebbieDingell_recording_roles_v2';base=root/'material/DebbieDingell';base.mkdir(parents=True)
            (base/'independent_material_qa.json').write_text(json.dumps(dict(passed=True)))
            outside=Path(directory)/'outside';outside.write_bytes(b'not-version-owned')
            try:(base/'external.wav').symlink_to(outside)
            except OSError:self.skipTest('OS requires symlink privilege')
            with self.assertRaises(AssertionError):seal.run(root,'DebbieDingell','material')
if __name__=='__main__':unittest.main()
