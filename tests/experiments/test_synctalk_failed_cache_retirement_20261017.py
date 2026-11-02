import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_multispeaker_full_20261017/retire_failed_canonical_remote.py'
spec=importlib.util.spec_from_file_location('failed_cache',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class FailedCacheControls(unittest.TestCase):
    def fixture(self,directory):
        root=Path(directory)/'synctalk-multispeaker-data-20261017';folder=root/'material/DebbieDingell';folder.mkdir(parents=True)
        rows=[]
        for role in ['main','test']:
            file=folder/(role+'.mkv');file.write_bytes(role.encode()*20);rows.append(dict(role=role,bytes=file.stat().st_size,sha256=mod.sha(file)))
        (folder/'material_receipt.json').write_text(json.dumps(dict(recordings=rows)))
        geometry=root/'prepared/DebbieDingell/geometry_v2/geometry_qualification.json';geometry.parent.mkdir(parents=True)
        geometry.write_text(json.dumps(dict(splits=[dict(passed=False)])))
        return root,folder,dict(passed=True,identity='DebbieDingell',failed_pair_evidence_independently_archived=True,failed_geometry_sha256=mod.sha(geometry),files=rows,failure_evidence_archive_sha256='sealed')
    def test_all_bytes_verified_before_first_unlink(self):
        with tempfile.TemporaryDirectory() as d:
            root,folder,proof=self.fixture(d);(folder/'test.mkv').write_bytes(b'changed')
            with self.assertRaises(AssertionError):mod.run(root,proof,False)
            self.assertTrue((folder/'main.mkv').exists());self.assertTrue((folder/'test.mkv').exists())
    def test_qualified_geometry_is_not_failed_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root,folder,proof=self.fixture(d);p=root/'prepared/DebbieDingell/geometry_v2/geometry_qualification.json';p.write_text(json.dumps(dict(passed=True,splits=[])));proof['failed_geometry_sha256']=mod.sha(p)
            with self.assertRaises(AssertionError):mod.run(root,proof,False)
            self.assertTrue((folder/'main.mkv').exists())
    def test_only_two_owned_copies_retired(self):
        with tempfile.TemporaryDirectory() as d:
            root,folder,proof=self.fixture(d);(folder/'test_original.wav').write_bytes(b'pcm');result=mod.run(root,proof,False)
            self.assertTrue(result['passed']);self.assertTrue((folder/'material_receipt.json').exists());self.assertTrue((folder/'test_original.wav').exists())
            self.assertFalse((folder/'main.mkv').exists());self.assertFalse((folder/'test.mkv').exists())
if __name__=='__main__':unittest.main()
