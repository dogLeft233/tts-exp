import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
p=Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_multispeaker_full_20261017/cloud_archive_readback_stream.py'
spec=importlib.util.spec_from_file_location('stream_cloud',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class StreamReadbackControls(unittest.TestCase):
    def fixture(self,directory):
        root=Path(directory);cloud=root/'cloud';cloud.mkdir();mod.ROOT=root/'workspace';mod.ROOT.mkdir();mod.CLOUD_PREFIX=cloud
        parts=[];whole=b''
        for i,payload in enumerate([b'A'*5000000,b'B'*17000,b'C'*123]):
            name=f'bundle.part{i:03d}';(cloud/name).write_bytes(payload);whole+=payload
            parts.append(dict(file=name,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
        manifest=dict(name='fixture',attempt=1,linux_cloud_directory=str(cloud),parts=parts,bytes=len(whole),sha256=hashlib.sha256(whole).hexdigest())
        path=root/'manifest.json';path.write_text(json.dumps(manifest));return root,cloud,path,manifest
    def test_full_SHA_and_parts_checked_without_payload_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root,cloud,path,manifest=self.fixture(d);receipt=mod.run(path,2)
            self.assertEqual(receipt['sha256'],manifest['sha256']);self.assertEqual(receipt['payload_staged_bytes'],0)
            directory=mod.ROOT/'cloud_readback/fixture/attempt02';self.assertEqual([p.name for p in directory.iterdir()],['receipt.json'])
            proof=root/'proof.json';proof.write_text(json.dumps(dict(passed=True,readback_attempt=2,F_verified_receipt_sha256=mod.sha(directory/'receipt.json'),archive_sha256=receipt['sha256'])))
            mod.retire(path,proof);self.assertTrue((directory/'staging_retirement.json').exists())
    def test_changed_cloud_bytes_fail_without_success_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root,cloud,path,manifest=self.fixture(d);(cloud/'bundle.part001').write_bytes(b'X'*17000)
            with self.assertRaises(AssertionError):mod.run(path)
            self.assertFalse((mod.ROOT/'cloud_readback/fixture/attempt01/receipt.json').exists())
if __name__=='__main__':unittest.main()
