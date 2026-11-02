import hashlib,io,json,sys,tarfile,tempfile,unittest
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_multispeaker_full_20261017'
sys.path.insert(0,str(SOURCE))
import reclaim_stage1_raw_F as cache

class CacheRetirement(unittest.TestCase):
    def fixture(self,root):
        identity='PatrickLeahy';packages=root/'preprocessed_packages';packages.mkdir()
        values={'stage1_receipt.json':b'{"passed":true}','train/ori_imgs/0.jpg':b'original image','train/parsing/0.png':b'original parsing'}
        rows=[]
        for name,raw in values.items():
            path=root/'preprocessed'/identity/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
            rows.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
        archive=packages/f'{identity}_stage1.tar.gz'
        with tarfile.open(archive,'w:gz') as stream:
            for name,raw in values.items():
                item=tarfile.TarInfo(identity+'/'+name);item.size=len(raw);stream.addfile(item,io.BytesIO(raw))
        manifest=packages/f'{identity}_stage1_manifest.json';manifest.write_text(json.dumps(rows))
        source=dict(bytes=archive.stat().st_size,sha256=cache.sha(archive),manifest_sha256=cache.sha(manifest))
        (packages/f'{identity}_verification.json').write_text(json.dumps(dict(identity=identity,passed=True,all_extracted_files_verified=True,files=len(rows),archive_sha256=source['sha256'])))
        return identity,source,rows,archive

    def test_only_verified_duplicate_files_are_retired(self):
        with tempfile.TemporaryDirectory(prefix='cache space ') as tmp:
            root=Path(tmp);identity,source,rows,archive=self.fixture(root)
            self.assertEqual(cache.verify_backup(root,identity,source),rows)
            result=cache.reclaim(root,identity,rows,{'stage1_receipt.json'})
            self.assertEqual(result['retired_files'],2)
            self.assertTrue((root/'preprocessed'/identity/'stage1_receipt.json').is_file())
            self.assertTrue(archive.is_file());self.assertFalse((root/'preprocessed'/identity/'train/ori_imgs/0.jpg').exists())

    def test_mutated_archive_blocks_every_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);identity,source,rows,archive=self.fixture(root)
            with archive.open('ab') as f:f.write(b'corrupt')
            with self.assertRaises(AssertionError):cache.verify_backup(root,identity,source)
            self.assertTrue((root/'preprocessed'/identity/'train/ori_imgs/0.jpg').is_file())

    def test_changed_loose_file_blocks_every_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);identity,source,rows,archive=self.fixture(root)
            (root/'preprocessed'/identity/'train/ori_imgs/0.jpg').write_bytes(b'changed')
            with self.assertRaises(AssertionError):cache.verify_backup(root,identity,source)
            self.assertTrue((root/'preprocessed'/identity/'train/parsing/0.png').is_file())

if __name__=='__main__':unittest.main()
