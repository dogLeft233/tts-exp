import importlib.util,tempfile,unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
p=Path(__file__).resolve().parents[2]/'scripts/experiments/synctalk_multispeaker_full_20261017/cloud_archive_readback.py'
spec=importlib.util.spec_from_file_location('cloud_readback',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class CloudReadbackControls(unittest.TestCase):
    def test_concurrent_parts_have_distinct_temporary_paths(self):
        with tempfile.TemporaryDirectory() as d:
            targets=[Path(d)/f'bundle.part{i:03d}' for i in range(3)]
            paths=[mod.staging_path(p) for p in targets];self.assertEqual(len(set(paths)),3)
            def write(i):
                with paths[i].open('xb') as f:f.write(bytes([i])*1000)
                paths[i].rename(targets[i])
            with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(write,range(3)))
            for i,target in enumerate(targets):self.assertEqual(target.read_bytes(),bytes([i])*1000)
if __name__=='__main__':unittest.main()
