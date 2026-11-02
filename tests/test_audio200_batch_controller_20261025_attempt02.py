import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('controller02',Path(__file__).resolve().parents[1]/'scripts/experiments/audio200_batch_controller_20261025_attempt02.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class RecoveryControllerContracts(unittest.TestCase):
    def test_cumulative_cap_counts_both_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'old';b=root/'new';a.mkdir();b.mkdir()
            (a/'x').write_bytes(b'123');(b/'y').write_bytes(b'4567')
            self.assertEqual(m.material_bytes([a,b]),7)
            with patch.object(m,'MATERIAL_ROOTS',[str(a),str(b)]),patch.object(m,'MATERIAL_CAP',6):
                with self.assertRaisesRegex(RuntimeError,'combined'):
                    m.material_guard()

    def test_failed_or_unresolved_task_never_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);m.put(p/'producer_launch.json',dict(command=['unused']))
            with patch.object(m.subprocess,'Popen') as called:
                with self.assertRaisesRegex(RuntimeError,'unresolved'):
                    m.checked(p,p,'producer',['unused'],float('inf'),1)
                called.assert_not_called()
            m.put(p/'producer_exit.json',dict(returncode=1,error='actual failure'))
            with patch.object(m.subprocess,'Popen') as called:
                with self.assertRaisesRegex(RuntimeError,'previous failure'):
                    m.checked(p,p,'producer',['unused'],float('inf'),1)
                called.assert_not_called()

    def test_existing_controller_completion_remains_verifiable(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);old=p/'old.log';old.write_text('complete')
            receipt=dict(command=['same'],returncode=0,error=None,log=str(old),log_sha256=m.sha(old))
            m.put(p/'producer_exit.json',receipt)
            with patch.object(m.subprocess,'Popen') as called:
                self.assertEqual(m.checked(p,p,'producer',['same'],float('inf'),1)['returncode'],0)
                called.assert_not_called()
            old.write_text('drift')
            with self.assertRaises(ValueError):
                m.checked(p,p,'producer',['same'],float('inf'),1)

    def test_training_not_authorized(self):
        with self.assertRaises(AssertionError):
            m.validate_authorization(dict(allow_media_generation=True,allow_training=True))

    def test_terminal_requires_matching_final_content_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);data=p/'dataset_index.json';data.write_text('verified')
            manifest=p/'final_manifest.json'
            m.put(manifest,dict(files=[dict(path=data.name,sha256=m.sha(data),bytes=data.stat().st_size)]))
            m.put(p/'final_status.json',dict(status=m.FINAL_PASS,no_training=True,units=4200,source_count=600,speaker_count=200,
                reused=280,new=3920,pairs=9000,final_manifest_sha256=m.sha(manifest)))
            self.assertEqual(m.accept_final_output(p),p/'final_status.json')
            data.write_text('tampered')
            with self.assertRaises(ValueError):m.accept_final_output(p)


if __name__=='__main__':unittest.main()
