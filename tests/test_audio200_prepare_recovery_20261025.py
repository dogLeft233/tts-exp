import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('repair',Path(__file__).resolve().parents[1]/'scripts/experiments/audio200_prepare_recovery_20261025.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class RecoveryContracts(unittest.TestCase):
    def test_only_guard_before_any_score_is_accepted(self):
        terminal=dict(type='RuntimeError',message='other GPU process present: [other]',traceback='runner.score() self.assert_gpu_idle()')
        summary=dict(status='INCOMPLETE',expected=10,completed=0,failed=0,pending=10,terminal_error=terminal)
        receipt=dict(status='FAILED',returncode=1)
        m.guard_only_failure(summary,terminal,receipt)
        bad=copy.deepcopy(summary);bad['completed']=1
        with self.assertRaises(AssertionError):m.guard_only_failure(bad,terminal,receipt)
        bad=copy.deepcopy(terminal);bad['message']='numeric failure'
        with self.assertRaises(AssertionError):m.guard_only_failure(summary,bad,receipt)

    def test_route_only_diff_and_generation_change_rejected(self):
        oldroot,newroot=Path('/old'),Path('/new')
        review=newroot/'reviewed_recovery.json'
        old=dict(run=str(oldroot),seed_bases=[1,2,3],batches=[dict(id=f'batch{i:03d}',split='fit',groups=[dict(arm='N',path=str(oldroot/str(i)/'N'))],media_manifest_path=str(oldroot/str(i)/'media.json')) for i in range(4)])
        new=copy.deepcopy(old);new.update(run=str(newroot),original_protocol_path=str(oldroot/'protocol.json'),original_protocol_sha256=m.OLD_SHA,
            material_roots=[str(oldroot),str(newroot)],reviewed_recovery_path=str(review),reviewed_recovery_sha256='r')
        new['batches'][3]['groups'][0]['path']=str(newroot/'batches/fit/batch003/N')
        new['batches'][3]['media_manifest_path']=str(newroot/'batches/fit/batch003/media_manifest.json')
        m.check_protocol_diff(old,new,review,'r')
        new['seed_bases'][0]=99
        with self.assertRaises(AssertionError):m.check_protocol_diff(old,new,review,'r')

    def test_exact_copy_never_overwrites_and_detects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'old';b=Path(tmp)/'new';a.write_bytes(b'original')
            self.assertEqual(m.exact_copy(a,b),m.sha(a))
            with self.assertRaises(FileExistsError):m.exact_copy(a,b)
            b.write_bytes(b'changed')
            with self.assertRaises(ValueError):m.verify({str(b):m.sha(a)})


if __name__=='__main__':unittest.main()
