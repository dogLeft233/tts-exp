import importlib.util
from pathlib import Path
import unittest
import torch
import sys

P = Path(__file__).resolve().parents[1]/'scripts/experiments/qwen_fixed_coarse_20261023/probe.py'
spec=importlib.util.spec_from_file_location('coarse_probe',P)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.path.insert(0,str(P.parents[3]))
from scripts.experiments.qwen_fixed_coarse_20261023.audio_checks import timing_ok


class ForceTests(unittest.TestCase):
    def test_timing_rejects_empty_unknown_and_hidden_outlier(self):
        self.assertFalse(timing_ok(True,[],0,0,True))
        self.assertFalse(timing_ok(True,[0,0],1,0,True))
        self.assertFalse(timing_ok(True,[0]*99+[90],0,0,True))
        self.assertFalse(timing_ok(True,[0,10],0,2,True))
        self.assertTrue(timing_ok(True,[0,10,20],0,1,True))

    def test_forces_sequence_and_preserves_rng(self):
        f=m.ForceSequence([2,1])
        s=torch.randn(1,5)
        before=torch.get_rng_state().clone()
        out=f(torch.empty(1,0,dtype=torch.long),s)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertEqual(torch.softmax(out,-1).tolist(),[[0.,0.,1.,0.,0.]])
        self.assertEqual(f(None,s).argmax().item(),1)
        with self.assertRaises(ValueError): f(None,s)

    def test_rejects_invalid_batch_and_token(self):
        with self.assertRaises(ValueError): m.ForceSequence([0])(None,torch.zeros(2,3))
        with self.assertRaises(ValueError): m.ForceSequence([3])(None,torch.zeros(1,3))

    def test_hook_restored_after_exception(self):
        class T:
            def generate(self,**kw): return kw
        t=T(); original=t.generate
        with self.assertRaises(RuntimeError):
            with m.constrain(t,[1]):
                self.assertIn('logits_processor',t.generate())
                raise RuntimeError('failure')
        self.assertEqual(t.generate,original)


if __name__=='__main__': unittest.main()
