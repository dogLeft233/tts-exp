import importlib.util
from pathlib import Path
import unittest
import numpy as np

P=Path(__file__).resolve().parents[2]/'scripts/experiments/gain_shared_direction_20261018/core.py'
spec=importlib.util.spec_from_file_location('gaincore',P)
core=importlib.util.module_from_spec(spec); spec.loader.exec_module(core)

class Contracts(unittest.TestCase):
    def test_identity_and_radial_change(self):
        x=np.random.default_rng(1).uniform(.1,1,(8,512)).astype('float32')
        out,q=core.make_conditions(x,x*2,np.zeros(512))
        np.testing.assert_array_equal(out['BASE'],x)
        for k in ['COMMON','RESIDUAL','FULL_DIRECTION']:
            np.testing.assert_allclose(out[k],x,rtol=1e-6)
        self.assertLess(q['reconstruction_max_abs'],1e-12)

    def test_common_is_test_gain_independent(self):
        rng=np.random.default_rng(2); x=rng.uniform(.1,1,(9,512)).astype('float32')
        y=rng.uniform(.1,1,(9,512)).astype('float32'); g=rng.normal(0,.001,512)
        a,_=core.make_conditions(x,y,g); b,_=core.make_conditions(x,y[::-1],g)
        np.testing.assert_array_equal(a['COMMON'],b['COMMON'])
        self.assertFalse(np.array_equal(a['RESIDUAL'],b['RESIDUAL']))

    def test_speaker_disjoint_and_equal_weight(self):
        with self.assertRaises(ValueError): core.check_disjoint(['a'],['a'])
        x=np.ones((3,512)); y=x.copy(); y[:,0]=2
        g=core.fit_common([('a',x,y),('b',x,y)])
        g2=core.fit_common([('a',np.repeat(x,2,0),np.repeat(y,2,0)),('b',x,y)])
        np.testing.assert_allclose(g,g2,atol=1e-15)

    def test_nonnegative_sensitivity_and_norm(self):
        x=np.ones((8,512),dtype='float32'); x[:,0]=0
        y=x.copy(); y[:,1]=2; g=np.zeros(512); g[0]=-.01
        out,q=core.make_conditions(x,y,g)
        self.assertTrue(np.any(out['COMMON']<0))
        self.assertTrue(np.all(out['COMMON_NONNEG']>=0))
        self.assertLess(q['conditions']['COMMON']['norm_relative_error'],1e-6)

if __name__=='__main__': unittest.main()
