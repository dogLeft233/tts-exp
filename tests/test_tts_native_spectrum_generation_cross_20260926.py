"""Frozen spectrum engine and scope contracts."""
import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1]/'scripts/experiments/tts_native_spectrum_generation_cross_20260926.py'
spec=importlib.util.spec_from_file_location('spectrum_cross',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_frozen_eq_identity_and_frequency_gain():
    x=np.random.default_rng(614).normal(0,.1,19003)
    y,info=m.cal.eq(x,np.zeros(257))
    np.testing.assert_allclose(y,x,atol=1e-12,rtol=0)
    assert info['design_zero_bins_preserved']

def test_original_waveform_frontend_and_delay_support():
    x=np.random.default_rng(12).integers(-5000,5000,64000,dtype=np.int16)
    w,_=m.base.frontend_windows(x.astype(float)/32768)
    assert len(w)==95
    a=np.random.default_rng(13).normal(size=(100,16)).astype(np.float32)
    shifted=np.zeros_like(a);shifted[5:]=a[:-5]
    assert m.base.summarize(m.base.matrix(a,shifted),'guard20')['best_lag']==5

def test_condition_and_engine_scope():
    assert m.COND==['raw','identity','EQ','ENVeq']
    assert m.base.OUT==m.OUT and m.cal.OUT==m.CAL
    assert m.base.resource_check is m.resource_check
    assert m.base.bootstrap([1,1,5],['A','A','B'])['mean']==3
