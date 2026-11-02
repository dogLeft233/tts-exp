"""Scientific contracts for fixed-video waveform spectral intervention."""
import importlib.util
from pathlib import Path
import numpy as np
s=importlib.util.spec_from_file_location('eq',Path(__file__).parents[1]/'scripts/experiments/syncnet_fixed_pcm_eq_20261025.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_sham_exact_and_phase():
    rng=np.random.default_rng(1);x=rng.integers(-5000,5000,16000,dtype=np.int16)
    y,d=m.transform(x,np.zeros(257));assert np.array_equal(x,y) and d['phase_error_radians']<1e-12
    db=np.linspace(-3,3,257);y,d=m.transform(x,db)
    assert len(x)==len(y) and d['phase_error_radians']<1e-12 and d['rms_relative_error']<1e-4

def test_loo_excludes_recipient():
    rng=np.random.default_rng(2);shapes={str(i):{a:rng.normal(size=257) for a in ['N','L06','L17']} for i in range(20)}
    p=m.profile_for('0',shapes)
    shapes['0']['N']+=100;shapes['0']['L06']-=100
    q=m.profile_for('0',shapes)
    assert p==q and '0' not in p['donors'] and len(p['donors'])==19
    assert np.linalg.norm(p['direction_db'])==np.linalg.norm(p['reversed_db'])

def test_identity_timing_and_source_bootstrap():
    t=np.arange(16000)/16000;x=np.rint(2000*np.sin(2*np.pi*300*t)*(1+.3*np.sin(2*np.pi*4*t))).astype(np.int16)
    y,p=m.transform(x,np.zeros(257));shape=m.logshape(x).tolist()
    d=m.diagnostics(x,y,p,dict(natural_logshape=shape,tts_logshape=shape))
    assert d['timing_pass'] and d['waveform_lag_samples']==0 and d['centroid_q95_abs_ms']<1e-12
    np.testing.assert_allclose(m.bootstrap([.2]*20)['ci99'],[.2,.2],atol=1e-15)
