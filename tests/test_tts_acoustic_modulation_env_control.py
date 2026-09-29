import importlib.util
from pathlib import Path
import numpy as np

P=Path(__file__).resolve().parents[1]/'scripts/experiments/tts_acoustic_modulation_env_control.py'
spec=importlib.util.spec_from_file_location('env_control',P); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_zero_and_identity_twenty_iterations():
    for x in [np.zeros(1),np.zeros(2003),np.random.default_rng(42).normal(0,.1,3001)]:
        y,gain,trace=m.envelope_match(x,x)
        assert len(trace['iterations'])==20
        np.testing.assert_allclose(y,x,atol=1e-14)
        assert np.isfinite(y).all() and len(y)==len(x)
        assert trace['zero_samples_preserved']


def test_gain_tracking_zero_samples_and_shared_peak():
    x=np.random.default_rng(43).normal(0,.1,3001); x[::17]=0
    target=x*(1+.4*np.sin(np.arange(len(x))*.01))
    y,gain,trace=m.envelope_match(x,target)
    np.testing.assert_array_equal(y,x*gain)
    assert np.all(y[x==0]==0)
    assert abs(m.v1.rms(y)/m.v1.rms(x)-1)<1e-12
    factor=m.shared_headroom({'N':{'raw':x,'env':y},'T':{'raw':x*20,'env':y*20}})
    assert max(np.max(abs(z*factor)) for z in [x,y,x*20,y*20]) <= .98000000001


def test_hann_envelope_against_explicit_windows():
    x=np.arange(17)/17
    pad=np.pad(x*x,(256,255),mode='reflect')
    expected=np.sqrt(np.array([sum(pad[i:i+512]*m.WINDOW[::-1]) for i in range(len(x))])+1e-16)
    np.testing.assert_allclose(m.envelope(x),expected,atol=1e-15)
