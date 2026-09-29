import importlib.util
from pathlib import Path
import numpy as np
import python_speech_features

P=Path(__file__).resolve().parents[1]/'scripts/experiments/tts_acoustic_generation_cross_20260926.py'
spec=importlib.util.spec_from_file_location('cross',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_original_pcm_mfcc_parity():
    x=np.random.default_rng(13).integers(-5000,5000,19003,dtype=np.int16)
    windows,mfcc=m.frontend_windows(x.astype(np.float32)/32768)
    expected=python_speech_features.mfcc(x,16000)
    np.testing.assert_array_equal(mfcc,expected)
    n=min(len(x)//640-5,(len(expected)-20)//4+1)
    np.testing.assert_array_equal(windows,np.stack([expected[i*4:i*4+20].T for i in range(n)]).astype(np.float32))


def test_four_cell_closure_and_null_paths():
    q=np.array([3.,3.4,3.7,4.3]);e=m.effects(q)
    assert abs(e['G']+e['E']+e['I']-e['total'])<1e-14
    assert m.effects([3,3,4,4])=={'G':0,'E':1,'I':0,'total':1}
    assert m.effects([3,4,3,4])=={'G':1,'E':0,'I':0,'total':1}


def test_distance_lag_orientation_and_support():
    rng=np.random.default_rng(70);a=rng.normal(size=(100,16)).astype(np.float32);v=a.copy();delayed=np.zeros_like(a);delayed[5:]=a[:-5]
    raw=m.matrix(v,a);shift=m.matrix(v,delayed)
    assert m.summarize(raw,'guard20')['best_lag']==0
    assert m.summarize(shift,'guard20')['best_lag']==5
    np.testing.assert_allclose(raw[25:75,:-5],shift[25:75,5:],atol=0,rtol=0)
    assert m.summarize(raw[:40],'guard20') is None


def test_equal_speaker_bootstrap_and_covariance_identity():
    s=m.bootstrap([1,1,1,5],['A','A','A','B']);assert s['mean']==3
    rng=np.random.default_rng(82);v=rng.normal(size=(80,10)).astype(np.float32);a=rng.normal(size=(80,10)).astype(np.float32)
    d=m.covariance_features(v,a,80)
    np.testing.assert_allclose(d['sigmaM']**2,(d['sigmaA']**2+d['sigmaV']**2+2*d['covAV'])/4,atol=1e-13)
