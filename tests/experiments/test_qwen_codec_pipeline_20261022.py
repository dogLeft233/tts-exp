"""Controls for acoustic timing diagnostics, without model inference."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location('codec_pipeline',Path(__file__).parents[2]/'scripts/experiments/qwen_codec_roundtrip_20261022/pipeline.py')
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_known_delay_has_correct_sign_and_10ms_clock():
    rng=np.random.default_rng(6)
    x=rng.uniform(.1,1,300)
    y=np.r_[np.zeros(4),x[:-4]]
    r=M.lag_proxy(x,y)
    assert r['lag_s']==.04
    assert r['correlation']==pytest.approx(1)
    assert M.lag_proxy(y,x)['lag_s']==-.04


def test_scalar_gain_cannot_be_mistaken_for_envelope_delay():
    t=np.arange(32000)/16000
    x=np.sin(2*np.pi*200*t)*(1+.5*np.sin(2*np.pi*3*t))
    r=M.lag_proxy(M.envelope(x),M.envelope(x*3))
    assert r['lag_s']==0
    assert r['correlation']==pytest.approx(1)


def test_silence_is_unmeasurable_not_a_fabricated_zero_lag():
    assert M.lag_proxy(np.zeros(100),np.zeros(100))=={'measurable':False}


def test_tail_sensitive_statistic_does_not_hide_outlier_in_median():
    r=M.stats([0]*19+[.1])
    assert r['median_abs']==0 and r['max_abs']==.1 and r['mae']==.005


def test_cluster_bootstrap_resamples_speakers_not_frames():
    r=M.bootstrap([1.,1.,1.])
    assert r['n_speakers']==3 and r['CI95']==[1.,1.]
