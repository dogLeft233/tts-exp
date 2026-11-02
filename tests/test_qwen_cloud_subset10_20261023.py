import importlib.util
from pathlib import Path
import numpy as np
import pytest

@pytest.fixture(params=['qwen_cloud_subset10_20261023.py','qwen_audio31_cloud_subset10_20261023.py'])
def m(request):
    source=Path(__file__).resolve().parents[1]/'scripts/experiments'/request.param
    sp=importlib.util.spec_from_file_location('cloud_pilot',source)
    module=importlib.util.module_from_spec(sp);sp.loader.exec_module(module)
    return module

def test_normalization_preserves_length_and_relative_waveform_without_time_editing(m):
    x=np.array([-.2,.1,.05,-.08],dtype=np.float32)
    y,r,g=m.canonical_values(x,.0376838172675746)
    assert len(y)==len(x)
    np.testing.assert_array_equal(y,x*np.float32(g))
    assert np.sqrt(np.mean(y.astype(float)**2))==pytest.approx(.0376838172675746,abs=1e-8)

def test_silent_nonfinite_and_clipping_outputs_stop_instead_of_changing_protocol(m):
    for x in [np.zeros(10),np.array([np.nan]),np.array([1]+[0]*9999)]:
        with pytest.raises(ValueError):m.canonical_values(x,.0376838172675746)

def test_temporary_signed_audio_url_is_not_persisted(m):
    assert m.safe_url('https://sample.example/a.wav?Signature=private#fragment')=='https://sample.example/a.wav'
