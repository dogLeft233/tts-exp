from pathlib import Path
import importlib.util
import numpy as np
import pytest

P=Path(__file__).resolve().parents[2]/'scripts/experiments/nrt_two_arm_current_pipeline_20261023/canonical_continuation.py'
spec=importlib.util.spec_from_file_location('canonical_continuation',P);C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)

def spike():
    x=np.full(654080,.01,np.float32);x[C.INDICES]=[-.8,-.9,-.85];return x

def test_only_explicit_three_indices_may_saturate():
    x=spike();y,rms,gain,exception=C.normalize(x,C.EXCEPTION_KEY,C.EXCEPTION_FLOAT_SHA)
    assert exception['indices_zero_based']==C.INDICES
    assert np.all(y[C.INDICES]==-1.) and len(y)==len(x)
    original=x*np.float32(gain);keep=np.ones(len(x),bool);keep[C.INDICES]=False
    np.testing.assert_array_equal(y[keep],original[keep])
    assert np.all(C.pcm_in_memory(y)[C.INDICES]==-32768)

def test_same_wave_other_cell_keeps_original_gate():
    with pytest.raises(ValueError,match='unapproved'):C.normalize(spike(),'other_T2',C.EXCEPTION_FLOAT_SHA)

def test_wrong_exception_source_and_extra_point_rejected():
    with pytest.raises(ValueError,match='source differs'):C.normalize(spike(),C.EXCEPTION_KEY,'wrong')
    x=spike();x[1]=-.9
    with pytest.raises(ValueError,match='exactly frozen three'):C.normalize(x,C.EXCEPTION_KEY,C.EXCEPTION_FLOAT_SHA)

def test_valid_cell_keeps_fixed_gain_and_sample_count():
    x=np.tile(np.array([-.1,.1],np.float32),8000);y,rms,gain,exc=C.normalize(x,'N','irrelevant')
    assert exc is None and len(y)==len(x)
    np.testing.assert_array_equal(y,x*np.float32(C.TARGET/rms))

def test_nonfinite_and_silence_still_rejected():
    for x in [np.zeros(10,np.float32),np.array([np.nan],np.float32)]:
        with pytest.raises(ValueError):C.normalize(x,'N','irrelevant')
