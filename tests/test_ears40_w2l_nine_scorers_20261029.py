import importlib.util
from pathlib import Path
import numpy as np

path=Path(__file__).resolve().parents[1]/'scripts/experiments/ears40_w2l_nine_scorers_20261029.py'
spec=importlib.util.spec_from_file_location('ears40_main',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_rms_preserves_clock_and_rejects_clipping():
    x=.1*np.sin(np.arange(16000)*.2)
    y,_=module.normalize(x)
    assert len(y)==len(x)
    assert np.isclose(np.sqrt(np.mean(y*y)),module.RMS)
    assert np.argmax(x)==np.argmax(y)
    impulse=np.zeros(16000);impulse[10]=1
    import pytest
    with pytest.raises(AssertionError,match='clip'):module.normalize(impulse)


def test_pcm_shift_has_no_wrap_and_same_length():
    x=np.arange(16000,dtype=np.int16)
    late=module.shift_pcm(x,6400);early=module.shift_pcm(x,-6400)
    assert len(late)==len(early)==len(x)
    assert np.all(late[:6400]==0) and np.array_equal(late[6400:],x[:-6400])
    assert np.all(early[-6400:]==0) and np.array_equal(early[:-6400],x[6400:])
