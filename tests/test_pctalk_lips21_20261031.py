import importlib.util
from pathlib import Path
import numpy as np

PATH=Path(__file__).resolve().parents[1]/'scripts/experiments/pctalk_lips21_20261031/prepare.py'
spec=importlib.util.spec_from_file_location('lips_prepare',PATH);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_selection_one_existing_id_per_speaker_is_order_independent():
    rows=[dict(id='b_9',speaker='b',flags=[]),dict(id='a_2',speaker='a',flags=[]),dict(id='a_1',speaker='a',flags=['clip'])]
    assert [r['id'] for r in module.select(rows)]==['a_1','b_9']
    assert module.select(rows)==module.select(list(reversed(rows)))

def test_scalar_normalization_preserves_clock_and_guard():
    x=np.sin(np.arange(16000)*.07)*.2
    y,gain=module.normalize(x)
    assert y.shape==x.shape and y.dtype==np.dtype('<i2')
    assert np.max(np.abs(y/32768-x*gain))<=.5/32768
    assert abs(20*np.log10(np.sqrt(np.mean((y/32768)**2)))+24)<.002
    import pytest
    impulse=np.zeros(16000);impulse[500]=1
    with pytest.raises(AssertionError):module.normalize(impulse)
