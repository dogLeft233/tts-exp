"""Protect pair support and control qualification in the EARS adapters."""
import importlib.util
from pathlib import Path
import pytest


def load():
    source=Path(__file__).parents[1]/'scripts/experiments/ears40_extra_scorers_20261029.py'
    spec=importlib.util.spec_from_file_location('ears_extra',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def rows():
    return [dict(id=f'p{i:03d}_{arm}_0',sample_id=f'p{i:03d}',arm=arm,native_frames=250,
        eligible_5s=True,center_start_5s=62) for i in range(40) for arm in ('N','T')]


def test_support_is_validated_per_pair():
    m=load();assert len(m.validate_rows(rows()))==80
    bad=rows();bad[-1]['sample_id']='p000'
    with pytest.raises(ValueError,match='40 speakers'):m.validate_rows(bad)
    bad=rows();bad[0]['native_frames']=124
    with pytest.raises(ValueError,match='eligibility'):m.validate_rows(bad)


def test_eighty_controls_require_sixty_four_strict_decreases():
    m=load()
    assert m.control_summary([dict(lower=i<64) for i in range(80)])['status']=='PASS'
    assert m.control_summary([dict(lower=i<63) for i in range(80)])['status']=='CONTROL_FAILED'
    assert m.control_summary([])['status']=='UNMEASURABLE'
@pytest.mark.parametrize('start',[114,115,58,57,171])
@pytest.mark.parametrize('shift',[-6400,0,6400])
def test_center_selection_uses_exact_integer_clock_and_shift_before_crop(start,shift):
    import numpy as np
    from importlib.util import spec_from_file_location, module_from_spec
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/experiments/ears40_extra_scorers_20261029.py'
    spec=spec_from_file_location('ears_center',path);mod=module_from_spec(spec);spec.loader.exec_module(mod)
    video=np.arange(500);audio=np.arange(320000,dtype=np.int32)
    shifted=np.zeros_like(audio)
    if shift>0:shifted[shift:]=audio[:-shift]
    elif shift<0:shifted[:shift]=audio[-shift:]
    else:shifted[:]=audio
    item=dict(video=video,audio=shifted,meta={'video':{'fps':[25]},'audio':{'framerate':[16000]}},targets={'v_start_i_sec':start/25,'offset_sec':0.})
    out=mod.pretrim_official_center(item,start)
    assert np.array_equal(out['video'],video[start:start+125])
    assert np.array_equal(out['audio'],shifted[start*640:(start+125)*640])
    assert len(out['video'])==125 and len(out['audio'])==80000
    assert out['targets']['v_start_i_sec']==0 and item['targets']['v_start_i_sec']==start/25

