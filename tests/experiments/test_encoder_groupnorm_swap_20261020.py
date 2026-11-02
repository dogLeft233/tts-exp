import importlib.util
from pathlib import Path
import numpy as np

def load():
    path=Path(__file__).parents[2]/'scripts/experiments/encoder_groupnorm_swap_20261020/geometry.py'
    spec=importlib.util.spec_from_file_location('swap_geometry',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_rotation_changes_raw_directions_but_preserves_cka():
    m=load();x=np.random.default_rng(42).normal(size=(100,8));q,_=np.linalg.qr(np.random.default_rng(43).normal(size=(8,8)))
    result=m.directions(x,x@q)
    assert result['mean_cosine']<.8
    assert abs(result['unit_direction_CKA']-1)<1e-12

def test_direction_response_ignores_pure_positive_scaling():
    m=load();x=np.random.default_rng(44).normal(size=(100,8));result=m.changes(x,x*.5,x,x*2)
    assert result['first_change_rms']<1e-14 and result['second_change_rms']<1e-14
    assert result['valid_frame_count']==0
