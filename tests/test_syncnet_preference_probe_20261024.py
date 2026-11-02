"""Tests for pair design, grouped reward metrics and antisymmetric heads."""
import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'scripts/experiments/syncnet_preference_probe_20261024.py'
spec=importlib.util.spec_from_file_location('probe',P);probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
def rows():
    return [dict(id=f's_{a}_{d}',sample_id='s',speaker_id='speaker',group_id='speaker',cohort='old',arm=a,draw=d,qualified=not(a=='L06' and d==3),score_float32=float(i)) for i,(a,d) in enumerate([('N',0)]+[(a,d) for a in ['L06','L17'] for d in [1,2,3]])]
def test_pair_protocol_and_common_draw_sensitivity():
    r=rows();full=probe.make_pairs(r);strict=probe.make_pairs(r,True)
    assert len(full)==15 and len(strict)==8
    assert sum(x['family'] in probe.PRIMARY for x in full)==9
    assert all(x['a_draw']!=3 and x['b_draw']!=3 for x in strict)
    assert len({p['a_id'] for p in full if p['a_id'].startswith('s_N')})==1
def test_hand_computed_macro_and_zero_prediction():
    p=[dict(source='s',family=f,target=2.) for f in probe.PRIMARY]
    s=probe.source_metrics(p,[1.,2.,3.]);assert s['mae']==2/3 and s['accuracy']==1
    assert probe.source_metrics(p,[0,0,0])['accuracy']==.5
def test_reward_antisymmetry_and_input_schema():
    p=probe.make_pairs(rows());z=np.arange(21.).reshape(7,3)
    model=probe.fit_model(p,z,10);a=probe.predict(p,z,model)
    swapped=[dict(x,a=x['b'],b=x['a']) for x in p]
    assert np.allclose(a,-probe.predict(swapped,z,model))
    assert not set(probe.ACOUSTIC)&{'score_float32','score_d','score_b','av_offset'}
