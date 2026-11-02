"""Unequal-clock support and paired-seed aggregation controls."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location('tts_extension_pipeline',Path(__file__).parents[2]/'scripts/experiments/qwen_full_tts_extension_20261022/pipeline.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


def test_all_policies_only_use_real_lag_support_and_equal_count_is_unique():
    counts=dict(N=100,R=102,T1=80,T2=120,T3=70)
    p=M.support_policies(counts)
    assert all(len(q)==40 for q in p['common'].values())
    for policy,roles in p.items():
        for role,q in roles.items():
            assert np.all(q-15>=0) and np.all(q+15<counts[role])
            assert len(np.unique(q))==len(q)
    assert len(p['native_full']['T2'])==90
    assert p['equal_count_spread']['T2'][-1]==104


def test_no_support_does_not_drop_a_short_seed():
    with pytest.raises(ValueError,match='real all-lag'):
        M.support_policies(dict(N=100,R=100,T1=29,T2=100,T3=100))


def test_seed_average_is_within_speaker_before_inference():
    x=np.array([[1,2,0,3,6],[10,12,8,11,14]],float)
    c=M.paired_contrasts(x)
    assert np.array_equal(c['T-N'],[2,1])
    assert np.array_equal(c['T-R'],[1,-1])
    assert M.summary(c['T-N'])['n_speakers']==2


def test_C_best_is_not_D0_and_offset_sign_is_preserved():
    x=np.tile(np.arange(31,dtype=float),(70,1));x[:,12]=-1
    r=M.metrics(x,np.arange(15,55))
    assert r['offset']==3 and r['Dmin']==-1 and r['D0']==15
    assert r['C']==r['B']-r['Dmin']


def test_two_comparison_interval_is_wider_than_descriptive_interval():
    r=M.summary([-.4,0,.2,.4,.8,.9,1,1.1,1.2,2])
    assert r['CI97_5'][0]<=r['CI95'][0]<=r['CI95'][1]<=r['CI97_5'][1]
