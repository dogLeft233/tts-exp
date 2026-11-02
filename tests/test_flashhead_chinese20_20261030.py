import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]/'scripts/experiments/flashhead_chinese20_20261030'
def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_normalization_preserves_timing_and_zeros_and_equalizes_rms():
    m=load('prepare');x=np.r_[np.zeros(400),.03*np.sin(np.arange(16000)*.12),np.zeros(800)]
    first,gain=m.normalized_pcm(x);second,other=m.normalized_pcm(x*.17)
    assert first.shape==x.shape and np.array_equal(first,second)
    assert not first[:400].any() and not first[-800:].any()
    assert abs(20*np.log10(np.sqrt(np.mean((first.astype(float)/32768)**2)))+24)<.002
    assert gain>0 and other>gain

def test_unsafe_target_rejected_instead_of_clipping():
    x=np.zeros(16000);x[10]=.1
    with pytest.raises(ValueError):load('prepare').normalized_pcm(x)

def test_pairing_keeps_draws_within_source_and_requires_all_three():
    m=load('analyze');rows=[dict(id=f's_{arm}{d}',source='s',speaker='speaker',arm=arm,strict_reference=True) for arm,d in [('N',0),('C',1),('C',2),('C',3)]]
    values={'s_N0':1.,'s_C1':2.,'s_C2':3.,'s_C3':4.}
    pairs,missing=m.make_pairs(rows,values)
    assert not missing and len(pairs)==1 and pairs[0]['T']==3 and pairs[0]['delta']==2
    values.pop('s_C3');pairs,missing=m.make_pairs(rows,values)
    assert not pairs and missing[0]['unmeasurable_ids']==['s_C3']

def test_paired_statistics_and_holm():
    m=load('analyze');s=m.statistics([1,2,3],[2,3,4])
    assert s['delta']==1 and s['ci95']==[1,1] and s['n_sources']==3
    assert m.holm([.01,.04,.03])==pytest.approx([.03,.06,.06])
