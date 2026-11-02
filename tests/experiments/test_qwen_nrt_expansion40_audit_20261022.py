import importlib.util
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[2]/'scripts/experiments/qwen_nrt_expansion40_20261022/independent_downstream_audit.py'
spec=importlib.util.spec_from_file_location('independent40audit',path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_unicode_cer_uses_character_insert_delete_substitute():
    assert audit.edit_distance('你好世界','你好世间')==1
    assert audit.edit_distance('库存10','库存十')==2
    assert audit.edit_distance('','测试')==2
    assert audit.edit_distance('自然音频','自然频')==1


def test_metric_offset_convention_and_included_zero_lag():
    matrix=np.full((4,31),7.,dtype=np.float64);matrix[:,17]=2.;matrix[:,15]=5.
    result=audit.summarize_matrix(matrix,np.arange(4))
    assert result['C']==5. and result['B']==7. and result['Dmin']==2.
    assert result['D0']==5. and result['offset']==-2 and result['query_count']==4
    assert result['best_lag_at_boundary'] is False


def test_speaker_bootstrap_has_fixed_units_and_both_intervals():
    a=audit.bootstrap([2.,2.,2.]);b=audit.bootstrap([2.,2.,2.])
    assert a==b and a['n_speakers']==3 and a['CI95']==a['CI97_5']==[2.,2.]


def test_array_shape_and_nonfinite_are_rejected():
    with pytest.raises(AssertionError):audit.close([1,2],[1],'different shape')
    with pytest.raises(AssertionError):audit.close([np.nan],[np.nan],'nonfinite')


def test_distance_matrix_matches_scalar_pairwise_formula():
    rng=np.random.default_rng(1);a=rng.normal(size=(35,1024)).astype(np.float32);v=rng.normal(size=(35,1024)).astype(np.float32)
    q=np.arange(15,20);j=q[:,None]+np.arange(-15,16)[None,:]
    delta=v[q,None,:]-a[j]+np.float32(1e-6)
    distances=np.sqrt(np.add.reduce(delta*delta,axis=2,dtype=np.float32),dtype=np.float32)
    assert j.min()==0 and j.max()==34
    for i,query in enumerate(q):
        for k,lag in enumerate(range(-15,16)):
            z=v[query]-a[query+lag]+np.float32(1e-6)
            assert abs(float(distances[i,k])-float(np.sqrt(np.sum(z*z,dtype=np.float32))))<1e-5
