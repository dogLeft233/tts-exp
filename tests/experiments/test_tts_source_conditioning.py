import numpy as np
from scripts.experiments.tts_source_conditioning import source_indices,score_roi,matrix,summarize_matrix

def test_source_phase_and_static_indices():
    n=280
    assert np.array_equal(source_indices(n,'D0'),np.tile(np.arange(132),3)[:n])
    assert np.array_equal(source_indices(n,'D66'),(np.arange(n)+66)%132)
    assert set(source_indices(n,'S0'))=={0}
    assert set(source_indices(n,'S66'))=={66}

def test_roi_is_one_fixed_union_square_with_padding():
    assert score_roi([[10,20,30,60],[20,10,50,40]])==[-8,-3,67,72]

def test_positive_delay_moves_best_lag_positive_on_common_support():
    rng=np.random.default_rng(1);v=rng.normal(size=(100,12)).astype(np.float32);a=v.copy();delayed=np.zeros_like(a);delayed[5:]=a[:-5]
    base=matrix(v,a);shifted=matrix(v,delayed)
    assert summarize_matrix(base,'guard20',0)['best_lag']==0
    z=summarize_matrix(shifted,'guard20',0)
    assert z['best_lag']==5 and z['offset']==-5
    np.testing.assert_allclose(shifted[25:-25,5:],base[25:-25,:-5],atol=0,rtol=0)

def test_valid_pair_policy_ignores_padding_and_common_guard_does_too():
    m=np.ones((50,31))
    for lag in range(-15,16):
        for i in range(50):
            if not 0<=i+lag<50:m[i,lag+15]=100
    assert summarize_matrix(m,'valid',0)['C']==0
    assert summarize_matrix(m,'guard20',0)['C']==0
    assert summarize_matrix(m,'guard0',0)['C']>0
