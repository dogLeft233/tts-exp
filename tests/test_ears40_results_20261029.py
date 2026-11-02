import importlib.util
from pathlib import Path

import numpy as np

path=Path(__file__).resolve().parents[1]/'scripts/experiments/ears40_results_20261029.py'
spec=importlib.util.spec_from_file_location('ears_results',path)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_holm_adjustment_preserves_original_order_and_stepdown_monotonicity():
    np.testing.assert_allclose(mod.holm([.04,.001,.03,.5]),[.09,.004,.09,.5])


def test_missing_arm_fails_instead_of_shrinking_denominator():
    import pytest
    with pytest.raises(ValueError,match='Incomplete'):
        mod.metric_pairs([dict(source='p1',arm='N',score=1)],'score',['p1'])


def test_identical_pairs_have_all_ties_and_no_advantage():
    s=mod.statistics([1,2,3],[1,2,3],bootstraps=20,permutations=20)
    assert s['mean_difference']==0 and s['ci95']==[0,0]
    assert s['signflip_p']==1 and s['ties']==3 and s['paired_dz'] is None


def test_retrieval_zero_tolerance_and_temporal_smoothing():
    z=np.zeros((3,31),dtype=np.float32)
    z[0,14]=1;z[1,16]=1;z[2,17]=1
    assert mod.k_accuracy(z)==2/3
    assert mod.k_accuracy(z,7)==1
