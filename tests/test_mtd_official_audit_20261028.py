import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('mtd_audit',Path(__file__).parents[1]/'scripts/experiments/mtd_official_audit_20261028.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_retrieval_does_not_reward_shifted_pair_as_in_sync():
    import numpy as np
    z=np.zeros((80,31));z[:,25]=10
    assert m.retrieval_metrics(z)['k5_accuracy']==0
    assert m.retrieval_metrics(z,expected_offset=-10)['k5_accuracy']==1
    assert m.retrieval_metrics(z)['global_offset']==-10


def test_interior_rejects_boundary_only_peak():
    import numpy as np
    z=np.zeros((80,31));z[:,15]=1;z[:20,0]=9;z[-20:,30]=9
    assert m.retrieval_metrics(z,guard=20)['k5_accuracy']==1
    assert m.retrieval_metrics(z)['k5_accuracy']==.5


def test_offset_change_control_handles_native_nonzero_and_rejects_constant():
    # Actual audit symptom: native -3; delayed audio -13, advanced audio +7.
    native=dict(global_offset=-3)
    assert m.shift_tracking(native,dict(global_offset=-13),6400)['passed']
    assert m.shift_tracking(native,dict(global_offset=7),-6400)['passed']
    assert not m.shift_tracking(native,native,6400)['passed']
