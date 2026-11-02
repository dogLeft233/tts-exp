"""Independent replay uses original epsilon and padded lag conventions."""
import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('parent',Path(__file__).parents[1]/'scripts/experiments/syncnet_fixed_pcm_eq_parent_audit_20261025.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_epsilon_padding_and_anchor_identity():
    a=np.ones((8,2),np.float32)
    d=m.distance(a,a)
    np.testing.assert_allclose(d[:,15],np.sqrt(2)*1e-6,rtol=1e-6)
    assert np.all(d[:,0]>1)
    s=m.scalar(d,15)
    assert abs(s['C']-s['C_anchor']-s['search_uplift'])<1e-12
