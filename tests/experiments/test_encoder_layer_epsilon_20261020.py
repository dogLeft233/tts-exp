"""Numerical checks for the diagnostic's measured norms and epsilon algebra."""
import importlib.util
from pathlib import Path
import numpy as np

def module():
    path=Path(__file__).parents[2]/'scripts/experiments/encoder_layer_epsilon_20261020/run.py'
    spec=importlib.util.spec_from_file_location('epsilon_diag',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value

def test_metrics_detect_direction_change_at_constant_norm():
    m=module();x=np.array([[1.,0.],[0.,1.]],dtype=np.float32);y=x[:,::-1].copy()
    stats=m.summarize(m.moments(x,y))
    assert np.isclose(stats['relative_l2'],np.sqrt(2))
    assert np.isclose(stats['mean_angle_degrees'],90)
    assert stats['mean_norm_ratio']==1

def test_epsilon_compensation_with_affine():
    rng=np.random.default_rng(42);z=rng.normal(size=(100,4));g=rng.normal(size=4);b=rng.normal(size=4)
    def norm(z,eps):return (z-z.mean(0))/np.sqrt(z.var(0)+eps)*g+b
    for scale in (.5,2.):
        assert np.array_equal(norm(z,1e-5),norm(z*scale,1e-5*scale**2))
