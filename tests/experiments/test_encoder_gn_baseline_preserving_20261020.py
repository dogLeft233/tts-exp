import importlib.util
from pathlib import Path
import numpy as np

def load():
    path=Path(__file__).parents[2]/'scripts/experiments/encoder_gn_baseline_preserving_20261020/metrics.py'
    spec=importlib.util.spec_from_file_location('anchor_metrics',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_anchor_exactly_preserves_arbitrary_original_baseline():
    m=load();r=np.random.default_rng(1);base=r.normal(size=(100,8)).astype(np.float32);other=(r.normal(size=(100,8))*100).astype(np.float32)
    assert np.array_equal(m.anchor(base,other,other),base)

def test_native_anchor_roundtrip_and_dose():
    m=load();r=np.random.default_rng(2);base=r.normal(size=(100,8)).astype(np.float32);changed=base+r.normal(scale=.001,size=base.shape).astype(np.float32)
    assert np.array_equal(m.anchor(base,changed,base),changed)
    target=m.matched(base,changed.astype(float)-base,1e-3)
    assert abs(m.norm(target.astype(float)-base)/m.norm(base)-1e-3)<1e-8

def test_direction_metric_ignores_scalar_and_has_exact_zero():
    m=load();x=np.random.default_rng(3).normal(size=(100,8)).astype(np.float32)
    assert m.summarize(m.moments(x,x))['unit_change_rms']==0
    assert m.summarize(m.moments(x,x*.5))['unit_change_rms']==0
