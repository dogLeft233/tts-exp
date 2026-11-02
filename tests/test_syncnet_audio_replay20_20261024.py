"""Boundary arithmetic and padding-preserving replay helpers."""
import importlib.util
from pathlib import Path
import numpy as np
P=Path(__file__).parents[1]/'scripts/experiments/syncnet_audio_replay20_20261024.py'
spec=importlib.util.spec_from_file_location('replay20',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def test_mfcc_rf_and_actual_centers():
    assert m.row_support(0)['raw_samples_inclusive']==[0,3439]
    assert m.row_support(1)['raw_samples_inclusive']==[639,4079]
    assert m.row_support(1)['center_s']==2359/16000
    assert m.row_support(15)['mfcc_columns']==[60,80]
def test_patch_edges_overlap_and_sensitive_nearest():
    assert m.patch_index(4,20) is None and m.patch_index(15,20) is None
    assert m.patch_index(5,20)==list(range(11))
    assert len(set(m.patch_index(6,20))&set(m.patch_index(5,20)))==10
    t=m.row_support(8)['center_s'];assert m.nearest_row(t,20)==8
    assert m.nearest_row(t+.04,20)==9 and m.nearest_row(t-.04,20)==7
def test_curve_scalar_keeps_edge_padding():
    z=np.arange(9*31,dtype=np.float32).reshape(9,31)
    a=m.scalar(z);mu=z.mean(0)
    assert a['C']==float(np.median(mu)-mu.min()) and a['offset']==15
def test_clock_delay_not_applied_to_scores():
    rng=np.random.default_rng(5);x=rng.normal(size=8000);y=np.r_[np.zeros(1105),x,np.zeros(100)]
    rec=m.clock_lag(x,y);assert rec['lag_samples_scoring_minus_canonical']==1105 and rec['correlation']>.999999
