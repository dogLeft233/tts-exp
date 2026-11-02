import importlib.util
from pathlib import Path
import numpy as np


def load():
    path=Path(__file__).resolve().parents[1]/'scripts/experiments/native_warp_sync_parent_audit_20261025.py'
    spec=importlib.util.spec_from_file_location('native_warp_parent',path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_official_median_and_offset():
    m=load()
    a=np.broadcast_to(np.arange(31,dtype=np.float32),(17,31)).copy()
    s=m.official_scalar(a)
    assert s['B']==15 and s['D']==0 and s['C']==15 and s['offset']==15


def test_source_equal_and_exclude_repeat():
    m=load();rows=[]
    for i in range(6):
        for j,k in enumerate(m.ROLES):
            rows.append(dict(sample_id=str(i),role=k,C=i+j,D=5-j,B=i+5))
    rows.append(dict(sample_id='0',role='N_repeat',C=999,D=999,B=999))
    a=m.aggregate(rows)
    assert a['effects']['T']['C']['mean']==4
    assert a['effects']['pause']['C']['positive']==6
