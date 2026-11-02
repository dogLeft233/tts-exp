import importlib.util
from pathlib import Path
import numpy as np


def test_cogen_identity_and_shift():
    path=Path(__file__).parents[1]/'scripts/experiments/cogenav_fixed_video_cross_20261026.py'
    spec=importlib.util.spec_from_file_location('cross',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    a=np.random.default_rng(1).normal(size=(100,32))
    assert abs(m.cogen(a,a)['eq1']-1)<1e-12
    v=np.roll(a,-4,axis=0)
    assert m.cogen(a,v)['lag']==4


def test_source_weighting():
    path=Path(__file__).parents[1]/'scripts/experiments/cogenav_fixed_video_cross_20261026.py'
    spec=importlib.util.spec_from_file_location('cross',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    rows=[]
    for video in ['N','T']:
        for source,delta,count in [('a',1,3),('b',-1,1)]:
            for _ in range(count):
                row=dict(source=source,video=video)
                for metric in ['sync_C','co_eq1','co_max']:
                    row[metric+'_N']=2;row[metric+'_T']=2+delta
                rows.append(row)
    assert m.summarize(rows)['N']['sync_C']['delta']==0
