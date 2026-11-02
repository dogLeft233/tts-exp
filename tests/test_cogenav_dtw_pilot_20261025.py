import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('pilot',Path(__file__).parents[1]/'scripts/experiments/cogenav_dtw_pilot_20261025.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def test_oracle_discontinuous_pause():
    segments=np.asarray([[0,16000,0,20000],[16000,32000,25120,37920]])
    times=np.asarray([0,.5,.99,1.,1.5])
    truth=p.oracle(times,segments)
    assert np.allclose(truth,[0,.625,1.2375,1.57,1.97])
    assert not np.any((truth>1.25)&(truth<1.57))


def test_identity_dtw():
    x=np.eye(10)
    path=p.dtw(p.distance(x,x))
    assert np.array_equal(path,np.column_stack([np.arange(10)]*2))
    assert np.allclose(p.mapping(path,10,np.arange(10)*.04),np.arange(10)*.04)


def test_monotone_and_selection():
    x=np.eye(8);y=np.repeat(x,2,axis=0)
    path=p.dtw(p.distance(x,y))
    assert np.array_equal(path[0],[0,0]) and np.array_equal(path[-1],[7,15])
    delta=np.diff(path,axis=0)
    assert np.all(delta>=0) and np.all(delta<=1) and np.all(delta.sum(axis=1)>=1)
    rows=[dict(sample_id='a',id='N',arm='N',draw=0,qualified=True,score_float32=1.),
          dict(sample_id='a',id='T',arm='L06',draw=1,qualified=True,score_float32=2.)]
    assert [r['id'] for r in p.select(rows)]==['N','T']


if __name__=='__main__':
    test_oracle_discontinuous_pause();test_identity_dtw();test_monotone_and_selection()
    print('3 contract functions PASS (direct execution, not pytest)')
