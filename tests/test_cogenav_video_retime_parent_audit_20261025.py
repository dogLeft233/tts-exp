import importlib.util
from pathlib import Path
import numpy as np


def load():
    path=Path(__file__).resolve().parents[1]/'scripts/experiments/cogenav_video_retime_parent_audit_20261025.py'
    spec=importlib.util.spec_from_file_location('retime_parent',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_path_direction_and_median_endpoint_projection():
    m=load()
    path=np.array([[0,0],[0,1],[1,2],[2,3],[3,4],[3,5]])
    m.validate_path(path,4,6)
    values,indices=m.project(path,4,6)
    assert values.tolist()==[0,2,3,5] and indices.tolist()==[0,2,3,5]
    try:
        m.validate_path(path[:,::-1],4,6)
    except AssertionError:
        pass
    else:
        raise AssertionError('reversed N/T path accepted')


def test_independent_optimum_and_official_scalar():
    m=load()
    cost=np.ones((7,7));np.fill_diagonal(cost,0)
    assert m.solve_cost(cost)==0
    distance=np.broadcast_to(np.arange(31,dtype=np.float32),(13,31)).copy()
    score=m.official_scalar(distance)
    assert (score['C'],score['D'],score['B'],score['offset'])==(15,0,15,15)


def test_codec_gain_cannot_be_reported_as_retiming_gain():
    m=load();rows=[]
    for i in range(6):
        rows.extend([
            dict(sample_id=str(i),role='N_codec',C=i+2.,D=4.,B=i+6.,original_N_C=i+1.),
            dict(sample_id=str(i),role='aligned',C=i+1.5,D=4.5,B=i+6.,original_N_C=i+1.)])
    result=m.aggregate(rows,['N_codec','aligned'])['effects']['aligned']
    assert result['delta_C']==-.5 and result['positive']==0
    assert result['delta_original_C']==.5 and result['positive_original']==6
