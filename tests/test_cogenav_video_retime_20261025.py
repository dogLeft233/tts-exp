from importlib.util import spec_from_file_location,module_from_spec
from pathlib import Path
import numpy as np
s=spec_from_file_location('cvr',Path(__file__).resolve().parents[1]/'scripts/experiments/cogenav_video_retime_20261025.py');m=module_from_spec(s);s.loader.exec_module(m)
def test_dtw_recovers_known_pause_and_identity():
 a=np.eye(8);cost=m.cosine_cost(a,a);path=m.dtw(cost);pred,ix=m.project(path,8,8)
 assert np.array_equal(path,np.column_stack([np.arange(8)]*2)) and np.array_equal(ix,np.arange(8))
 b=np.concatenate([a[:4],np.repeat(a[3:4],2,axis=0),a[4:]])
 p=m.dtw(m.cosine_cost(a,b));pred,ix=m.project(p,8,10)
 assert np.array_equal(ix[:3],[0,1,2]) and np.array_equal(ix[4:],[6,7,8,9]) and np.all(np.diff(ix)>=0)
def test_projection_direction_rounding_and_relu():
 p=np.asarray([[0,0],[0,1],[1,1],[2,2],[2,3],[3,4]])
 pred,ix=m.project(p,4,5)
 assert np.array_equal(pred,[0,1,2.5,4]) and np.array_equal(ix,[0,1,3,4])
 assert np.array_equal(m.nearest([-1,.49,.5,9],5),[0,0,1,4])
 assert m.cosine_cost(np.asarray([[1.,0]]),np.asarray([[-1.,0]]),True)[0,0]==1
 assert m.cosine_cost(np.asarray([[1.,0]]),np.asarray([[-1.,0]]),False)[0,0]==2
