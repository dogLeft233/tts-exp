import numpy as np
from scripts.experiments.tts_instance_audio_dtw import dtw,cmvn,mapped_time

def test_identity_and_tie_diagonal():
    path,cost=dtw(np.zeros((4,2)),np.zeros((4,2)))
    assert path==[[i,i] for i in range(4)] and cost==0

def test_path_endpoints_and_steps():
    x=np.array([[0.],[1.],[2.]])
    y=np.array([[0.],[0.],[1.],[2.]])
    path,cost=dtw(x,y)
    assert path[0]==[0,0] and path[-1]==[2,3] and cost==0
    assert all(tuple(d) in ((1,0),(0,1),(1,1)) for d in np.diff(path,axis=0))

def test_cmvn_constant_finite():
    assert np.array_equal(cmvn(np.ones((5,3))),np.zeros((5,3)))

def test_bidirectional_common_path_mapping():
    p={'path':[[0,0],[1,1],[1,2],[2,3]]}
    assert mapped_time(p,'Q1Q2',1/80)[0]==1.5/80
    assert mapped_time(p,'Q2Q1',2/80)[0]==1/80
