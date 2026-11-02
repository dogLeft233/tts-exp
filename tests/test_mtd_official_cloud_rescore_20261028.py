import importlib.util
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experiments'))
import mtd_official_cloud_rescore_20261028 as m


def test_official_k_context_changes_averaging_not_offset_tolerance():
    import numpy as np
    z=np.zeros((40,31),dtype=np.float32);z[:,18]=10
    assert all(v==0 for v in m.independent_reduction(z).values())
    z[:,18]=0;z[:,15]=10
    assert all(v==1 for v in m.independent_reduction(z).values())


def test_author_reduction_matches_independent_on_real_matrix():
    import numpy as np
    import torch
    path=Path('/home/wjj/tts-audio/tts-exp/runs/mtd_official_audit_20261028_attempt01/BAC009S0002W0122_N_0_0_512_matrix.npy')
    if not path.exists():return
    matrix=np.load(path);n=len(matrix);env=m.official_env()
    captured={}
    env['calc_pdist']=lambda *args,**kwargs:list(torch.from_numpy(matrix).unbind())
    class Dummy:
        def eval(self):return self
    def profile(frame,event,arg):
        if event=='return' and frame.f_code is env['eval_model'].__code__:
            captured.update({'K'+str(k):float(frame.f_locals['samplewise_acc_k'+str(k)][0]) for k in m.KS})
    sys.setprofile(profile)
    try:env['eval_model']([(torch.zeros(1,(n+5)*3,48,96),torch.zeros(1,1,80,int(80*((n+5)/25))+16),torch.tensor([n]))],torch.device('cpu'),Dummy())
    finally:sys.setprofile(None)
    assert captured==m.independent_reduction(matrix)
