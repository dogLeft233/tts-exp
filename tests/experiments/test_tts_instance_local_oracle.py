import numpy as np
from scripts.experiments.tts_instance_local_oracle import candidates

def test_radius_nesting_and_identity():
    node={'j':{'Q2':40},'span':{'Q2':[1.4,2.]}}
    pools=[set(candidates(node,'Q2',100,3,r,True)) for r in (0,1,2,4,None)]
    assert pools[0]=={40}
    assert all(a<=b for a,b in zip(pools,pools[1:]))

def test_guard_and_phone_constraints():
    node={'j':{'Q2':24},'span':{'Q2':[.9,1.3]}}
    pool=candidates(node,'Q2',60,3,4,True)
    assert np.all(pool>=23) and np.all(pool-3<40)
    assert np.all(pool/25+.1075<1.3)
    assert set(pool)<=set(candidates(node,'Q2',60,3,4,False))
