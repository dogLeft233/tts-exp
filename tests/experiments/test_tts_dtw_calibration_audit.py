from scripts.experiments.tts_dtw_calibration_audit import boot,quantiles

def test_clip_equal_distribution_not_point_equal():
    clips=[[{'error':0.}], [{'error':10.}]*9]
    q=quantiles(clips,'error')
    assert q['q25']==0 and q['median']==0 and q['q75']==10

def test_clip_bootstrap_constant():
    z=boot([3.,3.,3.])
    assert z['mean']==3 and z['ci99']==[3,3] and z['n']==3
