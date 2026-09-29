import numpy as np
from scripts.experiments.real_av_geometry_calibration_20260927 import angles,similarity,longest_false,geometry,input_qc

def test_uniform_pose_scale_and_translation_do_not_change_angles():
    t=np.deg2rad(15);m=np.eye(4);m[:3,:3]=2*np.array([[np.cos(t),0,np.sin(t)],[0,1,0],[-np.sin(t),0,np.cos(t)]]);m[:3,3]=[2,3,-9]
    np.testing.assert_allclose(angles(m),[15,0,0],atol=1e-12)

def test_similarity_uses_stable_points_and_preserves_lip_motion():
    rng=np.random.default_rng(42);r=rng.normal(size=(13,2));a=r*2+3;s,rot,tr=similarity(a[:7],r[:7]);np.testing.assert_allclose(s*a@rot+tr,r,atol=1e-12)
    a[7,1]+=1;s,rot,tr=similarity(a[:7],r[:7]);assert abs((s*a@rot+tr)[7,1]-r[7,1]-.5)<1e-12

def test_missing_runs_include_edges():
    assert longest_false([False,False,True,False,False,False])==3
    assert longest_false([True]*100)==0

def test_constant_width_does_not_block_aperture_gate():
    n=100;g={'valid':np.ones(n,bool),'reference_angles':np.zeros(3),'pose':np.zeros((n,3)),'residual':np.zeros(n),'aperture':.05+.02*np.sin(np.arange(n)),'width':np.ones(n)}
    assert input_qc(g)['passed']
