"""Train-only diagnostic, never a data-admission or generator-training path.

Freeze first 4800 train frames for landmark geometry calibration and last 1200
for camera recovery. Keep BA and original files immutable. Static landmark
offsets test whether BA's changed shape explains non-similarity camera errors.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from geometry import fit_training_gauge,map_poses_to_training_gauge,project

def run(root,identity):
    output=root/'diagnostics'/identity/'training_geometry_attempt02'
    output.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).read_bytes();(output/Path(__file__).name).write_bytes(source)
    folder=root/'prepared'/identity/'train'
    sys.path.insert(0,str(root/'official/data_utils/face_tracking'))
    from util import euler2rot,forward_transform
    from facemodel import Face_3DMM
    track=torch.load(folder/'track_params.pt',weights_only=False)
    bundle=torch.load(folder/'bundle_adjustment.pt',weights_only=False)
    assert len(track['euler'])==len(bundle['euler'])==6000
    r0=euler2rot(track['euler']).numpy().astype(np.float64)
    r1=euler2rot(bundle['euler']).numpy().astype(np.float64)
    t0=track['trans'].numpy().astype(np.float64)/1000
    t1=bundle['trans'].numpy().astype(np.float64)
    gauge=fit_training_gauge(r0[:4800],t0[:4800],r1[:4800],t1[:4800])
    model=Face_3DMM(str(root/'official/data_utils/face_tracking/3DMM'),100,79,100,34650)
    indices=torch.as_tensor(np.loadtxt(root/'official/data_utils/face_tracking/3DMM/lands_info.txt',dtype=np.int32)).cuda()
    with torch.no_grad():
        geo=model.forward_geo_sub(track['id'].cuda().expand(6000,-1),track['exp'].cuda(),indices[-51:].long()).cpu().numpy().astype(np.float64)/1000
    q=np.array(gauge['rotation']);shift=np.array(gauge['translation'])
    geo=gauge['scale']*np.einsum('ij,nkj->nki',q,geo)+shift
    lms=np.stack([np.loadtxt(folder/'ori_imgs'/f'{i}.lms')[-51:] for i in range(6000)])
    focal=float(track['focal'][0])
    u=(lms[:4800,:,0]-256)/focal;v=(lms[:4800,:,1]-256)/focal
    rot=r1[:4800];trans=t1[:4800]
    a=np.stack([rot[:,None,0,:]+u[:,:,None]*rot[:,None,2,:],rot[:,None,1,:]-v[:,:,None]*rot[:,None,2,:]],axis=2)
    b=np.stack([-trans[:,None,0]-u*trans[:,None,2],-trans[:,None,1]+v*trans[:,None,2]],axis=2)
    residual=b-np.einsum('nkij,nkj->nki',a,geo[:4800])
    aa=np.einsum('nkij,nkil->kjl',a,a);ab=np.einsum('nkij,nki->kj',a,residual)
    delta=np.linalg.solve(aa,ab[:,:,None])[:,:,0]
    adjusted=geo+delta[None]
    np.savez(output/'training_calibration.npz',landmark_offsets=delta,gauge_rotation=q,gauge_translation=shift,gauge_scale=gauge['scale'])
    result=dict(identity=identity,calibration_frames=[0,4800],diagnostic_frames=[4800,6000],
        source_sha256=hashlib.sha256(source).hexdigest(),training_admitted=False,
        threshold_changes=False,method='train-only static landmark correction with frozen BFM expression deformation',
        landmark_offset_m_median=float(np.median(np.linalg.norm(delta,axis=1))),
        landmark_offset_m_max=float(np.linalg.norm(delta,axis=1).max()),
        calibration_geometry_condition_max=float(np.linalg.cond(aa).max()),gauge=gauge)
    result['landmark_rms_at_official_BA_px']={name:float(np.sqrt(np.mean((project(adjusted[s],r1[s],t1[s],focal)-lms[s])**2))) for name,s in [('fit',slice(0,4800)),('heldout_train',slice(4800,6000))]}
    # Two declared diagnostics, neither modifies production poses or thresholds.
    configurations={'all_51':np.arange(51),'rigid_13':np.array([*range(10,19),19,22,25,28])}
    result['camera_recovery']={}
    from scipy.spatial.transform import Rotation
    for name,selected in configurations.items():
        initial_r,initial_t=map_poses_to_training_gauge(r0[4800:],t0[4800:],gauge)
        # Official Euler is Rx(theta) Ry(phi) Rz(-psi).
        angles=Rotation.from_matrix(initial_r).as_euler('XYZ')
        angles[:,2]*=-1
        euler=torch.tensor(angles,dtype=torch.float32,device='cuda',requires_grad=True)
        assert np.max(np.abs(euler2rot(euler.detach()).cpu().numpy()-initial_r))<1e-6
        trans=torch.tensor(initial_t,dtype=torch.float32,device='cuda',requires_grad=True)
        geometry=torch.tensor(adjusted[4800:,selected],dtype=torch.float32,device='cuda')
        target=torch.tensor(lms[4800:,selected],dtype=torch.float32,device='cuda')
        optimizer=torch.optim.Adam([{'params':[euler],'lr':.003},{'params':[trans],'lr':.0003}])
        cxy=torch.tensor([256.,256.],device='cuda');f=torch.tensor([focal],device='cuda')
        initial_rotation_error=float(np.quantile(np.degrees(Rotation.from_matrix(initial_r@np.swapaxes(r1[4800:],1,2)).magnitude()),.95))
        for iteration in range(2000):
            projection=forward_transform(geometry,euler,trans,f,cxy)[:,:,:2]
            loss=torch.mean((projection-target)**2)
            optimizer.zero_grad();loss.backward();optimizer.step()
        recovered=euler2rot(euler.detach()).cpu().numpy().astype(np.float64)
        recovered_t=trans.detach().cpu().numpy().astype(np.float64)
        errors=np.degrees(Rotation.from_matrix(recovered@np.swapaxes(r1[4800:],1,2)).magnitude())
        translations=np.linalg.norm(recovered_t-t1[4800:],axis=1)
        result['camera_recovery'][name]=dict(initial_rotation_p95_degrees=initial_rotation_error,
            rotation_p95_degrees=float(np.quantile(errors,.95)),translation_p95_m=float(np.quantile(translations,.95)),
            landmark_rms_px=float(torch.sqrt(torch.mean((forward_transform(geometry,euler,trans,f,cxy)[:,:,:2]-target)**2)).detach()),
            passes_existing_camera_limits=bool(np.quantile(errors,.95)<=2 and np.quantile(translations,.95)<=.005))
        np.savez(output/(name+'_heldout_train_camera.npz'),rotation=recovered,translation=recovered_t)
        print(json.dumps(result['camera_recovery'][name]),flush=True)
    result['completed']=True
    (output/'diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True)
    a=p.parse_args();run(a.root,a.identity)
