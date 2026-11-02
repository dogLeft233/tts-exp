"""Qualify training-owned geometry and bind heldout cameras to its coordinate frame."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from geometry import fit_training_gauge,map_poses_to_training_gauge,project

def triangulate(rot,trans,tracks,focal):
    xy=np.asarray(tracks,dtype=np.float64).copy();xy[:,:,0]=-(xy[:,:,0]-256)/focal;xy[:,:,1]=(xy[:,:,1]-256)/focal
    a=rot[:,None,:2,:]-xy[:,:,:,None]*rot[:,None,2:3,:]
    b=xy*trans[:,None,2:3]-trans[:,None,:2]
    aa=np.einsum('nkij,nkil->kjl',a,a);ab=np.einsum('nkij,nki->kj',a,b)
    return np.linalg.solve(aa,ab[:,:,None])[:,:,0]

def run(root,identity):
    folder=root/'prepared'/identity
    sys.path.insert(0,str(root/'official/data_utils/face_tracking'))
    from util import euler2rot,forward_transform
    from facemodel import Face_3DMM
    contract=json.loads((root/'preprocessing_contract_v1.json').read_text());limits=contract['geometry_gates']
    training=torch.load(folder/'train/track_params.pt',weights_only=False)
    train_bundle=torch.load(folder/'train/bundle_adjustment.pt',weights_only=False)
    r0=euler2rot(training['euler']).numpy().astype(np.float64);t0=training['trans'].numpy().astype(np.float64)/1000
    r1=euler2rot(train_bundle['euler']).numpy().astype(np.float64);t1=train_bundle['trans'].numpy().astype(np.float64)
    gauge=fit_training_gauge(r0,t0,r1,t1)
    gauge_checks=dict(rotation=gauge['rotation_error_degrees_p95']<=limits['gauge_rotation_error_degrees_p95_max'],
        translation=gauge['camera_translation_error_m_p95']<=limits['gauge_camera_translation_error_m_p95_max'],
        conditioning=gauge['condition']<=limits['gauge_condition_max'],
        scale=limits['gauge_scale_min']<=gauge['scale']<=limits['gauge_scale_max'])
    gauge['checks']=gauge_checks;gauge['source']='training camera pairs only'
    (folder/'train_fitted_gauge.json').write_text(json.dumps(gauge,indent=2))
    model=Face_3DMM(str(root/'official/data_utils/face_tracking/3DMM'),100,79,100,34650)
    lands=torch.as_tensor(np.loadtxt(root/'official/data_utils/face_tracking/3DMM/lands_info.txt',dtype=np.int32)).cuda()
    rows=[]
    for split,count in [('train',6000),('val',750),('test',1500)]:
        data=folder/split;track=torch.load(data/'track_params.pt',weights_only=False)
        bundle=torch.load(data/'bundle_adjustment.pt',weights_only=False)
        assert len(track['euler'])==count and len(bundle['euler'])==count
        assert torch.equal(track['id'],training['id']) and torch.equal(track['focal'],training['focal'])
        assert torch.equal(bundle['focal'],training['focal'])
        lms=np.stack([np.loadtxt(data/'ori_imgs'/f'{i}.lms')[-51:,:] for i in range(count)])
        with torch.no_grad():
            geo=model.forward_geo_sub(track['id'].cuda().expand(count,-1),track['exp'].cuda(),lands[-51:].long())
            projection=forward_transform(geo,track['euler'].cuda(),track['trans'].cuda(),track['focal'].cuda(),torch.tensor([256.,256.],device='cuda'))[:,:,:2].cpu().numpy()
        landmark_errors=np.sqrt(np.mean((projection-lms)**2,axis=(1,2)))
        rot=euler2rot(bundle['euler']).numpy().astype(np.float64);trans=bundle['trans'].numpy().astype(np.float64)
        tracks=np.load(data/'track_xys.npy').astype(np.float64);focal=float(track['focal'][0])
        if split=='train':points=triangulate(rot,trans,tracks,focal)
        else:points=bundle['fixed_points'].numpy().astype(np.float64)
        reconstructed=project(np.broadcast_to(points,(count,)+points.shape),rot,trans,focal)
        flow_errors=np.sqrt(np.mean((reconstructed-tracks)**2,axis=(1,2)))
        invariance=0.
        if split!='train':
            original_r,original_t=rot.copy(),trans.copy();rot,trans=map_poses_to_training_gauge(rot,trans,gauge)
            selection=np.linspace(0,count-1,min(64,count)).astype(int)
            test_points=geo[torch.as_tensor(selection,device=geo.device)].cpu().numpy().astype(np.float64)/1000
            transformed=gauge['scale']*np.einsum('ij,nkj->nki',np.array(gauge['rotation']),test_points)+np.array(gauge['translation'])
            invariance=float(np.abs(project(test_points,original_r[selection],original_t[selection],focal)-project(transformed,rot[selection],trans[selection],focal)).max())
        matrices=np.repeat(np.eye(4)[None],count,axis=0)
        matrices[:,:3,:3]=np.swapaxes(rot,1,2)
        matrices[:,:3,3]=-np.einsum('nij,nj->ni',matrices[:,:3,:3],trans)
        assert np.isfinite(matrices).all() and np.allclose(np.linalg.det(rot),1,atol=1e-5)
        np.save(data/'poses_training_gauge.npy',matrices.astype(np.float32))
        checks=dict(landmark_median=float(np.median(landmark_errors))<=limits['landmark_rms_px_median_max'],
            landmark_p95=float(np.quantile(landmark_errors,.95))<=limits['landmark_rms_px_p95_max'],
            flow_median=float(np.median(flow_errors))<=limits['flow_reprojection_rms_px_median_max'],
            flow_p95=float(np.quantile(flow_errors,.95))<=limits['flow_reprojection_rms_px_p95_max'],
            projection_invariance=invariance<=limits['heldout_projection_invariance_max_px'])
        rows.append(dict(split=split,frames=count,identity_and_focal_exact=True,
            landmark_rms_px_median=float(np.median(landmark_errors)),landmark_rms_px_p95=float(np.quantile(landmark_errors,.95)),
            flow_rms_px_median=float(np.median(flow_errors)),flow_rms_px_p95=float(np.quantile(flow_errors,.95)),
            projection_invariance_max_px=invariance,checks=checks,passed=all(checks.values())))
    result=dict(identity=identity,passed=all(gauge_checks.values()) and all(x['passed'] for x in rows),gauge=gauge,splits=rows,
        contract_sha256=hashlib.sha256((root/'preprocessing_contract_v1.json').read_bytes()).hexdigest(),training_admitted=False)
    (folder/'pose_qualification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    assert result['passed'],'pose qualification failed; do not train'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True)
    a=p.parse_args();run(a.root,a.identity)
