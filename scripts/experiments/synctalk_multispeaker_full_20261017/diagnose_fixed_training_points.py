"""Train-only camera recovery against fixed official training flow geometry."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from geometry import fit_training_gauge,map_poses_to_training_gauge,project
from qualify_pose import triangulate

def run(root,identity):
    output=root/'diagnostics'/identity/'fixed_training_points_attempt02'
    output.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).read_bytes();(output/Path(__file__).name).write_bytes(source)
    sys.path.insert(0,str(root/'official/data_utils/face_tracking'))
    from util import euler2rot,forward_transform
    folder=root/'prepared'/identity/'train'
    initial=torch.load(folder/'track_params.pt',weights_only=False)
    bundle=torch.load(folder/'bundle_adjustment.pt',weights_only=False)
    r0=euler2rot(initial['euler']).numpy().astype(np.float64);t0=initial['trans'].numpy().astype(np.float64)/1000
    r1=euler2rot(bundle['euler']).numpy().astype(np.float64);t1=bundle['trans'].numpy().astype(np.float64)
    tracks=np.load(folder/'track_xys.npy').astype(np.float64);focal=float(bundle['focal'][0])
    gauge=fit_training_gauge(r0[:4800],t0[:4800],r1[:4800],t1[:4800])
    points=triangulate(r1[:4800],t1[:4800],tracks[:4800],focal)
    np.save(output/'training_fixed_points.npy',points)
    recovery={}
    # Two fixed calibration ranges test ordinary and late trajectory support.
    for name,start,stop in [('early_control',3600,4800),('late_control',4800,6000)]:
        rot,trans=map_poses_to_training_gauge(r0[start:stop],t0[start:stop],gauge)
        angle=Rotation.from_matrix(rot).as_euler('XYZ');angle[:,2]*=-1
        euler=torch.tensor(angle,dtype=torch.float32,device='cuda',requires_grad=True)
        translation=torch.tensor(trans,dtype=torch.float32,device='cuda',requires_grad=True)
        assert np.max(np.abs(euler2rot(euler.detach()).cpu().numpy()-rot))<1e-6
        geo=torch.tensor(points,dtype=torch.float32,device='cuda')[None].expand(stop-start,-1,-1)
        target=torch.tensor(tracks[start:stop],dtype=torch.float32,device='cuda')
        focal_t=torch.tensor([focal],device='cuda');cxy=torch.tensor([256.,256.],device='cuda')
        optimizer=torch.optim.Adam([euler,translation],lr=.0001)
        for iteration in range(8000):
            prediction=forward_transform(geo,euler,translation,focal_t,cxy)[:,:,:2]
            loss=torch.mean((prediction-target)**2)
            optimizer.zero_grad();loss.backward();optimizer.step()
        rot=euler2rot(euler.detach()).cpu().numpy().astype(np.float64)
        trans=translation.detach().cpu().numpy().astype(np.float64)
        angle_errors=np.degrees(Rotation.from_matrix(rot@np.swapaxes(r1[start:stop],1,2)).magnitude())
        translation_errors=np.linalg.norm(trans-t1[start:stop],axis=1)
        flow_errors=np.sqrt(np.mean((project(np.broadcast_to(points,(stop-start,)+points.shape),rot,trans,focal)-tracks[start:stop])**2,axis=(1,2)))
        recovery[name]=dict(frames=[start,stop],rotation_p95_degrees=float(np.quantile(angle_errors,.95)),
            translation_p95_m=float(np.quantile(translation_errors,.95)),flow_rms_px_median=float(np.median(flow_errors)),
            flow_rms_px_p95=float(np.quantile(flow_errors,.95)),
            passes_existing_camera_limits=bool(np.quantile(angle_errors,.95)<=2 and np.quantile(translation_errors,.95)<=.005))
        print(json.dumps(recovery[name]),flush=True)
        np.savez(output/(name+'_camera.npz'),rotation=rot,translation=trans)
    result=dict(identity=identity,completed=True,training_admitted=False,
        source_sha256=hashlib.sha256(source).hexdigest(),calibration_frames=[0,4800],
        reference='original official training BA cameras and fixed training-reference flow points',recovery=recovery)
    (output/'diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True)
    a=p.parse_args();run(a.root,a.identity)
