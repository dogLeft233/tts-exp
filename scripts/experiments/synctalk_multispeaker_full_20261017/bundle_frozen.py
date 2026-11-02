"""Heldout official flow loss with points fixed in the training BFM coordinate frame."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

def run(root,identity,split):
    assert split in ['val','test']
    sys.path.insert(0,str(root/'official/data_utils/face_tracking'))
    from util import forward_transform,cal_lan_loss
    folder=root/'prepared'/identity/split
    assert not (folder/'bundle_adjustment.pt').exists()
    initial=torch.load(folder/'track_params.pt',weights_only=False)
    euler=initial['euler'].cuda().detach().requires_grad_()
    trans=(initial['trans'].cuda()/1000).detach().requires_grad_()
    focal=initial['focal'].cuda().detach()
    tracks=torch.as_tensor(np.load(folder/'track_xys.npy')).float().cuda()
    count,point_count,_=tracks.shape
    assert count=={'val':750,'test':1500}[split]
    pts=torch.zeros((point_count,3),dtype=torch.float32,device='cuda',requires_grad=True)
    cxy=torch.tensor([256.,256.],device='cuda');optimizer_pts=torch.optim.Adam([pts],lr=.01)
    for iteration in range(500):
        projection=forward_transform(pts[None].expand(count,-1,-1),euler,trans,focal,cxy)
        loss=cal_lan_loss(projection[:,:,:2],tracks)
        optimizer_pts.zero_grad();loss.backward();optimizer_pts.step()
    pts=pts.detach();before=pts.cpu().numpy().tobytes();euler.grad=None;trans.grad=None
    optimizer_pose=torch.optim.Adam([euler,trans],lr=.0001)
    for iteration in range(8000):
        projection=forward_transform(pts[None].expand(count,-1,-1),euler,trans,focal,cxy)
        loss=cal_lan_loss(projection[:,:,:2],tracks)
        optimizer_pose.zero_grad();loss.backward();optimizer_pose.step()
        if (iteration+1)%1000==0:print(json.dumps(dict(identity=identity,split=split,iteration=iteration+1,flow_mse=float(loss))),flush=True)
    assert pts.grad is None and pts.cpu().numpy().tobytes()==before
    with torch.no_grad():projection=forward_transform(pts[None].expand(count,-1,-1),euler,trans,focal,cxy)
    errors=torch.sqrt(torch.mean((projection[:,:,:2]-tracks)**2,dim=(1,2))).cpu().numpy()
    torch.save(dict(euler=euler.detach().cpu(),trans=trans.detach().cpu(),focal=focal.cpu(),fixed_points=pts.cpu()),folder/'bundle_adjustment.pt')
    receipt=dict(identity=identity,split=split,frames=count,keypoints=point_count,point_fit_iterations=500,pose_iterations=8000,
        fixed_points_unchanged=True,fixed_points_sha256=hashlib.sha256(before).hexdigest(),
        flow_rms_px_median=float(np.median(errors)),flow_rms_px_p95=float(np.quantile(errors,.95)),
        focal_exact_to_training=True,training_admitted=False,
        method='official reprojection objective; point fitting at frozen-identity initial cameras then heldout pose-only refinement')
    (folder/'bundle_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True);p.add_argument('--split',required=True)
    a=p.parse_args();run(a.root,a.identity,a.split)
