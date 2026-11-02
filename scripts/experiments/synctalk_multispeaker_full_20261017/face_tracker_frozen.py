"""Official BFM landmark pose fit with training identity and focal frozen."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

def run(root,identity,split):
    assert split in ['val','test']
    tracking=root/'official/data_utils/face_tracking';sys.path.insert(0,str(tracking))
    from facemodel import Face_3DMM
    from data_loader import load_dir
    from util import forward_transform,cal_lan_loss
    folder=root/'prepared'/identity/split
    target=folder/'track_params.pt';assert not target.exists()
    training_path=root/'prepared'/identity/'train/track_params.pt'
    trained=torch.load(training_path,map_location='cpu',weights_only=False)
    count=len(list((folder/'ori_imgs').glob('*.lms')))
    assert count=={'val':750,'test':1500}[split]
    lms=load_dir(str(folder/'ori_imgs'),0,count)
    model=Face_3DMM(str(tracking/'3DMM'),100,79,100,34650)
    lands=torch.as_tensor(np.loadtxt(tracking/'3DMM/lands_info.txt',dtype=np.int32)).cuda()
    id_para=trained['id'].cuda().detach();focal=trained['focal'].cuda().detach()
    exp=lms.new_zeros((count,79),requires_grad=True)
    euler=lms.new_zeros((count,3),requires_grad=True)
    trans=lms.new_zeros((count,3),requires_grad=True);trans.data[:,2]-=600
    optimizer_exp=torch.optim.Adam([exp],lr=.3)
    optimizer_pose=torch.optim.Adam([euler,trans],lr=.3)
    cxy=torch.tensor([256.,256.],device='cuda')
    for iteration in range(2000):
        geo=model.forward_geo_sub(id_para.expand(count,-1),exp,lands[-51:].long())
        projection=forward_transform(geo,euler,trans,focal,cxy)
        landmark_loss=cal_lan_loss(projection[:,:,:2],lms[:,-51:,:].detach())
        loss=landmark_loss+torch.mean(id_para*id_para)*8+torch.mean(exp*exp)*.5
        optimizer_exp.zero_grad();optimizer_pose.zero_grad();loss.backward()
        if iteration>1000:optimizer_exp.step()
        optimizer_pose.step()
        if (iteration+1)%250==0:print(json.dumps(dict(identity=identity,split=split,iteration=iteration+1,landmark_mse=float(landmark_loss))),flush=True)
    assert id_para.grad is None and focal.grad is None
    assert torch.equal(id_para.cpu(),trained['id']) and torch.equal(focal.cpu(),trained['focal'])
    torch.save(dict(id=id_para.cpu(),exp=exp.detach().cpu(),euler=euler.detach().cpu(),trans=trans.detach().cpu(),focal=focal.cpu()),target)
    with torch.no_grad():
        geo=model.forward_geo_sub(id_para.expand(count,-1),exp,lands[-51:].long())
        projection=forward_transform(geo,euler,trans,focal,cxy)
    errors=torch.sqrt(torch.mean((projection[:,:,:2]-lms[:,-51:,:])**2,dim=(1,2))).detach().cpu().numpy()
    receipt=dict(identity=identity,split=split,training_track_sha256=hashlib.sha256(training_path.read_bytes()).hexdigest(),
        frozen_identity_exact=True,frozen_focal_exact=True,frames=count,iterations=2000,
        landmark_rms_px_median=float(np.median(errors)),landmark_rms_px_p95=float(np.quantile(errors,.95)),
        training_admitted=False,method='official second-stage objective and Adam schedule, identity/focal held fixed to training result')
    (folder/'frozen_track_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True);p.add_argument('--split',required=True)
    a=p.parse_args();run(a.root,a.identity,a.split)
