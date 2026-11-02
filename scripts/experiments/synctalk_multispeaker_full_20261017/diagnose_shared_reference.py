"""Diagnostic only: heldout flow and pose against one fixed training point cloud."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import cv2,numpy as np,torch
from scipy.spatial.transform import Rotation
from geometry import fit_training_gauge,map_poses_to_training_gauge,project
from qualify_pose import triangulate
from flow_stream import sha

def run(root,identity):
    output=root/'diagnostics'/identity/'shared_reference_attempt01'
    output.mkdir(parents=True,exist_ok=False);os.chdir(output)
    raw=Path(__file__).read_bytes();(output/Path(__file__).name).write_bytes(raw)
    training=root/'prepared'/identity/'train'
    sys.path.insert(0,str(root/'official/data_utils/face_tracking'))
    from util import euler2rot,forward_transform
    train=torch.load(training/'track_params.pt',weights_only=False)
    bundle=torch.load(training/'bundle_adjustment.pt',weights_only=False)
    r0=euler2rot(train['euler']).numpy().astype(np.float64);t0=train['trans'].numpy().astype(np.float64)/1000
    r1=euler2rot(bundle['euler']).numpy().astype(np.float64);t1=bundle['trans'].numpy().astype(np.float64)
    focal=float(bundle['focal'][0]);tracks=np.load(training/'track_xys.npy').astype(np.float64)
    points=triangulate(r1,t1,tracks,focal);np.save(output/'training_fixed_points.npy',points)
    keys=np.loadtxt(training/'keypoints.txt',dtype=np.int32).reshape(-1,2)
    assert len(keys)==len(points)
    gauge=fit_training_gauge(r0,t0,r1,t1)
    flow_code=root/'official/data_utils/UNFaceFlow'
    sys.path.insert(0,str(flow_code));sys.path.insert(0,str(flow_code/'core'))
    from data_test_flow.dd_dataset import DDDataset
    from options_test_flow import TestOptions
    from models.network_test_flow import NeuralNRT
    items=[('train',x) for x in [0,1,2,2500,4500,5999]]
    for split,count in [('val',750),('test',1500)]:
        items.extend((split,int(x)) for x in np.linspace(0,count-1,20))
    paths=[]
    for split,i in items:
        folder=root/'prepared'/identity/split
        paths.append(f'{training}/ori_imgs/2.jpg {training}/flow_masks/2.png {folder}/ori_imgs/{i}.jpg {folder}/flow_masks/{i}.png')
    (output/'flow_list.txt').write_text('\n'.join(paths)+'\n')
    argv=sys.argv[:];sys.argv=[str(flow_code/'test_flow.py'),'--datapath',str(output/'flow_list.txt')]
    opt=TestOptions().parse();sys.argv=argv
    opt.width=512;opt.height=512;opt.num_threads=0;opt.shuffle=False;opt.batch_size=1
    dataset=DDDataset();dataset.initialize(opt)
    model=NeuralNRT(opt,str(flow_code/'pretrain_model/raft-small.pth'))
    weights=flow_code/'sgd_NNRT_model_epoch19008_50000.pth'
    state=torch.load(weights,map_location='cpu',weights_only=False)
    model.CorresPred.load_state_dict(state['net_C'],strict=True)
    model.ImportanceW.load_state_dict(state['net_W'],strict=True);model=model.cuda().eval()
    target_tracks=[]
    for index,(split,i) in enumerate(items):
        data=dataset[index];get=lambda key:data[key].unsqueeze(0).cuda()
        with torch.no_grad():
            flow=(model(get('src_crop_color'),get('tar_crop_color'),get('src_color'),get('tar_color'),get('Crop_param'))[0]*get('src_mask')[0]).cpu().numpy()
        row=(keys+flow[:,keys[:,1],keys[:,0]].T).astype(np.float32)
        if split=='train':assert np.array_equal(row,tracks[i].astype(np.float32))
        target_tracks.append(row)
    target_tracks=np.stack(target_tracks)
    np.save(output/'shared_tracks.npy',target_tracks)
    del model;torch.cuda.empty_cache()
    results={}
    for split in ['val','test']:
        chosen=[(n,i) for n,(role,i) in enumerate(items) if role==split];indexes=[i for _,i in chosen]
        initial=torch.load(root/'prepared'/identity/split/'track_params.pt',weights_only=False)
        r=euler2rot(initial['euler'][indexes]).numpy().astype(np.float64)
        t=initial['trans'][indexes].numpy().astype(np.float64)/1000
        r,t=map_poses_to_training_gauge(r,t,gauge)
        angles=Rotation.from_matrix(r).as_euler('XYZ');angles[:,2]*=-1
        euler=torch.tensor(angles,dtype=torch.float32,device='cuda',requires_grad=True)
        trans=torch.tensor(t,dtype=torch.float32,device='cuda',requires_grad=True)
        assert np.max(np.abs(euler2rot(euler.detach()).cpu().numpy()-r))<1e-6
        geo=torch.tensor(points,dtype=torch.float32,device='cuda')[None].expand(len(chosen),-1,-1)
        target=torch.tensor(target_tracks[[n for n,_ in chosen]],device='cuda')
        focal_t=torch.tensor([focal],device='cuda');cxy=torch.tensor([256.,256.],device='cuda')
        optimizer=torch.optim.Adam([euler,trans],lr=.0001)
        for iteration in range(8000):
            projection=forward_transform(geo,euler,trans,focal_t,cxy)[:,:,:2]
            loss=torch.mean((projection-target)**2)
            optimizer.zero_grad();loss.backward();optimizer.step()
        final_r=euler2rot(euler.detach()).cpu().numpy().astype(np.float64)
        final_t=trans.detach().cpu().numpy().astype(np.float64)
        projected=project(np.broadcast_to(points,(len(chosen),)+points.shape),final_r,final_t,focal)
        errors=np.sqrt(np.mean((projected-target.cpu().numpy())**2,axis=(1,2)))
        results[split]=dict(frames=indexes,flow_rms_px_median=float(np.median(errors)),flow_rms_px_p95=float(np.quantile(errors,.95)),
            flow_rms_px_per_frame=errors.tolist(),passes_existing_flow_limits=bool(np.median(errors)<=2 and np.quantile(errors,.95)<=5))
        np.savez(output/(split+'_camera.npz'),rotation=final_r,translation=final_t,projected=projected)
        print(json.dumps(results[split]),flush=True)
        # Ordered qualitative panels, never use score to select or drop frames.
        panels=[]
        for n,((_,i),xy) in enumerate(zip(chosen,projected)):
            image=cv2.imread(str(root/'prepared'/identity/split/'ori_imgs'/f'{i}.jpg'))
            for observation,prediction in zip(target[n].cpu().numpy(),xy):
                cv2.circle(image,tuple(np.round(observation).astype(int)),2,(0,255,0),-1)
                cv2.circle(image,tuple(np.round(prediction).astype(int)),1,(0,0,255),-1)
            cv2.putText(image,f'{split} frame {i} RMS {errors[n]:.2f}px',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            panels.append(cv2.resize(image,(256,256)))
        gallery=np.concatenate([np.concatenate(panels[i:i+5],axis=1) for i in range(0,20,5)],axis=0)
        cv2.imwrite(str(output/(split+'_correspondence_review.jpg')),gallery)
    result=dict(identity=identity,completed=True,training_admitted=False,source_sha256=hashlib.sha256(raw).hexdigest(),
        weights_sha256=sha(weights),training_control_six_forward_exact=True,frozen_training_points=True,
        diagnostic_only=True,results=results)
    (output/'diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True)
    a=p.parse_args();run(a.root,a.identity)
