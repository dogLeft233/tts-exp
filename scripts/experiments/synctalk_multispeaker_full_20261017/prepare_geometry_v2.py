"""Shared training-reference geometry; explicit successor of failed v1 calibration."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from scipy.spatial.transform import Rotation
from geometry import alpha_flow_mask,fit_training_gauge,map_poses_to_training_gauge,official_euler_from_rotation,project
from qualify_pose import triangulate
from flow_stream import sha

def matrices(rot,trans):
    result=np.repeat(np.eye(4)[None],len(rot),axis=0)
    result[:,:3,:3]=np.swapaxes(rot,1,2)
    result[:,:3,3]=-np.einsum('nij,nj->ni',result[:,:3,:3],trans)
    return result

def optimize_pose(points,rot,trans,tracks,focal,forward_transform,euler2rot):
    euler=torch.tensor(official_euler_from_rotation(rot),dtype=torch.float32,device='cuda',requires_grad=True)
    translation=torch.tensor(trans,dtype=torch.float32,device='cuda',requires_grad=True)
    assert np.abs(euler2rot(euler.detach()).cpu().numpy()-rot).max()<1e-6
    fixed=torch.tensor(points,dtype=torch.float32,device='cuda');before=fixed.cpu().numpy().tobytes()
    geo=fixed[None].expand(len(rot),-1,-1);target=torch.tensor(tracks,dtype=torch.float32,device='cuda')
    f=torch.tensor([focal],device='cuda');cxy=torch.tensor([256.,256.],device='cuda')
    optimizer=torch.optim.Adam([euler,translation],lr=.0001)
    for iteration in range(8000):
        predicted=forward_transform(geo,euler,translation,f,cxy)[:,:,:2]
        loss=torch.mean((predicted-target)**2)
        optimizer.zero_grad();loss.backward();optimizer.step()
    assert fixed.grad is None and fixed.cpu().numpy().tobytes()==before
    rot=euler2rot(euler.detach()).cpu().numpy().astype(np.float64)
    trans=translation.detach().cpu().numpy().astype(np.float64)
    p=project(np.broadcast_to(fixed.cpu().numpy().astype(np.float64),(len(rot),)+points.shape),rot,trans,focal)
    torch_p=forward_transform(geo,euler,translation,f,cxy)[:,:,:2].detach().cpu().numpy()
    parity=float(np.abs(p-torch_p).max())
    errors=np.sqrt(np.mean((p-np.asarray(tracks,dtype=np.float64))**2,axis=(1,2)))
    return rot,trans,dict(flow_rms_px_median=float(np.median(errors)),flow_rms_px_p95=float(np.quantile(errors,.95)),
        independent_projection_max_px=parity,fixed_training_points_unchanged=True,
        fixed_float32_points_sha256=hashlib.sha256(before).hexdigest())

def run(root,identity):
    folder=root/'prepared'/identity;output=folder/'geometry_v2';output.mkdir(exist_ok=False)
    contract_path=root/'preprocessing_contract_v2.json';contract=json.loads(contract_path.read_text());limits=contract['geometry_gates_v2']
    raw=Path(__file__).read_bytes();(output/Path(__file__).name).write_bytes(raw)
    sys.path.insert(0,str(root/'official/data_utils/face_tracking'))
    from util import euler2rot,forward_transform
    from facemodel import Face_3DMM
    initial=torch.load(folder/'train/track_params.pt',weights_only=False)
    bundle=torch.load(folder/'train/bundle_adjustment.pt',weights_only=False)
    r0=euler2rot(initial['euler']).numpy().astype(np.float64);t0=initial['trans'].numpy().astype(np.float64)/1000
    r1=euler2rot(bundle['euler']).numpy().astype(np.float64);t1=bundle['trans'].numpy().astype(np.float64)
    training_tracks=np.load(folder/'train/track_xys.npy').astype(np.float64);focal=float(bundle['focal'][0])
    assert len(r0)==len(r1)==len(training_tracks)==6000 and torch.equal(initial['focal'],bundle['focal'])
    gauge=fit_training_gauge(r0,t0,r1,t1)
    assert gauge['condition']<=limits['initialization_similarity_condition_max']
    assert limits['initialization_similarity_scale_min']<=gauge['scale']<=limits['initialization_similarity_scale_max']
    points=triangulate(r1,t1,training_tracks,focal)
    np.save(output/'fixed_training_points_float64.npy',points)
    np.save(output/'fixed_training_points_float32.npy',points.astype(np.float32))
    np.save(output/'train_poses.npy',matrices(r1,t1))
    result=dict(identity=identity,stage='shared_training_reference_geometry_v2',training_admitted=False,
        contract_sha256=sha(contract_path),execution_source_sha256=hashlib.sha256(raw).hexdigest(),
        previous_v1_qualification_sha256=sha(folder/'pose_qualification.json') if (folder/'pose_qualification.json').exists() else None,
        training_bundle_sha256=sha(folder/'train/bundle_adjustment.pt'),training_tracks_sha256=sha(folder/'train/track_xys.npy'),
        initialization_only_similarity=gauge,controls=[],splits=[],fixed_training_points_sha256=sha(output/'fixed_training_points_float32.npy'))
    def save():(output/'geometry_qualification.json').write_text(json.dumps(result,indent=2))
    save()
    control_gauge=fit_training_gauge(r0[:4800],t0[:4800],r1[:4800],t1[:4800])
    control_points=triangulate(r1[:4800],t1[:4800],training_tracks[:4800],focal)
    np.save(output/'control_points_first4800.npy',control_points)
    for name,start,stop in [('early',3600,4800),('late',4800,6000)]:
        rot,trans=map_poses_to_training_gauge(r0[start:stop],t0[start:stop],control_gauge)
        rot,trans,control=optimize_pose(control_points,rot,trans,training_tracks[start:stop],focal,forward_transform,euler2rot)
        angles=np.degrees(Rotation.from_matrix(rot@np.swapaxes(r1[start:stop],1,2)).magnitude())
        translations=np.linalg.norm(trans-t1[start:stop],axis=1)
        control.update(name=name,frames=[start,stop],rotation_p95_degrees=float(np.quantile(angles,.95)),translation_p95_m=float(np.quantile(translations,.95)))
        control['passed']=bool(control['rotation_p95_degrees']<=limits['training_control_camera_rotation_p95_degrees_max'] and control['translation_p95_m']<=limits['training_control_camera_translation_p95_m_max'])
        np.savez(output/(name+'_control_poses.npz'),rotation=rot,translation=trans)
        result['controls'].append(control);save();print(json.dumps(dict(identity=identity,control=control)),flush=True)
        assert control['passed'],'training camera recovery control failed; no heldout qualification'
    # Independent multi-view conditioning, identical to the triangulation normal matrix.
    u=(training_tracks[:,:,0]-256)/focal;v=(training_tracks[:,:,1]-256)/focal
    a=np.stack([r1[:,None,0,:]+u[:,:,None]*r1[:,None,2,:],r1[:,None,1,:]-v[:,:,None]*r1[:,None,2,:]],axis=2)
    normals=np.einsum('nkij,nkil->kjl',a,a)
    result['triangulation_condition_max']=float(np.linalg.cond(normals).max())
    assert result['triangulation_condition_max']<=limits['triangulation_normal_condition_max']
    model_bfm=Face_3DMM(str(root/'official/data_utils/face_tracking/3DMM'),100,79,100,34650)
    lands=torch.as_tensor(np.loadtxt(root/'official/data_utils/face_tracking/3DMM/lands_info.txt',dtype=np.int32)).cuda()
    for split,count in [('train',6000),('val',750),('test',1500)]:
        track=torch.load(folder/split/'track_params.pt',weights_only=False)
        assert torch.equal(track['id'],initial['id']) and torch.equal(track['focal'],initial['focal'])
        target=np.stack([np.loadtxt(folder/split/'ori_imgs'/f'{i}.lms')[-51:] for i in range(count)])
        with torch.no_grad():
            geo=model_bfm.forward_geo_sub(track['id'].cuda().expand(count,-1),track['exp'].cuda(),lands[-51:].long())
            predicted=forward_transform(geo,track['euler'].cuda(),track['trans'].cuda(),track['focal'].cuda(),torch.tensor([256.,256.],device='cuda'))[:,:,:2].cpu().numpy()
        errors=np.sqrt(np.mean((predicted-target)**2,axis=(1,2)))
        row=dict(split=split,frames=count,frozen_identity_and_focal_exact=True,landmark_rms_px_median=float(np.median(errors)),landmark_rms_px_p95=float(np.quantile(errors,.95)))
        assert row['landmark_rms_px_median']<=5 and row['landmark_rms_px_p95']<=10
        result['splits'].append(row)
    del model_bfm;torch.cuda.empty_cache()
    flow_code=root/'official/data_utils/UNFaceFlow';sys.path.insert(0,str(flow_code));sys.path.insert(0,str(flow_code/'core'))
    from data_test_flow.dd_dataset import DDDataset
    from options_test_flow import TestOptions
    from models.network_test_flow import NeuralNRT
    keys=np.loadtxt(folder/'train/keypoints.txt',dtype=np.int32).reshape(-1,2)
    assert len(keys)==len(points)
    reference=folder/'train/ori_imgs/2.jpg';reference_mask=folder/'train/flow_masks/2.png'
    expected_mask=alpha_flow_mask(cv2.imread(str(folder/'train/parsing/2_face.png')))
    assert np.array_equal(cv2.imread(str(reference_mask)),expected_mask)
    result['reference']=dict(image_sha256=sha(reference),mask_sha256=sha(reference_mask),keypoints_sha256=sha(folder/'train/keypoints.txt'))
    weights=flow_code/'sgd_NNRT_model_epoch19008_50000.pth'
    for split,count in [('val',750),('test',1500)]:
        data=folder/split;out=output/split;out.mkdir();os.chdir(out)
        masks=out/'flow_masks';masks.mkdir()
        lines=[]
        controls=[0,1,2,2500,4500,5999]
        for i in controls:lines.append(f'{reference} {reference_mask} {folder}/train/ori_imgs/{i}.jpg {folder}/train/flow_masks/{i}.png')
        for i in range(count):
            mask=alpha_flow_mask(cv2.imread(str(data/'parsing'/f'{i}_face.png')))
            path=masks/f'{i}.png';assert cv2.imwrite(str(path),mask)
            assert np.array_equal(cv2.imread(str(path)),mask)
            lines.append(f'{reference} {reference_mask} {data}/ori_imgs/{i}.jpg {path}')
        (out/'flow_list.txt').write_text('\n'.join(lines)+'\n')
        argv=sys.argv[:];sys.argv=[str(flow_code/'test_flow.py'),'--datapath',str(out/'flow_list.txt')]
        opt=TestOptions().parse();sys.argv=argv
        opt.width=512;opt.height=512;opt.num_threads=0;opt.shuffle=False;opt.batch_size=1
        dataset=DDDataset();dataset.initialize(opt)
        model=NeuralNRT(opt,str(flow_code/'pretrain_model/raft-small.pth'))
        state=torch.load(weights,map_location='cpu',weights_only=False)
        model.CorresPred.load_state_dict(state['net_C'],strict=True);model.ImportanceW.load_state_dict(state['net_W'],strict=True)
        model=model.cuda().eval()
        def predict(index):
            item=dataset[index];get=lambda key:item[key].unsqueeze(0).cuda()
            with torch.no_grad():
                flow=(model(get('src_crop_color'),get('tar_crop_color'),get('src_color'),get('tar_color'),get('Crop_param'))[0]*get('src_mask')[0]).cpu().numpy()
            return (keys+flow[:,keys[:,1],keys[:,0]].T).astype(np.float32)
        for n,i in enumerate(controls):assert np.array_equal(predict(n),training_tracks[i].astype(np.float32))
        tracks=np.empty((count,len(keys),2),dtype=np.float32)
        for i in range(count):
            tracks[i]=predict(i+len(controls))
            if (i+1)%250==0:print(json.dumps(dict(identity=identity,split=split,phase='shared_reference_flow',frames=i+1)),flush=True)
        for i in [0,count//2,count-1]:assert np.array_equal(predict(i+len(controls)),tracks[i])
        assert np.isfinite(tracks).all();np.save(out/'track_xys.npy',tracks)
        del model;torch.cuda.empty_cache()
        original=torch.load(data/'track_params.pt',weights_only=False)
        rot=euler2rot(original['euler']).numpy().astype(np.float64);trans=original['trans'].numpy().astype(np.float64)/1000
        rot,trans=map_poses_to_training_gauge(rot,trans,gauge)
        rot,trans,quality=optimize_pose(points,rot,trans,tracks,focal,forward_transform,euler2rot)
        quality.update(training_control_dense_flow_exact=True,heldout_repeated_flow_exact=True,tracks_sha256=sha(out/'track_xys.npy'),weights_sha256=sha(weights),pose_initialization_only=True)
        quality['passed']=bool(quality['flow_rms_px_median']<=2 and quality['flow_rms_px_p95']<=5 and quality['independent_projection_max_px']<=.0001)
        np.savez(out/'pose_arrays.npz',rotation=rot,translation=trans)
        np.save(out/'poses.npy',matrices(rot,trans))
        (out/'pose_qualification.json').write_text(json.dumps(quality,indent=2))
        row=next(x for x in result['splits'] if x['split']==split);row.update(quality);save();print(json.dumps(dict(identity=identity,split=split,quality=quality)),flush=True)
        assert quality['passed'],'full heldout shared-reference geometry failed; do not train'
    reconstruction=project(np.broadcast_to(points.astype(np.float32).astype(np.float64),(6000,)+points.shape),r1,t1,focal)
    errors=np.sqrt(np.mean((reconstruction-training_tracks)**2,axis=(1,2)))
    train_row=result['splits'][0];train_row.update(flow_rms_px_median=float(np.median(errors)),flow_rms_px_p95=float(np.quantile(errors,.95)))
    assert train_row['flow_rms_px_median']<=2 and train_row['flow_rms_px_p95']<=5
    assert sha(output/'fixed_training_points_float32.npy')==result['fixed_training_points_sha256']
    result.update(passed=True,train_original_BA_preserved=True,heldout_geometry_never_fitted=True)
    save()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True)
    a=p.parse_args();run(a.root,a.identity)
