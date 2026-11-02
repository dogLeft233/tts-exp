"""Original NeuralNRT inference/keypoint selection with bounded dense-flow storage."""
import argparse,hashlib,json,os,sys,textwrap,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from geometry import alpha_flow_mask

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def run(root,identity,split,smoke=False):
    source=root/'preprocessed'/identity/split;folder=root/'prepared'/identity/split
    if smoke:folder=root/'flow_smoke'/identity/split
    folder.mkdir(parents=True,exist_ok=True)
    os.chdir(folder)
    mask_dir=folder/'flow_masks';mask_dir.mkdir(exist_ok=True)
    count=len(list((source/'ori_imgs').glob('*.jpg')))
    assert count=={'train':6000,'val':750,'test':1500}[split]
    targets=list(range(3)) if smoke else list(range(count))
    for i in set(targets+[2]):
        path=mask_dir/f'{i}.png'
        assert not path.exists(),'do not overwrite an executed mask'
        raw=cv2.imread(str(source/'parsing'/f'{i}_face.png'))
        assert raw.shape==(512,512,3)
        mask=alpha_flow_mask(raw);cv2.imwrite(str(path),mask)
        assert 0<float(np.mean(mask[:,:,0]>250))<.8
        assert np.array_equal(cv2.imread(str(path)),mask)
    flow_code=root/'official/data_utils/UNFaceFlow'
    sys.path.insert(0,str(flow_code));sys.path.insert(0,str(flow_code/'core'))
    from data_test_flow.dd_dataset import DDDataset
    from options_test_flow import TestOptions
    from models.network_test_flow import NeuralNRT
    argv=sys.argv[:];sys.argv=[str(flow_code/'test_flow.py'),'--datapath',str(folder/'flow_list.txt')]
    opt=TestOptions().parse();sys.argv=argv
    opt.width=512;opt.height=512;opt.num_threads=0;opt.shuffle=False;opt.batch_size=1
    lines=[f'{source}/ori_imgs/2.jpg {mask_dir}/2.png {source}/ori_imgs/{i}.jpg {mask_dir}/{i}.png' for i in targets]
    (folder/'flow_list.txt').write_text('\n'.join(lines)+'\n')
    dataset=DDDataset();dataset.initialize(opt);assert len(dataset)==len(targets)
    model=NeuralNRT(opt,str(flow_code/'pretrain_model/raft-small.pth'))
    state=torch.load(flow_code/'sgd_NNRT_model_epoch19008_50000.pth',map_location='cpu',weights_only=False)
    model.CorresPred.load_state_dict(state['net_C'],strict=True)
    model.ImportanceW.load_state_dict(state['net_W'],strict=True);model=model.cuda().eval()
    rigid_mask=cv2.imread(str(mask_dir/'2.png'))[:,:,0]>250
    rigid_num=int(rigid_mask.sum());first=min(2500,len(targets))
    rigid_path=folder/'rigid_flow.npy';assert not rigid_path.exists()
    required=first*2*rigid_num*4
    import shutil
    assert shutil.disk_usage(folder).free>required+512*1024**2,'insufficient bounded flow workspace'
    rigid=np.lib.format.open_memmap(rigid_path,mode='w+',dtype=np.float32,shape=(first,2,rigid_num))
    def predict(i):
        data=dataset[i]
        get=lambda key:data[key].unsqueeze(0).cuda()
        with torch.no_grad():
            flow=model(get('src_crop_color'),get('tar_crop_color'),get('src_color'),get('tar_color'),get('Crop_param'))
            return (flow[0]*get('src_mask')[0]).cpu().numpy()
    started=time.time();smoke_paths=[]
    for i in range(first):
        flow=predict(i);rigid[i]=flow[:,rigid_mask]
        if i<3:
            path=folder/f'dense_control_{i}.npy';np.save(path,flow);smoke_paths.append(path)
            assert np.array_equal(np.load(path)[:,rigid_mask],rigid[i])
        if (i+1)%250==0:print(json.dumps(dict(identity=identity,split=split,phase='rigid_flow',frames=i+1)),flush=True)
    rigid.flush()
    if smoke:
        repeat=predict(0);assert np.array_equal(repeat,np.load(smoke_paths[0]))
        receipt=dict(passed=True,identity=identity,split=split,frames=3,rigid_pixels=rigid_num,
            dense_rigid_storage_exact=True,repeated_forward_exact=True,mask_alpha_semantics_passed=True,
            network_weights_sha256=sha(flow_code/'sgd_NNRT_model_epoch19008_50000.pth'),
            rigid_flow_sha256=sha(rigid_path),training_admitted=False)
        (folder/'smoke_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True);return
    official_source=(root/'official/data_utils/process.py').read_text()
    begin=official_source.index('    rigid_flow = rigid_flow.transpose((2, 1, 0))')
    end=official_source.index('    key_xys = np.loadtxt',begin)
    selection=textwrap.dedent(official_source[begin:end])
    # Execute the pinned original block verbatim, including uint8 conversion, grid and neighborhood rules.
    namespace=dict(np=np,torch=torch,F=F,cv2=cv2,os=os,rigid_flow=np.asarray(rigid),rigid_mask=rigid_mask,
        h=512,w=512,base_dir=str(folder),face_img=cv2.imread(str(source/'ori_imgs/2.jpg')))
    exec(compile(selection,str(root/'official/data_utils/process.py')+'::rigid_selection','exec'),namespace)
    keys=np.loadtxt(folder/'keypoints.txt',dtype=np.int32).reshape(-1,2)
    assert len(keys)>=6,'fewer than six rigid reference points'
    indexes=np.full((512,512),-1,dtype=np.int64);indexes[rigid_mask]=np.arange(rigid_num)
    positions=indexes[keys[:,1],keys[:,0]];assert (positions>=0).all()
    tracks=np.empty((count,len(keys),2),dtype=np.float32)
    tracks[:first]=keys[None]+np.asarray(rigid)[:,:,positions].transpose(0,2,1)
    for i in range(first,count):
        flow=predict(i);tracks[i]=keys+flow[:,keys[:,1],keys[:,0]].T
        if (i+1)%250==0:print(json.dumps(dict(identity=identity,split=split,phase='keypoint_flow',frames=i+1)),flush=True)
    assert np.isfinite(tracks).all();np.save(folder/'track_xys.npy',tracks)
    for i,path in enumerate(smoke_paths):
        dense=np.load(path)
        assert np.array_equal(tracks[i],(keys+dense[:,keys[:,1],keys[:,0]].T).astype(np.float32))
    receipt=dict(passed=True,identity=identity,split=split,frames=count,rigid_frames=first,rigid_pixels=rigid_num,
        keypoints=len(keys),dense_control_track_parity=True,rigid_selection_source_sha256=hashlib.sha256(selection.encode()).hexdigest(),
        network_weights_sha256=sha(flow_code/'sgd_NNRT_model_epoch19008_50000.pth'),
        rigid_flow_sha256=sha(rigid_path),tracks_sha256=sha(folder/'track_xys.npy'),elapsed_seconds=time.time()-started,
        training_admitted=False)
    (folder/'flow_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True);p.add_argument('--split',required=True);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.root,a.identity,a.split,a.smoke)
