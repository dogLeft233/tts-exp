"""Independently replay completed flow outputs after correcting reference storage dtype."""
import hashlib,json,os,sys,textwrap
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017');folder=ROOT/'prepared/SherrodBrown/train'
assert not (folder/'flow_receipt.json').exists()
original_source_sha='169435f3b649a21458e18af54ccbcec26397f2f89a143bb0828bacceb392499f'
original=ROOT/'executed_sources'/original_source_sha/'flow_stream.py'
assert hashlib.sha256(original.read_bytes()).hexdigest()==original_source_sha
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
before=sha(folder/'track_xys.npy');tracks=np.load(folder/'track_xys.npy')
keys=np.loadtxt(folder/'keypoints.txt',dtype=np.int32).reshape(-1,2)
assert tracks.shape==(6000,len(keys),2) and tracks.dtype==np.float32 and np.isfinite(tracks).all()
mask=cv2.imread(str(folder/'flow_masks/2.png'))[:,:,0]>250
rigid=np.load(folder/'rigid_flow.npy',mmap_mode='r');assert rigid.shape==(2500,2,int(mask.sum()))
indexes=np.full((512,512),-1,dtype=np.int64);indexes[mask]=np.arange(mask.sum());positions=indexes[keys[:,1],keys[:,0]]
assert (positions>=0).all()
reference=(keys[None]+np.asarray(rigid)[:,:,positions].transpose(0,2,1)).astype(np.float32)
assert np.array_equal(tracks[:2500],reference)
flow_code=ROOT/'official/data_utils/UNFaceFlow';sys.path.insert(0,str(flow_code));sys.path.insert(0,str(flow_code/'core'))
from data_test_flow.dd_dataset import DDDataset
from options_test_flow import TestOptions
from models.network_test_flow import NeuralNRT
sys.argv=[str(flow_code/'test_flow.py'),'--datapath',str(folder/'flow_list.txt')]
opt=TestOptions().parse();opt.num_threads=0;opt.shuffle=False;opt.batch_size=1
dataset=DDDataset();dataset.initialize(opt);assert len(dataset)==6000
model=NeuralNRT(opt,str(flow_code/'pretrain_model/raft-small.pth'))
state=torch.load(flow_code/'sgd_NNRT_model_epoch19008_50000.pth',map_location='cpu',weights_only=False)
model.CorresPred.load_state_dict(state['net_C'],strict=True);model.ImportanceW.load_state_dict(state['net_W'],strict=True);model=model.cuda().eval()
audit=folder/'flow_output_verification';audit.mkdir();os.chdir(audit);replays=[]
for i in [0,1,2,2500,4500,5999]:
    data=dataset[i];get=lambda key:data[key].unsqueeze(0).cuda()
    with torch.no_grad():flow=(model(get('src_crop_color'),get('tar_crop_color'),get('src_color'),get('tar_color'),get('Crop_param'))[0]*get('src_mask')[0]).cpu().numpy()
    expected=(keys+flow[:,keys[:,1],keys[:,0]].T).astype(np.float32)
    assert np.array_equal(tracks[i],expected),(i,float(np.abs(tracks[i]-expected).max()))
    if i<3:assert np.array_equal(flow,np.load(folder/f'dense_control_{i}.npy'))
    np.save(audit/f'dense_replay_{i}.npy',flow);replays.append(dict(frame=i,stored_tracks_exact_to_replay=True))
source=(ROOT/'official/data_utils/process.py').read_text();begin=source.index('    rigid_flow = rigid_flow.transpose((2, 1, 0))');end=source.index('    key_xys = np.loadtxt',begin)
block=textwrap.dedent(source[begin:end]);namespace=dict(np=np,torch=torch,F=F,cv2=cv2,os=os,rigid_flow=np.asarray(rigid),rigid_mask=mask,h=512,w=512,base_dir=str(audit),face_img=cv2.imread(str(ROOT/'preprocessed/SherrodBrown/train/ori_imgs/2.jpg')))
exec(compile(block,'official_process::rigid_selection_replay','exec'),namespace)
assert np.array_equal(keys,np.loadtxt(audit/'keypoints.txt',dtype=np.int32).reshape(-1,2))
np.save(audit/'rigid_laplacian_mean.npy',namespace['rigid_flow_mean'].cpu().numpy())
np.save(audit/'valid_laplacian_frames.npy',namespace['valid_frame'].cpu().numpy())
assert sha(folder/'track_xys.npy')==before
receipt=dict(passed=True,identity='SherrodBrown',split='train',frames=6000,rigid_frames=2500,rigid_pixels=int(mask.sum()),keypoints=len(keys),
    dense_control_track_parity=True,all_first2500_tracks_exact_to_rigid_storage=True,keypoint_selection_recomputed_exact=True,
    independently_replayed_frames=replays,network_weights_sha256=sha(flow_code/'sgd_NNRT_model_epoch19008_50000.pth'),
    rigid_selection_source_sha256=hashlib.sha256(block.encode()).hexdigest(),rigid_flow_sha256=sha(folder/'rigid_flow.npy'),tracks_sha256=before,
    producer_source_sha256=original_source_sha,verification_source_sha256=sha(Path(__file__)),
    verification_repair='producer completed all 6000 frames; original final assertion compared float32 storage against promoted float64 instead of original float32 assignment. No flow, keys or track output modified.',
    original_failure_log_sha256=sha(ROOT/'flow_SherrodBrown_train.log'),training_admitted=False)
(folder/'flow_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
