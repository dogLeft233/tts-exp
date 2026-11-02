"""Original-PCM, source-owned-ROI evaluation of both normalization axes."""
from pathlib import Path
from types import SimpleNamespace
import argparse,sys,json,subprocess,hashlib,time
import cv2,numpy as np,torch,soundfile as sf
from torch_ema import ExponentialMovingAverage
from runtime import *
SYNC=Path('/home/wjj/tts-audio/tts-exp/third_party/syncnet_python')
def crops(images,track):
 out=[]
 for image,(s,x,y) in zip(images,track):
  cs=.4;bsi=int(s*(1+2*cs));p=np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)),constant_values=110);mx=x+bsi;my=y+bsi
  z=p[int(my-s):int(my+s*(1+2*cs)),int(mx-s*(1+cs)):int(mx+s*(1+cs))];out.append(cv2.resize(z,(224,224)))
 return np.stack(out)
def summary(matrix,indices):
 curve=matrix[indices].mean(0);v,i=torch.min(curve,0)
 return dict(sync_c=float(torch.median(curve)-v),sync_d=float(v),distance_d0=float(curve[15]),background_median=float(torch.median(curve)),av_offset=int(15-i),windows=len(indices),curve=curve.tolist())
def render(arm):
 from nerf_triplane.utils import Trainer
 rec=json.loads((ROOT/'training'/arm['id']/'receipt.json').read_text());assert rec['passed'] and rec['success']==12000 and rec['skip']==0 and rec['arm']==arm
 cp=ROOT/'training'/arm['id']/rec['checkpoint'];assert sha(cp)==rec['checkpoint_sha256']
 state=torch.load(cp,map_location='cpu',weights_only=False);assert state['step']==12000 and state['arm']==arm and state['initial_sha256']==rec['initial_sha256']==sha(ROOT/'initialization.pt')
 p=protocol();track=np.load(ROOT/'historical'/'shared_gt_track.npy');manifest=json.loads((DATA/'manifest.json').read_text())
 lo=next(s['supervision_start_frame'] for s in manifest['splits'] if s['name']=='test')
 # Original source-owned track is read back against its frozen historical body.
 pins=json.loads((ROOT/'historical'/'pins.json').read_text())
 for row in pins:assert sha(ROOT/'historical'/row['file'])==row['sha256']
 for norm in [True,False]:
  for gain in [-6,0,6]:
   cell=ROOT/'renders'/arm['id']/f'{"on" if norm else "off"}_g{gain:+d}';cell.mkdir(parents=True,exist_ok=False)
   d=dataset(arm['encoder'],'test',norm,gain);model=network(d.opt);model.load_state_dict(state['model'],strict=True);restore_metadata(model,state['metadata'])
   ema=ExponentialMovingAverage(model.parameters(),decay=.95);ema.load_state_dict(state['ema']);assert ema.num_updates==12;ema.copy_to();model.eval()
   owner=SimpleNamespace(model=model,opt=d.opt,criterion=torch.nn.L1Loss(reduction='none'))
   clips=[]
   for clip in p['test_clips']:
    start=clip['source_start_frame']-lo;folder=cell/clip['id'];folder.mkdir()
    audio=DATA/clip['score_audio_path'];assert sha(audio)==clip['score_audio_sha256']
    pcm,sr=sf.read(audio,dtype='int16');assert sr==16000 and pcm.shape==(160000,)
    path=folder/'original_pcm.mkv'
    ff=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','512x512','-r','25','-i','pipe:0','-i',str(audio),'-map','0:v:0','-map','1:a:0','-frames:v','250','-c:v','ffv1','-level','3','-pix_fmt','bgr0','-c:a','pcm_s16le',str(path)],stdin=subprocess.PIPE)
    frames=[];selected={}
    with torch.inference_mode():
     for i in range(start,start+250):
      result=Trainer.eval_step(owner,d.collate([i]));pred=result[0][0];truth=result[5][0];assert torch.isfinite(pred).all()
      x=pred.cpu().numpy().astype(np.float32);assert x.min()>=-1e-6 and x.max()<=1+1e-6
      rgb=np.rint(np.clip(x,0,1)*255).astype('uint8');ff.stdin.write(rgb.tobytes())
      frames.append(dict(index=i,source_id=d.source_frames[i]['img_id'],float_sha256=hashlib.sha256(x.tobytes()).hexdigest(),rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),**metrics(pred[None],truth[None],d.source_frames[i]['img_id'])))
      if i-start in [0,125,249]:
       selected[str(i-start)]=x;cv2.imwrite(str(folder/f'frame{i-start:03d}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    ff.stdin.close();assert ff.wait()==0
    np.savez_compressed(folder/'selected_float.npz',**selected)
    cap=cv2.VideoCapture(str(path));count=0
    while True:
     ok,bgr=cap.read()
     if not ok:break
     assert hashlib.sha256(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()==frames[count]['rgb_sha256'];count+=1
    cap.release();assert count==250
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(path),'-map','0:a:0','-f','s16le','pipe:1']);assert raw==pcm.tobytes()
    # Fixed baseline replay uses the same generator and exact provider indices.
    repeat=None
    if norm==arm['norm'] and gain==0:
     errors=[]
     with torch.inference_mode():
      for j in [0,125,249]:
       x=Trainer.eval_step(owner,d.collate([start+j]))[0][0].cpu().numpy().astype(np.float32);errors.append(float(np.max(abs(x-selected[str(j)]))));assert np.array_equal(x,selected[str(j)])
     repeat=dict(selected_frames=[0,125,249],maxabs=max(errors),float_exact=True)
    receipt=dict(arm=arm,test_normalize=norm,gain_db=gain,checkpoint_sha256=sha(cp),weights='EMA12',clip=clip,frames=frames,pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),audio_file_sha256=sha(audio),mux_sha256=sha(path),rgb_pcm_readback=True,repeat=repeat,mean={k:float(np.mean([r[k] for r in frames])) for k in ['MSE','L1','mouth_bbox_MSE','mouth_bbox_L1']})
    write(folder/'receipt.json',receipt);clips.append(dict(clip_id=clip['id'],receipt_sha256=sha(folder/'receipt.json')))
   write(cell/'receipt.json',dict(passed=True,arm=arm,test_normalize=norm,gain_db=gain,clips=clips))
   print('RENDERED',arm['id'],norm,gain,flush=True);del model,ema,d,owner;torch.cuda.empty_cache()
def score():
 import python_speech_features
 sys.path.insert(0,str(SYNC));from SyncNetModel import S
 from SyncNetInstance import calc_pdist
 os.chdir(SYNC);p=protocol();output=ROOT/'scores';output.mkdir(exist_ok=False)
 model=S(num_layers_in_fc_layers=1024).cuda().eval();weights=SYNC/'data/syncnet_v2.model';model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
 old=json.loads((ROOT/'historical'/'scores.json').read_text());assert sha(weights)==old['syncnet_weight_sha256']
 assert sha(SYNC/'SyncNetModel.py')==old['network_source_sha256'] and sha(SYNC/'SyncNetInstance.py')==old['instance_source_sha256']
 track=np.load(ROOT/'historical'/'shared_gt_track.npy');lo=4925;audios={}
 for clip in p['test_clips']:
  pcm,sr=sf.read(DATA/clip['score_audio_path'],dtype='int16');mfcc=python_speech_features.mfcc(pcm,sr).T
  audio=torch.from_numpy(mfcc[None,None].astype('float32'));parts=[]
  with torch.inference_mode():
   for i in range(0,245,20):parts.append(model.forward_aud(torch.cat([audio[:,:,:,j*4:j*4+20] for j in range(i,min(i+20,245))]).cuda()).cpu())
  audios[clip['id']]=torch.cat(parts);np.save(output/(clip['id']+'_audio.npy'),audios[clip['id']].numpy())
 result=dict(cells=[],protocol_sha256=sha(ROOT/'protocol.json'),syncnet_weight_sha256=sha(weights),precision='FP32',tf32=False,scoring_audio='original PCM16 fixed',roi='original GT shared historical SHA')
 for arm in p['arms']:
  for norm in [True,False]:
   for gain in [-6,0,6]:
    root=ROOT/'renders'/arm['id']/f'{"on" if norm else "off"}_g{gain:+d}';rows=[]
    for clip in p['test_clips']:
     src=root/clip['id'];rec=json.loads((src/'receipt.json').read_text());assert rec['rgb_pcm_readback']
     cap=cv2.VideoCapture(str(src/'original_pcm.mkv'));images=[]
     while True:
      ok,bgr=cap.read()
      if not ok:break
      assert hashlib.sha256(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).tobytes()).hexdigest()==rec['frames'][len(images)]['rgb_sha256'];images.append(bgr)
     cap.release();assert len(images)==250
     start=clip['source_start_frame']-lo;x=crops(images,track[start:start+250]);visual=torch.from_numpy(x).permute(3,0,1,2).unsqueeze(0).float();parts=[]
     with torch.inference_mode():
      for i in range(0,245,20):parts.append(model.forward_lip(torch.cat([visual[:,:,j:j+5] for j in range(i,min(i+20,245))]).cuda()).cpu())
     vf=torch.cat(parts);matrix=torch.stack(calc_pdist(vf,audios[clip['id']],vshift=15));assert torch.isfinite(matrix).all()
     folder=output/arm['id']/f'{"on" if norm else "off"}_g{gain:+d}'/clip['id'];folder.mkdir(parents=True)
     np.save(folder/'visual.npy',vf.numpy());np.save(folder/'matrix.npy',matrix.numpy())
     write(folder/'crop_frame_hashes.json',[hashlib.sha256(z.tobytes()).hexdigest() for z in x])
     if arm==p['arms'][0] and norm is True and gain==-6 and clip==p['test_clips'][0]:
      repeated=[]
      with torch.inference_mode():
       for i in range(0,245,20):repeated.append(model.forward_lip(torch.cat([visual[:,:,j:j+5] for j in range(i,min(i+20,245))]).cuda()).cpu())
      repeated=torch.cat(repeated);assert torch.equal(repeated,vf)
      repeated_matrix=torch.stack(calc_pdist(repeated,audios[clip['id']],vshift=15));assert torch.equal(repeated_matrix,matrix)
      write(folder/'repeat_receipt.json',dict(visual_exact=True,matrix_exact=True,maxabs=0))
     row=dict(clip_id=clip['id'],guard20=summary(matrix,np.arange(20,225)),full=summary(matrix,np.arange(245)),mouth_metrics=rec['mean'],matrix_sha256=sha(folder/'matrix.npy'),source_pcm_sha256=rec['pcm_sha256']);rows.append(row)
    result['cells'].append(dict(arm=arm,test_normalize=norm,gain_db=gain,clips=rows,mean_guard20={k:float(np.mean([r['guard20'][k] for r in rows])) for k in ['sync_c','sync_d','distance_d0','background_median']},mean_mouth_MSE=float(np.mean([r['mouth_metrics']['mouth_bbox_MSE'] for r in rows]))))
    write(output/'results.json',result);print('SCORED',arm['id'],norm,gain,flush=True)
 assert len(result['cells'])==48;write(output/'complete.json',dict(passed=True,cells=48,matrices=96))
if __name__=='__main__':
 configure()
 if sys.argv[1]=='render':render(next(a for a in protocol()['arms'] if a['id']==sys.argv[2]))
 else:assert sys.argv[1]=='score';score()
