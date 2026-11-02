"""Read-only full provider and AVE audit. No generator or optimizer exists here."""
from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
import torch,cv2,soundfile as sf,librosa
R=Path('/dev/shm/may-tts-purified-training-20261020-v1');O=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):(O/n).write_text(json.dumps(v,indent=2,allow_nan=False))
def context(full,idx):
 out=np.zeros((8,1,512),dtype=np.float32)
 for j,a in enumerate(range(idx-4,idx+4)):
  if 0<=a<len(full):out[j]=full[a]
 return out
def main():
 assert not (O/'loader_ave_receipt.json').exists()
 spec=importlib.util.spec_from_file_location('frozen_training_common',R/'common.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);c.configure();torch.set_grad_enabled(False)
 pins=json.loads((R/'formal_admission.json').read_text())['source_sha256']
 for n,h in pins.items():assert sha(R/n)==h,n
 p=json.loads((R/'protocol.json').read_text());d=c.dataset('train',0,p);assert d.opt.att==2
 import nerf_triplane.provider as provider
 rows=d.source_frames;ids=np.array([x['img_id'] for x in rows]);aids=np.array([x['aud_id'] for x in rows]);cache=np.load(d.opt.verified_ave_cache);aligned=cache[aids,None,:]
 assert np.array_equal(aligned,d.auds.cpu().numpy())
 poses=np.stack([provider.nerf_matrix_to_ngp(np.array(x['transform_matrix'],np.float32),scale=d.scale,offset=d.offset) for x in rows]);assert np.array_equal(poses,d.poses.cpu().numpy())
 bs=np.load(R/'data/source/May/bs.npy')
 if d.opt.bs_area=='upper':bs=np.hstack((bs[:,:5],bs[:,8:10]))
 elif d.opt.bs_area=='single':bs=bs[:,[0,2,3,8]]
 elif d.opt.bs_area=='eye':bs=bs[:,8:10]
 assert not d.opt.smooth_eye and not d.opt.smooth_path
 assert np.array_equal(bs[ids].astype(np.float32),d.eye_area.cpu().numpy())
 assert all(Path(path).resolve()==(R/'data/source/May/gt_imgs'/f'{row["img_id"]}.jpg').resolve() for path,row in zip(d.images,rows))
 for i in range(len(rows)):
  actual=provider.get_audio_features(d.auds,2,i).cpu().numpy();assert np.array_equal(actual,context(aligned,i))
 cuts=np.where(np.diff(ids)==0)[0];sample=set(np.linspace(0,len(rows)-1,64).astype(int).tolist()+[0,1,2,3,len(rows)-4,len(rows)-3,len(rows)-2,len(rows)-1])
 for a in [358,648,954,1272,1622,1958,2274,2632,2970,3376,3724,2641,2685]:
  for j in [-1,0,1]:sample.add(int(np.argmin(np.abs(aids-(a+j)))))
 capture={};original=provider.get_rays
 def observe(*args,**kwargs):
  result=original(*args,**kwargs);capture['inds']=result['inds'].cpu().numpy()[0];return result
 checked=[];bg_error=0
 for i in sorted(sample):
  c.seed_domain(42,i,'diagnostic_loader');provider.get_rays=observe
  try:batch=d.collate([i])
  finally:provider.get_rays=original
  f=rows[i];source=R/'data/source/May';inds=capture['inds'];rgb=cv2.cvtColor(cv2.imread(str(source/'gt_imgs'/f"{f['img_id']}.jpg")),cv2.COLOR_BGR2RGB).astype(np.float32)/255
  assert np.array_equal(batch['images'][0].cpu().numpy(),rgb.reshape(-1,3)[inds])
  assert np.array_equal(batch['auds'].cpu().numpy(),context(aligned,i))
  assert np.array_equal(batch['poses'][0].cpu().numpy(),poses[i]);assert np.array_equal(batch['eye'][0].cpu().numpy(),bs[ids[i]].astype(np.float32))
  torso=cv2.cvtColor(cv2.imread(str(source/'torso_imgs'/f"{f['img_id']}.png"),cv2.IMREAD_UNCHANGED),cv2.COLOR_BGRA2RGBA).astype(np.float32)/255
  expected=torso[:,:,:3]*torso[:,:,3:]+d.bg_img.cpu().numpy()*(1-torso[:,:,3:]);error=float(np.max(np.abs(expected.reshape(-1,3)[inds]-batch['bg_color'][0].cpu().numpy())));assert error<1e-7;bg_error=max(bg_error,error)
  lm=np.loadtxt(source/'ori_imgs'/f"{f['img_id']}.lms");rect=[int(lm[:,1].min()),int(lm[:,1].max()),int(lm[:,0].min()),int(lm[:,0].max())];assert rect==d.face_rect[i]
  checked.append(dict(index=i,aud_id=f['aud_id'],source_id=f['img_id'],pixels_exact=True,attention_context_exact=True,pose_eye_torso_exact=True))
 # Recompute the entire AVE with the original checkpoint on the actual training WAV.
 from ave_cache_original import load_original_classes
 Encoder,Dataset=load_original_classes(R/'official');wav=R/'data/arms/train/gain_0/aud.wav';y,sr=sf.read(wav,dtype='float64');assert sr==16000 and np.array_equal(librosa.load(str(wav),sr=16000)[0],y.astype(np.float32))
 ds=Dataset(str(wav));model=Encoder().cuda().eval();model.requires_grad_(False)
 weights=R/'official/nerf_triplane/checkpoints/audio_visual_encoder.pth';assert sha(weights)=='1f75cdb5556938590c64221f6c47a8d20ab3e368d53c166023eedb12279db8c4'
 state=torch.load(weights,map_location='cpu',weights_only=True);model.load_state_dict({'audio_encoder.'+k:v for k,v in state.items()},strict=True)
 outputs=[]
 for start in range(0,len(ds),64):
  mel=torch.stack([ds[i] for i in range(start,min(start+64,len(ds)))]).cuda();outputs.append(model(mel).cpu())
 raw=torch.cat(outputs);actual=torch.cat([raw[:1].repeat(2,1),raw,raw[-1:].repeat(2,1)]).numpy();assert np.array_equal(actual,cache) and np.isfinite(actual).all();np.save(O/'recomputed_ave.npy',actual)
 windows=[]
 for a in sorted(set([int(aids[i]) for i in sample])):
  idx=a-2;start=int(80*(idx/25));independent=ds.orig_mel[start:start+16,:].T[None].astype(np.float32)
  assert np.array_equal(ds[idx].numpy(),independent);windows.append(dict(audio_frame=a,raw_feature_index=idx,mel_start=start,mel_count=16))
 save('loader_ave_receipt.json',dict(passed=True,all_rows=len(rows),all_feature_centers_exact=True,all_8frame_contexts_exact=True,all_poses_and_eye_exact=True,all_image_paths_source_owned=True,
  actual_collate_rows=len(checked),actual_checks=checked,torso_max_error=bg_error,AVE_entire_recomputed_bitwise_exact=True,recomputed_shape=list(actual.shape),recomputed_sha256=sha(O/'recomputed_ave.npy'),wave_sha256=sha(wav),
  mel_windows=windows,AVE_original_two_frame_padding=True,feature_context_only_supervision_edge_zero_padding_as_baseline=True,
  frozen_source_pins_fresh=True,generator_models_constructed=0,optimizer_updates=0,AVE_eval_models=1))
 print(json.dumps(dict(passed=True,all_rows=len(rows),actual_collate_rows=len(checked),AVE_full_exact=True,optimizer_updates=0)),flush=True)
if __name__=='__main__':main()
