"""Deterministic streaming Wav2Lip bottleneck replacement; no execution at import."""
from pathlib import Path
import hashlib,json,os,resource,signal,subprocess
import numpy as np

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for data in iter(lambda:f.read(1<<20),b''):h.update(data)
 return h.hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def statehash(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():
  h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def template_table(raw_mu,speaker_counts,labels,fold_index,source_index):
 """Float64 shrinkage first; one float32 cast per resulting template."""
 raw=np.asarray(raw_mu);counts=np.asarray(speaker_counts)
 assert raw.dtype==np.float64 and raw.ndim==4 and raw.shape[2]==len(labels)+1 and raw.shape[3]==512
 assert counts.shape==raw.shape[:1]+(len(labels),) and np.isfinite(raw).all() and (raw>=0).all()
 global64=raw[fold_index,source_index,-1];result={}
 for i,label in enumerate(labels):
  k=int(counts[fold_index,i]);assert k>=0;w=np.float64(k)/np.float64(k+4)
  if k==0:continue
  value=w*raw[fold_index,source_index,i]+(np.float64(1)-w)*global64
  result[label]=value.astype(np.float32)
 return result,global64.astype(np.float32)
def replacement_table(z,phone_labels,speech_mask,phone_mu,global_mu,kind):
 assert z.dtype==np.float32 and z.shape==(len(phone_labels),512) and len(speech_mask)==len(z)
 assert np.isfinite(z).all() and (z>=0).all() and kind in ('phone','global')
 out=z.copy();mask=np.asarray(speech_mask,dtype=bool);fallback=0
 for i in np.flatnonzero(mask):
  label=phone_labels[i];assert isinstance(label,str) and label
  if kind=='phone' and label in phone_mu:mu=phone_mu[label]
  else:mu=global_mu;fallback+=int(kind=='phone')
  assert mu.dtype==np.float32 and mu.shape==(512,) and np.isfinite(mu).all() and (mu>=0).all()
  left=np.multiply(z[i],np.float32(.5),dtype=np.float32)
  right=np.multiply(mu,np.float32(.5),dtype=np.float32)
  out[i]=np.add(left,right,dtype=np.float32)
 assert np.array_equal(out[~mask],z[~mask]) and np.isfinite(out).all() and (out>=0).all()
 return out,{'speech_frames':int(mask.sum()),'identity_frames':int((~mask).sum()),'phone_global_fallback_frames':fallback,'input_z_raw_sha256':ah(z),'replacement_z_raw_sha256':ah(out),'replacement_min':float(out.min()),'replacement_max':float(out.max()),'mixed_float32':'multiply(.5,z) then multiply(.5,template), then add; inactive direct-copy'}
def render(model,chunks,frame,image_tensor,box,roi,path,expected_z,*,mode,replacement=None,resource_check,file_cap):
 """mode none/noop/cached/processed. Actual native z must match parent cache."""
 import cv2,torch
 from scripts.experiments.static_image_bridge.score_worker import crop_zero_padded
 path=Path(path);assert not path.exists() and mode in ('none','noop','cached','processed')
 n=len(chunks);assert expected_z.dtype==np.float32 and expected_z.shape==(n,512)
 assert np.isfinite(expected_z).all() and (expected_z>=0).all()
 if mode in ('cached','processed'):
  assert replacement is not None and replacement.dtype==np.float32 and replacement.shape==expected_z.shape
  assert np.isfinite(replacement).all() and (replacement>=0).all()
  if mode=='cached':assert np.array_equal(replacement,expected_z)
 else:assert replacement is None
 file_cap=int(file_cap);assert file_cap>=4096;resource_check()
 oldlimit=resource.getrlimit(resource.RLIMIT_FSIZE);oldsignal=signal.getsignal(signal.SIGXFSZ)
 assert oldlimit[1]==resource.RLIM_INFINITY or file_cap<=oldlimit[1]
 def too_large(signum,stack):raise RuntimeError('temporary FFV1 file cap exceeded; preserve scene')
 resource.setrlimit(resource.RLIMIT_FSIZE,(file_cap,oldlimit[1]));signal.signal(signal.SIGXFSZ,too_large)
 writer=None;handle=None;digest=hashlib.sha256();native=[];used=[];batch=[0,0];calls=0;x1,y1,x2,y2=box
 def check_native(tensor):
  assert tensor.dtype==torch.float32 and tuple(tensor.shape)==(batch[1],512,1,1)
  value=tensor.detach().cpu().numpy()[:,:,0,0].copy()
  assert np.array_equal(value,expected_z[batch[0]:batch[0]+batch[1]]),'native Wav2Lip z not exact parent cache'
  native.append(value);return value
 def hook(module,inputs,output):
  nonlocal calls
  calls+=1;check_native(output)
  if mode=='noop':used.append(native[-1]);return None
  value=replacement[batch[0]:batch[0]+batch[1]];used.append(value.copy())
  result=torch.from_numpy(np.ascontiguousarray(value[:,:,None,None])).to(device=output.device,dtype=output.dtype)
  assert result.shape==output.shape and result.dtype==output.dtype and result.device==output.device
  return result
 try:
  path.parent.mkdir(parents=True,exist_ok=True)
  writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),25,(224,224));assert writer.isOpened()
  if mode!='none':handle=model.audio_encoder.register_forward_hook(hook)
  for start in range(0,n,32):
   resource_check();mt=torch.from_numpy(np.asarray(chunks[start:start+32],dtype=np.float32)[:,None]).to(image_tensor.device)
   batch[:]=[start,len(mt)]
   with torch.inference_mode():
    if mode=='none':check_native(model.audio_encoder(mt));used.append(native[-1])
    before=calls;pred=model(mt,image_tensor.expand(len(mt),-1,-1,-1)).cpu().numpy().transpose(0,2,3,1)*255
    if mode!='none':assert calls==before+1
   for fo in pred:
    full=frame.copy();full[y1:y2,x1:x2]=cv2.resize(fo.astype(np.uint8),(x2-x1,y2-y1))
    cropped=crop_zero_padded(full,roi);assert cropped.shape==(224,224,3) and cropped.dtype==np.uint8
    writer.write(cropped);digest.update(cropped.tobytes())
   resource_check()
 finally:
  try:
   if handle is not None:handle.remove()
   if writer is not None:writer.release()
  finally:resource.setrlimit(resource.RLIMIT_FSIZE,oldlimit);signal.signal(signal.SIGXFSZ,oldsignal)
 assert path.stat().st_size<=file_cap;resource_check()
 native=np.concatenate(native);applied=np.concatenate(used);assert np.array_equal(native,expected_z)
 if mode in ('cached','processed'):assert np.array_equal(applied,replacement)
 else:assert np.array_equal(applied,expected_z)
 return {'path':str(path),'sha256':sha(path),'pixel_sha256':digest.hexdigest(),'frames':n,'file_cap':file_cap,'hook_mode':mode,'native_z_exact':True,'native_z_raw_sha256':ah(native),'applied_z_raw_sha256':ah(applied),'hook_calls':calls,'none_z_probe':'separate same-batch audio_encoder call before unchanged model forward' if mode=='none' else None,'native_range':[float(native.min()),float(native.max())],'applied_range':[float(applied.min()),float(applied.max())]},native,applied

def decode(video,ffmpeg,ffprobe,resource_check):
 resource_check();proc=subprocess.Popen([str(ffmpeg),'-v','error','-threads','1','-i',str(video),'-threads','1','-f','rawvideo','-pix_fmt','bgr24','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 h=hashlib.sha256();n=0
 try:
  while True:
   q=proc.stdout.read(224*224*3)
   if not q:break
   assert len(q)==224*224*3;h.update(q);n+=1
   if n%32==0:resource_check()
  err=proc.stderr.read();assert proc.wait()==0,err[-500:]
 except BaseException:
  if proc.poll() is None:proc.kill()
  proc.wait();raise
 frames=json.loads(subprocess.check_output([str(ffprobe),'-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=pts,best_effort_timestamp_time','-of','json',str(video)],text=True))['frames']
 assert len(frames)==n;t=np.array([float(x['best_effort_timestamp_time']) for x in frames])
 assert np.allclose(t,np.arange(n)/25,atol=1e-9,rtol=0);resource_check()
 return {'pixel_sha256':h.hexdigest(),'frames':n,'PTS':[int(x['pts']) for x in frames],'time_seconds_sha256':ah(t),'decode':'producer separate FFmpeg BGR24 and ffprobe; not external-auditor claim'}

def visual(engine,path,resource_check,*,full=False):
 """Same official JPEG input and batch32; full path is identity diagnostic only."""
 torch=engine._torch;assert engine.batch_size==32
 if full:
  frames=list(engine._stream_mjpeg(path));out=[];resource_check()
  with torch.inference_mode():
   for start in range(0,len(frames)-4,32):
    resource_check();windows=[frames[j:j+5] for j in range(start,min(start+32,len(frames)-4))]
    out.append(engine.network.forward_lip(torch.from_numpy(engine._visual_batch(windows)).to(engine.device)).cpu().numpy().astype(np.float32));resource_check()
  result=np.concatenate(out);meta={'frame_count':len(frames),'full_reference':True}
 else:
  original=engine._visual_batch
  def guarded(windows):resource_check();return original(windows)
  engine._visual_batch=guarded
  try:result,meta=engine.extract_visual(path)
  finally:engine._visual_batch=original
  resource_check()
 assert result.dtype==np.float32 and result.shape==(meta['frame_count']-4,1024) and np.isfinite(result).all()
 return result,meta
