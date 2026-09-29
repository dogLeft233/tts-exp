"""Old-runtime exact native80 replay, then frozen signed bottleneck shifts."""
from pathlib import Path
from contextlib import contextmanager
import argparse,fcntl,json,math,os,resource,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments import tts_fixed_generator_shift_holdout80_20260927 as core
from scripts.experiments import tts_fixed_generator_phone_stream_20260927 as stream
from scripts.experiments.tts_fixed_generator_shift_ops_20260927 import shift_table
OUT=core.OUT;PARENT=core.FIT;TMP=core.TMP;W2L=ROOT/'third_party/Wav2Lip';PY=core.PY
CONDITIONS=core.CONDS;read=core.read;sha=core.sha;ah=core.ah;allocated=core.allocated
STAGES=['identity','baseline','calibration','evaluation']
def config():return read(OUT/'protocol.json')['gpu_config']
def limits(extra=0):
 z=core.limits(extra);elapsed=0.;now=time.time()
 for stage in STAGES:
  start=OUT/'gpu_runtime'/('timer_start_'+stage+'.json');end=OUT/'gpu_runtime'/('worker_exit_'+stage+'.json')
  if start.exists():elapsed+=max(0.,(read(end)['created_epoch'] if end.exists() else now)-read(start)['created_epoch'])
 assert elapsed<=5400,('GPU cumulative cap',elapsed)
 return {**z,'GPU_cumulative_wall_seconds':elapsed}
def write(p,z):
 limits();core.write(p,z);limits()
def save(p,a):
 assert a.dtype==np.float32 and a.ndim==2 and a.shape[1] in [512,1024] and np.isfinite(a).all()
 p=Path(p);bound=math.ceil((a.nbytes+128)/4096)*4096;limits(bound);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:np.save(f,a,allow_pickle=False)
 assert p.stat().st_blocks*512<=bound and np.array_equal(np.load(p),a);limits()
 return {'path':str(p),'sha256':sha(p),'raw_sha256':ah(a),'shape':list(a.shape),'dtype':'float32'}
def protocol(stage):
 p,_=core.locked();c=p['gpu_config'];assert core.runtime()==c['runtime'] and Path(sys.executable).resolve()==PY.resolve()
 for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ.get(k)=='1'
 seal=read(OUT/'input_seal.json');assert seal['status']=='PASS' and seal['protocol_sha256']==sha(OUT/'protocol.json')
 for f,h in seal['files'].items():assert sha(f)==h
 rv=read(OUT/'input_reviewer_pass.json');assert rv['status']=='PASS' and rv['input_seal_sha256']==sha(OUT/'input_seal.json') and sha(rv['receipt'])==rv['receipt_sha256']
 assert read(OUT/'baseline_validation.json')['status']=='PASS'
 from scripts.experiments.tts_native_gain_attribution import config as sc
 assert Path(sc.FFMPEG).resolve()==Path(c['ffmpeg']).resolve() and Path(sc.FFPROBE).resolve()==Path(c['ffprobe']).resolve()
 required={'baseline':'identity','calibration':'baseline','evaluation':'calibration'}
 if stage in required:assert read(OUT/(required[stage]+'_gate.json'))['passed']
 if stage in ['calibration','evaluation']:
  for f,h in read(OUT/'baseline_gate.json')['files'].items():assert sha(f)==h,f
 limits();return p,read(OUT/'input_rows.json')
def frontend(row):
 import soundfile as sf,python_speech_features
 a=row['arms']['N']
 for key in ['features','frontend','waveform','metadata']:assert sha(a[key+'_path'])==a[key+'_sha256']
 assert sha(a['TextGrid'])==a['TextGrid_sha256']
 x,sr=sf.read(a['waveform_path'],dtype='float64');assert sr==16000 and len(x)==a['samples']
 sys.path.insert(0,str(W2L));import audio
 mel=audio.melspectrogram(x.astype(np.float32)).astype(np.float32)
 mfcc=np.asarray(python_speech_features.mfcc(x*32768,16000),dtype=np.float64)
 n=min(len(x)//640-5,(len(mfcc)-20)//4+1);windows=np.stack([mfcc[i*4:i*4+20].T for i in range(n)]).astype(np.float32)
 with np.load(a['frontend_path']) as f:assert np.array_equal(mel,f['mel']) and np.array_equal(windows,f['windows'])
 assert np.isfinite(mel).all() and np.isfinite(windows).all()
 from scripts.experiments.static_image_bridge.render_worker import chunk_mels
 chunks=chunk_mels(mel,25);assert core.starts(mel.shape[1])==a['mel_starts'] and len(chunks)==a['frames']
 for chunk,start in zip(chunks,a['mel_starts']):assert np.array_equal(chunk,mel[:,start:start+16])
 parent=np.load(a['features_path']);assert parent['visual'].shape==(len(chunks)-4,1024)
 return chunks,windows,parent,{'mel_sha256':ah(mel),'windows_sha256':ah(windows),'chunks_sha256':ah(np.asarray(chunks,dtype=np.float32)),'starts':a['mel_starts'],'MFCC_starts':[i*4 for i in range(n)],'parent_features_sha256':a['features_sha256'],'L':a['L']}
def native_z(model,chunks):
 import torch
 vals=[]
 with torch.inference_mode():
  for start in range(0,len(chunks),32):
   gpucheck();x=torch.from_numpy(np.asarray(chunks[start:start+32],np.float32)[:,None]).cuda();y=model.audio_encoder(x).cpu().numpy()[:,:,0,0].copy();assert y.dtype==np.float32;vals.append(y)
 z=np.concatenate(vals);assert z.shape==(len(chunks),512) and np.isfinite(z).all() and (z>=0).all();return z
def audio_forward(engine,windows):
 torch=engine._torch;outputs=[]
 with torch.inference_mode():
  for start in range(0,len(windows),32):
   gpucheck();outputs.append(engine.network.forward_aud(torch.from_numpy(windows[start:start+32])[:,None].cuda()).cpu().numpy().astype(np.float32))
 return np.concatenate(outputs)


def processes():
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip();rows=[]
 for line in out.splitlines():
  if not line.strip():continue
  fields=[q.strip() for q in line.split(',')];rows.append({'pid':int(fields[0]),'name':fields[1],'memory_MiB':int(fields[2])})
 return rows

def gpucheck(start=False):
 z=limits();jobs=processes();assert all(q['pid']==os.getpid() for q in jobs),('GPU_OCCUPIED',jobs)
 used,util=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
 if start:assert used<=128 and util<=10,('GPU_OCCUPIED',used,util)
 return {**z,'compute':jobs,'gpu_MiB':used,'gpu_util':util}

@contextmanager
def lease(stage):
 with open('/tmp/tts-exp-gpu.lock','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   snapshots=[]
   for i in range(3):
    snapshots.append(gpucheck(True))
    if i<2:time.sleep(5)
   write(OUT/'gpu_runtime'/('resources_'+stage+'.json'),snapshots);yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)

def one(model,engine,chunks,frame,image_tensor,c,row,arm,condition,expected,replacement,mode,*,full=False):
 path=TMP/(row['id']+'_'+arm+'_'+condition+'.avi');limits();remaining=c['tmp_cap_bytes']-allocated(TMP)-2**20;cap=(remaining//4096)*4096;assert cap>4096
 video,native,applied=stream.render(model,chunks,frame,image_tensor,c['image']['generation_box'],c['image']['score_box']['box'],path,expected,mode=mode,replacement=replacement,resource_check=gpucheck,file_cap=cap)
 v,vm=stream.visual(engine,path,gpucheck);dec=stream.decode(path,c['ffmpeg'],c['ffprobe'],gpucheck)
 assert video['frames']==dec['frames']==vm['frame_count']==row['arms'][arm]['frames'] and video['pixel_sha256']==dec['pixel_sha256']
 if full:
  fv,_=stream.visual(engine,path,gpucheck,full=True);assert np.array_equal(fv,v)
 assert sha(path)==video['sha256']
 return v,{'video_at_creation':video,'decode':dec,'visual_metadata':vm,'native_z_exact':True,'full_stream_exact':True if full else None,'retained_video':False}

def live_gate(stage):
 files={str(p):sha(p) for p in sorted(TMP.rglob('*')) if p.is_file()};assert files
 write(OUT/(stage+'_live.json'),{'created_epoch':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'files':files,'no_following_stage_before_audit':True})
 print(stage.upper()+'_LIVE_WAIT',str(OUT/(stage+'_live.json')),flush=True);start=time.time();binding=OUT/(stage+'_media_audit.json')
 while not binding.exists():
  limits();assert time.time()-start<1800,'external live audit timeout; preserve scene';time.sleep(2)
 b=read(binding);assert b['status']=='PASS' and sha(b['receipt'])==b['receipt_sha256'];receipt=read(b['receipt'])
 assert receipt['status']=='PASS' and receipt['protocol_sha256']==sha(OUT/'protocol.json') and receipt['media_hashes']==files
 for path,h in files.items():assert Path(path).is_relative_to(TMP) and sha(path)==h
 for path in files:Path(path).unlink()
 write(OUT/(stage+'_release.json'),{'created_epoch':time.time(),'receipt':b,'released':files,'only_own_new_temporary_media':True});limits()
 return b

def release(meta):
 path=Path(meta['video_at_creation']['path']);assert path.is_relative_to(TMP) and sha(path)==meta['video_at_creation']['sha256'];path.unlink();limits()
def run(stage):
 p,rows=protocol(stage);c=p['gpu_config'];first=rows[0];assert len(rows)==80 and len({r['speaker'] for r in rows})==40
 write(OUT/'gpu_runtime'/('timer_start_'+stage+'.json'),{'created_epoch':time.time(),'stage':stage,'cap_seconds_cumulative':5400,'includes_live_wait':True})
 import cv2,torch
 from scripts.experiments.masked_tts_tfg_probe.direct_mel import _load_model
 from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
 cv2.setNumThreads(1);torch.set_num_threads(2);torch.manual_seed(20260926);torch.cuda.manual_seed_all(20260926);np.random.seed(20260926);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 TMP.mkdir(exist_ok=True);assert not any(x.is_file() for x in TMP.rglob('*'))
 fitmeta=read(PARENT/'fit.json');fit=np.load(PARENT/'fit.npz');fold=fitmeta['folds'].index('S0912');assert fitmeta['sources']==['N','T']
 mu=[stream.template_table(fit['raw_mu'],fit['speaker_counts'],fitmeta['labels'],fold,i) for i in [0,1]]
 targets=rows[:2] if stage=='identity' else [first] if stage=='calibration' else rows;completed=[]
 with lease(stage):
  sys.path.insert(0,str(W2L));model=_load_model(Path(c['checkpoint']),'cuda');engine=SyncNetEngine(batch_size=32,device='cuda')
  write(OUT/'gpu_runtime'/('loaded_'+stage+'.json'),{'created_epoch':time.time(),'runtime':core.runtime(),'wav2lip_state_sha256':stream.statehash(model),'syncnet_state_sha256':stream.statehash(engine.network),'seed':20260926,'torch_threads':2,'batch':32})
  frame=cv2.imread(c['image']['path']);assert frame is not None and sha(c['image']['path'])==c['image']['sha256'];x1,y1,x2,y2=c['image']['generation_box'];face=cv2.resize(frame[y1:y2,x1:x2],(96,96));masked=face.copy();masked[48:]=0
  image_tensor=torch.from_numpy(np.concatenate([masked,face],2).transpose(2,0,1).astype(np.float32)/255).unsqueeze(0).cuda()
  try:
   for row in targets:
    chunks,windows,parent,front=frontend(row);sid=row['id'];zp=OUT/'z'/(sid+'.npy')
    if stage in ['identity','baseline']:
     z=native_z(model,chunks)
     if zp.exists():assert np.array_equal(z,np.load(zp));zd={'path':str(zp),'sha256':sha(zp),'raw_sha256':ah(z),'shape':list(z.shape),'dtype':'float32'}
     else:zd=save(zp,z)
     aa=audio_forward(engine,windows);assert np.array_equal(aa,parent['audio'])
     checks={};pm=read(row['arms']['N']['metadata_path'])
     for mode in (['none','noop','cached'] if stage=='identity' else ['none']):
      v,m=one(model,engine,chunks,frame,image_tensor,c,row,'N',stage+'_'+mode,z,z.copy() if mode=='cached' else None,mode,full=stage=='identity')
      assert np.array_equal(v,parent['visual']) and m['video_at_creation']['pixel_sha256']==pm['video']['pixel_sha256']
      checks[mode]={'V_exact':True,'A_exact':True,'pixel_exact':True,'V_raw_sha256':ah(v),'A_raw_sha256':ah(aa),**m}
      if stage=='baseline':release(m)
     record={'id':sid,'speaker':row['speaker'],'passed':True,'z':zd,'frontend':front,'checks':checks,'parentbindings':row['arms']['N'],'historical_z_available':False,'new_z_repeat_native_forward_exact':True}
     write(OUT/('identity' if stage=='identity' else 'baseline')/(sid+'.json'),record);completed.append(sid)
    else:
     baseline=read(OUT/'baseline'/(sid+'.json'));assert baseline['passed'] and sha(zp)==baseline['z']['sha256'];z=np.load(zp)
     for condition in CONDITIONS:
      kind,sign=condition.split('_');own,other=(mu[0],mu[1]) if sign=='plus' else (mu[1],mu[0])
      replacement,mix=shift_table(z,row['arms']['N']['phone_labels'],row['arms']['N']['speech_mask'],own[0],own[1],other[0],other[1],kind)
      v,m=one(model,engine,chunks,frame,image_tensor,c,row,'N',stage+'_'+condition,z,replacement,'processed',full=stage=='calibration')
      vp=OUT/'gpu_controls/calibration'/(condition+'.npy') if stage=='calibration' else OUT/'visual'/sid/'N'/(condition+'.npy');vf=save(vp,v)
      if stage=='calibration':
       repeat,rm=one(model,engine,chunks,frame,image_tensor,c,row,'N',stage+'_'+condition+'_repeat',z,replacement,'processed',full=True)
       assert np.array_equal(v,repeat) and m['video_at_creation']['pixel_sha256']==rm['video_at_creation']['pixel_sha256'];m['repeat']={'V_exact':True,'pixel_exact':True,'metadata':rm}
      elif sid==first['id']:
       assert np.array_equal(v,np.load(OUT/'gpu_controls/calibration'/(condition+'.npy')))
      record={'id':sid,'speaker':row['speaker'],'arm':'N','condition':condition,'V':vf,'frontend':front,'parentbindings':row['arms']['N'],'native_z':baseline['z'],'fit_sha256':sha(PARENT/'fit.npz'),'fit_metadata_sha256':sha(PARENT/'fit.json'),'fold':'S0912','mix':mix,**m}
      dest=OUT/'gpu_controls/calibration'/(condition+'.json') if stage=='calibration' else OUT/'metadata'/sid/'N'/(condition+'.json');write(dest,record);completed.append(sid+'/'+condition)
      if stage=='evaluation':release(m)
    print('produced',stage,sid,flush=True)
   if stage in ['identity','calibration']:
    receipt=live_gate(stage);write(OUT/(stage+'_gate.json'),{'passed':True,'created_epoch':time.time(),'external_media_audit_passed':True,'receipt':receipt,'completed':completed,'no_effect_scores':True})
   if stage=='baseline':
    assert len(completed)==80;files={str(f):sha(f) for tree in ['z','baseline'] for f in (OUT/tree).rglob('*') if f.is_file()}
    write(OUT/'baseline_gate.json',{'passed':True,'created_epoch':time.time(),'completed':completed,'files':files,'all80_old_pixels_A_V_exact':True,'new_z_not_historical_z_claim':True})
  finally:engine.close();del model
 write(OUT/'gpu_runtime'/('worker_exit_'+stage+'.json'),{'created_epoch':time.time(),'stage_complete':True,'completed':len(completed),'lease_released':True,'temporary_files':len([p for p in TMP.rglob('*') if p.is_file()]),'external_compute_empty_check_required':True})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=STAGES);run(a.parse_args().stage)
