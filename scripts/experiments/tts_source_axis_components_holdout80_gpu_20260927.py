"""Actual generator axis components with strict post-displacement projection."""
from pathlib import Path
from contextlib import contextmanager
import argparse,fcntl,json,math,os,resource,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments import tts_source_axis_components_holdout80_core_20260927 as core
from scripts.experiments import tts_fixed_generator_phone_stream_20260927 as stream
from scripts.experiments.tts_fixed_generator_shift_ops_20260927 import shift_table
OUT=core.OUT;PARENT=core.PARENT;TMP=core.TMP;W2L=ROOT/'third_party/Wav2Lip';PY=core.PY
CONDITIONS=core.CONDS;read=core.read;sha=core.sha;ah=core.ah;allocated=core.allocated
STAGES=['calibration','evaluation']
def config():return read(OUT/'protocol.json')['gpu_config']
def limits(extra=0):
 z=core.limits(extra);elapsed=0.;now=time.time()
 for stage in STAGES:
  start=OUT/'gpu_runtime'/('timer_start_'+stage+'.json');end=OUT/'gpu_runtime'/('worker_exit_'+stage+'.json')
  if start.exists():elapsed+=max(0.,(read(end)['created_epoch'] if end.exists() else now)-read(start)['created_epoch'])
 assert elapsed<=3600,('GPU cumulative cap',elapsed)
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
 p,rows=core.locked();c=p['gpu_config'];assert core.runtime()==c['runtime'] and Path(sys.executable).resolve()==PY.resolve()
 for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ.get(k)=='1'
 assert read(OUT/'baseline_validation.json')['status']=='PASS'
 from scripts.experiments.tts_native_gain_attribution import config as sc
 assert Path(sc.FFMPEG).resolve()==Path(c['ffmpeg']).resolve() and Path(sc.FFPROBE).resolve()==Path(c['ffprobe']).resolve()
 if stage=='evaluation':assert read(OUT/'calibration_gate.json')['passed']
 limits();return p,rows
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
def gained(model,row,beta,repeat=False):
 import soundfile as sf
 sys.path.insert(0,str(W2L));import audio
 from scripts.experiments.static_image_bridge.render_worker import chunk_mels
 a=row['arms']['N'];x,sr=sf.read(a['waveform_path'],dtype='float32');assert sr==16000 and len(x)==a['samples']
 y,desc=core.ops.gain_waveform(x,beta);mel=audio.melspectrogram(y).astype(np.float32);chunks=chunk_mels(mel,25)
 assert np.isfinite(mel).all() and core.starts(mel.shape[1])==a['mel_starts'] and len(chunks)==a['frames']
 for start,chunk in zip(a['mel_starts'],chunks):assert np.array_equal(chunk,mel[:,start:start+16])
 z=native_z(model,chunks)
 if repeat:assert np.array_equal(z,native_z(model,chunks))
 target=OUT/'gain_z'/(row['id']+'.npy')
 if target.exists():
  assert np.array_equal(z,np.load(target));f={'path':str(target),'sha256':sha(target),'raw_sha256':ah(z),'shape':list(z.shape),'dtype':'float32'}
 else:f=save(target,z)
 ideal=desc.pop('ideal_wave64');desc={k:float(v) if isinstance(v,np.floating) else v for k,v in desc.items()}
 desc.update(beta64=float(beta),waveform_raw_sha256=ah(y),waveform_shape=list(y.shape),waveform_dtype='float32',original_raw_sha256=ah(x),mel_raw_sha256=ah(mel),mel_shape=list(mel.shape),chunks_raw_sha256=ah(np.asarray(chunks,np.float32)),mel_starts=a['mel_starts'],rounding_max=float(np.max(abs(y.astype(np.float64)-ideal))),no_explicit_clip=True,repeat_encoder_exact=True if repeat else None)
 return z,f,desc

def run(stage):
 p,rows=protocol(stage);c=p['gpu_config'];first=rows[0];assert len(rows)==80 and len({r['speaker'] for r in rows})==40
 write(OUT/'gpu_runtime'/('timer_start_'+stage+'.json'),{'created_epoch':time.time(),'stage':stage,'cap_seconds_cumulative':3600,'includes_live_wait':True})
 import cv2,torch
 from scripts.experiments.masked_tts_tfg_probe.direct_mel import _load_model
 from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
 cv2.setNumThreads(1);torch.set_num_threads(2);torch.manual_seed(20260926);torch.cuda.manual_seed_all(20260926);np.random.seed(20260926);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 TMP.mkdir(exist_ok=True);assert not any(x.is_file() for x in TMP.rglob('*'))
 with np.load(core.AXIS/'vectors.npz') as a:vectors={k:a[k].copy() for k in a.files}
 u=vectors['u_level'];assert np.linalg.norm(u)>1e-12
 targets=[first] if stage=='calibration' else rows;completed=[]
 with lease(stage):
  sys.path.insert(0,str(W2L));model=_load_model(Path(c['checkpoint']),'cuda');engine=SyncNetEngine(batch_size=32,device='cuda')
  write(OUT/'gpu_runtime'/('loaded_'+stage+'.json'),{'created_epoch':time.time(),'runtime':core.runtime(),'wav2lip_state_sha256':stream.statehash(model),'syncnet_state_sha256':stream.statehash(engine.network),'seed':20260926,'torch_threads':2,'batch':32})
  frame=cv2.imread(c['image']['path']);assert frame is not None and sha(c['image']['path'])==c['image']['sha256'];x1,y1,x2,y2=c['image']['generation_box'];face=cv2.resize(frame[y1:y2,x1:x2],(96,96));masked=face.copy();masked[48:]=0
  image_tensor=torch.from_numpy(np.concatenate([masked,face],2).transpose(2,0,1).astype(np.float32)/255).unsqueeze(0).cuda()
  try:
   for row in targets:
    chunks,windows,parent,front=frontend(row);sid=row['id'];baseline=read(PARENT/'baseline'/(sid+'.json'));zp=PARENT/'z'/(sid+'.npy')
    assert sha(zp)==baseline['z']['sha256'];z=np.load(zp);mask=np.asarray(row['arms']['N']['speech_mask'],bool)
    # Historical full is reconstructed exactly only for input diagnostics / first-cal replay.
    full=z.copy();half=np.multiply(vectors['actual_delta32'],np.float32(.5),dtype=np.float32);before=np.add(z[mask],half,dtype=np.float32);full[mask]=np.maximum(before,np.float32(0))
    fm=read(PARENT/'metadata'/sid/'N/global_plus.json');assert ah(full)==fm['mix']['replacement_z_raw_sha256']
    olddose={'baseline':core.dose(z,z,mask,u),'global_plus':core.dose(z,full,mask,u,before)}
    if stage=='calibration':
     checks={}
     for name,replacement,mode,oldmeta,oldv in [('baseline',None,'none',read(row['arms']['N']['metadata_path'])['video'],parent['visual']),('global_plus',full,'processed',fm['video_at_creation'],np.load(fm['V']['path']))]:
      vv,mm=one(model,engine,chunks,frame,image_tensor,c,row,'N','calibration_'+name,z,replacement,mode,full=True)
      assert np.array_equal(vv,oldv) and mm['video_at_creation']['pixel_sha256']==oldmeta['pixel_sha256']
      checks[name]={'V_exact':True,'pixel_exact':True,'historical_video_not_retained':True,'V_raw_sha256':ah(vv),**mm}
     write(OUT/'gpu_controls/calibration/old_cell_replay.json',{'passed':True,'id':sid,'checks':checks})
    gz,gf,gd=gained(model,row,p['gain_beta64'],repeat=stage=='calibration')
    for condition in CONDITIONS:
     special={};pre=None
     if condition in ['parallel','orthogonal']:
      replacement,detail=core.ops.component_replacement(z,mask,vectors[condition]);pre=detail['pre_relu_speech'];special={'part32_raw_sha256':ah(detail['part32']),'half32_raw_sha256':ah(detail['half32'])}
     elif condition=='masked_actual_gain':replacement=core.ops.masked_gain_replacement(z,mask,gz);special={'gain_z':gf,'gain_frontend':gd}
     elif condition=='strict_orth':
      replacement,ds=core.strict.strict_replacement(z,mask,vectors['orthogonal'],u);special={'strict':core.strict_summary(ds)};pre=z[mask].astype(np.float64)+np.float64(.5)*vectors['orthogonal']
     else:raise AssertionError(condition)
     mix=core.dose(z,replacement,mask,u,pre);mix.update(special)
     v,m=one(model,engine,chunks,frame,image_tensor,c,row,'N',stage+'_'+condition,z,replacement,'processed',full=stage=='calibration')
     vp=OUT/'gpu_controls/calibration'/(condition+'.npy') if stage=='calibration' else OUT/'visual'/sid/'N'/(condition+'.npy');vf=save(vp,v)
     if stage=='calibration':
      repeat,rm=one(model,engine,chunks,frame,image_tensor,c,row,'N',stage+'_'+condition+'_repeat',z,replacement,'processed',full=True)
      assert np.array_equal(v,repeat) and m['video_at_creation']['pixel_sha256']==rm['video_at_creation']['pixel_sha256'];m['repeat']={'V_exact':True,'pixel_exact':True,'metadata':rm}
     elif sid==first['id']:assert np.array_equal(v,np.load(OUT/'gpu_controls/calibration'/(condition+'.npy')))
     record={'id':sid,'speaker':row['speaker'],'arm':'N','condition':condition,'V':vf,'frontend':front,'parentbindings':row['arms']['N'],'native_z':baseline['z'],'axis_vectors_sha256':sha(core.AXIS/'vectors.npz'),'mix':mix,**m}
     dest=OUT/'gpu_controls/calibration'/(condition+'.json') if stage=='calibration' else OUT/'metadata'/sid/'N'/(condition+'.json');write(dest,record);completed.append(sid+'/'+condition)
     if stage=='evaluation':release(m)
    if stage=='evaluation':write(OUT/'dose_reference'/(sid+'.json'),{'id':sid,'speaker':row['speaker'],'old_cells':olddose,'gain_z':gf,'gain_frontend':gd})
    parent.close();print('produced',stage,sid,flush=True)
   if stage=='calibration':
    receipt=live_gate(stage);write(OUT/'calibration_gate.json',{'passed':True,'created_epoch':time.time(),'external_media_audit_passed':True,'receipt':receipt,'completed':completed,'no_effect_scores':True})
  finally:engine.close();del model
 write(OUT/'gpu_runtime'/('worker_exit_'+stage+'.json'),{'created_epoch':time.time(),'stage_complete':True,'completed':len(completed),'lease_released':True,'temporary_files':len([p for p in TMP.rglob('*') if p.is_file()]),'external_compute_empty_check_required':True})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=STAGES);run(a.parse_args().stage)
