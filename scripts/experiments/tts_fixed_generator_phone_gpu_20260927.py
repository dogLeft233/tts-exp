"""GPU stages for frozen generator-phone prototypes; CPU fit/scoring live elsewhere."""
from pathlib import Path
from contextlib import contextmanager
import argparse,fcntl,json,math,os,platform,resource,shutil,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments import tts_fixed_generator_phone_stream_20260927 as stream
OUT=ROOT/'runs/tts_fixed_generator_phone_prototype_20260927'
TMP=Path('/dev/shm/tts_fixed_generator_phone_prototype_20260927')
W2L=ROOT/'third_party/Wav2Lip';PY=ROOT/'.venv/bin/python';CONDITIONS=('phoneN','phoneT','globalN','globalT')
sha=stream.sha;ah=stream.ah

def read(p):return json.loads(Path(p).read_text())
def allocated(p):return sum(x.stat().st_blocks*512 for x in p.rglob('*') if x.is_file()) if p.exists() else 0
def config():return read(OUT/'protocol.json')['gpu_config']
def limits(extra=0):
 c=config();used=allocated(OUT);free=shutil.disk_usage(OUT).free;tmp=allocated(TMP)
 assert used+extra<=c['own_cap_bytes'] and free-max(0,c['own_cap_bytes']-used)-c['other_reserved_bytes']>=c['floor_bytes'],('disk_gate',used,extra,free)
 assert tmp<=c['tmp_cap_bytes'],('tmp_gate',tmp)
 av=next(int(q.split()[1])*1024 for q in Path('/proc/meminfo').read_text().splitlines() if q.startswith('MemAvailable:'))
 assert av>=2**30
 return {'time':time.time(),'allocated_bytes':used,'free_bytes':free,'remaining_commitment':max(0,c['own_cap_bytes']-used),'other_reserved_bytes':c['other_reserved_bytes'],'floor_bytes':c['floor_bytes'],'tmp_bytes':tmp,'MemAvailable_bytes':av,'maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
def write(p,z):
 data=(json.dumps(z,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode();limits(math.ceil(len(data)/4096)*4096)
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(data)
 limits()
def save(p,a):
 assert a.dtype==np.float32 and a.ndim==2 and a.shape[1]==1024 and np.isfinite(a).all()
 p=Path(p);bound=math.ceil((a.nbytes+128)/4096)*4096;limits(bound);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:np.save(f,a,allow_pickle=False)
 assert p.stat().st_blocks*512<=bound and np.array_equal(np.load(p),a);limits()
 return {'path':str(p),'sha256':sha(p),'raw_sha256':ah(a),'shape':list(a.shape),'dtype':'float32'}
def runtime():
 import cv2,torch,scipy,soundfile
 return {'python':platform.python_version(),'executable':sys.executable,'numpy':np.__version__,'scipy':scipy.__version__,'soundfile':soundfile.__version__,'opencv':cv2.__version__,'torch':torch.__version__,'cuda':torch.version.cuda}
def protocol(stage):
 p=read(OUT/'protocol.json');c=p['gpu_config'];review=read(OUT/'reviewer_pass.json')
 assert review['status']=='PASS' and sha(review['receipt'])==review['receipt_sha256'] and review['protocol_sha256']==sha(OUT/'protocol.json')
 for path,h in p['dependencies'].items():assert sha(path)==h,path
 for key in ['row_path','parent_protocol_path','checkpoint','ffmpeg','ffprobe']:
  assert sha(c[key])==c[key+'_sha256'],key
 prereq=c['prerequisite_closure'];assert sha(prereq['path'])==prereq['sha256'] and read(prereq['path'])['status'] in ('PASS','concluded')
 assert runtime()==c['runtime'] and Path(sys.executable).resolve()==PY.resolve()
 for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ.get(k)=='1'
 from scripts.experiments.tts_native_gain_attribution import config as sc
 assert Path(sc.FFMPEG).resolve()==Path(c['ffmpeg']).resolve() and Path(sc.FFPROBE).resolve()==Path(c['ffprobe']).resolve()
 if stage!='identity':
  gate=read(OUT/'identity_gate.json');assert gate['passed'] and gate['external_media_audit_passed']
  fs=read(OUT/'fit_seal.json')
  assert fs['protocol_sha256']==sha(OUT/'protocol.json') and fs['identity_gate_sha256']==sha(OUT/'identity_gate.json')
  for path,h in fs['files'].items():assert sha(path)==h,path
 if stage=='evaluation':assert read(OUT/'calibration_gate.json')['passed']
 limits();return p,read(c['row_path'])
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
def frontend(row,arm):
 z=row['arms'][arm]
 for key in ['features','frontend','waveform']:assert sha(z[key+'_path'])==z[key+'_sha256']
 assert sha(z['metadata_path'])==z['metadata_sha256']
 f=np.load(z['features_path']);fr=np.load(z['frontend_path']);mel=fr['mel']
 sys.path.insert(0,str(W2L));import audio
 actual=audio.melspectrogram(audio.load_wav(z['waveform_path'],16000)).astype(np.float32)
 assert np.array_equal(actual,mel) and np.isfinite(mel).all()
 from scripts.experiments.static_image_bridge.render_worker import chunk_mels
 chunks=chunk_mels(mel,25);starts=[]
 for i in range(len(chunks)):
  start=int(i*(80.0/25));starts.append(start if start+16<=mel.shape[1] else mel.shape[1]-16)
 assert starts==z['mel_starts'] and len(chunks)==z['frames']
 assert all(np.array_equal(chunk,mel[:,start:start+16]) for chunk,start in zip(chunks,starts))
 assert f['z'].dtype==np.float32 and f['z'].shape==(z['frames'],512) and f['visual'].shape==(z['frames']-4,1024)
 assert len(z['phone_labels'])==len(z['speech_mask'])==len(chunks)
 return chunks,f,{'mel_sha256':ah(mel),'chunks_sha256':ah(np.asarray(chunks,dtype=np.float32)),'starts_sha256':ah(np.asarray(starts,dtype=np.int64)),'parent_features_sha256':z['features_sha256'],'frontend_sha256':z['frontend_sha256'],'waveform_sha256':z['waveform_sha256'],'waveform_samples':z['samples'],'L':z['L']}
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

def run(stage):
 p,rows=protocol(stage);c=p['gpu_config'];first=next(r for r in rows if r['id']==c['first_cal'])
 import cv2,torch
 from scripts.experiments.masked_tts_tfg_probe.direct_mel import _load_model
 from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
 cv2.setNumThreads(1);torch.set_num_threads(2);torch.manual_seed(20260926);torch.cuda.manual_seed_all(20260926);np.random.seed(20260926);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 TMP.mkdir(exist_ok=True);assert not any(x.is_file() for x in TMP.rglob('*'))
 fit=None
 if stage!='identity':
  fitmeta=read(OUT/'fit.json');fit=np.load(OUT/'fit.npz');assert fitmeta['sources']==['N','T']
 targets=[first] if stage in ('identity','calibration') else [r for r in rows if r['split']=='evaluation'];completed=[]
 with lease(stage):
  sys.path.insert(0,str(W2L))
  model=_load_model(Path(c['checkpoint']),'cuda');engine=SyncNetEngine(batch_size=32,device='cuda')
  write(OUT/'gpu_runtime'/('loaded_'+stage+'.json'),{'time':time.time(),'runtime':runtime(),'wav2lip_state_sha256':stream.statehash(model),'syncnet_state_sha256':stream.statehash(engine.network),'seed':20260926,'torch_threads':2,'batch':32})
  frame=cv2.imread(c['image']['path']);assert frame is not None and sha(c['image']['path'])==c['image']['sha256'];x1,y1,x2,y2=c['image']['generation_box'];face=cv2.resize(frame[y1:y2,x1:x2],(96,96));masked=face.copy();masked[48:]=0
  image_tensor=torch.from_numpy(np.concatenate([masked,face],2).transpose(2,0,1).astype(np.float32)/255).unsqueeze(0).cuda()
  try:
   for row in targets:
    for arm in 'NT':
     chunks,parent,front=frontend(row,arm);expected=parent['z']
     if stage=='identity':
      checks={};original_pixel=None
      parentmeta=Path(row['arms'][arm]['metadata_path']);pm=read(parentmeta);assert pm['sha256']==row['arms'][arm]['features_sha256']
      for mode in ['none','noop','cached']:
       vv,meta=one(model,engine,chunks,frame,image_tensor,c,row,arm,mode,expected,expected.copy() if mode=='cached' else None,mode,full=True)
       assert np.array_equal(vv,parent['visual']) and meta['video_at_creation']['pixel_sha256']==pm['video']['pixel_sha256']
       checks[mode]={'V_exact':True,'pixel_exact':True,'native_z_exact':True,**meta}
       if mode=='none':keep=save(OUT/'gpu_controls'/arm/'identity_V.npy',vv)
      write(OUT/'gpu_controls'/arm/'identity.json',{'passed':True,'checks':checks,'V':keep,'frontend':front,'parent_metadata_sha256':sha(parentmeta),'audio_forward_calls':0});completed.append(row['id']+'/'+arm)
     else:
      fold=fitmeta['folds'].index(row['speaker'])
      for condition in CONDITIONS:
       kind='phone' if condition.startswith('phone') else 'global';source=condition[-1];ix=fitmeta['sources'].index(source)
       phones,global_mu=stream.template_table(fit['raw_mu'],fit['speaker_counts'],fitmeta['labels'],fold,ix)
       replacement,mix=stream.replacement_table(expected,row['arms'][arm]['phone_labels'],row['arms'][arm]['speech_mask'],phones,global_mu,kind)
       v,meta=one(model,engine,chunks,frame,image_tensor,c,row,arm,condition,expected,replacement,'processed',full=stage=='calibration')
       vf=save(OUT/'visual'/row['id']/arm/(condition+'.npy'),v)
       if stage=='calibration':
        repeat,rm=one(model,engine,chunks,frame,image_tensor,c,row,arm,condition+'_repeat',expected,replacement,'processed',full=True)
        assert np.array_equal(v,repeat) and meta['video_at_creation']['pixel_sha256']==rm['video_at_creation']['pixel_sha256'];meta['repeat']={'V_exact':True,'pixel_exact':True,'metadata':rm}
       record={'id':row['id'],'arm':arm,'condition':condition,'V':vf,'frontend':front,'parentbindings':row['arms'][arm],'fit_sha256':sha(OUT/'fit.npz'),'fit_metadata_sha256':sha(OUT/'fit.json'),'fold':row['speaker'],'mix':mix,**meta}
       write(OUT/'metadata'/row['id']/arm/(condition+'.json'),record);completed.append(row['id']+'/'+arm+'/'+condition)
       if stage=='evaluation':
        path=Path(meta['video_at_creation']['path']);assert path.is_relative_to(TMP) and sha(path)==meta['video_at_creation']['sha256'];path.unlink();limits()
     print('produced',stage,row['id'],arm,flush=True)
   if stage in ('identity','calibration'):
    receipt=live_gate(stage);write(OUT/(stage+'_gate.json'),{'passed':True,'created_epoch':time.time(),'external_media_audit_passed':True,'receipt':receipt,'completed':completed,'no_effect_scores':True,'no_parameter_selection':True})
  finally:engine.close();del model
 write(OUT/'gpu_runtime'/('worker_exit_'+stage+'.json'),{'created_epoch':time.time(),'stage_complete':True,'completed':len(completed),'lease_released':True,'temporary_files':len([p for p in TMP.rglob('*') if p.is_file()]),'external_compute_empty_check_required':True})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['identity','calibration','evaluation']);args=a.parse_args();run(args.stage)
