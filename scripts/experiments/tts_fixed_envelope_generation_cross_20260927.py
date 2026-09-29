"""Frozen FIXED-background local-envelope intervention; serial bounded retention."""
from pathlib import Path
from contextlib import contextmanager
import argparse,fcntl,hashlib,json,math,os,platform,resource,shutil,subprocess,sys,time,types,signal
import numpy as np
import soundfile as sf
from scipy.signal import convolve
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/tts_fixed_envelope_generation_cross_20260927';OLD=ROOT/'runs/tts_fixed_level_generation_cross_20260927';FEAS=ROOT/'runs/tts_fixed_envelope_feasibility_20260927';TMP=Path('/dev/shm/tts_fixed_envelope_generation_20260927')
PY=ROOT/'.venv/bin/python';W2L=ROOT/'third_party/Wav2Lip';FFMPEG=Path('/home/wjj/miniconda3/bin/ffmpeg');FFPROBE=Path('/home/wjj/miniconda3/bin/ffprobe')
BASE=ROOT/'runs/tts_acoustic_generation_cross_20260926/code_snapshot/tts_acoustic_generation_cross_20260926.py'
b=types.ModuleType('sealed_envelope_base');b.__file__=str(ROOT/'scripts/experiments/tts_acoustic_generation_cross_20260926.py');exec(compile(BASE.read_text(),str(BASE),'exec'),b.__dict__)
CAP=200*2**20;OTHER=56*2**20;FLOOR=int(4.5*2**30);W=np.hanning(513);W/=W.sum()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for q in iter(lambda:f.read(2**20),b''):h.update(q)
 return h.hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def allocated(path):return sum(f.stat().st_blocks*512 for f in path.rglob('*') if f.is_file()) if path.exists() else 0
def limits(next_bytes=0,category='other'):
 used=allocated(OUT);free=shutil.disk_usage(OUT).free;tmp=allocated(TMP);extra=used-allocated(OUT/'features')-allocated(OUT/'controls')
 assert used+next_bytes<=CAP and free-max(0,CAP-used)-OTHER>=FLOOR,('disk',used,next_bytes,free)
 if category=='other':assert extra+next_bytes<=12*2**20,('nonfeature_cap',extra,next_bytes)
 if category=='control':assert allocated(OUT/'controls')+next_bytes<=5*2**20
 assert tmp<=64*2**20,('temporary_cap',tmp)
 available=next(int(q.split()[1])*1024 for q in Path('/proc/meminfo').read_text().splitlines() if q.startswith('MemAvailable:'))
 assert available>=2**30
 return {'time':time.time(),'allocated_bytes':used,'free_bytes':free,'remaining_own_commitment':CAP-used,'other_commitment':OTHER,'floor_bytes':FLOOR,'tmp_bytes':tmp,'maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'MemAvailable_bytes':available}
def write(path,value):
 path=Path(path);s=json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n';n=math.ceil(len(s.encode())/4096)*4096;cat='control' if path.is_relative_to(OUT/'controls') else 'other';limits(n,cat);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:f.write(s)
 assert path.stat().st_blocks*512<=n;limits()
def save(path,x,control=False):
 path=Path(path);x=np.asarray(x);assert x.dtype==np.float32 and x.ndim==2 and x.shape[1]==1024 and np.isfinite(x).all();n=math.ceil((x.nbytes+128)/4096)*4096;limits(n,'control' if control else 'feature');path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:np.save(f,x,allow_pickle=False)
 assert path.stat().st_blocks*512<=n;limits();return {'path':str(path),'sha256':sha(path),'shape':list(x.shape),'dtype':'float32','raw_sha256':ah(x)}
def runtime():
 import scipy,cv2,torch,python_speech_features
 return {'python':platform.python_version(),'executable':sys.executable,'numpy':np.__version__,'scipy':scipy.__version__,'soundfile':sf.__version__,'opencv':cv2.__version__,'torch':torch.__version__,'cuda':torch.version.cuda}
def protocol():
 review=read(OUT/'reviewer_pass.json');assert review['status']=='PASS' and sha(review['receipt'])==review['receipt_sha256'] and review['protocol_sha256']==sha(OUT/'protocol.json')
 from scripts.experiments.tts_native_gain_attribution import config as actual_config
 assert actual_config.FFMPEG.resolve()==FFMPEG.resolve() and actual_config.FFPROBE.resolve()==FFPROBE.resolve()
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==read(OUT/'seal.json')['protocol_sha256'];assert str(Path(sys.executable).resolve())==str(PY.resolve());assert runtime()==p['runtime']
 for path,h in p['dependencies'].items():assert sha(path)==h,path
 return p
def envelope(x):return np.sqrt(np.maximum(convolve(np.pad(x*x,(256,256),mode='reflect'),W,mode='valid',method='direct'),0))
def transform(x,R):
 e=envelope(x);g=np.maximum(e/R,.1)**(-.5);z=x*g;restore=R/np.sqrt(np.mean(z*z));y=z*restore;return y.astype(np.float32),g*restore,e,float(restore)
def frontend(y):
 sys.path.insert(0,str(W2L));import audio
 from scripts.experiments.static_image_bridge.render_worker import chunk_mels
 mel=audio.melspectrogram(np.asarray(y,dtype=np.float32)).astype(np.float32);windows,mfcc=b.frontend_windows(np.asarray(y,dtype=np.float64));chunks=chunk_mels(mel,25);assert all(np.isfinite(z).all() for z in [mel,windows,mfcc,np.asarray(chunks)]);return mel,windows,mfcc,chunks
def construct(row,arm,p):
 z=row['arms'][arm];assert sha(z['FIXED']['waveform'])==z['FIXED']['sha256'];x,sr=sf.read(z['FIXED']['waveform'],dtype='float64');assert sr==16000 and x.ndim==1 and len(x)==z['FIXED']['samples'];R=p['target_RMS'];assert np.sqrt(np.mean(x*x))>1e-8
 y,g,e,restore=transform(x,R);yy=y.astype(float);ey=envelope(yy);S=e>=.1*R;spread=float(np.ptp(np.percentile(20*np.log10(e[S]),[10,90]))) if S.any() else None;after=float(np.ptp(np.percentile(20*np.log10(ey[S]),[10,90]))) if S.any() else None
 checks={'finite':bool(np.isfinite(y).all()),'same_length':len(y)==len(x),'zero_preserved':bool(np.all(y[x==0]==0)),'positive_gain':bool(np.all(g>0)),'no_clip_peak':float(abs(y).max())<=.98,'RMS_relative':bool(abs(np.sqrt(np.mean(yy*yy))/R-1)<=1e-6),'inverse_saved':float(abs(yy/g-x).max())<=1e-7,'identity_exact':bool(np.array_equal(x.astype(np.float32).astype(float),x))}
 mel,win,mfcc,chunks=frontend(y);starts=[min(int(i*(80./25)),mel.shape[1]-16) for i in range(len(chunks))];assert all(np.array_equal(c,mel[:,k:k+16]) for c,k in zip(chunks,starts));assert all(np.array_equal(w,mfcc[i*4:i*4+20].T.astype(np.float32)) for i,w in enumerate(win));assert len(chunks)==z['FIXED']['frames'] and len(win)==z['FIXED']['audio_windows']
 d={'id':row['id'],'arm':arm,'split':row['split'],'checks':checks,'passed':all(checks.values()),'samples':len(x),'waveform_float32_sha256':ah(y),'gain_float64_sha256':ah(g),'mel_sha256':ah(mel),'MFCC_float64_sha256':ah(mfcc),'mel_chunk_starts':starts,'MFCC_window_starts':[i*4 for i in range(len(win))],'chunks_float32_sha256':ah(np.asarray(chunks,dtype=np.float32)),'frontend_all_finite':True,'windows_sha256':ah(win),'frames':len(chunks),'audio_windows':len(win),'gain_restore':restore,'gain_min':float(g.min()),'gain_max':float(g.max()),'peak':float(abs(y).max()),'RMS_relative_error':float(abs(np.sqrt(np.mean(yy*yy))/R-1)),'inverse_maxabs':float(abs(yy/g-x).max()),'S_count':int(S.sum()),'spread_input_dB':spread,'spread_output_dB':after,'spread_ratio':after/spread if spread is not None and spread>0 else None,'input_FIXED_sha256':z['FIXED']['sha256']}
 return y,mel,win,chunks,d

def freeze():
 assert not (OUT/'protocol.json').exists();old=read(OLD/'protocol.json');support=read(OLD/'support.json');assert len(support)==100;rows=[];bindings={};total=0
 for r in support:
  q={'id':r['id'],'speaker':r['speaker'],'split':r['split'],'guard20_eligible':r['guard20_eligible'],'arms':{}}
  orig=next(x for x in old['rows'] if x['id']==r['id'])
  for a in ['N','T']:
   info=r['arms'][a]['FIXED'];raw=r['arms'][a]['raw'];metap=OLD/'features'/r['id']/a/'FIXED.json';meta=read(metap);rawmp=OLD/'features'/r['id']/a/'raw.json';rawmeta=read(rawmp)
   q['arms'][a]={'FIXED':info,'RAW':raw,'source':orig['audio'][a],'joint_L':r['arms'][a]['joint_L'],'baseline_metadata':str(metap),'baseline_metadata_sha256':sha(metap),'baseline_features':meta['features'],'baseline_features_sha256':meta['sha256'],'RAW_metadata':str(rawmp),'RAW_metadata_sha256':sha(rawmp),'RAW_features':rawmeta['features'],'RAW_features_sha256':rawmeta['sha256']}
   total+=sum(math.ceil((n*4096+128)/4096)*4096 for n in [info['frames']-4,info['audio_windows']])
  rows.append(q)
 main=[r['id'] for r in rows if r['split']=='evaluation' and r['guard20_eligible']];assert len(main)==71 and len(set(r['speaker'] for r in rows if r['id'] in main))==15
 import python_speech_features as psf
 dep=[Path(__file__),BASE,OLD/'protocol.json',OLD/'support.json',OLD/'cal_controls.json',OLD/'runtime_calibration.json',OLD/'runtime_evaluation.json',FEAS/'v2_floor01/plan.json',FEAS/'v2_floor01/diagnostics.json',FEAS/'version_transition.json',Path(old['image']['path']),W2L/'audio.py',W2L/'hparams.py',W2L/'models/wav2lip.py',W2L/'models/conv.py',W2L/'checkpoints/wav2lip_gan.pth',ROOT/'third_party/syncnet_python/SyncNetInstance.py',ROOT/'third_party/syncnet_python/SyncNetModel.py',ROOT/'third_party/syncnet_python/data/syncnet_v2.model',FFMPEG,FFPROBE,Path(psf.__file__),Path(psf.__file__).parent/'base.py',Path(psf.__file__).parent/'sigproc.py']
 dep += [ROOT/'scripts/experiments'/x for x in ['static_image_bridge/render_worker.py','static_image_bridge/score_worker.py','masked_tts_tfg_probe/direct_mel.py','tts_native_gain_attribution/syncnet.py','tts_native_gain_attribution/audio.py','tts_native_gain_attribution/config.py','tts_native_gain_attribution/common.py']]
 oldruntime=read(OLD/'runtime_calibration.json');current=runtime();assert all(current[k if k!='torch_cuda' else 'cuda']==v for k,v in oldruntime.items())
 p={'version':2,'revision':'pre-forward engineering exact cache, project runtime, index/finite checks, bounded render and live independent firstcal hold; v1 preserved','old_runtime':oldruntime,'first_cal_external_gate':{'before_rest_cal_and_eval':True,'artifacts':'8videos,2candidateWAV,2frontendNPZ; exactlivehash binding','receipt_schema':'status PASS, protocol_sha256, media_hashes fullpath->sha256 must equal all12 live files','timeout_s':1800},'status':'frozen_before_new_eval_manipulation_or_model_forward','time':time.time(),'rows':rows,'main_ids':main,'target_RMS':old['fixed_target']['target_RMS'],'operation':{'window':'Hann513 symmetric normalized, reflect256 each side','gain':'max(sqrt(conv(x*x))/R,.1)^(-.5)','restore':'float64 whole waveform RMS to R, cast float32 no clipping','input':'exact cached FIXED float32 promoted float64','identity':'exponent0, original cached FIXED untouched; original RAW bridge only'},'cal_manipulation':{'S':'input env>=.1R, fixed sample indices','spread':'P90-P10 of20log10env; output evaluated on same S','per_arm_count_ratio_lt_09_atleast':23,'per_arm_median_ratio_lt':.8,'non_target':'low energy amplification/LTAS/F0 and voicing coupled; no purity claim'},'engineer_all200':{'finite_length_zeros_positive_gain':True,'peak_max':.98,'saved_RMS_relative_max':1e-6,'inverse_saved_maxabs':1e-7,'no_replacement':True},'image':old['image'],'generation':old['generation'],'scoring':old['scoring'],'statistics':old['statistics'],'cells':{'q00':['FIXED','FIXED'],'q10':['ENV','FIXED'],'q01':['FIXED','ENV'],'q11':['ENV','ENV']},'first_question':'Does total T-minus-N response shrink FIXED native C gap? primary raw/guard20 common71/15; 99CI upper<0 supports shrink, lower>0 expansion, otherwise unconfirmed','effects':'full G=q10-q00,E=q01-q00,I=q11-q10-q01+q00,total=q11-q00; N,T,T-N and native before/after; no cal effect statistics','controls':{'first_cal':min(r['id'] for r in rows if r['split']=='calibration'),'repeat':'first N/T RAW,FIXED reproduce old pixel and A/V EXACT; ENV duplicate pixel/A/V EXACT and full-vs-stream visual exact; PCM16 vs float MFCC/A bridge','delay':'first N/T RAW +3200 samples head zero exact length; query25:L-25; shift+5±1, overlapping31lag curves max<=.15','cal_migration':'all52 FIXED/ENV native raw guard20; each arm median vs old FIXED<=1 and vs k3<=1; N/T difference<=1; fail stops eval, no anchor search','pixel':'all new video FFmpeg independent raw BGR decode plus ffprobe PTS before temporary release; firstcal additionally held for external live auditor PASS before remaining cal/eval'},'retention':'full new float32 A/V; recipe/input/gain/mel/MFCC/pixel/file hashes and all31lag curves; no persistent new fullvideo/WAV/mel/z/distance matrices; first controls A/V retained','budget':{'feature_allocated_exact_bound':total,'controls_cap':5*2**20,'other_cap':12*2**20,'proved_total_bound':total+17*2**20,'own_cap':CAP,'other_reserved':OTHER,'floor_bytes':FLOOR,'temporary_cap':64*2**20,'temporary_enforcement':'per-video RLIMIT_FSIZE <= remaining aggregate tmp capacity minus1MiB, restored after writer close; batch pre/post checks; other temporary writes bounded before creation'},'runtime':runtime(),'dependencies':{str(x.resolve()):sha(x) for x in dep},'history':'oldENV different; old dynamic compression exists. Floor.01 development ran but numerical diagnostics not saved/seen due budget; floor.1 and manipulation gates fixed by root before floor.1 outcomes. Historical reused cohort not unseen confirmation. Only joint operation, no physical mouth or pure envelope claim.'}
 assert p['budget']['proved_total_bound']<194*2**20
 write(OUT/'protocol.json',p);write(OUT/'seal.json',{'time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'worker_sha256':sha(__file__),'no_new_forward':True});dest=OUT/'code_snapshot'/Path(__file__).name;dest.parent.mkdir();shutil.copyfile(__file__,dest);limits();print('FROZEN',sha(OUT/'protocol.json'),p['budget'],flush=True)

def prepare():
 p=protocol();assert not (OUT/'input_gate.json').exists();allrows=[]
 for row in p['rows']:
  for arm in ['N','T']:
   z=row['arms'][arm]
   for f,h in [(z['source']['path'],z['source']['sha256']),(z['baseline_metadata'],z['baseline_metadata_sha256']),(z['baseline_features'],z['baseline_features_sha256']),(z['FIXED']['frontend'],z['FIXED']['frontend_sha256'])]:assert sha(f)==h,f
   pcm,sr=sf.read(z['source']['path'],dtype='int16');raw=pcm.astype(float)/32768;fixed,sr=sf.read(z['FIXED']['waveform'],dtype='float64');assert sr==16000 and np.array_equal((raw*(p['target_RMS']/np.sqrt(np.mean(raw*raw)))).astype(np.float32).astype(float),fixed)
   mel,win,_,_=frontend(fixed);oldfront=np.load(z['FIXED']['frontend']);assert np.array_equal(mel,oldfront['mel']) and np.array_equal(win,oldfront['windows'])
   _,_,_,_,d=construct(row,arm,p);write(OUT/'operations'/row['id']/(arm+'.json'),d);allrows.append(d);limits()
  print('prepared',row['id'],flush=True)
 cal=[r for r in allrows if r['split']=='calibration'];counts={a:sum(r['spread_ratio'] is not None and r['spread_ratio']<.9 for r in cal if r['arm']==a) for a in ['N','T']};medians={a:float(np.median([r['spread_ratio'] for r in cal if r['arm']==a])) for a in ['N','T']};gates={'all200_engineering':all(r['passed'] for r in allrows),'cal_N_spread':counts['N']>=23 and medians['N']<.8,'cal_T_spread':counts['T']>=23 and medians['T']<.8};write(OUT/'input_gate.json',{'time':time.time(),'passed':all(gates.values()),'gates':gates,'counts':counts,'medians':medians,'all200_count':len(allrows),'calibration':len(cal),'evaluation':len(allrows)-len(cal),'no_eval_effect_scores':True})
 write(OUT/'input_seal.json',{'time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'input_gate_sha256':sha(OUT/'input_gate.json'),'operations':{str(x):sha(x) for x in sorted((OUT/'operations').rglob('*.json'))}});assert all(gates.values());print('INPUT_PASS',flush=True)

def gpucheck(start=False):
 r=limits();jobs=b.compute_processes();assert all(q['pid']==os.getpid() for q in jobs),jobs
 used,util=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
 if start:assert used<=128 and util<=10,(used,util)
 return {**r,'compute':jobs,'GPU_memory_MiB':used,'GPU_utilization':util}
@contextmanager
def lease(split):
 with open('/tmp/tts-exp-gpu.lock','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);checks=[]
  try:
   for i in range(3):
    checks.append(gpucheck(True))
    if i<2:time.sleep(5)
   write(OUT/'runtime'/('resource_'+split+'.json'),checks);yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)
def statehash(model):
 h=hashlib.sha256()
 for k,v in model.state_dict().items():h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def decoded(video):
 argv=[str(FFMPEG),'-v','error','-threads','1','-i',str(video),'-threads','1','-f','rawvideo','-pix_fmt','bgr24','pipe:1'];proc=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE);h=hashlib.sha256();n=0
 while True:
  q=proc.stdout.read(224*224*3)
  if not q:break
  assert len(q)==224*224*3;h.update(q);n+=1
 err=proc.stderr.read();assert proc.wait()==0,err[-500:]
 frames=json.loads(subprocess.check_output([str(FFPROBE),'-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=pts,best_effort_timestamp_time','-of','json',str(video)],text=True))['frames'];assert len(frames)==n
 t=np.array([float(x['best_effort_timestamp_time']) for x in frames]);assert np.allclose(t,np.arange(n)/25,atol=1e-9,rtol=0)
 return {'pixel_sha256':h.hexdigest(),'frames':n,'PTS':[int(x['pts']) for x in frames],'time_seconds_sha256':ah(t),'decode':'independent FFmpeg BGR24 plus ffprobe continuous25fps'}
def full_visual(engine,path):
 frames=list(engine._stream_mjpeg(path));out=[];torch=engine._torch
 with torch.inference_mode():
  for i in range(0,len(frames)-4,32):
   windows=[frames[j:j+5] for j in range(i,min(i+32,len(frames)-4))];out.append(engine.network.forward_lip(torch.from_numpy(engine._visual_batch(windows)).cuda()).cpu().numpy().astype(np.float32))
 return np.concatenate(out)

def render_cell(model,chunks,frame,image_tensor,box,roi,path,capture_z=False):
 assert capture_z is False and not path.exists()
 import cv2,torch
 from scripts.experiments.static_image_bridge.score_worker import crop_zero_padded
 limits();remaining=64*2**20-allocated(TMP)-2**20;assert remaining>4096
 cap=(remaining//4096)*4096;oldlimit=resource.getrlimit(resource.RLIMIT_FSIZE);oldsignal=signal.getsignal(signal.SIGXFSZ)
 def too_large(signum,stack):raise RuntimeError('temporary FFV1 hard file-size bound exceeded; preserve scene')
 resource.setrlimit(resource.RLIMIT_FSIZE,(cap,oldlimit[1]));signal.signal(signal.SIGXFSZ,too_large);writer=None;digest=hashlib.sha256();x1,y1,x2,y2=box
 try:
  path.parent.mkdir(parents=True,exist_ok=True);writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),25,(224,224));assert writer.isOpened()
  for start in range(0,len(chunks),32):
   gpucheck();mt=torch.from_numpy(np.asarray(chunks[start:start+32],dtype=np.float32)[:,None]).cuda()
   with torch.inference_mode():pred=model(mt,image_tensor.expand(len(mt),-1,-1,-1)).cpu().numpy().transpose(0,2,3,1)*255
   for fo in pred:
    full=frame.copy();full[y1:y2,x1:x2]=cv2.resize(fo.astype(np.uint8),(x2-x1,y2-y1));cropped=crop_zero_padded(full,roi);writer.write(cropped);digest.update(cropped.tobytes())
   gpucheck()
 finally:
  try:
   if writer is not None:writer.release()
  finally:resource.setrlimit(resource.RLIMIT_FSIZE,oldlimit);signal.signal(signal.SIGXFSZ,oldsignal)
 assert path.stat().st_size<=cap;limits()
 return {'path':str(path),'sha256':sha(path),'pixel_sha256':digest.hexdigest(),'frames':len(chunks),'hard_file_size_cap':cap},None

def temp_reserve(n):
 assert allocated(TMP)+math.ceil(n/4096)*4096<=64*2**20;limits()
def wait_firstcal(p):
 files={str(f):sha(f) for f in sorted(TMP.rglob('*')) if f.is_file()};assert sum(x.endswith('.avi') for x in files)==8 and sum(x.endswith('.wav') for x in files)==2 and sum(x.endswith('.npz') for x in files)==2
 write(OUT/'first_cal_live.json',{'time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'files':files,'no_rest_cal_or_eval_before_audit':True});print('FIRST_CAL_LIVE_WAIT',str(OUT/'first_cal_live.json'),flush=True);t=time.time()
 while not (OUT/'first_cal_media_audit.json').exists():
  assert time.time()-t<1800,'external firstcal audit timed out; preserve all temporary media';limits();time.sleep(2)
 binding=read(OUT/'first_cal_media_audit.json');assert binding['status']=='PASS' and sha(binding['receipt'])==binding['receipt_sha256'];receipt=read(binding['receipt']);assert receipt['status']=='PASS' and receipt['protocol_sha256']==sha(OUT/'protocol.json') and receipt['media_hashes']==files
 for f,h in files.items():assert Path(f).is_relative_to(TMP) and sha(f)==h
 for f in files:Path(f).unlink()
 write(OUT/'first_cal_release.json',{'time':time.time(),'receipt':binding,'released':files,'only_own_new_temporary_media':True});limits()

def produce(split):
 p=protocol();assert read(OUT/'input_gate.json')['passed'];assert sha(OUT/'input_gate.json')==read(OUT/'input_seal.json')['input_gate_sha256']
 if split=='evaluation':assert read(OUT/'cal_controls.json')['passed']
 import cv2,torch
 from scripts.experiments.masked_tts_tfg_probe.direct_mel import _load_model
 from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
 cv2.setNumThreads(1);torch.set_num_threads(2);torch.manual_seed(20260926);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;TMP.mkdir(exist_ok=True);assert not any(x.is_file() for x in TMP.rglob('*'))
 rows=[r for r in p['rows'] if r['split']==split];first=p['controls']['first_cal']
 with lease(split):
  model=_load_model(W2L/'checkpoints/wav2lip_gan.pth','cuda');engine=SyncNetEngine(batch_size=32,device='cuda');write(OUT/'runtime'/('loaded_'+split+'.json'),{'runtime':runtime(),'wav2lip_loaded_state':statehash(model),'syncnet_loaded_state':statehash(engine.network),'seed':20260926,'batch32':True})
  im=p['image'];frame=cv2.imread(im['path']);x1,y1,x2,y2=im['generation_box'];face=cv2.resize(frame[y1:y2,x1:x2],(96,96));mask=face.copy();mask[48:]=0;it=torch.from_numpy(np.concatenate([mask,face],2).transpose(2,0,1).astype(np.float32)/255).unsqueeze(0).cuda()
  try:
   for row in rows:
    for arm in ['N','T']:
     gpucheck();z=row['arms'][arm];y,mel,win,chunks,d=construct(row,arm,p);saved=read(OUT/'operations'/row['id']/(arm+'.json'));assert d==saved and d['passed'];path=TMP/(row['id']+'_'+arm+'_ENV.avi');assert not path.exists();video,_=render_cell(model,chunks,frame,it,im['generation_box'],im['score_box']['box'],path,False);v,vm=engine.extract_visual(path);a=b.audio_forward(engine,win);dec=decoded(path);assert dec['pixel_sha256']==video['pixel_sha256'] and dec['frames']==video['frames']==d['frames'];assert v.shape==(d['frames']-4,1024) and a.shape==(d['audio_windows'],1024)
     dest=OUT/'features'/row['id']/arm;aa=save(dest/'A.npy',a);vv=save(dest/'V.npy',v);meta={'id':row['id'],'arm':arm,'condition':'ENV','audio':aa,'visual':vv,'video_at_creation':video,'decode':dec,'visual_metadata':vm,'operation_sha256':sha(OUT/'operations'/row['id']/(arm+'.json')),'retained_video':False,'joint_L':z['joint_L']};write(dest/'metadata.json',meta)
     if row['id']==first:
      checks={};cp=OUT/'controls'/arm
      for cond in ['RAW','FIXED','ENV']:
       if cond=='ENV':wave=y.astype(float);refv=v;refa=a;refpixel=video['pixel_sha256'];cmel,cwin,_,cc=frontend(wave)
       else:
        wave,sr=sf.read(z[cond]['waveform'],dtype='float64');cmel,cwin,_,cc=frontend(wave);mf=read(z['RAW_metadata'] if cond=='RAW' else z['baseline_metadata']);old=np.load(mf['features']);refv=old['visual'];refa=old['audio'];refpixel=mf['video']['pixel_sha256']
       rep=TMP/(row['id']+'_'+arm+'_'+cond+'_repeat.avi');rv,_=render_cell(model,cc,frame,it,im['generation_box'],im['score_box']['box'],rep,False);av=b.audio_forward(engine,cwin);ve,_=engine.extract_visual(rep);rd=decoded(rep);full=full_visual(engine,rep);tests={'pixel_equal':rv['pixel_sha256']==refpixel==rd['pixel_sha256'],'audio_max':float(abs(av-refa).max()),'visual_max':float(abs(ve-refv).max()),'audio_exact':bool(np.array_equal(av,refa)),'visual_exact':bool(np.array_equal(ve,refv)),'stream_full_exact':bool(np.array_equal(full,ve))};tests['passed']=tests['pixel_equal'] and tests['audio_exact'] and tests['visual_exact'] and tests['stream_full_exact'];rr={'condition':cond,'tests':tests,'video_at_creation':rv,'decode':rd,'A':save(cp/(cond+'_A.npy'),av,True),'V':save(cp/(cond+'_V.npy'),ve,True)};write(cp/(cond+'.json'),rr);checks[cond]=tests;assert tests['passed'];assert sha(rep)==rv['sha256']
      pcm,sr=sf.read(z['source']['path'],dtype='int16');oldwin,_=engine._audio_windows(pcm);newwin,_=b.frontend_windows(pcm.astype(float)/32768);assert np.array_equal(oldwin,newwin);oldae,_=engine.extract_audio(z['source']['path']);float_ae=b.audio_forward(engine,newwin);assert np.array_equal(oldae,float_ae);shift=np.zeros(len(pcm));shift[3200:]=pcm.astype(float)[:-3200]/32768;dw,_=b.frontend_windows(shift);da=b.audio_forward(engine,dw);delay=save(cp/'delay5_A.npy',da,True)
      wavtmp=TMP/(row['id']+'_'+arm+'.wav');temp_reserve(y.nbytes+4096);sf.write(wavtmp,y,16000,subtype='FLOAT');sys.path.insert(0,str(W2L));import audio;loaded=audio.load_wav(str(wavtmp),16000);assert np.array_equal(loaded,y) and np.array_equal(audio.melspectrogram(loaded).astype(np.float32),mel);wavhash=sha(wavtmp);_,gg,_,_=transform(sf.read(z['FIXED']['waveform'],dtype='float64')[0],p['target_RMS']);_,_,mf,_=frontend(y);temp_reserve(sum(q.nbytes for q in [gg,mel,win,mf])+4096);np.savez(TMP/(row['id']+'_'+arm+'_frontend.npz'),gain=gg,mel=mel,windows=win,MFCC=mf);write(cp/'bridge.json',{'passed':True,'PCM_MFCC_exact':True,'PCM_FLOAT_A_exact':True,'FLOAT_WAV_mel_exact':True,'candidate_FLOAT_WAV_sha256':wavhash,'delay':delay,'checks':checks})
     assert np.array_equal(np.load(aa['path']),a) and np.array_equal(np.load(vv['path']),v) and sha(path)==video['sha256']
     if row['id']!=first:path.unlink()
     limits()
    if row['id']==first:wait_firstcal(p)
    print('produced',split,row['id'],flush=True)
  finally:engine.close();del model
 write(OUT/'runtime'/('worker_exit_'+split+'.json'),{'time':time.time(),'stage_complete':True,'lease_released':True,'temporary_files':len([q for q in TMP.rglob('*') if q.is_file()]),'note':'external parent confirms process exit and compute-empty'})

def features(row,arm,cond):
 z=row['arms'][arm]
 if cond=='FIXED':
  assert sha(z['baseline_features'])==z['baseline_features_sha256'];f=np.load(z['baseline_features']);return f['visual'],f['audio']
 dest=OUT/'features'/row['id']/arm;m=read(dest/'metadata.json');assert sha(m['audio']['path'])==m['audio']['sha256'] and sha(m['visual']['path'])==m['visual']['sha256'];return np.load(m['visual']['path']),np.load(m['audio']['path'])
def score(split):
 import torch
 torch.set_num_threads(2);p=protocol();rows=[r for r in p['rows'] if r['split']==split]
 if split=='evaluation':assert read(OUT/'cal_controls.json')['passed']
 seals={str(f):sha(f) for r in rows for f in sorted((OUT/'features'/r['id']).rglob('*')) if f.is_file()};write(OUT/('feature_seal_'+split+'.json'),seals);write(OUT/('score_lock_'+split+'.json'),{'time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'feature_seal_sha256':sha(OUT/('feature_seal_'+split+'.json')),'input_seal_sha256':sha(OUT/'input_seal.json')})
 for row in rows:
  cells={}
  for arm in ['N','T']:
   L=row['arms'][arm]['joint_L'];f={c:features(row,arm,c) for c in ['FIXED','ENV']};cells[arm]={}
   for geom in ['raw','unit']:
    x={c:tuple(b.unit(z[:L]) if geom=='unit' else z[:L] for z in vv) for c,vv in f.items()};cells[arm][geom]={}
    for cell,(vc,ac) in p['cells'].items():
     if split=='calibration' and cell not in ['q00','q11']:continue
     m=b.matrix(x[vc][0],x[ac][1]);cells[arm][geom][cell]={pol:b.summarize(m,pol,3) for pol in p['scoring']['policies']}
  write(OUT/'scores'/(row['id']+'.json'),{'id':row['id'],'speaker':row['speaker'],'split':split,'guard20_eligible':row['guard20_eligible'],'cells':cells});print('scored',split,row['id'],flush=True)
 if split=='calibration':
  old=read(OLD/'cal_controls.json');lags={};gates=[]
  for cond,cell in [('FIXED','q00'),('ENV','q11')]:
   lags[cond]={a:float(np.median([read(OUT/'scores'/(r['id']+'.json'))['cells'][a]['raw'][cell]['guard20']['best_lag'] for r in rows])) for a in ['N','T']}
   gates += [abs(lags[cond][a]-old['condition_arm_medianlags']['FIXED'][a])<=1 and abs(lags[cond][a]-3)<=1 for a in ['N','T']];gates.append(abs(lags[cond]['N']-lags[cond]['T'])<=1)
  delays={};first=next(r for r in rows if r['id']==p['controls']['first_cal'])
  for a in ['N','T']:
   z=first['arms'][a];raw=np.load(z['RAW_features']);delay=np.load(OUT/'controls'/a/'delay5_A.npy');L=z['joint_L'];m=b.matrix(raw['visual'][:L],raw['audio'][:L]);dm=b.matrix(raw['visual'][:L],delay[:L]);curve=m[25:L-25].mean(0);dc=dm[25:L-25].mean(0);be=int(curve.argmin())-15;de=int(dc.argmin())-15;err=float(abs(dc[5:]-curve[:-5]).max());delays[a]={'base_lag':be,'delay_lag':de,'overlap_error':err,'passed':abs(de-be-5)<=1 and err<=.15};gates.append(delays[a]['passed']);gates.append(read(OUT/'controls'/a/'bridge.json')['passed'])
  write(OUT/'cal_controls.json',{'time':time.time(),'passed':all(gates),'condition_arm_medianlags':lags,'old_FIXED_medianlags':old['condition_arm_medianlags']['FIXED'],'delay':delays,'anchor':3,'no_parameter_support_or_k_change':True,'no_cal_effect_summary':True});assert all(gates),'cal migration/controls failed'

def analyze():
 p=protocol();assert read(OUT/'cal_controls.json')['passed'];rows=[r for r in p['rows'] if r['split']=='evaluation'];scores={r['id']:read(OUT/'scores'/(r['id']+'.json')) for r in rows};out={};per=[];closure=0
 for geom in ['raw','unit']:
  for pol in ['guard20','valid','guard0']:
   supports=['common71'] if pol=='guard20' else ['common71','all74']
   for support in supports:
    use=[r for r in rows if support=='all74' or r['id'] in p['main_ids']];groups=[r['speaker'] for r in use];view={}
    for metric in ['C','B','D','C_anchor','D_anchor','best_lag']:
     armdata={a:[] for a in ['N','T']};q0={a:[] for a in ['N','T']};q1={a:[] for a in ['N','T']}
     for r in use:
      for a in ['N','T']:
       c=scores[r['id']]['cells'][a][geom];q=[c[k][pol][metric] for k in ['q00','q10','q01','q11']];e=b.effects(q);closure=max(closure,abs(e['G']+e['E']+e['I']-e['total']));armdata[a].append(e);q0[a].append(q[0]);q1[a].append(q[3]);per.append({'id':r['id'],'speaker':r['speaker'],'view':geom+'/'+pol+'/'+support,'metric':metric,'arm':a,'q00':q[0],'q10':q[1],'q01':q[2],'q11':q[3],**e})
     v={'baseline_T_minus_N':b.bootstrap(np.array(q0['T'])-q0['N'],groups),'processed_T_minus_N':b.bootstrap(np.array(q1['T'])-q1['N'],groups),'effects':{}}
     for key in ['G','E','I','total']:
      n=np.array([x[key] for x in armdata['N']]);t=np.array([x[key] for x in armdata['T']]);v['effects'][key]={'N':b.bootstrap(n,groups),'T':b.bootstrap(t,groups),'T_minus_N':b.bootstrap(t-n,groups)}
     view[metric]=v
    out[geom+'/'+pol+'/'+support]=view
 main=out['raw/guard20/common71']['C']['effects']['total']['T_minus_N'];status='GAP_SHRINK_CONFIRMED' if main['ci99'][1]<0 else ('GAP_EXPANSION_CONFIRMED' if main['ci99'][0]>0 else 'GAP_CHANGE_NOT_CONFIRMED')
 write(OUT/'effects.json',per);write(OUT/'summary.json',{'status':status,'primary_total_T_minus_N':main,'views':out,'maximum_fourcell_closure':closure,'no_physical_quality_or_pure_envelope_claim':True,'no_sample_deletion':True});print(status,json.dumps(main),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','prepare','produce','score','analyze']);ap.add_argument('--split',choices=['calibration','evaluation']);a=ap.parse_args();globals()[a.stage](a.split) if a.stage in ['produce','score'] else globals()[a.stage]()
