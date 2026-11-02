"""Frozen selected-six CoGenAV feature/DTW video retiming adaptation, not paper inference."""
from pathlib import Path
import datetime, hashlib, importlib.util, json, os, pickle, shutil, subprocess, sys, time
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/cogenav_video_retime_20261025_attempt01'
PREV=ROOT/'runs/native_warp_sync_pilot_20261025_attempt01'
MODEL=ROOT/'runs/cogenav_sync_audit_20261024_attempt01'
SYNC=ROOT/'third_party/syncnet_python'
SPY='/home/wjj/.venvs/syncnet/bin/python'
ROLES=('N_codec','linear','cogenav_av','cogenav_aa','mfcc_aa')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def module(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def cosine_cost(a,b,relu=False):
 a=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
 b=b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12)
 c=np.clip(a@b.T,-1,1)
 return 1-np.maximum(c,0) if relu else 1-c
def dtw(cost):
 n,m=cost.shape;acc=np.full((n+1,m+1),np.inf);back=np.full((n,m),-1,dtype=np.int8);acc[0,0]=0
 for i in range(n):
  for j in range(m):
   if abs(i/max(n-1,1)-j/max(m-1,1))>.25:continue
   opts=(acc[i,j],acc[i,j+1]+.05,acc[i+1,j]+.05);k=int(np.argmin(opts));acc[i+1,j+1]=opts[k]+cost[i,j];back[i,j]=k
 if not np.isfinite(acc[n,m]):raise ValueError('no finite path')
 path=[];i,j=n-1,m-1
 while i>=0 and j>=0:
  path.append((i,j));k=back[i,j]
  if k==0:i,j=i-1,j-1
  elif k==1:i-=1
  elif k==2:j-=1
  else:raise ValueError('invalid traceback')
 return np.asarray(path[::-1],dtype=np.int64)
def project(path,n,m):
 pred=np.asarray([np.median(path[path[:,0]==i,1]) for i in range(n)])
 pred[0]=0;pred[-1]=m-1
 if not np.isfinite(pred).all() or np.any(np.diff(pred)<0):raise ValueError('nonmonotone map')
 return pred,nearest(pred,m)
def nearest(pred,m):return np.clip(np.floor(np.asarray(pred)+.5).astype(np.int64),0,m-1)
def scalar(z):
 import torch
 mu=torch.from_numpy(np.ascontiguousarray(np.asarray(z).T)).mean(1);val,ix=mu.min(0)
 return dict(C=float(torch.median(mu)-val),D=float(val),B=float(torch.median(mu)),offset=int(15-ix),windows=len(z),curve=mu.tolist())
def decode(p):
 import cv2
 cap=cv2.VideoCapture(str(p));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
 assert abs(fps-25)<1e-6
 while True:
  ok,x=cap.read()
  if not ok:break
  frames.append(x)
 cap.release();return np.asarray(frames)
def frame_digests(frames):return [hashlib.sha256(x.tobytes()).hexdigest() for x in frames]

class Runner:
 def __init__(self):self.start=time.monotonic()
 def cmd(self,args,name,cwd=ROOT,timeout=300,input=None):
  if time.monotonic()-self.start>3600:raise TimeoutError('one hour global limit')
  if shutil.disk_usage(OUT).free<5*1024**3:raise RuntimeError('below5GiBfree')
  log=OUT/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True);t=time.monotonic()
  with log.open('x') as f:
   f.write(json.dumps(list(map(str,args)))+'\n');f.flush()
   p=subprocess.run(list(map(str,args)),cwd=cwd,input=input,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
  put(OUT/'receipts'/f'{name}.json',dict(command=list(map(str,args)),cwd=str(cwd),returncode=p.returncode,elapsed_s=time.monotonic()-t,log_sha256=sha(log)))
  if p.returncode:raise RuntimeError('command failed '+name)
 def capture(self,args):return subprocess.check_output(list(map(str,args)))
 def prepare(self):
  OUT.mkdir(exist_ok=False);old=read(PREV/'video_manifest.json');ids=sorted({r['sample_id'] for r in old});inputs=[]
  for sid in ids:
   p={role:next(r for r in old if r['sample_id']==sid and r['role']==role) for role in ('N','T')}
   for role,r in p.items():
    for kind in ('audio','video'):assert sha(r[kind+'_path'])==r[kind+'_sha256']
    frames=decode(r['video_path']);r['video_frames']=len(frames);r['decoded_bgr_sha256']=frame_digests(frames)
    assert len(frames)<=r['samples']//640
   inputs.append(dict(sample_id=sid,N=p['N'],T=p['T'],output_frames=p['N']['video_frames']))
  pins={str(p):sha(p) for base in (MODEL/'upstream',SYNC) for p in base.rglob('*.py') if '__pycache__' not in p.parts}
  for p in [Path(__file__),MODEL/'weights/base_cogen_mirror.pt',MODEL/'upstream/config/base.yaml',MODEL/'autoavsr_detector.py',MODEL/'autoavsr_video_process.py',MODEL/'20words_mean_face.npy',SYNC/'data/syncnet_v2.model',SYNC/'detectors/s3fd/weights/sfd_face.pth',PREV/'protocol.json',PREV/'parent_independent_audit.json']:pins[str(p)]=sha(p)
  assert pins[str(MODEL/'weights/base_cogen_mirror.pt')]=='90f8c5165282c190c81c23d8fe92db2aa700862e021b1312fb96779caf9cfe61'
  protocol=dict(status='FROZEN_BEFORE_ALL_FORWARD',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection='same6 historical T>N sources; use fresh N/T Wav2Lip videos, no new generation',roles=ROLES,expected_scores=30,primary='CoGenAV audio(N) to video(T) cost1-ReLUcos, fixed DTW; compare with Nnative and matched Ncodec independently',controls='global normalized frame-count linear, CoGenAV audio(N)-audio(T)1-cos DTW, MFCC13CMVN1-cos DTW; no SyncNet based selection/tuning',features='pinned official CoGenAV Base strict-load FP32 eval separate audio-only/video-only pre-upsampler; same AutoAVSR MediaPipe96x96 RGB crop and upstream grayscale dot[.114,.587,.2989], mean0std1 /255, black pad750; Whisper80mel default pad30s; audio/video features sliced to respective native video counts',mfcc='librosa400FFT400Hann160hop80mel0-8kSlaney centered power2 logMel ref1 top_dbNone ->13MFCC; per utterance dim CMVN stdfloor1e-5; every4frames truncated nativevideo',dtw='closed endpoints; normalizedindexband.25; diagonal/horizontal/vertical; additive.05non-diagonal; tiesdiagonalfirst',projection='Nindex->median targetindex among path; anchor first0,lastm-1; nearest floor(x+.5),clamp[0,m-1]; no global lag correction',retiming='same output count as original N decoded frames,25fpszeroPTS; select decoded sourceT BGRframes without interpolation; lossless FFV1 bgr0 inMKV, original NPCM16 unchanged; Ncodec selects identity Nframes through same path',scoring='official unmodified run_pipeline min_track25 andrun_syncnet batch20 vshift15; firstsortedtrack; assert1trackframe0 andsamewindows; canonical Torch contiguous[31,windows]float32',analysis='source equal paired delta vs Ncodec and originalfreshNnative; six selectedsources descriptive; no bestmethod inference, perception claim, or general TTSbias conclusion',pins=pins,limits=dict(max_scores=30,global_seconds=3600,min_free_GiB=5,new_TTS=False,training=False),inputs=inputs)
  put(OUT/'protocol.json',protocol);put(OUT/'inputs.json',inputs);shutil.copyfile(__file__,OUT/'source_frozen.py')
  # Package exact helper sources; model weights remain pinned external inputs.
  for p in [MODEL/'autoavsr_detector.py',MODEL/'autoavsr_video_process.py',MODEL/'20words_mean_face.npy']:
   d=OUT/'frontend'/p.name;d.parent.mkdir(exist_ok=True);shutil.copyfile(p,d)
  shutil.copytree(MODEL/'upstream',OUT/'upstream',ignore=shutil.ignore_patterns('.git','__pycache__'))
  put(OUT/'runtime.json',dict(python=sys.version,numpy=np.__version__,soundfile=sf.__version__,GPU=self.capture(['nvidia-smi']).decode(),ffmpeg=self.capture(['ffmpeg','-version']).decode()))
  print('FROZEN',sha(OUT/'protocol.json'),flush=True)
 def features(self):
  import cv2,librosa,torch
  from transformers import WhisperFeatureExtractor
  from omegaconf import OmegaConf
  from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Config
  torch.set_num_threads(4);torch.manual_seed(20261025);np.random.seed(20261025)
  sys.path.insert(0,str(OUT/'upstream'))
  from models.cogenav import CoGenAV
  cfg=OmegaConf.merge(OmegaConf.structured(Wav2Vec2Config()),OmegaConf.load(OUT/'upstream/config/base.yaml'));OmegaConf.save(cfg,OUT/'base_resolved.yaml')
  model=CoGenAV(str(OUT/'base_resolved.yaml'),model_tensor=None);model.load_state_dict(torch.load(MODEL/'weights/base_cogen_mirror.pt',map_location='cpu',weights_only=True),strict=True);model.eval().cuda()
  det=module('avdetector',OUT/'frontend/autoavsr_detector.py').LandmarksDetector();cropper=module('avcrop',OUT/'frontend/autoavsr_video_process.py').VideoProcess(convert_gray=False);fe=WhisperFeatureExtractor()
  for row in read(OUT/'inputs.json'):
   out={};sid=row['sample_id'];directory=OUT/'features'/sid;directory.mkdir(parents=True)
   for role in ('N','T'):
    r=row[role];x,sr=sf.read(r['audio_path'],dtype='float32');assert sr==16000;n=r['video_frames'];mel=torch.from_numpy(fe(x,sampling_rate=sr).input_features[0])[None].cuda()
    with torch.inference_mode():out['audio_'+role]=model(None,mel,use_upsampler=False)[0,:n].cpu().numpy()
    power=librosa.feature.melspectrogram(y=x,sr=sr,n_fft=400,hop_length=160,win_length=400,n_mels=80,fmin=0,fmax=8000,center=True,power=2);lm=librosa.power_to_db(power,ref=1.,top_db=None);mf=librosa.feature.mfcc(S=lm,n_mfcc=13)[:,::4][:,:n].T
    out['mfcc_'+role]=((mf-mf.mean(0,keepdims=True))/np.maximum(mf.std(0,keepdims=True),1e-5)).astype(np.float32)
   frames=decode(row['T']['video_path']);rgb=np.asarray([cv2.cvtColor(x,cv2.COLOR_BGR2RGB) for x in frames]);land=det(rgb);detected=sum(x is not None for x in land);assert detected/len(frames)>=.95
   patches=cropper(rgb,land);assert patches.shape==(len(frames),96,96,3);gray=np.dot(patches[...,:3],[.114,.587,.2989]).astype(np.uint8);padded=np.zeros((750,96,96),np.float32);padded[:len(gray)]=gray/255.
   video=torch.from_numpy(padded)[None,...,None].permute(0,2,3,4,1).cuda()
   with torch.inference_mode():out['video_T']=model(video,None,use_upsampler=False)[0,:len(frames)].cpu().numpy()
   np.savez_compressed(directory/'features.npz',**out)
   np.savez_compressed(directory/'crop_and_landmarks.npz',gray=gray,landmarks=np.asarray(land),detected=detected)
   grid=np.concatenate([patches[k] for k in np.linspace(0,len(patches)-1,8).astype(int)],axis=1);cv2.imwrite(str(directory/'mouth_montage.jpg'),cv2.cvtColor(grid,cv2.COLOR_RGB2BGR))
   print('FEATURE',sid,flush=True)
  put(OUT/'model_load.json',dict(strict=True,torch=torch.__version__,librosa=librosa.__version__,weight_sha256=sha(MODEL/'weights/base_cogen_mirror.pt'),resolved_config_sha256=sha(OUT/'base_resolved.yaml'),forward_calls=18))
 def align_render(self):
  rows=[]
  for row in read(OUT/'inputs.json'):
   sid=row['sample_id'];n=row['N']['video_frames'];m=row['T']['video_frames'];f=np.load(OUT/'features'/sid/'features.npz');maps={};alignment={}
   for name,a,b,relu in [('cogenav_av','audio_N','video_T',True),('cogenav_aa','audio_N','audio_T',False),('mfcc_aa','mfcc_N','mfcc_T',False)]:
    cost=cosine_cost(f[a],f[b],relu);path=dtw(cost);pred,indices=project(path,n,m);maps[name]=indices
    alignment[name+'_cost']=cost;alignment[name+'_path']=path;alignment[name+'_prediction']=pred;alignment[name+'_indices']=indices
   pred=np.linspace(0,m-1,n);maps['linear']=nearest(pred,m);alignment['linear_prediction']=pred;alignment['linear_indices']=maps['linear'];maps['N_codec']=np.arange(n)
   np.savez_compressed(OUT/'features'/sid/'alignment.npz',**alignment)
   nt={r:decode(row[r]['video_path']) for r in ('N','T')}
   for role in ROLES:
    key=sid+'_'+role;source=nt['N' if role=='N_codec' else 'T'];frames=source[maps[role]];h,w=frames.shape[1:3];dest=OUT/'video'/sid/f'{role}.mkv';dest.parent.mkdir(parents=True,exist_ok=True)
    self.cmd(['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pixel_format','bgr24','-video_size',f'{w}x{h}','-framerate','25','-i','pipe:0','-i',row['N']['audio_path'],'-map','0:v:0','-map','1:a:0','-c:v','ffv1','-level','3','-pix_fmt','bgr0','-c:a','pcm_s16le',dest],'render_'+key,input=frames.tobytes())
    decoded=decode(dest);assert np.array_equal(decoded,frames)
    raw=self.capture(['ffmpeg','-v','error','-i',dest,'-map','0:a:0','-ac','1','-ar','16000','-f','s16le','-']);original=sf.read(row['N']['audio_path'],dtype='int16')[0].astype('<i2').tobytes();assert raw==original
    probe=json.loads(self.capture(['ffprobe','-v','error','-show_streams','-show_format','-of','json',dest]));put(OUT/'probes'/f'{key}.json',probe)
    stream=next(x for x in probe['streams'] if x['codec_type']=='video');assert stream['avg_frame_rate']=='25/1' and float(stream['start_time'])==0
    rec=dict(sample_id=sid,role=role,video_path=str(dest),video_sha256=sha(dest),audio_path=row['N']['audio_path'],audio_sha256=row['N']['audio_sha256'],pcm_sha256=hashlib.sha256(raw).hexdigest(),audio_samples=len(raw)//2,output_frames=n,source_role='N' if role=='N_codec' else 'T',source_video_path=row['N' if role=='N_codec' else 'T']['video_path'],source_video_sha256=row['N' if role=='N_codec' else 'T']['video_sha256'],indices=maps[role].tolist(),decoded_frame_sha256=frame_digests(decoded),pcm_exact=True,selected_frames_exact=True)
    rows.append(rec);print('RETIME',key,flush=True)
  put(OUT/'video_manifest.json',rows)
 def score(self):
  rows=[]
  for row in read(OUT/'video_manifest.json'):
   sid,role=row['sample_id'],row['role'];key=sid+'_'+role;ref='cvr25_'+key
   self.cmd([SPY,SYNC/'run_pipeline.py','--videofile',row['video_path'],'--reference',ref,'--data_dir',OUT/'data','--min_track','25','--overwrite'],'pipeline_'+key,SYNC)
   self.cmd([SPY,SYNC/'run_syncnet.py','--videofile',row['video_path'],'--reference',ref,'--data_dir',OUT/'data','--initial_model',SYNC/'data/syncnet_v2.model'],'syncnet_'+key,SYNC)
   ap=OUT/'data/pywork'/ref/'activesd.pckl';dists=pickle.loads(ap.read_bytes());assert len(dists)==1;z=np.asarray(dists[0]);assert z.dtype==np.float32 and z.shape[1]==31
   tracks=pickle.loads((OUT/'data/pywork'/ref/'tracks.pckl').read_bytes());assert len(tracks)==1 and int(tracks[0]['track']['frame'][0])==0
   item=OUT/'items'/key;item.mkdir(parents=True);np.savez_compressed(item/'distance.npz',distance=z);actual=OUT/'data/pytmp'/ref/'audio.wav';shutil.copyfile(actual,item/'scoring_audio.wav')
   rec=dict(row,**scalar(z),track_count=1,track_start_frame=0,track_end_frame=int(tracks[0]['track']['frame'][-1]),reference=ref,activesd_path=str(ap),activesd_sha256=sha(ap),distance_sha256=sha(item/'distance.npz'),scoring_audio_path=str(item/'scoring_audio.wav'),scoring_audio_sha256=sha(item/'scoring_audio.wav'))
   put(OUT/'scores'/f'{key}.json',rec);rows.append(rec);print('SCORE',key,rec['C'],flush=True)
  put(OUT/'scores.json',rows);prev=read(PREV/'parent_independent_audit.json');old=read(PREV/'scores.json');table=[]
  # Recompute old official distance arrays with the same canonical layout, not rounded log scores.
  for sid in sorted({r['sample_id'] for r in rows}):
   rr=[r for r in rows if r['sample_id']==sid];assert len({r['windows'] for r in rr})==1
   vals={r['role']:r['C'] for r in rr};native={r:scalar(np.load(PREV/'items'/f'{sid}_{r}'/'distance.npz')['distance'])['C'] for r in ('N','T')}
   table.append(dict(sample_id=sid,**vals,N_native=native['N'],T_native=native['T'],delta_vs_Ncodec={r:vals[r]-vals['N_codec'] for r in ROLES[1:]},delta_vs_Nnative={r:vals[r]-native['N'] for r in ROLES[1:]},codec_delta=vals['N_codec']-native['N'],windows=rr[0]['windows']))
  means={r:float(np.mean([x[r] for x in table])) for r in (*ROLES,'N_native','T_native')};effects={base:{r:dict(mean=float(np.mean([x[base][r] for x in table])),positive=sum(x[base][r]>0 for x in table),values=[x[base][r] for x in table]) for r in ROLES[1:]} for base in ('delta_vs_Ncodec','delta_vs_Nnative')}
  put(OUT/'summary.json',dict(status='COMPLETE',sources=6,scores=30,protocol_sha256=sha(OUT/'protocol.json'),means=means,effects=effects,per_source=table))
  lines=['# CoGenAV feature/DTW retiming TTS video to natural audio','','Our adaptation of official CoGenAV features, not an original paper retiming algorithm. Fixed30-case pilot; no score-guided tuning. All videos scored against exactly unchanged naturalPCM.','','|Source|N native|N codec|T native|linear|CoGenAV AV|CoGenAV AA|MFCC AA|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
  for x in table:lines.append('|'+x['sample_id']+'|'+'|'.join(f'{x[k]:.3f}' for k in ('N_native','N_codec','T_native','linear','cogenav_av','cogenav_aa','mfcc_aa'))+'|')
  lines.extend(['','Means: '+json.dumps(means),'','Effects: '+json.dumps(effects),'','Six sources preselected on historical T>N; no population inference. NativeT scores useTaudio and provide contextual advantage only. Retimed scores use natural audio. DTW path lacks phonetic ground truth. Nearest-frame repeats/skips may alter motion and metric. Sync-C gain is evaluator evidence, not perceptual proof. NaturalPCM full unchanged, output frames match freshNvideo; waveform trailing2–3frame tail not expanded. All scored supports match within source.'])
  (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 def run(self):
  try:self.prepare();self.features();self.align_render();self.score()
  except BaseException as exc:
   import traceback
   if OUT.exists():put(OUT/'terminal_failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
   raise
if __name__=='__main__':Runner().run()
