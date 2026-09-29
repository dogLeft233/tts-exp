"""Frozen natural80 single-input FIXED GxE, old runtime and exact RAW replay gate."""
from pathlib import Path
import argparse,copy,hashlib,importlib.metadata,json,os,shutil,sys,time,types
import numpy as np,soundfile as sf
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/tts_fixed_natural_holdout80_20260927';OLD=ROOT/'runs/tts_native_holdout_confirmation_20260926';FEAS=ROOT/'runs/tts_fixed_natural_holdout80_feasibility_20260927';FIX=ROOT/'runs/tts_fixed_level_generation_cross_20260927';CAL=ROOT/'runs/tts_native_level_match_calibration_20260927';ENGINE=ROOT/'runs/tts_acoustic_generation_cross_20260926/code_snapshot/tts_acoustic_generation_cross_20260926.py'
POLICIES=['guard20','valid','guard0','guard15'];GEOMETRIES=['raw','unit'];FIELDS=['C','B','D','C_anchor','D_anchor','best_lag'];CELLS={'q00':('RAW','RAW'),'q10':('FIXED','RAW'),'q01':('RAW','FIXED'),'q11':('FIXED','FIXED')}
def load(name,path):
 m=types.ModuleType(name);m.__file__=str(ROOT/'scripts/experiments'/Path(path).name);exec(compile(Path(path).read_text(),str(path),'exec'),m.__dict__);return m
base=load('natural80_engine',ENGINE);base.OUT=OUT
cal=load('fixed_scalar_cal',CAL/'code_snapshot/tts_native_level_match_calibration.py');spectral=cal.parent_module()
read,write,sha=base.read,base.write,base.sha
W2L=base.W2L

def allocation():
 seen=set();n=0
 for f in OUT.rglob('*'):
  if f.is_file():
   s=f.stat();k=(s.st_dev,s.st_ino)
   if k not in seen and s.st_nlink==1:n+=s.st_blocks*512
   seen.add(k)
 return n

def disk_check():
 free=shutil.disk_usage(OUT).free;n=allocation();assert free>=5<<30,('disk reserve',free);assert n<=int(.20*2**30),('new allocation budget',n)
 return {'free_bytes':free,'new_allocated_bytes':n}

def resource_check(p,startup=False):
 r=base.compute_processes();assert all(x['pid']==os.getpid() for x in r),('foreign compute',r)
 used,util=map(int,base.subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
 if startup:assert used<=128 and util<=10,(used,util)
 return {'time':time.time(),'compute':r,'memory_mib':used,'utilization':util,**disk_check()}
base.resource_check=resource_check

def environment():
 return {'python':sys.executable,'versions':{k:importlib.metadata.version(k) for k in ['torch','numpy','scipy','librosa','python_speech_features','soundfile']}}

def protocol():
 p=base.protocol();assert environment()==p['runtime'],('wrong runtime',environment(),p['runtime'])
 for f,h in p['new_code_hashes'].items():assert sha(f)==h
 return p

def freeze():
 assert not (OUT/'protocol.json').exists();assert read(FEAS/'independent_validation.json')['status']=='PASS';f=read(FEAS/'summary.json');assert environment()==f['old_runtime']
 old=read(OLD/'protocol.json');fixed=read(FIX/'protocol.json');rows=old['rows'];assert len(rows)==80 and len({r['speaker'] for r in rows})==40
 deps=copy.deepcopy(fixed['dependencies']);deps.update(read(FEAS/'input_seal.json')['hashes'])
 for path in [FEAS/'report.md',FEAS/'final.json',FEAS/'rows.json',FEAS/'independent_validation.json',OLD/'feature_records.json',OLD/'feature_runtime.json',OLD/'support.json',CAL/'code_snapshot/tts_native_level_match_calibration.py',CAL/'code_snapshot/check_tts_native_level_match_calibration.py',ENGINE]:deps[str(path)]=sha(path)
 files=[Path(__file__),ROOT/'scripts/experiments/check_tts_fixed_natural_holdout80_20260927.py']
 p={'status':'frozen_before_new_waveforms_features_scores','created_epoch':time.time(),'root_approval':'Explicitly approved complete CPU feasibility proposal; GPU waits visual final release; no further approval after engineering PASS','proposal_sha256':sha(FEAS/'report.md'),'rows':rows,'image':old['images'][0],'runtime':environment(),'target_RMS':.0376838172675746,'target_source':fixed['fixed_target'],'conditions':['RAW','identity','FIXED'],'identity':'exact alias of original RAW PCM/32768 and cached model frontend; no STFT or amplitude change','acoustics':fixed['acoustics'],'primary':'G=q10-q00 with unchanged original natural evaluation audio; all80/40speaker','cells':CELLS,'scoring':{'L':'min(F,PCM//640)-5, allsame within utterance','lags':list(range(-15,16)),'k':3,'policies':POLICIES,'geometries':GEOMETRIES,'fields':FIELDS,'distance':'torch pairwise_distance float32 eps1e-6, official zero padded31lags; MFCC float64 amplitude*32768 to float32 windows'},'statistics':{'bootstrap_draws':20000,'seed':20260926,'CI':.99,'aggregation':'2 utterance effects averaged within speaker then equal40speaker','effects':['G','E','I','total'],'FWER':False,'support':'all80 natural inputs; all frozen views; no Q/TTS dependent selection'},'generation':{'fps':25,'codec':'FFV1','batch':32,'seed':20260926,'source':'static image3','boxes':'frozen old holdout image metadata','audio':'FLOAT32 waveform no new time resampling; own original PCM time axis','score_video':'224 ROI then original SyncNet JPEG stream'},'controls':{'ids':[r['id'] for r in rows[:2]],'RAW':'first2 fresh pixel and A/V exact, PCM/FLOAT windows and audio exact; fail immediately stop; no automatic RAW full rerender or tolerance adjustment','delay':'+3200samples same length on first2 raw PCM, common query25:L-25, curve overlap<=.15/lag shift5±1; all k remains3','FIXED_stream':'first2 full and streamed repeat pixel/MFCC/A/V/query exact; then continue78','lag_transfer':'all80 diagonal medianlags abs(median-3)<=1 and RAW/FIXED range<=1; diagnostic limitation only no k/support changes'},'resources':{'lease':'/tmp/tts-exp-gpu.lock','startup':'three5s snapshots no foreigncompute, <=128MiB/10percentutil','budget_bytes':int(.20*2**30),'reserve_bytes':5<<30,'estimated_GiB':f['storage_upper_estimate']['GiB'],'watchdog':'disk/budget checked every32frame render batch and each persisted cell'},'retention':{'keep_new_full_videos':[r['id'] for r in rows[:2]],'others':'78FIXED and RAW/FIXED control duplicate newtemp videos removed only after pixel/file hashes, A/V/frontend/waveform, metadata atomic commit and reread; all existing files untouched','not_retained_video_claim':'creation-time hash only, not final existing-file hash verification','z':False},'independence':{'scope':'40canonical audio speakers and80PCM outside currentcloud100/old26cal development; not absolute allproject human or modeltraining independence','earlier_use':'80previously analyzed for other endpoints; not wholly unseen data','not_claimed':'cloud TTS-N gap, true lip quality, newface generalization'},'dependencies':deps,'new_code_hashes':{str(x):sha(x) for x in files}}
 assert p['target_RMS']==fixed['fixed_target']['target_RMS']
 for path in files:
  dest=OUT/'code_snapshot'/path.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
 write(OUT/'protocol.json',p);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n');print('FROZEN',sha(OUT/'protocol.json'),flush=True)

def prepare():
 p=protocol();assert not (OUT/'support.json').exists();sys.path.insert(0,str(W2L));import audio
 from scripts.experiments.static_image_bridge.render_worker import chunk_mels
 from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
 ff={r['id']:r for r in read(FEAS/'rows.json')};old={(r['id'],r['arm']):r for r in read(OLD/'feature_records.json')};support=[];checks=[];reuse=[]
 for r in p['rows']:
  disk_check();sid=r['id'];src=r['audio']['N'];assert sha(src['path'])==src['sha256'];pcm,sr=sf.read(src['path'],dtype='int16');assert sr==16000 and sf.info(src['path']).subtype=='PCM_16';x=pcm.astype(float)/32768;rr=cal.rms(x);assert rr>1e-8;g=p['target_RMS']/rr;y=x*g
  path=OUT/'audio'/(sid+'.wav');path.parent.mkdir(parents=True,exist_ok=True);sf.write(path,y,16000,subtype='FLOAT');stored=sf.read(path,dtype='float64')[0];assert np.isfinite(y).all() and np.isfinite(stored).all() and len(stored)==len(x) and max(abs(y).max(),abs(stored).max())<=.98
  f0=cal.features(spectral,x);fy=cal.features(spectral,y);fs=cal.features(spectral,stored);iv64=cal.invariant(x,y,f0,fy,g);iv32=cal.invariant(x,stored,f0,fs,g);relative=abs(cal.rms(stored)/p['target_RMS']-1)
  good={'target':relative<=1e-5,'float64':iv64['shape_max_abs_db']<=1e-8 and iv64['R_rms_abs_db']<=1e-8 and iv64['cosine_error']<=1e-10 and iv64['zero_preserved'],'saved':iv32['divide_gain_max_abs']<=1e-7 and iv32['shape_max_abs_db']<=1e-4 and iv32['R_rms_abs_db']<=1e-5 and iv32['cosine_error']<=1e-10 and iv32['zero_preserved']}
  oldw,_=SyncNetEngine._audio_windows(None,pcm);raw_w,_=base.frontend_windows(x);assert np.array_equal(oldw,raw_w)
  rawmel=audio.melspectrogram(audio.load_wav(src['path'],16000)).astype(np.float32);identitymel=audio.melspectrogram(x.astype(np.float32)).astype(np.float32);assert np.array_equal(rawmel,identitymel)
  mel=audio.melspectrogram(stored.astype(np.float32)).astype(np.float32);windows,_=base.frontend_windows(stored);frames=len(chunk_mels(mel,25));L=min(frames,len(x)//640)-5;assert frames==len(chunk_mels(rawmel,25))==ff[sid]['frames'] and L==ff[sid]['L'] and L>40
  front=OUT/'frontend'/(sid+'.npz');front.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(front,mel=mel,windows=windows)
  oo=old[sid,'N'];assert sha(oo['path'])==oo['sha256'] and sha(oo['video_path'])==oo['video_sha256'];dest=OUT/'features'/sid/'RAW.npz';dest.parent.mkdir(parents=True,exist_ok=True);os.link(oo['path'],dest)
  write(dest.with_suffix('.json'),{'id':sid,'condition':'RAW','features':str(dest),'sha256':sha(dest),'source_metadata':str(OLD/'features'/sid/'N.json'),'source_metadata_sha256':sha(OLD/'features'/sid/'N.json'),'video':{'path':oo['video_path'],'sha256':oo['video_sha256'],'pixel_sha256':ff[sid]['decoded_pixel_sha256'],'frames':frames},'retained_video':True,'cached_old':True})
  row={'id':sid,'speaker':r['speaker'],'source':src,'FIXED':{'waveform':str(path),'sha256':sha(path),'frontend':str(front),'frontend_sha256':sha(front),'frames':frames,'samples':len(x),'audio_windows':len(windows)},'L':L,'raw_frontend_hashes':{'mel':hashlib.sha256(rawmel.tobytes()).hexdigest(),'MFCC_windows':hashlib.sha256(raw_w.tobytes()).hexdigest()},'gain':g,'original_RMS':rr,'target_RMS':p['target_RMS']};support.append(row);checks.append({'id':sid,'float64':iv64,'float32':iv32,'target_relative':relative,'peak':float(max(abs(y).max(),abs(stored).max())),'checks':good,'passed':all(good.values())});reuse.append({'id':sid,'source_metadata':str(OLD/'features'/sid/'N.json'),'source_metadata_sha256':sha(OLD/'features'/sid/'N.json'),'feature_sha256':oo['sha256'],'video_sha256':oo['video_sha256'],'CPU_eligible':True,'GPU_replay_pending':True})
  print('prepared',sid,flush=True)
 write(OUT/'support.json',support);write(OUT/'acoustic_gate.json',{'passed':all(r['passed'] for r in checks),'rows':checks});write(OUT/'reuse_manifest.json',reuse)

def seal():
 p=protocol();assert read(OUT/'acoustic_gate.json')['passed'] and read(OUT/'independent_acoustic.json')['status']=='PASS';assert not list((OUT/'features').rglob('FIXED.npz'))
 write(OUT/'cpu_seal.json',{'status':'before_new_GPU_features_scores','protocol_sha256':sha(OUT/'protocol.json'),'support_sha256':sha(OUT/'support.json'),'acoustic_sha256':sha(OUT/'acoustic_gate.json'),'independent_acoustic_sha256':sha(OUT/'independent_acoustic.json'),'code':p['new_code_hashes'],'time':time.time()})

def decode_hash(path):
 import cv2
 cap=cv2.VideoCapture(str(path));d=hashlib.sha256();n=0
 while True:
  ok,f=cap.read()
  if not ok:break
  d.update(f.tobytes());n+=1
 cap.release();return d.hexdigest(),n

def render(model,chunks,frame,it,im,path):
 import cv2,torch
 from scripts.experiments.static_image_bridge.score_worker import crop_zero_padded
 path.parent.mkdir(parents=True,exist_ok=True);writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),25,(224,224));assert writer.isOpened();d=hashlib.sha256();x1,y1,x2,y2=im['generation_box']
 try:
  for start in range(0,len(chunks),32):
   disk_check();mt=torch.from_numpy(np.asarray(chunks[start:start+32],dtype=np.float32)[:,None]).cuda()
   with torch.inference_mode():pred=model(mt,it.expand(len(mt),-1,-1,-1)).cpu().numpy().transpose(0,2,3,1)*255
   for fo in pred:
    full=frame.copy();full[y1:y2,x1:x2]=cv2.resize(fo.astype(np.uint8),(x2-x1,y2-y1));crop=crop_zero_padded(full,im['score_box']['box']);writer.write(crop);d.update(crop.tobytes())
 finally:writer.release()
 dh,n=decode_hash(path);assert dh==d.hexdigest() and n==len(chunks)
 return {'path':str(path),'sha256':sha(path),'pixel_sha256':dh,'frames':n,'decoded_pixel_equals_render':True}

def persist(dest,visual,audio,meta):
 dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.pending.npz');np.savez_compressed(tmp,visual=visual,audio=audio);os.replace(tmp,dest.with_suffix('.npz'));z=np.load(dest.with_suffix('.npz'));assert np.array_equal(z['visual'],visual) and np.array_equal(z['audio'],audio) and np.isfinite(visual).all() and np.isfinite(audio).all()
 meta.update(features=str(dest.with_suffix('.npz')),sha256=sha(dest.with_suffix('.npz')),persisted_arrays_exact=True);tmp=dest.with_suffix('.pending.json');write(tmp,meta);os.replace(tmp,dest.with_suffix('.json'));return meta

def remove_committed(video,meta_path,commit_path):
 p=Path(video['path']);assert p.is_relative_to(OUT/'temporary') and sha(p)==video['sha256'];m=read(meta_path);assert sha(m['features'])==m['sha256'] and m['persisted_arrays_exact']
 write(commit_path,{'video_sha256_at_creation':video['sha256'],'decoded_pixel_sha256_at_creation':video['pixel_sha256'],'metadata_sha256':sha(meta_path),'features_sha256':m['sha256'],'video_not_retained':True,'time':time.time()});p.unlink()

def summarize(m,policy):
 L=len(m)
 if policy=='valid':curve=np.array([m[max(0,-lag):min(L,L-lag),lag+15].mean(0) for lag in range(-15,16)])
 else:
  g=int(policy[5:]);assert L>2*g;curve=m[g:L-g].mean(0) if g else m.mean(0)
 j=int(np.argmin(curve));B=float(np.median(curve));D=float(curve[j]);anchor=float(curve[18]);return {'C':B-D,'B':B,'D':D,'C_anchor':B-anchor,'D_anchor':anchor,'best_lag':j-15,'curve':curve.tolist()}

def produce():
 import cv2,torch
 from scripts.experiments.masked_tts_tfg_probe.direct_mel import _load_model
 from scripts.experiments.static_image_bridge.render_worker import chunk_mels
 from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
 p=protocol();lock=read(OUT/'cpu_seal.json');assert sha(OUT/'support.json')==lock['support_sha256'];assert read(OUT/'independent_acoustic.json')['status']=='PASS'
 cv2.setNumThreads(1);torch.set_num_threads(2);torch.manual_seed(20260926);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 rows=read(OUT/'support.json');first=p['controls']['ids'];controls=[]
 with base.lease('produce',p):
  write(OUT/'runtime.json',environment());np.savez_compressed(OUT/'rng_before.npz',torch_cpu=torch.get_rng_state().numpy(),torch_cuda=torch.cuda.get_rng_state().cpu().numpy())
  model=_load_model(W2L/'checkpoints/wav2lip_gan.pth','cuda');engine=SyncNetEngine(batch_size=32,device='cuda');im=p['image'];frame=cv2.imread(im['path']);x1,y1,x2,y2=im['generation_box'];face=cv2.resize(frame[y1:y2,x1:x2],(96,96));mask=face.copy();mask[48:]=0;it=torch.from_numpy(np.concatenate([mask,face],2).transpose(2,0,1).astype(np.float32)/255).unsqueeze(0).cuda()
  try:
   sys.path.insert(0,str(W2L));import audio
   # Both RAW controls must complete before the first FIXED video is rendered.
   for row in rows[:2]:
    resource_check(p);sid=row['id'];pcm=sf.read(row['source']['path'],dtype='int16')[0];x=pcm.astype(float)/32768;windows,_=base.frontend_windows(x);oldwindows,_=engine._audio_windows(pcm);mel=audio.melspectrogram(x.astype(np.float32)).astype(np.float32)
    assert hashlib.sha256(mel.tobytes()).hexdigest()==row['raw_frontend_hashes']['mel'] and hashlib.sha256(windows.tobytes()).hexdigest()==row['raw_frontend_hashes']['MFCC_windows']
    ae=base.audio_forward(engine,windows);oldae,_=engine.extract_audio(row['source']['path']);video=render(model,chunk_mels(mel,25),frame,it,im,OUT/'temporary'/'RAW_repeat'/(sid+'.avi'));ve,vm=engine.extract_visual(video['path']);saved=np.load(OUT/'features'/sid/'RAW.npz');rawmeta=read(OUT/'features'/sid/'RAW.json')
    tests={'pixel_exact':video['pixel_sha256']==rawmeta['video']['pixel_sha256'],'audio_exact':bool(np.array_equal(ae,saved['audio'])),'visual_exact':bool(np.array_equal(ve,saved['visual'])),'PCM_FLOAT_MFCC_exact':bool(np.array_equal(windows,oldwindows)),'PCM_FLOAT_A_exact':bool(np.array_equal(ae,oldae)),'frames_L_exact':video['frames']==row['FIXED']['frames'] and min(video['frames'],len(x)//640)-5==row['L']}
    d=OUT/'controls'/sid/'RAW_repeat';meta=persist(d,ve,ae,{'id':sid,'kind':'RAW_repeat','video':video,'retained_video':False,'tests':tests,'passed':all(tests.values()),'max_A_error':float(abs(ae-saved['audio']).max()),'max_V_error':float(abs(ve-saved['visual']).max())})
    if not all(tests.values()):write(OUT/'engineering_gate.json',{'passed':False,'failed_id':sid,'tests':tests,'action':'STOP no FIXED render or scores; no automatic rerender/retuning'});raise AssertionError(('RAW exact gate failed',sid,tests))
    remove_committed(video,d.with_suffix('.json'),OUT/'retention_commits'/'controls'/sid/'RAW_repeat.json')
    shifted=np.zeros_like(x);shifted[3200:]=x[:-3200];sw,_=base.frontend_windows(shifted);da=base.audio_forward(engine,sw);cp=OUT/'controls'/sid/'delay5.npz';np.savez_compressed(cp,delay_audio=da);L=row['L'];q=slice(25,L-25);bm=base.matrix(saved['visual'][:L],saved['audio'][:L]);dm=base.matrix(saved['visual'][:L],da[:L]);bc=bm[q].mean(0);dc=dm[q].mean(0);b=int(np.argmin(bc))-15;dlag=int(np.argmin(dc))-15;err=float(abs(dc[5:]-bc[:-5]).max());delay={'base_lag':b,'delay_lag':dlag,'curve_overlap_max':err,'query':[25,L-25],'passed':abs(dlag-b-5)<=1 and err<=.15,'array_sha256':sha(cp),'shifted_waveform_sha256':hashlib.sha256(shifted.astype(np.float32).tobytes()).hexdigest()};write(OUT/'controls'/sid/'delay5.json',delay);assert delay['passed'];controls.append({'id':sid,'RAW':tests,'delay':delay})
   write(OUT/'engineering_gate.json',{'passed':True,'controls':controls,'before_any_FIXED_video':True,'fixed_k':3})
   for row in rows:
    resource_check(p);sid=row['id'];dest=OUT/'features'/sid/'FIXED';info=row['FIXED'];assert sha(info['waveform'])==info['sha256'] and sha(info['frontend'])==info['frontend_sha256'];front=np.load(info['frontend']);ae=base.audio_forward(engine,front['windows']);chunks=chunk_mels(front['mel'],25);keep=sid in first;vp=OUT/('videos' if keep else 'temporary/FIXED')/(sid+'.avi');video=render(model,chunks,frame,it,im,vp);ve,vm=engine.extract_visual(vp);assert min(len(ae),len(ve))>=row['L'] and video['frames']==info['frames']
    meta=persist(dest,ve,ae,{'id':sid,'condition':'FIXED','video':video,'retained_video':keep,'audio_frontend':info,'joint_L':row['L'],'visual_metadata':vm,'protocol_sha256':sha(OUT/'protocol.json')})
    if not keep:remove_committed(video,dest.with_suffix('.json'),OUT/'retention_commits'/'FIXED'/(sid+'.json'))
    if keep:
     tv=render(model,chunks,frame,it,im,OUT/'temporary'/'FIXED_stream_control'/(sid+'.avi'));vv,vvm=engine.extract_visual(tv['path']);aa=base.audio_forward(engine,front['windows']);xx=sf.read(info['waveform'],dtype='float64')[0];ww,_=base.frontend_windows(xx);tests={'pixel_exact':tv['pixel_sha256']==video['pixel_sha256'],'A_exact':bool(np.array_equal(aa,ae)),'V_exact':bool(np.array_equal(vv,ve)),'MFCC_exact':bool(np.array_equal(ww,front['windows'])),'query_exact':tv['frames']==video['frames'] and min(tv['frames'],len(xx)//640)-5==row['L']};dd=OUT/'controls'/sid/'FIXED_stream';persist(dd,vv,aa,{'id':sid,'kind':'FIXED_stream','video':tv,'retained_video':False,'tests':tests,'passed':all(tests.values())});assert all(tests.values());remove_committed(tv,dd.with_suffix('.json'),OUT/'retention_commits'/'controls'/sid/'FIXED_stream.json')
    disk_check();print('produced',sid,flush=True)
  finally:engine.close()
 write(OUT/'gpu_release.json',{'time':time.time(),'pid':os.getpid(),'lease_released':True,'FIXED_cells':80,'retained_FIXED_videos':2,'streamed_FIXED_videos':78,**disk_check()})

def score():
 import torch
 torch.set_num_threads(2);p=protocol();assert read(OUT/'engineering_gate.json')['passed'];assert all(read(OUT/'controls'/sid/'FIXED_stream.json')['passed'] for sid in p['controls']['ids']);assert read(OUT/'gpu_release.json')['FIXED_cells']==80
 seals={str(f):sha(f) for f in (OUT/'features').rglob('*') if f.is_file()};write(OUT/'feature_seal.json',seals);write(OUT/'score_lock.json',{'time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'feature_seal_sha256':sha(OUT/'feature_seal.json'),'support_sha256':sha(OUT/'support.json'),'status':'before_all80_new_GxE_distance_scores'})
 lags={g:{c:[] for c in ['RAW','FIXED']} for g in GEOMETRIES}
 for row in read(OUT/'support.json'):
  disk_check();sid=row['id'];L=row['L'];fs={c:np.load(OUT/'features'/sid/(c+'.npz')) for c in ['RAW','FIXED']};matrices={};cells={}
  for geom in GEOMETRIES:
   a={c:(base.unit(f['audio'][:L]) if geom=='unit' else f['audio'][:L]) for c,f in fs.items()};v={c:(base.unit(f['visual'][:L]) if geom=='unit' else f['visual'][:L]) for c,f in fs.items()};cells[geom]={}
   for name,(vc,ac) in CELLS.items():
    m=base.matrix(v[vc],a[ac]);matrices[geom+'__'+name]=m;cells[geom][name]={'policies':{pol:summarize(m,pol) for pol in POLICIES},'bridge':base.covariance_features(v[vc],a[ac],L)}
   for c,q in [('RAW','q00'),('FIXED','q11')]:lags[geom][c].append(cells[geom][q]['policies']['guard20']['best_lag'])
  dest=OUT/'scores'/sid;dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest.with_suffix('.npz'),**matrices);write(dest.with_suffix('.json'),{'id':sid,'speaker':row['speaker'],'L':L,'cells':cells,'matrices_sha256':sha(dest.with_suffix('.npz'))});print('scored',sid,flush=True)
 med={g:{c:float(np.median(v)) for c,v in vals.items()} for g,vals in lags.items()};raw=med['raw'];write(OUT/'lag_transfer.json',{'fixed_k':3,'medians':med,'raw_passed':all(abs(v-3)<=1 for v in raw.values()) and max(raw.values())-min(raw.values())<=1,'diagnostic_only':True,'no_change_to_k_support_target':True})

def analyze():
 p=protocol();rows=[read(f) for f in sorted((OUT/'scores').glob('*.json'))];assert len(rows)==80;groups=[r['speaker'] for r in rows];assert len(set(groups))==40;result={};effects=[];closure=0.
 for geom in GEOMETRIES:
  for pol in POLICIES:
   view={}
   for metric in FIELDS:
    qq=np.array([[r['cells'][geom][q]['policies'][pol][metric] for q in ['q00','q10','q01','q11']] for r in rows]);ee=[base.effects(q) for q in qq];d={'baseline':base.bootstrap(qq[:,0],groups),'FIXED_native':base.bootstrap(qq[:,3],groups),'effects':{e:base.bootstrap([x[e] for x in ee],groups) for e in ['G','E','I','total']}};view[metric]=d
    for r,q,e in zip(rows,qq,ee):closure=max(closure,abs(e['G']+e['E']+e['I']-e['total']));effects.append({'id':r['id'],'speaker':r['speaker'],'geometry':geom,'policy':pol,'metric':metric,'q00':q[0],'q10':q[1],'q01':q[2],'q11':q[3],**e})
   result[geom+'/'+pol]=view
 write(OUT/'summary.json',{'results':result,'primary':'raw/guard20 C G','max_closure':closure,'utterances':80,'speakers':40,'lag_transfer':read(OUT/'lag_transfer.json')});write(OUT/'effects.json',effects);print(json.dumps(result['raw/guard20']['C'],indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','prepare','seal','produce','score','analyze']);args=ap.parse_args();globals()[args.stage]()
