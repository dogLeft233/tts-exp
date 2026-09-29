"""Prospectively frozen GRID label evaluation, staged input-only selection and paired outputs."""
from pathlib import Path
import argparse,contextlib,fcntl,hashlib,importlib.metadata,json,math,os,resource,shutil,subprocess,sys,time,wave
import numpy as np
import grid_geometry_calibration_20260927 as g
import check_grid_geometry_calibration_20260927 as geom
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/grid_shape_correlation_evaluation_20260927';TMP=Path('/dev/shm/tts_exp_grid16eval_20260927')
VIEWS=['BASE','TX_-8','TX_8','TY_-8','TY_8','ROT_-5','ROT_5','SCALE_0.9','SCALE_1.1'];DOMAINS=['REAL','RAW','FIXED'];LAGS=[0,-3];T=np.arange(14,55)
read=g.read;sha=g.sha;ah=g.ah
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);limits()
 with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x));f.write('\n')
def limits():
 size=sum(f.stat().st_blocks*512 for f in OUT.rglob('*') if f.is_file());free=shutil.disk_usage(OUT).free
 assert size<=40*2**20 and free-max(0,40*2**20-size)-8*2**20>=5*2**30,('disk',size,free)
 tmp=sum(f.stat().st_blocks*512 for f in TMP.rglob('*') if f.is_file()) if TMP.exists() else 0
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;child=g.child_rss(os.getpid());available=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
 assert tmp<=64*2**20 and rss+child+tmp<=2**30 and available>=2**30,('RAM',tmp,rss,child,available)
 return {'allocated_bytes':size,'free_bytes':free,'tmp_bytes':tmp,'peak_rss_bytes':rss,'child_rss_bytes':child,'memavailable_bytes':available}
def protocol():
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==read(OUT/'seal.json')['protocol_sha256']
 for f,h in p['dependencies'].items():assert sha(f)==h,f
 assert g.environment()==p['runtime'][sys.executable]
 if sys.executable==g.GPU_PY:assert importlib.metadata.version('python_speech_features')==p['python_speech_features_version']
 r=read(OUT/'reviewer_pass.json');assert r['status']=='PASS' and r['protocol_sha256']==sha(OUT/'protocol.json')
 assert sha(r['receipt'])==r['receipt_sha256'];limits();return p
g.OUT=OUT;g.TMP=TMP;g.limits=limits;g.write=write
def selected(p,block=None):
 e=read(OUT/'eligibility_lock.json');assert e['status']=='PASS' and e['protocol_sha256']==sha(OUT/'protocol.json')
 assert e['input_sha256']==sha(OUT/'input_gate.json')
 rows=[r for r in p['rows'] if r['id'] in e['eligible']]
 assert [r['id'] for r in rows]==e['eligible'] and len(rows)>=12
 return rows if block is None else rows[block*4:(block+1)*4]
def input_stage():
 import cv2
 cv2.setNumThreads(1);p=protocol();assert sys.executable==g.CPU_PY;rows=[]
 for r in p['rows']:
  assert r['split']=='eval' and sha(r['video'])==r['video_sha256'];fs=g.frames(r['video']);pts=g.old.read_probe_pts(r['video']);assert np.allclose(pts,np.arange(75)/25,atol=1e-6,rtol=0)
  m,_=g.save_detection(r['id'],'REAL','BASE',fs)
  assert g.decode_hashes_ffmpeg(r['video'])==m['pixel_hashes']
  z=np.load(OUT/'landmarks'/r['id']/'REAL/BASE.npz');q=geom.qc(geom.geom(z,z)) if z['valid'][0] else {'passed':False};assert q['passed']==m['qc']['passed'];z.close()
  rows.append({'id':r['id'],'qc':m['qc'],'box':m['box_from_first_frame'],'independent_pixel_QC_pass':True});print('INPUT',r['id'],m['qc']['passed'],flush=True)
 ids=[r['id'] for r in rows if r['qc']['passed']];status='PASS' if len(ids)>=12 else 'INPUT_QC_FAILED'
 write(OUT/'input_gate.json',{'status':status,'rows':rows,'fixed_denominator':16,'eligible':ids,'no_generated_or_score_selection':True})
 write(OUT/'eligibility_lock.json',{'status':status,'eligible':ids,'input_sha256':sha(OUT/'input_gate.json'),'protocol_sha256':sha(OUT/'protocol.json'),'time':time.time(),'minimum':12,'no_substitution':True})
def frontend():
 p=protocol();assert sys.executable==g.GPU_PY;sys.path.insert(0,str(ROOT/'third_party/Wav2Lip'));import audio
 rows=[]
 for r in selected(p):
  x=g.canonical_pcm(r);rms=float(np.sqrt(np.mean(x.astype(float)**2)));gain=g.TARGET/rms;y=(x*gain).astype(np.float32)
  assert np.max(abs(y))<=.98 and np.isfinite(y).all();items={}
  for c,a in [('RAW',x),('FIXED',y)]:
   mel=audio.melspectrogram(a).astype(np.float32);ix=np.minimum(np.floor(np.arange(75)*3.2).astype(int),mel.shape[1]-16);assert np.array_equal(ix[:69],np.floor(np.arange(69)*3.2).astype(int))
   path=OUT/'frontends'/r['id']/(c+'.npz');path.parent.mkdir(parents=True,exist_ok=True);limits();assert not path.exists();np.savez_compressed(path,mel=mel,indices=ix)
   items[c]={'waveform_hash':ah(a),'mel_hash':ah(mel),'npz_sha256':sha(path),'peak':float(max(abs(a)))}
  # Independent PCM API and full-waveform numerical bridge, not a second source.
  import soundfile as sf
  exact,rate=sf.read(r['pcm'],dtype='float32');assert rate==16000 and np.array_equal(exact,x)
  independent_rms=math.sqrt(sum(float(v)*float(v) for v in exact)/len(exact));assert abs(independent_rms-rms)<1e-12
  for c,a in [('RAW',exact),('FIXED',(exact*(g.TARGET/independent_rms)).astype(np.float32))]:
   z=np.load(OUT/'frontends'/r['id']/(c+'.npz'));assert np.array_equal(audio.melspectrogram(a).astype(np.float32),z['mel']);z.close()
  rows.append({'id':r['id'],'raw_rms':rms,'gain':gain,'gain_dB':20*math.log10(gain),'direction':'up' if gain>1 else 'down' if gain<1 else 'unchanged','conditions':items})
 write(OUT/'frontend_gate.json',{'status':'PASS','rows':rows,'PCM_fullmel_independent_bridge':'exact','eligibility_sha256':sha(OUT/'eligibility_lock.json')})
@contextlib.contextmanager
def lease(stage,block):
 limits();lock=open('/tmp/tts-exp-gpu.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  before=[]
  for i in range(3):
   z=g.gpu_state();assert not z['processes'] and z['memory_mib']<=128 and z['util']<=10;before.append(z)
   if i<2:time.sleep(5)
  write(OUT/'runtime'/f'{stage}_{block}_start.json',{'before':before,'runtime':g.environment()});yield
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 write(OUT/'runtime'/f'{stage}_{block}_exit.json',{'lease_released':True,'time':time.time(),'resources':limits()})
def model_state(model):
 h=hashlib.sha256()
 for name,t in sorted(model.state_dict().items()):
  h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def generate(block):
 import torch,cv2
 p=protocol();assert sys.executable==g.GPU_PY and read(OUT/'frontend_gate.json')['status']=='PASS';rows=selected(p,block);assert rows
 TMP.mkdir(exist_ok=True);torch.set_num_threads(1);cv2.setNumThreads(1)
 with lease('generate',block):
  sys.path.insert(0,str(ROOT/'third_party/Wav2Lip'));from models import Wav2Lip
  torch.manual_seed(20260927);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
  with torch.device('cuda'):model=Wav2Lip()
  ck=torch.load(str(ROOT/'third_party/Wav2Lip/checkpoints/wav2lip_gan.pth'),map_location='cuda');model.load_state_dict({k.replace('module.',''):v for k,v in ck['state_dict'].items()});del ck;model.eval()
  write(OUT/'runtime'/f'wav2lip_state_{block}.json',{'loaded_state_sha256':model_state(model),'checkpoint_sha256':sha(ROOT/'third_party/Wav2Lip/checkpoints/wav2lip_gan.pth')})
  for row in rows:
   sid=row['id'];m=read(OUT/'landmarks'/sid/'REAL/BASE.json');f=g.frames(row['video'])[0];x1,y1,x2,y2=m['box_from_first_frame'];face=cv2.resize(f[y1:y2,x1:x2],(96,96));masked=face.copy();masked[48:]=0;it=torch.from_numpy(np.concatenate([masked,face],2).transpose(2,0,1).astype(np.float32)[None]/255).cuda()
   for c in ['RAW','FIXED']:
    z=np.load(OUT/'frontends'/sid/(c+'.npz'));mel=z['mel'];ix=z['indices'];z.close();dest=TMP/(sid+'_'+c+'.mkv');assert not dest.exists()
    proc=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-f','rawvideo','-pixel_format','bgr24','-video_size','360x288','-framerate','25','-i','pipe:0','-an','-c:v','ffv1','-level','3','-threads','1','-pix_fmt','bgr0',str(dest)],stdin=subprocess.PIPE);hs=[];predhash=[]
    for i in range(75):
     limits();mt=torch.from_numpy(mel[:,ix[i]:ix[i]+16][None,None]).cuda()
     with torch.inference_mode():pred=model(mt,it).cpu().numpy()[0].transpose(1,2,0)*255
     frame=f.copy();frame[y1:y2,x1:x2]=cv2.resize(pred.astype(np.uint8),(x2-x1,y2-y1));predhash.append(ah(pred));hs.append(ah(frame));proc.stdin.write(frame.tobytes())
    proc.stdin.close();assert proc.wait()==0;assert g.decode_hashes_ffmpeg(dest)==hs
    write(OUT/'generation'/sid/(c+'.json'),{'id':sid,'condition':c,'temporary':str(dest),'video_sha256':sha(dest),'pixel_hashes':hs,'prediction_hashes':predhash,'decoded_bridge_exact':True,'box':[x1,y1,x2,y2],'frame0hash':ah(f),'mel_npz_sha256':sha(OUT/'frontends'/sid/(c+'.npz'))});print('GENERATED',sid,c,flush=True)
  del model;torch.cuda.empty_cache()
def controls(block):
 import cv2
 p=protocol();assert sys.executable==g.CPU_PY;cv2.setNumThreads(1)
 for r in selected(p,block):
  sid=r['id'];z=np.load(OUT/'landmarks'/sid/'REAL/BASE.npz');ref=(z['raw'][0],z['matrices'][0],bool(z['valid'][0]));z.close()
  for domain in DOMAINS:
   path=r['video'] if domain=='REAL' else read(OUT/'generation'/sid/(domain+'.json'))['temporary'];fs=g.frames(path)
   for v in VIEWS:
    if domain=='REAL' and v=='BASE':continue
    g.save_detection(sid,domain,v,g.transformed(fs,v),ref);print('VIEW',sid,domain,v,flush=True)
   if domain!='REAL':assert [ah(f) for f in fs]==read(OUT/'generation'/sid/(domain+'.json'))['pixel_hashes']
 write(OUT/'blocks'/f'geometry_{block}.json',{'status':'COMPLETE','ids':[r['id'] for r in selected(p,block)]})
def square_crop(frame,box):
 import cv2
 x1,y1,x2,y2=box;side=max(x2-x1,y2-y1);left=math.floor((x1+x2-side)/2);top=math.floor((y1+y2-side)/2);out=np.zeros((side,side,3),np.uint8)
 a,b=max(0,left),max(0,top);c,d=min(360,left+side),min(288,top+side);out[b-top:d-top,a-left:c-left]=frame[b:d,a:c]
 return cv2.resize(out,(224,224)),[left,top,left+side,top+side]
def sync(block):
 import torch,cv2
 from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
 p=protocol();assert sys.executable==g.GPU_PY;assert read(OUT/'blocks'/f'geometry_{block}.json')['status']=='COMPLETE';torch.set_num_threads(1);cv2.setNumThreads(1)
 torch.manual_seed(20260927);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 with lease('sync',block):
  engine=SyncNetEngine(batch_size=1,device='cuda');write(OUT/'runtime'/f'sync_state_{block}.json',{'loaded_state_sha256':model_state(engine.network),'checkpoint_sha256':engine.model_hash})
  for r in selected(p,block):
   sid=r['id'];box=read(OUT/'landmarks'/sid/'REAL/BASE.json')['box_from_first_frame'];dest=OUT/'sync'/sid;dest.mkdir(parents=True,exist_ok=True)
   a,am=engine.extract_audio(r['pcm']);assert a.shape==(69,1024);limits();np.save(dest/'A.npy',a)
   with wave.open(r['pcm']) as w:pcm=np.frombuffer(w.readframes(w.getnframes()),'<i2').copy()
   aw,_=engine._audio_windows(pcm);assert aw.shape==(69,13,20)
   metadata={'id':sid,'A_sha256':sha(dest/'A.npy'),'audio':am,'MFCC_windows_sha256':ah(aw),'conditions':{}}
   for c in ['RAW','FIXED']:
    rec=read(OUT/'generation'/sid/(c+'.json'));vp=Path(rec['temporary']);assert sha(vp)==rec['video_sha256'];crop=TMP/(sid+'_'+c+'_sync.avi');assert not crop.exists();writer=cv2.VideoWriter(str(crop),cv2.VideoWriter_fourcc(*'FFV1'),25,(224,224));assert writer.isOpened();hs=[]
    for frame in g.frames(vp):
     limits();cr,roi=square_crop(frame,box);writer.write(cr);hs.append(ah(cr))
    writer.release();cap=cv2.VideoCapture(str(crop));decoded=[]
    while True:
     ok,f=cap.read()
     if not ok:break
     decoded.append(ah(f))
    cap.release();assert decoded==hs
    v,vm=engine.extract_visual(crop);assert v.shape==(71,1024);np.save(dest/(c+'_V.npy'),v)
    bridge=None
    if sid==selected(p)[0]['id']:
     # Full decoded JPEG sequence versus streaming official frontend, same batch=1.
     full=list(engine._stream_mjpeg(crop));repeat=[]
     with torch.inference_mode():
      for i in range(71):repeat.append(engine.network.forward_lip(torch.from_numpy(engine._visual_batch([full[i:i+5]])).cuda()).cpu().numpy())
     repeat=np.concatenate(repeat).astype(np.float32);assert np.array_equal(v,repeat)
     import python_speech_features
     mfcc=python_speech_features.mfcc(pcm,16000);ind=np.stack([mfcc[4*i:4*i+20].T for i in range(69)]).astype(np.float32);assert np.array_equal(ind,aw)
     aa=[]
     with torch.inference_mode():
      for i in range(69):aa.append(engine.network.forward_aud(torch.from_numpy(ind[i:i+1,None]).cuda()).cpu().numpy())
     assert np.array_equal(a,np.concatenate(aa).astype(np.float32));bridge={'full_stream_V_exact':True,'independent_MFCC_A_exact':True};del full,repeat,aa
    metadata['conditions'][c]={'V_sha256':sha(dest/(c+'_V.npy')),'visual':vm,'square_ROI':roi,'crop_video_creation_sha256':sha(crop),'crop_pixel_hashes':hs,'bridge':bridge}
    assert np.array_equal(np.load(dest/(c+'_V.npy')),v);crop.unlink()
   write(dest/'metadata.json',metadata)
   for c in ['RAW','FIXED']:
    rec=read(OUT/'generation'/sid/(c+'.json'));vp=Path(rec['temporary']);assert vp.is_relative_to(TMP) and sha(vp)==rec['video_sha256']
    features={str(OUT/'landmarks'/sid/c/(v+'.npz')):sha(OUT/'landmarks'/sid/c/(v+'.npz')) for v in VIEWS}
    write(OUT/'retention'/sid/(c+'.json'),{'creation_video_sha256':rec['video_sha256'],'feature_hashes':features,'sync_metadata_sha256':sha(dest/'metadata.json'),'deleted_only_new_temporary':str(vp)});vp.unlink()
   print('SYNC_RETAINED',sid,flush=True)
  engine.close()
 write(OUT/'blocks'/f'sync_{block}.json',{'status':'COMPLETE','ids':[r['id'] for r in selected(p,block)]})
def stats(x,draws):
 x=np.asarray(x,float)
 if not np.isfinite(x).all():return {'n':len(x),'mean':None,'ci95':None,'ci99':None,'computable':False}
 means=np.mean(x[draws],axis=1);return {'n':len(x),'mean':float(np.mean(x)),'ci95':np.quantile(means,[.025,.975]).tolist(),'ci99':np.quantile(means,[.005,.995]).tolist(),'computable':True}
def analyze():
 import torch
 p=protocol();assert sys.executable==g.GPU_PY;torch.set_num_threads(1);rows=selected(p);n=len(rows);assert all((OUT/'blocks'/f'sync_{b}.json').exists() for b in range(math.ceil(n/4)))
 files={str(f):sha(f) for f in (OUT/'landmarks').rglob('*') if f.is_file()};files.update({str(f):sha(f) for f in (OUT/'sync').rglob('*') if f.is_file()})
 write(OUT/'score_lock.json',{'time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'eligibility_sha256':sha(OUT/'eligibility_lock.json'),'scientific_files':files,'before_effects':True})
 records=[];deltas=np.full((n,2,9,9),np.nan)
 for j,r in enumerate(rows):
  sid=r['id'];ref=np.load(OUT/'landmarks'/sid/'REAL/BASE.npz');data={}
  for domain in DOMAINS:
   for v in VIEWS:
    with np.load(OUT/'landmarks'/sid/domain/(v+'.npz')) as z:data[domain,v]=geom.geom(z,ref)
  # One support for every domain, spatial view and both time axes, no output filtering.
  t=np.array([i for i in T if all(z[1][i+l] for z in data.values() for l in LAGS)],dtype=int)
  row={'id':sid,'indices':t.tolist(),'support_pass':len(t)>=33,'domains':{},'correlations':{}}
  for domain in DOMAINS:
   q=geom.qc(data[domain,'BASE'],real=domain=='REAL');d={'BASE_QC_pass':q['passed'],'passed':False}
   if len(t)>=33:
    y=data[domain,'BASE'][0][t];sd=float(np.std(y,ddof=0));ss={v:geom.metric(y,data[domain,v][0][t]) for v in VIEWS[1:]};noise=max(s['centered_rmse'] for s in ss.values());d.update(native_sd=sd,max_spatial_centered_rmse=noise,spatial={v:{k:s[k] for k in ['r','centered_rmse']} for v,s in ss.items()});d['passed']=bool(q['passed'] and sd>=.005 and sd>=2*noise and all(s['r'] is not None and s['r']>=.95 for s in ss.values()))
   row['domains'][domain]=d
  computable=len(t)>=33
  for li,lag in enumerate(LAGS):
   rr={}
   for a,rv in enumerate(VIEWS):
    for b,gv in enumerate(VIEWS):
     rs={c:geom.metric(data['REAL',rv][0][t],data[c,gv][0][t+lag])['r'] for c in ['RAW','FIXED']} if len(t)>=33 else {'RAW':None,'FIXED':None}
     ok=all(v is not None and np.isfinite(v) for v in rs.values());computable=computable and ok
     delta=rs['FIXED']-rs['RAW'] if ok else None
     rr[rv+'/'+gv]={**rs,'delta':delta}
     if ok:deltas[j,li,a,b]=delta
   row['correlations'][str(lag)]=rr
  row['all_required_computable']=bool(computable);row['measurement_pass']=bool(len(t)>=33 and all(x['passed'] for x in row['domains'].values()));records.append(row);ref.close()
 draws=np.random.default_rng(20260927).integers(0,n,size=(100000,n));summary={}
 for li,lag in enumerate(LAGS):
  summary[str(lag)]={VIEWS[a]+'/'+VIEWS[b]:stats(deltas[:,li,a,b],draws) for a in range(9) for b in range(9)}
 measurable=sum(r['measurement_pass'] for r in records);complete=all(r['all_required_computable'] for r in records);gate=measurable>=math.ceil(.75*n);primary=summary['0'];valid=gate and complete
 conclusion='ROBUST_IMPROVEMENT' if valid and all(s['ci95'][0]>0 for s in primary.values()) else 'ROBUST_DECLINE' if valid and all(s['ci95'][1]<0 for s in primary.values()) else 'NOT_CONFIRMED'
 write(OUT/'geometry_results.json',{'rows':records,'eligible_denominator':n,'fixed_input_denominator':16,'measurement_passed_sources':measurable,'measurement_required':math.ceil(.75*n),'measurement_gate_pass':gate,'all_required_computable':complete,'summary':summary,'primary':'lag0 BASE/BASE','robustness_conclusion':conclusion,'bootstrap_draws':100000,'bootstrap_indices_hash':ah(draws),'seed':20260927,'no_output_QC_sample_removal':True,'FWER_simultaneous_coverage_claim':False})
 # Auxiliary same-label SyncNet; no geometry-window equivalence claim.
 from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
 sr=[]
 for r in rows:
  dest=OUT/'sync'/r['id'];a=np.load(dest/'A.npy');cells={}
  for c in ['RAW','FIXED']:
   v=np.load(dest/(c+'_V.npy'));m=SyncNetEngine.distance_matrix(v[:69],a);curve=m[20:49].mean(0);arg=int(np.argmin(curve));D=float(curve[arg]);C=float(np.median(curve)-D);cells[c]={'C':C,'D':D,'offset':15-arg,'best_lag':arg-15,'curve':curve.tolist(),'query_start':20,'query_stop':49};np.save(dest/(c+'_distances.npy'),m)
  sr.append({'id':r['id'],'cells':cells,'deltas':{k:cells['FIXED'][k]-cells['RAW'][k] for k in ['C','D','offset']}})
 write(OUT/'sync_results.json',{'rows':sr,'eligible_denominator':n,'summary':{k:stats([r['deltas'][k] for r in sr],draws) for k in ['C','D','offset']},'support':'same eligible labels, different temporal windows from geometry','primary_rescue_forbidden':True})
 print('RESULT',conclusion,'measurement',measurable,'/',n,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['input_stage','frontend','generate','controls','sync','analyze']);parser.add_argument('--block',type=int,default=0);args=parser.parse_args();globals()[args.stage](args.block) if args.stage in ['generate','controls','sync'] else globals()[args.stage]()
