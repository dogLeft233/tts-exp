"""Independent numerical/pixel/input audit. Never opens eval media or computes treatment deltas."""
from pathlib import Path
import argparse,hashlib,importlib.metadata,itertools,json,math,subprocess,wave,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/grid_geometry_calibration_20260927'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def binding_check(stage):
 import cv2
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==read(OUT/'seal.json')['protocol_sha256']
 for path,digest in p['dependencies'].items():assert sha(path)==digest,path
 expected='/home/wjj/.venvs/syncnet/bin/python' if stage=='frontend_check' else '/home/wjj/miniconda3/envs/autoavsr/bin/python';assert sys.executable==expected
 versions={}
 for name in ['numpy','torch','librosa','soundfile','mediapipe','opencv-python','opencv-python-headless','opencv-contrib-python']:
  try:versions[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:versions[name]=None
 actual={'python':sys.executable,'version':sys.version,'packages':versions,'cv2_version':cv2.__version__,'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]};assert actual==p['runtime'][sys.executable]
def euler(m):
 u,s,vt=np.linalg.svd(m[:3,:3]);r=u@vt;assert np.linalg.det(r)>0 and max(s)/min(s)<=1.02
 return np.array([np.arctan2(-r[2,0],np.hypot(r[0,0],r[1,0])),np.arctan2(r[2,1],r[2,2]),np.arctan2(r[1,0],r[0,0])])*180/np.pi
def geom(z,ref):
 target=ref['raw'][0,:,:2].astype(float)*[360,288];eye=np.linalg.norm(target[0]-target[3]);baseline=euler(ref['matrices'][0]);valid=z['valid'].copy();ap=np.full(75,np.nan);res=ap.copy();pose=np.full((75,3),np.nan)
 for i in np.flatnonzero(valid):
  a=z['raw'][i,:,:2].astype(float)*[360,288];ac=a[:7]-a[:7].mean(0);bc=target[:7]-target[:7].mean(0);dot=(ac*bc).sum();cross=(ac[:,0]*bc[:,1]-ac[:,1]*bc[:,0]).sum();theta=np.arctan2(cross,dot);rot=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]]);scale=np.hypot(dot,cross)/(ac*ac).sum();aligned=scale*(a-a[:7].mean(0))@rot+target[:7].mean(0)
  try:pose[i]=euler(z['matrices'][i])
  except (ValueError,AssertionError,np.linalg.LinAlgError):valid[i]=False;continue
  ap[i]=np.hypot(*(aligned[7]-aligned[8]))/eye;res[i]=np.sqrt(((aligned[:7]-target[:7])**2).sum(1).mean())/eye
 return ap,valid,res,pose,baseline

def qc(g,real=True):
 ap,v,res,pose,base=g;bad=np.flatnonzero(~v);longest=max([len(list(g)) for _,g in itertools.groupby(enumerate(bad),lambda x:x[1]-x[0])]+[0]);n=v.sum()
 if n==0:return {'valid_fraction':0.,'missing_run':longest,'reference_angles_deg':base.tolist(),'pose_change_p95_deg':None,'stable_residual_median':None,'stable_residual_p95':None,'aperture_sd':None,'passed':False}
 diff=abs((pose[v,:2]-base[:2]+180)%360-180);p95=np.percentile(diff,95,axis=0);sd=float(np.std(ap[v]));q={'valid_fraction':float(v.mean()),'missing_run':longest,'reference_angles_deg':base.tolist(),'pose_change_p95_deg':p95.tolist(),'stable_residual_median':float(np.median(res[v])),'stable_residual_p95':float(np.percentile(res[v],95)),'aperture_sd':sd};q['passed']=bool(v.mean()>=.95 and longest<=2 and max(abs(base[:2]))<=20 and max(p95)<=10 and np.median(res[v])<=.02 and np.percentile(res[v],95)<=.04 and (sd>=.005 or not real));return q

def pixels(path):
 cmd=['ffmpeg','-v','error','-threads','1','-i',str(path),'-an','-threads','1','-pix_fmt','bgr24','-f','rawvideo','pipe:1'];p=subprocess.Popen(cmd,stdout=subprocess.PIPE);out=[]
 while True:
  b=p.stdout.read(360*288*3)
  if not b:break
  assert len(b)==360*288*3;out.append(hashlib.sha256(b).hexdigest())
 assert p.wait()==0 and len(out)==75;return out

def input_check():
 p=read(OUT/'protocol.json');records=[];err=0.;count=0
 for r in p['calibration']:
  sid=r['id'];base=OUT/'landmarks'/sid/'REAL/BASE';meta=read(base.with_suffix('.json'));assert sha(base.with_suffix('.npz'))==meta['npz_sha256'];z=np.load(base.with_suffix('.npz'));assert np.array_equal(z['pts'],np.arange(75)/25)
  if z['valid'][0]:
   q=qc(geom(z,z));assert q['passed']==meta['qc']['passed']
   for k,v in q.items():
    if k=='passed':continue
    e=float(abs(np.asarray(v)-np.asarray(meta['qc'][k])).max());err=max(err,e);assert e<1e-7,(sid,k,e)
  else:q={'passed':False,'reference_invalid':True};assert not meta['qc']['passed']
  assert pixels(r['video'])==meta['pixel_hashes'];count+=75;records.append({'id':sid,'qc':q})
 gate=read(OUT/'input_gate.json');assert sum(r['qc']['passed'] for r in records)==gate['passed_sources'];write(OUT/'independent_input_check.json',{'status':'PASS','sources':8,'pixels':count,'numeric_max':err,'records':records,'checker_sha256':sha(__file__)})

def frontend_check():
 import soundfile as sf
 p=read(OUT/'protocol.json');front=read(OUT/'frontend_gate.json');by={r['id']:r for r in front['rows']};sys.path.insert(0,str(ROOT/'third_party/Wav2Lip'));import audio
 records=[]
 for r in p['calibration']:
  data,rate=sf.read(r['pcm'],dtype='float32');assert rate==16000 and data.shape==(47648,)
  with wave.open(r['pcm']) as w:raw=w.readframes(w.getnframes())
  exact=np.frombuffer(raw,'<i2').astype(np.float32)/32768.;assert np.array_equal(exact,data);f=by[r['id']];rms=math.sqrt(sum(float(v)*float(v) for v in data)/len(data));assert abs(rms-f['raw_rms'])<1e-12
  for c,y in [('RAW',data),('FIXED',(data*(.0376838172675746/rms)).astype(np.float32))]:
   z=np.load(OUT/'frontends'/r['id']/(c+'.npz'));mel=audio.melspectrogram(y).astype(np.float32);assert np.array_equal(mel,z['mel']);ix=np.minimum(np.floor(np.arange(75)*3.2).astype(int),mel.shape[1]-16);assert np.array_equal(ix,z['indices']);assert np.all(ix[:69]==np.floor(np.arange(69)*3.2).astype(int));assert hashlib.sha256(y.tobytes()).hexdigest()==f['conditions'][c]['waveform_hash']
  records.append({'id':r['id'],'canonical_float32_exact':True,'mel_exact':True,'indices_exact':True})
 write(OUT/'independent_inputs.json',{'status':'PASS','rows':records,'checker_sha256':sha(__file__)})

def metric(y,x):
 yc=y-np.mean(y);xc=x-np.mean(x);vy=float(np.dot(yc,yc)/len(y));vx=float(np.dot(xc,xc)/len(x));mse=float(np.dot(xc-yc,xc-yc)/len(y));return {'E':1-mse/vy if vy>1e-12 else None,'r':float(np.dot(xc,yc)/len(y)/math.sqrt(vy*vx)) if min(vy,vx)>1e-12 else None,'centered_rmse':math.sqrt(mse),'reference_sd':math.sqrt(vy),'candidate_sd':math.sqrt(vx),'uncentered_rmse':float(np.linalg.norm(x-y)/math.sqrt(len(y)))}
def choose(vals,ls):
 pairs=[(int(l),v) for l,v in zip(ls,vals) if v is not None and np.isfinite(v)]
 if not pairs:return None
 mx=max(v for _,v in pairs);return sorted((l for l,v in pairs if abs(v-mx)<=1e-12),key=lambda q:(abs(q),q))[0]

def calibration_check():
 p=read(OUT/'protocol.json');diag=read(OUT/'calibration_diagnostics.json');lock=read(OUT/'lag_lock.json');T=np.arange(14,55);ls=np.arange(-5,6);cache={};err=0.;checks=0
 def fetch(s,d,v='BASE'):
  key=(s,d,v)
  if key not in cache:
   ref=np.load(OUT/'landmarks'/s/'REAL/BASE.npz');z=np.load(OUT/'landmarks'/s/d/(v+'.npz'));cache[key]=geom(z,ref)[:2]
  return cache[key]
 def compare(a,b):
  nonlocal err,checks
  for k,v in a.items():
   if v is None:assert b[k] is None
   else:e=abs(v-b[k]);assert e<1e-7,(k,e);err=max(err,e)
   checks+=1
 good=[]
 gate={q['id']:q for q in read(OUT/'input_gate.json')['rows']}
 assert [r['id'] for r in lock['rows']]==[r['id'] for r in p['calibration']]
 for r in lock['rows']:
  ref=np.load(OUT/'landmarks'/r['id']/'REAL/BASE.npz');input_ok=bool(ref['valid'][0]) and qc(geom(ref,ref))['passed'];assert input_ok==gate[r['id']]['qc']['passed']
  if not input_ok:
   assert not r['eligible'] and 'r_by_lag' not in r;continue
  y,yv=fetch(r['id'],'REAL');x,xv=fetch(r['id'],'RAW');t=np.array([i for i in T if yv[i] and all(xv[i+l] for l in ls)]);assert t.tolist()==r['indices'];rr=[metric(y[t],x[t+l])['r'] for l in ls] if len(t)>=33 else [];assert len(rr)==len(r['r_by_lag'])
  for value,saved in zip(rr,r['r_by_lag']):
   if value is None:assert saved is None
   else:assert saved is not None and abs(value-saved)<1e-12
  eligible=len(rr)==11 and all(v is not None and np.isfinite(v) for v in rr);assert eligible==r['eligible']
  if eligible:good.append(rr)
 means=np.mean(good,0).tolist() if len(good)>=6 else [];assert len(good)==lock['source_count'];assert len(means)==len(lock['mean_r_by_lag']);assert not means or np.allclose(means,lock['mean_r_by_lag'],atol=1e-12);assert (choose(means,ls) if means else None)==lock['lag'];lag=lock['lag'];passed=0
 for row in diag['rows']:
  sid=row['id']
  if not row['input_pass']:assert not row['passed'];continue
  flags=[]
  for domain,d in row['domains'].items():
   data={v:fetch(sid,domain,v) for v in p['views']};t=np.array([i for i in T if all(all(valid[i+l] for l in ls) for _,valid in data.values())]);assert t.tolist()==d['indices']
   ref=np.load(OUT/'landmarks'/sid/'REAL/BASE.npz');dz=np.load(OUT/'landmarks'/sid/domain/'BASE.npz');q=qc(geom(dz,ref),real=domain=='REAL');actualqc=read(OUT/'landmarks'/sid/domain/'BASE.json')['qc'];baselineqc=q['passed'];assert baselineqc==actualqc['passed']==d['baseline_QC_pass']
   for key,val in q.items():
    if key=='passed':continue
    if val is None:assert actualqc[key] is None
    else:assert np.allclose(val,actualqc[key],atol=1e-7)
   if len(t)<33:assert not d['passed'];flags.append(False);continue
   y=data['BASE'][0];scores={v:metric(y[t],x[t]) for v,(x,_) in data.items()};ff=[]
   for v,s in scores.items():
    c=d['controls'][v];compare(s,c);ok=True
    if v=='REPEAT':ok=s['r'] is not None and s['r']>=.99 and s['centered_rmse']<=.001
    elif v.startswith('SHIFT_'):
     delta=int(v.split('_')[1]);rr=[metric(y[t],data[v][0][t+l])['r'] for l in ls];b=choose(rr,ls);assert b==c['recovered_lag'];ok=b is not None and b*delta>0 and abs(b-delta)<=1
    elif v=='FROZEN':ok=s['E'] is not None and s['E']<=.05
    elif v=='REVERSE' or v.startswith('WARP_'):ok=s['E'] is not None and scores['REPEAT']['E'] is not None and scores['REPEAT']['E']-s['E']>=.1
    elif v.startswith(('TX_','TY_','ROT_','SCALE_')):ok=s['r'] is not None and s['r']>=.95 and s['centered_rmse']<=.01
    assert bool(ok)==c['passed']
    if v!='BASE':ff.append(ok)
   dose=all(scores[f'SHIFT_{s*4}']['E']<=scores[f'SHIFT_{s*2}']['E']+1e-12 for s in [-1,1]) if all(scores[f'SHIFT_{d}']['E'] is not None for d in [-4,-2,2,4]) else None;assert dose==d['shift_dose_monotone_diagnostic'] and not d['shift_dose_is_gate'];want=all(ff) and baselineqc;assert want==d['passed'];flags.append(want)
  c=row['cross_domain'];cp=False
  if lag is not None:
   y,yv=fetch(sid,'REAL');raw={v:fetch(sid,'RAW',v) for v in ['BASE','SHIFT_-4','SHIFT_-2','SHIFT_2','SHIFT_4']};search=ls+lag;t=np.array([i for i in T if yv[i] and all(all(valid[i+l] for l in search) for _,valid in raw.values())]);assert t.tolist()==c['indices']
   if len(t)>=33:
    m=metric(y[t],raw['BASE'][0][t+lag]);compare(m,c['RAW']);compare(metric(y[t],raw['BASE'][0][t]),c['RAW_lag0_same_support']);b=choose([metric(y[t],raw['BASE'][0][t+l])['r'] for l in search],search);ff=[]
    for v in list(raw)[1:]:
     d=int(v.split('_')[1]);best=choose([metric(y[t],raw[v][0][t+l])['r'] for l in search],search);ok=best is not None and b is not None and abs(best-b-d)<=1 and (best-b)*d>0;assert ok==c['shift'][v]['passed'];ff.append(ok)
    cp=m['r'] is not None and m['r']>=.3 and all(ff)
   a,av=fetch(sid,'RAW');f,fv=fetch(sid,'FIXED');main=[i for i in T if yv[i] and all(av[i+l] and fv[i+l] for l in ls)];assert main==c['three_domain_main_indices'];cp=cp and len(main)>=33
  assert bool(cp)==c['passed'];want=all(flags) and cp;assert want==row['passed'];passed+=want
 assert passed==diag['passed_sources'];write(OUT/'independent_calibration_check.json',{'status':'PASS','numeric_max':err,'metric_scalar_checks':checks,'passed_cal_sources':passed,'lag':lag,'checker_sha256':sha(__file__),'no_treatment_deltas':True})
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['input_check','frontend_check','calibration_check']);args=ap.parse_args();binding_check(args.stage);globals()[args.stage]()
