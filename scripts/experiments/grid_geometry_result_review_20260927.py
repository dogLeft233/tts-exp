"""Read-only cal8 independent result review. Execute only after delivery; no effects/GPU."""
from pathlib import Path
import hashlib,itertools,json,math,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/grid_geometry_calibration_20260927'
OUT=ROOT/'runs/grid_geometry_protocol_audit_20260927/result_review'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def angles(m):
 u,s,v=np.linalg.svd(m[:3,:3]);q=u@v
 if np.linalg.det(q)<=0 or s[-1]<=0 or s[0]/s[-1]>1.02:raise ValueError('invalid pose matrix')
 return np.rad2deg([math.atan2(-q[2,0],math.hypot(q[0,0],q[1,0])),math.atan2(q[2,1],q[2,2]),math.atan2(q[1,0],q[0,0])])
def geometry(z,ref):
 target=ref['raw'][0,:,:2].astype(float)*[360,288];b=target[:,0]+1j*target[:,1];eye=abs(b[0]-b[3]);assert eye>1
 bm=b[:7].mean();bc=b[:7]-bm;v=z['valid'].copy();ap=np.full(75,np.nan);res=ap.copy();pose=np.full((75,3),np.nan)
 for i in np.flatnonzero(v):
  points=z['raw'][i,:,:2].astype(float)*[360,288];a=points[:,0]+1j*points[:,1];am=a[:7].mean();ac=a[:7]-am
  try:
   if np.vdot(ac,ac).real<=0:raise ValueError('degenerate similarity')
   c=np.vdot(ac,bc)/np.vdot(ac,ac).real;aligned=c*(a-am)+bm;pose[i]=angles(z['matrices'][i]);ap[i]=abs(aligned[7]-aligned[8])/eye;res[i]=np.sqrt(np.mean(abs(aligned[:7]-b[:7])**2))/eye
  except (ValueError,np.linalg.LinAlgError):v[i]=False
 return {'ap':ap,'valid':v,'res':res,'pose':pose,'reference':angles(ref['matrices'][0])}
def qc(g,real):
 v=g['valid'];runs=[len(list(group)) for value,group in itertools.groupby(v) if not value];n=int(v.sum());ref=g['reference'];diff=abs((g['pose'][v,:2]-ref[:2]+180)%360-180)
 flags={'coverage':float(v.mean())>=.95,'missing':max(runs,default=0)<=2,'reference_pose':max(abs(ref[:2]))<=20,'relative_pose':bool(n and max(np.percentile(diff,95,axis=0))<=10),'similarity':bool(n and np.median(g['res'][v])<=.02 and np.percentile(g['res'][v],95)<=.04)}
 if real:flags['motion']=bool(n and np.std(g['ap'][v])>=.005)
 return {k:bool(x) for k,x in flags.items()}
def metric(y,x):
 yc=y-y.mean();xc=x-x.mean();vy=float(np.vdot(yc,yc)/len(y));vx=float(np.vdot(xc,xc)/len(x));mse=float(np.vdot(xc-yc,xc-yc)/len(y))
 return {'E':1-mse/vy if vy>1e-12 else None,'r':float(np.vdot(xc,yc)/len(y)/np.sqrt(vx*vy)) if min(vx,vy)>1e-12 else None,'centered_rmse':math.sqrt(mse),'reference_sd':math.sqrt(vy),'candidate_sd':math.sqrt(vx),'uncentered_rmse':float(np.sqrt(np.mean((x-y)**2)))}
def choose(curve,lags):
 pairs=[(int(k),v) for k,v in zip(lags,curve) if v is not None and np.isfinite(v)]
 if not pairs:return None
 mx=max(v for _,v in pairs);return min([k for k,v in pairs if abs(v-mx)<=1e-12],key=lambda k:(abs(k),k))
def main():
 p=read(RUN/'protocol.json');diag=read(RUN/'calibration_diagnostics.json');lock=read(RUN/'lag_lock.json');inputs=read(RUN/'input_gate.json');assert sha(RUN/'protocol.json')==read(RUN/'seal.json')['protocol_sha256']
 assert [x['id'] for x in p['calibration']]==['s'+str(i) for i in range(2,10)]
 T=np.arange(14,55);L=np.arange(-5,6);views=p['views'];ids=[x['id'] for x in p['calibration']];cache={};hashes={};maximum=0.;checks=0
 compatible={sha(RUN/'protocol.json'),read(OUT/'filename_mapping_before.json')['protocol_sha256']}
 def get(s,d,v='BASE'):
  key=(s,d,v)
  if key not in cache:
   base=RUN/'landmarks'/s/d;f=base/(v+'.npz');meta=read(base/(v+'.json'));assert (meta['id'],meta['domain'],meta['view'])==key;assert meta['protocol_sha256'] in compatible;assert sha(f)==meta['npz_sha256'];hashes[str(f)]=sha(f);hashes[str(base/(v+'.json'))]=sha(base/(v+'.json'))
   with np.load(RUN/'landmarks'/s/'REAL/BASE.npz') as z:ref={k:z[k] for k in z.files}
   with np.load(f) as z:data={k:z[k] for k in z.files}
   assert np.array_equal(data['pts'],np.arange(75)/25)
   cache[key]=geometry(data,ref)
  return cache[key]
 def compare(a,b):
  nonlocal maximum,checks
  for k,v in a.items():
   if v is None:assert b[k] is None,(k,b[k])
   else:
    assert b[k] is not None;error=abs(v-b[k]);maximum=max(maximum,error);assert error<1e-7,(k,error)
   checks+=1
 qcs={s:qc(get(s,'REAL'),True) for s in ids};byinput={x['id']:x for x in inputs['rows']};good=[];lagrows={x['id']:x for x in lock['rows']}
 for s in ids:
  passed=all(qcs[s].values());assert passed==byinput[s]['qc']['passed'];row=lagrows[s]
  if not passed:assert not row['eligible'];continue
  y=get(s,'REAL');x=get(s,'RAW');t=np.array([i for i in T if y['valid'][i] and all(x['valid'][i+k] for k in L)]);assert t.tolist()==row['indices'];curve=[metric(y['ap'][t],x['ap'][t+k])['r'] for k in L] if len(t)>=33 else [];assert len(curve)==len(row['r_by_lag'])
  for actual,saved in zip(curve,row['r_by_lag']):
   if actual is None:assert saved is None
   else:assert saved is not None and abs(actual-saved)<1e-12
  eligible=len(curve)==11 and all(x is not None and np.isfinite(x) for x in curve);assert eligible==row['eligible']
  if eligible:good.append(curve)
 means=np.mean(good,axis=0).tolist() if len(good)>=6 else [];lag=choose(means,L) if means else None;assert lag==lock['lag'] and len(good)==lock['source_count'];assert len(means)==len(lock['mean_r_by_lag']) and all(abs(a-b)<1e-12 for a,b in zip(means,lock['mean_r_by_lag']));assert sum(all(x.values()) for x in qcs.values())==inputs['passed_sources'];records=[]
 for row in diag['rows']:
  s=row['id'];assert s in ids and row['input_pass']==all(qcs[s].values());record={'id':s,'input_pass':row['input_pass'],'input_flags':qcs[s],'domains':{},'cross':{}}
  if not row['input_pass']:assert not row['passed'];record['passed']=False;records.append(record);continue
  for d in ['REAL','RAW','FIXED']:
   data={v:get(s,d,v) for v in views};t=np.array([i for i in T if all(all(z['valid'][i+k] for k in L) for z in data.values())]);saved=row['domains'][d];assert t.tolist()==saved['indices'];q=qc(data['BASE'],d=='REAL');basepass=all(q.values());assert basepass==saved['baseline_QC_pass'];control=[];flags=[]
   if len(t)>=33:
    y=data['BASE']['ap'];scores={v:metric(y[t],z['ap'][t]) for v,z in data.items()}
    for v,z in scores.items():
     compare(z,saved['controls'][v]);passed=True;extra={}
     if v=='REPEAT':passed=z['r'] is not None and z['r']>=.99 and z['centered_rmse']<=.001
     elif v.startswith('SHIFT_'):
      delta=int(v.split('_')[1]);curve=[metric(y[t],data[v]['ap'][t+k])['r'] for k in L];best=choose(curve,L);assert best==saved['controls'][v]['recovered_lag'];passed=best is not None and best*delta>0 and abs(best-delta)<=1;extra={'expected':delta,'recovered':best}
     elif v=='FROZEN':passed=z['E'] is not None and z['E']<=.05
     elif v=='REVERSE' or v.startswith('WARP_'):passed=z['E'] is not None and scores['REPEAT']['E'] is not None and scores['REPEAT']['E']-z['E']>=.1
     elif v.startswith(('TX_','TY_','ROT_','SCALE_')):passed=z['r'] is not None and z['r']>=.95 and z['centered_rmse']<=.01
     assert bool(passed)==saved['controls'][v]['passed'];control.append({'view':v,**z,**extra,'passed':bool(passed)})
     if v!='BASE':flags.append(passed)
    dose=all(scores[f'SHIFT_{sg*4}']['E']<=scores[f'SHIFT_{sg*2}']['E']+1e-12 for sg in [-1,1]) if all(scores[f'SHIFT_{q}']['E'] is not None for q in [-4,-2,2,4]) else None;assert dose==saved['shift_dose_monotone_diagnostic'] and not saved['shift_dose_is_gate'];want=all(flags) and basepass
   else:want=False;dose=None
   assert bool(want)==saved['passed'];record['domains'][d]={'passed':bool(want),'valid_frames':len(t),'qc':q,'failed_controls':[x['view'] for x in control if not x['passed']],'controls':control,'dose_diagnostic':dose}
  crosspass=False;cs=row['cross_domain'];cinfo={'lag':lag,'passed':False}
  if lag is not None:
   y=get(s,'REAL');raw={v:get(s,'RAW',v) for v in ['BASE','SHIFT_-4','SHIFT_-2','SHIFT_2','SHIFT_4']};search=L+lag;t=np.array([i for i in T if y['valid'][i] and all(all(z['valid'][i+k] for k in search) for z in raw.values())]);assert t.tolist()==cs['indices'];cinfo['valid_frames']=len(t)
   if len(t)>=33:
    m=metric(y['ap'][t],raw['BASE']['ap'][t+lag]);compare(m,cs['RAW']);zero=metric(y['ap'][t],raw['BASE']['ap'][t]);compare(zero,cs['RAW_lag0_same_support']);best=choose([metric(y['ap'][t],raw['BASE']['ap'][t+k])['r'] for k in search],search);shifts=[]
    for v in list(raw)[1:]:
     delta=int(v.split('_')[1]);b=choose([metric(y['ap'][t],raw[v]['ap'][t+k])['r'] for k in search],search);ok=b is not None and best is not None and (b-best)*delta>0 and abs(b-best-delta)<=1;assert ok==cs['shift'][v]['passed'];shifts.append({'view':v,'expected':delta,'recovered':b-best if b is not None and best is not None else None,'passed':ok})
    crosspass=m['r'] is not None and m['r']>=.3 and all(z['passed'] for z in shifts);cinfo.update({'RAW':m,'RAW_lag0':zero,'shift_controls':shifts})
   a=get(s,'RAW');f=get(s,'FIXED');main=[i for i in T if y['valid'][i] and all(a['valid'][i+k] and f['valid'][i+k] for k in L)];assert main==cs['three_domain_main_indices'];crosspass=crosspass and len(main)>=33;cinfo['main_valid_frames']=len(main)
  assert bool(crosspass)==cs['passed'];cinfo['passed']=bool(crosspass);record['cross']=cinfo;want=all(x['passed'] for x in record['domains'].values()) and crosspass;assert bool(want)==row['passed'];record['passed']=bool(want);records.append(record)
 assert [x['id'] for x in records]==ids;count=sum(x['passed'] for x in records);assert count==diag['passed_sources'] and diag['status']==('PASS' if count>=6 else 'FAIL') and diag['fixed_denominator']==8
 assert diag['lag_lock_sha256']==sha(RUN/'lag_lock.json') and diag['no_treatment_effect_summary'] and diag['SyncNet_NOT_RUN'] and diag['eval_locked']
 scopes={}
 for sub in ['landmarks','frontends','generation','retention']:
  actual=sorted(x.name for x in (RUN/sub).iterdir() if x.is_dir());assert set(actual)<=set(ids);scopes[sub]=actual
 result={'status':'PASS','review_time':time.time(),'calibration_status':diag['status'],'passed_sources':count,'fixed_denominator':8,'input_passed_sources':sum(all(x.values()) for x in qcs.values()),'lag':lag,'lag_sources':len(good),'numeric_max_error':maximum,'scalar_checks':checks,'records':records,'source_scopes':scopes,'chronology':{'lag_created':lock['created'],'diagnostics_mtime':(RUN/'calibration_diagnostics.json').stat().st_mtime,'lag_precedes_diagnostics':lock['created']<=(RUN/'calibration_diagnostics.json').stat().st_mtime},'protocol_sha256':sha(RUN/'protocol.json'),'code_sha256':sha(__file__),'no_treatment_deltas_computed':True,'GPU':0,'artifact_hashes':hashes}
 assert result['chronology']['lag_precedes_diagnostics'];dest=OUT/'independent_results.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['records','artifact_hashes']}))
if __name__=='__main__':main()
