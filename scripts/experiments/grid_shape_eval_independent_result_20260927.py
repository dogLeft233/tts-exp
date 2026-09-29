"""Independent CPU geometry, statistics, distances and provenance review of sealed GRID16."""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
import torch
import grid_geometry_result_review_20260927 as independent
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/grid_shape_correlation_evaluation_20260927'
OUT=ROOT/'runs/grid_shape_eval_audit_20260927/result_review'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch.set_num_threads(1)
maximum=0.;checks=0;hashes={}
def compare(a,b,tolerance=1e-7):
 global maximum,checks
 if a is None or b is None:assert a is None and b is None;return
 err=float(np.max(abs(np.asarray(a)-np.asarray(b))))
 maximum=max(maximum,err);checks+=np.asarray(a).size
 assert err<=tolerance,(err,tolerance,a,b)
def verify(path,expected=None):
 h=sha(path)
 if expected is not None:assert h==expected,str(path)
 hashes[str(path)]=h;return h
def stats(x,draws):
 x=np.asarray(x,float)
 if not np.isfinite(x).all():return {'n':len(x),'mean':None,'ci95':None,'ci99':None,'computable':False}
 # Column accumulation is independent of the producer's x[draws].mean(axis=1).
 mean=sum(x[draws[:,k]] for k in range(len(x)))/len(x)
 return {'n':len(x),'mean':float(sum(x)/len(x)),'ci95':np.percentile(mean,[2.5,97.5]).tolist(),'ci99':np.percentile(mean,[.5,99.5]).tolist(),'computable':True}
def compare_stats(actual,saved,tol=1e-12):
 assert actual['n']==saved['n'] and actual['computable']==saved['computable']
 for k in ['mean','ci95','ci99']:compare(actual[k],saved[k],tol)

p=read(RUN/'protocol.json');ph=verify(RUN/'protocol.json',read(RUN/'seal.json')['protocol_sha256'])
assert ph=='4a6da1c680812a73f277b966c96d907022c7e0124b399347d4459f5949fa1aa7'
for path,h in p['dependencies'].items():verify(path,h)
e=read(RUN/'eligibility_lock.json');verify(RUN/'input_gate.json',e['input_sha256']);inputs=read(RUN/'input_gate.json')
ids=e['eligible'];assert len(ids)>=12 and [r['id'] for r in p['rows']]==ids
saved=read(RUN/'geometry_results.json');syn=read(RUN/'sync_results.json');lock=read(RUN/'score_lock.json')
assert lock['protocol_sha256']==ph and lock['before_effects'];verify(RUN/'eligibility_lock.json',lock['eligibility_sha256'])
for path,h in lock['scientific_files'].items():verify(path,h)
assert [r['id'] for r in saved['rows']]==[r['id'] for r in syn['rows']]==ids
views=p['views'];lags=[0,-3];domains=['REAL','RAW','FIXED'];T=np.arange(14,55)
delta=np.full((len(ids),2,9,9),np.nan);records=[];source_bindings=[]
for j,sid in enumerate(ids):
 with np.load(RUN/'landmarks'/sid/'REAL/BASE.npz') as f:ref={k:f[k] for k in f.files}
 data={};row=saved['rows'][j]
 for d in domains:
  for v in views:
   f=RUN/'landmarks'/sid/d/(v+'.npz');meta=read(f.with_suffix('.json'))
   assert (meta['id'],meta['domain'],meta['view'])==(sid,d,v) and meta['protocol_sha256']==ph
   verify(f,meta['npz_sha256'])
   with np.load(f) as z:
    assert np.array_equal(z['pts'],np.arange(75)/25)
    data[d,v]=independent.geometry(z,ref)
 realflags=independent.qc(data['REAL','BASE'],True)
 assert all(realflags.values())==inputs['rows'][j]['qc']['passed']
 t=T[np.logical_and.reduce([z['valid'][T+l] for z in data.values() for l in lags])]
 assert t.tolist()==row['indices'] and (len(t)>=33)==row['support_pass']
 gates={};failed=[]
 for d in domains:
  q=independent.qc(data[d,'BASE'],d=='REAL');sd=float(np.std(data[d,'BASE']['ap'][t],ddof=0))
  space={v:independent.metric(data[d,'BASE']['ap'][t],data[d,v]['ap'][t]) for v in views[1:]}
  noise=max(x['centered_rmse'] for x in space.values());s=row['domains'][d]
  assert all(q.values())==s['BASE_QC_pass'];compare(sd,s['native_sd']);compare(noise,s['max_spatial_centered_rmse'])
  for v,x in space.items():
   for k in ['r','centered_rmse']:compare(x[k],s['spatial'][v][k])
  gate=all(q.values()) and len(t)>=33 and sd>=.005 and sd>=2*noise and all(x['r'] is not None and x['r']>=.95 for x in space.values())
  assert gate==s['passed'];gates[d]={'passed':bool(gate),'qc_flags':q,'sd':sd,'noise_max':noise,'spatial_r_failed':[v for v,x in space.items() if x['r'] is None or x['r']<.95]}
  if not gate:failed.append(d)
 complete=True
 for li,lag in enumerate(lags):
  for a,rv in enumerate(views):
   for b,gv in enumerate(views):
    actual={d:independent.metric(data['REAL',rv]['ap'][t],data[d,gv]['ap'][t+lag])['r'] for d in ['RAW','FIXED']}
    s=row['correlations'][str(lag)][rv+'/'+gv]
    for d,x in actual.items():compare(x,s[d],1e-12)
    ok=all(x is not None and np.isfinite(x) for x in actual.values());complete &= ok
    x=actual['FIXED']-actual['RAW'] if ok else None;compare(x,s['delta'],1e-12)
    if ok:delta[j,li,a,b]=x
 gate=all(d['passed'] for d in gates.values())
 assert gate==row['measurement_pass'] and complete==row['all_required_computable']
 records.append({'id':sid,'support':len(t),'measurement_pass':bool(gate),'domains':gates,'all_required_computable':bool(complete)})
 # Source identity/retention is verified by file/payload hashes, without rerendering.
 src=next(r for r in p['rows'] if r['id']==sid)
 for name in ['video','pcm','timeline']:verify(src[name],src[name+'_sha256'])
 meta=read(RUN/'sync'/sid/'metadata.json');verify(RUN/'sync'/sid/'A.npy',meta['A_sha256'])
 for c in ['RAW','FIXED']:
  gen=read(RUN/'generation'/sid/(c+'.json'));ret=read(RUN/'retention'/sid/(c+'.json'))
  assert gen['decoded_bridge_exact'] and gen['video_sha256']==ret['creation_video_sha256'] and len(gen['pixel_hashes'])==75
  assert not Path(gen['temporary']).exists();verify(RUN/'sync'/sid/'metadata.json',ret['sync_metadata_sha256'])
  for path,h in ret['feature_hashes'].items():verify(path,h)
  verify(RUN/'sync'/sid/(c+'_V.npy'),meta['conditions'][c]['V_sha256'])
  if j==0:assert meta['conditions'][c]['bridge']=={'full_stream_V_exact':True,'independent_MFCC_A_exact':True}
  assert e['time'] <= (RUN/'generation'/sid/(c+'.json')).stat().st_mtime < lock['time']
 source_bindings.append(sid)

draws=np.random.default_rng(20260927).integers(0,len(ids),size=(100000,len(ids)))
assert hashlib.sha256(draws.tobytes()).hexdigest()==saved['bootstrap_indices_hash']
summary={}
for li,lag in enumerate(lags):
 summary[str(lag)]={}
 for a,rv in enumerate(views):
  for b,gv in enumerate(views):
   s=stats(delta[:,li,a,b],draws);compare_stats(s,saved['summary'][str(lag)][rv+'/'+gv]);summary[str(lag)][rv+'/'+gv]=s
count=sum(r['measurement_pass'] for r in records);complete=all(r['all_required_computable'] for r in records);gate=count>=math.ceil(.75*len(ids))
assert count==saved['measurement_passed_sources'] and complete==saved['all_required_computable'] and gate==saved['measurement_gate_pass']
positive=gate and complete and all(s['ci95'][0]>0 for s in summary['0'].values())
negative=gate and complete and all(s['ci95'][1]<0 for s in summary['0'].values())
conclusion='ROBUST_IMPROVEMENT' if positive else 'ROBUST_DECLINE' if negative else 'NOT_CONFIRMED'
assert conclusion==saved['robustness_conclusion']

distance_max=0.;distance_entries=0;sync_rows=[]
for j,sid in enumerate(ids):
 path=RUN/'sync'/sid;a=np.load(path/'A.npy');assert a.shape==(69,1024) and a.dtype==np.float32
 padded=np.pad(a,((15,15),(0,0)));cells={}
 for c in ['RAW','FIXED']:
  v=np.load(path/(c+'_V.npy'));assert v.shape==(71,1024) and v.dtype==np.float32
  # Explicit float32 subtraction, epsilon addition, and vector norm; no producer
  # distance function, model object, CUDA context, or feature normalization.
  m=np.stack([torch.linalg.vector_norm(torch.from_numpy(v[i:i+1]-padded[i:i+31])+1e-6,dim=1).numpy().astype(float) for i in range(69)])
  original=np.load(path/(c+'_distances.npy'));err=float(np.max(abs(m-original)));distance_max=max(distance_max,err);distance_entries+=m.size
  assert np.array_equal(m,original),(sid,c,err)
  curve=np.sum(m[20:49],axis=0)/29;arg=int(np.argmin(curve));D=float(curve[arg]);C=float(np.median(curve)-D)
  s=syn['rows'][j]['cells'][c];compare(curve,s['curve'],1e-12);compare(C,s['C'],1e-12);compare(D,s['D'],1e-12)
  assert (15-arg,arg-15,s['query_start'],s['query_stop'])==(s['offset'],s['best_lag'],20,49)
  cells[c]={'C':C,'D':D,'offset':15-arg}
 dr={k:cells['FIXED'][k]-cells['RAW'][k] for k in ['C','D','offset']}
 for k,x in dr.items():compare(x,syn['rows'][j]['deltas'][k],1e-12)
 sync_rows.append({'id':sid,'deltas':dr})
sync_summary={k:stats([r['deltas'][k] for r in sync_rows],draws) for k in ['C','D','offset']}
for k,s in sync_summary.items():compare_stats(s,syn['summary'][k])

for stage in ['generate','sync']:
 states=[read(RUN/'runtime'/f'{"wav2lip" if stage=="generate" else "sync"}_state_{i}.json')['loaded_state_sha256'] for i in range(4)]
 assert len(set(states))==1
 for i in range(4):
  r=read(RUN/'runtime'/f'{stage}_{i}_compute_empty.json');assert r['compute_empty'] and r['worker_exit_code']==0
assert e['time']<lock['time']<(RUN/'geometry_results.json').stat().st_mtime
verify(RUN/'geometry_results.json','b664288eb43c95fd349d8ece6c88336f058bf8679fac44ad2a95a07f02cff7a3')
verify(RUN/'sync_results.json','5b60f1cac508cf305596957cee58a42960ca7a42000b12a12fc670cfbd9b491a')
verify(RUN/'score_lock.json','3afb5fbae3142facf4fdf442ace757f277bd2c9db020bfb5181c21717b95b397')
result={'status':'PASS','time':time.time(),'protocol_sha256':ph,'auditor_sha256':sha(__file__),
        'independent_geometry_sha256':sha(independent.__file__),'source_count':len(ids),'views':len(ids)*27,
        'measurement_passed':count,'required':math.ceil(.75*len(ids)),'records':records,'geometry_summary':summary,
        'robustness_conclusion':conclusion,'sync_summary':sync_summary,'scalar_max_error':maximum,'scalar_checks':int(checks),
        'distance_entries_exact':distance_entries,'distance_max_error':distance_max,'hashes_checked':len(hashes),
        'chronology':{'eligibility_time':e['time'],'score_lock_time':lock['time'],'result_mtime':(RUN/'geometry_results.json').stat().st_mtime},
        'GPU':0,'model_forward':0,'media_decode':0,'eval_sources_dropped_after_eligibility':0}
OUT.mkdir(exist_ok=True)
(OUT/'independent_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
(OUT/'verified_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['records','geometry_summary']},indent=2))
