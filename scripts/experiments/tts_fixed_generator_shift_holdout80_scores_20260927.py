"""CPU-only frozen natural80 bridge, projected source-shift scores and paired statistics."""
from pathlib import Path
import argparse, ast, os, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments import tts_fixed_generator_shift_holdout80_20260927 as core
OUT=core.OUT;OLD=core.OLD;read=core.read;write=core.write;sha=core.sha
GEOMS=['raw','unit'];POLICIES=['guard20','valid','guard0','guard15']
METRICS=['C','B','D','C_anchor','D_anchor','search_uplift','best_lag','offset']
CONDS=['baseline']+core.CONDS
ENGINE=ROOT/'runs/tts_acoustic_generation_cross_20260926/code_snapshot/tts_acoustic_generation_cross_20260926.py'
OLD_CODE=OLD/'code_snapshot/tts_fixed_natural_holdout80_20260927.py'
def frozen_functions():
 ns={'np':np,'LAGS':list(range(-15,16))}
 for path,names in [(ENGINE,{'matrix','unit'}),(OLD_CODE,{'summarize'})]:
  tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];assert {n.name for n in nodes}==names
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 return ns
F=frozen_functions()
def runtime_gate(p):
 import torch
 assert core.runtime()==p['gpu_config']['runtime'];torch.set_num_threads(2)
 for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ.get(k)=='1'
 assert p['scoring']['k']==3 and p['scoring']['lags']==list(range(-15,16))
 assert p['scoring']['policies']==POLICIES and p['scoring']['geometries']==GEOMS

def scores(v,a,L):
 assert v.dtype==a.dtype==np.float32 and v.ndim==a.ndim==2 and v.shape[1]==a.shape[1]==1024
 assert len(v)>=L and len(a)>=L and L>40 and np.isfinite(v).all() and np.isfinite(a).all()
 result={}
 for g in GEOMS:
  vv,aa=v[:L],a[:L]
  if g=='unit':vv,aa=F['unit'](vv),F['unit'](aa)
  matrix=F['matrix'](vv,aa);result[g]={}
  for pol in POLICIES:
   z=F['summarize'](matrix,pol);z['search_uplift']=z['D_anchor']-z['D'];z['offset']=-z['best_lag'];result[g][pol]=z
 return result

def bootstrap(values,speakers):
 values=np.asarray(values,dtype=np.float64);assert values.shape==(80,) and np.isfinite(values).all()
 groups=sorted(set(speakers));assert len(groups)==40 and all(speakers.count(g)==2 for g in groups)
 means=np.array([np.mean([v for v,g in zip(values,speakers) if g==name]) for name in groups])
 ix=np.random.Generator(np.random.PCG64(20260926)).integers(0,40,(20000,40));draws=means[ix].mean(1)
 return {'mean':float(means.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),'ci99':np.quantile(draws,[.005,.995]).tolist(),'n':80,'speakers':40,'group_means':dict(zip(groups,means.tolist()))}

def baseline():
 p,rows=core.locked();runtime_gate(p);assert len(rows)==80
 bindings={};err=0.;cached=[];oldsummary=read(OLD/'summary.json');bindings[str(OLD/'summary.json')]=sha(OLD/'summary.json')
 for r in rows:
  core.limits();a=r['arms']['N'];assert sha(a['features_path'])==a['features_sha256']
  with np.load(a['features_path']) as f:z=scores(f['visual'],f['audio'],r['L'])
  op=OLD/'scores'/(r['id']+'.json');old=read(op);bindings[str(op)]=sha(op)
  assert old['id']==r['id'] and old['speaker']==r['speaker'] and old['L']==r['L']
  for g in GEOMS:
   for pol in POLICIES:
    want=old['cells'][g]['q11']['policies'][pol]
    for k,v in want.items():
     e=float(np.max(np.abs(np.asarray(z[g][pol][k])-np.asarray(v))));err=max(err,e);assert e==0.,(r['id'],g,pol,k,e)
  write(OUT/'baseline_scores'/(r['id']+'.json.gz'),{'id':r['id'],'speaker':r['speaker'],'L':r['L'],'scores':z});cached.append((r,z))
 for g in GEOMS:
  for pol in POLICIES:
   for m in p['scoring']['fields']:
    got=bootstrap([z[g][pol][m] for r,z in cached],[r['speaker'] for r,z in cached]);want=oldsummary['results'][g+'/'+pol][m]['FIXED_native']
    for k in ['mean','ci99','n','speakers']:assert got[k]==want[k],(g,pol,m,k)
    assert got['group_means']==want['group_means']
 write(OUT/'baseline_validation.json',{'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'input_rows_sha256':sha(OUT/'rows.json'),'clips':80,'speakers':40,'views':8,'curve_and_metric_max_error':err,'summary99_exact':True,'old_score_bindings':bindings,'cached_scores':{str(f):sha(f) for f in sorted((OUT/'baseline_scores').glob('*.json.gz'))},'runtime':core.runtime(),'new_intervention_scores':False,'resources':core.limits()})
 print('BASELINE_PASS',err,flush=True)

def validate_baseline():
 b=read(OUT/'baseline_validation.json');assert b['status']=='PASS' and b['protocol_sha256']==sha(OUT/'protocol.json') and b['input_rows_sha256']==sha(OUT/'rows.json')
 for group in ['old_score_bindings','cached_scores']:
  for f,h in b[group].items():assert sha(f)==h,f
 return b

def score():
 p,rows=core.locked();runtime_gate(p);validate_baseline()
 inp=read(OUT/'input_seal.json');assert inp['status']=='PASS' and inp['protocol_sha256']==sha(OUT/'protocol.json')
 for f,h in inp['files'].items():assert sha(f)==h,f
 rv=read(OUT/'input_reviewer_pass.json');assert rv['status']=='PASS' and rv['input_seal_sha256']==sha(OUT/'input_seal.json') and sha(rv['receipt'])==rv['receipt_sha256']
 for name in ['baseline','calibration']:assert read(OUT/(name+'_gate.json'))['passed']
 e=read(OUT/'gpu_runtime/worker_exit_evaluation.json');assert e['stage_complete'] and e['completed']==320 and e['lease_released'] and e['temporary_files']==0
 seal={str(OUT/'input_seal.json'):sha(OUT/'input_seal.json'),str(OUT/'gpu_runtime/worker_exit_evaluation.json'):sha(OUT/'gpu_runtime/worker_exit_evaluation.json')}
 for r in rows:
  for cond in core.CONDS:
   mp=OUT/'metadata'/r['id']/'N'/(cond+'.json');m=read(mp);vp=OUT/'visual'/r['id']/'N'/(cond+'.npy')
   assert m['id']==r['id'] and m['speaker']==r['speaker'] and m['condition']==cond and m['arm']=='N'
   assert Path(m['V']['path'])==vp and sha(vp)==m['V']['sha256']
   assert m['parentbindings']['features_sha256']==r['arms']['N']['features_sha256']
   seal[str(mp)]=sha(mp);seal[str(vp)]=sha(vp)
 write(OUT/'feature_seal.json',{'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'files':seal})
 write(OUT/'score_lock.json',{'created_epoch':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'feature_seal_sha256':sha(OUT/'feature_seal.json'),'status':'BEFORE_ALL320_NEW_DISTANCE_SCORES'})
 for r in rows:
  core.limits();a=r['arms']['N'];assert sha(a['features_path'])==a['features_sha256']
  with np.load(a['features_path']) as f:A=f['audio']
  cells={'baseline':read(OUT/'baseline_scores'/(r['id']+'.json.gz'))['scores']}
  for cond in core.CONDS:
   vp=OUT/'visual'/r['id']/'N'/(cond+'.npy');assert sha(vp)==seal[str(vp)];V=np.load(vp,allow_pickle=False);assert V.shape==(a['frames']-4,1024)
   cells[cond]=scores(V,A,r['L'])
  write(OUT/'scores'/(r['id']+'.json.gz'),{'id':r['id'],'speaker':r['speaker'],'L':r['L'],'cells':cells});print('SCORED',r['id'],flush=True)
 write(OUT/'score_complete.json',{'status':'SCORED_PENDING_ANALYSIS','created_epoch':time.time(),'clips':80,'new_cells':320,'score_files':{str(f):sha(f) for f in sorted((OUT/'scores').glob('*.json.gz'))}})

def contrasts(x):
 # x=[clip, baseline/phone+/phone-/global+/global-], no support changes.
 out={c+'/value':x[:,i] for i,c in enumerate(CONDS)}
 for family in ['phone','global']:
  plus=x[:,CONDS.index(family+'_plus')];minus=x[:,CONDS.index(family+'_minus')]
  out[family+'/plus_minus_baseline']=plus-x[:,0]
  out[family+'/minus_minus_baseline']=minus-x[:,0]
  out[family+'/plus_minus_minus']=plus-minus
 for name in ['plus_minus_baseline','minus_minus_baseline','plus_minus_minus']:
  out['phone_minus_global/'+name]=out['phone/'+name]-out['global/'+name]
 return out

def analyze():
 p,rows=core.locked();runtime_gate(p);done=read(OUT/'score_complete.json');assert done['clips']==80 and done['new_cells']==320
 for f,h in done['score_files'].items():assert sha(f)==h
 rs=[read(OUT/'scores'/(r['id']+'.json.gz')) for r in rows];names=[r['speaker'] for r in rs];summary={};closure=0.
 for g in GEOMS:
  for pol in POLICIES:
   for metric in METRICS:
    x=np.array([[r['cells'][c][g][pol][metric] for c in CONDS] for r in rs]);series=contrasts(x)
    for key,v in series.items():summary[g+'/'+pol+'/'+metric+'/'+key]=bootstrap(v,names)
    for family in ['phone','global']:
     closure=max(closure,float(np.max(abs(series[family+'/plus_minus_baseline']-series[family+'/minus_minus_baseline']-series[family+'/plus_minus_minus']))))
 assert closure<=1e-10
 criteria={m+'/'+contrast:summary['raw/guard20/'+m+'/global/'+contrast]['ci99'][0]>0 for m in ['C','C_anchor'] for contrast in ['plus_minus_baseline','plus_minus_minus']}
 medians={g:{c:float(np.median([r['cells'][c][g]['guard20']['best_lag'] for r in rs])) for c in CONDS} for g in GEOMS}
 write(OUT/'summary.json.gz',summary)
 write(OUT/'analysis.json',{'status':'PENDING_INDEPENDENT','created_epoch':time.time(),'four_primary_tests':criteria,'global_direction_transfer_supported':all(criteria.values()),'statistic_count':len(summary),'clips':80,'speakers':40,'support':'identical all80 for all8views','paired_contrast_closure_max':closure,'fixed_k':3,'median_guard20_best_lag':medians,'lag_diagnostic_only_no_reselection':True,'resources':core.limits()})
 print('ANALYZED',criteria,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['baseline','score','analyze']);globals()[ap.parse_args().stage]()
