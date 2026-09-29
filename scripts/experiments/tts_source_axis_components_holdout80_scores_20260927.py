"""Frozen six-cell CPU scoring; old baseline/full source exact reuse, four strict gates."""
from pathlib import Path
import argparse,importlib.util,os,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments import tts_source_axis_components_holdout80_core_20260927 as core
from scripts.experiments import tts_fixed_generator_shift_holdout80_scores_20260927 as oldscore
O=core.OUT;P=core.PARENT;read=core.read;sha=core.sha;write=core.write
PREP=ROOT/'runs/tts_source_axis_components_holdout80_preparation_20260927'
spec=importlib.util.spec_from_file_location('frozen_axis_ops',PREP/'ops.py');ops=importlib.util.module_from_spec(spec);spec.loader.exec_module(ops)
CELLS=['baseline','parallel','orthogonal','full','masked_actual_gain','strict_orth']
CONDS=['parallel','orthogonal','masked_actual_gain','strict_orth']
GEOMS=oldscore.GEOMS;POLICIES=oldscore.POLICIES;METRICS=oldscore.METRICS

def gate():
 p,rows=core.locked();oldscore.runtime_gate(p);assert len(rows)==80 and len({r['speaker'] for r in rows})==40
 assert all(r['L']==r['arms']['N']['L'] for r in rows)
 return p,rows

def baseline():
 p,rows=gate();bindings={};cached=[];summary=read(P/'summary.json.gz');bindings[str(P/'summary.json.gz')]=sha(P/'summary.json.gz')
 for r in rows:
  core.limits();a=r['arms']['N'];assert sha(a['features_path'])==a['features_sha256']
  with np.load(a['features_path']) as f:A=f['audio'];native=f['visual']
  mp=P/'metadata'/r['id']/'N'/'global_plus.json';meta=read(mp);vp=Path(meta['V']['path']);assert sha(vp)==meta['V']['sha256']
  source=read(P/'scores'/(r['id']+'.json.gz'));baseline_source=read(P/'baseline_scores'/(r['id']+'.json.gz'));assert source['id']==r['id'] and source['speaker']==r['speaker'] and source['L']==r['L']
  for path in [mp,vp,P/'scores'/(r['id']+'.json.gz'),P/'baseline_scores'/(r['id']+'.json.gz')]:bindings[str(path)]=sha(path)
  cells={}
  for c,V,old in [('baseline',native,baseline_source['scores']),('full',np.load(vp),source['cells']['global_plus'])]:
   z=oldscore.scores(V,A,r['L']);assert z==old,(r['id'],c,'old curve/metric exact failed');cells[c]=z
  assert cells['baseline']==source['cells']['baseline']
  write(O/'baseline_scores'/(r['id']+'.json.gz'),{'id':r['id'],'speaker':r['speaker'],'L':r['L'],'cells':cells});cached.append((r,cells))
 for g in GEOMS:
  for pol in POLICIES:
   for m in METRICS:
    for c,old in [('baseline','baseline'),('full','global_plus')]:
     b=oldscore.bootstrap([z[c][g][pol][m] for r,z in cached],[r['speaker'] for r,z in cached]);assert b==summary[g+'/'+pol+'/'+m+'/'+old+'/value'],(g,pol,m,c)
 write(O/'baseline_validation.json',{'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(O/'protocol.json'),'clips':80,'speakers':40,'old_cells':['baseline','global_plus'],'views':8,'curve_metric_and_summary_max':0.,'files':bindings,'cached_scores':{str(f):sha(f) for f in sorted((O/'baseline_scores').glob('*.json.gz'))},'no_new_effect_scores':True,'resources':core.limits()})
 print('OLD_TWO_CELL_BRIDGE_PASS',flush=True)

def score():
 p,rows=gate();b=read(O/'baseline_validation.json');assert b['status']=='PASS' and b['protocol_sha256']==sha(O/'protocol.json')
 for key in ['files','cached_scores']:
  for path,h in b[key].items():assert sha(path)==h,path
 inp=read(O/'input_seal.json');assert inp['status']=='PASS' and inp['protocol_sha256']==sha(O/'protocol.json')
 for f,h in inp['files'].items():assert sha(f)==h,f
 assert read(O/'calibration_gate.json')['passed']
 e=read(O/'gpu_runtime/worker_exit_evaluation.json');assert e['stage_complete'] and e['completed']==320 and e['lease_released'] and e['temporary_files']==0
 bindings={str(O/'input_seal.json'):sha(O/'input_seal.json'),str(O/'gpu_runtime/worker_exit_evaluation.json'):sha(O/'gpu_runtime/worker_exit_evaluation.json')}
 for r in rows:
  for c in CONDS:
   mp=O/'metadata'/r['id']/'N'/(c+'.json');meta=read(mp);vp=O/'visual'/r['id']/'N'/(c+'.npy');assert meta['id']==r['id'] and meta['speaker']==r['speaker'] and meta['condition']==c and meta['arm']=='N'
   assert Path(meta['V']['path'])==vp and sha(vp)==meta['V']['sha256'];bindings[str(mp)]=sha(mp);bindings[str(vp)]=sha(vp)
 write(O/'feature_seal.json',{'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(O/'protocol.json'),'files':bindings})
 write(O/'score_lock.json',{'status':'BEFORE_ALL320_NEW_DISTANCE_SCORES','created_epoch':time.time(),'protocol_sha256':sha(O/'protocol.json'),'feature_seal_sha256':sha(O/'feature_seal.json')})
 for r in rows:
  core.limits();a=r['arms']['N'];assert sha(a['features_path'])==a['features_sha256']
  with np.load(a['features_path']) as f:A=f['audio']
  cells=read(O/'baseline_scores'/(r['id']+'.json.gz'))['cells']
  for c in CONDS:
   vp=O/'visual'/r['id']/'N'/(c+'.npy');assert sha(vp)==bindings[str(vp)];V=np.load(vp,allow_pickle=False);assert V.shape==(a['frames']-4,1024);cells[c]=oldscore.scores(V,A,r['L'])
  assert set(cells)==set(CELLS);write(O/'scores'/(r['id']+'.json.gz'),{'id':r['id'],'speaker':r['speaker'],'L':r['L'],'cells':cells});print('SCORED',r['id'],flush=True)
 write(O/'score_complete.json',{'status':'SCORED_PENDING_ANALYSIS','created_epoch':time.time(),'clips':80,'new_cells':320,'score_files':{str(f):sha(f) for f in sorted((O/'scores').glob('*.json.gz'))}})

def analyze():
 p,rows=gate();done=read(O/'score_complete.json');assert done['clips']==80 and done['new_cells']==320
 for f,h in done['score_files'].items():assert sha(f)==h
 rs=[read(O/'scores'/(r['id']+'.json.gz')) for r in rows];names=[r['speaker'] for r in rs];summary={};closure=0.
 for g in GEOMS:
  for pol in POLICIES:
   for m in METRICS:
    x=np.array([[r['cells'][c][g][pol][m] for c in CELLS] for r in rs]);effects=ops.contrasts(x);assert len(effects)==17
    closure=max(closure,float(np.max(abs(effects['parallel_effect']+effects['orthogonal_effect']+effects['interaction']-effects['total']))))
    series={c+'/value':x[:,i] for i,c in enumerate(CELLS)};series.update(effects)
    for key,value in series.items():summary[g+'/'+pol+'/'+m+'/'+key]=oldscore.bootstrap(value,names)
 assert closure<=1e-10 and len(summary)==1472
 criteria={m+'/'+c:summary['raw/guard20/'+m+'/'+c]['ci99'][0]>0 for m in ['C','C_anchor'] for c in ['strict_orth_minus_baseline','strict_orth_minus_parallel']}
 secondary={m+'/'+c:summary['raw/guard20/'+m+'/'+c]['ci99'][0]>0 for m in ['C','C_anchor'] for c in ['orthogonal_effect','orthogonal_minus_parallel']}
 medians={g:{c:float(np.median([r['cells'][c][g]['guard20']['best_lag'] for r in rs])) for c in CELLS} for g in GEOMS}
 write(O/'summary.json.gz',summary);write(O/'analysis.json',{'status':'PENDING_INDEPENDENT','created_epoch':time.time(),'four_primary_tests':criteria,'strict_orthogonal_transfer_supported':all(criteria.values()),'secondary_preorth_four_tests':secondary,'secondary_preorth_conjunction':all(secondary.values()),'statistic_count':len(summary),'clips':80,'speakers':40,'same_support_all8views':True,'original_preorth_factorial_closure_max':closure,'fixed_k':3,'median_guard20_best_lag':medians,'no_equivalence_claim_from_crossing_zero':True,'resources':core.limits()})
 print('ANALYZED',criteria,flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['baseline','score','analyze']);globals()[a.parse_args().stage]()
