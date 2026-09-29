"""Frozen projected-source displacement factorial; no fitting or audio forward."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')
import argparse,gzip,hashlib,json,math,shutil,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments import tts_fixed_generator_phone_prototype_20260927 as parent
from scripts.experiments import tts_fixed_generator_phone_stream_20260927 as stream
from scripts.experiments.tts_fixed_generator_shift_ops_20260927 import shift_table
OUT=ROOT/'runs/tts_fixed_generator_shift_cross_20260927';PARENT=parent.OUT
KINDS=['phone','global'];CELLS=['q00','q10','q01','q11'];GEOMS=parent.GEOMS;METRICS=parent.METRICS
CAP=144*2**20;AUDIT=16*2**20;FLOOR=int(4.05*2**30)
read=parent.read;sha=parent.sha;event=parent.event

def limits(extra=0,preparing=False):
 used=parent.allocated(OUT);free=shutil.disk_usage(OUT).free
 assert used+extra<=CAP and free-max(0,CAP-used)-AUDIT>=FLOOR,('new_resource',used,extra,free)
 if preparing:
  oldused=parent.allocated(PARENT)
  assert free-extra-max(0,parent.CAP-oldused)-parent.AUDIT>=parent.FLOOR,'parent remaining commitment floor'
 return {'allocated':used,'free':free,'remaining_commitment':max(0,CAP-used)+AUDIT,'floor':FLOOR}

def write(path,obj):
 data=json.dumps(obj,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()+b'\n';path=Path(path)
 if path.suffix=='.gz':data=gzip.compress(data,mtime=0)
 limits(math.ceil(len(data)/4096)*4096);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(data)

def save(path,**arrays):
 limits(sum(a.nbytes for a in arrays.values())+4096);assert not Path(path).exists();np.savez_compressed(path,**arrays);limits()

def freeze():
 assert not (OUT/'protocol.json').exists() and not (PARENT/'score_lock.json').exists()
 p=read(PARENT/'protocol.json');deps=dict(p['dependencies'])
 for f in [PARENT/'protocol.json',PARENT/'rows.json',PARENT/'fit_seal.json',PARENT/'fit.json',PARENT/'fit.npz',PARENT/'identity_gate.json',PARENT/'calibration_gate.json',OUT/'design.md',Path(__file__),ROOT/'scripts/experiments/tts_fixed_generator_shift_ops_20260927.py',ROOT/'scripts/experiments/tts_fixed_generator_shift_gpu_20260927.py']:
  deps[str(f)]=sha(f)
 c=dict(p['gpu_config']);c.update(own_cap_bytes=CAP,other_reserved_bytes=AUDIT,floor_bytes=FLOOR)
 c['prerequisite_closure']={'path':str(PARENT/'final.json'),'sha256':'BOUND_AFTER_PARENT_CONCLUSION'}
 rows=read(PARENT/'rows.json');ev=[r for r in rows if r['split']=='evaluation'];sizes=[(r['arms'][a]['frames']-4)*4096 for r in ev for a in 'NT' for _ in KINDS]
 q={'status':'FROZEN_BEFORE_PARENT_EFFECT_SCORES','created_epoch':time.time(),'gpu_config':c,'dependencies':deps,'conditions':KINDS,'cells':CELLS,'geometries':GEOMS,'metrics':METRICS,'event_metrics':event.METRICS,'parent_protocol_sha256':sha(PARENT/'protocol.json'),'parent_fit_sha256':sha(PARENT/'fit.npz'),'primary':'raw/guard20/common71/C/phone/source_S/N ci99 lower>0','components':'B=q10-q00; S=q01-q00; I=q11-q10-q01+q00; total=q11-q00','arithmetic':'old q00/q10/q11 byte references; new subtract32(other,own),multiply32(.5),add32(z),maximum32(0),inactive copy','name':'相对校准原型的残差收缩 × 带ReLU投影的来源位移','new_fit':False,'stats':p['stats'],'unit_definitions':p['unit_definitions'],'resources':{'main':CAP,'audit':AUDIT,'total':CAP+AUDIT,'floor':FLOOR,'new_eval_cells':296,'eval_V_raw_bytes':sum(sizes),'eval_V_allocated_bound':sum(math.ceil((s+128)/4096)*4096 for s in sizes),'cal_V_allocated_bound':1417216,'tmp_cap':96*2**20,'GPU_cumulative_seconds_including_live':3600,'CPU_max_threads':2},'parent_expected_dynamic_binding':{'final':str(PARENT/'final.json'),'all_old_features':str(PARENT/'feature_seal_evaluation.json'),'summaries':[str(PARENT/'summary.json.gz'),str(PARENT/'event_summary.json.gz')],'must_be_complete_before_GPU':True},'no_new_score_before_all_generation':True}
 limits(preparing=True);write(OUT/'protocol.json',q);write(OUT/'seal.json',{'protocol_sha256':sha(OUT/'protocol.json'),'design_sha256':sha(OUT/'design.md')});print('FROZEN',sha(OUT/'protocol.json'),flush=True)

def locked(require_parent=True):
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==read(OUT/'seal.json')['protocol_sha256']
 for path,h in p['dependencies'].items():assert sha(path)==h,path
 rv=read(OUT/'reviewer_pass.json');assert rv['status']=='PASS' and rv['protocol_sha256']==sha(OUT/'protocol.json') and sha(rv['receipt'])==rv['receipt_sha256']
 if require_parent:
  b=read(OUT/'parent_binding.json');assert b['status']=='PASS'
  for path,h in b['files'].items():assert sha(path)==h,path
  assert read(PARENT/'final.json')['status']=='concluded'
 limits();return p,read(PARENT/'rows.json')

def bind_parent():
 p,rows=locked(False);assert read(PARENT/'final.json')['status']=='concluded'
 assert p['created_epoch']<read(PARENT/'score_lock.json')['created_epoch'],'must precede all parent effects'
 assert read(PARENT/'independent_validation.json')['status']=='PASS'
 assert read(PARENT/'resource_closure.json')['passed']
 files={}
 for f in [PARENT/'final.json',PARENT/'resource_closure.json',PARENT/'independent_validation.json',PARENT/'score_lock.json',PARENT/'feature_seal_evaluation.json',PARENT/'summary.json.gz',PARENT/'event_summary.json.gz',PARENT/'event_query_metrics.npz',PARENT/'event_clip_metrics.npz',PARENT/'analysis.json']:
  files[str(f)]=sha(f)
 for path,h in read(PARENT/'feature_seal_evaluation.json').items():assert sha(path)==h;files[path]=h
 for r in rows:
  if r['split']=='evaluation':f=PARENT/'scores'/(r['id']+'.json.gz');files[str(f)]=sha(f)
  elif r['id']=='a1_001':
   for a in 'NT':
    for c in parent.CONDS[1:]:
     for f in [PARENT/'metadata'/r['id']/a/(c+'.json'),PARENT/'visual'/r['id']/a/(c+'.npy')]:files[str(f)]=sha(f)
 write(OUT/'parent_binding.json',{'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'parent_effects_follow_design':True,'files':files,'resources':limits()});print('PARENT_BOUND',flush=True)

def error(a,b):
 if isinstance(a,dict):assert a.keys()==b.keys();return max([error(a[k],b[k]) for k in a]+[0.])
 if isinstance(a,(list,tuple)):assert len(a)==len(b);return max([error(x,y) for x,y in zip(a,b)]+[0.])
 if isinstance(a,(int,float,np.number)):return abs(float(a)-float(b))
 assert a==b;return 0.

def bridge():
 import torch;torch.set_num_threads(2)
 p,rows=locked();fm=read(PARENT/'fit.json');fit=np.load(PARENT/'fit.npz');maxscore=0.;checked=0;description={}
 for r in rows:
  if r['split']!='evaluation' and r['id']!='a1_001':continue
  score=read(PARENT/'scores'/(r['id']+'.json.gz')) if r['split']=='evaluation' else None
  fi=fm['folds'].index(r['speaker'])
  for ai,a in enumerate('NT'):
   q=r['arms'][a]
   with np.load(q['features_path']) as f:z=f['z'];A=f['audio'];V=f['visual']
   if score:maxscore=max(maxscore,error(parent.official_scores(V,A,q['L']),score['cells']['baseline'][a]))
   own,og=stream.template_table(fit['raw_mu'],fit['speaker_counts'],fm['labels'],fi,ai)
   other,tg=stream.template_table(fit['raw_mu'],fit['speaker_counts'],fm['labels'],fi,1-ai)
   for kind in KINDS:
    _,m=shift_table(z,q['phone_labels'],q['speech_mask'],own,og,other,tg,kind)
    for mode,source in [('own',a),('other','T' if a=='N' else 'N')]:
     cond=kind+source;meta=read(PARENT/'metadata'/r['id']/a/(cond+'.json'))
     assert m[mode+'_z_raw_sha256']==meta['mix']['replacement_z_raw_sha256']
     assert m['input_z_raw_sha256']==meta['mix']['input_z_raw_sha256']
     if score:
      v=np.load(meta['V']['path']);assert sha(meta['V']['path'])==meta['V']['sha256']
      maxscore=max(maxscore,error(parent.official_scores(v,A,q['L']),score['cells'][cond][a]))
     checked+=1
    description[r['id']+'/'+a+'/'+kind]=m
 assert maxscore==0.
 write(OUT/'bridge_validation.json',{'status':'PASS','created_epoch':time.time(),'parent_cell_input_hash_checks':checked,'parent_official_curves_metrics_max':maxscore,'old_event_bytes_reused_sha256':sha(PARENT/'event_query_metrics.npz'),'no_new_scores':True})
 write(OUT/'operation_descriptions.json.gz',description);print('BRIDGE_PASS',checked,maxscore,flush=True)

def score():
 p,rows=locked();assert read(OUT/'calibration_gate.json')['passed'] and read(OUT/'bridge_validation.json')['status']=='PASS';assert read(OUT/'gpu_runtime/worker_exit_evaluation.json')['stage_complete']
 import torch;torch.set_num_threads(2)
 ev=[r for r in rows if r['split']=='evaluation'];idx=read(parent.EVENT/'indices.json');byid={r['id']:r for r in idx['rows']};spans={r['id']:r for r in idx['spans']}
 with np.load(PARENT/'event_query_metrics.npz') as f:old=f['metrics']
 eq=np.zeros((1411,2,2,4,2,4));seal={}
 for r in ev:
  for a in 'NT':
   for kind in KINDS:
    mp=OUT/'metadata'/r['id']/a/(kind+'.json');m=read(mp);vp=Path(m['V']['path']);assert sha(vp)==m['V']['sha256'];seal[str(mp)]=sha(mp);seal[str(vp)]=sha(vp)
 write(OUT/'feature_seal_evaluation.json',seal);write(OUT/'score_lock.json',{'created_epoch':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'features_sha256':sha(OUT/'feature_seal_evaluation.json')})
 for r in ev:
  baseline=read(PARENT/'scores'/(r['id']+'.json.gz'))['cells'];result={};audio={}
  for a in 'NT':
   with np.load(r['arms'][a]['features_path']) as f:audio[a]=f['audio']
  for ki,kind in enumerate(KINDS):
   result[kind]={c:{} for c in CELLS};arrays={}
   for ai,a in enumerate('NT'):
    for ci,c in enumerate(CELLS):
     if c!='q01':
      cond='baseline' if c=='q00' else kind+(a if c=='q10' else ('T' if a=='N' else 'N'))
      result[kind][c][a]=baseline[cond][a]
      if r['id'] in spans:
       s=spans[r['id']];eq[s['start']:s['stop'],:,ki,ci,ai]=old[s['start']:s['stop'],:,parent.CONDS.index(cond),ai]
     else:
      v=np.load(OUT/'visual'/r['id']/a/(kind+'.npy'));assert v.shape==(r['arms'][a]['frames']-4,1024)
      result[kind][c][a]=parent.official_scores(v,audio[a],r['arms'][a]['L']);arrays['A','FIXED',a]=audio[a];arrays['V','FIXED',a]=v
   if r['id'] in byid:
    z=event.geometry(byid[r['id']],arrays,[{'video_state':'FIXED','audio_state':'FIXED','video_source':a,'audio_source':a} for a in 'NT']);s=spans[r['id']];eq[s['start']:s['stop'],:,ki,2]=z
  write(OUT/'scores'/(r['id']+'.json.gz'),{'id':r['id'],'speaker':r['speaker'],'eligible':r['guard20_eligible'],'cells':result});print('scored',r['id'],flush=True)
 save(OUT/'event_query_metrics.npz',metrics=eq);save(OUT/'event_clip_metrics.npz',metrics=event.clipmeans(eq,idx['spans']))
 write(OUT/'score_complete.json',{'status':'SCORED_PENDING_ANALYSIS','created_epoch':time.time(),'all74':True,'new_cells':296})

def contrasts(x):
 # x=[clip,kind,cell,source], cell order 00,10,01,11.
 series={};closure=0.
 for ki,kind in enumerate(KINDS):
  v=x[:,ki];B=v[:,1]-v[:,0];S=v[:,2]-v[:,0];I=v[:,3]-v[:,1]-v[:,2]+v[:,0];total=v[:,3]-v[:,0]
  components={'component_B':B,'source_S':S,'interaction_I':I,'total':total,'S_after_B':v[:,3]-v[:,1],'B_after_S':v[:,3]-v[:,2]}
  closure=max(closure,float(np.max(np.abs(B+S+I-total))))
  for ci,c in enumerate(CELLS):
   for ai,a in enumerate('NT'):series[kind+'/cell/'+c+'/'+a]=v[:,ci,ai]
   series[kind+'/gap/'+c]=v[:,ci,1]-v[:,ci,0]
  for name,values in components.items():
   for ai,a in enumerate('NT'):series[kind+'/'+name+'/'+a]=values[:,ai]
   series[kind+'/'+name+'/TminusN']=values[:,1]-values[:,0]
 for key in list(series):
  if key.startswith('phone/'):
   suffix=key[6:];series['phone_minus_global/'+suffix]=series[key]-series['global/'+suffix]
 return series,closure

def analyze():
 p,rows=locked();assert read(OUT/'score_complete.json')['all74'];rs=[read(OUT/'scores'/(r['id']+'.json.gz')) for r in rows if r['split']=='evaluation'];ss={};maxerr=0.
 for g in GEOMS:
  for policy,support in [('guard20','common71'),('valid','common71'),('valid','all74'),('guard0','common71'),('guard0','all74')]:
   rr=[r for r in rs if r['eligible']] if support=='common71' else rs;names=[r['speaker'] for r in rr]
   for metric in METRICS:
    vals=np.array([[[[r['cells'][k][c][a][g][policy][metric] for a in 'NT'] for c in CELLS] for k in KINDS] for r in rr]);series,err=contrasts(vals);maxerr=max(maxerr,err)
    for key,arr in series.items():ss[f'{g}/{policy}/{support}/{metric}/{key}']=parent.F['bootstrap'](arr,names)
 idx=read(parent.EVENT/'indices.json');names=[r['speaker'] for r in idx['spans']];es={}
 with np.load(OUT/'event_clip_metrics.npz') as f:cm=f['metrics']
 for gi,g in enumerate(GEOMS):
  for mi,m in enumerate(event.METRICS):
   series,err=contrasts(cm[:,gi,:,:,:,mi]);maxerr=max(maxerr,err)
   for key,arr in series.items():es[g+'/'+m+'/'+key]=parent.F['bootstrap'](arr,names)
 assert maxerr<=1e-10;write(OUT/'summary.json.gz',ss);write(OUT/'event_summary.json.gz',es)
 main=ss['raw/guard20/common71/C/phone/source_S/N']
 write(OUT/'analysis.json',{'status':'PENDING_INDEPENDENT','created_epoch':time.time(),'natural_phone_source_shift_gain':main['ci99'][0]>0,'primary':main,'factorial_closure_max':maxerr,'official_endpoints':len(ss),'event_endpoints':len(es),'resources':limits()});print('ANALYZED',main,flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['freeze','bind_parent','bridge','score','analyze']);globals()[a.parse_args().stage]()
