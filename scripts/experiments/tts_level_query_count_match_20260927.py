"""Frozen native-query-count subsampling diagnostic; no waveform or time edits."""
from pathlib import Path
import sys,types,argparse,time,shutil,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));OUT=ROOT/'runs/tts_level_query_count_match_20260927';PARENT=ROOT/'runs/tts_level_residual_representation_20260927'
code=PARENT/'code_snapshot/tts_level_residual_representation_20260927.py';m=types.ModuleType('residual_parent');m.__file__=str(ROOT/'scripts/experiments'/code.name);exec(compile(code.read_text(),str(code),'exec'),m.__dict__)
read,write,sha=m.read,m.write,m.sha;FIELDS=m.FIELDS+['search_uplift'];BGS=m.BGS;GEOMS=m.GEOMS

def resource():
 free=shutil.disk_usage(ROOT).free;size=sum(x.stat().st_size for x in OUT.rglob('*') if x.is_file());assert free>=5*2**30 and size<=100*2**20
 return {'free_bytes':free,'run_bytes':size}
def frozen():
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
 for f,h in p['bindings'].items():assert sha(f)==h,f
 resource();return p
def metric(curves):
 z=m.perm.curve_metrics(curves,3);return np.column_stack([z,z[:,3]-z[:,2]])
def freeze():
 assert not (OUT/'protocol.json').exists();p=read(OUT/'protocol_design.json');assert sha(OUT/'protocol_design.json')==(OUT/'protocol_design.sha256').read_text().strip();pp=m.frozen();by={r['id']:r for r in pp['rows']};bindings=dict(p['bindings']);rows=[]
 for info in p['support']:
  row=by[info['id']];rows.append(row);sid=row['id'];q=info['q']
  for a in ['N','T']:
   n=row['arms'][a]['joint_L'];ii=np.arange(20,n-20);assert len(ii)==info['I_counts'][a];sub=np.empty((1024,q),dtype=np.uint16)
   for b in range(1024):
    seed=int.from_bytes(hashlib.sha256(f'20260927|query_count|{sid}|{a}|{b}'.encode()).digest()[:16],'big');rng=np.random.Generator(np.random.PCG64(seed));sub[b]=np.sort(rng.choice(ii,q,replace=False))
   assert sub.min()>=ii.min() and sub.max()<=ii.max() and (q==1 or np.all(np.diff(sub.astype(int),axis=1)>0))
   path=OUT/'indices'/sid/(a+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,indices=sub);bindings[str(path)]=sha(path)
   for c in ['raw','LEVEL']:
    for ext in ['.npz','.json']:
     f=m.LEVEL/'features'/sid/a/(c+ext);assert sha(f)==pp['bindings'][str(f)];bindings[str(f)]=sha(f)
  f=PARENT/'baseline'/(sid+'.json');bindings[str(f)]=sha(f)
 for f in [code,m.ENGINE,m.OLD/'code_at_freeze.py',ROOT/'scripts/experiments/tts_native_midpoint.py',ROOT/'scripts/experiments/tts_native_boundary_audit.py',Path(__file__),ROOT/'scripts/experiments/check_tts_level_query_count_match_20260927.py']:bindings[str(f)]=sha(f)
 p.update({'status':'frozen_all_indices_before_any_subset_score','approved_parent':True,'created_epoch_fullseal':time.time(),'rows':rows,'bindings':bindings});write(OUT/'protocol.json',p);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
 for f in [Path(__file__),ROOT/'scripts/experiments/check_tts_level_query_count_match_20260927.py']:
  dest=OUT/'code_snapshot'/f.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
 print('FROZEN',sha(OUT/'protocol.json'),len(bindings),resource(),flush=True)
def baseline():
 import torch
 torch.set_num_threads(2);p=frozen();err=0.;cells=0
 for row in p['rows']:
  sid=row['id'];old=read(PARENT/'baseline'/(sid+'.json'));arrays={}
  for bg in BGS:
   for geom in GEOMS:
    for a in ['N','T']:
     v,au=m.inputs(row,a,bg,geom);d=m.base.matrix(v,au);cur=d[20:-20].mean(0);expected=old['cells'][f'{bg}/{geom}/{a}']['guard20'];val=metric(cur[None,:])[0];err=max(err,float(abs(cur-expected['curve']).max()),max(abs(val[j]-expected[f]) for j,f in enumerate(m.FIELDS)));arrays['/'.join([bg,geom,a])]=d;cells+=1
  dest=OUT/'distances'/(sid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**arrays)
 result={'passed':err<=1e-5,'max':err,'cells':cells};write(OUT/'baseline_validation.json',result);assert result['passed'];lock={str(f):sha(f) for f in (OUT/'distances').glob('*.npz')};write(OUT/'score_lock.json',{'time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'baseline_sha256':sha(OUT/'baseline_validation.json'),'distances':lock,'status':'all_full_query_baselines_pass_before_subset_scoring'});print('BASELINE',result,flush=True)
def score():
 p=frozen();lock=read(OUT/'score_lock.json');assert read(OUT/'baseline_validation.json')['passed'] and sha(OUT/'baseline_validation.json')==lock['baseline_sha256'];controls=[]
 for f,h in lock['distances'].items():assert sha(f)==h
 for row in p['rows']:
  sid=row['id'];d=np.load(OUT/'distances'/(sid+'.npz'))
  for bg in BGS:
   for geom in GEOMS:
    arrays={}
    for a in ['N','T']:
     matrix=d['/'.join([bg,geom,a])];sub=np.load(OUT/'indices'/sid/(a+'.npz'))['indices'];curves=np.empty((1024,31))
     for start in range(0,1024,64):curves[start:start+64]=matrix[sub[start:start+64]].mean(1)
     full=matrix[20:-20].mean(0);mean=curves.mean(0);arrays[a+'/full_curve']=full;arrays[a+'/mean_curve']=mean;arrays[a+'/full_metrics']=metric(full[None,:]);arrays[a+'/subset_metrics']=metric(curves);arrays[a+'/mean_curve_metrics']=metric(mean[None,:]);q=sub.shape[1];identity_err=float(abs(curves-full).max()) if q==len(matrix)-40 else None
     if identity_err is not None:assert identity_err<1e-12
     controls.append({'id':sid,'background':bg,'geometry':geom,'arm':a,'q':q,'I':len(matrix)-40,'full_query_identity_max':identity_err,'mean_curve_vs_full_MC_max':float(abs(mean-full).max()),'min_jensen':float(arrays[a+'/mean_curve_metrics'][0,2]-arrays[a+'/subset_metrics'][:,2].mean()),'finite':bool(np.isfinite(curves).all())});assert np.isfinite(curves).all()
    dest=OUT/'scores'/bg/geom/(sid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**arrays)
  resource();print('scored',sid,flush=True)
 write(OUT/'controls.json',controls);print('SCORED',resource(),flush=True)
def values(sid,bg,geom,count=1024):
 z=np.load(OUT/'scores'/bg/geom/(sid+'.npz'));result={}
 for a in ['N','T']:
  full=z[a+'/full_metrics'][0];matched=z[a+'/subset_metrics'][:count].mean(0);mean=z[a+'/mean_curve_metrics'][0];result[a]={'full':full,'matched':matched,'matched_minus_full':matched-full,'metric_of_mean_curve':mean,'mean_curve_minus_full':mean-full,'mean_metric_minus_metric_mean':matched-mean}
 result['gap']={k:result['T'][k]-result['N'][k] for k in result['N']};return result

def analyze():
 p=frozen();rows=p['rows'];sp=[r['speaker'] for r in rows];result={};mc={}
 for geom in GEOMS:
  vv={bg:[values(r['id'],bg,geom) for r in rows] for bg in BGS}
  for bg in BGS+['LEVEL_minus_RAW']:
   for a in ['N','T','gap']:
    for term in vv['RAW'][0][a]:
     vals=np.stack([r[a][term] for r in vv[bg]]) if bg in BGS else np.stack([x[a][term]-y[a][term] for x,y in zip(vv['LEVEL'],vv['RAW'])]);result[f'{geom}/{bg}/{a}/{term}']={f:m.base.bootstrap(vals[:,j],sp) for j,f in enumerate(FIELDS)}
  for count in [256,512]:
   part={bg:[values(r['id'],bg,geom,count) for r in rows] for bg in BGS}
   for bg in BGS+['LEVEL_minus_RAW']:
    for a in ['N','T','gap']:
     diff=np.stack([x[a]['matched']-y[a]['matched'] for x,y in zip(part[bg],vv[bg])]) if bg in BGS else np.stack([(u[a]['matched']-v[a]['matched'])-(x[a]['matched']-y[a]['matched']) for u,v,x,y in zip(part['LEVEL'],part['RAW'],vv['LEVEL'],vv['RAW'])]);mean=np.stack([diff[np.array(sp)==s].mean(0) for s in sorted(set(sp))]).mean(0);mc[f'{count}/{geom}/{bg}/{a}/matched']=dict(zip(FIELDS,mean.tolist()))
 support=p['support'];desc={'q':m.base.bootstrap([r['q'] for r in support],sp),'q_min':min(r['q'] for r in support),'q_max':max(r['q'] for r in support),'q_le5':[r['id'] for r in support if r['q']<=5],'arms':{}}
 for a in ['N','T']:desc['arms'][a]={'I':m.base.bootstrap([r['I_counts'][a] for r in support],sp),'removed':m.base.bootstrap([r['removed'][a] for r in support],sp),'removed_fraction':m.base.bootstrap([r['removed_fraction'][a] for r in support],sp),'unchanged_clips':sum(r['removed'][a]==0 for r in support)}
 write(OUT/'summary.json',result);write(OUT/'mc_precision.json',mc);write(OUT/'support_summary.json',desc);print('ANALYZED',len(result),resource(),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','baseline','score','analyze']);globals()[ap.parse_args().stage]()
