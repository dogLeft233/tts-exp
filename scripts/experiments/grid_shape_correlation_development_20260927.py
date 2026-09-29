"""One sealed CPU-only measurement-development calculation; no media or model access."""
from pathlib import Path
import hashlib, json, shutil, sys, resource
import numpy as np
import check_grid_geometry_calibration_20260927 as geometry

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/grid_shape_correlation_development_20260927'
OLD=ROOT/'runs/grid_geometry_calibration_20260927'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
 p=read(OUT/'protocol.json');seal=read(OUT/'seal.json')
 assert sha(OUT/'protocol.json')==seal['protocol_sha256']
 review=read(OUT/'reviewer_pass.json')
 assert review['status']=='PASS' and review['protocol_sha256']==seal['protocol_sha256']
 assert not (OUT/'diagnostics.json').exists() and not (OUT/'calculation_started.json').exists()
 assert sys.executable==p['python'] and np.__version__==p['numpy']
 for path,digest in p['bindings'].items():assert sha(ROOT/path)==digest,path
 assert shutil.disk_usage(ROOT).free-(1024**2)>=5*1024**3
 hashes=read(OLD/'scientific_feature_seal.json')
 for path,digest in hashes.items():
  q=Path(path);assert q.parts[:3]==('runs','grid_geometry_calibration_20260927','landmarks') and q.parts[3] in p['ids'];assert sha(ROOT/q)==digest
 write(OUT/'calculation_started.json',{'protocol_sha256':seal['protocol_sha256'],'old_scientific_files_verified':len(hashes),'review_sha256':sha(OUT/'reviewer_pass.json')})
 old=read(OLD/'calibration_diagnostics.json');oldrows={x['id']:x for x in old['rows']}
 T=np.arange(14,55);ls=np.arange(-5,6);lag=read(OLD/'lag_lock.json')['lag'];assert lag==-3
 records=[]
 for sid in p['ids']:
  ref=np.load(OLD/'landmarks'/sid/'REAL/BASE.npz');cache={}
  def fetch(domain,view='BASE'):
   key=(domain,view)
   if key not in cache:
    with np.load(OLD/'landmarks'/sid/domain/(view+'.npz')) as z:
     assert np.array_equal(z['pts'],np.arange(75)/25)
     cache[key]=geometry.geom(z,ref)
   return cache[key]
  input_pass=bool(ref['valid'][0]) and geometry.qc(fetch('REAL'))['passed']
  assert input_pass==oldrows[sid]['input_pass']
  row={'id':sid,'input_pass':input_pass,'domains':{}}
  for domain in p['domains']:
   data={v:fetch(domain,v) for v in p['views']}
   t=np.array([i for i in T if all(all(g[1][i+l] for l in ls) for g in data.values())],dtype=int)
   assert t.tolist()==oldrows[sid]['domains'][domain]['indices']
   q=geometry.qc(data['BASE'],real=domain=='REAL');assert q['passed']==oldrows[sid]['domains'][domain]['baseline_QC_pass']
   d={'indices':t.tolist(),'support_pass':len(t)>=33,'base_QC_pass':q['passed'],'controls':{}}
   checks=[len(t)>=33,q['passed']]
   if len(t)>=33:
    y=data['BASE'][0];scores={v:geometry.metric(y[t],g[0][t]) for v,g in data.items()}
    sd=float(np.std(y[t],ddof=0));noise=max(scores[v]['centered_rmse'] for v in p['spatial'])
    d.update(native_sd=sd,max_spatial_centered_rmse=noise,motion_noise_ratio=sd/noise if noise>0 else None,motion_pass=bool(sd>=.005 and sd>=2*noise))
    checks.append(d['motion_pass'])
    for v,s in scores.items():
     c={k:s[k] for k in ['r','centered_rmse','reference_sd','candidate_sd']};ok=True
     if v=='REPEAT':ok=s['r'] is not None and s['r']>=.99
     elif v in p['spatial']:ok=s['r'] is not None and s['r']>=.95
     elif v in ['REVERSE','WARP_0.8','WARP_1.2']:
      drop=scores['REPEAT']['r']-s['r'] if scores['REPEAT']['r'] is not None and s['r'] is not None else None
      c['repeat_minus_control_r']=drop;ok=drop is not None and drop>=.05
     elif v.startswith('SHIFT_'):
      delta=int(v.split('_')[1]);rr=[geometry.metric(y[t],data[v][0][t+l])['r'] for l in ls];best=geometry.choose(rr,ls)
      c.update(r_by_lag=rr,recovered_lag=best,expected=delta);ok=best is not None and best*delta>0 and abs(best-delta)<=1
     elif v=='FROZEN':
      c['zero_dynamic_NA_check']=bool(s['candidate_sd']<=1e-6 and s['r'] is None)
      assert c['zero_dynamic_NA_check'],'FROZEN engineering definition mismatch'
      c['scientific_gate']=False
     c['passed']=bool(ok);d['controls'][v]=c
     if v not in ['BASE','FROZEN']:checks.append(ok)
   d['passed']=bool(all(checks));row['domains'][domain]=d
  y,yv,*_=fetch('REAL');raw={v:fetch('RAW',v) for v in ['BASE','SHIFT_-4','SHIFT_-2','SHIFT_2','SHIFT_4']};search=ls+lag
  t=np.array([i for i in T if yv[i] and all(all(g[1][i+l] for l in search) for g in raw.values())],dtype=int)
  assert t.tolist()==oldrows[sid]['cross_domain']['indices']
  c={'indices':t.tolist(),'support_pass':len(t)>=33,'shift':{}};cp=False
  if len(t)>=33:
   r=geometry.metric(y[t],raw['BASE'][0][t+lag])['r'];r0=geometry.metric(y[t],raw['BASE'][0][t])['r']
   b=geometry.choose([geometry.metric(y[t],raw['BASE'][0][t+l])['r'] for l in search],search)
   c.update(RAW_r=r,RAW_lag0_r=r0,base_optimum=b);flags=[]
   for v in list(raw)[1:]:
    delta=int(v.split('_')[1]);rr=[geometry.metric(y[t],raw[v][0][t+l])['r'] for l in search];best=geometry.choose(rr,search)
    ok=best is not None and b is not None and (best-b)*delta>0 and abs(best-b-delta)<=1
    c['shift'][v]={'r_by_lag':rr,'recovered_lag':best,'relative_shift':best-b if best is not None and b is not None else None,'passed':bool(ok)};flags.append(ok)
   cp=r is not None and r>=.30 and all(flags)
  av=fetch('RAW')[1];fv=fetch('FIXED')[1];common=[int(i) for i in T if yv[i] and all(av[i+l] and fv[i+l] for l in ls)]
  assert common==oldrows[sid]['cross_domain']['three_domain_main_indices']
  c['three_domain_main_indices']=common;c['passed']=bool(cp and len(common)>=33);row['cross_domain']=c
  row['passed']=bool(input_pass and all(d['passed'] for d in row['domains'].values()) and c['passed']);records.append(row);ref.close()
 n=sum(x['passed'] for x in records)
 result={'status':'DEVELOPMENT_GATE_PASS' if n>=6 else 'DEVELOPMENT_GATE_FAILED','passed_sources':n,'fixed_denominator':8,'rows':records,'inherited_global_lag':lag,'eval_locked':True,'no_treatment_deltas':True,'no_forward_or_media':True,'protocol_sha256':seal['protocol_sha256'],'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 write(OUT/'diagnostics.json',result)
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<1024**3
 assert sum(f.stat().st_blocks*512 for f in OUT.rglob('*') if f.is_file())<1024**2
 print(json.dumps({'status':result['status'],'passed_sources':n,'diagnostics_sha256':sha(OUT/'diagnostics.json')}))
if __name__=='__main__':main()
