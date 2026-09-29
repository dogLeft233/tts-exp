"""Independent complex-LS recomputation of the separately frozen r development gates."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import grid_geometry_result_review_20260927 as independent
ROOT=Path(__file__).resolve().parents[2]
NEW=ROOT/'runs/grid_shape_correlation_development_20260927'
OLD=ROOT/'runs/grid_geometry_calibration_20260927'
OUT=ROOT/'runs/grid_geometry_protocol_audit_20260927/r_development_review'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 p=read(NEW/'protocol.json');prod=read(NEW/'diagnostics.json');assert sha(NEW/'protocol.json')==read(OUT/'static_review.json')['protocol_sha256']==prod['protocol_sha256'];assert sha(independent.__file__)=='849fbb52e8b38206f141ffeba1a2f4ffe873bc05d58b85fb534f7c6d0ff870f2'
 seal=read(OLD/'scientific_feature_seal.json');assert sha(OLD/'scientific_feature_seal.json')==p['bindings']['runs/grid_geometry_calibration_20260927/scientific_feature_seal.json'];assert all(sha(ROOT/k)==h for k,h in seal.items())
 T=np.arange(14,55);L=np.arange(-5,6);lag=-3;rows=[];errors=[];scalar_checks=0;load_count=0;by={r['id']:r for r in prod['rows']}
 def check(a,b):
  nonlocal scalar_checks
  scalar_checks+=1
  if a is None:assert b is None;return
  assert b is not None;error=abs(a-b);assert error<1e-7,error;errors.append(error)
 for sid in p['ids']:
  with np.load(OLD/'landmarks'/sid/'REAL/BASE.npz') as z:ref={k:z[k] for k in z.files}
  cache={}
  def get(d,v='BASE'):
   nonlocal load_count
   key=(d,v)
   if key not in cache:
    f=OLD/'landmarks'/sid/d/(v+'.npz')
    with np.load(f) as z:data={k:z[k] for k in z.files}
    assert np.array_equal(data['pts'],np.arange(75)/25);cache[key]=independent.geometry(data,ref);load_count+=1
   return cache[key]
  saved=by[sid];inputpass=bool(ref['valid'][0]) and all(independent.qc(get('REAL'),True).values());assert inputpass==saved['input_pass'];row={'id':sid,'input_pass':inputpass,'domains':{}}
  for d in p['domains']:
   data={v:get(d,v) for v in p['views']};t=np.array([i for i in T if all(all(g['valid'][i+k] for k in L) for g in data.values())],dtype=int);s=saved['domains'][d];assert t.tolist()==s['indices'];qpass=all(independent.qc(data['BASE'],d=='REAL').values());assert qpass==s['base_QC_pass'];flags=[len(t)>=33,qpass];rec={'indices':t.tolist(),'base_QC_pass':qpass,'controls':{}}
   if len(t)>=33:
    y=data['BASE']['ap'];scores={v:independent.metric(y[t],z['ap'][t]) for v,z in data.items()};sd=float(np.std(y[t],ddof=0));noise=max(scores[v]['centered_rmse'] for v in p['spatial']);motion=bool(sd>=.005 and sd>=2*noise);ratio=sd/noise if noise>0 else None;check(sd,s['native_sd']);check(noise,s['max_spatial_centered_rmse']);check(ratio,s['motion_noise_ratio']);assert motion==s['motion_pass'];flags.append(motion);rec.update(native_sd=sd,max_spatial_centered_rmse=noise,motion_noise_ratio=ratio,motion_pass=motion)
    for v,m in scores.items():
     c=s['controls'][v]
     for key in ['r','centered_rmse','reference_sd','candidate_sd']:check(m[key],c[key])
     ok=True;extra={}
     if v=='REPEAT':ok=m['r'] is not None and m['r']>=.99
     elif v in p['spatial']:ok=m['r'] is not None and m['r']>=.95
     elif v in ['REVERSE','WARP_0.8','WARP_1.2']:
      delta=scores['REPEAT']['r']-m['r'] if scores['REPEAT']['r'] is not None and m['r'] is not None else None;check(delta,c['repeat_minus_control_r']);ok=delta is not None and delta>=.05;extra['r_drop']=delta
     elif v.startswith('SHIFT_'):
      expected=int(v.split('_')[1]);curve=[independent.metric(y[t],data[v]['ap'][t+k])['r'] for k in L];found=independent.choose(curve,L);assert found==c['recovered_lag'];ok=found is not None and found*expected>0 and abs(found-expected)<=1
      for a,b in zip(curve,c['r_by_lag']):check(a,b)
      extra.update(recovered=found,expected=expected)
     elif v=='FROZEN':assert m['candidate_sd']<=1e-6 and m['r'] is None and c['zero_dynamic_NA_check'] and not c['scientific_gate']
     assert bool(ok)==c['passed'];rec['controls'][v]={'r':m['r'],'centered_rmse':m['centered_rmse'],'passed':bool(ok),**extra}
     if v not in ['BASE','FROZEN']:flags.append(ok)
   want=all(flags);assert bool(want)==s['passed'];rec['passed']=bool(want);row['domains'][d]=rec
  y=get('REAL');raw={v:get('RAW',v) for v in ['BASE','SHIFT_-4','SHIFT_-2','SHIFT_2','SHIFT_4']};search=L+lag;t=np.array([i for i in T if y['valid'][i] and all(all(g['valid'][i+k] for k in search) for g in raw.values())],dtype=int);s=saved['cross_domain'];assert t.tolist()==s['indices'];cp=False;cross={'indices':t.tolist(),'shift':{}}
  if len(t)>=33:
   r=independent.metric(y['ap'][t],raw['BASE']['ap'][t+lag])['r'];r0=independent.metric(y['ap'][t],raw['BASE']['ap'][t])['r'];check(r,s['RAW_r']);check(r0,s['RAW_lag0_r']);b=independent.choose([independent.metric(y['ap'][t],raw['BASE']['ap'][t+k])['r'] for k in search],search);assert b==s['base_optimum'];shiftflags=[];cross.update(RAW_r=r,RAW_lag0_r=r0)
   for v in list(raw)[1:]:
    expected=int(v.split('_')[1]);curve=[independent.metric(y['ap'][t],raw[v]['ap'][t+k])['r'] for k in search];found=independent.choose(curve,search);ok=found is not None and b is not None and (found-b)*expected>0 and abs(found-b-expected)<=1;assert ok==s['shift'][v]['passed'] and found==s['shift'][v]['recovered_lag'];shiftflags.append(ok)
    for aa,bb in zip(curve,s['shift'][v]['r_by_lag']):check(aa,bb)
    cross['shift'][v]={'passed':ok,'relative':found-b if found is not None and b is not None else None}
   cp=r is not None and r>=.30 and all(shiftflags)
  av=get('RAW')['valid'];fv=get('FIXED')['valid'];common=[int(i) for i in T if y['valid'][i] and all(av[i+k] and fv[i+k] for k in L)];assert common==s['three_domain_main_indices'];cp=cp and len(common)>=33;assert bool(cp)==s['passed'];cross.update(passed=bool(cp),main_indices=common);row['cross_domain']=cross;want=inputpass and all(d['passed'] for d in row['domains'].values()) and cp;assert bool(want)==saved['passed'];row['passed']=bool(want);rows.append(row)
 count=sum(r['passed'] for r in rows);status='DEVELOPMENT_GATE_PASS' if count>=6 else 'DEVELOPMENT_GATE_FAILED';assert count==prod['passed_sources'] and status==prod['status'] and prod['fixed_denominator']==8 and prod['eval_locked'] and prod['no_treatment_deltas'];assert load_count==432
 result={'status':'PASS','time':time.time(),'development_status':status,'passed_sources':count,'fixed_denominator':8,'rows':rows,'independent_array_loads':load_count,'source_files_verified':len(seal),'numeric_scalar_checks':scalar_checks,'max_numeric_error':max(errors,default=0.),'protocol_sha256':sha(NEW/'protocol.json'),'source_diagnostics_sha256':sha(NEW/'diagnostics.json'),'code_sha256':sha(__file__),'complex_LS_helper_sha256':sha(independent.__file__),'eval_locked':True,'no_treatment_deltas':True,'model_GPU_calls':0}
 path=OUT/'independent_gate.json';assert not path.exists();path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
