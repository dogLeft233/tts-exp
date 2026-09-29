"""Pure synthetic clock/sign/closed-mask checks; never imports production model code."""
from pathlib import Path
import json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/grid_geometry_protocol_audit_20260927'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x,y):
 x=x-x.mean();y=y-y.mean();return float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))
def choose(curve):
 m=max(curve.values());ties=[k for k,v in curve.items() if abs(v-m)<=1e-12];return min(ties,key=lambda k:(abs(k),k))
if __name__=='__main__':
 OUT.mkdir(exist_ok=True);t=np.arange(14,55);n=75;rng=np.random.default_rng(20260927);real=rng.normal(size=n);ix=np.arange(n);records=[]
 for g in range(-5,6):
  raw=real[np.clip(ix-g,0,n-1)];curve={k:stats(raw[t+k],real[t]) for k in range(-5,6)};assert choose(curve)==g
  for d in [-4,-2,2,4]:
   shifted=raw[np.clip(ix-d,0,n-1)];cc={relative:stats(shifted[t+g+relative],real[t]) for relative in range(-5,6)};found=choose(cc);assert found==d
   queried=np.concatenate([t+g+k for k in range(-5,6)]);source=queried-d;assert source.min()>=0 and source.max()<=68
   records.append({'global':g,'shift_d':d,'recovered_relative':found,'queried_min':int(queried.min()),'queried_max':int(queried.max()),'source_min':int(source.min()),'source_max':int(source.max())})
 # Closed validity mask must include the actual shifted-domain index for every candidate.
 valid=np.ones(n,dtype=bool);valid[40]=False;common=np.ones(len(t),dtype=bool)
 for g in range(-5,6):
  for rel in range(-5,6):common&=valid[t+g+rel]
 expected=~np.isin(t,np.arange(30,51));assert np.array_equal(common,expected)
 fixed_global_common=np.logical_and.reduce([valid[t+k] for k in range(-5,6)])
 assert np.array_equal(fixed_global_common,~np.isin(t,np.arange(35,46)))
 # Float WAV MFCC is not evaluated here; only canonical time grid arithmetic.
 starts=np.array([int(i*80/25) for i in range(n)]);minimum_mel_columns=int(starts[68]+16);assert minimum_mel_columns==233
 for samples in [47360,47647,48000,48640]:
  columns=1+samples//200;first_clamp=next((i for i,s in enumerate(starts) if s+16>columns),None);assert all(s+16<=columns for s in starts[:69])
 # Tie ordering is explicit and symmetric closest-to-zero, then negative.
 assert choose({-2:1.,2:1.,-1:.5,0:.2})==-2 and choose({-1:1.,1:1.,0:1.})==0
 y=real[t]-real[t].mean();energy_cases=[]
 for scale in [0.,.5,1.,2.,3.]:
  x=scale*y+7.;xc=x-x.mean();E=1-np.mean((xc-y)**2)/np.mean(y*y)
  assert abs(E-(2*scale-scale*scale))<1e-12
  energy_cases.append({'scale':scale,'E':float(E),'expected':2*scale-scale*scale,'r_defined':scale>0})
 data={'status':'PASS','time':time.time(),'scope':'pure synthetic protocol math; no model/data/media reads','interior_count':len(t),'minimum_valid':33,'global_and_control_sign_cases':len(records),'cases':records,'centered_E_cases':energy_cases,'closed_mask_with_one_missing40':{'fixed_global0_relative_search5_remaining':int(fixed_global_common.sum()),'fixed_global_removed_times':t[~fixed_global_common].tolist(),'unnecessarily_intersect_all_global_and_relative_remaining':int(common.sum()),'note':'Actual closed search at one frozen global can remove11 base positions; do not add an unnecessary intersection across every global in the later relative search. Never interpolate/relax33.'},'minimum_mel_columns_for_no_clamp_0_to68':minimum_mel_columns,'max_original_source_index':max(r['source_max'] for r in records),'min_original_source_index':min(r['source_min'] for r in records),'code_sha256':sha(__file__)}
 (OUT/'synthetic_checks.json').write_text(json.dumps(data,indent=2)+'\n');print({k:v for k,v in data.items() if k!='cases'})
