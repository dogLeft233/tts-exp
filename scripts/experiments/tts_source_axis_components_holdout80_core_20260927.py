"""Frozen v4 source-axis mechanism follow-up; no refit or output-driven selection."""
from pathlib import Path
import argparse,gzip,importlib.util,json,math,shutil,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments import tts_fixed_generator_shift_holdout80_20260927 as parent
from scripts.experiments.tts_fixed_generator_phone_stream_20260927 import sha,ah
OUT=ROOT/'runs/tts_source_axis_components_holdout80_20260927'
PARENT=parent.OUT;OLD=parent.OLD;FIT=parent.FIT;PY=parent.PY
PREP=ROOT/'runs/tts_source_axis_components_holdout80_preparation_20260927'
AXIS=ROOT/'runs/tts_cal_source_level_axis_geometry_20260927'
TMP=Path('/dev/shm/tts_source_axis_components_holdout80_20260927')
CAP=176*2**20;AUDIT=16*2**20;FLOOR=int(3.80*2**30)
CONDITIONS=CONDS=['parallel','orthogonal','masked_actual_gain','strict_orth']
read=parent.read;allocated=parent.allocated;runtime=parent.runtime;starts=parent.starts

def module(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
ops=module('axis_ops',PREP/'ops.py');strict=module('axis_strict',PREP/'strict_solver.py')

def limits(extra=0):
 used=allocated(OUT);free=shutil.disk_usage(OUT if OUT.exists() else ROOT).free
 assert used+extra<=CAP and free-max(0,CAP-used)-AUDIT>=FLOOR,('resource',used,extra,free)
 tmp=allocated(TMP);assert tmp<=96*2**20
 av=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'));assert av>=4*2**30
 return {'allocated':used,'free':free,'remaining_commitment':max(0,CAP-used)+AUDIT,'GPU_tmp':tmp,'MemAvailable':av}

def write(p,x):
 p=Path(p);data=json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()+b'\n'
 if p.suffix=='.gz':data=gzip.compress(data,mtime=0)
 limits(math.ceil(len(data)/4096)*4096);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(data)
 limits()

def stats(a):
 x=np.asarray(a,dtype=np.float64).reshape(-1);assert np.isfinite(x).all()
 return {'n':len(x),'mean':float(np.mean(x)) if len(x) else None,'median':float(np.median(x)) if len(x) else None,'min':float(np.min(x)) if len(x) else None,'max':float(np.max(x)) if len(x) else None}

def dose(native,replacement,mask,u,pre=None):
 mask=np.asarray(mask,bool);assert np.array_equal(native[~mask],replacement[~mask])
 z=native[mask].astype(np.float64);y=replacement[mask].astype(np.float64);d=y-z
 uu=float(np.dot(u,u));assert uu>1e-24
 dot=d@u;norm=np.linalg.norm(d,axis=1);ok=norm>0
 result={'speech_frames':int(mask.sum()),'identity_frames':int((~mask).sum()),'input_z_raw_sha256':ah(native),'replacement_z_raw_sha256':ah(replacement),'inactive_exact':True,'dot_u':stats(dot),'axis_coefficient':stats(dot/uu),'parallel_norm':stats(abs(dot)/np.sqrt(uu)),'displacement_norm':stats(norm),'cosine':stats(dot[ok]/norm[ok]/np.sqrt(uu)),'cosine_undefined_count':int((~ok).sum()),'input_RMS':float(np.sqrt(np.mean(z*z))),'output_RMS':float(np.sqrt(np.mean(y*y))),'descriptive_only':True}
 if pre is not None:
  pre=np.asarray(pre,np.float64);assert pre.shape==z.shape
  result.update(pre_negative_fraction=float(np.mean(pre<0)),projection_change_l2=stats(np.linalg.norm(y-pre,axis=1)))
 return result

def strict_summary(ds):
 keys=['alpha','bracket_expansions','bisections','residual64','tolerance64','residual32','residual32_bound','rounding_max','stationarity_max','objective_squared64','projection_change_l2_64']
 r={k:stats([x[k] for x in ds]) for k in keys}
 ratios=[abs(x['residual32'])/x['residual32_bound'] if x['residual32_bound'] else 0. for x in ds]
 r.update(residual32_bound_ratio=stats(ratios),all_within_bounds=all(abs(x['residual32'])<=x['residual32_bound'] for x in ds),float32_exact_count=sum(x['float32_constraint_exact'] for x in ds),cases={k:sum(x['case']==k for x in ds) for k in sorted({x['case'] for x in ds})})
 return r

def freeze():
 assert not (OUT/'protocol.json').exists()
 assert read(PREP/'protocol.json')['version']==4
 final=read(PARENT/'final.json');assert str(final['status']).lower() in ['concluded','complete','pass']
 # Parent scientific PASS and independent closure are explicit input prerequisites.
 review=ROOT/'runs/tts_fixed_generator_shift_holdout80_independent_audit_20260927'
 closure=read(OUT/'parent_closure_binding.json')
 assert closure['status']=='PASS' and sha(closure['receipt'])==closure['receipt_sha256']
 assert read(closure['receipt'])['status']=='PASS'
 summary=read(PARENT/'summary.json.gz')
 for m in ['C','C_anchor']:
  for q in ['plus_minus_baseline','plus_minus_minus']:
   assert summary['raw/guard20/'+m+'/global/'+q]['ci99'][0]>0
 p=read(PARENT/'protocol.json');deps=dict(p['dependencies']);rows=read(PARENT/'input_rows.json')
 assert len(rows)==80 and len({r['speaker'] for r in rows})==40
 files=[PARENT/n for n in ['protocol.json','seal.json','input_rows.json','input_seal.json','input_reviewer_pass.json','baseline_gate.json','baseline_validation.json','feature_seal.json','score_lock.json','analysis.json','summary.json.gz','report.md','final.json','resource_closure.json']]
 files += [review/'result_receipt.json',Path(closure['receipt']),OUT/'parent_closure_binding.json']
 files += [PREP/n for n in ['protocol.json','design.md','ops.py','strict_solver.py','audit_v4/result_receipt.json','preparation_receipt.json']]
 files += [AXIS/n for n in ['vectors.npz','result.json','protocol.json','final.json','audit/result_receipt.json']]
 for r in rows:
  sid=r['id'];files += [PARENT/'z'/(sid+'.npy'),PARENT/'baseline'/(sid+'.json'),PARENT/'metadata'/sid/'N/global_plus.json',PARENT/'visual'/sid/'N/global_plus.npy',PARENT/'scores'/(sid+'.json.gz'),PARENT/'baseline_scores'/(sid+'.json.gz')]
  for k in ['features','frontend','waveform','metadata']:deps[r['arms']['N'][k+'_path']]=r['arms']['N'][k+'_sha256']
  deps[r['arms']['N']['TextGrid']]=r['arms']['N']['TextGrid_sha256']
 for suffix in ['core','gpu','scores']:
  files.append(ROOT/('scripts/experiments/tts_source_axis_components_holdout80_'+suffix+'_20260927.py'))
 files += [OUT/'protocol.md',OUT/'gpu_supervisor.py']
 for f in files:deps[str(f.resolve())]=sha(f)
 for f,h in deps.items():assert sha(f)==h,f
 c=dict(p['gpu_config']);c.update(own_cap_bytes=CAP,other_reserved_bytes=AUDIT,floor_bytes=FLOOR)
 with np.load(AXIS/'vectors.npz') as v:
  assert v['u_level'].dtype==np.float64 and np.linalg.norm(v['u_level'])>1e-12
  assert np.array_equal(np.subtract(v['source_mu32'][1],v['source_mu32'][0],dtype=np.float32),v['actual_delta32'])
  rounding=ops.input_rounding_descriptions(v['actual_delta32'],v['parallel'],v['orthogonal'])
 write(OUT/'input_rows.json',rows)
 deps[str(OUT/'input_rows.json')]=sha(OUT/'input_rows.json')
 proto={'status':'FROZEN_BEFORE_NEW_FORWARD_OR_EFFECT_SCORE','created_epoch':time.time(),'preparation_version':4,'preparation_protocol_sha256':sha(PREP/'protocol.json'),'dependencies':deps,'gpu_config':c,'conditions':CONDITIONS,'primary':'strict_orth-baseline and strict_orth-parallel each rawguard20 C/C_anchor four99CI lower>0','statistics':p['statistics'],'scoring':p['scoring'],'resources':{'main':CAP,'audit':AUDIT,'floor':FLOOR,'GPU_cumulative_seconds_including_live':3600,'GPU_tmp':96*2**20,'minimum_MemAvailable':4*2**30,'V_allocated_bound':146423808,'gain_z_allocated_bound':19034112,'calV_allocated_bound':2359296},'gain_beta64':read(AXIS/'result.json')['beta_linear_secant_dB'],'input_rounding_descriptions':rounding,'BM_title':'80句来源方向电平轴分量与严格正交生成干预 2026-09-27','same80_followup_not_independent_confirmation':True}
 write(OUT/'protocol.json',proto);write(OUT/'seal.json',{'protocol_sha256':sha(OUT/'protocol.json'),'created_epoch':time.time(),'no_new_forward_or_scores':True})
 write(OUT/'input_seal.json',{'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'files':{str(OUT/'input_rows.json'):sha(OUT/'input_rows.json')},'parent_input_seal_sha256':sha(PARENT/'input_seal.json'),'all80_and_original_mask_exact':True})
 for f in files:
  if f.suffix=='.py' and (f.parent==ROOT/'scripts/experiments' or f==OUT/'gpu_supervisor.py'):
   dest=OUT/'code_snapshot'/f.name;data=f.read_bytes();limits(math.ceil(len(data)/4096)*4096);dest.parent.mkdir(exist_ok=True)
   with dest.open('xb') as out:out.write(data)
 print('FROZEN',sha(OUT/'protocol.json'),flush=True)

def locked():
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==read(OUT/'seal.json')['protocol_sha256']
 for f,h in p['dependencies'].items():assert sha(f)==h,f
 rv=read(OUT/'reviewer_pass.json');assert rv['status']=='PASS' and rv['protocol_sha256']==sha(OUT/'protocol.json') and sha(rv['receipt'])==rv['receipt_sha256']
 s=read(OUT/'input_seal.json');assert s['protocol_sha256']==sha(OUT/'protocol.json')
 for f,h in s['files'].items():assert sha(f)==h
 limits();return p,read(OUT/'input_rows.json')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['freeze']);globals()[a.parse_args().stage]()
