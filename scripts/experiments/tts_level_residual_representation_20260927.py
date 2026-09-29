"""Frozen CPU LEVEL residual midpoint/paired-permutation diagnostic."""
from pathlib import Path
import argparse,json,hashlib,types,sys,shutil,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'runs/tts_level_residual_representation_20260927';LEVEL=ROOT/'runs/tts_native_level_generation_cross_20260927';OLD=ROOT/'runs/tts_native_permutation_20260926'
ENGINE=ROOT/'runs/tts_acoustic_generation_cross_20260926/code_snapshot/tts_acoustic_generation_cross_20260926.py'
def load(path,name):
 m=types.ModuleType(name);m.__file__=str(ROOT/'scripts/experiments'/path.name);exec(compile(path.read_text(),str(path),'exec'),m.__dict__);return m
base=load(ENGINE,'frozen_gxe');perm=load(OLD/'code_at_freeze.py','frozen_perm')
read,write,sha=base.read,base.write,base.sha
FIELDS=['C','B','D','D_anchor','C_anchor','best_lag'];POLICIES=['guard20','valid','guard0'];GEOMS=['raw','unit'];BGS=['RAW','LEVEL'];LABELS=['N_base','T_base','N_M','T_M']
def resources():
 assert shutil.disk_usage(ROOT).free>=5*2**30,'5GiB disk reserve'
 size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert size<=250*2**20,'250MiB run budget'
 return {'free_bytes':shutil.disk_usage(ROOT).free,'run_bytes':size}
def frozen():
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
 for f,h in p['bindings'].items():assert sha(f)==h,f
 resources();return p

def inputs(row,arm,bg,geom):
 n=row['arms'][arm]['joint_L'];z=np.load(LEVEL/'features'/row['id']/arm/(('raw' if bg=='RAW' else 'LEVEL')+'.npz'));v,a=z['visual'][:n],z['audio'][:n]
 assert len(v)==len(a)==n and np.isfinite(v).all() and np.isfinite(a).all()
 return (base.unit(v),base.unit(a)) if geom=='unit' else (v,a)
def pi_for(sid,arm):return np.load(OLD/'permutations/static_cloud'/sid/(('N' if arm=='N' else 'C')+'.npz'))['whole']
def freeze():
 assert not (OUT/'protocol.json').exists();p=read(OUT/'protocol_design.json');assert sha(OUT/'protocol_design.json')==(OUT/'protocol_design.sha256').read_text().strip()
 for f,h in p['source_hashes'].items():assert sha(f)==h
 support=read(LEVEL/'support.json');seals={}
 for split in ['calibration','evaluation']:seals.update(read(LEVEL/('feature_seal_'+split+'.json')))
 ps=read(OLD/'permutation_seal.json');bindings=dict(p['source_hashes']);rawid=0.;control=[]
 for row in support:
  sid=row['id'];fp=LEVEL/'scores'/(sid+'.json');bindings[str(fp)]=sha(fp)
  for a in ['N','T']:
   cached={}
   for c in ['raw','identity','LEVEL']:
    fp=LEVEL/'features'/sid/a/(c+'.npz');jp=fp.with_suffix('.json')
    for f in [fp,jp]:assert sha(f)==seals[str(f)];bindings[str(f)]=seals[str(f)]
    meta=read(jp);assert meta['audio_frontend']==row['arms'][a][c];cached[c]=np.load(fp)
   n=row['arms'][a]['joint_L']
   for k in ['visual','audio']:rawid=max(rawid,float(abs(cached['raw'][k][:n]-cached['identity'][k][:n]).max()))
   assert all(row['arms'][a][c]['L']==n for c in ['raw','identity','LEVEL'])
   if sid in p['support']['primary']:
    fp=OLD/'permutations/static_cloud'/sid/(('N' if a=='N' else 'C')+'.npz');assert sha(fp)==ps[str(fp)];bindings[str(fp)]=sha(fp);pi=pi_for(sid,a);assert pi.shape==(256,n)
    j=np.arange(n-3);i=np.arange(20,n-20);reg=np.where(j<20,0,np.where(j<n-20,1,2));assert np.array_equal(reg[pi[:,j]],np.broadcast_to(reg,(256,len(j))))
    assert np.array_equal(np.sort(pi[:,i],axis=1),np.broadcast_to(i,(256,len(i))));control.append({'id':sid,'arm':a,'L':n,'moved_fraction':float((pi[:,j]!=j).mean()),'block_sizes':[int((reg==x).sum()) for x in range(3)]})
 assert rawid==0.
 files=[Path(__file__),ROOT/'scripts/experiments/check_tts_level_residual_representation_20260927.py',ENGINE,ROOT/'scripts/experiments/tts_native_midpoint.py',ROOT/'scripts/experiments/tts_native_boundary_audit.py']
 for f in files:bindings[str(f)]=sha(f)
 p.update({'status':'frozen_before_any_new_intervention_scores','approved_parent':True,'approved_epoch':time.time(),'rows':support,'bindings':bindings,'adaptive_alpha_constraint':'Each background estimates alpha separately. LEVEL x M is an interaction of the same adaptive matching rule, not an independent fixed-alpha factor. Report bidirectional alpha and CI for each background/geometry.','unit_constraint':'Normalize ORIGINAL vectors before defining own M/R/alpha. No normalization after transform.','primary_inference':'Closed guard20 only; valid/guard0 non-conserving supplements.','raw_identity_max':rawid})
 write(OUT/'protocol.json',p);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n');write(OUT/'permutation_controls.json',control)
 for f in files:
  dest=OUT/'code_snapshot'/f.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
 print('FROZEN',sha(OUT/'protocol.json'),len(bindings),flush=True)

def baseline():
 import torch
 torch.set_num_threads(2);p=frozen();rows=[];maxerr=0.;maxcurve=0.;cellschecked=0
 for row in p['rows']:
  sid=row['id'];old=read(LEVEL/'scores'/(sid+'.json'));cells={}
  for bg in BGS:
   for geom in GEOMS:
    for a in ['N','T']:
     v,au=inputs(row,a,bg,geom);m=base.matrix(v,au);key='/'.join([bg,geom,a]);cells[key]={pol:base.summarize(m,pol,3) for pol in POLICIES};orig=old['cells'][a][geom][('raw__raw' if bg=='RAW' else 'LEVEL__LEVEL')]['policies']
     for pol in POLICIES:
      new=cells[key][pol];expected=orig[pol];assert (new is None)==(expected is None)
      if new is None:continue
      maxerr=max(maxerr,max(abs(new[f]-expected[f]) for f in FIELDS));maxcurve=max(maxcurve,float(np.max(abs(np.array(new['curve'])-expected['curve']))));cellschecked+=1
  rec={'id':sid,'speaker':row['speaker'],'split':row['split'],'eligible':row['guard20_eligible'],'cells':cells};write(OUT/'baseline'/(sid+'.json'),rec);rows.append(rec)
 result={'passed':max(maxerr,maxcurve)<=1e-5,'metric_max':maxerr,'curve_max':maxcurve,'cells':cellschecked,'all100_all_baselines_before_intervention':True};write(OUT/'baseline_validation.json',result);assert result['passed']
 cal={bg:{a:float(np.median([r['cells'][f'{bg}/raw/{a}']['guard20']['best_lag'] for r in rows if r['split']=='calibration'])) for a in ['N','T']} for bg in BGS};write(OUT/'calibration_transfer.json',{'k':3,'medians':cal,'sufficient':all(abs(v-3)<=1 for d in cal.values() for v in d.values()) and all(abs(d['N']-d['T'])<=1 for d in cal.values()),'refit':False})
 write(OUT/'score_lock.json',{'time':time.time(),'baseline_sha256':sha(OUT/'baseline_validation.json'),'protocol_sha256':sha(OUT/'protocol.json'),'status':'all_baseline_gates_pass_before_new_intervention'})
 print('BASELINE',result,cal,flush=True)

def curves_from_gram(d,pi):
 n=d.shape[0];b=len(pi);iv=np.broadcast_to(np.arange(n),(b,n)).copy();ia=iv.copy();iv[:,:n-3]=pi[:,:n-3];ia[:,3:]=pi[:,:n-3]+3
 out={p:np.empty((b,31)) for p in POLICIES};ix=np.arange(n)
 for col,s in enumerate(range(-15,16)):
  q=ix+s;valid=(q>=0)&(q<n);aj=np.full((b,n),n,dtype=int);aj[:,valid]=ia[:,q[valid]];vals=d[iv,aj]
  out['guard20'][:,col]=vals[:,20:n-20].mean(1);out['guard0'][:,col]=vals.mean(1);out['valid'][:,col]=vals[:,valid].mean(1)
 return out

def score():
 import torch
 torch.set_num_threads(2);p=frozen();gate=read(OUT/'baseline_validation.json');lock=read(OUT/'score_lock.json');assert gate['passed'] and sha(OUT/'baseline_validation.json')==lock['baseline_sha256']
 controls=[];coefs=[]
 for row in p['rows']:
  sid=row['id']
  if sid not in p['support']['primary']:continue
  for bg in BGS:
   for geom in GEOMS:
    cc={a:perm.coordinates(*inputs(row,a,bg,geom),3) for a in ['N','T']};ratio=cc['N']['sigma']/cc['T']['sigma'];coefs.append({'id':sid,'speaker':row['speaker'],'background':bg,'geometry':geom,'T_to_N':ratio,'N_to_T':1/ratio,'sigma_N':cc['N']['sigma'],'sigma_T':cc['T']['sigma'],'rho_N':cc['N']['rho'],'rho_T':cc['T']['rho']});arrays={}
    for label,(a,alpha) in {'N_base':('N',1.),'T_base':('T',1.),'N_M':('N',1/ratio),'T_M':('T',ratio)}.items():
     c=cc[a];v,au=perm.transform(c,alpha,1.);n=len(v);j=c['t'];ix=c['i'];anchor64=float(abs((au[j+3]-v[j])-(c['a'][j+3]-c['v'][j])).max());anchor32=float(abs((au.astype(np.float32)[j+3]-v.astype(np.float32)[j])-(c['a'].astype(np.float32)[j+3]-c['v'].astype(np.float32)[j])).max())
     backm=c['mu']+(v[j]+au[j+3]-2*c['mu'])/(2*alpha);rr=(au[j+3]-v[j])/2;inv=float(max(abs(backm-rr-c['v'][j]).max(),abs(backm+rr-c['a'][j+3]).max()));assert anchor64<1e-12 and inv<1e-12
     vf=np.asarray(v,dtype=np.float32);af=np.asarray(au,dtype=np.float32);d=perm.gram(vf,np.vstack([af,np.zeros((1,af.shape[1]),np.float32)]));ident=curves_from_gram(d,np.arange(n)[None,:]);pi=pi_for(sid,a);curves=curves_from_gram(d,pi)
     diag=d[ix,ix+3];assert np.array_equal(np.sort(d[pi[:,ix],pi[:,ix]+3],axis=1),np.broadcast_to(np.sort(diag),(256,len(ix))))
     energies=((c['M'][ix]-c['mu'])**2).sum(1);renergy=(c['R'][ix]**2).sum(1);order=pi[:,ix]-20;energyerr=float(max(abs(energies[order].mean(1)-energies.mean()).max(),abs(renergy[order].mean(1)-renergy.mean()).max()))
     for policy in POLICIES:
      arrays[label+'/'+policy+'/identity']=perm.curve_metrics(ident[policy],3);arrays[label+'/'+policy+'/whole']=perm.curve_metrics(curves[policy],3)
     arrays[label+'/guard20/identity_curve']=ident['guard20'];arrays[label+'/guard20/whole_curves']=curves['guard20']
     old=read(OUT/'baseline'/(sid+'.json'))['cells'][f'{bg}/{geom}/{a}'];err=max(abs(arrays[label+'/'+pol+'/identity'][0,z]-old[pol][f]) for pol in POLICIES for z,f in enumerate(FIELDS)) if label.endswith('base') else 0.;assert err<1e-5
     rng=np.random.Generator(np.random.PCG64(20260926));probes=rng.integers(n,size=(512,2));direct=np.linalg.norm(vf[probes[:,0]].astype(float)-af[probes[:,1]].astype(float)+1e-6,axis=1);derr=float(abs(d[probes[:,0],probes[:,1]]-direct).max());assert derr<1e-5
     # A fixed common translation of M should leave every real-real distance invariant.
     shift=np.full(v.shape[1],.125);translated=perm.gram(v+shift,au+shift);shift_err=float(abs(translated-d[:,:n]).max());assert shift_err<1e-5
     controls.append({'id':sid,'background':bg,'geometry':geom,'label':label,'anchor64':anchor64,'anchor32':anchor32,'inverse':inv,'energy':energyerr,'baseline':err,'direct_probe':derr,'constant_M_shift':shift_err,'anchor_multiset_exact':True})
    dest=OUT/'scores'/bg/geom/(sid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**arrays)
  resources();print('scored',sid,flush=True)
 write(OUT/'coefficients.json',coefs);write(OUT/'controls.json',controls);print('SCORED',resources(),flush=True)

def point_contrasts(z,policy,repeats=256):
 def val(label,op):return z[label+'/'+policy+'/'+op][:repeats].mean(0)
 n,t,nm,tm=[val(label,'identity') for label in LABELS];np_,tp,nmp,tmp=[val(label,'whole') for label in LABELS]
 out={}
 for direction,quad in [('T_to_N',[(n,t),(n,tm),(np_,tp),(np_,tmp)]),('N_to_T',[(n,t),(nm,t),(np_,tp),(nmp,tp)])]:
  cells={}
  for name,(nv,tv) in zip(['native','M','whole','M_whole'],quad):
   cells[name]={'N':nv,'T':tv,'gap':tv-nv}
   for arm,vals in cells[name].items():out[f'{direction}/{name}/{arm}']=vals
  for arm in ['N','T','gap']:
   q={name:values[arm] for name,values in cells.items()}
   for effect,value in {'M_effect':q['M']-q['native'],'whole_effect':q['whole']-q['native'],'M_after_whole':q['M_whole']-q['whole'],'whole_after_M':q['M_whole']-q['M'],'interaction':q['M_whole']-q['M']-q['whole']+q['native']}.items():out[f'{direction}/{effect}/{arm}']=value
 return out

def analyze():
 p=frozen();rows=[r for r in p['rows'] if r['id'] in p['support']['primary']];speakers=[r['speaker'] for r in rows];result={};mc={};point={}
 for geom in GEOMS:
  for policy in POLICIES:
   by={bg:[point_contrasts(np.load(OUT/'scores'/bg/geom/(r['id']+'.npz')),policy) for r in rows] for bg in BGS};point[geom+'/'+policy]=by
   for bg in BGS+['LEVEL_minus_RAW']:
    for key in by['RAW'][0]:
     vals=np.stack([x[key] for x in by[bg]]) if bg in BGS else np.stack([x[key]-y[key] for x,y in zip(by['LEVEL'],by['RAW'])]);result['/'.join([geom,policy,bg,key])]={f:base.bootstrap(vals[:,j],speakers) for j,f in enumerate(FIELDS)}
   for count in [64,128]:
    small={bg:[point_contrasts(np.load(OUT/'scores'/bg/geom/(r['id']+'.npz')),policy,count) for r in rows] for bg in BGS}
    for bg in BGS+['LEVEL_minus_RAW']:
     vals=[]
     for key in by['RAW'][0]:
      delta=np.stack([x[key]-y[key] for x,y in zip(small[bg],by[bg])]) if bg in BGS else np.stack([(u[key]-v[key])-(x[key]-y[key]) for u,v,x,y in zip(small['LEVEL'],small['RAW'],by['LEVEL'],by['RAW'])]);means=np.array([delta[np.array(speakers)==s].mean(0) for s in sorted(set(speakers))]).mean(0);vals.extend(abs(means[:5]).tolist())
     mc[f'{geom}/{policy}/{bg}/{count}_minus_256_max_C_B_D_anchors']=max(vals)
 baseline={}
 for subset in ['common71','all74_appendix']:
  use=[r for r in p['rows'] if r['split']=='evaluation' and (subset!='common71' or r['guard20_eligible'])];ss=[r['speaker'] for r in use];rr=[read(OUT/'baseline'/(r['id']+'.json')) for r in use]
  for geom in GEOMS:
   for policy in (POLICIES if subset=='common71' else ['valid','guard0']):
    for bg in BGS:
     vals={a:np.array([[r['cells'][f'{bg}/{geom}/{a}'][policy][f] for f in FIELDS] for r in rr]) for a in ['N','T']};vals['gap']=vals['T']-vals['N']
     for a,v in vals.items():baseline[f'{subset}/{geom}/{policy}/{bg}/{a}']={f:base.bootstrap(v[:,j],ss) for j,f in enumerate(FIELDS)}
 coefs=read(OUT/'coefficients.json');cs={}
 for geom in GEOMS:
  for bg in BGS:
   use=[r for r in coefs if r['geometry']==geom and r['background']==bg];cs[f'{geom}/{bg}']={key:base.bootstrap([r[key] for r in use],[r['speaker'] for r in use]) for key in ['T_to_N','N_to_T','sigma_N','sigma_T','rho_N','rho_T']}
 write(OUT/'summary.json',result);write(OUT/'baseline_summary.json',baseline);write(OUT/'coefficient_summary.json',cs);write(OUT/'mc_precision.json',mc)
 print('ANALYZED',len(result),resources(),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','baseline','score','analyze']);globals()[ap.parse_args().stage]()
