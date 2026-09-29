"""CPU paired vs independent block-permutation null, fixed 256 indices."""
from pathlib import Path
import sys,types,argparse,time,shutil,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));OUT=ROOT/'runs/tts_level_independent_permutation_null_20260927';PARENT=ROOT/'runs/tts_level_residual_representation_20260927'
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
def paths(row,arm):return OUT/'rho'/row['id']/(arm+'.npz')
def parent_curves(sid,bg,geom,a):
 z=np.load(PARENT/'scores'/bg/geom/(sid+'.npz'));return z[a+'_base/guard20/identity_curve'],z[a+'_base/guard20/whole_curves']
def freeze():
 assert not (OUT/'protocol.json').exists();design=read(OUT/'protocol_design.json');assert sha(OUT/'protocol_design.json')==(OUT/'protocol_design.sha256').read_text().strip();ad=read(OUT/'protocol_design_addendum.json');assert ad['base_design_sha256']==sha(OUT/'protocol_design.json');pp=m.frozen();rows=[r for r in pp['rows'] if r['id'] in design['support']];bindings=dict(design['parent_hashes']);descriptions=[]
 for f,h in bindings.items():assert sha(f)==h
 for r in rows:
  sid=r['id']
  for a in ['N','T']:
   n=r['arms'][a]['joint_L'];j=np.arange(n-3);i=np.arange(20,n-20);blocks=[j[j<20],j[(j>=20)&(j<n-20)],j[j>=n-20]];rho=np.tile(np.arange(n,dtype=np.int32),(256,1));pi=m.pi_for(sid,a)
   for b in range(256):
    seed=int.from_bytes(hashlib.sha256(f'20260927|{sid}|{a}|3|whole|{b}'.encode()).digest()[:16],'big');rng=np.random.Generator(np.random.PCG64(seed))
    for block in blocks:rho[b,block]=rng.permutation(block)
   for block in blocks:assert np.array_equal(np.sort(rho[:,block],axis=1),np.tile(block,(256,1)))
   dest=paths(r,a);dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,rho=rho);bindings[str(dest)]=sha(dest);old=m.OLD/'permutations/static_cloud'/sid/(('N' if a=='N' else 'C')+'.npz');bindings[str(old)]=sha(old)
   frac=np.mean(rho[:,i]==pi[:,i],axis=1);descriptions.append({'id':sid,'speaker':r['speaker'],'arm':a,'L':n,'domain_lengths':[len(x) for x in blocks],'I':len(i),'accidental_fractions':frac.tolist(),'observed_mean':float(frac.mean()),'expected':1/len(i),'shared_backgrounds':BGS,'shared_geometries':GEOMS,'no_rejection':True})
   for c in ['raw','LEVEL']:
    for ext in ['.npz','.json']:
     f=m.LEVEL/'features'/sid/a/(c+ext);assert sha(f)==pp['bindings'][str(f)];bindings[str(f)]=sha(f)
  for bg in BGS:
   for geom in GEOMS:
    f=PARENT/'scores'/bg/geom/(sid+'.npz');assert sha(f)==read(PARENT/'artifact_hashes.json')[str(f)];bindings[str(f)]=sha(f)
  f=PARENT/'baseline'/(sid+'.json');bindings[str(f)]=sha(f)
 for f in [code,m.ENGINE,m.OLD/'code_at_freeze.py',ROOT/'scripts/experiments/tts_native_midpoint.py',ROOT/'scripts/experiments/tts_native_boundary_audit.py',Path(__file__),ROOT/'scripts/experiments/check_tts_level_independent_permutation_null_20260927.py']:bindings[str(f)]=sha(f)
 write(OUT/'accidental_correspondence.json',descriptions);bindings[str(OUT/'accidental_correspondence.json')]=sha(OUT/'accidental_correspondence.json');p={**design,'status':'frozen_all_rho_before_new_distance','parent_approved':True,'created_epoch_fullseal':time.time(),'addendum':ad,'rows':rows,'bindings':bindings,'endpoints':FIELDS};write(OUT/'protocol.json',p);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
 for f in [Path(__file__),ROOT/'scripts/experiments/check_tts_level_independent_permutation_null_20260927.py']:
  d=OUT/'code_snapshot'/f.name;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,d)
 print('FROZEN',sha(OUT/'protocol.json'),len(bindings),resource(),flush=True)

def aligned(row,a,bg,geom):
 v,au=m.inputs(row,a,bg,geom);return v[:-3],au[3:]
def curve(d,pv,pa,n):
 ix=np.arange(20,n-20);q=ix[:,None]+np.arange(-15,16)-3
 return np.array([d[x[ix,None],y[q]].mean(0) for x,y in zip(pv,pa)])
def baseline():
 p=frozen();err=0.;nativeerr=0.;cells=0
 for row in p['rows']:
  sid=row['id'];old=m.read(PARENT/'baseline'/(sid+'.json'))
  for bg in BGS:
   for geom in GEOMS:
    for a in ['N','T']:
     v,au=aligned(row,a,bg,geom);n=len(v)+3;d=m.perm.gram(v,au);pi=m.pi_for(sid,a);ident=np.arange(n)[None,:];ncur=curve(d,ident,ident,n);pcur=curve(d,pi,pi,n);pn,pp=parent_curves(sid,bg,geom,a);err=max(err,float(abs(ncur-pn).max()),float(abs(pcur-pp).max()));nativeerr=max(nativeerr,float(abs(ncur[0]-np.array(old['cells'][f'{bg}/{geom}/{a}']['guard20']['curve'])).max()));cells+=1
 result={'passed':err<=1e-12 and nativeerr<=1e-5,'parent_curves_max':err,'source_native_max':nativeerr,'cells':cells,'all_rho_sealed':True};write(OUT/'baseline_validation.json',result);assert result['passed'];write(OUT/'score_lock.json',{'time':time.time(),'protocol_sha256':sha(OUT/'protocol.json'),'baseline_sha256':sha(OUT/'baseline_validation.json'),'status':'all_parent_curves_reproduced_before_new_independent_score'});print('BASELINE',result,flush=True)

def score():
 p=frozen();assert read(OUT/'baseline_validation.json')['passed'];assert sha(OUT/'baseline_validation.json')==read(OUT/'score_lock.json')['baseline_sha256']
 for row in p['rows']:
  sid=row['id']
  for bg in BGS:
   for geom in GEOMS:
    arrays={}
    for a in ['N','T']:
     v,au=aligned(row,a,bg,geom);n=len(v)+3;d=m.perm.gram(v,au);rho=np.load(paths(row,a))['rho'];pi=m.pi_for(sid,a);cur=curve(d,pi,rho,n);arrays[a+'/independent_curves']=cur;arrays[a+'/independent_metrics']=metric(cur)
    dest=OUT/'scores'/bg/geom/(sid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**arrays)
  resource();print('scored',sid,flush=True)
 print('SCORED',resource(),flush=True)

def values(sid,bg,geom,repeats=256):
 z=np.load(OUT/'scores'/bg/geom/(sid+'.npz'));out={}
 for a in ['N','T']:
  native,paired=parent_curves(sid,bg,geom,a);native=metric(native)[0];paired=metric(paired[:repeats]).mean(0);ind=z[a+'/independent_metrics'][:repeats].mean(0)
  out[a]={'native':native,'paired_whole':paired,'independent_whole':ind,'native_minus_paired':native-paired,'paired_minus_independent':paired-ind,'independent_residual':ind}
 out['gap']={k:out['T'][k]-out['N'][k] for k in out['N']};return out

def analyze():
 p=frozen();rows=p['rows'];ss=[r['speaker'] for r in rows];summary={};closure=0.;mc={}
 for geom in GEOMS:
  vals={bg:[values(r['id'],bg,geom) for r in rows] for bg in BGS}
  for bg in BGS+['LEVEL_minus_RAW']:
   for a in ['N','T','gap']:
    for key in vals['RAW'][0][a]:
     vv=np.stack([r[a][key] for r in vals[bg]]) if bg in BGS else np.stack([x[a][key]-y[a][key] for x,y in zip(vals['LEVEL'],vals['RAW'])]);summary[f'{geom}/{bg}/{a}/{key}']={f:m.base.bootstrap(vv[:,j],ss) for j,f in enumerate(FIELDS)}
   for r in (vals[bg] if bg in BGS else [{a:{key:x[a][key]-y[a][key] for key in x[a]} for a in x} for x,y in zip(vals['LEVEL'],vals['RAW'])]):
    for a in r:closure=max(closure,float(abs(r[a]['native_minus_paired']+r[a]['paired_minus_independent']+r[a]['independent_residual']-r[a]['native']).max()))
  for count in [64,128]:
   small={bg:[values(r['id'],bg,geom,count) for r in rows] for bg in BGS}
   for bg in BGS+['LEVEL_minus_RAW']:
    maximum=0.
    for a in ['N','T','gap']:
     for key in vals['RAW'][0][a]:
      delta=np.stack([x[a][key]-y[a][key] for x,y in zip(small[bg],vals[bg])]) if bg in BGS else np.stack([(u[a][key]-v[a][key])-(x[a][key]-y[a][key]) for u,v,x,y in zip(small['LEVEL'],small['RAW'],vals['LEVEL'],vals['RAW'])]);means=np.stack([delta[np.array(ss)==s].mean(0) for s in sorted(set(ss))]).mean(0);maximum=max(maximum,float(abs(means[[0,1,2,3,4,6]]).max()))
    mc[f'{geom}/{bg}/{count}_minus_256_max']=maximum
 desc=read(OUT/'accidental_correspondence.json');acc={}
 for a in ['N','T']:
  use=[r for r in desc if r['arm']==a];acc[a]={'I_min':min(r['I'] for r in use),'I_max':max(r['I'] for r in use),'observed':m.base.bootstrap([r['observed_mean'] for r in use],[r['speaker'] for r in use]),'expected':m.base.bootstrap([r['expected'] for r in use],[r['speaker'] for r in use]),'largest_observed_clips':sorted([{'id':r['id'],'I':r['I'],'observed':r['observed_mean'],'expected':r['expected']} for r in use],key=lambda r:r['observed'],reverse=True)}
 assert closure<1e-10;write(OUT/'summary.json',summary);write(OUT/'closure.json',{'max':closure,'passed':True});write(OUT/'mc_precision.json',mc);write(OUT/'accidental_summary.json',acc);print('ANALYZED',len(summary),resource(),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','baseline','score','analyze']);globals()[ap.parse_args().stage]()
