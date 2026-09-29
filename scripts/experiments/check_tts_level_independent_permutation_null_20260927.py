"""Independent null distances, rho generation, statistics and closure verification."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_level_independent_permutation_null_20260927';PARENT=ROOT/'runs/tts_level_residual_representation_20260927';LEVEL=ROOT/'runs/tts_native_level_generation_cross_20260927';OLD=ROOT/'runs/tts_native_permutation_20260926'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();FIELDS=['C','B','D','D_anchor','C_anchor','best_lag','search_uplift']
def met(z):
 med=np.median(z,axis=1);minimum=np.min(z,axis=1);anchor=z[:,18]
 return np.column_stack([med-minimum,med,minimum,anchor,med-anchor,np.argmin(z,axis=1)-15,anchor-minimum])
def main():
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
 for f,h in p['bindings'].items():assert sha(f)==h,f
 desc={(r['id'],r['arm']):r for r in read(OUT/'accidental_correspondence.json')};maxdist=0.;maxmetric=0.;nativeerr=0.;count=0.;records={};marginals=0;coincidenceerr=0.
 for row in p['rows']:
  sid=row['id']
  for a in ['N','T']:
   n=row['arms'][a]['joint_L'];ii=np.arange(20,n-20);jj=np.arange(n-3);rho=np.load(OUT/'rho'/sid/(a+'.npz'))['rho'];pi=np.load(OLD/'permutations/static_cloud'/sid/(('N' if a=='N' else 'C')+'.npz'))['whole'];reg=np.where(jj<20,0,np.where(jj<n-20,1,2));reconstructed=np.tile(np.arange(n,dtype=np.int32),(256,1))
   for b in range(256):
    seed=int.from_bytes(hashlib.sha256(f'20260927|{sid}|{a}|3|whole|{b}'.encode()).digest()[:16],'big');rng=np.random.Generator(np.random.PCG64(seed))
    for region in range(3):
     block=jj[reg==region];reconstructed[b,block]=rng.permutation(block);assert np.array_equal(np.sort(rho[b,block]),block);assert np.array_equal(np.sort(pi[b,block]),block);marginals+=2
   assert np.array_equal(rho,reconstructed);assert desc[sid,a]['domain_lengths']==[int((reg==r).sum()) for r in range(3)];assert desc[sid,a]['expected']==1/len(ii);coincidenceerr=max(coincidenceerr,float(abs(np.mean(rho[:,ii]==pi[:,ii],axis=1)-desc[sid,a]['accidental_fractions']).max()))
  for bg in ['RAW','LEVEL']:
   for geom in ['raw','unit']:
    result=np.load(OUT/'scores'/bg/geom/(sid+'.npz'));prior=np.load(PARENT/'scores'/bg/geom/(sid+'.npz'));records[sid,bg,geom]={}
    for a in ['N','T']:
     n=row['arms'][a]['joint_L'];z=np.load(LEVEL/'features'/sid/a/(('raw' if bg=='RAW' else 'LEVEL')+'.npz'));v=z['visual'][:n];au=z['audio'][:n]
     if geom=='unit':v=(v.astype(float)/np.sqrt((v.astype(float)**2).sum(1))[:,None]).astype(np.float32);au=(au.astype(float)/np.sqrt((au.astype(float)**2).sum(1))[:,None]).astype(np.float32)
     v=v[:-3].astype(float);au=au[3:].astype(float);d=np.empty((len(v),len(au)))
     for start in range(0,len(v),8):
      diff=v[start:start+8,None,:]-au[None,:,:]+1e-6;d[start:start+8]=np.sqrt(np.sum(diff**2,axis=2))
     count+=d.size;rho=np.load(OUT/'rho'/sid/(a+'.npz'))['rho'];pi=np.load(OLD/'permutations/static_cloud'/sid/(('N' if a=='N' else 'C')+'.npz'))['whole'];idx=np.arange(20,n-20);cols=idx[:,None]+np.arange(-15,16)-3;curves=np.stack([d[x[idx[:,None]],y[cols]].mean(0) for x,y in zip(pi,rho)]);saved=result[a+'/independent_curves'];maxdist=max(maxdist,float(abs(curves-saved).max()));maxmetric=max(maxmetric,float(abs(met(curves)-result[a+'/independent_metrics']).max()))
     pc=prior[a+'_base/guard20/whole_curves'];nc=prior[a+'_base/guard20/identity_curve'];ncur=d[idx[:,None],cols].mean(0);pcur=np.stack([d[x[idx[:,None]],x[cols]].mean(0) for x in pi]);nativeerr=max(nativeerr,float(abs(ncur-nc[0]).max()),float(abs(pcur-pc).max()));records[sid,bg,geom][a]=np.stack([met(nc)[0],met(pc).mean(0),met(curves).mean(0)])
  print('independent',sid,flush=True)
 assert max(maxdist,maxmetric,nativeerr)<1e-10 and coincidenceerr==0.
 summary=read(OUT/'summary.json');sp=[r['speaker'] for r in p['rows']];names=sorted(set(sp));boot=np.random.Generator(np.random.PCG64(20260926)).integers(0,len(names),(20000,len(names)));maxstat=0.;closure=0.;statcount=0
 weights={'native':[1,0,0],'paired_whole':[0,1,0],'independent_whole':[0,0,1],'native_minus_paired':[1,-1,0],'paired_minus_independent':[0,1,-1],'independent_residual':[0,0,1]}
 for key,expected in summary.items():
  geom,bg,arm,term=key.split('/');vals=[]
  for row in p['rows']:
   def get(b):
    q=records[row['id'],b,geom];return q['T']-q['N'] if arm=='gap' else q[arm]
   full=get(bg) if bg!='LEVEL_minus_RAW' else get('LEVEL')-get('RAW');closure=max(closure,float(abs((full[0]-full[1])+(full[1]-full[2])+full[2]-full[0]).max()));vals.append(np.array(weights[term])@full)
  vals=np.stack(vals);groupmeans=np.stack([vals[np.array(sp)==g].mean(0) for g in names]);ci=np.quantile(groupmeans[boot].mean(1),[.005,.995],axis=0)
  for j,f in enumerate(FIELDS):maxstat=max(maxstat,abs(float(groupmeans[:,j].mean())-expected[f]['mean']),float(abs(ci[:,j]-expected[f]['ci99']).max()));statcount+=1
 assert maxstat<1e-10 and closure<1e-10
 result={'status':'PASS','direct_distances':int(count),'curve_max':maxdist,'metric_max':maxmetric,'parent_curves_max':nativeerr,'marginal_multiset_checks':marginals,'rho_regenerated_exact':True,'coincidence_max':coincidenceerr,'statistics':statcount,'statistics_max':maxstat,'closure_max':closure,'source_hashes':len(p['bindings'])};(OUT/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
