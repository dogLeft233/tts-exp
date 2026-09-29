"""Independent CPU vector, permutation, metric and cluster-statistic checks."""
from pathlib import Path
import argparse,json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_level_residual_representation_20260927';LEVEL=ROOT/'runs/tts_native_level_generation_cross_20260927';OLD=ROOT/'runs/tts_native_permutation_20260926'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
FIELDS=['C','B','D','D_anchor','C_anchor','best_lag'];POLS=['guard20','valid','guard0']
def metric(curves):
 b=np.median(curves,axis=1);d=np.min(curves,axis=1);a=curves[:,18]
 return np.column_stack([b-d,b,d,a,b-a,np.argmin(curves,axis=1)-15])
def check():
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
 for f,h in p['bindings'].items():assert sha(f)==h,f
 maximum=0.;curveerr=0.;anchorerr=0.;count=0;checks=0
 for row in p['rows']:
  sid=row['id']
  if sid not in p['support']['primary']:continue
  for bg in ['RAW','LEVEL']:
   for geom in ['raw','unit']:
    ca={};data=np.load(OUT/'scores'/bg/geom/(sid+'.npz'))
    for a in ['N','T']:
     n=row['arms'][a]['joint_L'];z=np.load(LEVEL/'features'/sid/a/(('raw' if bg=='RAW' else 'LEVEL')+'.npz'));v=z['visual'][:n];au=z['audio'][:n]
     if geom=='unit':v=(v.astype(float)/np.sqrt((v.astype(float)**2).sum(1))[:,None]).astype(np.float32);au=(au.astype(float)/np.sqrt((au.astype(float)**2).sum(1))[:,None]).astype(np.float32)
     v,au=v.astype(float),au.astype(float);m=(v[:-3]+au[3:])/2;r=(au[3:]-v[:-3])/2;mu=np.mean(m[20:n-20],axis=0);sigma=np.sqrt(np.mean(np.einsum('ij,ij->i',m[20:n-20]-mu,m[20:n-20]-mu)));ca[a]=(v,au,m,r,mu,sigma)
    for label in ['N_base','T_base','N_M','T_M']:
     a=label[0];v,au,m,r,mu,sigma=ca[a];n=len(v);alpha=ca['T' if a=='N' else 'N'][5]/sigma if label.endswith('_M') else 1.;vv=v.copy();aa=au.copy();mm=mu+alpha*(m-mu);vv[:-3]=mm-r;aa[3:]=mm+r
     anchorerr=max(anchorerr,float(abs((aa[3:]-vv[:-3])-2*r).max()));vv=vv.astype(np.float32).astype(float);aa=aa.astype(np.float32).astype(float);aap=np.vstack([aa,np.zeros((1,aa.shape[1]))]);d=np.empty((n,n+1))
     for start in range(0,n,8):
      dif=vv[start:start+8,None,:]-aap[None,:,:]+1e-6;d[start:start+8]=np.sqrt(np.einsum('ijk,ijk->ij',dif,dif))
     count+=n*(n+1);pi=np.load(OLD/'permutations/static_cloud'/sid/(('N' if a=='N' else 'C')+'.npz'))['whole'];allpi=np.vstack([np.arange(n)[None,:],pi]);ix=np.arange(n);vp=allpi.copy();vp[:,n-3:]=ix[n-3:];ap=np.tile(ix,(257,1));ap[:,3:]=allpi[:,:n-3]+3;curves={pol:np.empty((257,31)) for pol in POLS}
     for col,s in enumerate(range(-15,16)):
      q=ix+s;valid=(q>=0)&(q<n);aq=np.full((257,n),n);aq[:,valid]=ap[:,q[valid]];values=d[vp,aq];curves['guard20'][:,col]=values[:,20:-20].mean(1);curves['guard0'][:,col]=values.mean(1);curves['valid'][:,col]=values[:,valid].mean(1)
     for pol in POLS:
      got=metric(curves[pol]);expected=np.vstack([data[label+'/'+pol+'/identity'],data[label+'/'+pol+'/whole']]);maximum=max(maximum,float(abs(got-expected).max()));checks+=len(got)
     saved=np.vstack([data[label+'/guard20/identity_curve'],data[label+'/guard20/whole_curves']]);curveerr=max(curveerr,float(abs(curves['guard20']-saved).max()))
     ii=np.arange(20,n-20);assert np.array_equal(np.sort(pi[:,ii],axis=1),np.tile(ii,(256,1)))
  print('independent',sid,flush=True)
 assert maximum<1e-5 and curveerr<1e-5 and anchorerr<1e-12
 summary=read(OUT/'summary.json');rows=[r for r in p['rows'] if r['id'] in p['support']['primary']];groups=[r['speaker'] for r in rows];names=sorted(set(groups));inds=np.random.Generator(np.random.PCG64(20260926)).integers(len(names),size=(20000,len(names)));staterr=0.;closure=0.;statcount=0
 cache={}
 def vector(sid,bg,geom,pol,direction,effect,arm):
  key=(sid,bg,geom)
  if key not in cache:cache[key]=dict(np.load(OUT/'scores'/bg/geom/(sid+'.npz')))
  z=cache[key];labels=[('N_base','T_base'),('N_base','T_M'),('N_base','T_base'),('N_base','T_M')] if direction=='T_to_N' else [('N_base','T_base'),('N_M','T_base'),('N_base','T_base'),('N_M','T_base')];q=[]
  for index,(nl,tl) in enumerate(labels):
   op='identity' if index<2 else 'whole';nv=z[nl+'/'+pol+'/'+op].mean(0);tv=z[tl+'/'+pol+'/'+op].mean(0);q.append(nv if arm=='N' else tv if arm=='T' else tv-nv)
  coeff={'native':[1,0,0,0],'M':[0,1,0,0],'whole':[0,0,1,0],'M_whole':[0,0,0,1],'M_effect':[-1,1,0,0],'whole_effect':[-1,0,1,0],'M_after_whole':[0,0,-1,1],'whole_after_M':[0,-1,0,1],'interaction':[1,-1,-1,1]}[effect]
  return np.asarray(coeff)@np.asarray(q)
 for key,value in summary.items():
  geom,pol,bg,direction,effect,arm=key.split('/');vals=np.stack([vector(r['id'],bg,geom,pol,direction,effect,arm) if bg!='LEVEL_minus_RAW' else vector(r['id'],'LEVEL',geom,pol,direction,effect,arm)-vector(r['id'],'RAW',geom,pol,direction,effect,arm) for r in rows]);closure=max(closure,float(abs(vals[:,0]-vals[:,1]+vals[:,2]).max()),float(abs(vals[:,4]-vals[:,1]+vals[:,3]).max()));means=np.stack([vals[np.array(groups)==name].mean(0) for name in names]);ci=np.quantile(means[inds].mean(1),[.005,.995],axis=0)
  for j,f in enumerate(FIELDS):staterr=max(staterr,abs(float(means[:,j].mean())-value[f]['mean']),float(abs(ci[:,j]-value[f]['ci99']).max()));statcount+=1
 assert staterr<1e-10 and closure<1e-10
 write(OUT/'independent_validation.json',{'status':'PASS','direct_distances':count,'permutation_metric_rows':checks,'metric_max':maximum,'curve_max':curveerr,'anchor64_max':anchorerr,'statistics':statcount,'statistics_max':staterr,'closure_max':closure,'input_hashes':len(p['bindings'])})
 print('PASS',maximum,curveerr,staterr,flush=True)
if __name__=='__main__':check()
