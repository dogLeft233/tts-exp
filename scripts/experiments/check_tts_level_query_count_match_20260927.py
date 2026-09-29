"""Independent source-distance, subset, operator-order and bootstrap validation."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_level_query_count_match_20260927';LEVEL=ROOT/'runs/tts_native_level_generation_cross_20260927';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();FIELDS=['C','B','D','D_anchor','C_anchor','best_lag','search_uplift']
def metric(z):
 med=np.median(z,axis=1);d=z.min(1);a=z[:,18]
 return np.column_stack([med-d,med,d,a,med-a,z.argmin(1)-15,a-d])
def main():
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
 for f,h in p['bindings'].items():assert sha(f)==h,f
 for f,h in read(OUT/'score_lock.json')['distances'].items():assert sha(f)==h
 records={};distmax=0.;curveerr=0.;metricerr=0.;source_metricerr=0.;distcount=0;subsetcount=0;close_lag_disagreements=0
 for row,info in zip(p['rows'],p['support']):
  sid=row['id'];assert sid==info['id'];mats=np.load(OUT/'distances'/(sid+'.npz'))
  for a in ['N','T']:
   n=row['arms'][a]['joint_L'];q=info['q'];ii=np.arange(20,n-20);sub=np.load(OUT/'indices'/sid/(a+'.npz'))['indices'];assert sub.shape==(1024,q)
   for b in range(1024):
    seed=int.from_bytes(hashlib.sha256(f'20260927|query_count|{sid}|{a}|{b}'.encode()).digest()[:16],'big');sample=np.sort(np.random.Generator(np.random.PCG64(seed)).choice(ii,q,replace=False));assert np.array_equal(sample,sub[b]);assert len(set(map(int,sub[b])))==q and set(map(int,sub[b])).issubset(set(ii));subsetcount+=1
  for bg in ['RAW','LEVEL']:
   for geom in ['raw','unit']:
    saved=np.load(OUT/'scores'/bg/geom/(sid+'.npz'));records[sid,bg,geom]={}
    for a in ['N','T']:
     n=row['arms'][a]['joint_L'];z=np.load(LEVEL/'features'/sid/a/(('raw' if bg=='RAW' else 'LEVEL')+'.npz'));v=z['visual'][:n];au=z['audio'][:n]
     if geom=='unit':v=(v.astype(float)/np.linalg.norm(v.astype(float),axis=1,keepdims=True)).astype(np.float32);au=(au.astype(float)/np.linalg.norm(au.astype(float),axis=1,keepdims=True)).astype(np.float32)
     v=v.astype(float);au=np.pad(au.astype(float),((15,15),(0,0)));direct=np.column_stack([np.sqrt(np.sum((v-au[15+s:15+s+n]+1e-6)**2,axis=1)) for s in range(-15,16)]);matrix=mats['/'.join([bg,geom,a])];distmax=max(distmax,float(abs(direct-matrix).max()));distcount+=direct.size;sub=np.load(OUT/'indices'/sid/(a+'.npz'))['indices'];curves=np.stack([matrix[ix].mean(axis=0) for ix in sub]);direct_curves=np.stack([direct[ix].mean(axis=0) for ix in sub]);full=matrix[20:-20].mean(0);mean=curves.mean(0);fullmet=metric(full[None,:]);allmet=metric(curves);meanmet=metric(mean[None,:]);dmet=metric(direct_curves);source_metricerr=max(source_metricerr,float(abs(dmet[:,[0,1,2,3,4,6]]-allmet[:,[0,1,2,3,4,6]]).max()));diff=allmet[:,5]!=dmet[:,5]
     if np.any(diff):
      for ix in np.flatnonzero(diff):assert abs(direct_curves[ix,int(allmet[ix,5]+15)]-direct_curves[ix,int(dmet[ix,5]+15)])<=2*distmax
      close_lag_disagreements+=int(diff.sum())
     curveerr=max(curveerr,float(abs(full-saved[a+'/full_curve']).max()),float(abs(mean-saved[a+'/mean_curve']).max()));metricerr=max(metricerr,float(abs(allmet-saved[a+'/subset_metrics']).max()),float(abs(fullmet-saved[a+'/full_metrics']).max()),float(abs(meanmet-saved[a+'/mean_curve_metrics']).max()));records[sid,bg,geom][a]=np.stack([fullmet[0],allmet.mean(0),meanmet[0]])
  print('independent',sid,flush=True)
 assert distmax<1e-5 and source_metricerr<2e-5 and max(curveerr,metricerr)<1e-12
 summary=read(OUT/'summary.json');groups=[r['speaker'] for r in p['rows']];names=sorted(set(groups));bi=np.random.Generator(np.random.PCG64(20260926)).integers(len(names),size=(20000,len(names)));weights={'full':[1,0,0],'matched':[0,1,0],'matched_minus_full':[-1,1,0],'metric_of_mean_curve':[0,0,1],'mean_curve_minus_full':[-1,0,1],'mean_metric_minus_metric_mean':[0,1,-1]};staterr=0.;closure=0.;checks=0
 for key,expect in summary.items():
  geom,bg,arm,term=key.split('/');vals=[]
  for row in p['rows']:
   def get(b):
    d=records[row['id'],b,geom];return d['T']-d['N'] if arm=='gap' else d[arm]
   z=get(bg) if bg!='LEVEL_minus_RAW' else get('LEVEL')-get('RAW');vals.append(np.array(weights[term])@z)
  vals=np.stack(vals);closure=max(closure,float(abs(vals[:,0]-vals[:,4]-vals[:,6]).max()));means=np.stack([vals[np.array(groups)==g].mean(0) for g in names]);ci=np.quantile(means[bi].mean(1),[.005,.995],axis=0)
  for j,f in enumerate(FIELDS):staterr=max(staterr,abs(float(means[:,j].mean())-expect[f]['mean']),float(abs(ci[:,j]-expect[f]['ci99']).max()));checks+=1
 assert staterr<1e-10 and closure<1e-10
 result={'status':'PASS','direct_distances':distcount,'distance_float64_vs_torch_float32_max':distmax,'independent_source_metric_max':source_metricerr,'near_tie_bestlag_rounding_cases':close_lag_disagreements,'subset_indices_regenerated':subsetcount,'curve_reconstruction_max':curveerr,'metric_max':metricerr,'statistics':checks,'statistics_max':staterr,'closure_max':closure,'input_hashes':len(p['bindings'])};(OUT/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
