"""Independent frozen field x phase joint replay. No field refit or model forward."""
import gzip,hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'runs/tts_fixed_phone_shared_field_20260927';O=ROOT/'runs/tts_fixed_phone_field_phase_independent_audit_20260927'
j=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ix=j(R/'indices.json');binding=j(R/'phase_binding.json');assert j(R/'phase_review.json')['phase_binding_sha256']==sha(R/'phase_binding.json');assert binding['protocol_sha256']==sha(R/'protocol.json')
 for f,h in binding['inputs'].items():assert sha(f)==h,f
 earlier=j(O/'field_result_receipt.json');assert earlier['status']=='PASS' and earlier['hashes']['fit.npz']==sha(R/'fit.npz');fit=np.load(R/'fit.npz');models,counts=fit['raw_delta'],fit['speaker_counts'];labels={x:i for i,x in enumerate(ix['labels'])};p=j(binding['producer_binding_path']);assert p['created_epoch']<j(binding['phase_execution_lock_path'])['created_epoch']
 actual=np.load(R/'phase_query_metrics.npz')['metrics'];calc=np.empty_like(actual);same=0.;sqerr=0.
 for row,span in zip(ix['eval_rows'],ix['spans']):
  raw={}
  for a in 'NT':
   old=np.load(ROOT/'runs/tts_fixed_level_generation_cross_20260927/features'/row['id']/a/'FIXED.npz');entry=binding['features'][row['id']][a];new=np.load(entry['path']);raw[a]=(old['visual'],old['audio'],new)
  fi=ix['folds'].index(row['speaker'])
  for gi in range(2):
   arr={a:tuple((x/np.linalg.norm(x,axis=1)[:,None] if gi else x).astype(float) for x in xs) for a,xs in raw.items()};glob=models[fi,gi,-1];deltas=np.vstack([glob+(models[fi,gi,:-1]-glob)*(counts[fi]/(counts[fi]+4))[:,None],glob])
   for di,target in enumerate('TN'):
    for pi in range(2):
     for on in range(2):
      for ai,a in enumerate('NT'):
       sign=(-1 if a=='T' else 1) if on and a==target else 0
       for qi,q in enumerate(row['queries']):
        n=row['nodes'][q['node']];ns=[n]+[row['nodes'][z] for z in q['donors']];shifts=sign*deltas[[labels.get(v['phone'],63) for v in ns]];v=arr[a][0][n['j'][a]-3];audio=arr[a][1+pi][[v['j'][a] for v in ns]];distvec=(v+shifts[0])-(audio+shifts)+1e-6;orig=v-audio+1e-6;w=shifts[0]-shifts;same=max(same,float(abs(distvec[0]-orig[0]).max()));sqerr=max(sqerr,float(abs((distvec*distvec).sum(1)-((orig*orig).sum(1)+2*(orig*w).sum(1)+(w*w).sum(1))).max()));ds=np.linalg.norm(distvec,axis=1);pos=ds[0];neg=ds[1:];calc[span['start']+qi,gi,di,2*pi+on,ai]=[pos,neg.mean(),neg.mean()-pos,(np.count_nonzero(neg>pos)+.5*np.count_nonzero(neg==pos))/len(neg)]
 qe=float(abs(calc-actual).max());assert qe<1e-10 and same<1e-12 and sqerr<1e-9
 cm=np.array([calc[s['start']:s['stop']].mean(0) for s in ix['spans']]);ce=float(abs(cm-np.load(R/'phase_clip_metrics.npz')['metrics']).max());assert ce<1e-10
 baseline=np.load(R/'baseline.npz')['metrics'];assert abs(calc[:,:,:,0]-baseline[:,:,None]).max()<1e-10
 groups=[s['speaker'] for s in ix['spans']];names=sorted(set(groups));masks=[np.array(groups)==s for s in names];counts=np.array([m.sum() for m in masks]);draw=np.random.default_rng(20260926).integers(len(names),size=(20000,len(names)));summary=j(R/'phase_summary.json');series={};closure=0.
 for gi,g in enumerate(['raw','unit']):
  for di,target in enumerate('TN'):
   for mi,m in enumerate(['positive','negative','margin','rank']):
    z=cm[:,gi,di,:,:,mi];values={name:z[:,ci] for ci,name in enumerate(['identity','field','phase','both'])};values.update(field_effect=z[:,1]-z[:,0],phase_effect=z[:,2]-z[:,0],interaction=z[:,3]-z[:,1]-z[:,2]+z[:,0],total=z[:,3]-z[:,0]);closure=max(closure,float(abs(values['field_effect']+values['phase_effect']+values['interaction']-values['total']).max()))
    for name,x in values.items():
     key=f'{g}/{m}/only_field_{target}/{name}/'
     for ai,a in enumerate('NT'):series[key+a]=x[:,ai]
     series[key+'T_minus_N']=x[:,1]-x[:,0]
 assert set(series)==set(summary);err=0.;scalar=0
 for key,v in series.items():
  means=np.array([v[mask].mean() for mask in masks]);sim=means[draw];weighted=(sim*counts[draw]).sum(1)/counts[draw].sum(1);got={'n':32,'speakers':13,'speaker_mean':means.mean(),'speaker_ci95':np.quantile(sim.mean(1),[.025,.975]),'speaker_ci99':np.quantile(sim.mean(1),[.005,.995]),'utterance_mean':v.mean(),'utterance_cluster_ci95':np.quantile(weighted,[.025,.975]),'positive':int((v>0).sum()),'speaker_positive':int((means>0).sum()),'per_speaker':dict(zip(names,means))}
  for k,z in got.items():
   diff=np.array([z[s]-summary[key][k][s] for s in names]) if k=='per_speaker' else np.asarray(z)-np.asarray(summary[key][k]);err=max(err,float(abs(diff).max()));scalar+=diff.size
 assert err<1e-10 and closure<1e-10
 receipt={'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(R/'protocol.json'),'binding_sha256':sha(R/'phase_binding.json'),'audit_code_sha256':sha(__file__),'original_fit_sha256':sha(R/'fit.npz'),'query_entries':calc.size,'query_max':qe,'clip_max':ce,'stat_endpoints':len(series),'stat_scalars':scalar,'stat_max':err,'positive_vector_max':same,'squared_expansion_max':sqerr,'closure_max':closure,'no_refit_GPU_or_support_selection':True,'hashes':{f:sha(R/f) for f in ['phase_query_metrics.npz','phase_clip_metrics.npz','phase_summary.json','phase_execution.json']}}
 out=O/'joint_result_receipt.json';assert not out.exists();out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'receipt':str(out),'sha256':sha(out)}))
if __name__=='__main__':main()
