"""Independent real-array phone-field fit, distance and statistical audit; CPU only."""
import hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'runs/tts_fixed_phone_shared_field_20260927';O=ROOT/'runs/tts_fixed_phone_field_phase_independent_audit_20260927';F=ROOT/'runs/tts_fixed_level_generation_cross_20260927/features'
j=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(2**20),b''):h.update(x)
 return h.hexdigest()
def load(row,g):
 out={}
 for a in 'NT':
  with np.load(F/row['id']/a/'FIXED.npz') as z:
   for k in ['audio','visual']:
    x=z[k];out[a,k]=(x if not g else x/np.linalg.norm(x,axis=1)[:,None]).astype(np.float64)
 return out
def main():
 p=j(R/'protocol.json');ix=j(R/'indices.json');lab=ix['labels'];folds=ix['folds'];li={x:i for i,x in enumerate(lab)}
 for n,h in j(R/'seal.json').items():assert sha(R/n)==h
 for n,h in {**j(R/'inputs.json'),**p['code']}.items():assert sha(n)==h,n
 fitj=j(R/'fit.json');exe=j(R/'execution.json');assert j(R/'review.json')['protocol_sha256']==sha(R/'protocol.json')
 assert j(R/'bridge.json')['created_epoch']<fitj['created_epoch']<exe['created_epoch'];assert fitj['fit_sha256']==sha(R/'fit.npz')
 records=[]
 for row in ix['cal_rows']:
  z=np.full((2,64,1024),np.nan)
  for g in range(2):
   f=load(row,g);by={x:[] for x in lab};allv=[]
   for q in row['queries']:
    n=row['nodes'][q['node']];v=((f['T','audio'][n['j']['T']]+f['T','visual'][n['j']['T']-3])-(f['N','audio'][n['j']['N']]+f['N','visual'][n['j']['N']-3]))/2
    by[n['phone']].append(v);allv.append(v)
   z[g,-1]=np.mean(allv,axis=0)
   for l,v in by.items():
    if v:z[g,li[l]]=np.mean(v,axis=0)
  records.append(z)
 records=np.asarray(records);models=np.zeros((len(folds),2,64,1024));counts=np.zeros((len(folds),63),np.int16)
 for fi,held in enumerate(folds):
  use=[k for k,r in enumerate(ix['cal_rows']) if r['speaker']!=held];assert fitj['training_ids'][held]==[ix['cal_rows'][k]['id'] for k in use]
  for l in range(64):
   means=[]
   for sp in sorted({ix['cal_rows'][k]['speaker'] for k in use}):
    take=[k for k in use if ix['cal_rows'][k]['speaker']==sp and np.isfinite(records[k,0,l]).all()]
    if take:means.append(records[take,:,l].mean(0))
   if means:models[fi,:,l]=np.mean(means,axis=0)
   if l<63:counts[fi,l]=len(means)
  for l in range(63):
   if not counts[fi,l]:models[fi,:,l]=models[fi,:,-1]
 saved=np.load(R/'fit.npz');fiterr=float(abs(saved['raw_delta']-models).max());assert fiterr<1e-12 and np.array_equal(saved['speaker_counts'],counts)
 Q=dict(np.load(R/'Q.npz'));rng=np.random.default_rng(20260927)
 for k in range(16):assert np.array_equal(Q['permutation'][k],rng.permutation(1024)) and np.array_equal(Q['sign'][k],rng.choice(np.array([-1,1],np.int8),1024))
 expected=np.load(R/'query_metrics.npz')['metrics'];calc=np.empty_like(expected);square=0.;same=0.
 for row,span in zip(ix['eval_rows'],ix['spans']):
  for g in range(2):
   f=load(row,g)
   for ci,spec in enumerate(p['specs']):
    fam=spec['family'];delta=np.zeros((64,1024))
    if fam:
     fi=folds.index('pooled' if fam=='pooled_shrink' else row['speaker']);glob=models[fi,g,-1];eta=models[fi,g,:-1]-glob
     if fam=='loso_global':eta*=0
     elif fam!='loso_raw':eta*= (counts[fi]/(counts[fi]+4))[:,None]
     if 'Q' in spec:eta=eta[:,Q['permutation'][spec['Q']]]*Q['sign'][spec['Q']]
     delta[:-1]=glob+eta;delta[-1]=glob
    for ai,a in enumerate('NT'):
     sign=(-1 if a=='T' else 1) if spec['sign']=='main' else spec['sign']
     for qi,q in enumerate(row['queries']):
      n=row['nodes'][q['node']];nds=[n]+[row['nodes'][d] for d in q['donors']];dlabels=[li.get(x['phone'],63) for x in nds];audio=f[a,'audio'][[x['j'][a] for x in nds]];video=f[a,'visual'][n['j'][a]-3]
      shifts=sign*delta[dlabels];vshift=shifts[0];d=(video+vshift)-(audio+shifts)+1e-6;orig=video-audio+1e-6;w=vshift-shifts
      square=max(square,float(abs((d*d).sum(1)-((orig*orig).sum(1)+2*(orig*w).sum(1)+(w*w).sum(1))).max()))
      mask=np.array([x['phone']==n['phone'] for x in nds]);same=max(same,float(abs(d[mask]-orig[mask]).max()))
      ds=np.linalg.norm(d,axis=1);pos=ds[0];neg=ds[1:];avg=neg.mean();calc[span['start']+qi,g,ci,ai]=[pos,avg,avg-pos,(np.count_nonzero(neg>pos)+.5*np.count_nonzero(neg==pos))/len(neg)]
 qe=float(abs(calc-expected).max());assert qe<1e-10 and square<1e-9 and same<1e-12,(qe,square,same)
 cm=np.asarray([calc[s['start']:s['stop']].mean(0) for s in ix['spans']]);ce=float(abs(cm-np.load(R/'field_clips.npz')['metrics']).max());assert ce<1e-10
 speakers=[s['speaker'] for s in ix['spans']];names=sorted(set(speakers));masks=[np.array(speakers)==s for s in names];cnt=np.array([m.sum() for m in masks]);draw=np.random.default_rng(20260926).integers(len(names),size=(20000,len(names)))
 summary=j(R/'field_summary.json');derived=j(R/'derived_summary.json');target={**summary,**derived};series={}
 for g,gn in enumerate(['raw','unit']):
  for m,mn in enumerate(['positive','negative','margin','rank']):
   prefix=gn+'/'+mn+'/';base=cm[:,g,0,:,m]
   for ci,spec in enumerate(p['specs']):
    val=cm[:,g,ci,:,m]
    for ai,a in enumerate('NT'):
     key=prefix+spec['name']+'/';response=val[:,ai]-base[:,ai];series[key+a]=val[:,ai];series[key+'response_'+a]=response;series[key+'gap_change_only_'+a]=response*(1 if a=='T' else -1)
    series[prefix+spec['name']+'/both_shifted_T_minus_N']=val[:,1]-val[:,0]
   series[prefix+'sign_specificity/Tminus_minus_Tplus']=cm[:,g,1,1,m]-cm[:,g,2,1,m];series[prefix+'sign_specificity/Nplus_minus_Nminus']=cm[:,g,2,0,m]-cm[:,g,1,0,m]
   for ai,a in enumerate('NT'):series[prefix+'main_minus_meanQ/'+a]=cm[:,g,1 if a=='T' else 2,ai,m]-cm[:,g,9:,:,m].mean(1)[:,ai]
   series[prefix+'gap_after_only_Tminus']=cm[:,g,1,1,m]-base[:,0];series[prefix+'gap_after_only_Nplus']=base[:,1]-cm[:,g,2,0,m]
 assert set(series)==set(target);err=0.;nscalar=0
 for key,v in series.items():
  means=np.asarray([v[mask].mean() for mask in masks]);sim=means[draw];weighted=(sim*cnt[draw]).sum(1)/cnt[draw].sum(1)
  got={'n':32,'speakers':13,'speaker_mean':means.mean(),'speaker_ci95':np.quantile(sim.mean(1),[.025,.975]),'speaker_ci99':np.quantile(sim.mean(1),[.005,.995]),'utterance_mean':v.mean(),'utterance_cluster_ci95':np.quantile(weighted,[.025,.975]),'positive':int((v>0).sum()),'speaker_positive':int((means>0).sum()),'per_speaker':dict(zip(names,means))}
  for k,z in got.items():
   vals=np.array([z[s]-target[key][k][s] for s in names]) if k=='per_speaker' else np.asarray(z)-np.asarray(target[key][k]);
   if float(abs(vals).max())>1e-10:print('STATDIFF',key,k,float(abs(vals).max()),flush=True)
   err=max(err,float(abs(vals).max()));nscalar+=vals.size
 print('FIT_QUERY',fiterr,qe,ce,flush=True);assert err<1e-10,err
 t=target['raw/margin/loso_shrink_-1/response_T']['speaker_ci99'];n=target['raw/margin/loso_shrink_+1/response_N']['speaker_ci99'];ts=target['raw/margin/sign_specificity/Tminus_minus_Tplus']['speaker_ci99'];ns=target['raw/margin/sign_specificity/Nplus_minus_Nminus']['speaker_ci99']
 criteria={'raw_Tminus_decreases_CI99':t[1]<0,'raw_Nplus_increases_CI99':n[0]>0,'raw_both_directions':t[1]<0 and n[0]>0,'raw_T_sign_specificity':ts[1]<0,'raw_N_sign_specificity':ns[0]>0};assert criteria==j(R/'field_criteria.json')
 receipt={'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(R/'protocol.json'),'audit_code_sha256':sha(__file__),'fit_max':fiterr,'counts_training_ids_exact':True,'query_entries':calc.size,'query_max':qe,'clip_max':ce,'squared_expansion_max':square,'same_phone_vector_max':same,'statistics_max':err,'stat_endpoints':len(series),'stat_scalars':nscalar,'criteria_exact':criteria,'support':[20,890,32,13,1411],'no_GPU':True,'hashes':{f:sha(R/f) for f in ['fit.npz','fit.json','query_metrics.npz','field_clips.npz','field_summary.json','derived_summary.json','execution.json','field_criteria.json']}}
 dest=O/'field_result_receipt.json';assert not dest.exists();dest.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'status':'PASS','path':str(dest),'sha256':sha(dest),'max':err}))
if __name__=='__main__':main()
