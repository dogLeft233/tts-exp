"""Independent cached-cal LOSO prototype reconstruction; no new model/eval fit."""
import hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'runs/tts_fixed_generator_phone_prototype_20260927';O=ROOT/'runs/tts_fixed_generator_phone_prototype_independent_audit_20260927';j=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 fm=j(R/'fit.json');p=j(R/'protocol.json');assert j(R/'identity_gate.json')['passed'];assert fm['created_epoch']>j(R/'identity_gate.json')['created_epoch'];support=j(R/'feasibility.json')['rows'];rows={r['id']:r for r in j(R/'rows.json')};labels=fm['labels'];folds=fm['folds'];assert len(labels)==82 and len(folds)==15;perclip=[]
 for r in support:
  zz=np.full((2,83,512),np.nan)
  for ai,a in enumerate('NT'):
   z=np.load(rows[r['id']]['arms'][a]['features_path'])['z'].astype(np.float64);means=np.array([np.mean(z[e['indices'][a]],axis=0) for e in r['occurrences']]);phones=np.array([e['phone'] for e in r['occurrences']]);zz[ai,-1]=means.mean(0)
   for li,label in enumerate(labels):
    take=phones==label
    if take.any():zz[ai,li]=means[take].mean(0)
  perclip.append(zz)
 perclip=np.array(perclip);models=np.zeros((15,2,83,512));counts=np.zeros((15,82),np.int16)
 for fi,held in enumerate(folds):
  ids=[i for i,r in enumerate(support) if r['speaker']!=held];assert fm['training_ids'][held]==[support[i]['id'] for i in ids]
  for li in range(83):
   sm=[]
   for sp in sorted({support[k]['speaker'] for k in ids}):
    take=[i for i in ids if support[i]['speaker']==sp and np.isfinite(perclip[i,0,li]).all()]
    if take:sm.append(perclip[take,:,li].mean(0))
   if li==82:assert len(sm)>0
   if sm:models[fi,:,li]=np.array(sm).mean(0)
   if li<82:counts[fi,li]=len(sm)
  for li in range(82):
   if counts[fi,li]==0:models[fi,:,li]=models[fi,:,-1]
 actual=np.load(R/'fit.npz');err=float(abs(models-actual['raw_mu']).max());assert err<1e-12 and np.array_equal(counts,actual['speaker_counts']) and counts.tolist()==fm['counts'];assert np.isfinite(models).all() and (models>=0).all();fs=j(R/'fit_seal.json');assert fs['protocol_sha256']==sha(R/'protocol.json') and fs['identity_gate_sha256']==sha(R/'identity_gate.json')
 for f,h in fs['files'].items():assert sha(f)==h
 out=O/'fit_receipt.json';assert not out.exists();out.write_text(json.dumps({'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(R/'protocol.json'),'code_sha256':sha(__file__),'fit_sha256':sha(R/'fit.npz'),'fit_meta_sha256':sha(R/'fit.json'),'fit_max':err,'counts_and_training_ids_exact':True,'cal_only_occurrences':723,'folds':15,'labels':82,'global_support_nonempty':True,'identity_before_fit':True,'no_GPU_or_eval_fit':True},indent=2)+'\n');print(json.dumps({'receipt':str(out),'sha256':sha(out)}))
if __name__=='__main__':main()
