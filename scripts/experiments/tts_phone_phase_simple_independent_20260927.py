"""Independent conditional additions from already independently checked joint clips."""
import hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'runs/tts_fixed_phone_shared_field_20260927';O=ROOT/'runs/tts_fixed_phone_field_phase_independent_audit_20260927';j=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert sha(R/'phase_clip_metrics.npz')==j(O/'joint_result_receipt.json')['hashes']['phase_clip_metrics.npz'];cm=np.load(R/'phase_clip_metrics.npz')['metrics'];idx=j(R/'indices.json');target=j(R/'phase_derived_summary.json');groups=[x['speaker'] for x in idx['spans']];names=sorted(set(groups));masks=[np.array(groups)==s for s in names];counts=np.array([m.sum() for m in masks]);draw=np.random.default_rng(20260926).integers(13,size=(20000,13));series={}
 for gi,g in enumerate(['raw','unit']):
  for di,t in enumerate('TN'):
   for mi,m in enumerate(['positive','negative','margin','rank']):
    for name,base in [('field_at_phase',2),('phase_at_field',1)]:
     val=cm[:,gi,di,3,:,mi]-cm[:,gi,di,base,:,mi];key=f'{g}/{m}/only_field_{t}/{name}/'
     for ai,a in enumerate('NT'):series[key+a]=val[:,ai]
     series[key+'T_minus_N']=val[:,1]-val[:,0]
 assert set(series)==set(target);err=0.
 for key,v in series.items():
  means=np.array([v[mask].mean() for mask in masks]);sim=means[draw];weighted=(sim*counts[draw]).sum(1)/counts[draw].sum(1);got={'n':32,'speakers':13,'speaker_mean':means.mean(),'speaker_ci95':np.quantile(sim.mean(1),[.025,.975]),'speaker_ci99':np.quantile(sim.mean(1),[.005,.995]),'utterance_mean':v.mean(),'utterance_cluster_ci95':np.quantile(weighted,[.025,.975]),'positive':int((v>0).sum()),'speaker_positive':int((means>0).sum()),'per_speaker':dict(zip(names,means))}
  for k,z in got.items():
   dif=np.array([z[s]-target[key][k][s] for s in names]) if isinstance(z,dict) else np.asarray(z)-np.asarray(target[key][k]);err=max(err,float(abs(dif).max()))
 assert err<1e-10;out=O/'joint_simple_receipt.json';assert not out.exists();out.write_text(json.dumps({'status':'PASS','created_epoch':time.time(),'code_sha256':sha(__file__),'source_clip_sha256':sha(R/'phase_clip_metrics.npz'),'derived_sha256':sha(R/'phase_derived_summary.json'),'statistics':len(series),'max_error':err,'no_new_fit_forward_or_intervention':True},indent=2)+'\n');print(json.dumps({'receipt':str(out),'sha256':sha(out),'N_raw':target['raw/margin/only_field_N/field_at_phase/N'],'N_unit':target['unit/margin/only_field_N/field_at_phase/N']}))
if __name__=='__main__':main()
