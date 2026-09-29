"""Full CPU-only independent natural80 source-direction result audit."""
from pathlib import Path
import gzip,hashlib,json,time
import numpy as np,torch
from tts_generator_holdout80_independent_common_20260927 import rebuild,ah
from tts_generator_shift_independent_common_20260927 import compare_dict
from tts_fixed_envelope_independent_result_20260927 import distance
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'runs/tts_fixed_generator_shift_holdout80_20260927';P=ROOT/'runs/tts_fixed_generator_phone_prototype_20260927';OLD=ROOT/'runs/tts_fixed_natural_holdout80_20260927';O=ROOT/'runs/tts_fixed_generator_shift_holdout80_independent_audit_20260927';CONDS=['baseline','phone_plus','phone_minus','global_plus','global_minus'];POLS=['guard20','valid','guard0','guard15'];METRICS=['C','B','D','C_anchor','D_anchor','search_uplift','best_lag','offset']
def j(p):
 p=Path(p);return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def summarize(m,policy):
 L=len(m)
 if policy=='valid':curve=np.asarray([m[max(0,-l):min(L,L-l),l+15].mean() for l in range(-15,16)])
 else:
  g=int(policy[5:]);curve=m[g:L-g].mean(0) if g else m.mean(0)
 k=int(np.argmin(curve));b=float(np.median(curve));d=float(curve[k]);anchor=float(curve[18]);return {'C':b-d,'B':b,'D':d,'C_anchor':b-anchor,'D_anchor':anchor,'best_lag':k-15,'offset':15-k,'curve':curve.tolist(),'search_uplift':anchor-d}
def statistics(x,names):
 labels=sorted(set(names));assert len(x)==80 and len(labels)==40 and all(names.count(s)==2 for s in labels);means=np.array([np.mean([v for v,s in zip(x,names) if s==a]) for a in labels]);ix=np.random.default_rng(20260926).integers(0,40,(20000,40));boot=means[ix].mean(1);return {'mean':float(means.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'ci99':np.quantile(boot,[.005,.995]).tolist(),'n':80,'speakers':40,'group_means':dict(zip(labels,means.tolist()))}
def contrasts(x):
 out={c+'/value':x[:,i] for i,c in enumerate(CONDS)}
 for family,plus,minus in [('phone',1,2),('global',3,4)]:
  out[family+'/plus_minus_baseline']=x[:,plus]-x[:,0];out[family+'/minus_minus_baseline']=x[:,minus]-x[:,0];out[family+'/plus_minus_minus']=x[:,plus]-x[:,minus]
 for name in ['plus_minus_baseline','minus_minus_baseline','plus_minus_minus']:out['phone_minus_global/'+name]=out['phone/'+name]-out['global/'+name]
 return out

def main():
 torch.set_num_threads(2);p=j(R/'protocol.json');rows=j(R/'input_rows.json');assert len(rows)==80;assert j(O/'input_receipt.json')['input_seal_sha256']==sha(R/'input_seal.json');assert j(O/'calibration_live_receipt.json')['status']=='PASS'
 for source in [p['dependencies'],j(R/'input_seal.json')['files'],j(R/'baseline_gate.json')['files'],j(R/'feature_seal.json')['files']]:
  for f,h in source.items():assert sha(f)==h,f
 lock=j(R/'score_lock.json');assert lock['protocol_sha256']==sha(R/'protocol.json') and lock['feature_seal_sha256']==sha(R/'feature_seal.json');assert j(R/'baseline_gate.json')['created_epoch']<j(R/'calibration_gate.json')['created_epoch']<lock['created_epoch'];assert j(R/'gpu_runtime/worker_exit_evaluation.json')['completed']==320;fit=np.load(P/'fit.npz');fm=j(P/'fit.json');scores={};nmat=0;curveerr=0.;mixerr=0.;cells=0;oldsummary=j(OLD/'summary.json');names=[r['speaker'] for r in rows]
 for row in rows:
  sid=row['id'];q=row['arms']['N'];cache=np.load(q['features_path']);A=cache['audio'];baseline=j(R/'baseline'/(sid+'.json'));assert baseline['passed'] and baseline['parentbindings']==q;z=np.load(baseline['z']['path']);assert ah(z)==baseline['z']['raw_sha256'] and sha(baseline['z']['path'])==baseline['z']['sha256'];oldmeta=j(q['metadata_path']);native=baseline['checks']['none'];assert native['A_exact'] and native['V_exact'] and native['pixel_exact'];assert native['A_raw_sha256']==ah(A) and native['V_raw_sha256']==ah(cache['visual']);assert native['video_at_creation']['pixel_sha256']==oldmeta['video']['pixel_sha256'];assert native['video_at_creation']['native_z_raw_sha256']==ah(z);want=j(R/'scores'/(sid+'.json.gz'))['cells'];oldscore=j(OLD/'scores'/(sid+'.json'));scores[sid]={}
  for condition in CONDS:
   if condition=='baseline':V=cache['visual']
   else:
    m=j(R/'metadata'/sid/'N'/(condition+'.json'));assert m['parentbindings']==q and m['native_z']==baseline['z'] and m['frontend']==baseline['frontend'];assert m['fit_sha256']==sha(P/'fit.npz') and m['fold']=='S0912';rep,d=rebuild(z,q,condition,fit,fm);mixerr=max(mixerr,compare_dict(d,m['mix']));vm=m['video_at_creation'];assert vm['native_z_raw_sha256']==ah(z) and vm['applied_z_raw_sha256']==ah(rep) and vm['hook_mode']=='processed';assert vm['pixel_sha256']==m['decode']['pixel_sha256'] and vm['frames']==m['decode']['frames']==q['frames'];assert len(m['decode']['PTS'])==q['frames'];V=np.load(m['V']['path']);assert V.dtype==np.float32 and V.shape==(q['frames']-4,1024) and np.isfinite(V).all() and ah(V)==m['V']['raw_sha256'] and sha(m['V']['path'])==m['V']['sha256'];cells+=1
    if sid==rows[0]['id']:assert np.array_equal(V,np.load(R/'gpu_controls/calibration'/(condition+'.npy')))
   scores[sid][condition]={}
   for geometry in ['raw','unit']:
    v,a=V[:row['L']],A[:row['L']]
    if geometry=='unit':v,a=[(x.astype(float)/np.linalg.norm(x.astype(float),axis=1)[:,None]).astype(np.float32) for x in [v,a]]
    matrix=distance(v,a);nmat+=matrix.size;scores[sid][condition][geometry]={}
    for policy in POLS:
     got=summarize(matrix,policy)
     for k,value in got.items():curveerr=max(curveerr,float(abs(np.asarray(value)-np.asarray(want[condition][geometry][policy][k])).max()))
     if condition=='baseline':
      for k,value in oldscore['cells'][geometry]['q11']['policies'][policy].items():assert np.array_equal(np.asarray(got[k]),np.asarray(value))
     scores[sid][condition][geometry][policy]=got
  print('checked',sid,flush=True)
 assert cells==320 and curveerr<1e-10 and mixerr<1e-10;summary=j(R/'summary.json.gz');seen=set();staterr=0.
 for g in ['raw','unit']:
  for pol in POLS:
   for metric in METRICS:
    x=np.array([[scores[r['id']][c][g][pol][metric] for c in CONDS] for r in rows]);terms=contrasts(x)
    for name,values in terms.items():
     key=f'{g}/{pol}/{metric}/{name}';got=statistics(values,names);target=summary[key];staterr=max(staterr,compare_dict(got,target));seen.add(key)
    if metric in p['scoring']['fields']:
     got=statistics(x[:,0],names);target=oldsummary['results'][g+'/'+pol][metric]['FIXED_native']
     for k in ['mean','ci99','n','speakers','group_means']:assert got[k]==target[k]
 assert seen==set(summary) and staterr<1e-10;criteria={m+'/'+c:summary['raw/guard20/'+m+'/global/'+c]['ci99'][0]>0 for m in ['C','C_anchor'] for c in ['plus_minus_baseline','plus_minus_minus']};analysis=j(R/'analysis.json');assert criteria==analysis['four_primary_tests'] and all(criteria.values())==analysis['global_direction_transfer_supported'];medians={g:{c:float(np.median([scores[r['id']][c][g]['guard20']['best_lag'] for r in rows])) for c in CONDS} for g in ['raw','unit']};assert medians==analysis['median_guard20_best_lag']
 out=O/'result_receipt.json';assert not out.exists();out.write_text(json.dumps({'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(R/'protocol.json'),'audit_code_sha256':sha(__file__),'new_cells':cells,'distance_entries':nmat,'curve_max':curveerr,'mix_max':mixerr,'statistics':len(seen),'statistics_max':staterr,'oldbaseline_all8views_and_summary_exact':True,'primary_four':criteria,'primary_conjunction':all(criteria.values()),'all80_40speaker_preserved':True,'no_GPU_or_model_forward':True,'media_scope':'external first2 sixidentity+firstid eighttreatment live and two old physical videos; other deletedbaseline/eval videos only creation-time producer metadata checked','hashes':{f:sha(R/f) for f in ['summary.json.gz','analysis.json','feature_seal.json','input_seal.json','score_lock.json']}},indent=2)+'\n');print(sha(out))
if __name__=='__main__':main()
