"""Independent report-only cached prototype displacement dose checks."""
import gzip,json,hashlib,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'runs/tts_fixed_generator_shift_cross_20260927';P=ROOT/'runs/tts_fixed_generator_phone_prototype_20260927';O=ROOT/'runs/tts_fixed_generator_shift_independent_audit_20260927'
def j(p):
 p=Path(p);return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def error(a,b):
 if isinstance(a,dict):assert a.keys()==b.keys();return max([error(a[k],b[k]) for k in a]+[0.])
 if isinstance(a,(int,float,np.number)):return abs(float(a)-float(b))
 assert a==b,(a,b);return 0.
def main():
 x=j(R/'dose_descriptions.json.gz');assert x['protocol_sha256']==sha(R/'protocol.json') and x['fit_sha256']==sha(P/'fit.npz');assert x['script_sha256']==sha(ROOT/'scripts/experiments/tts_generator_prototype_doses_20260927.py');assert j(O/'result_receipt.json')['status']=='PASS';desc=j(R/'operation_descriptions.json.gz');rows={r['id']:r for r in j(P/'rows.json')};fit=np.load(P/'fit.npz');fm=j(P/'fit.json');labels={s:i for i,s in enumerate(fm['labels'])};err=0.;used=set();rebuilt=[]
 for c in x['clips']:
  sid,a,kind=c['id'],c['arm'],c['kind'];assert (sid,a,kind) not in used;used.add((sid,a,kind));r=rows[sid];assert r['split']=='evaluation';q=r['arms'][a];z=np.load(q['features_path'])['z'];active=np.asarray(q['speech_mask']);fi=fm['folds'].index(r['speaker']);mu=fit['raw_mu'][fi,fm['sources'].index(a)];count=fit['speaker_counts'][fi];rep=z.copy()
  for i in np.flatnonzero(active):
   ix=labels.get(q['phone_labels'][i]);template=mu[-1].astype(np.float32)
   if kind=='phone' and ix is not None and count[ix]>0:
    w=np.float64(count[ix])/np.float64(count[ix]+4);template=(w*mu[ix]+(1-w)*mu[-1]).astype(np.float32)
   rep[i]=np.multiply(z[i],np.float32(.5),dtype=np.float32)+np.multiply(template,np.float32(.5),dtype=np.float32)
  delta=rep[active].astype(float)-z[active].astype(float);B={'RMS':float(np.sqrt(np.mean(delta*delta))),'L2':float(np.sqrt((delta*delta).sum())),'max_abs':float(abs(delta).max()),'frame_L2_mean':float(np.sqrt((delta*delta).sum(1)).mean())};d=desc[sid+'/'+a+'/'+kind];out={'id':sid,'speaker':r['speaker'],'arm':a,'kind':kind,'primary71':r['guard20_eligible'],'speech_frames':int(active.sum()),'B_own_prototype_change':B,'S_applied_projected_change':d['applied'],'S_preprojection_half_delta':d['half_delta'],'projection':d['projection'],'projection_fraction':d['pre_relu_negative_fraction']};err=max(err,error(out,c));rebuilt.append(out)
 assert len(used)==296 and err<1e-10
 summary={}
 for support in ['all74','common71']:
  for a in 'NT':
   for kind in ['phone','global']:
    cs=[c for c in rebuilt if c['arm']==a and c['kind']==kind and (support=='all74' or c['primary71'])];cell={};names=sorted({c['speaker'] for c in cs})
    for key in ['B_own_prototype_change','S_applied_projected_change','S_preprojection_half_delta','projection']:
     v=np.array([c[key]['RMS'] for c in cs]);means=[np.mean([c[key]['RMS'] for c in cs if c['speaker']==s]) for s in names];cell[key]={'speaker_equal_mean_RMS':float(np.mean(means)),'clip_median_RMS':float(np.median(v)),'clip_min_RMS':float(v.min()),'clip_max_RMS':float(v.max())}
    f=[c['projection_fraction'] for c in cs];cell['projection_fraction']={'clip_median':float(np.median(f)),'min':min(f),'max':max(f)};cell['clips']=len(cs);summary[support+'/'+a+'/'+kind]=cell
 err=max(err,error(summary,x['summaries']));assert err<1e-10
 out=O/'dose_receipt.json';assert not out.exists();out.write_text(json.dumps({'status':'PASS','created_epoch':time.time(),'audit_code_sha256':sha(__file__),'dose_file_sha256':sha(R/'dose_descriptions.json.gz'),'clips':296,'max_error':err,'note':'B own-template displacement reconstructed independently; S/projection references operation descriptions already independently checked by result_receipt','new_scientific_effects':False,'GPU':0},indent=2)+'\n');print(sha(out))
if __name__=='__main__':main()
