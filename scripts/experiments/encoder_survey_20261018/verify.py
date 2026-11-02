"""Data-only independent check of saved common-support geometry; no model calls."""
from pathlib import Path
import json,hashlib,numpy as np

BASE=Path('F:/Datasets/tts-exp/encoder_survey_20261018')
def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def u(v):
 v=np.asarray(v,dtype=float);n=np.sqrt((v*v).sum(axis=-1,keepdims=True));assert (n>1e-12).all();return v/n
def main():
 p=json.loads((BASE/'protocol.json').read_text(encoding='utf8'))
 result=json.loads((BASE/'analysis/results.json').read_text(encoding='utf8'))
 errors=[];maxnorm=0.;maxstat=0.;checks=0;artifacts=[]
 for mid in result['completed_models']:
  root=BASE/'models'/mid;r=json.loads((root/'receipt.json').read_text(encoding='utf8'))
  assert r['status']=='COMPLETED' and r['forward_count']==(0 if r.get('cached_reference') else 121) and len(r['clips'])==20
  if not r.get('cached_reference'):
   assert not r['loading_info']['missing_keys'] and not r['loading_info']['mismatched_keys']
   assert max(r['repeat_max_abs'])<1e-5
  cov={d:{x['id']:x['common_phone_indices'] for x in rows} for d,rows in result['coverage'].items()}
  data={}
  for row in p['rows']:
   rec=next(x for x in r['clips'] if x['id']==row['id']);f=root/rec['file'];assert h(f)==rec['sha256']
   with np.load(f,allow_pickle=False) as z:data[row['id']]={k:z[k] for k in z.files}
   for key,x in data[row['id']].items():
    assert np.isfinite(x).all()
    if '_directions_' in key:
     err=float(abs(np.sqrt((x.astype(float)**2).sum(-1))-1).max());maxnorm=max(maxnorm,err);assert err<2e-7
    if '_radii_' in key:assert (x>0).all()
   assert [x['samples'] for x in rec['conditions']]==[row['samples'][k] for k in p['conditions']]
   assert [x['pcm_sha256'] for x in rec['conditions']]==[row['pcm_sha256'][k] for k in p['conditions']]
   artifacts.append({'model':mid,'clip':row['id'],'sha256':rec['sha256']})
  for t in [x for x in result['results'] if x['model']==mid]:
   ca,cb=p['endpoints'][t['endpoint']];clip=[];tokens=[]
   for row in p['rows']:
    d=data[row['id']];ids=d[t['domain']+'_phone_indices'].tolist();sel=[ids.index(i) for i in cov[t['domain']][row['id']]]
    a=d[f"{t['domain']}_directions_l{t['layer']}"][:,sel,:].astype(float)
    clip.append((row['speaker'],(a[cb]-a[ca]).mean(axis=0)))
    tokens.append((row['speaker'],a[cb]-a[ca]))
   speakers=sorted({s for s,_ in clip});v=np.stack([np.mean([a for s,a in clip if s==w],axis=0) for w in speakers])
   if t['status']=='DEGENERATE_SPEAKER_DELTA':continue
   pair=[];loso=[]
   for i in range(10):
    rest=np.sum([v[j] for j in range(10) if j!=i],axis=0)/9
    loso.append(float(np.dot(u(v[i]),u(rest))))
    for j in range(i):pair.append(float(np.dot(u(v[i]),u(v[j]))))
   err=max(abs(np.mean(pair)-t['pairwise_cos_mean']),abs(np.mean(loso)-t['loso_cos_mean']),float(abs(np.sqrt((v*v).sum(1))-np.asarray(t['speaker_delta_norms'])).max()))
   maxstat=max(maxstat,err);assert err<1e-10;checks+=1
   local=[]
   for si,s in enumerate(speakers):
    axis=u(np.mean([v[k] for k in range(10) if k!=si],axis=0));sent=[]
    for speaker,delta in tokens:
     if speaker!=s:continue
     valid=np.sqrt((delta*delta).sum(1))>1e-12
     if valid.any():sent.append(float(np.mean([np.dot(u(x),axis) for x in delta[valid]])))
    if sent:local.append(float(np.mean(sent)))
   if local:
    err=abs(np.mean(local)-t['loso_local_phone_cos_mean']);maxstat=max(maxstat,err);assert err<1e-10
 qa={'status':'PASS_DATA_ONLY','models':len(result['completed_models']),'files':len(artifacts),
  'common_support_stat_checks':checks,'max_direction_norm_error':maxnorm,'max_stat_error':maxstat,
  'result_sha256':h(BASE/'analysis/results.json'),'protocol_sha256':h(BASE/'protocol.json'),
  'verifier_source_sha256':h(__file__),'no_new_encoder_forward':True,'artifacts':artifacts}
 (BASE/'analysis/qa.json').write_text(json.dumps(qa,indent=2),encoding='utf8')
 print(json.dumps({k:v for k,v in qa.items() if k!='artifacts'}))
if __name__=='__main__':main()
