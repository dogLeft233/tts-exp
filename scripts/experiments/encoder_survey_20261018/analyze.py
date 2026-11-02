"""Compare saved encoder geometry on an explicit shared matched-phone support."""
from pathlib import Path
import json,hashlib,csv,itertools
import numpy as np
from prepare import sha
from run import BASE,unit,write,geometry_summary

def common_support(p,models,domain):
 common={};coverage=[]
 for row in p['rows']:
  per={}
  for spec in models:
   receipt=json.loads((BASE/'models'/spec['id']/'receipt.json').read_text(encoding='utf8'))
   clip=next(r for r in receipt['clips'] if r['id']==row['id'])
   f=BASE/'models'/spec['id']/clip['file'];assert sha(f)==clip['sha256']
   with np.load(f,allow_pickle=False) as z:per[spec['id']]=set(z[domain+'_phone_indices'].tolist())
  ids=sorted(set.intersection(*per.values()))
  if not ids:raise RuntimeError('empty shared support '+row['id'])
  common[row['id']]=ids
  coverage.append({'id':row['id'],'planned_matched_phones':len(row['phones']),
    'common_phones':len(ids),'per_model':{k:len(v) for k,v in per.items()},'common_phone_indices':ids})
 return common,coverage

def bootstrap_speaker_mean(x):
 x=np.asarray(x,dtype=float);rng=np.random.default_rng(20261018)
 vals=x[rng.integers(len(x),size=(20000,len(x)))].mean(1)
 return np.quantile(vals,[.025,.975]).tolist()

def summarize_complete():
 p=json.loads((BASE/'protocol.json').read_text(encoding='utf8'))
 models=[];failures=[]
 candidates=list(p['models'])
 if (BASE/'cached_reference_extension.json').exists():candidates.append(json.loads((BASE/'cached_reference_extension.json').read_text(encoding='utf8'))['model'])
 for spec in candidates:
  f=BASE/'models'/spec['id']/'receipt.json'
  if not f.exists():failures.append({'id':spec['id'],'status':'NOT_STARTED'});continue
  r=json.loads(f.read_text(encoding='utf8'))
  if r['status']!='COMPLETED':failures.append({'id':spec['id'],'status':r['status'],'error':r.get('error')});continue
  assert len(r['clips'])==20 and r['forward_count']==(0 if r.get('cached_reference') else 121)
  models.append(spec)
 if not models:raise RuntimeError('no completed model')
 out=BASE/'analysis';out.mkdir(exist_ok=False)
 results=[];coverage={};paircomparisons=[];sources=[]
 for domain in ['full_phone','inner50']:
  common,cov=common_support(p,models,domain);coverage[domain]=cov
  for spec in models:
   root=BASE/'models'/spec['id'];r=json.loads((root/'receipt.json').read_text(encoding='utf8'))
   native=json.loads((root/'summary_native_support.json').read_text(encoding='utf8'))['results']
   nl=r['layer_count'];selected=sorted(v for v in {0,1,2,nl//2,nl-1} if v<nl)
   clips={};count=0
   for row in p['rows']:
    f=root/(row['id']+'.npz')
    with np.load(f,allow_pickle=False) as z:
     ids=z[domain+'_phone_indices'].tolist();take=[ids.index(k) for k in common[row['id']]]
     clips[row['id']]={li:z[f'{domain}_directions_l{li}'][:,take,:].astype(float) for li in selected}
   endpoint_vectors={}
   for li in selected:
    for ep,(ca,cb) in p['endpoints'].items():
     records=[];angles=[]
     for row in p['rows']:
      d=clips[row['id']][li];delta=d[cb]-d[ca]
      records.append({'speaker':row['speaker'],'id':row['id'],'delta':delta.mean(0),'token_delta':delta})
      angles.append(float(np.degrees(np.arccos(np.clip(np.sum(unit(d[ca])*unit(d[cb]),axis=-1),-1,1))).mean()))
     stat=geometry_summary(records)
     base=next(t for t in native if (t['domain'],t['layer'],t['endpoint'])==(domain,li,ep))
     stat.update(model=spec['id'],repo=spec['repo'],domain=domain,layer=li,
       layer_fraction=li/(nl-1) if nl>1 else 1.0,endpoint=ep,direction_resolved=base['direction_resolved'],
       mean_chord_rms_native=base['mean_chord_rms'],repeat_chord_rms=base['repeat_chord_rms'],
       common_phone_count=sum(len(v) for v in common.values()),mean_angle_deg=float(np.mean(angles)),
       do_normalize=r['preprocessor'].get('do_normalize'),
       native_radius_ratio=base['mean_radius_ratio'])
     if stat['status']=='DESCRIPTIVE':
      # Fold references share training speakers; do not present a naive bootstrap
      # of these dependent fold scores as an independent-sample confidence interval.
      stat['loso_cos_range']=[min(stat['loso_cos_per_speaker']),max(stat['loso_cos_per_speaker'])]
      vectors=np.asarray(stat['speaker_deltas']);local=[];undefined=0
      for si,speaker in enumerate(stat['speaker_ids']):
       axis=unit(np.delete(vectors,si,axis=0).mean(0));sentence=[]
       for rec in records:
        if rec['speaker']!=speaker:continue
        d=rec['token_delta'];valid=np.linalg.norm(d,axis=1)>1e-12;undefined+=int((~valid).sum())
        if valid.any():sentence.append(float(np.mean(unit(d[valid])@axis)))
       local.append(float(np.mean(sentence)) if sentence else None)
      stat['loso_local_phone_cos_per_speaker']=local
      stat['loso_local_phone_cos_mean']=float(np.mean([v for v in local if v is not None])) if any(v is not None for v in local) else None
      stat['undefined_zero_token_deltas']=undefined
      endpoint_vectors[(li,ep)]=np.asarray(stat['speaker_deltas'])
      stat['speaker_delta_norms']=np.linalg.norm(endpoint_vectors[(li,ep)],axis=1).tolist()
      # Per-token NPZs already contain the exact reconstructable primitives.
      # Avoid duplicating millions of vector coordinates in a presentation JSON.
      del stat['speaker_deltas']
     results.append(stat)
    for ep in ['TTS_RAW','TTS_FIXED']:
     if (li,ep) not in endpoint_vectors or (li,'GAIN_P3') not in endpoint_vectors:continue
     a=endpoint_vectors[(li,ep)];b=endpoint_vectors[(li,'GAIN_P3')]
     vals=[float(unit(a[i])@unit(np.delete(b,i,axis=0).mean(0))) for i in range(len(a))]
     gain_resolved=next(t for t in native if (t['domain'],t['layer'],t['endpoint'])==(domain,li,'GAIN_P3'))['direction_resolved']
     paircomparisons.append({'model':spec['id'],'domain':domain,'layer':li,'TTS_endpoint':ep,
       'gain_direction_resolved':gain_resolved,'grand_mean_cos':float(unit(a.mean(0))@unit(b.mean(0))),
       'TTS_to_other_speakers_gain_cos_mean':float(np.mean(vals)),
       'interpretation':'DESCRIPTIVE' if gain_resolved else 'GAIN_BELOW_DIRECTION_RESOLUTION'})
   sources.append({'model':spec['id'],'receipt_sha256':sha(root/'receipt.json'),'native_summary_sha256':sha(root/'summary_native_support.json'),
     'download_source_sha256':sha(root/'model_source.json')})
 write(out/'results.json',{'status':'COMPLETE' if not failures else 'PARTIAL_WITH_EXPLICIT_FAILURES',
   'protocol_sha256':sha(BASE/'protocol.json'),'completed_models':[m['id'] for m in models],'failures':failures,
   'coverage':coverage,'results':results,'gain_TTS_comparisons':paircomparisons,'sources':sources,
   'limits':'model/layer coordinate specific; common support across successful models; unresolved gain cos shown only for numeric audit, never interpreted; CAL20 descriptive; no TFG or perceptual claims'})
 flat=[]
 for t in results:
  flat.append({k:t.get(k) for k in ['model','domain','layer','layer_fraction','endpoint','direction_resolved','mean_chord_rms_native',
   'mean_angle_deg','pairwise_cos_mean','loso_cos_mean','loso_local_phone_cos_mean','common_phone_count','do_normalize','native_radius_ratio']})
 with (out/'table.csv').open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 print(json.dumps({'models':len(models),'failures':failures,'common_phones':{k:sum(t['common_phones'] for t in v) for k,v in coverage.items()},'results_sha256':sha(out/'results.json')}))

if __name__=='__main__':summarize_complete()
