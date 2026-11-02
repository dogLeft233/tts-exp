"""Reuse verified Wav2Lip CAL encodings on the exact new shared phone support."""
from pathlib import Path
import json,numpy as np
from prepare import ROOT,sha,digest
from run import BASE,unit,write,geometry_summary

def main():
 p=json.loads((BASE/'protocol.json').read_text(encoding='utf8'))
 cal=ROOT/'runs/recovered_20260930/remote/calibration_20260930'
 sm=json.loads((cal/'calibration_summary.json').read_text(encoding='utf8'))
 out=BASE/'models/wav2lip_cached_ref';out.mkdir(exist_ok=False)
 keys=['natural_z_raw','tts_z_raw','natural_z_fixed','tts_z_fixed','natural_z_plus3db','natural_z_minus3db']
 receipt={'id':'wav2lip_cached_ref','status':'RUNNING','cached_reference':True,'forward_count':0,
  'layer_count':1,'hidden_dimension':512,'repeat_control':'not rerun; cached historical encodings only',
  'preprocessor':{'source':'historical calibration_summary.json','do_normalize':False},
  'source_sha256':sha(__file__),'clips':[],'protocol_sha256':sha(BASE/'protocol.json')}
 records=[]
 for ri,row in enumerate(p['rows']):
  source=cal/'calibration_features'/(row['id']+'.npz')
  assert sha(source)==row['source_npz_sha256']
  with np.load(source,allow_pickle=False) as z:
   seq=[z[k] for k in keys]
   ts=[(z[arm+'_mel_starts'].astype(float)+7.5)/80 for arm in ['natural','tts','natural','tts','natural','natural']]
  for key,z in zip(keys,seq):
   arm,_,mode=key.partition('_z_');assert digest(z)==sm['clips'][row['id']][arm+'_frontend'][mode]['latent_sha256']
  saved={};u=[unit(z) for z in seq]
  if ri==0:
   for ci in range(6):saved['last_frames_'+str(ci)]=seq[ci];saved['times_'+str(ci)]=ts[ci]
  for inner in [False,True]:
   support=[]
   for phone in row['phones']:
    ix=[]
    for ci in range(6):
     lo,hi=phone['tts' if ci in [1,3] else 'natural']
     if inner:lo,hi=lo+.25*(hi-lo),hi-.25*(hi-lo)
     ix.append(np.flatnonzero((ts[ci]>=lo)&(ts[ci]<hi)))
    if all(len(v) for v in ix):support.append((phone,ix))
   domain='inner50' if inner else 'full_phone'
   d=np.stack([np.stack([unit(u[ci][ix[ci]].mean(0)) for _,ix in support]) for ci in range(6)])
   rad=np.stack([[np.linalg.norm(seq[ci][ix[ci]].astype(float),axis=1).mean() for _,ix in support] for ci in range(6)])
   saved[domain+'_phone_indices']=np.array([ph['index'] for ph,_ in support],dtype='int32')
   saved[domain+'_directions_l0']=d.astype('float32');saved[domain+'_radii_l0']=rad.astype('float32')
   for ep,(a,b) in p['endpoints'].items():
    delta=d[b]-d[a];local=u[b]-u[a] if ep.startswith('GAIN') else delta
    records.append({'speaker':row['speaker'],'domain':domain,'layer':0,'endpoint':ep,'delta':delta.mean(0),
     'chord_rms':float(np.sqrt(np.mean(np.sum(local**2,axis=1)))),
     'angle':float(np.degrees(np.arccos(np.clip(np.sum(d[a]*d[b],axis=1),-1,1))).mean()),'radius':float(np.mean(rad[b]/rad[a]))})
  f=out/(row['id']+'.npz');np.savez_compressed(f,**saved)
  receipt['clips'].append({'id':row['id'],'file':f.name,'sha256':sha(f),'original_npz_sha256':sha(source),
   'conditions':[{'samples':row['samples'][k],'pcm_sha256':row['pcm_sha256'][k]} for k in p['conditions']]})
 results=[]
 for domain in ['full_phone','inner50']:
  for ep in p['endpoints']:
   selected=[r for r in records if r['domain']==domain and r['endpoint']==ep]
   s=geometry_summary(selected);chord=float(np.mean([r['chord_rms'] for r in selected]))
   s.update(domain=domain,layer=0,endpoint=ep,mean_chord_rms=chord,repeat_chord_rms=None,
    direction_resolved=chord>=1e-5,mean_angle_deg=float(np.mean([r['angle'] for r in selected])),mean_radius_ratio=float(np.mean([r['radius'] for r in selected])))
   results.append(s)
 write(out/'summary_native_support.json',{'results':results,'cached_reference':True,'no_fresh_repeat':True})
 write(out/'model_source.json',{'source':'historical verified CAL arrays','calibration_summary_sha256':sha(cal/'calibration_summary.json'),
  'wav2lip_weight_sha256':'ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8'})
 receipt['status']='COMPLETED';write(out/'receipt.json',receipt)
 write(BASE/'cached_reference_extension.json',{'model':{'id':'wav2lip_cached_ref','repo':'historical Wav2Lip GAN','kind':'cached'},
  'reason':'same-data native generator reference included in cross-encoder common support; no new encoder execution',
  'source_sha256':sha(__file__),'forward_count':0})
 print(json.dumps({'cached_clips':20,'new_forward_count':0}))
if __name__=='__main__':main()
