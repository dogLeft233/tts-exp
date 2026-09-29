"""Independent cached A/new V distances, event geometry, contrast algebra and CIs."""
import gzip,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from tts_fixed_envelope_independent_result_20260927 import distance,endpoints
from tts_mfcc_phase_independent_result_20260927 import stat
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'runs/tts_fixed_generator_phone_prototype_20260927';O=ROOT/'runs/tts_fixed_generator_phone_prototype_independent_audit_20260927';CON=['baseline','phoneN','phoneT','globalN','globalT']
def j(p):
 p=Path(p);return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for q in iter(lambda:f.read(2**20),b''):h.update(q)
 return h.hexdigest()
ah=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def contrasts(val):
 out={}
 for ci,c in enumerate(CON):
  for ai,a in enumerate('NT'):out['cell/'+c+'/'+a]=val[:,ci,ai];out['response/'+c+'/'+a]=val[:,ci,ai]-val[:,0,ai]
  out['gap/'+c]=val[:,ci,1]-val[:,ci,0];out['gap_change/'+c]=out['gap/'+c]-(val[:,0,1]-val[:,0,0])
 for kind,lo,hi in [('phone',1,2),('global',3,4)]:
  for ai,a in enumerate('NT'):out[kind+'/TminusN_template/'+a]=val[:,hi,ai]-val[:,lo,ai]
  out[kind+'/source_template_interaction']=val[:,hi,1]-val[:,lo,1]-val[:,hi,0]+val[:,lo,0]
 for ai,a in enumerate('NT'):
  for label,pc,gc in [('N',1,3),('T',2,4)]:out['phone_minus_global/template'+label+'/'+a]=val[:,pc,ai]-val[:,gc,ai]
  out['phone_global_template_interaction/'+a]=(val[:,2,ai]-val[:,1,ai])-(val[:,4,ai]-val[:,3,ai])
 return out
def main():
 torch.set_num_threads(2);p=j(R/'protocol.json');rows=j(R/'rows.json');ev=[r for r in rows if r['split']=='evaluation'];assert len(ev)==74;lock=j(R/'score_lock.json');assert lock['protocol_sha256']==sha(R/'protocol.json') and lock['fit_seal_sha256']==sha(R/'fit_seal.json');assert sha(R/'fit.npz')==j(O/'fit_receipt.json')['fit_sha256'];assert j(R/'calibration_gate.json')['passed'] and j(R/'calibration_gate.json')['created_epoch']<lock['created_epoch']
 for f,h in p['dependencies'].items():assert sha(f)==h,f
 assert sha(R/'feature_seal_evaluation.json')==lock['feature_seal_sha256']
 for f,h in j(R/'feature_seal_evaluation.json').items():assert sha(f)==h,f
 live=j(O/'calibration_live_receipt.json');release=j(R/'calibration_release.json');assert live['media_hashes']==release['released'] and live['created_epoch']<release['created_epoch'];fit=np.load(R/'fit.npz');fm=j(R/'fit.json');labels={x:i for i,x in enumerate(fm['labels'])};scores={};vfiles={};nmat=0;curve_max=0.;metadata_n=0
 for row in ev:
  actual=j(R/'scores'/(row['id']+'.json.gz'));scores[row['id']]={c:{} for c in CON};fold=fm['folds'].index(row['speaker'])
  for ai,a in enumerate('NT'):
   z=row['arms'][a];old=np.load(z['features_path']);aa=old['audio'];native=old['z'];L=z['L'];mask=np.asarray(z['speech_mask']);vfiles[row['id'],a]={'baseline':old['visual']}
   for cond in CON[1:]:
    meta=j(R/'metadata'/row['id']/a/(cond+'.json'));mu=fit['raw_mu'][fold,fm['sources'].index(cond[-1])];cnt=fit['speaker_counts'][fold];glob=mu[-1].astype(np.float32);rep=native.copy();fb=0
    for k in np.flatnonzero(mask):
     idx=labels.get(z['phone_labels'][k]);template=glob
     if cond.startswith('phone') and idx is not None and cnt[idx]>0:
      w=np.float64(cnt[idx])/np.float64(cnt[idx]+4);template=(w*mu[idx]+(1-w)*mu[-1]).astype(np.float32)
     elif cond.startswith('phone'):fb+=1
     rep[k]=np.multiply(native[k],np.float32(.5),dtype=np.float32)+np.multiply(template,np.float32(.5),dtype=np.float32)
    assert np.array_equal(rep[~mask],native[~mask]) and ah(rep)==meta['mix']['replacement_z_raw_sha256'];assert fb==meta['mix']['phone_global_fallback_frames'] and ah(native)==meta['mix']['input_z_raw_sha256'];assert meta['fit_sha256']==sha(R/'fit.npz');vm=meta['video_at_creation'];assert vm['native_z_raw_sha256']==ah(native) and vm['applied_z_raw_sha256']==ah(rep);assert vm['pixel_sha256']==meta['decode']['pixel_sha256'] and vm['frames']==meta['decode']['frames']==z['frames'];assert len(meta['decode']['PTS'])==z['frames'];v=np.load(meta['V']['path']);assert sha(meta['V']['path'])==meta['V']['sha256'] and ah(v)==meta['V']['raw_sha256'];assert v.shape==(z['frames']-4,1024);vfiles[row['id'],a][cond]=v;metadata_n+=1
   for cond in CON:
    vv=vfiles[row['id'],a][cond];scores[row['id']][cond][a]={}
    for g in ['raw','unit']:
     v,audio=vv[:L],aa[:L]
     if g=='unit':v,audio=[(x.astype(float)/np.linalg.norm(x.astype(float),axis=1)[:,None]).astype(np.float32) for x in [v,audio]]
     mat=distance(v,audio);nmat+=mat.size;scores[row['id']][cond][a][g]={}
     for pol in ['guard20','valid','guard0']:
      got=endpoints(mat,pol);expect=actual['cells'][cond][a][g][pol]
      if got is None:assert expect is None
      else:
       got['search_uplift']=got['C']-got['C_anchor']
       for k,x in got.items():curve_max=max(curve_max,float(abs(np.asarray(x)-np.asarray(expect[k])).max()))
      scores[row['id']][cond][a][g][pol]=got
  print('checked',row['id'],flush=True)
 assert curve_max<1e-10 and metadata_n==592
 idx=j(ROOT/'runs/tts_level_event_cross_20260927/indices.json');event=np.zeros((1411,2,5,2,4));rowmap={r['id']:r for r in rows}
 for row,span in zip(idx['rows'],idx['spans']):
  for ai,a in enumerate('NT'):
   audio=np.load(rowmap[row['id']]['arms'][a]['features_path'])['audio']
   for gi in range(2):
    aa=(audio/np.linalg.norm(audio,axis=1)[:,None] if gi else audio).astype(float)
    for ci,cond in enumerate(CON):
     v=vfiles[row['id'],a][cond];v=(v/np.linalg.norm(v,axis=1)[:,None] if gi else v).astype(float)
     for qi,q in enumerate(row['queries']):
      n=row['nodes'][q['node']];nodes=[n]+[row['nodes'][k] for k in q['donors']];d=np.linalg.norm(v[n['j'][a]-3]-aa[[n['j'][a] for n in nodes]]+1e-6,axis=1);positive=d[0];negative=d[1:];event[span['start']+qi,gi,ci,ai]=[positive,negative.mean(),negative.mean()-positive,(np.count_nonzero(negative>positive)+.5*np.count_nonzero(negative==positive))/len(negative)]
 qe=float(abs(event-np.load(R/'event_query_metrics.npz')['metrics']).max());assert qe<1e-10;cm=np.array([event[s['start']:s['stop']].mean(0) for s in idx['spans']]);ce=float(abs(cm-np.load(R/'event_clip_metrics.npz')['metrics']).max());assert ce<1e-10
 official=j(R/'summary.json.gz');events=j(R/'event_summary.json.gz');seen=set();eseen=set();statmax=0.;nstat=0
 def check(values,groups,target):
  nonlocal statmax,nstat
  got=stat(values,groups)
  for k,z in got.items():
   err=max(abs(z[s]-target[k][s]) for s in z) if isinstance(z,dict) else float(abs(np.asarray(z)-np.asarray(target[k])).max());statmax=max(statmax,err)
  nstat+=1
 for g in ['raw','unit']:
  for pol in ['guard20','valid','guard0']:
   for support in (['common71'] if pol=='guard20' else ['common71','all74']):
    use=[r for r in ev if support=='all74' or r['id'] in p['primary_support']];groups=[r['speaker'] for r in use]
    for m in p['official_metrics']:
     val=np.array([[[scores[r['id']][c][a][g][pol][m] for a in 'NT'] for c in CON] for r in use])
     for key,values in contrasts(val).items():k=f'{g}/{pol}/{support}/{m}/{key}';check(values,groups,official[k]);seen.add(k)
 for gi,g in enumerate(['raw','unit']):
  for mi,m in enumerate(['positive','negative','margin','rank']):
   for key,v in contrasts(cm[:,gi,:,:,mi]).items():k=f'{g}/{m}/{key}';check(v,[r['speaker'] for r in idx['spans']],events[k]);eseen.add(k)
 assert seen==set(official) and eseen==set(events) and statmax<1e-10
 a=official['raw/guard20/common71/C/phone/TminusN_template/N'];b=official['raw/guard20/common71/C/response/phoneT/N'];criteria={'phoneT_beats_phoneN_99CI':a['ci99'][0]>0,'phoneT_beats_baseline_99CI':b['ci99'][0]>0,'natural_generator_gain_both':a['ci99'][0]>0 and b['ci99'][0]>0};assert criteria==j(R/'analysis.json')['criteria']
 out=O/'result_receipt.json';assert not out.exists();out.write_text(json.dumps({'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(R/'protocol.json'),'audit_code_sha256':sha(__file__),'all592_replacement_and_creation_metadata_verified':True,'distance_entries':nmat,'curve_max':curve_max,'event_query_entries':event.size,'event_max':qe,'event_clip_max':ce,'statistics':nstat,'official_endpoints':len(seen),'event_endpoints':len(eseen),'stat_max':statmax,'criteria':criteria,'media_scope':'creation-time producer FFmpeg metadata for592; only firstcal live media independently redecoded, eval deleted videos not claimed independently redecoded','no_GPU_or_model_forward':True,'hashes':{f:sha(R/f) for f in ['summary.json.gz','event_summary.json.gz','analysis.json','event_query_metrics.npz','event_clip_metrics.npz','score_lock.json','feature_seal_evaluation.json']}},indent=2)+'\n');print(json.dumps({'receipt':str(out),'sha256':sha(out)}))
if __name__=='__main__':main()
