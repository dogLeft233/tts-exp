"""Independent projected-shift input, cached-cell, distance and complete statistics audit."""
import gzip,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from tts_generator_shift_independent_common_20260927 import rebuild,compare_dict,ah
from tts_fixed_envelope_independent_result_20260927 import distance,endpoints
from tts_mfcc_phase_independent_result_20260927 import stat
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'runs/tts_fixed_generator_shift_cross_20260927';P=ROOT/'runs/tts_fixed_generator_phone_prototype_20260927';O=ROOT/'runs/tts_fixed_generator_shift_independent_audit_20260927'
KINDS=['phone','global'];CELLS=['q00','q10','q01','q11'];CON=['baseline','phoneN','phoneT','globalN','globalT']
def j(p):
 p=Path(p);return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for q in iter(lambda:f.read(2**20),b''):h.update(q)
 return h.hexdigest()
def contrasts(x):
 out={};closure=0.
 for ki,kind in enumerate(KINDS):
  v=x[:,ki];parts={'component_B':v[:,1]-v[:,0],'source_S':v[:,2]-v[:,0],'interaction_I':v[:,3]-v[:,1]-v[:,2]+v[:,0],'total':v[:,3]-v[:,0],'S_after_B':v[:,3]-v[:,1],'B_after_S':v[:,3]-v[:,2]}
  closure=max(closure,float(abs(parts['component_B']+parts['source_S']+parts['interaction_I']-parts['total']).max()))
  for ci,c in enumerate(CELLS):
   for ai,a in enumerate('NT'):out[kind+'/cell/'+c+'/'+a]=v[:,ci,ai]
   out[kind+'/gap/'+c]=v[:,ci,1]-v[:,ci,0]
  for name,vv in parts.items():
   for ai,a in enumerate('NT'):out[kind+'/'+name+'/'+a]=vv[:,ai]
   out[kind+'/'+name+'/TminusN']=vv[:,1]-vv[:,0]
 for k in list(out):
  if k.startswith('phone/'):out['phone_minus_global/'+k[6:]]=out[k]-out['global/'+k[6:]]
 return out,closure

def main():
 torch.set_num_threads(2);p=j(R/'protocol.json');rows=j(P/'rows.json');ev=[r for r in rows if r['split']=='evaluation'];assert len(ev)==74
 assert p['created_epoch']<j(P/'score_lock.json')['created_epoch'];assert j(P/'final.json')['status']=='concluded';assert j(R/'calibration_gate.json')['passed'];lock=j(R/'score_lock.json');assert lock['protocol_sha256']==sha(R/'protocol.json') and lock['features_sha256']==sha(R/'feature_seal_evaluation.json')
 for source in [p['dependencies'],j(R/'parent_binding.json')['files'],j(R/'feature_seal_evaluation.json')]:
  for f,h in source.items():assert sha(f)==h,f
 assert sha(P/'fit.npz')==p['parent_fit_sha256'];assert j(P/'independent_validation.json')['status']=='PASS';fit=np.load(P/'fit.npz');fm=j(P/'fit.json');descs=j(R/'operation_descriptions.json.gz');opsmax=0.;opcount=0
 for r in rows:
  if r['split']!='evaluation' and r['id']!='a1_001':continue
  for a in 'NT':
   for kind in KINDS:
    rep,d=rebuild(r,a,kind,fit,fm);opsmax=max(opsmax,compare_dict(d,descs[r['id']+'/'+a+'/'+kind]));m=j(R/'metadata'/r['id']/a/(kind+'.json'));opsmax=max(opsmax,compare_dict(d,m['mix']));assert ah(rep)==m['video_at_creation']['applied_z_raw_sha256'];assert d['input_z_raw_sha256']==m['video_at_creation']['native_z_raw_sha256']
    for mode,source in [('own',a),('other','T' if a=='N' else 'N')]:assert d[mode+'_z_raw_sha256']==j(P/'metadata'/r['id']/a/(kind+source+'.json'))['mix']['replacement_z_raw_sha256']
    opcount+=1
 assert opcount==300 and opsmax<1e-10
 scores={};vfiles={};curve=0.;nmat=0;newcount=0
 for r in ev:
  sid=r['id'];actual=j(R/'scores'/(sid+'.json.gz'));old=j(P/'scores'/(sid+'.json.gz'))['cells'];scores[sid]={}
  for kind in KINDS:
   scores[sid][kind]={c:{} for c in CELLS}
   for a in 'NT':
    q=r['arms'][a];native=np.load(q['features_path']);aa=native['audio'];L=q['L'];m=j(R/'metadata'/sid/a/(kind+'.json'));v=np.load(m['V']['path']);assert sha(m['V']['path'])==m['V']['sha256'] and ah(v)==m['V']['raw_sha256'] and v.shape==(q['frames']-4,1024) and np.isfinite(v).all();vfiles[sid,a,kind]=v;newcount+=1
    assert m['decode']['frames']==m['video_at_creation']['frames']==q['frames'] and m['decode']['pixel_sha256']==m['video_at_creation']['pixel_sha256'];assert len(m['decode']['PTS'])==q['frames']
    for c in CELLS:
     if c!='q01':
      cond='baseline' if c=='q00' else kind+(a if c=='q10' else ('T' if a=='N' else 'N'));assert actual['cells'][kind][c][a]==old[cond][a];scores[sid][kind][c][a]=old[cond][a]
     else:
      scores[sid][kind][c][a]={}
      for g in ['raw','unit']:
       vv,au=v[:L],aa[:L]
       if g=='unit':vv,au=[(x.astype(float)/np.linalg.norm(x.astype(float),axis=1)[:,None]).astype(np.float32) for x in [vv,au]]
       mat=distance(vv,au);nmat+=mat.size;scores[sid][kind][c][a][g]={}
       for pol in ['guard20','valid','guard0']:
        got=endpoints(mat,pol);expected=actual['cells'][kind][c][a][g][pol]
        if got is None:assert expected is None
        else:
         got['search_uplift']=got['C']-got['C_anchor']
         for k,x in got.items():curve=max(curve,float(abs(np.asarray(x)-np.asarray(expected[k])).max()))
        scores[sid][kind][c][a][g][pol]=got
  print('checked',sid,flush=True)
 assert curve<1e-10 and newcount==296
 idx=j(ROOT/'runs/tts_level_event_cross_20260927/indices.json');oldq=np.load(P/'event_query_metrics.npz')['metrics'];event=np.zeros((1411,2,2,4,2,4));rowmap={r['id']:r for r in rows}
 for row,span in zip(idx['rows'],idx['spans']):
  for ai,a in enumerate('NT'):
   audio=np.load(rowmap[row['id']]['arms'][a]['features_path'])['audio']
   for ki,kind in enumerate(KINDS):
    for ci,c in enumerate(CELLS):
     if c!='q01':
      cond='baseline' if c=='q00' else kind+(a if c=='q10' else ('T' if a=='N' else 'N'));event[span['start']:span['stop'],:,ki,ci,ai]=oldq[span['start']:span['stop'],:,CON.index(cond),ai]
     else:
      for gi in range(2):
       aa=(audio/np.linalg.norm(audio,axis=1)[:,None] if gi else audio).astype(float);v=vfiles[row['id'],a,kind];v=(v/np.linalg.norm(v,axis=1)[:,None] if gi else v).astype(float)
       for qi,q in enumerate(row['queries']):
        n=row['nodes'][q['node']];nodes=[n]+[row['nodes'][k] for k in q['donors']];d=np.linalg.norm(v[n['j'][a]-3]-aa[[n['j'][a] for n in nodes]]+1e-6,axis=1);pos=d[0];neg=d[1:];event[span['start']+qi,gi,ki,ci,ai]=[pos,neg.mean(),neg.mean()-pos,(np.count_nonzero(neg>pos)+.5*np.count_nonzero(neg==pos))/len(neg)]
 qe=float(abs(event-np.load(R/'event_query_metrics.npz')['metrics']).max());assert qe<1e-10;cm=np.array([event[s['start']:s['stop']].mean(0) for s in idx['spans']]);ce=float(abs(cm-np.load(R/'event_clip_metrics.npz')['metrics']).max());assert ce<1e-10
 official=j(R/'summary.json.gz');es=j(R/'event_summary.json.gz');seen=set();eseen=set();statmax=0.;nstat=0;closure=0.
 def check(v,groups,target):
  nonlocal nstat,statmax
  got=stat(v,groups)
  for k,x in got.items():statmax=max(statmax,max(abs(x[s]-target[k][s]) for s in x) if isinstance(x,dict) else float(abs(np.asarray(x)-np.asarray(target[k])).max()))
  nstat+=1
 for g in ['raw','unit']:
  for pol,support in [('guard20','common71'),('valid','common71'),('valid','all74'),('guard0','common71'),('guard0','all74')]:
   use=[r for r in ev if support=='all74' or r['guard20_eligible']];assert len(use)==(74 if support=='all74' else 71)
   for metric in p['metrics']:
    val=np.array([[[[scores[r['id']][kind][c][a][g][pol][metric] for a in 'NT'] for c in CELLS] for kind in KINDS] for r in use]);terms,e=contrasts(val);closure=max(closure,e)
    for key,v in terms.items():k=f'{g}/{pol}/{support}/{metric}/{key}';check(v,[r['speaker'] for r in use],official[k]);seen.add(k)
 for gi,g in enumerate(['raw','unit']):
  for mi,m in enumerate(['positive','negative','margin','rank']):
   terms,e=contrasts(cm[:,gi,:,:,:,mi]);closure=max(closure,e)
   for key,v in terms.items():k=g+'/'+m+'/'+key;check(v,[r['speaker'] for r in idx['spans']],es[k]);eseen.add(k)
 assert set(official)==seen and set(es)==eseen and statmax<1e-10 and closure<1e-10;ana=j(R/'analysis.json');assert ana['natural_phone_source_shift_gain']==(official['raw/guard20/common71/C/phone/source_S/N']['ci99'][0]>0)
 result={'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(R/'protocol.json'),'audit_code_sha256':sha(__file__),'operations':opcount,'operation_description_max':opsmax,'new_V':newcount,'old_cells_and_events_exact':True,'distance_entries':nmat,'curve_max':curve,'event_query_entries':event.size,'event_max':qe,'event_clip_max':ce,'statistics':nstat,'stat_max':statmax,'closure_max':closure,'primary_pass':ana['natural_phone_source_shift_gain'],'media_scope':'8 firstcal physical live files independently decoded;296 eval deleted media checked through producer creation metadata, not independently physically decoded','no_GPU_or_model_forward':True,'hashes':{f:sha(R/f) for f in ['protocol.json','summary.json.gz','event_summary.json.gz','event_query_metrics.npz','event_clip_metrics.npz','analysis.json','feature_seal_evaluation.json','parent_binding.json']}};out=O/'result_receipt.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'receipt':str(out),'sha256':sha(out)}))
if __name__=='__main__':main()
