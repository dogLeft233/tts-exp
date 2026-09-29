"""Independent full-sequence MFCC operation, distance and statistics replay (CPU)."""
import hashlib,json,time,re,sys,gzip
from decimal import Decimal,ROUND_CEILING
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import python_speech_features as psf
from tts_fixed_envelope_independent_result_20260927 import distance,endpoints
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'runs/tts_fixed_mfcc_phase_reversal_20260927';O=ROOT/'runs/tts_fixed_phone_field_phase_independent_audit_20260927'
def j(p):
 p=Path(p);return json.loads(p.read_text() if p.exists() else gzip.decompress(Path(str(p)+'.gz').read_bytes()))
ah=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def sha(p):
 p=Path(p)
 if not p.exists():p=Path(str(p)+'.gz')
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def stat(v,groups):
 names=sorted(set(groups));means=np.array([np.mean([x for x,g in zip(v,groups) if g==s]) for s in names]);draws=np.random.default_rng(20260926).integers(len(names),size=(20000,len(names)))
 return {'mean':float(means.mean()),'ci99':np.quantile(means[draws].mean(1),[.005,.995]).tolist(),'n':len(v),'speakers':len(names),'group_means':dict(zip(names,means.tolist()))}
def main():
 torch.set_num_threads(2);p=j(R/'protocol.json');summary=j(R/'summary.json');lock=j(R/'score_lock.json');assert sha(R/'protocol.json')==j(R/'seal.json')['protocol_sha256']
 for f,h in {**p['dependencies'],**p['asset_hashes']}.items():assert sha(f)==h,f
 for name in ['input_seal.json','feature_seal.json']:
  assert sha(R/name)==lock[name.replace('.json','_sha256')]
  for f,h in j(R/name).items():assert sha(f)==h,f
 bind=j(R/'joint_protocol_binding.json');assert bind==lock['joint_protocol'] and bind['created_epoch']<lock['created_epoch'];assert sha(bind['protocol'])==bind['field_protocol_sha256'] and sha(bind['field_design_path'])==bind['field_design_sha256']
 assert j(R/'calibration_controls.json')['passed'] and j(R/'evaluation_controls.json')['passed'];assert j(R/'calibration_controls.json')['time']<j(R/'evaluation_controls.json')['time']<lock['time']
 loaded=[j(R/'runtime'/('loaded_'+split+'.json')) for split in ['calibration','evaluation']]
 assert loaded[0]['state_sha256']==loaded[1]['state_sha256'] and all(z['runtime']==p['runtime'] and z['seed']==20260926 and z['batch']==32 and z['torch_threads']==2 for z in loaded)
 for split in ['calibration','evaluation']:assert j(R/'runtime'/('worker_exit_'+split+'.json'))['complete']
 statsmax=0.;statcount=0;curveerr=0.;eventerr=0.;nmat=0;frontends=0;curves=0;sc={};ev={};exposure={'N':[],'T':[]}
 for row in p['rows']:
  eval_=row['split']=='evaluation'
  if eval_:sc[row['id']]={}
  for arm in 'NT':
   z=row['arms'][arm];op=j(R/'operations'/row['id']/(arm+'.json'));x,sr=sf.read(z['waveform'],dtype='float64');assert sr==16000 and len(x)==z['samples'];mf=np.asarray(psf.mfcc(x*32768,16000),np.float64);assert ah(mf)==op['original_MFCC_sha256'];tokens=z['tokens']
   if z['TextGrid']:
    text=Path(z['TextGrid']).read_text().split('name = "phones"',1)[1].split('item [',1)[0];raw=re.findall(r'intervals \[\d+\]:\s*xmin = ([\d.e+-]+)\s*xmax = ([\d.e+-]+)\s*text = "([^"]*)"',text);assert len(raw)==len(tokens)
    for (lo,hi,label),t in zip(raw,tokens):assert label==t['label'] and int((Decimal(lo)*16000).to_integral_value(rounding=ROUND_CEILING))==t['lo_sample'] and int((Decimal(hi)*16000).to_integral_value(rounding=ROUND_CEILING))==t['hi_sample']
   order=np.arange(len(mf));groups=[];selected=np.zeros(len(mf),bool)
   for t in tokens:
    take=[i for i in range(len(mf)) if t['usable'] and max(0,160*i-1)>=t['lo_sample'] and 160*i+400<=min(t['hi_sample'],len(x))]
    if len(take)>1:assert not selected[take].any();selected[take]=True;order[take]=take[::-1];groups.append({'tier_index':t['tier_index'],'label':t['label'],'frames':take})
   rev=mf.copy();rev[:,1:]=mf[order,1:];back=rev.copy();back[:,1:]=rev[order,1:];assert np.array_equal(back,mf) and np.array_equal(rev[:,0],mf[:,0]);assert groups==op['groups'] and order.tolist()==op['reverse_source_rows'] and ah(rev)==op['reversed_MFCC_sha256']
   n=z['audio_windows'];orig=np.array([mf[4*i:4*i+20].T for i in range(n)],np.float32);new=np.array([rev[4*i:4*i+20].T for i in range(n)],np.float32);assert ah(orig)==op['original_windows_sha256'] and ah(new)==op['new_windows_sha256'];assert np.array_equal(orig,np.load(z['frontend'])['windows'])
   cov=[int(np.count_nonzero(order[4*i:4*i+20]!=np.arange(4*i,4*i+20))) for i in range(n)];assert cov==op['window_changed_column_counts'];exposure[arm].append({'id':row['id'],'target_frames':int(selected.sum()),'total_frames':len(mf),'mean_abs_displacement':float(np.mean(abs(order-np.arange(len(order))))),'shape_L2':float(np.linalg.norm(rev[:,1:]-mf[:,1:]))});assert selected.sum()==op['target_frames']
   old=np.load(z['baseline_features']);v,a=old['visual'],old['audio'];c=j(R/'controls'/row['id']/(arm+'.json'));assert c['baseline_exact'] and c['inverse_exact'] and c['baseline_max']==c['inverse_max']==0 and c['baseline_raw_sha256']==c['inverse_raw_sha256']==ah(a)
   meta=j(R/'metadata'/row['id']/(arm+'.json'));assert meta['operation_sha256']==sha(R/'operations'/row['id']/(arm+'.json')) and meta['control_sha256']==sha(R/'controls'/row['id']/(arm+'.json'));revA=np.load(meta['A']['path']);assert sha(meta['A']['path'])==meta['A']['sha256'] and ah(revA)==meta['A']['raw_sha256'] and revA.shape==a.shape and np.isfinite(revA).all();frontends+=1
   if not eval_:continue
   sc[row['id']][arm]={};L=z['joint_L'];actual=j(R/'scores'/(row['id']+'.json'))
   for geom in ['raw','unit']:
    arrays=[q[:L] for q in [v,a,revA]]
    if geom=='unit':arrays=[(q.astype(float)/np.linalg.norm(q.astype(float),axis=1)[:,None]).astype('float32') for q in arrays]
    sc[row['id']][arm][geom]={}
    for state,aa in [('FIXED',arrays[1]),('REV',arrays[2])]:
     mat=distance(arrays[0],aa);nmat+=mat.size;pols={}
     for policy in ['guard20','valid','guard0']:
      out=endpoints(mat,policy);pols[policy]=out;expected=actual['cells'][arm][geom][state][policy]
      if out is None:assert expected is None;continue
      for k,xv in out.items():curveerr=max(curveerr,float(abs(np.asarray(xv)-np.asarray(expected[k])).max()))
      curves+=1
     sc[row['id']][arm][geom][state]=pols
  print('audited',row['id'],flush=True)
 assert curveerr<1e-10 and frontends==200
 events=j(R/'event_indices.json')['rows'];table={r['id']:r for r in p['rows']};parent=np.load(ROOT/'runs/tts_level_event_cross_20260927/query_metrics.npz')['metrics'];cursor=0;baselineerr=0.
 for r in events:
  arrays={};nr=len(r['queries']);calc=np.zeros((nr,2,2,4,4));prod=j(R/'event_scores'/(r['id']+'.json'))
  for a in 'NT':
   zz=np.load(table[r['id']]['arms'][a]['baseline_features']);arrays[a]=(zz['visual'],zz['audio'],np.load(R/'features'/r['id']/(a+'.npy')))
  for gi,g in enumerate(['raw','unit']):
   f={a:tuple((x/np.linalg.norm(x,axis=1)[:,None] if gi else x).astype(float) for x in xs) for a,xs in arrays.items()}
   for qi,q in enumerate(r['queries']):
    node=r['nodes'][q['node']];dn=[node]+[r['nodes'][k] for k in q['donors']]
    for si,state in enumerate(['FIXED','REV']):
     for ci,cell in enumerate(['NN','NT','TN','TT']):
      ds=np.linalg.norm(f[cell[0]][0][node['j'][cell[0]]-3]-f[cell[1]][si+1][[n['j'][cell[1]] for n in dn]]+1e-6,axis=1);pos=ds[0];neg=ds[1:];vals=[pos,neg.mean(),neg.mean()-pos,(np.count_nonzero(neg>pos)+.5*np.count_nonzero(neg==pos))/len(neg)];calc[qi,gi,si,ci]=vals
      target=prod['queries'][qi]['values'][g][state][cell];eventerr=max(eventerr,max(abs(vv-target[m]) for vv,m in zip(vals,['positive','negative','margin','rank'])))
  baselineerr=max(baselineerr,float(abs(calc[:,:,0]-parent[cursor:cursor+nr,:,12:16]).max()));cursor+=nr;ev[r['id']]=calc.mean(0)
 assert cursor==1411 and eventerr<1e-10 and baselineerr<1e-10
 def check(got,expected):
  nonlocal statsmax,statcount
  for k,v in got.items():
   err=max(abs(v[s]-expected[k][s]) for s in v) if isinstance(v,dict) else float(abs(np.asarray(v)-np.asarray(expected[k])).max());statsmax=max(statsmax,err)
  statcount+=1
 for view,res in summary['views'].items():
  geom,pol,support=view.split('/');rows=[r for r in p['rows'] if r['split']=='evaluation' and (support=='all74' or r['id'] in p['main_ids'])];groups=[r['speaker'] for r in rows]
  for m,expected in res.items():
   vals={state:{a:np.array([sc[r['id']][a][geom][state][pol][m] for r in rows]) for a in 'NT'} for state in ['FIXED','REV']};dn=vals['REV']['N']-vals['FIXED']['N'];dt=vals['REV']['T']-vals['FIXED']['T'];check(stat(vals['FIXED']['T']-vals['FIXED']['N'],groups),expected['baseline_T_minus_N']);check(stat(vals['REV']['T']-vals['REV']['N'],groups),expected['processed_T_minus_N'])
   for a,v in [('N',dn),('T',dt),('T_minus_N',dt-dn)]:check(stat(v,groups),expected['response'][a])
 groups=[r['speaker'] for r in events];weights={'visual_at_N_audio':[-1,0,1,0],'audio_at_N_visual':[-1,1,0,0],'interaction':[1,-1,-1,1],'diagonal':[-1,0,0,1]}
 for gi,g in enumerate(['raw','unit']):
  for mi,m in enumerate(['positive','negative','margin','rank']):
   values=np.array([ev[r['id']][gi,:,:,mi] for r in events]);expected=summary['event'][g][m]
   for name,w in weights.items():
    before=sum(values[:,0,k]*ww for k,ww in enumerate(w));after=sum(values[:,1,k]*ww for k,ww in enumerate(w))
    for key,v in [('baseline',before),('reversed',after),('response',after-before)]:check(stat(v,groups),expected[name][key])
   for a,k in [('N',0),('T',3)]:check(stat(values[:,1,k]-values[:,0,k],groups),expected['native_arm_response'][a])
 assert statsmax<1e-10
 effectmax=0.;effects=j(R/'effects.json')
 for z in effects:
  g,pol,support=z['view'].split('/');v=sc[z['id']];dn=v['N'][g]['REV'][pol][z['metric']]-v['N'][g]['FIXED'][pol][z['metric']];dt=v['T'][g]['REV'][pol][z['metric']]-v['T'][g]['FIXED'][pol][z['metric']];effectmax=max(effectmax,abs(dn-z['N_response']),abs(dt-z['T_response']),abs(dt-dn-z['gap_response']))
 assert effectmax<1e-10
 ci=summary['primary_gap_response']['ci99'];status='GAP_SHRINK_CONFIRMED' if ci[1]<0 else 'GAP_EXPANSION_CONFIRMED' if ci[0]>0 else 'GAP_CHANGE_NOT_CONFIRMED';assert status==summary['status']
 receipt={'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(R/'protocol.json'),'audit_code_sha256':sha(__file__),'frontends_and_control_hashes':frontends,'distance_entries':nmat,'curves':curves,'curve_max':curveerr,'event_query_max':eventerr,'legacy_baseline_max':baselineerr,'stat_endpoints':statcount,'stat_max':statsmax,'clip_effect_rows':len(effects),'clip_effect_max':effectmax,'joint_binding_before_scores':True,'GPU_or_model_forward':False,'primary_status':status,'hashes':{f:sha(R/f) for f in ['summary.json','effects.json','score_lock.json','feature_seal.json','input_seal.json']}}
 dest=O/'phase_result_receipt.json';assert not dest.exists();dest.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'path':str(dest),'sha256':sha(dest)}))
if __name__=='__main__':main()
