"""Frozen legacy-donor label composition; cached data only, no fitting or forward."""
import argparse,hashlib,json,math,shutil,time
from pathlib import Path
import numpy as np
from tts_level_event_cross_20260927 import stat
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_fixed_label_composition_20260927';FIELD=ROOT/'runs/tts_fixed_phone_shared_field_20260927';FIX=ROOT/'runs/tts_fixed_level_generation_cross_20260927';SHIFT=ROOT/'runs/tts_fixed_generator_shift_cross_20260927';CAP=8*2**20;FLOOR=int(4.05*2**30);GROUPS=['exact_same','same_after_five_tone_marks_only','different_base'];TRANS=str.maketrans('','','˥˦˧˨˩')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def allocated(r):return sum(p.stat().st_blocks*512 for p in r.rglob('*') if p.is_file())
def resource(extra=0):
 used=allocated(OUT) if OUT.exists() else 0;free=shutil.disk_usage(ROOT).free;other=max(0,144*2**20-allocated(SHIFT))+16*2**20
 assert used+extra<=CAP and free-other-max(0,CAP-used)>=FLOOR,('resource',used,free,other,extra)
 return {'used':used,'free':free,'shift_remaining_including_audit16':other,'own_remaining':max(0,CAP-used),'floor':FLOOR}
def write(path,value):
 data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();resource(math.ceil(len(data)/4096)*4096);assert not path.exists();path.write_bytes(data)
def save(path,**arrays):
 resource(sum(a.nbytes for a in arrays.values())+8192);assert not path.exists();np.savez_compressed(path,**arrays);resource()
def freeze():
 assert not OUT.exists();resource();OUT.mkdir();idx=read(FIELD/'indices.json');groups=[];offsets=[0];counts=[];labels={}
 for row in idx['eval_rows']:
  for q in row['queries']:
   a=row['nodes'][q['node']]['phone'];labels[a]=a.translate(TRANS);gg=[]
   for k in q['donors']:
    b=row['nodes'][k]['phone'];labels[b]=b.translate(TRANS);gg.append(0 if a==b else 1 if a.translate(TRANS)==b.translate(TRANS) else 2)
   groups.extend(gg);offsets.append(len(groups));counts.append(np.bincount(gg,minlength=3))
 assert len(counts)==1411 and len(groups)==20647 and len(idx['eval_rows'])==32 and len({r['speaker'] for r in idx['eval_rows']})==13
 save(OUT/'classification.npz',group=np.array(groups,np.uint8),offsets=np.array(offsets,np.int32),counts=np.array(counts,np.int16));write(OUT/'label_map.json',labels)
 deps={}
 for f in [FIELD/n for n in ['protocol.json','indices.json','fit.npz','fit.json','query_metrics.npz','final.json']]+[Path(__file__).resolve(),ROOT/'scripts/experiments/tts_level_event_cross_20260927.py',FIX/'feature_seal_evaluation.json']:
  deps[str(f)]=sha(f)
 seal=read(FIX/'feature_seal_evaluation.json')
 for row in idx['eval_rows']:
  for a in 'NT':
   f=FIX/'features'/row['id']/a/'FIXED.npz';assert sha(f)==seal[str(f)];deps[str(f)]=sha(f)
 protocol={'status':'FROZEN_BEFORE_NEW_COMPOSITION','created_epoch':time.time(),'dependencies':deps,'classification_sha256':sha(OUT/'classification.npz'),'label_map_sha256':sha(OUT/'label_map.json'),'groups':GROUPS,'delete_only':'˥˦˧˨˩','support':{'clips':32,'speakers':13,'queries':1411,'donor_assignments':20647,'no_filter':True},'conditions':['native_FIXED','frozen_LOSO_Nplus_Tminus'],'geometries':['raw','unit'],'unit':'old float32 norm/division, then float64; field after normalization without renormalization','field':'reuse sealed raw_delta; global+k/(k+4)*(raw_phone-global), missing global; Nplus/Tminus; same field to native A,V; donor own phone','distance':'sqrt(sum((V-A+1e-6)^2)); original k3 query/donor IDs and time indices','contributions':'negative=sum(group_dist)/original_total_donors; margin=(sum(group_dist)-group_count*positive)/original_total_donors; empty0; never group-conditioned renormalization','contrasts':'native,field,field_response each N,T,TminusN; all3 groups negative/margin; absolute positive and response; count/weight descriptions; no rank/new conditions','statistics':'query mean within clip; speaker equal;20000 paired bootstrap seed20260926;99CI each comparison,no FWER; raw/unit all reported; original stat additionally retains95CI/utterance descriptors','gates':{'replay_and_closure_max':1e-10,'same_exact_and_positive_distance_max':1e-10,'same_exact_assignments':512},'boundary':'historical descriptive composition, not71 official C decomposition, tone/F0 causal effect or viseme; secondgroup only same label after deleting five tone marks','resources':{'cap':CAP,'floor':FLOOR,'priority':'S full audit first','RAM_target':256*2**20,'budget_upper':{'arrays':2*2**20,'metadata_report_review':2*2**20,'headroom':4*2**20}},'resource_at_freeze':resource()}
 write(OUT/'protocol.json',protocol);write(OUT/'seal.json',{'protocol_sha256':sha(OUT/'protocol.json')});print(sha(OUT/'protocol.json'))
def run():
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==read(OUT/'seal.json')['protocol_sha256'];review=read(OUT/'reviewer_pass.json');assert review['status']=='PASS' and review['protocol_sha256']==sha(OUT/'protocol.json')
 for f,h in p['dependencies'].items():assert sha(f)==h,f
 assert sha(OUT/'classification.npz')==p['classification_sha256'];idx=read(FIELD/'indices.json');fit=np.load(FIELD/'fit.npz');original=np.load(FIELD/'query_metrics.npz')['metrics'];cl=np.load(OUT/'classification.npz');groups=cl['group'];offsets=cl['offsets'];cnt=cl['counts'];assert int(cnt[:,0].sum())==512
 contrib=np.zeros((1411,2,2,2,3,2));pos=np.zeros((1411,2,2,2));dist=np.zeros((20647,2,2,2));checks={'replay':0.,'negative_margin_closure':0.,'positive_invariance':0.,'same_exact_invariance':0.};li={x:i for i,x in enumerate(idx['labels'])}
 for row,span in zip(idx['eval_rows'],idx['spans']):
  resource();fi=idx['folds'].index(row['speaker']);raw=fit['raw_delta'][fi];k=fit['speaker_counts'][fi]
  for ai,a in enumerate('NT'):
   cache=np.load(FIX/'features'/row['id']/a/'FIXED.npz')
   for gi in range(2):
    aa,vv=[(cache[key] if gi==0 else cache[key]/np.linalg.norm(cache[key],axis=1)[:,None]).astype(np.float64) for key in ['audio','visual']];glob=raw[gi,-1];deltas=glob+(raw[gi,:-1]-glob)*(k/(k+4))[:,None];sign=1 if a=='N' else -1
    def delta(phone):return sign*(deltas[li[phone]] if phone in li else glob)
    for local,q in enumerate(row['queries']):
     qi=span['start']+local;n=row['nodes'][q['node']];don=[row['nodes'][d] for d in q['donors']];av=aa[n['j'][a]];v=vv[n['j'][a]-3];neg=aa[[d['j'][a] for d in don]];sl=slice(offsets[qi],offsets[qi+1]);group=groups[sl];total=len(don);assert total==cnt[qi].sum()
     for si in range(2):
      v1,a1,n1=v,av,neg
      if si:dd=delta(n['phone']);v1=v+dd;a1=av+dd;n1=neg+np.stack([delta(d['phone']) for d in don])
      dp=float(np.sqrt(np.square(v1-a1+1e-6).sum()));dn=np.sqrt(np.square(v1-n1+1e-6).sum(1));dist[sl,gi,si,ai]=dn;pos[qi,gi,si,ai]=dp
      for gr in range(3):
       subtotal=dn[group==gr].sum();contrib[qi,gi,si,ai,gr]=[subtotal/total,(subtotal-int(cnt[qi,gr])*dp)/total]
      expected=original[qi,gi,0 if si==0 else (2 if ai==0 else 1),ai];checks['replay']=max(checks['replay'],float(abs(np.array([dp,dn.mean(),dn.mean()-dp])-expected[:3]).max()));checks['negative_margin_closure']=max(checks['negative_margin_closure'],float(abs(contrib[qi,gi,si,ai].sum(0)-expected[1:3]).max()))
     checks['positive_invariance']=max(checks['positive_invariance'],abs(pos[qi,gi,1,ai]-pos[qi,gi,0,ai]))
     if (group==0).any():checks['same_exact_invariance']=max(checks['same_exact_invariance'],float(abs(dist[sl,gi,1,ai][group==0]-dist[sl,gi,0,ai][group==0]).max()))
  print('decomposed',row['id'],flush=True)
 assert max(checks.values())<=1e-10
 cm=np.array([contrib[s['start']:s['stop']].mean(0) for s in idx['spans']]);pm=np.array([pos[s['start']:s['stop']].mean(0) for s in idx['spans']]);weights=cnt/cnt.sum(1)[:,None];wm=np.array([weights[s['start']:s['stop']].mean(0) for s in idx['spans']]);speakers=[s['speaker'] for s in idx['spans']];summary={}
 for gi,g in enumerate(['raw','unit']):
  for si,state in enumerate(['native','field','field_response']):
   val=cm[:,gi,si] if si<2 else cm[:,gi,1]-cm[:,gi,0];pv=pm[:,gi,si] if si<2 else pm[:,gi,1]-pm[:,gi,0]
   for label,values,positive in [('N',val[:,0],pv[:,0]),('T',val[:,1],pv[:,1]),('TminusN',val[:,1]-val[:,0],pv[:,1]-pv[:,0])]:
    summary[f'{g}/{state}/{label}/positive']=stat(positive,speakers)
    for gr,name in enumerate(GROUPS):
     for mi,metric in enumerate(['negative','margin']):summary[f'{g}/{state}/{label}/{name}/{metric}']=stat(values[:,gr,mi],speakers)
 counts={'assignment_count':cnt.sum(0).tolist(),'assignment_weight':(cnt.sum(0)/cnt.sum()).tolist(),'queries_with_group':(cnt>0).sum(0).tolist(),'query_total':1411,'donor_total':int(cnt.sum()),'speaker_equal_query_weights':{name:stat(wm[:,gr],speakers) for gr,name in enumerate(GROUPS)}}
 save(OUT/'query_arrays.npz',contributions=contrib,positive=pos,negative_distances=dist);save(OUT/'clip_arrays.npz',contributions=cm,positive=pm,weights=wm);write(OUT/'summary.json',summary);write(OUT/'counts.json',counts);write(OUT/'execution.json',{'status':'PENDING_INDEPENDENT','created_epoch':time.time(),'checks':checks,'statistics':len(summary),'same_exact_assignments':512,'resources':resource(),'GPU':0,'fit':False,'hashes':{f:sha(OUT/f) for f in ['query_arrays.npz','clip_arrays.npz','summary.json','counts.json']}});print(json.dumps(checks))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','run']);globals()[ap.parse_args().stage]()
