"""Minimal pre-forward protocol/support/convex-operation audit, CPU only."""
import ast,hashlib,json,time
from pathlib import Path
from decimal import Decimal
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'runs/tts_fixed_generator_phone_prototype_20260927';O=ROOT/'runs/tts_fixed_generator_phone_prototype_independent_audit_20260927';S=ROOT/'scripts/experiments/tts_fixed_generator_phone_stream_20260927.py'
j=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for q in iter(lambda:f.read(2**20),b''):h.update(q)
 return h.hexdigest()
def main():
 p=j(R/'protocol.json');rows=j(R/'rows.json');feas=j(R/'feasibility.json');old=j(ROOT/'runs/tts_shared_translation_20260926/calibration_support.json');cal=[r for r in old['rows'] if r['eligible']];sp={r['id']:r for r in rows};assert len(rows)==94 and len(cal)==20
 for n,h in j(R/'seal.json').items():assert sha(R/n)==h,n
 for f,h in p['dependencies'].items():assert sha(f)==h,f
 for f,h in feas['bindings'].items():assert sha(f)==h,f
 assert {r['id'] for r in rows if r['split']=='calibration'}=={r['id'] for r in cal};assert len(p['primary_support'])==71 and p['resources']['main']+p['resources']['audit']==320*2**20
 assert j(p['gpu_config']['prerequisite_closure']['path'])['status'] in ['PASS','concluded']
 phase={r['id']:r for r in j(ROOT/'runs/tts_fixed_mfcc_phase_reversal_20260927/protocol.json')['rows']};frames=0
 for row in rows:
  for a,z in row['arms'].items():
   front=np.load(z['frontend_path']);M=front['mel'].shape[1];starts=[];i=0
   while int(i*80/25)+16<=M:starts.append(int(i*80/25));i+=1
   starts.append(M-16);assert starts==z['mel_starts'] and len(starts)==z['frames'];f=np.load(z['features_path']);assert f['z'].shape==(len(starts),512) and (f['z']>=0).all();assert f['visual'].shape==(len(starts)-4,1024)
   labels=[]
   for s in starts:
    t=Decimal(2*s+15)/160;match=[tkn for tkn in phase[row['id']]['arms'][a]['tokens'] if Decimal(tkn['lo_decimal'])<=t<Decimal(tkn['hi_decimal'])];assert len(match)<=1;labels.append(match[0]['label'] if match and match[0]['usable'] else None)
   assert labels==z['phone_labels'] and [x is not None for x in labels]==z['speech_mask']
   if row['split']=='evaluation':frames+=z['frames']
 rebuilt=[];excluded=[]
 for r in cal:
  occ=[];times={a:(np.asarray(sp[r['id']]['arms'][a]['mel_starts'])+7.5)/80 for a in 'NT'}
  for e in r['events']:
   if e['phone']=='__gap__':continue
   ix={a:[i for i,t in enumerate(times[a]) if e['span'][a][0]<=t<e['span'][a][1]] for a in 'NT'};entry={'id':r['id'],'speaker':r['speaker'],'event':e['event'],'phone':e['phone'],'span':e['span'],'indices':ix}
   (occ if all(ix.values()) else excluded).append(entry)
  rebuilt.append(occ)
 assert rebuilt==[r['occurrences'] for r in feas['rows']] and excluded==feas['excluded_cal_occurrences'];assert sum(map(len,rebuilt))==723 and len(excluded)==27
 labels=sorted({e['phone'] for rr in rebuilt for e in rr});assert len(labels)==82
 for held in sorted({r['speaker'] for r in rows if r['split']=='evaluation'}):assert any(r['speaker']!=held and rr for r,rr in zip(cal,rebuilt))
 tree=ast.parse(S.read_text());ns={'np':np,'ah':lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()};fns=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['template_table','replacement_table']];exec(compile(ast.Module(body=fns,type_ignores=[]),'<sealed pure convex functions>','exec'),ns)
 raw=np.zeros((1,2,3,512),np.float64);raw[0,:,0]=2;raw[0,:,1]=7;raw[0,:,-1]=4;count=np.array([[4,0]]);table,glob=ns['template_table'](raw,count,['p','q'],0,0);assert np.array_equal(table['p'],np.full(512,3,np.float32)) and 'q' not in table
 z=np.arange(4*512,dtype=np.float32).reshape(4,512)/100;out,info=ns['replacement_table'](z,['p','q',None,'unknown'],[1,1,0,1],table,glob,'phone');expected=z.copy()
 for i,mu in [(0,table['p']),(1,glob),(3,glob)]:expected[i]=np.multiply(z[i],np.float32(.5),dtype=np.float32)+np.multiply(mu,np.float32(.5),dtype=np.float32)
 assert np.array_equal(out,expected) and np.array_equal(out[2],z[2]) and info['phone_global_fallback_frames']==2
 receipt={'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(R/'protocol.json'),'audit_code_sha256':sha(__file__),'dependency_hashes':len(p['dependencies']),'cal_occurrences_rebuilt':723,'cal_excluded_rebuilt':27,'exact_labels':82,'cal_eval_rows':[20,74],'eval_frames':frames,'LOSO_global_support_nonempty_all15':True,'all_metadata_frame_centers_masks_and_cached_z_shapes_verified':True,'synthetic_convex_shrink_float32_fallback_identity_pass':True,'reviewed':['cal-only frame->occurrence->clip->speaker weights and LOSO allspeaker exclusion','fixed original A and full visual temporal support','none/noop/cached exact live gate beforefit, cal fourcondition/repeat live beforeeval','fixed fourconditions and dual99CI naturalgain rule','official/legacy distinct frozen unit arithmetic and k3','304MiB main plus16MiB audit floor4.25GiB temporary96MiB beforebatch and filecap gates'],'no_new_fit_forward_score':True}
 out=O/'static_receipt.json';assert not out.exists();out.write_text(json.dumps(receipt,indent=2)+'\n');rv=R/'reviewer_pass.json';assert not rv.exists();rv.write_text(json.dumps({'status':'PASS','protocol_sha256':sha(R/'protocol.json'),'receipt':str(out),'receipt_sha256':sha(out)},indent=2)+'\n');print(json.dumps({'receipt':str(out),'sha256':sha(out),'review_sha256':sha(rv)}))
if __name__=='__main__':main()
