"""Pre-fit field review: bindings and small synthetic functions, no real scores."""
import ast,hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'runs/tts_fixed_phone_shared_field_20260927';OUT=ROOT/'runs/tts_fixed_phone_field_phase_independent_audit_20260927'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for q in iter(lambda:f.read(2**20),b''):h.update(q)
 return h.hexdigest()
def main():
 p=read(RUN/'protocol.json');source=ROOT/'scripts/experiments/tts_fixed_phone_shared_field_20260927.py'
 for n,h in read(RUN/'seal.json').items():assert sha(RUN/n)==h,n
 for n,h in {**read(RUN/'inputs.json'),**p['code']}.items():assert sha(n)==h,n
 idx=read(RUN/'indices.json');assert len(idx['cal_rows'])==20 and len(idx['eval_rows'])==32
 assert idx['cal_rows']==[r for r in read(ROOT/'runs/tts_shared_translation_20260926/calibration_support.json')['rows'] if r['eligible']]
 assert idx['eval_rows']==read(ROOT/'runs/tts_level_event_cross_20260927/indices.json')['rows']
 tree=ast.parse(source.read_text());fns=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['geom','field_model','score_row']]
 ns={'np':np};exec(compile(ast.Module(body=fns,type_ignores=[]),'<frozen pure field functions>','exec'),ns)
 models=np.zeros((2,2,3,1024));models[0,:,0,0]=3;models[0,:,1,1]=4;models[0,:,-1,0]=1;models[1]=models[0]*2
 counts=np.array([[2,0],[3,1]]);ix={'folds':['held','pooled'],'labels':['p','q']}
 field=ns['field_model'](ix,models,counts,'held',0,{'family':'loso_shrink','sign':-1},'T',None)
 expected=models[0,0,-1]+2/6*(models[0,0,0]-models[0,0,-1]);assert np.array_equal(field('p'),-expected)
 assert np.array_equal(field('q'),-models[0,0,-1]) and np.array_equal(field('unknown'),-models[0,0,-1])
 Q=dict(np.load(RUN/'Q.npz'));rng=np.random.default_rng(20260927)
 for i in range(16):
  assert np.array_equal(Q['permutation'][i],rng.permutation(1024));assert np.array_equal(Q['sign'][i],rng.choice(np.array([-1,1],np.int8),1024))
  z=rng.normal(size=(3,1024));zz=z[:,Q['permutation'][i]]*Q['sign'][i]
  assert np.max(abs(np.sum((z[:,None]-z[None])**2,axis=-1)-np.sum((zz[:,None]-zz[None])**2,axis=-1)))<1e-10
  # Restore RNG used solely to verify frozen Q generation; synthetic z uses separate RNG.
  rng=np.random.default_rng(20260927)
  for _ in range(i+1):rng.permutation(1024);rng.choice(np.array([-1,1],np.int8),1024)
 rng=np.random.default_rng(30);arrays={(a,m):rng.normal(size=(12,1024)) for a in 'NT' for m in ['A','V']}
 row={'nodes':[{'phone':'p','j':{'N':5,'T':5}},{'phone':'p','j':{'N':8,'T':8}},{'phone':'q','j':{'N':10,'T':10}}],'queries':[{'node':0,'donors':[1,2]}]}
 checks={k:0. for k in ['positive_vector','same_phone_vector','same_phone_distance','squared_expansion','odd_even']}
 native=ns['score_row'](row,arrays,'T');got=ns['score_row'](row,arrays,'T',field,checks)
 assert abs(native[0,0]-got[0,0])<1e-10 and max(checks.values())<1e-9
 # Verify actual fit's speaker exclusion/read path statically; real fitting remains untouched.
 fit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fit');txt=ast.unparse(fit)
 assert "idx['cal_rows']" in txt and "r['speaker'] != fold" in txt and "idx['eval_rows']" not in txt
 receipt={'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(RUN/'protocol.json'),'worker_sha256':sha(source),'audit_code_sha256':sha(__file__),
 'input_hashes':len(read(RUN/'inputs.json')),'legacy_cal_eval_support_exact':True,'synthetic_shrink_missing_sign_Q_and_distance_pass':True,
 'synthetic_checks':checks,'LOSO_fit_reads_only_cal_and_excludes_all_held_speaker_clips':True,'reviewed':['query->clip->speaker weights; k distinct cal speakers','frozen exact63 labels and pooled/global/unshrunk sensitivities','paired opposite-sign and main-minus-meanQ contrasts','separate single-arm directional gap changes','phase uses original fitted fields and requires separate immutable binding review','fixed legacy support and no unit renormalization'],
 'real_fit_or_new_score_executed':False}
 dest=OUT/'field_static_receipt.json';assert not dest.exists();dest.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'receipt':str(dest),'sha256':sha(dest)}))
if __name__=='__main__':main()
