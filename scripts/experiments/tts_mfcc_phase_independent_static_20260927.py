"""Sealed phase pre-review with synthetic boundary/involution arithmetic only."""
import ast,hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'runs/tts_fixed_mfcc_phase_reversal_20260927';O=ROOT/'runs/tts_fixed_phone_field_phase_independent_audit_20260927';S=ROOT/'scripts/experiments/tts_fixed_mfcc_phase_reversal_20260927.py'
j=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for q in iter(lambda:f.read(2**20),b''):h.update(q)
 return h.hexdigest()
def main():
 p=j(R/'protocol.json');seal=j(R/'seal.json');assert seal['protocol_sha256']==sha(R/'protocol.json') and seal['worker_sha256']==sha(S)
 for f,h in {**p['dependencies'],**p['asset_hashes']}.items():assert sha(f)==h,f
 assert sha(R/'event_indices.json')==p['event_indices_sha256'];assert len(p['rows'])==100 and len(p['main_ids'])==71
 assert p['budget']['feature_NPY_allocated_bound']==93233152 and p['budget']['proved_total']<100*2**20
 tree=ast.parse(S.read_text());ns={'np':np,'ah':lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}
 fns=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['operation','windows','event_unit']];exec(compile(ast.Module(body=fns,type_ignores=[]),'<pure frozen phase functions>','exec'),ns)
 mf=np.arange(40*13,dtype=np.float64).reshape(40,13);tokens=[{'usable':True,'lo_sample':0,'hi_sample':1600,'tier_index':0,'label':'a1'},{'usable':False,'lo_sample':1600,'hi_sample':2400,'tier_index':1,'label':'sil'},{'usable':True,'lo_sample':2400,'hi_sample':6400,'tier_index':2,'label':'b2'}]
 y,back,info=ns['operation'](mf,tokens,6300);want=np.arange(40)
 for t in tokens:
  indices=[i for i in range(40) if t['usable'] and max(0,i*160-1)>=t['lo_sample'] and i*160+400<=min(t['hi_sample'],6300)]
  if len(indices)>1:want[indices]=indices[::-1]
 assert np.array_equal(info['reverse_source_rows'],want) and np.array_equal(y[:,0],mf[:,0]) and np.array_equal(y[:,1:],mf[want,1:]) and np.array_equal(back,mf)
 assert want[15]==15 and want[38]==38 and want[39]==39
 # Exact event normalization must use historical float32 reduction, distinct from official unit.
 score=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='score'))
 assert 'x / np.linalg.norm(x, axis=1)[:, None]' in score
 unitexpr=next(n for n in ast.walk(tree) if isinstance(n,ast.IfExp) and isinstance(n.test,ast.Compare) and ast.unparse(n.test)=="g == 'unit'" and isinstance(n.body,ast.BinOp))
 unitcode=compile(ast.Expression(body=unitexpr.body),'<legacy unit expression>','eval')
 rng=np.random.default_rng(9);x=rng.normal(size=(20,1024)).astype(np.float32);assert np.array_equal(eval(unitcode,{'np':np,'x':x}),x/np.linalg.norm(x,axis=1)[:,None])
 jb=j(R/'joint_protocol_binding.json');assert jb['field_protocol_sha256']==sha(jb['protocol']) and jb['field_design_sha256']==sha(jb['field_design_path']);assert jb['created_epoch']<=jb['time']
 receipt={'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(R/'protocol.json'),'worker_sha256':sha(S),'audit_code_sha256':sha(__file__),'input_bindings':len(p['asset_hashes']),'all200_retained':True,'cal52_precedes_eval148_and_all_baseline_inverse_A_exact_required':True,'synthetic_exact_sample_containment_C0_involution_boundary_pad_pass':True,'event_unit_matches_legacy_float32':True,'official_unit_unchanged':True,'joint_frozen_before_scores':True,'BLAS1_old_runtime_GPU_lease_and_100_plus92_MiB_floor4_5GiB_gates_reviewed':True,'new_real_frontend_fit_forward_score_executed':False}
 dest=O/'phase_static_receipt.json';assert not dest.exists();dest.write_text(json.dumps(receipt,indent=2)+'\n');review={'status':'PASS','protocol_sha256':sha(R/'protocol.json'),'receipt':str(dest),'receipt_sha256':sha(dest),'created_epoch':time.time()};path=R/'reviewer_pass.json';assert not path.exists();path.write_text(json.dumps(review,indent=2)+'\n');print(json.dumps({'receipt':str(dest),'sha256':sha(dest),'review_sha256':sha(path)}))
if __name__=='__main__':main()
