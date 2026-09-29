"""Pre-effect mathematical/float32, alias and cumulative-timer review; no models."""
import ast,hashlib,json,resource,tempfile,time,types
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'runs/tts_fixed_generator_shift_cross_20260927';O=ROOT/'runs/tts_fixed_generator_shift_independent_audit_20260927';P=ROOT/'runs/tts_fixed_generator_phone_prototype_20260927';j=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for q in iter(lambda:f.read(2**20),b''):h.update(q)
 return h.hexdigest()
def funcs(path,names,ns):
 tree=ast.parse(path.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'<frozen pure audit>','exec'),ns)
def main():
 p=j(R/'protocol.json');assert sha(R/'protocol.json')==j(R/'seal.json')['protocol_sha256'];assert sha(R/'design.md')==j(R/'seal.json')['design_sha256'];assert not (P/'score_lock.json').exists()
 for f,h in p['dependencies'].items():assert sha(f)==h,f
 assert p['parent_fit_sha256']==sha(P/'fit.npz')==j(ROOT/'runs/tts_fixed_generator_phone_prototype_independent_audit_20260927/fit_receipt.json')['fit_sha256'];assert p['resources']['total']==160*2**20 and p['resources']['main']==144*2**20 and p['resources']['audit']==16*2**20 and p['resources']['GPU_cumulative_seconds_including_live']==3600
 oldns={'np':np,'ah':lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()};funcs(ROOT/'scripts/experiments/tts_fixed_generator_phone_stream_20260927.py',{'replacement_table'},oldns);old=types.SimpleNamespace(ah=oldns['ah'],replacement_table=oldns['replacement_table']);ns={'np':np,'old':old};funcs(ROOT/'scripts/experiments/tts_fixed_generator_shift_ops_20260927.py',{'shift_table'},ns)
 rng=np.random.default_rng(84);z=abs(rng.normal(size=(5,512))).astype(np.float32);z[0]=0;labels=['p','q','p',None,'missing'];mask=[True,True,True,False,True];own={'p':np.full(512,4,np.float32),'q':np.full(512,.125,np.float32)};other={'p':np.full(512,.2,np.float32),'q':np.full(512,4.5,np.float32)};og=np.full(512,1.25,np.float32);tg=np.full(512,2.125,np.float32)
 for kind in ['phone','global']:
  got,meta=ns['shift_table'](z,labels,mask,own,og,other,tg,kind);expect=z.copy();neg=0;count=0;realerr=0.;preerr=[];posterr=[];proj=[]
  q10,_=old.replacement_table(z,labels,mask,own,og,kind);q11,_=old.replacement_table(z,labels,mask,other,tg,kind)
  for i in np.flatnonzero(mask):
   b=own.get(labels[i],og) if kind=='phone' else og;a=other.get(labels[i],tg) if kind=='phone' else tg;half=np.multiply(np.subtract(a,b,dtype=np.float32),np.float32(.5),dtype=np.float32);pre=np.add(z[i],half,dtype=np.float32);expect[i]=np.maximum(pre,np.float32(0));neg+=np.count_nonzero(pre<0);count+=512;seq=np.add(q10[i],half,dtype=np.float32);preerr.append(seq.astype(float)-q11[i]);posterr.append(np.maximum(seq,np.float32(0)).astype(float)-q11[i]);proj.append(expect[i].astype(float)-pre)
   z64=z[i].astype(float);a64=a.astype(float);b64=b.astype(float);realerr=max(realerr,float(abs(z64+.5*(b64-z64)+.5*(a64-b64)-(.5*z64+.5*a64)).max()))
  assert realerr<1e-12 and np.array_equal(got,expect) and np.array_equal(got[3],z[3]);assert meta['pre_relu_negative_coordinates']==neg and meta['speech_coordinates']==count;assert meta['own_z_raw_sha256']==old.ah(q10) and meta['other_z_raw_sha256']==old.ah(q11)
  for key,arr in [('rounding_pre',preerr),('rounding_post',posterr),('projection',proj)]:
   arr=np.asarray(arr);assert abs(meta[key]['RMS']-np.sqrt(np.square(arr).mean()))<1e-12 and abs(meta[key]['max_abs']-abs(arr).max())<1e-12
 cpu=ROOT/'scripts/experiments/tts_fixed_generator_shift_cross_20260927.py';cs={'np':np,'KINDS':['phone','global'],'CELLS':['q00','q10','q01','q11']};funcs(cpu,{'contrasts'},cs);x=rng.normal(size=(3,2,4,2));series,err=cs['contrasts'](x);assert err<1e-10 and np.array_equal(series['phone/source_S/N'],x[:,0,2,0]-x[:,0,0,0])
 gpu=ROOT/'scripts/experiments/tts_fixed_generator_shift_gpu_20260927.py';source=gpu.read_text();assert "timer_start_" in source and "write(OUT/'gpu_runtime'/('timer_start_'+stage+'.json')" in source
 with tempfile.TemporaryDirectory(prefix='shift-timer-audit-') as tmp:
  out=Path(tmp);(out/'gpu_runtime').mkdir();ctx={'Path':Path,'OUT':out,'TMP':out/'temp','config':lambda:{'own_cap_bytes':10**9,'other_reserved_bytes':0,'floor_bytes':0,'tmp_cap_bytes':10**9},'allocated':lambda p:0,'shutil':types.SimpleNamespace(disk_usage=lambda p:types.SimpleNamespace(free=10**12)),'time':types.SimpleNamespace(time=lambda:5000.),'read':j,'resource':resource};funcs(gpu,{'limits'},ctx)
  (out/'gpu_runtime/timer_start_calibration.json').write_text('{"created_epoch": 1000}');(out/'gpu_runtime/worker_exit_calibration.json').write_text('{"created_epoch": 1010}');(out/'gpu_runtime/timer_start_evaluation.json').write_text('{"created_epoch": 4980}');assert ctx['limits']()['GPU_cumulative_wall_seconds']==30
  (out/'gpu_runtime/timer_start_evaluation.json').write_text('{"created_epoch": 1000}')
  try:ctx['limits']();raise RuntimeError('over-budget gate failed')
  except AssertionError:pass
 receipt={'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(R/'protocol.json'),'audit_code_sha256':sha(__file__),'dependencies':len(p['dependencies']),'parent_scores_absent_at_review':True,'no_new_fit_forward_or_effects':True,'synthetic_subtract_multiply_add_ReLU_inactive_fallback_and_old_cell_hash_pass':True,'real_algebra_and_float_rounding_separated':True,'synthetic_component_alias_and_closure_pass':True,'timer_v2_prior_stage_plus_live_cumulative_gate_pass':True,'parent_conclusion_independent_receipt_resource_and_full_bridge_required_before_GPU':True,'resource_total160_MiB_audit16_floor4_05_GiB':True}
 dest=O/'static_receipt.json';assert not dest.exists();dest.write_text(json.dumps(receipt,indent=2)+'\n');rv=R/'reviewer_pass.json';assert not rv.exists();rv.write_text(json.dumps({'status':'PASS','protocol_sha256':sha(R/'protocol.json'),'receipt':str(dest),'receipt_sha256':sha(dest)},indent=2)+'\n');print(json.dumps({'receipt':str(dest),'sha256':sha(dest),'review_sha256':sha(rv)}))
if __name__=='__main__':main()
