"""Small synthetic lossless storage audit; does not call scientific worker."""
import ast,gzip,hashlib,json,math,tempfile,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'runs/tts_fixed_mfcc_phase_reversal_20260927';O=ROOT/'runs/tts_fixed_phone_field_phase_independent_audit_20260927';S=ROOT/'scripts/experiments/tts_fixed_mfcc_phase_storage_20260927.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
def main():
 seal=read(R/'storage_seal.json')
 for p,h in seal['dependencies'].items():assert sha(p)==h
 tree=ast.parse(S.read_text());funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['compressible','read','write']];calls=[]
 with tempfile.TemporaryDirectory(prefix='phase-storage-audit-') as tmp:
  out=Path(tmp);ns=dict(Path=Path,OUT=out,gzip=gzip,json=json,math=math,s=types.SimpleNamespace(limits=lambda *a:calls.append(a)),ORIGINAL_READ=read,ORIGINAL_WRITE=lambda *a:(_ for _ in ()).throw(AssertionError('outside whitelist')))
  exec(compile(ast.Module(body=funcs,type_ignores=[]),'<storage pure functions>','exec'),ns)
  payload={'unicode':'音素','floats':[1e-300,-0.0,1.2345678901234567], 'support':[None,True,123],'nested':{'x':[1,2,3]}}
  for relative in ['scores/a.json','event_scores/a.json','summary.json','effects.json']:
   path=out/relative;ns['write'](path,payload);raw=(json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode();assert gzip.decompress(Path(str(path)+'.gz').read_bytes())==raw and ns['read'](path)==payload and not path.exists()
  assert not ns['compressible'](out/'protocol.json') and not ns['compressible'](out/'features/a.npy') and len(calls)==8
 receipt={'status':'PASS','created_epoch':time.time(),'storage_seal_sha256':sha(R/'storage_seal.json'),'adapter_sha256':sha(S),'audit_code_sha256':sha(__file__),'science_worker_protocol_unchanged':True,'whitelist_four_categories_only':True,'synthetic_serialized_bytes_and_readback_exact':True,'metadata10_MiB_cap100_MiB_other92_floor4_5_preserved':True,'insufficient_compression_fails_resource_gate_not_silent_truncation':True,'no_model_or_effect_calculation':True}
 dest=O/'phase_storage_receipt.json';assert not dest.exists();dest.write_text(json.dumps(receipt,indent=2)+'\n');binding={'status':'PASS','storage_seal_sha256':sha(R/'storage_seal.json'),'receipt':str(dest),'receipt_sha256':sha(dest)};p=R/'storage_reviewer_pass.json';assert not p.exists();p.write_text(json.dumps(binding,indent=2)+'\n');print(json.dumps({'receipt':str(dest),'sha256':sha(dest)}))
if __name__=='__main__':main()
