"""Independent firstcal eight live videos; CPU FFmpeg/JPEG and float32 source audit."""
import hashlib,json,time
from pathlib import Path
import numpy as np,cv2
from tts_generator_phone_independent_live_20260927 import decode,check_meta
from tts_generator_shift_independent_common_20260927 import rebuild,compare_dict,ah
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'runs/tts_fixed_generator_shift_cross_20260927';P=ROOT/'runs/tts_fixed_generator_phone_prototype_20260927';O=ROOT/'runs/tts_fixed_generator_shift_independent_audit_20260927';j=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 cv2.setNumThreads(1);p=j(R/'protocol.json');c=p['gpu_config'];live=j(R/'calibration_live.json');assert live['protocol_sha256']==sha(R/'protocol.json');assert j(P/'final.json')['status']=='concluded' and j(P/'resource_closure.json')['passed'];binding=j(R/'parent_binding.json')
 for f,h in binding['files'].items():assert sha(f)==h,f
 assert j(R/'bridge_validation.json')['status']=='PASS';fit=np.load(P/'fit.npz');fm=j(P/'fit.json');assert sha(P/'fit.npz')==p['parent_fit_sha256'];first=next(r for r in j(P/'rows.json') if r['id']=='a1_001');files=live['files'];assert len(files)==8;checked={};saved={};err=0.
 for a in 'NT':
  for kind in ['phone','global']:
   m=j(R/'metadata'/first['id']/a/(kind+'.json'));rep,desc=rebuild(first,a,kind,fit,fm);err=max(err,compare_dict(desc,m['mix']));assert err<1e-10;v=np.load(m['V']['path']);assert sha(m['V']['path'])==m['V']['sha256'] and ah(v)==m['V']['raw_sha256'] and v.shape==(first['arms'][a]['frames']-4,1024) and np.isfinite(v).all();saved[a+'/'+kind]=m['V']['sha256'];assert m['repeat']['V_exact'] and m['repeat']['pixel_exact'];pair=[]
   for meta in [m,m['repeat']['metadata']]:
    vm=meta['video_at_creation'];path=vm['path'];assert path in files and files[path]==vm['sha256']==sha(path);got=decode(path,c);check_meta(meta,got);assert vm['native_z_raw_sha256']==desc['input_z_raw_sha256'] and vm['applied_z_raw_sha256']==ah(rep);checked[path]=got;pair.append(got)
   assert pair[0]==pair[1]
 assert set(files)==set(checked)
 for path,h in files.items():assert sha(path)==h
 out=O/'calibration_live_receipt.json';assert not out.exists();out.write_text(json.dumps({'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(R/'protocol.json'),'audit_code_sha256':sha(__file__),'helper_sha256':sha(ROOT/'scripts/experiments/tts_generator_shift_independent_common_20260927.py'),'media_hashes':files,'decoded':checked,'saved_V_hashes':saved,'all4_shift_and_oldcell_rebuild_max':err,'no_GPU_or_model_forward':True,'V_scope':'saved four V arrays finite/hash/shape independently checked; repeat/full-stream V exact bound producer assertions; all8 physical pixel/PTS/JPEG repeats independently exact','parent_complete_before_new_GPU':True},indent=2)+'\n');rv=R/'calibration_media_audit.json';assert not rv.exists();rv.write_text(json.dumps({'status':'PASS','receipt':str(out),'receipt_sha256':sha(out)},indent=2)+'\n');print(json.dumps({'receipt':str(out),'sha256':sha(out)}))
if __name__=='__main__':main()
