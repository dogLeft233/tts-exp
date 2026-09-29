"""CPU-only independent live media/input audit of native80 controls."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np,cv2
from tts_generator_phone_independent_live_20260927 import decode,check_meta
from tts_generator_holdout80_independent_common_20260927 import rebuild,ah
from tts_generator_shift_independent_common_20260927 import compare_dict
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'runs/tts_fixed_generator_shift_holdout80_20260927';P=ROOT/'runs/tts_fixed_generator_phone_prototype_20260927';O=ROOT/'runs/tts_fixed_generator_shift_holdout80_independent_audit_20260927';j=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(stage):
 cv2.setNumThreads(1);p=j(R/'protocol.json');c=p['gpu_config'];live=j(R/(stage+'_live.json'));assert live['protocol_sha256']==sha(R/'protocol.json');assert j(O/'input_receipt.json')['input_seal_sha256']==sha(R/'input_seal.json');rows=j(R/'input_rows.json');media=live['files'];assert len(media)==(6 if stage=='identity' else 8);checked={};extra={};err=0.;zfiles={};vfiles={}
 for f,h in media.items():assert sha(f)==h
 if stage=='identity':
  for row in rows[:2]:
   sid=row['id'];q=row['arms']['N'];record=j(R/'identity'/(sid+'.json'));assert record['passed'] and record['parentbindings']==q;z=np.load(record['z']['path']);assert z.dtype==np.float32 and z.shape==(q['frames'],512) and np.isfinite(z).all() and (z>=0).all() and ah(z)==record['z']['raw_sha256'] and sha(record['z']['path'])==record['z']['sha256'];zfiles[sid]=record['z']['sha256'];old=np.load(q['features_path']);om=j(q['metadata_path']);ov=om['video'];assert sha(ov['path'])==ov['sha256'];oldgot=decode(ov['path'],c);assert oldgot['pixel_sha256']==ov['pixel_sha256'];extra[ov['path']]=sha(ov['path'])
   for mode in ['none','noop','cached']:
    m=record['checks'][mode];vm=m['video_at_creation'];path=vm['path'];assert path in media and sha(path)==vm['sha256'];got=decode(path,c);check_meta(m,got);assert got==oldgot;assert vm['native_z_raw_sha256']==vm['applied_z_raw_sha256']==ah(z);assert vm['hook_mode']==mode and vm['hook_calls']==(0 if mode=='none' else (len(z)+31)//32);assert m['A_exact'] and m['V_exact'] and m['pixel_exact'];assert m['A_raw_sha256']==ah(old['audio']) and m['V_raw_sha256']==ah(old['visual']);checked[path]=got
 else:
  assert j(R/'baseline_gate.json')['passed'];row=rows[0];sid=row['id'];q=row['arms']['N'];bm=j(R/'baseline'/(sid+'.json'));z=np.load(bm['z']['path']);assert sha(bm['z']['path'])==bm['z']['sha256'];fit=np.load(P/'fit.npz');fm=j(P/'fit.json')
  for condition in p['conditions']:
   m=j(R/'gpu_controls/calibration'/(condition+'.json'));assert m['parentbindings']==q and m['fold']=='S0912' and m['fit_sha256']==sha(P/'fit.npz');rep,d=rebuild(z,q,condition,fit,fm);err=max(err,compare_dict(d,m['mix']));v=np.load(m['V']['path']);assert v.shape==(q['frames']-4,1024) and np.isfinite(v).all() and sha(m['V']['path'])==m['V']['sha256'] and ah(v)==m['V']['raw_sha256'];vfiles[condition]=m['V']['sha256'];assert m['repeat']['V_exact'] and m['repeat']['pixel_exact'];pair=[]
   for mm in [m,m['repeat']['metadata']]:
    vm=mm['video_at_creation'];path=vm['path'];assert path in media and sha(path)==vm['sha256'];got=decode(path,c);check_meta(mm,got);assert vm['native_z_raw_sha256']==ah(z) and vm['applied_z_raw_sha256']==ah(rep) and vm['hook_mode']=='processed';checked[path]=got;pair.append(got)
   assert pair[0]==pair[1]
 assert set(checked)==set(media) and err<1e-10
 for f,h in media.items():assert sha(f)==h
 out=O/(stage+'_live_receipt.json');assert not out.exists();out.write_text(json.dumps({'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(R/'protocol.json'),'audit_code_sha256':sha(__file__),'helper_sha256':sha(ROOT/'scripts/experiments/tts_generator_holdout80_independent_common_20260927.py'),'media_hashes':media,'old_retained_media_hashes':extra,'decoded':checked,'new_z_files':zfiles,'saved_V_files':vfiles,'mix_max_error':err,'no_GPU_or_model_forward':True,'scope':'physical FFmpeg pixel/PTS/JPEG inputs independent; saved arrays/hash/source checked; neural baseline/repeat/fullstream exact bound producer assertions, no invented historical z'},indent=2)+'\n');rv=R/(stage+'_media_audit.json');assert not rv.exists();rv.write_text(json.dumps({'status':'PASS','receipt':str(out),'receipt_sha256':sha(out)},indent=2)+'\n');print(sha(out))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['identity','calibration']);main(ap.parse_args().stage)
