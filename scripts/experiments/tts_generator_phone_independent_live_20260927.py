"""Read live temporary media independently; CPU decoding only, never model forward."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2];R=ROOT/'runs/tts_fixed_generator_phone_prototype_20260927';O=ROOT/'runs/tts_fixed_generator_phone_prototype_independent_audit_20260927';j=lambda p:json.loads(Path(p).read_text());ah=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for q in iter(lambda:f.read(2**20),b''):h.update(q)
 return h.hexdigest()
def decode(path,c):
 blob=subprocess.check_output([c['ffmpeg'],'-v','error','-threads','1','-i',str(path),'-threads','1','-f','rawvideo','-pix_fmt','bgr24','pipe:1']);assert len(blob)%(224*224*3)==0;n=len(blob)//(224*224*3);pix=hashlib.sha256(blob).hexdigest();del blob
 frames=json.loads(subprocess.check_output([c['ffprobe'],'-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=pts,best_effort_timestamp_time','-of','json',str(path)]))['frames'];assert len(frames)==n;ts=np.array([float(f['best_effort_timestamp_time']) for f in frames]);assert np.allclose(ts,np.arange(n)/25,rtol=0,atol=1e-9)
 data=subprocess.check_output([c['ffmpeg'],'-y','-loglevel','error','-i',str(path),'-threads','1','-f','image2pipe','-vcodec','mjpeg','pipe:1']);offset=0;h=hashlib.sha256();count=0
 while offset<len(data):
  assert data[offset:offset+2]==b'\xff\xd8';end=data.index(b'\xff\xd9',offset+2)+2;im=cv2.imdecode(np.frombuffer(data[offset:end],np.uint8),cv2.IMREAD_COLOR);assert im.shape==(224,224,3) and im.dtype==np.uint8;h.update(im.tobytes());count+=1;offset=end
 assert count==n;return {'frames':n,'pixel_sha256':pix,'JPEG_decoded_sha256':h.hexdigest(),'PTS':[int(f['pts']) for f in frames],'time_sha256':ah(ts)}
def check_meta(meta,got):
 v=meta['video_at_creation'];assert got['frames']==v['frames']==meta['decode']['frames'] and got['pixel_sha256']==v['pixel_sha256']==meta['decode']['pixel_sha256'];assert got['PTS']==meta['decode']['PTS'] and got['time_sha256']==meta['decode']['time_seconds_sha256'];assert meta['native_z_exact'] and meta['full_stream_exact']
def main(stage):
 cv2.setNumThreads(1);p=j(R/'protocol.json');c=p['gpu_config'];live=j(R/(stage+'_live.json'));assert live['protocol_sha256']==sha(R/'protocol.json');first=next(r for r in j(R/'rows.json') if r['id']==c['first_cal']);assert j(O/'static_receipt.json')['protocol_sha256']==sha(R/'protocol.json');media={};records={};savedV={};replacement_checked=0
 for path,h in live['files'].items():assert Path(path).is_file() and sha(path)==h;media[path]=h
 assert len(media)==(6 if stage=='identity' else 16)
 if stage=='calibration':
  fit=np.load(R/'fit.npz');fm=j(R/'fit.json');assert sha(R/'fit.npz')==fm['fit_sha256'];assert j(O/'fit_receipt.json')['fit_sha256']==sha(R/'fit.npz');fold=fm['folds'].index(first['speaker']);labels={v:i for i,v in enumerate(fm['labels'])}
 for arm,z in first['arms'].items():
  assert sha(z['features_path'])==z['features_sha256'];old=np.load(z['features_path']);native=old['z'];oldmeta=j(z['metadata_path']);assert sha(z['metadata_path'])==z['metadata_sha256']
  if stage=='identity':
   ctrl=j(R/'gpu_controls'/arm/'identity.json');v=np.load(ctrl['V']['path']);assert np.array_equal(v,old['visual']) and sha(ctrl['V']['path'])==ctrl['V']['sha256'];savedV[arm]=ctrl['V']['sha256'];oldvideo=oldmeta['video'];assert sha(oldvideo['path'])==oldvideo['sha256'];baseline=decode(oldvideo['path'],c);assert baseline['pixel_sha256']==oldvideo['pixel_sha256']
   for mode in ['none','noop','cached']:
    meta=ctrl['checks'][mode];vm=meta['video_at_creation'];path=vm['path'];assert path in media and media[path]==vm['sha256'];got=decode(path,c);check_meta(meta,got);assert got==baseline;assert vm['native_z_raw_sha256']==vm['applied_z_raw_sha256']==ah(native) and vm['hook_mode']==mode;assert vm['hook_calls']==(0 if mode=='none' else (len(native)+31)//32);assert meta['V_exact'] and meta['pixel_exact'];records[path]=got
  else:
   for cond in ['phoneN','phoneT','globalN','globalT']:
    meta=j(R/'metadata'/first['id']/arm/(cond+'.json'));assert meta['fit_sha256']==sha(R/'fit.npz') and meta['fold']==first['speaker'];mu=fit['raw_mu'][fold,fm['sources'].index(cond[-1])];counts=fit['speaker_counts'][fold];glob=mu[-1].astype(np.float32);repl=native.copy();mask=np.asarray(z['speech_mask']);fallback=0
    for k in np.flatnonzero(mask):
     label=z['phone_labels'][k];idx=labels.get(label);template=glob
     if cond.startswith('phone') and idx is not None and counts[idx]>0:
      w=np.float64(counts[idx])/np.float64(counts[idx]+4);template=(w*mu[idx]+(np.float64(1)-w)*mu[-1]).astype(np.float32)
     elif cond.startswith('phone'):fallback+=1
     repl[k]=np.multiply(native[k],np.float32(.5),dtype=np.float32)+np.multiply(template,np.float32(.5),dtype=np.float32)
    assert np.array_equal(repl[~mask],native[~mask]) and np.isfinite(repl).all() and (repl>=0).all();assert meta['mix']['replacement_z_raw_sha256']==ah(repl) and meta['mix']['phone_global_fallback_frames']==fallback;assert meta['mix']['input_z_raw_sha256']==ah(native)
    v=np.load(meta['V']['path']);assert v.dtype==np.float32 and v.shape==(z['frames']-4,1024) and np.isfinite(v).all() and sha(meta['V']['path'])==meta['V']['sha256'] and ah(v)==meta['V']['raw_sha256'];savedV[arm+'/'+cond]=meta['V']['sha256'];assert meta['repeat']['V_exact'] and meta['repeat']['pixel_exact'];pair=[]
    for m in [meta,meta['repeat']['metadata']]:
     vm=m['video_at_creation'];path=vm['path'];assert path in media and media[path]==vm['sha256'];got=decode(path,c);check_meta(m,got);assert vm['native_z_raw_sha256']==ah(native) and vm['applied_z_raw_sha256']==ah(repl) and vm['hook_mode']=='processed' and vm['hook_calls']==(len(native)+31)//32;pair.append(got);records[path]=got
    assert pair[0]==pair[1];replacement_checked+=1
 assert set(records)==set(media)
 for path,h in media.items():assert sha(path)==h
 receipt={'status':'PASS','created_epoch':time.time(),'protocol_sha256':sha(R/'protocol.json'),'audit_code_sha256':sha(__file__),'stage':stage,'media_hashes':media,'decoded':records,'saved_V_hashes':savedV,'replacement_tables_independently_rebuilt':replacement_checked,'all_new_media_live_at_verification':True,'independent_CPU_FFmpeg_pixels_PTS_JPEG_input':True,'V_scope':'saved identity none V independently equals parent; noop/cached or repeat V exact are producer runtime assertions bound to metadata; all six/sixteen media and JPEG inputs independently equal relevant parent/repeat','GPU_or_neural_forward':False}
 out=O/(stage+'_live_receipt.json');assert not out.exists();out.write_text(json.dumps(receipt,indent=2)+'\n');binding={'status':'PASS','receipt':str(out),'receipt_sha256':sha(out)};dest=R/(stage+'_media_audit.json');assert not dest.exists();dest.write_text(json.dumps(binding,indent=2)+'\n');print(json.dumps({'receipt':str(out),'sha256':sha(out)}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['identity','calibration']);main(ap.parse_args().stage)
