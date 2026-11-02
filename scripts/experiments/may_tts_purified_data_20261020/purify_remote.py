"""Train-only NAS frame purification and independent post-map Sync-C calculation.

No generator/optimizer exists here. All RGB and geometry remain source owned.
"""
from pathlib import Path
import json,sys,os,hashlib,importlib.util,dataclasses,time
import numpy as np
import torch
import cv2
import soundfile as sf
import python_speech_features
from scipy import signal
ROOT=Path(sys.argv[1]);SYNC=Path('/home/wjj/tts-audio/tts-exp/third_party/syncnet_python')
sys.path.insert(0,str(SYNC))
from SyncNetModel import S
from SyncNetInstance import calc_pdist
from detectors import S3FD
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def crop(image,track):
 s,x,y=track;bsi=int(s*1.8);p=np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)),constant_values=110)
 return cv2.resize(p[int(y+bsi-s):int(y+bsi+s*1.8),int(x+bsi-s*1.4):int(x+bsi+s*1.4)],(224,224))
def scores(v,a):
 m=torch.stack(calc_pdist(torch.from_numpy(v),torch.from_numpy(a),vshift=15)).numpy()
 rows=np.arange(20,len(m)-20);assert len(rows)>0
 curve=m[rows].mean(0);index=int(np.argmin(curve))
 # torch.median is the baseline convention, including an odd 31-point curve.
 c=float(torch.median(torch.from_numpy(curve)).item()-curve[index])
 return dict(sync_c=c,av_offset=15-index,windows=len(rows),guard=20),m
def main():
 revision02='--revision02' in sys.argv
 out=ROOT/('results_revision02' if revision02 else 'results');assert not out.exists();out.mkdir()
 os.chdir(SYNC);torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 spec=importlib.util.spec_from_file_location('nas_purify',ROOT/'project_purify_syncdrift.py');nas=importlib.util.module_from_spec(spec);sys.modules[spec.name]=nas;spec.loader.exec_module(nas)
 protocol=json.loads((ROOT/'protocol.json').read_text());aud=json.loads((ROOT/('audio_receipt_revision02.json' if revision02 else 'audio_receipt.json')).read_text());assert aud['passed']
 pins=json.loads((ROOT/'source_image_pins.json').read_text())
 for r in pins:assert sha(ROOT/r['path'])==r['sha256']
 model=S(num_layers_in_fc_layers=1024).cuda().eval();model.load_state_dict(torch.load(SYNC/'data/syncnet_v2.model',map_location='cpu',weights_only=True),strict=True)
 if revision02:
  track=np.load(ROOT/'results/source_owned_track.npy');boxes=np.load(ROOT/'results/source_face_boxes.npy')
 else:
  detector=S3FD(device='cuda');boxes=[]
  for i in range(4200):
   bgr=cv2.imread(str(ROOT/'source_images'/f'{i}.jpg'));assert bgr.shape==(512,512,3)
   rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);f=detector.detect_faces(rgb,conf_th=.9,scales=[.25])
   if len(f)==0:f=detector.detect_faces(rgb,conf_th=.9,scales=[1.])
   assert len(f)>0,('source face missing',i)
   boxes.append(max(f,key=lambda b:(b[2]-b[0])*(b[3]-b[1]))[:4])
   if i%500==0:print(json.dumps(dict(source_face_frame=i)),flush=True)
  del detector;torch.cuda.empty_cache();boxes=np.array(boxes)
  size=np.maximum(boxes[:,3]-boxes[:,1],boxes[:,2]-boxes[:,0])/2;x=(boxes[:,0]+boxes[:,2])/2;y=(boxes[:,1]+boxes[:,3])/2
  track=np.stack([signal.medfilt(z,13) for z in [size,x,y]],1)
 np.save(out/'source_face_boxes.npy',boxes);np.save(out/'source_owned_track.npy',track)
 source=np.empty((4200,224,224,3),dtype=np.uint8)
 for i in range(4200):source[i]=crop(cv2.imread(str(ROOT/'source_images'/f'{i}.jpg')),track[i])
 def visual(frames):
  n=len(frames)-5;rows=[]
  with torch.no_grad():
   for i in range(0,n,20):
    batch=np.stack([frames[j:j+5].transpose(3,0,1,2) for j in range(i,min(i+20,n))]).astype(np.float32)
    rows.append(model.forward_lip(torch.from_numpy(batch).cuda()).cpu().numpy())
  return np.concatenate(rows)
 def audio(y,n):
  mfcc=python_speech_features.mfcc(y*32768.0,16000).T.astype(np.float32);rows=[]
  with torch.no_grad():
   for i in range(0,n,20):
    batch=np.stack([mfcc[:,j*4:j*4+20][None] for j in range(i,min(i+20,n))])
    rows.append(model.forward_aud(torch.from_numpy(batch).cuda()).cpu().numpy())
  return np.concatenate(rows)
 rows=[];allmaps=[];allbase=[];waves=[];offset=0
 for r in aud['chunks']:
  if revision02 and r['id']!=1:continue
  d=out/f"chunk_{r['id']:03d}";d.mkdir();wav=ROOT/('chunks_revision02' if revision02 else 'chunks')/f"{r['id']:03d}"/'tts_16k.wav'
  assert sha(wav)==r['resampled_sha256'];y,sr=sf.read(wav,dtype='float64');assert sr==16000
  n=int(np.ceil(len(y)/640));y=np.pad(y,(0,n*640-len(y)))
  base=np.rint(np.linspace(r['source_start_frame'],r['source_end_frame_exclusive']-1,n)).astype(np.int64)
  before=source[base];af=audio(y,n-5);vf=visual(before)
  eb=nas.Embeddings(audio=af,lip=vf,crop=nas.SyncNetCrop(crop_avi=Path('not_materialized'),track_start_frame=0,track_end_frame=n-1,crop_index=0))
  info=nas.VideoInfo(fps=25,frame_count=n,width=512,height=512,video_duration_sec=n/25,audio_duration_sec=len(y)/16000)
  q=protocol['purification'];est=nas.estimate_correction(eb,info,'window',q['window_sec'],q['stride_sec'],q['search_min_ms'],q['search_max_ms'],q['median_filter_size'],q['min_valid_feats'],q['flat_score_epsilon'])
  mapping=base[est.mapped_indices];assert np.all(np.diff(mapping)>=0) and mapping.min()>=r['source_start_frame'] and mapping.max()<r['source_end_frame_exclusive']<=4200
  after=source[mapping];newvf=visual(after);s0,m0=scores(vf,af);s1,m1=scores(newvf,af)
  np.save(d/'audio_features.npy',af);np.save(d/'before_visual.npy',vf);np.save(d/'after_visual.npy',newvf)
  np.save(d/'before_distances.npy',m0);np.save(d/'after_distances.npy',m1);np.save(d/'search_distances.npy',est.score_matrix)
  np.save(d/'uniform_source_frame_ids.npy',base);np.save(d/'purified_source_frame_ids.npy',mapping)
  np.save(d/'local_correction_ms.npy',est.frame_correction_ms)
  sf.write(d/'tts_frame_aligned.wav',y,16000,subtype='DOUBLE')
  # CSV preserves invalid windows explicitly; do not write JSON NaN or drop rows.
  import csv
  with (d/'window_estimates.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=est.window_rows[0].keys());w.writeheader();w.writerows(est.window_rows)
  delta=np.diff(mapping);row=dict(id=r['id'],source_range=[r['source_start_frame'],r['source_end_frame_exclusive']],tts_frames=n,global_audio_start_frame=offset,before=s0,after=s1,
    passed_sync_c=bool(s1['sync_c']>5),valid_window_ratio=est.valid_window_ratio,repeated_frame_ratio=float(np.mean(delta==0)),skipped_transition_ratio=float(np.mean(delta>1)),unique_source_frames=len(np.unique(mapping)))
  save(d/'receipt.json',row);rows.append(row);allmaps.extend(mapping.tolist());allbase.extend(base.tolist());waves.append(y);offset+=n
  print(json.dumps(row),flush=True)
 joined=np.concatenate(waves);sf.write(out/'aud.wav',joined,16000,subtype='DOUBLE')
 np.save(out/'purified_source_frame_ids.npy',np.array(allmaps,dtype=np.int64));np.save(out/'uniform_source_frame_ids.npy',np.array(allbase,dtype=np.int64))
 mean=float(np.mean([r['after']['sync_c'] for r in rows]));passed=all(r['passed_sync_c'] for r in rows) and mean>5
 save(out/'purification_receipt.json',dict(status='DATA_ACCEPTED' if passed else 'DATA_QC_FAILED',passed=passed,chunks=rows,mean_sync_c=mean,sync_c_threshold_exclusive=5.0,
  every_chunk_required=True,dropped_chunks=[],native_audio_timing=True,normalization=False,local_tts_calls=0,training_jobs=0,
  syncnet_weight_sha256=sha(SYNC/'data/syncnet_v2.model'),nas_source_sha256=sha(ROOT/'project_purify_syncdrift.py'),source_owned_roi=True,precision='FP32_TF32_OFF',audio_units='unclipped float64 samples *32768 for MFCC',output_audio_frames=offset))
if __name__=='__main__':main()
