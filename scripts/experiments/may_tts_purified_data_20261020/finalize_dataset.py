"""Independently rebuild every Sync-C and bind train geometry to native TTS clock."""
from pathlib import Path
import json,hashlib,shutil,copy,csv
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from prepare_audio import OUT,PARENT,BASELINE,sha,save
def independent_sync_c(folder):
 a=np.load(folder/'audio_features.npy').astype(np.float64);v=np.load(folder/'after_visual.npy').astype(np.float64)
 stored=np.load(folder/'after_distances.npy');n=len(v);assert a.shape==v.shape
 # calc_pdist pads the audio embedding in the original integer frame clock.
 padded=np.pad(a,((15,15),(0,0)),constant_values=0.)
 rebuilt=np.stack([np.sqrt(np.sum((v-padded[j:j+n])**2,axis=1)) for j in range(31)],axis=1)
 error=float(np.max(np.abs(rebuilt-stored)))
 assert error<1e-4
 curve=rebuilt[20:n-20].mean(0);c=float(np.median(curve)-curve.min())
 return c,error,stored.size
def main():
 cloud=OUT/'cloud31_v1';dest=OUT/'prepared_dataset_v1';assert not dest.exists();dest.mkdir()
 audio=json.loads((cloud/'audio_receipt_revision02.json').read_text());assert all(r['word_error_rate']==0 for r in audio['chunks'])
 old=OUT/'purification_readback/results';repair=OUT/'purification_readback/results_revision02'
 rows=[];waves=[];maps=[];bases=[];total=0;distances=0;maxerror=0
 for r in audio['chunks']:
  i=r['id'];folder=(repair if i==1 else old)/f'chunk_{i:03d}'
  receipt=json.loads((folder/'receipt.json').read_text());c,err,count=independent_sync_c(folder);distances+=count;maxerror=max(maxerror,err)
  assert abs(c-receipt['after']['sync_c'])<1e-4 and c>5
  expected=cloud/('chunk001_content_repair_v2' if i==1 else f'chunks/{i:03d}')
  raw,sr=sf.read(expected/'tts_native.wav',dtype='float64');assert sr==24000 and sha(expected/'tts_native.wav')==r['native_sha256']
  res,sr=sf.read(expected/'tts_16k.wav',dtype='float64');assert sr==16000 and np.array_equal(res,resample_poly(raw,2,3))
  framewave,sr=sf.read(folder/'tts_frame_aligned.wav',dtype='float64');assert np.array_equal(framewave[:len(res)],res) and np.all(framewave[len(res):]==0)
  mapping=np.load(folder/'purified_source_frame_ids.npy');base=np.load(folder/'uniform_source_frame_ids.npy')
  assert len(mapping)*640==len(framewave) and len(base)==len(mapping)
  assert np.all(np.diff(mapping)>=0) and mapping.min()>=r['source_start_frame'] and mapping.max()<r['source_end_frame_exclusive']<=4200
  rows.append(dict(id=i,before_sync_c=receipt['before']['sync_c'],sync_c=receipt['after']['sync_c'],independent_float64_sync_c=c,passed=True,global_start_frame=total,frames=len(mapping),repeated_frame_ratio=receipt['repeated_frame_ratio'],skipped_transition_ratio=receipt['skipped_transition_ratio'],content_wer=0.0,score_receipt_sha256=sha(folder/'receipt.json')))
  waves.append(framewave);maps.append(mapping);bases.append(base);total+=len(mapping)
 mapping=np.concatenate(maps);base=np.concatenate(bases);y=np.concatenate(waves)
 assert len(mapping)==total and np.all(np.diff(mapping)>=0)
 train=dest/'tts_purified/train';train.mkdir(parents=True);sf.write(train/'aud.wav',y,16000,subtype='DOUBLE')
 back,sr=sf.read(train/'aud.wav',dtype='float64');assert np.array_equal(back,y)
 np.save(dest/'purified_source_frame_ids.npy',mapping);np.save(dest/'uniform_source_frame_ids.npy',base)
 original=json.loads((PARENT/'source/May/transforms_train.json').read_text());frames={r['img_id']:r for r in original['frames']}
 targets=[];uniform=[]
 for aid,source in enumerate(mapping):
  if aid<25 or aid>=total-25 or not 25<=source<4175:continue
  r=copy.deepcopy(frames[int(source)]);r['aud_id']=aid;targets.append(r)
 for aid,source in enumerate(base):
  if aid<25 or aid>=total-25 or not 25<=source<4175:continue
  r=copy.deepcopy(frames[int(source)]);r['aud_id']=aid;uniform.append(r)
 t={k:v for k,v in original.items() if k!='frames'};t.update(h=512,w=512,frames=targets);save(train/'transforms_train.json',t)
 u={k:v for k,v in t.items() if k!='frames'};u['frames']=uniform;save(dest/'uniform_only_transforms_train.json',u)
 # Explicit source table; never pretend that original image ids are new clock ids.
 save(dest/'source_binding.json',dict(source_root=(PARENT/'source/May').as_posix(),image_id='original source id for RGB/landmarks/parsing/torso/masks/eye conditions',audio_id='native TTS 25fps frame index',all_geometry_follows_source_image=True,background=(PARENT/'source/May/bc.jpg').as_posix(),bs=(PARENT/'source/May/bs.npy').as_posix(),no_new_geometry_or_video_compression=True))
 # Link existing verified scientific source folders into an immediately usable data tree.
 for name in ['gt_imgs','ori_imgs','torso_imgs']:
  link=train/name;src=(PARENT/'source/May'/name).resolve()
  import subprocess
  subprocess.run(['powershell','-NoProfile','-Command',f"New-Item -ItemType Junction -Path '{link}' -Target '{src}' | Out-Null"],check=True,capture_output=True)
 for name in ['bc.jpg','bs.npy']:shutil.copyfile(PARENT/'source/May'/name,train/name)
 for split in ['val','test']:
  d=dest/'natural_heldout'/split;d.mkdir(parents=True)
  for name in ['aud.wav',f'transforms_{split}.json']:
   src=OUT/'baseline_binding'/split/name;shutil.copyfile(src,d/name);assert sha(src)==sha(d/name)
 for name in ['initialization.pt','initialization.json','frame_order.npy','protocol.json']:
  p=dest/'baseline_reference';p.mkdir(exist_ok=True);shutil.copyfile(OUT/'baseline_binding'/name,p/name)
 with (dest/'sync_c.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 mean=float(np.mean([r['sync_c'] for r in rows]));assert mean>5
 binding=json.loads((OUT/'baseline_binding/receipt.json').read_text())
 save(dest/'readiness_receipt.json',dict(status='DATA_READY_NO_TRAINING',passed=True,cloud_only=True,local_tts_calls=0,
   asr_model='qwen-audio-3.1-asr-flash',tts_model='qwen-audio-3.1-tts-flash',normalization=False,chunks=rows,
   minimum_sync_c=min(r['sync_c'] for r in rows),mean_sync_c=mean,strict_each_chunk_gt5=True,content_wer_all_zero=True,
   total_audio_frames=total,total_audio_seconds=len(y)/16000,supervised_train_rows=len(targets),original_baseline_train_rows=4150,
   row_count_difference_reason='Native TTS duration retained; data timing/duplication changes are part of treatment, not hidden audio stretching',
   independent_rebuilt_distance_cells=distances,independent_max_distance_error=maxerror,
   initial_model_tensor_sha256=binding['initial_model_tensor_sha256'],baseline_compatible_training=binding['future_training'],
   validation_test_original_exact=True,training_jobs=0,ave_feature_extraction_done=False,
   training_entry_gate='Needs separate AVE cache and loader qualification before any later authorized training',
   limits='NAS optimizes the same SyncNet used by this technical QC; passing C>5 does not prove human synchronization or downstream training benefit'))
 print(json.dumps(dict(passed=True,min_sync_c=min(r['sync_c'] for r in rows),mean_sync_c=mean,training_rows=len(targets),audio_frames=total,distance_cells=distances,maxerror=maxerror)))
if __name__=='__main__':main()
