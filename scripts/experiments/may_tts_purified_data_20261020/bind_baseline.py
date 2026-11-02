"""Freshly pin the report-50 comparator and exact source-owned training assets."""
from pathlib import Path
import json,hashlib,shutil,concurrent.futures
import numpy as np
import soundfile as sf
from prepare_audio import OUT,PARENT,BASELINE,sha,save
def main():
 dest=OUT/'baseline_binding';assert not dest.exists();dest.mkdir()
 p=json.loads((BASELINE/'protocol.json').read_text());init=json.loads((BASELINE/'initialization.json').read_text())
 assert p['official_commit']=='43079fb3f349675f6cb33031bb07262f7566cdd5'
 assert p['audio_encoder'].startswith('official AVE') and not p['fp16_training']
 assert p['steps_per_model']==60000 and p['num_rays']==4096 and p['max_steps_per_ray']==16
 assert sha(BASELINE/'initialization.pt')==init['file_sha256']
 assert sha(BASELINE/'frame_order.npy')==init['frame_order_sha256']
 for name in ['protocol.json','initialization.json','initialization.pt','frame_order.npy']:
  shutil.copyfile(BASELINE/name,dest/name);assert sha(dest/name)==sha(BASELINE/name)
 original,sr=sf.read(PARENT/'source/May/aud.wav',dtype='float64',always_2d=True);assert sr==16000
 splitrows=[]
 for split,a,b,lo,hi in [('train',0,4200,25,4175),('val',4250,4850,4275,4825),('test',4900,6050,4925,6025)]:
  root=PARENT/'arms'/split/'gain_0';audio=root/'aud.wav';data,sr=sf.read(audio,dtype='float64')
  assert sr==16000 and np.array_equal(data,original[a*640:b*640,0])
  src=root/f'transforms_{split}.json';t=json.loads(src.read_text())
  assert [x['img_id'] for x in t['frames']]==list(range(lo,hi))
  assert [x['aud_id'] for x in t['frames']]==list(range(lo-a,hi-a))
  folder=dest/split;folder.mkdir();shutil.copyfile(audio,folder/'aud.wav');shutil.copyfile(src,folder/src.name)
  splitrows.append(dict(split=split,support=[a,b],supervision=[lo,hi],rows=len(t['frames']),audio_sha256=sha(audio),transform_sha256=sha(src),amplitude_exact_original_channel0=True))
 # Verify source-owned geometry/pixels rather than synthesizing new geometry.
 m=json.loads((PARENT/'manifest.json').read_text());selected=[]
 for r in m['source_entries']:
  rel=Path(r['path'])
  if len(rel.parts)<4:continue
  stem=rel.stem.split('_')[0]
  if stem.isdecimal() and 0<=int(stem)<4200:selected.append(r)
 def check(r):
  path=PARENT/r['path'];assert path.stat().st_size==r['bytes'] and sha(path)==r['sha256'],r['path']
  return r
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(check,selected))
 assert len(verified)>=12600
 save(dest/'source_training_asset_pins.json',verified)
 old=Path('scripts/purify_syncdrift.py');history=OUT/'history';history.mkdir(exist_ok=True)
 shutil.copyfile(old,history/'project_purify_syncdrift.py')
 comparison=dict(passed=True,baseline_report=50,natural_control_train_gain_db=0,
  natural_control_test_gain_db=0,reference_sync_c=8.909,reference_not_new_result=True,
  splits=splitrows,verified_source_training_members=len(verified),initial_model_tensor_sha256=init['tensor_state_sha256'],
  initialization_file_sha256=init['file_sha256'],frame_order_sha256=init['frame_order_sha256'],
  future_training=dict(seed=42,successful_updates=60000,precision='FP32',rays=4096,max_steps=16,encoder='same original AVE',input_normalization=False,gain_db=0,loss='same official',optimizer=p['optimizer'],lr_schedule=p['lr_schedule'],ema_decay=p['ema_decay'],ema_update_interval=p['ema_update_interval']),
  heldout_natural_unchanged=True,training_jobs_started=0,tts_ready=False,purification_ready=False,
  source_snapshot_sha256=sha(history/'project_purify_syncdrift.py'))
 save(dest/'receipt.json',comparison);print(json.dumps(dict(passed=True,verified_training_members=len(verified),split_rows=[r['rows'] for r in splitrows])))
if __name__=='__main__':main()
