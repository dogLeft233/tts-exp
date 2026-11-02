"""Scientific data guards: no heldout leakage or amplitude normalization."""
import importlib.util,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/experiments/may_tts_purified_data_20261020'))
from prepare_audio import stats
def test_actual_chunks_rejoin_exact_original_pcm_without_gain_or_missing_samples():
 import soundfile as sf,json
 from prepare_audio import OUT,PARENT
 p=json.loads((OUT/'protocol.json').read_text());pieces=[];end=0
 for r in p['chunks']:
  assert r['source_start_frame']==end;end=r['source_end_frame_exclusive']
  y,sr=sf.read(OUT/'chunks'/f"{r['id']:03d}"/'natural.wav',dtype='int16');assert sr==16000
  pieces.append(y)
 original,sr=sf.read(PARENT/'source/May/aud.wav',dtype='int16',always_2d=True)
 assert end==4200 and np.array_equal(np.concatenate(pieces),original[:4200*640,0])
def test_heldout_contract_exact_and_disjoint():
 import json
 from prepare_audio import OUT,PARENT,sha
 for split,lo,hi in [('val',4275,4825),('test',4925,6025)]:
  p=OUT/'baseline_binding'/split/f'transforms_{split}.json'
  assert sha(p)==sha(PARENT/'arms'/split/'gain_0'/p.name)
  assert [r['img_id'] for r in json.loads(p.read_text())['frames']]==list(range(lo,hi))
  assert lo>4200
def test_transcription_covers_all_train_only_chunks_and_nonempty():
 import json
 from prepare_audio import OUT
 p=json.loads((OUT/'transcription_manifest.json').read_text())
 assert p['passed'] and len(p['chunks'])==12
 assert all(r['text'].strip() and 0<=r['source_start_frame']<r['source_end_frame_exclusive']<=4200 for r in p['chunks'])
def test_final_native_clock_rows_keep_all_geometry_with_source_image():
 import json
 from prepare_audio import OUT,PARENT
 original=json.loads((PARENT/'source/May/transforms_train.json').read_text())
 source={r['img_id']:r for r in original['frames']}
 folder=OUT/'prepared_dataset_v1'
 mapping=np.load(folder/'purified_source_frame_ids.npy')
 frames=json.loads((folder/'tts_purified/train/transforms_train.json').read_text())['frames']
 assert len(frames)==3849 and np.all(np.diff([r['aud_id'] for r in frames])>0)
 for row in frames:
  assert 25<=row['img_id']<4175 and mapping[row['aud_id']]==row['img_id']
  old=dict(source[row['img_id']]);old['aud_id']=row['aud_id'];assert old==row
def test_final_cloud31_all_chunks_content_and_sync_threshold():
 import json
 from prepare_audio import OUT
 r=json.loads((OUT/'prepared_dataset_v1/readiness_receipt.json').read_text())
 assert r['passed'] and r['cloud_only'] and r['local_tts_calls']==0 and not r['normalization']
 assert r['asr_model']=='qwen-audio-3.1-asr-flash' and r['tts_model']=='qwen-audio-3.1-tts-flash'
 assert len(r['chunks'])==12 and all(x['sync_c']>5 and x['independent_float64_sync_c']>5 and x['content_wer']==0 for x in r['chunks'])
 assert r['training_jobs']==0 and not r['ave_feature_extraction_done']
