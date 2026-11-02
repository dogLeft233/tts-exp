"""Real-artifact evaluation controls, including historical metric reconstruction."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import soundfile as sf
O=Path('F:/Datasets/tts-exp/may_tts_purified_eval_20261020_v1');B=Path('F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017/remote')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validator():
 s=importlib.util.spec_from_file_location('independent_eval',O/'independent_validate.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_final_checkpoint_and_original_contract():
 c=json.loads((O/'evaluation_contract.json').read_text());t=Path('F:/Datasets/tts-exp/may_tts_purified_training_20261020_v1/remote_artifacts/training/gain_0')
 assert sha(t/'checkpoints/step060000.pt')==c['checkpoint_sha256']
 assert json.loads((t/'training_receipt.json').read_text())['ema_updates']==60
 assert c['testing_gains_db']==[-6,0,6] and c['scored_clips']==[4929,5179,5429,5679] and c['clip_frames']==250
 assert c['same_GT_ROI_from_baseline'] and c['FP32'] and not c['TF32'] and c['guard20_primary']
def test_historical_matrix_reconstruction():
 m=validator();old=json.loads((B/'scores/scores.json').read_text())
 for gain in [-6,0,6]:
  cell=next(x for x in old['cells'] if x['training_gain_db']==0 and x['test_gain_db']==gain)
  label={-6:'gain_m6',0:'gain_0',6:'gain_p6'}[gain]
  for row in cell['clips']:
   root=B/'scores'/row['clip_id'];sub=root/f'train_gain_0_test_{label}'
   matrix=np.load(sub/'distances.npy');rebuilt=m.independent_matrix(np.load(sub/'visual_features.npy'),np.load(root/'original_audio_features.npy'))
   assert np.max(np.abs(matrix-rebuilt))<1e-5
   for guard,key in [(True,'guard20'),(False,'full')]:
    for n,v in m.metric(rebuilt,guard).items():assert abs(v-row[key][n])<1e-5
def test_original_pcm_mutation_rejected():
 m=validator();pcm,sr=sf.read('F:/Datasets/tts-exp/synctalk_gain_training_20261017/splits/test/test_00_original_pcm.wav',dtype='int16');assert sr==16000
 raw=pcm.tobytes();m.verify_pcm(raw,raw)
 changed=bytearray(raw);changed[640]^=1
 try:m.verify_pcm(bytes(changed),raw)
 except m.ContractError:pass
 else:raise AssertionError('changed original PCM accepted')
def test_actual_qualification_passed():
 q=json.loads((O/'qualification_live_readback.json').read_text())['qualification']
 assert q['passed'] and q['baseline_fixed_frames_float_RGB_exact'] and q['new_model_repeat_exact'] and q['codec_pixels_PCM_exact']
 assert q['optimizer_updates']==0 and len(q['frames'])==9
