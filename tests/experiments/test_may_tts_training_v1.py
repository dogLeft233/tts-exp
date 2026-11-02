"""Admission checks against real artifacts; no synthetic result fixtures."""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
O=Path('F:/Datasets/tts-exp/may_tts_purified_training_20261020_v1')
B=Path('F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def test_baseline_optimization_unchanged():
 p=json.loads((O/'protocol.json').read_text());b=json.loads((B/'protocol.json').read_text())
 for k in ['seed','steps_per_model','num_rays','max_steps_per_ray','fp16_training','optimizer','criterion','lr_schedule','ema_decay','ema_update_interval','finetune_lips','patch_size','validation_interval','validation_frames','validation_selection','validation_gains_db','checkpoint_interval','log_interval']:
  assert p[k]==b[k],k
 assert p['training_gains_db']==[0] and p['input_normalization'] is False
 assert sha(O/'initialization.pt')==sha(B/'initialization.pt')==p['baseline_initialization_file_sha256']
def test_new_frame_order_domain():
 order=np.load(O/'qualification_readback/frame_order.npy');rng=np.random.default_rng(42)
 assert np.array_equal(order,np.concatenate([rng.permutation(3849) for _ in range(16)])[:60000].astype(np.int32))
 assert order.min()==0 and order.max()==3848
def test_real_cache_and_loader_receipts():
 d=O/'qualification_readback';q=json.loads((d/'qualification/receipt.json').read_text());f=json.loads((d/'features/receipt.json').read_text())
 assert q['passed'] and q['successful_entry_updates']==2 and q['skip']==0 and len(q['provider_checks'])==36
 assert all(x['pixels_exact'] and x['pose_exact'] and x['feature_exact'] for x in q['provider_checks'])
 assert f['passed'] and all(x['repeat_exact'] and x['no_waveform_normalization'] for x in f['rows'])
 val=np.load(d/'features/val_padded.npy')
 assert np.array_equal(val,np.load('F:/Datasets/tts-exp/synctalk_gain_training_20261017/features_attempt03/val/gain_0/ave_padded.npy'))
 assert np.load(d/'features/train_aligned.npy').shape==(3849,1,512)
def test_unchanged_official_model_loss_renderer():
 d=O/'qualification_readback/official/nerf_triplane'
 for name,digest in {'network.py':'0a1fb0f2e6c8e217329c5a64dfa3903ac580525ba12780b6effa69bca98aa16f','utils.py':'b449f5ce08e3c9178d69ec9e4923a8cc65cc3aac066aed68b8cdda6edf9306c2','renderer.py':'5bd3507b312a50d5ec4ec4022a05bcc3386b7ab8de82ea4150c03979d216c3ea','provider.py':'1398acaa33d609853f2fe822f099e14645fce83131372a93661f91999b3c04a3'}.items():assert sha(d/name)==digest
def test_backup_containment():
 p=Path('scripts/experiments/may_tts_purified_training_20261020/host_backup.py');s=importlib.util.spec_from_file_location('backup_qa',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 assert m.safe_relative('training/gain_0/checkpoints/step002000.pt').as_posix()=='training/gain_0/checkpoints/step002000.pt'
 for name in ['../old.pt','/tmp/a','C:/old.pt','training\\old.pt']:
  try:m.safe_relative(name)
  except AssertionError:pass
  else:raise AssertionError(name)
