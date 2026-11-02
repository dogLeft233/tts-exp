"""Small, within-checkpoint normalization experiment; no old trial is resumed."""
from pathlib import Path
import hashlib,json
OUT=Path('F:/Datasets/tts-exp/may_encoder_normalization_pilot_20261019')
REMOTE='/dev/shm/may-encoder-norm-pilot-20261019'
DATA=Path('F:/Datasets/tts-exp/synctalk_gain_training_20261017')
OLD=Path('F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017/remote')
SURVEY=Path('F:/Datasets/tts-exp/encoder_survey_20261018')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def validate_pair(a,b):
 """A normalization contrast cannot silently compare different encoders or clocks."""
 for key in ['encoder','weight_sha256','config_sha256','pcm_sha256','generator_initial_sha256','steps','frame_order_sha256','ray_schedule_sha256','source_ids','precision','lr_policy','feature_layout']:
  if a[key]!=b[key]:raise ValueError('uncontrolled contrast: '+key)
 if {a['normalize'],b['normalize']}!={True,False}:raise ValueError('contrast must change exactly normalization')
 return True
def build():
 manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
 models=[]
 for mid in ['hubert_base','wav2vec2_base']:
  source=json.loads((SURVEY/'models'/mid/'model_source.json').read_text(encoding='utf-8'))
  models.append(dict(id=mid,source=source,hidden_size=768))
 return dict(schema='may_encoder_norm_pilot_v1',status='FROZEN_BEFORE_NEW_SCIENCE',
  question='Within each frozen encoder, does input waveform normalization affect trainability and held-out generation, including train/inference preprocessing mismatch?',
  scope='May exposed English engineering pilot, one seed; no perceptual, general-population or cross-encoder normalization causal claim',
  official_commit='43079fb3f349675f6cb33031bb07262f7566cdd5',models=models,
  intervention='Only AutoFeatureExtractor.do_normalize True/False; actual native zero-mean/unit-variance epsilon formula; no internal LN/GN/weight changes; normalize whole split before fixed 20s chunks',
  train_norms=[True,False],test_norms=[True,False],train_gain_db=0,test_gains_db=[-6,0,6],
  arms=[dict(encoder=m['id'],norm=n,id=m['id']+('_on' if n else '_off')) for m in models for n in [True,False]],
  seed=42,steps_per_model=12000,num_rays=4096,max_steps_per_ray=16,fp16_training=False,
  learning_rate='original first-stage .5**(step/60000), same first12000 prefix as report50; no accelerated schedule',
  ema_decay=.95,ema_update_interval=1000,checkpoint_interval=2000,validation_interval=2000,validation_frames=16,
  selection='fixed final step12000/EMA12; no test-based selection, no automatic extension or best model selection',
  initialization='one newly created complete random 768-input generator state shared by all four arms; no learned generator warmstart; encoder weights frozen',
  generator='official SyncTalk HuBERT AudioNet path; replace only freshly initialized input module with AudioNet(dim_in=768,dim_aud=32); common within every comparison',
  feature_layout='native50Hz last hidden states paired [25fps,2,768], original provider permute to [25fps,768,2]; no projection/interpolation, supervised rows strictly within support',
  data_manifest_sha256=sha(DATA/'manifest.json'),data_protocol_sha256=sha(DATA/'protocol.json'),
  splits=manifest['splits'],test_clips=[c for c in manifest['eval_clips'] if c['id'] in ['test_00','test_03']],
  evaluation='two prespecified10s clips, first and last original test clips; 24 cells=2encoder x2train_norm x2test_norm x3drivergain; original PCM16/clock, shared original GT ROI, FP32 SyncNet TF32off guard20 primary/full secondary',
  endpoints=['training loss and successful updates','held-out image/mouth-box MSE and L1','Sync-C','Dmin','D0','background median','AV offset','finite-state failures'],
  historical='report50 exact same selected clips, original fixed-final60000 results descriptive only; additionally compare saved AVE12000 checkpoint where rendered; encoder architecture/weights differ, so never attribute cross-encoder delta to normalization',
  controls='same model weights/source/config within on/off; common generator fullinitial, frame-order and actual ray hashes, independent RNG domains, frozen validation rows; duplicate encoder and selected render controls, finite loss/gradient/model/EMA checks and fullstate saves',
  stopping='first scientific nonfinite/job failure seals fullstate/trace and stops all new scientific jobs; no automatic replay/precision change/scaler tune',
  paused_trials='Patrick repeat coarse7 and all previous trials remain paused; no old controller restart',
  persistence='fresh SHA F readback after each stage, retain all scientific files; temporary public weight cache removable only by exact pinned SHA; no old science or shared environment deletion')
if __name__=='__main__':
 assert not (OUT/'protocol.json').exists()
 write(OUT/'protocol.json',build());print(sha(OUT/'protocol.json'))
