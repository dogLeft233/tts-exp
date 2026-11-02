"""Small, within-checkpoint normalization experiment; no old trial is resumed."""
from pathlib import Path
import hashlib,json
OUT=Path('F:/Datasets/tts-exp/may_postnorm_scale_20261020')
REMOTE='/dev/shm/may-postnorm-scale-20261020'
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
 return json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
