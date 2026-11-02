from pathlib import Path
import hashlib,json
OUT=Path('F:/Datasets/tts-exp/may_four_noise_training_20261020_v1')
DATA=Path('F:/Datasets/tts-exp/synctalk_gain_training_20261017')
REMOTE='/home/wjj/tts-audio/may-four-noise-training-20261020-v1'
SSH=['ssh','-b','192.168.50.1','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=12','wjj@192.168.50.2']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial')
 t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
