import json,soundfile as sf,numpy as np,librosa,hashlib
from pathlib import Path
r=Path('/home/wjj/tts-audio/synctalk-amplitude-audit-20261016')
x,sr=sf.read(r/'assets/May/aud.wav')
y=librosa.load(r/'assets/May/aud.wav',sr=16000)[0]
print('May stereo peak',np.max(np.abs(x)),'mono peak',np.max(np.abs(y)),'mono RMS',np.sqrt(np.mean(y.astype(float)**2)))
for k in ['train','val']:
 t=json.loads((r/f'assets/May/transforms_{k}.json').read_text());print('split',k,len(t['frames']),t['frames'][0]['img_id'],t['frames'][-1]['img_id'])
for p in [r/'assets/trial_may/checkpoints/ngp.pth',Path('/dev/shm/synctalk-amplitude-audit-20261016/hubert-large/pytorch_model.bin')]:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 print(p,p.stat().st_size,h.hexdigest())
