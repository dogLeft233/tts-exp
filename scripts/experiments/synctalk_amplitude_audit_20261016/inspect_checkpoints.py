import torch
from pathlib import Path
r=Path('/home/wjj/tts-audio/synctalk-amplitude-audit-20261016')
x=torch.load(r/'assets/trial_may/checkpoints/ngp.pth',map_location='cpu',weights_only=False)
print('may keys',list(x));print('audio keys',[(k,tuple(v.shape)) for k,v in x['model'].items() if 'audio' in k][:40])
h=torch.load('/dev/shm/synctalk-amplitude-audit-20261016/hubert-large/pytorch_model.bin',map_location='cpu',weights_only=True)
print('hubert pos keys',[(k,tuple(v.shape)) for k,v in h.items() if 'pos_conv' in k])
