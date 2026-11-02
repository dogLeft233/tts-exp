from pathlib import Path
import soundfile as sf,numpy as np
root=Path('/home/wjj/tts-audio/tts-exp')
paths=[Path('/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/official/demo/test.wav')]+[root/'data/data/audio'/f'{i}.wav' for i in [1,2,3,100,101]]
for p in paths:
 if p.is_file():
  x,sr=sf.read(p); print(str(p),sr,x.shape,float(np.max(np.abs(x))),float(np.sqrt(np.mean(x*x))))
