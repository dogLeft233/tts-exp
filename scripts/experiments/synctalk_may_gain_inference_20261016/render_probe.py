import importlib.util
from pathlib import Path
import json
import soundfile as sf
import numpy as np

for module in ['trimesh','tensorboardX','mcubes','rich','torch_ema','lpips','ninja','scipy','cv2','imageio','torchvision','face_alignment']:
    s=importlib.util.find_spec(module)
    print(module, s.origin if s else None)
r=Path('/home/wjj/tts-audio/synctalk-amplitude-audit-20261016')
x,sr=sf.read(r/'assets/May/aud.wav')
print('May channels equal',np.array_equal(x[:,0],x[:,1]))
for path in [Path('/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01/lib/python3.10/site-packages/nvidia/cuda_runtime/include')]:
    print('CUDA runtime headers', [p.name for p in path.glob('cuda*.h')][:15])
