"""Create a private official preprocessing copy and convert authorized BFM."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
source=Path('/dev/shm/synctalk-may-gain-inference-20261016/official')
target=ROOT/'official'
assert not target.exists(),'never replace a prepared runtime'
shutil.copytree(source,target)
bfm=target/'data_utils/face_tracking/3DMM'
for p in (ROOT/'assets/3DMM').iterdir():shutil.copy2(p,bfm/p.name)
shutil.copy2(ROOT/'assets/bfm2009/01_MorphableModel.mat',bfm/'01_MorphableModel.mat')
shutil.copy2(ROOT/'assets/face_parsing/79999_iter.pth',target/'data_utils/face_parsing/79999_iter.pth')
landmarker=Path('/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/official/data_utils/blendshape_capture/face_landmarker.task')
shutil.copy2(landmarker,target/'data_utils/blendshape_capture/face_landmarker.task')
import numpy as np
from scipy.io import loadmat
m=loadmat(bfm/'01_MorphableModel.mat')
fields=['shapePC','shapeEV','shapeMU','texPC','texEV','texMU']
assert all(k in m and np.isfinite(m[k]).all() for k in fields)
subprocess.run([sys.executable,'convert_BFM.py'],cwd=bfm.parent,check=True)
converted=np.load(bfm/'3DMM_info.npy',allow_pickle=True).item()
assert all(np.isfinite(v).all() for v in converted.values())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
receipt=dict(passed=True,official_revision='43079fb3f349675f6cb33031bb07262f7566cdd5',
    mat_sha256=sha(bfm/'01_MorphableModel.mat'),converted_sha256=sha(bfm/'3DMM_info.npy'),
    mat_shapes={k:list(m[k].shape) for k in fields},converted_shapes={k:list(v.shape) for k,v in converted.items()},
    dependencies_qualified=False,training_admitted=False)
(ROOT/'assets/bfm2009/conversion_receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
