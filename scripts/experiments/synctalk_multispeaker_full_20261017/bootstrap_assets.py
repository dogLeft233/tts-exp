import hashlib
import json
from pathlib import Path
import requests

root=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/assets')
root.mkdir(exist_ok=True)
commit=requests.get('https://api.github.com/repos/YudongGuo/AD-NeRF/commits/master',timeout=30).json()['sha']
items=['data_util/face_tracking/3DMM/exp_info.npy','data_util/face_tracking/3DMM/keys_info.npy',
       'data_util/face_tracking/3DMM/sub_mesh.obj','data_util/face_tracking/3DMM/topology_info.npy',
       'data_util/face_parsing/79999_iter.pth']
rows=[]
for item in items:
    url=f'https://raw.githubusercontent.com/YudongGuo/AD-NeRF/{commit}/{item}'
    target=root/('face_parsing' if 'face_parsing' in item else '3DMM')/Path(item).name
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        response=requests.get(url,timeout=(20,120));response.raise_for_status();target.write_bytes(response.content)
    rows.append(dict(path=target.relative_to(root).as_posix(),source_url=url,bytes=target.stat().st_size,
                     sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
(root/'auxiliary_receipt.json').write_text(json.dumps(dict(official_source_commit=commit,files=rows),indent=2)+'\n')
print(json.dumps(rows,indent=2),flush=True)
