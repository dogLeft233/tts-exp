"""Download user-authorized BFM09; credentials only from process environment."""
import hashlib
import json
import os
import tarfile
from pathlib import Path
import requests

root=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/assets/bfm2009')
root.mkdir(parents=True,exist_ok=True)
target=root/'BaselFaceModel.tgz'
assert not target.exists()
url='https://faces.dmi.unibas.ch/bfm/content/basel_face_model/downloads/restricted/BaselFaceModel.tgz'
hasher=hashlib.sha256();total=0
with requests.get(url,auth=(os.environ['BFM_DOWNLOAD_USER'],os.environ['BFM_DOWNLOAD_PASSWORD']),
                  stream=True,timeout=(20,120)) as response:
    response.raise_for_status()
    with target.with_suffix('.partial').open('xb') as stream:
        for block in response.iter_content(1024**2):
            stream.write(block);hasher.update(block);total+=len(block)
target.with_suffix('.partial').replace(target)
with tarfile.open(target,'r:gz') as archive:
    member=next(x for x in archive.getmembers() if Path(x.name).name=='01_MorphableModel.mat')
    payload=archive.extractfile(member).read()
    (root/'01_MorphableModel.mat').write_bytes(payload)
    license_members=[]
    for item in archive.getmembers():
        if item.isfile() and any(k in Path(item.name).name.lower() for k in ['license','readme','copyright']):
            (root/Path(item.name).name).write_bytes(archive.extractfile(item).read());license_members.append(item.name)
receipt=dict(passed=True,source_url=url,authorization='user supplied official emailed download credentials; not persisted',
    archive_bytes=total,archive_sha256=hasher.hexdigest(),model_bytes=len(payload),
    model_sha256=hashlib.sha256(payload).hexdigest(),license_members=license_members)
(root/'download_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
