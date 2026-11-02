"""Download a bounded CUDA compiler toolchain from NVIDIA's public channel."""
import hashlib
import json
from pathlib import Path
import requests

out=Path('runs/synctalk_may_gain_inference_20261016/resources')
out.mkdir(parents=True,exist_ok=True)
files=['cuda-nvcc-12.1.105-0.tar.bz2','cuda-cudart-dev-12.1.105-0.tar.bz2',
       'cuda-cccl-12.1.109-0.tar.bz2','cuda-driver-dev-12.1.105-0.tar.bz2']
rows=[]
for name in files:
    url='https://conda.anaconda.org/nvidia/linux-64/'+name
    r=requests.get(url,stream=True,timeout=90);r.raise_for_status()
    p=out/name;h=hashlib.sha256();size=0
    with p.open('wb') as f:
        for b in r.iter_content(1048576):f.write(b);h.update(b);size+=len(b)
    rows.append(dict(url=url,path=str(p.resolve()),bytes=size,sha256=h.hexdigest()))
    print(name,size,flush=True)
(out/'toolchain_download_manifest.json').write_text(json.dumps(rows,indent=2))
