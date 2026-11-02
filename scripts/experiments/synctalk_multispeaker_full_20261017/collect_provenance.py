import hashlib
import json
import re
from pathlib import Path
import requests

root=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017')
folder=root/'provenance';folder.mkdir(exist_ok=True)
commit=requests.get('https://api.github.com/repos/MRzzm/HDTF/commits/main',timeout=30).json()['sha']
catalog=requests.get(f'https://api.github.com/repos/MRzzm/HDTF/contents/HDTF_dataset?ref={commit}',timeout=30).json()
files=[];mapping={}
for entry in catalog:
    if entry['type']!='file' or not entry['name'].endswith('.txt'):continue
    payload=requests.get(entry['download_url'],timeout=30).content
    (folder/entry['name']).write_bytes(payload)
    files.append(dict(name=entry['name'],sha256=hashlib.sha256(payload).hexdigest(),source_url=entry['download_url']))
    if entry['name'].endswith('video_url.txt'):
        prefix=entry['name'].split('_')[0]
        for line in payload.decode('utf-8-sig').splitlines():
            match=re.search(r'(https?://\S+)',line)
            if match:
                name=line[:match.start()].strip(' |\t')
                mapping[prefix+'_'+name]=match.group(1).strip('|')
for name in ['README.md','LICENSE']:
    url=f'https://raw.githubusercontent.com/MRzzm/HDTF/{commit}/{name}'
    response=requests.get(url,timeout=30)
    if response.ok:
        (folder/name).write_bytes(response.content)
        files.append(dict(name=name,sha256=hashlib.sha256(response.content).hexdigest(),source_url=url))
rows=[]
for path in sorted((root/'source').glob('*.mp4')):
    origin=re.sub(r'_\d{3}$','',path.stem)
    assert origin in mapping,origin
    url=mapping[origin];video_id=re.search(r'[?&]v=([^&]+)',url).group(1)
    rows.append(dict(file=path.name,original_recording_key=origin,original_video_id=video_id,
                     official_source_url=url,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
receipt=dict(official_repository='https://github.com/MRzzm/HDTF',official_commit=commit,metadata_files=files,sources=rows)
(root/'source_provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(official_commit=commit,sources=rows),indent=2))
