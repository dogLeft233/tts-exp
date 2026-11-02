import json,hashlib,shutil
from pathlib import Path
r=Path('/home/wjj/tts-audio/synctalk-amplitude-audit-20261016')
dst=Path('/dev/shm/synctalk-amplitude-audit-20261016/transport')
dst.mkdir(exist_ok=True)
expected=json.loads((r/'transport_cleanup_expected.json').read_text(encoding='utf-8-sig'))
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
records=[]
for name,meta in expected.items():
 assert name in ['source.bundle','trial_may.zip','may_subset.tar.gz']
 p=r/name
 assert p.resolve().parent==r.resolve()
 assert p.stat().st_size==meta['bytes'] and digest(p)==meta['sha256']
 q=dst/name
 shutil.copy2(p,q)
 assert digest(q)==meta['sha256']
 p.unlink()
 records.append(dict(meta,source=str(p),moved_to=str(q),verified=True))
(r/'transport_cleanup_receipt.json').write_text(json.dumps(records,indent=2))
print('Reclaimed task transport bytes',sum(x['bytes'] for x in records))
