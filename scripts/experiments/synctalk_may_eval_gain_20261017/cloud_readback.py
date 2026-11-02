"""Independent Linux-mounted cloud archive readback, never extracts to root disk."""
import hashlib
import json
import tarfile
import time
from pathlib import Path

root=Path('/dev/shm/synctalk-may-eval-gain-cloud-readback-20261017')
expected=json.loads((root/'archive_receipt.json').read_text())
cloud=Path('/home/wjj/百度网盘/tts-exp-research-models/synctalk-may-eval-gain-20261017')/expected['archive']
deadline=time.monotonic()+240
while not cloud.exists():
    if time.monotonic()>deadline:raise TimeoutError('cloud archive not visible')
    time.sleep(5)
h=hashlib.sha256();count=0
with cloud.open('rb') as stream:
    for block in iter(lambda:stream.read(4*1024**2),b''):
        h.update(block);count+=len(block)
assert count==expected['bytes'] and h.hexdigest()==expected['sha256']
with tarfile.open(cloud,'r:gz') as tar:
    manifest=json.load(tar.extractfile('delivery_manifest.json'))
    checked=0
    for entry in manifest['files']:
        payload=tar.extractfile('remote/'+entry['path']).read()
        assert len(payload)==entry['bytes'] and hashlib.sha256(payload).hexdigest()==entry['sha256']
        checked+=1
    assert checked==262
    report=tar.extractfile('metadata/51-synctalk-may-eval-gain.md').read()
    assert b'820260' in report
receipt=dict(passed=True,cloud_path=str(cloud),bytes=count,sha256=h.hexdigest(),
             raw_members_verified=checked,report_included=True,readback_host='wjj@192.168.50.2',
             method='independent Linux mount full stream SHA256 and tar member SHA256; no extraction')
(root/'cloud_readback_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False),flush=True)
