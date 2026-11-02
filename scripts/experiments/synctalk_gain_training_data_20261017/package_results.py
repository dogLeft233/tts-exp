"""Package derived data and verification; source archive is separately sealed."""
import hashlib
import json
import tarfile
from pathlib import Path

root = Path('/dev/shm/synctalk-gain-training-data-20261017')
archive = root / 'prepared_data_verified.tar.gz'
assert not archive.exists()
names = ['features_attempt03', 'independent_qa.json', 'source_clock_qa.json',
         'source_media_probe.json', 'cloud_storage_receipt.json', 'provider_attempt01.log',
         'provider_attempt02.log', 'provider_attempt03.log',
         'dataset/manifest.json', 'dataset/protocol.json', 'dataset/arms', 'dataset/splits',
         'dataset/verification_previews']
with tarfile.open(archive, 'w:gz', dereference=False, compresslevel=3) as t:
    for name in names:
        t.add(root / name, arcname=name)
h = hashlib.sha256()
with archive.open('rb') as f:
    for b in iter(lambda: f.read(4 * 1024**2), b''):
        h.update(b)
receipt = {'file': archive.name, 'bytes': archive.stat().st_size, 'sha256': h.hexdigest(),
           'source_archive_required': 'May.zip',
           'source_sha256': '008fe3fb46430a92beff334e26f37eab229c717ff5b577dfbba90f46c3848d4b',
           'includes_source_images': False, 'training_runs': 0}
(root / 'bundle_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
