"""Independent Linux readback of the derived archive uploaded by Windows."""
import hashlib
import json
import shutil
from pathlib import Path

root = Path('/dev/shm/synctalk-gain-training-data-20261017')
expected = json.loads((root / 'bundle_receipt.json').read_text())
source = Path('/home/wjj/百度网盘/tts-exp-research-models/synctalk-gain-training-data-20261017') / expected['file']
before = shutil.disk_usage('/').free
h = hashlib.sha256()
total = 0
with source.open('rb') as f:
    while block := f.read(4 * 1024**2):
        assert shutil.disk_usage('/').free > 128 * 1024**2
        total += len(block)
        h.update(block)
assert total == expected['bytes'] and h.hexdigest() == expected['sha256']
receipt = dict(expected, cloud_path=str(source), independent_linux_readback=True,
               root_free_before=before, root_free_after=shutil.disk_usage('/').free,
               passed=True, training_runs=0)
(root / 'cloud_bundle_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
