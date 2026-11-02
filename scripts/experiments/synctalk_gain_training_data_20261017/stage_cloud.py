"""Read the independent Linux cloud mount into task-owned volatile staging."""
import hashlib
import json
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

root = Path('/dev/shm/synctalk-gain-training-data-20261017')
root.mkdir(exist_ok=True)
source = Path('/home/wjj/百度网盘/tts-exp-research-models/synctalk-gain-training-data-20261017/May.zip')
dest = root / 'May.zip'
expected = '008fe3fb46430a92beff334e26f37eab229c717ff5b577dfbba90f46c3848d4b'
assert source.stat().st_size == 1225104190
verify_only = '--verify-only' in sys.argv
assert verify_only or not dest.exists(), 'never overwrite a previous transport attempt'
h = hashlib.sha256()
before = shutil.disk_usage('/').free
total = 0
with source.open('rb') as src, (nullcontext(None) if verify_only else dest.open('xb')) as dst:
    while True:
        block = src.read(4 * 1024**2)
        if not block:
            break
        if shutil.disk_usage('/').free < 128 * 1024**2:
            raise RuntimeError('root free-space guard: cloud read must not exhaust root')
        if dst:
            dst.write(block)
        h.update(block)
        total += len(block)
assert h.hexdigest() == expected
receipt = {'source': str(source), 'staging': None if verify_only else str(dest), 'bytes': total,
           'sha256': h.hexdigest(), 'independent_linux_cloud_readback': True,
           'root_free_before': before, 'root_free_after': shutil.disk_usage('/').free,
           'staging_is_volatile': True, 'training_runs': 0}
(root / 'cloud_storage_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
