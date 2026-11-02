"""Remove one owned volatile archive only after a verified F persistence receipt."""
import argparse,hashlib,json
from pathlib import Path
import psutil
p=argparse.ArgumentParser();p.add_argument('--identity',required=True);p.add_argument('--archive-sha',required=True);a=p.parse_args()
assert a.identity in ['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers']
root=Path('/dev/shm/synctalk-multispeaker-data-20261017')
receipt=json.loads((root/f'{a.identity}_persistent_verification.json').read_text())
assert receipt['passed'] and receipt['all_extracted_files_verified'] and receipt['archive_sha256']==a.archive_sha
target=root/f'{a.identity}_stage1.tar.gz'
assert not target.is_symlink() and target.resolve()==target and target.parent==root
before=target.stat();h=hashlib.sha256()
with target.open('rb') as f:
    for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
assert h.hexdigest()==a.archive_sha and target.stat().st_mtime_ns==before.st_mtime_ns
for process in psutil.process_iter():
    try:assert not any(x.path==str(target) for x in process.open_files()),f'archive busy: {process.pid}'
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
target.unlink()
result=dict(passed=True,identity=a.identity,removed_volatile_archive=str(target),bytes=before.st_size,
    persistent_archive_sha256=a.archive_sha,all_source_preprocessed_files_retained=True)
(root/f'{a.identity}_stage1_reclamation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
