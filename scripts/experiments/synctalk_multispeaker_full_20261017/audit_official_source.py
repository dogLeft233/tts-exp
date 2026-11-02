"""Recover the exact upstream reference and compare the inherited working copy."""
import hashlib,json,zipfile
from pathlib import Path
root=Path('/dev/shm/synctalk-multispeaker-data-20261017')
archive=root/'official_pinned.zip';target=root/'official_exact';assert not target.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with zipfile.ZipFile(archive) as z:
    prefix='SyncTalk-43079fb3f349675f6cb33031bb07262f7566cdd5/'
    for item in z.infolist():
        assert item.filename.startswith(prefix)
        relative=item.filename[len(prefix):]
        if not relative:continue
        p=target/relative;assert p.resolve().is_relative_to(target.resolve())
        if item.is_dir():p.mkdir(parents=True,exist_ok=True)
        else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(item))
differences=[];checked=0
for p in target.rglob('*.py'):
    relative=p.relative_to(target);old=root/'official'/relative
    if old.exists():
        checked+=1
        if sha(p)!=sha(old):differences.append(dict(path=str(relative),reference_sha256=sha(p),working_sha256=sha(old)))
core=['nerf_triplane/network.py','nerf_triplane/renderer.py','nerf_triplane/utils.py']
result=dict(passed=all(sha(target/p)==sha(root/'official'/p) for p in core),reference_commit='43079fb3f349675f6cb33031bb07262f7566cdd5',
    archive_sha256=sha(archive),python_files_checked=checked,python_differences=differences,
    immutable_reference=str(target),training_admitted=False)
(root/'official_source_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
