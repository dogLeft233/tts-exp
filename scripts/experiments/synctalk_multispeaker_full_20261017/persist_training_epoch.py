"""Read back each completed epoch and its full state before any cache rotation."""
import hashlib,json,os,shlex,subprocess,tarfile
from pathlib import Path,PurePosixPath
from persist_prepared import ROOT,REMOTE,SSH,sha
from execution_journal import atomic_json

def verify_persisted(relative,allow_retired=False):
    out=ROOT/relative;proof=json.loads((out/'verification.json').read_text())
    assert proof['passed'] and proof['all_archived_member_bytes_verified'] and proof['relative']==Path(relative).as_posix()
    assert sha(out/'manifest.json')==proof['manifest_sha256'] and sha(out/'artifacts.tar.gz')==proof['archive_sha256']
    for row in json.loads((out/'manifest.json').read_text())['files']:
        file=out/'artifacts'/row['path'];assert file.resolve().is_relative_to((out/'artifacts').resolve())
        assert file.stat().st_size==row['bytes'] and sha(file)==row['sha256']
    checkpoint=Path(proof['checkpoint']);assert checkpoint.resolve().is_relative_to(out.resolve())
    if checkpoint.exists():assert sha(checkpoint)==proof['checkpoint_sha256']
    else:
        assert allow_retired
        retired=json.loads((out/'checkpoint_cache_retirement.json').read_text())
        assert retired['passed'] and retired['retired_checkpoint_sha256']==proof['checkpoint_sha256']
    return proof

def run(relative):
    relative=Path(relative)
    assert relative.parts[0]=='training' and len(relative.parts)==7 and '..' not in relative.parts
    remote=f'{REMOTE}/{relative.as_posix()}'
    raw=subprocess.check_output(SSH+[f'cat {shlex.quote(remote+"/epoch_receipt.json")}'])
    receipt=json.loads(raw);assert receipt['passed']
    checkpoint=PurePosixPath(receipt['checkpoint_path']);assert str(checkpoint).startswith(remote+'/workspace/checkpoints/')
    out=ROOT/relative
    if (out/'verification.json').exists():return verify_persisted(relative)
    if out.exists():
        # Preserve interrupted copies, then retry from the completed immutable remote epoch.
        assert out.resolve().is_relative_to((ROOT/'training').resolve()) and not out.is_symlink()
        attempt=1
        while out.with_name(out.name+f'_transfer_failed_{attempt:03d}').exists():attempt+=1
        failed=out.with_name(out.name+f'_transfer_failed_{attempt:03d}')
        assert failed.resolve().is_relative_to((ROOT/'training').resolve()) and not failed.exists()
        out.rename(failed)
    out.mkdir(parents=True,exist_ok=False)
    target=out/'full_checkpoint.pth';partial=target.with_suffix('.partial')
    with partial.open('xb') as f:
        subprocess.run(SSH+['cat '+shlex.quote(str(checkpoint))],stdout=f,check=True);f.flush();os.fsync(f.fileno())
    assert sha(partial)==receipt['checkpoint_sha256'];partial.rename(target)
    code="""import hashlib,json
from pathlib import Path
r=Path(%r);base=Path(%r);files=[]
for p in sorted(base.rglob('*')):
 if p.is_file() and not p.is_symlink() and 'checkpoints' not in p.relative_to(base).parts:
  raw=p.read_bytes();files.append(dict(path=str(p.relative_to(base)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
print(json.dumps(dict(files=files)))
"""%(REMOTE,remote)
    manifest_raw=subprocess.check_output(SSH+['python3 -c '+shlex.quote(code)])
    manifest=json.loads(manifest_raw);(out/'manifest.json').write_bytes(manifest_raw)
    files={x['path']:x for x in manifest['files']};archive=out/'artifacts.tar.gz'
    with archive.open('xb') as f:
        subprocess.run(SSH+[f'tar --hard-dereference -czf - -C {shlex.quote(remote)} --null -T -'],input=('\0'.join(files)+'\0').encode(),stdout=f,check=True)
        f.flush();os.fsync(f.fileno())
    seen=set()
    with tarfile.open(archive,'r|gz') as source:
        for item in source:
            assert item.isfile() and item.name in files and item.name not in seen
            row=files[item.name];assert item.size==row['bytes']
            destination=out/'artifacts'/item.name
            assert destination.resolve().is_relative_to((out/'artifacts').resolve())
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as f:
                stream=source.extractfile(item)
                for part in iter(lambda:stream.read(4*1024**2),b''):f.write(part)
            assert sha(destination)==row['sha256'];seen.add(item.name)
    assert seen==files.keys()
    assert (out/'artifacts/epoch_receipt.json').read_bytes()==raw
    proof=dict(passed=True,relative=relative.as_posix(),checkpoint_sha256=sha(target),checkpoint=str(target),
        archive_sha256=sha(archive),manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        all_archived_member_bytes_verified=True,files=len(files),remote_checkpoint=str(checkpoint))
    atomic_json(out/'verification.json',proof);return proof

def retire_consumed_previous(previous,current):
    old_path=PurePosixPath(previous['checkpoint_path']);new_path=PurePosixPath(current['checkpoint_path'])
    assert old_path!=new_path
    # The own arm's coarse-final and final-full states are durable experiment outputs.
    if previous['epoch'] in [10,17]:return
    old_relative=old_path.parent.parent.parent.relative_to(PurePosixPath(REMOTE))
    new_relative=new_path.parent.parent.parent.relative_to(PurePosixPath(REMOTE))
    old_dir=ROOT/old_relative.as_posix();new_dir=ROOT/new_relative.as_posix()
    retirement_path=old_dir/'checkpoint_cache_retirement.json'
    if retirement_path.exists():
        retired=json.loads(retirement_path.read_text())
        assert retired['passed'] and retired['retired_checkpoint_sha256']==previous['checkpoint_sha256'] and retired['superseded_by_checkpoint_sha256']==current['checkpoint_sha256']
        return
    old_proof=json.loads((old_dir/'verification.json').read_text());new_proof=json.loads((new_dir/'verification.json').read_text())
    assert old_proof['passed'] and new_proof['passed']
    old=Path(old_proof['checkpoint']);new=Path(new_proof['checkpoint'])
    assert old.resolve().is_relative_to((ROOT/'training').resolve()) and new.resolve().is_relative_to((ROOT/'training').resolve())
    assert old_proof['checkpoint_sha256']==previous['checkpoint_sha256']
    intent_path=old_dir/'checkpoint_cache_retirement_intent.json'
    if old.exists():assert sha(old)==previous['checkpoint_sha256']
    else:
        intent=json.loads(intent_path.read_text());assert intent['old_sha256']==previous['checkpoint_sha256'] and intent['new_sha256']==current['checkpoint_sha256']
    assert sha(new)==new_proof['checkpoint_sha256']==current['checkpoint_sha256']
    atomic_json(intent_path,dict(old_sha256=previous['checkpoint_sha256'],new_sha256=current['checkpoint_sha256']))
    code="""import hashlib,json
from pathlib import Path
r=Path(%r).resolve();old=Path(%r);new=Path(%r)
assert old.resolve().is_relative_to(r/'training') and new.resolve().is_relative_to(r/'training') and old!=new
assert hashlib.sha256(new.read_bytes()).hexdigest()==%r
if old.exists():
 assert hashlib.sha256(old.read_bytes()).hexdigest()==%r
 old.unlink()
"""%(REMOTE,str(old_path),str(new_path),current['checkpoint_sha256'],previous['checkpoint_sha256'])
    subprocess.run(SSH+['python3 -c '+shlex.quote(code)],check=True)
    if old.exists():old.unlink()
    atomic_json(retirement_path,dict(passed=True,retired_checkpoint_sha256=previous['checkpoint_sha256'],
        superseded_by_checkpoint_sha256=current['checkpoint_sha256'],reason='verified newer own-arm full state consumed prior checkpoint; fixed coarse-final and final states retained'))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--relative',required=True);print(json.dumps(run(p.parse_args().relative)))
