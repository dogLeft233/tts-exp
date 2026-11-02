"""Independently persist exact loader/capacity inputs and receipts; never retire data."""
import argparse,hashlib,json,subprocess,tarfile
from pathlib import Path
from persist_prepared import ROOT,REMOTE,SSH,sha

def run(identity,attempt=1):
    assert identity in ['SherrodBrown','PatrickLeahy','ChrisVanHollen','CathyMcMorrisRodgers'] and attempt>0
    out=ROOT/'runtime_qualification'/identity/f'attempt{attempt:02d}';out.mkdir(parents=True,exist_ok=False)
    code="""import hashlib,json
from pathlib import Path
r=Path(%r);i=%r
paths=list((r/'capacity_qualification'/i).rglob('*'))+list((r/'loading_qualification'/i).rglob('*'))+list((r/'datasets'/i).rglob('*.json'))
paths += [r/'snapshots'/i/'control_supervision_restoration.json',r/'execution_contract_v1.json',r/'execution_unit_tests_attempt02.log']
files=[]
for p in sorted(set(paths)):
 if p.is_file() and not p.is_symlink():
  raw=p.read_bytes();files.append(dict(path=str(p.relative_to(r)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
assert any(x['path'].endswith('capacity_qualification.json') for x in files)
print(json.dumps(dict(identity=i,files=files)))
"""%(REMOTE,identity)
    import shlex
    raw=subprocess.check_output(SSH+['python3 -c '+shlex.quote(code)])
    manifest=json.loads(raw);(out/'manifest.json').write_bytes(raw)
    files={x['path']:x for x in manifest['files']};names='\0'.join(files)+'\0'
    archive=out/'qualification.tar.gz'
    with archive.open('xb') as f:
        subprocess.run(SSH+[f'tar --hard-dereference -czf - -C {REMOTE} --null -T -'],input=names.encode(),stdout=f,check=True)
        f.flush()
        import os
        os.fsync(f.fileno())
    seen=set()
    with tarfile.open(archive,'r|gz') as source:
        for item in source:
            assert item.isfile() and item.name in files and item.name not in seen
            row=files[item.name];assert item.size==row['bytes']
            h=hashlib.sha256()
            stream=source.extractfile(item)
            for part in iter(lambda:stream.read(4*1024**2),b''):h.update(part)
            assert h.hexdigest()==row['sha256'];seen.add(item.name)
    assert seen==files.keys()
    result=dict(identity=identity,files=len(files),passed=True,all_archived_member_bytes_verified=True,
        archive=str(archive),archive_sha256=sha(archive),manifest_sha256=hashlib.sha256(raw).hexdigest(),training_admitted=False)
    (out/'verification.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);p.add_argument('--attempt',type=int,default=1)
    a=p.parse_args();print(json.dumps(run(a.identity,a.attempt)))
