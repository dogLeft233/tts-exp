"""Full independent archive readback before retiring two owned canonical caches."""
import hashlib,json,os,shlex,subprocess,tarfile
from pathlib import Path
from cohort_binding import FROOT,REMOTE,REVISIONS,expected_root,sha
from persist_prepared import SSH,HOST
from execution_journal import atomic_json


def run(identity):
    selected=expected_root(FROOT,identity);remote=expected_root(REMOTE,identity).as_posix()
    assert REVISIONS[identity] and identity=='DebbieDingell'
    flow=json.loads((selected/'flow_cache_persistence_receipt.json').read_text());assert flow['passed']
    for row in flow['verified']:assert sha(Path(row['path']))==row['gzip_sha256']
    source=Path(__file__).with_name('seal_revision_failure.py')
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',str(source),HOST+':'+REMOTE.as_posix()+'/'+source.name],check=True)
    command=['python3',(REMOTE/source.name).as_posix(),'--root',remote,'--identity',identity]
    subprocess.run(SSH+[shlex.join(command)],check=True)
    folder=remote+'/failure_snapshot_attempt01';out=selected/'failed_preparation_preservation_attempt01';out.mkdir(exist_ok=False)
    raw=subprocess.check_output(SSH+['cat '+shlex.quote(folder+'/manifest.json')]);manifest=json.loads(raw)
    assert manifest['geometry_qualified'] is False and manifest['revision']==REVISIONS[identity]
    (out/'manifest.json').write_bytes(raw);expected={r['path']:r for r in manifest['files']}
    archive=out/'failed_preparation.tar.gz'
    with archive.open('xb') as stream:
        subprocess.run(SSH+['tar --hard-dereference -czf - -C '+shlex.quote(remote)+' --null -T '+shlex.quote(folder+'/files.nul')],stdout=stream,check=True)
        stream.flush();os.fsync(stream.fileno())
    seen=set()
    with tarfile.open(archive,'r|gz') as tf:
        for member in tf:
            assert member.isfile() and member.name in expected and member.name not in seen
            row=expected[member.name];assert row['bytes']==member.size
            h=hashlib.sha256()
            with tf.extractfile(member) as stream:
                for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
            assert h.hexdigest()==row['sha256'];seen.add(member.name)
    assert seen==expected.keys()
    proof=dict(passed=True,identity=identity,revision=REVISIONS[identity],geometry_qualified=False,training_admitted=False,
        archive=str(archive),archive_sha256=sha(archive),manifest_sha256=sha(out/'manifest.json'),
        files=len(seen),raw_bytes=sum(r['bytes'] for r in expected.values()),all_archived_member_bytes_verified=True,
        exact_prepared_stage1_aliases=len(manifest['byte_exact_stage1_aliases']))
    atomic_json(out/'verification.json',proof)
    rows=[expected[f'material/{identity}/{role}.mkv'] for role in ['main','test']]
    code='''import hashlib,json,psutil
from pathlib import Path
r=Path(%r);rows=%r;proof=%r
assert r==Path(%r) and r.resolve()==r
assert json.loads((r/'geometry_resume_worker_receipt.json').read_text())['passed'] is False
paths=[r/row['path'] for row in rows]
for p,row in zip(paths,rows):
 assert p.parent==r/'material'/%r and not p.is_symlink() and p.stat().st_size==row['bytes']
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 assert h.hexdigest()==row['sha256']
for proc in psutil.process_iter():
 try:
  assert not any(Path(f.path) in paths for f in proc.open_files()),'active canonical reader'
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
result=dict(passed=True,retired_bytes=sum(x['bytes'] for x in rows),files=rows,archive_sha256=proof['archive_sha256'],
 geometry_qualified=False,original_bytes_persistently_preserved=True)
output=r/'failed_canonical_cache_retirement_attempt01.json';assert not output.exists()
for p in paths:p.unlink()
output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''%(remote,rows,proof,remote,identity)
    result=json.loads(subprocess.check_output(SSH+['/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python -c '+shlex.quote(code)]))
    atomic_json(out/'canonical_cache_retirement.json',result)
    print(json.dumps(proof),flush=True)


if __name__=='__main__':run('DebbieDingell')
