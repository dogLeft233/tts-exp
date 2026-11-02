"""Retire only fully archived, rejected namespace stage1 caches after all-byte checks."""
import hashlib,json,shlex,subprocess
from pathlib import Path
from persist_prepared import ROOT,REMOTE,SSH,sha


def run():
    revision='DebbieDingell_recording_roles_v2';identity='DebbieDingell'
    folder=ROOT/'dataset_revisions'/revision/'failed_preparation_preservation_attempt01'
    proof=json.loads((folder/'verification.json').read_text());archive=Path(proof['archive'])
    assert proof['passed'] and proof['all_archived_member_bytes_verified'] and not proof['geometry_qualified']
    assert archive.resolve().is_relative_to(folder.resolve()) and sha(archive)==proof['archive_sha256']
    assert sha(folder/'manifest.json')==proof['manifest_sha256']
    remote=REMOTE+'/dataset_revisions/'+revision
    code='''import hashlib,json,psutil,shutil
from pathlib import Path
r=Path(%r);proof=%r;identity=%r
assert r.resolve()==r and r.name=='DebbieDingell_recording_roles_v2'
assert json.loads((r/'geometry_resume_worker_receipt.json').read_text())['passed'] is False
manifest_path=r/'failure_snapshot_attempt01/manifest.json'
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==proof['manifest_sha256']
manifest=json.loads(manifest_path.read_text());rows={x['path']:x for x in manifest['files']}
for alias in manifest['byte_exact_stage1_aliases']:
 row=rows[alias['same_bytes_as']];assert row['sha256']==alias['sha256']
 rows[alias['path']]=dict(path=alias['path'],bytes=row['bytes'],sha256=row['sha256'])
targets=[r/'preprocessed'/identity/s for s in ['train','val','test']]
targets += [r/'prepared'/identity/s/k for s in ['train','val','test'] for k in ['ori_imgs','parsing']]
for target in targets:
 assert target.is_dir() and not target.is_symlink() and target.resolve().is_relative_to(r)
expected={name:row for name,row in rows.items() if any((r/name).is_relative_to(t) for t in targets)}
current={p.relative_to(r).as_posix() for t in targets for p in t.rglob('*') if p.is_file()}
assert current==expected.keys() and len(current)==74250
inodes={}
for name,row in expected.items():
 p=r/name;assert not p.is_symlink() and p.resolve().is_relative_to(r) and p.stat().st_size==row['bytes']
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 assert h.hexdigest()==row['sha256']
 st=p.stat();inodes[(st.st_dev,st.st_ino)]=st.st_size
for proc in psutil.process_iter():
 try:
  assert not any(any(Path(f.path).is_relative_to(t) for t in targets) for f in proc.open_files()),'active stage1 reader'
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
intent=r/'failed_stage1_cache_retirement_intent_attempt01.json';assert not intent.exists()
result=dict(passed=True,geometry_qualified=False,training_admitted=False,files=len(current),
 logical_bytes=sum(x['bytes'] for x in expected.values()),unique_inode_payload_bytes=sum(inodes.values()),
 targets=[p.relative_to(r).as_posix() for p in targets],archive_sha256=proof['archive_sha256'],
 original_full_bytes_persistently_verified=True,metadata_geometry_and_PCM_retained=True)
intent.write_text(json.dumps(result,indent=2))
for target in targets:shutil.rmtree(target)
out=r/'failed_stage1_cache_retirement_attempt01.json';assert not out.exists();out.write_text(json.dumps(result,indent=2))
print(json.dumps(result))
'''%(remote,proof,identity)
    result=json.loads(subprocess.check_output(SSH+['/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python -c '+shlex.quote(code)]))
    path=folder/'stage1_cache_retirement.json';assert not path.exists();path.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':run()
