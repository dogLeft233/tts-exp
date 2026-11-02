"""Wait for complete independent revision qualification before retiring verified video caches."""
import argparse,json,os,shlex,subprocess,time,traceback
from pathlib import Path
from persist_prepared import ROOT,REMOTE,SSH,sha
from execution_journal import atomic_json


def run(identity,revision):
    assert Path(revision).name==revision and revision in ['CathyMcMorrisRodgers_recording2_v2','ChrisVanHollen_metadata_order_v1']
    assert identity==revision.split('_')[0]
    folder=ROOT/'dataset_revisions'/revision;remote=REMOTE+'/dataset_revisions/'+revision
    output=folder/'canonical_cache_worker_attempt01.json';assert not output.exists()
    status=dict(pid=os.getpid(),state='waiting_full_independent_revision_qualification',passed=None,identity=identity,training_admitted=False)
    def save():atomic_json(output,status)
    save()
    try:
        deadline=time.monotonic()+43200
        while True:
            upstream=folder/'post_geometry_worker_attempt02_receipt.json'
            if upstream.exists():
                worker=json.loads(upstream.read_text());assert worker.get('passed') is not False,worker
                if worker.get('passed'):break
            assert time.monotonic()<deadline;time.sleep(30)
        package=folder/'material_packages'/identity
        proof=json.loads((package/'verification.json').read_text())
        assert proof['passed'] and proof['all_archived_member_bytes_verified']
        archive=Path(proof['archive']);assert archive.resolve().is_relative_to(package.resolve())
        assert sha(archive)==proof['archive_sha256'] and sha(package/'manifest.json')==proof['manifest_sha256']
        rows={x['path']:x for x in json.loads((package/'manifest.json').read_text())['files']}
        selected=[rows[f'material/{identity}/{role}.mkv'] for role in ['main','test']]
        code='''import hashlib,json,psutil
from pathlib import Path
r=Path(%r);identity=%r;rows=%r;proof=%r
assert r.resolve()==r and r.parent.name=='dataset_revisions'
for p in [r/'prepared'/identity/'geometry_v2/geometry_qualification.json',r/'score_reference'/identity/'score_reference_receipt.json',
          r/'score_reference'/identity/'independent_reference_qualification.json',r/'capacity_qualification'/identity/'capacity_qualification.json']:
 assert json.loads(p.read_text())['passed']
paths=[r/row['path'] for row in rows]
for p,row in zip(paths,rows):
 assert p.parent==r/'material'/identity and not p.is_symlink() and p.stat().st_size==row['bytes']
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(4194304),b''):h.update(block)
 assert h.hexdigest()==row['sha256']
for proc in psutil.process_iter():
 try:assert not any(Path(f.path) in paths for f in proc.open_files()),'active canonical reader'
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
result=dict(passed=True,identity=identity,retired_bytes=sum(row['bytes'] for row in rows),files=rows,
 archive_sha256=proof['archive_sha256'],original_bytes_persistently_preserved=True,PCM_and_RGB_manifests_and_score_references_retained=True)
out=r/'qualified_canonical_cache_retirement_attempt01.json';assert not out.exists()
for p in paths:p.unlink()
out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''%(remote,identity,selected,proof)
        status['state']='archive_fresh_verified_remote_byte_validation';save()
        result=json.loads(subprocess.check_output(SSH+['/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python -c '+shlex.quote(code)]))
        status.update(state='verified_canonical_cache_retired',passed=True,result=result);save()
    except BaseException:
        status.update(state='failed',passed=False,error=traceback.format_exc());save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);p.add_argument('--revision',required=True)
    a=p.parse_args();run(a.identity,a.revision)
