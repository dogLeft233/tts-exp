"""Independently retain every sealed failed-preparation byte; integrity is not qualification."""
import argparse,hashlib,json,os,subprocess,tarfile
from persist_prepared import ROOT,REMOTE,SSH,sha
def run(identity):
    assert identity in ['DebbieDingell','CathyMcMorrisRodgers']
    source=REMOTE+'/failure_snapshots/'+identity+'_recording_pair_v1'
    out=ROOT/'geometry_diagnostics'/(identity+'_failed_v2_attempt01')/'full_failure_bundle';out.mkdir(parents=True,exist_ok=False)
    raw=subprocess.check_output(SSH+['cat '+source+'/manifest.json']);manifest=json.loads(raw);assert manifest['identity']==identity and manifest['geometry_qualified'] is False
    (out/'manifest.json').write_bytes(raw);archive=out/'failure.tar.gz'
    with archive.open('xb') as f:subprocess.run(SSH+[f'tar --hard-dereference -czf - -C {REMOTE} --null -T {source}/files.nul'],stdout=f,check=True);f.flush();os.fsync(f.fileno())
    expected={r['path']:r for r in manifest['files']};seen=set()
    with tarfile.open(archive,'r|gz') as tf:
        for member in tf:
            assert member.isfile() and member.name in expected and member.name not in seen
            row=expected[member.name];assert member.size==row['bytes'];h=hashlib.sha256()
            with tf.extractfile(member) as stream:
                for b in iter(lambda:stream.read(4*1024**2),b''):h.update(b)
            assert h.hexdigest()==row['sha256'];seen.add(member.name)
    assert seen==expected.keys()
    proof=dict(passed=True,identity=identity,archive_integrity_verified=True,geometry_qualified=False,training_admitted=False,
        files=len(seen),raw_bytes=sum(r['bytes'] for r in expected.values()),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),manifest_sha256=sha(out/'manifest.json'))
    (out/'verification.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);a=p.parse_args();run(a.identity)
