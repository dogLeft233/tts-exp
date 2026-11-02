"""Stream a sealed remote tar to F, independently hash every archived member."""
import argparse,hashlib,json,subprocess,tarfile
from pathlib import Path
ROOT=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017')
REMOTE='/dev/shm/synctalk-multispeaker-data-20261017'
HOST='wjj@192.168.50.2'
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',HOST]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def run(identity,kind='derived'):
    assert identity in ['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers']
    assert kind in ['derived','qualification']
    snapshot='snapshots' if kind=='derived' else 'qualification_snapshots'
    out=ROOT/(kind+'_packages')/identity;out.mkdir(parents=True,exist_ok=False)
    raw=subprocess.check_output(SSH+[f'cat {REMOTE}/{snapshot}/{identity}/manifest.json'])
    manifest=json.loads(raw);assert manifest['identity']==identity
    (out/'manifest.json').write_bytes(raw)
    archive=out/(kind+'.tar.gz');partial=out/(kind+'.tar.gz.partial')
    with partial.open('xb') as f:
        subprocess.run(SSH+[f'tar --hard-dereference -czf - -C {REMOTE} --null -T {REMOTE}/{snapshot}/{identity}/files.nul'],stdout=f,check=True)
        f.flush()
        import os
        os.fsync(f.fileno())
    expected={x['path']:x for x in manifest['files']};seen=set()
    with tarfile.open(partial,'r|gz') as tar:
        for member in tar:
            assert member.isfile() and member.name in expected and member.name not in seen
            row=expected[member.name];assert member.size==row['bytes']
            h=hashlib.sha256()
            with tar.extractfile(member) as f:
                for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
            assert h.hexdigest()==row['sha256'],member.name
            seen.add(member.name)
    assert seen==expected.keys()
    packed_sha=sha(partial);partial.rename(archive)
    receipt=dict(identity=identity,passed=True,files=len(seen),raw_bytes=sum(x['bytes'] for x in expected.values()),
        archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=packed_sha,
        manifest_sha256=hashlib.sha256(raw).hexdigest(),all_archived_member_bytes_verified=True,
        training_admitted=False,verification='full sequential gzip decompression and every tar member SHA256, no extraction required')
    (out/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);a=p.parse_args();run(a.identity)
