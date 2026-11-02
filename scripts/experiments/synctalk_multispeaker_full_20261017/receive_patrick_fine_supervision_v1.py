"""Restore only manifest-pinned GT/torso bytes from an independently verified F tar."""
import argparse,hashlib,json,sys,tarfile
from pathlib import Path
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
IDS=['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers']
def frames(mode,count):
    if mode in ['full','test']:return set(range(count))
    return set([*range(4),*range(count//2,count//2+4),*range(count-4,count)])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def run(identity,mode):
    assert identity=='PatrickLeahy' and mode=='full'
    folder=ROOT/'prepared'/identity
    manifest_path=ROOT/'snapshots'/identity/'manifest.json'
    proof=json.loads((ROOT/'snapshots'/identity/'F_archive_verification.json').read_text())
    assert proof['passed'] and proof['all_archived_member_bytes_verified'] and sha(manifest_path)==proof['manifest_sha256']
    manifest=json.loads(manifest_path.read_text());expected={}
    for row in manifest['files']:
        parts=Path(row['path']).parts
        if len(parts)==5 and parts[:2]==('prepared',identity) and parts[2] in ['train','val','test'] and parts[3] in ['gt_imgs','torso_imgs']:
            if mode=='test' and parts[2]!='test':continue
            count=dict(train=6000,val=750,test=1500)[parts[2]]
            index=int(Path(parts[4]).stem)
            if index in frames(mode,count):expected[row['path']]=row
    assert len(expected)==dict(full=16500,test=3000,control=72)[mode]
    seen=set();restored=0;reused=0
    with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
        for item in tar:
            assert item.isfile() and item.name in expected and item.name not in seen
            row=expected[item.name];assert item.size==row['bytes'] and item.size<16*1024**2
            path=ROOT/item.name;assert path.resolve().is_relative_to(folder.resolve()) and not path.is_symlink()
            path.parent.mkdir(parents=True,exist_ok=True);assert not path.parent.is_symlink()
            payload=tar.extractfile(item).read();assert hashlib.sha256(payload).hexdigest()==row['sha256']
            if path.exists():
                assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'];reused+=1
            else:
                partial=path.with_suffix(path.suffix+'.restore_partial')
                with partial.open('xb') as f:f.write(payload)
                assert sha(partial)==row['sha256'];partial.rename(path);restored+=1
            seen.add(item.name)
    assert seen==expected.keys()
    result=dict(passed=True,identity=identity,mode=mode,verified_files=len(seen),restored_files=restored,reused_exact_files=reused,
        manifest_sha256=proof['manifest_sha256'],archive_sha256=proof['archive_sha256'],all_restored_bytes_verified=True,
        local_supervision_complete=(mode=='full'),training_admitted=False)
    output=ROOT/'trials/patrick_fp16_fine_v1/storage'/(identity+'_full_restore.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    assert not output.exists();output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',choices=IDS,required=True);p.add_argument('--mode',choices=['control','full','test'],required=True)
    a=p.parse_args();run(a.identity,a.mode)
