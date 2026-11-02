"""Fresh-verify an F archive and stream selected pinned images back to bounded SHM paths."""
import argparse,hashlib,io,json,subprocess,tarfile
from pathlib import Path
ROOT=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017')
REMOTE='/dev/shm/synctalk-multispeaker-data-20261017';HOST='wjj@192.168.50.2'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def run(identity,mode):
    directory=ROOT/'derived_packages'/identity
    proof=json.loads((directory/'verification.json').read_text());assert proof['passed'] and proof['all_archived_member_bytes_verified']
    archive=Path(proof['archive']);manifest_path=directory/'manifest.json'
    assert archive.is_relative_to(directory) and sha(archive)==proof['archive_sha256']
    assert sha(manifest_path)==proof['manifest_sha256']
    manifest=json.loads(manifest_path.read_text());expected={}
    for row in manifest['files']:
        parts=Path(row['path']).parts
        if len(parts)==5 and parts[:2]==('prepared',identity) and parts[2] in ['train','val','test'] and parts[3] in ['gt_imgs','torso_imgs']:
            if mode=='test' and parts[2]!='test':continue
            count=dict(train=6000,val=750,test=1500)[parts[2]];index=int(Path(parts[4]).stem)
            selected=mode in ['full','test'] or index in set([*range(4),*range(count//2,count//2+4),*range(count-4,count)])
            if selected:expected[row['path']]=row
    assert len(expected)==dict(full=16500,test=3000,control=72)[mode]
    log=directory/(mode+'_restore_transport.log')
    with log.open('xb') as errors:
        process=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',HOST,
            f'/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python {REMOTE}/receive_supervision_restore.py --identity {identity} --mode {mode}'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=errors)
        seen=set()
        with tarfile.open(archive,'r|gz') as source,tarfile.open(fileobj=process.stdin,mode='w|') as target:
            for item in source:
                if item.name not in expected:continue
                row=expected[item.name];assert item.isfile() and item.size==row['bytes'] and item.name not in seen
                payload=source.extractfile(item).read();assert hashlib.sha256(payload).hexdigest()==row['sha256']
                info=tarfile.TarInfo(item.name);info.size=len(payload);info.mode=0o600
                target.addfile(info,io.BytesIO(payload));seen.add(item.name)
        assert seen==expected.keys()
        process.stdin.close();output=process.stdout.read();assert process.wait()==0,log
    result=json.loads(output);assert result['passed'] and result['verified_files']==len(expected)
    (directory/(mode+'_restoration_verification.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',choices=['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers'],required=True);p.add_argument('--mode',choices=['control','full','test'],required=True)
    a=p.parse_args();run(a.identity,a.mode)
