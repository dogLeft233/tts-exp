"""Finish the authorized four-person preprocessing stage; never start training."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
IDENTITIES=['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers']
p=argparse.ArgumentParser();p.add_argument('--identities',nargs='+',choices=IDENTITIES,default=IDENTITIES)
IDENTITIES=p.parse_args().identities
status=dict(stage='official_jpeg_landmarks_parsing_only',identities=[],training_admitted=False,pid=os.getpid())
def save():(ROOT/'stage1_worker_receipt.json').write_text(json.dumps(status,indent=2))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
try:
    for identity in IDENTITIES:
        status['current_identity']=identity;save()
        qa=ROOT/'material'/identity/'independent_material_qa.json'
        deadline=time.time()+3600
        while not qa.exists():
            assert time.time()<deadline,'material QA timeout'
            time.sleep(10)
        assert json.loads(qa.read_text())['passed']
        folder=ROOT/'preprocessed'/identity;receipt=folder/'stage1_receipt.json'
        if folder.exists():
            deadline=time.time()+7200
            while not json.loads(receipt.read_text()).get('passed',False):
                running=any(p.info['name'].startswith('python') and any('preprocess_images_landmarks.py' in x for x in p.info['cmdline'] or [])
                    and identity in (p.info['cmdline'] or []) for p in psutil.process_iter(['name','cmdline']))
                assert running,f'{identity} previous stage1 stopped without passing'
                assert time.time()<deadline,'stage1 timeout'
                time.sleep(20)
        else:
            assert shutil.disk_usage(ROOT).free>2*1024**3,'need 2 GiB workspace reserve'
            with (ROOT/f'stage1_{identity}.log').open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'preprocess_images_landmarks.py'),'--root',str(ROOT),'--identity',identity],stdout=log,stderr=subprocess.STDOUT,check=True)
        r=json.loads(receipt.read_text());assert r['passed'] and sum(x['frames'] for x in r['splits'])==8250
        manifest=[dict(path=str(p.relative_to(folder)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(folder.rglob('*')) if p.is_file()]
        (ROOT/f'{identity}_stage1_manifest.json').write_text(json.dumps(manifest,indent=2))
        archive=ROOT/f'{identity}_stage1.tar.gz';assert not archive.exists()
        partial=archive.with_suffix('.partial')
        subprocess.run(['tar','-czf',str(partial),'-C',str(folder.parent),identity],check=True)
        partial.rename(archive)
        result=dict(identity=identity,passed=True,files=len(manifest),archive=str(archive),bytes=archive.stat().st_size,
            sha256=sha(archive),manifest_sha256=sha(ROOT/f'{identity}_stage1_manifest.json'),training_admitted=False)
        (ROOT/f'{identity}_stage1_package.json').write_text(json.dumps(result,indent=2))
        status['identities'].append(result);save();print(json.dumps(result),flush=True)
    status['passed']=True;save()
except Exception as error:
    status['passed']=False;status['error']=repr(error);save();raise
