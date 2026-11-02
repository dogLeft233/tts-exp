"""Persist completed remote stage1 packages on F and verify every extracted file."""
import argparse,hashlib,json,subprocess,tarfile,time
from pathlib import Path
ROOT=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017')
REMOTE='/dev/shm/synctalk-multispeaker-data-20261017'
HOST='wjj@192.168.50.2'
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',HOST]
SCP=['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes']
IDENTITIES=['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers']
p=argparse.ArgumentParser();p.add_argument('--identities',nargs='+',choices=IDENTITIES,default=IDENTITIES)
IDENTITIES=p.parse_args().identities
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def fetch_json(path):
    p=subprocess.run(SSH+['cat '+path],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=60)
    if p.returncode:return None
    try:return json.loads(p.stdout)
    except json.JSONDecodeError:return None
packages=ROOT/'preprocessed_packages';packages.mkdir(exist_ok=True)
destination=ROOT/'preprocessed';destination.mkdir(exist_ok=True)
old=ROOT/'stage1_persistence_receipt.json'
prior=json.loads(old.read_text()) if old.exists() else {}
retained=[x for x in prior.get('verified_identities',[]) if x['identity'] not in IDENTITIES]
status=dict(passed=False,state='running',verified_identities=retained,training_admitted=False)
def save():(ROOT/'stage1_persistence_receipt.json').write_text(json.dumps(status,indent=2))
save()
try:
    for identity in IDENTITIES:
        status['current_identity']=identity;save();deadline=time.time()+14400
        while True:
            package=fetch_json(f'{REMOTE}/{identity}_stage1_package.json')
            if package:break
            worker=fetch_json(f'{REMOTE}/stage1_worker_receipt.json')
            if worker and worker.get('passed') is False:raise RuntimeError(worker.get('error','remote worker failed'))
            assert time.time()<deadline,'stage1 package timeout'
            time.sleep(30)
        assert package['passed'] and package['identity']==identity
        archive=packages/f'{identity}_stage1.tar.gz';assert not archive.exists()
        partial=archive.with_suffix('.partial')
        subprocess.run(SCP+[HOST+':'+package['archive'],str(partial)],check=True,timeout=1800)
        assert partial.stat().st_size==package['bytes'] and sha(partial)==package['sha256']
        partial.rename(archive)
        manifest=packages/f'{identity}_stage1_manifest.json'
        subprocess.run(SCP+[f'{HOST}:{REMOTE}/{identity}_stage1_manifest.json',str(manifest)],check=True,timeout=120)
        assert sha(manifest)==package['manifest_sha256']
        assert not (destination/identity).exists()
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                target=(destination/member.name).resolve()
                assert target.is_relative_to(destination.resolve())
                assert not member.issym() and not member.islnk()
            tar.extractall(destination,filter='data')
        entries=json.loads(manifest.read_text())
        for row in entries:
            p=destination/identity/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
        result=dict(identity=identity,passed=True,files=len(entries),archive_sha256=package['sha256'],all_extracted_files_verified=True)
        (packages/f'{identity}_verification.json').write_text(json.dumps(result,indent=2))
        status['verified_identities'].append(result);save();print(json.dumps(result),flush=True)
    status['passed']=True;status['state']='completed';save()
except Exception as error:
    status['state']='failed';status['error']=repr(error);save();raise
