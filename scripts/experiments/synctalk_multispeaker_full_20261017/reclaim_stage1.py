"""Free SHM package copies after the existing full F restore has verified."""
import hashlib,json,subprocess,time
from pathlib import Path
root=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017');remote='/dev/shm/synctalk-multispeaker-data-20261017';host='wjj@192.168.50.2'
ids=['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers']
state=dict(state='running',reclaimed=[],passed=False)
def save():(root/'stage1_reclamation_receipt.json').write_text(json.dumps(state,indent=2))
save()
try:
    for identity in ids:
        verification=root/'preprocessed_packages'/f'{identity}_verification.json';deadline=time.time()+14400
        while not verification.exists():
            assert time.time()<deadline,'persistence timeout';time.sleep(30)
        receipt=json.loads(verification.read_text());assert receipt['passed'] and receipt['all_extracted_files_verified']
        archive=root/'preprocessed_packages'/f'{identity}_stage1.tar.gz';h=hashlib.sha256()
        with archive.open('rb') as f:
            for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
        assert h.hexdigest()==receipt['archive_sha256']
        subprocess.run(['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',str(verification),f'{host}:{remote}/{identity}_persistent_verification.json'],check=True,timeout=120)
        command=f'/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python {remote}/reclaim_stage1_archive_remote.py --identity {identity} --archive-sha {receipt["archive_sha256"]}'
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',host,command],capture_output=True,encoding='utf-8',errors='replace',timeout=180)
        assert result.returncode==0,result.stderr
        item=json.loads(result.stdout);assert item['passed'];state['reclaimed'].append(item);save();print(json.dumps(item),flush=True)
    state['state']='completed';state['passed']=True;save()
except Exception as error:
    state['state']='failed';state['error']=repr(error);save();raise
