"""Independently retain all newly qualified pose and scorer arrays on F."""
import json,subprocess,time
from pathlib import Path
from persist_prepared import run as persist
ROOT=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017')
REMOTE='/dev/shm/synctalk-multispeaker-data-20261017';HOST='wjj@192.168.50.2'
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',HOST]
state=dict(state='running',identities=[],training_admitted=False)
def save():(ROOT/'qualification_persistence_receipt.json').write_text(json.dumps(state,indent=2))
def read(path):
    r=subprocess.run(SSH+['cat '+path],capture_output=True,encoding='utf-8',errors='replace')
    return json.loads(r.stdout) if r.returncode==0 else None
save()
try:
    for identity in ['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers']:
        state['current_identity']=identity;save();deadline=time.time()+43200
        while True:
            geo=read(REMOTE+f'/prepared/{identity}/geometry_v2/geometry_qualification.json')
            score=read(REMOTE+f'/score_reference/{identity}/score_reference_receipt.json')
            if geo and geo.get('passed') and score and score.get('passed'):break
            worker=read(REMOTE+'/stage2_v2_worker_receipt.json');assert not worker or worker.get('passed') is not False,worker
            assert time.time()<deadline;time.sleep(20)
        subprocess.run(SSH+[f'/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python {REMOTE}/prepare_snapshot_manifest.py --root {REMOTE} --identity {identity} --kind qualification'],check=True)
        persist(identity,'qualification')
        proof=json.loads((ROOT/'qualification_packages'/identity/'verification.json').read_text());assert proof['passed']
        state['identities'].append(proof);save()
    state.update(state='completed',passed=True);save()
except Exception as error:
    state.update(state='failed',passed=False,error=repr(error));save();raise
