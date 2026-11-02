"""F-side preparation observer: verified archives, then narrowly scoped SHM retirement."""
import hashlib,json,subprocess,time
from pathlib import Path
from persist_prepared import run as persist
ROOT=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017')
REMOTE='/dev/shm/synctalk-multispeaker-data-20261017';HOST='wjj@192.168.50.2'
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',HOST]
SCP=['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes']
state=dict(state='running',identities=[],training_admitted=False)
def save():(ROOT/'derived_persistence_Cathy_independent_receipt.json').write_text(json.dumps(state,indent=2))
def read(path):
    r=subprocess.run(SSH+['cat '+path],capture_output=True,encoding='utf-8',errors='replace')
    return json.loads(r.stdout) if r.returncode==0 else None
save()
try:
    for identity in ['CathyMcMorrisRodgers']:
        state['current_identity']=identity;save();deadline=time.time()+43200
        while True:
            worker=read(REMOTE+'/stage2_v2_Cathy_independent_worker_receipt.json')
            assert not worker or worker.get('passed') is not False,worker
            if worker and any(x['identity']==identity for x in worker['identities']):break
            assert time.time()<deadline;time.sleep(20)
        subprocess.run(SSH+[f'/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python {REMOTE}/prepare_snapshot_manifest.py --root {REMOTE} --identity {identity}'],check=True)
        persist(identity)
        proof=ROOT/'derived_packages'/identity/'verification.json'
        record=json.loads(proof.read_text());assert record['passed'] and record['all_archived_member_bytes_verified']
        subprocess.run(SCP+[str(proof),HOST+f':{REMOTE}/snapshots/{identity}/F_archive_verification.json'],check=True)
        r=subprocess.run(SSH+[f'/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python {REMOTE}/reclaim_derived_supervision.py --root {REMOTE} --identity {identity} --proof {REMOTE}/snapshots/{identity}/F_archive_verification.json'],capture_output=True,encoding='utf-8',check=True)
        retirement=json.loads(r.stdout);assert retirement['passed']
        (ROOT/'derived_packages'/identity/'supervision_reclamation.json').write_text(json.dumps(retirement,indent=2))
        state['identities'].append(dict(identity=identity,archive_verified=True,supervision_retired=True,training_admitted=False));save()
    state.update(state='completed',passed=True);save()
except Exception as error:
    state.update(state='failed',passed=False,error=repr(error));save();raise
