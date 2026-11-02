"""Watch qualified identities, fresh-hash both F videos, retire only matching SHM caches."""
import hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017')
REMOTE='/dev/shm/synctalk-multispeaker-data-20261017';HOST='wjj@192.168.50.2'
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',HOST]
SCP=['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes']
state=dict(state='running',identities=[],training_admitted=False)
def save():(ROOT/'canonical_reclamation_receipt.json').write_text(json.dumps(state,indent=2))
def read(path):
    r=subprocess.run(SSH+['cat '+path],capture_output=True,encoding='utf-8',errors='replace')
    return json.loads(r.stdout) if r.returncode==0 else None
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
save()
try:
    for identity in ['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers']:
        state['current_identity']=identity;save();deadline=time.time()+43200
        while True:
            results=[read(REMOTE+f'/prepared/{identity}/geometry_v2/geometry_qualification.json'),
                read(REMOTE+f'/score_reference/{identity}/score_reference_receipt.json'),
                read(REMOTE+f'/snapshots/{identity}/supervision_reclamation.json')]
            if all(x and x.get('passed') for x in results):break
            worker=read(REMOTE+'/stage2_v2_worker_receipt.json')
            assert not worker or worker.get('passed') is not False,worker
            assert time.time()<deadline;time.sleep(20)
        source=json.loads((ROOT/'material'/identity/'material_receipt.json').read_text())
        files=[]
        for role in ['main','test']:
            p=ROOT/'material'/identity/(role+'.mkv');original=next(x for x in source['recordings'] if x['role']==role)
            assert p.stat().st_size==original['bytes'] and sha(p)==original['sha256']
            files.append(dict(role=role,path=str(p),bytes=p.stat().st_size,sha256=original['sha256']))
        proof=ROOT/'material'/identity/'canonical_fresh_verification.json'
        proof.write_text(json.dumps(dict(identity=identity,passed=True,files=files),indent=2))
        subprocess.run(SCP+[str(proof),HOST+f':{REMOTE}/material/{identity}/F_canonical_verification.json'],check=True)
        r=subprocess.run(SSH+[f'/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python {REMOTE}/reclaim_canonical_remote.py --identity {identity}'],capture_output=True,encoding='utf-8',check=True)
        result=json.loads(r.stdout);assert result['passed'];state['identities'].append(result);save()
    state.update(state='completed',passed=True);save()
except Exception as error:
    state.update(state='failed',passed=False,error=repr(error));save();raise
