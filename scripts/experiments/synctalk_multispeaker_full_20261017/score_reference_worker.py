"""Freeze all four source-owned score inputs independently of generator preparation."""
import hashlib,json,os,subprocess,time
from pathlib import Path
import psutil
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
state=dict(pid=os.getpid(),identities=[],training_admitted=False)
def save():(ROOT/'score_reference_worker_receipt.json').write_text(json.dumps(state,indent=2))
def active(identity):
    for p in psutil.process_iter(['name','cmdline']):
        cmd=p.info['cmdline'] or []
        if p.info['name'].startswith('python') and any(Path(x).name=='prepare_score_reference.py' for x in cmd) and identity in cmd:return True
    return False
try:
    for identity in ['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers']:
        state['current_identity']=identity;save();receipt=ROOT/'score_reference'/identity/'score_reference_receipt.json'
        if active(identity):
            deadline=time.time()+7200
            while active(identity):assert time.time()<deadline;time.sleep(20)
        elif not receipt.exists():
            with (ROOT/f'score_reference_worker_{identity}.log').open('x') as log:
                subprocess.run(['/home/wjj/.venvs/syncnet/bin/python',str(ROOT/'prepare_score_reference.py'),'--root',str(ROOT),'--identity',identity],stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads(receipt.read_text());assert result['passed'] and result['repeat_audio_and_visual_exact'] and len(result['clips'])==6
        state['identities'].append(dict(identity=identity,passed=True,receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest()));save()
    state['passed']=True;save()
except Exception as error:
    state.update(passed=False,error=repr(error));save();raise
