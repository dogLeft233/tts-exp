"""Windows preparation-only observer: restore controls, qualify original loading and budget."""
import json,os,subprocess,time,traceback
from pathlib import Path
from persist_prepared import ROOT,REMOTE,SSH
from persist_runtime_qualification import run as persist
from restore_supervision import run as restore
IDS=['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers']

def read_remote(path):
    result=subprocess.run(SSH+[f'cat {REMOTE}/{path}'],capture_output=True)
    return json.loads(result.stdout) if result.returncode==0 else None

def run():
    receipt_path=ROOT/'runtime_qualification_worker_receipt.json'
    assert not receipt_path.exists(),'inspect and explicitly version any failed worker before replacing'
    result=dict(pid=os.getpid(),identities=[],training_admitted=False,worker_can_start_training=False)
    def save():receipt_path.write_text(json.dumps(result,indent=2))
    save()
    for identity in IDS:
        result.update(current_identity=identity,current_phase='waiting_verified_archive_and_reclamation');save()
        while True:
            proof=ROOT/'derived_packages'/identity/'verification.json'
            reclaimed=read_remote(f'snapshots/{identity}/supervision_reclamation.json')
            if proof.exists() and json.loads(proof.read_text())['passed'] and reclaimed and reclaimed['passed']:break
            time.sleep(30)
        if not read_remote(f'snapshots/{identity}/control_supervision_restoration.json'):
            result['current_phase']='control_restore';save();restore(identity,'control')
        jobs=[('dataset_binding',f'datasets/{identity}/binding_receipt.json',f'build_datasets.py --root {REMOTE} --identity {identity} --persisted-supervision'),
              ('loading',f'loading_qualification/{identity}/loading_qualification.json',f'qualify_loading.py --identity {identity}'),
              ('capacity',f'capacity_qualification/{identity}/capacity_qualification.json',f'qualify_capacity.py --identity {identity}')]
        for phase,path,command in jobs:
            result['current_phase']=phase;save();prior=read_remote(path)
            if prior is None:
                log=ROOT/'runtime_qualification'/f'{identity}_{phase}_transport.log';log.parent.mkdir(exist_ok=True)
                with log.open('xb') as f:
                    subprocess.run(SSH+[f'bash {REMOTE}/runtime_exec.sh {REMOTE}/{command}'],stdout=f,stderr=subprocess.STDOUT,check=True)
                prior=read_remote(path)
            assert prior and prior['passed'],(identity,phase)
        result['current_phase']='F_full_member_readback';save()
        prior_attempts=list((ROOT/'runtime_qualification'/identity).glob('attempt*'))
        verification=persist(identity,1+max([int(p.name[7:]) for p in prior_attempts],default=0))
        result['identities'].append(verification);save();print(json.dumps(verification),flush=True)
    result.update(passed=True,current_phase='completed');save()

if __name__=='__main__':
    try:run()
    except BaseException:
        path=ROOT/'runtime_qualification_worker_failure.txt';path.write_text(traceback.format_exc());raise
