"""Windows coordinator for admitted24 models; full epochs and verified rolling state."""
import argparse,json,os,shlex,subprocess,time,traceback
from pathlib import Path
from persist_prepared import ROOT,REMOTE,SSH,sha
from persist_training_epoch import run as persist,retire_consumed_previous,verify_persisted
from paired_budget import GAINS,qualify_epoch,qualify_phase
from restore_cohort_supervision import run as restore
from cohort_binding import identity_root,expected_root,REMOTE as REMOTE_PATH
from execution_journal import atomic_json,exclusive_lock
IDS=['SherrodBrown','PatrickLeahy','ChrisVanHollen','CathyMcMorrisRodgers']

def read_remote(path):
    return json.loads(subprocess.check_output(SSH+['cat '+shlex.quote(f'{REMOTE}/{path}')]))

def publish(relative,value):
    # JSON passes on stdin; no shell interpolation of metadata or file payload.
    destination=f'{REMOTE}/{relative}'
    code="from pathlib import Path;import sys;p=Path(%r);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(sys.stdin.buffer.read())"%destination
    subprocess.run(SSH+['python3 -c '+shlex.quote(code)],input=json.dumps(value,indent=2).encode(),check=True)

def run():
    with exclusive_lock(ROOT/'training_coordinator.lock'):run_locked()

def wait_job(identity,seed,gain,phase,epoch,prior,reference,log):
    arguments=['--identity',identity,'--seed',str(seed),'--gain',str(gain),'--phase',phase,'--epoch',str(epoch)]
    if prior:arguments+=['--checkpoint',prior['checkpoint_path'],'--resume',prior['_remote_receipt']]
    if reference:arguments+=['--reference-trace',reference]
    prefix=['/home/wjj/tts-audio/synctalk-amplitude-audit-20261016/env/bin/python',f'{REMOTE}/remote_epoch_job.py']
    state=json.loads(subprocess.check_output(SSH+[shlex.join(prefix+['start']+arguments)]))
    while state['state'] in ['starting','running']:
        time.sleep(30)
        state=json.loads(subprocess.check_output(SSH+[shlex.join(prefix+['status']+arguments)]))
    assert state['state']=='completed' and state['passed'] and state['returncode']==0,('epoch job failed or lost; do not replay updates',state)
    remote=f'{REMOTE}/jobs/{identity}/seed{seed}/gain_{gain:+d}/{phase}/ep{epoch:04d}/attempt01'
    log.parent.mkdir(parents=True,exist_ok=True)
    if log.exists():assert sha(log)==state['stdout_sha256']
    else:
        partial=log.with_suffix('.partial')
        if partial.exists():partial.rename(log.with_name(log.name+f'.interrupted_{time.time_ns()}'))
        with partial.open('xb') as f:subprocess.run(SSH+['cat '+shlex.quote(remote+'/stdout.log')],stdout=f,check=True)
        assert sha(partial)==state['stdout_sha256'];partial.rename(log)
    atomic_json(log.with_suffix('.execution.json'),state)

def handoff(identity):
    if identity==IDS[-1]:return
    for seed in [42,2026]:
        block=json.loads((ROOT/'blocks'/identity/f'seed{seed}'/'fine_qualification.json').read_text());assert block['passed']
    inputs=identity_root(identity,ROOT,require_admission=True);remote_inputs=expected_root(REMOTE_PATH,identity).as_posix()
    package=inputs/'derived_packages'/identity
    proof=json.loads((package/'verification.json').read_text())
    assert proof['passed'] and proof['all_archived_member_bytes_verified']
    assert sha(Path(proof['archive']))==proof['archive_sha256'] and sha(package/'manifest.json')==proof['manifest_sha256']
    retired=package/'post_training_supervision_reclamation.json'
    if retired.exists():assert json.loads(retired.read_text())['passed'];return
    prior=subprocess.run(SSH+[f'cat {remote_inputs}/snapshots/{identity}/post_training_supervision_reclamation.json'],capture_output=True)
    if prior.returncode==0:retirement=json.loads(prior.stdout)
    else:
        command=['bash',f'{REMOTE}/runtime_exec.sh',f'{REMOTE}/reclaim_derived_supervision.py','--root',remote_inputs,'--identity',identity,
            '--proof',f'{remote_inputs}/snapshots/{identity}/F_archive_verification.json','--report-name','post_training_supervision_reclamation.json']
        retirement=json.loads(subprocess.check_output(SSH+[shlex.join(command)]))
    assert retirement['passed'];atomic_json(retired,retirement)

def run_locked():
    admission=read_remote('data_qualification.json')
    assert admission['passed'] and admission['training_ready'] and admission['identities']==IDS
    assert admission['execution_contract_sha256']==sha(Path(__file__).with_name('execution_contract_v1.json'))
    assert all(sha(Path(__file__).with_name(name))==digest for name,digest in admission['windows_controller_sources'].items())
    state_path=ROOT/'training_coordinator_receipt.json'
    if state_path.exists():
        status=json.loads(state_path.read_text());assert status['admission']==admission
        status.update(pid=os.getpid(),coordinator_restarts=status.get('coordinator_restarts',0)+1)
    else:status=dict(pid=os.getpid(),blocks=[],models=24,passed=False,current_phase='admitted',admission=admission,coordinator_restarts=0)
    def save():atomic_json(state_path,status)
    save()
    for identity in IDS:
        inputs=identity_root(identity,ROOT,require_admission=True)
        completed=[ROOT/'blocks'/identity/f'seed{seed}'/'fine_qualification.json' for seed in [42,2026]]
        if all(p.exists() and json.loads(p.read_text())['passed'] for p in completed):handoff(identity);continue
        status.update(current_identity=identity,current_phase='full_supervision_restore');save()
        restoration=inputs/'derived_packages'/identity/'full_restoration_verification.json'
        if restoration.exists():assert json.loads(restoration.read_text())['passed']
        else:restore(identity,'full')
        # One identity's complete supervision remains live until its two seeds finish.
        for seed in [42,2026]:
            status.update(current_seed=seed);save();previous={g:None for g in GAINS}
            for phase,epochs in [('coarse',range(1,11)),('fine',range(11,18))]:
                epoch_results=[]
                for epoch in epochs:
                    receipts={};traces={};reference=None
                    for gain in [0,-6,6]:
                        relative=f'training/{identity}/seed{seed}/gain_{gain:+d}/{phase}/ep{epoch:04d}/attempt01'
                        status.update(current_phase=phase,current_epoch=epoch,current_gain_db=gain);save()
                        prior=previous[gain]
                        log=ROOT/'training_logs'/identity/f'seed{seed}'/f'{phase}_ep{epoch:04d}_gain{gain:+d}.log'
                        if (ROOT/relative/'verification.json').exists():
                            proof=verify_persisted(relative,allow_retired=True)
                            receipt=json.loads((ROOT/relative/'artifacts/epoch_receipt.json').read_text())
                        else:
                            wait_job(identity,seed,gain,phase,epoch,prior,reference,log)
                            receipt=read_remote(relative+'/epoch_receipt.json')
                            proof=persist(relative);assert proof['passed']
                        receipt['_remote_receipt']=f'{REMOTE}/{relative}/epoch_receipt.json'
                        receipts[gain]=receipt;traces[gain]=ROOT/relative/'artifacts/attempts.jsonl'
                        if gain==0:reference=f'{REMOTE}/{relative}/attempts.jsonl'
                        if prior:retire_consumed_previous(prior,receipt)
                        previous[gain]=receipt
                    accepted=qualify_epoch(receipts,traces);epoch_results.append(accepted)
                    destination=ROOT/'blocks'/identity/f'seed{seed}'/f'{phase}_ep{epoch:04d}.json'
                    atomic_json(destination,accepted)
                block=qualify_phase(epoch_results,previous)
                path=ROOT/'blocks'/identity/f'seed{seed}'/f'{phase}_qualification.json';atomic_json(path,block)
                publish(f'blocks/{identity}/seed{seed}/{phase}_qualification.json',block)
                status['blocks']=[b for b in status['blocks'] if (b['identity'],b['seed'],b['phase'])!=(identity,seed,phase)]+[block];save()
                assert block['passed'],'actual optimizer budgets differ; retain CONTROL_FAILED, no scaler/step compensation'
        status.update(current_phase='identity_training_completed_verified_supervision_handoff');save()
        handoff(identity)
    status.update(passed=True,current_phase='training_completed');save()

if __name__=='__main__':
    try:run()
    except BaseException:
        from datetime import datetime,timezone
        out=ROOT/'training_coordinator_failures';out.mkdir(exist_ok=True)
        (out/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.txt')).write_text(traceback.format_exc());raise
