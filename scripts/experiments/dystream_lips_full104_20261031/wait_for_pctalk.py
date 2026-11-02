"""Wait for PC-Talk completion and successful server acceptance, then launch once."""
from pathlib import Path
import fcntl,json,subprocess,time
from prepare import B,PARENT,RUN,sha,write

def dependency_state(parent):
    parent=Path(parent)
    if (parent/'orchestrator_failed').exists():return 'PARENT_FAILED'
    if not (parent/'orchestrator_complete').exists():return 'WAITING_PARENT'
    for name in ['independent_audit.json','independent_statistics_audit.json','delivery_receipt.json','syncnet_geometry_audit.json']:
        p=parent/name
        if not p.exists():return 'PARENT_ACCEPTANCE_MISSING'
        if json.loads(p.read_text()).get('status')!='PASS':return 'PARENT_ACCEPTANCE_FAILED'
    return 'READY_FOR_TERMINAL_CHECK'

def main():
    lock=(RUN/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (RUN/'orchestrator_started').exists()
    while True:
        state=dependency_state(PARENT)
        write(RUN/'status.json',dict(status=state,stage='queued_after_pctalk',updated=time.time()))
        if state=='WAITING_PARENT':time.sleep(45);continue
        if state!='READY_FOR_TERMINAL_CHECK':
            write(RUN/'queue_blocked.json',dict(state=state,science_started=False,time=time.time()))
            raise RuntimeError(state)
        # The parent's independent watcher may briefly be checking terminal state
        # too. Retry only this read-only process check, never scientific inference.
        script=B/'experiments/pctalk_lips_full104_20261031/terminal_check.py'
        with (RUN/'logs/parent_terminal_check.log').open('a') as f:
            checked=subprocess.run([str(B/'envs/eval-base/bin/python'),str(script)],stdout=f,stderr=subprocess.STDOUT)
        if checked.returncode:time.sleep(45);continue
        terminal=json.loads((PARENT/'terminal_receipt.json').read_text())
        assert terminal['all_recorded_experiment_processes_exited']
        write(RUN/'dependency_acceptance.json',dict(status='PASS',parent_run=str(PARENT),
            terminal=terminal,checks={n:sha(PARENT/n) for n in ['delivery_receipt.json','independent_audit.json','independent_statistics_audit.json']},
            time=time.time(),scope='Server scientific/file/process acceptance; local agent visual review may still be pending'))
        break
    # Exec removes the waiting controller from the process tree. run.sh owns all
    # later failures; no GPU is touched by this queue before the dependency passes.
    import os
    os.execvp('bash',['bash',str(Path(__file__).with_name('run.sh'))])

if __name__=='__main__':main()
