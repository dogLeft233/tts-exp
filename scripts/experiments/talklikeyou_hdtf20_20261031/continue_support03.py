"""Continue unfinished scorer coverage; never regenerate media or completed SyncNet."""
import os,time,fcntl,json
from prepare import B,RUN,write
from orchestrate import stage

def main():
    lock=(RUN/'orchestrator_support03.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((RUN/'generation_complete.json').read_text())['status']=='PASS'
    assert (RUN/'stages/syncnet_complete.json').exists()
    p=B/'envs/eval-base/bin/python'
    stage('scoring_support03',p,'controller.py')
    stage('audit',p,'audit_precision_replay.py','--run',RUN)
    stage('plot',p,'plot.py')
    stage('package',p,'package.py')
    write(RUN/'orchestrator_status.json',dict(status='COMPLETE',stage='DONE',pid=os.getpid(),updated=time.time(),continuation='support03'))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(RUN/'orchestrator_failure_support03.json',dict(error=repr(e),time=time.time()));raise
