"""One private sequential job; record scientific exits, never silently retry failures."""
import fcntl,json,os,subprocess,time
from pathlib import Path
import prepare
from prepare import B,D,RUN,GPU,sha,write
S=Path(__file__).resolve().parent

def stage(name,interpreter,script,*args):
    log=RUN/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
    cmd=[str(interpreter),'-u',str(S/script),*map(str,args)]
    write(RUN/'orchestrator_status.json',dict(status='RUNNING',stage=name,pid=os.getpid(),updated=time.time()))
    with log.open('w') as f:
        result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=12*3600)
    write(RUN/'stages'/('orchestrate_'+name+'.json'),dict(returncode=result.returncode,command=cmd,log_sha256=sha(log),ended=time.time()))
    if result.returncode:raise RuntimeError(name+' failed; retained log '+str(log))

def main():
    prepare.main()
    lock=(RUN/'orchestrator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=B/'envs/eval-base/bin/python'
    stage('runtime',p,'verify_runtime.py')
    stage('generation_smoke',D/'env/bin/python','generate.py','--smoke')
    stage('generation',D/'env/bin/python','generate.py')
    stage('scoring',p,'controller.py')
    stage('audit',p,'audit_precision_replay.py','--run',RUN)
    stage('plot',p,'plot.py')
    stage('package',p,'package.py')
    write(RUN/'orchestrator_status.json',dict(status='COMPLETE',stage='DONE',pid=os.getpid(),updated=time.time()))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write(RUN/'orchestrator_failure.json',dict(error=repr(e),time=time.time()));raise
