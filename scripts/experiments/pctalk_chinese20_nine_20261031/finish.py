"""Only read, audit and package after the scoring controller has exited."""
import json,fcntl,subprocess,time
from pathlib import Path
from prepare import B,RUN,sha,write

def main():
    lock=(RUN/'finish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=Path(__file__).resolve().parent
    while not (RUN/'stages/analysis.json').exists():
        assert not (RUN/'failure.json').exists(),(RUN/'failure.json').read_text() if (RUN/'failure.json').exists() else ''
        time.sleep(20)
    assert json.loads((RUN/'stages/analysis.json').read_text())['returncode']==0
    for name,script,args in [
        ('audit','audit_precision_replay.py',['--run',str(RUN)]),
        ('runtime','verify_runtime.py',[]),
        ('plot','plot.py',['--run',str(RUN)]),
        ('package','package.py',[])]:
        stage=RUN/'stages'/('finish_'+name+'.json')
        assert not stage.exists(),'Do not replay finished stages'
        write(RUN/'finish_status.json',dict(status='RUNNING',stage=name,updated=time.time()))
        cmd=[str(B/'envs/eval-base/bin/python'),'-u',str(source/script),*args]
        log=RUN/'logs'/('finish_'+name+'.log')
        with log.open('w') as output:code=subprocess.call(cmd,stdout=output,stderr=subprocess.STDOUT)
        write(stage,dict(returncode=code,command=cmd,log_sha256=sha(log),ended=time.time()))
        assert code==0,name+' failed'
    write(RUN/'finish_status.json',dict(status='COMPLETE',updated=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(RUN/'finish_failure.json',dict(error=repr(error),time=time.time()));raise
