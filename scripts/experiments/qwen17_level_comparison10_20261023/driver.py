"""Serial one-shot execution with bounded children and actual exit receipts."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/qwen17_level_comparison10_20261023_attempt01'
CODE=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
    with (RUN/n).open('x') as f:json.dump(v,f,indent=2)
def main():
    sync='/home/wjj/.venvs/syncnet/bin/python';tts='/home/wjj/.venvs/qwen3-tts-1p7b/bin/python'
    protocol=RUN/'protocol.json';h=sha(protocol);start=time.monotonic()
    save('driver_launch.json',dict(pid=os.getpid(),command=sys.argv,source_sha256=sha(__file__),protocol_sha256=h,no_training=True))
    stages=[('generate',[tts,str(CODE/'experiment.py'),'generate'],7200),
      ('audio',[sync,str(CODE/'experiment.py'),'audio'],1800),
      *[(s,[sync,str(CODE/'downstream.py'),'--stage',s,'--run',str(RUN),'--protocol',str(protocol),'--protocol-sha256',h],10800) for s in ['render','official']],
      ('index',[sync,str(CODE/'experiment.py'),'index'],300)]
    failed=None
    for name,command,timeout in stages:
        print('STAGE',name,flush=True);t=time.monotonic()
        with (RUN/(name+'_console.log')).open('x') as f:
            process=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            timed=False
            try:rc=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed=True;os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                rc=process.returncode
        save(name+'_driver_exit.json',dict(stage=name,pid=process.pid,returncode=rc,timed_out=timed,elapsed_s=time.monotonic()-t,
             command=command,log_sha256=sha(RUN/(name+'_console.log'))))
        print('EXIT',name,rc,flush=True)
        if rc or timed:failed=name;break
    save('driver_exit.json',dict(status='COMPLETE' if failed is None else 'FAILED',failed_stage=failed,elapsed_s=time.monotonic()-start,no_training=True,pid=os.getpid()))
    return int(failed is not None)
if __name__=='__main__':raise SystemExit(main())
