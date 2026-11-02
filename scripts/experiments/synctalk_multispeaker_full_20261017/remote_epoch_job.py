"""Detached, immutable epoch jobs survive a Windows/SSH coordinator interruption."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
IDS=['SherrodBrown','PatrickLeahy','ChrisVanHollen','CathyMcMorrisRodgers']

def atomic_json(path,value):
    temporary=path.with_name(path.name+f'.{os.getpid()}.partial')
    with temporary.open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    temporary.replace(path)

def alive(pid,directory=None):
    try:
        os.kill(int(pid),0)
        if directory is not None:
            command=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
            assert str(directory).encode() in command and b'_run' in command,'runner PID was reused'
        return True
    except (ProcessLookupError,FileNotFoundError,PermissionError,AssertionError):return False

def status(directory):
    command=json.loads((directory/'command.json').read_text())
    execution=directory/'execution.json'
    if execution.exists():
        state=json.loads(execution.read_text())
        assert state['command_sha256']==command['command_sha256']
        if state['state']=='running' and not alive(state['pid'],directory):state.update(state='lost',passed=False)
        return state
    launcher=directory/'launcher.json'
    return dict(state='starting' if launcher.exists() and alive(json.loads(launcher.read_text())['pid'],directory) else 'lost',
        command_sha256=command['command_sha256'],passed=False)

def start_job(directory,command):
    raw=json.dumps(command,separators=(',',':')).encode();digest=hashlib.sha256(raw).hexdigest()
    if directory.exists():
        prior=json.loads((directory/'command.json').read_text());assert prior['command_sha256']==digest and prior['command']==command
        return status(directory)
    directory.mkdir(parents=True,exist_ok=False)
    atomic_json(directory/'command.json',dict(command=command,command_sha256=digest,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    with (directory/'launcher.log').open('xb') as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'_run',str(directory)],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    atomic_json(directory/'launcher.json',dict(pid=process.pid))
    return dict(state='starting',pid=process.pid,command_sha256=digest,passed=False)

def execute(directory):
    command=json.loads((directory/'command.json').read_text());start=time.time()
    result=dict(state='running',pid=os.getpid(),command_sha256=command['command_sha256'],passed=False,started_unix=start)
    atomic_json(directory/'execution.json',result)
    with (directory/'stdout.log').open('xb') as log:
        process=subprocess.Popen(command['command'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        result['child_pid']=process.pid;atomic_json(directory/'execution.json',result);code=process.wait()
    result.update(state='completed',returncode=code,passed=(code==0),elapsed_seconds=time.time()-start,
        stdout_sha256=hashlib.sha256((directory/'stdout.log').read_bytes()).hexdigest())
    atomic_json(directory/'execution.json',result)

def main():
    if len(sys.argv)>1 and sys.argv[1]=='_run':execute(Path(sys.argv[2]));return
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','status']);p.add_argument('--identity',choices=IDS,required=True)
    p.add_argument('--seed',type=int,choices=[42,2026],required=True);p.add_argument('--gain',type=int,choices=[-6,0,6],required=True)
    p.add_argument('--phase',choices=['coarse','fine'],required=True);p.add_argument('--epoch',type=int,required=True)
    p.add_argument('--checkpoint');p.add_argument('--resume');p.add_argument('--reference-trace');a=p.parse_args()
    directory=ROOT/'jobs'/a.identity/f'seed{a.seed}'/f'gain_{a.gain:+d}'/a.phase/f'ep{a.epoch:04d}'/'attempt01'
    if a.action=='status':print(json.dumps(status(directory)));return
    admission=json.loads((ROOT/'data_qualification.json').read_text());assert admission['passed'] and admission['training_ready']
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==admission['executing_sources']['remote_epoch_job.py']
    command=['bash',str(ROOT/'runtime_exec.sh'),str(ROOT/'train_epoch.py'),'--identity',a.identity,'--seed',str(a.seed),
        '--gain',str(a.gain),'--phase',a.phase,'--epoch',str(a.epoch),'--attempt','1']
    for name in ['checkpoint','resume','reference_trace']:
        value=getattr(a,name)
        if value:command+=['--'+name.replace('_','-'),value]
    print(json.dumps(start_job(directory,command)))
if __name__=='__main__':main()
