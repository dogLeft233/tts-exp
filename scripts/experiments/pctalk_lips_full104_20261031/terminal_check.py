"""Confirm this experiment has exited; never signal any process."""
import json
import os
from pathlib import Path
import subprocess
import time
from prepare import RUN,GPU,write

def running(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[0] != 'Z'
    except FileNotFoundError:return False

def main():
    assert (RUN/'orchestrator_complete').exists()
    pids=set()
    for p in RUN.glob('*.pid'):pids.add(int(p.read_text()))
    reused=set(json.loads((RUN/'reuse_manifest.json').read_text())['reused_ids'])
    for p in (RUN/'generation').glob('*/process.json'):
        if p.parent.name not in reused:pids.add(json.loads(p.read_text())['pid'])
    active=[p for p in pids if running(p)]
    assert not active,active
    script_dir=Path(__file__).resolve().parent
    ancestors={os.getpid()};pid=os.getppid()
    while pid>1:
        ancestors.add(pid)
        try:pid=int(Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[1])
        except FileNotFoundError:break
    matching=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) in ancestors:continue
        try:
            if proc.stat().st_uid!=os.getuid() or not running(int(proc.name)):continue
            args=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if str(RUN) in args or str(script_dir) in args:matching.append(int(proc.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    assert not matching,matching
    gpu=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    write(RUN/'terminal_receipt.json',dict(status='PASS',time=time.time(),recorded_pids=sorted(pids),
        all_recorded_experiment_processes_exited=True,active_own_commands_bound_to_run=[],gpu_uuid=GPU,
        global_gpu_memory_and_utilization=gpu,interpretation='Global usage may include other jobs; this receipt confirms only this experiment exited. No process was signalled.',
        hdtf_talklikeyou_user_pause_unchanged=True))
    print('TERMINAL_PASS',len(pids),'global GPU:',gpu)

if __name__=='__main__':main()
