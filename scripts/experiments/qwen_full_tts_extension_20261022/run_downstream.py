"""Bounded one-shot downstream execution; owns only its new process groups."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/qwen_full_tts_extension_20261022_attempt01'
PYTHON='/home/wjj/.venvs/syncnet/bin/python'


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def main():
    complete=json.loads((RUN/'tts/completion.json').read_text())
    if not complete['gpu_model_released'] or complete['count']!=30:raise RuntimeError('TTS incomplete')
    # Other jobs are never stopped. Wait for the designated TTS owner to exit.
    check=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
    active=[]
    for line in check.stdout.splitlines():
        pid,memory=line.split(',');mib=int(memory.strip().split()[0])
        if mib>1024:active.append((int(pid),mib))
    if active:raise RuntimeError(f'GPU still occupied: {active}')
    sources={}
    for name in ['pipeline.py','run_downstream.py','report_assets.py']:
        path=Path(__file__).with_name(name)
        sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        target=RUN/'source_snapshot'/name
        if target.exists():
            if target.read_bytes()!=path.read_bytes():raise RuntimeError('frozen source mismatch')
        else:target.write_bytes(path.read_bytes())
    write(RUN/'downstream_launch.json',dict(utc_unix=time.time(),source_sha256=sources,no_training=True))
    stages=[('audio',120),('render',1800),('extract',900),('analyze',300),('report_assets',120)]
    for stage,timeout in stages:
        script=Path(__file__).with_name('report_assets.py' if stage=='report_assets' else 'pipeline.py')
        cmd=[PYTHON,'-u',str(script)]+([] if stage=='report_assets' else [stage])
        log=RUN/f'{stage}_console.log';start=time.monotonic()
        print('STAGE_STARTED',stage,flush=True)
        with log.open('x') as f:
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            timed_out=False
            try:code=p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(p.pid,signal.SIGTERM)
                try:code=p.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
        write(RUN/f'{stage}_execution.json',dict(command=cmd,pid=p.pid,returncode=code,timed_out=timed_out,elapsed_s=time.monotonic()-start))
        print('STAGE_EXIT',stage,code,'elapsed_s',time.monotonic()-start,flush=True)
        if code or timed_out:raise RuntimeError(f'stage failed: {stage}; see {log}')
        size=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())
        if size>2_000_000_000:raise RuntimeError('frozen storage cap exceeded')
    write(RUN/'downstream_completion.json',dict(status='COMPLETE_PENDING_AUDIT',no_training=True,utc_unix=time.time()))


if __name__=='__main__':main()
