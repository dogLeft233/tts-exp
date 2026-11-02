"""One-shot bounded downstream execution, exclusive GPU; ASR uses CPU only."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/qwen_nrt_expansion40_20261022_attempt01'
PY='/home/wjj/.venvs/syncnet/bin/python'


def save(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')


def main():
    done=json.loads((RUN/'generation/completion.json').read_text())
    assert done['R_count']==30 and done['T_count']==90 and done['gpu_model_released']
    pid=done.get('pid')
    if pid:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise RuntimeError('generation process has not actually exited')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True)
    active=[line for line in apps.splitlines() if int(line.split(',')[1].strip().split()[0])>1024]
    assert not active,active
    sources={};snapshot=RUN/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in ['pipeline.py','asr.py','run_downstream.py']:
        p=Path(__file__).with_name(name);dest=snapshot/name;sources[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        if dest.exists():assert dest.read_bytes()==p.read_bytes()
        else:dest.write_bytes(p.read_bytes())
    save(RUN/'downstream_launch.json',dict(utc_unix=time.time(),sources=sources,no_training=True,ASR_device='CPU'))
    asr=None;asr_log=None
    for stage,limit in [('audio',300),('render',3600),('extract',1200),('analyze',600)]:
        cmd=[PY,'-u',str(Path(__file__).with_name('pipeline.py')),stage];start=time.monotonic()
        print('START',stage,flush=True)
        with (RUN/(stage+'_console.log')).open('x') as f:
            worker=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            timed_out=False
            try:code=worker.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(worker.pid,signal.SIGTERM)
                try:code=worker.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);code=worker.wait()
        save(RUN/(stage+'_execution.json'),dict(command=cmd,pid=worker.pid,returncode=code,timed_out=timed_out,elapsed_s=time.monotonic()-start))
        print('EXIT',stage,code,time.monotonic()-start,flush=True)
        if code or timed_out:raise RuntimeError('failed stage '+stage)
        if sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())>8_000_000_000:raise RuntimeError('storage cap exceeded')
        if stage=='audio':
            asr_log=(RUN/'asr_console.log').open('x');asr_cmd=['/home/wjj/.venvs/tts-asr-audit-20261022/bin/python','-u',str(Path(__file__).with_name('asr.py'))]
            asr_start=time.monotonic();asr=subprocess.Popen(asr_cmd,cwd=ROOT,stdout=asr_log,stderr=subprocess.STDOUT,start_new_session=True)
            print('ASR_CPU_STARTED',asr.pid,flush=True)
    try:asr_code=asr.wait(timeout=max(1,1860-(time.monotonic()-asr_start)))
    except subprocess.TimeoutExpired:
        os.killpg(asr.pid,signal.SIGTERM)
        try:asr_code=asr.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(asr.pid,signal.SIGKILL);asr_code=asr.wait()
    asr_log.close();save(RUN/'asr_execution.json',dict(command=asr_cmd,pid=asr.pid,returncode=asr_code,elapsed_s=time.monotonic()-asr_start,device='cpu'))
    if asr_code:raise RuntimeError('ASR diagnostic failed; primary scores retained')
    save(RUN/'downstream_completion.json',dict(status='COMPLETE_PENDING_AUDIT',new_video_count=150,ASR_count=150,no_training=True))
    print('DOWNSTREAM_COMPLETE',flush=True)


if __name__=='__main__':main()
